"""
记忆反思生成器 —— 混合触发（同步摘要 + 异步提炼）

设计思路：
- 同步阶段（after_task 立即执行）：generate_summary()
  输入 tool_call_logs 摘要 → ~400 token Prompt → 输出 <100 字一句话总结
  耗时约 1-2 秒，用户能接受

- 异步阶段（后台线程执行）：generate_detail()
  输入 summary + tool_call_logs 完整内容 → ~800 token Prompt → 输出结构化经验
  耗时约 3-5 秒，用户无感知

- 标签和任务类型分类用规则匹配（关键词 + 工具组合），快速且稳定
"""

import re
import threading
from typing import Any, Optional, Callable

from src.state import ToolCallRecord


# ============================================================
# 规则引擎：任务类型分类 + 错误类型识别 + 标签提取
# ============================================================

# 任务类型关键词（按优先级排列，匹配即停止）
_TASK_TYPE_PATTERNS: list[tuple[str, list[str]]] = [
    ("test",       ["测试", "test", "pytest", "unittest", "覆盖率", "coverage", "断言", "assertion"]),
    ("bug-fix",    ["修复", "fix", "bug", "报错", "错误", "失败", "error", "exception", "崩溃", "crash"]),
    ("refactor",   ["重构", "refactor", "拆分", "提取", "重命名", "优化结构"]),
    ("document",   ["文档", "doc", "readme", "注释", "docstring", "解释"]),
    ("feature",    ["实现", "添加", "新增", "开发", "创建", "feature", "add", "implement"]),
    ("debug",      ["调试", "debug", "排查", "分析", "定位", "日志", "log"]),
    ("git",        ["提交", "commit", "分支", "branch", "merge", "合并", "PR"]),
    ("performance", ["性能", "优化速度", "加速", "慢", "performance", "卡顿"]),
]


def classify_task_type(task: str, tool_logs: list[ToolCallRecord]) -> str:
    """根据任务描述和使用的工具分类任务类型"""
    task_lower = task.lower()
    for task_type, keywords in _TASK_TYPE_PATTERNS:
        for kw in keywords:
            if kw.lower() in task_lower:
                return task_type
    return "general"


# 错误类型关键词
_ERROR_TYPE_PATTERNS: list[tuple[str, list[str]]] = [
    ("AssertionError",    ["assertionerror", "assertion error", "断言失败", "assert"]),
    ("AttributeError",    ["attributeerror", "attribute error", "属性错误", "has no attribute"]),
    ("ImportError",       ["importerror", "import error", "导入错误", "no module"]),
    ("KeyError",          ["keyerror", "key error", "键错误"]),
    ("TypeError",         ["typeerror", "type error", "类型错误"]),
    ("ValueError",        ["valueerror", "value error", "值错误"]),
    ("SyntaxError",       ["syntaxerror", "syntax error", "语法错误"]),
    ("TimeoutError",      ["timeout", "超时", "timed out"]),
    ("FileNotFoundError", ["filenotfound", "file not found", "找不到文件", "no such file"]),
    ("PermissionError",   ["permission", "权限", "denied"]),
    ("test-failure",      ["test", "测试失败", "test failed", "fail"]),
    ("config-error",      ["配置", "config", "环境变量", ".env"]),
]


def extract_error_type(task: str, tool_logs: list[ToolCallRecord]) -> str:
    """从任务描述和工具执行结果中提取错误类型"""
    search_text = task.lower()
    # 也搜索工具调用结果中的错误信息
    for log in tool_logs:
        if not log.success and log.error:
            search_text += " " + log.error.lower()
        if log.result_content:
            search_text += " " + log.result_content.lower()[:500]

    for error_type, keywords in _ERROR_TYPE_PATTERNS:
        for kw in keywords:
            if kw.lower() in search_text:
                return error_type
    return ""


def extract_tags(task: str, summary: str = "", detail: dict = None) -> list[str]:
    """从任务和反思中提取标签（规则 + 关键词）"""
    text = f"{task} {summary}"
    if detail:
        text += f" {detail.get('问题', '')} {detail.get('方案', '')}"

    text_lower = text.lower()
    tags: set[str] = set()

    # 通用编程标签
    tag_patterns = [
        # 语言/框架
        (["python", ".py", "pytest", "unittest"], "python"),
        (["javascript", "js", "node", "npm"], "javascript"),
        (["sql", "数据库", "database", "mysql", "sqlite"], "database"),
        (["api", "接口", "endpoint", "路由"], "api"),
        # 操作类型
        (["重构", "refactor", "拆分", "提取函数"], "refactor"),
        (["测试", "test", "pytest", "覆盖率"], "testing"),
        (["文档", "doc", "readme", "注释"], "documentation"),
        (["git", "commit", "提交", "分支"], "git"),
        (["配置", "config", "环境变量", ".env"], "configuration"),
        # 技术领域
        (["认证", "auth", "登录", "token", "jwt"], "authentication"),
        (["异常", "错误处理", "exception", "try-catch"], "error-handling"),
        (["类型注解", "type hint", "mypy"], "type-hints"),
        (["性能", "优化", "慢", "performance", "卡顿"], "performance"),
        (["导入", "import", "依赖", "dependency"], "dependencies"),
        (["文件", "读写", "file", "io"], "file-io"),
        (["命令", "shell", "bash", "cmd", "运行"], "shell"),
    ]

    for keywords, tag in tag_patterns:
        for kw in keywords:
            if kw.lower() in text_lower:
                tags.add(tag)
                break

    # 错误相关标签
    if any(w in text_lower for w in ["报错", "错误", "失败", "error", "fail", "bug"]):
        tags.add("error-fix")

    return sorted(tags)[:8]  # 最多 8 个标签


# ============================================================
# Prompt 模板
# ============================================================

_SUMMARY_SYSTEM_PROMPT = """你是一个编程任务总结助手。根据工具调用日志，用一句话总结任务做了什么。"""


def _build_summary_prompt(task: str, tool_logs: list[ToolCallRecord]) -> str:
    """构建摘要生成的用户 Prompt"""
    log_lines = []
    for log in tool_logs[:20]:  # 最多 20 条
        status = "OK" if log.success else "FAIL"
        params_summary = ", ".join(f"{k}={v}" for k, v in list(log.params.items())[:3])
        log_lines.append(
            f"- [{status}] {log.tool_name}({params_summary}) "
            f"→ {log.result_content[:100]}"
        )
    logs_text = "\n".join(log_lines) if log_lines else "（无工具调用）"

    return f"""## 原始任务
{task}

## 工具调用记录
{logs_text}

请用一句话总结本次任务做了什么（不超过 100 字）。只输出这句话，不要其他内容。"""


_DETAIL_SYSTEM_PROMPT = """你是一个编程经验提炼助手。将任务执行记录提炼为结构化经验，便于后续复用。"""


def _build_detail_prompt(task: str, summary: str, tool_logs: list[ToolCallRecord]) -> str:
    """构建深度提炼的用户 Prompt"""
    log_lines = []
    for log in tool_logs:
        status = "成功" if log.success else "失败"
        log_lines.append(
            f"### {log.tool_name} [{status}]\n"
            f"参数: {log.params}\n"
            f"结果: {log.result_content[:200]}\n"
            f"错误: {log.error or '无'}"
        )
    logs_text = "\n\n".join(log_lines) if log_lines else "（无工具调用）"

    return f"""## 原始任务
{task}

## 执行摘要
{summary}

## 工具调用详情
{logs_text}

## 输出格式
请按以下结构化格式提炼经验：

问题: <描述遇到的具体问题>
根因: <问题的根本原因是什么>
方案: <如何解决的，具体步骤>
教训: <可复用的经验教训，供后续类似任务参考>
涉及文件: <逗号分隔的文件路径列表>

只输出上述格式的内容，不要额外解释。"""


# ============================================================
# MemoryReflector 类
# ============================================================

class MemoryReflector:
    """
    记忆反思生成器。

    职责：
    - generate_summary(): 同步生成任务摘要（< 2 秒）
    - generate_detail(): 异步生成结构化经验（后台线程）
    - 标签和分类：纯规则引擎，快速无成本
    """

    def __init__(self, llm_client, async_enabled: bool = True):
        self._llm = llm_client
        self._async_enabled = async_enabled
        self._active_threads: list[threading.Thread] = []

    # ============================================================
    # 同步：摘要生成
    # ============================================================

    def generate_summary(self, task: str, tool_logs: list[ToolCallRecord]) -> str:
        """
        同步生成一句话任务摘要。

        用轻量 Prompt（~400 tokens）调用 LLM，快速返回。
        如果 LLM 调用失败，返回规则生成的 fallback 摘要。
        """
        if not tool_logs:
            return f"完成任务: {task[:80]}（无工具调用）"

        try:
            messages = [
                {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": _build_summary_prompt(task, tool_logs)},
            ]
            response = self._llm.chat(messages=messages, tools=[])
            if response.content and response.content.strip():
                return response.content.strip()[:150]
        except Exception:
            pass

        # Fallback：规则生成
        tool_names = list(dict.fromkeys(log.tool_name for log in tool_logs))
        fail_count = sum(1 for log in tool_logs if not log.success)
        parts = [f"执行任务: {task[:60]}"]
        parts.append(f"使用工具: {', '.join(tool_names[:5])}")
        if fail_count:
            parts.append(f"({fail_count} 次失败)")
        return "。".join(parts)

    # ============================================================
    # 异步：深度提炼
    # ============================================================

    def generate_detail(
        self,
        task: str,
        summary: str,
        tool_logs: list[ToolCallRecord],
        callback: Optional[Callable[[dict], None]] = None,
    ) -> Optional[dict]:
        """
        生成结构化经验详情。

        async_enabled=True 时，在后台线程执行，结果通过 callback 返回。
        async_enabled=False 时，同步执行，直接返回 dict。
        """
        if self._async_enabled and callback:
            thread = threading.Thread(
                target=self._run_detail_generation,
                args=(task, summary, tool_logs, callback),
                daemon=True,
                name="memory-reflect",
            )
            thread.start()
            self._active_threads.append(thread)
            # 清理已完成线程
            self._active_threads = [t for t in self._active_threads if t.is_alive()]
            return None
        else:
            return self._do_generate_detail(task, summary, tool_logs)

    def _run_detail_generation(
        self, task: str, summary: str, tool_logs: list, callback: Callable
    ):
        """后台线程入口"""
        try:
            result = self._do_generate_detail(task, summary, tool_logs)
            callback(result)
        except Exception:
            callback({})  # 即使失败也回调空结果，避免调用方永远等待

    def _do_generate_detail(
        self, task: str, summary: str, tool_logs: list[ToolCallRecord]
    ) -> dict:
        """执行深度提炼的 LLM 调用"""
        try:
            messages = [
                {"role": "system", "content": _DETAIL_SYSTEM_PROMPT},
                {"role": "user", "content": _build_detail_prompt(task, summary, tool_logs)},
            ]
            response = self._llm.chat(messages=messages, tools=[])
            if response.content:
                return self._parse_detail(response.content)
        except Exception:
            pass
        return {
            "问题": summary,
            "根因": "未能自动分析",
            "方案": "参考工具调用日志",
            "教训": summary,
            "涉及文件": "",
        }

    def _parse_detail(self, content: str) -> dict:
        """解析 LLM 返回的结构化经验文本"""
        result = {}
        keys_map = {"问题": "问题", "根因": "根因", "方案": "方案", "教训": "教训", "涉及文件": "涉及文件"}
        current_key = None
        current_value: list[str] = []

        for line in content.strip().split("\n"):
            line = line.strip()
            found = False
            for cn_key, en_key in keys_map.items():
                if line.startswith(cn_key + ":") or line.startswith(cn_key + "："):
                    if current_key and current_value:
                        result[current_key] = "\n".join(current_value).strip()
                    current_key = cn_key
                    value_part = line.split(":", 1)[-1].split("：", 1)[-1].strip()
                    current_value = [value_part] if value_part else []
                    found = True
                    break
            if not found and current_key:
                current_value.append(line)

        # 处理最后一个字段
        if current_key and current_value:
            result[current_key] = "\n".join(current_value).strip()

        return result

    # ============================================================
    # 便捷方法
    # ============================================================

    def classify_and_tag(self, task: str, summary: str, tool_logs: list, detail: dict) -> tuple:
        """一站式：返回 (task_type, error_type, tags)"""
        task_type = classify_task_type(task, tool_logs)
        error_type = extract_error_type(task, tool_logs)
        tags = extract_tags(task, summary, detail)
        return task_type, error_type, tags
