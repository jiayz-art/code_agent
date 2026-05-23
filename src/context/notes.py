"""
结构化笔记生成器 —— NoteGenerator

设计思路：
- 每次工具调用后，在原始结果前追加一条结构化笔记
- 规则优先：80% 的工具调用用固定模板生成笔记（零成本）
- LLM 回退：规则无法覆盖的复杂情况，用轻量 Prompt 生成 1-2 句笔记
- 笔记格式统一且简洁，利于 LLM 快速理解工具做了什么
"""

from typing import Optional


class NoteGenerator:
    """
    工具结果结构化笔记生成器。

    职责：
    - generate(): 输入工具调用信息，返回一句话结构化笔记
    - 规则模板覆盖 80% 场景
    - 使用占位格式 [TOOL:name:结果]，LLM 看到极其简洁的笔记
    """

    def __init__(self, enabled: bool = True, use_llm_fallback: bool = False, llm_client=None):
        self._enabled = enabled
        self._use_llm = use_llm_fallback
        self._llm = llm_client

    def generate(
        self, tool_name: str, params: dict, result_content: str, success: bool
    ) -> str:
        """
        生成结构化笔记。

        Returns:
            一条简洁的笔记文本（1-3 行），设计为追加在原始 tool result 之后。
        """
        if not self._enabled:
            return ""

        # 规则模板
        handler = getattr(self, f"_note_{tool_name}", None)
        if handler:
            try:
                return handler(params, result_content, success)
            except Exception:
                pass

        # LLM 回退
        if self._use_llm and self._llm:
            return self._llm_note(tool_name, params, result_content, success)

        # 默认
        return self._default_note(tool_name, params, success)

    # ============================================================
    # 规则模板（按工具类型）
    # ============================================================

    def _note_read_file(self, params: dict, content: str, success: bool) -> str:
        path = params.get("path", "?")
        lines = content.count("\n") + (1 if content else 0)
        offset = params.get("offset", 1)
        return f"[NOTE] 读取 {path} 第{offset}行起, 共{lines}行"

    def _note_write_file(self, params: dict, content: str, success: bool) -> str:
        path = params.get("path", "?")
        written = params.get("content", "")
        lines = written.count("\n") + 1 if written else 0
        chars = len(written)
        return f"[NOTE] 写入 {path} ({lines}行, {chars}字符)"

    def _note_edit_file(self, params: dict, content: str, success: bool) -> str:
        path = params.get("path", "?")
        old = params.get("old_string", "")
        new = params.get("new_string", "")
        action = "替换" if old else "编辑"
        return f"[NOTE] {action} {path} (old:{len(old)}→new:{len(new)}字符)"

    def _note_run_command(self, params: dict, content: str, success: bool) -> str:
        cmd = params.get("command", "?")[:80]
        # 从 content 中提取 exit_code
        exit_code = "?" if not success else "0"
        if "[exit_code:" in content:
            import re
            m = re.search(r'\[exit_code:\s*(\d+)\]', content)
            if m:
                exit_code = m.group(1)
        status = f"exit={exit_code}" if success else f"exit={exit_code} (失败)"
        return f"[NOTE] 执行: {cmd} → {status}"

    def _note_grep_search(self, params: dict, content: str, success: bool) -> str:
        pattern = params.get("pattern", "?")[:60]
        search_path = params.get("path", ".")
        matches = content.count("\n") + (1 if content.strip() else 0)
        return f"[NOTE] 搜索 \"{pattern}\" in {search_path} → {matches}个匹配"

    def _note_git_status(self, params: dict, content: str, success: bool) -> str:
        if "干净" in content or "没有未提交" in content:
            return "[NOTE] Git 状态: 工作区干净"
        files = content.count("\n") + (1 if content.strip() else 0)
        return f"[NOTE] Git 状态: {files}个变更"

    def _note_git_diff(self, params: dict, content: str, success: bool) -> str:
        staged = "已暂存" if params.get("staged") else "未暂存"
        lines = content.count("\n") + (1 if content.strip() else 0)
        return f"[NOTE] Git diff ({staged}): {lines}行"

    def _note_git_commit(self, params: dict, content: str, success: bool) -> str:
        msg = params.get("message", "?")[:60]
        return f"[NOTE] Git commit: \"{msg}\""

    def _default_note(self, tool_name: str, params: dict, success: bool) -> str:
        status = "成功" if success else "失败"
        return f"[NOTE] {tool_name} [{status}]"

    # ============================================================
    # LLM 回退（规则无法覆盖时）
    # ============================================================

    def _llm_note(
        self, tool_name: str, params: dict, content: str, success: bool
    ) -> str:
        """用轻量 LLM 生成笔记"""
        try:
            summary = content[:400]  # 只发前 400 字符给 LLM
            messages = [
                {"role": "system", "content": "用一句话总结以下工具执行结果（不超过50字）。"},
                {"role": "user", "content": (
                    f"工具: {tool_name}\n参数: {params}\n"
                    f"结果({'成功' if success else '失败'}):\n{summary}"
                )},
            ]
            response = self._llm.chat(messages=messages, tools=[])
            if response.content:
                return f"[NOTE] {response.content.strip()[:100]}"
        except Exception:
            pass
        return self._default_note(tool_name, params, success)

    # ============================================================
    # Context Prompt 注入（告知 LLM 笔记和占位符机制）
    # ============================================================

    @staticmethod
    def get_context_prompt() -> str:
        """返回告知 LLM 上下文压缩机制的 system prompt 片段"""
        return """## 上下文压缩说明
为节省 Token，部分工具结果已生成结构化笔记（[NOTE]）或压缩为占位符。
- [NOTE] 开头的行：工具调用的简洁摘要
- [PLACEHOLDER:id:type:size]：被压缩的完整内容
- 如需查看完整内容，请回复 "展开 PLACEHOLDER:<id>"
- 工具调用的核心信息（文件路径、行号、退出码）全部保留"""
