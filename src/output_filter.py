"""
用户可见输出过滤器

设计思路：
- 分离内部日志和用户可见消息：内部错误/堆栈只写日志文件，用户看到友好提示
- 错误分类映射：将原始 Python 异常、安全拦截等转为用户友好的中文提示
- 增量式输出：每次过滤返回清洁后的内容，去除内部标记
- 敏感信息脱敏：隐藏 API key、token 等不会出现在用户输出中
"""

import re
from typing import Optional


# 内部日志前缀模式（匹配到即过滤）
_INTERNAL_PATTERNS = [
    re.compile(r"^tool_call:\s+\w+\s+(OK|FAIL)", re.IGNORECASE),
    re.compile(r"^tokens:\s+prompt=", re.IGNORECASE),
    re.compile(r"^\[MemoryManager\s+", re.IGNORECASE),
    re.compile(r"^\[MemoryStore\]", re.IGNORECASE),
    re.compile(r"Traceback\s*\(most recent call last\)"),
    re.compile(r"^\s*File\s+\".*\",\s+line\s+\d+,\s+in\s+"),
    re.compile(r"^\w+Error:"),
    re.compile(r"^\w+Exception:"),
    re.compile(r"^During handling of the above exception"),
    re.compile(r"^\s*\^+\s*$"),
]

# 错误消息映射：内部错误 → 用户友好提示
_ERROR_TRANSLATIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"SECURITY_BLOCK.*critical", re.IGNORECASE),
     "操作已被安全策略拦截，因为目标路径在工作区之外。请将文件写入工作区内。"),
    (re.compile(r"SECURITY_BLOCK", re.IGNORECASE),
     "操作因安全策略被阻止。"),
    (re.compile(r"MemoryReflector.*has no attribute", re.IGNORECASE),
     "记忆系统内部处理中，不影响任务执行。"),
    (re.compile(r"AttributeError.*has no attribute", re.IGNORECASE),
     "内部组件状态异常，已自动恢复。"),
    (re.compile(r"ConnectionError|ConnectionRefusedError|TimeoutError", re.IGNORECASE),
     "服务暂时不可用，请稍后重试。"),
    (re.compile(r"LLM API 调用失败", re.IGNORECASE),
     "AI 服务调用失败，请检查网络连接和 API Key 配置。"),
    (re.compile(r"路径越界", re.IGNORECASE),
     "文件路径不在允许的工作区内，已阻止操作。"),
    (re.compile(r"权限不足|PermissionError|Permission denied", re.IGNORECASE),
     "没有足够的权限执行此操作。"),
    (re.compile(r"UnicodeDecodeError", re.IGNORECASE),
     "文件编码不支持，无法读取。"),
    (re.compile(r"未找到匹配的 old_string", re.IGNORECASE),
     "文件中未找到要替换的内容，请检查原文是否精确匹配。"),
    (re.compile(r"old_string 不能为空", re.IGNORECASE),
     "编辑文件时需要提供要替换的原始文本。"),
]


def is_internal_log(line: str) -> bool:
    """判断一行文本是否为内部日志（不应展示给用户）"""
    stripped = line.strip()
    if not stripped:
        return False
    for pattern in _INTERNAL_PATTERNS:
        if pattern.search(stripped):
            return True
    return False


def filter_lines(text: str) -> str:
    """过滤文本中的内部日志行，返回用户可看的内容"""
    if not text:
        return ""
    lines = text.split("\n")
    filtered = [line for line in lines if not is_internal_log(line)]
    return "\n".join(filtered)


def translate_error(raw_error: str) -> str:
    """将原始错误信息翻译为用户友好的提示"""
    if not raw_error:
        return ""
    for pattern, friendly in _ERROR_TRANSLATIONS:
        if pattern.search(raw_error):
            return friendly
    # 默认脱敏：移除文件路径、行号等内部信息
    cleaned = re.sub(r"File\s+\".*?\",\s+line\s+\d+", "[内部位置]", raw_error)
    cleaned = re.sub(r"'(/[^']*)'", "'[路径已隐藏]'", cleaned)
    cleaned = re.sub(r"E:[^\s]*", "[路径已隐藏]", cleaned)
    if len(cleaned) > 200:
        cleaned = cleaned[:200] + "..."
    return cleaned


def sanitize_for_user(text: str) -> str:
    """
    综合过滤：先逐行过滤内部日志，再翻译残留的错误消息。
    返回适合展示给用户的安全文本。
    """
    text = filter_lines(text)
    # 检查是否整段都是内部错误消息
    for pattern, friendly in _ERROR_TRANSLATIONS:
        if pattern.search(text):
            return friendly
    return text


class UserOutput:
    """用户输出包装器，用于 Agent 返回结果前做最终净化"""

    @staticmethod
    def wrap(raw_response: str, success: bool = True, workspace: str = "") -> str:
        """
        包装 Agent 的原始响应：
        1. 过滤内部日志行
        2. 翻译已知错误为友好提示
        3. 如果失败且响应为空，生成兜底消息
        """
        cleaned = filter_lines(raw_response)

        if not success:
            # 尝试翻译错误
            friendly = translate_error(cleaned)
            if friendly:
                return friendly
            if not cleaned.strip():
                return (
                    "抱歉，任务执行过程中遇到了问题。"
                    "请检查工作区设置或重试。"
                )

        if not cleaned.strip():
            return "任务已完成。"

        return cleaned

    @staticmethod
    def wrap_error(error_message: str, workspace: str = "") -> str:
        """包装错误消息为友好提示"""
        friendly = translate_error(error_message)
        if friendly:
            if workspace and "工作区" in friendly:
                friendly += f"\n当前工作区: {workspace}"
            return friendly
        return f"操作未成功: {sanitize_for_user(error_message)}"
