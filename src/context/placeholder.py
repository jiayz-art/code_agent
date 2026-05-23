"""
占位替换引擎 —— PlaceholderEngine

设计思路：
- 扫描 tool 角色消息，超长内容替换为紧凑占位符
- 原始内容存入 placeholder_map，支持按需检索恢复
- 占位符格式固定且短小，利于 Prompt Cache 命中
- 阈值可配置：文件内容、命令输出分别设置
"""

import hashlib
from typing import Any, Optional


class PlaceholderEngine:
    """
    占位替换引擎。

    职责：
    - replace(): 扫描 messages，超长 tool result 替换为占位符
    - resolve(): 按占位符 ID 返回原始内容
    - 不修改原始 messages 对象，生成新的副本
    """

    def __init__(
        self,
        file_threshold_chars: int = 3000,
        output_threshold_chars: int = 2000,
        keep_head_lines: int = 30,
        keep_head_chars: int = 500,
    ):
        self._file_threshold = file_threshold_chars
        self._output_threshold = output_threshold_chars
        self._keep_head_lines = keep_head_lines
        self._keep_head_chars = keep_head_chars
        # 占位符映射: placeholder_id → 原始内容
        self._store: dict[str, str] = {}

    # ============================================================
    # 主入口
    # ============================================================

    def replace(self, messages: list[dict]) -> tuple[list[dict], dict[str, str]]:
        """
        扫描并替换 messages 中的超长 tool 结果。

        Args:
            messages: OpenAI 格式的消息列表

        Returns:
            (替换后的 messages 副本, placeholder_id → 原始内容 的映射表)
        """
        self._store = {}
        result = []

        for msg in messages:
            if msg.get("role") == "tool" and "content" in msg:
                replaced = self._replace_if_long(msg)
                result.append(replaced)
            else:
                result.append(msg.copy())

        return result, self._store.copy()

    def _replace_if_long(self, msg: dict) -> dict:
        """判断单条 tool 消息是否需要占位替换"""
        content = msg.get("content", "")
        tool_name = msg.get("name", "")

        if not content or not isinstance(content, str):
            return msg.copy()

        # 根据工具类型选择合适的阈值和替换策略
        if tool_name == "read_file":
            threshold = self._file_threshold
            lines = content.split("\n")
            if len(lines) > self._keep_head_lines * 2 or len(content) > threshold:
                return self._replace_read_file(msg, content, lines)
        elif tool_name == "run_command":
            threshold = self._output_threshold
            if len(content) > threshold:
                return self._replace_command_output(msg, content)
        elif len(content) > self._output_threshold:
            return self._replace_generic(msg, content, tool_name)

        return msg.copy()

    def _replace_read_file(self, msg: dict, content: str, lines: list[str]) -> dict:
        """替换 read_file 结果"""
        placeholder_id = self._make_id(msg)
        self._store[placeholder_id] = content

        head_lines = lines[: self._keep_head_lines]
        total_lines = len(lines)
        file_path = msg.get("name", "")  # 从 tool name 无法获取，用 params 推断

        head = "".join(f"{i+1:6}\t{line}" for i, line in enumerate(head_lines))
        placeholder = (
            f"\n... [{self._keep_head_lines} 行已显示, 剩余 {total_lines - self._keep_head_lines} 行已压缩] ...\n"
            f"[PLACEHOLDER:{placeholder_id}:read_file:{total_lines}lines]\n"
            f"如需查看完整内容, 请说 \"展开 PLACEHOLDER:{placeholder_id}\"\n"
        )

        new_msg = msg.copy()
        new_msg["content"] = head + placeholder
        return new_msg

    def _replace_command_output(self, msg: dict, content: str) -> dict:
        """替换 run_command 输出"""
        placeholder_id = self._make_id(msg)
        self._store[placeholder_id] = content

        head = content[: self._keep_head_chars]
        total = len(content)

        new_msg = msg.copy()
        new_msg["content"] = (
            f"{head}\n"
            f"... [输出已截断, 完整 {total} 字符] ...\n"
            f"[PLACEHOLDER:{placeholder_id}:run_command:{total}chars]\n"
            f"如需查看完整输出, 请说 \"展开 PLACEHOLDER:{placeholder_id}\"\n"
        )
        return new_msg

    def _replace_generic(self, msg: dict, content: str, tool_name: str) -> dict:
        """通用占位替换"""
        placeholder_id = self._make_id(msg)
        self._store[placeholder_id] = content

        head = content[: self._keep_head_chars]
        total = len(content)

        new_msg = msg.copy()
        new_msg["content"] = (
            f"{head}\n"
            f"... [后续 {total - self._keep_head_chars} 字符已压缩] ...\n"
            f"[PLACEHOLDER:{placeholder_id}:{tool_name}:{total}chars]\n"
            f"如需查看完整内容, 请说 \"展开 PLACEHOLDER:{placeholder_id}\"\n"
        )
        return new_msg

    # ============================================================
    # 按需检索
    # ============================================================

    def resolve(self, placeholder_id: str) -> Optional[str]:
        """根据占位符 ID 返回原始完整内容"""
        return self._store.get(placeholder_id)

    def has(self, placeholder_id: str) -> bool:
        return placeholder_id in self._store

    def resolve_all_in_text(self, text: str, max_expanded_chars: int = 5000) -> str:
        """自动展开文本中所有占位符（限制总展开长度）"""
        import re
        ids = self.extract_placeholder_ids(text)
        total_expanded = 0
        for pid in ids:
            content = self._store.get(pid)
            if content and total_expanded < max_expanded_chars:
                available = max_expanded_chars - total_expanded
                replacement = content[:available]
                if len(content) > available:
                    replacement += "\n... [内容过长已截断]"
                text = text.replace(f"[PLACEHOLDER:{pid}:", f"[已展开 PLACEHOLDER:{pid}:")
                total_expanded += len(replacement)
        return text

    # ============================================================
    # 辅助
    # ============================================================

    def _make_id(self, msg: dict) -> str:
        """为消息生成简短的唯一占位符 ID"""
        raw = msg.get("content", "")[:200] + str(hash(msg.get("tool_call_id", "")))
        return hashlib.md5(raw.encode()).hexdigest()[:8]

    @property
    def placeholder_count(self) -> int:
        return len(self._store)

    def clear(self):
        self._store.clear()
