"""
跨轮对话记忆 —— 解决多轮对话中丢失上下文的问题

设计思路：
- 在 REPL 模式下持久化跨 run() 调用的关键信息
- 追踪：创建/修改/读取过的文件列表、最近的工具调用摘要
- 每次新任务开始前，将上轮关键上下文注入 system prompt
- 不替代 MemoryManager（长期经验沉淀），而是补充短期工作记忆
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TurnRecord:
    """单轮对话记录"""
    task: str
    response: str
    files_created: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    success: bool = True
    error_summary: str = ""


class ConversationMemory:
    """
    跨轮对话记忆 —— 短期工作记忆。

    与 MemoryManager（长期经验）互补：
    - ConversationMemory 记住"这轮对话中做了什么"
    - MemoryManager 提炼"从这些经历中学到了什么"
    """

    def __init__(self, max_turns: int = 10, max_files: int = 30):
        self._turns: list[TurnRecord] = []
        self._max_turns = max_turns
        self._max_files = max_files
        # 聚合视图
        self._all_files_created: set[str] = set()
        self._all_files_modified: set[str] = set()
        self._all_files_read: set[str] = set()
        self._last_error: str = ""

    def record_turn(self, turn: TurnRecord):
        """记录一轮对话"""
        self._turns.append(turn)
        if len(self._turns) > self._max_turns:
            self._turns.pop(0)

        for f in turn.files_created:
            self._all_files_created.add(f)
        for f in turn.files_modified:
            self._all_files_modified.add(f)
        for f in turn.files_read:
            self._all_files_read.add(f)

        # 限制文件集合大小
        if len(self._all_files_created) > self._max_files:
            self._all_files_created = set(list(self._all_files_created)[-self._max_files:])
        if len(self._all_files_modified) > self._max_files:
            self._all_files_modified = set(list(self._all_files_modified)[-self._max_files:])

        if not turn.success and turn.error_summary:
            self._last_error = turn.error_summary

    def get_context_prompt(self) -> str:
        """
        生成可注入到新任务 system message 的上下文提示。
        只在有内容时返回非空字符串。
        """
        parts = []

        # 1. 最近一轮对话
        if self._turns:
            last = self._turns[-1]
            parts.append(f"## 上一轮对话\n- 任务: {last.task[:150]}")
            if last.files_created:
                parts.append(f"- 创建的文件: {', '.join(last.files_created[:5])}")
            if last.files_modified:
                parts.append(f"- 修改的文件: {', '.join(last.files_modified[:5])}")
            if not last.success:
                parts.append(f"- 上次操作未成功: {last.error_summary[:100]}")

        # 2. 本轮已操作的文件汇总（帮助记住已创建/修改的文件）
        recent_creations = list(self._all_files_created)[-10:]
        recent_modifications = list(self._all_files_modified)[-10:]
        if recent_creations:
            parts.append(f"\n## 本轮已创建的文件\n" + "\n".join(f"- {f}" for f in recent_creations))
        if recent_modifications:
            parts.append(f"\n## 本轮已修改的文件\n" + "\n".join(f"- {f}" for f in recent_modifications))

        # 3. 上次错误提醒
        if self._last_error and (not self._turns or not self._turns[-1].success):
            parts.append(f"\n## 注意\n上次操作遇到问题: {self._last_error[:150]}。请避免重复同样的操作。")

        if not parts:
            return ""

        return "\n".join(parts)

    def clear(self):
        """清空短期记忆（/clear 命令时调用）"""
        self._turns.clear()
        self._all_files_created.clear()
        self._all_files_modified.clear()
        self._all_files_read.clear()
        self._last_error = ""

    def get_recent_files(self) -> dict[str, list[str]]:
        """获取最近操作的文件列表"""
        return {
            "created": list(self._all_files_created),
            "modified": list(self._all_files_modified),
            "read": list(self._all_files_read),
        }

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def last_error(self) -> str:
        return self._last_error
