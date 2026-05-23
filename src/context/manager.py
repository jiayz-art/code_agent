"""
上下文压缩管理器 —— ContextManager

设计思路：
- 作为分层压缩的统一入口，对 Agent 暴露单一方法 prepare_messages()
- 三层流水线：占位替换 → 笔记生成 → 超限压缩
- 占位替换和笔记生成是"每次工具调用后"的实时操作
- 超限压缩是"每次 LLM 调用前"的批量检查
- 对外零侵入：Agent 只需调用 prepare_messages() 获取压缩后的消息列表
"""

from typing import Optional

from src.context.placeholder import PlaceholderEngine
from src.context.notes import NoteGenerator
from src.context.compressor import Compressor, CompressStats


class ContextManager:
    """
    分层上下文压缩管理器。

    职责：
    - prepare_messages(): LLM 调用前，对消息列表执行压缩流水线
    - process_tool_result(): 工具执行后，对单条结果生成笔记
    - get_stats(): 返回最近一次压缩的统计数据
    """

    def __init__(
        self,
        enabled: bool = True,
        max_tokens: int = 9000,
        keep_recent: int = 8,
        file_threshold_chars: int = 3000,
        output_threshold_chars: int = 2000,
        keep_head_lines: int = 30,
        keep_head_chars: int = 500,
        note_enabled: bool = True,
        note_use_llm: bool = False,
        summary_max_chars: int = 300,
        llm_client=None,
    ):
        self._enabled = enabled

        # 三层组件
        self._placeholder = PlaceholderEngine(
            file_threshold_chars=file_threshold_chars,
            output_threshold_chars=output_threshold_chars,
            keep_head_lines=keep_head_lines,
            keep_head_chars=keep_head_chars,
        )
        self._notes = NoteGenerator(
            enabled=note_enabled,
            use_llm_fallback=note_use_llm,
            llm_client=llm_client,
        )
        self._compressor = Compressor(
            llm_client=llm_client,
            max_tokens=max_tokens,
            keep_recent=keep_recent,
            enabled=enabled,
            max_summary_chars=summary_max_chars,
        )

        self._last_stats = CompressStats()
        self._placeholder_map: dict[str, str] = {}

    # ============================================================
    # 主入口
    # ============================================================

    def prepare_messages(self, messages: list[dict]) -> list[dict]:
        """
        LLM 调用前准备消息列表：占位替换 → 超限压缩。

        占位替换在每次工具调用后已实时执行（通过 process_tool_result），
        这里做兜底扫描 + 超限压缩。

        Returns:
            处理后的消息列表
        """
        if not self._enabled:
            return messages

        # 第1层：占位替换（兜底扫描，处理未被实时替换的超长内容）
        messages, placeholder_map = self._placeholder.replace(messages)
        self._placeholder_map.update(placeholder_map)

        # 第2+3层：超限压缩（含 LLM 摘要）
        messages = self._compressor.compress(
            messages,
            placeholder_count=self._placeholder.placeholder_count,
            notes_count=0,
        )
        self._last_stats = self._compressor.get_last_stats()

        return messages

    # ============================================================
    # 工具结果处理（实时）
    # ============================================================

    def process_tool_result(
        self, tool_name: str, params: dict, result_content: str, success: bool
    ) -> str:
        """
        对单条工具执行结果进行实时处理：笔记生成 + 占位替换。

        在 Agent._execute_tool_calls() 中每次工具调用后调用。

        Args:
            tool_name: 工具名称
            params: 工具参数
            result_content: 工具返回的原始内容
            success: 工具是否执行成功

        Returns:
            处理后的内容（可能追加了笔记，或替换为占位符）
        """
        if not self._enabled:
            return result_content

        # 生成结构化笔记
        note = self._notes.generate(tool_name, params, result_content, success)
        if note:
            result_content = note + "\n" + result_content

        return result_content

    # ============================================================
    # 占位符检索
    # ============================================================

    def resolve_placeholder(self, placeholder_id: str) -> Optional[str]:
        """按 ID 检索被压缩的原始内容"""
        return self._placeholder.resolve(placeholder_id)

    def has_placeholder(self, placeholder_id: str) -> bool:
        return self._placeholder.has(placeholder_id)

    # ============================================================
    # 统计
    # ============================================================

    def get_stats(self) -> CompressStats:
        return self._last_stats

    def get_placeholder_count(self) -> int:
        return self._placeholder.placeholder_count

    def get_cache_metrics(self) -> dict:
        """返回上下文压缩的缓存指标（用于 Prompt Cache 命中率监控）"""
        compressor_metrics = self._compressor.get_cache_metrics()
        return {
            "placeholder_count": self._placeholder.placeholder_count,
            "placeholder_store_size": len(self._placeholder._store),
            **compressor_metrics,
        }

    # ============================================================
    # Context Prompt
    # ============================================================

    @staticmethod
    def get_context_prompt() -> str:
        """返回告知 LLM 上下文压缩机制的 system prompt 片段"""
        return NoteGenerator.get_context_prompt()
