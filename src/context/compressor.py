"""
超限压缩器 —— Compressor

设计思路：
- 当 messages 的 Token 估算值超过阈值时，触发压缩
- 保留策略：最新 N 条消息 + 所有 system 消息 + 首条 user 消息
- 压缩策略：其余旧消息合成一段 LLM 结构化摘要（问题-已完成-当前状态-关键发现）
- 压缩结果作为 system 消息插入，其余旧消息被移除
- 未超限时零开销返回原样
"""

import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CompressStats:
    """压缩统计"""
    original_tokens: int = 0
    compressed_tokens: int = 0
    placeholder_count: int = 0
    notes_generated: int = 0
    summary_generated: bool = False

    @property
    def savings_percent(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (1 - self.compressed_tokens / self.original_tokens) * 100


# ============================================================
# Token 估算
# ============================================================

# ★ 优先使用 tiktoken 精确计数，不可用时回退到字符估算
try:
    import tiktoken
    try:
        _enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _enc = None
except ImportError:
    _enc = None


def estimate_tokens(text: str) -> int:
    """Token 估算：优先 tiktoken（精确），回退字符估算"""
    if not text:
        return 0
    if _enc is not None:
        try:
            return len(_enc.encode(text))
        except Exception:
            pass
    chinese = sum(1 for c in text if '一' <= c <= '鿿')
    other = len(text) - chinese
    return int(chinese / 1.5 + other / 4)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """估算消息列表的总 Token 数"""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            # 多模态 content 数组
            total += estimate_tokens(json.dumps(content, ensure_ascii=False))
        # 工具调用
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            total += estimate_tokens(func.get("name", ""))
            total += estimate_tokens(func.get("arguments", ""))
        # role 和 tool_call_id 等元数据 ~5 tokens
        total += 5
    return total


# ============================================================
# LLM 摘要 Prompt
# ============================================================

_SUMMARY_SYSTEM = """你是一个对话压缩助手。将代码助手的对话历史压缩为结构化摘要，保留所有关键信息。"""


def _build_summary_prompt(messages: list[dict]) -> str:
    """构建摘要生成的用户 Prompt"""
    # 提取关键信息而非全量发送
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            parts.append(f"[系统指令] {content[:200]}")
        elif role == "user":
            parts.append(f"[用户] {content[:300]}")
        elif role == "assistant":
            tc_names = [tc.get("function", {}).get("name", "") for tc in msg.get("tool_calls", [])]
            text = content[:200] if content else ""
            if tc_names:
                text += f" (调用工具: {', '.join(tc_names)})"
            parts.append(f"[助手] {text}")
        elif role == "tool":
            tool_name = msg.get("name", "")
            result_preview = content[:300] if isinstance(content, str) else ""
            success = "[工具执行失败]" not in result_preview
            parts.append(
                f"[工具结果:{tool_name} {'成功' if success else '失败'}] {result_preview[:200]}"
            )

    history = "\n".join(parts[-30:])  # 最多 30 条

    return f"""## 需要压缩的对话历史
{history}

## 输出格式
请按以下格式输出结构化摘要（控制在 300 字以内）：

需求: <用户的原始需求是什么>
完成: <完成了哪些操作，修改了哪些文件>
状态: <当前进展到哪一步，是否遇到问题>
发现: <值得保留的关键发现或教训>

只输出上述格式内容，不要其他解释。"""


# ============================================================
# Compressor
# ============================================================

class Compressor:
    """
    超限压缩器。

    职责：
    - compress(): 超限时压缩旧消息，未超限时返回原样
    - 保留策略 + LLM 摘要压缩旧消息
    """

    def __init__(
        self,
        llm_client=None,
        max_tokens: int = 9000,
        keep_recent: int = 8,
        enabled: bool = True,
        max_summary_chars: int = 300,
    ):
        self._llm = llm_client
        self._max_tokens = max_tokens
        self._keep_recent = keep_recent
        self._enabled = enabled
        self._max_summary_chars = max_summary_chars

        self._last_stats = CompressStats()
        self._last_preview: str = ""
        self._total_compressions: int = 0
        self._total_tokens_saved: int = 0

    # ============================================================
    # 主入口
    # ============================================================

    def compress(
        self, messages: list[dict], placeholder_count: int = 0, notes_count: int = 0
    ) -> list[dict]:
        """
        超限时压缩旧消息。

        未超限返回原 messages（零开销），超限才执行压缩。

        Args:
            messages: 当前消息列表
            placeholder_count: 已生成的占位符数量（用于统计）
            notes_count: 已生成的笔记数量（用于统计）

        Returns:
            压缩后的消息列表
        """
        self._last_stats = CompressStats(
            original_tokens=estimate_messages_tokens(messages),
            placeholder_count=placeholder_count,
            notes_generated=notes_count,
        )

        if not self._enabled:
            return messages

        current_tokens = self._last_stats.original_tokens

        if current_tokens <= self._max_tokens:
            # 未超限，不压缩
            self._last_stats.compressed_tokens = current_tokens
            return messages

        # ---- 执行压缩 ----
        # 生成被压缩消息的结构化预览
        preview = self._generate_preview(messages, keep_recent=self._keep_recent)
        self._last_preview = preview

        # 分离各类消息
        system_msgs: list[dict] = []
        user_msgs: list[dict] = []
        other_msgs: list[dict] = []

        for msg in messages:
            role = msg.get("role", "")
            if role == "system":
                system_msgs.append(msg)
            elif role == "user":
                user_msgs.append(msg)
            else:
                other_msgs.append(msg)

        # 保留：所有 system + 首条 user + 最近 N 条
        keep = system_msgs.copy()
        if user_msgs:
            keep.append(user_msgs[0])
        # 将最近的 assistant+tool 消息加入保留
        keep.extend(other_msgs[-(self._keep_recent):])

        # 被压缩的：不在 keep 中的消息
        keep_ids = {id(m) for m in keep}
        to_compress = [m for m in messages if id(m) not in keep_ids]

        if not to_compress:
            self._last_stats.compressed_tokens = estimate_messages_tokens(keep)
            return keep

        # LLM 摘要生成
        summary = self._generate_summary(to_compress)

        # 构建压缩后的消息列表
        compressed = system_msgs.copy()
        if user_msgs:
            compressed.append(user_msgs[0])
        # 插入预览作为 system 消息（轻量概述）
        if preview:
            compressed.append({
                "role": "system",
                "content": preview,
            })
        # 插入摘要作为 system 消息（详细结构化摘要）
        if summary:
            compressed.append({
                "role": "system",
                "content": f"[对话历史摘要]\n{summary}",
            })
        compressed.extend(other_msgs[-(self._keep_recent):])

        self._last_stats.compressed_tokens = estimate_messages_tokens(compressed)
        self._last_stats.summary_generated = bool(summary)

        # 累计指标
        self._total_compressions += 1
        self._total_tokens_saved += max(0, self._last_stats.original_tokens - self._last_stats.compressed_tokens)

        return compressed

    # ============================================================
    # LLM 摘要生成
    # ============================================================

    def _generate_preview(self, messages: list[dict], keep_recent: int = 8) -> str:
        """生成轻量预览：一行概述被压缩的消息范围（不消耗 LLM 调用）"""
        if not messages:
            return ""

        # 找到被压缩的范围：排除保留的最新 N 条
        non_system = [m for m in messages if m.get("role") != "system"]
        if len(non_system) <= keep_recent:
            return ""

        compressed_range = non_system[:-keep_recent] if keep_recent > 0 else non_system

        # 统计被压缩的消息类型
        roles: dict[str, int] = {}
        tool_names: set[str] = set()
        for m in compressed_range:
            role = m.get("role", "unknown")
            roles[role] = roles.get(role, 0) + 1
            if role == "tool":
                name = m.get("name", "")
                if name:
                    tool_names.add(name)

        parts = [f"共 {len(compressed_range)} 条旧消息"]
        for role, count in sorted(roles.items()):
            parts.append(f"{role}:{count}")
        if tool_names:
            parts.append(f"涉及工具: {', '.join(sorted(tool_names)[:6])}")

        return f"[上下文压缩] {' | '.join(parts)}"

    def get_cache_metrics(self) -> dict:
        """返回压缩缓存指标（用于 Prompt Cache 命中率分析）"""
        return {
            "total_compressions": self._total_compressions,
            "total_tokens_saved": self._total_tokens_saved,
            "last_original_tokens": self._last_stats.original_tokens,
            "last_compressed_tokens": self._last_stats.compressed_tokens,
            "last_savings_percent": self._last_stats.savings_percent,
            "last_preview": self._last_preview,
        }

    def _generate_summary(self, messages: list[dict]) -> str:
        """调用 LLM 生成结构化摘要"""
        if not self._llm or not messages:
            return self._rule_summary(messages)

        try:
            user_message = _build_summary_prompt(messages)
            response = self._llm.chat(
                messages=[
                    {"role": "system", "content": _SUMMARY_SYSTEM},
                    {"role": "user", "content": user_message},
                ],
                tools=[],
            )
            if response.content:
                return response.content.strip()[: self._max_summary_chars * 3]
        except Exception:
            pass

        return self._rule_summary(messages)

    def _rule_summary(self, messages: list[dict]) -> str:
        """LLM 不可用时的规则摘要回退"""
        tool_names = []
        user_requests = []
        files_touched = set()
        errors = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "") if isinstance(msg.get("content"), str) else ""

            if role == "user" and content:
                user_requests.append(content[:100])
            for tc in msg.get("tool_calls", []):
                name = tc.get("function", {}).get("name", "")
                if name:
                    tool_names.append(name)
            if role == "tool":
                name = msg.get("name", "")
                if name:
                    tool_names.append(name)
            if "文件" in content or "file" in content.lower():
                import re
                found = re.findall(r'[\w./-]+\.(py|js|ts|json|yaml|yml|md|txt)', content)
                files_touched.update(found)
            if "失败" in content or "error" in content.lower() or "错误" in content:
                errors.append(content[:80])

        parts = []
        if user_requests:
            parts.append(f"需求: {user_requests[0]}")
        if tool_names:
            unique = list(dict.fromkeys(tool_names))
            parts.append(f"完成: 使用了 {', '.join(unique[:8])}")
        if files_touched:
            parts.append(f"文件: {', '.join(list(files_touched)[:5])}")
        if errors:
            parts.append(f"问题: {errors[0]}")
        parts.append(f"状态: 共 {len(messages)} 条消息被压缩")

        return "。".join(parts)

    # ============================================================
    # 工具方法
    # ============================================================

    def get_last_stats(self) -> CompressStats:
        return self._last_stats

    @staticmethod
    def count_tokens(messages: list[dict]) -> int:
        """静态方法：估算消息列表 Token 数"""
        return estimate_messages_tokens(messages)
