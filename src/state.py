"""
会话状态管理模块

设计思路：
- 用 dataclass 而非字典管理状态，字段可见、类型可检查
- 对话历史严格遵循 OpenAI messages 格式：system / user / assistant / tool 四种角色
- tool_call_logs 是结构化的执行记录，方便后续统计和调试
- Token 消耗按轮次累计，最终可汇总输出
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional


# ============================================================
# 单次工具调用记录
# ============================================================

@dataclass
class ToolCallRecord:
    """单次工具调用的完整记录"""
    tool_name: str
    params: dict[str, Any]
    result_content: str
    success: bool
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


# ============================================================
# 会话状态
# ============================================================

@dataclass
class SessionState:
    """
    一次 Agent 对话的完整状态。

    messages 是核心数据结构，格式为 OpenAI API 兼容的 dict 列表：
        {"role": "system", "content": "..."}
        {"role": "user", "content": "..."}
        {"role": "assistant", "content": "...", "tool_calls": [...]}
        {"role": "tool", "tool_call_id": "...", "content": "..."}
    """

    # 对话历史（OpenAI messages 格式）
    messages: list[dict[str, Any]] = field(default_factory=list)

    # 工具调用日志（结构化记录，用于调试和统计）
    tool_call_logs: list[ToolCallRecord] = field(default_factory=list)

    # Token 消耗累计
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    # 时间统计
    start_time: float = field(default_factory=time.time)
    iteration_count: int = 0

    # 当前任务描述
    task: str = ""

    def add_system_message(self, content: str):
        """添加 system 角色消息（通常只有一条，在对话开始时设置）"""
        self.messages.append({"role": "system", "content": content})

    def add_user_message(self, content: str):
        """添加 user 角色消息"""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: Optional[str], tool_calls: Optional[list[dict]] = None):
        """
        添加 assistant 角色消息。

        content 和 tool_calls 至少有一个不为空：
        - 纯文本回复时只有 content
        - 工具调用时 content 可能为 None, tool_calls 包含调用列表
        """
        msg: dict[str, Any] = {"role": "assistant"}
        if content:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    def add_tool_result(self, tool_call_id: str, tool_name: str, result: str):
        """添加 tool 角色消息（工具执行结果反馈）"""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        })

    def add_tokens(self, prompt: int, completion: int):
        """累计 Token 消耗"""
        self.total_prompt_tokens += prompt
        self.total_completion_tokens += completion
        self.iteration_count += 1

    def record_tool_call(self, record: ToolCallRecord):
        """记录一次工具调用"""
        self.tool_call_logs.append(record)

    def get_elapsed_seconds(self) -> float:
        """获取已耗时（秒）"""
        return time.time() - self.start_time

    def get_messages(self) -> list[dict[str, Any]]:
        """返回对话历史（给 LLM 客户端调用）"""
        return self.messages

    def get_token_summary(self) -> dict[str, int]:
        """返回 Token 消耗汇总"""
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "iterations": self.iteration_count,
        }
