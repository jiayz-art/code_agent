"""
LLM 客户端封装

设计思路：
- 基于 OpenAI SDK，因为千问 DashScope 完全兼容 OpenAI 接口
- 封装 chat() 方法，输入 messages + tools，返回结构化响应
- 自动解析 tool_calls，将 JSON 字符串参数转为 dict
- Token 用量通过 response.usage 返回给 Agent 层统计
- 网络错误、API 错误都在本层捕获并转为自定义异常，上层统一处理
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI

from src.config import LLMConfig


# ============================================================
# LLM 响应结构（平台无关，解耦 OpenAI SDK 的具体返回格式）
# ============================================================

@dataclass
class LLMResponse:
    """LLM 单次调用的结构化响应"""
    content: Optional[str] = None                # 文本回复（可能为空）
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # 工具调用列表
    finish_reason: str = ""                      # stop | tool_calls | length
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ============================================================
# LLM 客户端
# ============================================================

class LLMClient:
    """
    千问 LLM 客户端。

    使用 OpenAI SDK，base_url 指向 DashScope 兼容接口。
    所有 API 通信细节在本层封装，上层只需调用 chat()。
    """

    def __init__(self, config: LLMConfig):
        self._config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """
        发送对话请求，返回结构化响应。

        Args:
            messages: OpenAI 格式的消息列表
            tools: OpenAI 格式的工具定义列表

        Returns:
            LLMResponse: 包含 content / tool_calls / token 用量

        Raises:
            LLMError: API 调用失败时抛出
        """
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }

        # 只有 tools 不为空时才传，避免 API 报错
        if tools:
            kwargs["tools"] = tools

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise LLMError(f"LLM API 调用失败: {e}") from e

        return self._parse_response(response)

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_token: callable = None,
    ):
        """
        流式对话请求，逐 token yield。

        Args:
            messages: OpenAI 格式的消息列表
            tools: OpenAI 格式的工具定义列表
            on_token: 可选回调，接收 (delta_text: str)

        Yields:
            dict: {"type": "token", "text": "..."} 或 {"type": "done", "response": LLMResponse}
        """
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools

        try:
            stream = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise LLMError(f"LLM 流式调用失败: {e}") from e

        collected_content = ""
        collected_tool_calls: dict[int, dict] = {}
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason = ""

        for chunk in stream:
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta if hasattr(choice, 'delta') else None
            if delta is None:
                continue

            finish_reason = choice.finish_reason or finish_reason

            # 文本增量
            if delta.content:
                collected_content += delta.content
                if on_token:
                    on_token(delta.content)
                yield {"type": "token", "text": delta.content}

            # 工具调用增量
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in collected_tool_calls:
                        collected_tool_calls[idx] = {
                            "id": tc_delta.id or "",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc_delta.id:
                        collected_tool_calls[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            collected_tool_calls[idx]["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            collected_tool_calls[idx]["function"]["arguments"] += tc_delta.function.arguments

        # 构建最终响应
        tool_calls = []
        import json
        for tc_data in collected_tool_calls.values():
            try:
                arguments = json.loads(tc_data["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                arguments = {}
            tool_calls.append({
                "id": tc_data["id"],
                "type": "function",
                "function": {
                    "name": tc_data["function"]["name"],
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
                "_name": tc_data["function"]["name"],
                "_arguments": arguments,
            })

        response = LLMResponse(
            content=collected_content.strip() or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        yield {"type": "done", "response": response}

    def _parse_response(self, response) -> LLMResponse:
        """解析非流式响应"""
        import json
        choice = response.choices[0]
        msg = choice.message
        finish = choice.finish_reason or ""

        content = msg.content

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                    "_name": tc.function.name,
                    "_arguments": arguments,
                })

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class LLMError(Exception):
    """LLM 调用错误"""
    pass
