"""
子Agent执行引擎 —— SubAgent

设计思路：
- 每个子Agent是一个受限的迷你Agent，只能使用白名单内的工具
- 不支持Skill路由、记忆、上下文压缩（保持简洁）
- 以同步方式运行，返回结构化结果
- 支持worktree隔离：在独立Git worktree中执行文件操作
"""

import json
import time
import os
import signal
from typing import Optional

from src.config import LLMConfig
from src.llm import LLMClient, LLMResponse, LLMError
from src.state import SessionState, ToolCallRecord
from src.tools.base import ToolRegistry, ToolResult
from src.logger import AgentLogger
from src.multi_agent.types import AgentType, SubAgentResult, AGENT_SYSTEM_PROMPTS


class SubAgentTimeout(Exception):
    """子Agent执行超时"""
    pass


class SubAgent:
    """
    受限子Agent —— 主Agent通过Tool Call触发执行。

    与主Agent的区别：
    - 无状态机（简单循环直到 DONE/MAX_ITER/ERROR）
    - 工具白名单限制（由Orchestrator在创建时指定）
    - 不接入记忆系统、Skill路由、上下文压缩
    - 执行完返回结构化结果，不持有会话
    """

    def __init__(
        self,
        agent_type: AgentType,
        llm_config: LLMConfig,
        registry: ToolRegistry,
        logger: AgentLogger,
        workspace_root: str = ".",
        max_iterations: int = 15,
    ):
        self._agent_type = agent_type
        self._llm = LLMClient(llm_config)
        self._registry = registry
        self._logger = logger
        self._workspace = workspace_root
        self._max_iterations = max_iterations

    def run(self, task: str, context: str = "", timeout_seconds: int = 300) -> SubAgentResult:
        """
        执行子任务，返回结构化结果。

        Args:
            task: 子任务描述
            context: 附加上下文（项目规范、相关代码片段等）
            timeout_seconds: 执行超时秒数（默认300秒）

        Returns:
            SubAgentResult: 结构化执行结果
        """
        start_time = time.perf_counter()
        state = SessionState(task=task)

        # 构建 System Prompt：角色定义 + 附加上下文
        system_prompt = AGENT_SYSTEM_PROMPTS.get(self._agent_type, "")
        if context:
            system_prompt += f"\n\n## 项目上下文\n{context}"

        state.add_system_message(system_prompt)
        state.add_user_message(task)

        self._logger.log_info(
            f"[SubAgent:{self._agent_type.value}] 开始执行: {task[:100]}..."
        )

        final_content = ""
        files_created: list[str] = []
        files_modified: list[str] = []

        try:
            for iteration in range(1, self._max_iterations + 1):
                # 超时检查
                elapsed = time.perf_counter() - start_time
                if elapsed > timeout_seconds:
                    files_created, files_modified = self._extract_file_changes(state)
                    return SubAgentResult(
                        success=False,
                        agent_type=self._agent_type.value,
                        task=task,
                        summary=f"执行超时 ({timeout_seconds}s), 已完成{iteration}轮",
                        files_created=files_created,
                        files_modified=files_modified,
                        error=f"SubAgentTimeout: {elapsed:.0f}s > {timeout_seconds}s",
                        iteration_count=iteration,
                        tokens_used=state.get_token_summary()["total_tokens"],
                    )

                # 获取可用工具
                tools = self._registry.get_schemas()
                messages = state.get_messages()

                try:
                    response = self._llm.chat(messages=messages, tools=tools)
                except LLMError as e:
                    return SubAgentResult(
                        success=False,
                        agent_type=self._agent_type.value,
                        task=task,
                        summary=f"LLM调用失败: {e}",
                        error=str(e),
                        iteration_count=iteration,
                    )

                state.add_tokens(response.prompt_tokens, response.completion_tokens)

                if response.content:
                    final_content = response.content

                # 无工具调用 → 完成
                if not response.has_tool_calls:
                    files_created, files_modified = self._extract_file_changes(state)
                    self._logger.log_info(
                        f"[SubAgent:{self._agent_type.value}] 完成, {iteration}轮迭代"
                    )
                    break

                # 执行工具调用
                self._execute_tool_calls(state, response)

            else:
                # 达到最大迭代次数
                self._logger.log_error(
                    f"[SubAgent:{self._agent_type.value}] 达到最大迭代次数 {self._max_iterations}"
                )
                files_created, files_modified = self._extract_file_changes(state)
                return SubAgentResult(
                    success=False,
                    agent_type=self._agent_type.value,
                    task=task,
                    summary=final_content[:300] if final_content else "达到最大迭代次数",
                    files_created=files_created,
                    files_modified=files_modified,
                    error=f"达到最大迭代次数 {self._max_iterations}",
                    iteration_count=self._max_iterations,
                    tokens_used=state.get_token_summary()["total_tokens"],
                )

        except Exception as e:
            self._logger.log_error(f"[SubAgent:{self._agent_type.value}] 异常: {e}")
            return SubAgentResult(
                success=False,
                agent_type=self._agent_type.value,
                task=task,
                summary=f"执行异常: {e}",
                error=str(e),
                iteration_count=0,
            )

        # 提取结果摘要
        summary = self._build_summary(final_content)
        tokens = state.get_token_summary()

        return SubAgentResult(
            success=True,
            agent_type=self._agent_type.value,
            task=task,
            summary=summary,
            files_created=files_created,
            files_modified=files_modified,
            iteration_count=state.iteration_count,
            tokens_used=tokens["total_tokens"],
        )

    def _execute_tool_calls(self, state: SessionState, response: LLMResponse):
        """执行LLM返回的工具调用（与主Agent逻辑一致但更简洁）"""
        # 构建 assistant 消息
        assistant_tool_calls = []
        for tc in response.tool_calls:
            assistant_tool_calls.append({
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["_name"],
                    "arguments": json.dumps(tc["_arguments"], ensure_ascii=False),
                }
            })

        state.add_assistant_message(
            content=response.content,
            tool_calls=assistant_tool_calls,
        )

        for tc in response.tool_calls:
            tool_name = tc["_name"]
            arguments = tc["_arguments"]
            tool_call_id = tc["id"]

            start = time.perf_counter()
            result: ToolResult = self._registry.execute(tool_name, arguments)
            duration = time.perf_counter() - start

            self._logger.log_tool_call(
                tool_name=tool_name,
                params=arguments,
                duration=duration,
                success=result.success,
                result_summary=result.content[:200] if result.content else "",
                error=result.error,
            )

            state.record_tool_call(ToolCallRecord(
                tool_name=tool_name,
                params=arguments,
                result_content=result.content,
                success=result.success,
                error=result.error,
                duration_ms=duration * 1000,
            ))

            state.add_tool_result(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                result=result.to_message(),
            )

    def _extract_file_changes(self, state: SessionState) -> tuple[list[str], list[str]]:
        """从工具调用日志中提取文件变更"""
        created = []
        modified = []
        for record in state.tool_call_logs:
            if not record.success:
                continue
            path = record.params.get("path", "") or record.params.get("file_path", "")
            if not path:
                continue
            if record.tool_name == "write_file":
                if path not in created:
                    created.append(path)
            elif record.tool_name == "edit_file":
                if path not in modified and path not in created:
                    modified.append(path)
        return created, modified

    def _build_summary(self, final_content: str) -> str:
        """从最终回复中提取摘要（截取前300字符的关键信息）"""
        if not final_content:
            return "无输出"
        # 去掉过长的代码块等，只保留文字描述
        return final_content[:300].replace("\n", " ").strip()
