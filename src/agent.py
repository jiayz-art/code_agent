"""
核心 Agent —— 状态机编排引擎

设计思路：
- Agent 只做编排，不处理细节：LLM 调用 → 工具执行 → 结果反馈 → 循环
- 状态转移清晰：INIT → THINKING → EXECUTING → DONE/MAX_ITER/ERROR
- 每轮迭代都检查迭代上限，防止死循环
- 所有工具调用都有日志记录（工具名、参数、耗时、结果）
- System Prompt 是唯一内置提示词，定义 Agent 的行为规范
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum, auto
from typing import Optional

from src.config import AppConfig
from src.llm import LLMClient, LLMResponse, LLMError
from src.state import SessionState, ToolCallRecord
from src.tools.base import ToolRegistry, ToolResult
from src.logger import AgentLogger
from src.output_filter import UserOutput

# SkillRouter 为可选依赖，不导入则回退到全量工具模式
try:
    from src.skills.router import SkillRouter
    _SKILL_ROUTER_AVAILABLE = True
except ImportError:
    SkillRouter = None  # type: ignore
    _SKILL_ROUTER_AVAILABLE = False

# MemoryManager 为可选依赖
try:
    from src.memory.manager import MemoryManager
    _MEMORY_AVAILABLE = True
except ImportError:
    MemoryManager = None  # type: ignore
    _MEMORY_AVAILABLE = False

# ContextManager 为可选依赖
try:
    from src.context.manager import ContextManager
    _CONTEXT_AVAILABLE = True
except ImportError:
    ContextManager = None  # type: ignore
    _CONTEXT_AVAILABLE = False


# ============================================================
# Agent 状态枚举
# ============================================================

class AgentState(Enum):
    """Agent 状态机状态"""
    INIT = auto()         # 初始状态，等待用户输入
    THINKING = auto()     # LLM 推理中
    EXECUTING = auto()    # 执行工具调用中
    DONE = auto()         # 任务完成
    MAX_ITER = auto()     # 达到最大迭代次数
    ERROR = auto()        # 发生不可恢复的错误


# ============================================================
# System Prompt —— Agent 的行为准则
# ============================================================

SYSTEM_PROMPT = """你是一个 AI Coding Agent，帮助用户完成编程任务。

## 核心原则
- 使用工具读取文件、编辑代码、执行命令、搜索代码库
- 修改代码前先阅读相关文件理解现有逻辑
- 使用 edit_file 进行精确修改，而非重写整个文件
- 每次工具调用后分析结果再决定下一步
- 用中文回复，修改后简要说明做了什么

## 诚实汇报原则（极其重要）
- 工具返回"[工具执行失败]"或"[工具执行被安全策略阻止]"时，操作并未执行
- 绝对不要声称"文件已创建"、"修改已完成"等，除非工具返回了"[工具执行成功]"
- 如果工具被安全策略阻止，如实告知用户并建议替代方案（如使用工作区内路径）
- 如果工具返回"[写入验证]"确认了文件存在，才能说文件创建成功
- 遇到不确定的情况，优先读回文件验证，而非假设操作成功

## 禁止行为
- 不执行 rm -rf / 等危险命令
- 不修改工作区以外的文件
- 不在没有理解代码的情况下盲目修改
- 不在工具执行失败时对用户撒谎说操作成功
"""


# ============================================================
# Agent 核心类
# ============================================================

class Agent:
    """
    AI Coding Agent 核心引擎。

    职责：接收用户任务 → 循环调用 LLM → 执行工具 → 返回结果
    依赖：LLMClient（模型通信）、ToolRegistry（工具执行）、SessionState（状态管理）
    """

    def __init__(self, config: AppConfig, logger: AgentLogger, registry: ToolRegistry, skill_router=None, memory_manager=None, context_manager=None, orchestrator=None, conversation_memory=None, project: str = ""):
        self._config = config
        self._logger = logger
        self._registry = registry
        self._llm = LLMClient(config.llm)
        self._skill_router = skill_router  # 可选：SkillRouter 实例
        self._memory = memory_manager      # 可选：MemoryManager 实例
        self._context_mgr = context_manager  # 可选：ContextManager 实例
        self._orchestrator = orchestrator  # 可选：Orchestrator 实例
        self._conv_memory = conversation_memory  # 可选：ConversationMemory 实例（跨轮短期记忆）
        self._project = project or self._extract_project_name()
        self._state: Optional[SessionState] = None
        self._agent_state = AgentState.INIT
        # 首轮路由标记
        self._routing_done = False
        self._routed_tool_names: set[str] = set()
        self._tool_cache: dict[str, tuple[float, ToolResult]] = {}  # key -> (timestamp, result)
        self._last_responses: list[str] = []  # 最近几轮的 LLM 文本回复

    # ============================================================
    # 主循环 —— 状态机
    # ============================================================

    def run(self, task: str) -> str:
        """
        执行一个用户任务，返回最终响应。

        这是 Agent 的唯一公开入口。调用方无需了解内部状态机细节。
        """
        # 初始化会话
        self._state = SessionState(task=task)
        self._state.add_system_message(SYSTEM_PROMPT)
        self._agent_state = AgentState.THINKING
        self._routing_done = False
        self._routed_tool_names = set()

        # ★ 多Agent编排：注入子Agent分派指令
        if self._orchestrator is not None:
            orch_prompt = self._orchestrator.get_orchestration_prompt()
            if orch_prompt:
                self._state.add_system_message(orch_prompt)
                self._logger.log_info("多Agent编排指令已注入")

        # ★ 首轮路由：确定可用 Skill 集，注入上下文提示词
        if self._skill_router is not None:
            skill_names = self._skill_router.route(task)
            context_prompt = self._skill_router.get_context_prompt(skill_names)
            if context_prompt:
                self._state.add_system_message(context_prompt)
            self._routed_tool_names = self._skill_router.get_active_tool_names()
            self._logger.log_info(
                f"路由完成: {len(skill_names)} 个 Skill, "
                f"{len(self._routed_tool_names)} 个底层工具"
            )
        self._routing_done = True
        self._state.add_user_message(task)

        # ★ 记忆复用：检索历史经验并注入上下文
        if self._memory is not None:
            memory_context = self._memory.before_task(task, self._project)
            if memory_context:
                self._state.add_system_message(memory_context)
                self._logger.log_info("记忆上下文已注入")

        # ★ 跨轮对话记忆：注入上轮操作的文件和上下文
        if self._conv_memory is not None:
            conv_context = self._conv_memory.get_context_prompt()
            if conv_context:
                self._state.add_system_message(conv_context)
                self._logger.log_info("跨轮对话上下文已注入")

        # ★ 安全审查：设置当前任务上下文（供AI风险检测使用）
        if self._registry.audit_chain is not None:
            self._registry.audit_chain.set_session(
                session_id=str(id(self._state)), task=task
            )

        self._logger.log_info(f"任务开始: {task[:100]}...")
        final_response = ""

        # 主循环
        for iteration in range(1, self._config.agent.max_iterations + 1):
            try:
                self._agent_state = AgentState.THINKING

                # 1. 调用 LLM 前处理占位符展开请求
                messages = self._state.get_messages()
                if self._context_mgr is not None and messages:
                    last_msg = messages[-1]
                    if last_msg.get("role") == "user":
                        content = last_msg.get("content", "")
                        if isinstance(content, str) and "展开 PLACEHOLDER:" in content:
                            self._expand_placeholders_in_messages(content)

                # 2. 调用 LLM
                llm_response = self._call_llm()

                # 2. 如果有文本内容，输出给用户
                if llm_response.content:
                    final_response = llm_response.content

                # 有工具调用时重置响应记录
                if llm_response.has_tool_calls:
                    self._last_responses = []

                # 无工具调用：任务完成判断
                if not llm_response.has_tool_calls:
                    self._last_responses.append(llm_response.content or "")
                    if len(self._last_responses) > 3:
                        self._last_responses.pop(0)
                    if (len(self._last_responses) >= 2 and
                        self._last_responses[-1] == self._last_responses[-2]):
                        self._agent_state = AgentState.DONE
                        self._logger.log_info("任务完成（提前终止：连续相同回复）")
                        final_response = llm_response.content or final_response
                        break
                    else:
                        self._agent_state = AgentState.DONE
                        self._logger.log_info("任务完成（无工具调用）")
                        final_response = llm_response.content or final_response
                        break

                # 4. 有工具调用 → 逐一执行
                self._agent_state = AgentState.EXECUTING
                self._execute_tool_calls(llm_response)

            except LLMError as e:
                self._agent_state = AgentState.ERROR
                self._logger.log_error(f"LLM 调用失败: {e}", exc_info=True)
                final_response = UserOutput.wrap_error(str(e))
                break
            except Exception as e:
                self._agent_state = AgentState.ERROR
                self._logger.log_error(f"Agent 内部异常: {e}", exc_info=True)
                final_response = UserOutput.wrap_error(str(e))
                break
        else:
            # 循环正常结束（达到 max_iterations）
            self._agent_state = AgentState.MAX_ITER
            self._logger.log_error(f"达到最大迭代次数: {self._config.agent.max_iterations}")
            final_response = UserOutput.wrap(
                f"任务未完成：已达到最大工具调用轮数 "
                f"({self._config.agent.max_iterations})。"
                f"当前结果：\n{final_response}",
                success=False,
            )

        # ★ 记忆沉淀：反思本次任务并存储经验
        if self._memory is not None and self._state is not None:
            success = self._agent_state in (AgentState.DONE,)
            try:
                self._memory.after_task(
                    task=task,
                    project=self._project,
                    tool_logs=self._state.tool_call_logs,
                    success=success,
                )
            except Exception as e:
                self._logger.log_error(f"记忆沉淀失败: {e}")

        # ★ 跨轮对话记忆：记录本轮操作
        if self._conv_memory is not None and self._state is not None:
            self._record_conversation_turn(task, final_response)

        # ★ 用户输出净化：过滤内部日志、翻译错误
        success = self._agent_state in (AgentState.DONE,)
        workspace = str(self._config.agent.workspace_root)
        final_response = UserOutput.wrap(final_response, success=success, workspace=workspace)

        # 输出任务统计
        self._log_summary()
        return final_response

    # ============================================================
    # 内部方法
    # ============================================================

    def _call_llm(self) -> LLMResponse:
        """调用 LLM，记录 token 消耗。通过 SkillRouter 裁剪 tools 列表降低 Token 开销。"""
        all_schemas = self._registry.get_schemas()

        # 路由过滤：如果启用了 SkillRouter，仅传选中 Skill 的底层 tools
        if self._skill_router is not None and self._routed_tool_names:
            # 确保基础工具始终可用
            core_tools = self._routed_tool_names | {"read_file", "grep_search"}
            tools = self._registry.get_schemas_for(core_tools)
        else:
            tools = all_schemas

        messages = self._state.get_messages()

        # 上下文压缩：LLM 调用前执行占位替换 + 超限压缩
        if self._context_mgr is not None:
            messages = self._context_mgr.prepare_messages(messages)

        response = self._llm.chat(messages=messages, tools=tools)

        # 记录 token 消耗
        self._state.add_tokens(response.prompt_tokens, response.completion_tokens)
        self._logger.log_tokens(response.prompt_tokens, response.completion_tokens)

        # 按需追加：LLM 请求了不在当前候选集的工具 → 动态扩展
        if self._skill_router is not None and response.has_tool_calls:
            for tc in response.tool_calls:
                tool_name = tc["_name"]
                if tool_name not in self._routed_tool_names:
                    added = self._skill_router.on_demand_add(tool_name)
                    if added:
                        self._logger.log_info(f"按需追加工具: {tool_name}")
            # 更新活跃工具集（一次更新，不在循环内重复计算）
            self._routed_tool_names = self._skill_router.get_active_tool_names()

        return response

    def _execute_tool_calls(self, llm_response: LLMResponse):
        """执行 LLM 返回的所有工具调用，无依赖的并行执行"""
        # 构建 assistant 消息（含 tool_calls）加入历史
        assistant_tool_calls = []
        for tc in llm_response.tool_calls:
            assistant_tool_calls.append({
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["_name"],
                    "arguments": json.dumps(tc["_arguments"], ensure_ascii=False),
                }
            })

        self._state.add_assistant_message(
            content=llm_response.content,
            tool_calls=assistant_tool_calls,
        )

        # 判断哪些工具调用可以并行（只读工具可并行，写操作串行）
        tool_calls = llm_response.tool_calls
        if len(tool_calls) <= 1:
            # 单个工具调用，直接串行
            for tc in tool_calls:
                self._execute_single_tool(tc)
        else:
            # 分组：只读工具并行执行，写操作逐个串行
            read_only = {"read_file", "grep_search", "git_status", "git_diff"}
            parallel_calls = [tc for tc in tool_calls if tc["_name"] in read_only]
            serial_calls = [tc for tc in tool_calls if tc["_name"] not in read_only]

            # 只读工具并行
            if parallel_calls:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {
                        executor.submit(self._execute_single_tool, tc): tc
                        for tc in parallel_calls
                    }
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            tc = futures[future]
                            self._logger.log_error(
                                f"并行工具执行异常 {tc['_name']}: {e}"
                            )

            # 写操作串行（保持顺序，避免文件冲突）
            for tc in serial_calls:
                self._execute_single_tool(tc)

    def _execute_single_tool(self, tc: dict):
        """执行单个工具调用（含缓存检查）"""
        tool_name = tc["_name"]
        arguments = tc["_arguments"]
        tool_call_id = tc["id"]

        # 只读工具缓存检查（30秒内相同参数复用）
        if tool_name in ("read_file", "grep_search", "git_status", "git_diff"):
            cache_key = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
            cached = self._tool_cache.get(cache_key)
            if cached and (time.time() - cached[0]) < 30:
                self._logger.log_info(f"缓存命中: {tool_name}")
                result = cached[1]
                self._state.record_tool_call(ToolCallRecord(
                    tool_name=tool_name,
                    params=arguments,
                    result_content=result.content,
                    success=result.success,
                    error=result.error,
                    duration_ms=0,
                ))
                self._state.add_tool_result(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    result=result.to_message(),
                )
                return

        start = time.perf_counter()
        result: ToolResult = self._registry.execute(tool_name, arguments)
        duration = time.perf_counter() - start

        # 缓存只读工具结果
        if tool_name in ("read_file", "grep_search", "git_status", "git_diff"):
            cache_key = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
            self._tool_cache[cache_key] = (time.time(), result)
            # 限制缓存大小
            if len(self._tool_cache) > 20:
                oldest_key = min(self._tool_cache, key=lambda k: self._tool_cache[k][0])
                del self._tool_cache[oldest_key]

        # 多Agent协作：对 delegate_agent 结果进行校验
        if tool_name == "delegate_agent" and self._orchestrator is not None:
            if result.success and result.metadata:
                from src.multi_agent.types import AgentType
                try:
                    agent_type = AgentType(result.metadata.get("agent_type", ""))
                    sub_result = result.content
                    is_valid, reason = self._orchestrator.validate_result_from_metadata(
                        agent_type, result.metadata
                    )
                    if not is_valid:
                        self._logger.log_info(
                            f"子Agent结果校验失败: {reason}"
                        )
                except (ValueError, TypeError):
                    pass

        # 记录日志
        self._logger.log_tool_call(
            tool_name=tool_name,
            params=arguments,
            duration=duration,
            success=result.success,
            result_summary=result.content[:200] if result.content else "",
            error=result.error,
        )

        # 记录到会话状态
        self._state.record_tool_call(ToolCallRecord(
            tool_name=tool_name,
            params=arguments,
            result_content=result.content,
            success=result.success,
            error=result.error,
            duration_ms=duration * 1000,
        ))

        # 工具结果反馈给 LLM
        result_content = result.to_message()
        if self._context_mgr is not None:
            result_content = self._context_mgr.process_tool_result(
                tool_name=tool_name,
                params=arguments,
                result_content=result_content,
                success=result.success,
            )
        self._state.add_tool_result(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            result=result_content,
        )

    def _log_summary(self):
        """输出任务总结日志"""
        if self._state is None:
            return
        elapsed = self._state.get_elapsed_seconds()
        tokens = self._state.get_token_summary()
        tool_calls = len(self._state.tool_call_logs)
        self._logger.log_info(
            f"任务结束 | 状态: {self._agent_state.name} | "
            f"耗时: {elapsed:.1f}s | 迭代: {tokens['iterations']}轮 | "
            f"工具调用: {tool_calls}次 | "
            f"Token: {tokens['total_tokens']} "
            f"(prompt={tokens['prompt_tokens']}, completion={tokens['completion_tokens']})"
        )

    def _record_conversation_turn(self, task: str, final_response: str):
        """将本轮对话操作录入跨轮记忆"""
        from src.conversation_memory import TurnRecord

        files_created = []
        files_modified = []
        files_read = []
        error_summary = ""

        for log in self._state.tool_call_logs:
            if log.tool_name == "write_file":
                if log.success:
                    if log.params.get("path"):
                        files_created.append(log.params["path"])
                else:
                    error_summary = log.error or "文件写入失败"
            elif log.tool_name == "edit_file":
                if log.success:
                    if log.params.get("path"):
                        files_modified.append(log.params["path"])
                else:
                    error_summary = log.error or "文件编辑失败"
            elif log.tool_name == "read_file":
                if log.params.get("path"):
                    files_read.append(log.params["path"])

        success = self._agent_state in (AgentState.DONE,)
        turn = TurnRecord(
            task=task,
            response=final_response[:300],
            files_created=files_created,
            files_modified=files_modified,
            files_read=files_read,
            success=success,
            error_summary=error_summary,
        )
        self._conv_memory.record_turn(turn)

    def _extract_project_name(self) -> str:
        """从 workspace_root 中提取项目名"""
        import os
        workspace = self._config.agent.workspace_root
        if workspace == ".":
            workspace = os.getcwd()
        return os.path.basename(os.path.abspath(workspace)) or "unknown"

    def _expand_placeholders_in_messages(self, user_content: str):
        """解析用户「展开 PLACEHOLDER:xxx」请求，将原始内容注入消息流"""
        import re
        if self._context_mgr is None:
            return
        pids = re.findall(r"PLACEHOLDER:(\w+)", user_content)
        if not pids:
            return
        parts = []
        for pid in pids:
            content = self._context_mgr.resolve_placeholder(pid)
            if content:
                parts.append(f"[已展开 PLACEHOLDER:{pid} 完整内容]\n{content}")
            else:
                parts.append(f"[PLACEHOLDER:{pid} 不存在或已过期]")
        if parts:
            self._state.add_system_message("\n\n".join(parts))
            self._logger.log_info(f"占位符展开: {len(pids)} 个 ({', '.join(pids[:5])})")

    @property
    def state(self) -> AgentState:
        """当前 Agent 状态"""
        return self._agent_state

    @property
    def session(self) -> Optional[SessionState]:
        """当前会话状态（调试用）"""
        return self._state
