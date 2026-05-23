"""
多Agent编排器 —— Orchestrator

设计思路：
- 不替代主Agent的思考循环，而是增强其System Prompt
- 提供任务拆解模板和子Agent使用指南
- 主Agent LLM 自主决定何时、如何分派子Agent
- 负责结果校验：检查子Agent返回的文件变更是否合规
- Fork/Join: 并行分派多个子Agent，主Agent负责归并结果
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Callable

from src.multi_agent.types import AgentType, SubAgentResult, AGENT_TOOL_WHITELIST


class ForkJoinResult:
    """Fork/Join 并行分派的归并结果"""

    def __init__(self):
        self.results: list[SubAgentResult] = []
        self.failed: list[tuple[str, str]] = []  # (agent_type, error)
        self.total_tokens: int = 0
        self.elapsed_seconds: float = 0.0

    @property
    def all_success(self) -> bool:
        return len(self.failed) == 0 and all(r.success for r in self.results)

    @property
    def files_created(self) -> list[str]:
        files = []
        for r in self.results:
            files.extend(r.files_created)
        return files

    @property
    def files_modified(self) -> list[str]:
        files = []
        for r in self.results:
            files.extend(r.files_modified)
        return files

    def to_summary(self) -> str:
        """生成归并摘要供主Agent LLM阅读"""
        lines = [f"[Fork/Join 完成] {len(self.results)}个子Agent完成, 耗时{self.elapsed_seconds:.1f}s"]
        for r in self.results:
            status = "OK" if r.success else "FAIL"
            lines.append(f"  [{r.agent_type}] {status}: {r.summary[:120]}")
        for at, err in self.failed:
            lines.append(f"  [{at}] ERROR: {err[:120]}")
        if self.total_tokens:
            lines.append(f"  总Token消耗: {self.total_tokens}")
        return "\n".join(lines)


class Orchestrator:
    """
    多Agent编排器。

    职责：
    - get_orchestration_prompt(): 返回注入主Agent System Prompt的编排指令
    - validate_result(): 校验子Agent结果（工具权限、文件范围）
    - fork_and_join(): 并行分派多个子Agent任务并归并结果
    - 不参与运行时调度（由主Agent LLM自主决策）
    """

    def __init__(self, enabled: bool = True, parallel_workers: int = 4):
        self._enabled = enabled
        self._parallel_workers = parallel_workers
        self._last_fork_result: Optional[ForkJoinResult] = None

    def get_orchestration_prompt(self) -> str:
        """返回注入主Agent System Prompt的多Agent编排指令"""
        if not self._enabled:
            return ""

        agent_desc = []
        for at in AgentType:
            tools = AGENT_TOOL_WHITELIST.get(at, set())
            agent_desc.append(f"  - **{at.value}**: 可用工具 [{', '.join(sorted(tools))}]")

        return f"""## 多Agent协作模式

你是一个主控Agent，可以通过 `delegate_agent` 工具将子任务分派给专用子Agent执行。

### 可用的子Agent类型
{chr(10).join(agent_desc)}

### 工作流程
1. **任务分析**: 判断任务复杂度。简单任务（单文件修改、简单查询）自己完成，复杂任务（多文件、跨模块）考虑分派
2. **任务拆解**: 将复杂任务拆解为独立的子任务，每个子任务分配给最合适的子Agent
3. **并行分派**: 相互独立的子任务应在一个回复中同时调用多个 delegate_agent（并行执行）
4. **结果审查**: 收到子Agent结果后，检查：
   - 文件变更是否合理（路径、范围是否符合预期）
   - reviewer 子Agent 发现的 issue 是否已解决
   - tester 子Agent 报告的失败是否需要修复
5. **集成决策**: 决定是接受变更还是要求子Agent重做

### 并行分派示例

当用户要求"开发一个用户管理CRUD接口"时：

**第1轮: 并行生成代码+测试**
- delegate_agent(agent_type="code_generator", task="创建User模型(model/user.py)、UserService(service/user_service.py)、UserController(controller/user_controller.py)，包含CRUD全部方法")
- delegate_agent(agent_type="tester", task="为User模型、UserService、UserController编写单元测试")

**第2轮: 审查质量**
- delegate_agent(agent_type="reviewer", task="审查第1轮生成的所有代码，检查安全、性能、风格")

**第3轮: 修复问题+写文档**
- delegate_agent(agent_type="code_generator", task="根据reviewer的审查报告修复问题")
- delegate_agent(agent_type="documenter", task="为CRUD接口生成API文档")

### 关键规则
- **子Agent不修改生产代码**: tester/reviewer/documenter 只能读代码，不能修改非测试/非文档文件
- **结果最小化**: 子Agent只返回摘要+文件列表，不返回完整代码（需要时用 read_file 查看）
- **worktree隔离**: 并行任务建议设置 worktree=true，避免文件冲突
- **控制权归你**: 只有你能做最终决策（合并、拒绝、要求重做）
- **简单任务自己来**: 如果任务只需要1-2个工具调用，不要分派子Agent"""

    def validate_result(self, result: SubAgentResult, requested_agent_type: AgentType) -> tuple[bool, str]:
        """
        校验子Agent结果是否合规。

        Returns:
            (是否通过, 原因说明)
        """
        if not result.success:
            return True, ""  # 失败的结果也需要主Agent看到

        # 检查文件变更是否在权限范围内
        whitelist = AGENT_TOOL_WHITELIST.get(requested_agent_type, set())

        # tester 不应该创建或修改非测试文件
        if requested_agent_type == AgentType.TESTER:
            for f in result.files_created + result.files_modified:
                if "test" not in f.lower() and "_test" not in f.lower():
                    return False, f"tester 子Agent 修改了非测试文件: {f}"

        # reviewer 不应该修改任何文件
        if requested_agent_type == AgentType.REVIEWER:
            if result.files_created or result.files_modified:
                return False, f"reviewer 子Agent 不应该修改文件: {result.files_created + result.files_modified}"

        return True, ""

    def validate_result_from_metadata(
        self, agent_type: AgentType, metadata: dict
    ) -> tuple[bool, str]:
        """从ToolResult.metadata校验子Agent结果（无需完整SubAgentResult对象）"""
        files_created = metadata.get("files_created", []) or []
        files_modified = metadata.get("files_modified", []) or []
        # 构造一个轻量的 SubAgentResult 用于复用校验逻辑
        dummy = SubAgentResult(
            success=True,
            agent_type=agent_type.value,
            task="",
            summary="",
            files_created=files_created,
            files_modified=files_modified,
        )
        return self.validate_result(dummy, agent_type)

    # ============================================================
    # Fork/Join 并行分派
    # ============================================================

    def fork_and_join(
        self,
        tasks: list[dict],
        runner: Callable[[dict], SubAgentResult],
        timeout_per_task: int = 120,
    ) -> ForkJoinResult:
        """
        Fork: 并行分派多个子任务到子Agent.
        Join: 等待全部完成并归并结果。

        Args:
            tasks: 任务列表，每项为 {agent_type, task, context, worktree}
            runner: 执行函数，签名为 (task_dict) -> SubAgentResult
            timeout_per_task: 每个子任务的超时秒数

        Returns:
            ForkJoinResult 包含所有子Agent的执行结果
        """
        result = ForkJoinResult()
        start = time.perf_counter()

        if not self._enabled or not tasks:
            result.elapsed_seconds = time.perf_counter() - start
            return result

        max_workers = min(self._parallel_workers, len(tasks))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, task in enumerate(tasks):
                future = executor.submit(runner, task)
                futures[future] = (i, task)

            for future in as_completed(futures, timeout=timeout_per_task * len(tasks)):
                idx, task = futures[future]
                try:
                    sub_result: SubAgentResult = future.result(timeout=timeout_per_task)
                    result.results.append(sub_result)
                    result.total_tokens += sub_result.tokens_used
                except Exception as e:
                    result.failed.append((
                        task.get("agent_type", f"task_{idx}"),
                        str(e),
                    ))

        result.elapsed_seconds = time.perf_counter() - start
        self._last_fork_result = result
        return result

    def get_last_fork_result(self) -> Optional[ForkJoinResult]:
        return self._last_fork_result
