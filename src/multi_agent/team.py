"""
Agent Team 协作模式 —— AgentTeam

设计思路：
- 主Agent将复杂任务分解后，由AgentTeam协调多个子Agent协同完成
- 三种模式：sequential(串行流水线)、parallel(Fork/Join并行)、mixed(混合)
- 每阶段可选验证步骤：上一阶段的输出质量检查
- 返回统一的团队执行摘要，便于主Agent做最终决策
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

from src.multi_agent.types import AgentType, SubAgentResult


class TeamMode(str, Enum):
    SEQUENTIAL = "sequential"    # 串行流水线：上一阶段输出→下一阶段输入
    PARALLEL = "parallel"        # Fork/Join：所有子Agent并行执行
    MIXED = "mixed"              # 混合：先并行再串行（如 code+test→review→fix）


@dataclass
class TeamPhase:
    """团队协作的一个阶段"""
    agent_type: AgentType
    task_template: str           # 任务模板，{context} 会被替换为上一阶段输出
    depends_on: int = -1         # 依赖的阶段索引，-1 表示无依赖
    max_iterations: int = 15
    timeout_seconds: int = 300

    def build_task(self, context: str = "") -> str:
        return self.task_template.replace("{context}", context)


@dataclass
class TeamResult:
    """Agent Team 整体执行结果"""
    success: bool
    phases: list[SubAgentResult] = field(default_factory=list)
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    error: str = ""

    @property
    def files_created(self) -> list[str]:
        files = []
        for r in self.phases:
            files.extend(r.files_created)
        return files

    @property
    def files_modified(self) -> list[str]:
        files = []
        for r in self.phases:
            files.extend(r.files_modified)
        return files

    def to_summary(self) -> str:
        lines = [f"[AgentTeam 完成] {len(self.phases)}个阶段, 耗时{self.elapsed_seconds:.1f}s"]
        for i, r in enumerate(self.phases):
            status = "OK" if r.success else "FAIL"
            lines.append(f"  Phase{i}[{r.agent_type}] {status}: {r.summary[:150]}")
        if self.error:
            lines.append(f"  错误: {self.error}")
        lines.append(f"  总Token: {self.total_tokens}, 创建{len(self.files_created)}文件, 修改{len(self.files_modified)}文件")
        return "\n".join(lines)


class AgentTeam:
    """
    多Agent协作团队。

    协调多个子Agent按预定阶段执行复杂任务。
    支持三种协作模式：
    - sequential: 流水线，每阶段输出作为下一阶段上下文
    - parallel: Fork/Join 并行执行
    - mixed: 先并行(如 code+test) 再串行(如 review→fix)
    """

    def __init__(
        self,
        runner: Callable[[dict], SubAgentResult],
        mode: TeamMode = TeamMode.SEQUENTIAL,
        max_parallel: int = 4,
    ):
        self._runner = runner
        self._mode = mode
        self._max_parallel = max_parallel

    def execute(self, phases: list[TeamPhase], initial_context: str = "") -> TeamResult:
        """
        按阶段执行团队协作任务。

        Args:
            phases: 阶段列表（定义每个阶段的任务和依赖）
            initial_context: 初始上下文（项目规范、代码片段等）

        Returns:
            TeamResult: 团队整体执行结果
        """
        if self._mode == TeamMode.PARALLEL:
            return self._execute_parallel(phases, initial_context)
        elif self._mode == TeamMode.MIXED:
            return self._execute_mixed(phases, initial_context)
        else:
            return self._execute_sequential(phases, initial_context)

    # ============================================================
    # 串行流水线
    # ============================================================

    def _execute_sequential(
        self, phases: list[TeamPhase], initial_context: str
    ) -> TeamResult:
        result = TeamResult(success=True)
        start = time.perf_counter()
        context = initial_context

        for i, phase in enumerate(phases):
            task = phase.build_task(context)
            task_dict = {
                "agent_type": phase.agent_type.value,
                "task": task,
                "context": context,
                "max_iterations": phase.max_iterations,
            }

            try:
                sub_result = self._runner(task_dict)
                result.phases.append(sub_result)
                result.total_tokens += sub_result.tokens_used

                if not sub_result.success:
                    result.success = False
                    result.error = f"Phase{i}[{phase.agent_type.value}] 失败: {sub_result.error or sub_result.summary}"
                    break

                # 将本阶段输出作为下一阶段上下文
                context = self._build_stage_context(sub_result)

            except Exception as e:
                result.success = False
                result.error = f"Phase{i}[{phase.agent_type.value}] 异常: {e}"
                break

        result.elapsed_seconds = time.perf_counter() - start
        return result

    # ============================================================
    # Fork/Join 并行
    # ============================================================

    def _execute_parallel(
        self, phases: list[TeamPhase], initial_context: str
    ) -> TeamResult:
        result = TeamResult(success=True)
        start = time.perf_counter()

        tasks = []
        for phase in phases:
            tasks.append({
                "agent_type": phase.agent_type.value,
                "task": phase.build_task(initial_context),
                "context": initial_context,
                "max_iterations": phase.max_iterations,
            })

        max_workers = min(self._max_parallel, len(tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._runner, t): t for t in tasks}
            for future in as_completed(futures):
                try:
                    sub_result = future.result(timeout=600)
                    result.phases.append(sub_result)
                    result.total_tokens += sub_result.tokens_used
                    if not sub_result.success:
                        result.success = False
                except Exception as e:
                    task = futures[future]
                    result.success = False
                    result.error = f"并行任务异常 [{task['agent_type']}]: {e}"

        result.elapsed_seconds = time.perf_counter() - start
        return result

    # ============================================================
    # 混合模式：先并行再串行
    # ============================================================

    def _execute_mixed(
        self, phases: list[TeamPhase], initial_context: str
    ) -> TeamResult:
        result = TeamResult(success=True)
        start = time.perf_counter()
        context = initial_context

        # 分组：连续的依赖=0 的阶段并行，依赖>0 的串行
        i = 0
        while i < len(phases):
            # 收集当前并行组
            parallel_group = []
            while i < len(phases) and phases[i].depends_on <= 0:
                parallel_group.append(phases[i])
                i += 1

            if parallel_group:
                # 并行执行当前组
                tasks = []
                for phase in parallel_group:
                    tasks.append({
                        "agent_type": phase.agent_type.value,
                        "task": phase.build_task(context),
                        "context": context,
                        "max_iterations": phase.max_iterations,
                    })

                max_workers = min(self._max_parallel, len(tasks))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(self._runner, t): t for t in tasks}
                    for future in as_completed(futures):
                        try:
                            sub_result = future.result(timeout=600)
                            result.phases.append(sub_result)
                            result.total_tokens += sub_result.tokens_used
                            if not sub_result.success:
                                result.success = False
                            context = self._build_stage_context(sub_result)
                        except Exception as e:
                            task = futures[future]
                            result.success = False
                            result.error = f"并行异常 [{task['agent_type']}]: {e}"

                if not result.success:
                    break

            # 处理串行阶段（depends_on > 0）
            while i < len(phases) and phases[i].depends_on > 0:
                phase = phases[i]
                task = phase.build_task(context)
                task_dict = {
                    "agent_type": phase.agent_type.value,
                    "task": task,
                    "context": context,
                    "max_iterations": phase.max_iterations,
                }
                try:
                    sub_result = self._runner(task_dict)
                    result.phases.append(sub_result)
                    result.total_tokens += sub_result.tokens_used
                    if not sub_result.success:
                        result.success = False
                        result.error = f"Phase{i}[{phase.agent_type.value}] 失败"
                        break
                    context = self._build_stage_context(sub_result)
                except Exception as e:
                    result.success = False
                    result.error = f"Phase{i}[{phase.agent_type.value}] 异常: {e}"
                    break
                i += 1

            if not result.success:
                break

        result.elapsed_seconds = time.perf_counter() - start
        return result

    def _build_stage_context(self, sub_result: SubAgentResult) -> str:
        """将阶段结果压缩为下一阶段的上下文"""
        parts = [f"[上一阶段: {sub_result.agent_type}]", sub_result.summary]
        if sub_result.files_created:
            parts.append(f"创建: {', '.join(sub_result.files_created)}")
        if sub_result.files_modified:
            parts.append(f"修改: {', '.join(sub_result.files_modified)}")
        if sub_result.error:
            parts.append(f"错误: {sub_result.error}")
        return "\n".join(parts)
