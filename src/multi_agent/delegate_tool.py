"""
委托工具 —— DelegateAgentTool

设计思路：
- 以 BaseTool 形式注册到主Agent的ToolRegistry
- 主Agent LLM 可像调用普通工具一样调用 delegate_agent
- 内部创建受限 SubAgent → 执行 → 返回结构化结果
- 支持 Git Worktree 隔离，并行子任务互不干扰
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.llm import LLMClient
from src.safety import SafetyChecker
from src.tools.base import BaseTool, ToolRegistry, ToolResult
from src.tools.file_tools import ReadFileTool, WriteFileTool, EditFileTool
from src.tools.shell_tools import RunCommandTool
from src.tools.git_tools import GitStatusTool, GitDiffTool, GitCommitTool
from src.tools.search_tools import GrepSearchTool
from src.logger import AgentLogger
from src.multi_agent.types import (
    AgentType, SubAgentResult, DelegateRequest, AGENT_TOOL_WHITELIST,
)
from src.multi_agent.sub_agent import SubAgent


# 工具类名 → 类对象的映射（用于动态创建受限Registry）
_TOOL_REGISTRY_MAP: dict[str, type[BaseTool]] = {
    "read_file": ReadFileTool,
    "write_file": WriteFileTool,
    "edit_file": EditFileTool,
    "run_command": RunCommandTool,
    "git_status": GitStatusTool,
    "git_diff": GitDiffTool,
    "git_commit": GitCommitTool,
    "grep_search": GrepSearchTool,
}


class DelegateAgentTool(BaseTool):
    """
    委托子Agent工具 —— 主Agent调用此工具将子任务分派给专用子Agent。

    LLM 看到的工具定义（OpenAI格式）由 get_schema() 返回，
    内部 execute() 创建并运行 SubAgent。
    """

    name = "delegate_agent"
    description = (
        "将子任务分派给专用子Agent执行。支持并行分派（一次调用多个delegate_agent）。"
        "子Agent类型: code_generator(代码生成)、tester(测试)、reviewer(审查)、documenter(文档)。"
        "不同子Agent有不同的工具权限，确保安全隔离。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "agent_type": {
                "type": "string",
                "enum": ["code_generator", "tester", "reviewer", "documenter"],
                "description": "子Agent类型，决定可用的工具集和System Prompt",
            },
            "task": {
                "type": "string",
                "description": "分配给子Agent的具体任务描述，越具体越好",
            },
            "context": {
                "type": "string",
                "description": "附加上下文：项目规范、相关代码片段、文件列表等",
            },
            "worktree": {
                "type": "boolean",
                "description": "是否在Git Worktree中隔离执行，并行任务建议开启",
            },
        },
        "required": ["agent_type", "task"],
    }

    def __init__(self, safety: SafetyChecker, config: AppConfig, logger: AgentLogger):
        super().__init__(safety)
        self._config = config
        self._logger = logger
        self._workspace = Path(config.agent.workspace_root).resolve()
        # 管理的worktree路径（用于清理）
        self._active_worktrees: list[Path] = []

    def execute(self, agent_type: str, task: str, context: str = "", worktree: bool = False) -> ToolResult:
        """
        执行委托任务。

        1. 解析 agent_type
        2. 创建受限 ToolRegistry（白名单过滤）
        3. 如果 worktree=True，创建 Git Worktree 隔离环境
        4. 创建并运行 SubAgent
        5. 返回结构化结果
        """
        try:
            at = AgentType(agent_type)
        except ValueError:
            return ToolResult(
                success=False,
                content=f"未知的子Agent类型: '{agent_type}'。可选: {[t.value for t in AgentType]}",
                error=f"Invalid agent_type: {agent_type}",
            )

        # 创建受限的 ToolRegistry
        worktree_path = None
        workspace_root = str(self._workspace)

        if worktree:
            wt_result = self._create_worktree()
            if wt_result[0]:
                worktree_path = wt_result[1]
                workspace_root = str(worktree_path)
                self._logger.log_info(f"[Delegate] Worktree 创建: {worktree_path}")
            else:
                return ToolResult(
                    success=False,
                    content=f"Worktree 创建失败: {wt_result[1]}",
                    error=wt_result[1],
                )

        try:
            registry = self._build_restricted_registry(at, workspace_root)
            self._logger.log_info(
                f"[Delegate] 子Agent类型={at.value}, "
                f"可用工具={registry.list_tools()}, "
                f"worktree={worktree_path is not None}"
            )

            # 创建并运行 SubAgent
            sub = SubAgent(
                agent_type=at,
                llm_config=self._config.llm,
                registry=registry,
                logger=self._logger,
                workspace_root=workspace_root,
                max_iterations=15,
            )
            result: SubAgentResult = sub.run(task=task, context=context)

            # 在摘要中附加 worktree 信息
            if worktree_path and result.success:
                result.summary += f" (worktree: {worktree_path})"

            self._logger.log_info(
                f"[Delegate] {at.value} 完成: success={result.success}, "
                f"files_created={len(result.files_created)}, "
                f"files_modified={len(result.files_modified)}, "
                f"tokens={result.tokens_used}"
            )

            return ToolResult(
                success=result.success,
                content=result.to_message(),
                metadata={
                    "agent_type": result.agent_type,
                    "files_created": result.files_created,
                    "files_modified": result.files_modified,
                    "worktree": str(worktree_path) if worktree_path else None,
                    "tokens_used": result.tokens_used,
                },
            )

        finally:
            # 如果使用了 worktree，保留给主Agent检查（不立即清理）
            if worktree_path:
                self._active_worktrees.append(worktree_path)

    def _build_restricted_registry(self, agent_type: AgentType, workspace_root: str) -> ToolRegistry:
        """构建受限于白名单的ToolRegistry"""
        from src.config import AppConfig

        # 构建一个临时 SafetyChecker，scope 到正确的 workspace
        import copy
        safety_config = copy.deepcopy(self._config)
        safety_config.agent.workspace_root = workspace_root
        restricted_safety = SafetyChecker(safety_config)

        registry = ToolRegistry(restricted_safety)
        whitelist = AGENT_TOOL_WHITELIST.get(agent_type, set())

        for tool_name, tool_class in _TOOL_REGISTRY_MAP.items():
            if tool_name in whitelist:
                registry.register(tool_class)

        return registry

    def _create_worktree(self) -> tuple[bool, str]:
        """创建 Git Worktree 用于隔离执行"""
        # 检查是否在 Git 仓库中
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=str(self._workspace),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return False, "当前目录不在Git仓库中，无法创建Worktree"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False, "Git命令不可用"

        # 生成唯一的worktree路径
        import uuid
        wt_id = uuid.uuid4().hex[:8]
        wt_path = self._workspace / ".claude" / "worktrees" / f"agent_{wt_id}"
        wt_path.parent.mkdir(parents=True, exist_ok=True)

        # 获取当前分支作为base
        try:
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(self._workspace),
                capture_output=True,
                text=True,
                timeout=5,
            )
            base_branch = branch_result.stdout.strip() or "main"
        except Exception:
            base_branch = "main"

        # 创建worktree
        try:
            result = subprocess.run(
                ["git", "worktree", "add", str(wt_path), base_branch],
                cwd=str(self._workspace),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return False, f"git worktree add 失败: {result.stderr}"
        except subprocess.TimeoutExpired:
            return False, "git worktree add 超时"

        return True, wt_path

    def cleanup_worktrees(self):
        """清理所有活跃的worktree"""
        for wt_path in self._active_worktrees:
            try:
                # 先移除worktree注册
                subprocess.run(
                    ["git", "worktree", "remove", str(wt_path), "--force"],
                    cwd=str(self._workspace),
                    capture_output=True,
                    timeout=10,
                )
                # 再删除残留目录
                if wt_path.exists():
                    shutil.rmtree(str(wt_path), ignore_errors=True)
                self._logger.log_info(f"[Delegate] Worktree 已清理: {wt_path}")
            except Exception as e:
                self._logger.log_error(f"[Delegate] Worktree 清理失败 {wt_path}: {e}")
        self._active_worktrees.clear()

    @property
    def active_worktrees(self) -> list[Path]:
        return list(self._active_worktrees)
