"""
Git 操作工具集：git_status / git_diff / git_commit

设计思路：
- 不直接用 gitpython 等第三方库，而是通过 git CLI 命令，
  因为 CLI 输出格式稳定，LLM 可以直接理解
- git_status 返回简洁的状态摘要
- git_diff 支持查看工作区或暂存区的变更
- git_commit 只允许提交，禁止 force push 等危险操作
- 受保护分支（main/master/release/*）拒绝直接提交，提醒走 PR 流程
"""

import subprocess
from pathlib import Path

from src.tools.base import BaseTool, ToolResult
from src.safety import SafetyChecker


def _run_git(args: list[str], cwd: str = ".", timeout: int = 30) -> tuple[int, str, str]:
    """执行 git 命令的内部辅助函数"""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Git 命令超时"
    except FileNotFoundError:
        return -1, "", "Git 未安装或不在 PATH 中"


class GitStatusTool(BaseTool):
    name = "git_status"
    description = "查看 Git 仓库当前状态：已修改、已暂存、未跟踪的文件列表。"
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self) -> ToolResult:
        exit_code, stdout, stderr = _run_git(["status", "--short"])
        if exit_code != 0:
            return ToolResult(
                success=False,
                content=f"git status 失败: {stderr}",
                error=stderr,
            )
        if not stdout.strip():
            return ToolResult(
                success=True,
                content="工作区干净，没有未提交的变更。",
                metadata={"changed_files": 0},
            )
        return ToolResult(
            success=True,
            content=f"Git 状态:\n{stdout}",
            metadata={"changed_files": len(stdout.strip().split("\n"))},
        )


class GitDiffTool(BaseTool):
    name = "git_diff"
    description = (
        "查看 Git 工作区或暂存区的变更差异。"
        "默认显示工作区未暂存的变更；添加 staged=true 查看已暂存的变更。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "staged": {
                "type": "boolean",
                "description": "是否查看已暂存(staged)的变更，默认 false（查看未暂存变更）"
            },
        },
        "required": [],
    }

    def execute(self, staged: bool = False) -> ToolResult:
        args = ["diff"]
        if staged:
            args.append("--staged")
        exit_code, stdout, stderr = _run_git(args)
        if exit_code != 0:
            return ToolResult(
                success=False,
                content=f"git diff 失败: {stderr}",
                error=stderr,
            )
        if not stdout.strip():
            label = "已暂存" if staged else "未暂存"
            return ToolResult(
                success=True,
                content=f"没有{label}的变更。",
            )
        return ToolResult(
            success=True,
            content=stdout,
            metadata={"lines": len(stdout.split("\n"))},
        )


class GitCommitTool(BaseTool):
    name = "git_commit"
    description = (
        "提交当前所有已暂存(staged)的变更。"
        "需要提供提交信息(-m message)。"
        "注意：此工具仅做提交，不会 push。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Git 提交信息（commit message）"
            },
        },
        "required": ["message"],
    }

    def _validate_params(self, kwargs: dict) -> tuple[bool, str]:
        msg = kwargs.get("message", "")
        if not msg or not msg.strip():
            return False, "commit message 不能为空"
        return True, ""

    def execute(self, message: str) -> ToolResult:
        # 1. 检查当前分支
        exit_code, branch_name, _ = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        if exit_code != 0:
            return ToolResult(success=False, content="无法获取当前分支", error=branch_name)
        branch_name = branch_name.strip()

        # 2. 安全校验：受保护分支
        valid, reason = self.safety.validate_git_branch(branch_name)
        if not valid:
            return ToolResult(
                success=False,
                content=f"禁止直接提交到 '{branch_name}': {reason}\n请创建新分支后提交。",
                error=reason,
            )

        # 3. 检查是否有暂存的文件
        exit_code, staged, _ = _run_git(["diff", "--staged", "--name-only"])
        if exit_code != 0:
            return ToolResult(success=False, content="检查暂存区失败", error=staged)
        if not staged.strip():
            return ToolResult(
                success=False,
                content="暂存区为空。请先用 git add 暂存文件后再提交。",
                error="nothing_to_commit",
            )

        # 4. 执行提交
        exit_code, stdout, stderr = _run_git(["commit", "-m", message])
        if exit_code != 0:
            return ToolResult(
                success=False,
                content=f"git commit 失败: {stderr}",
                error=stderr,
            )

        return ToolResult(
            success=True,
            content=stdout,
            metadata={"branch": branch_name, "staged_files": staged.strip().split("\n")},
        )
