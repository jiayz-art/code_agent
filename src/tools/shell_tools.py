"""
Shell 命令执行工具

设计思路：
- 通过 subprocess 执行 shell 命令，捕获 stdout + stderr
- 命令执行前经过 SafetyChecker 黑名单校验
- 超时保护：config 中设定的 timeout 秒后强制 kill
- 输出长度限制：stdout/stderr 各截断至 10000 字符，避免撑爆 LLM 上下文
- 返回 exit_code、stdout、stderr，方便 LLM 判断命令结果
"""

import subprocess
import platform

from src.tools.base import BaseTool, ToolResult
from src.safety import SafetyChecker


class RunCommandTool(BaseTool):
    name = "run_command"
    description = (
        "在系统 Shell 中执行命令并返回输出。"
        "命令在当前工作目录下执行。"
        "长时间运行的命令会在超时后自动终止。"
        "返回 exit_code（0=成功）、stdout 和 stderr。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 Shell 命令"
            },
        },
        "required": ["command"],
    }

    def execute(self, command: str) -> ToolResult:
        # 1. 安全校验：危险命令黑名单
        valid, reason = self.safety.validate_command(command)
        if not valid:
            return ToolResult(
                success=False,
                content=f"命令被安全策略拦截: {reason}",
                error=reason,
            )

        timeout = self.safety.get_shell_timeout()

        try:
            # 跨平台：Windows 用 cmd，其他用 bash
            if platform.system() == "Windows":
                shell_cmd = ["cmd", "/c", command]
            else:
                shell_cmd = ["bash", "-c", command]

            result = subprocess.run(
                shell_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                # 不传 shell=True，而是通过 cmd/bash -c 执行，
                # 避免 shell 注入风险（命令已由黑名单保护）
            )

            exit_code = result.returncode
            stdout = result.stdout
            stderr = result.stderr

            # 截断过长输出
            max_len = 10000
            stdout_truncated = len(stdout) > max_len
            stderr_truncated = len(stderr) > max_len
            stdout = stdout[:max_len]
            stderr = stderr[:max_len]

            # 构建返回内容
            parts = [f"$ {command}"]
            if stdout:
                suffix = "\n... (stdout 已截断)" if stdout_truncated else ""
                parts.append(f"[stdout]\n{stdout}{suffix}")
            if stderr:
                suffix = "\n... (stderr 已截断)" if stderr_truncated else ""
                parts.append(f"[stderr]\n{stderr}{suffix}")
            parts.append(f"[exit_code: {exit_code}]")

            success = exit_code == 0

            return ToolResult(
                success=success,
                content="\n".join(parts),
                metadata={
                    "exit_code": exit_code,
                    "stdout_length": len(result.stdout),
                    "stderr_length": len(result.stderr),
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                },
                error=f"命令返回非零退出码: {exit_code}" if not success else None,
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                content=f"命令超时 ({timeout}秒): {command}",
                error=f"TimeoutExpired: {timeout}s",
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                content=f"Shell 不可用或命令不存在: {command}",
                error="ShellNotFound",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content=f"命令执行异常: {e}",
                error=str(e),
            )
