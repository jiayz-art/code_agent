"""
第2层：工具自检 —— ToolSelfCheck

设计思路：
- 工具执行前的参数级校验
- 检查：路径范围、文件大小、参数合法性、操作影响范围
- 返回 RiskAssessment 而非直接拒绝，给上层决策空间
- 每种工具类型有专门的检查逻辑
"""

import os
from pathlib import Path

from src.security.types import (
    RiskLevel, RiskAssessment, AuditDecision, SecurityPolicy,
)


class ToolSelfCheck:
    """
    工具参数自检 —— 审查链路第二层。

    对每种工具的调用参数进行深度检查，评估风险等级。
    """

    def __init__(self, policy: SecurityPolicy, workspace_root: str = "."):
        self._policy = policy
        self._workspace = Path(workspace_root).resolve()

    def assess(self, tool_name: str, params: dict) -> RiskAssessment:
        """对一次工具调用执行自检"""
        handler = getattr(self, f"_check_{tool_name}", None)
        if handler:
            return handler(params)
        return RiskAssessment(
            risk_level=RiskLevel.MEDIUM,
            decision=AuditDecision.PASS,
            reason=f"工具 '{tool_name}' 无专用自检规则，默认放行",
            layer="tool_check",
        )

    def _check_read_file(self, params: dict) -> RiskAssessment:
        path = params.get("path", "")
        # 检查是否读取敏感文件
        file_name = Path(path).name
        sensitive_names = (
            ".env", ".env.local", ".env.production",
            "credentials.json", "secrets.yaml", "private.key",
            "id_rsa", "id_ed25519",
        )
        if file_name in sensitive_names:
            return RiskAssessment(
                risk_level=RiskLevel.HIGH,
                decision=AuditDecision.CONFIRM,
                reason=f"读取敏感文件: '{path}'",
                layer="tool_check",
                metadata={"path": path, "file": file_name},
            )

        return RiskAssessment(
            risk_level=RiskLevel.LOW,
            decision=AuditDecision.PASS,
            reason="读取操作，低风险",
            layer="tool_check",
        )

    def _check_write_file(self, params: dict) -> RiskAssessment:
        path = params.get("path", "")
        content = params.get("content", "")
        resolved = self._resolve_path(path)

        risks = []
        risk_level = RiskLevel.MEDIUM

        # 检查是否覆盖已存在的文件
        if resolved and resolved.exists():
            size = resolved.stat().st_size
            risks.append(f"将覆盖已有文件 '{path}' ({size} 字节)")
            if size > 10000:
                risk_level = RiskLevel.HIGH
                risks.append("文件较大，覆盖可能造成数据丢失")

        # 检查是否操作配置文件
        file_name = Path(path).name
        if file_name in ("config.yaml", "config.yml", "config.json",
                          "settings.py", "pyproject.toml", "package.json",
                          ".gitignore", "Dockerfile", "docker-compose.yml",
                          "Makefile", "CMakeLists.txt"):
            risk_level = RiskLevel.HIGH
            risks.append(f"操作项目配置文件: '{path}'")

        # 检查是否写入工作区外的路径
        if resolved and not str(resolved).startswith(str(self._workspace)):
            return RiskAssessment(
                risk_level=RiskLevel.CRITICAL,
                decision=AuditDecision.BLOCK,
                reason=(
                    f"文件路径在工作区外: '{path}'。"
                    f"当前工作区为 '{self._workspace}'，"
                    f"请将文件写入工作区内的相对路径。"
                ),
                layer="tool_check",
            )

        # 检查内容大小
        if len(content) > 50000:
            risk_level = RiskLevel.HIGH
            risks.append(f"写入大量内容 ({len(content)} 字符)")

        if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            return RiskAssessment(
                risk_level=risk_level,
                decision=AuditDecision.CONFIRM,
                reason="; ".join(risks) if risks else "高风险写操作",
                layer="tool_check",
                metadata={"path": path, "risks": risks},
            )

        return RiskAssessment(
            risk_level=risk_level,
            decision=AuditDecision.PASS,
            reason="写文件操作，中等风险" if risks else "写文件操作",
            layer="tool_check",
        )

    def _check_edit_file(self, params: dict) -> RiskAssessment:
        path = params.get("path", "")
        old_str = params.get("old_string", "")
        new_str = params.get("new_string", "")

        risk_level = RiskLevel.MEDIUM
        risks = []

        # 检查是否编辑配置文件
        file_name = Path(path).name
        if file_name in ("config.yaml", "config.yml", "config.json",
                          ".env", "settings.py", "pyproject.toml",
                          "Dockerfile", "docker-compose.yml"):
            return RiskAssessment(
                risk_level=RiskLevel.HIGH,
                decision=AuditDecision.CONFIRM,
                reason=f"编辑项目配置文件: '{path}'",
                layer="tool_check",
            )

        # 删除多于添加 → 可能破坏代码
        if len(old_str) > len(new_str) * 3 and len(old_str) > 100:
            risk_level = RiskLevel.HIGH
            risks.append(f"将删除大量代码 ({len(old_str)} → {len(new_str)} 字符)")

        # 检查是否修改导入语句（影响面大）
        if "import " in old_str or "from " in old_str:
            risks.append("修改导入语句可能影响其他模块")
            risk_level = RiskLevel.HIGH

        if risk_level == RiskLevel.HIGH:
            return RiskAssessment(
                risk_level=RiskLevel.HIGH,
                decision=AuditDecision.CONFIRM,
                reason="; ".join(risks) if risks else "高风险编辑",
                layer="tool_check",
            )

        return RiskAssessment(
            risk_level=RiskLevel.MEDIUM,
            decision=AuditDecision.PASS,
            reason="; ".join(risks) if risks else "编辑文件",
            layer="tool_check",
        )

    def _check_run_command(self, params: dict) -> RiskAssessment:
        command = params.get("command", "")
        cmd_lower = command.lower().strip()
        risks = []

        # 检查管道操作（可能链式执行危险命令）
        if "|" in command:
            risks.append("命令包含管道操作")

        # 检查重定向
        if ">" in command or ">>" in command:
            risks.append("命令包含输出重定向")

        # 检查后台执行
        if command.rstrip().endswith("&"):
            risks.append("命令将后台执行")

        # 检查网络请求
        if any(kw in cmd_lower for kw in ("curl ", "wget ", "nc ", "netcat ")):
            risks.append("命令包含网络请求")

        # 检查安装/卸载操作
        if any(kw in cmd_lower for kw in ("pip install", "npm install -g",
                                            "apt-get", "yum ", "brew ",
                                            "pip uninstall", "npm uninstall")):
            risks.append("命令可能安装/卸载系统级包")

        # 检查文件删除
        if any(kw in cmd_lower for kw in ("rm ", "del ", "rmdir ", "rd ")):
            risks.append("命令包含文件删除操作")
            # 检查是否删除重要文件
            if any(kw in cmd_lower for kw in ("-rf", "/s", "/q")):
                return RiskAssessment(
                    risk_level=RiskLevel.CRITICAL,
                    decision=AuditDecision.BLOCK,
                    reason=f"危险文件删除命令",
                    layer="tool_check",
                )

        if risks:
            return RiskAssessment(
                risk_level=RiskLevel.HIGH,
                decision=AuditDecision.CONFIRM,
                reason="; ".join(risks),
                layer="tool_check",
            )

        return RiskAssessment(
            risk_level=RiskLevel.MEDIUM,
            decision=AuditDecision.PASS,
            reason="命令通过自检",
            layer="tool_check",
        )

    def _check_git_commit(self, params: dict) -> RiskAssessment:
        message = params.get("message", "")

        # 检查是否有提交信息
        if not message or len(message.strip()) < 3:
            return RiskAssessment(
                risk_level=RiskLevel.HIGH,
                decision=AuditDecision.CONFIRM,
                reason="Git提交信息过短或为空",
                layer="tool_check",
            )

        return RiskAssessment(
            risk_level=RiskLevel.HIGH,
            decision=AuditDecision.CONFIRM,
            reason=f"Git 提交: '{message[:60]}'",
            layer="tool_check",
        )

    def _check_grep_search(self, params: dict) -> RiskAssessment:
        # 搜索操作，始终低风险
        return RiskAssessment(
            risk_level=RiskLevel.LOW,
            decision=AuditDecision.PASS,
            reason="搜索操作，低风险",
            layer="tool_check",
        )

    def _check_git_status(self, params: dict) -> RiskAssessment:
        return RiskAssessment(
            risk_level=RiskLevel.LOW,
            decision=AuditDecision.PASS,
            reason="Git 状态查询，低风险",
            layer="tool_check",
        )

    def _check_git_diff(self, params: dict) -> RiskAssessment:
        return RiskAssessment(
            risk_level=RiskLevel.LOW,
            decision=AuditDecision.PASS,
            reason="Git 差异查看，低风险",
            layer="tool_check",
        )

    def _resolve_path(self, path_str: str) -> Path | None:
        try:
            resolved = (self._workspace / path_str).resolve()
            return resolved
        except (OSError, ValueError):
            return None
