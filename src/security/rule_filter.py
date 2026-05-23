"""
第1层：规则过滤器 —— RuleFilter

设计思路：
- 纯静态规则，零 Token 开销，最先执行
- 命令黑名单 + 路径敏感文件模式 + 危险操作模式
- 匹配即拒绝（BLOCK），不匹配则放行（PASS）到下一层
- 规则从 SecurityPolicy 加载，支持不同项目定制
"""

import re
import os
from fnmatch import fnmatch
from pathlib import Path

from src.security.types import (
    RiskLevel, RiskAssessment, AuditDecision, SecurityPolicy,
)
from src.safety import DANGEROUS_PATTERNS, COMMAND_BLACKLIST


class RuleFilter:
    """
    静态规则过滤器 —— 审查链路第一层。

    执行顺序：
    1. 危险命令模式匹配
    2. 敏感文件路径匹配
    3. 敏感目录操作检测
    4. 工具直接阻止列表检查
    """

    def __init__(self, policy: SecurityPolicy):
        self._policy = policy

        # 编译敏感文件正则
        self._sensitive_file_res: list[re.Pattern] = []
        for pattern in policy.sensitive_file_patterns:
            # 将 glob 模式转为正则
            regex = re.escape(pattern)
            regex = regex.replace(r"\*", ".*").replace(r"\?", ".")
            self._sensitive_file_res.append(re.compile(regex, re.IGNORECASE))

        # 编译敏感目录正则（glob 风格 → 正则）
        self._sensitive_dir_res: list[re.Pattern] = []
        for pattern in policy.sensitive_dir_patterns:
            # 将 glob ** 转换为正则：**/ → 可选前缀，/** → 可选后缀
            regex = re.escape(pattern)
            regex = regex.replace(r"\*\*/", "(^|.*/)")   # **/ 匹配开头或任意前缀
            regex = regex.replace(r"/\*\*", "(/.*|$)")    # /** 匹配后缀或结尾
            regex = regex.replace(r"\*\*", ".*")          # 中间的 ** 匹配任意字符
            regex = regex.replace(r"\*", "[^/]*")         # 单 * 匹配非 / 字符
            regex = "^" + regex + "$"
            self._sensitive_dir_res.append(re.compile(regex, re.IGNORECASE))

        # 危险命令模式（引用 safety.py 的共享定义，消除重复）
        self._dangerous_patterns: list[re.Pattern] = DANGEROUS_PATTERNS

        # 命令黑名单关键词（引用 safety.py 的共享定义）
        self._command_blacklist_keywords = COMMAND_BLACKLIST

    def assess(self, tool_name: str, params: dict, workspace_root: str = ".") -> RiskAssessment:
        """
        对一次工具调用执行规则过滤。

        Returns:
            RiskAssessment with BLOCK if matched, PASS otherwise
        """
        # 1. 工具直接阻止列表
        if tool_name in self._policy.blocked_tools:
            return RiskAssessment(
                risk_level=RiskLevel.CRITICAL,
                decision=AuditDecision.BLOCK,
                reason=f"工具 '{tool_name}' 在阻止列表中",
                layer="rule_filter",
            )

        # 2. 命令检查
        if tool_name == "run_command":
            assessment = self._check_command(params)
            if assessment.decision != AuditDecision.PASS:
                return assessment

        # 3. 文件路径检查
        if tool_name in ("read_file", "write_file", "edit_file"):
            assessment = self._check_file_path(params, workspace_root)
            if assessment.decision != AuditDecision.PASS:
                return assessment

        # 4. 检查文件内容中是否包含敏感操作（仅对 write_file）
        if tool_name == "write_file":
            assessment = self._check_file_content(params)
            if assessment.decision != AuditDecision.PASS:
                return assessment

        return RiskAssessment(
            risk_level=RiskLevel.LOW,
            decision=AuditDecision.PASS,
            reason="规则过滤通过",
            layer="rule_filter",
        )

    def _check_command(self, params: dict) -> RiskAssessment:
        """检查Shell命令"""
        command = params.get("command", "")
        if not command:
            return RiskAssessment(RiskLevel.LOW, AuditDecision.PASS, layer="rule_filter")

        # 精确黑名单关键词
        cmd_lower = command.lower().strip()
        for keyword in self._command_blacklist_keywords:
            if keyword in cmd_lower:
                return RiskAssessment(
                    risk_level=RiskLevel.CRITICAL,
                    decision=AuditDecision.BLOCK,
                    reason=f"命令包含危险操作: '{keyword}'",
                    layer="rule_filter",
                    metadata={"matched_keyword": keyword},
                )

        # 危险命令正则
        for pattern in self._dangerous_patterns:
            if pattern.search(command):
                return RiskAssessment(
                    risk_level=RiskLevel.CRITICAL,
                    decision=AuditDecision.BLOCK,
                    reason=f"命令匹配危险模式: '{pattern.pattern}'",
                    layer="rule_filter",
                    metadata={"matched_pattern": pattern.pattern},
                )

        return RiskAssessment(
            risk_level=RiskLevel.MEDIUM,
            decision=AuditDecision.PASS,
            reason="命令通过黑名单检查",
            layer="rule_filter",
        )

    def _check_file_path(self, params: dict, workspace_root: str) -> RiskAssessment:
        """检查文件路径"""
        path = params.get("path", "") or params.get("file_path", "")
        if not path:
            return RiskAssessment(RiskLevel.LOW, AuditDecision.PASS, layer="rule_filter")

        # 敏感文件模式
        filename = Path(path).name
        for pattern in self._policy.sensitive_file_patterns:
            if fnmatch(filename, pattern) or fnmatch(path, pattern):
                return RiskAssessment(
                    risk_level=RiskLevel.CRITICAL,
                    decision=AuditDecision.BLOCK,
                    reason=f"操作敏感文件被阻止: '{path}' (匹配模式 '{pattern}')",
                    layer="rule_filter",
                    metadata={"path": path, "matched_pattern": pattern},
                )

        # 敏感目录模式
        for pattern in self._sensitive_dir_res:
            if pattern.search(path):
                return RiskAssessment(
                    risk_level=RiskLevel.HIGH,
                    decision=AuditDecision.BLOCK,
                    reason=f"操作敏感目录被阻止: '{path}' (匹配模式 '{pattern.pattern}')",
                    layer="rule_filter",
                    metadata={"path": path, "matched_pattern": pattern.pattern},
                )

        # 路径越界检测（../ 逃逸）
        normalized = os.path.normpath(path)
        if normalized.startswith("..") or os.path.isabs(normalized):
            # 绝对路径或相对于工作区外的路径
            abs_path = str(Path(workspace_root).resolve())
            resolved = str((Path(workspace_root) / path).resolve())
            if not resolved.startswith(abs_path):
                return RiskAssessment(
                    risk_level=RiskLevel.CRITICAL,
                    decision=AuditDecision.BLOCK,
                    reason=(
                        f"路径越界: '{path}' 不在工作区 '{abs_path}' 内。"
                        f"请使用工作区内的相对路径，例如将文件写入工作区目录下。"
                    ),
                    layer="rule_filter",
                )

        return RiskAssessment(
            risk_level=RiskLevel.LOW,
            decision=AuditDecision.PASS,
            reason="文件路径通过检查",
            layer="rule_filter",
        )

    def _check_file_content(self, params: dict) -> RiskAssessment:
        """检查要写入的文件内容（防止写入恶意内容）"""
        content = params.get("content", "")
        if not content:
            return RiskAssessment(RiskLevel.LOW, AuditDecision.PASS, layer="rule_filter")

        # 检测是否尝试写入包含后门/反弹shell的内容
        suspicious_patterns = [
            (r"import\s+socket.*\.connect\s*\(", "可能包含反弹Shell代码"),
            (r"subprocess\.(call|Popen|run)\s*\(.*shell\s*=\s*True", "危险subprocess调用"),
            (r"os\.(system|popen|exec[lv]*)", "危险OS命令执行"),
            (r"eval\s*\(|exec\s*\(|compile\s*\(", "动态代码执行"),
            (r"__import__\s*\(|importlib\.import_module", "动态模块导入"),
            (r"base64\.(b64|standard_b64)decode", "可疑编码解码"),
            (r"lambda\s*.*__builtins__", "权限提升尝试"),
        ]

        for pattern, description in suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                return RiskAssessment(
                    risk_level=RiskLevel.HIGH,
                    decision=AuditDecision.CONFIRM,
                    reason=f"写入内容{description}",
                    layer="rule_filter",
                    metadata={"matched_pattern": pattern},
                )

        return RiskAssessment(
            risk_level=RiskLevel.LOW,
            decision=AuditDecision.PASS,
            layer="rule_filter",
        )
