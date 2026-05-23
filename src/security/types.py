"""
安全审查数据类型

设计思路：
- RiskLevel 四级风险：LOW/MEDIUM/HIGH/CRITICAL
- AuditDecision 三种决策：PASS/BLOCK/CONFIRM
- 每个审查层输出 RiskAssessment，AuditChain 汇总为最终 AuditDecision
- SecurityPolicy 可配置：不同项目不同安全等级
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RiskLevel(str, Enum):
    """操作风险等级"""
    LOW = "low"            # 只读操作、搜索、git status
    MEDIUM = "medium"      # 写文件、编辑文件、安全命令
    HIGH = "high"          # 删除文件、修改配置、git commit
    CRITICAL = "critical"  # 危险命令、.env 访问、系统文件操作


class AuditDecision(str, Enum):
    """审查决策"""
    PASS = "pass"          # 自动放行
    BLOCK = "block"        # 直接拒绝
    CONFIRM = "confirm"    # 需要用户确认


class SecurityTier(str, Enum):
    """安全等级 —— 不同项目可设不同等级"""
    RELAXED = "relaxed"    # 宽松：仅拦截 CRITICAL
    STANDARD = "standard"  # 标准：HIGH 需确认，CRITICAL 拦截
    STRICT = "strict"      # 严格：MEDIUM 也需确认，HIGH+ 拦截


@dataclass
class RiskAssessment:
    """单层审查的风险评估结果"""
    risk_level: RiskLevel
    decision: AuditDecision
    reason: str = ""
    layer: str = ""             # 产生此评估的审查层名称
    requires_confirmation: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class AuditEvent:
    """一次完整审查事件记录"""
    tool_name: str
    params: dict
    risk_level: RiskLevel
    final_decision: AuditDecision
    assessments: list[RiskAssessment] = field(default_factory=list)
    user_confirmed: bool = False
    timestamp: float = 0.0
    session_id: str = ""
    task: str = ""

    def to_log_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "params_summary": {k: str(v)[:100] for k, v in self.params.items()},
            "risk_level": self.risk_level.value,
            "final_decision": self.final_decision.value,
            "assessments": [
                {"layer": a.layer, "risk": a.risk_level.value,
                 "decision": a.decision.value, "reason": a.reason}
                for a in self.assessments
            ],
            "user_confirmed": self.user_confirmed,
            "session_id": self.session_id,
        }


@dataclass
class SecurityPolicy:
    """可配置的安全策略"""
    tier: SecurityTier = SecurityTier.STANDARD

    # 敏感文件模式（路径匹配）
    sensitive_file_patterns: list[str] = field(default_factory=lambda: [
        "*.env", "*.env.*", ".env.*",
        "config.yaml", "config.yml", "config.json",
        "settings.py", "settings.json",
        "*.pem", "*.key", "*.pfx", "*.p12",
        "credentials*", "secret*", "password*",
        "id_rsa*", "*.keystore",
    ])

    # 敏感目录模式
    sensitive_dir_patterns: list[str] = field(default_factory=lambda: [
        ".git/**", ".svn/**",
        "**/node_modules/**", "**/__pycache__/**",
        "**/.venv/**", "**/venv/**",
    ])

    # 需要确认的工具（按 RiskLevel 映射）
    confirm_tools: dict[str, RiskLevel] = field(default_factory=lambda: {
        "git_commit": RiskLevel.HIGH,
        "write_file": RiskLevel.MEDIUM,
        "edit_file": RiskLevel.MEDIUM,
        "run_command": RiskLevel.MEDIUM,
    })

    # 被拦截的工具（硬编码阻止列表，任何安全等级都生效）
    blocked_tools: set[str] = field(default_factory=set)

    # 根据安全等级获得各风险等级的决策
    def get_decision_for(self, risk_level: RiskLevel, tool_name: str = "") -> AuditDecision:
        """根据当前安全等级和风险等级返回决策"""
        # 硬阻止
        if tool_name in self.blocked_tools:
            return AuditDecision.BLOCK

        if self.tier == SecurityTier.RELAXED:
            if risk_level == RiskLevel.CRITICAL:
                return AuditDecision.BLOCK
            if risk_level == RiskLevel.HIGH:
                return AuditDecision.CONFIRM
            return AuditDecision.PASS

        elif self.tier == SecurityTier.STRICT:
            if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                return AuditDecision.BLOCK
            if risk_level == RiskLevel.MEDIUM:
                return AuditDecision.CONFIRM
            return AuditDecision.PASS

        else:  # STANDARD
            if risk_level == RiskLevel.CRITICAL:
                return AuditDecision.BLOCK
            if risk_level == RiskLevel.HIGH:
                return AuditDecision.CONFIRM
            return AuditDecision.PASS


# 工具 → 默认风险等级映射
TOOL_RISK_LEVELS: dict[str, RiskLevel] = {
    "read_file": RiskLevel.LOW,
    "grep_search": RiskLevel.LOW,
    "git_status": RiskLevel.LOW,
    "git_diff": RiskLevel.LOW,
    "write_file": RiskLevel.MEDIUM,
    "edit_file": RiskLevel.MEDIUM,
    "run_command": RiskLevel.MEDIUM,
    "git_commit": RiskLevel.HIGH,
}
