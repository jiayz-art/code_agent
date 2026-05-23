"""
安全审查链路编排器 —— AuditChain

设计思路：
- 串联四层审查：RuleFilter → ToolSelfCheck → AIRiskClassifier → HumanGate
- 短路机制：BLOCK 立即终止，PASS 进入下一层
- 最终决策=∑各层决策中最严格的那个
- 统一入口 audit()，返回 (AuditDecision, AuditEvent)
- 集成到 ToolRegistry.execute() 中，对Agent透明
"""

import time
from typing import Optional

from src.llm import LLMClient
from src.security.types import (
    RiskLevel, RiskAssessment, AuditDecision, AuditEvent, SecurityPolicy,
    TOOL_RISK_LEVELS,
)
from src.security.rule_filter import RuleFilter
from src.security.tool_check import ToolSelfCheck
from src.security.ai_classifier import AIRiskClassifier
from src.security.human_gate import HumanGate


class AuditChain:
    """
    安全审查链路 —— 四层串联审查。

    使用方式：
        chain = AuditChain(policy)
        decision, event = chain.audit(tool_name, params, user_message)
        if decision == AuditDecision.BLOCK:
            raise/return blocked
        elif decision == AuditDecision.CONFIRM:
            # show prompt to user, wait for approval
    """

    def __init__(
        self,
        policy: SecurityPolicy,
        llm_client: Optional[LLMClient] = None,
        workspace_root: str = ".",
        enable_ai_classifier: bool = True,
        enable_human_gate: bool = True,
    ):
        self._policy = policy

        # 四层审查器
        self._rule_filter = RuleFilter(policy)
        self._tool_check = ToolSelfCheck(policy, workspace_root)
        self._ai_classifier = AIRiskClassifier(
            llm_client=llm_client,
            enabled=enable_ai_classifier,
        )
        self._human_gate = HumanGate(policy) if enable_human_gate else None

        self._session_id: str = ""
        self._current_task: str = ""

    def set_session(self, session_id: str, task: str = ""):
        self._session_id = session_id
        self._current_task = task

    def audit(
        self,
        tool_name: str,
        params: dict,
        user_message: str = "",
        auto_confirm: bool = False,  # 自动化测试/CI环境
    ) -> tuple[AuditDecision, AuditEvent]:
        """
        对一次工具调用执行完整审查链路。

        Args:
            tool_name: 工具名
            params: 工具参数
            user_message: 用户原始输入（用于AI检测）
            auto_confirm: True 则跳过人工确认（测试/CI模式）

        Returns:
            (最终决策, 审查事件记录)
        """
        assessments: list[RiskAssessment] = []
        workspace = str(self._tool_check._workspace)

        # ============================================================
        # 第1层：静态规则过滤
        # ============================================================
        rule_result = self._rule_filter.assess(tool_name, params, workspace)
        assessments.append(rule_result)

        if rule_result.decision == AuditDecision.BLOCK:
            return AuditDecision.BLOCK, self._build_event(
                tool_name, params, RiskLevel.CRITICAL,
                AuditDecision.BLOCK, assessments,
            )

        # ============================================================
        # 第2层：工具参数自检
        # ============================================================
        tool_result = self._tool_check.assess(tool_name, params)
        assessments.append(tool_result)

        if tool_result.decision == AuditDecision.BLOCK:
            return AuditDecision.BLOCK, self._build_event(
                tool_name, params, tool_result.risk_level,
                AuditDecision.BLOCK, assessments,
            )

        # 汇总前两层的风险等级（取最高）
        current_risk = max(
            rule_result.risk_level,
            tool_result.risk_level,
            key=lambda r: _RISK_ORDER[r],
        )

        # ============================================================
        # 第3层：AI风险分类（仅对MEDIUM+操作触发）
        # ============================================================
        ai_result = self._ai_classifier.assess(
            tool_name, params, user_message, current_risk,
        )
        assessments.append(ai_result)

        if ai_result.decision == AuditDecision.BLOCK:
            return AuditDecision.BLOCK, self._build_event(
                tool_name, params, ai_result.risk_level,
                AuditDecision.BLOCK, assessments,
            )

        # 更新风险等级
        current_risk = max(current_risk, ai_result.risk_level, key=lambda r: _RISK_ORDER[r])

        # ============================================================
        # 第4层：人工确认
        # ============================================================
        # 根据策略判定是否需要确认
        policy_decision = self._policy.get_decision_for(current_risk, tool_name)

        if policy_decision == AuditDecision.BLOCK:
            return AuditDecision.BLOCK, self._build_event(
                tool_name, params, current_risk,
                AuditDecision.BLOCK, assessments,
            )

        user_confirmed = False
        if policy_decision == AuditDecision.CONFIRM and self._human_gate is not None:
            if auto_confirm:
                # 自动化模式：默认通过
                user_confirmed = True
            else:
                # 找出最严重的那个风险评估作为展示
                worst = max(assessments, key=lambda a: _RISK_ORDER.get(a.risk_level, 0))
                gate_result = self._human_gate.request_confirmation(
                    tool_name, params, worst,
                )
                if gate_result == AuditDecision.BLOCK:
                    assessments.append(RiskAssessment(
                        risk_level=current_risk,
                        decision=AuditDecision.BLOCK,
                        reason="用户拒绝了此操作",
                        layer="human_gate",
                    ))
                    return AuditDecision.BLOCK, self._build_event(
                        tool_name, params, current_risk,
                        AuditDecision.BLOCK, assessments, user_confirmed=False,
                    )
                user_confirmed = True

        final_decision = AuditDecision.PASS
        if user_confirmed:
            assessments.append(RiskAssessment(
                risk_level=current_risk,
                decision=AuditDecision.PASS,
                reason="用户已确认",
                layer="human_gate",
            ))

        return final_decision, self._build_event(
            tool_name, params, current_risk,
            final_decision, assessments, user_confirmed,
        )

    def get_human_gate(self) -> Optional[HumanGate]:
        return self._human_gate

    def preauthorize(self, tool_name: str, params: dict):
        """预授权某个操作（减少重复确认）"""
        if self._human_gate:
            self._human_gate.preauthorize(tool_name, params)

    def _build_event(
        self,
        tool_name: str,
        params: dict,
        risk_level: RiskLevel,
        decision: AuditDecision,
        assessments: list[RiskAssessment],
        user_confirmed: bool = False,
    ) -> AuditEvent:
        return AuditEvent(
            tool_name=tool_name,
            params=params,
            risk_level=risk_level,
            final_decision=decision,
            assessments=assessments,
            user_confirmed=user_confirmed,
            timestamp=time.time(),
            session_id=self._session_id,
            task=self._current_task,
        )


# 风险等级排序权重
_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}
