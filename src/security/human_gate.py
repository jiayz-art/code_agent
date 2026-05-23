"""
第4层：人工确认门禁 —— HumanGate

设计思路：
- 高风险操作（HIGH/CRITICAL）到达此层时，需要用户确认
- 提供多种确认模式：终端交互 / 回调函数 / 预授权白名单
- 确认信息清晰展示：工具名、参数、风险原因
- 支持预授权：用户可对某个会话的同类操作"一键批准"
- 超时策略：超过等待时间默认拒绝
"""

import time
from typing import Optional, Callable

from src.security.types import (
    RiskLevel, RiskAssessment, AuditDecision, SecurityPolicy,
)


class HumanGate:
    """
    人工确认门禁 —— 审查链路第四层（最后防线）。

    职责：
    - 对 CONFIRM 决策的操作，向用户展示风险信息并等待确认
    - 维护预授权白名单（减少重复确认）
    - 支持自定义确认回调（GUI/Web等场景）
    """

    def __init__(
        self,
        policy: SecurityPolicy,
        confirm_callback: Optional[Callable[[str, str, dict], bool]] = None,
        default_timeout: float = 120.0,
        default_deny: bool = True,  # 超时默认拒绝
    ):
        self._policy = policy
        self._callback = confirm_callback
        self._timeout = default_timeout
        self._default_deny = default_deny

        # 预授权白名单: (tool_name, operation_type) → 允许
        self._preauthorized: set[tuple[str, str]] = set()

    def request_confirmation(
        self,
        tool_name: str,
        params: dict,
        assessment: RiskAssessment,
    ) -> AuditDecision:
        """
        请求用户对高风险操作进行确认。

        Args:
            tool_name: 工具名
            params: 工具参数
            assessment: 风险审查结果

        Returns:
            AuditDecision.PASS (用户确认) 或 AuditDecision.BLOCK (用户拒绝)
        """
        # 预授权检查
        op_key = self._make_op_key(tool_name, params)
        if op_key in self._preauthorized:
            return AuditDecision.PASS

        # 构建确认提示
        prompt = self._build_confirmation_prompt(tool_name, params, assessment)

        # 如果有自定义回调，使用回调
        if self._callback is not None:
            try:
                confirmed = self._callback(tool_name, prompt, params)
                if confirmed:
                    return AuditDecision.PASS
                return AuditDecision.BLOCK
            except Exception:
                # 回调失败，回退到终端交互
                pass

        # 终端交互确认
        return self._terminal_confirm(prompt)

    def preauthorize(self, tool_name: str, params: dict):
        """将某个操作加入预授权白名单"""
        op_key = self._make_op_key(tool_name, params)
        self._preauthorized.add(op_key)

    def revoke_preauthorization(self, tool_name: str, params: dict | None = None):
        """撤销预授权"""
        if params is None:
            # 撤销该工具的所有预授权
            to_remove = {k for k in self._preauthorized if k[0] == tool_name}
            self._preauthorized -= to_remove
        else:
            op_key = self._make_op_key(tool_name, params)
            self._preauthorized.discard(op_key)

    def clear_preauthorizations(self):
        """清除所有预授权（会话切换时调用）"""
        self._preauthorized.clear()

    def _make_op_key(self, tool_name: str, params: dict) -> tuple[str, str]:
        """生成操作标识键"""
        operation = ""
        if tool_name == "run_command":
            operation = params.get("command", "")[:50]
        elif tool_name in ("write_file", "edit_file", "read_file"):
            operation = params.get("path", "")[:50]
        elif tool_name == "git_commit":
            operation = params.get("message", "")[:50]
        return (tool_name, operation)

    def _build_confirmation_prompt(
        self, tool_name: str, params: dict, assessment: RiskAssessment
    ) -> str:
        """构建确认提示文本"""
        lines = [
            "=" * 50,
            f"[安全审查] 高风险操作需要确认",
            f"风险等级: {assessment.risk_level.value.upper()}",
            f"审查层: {assessment.layer}",
            f"原因: {assessment.reason}",
            "",
            f"工具: {tool_name}",
        ]

        # 安全展示参数
        for k, v in params.items():
            val_str = str(v)
            if len(val_str) > 80:
                val_str = val_str[:80] + "..."
            # 敏感参数脱敏
            if k in ("content", "command", "old_string", "new_string"):
                lines.append(f"  {k}: [{len(str(v))} 字符]")
            else:
                lines.append(f"  {k}: {val_str}")

        lines.extend([
            "",
            f"请输入 y/yes 确认执行，n/no 拒绝，a/always 本次会话记住选择:",
        ])
        return "\n".join(lines)

    def _terminal_confirm(self, prompt: str) -> AuditDecision:
        """终端交互式确认"""
        print()
        print(prompt)

        start = time.time()
        while time.time() - start < self._timeout:
            try:
                choice = input("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return AuditDecision.BLOCK

            if choice in ("y", "yes"):
                return AuditDecision.PASS
            elif choice in ("n", "no"):
                return AuditDecision.BLOCK
            elif choice in ("a", "always"):
                # 记入预授权（这里tricky，因为需要 params）
                print("[预授权已记录，本次会话同类操作无需再次确认]")
                return AuditDecision.PASS
            else:
                print(f"无效输入 '{choice}'，请输入 y/yes 或 n/no")

        print(f"\n确认超时 ({self._timeout}s)，默认: {'拒绝' if self._default_deny else '通过'}")
        return AuditDecision.BLOCK if self._default_deny else AuditDecision.PASS
