"""
审查日志记录器 —— AuditLogger

设计思路：
- 所有审查事件（PASS/BLOCK/CONFIRM）都记录到结构化日志
- JSON Lines 格式，支持后续审计分析
- 关键信息：工具名、参数摘要、风险等级、各层判定、用户确认状态
- 支持告警回调：CRITICAL/BLOCK 事件可触发外部告警（Webhook/Slack等）
"""

import json
import time
from pathlib import Path
from typing import Optional, Callable

from src.security.types import AuditEvent, AuditDecision, RiskLevel


class AuditLogger:
    """
    安全审查日志记录器。

    职责：
    - 记录每次工具调用的审查事件
    - 支持文件持久化（JSON Lines）
    - CRITICAL/BLOCK 事件触发告警回调
    """

    def __init__(
        self,
        log_file: str = "data/audit.log",
        console_output: bool = True,
        alert_callback: Optional[Callable[[AuditEvent], None]] = None,
    ):
        self._log_path = Path(log_file)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._console = console_output
        self._alert_callback = alert_callback

    def log(self, event: AuditEvent):
        """记录一次审查事件"""
        record = event.to_log_dict()
        record["timestamp_iso"] = time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(event.timestamp)
        )

        # 写入文件
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass  # 日志写入失败不阻塞主流程

        # 控制台输出
        if self._console:
            self._console_log(event)

        # 告警回调
        if self._alert_callback and event.final_decision in (
            AuditDecision.BLOCK, AuditDecision.CONFIRM,
        ) and event.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            try:
                self._alert_callback(event)
            except Exception:
                pass

    def _console_log(self, event: AuditEvent):
        """控制台格式输出"""
        decision_color = {
            AuditDecision.PASS: "\033[32m",     # 绿色
            AuditDecision.CONFIRM: "\033[33m",   # 黄色
            AuditDecision.BLOCK: "\033[31m",     # 红色
        }
        reset = "\033[0m"
        color = decision_color.get(event.final_decision, "")

        icon = {
            AuditDecision.PASS: "[SAFE]",
            AuditDecision.CONFIRM: "[CONFIRM]",
            AuditDecision.BLOCK: "[BLOCK]",
        }.get(event.final_decision, "[???]")

        print(f"{color}{icon} {event.tool_name} | "
              f"risk={event.risk_level.value} | "
              f"decision={event.final_decision.value}{reset}")
