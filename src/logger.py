"""
日志模块

设计思路：
- 三类日志分流：工具调用 / Token 消耗 / 通用错误
- JSON 格式输出，方便后续 grep/jq 分析或接入外部监控
- 基于标准库 logging，不引入第三方日志框架，降低依赖
- 控制台仅输出真正的系统级错误（CRITICAL），内部状态不泄漏给用户
- 所有工具调用、Token 统计、Info 日志仅写入文件
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Optional


# ============================================================
# 控制台过滤器 —— 阻止内部日志泄漏到用户终端
# ============================================================

class ConsoleFilter(logging.Filter):
    """
    控制台输出过滤器：只允许真正的系统级错误（CRITICAL）通过。
    工具调用失败、Token 统计、Info 日志等一律不显示在用户终端。
    """
    def filter(self, record: logging.LogRecord) -> bool:
        # 只放行 CRITICAL 级别（真正的系统故障）
        if record.levelno >= logging.CRITICAL:
            return True
        # 检查 extra_fields 中的 category
        if hasattr(record, "extra_fields"):
            category = record.extra_fields.get("category", "")
            if category in ("tool_call", "token_usage", "info"):
                return False
        # ERROR 级别也只记录到文件，不显示给用户（用户通过 Agent 响应了解结果）
        return False


# ============================================================
# 自定义 JSON 格式化器
# ============================================================

class JSONFormatter(logging.Formatter):
    """将日志记录格式化为单行 JSON，便于机器解析"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # 附加上下文字段（如有）
        if hasattr(record, "extra_fields") and record.extra_fields:
            log_entry.update(record.extra_fields)  # type: ignore
        # 异常信息
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """人类可读的文本格式"""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s [%(levelname)-7s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


# ============================================================
# Logger 工厂
# ============================================================

# 缓存已创建的 logger，避免重复创建 handler
_loggers: dict[str, logging.Logger] = {}


def get_logger(
    name: str,
    log_file: str = "agent.log",
    level: str = "INFO",
    fmt: str = "json",
    console: bool = True,
) -> logging.Logger:
    """
    获取或创建一个 logger 实例。

    每个 name 全局唯一，重复调用返回同一实例，避免日志重复输出。
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False  # 不向根 logger 传播，避免重复

    # 选择格式化器
    formatter: logging.Formatter
    if fmt == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()

    # 文件 handler —— 记录所有级别
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # 控制台 handler —— 仅显示 CRITICAL 级别，且过滤内部消息
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.DEBUG)  # 由 filter 决定放行哪些
        console_handler.addFilter(ConsoleFilter())
        logger.addHandler(console_handler)

    _loggers[name] = logger
    return logger


# ============================================================
# 便捷工具函数 —— 供 agent / tools 模块直接调用
# ============================================================

class AgentLogger:
    """
    Agent 专用日志封装，提供结构化记录方法。

    使用方式：
        agent_log = AgentLogger.from_config(cfg.logging)
        agent_log.log_tool_call("read_file", {"path": "a.py"}, duration=0.3, success=True)
        agent_log.log_tokens(prompt=500, completion=200)
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    @classmethod
    def from_config(cls, logging_cfg) -> "AgentLogger":
        lgr = get_logger(
            name="agent",
            log_file=logging_cfg.file,
            level=logging_cfg.level,
            fmt=logging_cfg.format,
            console=logging_cfg.console,
        )
        return cls(lgr)

    def log_tool_call(
        self,
        tool_name: str,
        params: dict,
        duration: float,
        success: bool,
        result_summary: str = "",
        error: Optional[str] = None,
    ):
        """记录一次工具调用"""
        extra = {
            "category": "tool_call",
            "tool": tool_name,
            "params": params,
            "duration_seconds": round(duration, 4),
            "success": success,
            "result_summary": result_summary[:200],  # 截断过长结果
        }
        if error:
            extra["error"] = error
        record = logging.LogRecord(
            name=self._logger.name,
            level=logging.INFO if success else logging.ERROR,
            pathname="",
            lineno=0,
            msg=f"tool_call: {tool_name} {'OK' if success else 'FAIL'} ({duration:.3f}s)",
            args=(),
            exc_info=None,
        )
        record.extra_fields = extra  # type: ignore
        self._logger.handle(record)

    def log_tokens(self, prompt: int, completion: int):
        """记录 Token 消耗"""
        extra = {
            "category": "token_usage",
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
        record = logging.LogRecord(
            name=self._logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=f"tokens: prompt={prompt}, completion={completion}",
            args=(),
            exc_info=None,
        )
        record.extra_fields = extra  # type: ignore
        self._logger.handle(record)

    def log_error(self, message: str, exc_info=None):
        """记录错误"""
        extra = {"category": "error"}
        record = logging.LogRecord(
            name=self._logger.name,
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=exc_info,
        )
        record.extra_fields = extra  # type: ignore
        self._logger.handle(record)

    def log_info(self, message: str):
        """记录通用信息"""
        extra = {"category": "info"}
        record = logging.LogRecord(
            name=self._logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None,
        )
        record.extra_fields = extra  # type: ignore
        self._logger.handle(record)

    def info(self, msg: str):
        """快捷 info 方法"""
        self.log_info(msg)

    def error(self, msg: str):
        """快捷 error 方法"""
        self.log_error(msg)
