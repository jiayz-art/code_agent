"""
工具抽象基类与注册中心

设计思路：
- BaseTool 定义统一接口：name / description / parameters(JSON Schema) / execute()
- ToolResult 统一返回结构，无论成功失败都返回此对象，不再抛异常到上层
- ToolRegistry 是工具注册中心：
  - 注册工具时校验唯一性
  - get_schemas() 生成 OpenAI 兼容的 tools 数组
  - execute() 统一入口，包装异常为 ToolResult
  - 集成安全审查链路（AuditChain），工具执行前自动审查
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Optional

from src.safety import SafetyChecker


# ============================================================
# 统一返回结构
# ============================================================

@dataclass
class ToolResult:
    """
    工具执行结果 —— 所有工具统一返回此结构。

    content 会被追加到 LLM 对话中，所以需要清晰描述发生了什么。
    LLM 会阅读 content 来决定下一步操作，因此要提供足够的上下文。
    """
    success: bool                                  # 是否成功
    content: str                                    # 结果内容（供 LLM 推理）
    error: Optional[str] = None                     # 错误详情
    metadata: dict[str, Any] = field(default_factory=dict)  # 附加元信息

    def to_message(self) -> str:
        """转为发给 LLM 的工具结果消息。

        失败时提供明确的状态标记和操作建议，帮助 LLM 诚实汇报结果。
        """
        if self.success:
            return f"[工具执行成功]\n{self.content}"
        else:
            error_type = self.error or "未知错误"
            # 安全拦截特殊处理：明确告知 LLM 操作被阻止，不能声称成功
            if "SECURITY_BLOCK" in error_type:
                return (
                    f"[工具执行被安全策略阻止]\n"
                    f"原因: {self.content}\n"
                    f"重要: 此操作未执行，请勿声称已完成。请告知用户文件未创建，并建议使用工作区内的路径。"
                )
            return (
                f"[工具执行失败]\n"
                f"错误类型: {error_type}\n"
                f"详情: {self.content}\n"
                f"重要: 此操作未成功。请根据错误信息调整操作，或向用户如实汇报失败原因。"
            )


# ============================================================
# 工具抽象基类
# ============================================================

class BaseTool(ABC):
    """
    所有工具的抽象基类。

    子类必须提供：
    - name: 工具名（对应 API function.name）
    - description: 用途说明（LLM 通过这个判断何时调用）
    - parameters: JSON Schema 格式的参数定义
    - execute(**kwargs): 具体实现

    可选覆盖：
    - _validate_params(kwargs): 参数校验逻辑（默认不做额外校验）
    """

    # 子类必须定义的类属性
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}

    def __init__(self, safety: SafetyChecker):
        """
        每个工具都持有 SafetyChecker 引用，
        在执行前完成安全校验。
        """
        self.safety = safety

    def get_schema(self) -> dict[str, Any]:
        """生成 OpenAI 兼容的工具定义"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def run(self, **kwargs) -> ToolResult:
        """
        工具执行的统一入口，包装异常处理。

        子类不应覆盖此方法，而应覆盖 execute()。
        此方法确保所有异常都被捕获并转为 ToolResult。
        """
        try:
            # 参数校验
            valid, err_msg = self._validate_params(kwargs)
            if not valid:
                return ToolResult(
                    success=False,
                    content=f"参数校验失败: {err_msg}",
                    error=err_msg,
                )
            # 执行
            return self.execute(**kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                content=f"工具内部异常: {str(e)}",
                error=str(e),
            )

    def _validate_params(self, kwargs: dict) -> tuple[bool, str]:
        """参数校验钩子，子类可选覆盖。返回 (是否通过, 错误信息)"""
        return True, ""

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """子类实现具体逻辑。kwargs 已通过参数校验。"""
        ...


# ============================================================
# 工具注册中心
# ============================================================

class ToolRegistry:
    """
    工具注册中心 —— Agent 通过它发现和调用工具。

    职责：
    - 注册/注销工具
    - 生成 OpenAI tools 数组（给 LLM API）
    - 按名称执行工具（给 Agent 调用）
    - 集成安全审查链路（AuditChain），工具执行前自动审查
    """

    def __init__(self, safety: SafetyChecker, audit_chain=None):
        self._safety = safety
        self._audit_chain = audit_chain  # 可选：AuditChain 实例
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool_class: type[BaseTool]):
        """
        注册一个工具类。实例化时自动注入 SafetyChecker。

        tool_class 必须是 BaseTool 的子类，且 name 不能重复。
        """
        instance = tool_class(self._safety)
        if instance.name in self._tools:
            raise ValueError(f"工具名冲突: '{instance.name}' 已注册")
        if not instance.name:
            raise ValueError(f"工具类 {tool_class.__name__} 未定义 name")
        self._tools[instance.name] = instance

    def register_instance(self, tool: "BaseTool"):
        """直接注册已构造的工具实例（用于需要额外构造参数的工具）"""
        if tool.name in self._tools:
            raise ValueError(f"工具名冲突: '{tool.name}' 已注册")
        if not tool.name:
            raise ValueError("工具实例未定义 name")
        self._tools[tool.name] = tool

    def get_schemas(self) -> list[dict[str, Any]]:
        """返回所有已注册工具的 OpenAI 格式定义，供 LLM API 调用"""
        return [tool.get_schema() for tool in self._tools.values()]

    def get_schemas_for(self, tool_names: set[str]) -> list[dict[str, Any]]:
        """只返回指定工具名的 Schema（供路由系统裁剪后使用）"""
        return [
            tool.get_schema()
            for name, tool in self._tools.items()
            if name in tool_names
        ]

    def get_by_names(self, tool_names: set[str]) -> dict[str, "BaseTool"]:
        """按名称批量获取工具实例"""
        return {name: self._tools[name] for name in tool_names if name in self._tools}

    def execute(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        """
        根据工具名执行工具。

        params 是 LLM 返回的 arguments（已由 LLM SDK 解析为 dict）。
        执行前经过安全审查链路（如果启用）。
        如果工具未注册，返回失败的 ToolResult 而非抛异常。
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                content=f"未知工具: '{tool_name}'",
                error=f"未注册的工具: {tool_name}",
            )

        # ★ 安全审查：执行前通过 AuditChain 审查
        if self._audit_chain is not None:
            from src.security.types import AuditDecision
            decision, event = self._audit_chain.audit(tool_name, params)
            # 记录审查日志（由Agent层的logger统一处理，这里先跳过）
            if decision == AuditDecision.BLOCK:
                return ToolResult(
                    success=False,
                    content=f"[安全审查拦截] 操作被阻止:\n{event.assessments[-1].reason if event.assessments else '安全策略拒绝'}",
                    error=f"SECURITY_BLOCK: {event.risk_level.value}",
                    metadata={"audit_event": event.to_log_dict()},
                )

        return tool.run(**params)

    def list_tools(self) -> list[str]:
        """列出所有已注册的工具名"""
        return list(self._tools.keys())

    def count(self) -> int:
        """已注册工具数量"""
        return len(self._tools)

    @property
    def audit_chain(self):
        return self._audit_chain
