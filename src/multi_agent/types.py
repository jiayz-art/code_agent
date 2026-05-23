"""
多Agent协作数据类型

设计思路：
- 子Agent结果最小化：只返回摘要+文件变更列表，完整内容在worktree中
- 工具权限白名单：每个AgentType对应一组可调用工具
- 结果结构化：主Agent可校验，避免非结构化文本带来的上下文污染
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AgentType(str, Enum):
    """子Agent类型 —— 每种类型有不同的工具权限和System Prompt"""
    CODE_GENERATOR = "code_generator"   # 代码生成：可读写文件+搜索
    TESTER = "tester"                   # 测试：只读代码+执行命令
    REVIEWER = "reviewer"               # 审查：只读代码+搜索+Git
    DOCUMENTER = "documenter"           # 文档：只读代码+写文档


# 每种AgentType的工具白名单
AGENT_TOOL_WHITELIST: dict[AgentType, set[str]] = {
    AgentType.CODE_GENERATOR: {
        "read_file", "write_file", "edit_file", "grep_search", "run_command",
    },
    AgentType.TESTER: {
        "read_file", "run_command", "grep_search",
    },
    AgentType.REVIEWER: {
        "read_file", "grep_search", "git_status", "git_diff",
    },
    AgentType.DOCUMENTER: {
        "read_file", "write_file", "grep_search",
    },
}

# 每种AgentType的System Prompt（角色定义）
AGENT_SYSTEM_PROMPTS: dict[AgentType, str] = {
    AgentType.CODE_GENERATOR: """你是一个代码生成专家。根据需求描述生成高质量、可运行的代码。

## 规则
- 先读取相关文件理解项目结构和现有代码风格
- 生成的代码要符合项目现有风格和命名规范
- 包含必要的错误处理和类型注解
- 如果需求不明确，先提出澄清问题再编码
- 完成后列出所有创建/修改的文件

## 禁止
- 不修改与任务无关的文件
- 不执行破坏性命令""",

    AgentType.TESTER: """你是一个测试工程师。为给定代码编写和运行测试。

## 规则
- 先阅读待测代码理解其行为
- 编写覆盖核心逻辑、边界条件、异常路径的测试
- 运行测试并报告结果
- 如果测试失败且是代码问题，记录问题但不修改代码（那是code_generator的职责）

## 禁止
- 不修改业务代码（只能修改测试文件）
- 不提交代码""",

    AgentType.REVIEWER: """你是一个代码审查专家。审查代码质量、安全性和一致性。

## 规则
- 检查代码风格、命名规范、类型安全
- 检查潜在bug、安全漏洞、性能问题
- 对比已有代码确保风格一致
- 输出结构化的审查报告：问题列表 + 严重程度 + 修复建议

## 禁止
- 不修改代码（只提建议）
- 不执行Shell命令""",

    AgentType.DOCUMENTER: """你是一个技术文档专家。为代码生成清晰的文档。

## 规则
- 阅读代码理解API和行为
- 生成API文档、使用示例、README
- 确保文档和代码一致
- 使用中文编写文档

## 禁止
- 不修改业务代码（只修改文档文件）
- 不执行Shell命令""",
}


@dataclass
class SubAgentResult:
    """子Agent执行结果 —— 最小化信息传递，避免上下文污染"""
    success: bool
    agent_type: str                              # AgentType 名称
    task: str                                    # 执行的子任务描述
    summary: str                                 # 执行摘要（300字以内）
    files_created: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)  # {key: content_summary}
    error: Optional[str] = None
    iteration_count: int = 0
    tokens_used: int = 0

    def to_message(self) -> str:
        """转为供主Agent LLM阅读的结构化结果"""
        status = "成功" if self.success else "失败"
        lines = [
            f"[子Agent: {self.agent_type}] {status}",
            f"任务: {self.task}",
            f"摘要: {self.summary}",
        ]
        if self.files_created:
            lines.append(f"创建文件: {', '.join(self.files_created)}")
        if self.files_modified:
            lines.append(f"修改文件: {', '.join(self.files_modified)}")
        if self.artifacts:
            for k, v in self.artifacts.items():
                lines.append(f"{k}: {v}")
        if self.error:
            lines.append(f"错误: {self.error}")
        lines.append(f"统计: {self.iteration_count}轮迭代, {self.tokens_used} tokens")
        return "\n".join(lines)


@dataclass
class DelegateRequest:
    """主Agent向子Agent发起的委托请求"""
    agent_type: AgentType
    task: str                                    # 子任务描述
    context: str = ""                            # 附加上下文（相关代码片段、项目规范等）
    worktree: bool = False                       # 是否使用Git Worktree隔离
    max_iterations: int = 15                     # 子Agent最大迭代次数
