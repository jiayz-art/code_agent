"""
记忆数据模型

设计思路：
- 三级记忆体系：用户画像(跨项目) → 程序性经验(如何做) → 情景记忆(发生过什么)
- MemoryEntry = 程序性经验，EpisodicMemory = 情景记忆，UserProfile = 用户画像
- 分类维度：project → task_type → error_type → difficulty
- to_context_text() 控制注入 Prompt 的长度，配合 Token 预算机制
- 所有字段可序列化为 JSON，写入文件系统和 ChromaDB
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


# ============================================================
# 用户画像 —— 跨项目持久化
# ============================================================

@dataclass
class UserProfile:
    """用户画像 —— 跨项目持久化，记录用户偏好和习惯"""

    # 沟通偏好
    language: str = "zh"                             # 首选语言
    verbosity: str = "concise"                       # concise / detailed
    confirm_before_action: bool = True               # 高风险操作前确认

    # 技术偏好
    tech_stack: list[str] = field(default_factory=list)   # ["python", "fastapi"]
    preferred_test_framework: str = "pytest"
    preferred_doc_style: str = "google"              # google / numpy / sphinx

    # 编码风格
    coding_style: dict = field(default_factory=lambda: {
        "comments": "minimal",                       # minimal / moderate / verbose
        "type_hints": "strict",                      # strict / moderate / none
        "error_handling": "explicit",                # explicit / minimal
        "naming": "descriptive",                     # descriptive / concise
    })

    # 常见上下文
    common_projects: list[str] = field(default_factory=list)
    common_patterns: list[dict] = field(default_factory=list)  # [{"name": "CRUD", "template": "..."}]
    known_constraints: list[str] = field(default_factory=list)  # ["不使用 ORM", "SQL 必须大写"]

    # 元信息
    update_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_interactions: int = 0
    total_tasks_completed: int = 0

    def to_context_text(self) -> str:
        """生成可注入 Prompt 的用户画像摘要"""
        lines = ["## 用户画像（从历史交互中学习）"]
        lines.append(f"- 语言偏好: {self.language}")
        lines.append(f"- 技术栈: {', '.join(self.tech_stack[:8])}" if self.tech_stack else "- 技术栈: 待学习")
        style = self.coding_style
        lines.append(f"- 编码风格: 注释={style.get('comments','?')}, 类型={style.get('type_hints','?')}, 命名={style.get('naming','?')}")
        if self.known_constraints:
            lines.append(f"- 约束条件: {'; '.join(self.known_constraints[:5])}")
        if self.common_patterns:
            names = [p.get("name", "?") for p in self.common_patterns[:5]]
            lines.append(f"- 常用模式: {', '.join(names)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "verbosity": self.verbosity,
            "confirm_before_action": self.confirm_before_action,
            "tech_stack": self.tech_stack,
            "preferred_test_framework": self.preferred_test_framework,
            "preferred_doc_style": self.preferred_doc_style,
            "coding_style": self.coding_style,
            "common_projects": self.common_projects,
            "common_patterns": self.common_patterns,
            "known_constraints": self.known_constraints,
            "update_timestamp": self.update_timestamp,
            "total_interactions": self.total_interactions,
            "total_tasks_completed": self.total_tasks_completed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        return cls(
            language=data.get("language", "zh"),
            verbosity=data.get("verbosity", "concise"),
            confirm_before_action=data.get("confirm_before_action", True),
            tech_stack=data.get("tech_stack", []),
            preferred_test_framework=data.get("preferred_test_framework", "pytest"),
            preferred_doc_style=data.get("preferred_doc_style", "google"),
            coding_style=data.get("coding_style", {}),
            common_projects=data.get("common_projects", []),
            common_patterns=data.get("common_patterns", []),
            known_constraints=data.get("known_constraints", []),
            update_timestamp=data.get("update_timestamp", ""),
            total_interactions=data.get("total_interactions", 0),
            total_tasks_completed=data.get("total_tasks_completed", 0),
        )


# ============================================================
# 程序性经验记忆
# ============================================================

@dataclass
class MemoryEntry:
    """
    程序性经验记忆 —— "如何做"。

    一条记忆 = 一次任务执行的完整经验提炼。
    包含：做了什么、遇到什么问题、怎么解决的、能复用的是什么。
    """

    # ---- 唯一标识 ----
    id: str                                         # 唯一 ID，格式: mem_{project}_{序号}

    # ---- 分类维度 ----
    project: str                                    # 所属项目名
    task: str                                       # 原始任务描述
    task_type: str = "general"                      # bug-fix / feature / refactor / debug / test / document / git
    error_type: str = ""                            # 错误类型（如有）
    difficulty: str = "medium"                      # easy / medium / hard
    preconditions: list[str] = field(default_factory=list)  # 前置条件
    applicable_scenarios: list[str] = field(default_factory=list)  # 适用场景边界

    # ---- 反思内容 ----
    summary: str = ""                               # 一句话摘要（同步生成，<100 字）
    detail: str = ""                                # 详细经验（异步生成，结构化）
    tags: list[str] = field(default_factory=list)   # 自动提取的标签

    # ---- 执行统计 ----
    tool_calls_count: int = 0                       # 工具调用次数
    success: bool = True                            # 任务是否成功
    related_files: list[str] = field(default_factory=list)  # 涉及的文件

    # ---- 质量指标 ----
    reuse_count: int = 0                            # 被检索复用的次数
    helpful_score: float = 0.0                      # 有效性评分 (0-5)
    last_reused: str = ""                           # 上次被复用的时间

    # ---- 元信息 ----
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    embedding_id: str = ""                          # ChromaDB 中的 embedding ID

    def to_context_text(self, max_chars: int = 300) -> str:
        """
        转为注入 Prompt 的上下文文本。

        Token 预算控制在此处精确执行：截断 detail 到 max_chars 字符。
        """
        detail_preview = self.detail[:max_chars] if self.detail else self.summary
        if len(self.detail) > max_chars:
            detail_preview += "..."

        status = "成功" if self.success else "失败"
        lines = [
            f"[历史经验] 任务: {self.task}",
            f"类型: {self.task_type} | 难度: {self.difficulty} | 结果: {status} | 复用: {self.reuse_count}次",
        ]
        if self.error_type:
            lines.append(f"错误类型: {self.error_type}")
        if self.related_files:
            lines.append(f"涉及文件: {', '.join(self.related_files[:5])}")
        if self.applicable_scenarios:
            lines.append(f"适用场景: {', '.join(self.applicable_scenarios[:3])}")
        if detail_preview:
            lines.append(f"经验: {detail_preview}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """序列化为字典（用于 JSON 持久化）"""
        return {
            "id": self.id,
            "project": self.project,
            "task": self.task,
            "task_type": self.task_type,
            "error_type": self.error_type,
            "difficulty": self.difficulty,
            "preconditions": self.preconditions,
            "applicable_scenarios": self.applicable_scenarios,
            "summary": self.summary,
            "detail": self.detail,
            "tags": self.tags,
            "tool_calls_count": self.tool_calls_count,
            "success": self.success,
            "related_files": self.related_files,
            "reuse_count": self.reuse_count,
            "helpful_score": self.helpful_score,
            "last_reused": self.last_reused,
            "timestamp": self.timestamp,
            "embedding_id": self.embedding_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        """从字典反序列化"""
        return cls(
            id=data.get("id", ""),
            project=data.get("project", ""),
            task=data.get("task", ""),
            task_type=data.get("task_type", "general"),
            error_type=data.get("error_type", ""),
            difficulty=data.get("difficulty", "medium"),
            preconditions=data.get("preconditions", []),
            applicable_scenarios=data.get("applicable_scenarios", []),
            summary=data.get("summary", ""),
            detail=data.get("detail", ""),
            tags=data.get("tags", []),
            tool_calls_count=data.get("tool_calls_count", 0),
            success=data.get("success", True),
            related_files=data.get("related_files", []),
            reuse_count=data.get("reuse_count", 0),
            helpful_score=data.get("helpful_score", 0.0),
            last_reused=data.get("last_reused", ""),
            timestamp=data.get("timestamp", ""),
            embedding_id=data.get("embedding_id", ""),
        )


# ============================================================
# 情景记忆
# ============================================================

@dataclass
class EpisodicMemory:
    """
    情景记忆 —— "发生过什么"。

    记录完整的执行轨迹和关键决策点，与程序性经验互补。
    程序性经验回答"遇到 X 应该怎么做"，情景记忆回答"上次做 Y 时发生了 Z"。
    """

    id: str
    project: str
    task: str
    task_type: str = "general"

    # 执行轨迹
    context_before: str = ""                         # 执行前状态摘要
    decision_chain: list[dict] = field(default_factory=list)  # [{"step": 1, "decision": "...", "reason": "..."}]
    context_after: str = ""                          # 执行后状态摘要

    # 关联的程序性经验
    derived_procedural_id: str = ""                  # 提炼出的 MemoryEntry ID

    # 元信息
    tool_calls_count: int = 0
    success: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project": self.project,
            "task": self.task,
            "task_type": self.task_type,
            "context_before": self.context_before,
            "decision_chain": self.decision_chain,
            "context_after": self.context_after,
            "derived_procedural_id": self.derived_procedural_id,
            "tool_calls_count": self.tool_calls_count,
            "success": self.success,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodicMemory":
        return cls(
            id=data.get("id", ""),
            project=data.get("project", ""),
            task=data.get("task", ""),
            task_type=data.get("task_type", "general"),
            context_before=data.get("context_before", ""),
            decision_chain=data.get("decision_chain", []),
            context_after=data.get("context_after", ""),
            derived_procedural_id=data.get("derived_procedural_id", ""),
            tool_calls_count=data.get("tool_calls_count", 0),
            success=data.get("success", True),
            timestamp=data.get("timestamp", ""),
        )
