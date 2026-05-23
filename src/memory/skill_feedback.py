"""
Skill 生长反馈循环 —— 从记忆中发掘能力缺口

设计思路：
- 定期分析记忆库，发现高频但未被 Skill 覆盖的任务模式
- 输出三类建议：新增 Skill / 优化现有 Skill / 废弃过时 Skill
- 与 SkillCatalog 解耦：只输出建议，由管理员或自动化流程决定是否应用
- 反馈质量取决于记忆数量和多样性，初期可能建议较少
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from src.memory.models import MemoryEntry


@dataclass
class SkillSuggestion:
    """Skill 优化建议"""
    action: str                              # add / refine / deprecate
    skill_name: str                          # 建议的 skill 名称
    reason: str                              # 建议理由
    suggested_tools: list[str] = field(default_factory=list)
    suggested_description: str = ""
    suggested_tags: list[str] = field(default_factory=list)
    evidence_count: int = 0                  # 支撑此建议的记忆条数
    avg_success_rate: float = 0.0            # 相关任务的平均成功率


class SkillFeedbackLoop:
    """
    Skill 生长反馈循环。

    从记忆库中分析任务模式，发现能力缺口并提出 Skill 优化建议。
    """

    def __init__(self, min_evidence: int = 3, similarity_threshold: float = 0.6):
        self._min_evidence = min_evidence
        self._similarity_threshold = similarity_threshold

    def analyze(
        self,
        memories: list[MemoryEntry],
        existing_skill_names: set[str],
        existing_tags: set[str],
    ) -> list[SkillSuggestion]:
        """
        分析记忆库并返回 Skill 优化建议。

        Args:
            memories: 记忆条目列表
            existing_skill_names: 已有的 Skill 名称集合
            existing_tags: 已有的标签集合

        Returns:
            Skill 优化建议列表
        """
        if len(memories) < self._min_evidence:
            return []

        suggestions: list[SkillSuggestion] = []

        # 1. 发现高频但未被覆盖的任务类型
        suggestions.extend(self._find_uncovered_patterns(
            memories, existing_skill_names, existing_tags
        ))

        # 2. 发现可优化的现有 Skill（成功率低的模式）
        suggestions.extend(self._find_refinement_opportunities(
            memories, existing_skill_names
        ))

        # 3. 发现可能过时的 Skill（长期未被触发的）
        suggestions.extend(self._find_deprecated_skills(
            memories, existing_skill_names
        ))

        return suggestions

    def _find_uncovered_patterns(
        self,
        memories: list[MemoryEntry],
        existing_skill_names: set[str],
        existing_tags: set[str],
    ) -> list[SkillSuggestion]:
        """发现未被现有 Skill 覆盖的高频任务模式"""
        suggestions = []

        # 按 task_type 聚合统计
        type_counter = Counter(m.task_type for m in memories)
        tag_counter = Counter()
        for m in memories:
            tag_counter.update(m.tags)

        tool_counter = Counter()
        for m in memories:
            tool_counter.update(m.tags)

        # 发现高频标签但无对应 Skill 覆盖
        for tag, count in tag_counter.most_common(20):
            if count < self._min_evidence:
                continue
            if tag in existing_tags:
                continue
            # 检查是否有 Skill 名称中包含此标签关键词
            covered = any(
                tag in skill_name.lower()
                for skill_name in existing_skill_names
            )
            if covered:
                continue

            # 收集使用此标签的记忆
            related = [
                m for m in memories
                if tag in m.tags
            ]
            success_rate = sum(1 for m in related if m.success) / len(related) if related else 0
            tools_used = Counter()
            for m in related:
                tools_used.update(m.related_files[:3])

            suggestions.append(SkillSuggestion(
                action="add",
                skill_name=f"handle_{tag.replace('-', '_')}",
                reason=f"发现 {count} 条记忆使用标签 '{tag}'，但无对应 Skill 覆盖",
                suggested_tags=[tag],
                suggested_description=f"处理 {tag} 相关任务",
                evidence_count=count,
                avg_success_rate=success_rate,
            ))

        return suggestions

    def _find_refinement_opportunities(
        self,
        memories: list[MemoryEntry],
        existing_skill_names: set[str],
    ) -> list[SkillSuggestion]:
        """发现需要优化的 Skill（高频率但低成功率）"""
        suggestions = []

        # 按 task_type 分组统计成功率
        type_stats: dict[str, dict] = {}
        for m in memories:
            if m.task_type not in type_stats:
                type_stats[m.task_type] = {"total": 0, "success": 0, "errors": Counter()}
            stats = type_stats[m.task_type]
            stats["total"] += 1
            if m.success:
                stats["success"] += 1
            else:
                stats["errors"][m.error_type] += 1

        for task_type, stats in type_stats.items():
            if stats["total"] < self._min_evidence:
                continue
            success_rate = stats["success"] / stats["total"]
            if success_rate < 0.5 and stats["total"] >= self._min_evidence:
                # 找出最常出现的错误
                top_errors = stats["errors"].most_common(3)
                suggestions.append(SkillSuggestion(
                    action="refine",
                    skill_name=task_type,
                    reason=(
                        f"'{task_type}' 类型任务成功率仅 {success_rate:.0%} "
                        f"({stats['success']}/{stats['total']})。"
                        f"常见错误: {', '.join(f'{e}({c}次)' for e, c in top_errors)}"
                    ),
                    evidence_count=stats["total"],
                    avg_success_rate=success_rate,
                ))

        return suggestions

    def _find_deprecated_skills(
        self,
        memories: list[MemoryEntry],
        existing_skill_names: set[str],
    ) -> list[SkillSuggestion]:
        """发现可能过时的 Skill（长期无相关记忆）"""
        # 获取所有 task_type 的最近使用时间
        type_last_used: dict[str, str] = {}
        for m in memories:
            if m.task_type not in type_last_used or m.timestamp > type_last_used[m.task_type]:
                type_last_used[m.task_type] = m.timestamp

        suggestions = []
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=60)

        for task_type, last_used in type_last_used.items():
            try:
                last = datetime.fromisoformat(last_used.replace("Z", "+00:00"))
                if last < cutoff:
                    suggestions.append(SkillSuggestion(
                        action="deprecate",
                        skill_name=task_type,
                        reason=f"'{task_type}' 类型最近使用于 {last_used[:10]}，已超过60天无活动",
                        evidence_count=0,
                    ))
            except (ValueError, TypeError):
                pass

        return suggestions

    def generate_user_profile_update(
        self,
        memories: list[MemoryEntry],
        current_profile,
    ) -> dict:
        """从记忆中更新用户画像"""
        updates = {}

        # 技术栈统计
        tech_counter = Counter()
        style_counter = Counter()
        for m in memories:
            for tag in m.tags:
                if tag in ("python", "javascript", "typescript", "go", "rust",
                           "java", "sql", "react", "vue", "fastapi", "django",
                           "docker", "kubernetes", "redis", "postgresql"):
                    tech_counter[tag] += 1

            for tag in m.tags:
                if tag in ("minimal-comments", "verbose-comments", "strict-typing",
                           "loose-typing", "functional-style", "oop-style"):
                    style_counter[tag] += 1

        if tech_counter:
            updates["tech_stack"] = [t for t, _ in tech_counter.most_common(10)]

        # 模式发现
        pattern_counter = Counter()
        for m in memories:
            if m.success and m.task_type != "general":
                pattern_counter[m.task_type] += 1
        if pattern_counter:
            updates["common_patterns"] = [
                {"name": f"{ptype}_tasks", "frequency": count}
                for ptype, count in pattern_counter.most_common(5)
            ]

        return updates
