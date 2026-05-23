"""
Skill 路由编排层 —— 串联召回与精排，对 Agent 提供统一接口

设计思路：
- 单入口 route(task) → 返回该任务需要的 skill name 列表
- 内部流程：Catalog(查) → Index(召回 15) → Ranker(精排 5)
- 缓存：同 task hash 路由结果，避免重复计算
- 按需追加：Agent 侧 LLM 请求了不在候选集的 tool → on_demand_add()
- 上下文注入：get_context_prompt(names) → 返回合并后的 Skill prompt templates
"""

import hashlib
from typing import Optional

from src.skills.catalog import SkillCatalog, Skill
from src.skills.index import SkillIndex
from src.skills.ranker import SkillRanker


class SkillRouter:
    """
    Skill 路由编排器。

    这是 Agent 直接使用的唯一接口。
    内部串联 Catalog → Index → Ranker 三层的完整路由流程。

    使用方式：
        router = SkillRouter(catalog, index, ranker)
        skill_names = router.route("修复 test_user 的测试失败")
        skill_schemas = registry.get_schemas_for(skill_names)
        # 只传 skill_schemas 给 LLM
    """

    def __init__(
        self,
        catalog: SkillCatalog,
        index: SkillIndex,
        ranker: SkillRanker,
        cache_size: int = 100,
    ):
        self._catalog = catalog
        self._index = index
        self._ranker = ranker

        # 简单 LRU 缓存：task_hash → [skill_name, ...]
        self._cache: dict[str, list[str]] = {}
        self._cache_order: list[str] = []  # 插入顺序，用于 LRU 淘汰
        self._max_cache = cache_size

        # 历史记录：最近一次路由使用的 skill 名称（供精排参考）
        self._last_routed: list[str] = []

        # 当前轮次活跃的 skill 集合（支持按需追加）
        self._active_skills: set[str] = set()

    # ============================================================
    # 主入口
    # ============================================================

    def route(self, task: str) -> list[str]:
        """
        根据用户任务，返回应暴露给 LLM 的 Skill 名称列表。

        流程：
        1. 查缓存（相同 task hash 直接返回）
        2. Index.recall(task) → 粗排 15 个候选
        3. Ranker.rank(task, candidates, history) → 精排 5 个
        4. 写缓存 + 更新活跃集
        """
        task_hash = self._hash(task)
        if task_hash in self._cache:
            cached = self._cache[task_hash]
            self._active_skills = set(cached)
            return cached

        # 一阶段：召回
        candidates = self._index.recall(task, top_k=15)

        if not candidates:
            # 召回为空 → 返回所有高频 Skill（兜底）
            fallback = self._get_fallback_skills()
            self._active_skills = set(fallback)
            return fallback

        # 二阶段：精排
        history = self._last_routed if self._last_routed else None
        ranked = self._ranker.rank(task, candidates, history_names=history)

        result_names = [s.name for s in ranked]
        self._last_routed = result_names
        self._active_skills = set(result_names)
        self._add_cache(task_hash, result_names)

        return result_names

    # ============================================================
    # 按需追加
    # ============================================================

    def on_demand_add(self, tool_name: str) -> bool:
        """
        Agent 侧 LLM 请求了不在当前候选集的工具 → 动态追加。

        通过反向查找确定 tool_name 属于哪个 Skill，
        然后将该 Skill 加入活跃集。

        Returns:
            True 如果工具被成功追加
        """
        # 检查活跃 Skill 已覆盖的 tool 名
        if tool_name in self.get_active_tool_names():
            return True

        # 反向查找：tool_name 属于哪个 Skill？
        skill = self._catalog.get_by_tool_name(tool_name)
        if skill is None:
            return False

        self._active_skills.add(skill.name)
        return True

    def get_active_tool_names(self) -> set[str]:
        """
        返回当前活跃 Skill 集所引用的所有底层 tool 名称。

        这是最终传给 ToolRegistry.get_schemas_for() 的参数。
        """
        tools: set[str] = set()
        for skill_name in self._active_skills:
            skill = self._catalog.get_by_name(skill_name)
            if skill:
                tools.update(skill.tools)
        return tools

    def get_active_skill_names(self) -> set[str]:
        """返回当前活跃的 Skill 名称集"""
        return self._active_skills.copy()

    # ============================================================
    # 上下文注入
    # ============================================================

    def get_context_prompt(self, skill_names: Optional[list[str]] = None) -> str:
        """
        合并指定 Skill 的 prompt_template，返回注入给 Agent 的上下文。

        如果未指定 skill_names，使用当前活跃集。
        """
        names = skill_names or list(self._active_skills)
        if not names:
            return ""

        parts = ["## 当前任务可用能力\n"]
        for name in names:
            skill = self._catalog.get_by_name(name)
            if skill and skill.prompt_template:
                parts.append(f"### {skill.display_name}\n{skill.prompt_template}\n")

        return "\n".join(parts) if len(parts) > 1 else ""

    # ============================================================
    # 缓存管理
    # ============================================================

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _add_cache(self, task_hash: str, result: list[str]):
        """写入缓存，超过容量时淘汰最旧条目"""
        if len(self._cache_order) >= self._max_cache:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)
        self._cache[task_hash] = result
        self._cache_order.append(task_hash)

    def _get_fallback_skills(self) -> list[str]:
        """召回为空时的兜底：返回覆盖四大基础场景的 Skill"""
        fallback = []
        for cat in ["develop", "debug", "test", "document"]:
            skills = self._catalog.get_by_category(cat)
            if skills:
                fallback.append(skills[0].name)
        return fallback[:5] if fallback else [s.name for s in self._catalog.get_all()[:5]]

    def clear_cache(self):
        """清空路由缓存"""
        self._cache.clear()
        self._cache_order.clear()

    # ============================================================
    # 统计
    # ============================================================

    def stats(self) -> dict:
        """路由统计信息"""
        return {
            "total_skills": self._catalog.count(),
            "total_categories": self._catalog.category_count(),
            "total_tags": self._catalog.tag_count(),
            "active_skills": len(self._active_skills),
            "cache_size": len(self._cache),
            "last_route": self._last_routed,
        }
