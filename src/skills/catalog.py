"""
Skill 数据层 —— Skill 定义加载、分类存储、查询

设计思路：
- 所有 Skill 定义集中在 definitions.yaml，加载后转为强类型 Skill dataclass
- 按 category（场景目录）和 tag（能力标签）建立索引，O(1) 查询
- 新增 Skill 只需在 YAML 加一段，Catalog 自动感知
- 与 ToolRegistry 解耦：Catalog 只存元数据，Tool 执行仍走 ToolRegistry
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ============================================================
# 数据模型
# ============================================================

@dataclass
class SkillCategory:
    """场景分类目录"""
    name: str                     # 内部标识: debug / develop / test / document / git
    label: str                    # 人类可读名: "调试与排错"
    description: str = ""


@dataclass
class Skill:
    """
    单个 Skill 的定义。

    这是路由系统的最小单元 —— LLM 看到的就是一个 Skill。
    每个 Skill 绑定一组底层 tool，被路由选中后，tool schema 才传给 LLM。
    """
    name: str                              # 唯一标识: fix_unit_test
    display_name: str                      # 人类可读: "修复单元测试"
    description: str                       # LLM排序用的用途说明
    category: str                          # 所属分类目录名
    tags: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    prompt_template: str = ""              # 注入给 Agent 的上下文
    examples: list[dict] = field(default_factory=list)
    priority: int = 5                      # 默认优先级 1-10

    def to_brief(self) -> str:
        """生成给精排 LLM 看的简短描述（省 Token）"""
        tags_str = ", ".join(self.tags[:5])  # 最多展示5个标签
        return f"{self.name}: {self.display_name} — {self.description} [标签: {tags_str}]"


# ============================================================
# Catalog 类
# ============================================================

class SkillCatalog:
    """
    Skill 数据仓库。

    职责：
    - 从 YAML 文件加载所有 Skill 定义和分类信息
    - 提供按分类/标签/名称的查询接口
    - 不涉及路由逻辑，纯数据层
    """

    def __init__(self, yaml_path: str = ""):
        self._skills: dict[str, Skill] = {}           # name → Skill
        self._categories: dict[str, SkillCategory] = {}  # name → Category
        self._by_category: dict[str, list[str]] = {}   # category → [skill_name]
        self._by_tag: dict[str, set[str]] = {}          # tag → {skill_name}
        if yaml_path:
            self.load(yaml_path)

    # ============================================================
    # 加载
    # ============================================================

    def load(self, yaml_path: str):
        """从 YAML 文件加载所有 Skill 定义"""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Skill 定义文件不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError("Skill 定义文件格式错误：顶层必须是字典")

        # 1. 加载分类目录
        for cat_data in raw.get("categories", []):
            cat = SkillCategory(
                name=cat_data["name"],
                label=cat_data.get("label", cat_data["name"]),
                description=cat_data.get("description", ""),
            )
            self._categories[cat.name] = cat
            self._by_category[cat.name] = []

        # 2. 加载 Skill 定义
        for skill_data in raw.get("skills", []):
            skill = Skill(
                name=skill_data["name"],
                display_name=skill_data.get("display_name", skill_data["name"]),
                description=skill_data.get("description", ""),
                category=skill_data.get("category", ""),
                tags=skill_data.get("tags", []),
                tools=skill_data.get("tools", []),
                prompt_template=skill_data.get("prompt_template", ""),
                examples=skill_data.get("examples", []),
                priority=skill_data.get("priority", 5),
            )
            self._skills[skill.name] = skill

            # 更新分类索引
            cat = skill.category
            if cat not in self._by_category:
                self._by_category[cat] = []
            self._by_category[cat].append(skill.name)

            # 更新标签索引
            for tag in skill.tags:
                tag_lower = tag.lower()
                if tag_lower not in self._by_tag:
                    self._by_tag[tag_lower] = set()
                self._by_tag[tag_lower].add(skill.name)

    # ============================================================
    # 查询接口
    # ============================================================

    def get_by_name(self, name: str) -> Optional[Skill]:
        """按名称精确查找"""
        return self._skills.get(name)

    def get_by_category(self, category: str) -> list[Skill]:
        """获取某个分类下的所有 Skill"""
        names = self._by_category.get(category, [])
        return [self._skills[n] for n in names if n in self._skills]

    def get_by_tag(self, tag: str) -> list[Skill]:
        """按标签查找 Skill"""
        names = self._by_tag.get(tag.lower(), set())
        return [self._skills[n] for n in names if n in self._skills]

    def get_by_tags(self, tags: list[str]) -> list[Skill]:
        """按多个标签联合查找（并集）"""
        result_names: set[str] = set()
        for tag in tags:
            result_names.update(self._by_tag.get(tag.lower(), set()))
        return [self._skills[n] for n in result_names if n in self._skills]

    def get_all(self) -> list[Skill]:
        """返回所有 Skill"""
        return list(self._skills.values())

    def get_all_tool_names(self) -> set[str]:
        """返回所有底层工具名（所有 Skill 引用的 tool 的并集）"""
        tools: set[str] = set()
        for skill in self._skills.values():
            tools.update(skill.tools)
        return tools

    def get_by_tool_name(self, tool_name: str) -> Optional[Skill]:
        """按底层工具名反向查找所属 Skill"""
        for skill in self._skills.values():
            if tool_name in skill.tools:
                return skill
        return None

    def list_categories(self) -> list[SkillCategory]:
        """列出所有分类"""
        return list(self._categories.values())

    def list_tags(self) -> list[str]:
        """列出所有已知标签"""
        return sorted(self._by_tag.keys())

    # ============================================================
    # 统计
    # ============================================================

    def count(self) -> int:
        """Skill 总数"""
        return len(self._skills)

    def category_count(self) -> int:
        """分类数"""
        return len(self._categories)

    def tag_count(self) -> int:
        """标签种类数"""
        return len(self._by_tag)
