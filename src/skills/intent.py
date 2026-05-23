"""
Skill 意图分类器 —— IntentClassifier

设计思路：
- 对用户输入进行粗粒度意图分类，作为 Skill 路由的前置筛选
- 规则优先：关键词+模式匹配覆盖 90% 场景（零 Token 开销）
- LLM 回退：规则无法匹配时用轻量 Prompt 分类（仅对模糊输入触发）
- 分类结果 + 置信度，供 SkillRanker 精排时加权调整
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class IntentCategory(str, Enum):
    """用户意图分类"""
    DEVELOP = "develop"        # 开发新功能、写代码
    DEBUG = "debug"            # 调试、修复 bug
    TEST = "test"              # 编写/运行测试
    REFACTOR = "refactor"      # 重构、代码优化
    EXPLORE = "explore"        # 探索代码库、理解代码
    DOCUMENT = "document"      # 文档、注释
    CONFIGURE = "configure"    # 配置、环境、依赖
    REVIEW = "review"          # 代码审查
    DEPLOY = "deploy"          # 部署、CI/CD
    CHAT = "chat"              # 闲聊、问答（非编码类）
    UNKNOWN = "unknown"        # 无法分类


# 意图 → 对应的 Skill 分类
INTENT_TO_CATEGORY: dict[IntentCategory, list[str]] = {
    IntentCategory.DEVELOP: ["develop"],
    IntentCategory.DEBUG: ["debug"],
    IntentCategory.TEST: ["test"],
    IntentCategory.REFACTOR: ["develop", "refactor"],
    IntentCategory.EXPLORE: ["explore"],
    IntentCategory.DOCUMENT: ["document"],
    IntentCategory.CONFIGURE: ["configure"],
    IntentCategory.REVIEW: ["review"],
    IntentCategory.DEPLOY: ["deploy"],
    IntentCategory.CHAT: [],
    IntentCategory.UNKNOWN: [],
}


@dataclass
class IntentResult:
    """意图分类结果"""
    category: IntentCategory
    confidence: float           # 0.0 ~ 1.0
    keywords_matched: list[str] = field(default_factory=list)
    secondary: list[IntentCategory] = field(default_factory=list)  # 次要意图


class IntentClassifier:
    """
    用户意图分类器。

    使用方式：
        classifier = IntentClassifier()
        result = classifier.classify("帮我修复 login 模块的 bug")
        # result.category == IntentCategory.DEBUG
        # result.confidence == 0.85
    """

    def __init__(self, llm_client=None):
        self._llm = llm_client

        # 意图关键词表（意图→置信度贡献因子）
        self._intent_keywords: dict[IntentCategory, list[tuple[str, float]]] = {
            IntentCategory.DEVELOP: [
                ("开发", 0.9), ("实现", 0.85), ("创建", 0.7), ("新增", 0.75),
                ("添加功能", 0.9), ("写一个", 0.85), ("生成代码", 0.9),
                ("implement", 0.85), ("create", 0.7), ("add feature", 0.9),
                ("build", 0.8), ("scaffold", 0.85),
            ],
            IntentCategory.DEBUG: [
                ("修复", 0.9), ("bug", 0.85), ("调试", 0.9), ("报错", 0.85),
                ("错误", 0.7), ("异常", 0.75), ("失败", 0.6), ("崩溃", 0.85),
                ("不工作", 0.8), ("问题", 0.5), ("排查", 0.85), ("定位", 0.75),
                ("fix", 0.85), ("debug", 0.9), ("error", 0.7), ("crash", 0.85),
                ("broken", 0.8), ("issue", 0.6),
            ],
            IntentCategory.TEST: [
                ("测试", 0.9), ("单元测试", 0.95), ("集成测试", 0.9),
                ("test", 0.85), ("unittest", 0.95), ("pytest", 0.9),
                ("覆盖率", 0.85), ("coverage", 0.85), ("mock", 0.75),
            ],
            IntentCategory.REFACTOR: [
                ("重构", 0.9), ("优化", 0.6), ("整理", 0.7), ("拆分", 0.75),
                ("提取", 0.7), ("重命名", 0.85), ("迁移", 0.7),
                ("refactor", 0.9), ("optimize", 0.6), ("clean up", 0.8),
                ("restructure", 0.85), ("rename", 0.8),
            ],
            IntentCategory.EXPLORE: [
                ("查找", 0.8), ("搜索", 0.8), ("定位", 0.65), ("在哪里", 0.9),
                ("怎么实现", 0.75), ("如何工作", 0.85), ("解释", 0.7),
                ("explain", 0.7), ("find", 0.75), ("search", 0.75),
                ("where is", 0.9), ("how does", 0.8), ("理解", 0.75),
                ("查看", 0.6), ("检查", 0.5),
            ],
            IntentCategory.DOCUMENT: [
                ("文档", 0.9), ("注释", 0.85), ("README", 0.9),
                ("docstring", 0.9), ("API 文档", 0.9),
                ("document", 0.85), ("comment", 0.75), ("说明", 0.65),
            ],
            IntentCategory.CONFIGURE: [
                ("配置", 0.85), ("安装", 0.75), ("依赖", 0.8), ("环境", 0.75),
                ("config", 0.85), ("install", 0.75), ("setup", 0.8),
                ("deploy", 0.6), ("Docker", 0.85), ("CI", 0.8),
                ("docker-compose", 0.9), ("pip install", 0.85),
            ],
            IntentCategory.REVIEW: [
                ("审查", 0.9), ("review", 0.9), ("检查代码", 0.8),
                ("审计", 0.85), ("audit", 0.85), ("安全审查", 0.9),
                ("code review", 0.95),
            ],
            IntentCategory.DEPLOY: [
                ("部署", 0.9), ("发布", 0.85), ("上线", 0.9),
                ("deploy", 0.9), ("release", 0.85), ("publish", 0.8),
                ("CI/CD", 0.9), ("构建", 0.7), ("打包", 0.75),
            ],
        }

    def classify(self, user_input: str) -> IntentResult:
        """
        对用户输入进行意图分类（规则匹配）。

        Args:
            user_input: 用户的自然语言输入

        Returns:
            IntentResult: 包含分类结果和置信度
        """
        text = user_input.lower().strip()
        if not text:
            return IntentResult(IntentCategory.UNKNOWN, 0.0)

        # 累积每个意图的得分
        scores: dict[IntentCategory, float] = {}
        all_matched: dict[IntentCategory, list[str]] = {}

        for category, keywords in self._intent_keywords.items():
            cat_score = 0.0
            matched = []
            for kw, weight in keywords:
                if kw.lower() in text:
                    cat_score = max(cat_score, weight)  # 取最大权重
                    matched.append(kw)
            if cat_score > 0:
                scores[category] = cat_score
                all_matched[category] = matched

        if not scores:
            # 规则无法匹配 → LLM 回退（如果可用）
            if self._llm:
                return self._llm_classify(user_input)
            return IntentResult(IntentCategory.UNKNOWN, 0.0)

        # 按得分排序，确定主意图和次要意图
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        primary_cat, primary_score = ranked[0]
        secondary = [cat for cat, _ in ranked[1:3]]

        return IntentResult(
            category=primary_cat,
            confidence=min(primary_score, 1.0),
            keywords_matched=all_matched.get(primary_cat, []),
            secondary=secondary,
        )

    def get_skill_categories(self, intent: IntentCategory) -> list[str]:
        """获取该意图对应的 Skill 分类列表"""
        return INTENT_TO_CATEGORY.get(intent, [])

    def _llm_classify(self, user_input: str) -> IntentResult:
        """轻量 LLM 分类回退（仅对关键词匹配失败的模糊输入触发）"""
        try:
            categories = "\n".join(
                f"- {c.value}: {c.name}" for c in IntentCategory
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是意图分类器。将用户输入分类为以下之一（只输出分类名）。\n"
                        f"{categories}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"分类以下输入: {user_input[:200]}",
                },
            ]
            response = self._llm.chat(messages=messages, tools=[])
            if response.content:
                content = response.content.strip().lower()
                for cat in IntentCategory:
                    if cat.value in content:
                        return IntentResult(cat, 0.6)
        except Exception:
            pass
        return IntentResult(IntentCategory.UNKNOWN, 0.0)
