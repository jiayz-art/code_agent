"""
Skill 召回层 —— 一阶段粗筛（倒排索引 + 双路匹配）

设计思路：
- 构建倒排索引：tag → skill_names，keyword → skill_names（加载时预建，运行时只查）
- 双路召回：标签匹配（精确命中 Skill 标签）+ 关键词匹配（分词后在 Skill 描述的倒排索引中查找）
- 粗排打分：标签匹配 × 2.0 + 关键词匹配 × 1.0 + priority × 0.1
- 结果截断：输出 top-15 候选，进入精排阶段
- 中文分词用 jieba（可选依赖）；未安装时退回 2-gram 切分
"""

import re
import math
from typing import Optional

from src.skills.catalog import SkillCatalog, Skill


# ============================================================
# 中文分词适配（尽量使用 jieba，不可用时回退到 2-gram）
# ============================================================

try:
    import jieba
    _jieba_available = True
except ImportError:
    _jieba_available = False


# 中英文停用词
_STOP_WORDS = {
    # 中文
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "这个", "那个", "可以", "还是", "如果", "的话", "吗", "吧", "呢", "啊",
    # 英文
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "it", "its", "this", "that", "these", "those", "and", "but", "or",
    "not", "no", "if", "then", "else", "when", "where", "how", "what",
    "which", "who", "whom", "i", "me", "my", "we", "our", "you", "your",
}


def _tokenize(text: str) -> list[str]:
    """将文本分词，返回有意义的词条列表"""
    if _jieba_available:
        words = jieba.lcut(text)
    else:
        # 回退：2-gram 中文 + 空格分词英文
        words = []
        # 提取中英文单词
        tokens = re.findall(r'[一-鿿]+|[a-zA-Z0-9_-]+', text)
        for token in tokens:
            if re.match(r'[一-鿿]+', token):
                # 中文 2-gram
                for i in range(len(token) - 1):
                    words.append(token[i:i+2])
                if len(token) >= 1:
                    words.append(token)  # 保留原词
            else:
                words.append(token.lower())
    # 去停用词 + 去重 + 去太短
    return list(dict.fromkeys(
        w for w in words
        if w.lower() not in _STOP_WORDS and len(w) >= 2
    ))


# ============================================================
# 倒排索引
# ============================================================

class SkillIndex:
    """
    Skill 倒排索引 —— 一阶段召回引擎。

    首次实例化时从 SkillCatalog 构建索引，之后查询为 O(1) 哈希查找。
    """

    def __init__(self, catalog: SkillCatalog):
        self._catalog = catalog

        # 倒排索引：tag → {skill_name}
        self._tag_index: dict[str, set[str]] = {}

        # 倒排索引：keyword → {skill_name}
        self._keyword_index: dict[str, set[str]] = {}

        # 每个 Skill 的关键词频率（用于粗排打分）
        self._skill_keyword_scores: dict[str, dict[str, float]] = {}

        # 已知标签集合（从 YAML 标签 + 通用编程标签合成）
        self._known_tags: set[str] = set()

        self._build()

    # ============================================================
    # 索引构建
    # ============================================================

    def _build(self):
        """一次性构建所有倒排索引"""

        # 1. 收集所有标签
        for skill in self._catalog.get_all():
            for tag in skill.tags:
                self._known_tags.add(tag.lower())
                if tag.lower() not in self._tag_index:
                    self._tag_index[tag.lower()] = set()
                self._tag_index[tag.lower()].add(skill.name)

        # 2. 为每个 Skill 的关键词建索引
        for skill in self._catalog.get_all():
            # 合成 Skill 的可搜索文本
            search_text = (
                f"{skill.display_name} {skill.description} "
                f"{' '.join(skill.tags)} "
                f"{' '.join(ex.get('task', '') for ex in skill.examples)}"
            )
            keywords = _tokenize(search_text)
            scores: dict[str, float] = {}

            for kw in keywords:
                # TF: 词频
                scores[kw] = scores.get(kw, 0) + 1

                # 倒排索引
                if kw not in self._keyword_index:
                    self._keyword_index[kw] = set()
                self._keyword_index[kw].add(skill.name)

            # TF 归一化
            total = sum(scores.values()) or 1
            self._skill_keyword_scores[skill.name] = {
                k: v / total for k, v in scores.items()
            }

    # ============================================================
    # 召回
    # ============================================================

    def recall(self, task: str, top_k: int = 15) -> list[tuple[Skill, float]]:
        """
        根据用户任务召回候选 Skill。

        流程：
        1. 从 task 中提取已知标签
        2. 对 task 分词做关键词匹配
        3. 合并结果并粗排打分
        4. 返回 top_k 候选

        Returns:
            [(skill, score), ...] 按分数降序排列
        """
        task_lower = task.lower()
        task_keywords = _tokenize(task)

        # ---- 路1：标签匹配 ----
        tag_hits: dict[str, float] = {}
        for tag in self._known_tags:
            if tag in task_lower or tag.replace("-", "") in task_lower.replace(" ", ""):
                for skill_name in self._tag_index.get(tag, set()):
                    tag_hits[skill_name] = tag_hits.get(skill_name, 0) + 1

        # ---- 路2：关键词匹配 ----
        kw_hits: dict[str, float] = {}
        for kw in task_keywords:
            for skill_name in self._keyword_index.get(kw, set()):
                # 用 Skill 中该关键词的 TF 作为权重
                tf = self._skill_keyword_scores.get(skill_name, {}).get(kw, 0.1)
                kw_hits[skill_name] = kw_hits.get(skill_name, 0) + tf

        # ---- 合并打分 ----
        all_candidates: dict[str, float] = {}
        all_skill_names = set(tag_hits.keys()) | set(kw_hits.keys())

        for skill_name in all_skill_names:
            skill = self._catalog.get_by_name(skill_name)
            if skill is None:
                continue
            tag_score = tag_hits.get(skill_name, 0) * 2.0
            kw_score = kw_hits.get(skill_name, 0) * 1.0
            priority_score = skill.priority * 0.1
            all_candidates[skill_name] = tag_score + kw_score + priority_score

        # ---- 排序 + 截断 ----
        sorted_items = sorted(
            all_candidates.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        result = []
        for name, score in sorted_items:
            skill = self._catalog.get_by_name(name)
            if skill:
                result.append((skill, score))

        return result

    # ============================================================
    # 查询
    # ============================================================

    def get_known_tags(self) -> set[str]:
        """返回所有已知标签"""
        return self._known_tags.copy()

    def get_tag_skills(self, tag: str) -> list[str]:
        """获取某个标签关联的 Skill 名称列表"""
        return list(self._tag_index.get(tag.lower(), set()))
