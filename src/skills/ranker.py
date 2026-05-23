"""
Skill 精排层 —— 二阶段 LLM 排序

设计思路：
- 候选 15 个 Skill 对工具调用仍太多，用量化 LLM 排序压缩到 top 3-5
- 精排 Prompt 极简：只发 skill_id + display_name + description + tags 摘要
- LLM 返回 JSON 数组，按相关度降序，temperature=0 保证确定性
- 支持传入历史调用记录作为辅助信号
- 解析失败时有 fallback：按粗排分数直接截断
"""

import json
import re
from typing import Optional

from src.skills.catalog import Skill


# ============================================================
# 精排 Prompt 模板
# ============================================================

_RANKING_SYSTEM_PROMPT = """你是一个任务与工具匹配专家。你的职责是根据用户任务，从候选工具列表中选择最匹配的3-5个。

## 选择标准
1. 相关性：工具是否能直接帮助完成该任务
2. 必要性：没有这个工具，任务是否无法完成
3. 互补性：选出的工具之间应各司其职，避免功能重叠

## 输出格式
严格输出 JSON 数组，按相关度降序，只包含 skill 名称：
["skill_name1", "skill_name2", "skill_name3"]

不要输出任何其他内容。不要输出解释、不要输出 markdown 代码块标记。"""


def build_ranking_prompt(task: str, candidates: list[Skill], history_names: Optional[list[str]] = None) -> str:
    """构建精排用的用户消息（尽量精简以省 Token）"""
    lines = [f"## 用户任务\n{task}\n"]

    lines.append("## 候选工具列表")
    for i, skill in enumerate(candidates, 1):
        lines.append(f"{i}. {skill.to_brief()}")

    if history_names:
        history_str = ", ".join(h for h in history_names if h)
        if history_str:
            lines.append(f"\n## 历史参考\n上次类似任务使用了: {history_str}")

    lines.append(f"\n## 请从以上 {len(candidates)} 个候选中选出最匹配的 3-5 个。")
    return "\n".join(lines)


# ============================================================
# Ranker 类
# ============================================================

class SkillRanker:
    """
    LLM 精排器。

    使用与 Agent 相同的 LLMClient，但一次精排只需 ~400 tokens 的 prompt
    + ~100 tokens 的 completion，相比传全量 40 个 tool schema 节约显著。
    """

    def __init__(self, llm_client, top_k: int = 5):
        """
        Args:
            llm_client: LLMClient 实例（复用现有 LLM 连接）
            top_k: 精排后保留的 Skill 数量
        """
        self._llm = llm_client
        self._top_k = top_k

    def rank(
        self,
        task: str,
        candidates: list[tuple[Skill, float]],
        history_names: Optional[list[str]] = None,
    ) -> list[Skill]:
        """
        对粗排候选进行 LLM 精排。

        Args:
            task: 用户原始任务描述
            candidates: 粗排结果 [(skill, score), ...]
            history_names: 历史上相似任务使用的 skill 名称

        Returns:
            精排后的 top_k 个 Skill，按相关度降序
        """
        if len(candidates) <= self._top_k:
            # 候选已经很少，直接返回
            return [s for s, _ in candidates]

        skills = [s for s, _ in candidates]

        # 构建 prompt
        user_message = build_ranking_prompt(task, skills, history_names)

        messages = [
            {"role": "system", "content": _RANKING_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        try:
            response = self._llm.chat(messages=messages, tools=[])
            ranked_names = self._parse_response(response.content or "")
        except Exception:
            # LLM 调用失败 → fallback：按粗排分数直接截断
            return [s for s, _ in candidates[:self._top_k]]

        # 按 LLM 返回的顺序重排
        ranked_skills: list[Skill] = []
        skill_map = {s.name: s for s in skills}

        for name in ranked_names:
            if name in skill_map:
                ranked_skills.append(skill_map[name])

        # 如果 LLM 返回的结果不够 top_k，用粗排结果补充
        if len(ranked_skills) < self._top_k:
            existing = {s.name for s in ranked_skills}
            for skill, _ in candidates:
                if skill.name not in existing and len(ranked_skills) < self._top_k:
                    ranked_skills.append(skill)
                    existing.add(skill.name)

        return ranked_skills[:self._top_k]

    def _parse_response(self, content: str) -> list[str]:
        """
        解析 LLM 返回的 JSON 数组。

        容错设计：LLM 可能返回带 markdown 标记的 JSON，
        先用正则提取 JSON 数组部分，再解析。
        """
        if not content:
            return []

        # 尝试直接解析
        try:
            result = json.loads(content.strip())
            if isinstance(result, list):
                return [str(item) for item in result]
        except json.JSONDecodeError:
            pass

        # 容错：提取方括号内的内容
        match = re.search(r'\[.*?\]', content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return [str(item) for item in result]
            except json.JSONDecodeError:
                pass

        # 最后容错：按行提取看起来像 skill name 的内容
        names = []
        for line in content.split("\n"):
            name = line.strip().strip('",[] ')
            if name and not name.startswith("{") and not name.startswith("//"):
                names.append(name)
        return names[:self._top_k]
