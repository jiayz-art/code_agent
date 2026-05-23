"""
记忆检索层 —— 语义搜索 + Token 预算控制

设计思路：
- 优先 ChromaDB 语义检索（理解意图），回退关键词匹配（纯文件模式）
- 多维加权：语义相似度 × task_type 匹配 × error_type 匹配 × 时间衰减
- Token 预算控制：逐条拼接 context，达到上限即截断，防止上下文爆炸
- 检索结果按最终得分降序，优先注入最有价值的经验
"""

import math
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

from src.memory.models import MemoryEntry
from src.memory.store import MemoryStore


# ============================================================
# Token 估算（简单字符/4 规则，适合中英文混合）
# ============================================================

def estimate_tokens(text: str) -> int:
    """粗略估算 token 数：英文 ~4 字符/token，中文 ~1.5 字符/token"""
    chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


# ============================================================
# MemoryRetriever
# ============================================================

class MemoryRetriever:
    """
    记忆检索器。

    流程：
    1. ChromaDB 语义检索 top_k=10 候选
    2. 维度加权：task_type ×1.5, error_type ×2.0, 时间衰减
    3. 重排序取 top N
    4. Token 预算控制拼接上下文
    """

    def __init__(self, store: MemoryStore, max_tokens: int = 800):
        self._store = store
        self._max_tokens = max_tokens

    # ============================================================
    # 主入口
    # ============================================================

    def retrieve_context(self, task: str, project: str) -> str:
        """
        检索相关记忆并返回可注入 Prompt 的上下文。

        Args:
            task: 当前用户任务
            project: 当前项目名

        Returns:
            记忆上下文文本（可注入 Agent system message）。
            如果没有相关记忆，返回空字符串。
        """
        entries = self.search(task, project, top_k=8)
        if not entries:
            return ""

        return self._build_context(task, entries)

    def search(
        self, task: str, project: str, top_k: int = 10
    ) -> list[MemoryEntry]:
        """
        语义搜索相关记忆条目。

        返回按加权得分降序排列的 MemoryEntry 列表。
        """
        # 1. ChromaDB 语义搜索
        chroma_results = self._store.search_similar(task, project=project, top_k=top_k)

        if not chroma_results:
            # 回退：关键词匹配
            return self._keyword_search(task, project, top_k)

        # 2. 加载完整 MemoryEntry
        entries: list[tuple[MemoryEntry, float]] = []
        for mem_id, similarity in chroma_results:
            entry = self._store.get_by_id(mem_id)
            if entry:
                entries.append((entry, similarity))

        if not entries:
            return self._keyword_search(task, project, top_k)

        # 3. 多维加权重排
        task_type = self._infer_task_type(task)
        error_type = self._infer_error_type(task)

        scored: list[tuple[MemoryEntry, float]] = []
        for entry, base_score in entries:
            final_score = self._calculate_score(entry, base_score, task_type, error_type)
            scored.append((entry, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in scored[:top_k]]

    # ============================================================
    # 评分函数
    # ============================================================

    def _calculate_score(
        self,
        entry: MemoryEntry,
        base_score: float,
        query_task_type: str,
        query_error_type: str,
    ) -> float:
        """多维加权计算最终得分"""
        score = base_score

        # 同类型任务加权 ×1.5
        if entry.task_type == query_task_type and query_task_type != "general":
            score *= 1.5

        # 同类型错误加权 ×2.0（错误模式匹配是最有价值的）
        if query_error_type and entry.error_type == query_error_type:
            score *= 2.0

        # ★ 复用次数加权：被反复验证的经验更有价值
        if entry.reuse_count > 0:
            score *= min(1.0 + entry.reuse_count * 0.15, 2.5)

        # ★ 有效性评分加权
        if entry.helpful_score > 0:
            score *= 0.8 + entry.helpful_score * 0.04  # score=5 → ×1.0

        # 时间衰减：7天前 ×0.8，30天前 ×0.5，90天前 ×0.3
        try:
            entry_time = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
            days_ago = (datetime.now(timezone.utc) - entry_time).days
            if days_ago > 90:
                score *= 0.3
            elif days_ago > 30:
                score *= 0.5
            elif days_ago > 7:
                score *= 0.8
        except (ValueError, TypeError):
            pass

        # 失败任务的教训比成功任务更有价值
        if not entry.success:
            score *= 1.2

        return score

    # ============================================================
    # Token 预算控制的上下文构建
    # ============================================================

    def _build_context(self, task: str, entries: list[MemoryEntry]) -> str:
        """
        将检索到的记忆拼成上下文文本，严格控制 Token 预算。

        优先保留高分、高信息量记忆；超出预算时截断最后一条。
        """
        header = "## 历史经验参考（从过往任务中沉淀）\n"
        # 简单说明用途（固定开销，约 30 tokens）
        usage_hint = (
            "以下是你过去在类似任务中积累的经验，请参考这些经验来提高效率、避免重复错误。\n\n"
        )
        context = header + usage_hint
        token_budget = self._max_tokens - estimate_tokens(header + usage_hint)

        if token_budget <= 0:
            return ""

        added = 0
        for entry in entries:
            # 动态调整每条记忆的字符上限
            remaining = token_budget - added
            if remaining <= 20:
                break

            max_chars = min(300, remaining * 3)  # 粗略：1 token ≈ 3-4 字符
            snippet = entry.to_context_text(max_chars=max_chars)
            snippet_tokens = estimate_tokens(snippet)

            if added + snippet_tokens > token_budget:
                # 截断最后一条
                available = token_budget - added - 10  # 留 10 token 余量
                if available > 30:
                    snippet = entry.to_context_text(max_chars=available * 3)
                    snippet = snippet[: available * 3] + "..."
                    context += snippet + "\n"
                break

            context += snippet + "\n\n"
            added += snippet_tokens

        # 如果没有加入任何有效条目，返回空
        if context.strip() == (header + usage_hint).strip():
            return ""

        return context

    # ============================================================
    # 关键词回退搜索
    # ============================================================

    def _keyword_search(self, task: str, project: str, top_k: int) -> list[MemoryEntry]:
        """ChromaDB 不可用时的关键词匹配回退"""
        all_entries = self._store.load(project)
        if not all_entries:
            return []

        keywords = self._tokenize(task)
        if not keywords:
            return all_entries[:top_k]

        scored: list[tuple[MemoryEntry, float]] = []
        for entry in all_entries:
            score = 0.0
            search_text = f"{entry.task} {entry.summary} {' '.join(entry.tags)}".lower()
            for kw in keywords:
                if kw.lower() in search_text:
                    score += 1.0
            if score > 0:
                scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in scored[:top_k]]

    def _tokenize(self, text: str) -> list[str]:
        """中文 2-gram + 英文空格分词"""
        tokens: list[str] = []
        # 英文词
        eng_words = re.findall(r'[a-zA-Z0-9_]+', text)
        tokens.extend(w.lower() for w in eng_words if len(w) >= 2)
        # 中文 2-gram
        chinese = re.sub(r'[^一-鿿]', '', text)
        for i in range(len(chinese) - 1):
            tokens.append(chinese[i:i+2])
        return list(dict.fromkeys(tokens))

    # ============================================================
    # 任务类型/错误类型推断（供评分用，轻量规则）
    # ============================================================

    def _infer_task_type(self, task: str) -> str:
        """从任务描述快速推断任务类型（不需要 tool_logs）"""
        t = task.lower()
        if any(w in t for w in ["测试", "test", "pytest", "覆盖率", "assert"]):
            return "test"
        if any(w in t for w in ["修复", "fix", "bug", "报错", "错误", "失败", "error"]):
            return "bug-fix"
        if any(w in t for w in ["重构", "refactor", "拆分", "提取"]):
            return "refactor"
        if any(w in t for w in ["文档", "doc", "readme", "注释"]):
            return "document"
        if any(w in t for w in ["实现", "添加", "新增", "开发", "创建", "add", "implement"]):
            return "feature"
        if any(w in t for w in ["调试", "debug", "排查", "分析"]):
            return "debug"
        if any(w in t for w in ["提交", "commit", "分支", "branch"]):
            return "git"
        return "general"

    def _infer_error_type(self, task: str) -> str:
        """从任务描述快速推断错误类型"""
        t = task.lower()
        if any(w in t for w in ["assertionerror", "assertion", "断言"]): return "AssertionError"
        if any(w in t for w in ["attributeerror", "has no attribute", "属性"]): return "AttributeError"
        if any(w in t for w in ["importerror", "no module", "导入"]): return "ImportError"
        if any(w in t for w in ["keyerror", "key error"]): return "KeyError"
        if any(w in t for w in ["typeerror", "type error", "类型"]): return "TypeError"
        if any(w in t for w in ["valueerror", "value error"]): return "ValueError"
        if any(w in t for w in ["timeout", "超时"]): return "TimeoutError"
        if any(w in t for w in ["filenotfound", "no such file", "找不到"]): return "FileNotFoundError"
        if any(w in t for w in ["permission", "权限", "denied"]): return "PermissionError"
        if any(w in t for w in ["测试失败", "test fail"]): return "test-failure"
        if any(w in t for w in ["配置", "config", ".env"]): return "config-error"
        return ""
