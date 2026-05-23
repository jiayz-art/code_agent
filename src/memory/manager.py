"""
记忆系统编排层 —— MemoryManager

设计思路：
- 三级记忆体系：用户画像(跨项目) → 程序性经验(如何做) → 情景记忆(发生过什么)
- before_task(task, project) → 检索程序性经验 + 注入用户画像 → 返回上下文
- after_task(task, project, tool_logs, success) → 摘要同步 + 情景记录 + 提炼异步
- enabled 开关：关闭时所有方法无操作，零开销
- Skill 生长反馈：定期分析记忆库，输出 Skill 优化建议
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.state import ToolCallRecord
from src.memory.models import MemoryEntry, UserProfile, EpisodicMemory
from src.memory.reflector import MemoryReflector
from src.memory.store import MemoryStore
from src.memory.retriever import MemoryRetriever
from src.memory.skill_feedback import SkillFeedbackLoop, SkillSuggestion

# 模块级 logger（写入文件不输出到控制台）
_logger = logging.getLogger("agent.memory")


class MemoryManager:
    """
    记忆系统总入口。

    Agent 只与此类交互，三个钩子：
    - before_task(): 任务开始前调用，返回记忆上下文（含用户画像）
    - after_task():  任务结束后调用，触发反思和存储
    - get_feedback(): 获取 Skill 优化建议
    """

    def __init__(
        self,
        reflector: MemoryReflector,
        store: MemoryStore,
        retriever: MemoryRetriever,
        enabled: bool = True,
        async_reflection: bool = True,
        profile_path: str = "data/profile.json",
    ):
        self._reflector = reflector
        self._store = store
        self._retriever = retriever
        self.enabled = enabled
        self._async = async_reflection
        self._profile_path = Path(profile_path)

        # 用户画像（跨项目持久化）
        self._profile: UserProfile = self._load_profile()

        # Skill 生长反馈
        self._feedback_loop = SkillFeedbackLoop()

        # 统计
        self._total_memories_created = 0
        self._total_reflections = 0
        self._total_episodic_recorded = 0

        # 待处理的异步回调计数器
        self._pending_async = 0
        self._lock = threading.Lock()

    # ============================================================
    # Agent 钩子 1：任务前
    # ============================================================

    def before_task(self, task: str, project: str) -> str:
        """
        在新任务开始前调用。

        检索与当前任务相关的历史记忆 + 用户画像，
        返回可注入到 system message 的上下文文本。

        Args:
            task: 用户任务描述
            project: 当前项目标识符

        Returns:
            记忆上下文文本（含用户画像）。没有相关记忆时仅返回用户画像。
        """
        if not self.enabled:
            return ""

        parts = []

        # ★ 用户画像
        profile_text = self.get_profile_context()
        if profile_text:
            parts.append(profile_text)

        # 程序性经验检索
        try:
            context = self._retriever.retrieve_context(task, project)
            if context:
                parts.append(context)
                lines_count = len(context.split("\n"))
                self._log(f"记忆检索: 找到 {lines_count} 行上下文")
        except Exception as e:
            self._log(f"记忆检索失败: {e}")

        return "\n\n".join(parts) if parts else ""

    # ============================================================
    # Agent 钩子 2：任务后
    # ============================================================

    def after_task(
        self,
        task: str,
        project: str,
        tool_logs: list[ToolCallRecord],
        success: bool,
    ):
        """
        在任务结束后调用。

        流程：
        1. [同步] 生成摘要并立即存储基本信息 + 情景记忆 + 更新画像
        2. [异步] 后台生成详细经验并更新存储（含 embedding 重建）

        Args:
            task: 原始任务描述
            project: 项目标识符
            tool_logs: 工具调用日志列表
            success: 任务是否成功
        """
        if not self.enabled:
            return

        self._total_reflections += 1

        # ---- 同步：摘要 + 分类 ----
        summary = self._reflector.generate_summary(task, tool_logs)
        task_type = self._reflector.classify_task_type(task, tool_logs)
        error_type = ""
        if not success:
            error_type = self._reflector.extract_error_type(task, tool_logs)
        tags = self._reflector.extract_tags(task, summary)

        # 评估难度
        difficulty = self._assess_difficulty(tool_logs, success)

        memory_id = self._generate_id(project)
        timestamp = datetime.now(timezone.utc).isoformat()
        related_files = self._extract_files(tool_logs)

        # ---- 情景记忆记录 ----
        episodic_id = f"epi_{project}_{self._total_episodic_recorded + 1:04d}"
        episodic = EpisodicMemory(
            id=episodic_id,
            project=project,
            task=task,
            task_type=task_type,
            context_before=self._summarize_context_before(project),
            decision_chain=self._extract_decisions(tool_logs),
            context_after=summary,
            derived_procedural_id=memory_id,
            tool_calls_count=len(tool_logs),
            success=success,
            timestamp=timestamp,
        )
        self._save_episodic(episodic)
        self._total_episodic_recorded += 1

        # 立即存储程序性经验摘要（后续异步更新详情）
        entry = MemoryEntry(
            id=memory_id,
            project=project,
            task=task,
            task_type=task_type,
            error_type=error_type,
            difficulty=difficulty,
            summary=summary,
            tags=tags,
            tool_calls_count=len(tool_logs),
            success=success,
            related_files=related_files,
            timestamp=timestamp,
        )
        self._store.save(entry)
        self._total_memories_created += 1

        # ---- 更新用户画像 ----
        self._update_profile_from_task(task, task_type, tags, success)

        self._log(
            f"记忆已存储: {memory_id} (情景: {episodic_id}) | "
            f"类型: {task_type} | 难度: {difficulty} | {'成功' if success else '失败'}"
        )

        # ---- 异步：深度提炼 ----
        def on_detail_ready(detail: dict):
            if detail:
                detail_text = ""
                if detail.get("问题"):
                    detail_text += f"问题: {detail['问题']}\n"
                if detail.get("根因"):
                    detail_text += f"根因: {detail['根因']}\n"
                if detail.get("方案"):
                    detail_text += f"方案: {detail['方案']}\n"
                if detail.get("教训"):
                    detail_text += f"教训: {detail['教训']}"

                entry.detail = detail_text
                files_str = detail.get("涉及文件", "")
                if files_str:
                    detail_files = [f.strip() for f in files_str.split(",") if f.strip()]
                    for f in detail_files:
                        if f not in entry.related_files:
                            entry.related_files.append(f)

                enhanced_tags = self._reflector.extract_tags(task, summary, detail)
                entry.tags = list(dict.fromkeys(entry.tags + enhanced_tags))[:8]

                # 提取适用场景
                if detail.get("教训"):
                    entry.applicable_scenarios = self._extract_scenarios(detail["教训"])

                # 重新保存 + 更新 embedding
                self._store.save(entry)

                # ★ 异步 detail 更新后重建向量索引
                if self._store.chroma_available:
                    try:
                        self._store._chroma.add(entry)
                    except Exception:
                        pass

                self._log(f"记忆详情已更新(含embedding): {memory_id}")

                with self._lock:
                    self._pending_async -= 1

        with self._lock:
            self._pending_async += 1

        self._reflector.generate_detail(
            task=task,
            summary=summary,
            tool_logs=tool_logs,
            callback=on_detail_ready,
        )

    # ============================================================
    # 用户画像
    # ============================================================

    def get_profile_context(self) -> str:
        """获取用户画像的 Prompt 上下文"""
        if self._profile.total_interactions < 3:
            return ""  # 交互太少，画像不可靠
        return self._profile.to_context_text()

    def get_profile(self) -> UserProfile:
        return self._profile

    def update_profile(self, updates: dict):
        """手动更新用户画像"""
        for key, value in updates.items():
            if hasattr(self._profile, key):
                setattr(self._profile, key, value)
        self._save_profile()

    def _update_profile_from_task(self, task: str, task_type: str, tags: list[str], success: bool):
        """从单次任务中增量更新用户画像"""
        self._profile.total_interactions += 1
        if success:
            self._profile.total_tasks_completed += 1

        # 渐进更新技术栈
        tech_tags = {"python", "javascript", "typescript", "go", "rust", "java",
                     "react", "vue", "fastapi", "django", "docker", "redis"}
        for tag in tags:
            if tag in tech_tags and tag not in self._profile.tech_stack:
                self._profile.tech_stack.append(tag)
        if len(self._profile.tech_stack) > 15:
            self._profile.tech_stack = self._profile.tech_stack[-15:]

        # 每 10 次交互保存一次画像
        if self._profile.total_interactions % 10 == 0:
            self._save_profile()

    def _load_profile(self) -> UserProfile:
        """加载用户画像"""
        try:
            if self._profile_path.exists():
                with open(self._profile_path, "r", encoding="utf-8") as f:
                    return UserProfile.from_dict(json.load(f))
        except Exception:
            pass
        return UserProfile()

    def _save_profile(self):
        """持久化用户画像"""
        try:
            self._profile_path.parent.mkdir(parents=True, exist_ok=True)
            self._profile.update_timestamp = datetime.now(timezone.utc).isoformat()
            with open(self._profile_path, "w", encoding="utf-8") as f:
                json.dump(self._profile.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ============================================================
    # 情景记忆
    # ============================================================

    def _save_episodic(self, episodic: EpisodicMemory):
        """保存情景记忆到 JSON 文件"""
        try:
            epi_dir = self._store._data_dir / episodic.project / "episodic"
            epi_dir.mkdir(parents=True, exist_ok=True)
            file_path = epi_dir / f"{episodic.id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(episodic.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _summarize_context_before(self, project: str) -> str:
        """生成执行前状态摘要"""
        recent = self._store.load(project)
        if not recent:
            return "首次执行，无历史上下文"
        last = recent[-1]
        return f"上次任务: {last.task_type} | {'成功' if last.success else '失败'} | 文件: {', '.join(last.related_files[:5])}"

    def _extract_decisions(self, tool_logs: list[ToolCallRecord]) -> list[dict]:
        """从工具日志中提取关键决策点"""
        decisions = []
        step = 0
        for log in tool_logs:
            step += 1
            if log.tool_name in ("write_file", "edit_file", "run_command", "git_commit"):
                decisions.append({
                    "step": step,
                    "decision": log.tool_name,
                    "reason": f"参数: {str(log.params)[:120]}",
                    "outcome": "成功" if log.success else f"失败: {log.error or '未知'}",
                })
        return decisions[:10]

    def _assess_difficulty(self, tool_logs: list[ToolCallRecord], success: bool) -> str:
        """评估任务难度"""
        count = len(tool_logs)
        if count <= 2:
            return "easy"
        elif count <= 8:
            return "medium"
        else:
            return "hard" if not success else "medium"

    def _extract_scenarios(self, lesson_text: str) -> list[str]:
        """从经验教训中提取适用场景"""
        import re
        scenarios = re.findall(r'适用于[：:]\s*(.+?)(?:[；;]|$)', lesson_text)
        if not scenarios:
            scenarios = re.findall(r'场景[：:]\s*(.+?)(?:[；;]|$)', lesson_text)
        return [s.strip() for s in scenarios[:5]]

    # ============================================================
    # Skill 生长反馈
    # ============================================================

    def get_feedback(self, project: str = "", existing_skills: set = None, existing_tags: set = None) -> list[SkillSuggestion]:
        """获取 Skill 优化建议"""
        memories = self._store.load(project) if project else []
        if not memories and project:
            # 加载所有项目记忆
            for proj in self._store.list_projects():
                memories.extend(self._store.load(proj))

        return self._feedback_loop.analyze(
            memories,
            existing_skill_names=existing_skills or set(),
            existing_tags=existing_tags or set(),
        )

    def update_profile_from_feedback(self):
        """基于记忆分析更新用户画像"""
        all_memories = []
        for proj in self._store.list_projects():
            all_memories.extend(self._store.load(proj))
        if not all_memories:
            return
        updates = self._feedback_loop.generate_user_profile_update(
            all_memories, self._profile
        )
        self.update_profile(updates)

    # ============================================================
    # 辅助方法
    # ============================================================

    def _generate_id(self, project: str) -> str:
        """生成唯一的记忆 ID"""
        count = self._total_memories_created + 1
        existing = self._store.load(project)
        count += len(existing)
        return f"mem_{project}_{count:04d}"

    def _extract_files(self, tool_logs: list[ToolCallRecord]) -> list[str]:
        """从工具调用日志中提取涉及的文件路径"""
        files = set()
        for log in tool_logs:
            if log.tool_name in ("read_file", "write_file", "edit_file"):
                path = log.params.get("path", "")
                if path:
                    files.add(path)
            elif log.tool_name == "grep_search":
                path = log.params.get("path", "")
                if path and path != ".":
                    files.add(path)
        return sorted(files)[:10]

    def _log(self, message: str):
        """内部日志输出 —— 写入文件日志，不泄漏到用户终端"""
        _logger.info(f"[MemoryManager] {message}")

    # ============================================================
    # 查询接口（调试/监控）
    # ============================================================

    def get_project_memories(self, project: str) -> list[MemoryEntry]:
        """获取某项目的所有记忆"""
        return self._store.load(project)

    def list_projects(self) -> list[str]:
        """列出有记忆的项目"""
        return self._store.list_projects()

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "enabled": self.enabled,
            "async_reflection": self._async,
            "chroma_available": self._store.chroma_available,
            "chroma_count": self._store.chroma_count(),
            "total_reflections": self._total_reflections,
            "total_memories": self._total_memories_created,
            "total_episodic": self._total_episodic_recorded,
            "pending_async": self._pending_async,
            "profile_interactions": self._profile.total_interactions,
            "profile_tasks_completed": self._profile.total_tasks_completed,
            "projects": self.list_projects(),
        }

    def reset(self, project: str = ""):
        """重置记忆（危险操作）"""
        self._store.reset(project)
        if not project:
            self._total_memories_created = 0
            self._total_reflections = 0
            self._total_episodic_recorded = 0
