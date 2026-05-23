"""
记忆持久化层 —— JSON（完整记忆）+ ChromaDB（向量索引）

设计思路：
- 三层写入：① JSON 文件存储完整记忆 ② index.json 维护轻量索引 ③ ChromaDB 维护向量
- 按项目隔离：data/memories/{project}/ 下存放该项目的所有记忆
- index.json 启动时加载到内存，避免每次遍历文件系统
- ChromaDB 用于语义检索，自动生成 embedding
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from src.memory.models import MemoryEntry

_logger = logging.getLogger("agent.memory.store")


# ============================================================
# ChromaDB 封装（可选依赖）
# ============================================================

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


class ChromaBackend:
    """ChromaDB 向量存储后端 —— 可选依赖，未安装时降级为纯文件模式"""

    COLLECTION_NAME = "agent_memories"

    def __init__(self, persist_path: str = "data/chroma"):
        self._available = _CHROMA_AVAILABLE
        self._client = None
        self._collection = None

        if self._available:
            try:
                os.makedirs(persist_path, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=persist_path,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                # 获取或创建 collection
                try:
                    self._collection = self._client.get_collection(self.COLLECTION_NAME)
                except Exception:
                    self._collection = self._client.create_collection(
                        name=self.COLLECTION_NAME,
                        metadata={"description": "AI Coding Agent 记忆向量索引"},
                    )
            except Exception:
                self._available = False
                self._client = None
                self._collection = None

    @property
    def is_available(self) -> bool:
        return self._available and self._collection is not None

    def add(self, memory: MemoryEntry):
        """将记忆加入向量索引"""
        if not self.is_available:
            return

        text = f"{memory.task} {memory.summary} {memory.detail[:500]}"
        try:
            self._collection.add(
                ids=[memory.id],
                documents=[text],
                metadatas=[{
                    "project": memory.project,
                    "task_type": memory.task_type,
                    "error_type": memory.error_type,
                    "success": memory.success,
                    "tags": ",".join(memory.tags),
                    "timestamp": memory.timestamp,
                }],
            )
            memory.embedding_id = memory.id
        except Exception:
            pass  # ChromaDB 写入失败不影响 JSON 持久化

    def query(
        self, query_text: str, project: str = "", top_k: int = 10
    ) -> list[tuple[str, float, dict]]:
        """
        语义搜索。

        Returns:
            [(memory_id, distance_score, metadata), ...] 按相似度降序
        """
        if not self.is_available:
            return []

        try:
            where_filter = None
            if project:
                where_filter = {"project": project}

            results = self._collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=where_filter,
                include=["metadatas", "distances"],
            )

            if not results["ids"] or not results["ids"][0]:
                return []

            output = []
            for i, mem_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 1.0
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                score = 1.0 / (1.0 + distance)  # distance → similarity
                output.append((mem_id, score, metadata))
            return output
        except Exception:
            return []

    def delete(self, memory_id: str):
        """从向量索引中删除"""
        if not self.is_available:
            return
        try:
            self._collection.delete(ids=[memory_id])
        except Exception:
            pass

    def count(self) -> int:
        """向量索引中的记忆数量"""
        if not self.is_available:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0


# ============================================================
# MemoryStore
# ============================================================

class MemoryStore:
    """
    记忆持久化管理器。

    三路存储：
    1. data/memories/{project}/mem_{id}.json — 完整记忆体
    2. data/memories/{project}/index.json — 轻量索引
    3. ChromaDB — 向量检索
    """

    def __init__(self, data_dir: str = "data/memories", chroma_path: str = "data/chroma"):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._chroma = ChromaBackend(chroma_path)
        # 内存中的索引缓存: project → list of entry summaries
        self._index_cache: dict[str, dict] = {}

    # ============================================================
    # 写入
    # ============================================================

    def save(self, entry: MemoryEntry) -> bool:
        """
        三路写入一条记忆。

        三路任一路失败不影响其他路；至少一路成功即返回 True。
        """
        success = False

        # 1. JSON 文件
        project_dir = self._data_dir / entry.project
        project_dir.mkdir(parents=True, exist_ok=True)
        try:
            file_path = project_dir / f"{entry.id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)
            success = True
        except Exception as e:
            _logger.error(f"JSON 写入失败: {e}")

        # 2. index.json
        try:
            self._update_index(entry)
        except Exception as e:
            _logger.error(f"Index 更新失败: {e}")

        # 3. ChromaDB
        self._chroma.add(entry)

        return success

    def _update_index(self, entry: MemoryEntry):
        """更新项目的 index.json"""
        index_path = self._data_dir / entry.project / "index.json"

        index = {"project": entry.project, "last_updated": entry.timestamp, "total_memories": 0, "entries": []}

        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    index["entries"] = existing.get("entries", [])
            except Exception:
                pass

        # 检查是否已存在同 ID 记录（更新 or 追加）
        found = False
        for i, item in enumerate(index["entries"]):
            if item.get("id") == entry.id:
                index["entries"][i] = self._entry_summary(entry)
                found = True
                break
        if not found:
            index["entries"].append(self._entry_summary(entry))

        index["total_memories"] = len(index["entries"])
        index["last_updated"] = entry.timestamp

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

        # 更新内存缓存
        self._index_cache[entry.project] = index

    def _entry_summary(self, entry: MemoryEntry) -> dict:
        """生成索引条目摘要"""
        return {
            "id": entry.id,
            "task": entry.task[:80],
            "task_type": entry.task_type,
            "error_type": entry.error_type,
            "tags": entry.tags,
            "success": entry.success,
            "timestamp": entry.timestamp,
            "embedding_id": entry.embedding_id,
        }

    # ============================================================
    # 读取
    # ============================================================

    def load(self, project: str) -> list[MemoryEntry]:
        """加载某项目的所有完整记忆"""
        project_dir = self._data_dir / project
        if not project_dir.exists():
            return []

        entries = []
        for file_path in sorted(project_dir.glob("mem_*.json")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    entries.append(MemoryEntry.from_dict(data))
            except Exception:
                continue
        return entries

    def get_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """按 ID 加载单条记忆"""
        # 从 ID 中解析 project（格式: mem_{project}_{序号}）
        parts = memory_id.split("_", 2)
        if len(parts) >= 3:
            project = parts[1]
            file_path = self._data_dir / project / f"{memory_id}.json"
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        return MemoryEntry.from_dict(json.load(f))
                except Exception:
                    pass
        return None

    def load_index(self, project: str) -> dict:
        """加载项目索引文件（轻量，适合检索时使用）"""
        if project in self._index_cache:
            return self._index_cache[project]

        index_path = self._data_dir / project / "index.json"
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
                    self._index_cache[project] = index
                    return index
            except Exception:
                pass
        return {"project": project, "total_memories": 0, "entries": []}

    def list_projects(self) -> list[str]:
        """列出所有有记忆的项目目录"""
        if not self._data_dir.exists():
            return []
        return [
            d.name for d in self._data_dir.iterdir()
            if d.is_dir() and (d / "index.json").exists()
        ]

    # ============================================================
    # 删除
    # ============================================================

    def delete(self, memory_id: str):
        """删除一条记忆（三路同步删除）"""
        # JSON 文件
        for project_dir in self._data_dir.iterdir():
            if not project_dir.is_dir():
                continue
            file_path = project_dir / f"{memory_id}.json"
            if file_path.exists():
                file_path.unlink()

            # Index 更新
            index_path = project_dir / "index.json"
            if index_path.exists():
                try:
                    with open(index_path, "r", encoding="utf-8") as f:
                        index = json.load(f)
                    index["entries"] = [
                        e for e in index.get("entries", []) if e.get("id") != memory_id
                    ]
                    index["total_memories"] = len(index["entries"])
                    with open(index_path, "w", encoding="utf-8") as f:
                        json.dump(index, f, ensure_ascii=False, indent=2)
                    project_name = project_dir.name
                    self._index_cache[project_name] = index
                except Exception:
                    pass
            break

        # ChromaDB
        self._chroma.delete(memory_id)

    # ============================================================
    # 向量检索代理
    # ============================================================

    def search_similar(
        self, query_text: str, project: str = "", top_k: int = 10
    ) -> list[tuple[str, float]]:
        """语义搜索代理 → [(memory_id, score), ...]"""
        results = self._chroma.query(query_text, project=project, top_k=top_k)
        return [(rid, score) for rid, score, _ in results]

    @property
    def chroma_available(self) -> bool:
        return self._chroma.is_available

    def chroma_count(self) -> int:
        return self._chroma.count()

    # ============================================================
    # 备用
    # ============================================================

    def reset(self, project: str = ""):
        """重置某项目或全部记忆（危险操作）"""
        if project:
            project_dir = self._data_dir / project
            if project_dir.exists():
                shutil.rmtree(project_dir)
            self._index_cache.pop(project, None)
        else:
            if self._data_dir.exists():
                shutil.rmtree(self._data_dir)
            self._index_cache.clear()
            self._data_dir.mkdir(parents=True, exist_ok=True)
