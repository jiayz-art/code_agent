"""
Glob 文件模式匹配工具

对标 Claude Code 的 Glob 工具，支持 **/*.py 等 glob 模式匹配。
"""

import os
from pathlib import Path

from src.tools.base import BaseTool, ToolResult


class GlobTool(BaseTool):
    """按 glob 模式匹配文件路径，返回排序后的文件列表"""

    name = "glob"
    description = (
        "按 glob 模式匹配文件路径（如 **/*.py 匹配所有 Python 文件）。"
        "返回排序后的匹配文件路径列表。适用于查找特定类型的文件、探索项目结构。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "glob 模式，如 **/*.py、src/**/*.ts、**/*_test.py",
            },
            "path": {
                "type": "string",
                "description": "搜索起始目录，默认为工作区根目录",
            },
        },
        "required": ["pattern"],
    }

    def execute(self, pattern: str, path: str = ".") -> ToolResult:
        resolved = (Path(path)).resolve()
        if not resolved.exists():
            return ToolResult(
                success=False,
                content=f"目录不存在: {path}",
                error=f"Path not found: {path}",
            )

        try:
            matches = sorted(
                str(p.relative_to(resolved))
                for p in resolved.glob(pattern)
                if not self._is_excluded(p)
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content=f"Glob 模式执行失败: {e}",
                error=str(e),
            )

        if not matches:
            return ToolResult(
                success=True,
                content=f"未找到匹配 '{pattern}' 的文件",
            )

        return ToolResult(
            success=True,
            content=f"匹配 '{pattern}' 的 {len(matches)} 个文件:\n" +
                    "\n".join(matches[:200]),  # 限制输出行数
            metadata={"count": len(matches), "pattern": pattern},
        )

    def _is_excluded(self, path: Path) -> bool:
        """排除常见无关目录"""
        excluded_dirs = {
            ".git", "node_modules", "__pycache__", ".venv", "venv",
            "dist", "build", ".next", ".cache", "coverage",
        }
        for part in path.parts:
            if part in excluded_dirs:
                return True
        return False
