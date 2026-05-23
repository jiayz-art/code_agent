"""
代码搜索工具：grep_search

设计思路：
- 优先使用 ripgrep (rg) 命令（速度快、默认尊重 .gitignore）
- 如果没有 rg，回退到 Python 标准库实现
- 自动排除 node_modules、.git 等大目录
- 结果行数限制 500 行，避免撑爆 LLM 上下文
- 返回匹配行及行号，方便 LLM 定位代码
"""

import os
import re
import subprocess
from pathlib import Path

from src.tools.base import BaseTool, ToolResult
from src.safety import SafetyChecker


class GrepSearchTool(BaseTool):
    name = "grep_search"
    description = (
        "在代码库中搜索匹配正则表达式的内容。"
        "返回匹配的文件路径、行号和内容。"
        "支持正则表达式语法。结果最多返回 500 行匹配。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "搜索的正则表达式模式"
            },
            "path": {
                "type": "string",
                "description": "搜索目录（相对于工作区），默认为整个工作区"
            },
            "glob": {
                "type": "string",
                "description": "文件名过滤 glob，如 '*.py' 或 '*.{js,ts}'"
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "是否区分大小写，默认不区分"
            },
        },
        "required": ["pattern"],
    }

    def execute(
        self,
        pattern: str,
        path: str = ".",
        glob: str = "",
        case_sensitive: bool = False,
    ) -> ToolResult:
        # 安全校验搜索路径
        valid, resolved_path = self.safety.validate_path(path)
        if not valid:
            return ToolResult(success=False, content=valid, error=resolved_path)

        # 优先尝试 ripgrep
        try:
            return self._search_with_rg(pattern, resolved_path, glob, case_sensitive)
        except (FileNotFoundError, Exception):
            # rg 不可用时回退到 Python 实现
            return self._search_with_python(pattern, resolved_path, glob, case_sensitive)

    def _search_with_rg(
        self, pattern: str, path: str, glob: str, case_sensitive: bool
    ) -> ToolResult:
        """使用 ripgrep 搜索"""
        args = ["rg", "--no-heading", "--with-filename", "--line-number"]

        if not case_sensitive:
            args.append("--ignore-case")

        if glob:
            args.extend(["--glob", glob])

        # 添加排除目录
        for d in self.safety.get_exclude_dirs():
            args.extend(["--glob", f"!{d}/**"])

        args.extend([pattern, path])

        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                content="搜索超时（30秒），请缩小搜索范围或使用更精确的正则。",
                error="TimeoutExpired",
            )

        # rg: exit 0 = 有匹配, exit 1 = 无匹配, exit 2 = 错误
        if result.returncode == 2:
            return ToolResult(
                success=False,
                content=f"搜索失败: {result.stderr}",
                error=result.stderr,
            )

        output = result.stdout
        if not output.strip():
            return ToolResult(
                success=True,
                content=f"未找到匹配 '{pattern}' 的内容。",
                metadata={"matches": 0},
            )

        lines = output.strip().split("\n")
        truncated = len(lines) > 500
        if truncated:
            lines = lines[:500]

        return ToolResult(
            success=True,
            content="\n".join(lines) + ("\n... (结果已截断至 500 行)" if truncated else ""),
            metadata={
                "matches": len(lines),
                "truncated": truncated,
                "search_path": path,
                "using": "ripgrep",
            },
        )

    def _search_with_python(
        self, pattern: str, path: str, glob: str, case_sensitive: bool
    ) -> ToolResult:
        """使用 Python 标准库回退搜索"""
        try:
            regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
        except re.error as e:
            return ToolResult(
                success=False,
                content=f"无效的正则表达式: {e}",
                error=str(e),
            )

        # 解析 glob 过滤
        if glob:
            glob_regex = re.compile(
                glob.replace(".", r"\.").replace("*", ".*").replace("{", "(")
                .replace("}", ")").replace(",", "|")
            )
        else:
            glob_regex = None

        exclude_dirs = self.safety.get_exclude_dirs()
        matches: list[str] = []
        search_root = Path(path)

        for root, dirs, files in os.walk(search_root):
            # 跳过排除的目录
            dirs[:] = [d for d in dirs if not self.safety.should_exclude_dir(d)]

            for file in files:
                if glob_regex and not glob_regex.fullmatch(file):
                    continue

                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        for line_no, line in enumerate(f, 1):
                            if regex.search(line):
                                # 格式对齐 rg 输出: file:line:content
                                rel_path = file_path.relative_to(search_root.parent)
                                matches.append(f"{rel_path}:{line_no}:{line.rstrip()}")
                                if len(matches) >= 500:
                                    break
                except Exception:
                    continue

                if len(matches) >= 500:
                    break
            if len(matches) >= 500:
                break

        if not matches:
            return ToolResult(
                success=True,
                content=f"未找到匹配 '{pattern}' 的内容。",
                metadata={"matches": 0},
            )

        truncated = len(matches) >= 500
        return ToolResult(
            success=True,
            content="\n".join(matches) + ("\n... (结果已截断至 500 行)" if truncated else ""),
            metadata={
                "matches": len(matches),
                "truncated": truncated,
                "search_path": path,
                "using": "python",
            },
        )
