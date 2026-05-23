"""
文件操作工具集：read_file / write_file / edit_file

设计思路：
- 每个工具独立成一个类，职责单一
- read_file 支持片段读取（offset + limit），避免大文件撑爆上下文
- write_file 自动创建父目录，无需用户手动 mkdir
- edit_file 采用精确字符串替换（old_string → new_string），类似 sed 但更安全
- 所有操作前都通过 SafetyChecker 做路径白名单校验
"""

import os
from pathlib import Path

from src.tools.base import BaseTool, ToolResult
from src.safety import SafetyChecker


class ReadFileTool(BaseTool):
    name = "read_file"
    description = (
        "读取文件内容。可以读取整个文件，或通过 offset/limit 参数读取部分行。"
        "读取大文件时建议指定 offset 和 limit，避免返回过多内容。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径（相对于工作区根目录）"
            },
            "offset": {
                "type": "integer",
                "description": "起始行号（从 1 开始），不指定则从头读取"
            },
            "limit": {
                "type": "integer",
                "description": "读取行数，不指定则读取全部"
            },
        },
        "required": ["path"],
    }

    def execute(self, path: str, offset: int = 0, limit: int = -1) -> ToolResult:
        # 1. 安全校验：路径必须在工作区内
        valid, resolved = self.safety.validate_path(path, must_exist=True)
        if not valid:
            return ToolResult(success=False, content=valid, error=resolved)

        # 2. 大小校验
        size_ok, size_msg = self.safety.validate_file_size(resolved)
        if not size_ok:
            return ToolResult(success=False, content=size_msg, error=size_msg)

        file_path = Path(resolved)
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)

            # 处理读取范围
            if offset > 0:
                start = offset - 1  # 用户输入从 1 开始，内部用 0-based
            else:
                start = 0

            if limit > 0:
                end = min(start + limit, total_lines)
            elif limit == -1:
                end = total_lines
            else:
                end = total_lines

            if start >= total_lines:
                return ToolResult(
                    success=False,
                    content=f"offset ({offset}) 超出文件行数 ({total_lines})",
                )

            selected = lines[start:end]
            # 添加行号前缀（类似 cat -n），方便 LLM 后续引用行号
            numbered = "".join(
                f"{i + start + 1:6}\t{line}"
                for i, line in enumerate(selected)
            )

            return ToolResult(
                success=True,
                content=numbered,
                metadata={
                    "file": file_path.as_posix(),
                    "total_lines": total_lines,
                    "shown_lines": len(selected),
                    "start_line": start + 1,
                    "end_line": start + len(selected),
                }
            )
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                content=f"文件不是有效的 UTF-8 文本: {path}",
                error="UnicodeDecodeError",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content=f"读取文件失败: {e}",
                error=str(e),
            )


class WriteFileTool(BaseTool):
    name = "write_file"
    description = (
        "创建新文件或覆盖已有文件。"
        "会自动创建不存在的父目录。"
        "此操作会覆盖文件的全部内容，请谨慎使用。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径（相对于工作区根目录）"
            },
            "content": {
                "type": "string",
                "description": "要写入的完整内容"
            },
        },
        "required": ["path", "content"],
    }

    def execute(self, path: str, content: str) -> ToolResult:
        # 安全校验
        valid, resolved = self.safety.validate_path(path)
        if not valid:
            return ToolResult(success=False, content=valid, error=resolved)

        file_path = Path(resolved)
        try:
            # 自动创建父目录
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # 判断是创建还是覆盖
            is_new = not file_path.exists()

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            file_size = file_path.stat().st_size
            action = "创建" if is_new else "覆盖写入"

            # ★ 写入后验证：读回文件前几行确认写入成功
            verify_info = ""
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    head = f.read(200)
                line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
                verify_info = (
                    f"\n[写入验证] 文件确认存在，大小 {file_size} 字节，"
                    f"约 {line_count} 行。开头预览:\n{head[:150]}"
                )
            except Exception:
                verify_info = "\n[写入验证] 文件写入后无法读取验证，请手动确认。"

            return ToolResult(
                success=True,
                content=f"文件{action}成功: {file_path.as_posix()} ({file_size} 字节){verify_info}",
                metadata={
                    "file": file_path.as_posix(),
                    "is_new": is_new,
                    "size_bytes": file_size,
                    "lines": content.count("\n") + (1 if content and not content.endswith("\n") else 0),
                }
            )
        except PermissionError:
            return ToolResult(
                success=False,
                content=f"权限不足，无法写入: {path}",
                error="PermissionError",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content=f"写入文件失败: {e}",
                error=str(e),
            )


class EditFileTool(BaseTool):
    name = "edit_file"
    description = (
        "对文件执行精确的字符串替换。"
        "找到 old_string 的第一次出现，替换为 new_string。"
        "old_string 必须精确匹配（包括空格、缩进、换行），否则操作失败。"
        "这是比 write_file 更安全的修改方式，只改需要改的部分。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径（相对于工作区根目录）"
            },
            "old_string": {
                "type": "string",
                "description": "要被替换的原始文本（必须精确匹配）"
            },
            "new_string": {
                "type": "string",
                "description": "替换后的新文本"
            },
        },
        "required": ["path", "old_string", "new_string"],
    }

    def _validate_params(self, kwargs: dict) -> tuple[bool, str]:
        if not kwargs.get("old_string"):
            return False, "old_string 不能为空"
        if kwargs["old_string"] == kwargs.get("new_string"):
            return False, "old_string 和 new_string 相同，无需修改"
        return True, ""

    def execute(self, path: str, old_string: str, new_string: str) -> ToolResult:
        # 安全校验
        valid, resolved = self.safety.validate_path(path, must_exist=True)
        if not valid:
            return ToolResult(success=False, content=valid, error=resolved)

        file_path = Path(resolved)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                original = f.read()

            # 精确匹配
            if old_string not in original:
                return ToolResult(
                    success=False,
                    content=(
                        f"未找到匹配的 old_string。\n"
                        f"文件: {file_path.as_posix()}\n"
                        f"提示: 请确保 old_string 与文件内容完全一致，包括缩进和换行。"
                    ),
                    error="old_string_not_found",
                )

            # 只替换第一次出现
            modified = original.replace(old_string, new_string, 1)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(modified)

            # ★ 编辑后验证：确认替换确实生效
            with open(file_path, "r", encoding="utf-8") as f:
                verified = f.read()
            if new_string in verified:
                verify_msg = "[编辑验证] 已确认替换生效。"
            else:
                verify_msg = "[编辑验证] 警告：替换后文件中未找到新内容，请人工确认。"

            return ToolResult(
                success=True,
                content=(
                    f"文件编辑成功: {file_path.as_posix()}\n"
                    f"替换了 1 处匹配。{verify_msg}"
                ),
                metadata={
                    "file": file_path.as_posix(),
                    "old_length": len(old_string),
                    "new_length": len(new_string),
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content=f"编辑文件失败: {e}",
                error=str(e),
            )
