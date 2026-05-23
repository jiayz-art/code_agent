"""
安全校验模块

设计思路：
- 两层防护：文件操作做路径白名单校验，Shell 命令做黑名单正则匹配
- 路径校验的核心是解析符号链接和相对路径后，确认最终路径在工作区范围内
- 所有校验方法返回 (allowed: bool, reason: str)，不抛异常，让调用方决策
- 安全策略集中在配置文件中，运维可调整不需要改代码
- 与 security/ 审查链路共享规则：命令模式委托给 RuleFilter 以消除重复
"""

import os
import re
from pathlib import Path
from typing import Optional

from src.config import AppConfig

# 共享的危险命令模式（security/rule_filter.py 也引用此处，消除重复）
DANGEROUS_PATTERNS: list[re.Pattern] = [
    re.compile(r"\brm\s+(-\w*r\w*f?\w*|--recursive)\s+", re.IGNORECASE),
    re.compile(r"\bdel\s+/[fsq]", re.IGNORECASE),
    re.compile(r"\bmkfs\.", re.IGNORECASE),
    re.compile(r"\bdd\s+if=", re.IGNORECASE),
    re.compile(r">\s*/dev/sd", re.IGNORECASE),
    re.compile(r"\bformat\s+\w:", re.IGNORECASE),
    re.compile(r"\b(shutdown|reboot|halt|poweroff)\b", re.IGNORECASE),
    re.compile(r"\bchmod\s+777\b", re.IGNORECASE),
    re.compile(r"\bchown\s+-R\s+", re.IGNORECASE),
    re.compile(r"\bcurl.*\|\s*(ba)?sh\b", re.IGNORECASE),
    re.compile(r"\bwget.*\|\s*(ba)?sh\b", re.IGNORECASE),
    re.compile(r":\(\)\s*\{", re.IGNORECASE),
    re.compile(r"\bsudo\s+su\b", re.IGNORECASE),
]

COMMAND_BLACKLIST = {
    "rm -rf /", "rm -rf ~/", "rm -rf --no-preserve-root",
    "dd if=/dev/zero", "mkfs.ext", "mkfs.ntfs", "mkfs.fat",
    ":(){ :|:& };:", "> /dev/sda", "> /dev/null",
}


class SafetyChecker:
    """
    统一安全校验入口。

    初始化时从配置加载策略，后续所有校验都通过此类完成。
    """

    def __init__(self, config: AppConfig):
        self._workspace = Path(config.agent.workspace_root).resolve()
        self._max_file_size = config.tools.file.max_file_size_mb * 1024 * 1024
        self._shell_timeout = config.tools.shell.timeout_seconds
        self._protected_branches = config.tools.git.protected_branches

        # 编译危险命令正则（优先使用配置中的自定义模式，否则使用共享默认模式）
        self._dangerous_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in config.tools.shell.dangerous_patterns
        ] if config.tools.shell.dangerous_patterns else DANGEROUS_PATTERNS

        # 搜索时排除的目录
        self._exclude_dirs = set(config.tools.search.exclude_dirs)

    # ============================================================
    # 路径安全校验
    # ============================================================

    def validate_path(self, file_path: str, must_exist: bool = False) -> tuple[bool, str]:
        """
        校验文件路径是否安全。

        规则：
        1. 解析符号链接和相对路径，得到真实绝对路径
        2. 确认路径在工作区根目录下（防止 ../ 逃逸）
        3. 可选：检查文件是否存在
        """
        try:
            # resolve() 会处理 ..、符号链接，得到规范化的绝对路径
            resolved = (self._workspace / file_path).resolve()
        except (OSError, ValueError) as e:
            return False, f"路径解析失败: {e}"

        # 核心检查：必须在工作区内
        try:
            resolved.relative_to(self._workspace)
        except ValueError:
            return False, (
                f"路径越界：'{file_path}' 不在工作区 '{self._workspace}' 内。\n"
                f"提示：请将目标路径改为工作区内的相对路径，"
                f"例如将文件写入 '{self._workspace.as_posix()}/<文件名>'。"
            )

        if must_exist and not resolved.exists():
            return False, f"文件不存在: {resolved}"

        return True, resolved.as_posix()

    def validate_file_size(self, file_path: str) -> tuple[bool, str]:
        """检查文件大小是否在限额内"""
        path = Path(file_path)
        if not path.is_file():
            return True, ""  # 不存在不报错，由读文件工具自己报
        size = path.stat().st_size
        if size > self._max_file_size:
            return False, (
                f"文件过大: {size / 1024 / 1024:.1f}MB "
                f"(上限 {self._max_file_size / 1024 / 1024:.0f}MB)"
            )
        return True, ""

    # ============================================================
    # Shell 命令安全校验
    # ============================================================

    def validate_command(self, command: str) -> tuple[bool, str]:
        """
        校验 Shell 命令是否安全。

        采用黑名单模式——匹配到危险模式即拒绝。
        """
        for pattern in self._dangerous_patterns:
            if pattern.search(command):
                return False, f"危险命令被拦截: 匹配模式 '{pattern.pattern}'"
        return True, ""

    def get_shell_timeout(self) -> int:
        return self._shell_timeout

    # ============================================================
    # Git 操作安全校验
    # ============================================================

    def validate_git_branch(self, branch: str) -> tuple[bool, str]:
        """检查是否对受保护分支执行危险操作"""
        for protected in self._protected_branches:
            # 支持通配符，如 "release/*"
            if protected.endswith("/*"):
                prefix = protected[:-1]
                if branch.startswith(prefix):
                    return False, f"禁止操作受保护分支 '{branch}' (规则: {protected})"
            elif branch == protected:
                return False, f"禁止操作受保护分支 '{branch}'"
        return True, ""

    # ============================================================
    # 搜索安全校验
    # ============================================================

    def should_exclude_dir(self, dir_name: str) -> bool:
        """判断目录是否应该被搜索排除"""
        return dir_name in self._exclude_dirs

    def get_exclude_dirs(self) -> set[str]:
        return self._exclude_dirs.copy()
