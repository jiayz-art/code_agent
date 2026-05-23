"""
配置加载模块

设计思路：
- YAML 配置文件 + 环境变量注入，避免敏感信息进配置文件
- 使用 dataclass 强类型化，IDE 友好，避免字典 key 拼写错误
- 配置加载即校验，失败即抛，不要在运行时发现配置错误
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ============================================================
# 配置数据类定义 —— 一个子类对应配置文件中的一个 section
# ============================================================

@dataclass
class LLMConfig:
    """LLM 模型配置"""
    api_key: str
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-coder-plus"
    temperature: float = 0.1
    max_tokens: int = 8192


@dataclass
class AgentConfig:
    """Agent 行为配置"""
    max_iterations: int = 30
    workspace_root: str = "."


@dataclass
class FileToolConfig:
    """文件工具配置"""
    enabled: bool = True
    max_file_size_mb: int = 5


@dataclass
class ShellToolConfig:
    """Shell 工具配置"""
    enabled: bool = True
    timeout_seconds: int = 60
    dangerous_patterns: list[str] = field(default_factory=lambda: [
        r"rm\s+(-rf?\s+)?/",
        r"shutdown",
        r"reboot",
        r"mkfs\.",
        r"dd\s+if=",
        r">\s*/dev/sd",
    ])


@dataclass
class GitToolConfig:
    """Git 工具配置"""
    enabled: bool = True
    protected_branches: list[str] = field(default_factory=lambda: ["main", "master"])


@dataclass
class SearchToolConfig:
    """搜索工具配置"""
    enabled: bool = True
    exclude_dirs: list[str] = field(default_factory=lambda: [
        ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"
    ])


@dataclass
class ToolsConfig:
    """工具总配置"""
    file: FileToolConfig = field(default_factory=FileToolConfig)
    shell: ShellToolConfig = field(default_factory=ShellToolConfig)
    git: GitToolConfig = field(default_factory=GitToolConfig)
    search: SearchToolConfig = field(default_factory=SearchToolConfig)


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    file: str = "agent.log"
    format: str = "json"
    console: bool = True


@dataclass
class MemoryConfig:
    """记忆系统配置"""
    enabled: bool = False
    max_context_tokens: int = 800
    data_dir: str = "data/memories"
    async_reflection: bool = True


@dataclass
class PlaceholderConfig:
    """占位替换引擎配置"""
    file_threshold_chars: int = 3000
    output_threshold_chars: int = 2000
    keep_head_lines: int = 30
    keep_head_chars: int = 500


@dataclass
class NoteConfig:
    """笔记生成器配置"""
    enabled: bool = True
    use_llm_fallback: bool = False


@dataclass
class SummaryConfig:
    """超限压缩摘要配置"""
    max_chars: int = 300
    enabled: bool = True


@dataclass
class ContextConfig:
    """分层上下文压缩配置"""
    enabled: bool = True
    max_tokens: int = 9000
    keep_recent: int = 8
    placeholder: PlaceholderConfig = field(default_factory=PlaceholderConfig)
    note: NoteConfig = field(default_factory=NoteConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)


@dataclass
class MultiAgentConfig:
    """多Agent协作配置"""
    enabled: bool = True
    parallel_workers: int = 4
    sub_agent_max_iterations: int = 15
    worktree_enabled: bool = True


@dataclass
class SecurityConfig:
    """安全审查配置"""
    enabled: bool = True
    tier: str = "standard"          # relaxed | standard | strict
    enable_ai_classifier: bool = True
    enable_human_gate: bool = True
    confirm_timeout: float = 120.0
    auto_confirm: bool = False       # CI/自动化模式跳过人工确认
    audit_log: str = "data/audit.log"
    # 可自定义敏感文件模式（追加到默认列表）
    extra_sensitive_patterns: list[str] = field(default_factory=list)
    # 可完全阻止的工具（即使在relaxed模式也阻止）
    blocked_tools: list[str] = field(default_factory=list)


@dataclass
class AppConfig:
    """应用总配置 —— 顶层聚合所有子配置"""
    llm: LLMConfig
    agent: AgentConfig = field(default_factory=AgentConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    multi_agent: MultiAgentConfig = field(default_factory=MultiAgentConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)


# ============================================================
# 配置加载逻辑
# ============================================================

# 匹配 ${VAR_NAME} 或 ${VAR_NAME:default_value} 模式
_ENV_VAR_PATTERN = re.compile(r'\$\{(\w+)(?::([^}]*))?\}')


def _resolve_env_vars(value: str) -> str:
    """将字符串中的 ${VAR} 替换为环境变量值"""
    def _replacer(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2)
        env_val = os.environ.get(var_name)
        if env_val is not None:
            return env_val
        if default is not None:
            return default
        raise KeyError(
            f"环境变量 {var_name} 未设置，且配置中未提供默认值。"
            f"请设置环境变量: export {var_name}=<your_value>"
        )
    return _ENV_VAR_PATTERN.sub(_replacer, value)


def _resolve_dict(obj: dict) -> dict:
    """递归解析字典中所有字符串值的环境变量引用"""
    result = {}
    for key, value in obj.items():
        if isinstance(value, str):
            result[key] = _resolve_env_vars(value)
        elif isinstance(value, dict):
            result[key] = _resolve_dict(value)
        elif isinstance(value, list):
            result[key] = [
                _resolve_env_vars(v) if isinstance(v, str) else v
                for v in value
            ]
        else:
            result[key] = value
    return result


def _build_dataclass(cls, data: dict):
    """递归将字典构建为嵌套 dataclass 实例"""
    import dataclasses
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in field_types:
            continue  # 忽略配置文件中多余的 key
        target_type = field_types[key]
        # 处理嵌套 dataclass（通过 __dataclass_fields__ 判断）
        if hasattr(target_type, '__dataclass_fields__') and isinstance(value, dict):
            kwargs[key] = _build_dataclass(target_type, value)
        elif (hasattr(target_type, '__origin__') and
              target_type.__origin__ is list and
              hasattr(target_type.__args__[0], '__dataclass_fields__') and
              isinstance(value, list)):
            # list[SomeDataclass] 类型，当前配置中无此场景，保留扩展点
            kwargs[key] = [_build_dataclass(target_type.__args__[0], item) for item in value]
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(config_path: str | Path = "config.yaml") -> AppConfig:
    """
    加载并返回完整配置。

    流程：读 YAML → 环境变量替换 → 构建强类型 dataclass
    任何步骤失败都会抛出明确错误，不在运行时默默降级。
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("配置文件格式错误：顶层必须是字典")

    # 环境变量替换
    raw = _resolve_dict(raw)

    # 构建强类型配置
    return _build_dataclass(AppConfig, raw)
