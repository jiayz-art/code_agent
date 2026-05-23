"""
AI Coding Agent — 入口模块

用法:
    python main.py                    # 交互式 REPL 模式
    python main.py --task "任务描述"  # 单次任务模式
    python main.py --config my.yaml   # 指定配置文件

设计思路：
- REPL 模式用 rich 库美化终端输出，清晰展示工具调用和结果
- 单次任务模式支持脚本化和 CI 集成
- 所有异常都在最外层捕获，保证程序不会意外崩溃
"""

import argparse
import sys
import os
from pathlib import Path

# 将项目根目录加入 sys.path，确保 src 包可导入
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.live import Live
from rich import box

from src.config import load_config, AppConfig
from src.logger import AgentLogger
from src.safety import SafetyChecker
from src.tools.base import ToolRegistry
from src.tools.file_tools import ReadFileTool, WriteFileTool, EditFileTool
from src.tools.shell_tools import RunCommandTool
from src.tools.git_tools import GitStatusTool, GitDiffTool, GitCommitTool
from src.tools.search_tools import GrepSearchTool
from src.tools.glob_tool import GlobTool
from src.tools.ask_tool import AskUserQuestionTool
from src.tools.todo_tool import TodoWriteTool
from src.agent import Agent
from src.output_filter import UserOutput
from src.conversation_memory import ConversationMemory

# Skill 路由系统
from src.skills.catalog import SkillCatalog
from src.skills.index import SkillIndex
from src.skills.ranker import SkillRanker
from src.skills.router import SkillRouter
from src.llm import LLMClient

# 记忆系统
from src.memory.reflector import MemoryReflector
from src.memory.store import MemoryStore
from src.memory.retriever import MemoryRetriever
from src.memory.manager import MemoryManager

# 上下文压缩系统
from src.context.manager import ContextManager

# 多Agent协作系统
from src.multi_agent.orchestrator import Orchestrator
from src.multi_agent.delegate_tool import DelegateAgentTool

# 权限与安全审查系统
from src.security.types import SecurityPolicy, SecurityTier
from src.security.audit_chain import AuditChain
from src.security.audit_logger import AuditLogger


# ============================================================
# 终端美化
# ============================================================

console = Console()


def banner():
    """打印欢迎横幅"""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]AI Coding Agent[/bold cyan] [dim]v0.1.0[/dim]\n"
            "[dim]基于 Qwen-Coder-Plus | 输入 /help 查看帮助 | Ctrl+C 退出[/dim]",
            border_style="cyan",
            padding=(1, 4),
        )
    )


def help_text():
    """帮助信息"""
    table = Table(title="可用命令", box=box.SIMPLE)
    table.add_column("命令", style="cyan", no_wrap=True)
    table.add_column("说明", style="white")
    table.add_row("/help", "显示此帮助")
    table.add_row("/clear", "清空对话历史（重新开始）")
    table.add_row("/stats", "显示 Token 消耗统计")
    table.add_row("/exit, /quit", "退出程序")
    table.add_row("<任意文本>", "向 Agent 发送任务")
    console.print(table)


# ============================================================
# 初始化
# ============================================================

def setup_agent(config: AppConfig) -> tuple[Agent, AgentLogger, SafetyChecker]:
    """创建 Agent 实例，注册所有工具"""
    # 日志
    logger = AgentLogger.from_config(config.logging)

    # 安全校验器
    safety = SafetyChecker(config)

    # ---- 安全审查链路初始化 ----
    audit_chain = None
    audit_logger = None
    try:
        sec_cfg = getattr(config, "security", None)
        if sec_cfg and getattr(sec_cfg, "enabled", True):
            # 构建安全策略
            tier_map = {
                "relaxed": SecurityTier.RELAXED,
                "standard": SecurityTier.STANDARD,
                "strict": SecurityTier.STRICT,
            }
            tier = tier_map.get(getattr(sec_cfg, "tier", "standard"), SecurityTier.STANDARD)

            policy = SecurityPolicy(
                tier=tier,
                blocked_tools=set(getattr(sec_cfg, "blocked_tools", [])),
            )
            # 追加额外敏感文件模式
            extra_patterns = getattr(sec_cfg, "extra_sensitive_patterns", [])
            if extra_patterns:
                policy.sensitive_file_patterns.extend(extra_patterns)

            # 创建审查日志记录器
            audit_log_path = getattr(sec_cfg, "audit_log", "data/audit.log")
            audit_logger = AuditLogger(log_file=audit_log_path, console_output=True)

            # 创建审查链路（可选LLM用于AI风险检测）
            ai_llm = None
            if getattr(sec_cfg, "enable_ai_classifier", True):
                ai_llm = LLMClient(config.llm)

            audit_chain = AuditChain(
                policy=policy,
                llm_client=ai_llm,
                workspace_root=config.agent.workspace_root,
                enable_ai_classifier=getattr(sec_cfg, "enable_ai_classifier", True),
                enable_human_gate=getattr(sec_cfg, "enable_human_gate", True),
            )

            tier_labels = {"relaxed": "宽松(仅拦截CRITICAL)", "standard": "标准(HIGH确认/CRITICAL拦截)", "strict": "严格(MEDIUM确认/HIGH+拦截)"}
            logger.info(
                f"安全审查系统就绪 | 等级: {tier_labels.get(getattr(sec_cfg, 'tier', 'standard'), 'standard')} | "
                f"AI检测: {'启用' if getattr(sec_cfg, 'enable_ai_classifier', True) else '禁用'} | "
                f"人工确认: {'启用' if getattr(sec_cfg, 'enable_human_gate', True) else '禁用'}"
            )
    except Exception as e:
        logger.error(f"安全审查系统初始化失败（将禁用审查功能）: {e}")

    # 工具注册中心（注入 AuditChain）
    registry = ToolRegistry(safety, audit_chain=audit_chain)

    # 注册所有工具（按需启用）
    if config.tools.file.enabled:
        registry.register(ReadFileTool)
        registry.register(WriteFileTool)
        registry.register(EditFileTool)
        logger.info("文件工具已注册 (read_file, write_file, edit_file)")

    if config.tools.shell.enabled:
        registry.register(RunCommandTool)
        logger.info("Shell 工具已注册 (run_command)")

    if config.tools.git.enabled:
        registry.register(GitStatusTool)
        registry.register(GitDiffTool)
        registry.register(GitCommitTool)
        logger.info("Git 工具已注册 (git_status, git_diff, git_commit)")

    if config.tools.search.enabled:
        registry.register(GrepSearchTool)
        registry.register(GlobTool)
        logger.info("搜索工具已注册 (grep_search, glob)")

    # 通用工具（始终注册）
    registry.register(AskUserQuestionTool)
    registry.register(TodoWriteTool)
    logger.info("通用工具已注册 (ask_user_question, todo_write)")

    logger.info(f"共注册 {registry.count()} 个原子工具")

    # ---- Skill 路由系统初始化 ----
    catalog = SkillCatalog()
    index = None
    ranker = None
    router = None

    definitions_path = Path(__file__).parent / "src" / "skills" / "definitions.yaml"
    if definitions_path.exists():
        try:
            catalog.load(str(definitions_path))
            logger.info(f"Skill 目录加载: {catalog.count()} 个 Skill, {catalog.category_count()} 个分类")

            index = SkillIndex(catalog)
            logger.info(f"Skill 索引构建: {catalog.tag_count()} 个标签")

            # 精排复用同一个 LLM 连接（千问 API）
            llm_client = LLMClient(config.llm)
            ranker = SkillRanker(llm_client, top_k=5)

            router = SkillRouter(catalog, index, ranker)
            logger.info("Skill 路由系统就绪")
        except Exception as e:
            logger.error(f"Skill 路由系统初始化失败（将使用全量工具模式）: {e}")
    else:
        logger.info("Skill 定义文件不存在，使用全量工具模式")

    # ---- 记忆系统初始化 ----
    memory_manager = None
    try:
        # 读取 memory 配置（兼容旧 config.yaml 无 memory 段的情况）
        memory_cfg = getattr(config, "memory", None)
        if memory_cfg and getattr(memory_cfg, "enabled", False):
            # 创建 LLM 客户端用于反思生成（复用已有连接或新建）
            reflect_llm = LLMClient(config.llm)

            reflector = MemoryReflector(
                reflect_llm,
                async_enabled=getattr(memory_cfg, "async_reflection", True),
            )

            store = MemoryStore(
                data_dir=getattr(memory_cfg, "data_dir", "data/memories"),
                chroma_path="data/chroma",
            )

            retriever = MemoryRetriever(
                store,
                max_tokens=getattr(memory_cfg, "max_context_tokens", 800),
            )

            memory_manager = MemoryManager(
                reflector=reflector,
                store=store,
                retriever=retriever,
                enabled=True,
                async_reflection=getattr(memory_cfg, "async_reflection", True),
            )

            logger.info(
                f"记忆系统就绪 | ChromaDB: {'可用' if store.chroma_available else '不可用（使用关键词匹配）'} | "
                f"Token 预算: {getattr(memory_cfg, 'max_context_tokens', 800)}"
            )
    except Exception as e:
        logger.error(f"记忆系统初始化失败（将禁用记忆功能）: {e}")

    # ---- 上下文压缩系统初始化 ----
    context_manager = None
    try:
        context_cfg = getattr(config, "context", None)
        if context_cfg and getattr(context_cfg, "enabled", True):
            context_manager = ContextManager(
                enabled=getattr(context_cfg, "enabled", True),
                max_tokens=getattr(context_cfg, "max_tokens", 9000),
                keep_recent=getattr(context_cfg, "keep_recent", 8),
                file_threshold_chars=getattr(context_cfg.placeholder, "file_threshold_chars", 3000),
                output_threshold_chars=getattr(context_cfg.placeholder, "output_threshold_chars", 2000),
                keep_head_lines=getattr(context_cfg.placeholder, "keep_head_lines", 30),
                keep_head_chars=getattr(context_cfg.placeholder, "keep_head_chars", 500),
                note_enabled=getattr(context_cfg.note, "enabled", True),
                note_use_llm=getattr(context_cfg.note, "use_llm_fallback", False),
                summary_max_chars=getattr(context_cfg.summary, "max_chars", 300),
                llm_client=LLMClient(config.llm),
            )
            logger.info(
                f"上下文压缩系统就绪 | Token阈值: {getattr(context_cfg, 'max_tokens', 9000)} | "
                f"占位替换: 文件>{getattr(context_cfg.placeholder, 'file_threshold_chars', 3000)}字符 "
                f"输出>{getattr(context_cfg.placeholder, 'output_threshold_chars', 2000)}字符"
            )
    except Exception as e:
        logger.error(f"上下文压缩系统初始化失败（将禁用压缩功能）: {e}")

    # ---- 多Agent协作系统初始化 ----
    orchestrator = None
    delegate_tool = None
    try:
        ma_cfg = getattr(config, "multi_agent", None)
        if ma_cfg and getattr(ma_cfg, "enabled", True):
            orchestrator = Orchestrator(
                enabled=True,
                parallel_workers=getattr(ma_cfg, "parallel_workers", 4),
            )
            # 创建 DelegateAgentTool 并注册到主 Registry
            delegate_tool = DelegateAgentTool(
                safety=safety,
                config=config,
                logger=logger,
            )
            registry.register_instance(delegate_tool)
            logger.info(
                f"多Agent协作系统就绪 | 并行worker: {getattr(ma_cfg, 'parallel_workers', 4)} | "
                f"子Agent最大迭代: {getattr(ma_cfg, 'sub_agent_max_iterations', 15)}"
            )
    except Exception as e:
        logger.error(f"多Agent协作系统初始化失败（将禁用多Agent模式）: {e}")

    # 创建跨轮对话记忆
    conv_memory = ConversationMemory(max_turns=10, max_files=30)

    # 创建 Agent（注入 router + memory + context_mgr + orchestrator + conv_memory）
    agent = Agent(
        config=config,
        logger=logger,
        registry=registry,
        skill_router=router,
        memory_manager=memory_manager,
        context_manager=context_manager,
        orchestrator=orchestrator,
        conversation_memory=conv_memory,
    )
    return agent, logger, safety


# ============================================================
# REPL 模式
# ============================================================

def _read_multiline(console, first_prompt="▸", cont_prompt="…") -> str:
    """多行输入读取，自动检测粘贴场景。

    普通单行输入：直接返回，无需额外操作。
    多行粘贴：自动检测缓冲区中的剩余数据，读取全部行后返回。
    兜底：如果检测失败（首行非空且看起来像被截断），进入续行模式。
    """
    import sys

    first = console.input(f"\n[bold green]{first_prompt}[/bold green] ").rstrip()
    if not first:
        return first

    # 快速路径：/ 开头的命令直接返回
    if first.startswith("/"):
        return first

    # 检测是否有更多数据在 stdin 缓冲区中（粘贴场景）
    has_more = False
    try:
        if sys.platform == "win32":
            import msvcrt
            has_more = msvcrt.kbhit()
        else:
            import select
            has_more = select.select([sys.stdin], [], [], 0.0)[0]
    except (ImportError, OSError):
        has_more = False

    if not has_more:
        return first

    # 粘贴检测成功：读取剩余行（非阻塞，读完缓冲区即止）
    import sys as _sys
    lines = [first]
    while True:
        # 每轮先检查是否还有数据在缓冲区
        more = False
        try:
            if _sys.platform == "win32":
                import msvcrt as _msvcrt
                more = _msvcrt.kbhit()
            else:
                import select as _select
                more = _select.select([_sys.stdin], [], [], 0.05)[0]
        except (ImportError, OSError):
            pass
        if not more:
            break
        try:
            line = input()
            if not line.strip():
                # 空行：可能是粘贴内容的段落分隔，积累后退出
                lines.append("")
                continue
            lines.append(line.rstrip())
        except EOFError:
            break
    return "\n".join(lines)


def repl_loop(agent: Agent, logger: AgentLogger, config: AppConfig):
    """交互式 REPL 主循环"""
    banner()

    while True:
        try:
            # 读取用户输入（自动检测并支持多行粘贴）
            user_input = _read_multiline(console).strip()

            if not user_input:
                continue

            # 处理内置命令
            if user_input.startswith("/"):
                cmd = user_input.lower()
                if cmd in ("/exit", "/quit"):
                    console.print("[dim]再见！[/dim]")
                    break
                elif cmd == "/help":
                    help_text()
                    continue
                elif cmd == "/clear":
                    # 重新创建 agent，重置会话和短期记忆
                    agent, logger, _ = setup_agent(config)
                    console.print("[dim]对话历史和短期记忆已清空。[/dim]")
                    continue
                elif cmd == "/stats":
                    _show_stats(agent)
                    continue
                else:
                    console.print(f"[yellow]未知命令: {user_input}[/yellow]")
                    console.print("[dim]输入 /help 查看可用命令[/dim]")
                    continue

            # 执行任务
            console.print()
            with console.status("[bold cyan]思考中...[/bold cyan]", spinner="dots"):
                result = agent.run(user_input)

            # 输出结果
            if result:
                console.print()
                console.print(
                    Panel(
                        Markdown(result),
                        title="Agent",
                        title_align="left",
                        border_style="blue",
                        padding=(1, 2),
                    )
                )

            # 显示本轮简要统计（/stats 命令可查看详情）
            if agent.session:
                elapsed = agent.session.get_elapsed_seconds()
                tool_calls = len(agent.session.tool_call_logs)
                console.print(
                    f"[dim]耗时 {elapsed:.1f}s | {tool_calls} 次工具调用[/dim]"
                )

        except KeyboardInterrupt:
            console.print("\n[dim]按 Ctrl+C 再次退出，或输入 /exit[/dim]")
            try:
                user_input = console.input("[bold green]▸[/bold green] ").strip()
                if user_input.lower() in ("/exit", "/quit"):
                    break
            except KeyboardInterrupt:
                console.print("\n[dim]再见！[/dim]")
                break
        except Exception as e:
            # 异常净化后展示，避免内部堆栈泄漏
            friendly = UserOutput.wrap_error(str(e))
            console.print(f"[red]{friendly}[/red]")


def _show_stats(agent: Agent):
    """显示统计信息"""
    if not agent.session:
        console.print("[dim]暂无统计数据。[/dim]")
        return
    s = agent.session
    tokens = s.get_token_summary()
    table = Table(title="会话统计", box=box.SIMPLE)
    table.add_column("指标", style="cyan")
    table.add_column("值", style="white")
    table.add_row("已用时间", f"{s.get_elapsed_seconds():.1f}s")
    table.add_row("迭代轮数", str(tokens["iterations"]))
    table.add_row("工具调用次数", str(len(s.tool_call_logs)))
    table.add_row("Prompt Tokens", str(tokens["prompt_tokens"]))
    table.add_row("Completion Tokens", str(tokens["completion_tokens"]))
    table.add_row("Total Tokens", str(tokens["total_tokens"]))
    console.print(table)


# ============================================================
# 单次任务模式
# ============================================================

def single_task_mode(agent: Agent, task: str):
    """单次任务模式 —— 执行完退出"""
    console.print(f"[dim]任务: {task}[/dim]")
    console.print()
    with console.status("[bold cyan]执行中...[/bold cyan]", spinner="dots"):
        result = agent.run(task)
    if result:
        console.print(result)
    # 输出统计
    if agent.session:
        tokens = agent.session.get_token_summary()
        console.print(
            f"\n[dim]Token 消耗: {tokens['total_tokens']} "
            f"(P:{tokens['prompt_tokens']} C:{tokens['completion_tokens']}) | "
            f"工具调用: {len(agent.session.tool_call_logs)} 次[/dim]"
        )


# ============================================================
# 程序入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="AI Coding Agent — 对标 Claude Code 的 Python 实现",
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default="",
        help="任务描述（指定后自动进入单次任务模式）",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="配置文件路径（默认 config.yaml）",
    )
    args = parser.parse_args()

    # 加载配置
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]配置加载失败: {e}[/red]")
        sys.exit(1)

    # 验证 API Key
    if not config.llm.api_key:
        console.print(
            "[red]错误: 未设置 QWEN_API_KEY 环境变量。[/red]\n"
            "[dim]请设置: export QWEN_API_KEY=<your_api_key>[/dim]"
        )
        sys.exit(1)

    # 初始化 Agent
    agent, logger, safety = setup_agent(config)

    if args.task:
        # 单次任务模式
        single_task_mode(agent, args.task)
    else:
        # REPL 模式
        repl_loop(agent, logger, config)


if __name__ == "__main__":
    main()
