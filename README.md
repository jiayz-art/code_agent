# AI Coding Agent

对标 Claude Code 的 Python 实现，基于千问 Qwen-Coder-Plus 模型。自主完成代码阅读、文件编辑、命令执行、代码搜索和多Agent协作。

## 快速开始

### 环境要求

- Python 3.12+
- 阿里云 DashScope API Key（千问模型）

### 安装运行

```bash
cd E:\Agent
pip install -r requirements.txt

# 设置 API Key
export QWEN_API_KEY="sk-xxx"            # Linux/macOS
$env:QWEN_API_KEY="sk-xxx"              # Windows PowerShell
set QWEN_API_KEY=sk-xxx                 # Windows CMD

# REPL 交互模式
python main.py

# 单次任务模式
python main.py --task "分析 src/agent.py 的核心循环逻辑"

# 指定配置文件
python main.py --config my_config.yaml
```

获取 API Key: https://dashscope.console.aliyun.com/apiKey

## 架构总览

```
用户任务 → Agent 状态机 → LLM（千问）→ 工具执行 → 结果反馈 → 循环
   │              │                │                │
   ├─ Skill 路由 (Index→Rank)      │                ├─ 安全校验 (路径/命令/黑名单)
   ├─ 记忆系统 (三层)              │                ├─ 安全审查链 (规则→AI→人工)
   ├─ 上下文压缩 (占位→笔记→压缩)  └─ 多Agent编排 (Fork/Join)
   └─ 跨轮对话记忆
```

### Agent 核心循环

- 状态机流转：`INIT → THINKING → EXECUTING → DONE/MAX_ITER/ERROR`
- 只读工具并行执行（ThreadPoolExecutor，max_workers=4）
- 连续相同回复自动终止
- 只读工具结果缓存（30秒 TTL）
- 全链路审计日志 + Token 追踪

## 核心模块

### Skill 路由系统

三层路由减少 Token 消耗——只把相关工具暴露给 LLM：

- **目录层** — YAML 定义 14 个 Skill，5 大分类（调试、开发、测试、文档、Git）
- **召回层** — 基于 jieba 分词 + Tag 召回 Top 15 候选
- **精排层** — LLM 打分精排 Top 5
- 按需追加：LLM 请求了路由外的工具时动态扩展

### 多Agent协作

主控 Agent 通过 Fork/Join 模式分派子任务：

| 子Agent | 职责 | 权限 |
|---------|------|------|
| `code_generator` | 编写/修改生产代码 | 完整写入权限 |
| `tester` | 编写和运行测试 | 仅测试文件 |
| `reviewer` | 代码审查 | 只读 |
| `documenter` | 生成文档 | 只读 |

- Git worktree 隔离并行任务
- 结果校验强制检查每个 Agent 的工具权限
- 可配置并行 worker 数（默认 4）

### 安全审查链

工具执行前三层防御：

1. **规则过滤** — 静态模式匹配（敏感文件、危险命令）
2. **AI 风险分类** — LLM 检测 Prompt 注入、恶意代码、社会工程、路径欺骗
3. **人工确认** — MEDIUM+ 风险操作需交互确认

- 三级安全策略：`relaxed`（宽松）/ `standard`（标准）/ `strict`（严格）
- 完整审计日志持久化到 `data/audit.log`

### 自进化记忆系统

三层记忆 + 异步反思：

- **用户画像** — 跨项目持久化（技术栈、偏好、交互历史）
- **程序性记忆** — 任务经验 + 向量检索（ChromaDB，降级为关键词匹配）
- **情景记忆** — 每个任务的决策链 + 前后状态

钩子机制：`before_task()` 注入历史上下文 → `after_task()` 触发异步深度反思 → `get_feedback()` 输出 Skill 改进建议

### 分层上下文压缩

三层流水线控制上下文在 Token 预算内：

1. **占位替换** — 超长文件内容/命令输出替换为紧凑占位符
2. **笔记生成** — 工具结果自动生成结构化摘要
3. **超限压缩** — 超阈值时对历史消息做 LLM 摘要

按需展开：用户说 `展开 PLACEHOLDER:xxx` 恢复原始内容

## 可用工具

| 工具 | 功能 | 安全措施 |
|------|------|---------|
| `read_file` | 读取文件内容 | 路径白名单、5MB 限制 |
| `write_file` | 创建/覆盖文件 | 路径白名单 |
| `edit_file` | 精确字符串替换 | 路径白名单 |
| `run_command` | 执行 Shell 命令 | 命令黑名单、60s 超时 |
| `grep_search` | 正则搜索文件内容 | 排除 node_modules 等 |
| `glob` | 文件模式匹配 | 排除目录 |
| `git_status` | 查看仓库状态 | — |
| `git_diff` | 查看代码变更 | — |
| `git_commit` | 提交变更 | 保护 main/master 分支 |
| `ask_user_question` | 交互式多选提问 | — |
| `todo_write` | 管理任务清单 | — |
| `delegate_agent` | 分派子Agent任务 | worktree 隔离 |

## 项目结构

```
Agent/
├── main.py                      # 入口（REPL 交互 + 单次任务）
├── config.yaml                  # YAML 配置 + 环境变量注入
├── requirements.txt             # 依赖：openai, pyyaml, rich, jieba, chromadb
├── src/
│   ├── agent.py                 # 核心 Agent 状态机引擎
│   ├── config.py                # dataclass 强类型配置加载器
│   ├── llm.py                   # OpenAI 兼容的 LLM 客户端（千问/DashScope）
│   ├── state.py                 # 会话状态 + Token 统计
│   ├── safety.py                # 路径/命令安全校验（SafetyChecker）
│   ├── logger.py                # JSON 结构化日志 + Token 追踪
│   ├── output_filter.py         # 用户输出净化
│   ├── conversation_memory.py   # 跨轮对话短期记忆
│   │
│   ├── tools/                   # 原子工具实现
│   │   ├── base.py              # BaseTool 抽象 + ToolRegistry + ToolResult
│   │   ├── file_tools.py        # read_file, write_file, edit_file
│   │   ├── shell_tools.py       # run_command
│   │   ├── git_tools.py         # git_status, git_diff, git_commit
│   │   ├── search_tools.py      # grep_search
│   │   ├── glob_tool.py         # glob 模式匹配
│   │   ├── ask_tool.py          # ask_user_question
│   │   └── todo_tool.py         # todo_write
│   │
│   ├── skills/                  # Skill 路由系统
│   │   ├── definitions.yaml     # Skill 目录（14个Skill, 5个分类）
│   │   ├── catalog.py           # SkillCatalog — YAML 加载 + 查询 API
│   │   ├── index.py             # SkillIndex — Tag 召回
│   │   ├── ranker.py            # SkillRanker — LLM 精排
│   │   ├── router.py            # SkillRouter — 路由编排层
│   │   └── intent.py            # 意图提取
│   │
│   ├── memory/                  # 自进化记忆系统
│   │   ├── manager.py           # MemoryManager — before/after task 钩子
│   │   ├── reflector.py         # MemoryReflector — 异步摘要 + 详情生成
│   │   ├── store.py             # MemoryStore — JSON 持久化 + ChromaDB 向量索引
│   │   ├── retriever.py         # MemoryRetriever — 语义 + 关键词检索
│   │   ├── models.py            # MemoryEntry, UserProfile, EpisodicMemory
│   │   └── skill_feedback.py    # SkillFeedbackLoop — 记忆驱动的 Skill 改进
│   │
│   ├── context/                 # 分层上下文压缩
│   │   ├── manager.py           # ContextManager — 统一入口
│   │   ├── placeholder.py       # PlaceholderEngine — 长内容占位替换
│   │   ├── notes.py             # NoteGenerator — 结构化自动摘要
│   │   └── compressor.py        # Compressor — LLM 超限压缩
│   │
│   ├── multi_agent/             # 多Agent协作系统
│   │   ├── orchestrator.py      # Orchestrator — Fork/Join 主控编排器
│   │   ├── sub_agent.py         # SubAgent — 隔离的子Agent实例
│   │   ├── delegate_tool.py     # DelegateAgentTool — 分派接口
│   │   ├── team.py              # 团队组合
│   │   └── types.py             # AgentType 枚举、SubAgentResult、白名单
│   │
│   └── security/                # 安全审查链
│       ├── types.py             # RiskLevel, RiskAssessment, AuditDecision
│       ├── audit_chain.py       # AuditChain — 三层审查编排
│       ├── rule_filter.py       # RuleFilter — 静态规则匹配（第1层）
│       ├── ai_classifier.py     # AIRiskClassifier — LLM 风险检测（第2层）
│       ├── human_gate.py        # HumanGate — 人工确认门禁（第3层）
│       ├── audit_logger.py      # AuditLogger — 审计日志持久化
│       └── tool_check.py        # 工具级安全检查
│
└── docs/
    └── superpowers/
        ├── specs/               # 技术设计文档
        └── plans/               # 实施计划
```

## REPL 命令

| 命令 | 说明 |
|------|------|
| `/help` | 显示帮助信息 |
| `/clear` | 清空对话历史 |
| `/stats` | 查看 Token 消耗统计 |
| `/exit` / `/quit` | 退出程序 |

## 配置说明

所有配置项在 `config.yaml` 中，支持 `${环境变量}` 注入：

```yaml
llm:
  api_key: "${QWEN_API_KEY}"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model: "qwen-coder-plus"
  temperature: 0.1              # 0=确定性, 2=创造性
  max_tokens: 8192

agent:
  max_iterations: 30            # 最大工具调用轮数
  workspace_root: "."           # 文件操作安全边界

memory:
  enabled: true
  async_reflection: true        # 异步深度反思
  data_dir: "data/memories"

context:
  enabled: true
  max_tokens: 9000              # 触发压缩的 Token 阈值

multi_agent:
  enabled: true
  parallel_workers: 4           # 最大并行子Agent数
  sub_agent_max_iterations: 15

security:
  enabled: true
  tier: "standard"              # relaxed | standard | strict
  enable_ai_classifier: true    # AI 风险检测
  enable_human_gate: true       # 人工确认
```

## 技术栈

- **Python** 3.12+
- **LLM**: 千问 Qwen（DashScope，OpenAI 兼容接口）
- **向量数据库**: ChromaDB（可选，降级为关键词匹配）
- **终端**: Rich（面板、表格、Markdown、实时动画）
- **NLP**: jieba（中文分词，用于 Skill 索引）
- **配置**: YAML + dataclass + 环境变量注入
- **并发**: ThreadPoolExecutor（并行工具调用 + Fork/Join 子Agent）

## 设计原则

1. **Agent 只做编排，不做实现** — 核心 Agent 只管理 LLM→工具→结果的循环，所有逻辑在独立模块
2. **安全优先** — 路径校验、命令黑名单、三层审查链，失败如实汇报不撒谎
3. **Token 高效** — Skill 路由裁剪工具列表、上下文压缩、工具缓存、Prompt 瘦身
4. **配置驱动** — 所有行为参数在 config.yaml，调参不改代码
5. **中文原生** — 系统提示词、用户输出、文档全部中文优先

## 扩展指南

### 新增工具

```python
# src/tools/my_tool.py
from src.tools.base import BaseTool, ToolResult

class MyTool(BaseTool):
    name = "my_tool"
    description = "工具用途说明（LLM 阅读此描述决定何时调用）"
    parameters = {
        "type": "object",
        "properties": {
            "arg1": {"type": "string", "description": "参数说明"}
        },
        "required": ["arg1"],
    }

    def execute(self, arg1: str) -> ToolResult:
        result = do_something(arg1)
        return ToolResult(success=True, content=f"执行结果: {result}")
```

在 `main.py` 注册：

```python
from src.tools.my_tool import MyTool
registry.register(MyTool)
```

### 新增 Skill

在 `src/skills/definitions.yaml` 追加定义即可，无需改代码：

```yaml
skills:
  - name: my_skill
    display_name: "我的 Skill"
    description: "Skill 描述"
    category: develop
    tags: [python, feature, implementation]
    tools: [read_file, write_file, edit_file, run_command]
    prompt_template: |
      ## 工作流程
      1. 第一步
      2. 第二步
```

Catalog 自动从 YAML 加载，路由系统自动生效。

## 安全设计

- **路径白名单**：所有文件操作限制在 `workspace_root` 内，`resolve().relative_to()` 防止符号链接逃逸和 `../` 越界
- **命令黑名单**：正则拦截 `rm -rf /`、`shutdown`、`mkfs`、fork bomb、`curl|sh` 等危险命令
- **三层审查链**：规则过滤 → AI 风险分类 → 人工确认，工具执行前逐层检查
- **分支保护**：禁止对 main/master/release 等受保护分支执行危险操作
- **Prompt 注入防御**：AI 分类器检测 Prompt 覆盖企图、恶意代码注入、社会工程欺骗、路径伪装
- **审计日志**：所有安全事件记录到 `data/audit.log`，含风险等级、决策、时间戳
