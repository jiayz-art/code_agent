🤖 AI Coding Agent

对标 Claude Code​ 的企业级 AI 编程助手 | 基于 Qwen-Coder-Plus​ | 六层架构 | 自进化记忆 | 生产级安全

https://img.shields.io/badge/Python-3.10+-blue.svg

https://img.shields.io/badge/License-MIT-green.svg

https://img.shields.io/badge/Code%20Style-Black-black.svg

https://img.shields.io/badge/PRs-welcome-brightgreen.svg

📖 目录

✨ 核心特性

🏗️ 架构概览

🚀 快速开始

💡 使用示例

⚙️ 配置说明

📊 性能数据

🧩 扩展开发

🛠️ 技术栈

🗺️ 路线图

🤝 贡献指南

📄 许可证

✨ 核心特性

特性

	

说明

	

效果




🎯 Skill 分层路由​

	

倒排索引召回 → LLM 精排 → 按需追加

	

Token -40%​




🛡️ 四层安全审查​

	

规则 → 自检 → AI 评估 → 人工确认

	

100% 拦截高危操作​




🧠 自进化记忆​

	

ChromaDB + 异步反思 (问题/根因/方案/教训)

	

召回准确率 85%​




📦 上下文压缩​

	

占位替换 → 笔记生成 → 超限压缩

	

Token -33%​




🔄 多 Agent 协作​

	

Fork/Join 并行 + Git Worktree 隔离

	

执行速度 +50%​

🏗️ 架构概览
纯文本
┌────────────────────────────────────────────┐
│               入口层 (main.py)              │
│  REPL 交互 · 单次任务 · 配置加载 · 初始化   │
├────────────────────────────────────────────┤
│            Agent 引擎层 (agent.py)          │
│  状态机编排 · 循环控制 · 路由集成           │
├──────────┬──────────┬──────────┬───────────┤
│ Skill    │ 上下文   │ 安全     │ 记忆      │
│ 路由系统  │ 压缩系统  │ 审查链路  │ 管理系统   │
├──────────┴──────────┴──────────┴───────────┤
│            工具执行层 (tools/)              │
│  read_file · write_file · edit_file        │
│  run_command · git_* · grep_search         │
├────────────────────────────────────────────┤
│           LLM 通信层 (llm.py)              │
│  千问 API · OpenAI 兼容 · Token 统计        │
├────────────────────────────────────────────┤
│           基础设施层                        │
│  config · state · safety · logger          │
└────────────────────────────────────────────┘
🚀 快速开始
前置要求

Python 3.10+

Qwen API Key
(阿里云百炼)

安装步骤
bash
# 1. 克隆仓库
git clone https://github.com/jiayz-art/code_agent.git
cd code_agent

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. 安装依赖
pip install -r requirements.txt
配置

复制配置模板并填写你的 API Key：

bash
cp config.example.yaml config.yaml

编辑 config.yaml：

yaml
llm:
  api_key: "sk-your-qwen-api-key"
  model: "qwen-coder-plus"
运行

交互式 REPL 模式：

bash
python main.py

单次任务模式：

bash
python main.py --task "为 utils.py 编写单元测试并修复潜在 Bug"
💡 使用示例
🔧 代码生成

用 FastAPI 编写一个带 JWT 认证的登录接口，包含限流中间件。

🐛 调试修复

分析报错 AttributeError: 'NoneType' object has no attribute 'split'并给出修复方案。

🧪 自动化测试

为 order_service.py生成 Pytest 单元测试，覆盖 90% 的分支。

🔄 多 Agent 协作

生成订单模块代码，编写测试用例，进行代码审查，最后生成 Swagger 文档。

⚙️ 配置说明
安全等级
yaml
security:
  tier: "standard"  # relaxed | standard | strict

等级

	

说明

	

适用场景




relaxed

	

仅拦截致命错误

	

个人沙盒




standard

	

高风险需确认

	

企业开发




strict

	

中风险及以上拦截

	

生产环境

记忆系统
yaml
memory:
  enabled: true
  chroma_path: "./memory/chroma"
  max_entries: 1000
上下文压缩
yaml
context:
  placeholder_threshold: 3000  # 超长内容替换为占位符
  max_tokens: 9000             # 超限触发 LLM 压缩

📖 详细调参请见 TUNING.md

📊 性能数据

指标

	

数值




Token 效率提升

	

40%​




记忆召回准确率

	

85%​




高危操作拦截率

	

100%​




并行执行加速

	

50%​




上下文压缩率

	

33%​




代码规模

	

4000+ 行​

🧩 扩展开发
新增工具
python
# src/tools/my_tool.py
from .base import BaseTool, ToolResult

class MyTool(BaseTool):
    name = "my_tool"
    description = "自定义工具描述"
    parameters = {
        "type": "object",
        "properties": {"param": {"type": "string"}},
        "required": ["param"]
    }
    
    def execute(self, param: str) -> ToolResult:
        return ToolResult(success=True, content="执行完成")

# 在 main.py 中注册即可，无需修改 Agent 核心
新增 Skill

编辑 definitions.yaml：

yaml
skills:
  - name: my_skill
    display_name: "我的技能"
    category: develop
    tags: [custom, example]
    tools: [read_file, write_file]
    prompt_template: |
      ## 执行流程
      1. 分析需求
      2. 生成代码
🛠️ 技术栈

类别

	

技术




语言​

	

Python 3.10+




模型​

	

Qwen-Coder-Plus (DashScope)




存储​

	

ChromaDB + JSON




NLP​

	

jieba




并发​

	

ThreadPoolExecutor




终端​

	

Rich




架构​

	

状态机 · 责任链 · 异步回调

🗺️ 路线图

[x] Phase 1: 核心 Agent 与工具系统

[x] Phase 2: Skill 路由与记忆系统

[ ] Phase 3: Web UI 界面 (FastAPI + Vue)

[ ] Phase 4: MCP (Model Context Protocol) 协议支持

[ ] Phase 5: 插件市场与社区生态

🤝 贡献指南

我们非常欢迎各种形式的贡献！

Fork 本仓库

创建特性分支 (git checkout -b feature/amazing-feature)

提交更改 (git commit -m 'feat: add amazing feature')

推送到分支 (git push origin feature/amazing-feature)

开启一个 Pull Request

📖 请阅读 CONTRIBUTING.md
了解详细信息。

📄 许可证

本项目采用 MIT License​ 开源协议。详见 LICENSE
文件。

<div align="center">

🤖 AI Coding Agent — 让 AI 真正懂你的代码

⭐ Star us on GitHub
• 🐛 Report Bug
• 💡 Request Feature

</div>
