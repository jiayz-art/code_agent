# AI Coding Agent — 集成整合与性能优化设计

日期: 2026-05-21 | 状态: 已确认

## 目标

1. 端到端可运行：`pip install -r requirements.txt && python main.py --task "..."` 全链路跑通
2. 响应速度优先优化：减少 LLM 调用次数、并行工具执行、System Prompt 瘦身
3. 输出三份文档：ARCHITECTURE.md、TUNING.md、RESUME.md

## 策略

并行工作流（策略 C）：修复 + 优化 + 文档三线推进，文档随优化过程同步记录真实数据。

---

## Part 1: 修复清单

### 1.1 `requirements.txt` 补完
当前只有一行内容，缺少关键依赖。需要补充：openai、pyyaml、rich、jieba、chromadb 等。

### 1.2 Skill 路由逻辑修正
- `get_schemas_for()` 在 agent.py:249-253 处，`base_tools` 合并逻辑可能导致路由选中的 tools 被覆盖
- `on_demand_add()` 按 tool_name 查找，但 `catalog.get_by_name()` 查的是 skill name → 需区分 tool name 和 skill name 查找

### 1.3 工具调用并行化
`_execute_tool_calls()` 当前串行 for 循环 → 改为 ThreadPoolExecutor 并行执行无依赖的工具调用。

### 1.4 上下文笔记生成位置
`process_tool_result()` 中笔记追加方式改进：笔记放在结果内容之后而非之前，避免 LLM 优先看到摘要而忽略原始内容。

---

## Part 2: 速度优化（三个快刀）

### 2.1 减少 LLM 调用次数
- **路由缓存增强**：SkillRouter 当前基于精确 MD5 hash 缓存 → 加 TTL + 前缀匹配相似任务复用
- **工具结果短缓存**：同一 session 内相同工具+参数 30s 内复用（仅限只读工具）
- **提前终止**：连续 2 轮无新增工具调用 + content 无实质变化 → 判定 DONE

### 2.2 并行工具调用
- 无依赖关系检测：通过工具类型判断（只读工具可并行，写操作与读操作有依赖）
- ThreadPoolExecutor 并行执行，max_workers=4
- 预期：3 个独立工具调用从 sum(t1,t2,t3) 降到 max(t1,t2,t3)

### 2.3 System Prompt 瘦身
- 基础 SYSTEM_PROMPT 从 ~1900 chars 砍到 ~400 chars（角色定义 + 核心禁止行为）
- 工具使用指南移到各 Skill 的 prompt_template，仅在路由命中时注入
- 预期：每轮省 ~500 prompt tokens

---

## Part 3: 文档交付

### ARCHITECTURE.md
- ASCII art 六层架构图
- 模块职责、接口、数据流
- 关键设计决策

### TUNING.md
- Token 诊断方法
- 三维度配置指南（上下文压缩、路由缓存、并行度）
- 场景推荐配置

### RESUME.md
- 一句话定位
- 三个核心亮点 + 量化指标
- 技术栈列表

---

## Part 4: 验证路径

```
python main.py --task "读取 src/agent.py 解释核心循环逻辑"

预期链路：
  SkillRouter.route() → explain_code Skill
  → tools=[read_file, grep_search]
  → Agent: read_file → LLM分析 → 返回
  → AuditChain(LOW→PASS)
  → ContextManager(未超限跳过)
  → MemoryManager.reflect()
```

---

## 改动文件清单

| 文件 | 改动类型 | 内容 |
|------|---------|------|
| `requirements.txt` | 重写 | 补全所有依赖 |
| `src/agent.py` | 修改 | 并行执行、提前终止、Prompt 瘦身 |
| `src/skills/router.py` | 修改 | 路由缓存增强、tool/skill 名区分 |
| `docs/ARCHITECTURE.md` | 新建 | 架构文档 |
| `docs/TUNING.md` | 新建 | 调优指南 |
| `docs/RESUME.md` | 新建 | 简历亮点 |
