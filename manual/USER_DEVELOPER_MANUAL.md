# A3S Code 用户与开发者手册

> **A3S Code** - Agentic Agent Framework  
> 一个基于 Rust 的智能代理框架，提供 Python 和 Node.js 原生绑定

---

## 📚 目录

- [第一部分：用户指南](#第一部分用户指南)
  - [1. 简介](#1-简介)
  - [2. 安装与配置](#2-安装与配置)
  - [3. 快速开始](#3-快速开始)
  - [4. 核心概念](#4-核心概念)
  - [5. 工具系统](#5-工具系统)
  - [6. Skills 系统](#6-skills-系统)
  - [7. 多 Agent 协作](#7-多-agent-协作)
  - [8. 安全与权限](#8-安全与权限)
  - [9. 斜杠命令](#9-斜杠命令)
  - [10. 计划任务](#10-计划任务)
  - [11. 会话管理](#11-会话管理)
- [第二部分：开发者指南](#第二部分开发者指南)
  - [12. 架构概览](#12-架构概览)
  - [13. 开发环境搭建](#13-开发环境搭建)
  - [14. 核心模块解析](#14-核心模块解析)
  - [15. 扩展开发](#15-扩展开发)
  - [16. Hook 系统](#16-hook-系统)
  - [17. 插件开发](#17-插件开发)
  - [18. 测试与调试](#18-测试与调试)
  - [19. 贡献指南](#19-贡献指南)

---

# 第一部分：用户指南

## 1. 简介

A3S Code 是一个功能强大的 **Agentic Agent 框架**，允许你赋予大型语言模型（LLM）以下能力：

- 📁 **文件操作** - 读取、写入、编辑、补丁文件
- 🔍 **代码搜索** - 使用 Grep、Glob 等工具搜索代码库
- 🖥️ **命令执行** - 在沙盒环境中运行 Shell 命令
- 🌐 **网络访问** - 网页抓取和搜索
- 🤝 **任务委派** - 将任务分派给子代理或多代理团队

### 支持的语言与平台

| 平台 | 安装方式 |
|------|----------|
| Python | `pip install a3s-code` |
| Node.js | `npm install @a3s-lab/code` |
| Rust | `cargo add a3s-code-core` |

### 支持的 LLM 提供商

- **Anthropic** (Claude 系列)
- **OpenAI** (GPT 系列)
- **DeepSeek**
- **Kimi** (Moonshot)
- **Together**
- **Groq**

---

## 2. 安装与配置

### 2.1 Python 安装

```bash
pip install a3s-code
```

### 2.2 Node.js 安装

```bash
npm install @a3s-lab/code
```

### 2.3 配置代理 (agent.hcl)

创建 `agent.hcl` 配置文件：

```hcl
# 默认模型
default_model = "anthropic/claude-sonnet-4-20250514"

# LLM 提供商配置
providers {
  name    = "anthropic"
  api_key = env("ANTHROPIC_API_KEY")
}

providers {
  name    = "openai"
  api_key = env("OPENAI_API_KEY")
}

# 存储后端: "memory", "file", 或 "custom"
storage_backend = "file"

# 会话目录
sessions_dir = "./sessions"

# Skill 目录
skill_dirs = ["./skills"]

# 最大工具执行轮数
max_tool_rounds = 50
```

### 2.4 环境变量

```bash
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
```

---

## 3. 快速开始

### 3.1 Python 示例

```python
from a3s_code import Agent

# 创建代理
agent = Agent.create("agent.hcl")

# 创建会话
session = agent.session("/my-project")

# 发送请求
result = session.send("分析这个项目的架构")
print(result.text)
```

### 3.2 Node.js 示例

```typescript
import { Agent } from '@a3s-lab/code';

const agent = await Agent.create('agent.hcl');
const session = agent.session('/my-project');

const result = await session.send('分析这个项目的架构');
console.log(result.text);
```

### 3.3 第一个任务

```python
# 查找所有处理认证错误的代码位置
result = session.send("查找所有处理认证错误的代码位置")

# 分析代码质量
result = session.send("审查 main.py 的代码质量并给出改进建议")

# 执行测试
result = session.send("运行测试套件并报告结果")
```

---

## 4. 核心概念

### 4.1 架构层级

```
Agent (配置 + 提供商注册表)
  └── Session (工作空间 + 工具 + LLM)
        └── AgentLoop (基于轮次的执行)
              ├── LlmClient      → 发送消息，接收工具调用
              ├── ToolExecutor   → 运行工具，强制执行权限
              ├── SkillRegistry  → 将 Skills 注入系统提示
              └── PluginManager  → 加载可选的工具+Skill 包
```

### 4.2 核心组件

| 组件 | 说明 |
|------|------|
| **Agent** | 顶层配置和工厂，管理提供商注册表 |
| **Session** | 工作空间容器，包含工具、LLM 客户端和状态 |
| **AgentLoop** | 执行循环，管理 LLM 和工具之间的交互 |
| **Skill** | Markdown 文件，定义行为和能力 |
| **Tool** | 代理可调用的功能 |

### 4.3 SessionOptions 配置

```python
from a3s_code import Agent, SessionOptions

opts = SessionOptions()

# 指定模型
opts.model = "openai/gpt-4o"

# 启用内置 Skills
opts.builtin_skills = True

# 加载自定义 Skills
opts.skill_dirs = ["./skills"]

# 添加插件
from a3s_code import AgenticSearch, AgenticParse
opts.plugins = [AgenticSearch(), AgenticParse()]

session = agent.session(".", opts)
```

---

## 5. 工具系统

### 5.1 内置工具（16个）

#### 文件工具

| 工具 | 说明 | 示例 |
|------|------|------|
| `read` | 读取文件内容 | `read: /path/to/file.py` |
| `write` | 写入文件 | `write: /path/to/file.py` |
| `edit` | 编辑文件 | `edit: /path/to/file.py` |
| `patch` | 应用补丁 | `patch: /path/to/file.py` |

#### 搜索工具

| 工具 | 说明 | 示例 |
|------|------|------|
| `grep` | 文本搜索 | `grep: "function name"` |
| `glob` | 文件匹配 | `glob: "**/*.py"` |
| `ls` | 目录列表 | `ls: /path/to/dir` |

#### 其他工具

| 工具 | 说明 |
|------|------|
| `bash` | 执行 Shell 命令 |
| `web_fetch` | 抓取网页内容 |
| `web_search` | 执行网络搜索 |
| `git_worktree` | Git 工作树操作 |

### 5.2 委派工具

| 工具 | 说明 |
|------|------|
| `task` | 委派给单个代理 |
| `parallel_task` | 并行委派多个任务 |
| `run_team` | 运行代理团队 |
| `batch` | 批量执行任务 |
| `Skill` | 调用特定 Skill |

### 5.3 插件工具

```python
# 启用 AgenticSearch - 自然语言代码搜索
from a3s_code import AgenticSearch
opts.plugins = [AgenticSearch()]

# 启用 AgenticParse - 增强解析
from a3s_code import AgenticParse
opts.plugins = [AgenticParse()]
```

---

## 6. Skills 系统

Skills 是 Markdown 文件，用于塑造 LLM 行为。

### 6.1 Skill 文件结构

```markdown
---
name: safe-reviewer
description: 审查代码而不修改文件
allowed-tools: "read(*), grep(*), glob(*)"
---

审查工作空间中的代码。你可以读取和搜索文件，但不能写入、编辑或执行任何操作。

审查清单：
1. 检查潜在的安全问题
2. 验证错误处理
3. 评估代码可读性
4. 提供改进建议
```

### 6.2 使用 Skills

```python
opts = SessionOptions()
opts.skill_dirs = ["./skills"]
opts.builtin_skills = True  # 启用内置 Skills
session = agent.session(".", opts)
```

### 6.3 内置 Skills

| Skill | 功能 |
|-------|------|
| `agentic-search` | 智能代码搜索 |
| `code-search` | 代码搜索辅助 |
| `code-review` | 代码审查 |
| `explain-code` | 代码解释 |
| `find-bugs` | 缺陷检测 |
| `builtin-tools` | 工具使用指导 |
| `delegate-task` | 任务委派 |
| `find-skills` | Skill 发现 |

---

## 7. 多 Agent 协作

### 7.1 单个子代理

```python
result = session.send('task: 探索代码库并总结架构')
```

### 7.2 并行任务

```python
result = session.send('parallel_task: [审计安全性, 检查性能, 审查测试]')
```

### 7.3 代理团队

```python
# 运行代理团队（负责人分解 → 执行者执行 → 审查者验证）
result = session.send('run_team: 重构认证模块')
```

### 7.4 代理类型

| 类型 | 描述 |
|------|------|
| `explore` | 只读探索 |
| `general` | 完整功能 |
| `plan` | 仅分析 |

---

## 8. 安全与权限

### 8.1 权限策略

```python
from a3s_code import SessionOptions, PermissionPolicy, PermissionRule

opts = SessionOptions()
opts.permission_policy = PermissionPolicy(
    allow=[
        PermissionRule("read(*)"),
        PermissionRule("grep(*)")
    ],
    deny=[
        PermissionRule("bash(*)")
    ],
    default_decision="deny",
)
session = agent.session(".", opts)
```

### 8.2 人机确认 (HITL)

```python
# 在每个工具调用前提示确认
opts.hitl_enabled = True
```

### 8.3 安全特性

| 特性 | 说明 |
|------|------|
| **显式权限** | 默认拒绝，需明确授权 |
| **人机确认** | 工具调用前提示确认 |
| **Skill 限制** | `allowed-tools` 限制可调用工具 |
| **AHP 集成** | 运行时拦截和清理工具调用 |
| **自动压缩** | 达到令牌限制前自动压缩上下文 |
| **熔断器** | 3次连续失败后停止，防止无限重试 |
| **延续注入** | 防止 LLM 提前停止任务 |

---

## 9. 斜杠命令

在会话中输入 `/help` 查看可用命令：

| 命令 | 说明 |
|------|------|
| `/help` | 列出可用命令 |
| `/model [provider/model]` | 显示或切换当前模型 |
| `/cost` | 显示令牌使用和估计成本 |
| `/clear` | 清除对话历史 |
| `/compact` | 手动触发上下文压缩 |
| `/tools` | 列出已注册工具 |
| `/loop [interval] <prompt>` | 安排重复提示 |
| `/cron-list` | 列出计划任务 |
| `/cron-cancel <id>` | 取消计划任务 |

### 9.1 自定义命令

```python
session.register_command(
    "status", 
    "显示状态", 
    lambda args, ctx: f"模型: {ctx['model']}"
)
result = session.send("/status")
```

---

## 10. 计划任务

### 10.1 使用斜杠命令

```python
# 每5分钟检查测试状态
session.send('/loop 5m 检查测试是否仍在通过')
```

### 10.2 程序化 API

```python
# 安排任务（间隔300秒）
task_id = session.schedule_task('总结上次检查以来的 git 日志', 300)

# 列出计划任务
tasks = session.list_scheduled_tasks()

# 取消任务
session.cancel_scheduled_task(task_id)
```

### 10.3 间隔语法

- `30s` - 30秒
- `5m` - 5分钟
- `2h` - 2小时
- `1d` - 1天

限制：每会话最多50个任务，3天后自动过期。

---

## 11. 会话管理

### 11.1 BTW - 临时问题

询问旁支问题而不影响对话历史：

```python
btw = session.btw("PostgreSQL 默认端口是多少？")
print(btw.answer)        # "5432"
print(btw.total_tokens)  # 仅此次查询的令牌使用
# 主对话继续 - btw 问题不在历史中
```

### 11.2 会话持久化

```python
from a3s_code import SessionOptions, FileSessionStore, FileMemoryStore

opts = SessionOptions()
opts.session_store = FileSessionStore('./sessions')
opts.memory_store = FileMemoryStore('./memory')
opts.session_id = 'my-session'
opts.auto_save = True

session = agent.session(".", opts)

# 恢复会话
resumed = agent.resume_session('my-session', opts)
```

### 11.3 多提供商切换

```python
# 按会话切换模型
session = agent.session(".", model="openai/gpt-4o")
```

---

# 第二部分：开发者指南

## 12. 架构概览

### 12.1 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        A3S Code                              │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Python SDK  │  │  Node.js SDK │  │   Rust Core  │      │
│  │  (PyO3)      │  │  (NAPI)      │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    Agent     │  │   Session    │  │  AgentLoop   │      │
│  │  (配置管理)   │  │  (状态管理)   │  │  (执行循环)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  LlmClient   │  │ ToolExecutor │  │SkillRegistry │      │
│  │  (LLM 通信)   │  │  (工具执行)   │  │  (技能管理)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │PluginManager │  │  HookSystem  │  │Permissions   │      │
│  │  (插件管理)   │  │  (钩子系统)   │  │  (权限控制)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 12.2 核心模块

| 模块 | 路径 | 说明 |
|------|------|------|
| `agent.rs` | `core/src/` | Agent 主逻辑 |
| `session.rs` | `core/src/session/` | 会话管理 |
| `tools/` | `core/src/tools/` | 工具实现 |
| `skills/` | `core/src/skills/` | Skill 系统 |
| `llm/` | `core/src/llm/` | LLM 客户端 |
| `permissions.rs` | `core/src/` | 权限控制 |
| `hooks/` | `core/src/hooks/` | 钩子系统 |

---

## 13. 开发环境搭建

### 13.1 前置要求

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python (用于 Python SDK)
python -m pip install maturin

# Node.js (用于 Node.js SDK)
npm install -g napi-rs
```

### 13.2 克隆与构建

```bash
git clone <repository-url>
cd a3s-code

# 构建核心
cargo build --release

# 构建 Python SDK
cd sdk/python
maturin develop

# 构建 Node.js SDK
cd sdk/node
npm install
npm run build
```

### 13.3 开发工具

```bash
# 运行测试
cargo test

# 代码检查
cargo clippy

# 格式化
cargo fmt

# 使用 just 执行任务
just --list
```

---

## 14. 核心模块解析

### 14.1 Agent 模块 (`agent.rs`)

```rust
// Agent 结构
pub struct Agent {
    config: Config,
    provider_registry: ProviderRegistry,
}

impl Agent {
    // 创建 Agent
    pub fn create(config_path: &str) -> Result<Self>;
    
    // 创建会话
    pub fn session(&self, workspace: &str) -> Session;
    
    // 恢复会话
    pub fn resume_session(&self, session_id: &str) -> Result<Session>;
}
```

### 14.2 Session 模块 (`session/`)

```rust
pub struct Session {
    id: String,
    workspace: PathBuf,
    tool_executor: ToolExecutor,
    llm_client: LlmClient,
    skill_registry: SkillRegistry,
}

impl Session {
    // 发送消息
    pub fn send(&mut self, prompt: &str) -> Result<Response>;
    
    // BTW 查询
    pub fn btw(&self, question: &str) -> Result<BtwResponse>;
    
    // 计划任务
    pub fn schedule_task(&mut self, task: &str, interval_secs: u64) -> String;
}
```

### 14.3 Tool 模块 (`tools/`)

```rust
// Tool trait
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn execute(&self, input: ToolInput) -> Result<ToolOutput>;
}

// 内置工具
pub mod file_tools;   // read, write, edit, patch
pub mod search_tools; // grep, glob, ls
pub mod shell_tools;  // bash
pub mod web_tools;    // web_fetch, web_search
```

### 14.4 Skill 模块 (`skills/`)

```rust
pub struct Skill {
    name: String,
    description: String,
    allowed_tools: Vec<String>,
    content: String,
}

pub struct SkillRegistry {
    skills: Vec<Skill>,
    builtin_enabled: bool,
}
```

---

## 15. 扩展开发

### 15.1 创建自定义工具

```rust
use a3s_code_core::tools::{Tool, ToolInput, ToolOutput};

pub struct MyTool;

impl Tool for MyTool {
    fn name(&self) -> &str {
        "my_tool"
    }
    
    fn description(&self) -> &str {
        "我的自定义工具描述"
    }
    
    fn execute(&self, input: ToolInput) -> Result<ToolOutput> {
        // 实现逻辑
        Ok(ToolOutput::new("结果"))
    }
}
```

### 15.2 创建自定义 Skill

在 `skills/` 目录创建 Markdown 文件：

```markdown
---
name: my-skill
description: 我的自定义 Skill
allowed-tools: "read(*), grep(*)"
---

# 我的 Skill

这是 Skill 的详细说明，LLM 将使用这些信息来执行任务。

## 使用场景

1. 场景一
2. 场景二

## 示例

```
示例代码或命令
```
```

### 15.3 扩展现有工具

```rust
// 扩展现有工具的行为
pub trait ToolExtension {
    fn pre_execute(&self, input: &ToolInput) -> Result<()>;
    fn post_execute(&self, output: &ToolOutput) -> Result<()>;
}
```

---

## 16. Hook 系统

### 16.1 可用钩子事件

| 事件 | 说明 | 可拦截 |
|------|------|--------|
| `PreToolUse` | 工具使用前 | ✅ |
| `PostToolUse` | 工具使用后 | ❌ |
| `GenerateStart` | 生成开始前 | ✅ |
| `GenerateEnd` | 生成结束后 | ❌ |
| `SessionStart` | 会话开始时 | ❌ |
| `SessionEnd` | 会话结束时 | ❌ |
| `SkillLoad` | Skill 加载时 | ❌ |
| `SkillUnload` | Skill 卸载时 | ❌ |
| `PrePrompt` | 提示前 | ✅ |
| `PostResponse` | 响应后 | ❌ |
| `OnError` | 错误发生时 | ❌ |

### 16.2 实现 HookHandler

```rust
use a3s_code::HookHandler;

struct MyHook;

impl HookHandler for MyHook {
    fn pre_tool_use(&self, tool_name: &str, tool_input: &Value, ctx: &Context) -> HookResult {
        if tool_name == "bash" && tool_input.contains("rm -rf") {
            return HookResult::block("拒绝破坏性命令");
        }
        HookResult::continue_()
    }
    
    fn generate_start(&self, prompt: &str, ctx: &Context) -> HookResult {
        // 修改提示或记录日志
        HookResult::continue_()
    }
}
```

### 16.3 使用钩子

```python
from a3s_code import SessionOptions, HookHandler

class SecurityHook(HookHandler):
    def pre_tool_use(self, tool_name, tool_input, ctx):
        # 安全检查逻辑
        return self.continue_()

opts = SessionOptions()
opts.hook_handler = SecurityHook()
session = agent.session(".", opts)
```

---

## 17. 插件开发

### 17.1 插件结构

```rust
pub trait Plugin: Send + Sync {
    fn name(&self) -> &str;
    fn initialize(&mut self, ctx: &PluginContext) -> Result<()>;
    fn tools(&self) -> Vec<Box<dyn Tool>>;
    fn skills(&self) -> Vec<Skill>;
}
```

### 17.2 实现插件

```rust
use a3s_code_core::plugin::{Plugin, PluginContext};

pub struct MyPlugin;

impl Plugin for MyPlugin {
    fn name(&self) -> &str {
        "my-plugin"
    }
    
    fn initialize(&mut self, ctx: &PluginContext) -> Result<()> {
        // 初始化逻辑
        Ok(())
    }
    
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![
            Box::new(MyTool),
        ]
    }
    
    fn skills(&self) -> Vec<Skill> {
        vec![
            Skill::new("my-skill", "描述", "..."),
        ]
    }
}
```

### 17.3 注册插件

```python
from a3s_code import SessionOptions

opts = SessionOptions()
opts.plugins = [MyPlugin()]
session = agent.session(".", opts)
```

---

## 18. 测试与调试

### 18.1 单元测试

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_my_tool() {
        let tool = MyTool;
        let input = ToolInput::new(json!({"key": "value"}));
        let output = tool.execute(input).unwrap();
        assert_eq!(output.text(), "expected");
    }
}
```

### 18.2 集成测试

```python
import pytest
from a3s_code import Agent, SessionOptions

@pytest.fixture
def agent():
    return Agent.create("test-agent.hcl")

def test_session_send(agent):
    session = agent.session("./test-workspace")
    result = session.send("Hello")
    assert result.text is not None
```

### 18.3 调试技巧

```bash
# 启用详细日志
export RUST_LOG=debug
export A3S_DEBUG=1

# 使用 just 运行特定测试
just test-core
just test-python

# 性能分析
cargo flamegraph
```

### 18.4 故障排除

| 问题 | 解决方案 |
|------|----------|
| 模型连接失败 | 检查 API 密钥和网络 |
| 工具权限被拒绝 | 检查 PermissionPolicy 配置 |
| 会话无法恢复 | 验证 session_store 路径 |
| 插件加载失败 | 检查插件初始化日志 |

---

## 19. 贡献指南

### 19.1 代码规范

- 遵循 Rust 标准代码风格
- 使用 `cargo fmt` 格式化代码
- 使用 `cargo clippy` 检查代码
- 所有公共 API 必须有文档注释

### 19.2 提交规范

```
feat: 新功能
fix: 修复问题
docs: 文档更新
style: 代码格式调整
refactor: 重构
test: 测试相关
chore: 构建/工具相关
```

### 19.3 Pull Request 流程

1. Fork 仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: 添加 amazing 功能'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

### 19.4 文档更新

- 更新 API 文档
- 更新用户手册
- 添加示例代码
- 更新 CHANGELOG

---

## 附录

### A. 配置参考

完整的 `agent.hcl` 配置示例见项目根目录的 `agent.example.hcl`。

### B. API 参考

详细 API 文档：[a3s.dev/docs/code](https://a3s.dev/docs/code)

### C. 示例代码

更多示例见 `examples/` 目录。

### D. 相关资源

- [官方文档](https://a3s.dev/docs/code)
- [GitHub 仓库](https://github.com/a3s-lab/a3s-code)
- [Crates.io](https://crates.io/crates/a3s-code-core)
- [PyPI](https://pypi.org/project/a3s-code)
- [npm](https://www.npmjs.com/package/@a3s-lab/code)

---

**许可证**: MIT  
**版本**: 详见各 SDK 的 CHANGELOG

*本手册最后更新: 2026-03-24*
