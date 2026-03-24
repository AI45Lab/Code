# A3S Code User & Developer Guide

> **Agentic Agent Framework** - A3S Code is a Rust library with native Python and Node.js bindings

---

## Table of Contents

- [Part 1: User Guide](#part-1-user-guide)
  - [1. Introduction](#1-introduction)
  - [2. Installation & Configuration](#2-installation--configuration)
  - [3. Quick Start](#3-quick-start)
  - [4. Core Concepts](#4-core-concepts)
  - [5. Tools System](#5-tools-system)
  - [6. Skills System](#6-skills-system)
  - [7. Multi-Agent Collaboration](#7-multi-agent-collaboration)
  - [8. Security & Permissions](#8-security--permissions)
  - [9. Slash Commands](#9-slash-commands)
  - [10. Scheduled Tasks](#10-scheduled-tasks)
  - [11. Session Management](#11-session-management)
- [Part 2: Developer Guide](#part-2-developer-guide)
  - [12. Architecture Overview](#12-architecture-overview)
  - [13. Development Environment](#13-development-environment)
  - [14. Core Modules](#14-core-modules)
  - [15. Extension Development](#15-extension-development)
  - [16. Hook System](#16-hook-system)
  - [17. Plugin Development](#17-plugin-development)
  - [18. Testing & Debugging](#18-testing--debugging)
  - [19. Contributing Guidelines](#19-contributing-guidelines)

---

# Part 1: User Guide

## 1. Introduction

A3S Code is a powerful **Agentic Agent Framework** that enables Large Language Models (LLMs) to:

- **File Operations** - Read, write, edit, and patch files
- **Code Search** - Search codebases using Grep, Glob, and more
- **Command Execution** - Run shell commands in sandboxed environments
- **Web Access** - Web scraping and search capabilities
- **Task Delegation** - Distribute tasks to sub-agents or multi-agent teams

### Supported Platforms

| Platform | Installation |
|----------|-------------|
| Python | `pip install a3s-code` |
| Node.js | `npm install @a3s-lab/code` |
| Rust | `cargo add a3s-code-core` |

### Supported LLM Providers

- **Anthropic** (Claude series)
- **OpenAI** (GPT series)
- **DeepSeek**
- **Kimi** (Moonshot)
- **Together**
- **Groq**

## 2. Installation & Configuration

### 2.1 Python Installation

```bash
pip install a3s-code
```

### 2.2 Node.js Installation

```bash
npm install @a3s-lab/code
```

### 2.3 Agent Configuration (agent.hcl)

Create `agent.hcl` configuration file:

```hcl
# Default model
default_model = "anthropic/claude-sonnet-4-20250514"

# LLM Provider Configuration
providers {
  name    = "anthropic"
  api_key = env("ANTHROPIC_API_KEY")
}

providers {
  name    = "openai"
  api_key = env("OPENAI_API_KEY")
}

# Storage backend: "memory", "file", or "custom"
storage_backend = "file"

# Sessions directory
sessions_dir = "./sessions"

# Skill directories
skill_dirs = ["./skills"]

# Maximum tool execution rounds
max_tool_rounds = 50
```

### 2.4 Environment Variables

```bash
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
```

## 3. Quick Start

### 3.1 Python Example

```python
from a3s_code import Agent

# Create agent
agent = Agent.create("agent.hcl")

# Create session
session = agent.session("/my-project")

# Send request
result = session.send("Analyze project architecture")
print(result.text)
```

### 3.2 Node.js Example

```typescript
import { Agent } from '@a3s-lab/code';

const agent = await Agent.create('agent.hcl');
const session = agent.session('/my-project');

const result = await session.send('Analyze project architecture');
console.log(result.text);
```

### 3.3 First Tasks

```python
# Find authentication error handling
result = session.send("Find all places handling authentication errors")

# Review code quality
result = session.send("Review main.py code quality and suggest improvements")

# Run tests
result = session.send("Run test suite and report results")
```

## 4. Core Concepts

### 4.1 Architecture Layers

```
Agent (Config + Provider Registry)
  └── Session (Workspace + Tools + LLM)
        └── AgentLoop (Turn-based Execution)
              ├── LlmClient      → Send messages, receive tool calls
              ├── ToolExecutor   → Run tools, enforce permissions
              ├── SkillRegistry  → Inject skills into system prompt
              └── PluginManager  → Load optional tool+skill bundles
```

### 4.2 Core Components

| Component | Description |
|-----------|-------------|
| **Agent** | Top-level configuration and factory |
| **Session** | Workspace container with tools, LLM client, and state |
| **AgentLoop** | Execution loop managing LLM and tool interactions |
| **Skill** | Markdown files defining behavior and capabilities |
| **Tool** | Functions the agent can invoke |

### 4.3 SessionOptions Configuration

```python
from a3s_code import Agent, SessionOptions

opts = SessionOptions()

# Specify model
opts.model = "openai/gpt-4o"

# Enable built-in skills
opts.builtin_skills = True

# Load custom skills
opts.skill_dirs = ["./skills"]

# Add plugins
from a3s_code import AgenticSearch, AgenticParse
opts.plugins = [AgenticSearch(), AgenticParse()]

session = agent.session(".", opts)
```


## 5. Tools System

### 5.1 Built-in Tools (16 total)

#### File Tools

| Tool | Description | Example |
|------|-------------|---------|
| `read` | Read file content | `read: /path/to/file.py` |
| `write` | Write file | `write: /path/to/file.py` |
| `edit` | Edit file | `edit: /path/to/file.py` |
| `patch` | Apply patch | `patch: /path/to/file.py` |

#### Search Tools

| Tool | Description | Example |
|------|-------------|---------|
| `grep` | Text search | `grep: "function name"` |
| `glob` | File matching | `glob: "**/*.py"` |
| `ls` | Directory listing | `ls: /path/to/dir` |

#### Other Tools

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands |
| `web_fetch` | Fetch web page content |
| `web_search` | Perform web search |
| `git_worktree` | Git worktree operations |

### 5.2 Delegation Tools

| Tool | Description |
|------|-------------|
| `task` | Delegate to single agent |
| `parallel_task` | Delegate multiple tasks in parallel |
| `run_team` | Run agent team |
| `batch` | Batch execute tasks |
| `Skill` | Invoke specific skill |

### 5.3 Plugin Tools

```python
# Enable AgenticSearch - Natural language code search
from a3s_code import AgenticSearch
opts.plugins = [AgenticSearch()]

# Enable AgenticParse - Enhanced parsing
from a3s_code import AgenticParse
opts.plugins = [AgenticParse()]
```

## 6. Skills System

Skills are Markdown files that shape LLM behavior.

### 6.1 Skill File Structure

```markdown
---
name: safe-reviewer
description: Review code without modifying files
allowed-tools: "read(*), grep(*), glob(*)"
---

Review code in the workspace. You may read and search files,
but you must not write, edit, or execute anything.

Review checklist:
1. Check for potential security issues
2. Verify error handling
3. Evaluate code readability
4. Provide improvement suggestions
```

### 6.2 Using Skills

```python
opts = SessionOptions()
opts.skill_dirs = ["./skills"]
opts.builtin_skills = True  # Enable built-in skills
session = agent.session(".", opts)
```

### 6.3 Built-in Skills

| Skill | Function |
|-------|----------|
| `agentic-search` | Intelligent code search |
| `code-search` | Code search assistance |
| `code-review` | Code review |
| `explain-code` | Code explanation |
| `find-bugs` | Bug detection |
| `builtin-tools` | Tool usage guidance |
| `delegate-task` | Task delegation |
| `find-skills` | Skill discovery |

## 7. Multi-Agent Collaboration

### 7.1 Single Sub-agent

```python
result = session.send('task: explore codebase and summarize architecture')
```

### 7.2 Parallel Tasks

```python
result = session.send('parallel_task: [audit security, check performance, review tests]')
```

### 7.3 Agent Teams

```python
# Run agent team (lead decomposes -> workers execute -> reviewer validates)
result = session.send('run_team: refactor authentication module')
```

### 7.4 Agent Types

| Type | Description |
|------|-------------|
| `explore` | Read-only exploration |
| `general` | Full capabilities |
| `plan` | Analysis only |

## 8. Security & Permissions

### 8.1 Permission Policy

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

### 8.2 Human-in-the-Loop (HITL)

```python
# Prompt confirmation before each tool call
opts.hitl_enabled = True
```

### 8.3 Security Features

| Feature | Description |
|---------|-------------|
| **Explicit Permissions** | Deny by default, explicit allow required |
| **Human Confirmation** | Prompt before tool execution |
| **Skill Restrictions** | `allowed-tools` limits callable tools |
| **AHP Integration** | Runtime interception and sanitization |
| **Auto-compact** | Auto compress context before token limits |
| **Circuit Breaker** | Stop after 3 consecutive failures |


## 9. Slash Commands

Type `/help` in any session to see available commands:

| Command | Description |
|---------|-------------|
| `/help` | List available commands |
| `/model [provider/model]` | Show or switch current model |
| `/cost` | Show token usage and estimated cost |
| `/clear` | Clear conversation history |
| `/compact` | Manually trigger context compaction |
| `/tools` | List registered tools |
| `/loop [interval] <prompt>` | Schedule recurring prompt |
| `/cron-list` | List scheduled tasks |
| `/cron-cancel <id>` | Cancel scheduled task |

### 9.1 Custom Commands

```python
session.register_command(
    "status", 
    "Show status", 
    lambda args, ctx: f"Model: {ctx['model']}"
)
result = session.send("/status")
```

## 10. Scheduled Tasks

### 10.1 Using Slash Commands

```python
# Check test status every 5 minutes
session.send('/loop 5m check if tests are still passing')
```

### 10.2 Programmatic API

```python
# Schedule task (300 second interval)
task_id = session.schedule_task('summarize git log since last check', 300)

# List scheduled tasks
tasks = session.list_scheduled_tasks()

# Cancel task
session.cancel_scheduled_task(task_id)
```

### 10.3 Interval Syntax

- `30s` - 30 seconds
- `5m` - 5 minutes
- `2h` - 2 hours
- `1d` - 1 day

Limit: Max 50 tasks per session, auto-expire after 3 days.

## 11. Session Management

### 11.1 BTW - Side Questions

Ask side questions without affecting conversation history:

```python
btw = session.btw("What is PostgreSQL default port?")
print(btw.answer)        # "5432"
print(btw.total_tokens)  # Token usage for this query only
# Main conversation continues - btw question not in history
```

### 11.2 Session Persistence

```python
from a3s_code import SessionOptions, FileSessionStore, FileMemoryStore

opts = SessionOptions()
opts.session_store = FileSessionStore('./sessions')
opts.memory_store = FileMemoryStore('./memory')
opts.session_id = 'my-session'
opts.auto_save = True

session = agent.session(".", opts)

# Resume session
resumed = agent.resume_session('my-session', opts)
```

### 11.3 Multi-Provider Switching

```python
# Switch model per session
session = agent.session(".", model="openai/gpt-4o")
```

---

# Part 2: Developer Guide

## 12. Architecture Overview

### 12.1 System Architecture

```
A3S Code
├── Python SDK (PyO3)
├── Node.js SDK (NAPI)
└── Rust Core
    ├── Agent (Configuration)
    ├── Session (State Management)
    └── AgentLoop (Execution)
        ├── LlmClient
        ├── ToolExecutor
        ├── SkillRegistry
        └── PluginManager
```

### 12.2 Core Modules

| Module | Path | Description |
|--------|------|-------------|
| `agent.rs` | `core/src/` | Agent main logic |
| `session.rs` | `core/src/session/` | Session management |
| `tools/` | `core/src/tools/` | Tool implementations |
| `skills/` | `core/src/skills/` | Skill system |
| `llm/` | `core/src/llm/` | LLM clients |
| `permissions.rs` | `core/src/` | Permission control |
| `hooks/` | `core/src/hooks/` | Hook system |

## 13. Development Environment

### 13.1 Prerequisites

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python (for Python SDK)
python -m pip install maturin

# Node.js (for Node.js SDK)
npm install -g napi-rs
```

### 13.2 Clone and Build

```bash
git clone <repository-url>
cd a3s-code

# Build core
cargo build --release

# Build Python SDK
cd sdk/python
maturin develop

# Build Node.js SDK
cd sdk/node
npm install
npm run build
```

### 13.3 Development Tools

```bash
# Run tests
cargo test

# Linting
cargo clippy

# Formatting
cargo fmt

# Use just for tasks
just --list
```


## 14. Core Modules

### 14.1 Agent Module (`agent.rs`)

```rust
pub struct Agent {
    config: Config,
    provider_registry: ProviderRegistry,
}

impl Agent {
    pub fn create(config_path: &str) -> Result<Self>;
    pub fn session(&self, workspace: &str) -> Session;
    pub fn resume_session(&self, session_id: &str) -> Result<Session>;
}
```

### 14.2 Session Module (`session/`)

```rust
pub struct Session {
    id: String,
    workspace: PathBuf,
    tool_executor: ToolExecutor,
    llm_client: LlmClient,
    skill_registry: SkillRegistry,
}

impl Session {
    pub fn send(&mut self, prompt: &str) -> Result<Response>;
    pub fn btw(&self, question: &str) -> Result<BtwResponse>;
    pub fn schedule_task(&mut self, task: &str, interval_secs: u64) -> String;
}
```

### 14.3 Tool Module (`tools/`)

```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn execute(&self, input: ToolInput) -> Result<ToolOutput>;
}
```

## 15. Extension Development

### 15.1 Creating Custom Tools

```rust
use a3s_code_core::tools::{Tool, ToolInput, ToolOutput};

pub struct MyTool;

impl Tool for MyTool {
    fn name(&self) -> &str { "my_tool" }
    fn description(&self) -> &str { "My custom tool description" }
    fn execute(&self, input: ToolInput) -> Result<ToolOutput> {
        Ok(ToolOutput::new("result"))
    }
}
```

### 15.2 Creating Custom Skills

Create Markdown file in `skills/` directory:

```markdown
---
name: my-skill
description: My custom skill
allowed-tools: "read(*), grep(*)"
---

# My Skill

Detailed description for LLM to use when executing tasks.
```

## 16. Hook System

### 16.1 Available Hook Events

| Event | Description | Blockable |
|-------|-------------|-----------|
| `PreToolUse` | Before tool use | Yes |
| `PostToolUse` | After tool use | No |
| `GenerateStart` | Before generation | Yes |
| `GenerateEnd` | After generation | No |
| `SessionStart` | Session start | No |
| `SessionEnd` | Session end | No |

### 16.2 Implementing HookHandler

```rust
use a3s_code::HookHandler;

struct MyHook;

impl HookHandler for MyHook {
    fn pre_tool_use(&self, tool_name: &str, tool_input: &Value, ctx: &Context) -> HookResult {
        if tool_name == "bash" && tool_input.contains("rm -rf") {
            return HookResult::block("Refusing destructive command");
        }
        HookResult::continue_()
    }
}
```

## 17. Plugin Development

### 17.1 Plugin Structure

```rust
pub trait Plugin: Send + Sync {
    fn name(&self) -> &str;
    fn initialize(&mut self, ctx: &PluginContext) -> Result<()>;
    fn tools(&self) -> Vec<Box<dyn Tool>>;
    fn skills(&self) -> Vec<Skill>;
}
```

## 18. Testing & Debugging

### 18.1 Unit Tests

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

### 18.2 Debugging Tips

```bash
# Enable verbose logs
export RUST_LOG=debug
export A3S_DEBUG=1
```

## 19. Contributing Guidelines

### 19.1 Code Style

- Follow Rust standard style
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Document all public APIs

### 19.2 Commit Convention

```
feat: new feature
fix: bug fix
docs: documentation
style: formatting
refactor: refactoring
test: testing
chore: build/tools
```

---

**License**: MIT  
**Version**: See CHANGELOG in each SDK

*Last Updated: 2026-03-24*
