# A3S Code

**A harness-driven runtime for coding agents, SDKs, filesystem-first agents,
and the `a3s code` terminal experience.**

A3S Code is the agent-loop runtime behind `a3s code` and the Rust, Node.js,
and Python SDKs. It lets the harness own the parts a coding agent should not
improvise: context assembly, tool visibility, permissions, delegation,
workspace access, persistence, verification evidence, and event replay.

[![crates.io](https://img.shields.io/crates/v/a3s-code-core)](https://crates.io/crates/a3s-code-core)
[![PyPI](https://img.shields.io/pypi/v/a3s-code)](https://pypi.org/project/a3s-code/)
[![npm](https://img.shields.io/npm/v/@a3s-lab/code)](https://www.npmjs.com/package/@a3s-lab/code)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## What It Is

A3S Code is a library runtime, not a hosted agent service and not the TUI
itself. The runtime provides a small, observable execution loop:

```text
prompt
  -> context assembly
  -> optional planning
  -> selected tools / delegated child tasks
  -> permission and confirmation checks
  -> execution
  -> events, artifacts, and verification evidence
  -> compaction and persistence
```

The default session runs against a local workspace. Embedders can replace the
workspace, memory, session store, security provider, LLM client, MCP manager,
hooks, and budget guard with typed objects instead of raw backend strings.

The surrounding A3S project uses that runtime across these layers:

| Name | What it is | Primary repo |
| --- | --- | --- |
| **A3S Code / `a3s-code`** | Rust core plus Node.js and Python SDKs for embedding coding-agent sessions in products. | <https://github.com/A3S-Lab/Code> |
| **`a3s code` TUI** | The interactive terminal coding agent. It is shipped by the `a3s` CLI, drives `a3s-code-core`, and renders the event stream with the `a3s-tui` framework. | <https://github.com/A3S-Lab/Cli> |
| **`a3s-tui`** | Terminal UI framework used by the CLI. It is a UI layer, not the agent runtime. | <https://github.com/A3S-Lab/TUI> |
| **A3S monorepo** | Product docs, submodule pins, release orchestration, and related crates. | <https://github.com/A3S-Lab/a3s> |

Use `a3s code` when you want a ready interactive coding agent in a terminal.
Use the A3S Code SDKs when you are building your own harness, IDE extension,
server worker, workflow runner, or product UI.

## Core Surfaces

A3S Code is deliberately split into surfaces so products can adopt the runtime
without inheriting the terminal UX:

| Surface | What you use | What it gives you |
| --- | --- | --- |
| Runtime sessions | Rust core, Node.js SDK, Python SDK | `send`, `run`, `stream`, direct tools, cancellation, persistence, verification, lifecycle cleanup. |
| Filesystem-first agents | `AGENTS.md`, `agent.acl`, `.a3s/agents/`, `.a3s/skills/`, AgentDir | Git-reviewable instructions, model policy, worker roles, reusable skills, directory-scoped tools, and schedules. |
| Terminal app | `a3s code` from the `a3s` CLI | Ready TUI with streamed events, tool activity, approvals, memory/git/file panels, and session controls. |
| Host extension points | Typed stores, workspaces, providers, hooks, MCP/AHP, command registry | Product-specific storage, sandboxing, tools, controls, observability, and slash-command behavior without forking the loop. |

## Capability Map

| Area | Current capability |
| --- | --- |
| Agent API | `Agent` and `AgentSession` expose `send`, `run`, `stream`, object-shaped requests, explicit-history calls, attachments, direct tool calls, run state, cancellation, persistence, and lifecycle cleanup. |
| Agent loop | Streaming text/tool events, tool-call repair, bounded parse-error recovery, compaction, planning modes, budget guards, active-tool state, and deterministic direct calls. |
| Config | ACL config files or inline ACL source; provider/model selection; skill and agent directories; storage, search, MCP, and delegation settings. |
| LLM clients | Built-in Anthropic, OpenAI-compatible, and Zhipu-compatible clients, plus `SessionOptions::with_llm_client(...)` for host-supplied clients. |
| Tools | Files, search, shell, git, web fetch/search, batch, structured output, programmatic QuickJS tool calling, skills, MCP tools, and task delegation. |
| Commands | Built-in slash commands and a host command registry for product-specific `/command` handlers; the TUI layers its own terminal commands over the same session path. |
| Filesystem-first | `AGENTS.md`, `.a3s/agents/`, `.a3s/skills/`, AgentDir `instructions.md`, `agent.acl`, `tools/`, and `schedules/` make agent behavior reviewable, diffable, and reusable as files. |
| Context | Project instructions, prompt slots, filesystem context, recent-file/ripgrep providers, memory recall, skills, MCP, and run observations. |
| Safety | Permission policies, human confirmation, workspace path checks, tool timeouts, sandbox handle for `bash`, security providers, prompt boundary injection, and redaction-aware logging paths. |
| Delegation | Built-in worker roles, custom Markdown/YAML agents, `task`, `parallel_task`, automatic delegation controls, and subagent task tracking. |
| Orchestration | Programmable fan-out, pipelines, resumable checkpoints, workflow phases, loop caps, and shared workflow budget ledgers. |
| Serving | `serveAgentDir` / `serve_agent_dir` load AgentDir schedules as full harness turns with stable `schedule:<name>` sessions. |
| Workspaces | Local filesystem by default; typed workspace services for custom hosts; optional S3-compatible backend and HTTP/JSON remote-git backend. |
| Persistence | Memory and file session stores, session IDs, auto-save, run snapshots/events, trace artifacts, memory store integration, loop/workflow checkpoints, and retention caps. |
| Verification | `verifyCommands`, verification presets, structured reports, summaries, run events, artifacts, and trace APIs for replayable evidence. |
| Integration | MCP client/manager, AHP hook integration, lifecycle hooks, lane queue options, OpenTelemetry feature flag, Node SDK, Python SDK, and the `a3s code` TUI. |

## Install

### Interactive TUI

Install the `a3s` CLI when you want the terminal app:

```bash
brew install A3S-Lab/tap/a3s

# or from crates.io
cargo install a3s

# or from source
cargo install --git https://github.com/A3S-Lab/Cli
```

Run it inside the workspace you want the agent to inspect:

```bash
a3s code
a3s code resume <session-id>
a3s code update
```

`a3s code` discovers config in this order: `A3S_CONFIG_FILE`, then
`.a3s/config.acl` walking upward from the current directory, then
`~/.a3s/config.acl`. First launch can create a starter user config. Treat any
real config as local credential-bearing state; commit templates, not credentials.

### SDKs

Install the SDK package when you are embedding A3S Code in another product:

```bash
npm install @a3s-lab/code
pip install a3s-code
cargo add a3s-code-core
```

The Python package is a small bootstrap that downloads the matching native
extension from the release manifest and verifies the downloaded artifact hash.
See the release notes for the current hardening plan and offline-mode details.

## Filesystem-First Model

A3S Code treats durable agent behavior as files before APIs. The point is not
magic discovery; it is code review. Roles, schedules, reusable skills, runtime
policy, and worker definitions can live beside the repository they operate on.

This is the mode to reach for when agent behavior should survive a process,
move with a repository, and be reviewed like normal engineering work. The SDK
can still construct everything programmatically; filesystem-first is the
portable source-of-truth form.

```text
repo/
├── AGENTS.md          # project instructions loaded into context
├── agent.acl          # model/provider/runtime policy for SDK sessions
└── .a3s/
    ├── agents/        # worker/subagent definitions
    └── skills/        # reusable project skills

release-agent/
├── instructions.md    # AgentDir main-agent role slot
├── agent.acl          # optional AgentDir runtime config
├── skills/            # AgentDir-private skills
├── tools/             # AgentDir MCP/script tools
└── schedules/         # cron-driven recurring turns
```

There are two related but different conventions:

- Workspace files such as `AGENTS.md`, `.a3s/agents/`, and `.a3s/skills/`
  shape interactive sessions, TUI runs, and SDK sessions bound to a repository.
- An **AgentDir** is a primary durable agent directory. `instructions.md` is
  required; optional `agent.acl`, `skills/`, `tools/`, and `schedules/` are
  loaded by `AgentDir::load` / `serve_agent_dir` for recurring agent work.

These files do not override harness boundaries. Permissions, HITL confirmation,
tool visibility, response contracts, sandboxing, and verification remain part
of the runtime execution path.

## A3S Code TUI

The `a3s code` TUI is the reference terminal application built on top of this
runtime. It uses `a3s-code-core::AgentSession::stream()` as the source of truth
and renders `AgentEvent` updates as a live transcript with tool activity,
planning state, side questions, memory, git/file panels, and inline approval
prompts for gated tool calls.

Important distinction:

- **A3S Code runtime**: core crates and SDK APIs for product developers.
- **`a3s code` TUI**: one application that embeds the runtime and supplies an
  opinionated terminal workflow.

The TUI adds UI affordances such as `/model`, `/config`, `/init`, `/btw`,
`/compact`, `/goal`, `/loop`, `/git`, `/memory`, `/ide`, `/top`, and `/update`.
Those commands are CLI features layered on the runtime; SDK embedders can build
different controls over the same session, tool, event, persistence, and
verification APIs.

## Minimal Config

Use ACL for product configuration. Keep real keys and private base URLs in the
environment; commit templates, not local secrets.

```acl
default_model = "provider/model-id"
max_parallel_tasks = 4
auto_parallel = false

providers "provider" {
  apiKey = env("PROVIDER_API_KEY")
  baseUrl = env("PROVIDER_BASE_URL")

  models "model-id" {
    tool_call = true
    limit = {
      context = 128000
      output = 4096
    }
  }
}

agent_dirs = ["./.a3s/agents"]
skill_dirs = ["./skills"]
storage_backend = "file"
sessions_dir = ".a3s/sessions"

auto_delegation {
  enabled                 = false
  auto_parallel           = false
  allow_manual_delegation = true
  min_confidence          = 0.72
  max_tasks               = 4
}
```

`storage_backend = "file"` is only useful for local session persistence when it
has a `sessions_dir`; otherwise pass a typed `FileSessionStore` from the SDK.

Do not commit `.a3s/config.acl`, local provider URLs, access tokens, API keys,
or real tenant/user identifiers. Prefer `env("...")` in examples and CI.

## Quick Start

### Node.js

```ts
import { Agent } from '@a3s-lab/code';

const agent = await Agent.create('agent.acl');
const session = agent.session('/path/to/workspace', {
  builtinSkills: true,
  planningMode: 'auto',
  permissionPolicy: {
    allow: ['read(*)', 'grep(*)', 'glob(*)'],
    ask: ['bash(*)', 'write(*)'],
    deny: ['write(**/.env*)', 'bash(rm -rf*)'],
    defaultDecision: 'ask',
    enabled: true,
  },
});

const result = await session.send('Find the authentication entry points.');
console.log(result.text);

session.close();
await agent.close();
```

### Python

```python
from a3s_code import Agent, PermissionPolicy, SessionOptions

agent = Agent.create("agent.acl")

opts = SessionOptions()
opts.builtin_skills = True
opts.planning_mode = "auto"
opts.permission_policy = PermissionPolicy(
    allow=["read(*)", "grep(*)", "glob(*)"],
    ask=["bash(*)", "write(*)"],
    deny=["write(**/.env*)", "bash(rm -rf*)"],
    default_decision="ask",
)

session = agent.session("/path/to/workspace", opts)
result = session.send("Find the authentication entry points.")
print(result.text)

session.close()
agent.close()
```

### Rust

```rust
use a3s_code_core::{Agent, AgentEvent, SessionOptions};

# async fn run() -> anyhow::Result<()> {
let agent = Agent::new("agent.acl").await?;
let session = agent.session(
    "/path/to/workspace",
    Some(SessionOptions::new().with_planning(true)),
)?;

let result = session.send("Find the authentication entry points.", None).await?;
println!("{}", result.text);

let (mut rx, _handle) = session.stream("Summarize the test strategy.", None).await?;
while let Some(event) = rx.recv().await {
    match event {
        AgentEvent::TextDelta { text } => print!("{text}"),
        AgentEvent::End { .. } => break,
        _ => {}
    }
}
# Ok(())
# }
```

## Direct Tools

The SDKs expose direct host calls for product code that wants deterministic
tool use without asking the model to choose the tool:

```ts
const source = await session.readFile('src/main.rs'); // string
const hits = await session.grep('PermissionPolicy'); // ripgrep text
const files = await session.glob('**/*.rs'); // string[]
const testOutput = await session.bash('cargo test -p a3s-code-core'); // string

const structured = await session.tool('generate_object', {
  schema: {
    type: 'object',
    required: ['summary'],
    properties: { summary: { type: 'string' } },
  },
  prompt: 'Summarize the current task in one sentence.',
  schema_name: 'task_summary',
});
if (structured.exitCode !== 0) {
  throw new Error(structured.output);
}

const { object } = JSON.parse(structured.output);
console.log(source.length, hits.split('\n').filter(Boolean).length, files.length);
console.log(testOutput);
console.log(object.summary);
```

Direct host calls are privileged. Gate them in the embedding application before
exposing them to end users. Typed read/search/shell helpers return simple
values; generic tools such as `tool`, `writeFile`, `ls`, `git`, `task`,
`tasks`, and `program` return `ToolResult` with `output`, `exitCode`, and
optional metadata.

## Programmatic Tool Calling

High-frequency tool chains can run inside the embedded QuickJS program tool.
This reduces model round trips while preserving the same tool registry, limits,
workspace boundary, artifacts, and result shape. When the model invokes
`program` inside an agent turn, normal permission, confirmation, hook/AHP, and
trace paths apply. When product code calls `session.program(...)` directly, it
is a host control-plane call and should be authorized before invoking the SDK.

```ts
const result = await session.program({
  source: `
    export default async function run(ctx, inputs) {
      const hits = await ctx.grep(inputs.query, { glob: '*.rs' });
      const files = await ctx.glob('crates/**/*.rs');
      return { hits, files: files.slice(0, 20) };
    }
  `,
  inputs: { query: 'PermissionPolicy' },
  allowedTools: ['grep', 'glob'],
  limits: { timeoutMs: 30000, maxToolCalls: 20, maxOutputBytes: 65536 },
});
if (result.exitCode !== 0) {
  throw new Error(result.output);
}

const metadata = result.metadataJson ? JSON.parse(result.metadataJson) : {};
console.log(metadata.script_result);
console.log(metadata.program?.tool_calls ?? []);
```

## Delegation And Orchestration

Model-driven delegation uses `task` and `parallel_task`; host-driven
orchestration uses deterministic SDK calls.

```ts
const delegated = await session.task({
  agent: 'explore',
  description: 'Find auth entry points',
  prompt: 'Inspect the workspace and return file-level evidence.',
});
if (delegated.exitCode !== 0) {
  throw new Error(delegated.output);
}

const outcomes = await session.parallel([
  { taskId: 'plan', agent: 'plan', description: 'Plan change', prompt: 'Plan the fix.' },
  { taskId: 'review', agent: 'review', description: 'Review risk', prompt: 'Review current diff.' },
]);

for (const outcome of outcomes) {
  console.log(outcome.taskId, outcome.success, outcome.output);
}
```

The orchestration layer includes parallel fan-out, pipelines, resumable
checkpoints, workflow phases, `execute_loop` with a mandatory hard cap, and a
shared workflow token-budget guard. It defines grammar and bookkeeping; a host
platform can still decide placement.

`session.task(...)` and `session.tasks(...)` are model-driven delegation
wrappers over the `task` and `parallel_task` tools, so they return
`ToolResult`. `session.parallel(...)`, `session.pipeline(...)`, and
`session.parallelResumable(...)` are host-driven orchestration primitives, so
they return typed step outcomes instead.

## Workspace Backends

By default, built-in tools operate on the local filesystem. Hosts can pass a
`WorkspaceServices` object so the same tool names target a browser workspace,
remote runner, DFS, object storage, or another controlled environment.

Tool visibility follows backend capabilities: file tools need read/write,
`grep` and `glob` need search, `bash` needs a command runner, and `git` needs a
workspace git provider. With the `s3` Cargo feature, file tools can target an
S3-compatible backend; remote git can be attached separately through the
HTTP/JSON `RemoteGitBackend`.

## Verification And Replay

Every turn can produce typed run snapshots, ordered run events, active-tool
state, verification reports, and compact artifact references. Product UIs and
harnesses should consume those APIs instead of scraping the final answer text.

Useful surfaces include:

```ts
const runs = await session.runs();
const latest = runs.at(-1);
if (latest) {
  console.log(await session.runSnapshot(latest.id));
  console.log(await session.runEvents(latest.id));
  console.log(await session.activeTools());
}
```

## Testing Evidence

The repository contains both hermetic tests and opt-in real-provider tests.
Examples:

```bash
cargo test -p a3s-code-core
cargo test -p a3s-code-core --test test_prompt_boundaries_and_log_redaction
node scripts/docs_api_contract_smoke.mjs
```

Real LLM tests are ignored by default and require explicit provider
configuration through `A3S_CONFIG_FILE` or the local git-ignored
`.a3s/config.acl`:

```bash
A3S_CONFIG_FILE=/path/to/local/config.acl \
  cargo test -p a3s-code-core --test test_real_config_env_integration -- --ignored --nocapture

A3S_CONFIG_FILE=/path/to/local/config.acl \
  cargo test -p a3s-code-core --test test_orchestration_real_llm -- --ignored --nocapture
```

Do not paste real provider values into test commands, test logs, commits, or
pull-request descriptions.

## Documentation

Full guides live in the docs site:

- [A3S Code docs](https://a3s-lab.github.io/a3s/docs/code)
- [A3S Code TUI](https://a3s-lab.github.io/a3s/docs/code/tui)
- [Filesystem-First](https://a3s-lab.github.io/a3s/docs/code/filesystem-first)
- [API Contract](https://a3s-lab.github.io/a3s/docs/code/api-contract)
- [Sessions](https://a3s-lab.github.io/a3s/docs/code/sessions)
- [Commands](https://a3s-lab.github.io/a3s/docs/code/commands)
- [Tools](https://a3s-lab.github.io/a3s/docs/code/tools)
- [Verification](https://a3s-lab.github.io/a3s/docs/code/verification)
- [Providers](https://a3s-lab.github.io/a3s/docs/code/providers)
- [Workspace Backends](https://a3s-lab.github.io/a3s/docs/code/workspace-backends)
- [Orchestration](https://a3s-lab.github.io/a3s/docs/code/orchestration)
- [Security](https://a3s-lab.github.io/a3s/docs/code/security)
- [Hooks](https://a3s-lab.github.io/a3s/docs/code/hooks)
- [Agent Directory](https://a3s-lab.github.io/a3s/docs/code/agent-dir)

## Development

Run commands from this crate workspace, not from the monorepo root:

```bash
cargo fmt --all
cargo test -p a3s-code-core
cargo clippy -p a3s-code-core -- -D warnings
```

Build SDK crates individually when needed:

```bash
cargo build -p a3s-code-node
cargo build -p a3s-code-py
```

## License

MIT
