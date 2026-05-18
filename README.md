# A3S Code

**A harness-driven runtime for coding agents.**

A3S Code is a Rust agent runtime with Python and Node.js bindings. It is built around a simple belief:

> A coding agent becomes reliable when the harness controls context, actions, safety, and verification.

The model should reason. The harness should decide what context is load-bearing, which tools are visible, which actions are safe, and how completion is verified.

[![crates.io](https://img.shields.io/crates/v/a3s-code-core)](https://crates.io/crates/a3s-code-core)
[![PyPI](https://img.shields.io/pypi/v/a3s-code)](https://pypi.org/project/a3s-code/)
[![npm](https://img.shields.io/npm/v/@a3s-lab/code)](https://www.npmjs.com/package/@a3s-lab/code)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## Why

Most coding agents fail for boring reasons:

- too many tools are injected into every prompt
- raw search results, test logs, and delegated-task transcripts flood the context
- memory, skills, MCP, hooks, and project hints all inject context through separate paths
- safety is split across permissions, confirmations, skills, and custom guards
- agents stop after "I changed it" instead of proving the change works

A3S Code treats the agent as an execution system:

```text
Intent -> Context -> Action -> Observation -> Verification -> Compaction
```

Everything else is an extension of that loop.

---

## Install

```bash
# Python
pip install a3s-code

# Node.js
npm install @a3s-lab/code
```

Rust users can depend on `a3s-code-core`.

---

## Quick Start

Create `agent.acl`:

```acl
default_model = "anthropic/claude-sonnet-4-20250514"

providers "anthropic" {
  apiKey = env("ANTHROPIC_API_KEY")
}
```

Send a prompt:

```python
from a3s_code import Agent

agent = Agent.create("agent.acl")
session = agent.session("/my-project")

result = session.send({"prompt": "Summarize how auth errors are handled."})
print(result.text)
```

```typescript
import { Agent } from '@a3s-lab/code';

const agent = await Agent.create('agent.acl');
const session = agent.session('/my-project');

const result = await session.send({ prompt: 'Summarize how auth errors are handled.' });
console.log(result.text);
session.close();
```

## Main APIs At A Glance

The same surface area is available from Python and Node. Both are shown below — Node uses `camelCase`, Python uses `snake_case`, and the call shapes match.

```python
from a3s_code import (
    Agent,
    SessionOptions,
    PermissionPolicy,
    ConfirmationPolicy,
    WorkerAgentSpec,
    FileMemoryStore,
    FileSessionStore,
    HttpTransport,
)

# 1. Configure a session — typed extension options, not raw flags.
opts = SessionOptions()
opts.skill_dirs = ["./skills"]
opts.planning_mode = "auto"                              # "auto" | "enabled" | "disabled"
opts.permission_policy = PermissionPolicy(
    allow=["read(*)", "grep(*)", "glob(*)"],
    ask=["bash(*)", "write(*)"],
    deny=["bash(rm -rf *)"],
    default_decision="ask",
)
opts.confirmation_policy = ConfirmationPolicy(
    enabled=True, default_timeout_ms=30_000, timeout_action="reject",
)
opts.memory_store = FileMemoryStore("./memory")
opts.session_store = FileSessionStore("./sessions")
opts.session_id = "my-session"
opts.auto_save = True
opts.ahp_transport = HttpTransport("http://localhost:8080/ahp")

agent = Agent.create("agent.acl")
session = agent.session("/my-project", opts)

# 2. Send / stream — string or object-shaped requests.
result = session.send({"prompt": "Refactor the auth module"})
print(result.text, result.verification_status)

for event in session.stream({"prompt": "Continue the refactor"}):
    if event.event_type == "text_delta":
        print(event.text, end="", flush=True)

# 3. Direct tools (bypass the LLM).
session.read_file("src/main.py")
session.bash("pytest -q")
session.glob("**/*.py")
session.grep("PermissionPolicy")
session.git({"command": "status"})
session.web_search({"query": "rust async cancellation"})

# 4. Delegation — isolate child context.
session.task({
    "agent": "explore",
    "description": "Find auth entry points",
    "prompt": "Inspect the repo and return a list of auth files with evidence.",
})
session.tasks([
    {"agent": "explore",      "description": "Find tests", "prompt": "Locate auth tests."},
    {"agent": "verification", "description": "Check risk", "prompt": "Review auth edge cases."},
])

# 5. Programmatic tool calling — bounded JS in embedded QuickJS.
session.program({
    "source": """
        export default async function run(ctx, inputs) {
          const hits  = await ctx.grep(inputs.query, { glob: '*.py' });
          const files = await ctx.glob('src/**/*.py');
          return { hits, files: files.slice(0, 10) };
        }
    """,
    "inputs": {"query": "PermissionPolicy"},
    "allowed_tools": ["grep", "glob"],
    "limits": {"timeoutMs": 30_000, "maxToolCalls": 20, "maxOutputBytes": 65_536},
})

# 6. Structured output — schema-validated JSON from any provider.
session.tool("generate_object", {
    "schema": {
        "type": "object",
        "required": ["sentiment", "confidence"],
        "properties": {
            "sentiment":  {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    },
    "prompt": "Classify: 'This product is amazing!'",
    "schema_name": "sentiment",
})

# 7. Runs and replay — typed runtime state, not text scraping.
runs = session.runs()
if runs:
    last = runs[-1]
    session.run_snapshot(last["id"])
    session.run_events(last["id"])
    session.active_tools()
    session.cancel_run(last["id"])      # only cancels if still active

# 8. Verification — completion requires a checked result.
session.verify_commands("auth refactor", [
    {"label": "tests", "command": "pytest -q"},
    {"label": "lint",  "command": "ruff check ."},
])
session.verification_summary_text()

# 9. HITL confirmations — single safety gate.
for pending in session.pending_confirmations():
    session.confirm_tool_use(pending["tool_id"], approved=True, reason="Reviewed")

# 10. MCP — attach servers to a live session, tools selected per turn.
session.add_mcp({
    "name": "github",
    "transport": {
        "type": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
    },
    "timeout_ms": 30_000,
})
session.mcps()
session.remove_mcp("github")

# 11. Memory — optional evidence, not auto-stuffing.
session.remember_success("refactor auth", ["edit", "bash"], "all tests passing")
session.recall_similar("auth refactor", limit=5)
session.memory_stats()

# 12. Tools, skills, slash commands, hooks, workers.
session.tool_names()
session.tool_definitions()
session.list_commands()
session.register_command("ping", "Health check", lambda args, ctx: "pong")
session.register_hook("audit", "pre_tool_use", handler_fn)
session.register_worker_agent(
    WorkerAgentSpec.verifier("verify-cow", "Run focused checks"),
)

# 13. Persistence and lifecycle.
session.save()
resumed = agent.resume_session("my-session", opts)
session.cancel()    # cancels in-flight send/stream
session.close()
```

```typescript
import {
  Agent,
  SessionOptions,
  PermissionPolicy,
  ConfirmationPolicy,
  WorkerAgentSpec,
  FileMemoryStore,
  FileSessionStore,
  HttpTransport,
} from '@a3s-lab/code';

// 1. Configure a session — typed extension options, not raw flags.
const opts: SessionOptions = {
  skillDirs: ['./skills'],
  planningMode: 'auto',                         // "auto" | "enabled" | "disabled"
  permissionPolicy: {
    allow: ['read(*)', 'grep(*)', 'glob(*)'],
    ask: ['bash(*)', 'write(*)'],
    deny: ['bash(rm -rf *)'],
    defaultDecision: 'ask',
  },
  confirmationPolicy: {
    enabled: true,
    defaultTimeoutMs: 30_000,
    timeoutAction: 'reject',
  },
  memoryStore: new FileMemoryStore('./memory'),
  sessionStore: new FileSessionStore('./sessions'),
  sessionId: 'my-session',
  autoSave: true,
  ahpTransport: new HttpTransport('http://localhost:8080/ahp'),
};

const agent = await Agent.create('agent.acl');
const session = agent.session('/my-project', opts);

// 2. Send / stream — string or object-shaped requests.
const result = await session.send({ prompt: 'Refactor the auth module' });
console.log(result.text, result.verificationStatus);

const stream = await session.stream({ prompt: 'Continue the refactor' });
for await (const event of stream) {
  if (event.eventType === 'text_delta') process.stdout.write(event.text ?? '');
}

// 3. Direct tools (bypass the LLM).
await session.readFile('src/main.ts');
await session.bash('npm test');
await session.glob('**/*.ts');
await session.grep('PermissionPolicy');
await session.git({ command: 'status' });
await session.webSearch({ query: 'rust async cancellation' });

// 4. Delegation — isolate child context.
await session.task({
  agent: 'explore',
  description: 'Find auth entry points',
  prompt: 'Inspect the repo and return a list of auth files with evidence.',
});
await session.tasks([
  { agent: 'explore',      description: 'Find tests', prompt: 'Locate auth tests.' },
  { agent: 'verification', description: 'Check risk', prompt: 'Review auth edge cases.' },
]);

// 5. Programmatic tool calling — bounded JS in embedded QuickJS.
await session.program({
  source: `
    export default async function run(ctx, inputs) {
      const hits  = await ctx.grep(inputs.query, { glob: '*.ts' });
      const files = await ctx.glob('src/**/*.ts');
      return { hits, files: files.slice(0, 10) };
    }
  `,
  inputs: { query: 'PermissionPolicy' },
  allowedTools: ['grep', 'glob'],
  limits: { timeoutMs: 30_000, maxToolCalls: 20, maxOutputBytes: 65_536 },
});

// 6. Structured output — schema-validated JSON from any provider.
await session.tool('generate_object', {
  schema: {
    type: 'object',
    required: ['sentiment', 'confidence'],
    properties: {
      sentiment:  { type: 'string', enum: ['positive', 'negative', 'neutral'] },
      confidence: { type: 'number', minimum: 0, maximum: 1 },
    },
  },
  prompt: "Classify: 'This product is amazing!'",
  schema_name: 'sentiment',
});

// 7. Runs and replay — typed runtime state, not text scraping.
const runs = await session.runs();
const last = runs?.at(-1);
if (last) {
  await session.runSnapshot(last.id);
  await session.runEvents(last.id);
  await session.activeTools();
  await session.cancelRun(last.id);             // only cancels if still active
}

// 8. Verification — completion requires a checked result.
await session.verifyCommands('auth refactor', [
  { label: 'tests', command: 'npm test' },
  { label: 'lint',  command: 'npm run lint' },
]);
session.verificationSummaryText();

// 9. HITL confirmations — single safety gate.
for (const pending of await session.pendingConfirmations()) {
  await session.confirmToolUse(pending.toolId, true, 'Reviewed');
}

// 10. MCP — attach servers to a live session, tools selected per turn.
await session.addMcp({
  name: 'github',
  transport: {
    type: 'stdio',
    command: 'npx',
    args: ['-y', '@modelcontextprotocol/server-github'],
  },
  timeoutMs: 30_000,
});
await session.mcps();
await session.removeMcp('github');

// 11. Memory — optional evidence, not auto-stuffing.
await session.rememberSuccess('refactor auth', ['edit', 'bash'], 'all tests passing');
await session.recallSimilar('auth refactor', 5);
await session.memoryStats();

// 12. Tools, skills, slash commands, hooks, workers.
session.toolNames();
session.toolDefinitions();
session.listCommands();
session.registerCommand('ping', 'Health check', (_args, _ctx) => 'pong');
session.registerHook('audit', 'pre_tool_use', { tool: 'bash' }, undefined, (event) => {
  return { action: 'continue' };
});
session.registerWorkerAgent({
  name: 'verify-cow',
  description: 'Run focused checks',
  kind: 'verifier',
});

// 13. Persistence and lifecycle.
await session.save();
const resumed = agent.resumeSession('my-session', opts);
session.cancel();   // cancels in-flight send/stream
session.close();
```

---

## Design Principles

### 1. Small Kernel

The core runtime should do only the irreversible work:

- maintain the agent loop
- call the LLM
- expose selected actions
- execute actions through a single executor
- record observations
- compact state when needed
- return events and results

Advanced capabilities belong in the harness, not in the kernel.

### 2. Context Is Budgeted

The model should see the smallest useful context for the current decision.

All context sources should eventually flow through one assembler:

```text
AGENTS.md
skills
memory
file search
MCP
AHP
delegated task runs
tool observations
        -> ContextItem
        -> rank
        -> dedupe
        -> budget
        -> render
```

Raw logs, full grep output, and complete delegated-task transcripts should be stored as artifacts or trace data, not repeatedly injected into the prompt.

### 3. Tools Are Selected, Not Dumped

A3S Code keeps a full tool registry, but the model only sees tools relevant to the current turn.

Default core tools:

| Category | Tools |
|----------|-------|
| Files | `read`, `write`, `edit`, `patch` |
| Search | `grep`, `glob`, `ls` |
| Shell | `bash` |
| Programmatic | `program` |
| Delegation | `task`, `parallel_task` |
| Skills | `search_skills`, `Skill` |
| Structured Output | `generate_object` |

Intent-gated tools:

| Category | Tools |
|----------|-------|
| Web | `web_fetch`, `web_search` |
| Git | `git` |
| Batch | `batch` |
| External | MCP tools |

This follows the same direction as modern agent harnesses: remove routine tool clutter from the model's context and expose capabilities only when the task asks for them.

Workspace backends are capability providers behind the stable built-in tool
contracts. By default, `read`, `write`, `edit`, `patch`, `ls`, `grep`, `glob`,
`bash`, and `git` operate on the local workspace. Embedded hosts can supply
`WorkspaceServices` through `SessionOptions::with_workspace_backend(...)` so
those same tools target a DFS, browser workspace, remote container, or other
host-managed environment.

```rust
use a3s_code_core::{Agent, SessionOptions, WorkspaceServices};

# async fn run() -> anyhow::Result<()> {
let agent = Agent::new("agent.acl").await?;
let workspace = WorkspaceServices::local("/repo");
let session = agent.session(
    "/repo",
    Some(SessionOptions::new().with_workspace_backend(workspace)),
)?;
# Ok(())
# }
```

For non-local backends, A3S Code exposes tools according to declared workspace
capabilities. `bash` is exposed only when a command runner is available,
`grep`/`glob` only when a search provider is available, and `git` only when a
workspace Git provider is available. Browser hosts can pair a virtual file
system with a browser Git implementation, while cloud hosts can route the same
tool contract through DFS or RPC-backed providers.

#### S3-compatible storage backend

When the `s3` Cargo feature is enabled, `S3WorkspaceBackend` lets built-in
file tools (`read`, `write`, `edit`, `patch`, `ls`) target any S3-compatible
endpoint — AWS S3, MinIO, RustFS, Cloudflare R2, Backblaze B2, and so on.
`bash`, `git`, `grep`, and `glob` are intentionally not registered because
object storage cannot service them.

```toml
# Cargo.toml
[dependencies]
a3s-code-core = { version = "2.6", features = ["s3"] }
```

```rust
use a3s_code_core::{Agent, S3BackendConfig, SessionOptions, WorkspaceServices};

# async fn run() -> anyhow::Result<()> {
let agent = Agent::new("agent.acl").await?;
let config = S3BackendConfig::new(
    "workspace",                     // bucket
    "users/u1/sessions/s1",          // workspace prefix inside the bucket
    "AKIA...",                       // access key id
    "...",                           // secret access key
)
.endpoint("https://minio.local:9000")  // omit for AWS S3
.region("us-east-1")
.force_path_style(true);               // true for MinIO/RustFS, false for AWS
let session = agent.session(
    "s3://workspace/users/u1/sessions/s1",
    Some(SessionOptions::new().with_workspace_backend(WorkspaceServices::s3(config))),
)?;
# Ok(())
# }
```

The Node and Python SDKs expose the same backend:

```js
// Node
import { Agent, S3WorkspaceBackend } from '@a3s-lab/code';
const session = agent.session(workspaceUri, {
    workspaceBackend: new S3WorkspaceBackend({
        endpoint: 'https://minio.local:9000',
        region: 'us-east-1',
        accessKeyId: 'AKIA...',
        secretAccessKey: '...',
        bucket: 'workspace',
        prefix: 'users/u1/sessions/s1',
        forcePathStyle: true,
    }),
});
```

```python
# Python
from a3s_code import Agent, S3WorkspaceBackend, SessionOptions
opts = SessionOptions()
opts.workspace_backend = S3WorkspaceBackend(
    bucket="workspace",
    prefix="users/u1/sessions/s1",
    access_key_id="AKIA...",
    secret_access_key="...",
    endpoint="https://minio.local:9000",
    region="us-east-1",
    force_path_style=True,
)
session = agent.session(workspace_uri, opts)
```

S3 has no atomic read-modify-write, so concurrent writers to the same key may
overwrite each other. Partition workspaces per session/user via the `prefix`
field when running multi-tenant.

### 4. Programmatic Tool Calling

High-frequency tool chains should move out of the LLM loop.

Instead of forcing the model through:

```text
grep -> read -> grep -> read -> summarize
```

the harness can run a bounded JavaScript program in the embedded QuickJS VM:

```js
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
```

The same capability is available from Python with `session.program({...})` and
from Rust by calling the core `program` tool. If an allow-list is omitted, the
script can call every registered tool except `program`; use `allowedTools` or
`allowed_tools` to narrow the surface. Programmatic tools should return
structured summaries, findings, artifact references, and suggested next actions.
Raw output belongs in trace storage.

### 5. Structured Output

When the agent needs to produce machine-readable results, `generate_object`
forces schema-validated JSON output from any LLM provider:

```typescript
const result = await session.tool('generate_object', {
  schema: {
    type: 'object',
    required: ['sentiment', 'confidence'],
    properties: {
      sentiment: { type: 'string', enum: ['positive', 'negative', 'neutral'] },
      confidence: { type: 'number', minimum: 0, maximum: 1 },
    },
  },
  prompt: 'Classify: "This product is amazing!"',
  schema_name: 'sentiment',
});

const { object } = JSON.parse(result.output);
// { sentiment: "positive", confidence: 0.95 }
```

The tool works in two modes:
- **Agent-driven**: The LLM sees `generate_object` in its tool list and calls it
  autonomously when structured output is needed.
- **Direct call**: `session.tool('generate_object', ...)` bypasses LLM decision-making
  for deterministic structured extraction.

Reliability comes from three layers: tool-call mode forces the LLM to produce
JSON as tool arguments, a built-in schema validator catches violations, and an
automatic repair loop feeds errors back to the model (up to `max_repair_attempts`
retries). Streaming mode emits partial objects as `tool_output_delta` events.

### 6. Runtime Observability Is A Contract

Product UIs and harnesses should build from typed runtime state rather than
parsing final answer text. Every `send(...)` or `stream(...)` creates run-scoped
state in the session; when a session store is configured, these records are
persisted with the rest of the session.

Durable run state has two layers:

| Record | Purpose |
|--------|---------|
| `RunSnapshot` | Stable per-run state: id, session_id, status, original prompt, timestamps, final result_text or error, and event_count. |
| `RunEventRecord` | Ordered audit trail: sequence, timestamp_ms, and the emitted `AgentEvent`. |

The event stream is organized around the agent loop:

| Loop phase | Representative events |
|------------|-----------------------|
| Intent | `agent_start`, `agent_mode_changed`, `goal_extracted`, `planning_start`, `planning_end`, `task_updated` |
| Context | `context_resolving`, `context_resolved`, `memory_recalled`, `memories_searched`, `context_compacted` |
| Action | `tool_start`, `tool_end`, `permission_denied`, `confirmation_required`, `confirmation_received`, `confirmation_timeout`, `subagent_start`, `subagent_progress`, `subagent_end` |
| Observation | `tool_output_delta`, `tool_end`, `task_updated`, `turn_end`, `error` |
| Verification | `agent_end` with `verification_summary`, plus `verification_reports()` and `verification_summary()` |
| Compaction | `context_compacted` |

Replay boundaries are explicit:

- Replayable means observable and reconstructible, not re-executable.
- Raw LLM messages remain in session history; run records capture state and
  runtime events.
- Full raw logs and large outputs should live in trace or artifact storage;
  events should stay typed and product-friendly.

Node and Python expose the same session controls as the Rust core:

```js
agent.session('/repo', { planningMode: 'disabled' }) // auto | enabled | disabled
await session.task({
  agent: 'explore',
  description: 'Find auth files',
  prompt: 'Inspect auth-related files and return evidence.',
})
console.log(session.toolDefinitions())
await session.git({ command: 'status' })
```

```python
session = agent.session("/repo", planning_mode="enabled")
session.task({
    "agent": "verification",
    "description": "Check release risk",
    "prompt": "Validate the current changes and summarize blockers.",
})
session.tool_definitions()
session.git({"command": "status"})
```

Planning is explicit and observable. In `auto` mode the runtime performs
structured pre-analysis without a brittle keyword gate; `enabled` forces it, and
`disabled` lets SDK callers opt out for latency-sensitive requests. Planning
state is emitted as run-scoped events so product UIs can render a TaskList and
update each item as work progresses.

Run tracking is also part of the public surface:

```js
const runs = await session.runs()
const latest = runs.at(-1)

if (latest) {
  console.log(await session.runSnapshot(latest.id))
  console.log(await session.runEvents(latest.id))
  console.log(await session.activeTools())
  await session.cancelRun(latest.id)
}
```

```python
runs = session.runs()
latest = runs[-1] if runs else None

if latest:
    print(session.run_snapshot(latest["id"]))
    print(session.run_events(latest["id"]))
    print(session.active_tools())
    session.cancel_run(latest["id"])
```

### 7. AHP-Supervised Background Advice

A3S Code keeps the core session runtime focused on the main agent. Background
advice, context supplements, and proposed PTC scripts are caller-owned AHP
harness behaviors rather than a separate in-core advisory runtime.

Attach an AHP hook executor to forward lifecycle hooks and durable run events to
the harness:

```python
from a3s_code import Agent, HttpTransport, SessionOptions

agent = Agent.create("agent.acl")
opts = SessionOptions()
opts.ahp_transport = HttpTransport("http://localhost:8080/ahp")
session = agent.session(".", opts)
result = session.send("Refactor the auth module")
```

The SDK event stream remains product/UI friendly. When AHP is enabled, selected
runtime events are projected into the harness-facing contract
(`RunLifecycle`, `TaskList`, `Verification`) by `agent_event_to_ahp_events`,
while tool, prompt, confirmation, idle, and error hooks continue to map to AHP
supervision events.

The harness can observe run lifecycle, task, verification, tool, confirmation,
idle, and error events; it can maintain its own background workers and publish
advice through the host UI or by explicitly calling session APIs. Proposed PTC
scripts remain proposals until the caller runs them through the normal
`program`, permission, confirmation, and trace paths.

### 8. Delegated Tasks Isolate Context

Delegated tasks are not there to create more chat. They isolate local work.

The parent agent delegates:

```text
task(role, prompt, budget)
parallel_task(tasks)
```

Delegated child runs should return:

- summary
- key findings
- files inspected or changed
- evidence references
- risks
- confidence
- trace reference

The parent should not ingest the full child transcript.

### 9. Safety Has One Gate

All side effects should pass through one authorization path.

Policies may be composed from workspace boundaries, permissions, confirmations, skill grants, and security providers, but execution should observe one effective decision:

```text
Allow | Ask | Deny
```

This keeps `bash`, writes, network calls, MCP calls, and release actions auditable.

### 10. Completion Requires Verification

A coding agent is not done because it produced text. It is done when the goal is satisfied and the result has been checked.

Verification can include:

- unit tests
- type checks
- lint
- command output
- git diff review
- delegated review
- explicit residual risk reporting

---

## Architecture

Current public API:

```text
Agent
  -> AgentSession
     -> ToolSelector
        -> ToolExecutor
        -> SkillRegistry
        -> Context providers
        -> Permission / confirmation
        -> Compaction
        -> Events
```

Target harness architecture:

```text
a3s-code
├── runtime kernel
│   ├── internal agent loop
│   ├── state
│   ├── events
│   └── trace
│
├── harness
│   ├── intent router
│   ├── context assembler
│   ├── tool selector
│   ├── program executor
│   ├── safety gate
│   ├── verification loop
│   └── compaction engine
│
├── capabilities
│   ├── core tools
│   ├── skills
│   ├── MCP
│   ├── memory
│   ├── web
│   └── git
│
├── delegation
│   ├── task
│   └── parallel_task
│
├── advanced control
│   └── session-level lane queues for external/hybrid dispatch
│
└── API
    ├── Rust
    ├── Python
    └── Node.js
```

The long-term direction is a small runtime kernel with powerful harness extensions.

---

## Skills

Skills are loaded on demand. A3S Code exposes `search_skills` so the model can discover relevant skills without injecting every skill description into the prompt.

Example skill:

```markdown
---
name: safe-reviewer
description: Review code without modifying files
allowed-tools: "read(*), grep(*), glob(*)"
---

Review the code in the workspace. Focus on correctness, regressions, and missing tests.
Do not modify files.
```

Use custom skill directories:

```python
from a3s_code import SessionOptions

opts = SessionOptions()
opts.skill_dirs = ["./skills"]
session = agent.session(".", opts)
```

Built-in skills include code search, code review, explanation, and bug finding helpers.

---

## Delegation

Use delegation when a task benefits from context isolation.

Core delegation primitives:

- `task` — run one focused delegated child run
- `parallel_task` — run independent delegated child runs concurrently

The older model-visible team shortcut and duplicate lifecycle control-plane API
are no longer part of the public surface. Multi-agent work enters through the
delegation core.

Optional lane queues are also outside the default path. They are for explicit
external/hybrid dispatch, priority experiments, and operational integrations;
ordinary sessions are queue-free unless a session queue configuration is supplied.
They are not part of the delegation path.

---

## AHP Integration

AHP, the Agent Harness Protocol, is best treated as a harness extension.

It should observe runtime events and provide suggestions:

- add or boost context
- enable an action
- require confirmation
- request compaction
- provide policy hints

Those suggestions should flow through the same systems as everything else:

```text
AHP suggestion
  -> ContextAssembler
  -> ToolSelector
  -> SafetyGate
  -> CompactionEngine
```

AHP should not bypass context budgets or directly stuff prompt text into the model.

Example:

```python
from a3s_code import SessionOptions
from a3s_code.ahp import AhpHookExecutor, AhpTransport

ahp = AhpHookExecutor.new_with_config(
    AhpTransport.http("http://harness:8080/ahp", None),
    idle_threshold_ms=10_000,
)

opts = SessionOptions()
opts.ahp_executor = ahp
session = agent.session("/workspace", opts)
```

---

## Memory

Memory is optional evidence, not automatic prompt stuffing.

Recommended model:

| Layer | Purpose |
|-------|---------|
| Conversation summary | Preserve load-bearing state across long sessions |
| Working memory | Current task state |
| Long-term memory | Optional retrievable evidence across sessions |

Enable persistent memory when your product needs it:

```python
from a3s_code import SessionOptions, FileMemoryStore

opts = SessionOptions()
opts.memory_store = FileMemoryStore("./memory")
session = agent.session(".", opts)
```

---

## Safety

Configure explicit permissions:

```python
from a3s_code import SessionOptions, PermissionPolicy

opts = SessionOptions()
opts.permission_policy = PermissionPolicy(
    allow=["read(*)", "grep(*)"],
    deny=["bash(*)", "write(*)"],
    default_decision="deny",
)

session = agent.session(".", opts)
```

Built-in safeguards include:

- permission policies
- human-in-the-loop confirmation
- workspace-scoped tool context
- tool timeouts
- duplicate tool-call protection
- LLM circuit breaker
- context compaction
- output sanitization hooks

---

## MCP

Connect to Model Context Protocol servers when external capabilities are needed:

```acl
mcp_servers = [
  {
    name = "filesystem"
    transport = "stdio"
    command = "npx"
    args = ["@modelcontextprotocol/server-filesystem", "./workspace"]
  }
]
```

MCP tools are selected per turn instead of being listed wholesale in the system prompt.

SDK callers can also attach MCP servers to a live session with object-shaped
configs:

```js
await session.addMcp({
  name: 'github',
  transport: { type: 'stdio', command: 'npx', args: ['-y', '@modelcontextprotocol/server-github'] },
  timeoutMs: 30000,
})
```

---

## Slash Commands

Sessions support slash commands:

| Command | Description |
|---------|-------------|
| `/help` | List available commands |
| `/model [provider/model]` | Show or switch model |
| `/cost` | Show token usage |
| `/clear` | Clear conversation history |
| `/compact` | Manually trigger context compaction |

---

## Configuration

The config language is ACL. Config files use the `.acl` extension and labeled
blocks such as `providers "anthropic" { ... }`.

```acl
default_model = "anthropic/claude-sonnet-4-20250514"

providers "anthropic" {
  apiKey = env("ANTHROPIC_API_KEY")
}

skill_dirs = ["./skills"]
mcp_servers = []

ahp = {
  enabled = true
  url     = "http://harness:8080/ahp"
  idle_ms = 10_000
}
```

---

## Development

```bash
cargo check -p a3s-code-core
cargo test -p a3s-code-core
cargo clippy -p a3s-code-core -- -D warnings
```

Build language bindings individually:

```bash
cargo build -p a3s-code-py
cargo build -p a3s-code-node
```

---

## Documentation

Full reference and guides: [a3s-lab.github.io/a3s/docs/code](https://a3s-lab.github.io/a3s/docs/code)

- [SDK API Design Contract](manual/SDK_API_DESIGN.md)
- [Sessions](https://a3s-lab.github.io/a3s/docs/code/sessions)
- [Tools & Structured Output](https://a3s-lab.github.io/a3s/docs/code/tools)
- [AHP Protocol](https://a3s-lab.github.io/a3s/docs/code/ahp-integration)
- [Skills](https://a3s-lab.github.io/a3s/docs/code/skills)
- [Memory](https://a3s-lab.github.io/a3s/docs/code/memory)
- [Security](https://a3s-lab.github.io/a3s/docs/code/security)
- [Hooks](https://a3s-lab.github.io/a3s/docs/code/hooks)
- [Examples](https://a3s-lab.github.io/a3s/docs/code/examples)
- [Tutorials](https://a3s-lab.github.io/a3s/tutorials)

---

## License

MIT
