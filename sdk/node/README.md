# A3S Code — Node SDK

Native Node.js bindings for the A3S Code AI coding agent.

## Installation

```bash
npm install @a3s-lab/code
```

## Quick Start

```js
const { Agent } = require('@a3s-lab/code')

async function main() {
  const agent = await Agent.create('agent.acl')
  const session = agent.session('/my-project')

  const result = await session.send({
    prompt: 'What files handle authentication?',
  })
  console.log(result.text)
}

main().catch(console.error)
```

## Programmatic Tool Calling

`session.program(...)` runs a bounded JavaScript script in the embedded QuickJS
runtime. It is the SDK-friendly wrapper around the core `program` tool.

```js
const result = await session.program({
  source: `
    export default async function run(ctx, inputs) {
      const hits = await ctx.grep(inputs.query, { glob: '*.ts' })
      const files = await ctx.glob('src/**/*.ts')
      return { hits, files: files.slice(0, 10) }
    }
  `,
  inputs: { query: 'PermissionPolicy' },
  allowedTools: ['grep', 'glob'],
  limits: { timeoutMs: 30000, maxToolCalls: 20, maxOutputBytes: 65536 },
})

console.log(result.output)
```

Omit `allowedTools` to allow every registered session tool except `program`.
Scripts can also be loaded from workspace-relative `.js` or `.mjs` files with
`{ path: 'scripts/ptc/search.js' }`.

## Workspace Backends And Direct Files

The default workspace backend is the local filesystem rooted at the session
workspace. SDK callers can pass an explicit typed backend through the same
option surface used by remote, browser, DFS, and container-backed workspaces:

```js
const { Agent, LocalWorkspaceBackend } = require('@a3s-lab/code')

const agent = await Agent.create('agent.acl')
const session = agent.session('/repo', {
  workspaceBackend: new LocalWorkspaceBackend('/repo'),
})

await session.writeFile('notes.txt', 'one\ntwo\n')
await session.readFile('notes.txt')
await session.ls()
await session.editFile('notes.txt', 'one', 'uno')
await session.patchFile('notes.txt', '@@ -1,2 +1,2 @@\n uno\n-two\n+dos')
```

### S3-compatible object storage

`S3WorkspaceBackend` lets built-in file tools (`read`, `write`, `edit`,
`patch`, `ls`) target any S3-compatible endpoint — AWS S3, MinIO, RustFS,
Cloudflare R2, Backblaze B2, etc. `bash`, `git`, `grep`, and `glob` are
automatically hidden from the model because object storage cannot service
them.

```js
const { Agent, S3WorkspaceBackend } = require('@a3s-lab/code')

const agent = await Agent.create('agent.acl')
const session = agent.session('s3://workspace/users/u1/sessions/s1', {
  workspaceBackend: new S3WorkspaceBackend({
    endpoint: 'https://minio.local:9000',     // omit for AWS S3
    region: 'us-east-1',
    accessKeyId: 'AKIA...',
    secretAccessKey: '...',
    bucket: 'workspace',
    prefix: 'users/u1/sessions/s1',
    forcePathStyle: true,                     // true for MinIO/RustFS/R2
  }),
})

await session.writeFile('notes/hello.txt', 'one\ntwo\n')
await session.readFile('notes/hello.txt')
await session.ls('notes')
```

S3 has no atomic read-modify-write, so concurrent writers to the same key
overwrite each other (last-writer-wins). Partition workspaces per session or
user via the `prefix` field when running multi-tenant.

## Planning Events

Planning is automatic by default. Prefer the explicit tri-state
`planningMode` contract for SDK callers:

```js
agent.session('/my-project', { planningMode: 'auto' })     // default
agent.session('/my-project', { planningMode: 'enabled' })  // force planning
agent.session('/my-project', { planningMode: 'disabled' }) // explicitly off
```

The legacy boolean shortcut still works: `{ planning: true }` forces planning
and `{ planning: false }` disables it.

When streaming, `task_updated` is the authoritative task-list snapshot for UI
rendering. `planning_end` contains the initial plan, while `step_start` and
`step_end` are fine-grained progress events.

## Durable Request Shape

`send(...)` and `stream(...)` accept either a prompt string or an object-shaped
request. Use the object shape when the call needs history, attachments, or
future request options:

```js
const result = await session.send({
  prompt: 'Explain the auth module',
  history: previousMessages,
  attachments: [{ data: imageBuffer, mediaType: 'image/png' }],
})
```

`sendRequest(...)`, `streamRequest(...)`, and attachment-specific positional
overloads remain for compatibility.

## Delegation And Tool Introspection

The SDK exposes the core `task` / `parallel_task` tools as direct helpers:

```js
await session.task({
  agent: 'explore',
  description: 'Find auth entry points',
  prompt: 'Inspect the repository and summarize the auth-related files.',
})

await session.tasks([
  { agent: 'explore', description: 'Find tests', prompt: 'Locate auth tests.' },
  { agent: 'verification', description: 'Check risk', prompt: 'Review auth edge cases.' },
])
```

For automatic subagent delegation, `autoParallel: false` disables automatic
parallel fan-out while keeping manual `parallel_task` / `session.tasks(...)`
available:

```js
const session = agent.session('/my-project', {
  autoDelegation: { enabled: true, maxTasks: 4 },
  maxParallelTasks: 8,
  autoParallel: false,
})
```

Use `session.toolNames()` for names and `session.toolDefinitions()` when a UI
needs the full model-visible schemas.

## Evidence And Artifacts

Tool outputs that exceed the inline display budget are retained as session
artifacts. Use `artifactStoreLimits` to tune retention and `getArtifact(...)`
to retrieve retained content by URI:

```js
const session = agent.session('/my-project', {
  artifactStoreLimits: { maxArtifacts: 64, maxBytes: 8 * 1024 * 1024 },
})

const artifact = session.getArtifact('a3s://tool-output/read/abc123')
if (artifact) console.log(artifact.content)
```

External verification systems can attach their reports to the same session
evidence stream:

```js
session.recordVerificationReports([{
  schema: 'a3s.verification_report.v1',
  subject: 'sdk:tests',
  status: 'passed',
  checks: [{
    id: 'check:sdk',
    kind: 'test',
    description: 'Run SDK tests',
    status: 'passed',
    required: true,
  }],
}])
```

## Object-Shaped Direct Tools

New direct helpers use option objects when the command can grow over time:

```js
await session.git({ command: 'status' })
await session.git({ command: 'worktree', subcommand: 'list' })
```

The older positional `git(...)` overload and `gitCommand(...)` remain for
compatibility.

## Disposable Worker Agents

A3S Code treats subagents as cattle, not pets: define reproducible worker specs
in code, register them on a session, and delegate by name through the existing
`task` tool.

```js
const session = agent.session('/my-project', {
  workerAgents: [
    {
      name: 'frontend-cow',
      description: 'Small verified frontend fixes',
      kind: 'implementer',
      model: 'openai/gpt-4o',
      maxSteps: 24,
      prompt: 'Keep patches focused and run the narrowest relevant check.',
      confirmationInheritance: 'auto_approve',  // child runs auto-approve Ask decisions
    },
    { name: 'review-cow', description: 'Adversarial review', kind: 'reviewer' },
  ],
})

await session.task({
  agent: 'frontend-cow',
  description: 'Fix admin chat loading state',
  prompt: 'Find and fix the loading-state regression, then summarize verification.',
})
```

You can also register workers after the session is running:

```js
session.registerWorkerAgent({
  name: 'verify-cow',
  description: 'Run focused checks without editing files',
  kind: 'verifier',
})
```

For a worker as the top-level actor, use `agent.sessionForWorker(workspace, spec)`.

### Confirmation Inheritance

Control how child runs resolve Ask decisions with `confirmationInheritance`:

- `'auto_approve'` (default): Child runs auto-approve all Ask decisions
- `'deny_on_ask'`: Child runs fail immediately when encountering an Ask
- `'inherit_parent'`: Child runs inherit the parent's confirmation policy

```js
const session = agent.session('/my-project', {
  workerAgents: [
    {
      name: 'restricted-writer',
      description: 'Write files with parent confirmation',
      kind: 'implementer',
      confirmationInheritance: 'inherit_parent',  // requires parent approval
    },
  ],
})
```

## AHP-Supervised Advice

Background advice belongs in the host or AHP harness. A3S Code forwards hooks,
run lifecycle events, task updates, verification summaries, confirmations, idle
signals, and errors; the harness decides when to surface suggestions, add host
context, or propose PTC scripts.

```js
const session = agent.session('/my-project', {
  ahpTransport: new HttpTransport('http://localhost:8080/ahp'),
})
```

PTC scripts proposed by an AHP harness are not executed automatically. Run them
explicitly through `session.program(...)` so existing permission, confirmation,
and trace policies stay in force.

## Live MCP Servers

Prefer the object-shaped MCP API for new code. It keeps transport-specific
fields grouped and leaves room for OAuth/env/timeout extensions:

```js
await session.addMcp({
  name: 'github',
  transport: {
    type: 'stdio',
    command: 'npx',
    args: ['-y', '@modelcontextprotocol/server-github'],
  },
  env: { GITHUB_TOKEN: process.env.GITHUB_TOKEN ?? '' },
  timeoutMs: 30000,
})

console.log(await session.mcps())
```

The positional `addMcpServer(...)` overload and longer
`addMcpServerConfig(...)` alias remain for compatibility.

## Filesystem-First Agents

Define a durable agent as a **directory** — `instructions.md` (required) plus
optional `agent.acl`, `skills/`, `schedules/` (cron), and `tools/` (`kind: mcp` or
`kind: script` sandboxed QuickJS) — and serve its schedules. Each fire is a full
harness turn (context, tool visibility, safety gate, verification). Returns a
handle you must keep and stop explicitly.

```js
const handle = await agent.serveAgentDir('./my-agent', './workspace', {
  // Optional: pass a sessionStore so each schedule resumes its accumulated
  // context across daemon restarts.
  sessionStore: new FileSessionStore('./sessions'),
})
// ... runs in the background until:
await handle.stop()
console.log(handle.isStopped()) // true
```

## HITL Confirmations

Use `permissionPolicy` to decide which tools ask, then `confirmationPolicy` to
control confirmation runtime behavior such as timeout and YOLO lanes.

```js
const session = agent.session('.', {
  permissionPolicy: { ask: ['bash*'], defaultDecision: 'allow' },
  confirmationPolicy: {
    enabled: true,
    defaultTimeoutMs: 30000,
    timeoutAction: 'reject',
    yoloLanes: ['query'],
  },
})

for (const pending of await session.pendingConfirmations()) {
  await session.confirmToolUse(pending.toolId, true, 'Reviewed')
}
```

For the streaming event-driven loop used by UIs, see
`examples/streaming/hitl_confirmation_loop.ts`.

## Run Replay

Each `send(...)` or `stream(...)` call records a run snapshot and replayable
runtime events:

```js
await session.send('Fix the failing test')

const [run] = await session.runs()
console.log(run.id, run.status)
console.log(await session.runEvents(run.id))
console.log(await session.activeTools())
```

Use `session.currentRun()` while a stream is active to inspect the current run.
Use `session.cancelRun(run.id)` to cancel only that run; stale IDs will not
cancel a newer operation.
