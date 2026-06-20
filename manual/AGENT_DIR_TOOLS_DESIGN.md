# Agent-Dir `tools/` Mapping — Design Doc

Status: IMPLEMENTED. Both backends ship: `kind = "mcp"` and `kind = "script"`.
The loader (`load_tools` in `core/src/config/agent_dir.rs`) parses both into
`ToolSpec::{Mcp, Script}`; `serve::install_agent_dir_tools` registers them at
session build — MCP via `add_mcp_server`, script via the new
`AgentDirScriptTool` (`core/src/tools/agent_dir_script_tool.rs`), a thin facade
over the existing `program` QuickJS path. Scope: how the optional `tools/`
subdirectory of an eve-style agent directory becomes *executable* tools in
A3S Code without ever running arbitrary host JavaScript or arbitrary host
processes.

> Harness guardrail (non-negotiable). A file dropped into `tools/` may become
> **either** an MCP server connection **or** a sandboxed QuickJS program. It is
> NEVER turned into a free-running host JS/native process, and it NEVER gets to
> define its own tool-visibility or safety policy. Tool *definition* is allowed
> from the directory; tool *visibility* and *safety* remain harness-owned. This
> is the deliberate divergence from eve's user-defined-tools model, documented in
> the `core/src/config/agent_dir.rs` module header, and is why the directory
> selects between two harness-owned backends rather than running arbitrary code.

---

## 1. Where this fits today

The agent-dir convention is loaded by `AgentDir::load(dir)` in
`core/src/config/agent_dir.rs:75`. It synthesizes existing config objects from a
directory by convention:

```text
agent/
├── instructions.md   (required)  → SystemPromptSlots.role
├── agent.acl          (optional)  → CodeConfig
├── skills/            (optional)  → CodeConfig.skill_dirs
├── schedules/         (optional)  → Vec<ScheduleSpec>   (serve layer)
└── tools/             (optional)  → Vec<ToolSpec>        (load_tools, THIS DOC)
```

`tools/` is parsed by `load_tools` into `AgentDir::tools` and installed at session
build time. Two runtime seams already exist that this design reuses verbatim; we
wired, we did not invent:

1. MCP registration — `AgentSession::add_mcp_server(McpServerConfig)`
   (`core/src/agent_api.rs:1305`) → `SessionExtensionRuntime::add_mcp_server`
   (`core/src/agent_api/session_extensions.rs:62`), which registers, connects,
   fetches `list_tools()`, wraps with `create_mcp_tools()`
   (`core/src/mcp/tools.rs:83`, prefix `mcp__<server>__<tool>`), and calls
   `tool_executor.register_dynamic_tool(...)`.
2. Sandboxed JS — the `program` tool runs user JS in an embedded QuickJS VM via
   `run_quickjs_script` (`core/src/tools/program_tool.rs:258`). It already loads
   a workspace-relative `.js`/`.mjs` by `path` (`load_script_source`,
   `program_tool.rs:177`), parse-time-blocks `import/eval/Function/Worker/
   WebSocket/fetch` (`validate_script_source`, `program_tool.rs:230`), and runs
   under memory/stack/timeout/tool-call/output limits with a frozen `ctx`
   (`embedded_script_bootstrap`, `program_tool.rs:406`).

The whole job of `tools/` is to declare, by file, *which* of these two existing
backends to instantiate and with what bounded parameters.

---

## 2. (a) `tools/` file conventions

### 2.1 One manifest file per tool, declarative, no executable manifest

Each entry is a single declarative file. We do NOT use a `.js` file as the
manifest (a bare `.js` is ambiguous — is it the program source or a config?),
and we do NOT execute the manifest. The accepted form (as shipped) is:

- `tools/<name>.md` — YAML frontmatter + body, identical mechanics to
  `schedules/*.md`. Reuses `split_frontmatter` and `md_files` (so only `*.md`
  files are read). The body, when present, is treated as the tool *description*
  surfaced to the model.

A `kind:` discriminant selects the backend — the same "a spec file names which
adapter handles it" pattern the schedule loader uses for its frontmatter.

```text
kind = "mcp"        → MCP server connection  (backend 1)
kind = "script"     → sandboxed QuickJS PTC  (backend 2)
```

Unknown `kind` is a hard load error (fail closed). A program-source `.js`/`.mjs`
referenced by a `kind = "script"` spec lives anywhere workspace-relative;
the spec points at it, the spec is never the program.

### 2.2 `kind = "mcp"` spec → `McpServerConfig`

The spec is a 1:1 surface over the *already-deserializable*
`McpServerConfig` (`core/src/mcp/protocol.rs:393`). That type already has a
hand-written `Deserialize` accepting the flat ACL form
(`transport = "stdio" | "http" | "streamable-http"`, plus `command`/`args` or
`url`/`headers`), `env`, `oauth`, and `tool_timeout_secs`. So the loader does
*no new parsing*: it deserializes the file's YAML frontmatter into
`McpServerConfig`.

```md
---
# tools/github.md
kind: mcp
name: github                      # used for the mcp__github__* prefix
transport: stdio
command: npx
args: ["-y", "@modelcontextprotocol/server-github"]
enabled: true
# secrets come from the process env, not the file (see Security)
env: { GITHUB_TOKEN: "${env:GITHUB_TOKEN}" }
tool_timeout_secs: 60
---
GitHub issues and PR tools.   (← optional body = model-facing description)
```

### 2.3 `kind = "script"` spec → bounded `program` invocation

The spec names a workspace-relative `.js`/`.mjs` source and pins the sandbox
limits + allow-list. The fields map directly onto the `program` tool's existing
arguments (`program_tool.rs:47` parameters, `ScriptLimits` at `:112`).

```markdown
---
kind: script
name: search-auth
path: scripts/ptc/search-auth.js   # must end .js/.mjs (program_tool.rs:188)
allowed_tools: [grep, glob, read]  # subset; program is always excluded
limits:
  timeoutMs: 30000
  maxToolCalls: 30
  maxOutputBytes: 65536
---
Find auth-related files and return an evidence list.  (← model-facing description)
```

The source must define `async function run(ctx, inputs)` — enforced by
`script_source_with_host_entrypoint` (`program_tool.rs:330`). The declared
`name` becomes the *model-visible tool name* (see §4 for the thin wrapper).

### 2.4 Loader output (parse, don't wire, in `AgentDir::load`)

Following the `schedules` precedent, `AgentDir::load` only *parses* `tools/` into
typed specs and stores them on `AgentDir`. Actual registration is
done at session build time (§3), exactly as `add_mcp_server` is a session-level
operation today. Proposed shape (names illustrative, not prescriptive):

```rust
pub enum ToolSpec {
    Mcp(McpServerConfig),                 // backend 1
    Script(ScriptToolSpec),               // backend 2
}
pub struct ScriptToolSpec {
    pub name: String,
    pub description: String,              // .md body, or frontmatter `description`
    pub path: PathBuf,                    // workspace-relative .js/.mjs
    pub allowed_tools: Option<Vec<String>>,
    pub limits: ScriptLimits,             // reuse program_tool::ScriptLimits
}
// AgentDir gains:  pub tools: Vec<ToolSpec>,
```

Parsing rules (fail closed, mirroring existing loaders):
- No frontmatter / missing `kind` → error, like `schedules`.
- `kind = "mcp"` with a malformed `McpServerConfig` → propagate the serde error.
- `kind = "script"` whose `path` does not end `.js`/`.mjs` → error at load (do
  not defer to first call).
- Duplicate tool `name` across files → error (names are the registry key).

---

## 3. (b) The two execution backends and when each applies

Decision is purely by `kind`; there is no auto-detection and no fallthrough.

| `kind` | Backend | Runtime | Use when |
|--------|---------|---------|----------|
| `mcp` | MCP server connection | child process (stdio) or remote HTTP, spoken over the MCP protocol; A3S only sends/receives JSON-RPC | the capability is an external, already-MCP-shaped integration (GitHub, Postgres, a remote tool service). The "process" is the MCP server itself, launched/owned by `StdioTransport`/`HttpSseTransport` — A3S never `exec`s arbitrary user code, it speaks a protocol. |
| `script` | sandboxed QuickJS PTC | in-process `rquickjs` VM, no fs/net/proc/env | the capability is local glue over *existing* A3S tools (grep→read→summarize), expressed as JS. The script cannot touch the host; it can only call back through the frozen `ctx` allow-list. |

Why these two and nothing else (Rule 1 / Rule 2): together they cover "reach an
external system" (MCP, isolated by being a separate protocol-speaking process)
and "compose our own tools in a loop" (QuickJS, isolated by the VM). Neither
path can become "run this `.js`/binary on the host with ambient authority,"
which is precisely the eve model we are refusing.

### 3.1 `mcp` registration flow (reused verbatim)

At session build, for each `ToolSpec::Mcp(cfg)` call
`session.add_mcp_server(cfg)` (`agent_api.rs:1305`). That already:
`register_server` → `connect` (creates transport, `McpClient::initialize`,
`list_tools`) → `create_mcp_tools` (prefix `mcp__<name>__<tool>`) →
`register_dynamic_tool`. Removal is `remove_mcp_server` (unregister by
`mcp__<name>__` prefix + disconnect). Idle servers can be reaped by the existing
`McpManager::disconnect_idle`. No new MCP code is required.

Invariant to preserve: the `mcp__<server>__<tool>` naming is baked into
`McpToolWrapper::new` (`tools.rs:27`) and `McpManager::parse_tool_name`
(`manager.rs:339`). A `tools/` MCP spec must not try to rename or unprefix —
visibility/naming stays harness-owned.

### 3.2 `script` registration flow (thin new wrapper, no new sandbox)

The QuickJS executor already exists and is the only place JS runs. The single
new piece is a thin `Tool` impl that, when called, invokes the same
`program`-tool execution path with the spec's pinned `path` + `allowed_tools` +
`limits` — i.e. it is a *named, pre-parameterized* `program` call. It adds NO new
capability: it reuses `validate_script_source`, `run_quickjs_script`, the frozen
`__a3sCtx`, the 64MB memory / 512KB stack / interrupt-timeout limits, and the
per-call `maxToolCalls`/`maxOutputBytes` caps. See §4 for the seam.

Why a wrapper instead of just telling the model "call `program` with this path":
a named tool (`search-auth`) with its own description is what makes a `tools/`
entry *discoverable* and selectable like any other tool, while the sandbox and
allow-list stay fixed by the spec rather than chosen per-call by the model.

---

## 4. (c) How a `tools/`-sourced call still flows through visibility + safety

Both backends terminate as ordinary `Arc<dyn Tool>` entries registered via
`ToolExecutor::register_dynamic_tool` (`core/src/tools/mod.rs:390`). That single
chokepoint is why "tool definition from the dir" cannot smuggle past
"visibility + safety stays with the harness":

1. Registration is mediated. `register_dynamic_tool` → `registry.register`
   (`registry.rs:74`) refuses to shadow builtins. A `tools/` entry can never
   replace `bash`, `read`, `program`, etc.
2. Visibility is harness-derived. The model only ever sees what
   `ToolRegistry::definitions()` (`registry.rs:119`) exposes — name +
   description + JSON schema. A `tools/` tool is visible exactly like a built-in
   or an MCP tool; it cannot inject itself into the prompt outside that surface,
   and it cannot opt itself into being always-selected.
3. Safety gate is unchanged and upstream of execution. Every model-driven tool
   call is checked by the session's `PermissionChecker`
   (`permissions/mod.rs:28`, Deny → Allow → Ask → HITL default) before the tool
   runs. A `tools/` entry is just another `tool_name` fed to `check(...)`;
   policy authored in `agent.acl`/host config still governs it. The directory
   cannot grant itself `Allow`.
4. Workspace boundary still applies. Execution goes through
   `ToolExecutor::execute*` (`tools/mod.rs:453`/`:472`), which runs
   `check_workspace_boundary` before dispatch.
5. Backend-internal guards remain:
   - `script`: the QuickJS VM has no fs/net/proc/env; the *only* outbound
     capability is `ctx.tool(...)`, and `execute_host_tool_json`
     (`program_tool.rs:436`) enforces the per-script `allowed_tools` set and the
     `maxToolCalls` counter on every hop. Note the boundary precisely: those inner
     `ctx` calls go through `ToolRegistry::execute_with_context` directly — they
     are NOT re-evaluated against the session `PermissionChecker`/HITL (that gate
     runs in the agent loop for the model-selected `program`/script call, not for
     the script's internal hops). So the **allow-list is the boundary** for what a
     directory script may reach, and the loader fails it closed (empty by default).
   - `mcp`: A3S never runs the server's code; it exchanges JSON-RPC. The server
     is a separate process/endpoint owned by the transport layer.

Net: a `tools/` file can *add a callable name*. The model-selected call to it is
permission-gated like any tool, and the name is non-shadowing and
harness-namespaced (`mcp__…` for MCP). A `script`'s inner tool calls are bounded
by its pinned allow-list + the QuickJS sandbox rather than the permission policy.
There is no path by which a directory file executes arbitrary host JS or an
arbitrary host process.

```text
tools/<name>.md
   │  AgentDir::load  (parse only)
   ▼
ToolSpec ──┬─ Mcp(McpServerConfig)  ─► session.add_mcp_server ─► McpToolWrapper (mcp__name__*)
           └─ Script(ScriptToolSpec) ─► AgentDirScriptTool      (name)
                                          │
                  register_dynamic_tool ◄─┘   (non-shadowing; builtins win)
                                          │
   model selects ──► PermissionChecker.check ──► ToolExecutor.execute ──► backend
                       (Deny/Allow/Ask)          (workspace boundary)
                                                   │
                          MCP: JSON-RPC to server  │  script: QuickJS VM,
                                                   │  frozen ctx, allow-list,
                                                   ▼  limits — ctx.tool allow-list
                                                      (NOT the permission policy)
```

---

## 5. (d) The minimal trait/seam (as built)

Goal: the smallest possible new surface, reusing both existing backends. Two
trait-free functions plus one tiny `Tool` impl. No new manager, no new sandbox,
no new permission path (Rule 2: this is an *extension*, the core is untouched).

### Seam 1 — parse (in `config/agent_dir.rs`, alongside `load_schedules`)

```rust
fn load_tools(dir: &Path) -> Result<Vec<ToolSpec>>;   // mirrors load_schedules
```
Called from `AgentDir::load` after `schedules`; stores `tools: Vec<ToolSpec>` on
`AgentDir`. Pure parsing — fail closed, no I/O beyond reading the files.

### Seam 2 — register (session build time, NOT inside `AgentDir::load`)

A free function (or `SessionExtensionRuntime` method, next to `add_mcp_server`):
```rust
async fn install_agent_dir_tools(session: &AgentSession, specs: &[ToolSpec]) -> Result<()>;
//   Mcp(cfg)    => session.add_mcp_server(cfg).await   (existing, unchanged)
//   Script(spec)=> session.tool_executor
//                         .register_dynamic_tool(Arc::new(AgentDirScriptTool::new(spec)))
```
Keeping registration at the session level preserves the existing rule that
capability mutation is a session operation (the `add_mcp_server` precedent) and
keeps `AgentDir::load` a pure, side-effect-free config synthesizer.

### Seam 3 — the only new `Tool` impl

`AgentDirScriptTool` is a named facade over the existing `program` execution. It
holds the parsed `ScriptToolSpec` and, in `execute`, builds the same args the
`program` tool already accepts (`{ type: "script", language: "javascript",
path, allowed_tools, limits }`) and runs them through the existing
`run_quickjs_script` path. It introduces no new sandbox primitive — it is a
pre-parameterized, discoverable `program` call.

```rust
struct AgentDirScriptTool { spec: ScriptToolSpec }  // name()/description() from spec
#[async_trait] impl Tool for AgentDirScriptTool {
    // execute(): delegate to the existing program-script executor with
    //            spec.path / spec.allowed_tools / spec.limits pinned.
}
```

That is the entire footprint: one parser, one installer, one wrapper. MCP and
QuickJS — and the permission + visibility chokepoints — are all reused as-is.

---

## 6. Non-goals / explicitly refused

- No arbitrary host JS or native processes from `tools/` (the guardrail).
- No per-tool permission policy authored in the `tools/` file — safety stays in
  `agent.acl`/host `PermissionPolicy`. A directory file cannot self-`Allow`.
- No renaming around the `mcp__<server>__<tool>` convention.
- No new sandbox: `script` reuses the QuickJS VM and its existing limits.
- `tools/` loading must not auto-connect MCP servers during pure config load —
  connection is a session-time, fallible operation (the `add_mcp_server`
  precedent), so a missing `npx`/network failure surfaces at session build, not
  at directory parse.

## 7. Open items for the implementing PR

- Decide whether secrets in `kind="mcp"` specs use `${env:VAR}` interpolation or
  rely on the host injecting `env` — keep secrets OUT of the file (Security in
  `mcp.mdx`).
- Confirm `ScriptLimits` is re-exportable from `tools::program_tool` for the
  spec (currently private; either re-export or mirror the three numeric fields).
- Tests (TDD, before code): load fixture `tools/` with one `mcp` + one `script`
  spec; assert specs parse; assert duplicate-name and unknown-kind fail closed;
  assert a registered `script` tool is non-shadowing and is gated by a Deny
  permission rule.
- Docs: a new `apps/docs/content/docs/en/code/agent-dir-tools.mdx` (YAML
  frontmatter + body, using `mcp.mdx` as the format template) plus a `tools/`
  section in `manual/USER_GUIDE.md`. Add to `meta.json`.

