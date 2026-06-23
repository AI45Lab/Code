# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.1.0] - 2026-06-23

### Changed

- Active skill `allowed-tools` no longer globally deny ordinary session tool calls
  by default. Tool calls continue through permission policy, hooks, HITL, and AHP;
  `SessionOptions::with_active_skill_tool_restrictions(true)` (Node:
  `enforceActiveSkillToolRestrictions`) restores the legacy global restriction
  behavior. Skill-local execution still enforces each skill's `allowed-tools`.

## [4.0.0] - 2026-06-21

Milestone release: **filesystem-first agents**. A single directory
now defines a durable agent by convention — `instructions.md` (role slot),
`agent.acl` (config), `skills/`, `schedules/` (cron), and `tools/` — served by a
`serve` daemon that runs each schedule as a full harness turn. No breaking changes
to existing APIs; the new surface is additive and gated behind the `serve` feature.

### Added

- **Filesystem-first agent directories (`AgentDir`) + the `serve` daemon.** A
  directory with a required `instructions.md` (injected as a prompt *slot*, so the
  harness keeps `BOUNDARIES`/response-format/verification authoritative) plus
  optional `agent.acl`, `skills/`, `schedules/`, and `tools/` loads via
  `AgentDir::load` into existing config objects — no new runtime, no new prompt
  system. `serve_agent_dir` runs each enabled cron schedule on its own durable
  `schedule:<name>` session; every fire is a FULL `AgentSession::send` turn
  (context, tool visibility, safety gate, verification), never a raw model call.
  Cron accepts 5- and 6-field expressions (UTC). Exposed in the Node and Python
  SDKs (`serveAgentDir` / `serve_agent_dir`) returning a `ServeHandle`. Gated
  behind the `serve` Cargo feature.
- **`tools/` declarative tools — `kind: mcp`.** A `tools/<name>.md` with
  `kind: mcp` registers an MCP server into each schedule session through the
  normal `add_mcp_server` path (namespaced `mcp__<server>__<tool>`, gated by the
  session permission policy). Duplicate names and unknown kinds fail closed at load.
- **Rehydrate-on-boot for the serve daemon.** When a `SessionStore` is configured
  (e.g. via `SessionOptions::with_file_session_store`), `serve_agent_dir` now
  *resumes* any schedule whose `schedule:<name>` session already exists in the
  store instead of starting it fresh, so a daemon restart keeps the accumulated
  conversation context. Resume restores history only — the current
  `instructions.md` / `skills/` / `tools/` are re-applied each boot, so editing
  the agent dir still takes effect. With no store configured, every boot starts
  fresh (unchanged). Reuses the existing `Agent::resume_session` path; no new
  persistence machinery.
- **Sandboxed `script` tools for filesystem-first agents (`tools/ kind: script`).**
  A `tools/<name>.md` with `kind: script` now becomes a model-visible tool backed
  by the existing sandboxed QuickJS `program` path — no new sandbox. The spec pins
  the workspace-relative `.js`/`.mjs` `path`, the `allowed_tools` allow-list, and
  the `limits` (timeout / tool-calls / output); the model supplies only `inputs`.
  - New `AgentDirScriptTool` registers through the same non-shadowing
    `register_dynamic_tool` path as builtins/MCP, so a `tools/` entry can add a
    name but never replace a builtin. The model's call to the script tool is
    permission-gated like any tool; the script's *inner* `ctx.tool` calls are
    bounded by the pinned `allowed_tools` list + the QuickJS sandbox (no
    fs/net/proc/env), but are NOT re-checked against the session permission policy.
    The complement to `kind: mcp` (both now ship).
  - The `allowed_tools` list is the security boundary for a directory script, so
    the loader **fails it closed**: an omitted list grants NO tools (not all of
    them); list only the minimum, and avoid high-authority tools unless the
    directory is fully trusted.
  - Fails closed at load (not at first call): a non-`.js`/`.mjs` `path`, a path
    that escapes the workspace (absolute / `..`), an out-of-range sandbox limit
    (zero, or an effectively-unbounded `timeoutMs`), an unknown `kind`, or a
    duplicate tool name is a directory-load error. A `tools/` file is semi-trusted,
    so limits are bounded (≤10 min / ≤1000 calls / ≤16 MiB).
  - The serve daemon installs the agent dir's `tools/` into every schedule
    session, so scheduled turns can call them.

## [3.6.2] - 2026-06-14

Release-engineering fix for 3.6.0/3.6.1 (no library code changes). Both prior
tags published `a3s-code-core` to crates.io but failed every native SDK build,
so npm / PyPI / GitHub Release were skipped.

True root cause: **`sdk/{node,python}/Cargo.lock` were git-ignored** (never
committed), so CI resolved the SDK dependency graph fresh on each release. A
newly-published `alloc-no-stdlib 3.0.0` then got pulled alongside `2.0.4`,
producing a duplicate that breaks `brotli 8.0.3`'s `StandardAlloc: Allocator`
impl. (The 3.6.1 toolchain pin was a misdiagnosis — the build fails the same way
on any toolchain when the lock isn't honored.)

### Fixed

- **Commit the SDK lockfiles** — `sdk/node/Cargo.lock` and
  `sdk/python/Cargo.lock` are now tracked (removed from `.gitignore`), pinning a
  single consistent `alloc-no-stdlib 2.0.4` + `brotli 8.0.3`. CI now builds the
  exact, locally-verified resolution instead of re-resolving. Verified with the
  real `cargo build --release` for both SDKs (not just `cargo check`).

## [3.6.1] - 2026-06-14

Release-engineering fix for 3.6.0 (no library code changes). The 3.6.0 tag
published `a3s-code-core` to crates.io, but the native SDK build jobs failed
because **brotli 8.0.3** (pulled transitively via `tower-http` under the
`ahp`/`s3` features) no longer compiles on the newest stable Rust — so the npm,
PyPI, and GitHub Release artifacts were skipped. This release ships those.

### Fixed

- **Pin the SDK build toolchain** — `publish-node.yml` / `publish-python.yml`
  now use `dtolnay/rust-toolchain@1.94.1` (and pass `rust-toolchain: 1.94.1` to
  `maturin-action`) instead of `@stable`, restoring the known-good build that
  shipped earlier releases. Revisit when the upstream brotli/toolchain break
  clears.
- **Refresh the SDK lockfiles** — `sdk/{node,python}/Cargo.lock` now pin
  `a3s-code-core` to the release version (previously left stale, which forced a
  dependency re-resolution that pulled a second `alloc-no-stdlib`).

## [3.6.0] - 2026-06-14

A system-prompt hardening pass plus framework-vs-host boundary tightening:
inject safety boundaries and live environment grounding, redact secrets from
logs, and wire the LLM-client extension seam through the public API.

### Added

- **`<env>` grounding block** — every augmented system prompt now carries a small,
  always-on environment block (today's date, host platform, working directory),
  computed fresh each turn in `turn_context.rs` (no shell-out). Most importantly
  it pins the current date, which the model otherwise cannot infer past its
  training cutoff.
- **`SessionOptions::with_llm_client`** — hosts can now inject a custom
  `Arc<dyn LlmClient>` (custom/unsupported provider, deterministic record/replay
  client, or HTTP proxy/audit wrapper). The `LlmClient` trait and `Arc<dyn>`
  engine already existed but were only injectable in test code; this wires the
  seam through the public API, bringing the Action-layer backend to parity with
  workspace/memory/store/security (all object-injectable). The `provider/model`
  factory remains the default when unset.

### Security

- **Tool-argument log redaction** — `ToolExecutor` no longer logs raw tool
  arguments at `info!` (which were also exported to OTLP). Bash commands and
  `write`/`edit` file contents can contain secrets; invocations now log only the
  tool name, sorted argument field names, and payload byte size. Full args remain
  available at `trace!` for local debugging. Backs the new "never log secrets"
  prompt boundary.

### Changed

- **System-prompt safety boundaries** — every assembled system prompt (all agent
  styles and delegated subagents) now carries a `## Boundaries` section
  (injection hygiene: treat file/tool/web content as untrusted data, not
  commands; secret handling; defensive-security-only) from a single source
  (`prompts/common/boundaries.md`), injected once in
  `SystemPromptSlots::build_with_style`.
- **Default prompt guidance** — added a library-availability rule (confirm a
  dependency before using it) and clarified that the dedicated read/search/edit
  tools are preferred over shelling out (`cat`/`sed`/`grep`/`find`); `bash` is
  for running commands, builds, and tests. Response-format guidance now
  discourages re-printing already-read code and creating unsolicited report
  `.md` files.

### Fixed

- **Framework no longer writes to the host's stderr** — replaced a stray
  `eprintln!("[DEBUG] HTTP error...")` in the OpenAI client (fired on every
  transport error, also leaking into the host terminal) with `tracing::error!`,
  consistent with the rest of the crate.

### Removed

- **Dead `Planner` trait** (`planning::Planner`) — it was re-exported but had no
  `dyn Planner` dispatch and no consumer; every call site uses `LlmPlanner`'s
  inherent methods directly. Removed per the pruning rule (the real variability,
  the LLM, is swappable via `with_llm_client`). `LlmPlanner` is unchanged.

## [3.4.0] - 2026-05-30

Programmable, deterministic multi-agent orchestration — a grammar for
expressing fan-out, pipelines, and resumable workflows in code (not only via
model-driven delegation), drawn along the framework / host boundary:
the framework owns the grammar + serializable contracts; the host owns
placement, transport, and scheduling. All additions are backward compatible
(new types/methods, new optional fields, new `SessionStore` methods with
default no-op impls).

### Added

- **`AgentExecutor` seam** (`orchestration` module) — the boundary between the
  orchestration grammar and the host's placement/transport/scheduling. The
  in-box `TaskExecutor` runs each step as a child agent locally; a host
  substitutes its own executor to place steps across a cluster.
  `concurrency_hint()` is advisory, not a hard local bound, so orchestration
  scales past a single process. `AgentSession::agent_executor()` /
  `session_store()` expose a session-backed executor + its store.
- **Serializable step contracts** — `AgentStepSpec` (`task_id` / `agent` /
  `description` / `prompt` / `max_steps?` / `parent_session_id?` /
  `output_schema?`) and `StepOutcome` (`+ structured?`), serializable for
  cross-node transport and checkpoints.
- **Combinators** —
  - `execute_steps_parallel` — barrier fan-out, input-order preserving,
    per-branch panic isolation, bounded by the executor's concurrency hint.
  - `execute_pipeline` — per-item chains through stages with **no inter-stage
    barrier** (item A can be in stage 3 while item B is still in stage 1);
    stages are pure spec-builders that branch on the prior outcome.
  - `execute_steps_parallel_resumable` — journals completed steps to a
    `SessionStore` at each step boundary; on resume it skips completed steps
    and re-dispatches the rest. Records only successful steps (a failed step
    retries on resume). The checkpoint is serializable, so a host can resume an
    interrupted workflow on a *different* node.
- **Schema-forced step output** — a step carrying `output_schema` returns a
  schema-validated object in `StepOutcome.structured` (reuses the
  structured-output coercion + repair). A coercion failure demotes the step to
  unsuccessful, so callers never treat unvalidated text as the promised object.
- **`WorkflowCheckpoint`** (`schema_version` / `workflow_id` / `steps` /
  `checkpoint_ms`) + `SessionStore::{save,load,delete}_workflow_checkpoint`
  (default no-ops; the file store writes crash-atomically). Loads from a
  future, incompatible schema version are rejected.
- **SDK grammar (Node + Python)** — `session.parallel(specs)`,
  `session.pipeline(items, stages)`, `session.parallelResumable(specs,
  workflowId)` (Node, camelCase) / `parallel` / `pipeline` /
  `parallel_resumable` (Python, snake_case). Pipeline stages are JS/Python
  callbacks `(ctx) -> spec | null`; the bridges fail closed — a hung,
  null-returning, or raising stage stops only its own chain. (A Node stage
  callback must not throw — return `null` on error, same constraint as
  `setBudgetGuard`.)
- **`LoopCheckpoint::ensure_loadable()`** — loads from a future, incompatible
  loop-checkpoint schema version are now rejected at the store layer (both
  file and memory), honoring the documented contract.

### Changed

- The resumable combinator now distinguishes "no checkpoint" from an
  *unreadable* one: an unreadable (e.g. future-version) checkpoint logs a
  warning and re-runs the workflow from scratch rather than silently swallowing
  the error.
- Documented the FFI panic-safety contract in each SDK's module doc (napi 2.x
  does not catch panics in sync `#[napi]` bodies by default; PyO3 0.23 catches
  `#[pyfunction]` / `#[pymethods]` bodies). No code change — both boundaries
  were audited panic-safe.

### Tests

- Persisted-schema round-trip fuzz extended to the new migratable types
  (`AgentStepSpec`, `StepOutcome`, `WorkflowCheckpoint`) — round-trip stability
  + forward/backward compat. Comprehensive unit **and** real-LLM integration
  tests for the orchestration layer (parallel fan-out, multi-item pipeline, the
  resume path, nested-schema coercion) run against `.a3s/config.acl`.

## [3.3.0] - 2026-05-29

Cluster-grade runtime: everything needed for a host platform
to run long-lived agent sessions across many nodes — graceful shutdown,
multi-tenant identity, cost governance, deterministic replay, crash-tolerant
runs, and bounded in-memory state — plus an adversarial-review hardening
pass. All additions are backward compatible (new methods, new optional
fields, new `SessionStore` trait methods with default no-op impls).

### Added

- **Session / Agent lifecycle control.**
  - `AgentSession::close()` is now a full graceful stop: flips `is_closed`
    (further `send`/`stream` fast-fail with `CodeError::SessionClosed`),
    cancels the active run, all in-flight delegated subagent tasks, and
    pending HITL confirmations. `AgentSession::is_closed()` accessor.
  - Agent-side session registry: `Agent::list_sessions()`,
    `close_session(id)`, `close()` (also disconnects global MCP), and
    `is_closed()`. Sessions are tracked by `Weak` ref and pruned lazily.
  - Session-level `CancellationToken` parent: every run derives its token
    via `child_token()`, so `close()` cascades to all in-flight work.
    `AgentSession::session_cancel_token()` exposes it for embedders.
- **Host-provided identity labels** — `tenant_id`, `principal`,
  `agent_template_id`, `correlation_id` on `SessionOptions` (builder
  methods + accessors), persisted in `SessionData`, restored on resume.
  Framework treats them as opaque; the host drives multi-tenant
  aggregation / billing / tracing. Exposed on both SDKs.
- **`BudgetGuard` cost/quota contract** (`budget` module) — host-supplied
  `check_before_llm` / `record_after_llm` / `check_before_tool`, consulted
  at the LLM call site. `Deny` aborts with `CodeError::BudgetExhausted`;
  `SoftLimit` emits an event and proceeds. SDK bridges: a Python class
  (`opts.budget_guard`) and Node `session.setBudgetGuard({...})`. The Node
  bridge fails **closed** (timeout / unreadable return → deny).
- **`HostEnv` (IdGenerator + Clock) injection** (`host_env` module) —
  replace the default UUID + wall-clock pair for deterministic replay of a
  run on another node. `SequentialIdGenerator` / `FixedClock` helpers.
- **Loop checkpoints + run resumption** (`loop_checkpoint` module) — the
  agent loop persists a `LoopCheckpoint` after each completed tool round
  (when a `SessionStore` is configured); `AgentSession::resume_run(run_id)`
  replays from the last boundary on any node sharing the store, continuing
  cumulative token/tool-call accounting. `SessionStore` gains
  `save/load/delete_loop_checkpoint`; file writes are crash-atomic.
- **`SessionRetentionLimits`** (`retention` module) — optional FIFO caps on
  the in-memory run store (runs + per-run events), trace sink, and terminal
  subagent task snapshots, so long-running sessions don't grow unbounded.
  Exposed on both SDKs. Default is unbounded (no behavior change).
- **MCP idle disconnect** — `McpManager::disconnect_idle(threshold_ms)` and
  `Agent::disconnect_idle_mcp(...)` (both SDKs) reap quiet MCP servers
  (releasing FDs / background workers) while keeping their config for
  on-demand reconnect.
- **Cluster `AgentEvent` variants** — `BudgetThresholdHit`,
  `PassivationRequested`, `PeerInvocation`: platform-level events a host
  emits via `HookExecutor` so in-session code can react uniformly.
- `SessionStore` now persists the subagent task tracker across
  save/resume (`save/load_subagent_tasks`), so a migrated session keeps a
  queryable history of its delegated child runs.
- New errors: `CodeError::SessionClosed`, `CodeError::BudgetExhausted`.

### Changed

- `resume_run` continues cumulative metrics (`total_usage`,
  `tool_calls_count`) from the checkpoint instead of restarting at zero.
- Run-store and subagent-tracker FIFO eviction now hold their parallel
  maps under a single canonical lock order, so eviction is atomic with
  respect to concurrent record/cancel (no transient map inconsistency).

### Fixed

- **Loop checkpoint leak**: checkpoints were written after every tool round
  but never deleted — unbounded disk/memory growth on every completed run.
  They are now removed when a run reaches a terminal state in-process; only
  a true crash leaves one for resume.
- **`event_count` corruption**: restoring a session whose per-run event
  buffer had been trimmed reset the cumulative `event_count` to the trimmed
  length. The persisted cumulative count is now preserved.
- **Node `BudgetGuard` fail-open**: a hung or slow guard silently *allowed*
  the LLM call (disabling enforcement). It now fails **closed** (deny) on
  timeout and on an unreadable return.
- **MCP timestamp leak**: `touch()`-without-connect orphan timestamps are
  now purged by `disconnect_idle`.
- Session registry dangling `Weak` entries are pruned on `Agent::close()`.

### Known limitations

- Node `BudgetGuard` callbacks **must not throw** — due to a napi-rs
  constraint a thrown exception aborts the host process at return-value
  conversion. Wrap guard logic in try/catch and return a decision. Hangs
  are handled safely (fail-closed timeout). The Python `BudgetGuard`
  catches exceptions and is unaffected.

## [3.2.1] - 2026-05-24

### Added

- Python SDK: small pure-Python **bootstrap** shim published to PyPI as
  `a3s-code`. On first `import a3s_code` it downloads the matching
  native wheel for the current interpreter/platform from this repo's
  GitHub Releases, verifies the wheel's sha256 against the release
  manifest, extracts the compiled `_native` extension into
  `~/.cache/a3s-code/<version>/`, and registers it as
  `sys.modules["a3s_code._native"]`. Subsequent imports use the cache.
  Source under `sdk/python-bootstrap/`.
  - Environment knobs: `A3S_CODE_CACHE_DIR`, `A3S_CODE_RELEASES_BASE_URL`,
    `A3S_CODE_SKIP_HASH_CHECK`.
  - 15 unit tests + 1 live download test gated on
    `A3S_CODE_BOOTSTRAP_LIVE=1`.
  - New workflow `publish-python-bootstrap.yml`, wired after
    `publish-python` in `release.yml`.
- `scripts/check_release_versions.sh` now also validates the bootstrap
  package version and the runtime `__version__` literal.
- `release.sh` now bumps the bootstrap version in lockstep with the
  core release.

### Fixed

- `pip install a3s-code` works again from v3.2.1, restored after v3.2.0
  could only push a single wheel to PyPI under the quota cap.

## [3.2.0] - 2026-05-24

### Added

- Added a queryable subagent task tracker so callers can observe delegated
  child runs by `task_id` instead of scanning `run_events()`. The tracker is
  a materialized view over the existing `SubagentStart` / `SubagentProgress`
  / `SubagentEnd` event stream — the stream remains the authoritative record.
- Added three new APIs on `AgentSession` (and mirrored bindings on the Node
  and Python SDKs):
  - `subagent_task(task_id)` — look up a task snapshot by id.
  - `subagent_tasks()` — list every delegated subagent task observed in this
    session, oldest first.
  - `pending_subagent_tasks()` — list only tasks still in `running` state.
- Added emission of `SubagentProgress` events from the child loop forwarder.
  Two milestones are surfaced today: `status = "tool_completed"` after each
  child tool ends (metadata: tool, exit_code, output_bytes, optional
  error_kind) and `status = "turn_completed"` after each child LLM turn
  (metadata: turn, prompt/completion/total tokens). Noisy events (TextDelta,
  ToolStart, ToolOutputDelta, nested subagent events) are intentionally not
  translated; consumers needing token-level streaming should subscribe to the
  raw event stream directly.
- Added `SubagentStatus::Cancelled` and `AgentSession::cancel_subagent_task(id)`
  for interrupting in-flight delegated child runs without cancelling the parent
  run. Bindings on both SDKs (`session.cancelSubagentTask(taskId)` /
  `session.cancel_subagent_task(task_id)`). A late `SubagentEnd` from a
  cancelled child does not downgrade the terminal status — it stays
  `Cancelled`.
- Added `SubagentTaskSnapshot` carrying `task_id`, `parent_session_id`,
  `child_session_id`, `agent`, `description`, `status`, `started_ms`,
  `updated_ms`, optional `finished_ms` / `output` / `success`, and a
  `progress` log. The Cancellation path also propagates a real cancellation
  token into the child loop via `AgentLoop::execute_with_session`, so the
  signal honors existing LLM-streaming yield points.
- Added `InMemorySubagentTaskTracker` and `SubagentProgressEntry` to the
  public crate-root re-exports of `a3s-code-core` alongside the existing
  `SubagentStatus` / `SubagentTaskSnapshot` types.

### Changed

- Marked `SubagentStatus` `#[non_exhaustive]` so future variants can be added
  without a major version bump.
- Reshaped the Node SDK type layout to survive `napi-rs` regeneration. The
  build now writes generated declarations to `generated.d.ts`; hand-authored
  types that mirror JSON wire shapes (`ToolErrorKind`, `VerificationStatus`,
  `VerificationCheck`, `VerificationReport`, `ToolArtifact`) now live in
  `extra-types.d.ts`; the published `index.d.ts` is a small hand-authored
  aggregator that re-exports both. The `types` field in `package.json` still
  points at `index.d.ts`, so consumer imports are unchanged. A new
  `npm run test:types` script type-checks the aggregator to guard against
  future regressions.

### Fixed

- Fixed `TaskExecutor::execute` and `execute_background` so the emitted
  `SubagentStart` carries the real parent session id (previously
  `String::new()`), and `execute_background` returns the same `task_id`
  that appears in lifecycle events (previously a throwaway id). The
  background path also pre-emits `SubagentStart` synchronously so callers
  that query the tracker immediately after scheduling do not race the
  spawned task.

### Packaging

- The Python SDK no longer ships native wheels to PyPI. The project
  grew past PyPI's default 10 GB per-project quota, and binary wheels
  for the full Rust × CPython × platform matrix consume that budget
  fast. From v3.2.0 onwards the canonical wheel host is GitHub
  Releases — see README for the install command. Versions up to 3.1.0
  remain installable from PyPI for backward compatibility.

### Breaking

- `TaskExecutor::execute`, `execute_parallel`, and `execute_background` now
  take an additional `parent_session_id: Option<&str>` (or
  `Option<String>` for the background variant) so the emitted lifecycle
  events can be correctly associated with the parent session. Direct
  callers of `TaskExecutor` need to pass `None` (or the parent session id)
  to keep current behavior.
- `register_task_with_mcp` gained a trailing
  `subagent_tracker: Option<Arc<InMemorySubagentTaskTracker>>` parameter so
  the session bootstrap path can share a single tracker Arc with the
  executor and the live `AgentSession`. Pass `None` to opt out.

## [3.1.0] - 2026-05-23

### Added

- Added Claude Code-style automatic subagent delegation. When a request matches
  multiple independent specialists, the runtime can pre-run those child agents
  through one bounded `parallel_task` call and feed the gathered context back
  into the main turn.
- Added `auto_delegation` configuration and session overrides, including the
  global `auto_parallel` kill switch. Setting `auto_parallel = false` disables
  automatic parallel child-agent fan-out while keeping manual `parallel_task`
  available.
- Added `max_parallel_tasks` as the shared sibling fan-out limit for
  `parallel_task`, delegated plan waves, and safe parallel write batches.
- Added a reusable ordered parallel executor so concurrent child results remain
  deterministic and individual task failures are isolated.
- Added native `.a3s/agents` and `~/.a3s/agents` subagent discovery with
  recursive loading. `.claude/agents` remains a compatibility source, but
  `.a3s/agents` wins when the same agent name appears in both locations.
- Added Claude-style markdown agent compatibility for `tools`,
  `allowedTools`, and `disallowedTools` frontmatter. `tools` behaves as an
  allowlist and `disallowedTools` is a denylist that takes precedence.
- Added direct worker/subagent APIs across Rust, Node.js, and Python:
  `WorkerAgentSpec`, `AgentDefinition`, `session_for_worker`, live worker
  registration, and `task` / `parallel_task` helpers.
- Added real-provider smoke coverage for automatic parallel delegation and
  built-in subagent execution using `.a3s/config.acl`.

### Changed

- Aligned built-in subagent names and aliases with Claude Code conventions:
  `general-purpose` aliases to `general`; `verify` / `verifier` alias to
  `verification`; `code-review` / `reviewer` alias to `review`.
- Tightened built-in subagent permission boundaries. Read-only and review-style
  agents now default-deny undeclared tools, and all built-ins deny recursive
  `task` / `parallel_task` delegation to prevent unbounded nesting.
- Collapsed independent delegated plan waves into a single `parallel_task`
  operation where possible, rather than launching serial sibling `task` calls.
- Updated Node and Python SDK documentation to prefer the single delegation
  surface backed by the core `task` and `parallel_task` tools.

### Fixed

- Fixed release helper version checks so they no longer reference the removed
  `cli/` package.
- Fixed explicit subagent trigger parsing so normal phrases such as "use the
  plan" are not mistaken for a `plan` subagent request.

## [3.0.0] - 2026-05-20

### Added (Phase 8 — typed-error SDK alignment)

- New public `ToolErrorKind` enum (`#[non_exhaustive]`, JSON-tagged on
  the `type` discriminator) carries structured tool failure reasons
  from the Rust core all the way to SDK callers without losing the
  type. Six variants: `version_conflict`, `remote_git_conflict`,
  `not_found`, `invalid_argument`, `unsupported`, `timeout`.
- New optional `error_kind` field on `ToolOutput`, `ToolResult`, and
  `ToolCallResult`, plus a matching field on `AgentEvent::ToolEnd` for
  streaming consumers.
- Built-in `edit` and `patch` tools populate `error_kind` via
  `ToolErrorKind::from_workspace_error` whenever a `WorkspaceError`
  variant maps to a typed kind. The human-readable `output` /
  `content` message is unchanged so the model still gets the retry
  hint; SDK callers now have a programmatic discriminator next to it.
- Node SDK: new `errorKindJson` field on `ToolResult` and `AgentEvent`
  (JSON-encoded `ToolErrorKind`) plus a new `ToolErrorKind` TypeScript
  discriminated-union type in `index.d.ts`.
- Python SDK: new `error_kind_json` (raw) and `error_kind` (parsed
  dict) properties on `ToolResult` and `AgentEvent`.

This closes the v3.0 typed-error gap: until this commit the typed
`WorkspaceError` enum on the Rust trait surface was effectively
re-stringified at the SDK boundary, forcing JS/Python callers to
regex-match the output to detect e.g. concurrent-modification
conflicts. They now `switch` / `match` on `error_kind.type` instead.

### ⚠️ Breaking changes (3.0.0)

- **`WorkspaceFileSystem` and `WorkspaceFileSystemExt` trait methods now
  return `WorkspaceResult<T>` instead of `anyhow::Result<T>`.** The new
  result type wraps the typed `WorkspaceError` enum
  (`#[non_exhaustive]`) with structured variants for `NotFound`,
  `VersionConflict`, `RemoteGitConflict`, `InvalidArgument`, `Timeout`,
  `Unsupported`, and a `Backend(anyhow::Error)` catch-all. Callers that
  used `?` to lift errors into `anyhow::Result` keep working unchanged
  thanks to the blanket `From<WorkspaceError> for anyhow::Error` impl;
  callers that previously did `err.downcast_ref::<WorkspaceVersionConflict>()`
  now `match` on the typed variant directly:
  ```rust
  // before:
  if e.downcast_ref::<WorkspaceVersionConflict>().is_some() { ... }
  // after:
  if matches!(e, WorkspaceError::VersionConflict(_)) { ... }
  ```
  `WorkspaceServices::read_for_edit`, `write_for_edit`, and the generic
  `run_with_timeout` (now polymorphic in the error type) follow the
  same shape. The other 5 traits (`WorkspaceCommandRunner`,
  `WorkspaceSearch`, `WorkspaceGit`, `WorkspaceGitStashProvider`,
  `WorkspaceGitWorktreeProvider`) **still return `anyhow::Result`** —
  their migration to `WorkspaceResult` will be additive (non-breaking)
  in a future v3.x release.

### Added

- Added `S3WorkspaceBackend` — an S3-compatible workspace backend that lets
  built-in file tools (`read`, `write`, `edit`, `patch`, `ls`) operate
  directly against any S3-compatible endpoint (AWS S3, MinIO, RustFS, R2,
  Backblaze B2, ...). Gated behind the new `s3` Cargo feature.
- Added `S3BackendConfig` builder for configuring endpoint, region, static
  or session-token credentials, force-path-style, request timeout, and
  bucket prefix.
- Added `WorkspaceServices::s3()` factory and `WorkspaceServices::from_s3_backend()`
  helper. The factory installs a 60s default per-operation timeout and
  declines `bash`, `git`, `grep`, and `glob` capabilities — capability
  gating automatically hides those tools from the model so it cannot
  call operations the backend cannot service.
- Exposed `S3WorkspaceBackend` in the Node and Python SDKs alongside
  `LocalWorkspaceBackend`. Configuration uses the same option surface
  (`workspaceBackend` / `workspace_backend`).
- `S3WorkspaceBackend::read_text` now enforces a configurable size ceiling
  (`S3BackendConfig::max_read_bytes`, default 10 MiB) by inspecting
  `Content-Length` on the `GetObject` response before consuming the body.
  Oversized objects are rejected with a clear error and never buffered
  into memory. Responses without a `Content-Length` header are refused
  rather than risking OOM.
- Added optional `WorkspaceFileSystemExt` trait for backends that expose
  compare-and-swap writes, plus a `WorkspaceVersionConflict` error type.
  `S3WorkspaceBackend` implements it via ETag + `If-Match` on `PutObject`.
  The `edit` and `patch` tools now capture the ETag during the read and
  reject the write on version mismatch (HTTP 412), surfacing a typed
  "Concurrent modification detected" error so the model can re-read and
  retry instead of silently clobbering a concurrent writer.
  `WorkspaceServices::read_for_edit` and `write_for_edit` are the new
  helpers tools should use for any read-modify-write cycle; backends
  without versioning (e.g. local) transparently fall through to plain
  `read_text` / `write_text`.
- `S3WorkspaceBackend` now implements `WorkspaceSearch` (degraded `grep` /
  `glob` via `LIST` + `GET` + regex). Off by default; opt in via
  `S3BackendConfig::enable_search(true)`. Hard ceilings on objects scanned
  per call (`max_objects_scanned`, default 500) and per-object body size
  for `grep` (`max_grep_bytes_per_object`, default 1 MiB) bound the API
  cost. Hitting either ceiling sets `WorkspaceGrepResult::truncated = true`.
  Glob patterns follow the local backend's recursion convention: `*.rs`
  matches the immediate level, `**/*.rs` recurses.
- `S3WorkspaceBackend::grep` now downloads candidate objects in parallel
  via `futures::stream::buffer_unordered`. Concurrency defaults to 8 and
  is configurable via `S3BackendConfig::search_concurrency` (also
  exposed on both SDKs). Output ordering remains deterministic — results
  are sorted by workspace path before assembly — so callers see the same
  layout regardless of S3 response timing.

### Added

- Internal `workspace::conformance` module (test-only) codifies the
  behavioural invariants every backend implementing
  `WorkspaceFileSystem` (and optionally `WorkspaceFileSystemExt`) must
  satisfy. Two public entry points, `assert_filesystem_conformance` and
  `assert_filesystem_ext_conformance`, are run against
  `LocalWorkspaceBackend` and a new `InMemoryFileSystem` reference
  backend so the contract is exercised both over real I/O and an ideal
  HashMap-backed implementation. Future backends (GCS, container,
  browser) gain a regression suite for free — when the conformance set
  grows after a production incident, every backend running it picks up
  the new test automatically.

### Fixed

- `WorkspaceServices::with_remote_git` previously rebuilt the services
  through `WorkspaceServicesBuilder`, which silently dropped `local_root`
  (and would silently drop any future field). The decorator now goes
  through a new internal `with_git_provider` helper that uses an explicit
  struct literal — adding a new field to `WorkspaceServices` now triggers
  a compile error in every decorator, forcing a deliberate decision.
- `RemoteGitBackend::diff` previously deserialised the entire response
  body before applying `max_diff_bytes`, so a misbehaving gitserver
  returning a multi-gigabyte JSON could exhaust client memory. The diff
  path now streams the body with a hard cap (`max_diff_bytes * 4`, floor
  64 KiB), rejecting requests upfront when `Content-Length` advertises an
  oversized body and aborting the stream mid-flight when chunked encoding
  hides the size. The soft `max_diff_bytes` display truncation is
  unchanged.

### Changed

- `S3WorkspaceBackend::list_dir` now errors with "S3 path not found" when
  the LIST returns zero entries on a non-root path, matching the local
  backend's behaviour. Previously a missing prefix silently returned
  `Ok(vec![])`, masking typos. Paths that exist only as S3 zero-byte
  directory markers still return `Ok(vec![])`.
- Every S3 API call (`GET`, `PUT`, `LIST`) on `S3WorkspaceBackend` now
  emits a structured `tracing::debug!` event with fields `op`, `bucket`,
  `target`, `bytes`, `outcome`, `duration_ms`. Hosts can meter S3 cost
  by subscribing to these events without the backend taking a dependency
  on any metrics framework.
- Node and Python SDKs now expose the workspace hardening options added
  in this release. The Node `JsS3BackendConfig` and Python
  `S3WorkspaceBackend` constructor accept `maxReadBytes` /
  `max_read_bytes`, `searchEnabled` / `search_enabled`,
  `maxObjectsScanned` / `max_objects_scanned`, and
  `maxGrepBytesPerObject` / `max_grep_bytes_per_object`. A new
  `RemoteGitBackendConfig` class (Python) / `JsRemoteGitBackendConfig`
  shape (Node) and a top-level `remoteGit` / `remote_git` session
  option let SDK callers attach `RemoteGitBackend` on top of any
  workspace backend. Passing `remoteGit` without `workspaceBackend`
  raises a clear error.
- Added `RemoteGitBackend` — an HTTP/JSON `WorkspaceGit` client that
  brings the `git` tool to non-local workspaces (S3 today; future
  container / DFS). Implements `WorkspaceGit` in full and
  `WorkspaceGitStashProvider`; deliberately omits `WorkspaceGitWorktreeProvider`
  because worktrees do not map to a remote service. The protocol is
  specified in `apps/docs/content/docs/en/code/rfcs/workspace-remote-git.mdx`.
  - New types: `RemoteGitBackend`, `RemoteGitBackendConfig`,
    `RemoteGitConflict` (anyhow-downcastable for recoverable 409 / 422
    responses such as `WORKING_TREE_DIRTY` and `BRANCH_EXISTS`).
  - New factory: `WorkspaceServices::with_remote_git(config)` on any
    existing `Arc<WorkspaceServices>` to attach remote git on top of an
    S3 (or local) filesystem backend.
  - Client-side ceilings: `request_timeout` (default 30 s),
    `max_log_entries` (default 200), `max_diff_bytes` (default 1 MiB).
  - Per-call `tracing::debug!` event with fields `op`, `repo_id`,
    `status`, `bytes`, `outcome`, `duration_ms`, mirroring the S3
    metering shape so a single subscriber meters both.
  - Authentication: bearer token (header `Authorization: Bearer <token>`)
    or mTLS via `client_cert_pem` + `client_key_pem` (PKCS#8 PEM key for
    the `rustls-tls` backend). Setting only one of the mTLS pair fails
    at construction.

### Changed

- Restructured `core/src/workspace.rs` into a `workspace/` module with
  `workspace/mod.rs` (abstract traits + `WorkspaceServices`),
  `workspace/local.rs` (`LocalWorkspaceBackend`), and `workspace/s3.rs`
  (`S3WorkspaceBackend`). No behavioural change for existing callers.

## [2.6.0] - 2026-05-18

### Added

- Added `WorkspaceServices` capability abstraction (`core/src/workspace.rs`)
  that lets the host supply file system, command runner, search, and Git
  providers behind the stable built-in tool contract. The default
  `LocalWorkspaceBackend` preserves existing local-filesystem behavior, while
  DFS, browser, container, and remote backends can be assembled via
  `WorkspaceServicesBuilder`.
- Added `SessionOptions::with_workspace_backend()` (alias
  `with_workspace_services`) so callers can opt-in to non-local workspaces
  without changing tool schemas.
- Added capability-driven tool gating: `bash`, `grep`, `glob`, and `git` are
  only registered when the workspace backend declares the matching capability,
  preventing models from invoking tools the backend cannot service.
- Added `Session::write_file`, `Session::ls`, `Session::edit_file`, and
  `Session::patch_file` direct-tool APIs in core, Node, and Python SDKs,
  alongside the existing `read_file` / `bash` / `glob` / `grep`.
- Added `LocalWorkspaceBackend` class to the Node and Python SDKs as the
  explicit typed form of the default backend and the option surface for future
  remote/browser/DFS workspaces.
- Added `workspace_services` to `ChildRunContext` so child runs inherit the
  parent's workspace backend.
- Added 17 unit + integration tests covering virtual path resolution, capability
  downgrade, contract-level tool routing for files / search / bash / git
  through pluggable backends, and session-level direct-tool dispatch.

### Changed

- Refactored built-in tools `read`, `write`, `edit`, `patch`, `ls`, `bash`,
  `grep`, `glob`, and `git` to route operations through `WorkspaceServices`
  instead of hard-coded local filesystem calls. Local behavior is unchanged.
- Centralized workspace-boundary path checks in
  `ToolContext::resolve_workspace_path`, removing duplicated canonicalization
  logic from `ToolExecutor::execute`.

### Fixed

- Removed two `clippy::useless_conversion` warnings in
  `core/tests/test_ahp_idle_with_llm.rs` so `cargo clippy --all-targets` is
  clean.

### Documentation

- Updated `README.md`, Node SDK README, and Python SDK README with workspace
  backend usage and the new direct-tool API surface.

## [2.5.0] - 2026-05-12

### Added

- Added `ConfirmationInheritance` enum for controlling how child runs resolve Ask
  decisions: `AutoApprove` (default), `DenyOnAsk`, and `InheritParent`.
- Added `confirmation_inheritance` field to `WorkerAgentSpec` in Node and Python
  SDKs, allowing fine-grained control over child run confirmation behavior.
- Added `ChildRunContext` for explicit parent capability inheritance, ensuring
  child runs properly inherit permission checkers and confirmation policies.
- Added comprehensive integration tests for task delegation with real LLM calls
  and mock LLM contract tests for permission and confirmation inheritance.
- Added SDK integration tests for `confirmation_inheritance` in both Node and
  Python SDKs with `.a3s/config.acl` configuration support.

### Fixed

- Fixed task delegation to properly inherit permission checker from agent
  definition in child runs (Issue #28).
- Fixed child runs to respect parent's confirmation policy when using
  `InheritParent` mode.

### Changed

- Unified `AgentDefinition` → `AgentConfig` conversion via `apply_to()` method
  for consistent configuration application.
- Refactored `ToolExecutor` to remove redundant `guard_policy` field, relying
  on `PermissionChecker` for all permission decisions.

### Documentation

- Updated Node and Python SDK READMEs with `confirmation_inheritance` examples
  and usage guidance.
- Updated English and Chinese documentation for teams and tasks with worker
  agent confirmation inheritance patterns.

## [2.4.0] - 2026-05-11

### Added

- Added `generate_object` built-in tool for structured JSON output with schema
  validation, automatic repair, and streaming partial objects. Works across all
  providers via tool-calling mode.
- Added `llm::structured` module with four output modes (tool, prompt, strict,
  json), robust JSON extraction from dirty LLM output, partial JSON parser for
  streaming, and a built-in JSON Schema validator supporting `anyOf`/`oneOf`,
  nullable types, `additionalProperties`, `pattern`, and numeric ranges.
- Added streaming partial object support: `generate_object` emits
  `tool_output_delta` events with progressively complete JSON snapshots.
- Added comprehensive documentation: structured output example (EN/CN), contract
  review tutorial (EN/CN), and 7 additional core mechanism tutorials (PTC,
  streaming, session persistence, skills, MCP, security/HITL, hooks, memory).

### Fixed

- Fixed Shiki build error in docs site caused by unsupported `acl` language
  identifier in code blocks (replaced with `text`).

## [2.3.0] - 2026-05-09

### Added

- Added compact, object-shaped SDK APIs for long-lived integrations:
  `send(...)`, `run(...)`, `stream(...)`, `task(...)`, `tasks(...)`,
  `git(...)`, `addMcp(...)` / `add_mcp(...)`, `removeMcp(...)` /
  `remove_mcp(...)`, and `mcps()`.
- Added live run/tool observability through active tool snapshots and richer
  run replay APIs across Rust, Node.js, and Python SDKs.
- Added a durable SDK API design contract under `manual/SDK_API_DESIGN.md`.
- Added Python SDK parity for worker agents, HITL confirmation policy/control,
  session-for-worker, live worker registration, and session close.

### Changed

- Split the large agent and session API implementation files into focused
  runtime modules for maintainability.
- Made AHP the single harness/advisory/control plane with richer event context,
  heartbeat state, runtime state snapshots, and decision mapping.
- Updated docs and examples to prefer short SDK method names while retaining
  long compatibility aliases.
- Re-exported `ActiveToolSnapshot` from the Rust core crate root.

### Removed

- Removed the obsolete sidecar/copilot/BTW/strategize/BTE mechanism and related
  prompts, docs, configs, and examples. Background advice, context supplements,
  and PTC proposals now belong to the caller or AHP harness.

---

## [2.0.0] - 2026-05-02

### Changed

- Promoted A3S Code package metadata to `2.0.0` across Rust core, Node.js SDK, and Python SDK.
- Standardized runtime configuration on ACL (`.acl`) and explicit `env(...)` credential injection.
- Reworked the public API surface around `Agent`, `AgentSession`, and 2.0-compatible session/control-plane primitives.

### Added

- Release-blocking real-provider integration test for `.a3s/config.acl` environment-variable injection.
- No-network integration coverage, script dry-run support, and literal-config extraction for MiniMax ACL `env(...)` resolution.
- Release validation scripts for local core tests, AHP feature tests, version consistency, patch hygiene, and real-provider ACL smoke tests.

### Removed

- Legacy HCL config artifacts and stale prompt tests that no longer match the 2.0 ACL runtime.

---

## [v1.8.6] - 2026-04-10

### Fixed

#### web_search Tool

- **Issue #25 Fix**: The `web_search` tool now returns an error when unknown parameters are passed (e.g., `engine` instead of `engines`). Previously, unknown parameters were silently ignored, causing confusion when users specified the wrong field name.

### Changed

- `engines` parameter type changed from `string` to `array` in schema to match actual API
- Updated a3s-search integration to v1.0.0

---

## [v1.8.5] - 2026-04-05

### Added

#### Git Built-in Tool

- **Built-in Git Client**: New `git` tool with auto-install support for Windows, macOS, and Linux. Downloads official pre-built git binaries to `~/.local/git/bin/` when git is not available - no package manager required.

  Full git operations: `status`, `log`, `branch`, `checkout`, `diff`, `stash`, `remote`, `worktree`

- **Git Convenience Methods**: Python SDK (`session.git(...)`) and Node SDK (`session.git(...)`) convenience methods for git operations.

#### System Prompt Updates

- Updated all system prompts to reference "A3S Code" instead of "Claude Code"
- Updated skill references to use `a3s-lab/code-skills`

### Removed

- **Document Parser**: Removed `composite_document_parser` and `document` modules and all related code. This feature was not fully implemented and has been removed to simplify the codebase.

- **Agentic Search/Parse Tools**: Removed `agentic_search` and `agentic_parse` built-in tools.

- **Git Worktree Tool**: Replaced by the new unified `git` tool with `worktree` subcommand.

### Changed

- **Tool Count**: Updated built-in tool count from 15 to 16 to reflect new git and box tools.
- **Documentation**: Updated all documentation to reflect new tool names and capabilities.

---

## [v1.6.0] - 2026-04-02

### Added

#### Document Parsing

- **XLSB (Excel Binary) Support**: Added calamine-based BIFF12 parsing for XLSB files with proper cell value extraction, supporting Float, Int, Bool, DateTime, DateTimeIso, and DurationIso types. Significantly improves table fidelity for .xlsb files.

- **HWPX Table Extraction**: Added structured table extraction from Korean HWPX documents. Parses `tbl/tr/tc` XML hierarchy and includes `structured_payload` for `tables[]` output.

- **Vision OCR Provider**: New OCR backend supporting OpenAI-compatible vision APIs for document OCR fallback.

  ```hcl
  document_parser {
    ocr {
      enabled  = true
      model    = "openai/gpt-4.1-mini"
      api_key  = "sk-..."
      base_url = "https://api.openai.com/v1"  # optional
      prompt   = "Extract all text from this document..."
      max_images = 8
      dpi     = 144
    }
  }
  ```

  Provider priority: External provider > Vision API (if model+api_key configured) > Builtin tesseract

#### Search Ranking

- **Tabular Query Intent Detection**: Automatically detects when queries relate to tables (keywords: table, column, row, spreadsheet, excel, csv, cell, data, record, etc.) and boosts table line matches by +10 keyword hits plus 1.3x relevance multiplier.

- **Heading Inheritance Boost**: When search matches appear under headings that also match the query, those matches receive a relevance boost (up to 1.3x). Looks backwards to find the closest preceding heading.

### Changed

#### Configuration

- `DocumentOcrConfig` extended with new fields:
  - `provider: Option<String>` - Backend selection ("vision" or "builtin")
  - `base_url: Option<String>` - Custom API endpoint
  - `api_key: Option<String>` - API authentication

#### Dependencies

- Added `calamine = "0.26"` for XLSB parsing
- Added `reqwest/blocking` feature for Vision API HTTP calls

### Fixed

- Test assertion: `paged_text_blocks_reflow_two_column_preserves_paragraph_breaks` - Corrected expected string "Parser metadata now tracks OCR" vs "Parser metadata now tracks OCR backend"

---

## [v1.5.8] - 2026-03-07

### Added

- Phase 1 structured result surfaces:
  - `structured_payload` exposed in `agentic_parse` output and metadata
  - Table payloads in stable machine-readable form
  - Page-level data in `agentic_parse` output and metadata
  - Stable `tables[]`, `pages[]`, `elements[]` outputs

- Phase 2 PDF extraction improvements:
  - lopdf position-aware text extraction
  - Reduced dependence on weak text fallbacks
  - Position-aware table detection

- `agentic_search` enhancements:
  - Chunk context consumption
  - Tabular content consumption
  - Page numbers and locators support

### Changed

- `ParsedDocument` extended with `tables: Vec<StructuredTable>` and `pages: Vec<PageInfo>`

### Fixed

- Windows shell compatibility improvements

---

## [v1.5.7] - 2026-02-28

### Added

- Runtime session header support for OpenAI configs
- Cross-platform environment variable expansion in tests

---

## [v1.5.6] - 2026-02-20

### Added

- Enhanced agent config, document parser, LLM, tools, and SDKs
- Host shell environment propagation to tool commands

---

## [v1.5.5] - 2026-02-10

### Added

- Zhipu AI client (`ZhipuClient` formerly `GlmClient`)
- Duplicate tool call circuit breaker
- Streaming fallback support
- `agentic_parse` skill

---

## [v1.5.4] - 2026-01-28

### Added

- Session-local skill registries

---

## [v1.5.3] - 2026-01-15

### Added

- Tool schema hardening
- Slash command output restoration
