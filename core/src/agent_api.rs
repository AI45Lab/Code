//! Agent Facade API
//!
//! High-level, ergonomic API for using A3S Code as an embedded library.
//!
//! ## Example
//!
//! ```rust,no_run
//! use a3s_code_core::Agent;
//!
//! # async fn run() -> anyhow::Result<()> {
//! let agent = Agent::new("agent.acl").await?;
//! let session = agent.session("/my-project", None)?;
//! let result = session.send("Explain the auth module", None).await?;
//! println!("{}", result.text);
//! # Ok(())
//! # }
//! ```

use crate::agent::{AgentConfig, AgentEvent, AgentResult};
use crate::commands::CommandRegistry;
use crate::config::CodeConfig;
use crate::error::Result;
use crate::hitl::PendingConfirmationInfo;
use crate::llm::{LlmClient, Message};
use crate::prompts::{PlanningMode, SystemPromptSlots};
use crate::queue::{
    ExternalTask, ExternalTaskResult, LaneHandlerConfig, SessionLane, SessionQueueConfig,
    SessionQueueStats,
};
use crate::tools::{ToolContext, ToolExecutor};
use a3s_lane::{DeadLetter, MetricsSnapshot};
use a3s_memory::MemoryStore;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
mod agent_binding;
mod agent_bootstrap;
mod agent_loop_runtime;
mod agent_sessions;
mod capabilities;
mod command_runtime;
mod conversation_runtime;
mod direct_tools;
mod hook_control;
mod run_lifecycle;
mod runtime;
mod runtime_events;
mod session_builder;
mod session_clock;
mod session_close;
mod session_commands;
mod session_config;
mod session_extensions;
mod session_hitl;
mod session_options;
mod session_persistence;
mod session_queue;
mod session_runs;
mod session_runtime;
mod session_save;
mod session_verification;
mod session_view;
use direct_tools::DirectToolRuntime;
use hook_control::HookControl;
use runtime_events::ActiveToolState;
use session_close::SessionCloseHandle;
use session_extensions::SessionExtensionRuntime;
use session_hitl::HitlControl;
use session_queue::QueueControl;
use session_runs::RunControl;
use session_verification::VerificationRuntime;
use session_view::SessionView;

/// Canonicalize a path, stripping the Windows `\\?\` UNC prefix to avoid
/// polluting workspace strings throughout the system (prompts, session data, etc.).
fn safe_canonicalize(path: &Path) -> PathBuf {
    match std::fs::canonicalize(path) {
        Ok(p) => strip_unc_prefix(p),
        Err(_) => path.to_path_buf(),
    }
}

/// Strip the Windows extended-length path prefix (`\\?\`) that `canonicalize()` adds.
/// On non-Windows this is a no-op.
fn strip_unc_prefix(path: PathBuf) -> PathBuf {
    #[cfg(windows)]
    {
        let s = path.to_string_lossy();
        if let Some(stripped) = s.strip_prefix(r"\\?\") {
            return PathBuf::from(stripped);
        }
    }
    path
}

// ============================================================================
// ToolCallResult
// ============================================================================

/// Result of a direct tool execution (no LLM).
#[derive(Debug, Clone)]
pub struct ToolCallResult {
    pub name: String,
    pub output: String,
    pub exit_code: i32,
    pub metadata: Option<serde_json::Value>,
    /// Structured discriminant for tool failures. `None` when the tool
    /// either succeeded or failed without a typed reason (the message in
    /// `output` is then the only diagnostic). Populated for known
    /// kinds such as `VersionConflict` so SDK callers can branch on the
    /// `type` field instead of regex-matching `output`.
    pub error_kind: Option<crate::tools::ToolErrorKind>,
}

// ============================================================================
// SessionOptions
// ============================================================================

/// Optional per-session overrides.
#[derive(Clone, Default)]
pub struct SessionOptions {
    /// Override the default model. Format: `"provider/model"` (e.g., `"openai/gpt-4o"`).
    pub model: Option<String>,
    /// Extra directories to scan for agent files.
    /// Merged with any global `agent_dirs` from [`CodeConfig`].
    pub agent_dirs: Vec<PathBuf>,
    /// Reproducible disposable workers registered for task delegation.
    /// Explicit session workers override agents loaded from directories by name.
    pub worker_agents: Vec<crate::subagent::WorkerAgentSpec>,
    /// Optional queue configuration for lane-based tool execution.
    ///
    /// When set, enables priority-based tool scheduling with parallel execution
    /// of read-only (Query-lane) tools, DLQ, metrics, and external task handling.
    pub queue_config: Option<SessionQueueConfig>,
    /// Optional security provider for taint tracking and output sanitization
    pub security_provider: Option<Arc<dyn crate::security::SecurityProvider>>,
    /// Optional host-supplied LLM client.
    ///
    /// When set, it is used directly, overriding the `provider/model`
    /// factory resolution — the one Action-layer backend that was previously
    /// only injectable in test code. Lets a host plug in a provider the
    /// built-in factory does not cover, a deterministic record/replay client,
    /// or an HTTP-layer proxy/audit wrapper. Mirrors `workspace_services`.
    pub llm_client: Option<Arc<dyn crate::llm::LlmClient>>,
    /// Optional context providers for RAG
    pub context_providers: Vec<Arc<dyn crate::context::ContextProvider>>,
    /// Optional confirmation manager for HITL
    pub confirmation_manager: Option<Arc<dyn crate::hitl::ConfirmationProvider>>,
    /// Optional confirmation policy (will be used to create ConfirmationManager if confirmation_manager is not set)
    pub confirmation_policy: Option<crate::hitl::ConfirmationPolicy>,
    /// Optional permission checker
    pub permission_checker: Option<Arc<dyn crate::permissions::PermissionChecker>>,
    /// Serializable permission policy used to build the checker, when available.
    pub permission_policy: Option<crate::permissions::PermissionPolicy>,
    /// Enable planning
    pub planning_mode: PlanningMode,
    /// Enable goal tracking
    pub goal_tracking: bool,
    /// Extra directories to scan for skill files (*.md).
    /// Merged with any global `skill_dirs` from [`CodeConfig`].
    pub skill_dirs: Vec<PathBuf>,
    /// Optional skill registry for instruction injection
    pub skill_registry: Option<Arc<crate::skills::SkillRegistry>>,
    /// Optional memory store for long-term memory persistence
    pub memory_store: Option<Arc<dyn MemoryStore>>,
    /// Deferred file memory directory — constructed async in `build_session()`
    pub(crate) file_memory_dir: Option<PathBuf>,
    /// Optional session store for persistence
    pub session_store: Option<Arc<dyn crate::store::SessionStore>>,
    /// Explicit session ID (auto-generated if not set)
    pub session_id: Option<String>,
    /// Multi-tenant identifier. Framework only transports this string;
    /// the host decides what "tenant" means and how to
    /// aggregate/bill on it. Emitted to hooks/traces, persisted in
    /// `SessionData`, never interpreted by core.
    pub tenant_id: Option<String>,
    /// Identity of the principal that triggered this session (user id,
    /// service account, etc). Treated as opaque.
    pub principal: Option<String>,
    /// Logical identifier of the agent template / definition the session
    /// was instantiated from. Lets the host aggregate sessions by
    /// "which agent recipe" independent of the concrete session id.
    pub agent_template_id: Option<String>,
    /// Distributed-trace correlation id. Propagated through hooks/traces
    /// so a session's events join with upstream/downstream work in the
    /// host's observability pipeline.
    pub correlation_id: Option<String>,
    /// Optional host-supplied budget / quota guard. The framework calls
    /// into it before each LLM call (and reports actuals after) so the
    /// host can refuse or rate-limit at the cluster level. Default is
    /// `None` (no enforcement — equivalent to
    /// [`NoopBudgetGuard`](crate::budget::NoopBudgetGuard)).
    pub budget_guard: Option<Arc<dyn crate::budget::BudgetGuard>>,
    /// Optional host-provided ID/Clock pair. Replaces the default
    /// random-UUID + wall-clock pair, enabling deterministic replay
    /// on another node. `None` keeps pre-P2 behaviour.
    pub host_env: Option<Arc<crate::host_env::HostEnv>>,
    /// Optional FIFO retention caps on the session's in-memory stores
    /// (run records, run events, trace events, terminal subagent
    /// tasks). `None` (default) keeps everything — fine for short
    /// sessions, a memory leak for hours-long cluster workloads.
    pub retention_limits: Option<crate::retention::SessionRetentionLimits>,
    /// Auto-save after each completed `send()` or default-history `stream()` call.
    pub auto_save: bool,
    /// Optional artifact retention limits for large tool/program outputs.
    pub artifact_store_limits: Option<crate::tools::ArtifactStoreLimits>,
    /// Max consecutive parse errors before aborting (overrides default of 2).
    /// `None` uses the `AgentConfig` default.
    pub max_parse_retries: Option<u32>,
    /// Per-tool execution timeout in milliseconds.
    /// `None` = no timeout (default).
    pub tool_timeout_ms: Option<u64>,
    /// Circuit-breaker threshold: max consecutive LLM API failures before
    /// aborting in non-streaming mode (overrides default of 3).
    /// `None` uses the `AgentConfig` default.
    pub circuit_breaker_threshold: Option<u32>,
    /// Optional concrete sandbox implementation.
    ///
    /// When set, `bash` tool commands are routed through this sandbox instead
    /// of `std::process::Command`. The host application constructs and owns
    /// the implementation (e.g., an A3S Box–backed handle).
    pub sandbox_handle: Option<Arc<dyn crate::sandbox::BashSandbox>>,
    /// Optional host-provided workspace backend.
    ///
    /// When set, built-in tools such as `read`, `write`, `ls`, and `bash`
    /// execute against these workspace capabilities instead of assuming the
    /// server-local filesystem. This is the primary extension point for DFS,
    /// browser, container, and remote workspace deployments.
    pub workspace_services: Option<Arc<crate::workspace::WorkspaceServices>>,
    /// Enable auto-compaction when context usage exceeds threshold.
    pub auto_compact: bool,
    /// Context usage percentage threshold for auto-compaction (0.0 - 1.0).
    /// Default: 0.80 (80%).
    pub auto_compact_threshold: Option<f32>,
    /// Inject a continuation message when the LLM stops without completing the task.
    /// `None` uses the `AgentConfig` default (true).
    pub continuation_enabled: Option<bool>,
    /// Maximum continuation injections per execution.
    /// `None` uses the `AgentConfig` default (3).
    pub max_continuation_turns: Option<u32>,
    /// Maximum execution time in milliseconds.
    /// `None` = no timeout (default).
    /// When set, the execution loop will abort if it exceeds this duration.
    pub max_execution_time_ms: Option<u64>,
    /// Optional MCP manager for connecting to external MCP servers.
    ///
    /// When set, all tools from connected MCP servers are registered and
    /// available during agent execution with names like `mcp__server__tool`.
    pub mcp_manager: Option<Arc<crate::mcp::manager::McpManager>>,
    /// Sampling temperature (0.0–1.0). Overrides the provider default.
    pub temperature: Option<f32>,
    /// Extended thinking budget in tokens (Anthropic only).
    pub thinking_budget: Option<usize>,
    /// Per-session tool round limit override.
    ///
    /// When set, overrides the agent-level `max_tool_rounds` for this session only.
    /// Maps directly from [`AgentDefinition::max_steps`] when creating sessions
    /// via [`Agent::session_for_agent`].
    pub max_tool_rounds: Option<usize>,
    /// Per-session parallel fan-out limit override.
    ///
    /// Applies to delegated `parallel_task`, plan wave execution, and safe
    /// parallel write batches.
    pub max_parallel_tasks: Option<usize>,
    /// Per-session automatic subagent delegation override.
    pub auto_delegation: Option<crate::config::AutoDelegationConfig>,
    /// Per-session kill switch for automatic parallel child-agent fan-out.
    ///
    /// This overlays the effective automatic delegation config instead of
    /// replacing it, so callers can disable auto fan-out without disabling
    /// automatic delegation itself.
    pub auto_parallel_delegation: Option<bool>,
    /// Slot-based system prompt customization.
    ///
    /// When set, overrides the agent-level prompt slots for this session.
    /// Users can customize role, guidelines, response style, and extra instructions
    /// without losing the core agentic capabilities.
    pub prompt_slots: Option<SystemPromptSlots>,
    /// Optional external hook executor (e.g. an AHP harness server).
    ///
    /// When set, **replaces** the built-in `HookEngine` for this session.
    /// All 11 lifecycle events are forwarded to the executor instead of being
    /// dispatched locally. The executor is also propagated to sub-agents via
    /// the sentinel hook mechanism.
    pub hook_executor: Option<Arc<dyn crate::hooks::HookExecutor>>,
}

// ============================================================================
// Agent
// ============================================================================

/// High-level agent facade.
///
/// Holds the LLM client and agent config. Workspace-independent.
/// Use [`Agent::session()`] to bind to a workspace.
pub struct Agent {
    code_config: CodeConfig,
    config: AgentConfig,
    /// Global MCP manager loaded from config.mcp_servers
    global_mcp: Option<Arc<crate::mcp::manager::McpManager>>,
    /// Pre-fetched MCP tool definitions from global_mcp (cached at creation time).
    /// Wrapped in Mutex so `refresh_mcp_tools()` can update the cache without `&mut self`.
    global_mcp_tools: std::sync::Mutex<Vec<(String, crate::mcp::McpTool)>>,
    /// Tracks every live session created by this agent via `Weak` refs so
    /// the agent can enumerate and forcibly close them. Sessions register
    /// themselves at construction and become dangling `Weak`s on drop —
    /// `list_sessions()` / `close_session()` prune dead entries on access.
    ///
    /// Uses a synchronous lock so the sync `Agent::session()` factory can
    /// insert without nesting tokio runtimes. The lock is only held for
    /// brief insert/scan operations — async close work happens after the
    /// lock is released.
    sessions: Arc<std::sync::Mutex<HashMap<String, std::sync::Weak<SessionCloseHandle>>>>,
    /// Set once `Agent::close()` has been called. Subsequent `session()` /
    /// `resume_session()` calls fail fast with `CodeError::SessionClosed`.
    closed: Arc<std::sync::atomic::AtomicBool>,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent").finish()
    }
}

impl Agent {
    /// Create from a config file path or inline ACL-compatible string.
    ///
    /// Auto-detects `.acl` file paths vs inline ACL-compatible config.
    pub async fn new(config_source: impl Into<String>) -> Result<Self> {
        let config = agent_bootstrap::load_code_config(config_source.into())?;
        Self::from_config(config).await
    }

    /// Create from a config file path or inline ACL-compatible string.
    ///
    /// Alias for [`Agent::new()`] — provides a consistent API with
    /// the Python and Node.js SDKs.
    pub async fn create(config_source: impl Into<String>) -> Result<Self> {
        Self::new(config_source).await
    }

    /// Create from a [`CodeConfig`] struct.
    pub async fn from_config(config: CodeConfig) -> Result<Self> {
        agent_bootstrap::build_agent_from_config(config).await
    }

    /// Re-fetch tool definitions from all connected global MCP servers and
    /// update the internal cache.
    ///
    /// Call this when an MCP server has added or removed tools since the
    /// agent was created. The refreshed tools will be visible to all
    /// **new** sessions created after this call; existing sessions are
    /// unaffected (their `ToolExecutor` snapshot is already built).
    pub async fn refresh_mcp_tools(&self) -> Result<()> {
        agent_sessions::refresh_mcp_tools(self).await
    }

    /// Bind to a workspace directory, returning an [`AgentSession`].
    ///
    /// Pass `None` for defaults, or `Some(SessionOptions)` to override
    /// the model, agent directories for this session.
    pub fn session(
        &self,
        workspace: impl Into<String>,
        options: Option<SessionOptions>,
    ) -> Result<AgentSession> {
        agent_sessions::create_session(self, workspace, options)
    }

    /// Create a session pre-configured from an [`AgentDefinition`].
    ///
    /// Maps the definition's `permissions`, `prompt`, `model`, and `max_steps`
    /// directly into [`SessionOptions`], so markdown/YAML-defined subagents can
    /// be used by delegation and advanced control-plane flows without manual wiring.
    ///
    /// The mapping follows the same logic as the built-in `task` tool:
    /// - `permissions` → `permission_checker`
    /// - `prompt`      → `prompt_slots.extra`
    /// - `max_steps`   → `max_tool_rounds`
    /// - `model`       → `model` (as `"provider/model"` string)
    ///
    /// `extra` can supply additional overrides (e.g. `planning_enabled`) that
    /// take precedence over the definition's values.
    pub fn session_for_agent(
        &self,
        workspace: impl Into<String>,
        def: &crate::subagent::AgentDefinition,
        extra: Option<SessionOptions>,
    ) -> Result<AgentSession> {
        agent_sessions::create_session_for_agent(self, workspace, def, extra)
    }

    /// Create a session from a reproducible disposable worker recipe.
    ///
    /// This is the cattle-mode companion to [`Agent::session_for_agent`]: callers
    /// provide a small [`WorkerAgentSpec`](crate::subagent::WorkerAgentSpec), and
    /// A3S Code compiles it into the same runtime definition used by delegated agents.
    pub fn session_for_worker(
        &self,
        workspace: impl Into<String>,
        spec: crate::subagent::WorkerAgentSpec,
        extra: Option<SessionOptions>,
    ) -> Result<AgentSession> {
        let def = spec.into_agent_definition();
        self.session_for_agent(workspace, &def, extra)
    }

    /// Resume a previously saved session by ID.
    ///
    /// Loads the session data from the store, rebuilds the `AgentSession` with
    /// the saved conversation history, and returns it ready for continued use.
    ///
    /// The `options` must include a `session_store` (or `with_file_session_store`)
    /// that contains the saved session.
    ///
    /// The resumed session uses the **workspace stored in the snapshot**, not a
    /// workspace from `options`. The store is therefore a trust boundary: its
    /// contents drive the resumed workspace and the persisted runtime policies.
    ///
    /// Runtime: this loads the snapshot via `block_in_place`, so it must be called
    /// on a multi-threaded Tokio runtime (it panics on a current-thread runtime).
    pub fn resume_session(
        &self,
        session_id: &str,
        options: SessionOptions,
    ) -> Result<AgentSession> {
        agent_sessions::resume_session(self, session_id, options)
    }

    /// Return the IDs of every live session created from this agent.
    ///
    /// "Live" means the caller still holds an [`AgentSession`] — sessions
    /// that have been dropped are pruned lazily on each call. The list is
    /// sorted to make output stable for tests/UIs.
    pub async fn list_sessions(&self) -> Vec<String> {
        agent_sessions::list_sessions(self).await
    }

    /// Close a specific live session by its session ID.
    ///
    /// Returns `true` when a live session with the given id was found and
    /// transitioned from open to closed by this call; `false` when no live
    /// session has that id, or when the session was already closed.
    ///
    /// This is the out-of-band counterpart to [`AgentSession::close`]: it
    /// performs exactly the same cleanup but can be invoked without holding
    /// a reference to the session itself — useful for control-plane code
    /// that only knows the session ID.
    pub async fn close_session(&self, session_id: &str) -> bool {
        agent_sessions::close_session(self, session_id).await
    }

    /// Close every live session created from this agent and tear down
    /// background resources owned by the agent (global MCP connections).
    ///
    /// After this call:
    /// - Every live `AgentSession` is closed (same effect as calling
    ///   [`AgentSession::close`] on each).
    /// - Subsequent [`Agent::session`] / [`Agent::resume_session`] calls
    ///   fail fast with [`CodeError::SessionClosed`](crate::error::CodeError::SessionClosed).
    ///
    /// Idempotent: subsequent calls are no-ops and are guaranteed not to
    /// panic.
    pub async fn close(&self) {
        agent_sessions::close_agent(self).await
    }

    /// Return whether [`close`](Self::close) has been called on this agent.
    pub fn is_closed(&self) -> bool {
        self.closed.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Disconnect every global MCP server whose last activity is older
    /// than `idle_threshold_ms`. Returns the names of disconnected
    /// servers (empty when there is no global MCP manager or when
    /// nothing is idle).
    ///
    /// Hosts running thousands of long-lived sessions should call this
    /// periodically (e.g. every 60s with a 5-min threshold) to release
    /// file descriptors and background workers from quiet MCP servers
    /// without losing the server's configuration. A subsequent tool
    /// call on the same server will require an explicit reconnect.
    pub async fn disconnect_idle_mcp(&self, idle_threshold_ms: u64) -> Vec<String> {
        match &self.global_mcp {
            Some(mcp) => mcp.disconnect_idle(idle_threshold_ms).await,
            None => Vec::new(),
        }
    }

    #[cfg(test)]
    fn build_session(
        &self,
        workspace: String,
        llm_client: Arc<dyn LlmClient>,
        opts: &SessionOptions,
    ) -> Result<AgentSession> {
        session_builder::build_agent_session(self, workspace, llm_client, opts)
    }
}

// ============================================================================
// AgentSession
// ============================================================================

/// Workspace-bound session. All LLM and tool operations happen here.
///
/// History is automatically accumulated after each `send()` call and after
/// `stream()` completes when no custom history is supplied.
/// Use `history()` to retrieve the current conversation log.
pub struct AgentSession {
    llm_client: Arc<dyn LlmClient>,
    tool_executor: Arc<ToolExecutor>,
    tool_context: ToolContext,
    config: AgentConfig,
    workspace: PathBuf,
    /// Unique session identifier.
    session_id: String,
    /// Internal conversation history, auto-updated after each `send()` and default-history `stream()`.
    history: Arc<RwLock<Vec<Message>>>,
    /// Optional lane queue for priority-based tool execution.
    command_queue: Option<Arc<crate::session_lane_queue::SessionLaneQueue>>,
    /// Optional long-term memory.
    memory: Option<Arc<crate::memory::AgentMemory>>,
    /// Optional session store for persistence.
    session_store: Option<Arc<dyn crate::store::SessionStore>>,
    /// Auto-save after each completed `send()` or default-history `stream()`.
    auto_save: bool,
    /// Hook engine for lifecycle event interception.
    hook_engine: Arc<crate::hooks::HookEngine>,
    /// Optional external hook executor (e.g. AHP harness). When set, replaces
    /// `hook_engine` as the executor passed to each `AgentLoop`.
    ahp_executor: Option<Arc<dyn crate::hooks::HookExecutor>>,
    /// Deferred init warning: emitted as PersistenceFailed on first send() if set.
    init_warning: Option<String>,
    /// Slash command registry for `/command` dispatch.
    /// Uses interior mutability so commands can be registered on a shared `Arc<AgentSession>`.
    command_registry: std::sync::Mutex<CommandRegistry>,
    /// Model identifier for display (e.g., "anthropic/claude-sonnet-4-20250514").
    model_name: String,
    /// Shared MCP manager — all add_mcp_server / remove_mcp_server calls go here.
    mcp_manager: Arc<crate::mcp::manager::McpManager>,
    /// Shared agent registry — populated at session creation; extended via register_agent_dir().
    agent_registry: Arc<crate::subagent::AgentRegistry>,
    /// Cancellation token for the current operation (send/stream).
    /// Stored so that cancel() can abort ongoing LLM calls.
    cancel_token: Arc<tokio::sync::Mutex<Option<tokio_util::sync::CancellationToken>>>,
    /// ID of the run currently attached to the active cancellation token.
    current_run_id: Arc<tokio::sync::Mutex<Option<String>>>,
    /// In-memory run snapshots and event replay buffer for this session.
    run_store: Arc<crate::run::InMemoryRunStore>,
    /// Materialized view of delegated subagent task lifecycle, populated from runtime events.
    subagent_tasks: Arc<crate::subagent_task_tracker::InMemorySubagentTaskTracker>,
    /// Currently executing tools observed from runtime events.
    active_tools: Arc<tokio::sync::RwLock<HashMap<String, ActiveToolState>>>,
    /// Compact execution traces for this session.
    trace_sink: crate::trace::InMemoryTraceSink,
    /// Structured completion evidence collected from agent and explicit verification runs.
    verification_reports: Arc<RwLock<Vec<crate::verification::VerificationReport>>>,
    /// Set once `close()` has been called. Subsequent send/stream calls
    /// fast-fail with [`crate::error::CodeError::SessionClosed`].
    closed: Arc<std::sync::atomic::AtomicBool>,
    /// Session-level parent cancellation token.
    ///
    /// Every in-flight run (blocking send, stream, delegated subagent task)
    /// derives its per-operation token from this one via `child_token()`,
    /// so `session_cancel.cancel()` cascades to all of them. `close()` fires
    /// this token first, after which any new `child_token()` returns an
    /// already-cancelled token (defending against close/spawn races).
    pub(crate) session_cancel: tokio_util::sync::CancellationToken,
    /// Shared `Arc`-handle used by both [`AgentSession::close`] and the
    /// parent [`Agent`]'s registry. The handle bundles every field needed
    /// to perform the close sequence so the two entry points cannot drift.
    close_handle: Arc<SessionCloseHandle>,
    /// Runtime-mutable override for the budget guard. When set, takes
    /// precedence over `config.budget_guard` on the next agent-loop
    /// build. Lets SDK callers (Node especially) install a host-side
    /// guard after `session()` has returned without ever putting a
    /// JS callable into `SessionOptions`.
    runtime_budget_guard: std::sync::Mutex<Option<Arc<dyn crate::budget::BudgetGuard>>>,
    /// Multi-tenant label. Framework only carries the string; semantics
    /// belong to the host.
    pub(crate) tenant_id: Option<String>,
    /// Principal that triggered the session (user / service / etc.).
    pub(crate) principal: Option<String>,
    /// Logical identifier of the agent template the session was
    /// instantiated from.
    pub(crate) agent_template_id: Option<String>,
    /// Distributed-trace correlation id propagated to hooks / traces.
    pub(crate) correlation_id: Option<String>,
}

impl std::fmt::Debug for AgentSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentSession")
            .field("session_id", &self.session_id)
            .field("workspace", &self.workspace.display().to_string())
            .field("auto_save", &self.auto_save)
            .finish()
    }
}

impl AgentSession {
    /// Get a snapshot of command entries (name, description, optional usage).
    ///
    /// Acquires the command registry lock briefly and returns owned data.
    pub fn command_registry(&self) -> std::sync::MutexGuard<'_, CommandRegistry> {
        session_commands::registry(self)
    }

    /// Register a custom slash command.
    ///
    /// Takes `&self` so it can be called on a shared `Arc<AgentSession>`.
    pub fn register_command(&self, cmd: Arc<dyn crate::commands::SlashCommand>) {
        session_commands::register(self, cmd);
    }

    /// Return whether [`close`](Self::close) has been called on this session.
    ///
    /// Once closed, `send`/`stream` and their attachment variants fast-fail
    /// with [`crate::error::CodeError::SessionClosed`] instead of starting a
    /// new run.
    pub fn is_closed(&self) -> bool {
        self.closed.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Clone the session-level [`CancellationToken`](tokio_util::sync::CancellationToken).
    ///
    /// All in-flight runs derive their per-operation token from this one via
    /// `child_token()`, so embedders can:
    ///
    /// - Observe the token (e.g. wire it into a host-side `select!`) to
    ///   react to session shutdown without polling [`is_closed`](Self::is_closed);
    /// - Call `.cancel()` on it to abort every operation in the session
    ///   without going through `close()` (no run-store / hook side effects).
    ///
    /// For graceful shutdown prefer [`close`](Self::close), which also marks
    /// runs as cancelled in the store and fires AHP hooks.
    pub fn session_cancel_token(&self) -> tokio_util::sync::CancellationToken {
        self.session_cancel.clone()
    }

    /// Return the host-defined tenant id, if any.
    ///
    /// The framework only transports this string — it never interprets
    /// or enforces tenant boundaries itself. Use this from custom
    /// `HookExecutor` / `PermissionChecker` / `BudgetGuard` impls to
    /// route logic by tenant.
    pub fn tenant_id(&self) -> Option<&str> {
        self.tenant_id.as_deref()
    }

    /// Return the principal that triggered the session, if any.
    pub fn principal(&self) -> Option<&str> {
        self.principal.as_deref()
    }

    /// Return the id of the agent template/definition the session was
    /// instantiated from, if any.
    pub fn agent_template_id(&self) -> Option<&str> {
        self.agent_template_id.as_deref()
    }

    /// Return the distributed-trace correlation id propagated through
    /// this session's events, if any.
    pub fn correlation_id(&self) -> Option<&str> {
        self.correlation_id.as_deref()
    }

    /// Install or replace a runtime budget guard. Takes effect on the
    /// next `send` / `stream` call (the guard is consulted at agent-
    /// loop build time, not on the live execution). Setting `None`
    /// clears the override so `config.budget_guard` takes over again.
    ///
    /// This is the entry point SDKs use to wire a host-supplied guard
    /// after the session has already been constructed — useful when
    /// the guard's transport (e.g. a JS callable) cannot live inside
    /// the value-typed `SessionOptions`.
    pub fn set_budget_guard(&self, guard: Option<Arc<dyn crate::budget::BudgetGuard>>) {
        let mut slot = self
            .runtime_budget_guard
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        *slot = guard;
    }

    /// Return the currently-installed runtime budget guard, if any.
    /// `None` means the loop falls back to `config.budget_guard`.
    pub fn budget_guard(&self) -> Option<Arc<dyn crate::budget::BudgetGuard>> {
        self.runtime_budget_guard
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Proactively close the session and release its in-flight work.
    ///
    /// On the first call this:
    /// 1. flips the session into the **closed** state so further `send`/`stream`
    ///    calls fast-fail with [`crate::error::CodeError::SessionClosed`];
    /// 2. fires the session-level cancellation token so every derived
    ///    run/subagent token cascades to cancelled;
    /// 3. marks the active run `Cancelled` in the run store and fires AHP
    ///    hook side effects;
    /// 4. cancels every still-running delegated subagent task spawned from
    ///    this session;
    /// 5. cancels all pending human-in-the-loop tool confirmations.
    ///
    /// Subsequent calls are no-ops and are guaranteed not to panic.
    pub async fn close(&self) {
        // Delegate to the shared handle so this entry point and
        // `Agent::close_session(id)` cannot drift in behaviour.
        self.close_handle.close().await;
    }

    /// Send a prompt and wait for the complete response.
    ///
    /// When `history` is `None`, uses (and auto-updates) the session's
    /// internal conversation history. When `Some`, uses the provided
    /// history instead (the internal history is **not** modified).
    ///
    /// If the prompt starts with `/`, it is dispatched as a slash command
    /// and the result is returned without calling the LLM.
    pub async fn send(&self, prompt: &str, history: Option<&[Message]>) -> Result<AgentResult> {
        conversation_runtime::send(self, prompt, history).await
    }

    /// Resume a previously-checkpointed run on this session.
    ///
    /// Loads the latest [`LoopCheckpoint`](crate::loop_checkpoint::LoopCheckpoint)
    /// stored under `checkpoint_run_id` and replays the agent loop from
    /// that boundary state. A **new** run id is allocated for the
    /// resumed work; the relationship between the old and new run is
    /// host-tracked — the framework does not interpret
    /// it.
    ///
    /// Returns an error when no `SessionStore` is configured on this
    /// session, or when no checkpoint exists for `checkpoint_run_id`.
    pub async fn resume_run(&self, checkpoint_run_id: &str) -> Result<AgentResult> {
        conversation_runtime::resume_run(self, checkpoint_run_id).await
    }

    /// Send a prompt with image attachments and wait for the complete response.
    ///
    /// Images are included as multi-modal content blocks in the user message.
    /// Requires a vision-capable model (e.g., Claude Sonnet, GPT-4o).
    pub async fn send_with_attachments(
        &self,
        prompt: &str,
        attachments: &[crate::llm::Attachment],
        history: Option<&[Message]>,
    ) -> Result<AgentResult> {
        conversation_runtime::send_with_attachments(self, prompt, attachments, history).await
    }

    /// Stream a prompt with image attachments.
    ///
    /// Images are included as multi-modal content blocks in the user message.
    /// Requires a vision-capable model (e.g., Claude Sonnet, GPT-4o).
    pub async fn stream_with_attachments(
        &self,
        prompt: &str,
        attachments: &[crate::llm::Attachment],
        history: Option<&[Message]>,
    ) -> Result<(mpsc::Receiver<AgentEvent>, JoinHandle<()>)> {
        conversation_runtime::stream_with_attachments(self, prompt, attachments, history).await
    }

    /// Send a prompt and stream events back.
    ///
    /// When `history` is `None`, uses the session's internal history
    /// and updates it when the stream completes.
    /// When `Some`, uses the provided history instead.
    ///
    /// If the prompt starts with `/`, it is dispatched as a slash command
    /// and the result is emitted as a single `TextDelta` + `End` event.
    pub async fn stream(
        &self,
        prompt: &str,
        history: Option<&[Message]>,
    ) -> Result<(mpsc::Receiver<AgentEvent>, JoinHandle<()>)> {
        conversation_runtime::stream(self, prompt, history).await
    }

    /// Cancel the current ongoing operation (send/stream).
    ///
    /// If an operation is in progress, this will trigger cancellation of the LLM streaming
    /// and tool execution. The operation will terminate as soon as possible.
    ///
    /// Returns `true` if an operation was cancelled, `false` if no operation was in progress.
    pub async fn cancel(&self) -> bool {
        RunControl::from_session(self).cancel_current().await
    }

    /// Cancel a specific run only if it is still the active run.
    ///
    /// This is useful for SDK callers that hold a previously observed run ID:
    /// stale run IDs will not cancel a newer operation.
    pub async fn cancel_run(&self, run_id: &str) -> bool {
        RunControl::from_session(self).cancel_run(run_id).await
    }

    /// Return snapshots for runs recorded by this session.
    pub async fn runs(&self) -> Vec<crate::run::RunSnapshot> {
        RunControl::from_session(self).runs().await
    }

    /// Return a snapshot for a recorded run.
    pub async fn run_snapshot(&self, run_id: &str) -> Option<crate::run::RunSnapshot> {
        RunControl::from_session(self).run_snapshot(run_id).await
    }

    /// Return recorded runtime events for a run.
    pub async fn run_events(&self, run_id: &str) -> Vec<crate::run::RunEventRecord> {
        RunControl::from_session(self).run_events(run_id).await
    }

    /// Return a handle for the currently running operation, if any.
    pub async fn current_run(&self) -> Option<crate::run::RunHandle> {
        RunControl::from_session(self).current_run().await
    }

    /// Return active tool calls observed for the currently running operation.
    pub async fn active_tools(&self) -> Vec<crate::run::ActiveToolSnapshot> {
        SessionView::from_session(self).active_tools().await
    }

    /// Look up a delegated subagent task by id. Returns `None` if no such task
    /// has been observed in this session.
    pub async fn subagent_task(
        &self,
        task_id: &str,
    ) -> Option<crate::subagent_task_tracker::SubagentTaskSnapshot> {
        self.subagent_tasks.get(task_id).await
    }

    /// Return snapshots of every delegated subagent task observed in this
    /// session (including completed and failed ones), oldest first.
    pub async fn subagent_tasks(&self) -> Vec<crate::subagent_task_tracker::SubagentTaskSnapshot> {
        self.subagent_tasks.list_for_parent(&self.session_id).await
    }

    /// Return snapshots of subagent tasks still in `Running` state.
    pub async fn pending_subagent_tasks(
        &self,
    ) -> Vec<crate::subagent_task_tracker::SubagentTaskSnapshot> {
        use crate::subagent_task_tracker::SubagentStatus;
        self.subagent_tasks
            .list_for_parent(&self.session_id)
            .await
            .into_iter()
            .filter(|task| task.status == SubagentStatus::Running)
            .collect()
    }

    /// Cancel an in-flight delegated subagent task by id. Returns `true`
    /// when a cancellation token was found and fired, `false` when the
    /// task id is unknown or the task has already finished. The eventual
    /// `SubagentEnd` from the cancelled child loop won't downgrade the
    /// terminal status — it stays `Cancelled`.
    pub async fn cancel_subagent_task(&self, task_id: &str) -> bool {
        self.subagent_tasks.cancel(task_id).await
    }

    /// Return a shared handle to the session's subagent task tracker.
    ///
    /// Advanced: embedders implementing a custom subagent execution path
    /// (i.e. spawning child loops outside the built-in `task` tool) can use
    /// this to register cancellation tokens and feed `AgentEvent`s into the
    /// tracker so the standard
    /// [`subagent_task`](Self::subagent_task) / [`pending_subagent_tasks`](Self::pending_subagent_tasks) /
    /// [`cancel_subagent_task`](Self::cancel_subagent_task) APIs and
    /// [`close`](Self::close) keep working uniformly across execution paths.
    pub fn subagent_tracker(
        &self,
    ) -> Arc<crate::subagent_task_tracker::InMemorySubagentTaskTracker> {
        Arc::clone(&self.subagent_tasks)
    }

    /// Return a snapshot of the session's conversation history.
    pub fn history(&self) -> Vec<Message> {
        SessionView::from_session(self).history()
    }

    /// Return pending HITL tool confirmations for this session.
    pub async fn pending_confirmations(&self) -> Vec<PendingConfirmationInfo> {
        HitlControl::from_session(self)
            .pending_confirmations()
            .await
    }

    /// Resolve a pending HITL tool confirmation.
    ///
    /// Returns `Ok(true)` when a pending confirmation was found and completed,
    /// `Ok(false)` when the tool ID is not pending or HITL is not configured.
    pub async fn confirm_tool_use(
        &self,
        tool_id: &str,
        approved: bool,
        reason: Option<String>,
    ) -> Result<bool> {
        HitlControl::from_session(self)
            .confirm_tool_use(tool_id, approved, reason)
            .await
    }

    /// Cancel all pending HITL confirmations for this session.
    pub async fn cancel_confirmations(&self) -> usize {
        HitlControl::from_session(self).cancel_confirmations().await
    }

    /// Return a reference to the session's memory, if configured.
    pub fn memory(&self) -> Option<&Arc<crate::memory::AgentMemory>> {
        SessionView::from_session(self).memory()
    }

    /// Return the session ID.
    pub fn id(&self) -> &str {
        SessionView::from_session(self).id()
    }

    /// Return the session workspace path.
    pub fn workspace(&self) -> &std::path::Path {
        SessionView::from_session(self).workspace()
    }

    /// Return any deferred init warning (e.g. memory store failed to initialize).
    pub fn init_warning(&self) -> Option<&str> {
        SessionView::from_session(self).init_warning()
    }

    /// Return the session ID.
    pub fn session_id(&self) -> &str {
        SessionView::from_session(self).id()
    }

    /// An [`AgentExecutor`](crate::orchestration::AgentExecutor) backed by this
    /// session — runs each orchestrated step as a child agent on this node,
    /// inheriting the session's agent registry, LLM client, workspace, MCP
    /// tools, and subagent tracker.
    ///
    /// This is what the orchestration combinators
    /// ([`execute_steps_parallel`](crate::orchestration::execute_steps_parallel),
    /// [`execute_pipeline`](crate::orchestration::execute_pipeline),
    /// [`execute_steps_parallel_resumable`](crate::orchestration::execute_steps_parallel_resumable))
    /// run against; a host can instead supply its own executor to place steps
    /// across a cluster.
    pub fn agent_executor(&self) -> Arc<dyn crate::orchestration::AgentExecutor> {
        Arc::new(self.build_task_executor(self.parent_run_context()))
    }

    /// Build the in-box [`TaskExecutor`](crate::tools::TaskExecutor) for this
    /// session, applying `parent` as the child-run capability context. Shared by
    /// [`agent_executor`](Self::agent_executor) and [`workflow`](Self::workflow)
    /// so both wire children identically.
    fn build_task_executor(
        &self,
        parent: crate::child_run::ChildRunContext,
    ) -> crate::tools::TaskExecutor {
        crate::tools::TaskExecutor::with_mcp(
            Arc::clone(&self.agent_registry),
            Arc::clone(&self.llm_client),
            self.workspace.display().to_string(),
            Arc::clone(&self.mcp_manager),
        )
        .with_parent_context(parent)
        .with_subagent_tracker(Arc::clone(&self.subagent_tasks))
        .with_max_parallel_tasks(self.config.max_parallel_tasks)
    }

    /// A programmable [`Workflow`](crate::orchestration::Workflow) bound to this
    /// session.
    ///
    /// Pre-wired with this session's executor (inheriting the same governance as
    /// model-driven delegation), persistence store (so each
    /// [`phase`](crate::orchestration::Workflow::phase) is a resume boundary),
    /// per-step event stream, and a session-derived stable root id. Control flow
    /// is ordinary Rust: `await` a verb, inspect the outcomes, decide what runs
    /// next.
    pub fn workflow(&self) -> crate::orchestration::Workflow {
        self.workflow_with_token_budget(None)
    }

    /// Like [`workflow`](Self::workflow) but with a hard token ceiling shared
    /// across every step. The cap is a best-effort *soft* cost ceiling — under a
    /// wide fan-out a few in-flight turns can race past it before the shared
    /// ledger catches up (see [`WorkflowBudget`](crate::orchestration::WorkflowBudget)).
    pub fn workflow_with_token_budget(
        &self,
        limit_tokens: Option<u64>,
    ) -> crate::orchestration::Workflow {
        use crate::budget::BudgetGuard;

        // One shared ledger for the whole workflow, wrapping the session's own
        // budget guard (if any) so a host's per-tenant accounting keeps working.
        let mut budget = crate::orchestration::WorkflowBudget::new(limit_tokens);
        if let Some(inner) = self.config.budget_guard.clone() {
            budget = budget.with_inner(inner);
        }
        let budget = Arc::new(budget);

        // Install the shared ledger as the child runs' budget guard so every
        // step's per-turn LLM accounting feeds it.
        let mut parent = self.parent_run_context();
        parent.budget_guard = Some(Arc::clone(&budget) as Arc<dyn BudgetGuard>);
        let executor: Arc<dyn crate::orchestration::AgentExecutor> =
            Arc::new(self.build_task_executor(parent));

        let mut builder = crate::orchestration::Workflow::builder(executor)
            .with_root_id(format!("wf-{}", self.session_id))
            .with_budget(Arc::clone(&budget));
        if let Some(store) = self.session_store.clone() {
            builder = builder.with_store(store);
        }
        if let Some(step_events) = self.tool_context.agent_event_tx.clone() {
            builder = builder.with_step_events(step_events);
        }
        builder.build()
    }

    /// Build the [`ChildRunContext`](crate::child_run::ChildRunContext) that
    /// orchestrated / delegated child runs inherit from this session.
    ///
    /// Mirrors the context the model-driven `task` / `parallel_task` path
    /// installs (see `register_task_capability` in `agent_api/capabilities.rs`)
    /// so a step run through [`agent_executor`](Self::agent_executor) carries the
    /// SAME governance — security provider, skill restrictions, confirmation,
    /// the shared workspace, and the safety limits — instead of weaker, ambient
    /// authority. Sourced from the session's resolved config; `hook_engine`
    /// stays `None` to match the model-driven path.
    pub(crate) fn parent_run_context(&self) -> crate::child_run::ChildRunContext {
        crate::child_run::ChildRunContext {
            security_provider: self.config.security_provider.clone(),
            hook_engine: None,
            skill_registry: self.config.skill_registry.clone(),
            tool_timeout_ms: self.config.tool_timeout_ms,
            max_parallel_tasks: Some(self.config.max_parallel_tasks),
            max_execution_time_ms: self.config.max_execution_time_ms,
            circuit_breaker_threshold: Some(self.config.circuit_breaker_threshold),
            confirmation_manager: self.config.confirmation_manager.clone(),
            workspace_services: Some(Arc::clone(&self.tool_context.workspace_services)),
            budget_guard: self.config.budget_guard.clone(),
        }
    }

    /// The session's persistence store, if one is configured — needed by the
    /// resumable orchestration combinator to journal workflow progress.
    pub fn session_store(&self) -> Option<Arc<dyn crate::store::SessionStore>> {
        self.session_store.clone()
    }

    /// Return the definitions of all tools currently registered in this session.
    ///
    /// The list reflects the live state of the tool executor — tools added via
    /// `add_mcp_server()` appear immediately; tools removed via
    /// `remove_mcp_server()` disappear immediately.
    pub fn tool_definitions(&self) -> Vec<crate::llm::ToolDefinition> {
        DirectToolRuntime::from_session(self).definitions()
    }

    /// Return the names of all tools currently registered on this session.
    ///
    /// Equivalent to `tool_definitions().into_iter().map(|t| t.name).collect()`.
    /// Tools added via [`add_mcp_server`] appear immediately; tools removed via
    /// [`remove_mcp_server`] disappear immediately.
    pub fn tool_names(&self) -> Vec<String> {
        DirectToolRuntime::from_session(self).names()
    }

    /// Return a stored tool artifact by URI, if it exists in this session.
    pub fn get_artifact(&self, artifact_uri: &str) -> Option<crate::tools::ToolArtifact> {
        DirectToolRuntime::from_session(self).artifact(artifact_uri)
    }

    /// Return compact execution trace events recorded for this session.
    pub fn trace_events(&self) -> Vec<crate::trace::TraceEvent> {
        SessionView::from_session(self).trace_events()
    }

    /// Return structured verification reports recorded for this session.
    pub fn verification_reports(&self) -> Vec<crate::verification::VerificationReport> {
        VerificationRuntime::from_session(self).reports()
    }

    /// Return a structured summary of all verification reports recorded for this session.
    pub fn verification_summary(&self) -> crate::verification::VerificationSummary {
        VerificationRuntime::from_session(self).summary()
    }

    /// Return a concise human-readable verification summary for this session.
    pub fn verification_summary_text(&self) -> String {
        VerificationRuntime::from_session(self).summary_text()
    }

    /// Add externally produced verification reports to this session's completion evidence.
    pub fn record_verification_reports(
        &self,
        reports: impl IntoIterator<Item = crate::verification::VerificationReport>,
    ) {
        VerificationRuntime::from_session(self).record(reports);
    }

    // ========================================================================
    // Hook API
    // ========================================================================

    /// Register a hook for lifecycle event interception.
    pub fn register_hook(&self, hook: crate::hooks::Hook) {
        HookControl::from_session(self).register_hook(hook);
    }

    /// Unregister a hook by ID.
    pub fn unregister_hook(&self, hook_id: &str) -> Option<crate::hooks::Hook> {
        HookControl::from_session(self).unregister_hook(hook_id)
    }

    /// Register a handler for a specific hook.
    pub fn register_hook_handler(
        &self,
        hook_id: &str,
        handler: Arc<dyn crate::hooks::HookHandler>,
    ) {
        HookControl::from_session(self).register_hook_handler(hook_id, handler);
    }

    /// Unregister a hook handler by hook ID.
    pub fn unregister_hook_handler(&self, hook_id: &str) {
        HookControl::from_session(self).unregister_hook_handler(hook_id);
    }

    /// Get the number of registered hooks.
    pub fn hook_count(&self) -> usize {
        HookControl::from_session(self).hook_count()
    }

    /// Save the session to the configured store.
    ///
    /// Returns `Ok(())` if saved successfully, or if no store is configured (no-op).
    pub async fn save(&self) -> Result<()> {
        session_save::save(self).await
    }

    /// Read a file from the workspace.
    pub async fn read_file(&self, path: &str) -> Result<String> {
        DirectToolRuntime::from_session(self).read_file(path).await
    }

    /// Write a file in the workspace.
    pub async fn write_file(&self, path: &str, content: &str) -> Result<ToolCallResult> {
        DirectToolRuntime::from_session(self)
            .write_file(path, content)
            .await
    }

    /// List a directory in the workspace.
    pub async fn ls(&self, path: Option<&str>) -> Result<ToolCallResult> {
        DirectToolRuntime::from_session(self).ls(path).await
    }

    /// Edit a file by replacing text in the workspace.
    pub async fn edit_file(
        &self,
        path: &str,
        old_string: &str,
        new_string: &str,
        replace_all: bool,
    ) -> Result<ToolCallResult> {
        DirectToolRuntime::from_session(self)
            .edit_file(path, old_string, new_string, replace_all)
            .await
    }

    /// Apply a unified diff patch to a workspace file.
    pub async fn patch_file(&self, path: &str, diff: &str) -> Result<ToolCallResult> {
        DirectToolRuntime::from_session(self)
            .patch_file(path, diff)
            .await
    }

    /// Execute a bash command in the workspace.
    ///
    /// When a sandbox handle is configured via
    /// [`SessionOptions::with_sandbox_handle()`], the command is routed through
    /// that sandbox.
    pub async fn bash(&self, command: &str) -> Result<String> {
        DirectToolRuntime::from_session(self).bash(command).await
    }

    /// Run verification commands through the session's tool execution path.
    pub async fn verify_commands(
        &self,
        subject: &str,
        commands: &[crate::verification::VerificationCommand],
    ) -> Result<crate::verification::VerificationReport> {
        VerificationRuntime::from_session(self)
            .verify_commands(subject, commands)
            .await
    }

    /// Return project-aware verification command presets for this workspace.
    pub fn verification_presets(&self) -> Vec<crate::verification::VerificationPreset> {
        VerificationRuntime::from_session(self).presets()
    }

    /// Search for files matching a glob pattern.
    pub async fn glob(&self, pattern: &str) -> Result<Vec<String>> {
        DirectToolRuntime::from_session(self).glob(pattern).await
    }

    /// Search file contents with a regex pattern.
    pub async fn grep(&self, pattern: &str) -> Result<String> {
        DirectToolRuntime::from_session(self).grep(pattern).await
    }

    /// Execute a tool by name, bypassing the LLM.
    pub async fn tool(&self, name: &str, args: serde_json::Value) -> Result<ToolCallResult> {
        DirectToolRuntime::from_session(self).call(name, args).await
    }

    // ========================================================================
    // Advanced optional Queue API
    // ========================================================================

    /// Returns whether this session has an advanced lane queue configured.
    pub fn has_queue(&self) -> bool {
        QueueControl::from_session(self).has_queue()
    }

    /// Configure a lane's handler mode for explicit external/hybrid dispatch.
    ///
    /// Only effective when a queue is configured via `SessionOptions::with_queue_config`.
    pub async fn set_lane_handler(&self, lane: SessionLane, config: LaneHandlerConfig) {
        QueueControl::from_session(self)
            .set_lane_handler(lane, config)
            .await;
    }

    /// Complete an external queue task by ID.
    ///
    /// Returns `true` if the task was found and completed, `false` if not found.
    pub async fn complete_external_task(&self, task_id: &str, result: ExternalTaskResult) -> bool {
        QueueControl::from_session(self)
            .complete_external_task(task_id, result)
            .await
    }

    /// Get pending external queue tasks awaiting completion by an external handler.
    pub async fn pending_external_tasks(&self) -> Vec<ExternalTask> {
        QueueControl::from_session(self)
            .pending_external_tasks()
            .await
    }

    /// Get optional queue statistics (pending, active, external counts per lane).
    pub async fn queue_stats(&self) -> SessionQueueStats {
        QueueControl::from_session(self).stats().await
    }

    /// Get a metrics snapshot from the optional queue (if metrics are enabled).
    pub async fn queue_metrics(&self) -> Option<MetricsSnapshot> {
        QueueControl::from_session(self).metrics().await
    }

    /// Get dead letters from the optional queue's DLQ (if DLQ is enabled).
    pub async fn dead_letters(&self) -> Vec<DeadLetter> {
        QueueControl::from_session(self).dead_letters().await
    }

    // ========================================================================
    // MCP API
    // ========================================================================

    /// Register all agents found in a directory with the live session.
    ///
    /// Scans `dir` for `*.yaml`, `*.yml`, and `*.md` agent definition files,
    /// parses them, and adds each one to the shared `AgentRegistry` used by the
    /// `task` tool.  New agents are immediately usable via `task(agent="…")` in
    /// the same session — no restart required.
    ///
    /// Returns the number of agents successfully loaded from the directory.
    pub fn register_agent_dir(&self, dir: &std::path::Path) -> usize {
        SessionExtensionRuntime::from_session(self).register_agent_dir(dir)
    }

    /// Register a disposable worker agent with the live session.
    ///
    /// The returned definition is immediately available to the `task` tool by
    /// worker name, so callers can create many reproducible workers without
    /// writing temporary agent files or restarting the session.
    pub fn register_worker_agent(
        &self,
        spec: crate::subagent::WorkerAgentSpec,
    ) -> crate::subagent::AgentDefinition {
        SessionExtensionRuntime::from_session(self).register_worker_agent(spec)
    }

    /// Register multiple disposable worker agents with the live session.
    pub fn register_worker_agents<I>(&self, specs: I) -> Vec<crate::subagent::AgentDefinition>
    where
        I: IntoIterator<Item = crate::subagent::WorkerAgentSpec>,
    {
        SessionExtensionRuntime::from_session(self).register_worker_agents(specs)
    }

    /// Add an MCP server to this session.
    ///
    /// Registers, connects, and makes all tools immediately available for the
    /// agent to call. Tool names follow the convention `mcp__<name>__<tool>`.
    ///
    /// Returns the number of tools registered from the server.
    pub async fn add_mcp_server(
        &self,
        config: crate::mcp::McpServerConfig,
    ) -> crate::error::Result<usize> {
        SessionExtensionRuntime::from_session(self)
            .add_mcp_server(config)
            .await
    }

    /// The session's tool executor, for installing agent-dir `tools/` entries
    /// (e.g. a `kind = "script"` tool) into the live registry. Internal seam used
    /// by [`serve::install_agent_dir_tools`](crate::serve::install_agent_dir_tools)
    /// (the only caller, hence the `serve` gate).
    #[cfg(feature = "serve")]
    pub(crate) fn tool_executor(&self) -> &Arc<crate::tools::ToolExecutor> {
        &self.tool_executor
    }

    /// Remove an MCP server from this session.
    ///
    /// Disconnects the server and unregisters all its tools from the executor.
    /// No-op if the server was never added.
    pub async fn remove_mcp_server(&self, server_name: &str) -> crate::error::Result<()> {
        SessionExtensionRuntime::from_session(self)
            .remove_mcp_server(server_name)
            .await
    }

    /// Return the connection status of all MCP servers registered with this session.
    pub async fn mcp_status(
        &self,
    ) -> std::collections::HashMap<String, crate::mcp::McpServerStatus> {
        SessionExtensionRuntime::from_session(self)
            .mcp_status()
            .await
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
