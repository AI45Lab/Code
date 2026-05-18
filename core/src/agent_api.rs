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
    pub fn resume_session(
        &self,
        session_id: &str,
        options: SessionOptions,
    ) -> Result<AgentSession> {
        agent_sessions::resume_session(self, session_id, options)
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
    /// Currently executing tools observed from runtime events.
    active_tools: Arc<tokio::sync::RwLock<HashMap<String, ActiveToolState>>>,
    /// Compact execution traces for this session.
    trace_sink: crate::trace::InMemoryTraceSink,
    /// Structured completion evidence collected from agent and explicit verification runs.
    verification_reports: Arc<RwLock<Vec<crate::verification::VerificationReport>>>,
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

    /// Cancel any active operation and release session resources.
    pub async fn close(&self) {
        let _ = self.cancel().await;
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
