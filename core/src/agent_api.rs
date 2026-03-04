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
//! let agent = Agent::new("agent.hcl").await?;
//! let session = agent.session("/my-project", None)?;
//! let result = session.send("Explain the auth module", None).await?;
//! println!("{}", result.text);
//! # Ok(())
//! # }
//! ```

use crate::agent::{AgentConfig, AgentEvent, AgentLoop, AgentResult};
use crate::commands::{CommandContext, CommandRegistry};
use crate::config::CodeConfig;
use crate::error::{read_or_recover, write_or_recover, Result};
use crate::llm::{LlmClient, Message};
use crate::prompts::SystemPromptSlots;
use crate::queue::{
    ExternalTask, ExternalTaskResult, LaneHandlerConfig, SessionLane, SessionQueueConfig,
    SessionQueueStats,
};
use crate::session_lane_queue::SessionLaneQueue;
use crate::tools::{ToolContext, ToolExecutor};
use a3s_lane::{DeadLetter, MetricsSnapshot};
use a3s_memory::{FileMemoryStore, MemoryStore};
use anyhow::Context;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;

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
    /// Optional permission checker
    pub permission_checker: Option<Arc<dyn crate::permissions::PermissionChecker>>,
    /// Enable planning
    pub planning_enabled: bool,
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
    /// Auto-save after each `send()` call
    pub auto_save: bool,
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
    /// Optional sandbox configuration.
    ///
    /// When set, `bash` tool commands are routed through an A3S Box MicroVM
    /// sandbox instead of `std::process::Command`. Requires the `sandbox`
    /// Cargo feature to be enabled.
    pub sandbox_config: Option<crate::sandbox::SandboxConfig>,
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
    /// Optional MCP manager for connecting to external MCP servers.
    ///
    /// When set, all tools from connected MCP servers are registered and
    /// available during agent execution with names like `mcp__server__tool`.
    pub mcp_manager: Option<Arc<crate::mcp::manager::McpManager>>,
    /// Sampling temperature (0.0–1.0). Overrides the provider default.
    pub temperature: Option<f32>,
    /// Extended thinking budget in tokens (Anthropic only).
    pub thinking_budget: Option<usize>,
    /// Slot-based system prompt customization.
    ///
    /// When set, overrides the agent-level prompt slots for this session.
    /// Users can customize role, guidelines, response style, and extra instructions
    /// without losing the core agentic capabilities.
    pub prompt_slots: Option<SystemPromptSlots>,
}

impl std::fmt::Debug for SessionOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionOptions")
            .field("model", &self.model)
            .field("agent_dirs", &self.agent_dirs)
            .field("skill_dirs", &self.skill_dirs)
            .field("queue_config", &self.queue_config)
            .field("security_provider", &self.security_provider.is_some())
            .field("context_providers", &self.context_providers.len())
            .field("confirmation_manager", &self.confirmation_manager.is_some())
            .field("permission_checker", &self.permission_checker.is_some())
            .field("planning_enabled", &self.planning_enabled)
            .field("goal_tracking", &self.goal_tracking)
            .field(
                "skill_registry",
                &self
                    .skill_registry
                    .as_ref()
                    .map(|r| format!("{} skills", r.len())),
            )
            .field("memory_store", &self.memory_store.is_some())
            .field("session_store", &self.session_store.is_some())
            .field("session_id", &self.session_id)
            .field("auto_save", &self.auto_save)
            .field("max_parse_retries", &self.max_parse_retries)
            .field("tool_timeout_ms", &self.tool_timeout_ms)
            .field("circuit_breaker_threshold", &self.circuit_breaker_threshold)
            .field("sandbox_config", &self.sandbox_config)
            .field("auto_compact", &self.auto_compact)
            .field("auto_compact_threshold", &self.auto_compact_threshold)
            .field("continuation_enabled", &self.continuation_enabled)
            .field("max_continuation_turns", &self.max_continuation_turns)
            .field("mcp_manager", &self.mcp_manager.is_some())
            .field("temperature", &self.temperature)
            .field("thinking_budget", &self.thinking_budget)
            .field("prompt_slots", &self.prompt_slots.is_some())
            .finish()
    }
}

impl SessionOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_agent_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.agent_dirs.push(dir.into());
        self
    }

    pub fn with_queue_config(mut self, config: SessionQueueConfig) -> Self {
        self.queue_config = Some(config);
        self
    }

    /// Enable default security provider with taint tracking and output sanitization
    pub fn with_default_security(mut self) -> Self {
        self.security_provider = Some(Arc::new(crate::security::DefaultSecurityProvider::new()));
        self
    }

    /// Set a custom security provider
    pub fn with_security_provider(
        mut self,
        provider: Arc<dyn crate::security::SecurityProvider>,
    ) -> Self {
        self.security_provider = Some(provider);
        self
    }

    /// Add a file system context provider for simple RAG
    pub fn with_fs_context(mut self, root_path: impl Into<PathBuf>) -> Self {
        let config = crate::context::FileSystemContextConfig::new(root_path);
        self.context_providers
            .push(Arc::new(crate::context::FileSystemContextProvider::new(
                config,
            )));
        self
    }

    /// Add a custom context provider
    pub fn with_context_provider(
        mut self,
        provider: Arc<dyn crate::context::ContextProvider>,
    ) -> Self {
        self.context_providers.push(provider);
        self
    }

    /// Set a confirmation manager for HITL
    pub fn with_confirmation_manager(
        mut self,
        manager: Arc<dyn crate::hitl::ConfirmationProvider>,
    ) -> Self {
        self.confirmation_manager = Some(manager);
        self
    }

    /// Set a permission checker
    pub fn with_permission_checker(
        mut self,
        checker: Arc<dyn crate::permissions::PermissionChecker>,
    ) -> Self {
        self.permission_checker = Some(checker);
        self
    }

    /// Allow all tool execution without confirmation (permissive mode).
    ///
    /// Use this for automated scripts, demos, and CI environments where
    /// human-in-the-loop confirmation is not needed. Without this (or a
    /// custom permission checker), the default is `Ask`, which requires a
    /// HITL confirmation manager to be configured.
    pub fn with_permissive_policy(self) -> Self {
        self.with_permission_checker(Arc::new(crate::permissions::PermissionPolicy::permissive()))
    }

    /// Enable planning
    pub fn with_planning(mut self, enabled: bool) -> Self {
        self.planning_enabled = enabled;
        self
    }

    /// Enable goal tracking
    pub fn with_goal_tracking(mut self, enabled: bool) -> Self {
        self.goal_tracking = enabled;
        self
    }

    /// Add a skill registry with built-in skills
    pub fn with_builtin_skills(mut self) -> Self {
        self.skill_registry = Some(Arc::new(crate::skills::SkillRegistry::with_builtins()));
        self
    }

    /// Add a custom skill registry
    pub fn with_skill_registry(mut self, registry: Arc<crate::skills::SkillRegistry>) -> Self {
        self.skill_registry = Some(registry);
        self
    }

    /// Add skill directories to scan for skill files (*.md).
    /// Merged with any global `skill_dirs` from [`CodeConfig`] at session build time.
    pub fn with_skill_dirs(mut self, dirs: impl IntoIterator<Item = impl Into<PathBuf>>) -> Self {
        self.skill_dirs.extend(dirs.into_iter().map(Into::into));
        self
    }

    /// Load skills from a directory (eager — scans immediately into a registry).
    pub fn with_skills_from_dir(mut self, dir: impl AsRef<std::path::Path>) -> Self {
        let registry = self
            .skill_registry
            .unwrap_or_else(|| Arc::new(crate::skills::SkillRegistry::new()));
        if let Err(e) = registry.load_from_dir(&dir) {
            tracing::warn!(
                dir = %dir.as_ref().display(),
                error = %e,
                "Failed to load skills from directory — continuing without them"
            );
        }
        self.skill_registry = Some(registry);
        self
    }

    /// Set a custom memory store
    pub fn with_memory(mut self, store: Arc<dyn MemoryStore>) -> Self {
        self.memory_store = Some(store);
        self
    }

    /// Use a file-based memory store at the given directory.
    ///
    /// The store is created lazily when the session is built (requires async).
    /// This stores the directory path; `FileMemoryStore::new()` is called during
    /// session construction.
    pub fn with_file_memory(mut self, dir: impl Into<PathBuf>) -> Self {
        self.file_memory_dir = Some(dir.into());
        self
    }

    /// Set a session store for persistence
    pub fn with_session_store(mut self, store: Arc<dyn crate::store::SessionStore>) -> Self {
        self.session_store = Some(store);
        self
    }

    /// Use a file-based session store at the given directory
    pub fn with_file_session_store(mut self, dir: impl Into<PathBuf>) -> Self {
        let dir = dir.into();
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                match tokio::task::block_in_place(|| {
                    handle.block_on(crate::store::FileSessionStore::new(dir))
                }) {
                    Ok(store) => {
                        self.session_store =
                            Some(Arc::new(store) as Arc<dyn crate::store::SessionStore>);
                    }
                    Err(e) => {
                        tracing::warn!("Failed to create file session store: {}", e);
                    }
                }
            }
            Err(_) => {
                tracing::warn!(
                    "No async runtime available for file session store — persistence disabled"
                );
            }
        }
        self
    }

    /// Set an explicit session ID (auto-generated UUID if not set)
    pub fn with_session_id(mut self, id: impl Into<String>) -> Self {
        self.session_id = Some(id.into());
        self
    }

    /// Enable auto-save after each `send()` call
    pub fn with_auto_save(mut self, enabled: bool) -> Self {
        self.auto_save = enabled;
        self
    }

    /// Set the maximum number of consecutive malformed-tool-args errors before
    /// the agent loop bails.
    ///
    /// Default: 2 (the LLM gets two chances to self-correct before the session
    /// is aborted).
    pub fn with_parse_retries(mut self, max: u32) -> Self {
        self.max_parse_retries = Some(max);
        self
    }

    /// Set a per-tool execution timeout.
    ///
    /// When set, each tool execution is wrapped in `tokio::time::timeout`.
    /// A timeout produces an error message that is fed back to the LLM
    /// (the session continues).
    pub fn with_tool_timeout(mut self, timeout_ms: u64) -> Self {
        self.tool_timeout_ms = Some(timeout_ms);
        self
    }

    /// Set the circuit-breaker threshold.
    ///
    /// In non-streaming mode, the agent retries transient LLM API failures up
    /// to this many times (with exponential backoff) before aborting.
    /// Default: 3 attempts.
    pub fn with_circuit_breaker(mut self, threshold: u32) -> Self {
        self.circuit_breaker_threshold = Some(threshold);
        self
    }

    /// Enable all resilience defaults with sensible values:
    ///
    /// - `max_parse_retries = 2`
    /// - `tool_timeout_ms = 120_000` (2 minutes)
    /// - `circuit_breaker_threshold = 3`
    pub fn with_resilience_defaults(self) -> Self {
        self.with_parse_retries(2)
            .with_tool_timeout(120_000)
            .with_circuit_breaker(3)
    }

    /// Route `bash` tool execution through an A3S Box MicroVM sandbox.
    ///
    /// The workspace directory is mounted read-write at `/workspace` inside
    /// the sandbox. Requires the `sandbox` Cargo feature; without it a warning
    /// is logged and bash commands continue to run locally.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use a3s_code_core::{SessionOptions, SandboxConfig};
    ///
    /// SessionOptions::new().with_sandbox(SandboxConfig {
    ///     image: "ubuntu:22.04".into(),
    ///     memory_mb: 512,
    ///     network: false,
    ///     ..SandboxConfig::default()
    /// });
    /// ```
    pub fn with_sandbox(mut self, config: crate::sandbox::SandboxConfig) -> Self {
        self.sandbox_config = Some(config);
        self
    }

    /// Enable auto-compaction when context usage exceeds threshold.
    ///
    /// When enabled, the agent loop automatically prunes large tool outputs
    /// and summarizes old messages when context usage exceeds the threshold.
    pub fn with_auto_compact(mut self, enabled: bool) -> Self {
        self.auto_compact = enabled;
        self
    }

    /// Set the auto-compact threshold (0.0 - 1.0). Default: 0.80 (80%).
    pub fn with_auto_compact_threshold(mut self, threshold: f32) -> Self {
        self.auto_compact_threshold = Some(threshold.clamp(0.0, 1.0));
        self
    }

    /// Enable or disable continuation injection (default: enabled).
    ///
    /// When enabled, the loop injects a continuation message when the LLM stops
    /// calling tools before the task appears complete, nudging it to keep working.
    pub fn with_continuation(mut self, enabled: bool) -> Self {
        self.continuation_enabled = Some(enabled);
        self
    }

    /// Set the maximum number of continuation injections per execution (default: 3).
    pub fn with_max_continuation_turns(mut self, turns: u32) -> Self {
        self.max_continuation_turns = Some(turns);
        self
    }

    /// Set an MCP manager to connect to external MCP servers.
    ///
    /// All tools from connected servers will be available during execution
    /// with names like `mcp__<server>__<tool>`.
    pub fn with_mcp(mut self, manager: Arc<crate::mcp::manager::McpManager>) -> Self {
        self.mcp_manager = Some(manager);
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_thinking_budget(mut self, budget: usize) -> Self {
        self.thinking_budget = Some(budget);
        self
    }

    /// Set slot-based system prompt customization for this session.
    ///
    /// Allows customizing role, guidelines, response style, and extra instructions
    /// without overriding the core agentic capabilities.
    pub fn with_prompt_slots(mut self, slots: SystemPromptSlots) -> Self {
        self.prompt_slots = Some(slots);
        self
    }
}

// ============================================================================
// Agent
// ============================================================================

/// High-level agent facade.
///
/// Holds the LLM client and agent config. Workspace-independent.
/// Use [`Agent::session()`] to bind to a workspace.
pub struct Agent {
    llm_client: Arc<dyn LlmClient>,
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
    /// Create from a config file path or inline config string.
    ///
    /// Auto-detects: file path (.hcl/.json) vs inline JSON vs inline HCL.
    pub async fn new(config_source: impl Into<String>) -> Result<Self> {
        let source = config_source.into();

        // Expand leading `~/` to the user's home directory (cross-platform)
        let expanded = if let Some(rest) = source.strip_prefix("~/") {
            let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"));
            if let Some(home) = home {
                PathBuf::from(home).join(rest).display().to_string()
            } else {
                source.clone()
            }
        } else {
            source.clone()
        };

        let path = Path::new(&expanded);

        let config = if path.extension().is_some() && path.exists() {
            CodeConfig::from_file(path)
                .with_context(|| format!("Failed to load config: {}", path.display()))?
        } else {
            // Try to parse as HCL string
            CodeConfig::from_hcl(&source).context("Failed to parse config as HCL string")?
        };

        Self::from_config(config).await
    }

    /// Create from a config file path or inline config string.
    ///
    /// Alias for [`Agent::new()`] — provides a consistent API with
    /// the Python and Node.js SDKs.
    pub async fn create(config_source: impl Into<String>) -> Result<Self> {
        Self::new(config_source).await
    }

    /// Create from a [`CodeConfig`] struct.
    pub async fn from_config(config: CodeConfig) -> Result<Self> {
        let llm_config = config
            .default_llm_config()
            .context("default_model must be set in 'provider/model' format with a valid API key")?;
        let llm_client = crate::llm::create_client_with_config(llm_config);

        let agent_config = AgentConfig {
            max_tool_rounds: config
                .max_tool_rounds
                .unwrap_or(AgentConfig::default().max_tool_rounds),
            ..AgentConfig::default()
        };

        // Load global MCP servers from config
        let (global_mcp, global_mcp_tools) = if config.mcp_servers.is_empty() {
            (None, vec![])
        } else {
            let manager = Arc::new(crate::mcp::manager::McpManager::new());
            for server in &config.mcp_servers {
                if !server.enabled {
                    continue;
                }
                manager.register_server(server.clone()).await;
                if let Err(e) = manager.connect(&server.name).await {
                    tracing::warn!(
                        server = %server.name,
                        error = %e,
                        "Failed to connect to MCP server — skipping"
                    );
                }
            }
            // Pre-fetch tool definitions while we're in async context
            let tools = manager.get_all_tools().await;
            (Some(manager), tools)
        };

        let mut agent = Agent {
            llm_client,
            code_config: config,
            config: agent_config,
            global_mcp,
            global_mcp_tools: std::sync::Mutex::new(global_mcp_tools),
        };

        // Always initialize the skill registry with built-in skills, then load any user-defined dirs
        let registry = Arc::new(crate::skills::SkillRegistry::with_builtins());
        for dir in &agent.code_config.skill_dirs.clone() {
            if let Err(e) = registry.load_from_dir(dir) {
                tracing::warn!(
                    dir = %dir.display(),
                    error = %e,
                    "Failed to load skills from directory — skipping"
                );
            }
        }
        agent.config.skill_registry = Some(registry);

        Ok(agent)
    }

    /// Re-fetch tool definitions from all connected global MCP servers and
    /// update the internal cache.
    ///
    /// Call this when an MCP server has added or removed tools since the
    /// agent was created. The refreshed tools will be visible to all
    /// **new** sessions created after this call; existing sessions are
    /// unaffected (their `ToolExecutor` snapshot is already built).
    pub async fn refresh_mcp_tools(&self) -> Result<()> {
        if let Some(ref mcp) = self.global_mcp {
            let fresh = mcp.get_all_tools().await;
            *self
                .global_mcp_tools
                .lock()
                .expect("global_mcp_tools lock poisoned") = fresh;
        }
        Ok(())
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
        let opts = options.unwrap_or_default();

        let llm_client = if let Some(ref model) = opts.model {
            let (provider_name, model_id) = model
                .split_once('/')
                .context("model format must be 'provider/model' (e.g., 'openai/gpt-4o')")?;

            let mut llm_config = self
                .code_config
                .llm_config(provider_name, model_id)
                .with_context(|| {
                    format!("provider '{provider_name}' or model '{model_id}' not found in config")
                })?;

            if let Some(temp) = opts.temperature {
                llm_config = llm_config.with_temperature(temp);
            }
            if let Some(budget) = opts.thinking_budget {
                llm_config = llm_config.with_thinking_budget(budget);
            }

            crate::llm::create_client_with_config(llm_config)
        } else {
            if opts.temperature.is_some() || opts.thinking_budget.is_some() {
                tracing::warn!(
                    "temperature/thinking_budget set without model override — these will be ignored. \
                     Use with_model() to apply LLM parameter overrides."
                );
            }
            self.llm_client.clone()
        };

        // Merge global MCP manager with any session-level one from opts.
        // If both exist, session-level servers are added into the global manager.
        let merged_opts = match (&self.global_mcp, &opts.mcp_manager) {
            (Some(global), Some(session)) => {
                let global = Arc::clone(global);
                let session_mgr = Arc::clone(session);
                match tokio::runtime::Handle::try_current() {
                    Ok(handle) => {
                        let global_for_merge = Arc::clone(&global);
                        tokio::task::block_in_place(|| {
                            handle.block_on(async move {
                                for config in session_mgr.all_configs().await {
                                    let name = config.name.clone();
                                    global_for_merge.register_server(config).await;
                                    if let Err(e) = global_for_merge.connect(&name).await {
                                        tracing::warn!(
                                            server = %name,
                                            error = %e,
                                            "Failed to connect session-level MCP server — skipping"
                                        );
                                    }
                                }
                            })
                        });
                    }
                    Err(_) => {
                        tracing::warn!(
                            "No async runtime available to merge session-level MCP servers \
                             into global manager — session MCP servers will not be available"
                        );
                    }
                }
                SessionOptions {
                    mcp_manager: Some(Arc::clone(&global)),
                    ..opts
                }
            }
            (Some(global), None) => SessionOptions {
                mcp_manager: Some(Arc::clone(global)),
                ..opts
            },
            _ => opts,
        };

        self.build_session(workspace.into(), llm_client, &merged_opts)
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
        let store = options.session_store.as_ref().ok_or_else(|| {
            crate::error::CodeError::Session(
                "resume_session requires a session_store in SessionOptions".to_string(),
            )
        })?;

        // Load session data from store
        let data = match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| handle.block_on(store.load(session_id)))
                .map_err(|e| {
                crate::error::CodeError::Session(format!(
                    "Failed to load session {}: {}",
                    session_id, e
                ))
            })?,
            Err(_) => {
                return Err(crate::error::CodeError::Session(
                    "No async runtime available for session resume".to_string(),
                ))
            }
        };

        let data = data.ok_or_else(|| {
            crate::error::CodeError::Session(format!("Session not found: {}", session_id))
        })?;

        // Build session with the saved workspace
        let mut opts = options;
        opts.session_id = Some(data.id.clone());

        let llm_client = if let Some(ref model) = opts.model {
            let (provider_name, model_id) = model
                .split_once('/')
                .context("model format must be 'provider/model'")?;
            let llm_config = self
                .code_config
                .llm_config(provider_name, model_id)
                .with_context(|| {
                    format!("provider '{provider_name}' or model '{model_id}' not found")
                })?;
            crate::llm::create_client_with_config(llm_config)
        } else {
            self.llm_client.clone()
        };

        let session = self.build_session(data.config.workspace.clone(), llm_client, &opts)?;

        // Restore conversation history
        *write_or_recover(&session.history) = data.messages;

        Ok(session)
    }

    fn build_session(
        &self,
        workspace: String,
        llm_client: Arc<dyn LlmClient>,
        opts: &SessionOptions,
    ) -> Result<AgentSession> {
        let canonical = safe_canonicalize(Path::new(&workspace));

        let tool_executor = Arc::new(ToolExecutor::new(canonical.display().to_string()));

        // Register task delegation tools (task, parallel_task).
        // These require an LLM client to spawn isolated child agent loops.
        // When MCP manager is available, pass it through so child sessions inherit MCP tools.
        {
            use crate::subagent::{load_agents_from_dir, AgentRegistry};
            use crate::tools::register_task_with_mcp;
            let agent_registry = AgentRegistry::new();
            for dir in self
                .code_config
                .agent_dirs
                .iter()
                .chain(opts.agent_dirs.iter())
            {
                for agent in load_agents_from_dir(dir) {
                    agent_registry.register(agent);
                }
            }
            register_task_with_mcp(
                tool_executor.registry(),
                Arc::clone(&llm_client),
                Arc::new(agent_registry),
                canonical.display().to_string(),
                opts.mcp_manager.clone(),
            );
        }

        // Register MCP tools before taking tool definitions snapshot.
        // Use pre-cached tools from Agent creation (avoids async in sync SDK context).
        if let Some(ref mcp) = opts.mcp_manager {
            // Prefer cached tools from Agent::from_config(); fall back to runtime fetch
            // only when a session-level MCP manager is provided (not the global one).
            let all_tools: Vec<(String, crate::mcp::McpTool)> = if std::ptr::eq(
                Arc::as_ptr(mcp),
                self.global_mcp
                    .as_ref()
                    .map(Arc::as_ptr)
                    .unwrap_or(std::ptr::null()),
            ) {
                // Same manager as global — use cached tools
                self.global_mcp_tools
                    .lock()
                    .expect("global_mcp_tools lock poisoned")
                    .clone()
            } else {
                // Session-level or merged manager — fetch at runtime
                match tokio::runtime::Handle::try_current() {
                    Ok(handle) => {
                        tokio::task::block_in_place(|| handle.block_on(mcp.get_all_tools()))
                    }
                    Err(_) => {
                        tracing::warn!(
                            "No async runtime available for session-level MCP tools — \
                                 MCP tools will not be registered"
                        );
                        vec![]
                    }
                }
            };

            let mut by_server: std::collections::HashMap<String, Vec<crate::mcp::McpTool>> =
                std::collections::HashMap::new();
            for (server, tool) in all_tools {
                by_server.entry(server).or_default().push(tool);
            }
            for (server_name, tools) in by_server {
                for tool in
                    crate::mcp::tools::create_mcp_tools(&server_name, tools, Arc::clone(mcp))
                {
                    tool_executor.register_dynamic_tool(tool);
                }
            }
        }

        let tool_defs = tool_executor.definitions();

        // Build prompt slots: start from session options or agent-level config
        let mut prompt_slots = opts
            .prompt_slots
            .clone()
            .unwrap_or_else(|| self.config.prompt_slots.clone());

        // Build effective skill registry: fork the agent-level registry (builtins + global
        // skill_dirs), then layer session-level skills on top. Forking ensures session skills
        // never pollute the shared agent-level registry.
        let base_registry = self
            .config
            .skill_registry
            .as_deref()
            .map(|r| r.fork())
            .unwrap_or_else(crate::skills::SkillRegistry::with_builtins);
        // Merge explicit session registry on top of the fork
        if let Some(ref r) = opts.skill_registry {
            for skill in r.all() {
                base_registry.register_unchecked(skill);
            }
        }
        // Load session-level skill dirs
        for dir in &opts.skill_dirs {
            if let Err(e) = base_registry.load_from_dir(dir) {
                tracing::warn!(
                    dir = %dir.display(),
                    error = %e,
                    "Failed to load session skill dir — skipping"
                );
            }
        }
        let effective_registry = Arc::new(base_registry);

        // Append skill directory listing to the extra prompt slot
        let skill_prompt = effective_registry.to_system_prompt();
        if !skill_prompt.is_empty() {
            prompt_slots.extra = match prompt_slots.extra {
                Some(existing) => Some(format!("{}\n\n{}", existing, skill_prompt)),
                None => Some(skill_prompt),
            };
        }

        // Resolve memory store: explicit store takes priority, then file_memory_dir
        let mut init_warning: Option<String> = None;
        let memory = {
            let store = if let Some(ref store) = opts.memory_store {
                Some(Arc::clone(store))
            } else if let Some(ref dir) = opts.file_memory_dir {
                match tokio::runtime::Handle::try_current() {
                    Ok(handle) => {
                        let dir = dir.clone();
                        match tokio::task::block_in_place(|| {
                            handle.block_on(FileMemoryStore::new(dir))
                        }) {
                            Ok(store) => Some(Arc::new(store) as Arc<dyn MemoryStore>),
                            Err(e) => {
                                let msg = format!("Failed to create file memory store: {}", e);
                                tracing::warn!("{}", msg);
                                init_warning = Some(msg);
                                None
                            }
                        }
                    }
                    Err(_) => {
                        let msg =
                            "No async runtime available for file memory store — memory disabled"
                                .to_string();
                        tracing::warn!("{}", msg);
                        init_warning = Some(msg);
                        None
                    }
                }
            } else {
                None
            };
            store.map(|s| Arc::new(crate::memory::AgentMemory::new(s)))
        };

        let base = self.config.clone();
        let config = AgentConfig {
            prompt_slots,
            tools: tool_defs,
            security_provider: opts.security_provider.clone(),
            permission_checker: opts.permission_checker.clone(),
            confirmation_manager: opts.confirmation_manager.clone(),
            context_providers: opts.context_providers.clone(),
            planning_enabled: opts.planning_enabled,
            goal_tracking: opts.goal_tracking,
            skill_registry: Some(effective_registry),
            max_parse_retries: opts.max_parse_retries.unwrap_or(base.max_parse_retries),
            tool_timeout_ms: opts.tool_timeout_ms.or(base.tool_timeout_ms),
            circuit_breaker_threshold: opts
                .circuit_breaker_threshold
                .unwrap_or(base.circuit_breaker_threshold),
            auto_compact: opts.auto_compact,
            auto_compact_threshold: opts
                .auto_compact_threshold
                .unwrap_or(crate::session::DEFAULT_AUTO_COMPACT_THRESHOLD),
            max_context_tokens: base.max_context_tokens,
            llm_client: Some(Arc::clone(&llm_client)),
            memory: memory.clone(),
            continuation_enabled: opts
                .continuation_enabled
                .unwrap_or(base.continuation_enabled),
            max_continuation_turns: opts
                .max_continuation_turns
                .unwrap_or(base.max_continuation_turns),
            ..base
        };

        // Create lane queue if configured
        // A shared broadcast channel is used for both queue events and subagent events.
        let (agent_event_tx, _) = broadcast::channel::<crate::agent::AgentEvent>(256);
        let command_queue = if let Some(ref queue_config) = opts.queue_config {
            let session_id = uuid::Uuid::new_v4().to_string();
            let rt = tokio::runtime::Handle::try_current();

            match rt {
                Ok(handle) => {
                    // We're inside an async runtime — use block_in_place
                    let queue = tokio::task::block_in_place(|| {
                        handle.block_on(SessionLaneQueue::new(
                            &session_id,
                            queue_config.clone(),
                            agent_event_tx.clone(),
                        ))
                    });
                    match queue {
                        Ok(q) => {
                            // Start the queue
                            let q = Arc::new(q);
                            let q2 = Arc::clone(&q);
                            tokio::task::block_in_place(|| {
                                handle.block_on(async { q2.start().await.ok() })
                            });
                            Some(q)
                        }
                        Err(e) => {
                            tracing::warn!("Failed to create session lane queue: {}", e);
                            None
                        }
                    }
                }
                Err(_) => {
                    tracing::warn!(
                        "No async runtime available for queue creation — queue disabled"
                    );
                    None
                }
            }
        } else {
            None
        };

        // Create tool context with search config if available
        let mut tool_context = ToolContext::new(canonical.clone());
        if let Some(ref search_config) = self.code_config.search {
            tool_context = tool_context.with_search_config(search_config.clone());
        }
        tool_context = tool_context.with_agent_event_tx(agent_event_tx);

        // Wire sandbox when configured.
        #[cfg(feature = "sandbox")]
        if let Some(ref sandbox_cfg) = opts.sandbox_config {
            let handle: Arc<dyn crate::sandbox::BashSandbox> =
                Arc::new(crate::sandbox::BoxSandboxHandle::new(
                    sandbox_cfg.clone(),
                    canonical.display().to_string(),
                ));
            // Update the registry's default context so that direct
            // `AgentSession::bash()` calls also use the sandbox.
            tool_executor.registry().set_sandbox(Arc::clone(&handle));
            tool_context = tool_context.with_sandbox(handle);
        }
        #[cfg(not(feature = "sandbox"))]
        if opts.sandbox_config.is_some() {
            tracing::warn!(
                "sandbox_config is set but the `sandbox` Cargo feature is not enabled \
                 — bash commands will run locally"
            );
        }

        let session_id = opts
            .session_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        // Resolve session store: explicit opts store > config sessions_dir > None
        let session_store = if opts.session_store.is_some() {
            opts.session_store.clone()
        } else if let Some(ref dir) = self.code_config.sessions_dir {
            match tokio::runtime::Handle::try_current() {
                Ok(handle) => {
                    let dir = dir.clone();
                    match tokio::task::block_in_place(|| {
                        handle.block_on(crate::store::FileSessionStore::new(dir))
                    }) {
                        Ok(store) => Some(Arc::new(store) as Arc<dyn crate::store::SessionStore>),
                        Err(e) => {
                            tracing::warn!(
                                "Failed to create session store from sessions_dir: {}",
                                e
                            );
                            None
                        }
                    }
                }
                Err(_) => {
                    tracing::warn!(
                        "No async runtime for sessions_dir store — persistence disabled"
                    );
                    None
                }
            }
        } else {
            None
        };

        Ok(AgentSession {
            llm_client,
            tool_executor,
            tool_context,
            memory: config.memory.clone(),
            config,
            workspace: canonical,
            session_id,
            history: RwLock::new(Vec::new()),
            command_queue,
            session_store,
            auto_save: opts.auto_save,
            hook_engine: Arc::new(crate::hooks::HookEngine::new()),
            init_warning,
            command_registry: CommandRegistry::new(),
            model_name: opts
                .model
                .clone()
                .or_else(|| self.code_config.default_model.clone())
                .unwrap_or_else(|| "unknown".to_string()),
            mcp_manager: opts
                .mcp_manager
                .clone()
                .or_else(|| self.global_mcp.clone())
                .unwrap_or_else(|| Arc::new(crate::mcp::manager::McpManager::new())),
        })
    }
}

// ============================================================================
// AgentSession
// ============================================================================

/// Workspace-bound session. All LLM and tool operations happen here.
///
/// History is automatically accumulated after each `send()` call.
/// Use `history()` to retrieve the current conversation log.
pub struct AgentSession {
    llm_client: Arc<dyn LlmClient>,
    tool_executor: Arc<ToolExecutor>,
    tool_context: ToolContext,
    config: AgentConfig,
    workspace: PathBuf,
    /// Unique session identifier.
    session_id: String,
    /// Internal conversation history, auto-updated after each `send()`.
    history: RwLock<Vec<Message>>,
    /// Optional lane queue for priority-based tool execution.
    command_queue: Option<Arc<SessionLaneQueue>>,
    /// Optional long-term memory.
    memory: Option<Arc<crate::memory::AgentMemory>>,
    /// Optional session store for persistence.
    session_store: Option<Arc<dyn crate::store::SessionStore>>,
    /// Auto-save after each `send()`.
    auto_save: bool,
    /// Hook engine for lifecycle event interception.
    hook_engine: Arc<crate::hooks::HookEngine>,
    /// Deferred init warning: emitted as PersistenceFailed on first send() if set.
    init_warning: Option<String>,
    /// Slash command registry for `/command` dispatch.
    command_registry: CommandRegistry,
    /// Model identifier for display (e.g., "anthropic/claude-sonnet-4-20250514").
    model_name: String,
    /// Shared MCP manager — all add_mcp_server / remove_mcp_server calls go here.
    mcp_manager: Arc<crate::mcp::manager::McpManager>,
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
    /// Build an `AgentLoop` with the session's configuration.
    ///
    /// Propagates the lane queue (if configured) for external task handling.
    fn build_agent_loop(&self) -> AgentLoop {
        let mut config = self.config.clone();
        config.hook_engine =
            Some(Arc::clone(&self.hook_engine) as Arc<dyn crate::hooks::HookExecutor>);
        // Always use live tool definitions so tools added via add_mcp_server() are visible
        // to the LLM. The config.tools snapshot taken at session creation misses dynamically
        // added MCP tools.
        config.tools = self.tool_executor.definitions();
        let mut agent_loop = AgentLoop::new(
            self.llm_client.clone(),
            self.tool_executor.clone(),
            self.tool_context.clone(),
            config,
        );
        if let Some(ref queue) = self.command_queue {
            agent_loop = agent_loop.with_queue(Arc::clone(queue));
        }
        agent_loop
    }

    /// Build a `CommandContext` from the current session state.
    fn build_command_context(&self) -> CommandContext {
        let history = read_or_recover(&self.history);

        // Collect tool names from config
        let tool_names: Vec<String> = self.config.tools.iter().map(|t| t.name.clone()).collect();

        // Derive MCP server info from tool names
        let mut mcp_map: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for name in &tool_names {
            if let Some(rest) = name.strip_prefix("mcp__") {
                if let Some((server, _)) = rest.split_once("__") {
                    *mcp_map.entry(server.to_string()).or_default() += 1;
                }
            }
        }
        let mut mcp_servers: Vec<(String, usize)> = mcp_map.into_iter().collect();
        mcp_servers.sort_by(|a, b| a.0.cmp(&b.0));

        CommandContext {
            session_id: self.session_id.clone(),
            workspace: self.workspace.display().to_string(),
            model: self.model_name.clone(),
            history_len: history.len(),
            total_tokens: 0,
            total_cost: 0.0,
            tool_names,
            mcp_servers,
        }
    }

    /// Get a reference to the slash command registry.
    pub fn command_registry(&self) -> &CommandRegistry {
        &self.command_registry
    }

    /// Register a custom slash command.
    pub fn register_command(&mut self, cmd: Arc<dyn crate::commands::SlashCommand>) {
        self.command_registry.register(cmd);
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
        // Slash command interception
        if CommandRegistry::is_command(prompt) {
            let ctx = self.build_command_context();
            if let Some(output) = self.command_registry.dispatch(prompt, &ctx) {
                return Ok(AgentResult {
                    text: output.text,
                    messages: history
                        .map(|h| h.to_vec())
                        .unwrap_or_else(|| read_or_recover(&self.history).clone()),
                    tool_calls_count: 0,
                    usage: crate::llm::TokenUsage::default(),
                });
            }
        }

        if let Some(ref w) = self.init_warning {
            tracing::warn!(session_id = %self.session_id, "Session init warning: {}", w);
        }
        let agent_loop = self.build_agent_loop();

        let use_internal = history.is_none();
        let effective_history = match history {
            Some(h) => h.to_vec(),
            None => read_or_recover(&self.history).clone(),
        };

        let result = agent_loop.execute(&effective_history, prompt, None).await?;

        // Auto-accumulate: only update internal history when no custom
        // history was provided.
        if use_internal {
            *write_or_recover(&self.history) = result.messages.clone();

            // Auto-save if configured
            if self.auto_save {
                if let Err(e) = self.save().await {
                    tracing::warn!("Auto-save failed for session {}: {}", self.session_id, e);
                }
            }
        }

        Ok(result)
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
        // Build a user message with text + images, then pass it as the last
        // history entry. We use an empty prompt so execute_loop doesn't add
        // a duplicate user message.
        let use_internal = history.is_none();
        let mut effective_history = match history {
            Some(h) => h.to_vec(),
            None => read_or_recover(&self.history).clone(),
        };
        effective_history.push(Message::user_with_attachments(prompt, attachments));

        let agent_loop = self.build_agent_loop();
        let result = agent_loop
            .execute_from_messages(effective_history, None, None)
            .await?;

        if use_internal {
            *write_or_recover(&self.history) = result.messages.clone();
            if self.auto_save {
                if let Err(e) = self.save().await {
                    tracing::warn!("Auto-save failed for session {}: {}", self.session_id, e);
                }
            }
        }

        Ok(result)
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
        let (tx, rx) = mpsc::channel(256);
        let mut effective_history = match history {
            Some(h) => h.to_vec(),
            None => read_or_recover(&self.history).clone(),
        };
        effective_history.push(Message::user_with_attachments(prompt, attachments));

        let agent_loop = self.build_agent_loop();
        let handle = tokio::spawn(async move {
            let _ = agent_loop
                .execute_from_messages(effective_history, None, Some(tx))
                .await;
        });

        Ok((rx, handle))
    }

    /// Send a prompt and stream events back.
    ///
    /// When `history` is `None`, uses the session's internal history
    /// (note: streaming does **not** auto-update internal history since
    /// the result is consumed asynchronously via the channel).
    /// When `Some`, uses the provided history instead.
    ///
    /// If the prompt starts with `/`, it is dispatched as a slash command
    /// and the result is emitted as a single `TextDelta` + `End` event.
    pub async fn stream(
        &self,
        prompt: &str,
        history: Option<&[Message]>,
    ) -> Result<(mpsc::Receiver<AgentEvent>, JoinHandle<()>)> {
        // Slash command interception for streaming
        if CommandRegistry::is_command(prompt) {
            let ctx = self.build_command_context();
            if let Some(output) = self.command_registry.dispatch(prompt, &ctx) {
                let (tx, rx) = mpsc::channel(256);
                let handle = tokio::spawn(async move {
                    let _ = tx
                        .send(AgentEvent::TextDelta {
                            text: output.text.clone(),
                        })
                        .await;
                    let _ = tx
                        .send(AgentEvent::End {
                            text: output.text.clone(),
                            usage: crate::llm::TokenUsage::default(),
                        })
                        .await;
                });
                return Ok((rx, handle));
            }
        }

        let (tx, rx) = mpsc::channel(256);
        let agent_loop = self.build_agent_loop();
        let effective_history = match history {
            Some(h) => h.to_vec(),
            None => read_or_recover(&self.history).clone(),
        };
        let prompt = prompt.to_string();

        let handle = tokio::spawn(async move {
            let _ = agent_loop
                .execute(&effective_history, &prompt, Some(tx))
                .await;
        });

        Ok((rx, handle))
    }

    /// Return a snapshot of the session's conversation history.
    pub fn history(&self) -> Vec<Message> {
        read_or_recover(&self.history).clone()
    }

    /// Return a reference to the session's memory, if configured.
    pub fn memory(&self) -> Option<&Arc<crate::memory::AgentMemory>> {
        self.memory.as_ref()
    }

    /// Return the session ID.
    pub fn id(&self) -> &str {
        &self.session_id
    }

    /// Return the session workspace path.
    pub fn workspace(&self) -> &std::path::Path {
        &self.workspace
    }

    /// Return any deferred init warning (e.g. memory store failed to initialize).
    pub fn init_warning(&self) -> Option<&str> {
        self.init_warning.as_deref()
    }

    /// Return the session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Return the definitions of all tools currently registered in this session.
    ///
    /// The list reflects the live state of the tool executor — tools added via
    /// `add_mcp_server()` appear immediately; tools removed via
    /// `remove_mcp_server()` disappear immediately.
    pub fn tool_definitions(&self) -> Vec<crate::llm::ToolDefinition> {
        self.tool_executor.definitions()
    }

    /// Return the names of all tools currently registered on this session.
    ///
    /// Equivalent to `tool_definitions().into_iter().map(|t| t.name).collect()`.
    /// Tools added via [`add_mcp_server`] appear immediately; tools removed via
    /// [`remove_mcp_server`] disappear immediately.
    pub fn tool_names(&self) -> Vec<String> {
        self.tool_executor
            .definitions()
            .into_iter()
            .map(|t| t.name)
            .collect()
    }

    // ========================================================================
    // Hook API
    // ========================================================================

    /// Register a hook for lifecycle event interception.
    pub fn register_hook(&self, hook: crate::hooks::Hook) {
        self.hook_engine.register(hook);
    }

    /// Unregister a hook by ID.
    pub fn unregister_hook(&self, hook_id: &str) -> Option<crate::hooks::Hook> {
        self.hook_engine.unregister(hook_id)
    }

    /// Register a handler for a specific hook.
    pub fn register_hook_handler(
        &self,
        hook_id: &str,
        handler: Arc<dyn crate::hooks::HookHandler>,
    ) {
        self.hook_engine.register_handler(hook_id, handler);
    }

    /// Unregister a hook handler by hook ID.
    pub fn unregister_hook_handler(&self, hook_id: &str) {
        self.hook_engine.unregister_handler(hook_id);
    }

    /// Get the number of registered hooks.
    pub fn hook_count(&self) -> usize {
        self.hook_engine.hook_count()
    }

    /// Save the session to the configured store.
    ///
    /// Returns `Ok(())` if saved successfully, or if no store is configured (no-op).
    pub async fn save(&self) -> Result<()> {
        let store = match &self.session_store {
            Some(s) => s,
            None => return Ok(()),
        };

        let history = read_or_recover(&self.history).clone();
        let now = chrono::Utc::now().timestamp();

        let data = crate::store::SessionData {
            id: self.session_id.clone(),
            config: crate::session::SessionConfig {
                name: String::new(),
                workspace: self.workspace.display().to_string(),
                system_prompt: Some(self.config.prompt_slots.build()),
                max_context_length: 200_000,
                auto_compact: false,
                auto_compact_threshold: crate::session::DEFAULT_AUTO_COMPACT_THRESHOLD,
                storage_type: crate::config::StorageBackend::File,
                queue_config: None,
                confirmation_policy: None,
                permission_policy: None,
                parent_id: None,
                security_config: None,
                hook_engine: None,
                planning_enabled: self.config.planning_enabled,
                goal_tracking: self.config.goal_tracking,
            },
            state: crate::session::SessionState::Active,
            messages: history,
            context_usage: crate::session::ContextUsage::default(),
            total_usage: crate::llm::TokenUsage::default(),
            total_cost: 0.0,
            model_name: None,
            cost_records: Vec::new(),
            tool_names: crate::store::SessionData::tool_names_from_definitions(&self.config.tools),
            thinking_enabled: false,
            thinking_budget: None,
            created_at: now,
            updated_at: now,
            llm_config: None,
            tasks: Vec::new(),
            parent_id: None,
        };

        store.save(&data).await?;
        tracing::debug!("Session {} saved", self.session_id);
        Ok(())
    }

    /// Read a file from the workspace.
    pub async fn read_file(&self, path: &str) -> Result<String> {
        let args = serde_json::json!({ "file_path": path });
        let result = self.tool_executor.execute("read", &args).await?;
        Ok(result.output)
    }

    /// Execute a bash command in the workspace.
    ///
    /// When a sandbox is configured via [`SessionOptions::with_sandbox()`],
    /// the command is routed through the A3S Box sandbox.
    pub async fn bash(&self, command: &str) -> Result<String> {
        let args = serde_json::json!({ "command": command });
        let result = self
            .tool_executor
            .execute_with_context("bash", &args, &self.tool_context)
            .await?;
        Ok(result.output)
    }

    /// Search for files matching a glob pattern.
    pub async fn glob(&self, pattern: &str) -> Result<Vec<String>> {
        let args = serde_json::json!({ "pattern": pattern });
        let result = self.tool_executor.execute("glob", &args).await?;
        let files: Vec<String> = result
            .output
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| l.to_string())
            .collect();
        Ok(files)
    }

    /// Search file contents with a regex pattern.
    pub async fn grep(&self, pattern: &str) -> Result<String> {
        let args = serde_json::json!({ "pattern": pattern });
        let result = self.tool_executor.execute("grep", &args).await?;
        Ok(result.output)
    }

    /// Execute a tool by name, bypassing the LLM.
    pub async fn tool(&self, name: &str, args: serde_json::Value) -> Result<ToolCallResult> {
        let result = self.tool_executor.execute(name, &args).await?;
        Ok(ToolCallResult {
            name: name.to_string(),
            output: result.output,
            exit_code: result.exit_code,
        })
    }

    // ========================================================================
    // Queue API
    // ========================================================================

    /// Returns whether this session has a lane queue configured.
    pub fn has_queue(&self) -> bool {
        self.command_queue.is_some()
    }

    /// Configure a lane's handler mode (Internal/External/Hybrid).
    ///
    /// Only effective when a queue is configured via `SessionOptions::with_queue_config`.
    pub async fn set_lane_handler(&self, lane: SessionLane, config: LaneHandlerConfig) {
        if let Some(ref queue) = self.command_queue {
            queue.set_lane_handler(lane, config).await;
        }
    }

    /// Complete an external task by ID.
    ///
    /// Returns `true` if the task was found and completed, `false` if not found.
    pub async fn complete_external_task(&self, task_id: &str, result: ExternalTaskResult) -> bool {
        if let Some(ref queue) = self.command_queue {
            queue.complete_external_task(task_id, result).await
        } else {
            false
        }
    }

    /// Get pending external tasks awaiting completion by an external handler.
    pub async fn pending_external_tasks(&self) -> Vec<ExternalTask> {
        if let Some(ref queue) = self.command_queue {
            queue.pending_external_tasks().await
        } else {
            Vec::new()
        }
    }

    /// Get queue statistics (pending, active, external counts per lane).
    pub async fn queue_stats(&self) -> SessionQueueStats {
        if let Some(ref queue) = self.command_queue {
            queue.stats().await
        } else {
            SessionQueueStats::default()
        }
    }

    /// Get a metrics snapshot from the queue (if metrics are enabled).
    pub async fn queue_metrics(&self) -> Option<MetricsSnapshot> {
        if let Some(ref queue) = self.command_queue {
            queue.metrics_snapshot().await
        } else {
            None
        }
    }

    /// Submit a command directly to the session's lane queue.
    ///
    /// Returns `Err` if no queue is configured (i.e. session was created without
    /// `SessionOptions::with_queue_config`). On success, returns a receiver that
    /// resolves to the command's result when execution completes.
    pub async fn submit(
        &self,
        lane: SessionLane,
        command: Box<dyn crate::queue::SessionCommand>,
    ) -> anyhow::Result<tokio::sync::oneshot::Receiver<anyhow::Result<serde_json::Value>>> {
        let queue = self
            .command_queue
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No queue configured on this session"))?;
        Ok(queue.submit(lane, command).await)
    }

    /// Submit multiple commands to the session's lane queue in a single batch.
    ///
    /// More efficient than calling `submit()` in a loop: handler config is fetched
    /// once and task IDs are generated atomically. Returns `Err` if no queue is
    /// configured. On success, returns one receiver per command in the same order
    /// as the input slice.
    pub async fn submit_batch(
        &self,
        lane: SessionLane,
        commands: Vec<Box<dyn crate::queue::SessionCommand>>,
    ) -> anyhow::Result<Vec<tokio::sync::oneshot::Receiver<anyhow::Result<serde_json::Value>>>>
    {
        let queue = self
            .command_queue
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No queue configured on this session"))?;
        Ok(queue.submit_batch(lane, commands).await)
    }

    /// Get dead letters from the queue's DLQ (if DLQ is enabled).
    pub async fn dead_letters(&self) -> Vec<DeadLetter> {
        if let Some(ref queue) = self.command_queue {
            queue.dead_letters().await
        } else {
            Vec::new()
        }
    }

    // ========================================================================
    // MCP API
    // ========================================================================

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
        let server_name = config.name.clone();
        self.mcp_manager.register_server(config).await;
        self.mcp_manager.connect(&server_name).await.map_err(|e| {
            crate::error::CodeError::Tool {
                tool: server_name.clone(),
                message: format!("Failed to connect MCP server: {}", e),
            }
        })?;

        let tools = self.mcp_manager.get_server_tools(&server_name).await;
        let count = tools.len();

        for tool in
            crate::mcp::tools::create_mcp_tools(&server_name, tools, Arc::clone(&self.mcp_manager))
        {
            self.tool_executor.register_dynamic_tool(tool);
        }

        tracing::info!(
            session_id = %self.session_id,
            server = server_name,
            tools = count,
            "MCP server added to live session"
        );

        Ok(count)
    }

    /// Remove an MCP server from this session.
    ///
    /// Disconnects the server and unregisters all its tools from the executor.
    /// No-op if the server was never added.
    pub async fn remove_mcp_server(&self, server_name: &str) -> crate::error::Result<()> {
        self.tool_executor
            .unregister_tools_by_prefix(&format!("mcp__{server_name}__"));
        self.mcp_manager
            .disconnect(server_name)
            .await
            .map_err(|e| crate::error::CodeError::Tool {
                tool: server_name.to_string(),
                message: format!("Failed to disconnect MCP server: {}", e),
            })?;
        tracing::info!(
            session_id = %self.session_id,
            server = server_name,
            "MCP server removed from live session"
        );
        Ok(())
    }

    /// Return the connection status of all MCP servers registered with this session.
    pub async fn mcp_status(
        &self,
    ) -> std::collections::HashMap<String, crate::mcp::McpServerStatus> {
        self.mcp_manager.get_status().await
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelConfig, ModelModalities, ProviderConfig};
    use crate::store::SessionStore;

    #[tokio::test]
    async fn test_session_submit_no_queue_returns_err() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let session = agent.session(".", None).unwrap();
        struct Noop;
        #[async_trait::async_trait]
        impl crate::queue::SessionCommand for Noop {
            async fn execute(&self) -> anyhow::Result<serde_json::Value> {
                Ok(serde_json::json!(null))
            }
            fn command_type(&self) -> &str {
                "noop"
            }
        }
        let result: anyhow::Result<
            tokio::sync::oneshot::Receiver<anyhow::Result<serde_json::Value>>,
        > = session.submit(SessionLane::Query, Box::new(Noop)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No queue"));
    }

    #[tokio::test]
    async fn test_session_submit_batch_no_queue_returns_err() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let session = agent.session(".", None).unwrap();
        struct Noop;
        #[async_trait::async_trait]
        impl crate::queue::SessionCommand for Noop {
            async fn execute(&self) -> anyhow::Result<serde_json::Value> {
                Ok(serde_json::json!(null))
            }
            fn command_type(&self) -> &str {
                "noop"
            }
        }
        let cmds: Vec<Box<dyn crate::queue::SessionCommand>> = vec![Box::new(Noop)];
        let result: anyhow::Result<
            Vec<tokio::sync::oneshot::Receiver<anyhow::Result<serde_json::Value>>>,
        > = session.submit_batch(SessionLane::Query, cmds).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No queue"));
    }

    fn test_config() -> CodeConfig {
        CodeConfig {
            default_model: Some("anthropic/claude-sonnet-4-20250514".to_string()),
            providers: vec![
                ProviderConfig {
                    name: "anthropic".to_string(),
                    api_key: Some("test-key".to_string()),
                    base_url: None,
                    models: vec![ModelConfig {
                        id: "claude-sonnet-4-20250514".to_string(),
                        name: "Claude Sonnet 4".to_string(),
                        family: "claude-sonnet".to_string(),
                        api_key: None,
                        base_url: None,
                        attachment: false,
                        reasoning: false,
                        tool_call: true,
                        temperature: true,
                        release_date: None,
                        modalities: ModelModalities::default(),
                        cost: Default::default(),
                        limit: Default::default(),
                    }],
                },
                ProviderConfig {
                    name: "openai".to_string(),
                    api_key: Some("test-openai-key".to_string()),
                    base_url: None,
                    models: vec![ModelConfig {
                        id: "gpt-4o".to_string(),
                        name: "GPT-4o".to_string(),
                        family: "gpt-4".to_string(),
                        api_key: None,
                        base_url: None,
                        attachment: false,
                        reasoning: false,
                        tool_call: true,
                        temperature: true,
                        release_date: None,
                        modalities: ModelModalities::default(),
                        cost: Default::default(),
                        limit: Default::default(),
                    }],
                },
            ],
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_from_config() {
        let agent = Agent::from_config(test_config()).await;
        assert!(agent.is_ok());
    }

    #[tokio::test]
    async fn test_session_default() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let session = agent.session("/tmp/test-workspace", None);
        assert!(session.is_ok());
        let debug = format!("{:?}", session.unwrap());
        assert!(debug.contains("AgentSession"));
    }

    #[tokio::test]
    async fn test_session_with_model_override() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let opts = SessionOptions::new().with_model("openai/gpt-4o");
        let session = agent.session("/tmp/test-workspace", Some(opts));
        assert!(session.is_ok());
    }

    #[tokio::test]
    async fn test_session_with_invalid_model_format() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let opts = SessionOptions::new().with_model("gpt-4o");
        let session = agent.session("/tmp/test-workspace", Some(opts));
        assert!(session.is_err());
    }

    #[tokio::test]
    async fn test_session_with_model_not_found() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let opts = SessionOptions::new().with_model("openai/nonexistent");
        let session = agent.session("/tmp/test-workspace", Some(opts));
        assert!(session.is_err());
    }

    #[tokio::test]
    async fn test_new_with_hcl_string() {
        let hcl = r#"
            default_model = "anthropic/claude-sonnet-4-20250514"
            providers {
                name    = "anthropic"
                api_key = "test-key"
                models {
                    id   = "claude-sonnet-4-20250514"
                    name = "Claude Sonnet 4"
                }
            }
        "#;
        let agent = Agent::new(hcl).await;
        assert!(agent.is_ok());
    }

    #[tokio::test]
    async fn test_create_alias_hcl() {
        let hcl = r#"
            default_model = "anthropic/claude-sonnet-4-20250514"
            providers {
                name    = "anthropic"
                api_key = "test-key"
                models {
                    id   = "claude-sonnet-4-20250514"
                    name = "Claude Sonnet 4"
                }
            }
        "#;
        let agent = Agent::create(hcl).await;
        assert!(agent.is_ok());
    }

    #[tokio::test]
    async fn test_create_and_new_produce_same_result() {
        let hcl = r#"
            default_model = "anthropic/claude-sonnet-4-20250514"
            providers {
                name    = "anthropic"
                api_key = "test-key"
                models {
                    id   = "claude-sonnet-4-20250514"
                    name = "Claude Sonnet 4"
                }
            }
        "#;
        let agent_new = Agent::new(hcl).await;
        let agent_create = Agent::create(hcl).await;
        assert!(agent_new.is_ok());
        assert!(agent_create.is_ok());

        // Both should produce working sessions
        let session_new = agent_new.unwrap().session("/tmp/test-ws-new", None);
        let session_create = agent_create.unwrap().session("/tmp/test-ws-create", None);
        assert!(session_new.is_ok());
        assert!(session_create.is_ok());
    }

    #[test]
    fn test_from_config_requires_default_model() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = CodeConfig {
            providers: vec![ProviderConfig {
                name: "anthropic".to_string(),
                api_key: Some("test-key".to_string()),
                base_url: None,
                models: vec![],
            }],
            ..Default::default()
        };
        let result = rt.block_on(Agent::from_config(config));
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_history_empty_on_new_session() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let session = agent.session("/tmp/test-workspace", None).unwrap();
        assert!(session.history().is_empty());
    }

    #[tokio::test]
    async fn test_session_options_with_agent_dir() {
        let opts = SessionOptions::new()
            .with_agent_dir("/tmp/agents")
            .with_agent_dir("/tmp/more-agents");
        assert_eq!(opts.agent_dirs.len(), 2);
        assert_eq!(opts.agent_dirs[0], PathBuf::from("/tmp/agents"));
        assert_eq!(opts.agent_dirs[1], PathBuf::from("/tmp/more-agents"));
    }

    // ========================================================================
    // Queue Integration Tests
    // ========================================================================

    #[test]
    fn test_session_options_with_queue_config() {
        let qc = SessionQueueConfig::default().with_lane_features();
        let opts = SessionOptions::new().with_queue_config(qc.clone());
        assert!(opts.queue_config.is_some());

        let config = opts.queue_config.unwrap();
        assert!(config.enable_dlq);
        assert!(config.enable_metrics);
        assert!(config.enable_alerts);
        assert_eq!(config.default_timeout_ms, Some(60_000));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_with_queue_config() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let qc = SessionQueueConfig::default();
        let opts = SessionOptions::new().with_queue_config(qc);
        let session = agent.session("/tmp/test-workspace-queue", Some(opts));
        assert!(session.is_ok());
        let session = session.unwrap();
        assert!(session.has_queue());
    }

    #[tokio::test]
    async fn test_session_without_queue_config() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let session = agent.session("/tmp/test-workspace-noqueue", None).unwrap();
        assert!(!session.has_queue());
    }

    #[tokio::test]
    async fn test_session_queue_stats_without_queue() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let session = agent.session("/tmp/test-workspace-stats", None).unwrap();
        let stats = session.queue_stats().await;
        // Without a queue, stats should have zero values
        assert_eq!(stats.total_pending, 0);
        assert_eq!(stats.total_active, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_queue_stats_with_queue() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let qc = SessionQueueConfig::default();
        let opts = SessionOptions::new().with_queue_config(qc);
        let session = agent
            .session("/tmp/test-workspace-qstats", Some(opts))
            .unwrap();
        let stats = session.queue_stats().await;
        // Fresh queue with no commands should have zero stats
        assert_eq!(stats.total_pending, 0);
        assert_eq!(stats.total_active, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_pending_external_tasks_empty() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let qc = SessionQueueConfig::default();
        let opts = SessionOptions::new().with_queue_config(qc);
        let session = agent
            .session("/tmp/test-workspace-ext", Some(opts))
            .unwrap();
        let tasks = session.pending_external_tasks().await;
        assert!(tasks.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_dead_letters_empty() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let qc = SessionQueueConfig::default().with_dlq(Some(100));
        let opts = SessionOptions::new().with_queue_config(qc);
        let session = agent
            .session("/tmp/test-workspace-dlq", Some(opts))
            .unwrap();
        let dead = session.dead_letters().await;
        assert!(dead.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_queue_metrics_disabled() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        // Metrics not enabled
        let qc = SessionQueueConfig::default();
        let opts = SessionOptions::new().with_queue_config(qc);
        let session = agent
            .session("/tmp/test-workspace-nomet", Some(opts))
            .unwrap();
        let metrics = session.queue_metrics().await;
        assert!(metrics.is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_queue_metrics_enabled() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let qc = SessionQueueConfig::default().with_metrics();
        let opts = SessionOptions::new().with_queue_config(qc);
        let session = agent
            .session("/tmp/test-workspace-met", Some(opts))
            .unwrap();
        let metrics = session.queue_metrics().await;
        assert!(metrics.is_some());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_set_lane_handler() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let qc = SessionQueueConfig::default();
        let opts = SessionOptions::new().with_queue_config(qc);
        let session = agent
            .session("/tmp/test-workspace-handler", Some(opts))
            .unwrap();

        // Set Execute lane to External mode
        session
            .set_lane_handler(
                SessionLane::Execute,
                LaneHandlerConfig {
                    mode: crate::queue::TaskHandlerMode::External,
                    timeout_ms: 30_000,
                },
            )
            .await;

        // No panic = success. The handler config is stored internally.
        // We can't directly read it back but we verify no errors.
    }

    // ========================================================================
    // Session Persistence Tests
    // ========================================================================

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_has_id() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let session = agent.session("/tmp/test-ws-id", None).unwrap();
        // Auto-generated UUID
        assert!(!session.session_id().is_empty());
        assert_eq!(session.session_id().len(), 36); // UUID format
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_explicit_id() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let opts = SessionOptions::new().with_session_id("my-session-42");
        let session = agent.session("/tmp/test-ws-eid", Some(opts)).unwrap();
        assert_eq!(session.session_id(), "my-session-42");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_save_no_store() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let session = agent.session("/tmp/test-ws-save", None).unwrap();
        // save() is a no-op when no store is configured
        session.save().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_save_and_load() {
        let store = Arc::new(crate::store::MemorySessionStore::new());
        let agent = Agent::from_config(test_config()).await.unwrap();

        let opts = SessionOptions::new()
            .with_session_store(store.clone())
            .with_session_id("persist-test");
        let session = agent.session("/tmp/test-ws-persist", Some(opts)).unwrap();

        // Save empty session
        session.save().await.unwrap();

        // Verify it was stored
        assert!(store.exists("persist-test").await.unwrap());

        let data = store.load("persist-test").await.unwrap().unwrap();
        assert_eq!(data.id, "persist-test");
        assert!(data.messages.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_save_with_history() {
        let store = Arc::new(crate::store::MemorySessionStore::new());
        let agent = Agent::from_config(test_config()).await.unwrap();

        let opts = SessionOptions::new()
            .with_session_store(store.clone())
            .with_session_id("history-test");
        let session = agent.session("/tmp/test-ws-hist", Some(opts)).unwrap();

        // Manually inject history
        {
            let mut h = session.history.write().unwrap();
            h.push(Message::user("Hello"));
            h.push(Message::user("How are you?"));
        }

        session.save().await.unwrap();

        let data = store.load("history-test").await.unwrap().unwrap();
        assert_eq!(data.messages.len(), 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_resume_session() {
        let store = Arc::new(crate::store::MemorySessionStore::new());
        let agent = Agent::from_config(test_config()).await.unwrap();

        // Create and save a session with history
        let opts = SessionOptions::new()
            .with_session_store(store.clone())
            .with_session_id("resume-test");
        let session = agent.session("/tmp/test-ws-resume", Some(opts)).unwrap();
        {
            let mut h = session.history.write().unwrap();
            h.push(Message::user("What is Rust?"));
            h.push(Message::user("Tell me more"));
        }
        session.save().await.unwrap();

        // Resume the session
        let opts2 = SessionOptions::new().with_session_store(store.clone());
        let resumed = agent.resume_session("resume-test", opts2).unwrap();

        assert_eq!(resumed.session_id(), "resume-test");
        let history = resumed.history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].text(), "What is Rust?");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_resume_session_not_found() {
        let store = Arc::new(crate::store::MemorySessionStore::new());
        let agent = Agent::from_config(test_config()).await.unwrap();

        let opts = SessionOptions::new().with_session_store(store.clone());
        let result = agent.resume_session("nonexistent", opts);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_resume_session_no_store() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let opts = SessionOptions::new();
        let result = agent.resume_session("any-id", opts);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("session_store"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_file_session_store_persistence() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = Arc::new(
            crate::store::FileSessionStore::new(dir.path())
                .await
                .unwrap(),
        );
        let agent = Agent::from_config(test_config()).await.unwrap();

        // Save
        let opts = SessionOptions::new()
            .with_session_store(store.clone())
            .with_session_id("file-persist");
        let session = agent
            .session("/tmp/test-ws-file-persist", Some(opts))
            .unwrap();
        {
            let mut h = session.history.write().unwrap();
            h.push(Message::user("test message"));
        }
        session.save().await.unwrap();

        // Load from a fresh store instance pointing to same dir
        let store2 = Arc::new(
            crate::store::FileSessionStore::new(dir.path())
                .await
                .unwrap(),
        );
        let data = store2.load("file-persist").await.unwrap().unwrap();
        assert_eq!(data.messages.len(), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_options_builders() {
        let opts = SessionOptions::new()
            .with_session_id("test-id")
            .with_auto_save(true);
        assert_eq!(opts.session_id, Some("test-id".to_string()));
        assert!(opts.auto_save);
    }

    // ========================================================================
    // Sandbox Tests
    // ========================================================================

    #[test]
    fn test_session_options_with_sandbox_sets_config() {
        use crate::sandbox::SandboxConfig;
        let cfg = SandboxConfig {
            image: "ubuntu:22.04".into(),
            memory_mb: 1024,
            ..SandboxConfig::default()
        };
        let opts = SessionOptions::new().with_sandbox(cfg);
        assert!(opts.sandbox_config.is_some());
        let sc = opts.sandbox_config.unwrap();
        assert_eq!(sc.image, "ubuntu:22.04");
        assert_eq!(sc.memory_mb, 1024);
    }

    #[test]
    fn test_session_options_default_has_no_sandbox() {
        let opts = SessionOptions::default();
        assert!(opts.sandbox_config.is_none());
    }

    #[tokio::test]
    async fn test_session_debug_includes_sandbox_config() {
        use crate::sandbox::SandboxConfig;
        let opts = SessionOptions::new().with_sandbox(SandboxConfig::default());
        let debug = format!("{:?}", opts);
        assert!(debug.contains("sandbox_config"));
    }

    #[tokio::test]
    async fn test_session_build_with_sandbox_config_no_feature_warn() {
        // When feature is not enabled, build_session should still succeed
        // (it just logs a warning). With feature enabled, it creates a handle.
        let agent = Agent::from_config(test_config()).await.unwrap();
        let opts = SessionOptions::new().with_sandbox(crate::sandbox::SandboxConfig::default());
        // build_session should not fail even if sandbox feature is off
        let session = agent.session("/tmp/test-sandbox-session", Some(opts));
        assert!(session.is_ok());
    }

    // ========================================================================
    // Memory Integration Tests
    // ========================================================================

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_with_memory_store() {
        use a3s_memory::InMemoryStore;
        let store = Arc::new(InMemoryStore::new());
        let agent = Agent::from_config(test_config()).await.unwrap();
        let opts = SessionOptions::new().with_memory(store);
        let session = agent.session("/tmp/test-ws-memory", Some(opts)).unwrap();
        assert!(session.memory().is_some());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_without_memory_store() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let session = agent.session("/tmp/test-ws-no-memory", None).unwrap();
        assert!(session.memory().is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_memory_wired_into_config() {
        use a3s_memory::InMemoryStore;
        let store = Arc::new(InMemoryStore::new());
        let agent = Agent::from_config(test_config()).await.unwrap();
        let opts = SessionOptions::new().with_memory(store);
        let session = agent
            .session("/tmp/test-ws-mem-config", Some(opts))
            .unwrap();
        // memory is accessible via the public session API
        assert!(session.memory().is_some());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_with_file_memory() {
        let dir = tempfile::TempDir::new().unwrap();
        let agent = Agent::from_config(test_config()).await.unwrap();
        let opts = SessionOptions::new().with_file_memory(dir.path());
        let session = agent.session("/tmp/test-ws-file-mem", Some(opts)).unwrap();
        assert!(session.memory().is_some());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_memory_remember_and_recall() {
        use a3s_memory::InMemoryStore;
        let store = Arc::new(InMemoryStore::new());
        let agent = Agent::from_config(test_config()).await.unwrap();
        let opts = SessionOptions::new().with_memory(store);
        let session = agent
            .session("/tmp/test-ws-mem-recall", Some(opts))
            .unwrap();

        let memory = session.memory().unwrap();
        memory
            .remember_success("write a file", &["write".to_string()], "done")
            .await
            .unwrap();

        let results = memory.recall_similar("write", 5).await.unwrap();
        assert!(!results.is_empty());
        let stats = memory.stats().await.unwrap();
        assert_eq!(stats.long_term_count, 1);
    }

    // ========================================================================
    // Tool timeout tests
    // ========================================================================

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_tool_timeout_configured() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let opts = SessionOptions::new().with_tool_timeout(5000);
        let session = agent.session("/tmp/test-ws-timeout", Some(opts)).unwrap();
        assert!(!session.id().is_empty());
    }

    // ========================================================================
    // Queue fallback tests
    // ========================================================================

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_without_queue_builds_ok() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let session = agent.session("/tmp/test-ws-no-queue", None).unwrap();
        assert!(!session.id().is_empty());
    }

    // ========================================================================
    // Concurrent history access tests
    // ========================================================================

    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_history_reads() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let session = Arc::new(agent.session("/tmp/test-ws-concurrent", None).unwrap());

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let s = Arc::clone(&session);
                tokio::spawn(async move { s.history().len() })
            })
            .collect();

        for h in handles {
            h.await.unwrap();
        }
    }

    // ========================================================================
    // init_warning tests
    // ========================================================================

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_no_init_warning_without_file_memory() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let session = agent.session("/tmp/test-ws-no-warn", None).unwrap();
        assert!(session.init_warning().is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_with_mcp_manager_builds_ok() {
        use crate::mcp::manager::McpManager;
        let mcp = Arc::new(McpManager::new());
        let agent = Agent::from_config(test_config()).await.unwrap();
        let opts = SessionOptions::new().with_mcp(mcp);
        // No servers connected — should build fine with zero MCP tools registered
        let session = agent.session("/tmp/test-ws-mcp", Some(opts)).unwrap();
        assert!(!session.id().is_empty());
    }

    #[test]
    fn test_session_command_is_pub() {
        // Compile-time check: SessionCommand must be importable from crate root
        use crate::SessionCommand;
        let _ = std::marker::PhantomData::<Box<dyn SessionCommand>>;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_session_submit_with_queue_executes() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let qc = SessionQueueConfig::default();
        let opts = SessionOptions::new().with_queue_config(qc);
        let session = agent
            .session("/tmp/test-ws-submit-exec", Some(opts))
            .unwrap();

        struct Echo(serde_json::Value);
        #[async_trait::async_trait]
        impl crate::queue::SessionCommand for Echo {
            async fn execute(&self) -> anyhow::Result<serde_json::Value> {
                Ok(self.0.clone())
            }
            fn command_type(&self) -> &str {
                "echo"
            }
        }

        let rx = session
            .submit(
                SessionLane::Query,
                Box::new(Echo(serde_json::json!({"ok": true}))),
            )
            .await
            .expect("submit should succeed with queue configured");

        let result = tokio::time::timeout(std::time::Duration::from_secs(2), rx)
            .await
            .expect("timed out waiting for command result")
            .expect("channel closed before result")
            .expect("command returned an error");

        assert_eq!(result["ok"], true);
    }
}
