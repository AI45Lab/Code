//! Session option builder interface.
//!
//! `SessionOptions` is the host-facing capability configuration for a session.
//! Keeping the builder implementation here lets `agent_api.rs` keep the type
//! shape visible while moving option construction behavior behind this module.

use super::SessionOptions;
use crate::prompts::{PlanningMode, SystemPromptSlots};
use crate::queue::SessionQueueConfig;
use crate::subagent::WorkerAgentSpec;
use a3s_memory::MemoryStore;
use std::path::PathBuf;
use std::sync::Arc;

impl std::fmt::Debug for SessionOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionOptions")
            .field("model", &self.model)
            .field("agent_dirs", &self.agent_dirs)
            .field("worker_agents", &self.worker_agents.len())
            .field("skill_dirs", &self.skill_dirs)
            .field("queue_config", &self.queue_config)
            .field("security_provider", &self.security_provider.is_some())
            .field("llm_client", &self.llm_client.is_some())
            .field("context_providers", &self.context_providers.len())
            .field("confirmation_manager", &self.confirmation_manager.is_some())
            .field("permission_checker", &self.permission_checker.is_some())
            .field("permission_policy", &self.permission_policy.is_some())
            .field("planning_mode", &self.planning_mode)
            .field("goal_tracking", &self.goal_tracking)
            .field(
                "skill_registry",
                &self
                    .skill_registry
                    .as_ref()
                    .map(|r| format!("{} skills", r.len())),
            )
            .field(
                "enforce_active_skill_tool_restrictions",
                &self.enforce_active_skill_tool_restrictions,
            )
            .field("memory_store", &self.memory_store.is_some())
            .field("session_store", &self.session_store.is_some())
            .field("session_id", &self.session_id)
            .field("auto_save", &self.auto_save)
            .field("artifact_store_limits", &self.artifact_store_limits)
            .field("max_parse_retries", &self.max_parse_retries)
            .field("tool_timeout_ms", &self.tool_timeout_ms)
            .field("circuit_breaker_threshold", &self.circuit_breaker_threshold)
            .field("sandbox_handle", &self.sandbox_handle.is_some())
            .field("workspace_services", &self.workspace_services.is_some())
            .field("auto_compact", &self.auto_compact)
            .field("auto_compact_threshold", &self.auto_compact_threshold)
            .field("continuation_enabled", &self.continuation_enabled)
            .field("max_continuation_turns", &self.max_continuation_turns)
            .field("mcp_manager", &self.mcp_manager.is_some())
            .field("temperature", &self.temperature)
            .field("thinking_budget", &self.thinking_budget)
            .field("max_tool_rounds", &self.max_tool_rounds)
            .field("max_parallel_tasks", &self.max_parallel_tasks)
            .field("auto_delegation", &self.auto_delegation)
            .field("manual_delegation_enabled", &self.manual_delegation_enabled)
            .field("auto_parallel_delegation", &self.auto_parallel_delegation)
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

    /// Register a cattle-style worker with this session's task delegation registry.
    pub fn with_worker_agent(mut self, spec: WorkerAgentSpec) -> Self {
        self.worker_agents.push(spec);
        self
    }

    /// Register multiple cattle-style workers with this session.
    pub fn with_worker_agents<I>(mut self, specs: I) -> Self
    where
        I: IntoIterator<Item = WorkerAgentSpec>,
    {
        self.worker_agents.extend(specs);
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

    /// Provide a custom LLM client for this session.
    ///
    /// When set, this client is used directly, overriding the `provider/model`
    /// factory resolution. Use it to plug in a provider the built-in factory
    /// does not cover, a deterministic record/replay client for tests, or an
    /// HTTP-layer proxy/audit wrapper. Mirrors [`Self::with_workspace_backend`];
    /// the `provider/model` config path remains the default when unset.
    pub fn with_llm_client(mut self, client: Arc<dyn crate::llm::LlmClient>) -> Self {
        self.llm_client = Some(client);
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

    /// Set a confirmation policy for HITL
    ///
    /// The policy will be used to create a ConfirmationManager when the session is built.
    /// This is the preferred way to configure HITL from the Node SDK.
    pub fn with_confirmation_policy(mut self, policy: crate::hitl::ConfirmationPolicy) -> Self {
        self.confirmation_policy = Some(policy);
        self
    }

    /// Set a serializable permission policy for tool execution.
    pub fn with_permission_policy(mut self, policy: crate::permissions::PermissionPolicy) -> Self {
        self.permission_checker = Some(Arc::new(policy.clone()));
        self.permission_policy = Some(policy);
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

    /// Set planning mode
    pub fn with_planning_mode(mut self, mode: PlanningMode) -> Self {
        self.planning_mode = mode;
        self
    }

    /// Enable planning (shortcut for `with_planning_mode(PlanningMode::Enabled)`)
    pub fn with_planning(mut self, enabled: bool) -> Self {
        self.planning_mode = if enabled {
            PlanningMode::Enabled
        } else {
            PlanningMode::Disabled
        };
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

    /// Enable or disable legacy global active-skill `allowed-tools` restrictions.
    ///
    /// The default is disabled: active skills do not block ordinary session
    /// tools before the host permission/AHP/HITL approval chain runs.
    pub fn with_active_skill_tool_restrictions(mut self, enabled: bool) -> Self {
        self.enforce_active_skill_tool_restrictions = Some(enabled);
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

    /// Tag the session with a host-defined tenant id. Opaque to the
    /// framework — propagated to `SessionData`, hooks, and traces.
    pub fn with_tenant_id(mut self, tenant: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant.into());
        self
    }

    /// Tag the session with the id of the principal (user / service
    /// account / etc.) that triggered it.
    pub fn with_principal(mut self, principal: impl Into<String>) -> Self {
        self.principal = Some(principal.into());
        self
    }

    /// Tag the session with the id of the agent template / definition it
    /// was instantiated from.
    pub fn with_agent_template_id(mut self, template_id: impl Into<String>) -> Self {
        self.agent_template_id = Some(template_id.into());
        self
    }

    /// Attach a distributed-trace correlation id so this session's events
    /// can be joined with upstream/downstream work.
    pub fn with_correlation_id(mut self, corr: impl Into<String>) -> Self {
        self.correlation_id = Some(corr.into());
        self
    }

    /// Install a host-supplied [`BudgetGuard`](crate::budget::BudgetGuard).
    ///
    /// The guard is consulted before every LLM call (and after, for
    /// usage accounting). When unset, no budget enforcement happens.
    pub fn with_budget_guard(mut self, guard: Arc<dyn crate::budget::BudgetGuard>) -> Self {
        self.budget_guard = Some(guard);
        self
    }

    /// Install a host-provided [`HostEnv`](crate::host_env::HostEnv) for
    /// deterministic ID generation and time. Replaces the framework
    /// default of `uuid::Uuid::new_v4()` + wall clock — used by
    /// host replay infrastructure to recreate a run bit-identical on
    /// another node.
    pub fn with_host_env(mut self, env: Arc<crate::host_env::HostEnv>) -> Self {
        self.host_env = Some(env);
        self
    }

    /// Install FIFO retention caps for the session's in-memory stores.
    ///
    /// Without these caps the in-memory run store, trace sink, and
    /// subagent task tracker grow unboundedly across long-running
    /// sessions. Hosts running thousands of long-lived sessions per
    /// node should set sensible caps (e.g. retain the last 100 runs,
    /// 5000 events per run, 10000 trace events, 1000 terminal subagent
    /// tasks). When unset, the framework keeps every record — the
    /// pre-existing behaviour.
    pub fn with_retention_limits(
        mut self,
        limits: crate::retention::SessionRetentionLimits,
    ) -> Self {
        self.retention_limits = Some(limits);
        self
    }

    /// Enable auto-save after each `send()` call
    pub fn with_auto_save(mut self, enabled: bool) -> Self {
        self.auto_save = enabled;
        self
    }

    /// Set artifact retention limits for this session.
    pub fn with_artifact_store_limits(mut self, limits: crate::tools::ArtifactStoreLimits) -> Self {
        self.artifact_store_limits = Some(limits);
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

    /// Provide a concrete [`BashSandbox`] implementation for this session.
    ///
    /// When set, `bash` tool commands are routed through the given sandbox
    /// instead of `std::process::Command`. The host application is responsible
    /// for constructing and lifecycle-managing the sandbox.
    ///
    /// [`BashSandbox`]: crate::sandbox::BashSandbox
    pub fn with_sandbox_handle(mut self, handle: Arc<dyn crate::sandbox::BashSandbox>) -> Self {
        self.sandbox_handle = Some(handle);
        self
    }

    /// Provide a workspace backend for this session.
    ///
    /// Built-in tools keep their stable names and schemas, while their backing
    /// implementation can target a DFS, browser workspace, remote runner, or
    /// any other host-provided backend.
    pub fn with_workspace_backend(
        mut self,
        services: Arc<crate::workspace::WorkspaceServices>,
    ) -> Self {
        self.workspace_services = Some(services);
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

    /// Override the maximum number of tool execution rounds for this session.
    ///
    /// Useful when binding a markdown-defined subagent to a session —
    /// pass the agent definition's `max_steps` value here to enforce its step budget.
    pub fn with_max_tool_rounds(mut self, rounds: usize) -> Self {
        self.max_tool_rounds = Some(rounds);
        self
    }

    /// Override the maximum number of sibling parallel branches for this session.
    pub fn with_max_parallel_tasks(mut self, tasks: usize) -> Self {
        self.max_parallel_tasks = Some(tasks.max(1));
        self
    }

    /// Override automatic subagent delegation for this session.
    pub fn with_auto_delegation(mut self, config: crate::config::AutoDelegationConfig) -> Self {
        self.auto_delegation = Some(config);
        self
    }

    /// Enable or disable automatic subagent delegation for this session.
    pub fn with_auto_delegation_enabled(mut self, enabled: bool) -> Self {
        let mut config = self.auto_delegation.take().unwrap_or_default();
        config.enabled = enabled;
        self.auto_delegation = Some(config);
        self
    }

    /// Enable or disable model-visible manual child-agent tools for this session.
    ///
    /// When false, `task` and `parallel_task` are not registered in the session
    /// tool surface. Worker agents remain registered for introspection and hosts
    /// that manage them directly. This is for cost control or debugging; it is
    /// not a security sandbox for the parent agent.
    pub fn with_manual_delegation_enabled(mut self, enabled: bool) -> Self {
        if let Some(config) = &mut self.auto_delegation {
            config.allow_manual_delegation = enabled;
        }
        self.manual_delegation_enabled = Some(enabled);
        self
    }

    /// Globally enable or disable automatic parallel child-agent fan-out.
    ///
    /// Manual `parallel_task` calls remain available when this is false.
    pub fn with_auto_parallel_delegation(mut self, enabled: bool) -> Self {
        if let Some(config) = &mut self.auto_delegation {
            config.auto_parallel = enabled;
        }
        self.auto_parallel_delegation = Some(enabled);
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

    /// Replace the built-in hook engine with an external hook executor.
    ///
    /// Use this to attach an AHP harness server (or any custom `HookExecutor`)
    /// to the session. All lifecycle events will be forwarded to the executor
    /// instead of the in-process `HookEngine`.
    pub fn with_hook_executor(mut self, executor: Arc<dyn crate::hooks::HookExecutor>) -> Self {
        self.hook_executor = Some(executor);
        self
    }
}
