//! Session construction runtime.
//!
//! This module owns the harness assembly path for a workspace-bound session:
//! capabilities, runtime wiring, memory, persistence, and live control state.

use super::{safe_canonicalize, Agent, AgentSession, SessionOptions};
use crate::agent::AgentConfig;
use crate::commands::CommandRegistry;
use crate::error::Result;
use crate::llm::LlmClient;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

use super::capabilities::{
    build_session_capabilities, register_skill_capability, SessionCapabilityInput,
};
use super::session_config::{
    resolve_auto_delegation_config, resolve_session_memory, resolve_session_store,
};
use super::session_runtime::{build_session_runtime, SessionRuntimeInput};

pub(super) fn prepare_session_options(agent: &Agent, opts: SessionOptions) -> SessionOptions {
    let mut opts = merge_mcp_managers(agent, opts);
    if opts.session_id.is_none() {
        // Use the host-provided ID generator if one was supplied via
        // SessionOptions — this is the entry point that enables
        // deterministic-replay tooling to pin session ids.
        let env = opts
            .host_env
            .clone()
            .unwrap_or_else(|| Arc::clone(&agent.config.host_env));
        opts.session_id = Some(env.next_id());
    }
    opts
}

fn merge_mcp_managers(agent: &Agent, opts: SessionOptions) -> SessionOptions {
    match (&agent.global_mcp, &opts.mcp_manager) {
        (Some(global), Some(session)) => {
            merge_session_mcp_into_global(global, session);
            SessionOptions {
                mcp_manager: Some(Arc::clone(global)),
                ..opts
            }
        }
        (Some(global), None) => SessionOptions {
            mcp_manager: Some(Arc::clone(global)),
            ..opts
        },
        _ => opts,
    }
}

fn merge_session_mcp_into_global(
    global: &Arc<crate::mcp::manager::McpManager>,
    session: &Arc<crate::mcp::manager::McpManager>,
) {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            let global = Arc::clone(global);
            let session = Arc::clone(session);
            tokio::task::block_in_place(|| {
                handle.block_on(async move {
                    for config in session.all_configs().await {
                        let name = config.name.clone();
                        global.register_server(config).await;
                        if let Err(e) = global.connect(&name).await {
                            tracing::warn!(
                                server = %name,
                                error = %e,
                                "Failed to connect session-level MCP server - skipping"
                            );
                        }
                    }
                })
            });
        }
        Err(_) => {
            tracing::warn!(
                "No async runtime available to merge session-level MCP servers \
                 into global manager - session MCP servers will not be available"
            );
        }
    }
}

pub(super) fn build_agent_session(
    agent: &Agent,
    workspace: String,
    llm_client: Arc<dyn LlmClient>,
    opts: &SessionOptions,
) -> Result<AgentSession> {
    let canonical = safe_canonicalize(Path::new(&workspace));

    let capabilities = build_session_capabilities(SessionCapabilityInput {
        code_config: &agent.code_config,
        base_config: &agent.config,
        workspace: &canonical,
        llm_client: Arc::clone(&llm_client),
        opts,
        global_mcp: agent.global_mcp.as_ref(),
        cached_global_mcp_tools: agent
            .global_mcp_tools
            .lock()
            .expect("global_mcp_tools lock poisoned")
            .clone(),
    });
    let tool_executor = capabilities.tool_executor;
    let trace_sink = capabilities.trace_sink;
    let agent_registry = capabilities.agent_registry;
    let tool_defs = capabilities.tool_defs;
    let context_providers = capabilities.context_providers;
    let effective_registry = capabilities.skill_registry;
    let subagent_tasks = capabilities.subagent_tasks;

    let prompt_slots = opts
        .prompt_slots
        .clone()
        .unwrap_or_else(|| agent.config.prompt_slots.clone());

    let session_id = opts
        .session_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let runtime = build_session_runtime(SessionRuntimeInput {
        code_config: &agent.code_config,
        workspace: &canonical,
        session_id: &session_id,
        opts,
        tool_executor: Arc::clone(&tool_executor),
    });

    let resolved_memory = resolve_session_memory(opts);
    let memory = resolved_memory.memory;
    let init_warning = resolved_memory.init_warning;

    let base = agent.config.clone();
    let auto_delegation = resolve_auto_delegation_config(&agent.code_config, opts);
    let config = AgentConfig {
        prompt_slots,
        tools: tool_defs,
        security_provider: opts.security_provider.clone(),
        permission_checker: opts.permission_checker.clone(),
        permission_policy: opts.permission_policy.clone(),
        confirmation_manager: runtime.confirmation_manager.clone(),
        confirmation_policy: opts.confirmation_policy.clone(),
        queue_config: opts.queue_config.clone(),
        context_providers,
        planning_mode: opts.planning_mode,
        goal_tracking: opts.goal_tracking,
        skill_registry: Some(Arc::clone(&effective_registry)),
        max_parse_retries: opts.max_parse_retries.unwrap_or(base.max_parse_retries),
        tool_timeout_ms: opts.tool_timeout_ms.or(base.tool_timeout_ms),
        circuit_breaker_threshold: opts
            .circuit_breaker_threshold
            .unwrap_or(base.circuit_breaker_threshold),
        auto_compact: opts.auto_compact,
        auto_compact_threshold: opts
            .auto_compact_threshold
            .unwrap_or(crate::store::DEFAULT_AUTO_COMPACT_THRESHOLD),
        max_context_tokens: base.max_context_tokens,
        memory: memory.clone(),
        continuation_enabled: opts
            .continuation_enabled
            .unwrap_or(base.continuation_enabled),
        max_continuation_turns: opts
            .max_continuation_turns
            .unwrap_or(base.max_continuation_turns),
        max_tool_rounds: opts.max_tool_rounds.unwrap_or(base.max_tool_rounds),
        max_parallel_tasks: opts
            .max_parallel_tasks
            .unwrap_or(base.max_parallel_tasks)
            .max(1),
        auto_delegation,
        agent_registry: Some(Arc::clone(&agent_registry)),
        max_execution_time_ms: opts.max_execution_time_ms.or(base.max_execution_time_ms),
        budget_guard: opts.budget_guard.clone().or(base.budget_guard.clone()),
        host_env: opts
            .host_env
            .clone()
            .unwrap_or_else(|| Arc::clone(&base.host_env)),
        ..base
    };

    // Register Skill after config is built so it can spawn child loops with
    // the same harness configuration while applying skill-local restrictions.
    register_skill_capability(
        Arc::clone(&tool_executor),
        Arc::clone(&llm_client),
        Arc::clone(&effective_registry),
        config.clone(),
    );

    let command_queue = runtime.command_queue;
    let tool_context = runtime.tool_context;
    let session_store = resolve_session_store(&agent.code_config, opts);
    let command_registry = CommandRegistry::new();

    let closed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let session_cancel = tokio_util::sync::CancellationToken::new();
    let cancel_token = Arc::new(tokio::sync::Mutex::new(None));
    let current_run_id = Arc::new(tokio::sync::Mutex::new(None));
    let run_store = Arc::new({
        let limits = opts.retention_limits;
        crate::run::InMemoryRunStore::with_retention(
            limits.and_then(|l| l.max_runs_retained),
            limits.and_then(|l| l.max_events_per_run),
        )
    });

    let close_handle = Arc::new(super::session_close::SessionCloseHandle {
        session_id: session_id.clone(),
        closed: Arc::clone(&closed),
        session_cancel: session_cancel.clone(),
        cancel_token: Arc::clone(&cancel_token),
        current_run_id: Arc::clone(&current_run_id),
        run_store: Arc::clone(&run_store),
        subagent_tasks: Arc::clone(&subagent_tasks),
        confirmation_manager: config.confirmation_manager.clone(),
        hook_executor: opts.hook_executor.clone(),
    });

    super::agent_sessions::register_session(agent, &close_handle);

    let session = AgentSession {
        llm_client,
        tool_executor,
        tool_context,
        memory: config.memory.clone(),
        config,
        workspace: canonical,
        session_id,
        history: Arc::new(RwLock::new(Vec::new())),
        command_queue,
        session_store,
        auto_save: opts.auto_save,
        hook_engine: Arc::new(crate::hooks::HookEngine::new()),
        ahp_executor: opts.hook_executor.clone(),
        init_warning,
        command_registry: std::sync::Mutex::new(command_registry),
        model_name: opts
            .model
            .clone()
            .or_else(|| agent.code_config.default_model.clone())
            .unwrap_or_else(|| "unknown".to_string()),
        mcp_manager: opts
            .mcp_manager
            .clone()
            .or_else(|| agent.global_mcp.clone())
            .unwrap_or_else(|| Arc::new(crate::mcp::manager::McpManager::new())),
        agent_registry,
        cancel_token,
        current_run_id,
        run_store,
        subagent_tasks,
        active_tools: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        trace_sink,
        verification_reports: Arc::new(RwLock::new(Vec::new())),
        closed,
        session_cancel,
        close_handle,
        tenant_id: opts.tenant_id.clone(),
        principal: opts.principal.clone(),
        agent_template_id: opts.agent_template_id.clone(),
        correlation_id: opts.correlation_id.clone(),
        runtime_budget_guard: std::sync::Mutex::new(None),
    };
    Ok(session)
}
