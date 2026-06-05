//! Session capability wiring.
//!
//! This module owns the model-visible and context-visible capability set for a
//! session: built-in tools, delegated agent tools, MCP tools, workspace
//! instructions, and skills. The `Agent` facade passes configuration in and gets
//! back a ready-to-wire capability set.

use super::SessionOptions;
use crate::agent::AgentConfig;
use crate::config::CodeConfig;
use crate::context::{ContextItem, ContextProvider, ContextType, StaticContextProvider};
use crate::llm::{LlmClient, ToolDefinition};
use crate::mcp::{manager::McpManager, McpTool};
use crate::skills::SkillRegistry;
use crate::subagent::AgentRegistry;
use crate::tools::ToolExecutor;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const DISABLE_DELEGATION_TOOLS_ENV: &str = "A3S_CODE_DISABLE_DELEGATION_TOOLS";

pub(super) struct SessionCapabilityInput<'a> {
    pub(super) code_config: &'a CodeConfig,
    pub(super) base_config: &'a AgentConfig,
    pub(super) workspace: &'a Path,
    pub(super) llm_client: Arc<dyn LlmClient>,
    pub(super) opts: &'a SessionOptions,
    pub(super) global_mcp: Option<&'a Arc<McpManager>>,
    pub(super) cached_global_mcp_tools: Vec<(String, McpTool)>,
}

pub(super) struct SessionCapabilities {
    pub(super) tool_executor: Arc<ToolExecutor>,
    pub(super) trace_sink: crate::trace::InMemoryTraceSink,
    pub(super) tool_defs: Vec<ToolDefinition>,
    pub(super) context_providers: Vec<Arc<dyn ContextProvider>>,
    pub(super) skill_registry: Arc<SkillRegistry>,
    pub(super) agent_registry: Arc<AgentRegistry>,
    pub(super) subagent_tasks: Arc<crate::subagent_task_tracker::InMemorySubagentTaskTracker>,
}

pub(super) fn build_session_capabilities(input: SessionCapabilityInput<'_>) -> SessionCapabilities {
    let artifact_limits = input.opts.artifact_store_limits.unwrap_or_default();
    let workspace_services = input
        .opts
        .workspace_services
        .clone()
        .unwrap_or_else(|| crate::workspace::WorkspaceServices::local(input.workspace));
    let tool_executor = Arc::new(
        ToolExecutor::new_with_workspace_services_and_artifact_limits(
            input.workspace.display().to_string(),
            workspace_services,
            artifact_limits,
        ),
    );
    let trace_sink = match input.opts.retention_limits.and_then(|l| l.max_trace_events) {
        Some(cap) => crate::trace::InMemoryTraceSink::with_max_events(cap),
        None => crate::trace::InMemoryTraceSink::new(),
    };
    tool_executor.set_trace_sink(Arc::new(trace_sink.clone()));

    if let Some(ref search_config) = input.code_config.search {
        tool_executor
            .registry()
            .set_search_config(search_config.clone());
    }

    let subagent_tasks = Arc::new(
        match input
            .opts
            .retention_limits
            .and_then(|l| l.max_terminal_subagent_tasks)
        {
            Some(cap) => {
                crate::subagent_task_tracker::InMemorySubagentTaskTracker::with_max_terminal_tasks(
                    cap,
                )
            }
            None => crate::subagent_task_tracker::InMemorySubagentTaskTracker::new(),
        },
    );
    let agent_registry = register_task_capability(
        input.code_config,
        input.opts,
        input.workspace,
        Arc::clone(&input.llm_client),
        &tool_executor,
        Arc::clone(&subagent_tasks),
    );

    // Register generate_object tool (structured JSON output)
    crate::tools::register_generate_object(tool_executor.registry(), Arc::clone(&input.llm_client));

    register_mcp_capabilities(
        &tool_executor,
        input.opts,
        input.global_mcp,
        input.cached_global_mcp_tools,
    );

    let skill_registry =
        build_effective_skill_registry(input.base_config.skill_registry.as_deref(), input.opts);
    let context_providers = build_context_providers(input.opts, input.workspace, &skill_registry);
    let tool_defs = tool_executor.definitions();

    SessionCapabilities {
        tool_executor,
        trace_sink,
        tool_defs,
        context_providers,
        skill_registry,
        agent_registry,
        subagent_tasks,
    }
}

pub(super) fn register_skill_capability(
    tool_executor: Arc<ToolExecutor>,
    llm_client: Arc<dyn LlmClient>,
    skill_registry: Arc<SkillRegistry>,
    config: AgentConfig,
) {
    let registry = Arc::clone(tool_executor.registry());
    crate::tools::register_skill(&registry, llm_client, skill_registry, tool_executor, config);
}

pub(super) fn build_effective_skill_registry(
    agent_registry: Option<&SkillRegistry>,
    opts: &SessionOptions,
) -> Arc<SkillRegistry> {
    let base_registry = agent_registry
        .map(|r| r.fork())
        .unwrap_or_else(SkillRegistry::with_builtins);

    if let Some(ref registry) = opts.skill_registry {
        for skill in registry.all() {
            base_registry.register_unchecked(skill);
        }
    }

    for dir in &opts.skill_dirs {
        if let Err(e) = base_registry.load_from_dir(dir) {
            tracing::warn!(
                dir = %dir.display(),
                error = %e,
                "Failed to load session skill dir - skipping"
            );
        }
    }

    Arc::new(base_registry)
}

fn register_task_capability(
    code_config: &CodeConfig,
    opts: &SessionOptions,
    workspace: &Path,
    llm_client: Arc<dyn LlmClient>,
    tool_executor: &Arc<ToolExecutor>,
    subagent_tasks: Arc<crate::subagent_task_tracker::InMemorySubagentTaskTracker>,
) -> Arc<AgentRegistry> {
    use crate::child_run::ChildRunContext;
    use crate::subagent::load_agents_from_dir;
    use crate::tools::register_task_with_mcp;

    let registry = AgentRegistry::new();
    let disable_delegation_tools = disable_delegation_tools_from_env(
        std::env::var(DISABLE_DELEGATION_TOOLS_ENV).ok().as_deref(),
    );
    let built_in_agent_dirs = built_in_agent_dirs(workspace);
    for dir in code_config
        .agent_dirs
        .iter()
        .chain(built_in_agent_dirs.iter())
        .chain(opts.agent_dirs.iter())
    {
        for agent in load_agents_from_dir(dir) {
            registry.register(agent);
        }
    }
    for worker in &opts.worker_agents {
        registry.register_worker(worker.clone());
    }

    if disable_delegation_tools {
        return Arc::new(registry);
    }

    let parent_context = ChildRunContext {
        security_provider: opts.security_provider.clone(),
        hook_engine: None,
        skill_registry: opts.skill_registry.clone(),
        tool_timeout_ms: opts.tool_timeout_ms,
        max_parallel_tasks: opts.max_parallel_tasks.or(code_config.max_parallel_tasks),
        max_execution_time_ms: opts.max_execution_time_ms,
        circuit_breaker_threshold: opts.circuit_breaker_threshold,
        confirmation_manager: opts.confirmation_manager.clone(),
        workspace_services: opts.workspace_services.clone(),
    };

    let registry = Arc::new(registry);
    register_task_with_mcp(
        tool_executor.registry(),
        llm_client,
        Arc::clone(&registry),
        workspace.display().to_string(),
        opts.mcp_manager.clone(),
        Some(parent_context),
        Some(subagent_tasks),
    );
    registry
}

fn disable_delegation_tools_from_env(value: Option<&str>) -> bool {
    value
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn built_in_agent_dirs(workspace: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        let home = PathBuf::from(home);
        dirs.push(home.join(".claude").join("agents"));
        dirs.push(home.join(".a3s").join("agents"));
    }
    dirs.push(workspace.join(".claude").join("agents"));
    dirs.push(workspace.join(".a3s").join("agents"));
    dirs
}

fn register_mcp_capabilities(
    tool_executor: &Arc<ToolExecutor>,
    opts: &SessionOptions,
    global_mcp: Option<&Arc<McpManager>>,
    cached_global_mcp_tools: Vec<(String, McpTool)>,
) {
    let Some(ref mcp) = opts.mcp_manager else {
        return;
    };

    let all_tools = if is_global_mcp(mcp, global_mcp) {
        cached_global_mcp_tools
    } else {
        fetch_session_mcp_tools(mcp)
    };

    for (server_name, tools) in group_mcp_tools_by_server(all_tools) {
        for tool in crate::mcp::tools::create_mcp_tools(&server_name, tools, Arc::clone(mcp)) {
            tool_executor.register_dynamic_tool(tool);
        }
    }
}

fn is_global_mcp(mcp: &Arc<McpManager>, global_mcp: Option<&Arc<McpManager>>) -> bool {
    std::ptr::eq(
        Arc::as_ptr(mcp),
        global_mcp.map(Arc::as_ptr).unwrap_or(std::ptr::null()),
    )
}

fn fetch_session_mcp_tools(mcp: &Arc<McpManager>) -> Vec<(String, McpTool)> {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(mcp.get_all_tools())),
        Err(_) => {
            tracing::warn!(
                "No async runtime available for session-level MCP tools - MCP tools will not be registered"
            );
            vec![]
        }
    }
}

fn group_mcp_tools_by_server(all_tools: Vec<(String, McpTool)>) -> HashMap<String, Vec<McpTool>> {
    let mut by_server = HashMap::new();
    for (server, tool) in all_tools {
        by_server.entry(server).or_insert_with(Vec::new).push(tool);
    }
    by_server
}

#[cfg(test)]
mod tests {
    use super::disable_delegation_tools_from_env;

    #[test]
    fn disable_delegation_tools_accepts_common_truthy_values() {
        for value in ["1", "true", "TRUE", " yes ", "on", "On"] {
            assert!(disable_delegation_tools_from_env(Some(value)));
        }
    }

    #[test]
    fn disable_delegation_tools_rejects_empty_missing_and_falsey_values() {
        for value in ["", "0", "false", "no", "off", "enabled"] {
            assert!(!disable_delegation_tools_from_env(Some(value)));
        }
        assert!(!disable_delegation_tools_from_env(None));
    }
}

fn build_context_providers(
    opts: &SessionOptions,
    workspace: &Path,
    skill_registry: &SkillRegistry,
) -> Vec<Arc<dyn ContextProvider>> {
    let mut providers = opts.context_providers.clone();
    push_agents_md_context(&mut providers, workspace);
    push_skill_catalog_context(&mut providers, skill_registry);
    providers
}

fn push_agents_md_context(providers: &mut Vec<Arc<dyn ContextProvider>>, workspace: &Path) {
    let agents_md_path = workspace.join("AGENTS.md");
    if !agents_md_path.exists() || !agents_md_path.is_file() {
        return;
    }

    match std::fs::read_to_string(&agents_md_path) {
        Ok(content) if !content.trim().is_empty() => {
            tracing::info!(
                path = %agents_md_path.display(),
                "Auto-loaded AGENTS.md from workspace root"
            );
            let token_count = content.split_whitespace().count().max(1);
            let item = ContextItem::new(
                "agents_md",
                ContextType::Resource,
                format!("# Project Instructions (AGENTS.md)\n\n{}", content),
            )
            .with_source(format!("file://{}", agents_md_path.display()))
            .with_provenance("workspace_instructions")
            .with_priority(0.95)
            .with_trust(0.95)
            .with_freshness(1.0)
            .with_relevance(0.95)
            .with_token_count(token_count);

            providers.push(Arc::new(
                StaticContextProvider::new("agents_md").with_item(item),
            ));
        }
        Ok(_) => {
            tracing::debug!(
                path = %agents_md_path.display(),
                "AGENTS.md exists but is empty - skipping"
            );
        }
        Err(e) => {
            tracing::warn!(
                path = %agents_md_path.display(),
                error = %e,
                "Failed to read AGENTS.md - skipping"
            );
        }
    }
}

fn push_skill_catalog_context(
    providers: &mut Vec<Arc<dyn ContextProvider>>,
    skill_registry: &SkillRegistry,
) {
    let skill_prompt = skill_registry.to_system_prompt();
    if skill_prompt.is_empty() {
        return;
    }

    let item = ContextItem::new("skills_catalog", ContextType::Skill, skill_prompt)
        .with_source("a3s://skills/catalog")
        .with_provenance("skill_registry")
        .with_priority(0.85)
        .with_trust(0.9)
        .with_freshness(1.0)
        .with_relevance(1.0);
    providers.push(Arc::new(
        StaticContextProvider::new("skills_catalog").with_item(item),
    ));
}
