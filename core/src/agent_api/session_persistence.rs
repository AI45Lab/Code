//! Persisted session runtime contract.
//!
//! This module owns how an in-memory session becomes `SessionData` and how a
//! saved `SessionData` rehydrates runtime options and non-message artifacts.

use super::{AgentSession, SessionOptions};
use crate::agent::{AgentConfig, AgentResult};
use crate::error::{read_or_recover, write_or_recover, CodeError, Result};
use crate::llm::Message;
use crate::store::{LlmConfigData, SessionData, SessionStore};
use crate::tools::ToolExecutor;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

#[derive(Clone)]
pub(super) struct SessionPersistenceContext {
    session_store: Option<Arc<dyn SessionStore>>,
    session_id: String,
    workspace: PathBuf,
    config: AgentConfig,
    model_name: String,
    tool_executor: Arc<ToolExecutor>,
    trace_sink: crate::trace::InMemoryTraceSink,
    run_store: Arc<crate::run::InMemoryRunStore>,
    history: Arc<RwLock<Vec<Message>>>,
    verification_reports: Arc<RwLock<Vec<crate::verification::VerificationReport>>>,
    subagent_tasks: Arc<crate::subagent_task_tracker::InMemorySubagentTaskTracker>,
    tenant_id: Option<String>,
    principal: Option<String>,
    agent_template_id: Option<String>,
    correlation_id: Option<String>,
    auto_save: bool,
}

impl SessionPersistenceContext {
    pub(super) fn from_session(session: &AgentSession) -> Self {
        Self {
            session_store: session.session_store.clone(),
            session_id: session.session_id.clone(),
            workspace: session.workspace.clone(),
            config: session.config.clone(),
            model_name: session.model_name.clone(),
            tool_executor: Arc::clone(&session.tool_executor),
            trace_sink: session.trace_sink.clone(),
            run_store: Arc::clone(&session.run_store),
            history: Arc::clone(&session.history),
            verification_reports: Arc::clone(&session.verification_reports),
            subagent_tasks: Arc::clone(&session.subagent_tasks),
            tenant_id: session.tenant_id.clone(),
            principal: session.principal.clone(),
            agent_template_id: session.agent_template_id.clone(),
            correlation_id: session.correlation_id.clone(),
            auto_save: session.auto_save,
        }
    }

    pub(super) fn record_result(&self, result: &AgentResult) {
        *write_or_recover(&self.history) = result.messages.clone();
        if !result.verification_reports.is_empty() {
            write_or_recover(&self.verification_reports)
                .extend(result.verification_reports.clone());
        }
    }

    pub(super) async fn save(&self) -> Result<()> {
        let store = match &self.session_store {
            Some(store) => store,
            None => return Ok(()),
        };

        let history = read_or_recover(&self.history).clone();
        let verification_reports = read_or_recover(&self.verification_reports).clone();
        let data = build_session_data_snapshot(SessionDataSnapshotInput {
            session_id: &self.session_id,
            workspace: &self.workspace,
            config: &self.config,
            model_name: &self.model_name,
            history,
            tenant_id: self.tenant_id.as_deref(),
            principal: self.principal.as_deref(),
            agent_template_id: self.agent_template_id.as_deref(),
            correlation_id: self.correlation_id.as_deref(),
        })
        .await;

        store.save(&data).await?;
        store
            .save_artifacts(&self.session_id, &self.tool_executor.artifact_store())
            .await?;
        store
            .save_trace_events(&self.session_id, &self.trace_sink.events())
            .await?;
        store
            .save_run_records(&self.session_id, &self.run_store.records().await)
            .await?;
        store
            .save_verification_reports(&self.session_id, &verification_reports)
            .await?;
        store
            .save_subagent_tasks(&self.session_id, &self.subagent_tasks.list().await)
            .await?;
        tracing::debug!("Session {} saved", self.session_id);
        Ok(())
    }

    pub(super) async fn auto_save_if_enabled(&self) {
        if self.auto_save {
            if let Err(e) = self.save().await {
                tracing::warn!("Auto-save failed for session {}: {}", self.session_id, e);
            }
        }
    }

    /// Delete the loop checkpoint for `run_id` once the run has reached a
    /// terminal state in-process. The checkpoint exists only to survive a
    /// process crash; once the run returns (completed / failed / cancelled)
    /// it is dead weight. No-op when no store is configured. Errors are
    /// warn-logged — a failed cleanup must never mask the run's result.
    pub(super) async fn clear_loop_checkpoint(&self, run_id: &str) {
        let Some(store) = &self.session_store else {
            return;
        };
        if let Err(e) = store.delete_loop_checkpoint(run_id).await {
            tracing::warn!(
                run_id = %run_id,
                session_id = %self.session_id,
                "Failed to delete loop checkpoint on run completion: {}",
                e
            );
        }
    }
}

pub(super) fn load_session_data(
    store: &Arc<dyn SessionStore>,
    session_id: &str,
) -> Result<SessionData> {
    let data = match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(store.load(session_id)))
            .map_err(|e| {
                CodeError::Session(format!("Failed to load session {}: {}", session_id, e))
            })?,
        Err(_) => {
            return Err(CodeError::Session(
                "No async runtime available for session resume".to_string(),
            ))
        }
    };

    data.ok_or_else(|| CodeError::Session(format!("Session not found: {}", session_id)))
}

pub(super) fn apply_persisted_runtime_options(
    mut opts: SessionOptions,
    data: &SessionData,
) -> SessionOptions {
    opts.session_id = Some(data.id.clone());

    if opts.model.is_none() {
        opts.model = persisted_model_ref(data);
    }
    if opts.queue_config.is_none() {
        opts.queue_config = data.config.queue_config.clone();
    }
    if opts.confirmation_manager.is_none() && opts.confirmation_policy.is_none() {
        opts.confirmation_policy = data.config.confirmation_policy.clone();
    }
    if opts.permission_checker.is_none() && opts.permission_policy.is_none() {
        if let Some(policy) = data.config.permission_policy.clone() {
            opts = opts.with_permission_policy(policy);
        }
    }
    if opts.enforce_active_skill_tool_restrictions.is_none() {
        opts.enforce_active_skill_tool_restrictions =
            Some(data.config.enforce_active_skill_tool_restrictions);
    }
    if opts.max_parallel_tasks.is_none() {
        opts.max_parallel_tasks = data.config.max_parallel_tasks;
    }
    if opts.auto_delegation.is_none() {
        opts.auto_delegation = data.config.auto_delegation.clone();
    }

    // Identity labels: caller-supplied values take precedence (the resume
    // caller may want to relabel for a new tenant/principal). Otherwise
    // restore from the persisted snapshot.
    if opts.tenant_id.is_none() {
        opts.tenant_id = data.tenant_id.clone();
    }
    if opts.principal.is_none() {
        opts.principal = data.principal.clone();
    }
    if opts.agent_template_id.is_none() {
        opts.agent_template_id = data.agent_template_id.clone();
    }
    if opts.correlation_id.is_none() {
        opts.correlation_id = data.correlation_id.clone();
    }

    opts
}

pub(super) fn restore_persisted_session_state(
    session: &AgentSession,
    store: &Arc<dyn SessionStore>,
    data: SessionData,
) -> Result<()> {
    let session_id = data.id.clone();
    *write_or_recover(&session.history) = data.messages;

    if let Some(artifacts) = load_artifacts(store, &session_id)? {
        let target_store = session.tool_executor.artifact_store();
        for artifact in artifacts.artifacts() {
            target_store.put(artifact);
        }
    }

    if let Some(events) = load_trace_events(store, &session_id)? {
        session.trace_sink.replace_events(events);
    }

    if let Some(records) = load_run_records(store, &session_id)? {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            tokio::task::block_in_place(|| {
                handle.block_on(session.run_store.replace_records(records))
            });
        }
    }

    if let Some(reports) = load_verification_reports(store, &session_id)? {
        *write_or_recover(&session.verification_reports) = reports;
    }

    if let Some(tasks) = load_subagent_tasks(store, &session_id)? {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            tokio::task::block_in_place(|| {
                handle.block_on(session.subagent_tasks.replace_snapshots(tasks))
            });
        }
    }

    Ok(())
}

struct SessionDataSnapshotInput<'a> {
    session_id: &'a str,
    workspace: &'a Path,
    config: &'a AgentConfig,
    model_name: &'a str,
    history: Vec<Message>,
    tenant_id: Option<&'a str>,
    principal: Option<&'a str>,
    agent_template_id: Option<&'a str>,
    correlation_id: Option<&'a str>,
}

async fn build_session_data_snapshot(input: SessionDataSnapshotInput<'_>) -> SessionData {
    let confirmation_policy = match &input.config.confirmation_manager {
        Some(manager) => Some(manager.policy().await),
        None => input.config.confirmation_policy.clone(),
    };
    let model_name = persisted_model_name(input.model_name);
    let now = chrono::Utc::now().timestamp();

    SessionData {
        id: input.session_id.to_string(),
        config: crate::store::SessionConfig {
            name: String::new(),
            workspace: input.workspace.display().to_string(),
            system_prompt: Some(input.config.prompt_slots.build()),
            max_context_length: input.config.max_context_tokens.min(u32::MAX as usize) as u32,
            auto_compact: input.config.auto_compact,
            auto_compact_threshold: input.config.auto_compact_threshold,
            storage_type: crate::config::StorageBackend::File,
            queue_config: input.config.queue_config.clone(),
            confirmation_policy,
            permission_policy: input.config.permission_policy.clone(),
            enforce_active_skill_tool_restrictions: input
                .config
                .enforce_active_skill_tool_restrictions,
            max_parallel_tasks: Some(input.config.max_parallel_tasks),
            auto_delegation: Some(input.config.auto_delegation.clone()),
            parent_id: None,
            security_config: None,
            hook_engine: None,
            planning_mode: input.config.planning_mode,
            goal_tracking: input.config.goal_tracking,
        },
        state: crate::store::SessionState::Active,
        messages: input.history,
        context_usage: crate::store::ContextUsage::default(),
        total_usage: crate::llm::TokenUsage::default(),
        total_cost: 0.0,
        model_name,
        cost_records: Vec::new(),
        tool_names: SessionData::tool_names_from_definitions(&input.config.tools),
        thinking_enabled: false,
        thinking_budget: None,
        created_at: now,
        updated_at: now,
        llm_config: model_config_data(input.model_name),
        tasks: Vec::new(),
        parent_id: None,
        tenant_id: input.tenant_id.map(str::to_string),
        principal: input.principal.map(str::to_string),
        agent_template_id: input.agent_template_id.map(str::to_string),
        correlation_id: input.correlation_id.map(str::to_string),
    }
}

fn persisted_model_ref(data: &SessionData) -> Option<String> {
    if let Some(llm_config) = &data.llm_config {
        return Some(format!("{}/{}", llm_config.provider, llm_config.model));
    }
    data.model_name
        .as_ref()
        .filter(|model_name| model_name.contains('/'))
        .cloned()
}

fn persisted_model_name(model_name: &str) -> Option<String> {
    if model_name.is_empty() || model_name == "unknown" {
        None
    } else {
        Some(model_name.to_string())
    }
}

fn model_config_data(model_name: &str) -> Option<LlmConfigData> {
    let (provider, model) = model_name.split_once('/')?;
    Some(LlmConfigData {
        provider: provider.to_string(),
        model: model.to_string(),
        api_key: None,
        base_url: None,
    })
}

fn load_artifacts(
    store: &Arc<dyn SessionStore>,
    session_id: &str,
) -> Result<Option<crate::tools::ArtifactStore>> {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            tokio::task::block_in_place(|| handle.block_on(store.load_artifacts(session_id)))
                .map_err(|e| {
                    CodeError::Session(format!(
                        "Failed to load artifacts for session {}: {}",
                        session_id, e
                    ))
                })
        }
        Err(_) => Ok(None),
    }
}

fn load_trace_events(
    store: &Arc<dyn SessionStore>,
    session_id: &str,
) -> Result<Option<Vec<crate::trace::TraceEvent>>> {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            tokio::task::block_in_place(|| handle.block_on(store.load_trace_events(session_id)))
                .map_err(|e| {
                    CodeError::Session(format!(
                        "Failed to load trace events for session {}: {}",
                        session_id, e
                    ))
                })
        }
        Err(_) => Ok(None),
    }
}

fn load_run_records(
    store: &Arc<dyn SessionStore>,
    session_id: &str,
) -> Result<Option<Vec<crate::run::RunRecord>>> {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            tokio::task::block_in_place(|| handle.block_on(store.load_run_records(session_id)))
                .map_err(|e| {
                    CodeError::Session(format!(
                        "Failed to load run records for session {}: {}",
                        session_id, e
                    ))
                })
        }
        Err(_) => Ok(None),
    }
}

fn load_subagent_tasks(
    store: &Arc<dyn SessionStore>,
    session_id: &str,
) -> Result<Option<Vec<crate::subagent_task_tracker::SubagentTaskSnapshot>>> {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            tokio::task::block_in_place(|| handle.block_on(store.load_subagent_tasks(session_id)))
                .map_err(|e| {
                    CodeError::Session(format!(
                        "Failed to load subagent tasks for session {}: {}",
                        session_id, e
                    ))
                })
        }
        Err(_) => Ok(None),
    }
}

fn load_verification_reports(
    store: &Arc<dyn SessionStore>,
    session_id: &str,
) -> Result<Option<Vec<crate::verification::VerificationReport>>> {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| {
            handle.block_on(store.load_verification_reports(session_id))
        })
        .map_err(|e| {
            CodeError::Session(format!(
                "Failed to load verification reports for session {}: {}",
                session_id, e
            ))
        }),
        Err(_) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{ContextUsage, SessionConfig, SessionState};

    fn persisted_data(model_name: Option<&str>, llm: Option<(&str, &str)>) -> SessionData {
        SessionData {
            id: "session-1".to_string(),
            config: SessionConfig::default(),
            state: SessionState::Active,
            messages: Vec::new(),
            context_usage: ContextUsage::default(),
            total_usage: crate::llm::TokenUsage::default(),
            total_cost: 0.0,
            model_name: model_name.map(ToOwned::to_owned),
            cost_records: Vec::new(),
            tool_names: Vec::new(),
            thinking_enabled: false,
            thinking_budget: None,
            created_at: 0,
            updated_at: 0,
            llm_config: llm.map(|(provider, model)| LlmConfigData {
                provider: provider.to_string(),
                model: model.to_string(),
                api_key: None,
                base_url: None,
            }),
            tasks: Vec::new(),
            parent_id: None,
            tenant_id: None,
            principal: None,
            agent_template_id: None,
            correlation_id: None,
        }
    }

    #[test]
    fn persisted_runtime_options_prefer_llm_config() {
        let data = persisted_data(Some("anthropic/old"), Some(("openai", "gpt-4o")));
        let opts = apply_persisted_runtime_options(SessionOptions::new(), &data);
        assert_eq!(opts.session_id.as_deref(), Some("session-1"));
        assert_eq!(opts.model.as_deref(), Some("openai/gpt-4o"));
    }

    #[test]
    fn persisted_runtime_options_fall_back_to_model_name() {
        let data = persisted_data(Some("openai/gpt-4o"), None);
        let opts = apply_persisted_runtime_options(SessionOptions::new(), &data);
        assert_eq!(opts.model.as_deref(), Some("openai/gpt-4o"));
    }

    #[test]
    fn model_config_never_persists_secret_material() {
        let data = model_config_data("openai/gpt-4o").expect("model config");
        assert_eq!(data.provider, "openai");
        assert_eq!(data.model, "gpt-4o");
        assert!(data.api_key.is_none());
        assert!(data.base_url.is_none());
    }
}
