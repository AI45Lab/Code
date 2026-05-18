//! Session runtime wiring.
//!
//! Capabilities describe what the agent can do. This module wires the per-session
//! runtime channels and adapters that control how those capabilities execute.

use super::SessionOptions;
use crate::agent::AgentEvent;
use crate::config::CodeConfig;
use crate::hitl::ConfirmationProvider;
use crate::session_lane_queue::SessionLaneQueue;
use crate::tools::{ToolContext, ToolExecutor};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::broadcast;

pub(super) struct SessionRuntimeInput<'a> {
    pub(super) code_config: &'a CodeConfig,
    pub(super) workspace: &'a Path,
    pub(super) session_id: &'a str,
    pub(super) opts: &'a SessionOptions,
    pub(super) tool_executor: Arc<ToolExecutor>,
}

pub(super) struct SessionRuntime {
    pub(super) confirmation_manager: Option<Arc<dyn ConfirmationProvider>>,
    pub(super) command_queue: Option<Arc<SessionLaneQueue>>,
    pub(super) tool_context: ToolContext,
}

pub(super) fn build_session_runtime(input: SessionRuntimeInput<'_>) -> SessionRuntime {
    let (agent_event_tx, _) = broadcast::channel::<AgentEvent>(2048);

    let confirmation_manager = build_confirmation_manager(input.opts, agent_event_tx.clone());
    let command_queue = build_command_queue(input.opts, input.session_id, agent_event_tx.clone());
    let tool_context = build_tool_context(
        input.code_config,
        input.workspace,
        input.opts,
        Arc::clone(&input.tool_executor),
        agent_event_tx,
    );

    SessionRuntime {
        confirmation_manager,
        command_queue,
        tool_context,
    }
}

fn build_confirmation_manager(
    opts: &SessionOptions,
    agent_event_tx: broadcast::Sender<AgentEvent>,
) -> Option<Arc<dyn ConfirmationProvider>> {
    if opts.confirmation_manager.is_some() {
        opts.confirmation_manager.clone()
    } else if let Some(policy) = &opts.confirmation_policy {
        let manager = Arc::new(crate::hitl::ConfirmationManager::new(
            policy.clone(),
            agent_event_tx,
        ));
        Some(manager as Arc<dyn ConfirmationProvider>)
    } else {
        None
    }
}

fn build_command_queue(
    opts: &SessionOptions,
    session_id: &str,
    agent_event_tx: broadcast::Sender<AgentEvent>,
) -> Option<Arc<SessionLaneQueue>> {
    let queue_config = opts.queue_config.as_ref()?;

    let rt = tokio::runtime::Handle::try_current();
    match rt {
        Ok(handle) => {
            let queue = tokio::task::block_in_place(|| {
                handle.block_on(SessionLaneQueue::new(
                    session_id,
                    queue_config.clone(),
                    agent_event_tx,
                ))
            });
            match queue {
                Ok(queue) => {
                    let queue = Arc::new(queue);
                    let queue_to_start = Arc::clone(&queue);
                    tokio::task::block_in_place(|| {
                        handle.block_on(async { queue_to_start.start().await.ok() })
                    });
                    Some(queue)
                }
                Err(e) => {
                    tracing::warn!("Failed to create session lane queue: {}", e);
                    None
                }
            }
        }
        Err(_) => {
            tracing::warn!("No async runtime available for queue creation - queue disabled");
            None
        }
    }
}

fn build_tool_context(
    code_config: &CodeConfig,
    _workspace: &Path,
    opts: &SessionOptions,
    tool_executor: Arc<ToolExecutor>,
    agent_event_tx: broadcast::Sender<AgentEvent>,
) -> ToolContext {
    let mut tool_context = tool_executor.registry().context();
    if let Some(ref search_config) = code_config.search {
        tool_context = tool_context.with_search_config(search_config.clone());
    }
    tool_context = tool_context.with_agent_event_tx(agent_event_tx);

    if let Some(handle) = opts.sandbox_handle.clone() {
        tool_executor.registry().set_sandbox(Arc::clone(&handle));
        tool_context = tool_context.with_sandbox(handle);
    }

    tool_context
}
