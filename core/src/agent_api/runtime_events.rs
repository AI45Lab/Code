//! Runtime event tracking for agent runs.
//!
//! This module owns the contract from `AgentEvent` to run records, hook
//! forwarding, and active-tool state. Run orchestration can start workers without
//! knowing which events mutate tracking state.

use super::{session_clock, AgentSession};
use crate::agent::AgentEvent;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

#[derive(Debug, Clone)]
pub(super) struct ActiveToolState {
    pub(super) tool_name: String,
    pub(super) started_at_ms: u64,
}

type ActiveToolMap = Arc<tokio::sync::RwLock<HashMap<String, ActiveToolState>>>;

pub(super) async fn active_tool_snapshots(
    active_tools: &ActiveToolMap,
) -> Vec<crate::run::ActiveToolSnapshot> {
    let mut snapshots = active_tools
        .read()
        .await
        .iter()
        .map(|(id, tool)| crate::run::ActiveToolSnapshot {
            id: id.clone(),
            name: tool.tool_name.clone(),
            started_at_ms: tool.started_at_ms,
        })
        .collect::<Vec<_>>();
    snapshots.sort_by(|a, b| {
        a.started_at_ms
            .cmp(&b.started_at_ms)
            .then_with(|| a.id.cmp(&b.id))
    });
    snapshots
}

#[derive(Clone)]
pub(super) struct RuntimeEventSink {
    run_store: Arc<crate::run::InMemoryRunStore>,
    run_id: String,
    session_id: String,
    hook_executor: Option<Arc<dyn crate::hooks::HookExecutor>>,
    active_tools: ActiveToolMap,
    subagent_tasks: Arc<crate::subagent_task_tracker::InMemorySubagentTaskTracker>,
}

impl RuntimeEventSink {
    pub(super) fn from_session(session: &AgentSession, run_id: &str) -> Self {
        Self::new(
            Arc::clone(&session.run_store),
            run_id.to_string(),
            session.session_id.clone(),
            session.ahp_executor.clone(),
            Arc::clone(&session.active_tools),
            Arc::clone(&session.subagent_tasks),
        )
    }

    fn new(
        run_store: Arc<crate::run::InMemoryRunStore>,
        run_id: String,
        session_id: String,
        hook_executor: Option<Arc<dyn crate::hooks::HookExecutor>>,
        active_tools: ActiveToolMap,
        subagent_tasks: Arc<crate::subagent_task_tracker::InMemorySubagentTaskTracker>,
    ) -> Self {
        Self {
            run_store,
            run_id,
            session_id,
            hook_executor,
            active_tools,
            subagent_tasks,
        }
    }

    pub(super) fn spawn_collector(
        self,
        mut runtime_rx: mpsc::Receiver<AgentEvent>,
    ) -> JoinHandle<()> {
        tokio::spawn(async move {
            while let Some(event) = runtime_rx.recv().await {
                self.observe(&event).await;
            }
        })
    }

    pub(super) fn spawn_forwarder(
        self,
        mut runtime_rx: mpsc::Receiver<AgentEvent>,
        tx: mpsc::Sender<AgentEvent>,
    ) -> JoinHandle<()> {
        tokio::spawn(async move {
            while let Some(event) = runtime_rx.recv().await {
                self.observe(&event).await;
                let send_ok = tx.send(event.clone()).await.is_ok();
                if !send_ok {
                    // Receiver dropped or buffer full; preserve the existing stream contract
                    // by stopping instead of silently dropping later terminal events.
                    tracing::warn!("stream forwarder: receiver dropped, stopping event forward");
                    break;
                }
            }
        })
    }

    pub(super) async fn observe(&self, event: &AgentEvent) {
        let _ = self
            .run_store
            .record_event(&self.run_id, event.clone())
            .await;
        if let Some(executor) = &self.hook_executor {
            executor
                .record_agent_event(event, &self.run_id, &self.session_id)
                .await;
        }
        self.subagent_tasks.record_event(event).await;
        self.apply(event).await;
    }

    async fn apply(&self, event: &AgentEvent) {
        match event {
            AgentEvent::ToolStart { id, name } => {
                self.active_tools.write().await.insert(
                    id.clone(),
                    ActiveToolState {
                        tool_name: name.clone(),
                        started_at_ms: session_clock::now_ms(),
                    },
                );
            }
            AgentEvent::ToolEnd { id, .. }
            | AgentEvent::PermissionDenied { tool_id: id, .. }
            | AgentEvent::ConfirmationRequired { tool_id: id, .. }
            | AgentEvent::ConfirmationReceived { tool_id: id, .. }
            | AgentEvent::ConfirmationTimeout { tool_id: id, .. } => {
                self.active_tools.write().await.remove(id);
            }
            _ => {}
        }
    }
}

#[derive(Clone)]
pub(super) struct RunCleanupState {
    run_id: String,
    active_tools: ActiveToolMap,
    current_run_id: Arc<tokio::sync::Mutex<Option<String>>>,
    cancel_token: Arc<tokio::sync::Mutex<Option<tokio_util::sync::CancellationToken>>>,
}

impl RunCleanupState {
    pub(super) fn from_session(session: &AgentSession, run_id: &str) -> Self {
        Self {
            run_id: run_id.to_string(),
            active_tools: Arc::clone(&session.active_tools),
            current_run_id: Arc::clone(&session.current_run_id),
            cancel_token: Arc::clone(&session.cancel_token),
        }
    }

    pub(super) fn run_id(&self) -> &str {
        &self.run_id
    }

    pub(super) async fn set_cancel_token(&self, token: tokio_util::sync::CancellationToken) {
        *self.cancel_token.lock().await = Some(token);
    }

    pub(super) async fn clear_cancel_token(&self) {
        *self.cancel_token.lock().await = None;
    }

    pub(super) async fn finish(&self) {
        self.active_tools.write().await.clear();
        let mut current = self.current_run_id.lock().await;
        if current.as_deref() == Some(self.run_id.as_str()) {
            *current = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn active_tools() -> ActiveToolMap {
        Arc::new(tokio::sync::RwLock::new(HashMap::new()))
    }

    #[tokio::test]
    async fn tool_events_update_active_tool_state() {
        let run_store = Arc::new(crate::run::InMemoryRunStore::new());
        let run = run_store.create_run("session-1", "prompt").await;
        let active_tools = active_tools();
        let sink = RuntimeEventSink::new(
            Arc::clone(&run_store),
            run.id.clone(),
            "session-1".to_string(),
            None,
            Arc::clone(&active_tools),
            Arc::new(crate::subagent_task_tracker::InMemorySubagentTaskTracker::new()),
        );

        sink.observe(&AgentEvent::ToolStart {
            id: "tool-1".to_string(),
            name: "bash".to_string(),
        })
        .await;
        assert_eq!(active_tools.read().await.len(), 1);
        assert_eq!(
            active_tools
                .read()
                .await
                .get("tool-1")
                .map(|tool| tool.tool_name.as_str()),
            Some("bash")
        );

        sink.observe(&AgentEvent::ToolEnd {
            id: "tool-1".to_string(),
            name: "bash".to_string(),
            output: "ok".to_string(),
            exit_code: 0,
            metadata: None,
            error_kind: None,
        })
        .await;
        assert!(active_tools.read().await.is_empty());
    }

    #[tokio::test]
    async fn observe_records_events_on_run_store() {
        let run_store = Arc::new(crate::run::InMemoryRunStore::new());
        let run = run_store.create_run("session-1", "prompt").await;
        let sink = RuntimeEventSink::new(
            Arc::clone(&run_store),
            run.id.clone(),
            "session-1".to_string(),
            None,
            active_tools(),
            Arc::new(crate::subagent_task_tracker::InMemorySubagentTaskTracker::new()),
        );

        sink.observe(&AgentEvent::TextDelta {
            text: "hello".to_string(),
        })
        .await;

        let events = run_store.events(&run.id).await;
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].event, AgentEvent::TextDelta { .. }));
        assert_eq!(run_store.snapshot(&run.id).await.unwrap().event_count, 1);
    }
}
