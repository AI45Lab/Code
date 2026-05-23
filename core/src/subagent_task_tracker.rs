//! In-memory tracker for delegated subagent tasks.
//!
//! Materializes a queryable view of subagent task lifecycle from the
//! `AgentEvent` stream. The event stream remains the authoritative record;
//! this module exists so callers can ask "what is task X doing right now?"
//! without scanning `run_events()`.

use crate::agent::AgentEvent;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentStatus {
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentProgressEntry {
    pub timestamp_ms: u64,
    pub status: String,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentTaskSnapshot {
    pub task_id: String,
    pub parent_session_id: String,
    pub child_session_id: String,
    pub agent: String,
    pub description: String,
    pub status: SubagentStatus,
    pub started_ms: u64,
    pub updated_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub success: Option<bool>,
    pub progress: Vec<SubagentProgressEntry>,
}

#[derive(Debug, Default)]
pub struct InMemorySubagentTaskTracker {
    tasks: RwLock<HashMap<String, SubagentTaskSnapshot>>,
}

impl InMemorySubagentTaskTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a single agent event to the tracker. Non-subagent events are ignored.
    pub async fn record_event(&self, event: &AgentEvent) {
        match event {
            AgentEvent::SubagentStart {
                task_id,
                session_id,
                parent_session_id,
                agent,
                description,
            } => {
                let now = now_ms();
                let mut tasks = self.tasks.write().await;
                tasks
                    .entry(task_id.clone())
                    .and_modify(|task| {
                        // Late start (e.g. background path) — keep the first-seen
                        // started_ms but refresh fields we now know.
                        task.parent_session_id = parent_session_id.clone();
                        task.child_session_id = session_id.clone();
                        task.agent = agent.clone();
                        task.description = description.clone();
                        task.updated_ms = now;
                    })
                    .or_insert_with(|| SubagentTaskSnapshot {
                        task_id: task_id.clone(),
                        parent_session_id: parent_session_id.clone(),
                        child_session_id: session_id.clone(),
                        agent: agent.clone(),
                        description: description.clone(),
                        status: SubagentStatus::Running,
                        started_ms: now,
                        updated_ms: now,
                        finished_ms: None,
                        output: None,
                        success: None,
                        progress: Vec::new(),
                    });
            }
            AgentEvent::SubagentProgress {
                task_id,
                session_id,
                status,
                metadata,
            } => {
                let now = now_ms();
                let mut tasks = self.tasks.write().await;
                let entry = tasks
                    .entry(task_id.clone())
                    .or_insert_with(|| SubagentTaskSnapshot {
                        task_id: task_id.clone(),
                        parent_session_id: String::new(),
                        child_session_id: session_id.clone(),
                        agent: String::new(),
                        description: String::new(),
                        status: SubagentStatus::Running,
                        started_ms: now,
                        updated_ms: now,
                        finished_ms: None,
                        output: None,
                        success: None,
                        progress: Vec::new(),
                    });
                entry.updated_ms = now;
                entry.progress.push(SubagentProgressEntry {
                    timestamp_ms: now,
                    status: status.clone(),
                    metadata: metadata.clone(),
                });
            }
            AgentEvent::SubagentEnd {
                task_id,
                session_id,
                agent,
                output,
                success,
            } => {
                let now = now_ms();
                let mut tasks = self.tasks.write().await;
                let entry = tasks
                    .entry(task_id.clone())
                    .or_insert_with(|| SubagentTaskSnapshot {
                        task_id: task_id.clone(),
                        parent_session_id: String::new(),
                        child_session_id: session_id.clone(),
                        agent: agent.clone(),
                        description: String::new(),
                        status: SubagentStatus::Running,
                        started_ms: now,
                        updated_ms: now,
                        finished_ms: None,
                        output: None,
                        success: None,
                        progress: Vec::new(),
                    });
                entry.status = if *success {
                    SubagentStatus::Completed
                } else {
                    SubagentStatus::Failed
                };
                entry.updated_ms = now;
                entry.finished_ms = Some(now);
                entry.output = Some(output.clone());
                entry.success = Some(*success);
            }
            _ => {}
        }
    }

    pub async fn get(&self, task_id: &str) -> Option<SubagentTaskSnapshot> {
        self.tasks.read().await.get(task_id).cloned()
    }

    pub async fn list(&self) -> Vec<SubagentTaskSnapshot> {
        let mut tasks = self
            .tasks
            .read()
            .await
            .values()
            .cloned()
            .collect::<Vec<_>>();
        tasks.sort_by_key(|task| task.started_ms);
        tasks
    }

    pub async fn list_pending(&self) -> Vec<SubagentTaskSnapshot> {
        self.list()
            .await
            .into_iter()
            .filter(|task| task.status == SubagentStatus::Running)
            .collect()
    }

    pub async fn list_for_parent(&self, parent_session_id: &str) -> Vec<SubagentTaskSnapshot> {
        self.list()
            .await
            .into_iter()
            .filter(|task| task.parent_session_id == parent_session_id)
            .collect()
    }
}

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn start_event(task_id: &str, parent: &str, child: &str) -> AgentEvent {
        AgentEvent::SubagentStart {
            task_id: task_id.to_string(),
            session_id: child.to_string(),
            parent_session_id: parent.to_string(),
            agent: "explore".to_string(),
            description: "find things".to_string(),
        }
    }

    fn progress_event(task_id: &str, child: &str, status: &str) -> AgentEvent {
        AgentEvent::SubagentProgress {
            task_id: task_id.to_string(),
            session_id: child.to_string(),
            status: status.to_string(),
            metadata: serde_json::json!({}),
        }
    }

    fn end_event(task_id: &str, child: &str, success: bool) -> AgentEvent {
        AgentEvent::SubagentEnd {
            task_id: task_id.to_string(),
            session_id: child.to_string(),
            agent: "explore".to_string(),
            output: "done".to_string(),
            success,
        }
    }

    #[tokio::test]
    async fn lifecycle_start_progress_end_transitions_status() {
        let tracker = InMemorySubagentTaskTracker::new();

        tracker
            .record_event(&start_event("task-1", "parent", "child"))
            .await;
        let snap = tracker.get("task-1").await.unwrap();
        assert_eq!(snap.status, SubagentStatus::Running);
        assert_eq!(snap.parent_session_id, "parent");
        assert_eq!(snap.child_session_id, "child");
        assert!(snap.finished_ms.is_none());

        tracker
            .record_event(&progress_event("task-1", "child", "tool_completed: bash"))
            .await;
        let snap = tracker.get("task-1").await.unwrap();
        assert_eq!(snap.status, SubagentStatus::Running);
        assert_eq!(snap.progress.len(), 1);

        tracker
            .record_event(&end_event("task-1", "child", true))
            .await;
        let snap = tracker.get("task-1").await.unwrap();
        assert_eq!(snap.status, SubagentStatus::Completed);
        assert_eq!(snap.success, Some(true));
        assert_eq!(snap.output.as_deref(), Some("done"));
        assert!(snap.finished_ms.is_some());
    }

    #[tokio::test]
    async fn failed_end_event_marks_status_failed() {
        let tracker = InMemorySubagentTaskTracker::new();
        tracker
            .record_event(&start_event("task-2", "parent", "child"))
            .await;
        tracker
            .record_event(&end_event("task-2", "child", false))
            .await;
        let snap = tracker.get("task-2").await.unwrap();
        assert_eq!(snap.status, SubagentStatus::Failed);
        assert_eq!(snap.success, Some(false));
    }

    #[tokio::test]
    async fn pending_list_excludes_completed_tasks() {
        let tracker = InMemorySubagentTaskTracker::new();
        tracker
            .record_event(&start_event("task-a", "parent", "child-a"))
            .await;
        tracker
            .record_event(&start_event("task-b", "parent", "child-b"))
            .await;
        tracker
            .record_event(&end_event("task-a", "child-a", true))
            .await;

        let pending = tracker.list_pending().await;
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].task_id, "task-b");
    }

    #[tokio::test]
    async fn list_for_parent_filters_by_session() {
        let tracker = InMemorySubagentTaskTracker::new();
        tracker
            .record_event(&start_event("task-a", "session-1", "child-a"))
            .await;
        tracker
            .record_event(&start_event("task-b", "session-2", "child-b"))
            .await;

        let mine = tracker.list_for_parent("session-1").await;
        assert_eq!(mine.len(), 1);
        assert_eq!(mine[0].task_id, "task-a");
    }

    #[tokio::test]
    async fn end_before_start_still_records_terminal_state() {
        let tracker = InMemorySubagentTaskTracker::new();
        tracker
            .record_event(&end_event("task-late", "child", true))
            .await;
        let snap = tracker.get("task-late").await.unwrap();
        assert_eq!(snap.status, SubagentStatus::Completed);
    }

    #[tokio::test]
    async fn non_subagent_events_are_ignored() {
        let tracker = InMemorySubagentTaskTracker::new();
        tracker
            .record_event(&AgentEvent::TextDelta {
                text: "ignore me".to_string(),
            })
            .await;
        assert!(tracker.list().await.is_empty());
    }
}
