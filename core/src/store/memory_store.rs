use super::{SessionData, SessionStore};
use crate::loop_checkpoint::LoopCheckpoint;
use crate::run::RunRecord;
use crate::subagent_task_tracker::SubagentTaskSnapshot;
use crate::tools::ArtifactStore;
use crate::trace::TraceEvent;
use crate::verification::VerificationReport;
use anyhow::Result;
use std::collections::HashMap;

// ============================================================================
// In-Memory Session Store (for testing)
// ============================================================================

/// In-memory session store for testing
pub struct MemorySessionStore {
    sessions: tokio::sync::RwLock<HashMap<String, SessionData>>,
    artifacts: tokio::sync::RwLock<HashMap<String, ArtifactStore>>,
    trace_events: tokio::sync::RwLock<HashMap<String, Vec<TraceEvent>>>,
    run_records: tokio::sync::RwLock<HashMap<String, Vec<RunRecord>>>,
    verification_reports: tokio::sync::RwLock<HashMap<String, Vec<VerificationReport>>>,
    subagent_tasks: tokio::sync::RwLock<HashMap<String, Vec<SubagentTaskSnapshot>>>,
    loop_checkpoints: tokio::sync::RwLock<HashMap<String, LoopCheckpoint>>,
}

impl MemorySessionStore {
    pub fn new() -> Self {
        Self {
            sessions: tokio::sync::RwLock::new(HashMap::new()),
            artifacts: tokio::sync::RwLock::new(HashMap::new()),
            trace_events: tokio::sync::RwLock::new(HashMap::new()),
            run_records: tokio::sync::RwLock::new(HashMap::new()),
            verification_reports: tokio::sync::RwLock::new(HashMap::new()),
            subagent_tasks: tokio::sync::RwLock::new(HashMap::new()),
            loop_checkpoints: tokio::sync::RwLock::new(HashMap::new()),
        }
    }
}

impl Default for MemorySessionStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl SessionStore for MemorySessionStore {
    async fn save(&self, session: &SessionData) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        sessions.insert(session.id.clone(), session.clone());
        Ok(())
    }

    async fn load(&self, id: &str) -> Result<Option<SessionData>> {
        let sessions = self.sessions.read().await;
        Ok(sessions.get(id).cloned())
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        sessions.remove(id);
        self.artifacts.write().await.remove(id);
        self.trace_events.write().await.remove(id);
        self.run_records.write().await.remove(id);
        self.verification_reports.write().await.remove(id);
        self.subagent_tasks.write().await.remove(id);
        // Loop checkpoints are keyed by run_id, not session_id, so a
        // session-level delete can't address them. They are removed by
        // `delete_loop_checkpoint(run_id)` — called automatically by the
        // run lifecycle when each run reaches a terminal state in-process.
        Ok(())
    }

    async fn list(&self) -> Result<Vec<String>> {
        let sessions = self.sessions.read().await;
        Ok(sessions.keys().cloned().collect())
    }

    async fn exists(&self, id: &str) -> Result<bool> {
        let sessions = self.sessions.read().await;
        Ok(sessions.contains_key(id))
    }

    async fn save_artifacts(&self, id: &str, artifacts: &ArtifactStore) -> Result<()> {
        self.artifacts
            .write()
            .await
            .insert(id.to_string(), artifacts.clone());
        Ok(())
    }

    async fn load_artifacts(&self, id: &str) -> Result<Option<ArtifactStore>> {
        Ok(self.artifacts.read().await.get(id).cloned())
    }

    async fn save_trace_events(&self, id: &str, events: &[TraceEvent]) -> Result<()> {
        self.trace_events
            .write()
            .await
            .insert(id.to_string(), events.to_vec());
        Ok(())
    }

    async fn load_trace_events(&self, id: &str) -> Result<Option<Vec<TraceEvent>>> {
        Ok(self.trace_events.read().await.get(id).cloned())
    }

    async fn save_run_records(&self, id: &str, records: &[RunRecord]) -> Result<()> {
        self.run_records
            .write()
            .await
            .insert(id.to_string(), records.to_vec());
        Ok(())
    }

    async fn load_run_records(&self, id: &str) -> Result<Option<Vec<RunRecord>>> {
        Ok(self.run_records.read().await.get(id).cloned())
    }

    async fn save_verification_reports(
        &self,
        id: &str,
        reports: &[VerificationReport],
    ) -> Result<()> {
        self.verification_reports
            .write()
            .await
            .insert(id.to_string(), reports.to_vec());
        Ok(())
    }

    async fn load_verification_reports(&self, id: &str) -> Result<Option<Vec<VerificationReport>>> {
        Ok(self.verification_reports.read().await.get(id).cloned())
    }

    async fn save_subagent_tasks(&self, id: &str, tasks: &[SubagentTaskSnapshot]) -> Result<()> {
        self.subagent_tasks
            .write()
            .await
            .insert(id.to_string(), tasks.to_vec());
        Ok(())
    }

    async fn load_subagent_tasks(&self, id: &str) -> Result<Option<Vec<SubagentTaskSnapshot>>> {
        Ok(self.subagent_tasks.read().await.get(id).cloned())
    }

    async fn save_loop_checkpoint(&self, run_id: &str, checkpoint: &LoopCheckpoint) -> Result<()> {
        self.loop_checkpoints
            .write()
            .await
            .insert(run_id.to_string(), checkpoint.clone());
        Ok(())
    }

    async fn load_loop_checkpoint(&self, run_id: &str) -> Result<Option<LoopCheckpoint>> {
        match self.loop_checkpoints.read().await.get(run_id).cloned() {
            // Enforce the same future-schema rejection as the file store so
            // the contract holds uniformly across backends.
            Some(cp) => {
                cp.ensure_loadable()?;
                Ok(Some(cp))
            }
            None => Ok(None),
        }
    }

    async fn delete_loop_checkpoint(&self, run_id: &str) -> Result<()> {
        self.loop_checkpoints.write().await.remove(run_id);
        Ok(())
    }

    fn backend_name(&self) -> &str {
        "memory"
    }
}
