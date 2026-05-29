//! Session persistence layer
//!
//! Provides pluggable session storage via the `SessionStore` trait.
//!
//! ## Default Implementation
//!
//! `FileSessionStore` stores each session as a JSON file:
//! - Session metadata (id, name, timestamps)
//! - Configuration (system prompt, policies)
//! - Conversation history (messages)
//! - Context usage statistics
//!
//! ## Custom Backends
//!
//! Implement `SessionStore` trait for custom backends (Redis, PostgreSQL, etc.):
//!
//! ```ignore
//! use a3s_code::store::{SessionStore, SessionData};
//!
//! struct RedisStore { /* ... */ }
//!
//! #[async_trait::async_trait]
//! impl SessionStore for RedisStore {
//!     async fn save(&self, session: &SessionData) -> Result<()> { /* ... */ }
//!     async fn load(&self, id: &str) -> Result<Option<SessionData>> { /* ... */ }
//!     async fn delete(&self, id: &str) -> Result<()> { /* ... */ }
//!     async fn list(&self) -> Result<Vec<String>> { /* ... */ }
//!     async fn exists(&self, id: &str) -> Result<bool> { /* ... */ }
//! }
//! ```

mod file_store;
mod memory_store;
mod session_data;

#[cfg(test)]
mod tests;

pub use file_store::FileSessionStore;
pub use memory_store::MemorySessionStore;
pub use session_data::{
    ContextUsage, LlmConfigData, SessionConfig, SessionData, SessionState,
    DEFAULT_AUTO_COMPACT_THRESHOLD,
};

use crate::loop_checkpoint::LoopCheckpoint;
use crate::run::RunRecord;
use crate::subagent_task_tracker::SubagentTaskSnapshot;
use crate::tools::ArtifactStore;
use crate::trace::TraceEvent;
use crate::verification::VerificationReport;
use anyhow::Result;

// ============================================================================
// Session Store Trait
// ============================================================================

/// Session storage trait
#[async_trait::async_trait]
pub trait SessionStore: Send + Sync {
    /// Save session data
    async fn save(&self, session: &SessionData) -> Result<()>;

    /// Load session data by ID
    async fn load(&self, id: &str) -> Result<Option<SessionData>>;

    /// Delete session data
    async fn delete(&self, id: &str) -> Result<()>;

    /// List all session IDs
    async fn list(&self) -> Result<Vec<String>>;

    /// Check if session exists
    async fn exists(&self, id: &str) -> Result<bool>;

    /// Save artifacts associated with a session.
    async fn save_artifacts(&self, _id: &str, _artifacts: &ArtifactStore) -> Result<()> {
        Ok(())
    }

    /// Load artifacts associated with a session.
    async fn load_artifacts(&self, _id: &str) -> Result<Option<ArtifactStore>> {
        Ok(None)
    }

    /// Save compact trace events associated with a session.
    async fn save_trace_events(&self, _id: &str, _events: &[TraceEvent]) -> Result<()> {
        Ok(())
    }

    /// Load compact trace events associated with a session.
    async fn load_trace_events(&self, _id: &str) -> Result<Option<Vec<TraceEvent>>> {
        Ok(None)
    }

    /// Save run snapshots and replayable runtime events associated with a session.
    async fn save_run_records(&self, _id: &str, _records: &[RunRecord]) -> Result<()> {
        Ok(())
    }

    /// Load run snapshots and replayable runtime events associated with a session.
    async fn load_run_records(&self, _id: &str) -> Result<Option<Vec<RunRecord>>> {
        Ok(None)
    }

    /// Save structured verification reports associated with a session.
    async fn save_verification_reports(
        &self,
        _id: &str,
        _reports: &[VerificationReport],
    ) -> Result<()> {
        Ok(())
    }

    /// Load structured verification reports associated with a session.
    async fn load_verification_reports(
        &self,
        _id: &str,
    ) -> Result<Option<Vec<VerificationReport>>> {
        Ok(None)
    }

    /// Save the session's delegated subagent task tracker snapshots.
    ///
    /// Cluster-grade hosts need this so a migrated session keeps a
    /// queryable history of its delegated child runs. Cancellers are
    /// **not** persisted — they are runtime-only and re-attaching them
    /// is the executor's job at task respawn time.
    async fn save_subagent_tasks(&self, _id: &str, _tasks: &[SubagentTaskSnapshot]) -> Result<()> {
        Ok(())
    }

    /// Load the session's delegated subagent task tracker snapshots.
    async fn load_subagent_tasks(&self, _id: &str) -> Result<Option<Vec<SubagentTaskSnapshot>>> {
        Ok(None)
    }

    /// Save the latest per-tool-round loop checkpoint for `run_id`.
    ///
    /// The agent loop calls this through the
    /// [`SessionStoreCheckpointSink`](crate::loop_checkpoint::SessionStoreCheckpointSink)
    /// adapter after each completed tool round. Implementations should
    /// **overwrite** any earlier checkpoint for the same `run_id` — the
    /// loop only ever needs the most recent boundary.
    async fn save_loop_checkpoint(
        &self,
        _run_id: &str,
        _checkpoint: &LoopCheckpoint,
    ) -> Result<()> {
        Ok(())
    }

    /// Load the latest loop checkpoint for `run_id`.
    async fn load_loop_checkpoint(&self, _run_id: &str) -> Result<Option<LoopCheckpoint>> {
        Ok(None)
    }

    /// Delete the loop checkpoint for `run_id`, if present.
    ///
    /// Called by the run lifecycle when a run reaches a terminal state
    /// **in-process** (completed, failed, or cancelled) — at that point
    /// the checkpoint is dead weight. Only a process crash (the agent
    /// loop never returns) should leave a checkpoint behind for
    /// crash-recovery resume. Without this, every tool-using run would
    /// leak a checkpoint forever — the dominant unbounded-growth source
    /// for long-running cluster deployments.
    ///
    /// Deleting a non-existent checkpoint is a no-op success.
    async fn delete_loop_checkpoint(&self, _run_id: &str) -> Result<()> {
        Ok(())
    }

    /// Persist a workflow checkpoint, overwriting any earlier one for the same
    /// `workflow_id`. The resumable orchestration combinators call this at each
    /// step boundary so an interrupted workflow resumes from the last
    /// completed step (here or, after migration, on another node).
    async fn save_workflow_checkpoint(
        &self,
        _workflow_id: &str,
        _checkpoint: &crate::orchestration::WorkflowCheckpoint,
    ) -> Result<()> {
        Ok(())
    }

    /// Load the latest workflow checkpoint for `workflow_id`.
    async fn load_workflow_checkpoint(
        &self,
        _workflow_id: &str,
    ) -> Result<Option<crate::orchestration::WorkflowCheckpoint>> {
        Ok(None)
    }

    /// Delete the workflow checkpoint for `workflow_id`, if present. Called
    /// when a workflow reaches a terminal state in-process; only a crash should
    /// leave one behind for resume. Deleting a non-existent checkpoint is a
    /// no-op success.
    async fn delete_workflow_checkpoint(&self, _workflow_id: &str) -> Result<()> {
        Ok(())
    }

    /// Health check — verify the store backend is reachable and operational
    async fn health_check(&self) -> Result<()> {
        Ok(())
    }

    /// Backend name for diagnostics
    fn backend_name(&self) -> &str {
        "unknown"
    }
}
