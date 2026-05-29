//! Per-tool-round loop checkpoints for crash-tolerant runs (P3 cut 1).
//!
//! The agent loop persists a [`LoopCheckpoint`] after each completed tool
//! round. The checkpoint captures the minimum state needed to recreate
//! the loop's position so a future process — typically on a different
//! node, dispatched by 书安OS after a crash or planned migration — can
//! resume from the last consistent boundary.
//!
//! Boundary policy: checkpoints are taken **only** between tool rounds,
//! never mid-tool. If a process dies while a tool is executing, the
//! work of that round is lost on resume; the LLM re-deliberates from
//! the previous checkpoint. This trades retry cost for correctness —
//! re-executing a non-idempotent tool (write, bash) on the wrong side
//! of the boundary is worse than re-asking the LLM.
//!
//! Resume API (cut 2 follow-up): not part of this cut. This module
//! lands the data contract + persistence wiring; an
//! `AgentSession::resume_run(run_id)` entry point will live on top.

use crate::llm::{Message, TokenUsage};
use crate::verification::VerificationReport;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Schema version. Bumped on incompatible format changes; impls of
/// [`LoopCheckpointSink`] should reject loads from a future version.
pub const LOOP_CHECKPOINT_SCHEMA_VERSION: u32 = 1;

/// Snapshot of the agent loop at the boundary between tool rounds.
///
/// Stored under `run_id` so resume tooling can address the correct run
/// without scanning all checkpoints of a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopCheckpoint {
    /// Schema version — see [`LOOP_CHECKPOINT_SCHEMA_VERSION`].
    #[serde(default)]
    pub schema_version: u32,

    /// Logical run identifier. Matches the `run_id` carried by
    /// [`crate::run::RunSnapshot`] and `AgentEvent`s.
    pub run_id: String,

    /// Parent session id — redundant with `run_id` lookup but useful
    /// for store layouts that key by `(session_id, run_id)`.
    pub session_id: String,

    /// 1-based tool round counter at checkpoint time.
    /// `0` is reserved for "no rounds completed yet".
    pub turn: usize,

    /// Conversation history including the just-returned tool results.
    /// On resume, the new agent loop starts from this exact message list.
    pub messages: Vec<Message>,

    /// Running token usage at checkpoint time. Lets resume re-emit
    /// progress metrics without re-querying the LLM provider.
    pub total_usage: TokenUsage,

    /// How many tool calls have been executed total in this run.
    pub tool_calls_count: usize,

    /// Verification reports collected so far in this run.
    #[serde(default)]
    pub verification_reports: Vec<VerificationReport>,

    /// Wall-clock timestamp when the checkpoint was written
    /// (Unix epoch ms — sourced from the session's
    /// [`HostEnv`](crate::host_env::HostEnv)).
    pub checkpoint_ms: u64,
}

impl LoopCheckpoint {
    /// Reject a checkpoint written by a *newer*, incompatible schema
    /// version than this build understands — honoring the contract on
    /// [`LOOP_CHECKPOINT_SCHEMA_VERSION`].
    ///
    /// Field *additions* are absorbed transparently by `#[serde(default)]`,
    /// so an older checkpoint (lower `schema_version`, including a pre-v1
    /// `0`) always remains loadable. A *future* version, however, may have
    /// changed the meaning of existing fields or the tool-round boundary
    /// semantics; resuming from one risks silent corruption (e.g.
    /// re-running a non-idempotent tool on the wrong side of the boundary).
    ///
    /// [`SessionStore`](crate::store::SessionStore) impls call this right
    /// after deserialization, so both `resume_run` (which surfaces the
    /// error to the caller) and the live-run [`LoopCheckpointSink`] (which
    /// logs and starts fresh) refuse to act on an unreadable checkpoint.
    pub fn ensure_loadable(&self) -> anyhow::Result<()> {
        if self.schema_version > LOOP_CHECKPOINT_SCHEMA_VERSION {
            anyhow::bail!(
                "loop checkpoint for run {} has schema version {} but this build supports at \
                 most {}; refusing to resume from an incompatible future checkpoint",
                self.run_id,
                self.schema_version,
                LOOP_CHECKPOINT_SCHEMA_VERSION
            );
        }
        Ok(())
    }
}

/// Receiver of per-tool-round checkpoints.
///
/// The framework ships one adapter:
/// [`SessionStoreCheckpointSink`] which forwards to a
/// [`crate::store::SessionStore`]. Hosts can implement custom sinks
/// (e.g. push directly to Redis) by implementing this trait.
#[async_trait]
pub trait LoopCheckpointSink: Send + Sync {
    /// Persist a checkpoint. Called from inside the agent loop after a
    /// successful tool round. Errors are logged at warn level and
    /// otherwise swallowed — losing a checkpoint must not halt the
    /// live run.
    async fn save_checkpoint(&self, checkpoint: &LoopCheckpoint);

    /// Load the latest checkpoint for `run_id`, if any. Returns `None`
    /// when no checkpoint has been recorded.
    async fn load_latest(&self, run_id: &str) -> Option<LoopCheckpoint>;
}

/// Default adapter that forwards checkpoints to a
/// [`SessionStore`](crate::store::SessionStore). Construct via
/// [`SessionStoreCheckpointSink::new`].
pub struct SessionStoreCheckpointSink {
    inner: std::sync::Arc<dyn crate::store::SessionStore>,
}

impl SessionStoreCheckpointSink {
    pub fn new(store: std::sync::Arc<dyn crate::store::SessionStore>) -> Self {
        Self { inner: store }
    }
}

#[async_trait]
impl LoopCheckpointSink for SessionStoreCheckpointSink {
    async fn save_checkpoint(&self, checkpoint: &LoopCheckpoint) {
        if let Err(e) = self
            .inner
            .save_loop_checkpoint(&checkpoint.run_id, checkpoint)
            .await
        {
            tracing::warn!(
                run_id = %checkpoint.run_id,
                error = %e,
                "Loop checkpoint save failed; live run continues"
            );
        }
    }

    async fn load_latest(&self, run_id: &str) -> Option<LoopCheckpoint> {
        match self.inner.load_loop_checkpoint(run_id).await {
            Ok(opt) => opt,
            Err(e) => {
                tracing::warn!(
                    run_id = %run_id,
                    error = %e,
                    "Loop checkpoint load failed"
                );
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(run_id: &str, turn: usize) -> LoopCheckpoint {
        LoopCheckpoint {
            schema_version: LOOP_CHECKPOINT_SCHEMA_VERSION,
            run_id: run_id.to_string(),
            session_id: "session-1".to_string(),
            turn,
            messages: vec![Message::user("hi")],
            total_usage: TokenUsage::default(),
            tool_calls_count: 0,
            verification_reports: Vec::new(),
            checkpoint_ms: 1_700_000_000_000,
        }
    }

    #[test]
    fn checkpoint_round_trips_through_json() {
        let cp = sample("run-1", 3);
        let json = serde_json::to_string(&cp).unwrap();
        let back: LoopCheckpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(back.run_id, "run-1");
        assert_eq!(back.turn, 3);
        assert_eq!(back.schema_version, LOOP_CHECKPOINT_SCHEMA_VERSION);
    }

    #[test]
    fn missing_schema_version_defaults_to_zero() {
        // Older payloads without the field must still load — they'll
        // be interpreted as a pre-v1 snapshot.
        let json = r#"{
            "run_id": "run-1",
            "session_id": "s",
            "turn": 1,
            "messages": [],
            "total_usage": {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0},
            "tool_calls_count": 0,
            "checkpoint_ms": 0
        }"#;
        let cp: LoopCheckpoint = serde_json::from_str(json).unwrap();
        assert_eq!(cp.schema_version, 0);
    }
}
