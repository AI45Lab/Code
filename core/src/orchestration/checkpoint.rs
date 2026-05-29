//! Workflow-level checkpoints: journal completed steps so an interrupted
//! orchestration resumes from the longest completed prefix — on this node or,
//! because the checkpoint is serializable and the executor is pluggable, on
//! another one (host-driven migration).
//!
//! This is the step-boundary analogue of [`LoopCheckpoint`](crate::loop_checkpoint::LoopCheckpoint),
//! which checkpoints at tool-round boundaries one level down.

use super::executor::StepOutcome;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Schema version. Bumped on incompatible format changes; loads from a future
/// version are rejected (see [`WorkflowCheckpoint::ensure_loadable`]).
pub const WORKFLOW_CHECKPOINT_SCHEMA_VERSION: u32 = 1;

/// One completed step within a workflow.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkflowStepRecord {
    /// Matches the [`AgentStepSpec::task_id`](super::AgentStepSpec) of the
    /// step that produced this outcome.
    pub task_id: String,
    /// The completed step's result.
    pub outcome: StepOutcome,
}

/// Snapshot of a workflow's completed steps at a step boundary.
///
/// (`StepOutcome` contains a `serde_json::Value`, which is not `Eq`, so this
/// derives `PartialEq` only.)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkflowCheckpoint {
    /// Schema version — see [`WORKFLOW_CHECKPOINT_SCHEMA_VERSION`].
    #[serde(default)]
    pub schema_version: u32,
    /// Logical workflow identifier the checkpoint is keyed by.
    pub workflow_id: String,
    /// The steps completed so far. A resuming run skips these and re-dispatches
    /// only the rest.
    pub steps: Vec<WorkflowStepRecord>,
    /// Wall-clock timestamp when the checkpoint was written (Unix epoch ms).
    pub checkpoint_ms: u64,
}

impl WorkflowCheckpoint {
    /// Build a checkpoint from a map of completed `task_id -> outcome`.
    pub fn from_completed(
        workflow_id: impl Into<String>,
        completed: &HashMap<String, StepOutcome>,
        checkpoint_ms: u64,
    ) -> Self {
        let steps = completed
            .iter()
            .map(|(task_id, outcome)| WorkflowStepRecord {
                task_id: task_id.clone(),
                outcome: outcome.clone(),
            })
            .collect();
        Self {
            schema_version: WORKFLOW_CHECKPOINT_SCHEMA_VERSION,
            workflow_id: workflow_id.into(),
            steps,
            checkpoint_ms,
        }
    }

    /// The completed steps as a `task_id -> outcome` map.
    pub fn completed(&self) -> HashMap<String, StepOutcome> {
        self.steps
            .iter()
            .map(|r| (r.task_id.clone(), r.outcome.clone()))
            .collect()
    }

    /// Reject a checkpoint written by a *newer*, incompatible schema version
    /// than this build understands — mirrors
    /// [`LoopCheckpoint::ensure_loadable`](crate::loop_checkpoint::LoopCheckpoint::ensure_loadable).
    /// Field additions are absorbed by `#[serde(default)]`, so older (incl.
    /// pre-v1 `0`) checkpoints always remain loadable.
    pub fn ensure_loadable(&self) -> anyhow::Result<()> {
        if self.schema_version > WORKFLOW_CHECKPOINT_SCHEMA_VERSION {
            anyhow::bail!(
                "workflow checkpoint {} has schema version {} but this build supports at \
                 most {}; refusing to resume from an incompatible future checkpoint",
                self.workflow_id,
                self.schema_version,
                WORKFLOW_CHECKPOINT_SCHEMA_VERSION
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(id: &str) -> StepOutcome {
        StepOutcome {
            task_id: id.to_string(),
            session_id: format!("task-run-{id}"),
            agent: "a".to_string(),
            output: "o".to_string(),
            success: true,
            structured: None,
        }
    }

    #[test]
    fn round_trips_and_exposes_completed_map() {
        let mut completed = HashMap::new();
        completed.insert("t1".to_string(), outcome("t1"));
        let cp = WorkflowCheckpoint::from_completed("wf", &completed, 123);
        let back: WorkflowCheckpoint =
            serde_json::from_str(&serde_json::to_string(&cp).unwrap()).unwrap();
        assert_eq!(back, cp);
        assert_eq!(back.schema_version, WORKFLOW_CHECKPOINT_SCHEMA_VERSION);
        assert_eq!(back.checkpoint_ms, 123);
        assert_eq!(back.completed().get("t1").unwrap().task_id, "t1");
    }

    #[test]
    fn ensure_loadable_rejects_only_future_versions() {
        let mut cp = WorkflowCheckpoint::from_completed("wf", &HashMap::new(), 0);
        cp.ensure_loadable().expect("current version loadable");
        cp.schema_version = 0;
        cp.ensure_loadable().expect("pre-v1 loadable");
        cp.schema_version = WORKFLOW_CHECKPOINT_SCHEMA_VERSION + 1;
        let err = cp.ensure_loadable().unwrap_err();
        assert!(err.to_string().contains("schema version"), "got: {err}");
    }

    #[test]
    fn pre_v1_payload_without_schema_version_loads() {
        let json = r#"{"workflow_id":"wf","steps":[],"checkpoint_ms":0}"#;
        let cp: WorkflowCheckpoint = serde_json::from_str(json).unwrap();
        assert_eq!(cp.schema_version, 0);
    }
}
