//! The `AgentExecutor` seam and its barrier (`parallel`) primitive.

use crate::agent::{AgentEvent, DEFAULT_MAX_PARALLEL_TASKS};
use crate::ordered_parallel::run_ordered_parallel_with_limit;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::broadcast;

/// A single unit of orchestrated agent work — *what* to run, independent of
/// *where* it runs.
///
/// Serializable on purpose: a host may ship it to another node, and
/// a future workflow checkpoint persists it. The orchestration layer assigns
/// `task_id`; everything else mirrors a delegated task.
// `serde_json::Value` (in `output_schema`) is not `Eq`, so this derives
// `PartialEq` only.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentStepSpec {
    /// Stable id for this step. Flows into lifecycle events (and, later,
    /// workflow checkpoints) so a step can be correlated and resumed.
    pub task_id: String,
    /// Registry key of the agent to run (e.g. `"explore"`, `"review"`).
    pub agent: String,
    /// Short human label for display/tracking.
    pub description: String,
    /// The instruction handed to the child agent.
    pub prompt: String,
    /// Optional per-step tool-round cap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_steps: Option<usize>,
    /// Parent session id, for lifecycle-event correlation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    /// When set, the step must return a value conforming to this JSON Schema.
    /// The executor fulfills it (the local default coerces the step's output
    /// with the structured-output machinery); the validated object lands in
    /// [`StepOutcome::structured`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<serde_json::Value>,
}

impl AgentStepSpec {
    /// A step that runs `agent` with `prompt`, identified by `task_id`.
    pub fn new(
        task_id: impl Into<String>,
        agent: impl Into<String>,
        description: impl Into<String>,
        prompt: impl Into<String>,
    ) -> Self {
        Self {
            task_id: task_id.into(),
            agent: agent.into(),
            description: description.into(),
            prompt: prompt.into(),
            max_steps: None,
            parent_session_id: None,
            output_schema: None,
        }
    }

    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    pub fn with_parent_session_id(mut self, parent_session_id: impl Into<String>) -> Self {
        self.parent_session_id = Some(parent_session_id.into());
        self
    }

    /// Require this step to return a value conforming to `schema`.
    pub fn with_output_schema(mut self, schema: serde_json::Value) -> Self {
        self.output_schema = Some(schema);
        self
    }
}

/// The result of running one [`AgentStepSpec`] to completion.
///
/// `structured` is `Some` only when the spec carried an `output_schema` and
/// the executor produced a value validated against it. (`serde_json::Value`
/// is not `Eq`, so this derives `PartialEq` only.)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StepOutcome {
    pub task_id: String,
    pub session_id: String,
    pub agent: String,
    pub output: String,
    pub success: bool,
    /// Schema-validated structured output, when the step requested one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured: Option<serde_json::Value>,
}

impl StepOutcome {
    /// A failed outcome for a step that could not start (e.g. unknown agent)
    /// or whose fan-out branch panicked. `session_id` mirrors the id the
    /// local executor would have derived, so failed steps remain addressable.
    pub fn failed(
        task_id: impl Into<String>,
        agent: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        let task_id = task_id.into();
        let session_id = format!("task-run-{task_id}");
        Self {
            task_id,
            session_id,
            agent: agent.into(),
            output: message.into(),
            success: false,
            structured: None,
        }
    }
}

/// Runs agent steps — the seam between the framework's orchestration grammar
/// and the host's placement / transport / scheduling.
///
/// The in-box [`TaskExecutor`](crate::tools::TaskExecutor) runs every step
/// locally (in-process, tokio). A host such as a cluster runtime implements this trait to
/// place steps on remote nodes; the orchestration combinators are written
/// purely against the trait and never observe where a step actually ran. The
/// framework deliberately does **not** own placement, transport, or
/// cross-node scheduling — those are the host's.
#[async_trait]
pub trait AgentExecutor: Send + Sync {
    /// Run one step to completion.
    ///
    /// Failures are reported as `StepOutcome { success: false, .. }` rather
    /// than a hard error, so a fan-out can continue when one branch fails.
    /// `event_tx`, when present, receives the step's lifecycle/progress
    /// [`AgentEvent`]s.
    async fn execute_step(
        &self,
        spec: AgentStepSpec,
        event_tx: Option<broadcast::Sender<AgentEvent>>,
    ) -> StepOutcome;

    /// Advisory ceiling on how many steps the orchestration layer should run
    /// concurrently. The local default returns its `max_parallel_tasks`; a
    /// scheduler-backed host may return its cluster-wide target. It is a
    /// *hint*, not a hard local bound — that is what lets orchestration scale
    /// past a single process.
    fn concurrency_hint(&self) -> usize {
        DEFAULT_MAX_PARALLEL_TASKS
    }
}

/// Fan `specs` out across the executor, bounded by its
/// [`concurrency_hint`](AgentExecutor::concurrency_hint), preserving input
/// order. A panicked branch becomes a failed [`StepOutcome`] without dropping
/// the others.
///
/// This is the barrier (`parallel`) primitive — it awaits every step. Later
/// combinators (pipeline, phases) build on the same seam.
pub async fn execute_steps_parallel(
    executor: Arc<dyn AgentExecutor>,
    specs: Vec<AgentStepSpec>,
    event_tx: Option<broadcast::Sender<AgentEvent>>,
) -> Vec<StepOutcome> {
    let limit = executor.concurrency_hint();
    // Keep (task_id, agent) by index so a panicked branch still yields a
    // correctly-labelled failed outcome (mirrors TaskExecutor's fallback).
    let labels: Vec<(String, String)> = specs
        .iter()
        .map(|s| (s.task_id.clone(), s.agent.clone()))
        .collect();

    let results = run_ordered_parallel_with_limit(specs, limit, move |_idx, spec| {
        let executor = Arc::clone(&executor);
        let event_tx = event_tx.clone();
        async move { executor.execute_step(spec, event_tx).await }
    })
    .await;

    results
        .into_iter()
        .map(|result| match result.output {
            Ok(outcome) => outcome,
            Err(error) => {
                let (task_id, agent) = labels
                    .get(result.index)
                    .cloned()
                    .unwrap_or_else(|| ("unknown".to_string(), "unknown".to_string()));
                StepOutcome::failed(task_id, agent, error.to_string())
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Executor with no LLM — records peak concurrency and synthesizes
    /// outcomes from the spec, so the combinator can be tested in isolation.
    struct MockExecutor {
        hint: usize,
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
    }

    impl MockExecutor {
        fn new(hint: usize) -> Self {
            Self {
                hint,
                active: Arc::new(AtomicUsize::new(0)),
                max_active: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    #[async_trait]
    impl AgentExecutor for MockExecutor {
        async fn execute_step(
            &self,
            spec: AgentStepSpec,
            _event_tx: Option<broadcast::Sender<AgentEvent>>,
        ) -> StepOutcome {
            let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(now, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(20)).await;
            self.active.fetch_sub(1, Ordering::SeqCst);

            // `boom` panics (exercise branch-panic isolation); `fail`
            // returns an unsuccessful outcome.
            assert!(spec.agent != "boom", "boom");
            StepOutcome {
                task_id: spec.task_id.clone(),
                session_id: format!("task-run-{}", spec.task_id),
                agent: spec.agent.clone(),
                output: format!("ran: {}", spec.prompt),
                success: spec.agent != "fail",
                structured: None,
            }
        }
        fn concurrency_hint(&self) -> usize {
            self.hint
        }
    }

    fn spec(id: &str, agent: &str) -> AgentStepSpec {
        AgentStepSpec::new(id, agent, "d", format!("prompt-{id}"))
    }

    #[tokio::test]
    async fn fans_out_in_input_order() {
        let exec: Arc<dyn AgentExecutor> = Arc::new(MockExecutor::new(8));
        let specs = vec![spec("a", "explore"), spec("b", "review"), spec("c", "plan")];
        let out = execute_steps_parallel(exec, specs, None).await;
        assert_eq!(
            out.iter().map(|o| o.task_id.as_str()).collect::<Vec<_>>(),
            vec!["a", "b", "c"],
            "results preserve input order"
        );
        assert!(out.iter().all(|o| o.success));
        assert_eq!(out[0].output, "ran: prompt-a");
    }

    #[tokio::test]
    async fn respects_concurrency_hint() {
        let mock = MockExecutor::new(2);
        let max_active = Arc::clone(&mock.max_active);
        let exec: Arc<dyn AgentExecutor> = Arc::new(mock);
        let specs = (0..6).map(|i| spec(&i.to_string(), "explore")).collect();
        let _ = execute_steps_parallel(exec, specs, None).await;
        assert_eq!(
            max_active.load(Ordering::SeqCst),
            2,
            "never more than concurrency_hint steps run at once"
        );
    }

    #[tokio::test]
    async fn isolates_failed_and_panicked_steps() {
        let exec: Arc<dyn AgentExecutor> = Arc::new(MockExecutor::new(8));
        let specs = vec![
            spec("ok", "explore"),
            spec("bad", "fail"),
            spec("crash", "boom"),
            spec("ok2", "review"),
        ];
        let out = execute_steps_parallel(exec, specs, None).await;
        assert_eq!(out.len(), 4, "every step yields a result");
        assert!(out[0].success);
        assert!(
            !out[1].success,
            "explicit failure surfaces as success=false"
        );
        assert!(
            !out[2].success && out[2].agent == "boom",
            "a panicked branch becomes a labelled failed outcome, not a drop"
        );
        assert!(out[3].success, "later steps unaffected by an earlier panic");
    }

    #[tokio::test]
    async fn default_concurrency_hint_is_the_framework_default() {
        struct Bare;
        #[async_trait]
        impl AgentExecutor for Bare {
            async fn execute_step(
                &self,
                spec: AgentStepSpec,
                _tx: Option<broadcast::Sender<AgentEvent>>,
            ) -> StepOutcome {
                StepOutcome::failed(spec.task_id, spec.agent, "unused")
            }
        }
        assert_eq!(Bare.concurrency_hint(), DEFAULT_MAX_PARALLEL_TASKS);
    }

    #[test]
    fn spec_and_outcome_round_trip_including_new_optional_fields() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "v": { "type": "string" } },
            "required": ["v"]
        });
        let spec = AgentStepSpec::new("t1", "explore", "d", "p")
            .with_max_steps(3)
            .with_parent_session_id("parent")
            .with_output_schema(schema.clone());
        let back: AgentStepSpec =
            serde_json::from_str(&serde_json::to_string(&spec).unwrap()).unwrap();
        assert_eq!(back, spec);
        assert_eq!(back.output_schema, Some(schema));

        let outcome = StepOutcome {
            task_id: "t1".into(),
            session_id: "task-run-t1".into(),
            agent: "explore".into(),
            output: "ok".into(),
            success: true,
            structured: Some(serde_json::json!({ "v": "x" })),
        };
        let back: StepOutcome =
            serde_json::from_str(&serde_json::to_string(&outcome).unwrap()).unwrap();
        assert_eq!(back, outcome);

        // Backward-compat: a pre-Phase-2 payload lacking the new optional
        // fields still loads (they default to None).
        let old_spec: AgentStepSpec =
            serde_json::from_str(r#"{"task_id":"t","agent":"a","description":"d","prompt":"p"}"#)
                .unwrap();
        assert_eq!(old_spec.output_schema, None);
        let old_outcome: StepOutcome = serde_json::from_str(
            r#"{"task_id":"t","session_id":"s","agent":"a","output":"o","success":true}"#,
        )
        .unwrap();
        assert_eq!(old_outcome.structured, None);
    }
}
