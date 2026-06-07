//! A typed, host-driven workflow facade over the [`AgentExecutor`] seam.
//!
//! [`Workflow`] is a thin orchestrator with **zero new execution machinery**:
//! each verb resolves to exactly one existing combinator call (see
//! [`executor`](super::executor) and [`combinators`](super::combinators)). The
//! genuinely new behavior is observability — workflow-level milestone events on
//! a dedicated [`WorkflowEvent`] broadcast — and a resume boundary at each
//! [`phase`](Workflow::phase).
//!
//! ## Control flow lives in the host
//!
//! Unlike the model-driven `task`/`parallel_task` tools (the LLM decides to fan
//! out) this is *programmable*: the caller `await`s a verb, inspects the
//! returned [`StepOutcome`]s, and decides what runs next with ordinary Rust
//! `if`/`for`/`while`/`?`/`match`. The host language IS the interpreter, so
//! there is no embedded scripting engine to sandbox and the grammar stays
//! minimal (4 verbs + `log`) — anything richer (retry, map/reduce, timeouts) is
//! a host concern.
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use a3s_code_core::orchestration::{Workflow, AgentExecutor, AgentStepSpec};
//! # async fn run(executor: Arc<dyn AgentExecutor>) {
//! let wf = Workflow::builder(executor).build();
//!
//! // One step.
//! let plan = wf
//!     .agent(AgentStepSpec::new("plan", "plan", "plan the work", "Plan the refactor"))
//!     .await;
//!
//! // Fan a *variable* number of steps out of the prior result — this is the
//! // "dynamic" part: the shape is computed at runtime, not declared up front.
//! if plan.success {
//!     let specs: Vec<AgentStepSpec> = plan
//!         .output
//!         .lines()
//!         .enumerate()
//!         .map(|(i, line)| AgentStepSpec::new(format!("impl-{i}"), "general", "impl", line))
//!         .collect();
//!     let results = wf.phase("implement", specs).await; // resumable barrier
//!     wf.log("info", &format!("implemented {} steps", results.len()), serde_json::Value::Null);
//! }
//! # }
//! ```

use super::combinators::{execute_pipeline, execute_steps_parallel_resumable, PipelineStage};
use super::executor::{execute_steps_parallel, AgentExecutor, AgentStepSpec, StepOutcome};
use super::workflow_budget::{BudgetSnapshot, WorkflowBudget};
use crate::agent::AgentEvent;
use crate::store::SessionStore;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::broadcast;

/// Default capacity of the [`WorkflowEvent`] broadcast channel. Slow
/// subscribers lag (and drop) rather than block the workflow — milestones are
/// best-effort; durable audit belongs on the trace/AHP path.
const DEFAULT_WORKFLOW_EVENT_CAPACITY: usize = 256;

/// Workflow-level milestone stream, distinct from the per-step
/// [`AgentEvent`] stream the combinators already emit.
///
/// Bridged onto whatever consumes `AgentEvent` is *not* automatic — a host that
/// wants these in its existing UI subscribes via [`Workflow::subscribe`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkflowEvent {
    /// A named [`phase`](Workflow::phase) began.
    PhaseStart {
        name: String,
        index: usize,
        step_count: usize,
    },
    /// A named [`phase`](Workflow::phase) finished.
    PhaseEnd {
        name: String,
        index: usize,
        succeeded: usize,
        failed: usize,
    },
    /// A structured log line emitted by [`Workflow::log`].
    Log {
        level: String,
        message: String,
        fields: serde_json::Value,
    },
    /// The shared workflow budget reached its cap; subsequent child LLM calls
    /// will be denied. Emitted at the phase boundary where exhaustion is first
    /// observed. The fan-out in flight is **not** force-killed.
    BudgetExhausted { resource: String, reason: String },
}

/// Builder for a [`Workflow`]. The executor is mandatory (it is the seam to the
/// host's placement / scheduling); everything else is optional.
pub struct WorkflowBuilder {
    executor: Arc<dyn AgentExecutor>,
    store: Option<Arc<dyn SessionStore>>,
    step_events: Option<broadcast::Sender<AgentEvent>>,
    root_id: Option<String>,
    budget: Option<Arc<WorkflowBudget>>,
}

impl WorkflowBuilder {
    /// Start a builder around `executor` (the default in-box executor is
    /// [`AgentSession::agent_executor`](crate::AgentSession::agent_executor)).
    pub fn new(executor: Arc<dyn AgentExecutor>) -> Self {
        Self {
            executor,
            store: None,
            step_events: None,
            root_id: None,
            budget: None,
        }
    }

    /// Attach the shared budget ledger this workflow reports on.
    ///
    /// The SAME `Arc<WorkflowBudget>` must also be installed as the executor's
    /// child `budget_guard` (the in-box wiring is done by
    /// [`AgentSession::workflow`](crate::AgentSession::workflow)); the workflow
    /// only *reads* it for [`budget_snapshot`](Workflow::budget_snapshot) and to
    /// emit [`WorkflowEvent::BudgetExhausted`].
    pub fn with_budget(mut self, budget: Arc<WorkflowBudget>) -> Self {
        self.budget = Some(budget);
        self
    }

    /// Persist progress so each [`phase`](Workflow::phase) becomes a resume
    /// boundary. Without a store, phases run as plain (non-resumable) barriers.
    pub fn with_store(mut self, store: Arc<dyn SessionStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Thread the per-step [`AgentEvent`] sender into every combinator call so
    /// existing subagent/AHP/trace listeners observe child-run lifecycle events.
    pub fn with_step_events(mut self, step_events: broadcast::Sender<AgentEvent>) -> Self {
        self.step_events = Some(step_events);
        self
    }

    /// Set the stable root id used to derive per-phase checkpoint keys. Supply a
    /// deterministic id (e.g. session-derived) for resume to work across runs;
    /// otherwise a random id is used and resume is effectively disabled.
    pub fn with_root_id(mut self, root_id: impl Into<String>) -> Self {
        self.root_id = Some(root_id.into());
        self
    }

    /// Finalize the [`Workflow`], creating its event channel.
    pub fn build(self) -> Workflow {
        let (events, _) = broadcast::channel(DEFAULT_WORKFLOW_EVENT_CAPACITY);
        let root_id = self
            .root_id
            .unwrap_or_else(|| format!("wf-{}", uuid::Uuid::new_v4()));
        Workflow {
            executor: self.executor,
            store: self.store,
            events,
            step_events: self.step_events,
            root_id,
            phase_seq: Arc::new(AtomicUsize::new(0)),
            budget: self.budget,
        }
    }
}

/// A cheaply-clonable (all-`Arc`) typed orchestration facade.
///
/// Every verb delegates to exactly one existing combinator; [`Workflow`] owns no
/// scheduling or LLM logic. It is `Clone` so a phase can itself spawn
/// sub-workflows that share the same event bus, store, and phase numbering.
#[derive(Clone)]
pub struct Workflow {
    executor: Arc<dyn AgentExecutor>,
    store: Option<Arc<dyn SessionStore>>,
    events: broadcast::Sender<WorkflowEvent>,
    step_events: Option<broadcast::Sender<AgentEvent>>,
    root_id: String,
    phase_seq: Arc<AtomicUsize>,
    budget: Option<Arc<WorkflowBudget>>,
}

impl Workflow {
    /// Begin building a workflow around `executor`.
    pub fn builder(executor: Arc<dyn AgentExecutor>) -> WorkflowBuilder {
        WorkflowBuilder::new(executor)
    }

    /// The stable root id phase checkpoint keys are derived from.
    pub fn root_id(&self) -> &str {
        &self.root_id
    }

    /// A snapshot of the shared budget ledger, if a budget is attached.
    pub fn budget_snapshot(&self) -> Option<BudgetSnapshot> {
        self.budget.as_ref().map(|b| b.snapshot())
    }

    /// Subscribe to this workflow's [`WorkflowEvent`] milestones.
    pub fn subscribe(&self) -> broadcast::Receiver<WorkflowEvent> {
        self.events.subscribe()
    }

    /// Run a single step. A one-element barrier — reuses the panic isolation and
    /// placement of [`parallel`](Self::parallel) so one and many steps share a
    /// single code path.
    pub async fn agent(&self, spec: AgentStepSpec) -> StepOutcome {
        let task_id = spec.task_id.clone();
        let agent = spec.agent.clone();
        execute_steps_parallel(
            Arc::clone(&self.executor),
            vec![spec],
            self.step_events.clone(),
        )
        .await
        .into_iter()
        .next()
        .unwrap_or_else(|| StepOutcome::failed(task_id, agent, "step produced no outcome"))
    }

    /// Barrier fan-out. Bounded by the executor's
    /// [`concurrency_hint`](AgentExecutor::concurrency_hint); input order is
    /// preserved and a panicked branch becomes a failed [`StepOutcome`].
    pub async fn parallel(&self, specs: Vec<AgentStepSpec>) -> Vec<StepOutcome> {
        execute_steps_parallel(Arc::clone(&self.executor), specs, self.step_events.clone()).await
    }

    /// A *named* barrier and a resume boundary.
    ///
    /// Emits [`WorkflowEvent::PhaseStart`]/[`WorkflowEvent::PhaseEnd`] around the
    /// inner combinator. When a store is configured the phase runs through
    /// [`execute_steps_parallel_resumable`] keyed by a deterministic
    /// `"{root_id}/{index}:{name}"` id, so an interrupted run skips already
    /// completed steps; otherwise it is a plain [`parallel`](Self::parallel).
    pub async fn phase(&self, name: &str, specs: Vec<AgentStepSpec>) -> Vec<StepOutcome> {
        let index = self.phase_seq.fetch_add(1, Ordering::SeqCst);
        let _ = self.events.send(WorkflowEvent::PhaseStart {
            name: name.to_string(),
            index,
            step_count: specs.len(),
        });

        let out = match &self.store {
            Some(store) => {
                let workflow_id = format!("{}/{index}:{name}", self.root_id);
                execute_steps_parallel_resumable(
                    Arc::clone(&self.executor),
                    specs,
                    &workflow_id,
                    Arc::clone(store),
                    self.step_events.clone(),
                )
                .await
            }
            None => self.parallel(specs).await,
        };

        let failed = out.iter().filter(|o| !o.success).count();
        let _ = self.events.send(WorkflowEvent::PhaseEnd {
            name: name.to_string(),
            index,
            succeeded: out.len() - failed,
            failed,
        });

        // Surface a budget cap once it is reached so the host can stop issuing
        // further phases. Enforcement itself lives in the child loops (via the
        // shared budget guard); this is observation only.
        if let Some(budget) = &self.budget {
            if budget.is_exhausted() {
                let snap = budget.snapshot();
                let _ = self.events.send(WorkflowEvent::BudgetExhausted {
                    resource: "workflow_tokens".to_string(),
                    reason: format!(
                        "workflow token budget exhausted ({} / {} tokens)",
                        snap.consumed_tokens,
                        snap.limit_tokens.unwrap_or(0)
                    ),
                });
            }
        }
        out
    }

    /// Per-item staged chains with **no barrier between stages** — delegates
    /// straight to [`execute_pipeline`]. Item A may be in stage 3 while item B is
    /// still in stage 1.
    pub async fn pipeline<I>(
        &self,
        items: Vec<I>,
        stages: Vec<PipelineStage<I>>,
    ) -> Vec<Option<StepOutcome>>
    where
        I: Send + 'static,
    {
        execute_pipeline(
            Arc::clone(&self.executor),
            items,
            stages,
            self.step_events.clone(),
        )
        .await
    }

    /// Emit a best-effort structured log line on the [`WorkflowEvent`] stream.
    /// Synchronous and non-failing: a full channel drops the line rather than
    /// blocking the workflow.
    pub fn log(&self, level: &str, message: &str, fields: serde_json::Value) {
        let _ = self.events.send(WorkflowEvent::Log {
            level: level.to_string(),
            message: message.to_string(),
            fields,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestration::WorkflowCheckpoint;
    use crate::store::MemorySessionStore;
    use async_trait::async_trait;
    use std::collections::HashMap;

    /// Echoes the prompt into the output; records which task ids it ran.
    struct EchoExecutor {
        ran: Arc<tokio::sync::Mutex<Vec<String>>>,
    }

    impl EchoExecutor {
        fn new() -> Self {
            Self {
                ran: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            }
        }
    }

    #[async_trait]
    impl AgentExecutor for EchoExecutor {
        async fn execute_step(
            &self,
            spec: AgentStepSpec,
            _event_tx: Option<broadcast::Sender<AgentEvent>>,
        ) -> StepOutcome {
            self.ran.lock().await.push(spec.task_id.clone());
            StepOutcome {
                task_id: spec.task_id.clone(),
                session_id: format!("task-run-{}", spec.task_id),
                agent: spec.agent.clone(),
                output: spec.prompt.clone(),
                success: spec.agent != "fail",
                structured: None,
            }
        }
        fn concurrency_hint(&self) -> usize {
            4
        }
    }

    fn spec(id: &str, agent: &str, prompt: &str) -> AgentStepSpec {
        AgentStepSpec::new(id, agent, "d", prompt)
    }

    #[tokio::test]
    async fn agent_runs_a_single_step() {
        let wf = Workflow::builder(Arc::new(EchoExecutor::new())).build();
        let out = wf.agent(spec("a", "explore", "hello")).await;
        assert_eq!(out.task_id, "a");
        assert_eq!(out.output, "hello");
        assert!(out.success);
    }

    #[tokio::test]
    async fn parallel_preserves_order_and_isolates_failure() {
        let wf = Workflow::builder(Arc::new(EchoExecutor::new())).build();
        let out = wf
            .parallel(vec![
                spec("a", "explore", "pa"),
                spec("b", "fail", "pb"),
                spec("c", "review", "pc"),
            ])
            .await;
        assert_eq!(
            out.iter().map(|o| o.task_id.as_str()).collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
        assert!(out[0].success);
        assert!(
            !out[1].success,
            "the failing branch surfaces as success=false"
        );
        assert!(out[2].success);
    }

    #[tokio::test]
    async fn phase_emits_start_and_end_milestones() {
        let wf = Workflow::builder(Arc::new(EchoExecutor::new())).build();
        let mut rx = wf.subscribe();
        let out = wf
            .phase(
                "review",
                vec![spec("a", "review", "p"), spec("b", "fail", "p")],
            )
            .await;
        assert_eq!(out.len(), 2);

        let start = rx.recv().await.unwrap();
        assert_eq!(
            start,
            WorkflowEvent::PhaseStart {
                name: "review".to_string(),
                index: 0,
                step_count: 2,
            }
        );
        let end = rx.recv().await.unwrap();
        assert_eq!(
            end,
            WorkflowEvent::PhaseEnd {
                name: "review".to_string(),
                index: 0,
                succeeded: 1,
                failed: 1,
            }
        );
    }

    #[tokio::test]
    async fn phases_get_monotonic_indices() {
        let wf = Workflow::builder(Arc::new(EchoExecutor::new())).build();
        let mut rx = wf.subscribe();
        wf.phase("one", vec![spec("a", "explore", "p")]).await;
        wf.phase("two", vec![spec("b", "explore", "p")]).await;

        let indices: Vec<usize> = {
            let mut seen = Vec::new();
            while let Ok(ev) = rx.try_recv() {
                if let WorkflowEvent::PhaseStart { index, .. } = ev {
                    seen.push(index);
                }
            }
            seen
        };
        assert_eq!(indices, vec![0, 1], "phase indices increment per call");
    }

    #[tokio::test]
    async fn phase_with_store_resumes_from_checkpoint() {
        let store: Arc<dyn SessionStore> = Arc::new(MemorySessionStore::new());
        let exec = Arc::new(EchoExecutor::new());
        let ran = Arc::clone(&exec.ran);
        let wf = Workflow::builder(exec)
            .with_store(Arc::clone(&store))
            .with_root_id("root")
            .build();

        // Pre-seed the checkpoint for phase 0 ("implement"): step "a" is already
        // done. The deterministic id must match what phase() derives.
        let mut done = HashMap::new();
        done.insert(
            "a".to_string(),
            StepOutcome {
                task_id: "a".into(),
                session_id: "task-run-a".into(),
                agent: "explore".into(),
                output: "cached-a".into(),
                success: true,
                structured: None,
            },
        );
        store
            .save_workflow_checkpoint(
                "root/0:implement",
                &WorkflowCheckpoint::from_completed("root/0:implement", &done, 1),
            )
            .await
            .unwrap();

        let out = wf
            .phase(
                "implement",
                vec![spec("a", "explore", "pa"), spec("b", "review", "pb")],
            )
            .await;

        assert_eq!(
            *ran.lock().await,
            vec!["b".to_string()],
            "only the not-yet-completed step actually runs"
        );
        assert_eq!(
            out[0].output, "cached-a",
            "completed step reuses its cached outcome"
        );
        assert!(out.iter().all(|o| o.success));
    }

    #[tokio::test]
    async fn pipeline_chains_stages_per_item() {
        let wf = Workflow::builder(Arc::new(EchoExecutor::new())).build();
        let stages: Vec<PipelineStage<&str>> = vec![
            Arc::new(|_prev: Option<&StepOutcome>, item: &&str| {
                Some(AgentStepSpec::new("s1", "explore", "d", *item))
            }),
            Arc::new(|prev: Option<&StepOutcome>, _item: &&str| {
                let prior = prev.map(|o| o.output.clone()).unwrap_or_default();
                Some(AgentStepSpec::new(
                    "s2",
                    "review",
                    "d",
                    format!("review of: {prior}"),
                ))
            }),
        ];
        let out = wf.pipeline(vec!["alpha", "beta"], stages).await;
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].as_ref().unwrap().output, "review of: alpha");
        assert_eq!(out[1].as_ref().unwrap().output, "review of: beta");
    }

    #[tokio::test]
    async fn log_emits_a_log_event() {
        let wf = Workflow::builder(Arc::new(EchoExecutor::new())).build();
        let mut rx = wf.subscribe();
        wf.log("info", "hello", serde_json::json!({ "k": 1 }));
        let ev = rx.recv().await.unwrap();
        assert_eq!(
            ev,
            WorkflowEvent::Log {
                level: "info".to_string(),
                message: "hello".to_string(),
                fields: serde_json::json!({ "k": 1 }),
            }
        );
    }

    #[tokio::test]
    async fn phase_emits_budget_exhausted_when_capped() {
        use crate::budget::BudgetGuard;
        use crate::llm::TokenUsage;

        let budget = Arc::new(WorkflowBudget::new(Some(10)));
        // Spend past the cap before the phase runs (simulates child loops having
        // already recorded usage into the shared ledger).
        budget
            .record_after_llm(
                "s",
                &TokenUsage {
                    total_tokens: 12,
                    ..Default::default()
                },
            )
            .await;

        let wf = Workflow::builder(Arc::new(EchoExecutor::new()))
            .with_budget(Arc::clone(&budget))
            .build();
        let mut rx = wf.subscribe();
        wf.phase("p", vec![spec("a", "explore", "p")]).await;

        let mut saw_exhausted = false;
        while let Ok(ev) = rx.try_recv() {
            if let WorkflowEvent::BudgetExhausted { resource, .. } = ev {
                assert_eq!(resource, "workflow_tokens");
                saw_exhausted = true;
            }
        }
        assert!(
            saw_exhausted,
            "phase boundary emits BudgetExhausted once capped"
        );
        assert_eq!(wf.budget_snapshot().unwrap().consumed_tokens, 12);
    }

    #[tokio::test]
    async fn no_budget_means_no_snapshot_and_no_exhausted_event() {
        let wf = Workflow::builder(Arc::new(EchoExecutor::new())).build();
        let mut rx = wf.subscribe();
        wf.phase("p", vec![spec("a", "explore", "p")]).await;
        assert!(wf.budget_snapshot().is_none());
        while let Ok(ev) = rx.try_recv() {
            assert!(
                !matches!(ev, WorkflowEvent::BudgetExhausted { .. }),
                "no budget → never a BudgetExhausted event"
            );
        }
    }

    #[tokio::test]
    async fn agent_returns_failed_outcome_when_executor_yields_nothing() {
        // A pathological executor whose fan-out yields no result still gives the
        // caller an addressable failed outcome rather than panicking.
        struct Empty;
        #[async_trait]
        impl AgentExecutor for Empty {
            async fn execute_step(
                &self,
                spec: AgentStepSpec,
                _tx: Option<broadcast::Sender<AgentEvent>>,
            ) -> StepOutcome {
                StepOutcome::failed(spec.task_id, spec.agent, "boom")
            }
        }
        let wf = Workflow::builder(Arc::new(Empty)).build();
        let out = wf.agent(spec("x", "explore", "p")).await;
        assert_eq!(out.task_id, "x");
        assert!(!out.success);
    }
}
