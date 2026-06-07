//! Orchestration combinators built on the [`AgentExecutor`] seam.
//!
//! [`execute_steps_parallel`](super::execute_steps_parallel) (in `executor`)
//! is the barrier (`parallel`) primitive. This module adds `pipeline`: the
//! one genuinely new scheduling shape, where each item flows through a chain
//! of stages independently — no barrier between stages.

use super::checkpoint::WorkflowCheckpoint;
use super::executor::{execute_steps_parallel, AgentExecutor, AgentStepSpec, StepOutcome};
use crate::agent::AgentEvent;
use crate::ordered_parallel::run_ordered_parallel_with_limit;
use crate::store::SessionStore;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::broadcast;

fn now_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// A pipeline stage: given the previous stage's outcome (`None` before the
/// first stage) and the original item, produce the next step to run — or
/// `None` to stop this item's chain early.
///
/// Stages are pure spec-builders; the executor runs them. A stage can branch
/// on the prior result (e.g. "verify the finding the review stage produced").
pub type PipelineStage<I> =
    Arc<dyn Fn(Option<&StepOutcome>, &I) -> Option<AgentStepSpec> + Send + Sync>;

/// Run each item through `stages` as an independent chain.
///
/// All chains run concurrently, bounded by the executor's
/// [`concurrency_hint`](AgentExecutor::concurrency_hint) — there is **no
/// barrier between stages**, so item A can be in stage 3 while item B is still
/// in stage 1. Wall-clock is the slowest single chain, not the
/// sum-of-slowest-per-stage that a barrier `parallel` per stage would incur.
///
/// A chain stops early when a stage returns `None` or when a step fails
/// (later stages would only build on a failed result). Returns each item's
/// last outcome (`None` if its first stage produced no spec), preserving input
/// order. A stage closure that panics isolates to that one chain (its result
/// becomes `None`) without dropping the others.
pub async fn execute_pipeline<I>(
    executor: Arc<dyn AgentExecutor>,
    items: Vec<I>,
    stages: Vec<PipelineStage<I>>,
    event_tx: Option<broadcast::Sender<AgentEvent>>,
) -> Vec<Option<StepOutcome>>
where
    I: Send + 'static,
{
    let limit = executor.concurrency_hint();
    let stages = Arc::new(stages);

    let results = run_ordered_parallel_with_limit(items, limit, move |_idx, item| {
        let executor = Arc::clone(&executor);
        let stages = Arc::clone(&stages);
        let event_tx = event_tx.clone();
        async move {
            let mut prev: Option<StepOutcome> = None;
            for stage in stages.iter() {
                let Some(spec) = stage(prev.as_ref(), &item) else {
                    break;
                };
                let outcome = executor.execute_step(spec, event_tx.clone()).await;
                let succeeded = outcome.success;
                prev = Some(outcome);
                if !succeeded {
                    break;
                }
            }
            prev
        }
    })
    .await;

    // A panicked chain (Err) yields `None`; a normal chain yields its last
    // outcome. Order is preserved by `run_ordered_parallel_with_limit`.
    results
        .into_iter()
        .map(|result| result.output.unwrap_or(None))
        .collect()
}

/// Like [`execute_steps_parallel`](super::execute_steps_parallel), but
/// **resumable**: progress is journaled to `store` under `workflow_id`, so an
/// interrupted run picks up from the last completed step.
///
/// On entry, any steps already recorded in a prior checkpoint are skipped and
/// their cached outcomes reused; only the rest are dispatched. As each step
/// completes, the checkpoint is rewritten (the step boundary), so a crash
/// mid-run loses at most the in-flight steps. Because the checkpoint is
/// serializable and the executor is a parameter, a host can resume an
/// interrupted workflow on a *different* node by passing that node's executor.
///
/// Results are returned in the original `specs` order. On full success the
/// checkpoint is deleted (the workflow is terminal); only a crash leaves one
/// behind for resume.
pub async fn execute_steps_parallel_resumable(
    executor: Arc<dyn AgentExecutor>,
    specs: Vec<AgentStepSpec>,
    workflow_id: &str,
    store: Arc<dyn SessionStore>,
    event_tx: Option<broadcast::Sender<AgentEvent>>,
) -> Vec<StepOutcome> {
    // Prior progress. An unreadable checkpoint — e.g. one written by a newer,
    // incompatible schema version, which the store rejects via
    // `ensure_loadable` — is treated as *no* prior progress: the workflow
    // re-runs from scratch rather than resuming from state it can't interpret.
    // That's a fail-safe (do the work), but surface it rather than swallow it.
    let done: HashMap<String, StepOutcome> = match store.load_workflow_checkpoint(workflow_id).await
    {
        Ok(Some(cp)) => cp.completed(),
        Ok(None) => HashMap::new(),
        Err(e) => {
            tracing::warn!(
                workflow_id = %workflow_id,
                error = %e,
                "workflow checkpoint unreadable; re-running the workflow from scratch"
            );
            HashMap::new()
        }
    };

    let pending: Vec<AgentStepSpec> = specs
        .iter()
        .filter(|s| !done.contains_key(&s.task_id))
        .cloned()
        .collect();
    let labels: Vec<(String, String)> = pending
        .iter()
        .map(|s| (s.task_id.clone(), s.agent.clone()))
        .collect();

    // Accumulator seeded with prior progress; persisted at every step boundary.
    let acc = Arc::new(tokio::sync::Mutex::new(done.clone()));
    let limit = executor.concurrency_hint();
    let workflow_id_owned = workflow_id.to_string();
    let store_steps = Arc::clone(&store);

    let results = run_ordered_parallel_with_limit(pending, limit, move |_idx, spec| {
        let executor = Arc::clone(&executor);
        let event_tx = event_tx.clone();
        let acc = Arc::clone(&acc);
        let store = Arc::clone(&store_steps);
        let workflow_id = workflow_id_owned.clone();
        async move {
            let outcome = executor.execute_step(spec, event_tx).await;
            // Step boundary: record only *successful* steps, so a failed step
            // is retried on resume (its effect didn't complete) while a
            // succeeded step's work is never redone.
            if outcome.success {
                let mut guard = acc.lock().await;
                guard.insert(outcome.task_id.clone(), outcome.clone());
                let checkpoint =
                    WorkflowCheckpoint::from_completed(&workflow_id, &guard, now_epoch_ms());
                if let Err(e) = store
                    .save_workflow_checkpoint(&workflow_id, &checkpoint)
                    .await
                {
                    // Losing a checkpoint must not fail the live run.
                    tracing::warn!(
                        workflow_id = %workflow_id,
                        error = %e,
                        "workflow checkpoint save failed; run continues"
                    );
                }
            }
            outcome
        }
    })
    .await;

    let mut fresh: HashMap<String, StepOutcome> = HashMap::new();
    for result in results {
        match result.output {
            Ok(outcome) => {
                fresh.insert(outcome.task_id.clone(), outcome);
            }
            Err(error) => {
                if let Some((task_id, agent)) = labels.get(result.index).cloned() {
                    fresh.insert(
                        task_id.clone(),
                        StepOutcome::failed(task_id, agent, error.to_string()),
                    );
                }
            }
        }
    }

    // Merge cached + freshly-run, in the original spec order.
    let merged: Vec<StepOutcome> = specs
        .iter()
        .map(|s| {
            done.get(&s.task_id)
                .cloned()
                .or_else(|| fresh.remove(&s.task_id))
                .unwrap_or_else(|| {
                    StepOutcome::failed(
                        s.task_id.clone(),
                        s.agent.clone(),
                        "step produced no outcome",
                    )
                })
        })
        .collect();

    if merged.iter().all(|o| o.success) {
        let _ = store.delete_workflow_checkpoint(workflow_id).await;
    }
    merged
}

/// What an [`execute_loop`] predicate decides after seeing a round's outcomes.
pub enum LoopDecision {
    /// Run another round with these specs.
    Continue(Vec<AgentStepSpec>),
    /// Stop now; the loop returns the round that just completed.
    Stop,
}

/// Run rounds until the predicate says [`Stop`](LoopDecision::Stop), a round is
/// asked to run no specs, or `max_iterations` is reached — whichever comes
/// first. Each round is a barrier ([`execute_steps_parallel`]); `next` receives
/// the just-completed round's outcomes and decides whether (and with what) to
/// continue. Returns the last round's outcomes (empty if `initial` was empty).
///
/// `max_iterations` is **mandatory and a hard cap**: it is clamped to at least
/// 1, and once reached the loop stops even if the predicate returns
/// [`Continue`](LoopDecision::Continue). This is the guard that makes an
/// LLM-driven, unknown-length loop (e.g. loop-until-dry) safe — the predicate
/// must never be the *only* termination condition.
///
/// This is the "loop" shape from the orchestration grammar; like the other
/// combinators it is written purely against the [`AgentExecutor`] seam and adds
/// no scheduling of its own.
pub async fn execute_loop<F>(
    executor: Arc<dyn AgentExecutor>,
    initial: Vec<AgentStepSpec>,
    max_iterations: usize,
    event_tx: Option<broadcast::Sender<AgentEvent>>,
    mut next: F,
) -> Vec<StepOutcome>
where
    F: FnMut(&[StepOutcome]) -> LoopDecision + Send,
{
    let cap = max_iterations.max(1);
    let mut specs = initial;
    let mut last = Vec::new();
    let mut iterations = 0;

    while !specs.is_empty() {
        let round = execute_steps_parallel(
            Arc::clone(&executor),
            std::mem::take(&mut specs),
            event_tx.clone(),
        )
        .await;
        iterations += 1;
        let decision = next(&round);
        last = round;
        match decision {
            LoopDecision::Continue(more) if iterations < cap => specs = more,
            _ => break,
        }
    }

    last
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Echoes the prompt into the output; fails for agent `"fail"`; panics for
    /// agent `"boom"`. Records peak concurrency.
    struct EchoExecutor {
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
    }

    impl EchoExecutor {
        fn new() -> Self {
            Self {
                active: Arc::new(AtomicUsize::new(0)),
                max_active: Arc::new(AtomicUsize::new(0)),
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
            let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(now, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(15)).await;
            self.active.fetch_sub(1, Ordering::SeqCst);
            assert!(spec.agent != "boom", "boom");
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

    fn stage<I, F>(f: F) -> PipelineStage<I>
    where
        F: Fn(Option<&StepOutcome>, &I) -> Option<AgentStepSpec> + Send + Sync + 'static,
    {
        Arc::new(f)
    }

    #[tokio::test]
    async fn each_item_chains_through_stages_and_later_stages_see_prior_output() {
        let exec: Arc<dyn AgentExecutor> = Arc::new(EchoExecutor::new());
        // Stage 1: run agent "explore" with the item as the prompt.
        // Stage 2: run agent "review" with a prompt derived from stage 1's output.
        let stages = vec![
            stage(|_prev: Option<&StepOutcome>, item: &&str| {
                Some(AgentStepSpec::new("s1", "explore", "d", *item))
            }),
            stage(|prev: Option<&StepOutcome>, _item: &&str| {
                let prior = prev.map(|o| o.output.clone()).unwrap_or_default();
                Some(AgentStepSpec::new(
                    "s2",
                    "review",
                    "d",
                    format!("review of: {prior}"),
                ))
            }),
        ];
        let out = execute_pipeline(exec, vec!["alpha", "beta"], stages, None).await;

        assert_eq!(out.len(), 2, "one result per item, order preserved");
        // Each item's final outcome is stage 2, whose prompt was derived from
        // stage 1's output (the item text).
        assert_eq!(out[0].as_ref().unwrap().output, "review of: alpha");
        assert_eq!(out[1].as_ref().unwrap().output, "review of: beta");
        assert!(out.iter().all(|o| o.as_ref().unwrap().success));
    }

    #[tokio::test]
    async fn chain_stops_on_failure_and_on_none_stage() {
        let exec: Arc<dyn AgentExecutor> = Arc::new(EchoExecutor::new());
        // First item: stage 1 fails (agent "fail") → stage 2 must not run.
        // Second item: stage 1 ok, stage 2 returns None → chain stops at stage 1.
        let stages = vec![
            stage(|_p: Option<&StepOutcome>, item: &&str| {
                let agent = if *item == "x" { "fail" } else { "explore" };
                Some(AgentStepSpec::new("s1", agent, "d", *item))
            }),
            stage(|_p: Option<&StepOutcome>, item: &&str| {
                if *item == "y" {
                    None // stop the second item's chain at stage 1
                } else {
                    Some(AgentStepSpec::new("s2", "review", "d", "second"))
                }
            }),
        ];
        let out = execute_pipeline(exec, vec!["x", "y"], stages, None).await;

        let first = out[0].as_ref().unwrap();
        assert!(!first.success, "failed stage 1 surfaces");
        assert_eq!(
            first.output, "x",
            "stage 2 did not run after stage 1 failed"
        );

        let second = out[1].as_ref().unwrap();
        assert!(second.success);
        assert_eq!(
            second.output, "y",
            "stage 2 returned None → chain stopped at stage 1"
        );
    }

    #[tokio::test]
    async fn no_barrier_between_stages_bounded_by_hint() {
        let echo = EchoExecutor::new();
        let max_active = Arc::clone(&echo.max_active);
        let exec: Arc<dyn AgentExecutor> = Arc::new(echo);
        let stages = vec![
            stage(|_p: Option<&StepOutcome>, item: &usize| {
                Some(AgentStepSpec::new(
                    format!("s1-{item}"),
                    "explore",
                    "d",
                    "p",
                ))
            }),
            stage(|_p: Option<&StepOutcome>, item: &usize| {
                Some(AgentStepSpec::new(format!("s2-{item}"), "review", "d", "p"))
            }),
        ];
        let items: Vec<usize> = (0..8).collect();
        let out = execute_pipeline(exec, items, stages, None).await;
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|o| o.is_some()));
        // concurrency_hint is 4: chains run concurrently but never exceed it.
        assert!(
            max_active.load(Ordering::SeqCst) <= 4,
            "concurrency never exceeds the executor's hint"
        );
    }

    #[tokio::test]
    async fn panicking_stage_isolates_to_its_chain() {
        let exec: Arc<dyn AgentExecutor> = Arc::new(EchoExecutor::new());
        let stages = vec![stage(|_p: Option<&StepOutcome>, item: &&str| {
            // The middle item routes to the panicking agent.
            Some(AgentStepSpec::new("s1", *item, "d", "p"))
        })];
        let out = execute_pipeline(exec, vec!["explore", "boom", "review"], stages, None).await;
        assert_eq!(out.len(), 3);
        assert!(out[0].as_ref().unwrap().success);
        assert!(out[1].is_none(), "panicked chain becomes None, not a drop");
        assert!(out[2].as_ref().unwrap().success, "later chains unaffected");
    }

    /// Records which task ids it actually ran; always succeeds.
    struct RecordingExecutor {
        ran: Arc<tokio::sync::Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl AgentExecutor for RecordingExecutor {
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
                output: format!("ran:{}", spec.task_id),
                success: true,
                structured: None,
            }
        }
        fn concurrency_hint(&self) -> usize {
            4
        }
    }

    #[tokio::test]
    async fn resumable_skips_completed_then_clears_on_success() {
        use crate::store::MemorySessionStore;
        let store: Arc<dyn SessionStore> = Arc::new(MemorySessionStore::new());

        // Pre-seed: step "a" already completed on a prior run (possibly on
        // another node — this exercises the migration path too).
        let mut done = std::collections::HashMap::new();
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
                "wf-1",
                &WorkflowCheckpoint::from_completed("wf-1", &done, 1),
            )
            .await
            .unwrap();

        // A FRESH executor resumes (the node that runs the rest is not the one
        // that completed "a").
        let ran = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let exec: Arc<dyn AgentExecutor> = Arc::new(RecordingExecutor {
            ran: Arc::clone(&ran),
        });
        let specs = vec![
            AgentStepSpec::new("a", "explore", "d", "pa"),
            AgentStepSpec::new("b", "review", "d", "pb"),
        ];

        let out =
            execute_steps_parallel_resumable(exec, specs, "wf-1", Arc::clone(&store), None).await;

        assert_eq!(
            *ran.lock().await,
            vec!["b".to_string()],
            "only the not-yet-completed step runs"
        );
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].task_id, "a");
        assert_eq!(
            out[0].output, "cached-a",
            "completed step returns its cached outcome, unchanged"
        );
        assert_eq!(out[1].task_id, "b");
        assert!(out.iter().all(|o| o.success));
        assert!(
            store
                .load_workflow_checkpoint("wf-1")
                .await
                .unwrap()
                .is_none(),
            "a fully-succeeded workflow clears its checkpoint"
        );
    }

    #[tokio::test]
    async fn resumable_retains_checkpoint_recording_only_successes_on_partial_failure() {
        use crate::store::MemorySessionStore;
        let store: Arc<dyn SessionStore> = Arc::new(MemorySessionStore::new());
        // EchoExecutor fails the agent named "fail".
        let exec: Arc<dyn AgentExecutor> = Arc::new(EchoExecutor::new());
        let specs = vec![
            AgentStepSpec::new("ok", "explore", "d", "p"),
            AgentStepSpec::new("bad", "fail", "d", "p"),
        ];

        let out =
            execute_steps_parallel_resumable(exec, specs, "wf-2", Arc::clone(&store), None).await;
        assert!(out[0].success);
        assert!(!out[1].success);

        // Not all succeeded → checkpoint retained, recording only the success
        // so the failed step retries on resume.
        let cp = store
            .load_workflow_checkpoint("wf-2")
            .await
            .unwrap()
            .expect("checkpoint retained on partial failure");
        let completed = cp.completed();
        assert!(completed.contains_key("ok"), "succeeded step is recorded");
        assert!(
            !completed.contains_key("bad"),
            "failed step is NOT recorded → it retries on resume"
        );
    }

    struct ZeroHintExecutor;
    #[async_trait]
    impl AgentExecutor for ZeroHintExecutor {
        async fn execute_step(
            &self,
            spec: AgentStepSpec,
            _event_tx: Option<broadcast::Sender<AgentEvent>>,
        ) -> StepOutcome {
            StepOutcome {
                task_id: spec.task_id.clone(),
                session_id: format!("task-run-{}", spec.task_id),
                agent: spec.agent.clone(),
                output: "ok".to_string(),
                success: true,
                structured: None,
            }
        }
        fn concurrency_hint(&self) -> usize {
            0
        }
    }

    #[tokio::test]
    async fn empty_inputs_return_empty() {
        let exec: Arc<dyn AgentExecutor> = Arc::new(EchoExecutor::new());
        assert!(
            crate::orchestration::execute_steps_parallel(Arc::clone(&exec), vec![], None)
                .await
                .is_empty()
        );
        let stages: Vec<PipelineStage<&str>> =
            vec![stage(|_p: Option<&StepOutcome>, item: &&str| {
                Some(AgentStepSpec::new("s", "explore", "d", *item))
            })];
        assert!(execute_pipeline(exec, Vec::<&str>::new(), stages, None)
            .await
            .is_empty());
    }

    #[tokio::test]
    async fn zero_concurrency_hint_still_makes_progress() {
        // The .max(1) clamp in run_ordered_parallel_with_limit keeps a 0-hint
        // executor serialized-but-live instead of deadlocking on 0 permits.
        let exec: Arc<dyn AgentExecutor> = Arc::new(ZeroHintExecutor);
        let specs = vec![
            AgentStepSpec::new("a", "explore", "d", "p"),
            AgentStepSpec::new("b", "explore", "d", "p"),
            AgentStepSpec::new("c", "explore", "d", "p"),
        ];
        let out = crate::orchestration::execute_steps_parallel(exec, specs, None).await;
        assert_eq!(
            out.iter().map(|o| o.task_id.as_str()).collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
        assert!(out.iter().all(|o| o.success));
    }

    #[tokio::test]
    async fn pipeline_first_stage_none_yields_none_outcome() {
        let exec: Arc<dyn AgentExecutor> = Arc::new(EchoExecutor::new());
        let stages: Vec<PipelineStage<&str>> =
            vec![stage(|_p: Option<&StepOutcome>, item: &&str| {
                if *item == "skip" {
                    None
                } else {
                    Some(AgentStepSpec::new("s", "explore", "d", *item))
                }
            })];
        let out = execute_pipeline(exec, vec!["skip", "run"], stages, None).await;
        assert!(
            out[0].is_none(),
            "a first-stage None yields a None outcome (chain never started)"
        );
        assert!(out[1].as_ref().unwrap().success);
    }

    fn cached(task_id: &str, agent: &str, output: &str) -> StepOutcome {
        StepOutcome {
            task_id: task_id.to_string(),
            session_id: format!("task-run-{task_id}"),
            agent: agent.to_string(),
            output: output.to_string(),
            success: true,
            structured: None,
        }
    }

    #[tokio::test]
    async fn resumable_reruns_all_when_checkpoint_load_errors() {
        use crate::store::MemorySessionStore;
        let store: Arc<dyn SessionStore> = Arc::new(MemorySessionStore::new());

        // A checkpoint written by a *newer*, incompatible schema version: the
        // store rejects it on load. The resumable combinator must treat that as
        // no prior progress and re-run everything (fail-safe), not panic or
        // silently resume from state it can't read.
        let mut done = std::collections::HashMap::new();
        done.insert("a".to_string(), cached("a", "explore", "old"));
        let mut cp = WorkflowCheckpoint::from_completed("wf-err", &done, 1);
        cp.schema_version = crate::orchestration::WORKFLOW_CHECKPOINT_SCHEMA_VERSION + 1;
        store.save_workflow_checkpoint("wf-err", &cp).await.unwrap();

        let ran = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let exec: Arc<dyn AgentExecutor> = Arc::new(RecordingExecutor {
            ran: Arc::clone(&ran),
        });
        let specs = vec![
            AgentStepSpec::new("a", "explore", "d", "pa"),
            AgentStepSpec::new("b", "review", "d", "pb"),
        ];
        let out =
            execute_steps_parallel_resumable(exec, specs, "wf-err", Arc::clone(&store), None).await;

        let mut ran_ids = ran.lock().await.clone();
        ran_ids.sort();
        assert_eq!(
            ran_ids,
            vec!["a".to_string(), "b".to_string()],
            "an unreadable (future-version) checkpoint is ignored → all steps re-run"
        );
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|o| o.success));
    }

    #[tokio::test]
    async fn resumable_ignores_checkpointed_steps_absent_from_new_specs() {
        use crate::store::MemorySessionStore;
        let store: Arc<dyn SessionStore> = Arc::new(MemorySessionStore::new());

        // Prior checkpoint completed {a, b}; the new run drops "a", reorders,
        // and adds "c". Output follows the NEW specs; "b" is reused; the stale
        // "a" simply doesn't appear; only "c" actually runs.
        let mut done = std::collections::HashMap::new();
        done.insert("a".to_string(), cached("a", "explore", "cached-a"));
        done.insert("b".to_string(), cached("b", "review", "cached-b"));
        store
            .save_workflow_checkpoint(
                "wf-x",
                &WorkflowCheckpoint::from_completed("wf-x", &done, 1),
            )
            .await
            .unwrap();

        let ran = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let exec: Arc<dyn AgentExecutor> = Arc::new(RecordingExecutor {
            ran: Arc::clone(&ran),
        });
        let specs = vec![
            AgentStepSpec::new("b", "review", "d", "pb"),
            AgentStepSpec::new("c", "plan", "d", "pc"),
        ];
        let out =
            execute_steps_parallel_resumable(exec, specs, "wf-x", Arc::clone(&store), None).await;

        assert_eq!(
            *ran.lock().await,
            vec!["c".to_string()],
            "cached b reused, stale a dropped, only new c runs"
        );
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].task_id, "b");
        assert_eq!(out[0].output, "cached-b");
        assert_eq!(out[1].task_id, "c");
        assert!(out.iter().all(|o| o.success));
    }

    #[tokio::test]
    async fn loop_stops_when_predicate_says_stop() {
        let exec: Arc<dyn AgentExecutor> = Arc::new(EchoExecutor::new());
        let mut rounds = 0;
        let out = crate::orchestration::execute_loop(
            exec,
            vec![AgentStepSpec::new("r0", "explore", "d", "p")],
            10,
            None,
            |outcomes| {
                rounds += 1;
                // Continue twice, then stop on the third round.
                if rounds < 3 {
                    LoopDecision::Continue(vec![AgentStepSpec::new(
                        format!("r{rounds}"),
                        "explore",
                        "d",
                        outcomes[0].output.clone(),
                    )])
                } else {
                    LoopDecision::Stop
                }
            },
        )
        .await;
        assert_eq!(rounds, 3, "predicate saw exactly three rounds");
        assert_eq!(out.len(), 1, "returns the last round's outcomes");
        assert!(out[0].success);
    }

    #[tokio::test]
    async fn loop_is_hard_capped_by_max_iterations() {
        let exec: Arc<dyn AgentExecutor> = Arc::new(EchoExecutor::new());
        let mut rounds = 0;
        // A predicate that NEVER stops — only max_iterations terminates it.
        let _ = crate::orchestration::execute_loop(
            exec,
            vec![AgentStepSpec::new("r", "explore", "d", "p")],
            3,
            None,
            |_outcomes| {
                rounds += 1;
                LoopDecision::Continue(vec![AgentStepSpec::new("r", "explore", "d", "p")])
            },
        )
        .await;
        assert_eq!(
            rounds, 3,
            "max_iterations is a hard cap on a never-stopping predicate"
        );
    }

    #[tokio::test]
    async fn loop_with_empty_initial_runs_nothing() {
        let exec: Arc<dyn AgentExecutor> = Arc::new(EchoExecutor::new());
        let mut called = false;
        let out = crate::orchestration::execute_loop(exec, vec![], 5, None, |_| {
            called = true;
            LoopDecision::Stop
        })
        .await;
        assert!(out.is_empty());
        assert!(!called, "predicate is not invoked when there is no work");
    }

    #[tokio::test]
    async fn loop_stops_when_predicate_requests_no_further_specs() {
        // Continue with an empty spec set ends the loop (no work left = dry).
        let exec: Arc<dyn AgentExecutor> = Arc::new(EchoExecutor::new());
        let mut rounds = 0;
        let out = crate::orchestration::execute_loop(
            exec,
            vec![AgentStepSpec::new("r0", "explore", "d", "p")],
            10,
            None,
            |_| {
                rounds += 1;
                LoopDecision::Continue(vec![]) // nothing more to do → loop ends
            },
        )
        .await;
        assert_eq!(rounds, 1);
        assert_eq!(out.len(), 1, "the completed round is still returned");
    }
}
