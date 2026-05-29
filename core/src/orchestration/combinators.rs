//! Orchestration combinators built on the [`AgentExecutor`] seam.
//!
//! [`execute_steps_parallel`](super::execute_steps_parallel) (in `executor`)
//! is the barrier (`parallel`) primitive. This module adds `pipeline`: the
//! one genuinely new scheduling shape, where each item flows through a chain
//! of stages independently — no barrier between stages.

use super::executor::{AgentExecutor, AgentStepSpec, StepOutcome};
use crate::agent::AgentEvent;
use crate::ordered_parallel::run_ordered_parallel_with_limit;
use std::sync::Arc;
use tokio::sync::broadcast;

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
}
