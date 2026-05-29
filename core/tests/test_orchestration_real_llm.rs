//! Real-LLM validation of the orchestration layer (Workflow phases).
//!
//! `#[ignore]` — requires a live provider in `.a3s/config.acl`. Run:
//!
//! ```bash
//! A3S_CONFIG_FILE=/abs/path/.a3s/config.acl \
//!   cargo test -p a3s-code-core --test test_orchestration_real_llm -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use a3s_code_core::config::CodeConfig;
use a3s_code_core::llm::create_client_with_config;
use a3s_code_core::orchestration::{
    execute_pipeline, execute_steps_parallel, execute_steps_parallel_resumable, AgentExecutor,
    AgentStepSpec, PipelineStage, StepOutcome, WorkflowCheckpoint,
};
use a3s_code_core::store::{MemorySessionStore, SessionStore};
use a3s_code_core::subagent::AgentRegistry;
use a3s_code_core::tools::TaskExecutor;

fn repo_config_path() -> PathBuf {
    std::env::var_os("A3S_CONFIG_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../..")
                .join(".a3s/config.acl")
        })
}

/// Returns the executor plus the workspace guard — keep the guard in scope so
/// the temp dir is cleaned up when the test ends (no stray temp files).
fn local_executor() -> (TaskExecutor, tempfile::TempDir) {
    let path = repo_config_path();
    let config = CodeConfig::from_file(&path)
        .unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()));
    let llm_client =
        create_client_with_config(config.default_llm_config().expect("default llm config"));
    let workspace = tempfile::tempdir().expect("temp workspace");
    let executor = TaskExecutor::new(
        Arc::new(AgentRegistry::new()),
        llm_client,
        workspace.path().to_string_lossy().to_string(),
    );
    (executor, workspace)
}

/// Phase 2: a step carrying an `output_schema` runs against a live model and
/// returns a value validated against the schema in `StepOutcome::structured`.
/// Mock clients can't validate the structured-output coercion against a real
/// provider's tool-calling behavior — this does.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires real provider credentials and network access"]
async fn real_execute_step_with_schema_returns_validated_object() {
    let (executor, _workspace) = local_executor();

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "language": { "type": "string" },
            "is_systems_language": { "type": "boolean" }
        },
        "required": ["language", "is_systems_language"]
    });
    let spec = AgentStepSpec::new(
        "real-schema-1",
        "general",
        "classify language",
        "Briefly describe the Rust programming language and whether it is a systems language.",
    )
    .with_output_schema(schema)
    .with_max_steps(2);

    let outcome = executor.execute_step(spec, None).await;

    assert!(
        outcome.success,
        "schema'd step should succeed: {}",
        outcome.output
    );
    let object = outcome
        .structured
        .expect("a schema'd step must return a validated structured object");
    assert!(
        object.get("language").and_then(|v| v.as_str()).is_some(),
        "object has a string `language`: {object}"
    );
    assert!(
        object
            .get("is_systems_language")
            .map(|v| v.is_boolean())
            .unwrap_or(false),
        "object has a boolean `is_systems_language`: {object}"
    );
}

/// Phase 3: a two-stage pipeline chains live agents — stage 2's prompt is
/// derived from stage 1's output, with no barrier between them.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires real provider credentials and network access"]
async fn real_pipeline_chains_two_agent_stages() {
    let (executor, _workspace) = local_executor();
    let exec: Arc<dyn AgentExecutor> = Arc::new(executor);

    let stages: Vec<PipelineStage<&str>> = vec![
        Arc::new(|_prev: Option<&StepOutcome>, topic: &&str| {
            Some(
                AgentStepSpec::new(
                    "real-p1",
                    "general",
                    "summarize",
                    format!("In one sentence, what is {topic}?"),
                )
                .with_max_steps(2),
            )
        }),
        Arc::new(|prev: Option<&StepOutcome>, _topic: &&str| {
            let summary = prev.map(|o| o.output.clone()).unwrap_or_default();
            Some(
                AgentStepSpec::new(
                    "real-p2",
                    "general",
                    "classify",
                    format!(
                        "Reply with exactly one word, YES or NO: does this describe a \
                         programming language?\n\nText: {summary}"
                    ),
                )
                .with_max_steps(2),
            )
        }),
    ];

    let out = execute_pipeline(exec, vec!["the Rust programming language"], stages, None).await;

    assert_eq!(out.len(), 1);
    let final_outcome = out[0].as_ref().expect("the chain produced a final outcome");
    assert!(
        final_outcome.success,
        "pipeline chain succeeded: {}",
        final_outcome.output
    );
    // Reaching stage 2 is the chaining proof: stage 2 only runs when stage 1
    // succeeded, and its prompt is built from stage 1's output by the closure.
    // We don't assert on stage 2's output *content* — some models return an
    // empty final turn for a one-word YES/NO instruction. Deterministic
    // prev-output visibility is covered by the mock test
    // `each_item_chains_through_stages_and_later_stages_see_prior_output`.
    assert_eq!(
        final_outcome.task_id, "real-p2",
        "the chain ran through to stage 2"
    );
}

/// Phase 4: a resumable workflow runs real steps, journals progress to a
/// store, and clears its checkpoint on full success.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires real provider credentials and network access"]
async fn real_resumable_workflow_runs_and_clears_checkpoint() {
    let (executor, _workspace) = local_executor();
    let exec: Arc<dyn AgentExecutor> = Arc::new(executor);
    let store: Arc<dyn SessionStore> = Arc::new(MemorySessionStore::new());

    let specs = vec![
        AgentStepSpec::new("rw-1", "general", "q1", "Reply with one word: ready").with_max_steps(2),
        AgentStepSpec::new("rw-2", "general", "q2", "Reply with one word: go").with_max_steps(2),
    ];

    let out =
        execute_steps_parallel_resumable(exec, specs, "real-wf", Arc::clone(&store), None).await;

    assert_eq!(out.len(), 2);
    assert!(
        out.iter().all(|o| o.success),
        "both steps succeed: {:?}",
        out.iter().map(|o| &o.output).collect::<Vec<_>>()
    );
    assert!(
        store
            .load_workflow_checkpoint("real-wf")
            .await
            .unwrap()
            .is_none(),
        "a fully-succeeded workflow clears its checkpoint"
    );
}

/// Wraps a real executor to record which task_ids actually reached it — used to
/// prove that a resumed workflow skips already-completed steps (no live call).
struct CountingExecutor {
    inner: Arc<dyn AgentExecutor>,
    ran: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl AgentExecutor for CountingExecutor {
    async fn execute_step(
        &self,
        spec: AgentStepSpec,
        event_tx: Option<tokio::sync::broadcast::Sender<a3s_code_core::AgentEvent>>,
    ) -> StepOutcome {
        self.ran.lock().await.push(spec.task_id.clone());
        self.inner.execute_step(spec, event_tx).await
    }
    fn concurrency_hint(&self) -> usize {
        self.inner.concurrency_hint()
    }
}

/// Phase 1 live: pure `execute_steps_parallel` fan-out of several distinct
/// agents — order preserved, each branch independent (its own sentinel).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires real provider credentials and network access"]
async fn real_parallel_fans_out_independent_agents() {
    let (executor, _ws) = local_executor();
    let exec: Arc<dyn AgentExecutor> = Arc::new(executor);

    let specs = vec![
        AgentStepSpec::new(
            "p-alpha",
            "general",
            "echo",
            "Reply with exactly the word ALPHA.",
        )
        .with_max_steps(2),
        AgentStepSpec::new(
            "p-bravo",
            "general",
            "echo",
            "Reply with exactly the word BRAVO.",
        )
        .with_max_steps(2),
        AgentStepSpec::new(
            "p-charlie",
            "general",
            "echo",
            "Reply with exactly the word CHARLIE.",
        )
        .with_max_steps(2),
    ];
    let out = execute_steps_parallel(exec, specs, None).await;

    assert_eq!(out.len(), 3, "one outcome per spec");
    assert_eq!(
        out.iter().map(|o| o.task_id.as_str()).collect::<Vec<_>>(),
        vec!["p-alpha", "p-bravo", "p-charlie"],
        "results preserve input order regardless of completion order"
    );
    assert!(
        out.iter().all(|o| o.success),
        "all branches succeed: {:?}",
        out.iter().map(|o| &o.output).collect::<Vec<_>>()
    );
    // Each branch saw only its own prompt (no cross-contamination).
    assert!(
        out[0].output.contains("ALPHA"),
        "p-alpha: {}",
        out[0].output
    );
    assert!(
        out[1].output.contains("BRAVO"),
        "p-bravo: {}",
        out[1].output
    );
    assert!(
        out[2].output.contains("CHARLIE"),
        "p-charlie: {}",
        out[2].output
    );
}

/// Phase 3 live: a multi-item pipeline — the no-inter-stage-barrier shape with
/// >1 item, each chaining stage 2 off stage 1's live output.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires real provider credentials and network access"]
async fn real_pipeline_runs_multiple_items_concurrently() {
    let (executor, _ws) = local_executor();
    let exec: Arc<dyn AgentExecutor> = Arc::new(executor);

    let stages: Vec<PipelineStage<&str>> = vec![
        Arc::new(|_p: Option<&StepOutcome>, topic: &&str| {
            Some(
                AgentStepSpec::new(
                    "p1",
                    "general",
                    "summarize",
                    format!("In one sentence: {topic}"),
                )
                .with_max_steps(2),
            )
        }),
        Arc::new(|prev: Option<&StepOutcome>, _t: &&str| {
            let s = prev.map(|o| o.output.clone()).unwrap_or_default();
            Some(
                AgentStepSpec::new(
                    "p2",
                    "general",
                    "classify",
                    format!("Reply YES or NO: is the following about software?\n\n{s}"),
                )
                .with_max_steps(2),
            )
        }),
    ];
    let items = vec![
        "the Rust programming language",
        "the Pacific Ocean",
        "the Linux kernel",
    ];
    let out = execute_pipeline(exec, items, stages, None).await;

    assert_eq!(out.len(), 3, "one result per item, order preserved");
    for (i, r) in out.iter().enumerate() {
        let outcome = r
            .as_ref()
            .unwrap_or_else(|| panic!("item {i} produced no outcome"));
        assert!(
            outcome.success,
            "item {i} chain succeeded: {}",
            outcome.output
        );
        // Reaching stage 2 proves chaining (see the 1-item test's note);
        // output content isn't asserted (model nondeterminism).
        assert_eq!(outcome.task_id, "p2", "item {i} ran through to stage 2");
    }
}

/// Phase 4 live: the RESUME path — a workflow with a pre-existing checkpoint
/// re-runs ONLY the not-yet-completed steps against the real model; the
/// completed step is served from the checkpoint with no live call.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires real provider credentials and network access"]
async fn real_resumable_workflow_resumes_skipping_completed_steps() {
    let (executor, _ws) = local_executor();
    let store: Arc<dyn SessionStore> = Arc::new(MemorySessionStore::new());

    // Pre-seed: step "rw-1" already completed on a prior run.
    let mut done = HashMap::new();
    done.insert(
        "rw-1".to_string(),
        StepOutcome {
            task_id: "rw-1".into(),
            session_id: "task-run-rw-1".into(),
            agent: "general".into(),
            output: "CACHED-SENTINEL".into(),
            success: true,
            structured: None,
        },
    );
    store
        .save_workflow_checkpoint(
            "real-W",
            &WorkflowCheckpoint::from_completed("real-W", &done, 1),
        )
        .await
        .unwrap();

    let ran = Arc::new(Mutex::new(Vec::new()));
    let counting: Arc<dyn AgentExecutor> = Arc::new(CountingExecutor {
        inner: Arc::new(executor),
        ran: Arc::clone(&ran),
    });
    let specs = vec![
        AgentStepSpec::new("rw-1", "general", "cached", "unused — should be skipped")
            .with_max_steps(2),
        AgentStepSpec::new(
            "rw-2",
            "general",
            "live",
            "Reply with exactly the word DONE.",
        )
        .with_max_steps(2),
    ];
    let out =
        execute_steps_parallel_resumable(counting, specs, "real-W", Arc::clone(&store), None).await;

    assert_eq!(
        *ran.lock().await,
        vec!["rw-2".to_string()],
        "only the uncompleted step hit the real executor; rw-1 came from the checkpoint"
    );
    assert_eq!(out[0].task_id, "rw-1");
    assert_eq!(
        out[0].output, "CACHED-SENTINEL",
        "completed step returns its cached outcome, no second LLM call"
    );
    assert!(
        out[1].success,
        "the resumed step ran live: {}",
        out[1].output
    );
    assert!(
        store
            .load_workflow_checkpoint("real-W")
            .await
            .unwrap()
            .is_none(),
        "the now-complete workflow clears its checkpoint"
    );
}

/// Phase 2 live: a step with a NESTED/array schema returns a validated object
/// through the real tool-mode coercion + repair loop.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires real provider credentials and network access"]
async fn real_execute_step_with_nested_schema_returns_validated_object() {
    let (executor, _ws) = local_executor();
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "languages": {
                "type": "array",
                "minItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "compiled": { "type": "boolean" }
                    },
                    "required": ["name", "compiled"]
                }
            }
        },
        "required": ["languages"]
    });
    let spec = AgentStepSpec::new(
        "nested",
        "general",
        "list languages",
        "List two well-known systems programming languages and whether each is compiled.",
    )
    .with_output_schema(schema)
    .with_max_steps(2);

    let outcome = executor.execute_step(spec, None).await;

    assert!(
        outcome.success,
        "nested-schema step should succeed: {}",
        outcome.output
    );
    let object = outcome.structured.expect("validated structured object");
    let languages = object
        .get("languages")
        .and_then(|v| v.as_array())
        .expect("languages is an array");
    assert!(languages.len() >= 2, "at least 2 languages: {object}");
    for lang in languages {
        assert!(
            lang.get("name").and_then(|v| v.as_str()).is_some(),
            "each language has a string name: {lang}"
        );
        assert!(
            lang.get("compiled")
                .map(|v| v.is_boolean())
                .unwrap_or(false),
            "each language has a boolean compiled: {lang}"
        );
    }
}
