//! Real-LLM validation of the orchestration layer (Workflow phases).
//!
//! `#[ignore]` — requires a live provider in `.a3s/config.acl`. Run:
//!
//! ```bash
//! A3S_CONFIG_FILE=/abs/path/.a3s/config.acl \
//!   cargo test -p a3s-code-core --test test_orchestration_real_llm -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use a3s_code_core::config::CodeConfig;
use a3s_code_core::llm::create_client_with_config;
use a3s_code_core::orchestration::{
    execute_pipeline, AgentExecutor, AgentStepSpec, PipelineStage, StepOutcome,
};
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
    assert_eq!(
        final_outcome.task_id, "real-p2",
        "the returned outcome is the last stage's"
    );
    assert!(
        !final_outcome.output.trim().is_empty(),
        "stage 2 produced output derived from stage 1"
    );
}
