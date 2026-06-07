//! Real-LLM integration tests for the programmable Workflow facade, the
//! `execute_loop` combinator, and the shared `WorkflowBudget` — the surface the
//! mock-backed unit tests can't fully exercise (real child-agent loops, real
//! token accounting feeding the shared ledger).
//!
//! `#[ignore]` — requires a live provider in `.a3s/config.acl`. Run:
//!
//! ```bash
//! A3S_CONFIG_FILE=/abs/path/.a3s/config.acl \
//!   cargo test -p a3s-code-core --test test_workflow_facade_real_llm -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use a3s_code_core::config::CodeConfig;
use a3s_code_core::llm::create_client_with_config;
use a3s_code_core::orchestration::{
    execute_loop, AgentExecutor, AgentStepSpec, LoopDecision, Workflow, WorkflowEvent,
};
use a3s_code_core::subagent::AgentRegistry;
use a3s_code_core::tools::TaskExecutor;
use a3s_code_core::{Agent, AgentSession};

fn repo_config_path() -> PathBuf {
    std::env::var_os("A3S_CONFIG_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../..")
                .join(".a3s/config.acl")
        })
}

/// A bare real-LLM executor over a throwaway workspace. Keep the returned guard
/// in scope so the temp dir is cleaned up (no stray temp files).
fn real_executor() -> (Arc<dyn AgentExecutor>, tempfile::TempDir) {
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
    (Arc::new(executor), workspace)
}

/// A real-LLM session built from the repo config, over a throwaway workspace.
async fn real_session() -> (AgentSession, tempfile::TempDir) {
    let path = repo_config_path();
    let config = CodeConfig::from_file(&path)
        .unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()));
    let agent = Agent::from_config(config)
        .await
        .expect("build agent from real config");
    let workspace = tempfile::tempdir().expect("temp workspace");
    let session = agent
        .session(workspace.path().to_string_lossy().to_string(), None)
        .expect("create session");
    (session, workspace)
}

/// The facade's `phase` runs a real fan-out and emits PhaseStart/PhaseEnd
/// milestones on the WorkflowEvent stream.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires real provider credentials and network access"]
async fn real_workflow_phase_runs_and_emits_milestones() {
    let (exec, _ws) = real_executor();
    let wf = Workflow::builder(exec).build();
    let mut rx = wf.subscribe();

    let out = wf
        .phase(
            "probe",
            vec![AgentStepSpec::new(
                "g1",
                "general",
                "echo",
                "Reply with exactly the word READY.",
            )
            .with_max_steps(2)],
        )
        .await;

    assert_eq!(out.len(), 1);
    assert!(out[0].success, "phase step succeeded: {}", out[0].output);

    let start = rx.recv().await.expect("PhaseStart milestone");
    assert!(
        matches!(start, WorkflowEvent::PhaseStart { ref name, step_count, .. } if name == "probe" && step_count == 1),
        "got: {start:?}"
    );
    let end = rx.recv().await.expect("PhaseEnd milestone");
    assert!(
        matches!(end, WorkflowEvent::PhaseEnd { succeeded, failed, .. } if succeeded == 1 && failed == 0),
        "got: {end:?}"
    );
}

/// `execute_loop` runs real rounds and `max_iterations` hard-caps a predicate
/// that never returns `Stop`.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires real provider credentials and network access"]
async fn real_workflow_loop_is_hard_capped() {
    let (exec, _ws) = real_executor();
    let mut rounds = 0;
    let step = || {
        AgentStepSpec::new("r", "general", "echo", "Reply with exactly the word OK.")
            .with_max_steps(2)
    };

    let _ = execute_loop(exec, vec![step()], 2, None, |_outcomes| {
        rounds += 1;
        // Never stop voluntarily — only max_iterations terminates us.
        LoopDecision::Continue(vec![step()])
    })
    .await;

    assert_eq!(rounds, 2, "max_iterations caps a never-stopping predicate");
}

/// `session.workflow().agent(..)` spawns a real child agent and its token usage
/// feeds the shared (uncapped) ledger.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires real provider credentials and network access"]
async fn real_session_workflow_runs_child_and_accumulates_budget() {
    let (session, _ws) = real_session().await;
    let wf = session.workflow();

    let outcome = wf
        .agent(
            AgentStepSpec::new("t1", "general", "echo", "Reply with exactly the word DONE.")
                .with_max_steps(2),
        )
        .await;

    assert!(outcome.success, "child step succeeded: {}", outcome.output);
    let snap = wf.budget_snapshot().expect("workflow has a shared ledger");
    assert!(
        snap.consumed_tokens > 0,
        "the child's real LLM usage fed the shared workflow ledger"
    );
    assert_eq!(snap.limit_tokens, None, "this workflow is uncapped");
}

/// A tiny token cap is enforced through the shared guard: the first sequential
/// step runs (ledger starts empty), and a step started after the ledger is
/// exhausted is denied (surfaces as a failed outcome). Sequential on purpose so
/// the assertion is deterministic (no in-flight overshoot race).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires real provider credentials and network access"]
async fn real_session_workflow_budget_denies_after_cap() {
    let (session, _ws) = real_session().await;
    let wf = session.workflow_with_token_budget(Some(1));

    let first = wf
        .agent(
            AgentStepSpec::new("b1", "general", "echo", "Reply with exactly the word ONE.")
                .with_max_steps(2),
        )
        .await;
    assert!(
        first.success,
        "first step runs before the ledger is exhausted: {}",
        first.output
    );
    assert!(
        wf.budget_snapshot().unwrap().consumed_tokens >= 1,
        "the first step recorded usage into the shared ledger"
    );

    let second = wf
        .agent(
            AgentStepSpec::new("b2", "general", "echo", "Reply with exactly the word TWO.")
                .with_max_steps(2),
        )
        .await;
    assert!(
        !second.success,
        "a step started after the cap is denied (budget exhausted): {}",
        second.output
    );
}
