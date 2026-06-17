//! Real-LLM integration tests for the serve layer (filesystem-first agents).
//!
//! `#[ignore]` — requires a live provider in `.a3s/config.acl`. Run:
//!
//! ```bash
//! cargo test -p a3s-code-core --features serve \
//!   --test test_serve_agent_dir_real_llm -- --ignored --nocapture
//! ```
//!
//! These exercise the cross-module path the in-crate unit tests stub out: a cron
//! schedule firing a FULL harness turn through a real `AgentSession::send`.
#![cfg(feature = "serve")]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use a3s_code_core::config::{AgentDir, CodeConfig, ScheduleSpec};
use a3s_code_core::serve::{serve_agent_dir, ScheduleSink, Scheduler};
use a3s_code_core::{Agent, AgentSession};

/// Same resolution as the other `*_real_llm` tests: `A3S_CONFIG_FILE` or the repo
/// root `.a3s/config.acl`.
fn repo_config_path() -> PathBuf {
    std::env::var_os("A3S_CONFIG_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../..")
                .join(".a3s/config.acl")
        })
}

async fn real_agent() -> Agent {
    let path = repo_config_path();
    let config = CodeConfig::from_file(&path)
        .unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()));
    Agent::from_config(config)
        .await
        .expect("build agent from real config")
}

/// Drives a schedule's prompt through a real `AgentSession::send` (the same path
/// the serve daemon's internal sink uses) and records each completed turn's text.
struct RecordingSink {
    session: Arc<AgentSession>,
    fires: Arc<Mutex<Vec<String>>>,
}

#[async_trait::async_trait]
impl ScheduleSink for RecordingSink {
    async fn fire(&self, spec: &ScheduleSpec) {
        match self.session.send(&spec.prompt, None).await {
            Ok(result) => self.fires.lock().unwrap().push(result.text),
            Err(e) => panic!("scheduled real turn failed: {e}"),
        }
    }
}

#[tokio::test]
#[ignore = "requires a live provider in .a3s/config.acl"]
async fn schedule_fires_a_real_harness_turn() {
    let agent = real_agent().await;
    let workspace = tempfile::tempdir().expect("temp workspace");
    let session = Arc::new(
        agent
            .session(workspace.path().to_string_lossy().to_string(), None)
            .expect("session"),
    );
    let fires = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::new(RecordingSink {
        session,
        fires: Arc::clone(&fires),
    });

    // Every-second cron → at least one fire within the window. Each fire is a FULL
    // harness turn (real LLM), exactly as serve_agent_dir does internally.
    let scheduler = Scheduler::new([ScheduleSpec {
        name: "tick".to_string(),
        cron: "* * * * * *".to_string(),
        prompt: "Reply with exactly one word: PONG".to_string(),
        enabled: true,
    }])
    .expect("scheduler");

    let cancel = CancellationToken::new();
    let handle = tokio::spawn(scheduler.run(sink, cancel.clone()));
    tokio::time::sleep(Duration::from_secs(12)).await;
    cancel.cancel();
    let _ = handle.await;

    let recorded = fires.lock().unwrap();
    assert!(
        !recorded.is_empty(),
        "the schedule should have fired at least one real harness turn"
    );
    assert!(
        recorded.iter().any(|t| !t.trim().is_empty()),
        "a real LLM turn should return non-empty text; got {recorded:?}"
    );
}

#[tokio::test]
#[ignore = "requires a live provider in .a3s/config.acl"]
async fn serve_agent_dir_runs_a_real_schedule_and_stops() {
    let agent = real_agent().await;

    // A minimal agent dir: instructions + one every-second schedule.
    let dir = tempfile::tempdir().expect("agent dir");
    std::fs::write(
        dir.path().join("instructions.md"),
        "You are a terse test agent. Answer in one word.",
    )
    .unwrap();
    std::fs::create_dir_all(dir.path().join("schedules")).unwrap();
    std::fs::write(
        dir.path().join("schedules/tick.md"),
        "---\ncron: \"* * * * * *\"\nname: tick\n---\nReply with exactly one word: PONG",
    )
    .unwrap();
    let agent_dir = AgentDir::load(dir.path()).expect("load agent dir");

    let workspace = tempfile::tempdir().expect("workspace");
    let cancel = CancellationToken::new();
    {
        let cancel = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(8)).await;
            cancel.cancel();
        });
    }

    // Runs until cancelled; a real schedule fire happens inside, then a clean stop.
    serve_agent_dir(
        &agent,
        &agent_dir,
        workspace.path().to_string_lossy().to_string(),
        None,
        cancel,
    )
    .await
    .expect("serve_agent_dir should run a real fire and stop cleanly");
}
