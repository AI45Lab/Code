//! Integration tests for two prompt/runtime improvements ported from the
//! claude-fable-5 prompt design:
//!
//! 1. Tool invocations must never log argument *values* (secret hygiene) — backs
//!    the new "never log secrets" Boundaries rule.
//! 2. The safety `## Boundaries` block must appear in the assembled system prompt
//!    for every agent style (single-source injection in `build_with_style`).
//!
//! These exercise the public crate API end to end (real `ToolExecutor::execute`
//! with a live `tracing` subscriber, and real `SystemPromptSlots::build`).
//!
//! Run with: cargo test --test test_prompt_boundaries_and_log_redaction

use std::io::Write;
use std::sync::{Arc, Mutex};

use a3s_code_core::tools::ToolExecutor;
use a3s_code_core::{AgentStyle, SystemPromptSlots};

/// A `tracing` writer that appends all output to a shared buffer for assertions.
#[derive(Clone)]
struct SharedBuf(Arc<Mutex<Vec<u8>>>);

impl Write for SharedBuf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn tool_invocation_log_omits_secret_arg_values() {
    let buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    let sink = SharedBuf(buf.clone());

    // Capture INFO-level logs (the always-on, OTLP-exported level). Full args are
    // only emitted at TRACE, which this subscriber intentionally excludes.
    let subscriber = tracing_subscriber::fmt()
        .with_writer(move || sink.clone())
        .with_max_level(tracing::Level::INFO)
        .with_ansi(false)
        .finish();
    let guard = tracing::subscriber::set_default(subscriber);

    let secret = "SUPER_SECRET_AKIA1234567890";
    let executor = ToolExecutor::new("/tmp".to_string());
    // The result is irrelevant: `log_tool_invocation` fires synchronously at the
    // start of `execute()`, before the command runs, so the assertions hold
    // whether or not bash succeeds in this environment.
    let _ = executor
        .execute(
            "bash",
            &serde_json::json!({ "command": format!("echo {secret}") }),
        )
        .await;

    drop(guard);
    let logs = String::from_utf8(buf.lock().unwrap().clone()).unwrap();

    assert!(
        logs.contains("Executing tool: bash"),
        "redacted invocation summary missing; logs: {logs}"
    );
    assert!(
        logs.contains("command"),
        "argument field names should be logged; logs: {logs}"
    );
    assert!(
        logs.contains("bytes"),
        "payload size should be logged; logs: {logs}"
    );
    assert!(
        !logs.contains(secret),
        "SECRET VALUE LEAKED into INFO logs: {logs}"
    );
}

#[test]
fn boundaries_present_in_every_assembled_style_prompt() {
    for style in [
        AgentStyle::GeneralPurpose,
        AgentStyle::Plan,
        AgentStyle::Verification,
        AgentStyle::Explore,
        AgentStyle::CodeReview,
    ] {
        let prompt = SystemPromptSlots::default().with_style(style).build();
        assert!(
            prompt.contains("## Boundaries"),
            "{style:?} prompt missing Boundaries section"
        );
        assert!(
            prompt.contains("untrusted data"),
            "{style:?} prompt missing injection-hygiene rule"
        );
        assert!(
            prompt.contains("secrets"),
            "{style:?} prompt missing secret-handling rule"
        );
    }
}

#[test]
fn custom_response_style_still_strips_default_block() {
    // The response-format strip relies on byte-identical blocks; confirm that
    // adding the Boundaries injection did not break that surgery.
    let prompt = SystemPromptSlots::default()
        .with_response_style("Terse answers only.")
        .build();
    assert!(prompt.contains("Terse answers only."));
    assert!(
        !prompt.contains("keep progress notes brief and useful"),
        "default response-format block was not stripped"
    );
    assert!(prompt.contains("## Boundaries"));
}
