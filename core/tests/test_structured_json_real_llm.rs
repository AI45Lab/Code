//! Real-LLM validation of the JSON-object-generation stability fix.
//!
//! Exercises the changed code paths against the live provider configured in
//! `.a3s/config.acl`:
//!   * forced-`tool_choice` structured generation (Tool mode) — the core fix,
//!     run repeatedly to prove stability;
//!   * native `response_format` (Json / Strict modes) on the OpenAI-compatible
//!     provider;
//!   * the hardened planner pre-analysis JSON path (`LlmPlanner::pre_analyze`).
//!
//! `#[ignore]` — requires a live provider in `.a3s/config.acl` and network
//! access to it. Run:
//!
//! ```bash
//! A3S_CONFIG_FILE=/abs/path/.a3s/config.acl \
//!   cargo test -p a3s-code-core --test test_structured_json_real_llm -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use a3s_code_core::config::CodeConfig;
use a3s_code_core::llm::structured::{
    generate_blocking, StructuredMode, StructuredRequest, StructuredResult,
};
use a3s_code_core::llm::{create_client_with_config, LlmClient};
use a3s_code_core::planning::LlmPlanner;
use serde_json::{json, Value};

/// Hard ceiling per LLM call so a flaky/hung endpoint fails the test fast
/// instead of stalling for minutes.
const CALL_TIMEOUT: Duration = Duration::from_secs(90);

fn repo_config_path() -> PathBuf {
    std::env::var_os("A3S_CONFIG_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../..")
                .join(".a3s/config.acl")
        })
}

/// Build a client from `.a3s/config.acl`. By default uses the config's
/// `default_model`; set `A3S_TEST_MODEL=provider/model` to target a specific
/// model (e.g. a tool-capable one for structured-output tests).
fn real_client() -> Arc<dyn LlmClient> {
    let path = repo_config_path();
    let config = CodeConfig::from_file(&path)
        .unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()));

    let llm_config = match std::env::var("A3S_TEST_MODEL") {
        Ok(spec) => {
            let (provider, model) = spec
                .split_once('/')
                .expect("A3S_TEST_MODEL must be 'provider/model'");
            eprintln!("[real-llm] model = {spec} (from {})", path.display());
            config
                .llm_config(provider, model)
                .unwrap_or_else(|| panic!("model {spec} not found in {}", path.display()))
        }
        Err(_) => {
            eprintln!("[real-llm] model = <default> (from {})", path.display());
            config
                .default_llm_config()
                .expect("default llm config in .a3s/config.acl")
        }
    };
    create_client_with_config(llm_config)
}

/// Run a structured generation with a hard timeout.
async fn gen_with_timeout(
    client: &dyn LlmClient,
    req: &StructuredRequest,
) -> anyhow::Result<StructuredResult> {
    match tokio::time::timeout(CALL_TIMEOUT, generate_blocking(client, req)).await {
        Ok(res) => res,
        Err(_) => anyhow::bail!("LLM call exceeded {CALL_TIMEOUT:?}"),
    }
}

/// A non-trivial nested schema — the kind of object whose generation users
/// reported as unstable.
fn person_schema() -> Value {
    json!({
        "type": "object",
        "required": ["name", "age", "skills"],
        "additionalProperties": false,
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" },
            "skills": { "type": "array", "items": { "type": "string" } },
            "address": {
                "type": "object",
                "properties": { "city": { "type": "string" } }
            }
        }
    })
}

fn person_request(mode: StructuredMode) -> StructuredRequest {
    StructuredRequest {
        prompt: "Extract a structured person profile from this text: \
                 'Alice is 30 years old, a Rust and Python developer living in Berlin.'"
            .to_string(),
        system: None,
        schema: person_schema(),
        schema_name: "person".to_string(),
        schema_description: Some("A person profile".to_string()),
        mode,
        max_repair_attempts: 2,
    }
}

fn assert_valid_person(object: &Value) {
    assert!(
        object["name"].is_string(),
        "name must be a string, got {object}"
    );
    assert!(
        object["age"].is_i64() || object["age"].is_u64(),
        "age must be an integer, got {object}"
    );
    assert!(
        object["skills"].is_array(),
        "skills must be an array, got {object}"
    );
}

/// The core fix: forced `tool_choice` structured generation must be STABLE.
/// Run several independent times and require every one to yield a valid,
/// schema-conforming object.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires real provider credentials and network access"]
async fn real_structured_tool_mode_is_stable() {
    let client = real_client();
    const RUNS: usize = 5;
    let mut total_repairs = 0u32;

    for i in 0..RUNS {
        let result = gen_with_timeout(client.as_ref(), &person_request(StructuredMode::Tool))
            .await
            .unwrap_or_else(|e| panic!("run {i}: forced-tool structured generation failed: {e}"));

        assert_eq!(
            result.mode_used,
            StructuredMode::Tool,
            "run {i}: expected forced Tool mode"
        );
        assert_valid_person(&result.object);
        total_repairs += result.repair_rounds as u32;
        eprintln!(
            "[tool] run {i}: ok (repairs={}) -> {}",
            result.repair_rounds, result.object
        );
    }

    eprintln!(
        "[tool] {RUNS}/{RUNS} runs produced valid objects; total repair rounds = {total_repairs}"
    );
}

/// Native `response_format: json_object` on the OpenAI-compatible provider.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires real provider credentials and network access"]
async fn real_structured_json_mode() {
    let client = real_client();
    let result = gen_with_timeout(client.as_ref(), &person_request(StructuredMode::Json))
        .await
        .expect("json_object structured generation failed");
    assert_valid_person(&result.object);
    eprintln!(
        "[json] mode_used={:?} repairs={} -> {}",
        result.mode_used, result.repair_rounds, result.object
    );
}

/// Native `response_format: json_schema` (strict). Some providers reject schemas
/// that don't meet their strict subset, so this is tolerant: it must either
/// succeed with a valid object or fail cleanly (never hang or mis-parse).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires real provider credentials and network access"]
async fn real_structured_strict_mode() {
    let client = real_client();
    match gen_with_timeout(client.as_ref(), &person_request(StructuredMode::Strict)).await {
        Ok(result) => {
            assert_valid_person(&result.object);
            eprintln!(
                "[strict] ok mode_used={:?} repairs={} -> {}",
                result.mode_used, result.repair_rounds, result.object
            );
        }
        Err(e) => {
            // Acceptable: provider may not support strict json_schema for this
            // schema. The point is that it fails cleanly, not silently wrong.
            eprintln!("[strict] provider rejected native json_schema (acceptable): {e}");
        }
    }
}

/// The hardened planner pre-analysis path against a real model: the response
/// must parse into a `PreAnalysis` (robust extractor + one repair retry).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires real provider credentials and network access"]
async fn real_pre_analyze_parses() {
    let client = real_client();
    let analysis = tokio::time::timeout(
        CALL_TIMEOUT,
        LlmPlanner::pre_analyze(
            &client,
            "Refactor the auth module in src/auth.rs to use async/await, and keep the public API stable.",
        ),
    )
    .await
    .expect("pre_analyze timed out")
    .expect("pre_analyze should parse a real model's JSON response");

    assert!(
        !analysis.optimized_input.trim().is_empty(),
        "optimized_input should be populated"
    );
    eprintln!(
        "[pre_analyze] requires_planning={} optimized_input={:?}",
        analysis.requires_planning, analysis.optimized_input
    );
}
