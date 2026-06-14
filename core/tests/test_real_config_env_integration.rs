//! Real-provider integration tests for the 2.0 ACL configuration path.
//!
//! These tests intentionally require explicit opt-in because they call the
//! configured provider in `.a3s/config.acl`.

use std::path::PathBuf;
use std::sync::OnceLock;

use a3s_code_core::config::CodeConfig;
use a3s_code_core::llm::{create_client_with_config, Message};
use a3s_code_core::{Agent, AgentEvent, PlanningMode, RunStatus, SessionOptions};

fn repo_config_path() -> PathBuf {
    std::env::var_os("A3S_CONFIG_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../..")
                .join(".a3s/config.acl")
        })
}

/// Self-contained ACL fixture for the offline resolution tests.
///
/// Hermetic on purpose: the offline tests must NOT read the developer-local,
/// git-ignored `.a3s/config.acl` — its `default_model` and provider secrets vary
/// per machine (which is what previously broke these tests). `env_style_config_file`
/// rewrites the literal `apiKey`/`baseUrl` below into `env(...)` references so the
/// env-injection resolution path is exercised against known, stable values.
const FIXTURE_CONFIG_ACL: &str = r#"default_model = "openai/MiniMax-M2.7-highspeed"

providers "openai" {
  apiKey = "fixture-openai-key"
  baseUrl = "http://fixture.invalid/v1/"

  models "MiniMax-M2.7-highspeed" {
    name = "MiniMax-M2.7-highspeed"
    family = ""
    attachment = false
    reasoning = false
    toolCall = true
    temperature = true
    releaseDate = null
    modalities = { input = ["text"], output = ["text"] }
    cost = { input = 0, output = 0, cacheRead = 0, cacheWrite = 0 }
    limit = { context = 128000, output = 4096 }
  }
}
"#;

fn env_style_config_file() -> tempfile::NamedTempFile {
    let content = FIXTURE_CONFIG_ACL.to_string();

    let mut output = String::new();
    let mut in_openai_provider = false;
    let mut replaced_api_key = false;
    let mut replaced_base_url = false;

    for line in content.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("providers \"openai\"") {
            in_openai_provider = true;
        }

        if in_openai_provider && !replaced_api_key && trimmed.starts_with("apiKey") {
            output.push_str("  apiKey = env(\"A3S_OPENAI_API_KEY\")\n");
            replaced_api_key = true;
            continue;
        }

        if in_openai_provider && !replaced_api_key && trimmed.starts_with("api_key") {
            output.push_str("  apiKey = env(\"A3S_OPENAI_API_KEY\")\n");
            replaced_api_key = true;
            continue;
        }

        if in_openai_provider && !replaced_base_url && trimmed.starts_with("baseUrl") {
            output.push_str("  baseUrl = env(\"A3S_OPENAI_BASE_URL\")\n");
            replaced_base_url = true;
            continue;
        }

        if in_openai_provider && !replaced_base_url && trimmed.starts_with("base_url") {
            output.push_str("  baseUrl = env(\"A3S_OPENAI_BASE_URL\")\n");
            replaced_base_url = true;
            continue;
        }

        output.push_str(line);
        output.push('\n');
    }

    assert!(
        replaced_api_key,
        "openai provider api key line was not found in test config"
    );
    assert!(
        replaced_base_url,
        "openai provider base URL line was not found in test config"
    );

    let mut file = tempfile::Builder::new()
        .prefix("a3s-code-test-config-")
        .suffix(".acl")
        .tempfile()
        .expect("temp env-style config");
    std::io::Write::write_all(&mut file, output.as_bytes()).expect("write env-style config");
    file
}

fn require_env(name: &str) {
    assert!(
        std::env::var_os(name).is_some(),
        "{name} must be injected in the environment before running real integration tests"
    );
}

fn env_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

fn inject_minimax_aliases() {
    if std::env::var_os("A3S_OPENAI_API_KEY").is_none() {
        if let Some(api_key) = std::env::var_os("MINIMAX_API_KEY") {
            std::env::set_var("A3S_OPENAI_API_KEY", api_key);
        }
    }
    if std::env::var_os("A3S_OPENAI_BASE_URL").is_none() {
        if let Some(base_url) = std::env::var_os("MINIMAX_BASE_URL") {
            std::env::set_var("A3S_OPENAI_BASE_URL", base_url);
        }
    }
}

struct EnvVarGuard {
    name: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn set(name: &'static str, value: &'static str) -> Self {
        let previous = std::env::var_os(name);
        std::env::set_var(name, value);
        Self { name, previous }
    }

    fn unset(name: &'static str) -> Self {
        let previous = std::env::var_os(name);
        std::env::remove_var(name);
        Self { name, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(value) = &self.previous {
            std::env::set_var(self.name, value);
        } else {
            std::env::remove_var(self.name);
        }
    }
}

#[test]
fn test_config_acl_minimax_env_injection_resolves_without_network() {
    let _guard = env_lock().blocking_lock();
    let _api_key = EnvVarGuard::set("A3S_OPENAI_API_KEY", "test-minimax-key");
    let _base_url = EnvVarGuard::set("A3S_OPENAI_BASE_URL", "https://minimax.example.test/v1");

    let config_file = env_style_config_file();
    let config_path = config_file.path();
    let config = CodeConfig::from_file(config_path)
        .unwrap_or_else(|err| panic!("failed to load {}: {err}", config_path.display()));

    assert_eq!(
        config.default_model.as_deref(),
        Some("openai/MiniMax-M2.7-highspeed")
    );

    let llm_config = config
        .default_llm_config()
        .expect("default llm config should resolve through env() values");
    assert_eq!(llm_config.provider, "openai");
    assert_eq!(llm_config.model, "MiniMax-M2.7-highspeed");
    assert!(
        llm_config.api_key.expose() == "test-minimax-key",
        "api key did not resolve from A3S_OPENAI_API_KEY"
    );
    assert!(
        llm_config.base_url.as_deref() == Some("https://minimax.example.test/v1"),
        "base URL did not resolve from A3S_OPENAI_BASE_URL"
    );
}

#[test]
fn test_config_acl_minimax_aliases_resolve_without_network() {
    let _guard = env_lock().blocking_lock();
    let _a3s_api_key = EnvVarGuard::unset("A3S_OPENAI_API_KEY");
    let _a3s_base_url = EnvVarGuard::unset("A3S_OPENAI_BASE_URL");
    let _minimax_api_key = EnvVarGuard::set("MINIMAX_API_KEY", "alias-minimax-key");
    let _minimax_base_url = EnvVarGuard::set("MINIMAX_BASE_URL", "https://alias.example.test/v1");

    inject_minimax_aliases();

    let config_file = env_style_config_file();
    let config_path = config_file.path();
    let config = CodeConfig::from_file(config_path)
        .unwrap_or_else(|err| panic!("failed to load {}: {err}", config_path.display()));

    let llm_config = config
        .default_llm_config()
        .expect("default llm config should resolve through MiniMax aliases");
    assert_eq!(llm_config.provider, "openai");
    assert_eq!(llm_config.model, "MiniMax-M2.7-highspeed");
    assert!(
        llm_config.api_key.expose() == "alias-minimax-key",
        "api key did not resolve from MINIMAX_API_KEY alias"
    );
    assert!(
        llm_config.base_url.as_deref() == Some("https://alias.example.test/v1"),
        "base URL did not resolve from MINIMAX_BASE_URL alias"
    );
}

#[tokio::test]
#[ignore = "requires real provider credentials and network access"]
async fn test_config_acl_env_default_llm_completion() {
    let _guard = env_lock().lock().await;
    inject_minimax_aliases();
    require_env("A3S_OPENAI_API_KEY");
    require_env("A3S_OPENAI_BASE_URL");

    let config_path = repo_config_path();
    let config = CodeConfig::from_file(&config_path)
        .unwrap_or_else(|err| panic!("failed to load {}: {err}", config_path.display()));

    assert_eq!(
        config.default_model.as_deref(),
        Some("openai/MiniMax-M2.7-highspeed")
    );

    let llm_config = config
        .default_llm_config()
        .expect("default llm config should resolve through env() values");
    assert_eq!(llm_config.provider, "openai");
    assert_eq!(llm_config.model, "MiniMax-M2.7-highspeed");

    let client = create_client_with_config(llm_config);
    let response = client
        .complete(
            &[Message::user(
                "Return exactly this token and nothing else: A3S_ENV_OK",
            )],
            None,
            &[],
        )
        .await
        .expect("real default model completion should succeed");

    let text = response.text();
    assert!(
        text.contains("A3S_ENV_OK"),
        "expected A3S_ENV_OK in model response, got: {text:?}"
    );
}

#[tokio::test]
#[ignore = "requires real provider credentials and network access"]
async fn test_agent_create_uses_config_acl_env_injection() {
    let _guard = env_lock().lock().await;
    inject_minimax_aliases();
    require_env("A3S_OPENAI_API_KEY");
    require_env("A3S_OPENAI_BASE_URL");

    let config_path = repo_config_path();
    let agent = Agent::create(config_path.to_string_lossy().to_string())
        .await
        .expect("agent should be created from .a3s/config.acl with injected env values");

    let workspace = tempfile::tempdir().expect("temp workspace");
    let session = agent
        .session(workspace.path().to_string_lossy().to_string(), None)
        .expect("session should be created");

    let result = session
        .send(
            "Return exactly this token and nothing else: A3S_AGENT_OK",
            None,
        )
        .await
        .expect("real agent session should complete");

    assert!(
        result.text.contains("A3S_AGENT_OK"),
        "expected A3S_AGENT_OK in agent response, got: {:?}",
        result.text
    );
}

#[tokio::test]
#[ignore = "requires real provider credentials and network access"]
async fn test_env_config_real_llm_planning_records_run_and_task_events() {
    let _guard = env_lock().lock().await;
    inject_minimax_aliases();
    require_env("A3S_OPENAI_API_KEY");
    require_env("A3S_OPENAI_BASE_URL");

    let config_path = repo_config_path();
    let agent = Agent::create(config_path.to_string_lossy().to_string())
        .await
        .expect("agent should be created from env-injected ACL config");

    let workspace = tempfile::tempdir().expect("temp workspace");
    let opts = SessionOptions::new()
        .with_session_id("real-env-planning-session")
        .with_planning_mode(PlanningMode::Enabled);
    let session = agent
        .session(workspace.path().to_string_lossy().to_string(), Some(opts))
        .expect("session should be created");

    let result = session
        .send(
            "Do not edit files. Use planning mode to answer, then include exactly this token in the final answer: A3S_PLANNING_OK",
            None,
        )
        .await
        .expect("real planning session should complete");

    assert!(
        !result.messages.is_empty(),
        "real planning session should produce a message history"
    );

    let runs = session.runs().await;
    assert_eq!(runs.len(), 1, "one run should be recorded");
    assert_eq!(runs[0].status, RunStatus::Completed);
    assert_eq!(runs[0].session_id, "real-env-planning-session");

    let events = session.run_events(&runs[0].id).await;
    assert!(
        events
            .iter()
            .any(|record| matches!(record.event, AgentEvent::PlanningStart { .. })),
        "planning start event should be recorded"
    );
    assert!(
        events
            .iter()
            .any(|record| matches!(record.event, AgentEvent::PlanningEnd { .. })),
        "planning end event should be recorded"
    );
    assert!(
        events
            .iter()
            .any(|record| matches!(record.event, AgentEvent::TaskUpdated { .. })),
        "task-list snapshots should be recorded for planning mode"
    );
    assert!(
        events
            .iter()
            .any(|record| matches!(record.event, AgentEvent::End { .. })),
        "end event should be recorded"
    );
}
