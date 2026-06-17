//! Integration tests for the agent-dir `tools/` install path.
//!
//! Hermetic (no LLM, no network): exercises the real
//! `AgentDir::load -> install_agent_dir_tools -> add_mcp_server -> connect` chain
//! against a bogus MCP command, asserting it fails CLOSED at install rather than
//! silently at first call. Requires the `serve` feature:
//!
//! ```bash
//! cargo test -p a3s-code-core --features serve --test test_agent_dir_tools
//! ```
#![cfg(feature = "serve")]

use a3s_code_core::config::{AgentDir, CodeConfig};
use a3s_code_core::serve::install_agent_dir_tools;
use a3s_code_core::Agent;

/// A provider config with a placeholder key. No LLM call is made by these tests
/// (only the local MCP-connect path runs), so the key is never used.
fn fake_config() -> CodeConfig {
    CodeConfig::from_acl(
        r#"
default_model = "anthropic/claude-sonnet-4-20250514"
providers "anthropic" {
  api_key = "test-key"
  models "claude-sonnet-4-20250514" { name = "Claude Sonnet 4" }
}
"#,
    )
    .unwrap()
}

#[tokio::test]
async fn tools_mcp_parsed_and_install_fails_closed_on_bad_command() {
    // An agent dir whose tools/ declares an MCP server pointing at a command that
    // does not exist.
    let dir = tempfile::tempdir().expect("agent dir");
    std::fs::write(dir.path().join("instructions.md"), "role").unwrap();
    std::fs::create_dir_all(dir.path().join("tools")).unwrap();
    std::fs::write(
        dir.path().join("tools/bad.md"),
        "---\nkind: mcp\nname: bad\ntransport: stdio\ncommand: a3s-nonexistent-binary-zzz\n---\nbogus server\n",
    )
    .unwrap();

    // Parse: the convention loader turns tools/bad.md into one ToolSpec::Mcp.
    let agent_dir = AgentDir::load(dir.path()).expect("load agent dir");
    assert_eq!(agent_dir.tools.len(), 1, "tools/ should parse one spec");
    assert_eq!(agent_dir.tools[0].kind(), "mcp");
    assert_eq!(agent_dir.tools[0].name(), "bad");

    // Install into a real session: the MCP command can't be spawned, so connect
    // fails and the error propagates (fail-closed at startup).
    let agent = Agent::from_config(fake_config()).await.expect("agent");
    let workspace = tempfile::tempdir().expect("workspace");
    let session = agent
        .session(workspace.path().to_string_lossy().to_string(), None)
        .expect("session");

    let result = install_agent_dir_tools(&session, &agent_dir.tools).await;
    assert!(
        result.is_err(),
        "install must fail closed when the MCP server command cannot be spawned"
    );
}

#[tokio::test]
async fn install_with_empty_tools_is_ok() {
    // No tools/ → nothing to install → Ok, no connection attempted.
    let dir = tempfile::tempdir().expect("agent dir");
    std::fs::write(dir.path().join("instructions.md"), "role").unwrap();
    let agent_dir = AgentDir::load(dir.path()).expect("load");
    assert!(agent_dir.tools.is_empty());

    let agent = Agent::from_config(fake_config()).await.expect("agent");
    let workspace = tempfile::tempdir().expect("workspace");
    let session = agent
        .session(workspace.path().to_string_lossy().to_string(), None)
        .expect("session");

    install_agent_dir_tools(&session, &agent_dir.tools)
        .await
        .expect("empty install is Ok");
}
