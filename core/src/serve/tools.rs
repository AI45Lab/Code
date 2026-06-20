//! Install an agent directory's `tools/` specs into a live session.
//!
//! Parsing happens in [`AgentDir::load`](crate::config::AgentDir) (always, like
//! `schedules`); *installation* is a session-time operation here so it
//! reuses the same fallible, harness-owned registration the SDK's
//! [`add_mcp_server`](crate::AgentSession::add_mcp_server) already exposes — tool
//! definition comes from the directory, but visibility and the safety gate stay
//! with the harness.

use std::sync::Arc;

use crate::agent_api::AgentSession;
use crate::config::ToolSpec;
use crate::error::Result;
use crate::tools::AgentDirScriptTool;

/// Install each parsed [`ToolSpec`] into `session`.
///
/// `mcp` specs are registered and connected via the existing `add_mcp_server`
/// path, so their tools land as `mcp__<server>__<tool>` and are gated by the
/// session's permission policy like any other tool. Connection is fallible and
/// surfaces here (e.g. a missing `command` binary), so a misconfigured tool fails
/// at serve startup rather than silently at first call.
///
/// `script` specs register a sandboxed QuickJS tool ([`AgentDirScriptTool`]) into
/// the session registry via the same non-shadowing `register_dynamic_tool` path
/// builtins/MCP use — it cannot replace a builtin, and the model's call to it is
/// permission-gated like any tool. The script's *inner* `ctx.tool` calls are
/// bounded by the spec's pinned (fail-closed) allow-list and the QuickJS sandbox
/// rather than the session permission policy — see [`AgentDirScriptTool`].
pub async fn install_agent_dir_tools(session: &AgentSession, specs: &[ToolSpec]) -> Result<()> {
    for spec in specs {
        match spec {
            ToolSpec::Mcp(config) => {
                session.add_mcp_server(config.clone()).await?;
            }
            ToolSpec::Script(script) => {
                let registry = Arc::clone(session.tool_executor().registry());
                let tool = Arc::new(AgentDirScriptTool::new(script.clone(), registry));
                session.tool_executor().register_dynamic_tool(tool);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_api::Agent;
    use crate::config::CodeConfig;

    fn test_config() -> CodeConfig {
        let acl = r#"
default_model = "anthropic/claude-sonnet-4-20250514"
providers "anthropic" {
  api_key = "test-key"
  models "claude-sonnet-4-20250514" { name = "Claude Sonnet 4" }
}
"#;
        CodeConfig::from_acl(acl).unwrap()
    }

    #[tokio::test]
    async fn install_with_no_tools_is_ok() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let session = agent.session("/tmp/ws", None).unwrap();
        // Empty specs → no MCP connect attempted, returns Ok without a live server.
        install_agent_dir_tools(&session, &[]).await.unwrap();
    }

    fn script_spec(name: &str, description: &str) -> ToolSpec {
        ToolSpec::Script(crate::config::ScriptToolSpec {
            name: name.to_string(),
            description: description.to_string(),
            path: std::path::PathBuf::from("scripts/x.js"),
            allowed_tools: None,
            limits: crate::config::ScriptToolLimits::default(),
        })
    }

    #[tokio::test]
    async fn install_registers_script_tool_visibly() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let session = agent.session("/tmp/ws", None).unwrap();

        install_agent_dir_tools(&session, &[script_spec("repo-search", "search the repo")])
            .await
            .unwrap();

        let registry = session.tool_executor().registry();
        assert!(
            registry.contains("repo-search"),
            "script tool is registered"
        );
        let tool = registry.get("repo-search").unwrap();
        assert_eq!(tool.description(), "search the repo", "it's our tool");
    }

    #[tokio::test]
    async fn install_mcp_with_bad_command_fails_at_startup() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let session = agent.session("/tmp/ws", None).unwrap();

        // A stdio MCP server whose command does not exist must surface at install
        // time (fail at startup, not silently at first call). Built via the same
        // YAML path the agent-dir loader uses.
        let cfg: crate::mcp::McpServerConfig = serde_yaml::from_str(
            "name: ghost\ntransport: stdio\ncommand: a3s-nonexistent-mcp-binary-xyz\n",
        )
        .unwrap();

        let result = install_agent_dir_tools(&session, &[ToolSpec::Mcp(cfg)]).await;
        assert!(
            result.is_err(),
            "a missing MCP command must fail install_agent_dir_tools, not be swallowed"
        );
    }

    #[tokio::test]
    async fn install_script_cannot_shadow_a_builtin() {
        let agent = Agent::from_config(test_config()).await.unwrap();
        let session = agent.session("/tmp/ws", None).unwrap();
        let registry = session.tool_executor().registry();
        let builtin_bash_desc = registry.get("bash").unwrap().description().to_string();

        // A script that tries to take the `bash` name must be rejected, not shadow it.
        install_agent_dir_tools(&session, &[script_spec("bash", "HIJACKED")])
            .await
            .unwrap();

        assert_eq!(
            registry.get("bash").unwrap().description(),
            builtin_bash_desc,
            "builtin bash must be unchanged (script registration rejected)"
        );
    }
}
