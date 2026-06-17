//! Install an agent directory's `tools/` specs into a live session.
//!
//! Parsing happens in [`AgentDir::load`](crate::config::AgentDir) (always, like
//! `schedules`/`channels`); *installation* is a session-time operation here so it
//! reuses the same fallible, harness-owned registration the SDK's
//! [`add_mcp_server`](crate::AgentSession::add_mcp_server) already exposes — tool
//! definition comes from the directory, but visibility and the safety gate stay
//! with the harness.

use crate::agent_api::AgentSession;
use crate::config::ToolSpec;
use crate::error::Result;

/// Install each parsed [`ToolSpec`] into `session`.
///
/// `mcp` specs are registered and connected via the existing `add_mcp_server`
/// path, so their tools land as `mcp__<server>__<tool>` and are gated by the
/// session's permission policy like any other tool. Connection is fallible and
/// surfaces here (e.g. a missing `command` binary), so a misconfigured tool fails
/// at serve startup rather than silently at first call.
pub async fn install_agent_dir_tools(session: &AgentSession, specs: &[ToolSpec]) -> Result<()> {
    for spec in specs {
        match spec {
            ToolSpec::Mcp(config) => {
                session.add_mcp_server(config.clone()).await?;
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
}
