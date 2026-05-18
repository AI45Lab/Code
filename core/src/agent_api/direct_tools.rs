//! Direct host tool access.
//!
//! These helpers expose the same tool executor used by the agent loop for host
//! control-plane calls. Keeping argument shaping and result projection here
//! prevents the public session facade from duplicating tool-specific knowledge.

use super::{AgentSession, ToolCallResult};
use crate::error::Result;
use crate::llm::ToolDefinition;
use crate::tools::{ToolArtifact, ToolContext, ToolExecutor};
use std::sync::Arc;

pub(super) struct DirectToolRuntime {
    tool_executor: Arc<ToolExecutor>,
    tool_context: ToolContext,
}

impl DirectToolRuntime {
    pub(super) fn from_session(session: &AgentSession) -> Self {
        Self {
            tool_executor: Arc::clone(&session.tool_executor),
            tool_context: session.tool_context.clone(),
        }
    }

    pub(super) fn definitions(&self) -> Vec<ToolDefinition> {
        self.tool_executor.definitions()
    }

    pub(super) fn names(&self) -> Vec<String> {
        self.tool_executor
            .definitions()
            .into_iter()
            .map(|tool| tool.name)
            .collect()
    }

    pub(super) fn artifact(&self, artifact_uri: &str) -> Option<ToolArtifact> {
        self.tool_executor.get_artifact(artifact_uri)
    }

    pub(super) async fn read_file(&self, path: &str) -> Result<String> {
        let args = serde_json::json!({ "file_path": path });
        let result = self.tool_executor.execute("read", &args).await?;
        Ok(result.output)
    }

    pub(super) async fn write_file(&self, path: &str, content: &str) -> Result<ToolCallResult> {
        let args = serde_json::json!({ "file_path": path, "content": content });
        self.call("write", args).await
    }

    pub(super) async fn ls(&self, path: Option<&str>) -> Result<ToolCallResult> {
        let args = match path {
            Some(path) => serde_json::json!({ "path": path }),
            None => serde_json::json!({}),
        };
        self.call("ls", args).await
    }

    pub(super) async fn edit_file(
        &self,
        path: &str,
        old_string: &str,
        new_string: &str,
        replace_all: bool,
    ) -> Result<ToolCallResult> {
        let args = serde_json::json!({
            "file_path": path,
            "old_string": old_string,
            "new_string": new_string,
            "replace_all": replace_all,
        });
        self.call("edit", args).await
    }

    pub(super) async fn patch_file(&self, path: &str, diff: &str) -> Result<ToolCallResult> {
        let args = serde_json::json!({ "file_path": path, "diff": diff });
        self.call("patch", args).await
    }

    pub(super) async fn bash(&self, command: &str) -> Result<String> {
        let args = serde_json::json!({ "command": command });
        let result = self
            .tool_executor
            .execute_with_context("bash", &args, &self.tool_context)
            .await?;
        Ok(result.output)
    }

    pub(super) async fn glob(&self, pattern: &str) -> Result<Vec<String>> {
        let args = serde_json::json!({ "pattern": pattern });
        let result = self.tool_executor.execute("glob", &args).await?;
        Ok(parse_glob_output(&result.output))
    }

    pub(super) async fn grep(&self, pattern: &str) -> Result<String> {
        let args = serde_json::json!({ "pattern": pattern });
        let result = self.tool_executor.execute("grep", &args).await?;
        Ok(result.output)
    }

    pub(super) async fn call(&self, name: &str, args: serde_json::Value) -> Result<ToolCallResult> {
        let result = self.tool_executor.execute(name, &args).await?;
        Ok(ToolCallResult {
            name: name.to_string(),
            output: result.output,
            exit_code: result.exit_code,
            metadata: result.metadata,
        })
    }
}

fn parse_glob_output(output: &str) -> Vec<String> {
    output
        .lines()
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_glob_output_ignores_empty_lines() {
        assert_eq!(
            parse_glob_output("src/lib.rs\n\nsrc/main.rs\n"),
            vec!["src/lib.rs".to_string(), "src/main.rs".to_string()]
        );
    }
}
