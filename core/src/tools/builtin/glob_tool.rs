//! Glob tool - Find files matching a glob pattern

use crate::tools::types::{Tool, ToolContext, ToolOutput};
use crate::workspace::WorkspaceGlobRequest;
use anyhow::Result;
use async_trait::async_trait;

pub struct GlobTool;

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern. Returns a list of file paths."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Required. Glob pattern to match, for example '**/*.rs' or 'src/**/*.ts'. Always provide this exact field name: 'pattern'."
                },
                "path": {
                    "type": "string",
                    "description": "Optional. Base directory for the search. Default: workspace root."
                }
            },
            "required": ["pattern"],
            "examples": [
                {
                    "pattern": "**/*.rs"
                },
                {
                    "pattern": "*.md",
                    "path": "docs"
                }
            ]
        })
    }

    async fn execute(&self, args: &serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let pattern = match args.get("pattern").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return Ok(ToolOutput::error("pattern parameter is required")),
        };

        let path_str = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let base = match ctx.resolve_workspace_path(path_str) {
            Ok(path) => path,
            Err(e) => return Ok(ToolOutput::error(format!("Failed to resolve path: {}", e))),
        };

        let Some(search) = ctx.workspace_services.search() else {
            return Ok(ToolOutput::error(
                "glob is not available: this workspace backend did not provide search",
            ));
        };

        let request = WorkspaceGlobRequest {
            base,
            pattern: pattern.to_string(),
        };
        let result = match ctx
            .workspace_services
            .run_with_timeout("glob", async move { search.glob(request).await })
            .await
        {
            Ok(result) => result,
            Err(e) => {
                return Ok(ToolOutput::error(format!(
                    "Invalid glob pattern '{}': {}",
                    pattern, e
                )))
            }
        };
        let matches: Vec<String> = result
            .matches
            .into_iter()
            .map(|path| path.as_str().to_string())
            .collect();

        if matches.is_empty() {
            Ok(ToolOutput::success(format!(
                "No files found matching pattern: {}",
                pattern
            )))
        } else {
            let count = matches.len();
            let mut output = matches.join("\n");
            output.push_str(&format!("\n\n{} file(s) found", count));
            Ok(ToolOutput::success(output))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_glob_find_files() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("a.txt"), "").unwrap();
        std::fs::write(temp.path().join("b.txt"), "").unwrap();
        std::fs::write(temp.path().join("c.rs"), "").unwrap();

        let tool = GlobTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(&serde_json::json!({"pattern": "*.txt"}), &ctx)
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.content.contains("a.txt"));
        assert!(result.content.contains("b.txt"));
        assert!(!result.content.contains("c.rs"));
        assert!(result.content.contains("2 file(s)"));
    }

    #[tokio::test]
    async fn test_glob_no_matches() {
        let temp = tempfile::tempdir().unwrap();
        let tool = GlobTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(&serde_json::json!({"pattern": "*.xyz"}), &ctx)
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.content.contains("No files found"));
    }

    #[tokio::test]
    async fn test_glob_recursive() {
        let temp = tempfile::tempdir().unwrap();
        let sub = temp.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(temp.path().join("root.rs"), "").unwrap();
        std::fs::write(sub.join("nested.rs"), "").unwrap();

        let tool = GlobTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(&serde_json::json!({"pattern": "**/*.rs"}), &ctx)
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.content.contains("root.rs"));
        assert!(result.content.contains("nested.rs"));
    }

    #[tokio::test]
    async fn test_glob_missing_pattern() {
        let tool = GlobTool;
        let ctx = ToolContext::new(PathBuf::from("/tmp"));

        let result = tool.execute(&serde_json::json!({}), &ctx).await.unwrap();
        assert!(!result.success);
    }

    #[test]
    fn test_glob_schema_is_canonical() {
        let tool = GlobTool;
        let params = tool.parameters();
        assert_eq!(params["additionalProperties"], false);
        assert_eq!(params["required"], serde_json::json!(["pattern"]));
        let examples = params["examples"].as_array().unwrap();
        assert_eq!(examples[0]["pattern"], "**/*.rs");
        assert!(examples[0].get("glob").is_none());
    }
}
