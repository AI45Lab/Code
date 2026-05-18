//! Grep tool - Search file contents with regex

use crate::tools::types::{Tool, ToolContext, ToolOutput};
use crate::tools::MAX_OUTPUT_SIZE;
use crate::workspace::WorkspaceGrepRequest;
use anyhow::Result;
use async_trait::async_trait;
use regex::Regex;

pub struct GrepTool;

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search for a pattern in files using ripgrep. Returns matching lines with file paths and line numbers."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Required. Regular expression pattern to search for. Always provide this exact field name: 'pattern'."
                },
                "path": {
                    "type": "string",
                    "description": "Optional. Directory or file to search in. Default: workspace root."
                },
                "glob": {
                    "type": "string",
                    "description": "Optional. Glob pattern to filter files, for example '*.rs' or '*.{ts,tsx}'."
                },
                "context": {
                    "type": "integer",
                    "description": "Optional. Number of context lines to show before and after matches."
                },
                "-i": {
                    "type": "boolean",
                    "description": "Optional. Case insensitive search."
                }
            },
            "required": ["pattern"],
            "examples": [
                {
                    "pattern": "TODO"
                },
                {
                    "pattern": "fn main",
                    "path": "src",
                    "glob": "*.rs",
                    "context": 2
                }
            ]
        })
    }

    async fn execute(&self, args: &serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let pattern_str = match args.get("pattern").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return Ok(ToolOutput::error("pattern parameter is required")),
        };

        let case_insensitive = args.get("-i").and_then(|v| v.as_bool()).unwrap_or(false);

        let regex_pattern = if case_insensitive {
            format!("(?i){}", pattern_str)
        } else {
            pattern_str.to_string()
        };

        if let Err(e) = Regex::new(&regex_pattern) {
            return Ok(ToolOutput::error(format!(
                "Invalid regex pattern '{}': {}",
                pattern_str, e
            )));
        }

        let path_str = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let base = match ctx.resolve_workspace_path(path_str) {
            Ok(path) => path,
            Err(e) => return Ok(ToolOutput::error(format!("Failed to resolve path: {}", e))),
        };

        let glob_filter = args.get("glob").and_then(|v| v.as_str());
        let context_lines = args.get("context").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        let Some(search) = ctx.workspace_services.search() else {
            return Ok(ToolOutput::error(
                "grep is not available: this workspace backend did not provide search",
            ));
        };
        let request = WorkspaceGrepRequest {
            base,
            pattern: pattern_str.to_string(),
            glob: glob_filter.map(str::to_string),
            context_lines,
            case_insensitive,
            max_output_size: MAX_OUTPUT_SIZE,
        };
        let result = match ctx
            .workspace_services
            .run_with_timeout("grep", async move { search.grep(request).await })
            .await
        {
            Ok(result) => result,
            Err(e) => return Ok(ToolOutput::error(format!("Grep search failed: {}", e))),
        };

        if result.match_count == 0 {
            Ok(ToolOutput::success(format!(
                "No matches found for pattern: {}",
                pattern_str
            )))
        } else if result.truncated {
            Ok(ToolOutput::success(format!(
                "{}\n... (output truncated)\nFound {} matches in {} files (output truncated)",
                result.output, result.match_count, result.file_count
            )))
        } else {
            Ok(ToolOutput::success(format!(
                "{}\n{} match(es) in {} file(s)",
                result.output, result.match_count, result.file_count
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_grep_find_pattern() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(
            temp.path().join("a.txt"),
            "hello world\nfoo bar\nhello again",
        )
        .unwrap();
        std::fs::write(temp.path().join("b.txt"), "no match here").unwrap();

        let tool = GrepTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(&serde_json::json!({"pattern": "hello"}), &ctx)
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.content.contains("hello world"));
        assert!(result.content.contains("hello again"));
        assert!(result.content.contains("2 match(es)"));
    }

    #[tokio::test]
    async fn test_grep_no_match() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("a.txt"), "hello").unwrap();

        let tool = GrepTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(&serde_json::json!({"pattern": "xyz"}), &ctx)
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.content.contains("No matches found"));
    }

    #[tokio::test]
    async fn test_grep_case_insensitive() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("a.txt"), "Hello World\nhello world").unwrap();

        let tool = GrepTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(&serde_json::json!({"pattern": "hello", "-i": true}), &ctx)
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.content.contains("2 match(es)"));
    }

    #[tokio::test]
    async fn test_grep_invalid_regex() {
        let tool = GrepTool;
        let ctx = ToolContext::new(PathBuf::from("/tmp"));

        let result = tool
            .execute(&serde_json::json!({"pattern": "[invalid"}), &ctx)
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.content.contains("Invalid regex"));
    }

    #[tokio::test]
    async fn test_grep_missing_pattern() {
        let tool = GrepTool;
        let ctx = ToolContext::new(PathBuf::from("/tmp"));

        let result = tool.execute(&serde_json::json!({}), &ctx).await.unwrap();
        assert!(!result.success);
    }

    #[test]
    fn test_grep_schema_is_canonical() {
        let tool = GrepTool;
        let params = tool.parameters();
        assert_eq!(params["additionalProperties"], false);
        assert_eq!(params["required"], serde_json::json!(["pattern"]));
        let examples = params["examples"].as_array().unwrap();
        assert_eq!(examples[0]["pattern"], "TODO");
        assert!(examples[0].get("query").is_none());
    }
}
