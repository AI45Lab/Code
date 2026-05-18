//! Read tool - Read file contents with line numbering

use crate::text::truncate_utf8;
use crate::tools::types::{Tool, ToolContext, ToolOutput};
use crate::tools::{MAX_LINE_LENGTH, MAX_READ_LINES};
use anyhow::Result;
use async_trait::async_trait;

pub struct ReadTool;

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Read the contents of a file. Returns line-numbered output. Supports text files and images. Large outputs are capped; use offset/limit for long files."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Required. Path to the file to read, absolute or relative to the workspace. Always provide this exact field name: 'file_path'."
                },
                "offset": {
                    "type": "integer",
                    "description": "Optional. Line number to start reading from. 0-indexed. Default: 0."
                },
                "limit": {
                    "type": "integer",
                    "description": "Optional. Maximum number of lines to read. Default: 2000."
                }
            },
            "required": ["file_path"],
            "examples": [
                {
                    "file_path": "src/main.rs"
                },
                {
                    "file_path": "src/main.rs",
                    "offset": 40,
                    "limit": 80
                }
            ]
        })
    }

    async fn execute(&self, args: &serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let file_path = match args.get("file_path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return Ok(ToolOutput::error("file_path parameter is required")),
        };

        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(MAX_READ_LINES);

        let workspace_path = match ctx.resolve_workspace_path(file_path) {
            Ok(p) => p,
            Err(e) => return Ok(ToolOutput::error(format!("Failed to resolve path: {}", e))),
        };

        let fs = ctx.workspace_services.fs();
        let path_for_read = workspace_path.clone();
        let content = match ctx
            .workspace_services
            .run_with_timeout(
                "read_text",
                async move { fs.read_text(&path_for_read).await },
            )
            .await
        {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolOutput::error(format!(
                    "Failed to read file {}: {}",
                    ctx.workspace_services.display_path(&workspace_path),
                    e
                )))
            }
        };

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        if offset >= total_lines && total_lines > 0 {
            return Ok(ToolOutput::error(format!(
                "Offset {} exceeds file length ({} lines)",
                offset, total_lines
            )));
        }

        let end = (offset + limit).min(total_lines);
        let selected = &lines[offset..end];

        let mut output = String::new();
        for (i, line) in selected.iter().enumerate() {
            let line_num = offset + i + 1; // 1-indexed
            let truncated = truncate_utf8(line, MAX_LINE_LENGTH);
            output.push_str(&format!("{:>6}\t{}\n", line_num, truncated));
        }

        if end < total_lines {
            output.push_str(&format!(
                "\n... ({} more lines not shown, use offset/limit to read more)\n",
                total_lines - end
            ));
        }

        Ok(ToolOutput::success(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_read_file() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("test.txt");
        std::fs::write(&file, "line1\nline2\nline3\n").unwrap();

        let tool = ReadTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());
        let result = tool
            .execute(&serde_json::json!({"file_path": "test.txt"}), &ctx)
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.content.contains("line1"));
        assert!(result.content.contains("line2"));
        assert!(result.content.contains("line3"));
    }

    #[tokio::test]
    async fn test_read_with_offset_and_limit() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("test.txt");
        std::fs::write(&file, "a\nb\nc\nd\ne\n").unwrap();

        let tool = ReadTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());
        let result = tool
            .execute(
                &serde_json::json!({"file_path": "test.txt", "offset": 1, "limit": 2}),
                &ctx,
            )
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.content.contains("b"));
        assert!(result.content.contains("c"));
        assert!(!result.content.contains("\ta\n"));
    }

    #[tokio::test]
    async fn test_read_missing_file() {
        let temp = tempfile::tempdir().unwrap();
        let tool = ReadTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());
        let result = tool
            .execute(&serde_json::json!({"file_path": "nonexistent.txt"}), &ctx)
            .await
            .unwrap();

        assert!(!result.success);
    }

    #[tokio::test]
    async fn test_read_missing_param() {
        let tool = ReadTool;
        let ctx = ToolContext::new(PathBuf::from("/tmp"));
        let result = tool.execute(&serde_json::json!({}), &ctx).await.unwrap();

        assert!(!result.success);
        assert!(result.content.contains("file_path"));
    }

    #[test]
    fn test_read_schema_is_canonical() {
        let tool = ReadTool;
        let params = tool.parameters();
        assert_eq!(params["additionalProperties"], false);
        assert_eq!(params["required"], serde_json::json!(["file_path"]));
        let examples = params["examples"].as_array().unwrap();
        assert_eq!(examples[0]["file_path"], "src/main.rs");
        assert!(examples[0].get("path").is_none());
    }

    #[tokio::test]
    async fn test_read_truncation_at_utf8_boundary() {
        // Regression test: truncation at byte 2000 should not panic
        // when byte 2000 falls inside a multibyte UTF-8 character.
        // "频" is 3 bytes (bytes 1999..2002). When byte 2000 is
        // inside '频', truncation must find a valid char boundary.
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("boundary.txt");
        // 1999 ASCII bytes + one 3-byte UTF-8 char + trailing ASCII.
        // Byte 2000 is inside the '频' character (bytes 1999..2002).
        let content = "a".repeat(1999) + "频" + &"z".repeat(20);
        std::fs::write(&file, &content).unwrap();

        let tool = ReadTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());
        // Should not panic
        let result = tool
            .execute(&serde_json::json!({"file_path": "boundary.txt"}), &ctx)
            .await
            .unwrap();

        assert!(
            result.success,
            "read should succeed, got error: {}",
            result.content
        );
        // Verify the truncated content is valid UTF-8
        assert!(!result.content.contains("byte index"));
    }
}
