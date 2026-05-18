//! Write tool - Write content to files

use crate::tools::types::{Tool, ToolContext, ToolOutput};
use anyhow::Result;
use async_trait::async_trait;

pub struct WriteTool;

#[async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates the file and parent directories if they don't exist."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Required. Path to the file to write. Always provide this exact field name: 'file_path'."
                },
                "content": {
                    "type": "string",
                    "description": "Required. Full content to write to the file. Always provide this exact field name: 'content'."
                }
            },
            "required": ["file_path", "content"],
            "examples": [
                {
                    "file_path": "notes.txt",
                    "content": "hello world"
                }
            ]
        })
    }

    async fn execute(&self, args: &serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let file_path = match args.get("file_path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return Ok(ToolOutput::error("file_path parameter is required")),
        };

        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return Ok(ToolOutput::error("content parameter is required")),
        };

        let workspace_path = match ctx.resolve_workspace_path(file_path) {
            Ok(p) => p,
            Err(e) => return Ok(ToolOutput::error(format!("Failed to resolve path: {}", e))),
        };

        // Read existing content for diff metadata (if file exists)
        let fs = ctx.workspace_services.fs();
        let path_for_before = workspace_path.clone();
        let fs_for_before = fs.clone();
        let before_content = ctx
            .workspace_services
            .run_with_timeout("read_text", async move {
                fs_for_before.read_text(&path_for_before).await
            })
            .await
            .ok();

        let path_for_write = workspace_path.clone();
        let content_for_write = content.to_string();
        match ctx
            .workspace_services
            .run_with_timeout("write_text", async move {
                fs.write_text(&path_for_write, &content_for_write).await
            })
            .await
        {
            Ok(outcome) => {
                // Attach diff metadata
                let mut metadata = serde_json::Map::new();
                metadata.insert("file_path".to_string(), serde_json::json!(file_path));
                metadata.insert("after".to_string(), serde_json::json!(content));
                if let Some(before) = before_content {
                    metadata.insert("before".to_string(), serde_json::json!(before));
                }

                Ok(ToolOutput::success(format!(
                    "Wrote {} bytes ({} lines) to {}",
                    outcome.bytes,
                    outcome.lines,
                    ctx.workspace_services.display_path(&workspace_path)
                ))
                .with_metadata(serde_json::Value::Object(metadata)))
            }
            Err(e) => Ok(ToolOutput::error(format!(
                "Failed to write file {}: {}",
                ctx.workspace_services.display_path(&workspace_path),
                e
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_write_new_file() {
        let temp = tempfile::tempdir().unwrap();
        let tool = WriteTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(
                &serde_json::json!({"file_path": "new.txt", "content": "hello world"}),
                &ctx,
            )
            .await
            .unwrap();

        assert!(result.success);
        let content = std::fs::read_to_string(temp.path().join("new.txt")).unwrap();
        assert_eq!(content, "hello world");
    }

    #[tokio::test]
    async fn test_write_creates_parent_dirs() {
        let temp = tempfile::tempdir().unwrap();
        let tool = WriteTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(
                &serde_json::json!({"file_path": "sub/dir/file.txt", "content": "nested"}),
                &ctx,
            )
            .await
            .unwrap();

        assert!(result.success);
        let content = std::fs::read_to_string(temp.path().join("sub/dir/file.txt")).unwrap();
        assert_eq!(content, "nested");
    }

    #[tokio::test]
    async fn test_write_overwrite() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("existing.txt"), "old").unwrap();

        let tool = WriteTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(
                &serde_json::json!({"file_path": "existing.txt", "content": "new"}),
                &ctx,
            )
            .await
            .unwrap();

        assert!(result.success);
        let content = std::fs::read_to_string(temp.path().join("existing.txt")).unwrap();
        assert_eq!(content, "new");
    }

    #[tokio::test]
    async fn test_write_missing_params() {
        let tool = WriteTool;
        let ctx = ToolContext::new(std::path::PathBuf::from("/tmp"));

        let result = tool.execute(&serde_json::json!({}), &ctx).await.unwrap();
        assert!(!result.success);

        let result = tool
            .execute(&serde_json::json!({"file_path": "x"}), &ctx)
            .await
            .unwrap();
        assert!(!result.success);
    }

    #[test]
    fn test_write_schema_is_canonical() {
        let tool = WriteTool;
        let params = tool.parameters();
        assert_eq!(params["additionalProperties"], false);
        assert_eq!(
            params["required"],
            serde_json::json!(["file_path", "content"])
        );
        let examples = params["examples"].as_array().unwrap();
        assert_eq!(examples[0]["file_path"], "notes.txt");
        assert!(examples[0].get("path").is_none());
    }
}
