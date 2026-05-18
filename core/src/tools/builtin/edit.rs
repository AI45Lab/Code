//! Edit tool - Edit files by string replacement

use crate::tools::types::{Tool, ToolContext, ToolOutput};
use anyhow::Result;
use async_trait::async_trait;

pub struct EditTool;

#[async_trait]
impl Tool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn description(&self) -> &str {
        "Edit a file by replacing a specific string with another. The old_string must be unique in the file unless replace_all is true."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Required. Path to the file to edit. Always provide this exact field name: 'file_path'."
                },
                "old_string": {
                    "type": "string",
                    "description": "Required. The exact string to replace. It must be unique unless replace_all=true."
                },
                "new_string": {
                    "type": "string",
                    "description": "Required. The replacement string."
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Optional. Replace all occurrences. Default: false."
                }
            },
            "required": ["file_path", "old_string", "new_string"],
            "examples": [
                {
                    "file_path": "src/lib.rs",
                    "old_string": "old_value",
                    "new_string": "new_value"
                },
                {
                    "file_path": "src/lib.rs",
                    "old_string": "foo",
                    "new_string": "bar",
                    "replace_all": true
                }
            ]
        })
    }

    async fn execute(&self, args: &serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let file_path = match args.get("file_path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return Ok(ToolOutput::error("file_path parameter is required")),
        };

        let old_string = match args.get("old_string").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return Ok(ToolOutput::error("old_string parameter is required")),
        };

        let new_string = match args.get("new_string").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return Ok(ToolOutput::error("new_string parameter is required")),
        };

        let replace_all = args
            .get("replace_all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let workspace_path = match ctx.resolve_workspace_path(file_path) {
            Ok(path) => path,
            Err(e) => return Ok(ToolOutput::error(format!("Failed to resolve path: {}", e))),
        };
        let display_path = ctx.workspace_services.display_path(&workspace_path);

        let fs = ctx.workspace_services.fs();
        let path_for_read = workspace_path.clone();
        let fs_for_read = fs.clone();
        let content = match ctx
            .workspace_services
            .run_with_timeout("read_text", async move {
                fs_for_read.read_text(&path_for_read).await
            })
            .await
        {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolOutput::error(format!(
                    "Failed to read file {}: {}",
                    display_path, e
                )))
            }
        };

        let count = content.matches(old_string).count();

        if count == 0 {
            return Ok(ToolOutput::error(format!(
                "old_string not found in {}",
                display_path
            )));
        }

        if count > 1 && !replace_all {
            return Ok(ToolOutput::error(format!(
                "old_string found {} times in {}. Use replace_all=true to replace all occurrences, or provide a more specific string.",
                count,
                display_path
            )));
        }

        let new_content = if replace_all {
            content.replace(old_string, new_string)
        } else {
            content.replacen(old_string, new_string, 1)
        };

        let path_for_write = workspace_path.clone();
        let content_for_write = new_content.clone();
        match ctx
            .workspace_services
            .run_with_timeout("write_text", async move {
                fs.write_text(&path_for_write, &content_for_write).await
            })
            .await
        {
            Ok(_) => {
                // Attach diff metadata so frontend can show Monaco diff
                let mut metadata = serde_json::Map::new();
                metadata.insert("file_path".to_string(), serde_json::json!(file_path));
                metadata.insert("before".to_string(), serde_json::json!(content));
                metadata.insert("after".to_string(), serde_json::json!(new_content));

                Ok(ToolOutput::success(format!(
                    "Replaced {} occurrence(s) in {}",
                    if replace_all { count } else { 1 },
                    display_path
                ))
                .with_metadata(serde_json::Value::Object(metadata)))
            }
            Err(e) => Ok(ToolOutput::error(format!(
                "Failed to write file {}: {}",
                display_path, e
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_edit_single_replace() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("test.txt"), "hello world").unwrap();

        let tool = EditTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(
                &serde_json::json!({
                    "file_path": "test.txt",
                    "old_string": "hello",
                    "new_string": "goodbye"
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(result.success);
        let content = std::fs::read_to_string(temp.path().join("test.txt")).unwrap();
        assert_eq!(content, "goodbye world");
    }

    #[tokio::test]
    async fn test_edit_replace_all() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("test.txt"), "aaa bbb aaa").unwrap();

        let tool = EditTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(
                &serde_json::json!({
                    "file_path": "test.txt",
                    "old_string": "aaa",
                    "new_string": "ccc",
                    "replace_all": true
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(result.success);
        let content = std::fs::read_to_string(temp.path().join("test.txt")).unwrap();
        assert_eq!(content, "ccc bbb ccc");
    }

    #[tokio::test]
    async fn test_edit_not_unique() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("test.txt"), "aaa bbb aaa").unwrap();

        let tool = EditTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(
                &serde_json::json!({
                    "file_path": "test.txt",
                    "old_string": "aaa",
                    "new_string": "ccc"
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.content.contains("2 times"));
    }

    #[tokio::test]
    async fn test_edit_not_found() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("test.txt"), "hello world").unwrap();

        let tool = EditTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(
                &serde_json::json!({
                    "file_path": "test.txt",
                    "old_string": "xyz",
                    "new_string": "abc"
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.content.contains("not found"));
    }

    #[test]
    fn test_edit_schema_is_canonical() {
        let tool = EditTool;
        let params = tool.parameters();
        assert_eq!(params["additionalProperties"], false);
        assert_eq!(
            params["required"],
            serde_json::json!(["file_path", "old_string", "new_string"])
        );
        let examples = params["examples"].as_array().unwrap();
        assert_eq!(examples[0]["file_path"], "src/lib.rs");
        assert!(examples[0].get("path").is_none());
    }
}
