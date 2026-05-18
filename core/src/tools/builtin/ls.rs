//! Ls tool - List directory contents

use crate::tools::types::{Tool, ToolContext, ToolOutput};
use anyhow::Result;
use async_trait::async_trait;

pub struct LsTool;

#[async_trait]
impl Tool for LsTool {
    fn name(&self) -> &str {
        "ls"
    }

    fn description(&self) -> &str {
        "List contents of a directory with file types and sizes."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Optional. Directory path to list. Default: workspace root."
                }
            },
            "required": [],
            "examples": [
                {},
                {
                    "path": "src"
                }
            ]
        })
    }

    async fn execute(&self, args: &serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let path_str = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        let workspace_path = match ctx.resolve_workspace_path(path_str) {
            Ok(p) => p,
            Err(e) => return Ok(ToolOutput::error(format!("Failed to resolve path: {}", e))),
        };

        let fs = ctx.workspace_services.fs();
        let path_for_list = workspace_path.clone();
        let mut entries = match ctx
            .workspace_services
            .run_with_timeout("list_dir", async move { fs.list_dir(&path_for_list).await })
            .await
        {
            Ok(entries) => entries,
            Err(e) => {
                return Ok(ToolOutput::error(format!(
                    "Failed to read directory {}: {}",
                    ctx.workspace_services.display_path(&workspace_path),
                    e
                )))
            }
        };

        entries.sort_by(|a, b| {
            // Directories first, then alphabetical
            let dir_order = (a.kind.as_tool_kind() != "dir").cmp(&(b.kind.as_tool_kind() != "dir"));
            dir_order.then(a.name.to_lowercase().cmp(&b.name.to_lowercase()))
        });

        let mut output = format!(
            "Directory: {}\n\n",
            ctx.workspace_services.display_path(&workspace_path)
        );

        if entries.is_empty() {
            output.push_str("(empty directory)\n");
        } else {
            for entry in &entries {
                let kind = entry.kind.as_tool_kind();
                let size_str = format_size(entry.size);
                let suffix = if kind == "dir" { "/" } else { "" };
                output.push_str(&format!(
                    "{:<6} {:>8}  {}{}\n",
                    kind, size_str, entry.name, suffix
                ));
            }
            output.push_str(&format!("\n{} entries\n", entries.len()));
        }

        Ok(ToolOutput::success(output))
    }
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{}B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1}GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ls_directory() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("file.txt"), "content").unwrap();
        std::fs::create_dir(temp.path().join("subdir")).unwrap();

        let tool = LsTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool.execute(&serde_json::json!({}), &ctx).await.unwrap();

        assert!(result.success);
        assert!(result.content.contains("file.txt"));
        assert!(result.content.contains("subdir"));
        assert!(result.content.contains("dir"));
    }

    #[tokio::test]
    async fn test_ls_empty_dir() {
        let temp = tempfile::tempdir().unwrap();
        let tool = LsTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool.execute(&serde_json::json!({}), &ctx).await.unwrap();

        assert!(result.success);
        assert!(result.content.contains("empty directory"));
    }

    #[tokio::test]
    async fn test_ls_nonexistent() {
        let temp = tempfile::tempdir().unwrap();
        let tool = LsTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(&serde_json::json!({"path": "nonexistent"}), &ctx)
            .await
            .unwrap();

        assert!(!result.success);
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(0), "0B");
        assert_eq!(format_size(512), "512B");
        assert_eq!(format_size(1024), "1.0KB");
        assert_eq!(format_size(1048576), "1.0MB");
    }

    #[test]
    fn test_ls_schema_is_canonical() {
        let tool = LsTool;
        let params = tool.parameters();
        assert_eq!(params["additionalProperties"], false);
        let examples = params["examples"].as_array().unwrap();
        assert!(examples[0].as_object().unwrap().is_empty());
        assert_eq!(examples[1]["path"], "src");
    }
}
