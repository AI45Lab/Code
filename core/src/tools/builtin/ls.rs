//! Ls tool - List directory contents

use crate::tools::types::{Tool, ToolContext, ToolOutput};
use anyhow::Result;
use async_trait::async_trait;
use std::path::PathBuf;

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
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (default workspace root)"
                }
            },
            "required": []
        })
    }

    async fn execute(&self, args: &serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let path_str = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        let target = if std::path::Path::new(path_str).is_absolute() {
            PathBuf::from(path_str)
        } else {
            ctx.workspace.join(path_str)
        };

        if !target.exists() {
            return Ok(ToolOutput::error(format!(
                "Directory not found: {}",
                target.display()
            )));
        }

        if !target.is_dir() {
            return Ok(ToolOutput::error(format!(
                "Not a directory: {}",
                target.display()
            )));
        }

        let mut entries = Vec::new();
        let mut dir = match tokio::fs::read_dir(&target).await {
            Ok(d) => d,
            Err(e) => {
                return Ok(ToolOutput::error(format!(
                    "Failed to read directory {}: {}",
                    target.display(),
                    e
                )))
            }
        };

        while let Ok(Some(entry)) = dir.next_entry().await {
            let name = entry.file_name().to_string_lossy().to_string();
            let file_type = entry.file_type().await;
            let metadata = entry.metadata().await;

            let (kind, size) = match (&file_type, &metadata) {
                (Ok(ft), Ok(m)) => {
                    let kind = if ft.is_dir() {
                        "dir"
                    } else if ft.is_symlink() {
                        "link"
                    } else {
                        "file"
                    };
                    (kind, m.len())
                }
                _ => ("unknown", 0),
            };

            entries.push((name, kind, size));
        }

        entries.sort_by(|a, b| {
            // Directories first, then alphabetical
            let dir_order = (a.1 != "dir").cmp(&(b.1 != "dir"));
            dir_order.then(a.0.to_lowercase().cmp(&b.0.to_lowercase()))
        });

        let mut output = format!("Directory: {}\n\n", target.display());

        if entries.is_empty() {
            output.push_str("(empty directory)\n");
        } else {
            for (name, kind, size) in &entries {
                let size_str = format_size(*size);
                let suffix = if *kind == "dir" { "/" } else { "" };
                output.push_str(&format!(
                    "{:<6} {:>8}  {}{}\n",
                    kind, size_str, name, suffix
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
}
