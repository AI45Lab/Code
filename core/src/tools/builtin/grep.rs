//! Grep tool - Search file contents with regex

use crate::tools::types::{Tool, ToolContext, ToolOutput};
use crate::tools::MAX_OUTPUT_SIZE;
use anyhow::Result;
use async_trait::async_trait;
use ignore::WalkBuilder;
use regex::Regex;
use std::path::PathBuf;

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
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (default workspace root)"
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g., '*.rs', '*.{ts,tsx}')"
                },
                "context": {
                    "type": "integer",
                    "description": "Number of context lines to show before and after matches"
                },
                "-i": {
                    "type": "boolean",
                    "description": "Case insensitive search"
                }
            },
            "required": ["pattern"]
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

        let regex = match Regex::new(&regex_pattern) {
            Ok(r) => r,
            Err(e) => {
                return Ok(ToolOutput::error(format!(
                    "Invalid regex pattern '{}': {}",
                    pattern_str, e
                )))
            }
        };

        let search_path = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) => {
                if std::path::Path::new(p).is_absolute() {
                    PathBuf::from(p)
                } else {
                    ctx.workspace.join(p)
                }
            }
            None => ctx.workspace.clone(),
        };

        let glob_filter = args.get("glob").and_then(|v| v.as_str());
        let context_lines = args.get("context").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        // Use ignore crate to respect .gitignore
        let mut builder = WalkBuilder::new(&search_path);
        builder.hidden(false).git_ignore(true).git_global(true);

        if let Some(glob_pat) = glob_filter {
            let mut types = ignore::types::TypesBuilder::new();
            types.add("custom", glob_pat).ok();
            types.select("custom");
            if let Ok(built) = types.build() {
                builder.types(built);
            }
        }

        let mut output = String::new();
        let mut match_count = 0;
        let mut file_count = 0;
        let mut total_size = 0;

        for entry in builder.build().flatten() {
            if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                continue;
            }

            let file_path = entry.path();
            let content = match std::fs::read_to_string(file_path) {
                Ok(c) => c,
                Err(_) => continue, // Skip binary/unreadable files
            };

            let lines: Vec<&str> = content.lines().collect();
            let mut file_matches = Vec::new();

            for (line_idx, line) in lines.iter().enumerate() {
                if regex.is_match(line) {
                    file_matches.push(line_idx);
                }
            }

            if file_matches.is_empty() {
                continue;
            }

            file_count += 1;
            // Normalize paths for consistent strip_prefix (avoids UNC prefix mismatch on Windows)
            let file_display = file_path.to_string_lossy().replace('\\', "/");
            let workspace_display = ctx.workspace.to_string_lossy().replace('\\', "/");
            let rel_path = if let Some(stripped) = file_display.strip_prefix(&workspace_display) {
                stripped.trim_start_matches('/').to_string()
            } else {
                file_path
                    .strip_prefix(&ctx.workspace)
                    .unwrap_or(file_path)
                    .to_string_lossy()
                    .to_string()
            };

            for &match_idx in &file_matches {
                if total_size > MAX_OUTPUT_SIZE {
                    output.push_str("\n... (output truncated)\n");
                    return Ok(ToolOutput::success(format!(
                        "{}Found {} matches in {} files (output truncated)",
                        output, match_count, file_count
                    )));
                }

                match_count += 1;

                let start = match_idx.saturating_sub(context_lines);
                let end = (match_idx + context_lines + 1).min(lines.len());

                for (i, line) in lines[start..end].iter().enumerate() {
                    let abs_i = start + i;
                    let prefix = if abs_i == match_idx { ">" } else { " " };
                    let line_str = format!("{}{}:{}: {}\n", prefix, rel_path, abs_i + 1, line);
                    total_size += line_str.len();
                    output.push_str(&line_str);
                }

                if context_lines > 0 {
                    output.push_str("--\n");
                    total_size += 3;
                }
            }
        }

        if match_count == 0 {
            Ok(ToolOutput::success(format!(
                "No matches found for pattern: {}",
                pattern_str
            )))
        } else {
            output.push_str(&format!(
                "\n{} match(es) in {} file(s)",
                match_count, file_count
            ));
            Ok(ToolOutput::success(output))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
