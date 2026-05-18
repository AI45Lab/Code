//! Patch tool - Apply unified diff patches to files

use crate::tools::types::{Tool, ToolContext, ToolOutput};
use anyhow::Result;
use async_trait::async_trait;

pub struct PatchTool;

/// A parsed hunk from a unified diff
struct Hunk {
    /// 1-indexed start line in the original file
    old_start: usize,
    /// Lines to remove (without leading '-')
    removals: Vec<String>,
    /// Lines to add (without leading '+')
    additions: Vec<String>,
    /// Context lines with their offsets from old_start
    context: Vec<(usize, String)>,
}

/// Parse a unified diff into hunks
fn parse_hunks(diff: &str) -> Result<Vec<Hunk>, String> {
    let mut hunks = Vec::new();
    let lines: Vec<&str> = diff.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];

        // Look for @@ header
        if line.starts_with("@@") {
            let hunk = parse_single_hunk(&lines, &mut i)?;
            hunks.push(hunk);
        } else {
            i += 1;
        }
    }

    if hunks.is_empty() {
        return Err("No @@ hunk headers found in diff".to_string());
    }

    Ok(hunks)
}

fn parse_single_hunk(lines: &[&str], i: &mut usize) -> Result<Hunk, String> {
    let header = lines[*i];
    let old_start = parse_hunk_header(header)?;

    *i += 1;

    let mut removals = Vec::new();
    let mut additions = Vec::new();
    let mut context = Vec::new();
    let mut old_offset = 0;

    while *i < lines.len() {
        let line = lines[*i];

        if line.starts_with("@@") {
            break;
        }

        if let Some(content) = line.strip_prefix('-') {
            removals.push(content.to_string());
            old_offset += 1;
        } else if let Some(content) = line.strip_prefix('+') {
            additions.push(content.to_string());
        } else if let Some(content) = line.strip_prefix(' ') {
            context.push((old_offset, content.to_string()));
            old_offset += 1;
        } else if line.is_empty() {
            // Treat empty line as context
            context.push((old_offset, String::new()));
            old_offset += 1;
        } else {
            // Unknown prefix, treat as context
            context.push((old_offset, line.to_string()));
            old_offset += 1;
        }

        *i += 1;
    }

    Ok(Hunk {
        old_start,
        removals,
        additions,
        context,
    })
}

/// Parse @@ -old_start,old_count +new_start,new_count @@ line
fn parse_hunk_header(header: &str) -> Result<usize, String> {
    // Example: @@ -1,3 +1,3 @@
    let stripped = header
        .strip_prefix("@@")
        .and_then(|s| s.find("@@").map(|end| &s[..end]))
        .ok_or_else(|| format!("Invalid hunk header: {}", header))?
        .trim();

    // Parse -old_start part
    let parts: Vec<&str> = stripped.split_whitespace().collect();
    if parts.is_empty() {
        return Err(format!("Invalid hunk header: {}", header));
    }

    let old_part = parts[0]
        .strip_prefix('-')
        .ok_or_else(|| format!("Invalid old range in hunk header: {}", header))?;

    let old_start: usize = old_part
        .split(',')
        .next()
        .unwrap_or("1")
        .parse()
        .map_err(|_| format!("Invalid old start line in hunk header: {}", header))?;

    Ok(old_start)
}

/// Apply hunks to file content. Hunks are applied bottom-to-top to preserve line numbers.
fn apply_hunks(content: &str, hunks: &[Hunk]) -> Result<String, String> {
    let mut file_lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();

    // Sort hunks by old_start descending (apply bottom-to-top)
    let mut sorted_hunks: Vec<&Hunk> = hunks.iter().collect();
    sorted_hunks.sort_by_key(|h| std::cmp::Reverse(h.old_start));

    for hunk in sorted_hunks {
        let start_idx = hunk.old_start.saturating_sub(1); // Convert 1-indexed to 0-indexed

        // Verify context lines match (basic verification)
        for (offset, ctx_line) in &hunk.context {
            let line_idx = start_idx + offset;
            if line_idx < file_lines.len() && file_lines[line_idx].trim() != ctx_line.trim() {
                tracing::warn!(
                    "Context mismatch at line {}: expected {:?}, found {:?}",
                    line_idx + 1,
                    ctx_line,
                    file_lines[line_idx]
                );
            }
        }

        // Find and remove the old lines, then insert new lines
        let mut lines_to_remove: Vec<usize> = Vec::new();
        let mut search_idx = start_idx;

        for removal in &hunk.removals {
            // Find the line to remove starting from search_idx
            let found_idx = file_lines[search_idx..]
                .iter()
                .position(|line| line.trim() == removal.trim())
                .map(|pos| search_idx + pos);
            match found_idx {
                Some(idx) => {
                    lines_to_remove.push(idx);
                    search_idx = idx + 1;
                }
                None => {
                    return Err(format!(
                        "Could not find line to remove: {:?} near line {}",
                        removal,
                        start_idx + 1
                    ));
                }
            }
        }

        // Remove lines bottom-to-top
        lines_to_remove.sort_unstable();
        lines_to_remove.reverse();
        for idx in &lines_to_remove {
            file_lines.remove(*idx);
        }

        // Insert additions at the position of the first removal (or at start_idx)
        let insert_at = lines_to_remove
            .last()
            .copied()
            .unwrap_or(start_idx)
            .min(file_lines.len());

        for (j, addition) in hunk.additions.iter().enumerate() {
            file_lines.insert(insert_at + j, addition.clone());
        }
    }

    Ok(file_lines.join("\n"))
}

#[async_trait]
impl Tool for PatchTool {
    fn name(&self) -> &str {
        "patch"
    }

    fn description(&self) -> &str {
        "Apply a unified diff patch to a file. Use this for complex multi-line edits where the edit tool would be cumbersome. The diff must be in unified diff format with @@ hunk headers."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Required. Path to the file to patch. Always provide this exact field name: 'file_path'."
                },
                "diff": {
                    "type": "string",
                    "description": "Required. Unified diff content with @@ hunk headers."
                }
            },
            "required": ["file_path", "diff"],
            "examples": [
                {
                    "file_path": "src/lib.rs",
                    "diff": "@@ -1,3 +1,3 @@\n line1\n-old_line\n+new_line\n line3"
                }
            ]
        })
    }

    async fn execute(&self, args: &serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let file_path = match args.get("file_path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return Ok(ToolOutput::error("file_path parameter is required")),
        };

        let diff = match args.get("diff").and_then(|v| v.as_str()) {
            Some(d) => d,
            None => return Ok(ToolOutput::error("diff parameter is required")),
        };

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

        let hunks = match parse_hunks(diff) {
            Ok(h) => h,
            Err(e) => return Ok(ToolOutput::error(format!("Failed to parse diff: {}", e))),
        };

        let new_content = match apply_hunks(&content, &hunks) {
            Ok(c) => c,
            Err(e) => return Ok(ToolOutput::error(format!("Failed to apply patch: {}", e))),
        };

        // Preserve trailing newline if original had one
        let final_content = if content.ends_with('\n') && !new_content.ends_with('\n') {
            format!("{}\n", new_content)
        } else {
            new_content
        };

        let path_for_write = workspace_path.clone();
        let content_for_write = final_content.clone();
        match ctx
            .workspace_services
            .run_with_timeout("write_text", async move {
                fs.write_text(&path_for_write, &content_for_write).await
            })
            .await
        {
            Ok(_) => Ok(ToolOutput::success(format!(
                "Applied {} hunk(s) to {}",
                hunks.len(),
                display_path
            ))),
            Err(e) => Ok(ToolOutput::error(format!(
                "Failed to write patched file {}: {}",
                display_path, e
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hunk_header() {
        assert_eq!(parse_hunk_header("@@ -1,3 +1,3 @@").unwrap(), 1);
        assert_eq!(parse_hunk_header("@@ -10,5 +12,7 @@").unwrap(), 10);
        assert_eq!(parse_hunk_header("@@ -1 +1 @@ function name").unwrap(), 1);
    }

    #[test]
    fn test_parse_hunks_simple() {
        let diff = "@@ -1,3 +1,3 @@\n line1\n-old_line\n+new_line\n line3";
        let hunks = parse_hunks(diff).unwrap();
        assert_eq!(hunks.len(), 1);
        assert_eq!(hunks[0].old_start, 1);
        assert_eq!(hunks[0].removals, vec!["old_line"]);
        assert_eq!(hunks[0].additions, vec!["new_line"]);
    }

    #[test]
    fn test_apply_hunks_simple() {
        let content = "line1\nold_line\nline3";
        let diff = "@@ -1,3 +1,3 @@\n line1\n-old_line\n+new_line\n line3";
        let hunks = parse_hunks(diff).unwrap();
        let result = apply_hunks(content, &hunks).unwrap();
        assert_eq!(result, "line1\nnew_line\nline3");
    }

    #[tokio::test]
    async fn test_patch_tool() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("test.txt"), "line1\nold_line\nline3\n").unwrap();

        let tool = PatchTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(
                &serde_json::json!({
                    "file_path": "test.txt",
                    "diff": "@@ -1,3 +1,3 @@\n line1\n-old_line\n+new_line\n line3"
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(result.success);
        let content = std::fs::read_to_string(temp.path().join("test.txt")).unwrap();
        assert!(content.contains("new_line"));
        assert!(!content.contains("old_line"));
    }

    #[test]
    fn test_parse_hunks_no_header() {
        let diff = "just some text\nwithout hunks";
        assert!(parse_hunks(diff).is_err());
    }

    #[tokio::test]
    async fn test_patch_missing_params() {
        let tool = PatchTool;
        let ctx = ToolContext::new(std::path::PathBuf::from("/tmp"));

        let result = tool.execute(&serde_json::json!({}), &ctx).await.unwrap();
        assert!(!result.success);
    }

    #[test]
    fn test_patch_schema_is_canonical() {
        let tool = PatchTool;
        let params = tool.parameters();
        assert_eq!(params["additionalProperties"], false);
        assert_eq!(params["required"], serde_json::json!(["file_path", "diff"]));
        let examples = params["examples"].as_array().unwrap();
        assert_eq!(examples[0]["file_path"], "src/lib.rs");
        assert!(examples[0].get("path").is_none());
    }
}
