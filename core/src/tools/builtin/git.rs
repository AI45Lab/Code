//! Git tool backed by workspace Git services.
//!
//! The tool keeps the model-facing contract stable while the concrete Git
//! implementation can be local, remote, browser-backed, or DFS-backed.

use crate::tools::types::{Tool, ToolContext, ToolOutput};
use crate::workspace::{
    WorkspaceGit, WorkspaceGitCheckoutRequest, WorkspaceGitCreateBranchRequest,
    WorkspaceGitCreateWorktreeRequest, WorkspaceGitDiffRequest, WorkspaceGitRemote,
    WorkspaceGitRemoveWorktreeRequest, WorkspaceGitStashProvider, WorkspaceGitStashRequest,
    WorkspaceGitWorktreeProvider,
};
use anyhow::Result;
use async_trait::async_trait;

pub struct GitTool;

#[async_trait]
impl Tool for GitTool {
    fn name(&self) -> &str {
        "git"
    }

    fn description(&self) -> &str {
        "Execute Git operations for the current workspace. Supports: status, log, branch, checkout, diff, stash, remote, and worktree management."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "command": {
                    "type": "string",
                    "enum": [
                        "status", "log", "branch", "checkout", "diff", "stash", "remote", "worktree"
                    ],
                    "description": "Required. Git command to execute."
                },
                "subcommand": {
                    "type": "string",
                    "enum": ["list", "create", "remove"],
                    "description": "Worktree subcommand. Defaults to list."
                },
                "name": {
                    "type": "string",
                    "description": "Branch name for branch/checkout/worktree operations."
                },
                "path": {
                    "type": "string",
                    "description": "Path for worktree operations."
                },
                "ref": {
                    "type": "string",
                    "description": "Reference (branch, tag, commit) for checkout."
                },
                "force": {
                    "type": "boolean",
                    "description": "Force checkout/create even if it loses changes."
                },
                "target": {
                    "type": "string",
                    "description": "Target ref for diff (e.g., HEAD~1, main). If omitted, diffs working tree."
                },
                "max_count": {
                    "type": "integer",
                    "description": "Maximum number of log entries to show (default 10)."
                },
                "message": {
                    "type": "string",
                    "description": "Message for stash."
                },
                "include_untracked": {
                    "type": "boolean",
                    "description": "Include untracked files in stash (default false)."
                },
                "remote_name": {
                    "type": "string",
                    "description": "Remote name (reserved for provider-specific filtering)."
                },
                "new_branch": {
                    "type": "boolean",
                    "description": "Create a new branch for worktree (default true)."
                },
                "base": {
                    "type": "string",
                    "description": "Base ref for new branch (default HEAD)."
                }
            },
            "required": ["command"],
            "examples": [
                {"command": "status"},
                {"command": "log", "max_count": 5},
                {"command": "branch"},
                {"command": "branch", "name": "feature-x"},
                {"command": "checkout", "ref": "feature-x"},
                {"command": "diff"},
                {"command": "diff", "target": "HEAD~1"},
                {"command": "stash"},
                {"command": "stash", "message": "WIP: work in progress"},
                {"command": "remote"},
                {"command": "worktree", "subcommand": "list"}
            ]
        })
    }

    async fn execute(&self, args: &serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let command = match args.get("command").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return Ok(ToolOutput::error("command parameter is required")),
        };

        let Some(git) = ctx.workspace_services.git() else {
            return Ok(ToolOutput::error(
                "Git is not available for this workspace backend",
            ));
        };

        match git.is_repository().await {
            Ok(true) => {}
            Ok(false) => {
                return Ok(ToolOutput::error(format!(
                    "Not a git repository: {}",
                    ctx.workspace_services.workspace_ref().display_root
                )))
            }
            Err(e) => {
                return Ok(ToolOutput::error(format!(
                    "Failed to inspect git repository: {e}"
                )))
            }
        }

        match command {
            "status" => self.status(ctx, git.as_ref()).await,
            "log" => self.log(args, git.as_ref()).await,
            "branch" => self.branch(args, git.as_ref()).await,
            "checkout" => self.checkout(args, git.as_ref()).await,
            "diff" => self.diff(args, git.as_ref()).await,
            "stash" => {
                let Some(stash) = ctx.workspace_services.git_stash() else {
                    return Ok(ToolOutput::error(
                        "Stash operations are not supported by this workspace backend",
                    ));
                };
                self.stash(args, stash.as_ref()).await
            }
            "remote" => self.remote(git.as_ref()).await,
            "worktree" => {
                let Some(worktree) = ctx.workspace_services.git_worktree() else {
                    return Ok(ToolOutput::error(
                        "Worktree operations are not supported by this workspace backend",
                    ));
                };
                self.worktree(args, worktree.as_ref()).await
            }
            _ => Ok(ToolOutput::error(format!(
                "Unknown command: {command}. Use: status, log, branch, checkout, diff, stash, remote, worktree"
            ))),
        }
    }
}

impl GitTool {
    /// Show repository status.
    async fn status(&self, ctx: &ToolContext, git: &dyn WorkspaceGit) -> Result<ToolOutput> {
        match git.status().await {
            Ok(status) => {
                let status_str = if status.is_dirty {
                    format!("{} uncommitted change(s)", status.dirty_count)
                } else {
                    "clean".to_string()
                };

                Ok(ToolOutput::success(format!(
                    "Workspace: {}\n\
                     Branch:    {}\n\
                     Commit:    {}\n\
                     Status:    {}\n\
                     Worktree:  {}",
                    ctx.workspace_services.workspace_ref().display_root,
                    status.branch,
                    status.commit,
                    status_str,
                    if status.is_worktree {
                        "yes (linked)"
                    } else {
                        "no (main)"
                    }
                ))
                .with_metadata(serde_json::json!({
                    "branch": status.branch,
                    "is_worktree": status.is_worktree,
                    "dirty_count": status.dirty_count,
                    "is_dirty": status.is_dirty,
                })))
            }
            Err(e) => Ok(ToolOutput::error(format!("Failed to get status: {e}"))),
        }
    }

    /// Show commit log.
    async fn log(&self, args: &serde_json::Value, git: &dyn WorkspaceGit) -> Result<ToolOutput> {
        let max_count = args.get("max_count").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

        match git.log(max_count).await {
            Ok(commits) => {
                if commits.is_empty() {
                    return Ok(ToolOutput::success("No commits found."));
                }

                let entries: Vec<String> = commits
                    .iter()
                    .map(|commit| {
                        format!(
                            "{} - {} ({})\n  {}",
                            short_commit_id(&commit.id),
                            commit.author,
                            commit.date,
                            commit.message
                        )
                    })
                    .collect();

                Ok(ToolOutput::success(format!(
                    "Commit log ({} entries):\n\n{}",
                    commits.len(),
                    entries.join("\n\n")
                ))
                .with_metadata(serde_json::json!({ "count": commits.len() })))
            }
            Err(e) => Ok(ToolOutput::error(format!("Failed to get log: {e}"))),
        }
    }

    /// Branch operations: list or create.
    async fn branch(&self, args: &serde_json::Value, git: &dyn WorkspaceGit) -> Result<ToolOutput> {
        let name = args.get("name").and_then(|v| v.as_str());

        if let Some(branch_name) = name {
            let base = args.get("base").and_then(|v| v.as_str()).unwrap_or("HEAD");

            match git
                .create_branch(WorkspaceGitCreateBranchRequest {
                    name: branch_name.to_string(),
                    base: base.to_string(),
                })
                .await
            {
                Ok(_) => Ok(ToolOutput::success(format!(
                    "Created branch: {} (based on {})",
                    branch_name, base
                ))
                .with_metadata(serde_json::json!({ "branch": branch_name, "base": base }))),
                Err(e) => Ok(ToolOutput::error(format!("Failed to create branch: {e}"))),
            }
        } else {
            match git.list_branches().await {
                Ok(branches) => {
                    if branches.is_empty() {
                        return Ok(ToolOutput::success("No branches found."));
                    }

                    let entries: Vec<String> = branches
                        .iter()
                        .map(|branch| {
                            let prefix = if branch.is_current { "* " } else { "  " };
                            format!("{}{}", prefix, branch.name)
                        })
                        .collect();

                    Ok(
                        ToolOutput::success(format!("Branches:\n{}", entries.join("\n")))
                            .with_metadata(serde_json::json!({ "count": branches.len() })),
                    )
                }
                Err(e) => Ok(ToolOutput::error(format!("Failed to list branches: {e}"))),
            }
        }
    }

    /// Checkout a branch or commit.
    async fn checkout(
        &self,
        args: &serde_json::Value,
        git: &dyn WorkspaceGit,
    ) -> Result<ToolOutput> {
        let refspec = match args.get("ref").and_then(|v| v.as_str()) {
            Some(r) => r,
            None => return Ok(ToolOutput::error("ref parameter is required for checkout")),
        };

        let force = args.get("force").and_then(|v| v.as_bool()).unwrap_or(false);

        match git
            .checkout(WorkspaceGitCheckoutRequest {
                refspec: refspec.to_string(),
                force,
            })
            .await
        {
            Ok(output) => Ok(ToolOutput::success(format!(
                "Checked out: {}{}",
                refspec,
                if output.stdout.trim().is_empty() {
                    String::new()
                } else {
                    format!("\n{}", output.stdout)
                }
            ))),
            Err(e) => Ok(ToolOutput::error(format!("Failed to checkout: {e}"))),
        }
    }

    /// Show diff.
    async fn diff(&self, args: &serde_json::Value, git: &dyn WorkspaceGit) -> Result<ToolOutput> {
        let target = args
            .get("target")
            .and_then(|v| v.as_str())
            .map(ToString::to_string);

        match git.diff(WorkspaceGitDiffRequest { target }).await {
            Ok(diff) => {
                if diff.trim().is_empty() {
                    return Ok(ToolOutput::success("No changes.".to_string()));
                }
                Ok(ToolOutput::success(diff))
            }
            Err(e) => Ok(ToolOutput::error(format!("Failed to get diff: {e}"))),
        }
    }

    /// Stash operations.
    async fn stash(
        &self,
        args: &serde_json::Value,
        stash: &dyn WorkspaceGitStashProvider,
    ) -> Result<ToolOutput> {
        let message = args
            .get("message")
            .and_then(|v| v.as_str())
            .map(ToString::to_string);
        let include_untracked = args
            .get("include_untracked")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if message.is_some() || include_untracked {
            match stash
                .stash(WorkspaceGitStashRequest {
                    message,
                    include_untracked,
                })
                .await
            {
                Ok(_) => Ok(ToolOutput::success("Created stash".to_string())),
                Err(e) => Ok(ToolOutput::error(format!("Failed to stash: {e}"))),
            }
        } else {
            match stash.list_stashes().await {
                Ok(stashes) => {
                    if stashes.is_empty() {
                        return Ok(ToolOutput::success("No stashes found.".to_string()));
                    }

                    let entries: Vec<String> = stashes
                        .iter()
                        .map(|stash| format!("{}: {}", stash.index, stash.message))
                        .collect();

                    Ok(
                        ToolOutput::success(format!("Stashes:\n{}", entries.join("\n")))
                            .with_metadata(serde_json::json!({ "count": stashes.len() })),
                    )
                }
                Err(e) => Ok(ToolOutput::error(format!("Failed to list stashes: {e}"))),
            }
        }
    }

    /// Show remote information.
    async fn remote(&self, git: &dyn WorkspaceGit) -> Result<ToolOutput> {
        match git.list_remotes().await {
            Ok(remotes) if remotes.is_empty() => {
                Ok(ToolOutput::success("No remotes configured.".to_string()))
            }
            Ok(remotes) => {
                let entries: Vec<String> = remotes.iter().map(format_remote).collect();
                Ok(ToolOutput::success(format!(
                    "Remotes:\n{}",
                    entries.join("\n")
                )))
            }
            Err(e) => Ok(ToolOutput::error(format!("Failed to list remotes: {e}"))),
        }
    }

    /// Worktree operations.
    async fn worktree(
        &self,
        args: &serde_json::Value,
        worktree: &dyn WorkspaceGitWorktreeProvider,
    ) -> Result<ToolOutput> {
        let subcommand = args
            .get("subcommand")
            .and_then(|v| v.as_str())
            .unwrap_or("list");

        match subcommand {
            "list" => self.list_worktrees(worktree).await,
            "create" => self.create_worktree(args, worktree).await,
            "remove" => self.remove_worktree(args, worktree).await,
            _ => Ok(ToolOutput::error(format!(
                "Unknown worktree subcommand: {subcommand}. Use: list, create, remove"
            ))),
        }
    }

    /// List all worktrees.
    async fn list_worktrees(
        &self,
        worktree: &dyn WorkspaceGitWorktreeProvider,
    ) -> Result<ToolOutput> {
        match worktree.list_worktrees().await {
            Ok(worktrees) => {
                if worktrees.is_empty() {
                    return Ok(ToolOutput::success("No worktrees found."));
                }

                let entries: Vec<String> = worktrees
                    .iter()
                    .map(|worktree| {
                        let suffix = if worktree.is_bare {
                            " (bare)".to_string()
                        } else if worktree.is_detached {
                            " (detached)".to_string()
                        } else {
                            format!(" [{}]", worktree.branch)
                        };
                        format!("  {}{}", worktree.path, suffix)
                    })
                    .collect();

                Ok(ToolOutput::success(format!(
                    "Worktrees ({}):\n{}",
                    worktrees.len(),
                    entries.join("\n")
                ))
                .with_metadata(serde_json::json!({ "count": worktrees.len() })))
            }
            Err(e) => Ok(ToolOutput::error(format!("Failed to list worktrees: {e}"))),
        }
    }

    /// Create a new worktree.
    async fn create_worktree(
        &self,
        args: &serde_json::Value,
        worktree: &dyn WorkspaceGitWorktreeProvider,
    ) -> Result<ToolOutput> {
        let branch = match args
            .get("name")
            .or_else(|| args.get("branch"))
            .and_then(|v| v.as_str())
        {
            Some(b) => b,
            None => {
                return Ok(ToolOutput::error(
                    "branch name is required for worktree create",
                ))
            }
        };

        let new_branch = args
            .get("new_branch")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .map(ToString::to_string);

        match worktree
            .create_worktree(WorkspaceGitCreateWorktreeRequest {
                branch: branch.to_string(),
                path,
                new_branch,
            })
            .await
        {
            Ok(result) => Ok(ToolOutput::success(format!(
                "Created worktree at: {}\nBranch: {branch}",
                result.path
            ))
            .with_metadata(serde_json::json!({
                "path": result.path,
                "branch": branch,
            }))),
            Err(e) => Ok(ToolOutput::error(format!("Failed to create worktree: {e}"))),
        }
    }

    /// Remove a worktree.
    async fn remove_worktree(
        &self,
        args: &serde_json::Value,
        worktree: &dyn WorkspaceGitWorktreeProvider,
    ) -> Result<ToolOutput> {
        let path = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => {
                return Ok(ToolOutput::error(
                    "path parameter is required for worktree remove",
                ))
            }
        };

        let force = args.get("force").and_then(|v| v.as_bool()).unwrap_or(false);

        match worktree
            .remove_worktree(WorkspaceGitRemoveWorktreeRequest {
                path: path.to_string(),
                force,
            })
            .await
        {
            Ok(result) => Ok(ToolOutput::success(format!(
                "Removed worktree at: {}",
                result.path
            ))),
            Err(e) => Ok(ToolOutput::error(format!("Failed to remove worktree: {e}"))),
        }
    }
}

fn short_commit_id(id: &str) -> &str {
    id.get(..7).unwrap_or(id)
}

fn format_remote(remote: &WorkspaceGitRemote) -> String {
    if remote.direction.is_empty() {
        format!("{}\t{}", remote.name, remote.url)
    } else {
        format!("{}\t{} ({})", remote.name, remote.url, remote.direction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_git_not_installed() {
        // This test checks that the local provider handles non-git repos properly.
        let tool = GitTool;
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(dir.path().to_path_buf());
        let args = serde_json::json!({"command": "status"});
        let out = tool.execute(&args, &ctx).await.unwrap();
        assert!(!out.success);
        assert!(out.content.contains("Not a git repository"));
    }

    #[tokio::test]
    async fn test_missing_command() {
        let tool = GitTool;
        let ctx = ToolContext::new(PathBuf::from("/tmp"));
        let args = serde_json::json!({});
        let out = tool.execute(&args, &ctx).await.unwrap();
        assert!(!out.success);
        assert!(out.content.contains("command parameter is required"));
    }
}
