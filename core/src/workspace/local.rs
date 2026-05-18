//! Local filesystem-backed workspace implementation.
//!
//! [`LocalWorkspaceBackend`] preserves the historical "agent runs on the host
//! filesystem" behavior. It implements every workspace capability trait so
//! local sessions get the full tool surface (read, write, edit, patch, ls,
//! bash, grep, glob, git, git_stash, git_worktree).

use super::{
    default_path_input, has_windows_path_prefix, normalize_relative_path,
    pathbuf_to_workspace_path, validate_relative_pattern, CommandOutput, CommandRequest,
    WorkspaceCommandRunner, WorkspaceDirEntry, WorkspaceFileSystem, WorkspaceFileType,
    WorkspaceGit, WorkspaceGitBranch, WorkspaceGitCheckoutOutput, WorkspaceGitCheckoutRequest,
    WorkspaceGitCommit, WorkspaceGitCreateBranchRequest, WorkspaceGitCreateWorktreeRequest,
    WorkspaceGitDiffRequest, WorkspaceGitRemote, WorkspaceGitRemoveWorktreeRequest,
    WorkspaceGitStash, WorkspaceGitStashProvider, WorkspaceGitStashRequest, WorkspaceGitStatus,
    WorkspaceGitWorktree, WorkspaceGitWorktreeMutation, WorkspaceGitWorktreeProvider,
    WorkspaceGlobRequest, WorkspaceGlobResult, WorkspaceGrepRequest, WorkspaceGrepResult,
    WorkspacePath, WorkspacePathResolver, WorkspaceSearch, WorkspaceWriteOutcome,
};
use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use std::path::{Component, Path, PathBuf};

/// Local filesystem-backed workspace implementation.
#[derive(Debug)]
pub struct LocalWorkspaceBackend {
    pub(super) root: PathBuf,
}

impl LocalWorkspaceBackend {
    pub fn new(root: PathBuf) -> Self {
        let canonical = root.canonicalize();
        let root = match canonical {
            Ok(canonical) => canonical,
            Err(e) => {
                tracing::warn!(
                    "LocalWorkspaceBackend: failed to canonicalize root '{}' at construction: {} \
                     (path resolution will fail-closed at first use)",
                    root.display(),
                    e
                );
                root
            }
        };
        Self { root }
    }

    fn local_path_for_read(&self, path: &WorkspacePath) -> Result<PathBuf> {
        a3s_common::tools::resolve_path(&self.root, path.as_str()).map_err(|e| anyhow!("{}", e))
    }

    fn local_path_for_write(&self, path: &WorkspacePath) -> Result<PathBuf> {
        let target = if path.is_root() {
            self.root.clone()
        } else {
            self.root.join(path.as_str())
        };

        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                anyhow!(
                    "Failed to create parent directories for {}: {}",
                    target.display(),
                    e
                )
            })?;
        }

        a3s_common::tools::resolve_path_for_write(&self.root, path.as_str())
            .map_err(|e| anyhow!("{}", e))
    }
}

impl WorkspacePathResolver for LocalWorkspaceBackend {
    fn normalize(&self, input: &str) -> Result<WorkspacePath> {
        normalize_local_path(&self.root, input)
    }
}

#[async_trait]
impl WorkspaceFileSystem for LocalWorkspaceBackend {
    async fn read_text(&self, path: &WorkspacePath) -> Result<String> {
        let resolved = self.local_path_for_read(path)?;
        tokio::fs::read_to_string(&resolved)
            .await
            .map_err(|e| anyhow!("Failed to read file {}: {}", resolved.display(), e))
    }

    async fn write_text(
        &self,
        path: &WorkspacePath,
        content: &str,
    ) -> Result<WorkspaceWriteOutcome> {
        let resolved = self.local_path_for_write(path)?;
        tokio::fs::write(&resolved, content)
            .await
            .map_err(|e| anyhow!("Failed to write file {}: {}", resolved.display(), e))?;

        Ok(WorkspaceWriteOutcome {
            bytes: content.len(),
            lines: content.lines().count(),
        })
    }

    async fn list_dir(&self, path: &WorkspacePath) -> Result<Vec<WorkspaceDirEntry>> {
        let target = self.local_path_for_read(path)?;
        if !target.exists() {
            bail!("Directory not found: {}", target.display());
        }
        if !target.is_dir() {
            bail!("Not a directory: {}", target.display());
        }

        let mut dir = tokio::fs::read_dir(&target)
            .await
            .map_err(|e| anyhow!("Failed to read directory {}: {}", target.display(), e))?;
        let mut entries = Vec::new();

        while let Some(entry) = dir.next_entry().await? {
            let name = entry.file_name().to_string_lossy().to_string();
            let file_type = entry.file_type().await;
            let metadata = entry.metadata().await;
            let (kind, size) = match (&file_type, &metadata) {
                (Ok(ft), Ok(m)) => {
                    let kind = if ft.is_dir() {
                        WorkspaceFileType::Directory
                    } else if ft.is_symlink() {
                        WorkspaceFileType::Symlink
                    } else {
                        WorkspaceFileType::File
                    };
                    (kind, m.len())
                }
                _ => (WorkspaceFileType::Unknown, 0),
            };
            entries.push(WorkspaceDirEntry { name, kind, size });
        }

        Ok(entries)
    }
}

#[async_trait]
impl WorkspaceSearch for LocalWorkspaceBackend {
    async fn glob(&self, request: WorkspaceGlobRequest) -> Result<WorkspaceGlobResult> {
        validate_relative_pattern(&request.pattern, "glob pattern")?;
        let base = self.local_path_for_read(&request.base)?;
        let full_pattern = base.join(&request.pattern);
        let full_pattern = full_pattern.to_string_lossy().replace('\\', "/");

        let entries = glob::glob(&full_pattern)
            .map_err(|e| anyhow!("Invalid glob pattern '{}': {}", request.pattern, e))?;

        let mut matches = Vec::new();
        for entry in entries {
            match entry {
                Ok(path) => {
                    if let Ok(relative) = path.strip_prefix(&self.root) {
                        matches.push(pathbuf_to_workspace_path(relative));
                    }
                }
                Err(e) => tracing::warn!("Glob entry error: {}", e),
            }
        }

        matches.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        Ok(WorkspaceGlobResult { matches })
    }

    async fn grep(&self, request: WorkspaceGrepRequest) -> Result<WorkspaceGrepResult> {
        if let Some(ref glob) = request.glob {
            validate_relative_pattern(glob, "grep glob filter")?;
        }

        let regex_pattern = if request.case_insensitive {
            format!("(?i){}", request.pattern)
        } else {
            request.pattern.clone()
        };
        let regex = regex::Regex::new(&regex_pattern)
            .map_err(|e| anyhow!("Invalid regex pattern '{}': {}", request.pattern, e))?;

        let search_path = self.local_path_for_read(&request.base)?;
        let mut builder = ignore::WalkBuilder::new(&search_path);
        builder.hidden(false).git_ignore(true).git_global(true);

        if let Some(ref glob_pat) = request.glob {
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
                Ok(content) => content,
                Err(_) => continue,
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
            let rel_path = file_path
                .strip_prefix(&self.root)
                .unwrap_or(file_path)
                .to_string_lossy()
                .replace('\\', "/");

            for &match_idx in &file_matches {
                if total_size > request.max_output_size {
                    return Ok(WorkspaceGrepResult {
                        output,
                        match_count,
                        file_count,
                        truncated: true,
                    });
                }

                match_count += 1;

                let start = match_idx.saturating_sub(request.context_lines);
                let end = (match_idx + request.context_lines + 1).min(lines.len());

                for (i, line) in lines[start..end].iter().enumerate() {
                    let abs_i = start + i;
                    let prefix = if abs_i == match_idx { ">" } else { " " };
                    let line = format!("{}{}:{}: {}\n", prefix, rel_path, abs_i + 1, line);
                    total_size += line.len();
                    output.push_str(&line);
                }

                if request.context_lines > 0 {
                    output.push_str("--\n");
                    total_size += 3;
                }
            }
        }

        Ok(WorkspaceGrepResult {
            output,
            match_count,
            file_count,
            truncated: false,
        })
    }
}

#[async_trait]
impl WorkspaceGit for LocalWorkspaceBackend {
    async fn is_repository(&self) -> Result<bool> {
        self.run_blocking_git(|root| Ok(crate::git::is_git_repo(&root)))
            .await
    }

    async fn status(&self) -> Result<WorkspaceGitStatus> {
        self.run_blocking_git(|root| {
            let status = crate::git::get_status(&root)?;
            Ok(WorkspaceGitStatus {
                branch: status.branch,
                commit: status.commit,
                is_worktree: status.is_worktree,
                is_dirty: status.is_dirty,
                dirty_count: status.dirty_count,
            })
        })
        .await
    }

    async fn log(&self, max_count: usize) -> Result<Vec<WorkspaceGitCommit>> {
        self.run_blocking_git(move |root| {
            Ok(crate::git::get_log(&root, max_count)?
                .into_iter()
                .map(|commit| WorkspaceGitCommit {
                    id: commit.id,
                    message: commit.message,
                    author: commit.author,
                    date: commit.date,
                })
                .collect())
        })
        .await
    }

    async fn list_branches(&self) -> Result<Vec<WorkspaceGitBranch>> {
        self.run_blocking_git(|root| {
            Ok(crate::git::list_branches(&root)?
                .into_iter()
                .map(|branch| WorkspaceGitBranch {
                    name: branch.name,
                    is_current: branch.is_current,
                })
                .collect())
        })
        .await
    }

    async fn create_branch(&self, request: WorkspaceGitCreateBranchRequest) -> Result<()> {
        self.run_blocking_git(move |root| {
            crate::git::create_branch(&root, &request.name, &request.base)
        })
        .await
    }

    async fn checkout(
        &self,
        request: WorkspaceGitCheckoutRequest,
    ) -> Result<WorkspaceGitCheckoutOutput> {
        let args = if request.force {
            vec![
                "checkout".to_string(),
                "--force".to_string(),
                request.refspec,
            ]
        } else {
            vec!["checkout".to_string(), request.refspec]
        };
        let (success, stdout, stderr) = self.run_git_command(args).await?;
        if !success {
            bail!("{}", stderr.trim_end());
        }
        Ok(WorkspaceGitCheckoutOutput { stdout })
    }

    async fn diff(&self, request: WorkspaceGitDiffRequest) -> Result<String> {
        self.run_blocking_git(move |root| crate::git::get_diff(&root, request.target.as_deref()))
            .await
    }

    async fn list_remotes(&self) -> Result<Vec<WorkspaceGitRemote>> {
        let (success, stdout, stderr) = self
            .run_git_command(vec!["remote".to_string(), "-v".to_string()])
            .await?;
        if !success {
            bail!("{}", stderr.trim_end());
        }

        Ok(stdout.lines().filter_map(parse_git_remote_line).collect())
    }
}

#[async_trait]
impl WorkspaceGitStashProvider for LocalWorkspaceBackend {
    async fn list_stashes(&self) -> Result<Vec<WorkspaceGitStash>> {
        self.run_blocking_git(|root| {
            Ok(crate::git::list_stashes(&root)?
                .into_iter()
                .map(|stash| WorkspaceGitStash {
                    index: stash.index,
                    message: stash.message,
                })
                .collect())
        })
        .await
    }

    async fn stash(&self, request: WorkspaceGitStashRequest) -> Result<()> {
        self.run_blocking_git(move |root| {
            crate::git::stash(&root, request.message.as_deref(), request.include_untracked)
        })
        .await
    }
}

#[async_trait]
impl WorkspaceGitWorktreeProvider for LocalWorkspaceBackend {
    async fn list_worktrees(&self) -> Result<Vec<WorkspaceGitWorktree>> {
        self.run_blocking_git(|root| {
            Ok(crate::git::list_worktrees(&root)?
                .into_iter()
                .map(|worktree| WorkspaceGitWorktree {
                    path: worktree.path,
                    branch: worktree.branch,
                    is_bare: worktree.is_bare,
                    is_detached: worktree.is_detached,
                })
                .collect())
        })
        .await
    }

    async fn create_worktree(
        &self,
        request: WorkspaceGitCreateWorktreeRequest,
    ) -> Result<WorkspaceGitWorktreeMutation> {
        let branch = request.branch;
        let path = request
            .path
            .map(|path| {
                let path = PathBuf::from(path);
                if path.is_absolute() {
                    path
                } else {
                    self.root.join(path)
                }
            })
            .unwrap_or_else(|| default_local_worktree_path(&self.root, &branch));
        let display_path = path.display().to_string();
        let new_branch = request.new_branch;
        let branch_for_git = branch.clone();

        self.run_blocking_git(move |root| {
            crate::git::create_worktree(&root, &branch_for_git, &path, new_branch)
        })
        .await?;

        Ok(WorkspaceGitWorktreeMutation {
            path: display_path,
            branch: Some(branch),
        })
    }

    async fn remove_worktree(
        &self,
        request: WorkspaceGitRemoveWorktreeRequest,
    ) -> Result<WorkspaceGitWorktreeMutation> {
        let path = PathBuf::from(request.path);
        let display_path = path.display().to_string();
        let force = request.force;

        self.run_blocking_git(move |root| crate::git::remove_worktree(&root, &path, force))
            .await?;

        Ok(WorkspaceGitWorktreeMutation {
            path: display_path,
            branch: None,
        })
    }
}

#[async_trait]
impl WorkspaceCommandRunner for LocalWorkspaceBackend {
    async fn exec(&self, request: CommandRequest) -> Result<CommandOutput> {
        #[cfg(windows)]
        if let Some(output) =
            crate::tools::builtin::bash::maybe_execute_simple_windows_http_command(&request.command)
                .await
        {
            let exit_code = output
                .metadata
                .as_ref()
                .and_then(|m| m.get("exit_code"))
                .and_then(|v| v.as_i64())
                .map(|v| v as i32)
                .unwrap_or(if output.success { 0 } else { -1 });
            return Ok(CommandOutput {
                output: output.content,
                exit_code,
                timed_out: false,
            });
        }

        let timeout_secs = request.timeout_ms / 1000;
        let mut child = crate::tools::builtin::bash::spawn_shell(
            &request.command,
            &self.root,
            request.env.as_deref(),
        )
        .map_err(|e| anyhow!("Failed to spawn shell: {}", e))?;

        let (output, timed_out) = crate::tools::process::read_process_output(
            &mut child,
            timeout_secs,
            request.output_observer.as_deref(),
        )
        .await;

        if timed_out {
            return Ok(CommandOutput {
                output,
                exit_code: -1,
                timed_out: true,
            });
        }

        let status = child
            .wait()
            .await
            .map_err(|e| anyhow!("Failed to wait for shell: {}", e))?;
        let exit_code = status.code().unwrap_or(-1);

        Ok(CommandOutput {
            output,
            exit_code,
            timed_out: false,
        })
    }
}

impl LocalWorkspaceBackend {
    async fn run_blocking_git<T, F>(&self, operation: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(PathBuf) -> Result<T> + Send + 'static,
    {
        let root = self.root.clone();
        tokio::task::spawn_blocking(move || operation(root))
            .await
            .map_err(|e| anyhow!("Git worker failed: {}", e))?
    }

    async fn run_git_command(&self, args: Vec<String>) -> Result<(bool, String, String)> {
        tokio::task::spawn_blocking(crate::git::ensure_git_installed)
            .await
            .map_err(|e| anyhow!("Git worker failed: {}", e))??;

        let output = tokio::process::Command::new("git")
            .arg("-C")
            .arg(self.root.as_os_str())
            .args(&args)
            .output()
            .await
            .map_err(|e| anyhow!("Failed to execute git: {}", e))?;

        Ok((
            output.status.success(),
            String::from_utf8_lossy(&output.stdout).to_string(),
            String::from_utf8_lossy(&output.stderr).to_string(),
        ))
    }
}

fn parse_git_remote_line(line: &str) -> Option<WorkspaceGitRemote> {
    let mut parts = line.split_whitespace();
    let name = parts.next()?;
    let url = parts.next()?;
    let direction = parts
        .next()
        .unwrap_or_default()
        .trim_start_matches('(')
        .trim_end_matches(')');

    Some(WorkspaceGitRemote {
        name: name.to_string(),
        url: url.to_string(),
        direction: direction.to_string(),
    })
}

fn default_local_worktree_path(root: &Path, branch: &str) -> PathBuf {
    let repo_name = root
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| "repo".to_string());
    root.parent()
        .unwrap_or(root)
        .join(format!("{repo_name}-{branch}"))
}

pub(super) fn normalize_local_path(root: &Path, input: &str) -> Result<WorkspacePath> {
    let input = default_path_input(input);
    let candidate = Path::new(input);

    if candidate.is_absolute() {
        let root = normalize_absolute_path(root)?;
        let target = normalize_absolute_path(candidate)?;
        if !target.starts_with(&root) {
            bail!(
                "Workspace boundary violation: path '{}' escapes workspace '{}'",
                input,
                root.display()
            );
        }
        let relative = target
            .strip_prefix(&root)
            .map_err(|_| anyhow!("Failed to compute workspace-relative path"))?;
        return Ok(pathbuf_to_workspace_path(relative));
    }

    if has_windows_path_prefix(input) {
        bail!("Absolute paths are not supported by this workspace backend");
    }

    let normalized_input = input.replace('\\', "/");
    let path = Path::new(&normalized_input);
    if path.is_absolute() {
        bail!("Absolute paths are not supported by this workspace backend");
    }

    let relative = normalize_relative_path(path)?;
    Ok(pathbuf_to_workspace_path(&relative))
}

fn normalize_absolute_path(path: &Path) -> Result<PathBuf> {
    let lexical = normalize_absolute_path_lexical(path)?;
    if let Ok(canonical) = lexical.canonicalize() {
        return Ok(canonical);
    }

    let mut current = lexical.as_path();
    let mut suffix = Vec::new();
    while !current.exists() {
        let Some(file_name) = current.file_name() else {
            return Ok(lexical);
        };
        suffix.push(file_name.to_os_string());
        let Some(parent) = current.parent() else {
            return Ok(lexical);
        };
        current = parent;
    }

    let mut normalized = current.canonicalize().unwrap_or_else(|_| {
        normalize_absolute_path_lexical(current).unwrap_or_else(|_| current.into())
    });
    for part in suffix.iter().rev() {
        normalized.push(part);
    }
    Ok(normalized)
}

fn normalize_absolute_path_lexical(path: &Path) -> Result<PathBuf> {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => out.push(prefix.as_os_str()),
            Component::RootDir => out.push(Path::new(std::path::MAIN_SEPARATOR_STR)),
            Component::CurDir => {}
            Component::Normal(part) => out.push(part),
            Component::ParentDir => {
                if !out.pop() {
                    bail!("Invalid absolute path");
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::WorkspaceServices;
    use super::*;

    #[tokio::test]
    async fn local_backend_reads_writes_and_lists() {
        let temp = tempfile::tempdir().unwrap();
        let services = WorkspaceServices::local(temp.path());
        let path = services.normalize_path("dir/file.txt").unwrap();

        let written = services
            .fs()
            .write_text(&path, "hello\nworld\n")
            .await
            .unwrap();
        assert_eq!(written.bytes, 12);
        assert_eq!(written.lines, 2);

        let content = services.fs().read_text(&path).await.unwrap();
        assert_eq!(content, "hello\nworld\n");

        let dir = services.normalize_path("dir").unwrap();
        let entries = services.fs().list_dir(&dir).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "file.txt");
    }

    #[tokio::test]
    async fn local_backend_searches_glob_and_grep() {
        let temp = tempfile::tempdir().unwrap();
        let services = WorkspaceServices::local(temp.path());
        services
            .fs()
            .write_text(
                &services.normalize_path("src/main.rs").unwrap(),
                "fn main() {\n    println!(\"hello\");\n}\n",
            )
            .await
            .unwrap();
        services
            .fs()
            .write_text(
                &services.normalize_path("README.md").unwrap(),
                "hello from docs\n",
            )
            .await
            .unwrap();

        let search = services.search().expect("local backend supports search");
        let glob = search
            .glob(WorkspaceGlobRequest {
                base: services.normalize_path("src").unwrap(),
                pattern: "*.rs".to_string(),
            })
            .await
            .unwrap();
        assert_eq!(glob.matches[0].as_str(), "src/main.rs");

        let grep = search
            .grep(WorkspaceGrepRequest {
                base: WorkspacePath::root(),
                pattern: "hello".to_string(),
                glob: Some("**/*.rs".to_string()),
                context_lines: 0,
                case_insensitive: false,
                max_output_size: 1024,
            })
            .await
            .unwrap();
        assert_eq!(grep.match_count, 1);
        assert_eq!(grep.file_count, 1);
        assert!(grep.output.contains("src/main.rs:2"));
    }

    #[test]
    fn local_backend_rejects_absolute_paths_outside_workspace() {
        let temp = tempfile::tempdir().unwrap();
        let services = WorkspaceServices::local(temp.path());
        let outside = temp.path().parent().unwrap().join("secret.txt");
        let err = services
            .normalize_path(outside.to_str().unwrap())
            .expect_err("outside absolute path should be rejected");
        assert!(err.to_string().contains("escapes workspace"));
    }

    #[test]
    fn local_backend_rejects_backslash_parent_escape() {
        let temp = tempfile::tempdir().unwrap();
        let services = WorkspaceServices::local(temp.path());
        let err = services
            .normalize_path(r"..\secret.txt")
            .expect_err("backslash parent traversal should be rejected");
        assert!(err.to_string().contains("escapes workspace"));
    }

    #[test]
    fn local_backend_allows_absolute_paths_inside_workspace() {
        let temp = tempfile::tempdir().unwrap();
        let services = WorkspaceServices::local(temp.path());
        let absolute = temp.path().join("src/main.rs");
        let path = services
            .normalize_path(absolute.to_str().unwrap())
            .expect("absolute path inside workspace should normalize");
        assert_eq!(path.as_str(), "src/main.rs");
    }
}
