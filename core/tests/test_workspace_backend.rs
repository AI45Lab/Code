use a3s_code_core::tools::{ArtifactStoreLimits, ToolExecutor};
use a3s_code_core::{
    CommandOutput, CommandRequest, WorkspaceCommandRunner, WorkspaceDirEntry, WorkspaceFileSystem,
    WorkspaceFileType, WorkspaceGit, WorkspaceGitBranch, WorkspaceGitCheckoutOutput,
    WorkspaceGitCheckoutRequest, WorkspaceGitCommit, WorkspaceGitCreateBranchRequest,
    WorkspaceGitCreateWorktreeRequest, WorkspaceGitDiffRequest, WorkspaceGitRemote,
    WorkspaceGitRemoveWorktreeRequest, WorkspaceGitStash, WorkspaceGitStashProvider,
    WorkspaceGitStashRequest, WorkspaceGitStatus, WorkspaceGitWorktree,
    WorkspaceGitWorktreeMutation, WorkspaceGitWorktreeProvider, WorkspaceGlobRequest,
    WorkspaceGlobResult, WorkspaceGrepRequest, WorkspaceGrepResult, WorkspacePath, WorkspaceRef,
    WorkspaceSearch, WorkspaceServices, WorkspaceWriteOutcome,
};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Default)]
struct MemoryWorkspace {
    files: RwLock<HashMap<String, String>>,
}

impl MemoryWorkspace {
    fn insert(&self, path: &str, content: &str) {
        self.files
            .write()
            .unwrap()
            .insert(path.to_string(), content.to_string());
    }

    fn read_raw(&self, path: &str) -> Option<String> {
        self.files.read().unwrap().get(path).cloned()
    }
}

#[async_trait]
impl WorkspaceFileSystem for MemoryWorkspace {
    async fn read_text(&self, path: &WorkspacePath) -> Result<String> {
        self.files
            .read()
            .unwrap()
            .get(path.as_str())
            .cloned()
            .ok_or_else(|| anyhow!("missing remote file: {}", path.as_str()))
    }

    async fn write_text(
        &self,
        path: &WorkspacePath,
        content: &str,
    ) -> Result<WorkspaceWriteOutcome> {
        self.insert(path.as_str(), content);
        Ok(WorkspaceWriteOutcome {
            bytes: content.len(),
            lines: content.lines().count(),
        })
    }

    async fn list_dir(&self, path: &WorkspacePath) -> Result<Vec<WorkspaceDirEntry>> {
        let prefix = if path.is_root() {
            String::new()
        } else {
            format!("{}/", path.as_str())
        };
        let files = self.files.read().unwrap();
        let mut entries = HashMap::<String, WorkspaceDirEntry>::new();

        for (file_path, content) in files.iter() {
            if !file_path.starts_with(&prefix) {
                continue;
            }

            let remaining = &file_path[prefix.len()..];
            if remaining.is_empty() {
                continue;
            }

            let (name, kind, size) = match remaining.split_once('/') {
                Some((dir, _)) => (dir.to_string(), WorkspaceFileType::Directory, 0),
                None => (
                    remaining.to_string(),
                    WorkspaceFileType::File,
                    content.len() as u64,
                ),
            };

            entries
                .entry(name.clone())
                .or_insert(WorkspaceDirEntry { name, kind, size });
        }

        Ok(entries.into_values().collect())
    }
}

#[async_trait]
impl WorkspaceSearch for MemoryWorkspace {
    async fn glob(&self, request: WorkspaceGlobRequest) -> Result<WorkspaceGlobResult> {
        let pattern =
            glob::Pattern::new(&request.pattern).map_err(|e| anyhow!("invalid glob: {}", e))?;
        let base_prefix = if request.base.is_root() {
            String::new()
        } else {
            format!("{}/", request.base.as_str())
        };
        let files = self.files.read().unwrap();
        let mut matches = Vec::new();

        for file_path in files.keys() {
            if !file_path.starts_with(&base_prefix) {
                continue;
            }
            let remaining = &file_path[base_prefix.len()..];
            if pattern.matches(remaining) {
                matches.push(WorkspacePath::from_normalized(file_path.clone()));
            }
        }

        matches.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        Ok(WorkspaceGlobResult { matches })
    }

    async fn grep(&self, request: WorkspaceGrepRequest) -> Result<WorkspaceGrepResult> {
        let pattern = if request.case_insensitive {
            format!("(?i){}", request.pattern)
        } else {
            request.pattern.clone()
        };
        let regex = regex::Regex::new(&pattern)?;
        let glob = request
            .glob
            .as_deref()
            .map(glob::Pattern::new)
            .transpose()
            .map_err(|e| anyhow!("invalid glob: {}", e))?;
        let base_prefix = if request.base.is_root() {
            String::new()
        } else {
            format!("{}/", request.base.as_str())
        };
        let files = self.files.read().unwrap();
        let mut output = String::new();
        let mut match_count = 0;
        let mut matched_files = 0;

        for (file_path, content) in files.iter() {
            if !file_path.starts_with(&base_prefix) {
                continue;
            }
            let remaining = &file_path[base_prefix.len()..];
            if glob.as_ref().is_some_and(|glob| !glob.matches(remaining)) {
                continue;
            }

            let mut file_matched = false;
            for (idx, line) in content.lines().enumerate() {
                if regex.is_match(line) {
                    file_matched = true;
                    match_count += 1;
                    output.push_str(&format!(">{}:{}: {}\n", file_path, idx + 1, line));
                }
            }
            if file_matched {
                matched_files += 1;
            }
        }

        Ok(WorkspaceGrepResult {
            output,
            match_count,
            file_count: matched_files,
            truncated: false,
        })
    }
}

#[derive(Default)]
struct RecordingRunner {
    calls: RwLock<Vec<(String, u64)>>,
}

impl RecordingRunner {
    fn calls(&self) -> Vec<(String, u64)> {
        self.calls.read().unwrap().clone()
    }
}

#[async_trait]
impl WorkspaceCommandRunner for RecordingRunner {
    async fn exec(&self, request: CommandRequest) -> Result<CommandOutput> {
        self.calls
            .write()
            .unwrap()
            .push((request.command.clone(), request.timeout_ms));
        Ok(CommandOutput {
            output: format!("remote runner executed: {}\n", request.command),
            exit_code: 0,
            timed_out: false,
        })
    }
}

#[derive(Default)]
struct RecordingGit {
    calls: RwLock<Vec<String>>,
}

impl RecordingGit {
    fn calls(&self) -> Vec<String> {
        self.calls.read().unwrap().clone()
    }

    fn record(&self, call: impl Into<String>) {
        self.calls.write().unwrap().push(call.into());
    }
}

#[async_trait]
impl WorkspaceGit for RecordingGit {
    async fn is_repository(&self) -> Result<bool> {
        self.record("is_repository");
        Ok(true)
    }

    async fn status(&self) -> Result<WorkspaceGitStatus> {
        self.record("status");
        Ok(WorkspaceGitStatus {
            branch: "remote-main".to_string(),
            commit: "abcdef1234567890 init".to_string(),
            is_worktree: false,
            is_dirty: true,
            dirty_count: 2,
        })
    }

    async fn log(&self, max_count: usize) -> Result<Vec<WorkspaceGitCommit>> {
        self.record(format!("log:{max_count}"));
        Ok(vec![WorkspaceGitCommit {
            id: "abcdef1234567890".to_string(),
            message: "Initial commit".to_string(),
            author: "A3S".to_string(),
            date: "2026-05-18 10:00".to_string(),
        }])
    }

    async fn list_branches(&self) -> Result<Vec<WorkspaceGitBranch>> {
        self.record("list_branches");
        Ok(vec![
            WorkspaceGitBranch {
                name: "remote-main".to_string(),
                is_current: true,
            },
            WorkspaceGitBranch {
                name: "feature-x".to_string(),
                is_current: false,
            },
        ])
    }

    async fn create_branch(&self, request: WorkspaceGitCreateBranchRequest) -> Result<()> {
        self.record(format!("create_branch:{}:{}", request.name, request.base));
        Ok(())
    }

    async fn checkout(
        &self,
        request: WorkspaceGitCheckoutRequest,
    ) -> Result<WorkspaceGitCheckoutOutput> {
        self.record(format!("checkout:{}:{}", request.refspec, request.force));
        Ok(WorkspaceGitCheckoutOutput {
            stdout: "Switched remotely\n".to_string(),
        })
    }

    async fn diff(&self, request: WorkspaceGitDiffRequest) -> Result<String> {
        self.record(format!(
            "diff:{}",
            request
                .target
                .unwrap_or_else(|| "<working-tree>".to_string())
        ));
        Ok("diff --git a/src/main.rs b/src/main.rs\n".to_string())
    }

    async fn list_remotes(&self) -> Result<Vec<WorkspaceGitRemote>> {
        self.record("list_remotes");
        Ok(vec![
            WorkspaceGitRemote {
                name: "origin".to_string(),
                url: "ssh://example/repo.git".to_string(),
                direction: "fetch".to_string(),
            },
            WorkspaceGitRemote {
                name: "origin".to_string(),
                url: "ssh://example/repo.git".to_string(),
                direction: "push".to_string(),
            },
        ])
    }
}

#[async_trait]
impl WorkspaceGitStashProvider for RecordingGit {
    async fn list_stashes(&self) -> Result<Vec<WorkspaceGitStash>> {
        self.record("list_stashes");
        Ok(vec![WorkspaceGitStash {
            index: 0,
            message: "WIP remote".to_string(),
        }])
    }

    async fn stash(&self, request: WorkspaceGitStashRequest) -> Result<()> {
        self.record(format!(
            "stash:{}:{}",
            request.message.unwrap_or_default(),
            request.include_untracked
        ));
        Ok(())
    }
}

#[async_trait]
impl WorkspaceGitWorktreeProvider for RecordingGit {
    async fn list_worktrees(&self) -> Result<Vec<WorkspaceGitWorktree>> {
        self.record("list_worktrees");
        Ok(vec![WorkspaceGitWorktree {
            path: "dfs://workspace".to_string(),
            branch: "remote-main".to_string(),
            is_bare: false,
            is_detached: false,
        }])
    }

    async fn create_worktree(
        &self,
        request: WorkspaceGitCreateWorktreeRequest,
    ) -> Result<WorkspaceGitWorktreeMutation> {
        let path = request
            .path
            .unwrap_or_else(|| format!("dfs://workspace-{}", request.branch));
        self.record(format!(
            "create_worktree:{}:{}:{}",
            request.branch, path, request.new_branch
        ));
        Ok(WorkspaceGitWorktreeMutation {
            path,
            branch: Some(request.branch),
        })
    }

    async fn remove_worktree(
        &self,
        request: WorkspaceGitRemoveWorktreeRequest,
    ) -> Result<WorkspaceGitWorktreeMutation> {
        self.record(format!(
            "remove_worktree:{}:{}",
            request.path, request.force
        ));
        Ok(WorkspaceGitWorktreeMutation {
            path: request.path,
            branch: None,
        })
    }
}

fn virtual_services(fs: Arc<MemoryWorkspace>) -> Arc<WorkspaceServices> {
    let fs_backend: Arc<dyn WorkspaceFileSystem> = fs;
    WorkspaceServices::builder(
        WorkspaceRef::new("browser-workspace", "browser://workspace"),
        fs_backend,
    )
    .build()
}

fn virtual_services_with_runner(
    fs: Arc<MemoryWorkspace>,
    runner: Arc<RecordingRunner>,
) -> Arc<WorkspaceServices> {
    let fs_backend: Arc<dyn WorkspaceFileSystem> = fs;
    let runner_backend: Arc<dyn WorkspaceCommandRunner> = runner;
    WorkspaceServices::builder(
        WorkspaceRef::new("dfs-workspace", "dfs://workspace"),
        fs_backend,
    )
    .command_runner(runner_backend)
    .build()
}

fn virtual_services_with_git(
    fs: Arc<MemoryWorkspace>,
    git: Arc<RecordingGit>,
) -> Arc<WorkspaceServices> {
    let fs_backend: Arc<dyn WorkspaceFileSystem> = fs;
    let git_backend: Arc<dyn WorkspaceGit> = git.clone();
    let stash_backend: Arc<dyn WorkspaceGitStashProvider> = git.clone();
    let worktree_backend: Arc<dyn WorkspaceGitWorktreeProvider> = git;
    WorkspaceServices::builder(
        WorkspaceRef::new("git-workspace", "dfs://workspace"),
        fs_backend,
    )
    .git(git_backend)
    .git_stash(stash_backend)
    .git_worktree(worktree_backend)
    .build()
}

fn virtual_services_with_search(fs: Arc<MemoryWorkspace>) -> Arc<WorkspaceServices> {
    let fs_backend: Arc<dyn WorkspaceFileSystem> = fs.clone();
    let search_backend: Arc<dyn WorkspaceSearch> = fs;
    WorkspaceServices::builder(
        WorkspaceRef::new("search-workspace", "search://workspace"),
        fs_backend,
    )
    .search(search_backend)
    .build()
}

#[tokio::test]
async fn custom_workspace_file_tools_do_not_touch_local_filesystem() {
    let local_placeholder = tempfile::tempdir().expect("local placeholder");
    let fs = Arc::new(MemoryWorkspace::default());
    fs.insert("src/main.rs", "fn main() {}\n");

    let executor = ToolExecutor::new_with_workspace_services(
        local_placeholder.path().to_string_lossy().to_string(),
        virtual_services(Arc::clone(&fs)),
    );

    let definitions = executor.definitions();
    assert!(definitions.iter().any(|tool| tool.name == "read"));
    assert!(definitions.iter().any(|tool| tool.name == "write"));
    assert!(definitions.iter().any(|tool| tool.name == "ls"));
    assert!(definitions.iter().any(|tool| tool.name == "edit"));
    assert!(definitions.iter().any(|tool| tool.name == "patch"));
    assert!(!definitions.iter().any(|tool| tool.name == "bash"));
    assert!(!definitions.iter().any(|tool| tool.name == "grep"));
    assert!(!definitions.iter().any(|tool| tool.name == "glob"));
    assert!(!definitions.iter().any(|tool| tool.name == "git"));

    let read = executor
        .execute("read", &json!({ "file_path": "src/main.rs" }))
        .await
        .expect("read tool");
    assert_eq!(read.exit_code, 0, "{}", read.output);
    assert!(read.output.contains("fn main"));

    let write = executor
        .execute(
            "write",
            &json!({ "file_path": "src/generated.rs", "content": "pub const VALUE: u8 = 7;\n" }),
        )
        .await
        .expect("write tool");
    assert_eq!(write.exit_code, 0, "{}", write.output);
    assert_eq!(
        fs.read_raw("src/generated.rs").as_deref(),
        Some("pub const VALUE: u8 = 7;\n")
    );
    assert!(!local_placeholder.path().join("src/generated.rs").exists());

    let edit = executor
        .execute(
            "edit",
            &json!({
                "file_path": "src/generated.rs",
                "old_string": "VALUE: u8 = 7",
                "new_string": "VALUE: u8 = 8"
            }),
        )
        .await
        .expect("edit tool");
    assert_eq!(edit.exit_code, 0, "{}", edit.output);
    assert_eq!(
        fs.read_raw("src/generated.rs").as_deref(),
        Some("pub const VALUE: u8 = 8;\n")
    );

    let patch = executor
        .execute(
            "patch",
            &json!({
                "file_path": "src/generated.rs",
                "diff": "@@ -1,1 +1,1 @@\n-pub const VALUE: u8 = 8;\n+pub const VALUE: u8 = 9;"
            }),
        )
        .await
        .expect("patch tool");
    assert_eq!(patch.exit_code, 0, "{}", patch.output);
    assert_eq!(
        fs.read_raw("src/generated.rs").as_deref(),
        Some("pub const VALUE: u8 = 9;\n")
    );

    let listing = executor
        .execute("ls", &json!({ "path": "src" }))
        .await
        .expect("ls tool");
    assert_eq!(listing.exit_code, 0, "{}", listing.output);
    assert!(listing.output.contains("main.rs"));
    assert!(listing.output.contains("generated.rs"));
}

#[tokio::test]
async fn custom_workspace_path_resolver_blocks_escape_before_backend_access() {
    let fs = Arc::new(MemoryWorkspace::default());
    fs.insert("safe.txt", "ok\n");
    let executor = ToolExecutor::new_with_workspace_services_and_artifact_limits(
        "/server/local-placeholder".to_string(),
        virtual_services(fs),
        ArtifactStoreLimits::default(),
    );

    let read = executor
        .execute("read", &json!({ "file_path": "../secret.txt" }))
        .await
        .expect("read tool");
    assert_eq!(read.exit_code, 1);
    assert!(read.output.contains("escapes workspace"));

    let write = executor
        .execute(
            "write",
            &json!({ "file_path": "/tmp/secret.txt", "content": "nope" }),
        )
        .await
        .expect("write tool");
    assert_eq!(write.exit_code, 1);
    assert!(write.output.contains("Absolute paths"));
}

#[tokio::test]
async fn custom_workspace_runner_drives_bash_tool() {
    let fs = Arc::new(MemoryWorkspace::default());
    let runner = Arc::new(RecordingRunner::default());
    let executor = ToolExecutor::new_with_workspace_services_and_artifact_limits(
        "/server/local-placeholder".to_string(),
        virtual_services_with_runner(fs, Arc::clone(&runner)),
        ArtifactStoreLimits::default(),
    );

    let definitions = executor.definitions();
    assert!(definitions.iter().any(|tool| tool.name == "bash"));

    let result = executor
        .execute("bash", &json!({ "command": "pwd && ls", "timeout": 1234 }))
        .await
        .expect("bash tool");
    assert_eq!(result.exit_code, 0, "{}", result.output);
    assert_eq!(result.output, "remote runner executed: pwd && ls\n");
    assert_eq!(runner.calls(), vec![("pwd && ls".to_string(), 1234)]);
}

#[tokio::test]
async fn custom_workspace_search_provider_drives_grep_and_glob_tools() {
    let fs = Arc::new(MemoryWorkspace::default());
    fs.insert("src/main.rs", "fn main() {\n    println!(\"Hello\");\n}\n");
    fs.insert("src/lib.rs", "pub fn helper() {}\n");
    fs.insert("README.md", "hello from docs\n");
    let executor = ToolExecutor::new_with_workspace_services_and_artifact_limits(
        "/server/local-placeholder".to_string(),
        virtual_services_with_search(Arc::clone(&fs)),
        ArtifactStoreLimits::default(),
    );

    let definitions = executor.definitions();
    assert!(definitions.iter().any(|tool| tool.name == "grep"));
    assert!(definitions.iter().any(|tool| tool.name == "glob"));
    assert!(!definitions.iter().any(|tool| tool.name == "git"));

    let glob = executor
        .execute("glob", &json!({ "pattern": "*.rs", "path": "src" }))
        .await
        .expect("glob tool");
    assert_eq!(glob.exit_code, 0, "{}", glob.output);
    assert!(glob.output.contains("src/main.rs"));
    assert!(glob.output.contains("src/lib.rs"));
    assert!(!glob.output.contains("README.md"));

    let grep = executor
        .execute(
            "grep",
            &json!({ "pattern": "hello", "path": ".", "glob": "**/*.rs", "-i": true }),
        )
        .await
        .expect("grep tool");
    assert_eq!(grep.exit_code, 0, "{}", grep.output);
    assert!(grep.output.contains("src/main.rs:2"));
    assert!(grep.output.contains("1 match(es) in 1 file(s)"));
    assert!(!grep.output.contains("README.md"));
}

#[tokio::test]
async fn custom_workspace_git_provider_drives_git_tool() {
    let local_placeholder = tempfile::tempdir().expect("local placeholder");
    let fs = Arc::new(MemoryWorkspace::default());
    let git = Arc::new(RecordingGit::default());
    let executor = ToolExecutor::new_with_workspace_services_and_artifact_limits(
        local_placeholder.path().to_string_lossy().to_string(),
        virtual_services_with_git(fs, Arc::clone(&git)),
        ArtifactStoreLimits::default(),
    );

    let definitions = executor.definitions();
    assert!(definitions.iter().any(|tool| tool.name == "git"));
    assert!(!definitions.iter().any(|tool| tool.name == "bash"));

    let status = executor
        .execute("git", &json!({ "command": "status" }))
        .await
        .expect("git status");
    assert_eq!(status.exit_code, 0, "{}", status.output);
    assert!(status.output.contains("Workspace: dfs://workspace"));
    assert!(status.output.contains("Branch:    remote-main"));
    assert!(status.output.contains("2 uncommitted change(s)"));

    let branch = executor
        .execute(
            "git",
            &json!({ "command": "branch", "name": "feature-y", "base": "remote-main" }),
        )
        .await
        .expect("git branch");
    assert_eq!(branch.exit_code, 0, "{}", branch.output);
    assert!(branch.output.contains("Created branch: feature-y"));

    let diff = executor
        .execute("git", &json!({ "command": "diff", "target": "HEAD~1" }))
        .await
        .expect("git diff");
    assert_eq!(diff.exit_code, 0, "{}", diff.output);
    assert!(diff.output.contains("diff --git"));

    let remote = executor
        .execute("git", &json!({ "command": "remote" }))
        .await
        .expect("git remote");
    assert_eq!(remote.exit_code, 0, "{}", remote.output);
    assert!(remote
        .output
        .contains("origin\tssh://example/repo.git (fetch)"));

    let worktree = executor
        .execute(
            "git",
            &json!({
                "command": "worktree",
                "subcommand": "create",
                "name": "feature-y",
                "path": "branches/feature-y",
                "new_branch": false
            }),
        )
        .await
        .expect("git worktree create");
    assert_eq!(worktree.exit_code, 0, "{}", worktree.output);
    assert!(worktree
        .output
        .contains("Created worktree at: branches/feature-y"));

    assert_eq!(
        git.calls(),
        vec![
            "is_repository",
            "status",
            "is_repository",
            "create_branch:feature-y:remote-main",
            "is_repository",
            "diff:HEAD~1",
            "is_repository",
            "list_remotes",
            "is_repository",
            "create_worktree:feature-y:branches/feature-y:false",
        ]
    );
}
