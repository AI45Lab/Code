//! Workspace capability abstractions.
//!
//! Built-in tools expose stable model-facing contracts (`read`, `write`, `ls`,
//! `bash`, ...). The concrete place where those operations happen is supplied
//! by a workspace capability backend. The default backend is the local
//! filesystem (see [`LocalWorkspaceBackend`]); hosts can provide remote,
//! browser, DFS, or container-backed implementations by assembling
//! [`WorkspaceServices`] through [`WorkspaceServicesBuilder`].

mod local;

pub use local::LocalWorkspaceBackend;

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

/// Identity and display metadata for a workspace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceRef {
    /// Stable workspace identifier used by host backends.
    pub id: String,
    /// Human-readable root shown in tool output.
    pub display_root: String,
}

impl WorkspaceRef {
    pub fn new(id: impl Into<String>, display_root: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            display_root: display_root.into(),
        }
    }
}

/// A normalized virtual path inside a workspace.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkspacePath {
    inner: String,
}

impl WorkspacePath {
    pub fn root() -> Self {
        Self {
            inner: ".".to_string(),
        }
    }

    pub fn from_normalized(path: impl Into<String>) -> Self {
        let path = path.into();
        let path = path.trim_matches('/');
        if path.is_empty() || path == "." {
            Self::root()
        } else {
            Self {
                inner: path.replace('\\', "/"),
            }
        }
    }

    pub fn as_str(&self) -> &str {
        &self.inner
    }

    pub fn is_root(&self) -> bool {
        self.inner == "."
    }
}

/// Workspace capability flags used to gate which built-in tools are registered.
///
/// Each flag corresponds to a provider trait on [`WorkspaceServices`]; flags
/// without a backing provider are deliberately omitted so the surface stays
/// minimal until a real consumer appears.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceCapabilities {
    pub read: bool,
    pub write: bool,
    pub exec: bool,
    pub search: bool,
    pub git: bool,
}

impl WorkspaceCapabilities {
    pub fn local_default() -> Self {
        Self {
            read: true,
            write: true,
            exec: true,
            search: true,
            git: true,
        }
    }

    pub fn read_write() -> Self {
        Self {
            read: true,
            write: true,
            exec: false,
            search: false,
            git: false,
        }
    }
}

impl Default for WorkspaceCapabilities {
    fn default() -> Self {
        Self::read_write()
    }
}

/// Directory entry kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceFileType {
    File,
    Directory,
    Symlink,
    Unknown,
}

impl WorkspaceFileType {
    pub fn as_tool_kind(self) -> &'static str {
        match self {
            Self::File => "file",
            Self::Directory => "dir",
            Self::Symlink => "link",
            Self::Unknown => "unknown",
        }
    }
}

/// Directory entry returned by a workspace backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceDirEntry {
    pub name: String,
    pub kind: WorkspaceFileType,
    pub size: u64,
}

/// Result metadata for a write operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceWriteOutcome {
    pub bytes: usize,
    pub lines: usize,
}

/// Glob request for workspace-backed search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceGlobRequest {
    pub base: WorkspacePath,
    pub pattern: String,
}

/// Glob result returned by a workspace search provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceGlobResult {
    pub matches: Vec<WorkspacePath>,
}

/// Grep request for workspace-backed search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceGrepRequest {
    pub base: WorkspacePath,
    pub pattern: String,
    pub glob: Option<String>,
    pub context_lines: usize,
    pub case_insensitive: bool,
    pub max_output_size: usize,
}

/// Grep result returned by a workspace search provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceGrepResult {
    pub output: String,
    pub match_count: usize,
    pub file_count: usize,
    pub truncated: bool,
}

/// Repository status returned by a workspace Git provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceGitStatus {
    pub branch: String,
    pub commit: String,
    pub is_worktree: bool,
    pub is_dirty: bool,
    pub dirty_count: usize,
}

/// Commit information returned by a workspace Git provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceGitCommit {
    pub id: String,
    pub message: String,
    pub author: String,
    pub date: String,
}

/// Branch information returned by a workspace Git provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceGitBranch {
    pub name: String,
    pub is_current: bool,
}

/// Branch creation request for a workspace Git provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceGitCreateBranchRequest {
    pub name: String,
    pub base: String,
}

/// Checkout request for a workspace Git provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceGitCheckoutRequest {
    pub refspec: String,
    pub force: bool,
}

/// Checkout output returned by a workspace Git provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceGitCheckoutOutput {
    pub stdout: String,
}

/// Diff request for a workspace Git provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceGitDiffRequest {
    pub target: Option<String>,
}

/// Stash information returned by a workspace Git provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceGitStash {
    pub index: usize,
    pub message: String,
}

/// Stash request for a workspace Git provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceGitStashRequest {
    pub message: Option<String>,
    pub include_untracked: bool,
}

/// Remote information returned by a workspace Git provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceGitRemote {
    pub name: String,
    pub url: String,
    pub direction: String,
}

/// Worktree information returned by a workspace Git provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceGitWorktree {
    pub path: String,
    pub branch: String,
    pub is_bare: bool,
    pub is_detached: bool,
}

/// Worktree creation request for a workspace Git provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceGitCreateWorktreeRequest {
    pub branch: String,
    pub path: Option<String>,
    pub new_branch: bool,
}

/// Worktree removal request for a workspace Git provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceGitRemoveWorktreeRequest {
    pub path: String,
    pub force: bool,
}

/// Mutation result for workspace Git worktree operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceGitWorktreeMutation {
    pub path: String,
    pub branch: Option<String>,
}

/// Observer that receives streaming output deltas from a workspace command.
///
/// Backend implementations call this on each chunk of stdout/stderr they
/// observe. Tool layers wire host event channels behind this trait, so the
/// workspace abstraction does not depend on any tool event type.
#[async_trait]
pub trait CommandOutputObserver: Send + Sync {
    async fn on_output_delta(&self, delta: &str);
}

/// Command execution request.
#[derive(Clone)]
pub struct CommandRequest {
    pub command: String,
    pub timeout_ms: u64,
    pub output_observer: Option<Arc<dyn CommandOutputObserver>>,
    pub env: Option<Arc<HashMap<String, String>>>,
}

impl std::fmt::Debug for CommandRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandRequest")
            .field("command", &self.command)
            .field("timeout_ms", &self.timeout_ms)
            .field("output_observer", &self.output_observer.is_some())
            .field("env", &self.env.as_ref().map(|env| env.len()))
            .finish()
    }
}

/// Command execution output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandOutput {
    pub output: String,
    pub exit_code: i32,
    pub timed_out: bool,
}

/// Normalizes and validates host-supplied paths before they reach a backend.
pub trait WorkspacePathResolver: Send + Sync {
    fn normalize(&self, input: &str) -> Result<WorkspacePath>;
}

/// File operations available to built-in file tools.
///
/// **Trait stability policy:** new methods added to this trait are a breaking
/// change for every external backend implementation. Until the workspace
/// extension story is stabilised, new methods will be added to a separate
/// `WorkspaceFileSystemExt` trait (with default implementations that fall back
/// to the core methods) rather than to this trait directly. Backend authors
/// can rely on this trait surface remaining additive only through extension
/// traits.
#[async_trait]
pub trait WorkspaceFileSystem: Send + Sync {
    async fn read_text(&self, path: &WorkspacePath) -> Result<String>;
    async fn write_text(
        &self,
        path: &WorkspacePath,
        content: &str,
    ) -> Result<WorkspaceWriteOutcome>;
    async fn list_dir(&self, path: &WorkspacePath) -> Result<Vec<WorkspaceDirEntry>>;
}

/// Shell/command execution available to the `bash` tool.
#[async_trait]
pub trait WorkspaceCommandRunner: Send + Sync {
    async fn exec(&self, request: CommandRequest) -> Result<CommandOutput>;
}

/// Search operations available to `glob` and `grep`.
#[async_trait]
pub trait WorkspaceSearch: Send + Sync {
    async fn glob(&self, request: WorkspaceGlobRequest) -> Result<WorkspaceGlobResult>;
    async fn grep(&self, request: WorkspaceGrepRequest) -> Result<WorkspaceGrepResult>;
}

/// Core Git operations supported by virtually every workspace Git backend.
///
/// Optional features (stash, worktrees) live in separate traits so backends
/// like browser-side `isomorphic-git` can implement only what they support
/// instead of returning runtime "unsupported" errors.
#[async_trait]
pub trait WorkspaceGit: Send + Sync {
    async fn is_repository(&self) -> Result<bool>;
    async fn status(&self) -> Result<WorkspaceGitStatus>;
    async fn log(&self, max_count: usize) -> Result<Vec<WorkspaceGitCommit>>;
    async fn list_branches(&self) -> Result<Vec<WorkspaceGitBranch>>;
    async fn create_branch(&self, request: WorkspaceGitCreateBranchRequest) -> Result<()>;
    async fn checkout(
        &self,
        request: WorkspaceGitCheckoutRequest,
    ) -> Result<WorkspaceGitCheckoutOutput>;
    async fn diff(&self, request: WorkspaceGitDiffRequest) -> Result<String>;
    async fn list_remotes(&self) -> Result<Vec<WorkspaceGitRemote>>;
}

/// Optional Git stash operations.
///
/// Browser-side libraries such as `isomorphic-git` do not implement stash;
/// backends that cannot stash simply do not implement this trait.
#[async_trait]
pub trait WorkspaceGitStashProvider: Send + Sync {
    async fn list_stashes(&self) -> Result<Vec<WorkspaceGitStash>>;
    async fn stash(&self, request: WorkspaceGitStashRequest) -> Result<()>;
}

/// Optional Git worktree operations.
///
/// Worktrees are a local-filesystem concept and are typically not supported
/// by remote or browser-backed git providers.
#[async_trait]
pub trait WorkspaceGitWorktreeProvider: Send + Sync {
    async fn list_worktrees(&self) -> Result<Vec<WorkspaceGitWorktree>>;
    async fn create_worktree(
        &self,
        request: WorkspaceGitCreateWorktreeRequest,
    ) -> Result<WorkspaceGitWorktreeMutation>;
    async fn remove_worktree(
        &self,
        request: WorkspaceGitRemoveWorktreeRequest,
    ) -> Result<WorkspaceGitWorktreeMutation>;
}

/// The host-provided workspace capability bundle used by tool execution.
pub struct WorkspaceServices {
    workspace_ref: WorkspaceRef,
    capabilities: WorkspaceCapabilities,
    path_resolver: Arc<dyn WorkspacePathResolver>,
    file_system: Arc<dyn WorkspaceFileSystem>,
    command_runner: Option<Arc<dyn WorkspaceCommandRunner>>,
    search: Option<Arc<dyn WorkspaceSearch>>,
    git: Option<Arc<dyn WorkspaceGit>>,
    git_stash: Option<Arc<dyn WorkspaceGitStashProvider>>,
    git_worktree: Option<Arc<dyn WorkspaceGitWorktreeProvider>>,
    /// Default timeout applied to non-bash workspace operations (file system,
    /// search, git). Bash uses its own per-call timeout in [`CommandRequest`].
    /// `None` means no enforced timeout — appropriate for the local backend.
    operation_timeout: Option<std::time::Duration>,
    local_root: Option<PathBuf>,
}

impl std::fmt::Debug for WorkspaceServices {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkspaceServices")
            .field("workspace_ref", &self.workspace_ref)
            .field("capabilities", &self.capabilities)
            .field("command_runner", &self.command_runner.is_some())
            .field("search", &self.search.is_some())
            .field("git", &self.git.is_some())
            .field("git_stash", &self.git_stash.is_some())
            .field("git_worktree", &self.git_worktree.is_some())
            .field("local_root", &self.local_root)
            .finish()
    }
}

impl WorkspaceServices {
    pub(crate) fn new_with_git(
        workspace_ref: WorkspaceRef,
        mut capabilities: WorkspaceCapabilities,
        path_resolver: Arc<dyn WorkspacePathResolver>,
        file_system: Arc<dyn WorkspaceFileSystem>,
        command_runner: Option<Arc<dyn WorkspaceCommandRunner>>,
        search: Option<Arc<dyn WorkspaceSearch>>,
        git: Option<Arc<dyn WorkspaceGit>>,
    ) -> Self {
        if command_runner.is_none() {
            capabilities.exec = false;
        }
        if search.is_none() {
            capabilities.search = false;
        }
        if git.is_none() {
            capabilities.git = false;
        }
        Self {
            workspace_ref,
            capabilities,
            path_resolver,
            file_system,
            command_runner,
            search,
            git,
            git_stash: None,
            git_worktree: None,
            operation_timeout: None,
            local_root: None,
        }
    }

    pub fn builder(
        workspace_ref: WorkspaceRef,
        file_system: Arc<dyn WorkspaceFileSystem>,
    ) -> WorkspaceServicesBuilder {
        WorkspaceServicesBuilder::new(workspace_ref, file_system)
    }

    pub fn local(root: impl Into<PathBuf>) -> Arc<Self> {
        let backend = Arc::new(LocalWorkspaceBackend::new(root.into()));
        let workspace_ref = WorkspaceRef::new(
            backend.root.display().to_string(),
            backend.root.display().to_string(),
        );
        let path_resolver: Arc<dyn WorkspacePathResolver> = backend.clone();
        let file_system: Arc<dyn WorkspaceFileSystem> = backend.clone();
        let command_runner: Arc<dyn WorkspaceCommandRunner> = backend.clone();
        let search: Arc<dyn WorkspaceSearch> = backend.clone();
        let git: Arc<dyn WorkspaceGit> = backend.clone();
        let git_stash: Arc<dyn WorkspaceGitStashProvider> = backend.clone();
        let git_worktree: Arc<dyn WorkspaceGitWorktreeProvider> = backend.clone();
        Arc::new(Self {
            workspace_ref,
            capabilities: WorkspaceCapabilities::local_default(),
            path_resolver,
            file_system,
            command_runner: Some(command_runner),
            search: Some(search),
            git: Some(git),
            git_stash: Some(git_stash),
            git_worktree: Some(git_worktree),
            operation_timeout: None,
            local_root: Some(backend.root.clone()),
        })
    }

    pub fn workspace_ref(&self) -> &WorkspaceRef {
        &self.workspace_ref
    }

    pub fn capabilities(&self) -> WorkspaceCapabilities {
        self.capabilities
    }

    pub fn normalize_path(&self, input: &str) -> Result<WorkspacePath> {
        self.path_resolver.normalize(input)
    }

    pub fn fs(&self) -> Arc<dyn WorkspaceFileSystem> {
        Arc::clone(&self.file_system)
    }

    pub fn command_runner(&self) -> Option<Arc<dyn WorkspaceCommandRunner>> {
        self.command_runner.clone()
    }

    pub fn search(&self) -> Option<Arc<dyn WorkspaceSearch>> {
        self.search.clone()
    }

    pub fn git(&self) -> Option<Arc<dyn WorkspaceGit>> {
        self.git.clone()
    }

    pub fn git_stash(&self) -> Option<Arc<dyn WorkspaceGitStashProvider>> {
        self.git_stash.clone()
    }

    pub fn git_worktree(&self) -> Option<Arc<dyn WorkspaceGitWorktreeProvider>> {
        self.git_worktree.clone()
    }

    /// Default timeout applied to non-bash workspace operations.
    ///
    /// `None` means no enforced timeout. Backends that may stall (remote,
    /// browser, DFS) should set this so tools using [`Self::run_with_timeout`]
    /// surface a timeout error instead of letting the agent loop hang.
    pub fn operation_timeout(&self) -> Option<std::time::Duration> {
        self.operation_timeout
    }

    /// Run a workspace future under the configured operation timeout.
    ///
    /// Tools that route through file system / search / git providers should
    /// wrap their calls with this helper so non-local backends never stall
    /// the agent loop indefinitely.
    pub async fn run_with_timeout<F, T>(&self, op: &'static str, fut: F) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>>,
    {
        match self.operation_timeout {
            Some(d) => tokio::time::timeout(d, fut)
                .await
                .map_err(|_| anyhow!("workspace operation '{}' timed out after {:?}", op, d))?,
            None => fut.await,
        }
    }

    pub fn local_root(&self) -> Option<&Path> {
        self.local_root.as_deref()
    }

    pub fn display_path(&self, path: &WorkspacePath) -> String {
        if path.is_root() {
            return self.workspace_ref.display_root.clone();
        }

        let root = self.workspace_ref.display_root.trim_end_matches('/');
        if root.is_empty() {
            path.as_str().to_string()
        } else {
            format!("{root}/{}", path.as_str())
        }
    }
}

/// Builder for assembling workspace services without constructor arity churn.
pub struct WorkspaceServicesBuilder {
    workspace_ref: WorkspaceRef,
    capabilities: WorkspaceCapabilities,
    path_resolver: Arc<dyn WorkspacePathResolver>,
    file_system: Arc<dyn WorkspaceFileSystem>,
    command_runner: Option<Arc<dyn WorkspaceCommandRunner>>,
    search: Option<Arc<dyn WorkspaceSearch>>,
    git: Option<Arc<dyn WorkspaceGit>>,
    git_stash: Option<Arc<dyn WorkspaceGitStashProvider>>,
    git_worktree: Option<Arc<dyn WorkspaceGitWorktreeProvider>>,
    operation_timeout: Option<std::time::Duration>,
}

impl WorkspaceServicesBuilder {
    pub fn new(workspace_ref: WorkspaceRef, file_system: Arc<dyn WorkspaceFileSystem>) -> Self {
        Self {
            workspace_ref,
            capabilities: WorkspaceCapabilities::read_write(),
            path_resolver: Arc::new(VirtualPathResolver),
            file_system,
            command_runner: None,
            search: None,
            git: None,
            git_stash: None,
            git_worktree: None,
            operation_timeout: None,
        }
    }

    pub fn capabilities(mut self, capabilities: WorkspaceCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    pub fn command_runner(mut self, command_runner: Arc<dyn WorkspaceCommandRunner>) -> Self {
        self.capabilities.exec = true;
        self.command_runner = Some(command_runner);
        self
    }

    pub fn search(mut self, search: Arc<dyn WorkspaceSearch>) -> Self {
        self.capabilities.search = true;
        self.search = Some(search);
        self
    }

    pub fn git(mut self, git: Arc<dyn WorkspaceGit>) -> Self {
        self.capabilities.git = true;
        self.git = Some(git);
        self
    }

    pub fn git_stash(mut self, git_stash: Arc<dyn WorkspaceGitStashProvider>) -> Self {
        self.git_stash = Some(git_stash);
        self
    }

    pub fn git_worktree(mut self, git_worktree: Arc<dyn WorkspaceGitWorktreeProvider>) -> Self {
        self.git_worktree = Some(git_worktree);
        self
    }

    /// Apply a default timeout to non-bash workspace operations (file system,
    /// search, git). Backends that may stall — remote, browser, DFS — should
    /// set this so tools surface a timeout error rather than hanging.
    pub fn operation_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.operation_timeout = Some(timeout);
        self
    }

    pub fn build(self) -> Arc<WorkspaceServices> {
        let mut services = WorkspaceServices::new_with_git(
            self.workspace_ref,
            self.capabilities,
            self.path_resolver,
            self.file_system,
            self.command_runner,
            self.search,
            self.git,
        );
        services.git_stash = self.git_stash;
        services.git_worktree = self.git_worktree;
        services.operation_timeout = self.operation_timeout;
        Arc::new(services)
    }
}

/// Lexical resolver suitable for virtual/browser/DFS workspaces.
#[derive(Debug, Default)]
pub struct VirtualPathResolver;

impl WorkspacePathResolver for VirtualPathResolver {
    fn normalize(&self, input: &str) -> Result<WorkspacePath> {
        normalize_virtual_path(input)
    }
}

fn normalize_virtual_path(input: &str) -> Result<WorkspacePath> {
    let input = default_path_input(input);
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

fn default_path_input(input: &str) -> &str {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        "."
    } else {
        trimmed
    }
}

fn has_windows_path_prefix(input: &str) -> bool {
    let bytes = input.as_bytes();
    if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        return true;
    }

    input.starts_with("\\\\") || input.starts_with("//")
}

fn validate_relative_pattern(pattern: &str, label: &str) -> Result<()> {
    let pattern = pattern.trim();
    if pattern.is_empty() {
        bail!("{label} cannot be empty");
    }
    if has_windows_path_prefix(pattern) || Path::new(pattern).is_absolute() {
        bail!("{label} must be relative to the workspace");
    }

    let normalized = pattern.replace('\\', "/");
    if normalized.split('/').any(|component| component == "..") {
        bail!("{label} must not contain parent directory traversal");
    }

    Ok(())
}

fn normalize_relative_path(path: &Path) -> Result<PathBuf> {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => out.push(part),
            Component::ParentDir => {
                if !out.pop() {
                    bail!("Workspace boundary violation: path escapes workspace");
                }
            }
            Component::RootDir | Component::Prefix(_) => {
                bail!("Absolute paths are not supported by this workspace backend");
            }
        }
    }
    Ok(out)
}

fn pathbuf_to_workspace_path(path: &Path) -> WorkspacePath {
    let display = path.to_string_lossy().replace('\\', "/");
    WorkspacePath::from_normalized(display)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn virtual_resolver_normalizes_relative_paths() {
        let resolver = VirtualPathResolver;
        let path = resolver.normalize("./src/../README.md").unwrap();
        assert_eq!(path.as_str(), "README.md");
    }

    #[test]
    fn virtual_resolver_normalizes_backslash_separators() {
        let resolver = VirtualPathResolver;
        let path = resolver.normalize(r"src\main.rs").unwrap();
        assert_eq!(path.as_str(), "src/main.rs");
    }

    #[test]
    fn virtual_resolver_rejects_escape() {
        let resolver = VirtualPathResolver;
        let err = resolver.normalize("../secret.txt").unwrap_err();
        assert!(err.to_string().contains("escapes workspace"));
    }

    #[test]
    fn virtual_resolver_rejects_backslash_escape() {
        let resolver = VirtualPathResolver;
        let err = resolver.normalize(r"..\secret.txt").unwrap_err();
        assert!(err.to_string().contains("escapes workspace"));
    }

    #[test]
    fn virtual_resolver_rejects_absolute_paths() {
        let resolver = VirtualPathResolver;
        let err = resolver.normalize("/tmp/secret.txt").unwrap_err();
        assert!(err.to_string().contains("Absolute paths"));
    }

    #[test]
    fn virtual_resolver_rejects_windows_absolute_paths() {
        let resolver = VirtualPathResolver;

        let drive_err = resolver.normalize(r"C:\Users\secret.txt").unwrap_err();
        assert!(drive_err.to_string().contains("Absolute paths"));

        let unc_err = resolver
            .normalize(r"\\server\share\secret.txt")
            .unwrap_err();
        assert!(unc_err.to_string().contains("Absolute paths"));
    }

    #[test]
    fn workspace_services_disable_exec_without_runner() {
        struct EmptyFs;

        #[async_trait]
        impl WorkspaceFileSystem for EmptyFs {
            async fn read_text(&self, _path: &WorkspacePath) -> Result<String> {
                bail!("not implemented")
            }

            async fn write_text(
                &self,
                _path: &WorkspacePath,
                _content: &str,
            ) -> Result<WorkspaceWriteOutcome> {
                bail!("not implemented")
            }

            async fn list_dir(&self, _path: &WorkspacePath) -> Result<Vec<WorkspaceDirEntry>> {
                bail!("not implemented")
            }
        }

        let fs_backend: Arc<dyn WorkspaceFileSystem> = Arc::new(EmptyFs);
        let services = WorkspaceServices::builder(
            WorkspaceRef::new("virtual", "virtual://workspace"),
            fs_backend,
        )
        .capabilities(WorkspaceCapabilities {
            exec: true,
            ..WorkspaceCapabilities::read_write()
        })
        .build();

        assert!(!services.capabilities().exec);
        assert!(services.command_runner().is_none());
    }
}
