//! Extensible Tool System
//!
//! Provides a trait-based abstraction for tools.
//!
//! ## Architecture
//!
//! ```text
//! ToolRegistry
//!   └── builtin tools (bash, read, write, edit, grep, glob, ls, patch, web_fetch, web_search)
//! ```

mod agent_dir_script_tool;
mod artifacts;
pub(crate) mod builtin;
pub(crate) mod process;
mod program_tool;
mod registry;
mod selector;
pub mod skill;
pub mod task;
mod types;

pub use agent_dir_script_tool::AgentDirScriptTool;
pub use artifacts::{ArtifactStore, ArtifactStoreLimits, ToolArtifact};
pub(crate) use builtin::register_skill;
pub use builtin::{
    register_generate_object, register_program, register_program_with_catalog, register_task,
    register_task_with_mcp,
};
pub use program_tool::ProgramTool;
pub use registry::ToolRegistry;
pub use selector::{select_tools_for_messages, select_tools_for_prompt};
pub use task::{
    parallel_task_params_schema, task_params_schema, ParallelTaskParams, ParallelTaskTool,
    TaskExecutor, TaskParams, TaskResult, TaskTool,
};
pub use types::{Tool, ToolContext, ToolErrorKind, ToolEventSender, ToolOutput, ToolStreamEvent};

use crate::file_history::{self, FileHistory};
use crate::llm::ToolDefinition;
use crate::text::truncate_utf8;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Maximum output size in bytes before truncation
pub const MAX_OUTPUT_SIZE: usize = 100 * 1024; // 100KB

/// Maximum lines to read from a file
pub const MAX_READ_LINES: usize = 2000;

/// Maximum line length before truncation
pub const MAX_LINE_LENGTH: usize = 2000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ToolOutputArtifact {
    pub artifact_id: String,
    pub artifact_uri: String,
    pub original_bytes: usize,
    pub shown_bytes: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct TruncatedToolOutput {
    pub content: String,
    pub artifact: Option<ToolOutputArtifact>,
}

pub(crate) fn truncate_tool_output_with_artifact(
    tool_name: &str,
    output: &str,
) -> TruncatedToolOutput {
    if output.len() <= MAX_OUTPUT_SIZE {
        return TruncatedToolOutput {
            content: output.to_string(),
            artifact: None,
        };
    }

    let shown = truncate_utf8(output, MAX_OUTPUT_SIZE);
    let artifact = tool_output_artifact(tool_name, output, shown.len());
    let artifact_uri = artifact.artifact_uri.clone();
    let content = format!(
        "{}\n\n[tool output truncated: showing the first {} of {} bytes. Full output artifact: {}. Use narrower arguments such as offset/limit or filtering when possible.]",
        shown,
        shown.len(),
        output.len(),
        artifact_uri,
    );

    TruncatedToolOutput {
        content,
        artifact: Some(artifact),
    }
}

pub(crate) fn tool_output_artifact(
    tool_name: &str,
    output: &str,
    shown_bytes: usize,
) -> ToolOutputArtifact {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    tool_name.hash(&mut hasher);
    output.len().hash(&mut hasher);
    output.hash(&mut hasher);
    let digest = hasher.finish();
    let sanitized_tool = tool_name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    let artifact_id = format!("tool-output:{sanitized_tool}:{digest:016x}");
    let artifact_uri = format!("a3s://tool-output/{sanitized_tool}/{digest:016x}");

    ToolOutputArtifact {
        artifact_id,
        artifact_uri,
        original_bytes: output.len(),
        shown_bytes,
    }
}

pub(crate) fn merge_tool_output_artifact_metadata(
    metadata: Option<serde_json::Value>,
    artifact: &ToolOutputArtifact,
) -> serde_json::Value {
    let artifact_json = serde_json::json!({
        "artifact_id": artifact.artifact_id,
        "artifact_uri": artifact.artifact_uri,
        "original_bytes": artifact.original_bytes,
        "shown_bytes": artifact.shown_bytes,
    });

    match metadata {
        Some(serde_json::Value::Object(mut object)) => {
            object.insert("artifact".to_string(), artifact_json);
            serde_json::Value::Object(object)
        }
        Some(value) => serde_json::json!({
            "artifact": artifact_json,
            "previous_metadata": value,
        }),
        None => serde_json::json!({
            "artifact": artifact_json,
        }),
    }
}

/// Tool execution result returned by direct tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub name: String,
    pub output: String,
    pub exit_code: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    /// Image attachments from tool execution (multi-modal output).
    #[serde(skip)]
    pub images: Vec<crate::llm::Attachment>,
    /// Structured discriminant for tool failures. Populated by built-in
    /// tools that can map their failure into a typed [`ToolErrorKind`]
    /// (e.g. `edit`/`patch` setting `VersionConflict` on a CAS rejection
    /// from `WorkspaceError`). Forwarded to the SDK so callers can react
    /// programmatically without parsing `output`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<types::ToolErrorKind>,
}

impl ToolResult {
    pub fn success(name: &str, output: String) -> Self {
        Self {
            name: name.to_string(),
            output,
            exit_code: 0,
            metadata: None,
            images: Vec::new(),
            error_kind: None,
        }
    }

    pub fn error(name: &str, message: String) -> Self {
        Self {
            name: name.to_string(),
            output: message,
            exit_code: 1,
            metadata: None,
            images: Vec::new(),
            error_kind: None,
        }
    }
}

impl From<ToolOutput> for ToolResult {
    fn from(output: ToolOutput) -> Self {
        Self {
            name: String::new(),
            output: output.content,
            exit_code: if output.success { 0 } else { 1 },
            metadata: output.metadata,
            images: output.images,
            error_kind: output.error_kind,
        }
    }
}

/// Tool executor with workspace sandboxing
///
/// This is the main entry point for tool execution. It wraps the ToolRegistry
/// and captures file snapshots before write/edit/patch operations.
pub struct ToolExecutor {
    workspace: PathBuf,
    registry: Arc<ToolRegistry>,
    file_history: Arc<FileHistory>,
    command_env: Option<Arc<HashMap<String, String>>>,
    workspace_services: Arc<crate::workspace::WorkspaceServices>,
}

/// Build a log line for a tool invocation that excludes argument *values*.
///
/// Argument values (full bash commands, file contents written by `write`/`edit`)
/// can contain secrets, so the summary records only the tool name, the sorted
/// argument field names, and the serialized payload size — never the values. This
/// keeps the always-on `info!` tool trace (also exported to OTLP) compliant with
/// the "never log secrets" boundary. Use `trace!` for full args when debugging.
fn redacted_tool_log_summary(name: &str, args: &serde_json::Value) -> String {
    let arg_keys: Vec<&str> = match args.as_object() {
        Some(map) => {
            let mut keys: Vec<&str> = map.keys().map(String::as_str).collect();
            keys.sort_unstable();
            keys
        }
        None => Vec::new(),
    };
    format!(
        "Executing tool: {} (arg_keys={:?}, {} bytes)",
        name,
        arg_keys,
        args.to_string().len()
    )
}

/// Log a tool invocation without leaking argument values. See
/// [`redacted_tool_log_summary`] for the redaction rationale.
fn log_tool_invocation(name: &str, args: &serde_json::Value) {
    tracing::info!("{}", redacted_tool_log_summary(name, args));
    tracing::trace!("Tool {} full args: {}", name, args);
}

impl ToolExecutor {
    pub fn new(workspace: String) -> Self {
        let workspace_services =
            crate::workspace::WorkspaceServices::local(PathBuf::from(&workspace));
        Self::build(
            workspace,
            None,
            ArtifactStoreLimits::default(),
            workspace_services,
        )
    }

    pub fn new_with_artifact_limits(
        workspace: String,
        artifact_limits: ArtifactStoreLimits,
    ) -> Self {
        let workspace_services =
            crate::workspace::WorkspaceServices::local(PathBuf::from(&workspace));
        Self::build(workspace, None, artifact_limits, workspace_services)
    }

    pub fn new_with_workspace_services(
        workspace: String,
        workspace_services: Arc<crate::workspace::WorkspaceServices>,
    ) -> Self {
        Self::build(
            workspace,
            None,
            ArtifactStoreLimits::default(),
            workspace_services,
        )
    }

    pub fn new_with_workspace_services_and_artifact_limits(
        workspace: String,
        workspace_services: Arc<crate::workspace::WorkspaceServices>,
        artifact_limits: ArtifactStoreLimits,
    ) -> Self {
        Self::build(workspace, None, artifact_limits, workspace_services)
    }

    fn build(
        workspace: String,
        command_env: Option<HashMap<String, String>>,
        artifact_limits: ArtifactStoreLimits,
        workspace_services: Arc<crate::workspace::WorkspaceServices>,
    ) -> Self {
        let workspace_path = PathBuf::from(&workspace);
        let command_env = command_env.map(Arc::new);
        let registry = Arc::new(ToolRegistry::with_artifact_limits_and_workspace_services(
            workspace_path.clone(),
            artifact_limits,
            Arc::clone(&workspace_services),
        ));
        if let Some(env) = command_env.clone() {
            registry.set_command_env(env);
        }

        // Register native Rust built-in tools — only those whose required
        // workspace capability is available, so the model never sees a tool
        // the backend cannot service.
        builtin::register_builtins(&registry, &workspace_services.capabilities());
        // Batch tool requires Arc<ToolRegistry>, registered separately
        builtin::register_batch(&registry);
        builtin::register_program(&registry);

        Self {
            workspace: workspace_path,
            registry,
            file_history: Arc::new(FileHistory::new(500)),
            command_env,
            workspace_services,
        }
    }

    fn check_workspace_boundary(
        name: &str,
        args: &serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<()> {
        let path_field = match name {
            "read" | "write" | "edit" | "patch" => Some("file_path"),
            "ls" | "grep" | "glob" => Some("path"),
            _ => None,
        };

        if let Some(field) = path_field {
            if let Some(path_str) = args.get(field).and_then(|v| v.as_str()) {
                ctx.resolve_workspace_path(path_str).map_err(|e| {
                    anyhow::anyhow!(
                        "Workspace boundary check failed for tool '{}' path '{}': {}",
                        name,
                        path_str,
                        e
                    )
                })?;
            }
        }

        Ok(())
    }

    pub fn workspace(&self) -> &PathBuf {
        &self.workspace
    }

    pub fn registry(&self) -> &Arc<ToolRegistry> {
        &self.registry
    }

    /// Get a stored tool artifact by URI.
    pub fn get_artifact(&self, artifact_uri: &str) -> Option<ToolArtifact> {
        self.registry.get_artifact(artifact_uri)
    }

    /// Return a clone of the executor's artifact store handle.
    pub fn artifact_store(&self) -> ArtifactStore {
        self.registry.artifact_store()
    }

    /// Replace the sink used for compact execution trace events.
    pub fn set_trace_sink(&self, sink: Arc<dyn crate::trace::TraceSink>) {
        self.registry.set_trace_sink(sink);
    }

    /// Return the currently configured execution trace sink.
    pub fn trace_sink(&self) -> Arc<dyn crate::trace::TraceSink> {
        self.registry.trace_sink()
    }

    pub fn command_env(&self) -> Option<Arc<HashMap<String, String>>> {
        self.command_env.clone()
    }

    pub fn register_dynamic_tool(&self, tool: Arc<dyn Tool>) {
        self.registry.register(tool);
    }

    pub fn unregister_dynamic_tool(&self, name: &str) {
        self.registry.unregister(name);
    }

    /// Unregister all dynamic tools whose names start with the given prefix.
    pub fn unregister_tools_by_prefix(&self, prefix: &str) {
        self.registry.unregister_by_prefix(prefix);
    }

    /// Replace the model-visible `program` tool with a custom PTC catalog.
    pub fn register_program_catalog(&self, catalog: crate::program::ProgramCatalog) {
        builtin::register_program_with_catalog(&self.registry, catalog);
    }

    fn capture_snapshot(&self, name: &str, args: &serde_json::Value) {
        let Some(local_root) = self.workspace_services.local_root() else {
            return;
        };

        if let Some(file_path) = file_history::extract_file_path(name, args) {
            let workspace_path = match self.workspace_services.normalize_path(&file_path) {
                Ok(path) => path,
                Err(e) => {
                    tracing::warn!(
                        "Skipping file snapshot for invalid path {}: {}",
                        file_path,
                        e
                    );
                    return;
                }
            };
            let path_to_read = if workspace_path.is_root() {
                local_root.to_path_buf()
            } else {
                local_root.join(workspace_path.as_str())
            };

            if !path_to_read.exists() {
                self.file_history.save_snapshot(&file_path, "", name);
                return;
            }

            match std::fs::read_to_string(&path_to_read) {
                Ok(content) => {
                    self.file_history.save_snapshot(&file_path, &content, name);
                    tracing::debug!(
                        "Captured file snapshot for {} before {} (version {})",
                        file_path,
                        name,
                        self.file_history.list_versions(&file_path).len() - 1,
                    );
                }
                Err(e) => {
                    tracing::warn!("Failed to capture snapshot for {}: {}", file_path, e);
                }
            }
        }
    }

    pub async fn execute(&self, name: &str, args: &serde_json::Value) -> Result<ToolResult> {
        let ctx = self.registry.context();
        if let Err(e) = Self::check_workspace_boundary(name, args, &ctx) {
            return Ok(ToolResult::error(name, e.to_string()));
        }

        log_tool_invocation(name, args);
        self.capture_snapshot(name, args);
        let mut result = self.registry.execute_with_context(name, args, &ctx).await;
        if let Ok(ref mut r) = result {
            self.attach_diff_metadata(name, args, r);
        }
        match &result {
            Ok(r) => tracing::info!("Tool {} completed with exit_code={}", name, r.exit_code),
            Err(e) => tracing::error!("Tool {} failed: {}", name, e),
        }
        result
    }

    pub async fn execute_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolResult> {
        Self::check_workspace_boundary(name, args, ctx)?;
        log_tool_invocation(name, args);
        self.capture_snapshot(name, args);
        let mut result = self.registry.execute_with_context(name, args, ctx).await;
        if let Ok(ref mut r) = result {
            self.attach_diff_metadata(name, args, r);
        }
        match &result {
            Ok(r) => tracing::info!("Tool {} completed with exit_code={}", name, r.exit_code),
            Err(e) => tracing::error!("Tool {} failed: {}", name, e),
        }
        result
    }

    fn attach_diff_metadata(&self, name: &str, args: &serde_json::Value, result: &mut ToolResult) {
        if !file_history::is_file_modifying_tool(name) {
            return;
        }
        let Some(file_path) = file_history::extract_file_path(name, args) else {
            return;
        };
        // Only store file_path in metadata, let translate_event read the actual content
        // using the session's correct workspace
        let meta = result.metadata.get_or_insert_with(|| serde_json::json!({}));
        meta["file_path"] = serde_json::Value::String(file_path);
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.registry.definitions()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::{
        CommandOutput, CommandRequest, WorkspaceCommandRunner, WorkspaceDirEntry, WorkspaceError,
        WorkspaceFileSystem, WorkspaceFileType, WorkspacePath, WorkspaceRef, WorkspaceResult,
        WorkspaceServices, WorkspaceWriteOutcome,
    };
    use async_trait::async_trait;
    use std::sync::RwLock;

    #[test]
    fn test_redacted_tool_log_summary_omits_values() {
        let args = serde_json::json!({
            "command": "export AWS_SECRET_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE && deploy",
            "timeout": 30
        });
        let summary = redacted_tool_log_summary("bash", &args);
        // Field names and size are logged...
        assert!(summary.contains("bash"));
        assert!(summary.contains("command"));
        assert!(summary.contains("timeout"));
        assert!(summary.contains("bytes"));
        // ...but never the values (the secret must not appear).
        assert!(!summary.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(!summary.contains("deploy"));
    }

    #[test]
    fn test_redacted_tool_log_summary_handles_non_object_args() {
        let summary = redacted_tool_log_summary("noop", &serde_json::json!("raw string"));
        assert!(summary.contains("noop"));
        assert!(summary.contains("arg_keys=[]"));
        assert!(!summary.contains("raw string"));
    }

    struct LargeArtifactTool;

    #[async_trait]
    impl Tool for LargeArtifactTool {
        fn name(&self) -> &str {
            "large_artifact"
        }

        fn description(&self) -> &str {
            "Produces large output for artifact API tests"
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {},
                "required": []
            })
        }

        async fn execute(
            &self,
            args: &serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput> {
            let suffix = args
                .get("suffix")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            Ok(ToolOutput::success(format!(
                "{}{}",
                "z".repeat(MAX_OUTPUT_SIZE + 1),
                suffix
            )))
        }
    }

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echoes the message argument"
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "message": { "type": "string" }
                },
                "required": ["message"]
            })
        }

        async fn execute(
            &self,
            args: &serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput> {
            Ok(ToolOutput::success(
                args["message"].as_str().unwrap_or_default(),
            ))
        }
    }

    #[derive(Default)]
    struct MemoryWorkspaceFs {
        files: RwLock<HashMap<String, String>>,
    }

    impl MemoryWorkspaceFs {
        fn insert(&self, path: &str, content: &str) {
            self.files
                .write()
                .unwrap()
                .insert(path.to_string(), content.to_string());
        }

        fn get(&self, path: &str) -> Option<String> {
            self.files.read().unwrap().get(path).cloned()
        }
    }

    #[async_trait]
    impl WorkspaceFileSystem for MemoryWorkspaceFs {
        async fn read_text(&self, path: &WorkspacePath) -> WorkspaceResult<String> {
            self.files
                .read()
                .unwrap()
                .get(path.as_str())
                .cloned()
                .ok_or_else(|| WorkspaceError::NotFound {
                    path: path.as_str().to_string(),
                })
        }

        async fn write_text(
            &self,
            path: &WorkspacePath,
            content: &str,
        ) -> WorkspaceResult<WorkspaceWriteOutcome> {
            self.insert(path.as_str(), content);
            Ok(WorkspaceWriteOutcome {
                bytes: content.len(),
                lines: content.lines().count(),
            })
        }

        async fn list_dir(&self, path: &WorkspacePath) -> WorkspaceResult<Vec<WorkspaceDirEntry>> {
            let prefix = if path.is_root() {
                String::new()
            } else {
                format!("{}/", path.as_str())
            };
            let files = self.files.read().unwrap();
            let mut entries = Vec::new();
            for name in files.keys() {
                if !name.starts_with(&prefix) {
                    continue;
                }
                let remaining = &name[prefix.len()..];
                if remaining.is_empty() || remaining.contains('/') {
                    continue;
                }
                entries.push(WorkspaceDirEntry {
                    name: remaining.to_string(),
                    kind: WorkspaceFileType::File,
                    size: files
                        .get(name)
                        .map(|content| content.len() as u64)
                        .unwrap_or(0),
                });
            }
            Ok(entries)
        }
    }

    struct MockCommandRunner;

    #[async_trait]
    impl WorkspaceCommandRunner for MockCommandRunner {
        async fn exec(&self, request: CommandRequest) -> Result<CommandOutput> {
            Ok(CommandOutput {
                output: format!("remote: {}\n", request.command),
                exit_code: 0,
                timed_out: false,
            })
        }
    }

    #[tokio::test]
    async fn test_tool_executor_creation() {
        let executor = ToolExecutor::new("/tmp".to_string());
        // Baseline tools on a raw ToolExecutor: 13
        assert_eq!(executor.registry.len(), 13);
    }

    #[tokio::test]
    async fn test_unknown_tool() {
        let executor = ToolExecutor::new("/tmp".to_string());
        let result = executor
            .execute("unknown", &serde_json::json!({}))
            .await
            .unwrap();
        assert_eq!(result.exit_code, 1);
        assert!(result.output.contains("Unknown tool"));
    }

    #[tokio::test]
    async fn test_builtin_tools_registered() {
        let executor = ToolExecutor::new("/tmp".to_string());
        let definitions = executor.definitions();

        assert!(definitions.iter().any(|t| t.name == "bash"));
        assert!(definitions.iter().any(|t| t.name == "read"));
        assert!(definitions.iter().any(|t| t.name == "write"));
        assert!(definitions.iter().any(|t| t.name == "edit"));
        assert!(definitions.iter().any(|t| t.name == "grep"));
        assert!(definitions.iter().any(|t| t.name == "glob"));
        assert!(definitions.iter().any(|t| t.name == "ls"));
        assert!(definitions.iter().any(|t| t.name == "patch"));
        assert!(definitions.iter().any(|t| t.name == "web_fetch"));
        assert!(definitions.iter().any(|t| t.name == "web_search"));
        assert!(definitions.iter().any(|t| t.name == "batch"));
    }

    #[tokio::test]
    async fn test_builtin_file_tools_use_workspace_services() {
        let fs = Arc::new(MemoryWorkspaceFs::default());
        fs.insert("remote.txt", "first\nsecond\n");
        let services = WorkspaceServices::builder(
            WorkspaceRef::new("browser-workspace", "browser://workspace"),
            fs.clone(),
        )
        .build();
        let executor = ToolExecutor::new_with_workspace_services_and_artifact_limits(
            "/server/local-placeholder".to_string(),
            services,
            ArtifactStoreLimits::default(),
        );
        let definitions = executor.definitions();
        assert!(definitions.iter().any(|tool| tool.name == "read"));
        assert!(definitions.iter().any(|tool| tool.name == "write"));
        assert!(definitions.iter().any(|tool| tool.name == "ls"));
        assert!(!definitions.iter().any(|tool| tool.name == "bash"));
        assert!(!definitions.iter().any(|tool| tool.name == "grep"));
        assert!(definitions.iter().any(|tool| tool.name == "edit"));
        assert!(definitions.iter().any(|tool| tool.name == "patch"));

        let read = executor
            .execute("read", &serde_json::json!({"file_path": "remote.txt"}))
            .await
            .unwrap();
        assert_eq!(read.exit_code, 0);
        assert!(read.output.contains("first"));

        let write = executor
            .execute(
                "write",
                &serde_json::json!({"file_path": "created.txt", "content": "remote write\n"}),
            )
            .await
            .unwrap();
        assert_eq!(write.exit_code, 0);
        assert_eq!(fs.get("created.txt").unwrap(), "remote write\n");

        let ls = executor
            .execute("ls", &serde_json::json!({}))
            .await
            .unwrap();
        assert_eq!(ls.exit_code, 0);
        assert!(ls.output.contains("created.txt"));
        assert!(ls.output.contains("remote.txt"));
    }

    #[tokio::test]
    async fn test_bash_uses_workspace_command_runner() {
        let fs = Arc::new(MemoryWorkspaceFs::default());
        let fs_backend: Arc<dyn WorkspaceFileSystem> = fs;
        let services = WorkspaceServices::builder(
            WorkspaceRef::new("remote-workspace", "remote://workspace"),
            fs_backend,
        )
        .command_runner(Arc::new(MockCommandRunner))
        .build();
        let executor = ToolExecutor::new_with_workspace_services_and_artifact_limits(
            "/server/local-placeholder".to_string(),
            services,
            ArtifactStoreLimits::default(),
        );
        assert!(executor
            .definitions()
            .iter()
            .any(|tool| tool.name == "bash"));

        let result = executor
            .execute("bash", &serde_json::json!({"command": "pwd"}))
            .await
            .unwrap();

        assert_eq!(result.exit_code, 0);
        assert_eq!(result.output, "remote: pwd\n");
    }

    #[tokio::test]
    async fn test_command_env_is_available_on_default_context() {
        let temp = tempfile::tempdir().unwrap();
        let mut env = HashMap::new();
        env.insert(
            "A3S_COMMAND_ENV_TEST".to_string(),
            "registry-env".to_string(),
        );

        let executor = ToolExecutor::new(temp.path().to_string_lossy().to_string());
        executor.registry().set_command_env(Arc::new(env));
        let context = executor.registry().context();
        assert_eq!(
            context
                .command_env
                .as_ref()
                .and_then(|env| env.get("A3S_COMMAND_ENV_TEST"))
                .map(String::as_str),
            Some("registry-env")
        );

        #[cfg(windows)]
        let command = "Write-Output $env:A3S_COMMAND_ENV_TEST";
        #[cfg(not(windows))]
        let command = "printf '%s' \"$A3S_COMMAND_ENV_TEST\"";

        let result = executor
            .execute("bash", &serde_json::json!({ "command": command }))
            .await
            .unwrap();

        assert_eq!(result.exit_code, 0, "{}", result.output);
        assert!(result.output.contains("registry-env"));
    }

    #[tokio::test]
    async fn test_execute_applies_workspace_boundary_for_default_context() {
        let workspace = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "secret").unwrap();

        let executor = ToolExecutor::new(workspace.path().to_string_lossy().to_string());
        let result = executor
            .execute(
                "grep",
                &serde_json::json!({
                    "pattern": "secret",
                    "path": outside.path().to_string_lossy()
                }),
            )
            .await
            .unwrap();

        assert_eq!(result.exit_code, 1);
        assert!(result.output.contains("Workspace boundary"));
        assert!(result.output.contains("escapes workspace"));
    }

    #[test]
    fn test_tool_result_success() {
        let result = ToolResult::success("test_tool", "output text".to_string());
        assert_eq!(result.name, "test_tool");
        assert_eq!(result.output, "output text");
        assert_eq!(result.exit_code, 0);
        assert!(result.metadata.is_none());
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("test_tool", "error message".to_string());
        assert_eq!(result.name, "test_tool");
        assert_eq!(result.output, "error message");
        assert_eq!(result.exit_code, 1);
        assert!(result.metadata.is_none());
    }

    #[test]
    fn test_tool_result_from_tool_output_success() {
        let output = ToolOutput {
            content: "success content".to_string(),
            success: true,
            metadata: None,
            images: Vec::new(),
            error_kind: None,
        };
        let result: ToolResult = output.into();
        assert_eq!(result.output, "success content");
        assert_eq!(result.exit_code, 0);
        assert!(result.metadata.is_none());
    }

    #[test]
    fn test_tool_result_from_tool_output_failure() {
        let output = ToolOutput {
            content: "failure content".to_string(),
            success: false,
            metadata: Some(serde_json::json!({"error": "test"})),
            images: Vec::new(),
            error_kind: None,
        };
        let result: ToolResult = output.into();
        assert_eq!(result.output, "failure content");
        assert_eq!(result.exit_code, 1);
        assert_eq!(result.metadata, Some(serde_json::json!({"error": "test"})));
    }

    #[test]
    fn test_tool_result_metadata_propagation() {
        let output = ToolOutput::success("content")
            .with_metadata(serde_json::json!({"_load_skill": true, "skill_name": "test"}));
        let result: ToolResult = output.into();
        assert_eq!(result.exit_code, 0);
        let meta = result.metadata.unwrap();
        assert_eq!(meta["_load_skill"], true);
        assert_eq!(meta["skill_name"], "test");
    }

    #[test]
    fn test_tool_executor_workspace() {
        let executor = ToolExecutor::new("/test/workspace".to_string());
        assert_eq!(executor.workspace().to_str().unwrap(), "/test/workspace");
    }

    #[test]
    fn test_tool_executor_registry() {
        let executor = ToolExecutor::new("/tmp".to_string());
        let registry = executor.registry();
        // Baseline tools on a raw ToolExecutor: 13
        assert_eq!(registry.len(), 13);
    }

    #[tokio::test]
    async fn test_tool_executor_get_artifact() {
        let executor = ToolExecutor::new("/tmp".to_string());
        executor.register_dynamic_tool(Arc::new(LargeArtifactTool));

        let result = executor
            .execute("large_artifact", &serde_json::json!({}))
            .await
            .unwrap();

        let artifact_uri = result.metadata.as_ref().unwrap()["artifact"]["artifact_uri"]
            .as_str()
            .unwrap();
        let artifact = executor.get_artifact(artifact_uri).expect("artifact");
        assert_eq!(artifact.tool_name, "large_artifact");
        assert_eq!(artifact.content.len(), MAX_OUTPUT_SIZE + 1);
        assert!(executor.artifact_store().get(artifact_uri).is_some());
    }

    #[tokio::test]
    async fn test_tool_executor_respects_artifact_limits() {
        let executor = ToolExecutor::new_with_artifact_limits(
            "/tmp".to_string(),
            ArtifactStoreLimits {
                max_artifacts: 1,
                max_bytes: usize::MAX,
            },
        );
        executor.register_dynamic_tool(Arc::new(LargeArtifactTool));

        let first = executor
            .execute("large_artifact", &serde_json::json!({}))
            .await
            .unwrap();
        let first_uri = first.metadata.as_ref().unwrap()["artifact"]["artifact_uri"]
            .as_str()
            .unwrap()
            .to_string();

        executor
            .execute("large_artifact", &serde_json::json!({ "suffix": "again" }))
            .await
            .unwrap();

        assert_eq!(executor.artifact_store().limits().max_artifacts, 1);
        assert_eq!(executor.artifact_store().len(), 1);
        assert!(executor.get_artifact(&first_uri).is_none());
    }

    #[tokio::test]
    async fn test_tool_executor_register_program_catalog_keeps_script_only_program_tool() {
        let executor = ToolExecutor::new("/tmp".to_string());
        let trace_sink = crate::trace::InMemoryTraceSink::default();
        executor.set_trace_sink(Arc::new(trace_sink.clone()));
        executor.register_dynamic_tool(Arc::new(EchoTool));
        let mut catalog = crate::program::ProgramCatalog::new();
        catalog.register(
            crate::program::ProgramTemplate::new("custom_echo", "Run a custom echo program")
                .with_parameter(crate::program::ProgramParameter::required(
                    "message",
                    "Message to echo",
                ))
                .with_step(
                    crate::program::ProgramStepTemplate::new(
                        "echo",
                        serde_json::json!({ "message": "{{message}}" }),
                    )
                    .with_label("echo_message"),
                ),
        );
        executor.register_program_catalog(catalog);

        let result = executor
            .execute(
                "program",
                &serde_json::json!({
                    "name": "custom_echo",
                    "inputs": {
                        "message": "hello from catalog"
                    }
                }),
            )
            .await
            .unwrap();

        assert_eq!(result.exit_code, 1);
        assert!(result.output.contains("type parameter is required"));

        let events = trace_sink.events();
        assert!(events.iter().any(|event| {
            event.kind == crate::trace::TraceEventKind::ToolExecution && event.name == "program"
        }));
        assert!(!events.iter().any(|event| {
            event.kind == crate::trace::TraceEventKind::ToolExecution && event.name == "echo"
        }));
    }

    #[test]
    fn test_max_output_size_constant() {
        assert_eq!(MAX_OUTPUT_SIZE, 100 * 1024);
    }

    #[test]
    fn test_max_read_lines_constant() {
        assert_eq!(MAX_READ_LINES, 2000);
    }

    #[test]
    fn test_max_line_length_constant() {
        assert_eq!(MAX_LINE_LENGTH, 2000);
    }

    #[test]
    fn test_truncate_tool_output_with_artifact_reference() {
        let output = "x".repeat(MAX_OUTPUT_SIZE + 1);
        let truncated = truncate_tool_output_with_artifact("test/tool", &output);

        let artifact = truncated.artifact.expect("artifact");
        assert!(truncated.content.contains("Full output artifact:"));
        assert_eq!(artifact.original_bytes, MAX_OUTPUT_SIZE + 1);
        assert_eq!(artifact.shown_bytes, MAX_OUTPUT_SIZE);
        assert!(artifact.artifact_id.starts_with("tool-output:test_tool:"));
        assert!(artifact
            .artifact_uri
            .starts_with("a3s://tool-output/test_tool/"));
    }

    #[test]
    fn test_tool_result_clone() {
        let result = ToolResult::success("test", "output".to_string());
        let cloned = result.clone();
        assert_eq!(result.name, cloned.name);
        assert_eq!(result.output, cloned.output);
        assert_eq!(result.exit_code, cloned.exit_code);
        assert_eq!(result.metadata, cloned.metadata);
    }

    #[test]
    fn test_tool_result_debug() {
        let result = ToolResult::success("test", "output".to_string());
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("test"));
        assert!(debug_str.contains("output"));
    }

    #[tokio::test]
    async fn test_execute_attaches_diff_metadata() {
        use tempfile::TempDir;
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("hello.txt");
        std::fs::write(&file, "before content\n").unwrap();

        let executor = ToolExecutor::new(dir.path().to_str().unwrap().to_string());
        let args = serde_json::json!({
            "file_path": "hello.txt",
            "content": "after content\n"
        });
        let result = executor.execute("write", &args).await.unwrap();

        let meta = result.metadata.expect("metadata should be present");
        assert_eq!(meta["before"], "before content\n");
        assert_eq!(meta["after"], "after content\n");
        assert_eq!(meta["file_path"], "hello.txt");
    }

    #[tokio::test]
    async fn test_execute_with_context_attaches_diff_metadata() {
        use tempfile::TempDir;
        let dir = TempDir::new().unwrap();
        let canonical_dir = dir.path().canonicalize().unwrap();
        let file = canonical_dir.join("ctx.txt");
        std::fs::write(&file, "original\n").unwrap();

        let executor = ToolExecutor::new(canonical_dir.to_str().unwrap().to_string());
        let ctx = ToolContext::new(canonical_dir.clone());
        let args = serde_json::json!({
            "file_path": "ctx.txt",
            "content": "updated\n"
        });
        let result = executor
            .execute_with_context("write", &args, &ctx)
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0, "write tool failed: {}", result.output);

        let meta = result.metadata.expect("metadata should be present");
        assert_eq!(meta["before"], "original\n");
        assert_eq!(meta["after"], "updated\n");
        assert_eq!(meta["file_path"], "ctx.txt");
    }
}
