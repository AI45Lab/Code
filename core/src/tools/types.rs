//! Core types for the extensible tool system

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc};

/// Sender for streaming tool output deltas during execution.
pub type ToolEventSender = mpsc::Sender<ToolStreamEvent>;

/// Events emitted by tools during execution
#[derive(Debug, Clone)]
pub enum ToolStreamEvent {
    /// Intermediate output delta (e.g., a line of stdout from bash)
    OutputDelta(String),
}

/// Tool execution context
///
/// Provides tools with access to workspace and other runtime information.
#[derive(Clone)]
pub struct ToolContext {
    /// Workspace root directory (sandbox boundary)
    pub workspace: PathBuf,
    /// Optional session ID for session-aware tools
    pub session_id: Option<String>,
    /// Optional sender for streaming tool output deltas during execution
    pub event_tx: Option<ToolEventSender>,
    /// Optional agent event sender for tools that emit high-level agent events (e.g., SubagentStart)
    pub agent_event_tx: Option<broadcast::Sender<crate::agent::AgentEvent>>,
    /// Optional search configuration for web_search tool
    pub search_config: Option<Arc<crate::config::SearchConfig>>,
    /// Optional sandbox for routing `bash` tool execution through A3S Box.
    pub sandbox: Option<std::sync::Arc<dyn crate::sandbox::BashSandbox>>,
    /// Optional command environment overrides for subprocess-based tools.
    pub command_env: Option<Arc<HashMap<String, String>>>,
    /// Host-provided workspace capabilities used by built-in tools.
    pub workspace_services: Arc<crate::workspace::WorkspaceServices>,
}

impl std::fmt::Debug for ToolContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolContext")
            .field("workspace", &self.workspace)
            .field("session_id", &self.session_id)
            .field("sandbox", &self.sandbox.is_some())
            .field("workspace_services", &self.workspace_services)
            .finish()
    }
}

impl ToolContext {
    pub fn new(workspace: PathBuf) -> Self {
        let canonical_workspace = workspace
            .canonicalize()
            .unwrap_or_else(|_| workspace.clone());
        Self {
            workspace: canonical_workspace,
            session_id: None,
            event_tx: None,
            agent_event_tx: None,
            search_config: None,
            sandbox: None,
            command_env: None,
            workspace_services: crate::workspace::WorkspaceServices::local(workspace),
        }
    }

    /// Set the session ID for this context
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set the event sender for streaming tool output
    pub fn with_event_tx(mut self, tx: ToolEventSender) -> Self {
        self.event_tx = Some(tx);
        self
    }

    /// Set the agent event sender for high-level agent events (e.g., SubagentStart/End)
    pub fn with_agent_event_tx(mut self, tx: broadcast::Sender<crate::agent::AgentEvent>) -> Self {
        self.agent_event_tx = Some(tx);
        self
    }

    /// Set the search configuration
    pub fn with_search_config(mut self, config: crate::config::SearchConfig) -> Self {
        self.search_config = Some(Arc::new(config));
        self
    }

    /// Set a sandbox executor for the `bash` tool.
    pub fn with_sandbox(
        mut self,
        sandbox: std::sync::Arc<dyn crate::sandbox::BashSandbox>,
    ) -> Self {
        self.sandbox = Some(sandbox);
        self
    }

    /// Set environment overrides for subprocess-based tools such as `bash`.
    pub fn with_command_env(mut self, env: Arc<HashMap<String, String>>) -> Self {
        self.command_env = Some(env);
        self
    }

    /// Set host-provided workspace capabilities for built-in tools.
    pub fn with_workspace_services(
        mut self,
        services: Arc<crate::workspace::WorkspaceServices>,
    ) -> Self {
        self.workspace_services = services;
        self
    }

    /// Normalize a user-supplied path through the configured workspace backend.
    pub fn resolve_workspace_path(&self, path: &str) -> Result<crate::workspace::WorkspacePath> {
        self.workspace_services.normalize_path(path)
    }

    /// Resolve path relative to workspace, ensuring it stays within sandbox.
    ///
    /// Deprecated: returns a host-filesystem `PathBuf` that is meaningless for
    /// virtual / DFS / browser workspace backends. New code should call
    /// [`Self::resolve_workspace_path`] and route I/O through
    /// `workspace_services.fs()` instead.
    #[deprecated(
        note = "Use resolve_workspace_path() and route I/O through workspace_services.fs() for non-local backends"
    )]
    pub fn resolve_path(&self, path: &str) -> Result<PathBuf> {
        if self.workspace_services.local_root().is_none() {
            anyhow::bail!(
                "resolve_path is only valid for local workspaces; this session uses a non-local workspace backend, call resolve_workspace_path() instead"
            );
        }
        a3s_common::tools::resolve_path(&self.workspace, path).map_err(|e| anyhow::anyhow!("{}", e))
    }

    /// Resolve path for writing (allows non-existent files).
    ///
    /// Deprecated: see [`Self::resolve_path`].
    #[deprecated(
        note = "Use resolve_workspace_path() and route I/O through workspace_services.fs() for non-local backends"
    )]
    pub fn resolve_path_for_write(&self, path: &str) -> Result<PathBuf> {
        if self.workspace_services.local_root().is_none() {
            anyhow::bail!(
                "resolve_path_for_write is only valid for local workspaces; this session uses a non-local workspace backend, call resolve_workspace_path() instead"
            );
        }
        a3s_common::tools::resolve_path_for_write(&self.workspace, path)
            .map_err(|e| anyhow::anyhow!("{}", e))
    }
}

/// Tool execution output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    /// Output content (text or base64 for binary)
    pub content: String,
    /// Whether execution was successful
    pub success: bool,
    /// Optional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    /// Optional image attachments from tool execution (e.g., screenshots).
    ///
    /// When present, these are included in the tool result message sent to
    /// the LLM as multi-modal content blocks alongside the text content.
    #[serde(skip)]
    pub images: Vec<crate::llm::Attachment>,
}

impl ToolOutput {
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            success: true,
            metadata: None,
            images: Vec::new(),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: message.into(),
            success: false,
            metadata: None,
            images: Vec::new(),
        }
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Attach images to the tool output.
    ///
    /// These will be included as multi-modal content blocks in the tool
    /// result message sent to the LLM.
    pub fn with_images(mut self, images: Vec<crate::llm::Attachment>) -> Self {
        self.images = images;
        self
    }
}

/// Tool trait - the core abstraction for all tools
///
/// Implement this trait to create custom tools that can be registered
/// with the ToolRegistry.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name (must be unique within registry)
    fn name(&self) -> &str;

    /// Human-readable description for LLM
    fn description(&self) -> &str;

    /// JSON Schema for tool parameters
    fn parameters(&self) -> serde_json::Value;

    /// Execute the tool with given arguments
    async fn execute(&self, args: &serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(deprecated)]
    fn test_tool_context_resolve_path() {
        let temp_dir = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(temp_dir.path().to_path_buf());

        // Create a test file
        let test_file = temp_dir.path().join("file.txt");
        std::fs::write(&test_file, "test").unwrap();

        // Relative path to existing file
        let resolved = ctx.resolve_path("file.txt");
        assert!(resolved.is_ok());

        // Non-existent file should return error
        let resolved = ctx.resolve_path("nonexistent.txt");
        assert!(resolved.is_err());
    }

    #[test]
    fn test_tool_output_success() {
        let output = ToolOutput::success("Hello");
        assert!(output.success);
        assert_eq!(output.content, "Hello");
    }

    #[test]
    fn test_tool_output_error() {
        let output = ToolOutput::error("Failed");
        assert!(!output.success);
        assert_eq!(output.content, "Failed");
    }

    #[test]
    fn test_tool_output_images_default_empty() {
        let output = ToolOutput::success("ok");
        assert!(output.images.is_empty());
    }

    #[test]
    fn test_tool_output_with_images() {
        let images = vec![crate::llm::Attachment::png(vec![1, 2, 3])];
        let output = ToolOutput::success("screenshot taken").with_images(images);
        assert_eq!(output.images.len(), 1);
        assert_eq!(output.images[0].media_type, "image/png");
    }

    #[test]
    fn test_tool_output_with_metadata_and_images() {
        let output = ToolOutput::success("done")
            .with_metadata(serde_json::json!({"key": "val"}))
            .with_images(vec![crate::llm::Attachment::jpeg(vec![0xFF])]);
        assert!(output.metadata.is_some());
        assert_eq!(output.images.len(), 1);
    }
}
