//! Tool Registry
//!
//! Central registry for all tools (built-in and dynamic).
//! Provides thread-safe registration, lookup, and execution.

use super::artifacts::{ArtifactStore, ArtifactStoreLimits, ToolArtifact};
use super::types::{Tool, ToolContext, ToolOutput};
use super::ToolResult;
use super::{
    merge_tool_output_artifact_metadata, truncate_tool_output_with_artifact, ToolOutputArtifact,
};
use crate::llm::ToolDefinition;
use crate::trace::{InMemoryTraceSink, TraceEvent, TraceSink};
use anyhow::Result;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Tool registry for managing all available tools
pub struct ToolRegistry {
    tools: RwLock<HashMap<String, Arc<dyn Tool>>>,
    /// Names of builtin tools that cannot be overridden
    builtins: RwLock<std::collections::HashSet<String>>,
    context: RwLock<ToolContext>,
    artifact_store: ArtifactStore,
    trace_sink: RwLock<Arc<dyn TraceSink>>,
}

impl ToolRegistry {
    /// Create a new tool registry
    pub fn new(workspace: PathBuf) -> Self {
        Self::with_artifact_limits(workspace, ArtifactStoreLimits::default())
    }

    /// Create a new tool registry with custom artifact retention limits.
    pub fn with_artifact_limits(workspace: PathBuf, artifact_limits: ArtifactStoreLimits) -> Self {
        Self::with_artifact_limits_and_workspace_services(
            workspace.clone(),
            artifact_limits,
            crate::workspace::WorkspaceServices::local(workspace),
        )
    }

    /// Create a new tool registry with custom artifact limits and workspace backend.
    pub fn with_artifact_limits_and_workspace_services(
        workspace: PathBuf,
        artifact_limits: ArtifactStoreLimits,
        workspace_services: Arc<crate::workspace::WorkspaceServices>,
    ) -> Self {
        let context = ToolContext::new(workspace).with_workspace_services(workspace_services);
        Self {
            tools: RwLock::new(HashMap::new()),
            builtins: RwLock::new(std::collections::HashSet::new()),
            context: RwLock::new(context),
            artifact_store: ArtifactStore::with_limits(artifact_limits),
            trace_sink: RwLock::new(Arc::new(InMemoryTraceSink::default())),
        }
    }

    /// Register a builtin tool (cannot be overridden by dynamic tools)
    pub fn register_builtin(&self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        let mut tools = self.tools.write().unwrap();
        let mut builtins = self.builtins.write().unwrap();
        tracing::debug!("Registering builtin tool: {}", name);
        tools.insert(name.clone(), tool);
        builtins.insert(name);
    }

    /// Register a tool
    ///
    /// If a tool with the same name already exists as a builtin, the registration
    /// is rejected to prevent shadowing of core tools.
    pub fn register(&self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        let builtins = self.builtins.read().unwrap();
        if builtins.contains(&name) {
            tracing::warn!(
                "Rejected registration of tool '{}': cannot shadow builtin",
                name
            );
            return;
        }
        drop(builtins);
        let mut tools = self.tools.write().unwrap();
        tracing::debug!("Registering tool: {}", name);
        tools.insert(name, tool);
    }

    /// Unregister a tool by name
    ///
    /// Returns true if the tool was found and removed.
    pub fn unregister(&self, name: &str) -> bool {
        let mut tools = self.tools.write().unwrap();
        tracing::debug!("Unregistering tool: {}", name);
        tools.remove(name).is_some()
    }

    /// Unregister all tools whose names start with the given prefix.
    pub fn unregister_by_prefix(&self, prefix: &str) {
        let mut tools = self.tools.write().unwrap();
        tools.retain(|name, _| !name.starts_with(prefix));
        tracing::debug!("Unregistered tools with prefix: {}", prefix);
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        let tools = self.tools.read().unwrap();
        tools.get(name).cloned()
    }

    /// Check if a tool exists
    pub fn contains(&self, name: &str) -> bool {
        let tools = self.tools.read().unwrap();
        tools.contains_key(name)
    }

    /// Get all tool definitions for LLM
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        let tools = self.tools.read().unwrap();
        tools
            .values()
            .map(|tool| ToolDefinition {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.parameters(),
            })
            .collect()
    }

    /// List all registered tool names
    pub fn list(&self) -> Vec<String> {
        let tools = self.tools.read().unwrap();
        tools.keys().cloned().collect()
    }

    /// Get the number of registered tools
    pub fn len(&self) -> usize {
        let tools = self.tools.read().unwrap();
        tools.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the tool context
    pub fn context(&self) -> ToolContext {
        self.context.read().unwrap().clone()
    }

    /// Return a clone of the registry's artifact store handle.
    pub fn artifact_store(&self) -> ArtifactStore {
        self.artifact_store.clone()
    }

    /// Get a stored tool artifact by URI.
    pub fn get_artifact(&self, artifact_uri: &str) -> Option<ToolArtifact> {
        self.artifact_store.get(artifact_uri)
    }

    /// Replace the trace sink used for compact tool/program execution events.
    pub fn set_trace_sink(&self, sink: Arc<dyn TraceSink>) {
        *self.trace_sink.write().unwrap() = sink;
    }

    /// Return the current trace sink.
    pub fn trace_sink(&self) -> Arc<dyn TraceSink> {
        Arc::clone(&self.trace_sink.read().unwrap())
    }

    /// Set the search configuration for the tool context
    pub fn set_search_config(&self, config: crate::config::SearchConfig) {
        let mut ctx = self.context.write().unwrap();
        *ctx = ctx.clone().with_search_config(config);
    }

    /// Set a sandbox executor so that `bash` tool calls use the sandbox even
    /// when executed without an explicit `ToolContext` (i.e., via `execute()`).
    pub fn set_sandbox(&self, sandbox: std::sync::Arc<dyn crate::sandbox::BashSandbox>) {
        let mut ctx = self.context.write().unwrap();
        *ctx = ctx.clone().with_sandbox(sandbox);
    }

    /// Set environment overrides used by subprocess-backed tools when executed
    /// without an explicit context.
    pub fn set_command_env(&self, env: Arc<HashMap<String, String>>) {
        let mut ctx = self.context.write().unwrap();
        *ctx = ctx.clone().with_command_env(env);
    }

    /// Execute a tool by name using the registry's default context
    pub async fn execute(&self, name: &str, args: &serde_json::Value) -> Result<ToolResult> {
        let ctx = self.context();
        self.execute_with_context(name, args, &ctx).await
    }

    /// Execute a tool by name with an external context
    pub async fn execute_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolResult> {
        let start = std::time::Instant::now();

        let tool = self.get(name);

        let result = match tool {
            Some(tool) => {
                let mut output = tool.execute(args, ctx).await?;
                let original_content = output.content.clone();
                let truncated = truncate_tool_output_with_artifact(name, &output.content);
                output.content = truncated.content;
                if let Some(artifact) = truncated.artifact {
                    self.store_tool_artifact(name, &original_content, &artifact);
                    output.metadata = Some(merge_tool_output_artifact_metadata(
                        output.metadata,
                        &artifact,
                    ));
                }
                Ok(ToolResult {
                    name: name.to_string(),
                    output: output.content,
                    exit_code: if output.success { 0 } else { 1 },
                    metadata: output.metadata,
                    images: output.images,
                })
            }
            None => Ok(ToolResult::error(name, format!("Unknown tool: {}", name))),
        };

        if let Ok(ref r) = result {
            crate::telemetry::record_tool_result(r.exit_code, start.elapsed());
            self.record_trace_event(name, r, start.elapsed());
        }

        result
    }

    /// Execute a tool and return raw output using the registry's default context
    pub async fn execute_raw(
        &self,
        name: &str,
        args: &serde_json::Value,
    ) -> Result<Option<ToolOutput>> {
        let ctx = self.context();
        self.execute_raw_with_context(name, args, &ctx).await
    }

    /// Execute a tool and return raw output with an external context
    pub async fn execute_raw_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<Option<ToolOutput>> {
        let tool = self.get(name);

        match tool {
            Some(tool) => {
                let mut output = tool.execute(args, ctx).await?;
                let original_content = output.content.clone();
                let truncated = truncate_tool_output_with_artifact(name, &output.content);
                output.content = truncated.content;
                if let Some(artifact) = truncated.artifact {
                    self.store_tool_artifact(name, &original_content, &artifact);
                    output.metadata = Some(merge_tool_output_artifact_metadata(
                        output.metadata,
                        &artifact,
                    ));
                }
                Ok(Some(output))
            }
            None => Ok(None),
        }
    }

    fn store_tool_artifact(&self, tool_name: &str, content: &str, artifact: &ToolOutputArtifact) {
        self.artifact_store.put(ToolArtifact {
            artifact_id: artifact.artifact_id.clone(),
            artifact_uri: artifact.artifact_uri.clone(),
            tool_name: tool_name.to_string(),
            content: content.to_string(),
            original_bytes: artifact.original_bytes,
            shown_bytes: artifact.shown_bytes,
        });
    }

    fn record_trace_event(&self, name: &str, result: &ToolResult, duration: std::time::Duration) {
        let sink = self.trace_sink();
        sink.record(TraceEvent::tool_execution(
            name,
            result.exit_code == 0,
            result.exit_code,
            duration,
            result.output.len(),
            result.metadata.as_ref(),
        ));

        if name == "program" {
            sink.record(TraceEvent::program_execution(
                name,
                result.exit_code == 0,
                result.exit_code,
                duration,
                result.output.len(),
                result.metadata.as_ref(),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{InMemoryTraceSink, TraceEventKind};
    use async_trait::async_trait;

    struct MockTool {
        name: String,
    }

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "A mock tool for testing"
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
            _args: &serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput> {
            Ok(ToolOutput::success("mock output"))
        }
    }

    #[test]
    fn test_registry_register_and_get() {
        let registry = ToolRegistry::new(PathBuf::from("/tmp"));

        let tool = Arc::new(MockTool {
            name: "test".to_string(),
        });
        registry.register(tool);

        assert!(registry.contains("test"));
        assert!(!registry.contains("nonexistent"));

        let retrieved = registry.get("test");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name(), "test");
    }

    #[test]
    fn test_registry_unregister() {
        let registry = ToolRegistry::new(PathBuf::from("/tmp"));

        let tool = Arc::new(MockTool {
            name: "test".to_string(),
        });
        registry.register(tool);

        assert!(registry.contains("test"));
        assert!(registry.unregister("test"));
        assert!(!registry.contains("test"));
        assert!(!registry.unregister("test")); // Already removed
    }

    #[test]
    fn test_registry_definitions() {
        let registry = ToolRegistry::new(PathBuf::from("/tmp"));

        registry.register(Arc::new(MockTool {
            name: "tool1".to_string(),
        }));
        registry.register(Arc::new(MockTool {
            name: "tool2".to_string(),
        }));

        let definitions = registry.definitions();
        assert_eq!(definitions.len(), 2);
    }

    #[tokio::test]
    async fn test_registry_execute() {
        let registry = ToolRegistry::new(PathBuf::from("/tmp"));

        registry.register(Arc::new(MockTool {
            name: "test".to_string(),
        }));

        let result = registry
            .execute("test", &serde_json::json!({}))
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0);
        assert_eq!(result.output, "mock output");
    }

    #[tokio::test]
    async fn test_registry_execute_unknown() {
        let registry = ToolRegistry::new(PathBuf::from("/tmp"));

        let result = registry
            .execute("unknown", &serde_json::json!({}))
            .await
            .unwrap();
        assert_eq!(result.exit_code, 1);
        assert!(result.output.contains("Unknown tool"));
    }

    #[tokio::test]
    async fn test_registry_execute_with_context_success() {
        let registry = ToolRegistry::new(PathBuf::from("/tmp"));
        let ctx = ToolContext::new(PathBuf::from("/tmp"));
        let trace_sink = InMemoryTraceSink::default();
        registry.set_trace_sink(Arc::new(trace_sink.clone()));

        registry.register(Arc::new(MockTool {
            name: "my_tool".to_string(),
        }));

        let result = registry
            .execute_with_context("my_tool", &serde_json::json!({}), &ctx)
            .await
            .unwrap();
        assert_eq!(result.name, "my_tool");
        assert_eq!(result.exit_code, 0);
        assert_eq!(result.output, "mock output");

        let events = trace_sink.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, TraceEventKind::ToolExecution);
        assert_eq!(events[0].name, "my_tool");
        assert!(events[0].success);
        assert_eq!(events[0].output_bytes, "mock output".len());
    }

    #[tokio::test]
    async fn test_registry_execute_with_context_unknown_tool() {
        let registry = ToolRegistry::new(PathBuf::from("/tmp"));
        let ctx = ToolContext::new(PathBuf::from("/tmp"));

        let result = registry
            .execute_with_context("nonexistent", &serde_json::json!({}), &ctx)
            .await
            .unwrap();
        assert_eq!(result.exit_code, 1);
        assert!(result.output.contains("Unknown tool: nonexistent"));
    }

    struct FailingTool;

    #[async_trait]
    impl Tool for FailingTool {
        fn name(&self) -> &str {
            "failing"
        }

        fn description(&self) -> &str {
            "A tool that returns failure"
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
            _args: &serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput> {
            Ok(ToolOutput::error("something went wrong"))
        }
    }

    #[tokio::test]
    async fn test_registry_execute_failing_tool() {
        let registry = ToolRegistry::new(PathBuf::from("/tmp"));
        registry.register(Arc::new(FailingTool));

        let result = registry
            .execute("failing", &serde_json::json!({}))
            .await
            .unwrap();
        assert_eq!(result.exit_code, 1);
        assert_eq!(result.output, "something went wrong");
    }

    struct LargeOutputTool;

    #[async_trait]
    impl Tool for LargeOutputTool {
        fn name(&self) -> &str {
            "large_output"
        }

        fn description(&self) -> &str {
            "A tool that returns more than the maximum output size"
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
            _args: &serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput> {
            Ok(ToolOutput::success(
                "x".repeat(super::super::MAX_OUTPUT_SIZE + 1),
            ))
        }
    }

    #[tokio::test]
    async fn test_registry_truncates_large_tool_output() {
        let registry = ToolRegistry::new(PathBuf::from("/tmp"));
        let trace_sink = InMemoryTraceSink::default();
        registry.set_trace_sink(Arc::new(trace_sink.clone()));
        registry.register(Arc::new(LargeOutputTool));

        let result = registry
            .execute("large_output", &serde_json::json!({}))
            .await
            .unwrap();

        assert_eq!(result.exit_code, 0);
        assert!(result.output.contains("[tool output truncated:"));
        assert!(result
            .output
            .contains("Full output artifact: a3s://tool-output/large_output/"));
        assert!(result.output.len() < super::super::MAX_OUTPUT_SIZE + 512);
        let metadata = result.metadata.expect("artifact metadata");
        assert_eq!(
            metadata["artifact"]["original_bytes"],
            serde_json::json!(super::super::MAX_OUTPUT_SIZE + 1)
        );
        assert_eq!(
            metadata["artifact"]["shown_bytes"],
            serde_json::json!(super::super::MAX_OUTPUT_SIZE)
        );
        assert!(metadata["artifact"]["artifact_id"]
            .as_str()
            .unwrap()
            .starts_with("tool-output:large_output:"));
        assert!(metadata["artifact"]["artifact_uri"]
            .as_str()
            .unwrap()
            .starts_with("a3s://tool-output/large_output/"));

        let artifact_uri = metadata["artifact"]["artifact_uri"].as_str().unwrap();
        let artifact = registry
            .get_artifact(artifact_uri)
            .expect("full output artifact");
        assert_eq!(artifact.tool_name, "large_output");
        assert_eq!(artifact.original_bytes, super::super::MAX_OUTPUT_SIZE + 1);
        assert_eq!(artifact.shown_bytes, super::super::MAX_OUTPUT_SIZE);
        assert_eq!(
            artifact.content,
            "x".repeat(super::super::MAX_OUTPUT_SIZE + 1)
        );

        let events = trace_sink.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].artifact_uris, vec![artifact_uri]);
    }

    #[tokio::test]
    async fn test_registry_execute_raw_success() {
        let registry = ToolRegistry::new(PathBuf::from("/tmp"));
        registry.register(Arc::new(MockTool {
            name: "raw_test".to_string(),
        }));

        let output = registry
            .execute_raw("raw_test", &serde_json::json!({}))
            .await
            .unwrap();
        assert!(output.is_some());
        let output = output.unwrap();
        assert!(output.success);
        assert_eq!(output.content, "mock output");
    }

    #[tokio::test]
    async fn test_registry_execute_raw_stores_truncated_artifact() {
        let registry = ToolRegistry::new(PathBuf::from("/tmp"));
        registry.register(Arc::new(LargeOutputTool));

        let output = registry
            .execute_raw("large_output", &serde_json::json!({}))
            .await
            .unwrap()
            .expect("raw output");

        assert!(output.content.contains("[tool output truncated:"));
        let metadata = output.metadata.expect("artifact metadata");
        let artifact_uri = metadata["artifact"]["artifact_uri"].as_str().unwrap();
        let artifact = registry
            .get_artifact(artifact_uri)
            .expect("full output artifact");
        assert_eq!(artifact.tool_name, "large_output");
        assert_eq!(artifact.content.len(), super::super::MAX_OUTPUT_SIZE + 1);
    }

    #[tokio::test]
    async fn test_registry_execute_raw_unknown() {
        let registry = ToolRegistry::new(PathBuf::from("/tmp"));

        let output = registry
            .execute_raw("missing", &serde_json::json!({}))
            .await
            .unwrap();
        assert!(output.is_none());
    }

    #[test]
    fn test_registry_list() {
        let registry = ToolRegistry::new(PathBuf::from("/tmp"));
        registry.register(Arc::new(MockTool {
            name: "alpha".to_string(),
        }));
        registry.register(Arc::new(MockTool {
            name: "beta".to_string(),
        }));

        let names = registry.list();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"alpha".to_string()));
        assert!(names.contains(&"beta".to_string()));
    }

    #[test]
    fn test_registry_len_and_is_empty() {
        let registry = ToolRegistry::new(PathBuf::from("/tmp"));
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);

        registry.register(Arc::new(MockTool {
            name: "t".to_string(),
        }));
        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_registry_replace_tool() {
        let registry = ToolRegistry::new(PathBuf::from("/tmp"));
        registry.register(Arc::new(MockTool {
            name: "dup".to_string(),
        }));
        registry.register(Arc::new(MockTool {
            name: "dup".to_string(),
        }));
        // Should still have only 1 tool (replaced)
        assert_eq!(registry.len(), 1);
    }
}
