//! Task tools for delegated child runs.
//!
//! The Task tool allows the main agent to delegate specialized work to focused
//! child runs. Each child run gets bounded context and the permissions declared
//! by its agent definition.
//!
//! ## Usage
//!
//! ```json
//! {
//!   "agent": "explore",
//!   "description": "Find authentication code",
//!   "prompt": "Search for files related to user authentication..."
//! }
//! ```

use crate::agent::{AgentConfig, AgentEvent, AgentLoop};
use crate::llm::LlmClient;
use crate::mcp::manager::McpManager;
use crate::subagent::AgentRegistry;
use crate::tools::types::{Tool, ToolContext, ToolOutput};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio::task::JoinSet;

const TASK_OUTPUT_CONTEXT_LIMIT: usize = 4_000;
const TASK_OUTPUT_CONTEXT_HEAD: usize = 3_000;
const TASK_OUTPUT_CONTEXT_TAIL: usize = 800;

/// Task tool parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskParams {
    /// Agent type to use (explore, general, plan, verification, review, etc.)
    pub agent: String,
    /// Short description of the task (for display)
    pub description: String,
    /// Detailed prompt for the agent
    pub prompt: String,
    /// Optional: run in background (default: false)
    #[serde(default)]
    pub background: bool,
    /// Optional: maximum steps for this task
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_steps: Option<usize>,
}

/// Task tool result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    /// Task output from the delegated child run.
    pub output: String,
    /// Child session ID
    pub session_id: String,
    /// Agent type used
    pub agent: String,
    /// Whether the task succeeded
    pub success: bool,
    /// Task ID for tracking
    pub task_id: String,
}

fn compact_task_output(output: &str) -> (String, bool) {
    if output.len() <= TASK_OUTPUT_CONTEXT_LIMIT {
        return (output.to_string(), false);
    }

    let head = crate::text::truncate_utf8(output, TASK_OUTPUT_CONTEXT_HEAD);
    let tail_start = output
        .char_indices()
        .find_map(|(idx, _)| {
            if output.len().saturating_sub(idx) <= TASK_OUTPUT_CONTEXT_TAIL {
                Some(idx)
            } else {
                None
            }
        })
        .unwrap_or(output.len());
    let tail = &output[tail_start..];

    (
        format!(
            "{}\n\n[{} bytes omitted from delegated task output]\n\n{}",
            head,
            output.len().saturating_sub(head.len() + tail.len()),
            tail
        ),
        true,
    )
}

fn task_artifact_id(result: &TaskResult) -> String {
    format!("task-output:{}", result.task_id)
}

fn task_artifact_uri(result: &TaskResult) -> String {
    format!(
        "a3s://tasks/{}/runs/{}/output",
        result.session_id, result.task_id
    )
}

fn format_task_result_for_context(result: &TaskResult) -> (String, bool) {
    let (output, truncated) = compact_task_output(&result.output);
    let status = if result.success {
        "completed"
    } else {
        "failed"
    };
    let artifact_id = task_artifact_id(result);
    let artifact_uri = task_artifact_uri(result);
    let mut formatted = format!(
        "Task {status}: {}\nAgent: {}\nSession: {}\nTask ID: {}\nArtifact ID: {}\nArtifact URI: {}\n",
        result.task_id, result.agent, result.session_id, result.task_id, artifact_id, artifact_uri
    );
    if truncated {
        formatted.push_str(
            "Output excerpt: truncated for parent context. Use the artifact URI or child run session/events if exact omitted content is needed.\n",
        );
    } else {
        formatted.push_str("Output:\n");
    }
    formatted.push_str(&output);
    (formatted, truncated)
}

/// Task executor for delegated child runs.
pub struct TaskExecutor {
    /// Agent registry for looking up agent definitions
    registry: Arc<AgentRegistry>,
    /// LLM client used to power child agent loops
    llm_client: Arc<dyn LlmClient>,
    /// Workspace path shared with child agents
    workspace: String,
    /// Optional MCP manager for registering MCP tools in child sessions
    mcp_manager: Option<Arc<McpManager>>,
    /// Parent capabilities to inherit into child runs.
    parent_context: Option<crate::child_run::ChildRunContext>,
}

impl TaskExecutor {
    /// Create a new task executor
    pub fn new(
        registry: Arc<AgentRegistry>,
        llm_client: Arc<dyn LlmClient>,
        workspace: String,
    ) -> Self {
        Self {
            registry,
            llm_client,
            workspace,
            mcp_manager: None,
            parent_context: None,
        }
    }

    /// Create a new task executor with MCP manager for tool inheritance
    pub fn with_mcp(
        registry: Arc<AgentRegistry>,
        llm_client: Arc<dyn LlmClient>,
        workspace: String,
        mcp_manager: Arc<McpManager>,
    ) -> Self {
        Self {
            registry,
            llm_client,
            workspace,
            mcp_manager: Some(mcp_manager),
            parent_context: None,
        }
    }

    /// Set parent session capabilities to inherit into child runs.
    pub fn with_parent_context(mut self, ctx: crate::child_run::ChildRunContext) -> Self {
        self.parent_context = Some(ctx);
        self
    }

    /// Execute a task by spawning an isolated child AgentLoop.
    pub async fn execute(
        &self,
        params: TaskParams,
        event_tx: Option<broadcast::Sender<AgentEvent>>,
    ) -> Result<TaskResult> {
        let task_id = format!("task-{}", uuid::Uuid::new_v4());
        let session_id = format!("task-run-{}", task_id);

        let agent = self
            .registry
            .get(&params.agent)
            .context(format!("Unknown agent type: '{}'", params.agent))?;

        if let Some(ref tx) = event_tx {
            let _ = tx.send(AgentEvent::SubagentStart {
                task_id: task_id.clone(),
                session_id: session_id.clone(),
                parent_session_id: String::new(),
                agent: params.agent.clone(),
                description: params.description.clone(),
            });
        }

        // Build a child ToolExecutor. Task tools are intentionally omitted
        // here to prevent unlimited delegation nesting.
        let child_executor = if let Some(ref parent_ctx) = self.parent_context {
            if let Some(ref services) = parent_ctx.workspace_services {
                crate::tools::ToolExecutor::new_with_workspace_services_and_artifact_limits(
                    self.workspace.clone(),
                    Arc::clone(services),
                    crate::tools::ArtifactStoreLimits::default(),
                )
            } else {
                crate::tools::ToolExecutor::new(self.workspace.clone())
            }
        } else {
            crate::tools::ToolExecutor::new(self.workspace.clone())
        };

        // Register MCP tools so child agents can access MCP servers.
        if let Some(ref mcp) = self.mcp_manager {
            let all_tools = mcp.get_all_tools().await;
            let mut by_server: std::collections::HashMap<
                String,
                Vec<crate::mcp::protocol::McpTool>,
            > = std::collections::HashMap::new();
            for (server, tool) in all_tools {
                by_server.entry(server).or_default().push(tool);
            }
            for (server_name, tools) in by_server {
                let wrappers =
                    crate::mcp::tools::create_mcp_tools(&server_name, tools, Arc::clone(mcp));
                for wrapper in wrappers {
                    child_executor.register_dynamic_tool(wrapper);
                }
            }
        }

        let child_executor = Arc::new(child_executor);

        let mut child_config = AgentConfig {
            tools: child_executor.definitions(),
            ..AgentConfig::default()
        };
        agent.apply_to(&mut child_config);
        if let Some(ref parent_ctx) = self.parent_context {
            parent_ctx.apply_to(&mut child_config);
        }
        if let Some(max_steps) = params.max_steps {
            child_config.max_tool_rounds = max_steps;
        }

        let mut tool_context =
            ToolContext::new(PathBuf::from(&self.workspace)).with_session_id(session_id.clone());
        if let Some(ref parent_ctx) = self.parent_context {
            if let Some(ref services) = parent_ctx.workspace_services {
                tool_context = tool_context.with_workspace_services(Arc::clone(services));
            }
        }

        let agent_loop = AgentLoop::new(
            Arc::clone(&self.llm_client),
            child_executor,
            tool_context,
            child_config,
        );

        // Create an mpsc channel for the child agent and forward events to broadcast
        let child_event_tx = if let Some(ref broadcast_tx) = event_tx {
            let (mpsc_tx, mut mpsc_rx) = tokio::sync::mpsc::channel(100);
            let broadcast_tx_clone = broadcast_tx.clone();

            // Spawn a task to forward events from mpsc to broadcast
            tokio::spawn(async move {
                while let Some(event) = mpsc_rx.recv().await {
                    let _ = broadcast_tx_clone.send(event);
                }
            });

            Some(mpsc_tx)
        } else {
            None
        };

        let (output, success) = match agent_loop
            .execute(&[], &params.prompt, child_event_tx)
            .await
        {
            Ok(result) => (result.text, true),
            Err(e) => (format!("Task failed: {}", e), false),
        };

        if let Some(ref tx) = event_tx {
            let _ = tx.send(AgentEvent::SubagentEnd {
                task_id: task_id.clone(),
                session_id: session_id.clone(),
                agent: params.agent.clone(),
                output: output.clone(),
                success,
            });
        }

        Ok(TaskResult {
            output,
            session_id,
            agent: params.agent,
            success,
            task_id,
        })
    }

    /// Execute a task in the background.
    ///
    /// Returns immediately with the task ID. Use events to track progress.
    pub fn execute_background(
        self: Arc<Self>,
        params: TaskParams,
        event_tx: Option<broadcast::Sender<AgentEvent>>,
    ) -> String {
        let task_id = format!("task-{}", uuid::Uuid::new_v4());
        let task_id_clone = task_id.clone();

        tokio::spawn(async move {
            if let Err(e) = self.execute(params, event_tx).await {
                tracing::error!("Background task {} failed: {}", task_id_clone, e);
            }
        });

        task_id
    }

    /// Execute multiple tasks in parallel.
    ///
    /// Spawns all tasks concurrently and waits for all to complete.
    /// Returns results in the same order as the input tasks.
    pub async fn execute_parallel(
        self: &Arc<Self>,
        tasks: Vec<TaskParams>,
        event_tx: Option<broadcast::Sender<AgentEvent>>,
    ) -> Vec<TaskResult> {
        let mut join_set: JoinSet<(usize, TaskResult)> = JoinSet::new();

        for (idx, params) in tasks.into_iter().enumerate() {
            let executor = Arc::clone(self);
            let tx = event_tx.clone();

            join_set.spawn(async move {
                let result = match executor.execute(params.clone(), tx).await {
                    Ok(result) => result,
                    Err(e) => TaskResult {
                        output: format!("Task failed: {}", e),
                        session_id: String::new(),
                        agent: params.agent,
                        success: false,
                        task_id: format!("task-{}", uuid::Uuid::new_v4()),
                    },
                };
                (idx, result)
            });
        }

        let mut indexed_results = Vec::new();
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok((idx, task_result)) => indexed_results.push((idx, task_result)),
                Err(e) => {
                    tracing::error!("Parallel task panicked: {}", e);
                    indexed_results.push((
                        usize::MAX,
                        TaskResult {
                            output: format!("Task panicked: {}", e),
                            session_id: String::new(),
                            agent: "unknown".to_string(),
                            success: false,
                            task_id: format!("task-{}", uuid::Uuid::new_v4()),
                        },
                    ));
                }
            }
        }

        indexed_results.sort_by_key(|(idx, _)| *idx);
        indexed_results.into_iter().map(|(_, r)| r).collect()
    }
}

/// Get the JSON schema for TaskParams
pub fn task_params_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "agent": {
                "type": "string",
                "description": "Required. Canonical agent type to use (for example: explore, general, plan, verification, review). Always provide this exact field name: 'agent'."
            },
            "description": {
                "type": "string",
                "description": "Required. Short task label for display and tracking. Always provide this exact field name: 'description'."
            },
            "prompt": {
                "type": "string",
                "description": "Required. Detailed instruction for the delegated child run. Always provide this exact field name: 'prompt'."
            },
            "background": {
                "type": "boolean",
                "description": "Optional. Run the task in the background. Default: false.",
                "default": false
            },
            "max_steps": {
                "type": "integer",
                "description": "Optional. Maximum number of steps for this task."
            }
        },
        "required": ["agent", "description", "prompt"],
        "examples": [
            {
                "agent": "explore",
                "description": "Find Rust files",
                "prompt": "Search the workspace for Rust files and summarize the layout."
            },
            {
                "agent": "general",
                "description": "Investigate test failure",
                "prompt": "Inspect the failing tests and explain the root cause.",
                "max_steps": 6
            }
        ]
    })
}

/// TaskTool wraps TaskExecutor as a Tool for registration in ToolExecutor.
/// This allows the LLM to delegate tasks through the standard tool interface.
pub struct TaskTool {
    executor: Arc<TaskExecutor>,
}

impl TaskTool {
    /// Create a new TaskTool
    pub fn new(executor: Arc<TaskExecutor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for TaskTool {
    fn name(&self) -> &str {
        "task"
    }

    fn description(&self) -> &str {
        "Delegate a bounded task to a specialized child run. Built-in agents: explore (read-only codebase search), general (full access multi-step), plan (read-only planning), verification (adversarial validation), review (code review). Custom agents from agent_dirs are also available."
    }

    fn parameters(&self) -> serde_json::Value {
        task_params_schema()
    }

    async fn execute(&self, args: &serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let params: TaskParams =
            serde_json::from_value(args.clone()).context("Invalid task parameters")?;

        if params.background {
            let task_id =
                Arc::clone(&self.executor).execute_background(params, ctx.agent_event_tx.clone());
            return Ok(ToolOutput::success(format!(
                "Task started in background. Task ID: {}",
                task_id
            )));
        }

        let result = self
            .executor
            .execute(params, ctx.agent_event_tx.clone())
            .await?;
        let (content, truncated) = format_task_result_for_context(&result);
        let metadata = serde_json::json!({
            "task_id": result.task_id,
            "session_id": result.session_id,
            "agent": result.agent,
            "success": result.success,
            "output_bytes": result.output.len(),
            "truncated_for_context": truncated,
            "artifact_id": task_artifact_id(&result),
            "artifact_uri": task_artifact_uri(&result),
        });

        if result.success {
            Ok(ToolOutput::success(content).with_metadata(metadata))
        } else {
            Ok(ToolOutput::error(content).with_metadata(metadata))
        }
    }
}

/// Parameters for parallel task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParallelTaskParams {
    /// List of tasks to execute concurrently
    pub tasks: Vec<TaskParams>,
}

/// Get the JSON schema for ParallelTaskParams
pub fn parallel_task_params_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "tasks": {
                "type": "array",
                "description": "List of tasks to execute in parallel. Each task runs as an independent delegated child run concurrently.",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "agent": {
                            "type": "string",
                            "description": "Required. Canonical agent type for this task."
                        },
                        "description": {
                            "type": "string",
                            "description": "Required. Short task label for display and tracking."
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Required. Detailed instruction for the delegated child run."
                        }
                    },
                    "required": ["agent", "description", "prompt"]
                },
                "minItems": 1
            }
        },
        "required": ["tasks"],
        "examples": [
            {
                "tasks": [
                    {
                        "agent": "explore",
                        "description": "Find Rust files",
                        "prompt": "List Rust files under src/."
                    },
                    {
                        "agent": "explore",
                        "description": "Find tests",
                        "prompt": "List test files and summarize their purpose."
                    }
                ]
            }
        ]
    })
}

/// ParallelTaskTool allows the LLM to fan out multiple delegated tasks concurrently.
///
/// All tasks execute in parallel and the tool returns when all complete.
pub struct ParallelTaskTool {
    executor: Arc<TaskExecutor>,
}

impl ParallelTaskTool {
    /// Create a new ParallelTaskTool
    pub fn new(executor: Arc<TaskExecutor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for ParallelTaskTool {
    fn name(&self) -> &str {
        "parallel_task"
    }

    fn description(&self) -> &str {
        "Execute multiple delegated child runs in parallel. All tasks run concurrently and results are returned when all complete. Built-in agents: explore (read-only codebase search), general (full access multi-step), plan (read-only planning), verification (adversarial validation), review (code review). Custom agents from agent_dirs are also available."
    }

    fn parameters(&self) -> serde_json::Value {
        parallel_task_params_schema()
    }

    async fn execute(&self, args: &serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let params: ParallelTaskParams =
            serde_json::from_value(args.clone()).context("Invalid parallel task parameters")?;

        if params.tasks.is_empty() {
            return Ok(ToolOutput::error("No tasks provided".to_string()));
        }

        let task_count = params.tasks.len();

        let results = self
            .executor
            .execute_parallel(params.tasks, ctx.agent_event_tx.clone())
            .await;

        // Format results with compact per-task excerpts for parent context.
        let mut output = format!("Executed {} tasks in parallel:\n\n", task_count);
        let mut metadata_results = Vec::new();
        for (i, result) in results.iter().enumerate() {
            let status = if result.success { "[OK]" } else { "[ERR]" };
            let (formatted, truncated) = format_task_result_for_context(result);
            metadata_results.push(serde_json::json!({
                "task_id": result.task_id,
                "session_id": result.session_id,
                "agent": result.agent,
                "success": result.success,
                "output_bytes": result.output.len(),
                "truncated_for_context": truncated,
                "artifact_id": task_artifact_id(result),
                "artifact_uri": task_artifact_uri(result),
            }));
            output.push_str(&format!(
                "--- Task {} ({}) {} ---\n{}\n\n",
                i + 1,
                result.agent,
                status,
                formatted
            ));
        }

        Ok(
            ToolOutput::success(output).with_metadata(serde_json::json!({
                "task_count": task_count,
                "results": metadata_results,
            })),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_params_deserialize() {
        let json = r#"{
            "agent": "explore",
            "description": "Find auth code",
            "prompt": "Search for authentication files"
        }"#;

        let params: TaskParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.agent, "explore");
        assert_eq!(params.description, "Find auth code");
        assert!(!params.background);
    }

    #[test]
    fn test_task_params_with_background() {
        let json = r#"{
            "agent": "general",
            "description": "Long task",
            "prompt": "Do something complex",
            "background": true
        }"#;

        let params: TaskParams = serde_json::from_str(json).unwrap();
        assert!(params.background);
    }

    #[test]
    fn test_task_params_with_max_steps() {
        let json = r#"{
            "agent": "plan",
            "description": "Planning task",
            "prompt": "Create a plan",
            "max_steps": 10
        }"#;

        let params: TaskParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.agent, "plan");
        assert_eq!(params.max_steps, Some(10));
        assert!(!params.background);
    }

    #[test]
    fn test_task_params_all_fields() {
        let json = r#"{
            "agent": "general",
            "description": "Complex task",
            "prompt": "Do everything",
            "background": true,
            "max_steps": 20
        }"#;

        let params: TaskParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.agent, "general");
        assert_eq!(params.description, "Complex task");
        assert_eq!(params.prompt, "Do everything");
        assert!(params.background);
        assert_eq!(params.max_steps, Some(20));
    }

    #[test]
    fn test_task_params_missing_required_field() {
        let json = r#"{
            "agent": "explore",
            "description": "Missing prompt"
        }"#;

        let result: Result<TaskParams, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_task_params_serialize() {
        let params = TaskParams {
            agent: "explore".to_string(),
            description: "Test task".to_string(),
            prompt: "Test prompt".to_string(),
            background: false,
            max_steps: Some(5),
        };

        let json = serde_json::to_string(&params).unwrap();
        assert!(json.contains("explore"));
        assert!(json.contains("Test task"));
        assert!(json.contains("Test prompt"));
    }

    #[test]
    fn test_task_params_clone() {
        let params = TaskParams {
            agent: "explore".to_string(),
            description: "Test".to_string(),
            prompt: "Prompt".to_string(),
            background: true,
            max_steps: None,
        };

        let cloned = params.clone();
        assert_eq!(params.agent, cloned.agent);
        assert_eq!(params.description, cloned.description);
        assert_eq!(params.background, cloned.background);
    }

    #[test]
    fn test_task_result_serialize() {
        let result = TaskResult {
            output: "Found 5 files".to_string(),
            session_id: "session-123".to_string(),
            agent: "explore".to_string(),
            success: true,
            task_id: "task-456".to_string(),
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("Found 5 files"));
        assert!(json.contains("explore"));
    }

    #[test]
    fn test_task_result_deserialize() {
        let json = r#"{
            "output": "Task completed",
            "session_id": "sess-789",
            "agent": "general",
            "success": false,
            "task_id": "task-123"
        }"#;

        let result: TaskResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.output, "Task completed");
        assert_eq!(result.session_id, "sess-789");
        assert_eq!(result.agent, "general");
        assert!(!result.success);
        assert_eq!(result.task_id, "task-123");
    }

    #[test]
    fn test_task_result_clone() {
        let result = TaskResult {
            output: "Output".to_string(),
            session_id: "session-1".to_string(),
            agent: "explore".to_string(),
            success: true,
            task_id: "task-1".to_string(),
        };

        let cloned = result.clone();
        assert_eq!(result.output, cloned.output);
        assert_eq!(result.success, cloned.success);
    }

    #[test]
    fn test_compact_task_output_preserves_small_output() {
        let (output, truncated) = compact_task_output("short result");
        assert_eq!(output, "short result");
        assert!(!truncated);
    }

    #[test]
    fn test_format_task_result_for_context_truncates_large_output() {
        let result = TaskResult {
            output: format!("{}TAIL", "x".repeat(TASK_OUTPUT_CONTEXT_LIMIT + 500)),
            session_id: "session-1".to_string(),
            agent: "explore".to_string(),
            success: true,
            task_id: "task-1".to_string(),
        };

        let (formatted, truncated) = format_task_result_for_context(&result);
        assert!(truncated);
        assert!(formatted.contains("Output excerpt"));
        assert!(formatted.contains("bytes omitted"));
        assert!(formatted.contains("Artifact ID: task-output:task-1"));
        assert!(formatted.contains("Artifact URI: a3s://tasks/session-1/runs/task-1/output"));
        assert!(formatted.contains("TAIL"));
        assert!(formatted.len() < result.output.len());
    }

    #[test]
    fn test_task_artifact_reference_is_stable() {
        let result = TaskResult {
            output: "done".to_string(),
            session_id: "session-1".to_string(),
            agent: "explore".to_string(),
            success: true,
            task_id: "task-1".to_string(),
        };

        assert_eq!(task_artifact_id(&result), "task-output:task-1");
        assert_eq!(
            task_artifact_uri(&result),
            "a3s://tasks/session-1/runs/task-1/output"
        );

        let (formatted, truncated) = format_task_result_for_context(&result);
        assert!(!truncated);
        assert!(formatted.contains("Artifact URI: a3s://tasks/session-1/runs/task-1/output"));
    }

    #[test]
    fn test_task_params_schema() {
        let schema = task_params_schema();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["additionalProperties"], false);
        assert!(schema["properties"]["agent"].is_object());
        assert!(schema["properties"]["prompt"].is_object());
    }

    #[test]
    fn test_task_params_schema_required_fields() {
        let schema = task_params_schema();
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("agent")));
        assert!(required.contains(&serde_json::json!("description")));
        assert!(required.contains(&serde_json::json!("prompt")));
    }

    #[test]
    fn test_task_params_schema_properties() {
        let schema = task_params_schema();
        let props = &schema["properties"];

        assert_eq!(props["agent"]["type"], "string");
        assert_eq!(props["description"]["type"], "string");
        assert_eq!(props["prompt"]["type"], "string");
        assert_eq!(props["background"]["type"], "boolean");
        assert_eq!(props["background"]["default"], false);
        assert_eq!(props["max_steps"]["type"], "integer");
    }

    #[test]
    fn test_task_params_schema_descriptions() {
        let schema = task_params_schema();
        let props = &schema["properties"];

        assert!(props["agent"]["description"].is_string());
        assert!(props["description"]["description"].is_string());
        assert!(props["prompt"]["description"].is_string());
        assert!(props["background"]["description"].is_string());
        assert!(props["max_steps"]["description"].is_string());
    }

    #[test]
    fn test_task_params_default_background() {
        let params = TaskParams {
            agent: "explore".to_string(),
            description: "Test".to_string(),
            prompt: "Test prompt".to_string(),
            background: false,
            max_steps: None,
        };
        assert!(!params.background);
    }

    #[test]
    fn test_task_params_serialize_skip_none() {
        let params = TaskParams {
            agent: "explore".to_string(),
            description: "Test".to_string(),
            prompt: "Test prompt".to_string(),
            background: false,
            max_steps: None,
        };
        let json = serde_json::to_string(&params).unwrap();
        // max_steps should not appear when None
        assert!(!json.contains("max_steps"));
    }

    #[test]
    fn test_task_params_serialize_with_max_steps() {
        let params = TaskParams {
            agent: "explore".to_string(),
            description: "Test".to_string(),
            prompt: "Test prompt".to_string(),
            background: false,
            max_steps: Some(15),
        };
        let json = serde_json::to_string(&params).unwrap();
        assert!(json.contains("max_steps"));
        assert!(json.contains("15"));
    }

    #[test]
    fn test_task_result_success_true() {
        let result = TaskResult {
            output: "Success".to_string(),
            session_id: "sess-1".to_string(),
            agent: "explore".to_string(),
            success: true,
            task_id: "task-1".to_string(),
        };
        assert!(result.success);
    }

    #[test]
    fn test_task_result_success_false() {
        let result = TaskResult {
            output: "Failed".to_string(),
            session_id: "sess-1".to_string(),
            agent: "explore".to_string(),
            success: false,
            task_id: "task-1".to_string(),
        };
        assert!(!result.success);
    }

    #[test]
    fn test_task_params_empty_strings() {
        let params = TaskParams {
            agent: "".to_string(),
            description: "".to_string(),
            prompt: "".to_string(),
            background: false,
            max_steps: None,
        };
        let json = serde_json::to_string(&params).unwrap();
        let deserialized: TaskParams = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.agent, "");
        assert_eq!(deserialized.description, "");
        assert_eq!(deserialized.prompt, "");
    }

    #[test]
    fn test_task_result_empty_output() {
        let result = TaskResult {
            output: "".to_string(),
            session_id: "sess-1".to_string(),
            agent: "explore".to_string(),
            success: true,
            task_id: "task-1".to_string(),
        };
        assert_eq!(result.output, "");
    }

    #[test]
    fn test_task_params_debug_format() {
        let params = TaskParams {
            agent: "explore".to_string(),
            description: "Test".to_string(),
            prompt: "Test prompt".to_string(),
            background: false,
            max_steps: None,
        };
        let debug_str = format!("{:?}", params);
        assert!(debug_str.contains("explore"));
        assert!(debug_str.contains("Test"));
    }

    #[test]
    fn test_task_result_debug_format() {
        let result = TaskResult {
            output: "Output".to_string(),
            session_id: "sess-1".to_string(),
            agent: "explore".to_string(),
            success: true,
            task_id: "task-1".to_string(),
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("Output"));
        assert!(debug_str.contains("explore"));
    }

    #[test]
    fn test_task_params_roundtrip() {
        let original = TaskParams {
            agent: "general".to_string(),
            description: "Roundtrip test".to_string(),
            prompt: "Test roundtrip serialization".to_string(),
            background: true,
            max_steps: Some(42),
        };
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: TaskParams = serde_json::from_str(&json).unwrap();
        assert_eq!(original.agent, deserialized.agent);
        assert_eq!(original.description, deserialized.description);
        assert_eq!(original.prompt, deserialized.prompt);
        assert_eq!(original.background, deserialized.background);
        assert_eq!(original.max_steps, deserialized.max_steps);
    }

    #[test]
    fn test_task_result_roundtrip() {
        let original = TaskResult {
            output: "Roundtrip output".to_string(),
            session_id: "sess-roundtrip".to_string(),
            agent: "plan".to_string(),
            success: false,
            task_id: "task-roundtrip".to_string(),
        };
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: TaskResult = serde_json::from_str(&json).unwrap();
        assert_eq!(original.output, deserialized.output);
        assert_eq!(original.session_id, deserialized.session_id);
        assert_eq!(original.agent, deserialized.agent);
        assert_eq!(original.success, deserialized.success);
        assert_eq!(original.task_id, deserialized.task_id);
    }

    #[test]
    fn test_parallel_task_params_deserialize() {
        let json = r#"{
            "tasks": [
                { "agent": "explore", "description": "Find auth", "prompt": "Search auth files" },
                { "agent": "general", "description": "Fix bug", "prompt": "Fix the login bug" }
            ]
        }"#;

        let params: ParallelTaskParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.tasks.len(), 2);
        assert_eq!(params.tasks[0].agent, "explore");
        assert_eq!(params.tasks[1].agent, "general");
    }

    #[test]
    fn test_parallel_task_params_single_task() {
        let json = r#"{
            "tasks": [
                { "agent": "plan", "description": "Plan work", "prompt": "Create a plan" }
            ]
        }"#;

        let params: ParallelTaskParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.tasks.len(), 1);
    }

    #[test]
    fn test_parallel_task_params_empty_tasks() {
        let json = r#"{ "tasks": [] }"#;
        let params: ParallelTaskParams = serde_json::from_str(json).unwrap();
        assert!(params.tasks.is_empty());
    }

    #[test]
    fn test_parallel_task_params_missing_tasks() {
        let json = r#"{}"#;
        let result: Result<ParallelTaskParams, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_parallel_task_params_serialize() {
        let params = ParallelTaskParams {
            tasks: vec![
                TaskParams {
                    agent: "explore".to_string(),
                    description: "Task 1".to_string(),
                    prompt: "Prompt 1".to_string(),
                    background: false,
                    max_steps: None,
                },
                TaskParams {
                    agent: "general".to_string(),
                    description: "Task 2".to_string(),
                    prompt: "Prompt 2".to_string(),
                    background: false,
                    max_steps: Some(10),
                },
            ],
        };
        let json = serde_json::to_string(&params).unwrap();
        assert!(json.contains("explore"));
        assert!(json.contains("general"));
        assert!(json.contains("Prompt 1"));
        assert!(json.contains("Prompt 2"));
    }

    #[test]
    fn test_parallel_task_params_roundtrip() {
        let original = ParallelTaskParams {
            tasks: vec![
                TaskParams {
                    agent: "explore".to_string(),
                    description: "Explore".to_string(),
                    prompt: "Find files".to_string(),
                    background: false,
                    max_steps: None,
                },
                TaskParams {
                    agent: "plan".to_string(),
                    description: "Plan".to_string(),
                    prompt: "Make plan".to_string(),
                    background: false,
                    max_steps: Some(5),
                },
            ],
        };
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: ParallelTaskParams = serde_json::from_str(&json).unwrap();
        assert_eq!(original.tasks.len(), deserialized.tasks.len());
        assert_eq!(original.tasks[0].agent, deserialized.tasks[0].agent);
        assert_eq!(original.tasks[1].agent, deserialized.tasks[1].agent);
        assert_eq!(original.tasks[1].max_steps, deserialized.tasks[1].max_steps);
    }

    #[test]
    fn test_parallel_task_params_clone() {
        let params = ParallelTaskParams {
            tasks: vec![TaskParams {
                agent: "explore".to_string(),
                description: "Test".to_string(),
                prompt: "Prompt".to_string(),
                background: false,
                max_steps: None,
            }],
        };
        let cloned = params.clone();
        assert_eq!(params.tasks.len(), cloned.tasks.len());
        assert_eq!(params.tasks[0].agent, cloned.tasks[0].agent);
    }

    #[test]
    fn test_parallel_task_params_schema() {
        let schema = parallel_task_params_schema();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["additionalProperties"], false);
        assert!(schema["properties"]["tasks"].is_object());
        assert_eq!(schema["properties"]["tasks"]["type"], "array");
        assert_eq!(schema["properties"]["tasks"]["minItems"], 1);
    }

    #[test]
    fn test_parallel_task_params_schema_required() {
        let schema = parallel_task_params_schema();
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("tasks")));
    }

    #[test]
    fn test_parallel_task_params_schema_items() {
        let schema = parallel_task_params_schema();
        let items = &schema["properties"]["tasks"]["items"];
        assert_eq!(items["type"], "object");
        assert_eq!(items["additionalProperties"], false);
        let item_required = items["required"].as_array().unwrap();
        assert!(item_required.contains(&serde_json::json!("agent")));
        assert!(item_required.contains(&serde_json::json!("description")));
        assert!(item_required.contains(&serde_json::json!("prompt")));
    }

    #[test]
    fn test_task_schema_examples_use_delegation_core() {
        let task = task_params_schema();
        let task_examples = task["examples"].as_array().unwrap();
        assert_eq!(task_examples[0]["agent"], "explore");
        assert!(task_examples[0].get("task").is_none());

        let parallel = parallel_task_params_schema();
        let parallel_examples = parallel["examples"].as_array().unwrap();
        assert!(!parallel_examples[0]["tasks"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_parallel_task_params_debug() {
        let params = ParallelTaskParams {
            tasks: vec![TaskParams {
                agent: "explore".to_string(),
                description: "Debug test".to_string(),
                prompt: "Test".to_string(),
                background: false,
                max_steps: None,
            }],
        };
        let debug_str = format!("{:?}", params);
        assert!(debug_str.contains("explore"));
        assert!(debug_str.contains("Debug test"));
    }

    #[test]
    fn test_parallel_task_params_large_count() {
        // Validate that ParallelTaskParams can hold 150 tasks without truncation
        let tasks: Vec<TaskParams> = (0..150)
            .map(|i| TaskParams {
                agent: "explore".to_string(),
                description: format!("Task {}", i),
                prompt: format!("Prompt for task {}", i),
                background: false,
                max_steps: Some(10),
            })
            .collect();

        let params = ParallelTaskParams { tasks };
        let json = serde_json::to_string(&params).unwrap();
        let deserialized: ParallelTaskParams = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.tasks.len(), 150);
        assert_eq!(deserialized.tasks[0].description, "Task 0");
        assert_eq!(deserialized.tasks[149].description, "Task 149");
    }

    #[test]
    fn test_task_params_max_steps_zero() {
        // max_steps = 0 is a valid edge case (callers decide enforcement)
        let params = TaskParams {
            agent: "explore".to_string(),
            description: "Edge case".to_string(),
            prompt: "Zero steps".to_string(),
            background: false,
            max_steps: Some(0),
        };
        let json = serde_json::to_string(&params).unwrap();
        let deserialized: TaskParams = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.max_steps, Some(0));
    }

    #[test]
    fn test_parallel_task_params_all_background() {
        let tasks: Vec<TaskParams> = (0..5)
            .map(|i| TaskParams {
                agent: "general".to_string(),
                description: format!("BG task {}", i),
                prompt: "Run in background".to_string(),
                background: true,
                max_steps: None,
            })
            .collect();
        let params = ParallelTaskParams { tasks };
        for task in &params.tasks {
            assert!(task.background);
        }
    }

    #[test]
    fn test_task_params_rejects_permissive_field() {
        let json = r#"{
            "agent": "general",
            "description": "Legacy field rejection",
            "prompt": "Verify legacy fields are rejected",
            "permissive": true
        }"#;

        let result: Result<TaskParams, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_task_params_schema_hides_permissive_field() {
        let schema = task_params_schema();
        let props = &schema["properties"];

        assert!(props.get("permissive").is_none());
    }

    // ========================================================================
    // Contract tests — verify task delegation with MockLlmClient (no network)
    // ========================================================================

    use crate::agent::tests::MockLlmClient;
    use crate::permissions::PermissionPolicy;
    use crate::subagent::AgentRegistry;

    fn test_registry_with_writer() -> Arc<AgentRegistry> {
        let registry = AgentRegistry::new();
        let spec = crate::subagent::WorkerAgentSpec::custom("writer", "Write files")
            .with_permissions(PermissionPolicy::new().allow("write(*)").allow("read(*)"))
            .with_prompt("Write files when asked.")
            .with_max_steps(3);
        registry.register(spec.into_agent_definition());
        Arc::new(registry)
    }

    #[tokio::test]
    async fn task_child_run_permission_allow() {
        let workspace = tempfile::tempdir().unwrap();
        let mock = Arc::new(MockLlmClient::new(vec![
            MockLlmClient::tool_call_response(
                "t1",
                "write",
                serde_json::json!({
                    "file_path": workspace.path().join("out.txt").to_string_lossy(),
                    "content": "WRITTEN"
                }),
            ),
            MockLlmClient::text_response("Done."),
        ]));

        let executor = TaskExecutor::new(
            test_registry_with_writer(),
            mock,
            workspace.path().to_string_lossy().to_string(),
        );

        let result = executor
            .execute(
                TaskParams {
                    agent: "writer".to_string(),
                    description: "Write file".to_string(),
                    prompt: "Write out.txt".to_string(),
                    background: false,
                    max_steps: Some(3),
                },
                None,
            )
            .await
            .unwrap();

        assert!(
            result.success,
            "child run should succeed: {}",
            result.output
        );
        assert!(
            !result.output.contains("Permission denied"),
            "no permission denial: {}",
            result.output
        );
        let content = std::fs::read_to_string(workspace.path().join("out.txt")).unwrap();
        assert_eq!(content, "WRITTEN");
    }

    #[tokio::test]
    async fn task_child_run_permission_deny() {
        let workspace = tempfile::tempdir().unwrap();
        let registry = AgentRegistry::new();
        let spec = crate::subagent::WorkerAgentSpec::custom("restricted", "Restricted agent")
            .with_permissions(PermissionPolicy::new().allow("read(*)").deny("bash(*)"))
            .with_max_steps(3);
        registry.register(spec.into_agent_definition());

        let mock = Arc::new(MockLlmClient::new(vec![
            MockLlmClient::tool_call_response(
                "t1",
                "bash",
                serde_json::json!({"command": "echo hello"}),
            ),
            MockLlmClient::text_response("Could not run bash."),
        ]));

        let executor = TaskExecutor::new(
            Arc::new(registry),
            mock,
            workspace.path().to_string_lossy().to_string(),
        );

        let result = executor
            .execute(
                TaskParams {
                    agent: "restricted".to_string(),
                    description: "Try bash".to_string(),
                    prompt: "Run echo hello".to_string(),
                    background: false,
                    max_steps: Some(3),
                },
                None,
            )
            .await
            .unwrap();

        // The agent completes (LLM responds after denial), but bash was denied.
        // The denial is sent as a tool result to the LLM, which then responds.
        assert!(result.success, "agent should complete: {}", result.output);
    }

    #[tokio::test]
    async fn task_child_run_confirmation_auto_approve() {
        let workspace = tempfile::tempdir().unwrap();
        let registry = AgentRegistry::new();
        // Agent with allow("read(*)") — write is not in allow list, so it returns Ask.
        // With AutoApproveConfirmation (default for agents with permissions), Ask → approve.
        let spec = crate::subagent::WorkerAgentSpec::custom("reader-writer", "Read and write")
            .with_permissions(PermissionPolicy::new().allow("read(*)"))
            .with_max_steps(3);
        registry.register(spec.into_agent_definition());

        let mock = Arc::new(MockLlmClient::new(vec![
            MockLlmClient::tool_call_response(
                "t1",
                "write",
                serde_json::json!({
                    "file_path": workspace.path().join("auto.txt").to_string_lossy(),
                    "content": "AUTO_APPROVED"
                }),
            ),
            MockLlmClient::text_response("Written."),
        ]));

        let executor = TaskExecutor::new(
            Arc::new(registry),
            mock,
            workspace.path().to_string_lossy().to_string(),
        );

        let result = executor
            .execute(
                TaskParams {
                    agent: "reader-writer".to_string(),
                    description: "Write via auto-approve".to_string(),
                    prompt: "Write auto.txt".to_string(),
                    background: false,
                    max_steps: Some(3),
                },
                None,
            )
            .await
            .unwrap();

        assert!(
            result.success,
            "Ask should be auto-approved: {}",
            result.output
        );
        assert!(
            !result.output.contains("MissingConfirmationManager"),
            "no MissingConfirmationManager: {}",
            result.output
        );
    }

    #[tokio::test]
    async fn task_child_run_step_budget_enforced() {
        let workspace = tempfile::tempdir().unwrap();
        let mock = Arc::new(MockLlmClient::new(vec![
            MockLlmClient::tool_call_response(
                "t1",
                "read",
                serde_json::json!({"file_path": "/tmp/a.txt"}),
            ),
            MockLlmClient::tool_call_response(
                "t2",
                "read",
                serde_json::json!({"file_path": "/tmp/b.txt"}),
            ),
            MockLlmClient::tool_call_response(
                "t3",
                "read",
                serde_json::json!({"file_path": "/tmp/c.txt"}),
            ),
            MockLlmClient::text_response("Should not reach here."),
        ]));

        let executor = TaskExecutor::new(
            test_registry_with_writer(),
            mock,
            workspace.path().to_string_lossy().to_string(),
        );

        let result = executor
            .execute(
                TaskParams {
                    agent: "writer".to_string(),
                    description: "Exceed budget".to_string(),
                    prompt: "Read many files".to_string(),
                    background: false,
                    max_steps: Some(2),
                },
                None,
            )
            .await
            .unwrap();

        // The agent should fail after exceeding 2 tool rounds
        assert!(
            !result.success,
            "should fail when exceeding step budget: {}",
            result.output
        );
        assert!(
            result.output.contains("Max tool rounds") || result.output.contains("max tool rounds"),
            "error should mention tool rounds: {}",
            result.output
        );
    }

    #[tokio::test]
    async fn parallel_task_both_inherit_permissions() {
        let workspace = tempfile::tempdir().unwrap();
        let mock = Arc::new(MockLlmClient::new(vec![
            // Task 1 responses
            MockLlmClient::tool_call_response(
                "t1",
                "write",
                serde_json::json!({
                    "file_path": workspace.path().join("p1.txt").to_string_lossy(),
                    "content": "P1"
                }),
            ),
            MockLlmClient::text_response("Done 1."),
            // Task 2 responses
            MockLlmClient::tool_call_response(
                "t2",
                "write",
                serde_json::json!({
                    "file_path": workspace.path().join("p2.txt").to_string_lossy(),
                    "content": "P2"
                }),
            ),
            MockLlmClient::text_response("Done 2."),
        ]));

        let executor = Arc::new(TaskExecutor::new(
            test_registry_with_writer(),
            mock,
            workspace.path().to_string_lossy().to_string(),
        ));

        let tasks = vec![
            TaskParams {
                agent: "writer".to_string(),
                description: "Write p1".to_string(),
                prompt: "Write p1.txt".to_string(),
                background: false,
                max_steps: Some(3),
            },
            TaskParams {
                agent: "writer".to_string(),
                description: "Write p2".to_string(),
                prompt: "Write p2.txt".to_string(),
                background: false,
                max_steps: Some(3),
            },
        ];

        let results = executor.execute_parallel(tasks, None).await;
        assert_eq!(results.len(), 2);

        for result in &results {
            assert!(
                result.success,
                "parallel child should succeed: {}",
                result.output
            );
        }
    }
}
