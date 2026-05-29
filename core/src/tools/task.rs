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
use crate::orchestration::{AgentExecutor, AgentStepSpec, StepOutcome};
use crate::subagent::AgentRegistry;
use crate::tools::types::{Tool, ToolContext, ToolOutput};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::broadcast;

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

/// Translate selected child-loop events into a `SubagentProgress` milestone
/// for the parent broadcast. Returns `None` for events that aren't worth
/// surfacing as progress (text deltas, tool starts, subagent events from
/// nested delegation, etc.).
///
/// Currently emits progress for:
/// - `ToolEnd`   → `status = "tool_completed"`,
///   `metadata = { tool, exit_code, output_bytes, error_kind? }`
/// - `TurnEnd`   → `status = "turn_completed"`,
///   `metadata = { turn, total_tokens, prompt_tokens, completion_tokens }`
fn synthesize_subagent_progress(
    event: &AgentEvent,
    task_id: &str,
    session_id: &str,
) -> Option<AgentEvent> {
    match event {
        AgentEvent::ToolEnd {
            name,
            output,
            exit_code,
            error_kind,
            ..
        } => {
            let mut metadata = serde_json::json!({
                "tool": name,
                "exit_code": exit_code,
                "output_bytes": output.len(),
            });
            if let Some(kind) = error_kind {
                metadata["error_kind"] =
                    serde_json::to_value(kind).unwrap_or(serde_json::Value::Null);
            }
            Some(AgentEvent::SubagentProgress {
                task_id: task_id.to_string(),
                session_id: session_id.to_string(),
                status: "tool_completed".to_string(),
                metadata,
            })
        }
        AgentEvent::TurnEnd { turn, usage } => Some(AgentEvent::SubagentProgress {
            task_id: task_id.to_string(),
            session_id: session_id.to_string(),
            status: "turn_completed".to_string(),
            metadata: serde_json::json!({
                "turn": turn,
                "total_tokens": usage.total_tokens,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            }),
        }),
        _ => None,
    }
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
    max_parallel_tasks: usize,
    /// Optional shared tracker — when present each task registers a
    /// `CancellationToken` so callers can cancel by `task_id`.
    subagent_tracker: Option<Arc<crate::subagent_task_tracker::InMemorySubagentTaskTracker>>,
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
            max_parallel_tasks: crate::agent::DEFAULT_MAX_PARALLEL_TASKS,
            subagent_tracker: None,
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
            max_parallel_tasks: crate::agent::DEFAULT_MAX_PARALLEL_TASKS,
            subagent_tracker: None,
        }
    }

    /// Set parent session capabilities to inherit into child runs.
    pub fn with_parent_context(mut self, ctx: crate::child_run::ChildRunContext) -> Self {
        if let Some(max_parallel_tasks) = ctx.max_parallel_tasks {
            self.max_parallel_tasks = max_parallel_tasks.max(1);
        }
        self.parent_context = Some(ctx);
        self
    }

    pub fn with_max_parallel_tasks(mut self, max_parallel_tasks: usize) -> Self {
        self.max_parallel_tasks = max_parallel_tasks.max(1);
        self
    }

    /// Share a tracker with this executor. When set, each task registers
    /// a `CancellationToken` against the tracker so the parent session
    /// can cancel by `task_id`.
    pub fn with_subagent_tracker(
        mut self,
        tracker: Arc<crate::subagent_task_tracker::InMemorySubagentTaskTracker>,
    ) -> Self {
        self.subagent_tracker = Some(tracker);
        self
    }

    /// Execute a task by spawning an isolated child AgentLoop.
    ///
    /// `parent_session_id` flows into the emitted `SubagentStart`/`SubagentEnd`
    /// events so dashboards can associate child runs with the parent session.
    pub async fn execute(
        &self,
        params: TaskParams,
        event_tx: Option<broadcast::Sender<AgentEvent>>,
        parent_session_id: Option<&str>,
    ) -> Result<TaskResult> {
        let task_id = format!("task-{}", uuid::Uuid::new_v4());
        self.execute_with_task_id(task_id, params, event_tx, parent_session_id, true)
            .await
    }

    /// Execute a task using a caller-supplied task id. Used by `execute_background`
    /// so the synchronously-returned task id matches the one in lifecycle events.
    /// When `emit_start` is `false` the caller is responsible for emitting
    /// `SubagentStart` themselves (e.g. to avoid a race against a tracker query).
    pub async fn execute_with_task_id(
        &self,
        task_id: String,
        params: TaskParams,
        event_tx: Option<broadcast::Sender<AgentEvent>>,
        parent_session_id: Option<&str>,
        emit_start: bool,
    ) -> Result<TaskResult> {
        let session_id = format!("task-run-{}", task_id);

        let agent = self
            .registry
            .get(&params.agent)
            .context(format!("Unknown agent type: '{}'", params.agent))?;

        if emit_start {
            if let Some(ref tx) = event_tx {
                let _ = tx.send(AgentEvent::SubagentStart {
                    task_id: task_id.clone(),
                    session_id: session_id.clone(),
                    parent_session_id: parent_session_id.unwrap_or_default().to_string(),
                    agent: params.agent.clone(),
                    description: params.description.clone(),
                });
            }
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

        // Create an mpsc channel for the child agent and forward events to broadcast.
        // Selected child events (ToolEnd, TurnEnd) are also surfaced to the parent
        // broadcast as synthetic `SubagentProgress` events so dashboards can observe
        // mid-task milestones without subscribing to the raw event stream.
        let child_event_tx = if let Some(ref broadcast_tx) = event_tx {
            let (mpsc_tx, mut mpsc_rx) = tokio::sync::mpsc::channel(100);
            let broadcast_tx_clone = broadcast_tx.clone();
            let progress_task_id = task_id.clone();
            let progress_session_id = session_id.clone();

            tokio::spawn(async move {
                while let Some(event) = mpsc_rx.recv().await {
                    if let Some(progress) = synthesize_subagent_progress(
                        &event,
                        &progress_task_id,
                        &progress_session_id,
                    ) {
                        let _ = broadcast_tx_clone.send(progress);
                    }
                    let _ = broadcast_tx_clone.send(event);
                }
            });

            Some(mpsc_tx)
        } else {
            None
        };

        // Register a CancellationToken with the tracker (if shared) so the
        // parent session's `cancel_subagent_task` can interrupt this run.
        let cancel_token = tokio_util::sync::CancellationToken::new();
        if let Some(ref tracker) = self.subagent_tracker {
            tracker
                .register_canceller(&task_id, cancel_token.clone())
                .await;
        }

        let (output, success) = match agent_loop
            .execute_with_session(
                &[],
                &params.prompt,
                Some(&session_id),
                child_event_tx,
                Some(&cancel_token),
            )
            .await
        {
            Ok(result) => (result.text, true),
            Err(e) if cancel_token.is_cancelled() => {
                (format!("Task cancelled by caller: {}", e), false)
            }
            Err(e) => (format!("Task failed: {}", e), false),
        };

        if let Some(ref tracker) = self.subagent_tracker {
            tracker.clear_canceller(&task_id).await;
        }

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
    /// Returns immediately with the task ID; the same id is used in the emitted
    /// `SubagentStart`/`SubagentEnd` events so callers can correlate. Pre-emits
    /// `SubagentStart` synchronously when an event channel is available so a
    /// caller that queries the subagent task tracker right after this call
    /// observes the task in `Running` state without a race window.
    pub fn execute_background(
        self: Arc<Self>,
        params: TaskParams,
        event_tx: Option<broadcast::Sender<AgentEvent>>,
        parent_session_id: Option<String>,
    ) -> String {
        let task_id = format!("task-{}", uuid::Uuid::new_v4());
        let session_id = format!("task-run-{}", task_id);

        if let Some(ref tx) = event_tx {
            let _ = tx.send(AgentEvent::SubagentStart {
                task_id: task_id.clone(),
                session_id,
                parent_session_id: parent_session_id.clone().unwrap_or_default(),
                agent: params.agent.clone(),
                description: params.description.clone(),
            });
        }

        let task_id_for_spawn = task_id.clone();
        let task_id_for_log = task_id.clone();
        tokio::spawn(async move {
            if let Err(e) = self
                .execute_with_task_id(
                    task_id_for_spawn,
                    params,
                    event_tx,
                    parent_session_id.as_deref(),
                    false,
                )
                .await
            {
                tracing::error!("Background task {} failed: {}", task_id_for_log, e);
            }
        });

        task_id
    }

    /// Execute multiple tasks in parallel.
    ///
    /// Spawns all tasks concurrently and waits for all to complete.
    /// Returns results in the same order as the input tasks. Routed through
    /// the [`AgentExecutor`](crate::orchestration::AgentExecutor) seam so the
    /// same fan-out works whether steps run locally (default) or are placed
    /// on remote nodes by a host.
    pub async fn execute_parallel(
        self: &Arc<Self>,
        tasks: Vec<TaskParams>,
        event_tx: Option<broadcast::Sender<AgentEvent>>,
        parent_session_id: Option<&str>,
    ) -> Vec<TaskResult> {
        let parent = parent_session_id.map(|s| s.to_string());
        let specs = tasks
            .into_iter()
            .map(|params| AgentStepSpec {
                task_id: format!("task-{}", uuid::Uuid::new_v4()),
                agent: params.agent,
                description: params.description,
                prompt: params.prompt,
                max_steps: params.max_steps,
                parent_session_id: parent.clone(),
            })
            .collect();

        let executor: Arc<dyn AgentExecutor> = Arc::<Self>::clone(self);
        crate::orchestration::execute_steps_parallel(executor, specs, event_tx)
            .await
            .into_iter()
            .map(TaskResult::from)
            .collect()
    }
}

impl From<TaskResult> for StepOutcome {
    fn from(r: TaskResult) -> Self {
        StepOutcome {
            task_id: r.task_id,
            session_id: r.session_id,
            agent: r.agent,
            output: r.output,
            success: r.success,
        }
    }
}

impl From<StepOutcome> for TaskResult {
    fn from(o: StepOutcome) -> Self {
        TaskResult {
            output: o.output,
            session_id: o.session_id,
            agent: o.agent,
            success: o.success,
            task_id: o.task_id,
        }
    }
}

/// The local, in-process executor: every step runs as a child `AgentLoop` on
/// this node's tokio runtime. This is the default; a host (书安OS) substitutes
/// its own [`AgentExecutor`] to place steps across a cluster.
#[async_trait]
impl AgentExecutor for TaskExecutor {
    async fn execute_step(
        &self,
        spec: AgentStepSpec,
        event_tx: Option<broadcast::Sender<AgentEvent>>,
    ) -> StepOutcome {
        let agent = spec.agent.clone();
        let task_id = spec.task_id.clone();
        let params = TaskParams {
            agent: spec.agent,
            description: spec.description,
            prompt: spec.prompt,
            background: false,
            max_steps: spec.max_steps,
        };
        match self
            .execute_with_task_id(
                task_id.clone(),
                params,
                event_tx,
                spec.parent_session_id.as_deref(),
                true,
            )
            .await
        {
            Ok(result) => result.into(),
            Err(e) => StepOutcome::failed(task_id, agent, format!("Task failed: {e}")),
        }
    }

    fn concurrency_hint(&self) -> usize {
        self.max_parallel_tasks
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
        "Delegate a bounded task to a specialized child run. Built-in agents: explore (read-only codebase search), general/general-purpose (full access multi-step), plan (read-only planning), verification (adversarial validation), review (code review). Custom agents from agent_dirs and .a3s/agents are also available; .claude/agents is read for compatibility."
    }

    fn parameters(&self) -> serde_json::Value {
        task_params_schema()
    }

    async fn execute(&self, args: &serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let params: TaskParams =
            serde_json::from_value(args.clone()).context("Invalid task parameters")?;

        if params.background {
            let task_id = Arc::clone(&self.executor).execute_background(
                params,
                ctx.agent_event_tx.clone(),
                ctx.session_id.clone(),
            );
            return Ok(ToolOutput::success(format!(
                "Task started in background. Task ID: {}",
                task_id
            )));
        }

        let result = self
            .executor
            .execute(
                params,
                ctx.agent_event_tx.clone(),
                ctx.session_id.as_deref(),
            )
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
        "Execute multiple delegated child runs in parallel. All tasks run concurrently and results are returned when all complete. Built-in agents: explore (read-only codebase search), general/general-purpose (full access multi-step), plan (read-only planning), verification (adversarial validation), review (code review). Custom agents from agent_dirs and .a3s/agents are also available; .claude/agents is read for compatibility."
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
            .execute_parallel(
                params.tasks,
                ctx.agent_event_tx.clone(),
                ctx.session_id.as_deref(),
            )
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
                "output": formatted.clone(),
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

        let all_success = results.iter().all(|result| result.success);
        let output = if all_success {
            ToolOutput::success(output)
        } else {
            ToolOutput::error(output)
        };

        Ok(output.with_metadata(serde_json::json!({
            "task_count": task_count,
            "results": metadata_results,
        })))
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
    use crate::llm::{ContentBlock, LlmResponse, Message, StreamEvent, TokenUsage, ToolDefinition};
    use crate::permissions::PermissionPolicy;
    use crate::subagent::AgentRegistry;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tokio::sync::{mpsc, Barrier};

    fn text_response(text: impl Into<String>) -> LlmResponse {
        LlmResponse {
            message: Message {
                role: "assistant".to_string(),
                content: vec![ContentBlock::Text { text: text.into() }],
                reasoning_content: None,
            },
            usage: TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                cache_read_tokens: None,
                cache_write_tokens: None,
            },
            stop_reason: Some("end_turn".to_string()),
            meta: None,
        }
    }

    fn pre_analysis_response(messages: &[Message]) -> LlmResponse {
        let prompt = last_text(messages);
        let response = serde_json::json!({
            "intent": "GeneralPurpose",
            "requires_planning": false,
            "goal": {
                "description": prompt,
                "success_criteria": []
            },
            "execution_plan": {
                "complexity": "Simple",
                "steps": [{
                    "id": "step-1",
                    "description": prompt,
                    "tool": null,
                    "dependencies": [],
                    "success_criteria": "Complete the request"
                }],
                "required_tools": []
            },
            "optimized_input": prompt
        });
        text_response(response.to_string())
    }

    fn last_text(messages: &[Message]) -> String {
        messages
            .last()
            .and_then(|message| {
                message.content.iter().find_map(|block| {
                    if let ContentBlock::Text { text } = block {
                        Some(text.clone())
                    } else {
                        None
                    }
                })
            })
            .unwrap_or_default()
    }

    struct StaticLlmClient {
        text: String,
    }

    impl StaticLlmClient {
        fn new(text: impl Into<String>) -> Self {
            Self { text: text.into() }
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for StaticLlmClient {
        async fn complete(
            &self,
            messages: &[Message],
            system: Option<&str>,
            _tools: &[ToolDefinition],
        ) -> Result<LlmResponse> {
            if system == Some(crate::prompts::PRE_ANALYSIS_SYSTEM) {
                return Ok(pre_analysis_response(messages));
            }
            Ok(text_response(self.text.clone()))
        }

        async fn complete_streaming(
            &self,
            _messages: &[Message],
            _system: Option<&str>,
            _tools: &[ToolDefinition],
            _cancel_token: tokio_util::sync::CancellationToken,
        ) -> Result<mpsc::Receiver<StreamEvent>> {
            anyhow::bail!("streaming is not used by task executor tests")
        }
    }

    struct ConcurrentLlmClient {
        barrier: Arc<Barrier>,
        active: AtomicUsize,
        max_active: AtomicUsize,
    }

    impl ConcurrentLlmClient {
        fn new(task_count: usize) -> Self {
            Self {
                barrier: Arc::new(Barrier::new(task_count)),
                active: AtomicUsize::new(0),
                max_active: AtomicUsize::new(0),
            }
        }

        fn max_active(&self) -> usize {
            self.max_active.load(Ordering::SeqCst)
        }

        fn record_active(&self) {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            let mut observed = self.max_active.load(Ordering::SeqCst);
            while active > observed {
                match self.max_active.compare_exchange(
                    observed,
                    active,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(next) => observed = next,
                }
            }
        }
    }

    struct LimitedConcurrencyLlmClient {
        active: AtomicUsize,
        max_active: AtomicUsize,
    }

    impl LimitedConcurrencyLlmClient {
        fn new() -> Self {
            Self {
                active: AtomicUsize::new(0),
                max_active: AtomicUsize::new(0),
            }
        }

        fn max_active(&self) -> usize {
            self.max_active.load(Ordering::SeqCst)
        }

        fn record_active(&self) {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(active, Ordering::SeqCst);
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for LimitedConcurrencyLlmClient {
        async fn complete(
            &self,
            messages: &[Message],
            system: Option<&str>,
            _tools: &[ToolDefinition],
        ) -> Result<LlmResponse> {
            if system == Some(crate::prompts::PRE_ANALYSIS_SYSTEM) {
                return Ok(pre_analysis_response(messages));
            }

            let prompt = last_text(messages);
            self.record_active();
            tokio::time::sleep(Duration::from_millis(40)).await;
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(text_response(format!("completed: {prompt}")))
        }

        async fn complete_streaming(
            &self,
            _messages: &[Message],
            _system: Option<&str>,
            _tools: &[ToolDefinition],
            _cancel_token: tokio_util::sync::CancellationToken,
        ) -> Result<mpsc::Receiver<StreamEvent>> {
            anyhow::bail!("streaming is not used by task executor tests")
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for ConcurrentLlmClient {
        async fn complete(
            &self,
            messages: &[Message],
            system: Option<&str>,
            _tools: &[ToolDefinition],
        ) -> Result<LlmResponse> {
            if system == Some(crate::prompts::PRE_ANALYSIS_SYSTEM) {
                return Ok(pre_analysis_response(messages));
            }

            let prompt = last_text(messages);
            self.record_active();
            self.barrier.wait().await;
            if prompt.contains("slow") {
                tokio::time::sleep(Duration::from_millis(120)).await;
            } else {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(text_response(format!("completed: {prompt}")))
        }

        async fn complete_streaming(
            &self,
            _messages: &[Message],
            _system: Option<&str>,
            _tools: &[ToolDefinition],
            _cancel_token: tokio_util::sync::CancellationToken,
        ) -> Result<mpsc::Receiver<StreamEvent>> {
            anyhow::bail!("streaming is not used by task executor tests")
        }
    }

    fn test_registry_with_writer() -> Arc<AgentRegistry> {
        let registry = AgentRegistry::new();
        let spec = crate::subagent::WorkerAgentSpec::custom("writer", "Write files")
            .with_permissions(PermissionPolicy::new().allow("write(*)").allow("read(*)"))
            .with_prompt("Write files when asked.")
            .with_max_steps(3);
        registry.register(spec.into_agent_definition());
        Arc::new(registry)
    }

    fn test_registry_with_text_worker() -> Arc<AgentRegistry> {
        let registry = AgentRegistry::new();
        let spec = crate::subagent::WorkerAgentSpec::custom("worker", "Text worker")
            .with_prompt("Return a concise result.")
            .with_max_steps(1);
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
    async fn parallel_task_executor_runs_children_concurrently_and_preserves_input_order() {
        let workspace = tempfile::tempdir().unwrap();
        let client = Arc::new(ConcurrentLlmClient::new(2));
        let executor = Arc::new(TaskExecutor::new(
            test_registry_with_text_worker(),
            client.clone(),
            workspace.path().to_string_lossy().to_string(),
        ));

        let tasks = vec![
            TaskParams {
                agent: "worker".to_string(),
                description: "Slow task".to_string(),
                prompt: "slow branch".to_string(),
                background: false,
                max_steps: Some(1),
            },
            TaskParams {
                agent: "worker".to_string(),
                description: "Fast task".to_string(),
                prompt: "fast branch".to_string(),
                background: false,
                max_steps: Some(1),
            },
        ];

        let results = tokio::time::timeout(
            Duration::from_secs(2),
            executor.execute_parallel(tasks, None, None),
        )
        .await
        .expect("parallel children should reach the barrier and complete");

        assert_eq!(results.len(), 2);
        assert!(
            client.max_active() >= 2,
            "expected concurrent child execution, max_active={}",
            client.max_active()
        );
        assert!(results[0].success);
        assert!(results[0].output.contains("slow branch"));
        assert!(results[1].success);
        assert!(results[1].output.contains("fast branch"));
    }

    #[tokio::test]
    async fn parallel_task_executor_respects_configured_concurrency_limit() {
        let workspace = tempfile::tempdir().unwrap();
        let client = Arc::new(LimitedConcurrencyLlmClient::new());
        let executor = Arc::new(
            TaskExecutor::new(
                test_registry_with_text_worker(),
                client.clone(),
                workspace.path().to_string_lossy().to_string(),
            )
            .with_max_parallel_tasks(2),
        );

        let tasks = (0..5)
            .map(|idx| TaskParams {
                agent: "worker".to_string(),
                description: format!("Task {idx}"),
                prompt: format!("branch {idx}"),
                background: false,
                max_steps: Some(1),
            })
            .collect::<Vec<_>>();

        let results = executor.execute_parallel(tasks, None, None).await;

        assert_eq!(results.len(), 5);
        assert!(results.iter().all(|result| result.success));
        assert_eq!(client.max_active(), 2);
    }

    #[tokio::test]
    async fn parallel_task_executor_isolates_unknown_agent_failure() {
        let workspace = tempfile::tempdir().unwrap();
        let executor = Arc::new(TaskExecutor::new(
            test_registry_with_text_worker(),
            Arc::new(StaticLlmClient::new("valid branch done")),
            workspace.path().to_string_lossy().to_string(),
        ));

        let tasks = vec![
            TaskParams {
                agent: "missing-agent".to_string(),
                description: "Missing".to_string(),
                prompt: "should fail".to_string(),
                background: false,
                max_steps: Some(1),
            },
            TaskParams {
                agent: "worker".to_string(),
                description: "Valid".to_string(),
                prompt: "should succeed".to_string(),
                background: false,
                max_steps: Some(1),
            },
        ];

        let results = executor.execute_parallel(tasks, None, None).await;

        assert_eq!(results.len(), 2);
        assert!(!results[0].success);
        assert_eq!(results[0].agent, "missing-agent");
        assert!(results[0].output.contains("Unknown agent type"));
        assert!(results[1].success);
        assert_eq!(results[1].agent, "worker");
        assert!(results[1].output.contains("valid branch done"));
    }

    #[tokio::test]
    async fn parallel_task_executor_emits_subagent_events_for_each_child() {
        let workspace = tempfile::tempdir().unwrap();
        let executor = Arc::new(TaskExecutor::new(
            test_registry_with_text_worker(),
            Arc::new(StaticLlmClient::new("done")),
            workspace.path().to_string_lossy().to_string(),
        ));
        let (tx, mut rx) = broadcast::channel(64);

        let tasks = vec![
            TaskParams {
                agent: "worker".to_string(),
                description: "One".to_string(),
                prompt: "first".to_string(),
                background: false,
                max_steps: Some(1),
            },
            TaskParams {
                agent: "worker".to_string(),
                description: "Two".to_string(),
                prompt: "second".to_string(),
                background: false,
                max_steps: Some(1),
            },
        ];

        let results = executor.execute_parallel(tasks, Some(tx), None).await;
        assert_eq!(results.len(), 2);
        tokio::time::sleep(Duration::from_millis(20)).await;

        let mut starts = Vec::new();
        let mut ends = Vec::new();
        let mut progress_statuses: Vec<String> = Vec::new();
        while let Ok(event) = rx.try_recv() {
            match event {
                AgentEvent::SubagentStart { description, .. } => starts.push(description),
                AgentEvent::SubagentEnd { agent, success, .. } => ends.push((agent, success)),
                AgentEvent::SubagentProgress { status, .. } => progress_statuses.push(status),
                _ => {}
            }
        }

        starts.sort();
        assert_eq!(starts, vec!["One".to_string(), "Two".to_string()]);
        assert_eq!(ends.len(), 2);
        assert!(ends
            .iter()
            .all(|(agent, success)| agent == "worker" && *success));
        // Each child loop emits at least one TurnEnd, so we expect at least
        // two synthesized turn_completed progress events across the run.
        assert!(
            progress_statuses
                .iter()
                .filter(|s| s == &"turn_completed")
                .count()
                >= 2,
            "expected at least two turn_completed progress events, got {:?}",
            progress_statuses
        );
    }

    #[tokio::test]
    async fn parallel_task_tool_reports_error_when_any_child_fails() {
        let workspace = tempfile::tempdir().unwrap();
        let executor = Arc::new(TaskExecutor::new(
            test_registry_with_text_worker(),
            Arc::new(StaticLlmClient::new("valid branch done")),
            workspace.path().to_string_lossy().to_string(),
        ));
        let tool = ParallelTaskTool::new(executor);
        let ctx = ToolContext::new(workspace.path().to_path_buf());

        let output = tool
            .execute(
                &serde_json::json!({
                    "tasks": [
                        {
                            "agent": "missing-agent",
                            "description": "Missing",
                            "prompt": "should fail"
                        },
                        {
                            "agent": "worker",
                            "description": "Valid",
                            "prompt": "should succeed"
                        }
                    ]
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(
            !output.success,
            "parallel_task should fail when any child result fails"
        );
        assert!(output.content.contains("[ERR]"));
        assert!(output.content.contains("[OK]"));
        let metadata = output.metadata.expect("metadata");
        assert_eq!(metadata["task_count"], 2);
        assert_eq!(metadata["results"][0]["success"], false);
        assert_eq!(metadata["results"][1]["success"], true);
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

        let results = executor.execute_parallel(tasks, None, None).await;
        assert_eq!(results.len(), 2);

        for result in &results {
            assert!(
                result.success,
                "parallel child should succeed: {}",
                result.output
            );
        }
    }

    #[test]
    fn synthesize_progress_emits_tool_completed_for_tool_end() {
        let event = AgentEvent::ToolEnd {
            id: "call-1".to_string(),
            name: "bash".to_string(),
            output: "hello".to_string(),
            exit_code: 0,
            metadata: None,
            error_kind: None,
        };
        let progress =
            synthesize_subagent_progress(&event, "task-1", "task-run-task-1").expect("some");
        match progress {
            AgentEvent::SubagentProgress {
                task_id,
                session_id,
                status,
                metadata,
            } => {
                assert_eq!(task_id, "task-1");
                assert_eq!(session_id, "task-run-task-1");
                assert_eq!(status, "tool_completed");
                assert_eq!(metadata["tool"], "bash");
                assert_eq!(metadata["exit_code"], 0);
                assert_eq!(metadata["output_bytes"], 5);
                assert!(metadata.get("error_kind").is_none());
            }
            other => panic!("expected SubagentProgress, got {:?}", other),
        }
    }

    #[test]
    fn synthesize_progress_includes_error_kind_when_present() {
        let event = AgentEvent::ToolEnd {
            id: "call-2".to_string(),
            name: "edit".to_string(),
            output: "boom".to_string(),
            exit_code: 1,
            metadata: None,
            error_kind: Some(crate::tools::ToolErrorKind::NotFound {
                path: "missing.txt".to_string(),
            }),
        };
        let progress =
            synthesize_subagent_progress(&event, "task-x", "task-run-task-x").expect("some");
        if let AgentEvent::SubagentProgress { metadata, .. } = progress {
            assert!(
                metadata.get("error_kind").is_some(),
                "error_kind should propagate into metadata"
            );
        } else {
            panic!("expected SubagentProgress");
        }
    }

    #[test]
    fn synthesize_progress_emits_turn_completed_for_turn_end() {
        let event = AgentEvent::TurnEnd {
            turn: 3,
            usage: crate::llm::TokenUsage {
                prompt_tokens: 100,
                completion_tokens: 25,
                total_tokens: 125,
                cache_read_tokens: None,
                cache_write_tokens: None,
            },
        };
        let progress =
            synthesize_subagent_progress(&event, "task-1", "task-run-task-1").expect("some");
        if let AgentEvent::SubagentProgress {
            status, metadata, ..
        } = progress
        {
            assert_eq!(status, "turn_completed");
            assert_eq!(metadata["turn"], 3);
            assert_eq!(metadata["total_tokens"], 125);
            assert_eq!(metadata["prompt_tokens"], 100);
            assert_eq!(metadata["completion_tokens"], 25);
        } else {
            panic!("expected SubagentProgress");
        }
    }

    #[test]
    fn synthesize_progress_ignores_unrelated_events() {
        let ignored = [
            AgentEvent::TextDelta {
                text: "hi".to_string(),
            },
            AgentEvent::ToolStart {
                id: "x".to_string(),
                name: "bash".to_string(),
            },
            AgentEvent::TurnStart { turn: 1 },
            AgentEvent::SubagentStart {
                task_id: "nested".to_string(),
                session_id: "nested-run".to_string(),
                parent_session_id: "parent".to_string(),
                agent: "explore".to_string(),
                description: "nested".to_string(),
            },
        ];
        for event in &ignored {
            assert!(
                synthesize_subagent_progress(event, "task", "session").is_none(),
                "{:?} should not emit progress",
                event
            );
        }
    }
}
