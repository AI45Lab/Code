//! Agent Loop Implementation
//!
//! The agent loop handles the core conversation cycle:
//! 1. User sends a prompt
//! 2. LLM generates a response (possibly with tool calls)
//! 3. If tool calls present, execute them and send results back
//! 4. Repeat until LLM returns without tool calls
//!
//! This implements agentic behavior where the LLM can use tools
//! to accomplish tasks agentically.

use crate::context::ContextProvider;
use crate::hitl::ConfirmationProvider;
use crate::hooks::HookExecutor;
#[cfg(test)]
use crate::llm::LlmResponse;
use crate::llm::{LlmClient, Message, TokenUsage, ToolDefinition};
use crate::permissions::{PermissionChecker, PermissionPolicy};
use crate::planning::{AgentGoal, ExecutionPlan, TaskStatus};
use crate::prompts::{PlanningMode, SystemPromptSlots};
use crate::queue::{SessionCommand, SessionQueueConfig};
use crate::session_lane_queue::SessionLaneQueue;
use crate::subagent::AgentRegistry;
use crate::tools::{ToolContext, ToolExecutor};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

mod auto_delegation;
mod completion_runtime;
mod context_perception;
mod execution_entry;
mod execution_mode;
mod execution_state;
pub(crate) use execution_state::ExecutionSeed;
mod hook_runtime;
mod llm_turn;
mod loop_builder;
mod loop_runtime;
mod parallel_tool_runtime;
mod plan_execution;
mod planning_runtime;
mod project_context;
mod prompt_runtime;
mod queue_forwarder;
mod telemetry_runtime;
mod tool_completion_runtime;
mod tool_execution_runtime;
mod tool_gate_runtime;
mod tool_guard_runtime;
mod tool_memory_runtime;
mod tool_result_runtime;
mod tool_turn;
mod turn_context;

/// Maximum number of tool execution rounds before stopping
pub(crate) const MAX_TOOL_ROUNDS: usize = 50;
pub(crate) const DEFAULT_MAX_PARALLEL_TASKS: usize = 8;

/// Internal agent loop configuration.
#[derive(Clone)]
pub(crate) struct AgentConfig {
    /// Slot-based system prompt customization.
    ///
    /// Users can customize specific parts (role, guidelines, response style, extra)
    /// without overriding the core agentic capabilities. The default agentic core
    /// (tool usage, autonomous behavior, completion criteria) is always preserved.
    pub prompt_slots: SystemPromptSlots,
    pub tools: Vec<ToolDefinition>,
    pub max_tool_rounds: usize,
    /// Optional security provider for input taint tracking and output sanitization
    pub security_provider: Option<Arc<dyn crate::security::SecurityProvider>>,
    /// Optional permission checker for tool execution control
    pub permission_checker: Option<Arc<dyn PermissionChecker>>,
    /// Serializable permission policy used to build the checker, when available.
    pub permission_policy: Option<PermissionPolicy>,
    /// Optional confirmation manager for HITL (Human-in-the-Loop)
    pub confirmation_manager: Option<Arc<dyn ConfirmationProvider>>,
    /// Serializable confirmation policy used to build the manager, when available.
    pub confirmation_policy: Option<crate::hitl::ConfirmationPolicy>,
    /// Serializable queue configuration used to build the optional command queue.
    pub queue_config: Option<SessionQueueConfig>,
    /// Context providers for augmenting prompts with external context
    pub context_providers: Vec<Arc<dyn ContextProvider>>,
    /// Planning mode — Auto (detect from message), Enabled, or Disabled.
    pub planning_mode: PlanningMode,
    /// Enable goal tracking
    pub goal_tracking: bool,
    /// Optional hook engine for firing lifecycle events (PreToolUse, PostToolUse, etc.)
    pub hook_engine: Option<Arc<dyn HookExecutor>>,
    /// Optional skill registry for tool permission enforcement
    pub skill_registry: Option<Arc<crate::skills::SkillRegistry>>,
    /// Max consecutive malformed-tool-args errors before aborting (default: 2).
    ///
    /// When the LLM returns tool arguments with `__parse_error`, the error is
    /// fed back as a tool result. After this many consecutive parse errors the
    /// loop bails instead of retrying indefinitely.
    pub max_parse_retries: u32,
    /// Per-tool execution timeout in milliseconds (`None` = no timeout).
    ///
    /// When set, each tool execution is wrapped in `tokio::time::timeout`.
    /// A timeout produces an error result sent back to the LLM rather than
    /// crashing the session.
    pub tool_timeout_ms: Option<u64>,
    /// Maximum number of sibling branches/tools to run concurrently in bounded
    /// parallel fan-out paths.
    pub max_parallel_tasks: usize,
    /// Runtime-driven automatic child-agent delegation.
    pub auto_delegation: crate::config::AutoDelegationConfig,
    /// Available child agents for automatic delegation.
    pub agent_registry: Option<Arc<AgentRegistry>>,
    /// Circuit-breaker threshold: max consecutive LLM API failures before
    /// aborting (default: 3).
    ///
    /// In non-streaming mode, transient LLM failures are retried up to this
    /// many times (with short exponential backoff) before the loop bails.
    /// In streaming mode, any failure is fatal (events cannot be replayed).
    pub circuit_breaker_threshold: u32,
    /// Max consecutive identical tool signatures before aborting (default: 3).
    ///
    /// A tool signature is the exact combination of tool name + compact JSON
    /// arguments. This prevents the agent from getting stuck repeating the same
    /// tool call in a loop, for example repeatedly fetching the same URL.
    pub duplicate_tool_call_threshold: u32,
    /// Enable auto-compaction when context usage exceeds threshold.
    pub auto_compact: bool,
    /// Context usage percentage threshold to trigger auto-compaction (0.0 - 1.0).
    /// Default: 0.80 (80%).
    pub auto_compact_threshold: f32,
    /// Maximum context window size in tokens (used for auto-compact calculation).
    /// Default: 200_000.
    pub max_context_tokens: usize,
    /// Optional agent memory for auto-remember after tool execution and recall before prompts.
    pub memory: Option<Arc<crate::memory::AgentMemory>>,
    /// Inject a continuation message when the LLM stops calling tools before the
    /// task is complete. Enabled by default. Set to `false` to disable.
    ///
    /// When enabled, if the LLM produces a response with no tool calls but the
    /// response text looks like an intermediate step (not a final answer), the
    /// loop injects [`crate::prompts::CONTINUATION`] as a user message and
    /// continues for up to `max_continuation_turns` additional turns.
    pub continuation_enabled: bool,
    /// Maximum number of continuation injections per execution (default: 3).
    ///
    /// Prevents infinite loops when the LLM repeatedly stops without completing.
    pub max_continuation_turns: u32,
    /// Maximum execution time in milliseconds (`None` = no timeout).
    ///
    /// When set, the entire execution loop is wrapped in a timeout check.
    /// If execution exceeds this duration, the loop bails with an error.
    /// This prevents runaway executions that consume excessive API quota.
    pub max_execution_time_ms: Option<u64>,
    /// Host-supplied budget guard consulted before every LLM call (and
    /// after, for usage accounting). `None` means no enforcement.
    pub budget_guard: Option<Arc<dyn crate::budget::BudgetGuard>>,
    /// Host-provided ID generator + clock. Defaults to wall-clock UUIDs.
    /// Replace via [`SessionOptions::with_host_env`](crate::agent_api::SessionOptions::with_host_env)
    /// when deterministic replay is needed.
    pub host_env: Arc<crate::host_env::HostEnv>,
}

impl std::fmt::Debug for AgentConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentConfig")
            .field("prompt_slots", &self.prompt_slots)
            .field("tools", &self.tools)
            .field("max_tool_rounds", &self.max_tool_rounds)
            .field("security_provider", &self.security_provider.is_some())
            .field("permission_checker", &self.permission_checker.is_some())
            .field("permission_policy", &self.permission_policy.is_some())
            .field("confirmation_manager", &self.confirmation_manager.is_some())
            .field("confirmation_policy", &self.confirmation_policy.is_some())
            .field("queue_config", &self.queue_config.is_some())
            .field("context_providers", &self.context_providers.len())
            .field("planning_mode", &self.planning_mode)
            .field("goal_tracking", &self.goal_tracking)
            .field("hook_engine", &self.hook_engine.is_some())
            .field(
                "skill_registry",
                &self.skill_registry.as_ref().map(|r| r.len()),
            )
            .field("max_parse_retries", &self.max_parse_retries)
            .field("tool_timeout_ms", &self.tool_timeout_ms)
            .field("max_parallel_tasks", &self.max_parallel_tasks)
            .field("auto_delegation", &self.auto_delegation)
            .field(
                "agent_registry",
                &self.agent_registry.as_ref().map(|registry| registry.len()),
            )
            .field("circuit_breaker_threshold", &self.circuit_breaker_threshold)
            .field(
                "duplicate_tool_call_threshold",
                &self.duplicate_tool_call_threshold,
            )
            .field("auto_compact", &self.auto_compact)
            .field("auto_compact_threshold", &self.auto_compact_threshold)
            .field("max_context_tokens", &self.max_context_tokens)
            .field("continuation_enabled", &self.continuation_enabled)
            .field("max_continuation_turns", &self.max_continuation_turns)
            .field("memory", &self.memory.is_some())
            .finish()
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            prompt_slots: SystemPromptSlots::default(),
            tools: Vec::new(), // Tools are provided by ToolExecutor
            max_tool_rounds: MAX_TOOL_ROUNDS,
            security_provider: None,
            permission_checker: None,
            permission_policy: None,
            confirmation_manager: None,
            confirmation_policy: None,
            queue_config: None,
            context_providers: Vec::new(),
            planning_mode: PlanningMode::default(),
            goal_tracking: false,
            hook_engine: None,
            skill_registry: Some(Arc::new(crate::skills::SkillRegistry::with_builtins())),
            max_parse_retries: 2,
            tool_timeout_ms: None,
            max_parallel_tasks: DEFAULT_MAX_PARALLEL_TASKS,
            auto_delegation: crate::config::AutoDelegationConfig::default(),
            agent_registry: None,
            circuit_breaker_threshold: 3,
            duplicate_tool_call_threshold: 3,
            auto_compact: false,
            auto_compact_threshold: 0.80,
            max_context_tokens: 200_000,
            memory: None,
            continuation_enabled: true,
            max_continuation_turns: 3,
            max_execution_time_ms: None,
            budget_guard: None,
            host_env: Arc::new(crate::host_env::HostEnv::system()),
        }
    }
}

/// Events emitted during agent execution
///
/// Subscribe via [`crate::AgentSession::stream`].
/// New variants may be added in minor releases — always include a wildcard arm
/// (`_ => {}`) when matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum AgentEvent {
    /// Agent started processing
    #[serde(rename = "agent_start")]
    Start { prompt: String },

    /// Runtime agent style/mode selected for the current execution.
    #[serde(rename = "agent_mode_changed")]
    AgentModeChanged {
        /// Stable UI/runtime mode label, e.g. "general", "planning", "explore".
        mode: String,
        /// Canonical built-in agent name associated with this mode.
        agent: String,
        /// Human-readable explanation of the selected style.
        description: String,
    },

    /// LLM turn started
    #[serde(rename = "turn_start")]
    TurnStart { turn: usize },

    /// Text delta from streaming
    #[serde(rename = "text_delta")]
    TextDelta { text: String },

    /// Reasoning/thinking delta from streaming (for models like kimi, deepseek)
    #[serde(rename = "reasoning_delta")]
    ReasoningDelta { text: String },

    /// Tool execution started
    #[serde(rename = "tool_start")]
    ToolStart { id: String, name: String },

    /// Tool input delta from streaming (partial JSON arguments)
    #[serde(rename = "tool_input_delta")]
    ToolInputDelta { delta: String },

    /// Tool execution completed
    #[serde(rename = "tool_end")]
    ToolEnd {
        id: String,
        name: String,
        output: String,
        exit_code: i32,
        #[serde(skip_serializing_if = "Option::is_none")]
        metadata: Option<serde_json::Value>,
        /// Structured discriminant set by tools that mapped their failure
        /// into a typed [`ToolErrorKind`](crate::tools::ToolErrorKind)
        /// (e.g. `edit` / `patch` on a `WorkspaceError::VersionConflict`).
        /// `None` on success or untyped failure.
        #[serde(skip_serializing_if = "Option::is_none")]
        error_kind: Option<crate::tools::ToolErrorKind>,
    },

    /// Intermediate tool output (streaming delta)
    #[serde(rename = "tool_output_delta")]
    ToolOutputDelta {
        id: String,
        name: String,
        delta: String,
    },

    /// LLM turn completed
    #[serde(rename = "turn_end")]
    TurnEnd { turn: usize, usage: TokenUsage },

    /// Agent completed
    #[serde(rename = "agent_end")]
    End {
        text: String,
        usage: TokenUsage,
        verification_summary: Box<crate::verification::VerificationSummary>,
        #[serde(skip_serializing_if = "Option::is_none")]
        meta: Option<crate::llm::LlmResponseMeta>,
    },

    /// Error occurred
    #[serde(rename = "error")]
    Error { message: String },

    /// Tool execution requires confirmation (HITL)
    #[serde(rename = "confirmation_required")]
    ConfirmationRequired {
        tool_id: String,
        tool_name: String,
        args: serde_json::Value,
        timeout_ms: u64,
    },

    /// Confirmation received from user (HITL)
    #[serde(rename = "confirmation_received")]
    ConfirmationReceived {
        tool_id: String,
        approved: bool,
        reason: Option<String>,
    },

    /// Confirmation timed out (HITL)
    #[serde(rename = "confirmation_timeout")]
    ConfirmationTimeout {
        tool_id: String,
        action_taken: String, // "rejected" or "auto_approved"
    },

    /// External task pending (needs SDK processing)
    #[serde(rename = "external_task_pending")]
    ExternalTaskPending {
        task_id: String,
        session_id: String,
        lane: crate::queue::SessionLane,
        command_type: String,
        payload: serde_json::Value,
        timeout_ms: u64,
    },

    /// External task completed
    #[serde(rename = "external_task_completed")]
    ExternalTaskCompleted {
        task_id: String,
        session_id: String,
        success: bool,
    },

    /// Tool execution denied by permission policy
    #[serde(rename = "permission_denied")]
    PermissionDenied {
        tool_id: String,
        tool_name: String,
        args: serde_json::Value,
        reason: String,
    },

    /// Context resolution started
    #[serde(rename = "context_resolving")]
    ContextResolving { providers: Vec<String> },

    /// Context resolution completed
    #[serde(rename = "context_resolved")]
    ContextResolved {
        total_items: usize,
        total_tokens: usize,
    },

    // ========================================================================
    // a3s-lane integration events
    // ========================================================================
    /// Command moved to dead letter queue after exhausting retries
    #[serde(rename = "command_dead_lettered")]
    CommandDeadLettered {
        command_id: String,
        command_type: String,
        lane: String,
        error: String,
        attempts: u32,
    },

    /// Command retry attempt
    #[serde(rename = "command_retry")]
    CommandRetry {
        command_id: String,
        command_type: String,
        lane: String,
        attempt: u32,
        delay_ms: u64,
    },

    /// Queue alert (depth warning, latency alert, etc.)
    #[serde(rename = "queue_alert")]
    QueueAlert {
        level: String,
        alert_type: String,
        message: String,
    },

    // ========================================================================
    // Task tracking events
    // ========================================================================
    /// Task list updated
    #[serde(rename = "task_updated")]
    TaskUpdated {
        session_id: String,
        tasks: Vec<crate::planning::Task>,
    },

    // ========================================================================
    // Memory System events (Phase 3)
    // ========================================================================
    /// Memory stored
    #[serde(rename = "memory_stored")]
    MemoryStored {
        memory_id: String,
        memory_type: String,
        importance: f32,
        tags: Vec<String>,
    },

    /// Memory recalled
    #[serde(rename = "memory_recalled")]
    MemoryRecalled {
        memory_id: String,
        content: String,
        relevance: f32,
    },

    /// Memories searched
    #[serde(rename = "memories_searched")]
    MemoriesSearched {
        query: Option<String>,
        tags: Vec<String>,
        result_count: usize,
    },

    /// Memory cleared
    #[serde(rename = "memory_cleared")]
    MemoryCleared {
        tier: String, // "long_term", "short_term", "working"
        count: u64,
    },

    // ========================================================================
    // Subagent events
    // ========================================================================
    /// Subagent task started
    #[serde(rename = "subagent_start")]
    SubagentStart {
        /// Unique task identifier
        task_id: String,
        /// Child session ID
        session_id: String,
        /// Parent session ID
        parent_session_id: String,
        /// Agent type (e.g., "explore", "general")
        agent: String,
        /// Short description of the task
        description: String,
    },

    /// Subagent task progress update
    #[serde(rename = "subagent_progress")]
    SubagentProgress {
        /// Task identifier
        task_id: String,
        /// Child session ID
        session_id: String,
        /// Progress status message
        status: String,
        /// Additional metadata
        metadata: serde_json::Value,
    },

    /// Subagent task completed
    #[serde(rename = "subagent_end")]
    SubagentEnd {
        /// Task identifier
        task_id: String,
        /// Child session ID
        session_id: String,
        /// Agent type
        agent: String,
        /// Task output/result
        output: String,
        /// Whether the task succeeded
        success: bool,
    },

    // ========================================================================
    // Planning and Goal Tracking Events (Phase 1)
    // ========================================================================
    /// Planning phase started
    #[serde(rename = "planning_start")]
    PlanningStart { prompt: String },

    /// Planning phase completed
    #[serde(rename = "planning_end")]
    PlanningEnd {
        plan: ExecutionPlan,
        estimated_steps: usize,
    },

    /// Step execution started
    #[serde(rename = "step_start")]
    StepStart {
        step_id: String,
        description: String,
        step_number: usize,
        total_steps: usize,
    },

    /// Step execution completed
    #[serde(rename = "step_end")]
    StepEnd {
        step_id: String,
        status: TaskStatus,
        step_number: usize,
        total_steps: usize,
    },

    /// Goal extracted from prompt
    #[serde(rename = "goal_extracted")]
    GoalExtracted { goal: AgentGoal },

    /// Goal progress update
    #[serde(rename = "goal_progress")]
    GoalProgress {
        goal: String,
        progress: f32,
        completed_steps: usize,
        total_steps: usize,
    },

    /// Goal achieved
    #[serde(rename = "goal_achieved")]
    GoalAchieved {
        goal: String,
        total_steps: usize,
        duration_ms: i64,
    },

    // ========================================================================
    // Context Compaction events
    // ========================================================================
    /// Context automatically compacted due to high usage
    #[serde(rename = "context_compacted")]
    ContextCompacted {
        session_id: String,
        before_messages: usize,
        after_messages: usize,
        percent_before: f32,
    },

    // ========================================================================
    // Persistence events
    // ========================================================================
    /// Session persistence failed — SDK clients should handle this
    #[serde(rename = "persistence_failed")]
    PersistenceFailed {
        session_id: String,
        operation: String,
        error: String,
    },

    // ========================================================================
    // Cluster / platform events
    //
    // These variants are emitted by the host platform via
    // `HookExecutor` and are not produced by the agent loop itself. They
    // give in-session code a uniform way to observe platform-level
    // decisions (budget exhaustion, scheduled passivation, peer
    // invocations) without coupling to the host's transport.
    // ========================================================================
    /// A budget threshold was crossed for this session/tenant.
    ///
    /// Emitted by a host `BudgetGuard` impl when LLM/tool spend hits a
    /// soft or hard threshold. The session is **not** automatically
    /// halted — `kind` lets in-session policy decide (e.g. fast-compact
    /// at "soft", refuse next LLM call at "hard").
    #[serde(rename = "budget_threshold_hit")]
    BudgetThresholdHit {
        /// Logical resource: "llm_tokens", "tool_calls", "wall_time",
        /// "usd_cost", or host-defined.
        resource: String,
        /// "soft" or "hard"; host-defined semantics beyond that.
        kind: String,
        /// Current consumed amount in the same unit as `limit`.
        consumed: f64,
        /// Threshold that was crossed.
        limit: f64,
        /// Optional explanation for logs / UI.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },

    /// The host is asking the session to release in-memory state.
    ///
    /// Emitted before the host calls `session.close()` or moves the
    /// session to another node. Session code that holds large caches
    /// can react (flush to memory store, drop derived state). The
    /// framework does not act on this event itself.
    #[serde(rename = "passivation_requested")]
    PassivationRequested {
        /// "idle_reaper", "node_drain", "migration", "manual", or
        /// host-defined.
        reason: String,
        /// Optional deadline (Unix epoch ms) before forced close.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        deadline_ms: Option<u64>,
    },

    /// Another session in the cluster has invoked this one.
    ///
    /// Lets in-session hooks distinguish "human-driven send" from
    /// "peer-driven send" without inspecting prompts. The host routes
    /// the actual prompt through the normal `send` / `stream` path;
    /// this event is metadata only.
    #[serde(rename = "peer_invocation")]
    PeerInvocation {
        /// Session id of the invoking peer (cluster-stable).
        from_session_id: String,
        /// Optional tenant of the invoking peer.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        from_tenant_id: Option<String>,
        /// Distributed-trace correlation id linking the two sessions.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
    },
}

/// Result of agent execution
#[derive(Debug, Clone)]
pub struct AgentResult {
    pub text: String,
    pub messages: Vec<Message>,
    pub usage: TokenUsage,
    pub tool_calls_count: usize,
    pub verification_reports: Vec<crate::verification::VerificationReport>,
}

impl AgentResult {
    pub fn verification_summary(&self) -> crate::verification::VerificationSummary {
        crate::verification::VerificationSummary::from_reports(&self.verification_reports)
    }

    pub fn verification_summary_text(&self) -> String {
        crate::verification::format_verification_summary(&self.verification_summary())
    }

    pub fn has_pending_verification(&self) -> bool {
        matches!(
            self.verification_summary().status,
            crate::verification::VerificationStatus::NeedsReview
        )
    }
}

// ============================================================================
// ToolCommand — bridges ToolExecutor to SessionCommand for queue submission
// ============================================================================

/// Adapter that implements `SessionCommand` for tool execution via the queue.
///
/// Wraps a `ToolExecutor` call so it can be submitted to `SessionLaneQueue`.
pub struct ToolCommand {
    tool_executor: Arc<ToolExecutor>,
    tool_name: String,
    tool_args: Value,
    tool_context: ToolContext,
    skill_registry: Option<Arc<crate::skills::SkillRegistry>>,
}

impl ToolCommand {
    /// Create a new ToolCommand
    pub fn new(
        tool_executor: Arc<ToolExecutor>,
        tool_name: String,
        tool_args: Value,
        tool_context: ToolContext,
        skill_registry: Option<Arc<crate::skills::SkillRegistry>>,
    ) -> Self {
        Self {
            tool_executor,
            tool_name,
            tool_args,
            tool_context,
            skill_registry,
        }
    }
}

#[async_trait]
impl SessionCommand for ToolCommand {
    async fn execute(&self) -> Result<Value> {
        // Check skill-based tool permissions
        if let Some(registry) = &self.skill_registry {
            // If there are instruction skills with tool restrictions, check permissions
            let restricting_skills = registry.global_tool_restricting_skills();

            if !restricting_skills.is_empty() {
                let mut allowed = false;

                for skill in &restricting_skills {
                    if skill.is_tool_allowed(&self.tool_name) {
                        allowed = true;
                        break;
                    }
                }

                if !allowed {
                    return Err(anyhow::anyhow!(
                        "Tool '{}' is not allowed by any active skill. Active skills restrict tools to their allowed-tools lists.",
                        self.tool_name
                    ));
                }
            }
        }

        // Execute the tool
        let result = self
            .tool_executor
            .execute_with_context(&self.tool_name, &self.tool_args, &self.tool_context)
            .await?;
        Ok(serde_json::json!({
            "output": result.output,
            "exit_code": result.exit_code,
            "metadata": result.metadata,
        }))
    }

    fn command_type(&self) -> &str {
        &self.tool_name
    }

    fn payload(&self) -> Value {
        self.tool_args.clone()
    }
}

// ============================================================================
// AgentLoop
// ============================================================================

/// Internal agent loop executor.
#[derive(Clone)]
pub(crate) struct AgentLoop {
    llm_client: Arc<dyn LlmClient>,
    tool_executor: Arc<ToolExecutor>,
    tool_context: ToolContext,
    config: AgentConfig,
    /// Optional lane queue for priority-based tool execution
    command_queue: Option<Arc<SessionLaneQueue>>,
    /// Optional sink for per-tool-round checkpoints. Populated by
    /// `build_agent_loop` when the session has a configured
    /// `SessionStore`. The agent loop uses
    /// [`AgentLoop::set_checkpoint_run`] to bind a run id before
    /// `execute_with_session`, then persists a checkpoint after each
    /// completed tool round.
    pub(crate) checkpoint_sink: Option<Arc<dyn crate::loop_checkpoint::LoopCheckpointSink>>,
    /// Run id under which checkpoints are stored. Reset per execution
    /// via [`AgentLoop::set_checkpoint_run`].
    pub(crate) checkpoint_run_id: Option<String>,
}

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod extra_agent_tests;
