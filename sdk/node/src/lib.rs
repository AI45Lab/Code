//! A3S Code Node.js Bindings
//!
//! Native Node.js addon via napi-rs that wraps `a3s-code-core`'s Agent API.
//!
//! ## Usage
//!
//! ```javascript
//! const { Agent } = require('@a3s-lab/code');
//!
//! const agent = await Agent.create('agent.acl');
//! const session = agent.session('/my-project');
//!
//! const result = await session.send('What files handle auth?');
//! console.log(result.text);
//! ```
//!
//! ## Panic safety at the FFI boundary
//!
//! napi 2.x does **not** wrap exported bodies in `catch_unwind` by default. A
//! Rust panic that reaches the `extern "C"` boundary aborts the whole Node
//! process (Rust ≥ 1.81) — it does *not* become a catchable JS error. Only two
//! contexts are panic-safe: a `#[napi]` **async** fn / `impl Future` (panic →
//! rejected Promise) and a sync fn explicitly tagged `#[napi(catch_unwind)]`.
//! Everything else aborts (or silently loses the panic): default **sync**
//! `#[napi]` fns, `ThreadsafeFunction` callbacks (a panic there — or a
//! return-value conversion `Err` — aborts via `napi_fatal_error` under *both*
//! `ErrorStrategy` variants), `tokio::spawn`'d task bodies (panic swallowed,
//! never surfaced), `Drop`/finalizers, and module init.
//!
//! Convention this crate follows so the boundary stays safe: never
//! `.unwrap()` / `.expect()` / `panic!` in those contexts. Propagate with `?`
//! into a `napi::Error`, or fail closed with `unwrap_or_else` inside
//! threadsafe callbacks. (Audited 2026-05: the only production panic site is
//! the lazy Tokio-runtime build in `fallback_runtime()`, reached from within
//! `#[napi]` bodies; the spawned-task and threadsafe-callback paths are
//! panic-free by construction.)

#[macro_use]
extern crate napi_derive;

mod js_slash_command;
use js_slash_command::{js_command_context_to_object, JsSlashCommand};

use a3s_code_core::commands::CommandContext as RustCommandContext;
use a3s_code_core::config::{
    BrowserBackend as RustBrowserBackend, HeadlessConfig as RustHeadlessConfig,
    SearchConfig as RustSearchConfig, SearchEngineConfig as RustSearchEngineConfig,
    SearchHealthConfig as RustSearchHealthConfig,
};
use a3s_code_core::hitl::{
    ConfirmationPolicy as RustConfirmationPolicy, TimeoutAction as RustTimeoutAction,
};
use a3s_code_core::hooks::{
    Hook as RustHook, HookConfig as RustHookConfig, HookEvent as RustHookEvent,
    HookEventType as RustHookEventType, HookHandler as RustHookHandler,
    HookMatcher as RustHookMatcher, HookResponse as RustHookResponse,
};
use a3s_code_core::llm::{ContentBlock as RustContentBlock, Message as RustMessage};
use a3s_code_core::orchestration::{
    execute_steps_parallel, execute_steps_parallel_resumable, AgentStepSpec as RustAgentStepSpec,
    StepOutcome as RustStepOutcome,
};
use a3s_code_core::permissions::{
    PermissionDecision as RustPermissionDecision, PermissionPolicy as RustPermissionPolicy,
    PermissionRule as RustPermissionRule,
};
use a3s_code_core::queue::{
    ExternalTaskResult as RustExternalTaskResult, LaneHandlerConfig as RustLaneHandlerConfig,
    MetricsSnapshot as RustMetricsSnapshot, SessionLane as RustSessionLane,
    SessionQueueConfig as RustSessionQueueConfig, TaskHandlerMode as RustTaskHandlerMode,
};
use a3s_code_core::skills::{builtin_skills as rust_builtin_skills, SkillKind as RustSkillKind};
use a3s_code_core::subagent::{
    AgentDefinition as RustAgentDefinition, ModelConfig as RustAgentModelConfig,
    WorkerAgentKind as RustWorkerAgentKind, WorkerAgentSpec as RustWorkerAgentSpec,
};
use a3s_code_core::verification::{
    format_verification_summary as rust_format_verification_summary,
    VerificationCommand as RustVerificationCommand, VerificationReport as RustVerificationReport,
    VerificationStatus as RustVerificationStatus, VerificationSummary as RustVerificationSummary,
};
use a3s_code_core::{
    Agent as RustAgent, AgentEvent as RustAgentEvent, AgentResult as RustAgentResult,
    AgentSession as RustAgentSession, PlanningMode as RustPlanningMode,
    SessionOptions as RustSessionOptions,
};
use napi::Either;

// AHP Type Bindings
mod ahp_types;

use std::future::Future;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, OnceLock,
};

// ============================================================================
// Tokio Runtime
// ============================================================================

struct NapiRuntime;

fn fallback_runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("a3s-code-node-worker")
            .build()
            .expect("failed to create Tokio runtime for Node bindings")
    })
}

impl NapiRuntime {
    fn spawn<F>(&self, fut: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        // Try the current runtime first; otherwise use the binding-owned runtime.
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(fut)
        } else {
            fallback_runtime().spawn(fut)
        }
    }

    fn block_on<F: Future>(&self, fut: F) -> F::Output {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.block_on(fut)
        } else {
            fallback_runtime().block_on(fut)
        }
    }
}

fn get_runtime() -> NapiRuntime {
    NapiRuntime
}

// ============================================================================
// AgentResult
// ============================================================================

#[napi(object)]
#[derive(Clone)]
pub struct AgentResult {
    pub text: String,
    pub tool_calls_count: u32,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub verification_status: String,
    pub pending_verification_count: u32,
    pub failed_verification_count: u32,
    pub verification_report_count: u32,
    pub verification_summary_json: String,
    pub verification_summary_text: String,
}

impl From<RustAgentResult> for AgentResult {
    fn from(r: RustAgentResult) -> Self {
        let verification_summary = r.verification_summary();
        let verification_summary_json = verification_summary.to_value().to_string();
        let verification_summary_text = rust_format_verification_summary(&verification_summary);
        Self {
            text: r.text,
            tool_calls_count: r.tool_calls_count as u32,
            prompt_tokens: r.usage.prompt_tokens as u32,
            completion_tokens: r.usage.completion_tokens as u32,
            total_tokens: r.usage.total_tokens as u32,
            verification_status: verification_status_label(verification_summary.status),
            pending_verification_count: verification_summary.pending_required_check_count as u32,
            failed_verification_count: verification_summary.failed_check_count as u32,
            verification_report_count: verification_summary.report_count as u32,
            verification_summary_json,
            verification_summary_text,
        }
    }
}

fn verification_status_label(status: RustVerificationStatus) -> String {
    match status {
        RustVerificationStatus::Passed => "passed",
        RustVerificationStatus::Failed => "failed",
        RustVerificationStatus::NeedsReview => "needs_review",
        RustVerificationStatus::Skipped => "skipped",
    }
    .to_string()
}

#[napi]
pub fn format_verification_summary(summary: serde_json::Value) -> napi::Result<String> {
    let summary: RustVerificationSummary = match summary {
        serde_json::Value::String(summary_json) => serde_json::from_str(&summary_json),
        value => serde_json::from_value(value),
    }
    .map_err(|e| napi::Error::from_reason(format!("Invalid verification summary: {e}")))?;
    Ok(rust_format_verification_summary(&summary))
}

// ============================================================================
// AgentEvent
// ============================================================================

#[napi(object)]
#[derive(Clone)]
pub struct AgentEvent {
    #[napi(js_name = "type")]
    pub event_type: String,
    pub text: Option<String>,
    pub tool_name: Option<String>,
    pub tool_id: Option<String>,
    pub tool_output: Option<String>,
    pub exit_code: Option<i32>,
    pub turn: Option<u32>,
    pub prompt: Option<String>,
    pub error: Option<String>,
    pub total_tokens: Option<u32>,
    pub verification_summary_json: Option<String>,
    pub verification_summary_text: Option<String>,
    /// Extra data for events that don't map to standard fields (JSON-encoded)
    pub data: Option<String>,
    /// Structured discriminant for tool failures on `tool_end` events
    /// (JSON-encoded with a `type` field). `None` on success or untyped
    /// failure. Lets streaming consumers branch on the failure kind
    /// without scanning `tool_output`.
    pub error_kind_json: Option<String>,
}

#[napi(object)]
#[derive(Clone)]
pub struct VerificationCommand {
    pub id: String,
    pub kind: String,
    pub description: String,
    pub command: String,
    pub required: Option<bool>,
    pub timeout_ms: Option<u32>,
}

impl From<VerificationCommand> for RustVerificationCommand {
    fn from(command: VerificationCommand) -> Self {
        let mut rust_command = if command.required.unwrap_or(true) {
            RustVerificationCommand::required(
                command.id,
                command.kind,
                command.description,
                command.command,
            )
        } else {
            RustVerificationCommand::optional(
                command.id,
                command.kind,
                command.description,
                command.command,
            )
        };

        if let Some(timeout_ms) = command.timeout_ms {
            rust_command = rust_command.with_timeout_ms(timeout_ms as u64);
        }

        rust_command
    }
}

impl AgentEvent {
    fn empty(event_type: &str) -> Self {
        Self {
            event_type: event_type.to_string(),
            text: None,
            tool_name: None,
            tool_id: None,
            tool_output: None,
            exit_code: None,
            turn: None,
            prompt: None,
            error: None,
            total_tokens: None,
            verification_summary_json: None,
            verification_summary_text: None,
            data: None,
            error_kind_json: None,
        }
    }
}

impl From<RustAgentEvent> for AgentEvent {
    fn from(e: RustAgentEvent) -> Self {
        match e {
            RustAgentEvent::Start { prompt } => Self {
                prompt: Some(prompt),
                ..Self::empty("start")
            },
            RustAgentEvent::AgentModeChanged {
                mode,
                agent,
                description,
            } => Self {
                data: Some(
                    serde_json::json!({
                        "mode": mode,
                        "agent": agent,
                        "description": description
                    })
                    .to_string(),
                ),
                ..Self::empty("agent_mode_changed")
            },
            RustAgentEvent::TurnStart { turn } => Self {
                turn: Some(turn as u32),
                ..Self::empty("turn_start")
            },
            RustAgentEvent::TextDelta { text } => Self {
                text: Some(text),
                ..Self::empty("text_delta")
            },
            RustAgentEvent::ReasoningDelta { text } => Self {
                text: Some(text),
                ..Self::empty("reasoning_delta")
            },
            RustAgentEvent::ToolStart { id, name } => Self {
                tool_id: Some(id),
                tool_name: Some(name),
                ..Self::empty("tool_start")
            },
            RustAgentEvent::ToolInputDelta { delta } => Self {
                text: Some(delta),
                ..Self::empty("tool_input_delta")
            },
            RustAgentEvent::ToolEnd {
                id,
                name,
                output,
                exit_code,
                metadata: _,
                error_kind,
            } => Self {
                tool_id: Some(id),
                tool_name: Some(name),
                tool_output: Some(output),
                exit_code: Some(exit_code),
                error_kind_json: error_kind
                    .as_ref()
                    .and_then(|k| serde_json::to_string(k).ok()),
                ..Self::empty("tool_end")
            },
            RustAgentEvent::ToolOutputDelta { id, name, delta } => Self {
                tool_id: Some(id),
                tool_name: Some(name),
                text: Some(delta),
                ..Self::empty("tool_output_delta")
            },
            RustAgentEvent::TurnEnd { turn, usage } => Self {
                turn: Some(turn as u32),
                total_tokens: Some(usage.total_tokens as u32),
                ..Self::empty("turn_end")
            },
            RustAgentEvent::End {
                text,
                usage,
                verification_summary,
                ..
            } => Self {
                text: Some(text),
                total_tokens: Some(usage.total_tokens as u32),
                verification_summary_text: Some(rust_format_verification_summary(
                    &verification_summary,
                )),
                verification_summary_json: Some(verification_summary.to_value().to_string()),
                ..Self::empty("end")
            },
            RustAgentEvent::Error { message } => Self {
                error: Some(message),
                ..Self::empty("error")
            },
            RustAgentEvent::ConfirmationRequired {
                tool_id,
                tool_name,
                args,
                timeout_ms,
            } => Self {
                tool_id: Some(tool_id),
                tool_name: Some(tool_name),
                data: Some(
                    serde_json::json!({
                        "args": args,
                        "timeout_ms": timeout_ms
                    })
                    .to_string(),
                ),
                ..Self::empty("confirmation_required")
            },
            RustAgentEvent::ConfirmationReceived {
                tool_id,
                approved,
                reason,
            } => Self {
                tool_id: Some(tool_id),
                data: Some(
                    serde_json::json!({
                        "approved": approved,
                        "reason": reason
                    })
                    .to_string(),
                ),
                ..Self::empty("confirmation_received")
            },
            RustAgentEvent::ConfirmationTimeout {
                tool_id,
                action_taken,
            } => Self {
                tool_id: Some(tool_id),
                data: Some(
                    serde_json::json!({
                        "action_taken": action_taken
                    })
                    .to_string(),
                ),
                ..Self::empty("confirmation_timeout")
            },
            RustAgentEvent::ExternalTaskPending {
                task_id,
                session_id,
                command_type,
                payload,
                timeout_ms,
                ..
            } => Self {
                data: Some(
                    serde_json::json!({
                        "task_id": task_id,
                        "session_id": session_id,
                        "command_type": command_type,
                        "payload": payload,
                        "timeout_ms": timeout_ms
                    })
                    .to_string(),
                ),
                ..Self::empty("external_task_pending")
            },
            RustAgentEvent::ExternalTaskCompleted {
                task_id,
                session_id,
                success,
            } => Self {
                data: Some(
                    serde_json::json!({
                        "task_id": task_id,
                        "session_id": session_id,
                        "success": success
                    })
                    .to_string(),
                ),
                ..Self::empty("external_task_completed")
            },
            RustAgentEvent::PermissionDenied {
                tool_id,
                tool_name,
                args,
                reason,
            } => Self {
                tool_id: Some(tool_id),
                tool_name: Some(tool_name),
                data: Some(
                    serde_json::json!({
                        "args": args,
                        "reason": reason
                    })
                    .to_string(),
                ),
                ..Self::empty("permission_denied")
            },
            RustAgentEvent::ContextResolving { providers } => Self {
                data: Some(serde_json::json!({ "providers": providers }).to_string()),
                ..Self::empty("context_resolving")
            },
            RustAgentEvent::ContextResolved {
                total_items,
                total_tokens,
            } => Self {
                data: Some(
                    serde_json::json!({
                        "total_items": total_items,
                        "total_tokens": total_tokens
                    })
                    .to_string(),
                ),
                ..Self::empty("context_resolved")
            },
            RustAgentEvent::CommandDeadLettered {
                command_id,
                command_type,
                lane,
                error,
                attempts,
            } => Self {
                data: Some(
                    serde_json::json!({
                        "command_id": command_id,
                        "command_type": command_type,
                        "lane": lane,
                        "error": error,
                        "attempts": attempts
                    })
                    .to_string(),
                ),
                ..Self::empty("command_dead_lettered")
            },
            RustAgentEvent::CommandRetry {
                command_id,
                command_type,
                lane,
                attempt,
                delay_ms,
            } => Self {
                data: Some(
                    serde_json::json!({
                        "command_id": command_id,
                        "command_type": command_type,
                        "lane": lane,
                        "attempt": attempt,
                        "delay_ms": delay_ms
                    })
                    .to_string(),
                ),
                ..Self::empty("command_retry")
            },
            RustAgentEvent::QueueAlert {
                level,
                alert_type,
                message,
            } => Self {
                data: Some(
                    serde_json::json!({
                        "level": level,
                        "alert_type": alert_type,
                        "message": message
                    })
                    .to_string(),
                ),
                ..Self::empty("queue_alert")
            },
            RustAgentEvent::TaskUpdated { session_id, tasks } => Self {
                data: Some(
                    serde_json::json!({
                        "session_id": session_id,
                        "tasks": tasks
                    })
                    .to_string(),
                ),
                ..Self::empty("task_updated")
            },
            RustAgentEvent::MemoryStored {
                memory_id,
                memory_type,
                importance,
                tags,
            } => Self {
                data: Some(
                    serde_json::json!({
                        "memory_id": memory_id,
                        "memory_type": memory_type,
                        "importance": importance,
                        "tags": tags
                    })
                    .to_string(),
                ),
                ..Self::empty("memory_stored")
            },
            RustAgentEvent::MemoryRecalled {
                memory_id,
                content,
                relevance,
            } => Self {
                data: Some(
                    serde_json::json!({
                        "memory_id": memory_id,
                        "content": content,
                        "relevance": relevance
                    })
                    .to_string(),
                ),
                ..Self::empty("memory_recalled")
            },
            RustAgentEvent::MemoriesSearched {
                query,
                tags,
                result_count,
            } => Self {
                data: Some(
                    serde_json::json!({
                        "query": query,
                        "tags": tags,
                        "result_count": result_count
                    })
                    .to_string(),
                ),
                ..Self::empty("memories_searched")
            },
            RustAgentEvent::MemoryCleared { tier, count } => Self {
                data: Some(serde_json::json!({ "tier": tier, "count": count }).to_string()),
                ..Self::empty("memory_cleared")
            },
            RustAgentEvent::SubagentStart {
                task_id,
                session_id,
                parent_session_id,
                agent,
                description,
            } => Self {
                data: Some(
                    serde_json::json!({
                        "task_id": task_id,
                        "session_id": session_id,
                        "parent_session_id": parent_session_id,
                        "agent": agent,
                        "description": description
                    })
                    .to_string(),
                ),
                ..Self::empty("subagent_start")
            },
            RustAgentEvent::SubagentProgress {
                task_id,
                session_id,
                status,
                metadata,
            } => Self {
                data: Some(
                    serde_json::json!({
                        "task_id": task_id,
                        "session_id": session_id,
                        "status": status,
                        "metadata": metadata
                    })
                    .to_string(),
                ),
                ..Self::empty("subagent_progress")
            },
            RustAgentEvent::SubagentEnd {
                task_id,
                session_id,
                agent,
                output,
                success,
            } => Self {
                data: Some(
                    serde_json::json!({
                        "task_id": task_id,
                        "session_id": session_id,
                        "agent": agent,
                        "output": output,
                        "success": success
                    })
                    .to_string(),
                ),
                ..Self::empty("subagent_end")
            },
            RustAgentEvent::PlanningStart { prompt } => Self {
                prompt: Some(prompt),
                ..Self::empty("planning_start")
            },
            RustAgentEvent::PlanningEnd {
                plan,
                estimated_steps,
            } => Self {
                data: Some(
                    serde_json::json!({
                        "plan": plan,
                        "estimated_steps": estimated_steps
                    })
                    .to_string(),
                ),
                ..Self::empty("planning_end")
            },
            RustAgentEvent::StepStart {
                step_id,
                description,
                step_number,
                total_steps,
            } => Self {
                data: Some(
                    serde_json::json!({
                        "step_id": step_id,
                        "description": description,
                        "step_number": step_number,
                        "total_steps": total_steps
                    })
                    .to_string(),
                ),
                ..Self::empty("step_start")
            },
            RustAgentEvent::StepEnd {
                step_id,
                status,
                step_number,
                total_steps,
            } => Self {
                data: Some(
                    serde_json::json!({
                        "step_id": step_id,
                        "status": status,
                        "step_number": step_number,
                        "total_steps": total_steps
                    })
                    .to_string(),
                ),
                ..Self::empty("step_end")
            },
            RustAgentEvent::ContextCompacted {
                session_id,
                before_messages,
                after_messages,
                percent_before,
            } => Self {
                data: Some(
                    serde_json::json!({
                        "session_id": session_id,
                        "before_messages": before_messages,
                        "after_messages": after_messages,
                        "percent_before": percent_before
                    })
                    .to_string(),
                ),
                ..Self::empty("context_compacted")
            },
            RustAgentEvent::PersistenceFailed {
                session_id,
                operation,
                error,
            } => Self {
                data: Some(
                    serde_json::json!({
                        "session_id": session_id,
                        "operation": operation,
                        "error": error
                    })
                    .to_string(),
                ),
                ..Self::empty("persistence_failed")
            },
            _ => Self::empty("unknown"),
        }
    }
}

// ============================================================================
// ToolResult
// ============================================================================

#[napi(object)]
#[derive(Clone)]
pub struct ToolResult {
    pub name: String,
    pub output: String,
    pub exit_code: i32,
    /// Raw JSON-encoded tool metadata returned by the Rust core API.
    pub metadata_json: Option<String>,
    /// Convenience JSON view of `metadata.document_runtime` when present.
    pub document_runtime_json: Option<String>,
    /// Structured discriminant for tool failures, JSON-encoded with a
    /// `type` field on the top level — e.g.
    /// `{"type":"version_conflict","path":"doc.md","expected":"etag-1","actual":"etag-2"}`.
    /// `None` on success or untyped failure. SDK callers parse it to
    /// branch on the failure kind without scanning the `output` string.
    pub error_kind_json: Option<String>,
}

/// Execution limits for `Session.program`.
#[napi(object)]
#[derive(Clone)]
pub struct ProgramScriptLimits {
    pub timeout_ms: Option<u32>,
    pub max_tool_calls: Option<u32>,
    pub max_output_bytes: Option<u32>,
}

/// Options for `Session.program`.
#[napi(object)]
#[derive(Clone)]
pub struct ProgramScriptOptions {
    pub source: Option<String>,
    pub path: Option<String>,
    pub inputs: Option<serde_json::Value>,
    pub allowed_tools: Option<Vec<String>>,
    pub limits: Option<ProgramScriptLimits>,
}

/// Options for `Session.delegateTask`.
#[napi(object)]
#[derive(Clone)]
pub struct DelegateTaskOptions {
    pub agent: String,
    pub description: String,
    pub prompt: String,
    pub background: Option<bool>,
    pub max_steps: Option<u32>,
}

/// Object-shaped request for `Session.sendRequest` and `Session.streamRequest`.
#[napi(object)]
#[derive(Clone)]
pub struct SessionRequestOptions {
    pub prompt: String,
    pub history: Option<Vec<MessageObject>>,
    pub attachments: Option<Vec<AttachmentObject>>,
}

fn session_request_parts(
    request: Either<String, SessionRequestOptions>,
    history: Option<Vec<MessageObject>>,
) -> napi::Result<(
    String,
    Option<Vec<RustMessage>>,
    Vec<a3s_code_core::llm::Attachment>,
)> {
    match request {
        Either::A(prompt) => {
            let rust_history = history.map(|h| js_messages_to_rust(&h)).transpose()?;
            Ok((prompt, rust_history, Vec::new()))
        }
        Either::B(request) => {
            let rust_history = request
                .history
                .map(|h| js_messages_to_rust(&h))
                .transpose()?;
            let rust_attachments = request
                .attachments
                .as_deref()
                .map(js_attachments_to_rust)
                .unwrap_or_default();
            Ok((request.prompt, rust_history, rust_attachments))
        }
    }
}

async fn send_session_request(
    session: Arc<RustAgentSession>,
    prompt: String,
    history: Option<Vec<RustMessage>>,
    attachments: Vec<a3s_code_core::llm::Attachment>,
) -> napi::Result<AgentResult> {
    let result = if attachments.is_empty() {
        get_runtime()
            .spawn(async move { session.send(&prompt, history.as_deref()).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
    } else {
        get_runtime()
            .spawn(async move {
                session
                    .send_with_attachments(&prompt, &attachments, history.as_deref())
                    .await
            })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
    }
    .map_err(|e| napi::Error::from_reason(format!("Agent execution failed: {e}")))?;

    Ok(AgentResult::from(result))
}

async fn stream_session_request(
    session: Arc<RustAgentSession>,
    prompt: String,
    history: Option<Vec<RustMessage>>,
    attachments: Vec<a3s_code_core::llm::Attachment>,
) -> napi::Result<EventStream> {
    let (rx, _handle) = if attachments.is_empty() {
        get_runtime()
            .spawn(async move { session.stream(&prompt, history.as_deref()).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
    } else {
        get_runtime()
            .spawn(async move {
                session
                    .stream_with_attachments(&prompt, &attachments, history.as_deref())
                    .await
            })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
    }
    .map_err(|e| napi::Error::from_reason(format!("Failed to start stream: {e}")))?;

    Ok(EventStream {
        rx: Arc::new(tokio::sync::Mutex::new(rx)),
        done: Arc::new(AtomicBool::new(false)),
    })
}

fn tool_result_from_core(result: a3s_code_core::ToolCallResult) -> ToolResult {
    ToolResult {
        name: result.name,
        output: result.output,
        exit_code: result.exit_code,
        metadata_json: result.metadata.as_ref().map(serde_json::Value::to_string),
        document_runtime_json: result
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("document_runtime"))
            .map(serde_json::Value::to_string),
        error_kind_json: result
            .error_kind
            .as_ref()
            .and_then(|k| serde_json::to_string(k).ok()),
    }
}

fn normalize_program_script_options(options: serde_json::Value) -> napi::Result<serde_json::Value> {
    let obj = options
        .as_object()
        .ok_or_else(|| napi::Error::from_reason("program options must be an object"))?;

    let mut args = serde_json::Map::new();
    args.insert("type".to_string(), serde_json::json!("script"));
    args.insert("language".to_string(), serde_json::json!("javascript"));

    for key in ["source", "path", "inputs", "limits"] {
        if let Some(value) = obj.get(key) {
            args.insert(key.to_string(), value.clone());
        }
    }

    if let Some(value) = obj.get("allowedTools").or_else(|| obj.get("allowed_tools")) {
        args.insert("allowed_tools".to_string(), value.clone());
    }

    Ok(serde_json::Value::Object(args))
}

fn delegate_task_options_to_args(options: DelegateTaskOptions) -> serde_json::Value {
    let mut args = serde_json::json!({
        "agent": options.agent,
        "description": options.description,
        "prompt": options.prompt,
    });
    if let Some(background) = options.background {
        args["background"] = serde_json::json!(background);
    }
    if let Some(max_steps) = options.max_steps {
        args["max_steps"] = serde_json::json!(max_steps);
    }
    args
}

fn parallel_task_options_to_args(tasks: Vec<DelegateTaskOptions>) -> serde_json::Value {
    let task_values = tasks
        .into_iter()
        .map(delegate_task_options_to_args)
        .collect::<Vec<_>>();
    serde_json::json!({ "tasks": task_values })
}

#[napi(object)]
#[derive(Clone)]
pub struct GitCommandOptions {
    pub command: String,
    pub subcommand: Option<String>,
    pub name: Option<String>,
    pub path: Option<String>,
    pub new_branch: Option<bool>,
    pub base: Option<String>,
    pub force: Option<bool>,
    pub max_count: Option<u32>,
    pub message: Option<String>,
    pub include_untracked: Option<bool>,
    pub target: Option<String>,
    pub r#ref: Option<String>,
    pub reference: Option<String>,
}

fn git_command_options_to_args(options: GitCommandOptions) -> serde_json::Value {
    let mut args = serde_json::json!({ "command": options.command });
    if let Some(value) = options.subcommand {
        args["subcommand"] = serde_json::json!(value);
    }
    if let Some(value) = options.name {
        args["name"] = serde_json::json!(value);
    }
    if let Some(value) = options.path {
        args["path"] = serde_json::json!(value);
    }
    if let Some(value) = options.new_branch {
        args["new_branch"] = serde_json::json!(value);
    }
    if let Some(value) = options.base {
        args["base"] = serde_json::json!(value);
    }
    if let Some(value) = options.force {
        args["force"] = serde_json::json!(value);
    }
    if let Some(value) = options.max_count {
        args["max_count"] = serde_json::json!(value);
    }
    if let Some(value) = options.message {
        args["message"] = serde_json::json!(value);
    }
    if let Some(value) = options.include_untracked {
        args["include_untracked"] = serde_json::json!(value);
    }
    if let Some(value) = options.target {
        args["target"] = serde_json::json!(value);
    }
    if let Some(value) = options.r#ref.or(options.reference) {
        args["ref"] = serde_json::json!(value);
    }
    args
}

fn normalize_git_args(mut args: serde_json::Value) -> napi::Result<serde_json::Value> {
    let obj = args
        .as_object_mut()
        .ok_or_else(|| napi::Error::from_reason("git options must be an object"))?;

    if !obj.contains_key("command") {
        return Err(napi::Error::from_reason(
            "git options must include a command field",
        ));
    }

    for (from, to) in [
        ("newBranch", "new_branch"),
        ("maxCount", "max_count"),
        ("includeUntracked", "include_untracked"),
    ] {
        if let Some(value) = obj.remove(from) {
            obj.entry(to.to_string()).or_insert(value);
        }
    }

    if let Some(value) = obj.remove("reference") {
        obj.entry("ref".to_string()).or_insert(value);
    }

    Ok(args)
}

fn timeout_ms_to_secs(timeout_ms: u64) -> u64 {
    timeout_ms.div_ceil(1000).max(1)
}

fn normalize_mcp_server_config(
    mut value: serde_json::Value,
) -> napi::Result<a3s_code_core::mcp::protocol::McpServerConfig> {
    let obj = value
        .as_object_mut()
        .ok_or_else(|| napi::Error::from_reason("MCP server config must be an object"))?;

    for key in [
        "timeoutMs",
        "timeout_ms",
        "toolTimeoutMs",
        "tool_timeout_ms",
    ] {
        if let Some(timeout_ms) = obj.remove(key) {
            let timeout_ms = timeout_ms
                .as_u64()
                .ok_or_else(|| napi::Error::from_reason(format!("{key} must be a number")))?;
            obj.entry("toolTimeoutSecs".to_string())
                .or_insert_with(|| serde_json::json!(timeout_ms_to_secs(timeout_ms)));
            break;
        }
    }

    if let Some(transport) = obj.get_mut("transport") {
        normalize_mcp_transport_alias(transport);
    }

    serde_json::from_value(value)
        .map_err(|e| napi::Error::from_reason(format!("Invalid MCP server config: {e}")))
}

fn normalize_mcp_transport_alias(transport: &mut serde_json::Value) {
    match transport {
        serde_json::Value::String(kind) => {
            if matches!(kind.as_str(), "streamable_http" | "streamableHttp") {
                *kind = "streamable-http".to_string();
            }
        }
        serde_json::Value::Object(obj) => {
            if let Some(serde_json::Value::String(kind)) = obj.get_mut("type") {
                if matches!(kind.as_str(), "streamable_http" | "streamableHttp") {
                    *kind = "streamable-http".to_string();
                }
            }
        }
        _ => {}
    }
}

// ============================================================================
// WebSearchParams
// ============================================================================

/// Parameters for the web_search tool.
#[napi(object)]
#[derive(Clone)]
pub struct JsWebSearchParams {
    /// The search query.
    pub query: String,
    /// List of search engines to use.
    pub engines: Option<Vec<String>>,
    /// Maximum number of results to return (default: 10, max: 50).
    pub limit: Option<u32>,
    /// Search timeout in seconds (default: 10, max: 60).
    pub timeout: Option<u32>,
    /// Proxy URL (e.g., http://127.0.0.1:8080 or socks5://127.0.0.1:1080).
    pub proxy: Option<String>,
    /// Output format: "text" or "json".
    pub format: Option<String>,
}

// ============================================================================
// EventStream
// ============================================================================

/// Result of a single `EventStream.next()` call.
#[napi(object)]
#[derive(Clone)]
pub struct NextResult {
    pub value: Option<AgentEvent>,
    pub done: bool,
}

/// Streaming event iterator. Use `for await (const event of stream)` or call `.next()` manually.
#[napi]
pub struct EventStream {
    rx: Arc<tokio::sync::Mutex<tokio::sync::mpsc::Receiver<RustAgentEvent>>>,
    done: Arc<AtomicBool>,
}

#[napi]
impl EventStream {
    /// Get the next event from the stream.
    ///
    /// Returns `{ value: AgentEvent | null, done: boolean }`.
    /// When `done` is true, the stream is exhausted.
    #[napi]
    pub async fn next(&self) -> napi::Result<NextResult> {
        if self.done.load(Ordering::Relaxed) {
            return Ok(NextResult {
                value: None,
                done: true,
            });
        }
        let rx = self.rx.clone();
        let done_flag = self.done.clone();
        let result = get_runtime()
            .spawn(async move {
                let mut guard = rx.lock().await;
                guard.recv().await
            })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?;
        match result {
            Some(event) => {
                let is_end = matches!(event, RustAgentEvent::End { .. });
                let is_error = matches!(event, RustAgentEvent::Error { .. });
                let js_event = AgentEvent::from(event);
                if is_end || is_error {
                    done_flag.store(true, Ordering::Relaxed);
                    Ok(NextResult {
                        value: Some(js_event),
                        done: true,
                    })
                } else {
                    Ok(NextResult {
                        value: Some(js_event),
                        done: false,
                    })
                }
            }
            None => {
                done_flag.store(true, Ordering::Relaxed);
                Ok(NextResult {
                    value: None,
                    done: true,
                })
            }
        }
    }
}

// ============================================================================
// SessionOptions
// ============================================================================

/// An inline skill registered programmatically (no file required).
///
/// Use `kind: "instruction"` for prompt injections or `kind: "persona"` to
/// replace the default role section of the system prompt.
#[napi(object)]
#[derive(Clone, Default)]
pub struct InlineSkill {
    /// Unique skill name (kebab-case recommended, e.g. "type-hints").
    pub name: String,
    /// Skill kind: `"instruction"` or `"persona"`.
    pub kind: String,
    /// Markdown content for the skill.
    pub content: String,
}

// ============================================================================
// Typed store / provider helpers
// ============================================================================

// Internal napi-rs compatibility shims.
//
// napi-rs `#[napi(object)]` structs cannot hold `#[napi]` class instances directly,
// so SessionOptions fields that accept store/provider objects are typed as these plain
// structs. Users work exclusively with the public classes (FileMemoryStore,
// FileSessionStore, MemorySessionStore, DefaultSecurityProvider); TypeScript structural
// compatibility ensures those instances satisfy these struct shapes automatically.
//
// These are NOT exported in the public TypeScript API surface (index.d.ts).

#[napi(object)]
#[derive(Clone, Default)]
pub struct JsMemoryStore {
    pub backend: String,
    pub dir: Option<String>,
}

#[napi(object)]
#[derive(Clone, Default)]
pub struct JsSessionStore {
    pub backend: String,
    pub dir: Option<String>,
}

#[napi(object)]
#[derive(Clone, Default)]
pub struct JsSecurityProvider {
    pub kind: String,
}

#[napi(object)]
#[derive(Clone, Default)]
pub struct JsWorkspaceBackend {
    pub kind: String,
    pub root: Option<String>,
    pub s3: Option<JsS3BackendConfig>,
}

/// Configuration for an S3-compatible workspace backend.
///
/// Use this with [`S3WorkspaceBackend`] to point a session's built-in file
/// tools at any S3-compatible endpoint (AWS S3, MinIO, RustFS, R2, etc.).
/// `endpoint` is optional — omit it to use the AWS default. `prefix` is
/// the logical workspace root inside the bucket; every workspace path
/// becomes `<prefix>/<path>` when sent to S3.
#[napi(object)]
#[derive(Clone, Default)]
pub struct JsS3BackendConfig {
    /// Optional S3 endpoint URL. Omit for AWS S3 (the SDK will compute it
    /// from `region`). Set to `https://...` for MinIO / RustFS / R2 / etc.
    pub endpoint: Option<String>,
    /// AWS region. Defaults to `us-east-1` when omitted.
    pub region: Option<String>,
    /// Static access key. Use `sessionToken` together when STS-issued.
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
    /// Bucket name.
    pub bucket: String,
    /// Logical workspace prefix inside the bucket (without leading/trailing
    /// slashes). Use `""` to make the bucket root the workspace.
    pub prefix: String,
    /// `true` for MinIO / RustFS / most non-AWS endpoints; `false` for AWS S3.
    pub force_path_style: Option<bool>,
    /// Maximum bytes a single `read` may return. The backend rejects any
    /// response with `Content-Length` greater than this without buffering
    /// the body. Defaults to 10 MiB on the Rust side when omitted.
    pub max_read_bytes: Option<i64>,
    /// Enable degraded `grep` / `glob` against this S3 backend. Off by
    /// default — object storage has no native search, so the only viable
    /// strategy is `LIST` + `GET` + regex, which can be slow and expensive.
    pub search_enabled: Option<bool>,
    /// Upper bound on objects considered per `grep` / `glob` call. Defaults
    /// to 500 on the Rust side. Ignored when `searchEnabled` is `false`.
    pub max_objects_scanned: Option<i64>,
    /// Per-object body-size ceiling for `grep` downloads. Larger objects are
    /// skipped (debug-traced). Defaults to 1 MiB on the Rust side. Ignored
    /// when `searchEnabled` is `false`.
    pub max_grep_bytes_per_object: Option<i64>,
    /// Concurrent object downloads during `grep`. Defaults to 8 on the
    /// Rust side. Set lower when the gitserver / S3 endpoint rate-limits
    /// aggressively; set higher when latency dominates. Ignored when
    /// `searchEnabled` is `false`.
    pub search_concurrency: Option<i64>,
}

/// Configuration for a [`RemoteGitBackend`] — an HTTP/JSON client that
/// brings the `git` tool to non-local workspaces (S3, future container /
/// DFS).
///
/// Pass alongside `workspaceBackend` on a session to attach remote git
/// on top of any filesystem backend. The protocol is specified in the
/// repository RFC `apps/docs/content/docs/en/code/rfcs/workspace-remote-git.mdx`.
#[napi(object)]
#[derive(Clone, Default)]
pub struct JsRemoteGitBackendConfig {
    /// Base URL of the gitserver, no trailing slash. The client builds
    /// `{baseUrl}/v1/repos/{repoId}/git/{op}` per the RFC.
    pub base_url: String,
    /// Opaque repository identifier, URL-safe. Negotiated out of band
    /// with the gitserver operator.
    pub repo_id: String,
    /// Bearer token sent as `Authorization: Bearer <token>`. Required in
    /// production; omitting it emits a `tracing::warn!` and is only safe
    /// on a trusted localhost gitserver.
    pub bearer_token: Option<String>,
    /// mTLS client certificate path (PEM). When set together with
    /// `clientKeyPem`, the backend reads both files at construction and
    /// configures mTLS on the HTTP client. Setting only one of the pair
    /// errors at construction.
    pub client_cert_pem: Option<String>,
    /// mTLS client private key path (PEM). PKCS#8 format expected for the
    /// `rustls-tls` backend. See `clientCertPem`.
    pub client_key_pem: Option<String>,
    /// Per-call HTTP timeout in milliseconds. Defaults to 30 000.
    pub request_timeout_ms: Option<i64>,
    /// Client-side cap on `diff` response bytes. Defaults to 1 MiB.
    pub max_diff_bytes: Option<i64>,
    /// Client-side cap on `log` `max_count`. Defaults to 200.
    pub max_log_entries: Option<i64>,
}

/// File-backed long-term memory store.
///
/// ```js
/// agent.session('.', { memoryStore: new FileMemoryStore('./memory') });
/// ```
#[napi]
pub struct FileMemoryStore {
    pub backend: String,
    pub dir: String,
}

#[napi]
impl FileMemoryStore {
    /// Create a file-backed memory store at `dir`.
    #[napi(constructor)]
    pub fn new(dir: String) -> Self {
        Self {
            backend: "file".to_string(),
            dir,
        }
    }
}

/// File-backed session store (persists sessions to disk for later resumption).
///
/// ```js
/// agent.session('.', {
///   sessionStore: new FileSessionStore('./sessions'),
///   sessionId: 'my-session',
///   autoSave: true,
/// });
/// ```
#[napi]
pub struct FileSessionStore {
    pub backend: String,
    pub dir: String,
}

#[napi]
impl FileSessionStore {
    /// Create a file-backed session store at `dir`.
    #[napi(constructor)]
    pub fn new(dir: String) -> Self {
        Self {
            backend: "file".to_string(),
            dir,
        }
    }
}

/// In-memory (non-persistent) session store.
///
/// Useful for testing, ephemeral runs, and CI pipelines where no disk state is needed.
///
/// ```js
/// agent.session('.', { sessionStore: new MemorySessionStore() });
/// ```
#[napi]
pub struct MemorySessionStore {
    pub backend: String,
}

#[napi]
impl MemorySessionStore {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            backend: "memory".to_string(),
        }
    }
}

impl Default for MemorySessionStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Default security provider: input taint tracking + output sanitisation.
///
/// ```js
/// agent.session('.', { securityProvider: new DefaultSecurityProvider() });
/// ```
#[napi]
pub struct DefaultSecurityProvider {
    pub kind: String,
}

#[napi]
impl DefaultSecurityProvider {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            kind: "default".to_string(),
        }
    }
}

impl Default for DefaultSecurityProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Local filesystem workspace backend.
///
/// This is the explicit typed form of the default local workspace behavior.
/// It is useful when callers want to pass workspace backends through the same
/// option surface that remote/browser backends will use.
///
/// ```js
/// agent.session('/repo', { workspaceBackend: new LocalWorkspaceBackend('/repo') });
/// ```
#[napi]
pub struct LocalWorkspaceBackend {
    pub kind: String,
    pub root: String,
}

#[napi]
impl LocalWorkspaceBackend {
    /// Create a local filesystem workspace backend rooted at `root`.
    #[napi(constructor)]
    pub fn new(root: String) -> Self {
        Self {
            kind: "local".to_string(),
            root,
        }
    }
}

/// S3-compatible object-storage workspace backend.
///
/// Points built-in file tools (`read`, `write`, `edit`, `patch`, `ls`) at an
/// S3-compatible bucket. Works with AWS S3, MinIO, RustFS, Cloudflare R2,
/// Backblaze B2, and other S3-API-compatible services.
///
/// `bash`, `git`, `grep`, and `glob` are intentionally **not** registered
/// when this backend is in use — object storage cannot service them.
///
/// ```js
/// const backend = new S3WorkspaceBackend({
///   endpoint: 'https://minio.local:9000',
///   region: 'us-east-1',
///   accessKeyId: 'AKIA...',
///   secretAccessKey: '...',
///   bucket: 'workspace',
///   prefix: 'users/u1/sessions/s1',
///   forcePathStyle: true,
/// });
/// agent.session('s3://workspace/users/u1/sessions/s1', { workspaceBackend: backend });
/// ```
#[napi]
pub struct S3WorkspaceBackend {
    pub kind: String,
    pub s3: JsS3BackendConfig,
}

#[napi]
impl S3WorkspaceBackend {
    /// Create an S3-compatible workspace backend.
    #[napi(constructor)]
    pub fn new(config: JsS3BackendConfig) -> Self {
        Self {
            kind: "s3".to_string(),
            s3: config,
        }
    }
}

// ============================================================================
// AHP Transport Classes
// ============================================================================

/// Stdio transport for AHP (Agent Harness Protocol).
///
/// Launches a child process and communicates via stdin/stdout using JSON-RPC 2.0.
///
/// ```js
/// agent.session('.', {
///   ahpTransport: new StdioTransport('python', ['ahp_server.py'])
/// });
/// ```
#[napi]
pub struct StdioTransport {
    pub kind: String,
    pub program: Option<String>,
    pub args: Option<Vec<String>>,
    pub url: Option<String>,
    pub auth_token: Option<String>,
    pub path: Option<String>,
}

#[napi]
impl StdioTransport {
    #[napi(constructor)]
    pub fn new(program: String, args: Vec<String>) -> Self {
        Self {
            kind: "stdio".to_string(),
            program: Some(program),
            args: Some(args),
            url: None,
            auth_token: None,
            path: None,
        }
    }
}

/// HTTP transport for AHP (Agent Harness Protocol).
///
/// Connects to a remote AHP harness server via HTTP.
///
/// ```js
/// agent.session('.', {
///   ahpTransport: new HttpTransport('http://localhost:8080/ahp')
/// });
/// ```
#[napi]
pub struct HttpTransport {
    pub kind: String,
    pub program: Option<String>,
    pub args: Option<Vec<String>>,
    pub url: Option<String>,
    pub auth_token: Option<String>,
    pub path: Option<String>,
}

#[napi]
impl HttpTransport {
    #[napi(constructor)]
    pub fn new(url: String, auth_token: Option<String>) -> Self {
        Self {
            kind: "http".to_string(),
            program: None,
            args: None,
            url: Some(url),
            auth_token,
            path: None,
        }
    }
}

/// WebSocket transport for AHP (Agent Harness Protocol).
///
/// Connects to a remote AHP harness server via WebSocket for bidirectional streaming.
///
/// ```js
/// agent.session('.', {
///   ahpTransport: new WebSocketTransport('ws://localhost:8080/ahp')
/// });
/// ```
#[napi]
pub struct WebSocketTransport {
    pub kind: String,
    pub program: Option<String>,
    pub args: Option<Vec<String>>,
    pub url: Option<String>,
    pub auth_token: Option<String>,
    pub path: Option<String>,
}

#[napi]
impl WebSocketTransport {
    #[napi(constructor)]
    pub fn new(url: String, auth_token: Option<String>) -> Self {
        Self {
            kind: "websocket".to_string(),
            program: None,
            args: None,
            url: Some(url),
            auth_token,
            path: None,
        }
    }
}

/// Unix socket transport for AHP (Agent Harness Protocol).
///
/// Connects to a local AHP harness server via Unix domain socket.
///
/// ```js
/// agent.session('.', {
///   ahpTransport: new UnixSocketTransport('/tmp/ahp.sock')
/// });
/// ```
#[napi]
pub struct UnixSocketTransport {
    pub kind: String,
    pub program: Option<String>,
    pub args: Option<Vec<String>>,
    pub url: Option<String>,
    pub auth_token: Option<String>,
    pub path: Option<String>,
}

#[napi]
impl UnixSocketTransport {
    #[napi(constructor)]
    pub fn new(path: String) -> Self {
        Self {
            kind: "unix_socket".to_string(),
            program: None,
            args: None,
            url: None,
            auth_token: None,
            path: Some(path),
        }
    }
}

// ============================================================================
// SessionOptions
// ============================================================================

/// Union type for AHP transport configuration.
/// Accepts any of: StdioTransport, HttpTransport, WebSocketTransport, UnixSocketTransport.
#[napi(object)]
#[derive(Clone, Default)]
pub struct JsAhpTransport {
    pub kind: String,
    pub program: Option<String>,
    pub args: Option<Vec<String>>,
    pub url: Option<String>,
    pub auth_token: Option<String>,
    pub path: Option<String>,
}

#[napi(object)]
#[derive(Default)]
pub struct PermissionPolicy {
    /// Tool invocation patterns that are always denied first.
    pub deny: Option<Vec<String>>,
    /// Tool invocation patterns that are auto-approved.
    pub allow: Option<Vec<String>>,
    /// Tool invocation patterns that always require confirmation.
    pub ask: Option<Vec<String>>,
    /// Default decision when no rule matches: "allow", "deny", or "ask".
    pub default_decision: Option<String>,
    /// Whether this policy is enabled. Defaults to true.
    pub enabled: Option<bool>,
}

/// Reproducible recipe for a disposable worker/subagent.
///
/// This is the Node.js cattle-mode interface: define workers in data, pass them
/// to SessionOptions.workerAgents, Agent.sessionForWorker(), or
/// Session.registerWorkerAgent(). The Rust core compiles each spec into the
/// normal delegated-agent runtime definition.
#[napi(object)]
#[derive(Default)]
pub struct WorkerAgentSpec {
    /// Stable worker name used by task delegation.
    pub name: String,
    /// Human-readable worker purpose.
    pub description: String,
    /// Preset role: "read_only", "planner", "implementer", "verifier", "reviewer", or "custom".
    pub kind: Option<String>,
    /// Hide from UI lists while allowing explicit delegation.
    pub hidden: Option<bool>,
    /// Optional permission policy override.
    pub permissions: Option<PermissionPolicy>,
    /// Optional model override in "provider/model" format.
    pub model: Option<String>,
    /// Optional worker-specific prompt.
    pub prompt: Option<String>,
    /// Maximum execution steps/tool rounds.
    pub max_steps: Option<u32>,
    /// How child runs resolve Ask decisions: "auto_approve" (default), "deny_on_ask", or "inherit_parent".
    pub confirmation_inheritance: Option<String>,
}

#[napi(object)]
pub struct AgentDefinition {
    pub name: String,
    pub description: String,
    pub native: bool,
    pub hidden: bool,
    pub model: Option<String>,
    pub prompt: Option<String>,
    pub max_steps: Option<u32>,
    /// How child runs resolve Ask decisions: "auto_approve", "deny_on_ask", or "inherit_parent".
    pub confirmation_inheritance: Option<String>,
}

/// HITL confirmation policy configuration.
///
/// Controls the runtime behavior of Human-in-the-Loop confirmation flow.
#[napi(object)]
#[derive(Default)]
pub struct ConfirmationPolicy {
    /// Whether HITL is enabled (default: false, all tools auto-approved).
    pub enabled: Option<bool>,
    /// Default timeout in milliseconds (default: 30000 = 30s).
    pub default_timeout_ms: Option<u32>,
    /// Action to take on timeout: "reject" or "auto_approve" (default: "reject").
    pub timeout_action: Option<String>,
    /// Lanes that should auto-approve without confirmation: "control", "query", "execute", or "generate".
    pub yolo_lanes: Option<Vec<String>>,
}

/// Snapshot of a pending HITL tool confirmation.
#[napi(object)]
pub struct PendingConfirmation {
    /// Tool call ID to pass to `confirmToolUse`.
    pub tool_id: String,
    /// Tool name awaiting confirmation.
    pub tool_name: String,
    /// Tool arguments for display in a confirmation UI.
    pub args: serde_json::Value,
    /// Milliseconds remaining before the confirmation times out.
    pub remaining_ms: f64,
}

impl From<a3s_code_core::hitl::PendingConfirmationInfo> for PendingConfirmation {
    fn from(info: a3s_code_core::hitl::PendingConfirmationInfo) -> Self {
        Self {
            tool_id: info.tool_id,
            tool_name: info.tool_name,
            args: info.args,
            remaining_ms: info.remaining_ms as f64,
        }
    }
}

#[napi(object)]
#[derive(Default)]
pub struct AutoDelegationOptions {
    /// Enable runtime-driven automatic child-agent delegation.
    pub enabled: Option<bool>,
    /// Allow automatic delegation to launch multiple child agents in parallel.
    ///
    /// Manual `parallel_task` calls remain available when this is false.
    pub auto_parallel: Option<bool>,
    /// Minimum local confidence required to auto-delegate a child task.
    pub min_confidence: Option<f64>,
    /// Maximum number of automatic child tasks per user request.
    pub max_tasks: Option<u32>,
}

#[napi(object)]
#[derive(Default)]
pub struct SessionOptions {
    /// Override the default model. Format: "provider/model" (e.g., "openai/gpt-4o").
    pub model: Option<String>,
    /// Enable built-in skills (4 skills: code-search, code-review, explain-code, find-bugs).
    pub builtin_skills: Option<bool>,
    /// Extra directories to scan for skill files (.md with YAML frontmatter).
    pub skill_dirs: Option<Vec<String>>,
    /// Extra directories to scan for agent files.
    pub agent_dirs: Option<Vec<String>>,
    /// Reproducible disposable workers to register for task delegation.
    pub worker_agents: Option<Vec<WorkerAgentSpec>>,
    /// Optional advanced queue configuration for explicit external/hybrid lane dispatch.
    ///
    /// Ordinary sessions are queue-free unless this is provided.
    pub queue_config: Option<SessionQueueConfig>,
    /// Explicit permission policy for tool execution.
    pub permission_policy: Option<PermissionPolicy>,
    /// Explicit planning mode: "auto", "enabled", or "disabled".
    ///
    /// Prefer this over `planning` when the caller needs an unambiguous SDK contract.
    /// If both are set, `planningMode` wins.
    pub planning_mode: Option<String>,
    /// Legacy planning shortcut. Omit for auto planning, true to force planning, false to disable.
    pub planning: Option<bool>,
    /// Enable goal tracking (default: false).
    pub goal_tracking: Option<bool>,
    /// Max consecutive parse errors before abort.
    pub max_parse_retries: Option<u32>,
    /// Per-tool execution timeout in milliseconds.
    pub tool_timeout_ms: Option<f64>,
    /// Max LLM API failures before abort.
    pub circuit_breaker_threshold: Option<u32>,
    /// Enable auto-compaction when context window fills up (default: false).
    pub auto_compact: Option<bool>,
    /// Context usage threshold (0.0–1.0) to trigger auto-compaction (default: 0.8).
    pub auto_compact_threshold: Option<f64>,
    /// Retention limits for large tool/program artifacts.
    pub artifact_store_limits: Option<ArtifactStoreLimits>,
    /// Long-term memory store backend.
    ///
    /// Pass `new FileMemoryStore("./memory")` for file-based persistence.
    /// ```js
    /// agent.session('.', { memoryStore: new FileMemoryStore('./memory') });
    /// ```
    pub memory_store: Option<JsMemoryStore>,
    /// Session persistence store backend.
    ///
    /// Pass `new FileSessionStore("./sessions")` to persist sessions to disk,
    /// or `new MemorySessionStore()` for an ephemeral in-process store.
    /// ```js
    /// agent.session('.', {
    ///   sessionStore: new FileSessionStore('./sessions'),
    ///   sessionId: 'my-session',
    ///   autoSave: true,
    /// });
    /// ```
    pub session_store: Option<JsSessionStore>,
    /// Security provider.
    ///
    /// Pass `new DefaultSecurityProvider()` to enable input taint tracking and
    /// output sanitisation. Omit to disable security (default: no security).
    /// ```js
    /// agent.session('.', { securityProvider: new DefaultSecurityProvider() });
    /// ```
    pub security_provider: Option<JsSecurityProvider>,
    /// Workspace backend used by built-in tools.
    ///
    /// Pass `new LocalWorkspaceBackend("/repo")` to explicitly use the local
    /// filesystem backend. This option is the SDK surface for future remote,
    /// browser, DFS, and container-backed workspace implementations.
    /// ```js
    /// agent.session('/repo', { workspaceBackend: new LocalWorkspaceBackend('/repo') });
    /// ```
    pub workspace_backend: Option<JsWorkspaceBackend>,
    /// Optional remote git provider. When set, the resulting session attaches
    /// a `RemoteGitBackend` on top of `workspaceBackend` so the built-in
    /// `git` tool is available even on object-storage workspaces.
    ///
    /// ```js
    /// agent.session('s3://workspace/u1/s1', {
    ///   workspaceBackend: new S3WorkspaceBackend({ ... }),
    ///   remoteGit: {
    ///     baseUrl: 'https://gitserver.internal',
    ///     repoId:  'u1/s1',
    ///     bearerToken: token,
    ///   },
    /// });
    /// ```
    pub remote_git: Option<JsRemoteGitBackendConfig>,
    /// Custom role/identity prepended before the core agentic prompt.
    /// Example: "You are a senior Python developer specializing in FastAPI."
    pub role: Option<String>,
    /// Custom coding guidelines appended after the core prompt.
    /// Example: "Always use type hints. Follow PEP 8."
    pub guidelines: Option<String>,
    /// Custom response style (replaces default Response Format section).
    pub response_style: Option<String>,
    /// Freeform extra instructions appended at the end.
    pub extra: Option<String>,
    /// Inline skills registered programmatically without needing skill files on disk.
    /// Each entry defines an instruction or persona skill injected into the system prompt.
    pub inline_skills: Option<Vec<InlineSkill>>,
    /// Override maximum number of tool-call rounds for this session.
    pub max_tool_rounds: Option<u32>,
    /// Override maximum sibling parallel branches for this session.
    pub max_parallel_tasks: Option<u32>,
    /// Override automatic child-agent delegation for this session.
    pub auto_delegation: Option<AutoDelegationOptions>,
    /// Global session-level kill switch for automatic parallel child-agent fan-out.
    ///
    /// Manual `parallel_task` calls remain available when this is false.
    pub auto_parallel: Option<bool>,
    /// Sampling temperature (0.0–1.0). Overrides the provider default.
    /// Only applied when `model` is also set.
    pub temperature: Option<f64>,
    /// Extended thinking token budget (e.g. 10_000). Enables chain-of-thought reasoning.
    /// Only applied when `model` is also set. Provider must support extended thinking.
    pub thinking_budget: Option<u32>,
    /// Enable continuation injection (default: true).
    /// When enabled, the loop injects a follow-up prompt when the LLM stops without completing.
    pub continuation_enabled: Option<bool>,
    /// Maximum continuation injections per execution (default: 3).
    pub max_continuation_turns: Option<u32>,
    /// Session ID (auto-generated if not set).
    ///
    /// Set a stable ID so the session can be saved and resumed later:
    /// ```js
    /// agent.session('.', { sessionId: 'my-session', sessionStore: new FileSessionStore('./sessions'), autoSave: true });
    /// // Later:
    /// agent.resumeSession('my-session', { sessionStore: new FileSessionStore('./sessions') });
    /// ```
    pub session_id: Option<String>,
    /// Host-defined tenant id. Opaque to the framework — propagated to
    /// SessionData, hooks, and traces for multi-tenant aggregation /
    /// billing. Pair with `principal` / `agentTemplateId` /
    /// `correlationId` for full identity context.
    pub tenant_id: Option<String>,
    /// Identity of the principal (user / service / etc.) that triggered
    /// this session. Treated as opaque.
    pub principal: Option<String>,
    /// Logical identifier of the agent template / definition the session
    /// was instantiated from.
    pub agent_template_id: Option<String>,
    /// Distributed-trace correlation id propagated through this
    /// session's events.
    pub correlation_id: Option<String>,
    /// Optional FIFO retention caps on the session's in-memory stores.
    /// Cap any subset; missing fields keep the unbounded default for
    /// that store. Use this to stop long-running cluster sessions
    /// from leaking memory in the run / trace / subagent trackers.
    pub retention_limits: Option<RetentionLimitsObject>,
    /// Automatically save the session to the configured store after each turn (default: false).
    pub auto_save: Option<bool>,
    /// AHP transport configuration for external agent supervision.
    ///
    /// Pass an AHP transport instance to enable Agent Harness Protocol supervision.
    /// All agent lifecycle events will be forwarded to the harness server.
    ///
    /// ```js
    /// // Stdio transport (local child process)
    /// agent.session('.', { ahpTransport: new StdioTransport('python', ['ahp_server.py']) });
    ///
    /// // HTTP transport (remote server)
    /// agent.session('.', { ahpTransport: new HttpTransport('http://localhost:8080/ahp') });
    ///
    /// // WebSocket transport (bidirectional streaming)
    /// agent.session('.', { ahpTransport: new WebSocketTransport('ws://localhost:8080/ahp') });
    ///
    /// // Unix socket transport (local IPC)
    /// agent.session('.', { ahpTransport: new UnixSocketTransport('/tmp/ahp.sock') });
    /// ```
    pub ahp_transport: Option<JsAhpTransport>,
    /// HITL confirmation policy configuration.
    ///
    /// Pass a confirmation policy to enable Human-in-the-Loop confirmation for tool execution.
    /// When enabled, tools that require confirmation will emit ConfirmationRequired events
    /// and wait for user approval before executing.
    ///
    /// ```js
    /// agent.session('.', {
    ///   confirmationPolicy: {
    ///     enabled: true,
    ///     defaultTimeoutMs: 30000,
    ///     timeoutAction: 'reject'
    ///   }
    /// });
    /// ```
    pub confirmation_policy: Option<ConfirmationPolicy>,
    /// Maximum execution time in milliseconds.
    ///
    /// When set, the execution loop will abort if it exceeds this duration.
    /// This prevents runaway executions and excessive API costs.
    ///
    /// ```js
    /// agent.session('.', {
    ///   maxExecutionTimeMs: 300000  // 5 minutes
    /// });
    /// ```
    pub max_execution_time_ms: Option<f64>,
}

/// Retention limits for large tool/program artifacts.
#[napi(object)]
#[derive(Clone)]
pub struct ArtifactStoreLimits {
    /// Maximum number of artifacts retained by a session.
    pub max_artifacts: Option<f64>,
    /// Maximum total artifact content bytes retained by a session.
    pub max_bytes: Option<f64>,
}

/// A single message in conversation history.
#[napi(object)]
#[derive(Clone)]
pub struct MessageObject {
    pub role: String,
    pub content: Vec<ContentBlockObject>,
}

/// A content block within a message.
#[napi(object)]
#[derive(Clone)]
pub struct ContentBlockObject {
    #[napi(js_name = "type")]
    pub block_type: String,
    /// Text content (for "text" blocks).
    pub text: Option<String>,
    /// Tool use ID (for "tool_use" blocks).
    pub id: Option<String>,
    /// Tool name (for "tool_use" blocks).
    pub name: Option<String>,
    /// Tool input (for "tool_use" blocks).
    pub input: Option<serde_json::Value>,
    /// Tool use ID reference (for "tool_result" blocks).
    pub tool_use_id: Option<String>,
    /// Tool result content (for "tool_result" blocks).
    pub result_content: Option<String>,
    /// Whether this is an error result (for "tool_result" blocks).
    pub is_error: Option<bool>,
}

/// An image attachment for multi-modal prompts.
#[napi(object)]
#[derive(Clone)]
pub struct AttachmentObject {
    /// Raw image bytes.
    pub data: napi::bindgen_prelude::Buffer,
    /// MIME type (e.g., "image/jpeg", "image/png").
    pub media_type: String,
}

// ============================================================================
// SessionQueueConfig
// ============================================================================

/// Configuration for the optional advanced session lane queue.
///
/// Ordinary sessions do not initialize queue infrastructure. Use this only for
/// explicit external/hybrid dispatch, priority experiments, or operational integrations.
#[napi(object)]
#[derive(Clone, Default)]
pub struct SessionQueueConfig {
    /// Max concurrency for Query lane (default: 4).
    pub query_concurrency: Option<u32>,
    /// Max concurrency for Execute lane (default: 2).
    pub execute_concurrency: Option<u32>,
    /// Max concurrency for Generate lane (default: 1).
    pub generate_concurrency: Option<u32>,
    /// Enable dead letter queue.
    pub enable_dlq: Option<bool>,
    /// Max DLQ size (default: 1000).
    pub dlq_max_size: Option<u32>,
    /// Enable metrics collection.
    pub enable_metrics: Option<bool>,
    /// Enable queue alerts.
    pub enable_alerts: Option<bool>,
    /// Default command timeout (ms).
    pub timeout_ms: Option<u32>,
    /// Enable all features with sensible defaults.
    pub enable_all_features: Option<bool>,
    /// Per-lane handler config. Keys: "control", "query", "execute", "generate".
    /// Values: LaneHandlerConfig with mode ("internal"/"external"/"hybrid") and timeoutMs.
    pub lane_handlers: Option<std::collections::HashMap<String, LaneHandlerConfig>>,
}

/// Result of an external task completion.
#[napi(object)]
#[derive(Clone)]
pub struct ExternalTaskResult {
    pub success: bool,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
}

/// Lane handler configuration.
#[napi(object)]
#[derive(Clone)]
pub struct LaneHandlerConfig {
    /// "internal", "external", or "hybrid"
    pub mode: String,
    /// Timeout for external processing (ms).
    pub timeout_ms: Option<u32>,
}

/// Queue statistics.
#[napi(object)]
#[derive(Clone)]
pub struct QueueStats {
    pub total_pending: u32,
    pub total_active: u32,
    pub external_pending: u32,
}

fn js_queue_config_to_rust(config: &SessionQueueConfig) -> napi::Result<RustSessionQueueConfig> {
    let mut c = if config.enable_all_features.unwrap_or(false) {
        RustSessionQueueConfig::default().with_lane_features()
    } else {
        RustSessionQueueConfig::default()
    };
    if let Some(n) = config.query_concurrency {
        c.query_max_concurrency = n as usize;
    }
    if let Some(n) = config.execute_concurrency {
        c.execute_max_concurrency = n as usize;
    }
    if let Some(n) = config.generate_concurrency {
        c.generate_max_concurrency = n as usize;
    }
    if let Some(true) = config.enable_dlq {
        c = c.with_dlq(config.dlq_max_size.map(|n| n as usize));
    }
    if let Some(true) = config.enable_metrics {
        c = c.with_metrics();
    }
    if let Some(true) = config.enable_alerts {
        c = c.with_alerts();
    }
    if let Some(ms) = config.timeout_ms {
        c = c.with_timeout(ms as u64);
    }
    if let Some(ref handlers) = config.lane_handlers {
        for (lane_str, handler) in handlers {
            let lane = parse_lane(lane_str)?;
            let mode = parse_handler_mode(&handler.mode)?;
            let lane_cfg = RustLaneHandlerConfig {
                mode,
                timeout_ms: handler.timeout_ms.map(|ms| ms as u64).unwrap_or(60_000),
            };
            c.lane_handlers.insert(lane, lane_cfg);
        }
    }
    Ok(c)
}

fn parse_lane(lane: &str) -> napi::Result<RustSessionLane> {
    match lane.trim().to_ascii_lowercase().as_str() {
        "control" => Ok(RustSessionLane::Control),
        "query" => Ok(RustSessionLane::Query),
        "execute" => Ok(RustSessionLane::Execute),
        "generate" => Ok(RustSessionLane::Generate),
        _ => Err(napi::Error::from_reason(format!(
            "Invalid lane '{}'. Must be: control, query, execute, or generate",
            lane
        ))),
    }
}

fn parse_handler_mode(mode: &str) -> napi::Result<RustTaskHandlerMode> {
    match mode.trim().to_ascii_lowercase().as_str() {
        "internal" => Ok(RustTaskHandlerMode::Internal),
        "external" => Ok(RustTaskHandlerMode::External),
        "hybrid" => Ok(RustTaskHandlerMode::Hybrid),
        _ => Err(napi::Error::from_reason(format!(
            "Invalid handler mode '{}'. Must be: internal, external, or hybrid",
            mode
        ))),
    }
}

fn s3_config_to_core(js: &JsS3BackendConfig) -> a3s_code_core::S3BackendConfig {
    let mut cfg = a3s_code_core::S3BackendConfig::new(
        js.bucket.clone(),
        js.prefix.clone(),
        js.access_key_id.clone(),
        js.secret_access_key.clone(),
    );
    if let Some(ref endpoint) = js.endpoint {
        cfg = cfg.endpoint(endpoint.clone());
    }
    if let Some(ref region) = js.region {
        cfg = cfg.region(region.clone());
    }
    if let Some(ref token) = js.session_token {
        cfg = cfg.session_token(token.clone());
    }
    if let Some(force) = js.force_path_style {
        cfg = cfg.force_path_style(force);
    }
    if let Some(n) = js.max_read_bytes {
        cfg = cfg.max_read_bytes(n.max(0) as u64);
    }
    if let Some(on) = js.search_enabled {
        cfg = cfg.enable_search(on);
    }
    if let Some(n) = js.max_objects_scanned {
        cfg = cfg.max_objects_scanned(n.max(0) as usize);
    }
    if let Some(n) = js.max_grep_bytes_per_object {
        cfg = cfg.max_grep_bytes_per_object(n.max(0) as u64);
    }
    if let Some(n) = js.search_concurrency {
        cfg = cfg.search_concurrency(n.max(0) as usize);
    }
    cfg
}

fn remote_git_config_to_core(
    js: &JsRemoteGitBackendConfig,
) -> a3s_code_core::RemoteGitBackendConfig {
    let mut cfg =
        a3s_code_core::RemoteGitBackendConfig::new(js.base_url.clone(), js.repo_id.clone());
    if let Some(ref t) = js.bearer_token {
        cfg = cfg.bearer_token(t.clone());
    }
    if let Some(ref p) = js.client_cert_pem {
        cfg = cfg.client_cert_pem(std::path::PathBuf::from(p));
    }
    if let Some(ref p) = js.client_key_pem {
        cfg = cfg.client_key_pem(std::path::PathBuf::from(p));
    }
    if let Some(ms) = js.request_timeout_ms {
        cfg = cfg.request_timeout(std::time::Duration::from_millis(ms.max(0) as u64));
    }
    if let Some(n) = js.max_diff_bytes {
        cfg = cfg.max_diff_bytes(n.max(0) as u64);
    }
    if let Some(n) = js.max_log_entries {
        cfg = cfg.max_log_entries(n.max(0) as usize);
    }
    cfg
}

fn js_optional_usize(
    value: Option<f64>,
    field_name: &str,
    default_value: usize,
) -> napi::Result<usize> {
    match value {
        Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => Ok(n as usize),
        Some(_) => Err(napi::Error::from_reason(format!(
            "{field_name} must be a non-negative integer"
        ))),
        None => Ok(default_value),
    }
}

fn js_artifact_store_limits_to_rust(
    limits: ArtifactStoreLimits,
) -> napi::Result<a3s_code_core::tools::ArtifactStoreLimits> {
    let defaults = a3s_code_core::tools::ArtifactStoreLimits::default();
    Ok(a3s_code_core::tools::ArtifactStoreLimits {
        max_artifacts: js_optional_usize(
            limits.max_artifacts,
            "artifactStoreLimits.maxArtifacts",
            defaults.max_artifacts,
        )?,
        max_bytes: js_optional_usize(
            limits.max_bytes,
            "artifactStoreLimits.maxBytes",
            defaults.max_bytes,
        )?,
    })
}

fn verification_reports_from_value(
    reports: serde_json::Value,
) -> napi::Result<Vec<RustVerificationReport>> {
    let reports = match reports {
        serde_json::Value::Array(_) => serde_json::from_value(reports),
        serde_json::Value::Object(_) => {
            serde_json::from_value::<RustVerificationReport>(reports).map(|report| vec![report])
        }
        _ => {
            return Err(napi::Error::from_reason(
                "verification reports must be an array or object",
            ));
        }
    };
    reports.map_err(|e| napi::Error::from_reason(format!("Invalid verification report: {e}")))
}

fn js_auto_delegation_to_rust(
    options: AutoDelegationOptions,
) -> a3s_code_core::AutoDelegationConfig {
    let mut config = a3s_code_core::AutoDelegationConfig::default();
    if let Some(enabled) = options.enabled {
        config.enabled = enabled;
    }
    if let Some(auto_parallel) = options.auto_parallel {
        config.auto_parallel = auto_parallel;
    }
    if let Some(min_confidence) = options.min_confidence {
        config.min_confidence = (min_confidence as f32).clamp(0.0, 1.0);
    }
    if let Some(max_tasks) = options.max_tasks {
        config.max_tasks = (max_tasks as usize).max(1);
    }
    config
}

/// Build RustSessionOptions from JS SessionOptions.
fn js_session_options_to_rust(options: Option<SessionOptions>) -> napi::Result<RustSessionOptions> {
    let Some(o) = options else {
        return Ok(RustSessionOptions::new());
    };
    let mut opts = RustSessionOptions::new();
    if let Some(model) = o.model {
        opts = opts.with_model(model);
    }
    if o.builtin_skills.unwrap_or(false) {
        opts = opts.with_builtin_skills();
    }
    if let Some(dirs) = o.skill_dirs {
        for d in dirs {
            opts = opts.with_skills_from_dir(d);
        }
    }
    if let Some(dirs) = o.agent_dirs {
        for d in dirs {
            opts = opts.with_agent_dir(d);
        }
    }
    if let Some(workers) = o.worker_agents {
        for worker in workers {
            opts = opts.with_worker_agent(js_worker_agent_spec_to_rust(worker)?);
        }
    }
    if let Some(qc) = o.queue_config {
        opts = opts.with_queue_config(js_queue_config_to_rust(&qc)?);
    }
    if let Some(policy) = o.permission_policy {
        opts = opts.with_permission_policy(js_permission_policy_to_rust(policy)?);
    }
    opts = apply_planning_mode(opts, o.planning_mode.as_deref(), o.planning)?;
    if o.goal_tracking.unwrap_or(false) {
        opts = opts.with_goal_tracking(true);
    }
    if let Some(n) = o.max_parse_retries {
        opts = opts.with_parse_retries(n);
    }
    if let Some(ms) = o.tool_timeout_ms {
        opts = opts.with_tool_timeout(ms as u64);
    }
    if let Some(n) = o.circuit_breaker_threshold {
        opts = opts.with_circuit_breaker(n);
    }
    if o.auto_compact.unwrap_or(false) {
        opts = opts.with_auto_compact(true);
    }
    if let Some(t) = o.auto_compact_threshold {
        opts = opts.with_auto_compact_threshold(t as f32);
    }
    if let Some(limits) = o.artifact_store_limits {
        opts = opts.with_artifact_store_limits(js_artifact_store_limits_to_rust(limits)?);
    }
    if let Some(ref store) = o.memory_store {
        if store.backend == "file" {
            if let Some(ref dir) = store.dir {
                opts = opts.with_file_memory(dir);
            }
        }
    }
    if let Some(ref store) = o.session_store {
        match store.backend.as_str() {
            "file" => {
                if let Some(ref dir) = store.dir {
                    opts = opts.with_file_session_store(dir);
                }
            }
            "memory" => {
                let s: std::sync::Arc<dyn a3s_code_core::store::SessionStore> =
                    std::sync::Arc::new(a3s_code_core::store::MemorySessionStore::new());
                opts = opts.with_session_store(s);
            }
            _ => {}
        }
    }
    if let Some(ref sec) = o.security_provider {
        if sec.kind.is_empty() || sec.kind == "default" {
            opts = opts.with_default_security();
        }
    }
    if let Some(ref backend) = o.workspace_backend {
        let services: std::sync::Arc<a3s_code_core::WorkspaceServices> = match backend.kind.as_str()
        {
            "" | "local" => {
                let root = backend.root.as_ref().ok_or_else(|| {
                    napi::Error::from_reason("LocalWorkspaceBackend requires a root path")
                })?;
                a3s_code_core::WorkspaceServices::local(root.clone())
            }
            "s3" => {
                let s3_config = backend.s3.as_ref().ok_or_else(|| {
                    napi::Error::from_reason(
                        "S3WorkspaceBackend requires the `s3` configuration field",
                    )
                })?;
                a3s_code_core::WorkspaceServices::s3(s3_config_to_core(s3_config))
            }
            other => {
                return Err(napi::Error::from_reason(format!(
                    "Unsupported workspace backend kind '{other}'"
                )));
            }
        };
        let services = if let Some(ref git_cfg) = o.remote_git {
            services
                .with_remote_git(remote_git_config_to_core(git_cfg))
                .map_err(|e| napi::Error::from_reason(format!("with_remote_git: {e}")))?
        } else {
            services
        };
        opts = opts.with_workspace_backend(services);
    } else if o.remote_git.is_some() {
        // `remoteGit` needs a base `WorkspaceServices` to attach to. The
        // session path is not available here (it's the first argument to
        // `agent.session(path, options)`, applied later by the runtime),
        // so we cannot synthesize a local backend on the user's behalf.
        return Err(napi::Error::from_reason(
            "remoteGit requires workspaceBackend to be set; pass a LocalWorkspaceBackend or S3WorkspaceBackend alongside it",
        ));
    }
    // Build prompt slots if any slot is set
    if o.role.is_some() || o.guidelines.is_some() || o.response_style.is_some() || o.extra.is_some()
    {
        let slots = a3s_code_core::SystemPromptSlots {
            style: None,
            role: o.role,
            guidelines: o.guidelines,
            response_style: o.response_style,
            extra: o.extra,
        };
        opts = opts.with_prompt_slots(slots);
    }
    // Inline skills registered without skill files
    if let Some(inline_skills) = o.inline_skills {
        if !inline_skills.is_empty() {
            let registry = a3s_code_core::skills::SkillRegistry::new();
            for skill in inline_skills {
                let raw = format!(
                    "---\nname: {}\nkind: {}\n---\n{}",
                    skill.name, skill.kind, skill.content
                );
                if let Some(parsed) = a3s_code_core::skills::Skill::parse(&raw) {
                    registry.register_unchecked(std::sync::Arc::new(parsed));
                }
            }
            opts = opts.with_skill_registry(std::sync::Arc::new(registry));
        }
    }
    if let Some(r) = o.max_tool_rounds {
        opts = opts.with_max_tool_rounds(r as usize);
    }
    if let Some(max_parallel_tasks) = o.max_parallel_tasks {
        opts = opts.with_max_parallel_tasks(max_parallel_tasks as usize);
    }
    if let Some(auto_delegation) = o.auto_delegation {
        opts = opts.with_auto_delegation(js_auto_delegation_to_rust(auto_delegation));
    }
    if let Some(auto_parallel) = o.auto_parallel {
        opts = opts.with_auto_parallel_delegation(auto_parallel);
    }
    if let Some(id) = o.session_id {
        opts = opts.with_session_id(id);
    }
    if let Some(t) = o.tenant_id {
        opts = opts.with_tenant_id(t);
    }
    if let Some(p) = o.principal {
        opts = opts.with_principal(p);
    }
    if let Some(t) = o.agent_template_id {
        opts = opts.with_agent_template_id(t);
    }
    if let Some(c) = o.correlation_id {
        opts = opts.with_correlation_id(c);
    }
    if let Some(rl) = o.retention_limits {
        let mut limits = a3s_code_core::retention::SessionRetentionLimits::new();
        if let Some(n) = rl.max_runs_retained {
            limits.max_runs_retained = Some(n as usize);
        }
        if let Some(n) = rl.max_events_per_run {
            limits.max_events_per_run = Some(n as usize);
        }
        if let Some(n) = rl.max_trace_events {
            limits.max_trace_events = Some(n as usize);
        }
        if let Some(n) = rl.max_terminal_subagent_tasks {
            limits.max_terminal_subagent_tasks = Some(n as usize);
        }
        opts = opts.with_retention_limits(limits);
    }
    if o.auto_save.unwrap_or(false) {
        opts = opts.with_auto_save(true);
    }
    if let Some(t) = o.temperature {
        opts = opts.with_temperature(t as f32);
    }
    if let Some(budget) = o.thinking_budget {
        opts = opts.with_thinking_budget(budget as usize);
    }
    if let Some(enabled) = o.continuation_enabled {
        opts = opts.with_continuation(enabled);
    }
    if let Some(turns) = o.max_continuation_turns {
        opts = opts.with_max_continuation_turns(turns);
    }

    // HITL confirmation policy configuration
    if let Some(policy) = o.confirmation_policy {
        opts = opts.with_confirmation_policy(js_confirmation_policy_to_rust(policy)?);
    }

    // Maximum execution time configuration
    if let Some(timeout_ms) = o.max_execution_time_ms {
        opts.max_execution_time_ms = Some(timeout_ms as u64);
    }

    // AHP transport configuration
    #[cfg(feature = "ahp")]
    if let Some(ref transport) = o.ahp_transport {
        use a3s_code_core::ahp::{AhpHookExecutor, AhpTransport, AuthConfig};

        let ahp_transport = match transport.kind.as_str() {
            "stdio" => {
                if let (Some(program), Some(args)) = (&transport.program, &transport.args) {
                    Some(AhpTransport::Stdio {
                        program: program.clone(),
                        args: args.clone(),
                    })
                } else {
                    None
                }
            }
            "http" => {
                if let Some(url) = &transport.url {
                    let auth = transport
                        .auth_token
                        .as_ref()
                        .map(|t| AuthConfig::bearer(t.clone()));
                    Some(AhpTransport::Http {
                        url: url.clone(),
                        auth,
                    })
                } else {
                    None
                }
            }
            "websocket" => {
                if let Some(url) = &transport.url {
                    let auth = transport
                        .auth_token
                        .as_ref()
                        .map(|t| AuthConfig::bearer(t.clone()));
                    Some(AhpTransport::WebSocket {
                        url: url.clone(),
                        auth,
                    })
                } else {
                    None
                }
            }
            "unix_socket" => {
                #[cfg(unix)]
                {
                    transport
                        .path
                        .as_ref()
                        .map(|path| AhpTransport::UnixSocket { path: path.clone() })
                }
                #[cfg(not(unix))]
                {
                    None
                }
            }
            _ => None,
        };

        if let Some(ahp_transport) = ahp_transport {
            match get_runtime().block_on(AhpHookExecutor::new(ahp_transport)) {
                Ok(executor) => {
                    let executor = std::sync::Arc::new(executor);
                    opts = opts.with_hook_executor(executor.clone());
                }
                Err(e) => {
                    eprintln!(
                        "a3s-code: failed to create AHP executor: {} — continuing without AHP",
                        e
                    );
                }
            }
        }
    }

    Ok(opts)
}

fn apply_planning_mode(
    opts: RustSessionOptions,
    planning_mode: Option<&str>,
    planning: Option<bool>,
) -> napi::Result<RustSessionOptions> {
    if let Some(mode) = planning_mode {
        return Ok(opts.with_planning_mode(parse_planning_mode(mode)?));
    }

    if let Some(enabled) = planning {
        Ok(opts.with_planning(enabled))
    } else {
        Ok(opts)
    }
}

fn parse_planning_mode(mode: &str) -> napi::Result<RustPlanningMode> {
    match mode.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(RustPlanningMode::Auto),
        "enabled" | "enable" | "on" | "force" | "forced" | "true" => Ok(RustPlanningMode::Enabled),
        "disabled" | "disable" | "off" | "false" => Ok(RustPlanningMode::Disabled),
        _ => Err(napi::Error::from_reason(format!(
            "Invalid planningMode '{}'. Must be: auto, enabled, or disabled",
            mode
        ))),
    }
}

fn parse_permission_decision(value: Option<String>) -> napi::Result<RustPermissionDecision> {
    match value
        .as_deref()
        .unwrap_or("ask")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "allow" => Ok(RustPermissionDecision::Allow),
        "deny" => Ok(RustPermissionDecision::Deny),
        "ask" => Ok(RustPermissionDecision::Ask),
        other => Err(napi::Error::from_reason(format!(
            "Invalid permission defaultDecision '{}'. Must be: allow, deny, or ask",
            other
        ))),
    }
}

fn parse_timeout_action(value: Option<&str>) -> napi::Result<RustTimeoutAction> {
    match value
        .unwrap_or("reject")
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_")
        .as_str()
    {
        "reject" => Ok(RustTimeoutAction::Reject),
        "auto_approve" | "autoapprove" => Ok(RustTimeoutAction::AutoApprove),
        other => Err(napi::Error::from_reason(format!(
            "Invalid confirmation timeoutAction '{}'. Must be: reject or auto_approve",
            other
        ))),
    }
}

fn js_confirmation_policy_to_rust(
    policy: ConfirmationPolicy,
) -> napi::Result<RustConfirmationPolicy> {
    let mut rust_policy = if policy.enabled.unwrap_or(false) {
        RustConfirmationPolicy::enabled()
    } else {
        RustConfirmationPolicy::default()
    };

    if let Some(timeout_ms) = policy.default_timeout_ms {
        rust_policy = rust_policy.with_timeout(
            timeout_ms as u64,
            parse_timeout_action(policy.timeout_action.as_deref())?,
        );
    } else {
        parse_timeout_action(policy.timeout_action.as_deref())?;
    }

    if let Some(lanes) = policy.yolo_lanes {
        let yolo_lanes = lanes
            .iter()
            .map(|lane| parse_lane(lane))
            .collect::<napi::Result<Vec<_>>>()?;
        if !yolo_lanes.is_empty() {
            rust_policy = rust_policy.with_yolo_lanes(yolo_lanes);
        }
    }

    Ok(rust_policy)
}

fn js_permission_policy_to_rust(policy: PermissionPolicy) -> napi::Result<RustPermissionPolicy> {
    Ok(RustPermissionPolicy {
        deny: policy
            .deny
            .unwrap_or_default()
            .into_iter()
            .map(|rule| RustPermissionRule::new(&rule))
            .collect(),
        allow: policy
            .allow
            .unwrap_or_default()
            .into_iter()
            .map(|rule| RustPermissionRule::new(&rule))
            .collect(),
        ask: policy
            .ask
            .unwrap_or_default()
            .into_iter()
            .map(|rule| RustPermissionRule::new(&rule))
            .collect(),
        default_decision: parse_permission_decision(policy.default_decision)?,
        enabled: policy.enabled.unwrap_or(true),
    })
}

fn js_worker_agent_spec_to_rust(spec: WorkerAgentSpec) -> napi::Result<RustWorkerAgentSpec> {
    if spec.name.trim().is_empty() {
        return Err(napi::Error::from_reason("worker agent name is required"));
    }
    if spec.description.trim().is_empty() {
        return Err(napi::Error::from_reason(
            "worker agent description is required",
        ));
    }

    let kind = parse_worker_agent_kind(spec.kind.as_deref())?;
    let mut worker = RustWorkerAgentSpec::new(kind, spec.name, spec.description);
    if spec.hidden.unwrap_or(false) {
        worker = worker.hidden(true);
    }
    if let Some(policy) = spec.permissions {
        worker = worker.with_permissions(js_permission_policy_to_rust(policy)?);
    }
    if let Some(model) = spec.model {
        worker = worker.with_model(RustAgentModelConfig::from_model_ref(model));
    }
    if let Some(prompt) = spec.prompt {
        worker = worker.with_prompt(prompt);
    }
    if let Some(max_steps) = spec.max_steps {
        worker = worker.with_max_steps(max_steps as usize);
    }
    if let Some(ci) = spec.confirmation_inheritance {
        worker = worker.with_confirmation(parse_confirmation_inheritance(&ci)?);
    }
    Ok(worker)
}

fn parse_worker_agent_kind(kind: Option<&str>) -> napi::Result<RustWorkerAgentKind> {
    kind.unwrap_or("custom")
        .parse::<RustWorkerAgentKind>()
        .map_err(|e| napi::Error::from_reason(e.to_string()))
}

fn parse_confirmation_inheritance(
    value: &str,
) -> napi::Result<a3s_code_core::subagent::ConfirmationInheritance> {
    use a3s_code_core::subagent::ConfirmationInheritance;
    match value {
        "auto_approve" => Ok(ConfirmationInheritance::AutoApprove),
        "deny_on_ask" => Ok(ConfirmationInheritance::DenyOnAsk),
        "inherit_parent" => Ok(ConfirmationInheritance::InheritParent),
        other => Err(napi::Error::from_reason(format!(
            "invalid confirmation_inheritance: '{}' (expected: auto_approve, deny_on_ask, inherit_parent)",
            other
        ))),
    }
}

fn confirmation_inheritance_to_js(ci: &a3s_code_core::subagent::ConfirmationInheritance) -> String {
    use a3s_code_core::subagent::ConfirmationInheritance;
    match ci {
        ConfirmationInheritance::AutoApprove => "auto_approve".to_string(),
        ConfirmationInheritance::DenyOnAsk => "deny_on_ask".to_string(),
        ConfirmationInheritance::InheritParent => "inherit_parent".to_string(),
    }
}

fn rust_agent_definition_to_js(def: RustAgentDefinition) -> AgentDefinition {
    AgentDefinition {
        name: def.name,
        description: def.description,
        native: def.native,
        hidden: def.hidden,
        model: def.model.map(|model| model.model_ref()),
        prompt: def.prompt,
        max_steps: def.max_steps.map(|steps| steps as u32),
        confirmation_inheritance: def
            .confirmation_inheritance
            .as_ref()
            .map(confirmation_inheritance_to_js),
    }
}

// ============================================================================
// Agent
// ============================================================================

/// AI coding agent. Create with `Agent.create()`, then call `agent.session()`.
#[napi]
pub struct Agent {
    inner: Arc<RustAgent>,
}

#[napi]
impl Agent {
    /// Create an Agent from a config file path or inline config string.
    ///
    /// Accepts ACL-compatible config files (.acl) or inline config strings.
    /// JSON config is not supported.
    ///
    /// @param configSource - Path to a config file (.acl), or inline config string
    #[napi(factory)]
    pub async fn create(config_source: String) -> napi::Result<Self> {
        let agent = get_runtime()
            .spawn(async move { RustAgent::new(config_source).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Failed to create agent: {e}")))?;

        Ok(Self {
            inner: Arc::new(agent),
        })
    }

    /// Re-fetch tool definitions from all connected global MCP servers and
    /// update the agent-level cache.
    ///
    /// New sessions created after this call will see the refreshed tool list.
    /// Existing sessions are unaffected.
    #[napi]
    pub async fn refresh_mcp_tools(&self) -> napi::Result<()> {
        let agent = self.inner.clone();
        agent
            .refresh_mcp_tools()
            .await
            .map_err(|e| napi::Error::from_reason(format!("refresh_mcp_tools failed: {e}")))?;
        Ok(())
    }

    /// Bind to a workspace directory, returning a Session.
    ///
    /// @param workspace - Path to the workspace directory
    /// @param options - Optional session overrides
    #[napi]
    pub fn session(
        &self,
        workspace: String,
        options: Option<SessionOptions>,
    ) -> napi::Result<Session> {
        let rust_opts = js_session_options_to_rust(options)?;
        let session = self
            .inner
            .session(workspace, Some(rust_opts))
            .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
        Ok(Session {
            inner: Arc::new(session),
        })
    }

    /// Resume a previously saved session by ID.
    ///
    /// `options.sessionStore` must be set to a `FileSessionStore` (or `MemorySessionStore`)
    /// that points to the directory where the session was originally saved.
    ///
    /// ```js
    /// const session = agent.resumeSession('my-session', {
    ///   sessionStore: new FileSessionStore('./sessions'),
    /// });
    /// ```
    ///
    /// @param sessionId - The session ID to resume
    /// @param options - Session options; `sessionStore` is required
    #[napi]
    pub fn resume_session(
        &self,
        session_id: String,
        options: SessionOptions,
    ) -> napi::Result<Session> {
        let opts = js_session_options_to_rust(Some(options))?;
        let session = self
            .inner
            .resume_session(&session_id, opts)
            .map_err(|e| napi::Error::from_reason(format!("Failed to resume session: {e}")))?;
        Ok(Session {
            inner: Arc::new(session),
        })
    }

    /// Create a session pre-configured from a named agent definition.
    ///
    /// Loads the agent by name from built-in agents and optionally from
    /// additional directories, then creates a session with the agent's
    /// permissions, system prompt, model, and step limit applied.
    ///
    /// @param workspace - Path to the workspace directory
    /// @param agentName - Name of the agent to load (e.g. "explore", "general")
    /// @param agentDirs - Optional directories to scan for agent files
    /// @param options - Optional session overrides layered on top of the agent definition
    #[napi]
    pub fn session_for_agent(
        &self,
        workspace: String,
        agent_name: String,
        agent_dirs: Option<Vec<String>>,
        options: Option<SessionOptions>,
    ) -> napi::Result<Session> {
        let registry = a3s_code_core::subagent::AgentRegistry::new();
        for dir in agent_dirs.unwrap_or_default() {
            let agents = a3s_code_core::subagent::load_agents_from_dir(std::path::Path::new(&dir));
            for agent in agents {
                registry.register(agent);
            }
        }
        let def = registry
            .get(&agent_name)
            .ok_or_else(|| napi::Error::from_reason(format!("agent '{}' not found", agent_name)))?;
        let session = self
            .inner
            .session_for_agent(
                workspace,
                &def,
                options
                    .map(|o| js_session_options_to_rust(Some(o)))
                    .transpose()?,
            )
            .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
        Ok(Session {
            inner: Arc::new(session),
        })
    }

    /// Create a session pre-configured from a disposable worker spec.
    ///
    /// This avoids writing temporary agent files for one-off cattle workers.
    ///
    /// @param workspace - Path to the workspace directory
    /// @param worker - Worker spec to compile into an agent definition
    /// @param options - Optional session overrides layered on top of the worker definition
    #[napi]
    pub fn session_for_worker(
        &self,
        workspace: String,
        worker: WorkerAgentSpec,
        options: Option<SessionOptions>,
    ) -> napi::Result<Session> {
        let worker = js_worker_agent_spec_to_rust(worker)?;
        let session = self
            .inner
            .session_for_worker(
                workspace,
                worker,
                options
                    .map(|o| js_session_options_to_rust(Some(o)))
                    .transpose()?,
            )
            .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
        Ok(Session {
            inner: Arc::new(session),
        })
    }

    /// List session IDs for every live session created from this agent.
    ///
    /// Sessions that have been dropped (no JS-side references remain) are
    /// pruned lazily on each call. Result is sorted for stable output.
    #[napi]
    pub async fn list_sessions(&self) -> Vec<String> {
        self.inner.list_sessions().await
    }

    /// Close a specific live session by its session ID.
    ///
    /// Returns `true` when a live session with the given id was found and
    /// transitioned from open to closed by this call; `false` when no live
    /// session has that id, or when it was already closed.
    ///
    /// Equivalent to calling `session.close()` directly, but does not
    /// require holding a reference to the session — handy for control-plane
    /// code that only knows the session ID.
    #[napi]
    pub async fn close_session(&self, session_id: String) -> bool {
        self.inner.close_session(&session_id).await
    }

    /// Close every live session created from this agent and disconnect
    /// background resources owned by the agent (global MCP connections).
    ///
    /// After this call, `agent.session(...)` and `agent.resumeSession(...)`
    /// reject with a "Session closed" error. Idempotent.
    #[napi]
    pub async fn close(&self) {
        self.inner.close().await
    }

    /// Whether `close()` has been called on this agent.
    #[napi]
    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }

    /// Disconnect every global MCP server idle longer than
    /// `idleThresholdMs`, returning the names disconnected. The server's
    /// registered config is kept — a later tool call reconnects on
    /// demand. Call periodically (e.g. every 60s with a 5-min threshold)
    /// from a host-side sweeper to release file descriptors and
    /// background workers from quiet MCP servers in long-running
    /// deployments.
    #[napi]
    pub async fn disconnect_idle_mcp(&self, idle_threshold_ms: i64) -> Vec<String> {
        self.inner
            .disconnect_idle_mcp(idle_threshold_ms.max(0) as u64)
            .await
    }
}

// ============================================================================
// McpServerStatusEntry
// ============================================================================

#[napi(object)]
#[derive(Clone)]
pub struct McpServerStatusEntry {
    pub name: String,
    pub connected: bool,
    pub tool_count: u32,
    pub error: Option<String>,
}

// ============================================================================
// Session
// ============================================================================

/// One unit of orchestrated agent work — what to run, independent of where.
#[napi(object)]
#[derive(Clone)]
pub struct AgentStepSpecObject {
    /// Stable id for this step (assigned by the caller).
    pub task_id: String,
    /// Registry key of the agent to run (e.g. "explore", "review").
    pub agent: String,
    /// Short label for display/tracking.
    pub description: String,
    /// Instruction handed to the child agent.
    pub prompt: String,
    /// Optional per-step tool-round cap.
    pub max_steps: Option<u32>,
    /// Optional parent session id for event correlation.
    pub parent_session_id: Option<String>,
    /// When set, the step must return JSON conforming to this schema; the
    /// validated object lands in `StepOutcomeObject.structured`.
    pub output_schema: Option<serde_json::Value>,
}

impl From<AgentStepSpecObject> for RustAgentStepSpec {
    fn from(o: AgentStepSpecObject) -> Self {
        RustAgentStepSpec {
            task_id: o.task_id,
            agent: o.agent,
            description: o.description,
            prompt: o.prompt,
            max_steps: o.max_steps.map(|n| n as usize),
            parent_session_id: o.parent_session_id,
            output_schema: o.output_schema,
        }
    }
}

/// The result of running one orchestrated step.
#[napi(object)]
#[derive(Clone)]
pub struct StepOutcomeObject {
    pub task_id: String,
    pub session_id: String,
    pub agent: String,
    pub output: String,
    pub success: bool,
    /// Schema-validated structured output, when the step requested one.
    pub structured: Option<serde_json::Value>,
}

impl From<RustStepOutcome> for StepOutcomeObject {
    fn from(o: RustStepOutcome) -> Self {
        StepOutcomeObject {
            task_id: o.task_id,
            session_id: o.session_id,
            agent: o.agent,
            output: o.output,
            success: o.success,
            structured: o.structured,
        }
    }
}

/// Workspace-bound session. All LLM and tool operations happen here.
#[napi]
pub struct Session {
    inner: Arc<RustAgentSession>,
}

#[napi]
impl Session {
    /// Send a prompt or request and wait for the complete response.
    ///
    /// `send("prompt")` is the compact prompt-first form. `send({ prompt,
    /// history, attachments })` is the compact object-shaped form for growth.
    #[napi(
        ts_args_type = "request: string | SessionRequestOptions, history?: Array<MessageObject> | null"
    )]
    pub async fn send(
        &self,
        request: Either<String, SessionRequestOptions>,
        history: Option<Vec<MessageObject>>,
    ) -> napi::Result<AgentResult> {
        let (prompt, rust_history, rust_attachments) = session_request_parts(request, history)?;
        send_session_request(self.inner.clone(), prompt, rust_history, rust_attachments).await
    }

    /// Alias for `send(...)` with a name that matches run/replay terminology.
    #[napi(
        ts_args_type = "request: string | SessionRequestOptions, history?: Array<MessageObject> | null"
    )]
    pub async fn run(
        &self,
        request: Either<String, SessionRequestOptions>,
        history: Option<Vec<MessageObject>>,
    ) -> napi::Result<AgentResult> {
        let (prompt, rust_history, rust_attachments) = session_request_parts(request, history)?;
        send_session_request(self.inner.clone(), prompt, rust_history, rust_attachments).await
    }

    /// Resume a previously-checkpointed run on this session.
    ///
    /// Loads the latest loop checkpoint stored under `checkpointRunId`
    /// from the configured `SessionStore` and replays the agent loop
    /// from that boundary. A new run id is allocated for the resumed
    /// work; the relationship between the old and new run is host
    /// metadata.
    ///
    /// Rejects when the session has no `sessionStore` configured, or
    /// when no checkpoint exists for `checkpointRunId`.
    #[napi]
    pub async fn resume_run(&self, checkpoint_run_id: String) -> napi::Result<AgentResult> {
        let session = self.inner.clone();
        let result = get_runtime()
            .spawn(async move { session.resume_run(&checkpoint_run_id).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
        Ok(AgentResult::from(result))
    }

    /// Run `specs` as a fan-out of agent steps, bounded by the session's
    /// configured parallelism, and resolve with each step's outcome in input
    /// order. A failed step surfaces as `success: false` without failing the
    /// batch.
    #[napi]
    pub async fn parallel(
        &self,
        specs: Vec<AgentStepSpecObject>,
    ) -> napi::Result<Vec<StepOutcomeObject>> {
        let session = self.inner.clone();
        let rust_specs: Vec<RustAgentStepSpec> = specs.into_iter().map(Into::into).collect();
        let outcomes = get_runtime()
            .spawn(async move {
                let executor = session.agent_executor();
                execute_steps_parallel(executor, rust_specs, None).await
            })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?;
        Ok(outcomes.into_iter().map(StepOutcomeObject::from).collect())
    }

    /// Like `parallel`, but resumable: progress is journaled under
    /// `workflowId` via the session's `sessionStore`, so an interrupted run
    /// skips already-completed steps. Rejects when no `sessionStore` is
    /// configured.
    #[napi]
    pub async fn parallel_resumable(
        &self,
        specs: Vec<AgentStepSpecObject>,
        workflow_id: String,
    ) -> napi::Result<Vec<StepOutcomeObject>> {
        let session = self.inner.clone();
        let rust_specs: Vec<RustAgentStepSpec> = specs.into_iter().map(Into::into).collect();
        let outcomes = get_runtime()
            .spawn(async move {
                let Some(store) = session.session_store() else {
                    return Err("parallelResumable requires a sessionStore on the session");
                };
                let executor = session.agent_executor();
                Ok(execute_steps_parallel_resumable(
                    executor,
                    rust_specs,
                    &workflow_id,
                    store,
                    None,
                )
                .await)
            })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(napi::Error::from_reason)?;
        Ok(outcomes.into_iter().map(StepOutcomeObject::from).collect())
    }

    /// Send a prompt or request and get a streaming event iterator.
    ///
    /// Returns an `EventStream`. Use `for await (const event of stream)` or call `.next()` manually.
    /// When `history` is omitted, the session history and verification evidence are
    /// updated after the stream completes. Supplying `history` keeps the stream isolated.
    #[napi(
        ts_args_type = "request: string | SessionRequestOptions, history?: Array<MessageObject> | null"
    )]
    pub async fn stream(
        &self,
        request: Either<String, SessionRequestOptions>,
        history: Option<Vec<MessageObject>>,
    ) -> napi::Result<EventStream> {
        let (prompt, rust_history, rust_attachments) = session_request_parts(request, history)?;
        stream_session_request(self.inner.clone(), prompt, rust_history, rust_attachments).await
    }

    /// Send a request using the long-lived object-shaped API.
    ///
    /// Prefer this for new integrations when the call may need history,
    /// attachments, or future request options.
    #[napi(js_name = "sendRequest")]
    pub async fn send_request(&self, request: SessionRequestOptions) -> napi::Result<AgentResult> {
        let (prompt, rust_history, rust_attachments) =
            session_request_parts(Either::B(request), None)?;
        send_session_request(self.inner.clone(), prompt, rust_history, rust_attachments).await
    }

    /// Stream a request using the long-lived object-shaped API.
    #[napi(js_name = "streamRequest")]
    pub async fn stream_request(
        &self,
        request: SessionRequestOptions,
    ) -> napi::Result<EventStream> {
        let (prompt, rust_history, rust_attachments) =
            session_request_parts(Either::B(request), None)?;
        stream_session_request(self.inner.clone(), prompt, rust_history, rust_attachments).await
    }

    /// Send a prompt with image attachments and wait for the complete response.
    ///
    /// @param prompt - The prompt to send
    /// @param attachments - Array of `{ data: Buffer, mediaType: string }`
    /// @param history - Optional conversation history
    #[napi]
    pub async fn send_with_attachments(
        &self,
        prompt: String,
        attachments: Vec<AttachmentObject>,
        history: Option<Vec<MessageObject>>,
    ) -> napi::Result<AgentResult> {
        let rust_attachments = js_attachments_to_rust(&attachments);
        let rust_history = history.map(|h| js_messages_to_rust(&h)).transpose()?;
        let session = self.inner.clone();
        let result = get_runtime()
            .spawn(async move {
                session
                    .send_with_attachments(&prompt, &rust_attachments, rust_history.as_deref())
                    .await
            })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Agent execution failed: {e}")))?;
        Ok(AgentResult::from(result))
    }

    /// Stream a prompt with image attachments.
    ///
    /// When `history` is omitted, the session history and verification evidence are
    /// updated after the stream completes. Supplying `history` keeps the stream isolated.
    ///
    /// @param prompt - The prompt to send
    /// @param attachments - Array of `{ data: Buffer, mediaType: string }`
    /// @param history - Optional conversation history
    #[napi]
    pub async fn stream_with_attachments(
        &self,
        prompt: String,
        attachments: Vec<AttachmentObject>,
        history: Option<Vec<MessageObject>>,
    ) -> napi::Result<EventStream> {
        let rust_attachments = js_attachments_to_rust(&attachments);
        let rust_history = history.map(|h| js_messages_to_rust(&h)).transpose()?;
        let session = self.inner.clone();
        let (rx, _handle) = get_runtime()
            .spawn(async move {
                session
                    .stream_with_attachments(&prompt, &rust_attachments, rust_history.as_deref())
                    .await
            })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Failed to start stream: {e}")))?;
        Ok(EventStream {
            rx: Arc::new(tokio::sync::Mutex::new(rx)),
            done: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Return the session's conversation history.
    #[napi]
    pub fn history(&self) -> Vec<MessageObject> {
        rust_messages_to_js(&self.inner.history())
    }

    /// Return run snapshots recorded by this session.
    #[napi]
    pub async fn runs(&self) -> napi::Result<serde_json::Value> {
        let session = self.inner.clone();
        let runs = get_runtime()
            .spawn(async move { session.runs().await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?;
        serde_json::to_value(runs)
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    /// Return a run snapshot by ID, or null when it is unknown.
    #[napi(js_name = "runSnapshot")]
    pub async fn run_snapshot(&self, run_id: String) -> napi::Result<serde_json::Value> {
        let session = self.inner.clone();
        let snapshot = get_runtime()
            .spawn(async move { session.run_snapshot(&run_id).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?;
        serde_json::to_value(snapshot)
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    /// Return recorded runtime events for a run.
    #[napi(js_name = "runEvents")]
    pub async fn run_events(&self, run_id: String) -> napi::Result<serde_json::Value> {
        let session = self.inner.clone();
        let events = get_runtime()
            .spawn(async move { session.run_events(&run_id).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?;
        serde_json::to_value(events)
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    /// Return the currently running operation, or null when idle.
    #[napi(js_name = "currentRun")]
    pub async fn current_run(&self) -> napi::Result<serde_json::Value> {
        let session = self.inner.clone();
        let snapshot = get_runtime()
            .spawn(async move {
                match session.current_run().await {
                    Some(run) => run.snapshot().await,
                    None => None,
                }
            })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?;
        serde_json::to_value(snapshot)
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    /// Return active tool calls observed for the currently running operation.
    #[napi(js_name = "activeTools")]
    pub async fn active_tools(&self) -> napi::Result<serde_json::Value> {
        let session = self.inner.clone();
        let active_tools = get_runtime()
            .spawn(async move { session.active_tools().await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?;
        serde_json::to_value(active_tools)
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    /// Look up a delegated subagent task by id. Resolves to `null` when no
    /// such task has been observed in this session.
    #[napi(js_name = "subagentTask")]
    pub async fn subagent_task(&self, task_id: String) -> napi::Result<serde_json::Value> {
        let session = self.inner.clone();
        let snapshot = get_runtime()
            .spawn(async move { session.subagent_task(&task_id).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?;
        serde_json::to_value(snapshot)
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    /// Return snapshots of every delegated subagent task observed in this
    /// session (including completed and failed ones), oldest first.
    #[napi(js_name = "subagentTasks")]
    pub async fn subagent_tasks(&self) -> napi::Result<serde_json::Value> {
        let session = self.inner.clone();
        let tasks = get_runtime()
            .spawn(async move { session.subagent_tasks().await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?;
        serde_json::to_value(tasks)
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    /// Return snapshots of subagent tasks still in `running` state.
    #[napi(js_name = "pendingSubagentTasks")]
    pub async fn pending_subagent_tasks(&self) -> napi::Result<serde_json::Value> {
        let session = self.inner.clone();
        let tasks = get_runtime()
            .spawn(async move { session.pending_subagent_tasks().await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?;
        serde_json::to_value(tasks)
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    /// Cancel an in-flight subagent task by id. Resolves to `true` when a
    /// cancellation token was found and fired, `false` when the task id
    /// is unknown or the task already finished.
    #[napi(js_name = "cancelSubagentTask")]
    pub async fn cancel_subagent_task(&self, task_id: String) -> napi::Result<bool> {
        let session = self.inner.clone();
        get_runtime()
            .spawn(async move { session.cancel_subagent_task(&task_id).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))
    }

    /// Cancel a specific run only if it is still the active run.
    #[napi(js_name = "cancelRun")]
    pub async fn cancel_run(&self, run_id: String) -> napi::Result<bool> {
        let session = self.inner.clone();
        get_runtime()
            .spawn(async move { session.cancel_run(&run_id).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))
    }

    /// Execute a tool by name, bypassing the LLM.
    #[napi]
    pub async fn tool(&self, name: String, args: serde_json::Value) -> napi::Result<ToolResult> {
        let session = self.inner.clone();
        let result = get_runtime()
            .spawn(async move { session.tool(&name, args).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Tool execution failed: {e}")))?;
        Ok(tool_result_from_core(result))
    }

    /// Delegate a bounded task to a child agent through the built-in `task` tool.
    #[napi(ts_args_type = "options: DelegateTaskOptions")]
    pub async fn task(&self, options: DelegateTaskOptions) -> napi::Result<ToolResult> {
        let args = delegate_task_options_to_args(options);

        let session = self.inner.clone();
        let result = get_runtime()
            .spawn(async move { session.tool("task", args).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Task delegation failed: {e}")))?;
        Ok(tool_result_from_core(result))
    }

    /// Delegate a bounded task to a child agent through the built-in `task` tool.
    #[napi(ts_args_type = "options: DelegateTaskOptions")]
    pub async fn delegate_task(&self, options: DelegateTaskOptions) -> napi::Result<ToolResult> {
        self.task(options).await
    }

    /// Execute several delegated child-agent tasks concurrently through `parallel_task`.
    #[napi(ts_args_type = "tasks: DelegateTaskOptions[]")]
    pub async fn tasks(&self, tasks: Vec<DelegateTaskOptions>) -> napi::Result<ToolResult> {
        let args = parallel_task_options_to_args(tasks);

        let session = self.inner.clone();
        let result = get_runtime()
            .spawn(async move { session.tool("parallel_task", args).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| {
                napi::Error::from_reason(format!("Parallel task delegation failed: {e}"))
            })?;
        Ok(tool_result_from_core(result))
    }

    /// Execute several delegated child-agent tasks concurrently through `parallel_task`.
    #[napi(
        js_name = "parallelTask",
        ts_args_type = "tasks: DelegateTaskOptions[]"
    )]
    pub async fn parallel_task(&self, tasks: Vec<DelegateTaskOptions>) -> napi::Result<ToolResult> {
        self.tasks(tasks).await
    }

    /// Run a bounded JavaScript script through the embedded QuickJS `program` tool.
    #[napi(ts_args_type = "options: ProgramScriptOptions")]
    pub async fn program(&self, options: serde_json::Value) -> napi::Result<ToolResult> {
        let args = normalize_program_script_options(options)?;
        let session = self.inner.clone();
        let result = get_runtime()
            .spawn(async move { session.tool("program", args).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Program execution failed: {e}")))?;
        Ok(tool_result_from_core(result))
    }

    /// Read a file from the workspace.
    #[napi]
    pub async fn read_file(&self, path: String) -> napi::Result<String> {
        let session = self.inner.clone();
        get_runtime()
            .spawn(async move { session.read_file(&path).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("{e}")))
    }

    /// Write a file in the workspace.
    #[napi]
    pub async fn write_file(&self, path: String, content: String) -> napi::Result<ToolResult> {
        let session = self.inner.clone();
        let result = get_runtime()
            .spawn(async move { session.write_file(&path, &content).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
        Ok(tool_result_from_core(result))
    }

    /// List a directory in the workspace.
    #[napi]
    pub async fn ls(&self, path: Option<String>) -> napi::Result<ToolResult> {
        let session = self.inner.clone();
        let result = get_runtime()
            .spawn(async move { session.ls(path.as_deref()).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
        Ok(tool_result_from_core(result))
    }

    /// Edit a file by replacing text in the workspace.
    #[napi]
    pub async fn edit_file(
        &self,
        path: String,
        old_string: String,
        new_string: String,
        replace_all: Option<bool>,
    ) -> napi::Result<ToolResult> {
        let session = self.inner.clone();
        let result = get_runtime()
            .spawn(async move {
                session
                    .edit_file(
                        &path,
                        &old_string,
                        &new_string,
                        replace_all.unwrap_or(false),
                    )
                    .await
            })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
        Ok(tool_result_from_core(result))
    }

    /// Apply a unified diff patch to a workspace file.
    #[napi]
    pub async fn patch_file(&self, path: String, diff: String) -> napi::Result<ToolResult> {
        let session = self.inner.clone();
        let result = get_runtime()
            .spawn(async move { session.patch_file(&path, &diff).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
        Ok(tool_result_from_core(result))
    }

    /// Execute a bash command in the workspace.
    #[napi]
    pub async fn bash(&self, command: String) -> napi::Result<String> {
        let session = self.inner.clone();
        get_runtime()
            .spawn(async move { session.bash(&command).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("{e}")))
    }

    /// Search for files matching a glob pattern.
    #[napi]
    pub async fn glob(&self, pattern: String) -> napi::Result<Vec<String>> {
        let session = self.inner.clone();
        get_runtime()
            .spawn(async move { session.glob(&pattern).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("{e}")))
    }

    /// Search file contents with a regex pattern.
    #[napi]
    pub async fn grep(&self, pattern: String) -> napi::Result<String> {
        let session = self.inner.clone();
        get_runtime()
            .spawn(async move { session.grep(&pattern).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("{e}")))
    }

    /// Search the web using multiple search engines.
    #[napi]
    pub async fn web_search(&self, params: JsWebSearchParams) -> napi::Result<ToolResult> {
        let session = self.inner.clone();
        let args = serde_json::json!({
            "query": params.query,
            "engines": params.engines,
            "limit": params.limit,
            "timeout": params.timeout,
            "proxy": params.proxy,
            "format": params.format,
        });
        get_runtime()
            .spawn(async move {
                session.tool("web_search", args).await.map(|r| ToolResult {
                    name: r.name,
                    output: r.output,
                    exit_code: r.exit_code,
                    metadata_json: r.metadata.and_then(|m| serde_json::to_string(&m).ok()),
                    document_runtime_json: None,
                    error_kind_json: r
                        .error_kind
                        .as_ref()
                        .and_then(|k| serde_json::to_string(k).ok()),
                })
            })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("{e}")))
    }

    /// Execute a git command.
    ///
    /// Prefer `git({ command: "status" })`; positional arguments remain for
    /// compatibility.
    #[allow(clippy::too_many_arguments)]
    #[napi(
        ts_args_type = "command: string | GitCommandOptions, subcommand?: string | null, name?: string | null, path?: string | null, newBranch?: boolean | null, base?: string | null, force?: boolean | null, maxCount?: number | null, message?: string | null, includeUntracked?: boolean | null, target?: string | null, reference?: string | null"
    )]
    pub async fn git(
        &self,
        command: Either<String, GitCommandOptions>,
        subcommand: Option<String>,
        name: Option<String>,
        path: Option<String>,
        new_branch: Option<bool>,
        base: Option<String>,
        force: Option<bool>,
        max_count: Option<u32>,
        message: Option<String>,
        include_untracked: Option<bool>,
        target: Option<String>,
        reference: Option<String>,
    ) -> napi::Result<ToolResult> {
        let mut args = match command {
            Either::A(command) => serde_json::json!({ "command": command }),
            Either::B(options) => git_command_options_to_args(options),
        };

        if args.is_object() {
            if let Some(sc) = subcommand {
                args["subcommand"] = serde_json::json!(sc);
            }
            if let Some(n) = name {
                args["name"] = serde_json::json!(n);
            }
            if let Some(p) = path {
                args["path"] = serde_json::json!(p);
            }
            if let Some(nb) = new_branch {
                args["new_branch"] = serde_json::json!(nb);
            }
            if let Some(b) = base {
                args["base"] = serde_json::json!(b);
            }
            if let Some(f) = force {
                args["force"] = serde_json::json!(f);
            }
            if let Some(mc) = max_count {
                args["max_count"] = serde_json::json!(mc);
            }
            if let Some(msg) = message {
                args["message"] = serde_json::json!(msg);
            }
            if let Some(iu) = include_untracked {
                args["include_untracked"] = serde_json::json!(iu);
            }
            if let Some(t) = target {
                args["target"] = serde_json::json!(t);
            }
            if let Some(r) = reference {
                args["ref"] = serde_json::json!(r);
            }
        }

        let session = self.inner.clone();
        let result = get_runtime()
            .spawn(async move { session.tool("git", args).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Tool execution failed: {e}")))?;
        Ok(tool_result_from_core(result))
    }

    /// Execute a git command with an object-shaped API.
    ///
    /// Preferred over the positional `git(...)` overload for new callers.
    ///
    /// ```js
    /// await session.gitCommand({ command: 'status' })
    /// await session.gitCommand({ command: 'worktree', subcommand: 'list' })
    /// ```
    #[napi(js_name = "gitCommand", ts_args_type = "args: GitCommandOptions")]
    pub async fn git_command(&self, args: serde_json::Value) -> napi::Result<ToolResult> {
        let args = normalize_git_args(args)?;
        let session = self.inner.clone();
        let result = get_runtime()
            .spawn(async move { session.tool("git", args).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Tool execution failed: {e}")))?;
        Ok(tool_result_from_core(result))
    }

    // ========================================================================
    // Advanced optional Queue API
    // ========================================================================

    /// Check if this session has an advanced lane queue configured.
    #[napi]
    pub fn has_queue(&self) -> bool {
        self.inner.has_queue()
    }

    /// Configure a lane's handler mode for explicit external/hybrid dispatch.
    ///
    /// @param lane - "control", "query", "execute", or "generate"
    /// @param config - { mode: "internal"|"external"|"hybrid", timeoutMs?: number }
    #[napi]
    pub async fn set_lane_handler(
        &self,
        lane: String,
        config: LaneHandlerConfig,
    ) -> napi::Result<()> {
        let rust_lane = parse_lane(&lane)?;
        let rust_mode = parse_handler_mode(&config.mode)?;
        let rust_config = RustLaneHandlerConfig {
            mode: rust_mode,
            timeout_ms: config.timeout_ms.unwrap_or(60000) as u64,
        };
        let session = self.inner.clone();
        get_runtime()
            .spawn(async move { session.set_lane_handler(rust_lane, rust_config).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?;
        Ok(())
    }

    /// Complete an external queue task by ID.
    ///
    /// @param taskId - The task identifier
    /// @param result - { success: boolean, result?: any, error?: string }
    /// @returns true if found, false if not found
    #[napi]
    pub async fn complete_external_task(
        &self,
        task_id: String,
        result: ExternalTaskResult,
    ) -> napi::Result<bool> {
        let ext_result = RustExternalTaskResult {
            success: result.success,
            result: result.result.unwrap_or(serde_json::json!({})),
            error: result.error,
        };
        let session = self.inner.clone();
        get_runtime()
            .spawn(async move { session.complete_external_task(&task_id, ext_result).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))
    }

    /// Get pending external queue tasks.
    #[napi]
    pub async fn pending_external_tasks(&self) -> napi::Result<serde_json::Value> {
        let session = self.inner.clone();
        let tasks = get_runtime()
            .spawn(async move { session.pending_external_tasks().await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?;
        serde_json::to_value(&tasks)
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    // ========================================================================
    // HITL confirmation API
    // ========================================================================

    /// Return pending HITL tool confirmations for this session.
    #[napi]
    pub async fn pending_confirmations(&self) -> napi::Result<Vec<PendingConfirmation>> {
        let session = self.inner.clone();
        let pending = get_runtime()
            .spawn(async move { session.pending_confirmations().await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?;
        Ok(pending.into_iter().map(PendingConfirmation::from).collect())
    }

    /// Resolve a pending HITL tool confirmation.
    ///
    /// @param toolId - Tool call ID from a `confirmation_required` event.
    /// @param approved - Whether the tool execution should proceed.
    /// @param reason - Optional human-readable reason for audit/UI display.
    /// @returns true if a pending confirmation was found and completed.
    #[napi]
    pub async fn confirm_tool_use(
        &self,
        tool_id: String,
        approved: bool,
        reason: Option<String>,
    ) -> napi::Result<bool> {
        let session = self.inner.clone();
        get_runtime()
            .spawn(async move { session.confirm_tool_use(&tool_id, approved, reason).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Confirmation failed: {e}")))
    }

    /// Cancel all pending HITL confirmations for this session.
    #[napi]
    pub async fn cancel_confirmations(&self) -> napi::Result<u32> {
        let session = self.inner.clone();
        let count = get_runtime()
            .spawn(async move { session.cancel_confirmations().await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?;
        Ok(count as u32)
    }

    /// Get optional queue statistics.
    #[napi]
    pub async fn queue_stats(&self) -> napi::Result<QueueStats> {
        let session = self.inner.clone();
        let stats = get_runtime()
            .spawn(async move { session.queue_stats().await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?;
        Ok(QueueStats {
            total_pending: stats.total_pending as u32,
            total_active: stats.total_active as u32,
            external_pending: stats.external_pending as u32,
        })
    }

    /// Return compact execution trace events recorded for this session.
    #[napi]
    pub fn trace_events(&self) -> napi::Result<serde_json::Value> {
        serde_json::to_value(self.inner.trace_events())
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    /// Return structured verification reports recorded for this session.
    #[napi]
    pub fn verification_reports(&self) -> napi::Result<serde_json::Value> {
        serde_json::to_value(self.inner.verification_reports())
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    /// Add externally produced verification reports to this session.
    #[napi]
    pub fn record_verification_reports(&self, reports: serde_json::Value) -> napi::Result<()> {
        let reports = verification_reports_from_value(reports)?;
        self.inner.record_verification_reports(reports);
        Ok(())
    }

    /// Return a structured verification summary for this session.
    #[napi]
    pub fn verification_summary(&self) -> napi::Result<serde_json::Value> {
        serde_json::to_value(self.inner.verification_summary())
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    /// Return a concise human-readable verification summary for this session.
    #[napi]
    pub fn verification_summary_text(&self) -> String {
        self.inner.verification_summary_text()
    }

    /// Run verification commands and return a structured verification report.
    #[napi]
    pub async fn verify_commands(
        &self,
        subject: String,
        commands: Vec<VerificationCommand>,
    ) -> napi::Result<serde_json::Value> {
        let rust_commands = commands
            .into_iter()
            .map(RustVerificationCommand::from)
            .collect::<Vec<_>>();
        let session = self.inner.clone();
        let report = get_runtime()
            .spawn(async move { session.verify_commands(&subject, &rust_commands).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Verification failed: {e}")))?;
        serde_json::to_value(report)
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    /// Return project-aware verification command presets for this workspace.
    #[napi]
    pub fn verification_presets(&self) -> napi::Result<serde_json::Value> {
        serde_json::to_value(self.inner.verification_presets())
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    /// Get dead letters from the optional queue's DLQ.
    #[napi]
    pub async fn dead_letters(&self) -> napi::Result<serde_json::Value> {
        let session = self.inner.clone();
        let letters = get_runtime()
            .spawn(async move { session.dead_letters().await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?;
        serde_json::to_value(&letters)
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    /// Get a detailed metrics snapshot from the queue.
    ///
    /// Returns `null` if metrics are not enabled (queue not configured or
    /// `enable_metrics` was not set in `SessionQueueConfig`).
    ///
    /// @returns Object with `counters`, `gauges`, and `histograms` maps, or null
    #[napi]
    pub async fn queue_metrics(&self) -> napi::Result<serde_json::Value> {
        let session = self.inner.clone();
        let snapshot = get_runtime()
            .spawn(async move { session.queue_metrics().await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?;
        Ok(metrics_snapshot_to_json(snapshot))
    }

    // ========================================================================
    // MCP API
    // ========================================================================

    /// Add an MCP server to this live session.
    ///
    /// Connects the server and registers all its tools immediately so the agent
    /// can call them. Tool names follow the convention `mcp__<name>__<tool>`.
    ///
    /// @param name - Server identifier (used as prefix in tool names)
    /// @param transport - Transport type: `"stdio"` (default), `"http"`, or `"streamable-http"`
    /// @param command - Executable to launch (stdio only, e.g. `"npx"`)
    /// @param args - Arguments for the command (stdio only)
    /// @param url - Server URL (http / streamable-http only)
    /// @param headers - HTTP headers (http / streamable-http only)
    /// @param env - Optional extra environment variables (stdio only)
    /// @returns Number of tools registered from the server
    #[allow(clippy::too_many_arguments)]
    #[napi]
    pub async fn add_mcp_server(
        &self,
        name: String,
        #[napi(ts_arg_type = "'stdio' | 'http' | 'streamable-http'")] transport: Option<String>,
        command: Option<String>,
        args: Option<Vec<String>>,
        url: Option<String>,
        headers: Option<std::collections::HashMap<String, String>>,
        env: Option<std::collections::HashMap<String, String>>,
        timeout_ms: Option<u32>,
    ) -> napi::Result<u32> {
        use a3s_code_core::mcp::protocol::{McpServerConfig, McpTransportConfig};

        let transport_str = transport.as_deref().unwrap_or("stdio");
        let transport_config = match transport_str {
            "stdio" => {
                let command = command.ok_or_else(|| {
                    napi::Error::from_reason("'command' is required for stdio transport")
                })?;
                McpTransportConfig::Stdio {
                    command,
                    args: args.unwrap_or_default(),
                }
            }
            "http" => {
                let url = url.ok_or_else(|| {
                    napi::Error::from_reason("'url' is required for http transport")
                })?;
                McpTransportConfig::Http {
                    url,
                    headers: headers.unwrap_or_default(),
                }
            }
            "streamable-http" | "streamable_http" => {
                let url = url.ok_or_else(|| {
                    napi::Error::from_reason("'url' is required for streamable-http transport")
                })?;
                McpTransportConfig::StreamableHttp {
                    url,
                    headers: headers.unwrap_or_default(),
                }
            }
            other => {
                return Err(napi::Error::from_reason(format!(
                    "Unknown transport '{}'. Use 'stdio', 'http', or 'streamable-http'",
                    other
                )))
            }
        };

        let tool_timeout_secs = timeout_ms
            .map(|ms| timeout_ms_to_secs(ms as u64))
            .unwrap_or(60);
        let session = self.inner.clone();
        let count = session
            .add_mcp_server(McpServerConfig {
                name,
                transport: transport_config,
                enabled: true,
                env: env.unwrap_or_default(),
                oauth: None,
                tool_timeout_secs,
            })
            .await
            .map_err(|e| napi::Error::from_reason(format!("add_mcp_server failed: {e}")))?;
        Ok(count as u32)
    }

    /// Add an MCP server with a typed object config.
    ///
    /// Preferred over the positional overload for new SDK callers.
    ///
    /// ```js
    /// await session.addMcpServerConfig({
    ///   name: 'github',
    ///   transport: { type: 'stdio', command: 'npx', args: ['-y', '@modelcontextprotocol/server-github'] },
    ///   env: { GITHUB_TOKEN: process.env.GITHUB_TOKEN },
    ///   timeoutMs: 30000,
    /// })
    /// ```
    #[napi(
        js_name = "addMcpServerConfig",
        ts_args_type = "config: McpServerConfig"
    )]
    pub async fn add_mcp_server_config(&self, config: serde_json::Value) -> napi::Result<u32> {
        let config = normalize_mcp_server_config(config)?;
        let session = self.inner.clone();
        let count = session
            .add_mcp_server(config)
            .await
            .map_err(|e| napi::Error::from_reason(format!("add_mcp_server failed: {e}")))?;
        Ok(count as u32)
    }

    /// Add an MCP server with the compact object-shaped API.
    #[napi(js_name = "addMcp", ts_args_type = "config: McpServerConfig")]
    pub async fn add_mcp(&self, config: serde_json::Value) -> napi::Result<u32> {
        self.add_mcp_server_config(config).await
    }

    /// Dynamically register agent definitions from a directory into the live session.
    ///
    /// Scans the directory for `*.yaml`, `*.yml`, and `*.md` agent definition files
    /// and registers them into the shared AgentRegistry used by the `task` tool.
    /// New agents are immediately callable via `task({ agent: "…", … })` without
    /// restarting the session.
    ///
    /// @param path - Directory to scan for agent definition files
    /// @returns Number of agents successfully loaded
    #[napi]
    pub fn register_agent_dir(&self, path: String) -> u32 {
        let dir = std::path::PathBuf::from(&path);
        self.inner.register_agent_dir(&dir) as u32
    }

    /// Register a disposable worker agent into the live session.
    ///
    /// The worker is immediately callable through the model-visible `task` tool.
    ///
    /// @param worker - Worker spec to register
    /// @returns Compiled agent definition
    #[napi]
    pub fn register_worker_agent(&self, worker: WorkerAgentSpec) -> napi::Result<AgentDefinition> {
        let worker = js_worker_agent_spec_to_rust(worker)?;
        let definition = self.inner.register_worker_agent(worker);
        Ok(rust_agent_definition_to_js(definition))
    }

    /// Register many disposable workers into the live session.
    ///
    /// @param workers - Worker specs to register
    /// @returns Compiled agent definitions
    #[napi]
    pub fn register_worker_agents(
        &self,
        workers: Vec<WorkerAgentSpec>,
    ) -> napi::Result<Vec<AgentDefinition>> {
        let workers = workers
            .into_iter()
            .map(js_worker_agent_spec_to_rust)
            .collect::<napi::Result<Vec<_>>>()?;
        Ok(self
            .inner
            .register_worker_agents(workers)
            .into_iter()
            .map(rust_agent_definition_to_js)
            .collect())
    }

    /// Disconnect and unregister an MCP server, removing its tools from the session.
    ///
    /// @param name - Server name (must match the name used in addMcpServer)
    #[napi]
    pub async fn remove_mcp_server(&self, name: String) -> napi::Result<()> {
        let session = self.inner.clone();
        session
            .remove_mcp_server(&name)
            .await
            .map_err(|e| napi::Error::from_reason(format!("remove_mcp_server failed: {e}")))?;
        Ok(())
    }

    /// Remove an MCP server with the compact API.
    #[napi(js_name = "removeMcp")]
    pub async fn remove_mcp(&self, name: String) -> napi::Result<()> {
        self.remove_mcp_server(name).await
    }

    /// Return connection status for all MCP servers registered on this session.
    ///
    /// @returns Array of `{ name, connected, toolCount }` entries
    #[napi]
    pub async fn mcp_status(&self) -> napi::Result<Vec<McpServerStatusEntry>> {
        let session = self.inner.clone();
        let status = session.mcp_status().await;
        Ok(status
            .into_iter()
            .map(|(name, s)| McpServerStatusEntry {
                name,
                connected: s.connected,
                tool_count: s.tool_count as u32,
                error: s.error,
            })
            .collect())
    }

    /// Return MCP server status with the compact API.
    #[napi]
    pub async fn mcps(&self) -> napi::Result<Vec<McpServerStatusEntry>> {
        self.mcp_status().await
    }

    /// Return the names of all tools currently registered on this session.
    ///
    /// @returns Array of tool name strings
    #[napi]
    pub fn tool_names(&self) -> Vec<String> {
        self.inner.tool_names()
    }

    /// Return full model-visible tool definitions currently registered on this session.
    #[napi]
    pub fn tool_definitions(&self) -> napi::Result<serde_json::Value> {
        serde_json::to_value(self.inner.tool_definitions())
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    /// Return a stored tool artifact by URI, or null if it is not retained.
    #[napi]
    pub fn get_artifact(&self, artifact_uri: String) -> napi::Result<serde_json::Value> {
        serde_json::to_value(self.inner.get_artifact(&artifact_uri))
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    // ========================================================================
    // Hook API
    // ========================================================================

    /// Register a hook for lifecycle event interception.
    ///
    /// Hooks registered on a session are automatically propagated to all sub-agents
    /// spawned by the `task` tool, including grandchild agents at arbitrary depth.
    /// This ensures security hooks (e.g. a sentinel) apply across the full agent tree
    /// without requiring explicit registration on each sub-agent session.
    ///
    /// @param hookId - Unique hook identifier
    /// @param eventType - Event type: "pre_tool_use", "post_tool_use", "generate_start",
    ///   "generate_end", "session_start", "session_end", "skill_load", "skill_unload",
    ///   "pre_prompt", "post_response", "on_error"
    /// @param matcher - Optional matcher: { tool?, pathPattern?, commandPattern?, sessionId?, skill? }
    /// @param config - Optional config: { priority?, timeoutMs?, asyncExecution?, maxRetries? }
    /// @param handler - Optional callback `(event: any) => { action: 'continue' | 'block' | 'skip',
    ///   reason?: string } | null`. When provided, the function is called for every matching event
    ///   and its return value controls execution. Return `{ action: 'block', reason: '...' }` to
    ///   cancel the operation, `{ action: 'skip' }` to skip remaining hooks, or `null`/`undefined`
    ///   for continue. Hooks with no handler still fire (observable via stream events) but always
    ///   continue.
    #[napi]
    pub fn register_hook(
        &self,
        hook_id: String,
        event_type: String,
        matcher: Option<HookMatcherObject>,
        config: Option<HookConfigObject>,
        #[napi(
            ts_arg_type = "((event: Record<string, unknown>) => { action: string; reason?: string } | null | undefined) | null | undefined"
        )]
        handler: Option<napi::JsFunction>,
    ) -> napi::Result<()> {
        use napi::threadsafe_function::{ErrorStrategy, ThreadSafeCallContext, ThreadsafeFunction};

        let rust_event_type = parse_hook_event_type(&event_type)?;
        let mut hook = RustHook::new(&hook_id, rust_event_type);

        if let Some(m) = matcher {
            let mut rust_matcher = RustHookMatcher::new();
            if let Some(tool) = m.tool {
                rust_matcher = rust_matcher.with_tool(tool);
            }
            if let Some(path) = m.path_pattern {
                rust_matcher = rust_matcher.with_path(path);
            }
            if let Some(cmd) = m.command_pattern {
                rust_matcher = rust_matcher.with_command(cmd);
            }
            if let Some(sid) = m.session_id {
                rust_matcher = rust_matcher.with_session(sid);
            }
            if let Some(skill) = m.skill {
                rust_matcher = rust_matcher.with_skill(skill);
            }
            hook = hook.with_matcher(rust_matcher);
        }

        if let Some(c) = config {
            hook = hook.with_config(RustHookConfig {
                priority: c.priority.unwrap_or(100),
                timeout_ms: c.timeout_ms.map(|v| v as u64).unwrap_or(30000),
                async_execution: c.async_execution.unwrap_or(false),
                max_retries: c.max_retries.unwrap_or(0),
            });
        }

        let timeout_ms = hook.config.timeout_ms;
        self.inner.register_hook(hook);

        if let Some(js_fn) = handler {
            let tsfn: ThreadsafeFunction<serde_json::Value, ErrorStrategy::CalleeHandled> = js_fn
                .create_threadsafe_function(
                0,
                |ctx: ThreadSafeCallContext<serde_json::Value>| {
                    let js_val = ctx.env.to_js_value(&ctx.value)?;
                    Ok(vec![js_val])
                },
            )?;
            self.inner.register_hook_handler(
                &hook_id,
                Arc::new(NodeCallbackHandler { tsfn, timeout_ms }),
            );
        }

        Ok(())
    }

    /// Unregister a hook by ID.
    ///
    /// @param hookId - The hook identifier to remove
    /// @returns true if the hook was found and removed
    #[napi]
    pub fn unregister_hook(&self, hook_id: String) -> bool {
        self.inner.unregister_hook_handler(&hook_id);
        self.inner.unregister_hook(&hook_id).is_some()
    }

    /// Get the number of registered hooks.
    #[napi]
    pub fn hook_count(&self) -> u32 {
        self.inner.hook_count() as u32
    }

    // ========================================================================
    // Session Metadata API
    // ========================================================================

    /// Return the session ID.
    #[napi(getter)]
    pub fn session_id(&self) -> String {
        self.inner.session_id().to_string()
    }

    /// Return the workspace path.
    #[napi(getter)]
    pub fn workspace(&self) -> String {
        self.inner.workspace().display().to_string()
    }

    /// Return any deferred init warning (e.g. memory store failed to initialize).
    #[napi(getter)]
    pub fn init_warning(&self) -> Option<String> {
        self.inner.init_warning().map(|s| s.to_string())
    }

    /// Host-defined tenant id attached at session creation, if any.
    #[napi(getter)]
    pub fn tenant_id(&self) -> Option<String> {
        self.inner.tenant_id().map(|s| s.to_string())
    }

    /// Identity of the principal that triggered the session, if any.
    #[napi(getter)]
    pub fn principal(&self) -> Option<String> {
        self.inner.principal().map(|s| s.to_string())
    }

    /// Logical agent template / definition id, if any.
    #[napi(getter)]
    pub fn agent_template_id(&self) -> Option<String> {
        self.inner.agent_template_id().map(|s| s.to_string())
    }

    /// Distributed-trace correlation id propagated through this session, if any.
    #[napi(getter)]
    pub fn correlation_id(&self) -> Option<String> {
        self.inner.correlation_id().map(|s| s.to_string())
    }

    // ========================================================================
    // Session Persistence API
    // ========================================================================

    /// Save the session to the configured store.
    #[napi]
    pub async fn save(&self) -> napi::Result<()> {
        let session = self.inner.clone();
        get_runtime()
            .spawn(async move { session.save().await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Save failed: {e}")))
    }

    // ========================================================================
    // Memory API
    // ========================================================================

    /// Check if memory is configured for this session.
    #[napi(getter)]
    pub fn has_memory(&self) -> bool {
        self.inner.memory().is_some()
    }

    /// Remember a successful task execution.
    ///
    /// @param task - Description of the task
    /// @param tools - List of tool names used
    /// @param result - Summary of the result
    #[napi]
    pub async fn remember_success(
        &self,
        task: String,
        tools: Vec<String>,
        result: String,
    ) -> napi::Result<()> {
        let memory = self
            .inner
            .memory()
            .ok_or_else(|| napi::Error::from_reason("Memory not configured for this session"))?
            .clone();
        get_runtime()
            .spawn(async move { memory.remember_success(&task, &tools, &result).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Remember failed: {e}")))
    }

    /// Remember a failed task execution.
    ///
    /// @param task - Description of the task
    /// @param error - Error message
    /// @param tools - List of tool names attempted
    #[napi]
    pub async fn remember_failure(
        &self,
        task: String,
        error: String,
        tools: Vec<String>,
    ) -> napi::Result<()> {
        let memory = self
            .inner
            .memory()
            .ok_or_else(|| napi::Error::from_reason("Memory not configured for this session"))?
            .clone();
        get_runtime()
            .spawn(async move { memory.remember_failure(&task, &error, &tools).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Remember failed: {e}")))
    }

    /// Recall memories similar to a query.
    ///
    /// @param query - Search query
    /// @param limit - Maximum number of results (default: 5)
    /// @returns Array of memory items
    #[napi]
    pub async fn recall_similar(
        &self,
        query: String,
        limit: Option<u32>,
    ) -> napi::Result<serde_json::Value> {
        let memory = self
            .inner
            .memory()
            .ok_or_else(|| napi::Error::from_reason("Memory not configured for this session"))?
            .clone();
        let limit = limit.unwrap_or(5) as usize;
        let items = get_runtime()
            .spawn(async move { memory.recall_similar(&query, limit).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Recall failed: {e}")))?;
        serde_json::to_value(&items)
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    /// Recall memories by tags.
    ///
    /// @param tags - Tags to search for
    /// @param limit - Maximum number of results (default: 10)
    /// @returns Array of memory items
    #[napi]
    pub async fn recall_by_tags(
        &self,
        tags: Vec<String>,
        limit: Option<u32>,
    ) -> napi::Result<serde_json::Value> {
        let memory = self
            .inner
            .memory()
            .ok_or_else(|| napi::Error::from_reason("Memory not configured for this session"))?
            .clone();
        let limit = limit.unwrap_or(10) as usize;
        let items = get_runtime()
            .spawn(async move { memory.recall_by_tags(&tags, limit).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Recall failed: {e}")))?;
        serde_json::to_value(&items)
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    /// Get recent memory items.
    ///
    /// @param limit - Maximum number of results (default: 10)
    /// @returns Array of memory items
    #[napi]
    pub async fn memory_recent(&self, limit: Option<u32>) -> napi::Result<serde_json::Value> {
        let memory = self
            .inner
            .memory()
            .ok_or_else(|| napi::Error::from_reason("Memory not configured for this session"))?
            .clone();
        let limit = limit.unwrap_or(10) as usize;
        let items = get_runtime()
            .spawn(async move { memory.get_recent(limit).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Recall failed: {e}")))?;
        serde_json::to_value(&items)
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    /// Get memory statistics.
    ///
    /// @returns Object with longTermCount, shortTermCount, workingCount
    #[napi]
    pub async fn memory_stats(&self) -> napi::Result<serde_json::Value> {
        let memory = self
            .inner
            .memory()
            .ok_or_else(|| napi::Error::from_reason("Memory not configured for this session"))?
            .clone();
        let stats = get_runtime()
            .spawn(async move { memory.stats().await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Stats failed: {e}")))?;
        serde_json::to_value(&stats)
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    /// Get current working memory items.
    ///
    /// Working memory holds the active context items for the current task.
    ///
    /// @returns Array of memory items currently in working memory
    #[napi]
    pub async fn get_working(&self) -> napi::Result<serde_json::Value> {
        let memory = self
            .inner
            .memory()
            .ok_or_else(|| napi::Error::from_reason("Memory not configured for this session"))?
            .clone();
        let items = get_runtime()
            .spawn(async move { memory.get_working().await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?;
        serde_json::to_value(&items)
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    /// Clear working memory.
    ///
    /// Removes all items from working memory without affecting short-term or long-term memory.
    #[napi]
    pub async fn clear_working(&self) -> napi::Result<()> {
        let memory = self
            .inner
            .memory()
            .ok_or_else(|| napi::Error::from_reason("Memory not configured for this session"))?
            .clone();
        get_runtime()
            .spawn(async move { memory.clear_working().await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))
    }

    /// Get current short-term memory items.
    ///
    /// Short-term memory contains items stored during this session.
    ///
    /// @returns Array of memory items in short-term memory
    #[napi]
    pub async fn get_short_term(&self) -> napi::Result<serde_json::Value> {
        let memory = self
            .inner
            .memory()
            .ok_or_else(|| napi::Error::from_reason("Memory not configured for this session"))?
            .clone();
        let items = get_runtime()
            .spawn(async move { memory.get_short_term().await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?;
        serde_json::to_value(&items)
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {e}")))
    }

    /// Clear short-term memory for this session.
    ///
    /// Removes all session-scoped memory items without affecting long-term or working memory.
    #[napi]
    pub async fn clear_short_term(&self) -> napi::Result<()> {
        let memory = self
            .inner
            .memory()
            .ok_or_else(|| napi::Error::from_reason("Memory not configured for this session"))?
            .clone();
        get_runtime()
            .spawn(async move { memory.clear_short_term().await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))
    }

    // ========================================================================
    // Slash Command & Scheduler API
    // ========================================================================

    /// Register a custom slash command.
    ///
    /// Slash commands are invoked via `session.send("/command args")` and execute
    /// before the LLM sees the input. The handler receives the command arguments
    /// and a context object with session metadata.
    ///
    /// @param name - Command name without the leading `/` (e.g., `"status"`)
    /// @param description - Short description shown in `/help`
    /// @param handler - Callback `(args: string, ctx: CommandContext) => string`
    ///
    /// @example
    /// ```typescript
    /// session.registerCommand("status", "Show session info", (args, ctx) => {
    ///   return `Session ${ctx.sessionId} in ${ctx.workspace}`;
    /// });
    /// await session.send("/status");
    /// ```
    #[napi(
        ts_args_type = "name: string, description: string, handler: (args: string, ctx: CommandContext) => string"
    )]
    pub fn register_command(
        &self,
        name: String,
        description: String,
        handler: napi::JsFunction,
    ) -> napi::Result<()> {
        use napi::threadsafe_function::ThreadSafeCallContext;

        // Create a threadsafe function that calls the JS handler
        let tsfn: napi::threadsafe_function::ThreadsafeFunction<
            (String, RustCommandContext),
            napi::threadsafe_function::ErrorStrategy::Fatal,
        > = handler.create_threadsafe_function(
            0,
            |ctx: ThreadSafeCallContext<(String, RustCommandContext)>| {
                // Extract the values
                let args = ctx.value.0;
                let cmd_ctx = ctx.value.1;

                // Convert to JS values
                let args_str = ctx.env.create_string(&args)?;
                let ctx_obj = js_command_context_to_object(&ctx.env, &cmd_ctx)?;

                // Return the arguments that will be passed to the JS function
                Ok(vec![args_str.into_unknown(), ctx_obj.into_unknown()])
            },
        )?;

        let cmd = Arc::new(JsSlashCommand {
            name,
            description,
            handler: Arc::new(tsfn),
        });
        self.inner.clone().register_command(cmd);
        Ok(())
    }

    /// List all registered slash commands.
    ///
    /// Returns each command's name, description, and optional usage hint.
    /// Slash commands can be invoked via `session.send("/command args")`.
    ///
    /// @returns Array of CommandInfo objects sorted by name
    #[napi]
    pub fn list_commands(&self) -> Vec<CommandInfo> {
        self.inner
            .command_registry()
            .list_full()
            .into_iter()
            .map(|(name, description, usage)| CommandInfo {
                name,
                description,
                usage,
            })
            .collect()
    }

    /// Cancel the current ongoing operation (send/stream).
    ///
    /// If an operation is in progress, this will trigger cancellation of the LLM streaming
    /// and tool execution. The operation will terminate as soon as possible.
    ///
    /// @returns `true` if an operation was cancelled, `false` if no operation was in progress
    #[napi]
    pub fn cancel(&self) -> bool {
        let session = self.inner.clone();
        get_runtime().block_on(session.cancel())
    }

    /// Close the session and cancel any active operation.
    ///
    /// Call this when the session will no longer be used so Node.js can exit
    /// cleanly without waiting on session-scoped background workers.
    #[napi]
    pub fn close(&self) {
        let session = self.inner.clone();
        get_runtime().block_on(session.close())
    }

    /// Whether [`close`](#method.close) has been called on this session.
    ///
    /// Once `true`, calls to `send` / `stream` reject with a "Session closed"
    /// error instead of starting a new run.
    #[napi]
    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }

    /// Install a host-supplied BudgetGuard on this session.
    ///
    /// Each callback receives a single context object:
    /// - `checkBeforeLlm({ sessionId, estimatedTokens }) -> BudgetDecision | null`
    /// - `recordAfterLlm({ sessionId, usage }) -> void`
    /// - `checkBeforeTool({ sessionId, toolName }) -> BudgetDecision | null`
    ///
    /// where `BudgetDecision` is one of:
    /// - `null` / `{ decision: 'allow' }`                                                     → allow
    /// - `{ decision: 'soft', resource, consumed, limit, message? }`                          → emits BudgetThresholdHit('soft'), proceeds
    /// - `{ decision: 'deny',  resource, reason }`                                            → aborts the call, throws "Budget exhausted"
    ///
    /// FAIL-CLOSED on hang: a `check*` callback that does not return
    /// within `timeoutMs` (default 5000) is treated as a **deny**, never
    /// a silent allow — a budget control must not disable itself when the
    /// guard stalls. A malformed/unreadable return likewise denies.
    ///
    /// ⚠️ The callbacks MUST NOT throw. Due to a napi-rs limitation a JS
    /// exception thrown from a budget-guard callback aborts the host
    /// process (the return value cannot be converted). Wrap your logic in
    /// try/catch and return a decision (e.g. a deny) instead of throwing.
    /// (The Python SDK's BudgetGuard catches exceptions safely; only the
    /// Node binding has this constraint.)
    ///
    /// The guard takes effect on the next `send` / `stream`. Pass `null`
    /// for a method to leave it unhandled (default allow / no-op). Pass
    /// `null` for the whole handlers arg to clear the guard.
    #[napi(
        ts_args_type = "handlers: { checkBeforeLlm?: ((ctx: { sessionId: string; estimatedTokens: number }) => any) | null; recordAfterLlm?: ((ctx: { sessionId: string; usage: any }) => void) | null; checkBeforeTool?: ((ctx: { sessionId: string; toolName: string }) => any) | null; timeoutMs?: number | null } | null"
    )]
    pub fn set_budget_guard(&self, handlers: Option<BudgetGuardHandlers>) -> napi::Result<()> {
        use napi::threadsafe_function::{ErrorStrategy, ThreadSafeCallContext, ThreadsafeFunction};

        let Some(h) = handlers else {
            self.inner.set_budget_guard(None);
            return Ok(());
        };

        // Pass the call context as a SINGLE object arg so the JS callback
        // signature is the clean `(ctx) => decision`. We use
        // `ErrorStrategy::Fatal` (no leading `err` param). NOTE: in this
        // napi-rs version a JS callback that THROWS aborts the host process
        // at the return-value-conversion stage regardless of ErrorStrategy
        // (CalleeHandled does not help) — so budget-guard callbacks MUST NOT
        // throw; wrap your logic in try/catch and return a decision. Hangs
        // are handled safely (fail-closed timeout below).
        let single_obj = |ctx: ThreadSafeCallContext<serde_json::Value>| {
            Ok(vec![ctx.env.to_js_value(&ctx.value)?])
        };

        let check_llm_tsfn: Option<ThreadsafeFunction<serde_json::Value, ErrorStrategy::Fatal>> = h
            .check_before_llm
            .map(|f| f.create_threadsafe_function(0, single_obj))
            .transpose()?;

        let record_tsfn: Option<ThreadsafeFunction<serde_json::Value, ErrorStrategy::Fatal>> = h
            .record_after_llm
            .map(|f| f.create_threadsafe_function(0, single_obj))
            .transpose()?;

        let check_tool_tsfn: Option<ThreadsafeFunction<serde_json::Value, ErrorStrategy::Fatal>> =
            h.check_before_tool
                .map(|f| f.create_threadsafe_function(0, single_obj))
                .transpose()?;

        let guard: Arc<dyn a3s_code_core::budget::BudgetGuard> = Arc::new(NodeBudgetGuard {
            check_before_llm: check_llm_tsfn,
            record_after_llm: record_tsfn,
            check_before_tool: check_tool_tsfn,
            // Configurable; default 5s. On timeout the guard fails CLOSED
            // (Deny), so a small value trades latency-on-hang for faster
            // denial of a stuck guard.
            timeout_ms: h.timeout_ms.map(|t| t as u64).unwrap_or(5_000),
        });
        self.inner.set_budget_guard(Some(guard));
        Ok(())
    }
}

// ============================================================================
// Node-side BudgetGuard wrapper
// ============================================================================

/// Shape of the JS handlers object accepted by `session.setBudgetGuard`.
/// Each field is optional — methods that aren't provided fall back to
/// the framework's default Allow / no-op behaviour.
#[napi(object)]
pub struct BudgetGuardHandlers {
    pub check_before_llm: Option<napi::JsFunction>,
    pub record_after_llm: Option<napi::JsFunction>,
    pub check_before_tool: Option<napi::JsFunction>,
    /// Max time (ms) to wait for a `check*` callback to return before
    /// the guard fails **closed** (denies). Default 5000. A guard that
    /// throws (so its return value never arrives) or hangs is denied
    /// after this deadline — budget enforcement never silently
    /// disables itself.
    pub timeout_ms: Option<u32>,
}

/// FIFO retention caps on the session's in-memory stores. All fields
/// optional; missing fields keep the unbounded default for that
/// store. Use to cap memory growth across long-running cluster
/// sessions.
#[napi(object)]
pub struct RetentionLimitsObject {
    /// Cap on the number of runs retained in InMemoryRunStore.
    /// When exceeded the oldest run is dropped along with its events.
    pub max_runs_retained: Option<u32>,
    /// Cap on event records retained per run. Oldest events
    /// FIFO-dropped from each run's buffer past this cap. The
    /// snapshot's cumulative `eventCount` is not decremented.
    pub max_events_per_run: Option<u32>,
    /// Cap on events retained in InMemoryTraceSink.
    pub max_trace_events: Option<u32>,
    /// Cap on **terminal** (Completed / Failed / Cancelled) subagent
    /// task snapshots. Running tasks are never evicted.
    pub max_terminal_subagent_tasks: Option<u32>,
}

struct NodeBudgetGuard {
    check_before_llm: Option<
        napi::threadsafe_function::ThreadsafeFunction<
            serde_json::Value,
            napi::threadsafe_function::ErrorStrategy::Fatal,
        >,
    >,
    record_after_llm: Option<
        napi::threadsafe_function::ThreadsafeFunction<
            serde_json::Value,
            napi::threadsafe_function::ErrorStrategy::Fatal,
        >,
    >,
    check_before_tool: Option<
        napi::threadsafe_function::ThreadsafeFunction<
            serde_json::Value,
            napi::threadsafe_function::ErrorStrategy::Fatal,
        >,
    >,
    timeout_ms: u64,
}

// SAFETY: ThreadsafeFunction is designed to be sent across threads.
unsafe impl Send for NodeBudgetGuard {}
unsafe impl Sync for NodeBudgetGuard {}

impl NodeBudgetGuard {
    fn call_decision(
        &self,
        tsfn: &napi::threadsafe_function::ThreadsafeFunction<
            serde_json::Value,
            napi::threadsafe_function::ErrorStrategy::Fatal,
        >,
        args: serde_json::Value,
    ) -> a3s_code_core::budget::BudgetDecision {
        let (tx, rx) = std::sync::mpsc::sync_channel::<a3s_code_core::budget::BudgetDecision>(1);
        tsfn.call_with_return_value(
            args,
            napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
            move |ret: napi::JsUnknown| {
                // FAIL-CLOSED: if the JS return value can't even be read as
                // a napi value, deny rather than allow. A budget guard is a
                // cost/quota control — silently permitting on a broken
                // response is the dangerous direction. (Explicit responses
                // like null / {decision:'allow'} are still parsed leniently
                // as Allow inside parse_js_budget_decision.)
                let decision = parse_js_budget_decision(ret).unwrap_or_else(|_| {
                    a3s_code_core::budget::BudgetDecision::Deny {
                        resource: "budget_guard_error".to_string(),
                        reason: "budget guard return value could not be read".to_string(),
                    }
                });
                let _ = tx.send(decision);
                Ok(())
            },
        );
        // FAIL-CLOSED on timeout: a hung or throwing guard (under Fatal
        // strategy a JS throw means the return closure never fires, so the
        // channel stays empty and we hit this timeout) must DENY, not
        // Allow. Previously this defaulted to Allow — meaning a slow/buggy
        // guard silently disabled budget enforcement (a fail-open hole).
        tokio::task::block_in_place(|| {
            rx.recv_timeout(std::time::Duration::from_millis(self.timeout_ms))
                .unwrap_or_else(|_| a3s_code_core::budget::BudgetDecision::Deny {
                    resource: "budget_guard_timeout".to_string(),
                    reason: format!("budget guard did not respond within {}ms", self.timeout_ms),
                })
        })
    }
}

#[async_trait::async_trait]
impl a3s_code_core::budget::BudgetGuard for NodeBudgetGuard {
    async fn check_before_llm(
        &self,
        session_id: &str,
        estimated_prompt_tokens: usize,
    ) -> a3s_code_core::budget::BudgetDecision {
        let Some(tsfn) = self.check_before_llm.as_ref() else {
            return a3s_code_core::budget::BudgetDecision::Allow;
        };
        self.call_decision(
            tsfn,
            serde_json::json!({
                "sessionId": session_id,
                "estimatedTokens": estimated_prompt_tokens,
            }),
        )
    }

    async fn record_after_llm(&self, session_id: &str, usage: &a3s_code_core::llm::TokenUsage) {
        let Some(tsfn) = self.record_after_llm.as_ref() else {
            return;
        };
        let _ = tsfn.call(
            serde_json::json!({
                "sessionId": session_id,
                "usage": {
                    "promptTokens": usage.prompt_tokens,
                    "completionTokens": usage.completion_tokens,
                    "totalTokens": usage.total_tokens,
                    "cacheReadTokens": usage.cache_read_tokens,
                    "cacheWriteTokens": usage.cache_write_tokens,
                },
            }),
            napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
        );
    }

    async fn check_before_tool(
        &self,
        session_id: &str,
        tool_name: &str,
    ) -> a3s_code_core::budget::BudgetDecision {
        let Some(tsfn) = self.check_before_tool.as_ref() else {
            return a3s_code_core::budget::BudgetDecision::Allow;
        };
        self.call_decision(
            tsfn,
            serde_json::json!({ "sessionId": session_id, "toolName": tool_name }),
        )
    }
}

/// Parse the return value of a JS BudgetGuard callback into a
/// [`BudgetDecision`](a3s_code_core::budget::BudgetDecision).
///
/// Accepted JS shapes mirror Python's:
/// - `null` / `undefined` / `{ decision: 'allow' }`                                                 → Allow
/// - `{ decision: 'soft', resource, consumed, limit, message? }`                                    → SoftLimit
/// - `{ decision: 'deny',  resource, reason }`                                                      → Deny
fn parse_js_budget_decision(
    val: napi::JsUnknown,
) -> napi::Result<a3s_code_core::budget::BudgetDecision> {
    use a3s_code_core::budget::BudgetDecision;
    use napi::{JsObject, ValueType};

    match val.get_type()? {
        ValueType::Null | ValueType::Undefined => Ok(BudgetDecision::Allow),
        ValueType::Object => {
            let obj = unsafe { val.cast::<JsObject>() };
            let decision: String = obj
                .get_named_property::<napi::JsString>("decision")
                .ok()
                .and_then(|s| s.into_utf8().ok())
                .and_then(|s| s.into_owned().ok())
                .unwrap_or_else(|| "allow".to_string());
            match decision.as_str() {
                "deny" => {
                    let resource = obj
                        .get_named_property::<napi::JsString>("resource")
                        .ok()
                        .and_then(|s| s.into_utf8().ok())
                        .and_then(|s| s.into_owned().ok())
                        .unwrap_or_else(|| "unspecified".to_string());
                    let reason = obj
                        .get_named_property::<napi::JsString>("reason")
                        .ok()
                        .and_then(|s| s.into_utf8().ok())
                        .and_then(|s| s.into_owned().ok())
                        .unwrap_or_else(|| "denied by host".to_string());
                    Ok(BudgetDecision::Deny { resource, reason })
                }
                "soft" => {
                    let resource = obj
                        .get_named_property::<napi::JsString>("resource")
                        .ok()
                        .and_then(|s| s.into_utf8().ok())
                        .and_then(|s| s.into_owned().ok())
                        .unwrap_or_else(|| "unspecified".to_string());
                    let consumed = obj
                        .get_named_property::<napi::JsNumber>("consumed")
                        .ok()
                        .and_then(|n| n.get_double().ok())
                        .unwrap_or(0.0);
                    let limit = obj
                        .get_named_property::<napi::JsNumber>("limit")
                        .ok()
                        .and_then(|n| n.get_double().ok())
                        .unwrap_or(0.0);
                    let message = obj
                        .get_named_property::<napi::JsString>("message")
                        .ok()
                        .and_then(|s| s.into_utf8().ok())
                        .and_then(|s| s.into_owned().ok());
                    Ok(BudgetDecision::SoftLimit {
                        resource,
                        consumed,
                        limit,
                        message,
                    })
                }
                _ => Ok(BudgetDecision::Allow),
            }
        }
        _ => Ok(BudgetDecision::Allow),
    }
}

// ============================================================================
// Slash Command Types
// ============================================================================

/// MCP server metadata exposed to slash command handlers.
#[napi(object)]
#[derive(Clone)]
pub struct CommandMcpServerInfo {
    /// MCP server name.
    pub name: String,
    /// Number of tools currently exposed by the server.
    pub tool_count: u32,
}

/// Context passed to custom slash command handlers.
#[napi(object)]
#[derive(Clone)]
pub struct CommandContext {
    /// Current session ID.
    pub session_id: String,
    /// Current workspace path.
    pub workspace: String,
    /// Current active model identifier.
    pub model: String,
    /// Number of messages in session history.
    pub history_len: u32,
    /// Total tokens used in this session so far.
    pub total_tokens: i64,
    /// Estimated session cost in USD.
    pub total_cost: f64,
    /// Registered tool names (builtin + MCP).
    pub tool_names: Vec<String>,
    /// Connected MCP servers and their tool counts.
    pub mcp_servers: Vec<CommandMcpServerInfo>,
}

/// Metadata about a registered slash command.
#[napi(object)]
#[derive(Clone)]
pub struct CommandInfo {
    /// Command name without the leading `/` (e.g., `"help"`, `"model"`)
    pub name: String,
    /// Short description shown in `/help`
    pub description: String,
    /// Optional usage hint (e.g., `"/model <provider/model>"`)
    pub usage: Option<String>,
}

// ============================================================================
// Hook Types
// ============================================================================

/// Matcher for filtering which events trigger a hook.
#[napi(object)]
#[derive(Clone)]
pub struct HookMatcherObject {
    /// Match specific tool name (exact match)
    pub tool: Option<String>,
    /// Match file path pattern (glob)
    pub path_pattern: Option<String>,
    /// Match command pattern (regex for Bash commands)
    pub command_pattern: Option<String>,
    /// Match session ID (exact match)
    pub session_id: Option<String>,
    /// Match skill name (supports glob patterns)
    pub skill: Option<String>,
}

/// Configuration for a hook.
#[napi(object)]
#[derive(Clone)]
pub struct HookConfigObject {
    /// Priority (lower values = higher priority, default: 100)
    pub priority: Option<i32>,
    /// Timeout in milliseconds (default: 30000)
    pub timeout_ms: Option<i64>,
    /// Whether to execute asynchronously (fire-and-forget)
    pub async_execution: Option<bool>,
    /// Maximum retry attempts
    pub max_retries: Option<u32>,
}

fn metrics_snapshot_to_json(snapshot: Option<RustMetricsSnapshot>) -> serde_json::Value {
    let s = match snapshot {
        None => return serde_json::Value::Null,
        Some(s) => s,
    };
    let counters: serde_json::Map<String, serde_json::Value> = s
        .counters
        .into_iter()
        .map(|(k, v)| (k, serde_json::Value::Number(v.into())))
        .collect();
    let gauges: serde_json::Map<String, serde_json::Value> = s
        .gauges
        .into_iter()
        .map(|(k, v)| {
            let n = serde_json::Number::from_f64(v).unwrap_or_else(|| 0.into());
            (k, serde_json::Value::Number(n))
        })
        .collect();
    let histograms: serde_json::Map<String, serde_json::Value> = s
        .histograms
        .into_iter()
        .map(|(k, h)| {
            let to_f = |v: f64| serde_json::Number::from_f64(v).unwrap_or_else(|| 0.into());
            let (min, max) = if h.count == 0 {
                (0.into(), 0.into())
            } else {
                (to_f(h.min), to_f(h.max))
            };
            let v = serde_json::json!({
                "count": h.count,
                "sum": to_f(h.sum),
                "min": min,
                "max": max,
                "mean": to_f(h.mean),
                "p50": to_f(h.percentiles.p50),
                "p90": to_f(h.percentiles.p90),
                "p95": to_f(h.percentiles.p95),
                "p99": to_f(h.percentiles.p99),
            });
            (k, v)
        })
        .collect();
    serde_json::json!({
        "counters": serde_json::Value::Object(counters),
        "gauges": serde_json::Value::Object(gauges),
        "histograms": serde_json::Value::Object(histograms),
    })
}

fn parse_hook_event_type(event_type: &str) -> napi::Result<RustHookEventType> {
    match event_type {
        "pre_tool_use" => Ok(RustHookEventType::PreToolUse),
        "post_tool_use" => Ok(RustHookEventType::PostToolUse),
        "generate_start" => Ok(RustHookEventType::GenerateStart),
        "generate_end" => Ok(RustHookEventType::GenerateEnd),
        "session_start" => Ok(RustHookEventType::SessionStart),
        "session_end" => Ok(RustHookEventType::SessionEnd),
        "skill_load" => Ok(RustHookEventType::SkillLoad),
        "skill_unload" => Ok(RustHookEventType::SkillUnload),
        "pre_prompt" => Ok(RustHookEventType::PrePrompt),
        "post_response" => Ok(RustHookEventType::PostResponse),
        "on_error" => Ok(RustHookEventType::OnError),
        _ => Err(napi::Error::from_reason(format!(
            "Invalid hook event type: '{}'. Expected one of: pre_tool_use, post_tool_use, \
             generate_start, generate_end, session_start, session_end, skill_load, \
             skill_unload, pre_prompt, post_response, on_error",
            event_type
        ))),
    }
}

// ============================================================================
// NodeCallbackHandler — bridges JS hook callbacks into the Rust HookHandler trait
// ============================================================================

struct NodeCallbackHandler {
    tsfn: napi::threadsafe_function::ThreadsafeFunction<
        serde_json::Value,
        napi::threadsafe_function::ErrorStrategy::CalleeHandled,
    >,
    timeout_ms: u64,
}

// SAFETY: ThreadsafeFunction is designed to be sent across threads.
unsafe impl Send for NodeCallbackHandler {}
unsafe impl Sync for NodeCallbackHandler {}

impl RustHookHandler for NodeCallbackHandler {
    fn handle(&self, event: &RustHookEvent) -> RustHookResponse {
        let Ok(event_json) = serde_json::to_value(event) else {
            return RustHookResponse::continue_();
        };

        let (tx, rx) = std::sync::mpsc::sync_channel::<RustHookResponse>(1);

        self.tsfn.call_with_return_value(
            Ok(event_json),
            napi::threadsafe_function::ThreadsafeFunctionCallMode::NonBlocking,
            move |ret: napi::JsUnknown| {
                let response =
                    parse_js_hook_response(ret).unwrap_or_else(|_| RustHookResponse::continue_());
                let _ = tx.send(response);
                Ok(())
            },
        );

        // block_in_place: signal to tokio that this thread will block;
        // valid only on multi-thread runtime (which get_runtime() always creates).
        tokio::task::block_in_place(|| {
            rx.recv_timeout(std::time::Duration::from_millis(self.timeout_ms))
                .unwrap_or_else(|_| RustHookResponse::continue_())
        })
    }
}

/// Parse the return value from a JS hook callback into a `HookResponse`.
///
/// Accepted JS return shapes:
/// - `null` / `undefined`              → continue
/// - `{ action: 'continue' }`          → continue
/// - `{ action: 'block', reason: '…' }` → block
/// - `{ action: 'skip' }`              → skip
/// - `{ action: 'retry', delayMs: N }` → retry after N ms
fn parse_js_hook_response(val: napi::JsUnknown) -> napi::Result<RustHookResponse> {
    use napi::{JsObject, ValueType};

    match val.get_type()? {
        ValueType::Null | ValueType::Undefined => Ok(RustHookResponse::continue_()),
        ValueType::Object => {
            let obj = unsafe { val.cast::<JsObject>() };
            let action: Option<String> = obj
                .get_named_property::<napi::JsString>("action")
                .ok()
                .and_then(|s| s.into_utf8().ok())
                .and_then(|s| s.into_owned().ok());

            match action.as_deref() {
                Some("block") => {
                    let reason = obj
                        .get_named_property::<napi::JsString>("reason")
                        .ok()
                        .and_then(|s| s.into_utf8().ok())
                        .and_then(|s| s.into_owned().ok())
                        .unwrap_or_else(|| "Blocked by hook".to_string());
                    Ok(RustHookResponse::block(reason))
                }
                Some("skip") => Ok(RustHookResponse::skip()),
                Some("retry") => {
                    let delay_ms = obj
                        .get_named_property::<napi::JsNumber>("delayMs")
                        .ok()
                        .and_then(|n| n.get_uint32().ok())
                        .unwrap_or(1000) as u64;
                    Ok(RustHookResponse::retry(delay_ms))
                }
                // "continue" or any other value → continue
                _ => Ok(RustHookResponse::continue_()),
            }
        }
        _ => Ok(RustHookResponse::continue_()),
    }
}

// ============================================================================
// SkillInfo
// ============================================================================

/// Metadata about a built-in skill.
#[napi(object)]
#[derive(Clone)]
pub struct SkillInfo {
    pub name: String,
    pub description: String,
    /// Skill kind: "instruction", "tool", or "agent".
    pub kind: String,
}

/// Return a list of built-in skills compiled into the library.
///
/// Each entry has `name`, `description`, and `kind` (instruction, tool, or agent).
#[napi]
pub fn builtin_skills() -> Vec<SkillInfo> {
    rust_builtin_skills()
        .into_iter()
        .map(|s| SkillInfo {
            name: s.name.clone(),
            description: s.description.clone(),
            kind: match s.kind {
                RustSkillKind::Instruction => "instruction".to_string(),
                RustSkillKind::Persona => "persona".to_string(),
                RustSkillKind::Tool => "tool".to_string(),
            },
        })
        .collect()
}

// ============================================================================
// Conversion Helpers
// ============================================================================

fn js_content_block_to_rust(block: &ContentBlockObject) -> RustContentBlock {
    match block.block_type.as_str() {
        "tool_use" => RustContentBlock::ToolUse {
            id: block.id.clone().unwrap_or_default(),
            name: block.name.clone().unwrap_or_default(),
            input: block.input.clone().unwrap_or(serde_json::Value::Null),
        },
        "tool_result" => RustContentBlock::ToolResult {
            tool_use_id: block.tool_use_id.clone().unwrap_or_default(),
            content: a3s_code_core::llm::ToolResultContentField::Text(
                block.result_content.clone().unwrap_or_default(),
            ),
            is_error: block.is_error,
        },
        _ => RustContentBlock::Text {
            text: block.text.clone().unwrap_or_default(),
        },
    }
}

fn rust_content_block_to_js(block: &RustContentBlock) -> ContentBlockObject {
    match block {
        RustContentBlock::Text { text } => ContentBlockObject {
            block_type: "text".to_string(),
            text: Some(text.clone()),
            id: None,
            name: None,
            input: None,
            tool_use_id: None,
            result_content: None,
            is_error: None,
        },
        RustContentBlock::ToolUse { id, name, input } => ContentBlockObject {
            block_type: "tool_use".to_string(),
            text: None,
            id: Some(id.clone()),
            name: Some(name.clone()),
            input: Some(input.clone()),
            tool_use_id: None,
            result_content: None,
            is_error: None,
        },
        RustContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => ContentBlockObject {
            block_type: "tool_result".to_string(),
            text: None,
            id: None,
            name: None,
            input: None,
            tool_use_id: Some(tool_use_id.clone()),
            result_content: Some(match content {
                a3s_code_core::llm::ToolResultContentField::Text(s) => s.clone(),
                a3s_code_core::llm::ToolResultContentField::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|b| {
                        if let a3s_code_core::llm::ToolResultContent::Text { text } = b {
                            Some(text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            }),
            is_error: *is_error,
        },
        RustContentBlock::Image { .. } => ContentBlockObject {
            block_type: "image".to_string(),
            text: None,
            id: None,
            name: None,
            input: None,
            tool_use_id: None,
            result_content: None,
            is_error: None,
        },
    }
}

/// Convert JS AttachmentObject array to Rust Attachment vec.
fn js_attachments_to_rust(attachments: &[AttachmentObject]) -> Vec<a3s_code_core::llm::Attachment> {
    attachments
        .iter()
        .map(|a| a3s_code_core::llm::Attachment::new(a.data.to_vec(), a.media_type.clone()))
        .collect()
}

fn js_messages_to_rust(messages: &[MessageObject]) -> napi::Result<Vec<RustMessage>> {
    Ok(messages
        .iter()
        .map(|m| RustMessage {
            role: m.role.clone(),
            content: m.content.iter().map(js_content_block_to_rust).collect(),
            reasoning_content: None,
        })
        .collect())
}

fn rust_messages_to_js(messages: &[RustMessage]) -> Vec<MessageObject> {
    messages
        .iter()
        .map(|m| MessageObject {
            role: m.role.clone(),
            content: m.content.iter().map(rust_content_block_to_js).collect(),
        })
        .collect()
}

// ============================================================================
// SearchConfig
// ============================================================================

/// Configuration for a search engine.
#[napi(object)]
#[derive(Clone)]
pub struct SearchEngineConfig {
    pub enabled: bool,
    pub weight: f64,
    pub timeout: Option<u32>,
}

impl From<SearchEngineConfig> for RustSearchEngineConfig {
    fn from(c: SearchEngineConfig) -> Self {
        Self {
            enabled: c.enabled,
            weight: c.weight,
            timeout: c.timeout.map(|t| t as u64),
        }
    }
}

/// Browser backend for headless search.
#[napi]
pub enum BrowserBackend {
    /// Chrome/Chromium headless.
    Chrome,
    /// Lightpanda headless browser (Linux/macOS only).
    Lightpanda,
}

impl From<BrowserBackend> for RustBrowserBackend {
    fn from(b: BrowserBackend) -> Self {
        match b {
            BrowserBackend::Chrome => RustBrowserBackend::Chrome,
            BrowserBackend::Lightpanda => RustBrowserBackend::Lightpanda,
        }
    }
}

/// Headless browser configuration.
#[napi(object)]
#[derive(Clone)]
pub struct HeadlessConfig {
    pub backend: BrowserBackend,
    pub browser_path: Option<String>,
    pub max_tabs: Option<u32>,
    pub launch_args: Option<Vec<String>>,
    pub proxy_url: Option<String>,
}

impl From<HeadlessConfig> for RustHeadlessConfig {
    fn from(c: HeadlessConfig) -> Self {
        Self {
            backend: c.backend.into(),
            browser_path: c.browser_path,
            max_tabs: c.max_tabs.unwrap_or(4) as usize,
            launch_args: c.launch_args.unwrap_or_default(),
            proxy_url: c.proxy_url,
        }
    }
}

/// Health monitor configuration for search engines.
#[napi(object)]
#[derive(Clone)]
pub struct SearchHealthConfig {
    pub max_failures: u32,
    pub suspend_seconds: u32,
}

impl From<SearchHealthConfig> for RustSearchHealthConfig {
    fn from(c: SearchHealthConfig) -> Self {
        Self {
            max_failures: c.max_failures,
            suspend_seconds: c.suspend_seconds as u64,
        }
    }
}

/// Search engine configuration (a3s-search integration).
#[napi(object)]
#[derive(Clone)]
pub struct SearchConfig {
    pub timeout: u32,
    pub health: Option<SearchHealthConfig>,
    pub engines: std::collections::HashMap<String, SearchEngineConfig>,
    pub headless: Option<HeadlessConfig>,
}

impl From<SearchConfig> for RustSearchConfig {
    fn from(c: SearchConfig) -> Self {
        Self {
            timeout: c.timeout as u64,
            health: c.health.map(|h| h.into()),
            engines: c.engines.into_iter().map(|(k, v)| (k, v.into())).collect(),
            headless: c.headless.map(|h| h.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orchestration_object_conversions_round_trip_fields() {
        let schema = serde_json::json!({ "type": "object" });
        let spec = AgentStepSpecObject {
            task_id: "t1".into(),
            agent: "explore".into(),
            description: "d".into(),
            prompt: "p".into(),
            max_steps: Some(5),
            parent_session_id: Some("parent".into()),
            output_schema: Some(schema.clone()),
        };
        let rust: RustAgentStepSpec = spec.into();
        assert_eq!(rust.task_id, "t1");
        assert_eq!(rust.agent, "explore");
        assert_eq!(rust.max_steps, Some(5));
        assert_eq!(rust.parent_session_id.as_deref(), Some("parent"));
        assert_eq!(rust.output_schema, Some(schema));

        let outcome = RustStepOutcome {
            task_id: "t1".into(),
            session_id: "task-run-t1".into(),
            agent: "explore".into(),
            output: "out".into(),
            success: true,
            structured: Some(serde_json::json!({ "k": 1 })),
        };
        let obj = StepOutcomeObject::from(outcome);
        assert_eq!(obj.task_id, "t1");
        assert!(obj.success);
        assert_eq!(obj.structured, Some(serde_json::json!({ "k": 1 })));
    }

    fn sdk_test_config() -> a3s_code_core::CodeConfig {
        a3s_code_core::CodeConfig {
            default_model: Some("openai/gpt-4o".to_string()),
            providers: vec![a3s_code_core::ProviderConfig {
                name: "openai".to_string(),
                api_key: Some("test-key".to_string()),
                base_url: None,
                headers: std::collections::HashMap::new(),
                session_id_header: None,
                models: vec![a3s_code_core::ModelConfig {
                    id: "gpt-4o".to_string(),
                    name: "GPT-4o".to_string(),
                    family: "gpt-4".to_string(),
                    api_key: None,
                    base_url: None,
                    headers: std::collections::HashMap::new(),
                    session_id_header: None,
                    attachment: false,
                    reasoning: false,
                    tool_call: true,
                    temperature: true,
                    release_date: None,
                    modalities: a3s_code_core::ModelModalities::default(),
                    cost: Default::default(),
                    limit: Default::default(),
                }],
            }],
            ..Default::default()
        }
    }

    fn build_test_session() -> Session {
        let agent = fallback_runtime()
            .block_on(RustAgent::from_config(sdk_test_config()))
            .unwrap();
        let session = agent.session("/tmp/a3s-code-node-sdk-api", None).unwrap();
        Session {
            inner: Arc::new(session),
        }
    }

    fn verification_report_json() -> serde_json::Value {
        serde_json::json!({
            "schema": "a3s.verification_report.v1",
            "subject": "sdk:test",
            "status": "passed",
            "checks": [{
                "id": "check:sdk",
                "kind": "test",
                "description": "Run SDK tests",
                "status": "passed",
                "required": true
            }]
        })
    }

    #[test]
    fn artifact_store_limits_maps_to_rust_session_options() {
        let opts = js_session_options_to_rust(Some(SessionOptions {
            artifact_store_limits: Some(ArtifactStoreLimits {
                max_artifacts: Some(3.0),
                max_bytes: Some(4096.0),
            }),
            ..Default::default()
        }))
        .unwrap();

        let limits = opts.artifact_store_limits.expect("limits");
        assert_eq!(limits.max_artifacts, 3);
        assert_eq!(limits.max_bytes, 4096);
    }

    #[test]
    fn artifact_store_limits_rejects_fractional_values() {
        let result = js_session_options_to_rust(Some(SessionOptions {
            artifact_store_limits: Some(ArtifactStoreLimits {
                max_artifacts: Some(1.5),
                max_bytes: Some(4096.0),
            }),
            ..Default::default()
        }));

        assert!(result.is_err());
    }

    #[test]
    fn verification_reports_from_value_accepts_array_and_single_report() {
        let single = verification_reports_from_value(verification_report_json()).unwrap();
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].subject, "sdk:test");

        let array =
            verification_reports_from_value(serde_json::json!([verification_report_json()]))
                .unwrap();
        assert_eq!(array.len(), 1);
        assert_eq!(array[0].checks[0].id, "check:sdk");
    }

    #[test]
    fn session_records_verification_reports() {
        let session = build_test_session();
        session
            .record_verification_reports(serde_json::json!([verification_report_json()]))
            .unwrap();

        let reports = session.verification_reports().unwrap();
        assert_eq!(reports.as_array().unwrap().len(), 1);
        assert_eq!(reports[0]["subject"], "sdk:test");

        let summary = session.verification_summary().unwrap();
        assert_eq!(summary["status"], "passed");
    }

    #[test]
    fn session_get_artifact_returns_null_for_missing_uri() {
        let session = build_test_session();
        let artifact = session
            .get_artifact("a3s://tool-output/missing".to_string())
            .unwrap();
        assert!(artifact.is_null());
    }

    /// Phase 8 alignment: when the Rust core surfaces a typed
    /// `ToolErrorKind`, `tool_result_from_core` must round-trip it into
    /// `error_kind_json` on the SDK shape. Tests both the JSON envelope
    /// and the discriminator (`type`) field.
    #[test]
    fn tool_result_from_core_threads_error_kind_json() {
        let core_result = a3s_code_core::ToolCallResult {
            name: "edit".to_string(),
            output: "Concurrent modification detected".to_string(),
            exit_code: 1,
            metadata: None,
            error_kind: Some(a3s_code_core::ToolErrorKind::VersionConflict {
                path: "doc.md".to_string(),
                expected: "etag-1".to_string(),
                actual: Some("etag-2".to_string()),
            }),
        };
        let sdk_result = tool_result_from_core(core_result);
        let json_str = sdk_result
            .error_kind_json
            .expect("typed error_kind must round-trip into error_kind_json");
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed["type"], "version_conflict");
        assert_eq!(parsed["path"], "doc.md");
        assert_eq!(parsed["expected"], "etag-1");
        assert_eq!(parsed["actual"], "etag-2");
    }

    #[test]
    fn tool_result_from_core_leaves_error_kind_json_none_on_success() {
        let core_result = a3s_code_core::ToolCallResult {
            name: "read".to_string(),
            output: "hello".to_string(),
            exit_code: 0,
            metadata: None,
            error_kind: None,
        };
        let sdk_result = tool_result_from_core(core_result);
        assert!(sdk_result.error_kind_json.is_none());
    }

    #[test]
    fn planning_mode_parser_accepts_explicit_tristate() {
        assert!(matches!(
            parse_planning_mode("auto").unwrap(),
            RustPlanningMode::Auto
        ));
        assert!(matches!(
            parse_planning_mode("enabled").unwrap(),
            RustPlanningMode::Enabled
        ));
        assert!(matches!(
            parse_planning_mode("disabled").unwrap(),
            RustPlanningMode::Disabled
        ));
        assert!(parse_planning_mode("sometimes").is_err());
    }

    #[test]
    fn planning_mode_takes_precedence_over_legacy_bool() {
        let opts =
            apply_planning_mode(RustSessionOptions::new(), Some("disabled"), Some(true)).unwrap();
        assert!(matches!(opts.planning_mode, RustPlanningMode::Disabled));

        let opts = apply_planning_mode(RustSessionOptions::new(), None, Some(true)).unwrap();
        assert!(matches!(opts.planning_mode, RustPlanningMode::Enabled));
    }

    #[test]
    fn session_options_maps_parallel_delegation_controls() {
        let opts = js_session_options_to_rust(Some(SessionOptions {
            max_parallel_tasks: Some(3),
            auto_delegation: Some(AutoDelegationOptions {
                enabled: Some(true),
                auto_parallel: Some(true),
                min_confidence: Some(0.8),
                max_tasks: Some(2),
            }),
            auto_parallel: Some(false),
            ..Default::default()
        }))
        .unwrap();

        assert_eq!(opts.max_parallel_tasks, Some(3));
        assert_eq!(opts.auto_parallel_delegation, Some(false));
        let auto = opts.auto_delegation.expect("auto delegation options");
        assert!(auto.enabled);
        assert!(!auto.auto_parallel);
        assert!((auto.min_confidence - 0.8).abs() < f32::EPSILON);
        assert_eq!(auto.max_tasks, 2);
    }

    #[test]
    fn confirmation_policy_maps_yolo_lanes_to_rust_options() {
        let opts = js_session_options_to_rust(Some(SessionOptions {
            confirmation_policy: Some(ConfirmationPolicy {
                enabled: Some(true),
                default_timeout_ms: Some(5_000),
                timeout_action: Some("auto_approve".to_string()),
                yolo_lanes: Some(vec!["query".to_string(), "execute".to_string()]),
            }),
            ..Default::default()
        }))
        .unwrap();

        let policy = opts.confirmation_policy.unwrap();
        assert!(policy.enabled);
        assert_eq!(policy.default_timeout_ms, 5_000);
        assert!(matches!(
            policy.timeout_action,
            RustTimeoutAction::AutoApprove
        ));
        assert!(policy.yolo_lanes.contains(&RustSessionLane::Query));
        assert!(policy.yolo_lanes.contains(&RustSessionLane::Execute));
    }

    #[test]
    fn worker_agent_spec_maps_to_rust_session_options() {
        let opts = js_session_options_to_rust(Some(SessionOptions {
            worker_agents: Some(vec![WorkerAgentSpec {
                name: "frontend-cow".to_string(),
                description: "Fix frontend bugs".to_string(),
                kind: Some("implementer".to_string()),
                model: Some("openai/gpt-4o".to_string()),
                max_steps: Some(8),
                ..Default::default()
            }]),
            ..Default::default()
        }))
        .unwrap();

        assert_eq!(opts.worker_agents.len(), 1);
        assert_eq!(opts.worker_agents[0].name, "frontend-cow");
        assert_eq!(opts.worker_agents[0].kind.as_str(), "implementer");
        assert_eq!(
            opts.worker_agents[0]
                .model
                .as_ref()
                .map(|model| model.model_ref()),
            Some("openai/gpt-4o".to_string())
        );
    }

    #[test]
    fn local_workspace_backend_maps_to_rust_session_options() {
        let opts = js_session_options_to_rust(Some(SessionOptions {
            workspace_backend: Some(JsWorkspaceBackend {
                kind: "local".to_string(),
                root: Some(".".to_string()),
                s3: None,
            }),
            ..Default::default()
        }))
        .unwrap();

        assert!(opts.workspace_services.is_some());
    }

    #[test]
    fn workspace_backend_rejects_missing_local_root() {
        let result = js_session_options_to_rust(Some(SessionOptions {
            workspace_backend: Some(JsWorkspaceBackend {
                kind: "local".to_string(),
                root: None,
                s3: None,
            }),
            ..Default::default()
        }));

        assert!(result.is_err());
    }

    #[test]
    fn s3_workspace_backend_maps_to_rust_session_options() {
        let opts = js_session_options_to_rust(Some(SessionOptions {
            workspace_backend: Some(JsWorkspaceBackend {
                kind: "s3".to_string(),
                root: None,
                s3: Some(JsS3BackendConfig {
                    endpoint: Some("https://minio.local:9000".to_string()),
                    region: Some("us-east-1".to_string()),
                    access_key_id: "AKIA".to_string(),
                    secret_access_key: "secret".to_string(),
                    session_token: None,
                    bucket: "workspace".to_string(),
                    prefix: "users/u1/sessions/s1".to_string(),
                    force_path_style: Some(true),
                    ..Default::default()
                }),
            }),
            ..Default::default()
        }))
        .unwrap();

        let services = opts.workspace_services.expect("s3 backend builds services");
        let caps = services.capabilities();
        assert!(caps.read);
        assert!(caps.write);
        assert!(!caps.exec, "S3 must not expose bash");
        assert!(!caps.git, "S3 must not expose git");
        assert!(!caps.search, "S3 must not expose grep/glob");
    }

    #[test]
    fn workspace_backend_rejects_missing_s3_config() {
        let result = js_session_options_to_rust(Some(SessionOptions {
            workspace_backend: Some(JsWorkspaceBackend {
                kind: "s3".to_string(),
                root: None,
                s3: None,
            }),
            ..Default::default()
        }));

        assert!(result.is_err());
    }

    #[test]
    fn s3_phase1_3_options_thread_through_to_core() {
        let opts = js_session_options_to_rust(Some(SessionOptions {
            workspace_backend: Some(JsWorkspaceBackend {
                kind: "s3".to_string(),
                root: None,
                s3: Some(JsS3BackendConfig {
                    access_key_id: "AKIA".to_string(),
                    secret_access_key: "secret".to_string(),
                    bucket: "workspace".to_string(),
                    prefix: "u1/s1".to_string(),
                    max_read_bytes: Some(4 * 1024 * 1024),
                    search_enabled: Some(true),
                    max_objects_scanned: Some(250),
                    max_grep_bytes_per_object: Some(512 * 1024),
                    ..Default::default()
                }),
            }),
            ..Default::default()
        }))
        .unwrap();

        let services = opts.workspace_services.expect("s3 backend builds services");
        assert!(
            services.capabilities().search,
            "searchEnabled=true must enable the search capability"
        );
        assert!(services.search().is_some());
    }

    #[test]
    fn remote_git_attaches_on_top_of_s3_backend() {
        let opts = js_session_options_to_rust(Some(SessionOptions {
            workspace_backend: Some(JsWorkspaceBackend {
                kind: "s3".to_string(),
                root: None,
                s3: Some(JsS3BackendConfig {
                    access_key_id: "AKIA".to_string(),
                    secret_access_key: "secret".to_string(),
                    bucket: "workspace".to_string(),
                    prefix: "u1/s1".to_string(),
                    ..Default::default()
                }),
            }),
            remote_git: Some(JsRemoteGitBackendConfig {
                base_url: "https://gitserver.internal".to_string(),
                repo_id: "u1/s1".to_string(),
                bearer_token: Some("tok".to_string()),
                request_timeout_ms: Some(10_000),
                ..Default::default()
            }),
            ..Default::default()
        }))
        .unwrap();

        let services = opts.workspace_services.expect("services built");
        assert!(
            services.git().is_some(),
            "remoteGit must register a git provider"
        );
        assert!(services.git_stash().is_some());
        // Worktree is intentionally not available — see RFC §8.
        assert!(services.git_worktree().is_none());
        assert!(services.capabilities().git);
    }

    #[test]
    fn remote_git_without_workspace_backend_errors_clearly() {
        let result = js_session_options_to_rust(Some(SessionOptions {
            workspace_backend: None,
            remote_git: Some(JsRemoteGitBackendConfig {
                base_url: "https://gitserver".to_string(),
                repo_id: "r".to_string(),
                ..Default::default()
            }),
            ..Default::default()
        }));

        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("workspaceBackend"),
            "error message must mention the missing field, got: {}",
            err
        );
    }

    #[test]
    fn confirmation_policy_rejects_invalid_yolo_lane() {
        let result = js_session_options_to_rust(Some(SessionOptions {
            confirmation_policy: Some(ConfirmationPolicy {
                enabled: Some(true),
                yolo_lanes: Some(vec!["unknown".to_string()]),
                ..Default::default()
            }),
            ..Default::default()
        }));

        assert!(result.is_err());
    }

    #[test]
    fn session_options_reject_invalid_permission_decision() {
        let result = js_session_options_to_rust(Some(SessionOptions {
            permission_policy: Some(PermissionPolicy {
                default_decision: Some("maybe".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        }));

        assert!(result.is_err());
    }

    #[test]
    fn queue_config_rejects_invalid_lane_handler() {
        let mut lane_handlers = std::collections::HashMap::new();
        lane_handlers.insert(
            "unknown".to_string(),
            LaneHandlerConfig {
                mode: "external".to_string(),
                timeout_ms: None,
            },
        );

        let result = js_session_options_to_rust(Some(SessionOptions {
            queue_config: Some(SessionQueueConfig {
                lane_handlers: Some(lane_handlers),
                ..Default::default()
            }),
            ..Default::default()
        }));

        assert!(result.is_err());
    }

    #[test]
    fn program_options_normalize_to_script_tool_contract() {
        let args = normalize_program_script_options(serde_json::json!({
            "source": "async function run(ctx, inputs) { return inputs; }",
            "inputs": { "needle": "auth" },
            "allowedTools": ["grep", "read"],
            "limits": { "maxToolCalls": 4 }
        }))
        .unwrap();

        assert_eq!(args["type"], "script");
        assert_eq!(args["language"], "javascript");
        assert_eq!(args["allowed_tools"], serde_json::json!(["grep", "read"]));
        assert_eq!(args["inputs"]["needle"], "auth");
    }

    #[test]
    fn delegate_task_options_use_core_task_schema() {
        let args = delegate_task_options_to_args(DelegateTaskOptions {
            agent: "explore".to_string(),
            description: "Find auth files".to_string(),
            prompt: "Inspect auth files".to_string(),
            background: Some(false),
            max_steps: Some(3),
        });

        assert_eq!(args["agent"], "explore");
        assert_eq!(args["description"], "Find auth files");
        assert_eq!(args["prompt"], "Inspect auth files");
        assert_eq!(args["background"], false);
        assert_eq!(args["max_steps"], 3);
        assert!(args.get("role").is_none());
    }

    #[test]
    fn parallel_task_options_use_core_parallel_task_schema() {
        let args = parallel_task_options_to_args(vec![
            DelegateTaskOptions {
                agent: "explore".to_string(),
                description: "Find tests".to_string(),
                prompt: "Locate tests".to_string(),
                background: None,
                max_steps: None,
            },
            DelegateTaskOptions {
                agent: "verification".to_string(),
                description: "Check risks".to_string(),
                prompt: "Review risks".to_string(),
                background: None,
                max_steps: Some(2),
            },
        ]);

        assert_eq!(args["tasks"].as_array().unwrap().len(), 2);
        assert_eq!(args["tasks"][0]["agent"], "explore");
        assert_eq!(args["tasks"][1]["agent"], "verification");
        assert_eq!(args["tasks"][1]["max_steps"], 2);
    }

    #[test]
    fn mcp_config_object_accepts_nested_transport_and_timeout_ms() {
        let config = normalize_mcp_server_config(serde_json::json!({
            "name": "github",
            "transport": {
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"]
            },
            "env": { "GITHUB_TOKEN": "test" },
            "timeoutMs": 1500
        }))
        .unwrap();

        assert_eq!(config.name, "github");
        assert_eq!(config.tool_timeout_secs, 2);
        match config.transport {
            a3s_code_core::mcp::protocol::McpTransportConfig::Stdio { command, args } => {
                assert_eq!(command, "npx");
                assert_eq!(args, vec!["-y", "@modelcontextprotocol/server-github"]);
            }
            _ => panic!("expected stdio transport"),
        }
    }

    #[test]
    fn mcp_config_object_accepts_streamable_http_alias() {
        let config = normalize_mcp_server_config(serde_json::json!({
            "name": "remote",
            "transport": {
                "type": "streamable_http",
                "url": "https://example.com/mcp",
                "headers": { "Authorization": "Bearer token" }
            }
        }))
        .unwrap();

        match config.transport {
            a3s_code_core::mcp::protocol::McpTransportConfig::StreamableHttp { url, headers } => {
                assert_eq!(url, "https://example.com/mcp");
                assert_eq!(
                    headers.get("Authorization").map(String::as_str),
                    Some("Bearer token")
                );
            }
            _ => panic!("expected streamable-http transport"),
        }
    }
}
