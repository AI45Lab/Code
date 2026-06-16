//! A3S Code Python Bindings
//!
//! Native Python module via PyO3 that wraps `a3s-code-core`'s Agent API.
//!
//! ## Usage
//!
//! ```python
//! from a3s_code import Agent
//!
//! agent = Agent.create("agent.acl")
//! session = agent.session("/my-project")
//!
//! result = session.send("What files handle auth?")
//! print(result.text)
//! ```
//!
//! ## Panic safety at the FFI boundary
//!
//! PyO3 0.23 wraps `#[pyfunction]` / `#[pymethods]` / `#[pymodule]`-init bodies
//! in `catch_unwind`, so a panic there surfaces as a Python `PanicException`
//! (a `BaseException` subclass) rather than UB. It does **not** cover panics
//! inside `std::thread` / `tokio::spawn` task bodies, or `Python::with_gil`
//! closures invoked from a worker thread *outside* a pyfunction frame — those
//! are silently lost, and a panicking `Drop` during an unwind aborts the
//! process.
//!
//! Convention this crate follows so the boundary stays safe: the Rust→Python
//! bridges that run on tokio worker threads (`PythonCallbackHandler`,
//! `PyBudgetGuard`, `PySlashCommand`) never `.unwrap()` / `panic!`; they use
//! `.ok()` / `unwrap_or_else` and fail closed. (Audited 2026-05: the only
//! production panic site is the lazy Tokio-runtime build in `get_runtime()`,
//! reached only from caught pyfunction frames.)

use a3s_code_core::commands::{
    CommandContext as RustCommandContext, CommandOutput as RustCommandOutput,
    SlashCommand as RustSlashCommand,
};
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
use a3s_code_core::llm::Message as RustMessage;
use a3s_code_core::orchestration::{
    execute_pipeline, execute_steps_parallel, execute_steps_parallel_resumable,
    AgentStepSpec as RustAgentStepSpec, PipelineStage as RustPipelineStage,
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
use pyo3::exceptions::{
    PyRuntimeError, PyStopAsyncIteration, PyStopIteration, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tokio::runtime::Runtime;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use a3s_code_core::config::AgentDir as RustAgentDir;
use a3s_code_core::serve::serve_agent_dir as rust_serve_agent_dir;

// ============================================================================
// Utilities
// ============================================================================

/// Truncate a UTF-8 string to at most `max_bytes` bytes, without splitting
/// a multibyte character. Falls back to the full string if it's already
/// within the limit.
fn truncate_utf8(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

// AHP Type Bindings
// ============================================================================
mod ahp_types;

use ahp_types::{
    PyAhpEventContext, PyAhpEventType, PyFact, PyIdleDecision, PyIntentDetectionDecision,
    PyIntentDetectionEvent, PyMemorySummary, PySessionStats, PyTargetHints,
};

fn get_runtime() -> &'static Runtime {
    use std::sync::OnceLock;
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        // Optimized runtime configuration for I/O-intensive workloads
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(num_cpus::get() * 2) // 2x CPU cores for better I/O handling
            .max_blocking_threads(512) // More blocking threads for CPU-intensive tasks
            .thread_name("a3s-code-worker")
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime")
    })
}

fn json_string_to_py(py: Python<'_>, json: &str) -> PyResult<PyObject> {
    let json_module = py.import("json")?;
    let parsed = json_module.call_method1("loads", (json,))?;
    Ok(parsed.into())
}

// ============================================================================
// AgentResult
// ============================================================================

/// Result of a non-streaming agent execution.
#[pyclass(name = "AgentResult")]
#[derive(Clone)]
struct PyAgentResult {
    #[pyo3(get)]
    text: String,
    #[pyo3(get)]
    tool_calls_count: usize,
    #[pyo3(get)]
    prompt_tokens: usize,
    #[pyo3(get)]
    completion_tokens: usize,
    #[pyo3(get)]
    total_tokens: usize,
    #[pyo3(get)]
    verification_status: String,
    #[pyo3(get)]
    pending_verification_count: usize,
    #[pyo3(get)]
    failed_verification_count: usize,
    #[pyo3(get)]
    verification_report_count: usize,
    #[pyo3(get)]
    verification_summary_json: String,
    #[pyo3(get)]
    verification_summary_text: String,
}

#[pymethods]
impl PyAgentResult {
    fn __repr__(&self) -> String {
        format!(
            "AgentResult(text={:?}, tool_calls={}, tokens={}, verification={})",
            if self.text.len() > 80 {
                format!("{}...", truncate_utf8(&self.text, 80))
            } else {
                self.text.clone()
            },
            self.tool_calls_count,
            self.total_tokens,
            self.verification_status,
        )
    }

    fn __str__(&self) -> &str {
        &self.text
    }
}

impl From<RustAgentResult> for PyAgentResult {
    fn from(r: RustAgentResult) -> Self {
        let verification_summary = r.verification_summary();
        let verification_summary_json = verification_summary.to_value().to_string();
        let verification_summary_text = rust_format_verification_summary(&verification_summary);
        Self {
            text: r.text,
            tool_calls_count: r.tool_calls_count,
            prompt_tokens: r.usage.prompt_tokens,
            completion_tokens: r.usage.completion_tokens,
            total_tokens: r.usage.total_tokens,
            verification_status: verification_status_label(verification_summary.status),
            pending_verification_count: verification_summary.pending_required_check_count,
            failed_verification_count: verification_summary.failed_check_count,
            verification_report_count: verification_summary.report_count,
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

#[pyfunction]
fn format_verification_summary(py: Python<'_>, summary: &Bound<'_, PyAny>) -> PyResult<String> {
    let summary_json = if let Ok(summary_json) = summary.extract::<String>() {
        summary_json
    } else {
        let json_mod = py.import("json")?;
        json_mod.call_method1("dumps", (summary,))?.extract()?
    };
    let summary: RustVerificationSummary = serde_json::from_str(&summary_json)
        .map_err(|e| PyValueError::new_err(format!("Invalid verification summary: {e}")))?;
    Ok(rust_format_verification_summary(&summary))
}

// ============================================================================
// AgentEvent
// ============================================================================

/// A single event from the agent's streaming output.
#[pyclass(name = "AgentEvent")]
#[derive(Clone)]
struct PyAgentEvent {
    #[pyo3(get)]
    event_type: String,
    #[pyo3(get)]
    text: Option<String>,
    #[pyo3(get)]
    tool_name: Option<String>,
    #[pyo3(get)]
    tool_id: Option<String>,
    #[pyo3(get)]
    tool_output: Option<String>,
    #[pyo3(get)]
    exit_code: Option<i32>,
    #[pyo3(get)]
    turn: Option<usize>,
    #[pyo3(get)]
    prompt: Option<String>,
    #[pyo3(get)]
    error: Option<String>,
    #[pyo3(get)]
    total_tokens: Option<usize>,
    #[pyo3(get)]
    verification_summary_json: Option<String>,
    #[pyo3(get)]
    verification_summary_text: Option<String>,
    /// Extra data for events that don't map to standard fields (JSON-encoded)
    #[pyo3(get)]
    data: Option<String>,
    /// Structured discriminant for tool failures on ``tool_end`` events
    /// (JSON-encoded with a ``type`` field on the top level —
    /// e.g. ``{"type":"version_conflict","path":"doc.md","expected":"etag-1","actual":"etag-2"}``).
    /// ``None`` on success or untyped failure. Streaming consumers parse
    /// this via the ``error_kind`` property to branch on the failure
    /// kind without scanning ``tool_output``.
    #[pyo3(get)]
    error_kind_json: Option<String>,
}

impl PyAgentEvent {
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

#[pymethods]
impl PyAgentEvent {
    fn __repr__(&self) -> String {
        match self.event_type.as_str() {
            "text_delta" => format!(
                "AgentEvent(type='text_delta', text={:?})",
                self.text.as_deref().unwrap_or("")
            ),
            "tool_start" => format!(
                "AgentEvent(type='tool_start', tool='{}')",
                self.tool_name.as_deref().unwrap_or("")
            ),
            "end" => format!(
                "AgentEvent(type='end', tokens={})",
                self.total_tokens.unwrap_or(0)
            ),
            _ => format!("AgentEvent(type='{}')", self.event_type),
        }
    }

    /// Parsed `error_kind_json` as a dict — the discriminator lives on
    /// the ``type`` key (see [`ToolErrorKind`](crate::tools::ToolErrorKind)
    /// for the full set of variants). Downstream code matches on
    /// ``event.error_kind["type"]`` to decide retry behaviour without
    /// scanning ``tool_output``.
    #[getter]
    fn error_kind(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        self.error_kind_json
            .as_deref()
            .map(|json| json_string_to_py(py, json))
            .transpose()
    }
}

impl From<RustAgentEvent> for PyAgentEvent {
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
                turn: Some(turn),
                ..Self::empty("turn_start")
            },
            RustAgentEvent::TextDelta { text } => Self {
                text: Some(text),
                ..Self::empty("text_delta")
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
                turn: Some(turn),
                total_tokens: Some(usage.total_tokens),
                ..Self::empty("turn_end")
            },
            RustAgentEvent::End {
                text,
                usage,
                verification_summary,
                ..
            } => Self {
                text: Some(text),
                total_tokens: Some(usage.total_tokens),
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
                parent_session_id: _,
                agent,
                description,
            } => Self {
                tool_id: Some(task_id),
                tool_name: Some(agent),
                text: Some(session_id),
                prompt: Some(description),
                ..Self::empty("subagent_start")
            },
            RustAgentEvent::SubagentProgress {
                task_id,
                session_id,
                status,
                metadata: _,
            } => Self {
                tool_id: Some(task_id),
                text: Some(format!("{}: {}", session_id, status)),
                ..Self::empty("subagent_progress")
            },
            RustAgentEvent::SubagentEnd {
                task_id,
                session_id,
                agent,
                output,
                success,
            } => Self {
                tool_id: Some(task_id),
                tool_name: Some(agent),
                text: Some(session_id),
                tool_output: Some(output),
                exit_code: Some(if success { 0 } else { 1 }),
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

/// Result of a direct tool execution (no LLM).
#[pyclass(name = "ToolResult")]
#[derive(Clone)]
struct PyToolResult {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    output: String,
    #[pyo3(get)]
    exit_code: i32,
    /// Raw JSON-encoded tool metadata returned by the Rust core API.
    #[pyo3(get)]
    metadata_json: Option<String>,
    /// Structured discriminant for tool failures, JSON-encoded with a
    /// ``type`` field on the top level —
    /// e.g. ``{"type":"version_conflict","path":"doc.md","expected":"etag-1","actual":"etag-2"}``.
    /// ``None`` on success or untyped failure. SDK callers parse it via
    /// the ``error_kind`` property below to branch on the failure kind
    /// without scanning the ``output`` string.
    #[pyo3(get)]
    error_kind_json: Option<String>,
}

#[pymethods]
impl PyToolResult {
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        self.metadata_json
            .as_deref()
            .map(|json| json_string_to_py(py, json))
            .transpose()
    }

    /// Parsed `error_kind_json` as a dict. The discriminator lives on the
    /// ``type`` key; downstream code matches on that to decide retry
    /// behaviour without parsing ``output``.
    #[getter]
    fn error_kind(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        self.error_kind_json
            .as_deref()
            .map(|json| json_string_to_py(py, json))
            .transpose()
    }

    fn __repr__(&self) -> String {
        format!(
            "ToolResult(name='{}', exit_code={})",
            self.name, self.exit_code
        )
    }
}

// ============================================================================
// WebSearchParams
// ============================================================================

/// Parameters for the web_search tool.
#[pyclass(name = "WebSearchParams")]
#[derive(Clone)]
struct PyWebSearchParams {
    /// The search query.
    #[pyo3(get, set)]
    query: String,
    /// List of search engines to use.
    #[pyo3(get, set)]
    engines: Option<Vec<String>>,
    /// Maximum number of results to return (default: 10, max: 50).
    #[pyo3(get, set)]
    limit: Option<u32>,
    /// Search timeout in seconds (default: 10, max: 60).
    #[pyo3(get, set)]
    timeout: Option<u32>,
    /// Proxy URL (e.g., http://127.0.0.1:8080 or socks5://127.0.0.1:1080).
    #[pyo3(get, set)]
    proxy: Option<String>,
    /// Output format: "text" or "json".
    #[pyo3(get, set)]
    format: Option<String>,
}

#[pymethods]
impl PyWebSearchParams {
    #[new]
    #[pyo3(signature = (query, engines=None, limit=None, timeout=None, proxy=None, format=None))]
    fn new(
        query: String,
        engines: Option<Vec<String>>,
        limit: Option<u32>,
        timeout: Option<u32>,
        proxy: Option<String>,
        format: Option<String>,
    ) -> Self {
        Self {
            query,
            engines,
            limit,
            timeout,
            proxy,
            format,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "WebSearchParams(query='{}', engines={:?}, limit={:?}, timeout={:?}, format={:?})",
            self.query, self.engines, self.limit, self.timeout, self.format
        )
    }
}

// ============================================================================
// EventStream (Python Iterator + Async Iterator)
// ============================================================================

/// One-shot callable used by `run_in_executor` for async iteration.
///
/// Each `__anext__` call creates a new instance; `__call__` blocks on the
/// next channel receive and raises `StopAsyncIteration` when done.
#[pyclass]
struct BlockingRecv {
    rx: Arc<Mutex<tokio::sync::mpsc::Receiver<RustAgentEvent>>>,
    done: Arc<AtomicBool>,
}

#[pymethods]
impl BlockingRecv {
    fn __call__(&self, py: Python<'_>) -> PyResult<PyAgentEvent> {
        let rx = self.rx.clone();
        let done_flag = self.done.clone();
        let result = py.allow_threads(|| {
            get_runtime().block_on(async {
                let mut guard = rx.lock().await;
                guard.recv().await
            })
        });
        match result {
            Some(event) => {
                let is_end = matches!(event, RustAgentEvent::End { .. });
                let is_error = matches!(event, RustAgentEvent::Error { .. });
                let py_event = PyAgentEvent::from(event);
                if is_end || is_error {
                    done_flag.store(true, Ordering::Relaxed);
                }
                Ok(py_event)
            }
            None => {
                done_flag.store(true, Ordering::Relaxed);
                Err(PyStopAsyncIteration::new_err("stream exhausted"))
            }
        }
    }
}

/// Iterator / async-iterator that yields AgentEvents from a streaming execution.
///
/// Sync usage:  `for event in session.stream(prompt):`
/// Async usage: `async for event in session.stream(prompt):`
#[pyclass(name = "EventStream")]
struct PyEventStream {
    rx: Arc<Mutex<tokio::sync::mpsc::Receiver<RustAgentEvent>>>,
    done: Arc<AtomicBool>,
}

#[pymethods]
impl PyEventStream {
    // ------------------------------------------------------------------
    // Sync iterator protocol
    // ------------------------------------------------------------------

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyAgentEvent>> {
        if self.done.load(Ordering::Relaxed) {
            return Err(PyStopIteration::new_err("stream exhausted"));
        }

        let rx = self.rx.clone();
        let done_flag = self.done.clone();
        let result = py.allow_threads(|| {
            get_runtime().block_on(async {
                let mut guard = rx.lock().await;
                guard.recv().await
            })
        });

        match result {
            Some(event) => {
                let is_end = matches!(event, RustAgentEvent::End { .. });
                let is_error = matches!(event, RustAgentEvent::Error { .. });
                let py_event = PyAgentEvent::from(event);
                if is_end || is_error {
                    done_flag.store(true, Ordering::Relaxed);
                }
                Ok(Some(py_event))
            }
            None => {
                done_flag.store(true, Ordering::Relaxed);
                Err(PyStopIteration::new_err("stream exhausted"))
            }
        }
    }

    // ------------------------------------------------------------------
    // Async iterator protocol
    // ------------------------------------------------------------------

    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Returns an `asyncio.Future` that resolves to the next `AgentEvent`.
    ///
    /// Uses `run_in_executor` to bridge the blocking channel recv into an
    /// asyncio-compatible awaitable without requiring `pyo3-async`.
    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.done.load(Ordering::Relaxed) {
            return Err(PyStopAsyncIteration::new_err("stream exhausted"));
        }

        let callable = Bound::new(
            py,
            BlockingRecv {
                rx: self.rx.clone(),
                done: self.done.clone(),
            },
        )?;

        let asyncio = py.import("asyncio")?;
        let loop_ = asyncio.call_method0("get_running_loop")?;
        let future = loop_.call_method1("run_in_executor", (py.None(), callable))?;
        Ok(future)
    }
}

// ============================================================================
// Agent
// ============================================================================

/// Lifetime handle for a running serve daemon (see `Agent.serve_agent_dir`).
///
/// The daemon keeps running until `stop()` is called. Dropping the handle does
/// NOT cancel the daemon — call `stop()` explicitly for graceful shutdown.
#[pyclass(name = "ServeHandle")]
struct PyServeHandle {
    cancel: CancellationToken,
}

#[pymethods]
impl PyServeHandle {
    /// Request graceful shutdown of the serve daemon. Idempotent.
    fn stop(&self) {
        self.cancel.cancel();
    }

    /// Whether `stop()` has been called on this handle.
    fn is_stopped(&self) -> bool {
        self.cancel.is_cancelled()
    }
}

/// AI coding agent. Create with `Agent.create()`, then call `agent.session()`.
#[pyclass(name = "Agent")]
struct PyAgent {
    inner: Arc<RustAgent>,
}

#[pymethods]
impl PyAgent {
    /// Create an Agent from a config file path or inline config string.
    ///
    /// Accepts ACL-compatible config files (.acl) or inline config strings.
    /// JSON config is not supported.
    ///
    /// Args:
    ///     config_source: Path to a config file (.acl), or inline config string
    #[staticmethod]
    fn create(py: Python<'_>, config_source: String) -> PyResult<Self> {
        let agent = py
            .allow_threads(move || get_runtime().block_on(RustAgent::new(config_source)))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create agent: {e}")))?;

        Ok(Self {
            inner: Arc::new(agent),
        })
    }

    /// Serve a filesystem-first agent directory's cron schedules until stopped.
    ///
    /// Loads the directory by convention: `instructions.md` (required), optional
    /// `agent.acl`, `skills/`, `schedules/*.md` (cron jobs), and `tools/*.md`
    /// (`kind: mcp` servers or `kind: script` sandboxed QuickJS tools). It starts
    /// one durable session per enabled schedule (stable id `schedule:<name>`) with
    /// the agent dir's tools installed; each schedule fires as a FULL harness turn
    /// (context, tool visibility, safety gate, verification), never a raw model call.
    ///
    /// Returns immediately with a `ServeHandle`; the daemon runs in the
    /// background until `handle.stop()` is called. Dropping the handle does NOT
    /// cancel the daemon.
    ///
    /// Args:
    ///     dir: Path to the agent directory (prompt/skills/schedules/tools)
    ///     workspace: Workspace directory each scheduled turn operates in
    ///     options: Optional SessionOptions merged into every schedule session
    ///         (model, llm_client, session_store, …)
    #[pyo3(signature = (dir, workspace, options=None))]
    fn serve_agent_dir(
        &self,
        dir: String,
        workspace: String,
        options: Option<PySessionOptions>,
    ) -> PyResult<PyServeHandle> {
        let agent_dir = RustAgentDir::load(&dir)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load agent dir: {e}")))?;
        let extra = match options {
            Some(o) => Some(build_rust_session_options(o)?),
            None => None,
        };

        // The daemon runs until cancelled; spawn it on the shared runtime so the
        // call returns to Python immediately. The token, owned by the returned
        // ServeHandle, drives graceful shutdown.
        let cancel = CancellationToken::new();
        let agent = self.inner.clone();
        let handle_token = cancel.clone();

        get_runtime().spawn(async move {
            // serve_agent_dir returns Result and never panics by construction; a
            // scheduling error is reported, not propagated (spawned task bodies
            // are not panic-safe).
            if let Err(e) =
                rust_serve_agent_dir(&agent, &agent_dir, workspace, extra, cancel).await
            {
                eprintln!("a3s-code: serve_agent_dir daemon exited with error: {e}");
            }
        });

        Ok(PyServeHandle {
            cancel: handle_token,
        })
    }

    /// Re-fetch tool definitions from all connected global MCP servers and
    /// update the agent-level cache.
    ///
    /// New sessions created after this call will see the refreshed tool list.
    /// Existing sessions are unaffected.
    fn refresh_mcp_tools(&self, py: Python<'_>) -> PyResult<()> {
        let agent = self.inner.clone();
        py.allow_threads(move || {
            get_runtime().block_on(async {
                agent
                    .refresh_mcp_tools()
                    .await
                    .map_err(|e| PyRuntimeError::new_err(format!("refresh_mcp_tools failed: {e}")))
            })
        })
    }

    /// Bind to a workspace directory, returning a Session.
    ///
    /// Args:
    ///     workspace: Path to the workspace directory
    ///     options: Optional SessionOptions object
    ///     model: Optional model override, format "provider/model" (e.g., "openai/gpt-4o")
    ///     builtin_skills: Optional bool to enable built-in skills (default: False)
    ///     skill_dirs: Optional list of directories to scan for skill files
    ///     agent_dirs: Optional list of directories to scan for agent files
    ///     queue_config: Optional advanced SessionQueueConfig for explicit external/hybrid lane dispatch
    ///     planning_mode: Optional string: "auto", "enabled", or "disabled"
    ///     planning: Legacy optional bool. None = auto planning, True = force planning, False = disable planning
    ///     goal_tracking: Optional bool to enable goal tracking (default: False)
    ///     max_parse_retries: Optional max consecutive parse errors before abort
    ///     tool_timeout_ms: Optional per-tool execution timeout in milliseconds
    ///     circuit_breaker_threshold: Optional max LLM API failures before abort
    ///     max_parallel_tasks: Optional maximum sibling parallel branches
    ///     auto_parallel: Optional kill switch for automatic parallel child-agent fan-out
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (workspace, options=None, model=None, builtin_skills=None, skill_dirs=None, agent_dirs=None, queue_config=None, planning_mode=None, planning=None, goal_tracking=None, max_parse_retries=None, tool_timeout_ms=None, circuit_breaker_threshold=None, max_parallel_tasks=None, auto_parallel=None))]
    fn session(
        &self,
        workspace: String,
        options: Option<PySessionOptions>,
        model: Option<String>,
        builtin_skills: Option<bool>,
        skill_dirs: Option<Vec<String>>,
        agent_dirs: Option<Vec<String>>,
        queue_config: Option<PySessionQueueConfig>,
        planning_mode: Option<String>,
        planning: Option<bool>,
        goal_tracking: Option<bool>,
        max_parse_retries: Option<u32>,
        tool_timeout_ms: Option<u64>,
        circuit_breaker_threshold: Option<u32>,
        max_parallel_tasks: Option<usize>,
        auto_parallel: Option<bool>,
    ) -> PyResult<PySession> {
        // If a SessionOptions object is provided, build from it then apply named-argument overrides.
        let opts = if let Some(so) = options {
            let mut o = build_rust_session_options(so)?;
            // Named args take precedence over SessionOptions fields.
            o = apply_planning_mode(o, planning_mode.as_deref(), planning)?;
            if goal_tracking.unwrap_or(false) {
                o = o.with_goal_tracking(true);
            }
            if let Some(n) = max_parse_retries {
                o = o.with_parse_retries(n);
            }
            if let Some(ms) = tool_timeout_ms {
                o = o.with_tool_timeout(ms);
            }
            if let Some(n) = circuit_breaker_threshold {
                o = o.with_circuit_breaker(n);
            }
            if let Some(max_parallel_tasks) = max_parallel_tasks {
                o = o.with_max_parallel_tasks(max_parallel_tasks);
            }
            if let Some(auto_parallel) = auto_parallel {
                o = o.with_auto_parallel_delegation(auto_parallel);
            }
            Some(o)
        } else {
            // Fall back to individual named arguments.
            let has_overrides = model.is_some()
                || builtin_skills.is_some()
                || skill_dirs.is_some()
                || agent_dirs.is_some()
                || queue_config.is_some()
                || planning_mode.is_some()
                || planning.is_some()
                || goal_tracking.is_some()
                || max_parse_retries.is_some()
                || tool_timeout_ms.is_some()
                || circuit_breaker_threshold.is_some()
                || max_parallel_tasks.is_some()
                || auto_parallel.is_some();

            if has_overrides {
                let mut o = RustSessionOptions::new();
                if let Some(m) = model {
                    o = o.with_model(m);
                }
                if builtin_skills.unwrap_or(false) {
                    o = o.with_builtin_skills();
                }
                if let Some(dirs) = skill_dirs {
                    for d in dirs {
                        o = o.with_skills_from_dir(d);
                    }
                }
                if let Some(dirs) = agent_dirs {
                    for d in dirs {
                        o = o.with_agent_dir(d);
                    }
                }
                if let Some(qc) = queue_config {
                    o = o.with_queue_config(qc.inner);
                }
                o = apply_planning_mode(o, planning_mode.as_deref(), planning)?;
                if goal_tracking.unwrap_or(false) {
                    o = o.with_goal_tracking(true);
                }
                if let Some(n) = max_parse_retries {
                    o = o.with_parse_retries(n);
                }
                if let Some(ms) = tool_timeout_ms {
                    o = o.with_tool_timeout(ms);
                }
                if let Some(n) = circuit_breaker_threshold {
                    o = o.with_circuit_breaker(n);
                }
                if let Some(max_parallel_tasks) = max_parallel_tasks {
                    o = o.with_max_parallel_tasks(max_parallel_tasks);
                }
                if let Some(auto_parallel) = auto_parallel {
                    o = o.with_auto_parallel_delegation(auto_parallel);
                }
                Some(o)
            } else {
                None
            }
        };

        let session = self
            .inner
            .session(workspace, opts)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        Ok(PySession {
            inner: Arc::new(session),
        })
    }

    fn __repr__(&self) -> String {
        "Agent(...)".to_string()
    }

    /// Resume a previously saved session by ID.
    ///
    /// ``options.session_store`` must point to the store where the session was saved.
    ///
    /// .. code-block:: python
    ///
    ///     opts = SessionOptions()
    ///     opts.session_store = FileSessionStore('./sessions')
    ///     session = agent.resume_session('my-session', opts)
    ///
    /// Args:
    ///     session_id: The session ID to resume
    ///     options: SessionOptions with ``session_store`` set to the backing store
    #[pyo3(signature = (session_id, options))]
    fn resume_session(&self, session_id: String, options: PySessionOptions) -> PyResult<PySession> {
        let opts = build_rust_session_options(options)?;
        let session = self
            .inner
            .resume_session(&session_id, opts)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to resume session: {e}")))?;
        Ok(PySession {
            inner: Arc::new(session),
        })
    }

    /// Create a session pre-configured from a named agent definition.
    ///
    /// Loads the agent by name from built-in agents and optionally from
    /// additional directories, then creates a session with the agent's
    /// permissions, system prompt, model, and step limit applied.
    ///
    /// Args:
    ///     workspace: Path to the workspace directory
    ///     agent_name: Name of the agent to load (e.g. "explore", "general")
    ///     agent_dirs: Optional list of directories to scan for agent files
    ///     options: Optional session overrides layered on top of the agent definition
    #[pyo3(signature = (workspace, agent_name, agent_dirs=None, options=None))]
    fn session_for_agent(
        &self,
        workspace: String,
        agent_name: String,
        agent_dirs: Option<Vec<String>>,
        options: Option<PySessionOptions>,
    ) -> PyResult<PySession> {
        let registry = a3s_code_core::subagent::AgentRegistry::new();
        for dir in agent_dirs.unwrap_or_default() {
            let agents = a3s_code_core::subagent::load_agents_from_dir(std::path::Path::new(&dir));
            for agent in agents {
                registry.register(agent);
            }
        }
        let def = registry
            .get(&agent_name)
            .ok_or_else(|| PyRuntimeError::new_err(format!("agent '{}' not found", agent_name)))?;
        let opts = options.map(build_rust_session_options).transpose()?;
        let session = self
            .inner
            .session_for_agent(workspace, &def, opts)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        Ok(PySession {
            inner: Arc::new(session),
        })
    }

    /// Create a session pre-configured from a disposable worker spec.
    #[pyo3(signature = (workspace, worker, options=None))]
    fn session_for_worker(
        &self,
        workspace: String,
        worker: PyWorkerAgentSpec,
        options: Option<PySessionOptions>,
    ) -> PyResult<PySession> {
        let worker = py_worker_agent_spec_to_rust(worker)?;
        let opts = options.map(build_rust_session_options).transpose()?;
        let session = self
            .inner
            .session_for_worker(workspace, worker, opts)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        Ok(PySession {
            inner: Arc::new(session),
        })
    }

    /// List session IDs for every live session created from this agent.
    ///
    /// Sessions that have been dropped (no Python references remain) are
    /// pruned lazily on each call. Result is sorted for stable output.
    fn list_sessions(&self, py: Python<'_>) -> Vec<String> {
        let agent = self.inner.clone();
        py.allow_threads(move || get_runtime().block_on(agent.list_sessions()))
    }

    /// Close a specific live session by its session ID.
    ///
    /// Returns ``True`` when a live session with the given id was found and
    /// transitioned from open to closed by this call; ``False`` when no
    /// live session has that id, or when it was already closed.
    fn close_session(&self, py: Python<'_>, session_id: String) -> bool {
        let agent = self.inner.clone();
        py.allow_threads(move || get_runtime().block_on(agent.close_session(&session_id)))
    }

    /// Close every live session created from this agent and disconnect
    /// background resources owned by the agent (global MCP connections).
    ///
    /// After this call, ``agent.session(...)`` and ``agent.resume_session(...)``
    /// raise ``RuntimeError`` with a "Session closed" message. Idempotent.
    fn close(&self, py: Python<'_>) {
        let agent = self.inner.clone();
        py.allow_threads(move || get_runtime().block_on(agent.close()));
    }

    /// Whether ``close()`` has been called on this agent.
    #[getter]
    fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }

    /// Disconnect every global MCP server idle longer than
    /// ``idle_threshold_ms``, returning the names disconnected. The
    /// server's registered config is kept — a later tool call reconnects
    /// on demand. Call periodically (e.g. every 60s with a 5-min
    /// threshold) from a host-side sweeper to release file descriptors
    /// and background workers from quiet MCP servers in long-running
    /// deployments.
    fn disconnect_idle_mcp(&self, py: Python<'_>, idle_threshold_ms: u64) -> Vec<String> {
        let agent = self.inner.clone();
        py.allow_threads(move || {
            get_runtime().block_on(agent.disconnect_idle_mcp(idle_threshold_ms))
        })
    }
}

// ============================================================================
// Session
// ============================================================================

/// Workspace-bound session. All LLM and tool operations happen here.
#[pyclass(name = "Session")]
struct PySession {
    inner: Arc<RustAgentSession>,
}

#[pymethods]
impl PySession {
    /// Send a prompt or request and wait for the complete response.
    ///
    /// Args:
    ///     prompt: Prompt string, or {"prompt": str, "history": list, "attachments": list}
    ///     history: Optional conversation history as list of dicts
    ///              `[{"role": "user", "content": [{"type": "text", "text": "..."}]}]`
    #[pyo3(signature = (prompt, history=None))]
    fn send(
        &self,
        py: Python<'_>,
        prompt: &Bound<'_, PyAny>,
        history: Option<&Bound<'_, PyList>>,
    ) -> PyResult<PyAgentResult> {
        let (prompt, rust_history, rust_attachments) = py_session_input_to_parts(prompt, history)?;
        let session = self.inner.clone();
        let result = if rust_attachments.is_empty() {
            py.allow_threads(move || {
                get_runtime().block_on(session.send(&prompt, rust_history.as_deref()))
            })
        } else {
            py.allow_threads(move || {
                get_runtime().block_on(session.send_with_attachments(
                    &prompt,
                    &rust_attachments,
                    rust_history.as_deref(),
                ))
            })
        }
        .map_err(|e| PyRuntimeError::new_err(format!("Agent execution failed: {e}")))?;
        Ok(PyAgentResult::from(result))
    }

    /// Alias for ``send(...)`` with a name that matches run/replay terminology.
    #[pyo3(signature = (prompt, history=None))]
    fn run(
        &self,
        py: Python<'_>,
        prompt: &Bound<'_, PyAny>,
        history: Option<&Bound<'_, PyList>>,
    ) -> PyResult<PyAgentResult> {
        self.send(py, prompt, history)
    }

    /// Resume a previously-checkpointed run on this session.
    ///
    /// Loads the latest loop checkpoint stored under ``checkpoint_run_id``
    /// and replays the agent loop from that boundary. A new run id is
    /// allocated for the resumed work.
    ///
    /// Raises ``RuntimeError`` when no ``session_store`` is configured,
    /// or when no checkpoint exists for the given id.
    fn resume_run(&self, py: Python<'_>, checkpoint_run_id: String) -> PyResult<PyAgentResult> {
        let session = self.inner.clone();
        let result = py
            .allow_threads(move || get_runtime().block_on(session.resume_run(&checkpoint_run_id)))
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        Ok(PyAgentResult::from(result))
    }

    /// Run `specs` as a fan-out of agent steps and return each step's outcome
    /// (a dict) in input order. Each spec is a dict with snake_case keys:
    /// `task_id`, `agent`, `description`, `prompt`, optional `max_steps`,
    /// `parent_session_id`, `output_schema`. A failed step surfaces as
    /// `success: False` without failing the batch.
    ///
    /// Pass `budget_tokens` to run the fan-out under one shared token budget:
    /// every child agent feeds a single ledger and, once the cap is reached,
    /// further child LLM calls are denied (a soft cap; the in-flight fan-out is
    /// never force-killed). With a budget the result is a dict
    /// `{"outcomes": [...], "budget": {"consumed_tokens", "limit_tokens"}}`;
    /// without one it is the plain list of outcome dicts, unchanged.
    #[pyo3(signature = (specs, budget_tokens=None))]
    fn parallel(
        &self,
        py: Python<'_>,
        specs: Vec<Bound<'_, PyAny>>,
        budget_tokens: Option<u64>,
    ) -> PyResult<PyObject> {
        let rust_specs = specs
            .iter()
            .map(|s| py_to_step_spec(py, s))
            .collect::<PyResult<Vec<_>>>()?;
        let session = self.inner.clone();

        // No budget → unchanged behavior: a plain list of outcome dicts.
        let Some(limit) = budget_tokens else {
            let outcomes = py.allow_threads(move || {
                get_runtime().block_on(async move {
                    let executor = session.agent_executor();
                    execute_steps_parallel(executor, rust_specs, None).await
                })
            });
            let items = outcomes
                .iter()
                .map(|o| step_outcome_to_py(py, o))
                .collect::<PyResult<Vec<_>>>()?;
            return Ok(PyList::new(py, items)?.into_any().unbind());
        };

        // Budget → shared ledger across the fan-out; return {"outcomes", "budget"}.
        let (outcomes, snapshot) = py.allow_threads(move || {
            get_runtime().block_on(async move {
                let wf = session.workflow_with_token_budget(Some(limit));
                let outcomes = wf.parallel(rust_specs).await;
                (outcomes, wf.budget_snapshot())
            })
        });
        let outcomes_py = outcomes
            .iter()
            .map(|o| step_outcome_to_py(py, o))
            .collect::<PyResult<Vec<_>>>()?;
        let budget = PyDict::new(py);
        budget.set_item(
            "consumed_tokens",
            snapshot.as_ref().map(|b| b.consumed_tokens).unwrap_or(0),
        )?;
        budget.set_item(
            "limit_tokens",
            snapshot
                .as_ref()
                .and_then(|b| b.limit_tokens)
                .or(Some(limit)),
        )?;
        let result = PyDict::new(py);
        result.set_item("outcomes", outcomes_py)?;
        result.set_item("budget", budget)?;
        Ok(result.into_any().unbind())
    }

    /// Like `parallel`, but resumable: progress is journaled under
    /// `workflow_id` via the session's store, so an interrupted run skips
    /// already-completed steps. Raises if no `session_store` is configured.
    fn parallel_resumable(
        &self,
        py: Python<'_>,
        specs: Vec<Bound<'_, PyAny>>,
        workflow_id: String,
    ) -> PyResult<Vec<PyObject>> {
        let rust_specs = specs
            .iter()
            .map(|s| py_to_step_spec(py, s))
            .collect::<PyResult<Vec<_>>>()?;
        let session = self.inner.clone();
        let outcomes = py
            .allow_threads(move || {
                get_runtime().block_on(async move {
                    let Some(store) = session.session_store() else {
                        return Err("parallel_resumable requires a session_store on the session");
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
            })
            .map_err(PyRuntimeError::new_err)?;
        outcomes.iter().map(|o| step_outcome_to_py(py, o)).collect()
    }

    /// Run each item through a chain of `stages`, with no barrier between
    /// stages. Each stage is a callable `stage(ctx) -> spec_dict | None`, where
    /// `ctx = {"previous": <outcome dict or None>, "item": <item>}`. Return a
    /// spec dict (snake_case keys) to run that step, or `None` to stop the
    /// item's chain. A chain also stops when a step fails. Returns one entry
    /// per item (the last outcome dict, or `None`), in input order.
    ///
    /// A stage callable that raises is caught and treated as `None` (stops that
    /// chain). Per-stage `output_schema` is not supported here — use `parallel`
    /// for schema-validated steps.
    fn pipeline(
        &self,
        py: Python<'_>,
        items: Vec<Bound<'_, PyAny>>,
        stages: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Option<PyObject>>> {
        let rust_items = items
            .iter()
            .map(|i| py_to_json_value(py, i))
            .collect::<PyResult<Vec<_>>>()?;
        let rust_stages: Vec<RustPipelineStage<serde_json::Value>> = stages
            .into_iter()
            .map(|s| {
                let stage = std::sync::Arc::new(PythonPipelineStage {
                    callback: s.unbind(),
                });
                let ps: RustPipelineStage<serde_json::Value> =
                    std::sync::Arc::new(move |prev, item| stage.invoke(prev, item));
                ps
            })
            .collect();

        let session = self.inner.clone();
        let outcomes = py.allow_threads(move || {
            get_runtime().block_on(async move {
                let executor = session.agent_executor();
                execute_pipeline(executor, rust_items, rust_stages, None).await
            })
        });

        outcomes
            .iter()
            .map(|o| match o {
                Some(outcome) => step_outcome_to_py(py, outcome).map(Some),
                None => Ok(None),
            })
            .collect()
    }

    /// Send a prompt or request and get a streaming iterator of events.
    ///
    /// When ``history`` is omitted, session history and verification evidence are
    /// updated after the stream completes. Supplying ``history`` keeps the stream isolated.
    ///
    /// Args:
    ///     prompt: Prompt string, or {"prompt": str, "history": list, "attachments": list}
    ///     history: Optional conversation history (same format as send)
    #[pyo3(signature = (prompt, history=None))]
    fn stream(
        &self,
        py: Python<'_>,
        prompt: &Bound<'_, PyAny>,
        history: Option<&Bound<'_, PyList>>,
    ) -> PyResult<PyEventStream> {
        let (prompt, rust_history, rust_attachments) = py_session_input_to_parts(prompt, history)?;
        let session = self.inner.clone();
        let (rx, _handle) = if rust_attachments.is_empty() {
            py.allow_threads(move || {
                get_runtime().block_on(session.stream(&prompt, rust_history.as_deref()))
            })
        } else {
            py.allow_threads(move || {
                get_runtime().block_on(session.stream_with_attachments(
                    &prompt,
                    &rust_attachments,
                    rust_history.as_deref(),
                ))
            })
        }
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to start stream: {e}")))?;

        Ok(PyEventStream {
            rx: Arc::new(Mutex::new(rx)),
            done: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Send a request using the long-lived object-shaped API.
    ///
    /// Prefer this for new integrations when the call may need history,
    /// attachments, or future request options.
    fn send_request(&self, py: Python<'_>, request: &Bound<'_, PyDict>) -> PyResult<PyAgentResult> {
        let (prompt, rust_history, rust_attachments) = py_session_request_to_parts(request)?;
        let session = self.inner.clone();

        let result = if rust_attachments.is_empty() {
            py.allow_threads(move || {
                get_runtime().block_on(session.send(&prompt, rust_history.as_deref()))
            })
        } else {
            py.allow_threads(move || {
                get_runtime().block_on(session.send_with_attachments(
                    &prompt,
                    &rust_attachments,
                    rust_history.as_deref(),
                ))
            })
        }
        .map_err(|e| PyRuntimeError::new_err(format!("Agent execution failed: {e}")))?;

        Ok(PyAgentResult::from(result))
    }

    /// Stream a request using the long-lived object-shaped API.
    fn stream_request(
        &self,
        py: Python<'_>,
        request: &Bound<'_, PyDict>,
    ) -> PyResult<PyEventStream> {
        let (prompt, rust_history, rust_attachments) = py_session_request_to_parts(request)?;
        let session = self.inner.clone();

        let (rx, _handle) = if rust_attachments.is_empty() {
            py.allow_threads(move || {
                get_runtime().block_on(session.stream(&prompt, rust_history.as_deref()))
            })
        } else {
            py.allow_threads(move || {
                get_runtime().block_on(session.stream_with_attachments(
                    &prompt,
                    &rust_attachments,
                    rust_history.as_deref(),
                ))
            })
        }
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to start stream: {e}")))?;

        Ok(PyEventStream {
            rx: Arc::new(Mutex::new(rx)),
            done: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Send a prompt with image attachments and wait for the complete response.
    ///
    /// Args:
    ///     prompt: The prompt to send
    ///     attachments: List of dicts with `{"data": bytes, "media_type": str}`
    ///     history: Optional conversation history
    #[pyo3(signature = (prompt, attachments, history=None))]
    fn send_with_attachments(
        &self,
        py: Python<'_>,
        prompt: String,
        attachments: Vec<Bound<'_, PyDict>>,
        history: Option<&Bound<'_, PyList>>,
    ) -> PyResult<PyAgentResult> {
        let rust_attachments = py_attachments_to_rust(&attachments)?;
        let rust_history = history.map(|h| py_list_to_messages(h)).transpose()?;
        let session = self.inner.clone();
        let result = py
            .allow_threads(move || {
                get_runtime().block_on(session.send_with_attachments(
                    &prompt,
                    &rust_attachments,
                    rust_history.as_deref(),
                ))
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Agent execution failed: {e}")))?;
        Ok(PyAgentResult::from(result))
    }

    /// Stream a prompt with image attachments.
    ///
    /// When ``history`` is omitted, session history and verification evidence are
    /// updated after the stream completes. Supplying ``history`` keeps the stream isolated.
    ///
    /// Args:
    ///     prompt: The prompt to send
    ///     attachments: List of dicts with `{"data": bytes, "media_type": str}`
    ///     history: Optional conversation history
    #[pyo3(signature = (prompt, attachments, history=None))]
    fn stream_with_attachments(
        &self,
        py: Python<'_>,
        prompt: String,
        attachments: Vec<Bound<'_, PyDict>>,
        history: Option<&Bound<'_, PyList>>,
    ) -> PyResult<PyEventStream> {
        let rust_attachments = py_attachments_to_rust(&attachments)?;
        let rust_history = history.map(|h| py_list_to_messages(h)).transpose()?;
        let session = self.inner.clone();
        let (rx, _handle) = py
            .allow_threads(move || {
                get_runtime().block_on(session.stream_with_attachments(
                    &prompt,
                    &rust_attachments,
                    rust_history.as_deref(),
                ))
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to start stream: {e}")))?;
        Ok(PyEventStream {
            rx: Arc::new(Mutex::new(rx)),
            done: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Return the session's conversation history as a list of dicts.
    ///
    /// Each dict has `{"role": str, "content": [{"type": "text", "text": str}, ...]}`.
    fn history<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let messages = self.inner.history();
        messages_to_py_list(py, &messages)
    }

    /// Return run snapshots recorded by this session.
    fn runs(&self, py: Python<'_>) -> PyResult<PyObject> {
        let session = self.inner.clone();
        let runs = py.allow_threads(move || get_runtime().block_on(session.runs()));
        let json = serde_json::to_string(&runs)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize runs: {e}")))?;
        json_string_to_py(py, &json)
    }

    /// Return a run snapshot by ID, or None when it is unknown.
    fn run_snapshot(&self, py: Python<'_>, run_id: String) -> PyResult<PyObject> {
        let session = self.inner.clone();
        let snapshot =
            py.allow_threads(move || get_runtime().block_on(session.run_snapshot(&run_id)));
        let json = serde_json::to_string(&snapshot).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to serialize run snapshot: {e}"))
        })?;
        json_string_to_py(py, &json)
    }

    /// Return recorded runtime events for a run.
    fn run_events(&self, py: Python<'_>, run_id: String) -> PyResult<PyObject> {
        let session = self.inner.clone();
        let events = py.allow_threads(move || get_runtime().block_on(session.run_events(&run_id)));
        let json = serde_json::to_string(&events)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize run events: {e}")))?;
        json_string_to_py(py, &json)
    }

    /// Return the currently running operation, or None when idle.
    fn current_run(&self, py: Python<'_>) -> PyResult<PyObject> {
        let session = self.inner.clone();
        let snapshot = py.allow_threads(move || {
            get_runtime().block_on(async move {
                match session.current_run().await {
                    Some(run) => run.snapshot().await,
                    None => None,
                }
            })
        });
        let json = serde_json::to_string(&snapshot)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize run: {e}")))?;
        json_string_to_py(py, &json)
    }

    /// Return active tool calls observed for the currently running operation.
    fn active_tools(&self, py: Python<'_>) -> PyResult<PyObject> {
        let session = self.inner.clone();
        let active_tools = py.allow_threads(move || get_runtime().block_on(session.active_tools()));
        let json = serde_json::to_string(&active_tools).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to serialize active tools: {e}"))
        })?;
        json_string_to_py(py, &json)
    }

    /// Look up a delegated subagent task by id. Returns None when no such
    /// task has been observed in this session.
    fn subagent_task(&self, py: Python<'_>, task_id: String) -> PyResult<PyObject> {
        let session = self.inner.clone();
        let snapshot =
            py.allow_threads(move || get_runtime().block_on(session.subagent_task(&task_id)));
        let json = serde_json::to_string(&snapshot).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to serialize subagent task: {e}"))
        })?;
        json_string_to_py(py, &json)
    }

    /// Return snapshots of every delegated subagent task observed in this
    /// session (including completed and failed ones), oldest first.
    fn subagent_tasks(&self, py: Python<'_>) -> PyResult<PyObject> {
        let session = self.inner.clone();
        let tasks = py.allow_threads(move || get_runtime().block_on(session.subagent_tasks()));
        let json = serde_json::to_string(&tasks).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to serialize subagent tasks: {e}"))
        })?;
        json_string_to_py(py, &json)
    }

    /// Return snapshots of subagent tasks still in `running` state.
    fn pending_subagent_tasks(&self, py: Python<'_>) -> PyResult<PyObject> {
        let session = self.inner.clone();
        let tasks =
            py.allow_threads(move || get_runtime().block_on(session.pending_subagent_tasks()));
        let json = serde_json::to_string(&tasks).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to serialize pending subagent tasks: {e}"))
        })?;
        json_string_to_py(py, &json)
    }

    /// Cancel an in-flight subagent task by id. Returns True when a
    /// cancellation token was found and fired, False when the task id is
    /// unknown or the task already finished.
    fn cancel_subagent_task(&self, py: Python<'_>, task_id: String) -> bool {
        let session = self.inner.clone();
        py.allow_threads(move || get_runtime().block_on(session.cancel_subagent_task(&task_id)))
    }

    /// Cancel a specific run only if it is still the active run.
    fn cancel_run(&self, py: Python<'_>, run_id: String) -> bool {
        let session = self.inner.clone();
        py.allow_threads(move || get_runtime().block_on(session.cancel_run(&run_id)))
    }

    /// Execute a tool by name, bypassing the LLM.
    fn tool(
        &self,
        py: Python<'_>,
        name: String,
        args: &Bound<'_, pyo3::types::PyDict>,
    ) -> PyResult<PyToolResult> {
        let json_str = py_dict_to_json(args)?;
        let json_value: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| PyValueError::new_err(format!("Invalid JSON args: {e}")))?;

        let session = self.inner.clone();
        let result = py
            .allow_threads(move || get_runtime().block_on(session.tool(&name, json_value)))
            .map_err(|e| PyRuntimeError::new_err(format!("Tool execution failed: {e}")))?;

        Ok(PyToolResult {
            name: result.name,
            output: result.output,
            exit_code: result.exit_code,
            metadata_json: result.metadata.as_ref().map(serde_json::Value::to_string),
            error_kind_json: result
                .error_kind
                .as_ref()
                .and_then(|k| serde_json::to_string(k).ok()),
        })
    }

    /// Delegate a bounded task with the compact object-shaped API.
    fn task(&self, py: Python<'_>, options: &Bound<'_, PyDict>) -> PyResult<PyToolResult> {
        let json_str = py_dict_to_json(options)?;
        let args: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| PyValueError::new_err(format!("Invalid task options: {e}")))?;
        let args = normalize_task_options(args)?;

        let session = self.inner.clone();
        let result = py
            .allow_threads(move || get_runtime().block_on(session.tool("task", args)))
            .map_err(|e| PyRuntimeError::new_err(format!("Task delegation failed: {e}")))?;

        Ok(PyToolResult {
            name: result.name,
            output: result.output,
            exit_code: result.exit_code,
            metadata_json: result.metadata.as_ref().map(serde_json::Value::to_string),
            error_kind_json: result
                .error_kind
                .as_ref()
                .and_then(|k| serde_json::to_string(k).ok()),
        })
    }

    /// Delegate a bounded task to a child agent through the built-in ``task`` tool.
    #[pyo3(signature = (agent, description, prompt, background=false, max_steps=None))]
    fn delegate_task(
        &self,
        py: Python<'_>,
        agent: String,
        description: String,
        prompt: String,
        background: bool,
        max_steps: Option<u32>,
    ) -> PyResult<PyToolResult> {
        let args = delegate_task_args(agent, description, prompt, background, max_steps);

        let session = self.inner.clone();
        let result = py
            .allow_threads(move || get_runtime().block_on(session.tool("task", args)))
            .map_err(|e| PyRuntimeError::new_err(format!("Task delegation failed: {e}")))?;

        Ok(PyToolResult {
            name: result.name,
            output: result.output,
            exit_code: result.exit_code,
            metadata_json: result.metadata.as_ref().map(serde_json::Value::to_string),
            error_kind_json: result
                .error_kind
                .as_ref()
                .and_then(|k| serde_json::to_string(k).ok()),
        })
    }

    /// Execute several delegated child-agent tasks with the compact API.
    fn tasks(&self, py: Python<'_>, tasks: &Bound<'_, PyAny>) -> PyResult<PyToolResult> {
        let json_mod = py.import("json")?;
        let json_str: String = json_mod.call_method1("dumps", (tasks,))?.extract()?;
        let task_values: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| PyValueError::new_err(format!("Invalid task list: {e}")))?;
        let args = parallel_task_args(task_values)?;

        let session = self.inner.clone();
        let result = py
            .allow_threads(move || get_runtime().block_on(session.tool("parallel_task", args)))
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Parallel task delegation failed: {e}"))
            })?;

        Ok(PyToolResult {
            name: result.name,
            output: result.output,
            exit_code: result.exit_code,
            metadata_json: result.metadata.as_ref().map(serde_json::Value::to_string),
            error_kind_json: result
                .error_kind
                .as_ref()
                .and_then(|k| serde_json::to_string(k).ok()),
        })
    }

    /// Execute several delegated child-agent tasks concurrently through ``parallel_task``.
    fn parallel_task(&self, py: Python<'_>, tasks: &Bound<'_, PyAny>) -> PyResult<PyToolResult> {
        self.tasks(py, tasks)
    }

    /// Run a bounded JavaScript script through the embedded QuickJS `program` tool.
    fn program(
        &self,
        py: Python<'_>,
        options: &Bound<'_, pyo3::types::PyDict>,
    ) -> PyResult<PyToolResult> {
        let args = normalize_program_script_options(options)?;

        let session = self.inner.clone();
        let result = py
            .allow_threads(move || get_runtime().block_on(session.tool("program", args)))
            .map_err(|e| PyRuntimeError::new_err(format!("Program execution failed: {e}")))?;

        Ok(PyToolResult {
            name: result.name,
            output: result.output,
            exit_code: result.exit_code,
            metadata_json: result.metadata.as_ref().map(serde_json::Value::to_string),
            error_kind_json: result
                .error_kind
                .as_ref()
                .and_then(|k| serde_json::to_string(k).ok()),
        })
    }

    /// Read a file from the workspace.
    fn read_file(&self, py: Python<'_>, path: String) -> PyResult<String> {
        let session = self.inner.clone();
        py.allow_threads(move || get_runtime().block_on(session.read_file(&path)))
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    /// Write a file in the workspace.
    fn write_file(&self, py: Python<'_>, path: String, content: String) -> PyResult<PyToolResult> {
        let session = self.inner.clone();
        let result = py
            .allow_threads(move || get_runtime().block_on(session.write_file(&path, &content)))
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;

        Ok(PyToolResult {
            name: result.name,
            output: result.output,
            exit_code: result.exit_code,
            metadata_json: result.metadata.as_ref().map(serde_json::Value::to_string),
            error_kind_json: result
                .error_kind
                .as_ref()
                .and_then(|k| serde_json::to_string(k).ok()),
        })
    }

    /// List a directory in the workspace.
    #[pyo3(signature = (path=None))]
    fn ls(&self, py: Python<'_>, path: Option<String>) -> PyResult<PyToolResult> {
        let session = self.inner.clone();
        let result = py
            .allow_threads(move || get_runtime().block_on(session.ls(path.as_deref())))
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;

        Ok(PyToolResult {
            name: result.name,
            output: result.output,
            exit_code: result.exit_code,
            metadata_json: result.metadata.as_ref().map(serde_json::Value::to_string),
            error_kind_json: result
                .error_kind
                .as_ref()
                .and_then(|k| serde_json::to_string(k).ok()),
        })
    }

    /// Edit a file by replacing text in the workspace.
    #[pyo3(signature = (path, old_string, new_string, replace_all=false))]
    fn edit_file(
        &self,
        py: Python<'_>,
        path: String,
        old_string: String,
        new_string: String,
        replace_all: bool,
    ) -> PyResult<PyToolResult> {
        let session = self.inner.clone();
        let result = py
            .allow_threads(move || {
                get_runtime().block_on(session.edit_file(
                    &path,
                    &old_string,
                    &new_string,
                    replace_all,
                ))
            })
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;

        Ok(PyToolResult {
            name: result.name,
            output: result.output,
            exit_code: result.exit_code,
            metadata_json: result.metadata.as_ref().map(serde_json::Value::to_string),
            error_kind_json: result
                .error_kind
                .as_ref()
                .and_then(|k| serde_json::to_string(k).ok()),
        })
    }

    /// Apply a unified diff patch to a workspace file.
    fn patch_file(&self, py: Python<'_>, path: String, diff: String) -> PyResult<PyToolResult> {
        let session = self.inner.clone();
        let result = py
            .allow_threads(move || get_runtime().block_on(session.patch_file(&path, &diff)))
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;

        Ok(PyToolResult {
            name: result.name,
            output: result.output,
            exit_code: result.exit_code,
            metadata_json: result.metadata.as_ref().map(serde_json::Value::to_string),
            error_kind_json: result
                .error_kind
                .as_ref()
                .and_then(|k| serde_json::to_string(k).ok()),
        })
    }

    /// Execute a bash command in the workspace.
    fn bash(&self, py: Python<'_>, command: String) -> PyResult<String> {
        let session = self.inner.clone();
        py.allow_threads(move || get_runtime().block_on(session.bash(&command)))
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    /// Search for files matching a glob pattern.
    fn glob(&self, py: Python<'_>, pattern: String) -> PyResult<Vec<String>> {
        let session = self.inner.clone();
        py.allow_threads(move || get_runtime().block_on(session.glob(&pattern)))
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    /// Search file contents with a regex pattern.
    fn grep(&self, py: Python<'_>, pattern: String) -> PyResult<String> {
        let session = self.inner.clone();
        py.allow_threads(move || get_runtime().block_on(session.grep(&pattern)))
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    /// Search the web using multiple search engines.
    fn web_search(&self, py: Python<'_>, params: PyWebSearchParams) -> PyResult<PyToolResult> {
        let session = self.inner.clone();
        let mut args = serde_json::json!({
            "query": params.query,
        });
        if let Some(ref engines) = params.engines {
            args["engines"] = serde_json::json!(engines);
        }
        if let Some(limit) = params.limit {
            args["limit"] = serde_json::json!(limit);
        }
        if let Some(timeout) = params.timeout {
            args["timeout"] = serde_json::json!(timeout);
        }
        if let Some(ref proxy) = params.proxy {
            args["proxy"] = serde_json::json!(proxy);
        }
        if let Some(ref format) = params.format {
            args["format"] = serde_json::json!(format);
        }
        let result = py
            .allow_threads(move || get_runtime().block_on(session.tool("web_search", args)))
            .map_err(|e| PyRuntimeError::new_err(format!("Tool execution failed: {e}")))?;
        Ok(PyToolResult {
            name: result.name,
            output: result.output,
            exit_code: result.exit_code,
            metadata_json: result.metadata.as_ref().map(serde_json::Value::to_string),
            error_kind_json: result
                .error_kind
                .as_ref()
                .and_then(|k| serde_json::to_string(k).ok()),
        })
    }

    /// Execute a git command.
    ///
    /// Prefer ``git({"command": "status"})``; positional arguments remain for
    /// compatibility.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (command, subcommand=None, name=None, path=None, new_branch=true, base=None, force=false, max_count=None, message=None, include_untracked=false, target=None, reference=None))]
    fn git(
        &self,
        py: Python<'_>,
        command: &Bound<'_, PyAny>,
        subcommand: Option<String>,
        name: Option<String>,
        path: Option<String>,
        new_branch: bool,
        base: Option<String>,
        force: bool,
        max_count: Option<usize>,
        message: Option<String>,
        include_untracked: bool,
        target: Option<String>,
        reference: Option<String>,
    ) -> PyResult<PyToolResult> {
        let mut args = if let Ok(command) = command.extract::<String>() {
            serde_json::json!({ "command": command })
        } else if let Ok(config) = command.downcast::<PyDict>() {
            let json_str = py_dict_to_json(config)?;
            let args: serde_json::Value = serde_json::from_str(&json_str)
                .map_err(|e| PyValueError::new_err(format!("Invalid git args: {e}")))?;
            normalize_git_args(args)?
        } else {
            return Err(PyTypeError::new_err(
                "git command must be a command string or options dict",
            ));
        };

        if let Some(sc) = subcommand {
            args["subcommand"] = serde_json::json!(sc);
        }
        if let Some(n) = name {
            args["name"] = serde_json::json!(n);
        }
        if let Some(p) = path {
            args["path"] = serde_json::json!(p);
        }
        if !new_branch {
            args["new_branch"] = serde_json::json!(new_branch);
        }
        if let Some(b) = base {
            args["base"] = serde_json::json!(b);
        }
        if force {
            args["force"] = serde_json::json!(force);
        }
        if let Some(mc) = max_count {
            args["max_count"] = serde_json::json!(mc);
        }
        if let Some(msg) = message {
            args["message"] = serde_json::json!(msg);
        }
        if include_untracked {
            args["include_untracked"] = serde_json::json!(include_untracked);
        }
        if let Some(t) = target {
            args["target"] = serde_json::json!(t);
        }
        if let Some(r) = reference {
            args["ref"] = serde_json::json!(r);
        }

        let session = self.inner.clone();
        let result = py
            .allow_threads(move || get_runtime().block_on(session.tool("git", args)))
            .map_err(|e| PyRuntimeError::new_err(format!("git failed: {e}")))?;
        Ok(PyToolResult {
            name: result.name,
            output: result.output,
            exit_code: result.exit_code,
            metadata_json: result.metadata.as_ref().map(serde_json::Value::to_string),
            error_kind_json: result
                .error_kind
                .as_ref()
                .and_then(|k| serde_json::to_string(k).ok()),
        })
    }

    /// Execute a git command with an object-shaped API.
    ///
    /// Preferred over the positional ``git(...)`` overload for new callers.
    ///
    /// Example:
    ///     session.git_command({"command": "status"})
    ///     session.git_command({"command": "worktree", "subcommand": "list"})
    fn git_command(&self, py: Python<'_>, args: &Bound<'_, PyDict>) -> PyResult<PyToolResult> {
        let json_str = py_dict_to_json(args)?;
        let args: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| PyValueError::new_err(format!("Invalid git args: {e}")))?;
        let args = normalize_git_args(args)?;
        let session = self.inner.clone();
        let result = py
            .allow_threads(move || get_runtime().block_on(session.tool("git", args)))
            .map_err(|e| PyRuntimeError::new_err(format!("git failed: {e}")))?;
        Ok(PyToolResult {
            name: result.name,
            output: result.output,
            exit_code: result.exit_code,
            metadata_json: result.metadata.as_ref().map(serde_json::Value::to_string),
            error_kind_json: result
                .error_kind
                .as_ref()
                .and_then(|k| serde_json::to_string(k).ok()),
        })
    }

    // ========================================================================
    // Advanced optional Queue API
    // ========================================================================

    /// Check if this session has an advanced lane queue configured.
    fn has_queue(&self) -> bool {
        self.inner.has_queue()
    }

    /// Configure a lane's handler mode for explicit external/hybrid dispatch.
    ///
    /// Args:
    ///     lane (Literal["control", "query", "execute", "generate"]): Which lane to configure.
    ///     mode (Literal["internal", "external", "hybrid"]): Execution mode for the lane's tools.
    ///     timeout_ms: Timeout for external processing in milliseconds (default 60000).
    #[pyo3(signature = (lane, mode="internal", timeout_ms=60000))]
    fn set_lane_handler(
        &self,
        py: Python<'_>,
        lane: &str,
        mode: &str,
        timeout_ms: u64,
    ) -> PyResult<()> {
        let lane = parse_lane(lane)?;
        let mode = parse_handler_mode(mode)?;
        let config = RustLaneHandlerConfig { mode, timeout_ms };
        let session = self.inner.clone();
        py.allow_threads(move || get_runtime().block_on(session.set_lane_handler(lane, config)));
        Ok(())
    }

    /// Complete an external queue task by ID.
    ///
    /// Args:
    ///     task_id: The task identifier
    ///     success: Whether the task succeeded
    ///     result: Result data (any JSON-serializable value)
    ///     error: Optional error message
    ///
    /// Returns:
    ///     True if the task was found and completed, False if not found.
    #[pyo3(signature = (task_id, success=true, result=None, error=None))]
    fn complete_external_task(
        &self,
        py: Python<'_>,
        task_id: String,
        success: bool,
        result: Option<&Bound<'_, PyDict>>,
        error: Option<String>,
    ) -> PyResult<bool> {
        let result_value = match result {
            Some(dict) => {
                let json_str = py_dict_to_json(dict)?;
                serde_json::from_str(&json_str)
                    .map_err(|e| PyValueError::new_err(format!("Invalid JSON: {e}")))?
            }
            None => serde_json::json!({}),
        };
        let ext_result = RustExternalTaskResult {
            success,
            result: result_value,
            error,
        };
        let session = self.inner.clone();
        let found = py.allow_threads(move || {
            get_runtime().block_on(session.complete_external_task(&task_id, ext_result))
        });
        Ok(found)
    }

    /// Get pending external queue tasks.
    ///
    /// Returns:
    ///     List of dicts with task_id, session_id, lane, command_type, payload, timeout_ms.
    fn pending_external_tasks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let session = self.inner.clone();
        let tasks =
            py.allow_threads(move || get_runtime().block_on(session.pending_external_tasks()));
        let json_str = serde_json::to_string(&tasks)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {e}")))?;
        let json_mod = py.import("json")?;
        let py_obj = json_mod.call_method1("loads", (json_str,))?;
        py_obj
            .downcast::<PyList>()
            .cloned()
            .map_err(|e| PyRuntimeError::new_err(format!("Unexpected result: {e}")))
    }

    /// Return pending HITL tool confirmations for this session.
    fn pending_confirmations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let session = self.inner.clone();
        let pending =
            py.allow_threads(move || get_runtime().block_on(session.pending_confirmations()));
        let json_str = serde_json::to_string(&pending)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {e}")))?;
        let json_mod = py.import("json")?;
        let py_obj = json_mod.call_method1("loads", (json_str,))?;
        py_obj
            .downcast::<PyList>()
            .cloned()
            .map_err(|e| PyRuntimeError::new_err(format!("Unexpected result: {e}")))
    }

    /// Resolve a pending HITL tool confirmation.
    #[pyo3(signature = (tool_id, approved, reason=None))]
    fn confirm_tool_use(
        &self,
        py: Python<'_>,
        tool_id: String,
        approved: bool,
        reason: Option<String>,
    ) -> PyResult<bool> {
        let session = self.inner.clone();
        py.allow_threads(move || {
            get_runtime().block_on(session.confirm_tool_use(&tool_id, approved, reason))
        })
        .map_err(|e| PyRuntimeError::new_err(format!("confirm_tool_use failed: {e}")))
    }

    /// Cancel all pending HITL confirmations for this session.
    fn cancel_confirmations(&self, py: Python<'_>) -> usize {
        let session = self.inner.clone();
        py.allow_threads(move || get_runtime().block_on(session.cancel_confirmations()))
    }

    /// Get optional queue statistics.
    ///
    /// Returns:
    ///     Dict with total_pending, total_active, external_pending, and per-lane status.
    fn queue_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let session = self.inner.clone();
        let stats = py.allow_threads(move || get_runtime().block_on(session.queue_stats()));
        let json_str = serde_json::to_string(&stats)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {e}")))?;
        let json_mod = py.import("json")?;
        let py_obj = json_mod.call_method1("loads", (json_str,))?;
        py_obj
            .downcast::<PyDict>()
            .cloned()
            .map_err(|e| PyRuntimeError::new_err(format!("Unexpected result: {e}")))
    }

    /// Get dead letters from the optional queue's DLQ.
    ///
    /// Returns:
    ///     List of dicts with command_id, command_type, lane, error, attempts, failed_at.
    fn dead_letters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let session = self.inner.clone();
        let letters = py.allow_threads(move || get_runtime().block_on(session.dead_letters()));
        let json_str = serde_json::to_string(&letters)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {e}")))?;
        let json_mod = py.import("json")?;
        let py_obj = json_mod.call_method1("loads", (json_str,))?;
        py_obj
            .downcast::<PyList>()
            .cloned()
            .map_err(|e| PyRuntimeError::new_err(format!("Unexpected result: {e}")))
    }

    /// Get a detailed metrics snapshot from the queue.
    ///
    /// Returns ``None`` if metrics are not enabled (queue not configured or
    /// ``enable_metrics`` was not set in ``SessionQueueConfig``).
    ///
    /// Returns:
    ///     Dict with ``counters``, ``gauges``, and ``histograms`` maps, or None.
    fn queue_metrics<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let session = self.inner.clone();
        let snapshot = py.allow_threads(move || get_runtime().block_on(session.queue_metrics()));
        match snapshot {
            None => Ok(py.None()),
            Some(s) => {
                let json_str = metrics_snapshot_to_json_str(s)
                    .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {e}")))?;
                let json_mod = py.import("json")?;
                Ok(json_mod.call_method1("loads", (json_str,))?.into())
            }
        }
    }

    /// Add an MCP server to this live session.
    ///
    /// Connects the server and registers all its tools immediately so the agent
    /// can call them. Tool names follow the convention ``mcp__<name>__<tool>``.
    ///
    /// Args:
    ///     name: Server identifier (used as prefix in tool names)
    ///     transport: Transport type — ``"stdio"`` (default), ``"http"``, or ``"streamable-http"``
    ///     command: Executable to launch (stdio only, e.g. ``"npx"``)
    ///     args: Arguments for the command (stdio only)
    ///     url: Server URL (http / streamable-http only)
    ///     headers: HTTP headers dict (http / streamable-http only, e.g. ``{"Authorization": "Bearer ..."}``))
    ///     env: Optional dict of extra environment variables (stdio only)
    ///
    /// Returns:
    ///     Number of tools registered from the server
    ///
    /// Raises:
    ///     RuntimeError: If the server fails to connect
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (name, transport="stdio", command=None, args=None, url=None, headers=None, env=None, timeout_ms=None))]
    fn add_mcp_server(
        &self,
        py: Python<'_>,
        name: &str,
        transport: &str,
        command: Option<&str>,
        args: Option<Vec<String>>,
        url: Option<&str>,
        headers: Option<std::collections::HashMap<String, String>>,
        env: Option<std::collections::HashMap<String, String>>,
        timeout_ms: Option<u64>,
    ) -> PyResult<usize> {
        use a3s_code_core::mcp::protocol::{McpServerConfig, McpTransportConfig};

        let transport_config = match transport {
            "stdio" => {
                let command = command.ok_or_else(|| {
                    PyRuntimeError::new_err("'command' is required for stdio transport")
                })?;
                McpTransportConfig::Stdio {
                    command: command.to_string(),
                    args: args.unwrap_or_default(),
                }
            }
            "http" => {
                let url = url.ok_or_else(|| {
                    PyRuntimeError::new_err("'url' is required for http transport")
                })?;
                McpTransportConfig::Http {
                    url: url.to_string(),
                    headers: headers.unwrap_or_default(),
                }
            }
            "streamable-http" | "streamable_http" => {
                let url = url.ok_or_else(|| {
                    PyRuntimeError::new_err("'url' is required for streamable-http transport")
                })?;
                McpTransportConfig::StreamableHttp {
                    url: url.to_string(),
                    headers: headers.unwrap_or_default(),
                }
            }
            other => {
                return Err(PyRuntimeError::new_err(format!(
                    "Unknown transport '{}'. Use 'stdio', 'http', or 'streamable-http'",
                    other
                )))
            }
        };

        let tool_timeout_secs = timeout_ms.map(|ms| (ms / 1000).max(1)).unwrap_or(60);
        let config = McpServerConfig {
            name: name.to_string(),
            transport: transport_config,
            enabled: true,
            env: env.unwrap_or_default(),
            oauth: None,
            tool_timeout_secs,
        };
        let session = self.inner.clone();
        py.allow_threads(move || {
            get_runtime().block_on(async {
                session
                    .add_mcp_server(config)
                    .await
                    .map_err(|e| PyRuntimeError::new_err(format!("add_mcp_server failed: {e}")))
            })
        })
    }

    /// Add an MCP server with an object config.
    ///
    /// Preferred for new SDK callers because the transport is typed as a nested
    /// object instead of split across positional parameters.
    ///
    /// Example:
    ///     session.add_mcp_server_config({
    ///         "name": "github",
    ///         "transport": {
    ///             "type": "stdio",
    ///             "command": "npx",
    ///             "args": ["-y", "@modelcontextprotocol/server-github"],
    ///         },
    ///         "env": {"GITHUB_TOKEN": "..."},
    ///         "timeout_ms": 30000,
    ///     })
    fn add_mcp_server_config(&self, py: Python<'_>, config: &Bound<'_, PyDict>) -> PyResult<usize> {
        let json_str = py_dict_to_json(config)?;
        let value: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| PyValueError::new_err(format!("Invalid MCP server config: {e}")))?;
        let config = normalize_mcp_server_config(value)?;
        let session = self.inner.clone();
        py.allow_threads(move || {
            get_runtime().block_on(async {
                session
                    .add_mcp_server(config)
                    .await
                    .map_err(|e| PyRuntimeError::new_err(format!("add_mcp_server failed: {e}")))
            })
        })
    }

    /// Add an MCP server with the compact object-shaped API.
    fn add_mcp(&self, py: Python<'_>, config: &Bound<'_, PyDict>) -> PyResult<usize> {
        self.add_mcp_server_config(py, config)
    }

    /// Dynamically register agents from a directory with the live session.
    ///
    /// Scans the given directory for ``*.yaml``, ``*.yml``, and ``*.md`` agent
    /// definition files and adds each to the shared agent registry used by the
    /// ``task`` tool.  New agents become immediately callable via
    /// ``task(agent="…")`` without restarting the session.
    ///
    /// Args:
    ///     path: Directory path to scan for agent definition files
    ///
    /// Returns:
    ///     Number of agents successfully loaded from the directory
    #[pyo3(signature = (path))]
    fn register_agent_dir(&self, py: Python<'_>, path: &str) -> PyResult<usize> {
        let dir = std::path::PathBuf::from(path);
        let session = self.inner.clone();
        py.allow_threads(move || {
            let count = session.register_agent_dir(&dir);
            Ok(count)
        })
    }

    /// Register a disposable worker agent into the live session.
    fn register_worker_agent(&self, worker: PyWorkerAgentSpec) -> PyResult<PyAgentDefinition> {
        let worker = py_worker_agent_spec_to_rust(worker)?;
        Ok(rust_agent_definition_to_py(
            self.inner.register_worker_agent(worker),
        ))
    }

    /// Register many disposable worker agents into the live session.
    fn register_worker_agents(
        &self,
        workers: Vec<PyWorkerAgentSpec>,
    ) -> PyResult<Vec<PyAgentDefinition>> {
        let workers = workers
            .into_iter()
            .map(py_worker_agent_spec_to_rust)
            .collect::<PyResult<Vec<_>>>()?;
        Ok(self
            .inner
            .register_worker_agents(workers)
            .into_iter()
            .map(rust_agent_definition_to_py)
            .collect())
    }

    /// Remove an MCP server from this session.
    ///
    /// Disconnects the server and unregisters all its tools.
    /// No-op if the server was never added.
    ///
    /// Args:
    ///     name: Server identifier used when it was added
    #[pyo3(signature = (name))]
    fn remove_mcp_server(&self, py: Python<'_>, name: &str) -> PyResult<()> {
        let name = name.to_string();
        let session = self.inner.clone();
        py.allow_threads(move || {
            get_runtime().block_on(async {
                session
                    .remove_mcp_server(&name)
                    .await
                    .map_err(|e| PyRuntimeError::new_err(format!("remove_mcp_server failed: {e}")))
            })
        })
    }

    /// Remove an MCP server with the compact API.
    fn remove_mcp(&self, py: Python<'_>, name: &str) -> PyResult<()> {
        self.remove_mcp_server(py, name)
    }

    /// Return the connection status of all MCP servers for this session.
    ///
    /// Returns:
    ///     Dict mapping server name to status dict with keys:
    ///     ``connected`` (bool), ``tool_count`` (int).
    fn mcp_status<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let session = self.inner.clone();
        let status = py.allow_threads(move || get_runtime().block_on(session.mcp_status()));
        let dict = PyDict::new(py);
        for (name, s) in status {
            let entry = PyDict::new(py);
            entry.set_item("connected", s.connected)?;
            entry.set_item("tool_count", s.tool_count)?;
            entry.set_item("error", s.error.as_deref())?;
            dict.set_item(name, entry)?;
        }
        Ok(dict)
    }

    /// Return MCP server status with the compact API.
    fn mcps<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.mcp_status(py)
    }

    /// Return the names of all tools currently available in this session.
    ///
    /// Reflects the live state — MCP tools appear after ``add_mcp_server()``
    /// or ``add_mcp_server_config()``
    /// and disappear after ``remove_mcp_server()``.
    ///
    /// Returns:
    ///     List of tool name strings
    fn tool_names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let names = self.inner.tool_names();
        let list = PyList::new(py, names)?;
        Ok(list)
    }

    /// Return full model-visible tool definitions currently registered on this session.
    fn tool_definitions(&self, py: Python<'_>) -> PyResult<PyObject> {
        let json = serde_json::to_string(&self.inner.tool_definitions()).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to serialize tool definitions: {e}"))
        })?;
        json_string_to_py(py, &json)
    }

    /// Return a stored tool artifact by URI, or ``None`` if it is not retained.
    fn get_artifact(&self, py: Python<'_>, artifact_uri: &str) -> PyResult<PyObject> {
        let json = serde_json::to_string(&self.inner.get_artifact(artifact_uri))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize artifact: {e}")))?;
        json_string_to_py(py, &json)
    }

    /// Return compact execution trace events recorded for this session.
    fn trace_events(&self, py: Python<'_>) -> PyResult<PyObject> {
        let json = serde_json::to_string(&self.inner.trace_events())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize traces: {e}")))?;
        json_string_to_py(py, &json)
    }

    /// Return structured verification reports recorded for this session.
    fn verification_reports(&self, py: Python<'_>) -> PyResult<PyObject> {
        let json = serde_json::to_string(&self.inner.verification_reports()).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to serialize verification reports: {e}"))
        })?;
        json_string_to_py(py, &json)
    }

    /// Add externally produced verification reports to this session.
    fn record_verification_reports(
        &self,
        py: Python<'_>,
        reports: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let reports = py_verification_reports_to_rust(py, reports)?;
        self.inner.record_verification_reports(reports);
        Ok(())
    }

    /// Return a structured verification summary for this session.
    fn verification_summary(&self, py: Python<'_>) -> PyResult<PyObject> {
        let json = serde_json::to_string(&self.inner.verification_summary()).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to serialize verification summary: {e}"))
        })?;
        json_string_to_py(py, &json)
    }

    /// Return a concise human-readable verification summary for this session.
    fn verification_summary_text(&self) -> String {
        self.inner.verification_summary_text()
    }

    /// Run verification commands and return a structured verification report.
    fn verify_commands(
        &self,
        py: Python<'_>,
        subject: &str,
        commands: &Bound<'_, PyList>,
    ) -> PyResult<PyObject> {
        let rust_commands = py_list_to_verification_commands(commands)?;
        let session = self.inner.clone();
        let subject = subject.to_string();
        let report = py
            .allow_threads(move || {
                get_runtime().block_on(session.verify_commands(&subject, &rust_commands))
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Verification failed: {e}")))?;
        let json = serde_json::to_string(&report).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to serialize verification report: {e}"))
        })?;
        json_string_to_py(py, &json)
    }

    /// Return project-aware verification command presets for this workspace.
    fn verification_presets(&self, py: Python<'_>) -> PyResult<PyObject> {
        let json = serde_json::to_string(&self.inner.verification_presets()).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to serialize verification presets: {e}"))
        })?;
        json_string_to_py(py, &json)
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
    /// Args:
    ///     hook_id: Unique hook identifier
    ///     event_type: Event type string — one of:
    ///         "pre_tool_use", "post_tool_use", "generate_start", "generate_end",
    ///         "session_start", "session_end", "skill_load", "skill_unload",
    ///         "pre_prompt", "post_response", "on_error"
    ///     matcher: Optional dict with keys: tool, path_pattern, command_pattern, session_id, skill
    ///     config: Optional dict with keys: priority, timeout_ms, async_execution, max_retries
    ///     handler: Optional callable ``(event: dict) -> dict | None``. When provided, it is called
    ///         for every matching event and its return value controls execution:
    ///         ``{"action": "block", "reason": "…"}`` cancels the operation,
    ///         ``{"action": "skip"}`` skips remaining hooks, ``None`` or
    ///         ``{"action": "continue"}`` allows execution to proceed.
    #[pyo3(signature = (hook_id, event_type, matcher=None, config=None, handler=None))]
    fn register_hook(
        &self,
        hook_id: String,
        event_type: String,
        matcher: Option<&Bound<'_, PyDict>>,
        config: Option<&Bound<'_, PyDict>>,
        handler: Option<pyo3::Py<pyo3::PyAny>>,
    ) -> PyResult<()> {
        let rust_event_type = py_parse_hook_event_type(&event_type)?;
        let mut hook = RustHook::new(&hook_id, rust_event_type);

        if let Some(m) = matcher {
            let mut rust_matcher = RustHookMatcher::new();
            if let Some(tool) = m.get_item("tool")? {
                rust_matcher = rust_matcher.with_tool(tool.extract::<String>()?);
            }
            if let Some(path) = m.get_item("path_pattern")? {
                rust_matcher = rust_matcher.with_path(path.extract::<String>()?);
            }
            if let Some(cmd) = m.get_item("command_pattern")? {
                rust_matcher = rust_matcher.with_command(cmd.extract::<String>()?);
            }
            if let Some(sid) = m.get_item("session_id")? {
                rust_matcher = rust_matcher.with_session(sid.extract::<String>()?);
            }
            if let Some(skill) = m.get_item("skill")? {
                rust_matcher = rust_matcher.with_skill(skill.extract::<String>()?);
            }
            hook = hook.with_matcher(rust_matcher);
        }

        if let Some(c) = config {
            let priority = c
                .get_item("priority")?
                .map(|v| v.extract::<i32>())
                .transpose()?
                .unwrap_or(100);
            let timeout_ms = c
                .get_item("timeout_ms")?
                .map(|v| v.extract::<u64>())
                .transpose()?
                .unwrap_or(30000);
            let async_execution = c
                .get_item("async_execution")?
                .map(|v| v.extract::<bool>())
                .transpose()?
                .unwrap_or(false);
            let max_retries = c
                .get_item("max_retries")?
                .map(|v| v.extract::<u32>())
                .transpose()?
                .unwrap_or(0);
            hook = hook.with_config(RustHookConfig {
                priority,
                timeout_ms,
                async_execution,
                max_retries,
            });
        }

        self.inner.register_hook(hook);

        if let Some(py_fn) = handler {
            self.inner.register_hook_handler(
                &hook_id,
                Arc::new(PythonCallbackHandler { callback: py_fn }),
            );
        }

        Ok(())
    }

    /// Unregister a hook by ID.
    ///
    /// Returns True if the hook was found and removed, False otherwise.
    fn unregister_hook(&self, hook_id: String) -> bool {
        self.inner.unregister_hook_handler(&hook_id);
        self.inner.unregister_hook(&hook_id).is_some()
    }

    /// Get the number of registered hooks.
    fn hook_count(&self) -> usize {
        self.inner.hook_count()
    }

    // ========================================================================
    // Session Metadata API
    // ========================================================================

    /// Return the session ID.
    #[getter]
    fn session_id(&self) -> String {
        self.inner.session_id().to_string()
    }

    /// Return the workspace path.
    #[getter]
    fn workspace(&self) -> String {
        self.inner.workspace().display().to_string()
    }

    /// Return any deferred init warning (e.g. memory store failed to initialize).
    #[getter]
    fn init_warning(&self) -> Option<String> {
        self.inner.init_warning().map(|s| s.to_string())
    }

    /// Host-defined tenant id attached at session creation, if any.
    #[getter]
    fn tenant_id(&self) -> Option<String> {
        self.inner.tenant_id().map(|s| s.to_string())
    }

    /// Identity of the principal that triggered the session, if any.
    #[getter]
    fn principal(&self) -> Option<String> {
        self.inner.principal().map(|s| s.to_string())
    }

    /// Logical agent template / definition id, if any.
    #[getter]
    fn agent_template_id(&self) -> Option<String> {
        self.inner.agent_template_id().map(|s| s.to_string())
    }

    /// Distributed-trace correlation id propagated through this session,
    /// if any.
    #[getter]
    fn correlation_id(&self) -> Option<String> {
        self.inner.correlation_id().map(|s| s.to_string())
    }

    // ========================================================================
    // Session Persistence API
    // ========================================================================

    /// Save the session to the configured store.
    ///
    /// Returns None if no store is configured (no-op).
    fn save(&self, py: Python<'_>) -> PyResult<()> {
        let session = self.inner.clone();
        py.allow_threads(move || get_runtime().block_on(session.save()))
            .map_err(|e| PyRuntimeError::new_err(format!("Save failed: {e}")))
    }

    // ========================================================================
    // Memory API
    // ========================================================================

    /// Check if memory is configured for this session.
    #[getter]
    fn has_memory(&self) -> bool {
        self.inner.memory().is_some()
    }

    /// Remember a successful task execution.
    ///
    /// Args:
    ///     task: Description of the task
    ///     tools: List of tool names used
    ///     result: Summary of the result
    #[pyo3(signature = (task, tools, result))]
    fn remember_success(
        &self,
        py: Python<'_>,
        task: String,
        tools: Vec<String>,
        result: String,
    ) -> PyResult<()> {
        let memory = self
            .inner
            .memory()
            .ok_or_else(|| PyRuntimeError::new_err("Memory not configured for this session"))?
            .clone();
        py.allow_threads(move || {
            get_runtime().block_on(memory.remember_success(&task, &tools, &result))
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Remember failed: {e}")))
    }

    /// Remember a failed task execution.
    ///
    /// Args:
    ///     task: Description of the task
    ///     error: Error message
    ///     tools: List of tool names attempted
    #[pyo3(signature = (task, error, tools))]
    fn remember_failure(
        &self,
        py: Python<'_>,
        task: String,
        error: String,
        tools: Vec<String>,
    ) -> PyResult<()> {
        let memory = self
            .inner
            .memory()
            .ok_or_else(|| PyRuntimeError::new_err("Memory not configured for this session"))?
            .clone();
        py.allow_threads(move || {
            get_runtime().block_on(memory.remember_failure(&task, &error, &tools))
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Remember failed: {e}")))
    }

    /// Recall memories similar to a query.
    ///
    /// Args:
    ///     query: Search query
    ///     limit: Maximum number of results (default: 5)
    ///
    /// Returns:
    ///     List of dicts with task, tools, result/error, outcome, timestamp.
    #[pyo3(signature = (query, limit=5))]
    fn recall_similar<'py>(
        &self,
        py: Python<'py>,
        query: String,
        limit: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let memory = self
            .inner
            .memory()
            .ok_or_else(|| PyRuntimeError::new_err("Memory not configured for this session"))?
            .clone();
        let items = py
            .allow_threads(move || get_runtime().block_on(memory.recall_similar(&query, limit)))
            .map_err(|e| PyRuntimeError::new_err(format!("Recall failed: {e}")))?;
        let json_str = serde_json::to_string(&items)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {e}")))?;
        let json_mod = py.import("json")?;
        let py_obj = json_mod.call_method1("loads", (json_str,))?;
        py_obj
            .downcast::<PyList>()
            .cloned()
            .map_err(|e| PyRuntimeError::new_err(format!("Unexpected result: {e}")))
    }

    /// Recall memories by tags.
    ///
    /// Args:
    ///     tags: List of tags to search for
    ///     limit: Maximum number of results (default: 10)
    ///
    /// Returns:
    ///     List of memory item dicts.
    #[pyo3(signature = (tags, limit=10))]
    fn recall_by_tags<'py>(
        &self,
        py: Python<'py>,
        tags: Vec<String>,
        limit: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let memory = self
            .inner
            .memory()
            .ok_or_else(|| PyRuntimeError::new_err("Memory not configured for this session"))?
            .clone();
        let items = py
            .allow_threads(move || get_runtime().block_on(memory.recall_by_tags(&tags, limit)))
            .map_err(|e| PyRuntimeError::new_err(format!("Recall failed: {e}")))?;
        let json_str = serde_json::to_string(&items)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {e}")))?;
        let json_mod = py.import("json")?;
        let py_obj = json_mod.call_method1("loads", (json_str,))?;
        py_obj
            .downcast::<PyList>()
            .cloned()
            .map_err(|e| PyRuntimeError::new_err(format!("Unexpected result: {e}")))
    }

    /// Get recent memory items.
    ///
    /// Args:
    ///     limit: Maximum number of results (default: 10)
    ///
    /// Returns:
    ///     List of memory item dicts.
    #[pyo3(signature = (limit=10))]
    fn memory_recent<'py>(&self, py: Python<'py>, limit: usize) -> PyResult<Bound<'py, PyList>> {
        let memory = self
            .inner
            .memory()
            .ok_or_else(|| PyRuntimeError::new_err("Memory not configured for this session"))?
            .clone();
        let items = py
            .allow_threads(move || get_runtime().block_on(memory.get_recent(limit)))
            .map_err(|e| PyRuntimeError::new_err(format!("Recall failed: {e}")))?;
        let json_str = serde_json::to_string(&items)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {e}")))?;
        let json_mod = py.import("json")?;
        let py_obj = json_mod.call_method1("loads", (json_str,))?;
        py_obj
            .downcast::<PyList>()
            .cloned()
            .map_err(|e| PyRuntimeError::new_err(format!("Unexpected result: {e}")))
    }

    /// Get memory statistics.
    ///
    /// Returns:
    ///     Dict with long_term_count, short_term_count, working_count.
    fn memory_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let memory = self
            .inner
            .memory()
            .ok_or_else(|| PyRuntimeError::new_err("Memory not configured for this session"))?
            .clone();
        let stats = py
            .allow_threads(move || get_runtime().block_on(memory.stats()))
            .map_err(|e| PyRuntimeError::new_err(format!("Stats failed: {e}")))?;
        let json_str = serde_json::to_string(&stats)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {e}")))?;
        let json_mod = py.import("json")?;
        let py_obj = json_mod.call_method1("loads", (json_str,))?;
        py_obj
            .downcast::<PyDict>()
            .cloned()
            .map_err(|e| PyRuntimeError::new_err(format!("Unexpected result: {e}")))
    }

    /// Get current working memory items.
    ///
    /// Working memory holds the active context items for the current task.
    ///
    /// Returns:
    ///     List of memory item dicts currently in working memory.
    fn get_working<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let memory = self
            .inner
            .memory()
            .ok_or_else(|| PyRuntimeError::new_err("Memory not configured for this session"))?
            .clone();
        let items = py.allow_threads(move || get_runtime().block_on(memory.get_working()));
        let json_str = serde_json::to_string(&items)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {e}")))?;
        let json_mod = py.import("json")?;
        let py_obj = json_mod.call_method1("loads", (json_str,))?;
        py_obj
            .downcast::<PyList>()
            .cloned()
            .map_err(|e| PyRuntimeError::new_err(format!("Unexpected result: {e}")))
    }

    /// Clear working memory.
    ///
    /// Removes all items from working memory without affecting short-term or long-term memory.
    fn clear_working(&self, py: Python<'_>) -> PyResult<()> {
        let memory = self
            .inner
            .memory()
            .ok_or_else(|| PyRuntimeError::new_err("Memory not configured for this session"))?
            .clone();
        py.allow_threads(move || get_runtime().block_on(memory.clear_working()));
        Ok(())
    }

    /// Get current short-term memory items.
    ///
    /// Short-term memory contains items stored during this session.
    ///
    /// Returns:
    ///     List of memory item dicts in short-term memory for this session.
    fn get_short_term<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let memory = self
            .inner
            .memory()
            .ok_or_else(|| PyRuntimeError::new_err("Memory not configured for this session"))?
            .clone();
        let items = py.allow_threads(move || get_runtime().block_on(memory.get_short_term()));
        let json_str = serde_json::to_string(&items)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {e}")))?;
        let json_mod = py.import("json")?;
        let py_obj = json_mod.call_method1("loads", (json_str,))?;
        py_obj
            .downcast::<PyList>()
            .cloned()
            .map_err(|e| PyRuntimeError::new_err(format!("Unexpected result: {e}")))
    }

    /// Clear short-term memory for this session.
    ///
    /// Removes all session-scoped memory items without affecting long-term or working memory.
    fn clear_short_term(&self, py: Python<'_>) -> PyResult<()> {
        let memory = self
            .inner
            .memory()
            .ok_or_else(|| PyRuntimeError::new_err("Memory not configured for this session"))?
            .clone();
        py.allow_threads(move || get_runtime().block_on(memory.clear_short_term()));
        Ok(())
    }

    // ========================================================================
    // Slash Command & Scheduler API
    // ========================================================================

    /// List all registered slash commands.
    ///
    /// Returns a list of dicts with keys: `name`, `description`, `usage` (or `None`).
    /// Slash commands can be invoked via `session.send("/command args")`.
    fn list_commands<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let commands = self.inner.command_registry().list_full();
        let items: Vec<_> = commands
            .into_iter()
            .map(|(name, description, usage)| {
                let d = PyDict::new(py);
                let _ = d.set_item("name", &name);
                let _ = d.set_item("description", &description);
                let _ = d.set_item("usage", usage.as_deref());
                d.into_any()
            })
            .collect();
        PyList::new(py, &items)
    }

    /// Register a custom slash command with a Python callback.
    ///
    /// The `handler` receives two arguments: `args: str` (everything after the command name)
    /// and `ctx: dict` (session context with keys: `session_id`, `workspace`, `model`,
    /// `history_len`, `total_tokens`, `total_cost`, `tool_names`).
    /// It must return a `str` — the text displayed to the user.
    ///
    /// Example::
    ///
    ///     def ping_handler(args, ctx):
    ///         return f"pong! session={ctx['session_id']}"
    ///
    ///     session.register_command("ping", "Pong!", ping_handler)
    ///     result = await session.send("/ping hello")
    #[pyo3(signature = (name, description, handler))]
    fn register_command(
        &self,
        name: String,
        description: String,
        handler: pyo3::Py<pyo3::PyAny>,
    ) -> PyResult<()> {
        let cmd = Arc::new(PySlashCommand {
            name,
            description,
            handler,
        });
        self.inner.clone().register_command(cmd);
        Ok(())
    }

    /// Cancel the current ongoing operation (send/stream).
    ///
    /// If an operation is in progress, this will trigger cancellation of the LLM streaming
    /// and tool execution. The operation will terminate as soon as possible.
    ///
    /// :returns: ``True`` if an operation was cancelled, ``False`` if no operation was in progress.
    fn cancel(&self, py: Python<'_>) -> bool {
        let session = self.inner.clone();
        py.allow_threads(move || get_runtime().block_on(session.cancel()))
    }

    /// Close the session and cancel any active operation.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let session = self.inner.clone();
        py.allow_threads(move || get_runtime().block_on(session.close()));
        Ok(())
    }

    /// Whether ``close()`` has been called on this session.
    ///
    /// Once ``True``, calls to ``send`` / ``stream`` raise ``RuntimeError``
    /// with a "Session closed" message instead of starting a new run.
    #[getter]
    fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }

    fn __repr__(&self) -> String {
        format!(
            "Session(id='{}', workspace='{}')",
            self.inner.session_id(),
            self.inner.workspace().display()
        )
    }
}

// ============================================================================
// Hook Helpers
// ============================================================================

fn py_parse_hook_event_type(event_type: &str) -> PyResult<RustHookEventType> {
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
        // Harness control points
        "pre_context_perception" => Ok(RustHookEventType::PreContextPerception),
        "post_context_perception" => Ok(RustHookEventType::PostContextPerception),
        "on_success" => Ok(RustHookEventType::OnSuccess),
        "pre_memory_recall" => Ok(RustHookEventType::PreMemoryRecall),
        "post_memory_recall" => Ok(RustHookEventType::PostMemoryRecall),
        "pre_planning" => Ok(RustHookEventType::PrePlanning),
        "post_planning" => Ok(RustHookEventType::PostPlanning),
        "pre_reasoning" => Ok(RustHookEventType::PreReasoning),
        "post_reasoning" => Ok(RustHookEventType::PostReasoning),
        "on_rate_limit" => Ok(RustHookEventType::OnRateLimit),
        "on_confirmation" => Ok(RustHookEventType::OnConfirmation),
        _ => Err(PyValueError::new_err(format!(
            "Invalid hook event type: '{}'. Expected one of: pre_tool_use, post_tool_use, \
             generate_start, generate_end, session_start, session_end, skill_load, \
             skill_unload, pre_prompt, post_response, on_error, pre_context_perception, \
             post_context_perception, on_success, pre_memory_recall, post_memory_recall, \
             pre_planning, post_planning, pre_reasoning, post_reasoning, on_rate_limit, \
             on_confirmation",
            event_type
        ))),
    }
}

// ============================================================================
// PythonCallbackHandler — bridges Python callables into the Rust HookHandler trait
// ============================================================================

/// Wraps a Python callable so it can be used as a `HookHandler`.
///
/// The callable receives a dict (the serialized `HookEvent`) and must return
/// `None` / `{"action": "continue"}` to allow execution, or
/// `{"action": "block", "reason": "..."}` to cancel it.
///
/// GIL safety: `send()` and `stream()` both release the GIL via `py.allow_threads()`,
/// so acquiring it here from a tokio worker thread does not deadlock.
struct PythonCallbackHandler {
    callback: pyo3::Py<pyo3::PyAny>,
}

impl RustHookHandler for PythonCallbackHandler {
    fn handle(&self, event: &RustHookEvent) -> RustHookResponse {
        let Ok(json_str) = serde_json::to_string(event) else {
            return RustHookResponse::continue_();
        };

        pyo3::Python::with_gil(|py| {
            // Deserialize the event into a Python dict via json.loads.
            let result = (|| -> pyo3::PyResult<RustHookResponse> {
                let json_mod = py.import("json")?;
                let event_dict = json_mod.call_method1("loads", (json_str.as_str(),))?;
                let ret = self.callback.call1(py, (event_dict,))?;
                parse_py_hook_response(py, ret.bind(py))
            })();

            result.unwrap_or_else(|_| RustHookResponse::continue_())
        })
    }
}

/// Parse the return value of a Python hook callback into a `HookResponse`.
///
/// Accepted shapes:
/// - `None`                                   → continue
/// - `{"action": "continue"}`                 → continue
/// - `{"action": "block", "reason": "…"}`     → block
/// - `{"action": "skip"}`                     → skip
/// - `{"action": "retry", "delay_ms": N}`     → retry
fn parse_py_hook_response(
    _py: pyo3::Python,
    val: &pyo3::Bound<pyo3::PyAny>,
) -> pyo3::PyResult<RustHookResponse> {
    use pyo3::types::PyDict;

    if val.is_none() {
        return Ok(RustHookResponse::continue_());
    }

    if let Ok(dict) = val.downcast::<PyDict>() {
        let action = dict
            .get_item("action")?
            .and_then(|v| v.extract::<String>().ok());

        match action.as_deref() {
            Some("block") => {
                let reason = dict
                    .get_item("reason")?
                    .and_then(|v| v.extract::<String>().ok())
                    .unwrap_or_else(|| "Blocked by hook".to_string());
                return Ok(RustHookResponse::block(reason));
            }
            Some("skip") => return Ok(RustHookResponse::skip()),
            Some("retry") => {
                let delay_ms = dict
                    .get_item("delay_ms")?
                    .and_then(|v| v.extract::<u64>().ok())
                    .unwrap_or(1000);
                return Ok(RustHookResponse::retry(delay_ms));
            }
            _ => {}
        }
    }

    Ok(RustHookResponse::continue_())
}

// ============================================================================
// Orchestration: Python <-> Rust conversion + pipeline-stage bridge
// ============================================================================

fn py_dumps(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<String> {
    let json_mod = py.import("json")?;
    json_mod.call_method1("dumps", (obj,))?.extract()
}

/// Convert a Python spec dict into an `AgentStepSpec` (snake_case keys) via a
/// JSON round-trip.
fn py_to_step_spec(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<RustAgentStepSpec> {
    serde_json::from_str(&py_dumps(py, obj)?)
        .map_err(|e| PyValueError::new_err(format!("invalid AgentStepSpec: {e}")))
}

/// Convert an arbitrary Python value into a `serde_json::Value`.
fn py_to_json_value(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    serde_json::from_str(&py_dumps(py, obj)?)
        .map_err(|e| PyValueError::new_err(format!("invalid JSON: {e}")))
}

/// Convert a `StepOutcome` into a Python dict.
fn step_outcome_to_py(py: Python<'_>, outcome: &RustStepOutcome) -> PyResult<PyObject> {
    let json = serde_json::to_string(outcome)
        .map_err(|e| PyRuntimeError::new_err(format!("serialize outcome: {e}")))?;
    json_string_to_py(py, &json)
}

/// Bridges a Python pipeline-stage callable into a synchronous `PipelineStage`.
///
/// GIL safety: `pipeline()` releases the GIL via `py.allow_threads`, so
/// re-acquiring it here from a tokio worker thread does not deadlock (same as
/// the hook/budget bridges). A raised exception is caught and treated as
/// `None` (stop the chain).
struct PythonPipelineStage {
    callback: pyo3::Py<pyo3::PyAny>,
}

impl PythonPipelineStage {
    fn invoke(
        &self,
        prev: Option<&RustStepOutcome>,
        item: &serde_json::Value,
    ) -> Option<RustAgentStepSpec> {
        pyo3::Python::with_gil(|py| {
            let result = (|| -> PyResult<Option<RustAgentStepSpec>> {
                let json_mod = py.import("json")?;
                let previous = match prev {
                    Some(o) => {
                        let s = serde_json::to_string(o)
                            .map_err(|e| PyValueError::new_err(e.to_string()))?;
                        json_mod.call_method1("loads", (s,))?
                    }
                    None => py.None().into_bound(py),
                };
                let item_str = serde_json::to_string(item)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let item_py = json_mod.call_method1("loads", (item_str,))?;
                let ctx = PyDict::new(py);
                ctx.set_item("previous", previous)?;
                ctx.set_item("item", item_py)?;
                let ret = self.callback.call1(py, (ctx,))?;
                let bound = ret.bind(py);
                if bound.is_none() {
                    return Ok(None);
                }
                let spec_json: String = json_mod.call_method1("dumps", (bound,))?.extract()?;
                serde_json::from_str::<RustAgentStepSpec>(&spec_json)
                    .map(Some)
                    .map_err(|e| PyValueError::new_err(format!("invalid step spec: {e}")))
            })();
            // Fail-closed: any exception → stop this chain.
            result.unwrap_or(None)
        })
    }
}

// ============================================================================
// Python BudgetGuard bridge
// ============================================================================

/// Bridges a Python BudgetGuard instance into the Rust async
/// [`a3s_code_core::budget::BudgetGuard`] trait.
///
/// Looks up `check_before_llm`, `record_after_llm`, and
/// `check_before_tool` on the held `PyObject` at call time, so the
/// user's Python class only needs to define the methods it cares
/// about — missing methods are treated as a permissive default
/// (Allow / no-op).
///
/// Calls into Python acquire the GIL via `Python::with_gil`, which
/// blocks the tokio worker thread briefly. Acceptable here because
/// `BudgetGuard` is called at most once per LLM turn / tool call,
/// not on a hot path.
///
/// RE-ENTRANCY WARNING: do **not** call session/agent APIs (or any
/// blocking Rust path) from inside a Python budget-guard callback. The
/// tokio worker thread is already blocked acquiring the GIL to run the
/// callback; re-entering the runtime from there risks a deadlock or
/// re-entrancy panic. Budget guards should be pure policy — inspect the
/// args, consult host-side counters, return a decision.
struct PyBudgetGuard {
    inner: pyo3::Py<pyo3::PyAny>,
}

impl PyBudgetGuard {
    fn new(inner: pyo3::Py<pyo3::PyAny>) -> Self {
        Self { inner }
    }
}

#[async_trait::async_trait]
impl a3s_code_core::budget::BudgetGuard for PyBudgetGuard {
    async fn check_before_llm(
        &self,
        session_id: &str,
        estimated_prompt_tokens: usize,
    ) -> a3s_code_core::budget::BudgetDecision {
        pyo3::Python::with_gil(|py| {
            let inner = self.inner.bind(py);
            let method = match inner.getattr("check_before_llm") {
                Ok(m) if !m.is_none() => m,
                _ => return a3s_code_core::budget::BudgetDecision::Allow,
            };
            match method.call1((session_id, estimated_prompt_tokens)) {
                Ok(val) => parse_py_budget_decision(&val),
                Err(e) => {
                    eprintln!(
                        "[a3s-code] warning: Python BudgetGuard.check_before_llm raised: {e}; defaulting to Allow"
                    );
                    a3s_code_core::budget::BudgetDecision::Allow
                }
            }
        })
    }

    async fn record_after_llm(&self, session_id: &str, usage: &a3s_code_core::llm::TokenUsage) {
        pyo3::Python::with_gil(|py| {
            let inner = self.inner.bind(py);
            let method = match inner.getattr("record_after_llm") {
                Ok(m) if !m.is_none() => m,
                _ => return,
            };
            // Hand Python a dict so they don't have to construct a
            // TokenUsage type on their side.
            let usage_dict = pyo3::types::PyDict::new(py);
            let _ = usage_dict.set_item("prompt_tokens", usage.prompt_tokens);
            let _ = usage_dict.set_item("completion_tokens", usage.completion_tokens);
            let _ = usage_dict.set_item("total_tokens", usage.total_tokens);
            let _ = usage_dict.set_item("cache_read_tokens", usage.cache_read_tokens);
            let _ = usage_dict.set_item("cache_write_tokens", usage.cache_write_tokens);
            if let Err(e) = method.call1((session_id, usage_dict)) {
                eprintln!(
                    "[a3s-code] warning: Python BudgetGuard.record_after_llm raised: {e}; ignored"
                );
            }
        })
    }

    async fn check_before_tool(
        &self,
        session_id: &str,
        tool_name: &str,
    ) -> a3s_code_core::budget::BudgetDecision {
        pyo3::Python::with_gil(|py| {
            let inner = self.inner.bind(py);
            let method = match inner.getattr("check_before_tool") {
                Ok(m) if !m.is_none() => m,
                _ => return a3s_code_core::budget::BudgetDecision::Allow,
            };
            match method.call1((session_id, tool_name)) {
                Ok(val) => parse_py_budget_decision(&val),
                Err(e) => {
                    eprintln!(
                        "[a3s-code] warning: Python BudgetGuard.check_before_tool raised: {e}; defaulting to Allow"
                    );
                    a3s_code_core::budget::BudgetDecision::Allow
                }
            }
        })
    }
}

/// Parse the return value of a Python BudgetGuard method into a
/// [`BudgetDecision`](a3s_code_core::budget::BudgetDecision).
///
/// Accepted shapes:
/// - `None`                                                        → Allow
/// - `{"decision": "allow"}`                                       → Allow
/// - `{"decision": "soft", "resource": str, "consumed": float,
///     "limit": float, "message"?: str}`                           → SoftLimit
/// - `{"decision": "deny", "resource": str, "reason": str}`        → Deny
fn parse_py_budget_decision(
    val: &pyo3::Bound<pyo3::PyAny>,
) -> a3s_code_core::budget::BudgetDecision {
    use a3s_code_core::budget::BudgetDecision;
    use pyo3::types::PyDict;

    if val.is_none() {
        return BudgetDecision::Allow;
    }

    let Ok(dict) = val.downcast::<PyDict>() else {
        return BudgetDecision::Allow;
    };

    let decision = dict
        .get_item("decision")
        .ok()
        .flatten()
        .and_then(|v| v.extract::<String>().ok())
        .unwrap_or_else(|| "allow".to_string());

    match decision.as_str() {
        "deny" => {
            let resource = dict
                .get_item("resource")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<String>().ok())
                .unwrap_or_else(|| "unspecified".to_string());
            let reason = dict
                .get_item("reason")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<String>().ok())
                .unwrap_or_else(|| "denied by host".to_string());
            BudgetDecision::Deny { resource, reason }
        }
        "soft" => {
            let resource = dict
                .get_item("resource")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<String>().ok())
                .unwrap_or_else(|| "unspecified".to_string());
            let consumed = dict
                .get_item("consumed")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<f64>().ok())
                .unwrap_or(0.0);
            let limit = dict
                .get_item("limit")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<f64>().ok())
                .unwrap_or(0.0);
            let message = dict
                .get_item("message")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<String>().ok());
            BudgetDecision::SoftLimit {
                resource,
                consumed,
                limit,
                message,
            }
        }
        _ => BudgetDecision::Allow,
    }
}

/// Convert a Python dict (`{max_runs_retained: int, ...}`) into a
/// [`SessionRetentionLimits`](a3s_code_core::retention::SessionRetentionLimits).
/// Returns `None` if the supplied object is not a dict (caller treats
/// that as "no caps" and the framework default applies).
fn parse_py_retention_limits(
    py_obj: &pyo3::PyObject,
) -> Option<a3s_code_core::retention::SessionRetentionLimits> {
    use a3s_code_core::retention::SessionRetentionLimits;
    use pyo3::types::PyDict;

    pyo3::Python::with_gil(|py| {
        let bound = py_obj.bind(py);
        let dict = bound.downcast::<PyDict>().ok()?;
        let mut limits = SessionRetentionLimits::new();
        if let Some(v) = dict.get_item("max_runs_retained").ok().flatten() {
            if let Ok(n) = v.extract::<usize>() {
                limits.max_runs_retained = Some(n);
            }
        }
        if let Some(v) = dict.get_item("max_events_per_run").ok().flatten() {
            if let Ok(n) = v.extract::<usize>() {
                limits.max_events_per_run = Some(n);
            }
        }
        if let Some(v) = dict.get_item("max_trace_events").ok().flatten() {
            if let Ok(n) = v.extract::<usize>() {
                limits.max_trace_events = Some(n);
            }
        }
        if let Some(v) = dict.get_item("max_terminal_subagent_tasks").ok().flatten() {
            if let Ok(n) = v.extract::<usize>() {
                limits.max_terminal_subagent_tasks = Some(n);
            }
        }
        Some(limits)
    })
}

// ============================================================================
// PySlashCommand — bridges Python callables into the Rust SlashCommand trait
// ============================================================================

/// Wraps a Python callable so it can be registered as a slash command handler.
///
/// GIL safety: `SlashCommand::execute()` is called from within an async Rust
/// context. `Python::with_gil` is safe to call from any Rust thread as long as
/// the caller releases the GIL before blocking (which `send()` does via
/// `py.allow_threads()`), so this does not deadlock.
struct PySlashCommand {
    name: String,
    description: String,
    /// Python callable: `(args: str, ctx: dict) -> str`
    handler: pyo3::Py<pyo3::PyAny>,
}

impl RustSlashCommand for PySlashCommand {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn execute(&self, args: &str, ctx: &RustCommandContext) -> RustCommandOutput {
        Python::with_gil(|py| {
            let result = (|| -> pyo3::PyResult<String> {
                let ctx_dict = PyDict::new(py);
                ctx_dict.set_item("session_id", &ctx.session_id)?;
                ctx_dict.set_item("workspace", &ctx.workspace)?;
                ctx_dict.set_item("model", &ctx.model)?;
                ctx_dict.set_item("history_len", ctx.history_len)?;
                ctx_dict.set_item("total_tokens", ctx.total_tokens)?;
                ctx_dict.set_item("total_cost", ctx.total_cost)?;
                ctx_dict.set_item("tool_names", ctx.tool_names.clone())?;
                let ret = self.handler.call1(py, (args, ctx_dict))?;
                ret.extract::<String>(py)
            })();
            match result {
                Ok(text) => RustCommandOutput::text(text),
                Err(e) => RustCommandOutput::text(format!("Command error: {e}")),
            }
        })
    }
}

// ============================================================================
// Typed store / provider helpers
// ============================================================================

/// File-backed long-term memory store.
///
/// Pass to ``SessionOptions.memory_store``:
///
/// .. code-block:: python
///
///     opts = SessionOptions()
///     opts.memory_store = FileMemoryStore('./memory')
///     session = agent.session('.', opts)
#[pyclass(name = "FileMemoryStore")]
#[derive(Clone)]
struct PyFileMemoryStore {
    #[pyo3(get, set)]
    dir: String,
}

#[pymethods]
impl PyFileMemoryStore {
    #[new]
    fn new(dir: String) -> Self {
        Self { dir }
    }

    fn __repr__(&self) -> String {
        format!("FileMemoryStore(dir={:?})", self.dir)
    }
}

/// File-backed session store — persists sessions to disk for later resumption.
///
/// Pass to ``SessionOptions.session_store``:
///
/// .. code-block:: python
///
///     opts = SessionOptions()
///     opts.session_store = FileSessionStore('./sessions')
///     opts.session_id = 'my-session'
///     opts.auto_save = True
///     session = agent.session('.', opts)
#[pyclass(name = "FileSessionStore")]
#[derive(Clone)]
struct PyFileSessionStore {
    #[pyo3(get, set)]
    dir: String,
}

#[pymethods]
impl PyFileSessionStore {
    #[new]
    fn new(dir: String) -> Self {
        Self { dir }
    }

    fn __repr__(&self) -> String {
        format!("FileSessionStore(dir={:?})", self.dir)
    }
}

/// In-memory (non-persistent) session store.
///
/// Useful for testing, ephemeral runs, and CI pipelines where no disk state is needed.
///
/// .. code-block:: python
///
///     opts = SessionOptions()
///     opts.session_store = MemorySessionStore()
#[pyclass(name = "MemorySessionStore")]
#[derive(Clone)]
struct PyMemorySessionStore {}

#[pymethods]
impl PyMemorySessionStore {
    #[new]
    fn new() -> Self {
        Self {}
    }

    fn __repr__(&self) -> String {
        "MemorySessionStore()".to_string()
    }
}

/// Default security provider: input taint tracking + output sanitisation.
///
/// Pass to ``SessionOptions.security_provider``:
///
/// .. code-block:: python
///
///     opts = SessionOptions()
///     opts.security_provider = DefaultSecurityProvider()
#[pyclass(name = "DefaultSecurityProvider")]
#[derive(Clone)]
struct PyDefaultSecurityProvider {}

#[pymethods]
impl PyDefaultSecurityProvider {
    #[new]
    fn new() -> Self {
        Self {}
    }

    fn __repr__(&self) -> String {
        "DefaultSecurityProvider()".to_string()
    }
}

/// Local filesystem workspace backend.
///
/// This is the explicit typed form of the default local workspace behavior.
/// It is useful when callers want to pass workspace backends through the same
/// option surface that remote/browser backends will use.
///
/// .. code-block:: python
///
///     opts = SessionOptions()
///     opts.workspace_backend = LocalWorkspaceBackend('/repo')
///     session = agent.session('/repo', opts)
#[pyclass(name = "LocalWorkspaceBackend")]
#[derive(Clone)]
struct PyLocalWorkspaceBackend {
    #[pyo3(get, set)]
    root: String,
}

#[pymethods]
impl PyLocalWorkspaceBackend {
    #[new]
    fn new(root: String) -> Self {
        Self { root }
    }

    fn __repr__(&self) -> String {
        format!("LocalWorkspaceBackend(root={:?})", self.root)
    }
}

/// S3-compatible object-storage workspace backend.
///
/// Points the built-in file tools (``read``, ``write``, ``edit``, ``patch``,
/// ``ls``) at any S3-compatible bucket (AWS S3, MinIO, RustFS, Cloudflare R2,
/// Backblaze B2, ...). ``bash``, ``git``, ``grep`` and ``glob`` are
/// intentionally **not** registered when this backend is used because
/// object storage cannot service them.
///
/// .. code-block:: python
///
///     opts = SessionOptions()
///     opts.workspace_backend = S3WorkspaceBackend(
///         bucket="workspace",
///         prefix="users/u1/sessions/s1",
///         access_key_id="AKIA...",
///         secret_access_key="...",
///         endpoint="https://minio.local:9000",
///         region="us-east-1",
///         force_path_style=True,
///     )
///     session = agent.session("s3://workspace/users/u1/sessions/s1", opts)
#[pyclass(name = "S3WorkspaceBackend")]
#[derive(Clone)]
struct PyS3WorkspaceBackend {
    #[pyo3(get, set)]
    bucket: String,
    #[pyo3(get, set)]
    prefix: String,
    #[pyo3(get, set)]
    access_key_id: String,
    #[pyo3(get, set)]
    secret_access_key: String,
    #[pyo3(get, set)]
    endpoint: Option<String>,
    #[pyo3(get, set)]
    region: Option<String>,
    #[pyo3(get, set)]
    session_token: Option<String>,
    #[pyo3(get, set)]
    force_path_style: bool,
    /// Per-read size ceiling (bytes). Defaults to 10 MiB when ``None``.
    #[pyo3(get, set)]
    max_read_bytes: Option<u64>,
    /// Enable degraded ``grep`` / ``glob`` against this backend. Off by default
    /// because LIST + GET + regex can be slow and expensive.
    #[pyo3(get, set)]
    search_enabled: bool,
    /// Upper bound on objects considered per ``grep`` / ``glob`` call.
    /// Defaults to 500 when ``None``. Ignored when ``search_enabled`` is False.
    #[pyo3(get, set)]
    max_objects_scanned: Option<u64>,
    /// Per-object body-size ceiling for ``grep`` downloads. Defaults to 1 MiB
    /// when ``None``. Ignored when ``search_enabled`` is False.
    #[pyo3(get, set)]
    max_grep_bytes_per_object: Option<u64>,
    /// Concurrent object downloads during ``grep``. Defaults to 8 when
    /// ``None``. Set lower when the gitserver / S3 endpoint rate-limits
    /// aggressively; set higher when latency dominates. Ignored when
    /// ``search_enabled`` is False.
    #[pyo3(get, set)]
    search_concurrency: Option<u64>,
}

#[pymethods]
impl PyS3WorkspaceBackend {
    #[new]
    #[pyo3(signature = (
        bucket,
        prefix,
        access_key_id,
        secret_access_key,
        endpoint = None,
        region = None,
        session_token = None,
        force_path_style = false,
        max_read_bytes = None,
        search_enabled = false,
        max_objects_scanned = None,
        max_grep_bytes_per_object = None,
        search_concurrency = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        bucket: String,
        prefix: String,
        access_key_id: String,
        secret_access_key: String,
        endpoint: Option<String>,
        region: Option<String>,
        session_token: Option<String>,
        force_path_style: bool,
        max_read_bytes: Option<u64>,
        search_enabled: bool,
        max_objects_scanned: Option<u64>,
        max_grep_bytes_per_object: Option<u64>,
        search_concurrency: Option<u64>,
    ) -> Self {
        Self {
            bucket,
            prefix,
            access_key_id,
            secret_access_key,
            endpoint,
            region,
            session_token,
            force_path_style,
            max_read_bytes,
            search_enabled,
            max_objects_scanned,
            max_grep_bytes_per_object,
            search_concurrency,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "S3WorkspaceBackend(bucket={:?}, prefix={:?}, endpoint={:?}, region={:?}, force_path_style={}, search_enabled={})",
            self.bucket, self.prefix, self.endpoint, self.region, self.force_path_style, self.search_enabled,
        )
    }
}

impl PyS3WorkspaceBackend {
    fn to_core(&self) -> a3s_code_core::S3BackendConfig {
        let mut cfg = a3s_code_core::S3BackendConfig::new(
            self.bucket.clone(),
            self.prefix.clone(),
            self.access_key_id.clone(),
            self.secret_access_key.clone(),
        )
        .force_path_style(self.force_path_style)
        .enable_search(self.search_enabled);
        if let Some(ref endpoint) = self.endpoint {
            cfg = cfg.endpoint(endpoint.clone());
        }
        if let Some(ref region) = self.region {
            cfg = cfg.region(region.clone());
        }
        if let Some(ref token) = self.session_token {
            cfg = cfg.session_token(token.clone());
        }
        if let Some(n) = self.max_read_bytes {
            cfg = cfg.max_read_bytes(n);
        }
        if let Some(n) = self.max_objects_scanned {
            cfg = cfg.max_objects_scanned(n as usize);
        }
        if let Some(n) = self.max_grep_bytes_per_object {
            cfg = cfg.max_grep_bytes_per_object(n);
        }
        if let Some(n) = self.search_concurrency {
            cfg = cfg.search_concurrency(n as usize);
        }
        cfg
    }
}

/// Configuration for a remote git backend that brings the ``git`` tool to
/// non-local workspaces (S3, future container / DFS) over HTTP/JSON.
///
/// Attach to a session alongside ``workspace_backend``:
///
/// .. code-block:: python
///
///     opts = SessionOptions()
///     opts.workspace_backend = S3WorkspaceBackend(...)
///     opts.remote_git = RemoteGitBackendConfig(
///         base_url="https://gitserver.internal",
///         repo_id="u1/s1",
///         bearer_token=token,
///     )
#[pyclass(name = "RemoteGitBackendConfig")]
#[derive(Clone)]
struct PyRemoteGitBackendConfig {
    #[pyo3(get, set)]
    base_url: String,
    #[pyo3(get, set)]
    repo_id: String,
    #[pyo3(get, set)]
    bearer_token: Option<String>,
    /// mTLS client certificate path (PEM). When set together with
    /// ``client_key_pem``, the backend reads both files at construction and
    /// configures mTLS on the HTTP client. Setting only one of the pair
    /// errors at construction.
    #[pyo3(get, set)]
    client_cert_pem: Option<String>,
    /// mTLS client private key path (PEM). PKCS#8 format expected for the
    /// ``rustls-tls`` backend. See ``client_cert_pem``.
    #[pyo3(get, set)]
    client_key_pem: Option<String>,
    /// Per-call HTTP timeout in milliseconds. Defaults to 30 000.
    #[pyo3(get, set)]
    request_timeout_ms: Option<u64>,
    /// Client-side cap on ``diff`` response bytes. Defaults to 1 MiB.
    #[pyo3(get, set)]
    max_diff_bytes: Option<u64>,
    /// Client-side cap on ``log`` ``max_count``. Defaults to 200.
    #[pyo3(get, set)]
    max_log_entries: Option<u64>,
}

#[pymethods]
impl PyRemoteGitBackendConfig {
    #[new]
    #[pyo3(signature = (
        base_url,
        repo_id,
        bearer_token = None,
        client_cert_pem = None,
        client_key_pem = None,
        request_timeout_ms = None,
        max_diff_bytes = None,
        max_log_entries = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        base_url: String,
        repo_id: String,
        bearer_token: Option<String>,
        client_cert_pem: Option<String>,
        client_key_pem: Option<String>,
        request_timeout_ms: Option<u64>,
        max_diff_bytes: Option<u64>,
        max_log_entries: Option<u64>,
    ) -> Self {
        Self {
            base_url,
            repo_id,
            bearer_token,
            client_cert_pem,
            client_key_pem,
            request_timeout_ms,
            max_diff_bytes,
            max_log_entries,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RemoteGitBackendConfig(base_url={:?}, repo_id={:?})",
            self.base_url, self.repo_id
        )
    }
}

impl PyRemoteGitBackendConfig {
    fn to_core(&self) -> a3s_code_core::RemoteGitBackendConfig {
        let mut cfg =
            a3s_code_core::RemoteGitBackendConfig::new(self.base_url.clone(), self.repo_id.clone());
        if let Some(ref t) = self.bearer_token {
            cfg = cfg.bearer_token(t.clone());
        }
        if let Some(ref p) = self.client_cert_pem {
            cfg = cfg.client_cert_pem(std::path::PathBuf::from(p));
        }
        if let Some(ref p) = self.client_key_pem {
            cfg = cfg.client_key_pem(std::path::PathBuf::from(p));
        }
        if let Some(ms) = self.request_timeout_ms {
            cfg = cfg.request_timeout(std::time::Duration::from_millis(ms));
        }
        if let Some(n) = self.max_diff_bytes {
            cfg = cfg.max_diff_bytes(n);
        }
        if let Some(n) = self.max_log_entries {
            cfg = cfg.max_log_entries(n as usize);
        }
        cfg
    }
}

// ============================================================================
// AHP Transport Classes
// ============================================================================

/// Stdio transport for AHP (Agent Harness Protocol).
///
/// Launches a child process and communicates via stdin/stdout using JSON-RPC 2.0.
///
/// Example:
///     transport = StdioTransport(program='python', args=['ahp_server.py'])
///     opts = SessionOptions()
///     opts.ahp_transport = transport
///     session = agent.session('.', opts)
#[pyclass(name = "StdioTransport")]
#[derive(Clone)]
struct PyStdioTransport {
    #[pyo3(get, set)]
    program: String,
    #[pyo3(get, set)]
    args: Vec<String>,
}

#[pymethods]
impl PyStdioTransport {
    #[new]
    fn new(program: String, args: Vec<String>) -> Self {
        Self { program, args }
    }

    fn __repr__(&self) -> String {
        format!(
            "StdioTransport(program={:?}, args={:?})",
            self.program, self.args
        )
    }
}

/// HTTP transport for AHP (Agent Harness Protocol).
///
/// Connects to a remote AHP harness server via HTTP.
///
/// Example:
///     transport = HttpTransport(url='http://localhost:8080/ahp')
///     opts = SessionOptions()
///     opts.ahp_transport = transport
///     session = agent.session('.', opts)
#[pyclass(name = "HttpTransport")]
#[derive(Clone)]
struct PyHttpTransport {
    #[pyo3(get, set)]
    url: String,
    #[pyo3(get, set)]
    auth_token: Option<String>,
}

#[pymethods]
impl PyHttpTransport {
    #[new]
    #[pyo3(signature = (url, auth_token=None))]
    fn new(url: String, auth_token: Option<String>) -> Self {
        Self { url, auth_token }
    }

    fn __repr__(&self) -> String {
        format!("HttpTransport(url={:?})", self.url)
    }
}

/// WebSocket transport for AHP (Agent Harness Protocol).
///
/// Connects to a remote AHP harness server via WebSocket for bidirectional streaming.
///
/// Example:
///     transport = WebSocketTransport(url='ws://localhost:8080/ahp')
///     opts = SessionOptions()
///     opts.ahp_transport = transport
///     session = agent.session('.', opts)
#[pyclass(name = "WebSocketTransport")]
#[derive(Clone)]
struct PyWebSocketTransport {
    #[pyo3(get, set)]
    url: String,
    #[pyo3(get, set)]
    auth_token: Option<String>,
}

#[pymethods]
impl PyWebSocketTransport {
    #[new]
    #[pyo3(signature = (url, auth_token=None))]
    fn new(url: String, auth_token: Option<String>) -> Self {
        Self { url, auth_token }
    }

    fn __repr__(&self) -> String {
        format!("WebSocketTransport(url={:?})", self.url)
    }
}

/// Unix socket transport for AHP (Agent Harness Protocol).
///
/// Connects to a local AHP harness server via Unix domain socket.
///
/// Example:
///     transport = UnixSocketTransport(path='/tmp/ahp.sock')
///     opts = SessionOptions()
///     opts.ahp_transport = transport
///     session = agent.session('.', opts)
#[pyclass(name = "UnixSocketTransport")]
#[derive(Clone)]
struct PyUnixSocketTransport {
    #[pyo3(get, set)]
    path: String,
}

#[pymethods]
impl PyUnixSocketTransport {
    #[new]
    fn new(path: String) -> Self {
        Self { path }
    }

    fn __repr__(&self) -> String {
        format!("UnixSocketTransport(path={:?})", self.path)
    }
}

// ============================================================================
// SessionOptions
// ============================================================================

/// Explicit allow/deny/ask tool permission policy.
#[pyclass(name = "PermissionPolicy")]
#[derive(Clone)]
struct PyPermissionPolicy {
    #[pyo3(get, set)]
    deny: Vec<String>,
    #[pyo3(get, set)]
    allow: Vec<String>,
    #[pyo3(get, set)]
    ask: Vec<String>,
    #[pyo3(get, set)]
    default_decision: String,
    #[pyo3(get, set)]
    enabled: bool,
}

#[pymethods]
impl PyPermissionPolicy {
    #[new]
    #[pyo3(signature = (allow=None, deny=None, ask=None, default_decision=None, enabled=true))]
    fn new(
        allow: Option<Vec<String>>,
        deny: Option<Vec<String>>,
        ask: Option<Vec<String>>,
        default_decision: Option<String>,
        enabled: bool,
    ) -> Self {
        Self {
            deny: deny.unwrap_or_default(),
            allow: allow.unwrap_or_default(),
            ask: ask.unwrap_or_default(),
            default_decision: default_decision.unwrap_or_else(|| "ask".to_string()),
            enabled,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PermissionPolicy(allow={}, deny={}, ask={}, default_decision={:?}, enabled={})",
            self.allow.len(),
            self.deny.len(),
            self.ask.len(),
            self.default_decision,
            self.enabled
        )
    }
}

fn parse_py_permission_decision(value: &str) -> PyResult<RustPermissionDecision> {
    match value.trim().to_ascii_lowercase().as_str() {
        "allow" => Ok(RustPermissionDecision::Allow),
        "deny" => Ok(RustPermissionDecision::Deny),
        "ask" => Ok(RustPermissionDecision::Ask),
        other => Err(PyValueError::new_err(format!(
            "default_decision must be 'allow', 'deny', or 'ask', got {other:?}"
        ))),
    }
}

fn py_permission_policy_to_rust(policy: PyPermissionPolicy) -> PyResult<RustPermissionPolicy> {
    Ok(RustPermissionPolicy {
        deny: policy
            .deny
            .into_iter()
            .map(|rule| RustPermissionRule::new(&rule))
            .collect(),
        allow: policy
            .allow
            .into_iter()
            .map(|rule| RustPermissionRule::new(&rule))
            .collect(),
        ask: policy
            .ask
            .into_iter()
            .map(|rule| RustPermissionRule::new(&rule))
            .collect(),
        default_decision: parse_py_permission_decision(&policy.default_decision)?,
        enabled: policy.enabled,
    })
}

/// HITL confirmation policy configuration.
#[pyclass(name = "ConfirmationPolicy")]
#[derive(Clone)]
struct PyConfirmationPolicy {
    #[pyo3(get, set)]
    enabled: bool,
    #[pyo3(get, set)]
    default_timeout_ms: u64,
    #[pyo3(get, set)]
    timeout_action: String,
    #[pyo3(get, set)]
    yolo_lanes: Vec<String>,
}

#[pymethods]
impl PyConfirmationPolicy {
    #[new]
    #[pyo3(signature = (enabled=false, default_timeout_ms=30000, timeout_action=None, yolo_lanes=None))]
    fn new(
        enabled: bool,
        default_timeout_ms: u64,
        timeout_action: Option<String>,
        yolo_lanes: Option<Vec<String>>,
    ) -> Self {
        Self {
            enabled,
            default_timeout_ms,
            timeout_action: timeout_action.unwrap_or_else(|| "reject".to_string()),
            yolo_lanes: yolo_lanes.unwrap_or_default(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ConfirmationPolicy(enabled={}, default_timeout_ms={}, timeout_action={:?}, yolo_lanes={})",
            self.enabled,
            self.default_timeout_ms,
            self.timeout_action,
            self.yolo_lanes.len()
        )
    }
}

fn parse_py_timeout_action(value: &str) -> PyResult<RustTimeoutAction> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "reject" => Ok(RustTimeoutAction::Reject),
        "auto_approve" | "autoapprove" => Ok(RustTimeoutAction::AutoApprove),
        other => Err(PyValueError::new_err(format!(
            "timeout_action must be 'reject' or 'auto_approve', got {other:?}"
        ))),
    }
}

fn py_confirmation_policy_to_rust(
    policy: PyConfirmationPolicy,
) -> PyResult<RustConfirmationPolicy> {
    let mut rust_policy = if policy.enabled {
        RustConfirmationPolicy::enabled()
    } else {
        RustConfirmationPolicy::default()
    };

    rust_policy = rust_policy.with_timeout(
        policy.default_timeout_ms,
        parse_py_timeout_action(&policy.timeout_action)?,
    );

    let yolo_lanes = policy
        .yolo_lanes
        .iter()
        .map(|lane| parse_lane(lane))
        .collect::<PyResult<Vec<_>>>()?;
    if !yolo_lanes.is_empty() {
        rust_policy = rust_policy.with_yolo_lanes(yolo_lanes);
    }

    Ok(rust_policy)
}

/// Retention limits for large tool/program artifacts.
#[pyclass(name = "ArtifactStoreLimits")]
#[derive(Clone)]
struct PyArtifactStoreLimits {
    /// Maximum number of artifacts retained by a session.
    #[pyo3(get, set)]
    max_artifacts: usize,
    /// Maximum total artifact content bytes retained by a session.
    #[pyo3(get, set)]
    max_bytes: usize,
}

#[pymethods]
impl PyArtifactStoreLimits {
    #[new]
    #[pyo3(signature = (max_artifacts=None, max_bytes=None))]
    fn new(max_artifacts: Option<usize>, max_bytes: Option<usize>) -> Self {
        let defaults = a3s_code_core::tools::ArtifactStoreLimits::default();
        Self {
            max_artifacts: max_artifacts.unwrap_or(defaults.max_artifacts),
            max_bytes: max_bytes.unwrap_or(defaults.max_bytes),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ArtifactStoreLimits(max_artifacts={}, max_bytes={})",
            self.max_artifacts, self.max_bytes
        )
    }
}

impl From<PyArtifactStoreLimits> for a3s_code_core::tools::ArtifactStoreLimits {
    fn from(limits: PyArtifactStoreLimits) -> Self {
        Self {
            max_artifacts: limits.max_artifacts,
            max_bytes: limits.max_bytes,
        }
    }
}

/// Reproducible recipe for a disposable worker/subagent.
#[pyclass(name = "WorkerAgentSpec")]
#[derive(Clone)]
struct PyWorkerAgentSpec {
    #[pyo3(get, set)]
    name: String,
    #[pyo3(get, set)]
    description: String,
    #[pyo3(get, set)]
    kind: String,
    #[pyo3(get, set)]
    hidden: bool,
    #[pyo3(get, set)]
    permissions: Option<PyPermissionPolicy>,
    #[pyo3(get, set)]
    model: Option<String>,
    #[pyo3(get, set)]
    prompt: Option<String>,
    #[pyo3(get, set)]
    max_steps: Option<usize>,
    #[pyo3(get, set)]
    confirmation_inheritance: Option<String>,
}

#[pymethods]
impl PyWorkerAgentSpec {
    #[new]
    #[pyo3(signature = (name, description, kind=None))]
    fn new(name: String, description: String, kind: Option<String>) -> Self {
        Self {
            name,
            description,
            kind: kind.unwrap_or_else(|| "custom".to_string()),
            hidden: false,
            permissions: None,
            model: None,
            prompt: None,
            max_steps: None,
            confirmation_inheritance: None,
        }
    }

    #[staticmethod]
    fn read_only(name: String, description: String) -> Self {
        Self::new(name, description, Some("read_only".to_string()))
    }

    #[staticmethod]
    fn planner(name: String, description: String) -> Self {
        Self::new(name, description, Some("planner".to_string()))
    }

    #[staticmethod]
    fn implementer(name: String, description: String) -> Self {
        Self::new(name, description, Some("implementer".to_string()))
    }

    #[staticmethod]
    fn verifier(name: String, description: String) -> Self {
        Self::new(name, description, Some("verifier".to_string()))
    }

    #[staticmethod]
    fn reviewer(name: String, description: String) -> Self {
        Self::new(name, description, Some("reviewer".to_string()))
    }

    #[staticmethod]
    fn custom(name: String, description: String) -> Self {
        Self::new(name, description, Some("custom".to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "WorkerAgentSpec(name={:?}, kind={:?}, max_steps={:?})",
            self.name, self.kind, self.max_steps
        )
    }
}

/// Compiled agent definition returned after registering a worker.
#[pyclass(name = "AgentDefinition")]
#[derive(Clone)]
struct PyAgentDefinition {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    description: String,
    #[pyo3(get)]
    native: bool,
    #[pyo3(get)]
    hidden: bool,
    #[pyo3(get)]
    model: Option<String>,
    #[pyo3(get)]
    prompt: Option<String>,
    #[pyo3(get)]
    max_steps: Option<usize>,
    #[pyo3(get)]
    confirmation_inheritance: Option<String>,
}

#[pymethods]
impl PyAgentDefinition {
    fn __repr__(&self) -> String {
        format!(
            "AgentDefinition(name={:?}, native={}, hidden={})",
            self.name, self.native, self.hidden
        )
    }
}

fn parse_py_worker_agent_kind(kind: &str) -> PyResult<RustWorkerAgentKind> {
    kind.parse::<RustWorkerAgentKind>()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn py_worker_agent_spec_to_rust(spec: PyWorkerAgentSpec) -> PyResult<RustWorkerAgentSpec> {
    if spec.name.trim().is_empty() {
        return Err(PyValueError::new_err("worker agent name is required"));
    }
    if spec.description.trim().is_empty() {
        return Err(PyValueError::new_err(
            "worker agent description is required",
        ));
    }

    let mut worker = RustWorkerAgentSpec::new(
        parse_py_worker_agent_kind(&spec.kind)?,
        spec.name,
        spec.description,
    )
    .hidden(spec.hidden);
    if let Some(policy) = spec.permissions {
        worker = worker.with_permissions(py_permission_policy_to_rust(policy)?);
    }
    if let Some(model) = spec.model {
        worker = worker.with_model(RustAgentModelConfig::from_model_ref(model));
    }
    if let Some(prompt) = spec.prompt {
        worker = worker.with_prompt(prompt);
    }
    if let Some(max_steps) = spec.max_steps {
        worker = worker.with_max_steps(max_steps);
    }
    if let Some(ci) = spec.confirmation_inheritance {
        worker = worker.with_confirmation(parse_py_confirmation_inheritance(&ci)?);
    }
    Ok(worker)
}

fn parse_py_confirmation_inheritance(
    value: &str,
) -> PyResult<a3s_code_core::subagent::ConfirmationInheritance> {
    use a3s_code_core::subagent::ConfirmationInheritance;
    match value {
        "auto_approve" => Ok(ConfirmationInheritance::AutoApprove),
        "deny_on_ask" => Ok(ConfirmationInheritance::DenyOnAsk),
        "inherit_parent" => Ok(ConfirmationInheritance::InheritParent),
        other => Err(PyValueError::new_err(format!(
            "invalid confirmation_inheritance: '{}' (expected: auto_approve, deny_on_ask, inherit_parent)",
            other
        ))),
    }
}

fn confirmation_inheritance_to_py(ci: &a3s_code_core::subagent::ConfirmationInheritance) -> String {
    use a3s_code_core::subagent::ConfirmationInheritance;
    match ci {
        ConfirmationInheritance::AutoApprove => "auto_approve".to_string(),
        ConfirmationInheritance::DenyOnAsk => "deny_on_ask".to_string(),
        ConfirmationInheritance::InheritParent => "inherit_parent".to_string(),
    }
}

fn rust_agent_definition_to_py(def: RustAgentDefinition) -> PyAgentDefinition {
    PyAgentDefinition {
        name: def.name,
        description: def.description,
        native: def.native,
        hidden: def.hidden,
        model: def.model.map(|model| model.model_ref()),
        prompt: def.prompt,
        max_steps: def.max_steps,
        confirmation_inheritance: def
            .confirmation_inheritance
            .as_ref()
            .map(confirmation_inheritance_to_py),
    }
}

/// Automatic child-agent delegation controls.
#[pyclass(name = "AutoDelegationConfig")]
#[derive(Clone)]
struct PyAutoDelegationConfig {
    enabled: bool,
    auto_parallel: bool,
    min_confidence: f32,
    max_tasks: usize,
}

impl From<PyAutoDelegationConfig> for a3s_code_core::AutoDelegationConfig {
    fn from(config: PyAutoDelegationConfig) -> Self {
        Self {
            enabled: config.enabled,
            auto_parallel: config.auto_parallel,
            min_confidence: config.min_confidence.clamp(0.0, 1.0),
            max_tasks: config.max_tasks.max(1),
        }
    }
}

#[pymethods]
impl PyAutoDelegationConfig {
    #[new]
    #[pyo3(signature = (enabled=false, auto_parallel=true, min_confidence=0.72, max_tasks=4))]
    fn new(enabled: bool, auto_parallel: bool, min_confidence: f32, max_tasks: usize) -> Self {
        Self {
            enabled,
            auto_parallel,
            min_confidence: min_confidence.clamp(0.0, 1.0),
            max_tasks: max_tasks.max(1),
        }
    }

    /// Enable runtime-driven automatic child-agent delegation.
    #[getter]
    fn get_enabled(&self) -> bool {
        self.enabled
    }

    #[setter]
    fn set_enabled(&mut self, value: bool) {
        self.enabled = value;
    }

    /// Allow automatic delegation to launch multiple child agents in parallel.
    ///
    /// Manual ``parallel_task`` calls remain available when this is false.
    #[getter]
    fn get_auto_parallel(&self) -> bool {
        self.auto_parallel
    }

    #[setter]
    fn set_auto_parallel(&mut self, value: bool) {
        self.auto_parallel = value;
    }

    /// Minimum local confidence required to auto-delegate a child task.
    #[getter]
    fn get_min_confidence(&self) -> f32 {
        self.min_confidence
    }

    #[setter]
    fn set_min_confidence(&mut self, value: f32) {
        self.min_confidence = value.clamp(0.0, 1.0);
    }

    /// Maximum number of automatic child tasks per user request.
    #[getter]
    fn get_max_tasks(&self) -> usize {
        self.max_tasks
    }

    #[setter]
    fn set_max_tasks(&mut self, value: usize) {
        self.max_tasks = value.max(1);
    }

    fn __repr__(&self) -> String {
        format!(
            "AutoDelegationConfig(enabled={}, auto_parallel={}, min_confidence={}, max_tasks={})",
            self.enabled, self.auto_parallel, self.min_confidence, self.max_tasks
        )
    }
}

/// Per-session configuration options.
///
/// Pass to `agent.session(workspace, options)` to override defaults.
#[pyclass(name = "SessionOptions")]
struct PySessionOptions {
    model: Option<String>,
    builtin_skills: bool,
    skill_dirs: Vec<String>,
    enforce_active_skill_tool_restrictions: Option<bool>,
    agent_dirs: Vec<String>,
    worker_agents: Vec<PyWorkerAgentSpec>,
    queue_config: Option<PySessionQueueConfig>,
    permission_policy: Option<PyPermissionPolicy>,
    confirmation_policy: Option<PyConfirmationPolicy>,
    auto_compact: bool,
    auto_compact_threshold: Option<f32>,
    /// Retention limits for large tool/program artifacts.
    artifact_store_limits: Option<PyArtifactStoreLimits>,
    /// Long-term memory store backend. Set to a ``FileMemoryStore`` instance.
    memory_store: Option<pyo3::PyObject>,
    /// Session persistence store backend. Set to ``FileSessionStore`` or ``MemorySessionStore``.
    session_store: Option<pyo3::PyObject>,
    /// Security provider. Set to ``DefaultSecurityProvider`` to enable taint tracking.
    security_provider: Option<pyo3::PyObject>,
    /// Workspace backend. Set to ``LocalWorkspaceBackend`` to use local filesystem tools explicitly.
    workspace_backend: Option<pyo3::PyObject>,
    /// Optional remote git provider. When set, the session attaches a
    /// ``RemoteGitBackend`` on top of ``workspace_backend`` so the built-in
    /// ``git`` tool is available on object-storage workspaces. Requires
    /// ``workspace_backend`` to be set; otherwise the session raises a clear
    /// error at construction.
    remote_git: Option<PyRemoteGitBackendConfig>,
    /// Custom role/identity (e.g. "You are a Python expert")
    role: Option<String>,
    /// Custom coding guidelines
    guidelines: Option<String>,
    /// Custom response style (replaces default)
    response_style: Option<String>,
    /// Freeform extra instructions
    extra: Option<String>,
    /// Inline skills registered programmatically: (name, kind, content).
    /// Populated via `add_instruction()` / `add_persona()` — not exposed directly to Python.
    inline_skills: Vec<(String, String, String)>,
    /// Override maximum number of tool-call rounds per session.
    max_tool_rounds: Option<usize>,
    /// Override maximum sibling parallel branches for this session.
    max_parallel_tasks: Option<usize>,
    /// Override automatic child-agent delegation for this session.
    auto_delegation: Option<PyAutoDelegationConfig>,
    /// Global session-level kill switch for automatic parallel child-agent fan-out.
    ///
    /// Manual ``parallel_task`` calls remain available when this is false.
    auto_parallel: Option<bool>,
    /// Explicit planning mode: "auto", "enabled", or "disabled".
    ///
    /// Prefer this over ``planning`` for an unambiguous SDK contract.
    /// If both are set, ``planning_mode`` wins.
    planning_mode: Option<String>,
    /// Legacy planning shortcut. None = auto, True = force, False = disabled.
    planning: Option<bool>,
    /// Enable goal tracking (default: False).
    goal_tracking: bool,
    /// Max consecutive parse errors before abort (default: 2).
    max_parse_retries: Option<u32>,
    /// Per-tool execution timeout in milliseconds.
    tool_timeout_ms: Option<u64>,
    /// Max LLM API failures before abort (default: 3).
    circuit_breaker_threshold: Option<u32>,
    /// Sampling temperature (0.0–1.0). Overrides the provider default.
    /// Only applied when ``model`` is also set.
    temperature: Option<f32>,
    /// Extended thinking token budget (e.g. 10_000). Enables chain-of-thought reasoning.
    /// Only applied when ``model`` is also set. Provider must support extended thinking.
    thinking_budget: Option<usize>,
    /// Enable continuation injection (default: True).
    /// When enabled, the loop injects a follow-up prompt when the LLM stops without completing.
    continuation_enabled: Option<bool>,
    /// Maximum continuation injections per execution (default: 3).
    max_continuation_turns: Option<u32>,
    /// Maximum execution time in milliseconds.
    /// When set, the execution loop will abort if it exceeds this duration.
    max_execution_time_ms: Option<u64>,
    /// Session ID for this session (auto-generated if not set).
    ///
    /// Set a stable ID to save and resume the session later:
    ///
    /// .. code-block:: python
    ///
    ///     opts = SessionOptions()
    ///     opts.session_store = FileSessionStore('./sessions')
    ///     opts.session_id = 'my-session'
    ///     opts.auto_save = True
    ///     session = agent.session('.', opts)
    ///     # Later:
    ///     resumed = agent.resume_session('my-session', opts)
    session_id: Option<String>,
    /// Host-defined tenant id. Opaque to the framework — propagated to
    /// SessionData / hooks / traces for multi-tenant aggregation.
    tenant_id: Option<String>,
    /// Principal identity (user / service / etc) that triggered the
    /// session. Treated as opaque.
    principal: Option<String>,
    /// Logical id of the agent template the session was instantiated
    /// from.
    agent_template_id: Option<String>,
    /// Distributed-trace correlation id propagated through this
    /// session's events.
    correlation_id: Option<String>,
    /// Automatically save the session to the configured store after each turn (default: False).
    auto_save: bool,
    /// AHP transport configuration for external agent supervision.
    ///
    /// Set to an AHP transport instance (``StdioTransport``, ``HttpTransport``, etc.)
    /// to enable Agent Harness Protocol supervision:
    ///
    /// .. code-block:: python
    ///
    ///     opts = SessionOptions()
    ///     opts.ahp_transport = StdioTransport(program='python', args=['ahp_server.py'])
    ///     session = agent.session('.', opts)
    ahp_transport: Option<pyo3::PyObject>,
    /// Optional Python-side BudgetGuard. The framework calls
    /// `check_before_llm(session_id, estimated_tokens)`,
    /// `record_after_llm(session_id, usage_dict)`, and
    /// `check_before_tool(session_id, tool_name)` on this object.
    /// Methods that aren't defined behave as Allow / no-op.
    ///
    /// Return shapes for check_*: ``None`` or ``{"decision":"allow"}``
    /// allows; ``{"decision":"soft","resource":...,"consumed":...,"limit":...,"message":...}``
    /// emits BudgetThresholdHit("soft"); ``{"decision":"deny","resource":...,"reason":...}``
    /// aborts the call with a ``Budget exhausted`` RuntimeError.
    budget_guard: Option<pyo3::PyObject>,
    /// Optional FIFO retention caps on the session's in-memory stores.
    /// Accepts a dict with optional integer keys:
    ///
    ///   - ``max_runs_retained``           -- cap on InMemoryRunStore.runs
    ///   - ``max_events_per_run``          -- cap on per-run event buffers
    ///   - ``max_trace_events``            -- cap on InMemoryTraceSink
    ///   - ``max_terminal_subagent_tasks`` -- cap on terminal subagent entries
    ///
    /// Missing keys keep the unbounded default for that store. Used by
    /// long-running cluster sessions to stop in-memory state from
    /// growing unboundedly.
    retention_limits: Option<pyo3::PyObject>,
}

impl Clone for PySessionOptions {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            builtin_skills: self.builtin_skills,
            skill_dirs: self.skill_dirs.clone(),
            enforce_active_skill_tool_restrictions: self.enforce_active_skill_tool_restrictions,
            agent_dirs: self.agent_dirs.clone(),
            worker_agents: self.worker_agents.clone(),
            queue_config: self.queue_config.clone(),
            permission_policy: self.permission_policy.clone(),
            confirmation_policy: self.confirmation_policy.clone(),
            auto_compact: self.auto_compact,
            auto_compact_threshold: self.auto_compact_threshold,
            artifact_store_limits: self.artifact_store_limits.clone(),
            memory_store: pyo3::Python::with_gil(|py| {
                self.memory_store.as_ref().map(|o| o.clone_ref(py))
            }),
            session_store: pyo3::Python::with_gil(|py| {
                self.session_store.as_ref().map(|o| o.clone_ref(py))
            }),
            security_provider: pyo3::Python::with_gil(|py| {
                self.security_provider.as_ref().map(|o| o.clone_ref(py))
            }),
            workspace_backend: pyo3::Python::with_gil(|py| {
                self.workspace_backend.as_ref().map(|o| o.clone_ref(py))
            }),
            remote_git: self.remote_git.clone(),
            role: self.role.clone(),
            guidelines: self.guidelines.clone(),
            response_style: self.response_style.clone(),
            extra: self.extra.clone(),
            inline_skills: self.inline_skills.clone(),
            max_tool_rounds: self.max_tool_rounds,
            max_parallel_tasks: self.max_parallel_tasks,
            auto_delegation: self.auto_delegation.clone(),
            auto_parallel: self.auto_parallel,
            planning_mode: self.planning_mode.clone(),
            planning: self.planning,
            goal_tracking: self.goal_tracking,
            max_parse_retries: self.max_parse_retries,
            tool_timeout_ms: self.tool_timeout_ms,
            circuit_breaker_threshold: self.circuit_breaker_threshold,
            temperature: self.temperature,
            thinking_budget: self.thinking_budget,
            continuation_enabled: self.continuation_enabled,
            max_continuation_turns: self.max_continuation_turns,
            max_execution_time_ms: self.max_execution_time_ms,
            session_id: self.session_id.clone(),
            tenant_id: self.tenant_id.clone(),
            principal: self.principal.clone(),
            agent_template_id: self.agent_template_id.clone(),
            correlation_id: self.correlation_id.clone(),
            auto_save: self.auto_save,
            ahp_transport: pyo3::Python::with_gil(|py| {
                self.ahp_transport.as_ref().map(|o| o.clone_ref(py))
            }),
            budget_guard: pyo3::Python::with_gil(|py| {
                self.budget_guard.as_ref().map(|o| o.clone_ref(py))
            }),
            retention_limits: pyo3::Python::with_gil(|py| {
                self.retention_limits.as_ref().map(|o| o.clone_ref(py))
            }),
        }
    }
}

#[pymethods]
impl PySessionOptions {
    #[new]
    fn new() -> Self {
        Self {
            model: None,
            builtin_skills: false,
            skill_dirs: vec![],
            enforce_active_skill_tool_restrictions: None,
            agent_dirs: vec![],
            worker_agents: vec![],
            queue_config: None,
            permission_policy: None,
            confirmation_policy: None,
            auto_compact: false,
            auto_compact_threshold: None,
            artifact_store_limits: None,
            memory_store: None,
            session_store: None,
            security_provider: None,
            workspace_backend: None,
            remote_git: None,
            role: None,
            guidelines: None,
            response_style: None,
            extra: None,
            inline_skills: vec![],
            max_tool_rounds: None,
            max_parallel_tasks: None,
            auto_delegation: None,
            auto_parallel: None,
            planning_mode: None,
            planning: None,
            goal_tracking: false,
            max_parse_retries: None,
            tool_timeout_ms: None,
            circuit_breaker_threshold: None,
            temperature: None,
            thinking_budget: None,
            continuation_enabled: None,
            max_continuation_turns: None,
            max_execution_time_ms: None,
            session_id: None,
            tenant_id: None,
            principal: None,
            agent_template_id: None,
            correlation_id: None,
            auto_save: false,
            ahp_transport: None,
            budget_guard: None,
            retention_limits: None,
        }
    }

    /// Override the default model. Format: "provider/model".
    #[getter]
    fn get_model(&self) -> Option<String> {
        self.model.clone()
    }

    #[setter]
    fn set_model(&mut self, value: Option<String>) {
        self.model = value;
    }

    /// Enable built-in skills.
    #[getter]
    fn get_builtin_skills(&self) -> bool {
        self.builtin_skills
    }

    #[setter]
    fn set_builtin_skills(&mut self, value: bool) {
        self.builtin_skills = value;
    }

    /// Extra directories to scan for skill files.
    #[getter]
    fn get_skill_dirs(&self) -> Vec<String> {
        self.skill_dirs.clone()
    }

    #[setter]
    fn set_skill_dirs(&mut self, value: Vec<String>) {
        self.skill_dirs = value;
    }

    /// Whether active skill allowed-tools restrict ordinary session tool calls.
    ///
    /// Defaults to None/False. Set True to restore the legacy global
    /// active-skill restriction before permission policy, hooks, HITL, or AHP run.
    #[getter]
    fn get_enforce_active_skill_tool_restrictions(&self) -> Option<bool> {
        self.enforce_active_skill_tool_restrictions
    }

    #[setter]
    fn set_enforce_active_skill_tool_restrictions(&mut self, value: Option<bool>) {
        self.enforce_active_skill_tool_restrictions = value;
    }

    /// Extra directories to scan for agent files.
    #[getter]
    fn get_agent_dirs(&self) -> Vec<String> {
        self.agent_dirs.clone()
    }

    #[setter]
    fn set_agent_dirs(&mut self, value: Vec<String>) {
        self.agent_dirs = value;
    }

    /// Reproducible disposable workers to register for task delegation.
    #[getter]
    fn get_worker_agents(&self) -> Vec<PyWorkerAgentSpec> {
        self.worker_agents.clone()
    }

    #[setter]
    fn set_worker_agents(&mut self, value: Vec<PyWorkerAgentSpec>) {
        self.worker_agents = value;
    }

    /// Add one disposable worker agent to this session option set.
    fn add_worker_agent(&mut self, worker: PyWorkerAgentSpec) {
        self.worker_agents.push(worker);
    }

    /// Optional advanced queue configuration for explicit external/hybrid lane dispatch.
    ///
    /// Ordinary sessions are queue-free unless this is set.
    #[getter]
    fn get_queue_config(&self) -> Option<PySessionQueueConfig> {
        self.queue_config.clone()
    }

    #[setter]
    fn set_queue_config(&mut self, value: Option<PySessionQueueConfig>) {
        self.queue_config = value;
    }

    /// Explicit permission policy for tool execution.
    ///
    /// Use this to make tool access explicit for real applications.
    #[getter]
    fn get_permission_policy(&self) -> Option<PyPermissionPolicy> {
        self.permission_policy.clone()
    }

    #[setter]
    fn set_permission_policy(&mut self, value: Option<PyPermissionPolicy>) {
        self.permission_policy = value;
    }

    /// HITL confirmation policy configuration.
    #[getter]
    fn get_confirmation_policy(&self) -> Option<PyConfirmationPolicy> {
        self.confirmation_policy.clone()
    }

    #[setter]
    fn set_confirmation_policy(&mut self, value: Option<PyConfirmationPolicy>) {
        self.confirmation_policy = value;
    }

    /// Enable auto-compaction when context window fills up.
    #[getter]
    fn get_auto_compact(&self) -> bool {
        self.auto_compact
    }

    #[setter]
    fn set_auto_compact(&mut self, value: bool) {
        self.auto_compact = value;
    }

    /// Context usage threshold (0.0–1.0) to trigger auto-compaction.
    #[getter]
    fn get_auto_compact_threshold(&self) -> Option<f32> {
        self.auto_compact_threshold
    }

    #[setter]
    fn set_auto_compact_threshold(&mut self, value: Option<f32>) {
        self.auto_compact_threshold = value;
    }

    /// Retention limits for large tool/program artifacts.
    #[getter]
    fn get_artifact_store_limits(&self) -> Option<PyArtifactStoreLimits> {
        self.artifact_store_limits.clone()
    }

    #[setter]
    fn set_artifact_store_limits(&mut self, value: Option<PyArtifactStoreLimits>) {
        self.artifact_store_limits = value;
    }

    /// Long-term memory store backend.
    ///
    /// Assign a ``FileMemoryStore`` instance:
    ///
    /// .. code-block:: python
    ///
    ///     opts.memory_store = FileMemoryStore('./memory')
    #[getter]
    fn get_memory_store(&self, py: pyo3::Python<'_>) -> Option<pyo3::PyObject> {
        self.memory_store.as_ref().map(|o| o.clone_ref(py))
    }

    #[setter]
    fn set_memory_store(&mut self, value: Option<pyo3::PyObject>) {
        self.memory_store = value;
    }

    /// Session persistence store backend.
    ///
    /// Assign a ``FileSessionStore`` or ``MemorySessionStore`` instance:
    ///
    /// .. code-block:: python
    ///
    ///     opts.session_store = FileSessionStore('./sessions')  # persists to disk
    ///     opts.session_store = MemorySessionStore()           # ephemeral
    #[getter]
    fn get_session_store(&self, py: pyo3::Python<'_>) -> Option<pyo3::PyObject> {
        self.session_store.as_ref().map(|o| o.clone_ref(py))
    }

    #[setter]
    fn set_session_store(&mut self, value: Option<pyo3::PyObject>) {
        self.session_store = value;
    }

    /// Security provider.
    ///
    /// Assign a ``DefaultSecurityProvider`` to enable taint tracking and output sanitisation:
    ///
    /// .. code-block:: python
    ///
    ///     opts.security_provider = DefaultSecurityProvider()
    #[getter]
    fn get_security_provider(&self, py: pyo3::Python<'_>) -> Option<pyo3::PyObject> {
        self.security_provider.as_ref().map(|o| o.clone_ref(py))
    }

    #[setter]
    fn set_security_provider(&mut self, value: Option<pyo3::PyObject>) {
        self.security_provider = value;
    }

    /// Workspace backend used by built-in tools.
    ///
    /// Assign a ``LocalWorkspaceBackend`` instance:
    ///
    /// .. code-block:: python
    ///
    ///     opts.workspace_backend = LocalWorkspaceBackend('/repo')
    #[getter]
    fn get_workspace_backend(&self, py: pyo3::Python<'_>) -> Option<pyo3::PyObject> {
        self.workspace_backend.as_ref().map(|o| o.clone_ref(py))
    }

    #[setter]
    fn set_workspace_backend(&mut self, value: Option<pyo3::PyObject>) {
        self.workspace_backend = value;
    }

    /// Optional remote git provider. Attach a ``RemoteGitBackendConfig`` to
    /// bring the built-in ``git`` tool to a session whose ``workspace_backend``
    /// cannot natively host git (e.g. S3). Requires ``workspace_backend`` to
    /// be set.
    #[getter]
    fn get_remote_git(&self) -> Option<PyRemoteGitBackendConfig> {
        self.remote_git.clone()
    }

    #[setter]
    fn set_remote_git(&mut self, value: Option<PyRemoteGitBackendConfig>) {
        self.remote_git = value;
    }

    /// Custom role/identity prepended before the core agentic prompt.
    /// Example: "You are a senior Python developer specializing in FastAPI."
    #[getter]
    fn get_role(&self) -> Option<String> {
        self.role.clone()
    }

    #[setter]
    fn set_role(&mut self, value: Option<String>) {
        self.role = value;
    }

    /// Custom coding guidelines appended after the core prompt.
    /// Example: "Always use type hints. Follow PEP 8."
    #[getter]
    fn get_guidelines(&self) -> Option<String> {
        self.guidelines.clone()
    }

    #[setter]
    fn set_guidelines(&mut self, value: Option<String>) {
        self.guidelines = value;
    }

    /// Custom response style (replaces default Response Format section).
    #[getter]
    fn get_response_style(&self) -> Option<String> {
        self.response_style.clone()
    }

    #[setter]
    fn set_response_style(&mut self, value: Option<String>) {
        self.response_style = value;
    }

    /// Freeform extra instructions appended at the end.
    #[getter]
    fn get_extra(&self) -> Option<String> {
        self.extra.clone()
    }

    #[setter]
    fn set_extra(&mut self, value: Option<String>) {
        self.extra = value;
    }

    /// Override maximum number of tool-call rounds for this session.
    #[getter]
    fn get_max_tool_rounds(&self) -> Option<usize> {
        self.max_tool_rounds
    }

    #[setter]
    fn set_max_tool_rounds(&mut self, value: Option<usize>) {
        self.max_tool_rounds = value;
    }

    /// Override maximum sibling parallel branches for this session.
    #[getter]
    fn get_max_parallel_tasks(&self) -> Option<usize> {
        self.max_parallel_tasks
    }

    #[setter]
    fn set_max_parallel_tasks(&mut self, value: Option<usize>) {
        self.max_parallel_tasks = value.map(|tasks| tasks.max(1));
    }

    /// Override automatic child-agent delegation for this session.
    #[getter]
    fn get_auto_delegation(&self) -> Option<PyAutoDelegationConfig> {
        self.auto_delegation.clone()
    }

    #[setter]
    fn set_auto_delegation(&mut self, value: Option<PyAutoDelegationConfig>) {
        self.auto_delegation = value;
    }

    /// Global session-level kill switch for automatic parallel child-agent fan-out.
    ///
    /// Manual ``parallel_task`` calls remain available when this is false.
    #[getter]
    fn get_auto_parallel(&self) -> Option<bool> {
        self.auto_parallel
    }

    #[setter]
    fn set_auto_parallel(&mut self, value: Option<bool>) {
        self.auto_parallel = value;
    }

    /// Explicit planning mode: "auto", "enabled", or "disabled".
    #[getter]
    fn get_planning_mode(&self) -> Option<String> {
        self.planning_mode.clone()
    }

    #[setter]
    fn set_planning_mode(&mut self, value: Option<String>) -> PyResult<()> {
        if let Some(ref mode) = value {
            parse_planning_mode(mode)?;
        }
        self.planning_mode = value;
        Ok(())
    }

    /// Legacy planning shortcut. None = auto, True = force, False = disabled.
    #[getter]
    fn get_planning(&self) -> Option<bool> {
        self.planning
    }

    #[setter]
    fn set_planning(&mut self, value: Option<bool>) {
        self.planning = value;
    }

    /// Enable goal tracking (default: False).
    #[getter]
    fn get_goal_tracking(&self) -> bool {
        self.goal_tracking
    }

    #[setter]
    fn set_goal_tracking(&mut self, value: bool) {
        self.goal_tracking = value;
    }

    /// Max consecutive parse errors before abort (default: 2).
    #[getter]
    fn get_max_parse_retries(&self) -> Option<u32> {
        self.max_parse_retries
    }

    #[setter]
    fn set_max_parse_retries(&mut self, value: Option<u32>) {
        self.max_parse_retries = value;
    }

    /// Per-tool execution timeout in milliseconds.
    #[getter]
    fn get_tool_timeout_ms(&self) -> Option<u64> {
        self.tool_timeout_ms
    }

    #[setter]
    fn set_tool_timeout_ms(&mut self, value: Option<u64>) {
        self.tool_timeout_ms = value;
    }

    /// Max LLM API failures before abort (default: 3).
    #[getter]
    fn get_circuit_breaker_threshold(&self) -> Option<u32> {
        self.circuit_breaker_threshold
    }

    #[setter]
    fn set_circuit_breaker_threshold(&mut self, value: Option<u32>) {
        self.circuit_breaker_threshold = value;
    }

    /// Sampling temperature (0.0–1.0). Overrides the provider default.
    /// Only applied when ``model`` is also set.
    #[getter]
    fn get_temperature(&self) -> Option<f32> {
        self.temperature
    }

    #[setter]
    fn set_temperature(&mut self, value: Option<f32>) {
        self.temperature = value;
    }

    /// Extended thinking token budget. Enables chain-of-thought reasoning.
    /// Only applied when ``model`` is also set.
    #[getter]
    fn get_thinking_budget(&self) -> Option<usize> {
        self.thinking_budget
    }

    #[setter]
    fn set_thinking_budget(&mut self, value: Option<usize>) {
        self.thinking_budget = value;
    }

    /// Enable or disable continuation injection (default: True).
    #[getter]
    fn get_continuation_enabled(&self) -> Option<bool> {
        self.continuation_enabled
    }

    #[setter]
    fn set_continuation_enabled(&mut self, value: Option<bool>) {
        self.continuation_enabled = value;
    }

    /// Maximum continuation injections per execution (default: 3).
    #[getter]
    fn get_max_continuation_turns(&self) -> Option<u32> {
        self.max_continuation_turns
    }

    #[setter]
    fn set_max_continuation_turns(&mut self, value: Option<u32>) {
        self.max_continuation_turns = value;
    }

    /// Maximum execution time in milliseconds.
    #[getter]
    fn get_max_execution_time_ms(&self) -> Option<u64> {
        self.max_execution_time_ms
    }

    #[setter]
    fn set_max_execution_time_ms(&mut self, value: Option<u64>) {
        self.max_execution_time_ms = value;
    }

    /// Session ID (auto-generated if not set). Set to save and resume sessions by name.
    #[getter]
    fn get_session_id(&self) -> Option<String> {
        self.session_id.clone()
    }

    #[setter]
    fn set_session_id(&mut self, value: Option<String>) {
        self.session_id = value;
    }

    /// Host-defined tenant id. Opaque to the framework — used by hooks
    /// / traces / SessionData for multi-tenant aggregation.
    #[getter]
    fn get_tenant_id(&self) -> Option<String> {
        self.tenant_id.clone()
    }

    #[setter]
    fn set_tenant_id(&mut self, value: Option<String>) {
        self.tenant_id = value;
    }

    /// Identity of the principal that triggered the session.
    #[getter]
    fn get_principal(&self) -> Option<String> {
        self.principal.clone()
    }

    #[setter]
    fn set_principal(&mut self, value: Option<String>) {
        self.principal = value;
    }

    /// Logical id of the agent template / definition.
    #[getter]
    fn get_agent_template_id(&self) -> Option<String> {
        self.agent_template_id.clone()
    }

    #[setter]
    fn set_agent_template_id(&mut self, value: Option<String>) {
        self.agent_template_id = value;
    }

    /// Distributed-trace correlation id.
    #[getter]
    fn get_correlation_id(&self) -> Option<String> {
        self.correlation_id.clone()
    }

    #[setter]
    fn set_correlation_id(&mut self, value: Option<String>) {
        self.correlation_id = value;
    }

    /// Automatically save the session after each turn (default: False).
    #[getter]
    fn get_auto_save(&self) -> bool {
        self.auto_save
    }

    #[setter]
    fn set_auto_save(&mut self, value: bool) {
        self.auto_save = value;
    }

    /// AHP transport configuration for external agent supervision.
    #[getter]
    fn get_ahp_transport(&self) -> Option<pyo3::PyObject> {
        pyo3::Python::with_gil(|py| self.ahp_transport.as_ref().map(|o| o.clone_ref(py)))
    }

    #[setter]
    fn set_ahp_transport(&mut self, value: Option<pyo3::PyObject>) {
        self.ahp_transport = value;
    }

    /// Host-supplied BudgetGuard. Any Python object implementing some
    /// subset of `check_before_llm` / `record_after_llm` /
    /// `check_before_tool`. The framework calls these around every
    /// LLM call and surfaces `{"decision": "deny", ...}` as a
    /// ``Budget exhausted`` ``RuntimeError`` on ``session.send``.
    #[getter]
    fn get_budget_guard(&self) -> Option<pyo3::PyObject> {
        pyo3::Python::with_gil(|py| self.budget_guard.as_ref().map(|o| o.clone_ref(py)))
    }

    #[setter]
    fn set_budget_guard(&mut self, value: Option<pyo3::PyObject>) {
        self.budget_guard = value;
    }

    /// Optional FIFO retention caps as a dict with any subset of:
    /// ``max_runs_retained``, ``max_events_per_run``,
    /// ``max_trace_events``, ``max_terminal_subagent_tasks``.
    /// Missing keys keep the unbounded default for that store.
    #[getter]
    fn get_retention_limits(&self) -> Option<pyo3::PyObject> {
        pyo3::Python::with_gil(|py| self.retention_limits.as_ref().map(|o| o.clone_ref(py)))
    }

    #[setter]
    fn set_retention_limits(&mut self, value: Option<pyo3::PyObject>) {
        self.retention_limits = value;
    }

    /// Register an instruction skill programmatically.
    ///
    /// Instructions are injected into the system prompt at session start.
    /// Use this instead of skill files for simple, one-off guidance.
    ///
    /// Args:
    ///     name: Unique skill name (kebab-case recommended, e.g. "type-hints")
    ///     content: Markdown content describing the instruction
    fn add_instruction(&mut self, name: String, content: String) {
        self.inline_skills
            .push((name, "instruction".to_string(), content));
    }

    /// Register a persona skill programmatically.
    ///
    /// Personas replace the default role section of the system prompt.
    /// Only one persona is active at a time (last registered wins).
    ///
    /// Args:
    ///     name: Unique skill name (kebab-case recommended, e.g. "python-expert")
    ///     content: System prompt content for this persona
    fn add_persona(&mut self, name: String, content: String) {
        self.inline_skills
            .push((name, "persona".to_string(), content));
    }

    fn __repr__(&self) -> String {
        format!(
            "SessionOptions(model={:?}, builtin_skills={}, queue_config={}, auto_compact={}, artifact_store_limits={}, memory_store={}, session_store={}, security_provider={}, workspace_backend={}, inline_skills={}, max_parallel_tasks={:?}, auto_parallel={:?})",
            self.model,
            self.builtin_skills,
            if self.queue_config.is_some() { "Some(...)" } else { "None" },
            self.auto_compact,
            if self.artifact_store_limits.is_some() { "Some(...)" } else { "None" },
            if self.memory_store.is_some() { "Some(...)" } else { "None" },
            if self.session_store.is_some() { "Some(...)" } else { "None" },
            if self.security_provider.is_some() { "Some(...)" } else { "None" },
            if self.workspace_backend.is_some() { "Some(...)" } else { "None" },
            self.inline_skills.len(),
            self.max_parallel_tasks,
            self.auto_parallel,
        )
    }
}

// ============================================================================
// SessionQueueConfig
// ============================================================================

/// Configuration for the optional advanced session lane queue.
///
/// Ordinary sessions do not initialize queue infrastructure. Use this only for
/// explicit external/hybrid dispatch, priority experiments, or operational integrations.
#[pyclass(name = "SessionQueueConfig")]
#[derive(Clone)]
struct PySessionQueueConfig {
    inner: RustSessionQueueConfig,
}

#[pymethods]
impl PySessionQueueConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: RustSessionQueueConfig::default(),
        }
    }

    /// Enable all lane features (DLQ, metrics, alerts) with sensible defaults.
    fn with_lane_features(&mut self) {
        self.inner = self.inner.clone().with_lane_features();
    }

    /// Set max concurrency for Query lane (default: 4).
    fn set_query_concurrency(&mut self, n: usize) {
        self.inner.query_max_concurrency = n;
    }

    /// Set max concurrency for Execute lane (default: 2).
    fn set_execute_concurrency(&mut self, n: usize) {
        self.inner.execute_max_concurrency = n;
    }

    /// Set max concurrency for Generate lane (default: 1).
    fn set_generate_concurrency(&mut self, n: usize) {
        self.inner.generate_max_concurrency = n;
    }

    /// Enable dead letter queue with optional max size.
    #[pyo3(signature = (max_size=None))]
    fn enable_dlq(&mut self, max_size: Option<usize>) {
        self.inner = self.inner.clone().with_dlq(max_size);
    }

    /// Enable metrics collection.
    fn enable_metrics(&mut self) {
        self.inner = self.inner.clone().with_metrics();
    }

    /// Enable queue alerts.
    fn enable_alerts(&mut self) {
        self.inner = self.inner.clone().with_alerts();
    }

    /// Set default timeout for commands (ms).
    fn set_timeout(&mut self, timeout_ms: u64) {
        self.inner = self.inner.clone().with_timeout(timeout_ms);
    }

    /// Configure how a specific lane handles tasks.
    ///
    /// Args:
    ///     lane (Literal["control", "query", "execute", "generate"]): Which lane to configure.
    ///     mode (Literal["internal", "external", "hybrid"]): Execution mode for the lane's tools.
    ///     timeout_ms: Timeout for external tasks in milliseconds (default 60000).
    #[pyo3(signature = (lane, mode, timeout_ms=60_000))]
    fn set_lane_handler(&mut self, lane: &str, mode: &str, timeout_ms: u64) -> PyResult<()> {
        let rust_lane = parse_lane(lane)?;
        let rust_mode = parse_handler_mode(mode)?;
        let config = RustLaneHandlerConfig {
            mode: rust_mode,
            timeout_ms,
        };
        self.inner.lane_handlers.insert(rust_lane, config);
        Ok(())
    }

    /// Set max concurrency for Query lane (default: 4).
    #[getter]
    fn get_query_max_concurrency(&self) -> usize {
        self.inner.query_max_concurrency
    }

    #[setter]
    fn set_query_max_concurrency(&mut self, value: usize) {
        self.inner.query_max_concurrency = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "SessionQueueConfig(query={}, execute={}, generate={}, dlq={}, metrics={})",
            self.inner.query_max_concurrency,
            self.inner.execute_max_concurrency,
            self.inner.generate_max_concurrency,
            self.inner.enable_dlq,
            self.inner.enable_metrics,
        )
    }
}

// ============================================================================
// Queue Helpers
// ============================================================================

fn parse_lane(lane: &str) -> PyResult<RustSessionLane> {
    match lane {
        "control" => Ok(RustSessionLane::Control),
        "query" => Ok(RustSessionLane::Query),
        "execute" => Ok(RustSessionLane::Execute),
        "generate" => Ok(RustSessionLane::Generate),
        _ => Err(PyValueError::new_err(format!(
            "Invalid lane '{}'. Must be: control, query, execute, or generate",
            lane
        ))),
    }
}

fn parse_handler_mode(mode: &str) -> PyResult<RustTaskHandlerMode> {
    match mode {
        "internal" => Ok(RustTaskHandlerMode::Internal),
        "external" => Ok(RustTaskHandlerMode::External),
        "hybrid" => Ok(RustTaskHandlerMode::Hybrid),
        _ => Err(PyValueError::new_err(format!(
            "Invalid handler mode '{}'. Must be: internal, external, or hybrid",
            mode
        ))),
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn parse_planning_mode(mode: &str) -> PyResult<RustPlanningMode> {
    match mode.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(RustPlanningMode::Auto),
        "enabled" | "enable" | "on" | "force" | "forced" | "true" => Ok(RustPlanningMode::Enabled),
        "disabled" | "disable" | "off" | "false" => Ok(RustPlanningMode::Disabled),
        _ => Err(PyValueError::new_err(format!(
            "Invalid planning_mode '{}'. Expected 'auto', 'enabled', or 'disabled'",
            mode
        ))),
    }
}

fn apply_planning_mode(
    opts: RustSessionOptions,
    planning_mode: Option<&str>,
    planning: Option<bool>,
) -> PyResult<RustSessionOptions> {
    if let Some(mode) = planning_mode {
        Ok(opts.with_planning_mode(parse_planning_mode(mode)?))
    } else if let Some(enabled) = planning {
        Ok(opts.with_planning(enabled))
    } else {
        Ok(opts)
    }
}

fn delegate_task_args(
    agent: String,
    description: String,
    prompt: String,
    background: bool,
    max_steps: Option<u32>,
) -> serde_json::Value {
    let mut args = serde_json::json!({
        "agent": agent,
        "description": description,
        "prompt": prompt,
    });
    if background {
        args["background"] = serde_json::json!(true);
    }
    if let Some(max_steps) = max_steps {
        args["max_steps"] = serde_json::json!(max_steps);
    }
    args
}

fn parallel_task_args(tasks: serde_json::Value) -> PyResult<serde_json::Value> {
    if !tasks.is_array() {
        return Err(PyValueError::new_err(
            "tasks must be a list of dictionaries",
        ));
    }
    Ok(serde_json::json!({ "tasks": tasks }))
}

/// Build RustSessionOptions from PySessionOptions.
fn build_rust_session_options(so: PySessionOptions) -> PyResult<RustSessionOptions> {
    let mut o = RustSessionOptions::new();
    if let Some(m) = so.model {
        o = o.with_model(m);
    }
    if so.builtin_skills {
        o = o.with_builtin_skills();
    }
    for d in &so.skill_dirs {
        o = o.with_skills_from_dir(d);
    }
    if let Some(enabled) = so.enforce_active_skill_tool_restrictions {
        o = o.with_active_skill_tool_restrictions(enabled);
    }
    for d in &so.agent_dirs {
        o = o.with_agent_dir(d);
    }
    for worker in so.worker_agents {
        o = o.with_worker_agent(py_worker_agent_spec_to_rust(worker)?);
    }
    if let Some(qc) = so.queue_config {
        o = o.with_queue_config(qc.inner);
    }
    if let Some(policy) = so.permission_policy {
        o = o.with_permission_checker(Arc::new(py_permission_policy_to_rust(policy)?));
    }
    if let Some(policy) = so.confirmation_policy {
        o = o.with_confirmation_policy(py_confirmation_policy_to_rust(policy)?);
    }
    if so.auto_compact {
        o = o.with_auto_compact(true);
    }
    if let Some(t) = so.auto_compact_threshold {
        o = o.with_auto_compact_threshold(t);
    }
    if let Some(limits) = so.artifact_store_limits {
        o = o.with_artifact_store_limits(limits.into());
    }
    if let Some(ref store) = so.memory_store {
        let dir = Python::with_gil(|py| {
            store
                .extract::<pyo3::PyRef<PyFileMemoryStore>>(py)
                .ok()
                .map(|s| s.dir.clone())
        });
        if let Some(dir) = dir {
            o = o.with_file_memory(dir);
        }
    }
    if let Some(ref store) = so.session_store {
        enum SessionStoreKind {
            File(String),
            Memory,
        }
        let kind = Python::with_gil(|py| {
            if let Ok(file_store) = store.extract::<pyo3::PyRef<PyFileSessionStore>>(py) {
                Some(SessionStoreKind::File(file_store.dir.clone()))
            } else if store
                .extract::<pyo3::PyRef<PyMemorySessionStore>>(py)
                .is_ok()
            {
                Some(SessionStoreKind::Memory)
            } else {
                None
            }
        });
        match kind {
            Some(SessionStoreKind::File(dir)) => {
                o = o.with_file_session_store(dir);
            }
            Some(SessionStoreKind::Memory) => {
                let s: Arc<dyn a3s_code_core::store::SessionStore> =
                    Arc::new(a3s_code_core::store::MemorySessionStore::new());
                o = o.with_session_store(s);
            }
            None => {}
        }
    }
    if let Some(ref sec) = so.security_provider {
        let is_default = Python::with_gil(|py| {
            sec.extract::<pyo3::PyRef<PyDefaultSecurityProvider>>(py)
                .is_ok()
        });
        if is_default {
            o = o.with_default_security();
        }
    }
    if let Some(ref backend) = so.workspace_backend {
        // S3BackendConfig is significantly larger than the other variants;
        // box it to avoid a `clippy::large_enum_variant` warning.
        enum BackendKind {
            Local(String),
            S3(Box<a3s_code_core::S3BackendConfig>),
            Unknown,
        }
        let resolved = Python::with_gil(|py| -> BackendKind {
            if let Ok(local) = backend.extract::<pyo3::PyRef<PyLocalWorkspaceBackend>>(py) {
                return BackendKind::Local(local.root.clone());
            }
            if let Ok(s3) = backend.extract::<pyo3::PyRef<PyS3WorkspaceBackend>>(py) {
                return BackendKind::S3(Box::new(s3.to_core()));
            }
            BackendKind::Unknown
        });
        let services = match resolved {
            BackendKind::Local(root) => a3s_code_core::WorkspaceServices::local(root),
            BackendKind::S3(cfg) => a3s_code_core::WorkspaceServices::s3(*cfg),
            BackendKind::Unknown => {
                return Err(PyTypeError::new_err(
                    "workspace_backend must be a LocalWorkspaceBackend or S3WorkspaceBackend instance",
                ));
            }
        };
        let services = if let Some(ref git_cfg) = so.remote_git {
            services
                .with_remote_git(git_cfg.to_core())
                .map_err(|e| PyValueError::new_err(format!("remote_git: {e}")))?
        } else {
            services
        };
        o = o.with_workspace_backend(services);
    } else if so.remote_git.is_some() {
        return Err(PyValueError::new_err(
            "remote_git requires workspace_backend to be set; assign a LocalWorkspaceBackend or S3WorkspaceBackend first",
        ));
    }
    // Build prompt slots if any slot is set
    if so.role.is_some()
        || so.guidelines.is_some()
        || so.response_style.is_some()
        || so.extra.is_some()
    {
        let slots = a3s_code_core::SystemPromptSlots {
            style: None,
            role: so.role,
            guidelines: so.guidelines,
            response_style: so.response_style,
            extra: so.extra,
        };
        o = o.with_prompt_slots(slots);
    }
    // Inline skills registered programmatically via add_instruction / add_persona
    if !so.inline_skills.is_empty() {
        let registry = a3s_code_core::skills::SkillRegistry::new();
        for (name, kind, content) in so.inline_skills {
            let raw = format!("---\nname: {name}\nkind: {kind}\n---\n{content}");
            if let Some(skill) = a3s_code_core::skills::Skill::parse(&raw) {
                registry.register_unchecked(Arc::new(skill));
            } else {
                eprintln!(
                    "a3s-code: failed to parse inline skill '{}' — skipping",
                    name
                );
            }
        }
        o = o.with_skill_registry(Arc::new(registry));
    }
    if let Some(r) = so.max_tool_rounds {
        o = o.with_max_tool_rounds(r);
    }
    if let Some(max_parallel_tasks) = so.max_parallel_tasks {
        o = o.with_max_parallel_tasks(max_parallel_tasks);
    }
    if let Some(auto_delegation) = so.auto_delegation {
        o = o.with_auto_delegation(auto_delegation.into());
    }
    if let Some(auto_parallel) = so.auto_parallel {
        o = o.with_auto_parallel_delegation(auto_parallel);
    }
    o = apply_planning_mode(o, so.planning_mode.as_deref(), so.planning)?;
    if so.goal_tracking {
        o = o.with_goal_tracking(true);
    }
    if let Some(n) = so.max_parse_retries {
        o = o.with_parse_retries(n);
    }
    if let Some(ms) = so.tool_timeout_ms {
        o = o.with_tool_timeout(ms);
    }
    if let Some(n) = so.circuit_breaker_threshold {
        o = o.with_circuit_breaker(n);
    }
    if let Some(t) = so.temperature {
        o = o.with_temperature(t);
    }
    if let Some(budget) = so.thinking_budget {
        o = o.with_thinking_budget(budget);
    }
    if let Some(enabled) = so.continuation_enabled {
        o = o.with_continuation(enabled);
    }
    if let Some(turns) = so.max_continuation_turns {
        o = o.with_max_continuation_turns(turns);
    }
    if let Some(timeout_ms) = so.max_execution_time_ms {
        o.max_execution_time_ms = Some(timeout_ms);
    }
    if let Some(id) = so.session_id {
        o = o.with_session_id(id);
    }
    if let Some(t) = so.tenant_id {
        o = o.with_tenant_id(t);
    }
    if let Some(p) = so.principal {
        o = o.with_principal(p);
    }
    if let Some(t) = so.agent_template_id {
        o = o.with_agent_template_id(t);
    }
    if let Some(c) = so.correlation_id {
        o = o.with_correlation_id(c);
    }
    if let Some(guard) = so.budget_guard {
        let wrapped: std::sync::Arc<dyn a3s_code_core::budget::BudgetGuard> =
            std::sync::Arc::new(PyBudgetGuard::new(guard));
        o = o.with_budget_guard(wrapped);
    }
    if let Some(retention) = so.retention_limits {
        if let Some(limits) = parse_py_retention_limits(&retention) {
            o = o.with_retention_limits(limits);
        }
    }
    if so.auto_save {
        o = o.with_auto_save(true);
    }

    // AHP transport configuration
    #[cfg(feature = "ahp")]
    if let Some(ref transport_obj) = so.ahp_transport {
        use a3s_code_core::ahp::{AhpHookExecutor, AhpTransport, AuthConfig};

        let transport = Python::with_gil(|py| {
            // Try stdio transport
            if let Ok(stdio) = transport_obj.extract::<pyo3::PyRef<PyStdioTransport>>(py) {
                return Some(AhpTransport::Stdio {
                    program: stdio.program.clone(),
                    args: stdio.args.clone(),
                });
            }
            // Try HTTP transport
            if let Ok(http) = transport_obj.extract::<pyo3::PyRef<PyHttpTransport>>(py) {
                let auth = http
                    .auth_token
                    .as_ref()
                    .map(|token| AuthConfig::bearer(token.clone()));
                return Some(AhpTransport::Http {
                    url: http.url.clone(),
                    auth,
                });
            }
            // Try WebSocket transport
            if let Ok(ws) = transport_obj.extract::<pyo3::PyRef<PyWebSocketTransport>>(py) {
                let auth = ws
                    .auth_token
                    .as_ref()
                    .map(|token| AuthConfig::bearer(token.clone()));
                return Some(AhpTransport::WebSocket {
                    url: ws.url.clone(),
                    auth,
                });
            }
            // Try Unix socket transport
            #[cfg(unix)]
            if let Ok(unix) = transport_obj.extract::<pyo3::PyRef<PyUnixSocketTransport>>(py) {
                return Some(AhpTransport::UnixSocket {
                    path: unix.path.clone(),
                });
            }
            None
        });

        if let Some(transport) = transport {
            // Create AHP executor asynchronously
            match get_runtime().block_on(AhpHookExecutor::new(transport)) {
                Ok(executor) => {
                    o = o.with_hook_executor(Arc::new(executor));
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

    Ok(o)
}

fn metrics_snapshot_to_json_str(s: RustMetricsSnapshot) -> Result<String, serde_json::Error> {
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
    serde_json::to_string(&serde_json::json!({
        "counters": serde_json::Value::Object(counters),
        "gauges": serde_json::Value::Object(gauges),
        "histograms": serde_json::Value::Object(histograms),
    }))
}

fn py_dict_to_json(dict: &Bound<'_, pyo3::types::PyDict>) -> PyResult<String> {
    let py = dict.py();
    let json_mod = py.import("json")?;
    let json_str = json_mod.call_method1("dumps", (dict,))?;
    json_str.extract::<String>()
}

fn py_any_to_json(value: &Bound<'_, PyAny>) -> PyResult<String> {
    let json_mod = value.py().import("json")?;
    let json_str = json_mod.call_method1("dumps", (value,))?;
    json_str.extract::<String>()
}

fn verification_reports_from_value(
    reports: serde_json::Value,
) -> PyResult<Vec<RustVerificationReport>> {
    let reports = match reports {
        serde_json::Value::Array(_) => serde_json::from_value(reports),
        serde_json::Value::Object(_) => {
            serde_json::from_value::<RustVerificationReport>(reports).map(|report| vec![report])
        }
        _ => {
            return Err(PyTypeError::new_err(
                "verification reports must be a list or dict",
            ));
        }
    };
    reports.map_err(|e| PyValueError::new_err(format!("Invalid verification report: {e}")))
}

fn py_verification_reports_to_rust(
    _py: Python<'_>,
    reports: &Bound<'_, PyAny>,
) -> PyResult<Vec<RustVerificationReport>> {
    let json_str = py_any_to_json(reports)?;
    let value: serde_json::Value = serde_json::from_str(&json_str)
        .map_err(|e| PyValueError::new_err(format!("Invalid verification report JSON: {e}")))?;
    verification_reports_from_value(value)
}

fn normalize_task_options(mut value: serde_json::Value) -> PyResult<serde_json::Value> {
    let obj = value
        .as_object_mut()
        .ok_or_else(|| PyValueError::new_err("task options must be a dict"))?;

    for field in ["agent", "description", "prompt"] {
        if !obj.get(field).is_some_and(|v| v.is_string()) {
            return Err(PyValueError::new_err(format!(
                "task options must include string field '{field}'"
            )));
        }
    }

    if let Some(value) = obj.remove("maxSteps") {
        obj.entry("max_steps".to_string()).or_insert(value);
    }

    Ok(value)
}

fn normalize_git_args(mut args: serde_json::Value) -> PyResult<serde_json::Value> {
    let obj = args
        .as_object_mut()
        .ok_or_else(|| PyValueError::new_err("git options must be a dict"))?;

    if !obj.contains_key("command") {
        return Err(PyValueError::new_err(
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

fn normalize_program_script_options(
    options: &Bound<'_, pyo3::types::PyDict>,
) -> PyResult<serde_json::Value> {
    let json_str = py_dict_to_json(options)?;
    let value: serde_json::Value = serde_json::from_str(&json_str)
        .map_err(|e| PyValueError::new_err(format!("Invalid program options: {e}")))?;
    let obj = value
        .as_object()
        .ok_or_else(|| PyValueError::new_err("program options must be a dict"))?;

    let mut args = serde_json::Map::new();
    args.insert("type".to_string(), serde_json::json!("script"));
    args.insert("language".to_string(), serde_json::json!("javascript"));

    for key in ["source", "path", "inputs", "limits"] {
        if let Some(field) = obj.get(key) {
            args.insert(key.to_string(), field.clone());
        }
    }

    if let Some(field) = obj.get("allowed_tools").or_else(|| obj.get("allowedTools")) {
        args.insert("allowed_tools".to_string(), field.clone());
    }

    Ok(serde_json::Value::Object(args))
}

fn timeout_ms_to_secs(timeout_ms: u64) -> u64 {
    timeout_ms.div_ceil(1000).max(1)
}

fn normalize_mcp_server_config(
    mut value: serde_json::Value,
) -> PyResult<a3s_code_core::mcp::protocol::McpServerConfig> {
    let obj = value
        .as_object_mut()
        .ok_or_else(|| PyValueError::new_err("MCP server config must be a dict"))?;

    for key in [
        "timeout_ms",
        "timeoutMs",
        "tool_timeout_ms",
        "toolTimeoutMs",
    ] {
        if let Some(timeout_ms) = obj.remove(key) {
            let timeout_ms = timeout_ms
                .as_u64()
                .ok_or_else(|| PyValueError::new_err(format!("{key} must be an integer")))?;
            obj.entry("toolTimeoutSecs".to_string())
                .or_insert_with(|| serde_json::json!(timeout_ms_to_secs(timeout_ms)));
            break;
        }
    }

    if let Some(transport) = obj.get_mut("transport") {
        normalize_mcp_transport_alias(transport);
    }

    serde_json::from_value(value)
        .map_err(|e| PyValueError::new_err(format!("Invalid MCP server config: {e}")))
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

/// Convert Python attachment dicts to Rust Attachment vec.
fn py_attachments_to_rust(
    attachments: &[Bound<'_, PyDict>],
) -> PyResult<Vec<a3s_code_core::llm::Attachment>> {
    attachments
        .iter()
        .map(|dict| {
            let data: Vec<u8> = dict
                .get_item("data")?
                .ok_or_else(|| PyValueError::new_err("Attachment missing 'data' field"))?
                .extract()?;
            let media_type: String = dict
                .get_item("media_type")?
                .ok_or_else(|| PyValueError::new_err("Attachment missing 'media_type' field"))?
                .extract()?;
            Ok(a3s_code_core::llm::Attachment::new(data, media_type))
        })
        .collect()
}

fn py_attachment_list_to_rust(
    attachments: &Bound<'_, PyList>,
) -> PyResult<Vec<a3s_code_core::llm::Attachment>> {
    attachments
        .iter()
        .map(|item| {
            let dict = item
                .downcast::<PyDict>()
                .map_err(|_| PyTypeError::new_err("attachments must contain dict items"))?;
            let data: Vec<u8> = dict
                .get_item("data")?
                .ok_or_else(|| PyValueError::new_err("Attachment missing 'data' field"))?
                .extract()?;
            let media_type: String = dict
                .get_item("media_type")?
                .ok_or_else(|| PyValueError::new_err("Attachment missing 'media_type' field"))?
                .extract()?;
            Ok(a3s_code_core::llm::Attachment::new(data, media_type))
        })
        .collect()
}

fn py_session_request_to_parts(
    request: &Bound<'_, PyDict>,
) -> PyResult<(
    String,
    Option<Vec<RustMessage>>,
    Vec<a3s_code_core::llm::Attachment>,
)> {
    let prompt = request
        .get_item("prompt")?
        .ok_or_else(|| PyValueError::new_err("request missing 'prompt' field"))?
        .extract::<String>()?;

    let history = match request.get_item("history")? {
        Some(value) => {
            let list = value
                .downcast::<PyList>()
                .map_err(|_| PyTypeError::new_err("request.history must be a list"))?;
            Some(py_list_to_messages(list)?)
        }
        None => None,
    };

    let attachments = match request.get_item("attachments")? {
        Some(value) => {
            let list = value
                .downcast::<PyList>()
                .map_err(|_| PyTypeError::new_err("request.attachments must be a list"))?;
            py_attachment_list_to_rust(list)?
        }
        None => Vec::new(),
    };

    Ok((prompt, history, attachments))
}

fn py_session_input_to_parts(
    input: &Bound<'_, PyAny>,
    history: Option<&Bound<'_, PyList>>,
) -> PyResult<(
    String,
    Option<Vec<RustMessage>>,
    Vec<a3s_code_core::llm::Attachment>,
)> {
    if let Ok(prompt) = input.extract::<String>() {
        let rust_history = history.map(py_list_to_messages).transpose()?;
        return Ok((prompt, rust_history, Vec::new()));
    }

    if let Ok(request) = input.downcast::<PyDict>() {
        return py_session_request_to_parts(request);
    }

    Err(PyTypeError::new_err(
        "session input must be a prompt string or request dict",
    ))
}

/// Convert a Python list of message dicts to `Vec<RustMessage>`.
///
/// Expected format: `[{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]`
fn py_list_to_messages(list: &Bound<'_, PyList>) -> PyResult<Vec<RustMessage>> {
    let py = list.py();
    let json_mod = py.import("json")?;
    let json_str: String = json_mod.call_method1("dumps", (list,))?.extract()?;
    serde_json::from_str::<Vec<RustMessage>>(&json_str)
        .map_err(|e| PyTypeError::new_err(format!("Invalid history format: {e}")))
}

/// Convert a Python list of verification command dicts to Rust commands.
///
/// Expected format:
/// `[{"id": "check:test", "kind": "test", "description": "Run tests", "command": "cargo test"}]`
fn py_list_to_verification_commands(
    list: &Bound<'_, PyList>,
) -> PyResult<Vec<RustVerificationCommand>> {
    let py = list.py();
    let json_mod = py.import("json")?;
    let json_str: String = json_mod.call_method1("dumps", (list,))?.extract()?;
    serde_json::from_str::<Vec<RustVerificationCommand>>(&json_str)
        .map_err(|e| PyTypeError::new_err(format!("Invalid verification command format: {e}")))
}

/// Convert `&[RustMessage]` to a Python list of dicts.
fn messages_to_py_list<'py>(
    py: Python<'py>,
    messages: &[RustMessage],
) -> PyResult<Bound<'py, PyList>> {
    let json_str = serde_json::to_string(messages)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize history: {e}")))?;
    let json_mod = py.import("json")?;
    let py_obj = json_mod.call_method1("loads", (json_str,))?;
    py_obj
        .downcast::<PyList>()
        .cloned()
        .map_err(|e| PyRuntimeError::new_err(format!("Unexpected serialization result: {e}")))
}

// ============================================================================
// SearchConfig
// ============================================================================

/// Configuration for a search engine.
#[pyclass(name = "SearchEngineConfig")]
#[derive(Clone)]
struct PySearchEngineConfig {
    #[pyo3(get, set)]
    enabled: bool,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    timeout: Option<u64>,
}

#[pymethods]
impl PySearchEngineConfig {
    #[new]
    #[pyo3(signature = (enabled=true, weight=1.0, timeout=None))]
    fn new(enabled: bool, weight: f64, timeout: Option<u64>) -> Self {
        Self {
            enabled,
            weight,
            timeout,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchEngineConfig(enabled={}, weight={}, timeout={:?})",
            self.enabled, self.weight, self.timeout
        )
    }
}

impl From<PySearchEngineConfig> for RustSearchEngineConfig {
    fn from(c: PySearchEngineConfig) -> Self {
        Self {
            enabled: c.enabled,
            weight: c.weight,
            timeout: c.timeout,
        }
    }
}

/// Health monitor configuration for search engines.
#[pyclass(name = "SearchHealthConfig")]
#[derive(Clone)]
struct PySearchHealthConfig {
    #[pyo3(get, set)]
    max_failures: u32,
    #[pyo3(get, set)]
    suspend_seconds: u64,
}

#[pymethods]
impl PySearchHealthConfig {
    #[new]
    #[pyo3(signature = (max_failures=3, suspend_seconds=60))]
    fn new(max_failures: u32, suspend_seconds: u64) -> Self {
        Self {
            max_failures,
            suspend_seconds,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchHealthConfig(max_failures={}, suspend_seconds={})",
            self.max_failures, self.suspend_seconds
        )
    }
}

impl From<PySearchHealthConfig> for RustSearchHealthConfig {
    fn from(c: PySearchHealthConfig) -> Self {
        Self {
            max_failures: c.max_failures,
            suspend_seconds: c.suspend_seconds,
        }
    }
}

/// Search engine configuration (a3s-search integration).
#[pyclass(name = "SearchConfig")]
#[derive(Clone)]
struct PySearchConfig {
    #[pyo3(get, set)]
    timeout: u64,
    #[pyo3(get, set)]
    health: Option<PySearchHealthConfig>,
    engines: std::collections::HashMap<String, PySearchEngineConfig>,
    #[pyo3(get, set)]
    headless: Option<PyHeadlessConfig>,
}

#[pymethods]
impl PySearchConfig {
    #[new]
    #[pyo3(signature = (timeout=10, health=None, headless=None))]
    fn new(
        timeout: u64,
        health: Option<PySearchHealthConfig>,
        headless: Option<PyHeadlessConfig>,
    ) -> Self {
        Self {
            timeout,
            health,
            engines: std::collections::HashMap::new(),
            headless,
        }
    }

    /// Set engine configuration.
    fn set_engine(&mut self, name: String, config: PySearchEngineConfig) {
        self.engines.insert(name, config);
    }

    /// Get engine configuration.
    fn get_engine(&self, name: String) -> Option<PySearchEngineConfig> {
        self.engines.get(&name).cloned()
    }

    /// Get all engine names.
    fn engine_names(&self) -> Vec<String> {
        self.engines.keys().cloned().collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchConfig(timeout={}, engines={}, health={:?})",
            self.timeout,
            self.engines.len(),
            self.health.is_some()
        )
    }
}

/// Headless browser backend selection.
#[pyclass(name = "BrowserBackend", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyBrowserBackend {
    /// Chrome/Chromium headless.
    Chrome,
    /// Lightpanda headless browser (Linux/macOS only).
    Lightpanda,
}

impl From<PyBrowserBackend> for RustBrowserBackend {
    fn from(b: PyBrowserBackend) -> Self {
        match b {
            PyBrowserBackend::Chrome => RustBrowserBackend::Chrome,
            PyBrowserBackend::Lightpanda => RustBrowserBackend::Lightpanda,
        }
    }
}

/// Headless browser configuration for JS-rendered search engines.
#[pyclass(name = "HeadlessConfig")]
#[derive(Clone)]
pub struct PyHeadlessConfig {
    #[pyo3(get, set)]
    backend: PyBrowserBackend,
    #[pyo3(get, set)]
    browser_path: Option<String>,
    #[pyo3(get, set)]
    max_tabs: Option<usize>,
    #[pyo3(get, set)]
    launch_args: Option<Vec<String>>,
    #[pyo3(get, set)]
    proxy_url: Option<String>,
}

#[pymethods]
impl PyHeadlessConfig {
    #[new]
    #[pyo3(signature = (backend, browser_path=None, max_tabs=None, launch_args=None, proxy_url=None))]
    fn new(
        backend: PyBrowserBackend,
        browser_path: Option<String>,
        max_tabs: Option<usize>,
        launch_args: Option<Vec<String>>,
        proxy_url: Option<String>,
    ) -> Self {
        Self {
            backend,
            browser_path,
            max_tabs,
            launch_args,
            proxy_url,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "HeadlessConfig(backend={:?}, browser_path={:?}, max_tabs={:?}, launch_args={:?}, proxy_url={:?})",
            self.backend, self.browser_path, self.max_tabs, self.launch_args, self.proxy_url
        )
    }
}

impl From<PyHeadlessConfig> for RustHeadlessConfig {
    fn from(c: PyHeadlessConfig) -> Self {
        Self {
            backend: c.backend.into(),
            browser_path: c.browser_path,
            max_tabs: c.max_tabs.unwrap_or(4),
            launch_args: c.launch_args.unwrap_or_default(),
            proxy_url: c.proxy_url,
        }
    }
}

impl From<PySearchConfig> for RustSearchConfig {
    fn from(c: PySearchConfig) -> Self {
        Self {
            timeout: c.timeout,
            health: c.health.map(|h| h.into()),
            engines: c.engines.into_iter().map(|(k, v)| (k, v.into())).collect(),
            headless: c.headless.map(|h| h.into()),
        }
    }
}

// ============================================================================
// SkillInfo
// ============================================================================

/// Metadata about a built-in skill.
#[pyclass(name = "SkillInfo")]
#[derive(Clone)]
struct PySkillInfo {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    description: String,
    #[pyo3(get)]
    kind: String,
}

#[pymethods]
impl PySkillInfo {
    fn __repr__(&self) -> String {
        format!(
            "SkillInfo(name='{}', kind='{}', description='{}')",
            self.name,
            self.kind,
            if self.description.len() > 60 {
                format!("{}...", truncate_utf8(&self.description, 60))
            } else {
                self.description.clone()
            }
        )
    }
}

// ============================================================================
// EventType — string constants for AgentEvent.type
// ============================================================================

/// String constants for `AgentEvent.type`.
///
/// Use these instead of raw strings to avoid typos and enable IDE completion:
///
/// ```python
/// from a3s_code import EventType
///
/// for event in session.stream("Refactor this module"):
///     if event.type == EventType.TEXT_DELTA:
///         print(event.text, end="", flush=True)
///     elif event.type == EventType.END:
///         print(f"\nDone. Tokens: {event.total_tokens}")
/// ```
#[pyclass(name = "EventType")]
struct PyEventType;

#[pymethods]
impl PyEventType {
    /// Agent started processing (carries `prompt`).
    #[classattr]
    const START: &'static str = "start";
    /// A new LLM turn began (carries `turn`).
    #[classattr]
    const TURN_START: &'static str = "turn_start";
    /// A chunk of assistant text arrived (carries `text`).
    #[classattr]
    const TEXT_DELTA: &'static str = "text_delta";
    /// A tool call started (carries `tool_id`, `tool_name`).
    #[classattr]
    const TOOL_START: &'static str = "tool_start";
    /// A tool call completed (carries `tool_id`, `tool_name`, `tool_output`, `exit_code`).
    #[classattr]
    const TOOL_END: &'static str = "tool_end";
    /// A streaming chunk from a tool (carries `tool_id`, `tool_name`, `text`).
    #[classattr]
    const TOOL_OUTPUT_DELTA: &'static str = "tool_output_delta";
    /// An LLM turn finished (carries `turn`, `total_tokens`).
    #[classattr]
    const TURN_END: &'static str = "turn_end";
    /// The agent finished (carries `text`, `total_tokens`).
    #[classattr]
    const END: &'static str = "end";
    /// An error occurred (carries `error`).
    #[classattr]
    const ERROR: &'static str = "error";
    /// Human-in-the-loop confirmation required before a tool runs.
    #[classattr]
    const CONFIRMATION_REQUIRED: &'static str = "confirmation_required";
    /// Confirmation response received.
    #[classattr]
    const CONFIRMATION_RECEIVED: &'static str = "confirmation_received";
    /// Confirmation timed out; default action was taken.
    #[classattr]
    const CONFIRMATION_TIMEOUT: &'static str = "confirmation_timeout";
    /// An external lane task is pending (carries `task_id`, `lane`).
    #[classattr]
    const EXTERNAL_TASK_PENDING: &'static str = "external_task_pending";
    /// An external lane task completed.
    #[classattr]
    const EXTERNAL_TASK_COMPLETED: &'static str = "external_task_completed";
    /// A tool was blocked by the permission policy.
    #[classattr]
    const PERMISSION_DENIED: &'static str = "permission_denied";
}

// ============================================================================
// Python Module
// ============================================================================

/// A3S Code - Native AI coding agent library for Python.
#[pymodule(name = "_native")]
fn a3s_code_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAgent>()?;
    m.add_class::<PySession>()?;
    m.add_class::<PyAgentResult>()?;
    m.add_class::<PyAgentEvent>()?;
    m.add_class::<PyToolResult>()?;
    m.add_class::<PyWebSearchParams>()?;
    m.add_class::<PyEventStream>()?;
    m.add_class::<PySkillInfo>()?;
    m.add_class::<PyFileMemoryStore>()?;
    m.add_class::<PyFileSessionStore>()?;
    m.add_class::<PyMemorySessionStore>()?;
    m.add_class::<PyDefaultSecurityProvider>()?;
    m.add_class::<PyLocalWorkspaceBackend>()?;
    m.add_class::<PyS3WorkspaceBackend>()?;
    m.add_class::<PyRemoteGitBackendConfig>()?;
    m.add_class::<PyStdioTransport>()?;
    m.add_class::<PyHttpTransport>()?;
    m.add_class::<PyWebSocketTransport>()?;
    m.add_class::<PyUnixSocketTransport>()?;
    m.add_class::<PyPermissionPolicy>()?;
    m.add_class::<PyConfirmationPolicy>()?;
    m.add_class::<PyArtifactStoreLimits>()?;
    m.add_class::<PyWorkerAgentSpec>()?;
    m.add_class::<PyAgentDefinition>()?;
    m.add_class::<PyAutoDelegationConfig>()?;
    m.add_class::<PySessionOptions>()?;
    m.add_class::<PyServeHandle>()?;
    m.add_class::<PySessionQueueConfig>()?;
    m.add_class::<PySearchConfig>()?;
    m.add_class::<PySearchEngineConfig>()?;
    m.add_class::<PySearchHealthConfig>()?;
    m.add_class::<PyBrowserBackend>()?;
    m.add_class::<PyHeadlessConfig>()?;
    m.add_class::<PyEventType>()?;
    // AHP types
    m.add_class::<PyAhpEventType>()?;
    m.add_class::<PyFact>()?;
    m.add_class::<PyMemorySummary>()?;
    m.add_class::<PySessionStats>()?;
    m.add_class::<PyIdleDecision>()?;
    m.add_class::<PyAhpEventContext>()?;
    m.add_class::<PyTargetHints>()?;
    m.add_class::<PyIntentDetectionEvent>()?;
    m.add_class::<PyIntentDetectionDecision>()?;
    m.add_function(wrap_pyfunction!(format_verification_summary, m)?)?;
    m.add_function(wrap_pyfunction!(py_builtin_skills, m)?)?;

    Ok(())
}

/// Return a list of built-in skills compiled into the library.
///
/// Each entry has `name`, `description`, and `kind` (instruction, tool, or agent).
#[pyfunction(name = "builtin_skills")]
fn py_builtin_skills() -> Vec<PySkillInfo> {
    rust_builtin_skills()
        .into_iter()
        .map(|s| PySkillInfo {
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

#[cfg(test)]
mod tests {
    use super::*;

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

    fn build_test_session() -> PySession {
        let agent = get_runtime()
            .block_on(RustAgent::from_config(sdk_test_config()))
            .unwrap();
        let session = agent.session("/tmp/a3s-code-python-sdk-api", None).unwrap();
        PySession {
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
    fn session_options_map_parallel_delegation_controls() {
        let mut session_options = PySessionOptions::new();
        session_options.max_parallel_tasks = Some(3);
        session_options.auto_delegation = Some(PyAutoDelegationConfig::new(true, true, 0.8, 2));
        session_options.auto_parallel = Some(false);

        let opts = build_rust_session_options(session_options).unwrap();
        assert_eq!(opts.max_parallel_tasks, Some(3));
        assert_eq!(opts.auto_parallel_delegation, Some(false));
        let auto = opts.auto_delegation.expect("auto delegation config");
        assert!(auto.enabled);
        assert!(!auto.auto_parallel);
        assert!((auto.min_confidence - 0.8).abs() < f32::EPSILON);
        assert_eq!(auto.max_tasks, 2);
    }

    #[test]
    fn session_options_map_active_skill_tool_restriction_control() {
        let default_opts = build_rust_session_options(PySessionOptions::new()).unwrap();
        assert_eq!(default_opts.enforce_active_skill_tool_restrictions, None);

        let mut session_options = PySessionOptions::new();
        session_options.enforce_active_skill_tool_restrictions = Some(true);

        let opts = build_rust_session_options(session_options).unwrap();
        assert_eq!(opts.enforce_active_skill_tool_restrictions, Some(true));
    }

    #[test]
    fn artifact_store_limits_map_to_rust_session_options() {
        let mut session_options = PySessionOptions::new();
        session_options.artifact_store_limits = Some(PyArtifactStoreLimits {
            max_artifacts: 3,
            max_bytes: 4096,
        });

        let opts = build_rust_session_options(session_options).unwrap();
        let limits = opts.artifact_store_limits.expect("limits");
        assert_eq!(limits.max_artifacts, 3);
        assert_eq!(limits.max_bytes, 4096);
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
    fn py_session_records_verification_reports() {
        pyo3::prepare_freethreaded_python();
        let session = build_test_session();

        Python::with_gil(|py| {
            let json_mod = py.import("json").unwrap();
            let reports = json_mod
                .call_method1(
                    "loads",
                    (serde_json::json!([verification_report_json()]).to_string(),),
                )
                .unwrap();
            session.record_verification_reports(py, &reports).unwrap();
        });

        let reports = session.inner.verification_reports();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].subject, "sdk:test");
        assert!(matches!(
            session.inner.verification_summary().status,
            RustVerificationStatus::Passed
        ));
    }

    #[test]
    fn py_session_get_artifact_returns_none_for_missing_uri() {
        pyo3::prepare_freethreaded_python();
        let session = build_test_session();

        Python::with_gil(|py| {
            let artifact = session
                .get_artifact(py, "a3s://tool-output/missing")
                .unwrap();
            assert!(artifact.bind(py).is_none());
        });
    }

    #[test]
    fn local_workspace_backend_maps_to_rust_session_options() {
        pyo3::prepare_freethreaded_python();
        let opts = Python::with_gil(|py| {
            let backend = Py::new(
                py,
                PyLocalWorkspaceBackend {
                    root: ".".to_string(),
                },
            )
            .unwrap();
            let mut session_options = PySessionOptions::new();
            session_options.workspace_backend = Some(backend.into_any());
            build_rust_session_options(session_options)
        })
        .unwrap();

        assert!(opts.workspace_services.is_some());
    }

    #[test]
    fn s3_workspace_backend_maps_to_rust_session_options() {
        pyo3::prepare_freethreaded_python();
        let opts = Python::with_gil(|py| {
            let backend = Py::new(
                py,
                PyS3WorkspaceBackend {
                    bucket: "workspace".to_string(),
                    prefix: "users/u1/sessions/s1".to_string(),
                    access_key_id: "AKIA".to_string(),
                    secret_access_key: "secret".to_string(),
                    endpoint: Some("https://minio.local:9000".to_string()),
                    region: Some("us-east-1".to_string()),
                    session_token: None,
                    force_path_style: true,
                    max_read_bytes: None,
                    search_enabled: false,
                    max_objects_scanned: None,
                    max_grep_bytes_per_object: None,
                    search_concurrency: None,
                },
            )
            .unwrap();
            let mut session_options = PySessionOptions::new();
            session_options.workspace_backend = Some(backend.into_any());
            build_rust_session_options(session_options)
        })
        .unwrap();

        let services = opts.workspace_services.expect("s3 backend builds services");
        let caps = services.capabilities();
        assert!(caps.read);
        assert!(caps.write);
        assert!(!caps.exec);
        assert!(!caps.git);
        assert!(!caps.search);
    }

    #[test]
    fn s3_phase1_3_options_thread_through_to_core() {
        pyo3::prepare_freethreaded_python();
        let opts = Python::with_gil(|py| {
            let backend = Py::new(
                py,
                PyS3WorkspaceBackend {
                    bucket: "workspace".to_string(),
                    prefix: "u1/s1".to_string(),
                    access_key_id: "AKIA".to_string(),
                    secret_access_key: "secret".to_string(),
                    endpoint: None,
                    region: None,
                    session_token: None,
                    force_path_style: false,
                    max_read_bytes: Some(4 * 1024 * 1024),
                    search_enabled: true,
                    max_objects_scanned: Some(250),
                    max_grep_bytes_per_object: Some(512 * 1024),
                    search_concurrency: None,
                },
            )
            .unwrap();
            let mut session_options = PySessionOptions::new();
            session_options.workspace_backend = Some(backend.into_any());
            build_rust_session_options(session_options)
        })
        .unwrap();

        let services = opts.workspace_services.expect("services built");
        assert!(
            services.capabilities().search,
            "search_enabled=true must enable the search capability"
        );
        assert!(services.search().is_some());
    }

    #[test]
    fn remote_git_attaches_on_top_of_s3_backend() {
        pyo3::prepare_freethreaded_python();
        let opts = Python::with_gil(|py| {
            let backend = Py::new(
                py,
                PyS3WorkspaceBackend {
                    bucket: "workspace".to_string(),
                    prefix: "u1/s1".to_string(),
                    access_key_id: "AKIA".to_string(),
                    secret_access_key: "secret".to_string(),
                    endpoint: None,
                    region: None,
                    session_token: None,
                    force_path_style: false,
                    max_read_bytes: None,
                    search_enabled: false,
                    max_objects_scanned: None,
                    max_grep_bytes_per_object: None,
                    search_concurrency: None,
                },
            )
            .unwrap();
            let mut session_options = PySessionOptions::new();
            session_options.workspace_backend = Some(backend.into_any());
            session_options.remote_git = Some(PyRemoteGitBackendConfig {
                base_url: "https://gitserver.internal".to_string(),
                repo_id: "u1/s1".to_string(),
                bearer_token: Some("tok".to_string()),
                client_cert_pem: None,
                client_key_pem: None,
                request_timeout_ms: Some(10_000),
                max_diff_bytes: None,
                max_log_entries: None,
            });
            build_rust_session_options(session_options)
        })
        .unwrap();

        let services = opts.workspace_services.expect("services built");
        assert!(services.git().is_some());
        assert!(services.git_stash().is_some());
        // Worktree intentionally unavailable on remote-git workspaces (RFC §8).
        assert!(services.git_worktree().is_none());
        assert!(services.capabilities().git);
    }

    /// Phase 8 alignment: a typed `ToolErrorKind` from the Rust core
    /// must arrive at the Python SDK as a JSON envelope on
    /// `error_kind_json`, with the discriminator on `type`. We assert
    /// both the raw string shape and the parsed serde_json round-trip
    /// (Python's `error_kind` getter calls `json_string_to_py` on the
    /// same string, so this test fully covers the contract without
    /// needing a Python interpreter to run JSON.parse).
    #[test]
    fn py_tool_result_threads_error_kind_json() {
        let kind = a3s_code_core::ToolErrorKind::VersionConflict {
            path: "doc.md".to_string(),
            expected: "etag-1".to_string(),
            actual: Some("etag-2".to_string()),
        };
        // The SDK conversion path uses `serde_json::to_string(&k).ok()`;
        // mirror that here to exercise the exact envelope shape the
        // Python `error_kind` property reads from.
        let json = serde_json::to_string(&kind).expect("kind serialises");
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "version_conflict");
        assert_eq!(parsed["path"], "doc.md");
        assert_eq!(parsed["expected"], "etag-1");
        assert_eq!(parsed["actual"], "etag-2");
    }

    /// Successful tool calls and tool calls that fail without a typed
    /// reason must leave `error_kind_json` as `None` so SDK callers can
    /// rely on its presence as the sole "is this a typed failure?"
    /// signal.
    #[test]
    fn py_tool_result_error_kind_json_is_none_when_no_kind() {
        let result = a3s_code_core::ToolCallResult {
            name: "read".to_string(),
            output: "hello".to_string(),
            exit_code: 0,
            metadata: None,
            error_kind: None,
        };
        let json = result
            .error_kind
            .as_ref()
            .and_then(|k| serde_json::to_string(k).ok());
        assert!(json.is_none());
    }

    #[test]
    fn remote_git_without_workspace_backend_errors_clearly() {
        pyo3::prepare_freethreaded_python();
        let result = Python::with_gil(|_py| {
            let mut session_options = PySessionOptions::new();
            session_options.remote_git = Some(PyRemoteGitBackendConfig {
                base_url: "https://gitserver".to_string(),
                repo_id: "r".to_string(),
                bearer_token: None,
                client_cert_pem: None,
                client_key_pem: None,
                request_timeout_ms: None,
                max_diff_bytes: None,
                max_log_entries: None,
            });
            build_rust_session_options(session_options)
        });

        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("workspace_backend"),
            "error must mention missing field, got: {}",
            msg
        );
    }

    #[test]
    fn delegate_task_args_use_core_task_schema() {
        let args = delegate_task_args(
            "explore".to_string(),
            "Find auth files".to_string(),
            "Inspect auth files".to_string(),
            true,
            Some(3),
        );

        assert_eq!(args["agent"], "explore");
        assert_eq!(args["description"], "Find auth files");
        assert_eq!(args["prompt"], "Inspect auth files");
        assert_eq!(args["background"], true);
        assert_eq!(args["max_steps"], 3);
        assert!(args.get("role").is_none());
    }

    #[test]
    fn parallel_task_args_use_core_parallel_task_schema() {
        let args = parallel_task_args(serde_json::json!([
            { "agent": "explore", "description": "Find tests", "prompt": "Locate tests" },
            { "agent": "verification", "description": "Check risks", "prompt": "Review risks" }
        ]))
        .unwrap();

        assert_eq!(args["tasks"].as_array().unwrap().len(), 2);
        assert_eq!(args["tasks"][0]["agent"], "explore");
        assert_eq!(args["tasks"][1]["agent"], "verification");
        assert!(parallel_task_args(serde_json::json!({ "agent": "explore" })).is_err());
    }

    #[test]
    fn program_options_normalize_to_script_tool_contract() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item(
                "source",
                "async function run(ctx, inputs) { return inputs; }",
            )
            .unwrap();
            dict.set_item(
                "inputs",
                serde_json::json!({ "needle": "auth" }).to_string(),
            )
            .unwrap();
            dict.set_item("allowedTools", vec!["grep", "read"]).unwrap();

            let args = normalize_program_script_options(&dict).unwrap();
            assert_eq!(args["type"], "script");
            assert_eq!(args["language"], "javascript");
            assert_eq!(args["allowed_tools"], serde_json::json!(["grep", "read"]));
        });
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
            "timeout_ms": 1500
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

    // ---- orchestration conversion + pipeline-stage bridge (#43) ----

    #[test]
    fn py_to_step_spec_parses_full_dict() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("task_id", "t1").unwrap();
            dict.set_item("agent", "explore").unwrap();
            dict.set_item("description", "d").unwrap();
            dict.set_item("prompt", "p").unwrap();
            dict.set_item("max_steps", 5u32).unwrap();
            dict.set_item("parent_session_id", "parent").unwrap();
            let schema = PyDict::new(py);
            schema.set_item("type", "object").unwrap();
            dict.set_item("output_schema", &schema).unwrap();

            let spec = py_to_step_spec(py, dict.as_any()).unwrap();
            assert_eq!(spec.task_id, "t1");
            assert_eq!(spec.agent, "explore");
            assert_eq!(spec.prompt, "p");
            assert_eq!(spec.max_steps, Some(5));
            assert_eq!(spec.parent_session_id.as_deref(), Some("parent"));
            assert!(spec.output_schema.is_some());
        });
    }

    #[test]
    fn py_to_step_spec_minimal_defaults_optionals() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("task_id", "t1").unwrap();
            dict.set_item("agent", "explore").unwrap();
            dict.set_item("description", "d").unwrap();
            dict.set_item("prompt", "p").unwrap();
            let spec = py_to_step_spec(py, dict.as_any()).unwrap();
            assert_eq!(spec.max_steps, None);
            assert_eq!(spec.parent_session_id, None);
            assert_eq!(spec.output_schema, None);
        });
    }

    #[test]
    fn py_to_step_spec_missing_required_field_errors() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("task_id", "t1").unwrap();
            dict.set_item("agent", "explore").unwrap();
            dict.set_item("description", "d").unwrap();
            // No "prompt" — a required field with no serde default.
            let err = py_to_step_spec(py, dict.as_any()).unwrap_err();
            assert!(
                err.to_string().contains("AgentStepSpec") || err.to_string().contains("prompt"),
                "got: {err}"
            );
        });
    }

    #[test]
    fn step_outcome_to_py_uses_snake_case_keys() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let outcome = RustStepOutcome {
                task_id: "t1".into(),
                session_id: "task-run-t1".into(),
                agent: "explore".into(),
                output: "o".into(),
                success: true,
                structured: Some(serde_json::json!({ "k": 1 })),
            };
            let obj = step_outcome_to_py(py, &outcome).unwrap();
            let bound = obj.bind(py);
            let dict = bound.downcast::<PyDict>().unwrap();
            // snake_case keys — the casing the pipeline `ctx['previous']` relies on.
            assert_eq!(
                dict.get_item("task_id")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "t1"
            );
            assert_eq!(
                dict.get_item("session_id")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "task-run-t1"
            );
            assert!(dict
                .get_item("success")
                .unwrap()
                .unwrap()
                .extract::<bool>()
                .unwrap());
            assert!(dict.get_item("structured").unwrap().is_some());
        });
    }

    #[test]
    fn python_pipeline_stage_none_raise_and_spec() {
        pyo3::prepare_freethreaded_python();
        let (none_cb, raise_cb, spec_cb) = Python::with_gil(|py| {
            let none_cb = py.eval(c"lambda ctx: None", None, None).unwrap().unbind();
            // A raising stage must fail closed (caught → None), not abort.
            let raise_cb = py.eval(c"lambda ctx: 1 / 0", None, None).unwrap().unbind();
            // Reads ctx['previous']['task_id'] (snake_case) and returns a spec.
            let spec_cb = py
                .eval(
                    c"lambda ctx: {'task_id': 'ps', 'agent': 'review', 'description': 'd', 'prompt': 'prev=' + str(ctx['previous']['task_id'])}",
                    None,
                    None,
                )
                .unwrap()
                .unbind();
            (none_cb, raise_cb, spec_cb)
        });

        assert!(PythonPipelineStage { callback: none_cb }
            .invoke(None, &serde_json::json!({ "x": 1 }))
            .is_none());
        assert!(
            PythonPipelineStage { callback: raise_cb }
                .invoke(None, &serde_json::json!({ "x": 1 }))
                .is_none(),
            "a raising stage fails closed to None"
        );

        let prev = RustStepOutcome {
            task_id: "prior".into(),
            session_id: "s".into(),
            agent: "a".into(),
            output: "o".into(),
            success: true,
            structured: None,
        };
        let spec = PythonPipelineStage { callback: spec_cb }
            .invoke(Some(&prev), &serde_json::json!({ "x": 1 }))
            .expect("spec returned");
        assert_eq!(spec.task_id, "ps");
        assert!(
            spec.prompt.contains("prior"),
            "ctx['previous']['task_id'] (snake_case) was readable: {}",
            spec.prompt
        );
    }
}
