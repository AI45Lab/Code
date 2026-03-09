//! A3S Code Python Bindings
//!
//! Native Python module via PyO3 that wraps `a3s-code-core`'s Agent API.
//!
//! ## Usage
//!
//! ```python
//! from a3s_code import Agent
//!
//! agent = Agent.create("agent.hcl")
//! session = agent.session("/my-project")
//!
//! result = session.send("What files handle auth?")
//! print(result.text)
//! ```

use a3s_code_core::agent::{AgentEvent as RustAgentEvent, AgentResult as RustAgentResult};
use a3s_code_core::commands::{
    CommandContext as RustCommandContext, CommandOutput as RustCommandOutput,
    SlashCommand as RustSlashCommand,
};
use a3s_code_core::agent_teams::{
    AgentTeam as RustAgentTeam, TaskStatus as RustTaskStatus, TeamConfig as RustTeamConfig,
    TeamRole as RustTeamRole, TeamRunner as RustTeamRunner, TeamTask as RustTeamTask,
    TeamTaskBoard as RustTeamTaskBoard,
};
use a3s_code_core::orchestrator::{
    AgentOrchestrator as RustOrchestrator, AgentSlot as RustAgentSlot,
    ControlSignal as RustControlSignal, OrchestratorEvent as RustOrchestratorEvent,
    SubAgentActivity as RustSubAgentActivity, SubAgentConfig as RustSubAgentConfig,
    SubAgentHandle as RustSubAgentHandle, SubAgentInfo as RustSubAgentInfo,
    SubAgentState as RustSubAgentState,
};
use a3s_code_core::config::{
    SearchConfig as RustSearchConfig, SearchEngineConfig as RustSearchEngineConfig,
    SearchHealthConfig as RustSearchHealthConfig,
};
use a3s_code_core::hooks::{
    Hook as RustHook, HookConfig as RustHookConfig, HookEvent as RustHookEvent,
    HookEventType as RustHookEventType, HookHandler as RustHookHandler,
    HookMatcher as RustHookMatcher, HookResponse as RustHookResponse,
};
use a3s_code_core::llm::Message as RustMessage;
use a3s_code_core::queue::{
    ExternalTaskResult as RustExternalTaskResult, LaneHandlerConfig as RustLaneHandlerConfig,
    SessionLane as RustSessionLane, SessionQueueConfig as RustSessionQueueConfig,
    TaskHandlerMode as RustTaskHandlerMode,
};
use a3s_code_core::{builtin_skills as rust_builtin_skills, SkillKind as RustSkillKind};
use a3s_code_core::{
    Agent as RustAgent, AgentSession as RustAgentSession, SessionOptions as RustSessionOptions,
};
use pyo3::exceptions::{
    PyRuntimeError, PyStopAsyncIteration, PyStopIteration, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tokio::runtime::Runtime;
use tokio::sync::Mutex;

// ============================================================================
// Tokio Runtime
// ============================================================================

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
}

#[pymethods]
impl PyAgentResult {
    fn __repr__(&self) -> String {
        format!(
            "AgentResult(text={:?}, tool_calls={}, tokens={})",
            if self.text.len() > 80 {
                format!("{}...", &self.text[..80])
            } else {
                self.text.clone()
            },
            self.tool_calls_count,
            self.total_tokens,
        )
    }

    fn __str__(&self) -> &str {
        &self.text
    }
}

impl From<RustAgentResult> for PyAgentResult {
    fn from(r: RustAgentResult) -> Self {
        Self {
            text: r.text,
            tool_calls_count: r.tool_calls_count,
            prompt_tokens: r.usage.prompt_tokens,
            completion_tokens: r.usage.completion_tokens,
            total_tokens: r.usage.total_tokens,
        }
    }
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
}

impl From<RustAgentEvent> for PyAgentEvent {
    fn from(e: RustAgentEvent) -> Self {
        match e {
            RustAgentEvent::Start { prompt } => Self {
                prompt: Some(prompt),
                ..Self::empty("start")
            },
            RustAgentEvent::TextDelta { text } => Self {
                text: Some(text),
                ..Self::empty("text_delta")
            },
            RustAgentEvent::TurnStart { turn } => Self {
                turn: Some(turn),
                ..Self::empty("turn_start")
            },
            RustAgentEvent::TurnEnd { turn, usage } => Self {
                turn: Some(turn),
                total_tokens: Some(usage.total_tokens),
                ..Self::empty("turn_end")
            },
            RustAgentEvent::ToolStart { id, name } => Self {
                tool_id: Some(id),
                tool_name: Some(name),
                ..Self::empty("tool_start")
            },
            RustAgentEvent::ToolEnd {
                id,
                name,
                output,
                exit_code,
                metadata: _,
            } => Self {
                tool_id: Some(id),
                tool_name: Some(name),
                tool_output: Some(output),
                exit_code: Some(exit_code),
                ..Self::empty("tool_end")
            },
            RustAgentEvent::ToolOutputDelta { id, name, delta } => Self {
                tool_id: Some(id),
                tool_name: Some(name),
                text: Some(delta),
                ..Self::empty("tool_output_delta")
            },
            RustAgentEvent::End { text, usage } => Self {
                text: Some(text),
                total_tokens: Some(usage.total_tokens),
                ..Self::empty("end")
            },
            RustAgentEvent::Error { message } => Self {
                error: Some(message),
                ..Self::empty("error")
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
            RustAgentEvent::ToolInputDelta { delta } => Self {
                text: Some(delta),
                ..Self::empty("tool_input_delta")
            },
            RustAgentEvent::SubagentProgress { task_id, session_id, status, metadata: _ } => Self {
                tool_id: Some(task_id),
                text: Some(format!("{}: {}", session_id, status)),
                ..Self::empty("subagent_progress")
            },
            // Catch-all for other event types (HITL, permissions, context, memory, planning, etc.)
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
}

#[pymethods]
impl PyToolResult {
    fn __repr__(&self) -> String {
        format!(
            "ToolResult(name='{}', exit_code={})",
            self.name, self.exit_code
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

/// AI coding agent. Create with `Agent.create()`, then call `agent.session()`.
#[pyclass(name = "Agent")]
struct PyAgent {
    inner: Arc<RustAgent>,
}

#[pymethods]
impl PyAgent {
    /// Create an Agent from a config file path or inline config string.
    ///
    /// Args:
    ///     config_source: Path to .hcl/.json file, or inline JSON/HCL string
    #[staticmethod]
    fn create(py: Python<'_>, config_source: String) -> PyResult<Self> {
        let agent = py
            .allow_threads(move || get_runtime().block_on(RustAgent::new(config_source)))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create agent: {e}")))?;

        Ok(Self {
            inner: Arc::new(agent),
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
    ///     queue_config: Optional SessionQueueConfig for lane-based tool execution
    ///     permissive: Optional bool to allow all tools without HITL confirmation (default: False)
    ///     planning: Optional bool to enable planning mode (default: False)
    ///     goal_tracking: Optional bool to enable goal tracking (default: False)
    ///     max_parse_retries: Optional max consecutive parse errors before abort
    ///     tool_timeout_ms: Optional per-tool execution timeout in milliseconds
    ///     circuit_breaker_threshold: Optional max LLM API failures before abort
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

    #[pyo3(signature = (workspace, options=None, model=None, builtin_skills=None, skill_dirs=None, agent_dirs=None, queue_config=None, permissive=None, planning=None, goal_tracking=None, max_parse_retries=None, tool_timeout_ms=None, circuit_breaker_threshold=None))]
    fn session(
        &self,
        workspace: String,
        options: Option<PySessionOptions>,
        model: Option<String>,
        builtin_skills: Option<bool>,
        skill_dirs: Option<Vec<String>>,
        agent_dirs: Option<Vec<String>>,
        queue_config: Option<PySessionQueueConfig>,
        permissive: Option<bool>,
        planning: Option<bool>,
        goal_tracking: Option<bool>,
        max_parse_retries: Option<u32>,
        tool_timeout_ms: Option<u64>,
        circuit_breaker_threshold: Option<u32>,
    ) -> PyResult<PySession> {
        // If a SessionOptions object is provided, build from it then apply keyword overrides
        let opts = if let Some(so) = options {
            let mut o = build_rust_session_options(so);
            // Keyword args take precedence over SessionOptions fields
            if permissive.unwrap_or(false) {
                o = o.with_permissive_policy();
            }
            if planning.unwrap_or(false) {
                o = o.with_planning(true);
            }
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
            Some(o)
        } else {
            // Fall back to individual keyword arguments
            let has_overrides = model.is_some()
                || builtin_skills.is_some()
                || skill_dirs.is_some()
                || agent_dirs.is_some()
                || queue_config.is_some()
                || permissive.is_some()
                || planning.is_some()
                || goal_tracking.is_some()
                || max_parse_retries.is_some()
                || tool_timeout_ms.is_some()
                || circuit_breaker_threshold.is_some();

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
                if permissive.unwrap_or(false) {
                    o = o.with_permissive_policy();
                }
                if planning.unwrap_or(false) {
                    o = o.with_planning(true);
                }
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
    fn resume_session(
        &self,
        session_id: String,
        options: PySessionOptions,
    ) -> PyResult<PySession> {
        let opts = build_rust_session_options(options);
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
    #[pyo3(signature = (workspace, agent_name, agent_dirs=None))]
    fn session_for_agent(
        &self,
        workspace: String,
        agent_name: String,
        agent_dirs: Option<Vec<String>>,
    ) -> PyResult<PySession> {
        let registry = a3s_code_core::AgentRegistry::new();
        for dir in agent_dirs.unwrap_or_default() {
            let agents =
                a3s_code_core::load_agents_from_dir(std::path::Path::new(&dir));
            for agent in agents {
                registry.register(agent);
            }
        }
        let def = registry
            .get(&agent_name)
            .ok_or_else(|| PyRuntimeError::new_err(format!("agent '{}' not found", agent_name)))?;
        let session = self
            .inner
            .session_for_agent(workspace, &def, None)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        Ok(PySession {
            inner: Arc::new(session),
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
    /// Send a prompt and wait for the complete response.
    ///
    /// Args:
    ///     prompt: The prompt to send
    ///     history: Optional conversation history as list of dicts
    ///              `[{"role": "user", "content": [{"type": "text", "text": "..."}]}]`
    #[pyo3(signature = (prompt, history=None))]
    fn send(
        &self,
        py: Python<'_>,
        prompt: String,
        history: Option<&Bound<'_, PyList>>,
    ) -> PyResult<PyAgentResult> {
        let rust_history = history.map(|h| py_list_to_messages(h)).transpose()?;
        let session = self.inner.clone();
        let result = py
            .allow_threads(move || {
                get_runtime().block_on(session.send(&prompt, rust_history.as_deref()))
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Agent execution failed: {e}")))?;
        Ok(PyAgentResult::from(result))
    }

    /// Send a prompt and get a streaming iterator of events.
    ///
    /// Args:
    ///     prompt: The prompt to send
    ///     history: Optional conversation history (same format as send)
    #[pyo3(signature = (prompt, history=None))]
    fn stream(
        &self,
        py: Python<'_>,
        prompt: String,
        history: Option<&Bound<'_, PyList>>,
    ) -> PyResult<PyEventStream> {
        let rust_history = history.map(|h| py_list_to_messages(h)).transpose()?;
        let session = self.inner.clone();
        let (rx, _handle) = py
            .allow_threads(move || {
                get_runtime().block_on(session.stream(&prompt, rust_history.as_deref()))
            })
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
        })
    }

    /// Run a goal through the built-in `run_team` tool.
    ///
    /// Spawns a Lead → Worker → Reviewer team as child subagents: the Lead
    /// decomposes `goal` into tasks, Workers execute them concurrently, and the
    /// Reviewer approves or rejects each result (rejected tasks are retried).
    ///
    /// This is a typed convenience wrapper over `session.tool("run_team", {...})`.
    /// All agent-type arguments default to `"general"`.
    ///
    /// Returns a `ToolResult` whose `output` field contains the formatted team
    /// run summary.
    #[pyo3(signature = (goal, lead_agent="general", worker_agent="general", reviewer_agent="general", max_steps=None))]
    fn run_team(
        &self,
        py: Python<'_>,
        goal: String,
        lead_agent: &str,
        worker_agent: &str,
        reviewer_agent: &str,
        max_steps: Option<usize>,
    ) -> PyResult<PyToolResult> {
        let mut args = serde_json::json!({
            "goal": goal,
            "lead_agent": lead_agent,
            "worker_agent": worker_agent,
            "reviewer_agent": reviewer_agent,
        });
        if let Some(steps) = max_steps {
            args["max_steps"] = serde_json::json!(steps);
        }
        let session = self.inner.clone();
        let result = py
            .allow_threads(move || get_runtime().block_on(session.tool("run_team", args)))
            .map_err(|e| PyRuntimeError::new_err(format!("run_team failed: {e}")))?;
        Ok(PyToolResult {
            name: result.name,
            output: result.output,
            exit_code: result.exit_code,
        })
    }

    /// Read a file from the workspace.
    fn read_file(&self, py: Python<'_>, path: String) -> PyResult<String> {
        let session = self.inner.clone();
        py.allow_threads(move || get_runtime().block_on(session.read_file(&path)))
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
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

    // ========================================================================
    // Queue API
    // ========================================================================

    /// Check if this session has a lane queue configured.
    fn has_queue(&self) -> bool {
        self.inner.has_queue()
    }

    /// Configure a lane's handler mode.
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

    /// Complete an external task by ID.
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

    /// Get pending external tasks.
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
            .map(|l| l.clone())
            .map_err(|e| PyRuntimeError::new_err(format!("Unexpected result: {e}")))
    }

    /// Get queue statistics.
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
            .map(|d| d.clone())
            .map_err(|e| PyRuntimeError::new_err(format!("Unexpected result: {e}")))
    }

    /// Get dead letters from the queue's DLQ.
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
            .map(|l| l.clone())
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
    #[pyo3(signature = (name, transport="stdio", command=None, args=None, url=None, headers=None, env=None))]
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

        let config = McpServerConfig {
            name: name.to_string(),
            transport: transport_config,
            enabled: true,
            env: env.unwrap_or_default(),
            oauth: None,
            tool_timeout_secs: 60,
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

    /// Return the names of all tools currently available in this session.
    ///
    /// Reflects the live state — MCP tools appear after ``add_mcp_server()``
    /// and disappear after ``remove_mcp_server()``.
    ///
    /// Returns:
    ///     List of tool name strings
    fn tool_names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let names = self.inner.tool_names();
        let list = PyList::new(py, names)?;
        Ok(list)
    }
    ///
    /// Args:
    ///     lane (Literal["control", "query", "execute", "generate"]): Which lane to submit to.
    ///     callable: A Python callable that takes no arguments and returns a
    ///               JSON-serializable value.
    ///
    /// Returns:
    ///     The result of the callable as a Python object (dict, list, str, etc.)
    ///
    /// Raises:
    ///     RuntimeError: If no queue is configured or the command fails
    fn submit(&self, py: Python<'_>, lane: &str, callable: PyObject) -> PyResult<PyObject> {
        let lane = parse_lane(lane)?;
        let cmd = PythonCommand { callable };
        let session = self.inner.clone();
        let rx = py
            .allow_threads(move || get_runtime().block_on(session.submit(lane, Box::new(cmd))))
            .map_err(|e| PyRuntimeError::new_err(format!("submit failed: {e}")))?;
        let result = py
            .allow_threads(move || get_runtime().block_on(rx))
            .map_err(|e| PyRuntimeError::new_err(format!("receiver dropped: {e}")))?
            .map_err(|e| PyRuntimeError::new_err(format!("command failed: {e}")))?;
        let json_str = serde_json::to_string(&result)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {e}")))?;
        let json_mod = py.import("json")?;
        Ok(json_mod.call_method1("loads", (json_str,))?.into())
    }

    /// Submit a batch of Python callables to the session's lane queue.
    ///
    /// More efficient than calling `submit()` in a loop.
    ///
    /// Args:
    ///     lane (Literal["control", "query", "execute", "generate"]): Which lane to submit to.
    ///     callables: List of Python callables, each taking no arguments and
    ///                returning a JSON-serializable value.
    ///
    /// Returns:
    ///     List of results in the same order as the input callables
    ///
    /// Raises:
    ///     RuntimeError: If no queue is configured or any command fails
    fn submit_batch<'py>(
        &self,
        py: Python<'py>,
        lane: &str,
        callables: Vec<PyObject>,
    ) -> PyResult<Bound<'py, PyList>> {
        let lane = parse_lane(lane)?;
        let commands: Vec<Box<dyn a3s_code_core::queue::SessionCommand>> = callables
            .into_iter()
            .map(|c| -> Box<dyn a3s_code_core::queue::SessionCommand> {
                Box::new(PythonCommand { callable: c })
            })
            .collect();
        let session = self.inner.clone();
        let receivers = py
            .allow_threads(move || get_runtime().block_on(session.submit_batch(lane, commands)))
            .map_err(|e| PyRuntimeError::new_err(format!("submit_batch failed: {e}")))?;
        // Await all receivers and collect results.
        let results = py
            .allow_threads(move || {
                get_runtime().block_on(async move {
                    let mut out = Vec::with_capacity(receivers.len());
                    for rx in receivers {
                        let val = rx
                            .await
                            .map_err(|e| anyhow::anyhow!("receiver dropped: {e}"))?
                            .map_err(|e| anyhow::anyhow!("command failed: {e}"))?;
                        out.push(val);
                    }
                    Ok::<Vec<serde_json::Value>, anyhow::Error>(out)
                })
            })
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        let json_mod = py.import("json")?;
        let py_results: Vec<PyObject> = results
            .into_iter()
            .map(|v| {
                let s = serde_json::to_string(&v)
                    .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {e}")))?;
                Ok(json_mod.call_method1("loads", (s,))?.into())
            })
            .collect::<PyResult<_>>()?;
        Ok(PyList::new(py, py_results)?)
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
            .map(|l| l.clone())
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
            .map(|l| l.clone())
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
            .map(|l| l.clone())
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
            .map(|d| d.clone())
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
            .map(|l| l.clone())
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
            .map(|l| l.clone())
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

    /// Schedule a recurring prompt to fire at a given interval.
    ///
    /// This is the programmatic equivalent of `/loop <interval>s <prompt>`.
    /// The scheduled prompt runs automatically after each `send()` call when it is due.
    ///
    /// :param prompt: The prompt to send at each interval.
    /// :param interval_secs: Interval in seconds (minimum: 1).
    /// :returns: 8-char hex task ID (use with :meth:`cancel_scheduled_task`).
    fn schedule_task(&self, prompt: String, interval_secs: u64) -> PyResult<String> {
        self.inner
            .cron_scheduler()
            .create_task(
                prompt,
                std::time::Duration::from_secs(interval_secs),
                true,
            )
            .map_err(PyRuntimeError::new_err)
    }

    /// List all active scheduled tasks for this session.
    ///
    /// :returns: List of dicts with keys: ``id``, ``prompt``, ``interval_secs``,
    ///     ``recurring``, ``fire_count``, ``next_fire_in_secs``.
    fn list_scheduled_tasks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let tasks = self.inner.cron_scheduler().list_tasks();
        let items: Vec<_> = tasks
            .into_iter()
            .map(|t| {
                let d = PyDict::new(py);
                let _ = d.set_item("id", &t.id);
                let _ = d.set_item("prompt", &t.prompt);
                let _ = d.set_item("interval_secs", t.interval_secs);
                let _ = d.set_item("recurring", t.recurring);
                let _ = d.set_item("fire_count", t.fire_count);
                let _ = d.set_item("next_fire_in_secs", t.next_fire_in_secs);
                d.into_any()
            })
            .collect();
        PyList::new(py, &items)
    }

    /// Cancel a scheduled task by ID.
    ///
    /// :param task_id: Task ID returned by :meth:`schedule_task` or listed by
    ///     :meth:`list_scheduled_tasks`.
    /// :returns: ``True`` if the task was found and cancelled.
    fn cancel_scheduled_task(&self, task_id: String) -> bool {
        self.inner.cron_scheduler().cancel_task(&task_id)
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
        _ => Err(PyValueError::new_err(format!(
            "Invalid hook event type: '{}'. Expected one of: pre_tool_use, post_tool_use, \
             generate_start, generate_end, session_start, session_end, skill_load, \
             skill_unload, pre_prompt, post_response, on_error",
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
    fn execute(
        &self,
        args: &str,
        ctx: &RustCommandContext,
    ) -> RustCommandOutput {
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

// ============================================================================
// SessionOptions
// ============================================================================

/// Per-session configuration options.
///
/// Pass to `agent.session(workspace, options)` to override defaults.
#[pyclass(name = "SessionOptions")]
struct PySessionOptions {
    model: Option<String>,
    builtin_skills: bool,
    skill_dirs: Vec<String>,
    agent_dirs: Vec<String>,
    queue_config: Option<PySessionQueueConfig>,
    auto_compact: bool,
    auto_compact_threshold: Option<f32>,
    /// Long-term memory store backend. Set to a ``FileMemoryStore`` instance.
    memory_store: Option<pyo3::PyObject>,
    /// Session persistence store backend. Set to ``FileSessionStore`` or ``MemorySessionStore``.
    session_store: Option<pyo3::PyObject>,
    /// Security provider. Set to ``DefaultSecurityProvider`` to enable taint tracking.
    security_provider: Option<pyo3::PyObject>,
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
    /// Automatically save the session to the configured store after each turn (default: False).
    auto_save: bool,
}

impl Clone for PySessionOptions {
    fn clone(&self) -> Self {
        pyo3::Python::with_gil(|py| Self {
            model: self.model.clone(),
            builtin_skills: self.builtin_skills,
            skill_dirs: self.skill_dirs.clone(),
            agent_dirs: self.agent_dirs.clone(),
            queue_config: self.queue_config.clone(),
            auto_compact: self.auto_compact,
            auto_compact_threshold: self.auto_compact_threshold,
            memory_store: self.memory_store.as_ref().map(|o| o.clone_ref(py)),
            session_store: self.session_store.as_ref().map(|o| o.clone_ref(py)),
            security_provider: self.security_provider.as_ref().map(|o| o.clone_ref(py)),
            role: self.role.clone(),
            guidelines: self.guidelines.clone(),
            response_style: self.response_style.clone(),
            extra: self.extra.clone(),
            inline_skills: self.inline_skills.clone(),
            max_tool_rounds: self.max_tool_rounds,
            session_id: self.session_id.clone(),
            auto_save: self.auto_save,
        })
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
            agent_dirs: vec![],
            queue_config: None,
            auto_compact: false,
            auto_compact_threshold: None,
            memory_store: None,
            session_store: None,
            security_provider: None,
            role: None,
            guidelines: None,
            response_style: None,
            extra: None,
            inline_skills: vec![],
            max_tool_rounds: None,
            session_id: None,
            auto_save: false,
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

    /// Extra directories to scan for agent files.
    #[getter]
    fn get_agent_dirs(&self) -> Vec<String> {
        self.agent_dirs.clone()
    }

    #[setter]
    fn set_agent_dirs(&mut self, value: Vec<String>) {
        self.agent_dirs = value;
    }

    /// Optional queue configuration for lane-based tool execution.
    #[getter]
    fn get_queue_config(&self) -> Option<PySessionQueueConfig> {
        self.queue_config.clone()
    }

    #[setter]
    fn set_queue_config(&mut self, value: Option<PySessionQueueConfig>) {
        self.queue_config = value;
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

    /// Session ID (auto-generated if not set). Set to save and resume sessions by name.
    #[getter]
    fn get_session_id(&self) -> Option<String> {
        self.session_id.clone()
    }

    #[setter]
    fn set_session_id(&mut self, value: Option<String>) {
        self.session_id = value;
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
            "SessionOptions(model={:?}, builtin_skills={}, queue_config={}, auto_compact={}, memory_store={}, session_store={}, security_provider={}, inline_skills={})",
            self.model,
            self.builtin_skills,
            if self.queue_config.is_some() { "Some(...)" } else { "None" },
            self.auto_compact,
            if self.memory_store.is_some() { "Some(...)" } else { "None" },
            if self.session_store.is_some() { "Some(...)" } else { "None" },
            if self.security_provider.is_some() { "Some(...)" } else { "None" },
            self.inline_skills.len(),
        )
    }
}

// ============================================================================
// SessionQueueConfig
// ============================================================================

/// Configuration for the session lane queue.
///
/// Enables priority-based tool scheduling with parallel execution
/// of read-only tools, DLQ, metrics, and external task handling.
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
// PythonCommand — wraps a Python callable as a SessionCommand
// ============================================================================

struct PythonCommand {
    callable: PyObject,
}

#[async_trait::async_trait]
impl a3s_code_core::queue::SessionCommand for PythonCommand {
    async fn execute(&self) -> anyhow::Result<serde_json::Value> {
        // Clone the callable with the GIL held, then release it before spawning.
        let callable = Python::with_gil(|py| self.callable.clone_ref(py));
        // Run the Python callable on a blocking thread to avoid holding the GIL
        // across an await point.
        tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                let result = callable
                    .call0(py)
                    .map_err(|e| anyhow::anyhow!("Python callable failed: {e}"))?;
                // Serialize via Python's json.dumps to handle arbitrary Python objects.
                let json_mod = py
                    .import("json")
                    .map_err(|e| anyhow::anyhow!("Failed to import json: {e}"))?;
                let json_str: String = json_mod
                    .call_method1("dumps", (result,))
                    .map_err(|e| anyhow::anyhow!("Failed to serialize result: {e}"))?
                    .extract()
                    .map_err(|e| anyhow::anyhow!("Failed to extract json string: {e}"))?;
                serde_json::from_str(&json_str)
                    .map_err(|e| anyhow::anyhow!("Failed to parse json: {e}"))
            })
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {e}"))?
    }

    fn command_type(&self) -> &str {
        "python"
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

/// Build a `TeamMemberOptions` from individual keyword arguments.
///
/// Returns `None` when all arguments are `None` (no overrides needed).
#[allow(clippy::too_many_arguments)]
fn py_build_team_member_options(
    workspace: Option<String>,
    model: Option<String>,
    role: Option<String>,
    guidelines: Option<String>,
    response_style: Option<String>,
    extra: Option<String>,
    max_tool_rounds: Option<usize>,
) -> Option<a3s_code_core::TeamMemberOptions> {
    if workspace.is_none()
        && model.is_none()
        && role.is_none()
        && guidelines.is_none()
        && response_style.is_none()
        && extra.is_none()
        && max_tool_rounds.is_none()
    {
        return None;
    }
    let has_slots =
        role.is_some() || guidelines.is_some() || response_style.is_some() || extra.is_some();
    let prompt_slots = has_slots.then(|| a3s_code_core::SystemPromptSlots {
        role,
        guidelines,
        response_style,
        extra,
    });
    Some(a3s_code_core::TeamMemberOptions {
        workspace,
        model,
        prompt_slots,
        max_tool_rounds,
    })
}

/// Build RustSessionOptions from PySessionOptions.
fn build_rust_session_options(so: PySessionOptions) -> RustSessionOptions {
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
    for d in &so.agent_dirs {
        o = o.with_agent_dir(d);
    }
    if let Some(qc) = so.queue_config {
        o = o.with_queue_config(qc.inner);
    }
    if so.auto_compact {
        o = o.with_auto_compact(true);
    }
    if let Some(t) = so.auto_compact_threshold {
        o = o.with_auto_compact_threshold(t);
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
            } else if store.extract::<pyo3::PyRef<PyMemorySessionStore>>(py).is_ok() {
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
        let is_default =
            Python::with_gil(|py| sec.extract::<pyo3::PyRef<PyDefaultSecurityProvider>>(py).is_ok());
        if is_default {
            o = o.with_default_security();
        }
    }
    // Build prompt slots if any slot is set
    if so.role.is_some()
        || so.guidelines.is_some()
        || so.response_style.is_some()
        || so.extra.is_some()
    {
        let slots = a3s_code_core::SystemPromptSlots {
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
            if let Some(skill) = a3s_code_core::Skill::parse(&raw) {
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
    if let Some(id) = so.session_id {
        o = o.with_session_id(id);
    }
    if so.auto_save {
        o = o.with_auto_save(true);
    }
    o
}

fn metrics_snapshot_to_json_str(
    s: a3s_code_core::MetricsSnapshot,
) -> Result<String, serde_json::Error> {
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
        .map(|l| l.clone())
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
}

#[pymethods]
impl PySearchConfig {
    #[new]
    #[pyo3(signature = (timeout=10, health=None))]
    fn new(timeout: u64, health: Option<PySearchHealthConfig>) -> Self {
        Self {
            timeout,
            health,
            engines: std::collections::HashMap::new(),
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

impl From<PySearchConfig> for RustSearchConfig {
    fn from(c: PySearchConfig) -> Self {
        Self {
            timeout: c.timeout,
            health: c.health.map(|h| h.into()),
            engines: c.engines.into_iter().map(|(k, v)| (k, v.into())).collect(),
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
                format!("{}...", &self.description[..60])
            } else {
                self.description.clone()
            }
        )
    }
}

// ============================================================================
// Agent Teams
// ============================================================================

/// Team configuration.
///
/// Args:
///     max_tasks: Maximum concurrent tasks on the board (default: 50)
///     channel_buffer: Message channel buffer size (default: 128)
///     max_rounds: Maximum coordinator rounds before run_until_done exits (default: 10)
///     poll_interval_ms: Worker/Reviewer polling interval in ms (default: 200)
#[pyclass(name = "TeamConfig")]
#[derive(Clone)]
struct PyTeamConfig {
    #[pyo3(get, set)]
    max_tasks: usize,
    #[pyo3(get, set)]
    channel_buffer: usize,
    #[pyo3(get, set)]
    max_rounds: usize,
    #[pyo3(get, set)]
    poll_interval_ms: u64,
}

#[pymethods]
impl PyTeamConfig {
    #[new]
    #[pyo3(signature = (max_tasks=50, channel_buffer=128, max_rounds=10, poll_interval_ms=200))]
    fn new(
        max_tasks: usize,
        channel_buffer: usize,
        max_rounds: usize,
        poll_interval_ms: u64,
    ) -> Self {
        Self {
            max_tasks,
            channel_buffer,
            max_rounds,
            poll_interval_ms,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TeamConfig(max_tasks={}, max_rounds={}, poll_interval_ms={})",
            self.max_tasks, self.max_rounds, self.poll_interval_ms
        )
    }
}

impl From<PyTeamConfig> for RustTeamConfig {
    fn from(c: PyTeamConfig) -> Self {
        Self {
            max_tasks: c.max_tasks,
            channel_buffer: c.channel_buffer,
            max_rounds: c.max_rounds,
            poll_interval_ms: c.poll_interval_ms,
        }
    }
}

/// A task on the team board (read-only snapshot).
#[pyclass(name = "TeamTask")]
#[derive(Clone)]
struct PyTeamTask {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    description: String,
    #[pyo3(get)]
    posted_by: String,
    #[pyo3(get)]
    assigned_to: Option<String>,
    /// Task status string: "open", "in_progress", "in_review", "done", "rejected".
    #[pyo3(get)]
    status: String,
    #[pyo3(get)]
    result: Option<String>,
    #[pyo3(get)]
    created_at: i64,
    #[pyo3(get)]
    updated_at: i64,
}

#[pymethods]
impl PyTeamTask {
    fn __repr__(&self) -> String {
        format!(
            "TeamTask(id='{}', status='{}', description={:?})",
            self.id,
            self.status,
            if self.description.len() > 60 {
                format!("{}...", &self.description[..60])
            } else {
                self.description.clone()
            }
        )
    }
}

impl From<RustTeamTask> for PyTeamTask {
    fn from(t: RustTeamTask) -> Self {
        Self {
            id: t.id,
            description: t.description,
            posted_by: t.posted_by,
            assigned_to: t.assigned_to,
            status: t.status.to_string(),
            result: t.result,
            created_at: t.created_at,
            updated_at: t.updated_at,
        }
    }
}

/// Result returned by `TeamRunner.run_until_done()`.
#[pyclass(name = "TeamRunResult")]
struct PyTeamRunResult {
    #[pyo3(get)]
    done_tasks: Vec<PyTeamTask>,
    #[pyo3(get)]
    rejected_tasks: Vec<PyTeamTask>,
    #[pyo3(get)]
    rounds: usize,
}

#[pymethods]
impl PyTeamRunResult {
    fn __repr__(&self) -> String {
        format!(
            "TeamRunResult(done={}, rejected={}, rounds={})",
            self.done_tasks.len(),
            self.rejected_tasks.len(),
            self.rounds
        )
    }
}

/// Shared task board for team coordination.
#[pyclass(name = "TeamTaskBoard")]
struct PyTeamTaskBoard {
    inner: Arc<RustTeamTaskBoard>,
}

#[pymethods]
impl PyTeamTaskBoard {
    /// Post a new task. Returns the task ID, or None if the board is full.
    ///
    /// Args:
    ///     description: Task description
    ///     posted_by: Member ID posting the task
    ///     assign_to: Optional member ID to pre-assign the task to
    #[pyo3(signature = (description, posted_by, assign_to=None))]
    fn post(
        &self,
        description: String,
        posted_by: String,
        assign_to: Option<String>,
    ) -> Option<String> {
        self.inner
            .post(&description, &posted_by, assign_to.as_deref())
    }

    /// Claim the next open or rejected task for a member.
    fn claim(&self, member_id: String) -> Option<PyTeamTask> {
        self.inner.claim(&member_id).map(PyTeamTask::from)
    }

    /// Mark a task as complete with a result. Returns True if found.
    fn complete(&self, task_id: String, result: String) -> bool {
        self.inner.complete(&task_id, &result)
    }

    /// Approve a task (reviewer action). Returns True if found in InReview state.
    fn approve(&self, task_id: String) -> bool {
        self.inner.approve(&task_id)
    }

    /// Reject a task back to open (reviewer action). Returns True if found.
    fn reject(&self, task_id: String) -> bool {
        self.inner.reject(&task_id)
    }

    /// Get a task by ID.
    fn get(&self, task_id: String) -> Option<PyTeamTask> {
        self.inner.get(&task_id).map(PyTeamTask::from)
    }

    /// Get all tasks with a given status string.
    ///
    /// Args:
    ///     status: "open", "in_progress", "in_review", "done", or "rejected"
    fn by_status(&self, status: String) -> PyResult<Vec<PyTeamTask>> {
        let s = py_parse_task_status(&status)?;
        Ok(self
            .inner
            .by_status(s)
            .into_iter()
            .map(PyTeamTask::from)
            .collect())
    }

    /// Get all tasks assigned to a member.
    fn by_assignee(&self, member_id: String) -> Vec<PyTeamTask> {
        self.inner
            .by_assignee(&member_id)
            .into_iter()
            .map(PyTeamTask::from)
            .collect()
    }

    /// Summary stats as (open, in_progress, in_review, done, rejected).
    fn stats(&self) -> (usize, usize, usize, usize, usize) {
        self.inner.stats()
    }

    /// Number of tasks on the board.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        let (o, p, r, d, rej) = self.inner.stats();
        format!(
            "TeamTaskBoard(total={}, open={}, in_progress={}, in_review={}, done={}, rejected={})",
            self.inner.len(),
            o,
            p,
            r,
            d,
            rej
        )
    }
}

fn py_parse_task_status(s: &str) -> PyResult<RustTaskStatus> {
    match s {
        "open" => Ok(RustTaskStatus::Open),
        "in_progress" => Ok(RustTaskStatus::InProgress),
        "in_review" => Ok(RustTaskStatus::InReview),
        "done" => Ok(RustTaskStatus::Done),
        "rejected" => Ok(RustTaskStatus::Rejected),
        _ => Err(PyValueError::new_err(format!(
            "Invalid task status '{}'. Expected: open, in_progress, in_review, done, rejected",
            s
        ))),
    }
}

fn py_parse_team_role(s: &str) -> PyResult<RustTeamRole> {
    match s {
        "lead" => Ok(RustTeamRole::Lead),
        "worker" => Ok(RustTeamRole::Worker),
        "reviewer" => Ok(RustTeamRole::Reviewer),
        _ => Err(PyValueError::new_err(format!(
            "Invalid role '{}'. Expected: lead, worker, reviewer",
            s
        ))),
    }
}

/// Multi-agent team coordinator.
///
/// Create the team, add members, then pass it to `TeamRunner` to execute.
///
/// Args:
///     name: Team name
///     config: Optional `TeamConfig` (uses defaults if not provided)
///
/// Example::
///
///     team = Team("refactor-auth")
///     team.add_member("lead", "lead")
///     team.add_member("worker-1", "worker")
///     team.add_member("reviewer", "reviewer")
///     runner = TeamRunner(team)
///     runner.bind_session("lead", lead_session)
///     result = runner.run_until_done("Refactor the auth module")
#[pyclass(name = "Team")]
struct PyTeam {
    inner: Arc<tokio::sync::Mutex<Option<RustAgentTeam>>>,
}

#[pymethods]
impl PyTeam {
    #[new]
    #[pyo3(signature = (name, config=None))]
    fn new(name: String, config: Option<PyTeamConfig>) -> Self {
        let rust_config = config.map(RustTeamConfig::from).unwrap_or_default();
        Self {
            inner: Arc::new(tokio::sync::Mutex::new(Some(RustAgentTeam::new(
                &name,
                rust_config,
            )))),
        }
    }

    /// Add a member to the team.
    ///
    /// Args:
    ///     member_id: Unique member identifier
    ///     role: "lead", "worker", or "reviewer"
    fn add_member(&self, member_id: String, role: String) -> PyResult<()> {
        let role = py_parse_team_role(&role)?;
        let guard = self.inner.blocking_lock();
        guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Team has been consumed by a TeamRunner"))?;
        // Temporarily take and replace to get mutable access
        drop(guard);
        let mut guard = self.inner.blocking_lock();
        if let Some(team) = guard.as_mut() {
            team.add_member(&member_id, role);
        }
        Ok(())
    }

    /// Remove a member from the team. Returns True if the member was found.
    fn remove_member(&self, member_id: String) -> bool {
        let mut guard = self.inner.blocking_lock();
        guard
            .as_mut()
            .map(|t| t.remove_member(&member_id))
            .unwrap_or(false)
    }

    /// Number of registered members.
    fn member_count(&self) -> usize {
        self.inner
            .blocking_lock()
            .as_ref()
            .map(|t| t.member_count())
            .unwrap_or(0)
    }

    /// Get the shared task board (can be used to inspect tasks after creation).
    fn task_board(&self) -> PyResult<PyTeamTaskBoard> {
        let guard = self.inner.blocking_lock();
        let team = guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Team has been consumed by a TeamRunner"))?;
        Ok(PyTeamTaskBoard {
            inner: team.task_board_arc(),
        })
    }

    fn __repr__(&self) -> String {
        let guard = self.inner.blocking_lock();
        match guard.as_ref() {
            Some(t) => format!("Team(name='{}', members={})", t.name(), t.member_count()),
            None => "Team(consumed)".to_string(),
        }
    }
}

/// Binds an agent team to real `Session` executors and runs the workflow.
///
/// The team is consumed on construction (it cannot be used after creating a runner).
///
/// Example::
///
///     runner = TeamRunner(team)
///     runner.bind_session("lead", lead_session)
///     runner.bind_session("worker-1", worker_session)
///     runner.bind_session("reviewer", reviewer_session)
///     result = runner.run_until_done("Build the feature")
///     for task in result.done_tasks:
///         print(task.id, task.result)
#[pyclass(name = "TeamRunner")]
struct PyTeamRunner {
    inner: Arc<tokio::sync::Mutex<RustTeamRunner>>,
}

#[pymethods]
impl PyTeamRunner {
    /// Create a runner from a team.
    ///
    /// The team is consumed: further calls on the original `Team` object will raise an error.
    #[new]
    fn new(team: &PyTeam) -> PyResult<Self> {
        let mut guard = team.inner.blocking_lock();
        let rust_team = guard.take().ok_or_else(|| {
            PyRuntimeError::new_err("Team has already been consumed by another TeamRunner")
        })?;
        Ok(Self {
            inner: Arc::new(tokio::sync::Mutex::new(RustTeamRunner::new(rust_team))),
        })
    }

    /// Create a runner with a default agent context.
    ///
    /// Stores the agent, workspace, and agent directories once so that
    /// subsequent calls to ``add_lead``, ``add_worker``, and ``add_reviewer``
    /// do not need to repeat them.
    ///
    /// Args:
    ///     agent: The ``Agent`` to create sessions from
    ///     workspace: Path to the workspace directory shared by all members
    ///     agent_dirs: Directories to scan for agent definition files
    #[staticmethod]
    #[pyo3(signature = (agent, workspace, agent_dirs=None))]
    fn create(
        agent: &PyAgent,
        workspace: String,
        agent_dirs: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let registry = a3s_code_core::AgentRegistry::new();
        for dir in agent_dirs.unwrap_or_default() {
            let agents = a3s_code_core::load_agents_from_dir(std::path::Path::new(&dir));
            for agent_def in agents {
                registry.register(agent_def);
            }
        }
        let team = a3s_code_core::AgentTeam::new("team", a3s_code_core::TeamConfig::default());
        let runner = RustTeamRunner::with_agent(
            team,
            agent.inner.clone(),
            &workspace,
            Arc::new(registry),
        );
        Ok(Self {
            inner: Arc::new(tokio::sync::Mutex::new(runner)),
        })
    }

    /// Add a Lead member bound to the named agent definition.
    ///
    /// Requires the runner to have been created with ``TeamRunner.create(...)``.
    /// The member ID is fixed to ``"lead"``.
    ///
    /// All keyword arguments are optional. Omitted arguments inherit from the
    /// agent definition file, then from the ``Agent`` base config::
    ///
    ///     per-call kwarg  →  AgentDefinition (.yaml/.md)  →  Agent (config.hcl)
    ///
    /// Args:
    ///     agent_name: Name of the agent definition (e.g. ``"orchestrator"``)
    ///     workspace: Override workspace (falls back to ``TeamRunner.create`` workspace)
    ///     model: Model override ``"provider/model"``; falls back to definition then Agent model
    ///     role: Custom role/identity; no definition-level default
    ///     guidelines: Custom coding guidelines; no definition-level default
    ///     response_style: Custom response style; no definition-level default
    ///     extra: Extra instructions; falls back to agent definition ``prompt`` field
    ///     max_tool_rounds: Round limit; falls back to definition ``max_steps`` then Agent config
    #[pyo3(signature = (agent_name, *, workspace=None, model=None, role=None, guidelines=None, response_style=None, extra=None, max_tool_rounds=None))]
    fn add_lead(
        &self,
        agent_name: String,
        workspace: Option<String>,
        model: Option<String>,
        role: Option<String>,
        guidelines: Option<String>,
        response_style: Option<String>,
        extra: Option<String>,
        max_tool_rounds: Option<usize>,
    ) -> PyResult<()> {
        let opts = py_build_team_member_options(workspace, model, role, guidelines, response_style, extra, max_tool_rounds);
        self.inner
            .blocking_lock()
            .add_lead(&agent_name, opts)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    /// Add a Worker member bound to the named agent definition.
    ///
    /// Requires the runner to have been created with ``TeamRunner.create(...)``.
    /// Member IDs are auto-generated as ``"worker-1"``, ``"worker-2"``, etc.
    /// Call this multiple times to add concurrent workers.
    ///
    /// Set ``workspace`` to a git worktree path to give each worker an isolated
    /// filesystem so concurrent writes do not conflict.
    /// All keyword arguments are optional and inherit from the agent definition
    /// then from the ``Agent`` base config when omitted.
    ///
    /// Args:
    ///     agent_name: Name of the agent definition (e.g. ``"general"``)
    ///     workspace: Override workspace; set to a git worktree for isolation
    ///     model: Model override ``"provider/model"``; falls back to definition then Agent model
    ///     role: Custom role/identity; no definition-level default
    ///     guidelines: Custom coding guidelines; no definition-level default
    ///     response_style: Custom response style; no definition-level default
    ///     extra: Extra instructions; falls back to agent definition ``prompt`` field
    ///     max_tool_rounds: Round limit; falls back to definition ``max_steps`` then Agent config
    #[pyo3(signature = (agent_name, *, workspace=None, model=None, role=None, guidelines=None, response_style=None, extra=None, max_tool_rounds=None))]
    fn add_worker(
        &self,
        agent_name: String,
        workspace: Option<String>,
        model: Option<String>,
        role: Option<String>,
        guidelines: Option<String>,
        response_style: Option<String>,
        extra: Option<String>,
        max_tool_rounds: Option<usize>,
    ) -> PyResult<()> {
        let opts = py_build_team_member_options(workspace, model, role, guidelines, response_style, extra, max_tool_rounds);
        self.inner
            .blocking_lock()
            .add_worker(&agent_name, opts)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    /// Add a Reviewer member bound to the named agent definition.
    ///
    /// Requires the runner to have been created with ``TeamRunner.create(...)``.
    /// The member ID is fixed to ``"reviewer"``.
    ///
    /// All keyword arguments are optional and inherit from the agent definition
    /// then from the ``Agent`` base config when omitted.
    ///
    /// Args:
    ///     agent_name: Name of the agent definition (e.g. ``"reviewer"``)
    ///     workspace: Override workspace (falls back to ``TeamRunner.create`` workspace)
    ///     model: Model override ``"provider/model"``; falls back to definition then Agent model
    ///     role: Custom role/identity; no definition-level default
    ///     guidelines: Custom coding guidelines; no definition-level default
    ///     response_style: Custom response style; no definition-level default
    ///     extra: Extra instructions; falls back to agent definition ``prompt`` field
    ///     max_tool_rounds: Round limit; falls back to definition ``max_steps`` then Agent config
    #[pyo3(signature = (agent_name, *, workspace=None, model=None, role=None, guidelines=None, response_style=None, extra=None, max_tool_rounds=None))]
    fn add_reviewer(
        &self,
        agent_name: String,
        workspace: Option<String>,
        model: Option<String>,
        role: Option<String>,
        guidelines: Option<String>,
        response_style: Option<String>,
        extra: Option<String>,
        max_tool_rounds: Option<usize>,
    ) -> PyResult<()> {
        let opts = py_build_team_member_options(workspace, model, role, guidelines, response_style, extra, max_tool_rounds);
        self.inner
            .blocking_lock()
            .add_reviewer(&agent_name, opts)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    /// Bind a `Session` to a team member.
    ///
    /// The session acts as the LLM executor for that member's role.
    ///
    /// Args:
    ///     member_id: The member ID (must match a member added to the team)
    ///     session: A `Session` object created from `Agent.session()`
    fn bind_session(&self, member_id: String, session: &PySession) -> PyResult<()> {
        let session_arc = session.inner.clone();
        self.inner
            .blocking_lock()
            .bind_session(&member_id, session_arc)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    /// Bind a team member to a named agent definition.
    ///
    /// Loads the agent by name from built-in agents and optionally from
    /// additional directories, then creates and binds a session with the
    /// agent's permissions, system prompt, model, and step limit applied.
    ///
    /// Args:
    ///     member_id: The member ID (must match a member added to the team)
    ///     agent: The `Agent` to create the session from
    ///     workspace: Path to the workspace directory
    ///     agent_name: Name of the agent to load (e.g. "explore", "general")
    ///     agent_dirs: Optional list of directories to scan for agent files
    #[pyo3(signature = (member_id, agent, workspace, agent_name, agent_dirs=None))]
    fn bind_agent(
        &self,
        member_id: String,
        agent: &PyAgent,
        workspace: String,
        agent_name: String,
        agent_dirs: Option<Vec<String>>,
    ) -> PyResult<()> {
        let registry = a3s_code_core::AgentRegistry::new();
        for dir in agent_dirs.unwrap_or_default() {
            let agents =
                a3s_code_core::load_agents_from_dir(std::path::Path::new(&dir));
            for agent_def in agents {
                registry.register(agent_def);
            }
        }
        self.inner
            .blocking_lock()
            .bind_agent(&member_id, &agent.inner, &workspace, &agent_name, &registry)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    /// Get the shared task board for inspection.
    fn task_board(&self) -> PyTeamTaskBoard {
        PyTeamTaskBoard {
            inner: self.inner.blocking_lock().task_board(),
        }
    }

    /// Run the Lead → Worker → Reviewer workflow until all tasks are done.
    ///
    /// 1. The Lead member decomposes `goal` into tasks via JSON response.
    /// 2. Worker members concurrently claim and execute tasks.
    /// 3. The Reviewer member approves or rejects completed tasks.
    /// 4. Rejected tasks re-enter the work queue for retry.
    ///
    /// Args:
    ///     goal: High-level goal to decompose and execute
    ///
    /// Returns:
    ///     `TeamRunResult` with done_tasks, rejected_tasks, rounds
    fn run_until_done(&self, py: Python<'_>, goal: String) -> PyResult<PyTeamRunResult> {
        let runner = self.inner.clone();
        let result = py
            .allow_threads(move || {
                get_runtime()
                    .block_on(async move { runner.lock().await.run_until_done(&goal).await })
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Team execution failed: {e}")))?;
        Ok(PyTeamRunResult {
            done_tasks: result
                .done_tasks
                .into_iter()
                .map(PyTeamTask::from)
                .collect(),
            rejected_tasks: result
                .rejected_tasks
                .into_iter()
                .map(PyTeamTask::from)
                .collect(),
            rounds: result.rounds,
        })
    }

    fn __repr__(&self) -> String {
        "TeamRunner(...)".to_string()
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
// Agent Orchestrator - Main-Sub Agent Coordination
// ============================================================================

/// SubAgent configuration for orchestrator.
#[pyclass(name = "SubAgentConfig")]
#[derive(Clone)]
struct PySubAgentConfig {
    inner: RustSubAgentConfig,
}

#[pymethods]
impl PySubAgentConfig {
    #[new]
    #[pyo3(signature = (agent_type, prompt, description=None, permissive=false, permissive_deny=None, max_steps=None, timeout_ms=None, parent_id=None, workspace=None, agent_dirs=None, lane_config=None))]
    fn new(
        agent_type: String,
        prompt: String,
        description: Option<String>,
        permissive: bool,
        permissive_deny: Option<Vec<String>>,
        max_steps: Option<usize>,
        timeout_ms: Option<u64>,
        parent_id: Option<String>,
        workspace: Option<String>,
        agent_dirs: Option<Vec<String>>,
        lane_config: Option<PySessionQueueConfig>,
    ) -> Self {
        let mut config = RustSubAgentConfig::new(agent_type, prompt);
        if let Some(desc) = description {
            config = config.with_description(desc);
        }
        config = config.with_permissive(permissive);
        if let Some(deny) = permissive_deny {
            config = config.with_permissive_deny(deny);
        }
        if let Some(steps) = max_steps {
            config = config.with_max_steps(steps);
        }
        if let Some(timeout) = timeout_ms {
            config = config.with_timeout_ms(timeout);
        }
        if let Some(parent) = parent_id {
            config = config.with_parent_id(parent);
        }
        if let Some(ws) = workspace {
            config = config.with_workspace(ws);
        }
        if let Some(dirs) = agent_dirs {
            config = config.with_agent_dirs(dirs);
        }
        if let Some(lc) = lane_config {
            config = config.with_lane_config(lc.inner);
        }
        Self { inner: config }
    }

    fn __repr__(&self) -> String {
        format!(
            "SubAgentConfig(agent_type={:?}, permissive={}, max_steps={:?})",
            self.inner.agent_type, self.inner.permissive, self.inner.max_steps
        )
    }
}

/// Unified agent slot — used for both standalone subagents and team members.
///
/// When `role` is None the slot describes a standalone subagent.
/// Valid role values: "lead", "worker", "reviewer".
#[pyclass(name = "AgentSlot")]
#[derive(Clone)]
struct PyAgentSlot {
    inner: RustAgentSlot,
}

#[pymethods]
impl PyAgentSlot {
    #[new]
    #[pyo3(signature = (agent_type, prompt, role=None, description=None, permissive=false, permissive_deny=None, max_steps=None, timeout_ms=None, parent_id=None, workspace=None, agent_dirs=None, lane_config=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        agent_type: String,
        prompt: String,
        role: Option<String>,
        description: Option<String>,
        permissive: bool,
        permissive_deny: Option<Vec<String>>,
        max_steps: Option<usize>,
        timeout_ms: Option<u64>,
        parent_id: Option<String>,
        workspace: Option<String>,
        agent_dirs: Option<Vec<String>>,
        lane_config: Option<PySessionQueueConfig>,
    ) -> Self {
        let rust_role = role.as_deref().and_then(|r| match r {
            "lead" => Some(RustTeamRole::Lead),
            "worker" => Some(RustTeamRole::Worker),
            "reviewer" => Some(RustTeamRole::Reviewer),
            _ => None,
        });
        let mut slot = RustAgentSlot::new(agent_type, prompt);
        if let Some(r) = rust_role {
            slot = slot.with_role(r);
        }
        if let Some(desc) = description {
            slot = slot.with_description(desc);
        }
        slot = slot.with_permissive(permissive);
        if let Some(deny) = permissive_deny {
            slot = slot.with_permissive_deny(deny);
        }
        if let Some(steps) = max_steps {
            slot = slot.with_max_steps(steps);
        }
        if let Some(timeout) = timeout_ms {
            slot = slot.with_timeout_ms(timeout);
        }
        if let Some(parent) = parent_id {
            slot = slot.with_parent_id(parent);
        }
        if let Some(ws) = workspace {
            slot = slot.with_workspace(ws);
        }
        if let Some(dirs) = agent_dirs {
            slot = slot.with_agent_dirs(dirs);
        }
        if let Some(lc) = lane_config {
            slot = slot.with_lane_config(lc.inner);
        }
        Self { inner: slot }
    }

    fn __repr__(&self) -> String {
        format!(
            "AgentSlot(agent_type={:?}, role={:?}, permissive={}, max_steps={:?})",
            self.inner.agent_type,
            self.inner.role.map(|r| format!("{}", r)),
            self.inner.permissive,
            self.inner.max_steps,
        )
    }
}

/// SubAgent handle for control and monitoring.
#[pyclass(name = "SubAgentHandle")]
struct PySubAgentHandle {
    inner: Arc<Mutex<RustSubAgentHandle>>,
}

#[pymethods]
impl PySubAgentHandle {
    #[getter]
    fn id(&self, py: Python<'_>) -> PyResult<String> {
        let handle = self.inner.clone();
        py.allow_threads(move || {
            get_runtime().block_on(async move {
                let h = handle.lock().await;
                Ok(h.id.clone())
            })
        })
    }

    /// Get current state (non-blocking).
    fn state(&self, py: Python<'_>) -> PyResult<String> {
        let handle = self.inner.clone();
        py.allow_threads(move || {
            get_runtime().block_on(async move {
                let h = handle.lock().await;
                let state = h.state_async().await;
                Ok(format!("{:?}", state))
            })
        })
    }

    /// Pause execution.
    fn pause(&self, py: Python<'_>) -> PyResult<()> {
        let handle = self.inner.clone();
        py.allow_threads(move || {
            get_runtime()
                .block_on(async move { handle.lock().await.pause().await })
                .map_err(|e| PyRuntimeError::new_err(format!("Pause failed: {e}")))
        })
    }

    /// Resume execution.
    fn resume(&self, py: Python<'_>) -> PyResult<()> {
        let handle = self.inner.clone();
        py.allow_threads(move || {
            get_runtime()
                .block_on(async move { handle.lock().await.resume().await })
                .map_err(|e| PyRuntimeError::new_err(format!("Resume failed: {e}")))
        })
    }

    /// Cancel execution.
    fn cancel(&self, py: Python<'_>) -> PyResult<()> {
        let handle = self.inner.clone();
        py.allow_threads(move || {
            get_runtime()
                .block_on(async move { handle.lock().await.cancel().await })
                .map_err(|e| PyRuntimeError::new_err(format!("Cancel failed: {e}")))
        })
    }

    /// Wait for completion and get result.
    fn wait(&self, py: Python<'_>) -> PyResult<String> {
        let handle = self.inner.clone();
        py.allow_threads(move || {
            get_runtime()
                .block_on(async move { handle.lock().await.wait().await })
                .map_err(|e| PyRuntimeError::new_err(format!("Wait failed: {e}")))
        })
    }

    fn __repr__(&self) -> String {
        "SubAgentHandle(...)".to_string()
    }
}

/// SubAgent activity type.
#[pyclass(name = "SubAgentActivity")]
#[derive(Clone)]
struct PySubAgentActivity {
    activity_type: String,
    data: Option<String>,
}

#[pymethods]
impl PySubAgentActivity {
    /// Get activity type (idle, calling_tool, requesting_llm, waiting_for_control).
    #[getter]
    fn activity_type(&self) -> String {
        self.activity_type.clone()
    }

    /// Get activity data (JSON string for tool args, etc.).
    #[getter]
    fn data(&self) -> Option<String> {
        self.data.clone()
    }

    fn __repr__(&self) -> String {
        format!("SubAgentActivity(type={})", self.activity_type)
    }
}

impl From<RustSubAgentActivity> for PySubAgentActivity {
    fn from(activity: RustSubAgentActivity) -> Self {
        match activity {
            RustSubAgentActivity::Idle => Self {
                activity_type: "idle".to_string(),
                data: None,
            },
            RustSubAgentActivity::CallingTool { tool_name, args } => Self {
                activity_type: "calling_tool".to_string(),
                data: Some(
                    serde_json::json!({
                        "tool_name": tool_name,
                        "args": args
                    })
                    .to_string(),
                ),
            },
            RustSubAgentActivity::RequestingLlm { message_count } => Self {
                activity_type: "requesting_llm".to_string(),
                data: Some(
                    serde_json::json!({
                        "message_count": message_count
                    })
                    .to_string(),
                ),
            },
            RustSubAgentActivity::WaitingForControl { reason } => Self {
                activity_type: "waiting_for_control".to_string(),
                data: Some(
                    serde_json::json!({
                        "reason": reason
                    })
                    .to_string(),
                ),
            },
        }
    }
}

/// SubAgent information with metadata and current activity.
#[pyclass(name = "SubAgentInfo")]
#[derive(Clone)]
struct PySubAgentInfo {
    id: String,
    agent_type: String,
    description: String,
    state: String,
    parent_id: Option<String>,
    created_at: u64,
    updated_at: u64,
    current_activity: Option<PySubAgentActivity>,
}

#[pymethods]
impl PySubAgentInfo {
    #[getter]
    fn id(&self) -> String {
        self.id.clone()
    }

    #[getter]
    fn agent_type(&self) -> String {
        self.agent_type.clone()
    }

    #[getter]
    fn description(&self) -> String {
        self.description.clone()
    }

    #[getter]
    fn state(&self) -> String {
        self.state.clone()
    }

    #[getter]
    fn parent_id(&self) -> Option<String> {
        self.parent_id.clone()
    }

    #[getter]
    fn created_at(&self) -> u64 {
        self.created_at
    }

    #[getter]
    fn updated_at(&self) -> u64 {
        self.updated_at
    }

    #[getter]
    fn current_activity(&self) -> Option<PySubAgentActivity> {
        self.current_activity.clone()
    }

    fn __repr__(&self) -> String {
        format!("SubAgentInfo(id={}, type={}, state={})", self.id, self.agent_type, self.state)
    }
}

impl From<RustSubAgentInfo> for PySubAgentInfo {
    fn from(info: RustSubAgentInfo) -> Self {
        Self {
            id: info.id,
            agent_type: info.agent_type,
            description: info.description,
            state: info.state,
            parent_id: info.parent_id,
            created_at: info.created_at,
            updated_at: info.updated_at,
            current_activity: info.current_activity.map(|a| a.into()),
        }
    }
}

/// Agent Orchestrator for main-sub agent coordination.
#[pyclass(name = "Orchestrator")]
struct PyOrchestrator {
    inner: Arc<Mutex<RustOrchestrator>>,
}

#[pymethods]
impl PyOrchestrator {
    /// Create a new orchestrator.
    ///
    /// Args:
    ///     agent: Optional `Agent` instance. When provided, spawned SubAgents
    ///            execute real LLM calls using the agent's configuration.
    ///            When omitted, SubAgents run in placeholder mode.
    #[staticmethod]
    #[pyo3(signature = (agent=None))]
    fn create(agent: Option<&PyAgent>) -> Self {
        let orch = match agent {
            Some(a) => RustOrchestrator::from_agent(a.inner.clone()),
            None => RustOrchestrator::new_memory(),
        };
        Self {
            inner: Arc::new(Mutex::new(orch)),
        }
    }

    /// Spawn a new SubAgent.
    fn spawn_subagent(
        &self,
        py: Python<'_>,
        config: PySubAgentConfig,
    ) -> PyResult<PySubAgentHandle> {
        let orch = self.inner.clone();
        let cfg = config.inner.clone();
        let handle = py
            .allow_threads(move || {
                get_runtime().block_on(async move { orch.lock().await.spawn_subagent(cfg).await })
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Spawn failed: {e}")))?;
        Ok(PySubAgentHandle {
            inner: Arc::new(Mutex::new(handle)),
        })
    }

    /// Spawn a subagent from a unified `AgentSlot` declaration.
    ///
    /// Convenience wrapper over `spawn_subagent` that accepts the unified slot
    /// type.  The `role` field is ignored for standalone spawning — use
    /// `run_team` for team-based workflows.
    fn spawn(&self, py: Python<'_>, slot: PyAgentSlot) -> PyResult<PySubAgentHandle> {
        let orch = self.inner.clone();
        let s = slot.inner.clone();
        let handle = py
            .allow_threads(move || {
                get_runtime().block_on(async move { orch.lock().await.spawn(s).await })
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Spawn failed: {e}")))?;
        Ok(PySubAgentHandle {
            inner: Arc::new(Mutex::new(handle)),
        })
    }

    /// Run a goal through a Lead → Worker → Reviewer team built from AgentSlots.
    ///
    /// Requires the orchestrator to be created with `create(agent=...)` — returns
    /// an error if no backing Agent is configured.  Each slot's `role` field
    /// determines its position in the team; slots without a role default to Worker.
    ///
    /// Returns a `TeamRunResult` with `done_tasks`, `rejected_tasks`, and `rounds`.
    fn run_team(
        &self,
        py: Python<'_>,
        goal: String,
        workspace: String,
        slots: Vec<PyAgentSlot>,
    ) -> PyResult<PyTeamRunResult> {
        let orch = self.inner.clone();
        let rust_slots: Vec<RustAgentSlot> = slots.into_iter().map(|s| s.inner).collect();
        let result = py
            .allow_threads(move || {
                get_runtime()
                    .block_on(async move { orch.lock().await.run_team(goal, workspace, rust_slots).await })
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Team run failed: {e}")))?;
        Ok(PyTeamRunResult {
            done_tasks: result.done_tasks.into_iter().map(PyTeamTask::from).collect(),
            rejected_tasks: result
                .rejected_tasks
                .into_iter()
                .map(PyTeamTask::from)
                .collect(),
            rounds: result.rounds,
        })
    }

    /// Get active SubAgent count.
    fn active_count(&self, py: Python<'_>) -> PyResult<usize> {
        let orch = self.inner.clone();
        py.allow_threads(move || {
            get_runtime().block_on(async move { Ok(orch.lock().await.active_count().await) })
        })
    }

    /// Get all SubAgent information list.
    fn list_subagents(&self, py: Python<'_>) -> PyResult<Vec<PySubAgentInfo>> {
        let orch = self.inner.clone();
        py.allow_threads(move || {
            get_runtime().block_on(async move {
                let infos = orch.lock().await.list_subagents().await;
                Ok(infos.into_iter().map(|i| i.into()).collect())
            })
        })
    }

    /// Get specific SubAgent information.
    fn get_subagent_info(&self, py: Python<'_>, id: String) -> PyResult<Option<PySubAgentInfo>> {
        let orch = self.inner.clone();
        py.allow_threads(move || {
            get_runtime().block_on(async move {
                Ok(orch.lock().await.get_subagent_info(&id).await.map(|i| i.into()))
            })
        })
    }

    /// Get all active SubAgent activities.
    fn get_active_activities(&self, py: Python<'_>) -> PyResult<Vec<(String, PySubAgentActivity)>> {
        let orch = self.inner.clone();
        py.allow_threads(move || {
            get_runtime().block_on(async move {
                let activities = orch.lock().await.get_active_activities().await;
                Ok(activities
                    .into_iter()
                    .map(|(id, activity)| (id, activity.into()))
                    .collect())
            })
        })
    }

    /// Get all SubAgent states.
    fn get_all_states(&self, py: Python<'_>) -> PyResult<Vec<(String, String)>> {
        let orch = self.inner.clone();
        py.allow_threads(move || {
            get_runtime().block_on(async move {
                let states = orch.lock().await.get_all_states().await;
                Ok(states
                    .into_iter()
                    .map(|(id, state)| (id, format!("{:?}", state)))
                    .collect())
            })
        })
    }

    /// Pause a SubAgent.
    fn pause_subagent(&self, py: Python<'_>, id: String) -> PyResult<()> {
        let orch = self.inner.clone();
        py.allow_threads(move || {
            get_runtime()
                .block_on(async move { orch.lock().await.pause_subagent(&id).await })
                .map_err(|e| PyRuntimeError::new_err(format!("Pause failed: {e}")))
        })
    }

    /// Resume a SubAgent.
    fn resume_subagent(&self, py: Python<'_>, id: String) -> PyResult<()> {
        let orch = self.inner.clone();
        py.allow_threads(move || {
            get_runtime()
                .block_on(async move { orch.lock().await.resume_subagent(&id).await })
                .map_err(|e| PyRuntimeError::new_err(format!("Resume failed: {e}")))
        })
    }

    /// Cancel a SubAgent.
    fn cancel_subagent(&self, py: Python<'_>, id: String) -> PyResult<()> {
        let orch = self.inner.clone();
        py.allow_threads(move || {
            get_runtime()
                .block_on(async move { orch.lock().await.cancel_subagent(&id).await })
                .map_err(|e| PyRuntimeError::new_err(format!("Cancel failed: {e}")))
        })
    }

    /// Wait for all SubAgents to complete.
    fn wait_all(&self, py: Python<'_>) -> PyResult<()> {
        let orch = self.inner.clone();
        py.allow_threads(move || {
            get_runtime()
                .block_on(async move { orch.lock().await.wait_all().await })
                .map_err(|e| PyRuntimeError::new_err(format!("Wait failed: {e}")))
        })
    }

    /// Return any external tasks currently waiting for the given SubAgent.
    ///
    /// Each item is a dict with keys: `task_id`, `command_type`, `payload` (JSON string), `lane`.
    /// Returns an empty list when no tasks are pending or the SubAgent is not found.
    fn pending_external_tasks_for(
        &self,
        py: Python<'_>,
        subagent_id: String,
    ) -> PyResult<Vec<PyObject>> {
        let orch = self.inner.clone();
        let tasks = py.allow_threads(move || {
            get_runtime().block_on(async move {
                orch.lock()
                    .await
                    .pending_external_tasks_for(&subagent_id)
                    .await
            })
        });
        let result: Vec<PyObject> = tasks
            .into_iter()
            .map(|t| {
                let dict = pyo3::types::PyDict::new(py);
                let _ = dict.set_item("task_id", &t.task_id);
                let _ = dict.set_item("command_type", &t.command_type);
                let _ = dict.set_item(
                    "payload",
                    serde_json::to_string(&t.payload).unwrap_or_default(),
                );
                let _ = dict.set_item("lane", format!("{:?}", t.lane));
                dict.into()
            })
            .collect();
        Ok(result)
    }

    /// Complete an external task dispatched to a remote worker.
    ///
    /// Returns `True` if the task was found and completed, `False` if no
    /// session with the given `subagent_id` is currently registered.
    ///
    /// `result` should be a JSON-serializable object or `None`.
    #[pyo3(signature = (subagent_id, task_id, success, result=None, error=None))]
    fn complete_external_task(
        &self,
        py: Python<'_>,
        subagent_id: String,
        task_id: String,
        success: bool,
        result: Option<String>,
        error: Option<String>,
    ) -> PyResult<bool> {
        let result_value = result
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or(serde_json::Value::Null);
        let ext_result = RustExternalTaskResult {
            success,
            result: result_value,
            error,
        };
        let orch = self.inner.clone();
        py.allow_threads(move || {
            Ok(get_runtime().block_on(async move {
                orch.lock()
                    .await
                    .complete_external_task(&subagent_id, &task_id, ext_result)
                    .await
            }))
        })
    }

    fn __repr__(&self) -> String {
        "Orchestrator(...)".to_string()
    }
}

// ============================================================================
// Python Module
// ============================================================================

/// A3S Code - Native AI coding agent library for Python.
#[pymodule]
fn a3s_code(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAgent>()?;
    m.add_class::<PySession>()?;
    m.add_class::<PyAgentResult>()?;
    m.add_class::<PyAgentEvent>()?;
    m.add_class::<PyToolResult>()?;
    m.add_class::<PyEventStream>()?;
    m.add_class::<PySkillInfo>()?;
    m.add_class::<PyFileMemoryStore>()?;
    m.add_class::<PyFileSessionStore>()?;
    m.add_class::<PyMemorySessionStore>()?;
    m.add_class::<PyDefaultSecurityProvider>()?;
    m.add_class::<PySessionOptions>()?;
    m.add_class::<PySessionQueueConfig>()?;
    m.add_class::<PySearchConfig>()?;
    m.add_class::<PySearchEngineConfig>()?;
    m.add_class::<PySearchHealthConfig>()?;
    m.add_class::<PyEventType>()?;
    // Agent Teams
    m.add_class::<PyTeamConfig>()?;
    m.add_class::<PyTeamTask>()?;
    m.add_class::<PyTeamRunResult>()?;
    m.add_class::<PyTeamTaskBoard>()?;
    m.add_class::<PyTeam>()?;
    m.add_class::<PyTeamRunner>()?;
    // Agent Orchestrator
    m.add_class::<PyOrchestrator>()?;
    m.add_class::<PySubAgentConfig>()?;
    m.add_class::<PyAgentSlot>()?;
    m.add_class::<PySubAgentHandle>()?;
    m.add_class::<PySubAgentInfo>()?;
    m.add_class::<PySubAgentActivity>()?;
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
            },
        })
        .collect()
}
