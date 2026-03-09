//! A3S Code Node.js Bindings
//!
//! Native Node.js addon via napi-rs that wraps `a3s-code-core`'s Agent API.
//!
//! ## Usage
//!
//! ```javascript
//! const { Agent } = require('@a3s-lab/code');
//!
//! const agent = await Agent.create('agent.hcl');
//! const session = agent.session('/my-project');
//!
//! const result = await session.send('What files handle auth?');
//! console.log(result.text);
//! ```

#[macro_use]
extern crate napi_derive;

use a3s_code_core::agent::{AgentEvent as RustAgentEvent, AgentResult as RustAgentResult};
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
use a3s_code_core::llm::{ContentBlock as RustContentBlock, Message as RustMessage};
use a3s_code_core::queue::{
    ExternalTaskResult as RustExternalTaskResult, LaneHandlerConfig as RustLaneHandlerConfig,
    SessionLane as RustSessionLane, SessionQueueConfig as RustSessionQueueConfig,
    TaskHandlerMode as RustTaskHandlerMode,
};
use a3s_code_core::{builtin_skills as rust_builtin_skills, SkillKind as RustSkillKind};
use a3s_code_core::{
    Agent as RustAgent, AgentSession as RustAgentSession, SessionOptions as RustSessionOptions,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tokio::runtime::Runtime;

// ============================================================================
// Tokio Runtime
// ============================================================================

fn get_runtime() -> &'static Runtime {
    use std::sync::OnceLock;
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| Runtime::new().expect("Failed to create tokio runtime"))
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
}

impl From<RustAgentResult> for AgentResult {
    fn from(r: RustAgentResult) -> Self {
        Self {
            text: r.text,
            tool_calls_count: r.tool_calls_count as u32,
            prompt_tokens: r.usage.prompt_tokens as u32,
            completion_tokens: r.usage.completion_tokens as u32,
            total_tokens: r.usage.total_tokens as u32,
        }
    }
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
            RustAgentEvent::TextDelta { text } => Self {
                text: Some(text),
                ..Self::empty("text_delta")
            },
            RustAgentEvent::TurnStart { turn } => Self {
                turn: Some(turn as u32),
                ..Self::empty("turn_start")
            },
            RustAgentEvent::TurnEnd { turn, usage } => Self {
                turn: Some(turn as u32),
                total_tokens: Some(usage.total_tokens as u32),
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
                total_tokens: Some(usage.total_tokens as u32),
                ..Self::empty("end")
            },
            RustAgentEvent::Error { message } => Self {
                error: Some(message),
                ..Self::empty("error")
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

// ============================================================================
// SessionOptions
// ============================================================================

#[napi(object)]
#[derive(Clone, Default)]
pub struct SessionOptions {
    /// Override the default model. Format: "provider/model" (e.g., "openai/gpt-4o").
    pub model: Option<String>,
    /// Enable built-in skills (7 skills: code-search, code-review, explain-code, find-bugs, builtin-tools, delegate-task, find-skills).
    pub builtin_skills: Option<bool>,
    /// Extra directories to scan for skill files (.md with YAML frontmatter).
    pub skill_dirs: Option<Vec<String>>,
    /// Extra directories to scan for agent files.
    pub agent_dirs: Option<Vec<String>>,
    /// Optional queue configuration for lane-based tool execution.
    pub queue_config: Option<SessionQueueConfig>,
    /// Allow all tools without HITL confirmation (default: false).
    pub permissive: Option<bool>,
    /// Enable planning mode (default: false).
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
    /// Session ID (auto-generated if not set).
    ///
    /// Set a stable ID so the session can be saved and resumed later:
    /// ```js
    /// agent.session('.', { sessionId: 'my-session', sessionStore: new FileSessionStore('./sessions'), autoSave: true });
    /// // Later:
    /// agent.resumeSession('my-session', { sessionStore: new FileSessionStore('./sessions') });
    /// ```
    pub session_id: Option<String>,
    /// Automatically save the session to the configured store after each turn (default: false).
    pub auto_save: Option<bool>,
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

/// Configuration for the session lane queue.
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

fn js_queue_config_to_rust(config: &SessionQueueConfig) -> RustSessionQueueConfig {
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
            if let (Ok(lane), Ok(mode)) = (
                parse_lane(lane_str),
                parse_handler_mode(&handler.mode),
            ) {
                let lane_cfg = RustLaneHandlerConfig {
                    mode,
                    timeout_ms: handler.timeout_ms.map(|ms| ms as u64).unwrap_or(60_000),
                };
                c.lane_handlers.insert(lane, lane_cfg);
            }
        }
    }
    c
}

fn parse_lane(lane: &str) -> napi::Result<RustSessionLane> {
    match lane {
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
    match mode {
        "internal" => Ok(RustTaskHandlerMode::Internal),
        "external" => Ok(RustTaskHandlerMode::External),
        "hybrid" => Ok(RustTaskHandlerMode::Hybrid),
        _ => Err(napi::Error::from_reason(format!(
            "Invalid handler mode '{}'. Must be: internal, external, or hybrid",
            mode
        ))),
    }
}

/// Convert JS `TeamMemberOptions` to the Rust core type.
fn js_team_member_options_to_rust(opts: TeamMemberOptions) -> a3s_code_core::TeamMemberOptions {
    let has_slots =
        opts.role.is_some() || opts.guidelines.is_some() || opts.response_style.is_some() || opts.extra.is_some();
    let prompt_slots = has_slots.then(|| a3s_code_core::SystemPromptSlots {
        role: opts.role,
        guidelines: opts.guidelines,
        response_style: opts.response_style,
        extra: opts.extra,
    });
    a3s_code_core::TeamMemberOptions {
        workspace: opts.workspace,
        model: opts.model,
        prompt_slots,
        max_tool_rounds: opts.max_tool_rounds.map(|n| n as usize),
    }
}

/// Build RustSessionOptions from JS SessionOptions.
fn js_session_options_to_rust(options: Option<SessionOptions>) -> RustSessionOptions {
    let Some(o) = options else {
        return RustSessionOptions::new();
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
    if let Some(qc) = o.queue_config {
        opts = opts.with_queue_config(js_queue_config_to_rust(&qc));
    }
    if o.permissive.unwrap_or(false) {
        opts = opts.with_permissive_policy();
    }
    if o.planning.unwrap_or(false) {
        opts = opts.with_planning(true);
    }
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
    // Build prompt slots if any slot is set
    if o.role.is_some() || o.guidelines.is_some() || o.response_style.is_some() || o.extra.is_some()
    {
        let slots = a3s_code_core::SystemPromptSlots {
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
                if let Some(parsed) = a3s_code_core::Skill::parse(&raw) {
                    registry.register_unchecked(std::sync::Arc::new(parsed));
                }
            }
            opts = opts.with_skill_registry(std::sync::Arc::new(registry));
        }
    }
    if let Some(r) = o.max_tool_rounds {
        opts = opts.with_max_tool_rounds(r as usize);
    }
    if let Some(id) = o.session_id {
        opts = opts.with_session_id(id);
    }
    if o.auto_save.unwrap_or(false) {
        opts = opts.with_auto_save(true);
    }
    opts
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
    /// @param configSource - Path to .hcl/.json file, or inline JSON/HCL string
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
        let rust_opts = js_session_options_to_rust(options);
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
        let opts = js_session_options_to_rust(Some(options));
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
    #[napi]
    pub fn session_for_agent(
        &self,
        workspace: String,
        agent_name: String,
        agent_dirs: Option<Vec<String>>,
    ) -> napi::Result<Session> {
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
            .ok_or_else(|| napi::Error::from_reason(format!("agent '{}' not found", agent_name)))?;
        let session = self
            .inner
            .session_for_agent(workspace, &def, None)
            .map_err(|e| napi::Error::from_reason(format!("{e}")))?;
        Ok(Session {
            inner: Arc::new(session),
        })
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

/// Workspace-bound session. All LLM and tool operations happen here.
#[napi]
pub struct Session {
    inner: Arc<RustAgentSession>,
}

#[napi]
impl Session {
    /// Send a prompt and wait for the complete response.
    ///
    /// @param prompt - The prompt to send
    /// @param history - Optional conversation history
    #[napi]
    pub async fn send(
        &self,
        prompt: String,
        history: Option<Vec<MessageObject>>,
    ) -> napi::Result<AgentResult> {
        let rust_history = history.map(|h| js_messages_to_rust(&h)).transpose()?;
        let session = self.inner.clone();
        let result = get_runtime()
            .spawn(async move { session.send(&prompt, rust_history.as_deref()).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Agent execution failed: {e}")))?;
        Ok(AgentResult::from(result))
    }

    /// Send a prompt and get a streaming event iterator.
    ///
    /// Returns an `EventStream`. Use `for await (const event of stream)` or call `.next()` manually.
    ///
    /// @param prompt - The prompt to send
    /// @param history - Optional conversation history
    #[napi]
    pub async fn stream(
        &self,
        prompt: String,
        history: Option<Vec<MessageObject>>,
    ) -> napi::Result<EventStream> {
        let rust_history = history.map(|h| js_messages_to_rust(&h)).transpose()?;
        let session = self.inner.clone();
        let (rx, _handle) = get_runtime()
            .spawn(async move { session.stream(&prompt, rust_history.as_deref()).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Failed to start stream: {e}")))?;

        Ok(EventStream {
            rx: Arc::new(tokio::sync::Mutex::new(rx)),
            done: Arc::new(AtomicBool::new(false)),
        })
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

    /// Execute a tool by name, bypassing the LLM.
    #[napi]
    pub async fn tool(&self, name: String, args: serde_json::Value) -> napi::Result<ToolResult> {
        let session = self.inner.clone();
        let result = get_runtime()
            .spawn(async move { session.tool(&name, args).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Tool execution failed: {e}")))?;
        Ok(ToolResult {
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
    /// All agent-type arguments default to `"general"` when omitted.
    ///
    /// @returns `ToolResult` whose `output` contains the formatted team run summary.
    #[napi]
    pub async fn run_team(
        &self,
        goal: String,
        lead_agent: Option<String>,
        worker_agent: Option<String>,
        reviewer_agent: Option<String>,
        max_steps: Option<u32>,
    ) -> napi::Result<ToolResult> {
        let mut args = serde_json::json!({
            "goal": goal,
            "lead_agent": lead_agent.unwrap_or_else(|| "general".to_string()),
            "worker_agent": worker_agent.unwrap_or_else(|| "general".to_string()),
            "reviewer_agent": reviewer_agent.unwrap_or_else(|| "general".to_string()),
        });
        if let Some(steps) = max_steps {
            args["max_steps"] = serde_json::json!(steps);
        }
        let session = self.inner.clone();
        let result = get_runtime()
            .spawn(async move { session.tool("run_team", args).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("run_team failed: {e}")))?;
        Ok(ToolResult {
            name: result.name,
            output: result.output,
            exit_code: result.exit_code,
        })
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

    // ========================================================================
    // Queue API
    // ========================================================================

    /// Check if this session has a lane queue configured.
    #[napi]
    pub fn has_queue(&self) -> bool {
        self.inner.has_queue()
    }

    /// Configure a lane's handler mode.
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

    /// Complete an external task by ID.
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

    /// Get pending external tasks.
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

    /// Get queue statistics.
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

    /// Submit a JSON payload as a command to the session's lane queue.
    ///
    /// The payload is stored and returned as-is when the queue schedules the
    /// command. Returns a Promise that resolves to the payload value.
    ///
    /// @param lane - "control", "query", "execute", or "generate"
    /// @param payload - Any JSON-serializable value
    #[napi]
    pub async fn submit(
        &self,
        lane: String,
        payload: serde_json::Value,
    ) -> napi::Result<serde_json::Value> {
        let rust_lane = parse_lane(&lane)?;
        struct JsonCommand(serde_json::Value);
        #[async_trait::async_trait]
        impl a3s_code_core::queue::SessionCommand for JsonCommand {
            async fn execute(&self) -> anyhow::Result<serde_json::Value> {
                Ok(self.0.clone())
            }
            fn command_type(&self) -> &str {
                "json"
            }
        }
        let cmd = JsonCommand(payload);
        let rx = self
            .inner
            .submit(rust_lane, Box::new(cmd))
            .await
            .map_err(|e| napi::Error::from_reason(format!("Submit failed: {e}")))?;
        rx.await
            .map_err(|e| napi::Error::from_reason(format!("Command dropped: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Command failed: {e}")))
    }

    /// Submit multiple JSON payloads as a batch to the session's lane queue.
    ///
    /// More efficient than calling `submit()` in a loop. Returns a Promise that
    /// resolves to an array of results in the same order as the input payloads.
    ///
    /// @param lane - "control", "query", "execute", or "generate"
    /// @param payloads - Array of JSON-serializable values
    #[napi]
    pub async fn submit_batch(
        &self,
        lane: String,
        payloads: Vec<serde_json::Value>,
    ) -> napi::Result<Vec<serde_json::Value>> {
        let rust_lane = parse_lane(&lane)?;
        struct JsonCommand(serde_json::Value);
        #[async_trait::async_trait]
        impl a3s_code_core::queue::SessionCommand for JsonCommand {
            async fn execute(&self) -> anyhow::Result<serde_json::Value> {
                Ok(self.0.clone())
            }
            fn command_type(&self) -> &str {
                "json"
            }
        }
        let commands: Vec<Box<dyn a3s_code_core::queue::SessionCommand>> = payloads
            .into_iter()
            .map(|p| Box::new(JsonCommand(p)) as Box<dyn a3s_code_core::queue::SessionCommand>)
            .collect();
        let receivers = self
            .inner
            .submit_batch(rust_lane, commands)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Submit batch failed: {e}")))?;
        let mut results = Vec::with_capacity(receivers.len());
        for rx in receivers {
            let val = rx
                .await
                .map_err(|e| napi::Error::from_reason(format!("Command dropped: {e}")))?
                .map_err(|e| napi::Error::from_reason(format!("Command failed: {e}")))?;
            results.push(val);
        }
        Ok(results)
    }

    /// Get dead letters from the DLQ.
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

        let session = self.inner.clone();
        let count = session
            .add_mcp_server(McpServerConfig {
                name,
                transport: transport_config,
                enabled: true,
                env: env.unwrap_or_default(),
                oauth: None,
                tool_timeout_secs: 60,
            })
            .await
            .map_err(|e| napi::Error::from_reason(format!("add_mcp_server failed: {e}")))?;
        Ok(count as u32)
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

    /// Return the names of all tools currently registered on this session.
    ///
    /// @returns Array of tool name strings
    #[napi]
    pub fn tool_names(&self) -> Vec<String> {
        self.inner.tool_names()
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
        #[napi(ts_arg_type = "((event: Record<string, unknown>) => { action: string; reason?: string } | null | undefined) | null | undefined")]
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
            let tsfn: ThreadsafeFunction<serde_json::Value, ErrorStrategy::CalleeHandled> =
                js_fn.create_threadsafe_function(0, |ctx: ThreadSafeCallContext<serde_json::Value>| {
                    let js_val = ctx.env.to_js_value(&ctx.value)?;
                    Ok(vec![js_val])
                })?;
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

    /// Schedule a recurring prompt to fire at a given interval.
    ///
    /// This is the programmatic equivalent of `/loop <interval>s <prompt>`.
    /// The scheduled prompt runs automatically after each `send()` call when it is due.
    ///
    /// @param prompt - The prompt to send at each interval
    /// @param intervalSecs - Interval in seconds (minimum: 1)
    /// @returns 8-char hex task ID (use with `cancelScheduledTask`)
    #[napi]
    pub fn schedule_task(&self, prompt: String, interval_secs: u32) -> napi::Result<String> {
        self.inner
            .cron_scheduler()
            .create_task(
                prompt,
                std::time::Duration::from_secs(interval_secs as u64),
                true,
            )
            .map_err(napi::Error::from_reason)
    }

    /// List all active scheduled tasks for this session.
    ///
    /// @returns Array of ScheduledTaskInfo objects sorted by task ID
    #[napi]
    pub fn list_scheduled_tasks(&self) -> Vec<ScheduledTaskInfo> {
        self.inner
            .cron_scheduler()
            .list_tasks()
            .into_iter()
            .map(|t| ScheduledTaskInfo {
                id: t.id,
                prompt: t.prompt,
                interval_secs: t.interval_secs as i64,
                recurring: t.recurring,
                fire_count: t.fire_count as i64,
                next_fire_in_secs: t.next_fire_in_secs as i64,
            })
            .collect()
    }

    /// Cancel a scheduled task by ID.
    ///
    /// @param id - Task ID returned by `scheduleTask` or listed by `listScheduledTasks`
    /// @returns `true` if the task was found and cancelled
    #[napi]
    pub fn cancel_scheduled_task(&self, id: String) -> bool {
        self.inner.cron_scheduler().cancel_task(&id)
    }
}

// ============================================================================
// Slash Command Types
// ============================================================================

/// Metadata about a registered slash command.
#[napi(object)]
#[derive(Clone)]
pub struct CommandInfo {
    /// Command name without the leading `/` (e.g., `"loop"`, `"help"`)
    pub name: String,
    /// Short description shown in `/help`
    pub description: String,
    /// Optional usage hint (e.g., `"/loop [interval] <prompt> [every <interval>]"`)
    pub usage: Option<String>,
}

/// Info about an active scheduled task.
#[napi(object)]
#[derive(Clone)]
pub struct ScheduledTaskInfo {
    /// 8-char hex task ID
    pub id: String,
    /// The prompt sent at each interval
    pub prompt: String,
    /// Interval between fires in seconds
    pub interval_secs: i64,
    /// Whether the task repeats (always `true` for tasks created via `/loop`)
    pub recurring: bool,
    /// Number of times this task has fired so far
    pub fire_count: i64,
    /// Seconds until the next fire (0 if overdue)
    pub next_fire_in_secs: i64,
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

fn metrics_snapshot_to_json(snapshot: Option<a3s_code_core::MetricsSnapshot>) -> serde_json::Value {
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
// Agent Teams
// ============================================================================

/// Team configuration.
#[napi(object)]
#[derive(Clone)]
pub struct TeamConfig {
    /// Maximum concurrent tasks on the board (default: 50).
    pub max_tasks: u32,
    /// Message channel buffer size (default: 128).
    pub channel_buffer: u32,
    /// Maximum coordinator rounds before `runUntilDone` exits (default: 10).
    pub max_rounds: u32,
    /// Worker/Reviewer polling interval in milliseconds (default: 200).
    pub poll_interval_ms: u32,
}

impl From<TeamConfig> for RustTeamConfig {
    fn from(c: TeamConfig) -> Self {
        Self {
            max_tasks: c.max_tasks as usize,
            channel_buffer: c.channel_buffer as usize,
            max_rounds: c.max_rounds as usize,
            poll_interval_ms: c.poll_interval_ms as u64,
        }
    }
}

/// A task snapshot from the team board (read-only).
#[napi(object)]
#[derive(Clone)]
pub struct TeamTask {
    pub id: String,
    pub description: String,
    pub posted_by: String,
    pub assigned_to: Option<String>,
    /// Task status: "open", "in_progress", "in_review", "done", or "rejected".
    pub status: String,
    pub result: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

impl From<RustTeamTask> for TeamTask {
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

/// Result returned by `TeamRunner.runUntilDone()`.
#[napi(object)]
pub struct TeamRunResult {
    pub done_tasks: Vec<TeamTask>,
    pub rejected_tasks: Vec<TeamTask>,
    pub rounds: u32,
}

fn js_parse_task_status(s: &str) -> napi::Result<RustTaskStatus> {
    match s {
        "open" => Ok(RustTaskStatus::Open),
        "in_progress" => Ok(RustTaskStatus::InProgress),
        "in_review" => Ok(RustTaskStatus::InReview),
        "done" => Ok(RustTaskStatus::Done),
        "rejected" => Ok(RustTaskStatus::Rejected),
        _ => Err(napi::Error::from_reason(format!(
            "Invalid task status '{}'. Expected: open, in_progress, in_review, done, rejected",
            s
        ))),
    }
}

fn js_parse_team_role(s: &str) -> napi::Result<RustTeamRole> {
    match s {
        "lead" => Ok(RustTeamRole::Lead),
        "worker" => Ok(RustTeamRole::Worker),
        "reviewer" => Ok(RustTeamRole::Reviewer),
        _ => Err(napi::Error::from_reason(format!(
            "Invalid role '{}'. Expected: lead, worker, reviewer",
            s
        ))),
    }
}

/// Shared task board for team coordination.
///
/// Use `Team.taskBoard()` or `TeamRunner.taskBoard()` to access the board.
#[napi]
pub struct TeamTaskBoard {
    inner: Arc<RustTeamTaskBoard>,
}

#[napi]
impl TeamTaskBoard {
    /// Post a new task. Returns the task ID, or null if the board is full.
    ///
    /// @param description - Task description
    /// @param postedBy - Member ID posting the task
    /// @param assignTo - Optional member ID to pre-assign the task to
    #[napi]
    pub fn post(
        &self,
        description: String,
        posted_by: String,
        assign_to: Option<String>,
    ) -> Option<String> {
        self.inner
            .post(&description, &posted_by, assign_to.as_deref())
    }

    /// Claim the next open or rejected task for a member.
    #[napi]
    pub fn claim(&self, member_id: String) -> Option<TeamTask> {
        self.inner.claim(&member_id).map(TeamTask::from)
    }

    /// Mark a task as complete with a result. Returns true if found.
    #[napi]
    pub fn complete(&self, task_id: String, result: String) -> bool {
        self.inner.complete(&task_id, &result)
    }

    /// Approve a task (reviewer action). Returns true if the task was in InReview state.
    #[napi]
    pub fn approve(&self, task_id: String) -> bool {
        self.inner.approve(&task_id)
    }

    /// Reject a task back to open (reviewer action). Returns true if found.
    #[napi]
    pub fn reject(&self, task_id: String) -> bool {
        self.inner.reject(&task_id)
    }

    /// Get a task by ID.
    #[napi]
    pub fn get(&self, task_id: String) -> Option<TeamTask> {
        self.inner.get(&task_id).map(TeamTask::from)
    }

    /// Get all tasks with the given status string.
    ///
    /// @param status - "open", "in_progress", "in_review", "done", or "rejected"
    #[napi]
    pub fn by_status(&self, status: String) -> napi::Result<Vec<TeamTask>> {
        let s = js_parse_task_status(&status)?;
        Ok(self
            .inner
            .by_status(s)
            .into_iter()
            .map(TeamTask::from)
            .collect())
    }

    /// Get all tasks assigned to a member.
    #[napi]
    pub fn by_assignee(&self, member_id: String) -> Vec<TeamTask> {
        self.inner
            .by_assignee(&member_id)
            .into_iter()
            .map(TeamTask::from)
            .collect()
    }

    /// Summary stats as `{ open, inProgress, inReview, done, rejected }`.
    #[napi]
    pub fn stats(&self) -> serde_json::Value {
        let (open, in_progress, in_review, done, rejected) = self.inner.stats();
        serde_json::json!({
            "open": open,
            "inProgress": in_progress,
            "inReview": in_review,
            "done": done,
            "rejected": rejected,
            "total": self.inner.len(),
        })
    }

    /// Total number of tasks on the board.
    #[napi(getter)]
    pub fn len(&self) -> u32 {
        self.inner.len() as u32
    }

    /// True if the board has no tasks.
    #[napi(getter)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// Multi-agent team coordinator.
///
/// Create the team, add members, then pass it to `TeamRunner` to execute.
///
/// @example
/// ```js
/// const team = new Team("refactor-auth");
/// team.addMember("lead", "lead");
/// team.addMember("worker-1", "worker");
/// team.addMember("reviewer", "reviewer");
/// const runner = new TeamRunner(team);
/// runner.bindSession("lead", leadSession);
/// const result = await runner.runUntilDone("Refactor the auth module");
/// ```
#[napi]
pub struct Team {
    inner: Arc<tokio::sync::Mutex<Option<RustAgentTeam>>>,
}

#[napi]
impl Team {
    /// Create a new team.
    ///
    /// @param name - Team name
    /// @param config - Optional `TeamConfig` (uses defaults if omitted)
    #[napi(constructor)]
    pub fn new(name: String, config: Option<TeamConfig>) -> Self {
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
    /// @param memberId - Unique member identifier
    /// @param role - "lead", "worker", or "reviewer"
    #[napi]
    pub fn add_member(&self, member_id: String, role: String) -> napi::Result<()> {
        let role = js_parse_team_role(&role)?;
        let mut guard = self.inner.blocking_lock();
        let team = guard
            .as_mut()
            .ok_or_else(|| napi::Error::from_reason("Team has been consumed by a TeamRunner"))?;
        team.add_member(&member_id, role);
        Ok(())
    }

    /// Remove a member. Returns true if the member was found.
    #[napi]
    pub fn remove_member(&self, member_id: String) -> bool {
        let mut guard = self.inner.blocking_lock();
        guard
            .as_mut()
            .map(|t| t.remove_member(&member_id))
            .unwrap_or(false)
    }

    /// Number of registered members.
    #[napi(getter)]
    pub fn member_count(&self) -> u32 {
        self.inner
            .blocking_lock()
            .as_ref()
            .map(|t| t.member_count())
            .unwrap_or(0) as u32
    }

    /// Get the shared task board for inspection.
    #[napi]
    pub fn task_board(&self) -> napi::Result<TeamTaskBoard> {
        let guard = self.inner.blocking_lock();
        let team = guard
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("Team has been consumed by a TeamRunner"))?;
        Ok(TeamTaskBoard {
            inner: team.task_board_arc(),
        })
    }
}

/// Per-member overrides for `TeamRunner.addLead`, `addWorker`, and `addReviewer`.
///
/// All fields are optional. Unset fields inherit from the agent definition
/// file (role-level config) and ultimately from the `Agent` base config:
///
/// ```
/// TeamMemberOptions  →  AgentDefinition (.yaml/.md)  →  Agent (config.hcl)
/// ```
///
/// Specifically:
/// - `model`: unset → inherits agent definition model → inherits Agent default model
/// - `extra`: unset → inherits agent definition `prompt` field
/// - `role`, `guidelines`, `responseStyle`: unset → empty (no definition-level equivalent)
/// - `workspace`: unset → inherits the workspace passed to `TeamRunner.create`
/// - `maxToolRounds`: unset → inherits agent definition `max_steps` → inherits Agent config
#[napi(object)]
#[derive(Clone, Default)]
pub struct TeamMemberOptions {
    /// Override the workspace for this member.
    ///
    /// Set this to an isolated git worktree path so concurrent workers do not
    /// conflict with each other on the filesystem.
    /// Falls back to the workspace supplied to `TeamRunner.create`.
    pub workspace: Option<String>,
    /// Model override. Format: `"provider/model"` (e.g. `"openai/gpt-4o"`).
    /// Falls back to the agent definition model, then the Agent default model.
    pub model: Option<String>,
    /// Custom role/identity prepended before the core agentic prompt.
    ///
    /// Example: `"You are a senior Python developer specializing in FastAPI."`
    /// No definition-level default — omit to use the standard agent identity.
    pub role: Option<String>,
    /// Custom coding guidelines appended after the core prompt.
    ///
    /// Example: `"Always write unit tests. Follow PEP 8."`
    /// No definition-level default — omit to use no extra guidelines.
    pub guidelines: Option<String>,
    /// Custom response style (replaces the default Response Format section).
    /// No definition-level default — omit to use the standard response format.
    pub response_style: Option<String>,
    /// Freeform extra instructions appended at the very end of the system prompt.
    /// Falls back to the agent definition `prompt` field when unset.
    pub extra: Option<String>,
    /// Override maximum number of tool-call rounds for this member's session.
    /// Falls back to the agent definition `max_steps`, then the Agent config.
    pub max_tool_rounds: Option<u32>,
}

/// Binds an agent team to real `Session` executors and runs the workflow.
///
/// The team object is consumed on construction.
///
/// @example
/// ```js
/// const runner = new TeamRunner(team);
/// runner.bindSession("lead", leadSession);
/// runner.bindSession("worker-1", workerSession);
/// runner.bindSession("reviewer", reviewerSession);
/// const result = await runner.runUntilDone("Build the feature");
/// for (const task of result.doneTasks) {
///   console.log(task.id, task.result);
/// }
/// ```
#[napi]
pub struct TeamRunner {
    inner: Arc<tokio::sync::Mutex<RustTeamRunner>>,
}

#[napi]
impl TeamRunner {
    /// Create a runner from a team.
    ///
    /// The team is consumed: further calls on the original `Team` object will throw.
    #[napi(constructor)]
    pub fn new(team: &Team) -> napi::Result<Self> {
        let mut guard = team.inner.blocking_lock();
        let rust_team = guard.take().ok_or_else(|| {
            napi::Error::from_reason("Team has already been consumed by another TeamRunner")
        })?;
        Ok(Self {
            inner: Arc::new(tokio::sync::Mutex::new(RustTeamRunner::new(rust_team))),
        })
    }

    /// Create a runner with a default agent context.
    ///
    /// Stores the agent, workspace, and agent directories once so that
    /// subsequent calls to `addLead`, `addWorker`, and `addReviewer` do not
    /// need to repeat them.
    ///
    /// @param agent - The `Agent` to create sessions from
    /// @param workspace - Path to the workspace directory shared by all members
    /// @param agentDirs - Directories to scan for agent definition files
    #[napi(factory)]
    pub fn create(
        agent: &Agent,
        workspace: String,
        agent_dirs: Option<Vec<String>>,
    ) -> napi::Result<Self> {
        let registry = a3s_code_core::AgentRegistry::new();
        for dir in agent_dirs.unwrap_or_default() {
            let agents = a3s_code_core::load_agents_from_dir(std::path::Path::new(&dir));
            for agent_def in agents {
                registry.register(agent_def);
            }
        }
        let team = a3s_code_core::AgentTeam::new("team", a3s_code_core::TeamConfig::default());
        let runner =
            RustTeamRunner::with_agent(team, agent.inner.clone(), &workspace, Arc::new(registry));
        Ok(Self {
            inner: Arc::new(tokio::sync::Mutex::new(runner)),
        })
    }

    /// Add a Lead member bound to the named agent definition.
    ///
    /// Requires the runner to have been created with `TeamRunner.create(...)`.
    /// The member ID is fixed to `"lead"`.
    ///
    /// Unset fields in `opts` inherit from the agent definition, then from the
    /// `Agent` base config (see `TeamMemberOptions` for the full inheritance chain).
    ///
    /// @param agentName - Name of the agent definition (e.g. `"orchestrator"`)
    /// @param opts - Optional per-member overrides; omit to use agent definition defaults
    #[napi]
    pub fn add_lead(
        &self,
        agent_name: String,
        opts: Option<TeamMemberOptions>,
    ) -> napi::Result<()> {
        let rust_opts = opts.map(js_team_member_options_to_rust);
        self.inner
            .blocking_lock()
            .add_lead(&agent_name, rust_opts)
            .map_err(|e| napi::Error::from_reason(format!("{e}")))
    }

    /// Add a Worker member bound to the named agent definition.
    ///
    /// Requires the runner to have been created with `TeamRunner.create(...)`.
    /// Member IDs are auto-generated as `"worker-1"`, `"worker-2"`, etc.
    /// Call this multiple times to add concurrent workers.
    ///
    /// Set `opts.workspace` to a git worktree path to give each worker an
    /// isolated filesystem so concurrent writes do not conflict.
    /// Unset fields inherit from the agent definition, then from the `Agent`
    /// base config (see `TeamMemberOptions` for the full inheritance chain).
    ///
    /// @param agentName - Name of the agent definition (e.g. `"general"`)
    /// @param opts - Optional per-member overrides; omit to use agent definition defaults
    #[napi]
    pub fn add_worker(
        &self,
        agent_name: String,
        opts: Option<TeamMemberOptions>,
    ) -> napi::Result<()> {
        let rust_opts = opts.map(js_team_member_options_to_rust);
        self.inner
            .blocking_lock()
            .add_worker(&agent_name, rust_opts)
            .map_err(|e| napi::Error::from_reason(format!("{e}")))
    }

    /// Add a Reviewer member bound to the named agent definition.
    ///
    /// Requires the runner to have been created with `TeamRunner.create(...)`.
    /// The member ID is fixed to `"reviewer"`.
    ///
    /// Unset fields in `opts` inherit from the agent definition, then from the
    /// `Agent` base config (see `TeamMemberOptions` for the full inheritance chain).
    ///
    /// @param agentName - Name of the agent definition (e.g. `"reviewer"`)
    /// @param opts - Optional per-member overrides; omit to use agent definition defaults
    #[napi]
    pub fn add_reviewer(
        &self,
        agent_name: String,
        opts: Option<TeamMemberOptions>,
    ) -> napi::Result<()> {
        let rust_opts = opts.map(js_team_member_options_to_rust);
        self.inner
            .blocking_lock()
            .add_reviewer(&agent_name, rust_opts)
            .map_err(|e| napi::Error::from_reason(format!("{e}")))
    }

    /// Bind a `Session` to a team member.
    ///
    /// @param memberId - The member ID (must match a member added to the team)
    /// @param session - A `Session` object from `Agent.session()`
    #[napi]
    pub fn bind_session(&self, member_id: String, session: &Session) -> napi::Result<()> {
        let session_arc = session.inner.clone();
        self.inner
            .blocking_lock()
            .bind_session(&member_id, session_arc)
            .map_err(|e| napi::Error::from_reason(format!("{e}")))
    }

    /// Bind a team member to a named agent definition.
    ///
    /// Loads the agent by name from built-in agents and optionally from
    /// additional directories, then creates and binds a session with the
    /// agent's permissions, system prompt, model, and step limit applied.
    ///
    /// @param memberId - The member ID (must match a member added to the team)
    /// @param agent - The `Agent` to create the session from
    /// @param workspace - Path to the workspace directory
    /// @param agentName - Name of the agent to load (e.g. "explore", "general")
    /// @param agentDirs - Optional directories to scan for agent files
    #[napi]
    pub fn bind_agent(
        &self,
        member_id: String,
        agent: &Agent,
        workspace: String,
        agent_name: String,
        agent_dirs: Option<Vec<String>>,
    ) -> napi::Result<()> {
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
            .map_err(|e| napi::Error::from_reason(format!("{e}")))
    }

    /// Get the shared task board for inspection.
    #[napi]
    pub fn task_board(&self) -> TeamTaskBoard {
        TeamTaskBoard {
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
    /// @param goal - High-level goal to decompose and execute
    /// @returns `TeamRunResult` with `doneTasks`, `rejectedTasks`, and `rounds`
    #[napi]
    pub async fn run_until_done(&self, goal: String) -> napi::Result<TeamRunResult> {
        let runner = self.inner.clone();
        let result = get_runtime()
            .spawn(async move { runner.lock().await.run_until_done(&goal).await })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Team execution failed: {e}")))?;
        Ok(TeamRunResult {
            done_tasks: result.done_tasks.into_iter().map(TeamTask::from).collect(),
            rejected_tasks: result
                .rejected_tasks
                .into_iter()
                .map(TeamTask::from)
                .collect(),
            rounds: result.rounds as u32,
        })
    }
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
}

impl From<SearchConfig> for RustSearchConfig {
    fn from(c: SearchConfig) -> Self {
        Self {
            timeout: c.timeout as u64,
            health: c.health.map(|h| h.into()),
            engines: c.engines.into_iter().map(|(k, v)| (k, v.into())).collect(),
        }
    }
}

// ============================================================================
// Agent Orchestrator - Main-Sub Agent Coordination
// ============================================================================

/// SubAgent configuration for orchestrator.
#[napi(object)]
#[derive(Clone)]
pub struct SubAgentConfig {
    /// Agent type (general, explore, plan, etc.)
    pub agent_type: String,
    /// Task description
    pub description: String,
    /// Execution prompt
    pub prompt: String,
    /// Enable permissive mode (bypass HITL)
    pub permissive: bool,
    /// Deny rules to enforce even in permissive mode (e.g., ["mcp__longvt__*"])
    pub permissive_deny: Option<Vec<String>>,
    /// Maximum execution steps
    pub max_steps: Option<u32>,
    /// Execution timeout (milliseconds)
    pub timeout_ms: Option<u32>,
    /// Parent SubAgent ID (for nesting)
    pub parent_id: Option<String>,
    /// Workspace directory for the SubAgent (defaults to ".")
    pub workspace: Option<String>,
    /// Extra directories to scan for agent definition files
    pub agent_dirs: Option<Vec<String>>,
    /// Lane queue config for External/Hybrid tool dispatch.
    /// When set, tools in the specified lanes are routed to external workers.
    pub lane_config: Option<SessionQueueConfig>,
}

impl From<SubAgentConfig> for RustSubAgentConfig {
    fn from(c: SubAgentConfig) -> Self {
        let mut config = RustSubAgentConfig::new(c.agent_type, c.prompt);
        if !c.description.is_empty() {
            config = config.with_description(c.description);
        }
        config = config.with_permissive(c.permissive);
        if let Some(deny) = c.permissive_deny {
            config = config.with_permissive_deny(deny);
        }
        if let Some(steps) = c.max_steps {
            config = config.with_max_steps(steps as usize);
        }
        if let Some(timeout) = c.timeout_ms {
            config = config.with_timeout_ms(timeout as u64);
        }
        if let Some(parent) = c.parent_id {
            config = config.with_parent_id(parent);
        }
        if let Some(ws) = c.workspace {
            config = config.with_workspace(ws);
        }
        if let Some(dirs) = c.agent_dirs {
            config = config.with_agent_dirs(dirs);
        }
        if let Some(lc) = c.lane_config {
            config = config.with_lane_config(js_queue_config_to_rust(&lc));
        }
        config
    }
}

/// Unified agent slot — used for both standalone subagents and team members.
///
/// When `role` is `undefined` the slot describes a standalone subagent.
/// Valid role values: `"lead"`, `"worker"`, `"reviewer"`.
#[napi(object)]
#[derive(Clone)]
pub struct AgentSlot {
    /// Agent type (general, explore, plan, etc.)
    pub agent_type: String,
    /// Team role: "lead", "worker", or "reviewer". Omit for standalone.
    pub role: Option<String>,
    /// Task description
    pub description: String,
    /// Execution prompt
    pub prompt: String,
    /// Enable permissive mode (bypass HITL)
    pub permissive: bool,
    /// Deny rules to enforce even in permissive mode (e.g., ["mcp__longvt__*"])
    pub permissive_deny: Option<Vec<String>>,
    /// Maximum execution steps
    pub max_steps: Option<u32>,
    /// Execution timeout (milliseconds)
    pub timeout_ms: Option<u32>,
    /// Parent SubAgent ID (for nesting)
    pub parent_id: Option<String>,
    /// Workspace directory (defaults to ".")
    pub workspace: Option<String>,
    /// Extra directories to scan for agent definition files
    pub agent_dirs: Option<Vec<String>>,
    /// Lane queue config for External/Hybrid tool dispatch
    pub lane_config: Option<SessionQueueConfig>,
}

impl From<AgentSlot> for RustAgentSlot {
    fn from(s: AgentSlot) -> Self {
        let rust_role = s.role.as_deref().and_then(|r| match r {
            "lead" => Some(RustTeamRole::Lead),
            "worker" => Some(RustTeamRole::Worker),
            "reviewer" => Some(RustTeamRole::Reviewer),
            _ => None,
        });
        let mut slot = RustAgentSlot::new(s.agent_type, s.prompt);
        if let Some(r) = rust_role {
            slot = slot.with_role(r);
        }
        if !s.description.is_empty() {
            slot = slot.with_description(s.description);
        }
        slot = slot.with_permissive(s.permissive);
        if let Some(deny) = s.permissive_deny {
            slot = slot.with_permissive_deny(deny);
        }
        if let Some(steps) = s.max_steps {
            slot = slot.with_max_steps(steps as usize);
        }
        if let Some(timeout) = s.timeout_ms {
            slot = slot.with_timeout_ms(timeout as u64);
        }
        if let Some(parent) = s.parent_id {
            slot = slot.with_parent_id(parent);
        }
        if let Some(ws) = s.workspace {
            slot = slot.with_workspace(ws);
        }
        if let Some(dirs) = s.agent_dirs {
            slot = slot.with_agent_dirs(dirs);
        }
        if let Some(lc) = s.lane_config {
            slot = slot.with_lane_config(js_queue_config_to_rust(&lc));
        }
        slot
    }
}

/// SubAgent handle for control and monitoring.
#[napi]
pub struct SubAgentHandle {
    inner: Arc<tokio::sync::Mutex<RustSubAgentHandle>>,
}

#[napi]
impl SubAgentHandle {
    /// Get SubAgent ID
    #[napi(getter)]
    pub fn id(&self) -> napi::Result<String> {
        let handle = self.inner.clone();
        Ok(get_runtime().block_on(async move {
            let h = handle.lock().await;
            h.id.clone()
        }))
    }

    /// Get current state (non-blocking)
    #[napi]
    pub fn state(&self) -> napi::Result<String> {
        let handle = self.inner.clone();
        Ok(get_runtime().block_on(async move {
            let h = handle.lock().await;
            let state = h.state_async().await;
            format!("{:?}", state)
        }))
    }

    /// Pause execution
    #[napi]
    pub fn pause(&self) -> napi::Result<()> {
        let handle = self.inner.clone();
        get_runtime()
            .block_on(async move { handle.lock().await.pause().await })
            .map_err(|e| napi::Error::from_reason(format!("Pause failed: {}", e)))
    }

    /// Resume execution
    #[napi]
    pub fn resume(&self) -> napi::Result<()> {
        let handle = self.inner.clone();
        get_runtime()
            .block_on(async move { handle.lock().await.resume().await })
            .map_err(|e| napi::Error::from_reason(format!("Resume failed: {}", e)))
    }

    /// Cancel execution
    #[napi]
    pub fn cancel(&self) -> napi::Result<()> {
        let handle = self.inner.clone();
        get_runtime()
            .block_on(async move { handle.lock().await.cancel().await })
            .map_err(|e| napi::Error::from_reason(format!("Cancel failed: {}", e)))
    }

    /// Wait for completion and get result
    #[napi]
    pub fn wait(&self) -> napi::Result<String> {
        let handle = self.inner.clone();
        get_runtime()
            .block_on(async move { handle.lock().await.wait().await })
            .map_err(|e| napi::Error::from_reason(format!("Wait failed: {}", e)))
    }
}

/// SubAgent activity type
#[napi(object)]
pub struct SubAgentActivity {
    /// Activity type: idle, calling_tool, requesting_llm, waiting_for_control
    pub activity_type: String,
    /// Activity data (JSON string)
    pub data: Option<String>,
}

impl From<RustSubAgentActivity> for SubAgentActivity {
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

/// SubAgent information with metadata and current activity
#[napi(object)]
pub struct SubAgentInfo {
    pub id: String,
    pub agent_type: String,
    pub description: String,
    pub state: String,
    pub parent_id: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
    pub current_activity: Option<SubAgentActivity>,
}

impl From<RustSubAgentInfo> for SubAgentInfo {
    fn from(info: RustSubAgentInfo) -> Self {
        Self {
            id: info.id,
            agent_type: info.agent_type,
            description: info.description,
            state: info.state,
            parent_id: info.parent_id,
            created_at: info.created_at as i64,
            updated_at: info.updated_at as i64,
            current_activity: info.current_activity.map(|a| a.into()),
        }
    }
}

/// SubAgent activity entry (id + activity)
#[napi(object)]
pub struct SubAgentActivityEntry {
    pub id: String,
    pub activity: SubAgentActivity,
}

/// SubAgent state entry (id + state)
#[napi(object)]
pub struct SubAgentStateEntry {
    pub id: String,
    pub state: String,
}

/// A pending external task waiting for a remote worker to process.
#[napi(object)]
pub struct PendingExternalTask {
    /// Unique task identifier — pass this to `completeExternalTask()`
    pub task_id: String,
    /// Tool type: "bash", "write", "edit", etc.
    pub command_type: String,
    /// JSON-encoded tool arguments
    pub payload: String,
    /// Lane name: "Execute", "Query", etc.
    pub lane: String,
}

/// Agent Orchestrator for main-sub agent coordination.
#[napi]
pub struct Orchestrator {
    inner: Arc<tokio::sync::Mutex<RustOrchestrator>>,
}

#[napi]
impl Orchestrator {
    /// Create a new orchestrator.
    ///
    /// @param agent - Optional `Agent` instance. When provided, spawned SubAgents
    ///                execute real LLM calls using the agent's configuration.
    ///                When omitted, SubAgents run in placeholder mode.
    #[napi(factory)]
    pub fn create(agent: Option<&Agent>) -> Self {
        let orch = match agent {
            Some(a) => RustOrchestrator::from_agent(a.inner.clone()),
            None => RustOrchestrator::new_memory(),
        };
        Self {
            inner: Arc::new(tokio::sync::Mutex::new(orch)),
        }
    }

    /// Spawn a new SubAgent
    #[napi]
    pub fn spawn_subagent(&self, config: SubAgentConfig) -> napi::Result<SubAgentHandle> {
        let orch = self.inner.clone();
        let cfg: RustSubAgentConfig = config.into();
        let handle = get_runtime()
            .block_on(async move { orch.lock().await.spawn_subagent(cfg).await })
            .map_err(|e| napi::Error::from_reason(format!("Spawn failed: {}", e)))?;
        Ok(SubAgentHandle {
            inner: Arc::new(tokio::sync::Mutex::new(handle)),
        })
    }

    /// Spawn a subagent from a unified `AgentSlot` declaration.
    ///
    /// Convenience wrapper over `spawnSubagent` that accepts the unified slot
    /// type.  The `role` field is ignored for standalone spawning — use
    /// `runTeam` for team-based workflows.
    #[napi]
    pub fn spawn(&self, slot: AgentSlot) -> napi::Result<SubAgentHandle> {
        let orch = self.inner.clone();
        let s: RustAgentSlot = slot.into();
        let handle = get_runtime()
            .block_on(async move { orch.lock().await.spawn(s).await })
            .map_err(|e| napi::Error::from_reason(format!("Spawn failed: {}", e)))?;
        Ok(SubAgentHandle {
            inner: Arc::new(tokio::sync::Mutex::new(handle)),
        })
    }

    /// Run a goal through a Lead → Worker → Reviewer team built from AgentSlots.
    ///
    /// Requires `Orchestrator.create(agent)` mode — returns an error if no backing
    /// Agent is configured.  Each slot's `role` field determines its position in the
    /// team; slots without a role default to Worker.
    ///
    /// @returns `TeamRunResult` with `doneTasks`, `rejectedTasks`, and `rounds`.
    #[napi]
    pub async fn run_team(
        &self,
        goal: String,
        workspace: String,
        slots: Vec<AgentSlot>,
    ) -> napi::Result<TeamRunResult> {
        let orch = self.inner.clone();
        let rust_slots: Vec<RustAgentSlot> = slots.into_iter().map(RustAgentSlot::from).collect();
        let result = get_runtime()
            .spawn(async move {
                orch.lock()
                    .await
                    .run_team(goal, workspace, rust_slots)
                    .await
            })
            .await
            .map_err(|e| napi::Error::from_reason(format!("Task join error: {e}")))?
            .map_err(|e| napi::Error::from_reason(format!("Team run failed: {e}")))?;
        Ok(TeamRunResult {
            done_tasks: result.done_tasks.into_iter().map(TeamTask::from).collect(),
            rejected_tasks: result
                .rejected_tasks
                .into_iter()
                .map(TeamTask::from)
                .collect(),
            rounds: result.rounds as u32,
        })
    }

    /// Get active SubAgent count
    #[napi]
    pub fn active_count(&self) -> napi::Result<u32> {
        let orch = self.inner.clone();
        Ok(get_runtime().block_on(async move { orch.lock().await.active_count().await }) as u32)
    }

    /// Get all SubAgent information list
    #[napi]
    pub fn list_subagents(&self) -> napi::Result<Vec<SubAgentInfo>> {
        let orch = self.inner.clone();
        let infos = get_runtime().block_on(async move { orch.lock().await.list_subagents().await });
        Ok(infos.into_iter().map(|i| i.into()).collect())
    }

    /// Get specific SubAgent information
    #[napi]
    pub fn get_subagent_info(&self, id: String) -> napi::Result<Option<SubAgentInfo>> {
        let orch = self.inner.clone();
        let info =
            get_runtime().block_on(async move { orch.lock().await.get_subagent_info(&id).await });
        Ok(info.map(|i| i.into()))
    }

    /// Get all active SubAgent activities
    #[napi]
    pub fn get_active_activities(&self) -> napi::Result<Vec<SubAgentActivityEntry>> {
        let orch = self.inner.clone();
        let activities =
            get_runtime().block_on(async move { orch.lock().await.get_active_activities().await });
        Ok(activities
            .into_iter()
            .map(|(id, activity)| SubAgentActivityEntry {
                id,
                activity: activity.into(),
            })
            .collect())
    }

    /// Get all SubAgent states
    #[napi]
    pub fn get_all_states(&self) -> napi::Result<Vec<SubAgentStateEntry>> {
        let orch = self.inner.clone();
        let states =
            get_runtime().block_on(async move { orch.lock().await.get_all_states().await });
        Ok(states
            .into_iter()
            .map(|(id, state)| SubAgentStateEntry {
                id,
                state: format!("{:?}", state),
            })
            .collect())
    }

    /// Pause a SubAgent
    #[napi]
    pub fn pause_subagent(&self, id: String) -> napi::Result<()> {
        let orch = self.inner.clone();
        get_runtime()
            .block_on(async move { orch.lock().await.pause_subagent(&id).await })
            .map_err(|e| napi::Error::from_reason(format!("Pause failed: {}", e)))
    }

    /// Resume a SubAgent
    #[napi]
    pub fn resume_subagent(&self, id: String) -> napi::Result<()> {
        let orch = self.inner.clone();
        get_runtime()
            .block_on(async move { orch.lock().await.resume_subagent(&id).await })
            .map_err(|e| napi::Error::from_reason(format!("Resume failed: {}", e)))
    }

    /// Cancel a SubAgent
    #[napi]
    pub fn cancel_subagent(&self, id: String) -> napi::Result<()> {
        let orch = self.inner.clone();
        get_runtime()
            .block_on(async move { orch.lock().await.cancel_subagent(&id).await })
            .map_err(|e| napi::Error::from_reason(format!("Cancel failed: {}", e)))
    }

    /// Wait for all SubAgents to complete
    #[napi]
    pub fn wait_all(&self) -> napi::Result<()> {
        let orch = self.inner.clone();
        get_runtime()
            .block_on(async move { orch.lock().await.wait_all().await })
            .map_err(|e| napi::Error::from_reason(format!("Wait failed: {}", e)))
    }

    /// Return any external tasks currently waiting for the given SubAgent.
    ///
    /// Returns an empty array when no tasks are pending or the SubAgent is not found.
    #[napi]
    pub fn pending_external_tasks_for(&self, subagent_id: String) -> napi::Result<Vec<PendingExternalTask>> {
        let orch = self.inner.clone();
        let tasks = get_runtime()
            .block_on(async move { orch.lock().await.pending_external_tasks_for(&subagent_id).await });
        Ok(tasks
            .into_iter()
            .map(|t| PendingExternalTask {
                task_id: t.task_id,
                command_type: t.command_type,
                payload: serde_json::to_string(&t.payload).unwrap_or_default(),
                lane: format!("{:?}", t.lane),
            })
            .collect())
    }

    /// Complete an external task dispatched to a remote worker.
    ///
    /// Returns `true` if the task was found and completed, `false` if no
    /// session with the given `subagent_id` is currently registered.
    #[napi]
    pub fn complete_external_task(
        &self,
        subagent_id: String,
        task_id: String,
        result: ExternalTaskResult,
    ) -> napi::Result<bool> {
        let ext_result = RustExternalTaskResult {
            success: result.success,
            result: result.result.unwrap_or(serde_json::Value::Null),
            error: result.error,
        };
        let orch = self.inner.clone();
        Ok(get_runtime().block_on(async move {
            orch.lock()
                .await
                .complete_external_task(&subagent_id, &task_id, ext_result)
                .await
        }))
    }
}
