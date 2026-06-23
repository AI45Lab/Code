use crate::llm::{Message, TokenUsage, ToolDefinition};
use crate::planning::Task;
use crate::prompts::PlanningMode;
use crate::queue::SessionQueueConfig;
use serde::{Deserialize, Serialize};

// ============================================================================
// Serializable Session Data
// ============================================================================

/// Session state persisted with saved sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SessionState {
    #[default]
    Unknown = 0,
    Active = 1,
    Paused = 2,
    Completed = 3,
    Error = 4,
}

/// Context usage statistics persisted with saved sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextUsage {
    pub used_tokens: usize,
    pub max_tokens: usize,
    pub percent: f32,
    pub turns: usize,
}

impl Default for ContextUsage {
    fn default() -> Self {
        Self {
            used_tokens: 0,
            max_tokens: 200_000,
            percent: 0.0,
            turns: 0,
        }
    }
}

/// Default auto-compact threshold (80% of context window).
pub const DEFAULT_AUTO_COMPACT_THRESHOLD: f32 = 0.80;

pub(crate) fn default_auto_compact_threshold() -> f32 {
    DEFAULT_AUTO_COMPACT_THRESHOLD
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// Serializable session configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    pub name: String,
    pub workspace: String,
    pub system_prompt: Option<String>,
    pub max_context_length: u32,
    pub auto_compact: bool,
    /// Context usage percentage threshold to trigger auto-compaction (0.0 - 1.0).
    /// Only used when `auto_compact` is true. Default: 0.80 (80%).
    #[serde(default = "default_auto_compact_threshold")]
    pub auto_compact_threshold: f32,
    /// Storage type for this session.
    #[serde(default)]
    pub storage_type: crate::config::StorageBackend,
    /// Optional advanced queue configuration.
    ///
    /// Queue infrastructure is initialized only when this is set. Ordinary
    /// sessions stay queue-free.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_config: Option<SessionQueueConfig>,
    /// Confirmation policy (optional, uses defaults if None).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confirmation_policy: Option<crate::hitl::ConfirmationPolicy>,
    /// Permission policy (optional, uses defaults if None).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub permission_policy: Option<crate::permissions::PermissionPolicy>,
    /// Whether active skill `allowed-tools` restrict ordinary session tools.
    #[serde(default, skip_serializing_if = "is_false")]
    pub enforce_active_skill_tool_restrictions: bool,
    /// Maximum sibling branches/tools to run concurrently.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_parallel_tasks: Option<usize>,
    /// Automatic subagent delegation settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_delegation: Option<crate::config::AutoDelegationConfig>,
    /// Parent session ID (for delegated child sessions).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    /// Security configuration (optional, enables security features).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub security_config: Option<crate::security::SecurityConfig>,
    /// Shared hook engine for lifecycle events.
    #[serde(skip)]
    pub hook_engine: Option<std::sync::Arc<dyn crate::hooks::HookExecutor>>,
    /// Enable planning phase before execution.
    #[serde(default)]
    pub planning_mode: PlanningMode,
    /// Enable goal tracking.
    #[serde(default)]
    pub goal_tracking: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            workspace: String::new(),
            system_prompt: None,
            max_context_length: 0,
            auto_compact: false,
            auto_compact_threshold: DEFAULT_AUTO_COMPACT_THRESHOLD,
            storage_type: crate::config::StorageBackend::default(),
            queue_config: None,
            confirmation_policy: None,
            permission_policy: None,
            enforce_active_skill_tool_restrictions: false,
            max_parallel_tasks: None,
            auto_delegation: None,
            parent_id: None,
            security_config: None,
            hook_engine: None,
            planning_mode: PlanningMode::default(),
            goal_tracking: false,
        }
    }
}

/// Serializable session data for persistence
///
/// Contains only the fields that can be serialized.
/// Non-serializable fields (event_tx, command_queue, etc.) are rebuilt on load.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionData {
    /// Session ID
    pub id: String,

    /// Session configuration
    pub config: SessionConfig,

    /// Current state
    pub state: SessionState,

    /// Conversation history
    pub messages: Vec<Message>,

    /// Context usage statistics
    pub context_usage: ContextUsage,

    /// Total token usage
    pub total_usage: TokenUsage,

    /// Cumulative dollar cost for this session
    #[serde(default)]
    pub total_cost: f64,

    /// Model name for cost calculation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,

    /// LLM cost records for this session
    #[serde(default)]
    pub cost_records: Vec<crate::telemetry::LlmCostRecord>,

    /// Tool definitions (names only, rebuilt from executor on load)
    pub tool_names: Vec<String>,

    /// Whether thinking mode is enabled
    pub thinking_enabled: bool,

    /// Thinking budget if set
    pub thinking_budget: Option<usize>,

    /// Creation timestamp (Unix epoch seconds)
    pub created_at: i64,

    /// Last update timestamp (Unix epoch seconds)
    pub updated_at: i64,

    /// LLM configuration for per-session client (if set)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_config: Option<LlmConfigData>,

    /// Task list for tracking
    #[serde(default, alias = "todos")]
    pub tasks: Vec<Task>,

    /// Parent session ID (for delegated child sessions)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,

    /// Multi-tenant identifier. The framework only transports this string;
    /// the host decides what "tenant" means and how to
    /// aggregate/bill on it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,

    /// Identity of the principal that triggered this session (user id,
    /// service account, etc). Framework treats as opaque; emitted to
    /// hooks/traces for accounting and audit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub principal: Option<String>,

    /// Logical identifier of the agent template / definition the session
    /// was instantiated from. Lets the host aggregate sessions by
    /// "which agent recipe" independent of the concrete session id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_template_id: Option<String>,

    /// Distributed-trace correlation id. Propagated through hooks/traces
    /// so a session's events can be joined with upstream/downstream work
    /// in the host's observability pipeline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
}

/// Serializable LLM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfigData {
    pub provider: String,
    pub model: String,
    /// API key is NOT stored - must be provided on session resume
    #[serde(skip_serializing, default)]
    pub api_key: Option<String>,
    pub base_url: Option<String>,
}

impl SessionData {
    /// Extract tool names from definitions
    pub fn tool_names_from_definitions(tools: &[ToolDefinition]) -> Vec<String> {
        tools.iter().map(|t| t.name.clone()).collect()
    }
}
