//! Configuration module for A3S Code
//!
//! Provides configuration for:
//! - LLM providers and models (defaultModel in "provider/model" format, providers)
//! - Queue configuration (a3s-lane integration)
//! - Search configuration (a3s-search integration)
//! - Directories for dynamic skill and agent loading
//!
//! Configuration is loaded from ACL-compatible files or strings.
//! Existing `.acl` config filenames are still accepted for compatibility.
//! JSON support has been removed.

pub mod agent_dir;
mod loader;
mod provider;
mod search;
#[cfg(test)]
mod tests;

pub use agent_dir::{AgentDir, ScheduleSpec, ScriptToolLimits, ScriptToolSpec, ToolSpec};
pub use provider::{ModelConfig, ModelCost, ModelLimit, ModelModalities, ProviderConfig};
pub use search::{
    BrowserBackend, DocumentCacheConfig, DocumentOcrConfig, DocumentParserConfig, HeadlessConfig,
    SearchConfig, SearchEngineConfig, SearchHealthConfig,
};

use crate::memory::MemoryConfig;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ============================================================================
// Storage Configuration
// ============================================================================

/// Session storage backend type
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum StorageBackend {
    /// In-memory storage (no persistence)
    Memory,
    /// File-based storage (JSON files)
    #[default]
    File,
    /// Custom external storage (Redis, PostgreSQL, etc.)
    ///
    /// Requires a `SessionStore` implementation registered on `AgentSession` options.
    /// Use `storage_url` in config to pass connection details.
    Custom,
}

// ============================================================================
// Main Configuration
// ============================================================================

/// Automatic subagent delegation controls.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, rename_all = "camelCase")]
pub struct AutoDelegationConfig {
    /// Enable runtime-driven automatic child agent delegation.
    pub enabled: bool,
    /// Allow automatic delegation to launch multiple child agents in parallel.
    ///
    /// Manual `parallel_task` calls remain available when this is false.
    #[serde(alias = "auto_parallel")]
    pub auto_parallel: bool,
    /// Minimum local confidence required to auto-delegate a child task.
    pub min_confidence: f32,
    /// Maximum number of automatic child tasks per user request.
    pub max_tasks: usize,
}

impl Default for AutoDelegationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            auto_parallel: true,
            min_confidence: 0.72,
            max_tasks: 4,
        }
    }
}

/// Configuration for A3S Code
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CodeConfig {
    /// Default model in "provider/model" format (e.g., "anthropic/claude-sonnet-4-20250514")
    #[serde(default, alias = "default_model")]
    pub default_model: Option<String>,

    /// Provider configurations
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,

    /// Session storage backend
    #[serde(default)]
    pub storage_backend: StorageBackend,

    /// Sessions directory (for file backend)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sessions_dir: Option<PathBuf>,

    /// Connection URL for custom storage backend (e.g., "redis://localhost:6379", "postgres://user:pass@localhost/a3s")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage_url: Option<String>,

    /// Directories to scan for skill files (*.md with tool definitions)
    #[serde(default, alias = "skill_dirs")]
    pub skill_dirs: Vec<PathBuf>,

    /// Directories to scan for agent files (*.yaml or *.md)
    #[serde(default, alias = "agent_dirs")]
    pub agent_dirs: Vec<PathBuf>,

    /// Maximum tool execution rounds per turn (default: 25)
    #[serde(default, alias = "max_tool_rounds")]
    pub max_tool_rounds: Option<usize>,

    /// Maximum sibling branches/tools to run concurrently in bounded parallel fan-out paths.
    #[serde(default, alias = "max_parallel_tasks")]
    pub max_parallel_tasks: Option<usize>,

    /// Global automatic child-agent delegation settings.
    #[serde(default, alias = "auto_delegation")]
    pub auto_delegation: AutoDelegationConfig,

    /// Convenience global kill switch for automatic parallel child-agent fan-out.
    ///
    /// When set, overrides `auto_delegation.auto_parallel`.
    #[serde(default, alias = "auto_parallel")]
    pub auto_parallel: Option<bool>,

    /// Thinking/reasoning budget in tokens
    #[serde(default, alias = "thinking_budget")]
    pub thinking_budget: Option<usize>,

    /// Memory system configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<MemoryConfig>,

    /// Queue configuration (a3s-lane integration)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue: Option<crate::queue::SessionQueueConfig>,

    /// Search configuration (a3s-search integration)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search: Option<SearchConfig>,

    /// Built-in document context extraction configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document_parser: Option<DocumentParserConfig>,

    /// MCP server configurations
    #[serde(default, alias = "mcp_servers")]
    pub mcp_servers: Vec<crate::mcp::McpServerConfig>,
}
