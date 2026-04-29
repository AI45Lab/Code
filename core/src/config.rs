//! Configuration module for A3S Code
//!
//! Provides configuration for:
//! - LLM providers and models (defaultModel in "provider/model" format, providers)
//! - Queue configuration (a3s-lane integration)
//! - Search configuration (a3s-search integration)
//! - Directories for dynamic skill and agent loading
//!
//! Configuration is loaded from ACL-compatible files or strings.
//! Existing `.hcl` config filenames are still accepted for compatibility.
//! JSON support has been removed.

use crate::error::{CodeError, Result};
use crate::llm::LlmConfig;
use crate::memory::MemoryConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ============================================================================
// Provider Configuration
// ============================================================================

/// Model cost information (per million tokens)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ModelCost {
    /// Input token cost
    #[serde(default)]
    pub input: f64,
    /// Output token cost
    #[serde(default)]
    pub output: f64,
    /// Cache read cost
    #[serde(default)]
    pub cache_read: f64,
    /// Cache write cost
    #[serde(default)]
    pub cache_write: f64,
}

/// Model limits
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelLimit {
    /// Maximum context tokens
    #[serde(default)]
    pub context: u32,
    /// Maximum output tokens
    #[serde(default)]
    pub output: u32,
}

/// Model modalities (input/output types)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelModalities {
    /// Supported input types
    #[serde(default)]
    pub input: Vec<String>,
    /// Supported output types
    #[serde(default)]
    pub output: Vec<String>,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelConfig {
    /// Model ID (e.g., "claude-sonnet-4-20250514")
    pub id: String,
    /// Display name
    #[serde(default)]
    pub name: String,
    /// Model family (e.g., "claude-sonnet")
    #[serde(default)]
    pub family: String,
    /// Per-model API key override
    #[serde(default)]
    pub api_key: Option<String>,
    /// Per-model base URL override
    #[serde(default)]
    pub base_url: Option<String>,
    /// Static HTTP headers for this model
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Header name to receive the runtime session ID
    #[serde(default)]
    pub session_id_header: Option<String>,
    /// Supports file attachments
    #[serde(default)]
    pub attachment: bool,
    /// Supports reasoning/thinking
    #[serde(default)]
    pub reasoning: bool,
    /// Supports tool calling
    #[serde(default = "default_true")]
    pub tool_call: bool,
    /// Supports temperature setting
    #[serde(default = "default_true")]
    pub temperature: bool,
    /// Release date
    #[serde(default)]
    pub release_date: Option<String>,
    /// Input/output modalities
    #[serde(default)]
    pub modalities: ModelModalities,
    /// Cost information
    #[serde(default)]
    pub cost: ModelCost,
    /// Token limits
    #[serde(default)]
    pub limit: ModelLimit,
}

fn default_true() -> bool {
    true
}

/// Provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderConfig {
    /// Provider name (e.g., "anthropic", "openai")
    pub name: String,
    /// API key for this provider
    #[serde(default)]
    pub api_key: Option<String>,
    /// Base URL for the API
    #[serde(default)]
    pub base_url: Option<String>,
    /// Static HTTP headers for this provider
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Header name to receive the runtime session ID
    #[serde(default)]
    pub session_id_header: Option<String>,
    /// Available models
    #[serde(default)]
    pub models: Vec<ModelConfig>,
}

/// Apply model capability flags to an LlmConfig.
///
/// - `temperature = false` → omit temperature (model ignores it, e.g. o1)
/// - `reasoning = true` + `thinking_budget` set → pass budget to client
/// - `limit.output > 0` → use as max_tokens
fn apply_model_caps(
    mut config: LlmConfig,
    model: &ModelConfig,
    thinking_budget: Option<usize>,
) -> LlmConfig {
    // reasoning=true + thinking_budget set → pass budget to client (Anthropic only)
    if model.reasoning {
        if let Some(budget) = thinking_budget {
            config = config.with_thinking_budget(budget);
        }
    }

    // limit.output > 0 → use as max_tokens cap
    if model.limit.output > 0 {
        config = config.with_max_tokens(model.limit.output as usize);
    }

    // temperature=false models (e.g. o1) must not receive a temperature param.
    // Store the flag so the LLM client can gate it at call time.
    if !model.temperature {
        config.disable_temperature = true;
    }

    config
}

impl ProviderConfig {
    /// Find a model by ID
    pub fn find_model(&self, model_id: &str) -> Option<&ModelConfig> {
        self.models.iter().find(|m| m.id == model_id)
    }

    /// Get the effective API key for a model (model override or provider default)
    pub fn get_api_key<'a>(&'a self, model: &'a ModelConfig) -> Option<&'a str> {
        model.api_key.as_deref().or(self.api_key.as_deref())
    }

    /// Get the effective base URL for a model (model override or provider default)
    pub fn get_base_url<'a>(&'a self, model: &'a ModelConfig) -> Option<&'a str> {
        model.base_url.as_deref().or(self.base_url.as_deref())
    }

    /// Get the effective static headers for a model (provider defaults with model overrides)
    pub fn get_headers(&self, model: &ModelConfig) -> HashMap<String, String> {
        let mut headers = self.headers.clone();
        headers.extend(model.headers.clone());
        headers
    }

    /// Get the header name that should carry the runtime session ID.
    pub fn get_session_id_header<'a>(&'a self, model: &'a ModelConfig) -> Option<&'a str> {
        model
            .session_id_header
            .as_deref()
            .or(self.session_id_header.as_deref())
    }
}

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
    /// Requires a `SessionStore` implementation registered via `SessionManager::with_store()`.
    /// Use `storage_url` in config to pass connection details.
    Custom,
}

// ============================================================================
// Main Configuration
// ============================================================================

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

    /// Agentic search tool configuration.
    #[serde(
        default,
        alias = "agentic_search",
        skip_serializing_if = "Option::is_none"
    )]
    pub agentic_search: Option<AgenticSearchConfig>,

    /// Agentic parse tool configuration.
    #[serde(
        default,
        alias = "agentic_parse",
        skip_serializing_if = "Option::is_none"
    )]
    pub agentic_parse: Option<AgenticParseConfig>,

    /// Built-in document context extraction configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document_parser: Option<DocumentParserConfig>,

    /// MCP server configurations
    #[serde(default, alias = "mcp_servers")]
    pub mcp_servers: Vec<crate::mcp::McpServerConfig>,
}

/// Search engine configuration (a3s-search integration)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchConfig {
    /// Default timeout in seconds for all engines
    #[serde(default = "default_search_timeout")]
    pub timeout: u64,

    /// Health monitor configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub health: Option<SearchHealthConfig>,

    /// Engine configurations
    #[serde(default, rename = "engine")]
    pub engines: std::collections::HashMap<String, SearchEngineConfig>,

    /// Headless browser configuration for JS-rendered engines (google, baidu, bing_cn).
    /// When enabled, the browser binary is auto-detected or downloaded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headless: Option<HeadlessConfig>,
}

/// Headless browser backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BrowserBackend {
    /// Chrome/Chromium headless. Auto-detected or downloaded from Google.
    Chrome,
    /// Lightpanda headless browser. Auto-detected or downloaded from GitHub.
    /// Supported on Linux and macOS only.
    Lightpanda,
}

#[allow(clippy::derivable_impls)]
impl Default for BrowserBackend {
    fn default() -> Self {
        BrowserBackend::Chrome
    }
}

/// Headless browser configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HeadlessConfig {
    /// Which headless backend to use.
    #[serde(default)]
    pub backend: BrowserBackend,

    /// Path to the browser executable. If None, auto-detected or downloaded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub browser_path: Option<String>,

    /// Maximum number of concurrent browser tabs.
    #[serde(default = "default_headless_max_tabs")]
    pub max_tabs: usize,

    /// Additional launch arguments for the browser.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub launch_args: Vec<String>,
}

impl Default for HeadlessConfig {
    fn default() -> Self {
        Self {
            backend: BrowserBackend::default(),
            browser_path: None,
            max_tabs: 4,
            launch_args: Vec::new(),
        }
    }
}

/// Default configuration for the built-in `agentic_search` tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgenticSearchConfig {
    /// Whether the tool is registered by default.
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Default search mode when tool input omits `mode`.
    #[serde(default = "default_agentic_search_mode")]
    pub default_mode: String,

    /// Default max results when tool input omits `max_results`.
    #[serde(default = "default_agentic_search_max_results")]
    pub max_results: usize,

    /// Default context lines when tool input omits `context_lines`.
    #[serde(default = "default_agentic_search_context_lines")]
    pub context_lines: usize,
}

impl Default for AgenticSearchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_mode: default_agentic_search_mode(),
            max_results: default_agentic_search_max_results(),
            context_lines: default_agentic_search_context_lines(),
        }
    }
}

impl AgenticSearchConfig {
    pub fn normalized(&self) -> Self {
        let default_mode = match self.default_mode.to_ascii_lowercase().as_str() {
            "fast" => "fast".to_string(),
            "deep" => "deep".to_string(),
            "filename_only" | "filename" => "filename_only".to_string(),
            _ => default_agentic_search_mode(),
        };

        Self {
            enabled: self.enabled,
            default_mode,
            max_results: self.max_results.clamp(1, 100),
            context_lines: self.context_lines.min(20),
        }
    }
}

/// Default configuration for the built-in `agentic_parse` tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgenticParseConfig {
    /// Whether the tool is registered by default.
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Default parse strategy when tool input omits `strategy`.
    #[serde(default = "default_agentic_parse_strategy")]
    pub default_strategy: String,

    /// Default maximum characters sent to the LLM when tool input omits `max_chars`.
    #[serde(default = "default_agentic_parse_max_chars")]
    pub max_chars: usize,
}

impl Default for AgenticParseConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_strategy: default_agentic_parse_strategy(),
            max_chars: default_agentic_parse_max_chars(),
        }
    }
}

impl AgenticParseConfig {
    pub fn normalized(&self) -> Self {
        let default_strategy = match self.default_strategy.to_ascii_lowercase().as_str() {
            "auto" => "auto".to_string(),
            "structured" => "structured".to_string(),
            "narrative" => "narrative".to_string(),
            "tabular" => "tabular".to_string(),
            "code" => "code".to_string(),
            _ => default_agentic_parse_strategy(),
        };

        Self {
            enabled: self.enabled,
            default_strategy,
            max_chars: self.max_chars.clamp(500, 200_000),
        }
    }
}

/// Default configuration for built-in document context extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentParserConfig {
    /// Whether the default document extraction stack is registered in the parser registry.
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Maximum file size accepted by the parser, in MiB.
    #[serde(default = "default_document_parser_max_file_size_mb")]
    pub max_file_size_mb: u64,

    /// Optional OCR / vision-model settings for image-heavy documents.
    ///
    /// These settings control OCR fallback when context extraction reaches
    /// scanned or image-heavy inputs. Current parsers may not execute OCR for
    /// every format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ocr: Option<DocumentOcrConfig>,

    /// Optional cache settings for parsed / normalized document context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<DocumentCacheConfig>,
}

impl Default for DocumentParserConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_file_size_mb: default_document_parser_max_file_size_mb(),
            ocr: None,
            cache: Some(DocumentCacheConfig::default()),
        }
    }
}

impl DocumentParserConfig {
    pub fn normalized(&self) -> Self {
        Self {
            enabled: self.enabled,
            max_file_size_mb: self.max_file_size_mb.clamp(1, 1024),
            ocr: self.ocr.as_ref().map(DocumentOcrConfig::normalized),
            cache: self.cache.as_ref().map(DocumentCacheConfig::normalized),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentCacheConfig {
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub directory: Option<PathBuf>,
}

impl Default for DocumentCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            directory: None,
        }
    }
}

impl DocumentCacheConfig {
    pub fn normalized(&self) -> Self {
        Self {
            enabled: self.enabled,
            directory: self.directory.clone(),
        }
    }
}

/// OCR / vision-model configuration for built-in document context extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentOcrConfig {
    /// Whether OCR fallback is enabled for image-heavy documents.
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Vision-capable model identifier, for example `openai/gpt-4.1-mini`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Optional custom OCR prompt / extraction instruction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,

    /// Maximum number of rendered images/pages to send for OCR fallback.
    #[serde(default = "default_document_ocr_max_images")]
    pub max_images: usize,

    /// Render DPI when rasterizing pages for OCR fallback.
    #[serde(default = "default_document_ocr_dpi")]
    pub dpi: u32,

    /// OCR provider backend. Defaults to "vision" when model is set.
    /// "vision" - Vision API (OpenAI-compatible)
    /// "builtin" - Local tesseract (requires tesseract + pdftoppm binaries)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,

    /// Base URL for vision API. Defaults to OpenAI API if not set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,

    /// API key for vision API.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}

impl Default for DocumentOcrConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: None,
            prompt: None,
            max_images: default_document_ocr_max_images(),
            dpi: default_document_ocr_dpi(),
            provider: None,
            base_url: None,
            api_key: None,
        }
    }
}

fn acl_attr<'a>(block: &'a a3s_acl::Block, keys: &[&str]) -> Option<&'a a3s_acl::Value> {
    keys.iter().find_map(|key| block.attributes.get(*key))
}

fn acl_string(value: &a3s_acl::Value) -> Option<String> {
    match value {
        a3s_acl::Value::String(s) => Some(s.clone()),
        a3s_acl::Value::Call(name, args) if name == "env" => {
            let var_name = args.first().and_then(acl_string)?;
            std::env::var(var_name).ok()
        }
        _ => None,
    }
}

fn acl_string_attr(block: &a3s_acl::Block, keys: &[&str]) -> Option<String> {
    acl_attr(block, keys).and_then(acl_string)
}

fn acl_label_or_attr(block: &a3s_acl::Block, keys: &[&str]) -> Option<String> {
    block
        .labels
        .first()
        .cloned()
        .or_else(|| acl_string_attr(block, keys))
}

fn acl_bool_attr(block: &a3s_acl::Block, keys: &[&str]) -> Option<bool> {
    match acl_attr(block, keys) {
        Some(a3s_acl::Value::Bool(value)) => Some(*value),
        _ => None,
    }
}

fn acl_usize_attr(block: &a3s_acl::Block, keys: &[&str]) -> Option<usize> {
    match acl_attr(block, keys) {
        Some(a3s_acl::Value::Number(value)) if *value >= 0.0 => Some(*value as usize),
        _ => None,
    }
}

fn acl_path_list_attr(block: &a3s_acl::Block, keys: &[&str]) -> Option<Vec<PathBuf>> {
    let value = acl_attr(block, keys)?;
    match value {
        a3s_acl::Value::List(items) => Some(
            items
                .iter()
                .filter_map(acl_string)
                .map(PathBuf::from)
                .collect(),
        ),
        _ => acl_string(value).map(|s| vec![PathBuf::from(s)]),
    }
}

impl DocumentOcrConfig {
    pub fn normalized(&self) -> Self {
        Self {
            enabled: self.enabled,
            model: self.model.clone(),
            prompt: self.prompt.clone(),
            max_images: self.max_images.clamp(1, 64),
            dpi: self.dpi.clamp(72, 600),
            provider: self.provider.clone(),
            base_url: self.base_url.clone(),
            api_key: self.api_key.clone(),
        }
    }
}

/// Search health monitor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchHealthConfig {
    /// Number of consecutive failures before suspending
    #[serde(default = "default_max_failures")]
    pub max_failures: u32,

    /// Suspension duration in seconds
    #[serde(default = "default_suspend_seconds")]
    pub suspend_seconds: u64,
}

/// Per-engine search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchEngineConfig {
    /// Whether the engine is enabled
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Weight for ranking (higher = more influence)
    #[serde(default = "default_weight")]
    pub weight: f64,

    /// Per-engine timeout override in seconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,
}

fn default_search_timeout() -> u64 {
    10
}

fn default_headless_max_tabs() -> usize {
    4
}

fn default_max_failures() -> u32 {
    3
}

fn default_suspend_seconds() -> u64 {
    60
}

fn default_enabled() -> bool {
    true
}

fn default_weight() -> f64 {
    1.0
}

fn default_agentic_search_mode() -> String {
    "fast".to_string()
}

fn default_agentic_search_max_results() -> usize {
    10
}

fn default_agentic_search_context_lines() -> usize {
    2
}

fn default_agentic_parse_strategy() -> String {
    "auto".to_string()
}

fn default_agentic_parse_max_chars() -> usize {
    8000
}

fn default_document_parser_max_file_size_mb() -> u64 {
    50
}

fn default_document_ocr_max_images() -> usize {
    8
}

fn default_document_ocr_dpi() -> u32 {
    144
}

impl CodeConfig {
    /// Create a new empty configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Load configuration from an ACL-compatible config file.
    ///
    /// `.acl` is the canonical extension. Existing `.hcl` paths are accepted
    /// for compatibility because ACL is HCL-like. JSON support has been removed.
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            CodeError::Config(format!(
                "Failed to read config file {}: {}",
                path.display(),
                e
            ))
        })?;

        Self::from_acl(&content).map_err(|e| {
            CodeError::Config(format!(
                "Failed to parse ACL config {}: {}",
                path.display(),
                e
            ))
        })
    }

    /// Parse configuration from an ACL string.
    ///
    /// ACL (Agent Configuration Language) is similar to HCL but uses labeled blocks
    /// like `providers "openai" { }`. For compatibility, HCL-style blocks such
    /// as `providers { name = "openai" }` are also accepted.
    pub fn from_acl(content: &str) -> Result<Self> {
        use a3s_acl::parse_acl;

        let doc = parse_acl(content)
            .map_err(|e| CodeError::Config(format!("Failed to parse ACL: {}", e)))?;

        let mut config = Self::default();

        for block in doc.blocks {
            match block.name.as_str() {
                "default_model" => {
                    // ACL: default_model = "openai/gpt-4" or just "openai/gpt-4" as label
                    if let Some(default_model) = acl_label_or_attr(&block, &["default_model"]) {
                        config.default_model = Some(default_model);
                    }
                }
                "storage_backend" => {
                    if let Some(backend) = acl_string_attr(&block, &["storage_backend"]) {
                        config.storage_backend = match backend.to_ascii_lowercase().as_str() {
                            "memory" => StorageBackend::Memory,
                            "custom" => StorageBackend::Custom,
                            _ => StorageBackend::File,
                        };
                    }
                }
                "sessions_dir" => {
                    if let Some(path) = acl_string_attr(&block, &["sessions_dir"]) {
                        config.sessions_dir = Some(PathBuf::from(path));
                    }
                }
                "storage_url" => {
                    if let Some(storage_url) = acl_string_attr(&block, &["storage_url"]) {
                        config.storage_url = Some(storage_url);
                    }
                }
                "skill_dirs" | "skills" => {
                    if let Some(paths) = acl_path_list_attr(&block, &["skill_dirs", "skills"]) {
                        config.skill_dirs = paths;
                    }
                }
                "agent_dirs" => {
                    if let Some(paths) = acl_path_list_attr(&block, &["agent_dirs"]) {
                        config.agent_dirs = paths;
                    }
                }
                "max_tool_rounds" => {
                    if let Some(max_tool_rounds) = acl_usize_attr(&block, &["max_tool_rounds"]) {
                        config.max_tool_rounds = Some(max_tool_rounds);
                    }
                }
                "thinking_budget" => {
                    if let Some(thinking_budget) = acl_usize_attr(&block, &["thinking_budget"]) {
                        config.thinking_budget = Some(thinking_budget);
                    }
                }
                "providers" => {
                    let provider_name = acl_label_or_attr(&block, &["name"]).ok_or_else(|| {
                        CodeError::Config(
                            "providers block requires a label or name attribute (e.g., providers \"openai\" or providers { name = \"openai\" })".into(),
                        )
                    })?;

                    let mut provider = ProviderConfig {
                        name: provider_name.clone(),
                        api_key: None,
                        base_url: None,
                        headers: HashMap::new(),
                        session_id_header: None,
                        models: Vec::new(),
                    };

                    for (key, value) in &block.attributes {
                        match key.as_str() {
                            "apiKey" | "api_key" => {
                                if let Some(api_key) = acl_string(value) {
                                    provider.api_key = Some(api_key);
                                }
                            }
                            "baseUrl" | "base_url" => {
                                if let Some(base_url) = acl_string(value) {
                                    provider.base_url = Some(base_url);
                                }
                            }
                            "sessionIdHeader" | "session_id_header" => {
                                if let Some(header) = acl_string(value) {
                                    provider.session_id_header = Some(header);
                                }
                            }
                            _ => {}
                        }
                    }

                    // Process nested models blocks
                    for model_block in &block.blocks {
                        if model_block.name == "models" {
                            let model_name = acl_label_or_attr(model_block, &["id", "name"])
                                .ok_or_else(|| {
                                    CodeError::Config(
                                        "models block requires a label or id attribute (e.g., models \"gpt-4\" or models { id = \"gpt-4\" })"
                                            .into(),
                                    )
                                })?;

                            let mut model = ModelConfig {
                                id: model_name.clone(),
                                name: model_name.clone(),
                                family: String::new(),
                                api_key: None,
                                base_url: None,
                                headers: HashMap::new(),
                                session_id_header: None,
                                attachment: false,
                                reasoning: false,
                                tool_call: true,
                                temperature: true,
                                release_date: None,
                                modalities: ModelModalities::default(),
                                cost: ModelCost::default(),
                                limit: ModelLimit::default(),
                            };

                            for (key, value) in &model_block.attributes {
                                match key.as_str() {
                                    "name" => {
                                        if let Some(s) = acl_string(value) {
                                            model.name = s;
                                        }
                                    }
                                    "family" => {
                                        if let Some(s) = acl_string(value) {
                                            model.family = s;
                                        }
                                    }
                                    "apiKey" | "api_key" => {
                                        if let Some(api_key) = acl_string(value) {
                                            model.api_key = Some(api_key);
                                        }
                                    }
                                    "baseUrl" | "base_url" => {
                                        if let Some(base_url) = acl_string(value) {
                                            model.base_url = Some(base_url);
                                        }
                                    }
                                    "sessionIdHeader" | "session_id_header" => {
                                        if let Some(header) = acl_string(value) {
                                            model.session_id_header = Some(header);
                                        }
                                    }
                                    "attachment" => {
                                        model.attachment =
                                            acl_bool_attr(model_block, &["attachment"])
                                                .unwrap_or(model.attachment);
                                    }
                                    "reasoning" => {
                                        model.reasoning =
                                            acl_bool_attr(model_block, &["reasoning"])
                                                .unwrap_or(model.reasoning);
                                    }
                                    "toolCall" | "tool_call" => {
                                        model.tool_call =
                                            acl_bool_attr(model_block, &["toolCall", "tool_call"])
                                                .unwrap_or(model.tool_call);
                                    }
                                    "temperature" => {
                                        model.temperature =
                                            acl_bool_attr(model_block, &["temperature"])
                                                .unwrap_or(model.temperature);
                                    }
                                    "releaseDate" | "release_date" => {
                                        if let Some(release_date) = acl_string(value) {
                                            model.release_date = Some(release_date);
                                        }
                                    }
                                    _ => {}
                                }
                            }

                            provider.models.push(model);
                        }
                    }

                    config.providers.push(provider);
                }
                _ => {
                    // Other top-level blocks are not mapped by the lightweight
                    // ACL loader yet (queue, search, memory, MCP, etc.).
                }
            }
        }

        Ok(config)
    }

    /// Save configuration to a JSON file (used for persistence)
    ///
    /// Note: This saves as JSON format for persistence/debugging. Agent config
    /// files should be authored as ACL-compatible `.acl` or legacy `.hcl` files.
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                CodeError::Config(format!(
                    "Failed to create config directory {}: {}",
                    parent.display(),
                    e
                ))
            })?;
        }

        let content = serde_json::to_string_pretty(self)
            .map_err(|e| CodeError::Config(format!("Failed to serialize config: {}", e)))?;

        std::fs::write(path, content).map_err(|e| {
            CodeError::Config(format!(
                "Failed to write config file {}: {}",
                path.display(),
                e
            ))
        })?;

        Ok(())
    }

    /// Find a provider by name
    pub fn find_provider(&self, name: &str) -> Option<&ProviderConfig> {
        self.providers.iter().find(|p| p.name == name)
    }

    /// Get the default provider configuration (parsed from `default_model` "provider/model" format)
    pub fn default_provider_config(&self) -> Option<&ProviderConfig> {
        let default = self.default_model.as_ref()?;
        let (provider_name, _) = default.split_once('/')?;
        self.find_provider(provider_name)
    }

    /// Get the default model configuration (parsed from `default_model` "provider/model" format)
    pub fn default_model_config(&self) -> Option<(&ProviderConfig, &ModelConfig)> {
        let default = self.default_model.as_ref()?;
        let (provider_name, model_id) = default.split_once('/')?;
        let provider = self.find_provider(provider_name)?;
        let model = provider.find_model(model_id)?;
        Some((provider, model))
    }

    /// Get LlmConfig for the default provider and model
    ///
    /// Returns None if default provider/model is not configured or API key is missing.
    pub fn default_llm_config(&self) -> Option<LlmConfig> {
        let (provider, model) = self.default_model_config()?;
        let api_key = provider.get_api_key(model)?;
        let base_url = provider.get_base_url(model);
        let headers = provider.get_headers(model);
        let session_id_header = provider.get_session_id_header(model);

        let mut config = LlmConfig::new(&provider.name, &model.id, api_key);
        if let Some(url) = base_url {
            config = config.with_base_url(url);
        }
        if !headers.is_empty() {
            config = config.with_headers(headers);
        }
        if let Some(header_name) = session_id_header {
            config = config.with_session_id_header(header_name);
        }
        config = apply_model_caps(config, model, self.thinking_budget);
        Some(config)
    }

    /// Get LlmConfig for a specific provider and model
    ///
    /// Returns None if provider/model is not found or API key is missing.
    pub fn llm_config(&self, provider_name: &str, model_id: &str) -> Option<LlmConfig> {
        let provider = self.find_provider(provider_name)?;
        let model = provider.find_model(model_id)?;
        let api_key = provider.get_api_key(model)?;
        let base_url = provider.get_base_url(model);
        let headers = provider.get_headers(model);
        let session_id_header = provider.get_session_id_header(model);

        let mut config = LlmConfig::new(&provider.name, &model.id, api_key);
        if let Some(url) = base_url {
            config = config.with_base_url(url);
        }
        if !headers.is_empty() {
            config = config.with_headers(headers);
        }
        if let Some(header_name) = session_id_header {
            config = config.with_session_id_header(header_name);
        }
        config = apply_model_caps(config, model, self.thinking_budget);
        Some(config)
    }

    /// List all available models across all providers
    pub fn list_models(&self) -> Vec<(&ProviderConfig, &ModelConfig)> {
        self.providers
            .iter()
            .flat_map(|p| p.models.iter().map(move |m| (p, m)))
            .collect()
    }

    /// Add a skill directory
    pub fn add_skill_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.skill_dirs.push(dir.into());
        self
    }

    /// Add an agent directory
    pub fn add_agent_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.agent_dirs.push(dir.into());
        self
    }

    /// Check if any directories are configured
    pub fn has_directories(&self) -> bool {
        !self.skill_dirs.is_empty() || !self.agent_dirs.is_empty()
    }

    /// Check if provider configuration is available
    pub fn has_providers(&self) -> bool {
        !self.providers.is_empty()
    }
}

// ============================================================================
// ACL Parsing Helpers
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = CodeConfig::default();
        assert!(config.skill_dirs.is_empty());
        assert!(config.agent_dirs.is_empty());
        assert!(config.providers.is_empty());
        assert!(config.default_model.is_none());
        assert_eq!(config.storage_backend, StorageBackend::File);
        assert!(config.sessions_dir.is_none());
    }

    #[test]
    fn test_storage_backend_default() {
        let backend = StorageBackend::default();
        assert_eq!(backend, StorageBackend::File);
    }

    #[test]
    fn test_storage_backend_serde() {
        // Test serialization
        let memory = StorageBackend::Memory;
        let json = serde_json::to_string(&memory).unwrap();
        assert_eq!(json, "\"memory\"");

        let file = StorageBackend::File;
        let json = serde_json::to_string(&file).unwrap();
        assert_eq!(json, "\"file\"");

        // Test deserialization
        let memory: StorageBackend = serde_json::from_str("\"memory\"").unwrap();
        assert_eq!(memory, StorageBackend::Memory);

        let file: StorageBackend = serde_json::from_str("\"file\"").unwrap();
        assert_eq!(file, StorageBackend::File);
    }

    #[test]
    fn test_config_with_storage_backend() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config_path = temp_dir.path().join("config.acl");

        std::fs::write(
            &config_path,
            r#"
                storage_backend = "memory"
                sessions_dir = "/tmp/sessions"
            "#,
        )
        .unwrap();

        let config = CodeConfig::from_file(&config_path).unwrap();
        assert_eq!(config.storage_backend, StorageBackend::Memory);
        assert_eq!(config.sessions_dir, Some(PathBuf::from("/tmp/sessions")));
    }

    #[test]
    fn test_config_supports_hcl_style_provider_blocks() {
        std::env::set_var("A3S_CODE_TEST_API_KEY", "sk-test");
        // Test HCL-style: unlabeled provider block with name attribute,
        // and labeled models block (ACL parser doesn't support unlabeled nested blocks
        // with id attribute, so we use the label form for models)
        let config = CodeConfig::from_acl(
            r#"
                default_model = "openai/gpt-4.1"
                max_tool_rounds = 12
                skill_dirs = ["./skills"]

                providers {
                  name     = "openai"
                  api_key  = env("A3S_CODE_TEST_API_KEY")
                  base_url = "https://api.openai.com/v1"

                  models "gpt-4.1" {
                    name      = "GPT 4.1"
                    reasoning = true
                    tool_call = false
                  }
                }
            "#,
        )
        .unwrap();

        assert_eq!(config.default_model.as_deref(), Some("openai/gpt-4.1"));
        assert_eq!(config.max_tool_rounds, Some(12));
        assert_eq!(config.skill_dirs, vec![PathBuf::from("./skills")]);
        assert_eq!(config.providers.len(), 1);
        let provider = &config.providers[0];
        assert_eq!(provider.name, "openai");
        assert_eq!(provider.api_key.as_deref(), Some("sk-test"));
        assert_eq!(
            provider.base_url.as_deref(),
            Some("https://api.openai.com/v1")
        );
        assert_eq!(provider.models.len(), 1);
        assert_eq!(provider.models[0].id, "gpt-4.1");
        assert_eq!(provider.models[0].name, "GPT 4.1");
        assert!(provider.models[0].reasoning);
        assert!(!provider.models[0].tool_call);
    }

    #[test]
    fn test_config_supports_acl_style_provider_labels() {
        let config = CodeConfig::from_acl(
            r#"
                default_model = "openai/gpt-4.1"

                providers "openai" {
                  apiKey  = "sk-test"
                  baseUrl = "https://api.openai.com/v1"

                  models "gpt-4.1" {
                    name     = "GPT 4.1"
                    toolCall = true
                  }
                }
            "#,
        )
        .unwrap();

        assert_eq!(config.default_model.as_deref(), Some("openai/gpt-4.1"));
        assert_eq!(config.providers[0].name, "openai");
        assert_eq!(config.providers[0].api_key.as_deref(), Some("sk-test"));
        assert_eq!(config.providers[0].models[0].id, "gpt-4.1");
        assert_eq!(config.providers[0].models[0].name, "GPT 4.1");
        assert!(config.providers[0].models[0].tool_call);
    }

    #[test]
    fn test_config_builder() {
        let config = CodeConfig::new()
            .add_skill_dir("/tmp/skills")
            .add_agent_dir("/tmp/agents");

        assert_eq!(config.skill_dirs.len(), 1);
        assert_eq!(config.agent_dirs.len(), 1);
    }

    #[test]
    fn test_find_provider() {
        let config = CodeConfig {
            providers: vec![
                ProviderConfig {
                    name: "anthropic".to_string(),
                    api_key: Some("key1".to_string()),
                    base_url: None,
                    headers: HashMap::new(),
                    session_id_header: None,
                    models: vec![],
                },
                ProviderConfig {
                    name: "openai".to_string(),
                    api_key: Some("key2".to_string()),
                    base_url: None,
                    headers: HashMap::new(),
                    session_id_header: None,
                    models: vec![],
                },
            ],
            ..Default::default()
        };

        assert!(config.find_provider("anthropic").is_some());
        assert!(config.find_provider("openai").is_some());
        assert!(config.find_provider("unknown").is_none());
    }

    #[test]
    fn test_default_llm_config() {
        let config = CodeConfig {
            default_model: Some("anthropic/claude-sonnet-4".to_string()),
            providers: vec![ProviderConfig {
                name: "anthropic".to_string(),
                api_key: Some("test-api-key".to_string()),
                base_url: Some("https://api.anthropic.com".to_string()),
                headers: HashMap::new(),
                session_id_header: None,
                models: vec![ModelConfig {
                    id: "claude-sonnet-4".to_string(),
                    name: "Claude Sonnet 4".to_string(),
                    family: "claude-sonnet".to_string(),
                    api_key: None,
                    base_url: None,
                    headers: HashMap::new(),
                    session_id_header: None,
                    attachment: false,
                    reasoning: false,
                    tool_call: true,
                    temperature: true,
                    release_date: None,
                    modalities: ModelModalities::default(),
                    cost: ModelCost::default(),
                    limit: ModelLimit::default(),
                }],
            }],
            ..Default::default()
        };

        let llm_config = config.default_llm_config().unwrap();
        assert_eq!(llm_config.provider, "anthropic");
        assert_eq!(llm_config.model, "claude-sonnet-4");
        assert_eq!(llm_config.api_key.expose(), "test-api-key");
        assert_eq!(
            llm_config.base_url,
            Some("https://api.anthropic.com".to_string())
        );
    }

    #[test]
    fn test_model_api_key_override() {
        let provider = ProviderConfig {
            name: "openai".to_string(),
            api_key: Some("provider-key".to_string()),
            base_url: Some("https://api.openai.com".to_string()),
            headers: HashMap::new(),
            session_id_header: None,
            models: vec![
                ModelConfig {
                    id: "gpt-4".to_string(),
                    name: "GPT-4".to_string(),
                    family: "gpt".to_string(),
                    api_key: None, // Uses provider key
                    base_url: None,
                    headers: HashMap::new(),
                    session_id_header: None,
                    attachment: false,
                    reasoning: false,
                    tool_call: true,
                    temperature: true,
                    release_date: None,
                    modalities: ModelModalities::default(),
                    cost: ModelCost::default(),
                    limit: ModelLimit::default(),
                },
                ModelConfig {
                    id: "custom-model".to_string(),
                    name: "Custom Model".to_string(),
                    family: "custom".to_string(),
                    api_key: Some("model-specific-key".to_string()), // Override
                    base_url: Some("https://custom.api.com".to_string()), // Override
                    headers: HashMap::new(),
                    session_id_header: None,
                    attachment: false,
                    reasoning: false,
                    tool_call: true,
                    temperature: true,
                    release_date: None,
                    modalities: ModelModalities::default(),
                    cost: ModelCost::default(),
                    limit: ModelLimit::default(),
                },
            ],
        };

        // Model without override uses provider key
        let model1 = provider.find_model("gpt-4").unwrap();
        assert_eq!(provider.get_api_key(model1), Some("provider-key"));
        assert_eq!(
            provider.get_base_url(model1),
            Some("https://api.openai.com")
        );

        // Model with override uses its own key
        let model2 = provider.find_model("custom-model").unwrap();
        assert_eq!(provider.get_api_key(model2), Some("model-specific-key"));
        assert_eq!(
            provider.get_base_url(model2),
            Some("https://custom.api.com")
        );
    }

    #[test]
    fn test_list_models() {
        let config = CodeConfig {
            providers: vec![
                ProviderConfig {
                    name: "anthropic".to_string(),
                    api_key: None,
                    base_url: None,
                    headers: HashMap::new(),
                    session_id_header: None,
                    models: vec![
                        ModelConfig {
                            id: "claude-1".to_string(),
                            name: "Claude 1".to_string(),
                            family: "claude".to_string(),
                            api_key: None,
                            base_url: None,
                            headers: HashMap::new(),
                            session_id_header: None,
                            attachment: false,
                            reasoning: false,
                            tool_call: true,
                            temperature: true,
                            release_date: None,
                            modalities: ModelModalities::default(),
                            cost: ModelCost::default(),
                            limit: ModelLimit::default(),
                        },
                        ModelConfig {
                            id: "claude-2".to_string(),
                            name: "Claude 2".to_string(),
                            family: "claude".to_string(),
                            api_key: None,
                            base_url: None,
                            headers: HashMap::new(),
                            session_id_header: None,
                            attachment: false,
                            reasoning: false,
                            tool_call: true,
                            temperature: true,
                            release_date: None,
                            modalities: ModelModalities::default(),
                            cost: ModelCost::default(),
                            limit: ModelLimit::default(),
                        },
                    ],
                },
                ProviderConfig {
                    name: "openai".to_string(),
                    api_key: None,
                    base_url: None,
                    headers: HashMap::new(),
                    session_id_header: None,
                    models: vec![ModelConfig {
                        id: "gpt-4".to_string(),
                        name: "GPT-4".to_string(),
                        family: "gpt".to_string(),
                        api_key: None,
                        base_url: None,
                        headers: HashMap::new(),
                        session_id_header: None,
                        attachment: false,
                        reasoning: false,
                        tool_call: true,
                        temperature: true,
                        release_date: None,
                        modalities: ModelModalities::default(),
                        cost: ModelCost::default(),
                        limit: ModelLimit::default(),
                    }],
                },
            ],
            ..Default::default()
        };

        let models = config.list_models();
        assert_eq!(models.len(), 3);
    }

    #[test]
    fn test_config_from_file_not_found() {
        let result = CodeConfig::from_file(Path::new("/nonexistent/config.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_config_has_directories() {
        let empty = CodeConfig::default();
        assert!(!empty.has_directories());

        let with_skills = CodeConfig::new().add_skill_dir("/tmp/skills");
        assert!(with_skills.has_directories());

        let with_agents = CodeConfig::new().add_agent_dir("/tmp/agents");
        assert!(with_agents.has_directories());
    }

    #[test]
    fn test_config_has_providers() {
        let empty = CodeConfig::default();
        assert!(!empty.has_providers());

        let with_providers = CodeConfig {
            providers: vec![ProviderConfig {
                name: "test".to_string(),
                api_key: None,
                base_url: None,
                headers: HashMap::new(),
                session_id_header: None,
                models: vec![],
            }],
            ..Default::default()
        };
        assert!(with_providers.has_providers());
    }

    #[test]
    fn test_storage_backend_equality() {
        assert_eq!(StorageBackend::Memory, StorageBackend::Memory);
        assert_eq!(StorageBackend::File, StorageBackend::File);
        assert_ne!(StorageBackend::Memory, StorageBackend::File);
    }

    #[test]
    fn test_storage_backend_serde_custom() {
        let custom = StorageBackend::Custom;
        // Custom variant is now serializable
        let json = serde_json::to_string(&custom).unwrap();
        assert_eq!(json, "\"custom\"");

        // And deserializable
        let parsed: StorageBackend = serde_json::from_str("\"custom\"").unwrap();
        assert_eq!(parsed, StorageBackend::Custom);
    }

    #[test]
    fn test_model_cost_default() {
        let cost = ModelCost::default();
        assert_eq!(cost.input, 0.0);
        assert_eq!(cost.output, 0.0);
        assert_eq!(cost.cache_read, 0.0);
        assert_eq!(cost.cache_write, 0.0);
    }

    #[test]
    fn test_model_cost_serialization() {
        let cost = ModelCost {
            input: 3.0,
            output: 15.0,
            cache_read: 0.3,
            cache_write: 3.75,
        };
        let json = serde_json::to_string(&cost).unwrap();
        assert!(json.contains("\"input\":3"));
        assert!(json.contains("\"output\":15"));
    }

    #[test]
    fn test_model_cost_deserialization_missing_fields() {
        let json = r#"{"input":3.0}"#;
        let cost: ModelCost = serde_json::from_str(json).unwrap();
        assert_eq!(cost.input, 3.0);
        assert_eq!(cost.output, 0.0);
        assert_eq!(cost.cache_read, 0.0);
        assert_eq!(cost.cache_write, 0.0);
    }

    #[test]
    fn test_model_limit_default() {
        let limit = ModelLimit::default();
        assert_eq!(limit.context, 0);
        assert_eq!(limit.output, 0);
    }

    #[test]
    fn test_model_limit_serialization() {
        let limit = ModelLimit {
            context: 200000,
            output: 8192,
        };
        let json = serde_json::to_string(&limit).unwrap();
        assert!(json.contains("\"context\":200000"));
        assert!(json.contains("\"output\":8192"));
    }

    #[test]
    fn test_model_limit_deserialization_missing_fields() {
        let json = r#"{"context":100000}"#;
        let limit: ModelLimit = serde_json::from_str(json).unwrap();
        assert_eq!(limit.context, 100000);
        assert_eq!(limit.output, 0);
    }

    #[test]
    fn test_model_modalities_default() {
        let modalities = ModelModalities::default();
        assert!(modalities.input.is_empty());
        assert!(modalities.output.is_empty());
    }

    #[test]
    fn test_model_modalities_serialization() {
        let modalities = ModelModalities {
            input: vec!["text".to_string(), "image".to_string()],
            output: vec!["text".to_string()],
        };
        let json = serde_json::to_string(&modalities).unwrap();
        assert!(json.contains("\"input\""));
        assert!(json.contains("\"text\""));
    }

    #[test]
    fn test_model_modalities_deserialization_missing_fields() {
        let json = r#"{"input":["text"]}"#;
        let modalities: ModelModalities = serde_json::from_str(json).unwrap();
        assert_eq!(modalities.input.len(), 1);
        assert!(modalities.output.is_empty());
    }

    #[test]
    fn test_model_config_serialization() {
        let config = ModelConfig {
            id: "gpt-4o".to_string(),
            name: "GPT-4o".to_string(),
            family: "gpt-4".to_string(),
            api_key: Some("sk-test".to_string()),
            base_url: None,
            headers: HashMap::new(),
            session_id_header: None,
            attachment: true,
            reasoning: false,
            tool_call: true,
            temperature: true,
            release_date: Some("2024-05-13".to_string()),
            modalities: ModelModalities::default(),
            cost: ModelCost::default(),
            limit: ModelLimit::default(),
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"id\":\"gpt-4o\""));
        assert!(json.contains("\"attachment\":true"));
    }

    #[test]
    fn test_model_config_deserialization_with_defaults() {
        let json = r#"{"id":"test-model"}"#;
        let config: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.id, "test-model");
        assert_eq!(config.name, "");
        assert_eq!(config.family, "");
        assert!(config.api_key.is_none());
        assert!(!config.attachment);
        assert!(config.tool_call);
        assert!(config.temperature);
    }

    #[test]
    fn test_model_config_all_optional_fields() {
        let json = r#"{
            "id": "claude-sonnet-4",
            "name": "Claude Sonnet 4",
            "family": "claude-sonnet",
            "apiKey": "sk-test",
            "baseUrl": "https://api.anthropic.com",
            "attachment": true,
            "reasoning": true,
            "toolCall": false,
            "temperature": false,
            "releaseDate": "2025-05-14"
        }"#;
        let config: ModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.id, "claude-sonnet-4");
        assert_eq!(config.name, "Claude Sonnet 4");
        assert_eq!(config.api_key, Some("sk-test".to_string()));
        assert_eq!(
            config.base_url,
            Some("https://api.anthropic.com".to_string())
        );
        assert!(config.attachment);
        assert!(config.reasoning);
        assert!(!config.tool_call);
        assert!(!config.temperature);
    }

    #[test]
    fn test_provider_config_serialization() {
        let provider = ProviderConfig {
            name: "anthropic".to_string(),
            api_key: Some("sk-test".to_string()),
            base_url: Some("https://api.anthropic.com".to_string()),
            headers: HashMap::new(),
            session_id_header: None,
            models: vec![],
        };
        let json = serde_json::to_string(&provider).unwrap();
        assert!(json.contains("\"name\":\"anthropic\""));
        assert!(json.contains("\"apiKey\":\"sk-test\""));
    }

    #[test]
    fn test_provider_config_deserialization_missing_optional() {
        let json = r#"{"name":"openai"}"#;
        let provider: ProviderConfig = serde_json::from_str(json).unwrap();
        assert_eq!(provider.name, "openai");
        assert!(provider.api_key.is_none());
        assert!(provider.base_url.is_none());
        assert!(provider.models.is_empty());
    }

    #[test]
    fn test_provider_config_find_model() {
        let provider = ProviderConfig {
            name: "anthropic".to_string(),
            api_key: None,
            base_url: None,
            headers: HashMap::new(),
            session_id_header: None,
            models: vec![ModelConfig {
                id: "claude-sonnet-4".to_string(),
                name: "Claude Sonnet 4".to_string(),
                family: "claude-sonnet".to_string(),
                api_key: None,
                base_url: None,
                headers: HashMap::new(),
                session_id_header: None,
                attachment: false,
                reasoning: false,
                tool_call: true,
                temperature: true,
                release_date: None,
                modalities: ModelModalities::default(),
                cost: ModelCost::default(),
                limit: ModelLimit::default(),
            }],
        };

        let found = provider.find_model("claude-sonnet-4");
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, "claude-sonnet-4");

        let not_found = provider.find_model("gpt-4o");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_provider_config_get_api_key() {
        let provider = ProviderConfig {
            name: "anthropic".to_string(),
            api_key: Some("provider-key".to_string()),
            base_url: None,
            headers: HashMap::new(),
            session_id_header: None,
            models: vec![],
        };

        let model_with_key = ModelConfig {
            id: "test".to_string(),
            name: "".to_string(),
            family: "".to_string(),
            api_key: Some("model-key".to_string()),
            base_url: None,
            headers: HashMap::new(),
            session_id_header: None,
            attachment: false,
            reasoning: false,
            tool_call: true,
            temperature: true,
            release_date: None,
            modalities: ModelModalities::default(),
            cost: ModelCost::default(),
            limit: ModelLimit::default(),
        };

        let model_without_key = ModelConfig {
            id: "test2".to_string(),
            name: "".to_string(),
            family: "".to_string(),
            api_key: None,
            base_url: None,
            headers: HashMap::new(),
            session_id_header: None,
            attachment: false,
            reasoning: false,
            tool_call: true,
            temperature: true,
            release_date: None,
            modalities: ModelModalities::default(),
            cost: ModelCost::default(),
            limit: ModelLimit::default(),
        };

        assert_eq!(provider.get_api_key(&model_with_key), Some("model-key"));
        assert_eq!(
            provider.get_api_key(&model_without_key),
            Some("provider-key")
        );
    }

    #[test]
    fn test_provider_config_get_headers_and_session_id_header() {
        let mut provider_headers = HashMap::new();
        provider_headers.insert("X-Provider".to_string(), "provider".to_string());
        provider_headers.insert("X-Shared".to_string(), "provider".to_string());

        let mut model_headers = HashMap::new();
        model_headers.insert("X-Model".to_string(), "model".to_string());
        model_headers.insert("X-Shared".to_string(), "model".to_string());

        let provider = ProviderConfig {
            name: "openai".to_string(),
            api_key: Some("provider-key".to_string()),
            base_url: None,
            headers: provider_headers,
            session_id_header: Some("X-Session-Id".to_string()),
            models: vec![],
        };

        let model = ModelConfig {
            id: "gpt-4o".to_string(),
            name: "".to_string(),
            family: "".to_string(),
            api_key: None,
            base_url: None,
            headers: model_headers,
            session_id_header: Some("X-Model-Session".to_string()),
            attachment: false,
            reasoning: false,
            tool_call: true,
            temperature: true,
            release_date: None,
            modalities: ModelModalities::default(),
            cost: ModelCost::default(),
            limit: ModelLimit::default(),
        };

        let headers = provider.get_headers(&model);
        assert_eq!(headers.get("X-Provider"), Some(&"provider".to_string()));
        assert_eq!(headers.get("X-Model"), Some(&"model".to_string()));
        assert_eq!(headers.get("X-Shared"), Some(&"model".to_string()));
        assert_eq!(
            provider.get_session_id_header(&model),
            Some("X-Model-Session")
        );
    }

    #[test]
    fn test_llm_config_includes_headers_and_runtime_session_header() {
        let mut provider_headers = HashMap::new();
        provider_headers.insert("X-Provider".to_string(), "provider".to_string());

        let config = CodeConfig {
            default_model: Some("openai/gpt-4o".to_string()),
            providers: vec![ProviderConfig {
                name: "openai".to_string(),
                api_key: Some("sk-test".to_string()),
                base_url: Some("https://api.example.com".to_string()),
                headers: provider_headers,
                session_id_header: Some("X-Session-Id".to_string()),
                models: vec![ModelConfig {
                    id: "gpt-4o".to_string(),
                    name: "".to_string(),
                    family: "".to_string(),
                    api_key: None,
                    base_url: None,
                    headers: HashMap::new(),
                    session_id_header: None,
                    attachment: false,
                    reasoning: false,
                    tool_call: true,
                    temperature: true,
                    release_date: None,
                    modalities: ModelModalities::default(),
                    cost: ModelCost::default(),
                    limit: ModelLimit::default(),
                }],
            }],
            ..Default::default()
        };

        let llm_config = config.default_llm_config().unwrap();
        assert_eq!(
            llm_config.headers.get("X-Provider"),
            Some(&"provider".to_string())
        );
        assert_eq!(
            llm_config.session_id_header.as_deref(),
            Some("X-Session-Id")
        );
    }

    #[test]
    fn test_code_config_default_provider_config() {
        let config = CodeConfig {
            default_model: Some("anthropic/claude-sonnet-4".to_string()),
            providers: vec![ProviderConfig {
                name: "anthropic".to_string(),
                api_key: Some("sk-test".to_string()),
                base_url: None,
                headers: HashMap::new(),
                session_id_header: None,
                models: vec![],
            }],
            ..Default::default()
        };

        let provider = config.default_provider_config();
        assert!(provider.is_some());
        assert_eq!(provider.unwrap().name, "anthropic");
    }

    #[test]
    fn test_code_config_default_model_config() {
        let config = CodeConfig {
            default_model: Some("anthropic/claude-sonnet-4".to_string()),
            providers: vec![ProviderConfig {
                name: "anthropic".to_string(),
                api_key: Some("sk-test".to_string()),
                base_url: None,
                headers: HashMap::new(),
                session_id_header: None,
                models: vec![ModelConfig {
                    id: "claude-sonnet-4".to_string(),
                    name: "Claude Sonnet 4".to_string(),
                    family: "claude-sonnet".to_string(),
                    api_key: None,
                    base_url: None,
                    headers: HashMap::new(),
                    session_id_header: None,
                    attachment: false,
                    reasoning: false,
                    tool_call: true,
                    temperature: true,
                    release_date: None,
                    modalities: ModelModalities::default(),
                    cost: ModelCost::default(),
                    limit: ModelLimit::default(),
                }],
            }],
            ..Default::default()
        };

        let result = config.default_model_config();
        assert!(result.is_some());
        let (provider, model) = result.unwrap();
        assert_eq!(provider.name, "anthropic");
        assert_eq!(model.id, "claude-sonnet-4");
    }

    #[test]
    fn test_code_config_default_llm_config() {
        let config = CodeConfig {
            default_model: Some("anthropic/claude-sonnet-4".to_string()),
            providers: vec![ProviderConfig {
                name: "anthropic".to_string(),
                api_key: Some("sk-test".to_string()),
                base_url: Some("https://api.anthropic.com".to_string()),
                headers: HashMap::new(),
                session_id_header: None,
                models: vec![ModelConfig {
                    id: "claude-sonnet-4".to_string(),
                    name: "Claude Sonnet 4".to_string(),
                    family: "claude-sonnet".to_string(),
                    api_key: None,
                    base_url: None,
                    headers: HashMap::new(),
                    session_id_header: None,
                    attachment: false,
                    reasoning: false,
                    tool_call: true,
                    temperature: true,
                    release_date: None,
                    modalities: ModelModalities::default(),
                    cost: ModelCost::default(),
                    limit: ModelLimit::default(),
                }],
            }],
            ..Default::default()
        };

        let llm_config = config.default_llm_config();
        assert!(llm_config.is_some());
    }

    #[test]
    fn test_code_config_list_models() {
        let config = CodeConfig {
            providers: vec![
                ProviderConfig {
                    name: "anthropic".to_string(),
                    api_key: None,
                    base_url: None,
                    headers: HashMap::new(),
                    session_id_header: None,
                    models: vec![ModelConfig {
                        id: "claude-sonnet-4".to_string(),
                        name: "".to_string(),
                        family: "".to_string(),
                        api_key: None,
                        base_url: None,
                        headers: HashMap::new(),
                        session_id_header: None,
                        attachment: false,
                        reasoning: false,
                        tool_call: true,
                        temperature: true,
                        release_date: None,
                        modalities: ModelModalities::default(),
                        cost: ModelCost::default(),
                        limit: ModelLimit::default(),
                    }],
                },
                ProviderConfig {
                    name: "openai".to_string(),
                    api_key: None,
                    base_url: None,
                    headers: HashMap::new(),
                    session_id_header: None,
                    models: vec![ModelConfig {
                        id: "gpt-4o".to_string(),
                        name: "".to_string(),
                        family: "".to_string(),
                        api_key: None,
                        base_url: None,
                        headers: HashMap::new(),
                        session_id_header: None,
                        attachment: false,
                        reasoning: false,
                        tool_call: true,
                        temperature: true,
                        release_date: None,
                        modalities: ModelModalities::default(),
                        cost: ModelCost::default(),
                        limit: ModelLimit::default(),
                    }],
                },
            ],
            ..Default::default()
        };

        let models = config.list_models();
        assert_eq!(models.len(), 2);
    }

    #[test]
    fn test_llm_config_specific_provider_model() {
        let model: ModelConfig = serde_json::from_value(serde_json::json!({
            "id": "claude-3",
            "name": "Claude 3"
        }))
        .unwrap();

        let config = CodeConfig {
            providers: vec![ProviderConfig {
                name: "anthropic".to_string(),
                api_key: Some("sk-test".to_string()),
                base_url: None,
                headers: HashMap::new(),
                session_id_header: None,
                models: vec![model],
            }],
            ..Default::default()
        };

        let llm = config.llm_config("anthropic", "claude-3");
        assert!(llm.is_some());
        let llm = llm.unwrap();
        assert_eq!(llm.provider, "anthropic");
        assert_eq!(llm.model, "claude-3");
    }

    #[test]
    fn test_llm_config_missing_provider() {
        let config = CodeConfig::default();
        assert!(config.llm_config("nonexistent", "model").is_none());
    }

    #[test]
    fn test_llm_config_missing_model() {
        let config = CodeConfig {
            providers: vec![ProviderConfig {
                name: "anthropic".to_string(),
                api_key: Some("sk-test".to_string()),
                base_url: None,
                headers: HashMap::new(),
                session_id_header: None,
                models: vec![],
            }],
            ..Default::default()
        };
        assert!(config.llm_config("anthropic", "nonexistent").is_none());
    }

    #[test]
    fn test_agentic_search_config_normalizes_invalid_values() {
        let config = AgenticSearchConfig {
            enabled: true,
            default_mode: "weird".to_string(),
            max_results: 0,
            context_lines: 999,
        }
        .normalized();

        assert_eq!(config.default_mode, "fast");
        assert_eq!(config.max_results, 1);
        assert_eq!(config.context_lines, 20);
    }

    #[test]
    fn test_agentic_parse_config_normalizes_invalid_values() {
        let config = AgenticParseConfig {
            enabled: true,
            default_strategy: "unknown".to_string(),
            max_chars: 1,
        }
        .normalized();

        assert_eq!(config.default_strategy, "auto");
        assert_eq!(config.max_chars, 500);
    }

    #[test]
    fn test_document_parser_config_normalizes_nested_ocr_values() {
        let config = DocumentParserConfig {
            enabled: true,
            max_file_size_mb: 0,
            cache: Some(DocumentCacheConfig {
                enabled: true,
                directory: Some(PathBuf::from("/tmp/cache")),
            }),
            ocr: Some(DocumentOcrConfig {
                enabled: true,
                model: Some("openai/gpt-4.1-mini".to_string()),
                prompt: None,
                max_images: 0,
                dpi: 10,
                provider: None,
                base_url: None,
                api_key: None,
            }),
        }
        .normalized();

        assert_eq!(config.max_file_size_mb, 1);
        let cache = config.cache.unwrap();
        assert!(cache.enabled);
        assert_eq!(cache.directory, Some(PathBuf::from("/tmp/cache")));
        let ocr = config.ocr.unwrap();
        assert_eq!(ocr.max_images, 1);
        assert_eq!(ocr.dpi, 72);
    }
}
