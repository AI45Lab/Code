//! Configuration module for A3S Code
//!
//! Provides configuration for:
//! - LLM providers and models (defaultModel in "provider/model" format, providers)
//! - Queue configuration (a3s-lane integration)
//! - Search configuration (a3s-search integration)
//! - Directories for dynamic skill and agent loading
//!
//! Configuration is loaded from HCL files or HCL strings only.
//! JSON support has been removed.

use crate::error::{CodeError, Result};
use crate::llm::LlmConfig;
use crate::memory::MemoryConfig;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
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

impl CodeConfig {
    /// Create a new empty configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Load configuration from an HCL file.
    ///
    /// Only `.hcl` files are supported. JSON support has been removed.
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            CodeError::Config(format!(
                "Failed to read config file {}: {}",
                path.display(),
                e
            ))
        })?;

        Self::from_hcl(&content).map_err(|e| {
            CodeError::Config(format!(
                "Failed to parse HCL config {}: {}",
                path.display(),
                e
            ))
        })
    }

    /// Parse configuration from an HCL string.
    ///
    /// HCL attributes use `snake_case` which is converted to `camelCase` for
    /// serde deserialization. Repeated blocks (e.g., `providers`, `models`)
    /// are collected into JSON arrays.
    pub fn from_hcl(content: &str) -> Result<Self> {
        let body: hcl::Body = hcl::from_str(content)
            .map_err(|e| CodeError::Config(format!("Failed to parse HCL: {}", e)))?;
        let json_value = hcl_body_to_json(&body);
        serde_json::from_value(json_value)
            .map_err(|e| CodeError::Config(format!("Failed to deserialize HCL config: {}", e)))
    }

    /// Save configuration to a JSON file (used for persistence)
    ///
    /// Note: This saves as JSON format. To use HCL format, manually create .hcl files.
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

        let mut config = LlmConfig::new(&provider.name, &model.id, api_key);
        if let Some(url) = base_url {
            config = config.with_base_url(url);
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

        let mut config = LlmConfig::new(&provider.name, &model.id, api_key);
        if let Some(url) = base_url {
            config = config.with_base_url(url);
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
// HCL Parsing Helpers
// ============================================================================

/// Block labels that should be collected into JSON arrays.
const HCL_ARRAY_BLOCKS: &[&str] = &["providers", "models", "mcp_servers"];

/// Block identifiers whose body attribute keys should be preserved verbatim
/// (not converted to camelCase). These blocks contain user-defined key-value maps
/// like environment variables or HTTP headers, not struct field names.
const HCL_VERBATIM_BLOCKS: &[&str] = &["env", "headers"];

/// Convert an HCL body into a JSON value with camelCase keys.
fn hcl_body_to_json(body: &hcl::Body) -> JsonValue {
    hcl_body_to_json_inner(body, false)
}

/// Inner conversion with `verbatim_keys` flag.
///
/// When `verbatim_keys` is true, attribute keys are preserved as-is
/// (used for blocks like `env { ... }` where keys are user data).
fn hcl_body_to_json_inner(body: &hcl::Body, verbatim_keys: bool) -> JsonValue {
    let mut map = serde_json::Map::new();

    // Process attributes (key = value)
    for attr in body.attributes() {
        let key = if verbatim_keys {
            attr.key.as_str().to_string()
        } else {
            snake_to_camel(attr.key.as_str())
        };
        let value = hcl_expr_to_json(attr.expr());
        map.insert(key, value);
    }

    // Process blocks (repeated structures like `providers { ... }`)
    for block in body.blocks() {
        let key = if verbatim_keys {
            block.identifier.as_str().to_string()
        } else {
            snake_to_camel(block.identifier.as_str())
        };
        // Blocks in HCL_VERBATIM_BLOCKS contain user-defined maps, not struct fields
        let child_verbatim = HCL_VERBATIM_BLOCKS.contains(&block.identifier.as_str());
        let block_value = hcl_body_to_json_inner(block.body(), child_verbatim);

        if HCL_ARRAY_BLOCKS.contains(&block.identifier.as_str()) {
            // Collect into array
            let arr = map
                .entry(key)
                .or_insert_with(|| JsonValue::Array(Vec::new()));
            if let JsonValue::Array(ref mut vec) = arr {
                vec.push(block_value);
            }
        } else {
            map.insert(key, block_value);
        }
    }

    JsonValue::Object(map)
}

/// Convert snake_case to camelCase.
fn snake_to_camel(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut capitalize_next = false;
    for ch in s.chars() {
        if ch == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            result.extend(ch.to_uppercase());
            capitalize_next = false;
        } else {
            result.push(ch);
        }
    }
    result
}

/// Convert an HCL expression to a JSON value.
fn hcl_expr_to_json(expr: &hcl::Expression) -> JsonValue {
    match expr {
        hcl::Expression::String(s) => JsonValue::String(s.clone()),
        hcl::Expression::Number(n) => {
            if let Some(i) = n.as_i64() {
                JsonValue::Number(i.into())
            } else if let Some(f) = n.as_f64() {
                serde_json::Number::from_f64(f)
                    .map(JsonValue::Number)
                    .unwrap_or(JsonValue::Null)
            } else {
                JsonValue::Null
            }
        }
        hcl::Expression::Bool(b) => JsonValue::Bool(*b),
        hcl::Expression::Null => JsonValue::Null,
        hcl::Expression::Array(arr) => JsonValue::Array(arr.iter().map(hcl_expr_to_json).collect()),
        hcl::Expression::Object(obj) => {
            // Object expression keys are user data (env vars, headers, etc.),
            // NOT struct field names — preserve them verbatim.
            let map: serde_json::Map<String, JsonValue> = obj
                .iter()
                .map(|(k, v)| {
                    let key = match k {
                        hcl::ObjectKey::Identifier(id) => id.as_str().to_string(),
                        hcl::ObjectKey::Expression(expr) => {
                            if let hcl::Expression::String(s) = expr {
                                s.clone()
                            } else {
                                format!("{:?}", expr)
                            }
                        }
                        _ => format!("{:?}", k),
                    };
                    (key, hcl_expr_to_json(v))
                })
                .collect();
            JsonValue::Object(map)
        }
        hcl::Expression::FuncCall(func_call) => eval_func_call(func_call),
        hcl::Expression::TemplateExpr(tmpl) => eval_template_expr(tmpl),
        _ => JsonValue::String(format!("{:?}", expr)),
    }
}

/// Evaluate an HCL function call expression.
///
/// Supported functions:
/// - `env("VAR_NAME")` — read environment variable, returns empty string if unset
fn eval_func_call(func_call: &hcl::expr::FuncCall) -> JsonValue {
    let name = func_call.name.name.as_str();
    match name {
        "env" => {
            if let Some(arg) = func_call.args.first() {
                let var_name = match arg {
                    hcl::Expression::String(s) => s.as_str(),
                    _ => {
                        tracing::warn!("env() expects a string argument, got: {:?}", arg);
                        return JsonValue::Null;
                    }
                };
                match std::env::var(var_name) {
                    Ok(val) => JsonValue::String(val),
                    Err(_) => {
                        tracing::debug!("env(\"{}\") is not set, returning null", var_name);
                        JsonValue::Null
                    }
                }
            } else {
                tracing::warn!("env() called with no arguments");
                JsonValue::Null
            }
        }
        _ => {
            tracing::warn!("Unsupported HCL function: {}()", name);
            JsonValue::String(format!("{}()", name))
        }
    }
}

/// Evaluate an HCL template expression (string interpolation).
///
/// For quoted strings like `"prefix-${env("VAR")}-suffix"`, the template contains
/// literal parts and interpolated expressions that we evaluate and concatenate.
fn eval_template_expr(tmpl: &hcl::expr::TemplateExpr) -> JsonValue {
    // TemplateExpr is either a quoted string or heredoc containing template directives.
    // We convert it to string representation — the hcl-rs library stores the raw template.
    // For simple cases, just return the string form. For interpolations, we'd need a
    // full template evaluator which hcl-rs doesn't provide.
    // Best effort: convert to display string.
    JsonValue::String(format!("{}", tmpl))
}

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
        let config_path = temp_dir.path().join("config.hcl");

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
                    models: vec![],
                },
                ProviderConfig {
                    name: "openai".to_string(),
                    api_key: Some("key2".to_string()),
                    base_url: None,
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
                models: vec![ModelConfig {
                    id: "claude-sonnet-4".to_string(),
                    name: "Claude Sonnet 4".to_string(),
                    family: "claude-sonnet".to_string(),
                    api_key: None,
                    base_url: None,
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
            models: vec![
                ModelConfig {
                    id: "gpt-4".to_string(),
                    name: "GPT-4".to_string(),
                    family: "gpt".to_string(),
                    api_key: None, // Uses provider key
                    base_url: None,
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
                    models: vec![
                        ModelConfig {
                            id: "claude-1".to_string(),
                            name: "Claude 1".to_string(),
                            family: "claude".to_string(),
                            api_key: None,
                            base_url: None,
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
                    models: vec![ModelConfig {
                        id: "gpt-4".to_string(),
                        name: "GPT-4".to_string(),
                        family: "gpt".to_string(),
                        api_key: None,
                        base_url: None,
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
            models: vec![ModelConfig {
                id: "claude-sonnet-4".to_string(),
                name: "Claude Sonnet 4".to_string(),
                family: "claude-sonnet".to_string(),
                api_key: None,
                base_url: None,
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
            models: vec![],
        };

        let model_with_key = ModelConfig {
            id: "test".to_string(),
            name: "".to_string(),
            family: "".to_string(),
            api_key: Some("model-key".to_string()),
            base_url: None,
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
    fn test_code_config_default_provider_config() {
        let config = CodeConfig {
            default_model: Some("anthropic/claude-sonnet-4".to_string()),
            providers: vec![ProviderConfig {
                name: "anthropic".to_string(),
                api_key: Some("sk-test".to_string()),
                base_url: None,
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
                models: vec![ModelConfig {
                    id: "claude-sonnet-4".to_string(),
                    name: "Claude Sonnet 4".to_string(),
                    family: "claude-sonnet".to_string(),
                    api_key: None,
                    base_url: None,
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
                models: vec![ModelConfig {
                    id: "claude-sonnet-4".to_string(),
                    name: "Claude Sonnet 4".to_string(),
                    family: "claude-sonnet".to_string(),
                    api_key: None,
                    base_url: None,
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
                    models: vec![ModelConfig {
                        id: "claude-sonnet-4".to_string(),
                        name: "".to_string(),
                        family: "".to_string(),
                        api_key: None,
                        base_url: None,
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
                    models: vec![ModelConfig {
                        id: "gpt-4o".to_string(),
                        name: "".to_string(),
                        family: "".to_string(),
                        api_key: None,
                        base_url: None,
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
                models: vec![],
            }],
            ..Default::default()
        };
        assert!(config.llm_config("anthropic", "nonexistent").is_none());
    }

    #[test]
    fn test_from_hcl_string() {
        let hcl = r#"
            default_model = "anthropic/claude-sonnet-4"

            providers {
                name    = "anthropic"
                api_key = "test-key"

                models {
                    id   = "claude-sonnet-4"
                    name = "Claude Sonnet 4"
                }
            }
        "#;

        let config = CodeConfig::from_hcl(hcl).unwrap();
        assert_eq!(
            config.default_model,
            Some("anthropic/claude-sonnet-4".to_string())
        );
        assert_eq!(config.providers.len(), 1);
        assert_eq!(config.providers[0].name, "anthropic");
        assert_eq!(config.providers[0].models.len(), 1);
        assert_eq!(config.providers[0].models[0].id, "claude-sonnet-4");
    }

    #[test]
    fn test_from_hcl_multi_provider() {
        let hcl = r#"
            default_model = "anthropic/claude-sonnet-4"

            providers {
                name    = "anthropic"
                api_key = "sk-ant-test"

                models {
                    id   = "claude-sonnet-4"
                    name = "Claude Sonnet 4"
                }

                models {
                    id        = "claude-opus-4"
                    name      = "Claude Opus 4"
                    reasoning = true
                }
            }

            providers {
                name    = "openai"
                api_key = "sk-test"

                models {
                    id   = "gpt-4o"
                    name = "GPT-4o"
                }
            }
        "#;

        let config = CodeConfig::from_hcl(hcl).unwrap();
        assert_eq!(config.providers.len(), 2);
        assert_eq!(config.providers[0].models.len(), 2);
        assert_eq!(config.providers[1].models.len(), 1);
        assert_eq!(config.providers[1].name, "openai");
    }

    #[test]
    fn test_snake_to_camel() {
        assert_eq!(snake_to_camel("default_model"), "defaultModel");
        assert_eq!(snake_to_camel("api_key"), "apiKey");
        assert_eq!(snake_to_camel("base_url"), "baseUrl");
        assert_eq!(snake_to_camel("name"), "name");
        assert_eq!(snake_to_camel("tool_call"), "toolCall");
    }

    #[test]
    fn test_from_file_auto_detect_hcl() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config_path = temp_dir.path().join("config.hcl");

        std::fs::write(
            &config_path,
            r#"
            default_model = "anthropic/claude-sonnet-4"

            providers {
                name    = "anthropic"
                api_key = "test-key"

                models {
                    id = "claude-sonnet-4"
                }
            }
        "#,
        )
        .unwrap();

        let config = CodeConfig::from_file(&config_path).unwrap();
        assert_eq!(
            config.default_model,
            Some("anthropic/claude-sonnet-4".to_string())
        );
    }

    #[test]
    fn test_from_hcl_with_queue_config() {
        let hcl = r#"
            default_model = "anthropic/claude-sonnet-4"

            providers {
                name    = "anthropic"
                api_key = "test-key"
            }

            queue {
                query_max_concurrency = 20
                execute_max_concurrency = 5
                enable_metrics = true
                enable_dlq = true
            }
        "#;

        let config = CodeConfig::from_hcl(hcl).unwrap();
        assert!(config.queue.is_some());
        let queue = config.queue.unwrap();
        assert_eq!(queue.query_max_concurrency, 20);
        assert_eq!(queue.execute_max_concurrency, 5);
        assert!(queue.enable_metrics);
        assert!(queue.enable_dlq);
    }

    #[test]
    fn test_from_hcl_with_search_config() {
        let hcl = r#"
            default_model = "anthropic/claude-sonnet-4"

            providers {
                name    = "anthropic"
                api_key = "test-key"
            }

            search {
                timeout = 30

                health {
                    max_failures = 5
                    suspend_seconds = 120
                }

                engine {
                    google {
                        enabled = true
                        weight = 1.5
                    }
                    bing {
                        enabled = true
                        weight = 1.0
                        timeout = 15
                    }
                }
            }
        "#;

        let config = CodeConfig::from_hcl(hcl).unwrap();
        assert!(config.search.is_some());
        let search = config.search.unwrap();
        assert_eq!(search.timeout, 30);
        assert!(search.health.is_some());
        let health = search.health.unwrap();
        assert_eq!(health.max_failures, 5);
        assert_eq!(health.suspend_seconds, 120);
        assert_eq!(search.engines.len(), 2);
        assert!(search.engines.contains_key("google"));
        assert!(search.engines.contains_key("bing"));
        let google = &search.engines["google"];
        assert!(google.enabled);
        assert_eq!(google.weight, 1.5);
        let bing = &search.engines["bing"];
        assert_eq!(bing.timeout, Some(15));
    }

    #[test]
    fn test_from_hcl_with_queue_and_search() {
        let hcl = r#"
            default_model = "anthropic/claude-sonnet-4"

            providers {
                name    = "anthropic"
                api_key = "test-key"
            }

            queue {
                query_max_concurrency = 10
                enable_metrics = true
            }

            search {
                timeout = 20
                engine {
                    duckduckgo {
                        enabled = true
                    }
                }
            }
        "#;

        let config = CodeConfig::from_hcl(hcl).unwrap();
        assert!(config.queue.is_some());
        assert!(config.search.is_some());
        assert_eq!(config.queue.unwrap().query_max_concurrency, 10);
        assert_eq!(config.search.unwrap().timeout, 20);
    }

    #[test]
    fn test_from_hcl_multiple_mcp_servers() {
        let hcl = r#"
            mcp_servers {
                name      = "fetch"
                transport = "stdio"
                command   = "npx"
                args      = ["-y", "@modelcontextprotocol/server-fetch"]
                enabled   = true
            }

            mcp_servers {
                name      = "puppeteer"
                transport = "stdio"
                command   = "npx"
                args      = ["-y", "@anthropic/mcp-server-puppeteer"]
                enabled   = true
            }

            mcp_servers {
                name      = "filesystem"
                transport = "stdio"
                command   = "npx"
                args      = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
                enabled   = false
            }
        "#;

        let config = CodeConfig::from_hcl(hcl).unwrap();
        assert_eq!(
            config.mcp_servers.len(),
            3,
            "all 3 mcp_servers blocks should be parsed"
        );
        assert_eq!(config.mcp_servers[0].name, "fetch");
        assert_eq!(config.mcp_servers[1].name, "puppeteer");
        assert_eq!(config.mcp_servers[2].name, "filesystem");
        assert!(config.mcp_servers[0].enabled);
        assert!(!config.mcp_servers[2].enabled);
    }

    #[test]
    fn test_from_hcl_with_advanced_queue_config() {
        let hcl = r#"
            default_model = "anthropic/claude-sonnet-4"

            providers {
                name    = "anthropic"
                api_key = "test-key"
            }

            queue {
                query_max_concurrency = 20
                enable_metrics = true

                retry_policy {
                    strategy = "exponential"
                    max_retries = 5
                    initial_delay_ms = 200
                }

                rate_limit {
                    limit_type = "per_second"
                    max_operations = 100
                }

                priority_boost {
                    strategy = "standard"
                    deadline_ms = 300000
                }

                pressure_threshold = 50
            }
        "#;

        let config = CodeConfig::from_hcl(hcl).unwrap();
        assert!(config.queue.is_some());
        let queue = config.queue.unwrap();

        assert_eq!(queue.query_max_concurrency, 20);
        assert!(queue.enable_metrics);

        // Test retry policy
        assert!(queue.retry_policy.is_some());
        let retry = queue.retry_policy.unwrap();
        assert_eq!(retry.strategy, "exponential");
        assert_eq!(retry.max_retries, 5);
        assert_eq!(retry.initial_delay_ms, 200);

        // Test rate limit
        assert!(queue.rate_limit.is_some());
        let rate = queue.rate_limit.unwrap();
        assert_eq!(rate.limit_type, "per_second");
        assert_eq!(rate.max_operations, Some(100));

        // Test priority boost
        assert!(queue.priority_boost.is_some());
        let boost = queue.priority_boost.unwrap();
        assert_eq!(boost.strategy, "standard");
        assert_eq!(boost.deadline_ms, Some(300000));

        // Test pressure threshold
        assert_eq!(queue.pressure_threshold, Some(50));
    }

    #[test]
    fn test_hcl_env_function_resolved() {
        // Set a test env var
        std::env::set_var("A3S_TEST_HCL_KEY", "test-secret-key-123");

        let hcl_str = r#"
            providers {
                name    = "test"
                api_key = env("A3S_TEST_HCL_KEY")
            }
        "#;

        let body: hcl::Body = hcl::from_str(hcl_str).unwrap();
        let json = hcl_body_to_json(&body);

        // The providers block should be an array
        let providers = json.get("providers").unwrap();
        let provider = providers.as_array().unwrap().first().unwrap();
        let api_key = provider.get("apiKey").unwrap();

        assert_eq!(api_key.as_str().unwrap(), "test-secret-key-123");

        // Clean up
        std::env::remove_var("A3S_TEST_HCL_KEY");
    }

    #[test]
    fn test_hcl_env_function_unset_returns_null() {
        // Make sure this var doesn't exist
        std::env::remove_var("A3S_TEST_NONEXISTENT_VAR_12345");

        let hcl_str = r#"
            providers {
                name    = "test"
                api_key = env("A3S_TEST_NONEXISTENT_VAR_12345")
            }
        "#;

        let body: hcl::Body = hcl::from_str(hcl_str).unwrap();
        let json = hcl_body_to_json(&body);

        let providers = json.get("providers").unwrap();
        let provider = providers.as_array().unwrap().first().unwrap();
        let api_key = provider.get("apiKey").unwrap();

        assert!(api_key.is_null(), "Unset env var should return null");
    }

    #[test]
    fn test_hcl_mcp_env_block_preserves_var_names() {
        // env block keys are environment variable names — must NOT be camelCase'd
        std::env::set_var("A3S_TEST_SECRET", "my-secret");

        let hcl_str = r#"
            mcp_servers {
                name      = "test-server"
                transport = "stdio"
                command   = "echo"
                env = {
                    API_KEY           = "sk-test-123"
                    ANTHROPIC_API_KEY = env("A3S_TEST_SECRET")
                    SIMPLE            = "value"
                }
            }
        "#;

        let body: hcl::Body = hcl::from_str(hcl_str).unwrap();
        let json = hcl_body_to_json(&body);

        let servers = json.get("mcpServers").unwrap().as_array().unwrap();
        let server = &servers[0];
        let env = server.get("env").unwrap().as_object().unwrap();

        // Keys must be preserved verbatim, not converted to camelCase
        assert_eq!(env.get("API_KEY").unwrap().as_str().unwrap(), "sk-test-123");
        assert_eq!(
            env.get("ANTHROPIC_API_KEY").unwrap().as_str().unwrap(),
            "my-secret"
        );
        assert_eq!(env.get("SIMPLE").unwrap().as_str().unwrap(), "value");

        // These mangled keys must NOT exist
        assert!(
            env.get("apiKey").is_none(),
            "env var key should not be camelCase'd"
        );
        assert!(
            env.get("APIKEY").is_none(),
            "env var key should not have underscores stripped"
        );
        assert!(env.get("anthropicApiKey").is_none());

        std::env::remove_var("A3S_TEST_SECRET");
    }

    #[test]
    fn test_hcl_mcp_env_as_block_syntax() {
        // Test block syntax: env { KEY = "value" } (no equals sign)
        let hcl_str = r#"
            mcp_servers {
                name      = "test-server"
                transport = "stdio"
                command   = "echo"
                env {
                    MY_VAR     = "hello"
                    OTHER_VAR  = "world"
                }
            }
        "#;

        let body: hcl::Body = hcl::from_str(hcl_str).unwrap();
        let json = hcl_body_to_json(&body);

        let servers = json.get("mcpServers").unwrap().as_array().unwrap();
        let server = &servers[0];
        let env = server.get("env").unwrap().as_object().unwrap();

        assert_eq!(env.get("MY_VAR").unwrap().as_str().unwrap(), "hello");
        assert_eq!(env.get("OTHER_VAR").unwrap().as_str().unwrap(), "world");
        assert!(
            env.get("myVar").is_none(),
            "block env keys should not be camelCase'd"
        );
    }

    #[test]
    fn test_hcl_mcp_full_deserialization_with_env() {
        // End-to-end: HCL string → CodeConfig with McpServerConfig.env populated
        std::env::set_var("A3S_TEST_MCP_KEY", "resolved-secret");

        let hcl_str = r#"
            mcp_servers {
                name      = "fetch"
                transport = "stdio"
                command   = "npx"
                args      = ["-y", "@modelcontextprotocol/server-fetch"]
                env = {
                    NODE_ENV = "production"
                    API_KEY  = env("A3S_TEST_MCP_KEY")
                }
                tool_timeout_secs = 120
            }
        "#;

        let config = CodeConfig::from_hcl(hcl_str).unwrap();
        assert_eq!(config.mcp_servers.len(), 1);

        let server = &config.mcp_servers[0];
        assert_eq!(server.name, "fetch");
        assert_eq!(server.env.get("NODE_ENV").unwrap(), "production");
        assert_eq!(server.env.get("API_KEY").unwrap(), "resolved-secret");
        assert_eq!(server.tool_timeout_secs, 120);

        std::env::remove_var("A3S_TEST_MCP_KEY");
    }
}
