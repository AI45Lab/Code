use super::provider::{
    apply_model_caps, ModelConfig, ModelCost, ModelLimit, ModelModalities, ProviderConfig,
};
use super::{AutoDelegationConfig, CodeConfig, StorageBackend};
use crate::error::{CodeError, Result};
use crate::llm::LlmConfig;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ============================================================================
// ACL Parsing Helpers
// ============================================================================

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

fn acl_f32_attr(block: &a3s_acl::Block, keys: &[&str]) -> Option<f32> {
    match acl_attr(block, keys) {
        Some(a3s_acl::Value::Number(value)) => Some(*value as f32),
        _ => None,
    }
}

fn parse_auto_delegation_block(
    block: &a3s_acl::Block,
    base: &AutoDelegationConfig,
) -> AutoDelegationConfig {
    let mut config = base.clone();
    if let Some(enabled) = acl_bool_attr(block, &["enabled"]) {
        config.enabled = enabled;
    }
    if let Some(auto_parallel) =
        acl_bool_attr(block, &["auto_parallel", "autoParallel", "parallel"])
    {
        config.auto_parallel = auto_parallel;
    }
    if let Some(allow_manual_delegation) = acl_bool_attr(
        block,
        &[
            "allow_manual_delegation",
            "allowManualDelegation",
            "manual_delegation",
            "manualDelegation",
        ],
    ) {
        config.allow_manual_delegation = allow_manual_delegation;
    }
    if let Some(min_confidence) = acl_f32_attr(block, &["min_confidence", "minConfidence"]) {
        config.min_confidence = min_confidence.clamp(0.0, 1.0);
    }
    if let Some(max_tasks) = acl_usize_attr(block, &["max_tasks", "maxTasks"]) {
        config.max_tasks = max_tasks.max(1);
    }
    config
}

fn acl_u32(value: &a3s_acl::Value) -> Option<u32> {
    match value {
        a3s_acl::Value::Number(value) if *value >= 0.0 => {
            Some((*value as usize).min(u32::MAX as usize) as u32)
        }
        _ => None,
    }
}

fn acl_object_u32_attr(value: &a3s_acl::Value, key: &str) -> Option<u32> {
    match value {
        a3s_acl::Value::Object(pairs) => pairs
            .iter()
            .find_map(|(candidate, value)| (candidate == key).then(|| acl_u32(value)).flatten()),
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

// ============================================================================
// CodeConfig Implementation
// ============================================================================

impl CodeConfig {
    /// Create a new empty configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Load configuration from an ACL-compatible config file.
    ///
    /// `.acl` is the only supported config file extension. JSON and legacy
    /// `.hcl` config files are not supported.
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
    /// ACL (Agent Configuration Language) uses labeled blocks like
    /// `providers "openai" { }`.
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
                "max_parallel_tasks" => {
                    if let Some(max_parallel_tasks) =
                        acl_usize_attr(&block, &["max_parallel_tasks"])
                    {
                        config.max_parallel_tasks = Some(max_parallel_tasks);
                    }
                }
                "auto_parallel" | "auto_parallel_enabled" => {
                    if let Some(auto_parallel) =
                        acl_bool_attr(&block, &["auto_parallel", "auto_parallel_enabled"])
                    {
                        config.auto_parallel = Some(auto_parallel);
                    }
                }
                "auto_delegation" => {
                    config.auto_delegation =
                        parse_auto_delegation_block(&block, &config.auto_delegation);
                }
                "thinking_budget" => {
                    if let Some(thinking_budget) = acl_usize_attr(&block, &["thinking_budget"]) {
                        config.thinking_budget = Some(thinking_budget);
                    }
                }
                "providers" => {
                    let provider_name = block.labels.first().cloned().ok_or_else(|| {
                        CodeError::Config(
                            "providers block requires a label (e.g., providers \"openai\" { ... })"
                                .into(),
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
                            let model_name =
                                model_block.labels.first().cloned().ok_or_else(|| {
                                    CodeError::Config(
                                        "models block requires a label (e.g., models \"gpt-4\" { ... })"
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
                                    "maxTokens" => {
                                        tracing::warn!(
                                            provider = %provider.name,
                                            model = %model.id,
                                            field = "maxTokens",
                                            "Flat ACL model token limit fields are deprecated; use limit = {{ output = ..., context = ... }}"
                                        );
                                        if let Some(output) = acl_u32(value) {
                                            model.limit.output = output;
                                        }
                                    }
                                    "contextTokens" => {
                                        tracing::warn!(
                                            provider = %provider.name,
                                            model = %model.id,
                                            field = "contextTokens",
                                            "Flat ACL model token limit fields are deprecated; use limit = {{ output = ..., context = ... }}"
                                        );
                                        if let Some(context) = acl_u32(value) {
                                            model.limit.context = context;
                                        }
                                    }
                                    "limit" => {
                                        if let Some(output) = acl_object_u32_attr(value, "output") {
                                            model.limit.output = output;
                                        }
                                        if let Some(context) = acl_object_u32_attr(value, "context")
                                        {
                                            model.limit.context = context;
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

        if let Some(auto_parallel) = config.auto_parallel {
            config.auto_delegation.auto_parallel = auto_parallel;
        }

        Ok(config)
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
