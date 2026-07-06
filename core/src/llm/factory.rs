//! LLM client factory

use super::anthropic::AnthropicClient;
use super::openai::OpenAiClient;
use super::types::SecretString;
use super::zhipu::ZhipuClient;
use super::LlmClient;
use crate::retry::RetryConfig;
use std::collections::HashMap;
use std::sync::Arc;

/// LLM client configuration
#[derive(Clone, Default)]
pub struct LlmConfig {
    pub provider: String,
    pub model: String,
    pub api_key: SecretString,
    pub base_url: Option<String>,
    pub headers: HashMap<String, String>,
    pub session_id_header: Option<String>,
    pub session_id: Option<String>,
    pub retry_config: Option<RetryConfig>,
    /// Sampling temperature (0.0–1.0). None uses the provider default.
    pub temperature: Option<f32>,
    /// Maximum tokens to generate. None uses the client default.
    pub max_tokens: Option<usize>,
    /// Extended thinking budget in tokens (Anthropic only).
    pub thinking_budget: Option<usize>,
    /// Request token-level log probabilities from OpenAI-compatible providers.
    pub logprobs: Option<bool>,
    /// Number of alternative logprobs per token when logprobs are requested.
    pub top_logprobs: Option<usize>,
    /// When true, temperature is never sent to the API (e.g., o1 models).
    pub disable_temperature: bool,
}

impl std::fmt::Debug for LlmConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmConfig")
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("api_key", &"[REDACTED]")
            .field("base_url", &self.base_url)
            .field("headers", &self.headers.keys().collect::<Vec<_>>())
            .field("session_id_header", &self.session_id_header)
            .field(
                "session_id",
                &self.session_id.as_ref().map(|_| "[REDACTED]"),
            )
            .field("retry_config", &self.retry_config)
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .field("thinking_budget", &self.thinking_budget)
            .field("logprobs", &self.logprobs)
            .field("top_logprobs", &self.top_logprobs)
            .field("disable_temperature", &self.disable_temperature)
            .finish()
    }
}

impl LlmConfig {
    pub fn new(
        provider: impl Into<String>,
        model: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            api_key: SecretString::new(api_key.into()),
            base_url: None,
            headers: HashMap::new(),
            session_id_header: None,
            session_id: None,
            retry_config: None,
            temperature: None,
            max_tokens: None,
            thinking_budget: None,
            logprobs: None,
            top_logprobs: None,
            disable_temperature: false,
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers = headers;
        self
    }

    pub fn with_session_id_header(mut self, header_name: impl Into<String>) -> Self {
        self.session_id_header = Some(header_name.into());
        self
    }

    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn with_retry_config(mut self, retry_config: RetryConfig) -> Self {
        self.retry_config = Some(retry_config);
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_thinking_budget(mut self, budget: usize) -> Self {
        self.thinking_budget = Some(budget);
        self
    }

    pub fn with_logprobs(mut self, enabled: bool) -> Self {
        self.logprobs = Some(enabled);
        self
    }

    pub fn with_top_logprobs(mut self, top_logprobs: usize) -> Self {
        self.logprobs = Some(true);
        self.top_logprobs = Some(top_logprobs);
        self
    }

    pub(crate) fn resolved_headers(&self) -> HashMap<String, String> {
        let mut headers = self.headers.clone();
        if let (Some(header_name), Some(session_id)) = (&self.session_id_header, &self.session_id) {
            headers.insert(header_name.clone(), session_id.clone());
        }
        headers
    }
}

/// Create LLM client with full configuration (supports custom base_url)
pub fn create_client_with_config(config: LlmConfig) -> Arc<dyn LlmClient> {
    let retry = config.retry_config.clone().unwrap_or_default();
    let api_key = config.api_key.expose().to_string();
    let headers = config.resolved_headers();

    match config.provider.as_str() {
        "anthropic" | "claude" => {
            let mut client = AnthropicClient::new(api_key, config.model)
                .with_provider_name(config.provider.clone())
                .with_retry_config(retry);
            if let Some(base_url) = config.base_url {
                client = client.with_base_url(base_url);
            }
            if !config.disable_temperature {
                if let Some(temp) = config.temperature {
                    client = client.with_temperature(temp);
                }
            }
            if let Some(max) = config.max_tokens {
                client = client.with_max_tokens(max);
            }
            if let Some(budget) = config.thinking_budget {
                client = client.with_thinking_budget(budget);
            }
            Arc::new(client)
        }
        "openai" | "gpt" => {
            let mut client = OpenAiClient::new(api_key, config.model)
                .with_provider_name(config.provider.clone())
                .with_retry_config(retry);
            if let Some(base_url) = config.base_url {
                client = client.with_base_url(base_url);
            }
            if !headers.is_empty() {
                client = client.with_headers(headers.clone());
            }
            if !config.disable_temperature {
                if let Some(temp) = config.temperature {
                    client = client.with_temperature(temp);
                }
            }
            if let Some(max) = config.max_tokens {
                client = client.with_max_tokens(max);
            }
            if let Some(enabled) = config.logprobs {
                client = client.with_logprobs(enabled);
            }
            if let Some(top_logprobs) = config.top_logprobs {
                client = client.with_top_logprobs(top_logprobs);
            }
            Arc::new(client)
        }
        "glm" | "zhipu" | "bigmodel" => {
            let mut client = ZhipuClient::new(api_key, config.model).with_retry_config(retry);
            if let Some(base_url) = config.base_url {
                client = client.with_base_url(base_url);
            }
            if !config.disable_temperature {
                if let Some(temp) = config.temperature {
                    client = client.with_temperature(temp);
                }
            }
            if let Some(max) = config.max_tokens {
                client = client.with_max_tokens(max);
            }
            if let Some(enabled) = config.logprobs {
                client = client.with_logprobs(enabled);
            }
            if let Some(top_logprobs) = config.top_logprobs {
                client = client.with_top_logprobs(top_logprobs);
            }
            Arc::new(client)
        }
        // OpenAI-compatible providers (deepseek, groq, together, ollama, etc.)
        _ => {
            tracing::info!(
                "Using OpenAI-compatible client for provider '{}'",
                config.provider
            );
            let mut client = OpenAiClient::new(api_key, config.model)
                .with_provider_name(config.provider.clone())
                .with_retry_config(retry);
            if let Some(base_url) = config.base_url {
                client = client.with_base_url(base_url);
            }
            if !headers.is_empty() {
                client = client.with_headers(headers.clone());
            }
            if !config.disable_temperature {
                if let Some(temp) = config.temperature {
                    client = client.with_temperature(temp);
                }
            }
            if let Some(max) = config.max_tokens {
                client = client.with_max_tokens(max);
            }
            if let Some(enabled) = config.logprobs {
                client = client.with_logprobs(enabled);
            }
            if let Some(top_logprobs) = config.top_logprobs {
                client = client.with_top_logprobs(top_logprobs);
            }
            Arc::new(client)
        }
    }
}
