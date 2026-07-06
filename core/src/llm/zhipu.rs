//! Zhipu AI (GLM) LLM client
//!
//! GLM uses an OpenAI-compatible API but with a different endpoint path.
//! This client wraps `OpenAiClient` with the correct GLM defaults.

use super::openai::OpenAiClient;
use super::structured;
use super::types::*;
use super::LlmClient;
use crate::retry::RetryConfig;
use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
#[cfg(test)]
use {super::http::HttpClient, std::sync::Arc};

const GLM_BASE_URL: &str = "https://open.bigmodel.cn";
const GLM_CHAT_PATH: &str = "/api/paas/v4/chat/completions";

/// Zhipu AI (GLM) client
pub struct ZhipuClient(OpenAiClient);

impl ZhipuClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self(
            OpenAiClient::new(api_key, model)
                .with_provider_name("zhipu")
                .with_base_url(GLM_BASE_URL.to_string())
                .with_chat_completions_path(GLM_CHAT_PATH),
        )
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.0 = self.0.with_temperature(temperature);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.0 = self.0.with_max_tokens(max_tokens);
        self
    }

    pub fn with_logprobs(mut self, enabled: bool) -> Self {
        self.0 = self.0.with_logprobs(enabled);
        self
    }

    pub fn with_top_logprobs(mut self, top_logprobs: usize) -> Self {
        self.0 = self.0.with_top_logprobs(top_logprobs);
        self
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.0 = self.0.with_base_url(base_url);
        self
    }

    pub fn with_retry_config(mut self, retry_config: RetryConfig) -> Self {
        self.0 = self.0.with_retry_config(retry_config);
        self
    }

    #[cfg(test)]
    pub fn with_http_client(mut self, http: Arc<dyn HttpClient>) -> Self {
        self.0 = self.0.with_http_client(http);
        self
    }
}

#[async_trait]
impl LlmClient for ZhipuClient {
    async fn complete(
        &self,
        messages: &[Message],
        system: Option<&str>,
        tools: &[ToolDefinition],
    ) -> Result<LlmResponse> {
        self.0.complete(messages, system, tools).await
    }

    async fn complete_streaming(
        &self,
        messages: &[Message],
        system: Option<&str>,
        tools: &[ToolDefinition],
        cancel_token: CancellationToken,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        self.0
            .complete_streaming(messages, system, tools, cancel_token)
            .await
    }

    fn native_structured_support(&self) -> structured::NativeStructuredSupport {
        self.0.native_structured_support()
    }

    async fn complete_structured(
        &self,
        messages: &[Message],
        system: Option<&str>,
        tools: &[ToolDefinition],
        directive: &structured::StructuredDirective,
    ) -> Result<LlmResponse> {
        self.0
            .complete_structured(messages, system, tools, directive)
            .await
    }

    async fn complete_streaming_structured(
        &self,
        messages: &[Message],
        system: Option<&str>,
        tools: &[ToolDefinition],
        directive: &structured::StructuredDirective,
        cancel_token: CancellationToken,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        self.0
            .complete_streaming_structured(messages, system, tools, directive, cancel_token)
            .await
    }
}
