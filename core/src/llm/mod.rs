//! LLM client abstraction layer
//!
//! Provides a unified interface for interacting with LLM providers
//! (Anthropic Claude, OpenAI, Zhipu AI GLM, and OpenAI-compatible providers).

pub mod anthropic;
pub mod factory;
pub mod http;
pub mod openai;
pub mod structured;
mod types;
pub mod zhipu;

// Re-export public types
pub use anthropic::AnthropicClient;
pub use factory::{create_client_with_config, LlmConfig};
pub use http::{
    clear_http_metrics_callback, default_http_client, set_http_metrics_callback, HttpClient,
    HttpMetricsCallback, HttpMetricsRecord, HttpResponse, StreamingHttpResponse,
};
pub use openai::OpenAiClient;
pub use types::*;
pub use zhipu::ZhipuClient;

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// LLM client trait
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Complete a conversation (non-streaming)
    async fn complete(
        &self,
        messages: &[Message],
        system: Option<&str>,
        tools: &[ToolDefinition],
    ) -> Result<LlmResponse>;

    /// Complete a conversation with streaming
    /// Returns a receiver for streaming events.
    /// The cancel_token is checked during the HTTP request; if cancelled, the request is aborted.
    async fn complete_streaming(
        &self,
        messages: &[Message],
        system: Option<&str>,
        tools: &[ToolDefinition],
        cancel_token: CancellationToken,
    ) -> Result<mpsc::Receiver<StreamEvent>>;

    /// Report the strongest provider-native structured-output enforcement this
    /// client supports. Used by [`structured`] to decide whether to force a
    /// tool call, request a native `response_format`, or fall back to
    /// prompt-and-parse. Defaults to no native support.
    fn native_structured_support(&self) -> structured::NativeStructuredSupport {
        structured::NativeStructuredSupport::None
    }

    /// Complete a conversation while honoring a structured-output directive
    /// (forced `tool_choice` and/or native `response_format`).
    ///
    /// The default implementation ignores the directive and behaves exactly
    /// like [`LlmClient::complete`], so existing clients keep working unchanged;
    /// providers that support native structured output override this.
    async fn complete_structured(
        &self,
        messages: &[Message],
        system: Option<&str>,
        tools: &[ToolDefinition],
        _directive: &structured::StructuredDirective,
    ) -> Result<LlmResponse> {
        self.complete(messages, system, tools).await
    }

    /// Streaming counterpart of [`LlmClient::complete_structured`]. Defaults to
    /// [`LlmClient::complete_streaming`], ignoring the directive.
    async fn complete_streaming_structured(
        &self,
        messages: &[Message],
        system: Option<&str>,
        tools: &[ToolDefinition],
        _directive: &structured::StructuredDirective,
        cancel_token: CancellationToken,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        self.complete_streaming(messages, system, tools, cancel_token)
            .await
    }
}

// Include test modules — these reference internal types via crate paths
#[cfg(test)]
#[path = "tests.rs"]
mod tests_file;
