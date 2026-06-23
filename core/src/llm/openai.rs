//! OpenAI-compatible LLM client

use super::http::{default_http_client, normalize_base_url, HttpClient};
use super::structured;
use super::types::*;
use super::LlmClient;
use crate::llm::types::{ToolResultContent, ToolResultContentField};
use crate::retry::{AttemptOutcome, RetryConfig};
use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

/// OpenAI client
pub struct OpenAiClient {
    pub(crate) provider_name: String,
    pub(crate) api_key: SecretString,
    pub(crate) model: String,
    pub(crate) base_url: String,
    pub(crate) chat_completions_path: String,
    pub(crate) headers: HashMap<String, String>,
    pub(crate) temperature: Option<f32>,
    pub(crate) max_tokens: Option<usize>,
    pub(crate) http: Arc<dyn HttpClient>,
    pub(crate) retry_config: RetryConfig,
}

impl OpenAiClient {
    pub(crate) fn parse_tool_arguments(tool_name: &str, arguments: &str) -> serde_json::Value {
        if arguments.trim().is_empty() {
            return serde_json::Value::Object(Default::default());
        }

        serde_json::from_str(arguments).unwrap_or_else(|e| {
            tracing::warn!(
                "Failed to parse tool arguments JSON for tool '{}': {}",
                tool_name,
                e
            );
            serde_json::json!({
                "__parse_error": format!(
                    "Malformed tool arguments: {}. Raw input: {}",
                    e, arguments
                )
            })
        })
    }

    fn merge_stream_text(text_content: &mut String, incoming: &str) -> Option<String> {
        if incoming.is_empty() {
            return None;
        }
        if text_content.is_empty() {
            text_content.push_str(incoming);
            return Some(incoming.to_string());
        }
        if incoming == text_content.as_str() || text_content.ends_with(incoming) {
            return None;
        }
        // If incoming contains text_content as a prefix (incoming is the full content),
        // replace text_content instead of appending (avoids duplicate full content)
        if incoming.starts_with(text_content.as_str()) && incoming.len() > text_content.len() {
            let suffix = &incoming[text_content.len()..];
            if !suffix.is_empty() {
                *text_content = incoming.to_string();
                return Some(suffix.to_string());
            }
            return None;
        }
        if let Some(suffix) = incoming.strip_prefix(text_content.as_str()) {
            if suffix.is_empty() {
                return None;
            }
            text_content.push_str(suffix);
            return Some(suffix.to_string());
        }
        text_content.push_str(incoming);
        Some(incoming.to_string())
    }

    pub fn new(api_key: String, model: String) -> Self {
        Self {
            provider_name: "openai".to_string(),
            api_key: SecretString::new(api_key),
            model,
            base_url: "https://api.openai.com".to_string(),
            chat_completions_path: "/v1/chat/completions".to_string(),
            headers: HashMap::new(),
            temperature: None,
            max_tokens: None,
            http: default_http_client(),
            retry_config: RetryConfig::default(),
        }
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = normalize_base_url(&base_url);
        self
    }

    pub fn with_provider_name(mut self, provider_name: impl Into<String>) -> Self {
        self.provider_name = provider_name.into();
        self
    }

    pub fn with_chat_completions_path(mut self, path: impl Into<String>) -> Self {
        let path = path.into();
        self.chat_completions_path = if path.starts_with('/') {
            path
        } else {
            format!("/{}", path)
        };
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers = headers;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_retry_config(mut self, retry_config: RetryConfig) -> Self {
        self.retry_config = retry_config;
        self
    }

    pub fn with_http_client(mut self, http: Arc<dyn HttpClient>) -> Self {
        self.http = http;
        self
    }

    pub(crate) fn request_headers(&self) -> Vec<(String, String)> {
        let mut headers = Vec::with_capacity(self.headers.len() + 1);
        let has_authorization = self
            .headers
            .keys()
            .any(|key| key.eq_ignore_ascii_case("authorization"));
        if !has_authorization {
            headers.push((
                "Authorization".to_string(),
                format!("Bearer {}", self.api_key.expose()),
            ));
        }
        headers.extend(
            self.headers
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
        headers
    }

    pub(crate) fn convert_messages(&self, messages: &[Message]) -> Vec<serde_json::Value> {
        messages
            .iter()
            .map(|msg| {
                let content: serde_json::Value = if msg.content.len() == 1 {
                    match &msg.content[0] {
                        ContentBlock::Text { text } => serde_json::json!(text),
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            let content_str = match content {
                                ToolResultContentField::Text(s) => s.clone(),
                                ToolResultContentField::Blocks(blocks) => blocks
                                    .iter()
                                    .filter_map(|b| {
                                        if let ToolResultContent::Text { text } = b {
                                            Some(text.clone())
                                        } else {
                                            None
                                        }
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                            };
                            return serde_json::json!({
                                "role": "tool",
                                "tool_call_id": tool_use_id,
                                "content": content_str,
                            });
                        }
                        _ => serde_json::json!(""),
                    }
                } else {
                    serde_json::json!(msg
                        .content
                        .iter()
                        .map(|block| {
                            match block {
                                ContentBlock::Text { text } => serde_json::json!({
                                    "type": "text",
                                    "text": text,
                                }),
                                ContentBlock::Image { source } => serde_json::json!({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": format!(
                                            "data:{};base64,{}",
                                            source.media_type, source.data
                                        ),
                                    }
                                }),
                                ContentBlock::ToolUse { id, name, input } => serde_json::json!({
                                    "type": "function",
                                    "id": id,
                                    "function": {
                                        "name": name,
                                        "arguments": input.to_string(),
                                    }
                                }),
                                _ => serde_json::json!({}),
                            }
                        })
                        .collect::<Vec<_>>())
                };

                // Handle assistant messages — kimi-k2.5 requires reasoning_content
                // on all assistant messages when thinking mode is enabled
                if msg.role == "assistant" {
                    let rc = msg.reasoning_content.as_deref().unwrap_or("");
                    let tool_calls: Vec<_> = msg.tool_calls();
                    if !tool_calls.is_empty() {
                        return serde_json::json!({
                            "role": "assistant",
                            "content": msg.text(),
                            "reasoning_content": rc,
                            "tool_calls": tool_calls.iter().map(|tc| {
                                serde_json::json!({
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": tc.args.to_string(),
                                    }
                                })
                            }).collect::<Vec<_>>(),
                        });
                    }
                    return serde_json::json!({
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": rc,
                    });
                }

                serde_json::json!({
                    "role": msg.role,
                    "content": content,
                })
            })
            .collect()
    }

    pub(crate) fn convert_tools(&self, tools: &[ToolDefinition]) -> Vec<serde_json::Value> {
        tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                })
            })
            .collect()
    }
}

impl OpenAiClient {
    /// Apply a structured-output directive to an OpenAI-compatible request.
    ///
    /// OpenAI-compatible APIs support both forced function `tool_choice` and
    /// native `response_format` (`json_object` / `json_schema` + `strict`).
    fn apply_directive(
        request: &mut serde_json::Value,
        directive: &structured::StructuredDirective,
    ) {
        if let Some(tool) = &directive.force_tool {
            request["tool_choice"] = serde_json::json!({
                "type": "function",
                "function": { "name": tool }
            });
        }
        if let Some(rf) = &directive.response_format {
            request["response_format"] = match rf {
                structured::ResponseFormat::JsonObject => {
                    serde_json::json!({ "type": "json_object" })
                }
                structured::ResponseFormat::JsonSchema { name, schema } => serde_json::json!({
                    "type": "json_schema",
                    "json_schema": { "name": name, "schema": schema, "strict": true }
                }),
            };
        }
    }

    /// Build a chat-completions request body, optionally applying a directive.
    fn build_chat_request(
        &self,
        messages: &[Message],
        system: Option<&str>,
        tools: &[ToolDefinition],
        directive: Option<&structured::StructuredDirective>,
    ) -> serde_json::Value {
        let mut openai_messages = Vec::new();

        if let Some(sys) = system {
            openai_messages.push(serde_json::json!({
                "role": "system",
                "content": sys,
            }));
        }

        openai_messages.extend(self.convert_messages(messages));

        let mut request = serde_json::json!({
            "model": self.model,
            "messages": openai_messages,
        });

        if let Some(temp) = self.temperature {
            request["temperature"] = serde_json::json!(temp);
        }
        if let Some(max) = self.max_tokens {
            request["max_tokens"] = serde_json::json!(max);
        }

        if !tools.is_empty() {
            request["tools"] = serde_json::json!(self.convert_tools(tools));
        }

        if let Some(directive) = directive {
            Self::apply_directive(&mut request, directive);
        }

        request
    }

    /// Execute a fully-built (non-streaming) chat-completions request.
    async fn send_request(&self, request: serde_json::Value) -> Result<LlmResponse> {
        {
            let request_started_at = Instant::now();
            let url = format!("{}{}", self.base_url, self.chat_completions_path);
            let request_headers = self.request_headers();

            let response = crate::retry::with_retry(&self.retry_config, |_attempt| {
                let http = &self.http;
                let url = &url;
                let request_headers = request_headers.clone();
                let request = &request;
                async move {
                    let headers = request_headers
                        .iter()
                        .map(|(key, value)| (key.as_str(), value.as_str()))
                        .collect::<Vec<_>>();
                    // Non-streaming: use a non-cancelled token for now
                    let cancel_token = tokio_util::sync::CancellationToken::new();
                    match http.post(url, headers, request, cancel_token).await {
                        Ok(resp) => {
                            let status = reqwest::StatusCode::from_u16(resp.status)
                                .unwrap_or(reqwest::StatusCode::INTERNAL_SERVER_ERROR);
                            if status.is_success() {
                                AttemptOutcome::Success(resp.body)
                            } else if self.retry_config.is_retryable_status(status) {
                                AttemptOutcome::Retryable {
                                    status,
                                    body: resp.body,
                                    retry_after: None,
                                }
                            } else {
                                AttemptOutcome::Fatal(anyhow::anyhow!(
                                    "OpenAI API error at {} ({}): {}",
                                    url,
                                    status,
                                    resp.body
                                ))
                            }
                        }
                        Err(e) => {
                            tracing::error!("HTTP error: {e:?}");
                            AttemptOutcome::Fatal(e)
                        }
                    }
                }
            })
            .await?;

            let parsed: OpenAiResponse =
                serde_json::from_str(&response).context("Failed to parse OpenAI response")?;

            let choice = parsed.choices.into_iter().next().context("No choices")?;

            let mut content = vec![];

            let reasoning_content = choice.message.reasoning_content;

            let text_content = choice.message.content;

            if let Some(text) = text_content {
                if !text.is_empty() {
                    content.push(ContentBlock::Text { text });
                }
            }

            if let Some(tool_calls) = choice.message.tool_calls {
                for tc in tool_calls {
                    content.push(ContentBlock::ToolUse {
                        id: tc.id,
                        name: tc.function.name.clone(),
                        input: Self::parse_tool_arguments(
                            &tc.function.name,
                            &tc.function.arguments,
                        ),
                    });
                }
            }

            let llm_response = LlmResponse {
                message: Message {
                    role: "assistant".to_string(),
                    content,
                    reasoning_content,
                },
                usage: TokenUsage {
                    prompt_tokens: parsed.usage.prompt_tokens,
                    completion_tokens: parsed.usage.completion_tokens,
                    total_tokens: {
                        let t = parsed.usage.total_tokens;
                        // MiniMax: fall back to total_characters when total_tokens is 0.
                        if t == 0 {
                            parsed.usage.total_characters.unwrap_or(0)
                        } else {
                            t
                        }
                    },
                    cache_read_tokens: parsed
                        .usage
                        .prompt_tokens_details
                        .as_ref()
                        .and_then(|d| d.cached_tokens),
                    cache_write_tokens: None,
                },
                stop_reason: choice.finish_reason,
                meta: Some(LlmResponseMeta {
                    provider: Some(self.provider_name.clone()),
                    request_model: Some(self.model.clone()),
                    request_url: Some(url.clone()),
                    response_id: parsed.id,
                    response_model: parsed.model,
                    response_object: parsed.object,
                    first_token_ms: None,
                    duration_ms: Some(request_started_at.elapsed().as_millis() as u64),
                }),
            };

            crate::telemetry::record_llm_usage(
                llm_response.usage.prompt_tokens,
                llm_response.usage.completion_tokens,
                llm_response.usage.total_tokens,
                llm_response.stop_reason.as_deref(),
            );

            Ok(llm_response)
        }
    }
}

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn complete(
        &self,
        messages: &[Message],
        system: Option<&str>,
        tools: &[ToolDefinition],
    ) -> Result<LlmResponse> {
        self.send_request(self.build_chat_request(messages, system, tools, None))
            .await
    }

    async fn complete_structured(
        &self,
        messages: &[Message],
        system: Option<&str>,
        tools: &[ToolDefinition],
        directive: &structured::StructuredDirective,
    ) -> Result<LlmResponse> {
        self.send_request(self.build_chat_request(messages, system, tools, Some(directive)))
            .await
    }

    fn native_structured_support(&self) -> structured::NativeStructuredSupport {
        structured::NativeStructuredSupport::JsonSchema
    }

    async fn complete_streaming(
        &self,
        messages: &[Message],
        system: Option<&str>,
        tools: &[ToolDefinition],
        cancel_token: tokio_util::sync::CancellationToken,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        {
            let request_started_at = Instant::now();
            let mut openai_messages = Vec::new();

            if let Some(sys) = system {
                openai_messages.push(serde_json::json!({
                    "role": "system",
                    "content": sys,
                }));
            }

            openai_messages.extend(self.convert_messages(messages));

            let mut request = serde_json::json!({
                "model": self.model,
                "messages": openai_messages,
                "stream": true,
                "stream_options": { "include_usage": true },
            });

            if let Some(temp) = self.temperature {
                request["temperature"] = serde_json::json!(temp);
            }
            if let Some(max) = self.max_tokens {
                request["max_tokens"] = serde_json::json!(max);
            }

            if !tools.is_empty() {
                request["tools"] = serde_json::json!(self.convert_tools(tools));
            }

            let url = format!("{}{}", self.base_url, self.chat_completions_path);
            let request_headers = self.request_headers();

            let streaming_resp = crate::retry::with_retry(&self.retry_config, |_attempt| {
                let http = &self.http;
                let url = &url;
                let request_headers = request_headers.clone();
                let request = &request;
                let cancel_token = cancel_token.clone();
                async move {
                    let headers = request_headers
                        .iter()
                        .map(|(key, value)| (key.as_str(), value.as_str()))
                        .collect::<Vec<_>>();
                    // Wrap in tokio::select! so cancellation aborts the HTTP request mid-flight
                    let resp = tokio::select! {
                        _ = cancel_token.cancelled() => {
                            return AttemptOutcome::Fatal(anyhow::anyhow!("HTTP request cancelled"));
                        }
                        result = http.post_streaming(url, headers, request, cancel_token.clone()) => {
                            match result {
                                Ok(r) => r,
                                Err(e) => {
                                    return AttemptOutcome::Fatal(anyhow::anyhow!("HTTP request failed: {}", e));
                                }
                            }
                        }
                    };
                    let status = reqwest::StatusCode::from_u16(resp.status)
                        .unwrap_or(reqwest::StatusCode::INTERNAL_SERVER_ERROR);
                    if status.is_success() {
                        AttemptOutcome::Success(resp)
                    } else {
                        let retry_after = resp
                            .retry_after
                            .as_deref()
                            .and_then(|v| RetryConfig::parse_retry_after(Some(v)));
                        if self.retry_config.is_retryable_status(status) {
                            AttemptOutcome::Retryable {
                                status,
                                body: resp.error_body,
                                retry_after,
                            }
                        } else {
                            AttemptOutcome::Fatal(anyhow::anyhow!(
                                "OpenAI API error at {} ({}): {}",
                                url,
                                status,
                                resp.error_body
                            ))
                        }
                    }
                }
            })
            .await?;

            let (tx, rx) = mpsc::channel(100);

            let mut stream = streaming_resp.byte_stream;
            let provider_name = self.provider_name.clone();
            let request_model = self.model.clone();
            let request_url = url.clone();
            tokio::spawn(async move {
                let mut buffer = String::new();
                let mut content_blocks: Vec<ContentBlock> = Vec::new();
                let mut text_content = String::new();
                let mut reasoning_content_accum = String::new();
                let mut tool_calls: std::collections::BTreeMap<usize, (String, String, String)> =
                    std::collections::BTreeMap::new();
                let mut usage = TokenUsage::default();
                let mut finish_reason = None;
                let mut response_id = None;
                let mut response_model = None;
                let mut response_object = None;
                let mut first_token_ms = None;
                let mut saw_done = false;
                let mut parsed_any_event = false;

                while let Some(chunk_result) = stream.next().await {
                    let chunk = match chunk_result {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::error!("Stream error: {}", e);
                            break;
                        }
                    };

                    buffer.push_str(&String::from_utf8_lossy(&chunk));

                    while let Some(event_end) = buffer.find("\n\n") {
                        let event_data: String = buffer.drain(..event_end).collect();
                        buffer.drain(..2);

                        for line in event_data.lines() {
                            if let Some(data) = line.strip_prefix("data: ") {
                                if data == "[DONE]" {
                                    saw_done = true;
                                    if !text_content.is_empty() {
                                        content_blocks.push(ContentBlock::Text {
                                            text: text_content.clone(),
                                        });
                                    }
                                    for (_, (id, name, args)) in tool_calls.iter() {
                                        content_blocks.push(ContentBlock::ToolUse {
                                            id: id.clone(),
                                            name: name.clone(),
                                            input: Self::parse_tool_arguments(name, args),
                                        });
                                    }
                                    tool_calls.clear();
                                    crate::telemetry::record_llm_usage(
                                        usage.prompt_tokens,
                                        usage.completion_tokens,
                                        usage.total_tokens,
                                        finish_reason.as_deref(),
                                    );
                                    let response = LlmResponse {
                                        message: Message {
                                            role: "assistant".to_string(),
                                            content: std::mem::take(&mut content_blocks),
                                            reasoning_content: if reasoning_content_accum.is_empty()
                                            {
                                                None
                                            } else {
                                                Some(std::mem::take(&mut reasoning_content_accum))
                                            },
                                        },
                                        usage: usage.clone(),
                                        stop_reason: std::mem::take(&mut finish_reason),
                                        meta: Some(LlmResponseMeta {
                                            provider: Some(provider_name.clone()),
                                            request_model: Some(request_model.clone()),
                                            request_url: Some(request_url.clone()),
                                            response_id: response_id.clone(),
                                            response_model: response_model.clone(),
                                            response_object: response_object.clone(),
                                            first_token_ms,
                                            duration_ms: Some(
                                                request_started_at.elapsed().as_millis() as u64,
                                            ),
                                        }),
                                    };
                                    let _ = tx.send(StreamEvent::Done(response)).await;
                                    continue;
                                }

                                if let Ok(event) = serde_json::from_str::<OpenAiStreamChunk>(data) {
                                    parsed_any_event = true;
                                    if response_id.is_none() {
                                        response_id = event.id.clone();
                                    }
                                    if response_model.is_none() {
                                        response_model = event.model.clone();
                                    }
                                    if response_object.is_none() {
                                        response_object = event.object.clone();
                                    }
                                    if let Some(u) = event.usage {
                                        usage.prompt_tokens = u.prompt_tokens;
                                        usage.completion_tokens = u.completion_tokens;
                                        usage.total_tokens = u.total_tokens;
                                        // MiniMax: fall back to total_characters when total_tokens is 0.
                                        if usage.total_tokens == 0 {
                                            usage.total_tokens = u.total_characters.unwrap_or(0);
                                        }
                                        usage.cache_read_tokens = u
                                            .prompt_tokens_details
                                            .as_ref()
                                            .and_then(|d| d.cached_tokens);
                                    }

                                    if let Some(choice) = event.choices.into_iter().next() {
                                        if let Some(reason) = choice.finish_reason {
                                            finish_reason = Some(reason);
                                        }

                                        if let Some(message) = choice.message {
                                            // If text_content already has content (from delta processing),
                                            // skip message.content to avoid sending duplicate full content
                                            let skip_content = !text_content.is_empty();
                                            if let Some(reasoning) = message.reasoning_content {
                                                reasoning_content_accum.push_str(&reasoning);
                                                if first_token_ms.is_none() {
                                                    first_token_ms = Some(
                                                        request_started_at.elapsed().as_millis()
                                                            as u64,
                                                    );
                                                }
                                                if let Some(delta) = Self::merge_stream_text(
                                                    &mut text_content,
                                                    &reasoning,
                                                ) {
                                                    let _ = tx
                                                        .send(StreamEvent::ReasoningDelta(delta))
                                                        .await;
                                                }
                                            }
                                            if !skip_content {
                                                if let Some(content) = message
                                                    .content
                                                    .filter(|value| !value.is_empty())
                                                {
                                                    if first_token_ms.is_none() {
                                                        first_token_ms = Some(
                                                            request_started_at.elapsed().as_millis()
                                                                as u64,
                                                        );
                                                    }
                                                    if let Some(delta) = Self::merge_stream_text(
                                                        &mut text_content,
                                                        &content,
                                                    ) {
                                                        let _ = tx
                                                            .send(StreamEvent::TextDelta(delta))
                                                            .await;
                                                    }
                                                }
                                            }
                                            if let Some(tcs) = message.tool_calls {
                                                for (index, tc) in tcs.into_iter().enumerate() {
                                                    tool_calls.insert(
                                                        index,
                                                        (
                                                            tc.id,
                                                            tc.function.name,
                                                            tc.function.arguments,
                                                        ),
                                                    );
                                                }
                                            }
                                        } else if let Some(delta) = choice.delta {
                                            if let Some(ref rc) = delta.reasoning_content {
                                                reasoning_content_accum.push_str(rc);
                                                if first_token_ms.is_none() {
                                                    first_token_ms = Some(
                                                        request_started_at.elapsed().as_millis()
                                                            as u64,
                                                    );
                                                }
                                                if let Some(delta) =
                                                    Self::merge_stream_text(&mut text_content, rc)
                                                {
                                                    let _ = tx
                                                        .send(StreamEvent::ReasoningDelta(delta))
                                                        .await;
                                                }
                                            }

                                            if let Some(content) = delta.content {
                                                if first_token_ms.is_none() {
                                                    first_token_ms = Some(
                                                        request_started_at.elapsed().as_millis()
                                                            as u64,
                                                    );
                                                }
                                                if let Some(delta) = Self::merge_stream_text(
                                                    &mut text_content,
                                                    &content,
                                                ) {
                                                    let _ = tx
                                                        .send(StreamEvent::TextDelta(delta))
                                                        .await;
                                                }
                                            }

                                            if let Some(tcs) = delta.tool_calls {
                                                for tc in tcs {
                                                    let entry = tool_calls
                                                        .entry(tc.index)
                                                        .or_insert_with(|| {
                                                            (
                                                                String::new(),
                                                                String::new(),
                                                                String::new(),
                                                            )
                                                        });

                                                    if let Some(id) = tc.id {
                                                        entry.0 = id;
                                                    }
                                                    if let Some(func) = tc.function {
                                                        if let Some(name) = func.name {
                                                            if first_token_ms.is_none() {
                                                                first_token_ms = Some(
                                                                    request_started_at
                                                                        .elapsed()
                                                                        .as_millis()
                                                                        as u64,
                                                                );
                                                            }
                                                            entry.1 = name.clone();
                                                            let _ = tx
                                                                .send(StreamEvent::ToolUseStart {
                                                                    id: entry.0.clone(),
                                                                    name,
                                                                })
                                                                .await;
                                                        }
                                                        if let Some(args) = func.arguments {
                                                            entry.2.push_str(&args);
                                                            let _ = tx
                                                                .send(
                                                                    StreamEvent::ToolUseInputDelta(
                                                                        args,
                                                                    ),
                                                                )
                                                                .await;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if saw_done {
                    return;
                }

                let trailing = buffer.trim();
                if !trailing.is_empty() {
                    if let Ok(event) = serde_json::from_str::<OpenAiStreamChunk>(trailing) {
                        parsed_any_event = true;
                        if response_id.is_none() {
                            response_id = event.id.clone();
                        }
                        if response_model.is_none() {
                            response_model = event.model.clone();
                        }
                        if response_object.is_none() {
                            response_object = event.object.clone();
                        }
                        if let Some(u) = event.usage {
                            usage.prompt_tokens = u.prompt_tokens;
                            usage.completion_tokens = u.completion_tokens;
                            usage.total_tokens = u.total_tokens;
                            usage.cache_read_tokens = u
                                .prompt_tokens_details
                                .as_ref()
                                .and_then(|d| d.cached_tokens);
                        }
                        if let Some(choice) = event.choices.into_iter().next() {
                            if let Some(reason) = choice.finish_reason {
                                finish_reason = Some(reason);
                            }
                            // If text_content already has content (from delta processing),
                            // skip message.content to avoid sending duplicate full content
                            let skip_content = !text_content.is_empty();
                            if let Some(message) = choice.message {
                                if let Some(reasoning) = message.reasoning_content {
                                    reasoning_content_accum.push_str(&reasoning);
                                    if first_token_ms.is_none() {
                                        first_token_ms =
                                            Some(request_started_at.elapsed().as_millis() as u64);
                                    }
                                    if let Some(delta) =
                                        Self::merge_stream_text(&mut text_content, &reasoning)
                                    {
                                        let _ = tx.send(StreamEvent::ReasoningDelta(delta)).await;
                                    }
                                }
                                if !skip_content {
                                    if let Some(content) =
                                        message.content.filter(|value| !value.is_empty())
                                    {
                                        if first_token_ms.is_none() {
                                            first_token_ms = Some(
                                                request_started_at.elapsed().as_millis() as u64,
                                            );
                                        }
                                        if let Some(delta) =
                                            Self::merge_stream_text(&mut text_content, &content)
                                        {
                                            let _ = tx.send(StreamEvent::TextDelta(delta)).await;
                                        }
                                    }
                                }
                                if let Some(tcs) = message.tool_calls {
                                    for (index, tc) in tcs.into_iter().enumerate() {
                                        tool_calls.insert(
                                            index,
                                            (tc.id, tc.function.name, tc.function.arguments),
                                        );
                                    }
                                }
                            } else if let Some(delta) = choice.delta {
                                if let Some(ref rc) = delta.reasoning_content {
                                    reasoning_content_accum.push_str(rc);
                                    if first_token_ms.is_none() {
                                        first_token_ms =
                                            Some(request_started_at.elapsed().as_millis() as u64);
                                    }
                                    if let Some(delta) =
                                        Self::merge_stream_text(&mut text_content, rc)
                                    {
                                        let _ = tx.send(StreamEvent::ReasoningDelta(delta)).await;
                                    }
                                }
                                if let Some(content) = delta.content {
                                    if first_token_ms.is_none() {
                                        first_token_ms =
                                            Some(request_started_at.elapsed().as_millis() as u64);
                                    }
                                    if let Some(delta) =
                                        Self::merge_stream_text(&mut text_content, &content)
                                    {
                                        let _ = tx.send(StreamEvent::TextDelta(delta)).await;
                                    }
                                }
                            }
                        }
                    } else if let Ok(response) = serde_json::from_str::<OpenAiResponse>(trailing) {
                        parsed_any_event = true;
                        response_id = response.id.clone();
                        response_model = response.model.clone();
                        response_object = response.object.clone();
                        usage.prompt_tokens = response.usage.prompt_tokens;
                        usage.completion_tokens = response.usage.completion_tokens;
                        usage.total_tokens = response.usage.total_tokens;
                        // MiniMax: fall back to total_characters when total_tokens is 0.
                        if usage.total_tokens == 0 {
                            usage.total_tokens = response.usage.total_characters.unwrap_or(0);
                        }
                        usage.cache_read_tokens = response
                            .usage
                            .prompt_tokens_details
                            .as_ref()
                            .and_then(|d| d.cached_tokens);

                        if let Some(choice) = response.choices.into_iter().next() {
                            finish_reason = choice.finish_reason;
                            if let Some(text) =
                                choice.message.content.filter(|text| !text.is_empty())
                            {
                                if first_token_ms.is_none() {
                                    first_token_ms =
                                        Some(request_started_at.elapsed().as_millis() as u64);
                                }
                                let _ = Self::merge_stream_text(&mut text_content, &text);
                            }
                            if let Some(reasoning) = choice.message.reasoning_content {
                                reasoning_content_accum.push_str(&reasoning);
                            }
                            if let Some(final_tool_calls) = choice.message.tool_calls {
                                for tc in final_tool_calls {
                                    tool_calls.insert(
                                        tool_calls.len(),
                                        (tc.id, tc.function.name, tc.function.arguments),
                                    );
                                }
                            }
                        }
                    }
                }

                if parsed_any_event
                    || !text_content.is_empty()
                    || !tool_calls.is_empty()
                    || !content_blocks.is_empty()
                {
                    tracing::warn!(
                        provider = %provider_name,
                        model = %request_model,
                        "OpenAI-compatible stream ended without [DONE]; finalizing buffered response"
                    );
                    if !text_content.is_empty() {
                        content_blocks.push(ContentBlock::Text {
                            text: text_content.clone(),
                        });
                    }
                    for (_, (id, name, args)) in tool_calls.iter() {
                        content_blocks.push(ContentBlock::ToolUse {
                            id: id.clone(),
                            name: name.clone(),
                            input: Self::parse_tool_arguments(name, args),
                        });
                    }
                    tool_calls.clear();
                    crate::telemetry::record_llm_usage(
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        usage.total_tokens,
                        finish_reason.as_deref(),
                    );
                    let response = LlmResponse {
                        message: Message {
                            role: "assistant".to_string(),
                            content: std::mem::take(&mut content_blocks),
                            reasoning_content: if reasoning_content_accum.is_empty() {
                                None
                            } else {
                                Some(std::mem::take(&mut reasoning_content_accum))
                            },
                        },
                        usage: usage.clone(),
                        stop_reason: std::mem::take(&mut finish_reason),
                        meta: Some(LlmResponseMeta {
                            provider: Some(provider_name.clone()),
                            request_model: Some(request_model.clone()),
                            request_url: Some(request_url.clone()),
                            response_id: response_id.clone(),
                            response_model: response_model.clone(),
                            response_object: response_object.clone(),
                            first_token_ms,
                            duration_ms: Some(request_started_at.elapsed().as_millis() as u64),
                        }),
                    };
                    let _ = tx.send(StreamEvent::Done(response)).await;
                } else {
                    tracing::warn!(
                        provider = %provider_name,
                        model = %request_model,
                        trailing = %trailing.chars().take(400).collect::<String>(),
                        "OpenAI-compatible stream ended without any parseable events"
                    );
                }
            });

            Ok(rx)
        }
    }
}

// OpenAI API response types (private)
#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiResponse {
    #[serde(default)]
    pub(crate) id: Option<String>,
    #[serde(default)]
    pub(crate) object: Option<String>,
    #[serde(default)]
    pub(crate) model: Option<String>,
    pub(crate) choices: Vec<OpenAiChoice>,
    pub(crate) usage: OpenAiUsage,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiChoice {
    pub(crate) message: OpenAiMessage,
    pub(crate) finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiMessage {
    pub(crate) reasoning_content: Option<String>,
    pub(crate) content: Option<String>,
    pub(crate) tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiToolCall {
    pub(crate) id: String,
    pub(crate) function: OpenAiFunction,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiFunction {
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiUsage {
    #[serde(default)]
    pub(crate) prompt_tokens: usize,
    #[serde(default)]
    pub(crate) completion_tokens: usize,
    #[serde(default)]
    pub(crate) total_tokens: usize,
    /// MiniMax uses `total_characters` instead of token counts.
    #[serde(default)]
    pub(crate) total_characters: Option<usize>,
    /// OpenAI returns cached token count in `prompt_tokens_details.cached_tokens`
    #[serde(default)]
    pub(crate) prompt_tokens_details: Option<OpenAiPromptTokensDetails>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiPromptTokensDetails {
    #[serde(default)]
    pub(crate) cached_tokens: Option<usize>,
}

// OpenAI streaming types
#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiStreamChunk {
    #[serde(default)]
    pub(crate) id: Option<String>,
    #[serde(default)]
    pub(crate) object: Option<String>,
    #[serde(default)]
    pub(crate) model: Option<String>,
    pub(crate) choices: Vec<OpenAiStreamChoice>,
    pub(crate) usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiStreamChoice {
    pub(crate) message: Option<OpenAiMessage>,
    pub(crate) delta: Option<OpenAiDelta>,
    pub(crate) finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiDelta {
    pub(crate) reasoning_content: Option<String>,
    pub(crate) content: Option<String>,
    pub(crate) tool_calls: Option<Vec<OpenAiToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiToolCallDelta {
    pub(crate) index: usize,
    pub(crate) id: Option<String>,
    pub(crate) function: Option<OpenAiFunctionDelta>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiFunctionDelta {
    pub(crate) name: Option<String>,
    pub(crate) arguments: Option<String>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{Message, ToolDefinition};

    fn make_client() -> OpenAiClient {
        OpenAiClient::new("test-key".to_string(), "gpt-test".to_string())
    }

    #[test]
    fn test_apply_directive_forced_function_tool_choice() {
        let mut req = serde_json::json!({ "model": "m" });
        OpenAiClient::apply_directive(
            &mut req,
            &structured::StructuredDirective {
                force_tool: Some("emit_person".to_string()),
                response_format: None,
            },
        );
        assert_eq!(req["tool_choice"]["type"], "function");
        assert_eq!(req["tool_choice"]["function"]["name"], "emit_person");
        assert!(req.get("response_format").is_none());
    }

    #[test]
    fn test_apply_directive_json_schema_strict() {
        let mut req = serde_json::json!({});
        OpenAiClient::apply_directive(
            &mut req,
            &structured::StructuredDirective {
                force_tool: None,
                response_format: Some(structured::ResponseFormat::JsonSchema {
                    name: "person".to_string(),
                    schema: serde_json::json!({ "type": "object" }),
                }),
            },
        );
        assert_eq!(req["response_format"]["type"], "json_schema");
        assert_eq!(req["response_format"]["json_schema"]["name"], "person");
        assert_eq!(req["response_format"]["json_schema"]["strict"], true);
        assert!(req.get("tool_choice").is_none());
    }

    #[test]
    fn test_apply_directive_json_object() {
        let mut req = serde_json::json!({});
        OpenAiClient::apply_directive(
            &mut req,
            &structured::StructuredDirective {
                force_tool: None,
                response_format: Some(structured::ResponseFormat::JsonObject),
            },
        );
        assert_eq!(req["response_format"]["type"], "json_object");
    }

    #[test]
    fn test_build_chat_request_applies_directive_and_system() {
        let req = make_client().build_chat_request(
            &[Message::user("hi")],
            Some("sys"),
            &[ToolDefinition {
                name: "emit_x".to_string(),
                description: "emit".to_string(),
                parameters: serde_json::json!({ "type": "object" }),
            }],
            Some(&structured::StructuredDirective {
                force_tool: Some("emit_x".to_string()),
                response_format: None,
            }),
        );
        assert_eq!(req["messages"][0]["role"], "system");
        assert_eq!(req["tool_choice"]["function"]["name"], "emit_x");
        assert_eq!(req["tools"][0]["function"]["name"], "emit_x");
    }

    #[test]
    fn test_build_chat_request_without_directive_is_plain() {
        let req = make_client().build_chat_request(&[Message::user("hi")], None, &[], None);
        assert!(req.get("tool_choice").is_none());
        assert!(req.get("response_format").is_none());
    }

    #[test]
    fn test_native_structured_support_is_json_schema() {
        assert_eq!(
            make_client().native_structured_support(),
            structured::NativeStructuredSupport::JsonSchema
        );
    }
}
