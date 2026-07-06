#[cfg(test)]
mod tests {
    use crate::llm::anthropic::*;
    use crate::llm::http::normalize_base_url;
    use crate::llm::openai::*;
    use crate::llm::*;
    use tokio_util::sync::CancellationToken;

    #[test]
    fn test_secret_string_redacts_debug() {
        let secret = SecretString::new("my-api-key-12345");
        let debug = format!("{:?}", secret);
        assert_eq!(debug, "[REDACTED]");
        assert!(!debug.contains("my-api-key"));
    }

    #[test]
    fn test_secret_string_redacts_display() {
        let secret = SecretString::new("sk-secret-value");
        let display = format!("{}", secret);
        assert_eq!(display, "[REDACTED]");
        assert!(!display.contains("sk-secret"));
    }

    #[test]
    fn test_secret_string_expose() {
        let secret = SecretString::new("the-real-key");
        assert_eq!(secret.expose(), "the-real-key");
    }

    #[test]
    fn test_secret_string_from_impls() {
        let _s1: SecretString = "literal".into();
        let _s2: SecretString = String::from("owned").into();
        let owned = String::from("ref");
        let _s3: SecretString = (&owned).into();
    }

    #[test]
    fn test_llm_config_debug_redacts_api_key() {
        let config = LlmConfig::new("openai", "gpt-4", "sk-super-secret");
        let debug = format!("{:?}", config);
        assert!(debug.contains("REDACTED"));
        assert!(!debug.contains("sk-super-secret"));
    }

    #[test]
    fn test_message_creation() {
        let msg = Message::user("Hello");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.text(), "Hello");
    }

    #[test]
    fn test_normalize_base_url() {
        assert_eq!(
            normalize_base_url("https://api.example.com"),
            "https://api.example.com"
        );
        assert_eq!(
            normalize_base_url("https://api.example.com/"),
            "https://api.example.com"
        );
        assert_eq!(
            normalize_base_url("https://api.example.com/v1"),
            "https://api.example.com"
        );
        assert_eq!(
            normalize_base_url("https://api.example.com/v1/"),
            "https://api.example.com"
        );
    }

    // ========================================================================
    // Integration Tests (require real LLM API)
    // ========================================================================
    // These tests are ignored by default. Run with:
    //   cargo test -p a3s-code --lib test_real_llm -- --ignored
    //
    // Requires config.json in workspace root with:
    // {
    //   "llm": {
    //     "api_base": "http://...",
    //     "api_key": "sk-...",
    //     "model": "..."
    //   }
    // }

    fn load_test_config() -> Option<(String, String, String)> {
        let config_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()?
            .parent()?
            .join("config.json");

        let content = std::fs::read_to_string(&config_path).ok()?;
        let config: serde_json::Value = serde_json::from_str(&content).ok()?;

        // Try new format first (providers array)
        if let Some(providers) = config.get("providers").and_then(|p| p.as_array()) {
            // Look for openai provider with kimi-k2.5 model
            for provider in providers {
                if let Some(models) = provider.get("models").and_then(|m| m.as_array()) {
                    for model in models {
                        if model.get("id")?.as_str()? == "kimi-k2.5" {
                            let api_base = model.get("baseUrl")?.as_str()?.to_string();
                            let api_key = model.get("apiKey")?.as_str()?.to_string();
                            let model_id = model.get("id")?.as_str()?.to_string();
                            return Some((api_base, api_key, model_id));
                        }
                    }
                }
            }

            // Fallback: use first provider's first model
            if let Some(provider) = providers.first() {
                let api_base = provider.get("baseUrl")?.as_str()?.to_string();
                let api_key = provider.get("apiKey")?.as_str()?.to_string();
                let models = provider.get("models")?.as_array()?;
                let model_id = models.first()?.get("id")?.as_str()?.to_string();
                return Some((api_base, api_key, model_id));
            }
        }

        // Try old format (llm object)
        if let Some(llm) = config.get("llm") {
            let api_base = llm.get("api_base")?.as_str()?.to_string();
            let api_key = llm.get("api_key")?.as_str()?.to_string();
            let model = llm.get("model")?.as_str()?.to_string();
            return Some((api_base, api_key, model));
        }

        None
    }

    #[tokio::test]
    #[ignore] // Run with: cargo test -p a3s-code --lib test_real_llm_openai_complete -- --ignored
    async fn test_real_llm_openai_complete() {
        let Some((api_base, api_key, model)) = load_test_config() else {
            eprintln!("Skipping test: config.json not found or invalid");
            return;
        };

        let client = OpenAiClient::new(api_key, model).with_base_url(api_base);

        let messages = vec![Message::user("Say 'Hello, World!' and nothing else.")];

        let response = client.complete(&messages, None, &[]).await;
        assert!(response.is_ok(), "LLM call failed: {:?}", response.err());

        let response = response.unwrap();
        let text = response.text().to_lowercase();
        assert!(
            text.contains("hello") && text.contains("world"),
            "Unexpected response: {}",
            text
        );

        println!("Response: {}", response.text());
        println!("Usage: {:?}", response.usage);
    }

    #[tokio::test]
    #[ignore] // Run with: cargo test -p a3s-code --lib test_real_llm_openai_streaming -- --ignored
    async fn test_real_llm_openai_streaming() {
        let Some((api_base, api_key, model)) = load_test_config() else {
            eprintln!("Skipping test: config.json not found or invalid");
            return;
        };

        let client = OpenAiClient::new(api_key, model).with_base_url(api_base);

        let messages = vec![Message::user("Count from 1 to 5, one number per line.")];

        let result = client
            .complete_streaming(&messages, None, &[], CancellationToken::new())
            .await;
        assert!(result.is_ok(), "Streaming call failed: {:?}", result.err());

        let mut rx = result.unwrap();
        let mut full_text = String::new();
        let mut event_count = 0;

        while let Some(event) = rx.recv().await {
            event_count += 1;
            match event {
                StreamEvent::TextDelta(delta) => {
                    full_text.push_str(&delta);
                    print!("{}", delta);
                }
                StreamEvent::Done(response) => {
                    println!("\n\nStreaming complete. Usage: {:?}", response.usage);
                }
                _ => {}
            }
        }

        assert!(event_count > 0, "No events received");
        assert!(full_text.contains("1"), "Response should contain '1'");
        println!("\nFull response: {}", full_text);
    }

    // ========================================================================
    // Unit Tests for LLM types and telemetry integration
    // ========================================================================

    #[test]
    fn test_message_tool_result() {
        let msg = Message::tool_result("tool-123", "result data", false);
        assert_eq!(msg.role, "user");
        match &msg.content[0] {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                assert_eq!(tool_use_id, "tool-123");
                assert_eq!(content, "result data");
                assert_eq!(*is_error, Some(false));
            }
            _ => panic!("Expected ToolResult content block"),
        }
    }

    #[test]
    fn test_message_tool_result_error() {
        let msg = Message::tool_result("tool-456", "error msg", true);
        match &msg.content[0] {
            ContentBlock::ToolResult { is_error, .. } => {
                assert_eq!(*is_error, Some(true));
            }
            _ => panic!("Expected ToolResult content block"),
        }
    }

    #[test]
    fn test_message_text_multiple_blocks() {
        let msg = Message {
            role: "assistant".to_string(),
            content: vec![
                ContentBlock::Text {
                    text: "Hello ".to_string(),
                },
                ContentBlock::Text {
                    text: "World".to_string(),
                },
            ],
            reasoning_content: None,
        };
        assert_eq!(msg.text(), "Hello World");
    }

    #[test]
    fn test_message_text_with_tool_use() {
        let msg = Message {
            role: "assistant".to_string(),
            content: vec![
                ContentBlock::Text {
                    text: "Let me run that.".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "t1".to_string(),
                    name: "bash".to_string(),
                    input: serde_json::json!({"command": "ls"}),
                },
            ],
            reasoning_content: None,
        };
        assert_eq!(msg.text(), "Let me run that.");
    }

    #[test]
    fn test_message_tool_calls() {
        let msg = Message {
            role: "assistant".to_string(),
            content: vec![
                ContentBlock::Text {
                    text: "Running tools".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "t1".to_string(),
                    name: "bash".to_string(),
                    input: serde_json::json!({"command": "ls"}),
                },
                ContentBlock::ToolUse {
                    id: "t2".to_string(),
                    name: "read".to_string(),
                    input: serde_json::json!({"file": "test.rs"}),
                },
            ],
            reasoning_content: None,
        };
        let calls = msg.tool_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[1].name, "read");
        assert_eq!(calls[0].id, "t1");
    }

    #[test]
    fn test_message_no_tool_calls() {
        let msg = Message::user("Hello");
        assert!(msg.tool_calls().is_empty());
    }

    #[test]
    fn test_token_usage_default() {
        let usage = TokenUsage::default();
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
        assert!(usage.cache_read_tokens.is_none());
        assert!(usage.cache_write_tokens.is_none());
    }

    #[test]
    fn test_llm_response_text() {
        let response = LlmResponse {
            message: Message {
                role: "assistant".to_string(),
                content: vec![ContentBlock::Text {
                    text: "Hello!".to_string(),
                }],
                reasoning_content: None,
            },
            usage: TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                cache_read_tokens: None,
                cache_write_tokens: None,
            },
            stop_reason: Some("end_turn".to_string()),
            token_logprobs: Vec::new(),
            meta: None,
        };
        assert_eq!(response.text(), "Hello!");
        assert!(response.tool_calls().is_empty());
        assert_eq!(response.usage.total_tokens, 15);
        assert_eq!(response.stop_reason.as_deref(), Some("end_turn"));
    }

    #[test]
    fn test_llm_response_with_tool_calls() {
        let response = LlmResponse {
            message: Message {
                role: "assistant".to_string(),
                content: vec![ContentBlock::ToolUse {
                    id: "call-1".to_string(),
                    name: "grep".to_string(),
                    input: serde_json::json!({"pattern": "fn main"}),
                }],
                reasoning_content: None,
            },
            usage: TokenUsage::default(),
            stop_reason: Some("tool_use".to_string()),
            token_logprobs: Vec::new(),
            meta: None,
        };
        let calls = response.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "grep");
        assert_eq!(calls[0].args["pattern"], "fn main");
    }

    #[test]
    fn test_tool_definition_creation() {
        let def = ToolDefinition {
            name: "bash".to_string(),
            description: "Execute shell commands".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            }),
        };
        assert_eq!(def.name, "bash");
        assert!(def.parameters["properties"]["command"].is_object());
    }

    #[test]
    fn test_content_block_serialization() {
        let text = ContentBlock::Text {
            text: "hello".to_string(),
        };
        let json = serde_json::to_string(&text).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"text\":\"hello\""));

        let tool = ContentBlock::ToolUse {
            id: "t1".to_string(),
            name: "bash".to_string(),
            input: serde_json::json!({}),
        };
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("\"type\":\"tool_use\""));
        assert!(json.contains("\"name\":\"bash\""));
    }

    #[test]
    fn test_anthropic_client_builder() {
        let client = AnthropicClient::new("sk-test".to_string(), "claude-sonnet".to_string());
        assert_eq!(client.model, "claude-sonnet");

        let client = client
            .with_base_url("https://custom.api.com".to_string())
            .with_max_tokens(2048);
        assert_eq!(client.base_url, "https://custom.api.com");
        assert_eq!(client.max_tokens, 2048);
    }

    #[test]
    fn test_openai_client_builder() {
        let client = OpenAiClient::new("sk-test".to_string(), "gpt-4o".to_string());
        assert_eq!(client.model, "gpt-4o");

        let client = client.with_base_url("https://custom.openai.com".to_string());
        assert_eq!(client.base_url, "https://custom.openai.com");
    }

    #[test]
    fn test_normalize_base_url_edge_cases() {
        assert_eq!(
            normalize_base_url("http://localhost:8080"),
            "http://localhost:8080"
        );
        assert_eq!(
            normalize_base_url("http://localhost:8080/"),
            "http://localhost:8080"
        );
        assert_eq!(
            normalize_base_url("http://localhost:8080/v1/"),
            "http://localhost:8080"
        );
    }

    #[test]
    fn test_llm_config_creation() {
        let config = LlmConfig::new("anthropic", "claude-sonnet", "sk-key");
        assert_eq!(config.provider, "anthropic");
        assert_eq!(config.model, "claude-sonnet");
        assert_eq!(config.api_key.expose(), "sk-key");
    }
}

#[cfg(test)]
mod extra_llm_tests {
    use crate::llm::anthropic::*;
    use crate::llm::http::normalize_base_url;
    use crate::llm::openai::*;
    use crate::llm::*;

    #[test]
    fn test_message_assistant_text() {
        let msg = Message {
            role: "assistant".into(),
            content: vec![ContentBlock::Text {
                text: "Hello".into(),
            }],
            reasoning_content: None,
        };
        assert_eq!(msg.text(), "Hello");
    }

    #[test]
    fn test_message_text_empty() {
        let msg = Message {
            role: "assistant".into(),
            content: vec![],
            reasoning_content: None,
        };
        assert_eq!(msg.text(), "");
    }

    #[test]
    fn test_message_text_mixed() {
        let msg = Message {
            role: "assistant".into(),
            content: vec![
                ContentBlock::Text { text: "A ".into() },
                ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "bash".into(),
                    input: serde_json::json!({}),
                },
                ContentBlock::Text { text: "B".into() },
            ],
            reasoning_content: None,
        };
        assert_eq!(msg.text(), "A B");
    }

    #[test]
    fn test_message_tool_calls_extraction() {
        let msg = Message {
            role: "assistant".into(),
            content: vec![
                ContentBlock::Text {
                    text: "help".into(),
                },
                ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "bash".into(),
                    input: serde_json::json!({"cmd":"ls"}),
                },
                ContentBlock::ToolUse {
                    id: "t2".into(),
                    name: "read".into(),
                    input: serde_json::json!({"p":"/"}),
                },
            ],
            reasoning_content: None,
        };
        let calls = msg.tool_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[1].name, "read");
    }

    #[test]
    fn test_message_tool_calls_empty() {
        let msg = Message {
            role: "assistant".into(),
            content: vec![ContentBlock::Text {
                text: "no tools".into(),
            }],
            reasoning_content: None,
        };
        assert!(msg.tool_calls().is_empty());
    }

    #[test]
    fn test_message_tool_result_success() {
        let msg = Message::tool_result("t1", "output", false);
        assert_eq!(msg.role, "user");
        match &msg.content[0] {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                assert_eq!(tool_use_id, "t1");
                assert_eq!(content, "output");
                assert_eq!(*is_error, Some(false));
            }
            _ => panic!("Expected ToolResult"),
        }
    }

    #[test]
    fn test_message_tool_result_error() {
        let msg = Message::tool_result("t1", "err", true);
        match &msg.content[0] {
            ContentBlock::ToolResult { is_error, .. } => assert_eq!(*is_error, Some(true)),
            _ => panic!("Expected ToolResult"),
        }
    }

    #[test]
    fn test_llm_response_text() {
        let r = LlmResponse {
            message: Message {
                role: "assistant".into(),
                content: vec![ContentBlock::Text {
                    text: "resp".into(),
                }],
                reasoning_content: None,
            },
            usage: TokenUsage::default(),
            stop_reason: None,
            token_logprobs: Vec::new(),
            meta: None,
        };
        assert_eq!(r.text(), "resp");
    }

    #[test]
    fn test_llm_response_tool_calls() {
        let r = LlmResponse {
            message: Message {
                role: "assistant".into(),
                content: vec![ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "bash".into(),
                    input: serde_json::json!({}),
                }],
                reasoning_content: None,
            },
            usage: TokenUsage::default(),
            stop_reason: None,
            token_logprobs: Vec::new(),
            meta: None,
        };
        assert_eq!(r.tool_calls().len(), 1);
    }

    #[test]
    fn test_token_usage_default() {
        let u = TokenUsage::default();
        assert_eq!(u.prompt_tokens, 0);
        assert_eq!(u.completion_tokens, 0);
        assert_eq!(u.total_tokens, 0);
        assert_eq!(u.cache_read_tokens, None);
        assert_eq!(u.cache_write_tokens, None);
    }

    #[test]
    fn test_llm_config_new() {
        let c = LlmConfig::new("anthropic", "claude-3", "sk-test");
        assert_eq!(c.provider, "anthropic");
        assert_eq!(c.model, "claude-3");
        assert!(c.base_url.is_none());
    }

    #[test]
    fn test_llm_config_with_base_url() {
        let c = LlmConfig::new("openai", "gpt-4", "k").with_base_url("https://x.com");
        assert_eq!(c.base_url, Some("https://x.com".into()));
    }

    #[test]
    fn test_anthropic_client_new() {
        let c = AnthropicClient::new("key".into(), "claude-3".into());
        assert_eq!(c.model, "claude-3");
        assert_eq!(c.max_tokens, 8192);
    }

    #[test]
    fn test_anthropic_client_max_tokens() {
        let c = AnthropicClient::new("k".into(), "m".into()).with_max_tokens(4096);
        assert_eq!(c.max_tokens, 4096);
    }

    #[test]
    fn test_openai_client_new() {
        let c = OpenAiClient::new("key".into(), "gpt-4".into());
        assert_eq!(c.model, "gpt-4");
        assert_eq!(c.base_url, "https://api.openai.com");
    }

    #[test]
    fn test_normalize_strips_v1() {
        assert_eq!(normalize_base_url("https://api.com/v1"), "https://api.com");
    }

    #[test]
    fn test_normalize_strips_trailing_slash() {
        assert_eq!(normalize_base_url("https://api.com/"), "https://api.com");
    }

    #[test]
    fn test_normalize_no_change() {
        assert_eq!(normalize_base_url("https://api.com"), "https://api.com");
    }

    #[test]
    fn test_create_client_anthropic() {
        let _c = create_client_with_config(LlmConfig::new("anthropic", "claude-3", "k"));
    }

    #[test]
    fn test_create_client_openai() {
        let _c = create_client_with_config(LlmConfig::new("openai", "gpt-4", "k"));
    }

    #[test]
    fn test_create_client_unknown() {
        let _c = create_client_with_config(LlmConfig::new("unknown", "m", "k"));
    }

    #[test]
    fn test_anthropic_build_request_basic() {
        let c = AnthropicClient::new("k".into(), "claude-3".into());
        let b = c.build_request(&[Message::user("Hi")], None, &[]);
        assert_eq!(b["model"], "claude-3");
        assert!(b.get("system").is_none());
        assert!(b.get("tools").is_none());
    }

    #[test]
    fn test_anthropic_build_request_system() {
        let c = AnthropicClient::new("k".into(), "claude-3".into());
        let b = c.build_request(&[Message::user("Hi")], Some("Be helpful"), &[]);
        // System is now an array with cache_control for prompt caching
        let system = b["system"].as_array().unwrap();
        assert_eq!(system.len(), 1);
        assert_eq!(system[0]["text"], "Be helpful");
        assert_eq!(system[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_anthropic_build_request_tools() {
        let c = AnthropicClient::new("k".into(), "claude-3".into());
        let tools = vec![ToolDefinition {
            name: "bash".into(),
            description: "Run".into(),
            parameters: serde_json::json!({"type":"object"}),
        }];
        let b = c.build_request(&[Message::user("Hi")], None, &tools);
        assert_eq!(b["tools"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_openai_convert_user_msg() {
        let c = OpenAiClient::new("k".into(), "gpt-4".into());
        let m = c.convert_messages(&[Message::user("Hello")]);
        assert_eq!(m[0]["role"], "user");
        assert_eq!(m[0]["content"], "Hello");
    }

    #[test]
    fn test_openai_convert_tool_result() {
        let c = OpenAiClient::new("k".into(), "gpt-4".into());
        let m = c.convert_messages(&[Message::tool_result("c1", "out", false)]);
        assert_eq!(m[0]["role"], "tool");
        assert_eq!(m[0]["tool_call_id"], "c1");
    }

    #[test]
    fn test_openai_convert_tools_empty() {
        let c = OpenAiClient::new("k".into(), "gpt-4".into());
        assert!(c.convert_tools(&[]).is_empty());
    }

    #[test]
    fn test_openai_convert_tools_single() {
        let c = OpenAiClient::new("k".into(), "gpt-4".into());
        let t = c.convert_tools(&[ToolDefinition {
            name: "read".into(),
            description: "Read".into(),
            parameters: serde_json::json!({"type":"object"}),
        }]);
        assert_eq!(t.len(), 1);
        assert_eq!(t[0]["type"], "function");
    }

    #[test]
    fn test_anthropic_response_with_cache() {
        let j = r#"{"content":[{"type":"text","text":"Hi"}],"stop_reason":"end_turn","usage":{"input_tokens":100,"output_tokens":50,"cache_read_input_tokens":80,"cache_creation_input_tokens":20}}"#;
        let r: AnthropicResponse = serde_json::from_str(j).unwrap();
        assert_eq!(r.usage.cache_read_input_tokens, Some(80));
        assert_eq!(r.usage.cache_creation_input_tokens, Some(20));
    }

    #[test]
    fn test_anthropic_response_no_cache() {
        let j = r#"{"content":[{"type":"text","text":"Hi"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}"#;
        let r: AnthropicResponse = serde_json::from_str(j).unwrap();
        assert!(r.usage.cache_read_input_tokens.is_none());
    }

    #[test]
    fn test_anthropic_response_tool_use() {
        let j = r#"{"content":[{"type":"text","text":"ok"},{"type":"tool_use","id":"t1","name":"bash","input":{"cmd":"ls"}}],"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":20}}"#;
        let r: AnthropicResponse = serde_json::from_str(j).unwrap();
        assert_eq!(r.content.len(), 2);
        assert_eq!(r.stop_reason, "tool_use");
    }

    #[test]
    fn test_openai_response_null_content() {
        let j = r#"{"id":"c1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#;
        let r: OpenAiResponse = serde_json::from_str(j).unwrap();
        assert!(r.choices[0].message.content.is_none());
    }

    #[test]
    fn test_openai_response_tool_calls() {
        let j = r#"{"id":"c1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"bash","arguments":"{\"cmd\":\"ls\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#;
        let r: OpenAiResponse = serde_json::from_str(j).unwrap();
        assert_eq!(r.choices[0].message.tool_calls.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_content_block_text_serialize() {
        let b = ContentBlock::Text { text: "hi".into() };
        let j = serde_json::to_value(&b).unwrap();
        assert_eq!(j["type"], "text");
    }

    #[test]
    fn test_content_block_tool_use_serialize() {
        let b = ContentBlock::ToolUse {
            id: "t1".into(),
            name: "bash".into(),
            input: serde_json::json!({}),
        };
        let j = serde_json::to_value(&b).unwrap();
        assert_eq!(j["type"], "tool_use");
    }

    #[test]
    fn test_content_block_tool_result_serialize() {
        let b = ContentBlock::ToolResult {
            tool_use_id: "t1".into(),
            content: "out".into(),
            is_error: Some(false),
        };
        let j = serde_json::to_value(&b).unwrap();
        assert_eq!(j["type"], "tool_result");
    }

    #[test]
    fn test_tool_definition() {
        let t = ToolDefinition {
            name: "grep".into(),
            description: "Search".into(),
            parameters: serde_json::json!({"type":"object"}),
        };
        assert_eq!(t.name, "grep");
    }
}

#[cfg(test)]
mod extra_llm_tests2 {
    use crate::llm::anthropic::*;
    use crate::llm::http::normalize_base_url;
    use crate::llm::http::{HttpClient, HttpResponse, StreamingHttpResponse};
    use crate::llm::openai::*;
    use crate::llm::*;
    use crate::retry::RetryConfig;
    use anyhow::Result;
    use async_trait::async_trait;
    use bytes::Bytes;
    use futures::stream;
    use std::sync::Arc;

    struct MockStreamingHttpClient {
        chunks: Vec<String>,
    }

    #[async_trait]
    impl HttpClient for MockStreamingHttpClient {
        async fn post(
            &self,
            _url: &str,
            _headers: Vec<(&str, &str)>,
            _body: &serde_json::Value,
            _cancel_token: CancellationToken,
        ) -> Result<HttpResponse> {
            anyhow::bail!("post() not expected in MockStreamingHttpClient tests")
        }

        async fn post_streaming(
            &self,
            _url: &str,
            _headers: Vec<(&str, &str)>,
            _body: &serde_json::Value,
            _cancel_token: CancellationToken,
        ) -> Result<StreamingHttpResponse> {
            let items = self
                .chunks
                .iter()
                .cloned()
                .map(|chunk| Ok(Bytes::from(chunk)))
                .collect::<Vec<_>>();
            Ok(StreamingHttpResponse {
                status: 200,
                retry_after: None,
                byte_stream: Box::pin(stream::iter(items)),
                error_body: String::new(),
            })
        }
    }

    // ========================================================================
    // AnthropicClient build_request
    // ========================================================================

    #[test]
    fn test_anthropic_build_request_basic() {
        let client =
            AnthropicClient::new("key".to_string(), "claude-sonnet-4-20250514".to_string());
        let msgs = vec![Message::user("Hello")];
        let req = client.build_request(&msgs, None, &[]);

        assert_eq!(req["model"], "claude-sonnet-4-20250514");
        assert!(req["system"].is_null());
        assert!(req["tools"].is_null());
        assert!(req["messages"].is_array());
    }

    #[test]
    fn test_anthropic_build_request_with_system() {
        let client =
            AnthropicClient::new("key".to_string(), "claude-sonnet-4-20250514".to_string());
        let msgs = vec![Message::user("Hello")];
        let req = client.build_request(&msgs, Some("You are helpful"), &[]);

        // System is now an array with cache_control for prompt caching
        let system = req["system"].as_array().unwrap();
        assert_eq!(system[0]["text"], "You are helpful");
        assert_eq!(system[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_anthropic_build_request_with_tools() {
        let client =
            AnthropicClient::new("key".to_string(), "claude-sonnet-4-20250514".to_string());
        let msgs = vec![Message::user("Hello")];
        let tools = vec![ToolDefinition {
            name: "bash".to_string(),
            description: "Run a command".to_string(),
            parameters: serde_json::json!({"type": "object", "properties": {"command": {"type": "string"}}}),
        }];
        let req = client.build_request(&msgs, None, &tools);

        assert!(req["tools"].is_array());
        assert_eq!(req["tools"][0]["name"], "bash");
        assert_eq!(req["tools"][0]["description"], "Run a command");
        assert!(req["tools"][0]["input_schema"].is_object());
    }

    #[test]
    fn test_anthropic_build_request_max_tokens() {
        let client =
            AnthropicClient::new("key".to_string(), "model".to_string()).with_max_tokens(4096);
        let req = client.build_request(&[], None, &[]);
        assert_eq!(req["max_tokens"], 4096);
    }

    // ========================================================================
    // OpenAiClient convert_messages
    // ========================================================================

    #[test]
    fn test_openai_convert_messages_simple_text() {
        let client = OpenAiClient::new("key".to_string(), "gpt-4".to_string());
        let msgs = vec![Message::user("Hello")];
        let converted = client.convert_messages(&msgs);

        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["role"], "user");
        assert_eq!(converted[0]["content"], "Hello");
    }

    #[test]
    fn test_openai_convert_messages_tool_result() {
        let client = OpenAiClient::new("key".to_string(), "gpt-4".to_string());
        let msgs = vec![Message::tool_result("call-1", "output data", false)];
        let converted = client.convert_messages(&msgs);

        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["role"], "tool");
        assert_eq!(converted[0]["tool_call_id"], "call-1");
        assert_eq!(converted[0]["content"], "output data");
    }

    #[test]
    fn test_openai_convert_messages_assistant_with_tool_calls() {
        let client = OpenAiClient::new("key".to_string(), "gpt-4".to_string());
        let msgs = vec![Message {
            role: "assistant".to_string(),
            content: vec![
                ContentBlock::Text {
                    text: "Let me help.".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "call-1".to_string(),
                    name: "bash".to_string(),
                    input: serde_json::json!({"command": "ls"}),
                },
            ],
            reasoning_content: None,
        }];
        let converted = client.convert_messages(&msgs);

        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["role"], "assistant");
        assert!(converted[0]["tool_calls"].is_array());
        assert_eq!(converted[0]["tool_calls"][0]["function"]["name"], "bash");
    }

    #[test]
    fn test_openai_convert_messages_multi_block_text() {
        let client = OpenAiClient::new("key".to_string(), "gpt-4".to_string());
        let msgs = vec![Message {
            role: "user".to_string(),
            content: vec![
                ContentBlock::Text {
                    text: "Part 1".to_string(),
                },
                ContentBlock::Text {
                    text: "Part 2".to_string(),
                },
            ],
            reasoning_content: None,
        }];
        let converted = client.convert_messages(&msgs);

        assert_eq!(converted.len(), 1);
        // Multi-block should be an array
        assert!(converted[0]["content"].is_array());
    }

    // ========================================================================
    // OpenAiClient convert_tools
    // ========================================================================

    #[test]
    fn test_openai_convert_tools() {
        let client = OpenAiClient::new("key".to_string(), "gpt-4".to_string());
        let tools = vec![
            ToolDefinition {
                name: "bash".to_string(),
                description: "Run command".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            },
            ToolDefinition {
                name: "read".to_string(),
                description: "Read file".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            },
        ];
        let converted = client.convert_tools(&tools);

        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0]["type"], "function");
        assert_eq!(converted[0]["function"]["name"], "bash");
        assert_eq!(converted[1]["function"]["name"], "read");
    }

    #[test]
    fn test_openai_convert_tools_empty() {
        let client = OpenAiClient::new("key".to_string(), "gpt-4".to_string());
        let converted = client.convert_tools(&[]);
        assert!(converted.is_empty());
    }

    // ========================================================================
    // Anthropic Response Deserialization
    // ========================================================================

    #[test]
    fn test_anthropic_response_text_only() {
        let json = r#"{
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }"#;
        let resp: AnthropicResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.stop_reason, "end_turn");
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 5);
        assert!(resp.usage.cache_read_input_tokens.is_none());
    }

    #[test]
    fn test_anthropic_response_with_tool_use() {
        let json = r#"{
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "ls"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 15}
        }"#;
        let resp: AnthropicResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.content.len(), 2);
        assert_eq!(resp.stop_reason, "tool_use");
    }

    #[test]
    fn test_anthropic_response_with_cache_tokens() {
        let json = r#"{
            "content": [{"type": "text", "text": "Hi"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 80,
                "cache_creation_input_tokens": 20
            }
        }"#;
        let resp: AnthropicResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.usage.cache_read_input_tokens, Some(80));
        assert_eq!(resp.usage.cache_creation_input_tokens, Some(20));
    }

    // ========================================================================
    // OpenAI Response Deserialization
    // ========================================================================

    #[test]
    fn test_openai_response_text_only() {
        let json = r#"{
            "choices": [{
                "message": {"content": "Hello!", "tool_calls": null},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }"#;
        let resp: OpenAiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].message.content, Some("Hello!".to_string()));
        assert_eq!(resp.choices[0].finish_reason, Some("stop".to_string()));
        assert_eq!(resp.usage.total_tokens, 15);
    }

    #[test]
    fn test_openai_response_with_tool_calls() {
        let json = r#"{
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call-1",
                        "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        }"#;
        let resp: OpenAiResponse = serde_json::from_str(json).unwrap();
        assert!(resp.choices[0].message.content.is_none());
        let tool_calls = resp.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call-1");
        assert_eq!(tool_calls[0].function.name, "bash");
    }

    #[test]
    fn test_openai_response_null_content_and_tool_calls() {
        let json = r#"{
            "choices": [{
                "message": {"content": null, "tool_calls": null},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        }"#;
        let resp: OpenAiResponse = serde_json::from_str(json).unwrap();
        assert!(resp.choices[0].message.content.is_none());
        assert!(resp.choices[0].message.tool_calls.is_none());
    }

    // ========================================================================
    // Anthropic Streaming Event Deserialization
    // ========================================================================

    #[test]
    fn test_anthropic_stream_message_start() {
        let json = r#"{
            "type": "message_start",
            "message": {
                "usage": {"input_tokens": 100, "output_tokens": 0}
            }
        }"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, AnthropicStreamEvent::MessageStart { .. }));
    }

    #[test]
    fn test_anthropic_stream_content_block_start_text() {
        let json = r#"{
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        }"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(
            event,
            AnthropicStreamEvent::ContentBlockStart { .. }
        ));
    }

    #[test]
    fn test_anthropic_stream_content_block_start_tool() {
        let json = r#"{
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "t1", "name": "bash", "input": {}}
        }"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(
            event,
            AnthropicStreamEvent::ContentBlockStart { .. }
        ));
    }

    #[test]
    fn test_anthropic_stream_text_delta() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"}
        }"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(
            event,
            AnthropicStreamEvent::ContentBlockDelta { .. }
        ));
    }

    #[test]
    fn test_anthropic_stream_input_json_delta() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": "{\"cmd\":"}
        }"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(
            event,
            AnthropicStreamEvent::ContentBlockDelta { .. }
        ));
    }

    #[tokio::test]
    async fn test_anthropic_stream_tool_input_from_content_block_start_is_preserved() {
        let sse = vec![
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg-1\",\"type\":\"message\",\"model\":\"claude-sonnet-4-20250514\",\"usage\":{\"input_tokens\":10}}}\n\n".to_string(),
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tool-1\",\"name\":\"Skill\",\"input\":{\"skill_name\":\"hello-skill\",\"prompt\":\"run\"}}}\n\n".to_string(),
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n".to_string(),
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":5}}\n\n".to_string(),
            "data: {\"type\":\"message_stop\"}\n\n".to_string(),
        ];

        let client = AnthropicClient::new("key".to_string(), "model".to_string())
            .with_http_client(Arc::new(MockStreamingHttpClient { chunks: sse }));

        let mut rx = client
            .complete_streaming(
                &[Message::user("run skill")],
                None,
                &[],
                CancellationToken::new(),
            )
            .await
            .unwrap();

        let mut saw_tool_start = false;
        let mut saw_input_delta = false;
        let mut final_response = None;

        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::ToolUseStart { id, name } => {
                    saw_tool_start = true;
                    assert_eq!(id, "tool-1");
                    assert_eq!(name, "Skill");
                }
                StreamEvent::ToolUseInputDelta(delta) => {
                    saw_input_delta = true;
                    assert_eq!(delta, r#"{"prompt":"run","skill_name":"hello-skill"}"#);
                }
                StreamEvent::Done(resp) => {
                    final_response = Some(resp);
                    break;
                }
                _ => {}
            }
        }

        assert!(saw_tool_start);
        assert!(saw_input_delta);

        let resp = final_response.expect("expected final response");
        let tool_calls = resp.tool_calls();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "Skill");
        assert_eq!(tool_calls[0].args["skill_name"], "hello-skill");
        assert_eq!(tool_calls[0].args["prompt"], "run");
    }

    #[test]
    fn test_anthropic_stream_message_delta() {
        let json = r#"{
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 50}
        }"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, AnthropicStreamEvent::MessageDelta { .. }));
    }

    #[test]
    fn test_anthropic_stream_message_stop() {
        let json = r#"{"type": "message_stop"}"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, AnthropicStreamEvent::MessageStop));
    }

    #[test]
    fn test_anthropic_stream_ping() {
        let json = r#"{"type": "ping"}"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, AnthropicStreamEvent::Ping));
    }

    #[test]
    fn test_anthropic_stream_error() {
        let json = r#"{
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Server overloaded"}
        }"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(event, AnthropicStreamEvent::Error { .. }));
    }

    // ========================================================================
    // OpenAI Streaming Event Deserialization
    // ========================================================================

    #[test]
    fn test_openai_stream_chunk_text() {
        let json = r#"{
            "choices": [{
                "delta": {"content": "Hello"},
                "finish_reason": null
            }],
            "usage": null
        }"#;
        let chunk: OpenAiStreamChunk = serde_json::from_str(json).unwrap();
        assert_eq!(
            chunk.choices[0].delta.as_ref().unwrap().content,
            Some("Hello".to_string())
        );
        assert!(chunk.choices[0].finish_reason.is_none());
    }

    #[test]
    fn test_openai_stream_chunk_tool_call() {
        let json = r#"{
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call-1",
                        "function": {"name": "bash", "arguments": ""}
                    }]
                },
                "finish_reason": null
            }],
            "usage": null
        }"#;
        let chunk: OpenAiStreamChunk = serde_json::from_str(json).unwrap();
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        let tool_calls = delta.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls[0].id, Some("call-1".to_string()));
        assert_eq!(
            tool_calls[0].function.as_ref().unwrap().name,
            Some("bash".to_string())
        );
    }

    #[test]
    fn test_openai_stream_chunk_done() {
        let json = r#"{
            "choices": [{
                "delta": {},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }"#;
        let chunk: OpenAiStreamChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices[0].finish_reason, Some("stop".to_string()));
        assert!(chunk.usage.is_some());
    }

    #[test]
    fn test_openai_stream_chunk_minimax_usage_defaults_missing_prompt_fields() {
        let json = r#"{
            "choices": [{
                "delta": {"content": ""},
                "finish_reason": null
            }],
            "usage": {"total_tokens": 0, "total_characters": 0}
        }"#;
        let chunk: OpenAiStreamChunk = serde_json::from_str(json).unwrap();
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn test_openai_stream_chunk_minimax_final_message_content() {
        let json = r#"{
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": "OK",
                    "role": "assistant",
                    "reasoning_content": "done"
                }
            }],
            "usage": {"total_tokens": 94, "prompt_tokens": 46, "completion_tokens": 48}
        }"#;
        let chunk: OpenAiStreamChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices[0].finish_reason, Some("stop".to_string()));
        let message = chunk.choices[0].message.as_ref().unwrap();
        assert_eq!(message.content.as_deref(), Some("OK"));
        assert_eq!(message.reasoning_content.as_deref(), Some("done"));
    }

    // ========================================================================
    // LlmConfig and create_client_with_config
    // ========================================================================

    #[test]
    fn test_llm_config_with_retry() {
        let retry = RetryConfig::default();
        let config =
            LlmConfig::new("anthropic", "claude-sonnet-4-20250514", "key").with_retry_config(retry);
        assert!(config.retry_config.is_some());
    }

    #[test]
    fn test_create_client_anthropic() {
        let config = LlmConfig::new("anthropic", "claude-sonnet-4-20250514", "key");
        let _client = create_client_with_config(config);
    }

    #[test]
    fn test_create_client_openai() {
        let config = LlmConfig::new("openai", "gpt-4", "key");
        let _client = create_client_with_config(config);
    }

    #[test]
    fn test_create_client_unknown_defaults_anthropic() {
        let config = LlmConfig::new("unknown_provider", "model", "key");
        let _client = create_client_with_config(config);
    }

    #[test]
    fn test_create_client_with_base_url() {
        let config =
            LlmConfig::new("openai", "gpt-4", "key").with_base_url("https://custom.api.com");
        let _client = create_client_with_config(config);
    }

    // ========================================================================
    // normalize_base_url
    // ========================================================================

    #[test]
    fn test_normalize_base_url_strips_trailing_slash() {
        assert_eq!(
            normalize_base_url("https://api.example.com/"),
            "https://api.example.com"
        );
    }

    #[test]
    fn test_normalize_base_url_strips_v1() {
        assert_eq!(
            normalize_base_url("https://api.example.com/v1"),
            "https://api.example.com"
        );
    }

    #[test]
    fn test_normalize_base_url_strips_v1_slash() {
        assert_eq!(
            normalize_base_url("https://api.example.com/v1/"),
            "https://api.example.com"
        );
    }

    #[test]
    fn test_normalize_base_url_no_change() {
        assert_eq!(
            normalize_base_url("https://api.example.com"),
            "https://api.example.com"
        );
    }

    // ========================================================================
    // AnthropicClient with_retry_config
    // ========================================================================

    #[test]
    fn test_anthropic_client_with_retry_config() {
        let retry = RetryConfig::default();
        let client = AnthropicClient::new("key".to_string(), "model".to_string())
            .with_retry_config(retry.clone());
        assert_eq!(client.retry_config.max_retries, retry.max_retries);
    }

    #[test]
    fn test_openai_client_with_retry_config() {
        let retry = RetryConfig::default();
        let client = OpenAiClient::new("key".to_string(), "model".to_string())
            .with_retry_config(retry.clone());
        assert_eq!(client.retry_config.max_retries, retry.max_retries);
    }

    #[test]
    fn test_openai_client_with_base_url() {
        let client = OpenAiClient::new("key".to_string(), "model".to_string())
            .with_base_url("https://custom.openai.com".to_string());
        assert_eq!(client.base_url, "https://custom.openai.com");
    }

    #[test]
    fn test_openai_client_request_headers_with_custom_headers() {
        let mut headers = std::collections::HashMap::new();
        headers.insert("X-Session-Id".to_string(), "sess-123".to_string());
        headers.insert("X-Test".to_string(), "value".to_string());

        let client =
            OpenAiClient::new("key".to_string(), "model".to_string()).with_headers(headers);
        let request_headers = client.request_headers();

        assert!(request_headers
            .iter()
            .any(|(key, value)| key == "Authorization" && value == "Bearer key"));
        assert!(request_headers
            .iter()
            .any(|(key, value)| key == "X-Session-Id" && value == "sess-123"));
        assert!(request_headers
            .iter()
            .any(|(key, value)| key == "X-Test" && value == "value"));
    }

    #[test]
    fn test_openai_client_request_headers_respects_custom_authorization() {
        let mut headers = std::collections::HashMap::new();
        headers.insert(
            "Authorization".to_string(),
            "Bearer override-token".to_string(),
        );

        let client =
            OpenAiClient::new("key".to_string(), "model".to_string()).with_headers(headers);
        let request_headers = client.request_headers();

        assert_eq!(
            request_headers
                .iter()
                .filter(|(key, _)| key.eq_ignore_ascii_case("authorization"))
                .count(),
            1
        );
        assert!(request_headers.iter().any(|(key, value)| {
            key.eq_ignore_ascii_case("authorization") && value == "Bearer override-token"
        }));
    }

    #[test]
    fn test_anthropic_client_with_base_url() {
        let client = AnthropicClient::new("key".to_string(), "model".to_string())
            .with_base_url("https://custom.anthropic.com".to_string());
        assert_eq!(client.base_url, "https://custom.anthropic.com");
    }

    // ========================================================================
    // Additional Coverage Tests
    // ========================================================================

    #[test]
    fn test_message_user_creates_text_block() {
        let msg = Message::user("test message");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content.len(), 1);
        match &msg.content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "test message"),
            _ => panic!("Expected Text block"),
        }
    }

    #[test]
    fn test_message_tool_result_with_error_flag() {
        let msg = Message::tool_result("id-123", "error occurred", true);
        assert_eq!(msg.role, "user");
        match &msg.content[0] {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                assert_eq!(tool_use_id, "id-123");
                assert_eq!(content, "error occurred");
                assert_eq!(*is_error, Some(true));
            }
            _ => panic!("Expected ToolResult block"),
        }
    }

    #[test]
    fn test_message_text_extracts_only_text_blocks() {
        let msg = Message {
            role: "assistant".to_string(),
            content: vec![
                ContentBlock::Text {
                    text: "First ".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "t1".to_string(),
                    name: "bash".to_string(),
                    input: serde_json::json!({}),
                },
                ContentBlock::Text {
                    text: "Second".to_string(),
                },
            ],
            reasoning_content: None,
        };
        assert_eq!(msg.text(), "First Second");
    }

    #[test]
    fn test_message_text_empty_content() {
        let msg = Message {
            role: "user".to_string(),
            content: vec![],
            reasoning_content: None,
        };
        assert_eq!(msg.text(), "");
    }

    #[test]
    fn test_message_tool_calls_extracts_multiple() {
        let msg = Message {
            role: "assistant".to_string(),
            content: vec![
                ContentBlock::ToolUse {
                    id: "call-1".to_string(),
                    name: "read".to_string(),
                    input: serde_json::json!({"path": "file.txt"}),
                },
                ContentBlock::Text {
                    text: "text".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "call-2".to_string(),
                    name: "write".to_string(),
                    input: serde_json::json!({"path": "out.txt"}),
                },
            ],
            reasoning_content: None,
        };
        let calls = msg.tool_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "call-1");
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[1].id, "call-2");
        assert_eq!(calls[1].name, "write");
    }

    #[test]
    fn test_message_tool_calls_empty_when_no_tool_use() {
        let msg = Message {
            role: "assistant".to_string(),
            content: vec![ContentBlock::Text {
                text: "hello".to_string(),
            }],
            reasoning_content: None,
        };
        assert!(msg.tool_calls().is_empty());
    }

    #[test]
    fn test_llm_response_text_delegates_to_message() {
        let response = LlmResponse {
            message: Message {
                role: "assistant".to_string(),
                content: vec![ContentBlock::Text {
                    text: "response text".to_string(),
                }],
                reasoning_content: None,
            },
            usage: TokenUsage::default(),
            stop_reason: None,
            token_logprobs: Vec::new(),
            meta: None,
        };
        assert_eq!(response.text(), "response text");
    }

    #[test]
    fn test_llm_response_tool_calls_delegates_to_message() {
        let response = LlmResponse {
            message: Message {
                role: "assistant".to_string(),
                content: vec![ContentBlock::ToolUse {
                    id: "t1".to_string(),
                    name: "bash".to_string(),
                    input: serde_json::json!({"cmd": "ls"}),
                }],
                reasoning_content: None,
            },
            usage: TokenUsage::default(),
            stop_reason: Some("tool_use".to_string()),
            token_logprobs: Vec::new(),
            meta: None,
        };
        let calls = response.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
    }

    #[test]
    fn test_token_usage_with_cache_tokens() {
        let usage = TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            cache_read_tokens: Some(80),
            cache_write_tokens: Some(20),
        };
        assert_eq!(usage.cache_read_tokens, Some(80));
        assert_eq!(usage.cache_write_tokens, Some(20));
    }

    #[test]
    fn test_anthropic_client_new_defaults() {
        let client = AnthropicClient::new("test-key".to_string(), "claude-3".to_string());
        assert_eq!(client.api_key.expose(), "test-key");
        assert_eq!(client.model, "claude-3");
        assert_eq!(client.base_url, "https://api.anthropic.com");
        assert_eq!(client.max_tokens, DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn test_anthropic_client_with_max_tokens() {
        let client =
            AnthropicClient::new("key".to_string(), "model".to_string()).with_max_tokens(16384);
        assert_eq!(client.max_tokens, 16384);
    }

    #[test]
    fn test_anthropic_client_builder_chain() {
        let client = AnthropicClient::new("key".to_string(), "model".to_string())
            .with_base_url("https://custom.com".to_string())
            .with_max_tokens(4096)
            .with_retry_config(RetryConfig::default());
        assert_eq!(client.base_url, "https://custom.com");
        assert_eq!(client.max_tokens, 4096);
    }

    #[test]
    fn test_anthropic_build_request_empty_messages() {
        let client = AnthropicClient::new("key".to_string(), "model".to_string());
        let req = client.build_request(&[], None, &[]);
        assert!(req["messages"].is_array());
        assert_eq!(req["messages"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_anthropic_build_request_multiple_tools() {
        let client = AnthropicClient::new("key".to_string(), "model".to_string());
        let tools = vec![
            ToolDefinition {
                name: "tool1".to_string(),
                description: "First tool".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            },
            ToolDefinition {
                name: "tool2".to_string(),
                description: "Second tool".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            },
        ];
        let req = client.build_request(&[], None, &tools);
        assert_eq!(req["tools"].as_array().unwrap().len(), 2);
        assert_eq!(req["tools"][0]["name"], "tool1");
        assert_eq!(req["tools"][1]["name"], "tool2");
    }

    #[test]
    fn test_openai_client_new_defaults() {
        let client = OpenAiClient::new("test-key".to_string(), "gpt-4".to_string());
        assert_eq!(client.api_key.expose(), "test-key");
        assert_eq!(client.model, "gpt-4");
        assert_eq!(client.base_url, "https://api.openai.com");
    }

    #[test]
    fn test_openai_client_builder_chain() {
        let client = OpenAiClient::new("key".to_string(), "model".to_string())
            .with_base_url("https://custom.com".to_string())
            .with_retry_config(RetryConfig::default());
        assert_eq!(client.base_url, "https://custom.com");
    }

    #[test]
    fn test_openai_convert_messages_empty() {
        let client = OpenAiClient::new("key".to_string(), "model".to_string());
        let converted = client.convert_messages(&[]);
        assert_eq!(converted.len(), 0);
    }

    #[test]
    fn test_openai_convert_messages_single_text_block() {
        let client = OpenAiClient::new("key".to_string(), "model".to_string());
        let msgs = vec![Message {
            role: "user".to_string(),
            content: vec![ContentBlock::Text {
                text: "Hello".to_string(),
            }],
            reasoning_content: None,
        }];
        let converted = client.convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["role"], "user");
        assert_eq!(converted[0]["content"], "Hello");
    }

    #[test]
    fn test_openai_convert_messages_tool_result_single_block() {
        let client = OpenAiClient::new("key".to_string(), "model".to_string());
        let msgs = vec![Message {
            role: "user".to_string(),
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "call-123".to_string(),
                content: "result".into(),
                is_error: Some(false),
            }],
            reasoning_content: None,
        }];
        let converted = client.convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["role"], "tool");
        assert_eq!(converted[0]["tool_call_id"], "call-123");
        assert_eq!(converted[0]["content"], "result");
    }

    #[test]
    fn test_openai_convert_messages_assistant_text_only() {
        let client = OpenAiClient::new("key".to_string(), "model".to_string());
        let msgs = vec![Message {
            role: "assistant".to_string(),
            content: vec![ContentBlock::Text {
                text: "Response".to_string(),
            }],
            reasoning_content: None,
        }];
        let converted = client.convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["role"], "assistant");
        assert_eq!(converted[0]["content"], "Response");
    }

    #[test]
    fn test_openai_convert_messages_multi_block_with_tool_use() {
        let client = OpenAiClient::new("key".to_string(), "model".to_string());
        let msgs = vec![Message {
            role: "user".to_string(),
            content: vec![
                ContentBlock::Text {
                    text: "Part1".to_string(),
                },
                ContentBlock::Text {
                    text: "Part2".to_string(),
                },
            ],
            reasoning_content: None,
        }];
        let converted = client.convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        assert!(converted[0]["content"].is_array());
        let content_arr = converted[0]["content"].as_array().unwrap();
        assert_eq!(content_arr.len(), 2);
        assert_eq!(content_arr[0]["type"], "text");
        assert_eq!(content_arr[0]["text"], "Part1");
    }

    #[test]
    fn test_openai_convert_tools_single() {
        let client = OpenAiClient::new("key".to_string(), "model".to_string());
        let tools = vec![ToolDefinition {
            name: "search".to_string(),
            description: "Search files".to_string(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
        }];
        let converted = client.convert_tools(&tools);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["type"], "function");
        assert_eq!(converted[0]["function"]["name"], "search");
        assert_eq!(converted[0]["function"]["description"], "Search files");
    }

    #[test]
    fn test_normalize_base_url_multiple_trailing_slashes() {
        assert_eq!(normalize_base_url("https://api.com///"), "https://api.com");
    }

    #[test]
    fn test_normalize_base_url_with_port() {
        assert_eq!(
            normalize_base_url("http://localhost:3000/v1"),
            "http://localhost:3000"
        );
    }

    #[test]
    fn test_normalize_base_url_already_normalized() {
        assert_eq!(
            normalize_base_url("https://api.example.com"),
            "https://api.example.com"
        );
    }

    #[test]
    fn test_llm_config_new_basic() {
        let config = LlmConfig::new("openai", "gpt-4", "sk-123");
        assert_eq!(config.provider, "openai");
        assert_eq!(config.model, "gpt-4");
        assert_eq!(config.api_key.expose(), "sk-123");
        assert!(config.base_url.is_none());
        assert!(config.retry_config.is_none());
    }

    #[test]
    fn test_llm_config_with_base_url() {
        let config =
            LlmConfig::new("openai", "gpt-4", "key").with_base_url("https://custom.api.com");
        assert_eq!(config.base_url, Some("https://custom.api.com".to_string()));
    }

    #[test]
    fn test_llm_config_resolved_headers_with_runtime_session() {
        let mut headers = std::collections::HashMap::new();
        headers.insert("X-Test".to_string(), "value".to_string());

        let config = LlmConfig::new("openai", "gpt-4", "key")
            .with_headers(headers)
            .with_session_id_header("X-Session-Id")
            .with_session_id("sess-456");

        let resolved_headers = config.resolved_headers();
        assert_eq!(
            resolved_headers.get("X-Session-Id"),
            Some(&"sess-456".to_string())
        );
        assert_eq!(resolved_headers.get("X-Test"), Some(&"value".to_string()));
    }

    #[test]
    fn test_llm_config_with_retry_config() {
        let retry = RetryConfig::default();
        let config = LlmConfig::new("openai", "gpt-4", "key").with_retry_config(retry.clone());
        assert!(config.retry_config.is_some());
        assert_eq!(config.retry_config.unwrap().max_retries, retry.max_retries);
    }

    #[test]
    fn test_llm_config_builder_chain() {
        let retry = RetryConfig::default();
        let config = LlmConfig::new("anthropic", "claude", "key")
            .with_base_url("https://api.com")
            .with_retry_config(retry);
        assert_eq!(config.provider, "anthropic");
        assert_eq!(config.base_url, Some("https://api.com".to_string()));
        assert!(config.retry_config.is_some());
    }

    #[test]
    fn test_create_client_with_config_anthropic() {
        let config = LlmConfig::new("anthropic", "claude-3", "key");
        let client = create_client_with_config(config);
        // Just verify it creates without panic
        assert!(Arc::strong_count(&client) >= 1);
    }

    #[test]
    fn test_create_client_with_config_claude_alias() {
        let config = LlmConfig::new("claude", "claude-3", "key");
        let client = create_client_with_config(config);
        assert!(Arc::strong_count(&client) >= 1);
    }

    #[test]
    fn test_create_client_with_config_openai() {
        let config = LlmConfig::new("openai", "gpt-4", "key");
        let client = create_client_with_config(config);
        assert!(Arc::strong_count(&client) >= 1);
    }

    #[test]
    fn test_create_client_with_config_gpt_alias() {
        let config = LlmConfig::new("gpt", "gpt-4", "key");
        let client = create_client_with_config(config);
        assert!(Arc::strong_count(&client) >= 1);
    }

    #[test]
    fn test_create_client_with_config_unknown_provider() {
        let config = LlmConfig::new("deepseek", "deepseek-chat", "key");
        let client = create_client_with_config(config);
        // Should default to OpenAI-compatible
        assert!(Arc::strong_count(&client) >= 1);
    }

    #[test]
    fn test_create_client_with_config_with_base_url() {
        let config =
            LlmConfig::new("openai", "gpt-4", "key").with_base_url("https://custom.openai.com");
        let client = create_client_with_config(config);
        assert!(Arc::strong_count(&client) >= 1);
    }

    #[test]
    fn test_create_client_with_config_with_retry() {
        let retry = RetryConfig::default();
        let config = LlmConfig::new("anthropic", "claude", "key").with_retry_config(retry);
        let client = create_client_with_config(config);
        assert!(Arc::strong_count(&client) >= 1);
    }

    #[test]
    fn test_content_block_text_deserialization() {
        let json = r#"{"type":"text","text":"hello world"}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        match block {
            ContentBlock::Text { text } => assert_eq!(text, "hello world"),
            _ => panic!("Expected Text block"),
        }
    }

    #[test]
    fn test_content_block_tool_use_deserialization() {
        let json = r#"{"type":"tool_use","id":"t1","name":"bash","input":{"cmd":"ls"}}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        match block {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "t1");
                assert_eq!(name, "bash");
                assert_eq!(input["cmd"], "ls");
            }
            _ => panic!("Expected ToolUse block"),
        }
    }

    #[test]
    fn test_content_block_tool_result_deserialization() {
        let json =
            r#"{"type":"tool_result","tool_use_id":"t1","content":"output","is_error":false}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        match block {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                assert_eq!(tool_use_id, "t1");
                assert_eq!(content, "output");
                assert_eq!(is_error, Some(false));
            }
            _ => panic!("Expected ToolResult block"),
        }
    }

    #[test]
    fn test_content_block_tool_result_no_error_flag() {
        let json = r#"{"type":"tool_result","tool_use_id":"t1","content":"output"}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        match block {
            ContentBlock::ToolResult { is_error, .. } => {
                assert_eq!(is_error, None);
            }
            _ => panic!("Expected ToolResult block"),
        }
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message::user("test");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"type\":\"text\""));
    }

    #[test]
    fn test_message_deserialization() {
        let json = r#"{"role":"assistant","content":[{"type":"text","text":"hi"}]}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.text(), "hi");
    }

    #[test]
    fn test_tool_definition_serialization() {
        let tool = ToolDefinition {
            name: "test".to_string(),
            description: "Test tool".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        };
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("\"name\":\"test\""));
        assert!(json.contains("\"description\":\"Test tool\""));
    }

    #[test]
    fn test_tool_definition_deserialization() {
        let json = r#"{"name":"bash","description":"Run command","parameters":{"type":"object"}}"#;
        let tool: ToolDefinition = serde_json::from_str(json).unwrap();
        assert_eq!(tool.name, "bash");
        assert_eq!(tool.description, "Run command");
    }

    #[test]
    fn test_llm_response_serialization() {
        let response = LlmResponse {
            message: Message::user("test"),
            usage: TokenUsage::default(),
            stop_reason: Some("end_turn".to_string()),
            token_logprobs: Vec::new(),
            meta: None,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"stop_reason\":\"end_turn\""));
    }

    #[test]
    fn test_llm_response_deserialization() {
        let json = r#"{"message":{"role":"assistant","content":[{"type":"text","text":"hi"}]},"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15},"stop_reason":"end_turn"}"#;
        let response: LlmResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.text(), "hi");
        assert_eq!(response.usage.total_tokens, 15);
    }

    #[test]
    fn test_tool_call_serialization() {
        let call = ToolCall {
            id: "call-1".to_string(),
            name: "bash".to_string(),
            args: serde_json::json!({"cmd": "ls"}),
        };
        let json = serde_json::to_string(&call).unwrap();
        assert!(json.contains("\"id\":\"call-1\""));
        assert!(json.contains("\"name\":\"bash\""));
    }

    #[test]
    fn test_tool_call_deserialization() {
        let json = r#"{"id":"call-1","name":"read","args":{"file":"test.txt"}}"#;
        let call: ToolCall = serde_json::from_str(json).unwrap();
        assert_eq!(call.id, "call-1");
        assert_eq!(call.name, "read");
        assert_eq!(call.args["file"], "test.txt");
    }

    #[test]
    fn test_openai_convert_messages_assistant_empty_text() {
        let client = OpenAiClient::new("key".to_string(), "model".to_string());
        let msgs = vec![Message {
            role: "assistant".to_string(),
            content: vec![ContentBlock::ToolUse {
                id: "call-1".to_string(),
                name: "bash".to_string(),
                input: serde_json::json!({"cmd": "ls"}),
            }],
            reasoning_content: None,
        }];
        let converted = client.convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["role"], "assistant");
        assert!(converted[0]["tool_calls"].is_array());
        assert_eq!(converted[0]["content"], "");
    }

    #[test]
    fn test_openai_convert_messages_multiple_messages() {
        let client = OpenAiClient::new("key".to_string(), "model".to_string());
        let msgs = vec![
            Message::user("Hello"),
            Message {
                role: "assistant".to_string(),
                content: vec![ContentBlock::Text {
                    text: "Hi".to_string(),
                }],
                reasoning_content: None,
            },
            Message::user("How are you?"),
        ];
        let converted = client.convert_messages(&msgs);
        assert_eq!(converted.len(), 3);
        assert_eq!(converted[0]["role"], "user");
        assert_eq!(converted[1]["role"], "assistant");
        assert_eq!(converted[2]["role"], "user");
    }
}
#[cfg(test)]
mod extra_llm_tests3 {
    use crate::llm::anthropic::*;
    use crate::llm::http::normalize_base_url;
    use crate::llm::openai::*;
    use crate::llm::*;
    use crate::retry::RetryConfig;

    // ========================================================================
    // OpenAiClient convert_messages - Additional Coverage
    // ========================================================================

    #[test]
    fn test_openai_convert_messages_single_non_text_block() {
        let client = OpenAiClient::new("key".to_string(), "model".to_string());
        let msgs = vec![Message {
            role: "user".to_string(),
            content: vec![ContentBlock::ToolUse {
                id: "t1".to_string(),
                name: "bash".to_string(),
                input: serde_json::json!({"cmd": "ls"}),
            }],
            reasoning_content: None,
        }];
        let converted = client.convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        // Single non-text block should result in empty string
        assert_eq!(converted[0]["content"], "");
    }

    #[test]
    fn test_openai_convert_messages_multi_block_with_tool_use() {
        let client = OpenAiClient::new("key".to_string(), "model".to_string());
        let msgs = vec![Message {
            role: "user".to_string(),
            content: vec![
                ContentBlock::Text {
                    text: "First".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "t1".to_string(),
                    name: "bash".to_string(),
                    input: serde_json::json!({"cmd": "ls"}),
                },
            ],
            reasoning_content: None,
        }];
        let converted = client.convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        // Multi-block should be an array
        assert!(converted[0]["content"].is_array());
        let content_arr = converted[0]["content"].as_array().unwrap();
        assert_eq!(content_arr.len(), 2);
        assert_eq!(content_arr[0]["type"], "text");
        assert_eq!(content_arr[1]["type"], "function");
    }

    #[test]
    fn test_openai_convert_messages_multi_block_with_tool_result() {
        let client = OpenAiClient::new("key".to_string(), "model".to_string());
        let msgs = vec![Message {
            role: "user".to_string(),
            content: vec![
                ContentBlock::Text {
                    text: "Text".to_string(),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "t1".to_string(),
                    content: "result".into(),
                    is_error: Some(false),
                },
            ],
            reasoning_content: None,
        }];
        let converted = client.convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        // Multi-block with ToolResult should be array with empty object for ToolResult
        assert!(converted[0]["content"].is_array());
        let content_arr = converted[0]["content"].as_array().unwrap();
        assert_eq!(content_arr.len(), 2);
        assert_eq!(content_arr[0]["type"], "text");
        assert_eq!(content_arr[1], serde_json::json!({}));
    }

    #[test]
    fn test_openai_convert_messages_assistant_with_text_and_tool_calls() {
        let client = OpenAiClient::new("key".to_string(), "model".to_string());
        let msgs = vec![Message {
            role: "assistant".to_string(),
            content: vec![
                ContentBlock::Text {
                    text: "Let me check".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "call-1".to_string(),
                    name: "bash".to_string(),
                    input: serde_json::json!({"cmd": "ls"}),
                },
            ],
            reasoning_content: None,
        }];
        let converted = client.convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["role"], "assistant");
        assert_eq!(converted[0]["content"], "Let me check");
        assert!(converted[0]["tool_calls"].is_array());
        let tool_calls = converted[0]["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["id"], "call-1");
        assert_eq!(tool_calls[0]["function"]["name"], "bash");
    }

    #[test]
    fn test_openai_convert_messages_assistant_no_tool_calls() {
        let client = OpenAiClient::new("key".to_string(), "model".to_string());
        let msgs = vec![Message {
            role: "assistant".to_string(),
            content: vec![ContentBlock::Text {
                text: "Hello".to_string(),
            }],
            reasoning_content: None,
        }];
        let converted = client.convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["role"], "assistant");
        assert_eq!(converted[0]["content"], "Hello");
        assert!(converted[0]["tool_calls"].is_null());
    }

    // ========================================================================
    // OpenAI Response Parsing - Error Cases
    // ========================================================================

    #[test]
    fn test_openai_response_empty_choices() {
        let json = r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#;
        let resp: OpenAiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.choices.len(), 0);
    }

    #[test]
    fn test_openai_response_empty_content_string() {
        let json = r#"{"choices":[{"message":{"content":"","tool_calls":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#;
        let resp: OpenAiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.choices[0].message.content, Some("".to_string()));
    }

    #[test]
    fn test_openai_response_invalid_tool_arguments() {
        let json = r#"{"choices":[{"message":{"content":null,"tool_calls":[{"id":"call-1","function":{"name":"bash","arguments":"invalid json"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#;
        let resp: OpenAiResponse = serde_json::from_str(json).unwrap();
        let tool_calls = resp.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls[0].function.arguments, "invalid json");
    }

    #[test]
    fn test_openai_parse_tool_arguments_preserves_parse_error() {
        let parsed = OpenAiClient::parse_tool_arguments("Skill", "invalid json");
        let err = parsed
            .get("__parse_error")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        assert!(err.contains("Malformed tool arguments"));
        assert!(err.contains("invalid json"));
    }

    // ========================================================================
    // Anthropic Response Parsing
    // ========================================================================

    #[test]
    fn test_anthropic_response_multiple_text_blocks() {
        let json = r#"{
            "content": [
                {"type": "text", "text": "First "},
                {"type": "text", "text": "Second"}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }"#;
        let resp: AnthropicResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.content.len(), 2);
    }

    #[test]
    fn test_anthropic_response_mixed_content() {
        let json = r#"{
            "content": [
                {"type": "text", "text": "Let me help"},
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}},
                {"type": "text", "text": "Done"}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 15}
        }"#;
        let resp: AnthropicResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.content.len(), 3);
        match &resp.content[0] {
            AnthropicContentBlock::Text { text } => assert_eq!(text, "Let me help"),
            _ => panic!("Expected text block"),
        }
        match &resp.content[1] {
            AnthropicContentBlock::ToolUse { id, name, .. } => {
                assert_eq!(id, "t1");
                assert_eq!(name, "bash");
            }
            _ => panic!("Expected tool_use block"),
        }
    }

    // ========================================================================
    // Stream Event Parsing - Additional Coverage
    // ========================================================================

    #[test]
    fn test_anthropic_stream_content_block_stop() {
        let json = r#"{"type": "content_block_stop", "index": 0}"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(
            event,
            AnthropicStreamEvent::ContentBlockStop { .. }
        ));
    }

    #[test]
    fn test_anthropic_stream_multiple_deltas() {
        let json1 = r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}"#;
        let json2 = r#"{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " World"}}"#;

        let event1: AnthropicStreamEvent = serde_json::from_str(json1).unwrap();
        let event2: AnthropicStreamEvent = serde_json::from_str(json2).unwrap();

        assert!(matches!(
            event1,
            AnthropicStreamEvent::ContentBlockDelta { .. }
        ));
        assert!(matches!(
            event2,
            AnthropicStreamEvent::ContentBlockDelta { .. }
        ));
    }

    #[test]
    fn test_openai_stream_chunk_empty_delta() {
        let json = r#"{"choices":[{"delta":{},"finish_reason":null}],"usage":null}"#;
        let chunk: OpenAiStreamChunk = serde_json::from_str(json).unwrap();
        assert!(chunk.choices[0].delta.is_some());
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        assert!(delta.content.is_none());
        assert!(delta.tool_calls.is_none());
    }

    #[test]
    fn test_openai_stream_chunk_tool_call_delta_no_id() {
        let json = r#"{
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": "{\"cmd\":"}
                    }]
                },
                "finish_reason": null
            }],
            "usage": null
        }"#;
        let chunk: OpenAiStreamChunk = serde_json::from_str(json).unwrap();
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        let tool_calls = delta.tool_calls.as_ref().unwrap();
        assert!(tool_calls[0].id.is_none());
        assert_eq!(
            tool_calls[0].function.as_ref().unwrap().arguments,
            Some("{\"cmd\":".to_string())
        );
    }

    #[test]
    fn test_openai_stream_chunk_multiple_tool_calls() {
        let json = r#"{
            "choices": [{
                "delta": {
                    "tool_calls": [
                        {"index": 0, "id": "call-1", "function": {"name": "bash"}},
                        {"index": 1, "id": "call-2", "function": {"name": "read"}}
                    ]
                },
                "finish_reason": null
            }],
            "usage": null
        }"#;
        let chunk: OpenAiStreamChunk = serde_json::from_str(json).unwrap();
        let delta = chunk.choices[0].delta.as_ref().unwrap();
        let tool_calls = delta.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].index, 0);
        assert_eq!(tool_calls[1].index, 1);
    }

    // ========================================================================
    // LlmConfig Builder Pattern
    // ========================================================================

    #[test]
    fn test_llm_config_default() {
        let config = LlmConfig::default();
        assert_eq!(config.provider, "");
        assert_eq!(config.model, "");
        assert_eq!(config.api_key.expose(), "");
        assert!(config.base_url.is_none());
        assert!(config.retry_config.is_none());
    }

    #[test]
    fn test_llm_config_full_builder() {
        let retry = RetryConfig::default();
        let config = LlmConfig::new("anthropic", "claude-3", "key")
            .with_base_url("https://custom.com")
            .with_retry_config(retry.clone());

        assert_eq!(config.provider, "anthropic");
        assert_eq!(config.model, "claude-3");
        assert_eq!(config.api_key.expose(), "key");
        assert_eq!(config.base_url, Some("https://custom.com".to_string()));
        assert!(config.retry_config.is_some());
    }

    // ========================================================================
    // create_client_with_config - Provider Variants
    // ========================================================================

    #[test]
    fn test_create_client_claude_provider() {
        let config = LlmConfig::new("claude", "claude-3", "key");
        let _client = create_client_with_config(config);
    }

    #[test]
    fn test_create_client_gpt_provider() {
        let config = LlmConfig::new("gpt", "gpt-4", "key");
        let _client = create_client_with_config(config);
    }

    #[test]
    fn test_create_client_deepseek_provider() {
        let config = LlmConfig::new("deepseek", "deepseek-chat", "key")
            .with_base_url("https://api.deepseek.com");
        let _client = create_client_with_config(config);
    }

    #[test]
    fn test_create_client_groq_provider() {
        let config = LlmConfig::new("groq", "llama-3", "key").with_base_url("https://api.groq.com");
        let _client = create_client_with_config(config);
    }

    #[test]
    fn test_create_client_ollama_provider() {
        let config =
            LlmConfig::new("ollama", "llama2", "key").with_base_url("http://localhost:11434");
        let _client = create_client_with_config(config);
    }

    #[test]
    fn test_create_client_with_retry_config() {
        let retry = RetryConfig::default();
        let config = LlmConfig::new("openai", "gpt-4", "key").with_retry_config(retry);
        let _client = create_client_with_config(config);
    }

    // ========================================================================
    // Client Builder Methods
    // ========================================================================

    #[test]
    fn test_anthropic_client_builder_all_options() {
        let retry = RetryConfig::default();
        let client = AnthropicClient::new("key".to_string(), "model".to_string())
            .with_base_url("https://custom.com".to_string())
            .with_max_tokens(16384)
            .with_retry_config(retry);

        assert_eq!(client.base_url, "https://custom.com");
        assert_eq!(client.max_tokens, 16384);
    }

    #[test]
    fn test_openai_client_builder_all_options() {
        let retry = RetryConfig::default();
        let client = OpenAiClient::new("key".to_string(), "model".to_string())
            .with_base_url("https://custom.com".to_string())
            .with_retry_config(retry);

        assert_eq!(client.base_url, "https://custom.com");
    }

    #[test]
    fn test_openai_client_new_defaults() {
        let client = OpenAiClient::new("test-key".to_string(), "gpt-4".to_string());
        assert_eq!(client.api_key.expose(), "test-key");
        assert_eq!(client.model, "gpt-4");
        assert_eq!(client.base_url, "https://api.openai.com");
    }

    // ========================================================================
    // ContentBlock Variants
    // ========================================================================

    #[test]
    fn test_content_block_text_deserialization() {
        let json = r#"{"type":"text","text":"Hello"}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        match block {
            ContentBlock::Text { text } => assert_eq!(text, "Hello"),
            _ => panic!("Expected Text block"),
        }
    }

    #[test]
    fn test_content_block_tool_use_deserialization() {
        let json = r#"{"type":"tool_use","id":"t1","name":"bash","input":{"cmd":"ls"}}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        match block {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "t1");
                assert_eq!(name, "bash");
                assert_eq!(input["cmd"], "ls");
            }
            _ => panic!("Expected ToolUse block"),
        }
    }

    #[test]
    fn test_content_block_tool_result_deserialization() {
        let json =
            r#"{"type":"tool_result","tool_use_id":"t1","content":"output","is_error":false}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        match block {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                assert_eq!(tool_use_id, "t1");
                assert_eq!(content, "output");
                assert_eq!(is_error, Some(false));
            }
            _ => panic!("Expected ToolResult block"),
        }
    }

    #[test]
    fn test_content_block_tool_result_no_error_flag() {
        let json = r#"{"type":"tool_result","tool_use_id":"t1","content":"output"}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        match block {
            ContentBlock::ToolResult { is_error, .. } => {
                assert_eq!(is_error, None);
            }
            _ => panic!("Expected ToolResult block"),
        }
    }

    // ========================================================================
    // Message Serialization/Deserialization
    // ========================================================================

    #[test]
    fn test_message_serialization() {
        let msg = Message::user("Hello");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"text\":\"Hello\""));
    }

    #[test]
    fn test_message_deserialization() {
        let json = r#"{"role":"assistant","content":[{"type":"text","text":"Hi"}]}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.text(), "Hi");
    }

    #[test]
    fn test_message_with_multiple_content_blocks() {
        let msg = Message {
            role: "assistant".to_string(),
            content: vec![
                ContentBlock::Text {
                    text: "First".to_string(),
                },
                ContentBlock::Text {
                    text: "Second".to_string(),
                },
            ],
            reasoning_content: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.content.len(), 2);
    }

    // ========================================================================
    // TokenUsage
    // ========================================================================

    #[test]
    fn test_token_usage_serialization() {
        let usage = TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            cache_read_tokens: Some(80),
            cache_write_tokens: Some(20),
        };
        let json = serde_json::to_string(&usage).unwrap();
        assert!(json.contains("\"prompt_tokens\":100"));
        assert!(json.contains("\"cache_read_tokens\":80"));
    }

    #[test]
    fn test_token_usage_deserialization() {
        let json = r#"{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,"cache_read_tokens":80,"cache_write_tokens":20}"#;
        let usage: TokenUsage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.cache_read_tokens, Some(80));
    }

    #[test]
    fn test_token_usage_deserialization_no_cache() {
        let json = r#"{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150}"#;
        let usage: TokenUsage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert!(usage.cache_read_tokens.is_none());
        assert!(usage.cache_write_tokens.is_none());
    }

    // ========================================================================
    // ToolDefinition
    // ========================================================================

    #[test]
    fn test_tool_definition_with_complex_parameters() {
        let tool = ToolDefinition {
            name: "search".to_string(),
            description: "Search files".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"}
                },
                "required": ["query"]
            }),
        };
        let json = serde_json::to_string(&tool).unwrap();
        let parsed: ToolDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "search");
        assert_eq!(parsed.parameters["properties"]["query"]["type"], "string");
    }

    // ========================================================================
    // normalize_base_url Edge Cases
    // ========================================================================

    #[test]
    fn test_normalize_base_url_multiple_trailing_slashes() {
        assert_eq!(normalize_base_url("https://api.com///"), "https://api.com");
    }

    #[test]
    fn test_normalize_base_url_v1_with_multiple_slashes() {
        assert_eq!(
            normalize_base_url("https://api.com/v1///"),
            "https://api.com"
        );
    }

    #[test]
    fn test_normalize_base_url_empty_string() {
        assert_eq!(normalize_base_url(""), "");
    }

    #[test]
    fn test_normalize_base_url_only_slashes() {
        assert_eq!(normalize_base_url("///"), "");
    }

    // ========================================================================
    // Anthropic build_request - Edge Cases
    // ========================================================================

    #[test]
    fn test_anthropic_build_request_empty_messages() {
        let client = AnthropicClient::new("key".to_string(), "model".to_string());
        let req = client.build_request(&[], None, &[]);
        assert_eq!(req["messages"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_anthropic_build_request_multiple_tools() {
        let client = AnthropicClient::new("key".to_string(), "model".to_string());
        let tools = vec![
            ToolDefinition {
                name: "bash".to_string(),
                description: "Run command".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            },
            ToolDefinition {
                name: "read".to_string(),
                description: "Read file".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            },
        ];
        let req = client.build_request(&[], None, &tools);
        assert_eq!(req["tools"].as_array().unwrap().len(), 2);
        assert_eq!(req["tools"][0]["name"], "bash");
        assert_eq!(req["tools"][1]["name"], "read");
    }

    #[test]
    fn test_anthropic_build_request_system_and_tools() {
        let client = AnthropicClient::new("key".to_string(), "model".to_string());
        let tools = vec![ToolDefinition {
            name: "bash".to_string(),
            description: "Run".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        }];
        let req = client.build_request(&[Message::user("Hi")], Some("Be helpful"), &tools);
        // System is now an array with cache_control for prompt caching
        let system = req["system"].as_array().unwrap();
        assert_eq!(system[0]["text"], "Be helpful");
        assert_eq!(req["tools"].as_array().unwrap().len(), 1);
        // Last tool should have cache_control
        assert_eq!(req["tools"][0]["cache_control"]["type"], "ephemeral");
    }

    // ========================================================================
    // OpenAI convert_tools - Edge Cases
    // ========================================================================

    #[test]
    fn test_openai_convert_tools_multiple() {
        let client = OpenAiClient::new("key".to_string(), "model".to_string());
        let tools = vec![
            ToolDefinition {
                name: "bash".to_string(),
                description: "Run command".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            },
            ToolDefinition {
                name: "read".to_string(),
                description: "Read file".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            },
            ToolDefinition {
                name: "write".to_string(),
                description: "Write file".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            },
        ];
        let converted = client.convert_tools(&tools);
        assert_eq!(converted.len(), 3);
        assert_eq!(converted[0]["function"]["name"], "bash");
        assert_eq!(converted[1]["function"]["name"], "read");
        assert_eq!(converted[2]["function"]["name"], "write");
    }

    #[test]
    fn test_openai_convert_tools_with_complex_parameters() {
        let client = OpenAiClient::new("key".to_string(), "model".to_string());
        let tools = vec![ToolDefinition {
            name: "search".to_string(),
            description: "Search".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "number"}
                },
                "required": ["query"]
            }),
        }];
        let converted = client.convert_tools(&tools);
        assert_eq!(
            converted[0]["function"]["parameters"]["properties"]["query"]["type"],
            "string"
        );
        assert_eq!(
            converted[0]["function"]["parameters"]["required"][0],
            "query"
        );
    }
}

// ============================================================================
// Phase 6: Multi-Modal Support Tests
// ============================================================================

#[cfg(test)]
mod multimodal_tests {
    use crate::llm::openai::*;
    use crate::llm::types::*;

    // --- Attachment ---

    #[test]
    fn test_attachment_jpeg() {
        let a = Attachment::jpeg(vec![0xFF, 0xD8, 0xFF]);
        assert_eq!(a.media_type, "image/jpeg");
        assert_eq!(a.data, vec![0xFF, 0xD8, 0xFF]);
    }

    #[test]
    fn test_attachment_png() {
        let a = Attachment::png(vec![0x89, 0x50, 0x4E, 0x47]);
        assert_eq!(a.media_type, "image/png");
    }

    #[test]
    fn test_attachment_gif() {
        let a = Attachment::gif(vec![0x47, 0x49, 0x46]);
        assert_eq!(a.media_type, "image/gif");
    }

    #[test]
    fn test_attachment_webp() {
        let a = Attachment::webp(vec![0x52, 0x49, 0x46, 0x46]);
        assert_eq!(a.media_type, "image/webp");
    }

    #[test]
    fn test_attachment_new() {
        let a = Attachment::new(vec![1, 2, 3], "image/svg+xml");
        assert_eq!(a.media_type, "image/svg+xml");
        assert_eq!(a.data.len(), 3);
    }

    #[test]
    fn test_attachment_base64_data() {
        let a = Attachment::jpeg(vec![0xFF, 0xD8, 0xFF]);
        let b64 = a.base64_data();
        assert!(!b64.is_empty());
        // Verify round-trip
        use base64::{engine::general_purpose::STANDARD, Engine as _};
        let decoded = STANDARD.decode(&b64).unwrap();
        assert_eq!(decoded, vec![0xFF, 0xD8, 0xFF]);
    }

    #[test]
    fn test_attachment_to_content_block() {
        let a = Attachment::png(vec![1, 2, 3]);
        let block = a.to_content_block();
        match block {
            ContentBlock::Image { source } => {
                assert_eq!(source.source_type, "base64");
                assert_eq!(source.media_type, "image/png");
                assert!(!source.data.is_empty());
            }
            _ => panic!("Expected Image content block"),
        }
    }

    #[test]
    fn test_attachment_from_file_jpeg() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jpg");
        std::fs::write(&path, [0xFF, 0xD8, 0xFF]).unwrap();
        let a = Attachment::from_file(&path).unwrap();
        assert_eq!(a.media_type, "image/jpeg");
        assert_eq!(a.data, vec![0xFF, 0xD8, 0xFF]);
    }

    #[test]
    fn test_attachment_from_file_png() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.png");
        std::fs::write(&path, [0x89, 0x50]).unwrap();
        let a = Attachment::from_file(&path).unwrap();
        assert_eq!(a.media_type, "image/png");
    }

    #[test]
    fn test_attachment_from_file_unknown_ext() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");
        std::fs::write(&path, [0x00]).unwrap();
        let a = Attachment::from_file(&path).unwrap();
        assert_eq!(a.media_type, "application/octet-stream");
    }

    // --- ImageSource ---

    #[test]
    fn test_image_source_serialize() {
        let src = ImageSource {
            source_type: "base64".to_string(),
            media_type: "image/jpeg".to_string(),
            data: "abc123".to_string(),
        };
        let json = serde_json::to_value(&src).unwrap();
        assert_eq!(json["type"], "base64");
        assert_eq!(json["media_type"], "image/jpeg");
        assert_eq!(json["data"], "abc123");
    }

    #[test]
    fn test_image_source_deserialize() {
        let json = r#"{"type":"base64","media_type":"image/png","data":"xyz"}"#;
        let src: ImageSource = serde_json::from_str(json).unwrap();
        assert_eq!(src.source_type, "base64");
        assert_eq!(src.media_type, "image/png");
        assert_eq!(src.data, "xyz");
    }

    // --- ContentBlock::Image ---

    #[test]
    fn test_content_block_image_serialize() {
        let block = ContentBlock::Image {
            source: ImageSource {
                source_type: "base64".to_string(),
                media_type: "image/jpeg".to_string(),
                data: "abc".to_string(),
            },
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "image");
        assert_eq!(json["source"]["type"], "base64");
        assert_eq!(json["source"]["media_type"], "image/jpeg");
    }

    #[test]
    fn test_content_block_image_deserialize() {
        let json =
            r#"{"type":"image","source":{"type":"base64","media_type":"image/png","data":"xyz"}}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        match block {
            ContentBlock::Image { source } => {
                assert_eq!(source.source_type, "base64");
                assert_eq!(source.media_type, "image/png");
                assert_eq!(source.data, "xyz");
            }
            _ => panic!("Expected Image block"),
        }
    }

    // --- ToolResultContentField ---

    #[test]
    fn test_tool_result_content_field_text_serialize() {
        let field = ToolResultContentField::Text("hello".to_string());
        let json = serde_json::to_value(&field).unwrap();
        assert_eq!(json, "hello");
    }

    #[test]
    fn test_tool_result_content_field_blocks_serialize() {
        let field = ToolResultContentField::Blocks(vec![
            ToolResultContent::Text {
                text: "output".to_string(),
            },
            ToolResultContent::Image {
                source: ImageSource {
                    source_type: "base64".to_string(),
                    media_type: "image/png".to_string(),
                    data: "abc".to_string(),
                },
            },
        ]);
        let json = serde_json::to_value(&field).unwrap();
        assert!(json.is_array());
        assert_eq!(json[0]["type"], "text");
        assert_eq!(json[1]["type"], "image");
    }

    #[test]
    fn test_tool_result_content_field_text_deserialize() {
        let json = r#""hello""#;
        let field: ToolResultContentField = serde_json::from_str(json).unwrap();
        assert_eq!(field, "hello");
    }

    #[test]
    fn test_tool_result_content_field_blocks_deserialize() {
        let json = r#"[{"type":"text","text":"out"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"x"}}]"#;
        let field: ToolResultContentField = serde_json::from_str(json).unwrap();
        match field {
            ToolResultContentField::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
            }
            _ => panic!("Expected Blocks variant"),
        }
    }

    #[test]
    fn test_tool_result_content_field_from_str() {
        let field: ToolResultContentField = "test".into();
        assert_eq!(field, "test");
    }

    #[test]
    fn test_tool_result_content_field_from_string() {
        let field: ToolResultContentField = "test".to_string().into();
        assert_eq!(field, "test");
    }

    #[test]
    fn test_tool_result_content_field_as_text_plain() {
        let field = ToolResultContentField::Text("hello".to_string());
        assert_eq!(field.as_text(), "hello");
    }

    #[test]
    fn test_tool_result_content_field_as_text_blocks() {
        let field = ToolResultContentField::Blocks(vec![
            ToolResultContent::Text {
                text: "line1".to_string(),
            },
            ToolResultContent::Image {
                source: ImageSource {
                    source_type: "base64".to_string(),
                    media_type: "image/png".to_string(),
                    data: "x".to_string(),
                },
            },
            ToolResultContent::Text {
                text: "line2".to_string(),
            },
        ]);
        assert_eq!(field.as_text(), "line1\nline2");
    }

    #[test]
    fn test_tool_result_content_field_partial_eq() {
        let field = ToolResultContentField::Text("hello".to_string());
        assert!(field == "hello");
        assert!(field != "world");

        let blocks = ToolResultContentField::Blocks(vec![]);
        assert!(blocks != "hello");
    }

    // --- Message::user_with_attachments ---

    #[test]
    fn test_message_user_with_attachments() {
        let attachments = vec![Attachment::jpeg(vec![1, 2, 3])];
        let msg = Message::user_with_attachments("Describe this image", &attachments);
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content.len(), 2); // 1 image + 1 text
        match &msg.content[0] {
            ContentBlock::Image { source } => {
                assert_eq!(source.media_type, "image/jpeg");
            }
            _ => panic!("Expected Image block first"),
        }
        match &msg.content[1] {
            ContentBlock::Text { text } => {
                assert_eq!(text, "Describe this image");
            }
            _ => panic!("Expected Text block second"),
        }
    }

    #[test]
    fn test_message_user_with_multiple_attachments() {
        let attachments = vec![Attachment::jpeg(vec![1]), Attachment::png(vec![2])];
        let msg = Message::user_with_attachments("Compare", &attachments);
        assert_eq!(msg.content.len(), 3); // 2 images + 1 text
    }

    #[test]
    fn test_message_user_with_empty_attachments() {
        let msg = Message::user_with_attachments("No images", &[]);
        assert_eq!(msg.content.len(), 1); // just text
        assert_eq!(msg.text(), "No images");
    }

    // --- Message::tool_result_with_images ---

    #[test]
    fn test_message_tool_result_with_images() {
        let images = vec![Attachment::png(vec![1, 2])];
        let msg = Message::tool_result_with_images("t1", "screenshot taken", &images, false);
        assert_eq!(msg.role, "user");
        match &msg.content[0] {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                assert_eq!(tool_use_id, "t1");
                assert_eq!(*is_error, Some(false));
                match content {
                    ToolResultContentField::Blocks(blocks) => {
                        assert_eq!(blocks.len(), 2); // text + image
                        match &blocks[0] {
                            ToolResultContent::Text { text } => {
                                assert_eq!(text, "screenshot taken");
                            }
                            _ => panic!("Expected text block"),
                        }
                        match &blocks[1] {
                            ToolResultContent::Image { source } => {
                                assert_eq!(source.media_type, "image/png");
                            }
                            _ => panic!("Expected image block"),
                        }
                    }
                    _ => panic!("Expected Blocks variant"),
                }
            }
            _ => panic!("Expected ToolResult block"),
        }
    }

    // --- OpenAI image conversion ---

    #[test]
    fn test_openai_convert_messages_with_image() {
        let client = OpenAiClient::new("key".to_string(), "gpt-4o".to_string());
        let msgs = vec![Message {
            role: "user".to_string(),
            content: vec![
                ContentBlock::Image {
                    source: ImageSource {
                        source_type: "base64".to_string(),
                        media_type: "image/jpeg".to_string(),
                        data: "abc123".to_string(),
                    },
                },
                ContentBlock::Text {
                    text: "What is this?".to_string(),
                },
            ],
            reasoning_content: None,
        }];
        let converted = client.convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["role"], "user");
        let content = converted[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "image_url");
        assert!(content[0]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/jpeg;base64,"));
        assert_eq!(content[1]["type"], "text");
    }

    #[test]
    fn test_openai_convert_tool_result_with_multimodal_content() {
        let client = OpenAiClient::new("key".to_string(), "gpt-4o".to_string());
        let msgs = vec![Message {
            role: "user".to_string(),
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "call-1".to_string(),
                content: ToolResultContentField::Blocks(vec![
                    ToolResultContent::Text {
                        text: "screenshot".to_string(),
                    },
                    ToolResultContent::Image {
                        source: ImageSource {
                            source_type: "base64".to_string(),
                            media_type: "image/png".to_string(),
                            data: "xyz".to_string(),
                        },
                    },
                ]),
                is_error: Some(false),
            }],
            reasoning_content: None,
        }];
        let converted = client.convert_messages(&msgs);
        assert_eq!(converted[0]["role"], "tool");
        // OpenAI tool results only support text, so images are stripped
        assert_eq!(converted[0]["content"], "screenshot");
    }

    // --- Backward compatibility ---

    #[test]
    fn test_tool_result_backward_compat_serialization() {
        // Plain string tool result should serialize as before
        let msg = Message::tool_result("t1", "output text", false);
        let json = serde_json::to_value(&msg).unwrap();
        let content = &json["content"][0];
        assert_eq!(content["type"], "tool_result");
        assert_eq!(content["content"], "output text");
    }

    #[test]
    fn test_tool_result_backward_compat_deserialization() {
        // Old-format JSON with string content should still deserialize
        let json =
            r#"{"type":"tool_result","tool_use_id":"t1","content":"output","is_error":false}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        match block {
            ContentBlock::ToolResult { content, .. } => {
                assert_eq!(content, "output");
            }
            _ => panic!("Expected ToolResult"),
        }
    }

    #[test]
    fn test_tool_result_multimodal_deserialization() {
        // New-format JSON with array content
        let json = r#"{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"out"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"x"}}],"is_error":false}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        match block {
            ContentBlock::ToolResult { content, .. } => match content {
                ToolResultContentField::Blocks(blocks) => {
                    assert_eq!(blocks.len(), 2);
                }
                _ => panic!("Expected Blocks variant"),
            },
            _ => panic!("Expected ToolResult"),
        }
    }

    // glm5.1 / GLM reasoning models stream + return their chain-of-thought under
    // `reasoning`, not the `reasoning_content` kimi/deepseek use. The serde alias must
    // capture it; otherwise the whole reasoning phase yields zero recognized deltas and
    // the stream-stall watchdog kills long reasoning (asset-diagnose "未返回结构化输出").
    #[test]
    fn glm_reasoning_aliases_to_reasoning_content_in_streaming_delta() {
        let d: OpenAiDelta = serde_json::from_str(r#"{"reasoning":"thinking"}"#).unwrap();
        assert_eq!(d.reasoning_content.as_deref(), Some("thinking"));
        // the canonical field still works (kimi/deepseek)
        let d2: OpenAiDelta = serde_json::from_str(r#"{"reasoning_content":"rc"}"#).unwrap();
        assert_eq!(d2.reasoning_content.as_deref(), Some("rc"));
    }

    #[test]
    fn glm_reasoning_aliases_to_reasoning_content_in_message() {
        let m: OpenAiMessage =
            serde_json::from_str(r#"{"reasoning":"thinking","content":"answer"}"#).unwrap();
        assert_eq!(m.reasoning_content.as_deref(), Some("thinking"));
        assert_eq!(m.content.as_deref(), Some("answer"));
    }
}
