use super::*;
use crate::llm::{
    ContentBlock, LlmClient, LlmResponse, Message, StreamEvent, TokenUsage, ToolDefinition,
};
use async_trait::async_trait;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

struct MockStructuredClient {
    responses: Mutex<Vec<LlmResponse>>,
}

impl MockStructuredClient {
    fn new(responses: Vec<LlmResponse>) -> Self {
        Self {
            responses: Mutex::new(responses),
        }
    }

    fn text_response(text: &str) -> LlmResponse {
        LlmResponse {
            message: Message {
                role: "assistant".to_string(),
                content: vec![ContentBlock::Text {
                    text: text.to_string(),
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
            meta: None,
        }
    }

    fn tool_call_response(tool_name: &str, args: serde_json::Value) -> LlmResponse {
        LlmResponse {
            message: Message {
                role: "assistant".to_string(),
                content: vec![ContentBlock::ToolUse {
                    id: "call_001".to_string(),
                    name: tool_name.to_string(),
                    input: args,
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
            stop_reason: Some("tool_use".to_string()),
            meta: None,
        }
    }
}

#[async_trait]
impl LlmClient for MockStructuredClient {
    async fn complete(
        &self,
        _messages: &[Message],
        _system: Option<&str>,
        _tools: &[ToolDefinition],
    ) -> anyhow::Result<LlmResponse> {
        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            anyhow::bail!("No more mock responses");
        }
        Ok(responses.remove(0))
    }

    async fn complete_streaming(
        &self,
        _messages: &[Message],
        _system: Option<&str>,
        _tools: &[ToolDefinition],
        _cancel_token: CancellationToken,
    ) -> anyhow::Result<mpsc::Receiver<StreamEvent>> {
        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            anyhow::bail!("No more mock responses");
        }
        let response = responses.remove(0);
        let (tx, rx) = mpsc::channel(10);
        tokio::spawn(async move {
            for block in &response.message.content {
                if let ContentBlock::ToolUse { name, input, .. } = block {
                    tx.send(StreamEvent::ToolUseStart {
                        id: "call_001".to_string(),
                        name: name.clone(),
                    })
                    .await
                    .ok();
                    let json_str = serde_json::to_string(input).unwrap();
                    for chunk in json_str.as_bytes().chunks(10) {
                        let s = String::from_utf8_lossy(chunk).to_string();
                        tx.send(StreamEvent::ToolUseInputDelta(s)).await.ok();
                    }
                } else if let ContentBlock::Text { text } = block {
                    for chunk in text.as_bytes().chunks(10) {
                        let s = String::from_utf8_lossy(chunk).to_string();
                        tx.send(StreamEvent::TextDelta(s)).await.ok();
                    }
                }
            }
            tx.send(StreamEvent::Done(response)).await.ok();
        });
        Ok(rx)
    }
}

// ========================================================================
// extract_json_value tests
// ========================================================================

#[test]
fn test_extract_json_direct() {
    let input = r#"{"name": "Alice", "age": 30}"#;
    let result = extract_json_value(input).unwrap();
    assert_eq!(result["name"], "Alice");
    assert_eq!(result["age"], 30);
}

#[test]
fn test_extract_json_with_whitespace() {
    let input = "  \n  {\"x\": 1}  \n  ";
    let result = extract_json_value(input).unwrap();
    assert_eq!(result["x"], 1);
}

#[test]
fn test_extract_json_from_code_fence() {
    let input = "```json\n{\"key\": \"value\"}\n```";
    let result = extract_json_value(input).unwrap();
    assert_eq!(result["key"], "value");
}

#[test]
fn test_extract_json_from_code_fence_no_lang() {
    let input = "```\n{\"key\": \"value\"}\n```";
    let result = extract_json_value(input).unwrap();
    assert_eq!(result["key"], "value");
}

#[test]
fn test_extract_json_with_surrounding_prose() {
    let input = "Here is the result:\n{\"status\": \"ok\"}\nDone!";
    let result = extract_json_value(input).unwrap();
    assert_eq!(result["status"], "ok");
}

#[test]
fn test_extract_json_nested_braces() {
    let input = r#"Result: {"outer": {"inner": [1, 2, 3]}} end"#;
    let result = extract_json_value(input).unwrap();
    assert_eq!(result["outer"]["inner"][1], 2);
}

#[test]
fn test_extract_json_with_escaped_quotes() {
    let input = r#"{"msg": "he said \"hello\""}"#;
    let result = extract_json_value(input).unwrap();
    assert_eq!(result["msg"], "he said \"hello\"");
}

#[test]
fn test_extract_json_array() {
    let input = r#"[{"a": 1}, {"a": 2}]"#;
    let result = extract_json_value(input).unwrap();
    assert_eq!(result[0]["a"], 1);
}

#[test]
fn test_extract_json_no_json() {
    let input = "This is just plain text with no JSON.";
    assert!(extract_json_value(input).is_err());
}

#[test]
fn test_extract_json_braces_in_string() {
    let input = r#"prefix {"template": "Hello {name}!"} suffix"#;
    let result = extract_json_value(input).unwrap();
    assert_eq!(result["template"], "Hello {name}!");
}

// ========================================================================
// try_parse_partial_json tests
// ========================================================================

#[test]
fn test_partial_json_complete() {
    let input = r#"{"name": "Alice", "age": 30}"#;
    let result = try_parse_partial_json(input).unwrap();
    assert_eq!(result["name"], "Alice");
}

#[test]
fn test_partial_json_unclosed_object() {
    let input = r#"{"name": "Alice", "age": 30"#;
    let result = try_parse_partial_json(input).unwrap();
    assert_eq!(result["name"], "Alice");
    assert_eq!(result["age"], 30);
}

#[test]
fn test_partial_json_unclosed_array() {
    let input = r#"{"items": [1, 2, 3"#;
    let result = try_parse_partial_json(input).unwrap();
    assert_eq!(result["items"][0], 1);
    assert_eq!(result["items"][2], 3);
}

#[test]
fn test_partial_json_unclosed_string() {
    let input = r#"{"name": "Ali"#;
    let result = try_parse_partial_json(input).unwrap();
    assert_eq!(result["name"], "Ali");
}

#[test]
fn test_partial_json_trailing_comma() {
    let input = r#"{"a": 1,"#;
    let result = try_parse_partial_json(input).unwrap();
    assert_eq!(result["a"], 1);
}

#[test]
fn test_partial_json_key_no_value() {
    let input = r#"{"a": 1, "b":"#;
    let result = try_parse_partial_json(input).unwrap();
    assert_eq!(result["a"], 1);
}

#[test]
fn test_partial_json_empty() {
    assert!(try_parse_partial_json("").is_none());
}

#[test]
fn test_partial_json_just_open_brace() {
    let result = try_parse_partial_json("{");
    assert!(result.is_some());
    assert!(result.unwrap().is_object());
}

#[test]
fn test_partial_json_nested_unclosed() {
    let input = r#"{"user": {"name": "Bob", "tags": ["admin""#;
    let result = try_parse_partial_json(input).unwrap();
    assert_eq!(result["user"]["name"], "Bob");
    assert_eq!(result["user"]["tags"][0], "admin");
}

// ========================================================================
// Schema validation tests
// ========================================================================

#[test]
fn test_validate_simple_object() {
    let schema = serde_json::json!({
        "type": "object",
        "required": ["name", "age"],
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    });
    let value = serde_json::json!({"name": "Alice", "age": 30});
    assert!(validate_against_schema(&value, &schema).is_ok());
}

#[test]
fn test_validate_missing_required() {
    let schema = serde_json::json!({
        "type": "object",
        "required": ["name", "age"],
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    });
    let value = serde_json::json!({"name": "Alice"});
    let errors = validate_against_schema(&value, &schema).unwrap_err();
    assert!(errors.iter().any(|e| e.contains("age")));
}

#[test]
fn test_validate_wrong_type() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "count": {"type": "integer"}
        }
    });
    let value = serde_json::json!({"count": "not a number"});
    let errors = validate_against_schema(&value, &schema).unwrap_err();
    assert!(errors.iter().any(|e| e.contains("integer")));
}

#[test]
fn test_validate_array_items() {
    let schema = serde_json::json!({
        "type": "array",
        "items": {"type": "string"}
    });
    let value = serde_json::json!(["a", "b", "c"]);
    assert!(validate_against_schema(&value, &schema).is_ok());

    let bad_value = serde_json::json!(["a", 123, "c"]);
    assert!(validate_against_schema(&bad_value, &schema).is_err());
}

#[test]
fn test_validate_enum() {
    let schema = serde_json::json!({
        "type": "string",
        "enum": ["red", "green", "blue"]
    });
    let good = serde_json::json!("red");
    assert!(validate_against_schema(&good, &schema).is_ok());

    let bad = serde_json::json!("purple");
    assert!(validate_against_schema(&bad, &schema).is_err());
}

#[test]
fn test_validate_nested_object() {
    let schema = serde_json::json!({
        "type": "object",
        "required": ["user"],
        "properties": {
            "user": {
                "type": "object",
                "required": ["email"],
                "properties": {
                    "email": {"type": "string"}
                }
            }
        }
    });
    let good = serde_json::json!({"user": {"email": "a@b.com"}});
    assert!(validate_against_schema(&good, &schema).is_ok());

    let bad = serde_json::json!({"user": {}});
    let errors = validate_against_schema(&bad, &schema).unwrap_err();
    assert!(errors.iter().any(|e| e.contains("email")));
}

// ========================================================================
// generate_blocking tests
// ========================================================================

#[tokio::test]
async fn test_generate_blocking_tool_mode_success() {
    let client = MockStructuredClient::new(vec![MockStructuredClient::tool_call_response(
        "emit_invoice",
        serde_json::json!({"amount": 100, "currency": "USD"}),
    )]);

    let req = StructuredRequest {
        prompt: "Extract invoice".to_string(),
        system: None,
        schema: serde_json::json!({
            "type": "object",
            "required": ["amount", "currency"],
            "properties": {
                "amount": {"type": "number"},
                "currency": {"type": "string"}
            }
        }),
        schema_name: "invoice".to_string(),
        schema_description: None,
        mode: StructuredMode::Tool,
        max_repair_attempts: 2,
    };

    let result = generate_blocking(&client, &req).await.unwrap();
    assert_eq!(result.object["amount"], 100);
    assert_eq!(result.object["currency"], "USD");
    assert_eq!(result.repair_rounds, 0);
}

#[tokio::test]
async fn test_generate_blocking_prompt_mode_success() {
    let client = MockStructuredClient::new(vec![MockStructuredClient::text_response(
        r#"{"amount": 50, "currency": "EUR"}"#,
    )]);

    let req = StructuredRequest {
        prompt: "Extract invoice".to_string(),
        system: None,
        schema: serde_json::json!({
            "type": "object",
            "required": ["amount", "currency"],
            "properties": {
                "amount": {"type": "number"},
                "currency": {"type": "string"}
            }
        }),
        schema_name: "invoice".to_string(),
        schema_description: None,
        mode: StructuredMode::Prompt,
        max_repair_attempts: 2,
    };

    let result = generate_blocking(&client, &req).await.unwrap();
    assert_eq!(result.object["amount"], 50);
    assert_eq!(result.object["currency"], "EUR");
}

#[tokio::test]
async fn test_generate_blocking_repair_on_invalid_json() {
    let client = MockStructuredClient::new(vec![
        MockStructuredClient::text_response("not json at all"),
        MockStructuredClient::text_response(r#"{"name": "fixed"}"#),
    ]);

    let req = StructuredRequest {
        prompt: "Generate".to_string(),
        system: None,
        schema: serde_json::json!({
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}}
        }),
        schema_name: "result".to_string(),
        schema_description: None,
        mode: StructuredMode::Prompt,
        max_repair_attempts: 2,
    };

    let result = generate_blocking(&client, &req).await.unwrap();
    assert_eq!(result.object["name"], "fixed");
    assert_eq!(result.repair_rounds, 1);
}

#[tokio::test]
async fn test_generate_blocking_repair_on_schema_violation() {
    let client = MockStructuredClient::new(vec![
        MockStructuredClient::text_response(r#"{"name": "Alice"}"#),
        MockStructuredClient::text_response(r#"{"name": "Alice", "age": 25}"#),
    ]);

    let req = StructuredRequest {
        prompt: "Generate".to_string(),
        system: None,
        schema: serde_json::json!({
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }),
        schema_name: "person".to_string(),
        schema_description: None,
        mode: StructuredMode::Prompt,
        max_repair_attempts: 2,
    };

    let result = generate_blocking(&client, &req).await.unwrap();
    assert_eq!(result.object["age"], 25);
    assert_eq!(result.repair_rounds, 1);
}

#[tokio::test]
async fn test_generate_blocking_exhausts_repairs() {
    let client = MockStructuredClient::new(vec![
        MockStructuredClient::text_response("bad1"),
        MockStructuredClient::text_response("bad2"),
        MockStructuredClient::text_response("bad3"),
    ]);

    let req = StructuredRequest {
        prompt: "Generate".to_string(),
        system: None,
        schema: serde_json::json!({
            "type": "object",
            "required": ["x"],
            "properties": {"x": {"type": "string"}}
        }),
        schema_name: "r".to_string(),
        schema_description: None,
        mode: StructuredMode::Prompt,
        max_repair_attempts: 2,
    };

    let err = generate_blocking(&client, &req).await.unwrap_err();
    assert!(err.to_string().contains("repair attempts"));
}

// ========================================================================
// generate_streaming tests
// ========================================================================

#[tokio::test]
async fn test_generate_streaming_tool_mode() {
    let client = MockStructuredClient::new(vec![MockStructuredClient::tool_call_response(
        "emit_result",
        serde_json::json!({"items": ["a", "b", "c"]}),
    )]);

    let req = StructuredRequest {
        prompt: "List items".to_string(),
        system: None,
        schema: serde_json::json!({
            "type": "object",
            "required": ["items"],
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}}
            }
        }),
        schema_name: "result".to_string(),
        schema_description: None,
        mode: StructuredMode::Tool,
        max_repair_attempts: 0,
    };

    let partials = Arc::new(Mutex::new(Vec::new()));
    let partials_clone = Arc::clone(&partials);
    let callback: PartialObjectCallback = Box::new(move |v: &Value| {
        partials_clone.lock().unwrap().push(v.clone());
    });

    let result = generate_streaming(&client, &req, callback).await.unwrap();
    assert_eq!(result.object["items"][0], "a");
    assert_eq!(result.object["items"][2], "c");

    let p = partials.lock().unwrap();
    assert!(!p.is_empty());
}

// ========================================================================
// Edge case: extract_json_value
// ========================================================================

#[test]
fn test_extract_json_unicode() {
    let input = r#"{"name": "日本語テスト", "emoji": "🎉"}"#;
    let result = extract_json_value(input).unwrap();
    assert_eq!(result["name"], "日本語テスト");
    assert_eq!(result["emoji"], "🎉");
}

#[test]
fn test_extract_json_deeply_nested() {
    let input = r#"{"a":{"b":{"c":{"d":{"e":"deep"}}}}}"#;
    let result = extract_json_value(input).unwrap();
    assert_eq!(result["a"]["b"]["c"]["d"]["e"], "deep");
}

#[test]
fn test_extract_json_empty_object() {
    let input = "{}";
    let result = extract_json_value(input).unwrap();
    assert!(result.as_object().unwrap().is_empty());
}

#[test]
fn test_extract_json_empty_array() {
    let input = "[]";
    let result = extract_json_value(input).unwrap();
    assert!(result.as_array().unwrap().is_empty());
}

#[test]
fn test_extract_json_multiple_objects_takes_first() {
    let input = r#"first: {"a": 1} second: {"b": 2}"#;
    let result = extract_json_value(input).unwrap();
    assert_eq!(result["a"], 1);
}

#[test]
fn test_extract_json_code_fence_with_trailing_whitespace() {
    let input = "```json\n{\"x\": 1}\n```\n\n";
    let result = extract_json_value(input).unwrap();
    assert_eq!(result["x"], 1);
}

#[test]
fn test_extract_json_code_fence_uppercase() {
    // Some models output ```JSON instead of ```json
    let input = "```JSON\n{\"x\": 1}\n```";
    // This won't match our fence stripper, but find_balanced will catch it
    let result = extract_json_value(input).unwrap();
    assert_eq!(result["x"], 1);
}

#[test]
fn test_extract_json_with_newlines_in_string() {
    let input = r#"{"text": "line1\nline2\nline3"}"#;
    let result = extract_json_value(input).unwrap();
    // \n in JSON becomes actual newline when parsed
    assert!(result["text"].as_str().unwrap().contains('\n'));
}

#[test]
fn test_extract_json_number_values() {
    let input = r#"{"int": 42, "float": 3.14, "neg": -1, "exp": 1e10}"#;
    let result = extract_json_value(input).unwrap();
    assert_eq!(result["int"], 42);
    assert_eq!(result["neg"], -1);
}

#[test]
fn test_extract_json_boolean_null() {
    let input = r#"{"flag": true, "other": false, "empty": null}"#;
    let result = extract_json_value(input).unwrap();
    assert_eq!(result["flag"], true);
    assert_eq!(result["other"], false);
    assert!(result["empty"].is_null());
}

#[test]
fn test_extract_json_scalar_rejected() {
    // A bare string/number is not a valid structured output
    assert!(extract_json_value(r#""just a string""#).is_err());
    assert!(extract_json_value("42").is_err());
    assert!(extract_json_value("true").is_err());
}

// ========================================================================
// Edge case: partial JSON parser
// ========================================================================

#[test]
fn test_partial_json_deeply_nested_unclosed() {
    let input = r#"{"a": {"b": {"c": [1, 2"#;
    let result = try_parse_partial_json(input).unwrap();
    assert_eq!(result["a"]["b"]["c"][0], 1);
    assert_eq!(result["a"]["b"]["c"][1], 2);
}

#[test]
fn test_partial_json_boolean_mid_value() {
    // "tru" is not valid, but we have an unclosed brace
    let input = r#"{"done": tru"#;
    // This will fail to parse even after closing — that's OK, returns None
    let result = try_parse_partial_json(input);
    // "tru}" is not valid JSON, so None is correct
    assert!(result.is_none());
}

#[test]
fn test_partial_json_number_mid_value() {
    let input = r#"{"count": 12"#;
    let result = try_parse_partial_json(input).unwrap();
    assert_eq!(result["count"], 12);
}

#[test]
fn test_partial_json_multiple_keys() {
    let input = r#"{"a": "hello", "b": 42, "c": [1, 2, 3], "d":"#;
    let result = try_parse_partial_json(input).unwrap();
    assert_eq!(result["a"], "hello");
    assert_eq!(result["b"], 42);
    assert_eq!(result["c"][2], 3);
    assert!(result["d"].is_null()); // colon with no value → null
}

#[test]
fn test_partial_json_escaped_quotes_in_string() {
    let input = r#"{"msg": "he said \"hi\""#;
    let result = try_parse_partial_json(input).unwrap();
    assert_eq!(result["msg"], "he said \"hi\"");
}

#[test]
fn test_partial_json_unicode_in_string() {
    let input = r#"{"name": "日本"#;
    let result = try_parse_partial_json(input).unwrap();
    assert_eq!(result["name"], "日本");
}

#[test]
fn test_partial_json_empty_array_unclosed() {
    let input = r#"{"items": ["#;
    let result = try_parse_partial_json(input).unwrap();
    assert!(result["items"].as_array().unwrap().is_empty());
}

#[test]
fn test_partial_json_nested_objects_and_arrays() {
    let input = r#"{"users": [{"name": "Alice", "tags": ["admin", "user"#;
    let result = try_parse_partial_json(input).unwrap();
    assert_eq!(result["users"][0]["name"], "Alice");
    assert_eq!(result["users"][0]["tags"][0], "admin");
    assert_eq!(result["users"][0]["tags"][1], "user");
}

// ========================================================================
// Edge case: schema validation
// ========================================================================

#[test]
fn test_validate_nullable_type() {
    let schema = serde_json::json!({
        "type": ["string", "null"]
    });
    assert!(validate_against_schema(&serde_json::json!("hello"), &schema).is_ok());
    assert!(validate_against_schema(&serde_json::json!(null), &schema).is_ok());
    assert!(validate_against_schema(&serde_json::json!(42), &schema).is_err());
}

#[test]
fn test_validate_any_of() {
    let schema = serde_json::json!({
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    });
    assert!(validate_against_schema(&serde_json::json!("hello"), &schema).is_ok());
    assert!(validate_against_schema(&serde_json::json!(42), &schema).is_ok());
    assert!(validate_against_schema(&serde_json::json!(true), &schema).is_err());
}

#[test]
fn test_validate_one_of() {
    let schema = serde_json::json!({
        "oneOf": [
            {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}},
            {"type": "object", "required": ["id"], "properties": {"id": {"type": "integer"}}}
        ]
    });
    assert!(validate_against_schema(&serde_json::json!({"name": "Alice"}), &schema).is_ok());
    assert!(validate_against_schema(&serde_json::json!({"id": 1}), &schema).is_ok());
    assert!(validate_against_schema(&serde_json::json!({"other": true}), &schema).is_err());
}

#[test]
fn test_validate_additional_properties_false() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "additionalProperties": false
    });
    assert!(validate_against_schema(&serde_json::json!({"name": "ok"}), &schema).is_ok());
    let errors =
        validate_against_schema(&serde_json::json!({"name": "ok", "extra": true}), &schema)
            .unwrap_err();
    assert!(errors.iter().any(|e| e.contains("extra")));
}

#[test]
fn test_validate_number_range() {
    let schema = serde_json::json!({
        "type": "number",
        "minimum": 0,
        "maximum": 100
    });
    assert!(validate_against_schema(&serde_json::json!(50), &schema).is_ok());
    assert!(validate_against_schema(&serde_json::json!(0), &schema).is_ok());
    assert!(validate_against_schema(&serde_json::json!(100), &schema).is_ok());
    assert!(validate_against_schema(&serde_json::json!(-1), &schema).is_err());
    assert!(validate_against_schema(&serde_json::json!(101), &schema).is_err());
}

#[test]
fn test_validate_exclusive_range() {
    let schema = serde_json::json!({
        "type": "number",
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 10
    });
    assert!(validate_against_schema(&serde_json::json!(5), &schema).is_ok());
    assert!(validate_against_schema(&serde_json::json!(0), &schema).is_err());
    assert!(validate_against_schema(&serde_json::json!(10), &schema).is_err());
}

#[test]
fn test_validate_string_pattern() {
    let schema = serde_json::json!({
        "type": "string",
        "pattern": "^[a-z]+$"
    });
    assert!(validate_against_schema(&serde_json::json!("hello"), &schema).is_ok());
    assert!(validate_against_schema(&serde_json::json!("Hello"), &schema).is_err());
    assert!(validate_against_schema(&serde_json::json!("123"), &schema).is_err());
}

#[test]
fn test_validate_string_length_unicode() {
    let schema = serde_json::json!({
        "type": "string",
        "minLength": 2,
        "maxLength": 5
    });
    // Unicode chars count as 1 each (char count, not byte count)
    assert!(validate_against_schema(&serde_json::json!("日本"), &schema).is_ok());
    assert!(validate_against_schema(&serde_json::json!("a"), &schema).is_err());
    assert!(validate_against_schema(&serde_json::json!("abcdef"), &schema).is_err());
}

#[test]
fn test_validate_integer_as_float() {
    let schema = serde_json::json!({"type": "integer"});
    // 30.0 should pass as integer
    assert!(validate_against_schema(&serde_json::json!(30.0), &schema).is_ok());
    // 30.5 should fail
    assert!(validate_against_schema(&serde_json::json!(30.5), &schema).is_err());
}

#[test]
fn test_validate_min_max_items() {
    let schema = serde_json::json!({
        "type": "array",
        "minItems": 2,
        "maxItems": 4,
        "items": {"type": "number"}
    });
    assert!(validate_against_schema(&serde_json::json!([1, 2]), &schema).is_ok());
    assert!(validate_against_schema(&serde_json::json!([1, 2, 3, 4]), &schema).is_ok());
    assert!(validate_against_schema(&serde_json::json!([1]), &schema).is_err());
    assert!(validate_against_schema(&serde_json::json!([1, 2, 3, 4, 5]), &schema).is_err());
}

#[test]
fn test_validate_nested_required_with_optional() {
    let schema = serde_json::json!({
        "type": "object",
        "required": ["id"],
        "properties": {
            "id": {"type": "integer"},
            "metadata": {
                "type": "object",
                "required": ["version"],
                "properties": {
                    "version": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}}
                }
            }
        }
    });
    // id present, metadata absent (not required) → ok
    assert!(validate_against_schema(&serde_json::json!({"id": 1}), &schema).is_ok());
    // id present, metadata present but missing version → error
    let errors = validate_against_schema(&serde_json::json!({"id": 1, "metadata": {}}), &schema)
        .unwrap_err();
    assert!(errors.iter().any(|e| e.contains("version")));
}

// ========================================================================
// Edge case: generate_blocking
// ========================================================================

#[tokio::test]
async fn test_generate_blocking_tool_mode_model_returns_text_instead() {
    // Model ignores tool instruction and returns text — should still extract JSON
    let client = MockStructuredClient::new(vec![MockStructuredClient::text_response(
        r#"{"fallback": true}"#,
    )]);

    let req = StructuredRequest {
        prompt: "test".to_string(),
        system: None,
        schema: serde_json::json!({
            "type": "object",
            "properties": {"fallback": {"type": "boolean"}}
        }),
        schema_name: "test".to_string(),
        schema_description: None,
        mode: StructuredMode::Tool,
        max_repair_attempts: 0,
    };

    let result = generate_blocking(&client, &req).await.unwrap();
    assert_eq!(result.object["fallback"], true);
}

#[tokio::test]
async fn test_generate_blocking_zero_repair_attempts() {
    let client = MockStructuredClient::new(vec![MockStructuredClient::text_response("bad")]);

    let req = StructuredRequest {
        prompt: "test".to_string(),
        system: None,
        schema: serde_json::json!({"type": "object", "required": ["x"], "properties": {"x": {"type": "string"}}}),
        schema_name: "r".to_string(),
        schema_description: None,
        mode: StructuredMode::Prompt,
        max_repair_attempts: 0,
    };

    let err = generate_blocking(&client, &req).await.unwrap_err();
    assert!(err.to_string().contains("parsing"));
}

#[tokio::test]
async fn test_generate_blocking_code_fence_output() {
    let client = MockStructuredClient::new(vec![MockStructuredClient::text_response(
        "```json\n{\"status\": \"ok\"}\n```",
    )]);

    let req = StructuredRequest {
        prompt: "test".to_string(),
        system: None,
        schema: serde_json::json!({
            "type": "object",
            "required": ["status"],
            "properties": {"status": {"type": "string"}}
        }),
        schema_name: "r".to_string(),
        schema_description: None,
        mode: StructuredMode::Prompt,
        max_repair_attempts: 0,
    };

    let result = generate_blocking(&client, &req).await.unwrap();
    assert_eq!(result.object["status"], "ok");
}

#[tokio::test]
async fn test_generate_blocking_repair_in_tool_mode() {
    // First call: tool call with missing required field
    // Second call: tool call with complete object
    let client = MockStructuredClient::new(vec![
        MockStructuredClient::tool_call_response("emit_person", serde_json::json!({"name": "Bob"})),
        MockStructuredClient::tool_call_response(
            "emit_person",
            serde_json::json!({"name": "Bob", "age": 30}),
        ),
    ]);

    let req = StructuredRequest {
        prompt: "test".to_string(),
        system: None,
        schema: serde_json::json!({
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }),
        schema_name: "person".to_string(),
        schema_description: None,
        mode: StructuredMode::Tool,
        max_repair_attempts: 2,
    };

    let result = generate_blocking(&client, &req).await.unwrap();
    assert_eq!(result.object["name"], "Bob");
    assert_eq!(result.object["age"], 30);
    assert_eq!(result.repair_rounds, 1);
}

// ========================================================================
// Edge case: streaming
// ========================================================================

#[tokio::test]
async fn test_generate_streaming_text_mode() {
    let client = MockStructuredClient::new(vec![MockStructuredClient::text_response(
        r#"{"color": "blue", "count": 3}"#,
    )]);

    let req = StructuredRequest {
        prompt: "test".to_string(),
        system: None,
        schema: serde_json::json!({
            "type": "object",
            "required": ["color", "count"],
            "properties": {
                "color": {"type": "string"},
                "count": {"type": "integer"}
            }
        }),
        schema_name: "result".to_string(),
        schema_description: None,
        mode: StructuredMode::Prompt,
        max_repair_attempts: 0,
    };

    let partials = Arc::new(Mutex::new(Vec::new()));
    let partials_clone = Arc::clone(&partials);
    let callback: PartialObjectCallback = Box::new(move |v: &Value| {
        partials_clone.lock().unwrap().push(v.clone());
    });

    let result = generate_streaming(&client, &req, callback).await.unwrap();
    assert_eq!(result.object["color"], "blue");
    assert_eq!(result.object["count"], 3);
}

#[tokio::test]
async fn test_generate_streaming_schema_validation_failure() {
    let client = MockStructuredClient::new(vec![MockStructuredClient::tool_call_response(
        "emit_result",
        serde_json::json!({"wrong_field": true}),
    )]);

    let req = StructuredRequest {
        prompt: "test".to_string(),
        system: None,
        schema: serde_json::json!({
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}}
        }),
        schema_name: "result".to_string(),
        schema_description: None,
        mode: StructuredMode::Tool,
        max_repair_attempts: 0,
    };

    let callback: PartialObjectCallback = Box::new(|_| {});
    let err = generate_streaming(&client, &req, callback)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("schema validation"));
}

// ========================================================================
// Edge case: find_balanced with tricky strings
// ========================================================================

#[test]
fn test_find_balanced_brace_inside_string() {
    let input = r#"text "contains {braces}" then {"real": true} end"#;
    let result = extract_json_value(input).unwrap();
    assert_eq!(result["real"], true);
}

#[test]
fn test_find_balanced_escaped_quote_before_brace() {
    let input = r#"prefix "escaped \" quote" {"key": "val"} suffix"#;
    let result = extract_json_value(input).unwrap();
    assert_eq!(result["key"], "val");
}

#[test]
fn test_find_balanced_nested_strings_with_braces() {
    let input = r#"{"template": "Hello {name}!", "data": {"name": "World"}}"#;
    let result = extract_json_value(input).unwrap();
    assert_eq!(result["template"], "Hello {name}!");
    assert_eq!(result["data"]["name"], "World");
}

// ========================================================================
// Integration tests (require real LLM API)
// Run with: cargo test -p a3s-code-core --lib structured_tests::test_integration -- --ignored
// ========================================================================

fn load_minimax_config() -> Option<(String, String, String)> {
    // CARGO_MANIFEST_DIR = .../crates/code/core
    // config.acl is at repo root: .../a3s/.a3s/config.acl
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("../../../.a3s/config.acl"), // crates/code/core → repo root
        manifest_dir.join("../../.a3s/config.acl"),
        manifest_dir.join("../.a3s/config.acl"),
    ];

    let config_path = candidates.iter().find(|p| p.exists())?;
    let content = std::fs::read_to_string(config_path).ok()?;

    // Parse ACL to extract openai provider with MiniMax model
    let api_key = extract_acl_field(&content, "providers \"openai\"", "apiKey")?;
    let base_url = extract_acl_field(&content, "providers \"openai\"", "baseUrl")?;
    Some((base_url, api_key, "MiniMax-M2.7-highspeed".to_string()))
}

fn extract_acl_field(content: &str, section: &str, field: &str) -> Option<String> {
    let section_start = content.find(section)?;
    let section_content = &content[section_start..];
    // Find the field within the section (before next top-level block)
    let field_pattern = format!("{} = \"", field);
    let field_start = section_content.find(&field_pattern)?;
    let value_start = field_start + field_pattern.len();
    let value_end = section_content[value_start..].find('"')?;
    Some(section_content[value_start..value_start + value_end].to_string())
}

#[tokio::test]
#[ignore]
async fn test_integration_generate_blocking_tool_mode() {
    let Some((base_url, api_key, model)) = load_minimax_config() else {
        eprintln!("Skipping: .a3s/config.acl not found or missing MiniMax config");
        return;
    };

    let config = crate::llm::LlmConfig::new("openai", &model, &api_key)
        .with_base_url(&base_url)
        .with_temperature(0.0);
    let client = crate::llm::factory::create_client_with_config(config);

    let req = StructuredRequest {
        prompt: "Extract the person's information: John Smith is 35 years old and works as a software engineer.".to_string(),
        system: None,
        schema: serde_json::json!({
            "type": "object",
            "required": ["name", "age", "occupation"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "occupation": {"type": "string"}
            }
        }),
        schema_name: "person".to_string(),
        schema_description: Some("A person's basic information".to_string()),
        mode: StructuredMode::Tool,
        max_repair_attempts: 2,
    };

    let result = generate_blocking(&*client, &req).await;
    assert!(
        result.is_ok(),
        "generate_blocking failed: {:?}",
        result.err()
    );

    let result = result.unwrap();
    assert_eq!(
        result.object["name"].as_str().unwrap().to_lowercase(),
        "john smith"
    );
    assert_eq!(result.object["age"], 35);
    assert!(result.object["occupation"]
        .as_str()
        .unwrap()
        .to_lowercase()
        .contains("software"));
    eprintln!(
        "Integration test passed: mode={:?}, repairs={}, tokens={}",
        result.mode_used, result.repair_rounds, result.usage.total_tokens
    );
}

#[tokio::test]
#[ignore]
async fn test_integration_generate_blocking_prompt_mode() {
    let Some((base_url, api_key, model)) = load_minimax_config() else {
        eprintln!("Skipping: .a3s/config.acl not found");
        return;
    };

    let config = crate::llm::LlmConfig::new("openai", &model, &api_key)
        .with_base_url(&base_url)
        .with_temperature(0.0);
    let client = crate::llm::factory::create_client_with_config(config);

    let req = StructuredRequest {
        prompt: "Classify the sentiment of this text: 'I absolutely love this product, it changed my life!' Return the classification.".to_string(),
        system: None,
        schema: serde_json::json!({
            "type": "object",
            "required": ["sentiment", "confidence"],
            "properties": {
                "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            }
        }),
        schema_name: "classification".to_string(),
        schema_description: None,
        mode: StructuredMode::Prompt,
        max_repair_attempts: 2,
    };

    let result = generate_blocking(&*client, &req).await;
    assert!(
        result.is_ok(),
        "generate_blocking failed: {:?}",
        result.err()
    );

    let result = result.unwrap();
    assert_eq!(result.object["sentiment"], "positive");
    let confidence = result.object["confidence"].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&confidence));
    eprintln!(
        "Integration test passed: sentiment={}, confidence={}, repairs={}",
        result.object["sentiment"], confidence, result.repair_rounds
    );
}

#[tokio::test]
#[ignore]
async fn test_integration_generate_streaming_tool_mode() {
    let Some((base_url, api_key, model)) = load_minimax_config() else {
        eprintln!("Skipping: .a3s/config.acl not found");
        return;
    };

    let config = crate::llm::LlmConfig::new("openai", &model, &api_key)
        .with_base_url(&base_url)
        .with_temperature(0.0);
    let client = crate::llm::factory::create_client_with_config(config);

    let req = StructuredRequest {
        prompt: "List 3 programming languages with their year of creation.".to_string(),
        system: None,
        schema: serde_json::json!({
            "type": "object",
            "required": ["languages"],
            "properties": {
                "languages": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "required": ["name", "year"],
                        "properties": {
                            "name": {"type": "string"},
                            "year": {"type": "integer", "minimum": 1950, "maximum": 2030}
                        }
                    }
                }
            }
        }),
        schema_name: "languages".to_string(),
        schema_description: Some("A list of programming languages".to_string()),
        mode: StructuredMode::Tool,
        max_repair_attempts: 0,
    };

    let partials = Arc::new(Mutex::new(Vec::new()));
    let partials_clone = Arc::clone(&partials);
    let callback: PartialObjectCallback = Box::new(move |v: &Value| {
        partials_clone.lock().unwrap().push(v.clone());
    });

    let result = generate_streaming(&*client, &req, callback).await;
    assert!(
        result.is_ok(),
        "generate_streaming failed: {:?}",
        result.err()
    );

    let result = result.unwrap();
    let languages = result.object["languages"].as_array().unwrap();
    assert_eq!(languages.len(), 3);
    for lang in languages {
        assert!(lang["name"].is_string());
        let year = lang["year"].as_i64().unwrap();
        assert!((1950..=2030).contains(&year));
    }

    let partial_count = partials.lock().unwrap().len();
    eprintln!(
        "Integration test passed: {} languages, {} partial updates, tokens={}",
        languages.len(),
        partial_count,
        result.usage.total_tokens
    );
    // Should have received at least a few partial updates during streaming
    assert!(
        partial_count >= 1,
        "Expected partial updates during streaming, got 0"
    );
}

#[tokio::test]
#[ignore]
async fn test_integration_generate_complex_nested_schema() {
    let Some((base_url, api_key, model)) = load_minimax_config() else {
        eprintln!("Skipping: .a3s/config.acl not found");
        return;
    };

    let config = crate::llm::LlmConfig::new("openai", &model, &api_key)
        .with_base_url(&base_url)
        .with_temperature(0.0);
    let client = crate::llm::factory::create_client_with_config(config);

    let req = StructuredRequest {
        prompt: "Generate a sample API error response for a 404 not found error when trying to access user with ID 123.".to_string(),
        system: None,
        schema: serde_json::json!({
            "type": "object",
            "required": ["error"],
            "properties": {
                "error": {
                    "type": "object",
                    "required": ["code", "message", "details"],
                    "properties": {
                        "code": {"type": "integer"},
                        "message": {"type": "string", "minLength": 5},
                        "details": {
                            "type": "object",
                            "required": ["resource", "id"],
                            "properties": {
                                "resource": {"type": "string"},
                                "id": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }),
        schema_name: "api_error".to_string(),
        schema_description: None,
        mode: StructuredMode::Tool,
        max_repair_attempts: 2,
    };

    let result = generate_blocking(&*client, &req).await;
    assert!(
        result.is_ok(),
        "generate_blocking failed: {:?}",
        result.err()
    );

    let result = result.unwrap();
    let error = &result.object["error"];
    assert_eq!(error["code"], 404);
    assert!(error["message"].as_str().unwrap().len() >= 5);
    assert!(error["details"]["resource"].is_string());
    assert!(error["details"]["id"].as_str().unwrap().contains("123"));
    eprintln!(
        "Integration test passed: error.code={}, repairs={}",
        error["code"], result.repair_rounds
    );
}

// ========================================================================
// Native structured-output routing (capability + directive) tests
//
// These verify the core stability fix: the structured engine resolves the
// mode against the client's native capability and passes the correct
// directive (forced tool_choice vs. response_format) to complete_structured.
// ========================================================================

/// A client that records the directive + tool surface it was asked to honor,
/// and reports a configurable native capability.
struct RecordingClient {
    support: NativeStructuredSupport,
    responses: Mutex<Vec<LlmResponse>>,
    last_directive: Mutex<Option<StructuredDirective>>,
    last_tool_names: Mutex<Vec<String>>,
    structured_calls: std::sync::atomic::AtomicUsize,
}

impl RecordingClient {
    fn new(support: NativeStructuredSupport, responses: Vec<LlmResponse>) -> Self {
        Self {
            support,
            responses: Mutex::new(responses),
            last_directive: Mutex::new(None),
            last_tool_names: Mutex::new(Vec::new()),
            structured_calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn pop(&self) -> anyhow::Result<LlmResponse> {
        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            anyhow::bail!("No more mock responses");
        }
        Ok(responses.remove(0))
    }
}

#[async_trait]
impl LlmClient for RecordingClient {
    async fn complete(
        &self,
        _messages: &[Message],
        _system: Option<&str>,
        _tools: &[ToolDefinition],
    ) -> anyhow::Result<LlmResponse> {
        self.pop()
    }

    async fn complete_streaming(
        &self,
        _messages: &[Message],
        _system: Option<&str>,
        _tools: &[ToolDefinition],
        _cancel_token: CancellationToken,
    ) -> anyhow::Result<mpsc::Receiver<StreamEvent>> {
        anyhow::bail!("streaming not used in routing tests")
    }

    fn native_structured_support(&self) -> NativeStructuredSupport {
        self.support
    }

    async fn complete_structured(
        &self,
        _messages: &[Message],
        _system: Option<&str>,
        tools: &[ToolDefinition],
        directive: &StructuredDirective,
    ) -> anyhow::Result<LlmResponse> {
        self.structured_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        *self.last_directive.lock().unwrap() = Some(directive.clone());
        *self.last_tool_names.lock().unwrap() = tools.iter().map(|t| t.name.clone()).collect();
        self.pop()
    }
}

fn person_request(mode: StructuredMode) -> StructuredRequest {
    StructuredRequest {
        prompt: "Extract the person".to_string(),
        system: None,
        schema: serde_json::json!({
            "type": "object",
            "required": ["name"],
            "properties": { "name": { "type": "string" } }
        }),
        schema_name: "person".to_string(),
        schema_description: None,
        mode,
        max_repair_attempts: 0,
    }
}

#[tokio::test]
async fn test_routing_tool_mode_forces_tool_choice() {
    let client = RecordingClient::new(
        NativeStructuredSupport::ForcedTool,
        vec![MockStructuredClient::tool_call_response(
            "emit_person",
            serde_json::json!({ "name": "Bob" }),
        )],
    );
    let result = generate_blocking(&client, &person_request(StructuredMode::Tool))
        .await
        .unwrap();

    assert_eq!(result.object["name"], "Bob");
    assert_eq!(result.mode_used, StructuredMode::Tool);
    assert_eq!(
        client
            .structured_calls
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );

    let directive = client.last_directive.lock().unwrap().clone().unwrap();
    assert_eq!(directive.force_tool.as_deref(), Some("emit_person"));
    assert!(directive.response_format.is_none());
    // The synthetic emit tool must be present in the tool surface.
    assert_eq!(
        client.last_tool_names.lock().unwrap().as_slice(),
        &["emit_person".to_string()]
    );
}

#[tokio::test]
async fn test_routing_auto_collapses_to_forced_tool() {
    let client = RecordingClient::new(
        NativeStructuredSupport::JsonSchema,
        vec![MockStructuredClient::tool_call_response(
            "emit_person",
            serde_json::json!({ "name": "Bob" }),
        )],
    );
    let result = generate_blocking(&client, &person_request(StructuredMode::Auto))
        .await
        .unwrap();

    assert_eq!(result.mode_used, StructuredMode::Tool);
    let directive = client.last_directive.lock().unwrap().clone().unwrap();
    assert_eq!(directive.force_tool.as_deref(), Some("emit_person"));
}

#[tokio::test]
async fn test_routing_strict_uses_response_format_when_supported() {
    let client = RecordingClient::new(
        NativeStructuredSupport::JsonSchema,
        vec![MockStructuredClient::text_response(r#"{"name": "Bob"}"#)],
    );
    let result = generate_blocking(&client, &person_request(StructuredMode::Strict))
        .await
        .unwrap();

    assert_eq!(result.object["name"], "Bob");
    assert_eq!(result.mode_used, StructuredMode::Strict);

    let directive = client.last_directive.lock().unwrap().clone().unwrap();
    assert!(directive.force_tool.is_none());
    match directive.response_format {
        Some(ResponseFormat::JsonSchema {
            ref name,
            ref schema,
        }) => {
            assert_eq!(name, "person");
            assert_eq!(schema["required"][0], "name");
        }
        other => panic!("expected json_schema response_format, got {:?}", other),
    }
    // Strict mode must not inject a tool.
    assert!(client.last_tool_names.lock().unwrap().is_empty());
}

#[tokio::test]
async fn test_routing_strict_falls_back_to_tool_without_support() {
    // A provider with no native json_schema support must NOT silently degrade
    // to unconstrained text — it falls back to forced tool mode.
    let client = RecordingClient::new(
        NativeStructuredSupport::ForcedTool,
        vec![MockStructuredClient::tool_call_response(
            "emit_person",
            serde_json::json!({ "name": "Bob" }),
        )],
    );
    let result = generate_blocking(&client, &person_request(StructuredMode::Strict))
        .await
        .unwrap();

    assert_eq!(result.mode_used, StructuredMode::Tool);
    let directive = client.last_directive.lock().unwrap().clone().unwrap();
    assert_eq!(directive.force_tool.as_deref(), Some("emit_person"));
    assert!(directive.response_format.is_none());
}

#[tokio::test]
async fn test_routing_json_uses_json_object_when_supported() {
    let client = RecordingClient::new(
        NativeStructuredSupport::JsonSchema,
        vec![MockStructuredClient::text_response(r#"{"name": "Bob"}"#)],
    );
    let result = generate_blocking(&client, &person_request(StructuredMode::Json))
        .await
        .unwrap();

    assert_eq!(result.mode_used, StructuredMode::Json);
    let directive = client.last_directive.lock().unwrap().clone().unwrap();
    assert_eq!(directive.response_format, Some(ResponseFormat::JsonObject));
    assert!(directive.force_tool.is_none());
}

// ========================================================================
// Adversarial JSON extraction edge cases
// ========================================================================

#[test]
fn test_extract_json_crlf_code_fence() {
    let input = "```json\r\n{\"a\": 1}\r\n```";
    let result = extract_json_value(input).unwrap();
    assert_eq!(result["a"], 1);
}

#[test]
fn test_extract_json_prose_with_brace_in_string() {
    // A `}` inside a string value, plus trailing prose: balanced extraction
    // must keep the in-string brace and stop at the real closing brace.
    let input = "Sure thing: {\"msg\": \"close with } please\"} — done.";
    let result = extract_json_value(input).unwrap();
    assert_eq!(result["msg"], "close with } please");
}

#[test]
fn test_extract_json_single_quotes_rejected() {
    // Single-quoted pseudo-JSON is NOT valid JSON; extraction must fail rather
    // than silently mis-parse. (Documents current, intentional behavior.)
    assert!(extract_json_value("{'name': 'Bob'}").is_err());
}

#[test]
fn test_extract_raw_output_tool_mode_falls_back_to_text() {
    // If the model ignores the forced tool and replies with text anyway, tool
    // mode must fall back to reading the message text.
    let msg = Message {
        role: "assistant".to_string(),
        content: vec![ContentBlock::Text {
            text: r#"{"name": "Bob"}"#.to_string(),
        }],
        reasoning_content: None,
    };
    let raw = extract_raw_output(&msg, StructuredMode::Tool);
    let value = extract_json_value(&raw).unwrap();
    assert_eq!(value["name"], "Bob");
}
