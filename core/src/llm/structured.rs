//! Structured object generation from LLM output.
//!
//! Provides reliable JSON object generation with schema validation, automatic
//! repair, and streaming partial object support. Works across all providers by
//! selecting the best available mode (strict JSON schema, json_mode, tool-call,
//! or prompt-only).

use super::{LlmClient, Message, StreamEvent, TokenUsage, ToolDefinition};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Mode selection for structured output generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredMode {
    /// Auto-select best mode based on provider capabilities.
    Auto,
    /// OpenAI native strict JSON schema (response_format.type = json_schema).
    Strict,
    /// OpenAI json_object mode (guarantees valid JSON, not schema-conformant).
    Json,
    /// Use tool-calling: inject a synthetic tool whose parameters IS the schema.
    /// Works on all providers that support tool use (Anthropic, OpenAI, etc).
    Tool,
    /// Prompt-only: append schema instructions to the prompt. Least reliable.
    Prompt,
}

/// Request specification for structured object generation.
#[derive(Debug, Clone)]
pub struct StructuredRequest {
    pub prompt: String,
    pub system: Option<String>,
    pub schema: Value,
    pub schema_name: String,
    pub schema_description: Option<String>,
    pub mode: StructuredMode,
    pub max_repair_attempts: u8,
}

/// Result of a successful structured generation.
#[derive(Debug, Clone, Serialize)]
pub struct StructuredResult {
    pub object: Value,
    pub raw_text: Option<String>,
    pub usage: TokenUsage,
    pub repair_rounds: u8,
    pub mode_used: StructuredMode,
}

/// Provider-native structured-output capability.
///
/// Each [`LlmClient`] reports this so the structured engine can request the
/// strongest enforcement the provider actually supports. Defaults to
/// [`NativeStructuredSupport::None`] for clients that don't override it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeStructuredSupport {
    /// No native enforcement — rely on prompt instructions + lenient extraction.
    None,
    /// Can force a specific tool call (Anthropic `tool_choice`, OpenAI function
    /// `tool_choice`). Guarantees the model emits the structured tool call
    /// instead of free-form prose.
    ForcedTool,
    /// Supports OpenAI-style `response_format` (`json_object` and
    /// `json_schema` + `strict`) in addition to forced tool calls.
    JsonSchema,
}

/// A native `response_format` request for OpenAI-compatible providers.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseFormat {
    /// `{"type":"json_object"}` — guarantees syntactically valid JSON, but not
    /// schema conformance.
    JsonObject,
    /// `{"type":"json_schema","json_schema":{name,schema,strict:true}}` —
    /// parser-enforced schema conformance.
    JsonSchema { name: String, schema: Value },
}

/// Instruction telling a provider how to enforce structured output for a call.
///
/// Carries the union of intents; each provider honors what it supports and
/// ignores the rest (e.g. Anthropic has no `response_format`, so it only acts
/// on `force_tool`). The default (`force_tool: None, response_format: None`)
/// reproduces an ordinary completion, which is why the trait's default
/// `complete_structured` impl is behavior-preserving.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StructuredDirective {
    /// Force the model to call exactly this tool (provider `tool_choice`).
    pub force_tool: Option<String>,
    /// Request a provider-native `response_format` (OpenAI-compatible only).
    pub response_format: Option<ResponseFormat>,
}

/// Callback for streaming partial object snapshots.
pub type PartialObjectCallback = Box<dyn Fn(&Value) + Send>;

// ---------------------------------------------------------------------------
// Core generation: blocking (non-streaming)
// ---------------------------------------------------------------------------

/// Generate a structured JSON object using the given LLM client.
///
/// Selects the best mode based on `req.mode`, calls the LLM, validates against
/// the schema, and retries with repair prompts if validation fails.
pub async fn generate_blocking(
    client: &dyn LlmClient,
    req: &StructuredRequest,
) -> Result<StructuredResult> {
    let mode = resolve_mode(req.mode, client.native_structured_support());
    let mut messages = build_initial_messages(req, mode);
    let system = build_system_prompt(req, mode);
    let tools = build_tools(req, mode);
    let directive = build_directive(req, mode);

    let mut total_usage = TokenUsage::default();
    let mut repair_rounds: u8 = 0;

    loop {
        let resp = client
            .complete_structured(&messages, Some(&system), &tools, &directive)
            .await
            .context("LLM call failed during structured generation")?;

        accumulate_usage(&mut total_usage, &resp.usage);

        let raw_text = extract_raw_output(&resp.message, mode);
        let parsed = extract_json_value(&raw_text);

        match parsed {
            Ok(value) => match validate_against_schema(&value, &req.schema) {
                Ok(()) => {
                    return Ok(StructuredResult {
                        object: value,
                        raw_text: Some(raw_text),
                        usage: total_usage,
                        repair_rounds,
                        mode_used: mode,
                    });
                }
                Err(errors) if repair_rounds < req.max_repair_attempts => {
                    repair_rounds += 1;
                    let repair_msg = build_repair_message(&raw_text, &errors);
                    append_repair_context(
                        &mut messages,
                        &resp.message,
                        &repair_msg,
                        mode,
                        &raw_text,
                    );
                }
                Err(errors) => {
                    bail!(
                            "Structured output failed schema validation after {} repair attempts. Errors: {}",
                            repair_rounds,
                            errors.join("; ")
                        );
                }
            },
            Err(parse_err) if repair_rounds < req.max_repair_attempts => {
                repair_rounds += 1;
                let repair_msg = format!(
                    "Your previous output could not be parsed as JSON:\n\n{}\n\nError: {}\n\nPlease return ONLY a valid JSON object matching the schema.",
                    raw_text, parse_err
                );
                append_repair_context(&mut messages, &resp.message, &repair_msg, mode, &raw_text);
            }
            Err(parse_err) => {
                bail!(
                    "Structured output failed JSON parsing after {} repair attempts: {}",
                    repair_rounds,
                    parse_err
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Core generation: streaming
// ---------------------------------------------------------------------------

/// Generate a structured JSON object with streaming partial updates.
///
/// Calls `on_partial` with progressively more complete partial objects as tokens
/// arrive. Returns the final validated object.
///
/// In streaming mode, `max_repair_attempts` defaults to 0 because a repair
/// would reset the partial object stream (confusing for consumers).
pub async fn generate_streaming(
    client: &dyn LlmClient,
    req: &StructuredRequest,
    on_partial: PartialObjectCallback,
) -> Result<StructuredResult> {
    let mode = resolve_mode(req.mode, client.native_structured_support());
    let messages = build_initial_messages(req, mode);
    let system = build_system_prompt(req, mode);
    let tools = build_tools(req, mode);
    let directive = build_directive(req, mode);

    let cancel_token = CancellationToken::new();
    let mut rx = client
        .complete_streaming_structured(&messages, Some(&system), &tools, &directive, cancel_token)
        .await
        .context("LLM streaming call failed during structured generation")?;

    let mut json_buffer = String::new();
    let mut last_valid_partial: Option<Value> = None;
    let mut final_response: Option<super::LlmResponse> = None;
    let mut last_parse_len: usize = 0;
    // Minimum bytes of new data before attempting a partial parse (reduces CPU)
    const PARSE_THRESHOLD: usize = 8;

    while let Some(event) = rx.recv().await {
        match event {
            StreamEvent::ToolUseInputDelta(delta) if mode == StructuredMode::Tool => {
                if final_response.is_some() {
                    continue;
                }
                json_buffer.push_str(&delta);
                if json_buffer.len() - last_parse_len >= PARSE_THRESHOLD {
                    if let Some(partial) = try_parse_partial_json(&json_buffer) {
                        if last_valid_partial.as_ref() != Some(&partial) {
                            on_partial(&partial);
                            last_valid_partial = Some(partial);
                        }
                    }
                    last_parse_len = json_buffer.len();
                }
            }
            StreamEvent::TextDelta(delta) if mode != StructuredMode::Tool => {
                if final_response.is_some() {
                    continue;
                }
                json_buffer.push_str(&delta);
                if json_buffer.len() - last_parse_len >= PARSE_THRESHOLD {
                    if let Some(json_start) = find_json_start(&json_buffer) {
                        let candidate = &json_buffer[json_start..];
                        if let Some(partial) = try_parse_partial_json(candidate) {
                            if last_valid_partial.as_ref() != Some(&partial) {
                                on_partial(&partial);
                                last_valid_partial = Some(partial);
                            }
                        }
                    }
                    last_parse_len = json_buffer.len();
                }
            }
            StreamEvent::Done(resp) => {
                final_response = Some(resp);
            }
            _ => {}
        }
    }

    let resp = final_response.context("Stream ended without Done event")?;
    let raw_text = extract_raw_output(&resp.message, mode);
    let value =
        extract_json_value(&raw_text).context("Failed to parse final streamed output as JSON")?;

    validate_against_schema(&value, &req.schema).map_err(|errors| {
        anyhow::anyhow!(
            "Streamed structured output failed schema validation: {}",
            errors.join("; ")
        )
    })?;

    // Emit final complete object
    on_partial(&value);

    Ok(StructuredResult {
        object: value,
        raw_text: Some(raw_text),
        usage: resp.usage,
        repair_rounds: 0,
        mode_used: mode,
    })
}

// ---------------------------------------------------------------------------
// JSON extraction and parsing
// ---------------------------------------------------------------------------

/// Extract a JSON value from potentially dirty LLM output.
///
/// Handles: raw JSON, markdown code fences, leading/trailing prose.
pub fn extract_json_value(text: &str) -> Result<Value> {
    let trimmed = text.trim();

    // 1. Direct parse
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        if v.is_object() || v.is_array() {
            return Ok(v);
        }
    }

    // 2. Strip markdown code fence
    if let Some(inner) = strip_code_fence(trimmed) {
        if let Ok(v) = serde_json::from_str::<Value>(inner.trim()) {
            if v.is_object() || v.is_array() {
                return Ok(v);
            }
        }
    }

    // 3. Find balanced JSON substring (first { to matching })
    if let Some(candidate) = find_balanced_json_object(trimmed) {
        if let Ok(v) = serde_json::from_str::<Value>(candidate) {
            return Ok(v);
        }
    }

    // 4. Try array
    if let Some(candidate) = find_balanced_json_array(trimmed) {
        if let Ok(v) = serde_json::from_str::<Value>(candidate) {
            return Ok(v);
        }
    }

    bail!("No valid JSON object found in LLM output")
}

/// Strip ```json ... ``` or ``` ... ``` fences.
fn strip_code_fence(text: &str) -> Option<&str> {
    let start_patterns = ["```json\n", "```json\r\n", "```\n", "```\r\n"];
    for pat in &start_patterns {
        if let Some(rest) = text.strip_prefix(pat) {
            // Find closing fence
            if let Some(end) = rest.rfind("```") {
                return Some(&rest[..end]);
            }
        }
    }
    // Also handle inline: ```json{...}```
    if let Some(inner) = text.strip_prefix("```json") {
        if let Some(end) = inner.rfind("```") {
            return Some(inner[..end].trim());
        }
    }
    if let Some(inner) = text.strip_prefix("```") {
        if let Some(end) = inner.rfind("```") {
            return Some(inner[..end].trim());
        }
    }
    None
}

/// Find the first balanced `{...}` substring using bracket counting.
fn find_balanced_json_object(text: &str) -> Option<&str> {
    find_balanced(text, '{', '}')
}

/// Find the first balanced `[...]` substring.
fn find_balanced_json_array(text: &str) -> Option<&str> {
    find_balanced(text, '[', ']')
}

fn find_balanced(text: &str, open: char, close: char) -> Option<&str> {
    let bytes = text.as_bytes();
    let open_byte = open as u8;
    let close_byte = close as u8;

    // Find the first unquoted occurrence of `open`
    let mut in_string = false;
    let mut escape_next = false;
    let mut start = None;

    for (i, &b) in bytes.iter().enumerate() {
        if escape_next {
            escape_next = false;
            continue;
        }
        match b {
            b'\\' if in_string => escape_next = true,
            b'"' => in_string = !in_string,
            _ if in_string => {}
            _ if b == open_byte => {
                start = Some(i);
                break;
            }
            _ => {}
        }
    }

    let start = start?;
    let mut depth = 0i32;
    in_string = false;
    escape_next = false;

    for (i, &b) in bytes[start..].iter().enumerate() {
        if escape_next {
            escape_next = false;
            continue;
        }
        match b {
            b'\\' if in_string => escape_next = true,
            b'"' => in_string = !in_string,
            _ if in_string => {}
            _ if b == open_byte => depth += 1,
            _ if b == close_byte => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start..start + i + 1]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Find the byte offset where JSON content starts in a text stream.
/// Skips leading prose/whitespace to find `{` or `[` that isn't inside a string.
fn find_json_start(text: &str) -> Option<usize> {
    // Skip past code fence markers if present
    let (search_text, offset) = if let Some(rest) = text.strip_prefix("```json") {
        (rest, 7)
    } else if let Some(rest) = text.strip_prefix("```") {
        (rest, 3)
    } else {
        (text, 0)
    };

    let mut in_string = false;
    let mut escape_next = false;
    for (i, &b) in search_text.as_bytes().iter().enumerate() {
        if escape_next {
            escape_next = false;
            continue;
        }
        match b {
            b'\\' if in_string => {
                escape_next = true;
            }
            b'"' => {
                in_string = !in_string;
            }
            b'{' | b'[' if !in_string => {
                return Some(offset + i);
            }
            _ => {}
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Partial JSON parsing (for streaming)
// ---------------------------------------------------------------------------

/// Attempt to parse a potentially incomplete JSON string into the most complete
/// valid partial object possible.
///
/// Strategy: try parsing as-is first. If that fails, progressively close open
/// braces/brackets and try again. This handles the common case where the LLM
/// has output `{"name": "foo", "items": [1, 2` — we close it to get a partial.
fn try_parse_partial_json(text: &str) -> Option<Value> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }

    // Fast path: already valid
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        if v.is_object() || v.is_array() {
            return Some(v);
        }
    }

    // Count unclosed brackets/braces (respecting strings)
    let mut closers = Vec::new();
    let mut in_string = false;
    let mut escape_next = false;
    // Track if we're mid-value (after a colon or comma, before the value is complete)
    let mut last_significant: Option<u8> = None;

    for &b in trimmed.as_bytes() {
        if escape_next {
            escape_next = false;
            continue;
        }
        match b {
            b'\\' if in_string => {
                escape_next = true;
            }
            b'"' => {
                in_string = !in_string;
                if !in_string {
                    last_significant = Some(b'"');
                }
            }
            _ if in_string => {}
            b'{' => {
                closers.push(b'}');
                last_significant = Some(b'{');
            }
            b'[' => {
                closers.push(b']');
                last_significant = Some(b'[');
            }
            b'}' | b']' => {
                closers.pop();
                last_significant = Some(b);
            }
            b':' | b',' => {
                last_significant = Some(b);
            }
            b if !b.is_ascii_whitespace() => {
                last_significant = Some(b);
            }
            _ => {}
        }
    }

    if closers.is_empty() {
        return None; // Already balanced but didn't parse — genuinely invalid
    }

    // Pre-allocate repair buffer: original + at most 6 extra chars (null + closers)
    let mut repaired = String::with_capacity(trimmed.len() + closers.len() + 6);
    repaired.push_str(trimmed);

    if in_string {
        repaired.push('"');
        last_significant = Some(b'"');
    }

    // If last significant char suggests an incomplete key or value, handle it
    if let Some(last) = last_significant {
        if last == b':' {
            // Key with no value yet — add null
            repaired.push_str("null");
        } else if last == b',' {
            // Trailing comma — some parsers choke on this, trim it
            if let Some(pos) = repaired.rfind(',') {
                repaired.truncate(pos);
            }
        }
    }

    // Close all open brackets/braces
    for &closer in closers.iter().rev() {
        repaired.push(closer as char);
    }

    serde_json::from_str::<Value>(&repaired)
        .ok()
        .filter(|v| v.is_object() || v.is_array())
}

// ---------------------------------------------------------------------------
// Schema validation
// ---------------------------------------------------------------------------

/// Validate a JSON value against a JSON Schema.
/// Returns Ok(()) on success, or a list of human-readable error strings.
fn validate_against_schema(value: &Value, schema: &Value) -> Result<(), Vec<String>> {
    // We do a basic recursive validation here. For production, consider using
    // the `jsonschema` crate, but to avoid adding a heavy dependency we implement
    // the subset of JSON Schema that matters for structured output.
    let errors = basic_schema_validate(value, schema, "");
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Basic JSON Schema validator covering the most common constraints.
fn basic_schema_validate(value: &Value, schema: &Value, path: &str) -> Vec<String> {
    let mut errors = Vec::new();

    // Handle $ref — not supported in basic validator, skip
    if schema.get("$ref").is_some() {
        return errors;
    }

    // Handle anyOf / oneOf: value must match at least one sub-schema
    if let Some(any_of) = schema
        .get("anyOf")
        .or_else(|| schema.get("oneOf"))
        .and_then(|v| v.as_array())
    {
        let matched = any_of
            .iter()
            .any(|sub| basic_schema_validate(value, sub, path).is_empty());
        if !matched {
            errors.push(format!(
                "{}: value does not match any variant in anyOf/oneOf",
                path_or_root(path),
            ));
        }
        return errors;
    }

    // Handle enum
    if let Some(enum_values) = schema.get("enum").and_then(|v| v.as_array()) {
        if !enum_values.contains(value) {
            errors.push(format!(
                "{}: value {:?} not in enum {:?}",
                path_or_root(path),
                value,
                enum_values
            ));
        }
        return errors;
    }

    // Handle const
    if let Some(const_val) = schema.get("const") {
        if value != const_val {
            errors.push(format!(
                "{}: expected const {:?}, got {:?}",
                path_or_root(path),
                const_val,
                value
            ));
        }
        return errors;
    }

    // Type checking (supports nullable via type array: ["string", "null"])
    if let Some(type_val) = schema.get("type") {
        let type_ok = if let Some(type_str) = type_val.as_str() {
            check_type(value, type_str)
        } else if let Some(type_arr) = type_val.as_array() {
            type_arr
                .iter()
                .filter_map(|t| t.as_str())
                .any(|t| check_type(value, t))
        } else {
            true
        };
        if !type_ok {
            errors.push(format!(
                "{}: expected type {:?}, got {:?}",
                path_or_root(path),
                type_val,
                value_type_name(value)
            ));
            return errors;
        }
    }

    // Object validation
    if let Some(obj) = value.as_object() {
        if let Some(properties) = schema.get("properties").and_then(|v| v.as_object()) {
            for (key, prop_schema) in properties {
                if let Some(child_value) = obj.get(key) {
                    let child_path = if path.is_empty() {
                        format!(".{}", key)
                    } else {
                        format!("{}.{}", path, key)
                    };
                    errors.extend(basic_schema_validate(child_value, prop_schema, &child_path));
                }
            }
        }

        if let Some(required) = schema.get("required").and_then(|v| v.as_array()) {
            for req_field in required {
                if let Some(field_name) = req_field.as_str() {
                    if !obj.contains_key(field_name) {
                        errors.push(format!(
                            "{}: missing required field '{}'",
                            path_or_root(path),
                            field_name
                        ));
                    }
                }
            }
        }

        // additionalProperties: false
        if schema.get("additionalProperties") == Some(&Value::Bool(false)) {
            if let Some(properties) = schema.get("properties").and_then(|v| v.as_object()) {
                for key in obj.keys() {
                    if !properties.contains_key(key) {
                        errors.push(format!(
                            "{}: unexpected additional property '{}'",
                            path_or_root(path),
                            key
                        ));
                    }
                }
            }
        }
    }

    // Array validation
    if let Some(arr) = value.as_array() {
        if let Some(items_schema) = schema.get("items") {
            for (i, item) in arr.iter().enumerate() {
                let child_path = format!("{}[{}]", path, i);
                errors.extend(basic_schema_validate(item, items_schema, &child_path));
            }
        }
        if let Some(min) = schema.get("minItems").and_then(|v| v.as_u64()) {
            if (arr.len() as u64) < min {
                errors.push(format!(
                    "{}: array has {} items, minimum is {}",
                    path_or_root(path),
                    arr.len(),
                    min
                ));
            }
        }
        if let Some(max) = schema.get("maxItems").and_then(|v| v.as_u64()) {
            if (arr.len() as u64) > max {
                errors.push(format!(
                    "{}: array has {} items, maximum is {}",
                    path_or_root(path),
                    arr.len(),
                    max
                ));
            }
        }
    }

    // String validation
    if let Some(s) = value.as_str() {
        if let Some(min_len) = schema.get("minLength").and_then(|v| v.as_u64()) {
            if (s.chars().count() as u64) < min_len {
                errors.push(format!(
                    "{}: string length {} < minLength {}",
                    path_or_root(path),
                    s.chars().count(),
                    min_len
                ));
            }
        }
        if let Some(max_len) = schema.get("maxLength").and_then(|v| v.as_u64()) {
            if (s.chars().count() as u64) > max_len {
                errors.push(format!(
                    "{}: string length {} > maxLength {}",
                    path_or_root(path),
                    s.chars().count(),
                    max_len
                ));
            }
        }
        if let Some(pattern) = schema.get("pattern").and_then(|v| v.as_str()) {
            if let Ok(re) = regex::Regex::new(pattern) {
                if !re.is_match(s) {
                    errors.push(format!(
                        "{}: string does not match pattern '{}'",
                        path_or_root(path),
                        pattern
                    ));
                }
            }
        }
    }

    // Number validation
    if let Some(n) = value.as_f64() {
        if let Some(min) = schema.get("minimum").and_then(|v| v.as_f64()) {
            if n < min {
                errors.push(format!(
                    "{}: value {} < minimum {}",
                    path_or_root(path),
                    n,
                    min
                ));
            }
        }
        if let Some(max) = schema.get("maximum").and_then(|v| v.as_f64()) {
            if n > max {
                errors.push(format!(
                    "{}: value {} > maximum {}",
                    path_or_root(path),
                    n,
                    max
                ));
            }
        }
        if let Some(exc_min) = schema.get("exclusiveMinimum").and_then(|v| v.as_f64()) {
            if n <= exc_min {
                errors.push(format!(
                    "{}: value {} <= exclusiveMinimum {}",
                    path_or_root(path),
                    n,
                    exc_min
                ));
            }
        }
        if let Some(exc_max) = schema.get("exclusiveMaximum").and_then(|v| v.as_f64()) {
            if n >= exc_max {
                errors.push(format!(
                    "{}: value {} >= exclusiveMaximum {}",
                    path_or_root(path),
                    n,
                    exc_max
                ));
            }
        }
    }

    errors
}

fn check_type(value: &Value, type_str: &str) -> bool {
    match type_str {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => {
            value.is_i64()
                || value.is_u64()
                || value
                    .as_f64()
                    .map(|f| f.fract() == 0.0 && f.is_finite())
                    .unwrap_or(false)
        }
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => true,
    }
}

fn path_or_root(path: &str) -> &str {
    if path.is_empty() {
        "$"
    } else {
        path
    }
}

fn value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

// ---------------------------------------------------------------------------
// Message/prompt construction helpers
// ---------------------------------------------------------------------------

/// Resolve the requested mode against the provider's native capability.
///
/// `Auto`/`Tool` always resolve to forced `Tool` mode — the most reliable
/// cross-provider strategy (the synthetic `emit_*` tool is made mandatory via
/// the provider's `tool_choice`). `Strict`/`Json` use native `response_format`
/// only when the provider reports [`NativeStructuredSupport::JsonSchema`];
/// otherwise they fall back to forced `Tool` mode rather than silently
/// degrading to unconstrained text.
fn resolve_mode(requested: StructuredMode, support: NativeStructuredSupport) -> StructuredMode {
    match requested {
        StructuredMode::Prompt => StructuredMode::Prompt,
        StructuredMode::Strict if support == NativeStructuredSupport::JsonSchema => {
            StructuredMode::Strict
        }
        StructuredMode::Json if support == NativeStructuredSupport::JsonSchema => {
            StructuredMode::Json
        }
        // Auto, Tool, or Strict/Json on a provider without json_schema support.
        _ => StructuredMode::Tool,
    }
}

/// Build the provider directive for an already-resolved mode.
fn build_directive(req: &StructuredRequest, mode: StructuredMode) -> StructuredDirective {
    match mode {
        StructuredMode::Tool => StructuredDirective {
            force_tool: Some(format!("emit_{}", req.schema_name)),
            response_format: None,
        },
        StructuredMode::Strict => StructuredDirective {
            force_tool: None,
            response_format: Some(ResponseFormat::JsonSchema {
                name: req.schema_name.clone(),
                schema: req.schema.clone(),
            }),
        },
        StructuredMode::Json => StructuredDirective {
            force_tool: None,
            response_format: Some(ResponseFormat::JsonObject),
        },
        StructuredMode::Auto | StructuredMode::Prompt => StructuredDirective::default(),
    }
}

fn build_initial_messages(req: &StructuredRequest, mode: StructuredMode) -> Vec<Message> {
    match mode {
        StructuredMode::Tool => {
            // For tool mode, the prompt is the user message; the LLM will respond
            // with a tool call whose input is the structured object.
            vec![Message::user(&req.prompt)]
        }
        StructuredMode::Prompt | StructuredMode::Json => {
            // Prompt mode and json_object mode both need the schema in the prompt:
            // json_object only guarantees *syntactic* validity, so the model still
            // has to be told the shape it should produce.
            let augmented = format!(
                "{}\n\nYou MUST respond with ONLY a valid JSON object (no markdown, no explanation) that conforms to this JSON Schema:\n\n```json\n{}\n```",
                req.prompt,
                serde_json::to_string_pretty(&req.schema).unwrap_or_default()
            );
            vec![Message::user(&augmented)]
        }
        _ => {
            // Strict mode: the schema constraint is enforced by the provider via
            // response_format.json_schema, so the user message is just the prompt.
            vec![Message::user(&req.prompt)]
        }
    }
}

fn build_system_prompt(req: &StructuredRequest, mode: StructuredMode) -> String {
    let base = req.system.as_deref().unwrap_or("");

    match mode {
        StructuredMode::Tool => {
            format!(
                "{}{}You MUST respond by calling the `emit_{}` tool exactly once with a valid argument matching the schema. Do not output any text outside the tool call.",
                base,
                if base.is_empty() { "" } else { "\n\n" },
                req.schema_name
            )
        }
        StructuredMode::Prompt | StructuredMode::Json => {
            format!(
                "{}{}You are a structured data extraction assistant. Always respond with valid JSON only, no markdown fences, no explanation text.",
                base,
                if base.is_empty() { "" } else { "\n\n" },
            )
        }
        _ => base.to_string(),
    }
}

fn build_tools(req: &StructuredRequest, mode: StructuredMode) -> Vec<ToolDefinition> {
    match mode {
        StructuredMode::Tool => {
            vec![ToolDefinition {
                name: format!("emit_{}", req.schema_name),
                description: req
                    .schema_description
                    .clone()
                    .unwrap_or_else(|| format!("Emit a structured {} object", req.schema_name)),
                parameters: req.schema.clone(),
            }]
        }
        _ => vec![],
    }
}

/// Extract the raw JSON string from the LLM response based on mode.
fn extract_raw_output(message: &super::Message, mode: StructuredMode) -> String {
    match mode {
        StructuredMode::Tool => {
            // Look for tool call input
            let calls = message.tool_calls();
            if let Some(call) = calls.first() {
                serde_json::to_string(&call.args).unwrap_or_default()
            } else {
                // Fallback: maybe the model responded with text anyway
                message.text()
            }
        }
        _ => message.text(),
    }
}

fn build_repair_message(raw_text: &str, errors: &[String]) -> String {
    // Truncate raw output in repair message to avoid blowing context
    let truncated_raw = if raw_text.len() > 2000 {
        format!(
            "{}...[truncated, {} bytes total]",
            &raw_text[..2000],
            raw_text.len()
        )
    } else {
        raw_text.to_string()
    };
    format!(
        "Your previous output failed schema validation:\n\n{}\n\nValidation errors:\n{}\n\nPlease return ONLY a corrected JSON object that fixes these errors. No explanation, no markdown.",
        truncated_raw,
        errors.iter().map(|e| format!("- {}", e)).collect::<Vec<_>>().join("\n")
    )
}

fn accumulate_usage(total: &mut TokenUsage, delta: &TokenUsage) {
    total.prompt_tokens += delta.prompt_tokens;
    total.completion_tokens += delta.completion_tokens;
    total.total_tokens += delta.total_tokens;
}

/// Append repair context to the message history, respecting conversation structure.
///
/// In tool mode, the LLM returned a tool_use block. The correct follow-up is:
///   assistant (tool_use) → user (tool_result with error) → assistant (retry)
/// In text modes, it's simply:
///   assistant (text) → user (repair request) → assistant (retry)
fn append_repair_context(
    messages: &mut Vec<Message>,
    assistant_msg: &Message,
    repair_text: &str,
    mode: StructuredMode,
    _raw_text: &str,
) {
    if mode == StructuredMode::Tool {
        // Push the original assistant message (with tool_use block intact)
        messages.push(assistant_msg.clone());
        // Find the tool_use ID to construct a proper tool_result
        let tool_use_id = assistant_msg
            .tool_calls()
            .first()
            .map(|tc| tc.id.clone())
            .unwrap_or_else(|| "unknown".to_string());
        // Return the error as a tool_result so the conversation stays valid
        messages.push(Message::tool_result(&tool_use_id, repair_text, true));
    } else {
        // Text modes: push assistant text then user repair request
        messages.push(assistant_msg.clone());
        messages.push(Message::user(repair_text));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "structured_tests.rs"]
mod structured_tests;
