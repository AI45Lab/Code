//! Built-in `generate_object` tool for structured JSON output.
//!
//! This tool allows the agent (or users via `session.tool()`) to generate a
//! JSON object that conforms to a given JSON Schema. It supports streaming
//! partial objects via `ToolStreamEvent::OutputDelta`.

use crate::llm::structured::{self, PartialObjectCallback, StructuredMode, StructuredRequest};
use crate::llm::LlmClient;
use crate::tools::types::{Tool, ToolContext, ToolOutput, ToolStreamEvent};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

pub struct GenerateObjectTool {
    llm_client: Arc<dyn LlmClient>,
}

impl GenerateObjectTool {
    pub fn new(llm_client: Arc<dyn LlmClient>) -> Self {
        Self { llm_client }
    }
}

#[async_trait]
impl Tool for GenerateObjectTool {
    fn name(&self) -> &str {
        "generate_object"
    }

    fn description(&self) -> &str {
        "Generate a JSON object that strictly conforms to a provided JSON Schema. \
         Use when you need structured output: extracting fields from text, classifying \
         data, converting natural language to typed records, or producing machine-readable \
         results. Returns the validated object on success."
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "required": ["schema", "prompt"],
            "additionalProperties": false,
            "properties": {
                "schema": {
                    "type": "object",
                    "description": "JSON Schema that the output object must conform to"
                },
                "schema_name": {
                    "type": "string",
                    "description": "Short name for the schema (used internally for tool naming)",
                    "default": "result"
                },
                "schema_description": {
                    "type": "string",
                    "description": "Optional description of what the schema represents"
                },
                "prompt": {
                    "type": "string",
                    "description": "The prompt describing what object to generate or extract"
                },
                "system": {
                    "type": "string",
                    "description": "Optional system prompt to guide generation"
                },
                "mode": {
                    "type": "string",
                    "enum": ["auto", "strict", "json", "tool", "prompt"],
                    "description": "Output mode. 'auto' selects the best mode for the provider. 'tool' uses tool-calling (most reliable cross-provider). 'strict' uses OpenAI native JSON schema. 'json' uses json_object mode. 'prompt' appends schema to prompt.",
                    "default": "auto"
                },
                "max_repair_attempts": {
                    "type": "integer",
                    "description": "Maximum repair attempts if output fails validation (0-5)",
                    "default": 2,
                    "minimum": 0,
                    "maximum": 5
                }
            }
        })
    }

    async fn execute(&self, args: &Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let schema = match args.get("schema") {
            Some(s) if s.is_object() => s.clone(),
            Some(_) => {
                return Ok(ToolOutput::error(
                    "'schema' must be a JSON object (a valid JSON Schema)",
                ));
            }
            None => {
                return Ok(ToolOutput::error("'schema' parameter is required"));
            }
        };

        let prompt = match args.get("prompt").and_then(|v| v.as_str()) {
            Some(p) if !p.is_empty() => p.to_string(),
            _ => {
                return Ok(ToolOutput::error(
                    "'prompt' parameter is required and must be non-empty",
                ));
            }
        };

        // Validate schema has at minimum a "type" or "properties" or "anyOf" field
        if schema.get("type").is_none()
            && schema.get("properties").is_none()
            && schema.get("anyOf").is_none()
            && schema.get("oneOf").is_none()
            && schema.get("enum").is_none()
        {
            return Ok(ToolOutput::error(
                "'schema' should contain at least one of: type, properties, anyOf, oneOf, or enum",
            ));
        }

        let schema_name: String = args
            .get("schema_name")
            .and_then(|v| v.as_str())
            .unwrap_or("result")
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
            .take(64)
            .collect();
        let schema_name = if schema_name.is_empty() {
            "result".to_string()
        } else {
            schema_name
        };

        let schema_description = args
            .get("schema_description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let system = args
            .get("system")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let mode = match args.get("mode").and_then(|v| v.as_str()) {
            Some("strict") => StructuredMode::Strict,
            Some("json") => StructuredMode::Json,
            Some("tool") => StructuredMode::Tool,
            Some("prompt") => StructuredMode::Prompt,
            _ => StructuredMode::Auto,
        };

        // Mode resolution is delegated to the structured engine, which inspects
        // the client's native capability: Auto/Tool become forced tool calls,
        // and Strict/Json use native `response_format` when the provider
        // supports it (falling back to forced Tool mode otherwise).
        let max_repair_attempts = args
            .get("max_repair_attempts")
            .and_then(|v| v.as_u64())
            .unwrap_or(2)
            .min(5) as u8;

        let req = StructuredRequest {
            prompt,
            system,
            schema,
            schema_name,
            schema_description,
            mode,
            max_repair_attempts,
        };

        let result = if let Some(ref tx) = ctx.event_tx {
            let tx_clone = tx.clone();
            let callback: PartialObjectCallback = Box::new(move |partial: &Value| {
                let delta = serde_json::json!({
                    "object_partial": partial,
                    "final": false,
                });
                let delta_str = serde_json::to_string(&delta).unwrap_or_default();
                let _ = tx_clone.try_send(ToolStreamEvent::OutputDelta(delta_str));
            });
            structured::generate_streaming(&*self.llm_client, &req, callback).await
        } else {
            structured::generate_blocking(&*self.llm_client, &req).await
        };

        match result {
            Ok(sr) => {
                let output = serde_json::json!({
                    "object": sr.object,
                    "repair_rounds": sr.repair_rounds,
                    "mode_used": sr.mode_used,
                });
                Ok(ToolOutput::success(serde_json::to_string(&output)?))
            }
            Err(e) => Ok(ToolOutput::error(format!("generate_object failed: {}", e))),
        }
    }
}
