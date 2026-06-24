//! RL trajectory recording primitives.
//!
//! The default runtime trace is intentionally lightweight and diagnostic. This
//! module records an opt-in training-oriented JSONL stream that can reconstruct LLM
//! turns, tool calls, tool observations, token usage, and termination reasons.
//! It is opt-in: normal sessions pay only a cheap disabled-recorder branch.

use crate::llm::{ContentBlock, LlmResponse, Message, TokenUsage, ToolCall};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

pub const RL_TRAJECTORY_SCHEMA: &str = "a3s.rl_trajectory.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RlTrajectoryMode {
    #[default]
    Off,
    On,
}

impl RlTrajectoryMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "off" | "0" | "false" | "none" => Some(Self::Off),
            "on" | "1" | "true" | "yes" | "enabled" | "rl" | "train" | "training"
            | "trajectory" | "trace" | "debug" | "full" | "compact" => Some(Self::On),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlTrajectoryConfig {
    pub mode: RlTrajectoryMode,
    pub path: PathBuf,
    pub max_text_bytes: usize,
    pub include_messages: bool,
}

impl RlTrajectoryConfig {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            mode: RlTrajectoryMode::On,
            path: path.into(),
            max_text_bytes: default_max_text_bytes(RlTrajectoryMode::On),
            include_messages: true,
        }
    }

    pub fn with_mode(mut self, mode: RlTrajectoryMode) -> Self {
        self.mode = mode;
        self.max_text_bytes = default_max_text_bytes(mode);
        self.include_messages = mode == RlTrajectoryMode::On;
        self
    }

    pub fn with_max_text_bytes(mut self, max_text_bytes: usize) -> Self {
        self.max_text_bytes = max_text_bytes;
        self
    }

    pub fn with_include_messages(mut self, include_messages: bool) -> Self {
        self.include_messages = include_messages;
        self
    }

    pub fn from_env() -> Result<Option<Self>> {
        let mode_env = env_first(&["A3S_CODE_TRAJECTORY_MODE", "A3S_CODE_RL_TRAJECTORY_MODE"]);
        let path_env = env_first(&["A3S_CODE_TRAJECTORY_PATH", "A3S_CODE_RL_TRAJECTORY_PATH"]);
        if mode_env.is_none() && path_env.is_none() {
            return Ok(None);
        }

        let mode = match mode_env.as_deref() {
            Some(raw) => RlTrajectoryMode::parse(raw)
                .with_context(|| format!("invalid A3S_CODE_RL_TRAJECTORY_MODE: {raw}"))?,
            None => RlTrajectoryMode::On,
        };
        if mode == RlTrajectoryMode::Off {
            return Ok(None);
        }

        let path = path_env
            .filter(|s| !s.trim().is_empty())
            .with_context(|| "A3S_CODE_TRAJECTORY_PATH is required when trajectory mode is on")?;

        let max_text_bytes = env_first(&[
            "A3S_CODE_TRAJECTORY_MAX_TEXT_BYTES",
            "A3S_CODE_RL_TRAJECTORY_MAX_TEXT_BYTES",
        ])
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_else(|| default_max_text_bytes(mode));

        let include_messages = env_first(&[
            "A3S_CODE_TRAJECTORY_INCLUDE_MESSAGES",
            "A3S_CODE_RL_TRAJECTORY_INCLUDE_MESSAGES",
        ])
        .and_then(|value| parse_bool(&value))
        .unwrap_or(true);

        Ok(Some(Self {
            mode,
            path: PathBuf::from(path),
            max_text_bytes,
            include_messages,
        }))
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RlTrajectoryContext {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replica_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_id: Option<String>,
}

impl RlTrajectoryContext {
    fn from_env() -> Self {
        Self {
            run_id: env_first(&["A3S_CODE_RL_RUN_ID", "A3S_CODE_RUN_ID", "A3S_RUN_ID"]),
            task_id: env_first(&["A3S_CODE_RL_TASK_ID", "A3S_CODE_TASK_ID", "TASK_ID"]),
            group_id: env_first(&["A3S_CODE_RL_GROUP_ID", "A3S_CODE_GROUP_ID"]),
            replica_id: env_first(&["A3S_CODE_RL_REPLICA_ID", "A3S_CODE_REPLICA_ID"]),
            sample_id: env_first(&["A3S_CODE_RL_SAMPLE_ID", "A3S_CODE_SAMPLE_ID"]),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturedText {
    pub byte_len: usize,
    pub sha256: String,
    pub truncated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preview: Option<String>,
}

#[derive(Clone)]
pub struct RlTrajectoryRecorder {
    inner: Option<Arc<RlTrajectoryRecorderInner>>,
}

struct RlTrajectoryRecorderInner {
    config: RlTrajectoryConfig,
    context: RlTrajectoryContext,
    sequence: AtomicU64,
    file: Mutex<File>,
}

impl std::fmt::Debug for RlTrajectoryRecorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RlTrajectoryRecorder")
            .field("enabled", &self.inner.is_some())
            .finish()
    }
}

impl Default for RlTrajectoryRecorder {
    fn default() -> Self {
        Self::disabled()
    }
}

impl RlTrajectoryRecorder {
    pub fn disabled() -> Self {
        Self { inner: None }
    }

    pub fn from_config(config: Option<RlTrajectoryConfig>) -> Result<Self> {
        let Some(config) = config else {
            return Ok(Self::disabled());
        };
        if config.mode == RlTrajectoryMode::Off {
            return Ok(Self::disabled());
        }

        if let Some(parent) = config.path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create RL trajectory directory {}",
                    parent.display()
                )
            })?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&config.path)
            .with_context(|| {
                format!(
                    "failed to open RL trajectory JSONL {}",
                    config.path.display()
                )
            })?;

        Ok(Self {
            inner: Some(Arc::new(RlTrajectoryRecorderInner {
                config,
                context: RlTrajectoryContext::from_env(),
                sequence: AtomicU64::new(0),
                file: Mutex::new(file),
            })),
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    pub fn record_execution_start(
        &self,
        session_id: &str,
        workspace: &Path,
        prompt: &str,
        history: &[Message],
        system_prompt: Option<&str>,
        max_tool_rounds: usize,
        planning_mode: &str,
    ) {
        let Some(inner) = &self.inner else {
            return;
        };
        let payload = json!({
            "workspace": workspace.display().to_string(),
            "prompt": inner.capture_text(prompt),
            "history_message_count": history.len(),
            "history": inner.capture_messages(history),
            "system_prompt": system_prompt.map(|s| inner.capture_text(s)),
            "max_tool_rounds": max_tool_rounds,
            "planning_mode": planning_mode,
        });
        inner.record("execution_start", session_id, payload);
    }

    pub fn record_llm_request(
        &self,
        session_id: &str,
        turn: usize,
        messages: &[Message],
        system: Option<&str>,
        available_tools: &[String],
        estimated_prompt_tokens: usize,
    ) {
        let Some(inner) = &self.inner else {
            return;
        };
        let payload = json!({
            "turn": turn,
            "messages_count": messages.len(),
            "messages": inner.capture_messages(messages),
            "system_prompt": system.map(|s| inner.capture_text(s)),
            "available_tools": available_tools,
            "estimated_prompt_tokens": estimated_prompt_tokens,
        });
        inner.record("llm_request", session_id, payload);
    }

    pub fn record_llm_response(
        &self,
        session_id: &str,
        turn: usize,
        response: &LlmResponse,
        duration_ms: u64,
    ) {
        let Some(inner) = &self.inner else {
            return;
        };
        let payload = json!({
            "turn": turn,
            "message": inner.capture_message(&response.message),
            "response_text": inner.capture_text(&response.text()),
            "reasoning_content": response.message.reasoning_content.as_ref().map(|s| inner.capture_text(s)),
            "tool_calls": response.tool_calls().iter().map(tool_call_value).collect::<Vec<_>>(),
            "usage": token_usage_value(&response.usage),
            "stop_reason": response.stop_reason.clone(),
            "meta": response.meta.clone(),
            "duration_ms": duration_ms,
        });
        inner.record("llm_response", session_id, payload);
    }

    pub fn record_tool_call(&self, session_id: &str, turn: usize, tool_call: &ToolCall) {
        let Some(inner) = &self.inner else {
            return;
        };
        let payload = json!({
            "turn": turn,
            "tool_call_id": tool_call.id,
            "tool": tool_call.name,
            "args": tool_call.args,
        });
        inner.record("tool_call", session_id, payload);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_tool_result(
        &self,
        session_id: &str,
        turn: usize,
        tool_call_id: &str,
        tool_name: &str,
        output: &str,
        exit_code: i32,
        duration_ms: u64,
        metadata: &Option<Value>,
        error_kind: Option<String>,
    ) {
        let Some(inner) = &self.inner else {
            return;
        };
        let payload = json!({
            "turn": turn,
            "tool_call_id": tool_call_id,
            "tool": tool_name,
            "success": exit_code == 0,
            "exit_code": exit_code,
            "duration_ms": duration_ms,
            "output": inner.capture_text(output),
            "metadata": metadata,
            "error_kind": error_kind,
        });
        inner.record("tool_result", session_id, payload);
    }

    pub fn record_context_compacted(
        &self,
        session_id: &str,
        before_messages: usize,
        after_messages: &[Message],
        percent_before: f32,
    ) {
        let Some(inner) = &self.inner else {
            return;
        };
        let payload = json!({
            "before_messages": before_messages,
            "after_messages": after_messages.len(),
            "percent_before": percent_before,
            "messages": inner.capture_messages(after_messages),
        });
        inner.record("context_compacted", session_id, payload);
    }

    pub fn record_execution_end(
        &self,
        session_id: &str,
        success: bool,
        response_text: Option<&str>,
        usage: Option<&TokenUsage>,
        tool_calls_count: Option<usize>,
        error_message: Option<&str>,
    ) {
        let Some(inner) = &self.inner else {
            return;
        };
        let payload = json!({
            "success": success,
            "response_text": response_text.map(|s| inner.capture_text(s)),
            "usage": usage.map(token_usage_value),
            "tool_calls_count": tool_calls_count,
            "error_message": error_message.map(|s| inner.capture_text(s)),
        });
        inner.record("execution_end", session_id, payload);
    }
}

impl RlTrajectoryRecorderInner {
    fn record(&self, event_type: &str, session_id: &str, payload: Value) {
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed) + 1;
        let record = json!({
            "schema": RL_TRAJECTORY_SCHEMA,
            "sequence": sequence,
            "timestamp_ms": chrono::Utc::now().timestamp_millis(),
            "event_type": event_type,
            "session_id": session_id,
            "mode": self.config.mode,
            "context": self.context,
            "payload": payload,
        });

        let line = match serde_json::to_string(&record) {
            Ok(line) => line,
            Err(err) => {
                tracing::warn!(error = %err, "Failed to serialize RL trajectory record");
                return;
            }
        };
        let mut file = match self.file.lock() {
            Ok(file) => file,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Err(err) = writeln!(file, "{line}") {
            tracing::warn!(error = %err, "Failed to write RL trajectory record");
        }
    }

    fn capture_messages(&self, messages: &[Message]) -> Value {
        if !self.config.include_messages {
            return json!({
                "included": false,
                "count": messages.len(),
                "roles": messages.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
            });
        }
        Value::Array(
            messages
                .iter()
                .enumerate()
                .map(|(index, message)| {
                    let mut value = self.capture_message(message);
                    if let Value::Object(ref mut object) = value {
                        object.insert("index".to_string(), json!(index));
                    }
                    value
                })
                .collect(),
        )
    }

    fn capture_message(&self, message: &Message) -> Value {
        json!({
            "role": message.role,
            "content": message.content.iter().map(|block| self.capture_content_block(block)).collect::<Vec<_>>(),
            "reasoning_content": message.reasoning_content.as_ref().map(|s| self.capture_text(s)),
        })
    }

    fn capture_content_block(&self, block: &ContentBlock) -> Value {
        match block {
            ContentBlock::Text { text } => json!({
                "type": "text",
                "text": self.capture_text(text),
            }),
            ContentBlock::Image { source } => json!({
                "type": "image",
                "source": {
                    "media_type": source.media_type,
                    "data": self.capture_text(&source.data),
                }
            }),
            ContentBlock::ToolUse { id, name, input } => json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input,
            }),
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": self.capture_text(&content.as_text()),
                "is_error": is_error,
            }),
        }
    }

    fn capture_text(&self, text: &str) -> CapturedText {
        let byte_len = text.len();
        let sha256 = sha256::digest(text);
        let (captured, truncated) = truncate_utf8(text, self.config.max_text_bytes);
        CapturedText {
            byte_len,
            sha256,
            truncated,
            text: Some(captured),
            preview: None,
        }
    }
}

fn tool_call_value(tool_call: &ToolCall) -> Value {
    json!({
        "tool_call_id": tool_call.id,
        "tool": tool_call.name,
        "args": tool_call.args,
    })
}

fn token_usage_value(usage: &TokenUsage) -> Value {
    json!({
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
    })
}

fn truncate_utf8(text: &str, max_bytes: usize) -> (String, bool) {
    if text.len() <= max_bytes {
        return (text.to_string(), false);
    }
    let mut end = max_bytes.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    (text[..end].to_string(), true)
}

fn default_max_text_bytes(mode: RlTrajectoryMode) -> usize {
    match mode {
        RlTrajectoryMode::Off => 0,
        RlTrajectoryMode::On => 1024 * 1024,
    }
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn env_first(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn rl_recorder_writes_jsonl() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("trajectory.jsonl");
        let recorder =
            RlTrajectoryRecorder::from_config(Some(RlTrajectoryConfig::new(&path))).unwrap();

        recorder.record_execution_start(
            "sess-1",
            Path::new("/workspace"),
            "solve task",
            &[],
            Some("system"),
            64,
            "disabled",
        );
        recorder.record_tool_result("sess-1", 1, "tool-1", "bash", "ok", 0, 3, &None, None);

        let lines = std::fs::read_to_string(path).unwrap();
        assert_eq!(lines.lines().count(), 2);
        let first: Value = serde_json::from_str(lines.lines().next().unwrap()).unwrap();
        assert_eq!(first["schema"], RL_TRAJECTORY_SCHEMA);
        assert_eq!(first["event_type"], "execution_start");
        assert_eq!(first["session_id"], "sess-1");
    }

    #[test]
    fn enabled_mode_records_text_with_truncation_flag() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("trajectory.jsonl");
        let recorder = RlTrajectoryRecorder::from_config(Some(
            RlTrajectoryConfig::new(&path).with_max_text_bytes(3),
        ))
        .unwrap();

        recorder.record_execution_start(
            "sess-1",
            Path::new("/workspace"),
            "abcdef",
            &[],
            None,
            64,
            "auto",
        );

        let text = std::fs::read_to_string(path).unwrap();
        let record: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        let prompt = &record["payload"]["prompt"];
        assert!(prompt.get("sha256").is_some());
        assert_eq!(prompt["text"], "abc");
        assert_eq!(prompt["truncated"], true);
    }
}
