use super::AgentResult;
use crate::llm::{ContentBlock, Message, TokenUsage};
use crate::verification::VerificationReport;
use serde_json::Value;
use std::time::Instant;

const RECENT_TOOL_SIGNATURE_LIMIT: usize = 8;

pub(super) struct ExecutionLoopState {
    pub(super) messages: Vec<Message>,
    pub(super) total_usage: TokenUsage,
    pub(super) tool_calls_count: usize,
    pub(super) verification_reports: Vec<VerificationReport>,
    turn: usize,
    parse_error_count: u32,
    continuation_count: u32,
    recent_tool_signatures: Vec<String>,
    execution_start: Instant,
}

pub(super) struct ParseErrorOutcome {
    pub(super) output: String,
    pub(super) count: u32,
    pub(super) fatal_message: Option<String>,
}

/// Seed for resuming a run from a [`LoopCheckpoint`](crate::loop_checkpoint::LoopCheckpoint):
/// the cumulative metrics accrued before the crash/migration so the
/// resumed run continues accounting from where it left off instead of
/// re-starting at zero (which would under-report token usage and tool
/// calls in the resulting `AgentResult`).
#[derive(Default)]
pub(crate) struct ExecutionSeed {
    pub(crate) total_usage: TokenUsage,
    pub(crate) tool_calls_count: usize,
    pub(crate) verification_reports: Vec<VerificationReport>,
}

impl ExecutionLoopState {
    /// Convenience constructor with no checkpoint seed. Only used by
    /// unit tests now; production paths go through `new_seeded` (the
    /// resume path threads checkpoint metrics, the normal path passes
    /// `None`).
    #[cfg(test)]
    pub(super) fn new(history: &[Message]) -> Self {
        Self::new_seeded(history, None)
    }

    /// Build loop state, optionally pre-seeded with cumulative metrics
    /// from a checkpoint (see [`ExecutionSeed`]).
    pub(super) fn new_seeded(history: &[Message], seed: Option<ExecutionSeed>) -> Self {
        let seed = seed.unwrap_or_default();
        Self {
            messages: history.to_vec(),
            total_usage: seed.total_usage,
            tool_calls_count: seed.tool_calls_count,
            verification_reports: seed.verification_reports,
            turn: 0,
            parse_error_count: 0,
            continuation_count: 0,
            recent_tool_signatures: Vec::new(),
            execution_start: Instant::now(),
        }
    }

    pub(super) fn next_turn(&mut self) -> usize {
        self.turn += 1;
        self.turn
    }

    pub(super) fn current_turn(&self) -> usize {
        self.turn
    }

    pub(super) fn continuation_count(&self) -> u32 {
        self.continuation_count
    }

    pub(super) fn check_execution_timeout(&self, max_time_ms: Option<u64>) -> Option<String> {
        let max_time_ms = max_time_ms?;
        let elapsed_ms = self.execution_start.elapsed().as_millis() as u64;
        if elapsed_ms <= max_time_ms {
            return None;
        }

        Some(format!(
            "Execution timeout after {} seconds (limit: {} seconds). Completed {} turns.",
            elapsed_ms / 1000,
            max_time_ms / 1000,
            self.turn.saturating_sub(1)
        ))
    }

    pub(super) fn elapsed_ms(&self) -> u64 {
        self.execution_start.elapsed().as_millis() as u64
    }

    pub(super) fn turn_limit_error(&self, max_tool_rounds: usize) -> Option<String> {
        (self.turn > max_tool_rounds)
            .then(|| format!("Max tool rounds ({}) exceeded", max_tool_rounds))
    }

    pub(super) fn record_usage(&mut self, usage: &TokenUsage) {
        self.total_usage.prompt_tokens += usage.prompt_tokens;
        self.total_usage.completion_tokens += usage.completion_tokens;
        self.total_usage.total_tokens += usage.total_tokens;
    }

    pub(super) fn record_tool_call(&mut self) {
        self.tool_calls_count += 1;
    }

    pub(super) fn duplicate_tool_call(
        &self,
        tool_name: &str,
        args: &Value,
        threshold: u32,
    ) -> Option<(usize, String)> {
        let signature = Self::tool_signature(tool_name, args);
        let duplicate_count = self
            .recent_tool_signatures
            .iter()
            .filter(|sig| sig.starts_with(&signature))
            .count();

        if duplicate_count < threshold as usize {
            return None;
        }

        Some((
            duplicate_count,
            format!(
                "Tool '{}' has been called {} times with identical arguments. \
                 Aborting to prevent infinite loop. Consider modifying your approach.",
                tool_name, duplicate_count
            ),
        ))
    }

    pub(super) fn record_parse_error(
        &mut self,
        parse_error: &str,
        max_parse_retries: u32,
    ) -> ParseErrorOutcome {
        self.parse_error_count += 1;
        let output = format!("Error: {}", parse_error);
        let fatal_message = (self.parse_error_count > max_parse_retries).then(|| {
            format!(
                "LLM produced malformed tool arguments {} time(s) in a row \
                 (max_parse_retries={}); giving up",
                self.parse_error_count, max_parse_retries
            )
        });

        ParseErrorOutcome {
            output,
            count: self.parse_error_count,
            fatal_message,
        }
    }

    pub(super) fn reset_parse_errors(&mut self) {
        self.parse_error_count = 0;
    }

    pub(super) fn recent_tool_signatures(&self) -> Vec<String> {
        self.recent_tool_signatures.clone()
    }

    pub(super) fn remember_tool_signature(
        &mut self,
        tool_name: &str,
        args: &Value,
        is_error: bool,
    ) {
        self.recent_tool_signatures.push(format!(
            "{}:{} => {}",
            tool_name,
            serde_json::to_string(args).unwrap_or_default(),
            if is_error { "error" } else { "ok" }
        ));

        if self.recent_tool_signatures.len() > RECENT_TOOL_SIGNATURE_LIMIT {
            let overflow = self.recent_tool_signatures.len() - RECENT_TOOL_SIGNATURE_LIMIT;
            self.recent_tool_signatures.drain(0..overflow);
        }
    }

    pub(super) fn should_inject_continuation(
        &mut self,
        looks_incomplete: bool,
        enabled: bool,
        max_continuation_turns: u32,
        max_tool_rounds: usize,
    ) -> bool {
        if enabled
            && self.continuation_count < max_continuation_turns
            && self.turn < max_tool_rounds
            && looks_incomplete
        {
            self.continuation_count += 1;
            return true;
        }

        false
    }

    pub(super) fn finish(self, text: String) -> AgentResult {
        AgentResult {
            text,
            messages: self.messages,
            usage: self.total_usage,
            tool_calls_count: self.tool_calls_count,
            verification_reports: self.verification_reports,
        }
    }

    /// Build a result from a turn that was cancelled mid-generation. Keeps the
    /// conversation accumulated so far (the user's message above all) so the next
    /// turn remembers it. Appends a short assistant marker when the log would
    /// otherwise end on a user message, so the next user turn still alternates.
    pub(super) fn finish_interrupted(mut self) -> AgentResult {
        let ends_on_user = self
            .messages
            .last()
            .map(|m| m.role != "assistant")
            .unwrap_or(false);
        if ends_on_user {
            self.messages.push(Message {
                role: "assistant".to_string(),
                content: vec![ContentBlock::Text {
                    text: "(Response interrupted by the user.)".to_string(),
                }],
                reasoning_content: None,
            });
        }
        AgentResult {
            text: String::new(),
            messages: self.messages,
            usage: self.total_usage,
            tool_calls_count: self.tool_calls_count,
            verification_reports: self.verification_reports,
        }
    }

    fn tool_signature(tool_name: &str, args: &Value) -> String {
        format!(
            "{}:{}",
            tool_name,
            serde_json::to_string(args).unwrap_or_default()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn finish_interrupted_keeps_user_message_and_alternates() {
        // A cancelled turn must keep the user's message (so the next turn
        // remembers it) and end on an assistant message (so it still alternates).
        let mut state = ExecutionLoopState::new(&[]);
        state.messages.push(Message::user("what is the plan?"));
        let result = state.finish_interrupted();
        assert!(
            result
                .messages
                .iter()
                .any(|m| m.role == "user" && m.text().contains("what is the plan?")),
            "user message must survive the interrupt"
        );
        assert_eq!(
            result.messages.last().unwrap().role,
            "assistant",
            "history must end on an assistant message to alternate"
        );
    }

    #[test]
    fn duplicate_tool_call_uses_recent_success_and_error_signatures() {
        let mut state = ExecutionLoopState::new(&[]);
        let args = json!({"path":"src/lib.rs"});

        state.remember_tool_signature("read_file", &args, false);
        state.remember_tool_signature("read_file", &args, true);

        let duplicate = state.duplicate_tool_call("read_file", &args, 2).unwrap();
        assert_eq!(duplicate.0, 2);
        assert!(duplicate.1.contains("read_file"));
    }

    #[test]
    fn parse_error_becomes_fatal_after_retry_budget() {
        let mut state = ExecutionLoopState::new(&[]);

        let first = state.record_parse_error("bad json", 1);
        assert_eq!(first.count, 1);
        assert!(first.fatal_message.is_none());

        let second = state.record_parse_error("bad json", 1);
        assert_eq!(second.count, 2);
        assert!(second
            .fatal_message
            .unwrap()
            .contains("max_parse_retries=1"));
    }

    #[test]
    fn continuation_budget_is_consumed_only_when_incomplete() {
        let mut state = ExecutionLoopState::new(&[]);
        state.next_turn();

        assert!(state.should_inject_continuation(true, true, 1, 5));
        assert!(!state.should_inject_continuation(true, true, 1, 5));
        assert!(!state.should_inject_continuation(false, true, 2, 5));
    }
}
