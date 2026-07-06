use super::execution_state::ExecutionLoopState;
use super::{AgentEvent, AgentLoop};
use crate::llm::{Message, ToolCall};
use tokio::sync::mpsc;

impl AgentLoop {
    /// Handles pre-execution guards that feed an immediate tool result back to the LLM.
    ///
    /// Returns `true` when the tool call has been fully handled and should not continue
    /// through safety gating or execution.
    pub(super) async fn handle_tool_preflight_guard(
        &self,
        tool_call: &ToolCall,
        state: &mut ExecutionLoopState,
        event_tx: &Option<mpsc::Sender<AgentEvent>>,
        session_id: Option<&str>,
    ) -> anyhow::Result<bool> {
        if let Some((duplicate_count, error_msg)) = state.duplicate_tool_call(
            &tool_call.name,
            &tool_call.args,
            self.config.duplicate_tool_call_threshold,
        ) {
            tracing::warn!(
                tool_name = tool_call.name.as_str(),
                duplicate_count = duplicate_count,
                threshold = self.config.duplicate_tool_call_threshold,
                "Duplicate tool call threshold exceeded"
            );

            if let Some(tx) = event_tx {
                tx.send(AgentEvent::Error {
                    message: error_msg.clone(),
                })
                .await
                .ok();
            }

            state
                .messages
                .push(Message::tool_result(&tool_call.id, &error_msg, true));
            self.config.rl_trajectory_recorder.record_tool_result(
                session_id.unwrap_or(""),
                state.current_turn(),
                &tool_call.id,
                &tool_call.name,
                &error_msg,
                1,
                0,
                &None,
                Some("duplicate_tool_call".to_string()),
            );
            return Ok(true);
        }

        if let Some(parse_error) = tool_call.args.get("__parse_error").and_then(|v| v.as_str()) {
            let parse_outcome =
                state.record_parse_error(parse_error, self.config.max_parse_retries);
            tracing::warn!(
                tool = tool_call.name.as_str(),
                parse_error_count = parse_outcome.count,
                max_parse_retries = self.config.max_parse_retries,
                "Malformed tool arguments from LLM"
            );

            if let Some(tx) = event_tx {
                tx.send(AgentEvent::ToolEnd {
                    id: tool_call.id.clone(),
                    name: tool_call.name.clone(),
                    output: parse_outcome.output.clone(),
                    exit_code: 1,
                    metadata: None,
                    error_kind: None,
                })
                .await
                .ok();
            }

            state.messages.push(Message::tool_result(
                &tool_call.id,
                &parse_outcome.output,
                true,
            ));
            self.config.rl_trajectory_recorder.record_tool_result(
                session_id.unwrap_or(""),
                state.current_turn(),
                &tool_call.id,
                &tool_call.name,
                &parse_outcome.output,
                1,
                0,
                &None,
                Some("parse_error".to_string()),
            );

            if let Some(msg) = parse_outcome.fatal_message {
                tracing::error!("{}", msg);
                if let Some(tx) = event_tx {
                    tx.send(AgentEvent::Error {
                        message: msg.clone(),
                    })
                    .await
                    .ok();
                }
                anyhow::bail!(msg);
            }
            return Ok(true);
        }

        state.reset_parse_errors();
        Ok(false)
    }
}
