use super::execution_state::ExecutionLoopState;
use super::tool_result_runtime::{push_tool_result_message, NormalizedToolResult};
use super::{AgentEvent, AgentLoop};
use crate::llm::ToolCall;
use std::time::Instant;
use tokio::sync::mpsc;

pub(super) struct ToolCompletionInput<'a> {
    pub(super) tool_call: &'a ToolCall,
    pub(super) event_tx: &'a Option<mpsc::Sender<AgentEvent>>,
    pub(super) session_id: Option<&'a str>,
    pub(super) effective_prompt: &'a str,
    pub(super) tool_start: Instant,
    pub(super) normalized: NormalizedToolResult,
}

impl AgentLoop {
    pub(super) async fn complete_tool_call(
        &self,
        state: &mut ExecutionLoopState,
        input: ToolCompletionInput<'_>,
    ) {
        let ToolCompletionInput {
            tool_call,
            event_tx,
            session_id,
            effective_prompt,
            tool_start,
            normalized,
        } = input;
        let tool_duration = tool_start.elapsed();
        crate::telemetry::record_tool_result(normalized.exit_code, tool_duration);
        Self::collect_verification_report(&mut state.verification_reports, &normalized.metadata);

        let output = if let Some(ref sp) = self.config.security_provider {
            sp.sanitize_output(&normalized.output)
        } else {
            normalized.output.clone()
        };

        state.remember_tool_signature(&tool_call.name, &tool_call.args, normalized.is_error);

        self.fire_post_tool_use(
            session_id.unwrap_or(""),
            &tool_call.name,
            &tool_call.args,
            &output,
            normalized.exit_code == 0,
            tool_duration.as_millis() as u64,
        )
        .await;

        self.config.rl_trajectory_recorder.record_tool_result(
            session_id.unwrap_or(""),
            state.current_turn(),
            &tool_call.id,
            &tool_call.name,
            &output,
            normalized.exit_code,
            tool_duration.as_millis() as u64,
            &normalized.metadata,
            normalized
                .error_kind
                .as_ref()
                .map(|kind| format!("{kind:?}")),
        );

        self.remember_tool_result(
            effective_prompt,
            &tool_call.name,
            &output,
            normalized.exit_code,
            event_tx,
        )
        .await;

        if let Some(tx) = event_tx {
            tx.send(AgentEvent::ToolEnd {
                id: tool_call.id.clone(),
                name: tool_call.name.clone(),
                output: output.clone(),
                exit_code: normalized.exit_code,
                metadata: normalized.metadata.clone(),
                error_kind: normalized.error_kind.clone(),
            })
            .await
            .ok();
        }

        push_tool_result_message(
            state,
            &tool_call.id,
            &output,
            normalized.is_error,
            normalized.images,
        );
    }
}
