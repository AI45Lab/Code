use super::execution_state::ExecutionLoopState;
use super::tool_completion_runtime::ToolCompletionInput;
use super::{AgentEvent, AgentLoop};
use crate::llm::ToolCall;
use tokio::sync::mpsc;

impl AgentLoop {
    pub(super) async fn execute_tool_turn(
        &self,
        tool_calls: Vec<ToolCall>,
        state: &mut ExecutionLoopState,
        event_tx: &Option<mpsc::Sender<AgentEvent>>,
        session_id: Option<&str>,
        effective_prompt: &str,
    ) -> anyhow::Result<()> {
        if self.can_run_parallel_write_batch(&tool_calls) {
            self.execute_parallel_write_batch(&tool_calls, state, event_tx)
                .await;
            return Ok(());
        }

        for tool_call in tool_calls {
            self.execute_sequential_tool_call(
                tool_call,
                state,
                event_tx,
                session_id,
                effective_prompt,
            )
            .await?;
        }

        Ok(())
    }

    async fn execute_sequential_tool_call(
        &self,
        tool_call: ToolCall,
        state: &mut ExecutionLoopState,
        event_tx: &Option<mpsc::Sender<AgentEvent>>,
        session_id: Option<&str>,
        effective_prompt: &str,
    ) -> anyhow::Result<()> {
        state.record_tool_call();
        let tool_start = std::time::Instant::now();
        let turn = state.current_turn();
        self.config.rl_trajectory_recorder.record_tool_call(
            session_id.unwrap_or(""),
            turn,
            &tool_call,
        );

        tracing::info!(
            tool_name = tool_call.name.as_str(),
            tool_id = tool_call.id.as_str(),
            "Tool execution started"
        );

        if self
            .handle_tool_preflight_guard(&tool_call, state, event_tx, session_id)
            .await?
        {
            return Ok(());
        }

        let gate_decision = self.decide_tool_gate(&tool_call, state, session_id).await;

        let normalized = self
            .resolve_tool_gate_decision(gate_decision, &tool_call, event_tx)
            .await;

        self.complete_tool_call(
            state,
            ToolCompletionInput {
                tool_call: &tool_call,
                event_tx,
                session_id,
                effective_prompt,
                tool_start,
                normalized,
            },
        )
        .await;
        Ok(())
    }
}
