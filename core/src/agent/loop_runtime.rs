use super::completion_runtime::CompletionFlow;
use super::execution_state::ExecutionLoopState;
use super::queue_forwarder::QueueEventForwarder;
use super::{AgentEvent, AgentLoop, AgentResult};
use crate::llm::Message;
use crate::prompts::AgentStyle;
use anyhow::Result;
use tokio::sync::mpsc;

impl AgentLoop {
    /// Core execution loop (without planning routing).
    ///
    /// This is the inner loop that runs LLM calls and tool executions.
    /// Called directly by `execute_with_session` (after planning check)
    /// and by `execute_plan` (for individual steps, bypassing planning).
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn execute_loop(
        &self,
        history: &[Message],
        prompt: &str,
        effective_style: AgentStyle,
        session_id: Option<&str>,
        event_tx: Option<mpsc::Sender<AgentEvent>>,
        cancel_token: &tokio_util::sync::CancellationToken,
        emit_end: bool,
    ) -> Result<AgentResult> {
        // When called via execute_loop, the prompt is used for both
        // message-adding and hook/memory/event purposes.
        self.execute_loop_inner(
            history,
            prompt,
            prompt,
            Some(effective_style),
            session_id,
            event_tx,
            cancel_token,
            emit_end,
            None,
        )
        .await
    }

    /// Inner execution loop.
    ///
    /// `msg_prompt` controls whether a user message is appended (empty = skip).
    /// `effective_prompt` is used for hooks, memory recall, taint tracking, and events.
    /// `effective_style` pre-computed style to skip redundant LLM-based intent detection.
    /// `emit_end` controls whether to send `AgentEvent::End` when the loop completes
    /// (should be false when called from `execute_plan` to avoid duplicate End events).
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn execute_loop_inner(
        &self,
        history: &[Message],
        msg_prompt: &str,
        effective_prompt: &str,
        effective_style: Option<AgentStyle>,
        session_id: Option<&str>,
        event_tx: Option<mpsc::Sender<AgentEvent>>,
        cancel_token: &tokio_util::sync::CancellationToken,
        emit_end: bool,
        seed: Option<super::execution_state::ExecutionSeed>,
    ) -> Result<AgentResult> {
        let mut state = ExecutionLoopState::new_seeded(history, seed);

        let style_prompt = if effective_prompt.is_empty() {
            msg_prompt
        } else {
            effective_prompt
        };
        let prompt_mode = self
            .resolve_prompt_mode(effective_style, style_prompt, &event_tx)
            .await;
        let effective_system_prompt = prompt_mode.system_prompt;

        // Send start event
        if let Some(tx) = &event_tx {
            tx.send(AgentEvent::Start {
                prompt: effective_prompt.to_string(),
            })
            .await
            .ok();
        }

        let _queue_forwarder = QueueEventForwarder::start(
            self.command_queue.as_ref(),
            event_tx.as_ref(),
            cancel_token,
        );

        let turn_context = self
            .prepare_turn_context(
                &effective_system_prompt,
                effective_prompt,
                state.messages.len(),
                session_id,
                &event_tx,
            )
            .await;
        let effective_prompt = turn_context.effective_prompt.as_str();
        let augmented_system = turn_context.augmented_system;

        self.config.rl_trajectory_recorder.record_execution_start(
            session_id.unwrap_or(""),
            &self.tool_context.workspace,
            effective_prompt,
            history,
            augmented_system.as_deref(),
            self.config.max_tool_rounds,
            &format!("{:?}", self.config.planning_mode),
        );

        // Add user message
        if !msg_prompt.is_empty() {
            state.messages.push(Message::user(msg_prompt));
        }

        loop {
            let llm_turn = match self
                .execute_llm_turn(
                    &mut state,
                    &augmented_system,
                    effective_prompt,
                    session_id,
                    &event_tx,
                    cancel_token,
                )
                .await
            {
                Ok(turn) => turn,
                // Interrupted mid-generation (Esc / cancel): keep the conversation
                // accumulated so far — above all the user's message — and return it
                // as the result so it is committed to history. Without this the
                // whole turn is dropped and the agent "forgets" what was just asked
                // when the user continues.
                Err(_) if cancel_token.is_cancelled() => {
                    return Ok(state.finish_interrupted());
                }
                Err(e) => return Err(e),
            };
            let turn = llm_turn.turn;
            let response = llm_turn.response;
            let tool_calls = llm_turn.tool_calls;

            if tool_calls.is_empty() {
                match self
                    .complete_no_tool_response(
                        &mut state,
                        turn,
                        &response,
                        effective_prompt,
                        session_id,
                        &event_tx,
                        emit_end,
                    )
                    .await
                {
                    CompletionFlow::Continue => continue,
                    CompletionFlow::Finished(final_text) => return Ok(state.finish(final_text)),
                }
            }

            self.execute_tool_turn(
                tool_calls,
                &mut state,
                &event_tx,
                session_id,
                effective_prompt,
            )
            .await?;

            // Quiescent boundary: the tool round has fully resolved and
            // `state.messages` is consistent. Persist a checkpoint so a
            // future process can resume from here (P3).
            self.persist_loop_checkpoint(turn, &state, session_id).await;
        }
    }

    /// Persist a `LoopCheckpoint` if both a sink and a bound run id are
    /// configured. Failures are swallowed (the sink already logs them)
    /// so an unavailable store cannot halt a live run.
    async fn persist_loop_checkpoint(
        &self,
        turn: usize,
        state: &super::execution_state::ExecutionLoopState,
        session_id: Option<&str>,
    ) {
        let Some(sink) = self.checkpoint_sink.as_ref() else {
            return;
        };
        let Some(run_id) = self.checkpoint_run_id.as_ref() else {
            return;
        };
        let checkpoint = crate::loop_checkpoint::LoopCheckpoint {
            schema_version: crate::loop_checkpoint::LOOP_CHECKPOINT_SCHEMA_VERSION,
            run_id: run_id.clone(),
            session_id: session_id.unwrap_or("").to_string(),
            turn,
            messages: state.messages.clone(),
            total_usage: state.total_usage.clone(),
            tool_calls_count: state.tool_calls_count,
            verification_reports: state.verification_reports.clone(),
            checkpoint_ms: self.config.host_env.now_ms(),
        };
        sink.save_checkpoint(&checkpoint).await;
    }
}
