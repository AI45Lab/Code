use super::execution_state::ExecutionLoopState;
use super::{AgentEvent, AgentLoop};
use crate::hooks::{
    ErrorType, GenerateEndEvent, GenerateStartEvent, HookEvent, TokenUsageInfo, ToolCallInfo,
};
use crate::llm::{LlmResponse, Message, ToolCall, ToolDefinition};
use anyhow::Context;
use std::time::Duration;
use tokio::sync::mpsc;

pub(super) struct LlmTurnOutput {
    pub(super) turn: usize,
    pub(super) response: LlmResponse,
    pub(super) tool_calls: Vec<ToolCall>,
}

struct LlmCallRequest<'a> {
    turn: usize,
    messages: &'a [Message],
    system: Option<&'a str>,
    tools: &'a [ToolDefinition],
    session_id: Option<&'a str>,
    event_tx: &'a Option<mpsc::Sender<AgentEvent>>,
    cancel_token: &'a tokio_util::sync::CancellationToken,
}

impl AgentLoop {
    pub(super) async fn execute_llm_turn(
        &self,
        state: &mut ExecutionLoopState,
        augmented_system: &Option<String>,
        effective_prompt: &str,
        session_id: Option<&str>,
        event_tx: &Option<mpsc::Sender<AgentEvent>>,
        cancel_token: &tokio_util::sync::CancellationToken,
    ) -> anyhow::Result<LlmTurnOutput> {
        let turn = state.next_turn();
        self.ensure_turn_can_start(turn, state, event_tx).await?;
        self.emit_turn_start(turn, event_tx).await;

        tracing::info!(
            a3s.llm.streaming = event_tx.is_some(),
            "LLM completion started"
        );

        let selected_tools =
            crate::tools::select_tools_for_messages(&self.config.tools, &state.messages);
        self.config.rl_trajectory_recorder.record_llm_request(
            session_id.unwrap_or(""),
            turn,
            &state.messages,
            augmented_system.as_deref(),
            &selected_tools,
            estimate_prompt_tokens(&state.messages, augmented_system.as_deref()),
        );

        self.fire_generate_start(session_id.unwrap_or(""), effective_prompt, augmented_system)
            .await;

        let llm_start = std::time::Instant::now();
        let response = self
            .call_llm_with_circuit_breaker(LlmCallRequest {
                turn,
                messages: &state.messages,
                system: augmented_system.as_deref(),
                tools: &selected_tools,
                session_id,
                event_tx,
                cancel_token,
            })
            .await?;

        state.record_usage(&response.usage);
        self.complete_llm_turn(
            turn,
            effective_prompt,
            &response,
            llm_start,
            event_tx,
            session_id,
        )
        .await;

        state.messages.push(response.message.clone());
        let tool_calls = response.tool_calls();
        self.emit_turn_end(turn, &response, event_tx).await;
        self.maybe_auto_compact(state, &response, session_id, event_tx)
            .await;

        Ok(LlmTurnOutput {
            turn,
            response,
            tool_calls,
        })
    }

    async fn ensure_turn_can_start(
        &self,
        turn: usize,
        state: &ExecutionLoopState,
        event_tx: &Option<mpsc::Sender<AgentEvent>>,
    ) -> anyhow::Result<()> {
        if let Some(error) = state.check_execution_timeout(self.config.max_execution_time_ms) {
            tracing::warn!(
                elapsed_ms = state.elapsed_ms(),
                max_time_ms = self.config.max_execution_time_ms.unwrap_or_default(),
                turns = turn.saturating_sub(1),
                "Execution timeout exceeded"
            );
            self.emit_error(event_tx, error.clone()).await;
            anyhow::bail!(error);
        }

        if let Some(error) = state.turn_limit_error(self.config.max_tool_rounds) {
            self.emit_error(event_tx, error.clone()).await;
            anyhow::bail!(error);
        }

        Ok(())
    }

    async fn emit_turn_start(&self, turn: usize, event_tx: &Option<mpsc::Sender<AgentEvent>>) {
        if let Some(tx) = event_tx {
            tx.send(AgentEvent::TurnStart { turn }).await.ok();
        }

        tracing::info!(
            turn = turn,
            max_turns = self.config.max_tool_rounds,
            "Agent turn started"
        );
    }

    async fn call_llm_with_circuit_breaker(
        &self,
        request: LlmCallRequest<'_>,
    ) -> anyhow::Result<LlmResponse> {
        // Consult the host's BudgetGuard once per turn (not per retry).
        // A `Deny` bails out before the LLM is touched; a `SoftLimit`
        // surfaces a BudgetThresholdHit event and proceeds.
        if let Some(guard) = &self.config.budget_guard {
            let sid = request.session_id.unwrap_or("");
            let estimate = estimate_prompt_tokens(request.messages, request.system);
            match guard.check_before_llm(sid, estimate).await {
                crate::budget::BudgetDecision::Allow => {}
                crate::budget::BudgetDecision::SoftLimit {
                    resource,
                    consumed,
                    limit,
                    message,
                } => {
                    if let Some(tx) = request.event_tx {
                        let _ = tx
                            .send(AgentEvent::BudgetThresholdHit {
                                resource,
                                kind: "soft".to_string(),
                                consumed,
                                limit,
                                message,
                            })
                            .await;
                    }
                }
                crate::budget::BudgetDecision::Deny { resource, reason } => {
                    if let Some(tx) = request.event_tx {
                        let _ = tx
                            .send(AgentEvent::BudgetThresholdHit {
                                resource: resource.clone(),
                                kind: "hard".to_string(),
                                consumed: 0.0,
                                limit: 0.0,
                                message: Some(reason.clone()),
                            })
                            .await;
                    }
                    anyhow::bail!("Budget exhausted on '{resource}': {reason}");
                }
            }
        }

        let threshold = self.config.circuit_breaker_threshold.max(1);
        let mut attempt = 0u32;

        loop {
            attempt += 1;
            let result = self
                .call_llm(
                    request.messages,
                    request.system,
                    request.tools,
                    request.event_tx,
                    request.cancel_token,
                )
                .await;
            match result {
                Ok(response) => {
                    if let Some(guard) = &self.config.budget_guard {
                        guard
                            .record_after_llm(request.session_id.unwrap_or(""), &response.usage)
                            .await;
                    }
                    return Ok(response);
                }
                Err(error) if request.cancel_token.is_cancelled() => {
                    anyhow::bail!(error);
                }
                Err(error)
                    if attempt < threshold && (request.event_tx.is_none() || attempt == 1) =>
                {
                    tracing::warn!(
                        turn = request.turn,
                        attempt = attempt,
                        threshold = threshold,
                        error = %error,
                        "LLM call failed, will retry"
                    );
                    tokio::time::sleep(Duration::from_millis(100 * attempt as u64)).await;
                }
                Err(error) => {
                    let msg = if attempt > 1 {
                        format!(
                            "LLM circuit breaker triggered: failed after {} attempt(s): {}",
                            attempt, error
                        )
                    } else {
                        format!("LLM call failed: {}", error)
                    };
                    tracing::error!(turn = request.turn, attempt = attempt, "{}", msg);
                    self.fire_on_error(
                        request.session_id.unwrap_or(""),
                        ErrorType::LlmFailure,
                        &msg,
                        serde_json::json!({"turn": request.turn, "attempt": attempt}),
                    )
                    .await;
                    self.emit_error(request.event_tx, msg.clone()).await;
                    anyhow::bail!(msg);
                }
            }
        }
    }

    async fn complete_llm_turn(
        &self,
        turn: usize,
        effective_prompt: &str,
        response: &LlmResponse,
        llm_start: std::time::Instant,
        event_tx: &Option<mpsc::Sender<AgentEvent>>,
        session_id: Option<&str>,
    ) {
        let llm_duration = llm_start.elapsed();
        tracing::info!(
            turn = turn,
            streaming = event_tx.is_some(),
            prompt_tokens = response.usage.prompt_tokens,
            completion_tokens = response.usage.completion_tokens,
            total_tokens = response.usage.total_tokens,
            stop_reason = response.stop_reason.as_deref().unwrap_or("unknown"),
            duration_ms = llm_duration.as_millis() as u64,
            "LLM completion finished"
        );

        self.fire_generate_end(
            session_id.unwrap_or(""),
            effective_prompt,
            response,
            llm_duration.as_millis() as u64,
        )
        .await;

        crate::telemetry::record_llm_usage(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response.usage.total_tokens,
            response.stop_reason.as_deref(),
        );
        tracing::info!(
            turn = turn,
            a3s.llm.total_tokens = response.usage.total_tokens,
            "Turn token usage"
        );
        self.config.rl_trajectory_recorder.record_llm_response(
            session_id.unwrap_or(""),
            turn,
            response,
            llm_duration.as_millis() as u64,
        );
    }

    async fn emit_turn_end(
        &self,
        turn: usize,
        response: &LlmResponse,
        event_tx: &Option<mpsc::Sender<AgentEvent>>,
    ) {
        if let Some(tx) = event_tx {
            tx.send(AgentEvent::TurnEnd {
                turn,
                usage: response.usage.clone(),
            })
            .await
            .ok();
        }
    }

    async fn maybe_auto_compact(
        &self,
        state: &mut ExecutionLoopState,
        response: &LlmResponse,
        session_id: Option<&str>,
        event_tx: &Option<mpsc::Sender<AgentEvent>>,
    ) {
        if !self.config.auto_compact {
            return;
        }

        let used = response.usage.prompt_tokens;
        let max = self.config.max_context_tokens;
        let threshold = self.config.auto_compact_threshold;

        if !crate::compaction::should_auto_compact(used, max, threshold) {
            return;
        }

        let before_len = state.messages.len();
        let percent_before = used as f32 / max as f32;

        tracing::info!(
            used_tokens = used,
            max_tokens = max,
            percent = percent_before,
            threshold = threshold,
            "Auto-compact triggered"
        );

        if let Some(pruned) = crate::compaction::prune_tool_outputs(&state.messages) {
            state.messages = pruned;
            tracing::info!("Tool output pruning applied");
        }

        if let Ok(Some(compacted)) = crate::compaction::compact_messages(
            session_id.unwrap_or(""),
            &state.messages,
            &self.llm_client,
        )
        .await
        {
            state.messages = compacted;
        }

        self.config.rl_trajectory_recorder.record_context_compacted(
            session_id.unwrap_or(""),
            before_len,
            &state.messages,
            percent_before,
        );

        if let Some(tx) = event_tx {
            tx.send(AgentEvent::ContextCompacted {
                session_id: session_id.unwrap_or("").to_string(),
                before_messages: before_len,
                after_messages: state.messages.len(),
                percent_before,
            })
            .await
            .ok();
        }
    }

    async fn emit_error(&self, event_tx: &Option<mpsc::Sender<AgentEvent>>, message: String) {
        if let Some(tx) = event_tx {
            tx.send(AgentEvent::Error { message }).await.ok();
        }
    }

    /// Call the LLM, handling streaming vs non-streaming internally.
    ///
    /// Streaming events (`TextDelta`, `ToolStart`) are forwarded to `event_tx`
    /// as they arrive. Non-streaming mode simply awaits the complete response.
    ///
    /// Tool definitions are selected once per turn by the centralized tool selector.
    ///
    /// Returns `Err` on any LLM API failure. The circuit breaker in
    /// `execute_loop` wraps this call with retry logic for non-streaming mode.
    async fn call_llm(
        &self,
        messages: &[Message],
        system: Option<&str>,
        tools: &[ToolDefinition],
        event_tx: &Option<mpsc::Sender<AgentEvent>>,
        cancel_token: &tokio_util::sync::CancellationToken,
    ) -> anyhow::Result<LlmResponse> {
        if event_tx.is_some() {
            let mut stream_rx = match self
                .llm_client
                .complete_streaming(messages, system, tools, cancel_token.clone())
                .await
            {
                Ok(rx) => rx,
                Err(stream_error) => {
                    // Do not fall back to non-streaming if cancelled — propagate cancellation
                    if cancel_token.is_cancelled() {
                        anyhow::bail!("Operation cancelled by user");
                    }
                    tracing::warn!(
                        error = %stream_error,
                        "LLM streaming setup failed; falling back to non-streaming completion"
                    );
                    return self
                        .llm_client
                        .complete(messages, system, tools)
                        .await
                        .with_context(|| {
                            format!(
                                "LLM streaming call failed ({stream_error}); non-streaming fallback also failed"
                            )
                        });
                }
            };

            let mut final_response: Option<LlmResponse> = None;
            loop {
                tokio::select! {
                    _ = cancel_token.cancelled() => {
                        tracing::info!("🛑 LLM streaming cancelled by CancellationToken");
                        anyhow::bail!("Operation cancelled by user");
                    }
                    event = stream_rx.recv() => {
                        match event {
                            Some(crate::llm::StreamEvent::TextDelta(text)) => {
                                if let Some(tx) = event_tx {
                                    tx.send(AgentEvent::TextDelta { text }).await.ok();
                                }
                            }
                            Some(crate::llm::StreamEvent::ReasoningDelta(text)) => {
                                if let Some(tx) = event_tx {
                                    tx.send(AgentEvent::ReasoningDelta { text }).await.ok();
                                }
                            }
                            Some(crate::llm::StreamEvent::ToolUseStart { id, name }) => {
                                if let Some(tx) = event_tx {
                                    tx.send(AgentEvent::ToolStart { id, name }).await.ok();
                                }
                            }
                            Some(crate::llm::StreamEvent::ToolUseInputDelta(delta)) => {
                                if let Some(tx) = event_tx {
                                    tx.send(AgentEvent::ToolInputDelta { delta }).await.ok();
                                }
                            }
                            Some(crate::llm::StreamEvent::Done(resp)) => {
                                final_response = Some(resp);
                                break;
                            }
                            None => break,
                        }
                    }
                }
            }
            final_response.context("Stream ended without final response")
        } else {
            self.llm_client
                .complete(messages, system, tools)
                .await
                .context("LLM call failed")
        }
    }

    /// Fire GenerateStart hook event before an LLM call.
    async fn fire_generate_start(
        &self,
        session_id: &str,
        prompt: &str,
        system_prompt: &Option<String>,
    ) {
        if let Some(he) = &self.config.hook_engine {
            let event = HookEvent::GenerateStart(GenerateStartEvent {
                session_id: session_id.to_string(),
                prompt: prompt.to_string(),
                system_prompt: system_prompt.clone(),
                model_provider: String::new(),
                model_name: String::new(),
                available_tools: self.config.tools.iter().map(|t| t.name.clone()).collect(),
            });
            let _ = he.fire(&event).await;
        }
    }

    /// Fire GenerateEnd hook event after an LLM call.
    async fn fire_generate_end(
        &self,
        session_id: &str,
        prompt: &str,
        response: &LlmResponse,
        duration_ms: u64,
    ) {
        if let Some(he) = &self.config.hook_engine {
            let tool_calls: Vec<ToolCallInfo> = response
                .tool_calls()
                .iter()
                .map(|tc| {
                    let args = if tc.args.is_null() {
                        serde_json::Value::Object(Default::default())
                    } else {
                        tc.args.clone()
                    };
                    ToolCallInfo {
                        name: tc.name.clone(),
                        args,
                    }
                })
                .collect();

            let event = HookEvent::GenerateEnd(GenerateEndEvent {
                session_id: session_id.to_string(),
                prompt: prompt.to_string(),
                response_text: response.text().to_string(),
                tool_calls,
                usage: TokenUsageInfo {
                    prompt_tokens: response.usage.prompt_tokens as i32,
                    completion_tokens: response.usage.completion_tokens as i32,
                    total_tokens: response.usage.total_tokens as i32,
                },
                duration_ms,
            });
            let _ = he.fire(&event).await;
        }
    }
}

/// Cheap, framework-internal estimator of prompt tokens for
/// `BudgetGuard::check_before_llm`. Roughly counts characters / 4
/// across system + messages, matching the well-known "1 token ≈ 4
/// English characters" heuristic. Impls that need precision should
/// rely on `record_after_llm` with the provider's actual usage.
fn estimate_prompt_tokens(messages: &[Message], system: Option<&str>) -> usize {
    let mut chars = system.map(|s| s.len()).unwrap_or(0);
    for msg in messages {
        chars += msg.text().len();
    }
    chars / 4
}
