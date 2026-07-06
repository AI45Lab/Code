use super::{AgentEvent, AgentLoop, AgentResult};
use crate::hooks::ErrorType;
use crate::llm::Message;
use crate::planning::{LlmPlanner, PreAnalysis};
use crate::prompts::{AgentStyle, PlanningMode};
use anyhow::Result;
use tokio::sync::mpsc;

struct ExecutionRoute {
    style: AgentStyle,
    use_planning: bool,
    effective_prompt: String,
    pre_analysis: Option<PreAnalysis>,
}

impl AgentLoop {
    pub(super) fn preserve_original_prompt_for_execution(
        original_prompt: &str,
        optimized_input: &str,
    ) -> String {
        let original = original_prompt.trim();
        let optimized = optimized_input.trim();

        if original.is_empty() {
            return optimized.to_string();
        }
        if optimized.is_empty() || optimized == original {
            return original.to_string();
        }
        if optimized.contains(original) {
            return optimized.to_string();
        }

        format!("Original user request:\n{original}\n\nPlanner-optimized request:\n{optimized}")
    }

    pub(super) fn should_run_pre_analysis(&self) -> bool {
        match self.config.planning_mode {
            PlanningMode::Disabled => false,
            PlanningMode::Enabled => true,
            PlanningMode::Auto => true,
        }
    }

    /// Execute the agent loop for a prompt with session context
    ///
    /// Takes the conversation history, user prompt, and optional session ID.
    /// When session_id is provided, context providers can use it for session-specific context.
    pub async fn execute_with_session(
        &self,
        history: &[Message],
        prompt: &str,
        session_id: Option<&str>,
        event_tx: Option<mpsc::Sender<AgentEvent>>,
        cancel_token: Option<&tokio_util::sync::CancellationToken>,
    ) -> Result<AgentResult> {
        let default_token = tokio_util::sync::CancellationToken::new();
        let token = cancel_token.unwrap_or(&default_token);
        tracing::info!(
            a3s.session.id = session_id.unwrap_or("none"),
            a3s.agent.max_turns = self.config.max_tool_rounds,
            "a3s.agent.execute started"
        );

        let route = self.resolve_execution_route(prompt).await;
        let mut effective_prompt = route.effective_prompt.clone();
        let mut auto_tool_calls_count = 0;
        if !route.use_planning {
            if let Some(outcome) = self
                .maybe_apply_auto_delegation(&effective_prompt, session_id, &event_tx)
                .await?
            {
                effective_prompt = outcome.prompt;
                auto_tool_calls_count = outcome.tool_calls_count;
            }
        }

        let mut result = if route.use_planning {
            self.execute_with_planning(
                history,
                &effective_prompt,
                session_id,
                event_tx,
                route.pre_analysis,
            )
            .await
        } else {
            self.execute_loop(
                history,
                &effective_prompt,
                route.style,
                session_id,
                event_tx,
                token,
                true,
            )
            .await
        };
        if let Ok(result) = &mut result {
            result.tool_calls_count += auto_tool_calls_count;
        }

        self.record_execution_result(session_id, &result).await;
        result
    }

    async fn resolve_execution_route(&self, prompt: &str) -> ExecutionRoute {
        let pre_analysis = self.run_pre_analysis(prompt).await;
        let style = self.resolve_execution_style(prompt, pre_analysis.as_ref());
        let use_planning = self.resolve_planning_decision(style, pre_analysis.as_ref());
        let effective_prompt = pre_analysis
            .as_ref()
            .map(|analysis| {
                Self::preserve_original_prompt_for_execution(prompt, &analysis.optimized_input)
            })
            .unwrap_or_else(|| prompt.to_string());

        ExecutionRoute {
            style,
            use_planning,
            effective_prompt,
            pre_analysis,
        }
    }

    async fn run_pre_analysis(&self, prompt: &str) -> Option<PreAnalysis> {
        if !self.should_run_pre_analysis() {
            return None;
        }

        match LlmPlanner::pre_analyze(&self.llm_client.clone(), prompt).await {
            Ok(analysis) => {
                tracing::debug!(
                    intent = ?analysis.intent,
                    requires_planning = analysis.requires_planning,
                    plan_steps = analysis.execution_plan.steps.len(),
                    "Pre-analysis completed"
                );
                Some(analysis)
            }
            Err(e) => {
                tracing::warn!(error = %e, "Pre-analysis failed; using local style fallback");
                None
            }
        }
    }

    fn resolve_execution_style(
        &self,
        prompt: &str,
        pre_analysis: Option<&PreAnalysis>,
    ) -> AgentStyle {
        if let Some(analysis) = pre_analysis {
            return analysis.intent;
        }

        let (style, confidence) = AgentStyle::detect_with_confidence(prompt);
        tracing::debug!(
            intent.classification = ?style,
            intent.confidence = ?confidence,
            intent.source = "local_fallback",
            "Intent classified locally"
        );
        style
    }

    fn resolve_planning_decision(
        &self,
        style: AgentStyle,
        pre_analysis: Option<&PreAnalysis>,
    ) -> bool {
        match self.config.planning_mode {
            PlanningMode::Disabled => false,
            PlanningMode::Enabled => true,
            PlanningMode::Auto => pre_analysis
                .map(|analysis| analysis.requires_planning)
                .unwrap_or_else(|| style.requires_planning()),
        }
    }

    async fn record_execution_result(
        &self,
        session_id: Option<&str>,
        result: &Result<AgentResult>,
    ) {
        match result {
            Ok(r) => {
                tracing::info!(
                    a3s.agent.tool_calls_count = r.tool_calls_count,
                    a3s.llm.total_tokens = r.usage.total_tokens,
                    "a3s.agent.execute completed"
                );
                self.config.rl_trajectory_recorder.record_execution_end(
                    session_id.unwrap_or(""),
                    true,
                    Some(&r.text),
                    Some(&r.usage),
                    Some(r.tool_calls_count),
                    None,
                );
                self.fire_post_response(
                    session_id.unwrap_or(""),
                    &r.text,
                    r.tool_calls_count,
                    &r.usage,
                    0,
                )
                .await;
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "a3s.agent.execute failed"
                );
                self.config.rl_trajectory_recorder.record_execution_end(
                    session_id.unwrap_or(""),
                    false,
                    None,
                    None,
                    None,
                    Some(&e.to_string()),
                );
                self.fire_on_error(
                    session_id.unwrap_or(""),
                    ErrorType::Other,
                    &e.to_string(),
                    serde_json::json!({"phase": "execute"}),
                )
                .await;
            }
        }
    }
}
