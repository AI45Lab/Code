use super::{AgentEvent, AgentLoop, AgentResult};
use crate::llm::Message;
use crate::planning::{AgentGoal, ExecutionPlan, LlmPlanner, PreAnalysis};
use anyhow::Result;
use tokio::sync::mpsc;

impl AgentLoop {
    pub(super) fn preserve_plan_goal_context(
        mut plan: ExecutionPlan,
        execution_prompt: &str,
    ) -> ExecutionPlan {
        let context = execution_prompt.trim();
        let goal = plan.goal.trim();

        if context.is_empty() || goal == context || goal.contains(context) {
            return plan;
        }

        plan.goal = format!(
            "Original user request and planning context:\n{context}\n\nPlanner goal:\n{goal}"
        );
        plan
    }

    pub(super) async fn emit_task_updated(
        &self,
        event_tx: &Option<mpsc::Sender<AgentEvent>>,
        session_id: &str,
        plan: &ExecutionPlan,
    ) {
        if let Some(tx) = event_tx {
            tx.send(AgentEvent::TaskUpdated {
                session_id: session_id.to_string(),
                tasks: plan.steps.clone(),
            })
            .await
            .ok();
        }
    }

    /// Create an execution plan for a prompt
    ///
    /// Delegates to [`LlmPlanner`] for structured JSON plan generation,
    /// falling back to heuristic planning if the LLM call fails.
    pub async fn plan(&self, prompt: &str, _context: Option<&str>) -> Result<ExecutionPlan> {
        match LlmPlanner::create_plan(&self.llm_client, prompt).await {
            Ok(plan) => Ok(plan),
            Err(e) => {
                tracing::warn!("LLM plan creation failed, using fallback: {}", e);
                Ok(LlmPlanner::fallback_plan(prompt))
            }
        }
    }

    /// Execute with planning phase.
    ///
    /// If `pre_analysis` is provided (from a single pre-analysis LLM call in
    /// `execute_with_session`), the goal and plan are already available and no
    /// additional LLM calls are needed for planning. Otherwise, falls back to
    /// calling `extract_goal` and `plan` individually.
    pub async fn execute_with_planning(
        &self,
        history: &[Message],
        prompt: &str,
        session_id: Option<&str>,
        event_tx: Option<mpsc::Sender<AgentEvent>>,
        pre_analysis: Option<PreAnalysis>,
    ) -> Result<AgentResult> {
        // Send planning start event
        if let Some(tx) = &event_tx {
            tx.send(AgentEvent::PlanningStart {
                prompt: prompt.to_string(),
            })
            .await
            .ok();
        }

        // Use pre-analysis result if available (goal + plan already computed in one LLM call).
        let (goal, plan) = if let Some(analysis) = pre_analysis {
            (
                Some(analysis.goal.clone()),
                Self::preserve_plan_goal_context(analysis.execution_plan.clone(), prompt),
            )
        } else {
            // Fall back: extract goal and create plan via separate LLM calls.
            let g = if self.config.goal_tracking {
                Some(self.extract_goal(prompt).await?)
            } else {
                None
            };
            let p = self.plan(prompt, None).await?;
            (g, p)
        };

        // Send GoalExtracted event if goal_tracking is enabled.
        if self.config.goal_tracking {
            if let Some(ref g) = goal {
                if let Some(tx) = &event_tx {
                    tx.send(AgentEvent::GoalExtracted { goal: g.clone() })
                        .await
                        .ok();
                }
            }
        }

        // Send planning end event
        if let Some(tx) = &event_tx {
            tx.send(AgentEvent::PlanningEnd {
                estimated_steps: plan.steps.len(),
                plan: plan.clone(),
            })
            .await
            .ok();
        }

        let plan_start = std::time::Instant::now();

        // Execute the plan step by step
        let result = self
            .execute_plan(history, &plan, session_id, event_tx.clone())
            .await?;

        // Emit the final End event (execute_loop_inner does not emit End in planning mode)
        if let Some(tx) = &event_tx {
            tx.send(AgentEvent::End {
                text: result.text.clone(),
                usage: result.usage.clone(),
                verification_summary: Box::new(result.verification_summary()),
                meta: None,
            })
            .await
            .ok();
        }

        // Check goal achievement when goal_tracking is enabled
        if self.config.goal_tracking {
            if let Some(ref g) = goal {
                let achieved = self.check_goal_achievement(g, &result.text).await?;
                if achieved {
                    if let Some(tx) = &event_tx {
                        tx.send(AgentEvent::GoalAchieved {
                            goal: g.description.clone(),
                            total_steps: result.messages.len(),
                            duration_ms: plan_start.elapsed().as_millis() as i64,
                        })
                        .await
                        .ok();
                    }
                }
            }
        }

        Ok(result)
    }

    /// Extract goal from prompt
    ///
    /// Delegates to [`LlmPlanner`] for structured JSON goal extraction,
    /// falling back to heuristic logic if the LLM call fails.
    pub async fn extract_goal(&self, prompt: &str) -> Result<AgentGoal> {
        match LlmPlanner::extract_goal(&self.llm_client, prompt).await {
            Ok(goal) => Ok(goal),
            Err(e) => {
                tracing::warn!("LLM goal extraction failed, using fallback: {}", e);
                Ok(LlmPlanner::fallback_goal(prompt))
            }
        }
    }

    /// Check if goal is achieved
    ///
    /// Delegates to [`LlmPlanner`] for structured JSON achievement check,
    /// falling back to heuristic logic if the LLM call fails.
    pub async fn check_goal_achievement(
        &self,
        goal: &AgentGoal,
        current_state: &str,
    ) -> Result<bool> {
        match LlmPlanner::check_achievement(&self.llm_client, goal, current_state).await {
            Ok(result) => Ok(result.achieved),
            Err(e) => {
                tracing::warn!("LLM achievement check failed, using fallback: {}", e);
                let result = LlmPlanner::fallback_check_achievement(goal, current_state);
                Ok(result.achieved)
            }
        }
    }
}
