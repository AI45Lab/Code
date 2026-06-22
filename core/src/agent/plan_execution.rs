use super::{AgentEvent, AgentLoop};
use crate::agent::AgentResult;
use crate::llm::{Message, TokenUsage};
use crate::planning::{ExecutionPlan, Task, TaskStatus};
use crate::prompts::AgentStyle;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc;

/// Result of a single parallel step execution, emitted as structured JSON
/// so the frontend can render it dynamically in the user's language.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ParallelStepResult {
    pub step_id: String,
    pub step_number: u32,
    pub status: String, // "completed" | "failed"
    /// Brief summary of what this step did (in the LLM's language).
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key_findings: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl ParallelStepResult {
    /// Build the full parallel-results JSON envelope injected into history.
    pub fn build_envelope(results: Vec<ParallelStepResult>) -> Value {
        json!({
            "type": "parallel_results",
            "steps": results
        })
    }
}

#[derive(Debug, Clone)]
struct DelegatedParallelChildResult {
    success: bool,
    output: Option<String>,
    data: Option<Value>,
}

impl AgentLoop {
    pub(super) fn normalized_plan_tool(step: &Task) -> Option<&str> {
        step.tool
            .as_deref()
            .map(str::trim)
            .filter(|tool| !tool.is_empty())
    }

    pub(super) fn should_delegate_plan_step(step: &Task) -> bool {
        matches!(
            Self::normalized_plan_tool(step),
            Some("task") | Some("parallel_task")
        )
    }

    pub(super) fn delegated_agent_for_step(step: &Task) -> &'static str {
        let text = format!(
            "{}\n{}",
            step.content,
            step.success_criteria.as_deref().unwrap_or_default()
        )
        .to_lowercase();

        if text.contains("review")
            || text.contains("code review")
            || text.contains("regression")
            || text.contains("审查")
            || text.contains("评审")
            || text.contains("回归")
        {
            "review"
        } else if text.contains("verify")
            || text.contains("verification")
            || text.contains("validate")
            || text.contains("test")
            || text.contains("release")
            || text.contains("smoke")
            || text.contains("验证")
            || text.contains("测试")
            || text.contains("发布")
        {
            "verification"
        } else if text.contains("plan")
            || text.contains("design")
            || text.contains("architecture")
            || text.contains("规划")
            || text.contains("设计")
            || text.contains("架构")
        {
            "plan"
        } else if text.contains("explore")
            || text.contains("find")
            || text.contains("search")
            || text.contains("locate")
            || text.contains("inspect")
            || text.contains("查找")
            || text.contains("搜索")
            || text.contains("定位")
            || text.contains("探索")
            || text.contains("检查")
        {
            "explore"
        } else {
            "general"
        }
    }

    pub(super) fn delegated_prompt_for_step_with_goal(
        plan_goal: Option<&str>,
        step: &Task,
        step_number: usize,
        total_steps: usize,
    ) -> String {
        let mut prompt = String::new();
        if let Some(goal) = plan_goal.map(str::trim).filter(|goal| !goal.is_empty()) {
            prompt.push_str("Plan goal/context:\n");
            prompt.push_str(goal);
            prompt.push_str("\n\n");
        }
        prompt.push_str(&format!(
            "Execute plan step {}/{}.\n\nTask:\n{}\n",
            step_number, total_steps, step.content
        ));
        if let Some(criteria) = step
            .success_criteria
            .as_deref()
            .filter(|s| !s.trim().is_empty())
        {
            prompt.push_str("\nSuccess criteria:\n");
            prompt.push_str(criteria);
            prompt.push('\n');
        }
        prompt.push_str("\nReturn a compact result with summary, evidence, risks, and confidence.");
        prompt
    }

    pub(super) fn delegated_task_args_with_goal(
        plan_goal: Option<&str>,
        step: &Task,
        step_number: usize,
        total_steps: usize,
    ) -> Value {
        json!({
            "agent": Self::delegated_agent_for_step(step),
            "description": step.content,
            "prompt": Self::delegated_prompt_for_step_with_goal(plan_goal, step, step_number, total_steps),
        })
    }

    pub(super) fn parallel_delegated_task_args_with_goal(
        plan_goal: Option<&str>,
        steps: &[(Task, usize)],
        total_steps: usize,
    ) -> Value {
        let tasks = steps
            .iter()
            .map(|(step, step_number)| {
                Self::delegated_task_args_with_goal(plan_goal, step, *step_number, total_steps)
            })
            .collect::<Vec<_>>();
        json!({ "tasks": tasks })
    }

    fn delegated_parallel_child_results(
        metadata: Option<&Value>,
        child_count: usize,
        fallback_success: bool,
    ) -> Vec<DelegatedParallelChildResult> {
        let metadata_results = metadata
            .and_then(|value| value.get("results"))
            .and_then(Value::as_array);

        (0..child_count)
            .map(|index| {
                let child = metadata_results.and_then(|results| results.get(index));
                let success = child
                    .and_then(|value| value.get("success"))
                    .and_then(Value::as_bool)
                    .unwrap_or(fallback_success);
                let output = child
                    .and_then(|value| value.get("output"))
                    .and_then(Value::as_str)
                    .map(str::to_string);

                DelegatedParallelChildResult {
                    success,
                    output,
                    data: child.cloned(),
                }
            })
            .collect()
    }

    /// Execute an execution plan using wave-based dependency-aware scheduling.
    ///
    /// Steps with no unmet dependencies are grouped into "waves". A wave with
    /// a single step executes sequentially (preserving the history chain). A
    /// wave with multiple independent steps executes them in parallel via
    /// `JoinSet`, then merges their results back into the shared history.
    pub(super) async fn execute_plan(
        &self,
        history: &[Message],
        plan: &ExecutionPlan,
        session_id: Option<&str>,
        event_tx: Option<mpsc::Sender<AgentEvent>>,
    ) -> Result<AgentResult> {
        let mut plan = plan.clone();
        let task_session_id = session_id.unwrap_or("").to_string();
        let mut current_history = history.to_vec();
        let mut total_usage = TokenUsage::default();
        let mut tool_calls_count = 0;
        let total_steps = plan.steps.len();

        // Add initial user message with the goal
        let steps_text = plan
            .steps
            .iter()
            .enumerate()
            .map(|(i, step)| format!("{}. {}", i + 1, step.content))
            .collect::<Vec<_>>()
            .join("\n");
        current_history.push(Message::user(&crate::prompts::render(
            crate::prompts::PLAN_EXECUTE_GOAL,
            &[("goal", &plan.goal), ("steps", &steps_text)],
        )));
        self.emit_task_updated(&event_tx, &task_session_id, &plan)
            .await;

        loop {
            let ready: Vec<String> = plan
                .get_ready_steps()
                .iter()
                .map(|s| s.id.clone())
                .collect();

            if ready.is_empty() {
                // All done or deadlock
                if plan.has_deadlock() {
                    tracing::warn!(
                        "Plan deadlock detected: {} pending steps with unresolvable dependencies",
                        plan.pending_count()
                    );
                }
                break;
            }

            if ready.len() == 1 {
                // === Single step: sequential execution (preserves history chain) ===
                let step_id = &ready[0];
                let step = plan
                    .steps
                    .iter()
                    .find(|s| s.id == *step_id)
                    .ok_or_else(|| anyhow::anyhow!("step '{}' not found in plan", step_id))?
                    .clone();
                let step_number = plan
                    .steps
                    .iter()
                    .position(|s| s.id == *step_id)
                    .unwrap_or(0)
                    + 1;

                // Send step start event
                plan.mark_status(&step.id, TaskStatus::InProgress);
                self.emit_task_updated(&event_tx, &task_session_id, &plan)
                    .await;

                if let Some(tx) = &event_tx {
                    tx.send(AgentEvent::StepStart {
                        step_id: step.id.clone(),
                        description: step.content.clone(),
                        step_number,
                        total_steps,
                    })
                    .await
                    .ok();
                }

                if Self::should_delegate_plan_step(&step) {
                    let tool_name = match Self::normalized_plan_tool(&step) {
                        Some("parallel_task") => "parallel_task",
                        _ => "task",
                    };
                    let args = if tool_name == "parallel_task" {
                        json!({ "tasks": [Self::delegated_task_args_with_goal(Some(&plan.goal), &step, step_number, total_steps)] })
                    } else {
                        Self::delegated_task_args_with_goal(
                            Some(&plan.goal),
                            &step,
                            step_number,
                            total_steps,
                        )
                    };
                    let (output, _exit_code, is_error, _metadata) = self
                        .execute_delegated_plan_tool(tool_name, &args, session_id, &event_tx)
                        .await;
                    tool_calls_count += 1;

                    if is_error {
                        tracing::error!("Delegated plan step '{}' failed: {}", step.id, output);
                        current_history.push(Message::user(&format!(
                            "Delegated plan step '{}' failed:\n{}",
                            step.content, output
                        )));
                        plan.mark_status(&step.id, TaskStatus::Failed);
                        self.emit_task_updated(&event_tx, &task_session_id, &plan)
                            .await;

                        if let Some(tx) = &event_tx {
                            tx.send(AgentEvent::StepEnd {
                                step_id: step.id.clone(),
                                status: TaskStatus::Failed,
                                step_number,
                                total_steps,
                            })
                            .await
                            .ok();
                        }
                    } else {
                        current_history.push(Message {
                            role: "assistant".to_string(),
                            content: vec![crate::llm::ContentBlock::Text { text: output }],
                            reasoning_content: None,
                        });
                        plan.mark_status(&step.id, TaskStatus::Completed);
                        self.emit_task_updated(&event_tx, &task_session_id, &plan)
                            .await;

                        if let Some(tx) = &event_tx {
                            tx.send(AgentEvent::StepEnd {
                                step_id: step.id.clone(),
                                status: TaskStatus::Completed,
                                step_number,
                                total_steps,
                            })
                            .await
                            .ok();
                        }
                    }
                } else {
                    let step_prompt = crate::prompts::render(
                        crate::prompts::PLAN_EXECUTE_STEP,
                        &[
                            ("step_num", &step_number.to_string()),
                            ("description", &step.content),
                        ],
                    );

                    match self
                        .execute_loop(
                            &current_history,
                            &step_prompt,
                            AgentStyle::GeneralPurpose,
                            None,
                            event_tx.clone(),
                            &tokio_util::sync::CancellationToken::new(),
                            false, // emit_end: false — End is emitted by execute_with_planning after execute_plan
                        )
                        .await
                    {
                        Ok(result) => {
                            current_history = result.messages.clone();
                            total_usage.prompt_tokens += result.usage.prompt_tokens;
                            total_usage.completion_tokens += result.usage.completion_tokens;
                            total_usage.total_tokens += result.usage.total_tokens;
                            tool_calls_count += result.tool_calls_count;
                            plan.mark_status(&step.id, TaskStatus::Completed);
                            self.emit_task_updated(&event_tx, &task_session_id, &plan)
                                .await;

                            if let Some(tx) = &event_tx {
                                tx.send(AgentEvent::StepEnd {
                                    step_id: step.id.clone(),
                                    status: TaskStatus::Completed,
                                    step_number,
                                    total_steps,
                                })
                                .await
                                .ok();
                            }
                        }
                        Err(e) => {
                            tracing::error!("Plan step '{}' failed: {}", step.id, e);
                            plan.mark_status(&step.id, TaskStatus::Failed);
                            self.emit_task_updated(&event_tx, &task_session_id, &plan)
                                .await;

                            if let Some(tx) = &event_tx {
                                tx.send(AgentEvent::StepEnd {
                                    step_id: step.id.clone(),
                                    status: TaskStatus::Failed,
                                    step_number,
                                    total_steps,
                                })
                                .await
                                .ok();
                            }
                        }
                    }
                }
            } else {
                // === Multiple steps: parallel execution via JoinSet ===
                // NOTE: Each parallel branch gets a clone of the base history.
                // Individual branch histories (tool calls, LLM turns) are NOT merged
                // back — only a summary message is appended. This is a deliberate
                // trade-off: merging divergent histories in a deterministic order is
                // complex and the summary approach keeps the context window manageable.
                let ready_steps: Vec<_> = ready
                    .iter()
                    .filter_map(|id| {
                        let step = plan.steps.iter().find(|s| s.id == *id)?.clone();
                        let step_number =
                            plan.steps.iter().position(|s| s.id == *id).unwrap_or(0) + 1;
                        Some((step, step_number))
                    })
                    .collect();

                // Mark all as InProgress and emit one task-list snapshot for the wave.
                for (step, _) in &ready_steps {
                    plan.mark_status(&step.id, TaskStatus::InProgress);
                }
                self.emit_task_updated(&event_tx, &task_session_id, &plan)
                    .await;

                // Emit StepStart events after the authoritative task-list snapshot.
                for (step, step_number) in &ready_steps {
                    if let Some(tx) = &event_tx {
                        tx.send(AgentEvent::StepStart {
                            step_id: step.id.clone(),
                            description: step.content.clone(),
                            step_number: *step_number,
                            total_steps,
                        })
                        .await
                        .ok();
                    }
                }

                let mut parallel_results: Vec<ParallelStepResult> = Vec::new();
                if ready_steps
                    .iter()
                    .all(|(step, _)| Self::should_delegate_plan_step(step))
                {
                    let args = Self::parallel_delegated_task_args_with_goal(
                        Some(&plan.goal),
                        &ready_steps,
                        total_steps,
                    );
                    let (output, _exit_code, is_error, metadata) = self
                        .execute_delegated_plan_tool("parallel_task", &args, session_id, &event_tx)
                        .await;
                    tool_calls_count += 1;

                    let child_results = Self::delegated_parallel_child_results(
                        metadata.as_ref(),
                        ready_steps.len(),
                        !is_error,
                    );
                    let wave_failed =
                        is_error || child_results.iter().any(|result| !result.success);

                    for ((step, step_number), child_result) in
                        ready_steps.iter().zip(child_results.iter())
                    {
                        let status = if child_result.success {
                            TaskStatus::Completed
                        } else {
                            TaskStatus::Failed
                        };
                        plan.mark_status(&step.id, status);
                        self.emit_task_updated(&event_tx, &task_session_id, &plan)
                            .await;

                        parallel_results.push(ParallelStepResult {
                            step_id: step.id.clone(),
                            step_number: *step_number as u32,
                            status: if child_result.success {
                                "completed"
                            } else {
                                "failed"
                            }
                            .to_string(),
                            summary: if child_result.success {
                                child_result
                                    .output
                                    .clone()
                                    .unwrap_or_else(|| output.trim().to_string())
                            } else {
                                String::new()
                            },
                            key_findings: None,
                            error: (!child_result.success).then(|| {
                                child_result
                                    .output
                                    .clone()
                                    .unwrap_or_else(|| output.clone())
                            }),
                            data: child_result.data.clone(),
                        });

                        if let Some(tx) = &event_tx {
                            tx.send(AgentEvent::StepEnd {
                                step_id: step.id.clone(),
                                status,
                                step_number: *step_number,
                                total_steps,
                            })
                            .await
                            .ok();
                        }
                    }

                    if wave_failed {
                        current_history.push(Message::user(&format!(
                            "Delegated parallel plan wave completed with failures:\n{}",
                            output
                        )));
                    } else {
                        current_history.push(Message {
                            role: "assistant".to_string(),
                            content: vec![crate::llm::ContentBlock::Text {
                                text: output.clone(),
                            }],
                            reasoning_content: None,
                        });
                    }
                } else {
                    let step_lookup = ready_steps
                        .iter()
                        .map(|(step, step_number)| (step.id.clone(), *step_number))
                        .collect::<Vec<_>>();
                    let outcomes = crate::ordered_parallel::run_ordered_parallel_with_limit(
                        ready_steps.clone(),
                        self.config.max_parallel_tasks,
                        {
                            let base_history = current_history.clone();
                            let agent = self.clone();
                            let tx = event_tx.clone();
                            move |_index, (step, step_number)| {
                                let base_history = base_history.clone();
                                let agent_clone = agent.clone();
                                let tx = tx.clone();
                                async move {
                                    let prompt = crate::prompts::render(
                                        crate::prompts::PLAN_EXECUTE_STEP,
                                        &[
                                            ("step_num", &step_number.to_string()),
                                            ("description", &step.content),
                                        ],
                                    );
                                    agent_clone
                                        .execute_loop(
                                            &base_history,
                                            &prompt,
                                            AgentStyle::GeneralPurpose,
                                            None,
                                            tx,
                                            &tokio_util::sync::CancellationToken::new(),
                                            false, // emit_end: false — End is emitted by execute_with_planning after execute_plan
                                        )
                                        .await
                                }
                            }
                        },
                    )
                    .await;

                    for outcome in outcomes {
                        let (step_id, step_number) = step_lookup
                            .get(outcome.index)
                            .cloned()
                            .unwrap_or_else(|| ("unknown".to_string(), 0));
                        match outcome.output {
                            Ok(step_result) => match step_result {
                                Ok(result) => {
                                    total_usage.prompt_tokens += result.usage.prompt_tokens;
                                    total_usage.completion_tokens += result.usage.completion_tokens;
                                    total_usage.total_tokens += result.usage.total_tokens;
                                    tool_calls_count += result.tool_calls_count;
                                    plan.mark_status(&step_id, TaskStatus::Completed);
                                    self.emit_task_updated(&event_tx, &task_session_id, &plan)
                                        .await;

                                    parallel_results.push(ParallelStepResult {
                                        step_id: step_id.clone(),
                                        step_number: step_number as u32,
                                        status: "completed".to_string(),
                                        summary: result.text.trim().to_string(),
                                        key_findings: None,
                                        error: None,
                                        data: None,
                                    });

                                    if let Some(tx) = &event_tx {
                                        tx.send(AgentEvent::StepEnd {
                                            step_id,
                                            status: TaskStatus::Completed,
                                            step_number,
                                            total_steps,
                                        })
                                        .await
                                        .ok();
                                    }
                                }
                                Err(e) => {
                                    tracing::error!("Plan step '{}' failed: {}", step_id, e);
                                    plan.mark_status(&step_id, TaskStatus::Failed);
                                    self.emit_task_updated(&event_tx, &task_session_id, &plan)
                                        .await;

                                    parallel_results.push(ParallelStepResult {
                                        step_id: step_id.clone(),
                                        step_number: step_number as u32,
                                        status: "failed".to_string(),
                                        summary: String::new(),
                                        key_findings: None,
                                        error: Some(e.to_string()),
                                        data: None,
                                    });

                                    if let Some(tx) = &event_tx {
                                        tx.send(AgentEvent::StepEnd {
                                            step_id,
                                            status: TaskStatus::Failed,
                                            step_number,
                                            total_steps,
                                        })
                                        .await
                                        .ok();
                                    }
                                }
                            },
                            Err(e) => {
                                tracing::error!("Plan step '{}' failed: {}", step_id, e);
                                plan.mark_status(&step_id, TaskStatus::Failed);
                                self.emit_task_updated(&event_tx, &task_session_id, &plan)
                                    .await;

                                parallel_results.push(ParallelStepResult {
                                    step_id: step_id.clone(),
                                    step_number: step_number as u32,
                                    status: "failed".to_string(),
                                    summary: String::new(),
                                    key_findings: None,
                                    error: Some(e.to_string()),
                                    data: None,
                                });

                                if let Some(tx) = &event_tx {
                                    tx.send(AgentEvent::StepEnd {
                                        step_id,
                                        status: TaskStatus::Failed,
                                        step_number,
                                        total_steps,
                                    })
                                    .await
                                    .ok();
                                }
                            }
                        }
                    }
                }

                // Merge parallel results into history for subsequent steps.
                // Emit as a structured JSON USER message so the frontend can
                // parse and render it in the user's language.
                if !parallel_results.is_empty() {
                    parallel_results.sort_by_key(|r| r.step_number);
                    let envelope = ParallelStepResult::build_envelope(parallel_results);
                    current_history.push(Message::user(
                        &serde_json::to_string(&envelope).unwrap_or_default(),
                    ));
                }
            }

            // Emit GoalProgress after each wave
            if self.config.goal_tracking {
                let completed = plan
                    .steps
                    .iter()
                    .filter(|s| s.status == TaskStatus::Completed)
                    .count();
                if let Some(tx) = &event_tx {
                    tx.send(AgentEvent::GoalProgress {
                        goal: plan.goal.clone(),
                        progress: plan.progress(),
                        completed_steps: completed,
                        total_steps,
                    })
                    .await
                    .ok();
                }
            }
        }

        // Get final response — find the last assistant message (not the last history
        // entry, which may be a user-summary injected after parallel execution)
        let final_text = current_history
            .iter()
            .rev()
            .find(|m| m.role == "assistant")
            .map(|m| {
                m.content
                    .iter()
                    .filter_map(|block| {
                        if let crate::llm::ContentBlock::Text { text } = block {
                            Some(text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();

        Ok(AgentResult {
            text: final_text,
            messages: current_history,
            usage: total_usage,
            tool_calls_count,
            verification_reports: Vec::new(),
        })
    }
}
