//! LLM-powered planning logic
//!
//! Provides intelligent plan generation, goal extraction, and achievement
//! evaluation by sending structured prompts to an LLM and parsing JSON responses.
//! Falls back to heuristic logic when no LLM client is available.

use crate::llm::{LlmClient, Message};
use crate::planning::{AgentGoal, Complexity, ExecutionPlan, Task};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Result of evaluating goal achievement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AchievementResult {
    /// Whether the goal has been achieved
    pub achieved: bool,
    /// Progress toward goal (0.0 - 1.0)
    pub progress: f32,
    /// Criteria that remain unmet
    pub remaining_criteria: Vec<String>,
}

/// Pre-analysis result — intent, goal, plan, and optimized input in one LLM call.
#[derive(Debug, Clone)]
pub struct PreAnalysis {
    pub intent: crate::prompts::AgentStyle,
    pub requires_planning: bool,
    pub goal: AgentGoal,
    pub execution_plan: ExecutionPlan,
    /// LLM-rewritten version of the user input with ambiguities resolved.
    pub optimized_input: String,
}

/// LLM-powered planner that generates plans, extracts goals, and evaluates achievement
pub struct LlmPlanner;

// ============================================================================
// JSON response schemas for LLM parsing
// ============================================================================

#[derive(Debug, Deserialize)]
struct PlanResponse {
    goal: String,
    complexity: String,
    steps: Vec<StepResponse>,
    #[serde(default)]
    required_tools: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct StepResponse {
    id: String,
    description: String,
    #[serde(default)]
    tool: Option<String>,
    #[serde(default)]
    dependencies: Vec<String>,
    #[serde(default)]
    success_criteria: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GoalResponse {
    description: String,
    success_criteria: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct AchievementResponse {
    achieved: bool,
    progress: f32,
    #[serde(default)]
    remaining_criteria: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct PreAnalysisResponse {
    intent: String,
    requires_planning: bool,
    goal: GoalResponse,
    execution_plan: PreAnalysisPlan,
    optimized_input: String,
}

#[derive(Debug, Deserialize)]
struct PreAnalysisPlan {
    complexity: String,
    steps: Vec<StepResponse>,
    #[serde(default)]
    required_tools: Vec<String>,
}

impl LlmPlanner {
    /// Generate an execution plan from a prompt using LLM
    pub async fn create_plan(llm: &Arc<dyn LlmClient>, prompt: &str) -> Result<ExecutionPlan> {
        let system = crate::prompts::LLM_PLAN_SYSTEM;

        let messages = vec![Message::user(prompt)];
        let response = llm
            .complete(&messages, Some(system), &[])
            .await
            .context("LLM call failed during plan creation")?;

        let text = response.text();
        Self::parse_plan_response(&text)
    }

    /// Extract a goal with success criteria from a prompt using LLM
    pub async fn extract_goal(llm: &Arc<dyn LlmClient>, prompt: &str) -> Result<AgentGoal> {
        let system = crate::prompts::LLM_GOAL_EXTRACT_SYSTEM;

        let messages = vec![Message::user(prompt)];
        let response = llm
            .complete(&messages, Some(system), &[])
            .await
            .context("LLM call failed during goal extraction")?;

        let text = response.text();
        Self::parse_goal_response(&text)
    }

    /// Evaluate whether a goal has been achieved given current state
    pub async fn check_achievement(
        llm: &Arc<dyn LlmClient>,
        goal: &AgentGoal,
        current_state: &str,
    ) -> Result<AchievementResult> {
        let system = crate::prompts::LLM_GOAL_CHECK_SYSTEM;

        let user_message = format!(
            "Goal: {}\nSuccess Criteria: {}\nCurrent State: {}",
            goal.description,
            goal.success_criteria.join("; "),
            current_state,
        );

        let messages = vec![Message::user(&user_message)];
        let response = llm
            .complete(&messages, Some(system), &[])
            .await
            .context("LLM call failed during achievement check")?;

        let text = response.text();
        Self::parse_achievement_response(&text)
    }

    /// Create a fallback plan using heuristic logic (no LLM required)
    pub fn fallback_plan(prompt: &str) -> ExecutionPlan {
        let complexity = if prompt.len() < 50 {
            Complexity::Simple
        } else if prompt.len() < 150 {
            Complexity::Medium
        } else if prompt.len() < 300 {
            Complexity::Complex
        } else {
            Complexity::VeryComplex
        };

        let mut plan = ExecutionPlan::new(prompt, complexity);

        let step_count = match complexity {
            Complexity::Simple => 2,
            Complexity::Medium => 4,
            Complexity::Complex => 7,
            Complexity::VeryComplex => 10,
        };

        for i in 0..step_count {
            let step = Task::new(
                format!("step-{}", i + 1),
                crate::prompts::render(
                    crate::prompts::PLAN_FALLBACK_STEP,
                    &[("step_num", &(i + 1).to_string())],
                ),
            );
            plan.add_step(step);
        }

        plan
    }

    /// Create a fallback goal using heuristic logic (no LLM required)
    pub fn fallback_goal(prompt: &str) -> AgentGoal {
        AgentGoal::new(prompt).with_criteria(vec![
            "Task is completed successfully".to_string(),
            "All requirements are met".to_string(),
        ])
    }

    /// Create a fallback achievement result using heuristic logic (no LLM required)
    pub fn fallback_check_achievement(goal: &AgentGoal, current_state: &str) -> AchievementResult {
        let state_lower = current_state.to_lowercase();
        let achieved = state_lower.contains("complete")
            || state_lower.contains("done")
            || state_lower.contains("finished");

        let progress = if achieved { 1.0 } else { goal.progress };

        let remaining_criteria = if achieved {
            Vec::new()
        } else {
            goal.success_criteria.clone()
        };

        AchievementResult {
            achieved,
            progress,
            remaining_criteria,
        }
    }

    /// Perform pre-analysis in a single LLM call: intent classification, goal extraction,
    /// execution plan, and input optimization. Falls back to heuristics on failure.
    pub async fn pre_analyze(llm: &Arc<dyn LlmClient>, prompt: &str) -> Result<PreAnalysis> {
        let system = crate::prompts::PRE_ANALYSIS_SYSTEM;

        // One initial attempt plus one repair round: if the model returns
        // unparseable JSON, re-prompt it once to emit strictly valid JSON before
        // giving up (callers fall back to heuristics on the returned error).
        const MAX_ATTEMPTS: usize = 2;
        let mut messages = vec![Message::user(prompt)];
        let mut last_err: Option<anyhow::Error> = None;

        for attempt in 0..MAX_ATTEMPTS {
            let response = llm
                .complete(&messages, Some(system), &[])
                .await
                .context("LLM pre-analysis call failed")?;

            let text = response.text();
            match Self::parse_pre_analysis_response(&text, prompt) {
                Ok(analysis) => return Ok(analysis),
                Err(e) => {
                    last_err = Some(e);
                    if attempt + 1 < MAX_ATTEMPTS {
                        messages.push(response.message.clone());
                        messages.push(Message::user(
                            "Your previous response was not valid JSON matching the required \
                             schema. Respond again with ONLY the JSON object — no markdown \
                             fences, no prose, no explanation.",
                        ));
                    }
                }
            }
        }

        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("pre-analysis produced no result")))
    }

    fn parse_pre_analysis_response(text: &str, original_prompt: &str) -> Result<PreAnalysis> {
        let parsed: PreAnalysisResponse = Self::parse_json_lenient(text)
            .context("Failed to parse pre-analysis JSON from LLM response")?;

        let intent = match parsed.intent.to_lowercase().as_str() {
            "plan" => crate::prompts::AgentStyle::Plan,
            "explore" => crate::prompts::AgentStyle::Explore,
            "verification" => crate::prompts::AgentStyle::Verification,
            "codereview" | "code review" => crate::prompts::AgentStyle::CodeReview,
            _ => crate::prompts::AgentStyle::GeneralPurpose,
        };

        let goal_description = parsed.goal.description.clone();
        let goal =
            AgentGoal::new(goal_description.clone()).with_criteria(parsed.goal.success_criteria);

        let complexity = match parsed.execution_plan.complexity.as_str() {
            "Simple" => Complexity::Simple,
            "Medium" => Complexity::Medium,
            "Complex" => Complexity::Complex,
            "VeryComplex" => Complexity::VeryComplex,
            _ => Complexity::Medium,
        };

        let mut plan = ExecutionPlan::new(goal_description, complexity);
        for step_resp in parsed.execution_plan.steps {
            let mut task = Task::new(step_resp.id, step_resp.description);
            if let Some(tool) = step_resp.tool {
                task = task.with_tool(tool);
            }
            if !step_resp.dependencies.is_empty() {
                task = task.with_dependencies(step_resp.dependencies);
            }
            if let Some(criteria) = step_resp.success_criteria {
                task = task.with_success_criteria(criteria);
            }
            plan.add_step(task);
        }
        for tool in parsed.execution_plan.required_tools {
            plan.add_required_tool(tool);
        }

        Ok(PreAnalysis {
            intent,
            requires_planning: parsed.requires_planning,
            goal,
            execution_plan: plan,
            optimized_input: if parsed.optimized_input.is_empty() {
                original_prompt.to_string()
            } else {
                parsed.optimized_input
            },
        })
    }

    // ========================================================================
    // JSON parsing helpers
    // ========================================================================

    fn parse_plan_response(text: &str) -> Result<ExecutionPlan> {
        let parsed: PlanResponse = Self::parse_json_lenient(text)
            .context("Failed to parse plan JSON from LLM response")?;

        let complexity = match parsed.complexity.as_str() {
            "Simple" => Complexity::Simple,
            "Medium" => Complexity::Medium,
            "Complex" => Complexity::Complex,
            "VeryComplex" => Complexity::VeryComplex,
            _ => Complexity::Medium,
        };

        let mut plan = ExecutionPlan::new(parsed.goal, complexity);

        for step_resp in parsed.steps {
            let mut task = Task::new(step_resp.id, step_resp.description);
            if let Some(tool) = step_resp.tool {
                task = task.with_tool(tool);
            }
            if !step_resp.dependencies.is_empty() {
                task = task.with_dependencies(step_resp.dependencies);
            }
            if let Some(criteria) = step_resp.success_criteria {
                task = task.with_success_criteria(criteria);
            }
            plan.add_step(task);
        }

        for tool in parsed.required_tools {
            plan.add_required_tool(tool);
        }

        Ok(plan)
    }

    fn parse_goal_response(text: &str) -> Result<AgentGoal> {
        let parsed: GoalResponse = Self::parse_json_lenient(text)
            .context("Failed to parse goal JSON from LLM response")?;

        Ok(AgentGoal::new(parsed.description).with_criteria(parsed.success_criteria))
    }

    fn parse_achievement_response(text: &str) -> Result<AchievementResult> {
        let parsed: AchievementResponse = Self::parse_json_lenient(text)
            .context("Failed to parse achievement JSON from LLM response")?;

        Ok(AchievementResult {
            achieved: parsed.achieved,
            progress: parsed.progress.clamp(0.0, 1.0),
            remaining_criteria: parsed.remaining_criteria,
        })
    }

    /// Parse JSON from possibly-dirty LLM output into the target type.
    ///
    /// Uses the shared robust extractor ([`crate::llm::structured::extract_json_value`]),
    /// which handles ```json fences, surrounding prose, and braces embedded in
    /// strings — unlike the previous naive first-`{`/last-`}` slice, which broke
    /// on fenced output or any `}` inside a string value.
    fn parse_json_lenient<T: serde::de::DeserializeOwned>(text: &str) -> Result<T> {
        let value = crate::llm::structured::extract_json_value(text)?;
        Ok(serde_json::from_value(value)?)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_plan_response() {
        let json = r#"{
            "goal": "Build a REST API",
            "complexity": "Complex",
            "steps": [
                {
                    "id": "step-1",
                    "description": "Set up project structure",
                    "tool": "bash",
                    "dependencies": [],
                    "success_criteria": "Project directory created"
                },
                {
                    "id": "step-2",
                    "description": "Implement endpoints",
                    "tool": "write",
                    "dependencies": ["step-1"],
                    "success_criteria": "Endpoints respond correctly"
                }
            ],
            "required_tools": ["bash", "write", "read"]
        }"#;

        let plan = LlmPlanner::parse_plan_response(json).unwrap();
        assert_eq!(plan.goal, "Build a REST API");
        assert_eq!(plan.complexity, Complexity::Complex);
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[0].id, "step-1");
        assert_eq!(plan.steps[0].tool, Some("bash".to_string()));
        assert_eq!(plan.steps[1].dependencies, vec!["step-1".to_string()]);
        assert_eq!(plan.required_tools, vec!["bash", "write", "read"]);
    }

    #[test]
    fn test_parse_plan_response_with_markdown_fences() {
        let json = "```json\n{\"goal\": \"Test\", \"complexity\": \"Simple\", \"steps\": [{\"id\": \"step-1\", \"description\": \"Do it\"}], \"required_tools\": []}\n```";

        let plan = LlmPlanner::parse_plan_response(json).unwrap();
        assert_eq!(plan.goal, "Test");
        assert_eq!(plan.complexity, Complexity::Simple);
        assert_eq!(plan.steps.len(), 1);
    }

    #[test]
    fn test_parse_plan_response_invalid() {
        let bad_json = "This is not JSON at all";
        let result = LlmPlanner::parse_plan_response(bad_json);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_plan_response_unknown_complexity() {
        let json =
            r#"{"goal": "Test", "complexity": "Unknown", "steps": [], "required_tools": []}"#;
        let plan = LlmPlanner::parse_plan_response(json).unwrap();
        assert_eq!(plan.complexity, Complexity::Medium); // falls back to Medium
    }

    #[test]
    fn test_parse_goal_response() {
        let json = r#"{
            "description": "Deploy the application to production",
            "success_criteria": [
                "All tests pass",
                "Application is accessible at production URL",
                "Health check returns 200"
            ]
        }"#;

        let goal = LlmPlanner::parse_goal_response(json).unwrap();
        assert_eq!(goal.description, "Deploy the application to production");
        assert_eq!(goal.success_criteria.len(), 3);
        assert_eq!(goal.success_criteria[0], "All tests pass");
    }

    #[test]
    fn test_parse_goal_response_invalid() {
        let result = LlmPlanner::parse_goal_response("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_achievement_response() {
        let json = r#"{
            "achieved": false,
            "progress": 0.65,
            "remaining_criteria": ["Health check not verified"]
        }"#;

        let result = LlmPlanner::parse_achievement_response(json).unwrap();
        assert!(!result.achieved);
        assert!((result.progress - 0.65).abs() < f32::EPSILON);
        assert_eq!(result.remaining_criteria, vec!["Health check not verified"]);
    }

    #[test]
    fn test_parse_achievement_response_achieved() {
        let json = r#"{"achieved": true, "progress": 1.0, "remaining_criteria": []}"#;
        let result = LlmPlanner::parse_achievement_response(json).unwrap();
        assert!(result.achieved);
        assert!((result.progress - 1.0).abs() < f32::EPSILON);
        assert!(result.remaining_criteria.is_empty());
    }

    #[test]
    fn test_parse_achievement_response_clamps_progress() {
        let json = r#"{"achieved": false, "progress": 1.5, "remaining_criteria": []}"#;
        let result = LlmPlanner::parse_achievement_response(json).unwrap();
        assert!((result.progress - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fallback_plan() {
        let short_prompt = "Fix bug";
        let plan = LlmPlanner::fallback_plan(short_prompt);
        assert_eq!(plan.complexity, Complexity::Simple);
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.goal, short_prompt);

        let long_prompt = "Implement a comprehensive authentication system with OAuth2 support, JWT tokens, refresh token rotation, multi-factor authentication, and role-based access control across all API endpoints with proper audit logging and session management capabilities for both web and mobile clients, including password reset flows, account lockout policies, and integration with external identity providers such as Google, GitHub, and SAML-based enterprise SSO systems";
        let plan = LlmPlanner::fallback_plan(long_prompt);
        assert_eq!(plan.complexity, Complexity::VeryComplex);
        assert_eq!(plan.steps.len(), 10);
    }

    #[test]
    fn test_fallback_goal() {
        let goal = LlmPlanner::fallback_goal("Fix the login bug");
        assert_eq!(goal.description, "Fix the login bug");
        assert_eq!(goal.success_criteria.len(), 2);
        assert_eq!(goal.success_criteria[0], "Task is completed successfully");
    }

    #[test]
    fn test_fallback_check_achievement_done() {
        let goal = AgentGoal::new("Test task").with_criteria(vec!["Criterion 1".to_string()]);

        let result = LlmPlanner::fallback_check_achievement(&goal, "The task is done.");
        assert!(result.achieved);
        assert!((result.progress - 1.0).abs() < f32::EPSILON);
        assert!(result.remaining_criteria.is_empty());
    }

    #[test]
    fn test_fallback_check_achievement_not_done() {
        let goal = AgentGoal::new("Test task")
            .with_criteria(vec!["Criterion 1".to_string(), "Criterion 2".to_string()]);

        let result = LlmPlanner::fallback_check_achievement(&goal, "Work in progress");
        assert!(!result.achieved);
        assert_eq!(result.remaining_criteria.len(), 2);
    }

    #[test]
    fn test_parse_json_lenient_plain() {
        let v: serde_json::Value = LlmPlanner::parse_json_lenient("  {\"a\": 1}  ").unwrap();
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn test_parse_json_lenient_with_fences() {
        let text = "```json\n{\"a\": 1}\n```";
        let v: serde_json::Value = LlmPlanner::parse_json_lenient(text).unwrap();
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn test_parse_json_lenient_with_surrounding_prose() {
        let text = "Here is the plan:\n{\"goal\": \"test\"}\nDone.";
        let v: serde_json::Value = LlmPlanner::parse_json_lenient(text).unwrap();
        assert_eq!(v["goal"], "test");
    }

    #[test]
    fn test_parse_json_lenient_brace_inside_string_value() {
        // The old naive first-`{`/last-`}` slice broke when a string value
        // contained a `}` followed by trailing prose; the robust extractor
        // balances braces while respecting string boundaries.
        let text = "Result: {\"note\": \"use a closing brace } here\"} -- end.";
        let v: serde_json::Value = LlmPlanner::parse_json_lenient(text).unwrap();
        assert_eq!(v["note"], "use a closing brace } here");
    }

    #[test]
    fn test_parse_json_lenient_fenced_with_trailing_prose() {
        // ```json fence followed by an explanation. The naive parser's
        // `rfind('}')` could grab a brace from the trailing prose.
        let text = "```json\n{\"goal\": \"ship\"}\n```\nNote: revisit the `plan` later.";
        let v: serde_json::Value = LlmPlanner::parse_json_lenient(text).unwrap();
        assert_eq!(v["goal"], "ship");
    }

    #[test]
    fn test_parse_json_lenient_rejects_non_json() {
        let err = LlmPlanner::parse_json_lenient::<serde_json::Value>("no json here at all");
        assert!(err.is_err());
    }

    /// Replays a fixed sequence of assistant text responses, one per call.
    struct ReplayClient {
        responses: std::sync::Mutex<Vec<String>>,
    }

    impl ReplayClient {
        fn new(responses: Vec<String>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses),
            }
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for ReplayClient {
        async fn complete(
            &self,
            _messages: &[Message],
            _system: Option<&str>,
            _tools: &[crate::llm::ToolDefinition],
        ) -> anyhow::Result<crate::llm::LlmResponse> {
            let text = {
                let mut r = self.responses.lock().unwrap();
                if r.is_empty() {
                    String::new()
                } else {
                    r.remove(0)
                }
            };
            Ok(crate::llm::LlmResponse {
                message: Message {
                    role: "assistant".to_string(),
                    content: vec![crate::llm::ContentBlock::Text { text }],
                    reasoning_content: None,
                },
                usage: crate::llm::TokenUsage::default(),
                stop_reason: None,
                token_logprobs: Vec::new(),
                meta: None,
            })
        }

        async fn complete_streaming(
            &self,
            _messages: &[Message],
            _system: Option<&str>,
            _tools: &[crate::llm::ToolDefinition],
            _cancel_token: tokio_util::sync::CancellationToken,
        ) -> anyhow::Result<tokio::sync::mpsc::Receiver<crate::llm::StreamEvent>> {
            anyhow::bail!("streaming not used in planner tests")
        }
    }

    #[tokio::test]
    async fn test_pre_analyze_repairs_invalid_json() {
        // First response is unparseable; pre_analyze must re-prompt and succeed
        // on the second (valid) response.
        let good = r#"{"intent":"explore","requires_planning":false,"goal":{"description":"Do x","success_criteria":["done"]},"execution_plan":{"complexity":"Simple","steps":[],"required_tools":[]},"optimized_input":"Do x carefully"}"#;
        let client: Arc<dyn LlmClient> = Arc::new(ReplayClient::new(vec![
            "Sorry — here's the plan, but not as JSON.".to_string(),
            good.to_string(),
        ]));
        let pa = LlmPlanner::pre_analyze(&client, "do x").await.unwrap();
        assert_eq!(pa.optimized_input, "Do x carefully");
    }

    #[tokio::test]
    async fn test_pre_analyze_first_try_with_fenced_json() {
        // A single ```json-fenced response must parse on the first attempt
        // (robust extractor), with no repair round needed.
        let good = format!(
            "```json\n{}\n```",
            r#"{"intent":"plan","requires_planning":true,"goal":{"description":"g","success_criteria":[]},"execution_plan":{"complexity":"Medium","steps":[],"required_tools":[]},"optimized_input":"opt"}"#
        );
        let client: Arc<dyn LlmClient> = Arc::new(ReplayClient::new(vec![good]));
        let pa = LlmPlanner::pre_analyze(&client, "do x").await.unwrap();
        assert_eq!(pa.optimized_input, "opt");
    }
}
