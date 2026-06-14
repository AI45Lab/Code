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

        let messages = vec![Message::user(prompt)];
        let response = llm
            .complete(&messages, Some(system), &[])
            .await
            .context("LLM pre-analysis call failed")?;

        let text = response.text();
        Self::parse_pre_analysis_response(&text, prompt)
    }

    fn parse_pre_analysis_response(text: &str, original_prompt: &str) -> Result<PreAnalysis> {
        let cleaned = Self::extract_json(text);
        let parsed: PreAnalysisResponse = serde_json::from_str(cleaned)
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
        let cleaned = Self::extract_json(text);
        let parsed: PlanResponse =
            serde_json::from_str(cleaned).context("Failed to parse plan JSON from LLM response")?;

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
        let cleaned = Self::extract_json(text);
        let parsed: GoalResponse =
            serde_json::from_str(cleaned).context("Failed to parse goal JSON from LLM response")?;

        Ok(AgentGoal::new(parsed.description).with_criteria(parsed.success_criteria))
    }

    fn parse_achievement_response(text: &str) -> Result<AchievementResult> {
        let cleaned = Self::extract_json(text);
        let parsed: AchievementResponse = serde_json::from_str(cleaned)
            .context("Failed to parse achievement JSON from LLM response")?;

        Ok(AchievementResult {
            achieved: parsed.achieved,
            progress: parsed.progress.clamp(0.0, 1.0),
            remaining_criteria: parsed.remaining_criteria,
        })
    }

    /// Extract JSON from LLM text that may contain markdown fences
    fn extract_json(text: &str) -> &str {
        let trimmed = text.trim();

        // Strip markdown code fences if present
        if let Some(start) = trimmed.find('{') {
            if let Some(end) = trimmed.rfind('}') {
                if start <= end {
                    return &trimmed[start..=end];
                }
            }
        }

        trimmed
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
    fn test_extract_json_plain() {
        assert_eq!(LlmPlanner::extract_json("  {\"a\": 1}  "), "{\"a\": 1}");
    }

    #[test]
    fn test_extract_json_with_fences() {
        let text = "```json\n{\"a\": 1}\n```";
        assert_eq!(LlmPlanner::extract_json(text), "{\"a\": 1}");
    }

    #[test]
    fn test_extract_json_with_surrounding_text() {
        let text = "Here is the plan:\n{\"goal\": \"test\"}\nDone.";
        assert_eq!(LlmPlanner::extract_json(text), "{\"goal\": \"test\"}");
    }
}
