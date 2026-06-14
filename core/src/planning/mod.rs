//! Planning, Goal Tracking, and Task Management
//!
//! Unified task tracking for both execution planning (decomposed steps with
//! dependencies) and user-facing task lists (priority, manual tracking).
//!
//! The [`Task`] struct replaces the former separate `PlanStep` and `Todo` types.

pub mod llm_planner;

pub use llm_planner::{AchievementResult, LlmPlanner, PreAnalysis};

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

// ============================================================================
// Task Status (unified from StepStatus + TodoStatus)
// ============================================================================

/// Task status — covers both execution steps and manual tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// Task is waiting to be started
    #[default]
    Pending,
    /// Task is currently being worked on
    InProgress,
    /// Task completed successfully
    Completed,
    /// Task failed during execution
    Failed,
    /// Task was skipped (dependency resolution, no longer needed)
    Skipped,
    /// Task was cancelled by user
    Cancelled,
}

impl fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaskStatus::Pending => write!(f, "pending"),
            TaskStatus::InProgress => write!(f, "in_progress"),
            TaskStatus::Completed => write!(f, "completed"),
            TaskStatus::Failed => write!(f, "failed"),
            TaskStatus::Skipped => write!(f, "skipped"),
            TaskStatus::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl FromStr for TaskStatus {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "pending" => TaskStatus::Pending,
            "in_progress" | "inprogress" => TaskStatus::InProgress,
            "completed" | "done" => TaskStatus::Completed,
            "failed" => TaskStatus::Failed,
            "skipped" => TaskStatus::Skipped,
            "cancelled" | "canceled" => TaskStatus::Cancelled,
            _ => TaskStatus::Pending,
        })
    }
}

impl TaskStatus {
    /// Check if task is still active (not completed, failed, skipped, or cancelled)
    pub fn is_active(&self) -> bool {
        matches!(self, TaskStatus::Pending | TaskStatus::InProgress)
    }
}

// ============================================================================
// Task Priority
// ============================================================================

/// Task priority level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TaskPriority {
    /// High priority — should be done first
    High,
    /// Medium priority — normal importance
    #[default]
    Medium,
    /// Low priority — can be deferred
    Low,
}

impl fmt::Display for TaskPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaskPriority::High => write!(f, "high"),
            TaskPriority::Medium => write!(f, "medium"),
            TaskPriority::Low => write!(f, "low"),
        }
    }
}

impl FromStr for TaskPriority {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "high" | "h" | "1" => TaskPriority::High,
            "medium" | "med" | "m" | "2" => TaskPriority::Medium,
            "low" | "l" | "3" => TaskPriority::Low,
            _ => TaskPriority::Medium,
        })
    }
}

// ============================================================================
// Task (unified from PlanStep + Todo)
// ============================================================================

/// A task item — used for both execution plan steps and user-facing task tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique identifier
    pub id: String,
    /// Brief description of the task
    pub content: String,
    /// Current status
    pub status: TaskStatus,
    /// Priority level (for user-facing ordering)
    #[serde(default)]
    pub priority: TaskPriority,
    /// Tool to use for this step (execution plans)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    /// IDs of tasks that must complete before this one
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dependencies: Vec<String>,
    /// Expected output or success criteria
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub success_criteria: Option<String>,
}

impl Task {
    /// Create a new task with pending status and medium priority
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            status: TaskStatus::Pending,
            priority: TaskPriority::Medium,
            tool: None,
            dependencies: Vec::new(),
            success_criteria: None,
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set status
    pub fn with_status(mut self, status: TaskStatus) -> Self {
        self.status = status;
        self
    }

    /// Set tool for execution
    pub fn with_tool(mut self, tool: impl Into<String>) -> Self {
        self.tool = Some(tool.into());
        self
    }

    /// Set dependency IDs
    pub fn with_dependencies(mut self, deps: Vec<String>) -> Self {
        self.dependencies = deps;
        self
    }

    /// Set success criteria
    pub fn with_success_criteria(mut self, criteria: impl Into<String>) -> Self {
        self.success_criteria = Some(criteria.into());
        self
    }

    /// Check if task is still active
    pub fn is_active(&self) -> bool {
        self.status.is_active()
    }
}

// ============================================================================
// Planning Structures
// ============================================================================

/// Task complexity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Complexity {
    /// Simple task (1-2 steps)
    Simple,
    /// Medium complexity (3-5 steps)
    Medium,
    /// Complex task (6-10 steps)
    Complex,
    /// Very complex task (10+ steps)
    VeryComplex,
}

/// Execution plan for a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// High-level goal
    pub goal: String,
    /// Decomposed steps
    pub steps: Vec<Task>,
    /// Estimated complexity
    pub complexity: Complexity,
    /// Required tools
    pub required_tools: Vec<String>,
    /// Estimated total steps
    pub estimated_steps: usize,
}

impl ExecutionPlan {
    pub fn new(goal: impl Into<String>, complexity: Complexity) -> Self {
        Self {
            goal: goal.into(),
            steps: Vec::new(),
            complexity,
            required_tools: Vec::new(),
            estimated_steps: 0,
        }
    }

    pub fn add_step(&mut self, step: Task) {
        self.steps.push(step);
        self.estimated_steps = self.steps.len();
    }

    pub fn add_required_tool(&mut self, tool: impl Into<String>) {
        let tool_str = tool.into();
        if !self.required_tools.contains(&tool_str) {
            self.required_tools.push(tool_str);
        }
    }

    /// Get steps that are ready to execute (dependencies met)
    pub fn get_ready_steps(&self) -> Vec<&Task> {
        self.steps
            .iter()
            .filter(|step| {
                step.status == TaskStatus::Pending
                    && step.dependencies.iter().all(|dep_id| {
                        self.steps
                            .iter()
                            .find(|s| &s.id == dep_id)
                            .map(|s| s.status == TaskStatus::Completed)
                            .unwrap_or(false)
                    })
            })
            .collect()
    }

    /// Update the status of a step by ID
    pub fn mark_status(&mut self, step_id: &str, status: TaskStatus) {
        if let Some(step) = self.steps.iter_mut().find(|s| s.id == step_id) {
            step.status = status;
        }
    }

    /// Count remaining Pending steps
    pub fn pending_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| s.status == TaskStatus::Pending)
            .count()
    }

    /// Detect deadlock: Pending steps remain but none are ready to execute.
    ///
    /// This happens when all Pending steps have dependencies that are not
    /// Completed (e.g., circular deps or all deps Failed/Skipped).
    pub fn has_deadlock(&self) -> bool {
        self.pending_count() > 0 && self.get_ready_steps().is_empty()
    }

    /// Get progress as a fraction (0.0 - 1.0)
    pub fn progress(&self) -> f32 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let completed = self
            .steps
            .iter()
            .filter(|s| s.status == TaskStatus::Completed)
            .count();
        completed as f32 / self.steps.len() as f32
    }
}

// ============================================================================
// Goal Tracking Structures
// ============================================================================

/// Agent goal with success criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentGoal {
    /// Goal description
    pub description: String,
    /// Success criteria (list of conditions)
    pub success_criteria: Vec<String>,
    /// Current progress (0.0 - 1.0)
    pub progress: f32,
    /// Is goal achieved?
    pub achieved: bool,
    /// Timestamp when goal was created
    pub created_at: i64,
    /// Timestamp when goal was achieved (if achieved)
    pub achieved_at: Option<i64>,
}

impl AgentGoal {
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            success_criteria: Vec::new(),
            progress: 0.0,
            achieved: false,
            created_at: chrono::Utc::now().timestamp(),
            achieved_at: None,
        }
    }

    pub fn with_criteria(mut self, criteria: Vec<String>) -> Self {
        self.success_criteria = criteria;
        self
    }

    pub fn update_progress(&mut self, progress: f32) {
        self.progress = progress.clamp(0.0, 1.0);
    }

    pub fn mark_achieved(&mut self) {
        self.achieved = true;
        self.progress = 1.0;
        self.achieved_at = Some(chrono::Utc::now().timestamp());
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // TaskStatus tests (merged from TodoStatus + StepStatus)
    // ========================================================================

    #[test]
    fn test_task_status_display() {
        assert_eq!(TaskStatus::Pending.to_string(), "pending");
        assert_eq!(TaskStatus::InProgress.to_string(), "in_progress");
        assert_eq!(TaskStatus::Completed.to_string(), "completed");
        assert_eq!(TaskStatus::Failed.to_string(), "failed");
        assert_eq!(TaskStatus::Skipped.to_string(), "skipped");
        assert_eq!(TaskStatus::Cancelled.to_string(), "cancelled");
    }

    #[test]
    fn test_task_status_from_str() {
        assert_eq!(
            TaskStatus::from_str("pending").unwrap(),
            TaskStatus::Pending
        );
        assert_eq!(
            TaskStatus::from_str("in_progress").unwrap(),
            TaskStatus::InProgress
        );
        assert_eq!(
            TaskStatus::from_str("inprogress").unwrap(),
            TaskStatus::InProgress
        );
        assert_eq!(
            TaskStatus::from_str("completed").unwrap(),
            TaskStatus::Completed
        );
        assert_eq!(TaskStatus::from_str("done").unwrap(), TaskStatus::Completed);
        assert_eq!(TaskStatus::from_str("failed").unwrap(), TaskStatus::Failed);
        assert_eq!(
            TaskStatus::from_str("skipped").unwrap(),
            TaskStatus::Skipped
        );
        assert_eq!(
            TaskStatus::from_str("cancelled").unwrap(),
            TaskStatus::Cancelled
        );
        assert_eq!(
            TaskStatus::from_str("canceled").unwrap(),
            TaskStatus::Cancelled
        );
        assert_eq!(
            TaskStatus::from_str("unknown").unwrap(),
            TaskStatus::Pending
        );
    }

    #[test]
    fn test_task_status_is_active() {
        assert!(TaskStatus::Pending.is_active());
        assert!(TaskStatus::InProgress.is_active());
        assert!(!TaskStatus::Completed.is_active());
        assert!(!TaskStatus::Failed.is_active());
        assert!(!TaskStatus::Skipped.is_active());
        assert!(!TaskStatus::Cancelled.is_active());
    }

    #[test]
    fn test_task_status_serialization() {
        assert_eq!(
            serde_json::to_string(&TaskStatus::InProgress).unwrap(),
            "\"in_progress\""
        );
        assert_eq!(
            serde_json::to_string(&TaskStatus::Failed).unwrap(),
            "\"failed\""
        );
    }

    // ========================================================================
    // TaskPriority tests (from TodoPriority)
    // ========================================================================

    #[test]
    fn test_task_priority_display() {
        assert_eq!(TaskPriority::High.to_string(), "high");
        assert_eq!(TaskPriority::Medium.to_string(), "medium");
        assert_eq!(TaskPriority::Low.to_string(), "low");
    }

    #[test]
    fn test_task_priority_from_str() {
        assert_eq!(TaskPriority::from_str("high").unwrap(), TaskPriority::High);
        assert_eq!(TaskPriority::from_str("h").unwrap(), TaskPriority::High);
        assert_eq!(
            TaskPriority::from_str("medium").unwrap(),
            TaskPriority::Medium
        );
        assert_eq!(TaskPriority::from_str("med").unwrap(), TaskPriority::Medium);
        assert_eq!(TaskPriority::from_str("low").unwrap(), TaskPriority::Low);
        assert_eq!(TaskPriority::from_str("l").unwrap(), TaskPriority::Low);
        assert_eq!(
            TaskPriority::from_str("unknown").unwrap(),
            TaskPriority::Medium
        );
    }

    // ========================================================================
    // Task tests (unified from PlanStep + Todo)
    // ========================================================================

    #[test]
    fn test_task_new() {
        let task = Task::new("1", "Test task");
        assert_eq!(task.id, "1");
        assert_eq!(task.content, "Test task");
        assert_eq!(task.status, TaskStatus::Pending);
        assert_eq!(task.priority, TaskPriority::Medium);
        assert!(task.tool.is_none());
        assert!(task.dependencies.is_empty());
        assert!(task.success_criteria.is_none());
    }

    #[test]
    fn test_task_builder() {
        let task = Task::new("1", "Test task")
            .with_priority(TaskPriority::High)
            .with_status(TaskStatus::InProgress)
            .with_tool("bash")
            .with_dependencies(vec!["step-0".to_string()])
            .with_success_criteria("Command exits with 0");

        assert_eq!(task.priority, TaskPriority::High);
        assert_eq!(task.status, TaskStatus::InProgress);
        assert_eq!(task.tool, Some("bash".to_string()));
        assert_eq!(task.dependencies, vec!["step-0".to_string()]);
        assert_eq!(
            task.success_criteria,
            Some("Command exits with 0".to_string())
        );
    }

    #[test]
    fn test_task_is_active() {
        let pending = Task::new("1", "Pending task");
        let in_progress = Task::new("2", "In progress").with_status(TaskStatus::InProgress);
        let completed = Task::new("3", "Completed").with_status(TaskStatus::Completed);
        let failed = Task::new("4", "Failed").with_status(TaskStatus::Failed);
        let cancelled = Task::new("5", "Cancelled").with_status(TaskStatus::Cancelled);

        assert!(pending.is_active());
        assert!(in_progress.is_active());
        assert!(!completed.is_active());
        assert!(!failed.is_active());
        assert!(!cancelled.is_active());
    }

    #[test]
    fn test_task_serialization() {
        let task = Task::new("1", "Test task")
            .with_priority(TaskPriority::High)
            .with_status(TaskStatus::InProgress);

        let json = serde_json::to_string(&task).unwrap();
        let parsed: Task = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.id, task.id);
        assert_eq!(parsed.content, task.content);
        assert_eq!(parsed.status, task.status);
        assert_eq!(parsed.priority, task.priority);
    }

    // ========================================================================
    // ExecutionPlan tests
    // ========================================================================

    #[test]
    fn test_execution_plan() {
        let mut plan = ExecutionPlan::new("Test goal", Complexity::Medium);

        plan.add_step(Task::new("step-1", "First step"));
        plan.add_step(
            Task::new("step-2", "Second step").with_dependencies(vec!["step-1".to_string()]),
        );

        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.estimated_steps, 2);
        assert_eq!(plan.progress(), 0.0);

        // Mark first step as completed
        plan.steps[0].status = TaskStatus::Completed;
        assert_eq!(plan.progress(), 0.5);

        // Check ready steps
        let ready = plan.get_ready_steps();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, "step-2");
    }

    // ========================================================================
    // ExecutionPlan helper method tests
    // ========================================================================

    #[test]
    fn test_mark_status() {
        let mut plan = ExecutionPlan::new("Test", Complexity::Simple);
        plan.add_step(Task::new("s1", "Step 1"));
        plan.add_step(Task::new("s2", "Step 2"));

        assert_eq!(plan.steps[0].status, TaskStatus::Pending);
        plan.mark_status("s1", TaskStatus::InProgress);
        assert_eq!(plan.steps[0].status, TaskStatus::InProgress);
        plan.mark_status("s1", TaskStatus::Completed);
        assert_eq!(plan.steps[0].status, TaskStatus::Completed);
        // Non-existent ID is a no-op
        plan.mark_status("s999", TaskStatus::Failed);
        assert_eq!(plan.steps[1].status, TaskStatus::Pending);
    }

    #[test]
    fn test_pending_count() {
        let mut plan = ExecutionPlan::new("Test", Complexity::Simple);
        plan.add_step(Task::new("s1", "Step 1"));
        plan.add_step(Task::new("s2", "Step 2"));
        plan.add_step(Task::new("s3", "Step 3"));

        assert_eq!(plan.pending_count(), 3);
        plan.mark_status("s1", TaskStatus::Completed);
        assert_eq!(plan.pending_count(), 2);
        plan.mark_status("s2", TaskStatus::Failed);
        assert_eq!(plan.pending_count(), 1);
        plan.mark_status("s3", TaskStatus::InProgress);
        assert_eq!(plan.pending_count(), 0);
    }

    #[test]
    fn test_has_deadlock() {
        // Circular dependency: s1 depends on s2, s2 depends on s1
        let mut plan = ExecutionPlan::new("Test", Complexity::Simple);
        plan.add_step(Task::new("s1", "Step 1").with_dependencies(vec!["s2".to_string()]));
        plan.add_step(Task::new("s2", "Step 2").with_dependencies(vec!["s1".to_string()]));

        assert!(plan.has_deadlock());

        // No deadlock when steps have no deps
        let mut plan2 = ExecutionPlan::new("Test", Complexity::Simple);
        plan2.add_step(Task::new("s1", "Step 1"));
        assert!(!plan2.has_deadlock());

        // Deadlock when dependency failed
        let mut plan3 = ExecutionPlan::new("Test", Complexity::Simple);
        plan3.add_step(Task::new("s1", "Step 1"));
        plan3.add_step(Task::new("s2", "Step 2").with_dependencies(vec!["s1".to_string()]));
        plan3.mark_status("s1", TaskStatus::Failed);
        assert!(plan3.has_deadlock()); // s2 depends on s1 which failed, not completed
    }

    #[test]
    fn test_get_ready_steps_parallel() {
        // Three independent steps — all should be ready simultaneously
        let mut plan = ExecutionPlan::new("Test", Complexity::Medium);
        plan.add_step(Task::new("s1", "Step 1"));
        plan.add_step(Task::new("s2", "Step 2"));
        plan.add_step(Task::new("s3", "Step 3"));

        let ready = plan.get_ready_steps();
        assert_eq!(ready.len(), 3);
    }

    #[test]
    fn test_get_ready_steps_wave() {
        // s1 and s2 are independent; s3 depends on both
        let mut plan = ExecutionPlan::new("Test", Complexity::Medium);
        plan.add_step(Task::new("s1", "Step 1"));
        plan.add_step(Task::new("s2", "Step 2"));
        plan.add_step(
            Task::new("s3", "Step 3").with_dependencies(vec!["s1".to_string(), "s2".to_string()]),
        );

        // Wave 1: s1 and s2
        let ready = plan.get_ready_steps();
        assert_eq!(ready.len(), 2);
        let ids: Vec<&str> = ready.iter().map(|s| s.id.as_str()).collect();
        assert!(ids.contains(&"s1"));
        assert!(ids.contains(&"s2"));

        // Complete wave 1
        plan.mark_status("s1", TaskStatus::Completed);
        plan.mark_status("s2", TaskStatus::Completed);

        // Wave 2: s3
        let ready = plan.get_ready_steps();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, "s3");
    }

    // ========================================================================
    // AgentGoal tests
    // ========================================================================

    #[test]
    fn test_agent_goal() {
        let mut goal = AgentGoal::new("Complete task")
            .with_criteria(vec!["Criterion 1".to_string(), "Criterion 2".to_string()]);

        assert_eq!(goal.description, "Complete task");
        assert_eq!(goal.success_criteria.len(), 2);
        assert_eq!(goal.progress, 0.0);
        assert!(!goal.achieved);

        goal.update_progress(0.5);
        assert_eq!(goal.progress, 0.5);

        goal.mark_achieved();
        assert!(goal.achieved);
        assert_eq!(goal.progress, 1.0);
        assert!(goal.achieved_at.is_some());
    }

    #[test]
    fn test_complexity_levels() {
        assert_eq!(
            serde_json::to_string(&Complexity::Simple).unwrap(),
            "\"Simple\""
        );
        assert_eq!(
            serde_json::to_string(&Complexity::Complex).unwrap(),
            "\"Complex\""
        );
    }
}
