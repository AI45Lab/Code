use super::*;
use crate::agent::tests::MockLlmClient;
use crate::queue::SessionQueueConfig;
use crate::tools::ToolExecutor;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::mpsc;

fn test_tool_context() -> ToolContext {
    ToolContext::new(PathBuf::from("/tmp"))
}

// ========================================================================
// AgentConfig
// ========================================================================

#[test]
fn test_agent_config_debug() {
    let config = AgentConfig {
        prompt_slots: SystemPromptSlots {
            extra: Some("You are helpful".to_string()),
            ..Default::default()
        },
        tools: vec![],
        max_tool_rounds: 10,
        permission_checker: None,
        confirmation_manager: None,
        context_providers: vec![],
        planning_mode: PlanningMode::Enabled,
        goal_tracking: false,
        hook_engine: None,
        skill_registry: None,
        ..AgentConfig::default()
    };
    let debug = format!("{:?}", config);
    assert!(debug.contains("AgentConfig"));
    assert!(debug.contains("planning_mode"));
}

#[test]
fn test_agent_config_default_values() {
    let config = AgentConfig::default();
    assert_eq!(config.max_tool_rounds, MAX_TOOL_ROUNDS);
    assert_eq!(config.max_parallel_tasks, DEFAULT_MAX_PARALLEL_TASKS);
    assert_eq!(config.planning_mode, PlanningMode::Auto);
    assert!(!config.goal_tracking);
    assert!(config.context_providers.is_empty());
}

#[test]
fn test_auto_pre_analysis_runs_without_keyword_gate() {
    let mock_client = Arc::new(MockLlmClient::new(vec![]));
    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let agent = AgentLoop::new(
        mock_client,
        tool_executor,
        test_tool_context(),
        AgentConfig::default(),
    );

    assert!(agent.should_run_pre_analysis());
}

#[test]
fn test_disabled_planning_never_runs_pre_analysis() {
    let mock_client = Arc::new(MockLlmClient::new(vec![]));
    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let config = AgentConfig {
        planning_mode: PlanningMode::Disabled,
        ..AgentConfig::default()
    };
    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);

    assert!(!agent.should_run_pre_analysis());
}

// ========================================================================
// AgentEvent serialization
// ========================================================================

#[test]
fn test_agent_event_serialize_start() {
    let event = AgentEvent::Start {
        prompt: "Hello".to_string(),
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("agent_start"));
    assert!(json.contains("Hello"));
}

#[test]
fn test_agent_event_serialize_text_delta() {
    let event = AgentEvent::TextDelta {
        text: "chunk".to_string(),
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("text_delta"));
}

#[test]
fn test_agent_event_serialize_tool_start() {
    let event = AgentEvent::ToolStart {
        id: "t1".to_string(),
        name: "bash".to_string(),
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("tool_start"));
    assert!(json.contains("bash"));
}

#[test]
fn test_agent_event_serialize_tool_end() {
    let event = AgentEvent::ToolEnd {
        id: "t1".to_string(),
        name: "bash".to_string(),
        output: "hello".to_string(),
        exit_code: 0,
        metadata: None,
        error_kind: None,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("tool_end"));
}

#[test]
fn test_agent_event_tool_end_has_metadata_field() {
    let event = AgentEvent::ToolEnd {
        id: "t1".to_string(),
        name: "write".to_string(),
        output: "Wrote 5 bytes".to_string(),
        exit_code: 0,
        metadata: Some(
            serde_json::json!({ "before": "old", "after": "new", "file_path": "f.txt" }),
        ),
        error_kind: None,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("\"before\""));
}

#[test]
fn test_agent_event_serialize_error() {
    let event = AgentEvent::Error {
        message: "oops".to_string(),
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("error"));
    assert!(json.contains("oops"));
}

#[test]
fn test_agent_event_serialize_confirmation_required() {
    let event = AgentEvent::ConfirmationRequired {
        tool_id: "t1".to_string(),
        tool_name: "bash".to_string(),
        args: serde_json::json!({"cmd": "rm"}),
        timeout_ms: 30000,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("confirmation_required"));
}

#[test]
fn test_agent_event_serialize_confirmation_received() {
    let event = AgentEvent::ConfirmationReceived {
        tool_id: "t1".to_string(),
        approved: true,
        reason: Some("safe".to_string()),
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("confirmation_received"));
}

#[test]
fn test_agent_event_serialize_confirmation_timeout() {
    let event = AgentEvent::ConfirmationTimeout {
        tool_id: "t1".to_string(),
        action_taken: "rejected".to_string(),
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("confirmation_timeout"));
}

#[test]
fn test_agent_event_serialize_external_task_pending() {
    let event = AgentEvent::ExternalTaskPending {
        task_id: "task-1".to_string(),
        session_id: "sess-1".to_string(),
        lane: crate::queue::SessionLane::Execute,
        command_type: "bash".to_string(),
        payload: serde_json::json!({}),
        timeout_ms: 60000,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("external_task_pending"));
}

#[test]
fn test_agent_event_serialize_external_task_completed() {
    let event = AgentEvent::ExternalTaskCompleted {
        task_id: "task-1".to_string(),
        session_id: "sess-1".to_string(),
        success: false,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("external_task_completed"));
}

#[test]
fn test_agent_event_serialize_permission_denied() {
    let event = AgentEvent::PermissionDenied {
        tool_id: "t1".to_string(),
        tool_name: "bash".to_string(),
        args: serde_json::json!({}),
        reason: "denied".to_string(),
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("permission_denied"));
}

#[test]
fn test_agent_event_serialize_context_compacted() {
    let event = AgentEvent::ContextCompacted {
        session_id: "sess-1".to_string(),
        before_messages: 100,
        after_messages: 20,
        percent_before: 0.85,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("context_compacted"));
}

#[test]
fn test_agent_event_serialize_turn_start() {
    let event = AgentEvent::TurnStart { turn: 3 };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("turn_start"));
}

#[test]
fn test_agent_event_serialize_turn_end() {
    let event = AgentEvent::TurnEnd {
        turn: 3,
        usage: TokenUsage::default(),
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("turn_end"));
}

#[test]
fn test_agent_event_serialize_end() {
    let event = AgentEvent::End {
        text: "Done".to_string(),
        usage: TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            cache_read_tokens: None,
            cache_write_tokens: None,
        },
        verification_summary: Box::new(crate::verification::VerificationSummary::from_reports(&[])),
        meta: None,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("agent_end"));
    assert!(json.contains("verification_summary"));
}

// ========================================================================
// AgentResult
// ========================================================================

#[test]
fn test_agent_result_fields() {
    let result = AgentResult {
        text: "output".to_string(),
        messages: vec![Message::user("hello")],
        usage: TokenUsage::default(),
        tool_calls_count: 3,
        verification_reports: Vec::new(),
    };
    assert_eq!(result.text, "output");
    assert_eq!(result.messages.len(), 1);
    assert_eq!(result.tool_calls_count, 3);
    assert!(result.verification_reports.is_empty());
    assert_eq!(
        result.verification_summary().status,
        crate::verification::VerificationStatus::Skipped
    );
    assert!(!result.has_pending_verification());
}

#[test]
fn test_collect_verification_report_from_tool_metadata() {
    let report = crate::verification::VerificationReport::new(
        "program:example",
        vec![crate::verification::VerificationCheck::required(
            "check:inspect",
            "inspect_artifacts",
            "Inspect artifacts",
        )],
    );
    let metadata = Some(serde_json::json!({
        "verification_report": report.to_value()
    }));
    let mut reports = Vec::new();

    AgentLoop::collect_verification_report(&mut reports, &metadata);

    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].subject, "program:example");
    assert_eq!(
        reports[0].status,
        crate::verification::VerificationStatus::NeedsReview
    );
}

#[test]
fn test_agent_result_verification_summary() {
    let report = crate::verification::VerificationReport::new(
        "program:example",
        vec![crate::verification::VerificationCheck::required(
            "check:inspect",
            "inspect_artifacts",
            "Inspect artifacts",
        )],
    );
    let result = AgentResult {
        text: "output".to_string(),
        messages: Vec::new(),
        usage: TokenUsage::default(),
        tool_calls_count: 1,
        verification_reports: vec![report],
    };

    let summary = result.verification_summary();

    assert_eq!(
        summary.status,
        crate::verification::VerificationStatus::NeedsReview
    );
    assert_eq!(summary.pending_required_check_count, 1);
    assert!(result
        .verification_summary_text()
        .contains("Verification needs review"));
    assert!(result.has_pending_verification());
}

// ========================================================================
// Missing AgentEvent serialization tests
// ========================================================================

#[test]
fn test_agent_event_serialize_context_resolving() {
    let event = AgentEvent::ContextResolving {
        providers: vec!["provider1".to_string(), "provider2".to_string()],
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("context_resolving"));
    assert!(json.contains("provider1"));
}

#[test]
fn test_agent_event_serialize_context_resolved() {
    let event = AgentEvent::ContextResolved {
        total_items: 5,
        total_tokens: 1000,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("context_resolved"));
    assert!(json.contains("1000"));
}

#[test]
fn test_agent_event_serialize_command_dead_lettered() {
    let event = AgentEvent::CommandDeadLettered {
        command_id: "cmd-1".to_string(),
        command_type: "bash".to_string(),
        lane: "execute".to_string(),
        error: "timeout".to_string(),
        attempts: 3,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("command_dead_lettered"));
    assert!(json.contains("cmd-1"));
}

#[test]
fn test_agent_event_serialize_command_retry() {
    let event = AgentEvent::CommandRetry {
        command_id: "cmd-2".to_string(),
        command_type: "read".to_string(),
        lane: "query".to_string(),
        attempt: 2,
        delay_ms: 1000,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("command_retry"));
    assert!(json.contains("cmd-2"));
}

#[test]
fn test_agent_event_serialize_queue_alert() {
    let event = AgentEvent::QueueAlert {
        level: "warning".to_string(),
        alert_type: "depth".to_string(),
        message: "Queue depth exceeded".to_string(),
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("queue_alert"));
    assert!(json.contains("warning"));
}

#[test]
fn test_agent_event_serialize_task_updated() {
    let event = AgentEvent::TaskUpdated {
        session_id: "sess-1".to_string(),
        tasks: vec![],
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("task_updated"));
    assert!(json.contains("sess-1"));
}

#[test]
fn test_agent_event_serialize_memory_stored() {
    let event = AgentEvent::MemoryStored {
        memory_id: "mem-1".to_string(),
        memory_type: "conversation".to_string(),
        importance: 0.8,
        tags: vec!["important".to_string()],
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("memory_stored"));
    assert!(json.contains("mem-1"));
}

#[test]
fn test_agent_event_serialize_memory_recalled() {
    let event = AgentEvent::MemoryRecalled {
        memory_id: "mem-2".to_string(),
        content: "Previous conversation".to_string(),
        relevance: 0.9,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("memory_recalled"));
    assert!(json.contains("mem-2"));
}

#[test]
fn test_agent_event_serialize_memories_searched() {
    let event = AgentEvent::MemoriesSearched {
        query: Some("search term".to_string()),
        tags: vec!["tag1".to_string()],
        result_count: 5,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("memories_searched"));
    assert!(json.contains("search term"));
}

#[test]
fn test_agent_event_serialize_memory_cleared() {
    let event = AgentEvent::MemoryCleared {
        tier: "short_term".to_string(),
        count: 10,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("memory_cleared"));
    assert!(json.contains("short_term"));
}

#[test]
fn test_agent_event_serialize_subagent_start() {
    let event = AgentEvent::SubagentStart {
        task_id: "task-1".to_string(),
        session_id: "child-sess".to_string(),
        parent_session_id: "parent-sess".to_string(),
        agent: "explore".to_string(),
        description: "Explore codebase".to_string(),
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("subagent_start"));
    assert!(json.contains("explore"));
}

#[test]
fn test_agent_event_serialize_subagent_progress() {
    let event = AgentEvent::SubagentProgress {
        task_id: "task-1".to_string(),
        session_id: "child-sess".to_string(),
        status: "processing".to_string(),
        metadata: serde_json::json!({"progress": 50}),
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("subagent_progress"));
    assert!(json.contains("processing"));
}

#[test]
fn test_agent_event_serialize_subagent_end() {
    let event = AgentEvent::SubagentEnd {
        task_id: "task-1".to_string(),
        session_id: "child-sess".to_string(),
        agent: "explore".to_string(),
        output: "Found 10 files".to_string(),
        success: true,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("subagent_end"));
    assert!(json.contains("Found 10 files"));
}

#[test]
fn test_agent_event_serialize_planning_start() {
    let event = AgentEvent::PlanningStart {
        prompt: "Build a web app".to_string(),
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("planning_start"));
    assert!(json.contains("Build a web app"));
}

#[test]
fn test_agent_event_serialize_planning_end() {
    use crate::planning::{Complexity, ExecutionPlan};
    let plan = ExecutionPlan::new("Test goal".to_string(), Complexity::Simple);
    let event = AgentEvent::PlanningEnd {
        plan,
        estimated_steps: 3,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("planning_end"));
    assert!(json.contains("estimated_steps"));
}

#[test]
fn test_agent_event_serialize_step_start() {
    let event = AgentEvent::StepStart {
        step_id: "step-1".to_string(),
        description: "Initialize project".to_string(),
        step_number: 1,
        total_steps: 5,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("step_start"));
    assert!(json.contains("Initialize project"));
}

#[test]
fn test_agent_event_serialize_step_end() {
    let event = AgentEvent::StepEnd {
        step_id: "step-1".to_string(),
        status: TaskStatus::Completed,
        step_number: 1,
        total_steps: 5,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("step_end"));
    assert!(json.contains("step-1"));
}

#[test]
fn test_agent_event_serialize_goal_extracted() {
    use crate::planning::AgentGoal;
    let goal = AgentGoal::new("Complete the task".to_string());
    let event = AgentEvent::GoalExtracted { goal };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("goal_extracted"));
}

#[test]
fn test_agent_event_serialize_goal_progress() {
    let event = AgentEvent::GoalProgress {
        goal: "Build app".to_string(),
        progress: 0.5,
        completed_steps: 2,
        total_steps: 4,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("goal_progress"));
    assert!(json.contains("0.5"));
}

#[test]
fn test_agent_event_serialize_goal_achieved() {
    let event = AgentEvent::GoalAchieved {
        goal: "Build app".to_string(),
        total_steps: 4,
        duration_ms: 5000,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("goal_achieved"));
    assert!(json.contains("5000"));
}

#[tokio::test]
async fn test_extract_goal_with_json_response() {
    // LlmPlanner expects JSON with "description" and "success_criteria" fields
    let mock_client = Arc::new(MockLlmClient::new(vec![MockLlmClient::text_response(
        r#"{"description": "Build web app", "success_criteria": ["App runs on port 3000", "Has login page"]}"#,
    )]));
    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let agent = AgentLoop::new(
        mock_client,
        tool_executor,
        test_tool_context(),
        AgentConfig::default(),
    );

    let goal = agent.extract_goal("Build a web app").await.unwrap();
    assert_eq!(goal.description, "Build web app");
    assert_eq!(goal.success_criteria.len(), 2);
    assert_eq!(goal.success_criteria[0], "App runs on port 3000");
}

#[tokio::test]
async fn test_extract_goal_fallback_on_non_json() {
    // Non-JSON response triggers fallback: returns the original prompt as goal
    let mock_client = Arc::new(MockLlmClient::new(vec![MockLlmClient::text_response(
        "Some non-JSON response",
    )]));
    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let agent = AgentLoop::new(
        mock_client,
        tool_executor,
        test_tool_context(),
        AgentConfig::default(),
    );

    let goal = agent.extract_goal("Do something").await.unwrap();
    // Fallback uses the original prompt as description
    assert_eq!(goal.description, "Do something");
    // Fallback adds 2 generic criteria
    assert_eq!(goal.success_criteria.len(), 2);
}

#[tokio::test]
async fn test_check_goal_achievement_json_yes() {
    let mock_client = Arc::new(MockLlmClient::new(vec![MockLlmClient::text_response(
        r#"{"achieved": true, "progress": 1.0, "remaining_criteria": []}"#,
    )]));
    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let agent = AgentLoop::new(
        mock_client,
        tool_executor,
        test_tool_context(),
        AgentConfig::default(),
    );

    let goal = crate::planning::AgentGoal::new("Test goal".to_string());
    let achieved = agent
        .check_goal_achievement(&goal, "All done")
        .await
        .unwrap();
    assert!(achieved);
}

#[tokio::test]
async fn test_check_goal_achievement_fallback_not_done() {
    // Non-JSON response triggers heuristic fallback
    let mock_client = Arc::new(MockLlmClient::new(vec![MockLlmClient::text_response(
        "invalid json",
    )]));
    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let agent = AgentLoop::new(
        mock_client,
        tool_executor,
        test_tool_context(),
        AgentConfig::default(),
    );

    let goal = crate::planning::AgentGoal::new("Test goal".to_string());
    // "still working" doesn't contain "complete"/"done"/"finished"
    let achieved = agent
        .check_goal_achievement(&goal, "still working")
        .await
        .unwrap();
    assert!(!achieved);
}

// ========================================================================
// build_augmented_system_prompt Tests
// ========================================================================

#[test]
fn test_build_augmented_system_prompt_empty_context() {
    let mock_client = Arc::new(MockLlmClient::new(vec![]));
    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let config = AgentConfig {
        prompt_slots: SystemPromptSlots {
            extra: Some("Base prompt".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);

    let result = agent.build_augmented_system_prompt(&[]);
    assert!(result.unwrap().contains("Base prompt"));
}

#[test]
fn test_build_augmented_system_prompt_no_custom_slots() {
    let mock_client = Arc::new(MockLlmClient::new(vec![]));
    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let agent = AgentLoop::new(
        mock_client,
        tool_executor,
        test_tool_context(),
        AgentConfig::default(),
    );

    let result = agent.build_augmented_system_prompt(&[]);
    // Default slots still produce the default agentic prompt
    assert!(result.is_some());
    let text = result.unwrap();
    assert!(text.contains("Core Behaviour"));
    // The always-on <env> grounding block is injected at augmentation time.
    assert!(text.contains("<env>"));
    assert!(text.contains("Today's date:"));
    assert!(text.contains("## Boundaries"));
}

#[test]
fn test_project_hint_is_assembled_as_context_item() {
    let temp_dir = tempfile::tempdir().unwrap();
    std::fs::write(
        temp_dir.path().join("Cargo.toml"),
        "[package]\nname = \"demo\"\n",
    )
    .unwrap();

    let mock_client = Arc::new(MockLlmClient::new(vec![]));
    let tool_executor = Arc::new(ToolExecutor::new(temp_dir.path().display().to_string()));
    let agent = AgentLoop::new(
        mock_client,
        tool_executor,
        ToolContext::new(temp_dir.path().to_path_buf()),
        AgentConfig::default(),
    );

    let assembly = agent.assemble_context_results(&[]);
    assert_eq!(assembly.items.len(), 1);
    assert_eq!(
        assembly.items[0].source.as_deref(),
        Some("a3s://project-hint")
    );
    assert!(assembly.items[0].content.contains("Rust"));

    let text = agent.build_augmented_system_prompt(&[]).unwrap();
    assert!(text.contains("<context source=\"a3s://project-hint\" type=\"Resource\">"));
}

#[test]
fn test_build_augmented_system_prompt_with_context_no_base() {
    use crate::context::{ContextItem, ContextResult, ContextType};

    let mock_client = Arc::new(MockLlmClient::new(vec![]));
    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let agent = AgentLoop::new(
        mock_client,
        tool_executor,
        test_tool_context(),
        AgentConfig::default(),
    );

    let context = vec![ContextResult {
        provider: "test".to_string(),
        items: vec![ContextItem::new("id1", ContextType::Resource, "Content")],
        total_tokens: 10,
        truncated: false,
    }];

    let result = agent.build_augmented_system_prompt(&context);
    assert!(result.is_some());
    let text = result.unwrap();
    assert!(text.contains("<context"));
    assert!(text.contains("Content"));
}

// ========================================================================
// AgentResult Clone and Debug
// ========================================================================

#[test]
fn test_agent_result_clone() {
    let result = AgentResult {
        text: "output".to_string(),
        messages: vec![Message::user("hello")],
        usage: TokenUsage::default(),
        tool_calls_count: 3,
        verification_reports: Vec::new(),
    };
    let cloned = result.clone();
    assert_eq!(cloned.text, result.text);
    assert_eq!(cloned.tool_calls_count, result.tool_calls_count);
}

#[test]
fn test_agent_result_debug() {
    let result = AgentResult {
        text: "output".to_string(),
        messages: vec![Message::user("hello")],
        usage: TokenUsage::default(),
        tool_calls_count: 3,
        verification_reports: Vec::new(),
    };
    let debug = format!("{:?}", result);
    assert!(debug.contains("AgentResult"));
    assert!(debug.contains("output"));
}

// ========================================================================
// handle_post_execution_metadata Tests
// ========================================================================

// ========================================================================
// ToolCommand adapter tests
// ========================================================================

#[tokio::test]
async fn test_tool_command_command_type() {
    let executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let cmd = ToolCommand {
        tool_executor: executor,
        tool_name: "read".to_string(),
        tool_args: serde_json::json!({"file": "test.rs"}),
        skill_registry: None,
        tool_context: test_tool_context(),
    };
    assert_eq!(cmd.command_type(), "read");
}

#[tokio::test]
async fn test_tool_command_payload() {
    let executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let args = serde_json::json!({"file": "test.rs", "offset": 10});
    let cmd = ToolCommand {
        tool_executor: executor,
        tool_name: "read".to_string(),
        tool_args: args.clone(),
        skill_registry: None,
        tool_context: test_tool_context(),
    };
    assert_eq!(cmd.payload(), args);
}

// ========================================================================
// AgentLoop with queue builder tests
// ========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_agent_loop_with_queue() {
    use tokio::sync::broadcast;

    let mock_client = Arc::new(MockLlmClient::new(vec![MockLlmClient::text_response(
        "Hello",
    )]));
    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let config = AgentConfig::default();

    let (event_tx, _) = broadcast::channel(100);
    let queue = SessionLaneQueue::new("test-session", SessionQueueConfig::default(), event_tx)
        .await
        .unwrap();

    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config)
        .with_queue(Arc::new(queue));

    assert!(agent.command_queue.is_some());
}

#[tokio::test]
async fn test_agent_loop_without_queue() {
    let mock_client = Arc::new(MockLlmClient::new(vec![MockLlmClient::text_response(
        "Hello",
    )]));
    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let config = AgentConfig::default();

    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);

    assert!(agent.command_queue.is_none());
}

// ========================================================================
// Parallel Plan Execution Tests
// ========================================================================

#[tokio::test]
async fn test_execute_plan_parallel_independent() {
    use crate::planning::{Complexity, ExecutionPlan, Task};

    // 3 independent steps (no dependencies) — should all execute.
    // MockLlmClient needs one response per execute_loop call per step.
    let mock_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::text_response("Step 1 done"),
        MockLlmClient::text_response("Step 2 done"),
        MockLlmClient::text_response("Step 3 done"),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let config = AgentConfig::default();
    let agent = AgentLoop::new(
        mock_client.clone(),
        tool_executor,
        test_tool_context(),
        config,
    );

    let mut plan = ExecutionPlan::new("Test parallel", Complexity::Simple);
    plan.add_step(Task::new("s1", "First step"));
    plan.add_step(Task::new("s2", "Second step"));
    plan.add_step(Task::new("s3", "Third step"));

    let (tx, mut rx) = mpsc::channel(100);
    let result = agent
        .execute_plan(&[], &plan, Some("test-session"), Some(tx))
        .await
        .unwrap();

    // All 3 steps should have been executed (3 * 15 = 45 total tokens)
    assert_eq!(result.usage.total_tokens, 45);

    // Verify we received StepStart and StepEnd events for all 3 steps
    let mut step_starts = Vec::new();
    let mut step_ends = Vec::new();
    rx.close();
    while let Some(event) = rx.recv().await {
        match event {
            AgentEvent::StepStart { step_id, .. } => step_starts.push(step_id),
            AgentEvent::StepEnd {
                step_id, status, ..
            } => {
                assert_eq!(status, TaskStatus::Completed);
                step_ends.push(step_id);
            }
            _ => {}
        }
    }
    assert_eq!(step_starts.len(), 3);
    assert_eq!(step_ends.len(), 3);
}

#[tokio::test]
async fn test_execute_plan_emits_task_list_snapshots() {
    use crate::planning::{Complexity, ExecutionPlan, Task};

    let mock_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::text_response("Step 1 done"),
        MockLlmClient::text_response("Step 2 done"),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let agent = AgentLoop::new(
        mock_client,
        tool_executor,
        test_tool_context(),
        AgentConfig::default(),
    );

    let mut plan = ExecutionPlan::new("Track task list", Complexity::Simple);
    plan.add_step(Task::new("s1", "First step"));
    plan.add_step(Task::new("s2", "Second step").with_dependencies(vec!["s1".to_string()]));

    let (tx, mut rx) = mpsc::channel(100);
    let _ = agent
        .execute_plan(&[], &plan, Some("task-session"), Some(tx))
        .await
        .unwrap();

    let mut snapshots = Vec::new();
    rx.close();
    while let Some(event) = rx.recv().await {
        if let AgentEvent::TaskUpdated { session_id, tasks } = event {
            assert_eq!(session_id, "task-session");
            snapshots.push(tasks);
        }
    }

    assert!(
        snapshots
            .first()
            .unwrap()
            .iter()
            .all(|task| task.status == TaskStatus::Pending),
        "initial snapshot should expose the pending task list"
    );
    assert!(snapshots.iter().any(|tasks| tasks
        .iter()
        .any(|task| task.id == "s1" && task.status == TaskStatus::InProgress)));
    assert!(snapshots.iter().any(|tasks| tasks
        .iter()
        .any(|task| task.id == "s1" && task.status == TaskStatus::Completed)));
    assert!(snapshots
        .last()
        .unwrap()
        .iter()
        .all(|task| task.status == TaskStatus::Completed));
}

#[tokio::test]
async fn test_execute_plan_delegates_task_tool_steps() {
    use crate::planning::{Complexity, ExecutionPlan, Task};
    use crate::subagent::AgentRegistry;
    use crate::tools::register_task;

    let mock_client = Arc::new(MockLlmClient::new(vec![MockLlmClient::text_response(
        "delegated search complete",
    )]));
    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    register_task(
        tool_executor.registry(),
        mock_client,
        Arc::new(AgentRegistry::new()),
        "/tmp".to_string(),
    );
    let agent = AgentLoop::new(
        Arc::new(MockLlmClient::new(vec![])),
        tool_executor,
        test_tool_context(),
        AgentConfig::default(),
    );

    let mut plan = ExecutionPlan::new("Delegate a step", Complexity::Simple);
    plan.add_step(Task::new("s1", "Find the relevant docs").with_tool("task"));

    let (tx, mut rx) = mpsc::channel(100);
    let result = agent
        .execute_plan(&[], &plan, Some("task-session"), Some(tx))
        .await
        .unwrap();

    assert_eq!(result.tool_calls_count, 1);
    assert!(result.text.contains("delegated search complete"));

    let mut saw_task_tool_start = false;
    let mut saw_completed_step = false;
    rx.close();
    while let Some(event) = rx.recv().await {
        match event {
            AgentEvent::ToolStart { name, .. } if name == "task" => {
                saw_task_tool_start = true;
            }
            AgentEvent::StepEnd {
                status: TaskStatus::Completed,
                ..
            } => {
                saw_completed_step = true;
            }
            _ => {}
        }
    }

    assert!(saw_task_tool_start);
    assert!(saw_completed_step);
}

#[tokio::test]
async fn test_execute_plan_delegates_parallel_task_wave_once() {
    use crate::planning::{Complexity, ExecutionPlan, Task};
    use crate::subagent::AgentRegistry;
    use crate::tools::register_task;

    let child_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::text_response("delegated docs complete"),
        MockLlmClient::text_response("delegated tests complete"),
    ]));
    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    register_task(
        tool_executor.registry(),
        child_client,
        Arc::new(AgentRegistry::new()),
        "/tmp".to_string(),
    );
    let agent = AgentLoop::new(
        Arc::new(MockLlmClient::new(vec![])),
        tool_executor,
        test_tool_context(),
        AgentConfig::default(),
    );

    let mut plan = ExecutionPlan::new("Delegate independent wave", Complexity::Medium);
    plan.add_step(Task::new("s1", "Find relevant docs").with_tool("task"));
    plan.add_step(Task::new("s2", "Run verification tests").with_tool("task"));

    let (tx, mut rx) = mpsc::channel(100);
    let result = agent
        .execute_plan(&[], &plan, Some("parallel-task-session"), Some(tx))
        .await
        .unwrap();

    assert_eq!(
        result.tool_calls_count, 1,
        "independent delegated wave should be collapsed into one parallel_task call"
    );
    assert!(result.text.contains("delegated docs complete"));
    assert!(result.text.contains("delegated tests complete"));

    let mut parallel_task_starts = 0;
    let mut completed_steps = Vec::new();
    let mut task_snapshots = Vec::new();
    rx.close();
    while let Some(event) = rx.recv().await {
        match event {
            AgentEvent::ToolStart { name, .. } if name == "parallel_task" => {
                parallel_task_starts += 1;
            }
            AgentEvent::StepEnd {
                step_id,
                status: TaskStatus::Completed,
                ..
            } => completed_steps.push(step_id),
            AgentEvent::TaskUpdated { tasks, .. } => task_snapshots.push(tasks),
            _ => {}
        }
    }

    completed_steps.sort();
    assert_eq!(parallel_task_starts, 1);
    assert_eq!(completed_steps, vec!["s1".to_string(), "s2".to_string()]);
    assert!(task_snapshots.iter().any(|tasks| tasks
        .iter()
        .all(|task| task.status == TaskStatus::InProgress)));
    assert!(task_snapshots
        .last()
        .unwrap()
        .iter()
        .all(|task| task.status == TaskStatus::Completed));
}

#[tokio::test]
async fn test_execute_plan_delegated_parallel_wave_maps_child_failure() {
    use crate::planning::{Complexity, ExecutionPlan, Task};
    use crate::subagent::AgentRegistry;
    use crate::tools::register_task;

    let child_client = Arc::new(MockLlmClient::new(vec![MockLlmClient::text_response(
        "delegated docs complete",
    )]));
    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let registry = Arc::new(AgentRegistry::new());
    registry.unregister("verification");
    register_task(
        tool_executor.registry(),
        child_client,
        registry,
        "/tmp".to_string(),
    );
    let agent = AgentLoop::new(
        Arc::new(MockLlmClient::new(vec![])),
        tool_executor,
        test_tool_context(),
        AgentConfig::default(),
    );

    let mut plan = ExecutionPlan::new("Delegate partially failing wave", Complexity::Medium);
    plan.add_step(Task::new("s1", "Find relevant docs").with_tool("task"));
    plan.add_step(Task::new("s2", "Run verification tests").with_tool("task"));

    let (tx, mut rx) = mpsc::channel(100);
    let result = agent
        .execute_plan(&[], &plan, Some("parallel-task-failure-session"), Some(tx))
        .await
        .unwrap();

    let mut completed_steps = Vec::new();
    let mut failed_steps = Vec::new();
    rx.close();
    while let Some(event) = rx.recv().await {
        if let AgentEvent::StepEnd {
            step_id, status, ..
        } = event
        {
            match status {
                TaskStatus::Completed => completed_steps.push(step_id),
                TaskStatus::Failed => failed_steps.push(step_id),
                _ => {}
            }
        }
    }

    completed_steps.sort();
    failed_steps.sort();
    assert_eq!(completed_steps, vec!["s1".to_string()]);
    assert_eq!(failed_steps, vec!["s2".to_string()]);

    let envelope = result
        .messages
        .iter()
        .rev()
        .filter(|message| message.role == "user")
        .find_map(|message| {
            let text = message
                .content
                .iter()
                .filter_map(|block| {
                    if let crate::llm::ContentBlock::Text { text } = block {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            serde_json::from_str::<serde_json::Value>(&text).ok()
        })
        .expect("parallel result envelope");
    assert_eq!(envelope["type"], "parallel_results");
    let steps = envelope["steps"].as_array().expect("steps");
    assert_eq!(steps[0]["step_id"], "s1");
    assert_eq!(steps[0]["status"], "completed");
    assert_eq!(steps[1]["step_id"], "s2");
    assert_eq!(steps[1]["status"], "failed");
}

#[tokio::test]
async fn test_auto_delegation_runs_parallel_specialists_when_enabled() {
    use crate::prompts::PlanningMode;
    use crate::subagent::AgentRegistry;
    use crate::tools::register_task;

    let mock_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::text_response("review child complete"),
        MockLlmClient::text_response("verification child complete"),
        MockLlmClient::text_response("final answer with automatic context"),
    ]));
    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let agent_registry = Arc::new(AgentRegistry::new());
    register_task(
        tool_executor.registry(),
        mock_client.clone(),
        agent_registry.clone(),
        "/tmp".to_string(),
    );
    let mut auto_delegation = crate::config::AutoDelegationConfig::default();
    auto_delegation.enabled = true;
    auto_delegation.max_tasks = 2;
    let config = AgentConfig {
        planning_mode: PlanningMode::Disabled,
        auto_delegation,
        agent_registry: Some(agent_registry),
        ..AgentConfig::default()
    };
    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);

    let (tx, mut rx) = mpsc::channel(100);
    let result = agent
        .execute_with_session(
            &[],
            "Review the current diff and run regression tests",
            Some("auto-parallel-session"),
            Some(tx),
            None,
        )
        .await
        .unwrap();

    assert_eq!(result.tool_calls_count, 1);
    assert!(result.text.contains("final answer"));

    let mut parallel_task_starts = 0;
    rx.close();
    while let Some(event) = rx.recv().await {
        if let AgentEvent::ToolStart { name, .. } = event {
            if name == "parallel_task" {
                parallel_task_starts += 1;
            }
        }
    }
    assert_eq!(parallel_task_starts, 1);
}

#[tokio::test]
async fn test_auto_delegation_global_parallel_switch_uses_single_task() {
    use crate::prompts::PlanningMode;
    use crate::subagent::AgentRegistry;
    use crate::tools::register_task;

    let mock_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::text_response("single child complete"),
        MockLlmClient::text_response("final answer"),
    ]));
    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let agent_registry = Arc::new(AgentRegistry::new());
    register_task(
        tool_executor.registry(),
        mock_client.clone(),
        agent_registry.clone(),
        "/tmp".to_string(),
    );
    let mut auto_delegation = crate::config::AutoDelegationConfig::default();
    auto_delegation.enabled = true;
    auto_delegation.auto_parallel = false;
    auto_delegation.max_tasks = 2;
    let config = AgentConfig {
        planning_mode: PlanningMode::Disabled,
        auto_delegation,
        agent_registry: Some(agent_registry),
        ..AgentConfig::default()
    };
    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);

    let (tx, mut rx) = mpsc::channel(100);
    let result = agent
        .execute_with_session(
            &[],
            "Review the current diff and run regression tests",
            Some("auto-single-session"),
            Some(tx),
            None,
        )
        .await
        .unwrap();

    assert_eq!(result.tool_calls_count, 1);

    let mut task_starts = 0;
    let mut parallel_task_starts = 0;
    rx.close();
    while let Some(event) = rx.recv().await {
        if let AgentEvent::ToolStart { name, .. } = event {
            if name == "task" {
                task_starts += 1;
            } else if name == "parallel_task" {
                parallel_task_starts += 1;
            }
        }
    }
    assert_eq!(task_starts, 1);
    assert_eq!(parallel_task_starts, 0);
}

#[tokio::test]
async fn test_auto_delegation_disabled_does_not_start_subagents() {
    use crate::prompts::PlanningMode;
    use crate::subagent::AgentRegistry;
    use crate::tools::register_task;

    let mock_client = Arc::new(MockLlmClient::new(vec![MockLlmClient::text_response(
        "final answer without delegation",
    )]));
    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let agent_registry = Arc::new(AgentRegistry::new());
    register_task(
        tool_executor.registry(),
        mock_client.clone(),
        agent_registry.clone(),
        "/tmp".to_string(),
    );
    let config = AgentConfig {
        planning_mode: PlanningMode::Disabled,
        auto_delegation: crate::config::AutoDelegationConfig {
            enabled: false,
            ..Default::default()
        },
        agent_registry: Some(agent_registry),
        ..AgentConfig::default()
    };
    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);

    let (tx, mut rx) = mpsc::channel(100);
    let result = agent
        .execute_with_session(
            &[],
            "Review the current diff and run regression tests",
            Some("auto-disabled-session"),
            Some(tx),
            None,
        )
        .await
        .unwrap();

    assert_eq!(result.tool_calls_count, 0);

    let mut task_tool_starts = 0;
    rx.close();
    while let Some(event) = rx.recv().await {
        if let AgentEvent::ToolStart { name, .. } = event {
            if name == "task" || name == "parallel_task" {
                task_tool_starts += 1;
            }
        }
    }
    assert_eq!(task_tool_starts, 0);
}

#[tokio::test]
async fn test_execute_plan_respects_dependencies() {
    use crate::planning::{Complexity, ExecutionPlan, Task};

    // s1 and s2 are independent (wave 1), s3 depends on both (wave 2).
    // This requires 3 responses total.
    let mock_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::text_response("Step 1 done"),
        MockLlmClient::text_response("Step 2 done"),
        MockLlmClient::text_response("Step 3 done"),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let config = AgentConfig::default();
    let agent = AgentLoop::new(
        mock_client.clone(),
        tool_executor,
        test_tool_context(),
        config,
    );

    let mut plan = ExecutionPlan::new("Test deps", Complexity::Medium);
    plan.add_step(Task::new("s1", "Independent A"));
    plan.add_step(Task::new("s2", "Independent B"));
    plan.add_step(
        Task::new("s3", "Depends on A+B")
            .with_dependencies(vec!["s1".to_string(), "s2".to_string()]),
    );

    let (tx, mut rx) = mpsc::channel(100);
    let result = agent
        .execute_plan(&[], &plan, Some("test-session"), Some(tx))
        .await
        .unwrap();

    // All 3 steps should have been executed (3 * 15 = 45 total tokens)
    assert_eq!(result.usage.total_tokens, 45);

    // Verify ordering: s3's StepStart must come after s1 and s2's StepEnd
    let mut events = Vec::new();
    rx.close();
    while let Some(event) = rx.recv().await {
        match &event {
            AgentEvent::StepStart { step_id, .. } => {
                events.push(format!("start:{}", step_id));
            }
            AgentEvent::StepEnd { step_id, .. } => {
                events.push(format!("end:{}", step_id));
            }
            _ => {}
        }
    }

    // s3 start must occur after both s1 end and s2 end
    let s1_end = events.iter().position(|e| e == "end:s1").unwrap();
    let s2_end = events.iter().position(|e| e == "end:s2").unwrap();
    let s3_start = events.iter().position(|e| e == "start:s3").unwrap();
    assert!(
        s3_start > s1_end,
        "s3 started before s1 ended: {:?}",
        events
    );
    assert!(
        s3_start > s2_end,
        "s3 started before s2 ended: {:?}",
        events
    );

    // Final result should reflect step 3 (last sequential step)
    assert!(result.text.contains("Step 3 done") || !result.text.is_empty());
}

#[tokio::test]
async fn test_execute_plan_handles_step_failure() {
    use crate::planning::{Complexity, ExecutionPlan, Task};

    // s1 succeeds, s2 depends on s1 (succeeds), s3 depends on nothing (succeeds),
    // s4 depends on a step that will fail (s_fail).
    // We simulate failure by providing no responses for s_fail's execute_loop.
    //
    // Simpler approach: s1 succeeds, s2 depends on s1 (will fail because no
    // mock response left), s3 is independent.
    // Layout: s1 (independent), s3 (independent) → wave 1 parallel
    //         s2 depends on s1 → wave 2
    //         s4 depends on s2 → wave 3 (should deadlock since s2 fails)
    let mock_client = Arc::new(MockLlmClient::new(vec![
        // Wave 1: s1 and s3 execute in parallel
        MockLlmClient::text_response("s1 done"),
        MockLlmClient::text_response("s3 done"),
        // Wave 2: s2 executes — but we give it no response, causing failure
        // Actually the MockLlmClient will fail with "No more mock responses"
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let config = AgentConfig::default();
    let agent = AgentLoop::new(
        mock_client.clone(),
        tool_executor,
        test_tool_context(),
        config,
    );

    let mut plan = ExecutionPlan::new("Test failure", Complexity::Medium);
    plan.add_step(Task::new("s1", "Independent step"));
    plan.add_step(Task::new("s2", "Depends on s1").with_dependencies(vec!["s1".to_string()]));
    plan.add_step(Task::new("s3", "Another independent"));
    plan.add_step(Task::new("s4", "Depends on s2").with_dependencies(vec!["s2".to_string()]));

    let (tx, mut rx) = mpsc::channel(100);
    let _result = agent
        .execute_plan(&[], &plan, Some("test-session"), Some(tx))
        .await
        .unwrap();

    // s1 and s3 should succeed (wave 1), s2 should fail (wave 2),
    // s4 should never execute (deadlock — dep s2 failed, not completed)
    let mut completed_steps = Vec::new();
    let mut failed_steps = Vec::new();
    rx.close();
    while let Some(event) = rx.recv().await {
        if let AgentEvent::StepEnd {
            step_id, status, ..
        } = event
        {
            match status {
                TaskStatus::Completed => completed_steps.push(step_id),
                TaskStatus::Failed => failed_steps.push(step_id),
                _ => {}
            }
        }
    }

    assert!(
        completed_steps.contains(&"s1".to_string()),
        "s1 should complete"
    );
    assert!(
        completed_steps.contains(&"s3".to_string()),
        "s3 should complete"
    );
    assert!(failed_steps.contains(&"s2".to_string()), "s2 should fail");
    // s4 should NOT appear in either list — it was never started
    assert!(
        !completed_steps.contains(&"s4".to_string()),
        "s4 should not complete"
    );
    assert!(
        !failed_steps.contains(&"s4".to_string()),
        "s4 should not fail (never started)"
    );
}

// ========================================================================
// Phase 4: Error Recovery & Resilience Tests
// ========================================================================

#[test]
fn test_agent_config_resilience_defaults() {
    let config = AgentConfig::default();
    assert_eq!(config.max_parse_retries, 2);
    assert_eq!(config.tool_timeout_ms, None);
    assert_eq!(config.circuit_breaker_threshold, 3);
}

/// 4.1 — Parse error recovery: bails after max_parse_retries exceeded
#[tokio::test]
async fn test_parse_error_recovery_bails_after_threshold() {
    // 3 parse errors with max_parse_retries=2: count reaches 3 > 2 → bail
    let mock_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::tool_call_response(
            "c1",
            "bash",
            serde_json::json!({"__parse_error": "unexpected token at position 5"}),
        ),
        MockLlmClient::tool_call_response(
            "c2",
            "bash",
            serde_json::json!({"__parse_error": "missing closing brace"}),
        ),
        MockLlmClient::tool_call_response(
            "c3",
            "bash",
            serde_json::json!({"__parse_error": "still broken"}),
        ),
        MockLlmClient::text_response("Done"), // never reached
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let config = AgentConfig {
        max_parse_retries: 2,
        ..AgentConfig::default()
    };
    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);
    let result = agent.execute(&[], "Do something", None).await;
    assert!(result.is_err(), "should bail after parse error threshold");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("malformed tool arguments"),
        "error should mention malformed tool arguments, got: {}",
        err
    );
}

/// 4.1 — Parse error recovery: counter resets after a valid tool execution
#[tokio::test]
async fn test_parse_error_counter_resets_on_success() {
    // 2 parse errors (= max_parse_retries, not yet exceeded)
    // Then a valid tool call (resets counter)
    // Then final text — should NOT bail
    let mock_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::tool_call_response(
            "c1",
            "bash",
            serde_json::json!({"__parse_error": "bad args"}),
        ),
        MockLlmClient::tool_call_response(
            "c2",
            "bash",
            serde_json::json!({"__parse_error": "bad args again"}),
        ),
        // Valid call — resets parse_error_count to 0
        MockLlmClient::tool_call_response("c3", "bash", serde_json::json!({"command": "echo ok"})),
        MockLlmClient::text_response("All done"),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let config = AgentConfig {
        max_parse_retries: 2,
        ..AgentConfig::default()
    };
    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);
    let result = agent.execute(&[], "Do something", None).await;
    assert!(
        result.is_ok(),
        "should not bail — counter reset after successful tool, got: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().text, "All done");
}

/// 4.2 — Tool timeout: slow tool produces a timeout error result; session continues
#[tokio::test]
async fn test_tool_timeout_produces_error_result() {
    let mock_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::tool_call_response("t1", "bash", serde_json::json!({"command": "sleep 10"})),
        MockLlmClient::text_response("The command timed out."),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let config = AgentConfig {
        // 50ms — sleep 10 will never finish
        tool_timeout_ms: Some(50),
        ..AgentConfig::default()
    };
    let agent = AgentLoop::new(
        mock_client.clone(),
        tool_executor,
        test_tool_context(),
        config,
    );
    let result = agent.execute(&[], "Run sleep", None).await;
    assert!(
        result.is_ok(),
        "session should continue after tool timeout: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().text, "The command timed out.");
    // LLM called twice: initial request + response after timeout error
    assert_eq!(mock_client.call_count.load(Ordering::SeqCst), 2);
}

/// 4.2 — Tool timeout: tool that finishes before the deadline succeeds normally
#[tokio::test]
async fn test_tool_within_timeout_succeeds() {
    let mock_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::tool_call_response(
            "t1",
            "bash",
            serde_json::json!({"command": "echo fast"}),
        ),
        MockLlmClient::text_response("Command succeeded."),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let config = AgentConfig {
        tool_timeout_ms: Some(5_000), // 5 s — echo completes in <100ms
        ..AgentConfig::default()
    };
    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);
    let result = agent.execute(&[], "Run something fast", None).await;
    assert!(
        result.is_ok(),
        "fast tool should succeed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().text, "Command succeeded.");
}

/// 4.3 — Circuit breaker: retries non-streaming LLM failures up to threshold
#[tokio::test]
async fn test_circuit_breaker_retries_non_streaming() {
    // Empty response list → every call bails with "No more mock responses"
    // threshold=2 → tries twice, then bails with circuit-breaker message
    let mock_client = Arc::new(MockLlmClient::new(vec![]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let config = AgentConfig {
        circuit_breaker_threshold: 2,
        ..AgentConfig::default()
    };
    let agent = AgentLoop::new(
        mock_client.clone(),
        tool_executor,
        test_tool_context(),
        config,
    );
    let result = agent.execute(&[], "Hello", None).await;
    assert!(result.is_err(), "should fail when LLM always errors");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("circuit breaker"),
        "error should mention circuit breaker, got: {}",
        err
    );
    assert_eq!(
        mock_client.call_count.load(Ordering::SeqCst),
        2,
        "should make exactly threshold=2 LLM calls"
    );
}

/// 4.3 — Circuit breaker: threshold=1 bails on the very first failure
#[tokio::test]
async fn test_circuit_breaker_threshold_one_no_retry() {
    let mock_client = Arc::new(MockLlmClient::new(vec![]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let config = AgentConfig {
        circuit_breaker_threshold: 1,
        ..AgentConfig::default()
    };
    let agent = AgentLoop::new(
        mock_client.clone(),
        tool_executor,
        test_tool_context(),
        config,
    );
    let result = agent.execute(&[], "Hello", None).await;
    assert!(result.is_err());
    assert_eq!(
        mock_client.call_count.load(Ordering::SeqCst),
        1,
        "with threshold=1 exactly one attempt should be made"
    );
}

/// 4.3 — Circuit breaker: succeeds when LLM recovers before hitting threshold
#[tokio::test]
async fn test_circuit_breaker_succeeds_if_llm_recovers() {
    // First call fails, second call succeeds; threshold=3 — recovery within threshold
    struct FailOnceThenSucceed {
        inner: MockLlmClient,
        failed_once: std::sync::atomic::AtomicBool,
        call_count: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl LlmClient for FailOnceThenSucceed {
        async fn complete(
            &self,
            messages: &[Message],
            system: Option<&str>,
            tools: &[ToolDefinition],
        ) -> Result<LlmResponse> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let already_failed = self
                .failed_once
                .swap(true, std::sync::atomic::Ordering::SeqCst);
            if !already_failed {
                anyhow::bail!("transient network error");
            }
            self.inner.complete(messages, system, tools).await
        }

        async fn complete_streaming(
            &self,
            messages: &[Message],
            system: Option<&str>,
            tools: &[ToolDefinition],
            cancel_token: tokio_util::sync::CancellationToken,
        ) -> Result<tokio::sync::mpsc::Receiver<crate::llm::StreamEvent>> {
            self.inner
                .complete_streaming(messages, system, tools, cancel_token)
                .await
        }
    }

    let mock = Arc::new(FailOnceThenSucceed {
        inner: MockLlmClient::new(vec![MockLlmClient::text_response("Recovered!")]),
        failed_once: std::sync::atomic::AtomicBool::new(false),
        call_count: AtomicUsize::new(0),
    });

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let config = AgentConfig {
        circuit_breaker_threshold: 3,
        ..AgentConfig::default()
    };
    let agent = AgentLoop::new(mock.clone(), tool_executor, test_tool_context(), config);
    let result = agent.execute(&[], "Hello", None).await;
    assert!(
        result.is_ok(),
        "should succeed when LLM recovers within threshold: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().text, "Recovered!");
    assert_eq!(
        mock.call_count.load(Ordering::SeqCst),
        2,
        "should have made exactly 2 calls (1 fail + 1 success)"
    );
}

// ── Continuation detection tests ─────────────────────────────────────────

#[test]
fn test_looks_incomplete_empty() {
    assert!(AgentLoop::looks_incomplete(""));
    assert!(AgentLoop::looks_incomplete("   "));
}

#[test]
fn test_looks_incomplete_trailing_colon() {
    assert!(AgentLoop::looks_incomplete("Let me check the file:"));
    assert!(AgentLoop::looks_incomplete("Next steps:"));
}

#[test]
fn test_looks_incomplete_ellipsis() {
    assert!(AgentLoop::looks_incomplete("Working on it..."));
    assert!(AgentLoop::looks_incomplete("Processing…"));
}

#[test]
fn test_looks_incomplete_intent_phrases() {
    assert!(AgentLoop::looks_incomplete(
        "I'll start by reading the file."
    ));
    assert!(AgentLoop::looks_incomplete(
        "Let me check the configuration."
    ));
    assert!(AgentLoop::looks_incomplete("I will now run the tests."));
    assert!(AgentLoop::looks_incomplete(
        "I need to update the Cargo.toml."
    ));
}

#[test]
fn test_looks_complete_final_answer() {
    // Clear final answers should NOT trigger continuation
    assert!(!AgentLoop::looks_incomplete(
        "The tests pass. All changes have been applied successfully."
    ));
    assert!(!AgentLoop::looks_incomplete(
        "Done. I've updated the three files and verified the build succeeds."
    ));
    assert!(!AgentLoop::looks_incomplete("42"));
    assert!(!AgentLoop::looks_incomplete("Yes."));
}

#[test]
fn test_looks_incomplete_multiline_complete() {
    let text =
        "Here is the summary:\n\n- Fixed the bug in agent.rs\n- All tests pass\n- Build succeeds";
    assert!(!AgentLoop::looks_incomplete(text));
}
