use super::*;
use crate::context::{ContextItem, ContextProvider, ContextQuery, ContextResult, ContextType};
use crate::llm::{ContentBlock, StreamEvent};
use crate::permissions::PermissionPolicy;
use crate::prompts::AgentStyle;
use crate::tools::ToolExecutor;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::mpsc;

/// Create a default ToolContext for tests
fn test_tool_context() -> ToolContext {
    ToolContext::new(PathBuf::from("/tmp"))
}

#[test]
fn test_plan_step_delegation_detection() {
    use crate::planning::Task;

    assert!(AgentLoop::should_delegate_plan_step(
        &Task::new("s1", "Find relevant files").with_tool("task")
    ));
    assert!(AgentLoop::should_delegate_plan_step(
        &Task::new("s2", "Check independent areas").with_tool("parallel_task")
    ));
    assert!(!AgentLoop::should_delegate_plan_step(&Task::new(
        "s3",
        "Implement directly"
    )));
}

#[test]
fn test_delegated_agent_selection_from_step_text() {
    use crate::planning::Task;

    assert_eq!(
        AgentLoop::delegated_agent_for_step(&Task::new("s1", "查找相关实现")),
        "explore"
    );
    assert_eq!(
        AgentLoop::delegated_agent_for_step(&Task::new("s2", "Run release verification tests")),
        "verification"
    );
    assert_eq!(
        AgentLoop::delegated_agent_for_step(&Task::new("s3", "Review risky code changes")),
        "review"
    );
    assert_eq!(
        AgentLoop::delegated_agent_for_step(&Task::new("s4", "Design the architecture")),
        "plan"
    );
    assert_eq!(
        AgentLoop::delegated_agent_for_step(&Task::new("s5", "Implement the change")),
        "general"
    );
}

#[test]
fn test_delegated_task_args_include_prompt_contract() {
    use crate::planning::Task;

    let task = Task::new("s1", "验证 program 工具")
        .with_tool("task")
        .with_success_criteria("All integration checks pass.");
    let args = AgentLoop::delegated_task_args(&task, 2, 5);

    assert_eq!(args["agent"], "verification");
    assert_eq!(args["description"], "验证 program 工具");
    assert!(args["prompt"].as_str().unwrap().contains("2/5"));
    assert!(args["prompt"]
        .as_str()
        .unwrap()
        .contains("All integration checks pass."));
}

#[test]
fn test_parallel_delegated_task_args_preserve_order() {
    use crate::planning::Task;

    let steps = vec![
        (Task::new("s1", "Find docs").with_tool("task"), 1),
        (Task::new("s2", "Run tests").with_tool("task"), 2),
    ];
    let args = AgentLoop::parallel_delegated_task_args(&steps, 2);
    let tasks = args["tasks"].as_array().unwrap();

    assert_eq!(tasks.len(), 2);
    assert_eq!(tasks[0]["agent"], "explore");
    assert_eq!(tasks[1]["agent"], "verification");
}

#[test]
fn test_memory_items_become_context_result() {
    let item = a3s_memory::MemoryItem::new("Use focused regression tests for context changes.")
        .with_importance(0.8);

    let result = crate::memory::memory_items_to_context_result("memory", vec![item.clone()]);

    assert_eq!(result.provider, "memory");
    assert_eq!(result.items.len(), 1);
    assert_eq!(result.items[0].id, item.id.as_str());
    assert_eq!(result.items[0].context_type, ContextType::Memory);
    let expected_source = format!("memory://{}", item.id);
    assert_eq!(
        result.items[0].source.as_deref(),
        Some(expected_source.as_str())
    );
    assert!(result.items[0].content.contains("focused regression tests"));
    assert!(result.items[0].token_count > 0);
}

#[cfg(feature = "ahp")]
#[test]
fn test_injected_context_to_results_includes_all_context_shapes() {
    let injected = a3s_ahp::InjectedContext {
        facts: vec![a3s_ahp::Fact {
            content: "Fact from harness".to_string(),
            source: "ahp://fact/source".to_string(),
            confidence: 0.92,
        }],
        file_contents: Some(vec![a3s_ahp::FileContentSnippet {
            path: "src/lib.rs".to_string(),
            snippet: "pub fn important() {}".to_string(),
            relevance_score: 0.88,
        }]),
        project_summary: Some(a3s_ahp::ProjectSummary {
            project_name: "demo".to_string(),
            language: Some("Rust".to_string()),
            key_files: Some(vec!["Cargo.toml".to_string(), "src/lib.rs".to_string()]),
            structure_description: "Small Rust crate".to_string(),
        }),
        knowledge: Some(vec!["Use context budgets".to_string()]),
        suggestions: Some(vec!["Prefer focused verification".to_string()]),
    };

    let results = context_perception::injected_context_to_results(injected);
    let items = results
        .iter()
        .flat_map(|result| result.items.iter())
        .collect::<Vec<_>>();

    assert_eq!(results.len(), 5);
    assert!(items.iter().any(|item| item.content == "Fact from harness"
        && item.source.as_deref() == Some("ahp://fact/source")));
    assert!(items
        .iter()
        .any(|item| item.content == "pub fn important() {}"
            && item.source.as_deref() == Some("src/lib.rs")));
    assert!(items
        .iter()
        .any(|item| item.content.contains("Key files: Cargo.toml, src/lib.rs")));
    assert!(items
        .iter()
        .any(|item| item.source.as_deref() == Some("ahp://suggestions")
            && item.content.contains("Prefer focused verification")));
    assert!(results
        .iter()
        .all(|result| result.provider == "ahp_harness"));
}

#[test]
fn test_agent_config_default() {
    let config = AgentConfig::default();
    assert!(config.prompt_slots.is_empty());
    assert!(config.tools.is_empty()); // Tools are provided externally
    assert_eq!(config.max_tool_rounds, MAX_TOOL_ROUNDS);
    assert_eq!(config.max_parallel_tasks, DEFAULT_MAX_PARALLEL_TASKS);
    assert!(config.permission_checker.is_none());
    assert!(config.context_providers.is_empty());
    // Built-in skills are always present by default
    let registry = config
        .skill_registry
        .expect("skill_registry must be Some by default");
    assert!(registry.len() >= 4, "expected at least 4 built-in skills");
    assert!(registry.get("code-search").is_some());
    assert!(registry.get("find-bugs").is_some());
}

// ========================================================================
// Mock LLM Client for Testing
// ========================================================================

/// Mock LLM client that returns predefined responses
pub(crate) struct MockLlmClient {
    /// Responses to return (consumed in order)
    responses: std::sync::Mutex<Vec<LlmResponse>>,
    /// Number of calls made
    pub(crate) call_count: AtomicUsize,
}

impl MockLlmClient {
    pub(crate) fn new(responses: Vec<LlmResponse>) -> Self {
        Self {
            responses: std::sync::Mutex::new(responses),
            call_count: AtomicUsize::new(0),
        }
    }

    /// Create a response with text only (no tool calls)
    pub(crate) fn text_response(text: &str) -> LlmResponse {
        LlmResponse {
            message: Message {
                role: "assistant".to_string(),
                content: vec![ContentBlock::Text {
                    text: text.to_string(),
                }],
                reasoning_content: None,
            },
            usage: TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                cache_read_tokens: None,
                cache_write_tokens: None,
            },
            stop_reason: Some("end_turn".to_string()),
            meta: None,
        }
    }

    /// Create a response with a tool call
    pub(crate) fn tool_call_response(
        tool_id: &str,
        tool_name: &str,
        args: serde_json::Value,
    ) -> LlmResponse {
        LlmResponse {
            message: Message {
                role: "assistant".to_string(),
                content: vec![ContentBlock::ToolUse {
                    id: tool_id.to_string(),
                    name: tool_name.to_string(),
                    input: args,
                }],
                reasoning_content: None,
            },
            usage: TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                cache_read_tokens: None,
                cache_write_tokens: None,
            },
            stop_reason: Some("tool_use".to_string()),
            meta: None,
        }
    }
}

#[async_trait::async_trait]
impl LlmClient for MockLlmClient {
    async fn complete(
        &self,
        messages: &[Message],
        system: Option<&str>,
        _tools: &[ToolDefinition],
    ) -> Result<LlmResponse> {
        if system == Some(crate::prompts::PRE_ANALYSIS_SYSTEM) {
            let prompt = messages
                .last()
                .and_then(|m| {
                    m.content.iter().find_map(|block| {
                        if let ContentBlock::Text { text } = block {
                            Some(text.as_str())
                        } else {
                            None
                        }
                    })
                })
                .unwrap_or("");
            let response = serde_json::json!({
                "intent": "GeneralPurpose",
                "requires_planning": false,
                "goal": {
                    "description": prompt,
                    "success_criteria": []
                },
                "execution_plan": {
                    "complexity": "Simple",
                    "steps": [
                        {
                            "id": "step-1",
                            "description": prompt,
                            "tool": null,
                            "dependencies": [],
                            "success_criteria": "Complete the request"
                        }
                    ],
                    "required_tools": []
                },
                "optimized_input": prompt
            });
            return Ok(MockLlmClient::text_response(&response.to_string()));
        }
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            anyhow::bail!("No more mock responses available");
        }
        Ok(responses.remove(0))
    }

    async fn complete_streaming(
        &self,
        _messages: &[Message],
        _system: Option<&str>,
        _tools: &[ToolDefinition],
        _cancel_token: tokio_util::sync::CancellationToken,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            anyhow::bail!("No more mock responses available");
        }
        let response = responses.remove(0);

        let (tx, rx) = mpsc::channel(10);
        tokio::spawn(async move {
            // Send text deltas if any
            for block in &response.message.content {
                if let ContentBlock::Text { text } = block {
                    tx.send(StreamEvent::TextDelta(text.clone())).await.ok();
                }
            }
            tx.send(StreamEvent::Done(response)).await.ok();
        });

        Ok(rx)
    }
}

// ========================================================================
// Agent Loop Tests
// ========================================================================

#[tokio::test]
async fn test_agent_simple_response() {
    let mock_client = Arc::new(MockLlmClient::new(vec![MockLlmClient::text_response(
        "Hello, I'm an AI assistant.",
    )]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let config = AgentConfig::default();

    let agent = AgentLoop::new(
        mock_client.clone(),
        tool_executor,
        test_tool_context(),
        config,
    );
    let result = agent.execute(&[], "Hello", None).await.unwrap();

    assert_eq!(result.text, "Hello, I'm an AI assistant.");
    assert_eq!(result.tool_calls_count, 0);
    assert_eq!(mock_client.call_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_agent_with_tool_call() {
    let mock_client = Arc::new(MockLlmClient::new(vec![
        // First response: tool call
        MockLlmClient::tool_call_response(
            "tool-1",
            "bash",
            serde_json::json!({"command": "echo hello"}),
        ),
        // Second response: final text
        MockLlmClient::text_response("The command output was: hello"),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let config = AgentConfig::default();

    let agent = AgentLoop::new(
        mock_client.clone(),
        tool_executor,
        test_tool_context(),
        config,
    );
    let result = agent.execute(&[], "Run echo hello", None).await.unwrap();

    assert_eq!(result.text, "The command output was: hello");
    assert_eq!(result.tool_calls_count, 1);
    assert_eq!(mock_client.call_count.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn test_agent_permission_deny() {
    let mock_client = Arc::new(MockLlmClient::new(vec![
        // First response: tool call that will be denied
        MockLlmClient::tool_call_response(
            "tool-1",
            "bash",
            serde_json::json!({"command": "rm -rf /tmp/test"}),
        ),
        // Second response: LLM responds to the denial
        MockLlmClient::text_response(
            "I cannot execute that command due to permission restrictions.",
        ),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));

    // Create permission policy that denies rm commands
    let permission_policy = PermissionPolicy::new().deny("bash(rm:*)");

    let config = AgentConfig {
        permission_checker: Some(Arc::new(permission_policy)),
        ..Default::default()
    };

    let (tx, mut rx) = mpsc::channel(100);
    let agent = AgentLoop::new(
        mock_client.clone(),
        tool_executor,
        test_tool_context(),
        config,
    );
    let result = agent.execute(&[], "Delete files", Some(tx)).await.unwrap();

    // Check that we received a PermissionDenied event
    let mut found_permission_denied = false;
    while let Ok(event) = rx.try_recv() {
        if let AgentEvent::PermissionDenied { tool_name, .. } = event {
            assert_eq!(tool_name, "bash");
            found_permission_denied = true;
        }
    }
    assert!(
        found_permission_denied,
        "Should have received PermissionDenied event"
    );

    assert_eq!(result.tool_calls_count, 1);
}

#[tokio::test]
async fn test_agent_permission_allow() {
    let mock_client = Arc::new(MockLlmClient::new(vec![
        // First response: tool call that will be allowed
        MockLlmClient::tool_call_response(
            "tool-1",
            "bash",
            serde_json::json!({"command": "echo hello"}),
        ),
        // Second response: final text
        MockLlmClient::text_response("Done!"),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));

    // Create permission policy that allows echo commands
    let permission_policy = PermissionPolicy::new()
        .allow("bash(echo:*)")
        .deny("bash(rm:*)");

    let config = AgentConfig {
        permission_checker: Some(Arc::new(permission_policy)),
        ..Default::default()
    };

    let agent = AgentLoop::new(
        mock_client.clone(),
        tool_executor,
        test_tool_context(),
        config,
    );
    let result = agent.execute(&[], "Echo hello", None).await.unwrap();

    assert_eq!(result.text, "Done!");
    assert_eq!(result.tool_calls_count, 1);
}

#[tokio::test]
async fn test_agent_streaming_events() {
    let mock_client = Arc::new(MockLlmClient::new(vec![MockLlmClient::text_response(
        "Hello!",
    )]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let config = AgentConfig::default();

    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);
    let (tx, mut rx) = mpsc::channel(100);
    let cancel_token = tokio_util::sync::CancellationToken::new();

    let result = agent
        .execute_with_session(&[], "Hi", None, Some(tx), Some(&cancel_token))
        .await
        .unwrap();
    let mut events = Vec::new();
    while let Some(event) = rx.recv().await {
        events.push(event);
    }

    assert_eq!(result.text, "Hello!");

    // Check we received Start and End events
    assert!(events.iter().any(|e| matches!(e, AgentEvent::Start { .. })));
    assert!(events.iter().any(|e| matches!(e, AgentEvent::End { .. })));
}

#[tokio::test]
async fn test_agent_max_tool_rounds() {
    // Create a mock that always returns tool calls (infinite loop)
    let responses: Vec<LlmResponse> = (0..100)
        .map(|i| {
            MockLlmClient::tool_call_response(
                &format!("tool-{}", i),
                "bash",
                serde_json::json!({"command": "echo loop"}),
            )
        })
        .collect();

    let mock_client = Arc::new(MockLlmClient::new(responses));
    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));

    let config = AgentConfig {
        max_tool_rounds: 3,
        ..Default::default()
    };

    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);
    let result = agent.execute(&[], "Loop forever", None).await;

    // Should fail due to max tool rounds exceeded
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Max tool rounds"));
}

#[tokio::test]
async fn test_agent_no_permission_policy_defaults_to_ask() {
    // When no permission policy is set, tools default to Ask.
    // Without a confirmation manager, Ask = safe deny.
    let mock_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::tool_call_response(
            "tool-1",
            "bash",
            serde_json::json!({"command": "rm -rf /tmp/test"}),
        ),
        MockLlmClient::text_response("Denied!"),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let config = AgentConfig {
        permission_checker: None, // No policy → defaults to Ask
        // No confirmation_manager → safe deny
        ..Default::default()
    };

    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);
    let result = agent.execute(&[], "Delete", None).await.unwrap();

    // Should be denied (no policy + no CM = safe deny)
    assert_eq!(result.text, "Denied!");
    assert_eq!(result.tool_calls_count, 1);
}

#[tokio::test]
async fn test_agent_permission_ask_without_cm_denies() {
    // When permission is Ask and no confirmation manager configured,
    // tool execution should be denied (safe default).
    let mock_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::tool_call_response(
            "tool-1",
            "bash",
            serde_json::json!({"command": "echo test"}),
        ),
        MockLlmClient::text_response("Denied!"),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));

    // Create policy where bash falls through to Ask (default)
    let permission_policy = PermissionPolicy::new(); // Default decision is Ask

    let config = AgentConfig {
        permission_checker: Some(Arc::new(permission_policy)),
        // No confirmation_manager — safe deny
        ..Default::default()
    };

    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);
    let result = agent.execute(&[], "Echo", None).await.unwrap();

    // Should deny (Ask without CM = safe deny)
    assert_eq!(result.text, "Denied!");
    // The tool result should contain the denial message
    assert!(result.tool_calls_count >= 1);
}

// ========================================================================
// HITL (Human-in-the-Loop) Tests
// ========================================================================

#[tokio::test]
async fn test_agent_hitl_approved() {
    use crate::hitl::{ConfirmationManager, ConfirmationPolicy};
    use tokio::sync::broadcast;

    let mock_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::tool_call_response(
            "tool-1",
            "bash",
            serde_json::json!({"command": "echo hello"}),
        ),
        MockLlmClient::text_response("Command executed!"),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));

    // Create HITL confirmation manager with policy enabled
    let (event_tx, _event_rx) = broadcast::channel(100);
    let hitl_policy = ConfirmationPolicy {
        enabled: true,
        ..Default::default()
    };
    let confirmation_manager = Arc::new(ConfirmationManager::new(hitl_policy, event_tx));

    // Create permission policy that returns Ask for bash
    let permission_policy = PermissionPolicy::new(); // Default is Ask

    let config = AgentConfig {
        permission_checker: Some(Arc::new(permission_policy)),
        confirmation_manager: Some(confirmation_manager.clone()),
        ..Default::default()
    };

    // Spawn a task to approve the confirmation
    let cm_clone = confirmation_manager.clone();
    tokio::spawn(async move {
        // Wait a bit for the confirmation request to be created
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        // Approve it
        cm_clone.confirm("tool-1", true, None).await.ok();
    });

    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);
    let result = agent.execute(&[], "Run echo", None).await.unwrap();

    assert_eq!(result.text, "Command executed!");
    assert_eq!(result.tool_calls_count, 1);
}

#[tokio::test]
async fn test_agent_hitl_rejected() {
    use crate::hitl::{ConfirmationManager, ConfirmationPolicy};
    use tokio::sync::broadcast;

    let mock_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::tool_call_response(
            "tool-1",
            "bash",
            serde_json::json!({"command": "rm -rf /"}),
        ),
        MockLlmClient::text_response("Understood, I won't do that."),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));

    // Create HITL confirmation manager
    let (event_tx, _event_rx) = broadcast::channel(100);
    let hitl_policy = ConfirmationPolicy {
        enabled: true,
        ..Default::default()
    };
    let confirmation_manager = Arc::new(ConfirmationManager::new(hitl_policy, event_tx));

    // Permission policy returns Ask
    let permission_policy = PermissionPolicy::new();

    let config = AgentConfig {
        permission_checker: Some(Arc::new(permission_policy)),
        confirmation_manager: Some(confirmation_manager.clone()),
        ..Default::default()
    };

    // Spawn a task to reject the confirmation
    let cm_clone = confirmation_manager.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        cm_clone
            .confirm("tool-1", false, Some("Too dangerous".to_string()))
            .await
            .ok();
    });

    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);
    let result = agent.execute(&[], "Delete everything", None).await.unwrap();

    // LLM should respond to the rejection
    assert_eq!(result.text, "Understood, I won't do that.");
}

#[tokio::test]
async fn test_agent_hitl_timeout_reject() {
    use crate::hitl::{ConfirmationManager, ConfirmationPolicy, TimeoutAction};
    use tokio::sync::broadcast;

    let mock_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::tool_call_response(
            "tool-1",
            "bash",
            serde_json::json!({"command": "echo test"}),
        ),
        MockLlmClient::text_response("Timed out, I understand."),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));

    // Create HITL with very short timeout and Reject action
    let (event_tx, _event_rx) = broadcast::channel(100);
    let hitl_policy = ConfirmationPolicy {
        enabled: true,
        default_timeout_ms: 50, // Very short timeout
        timeout_action: TimeoutAction::Reject,
        ..Default::default()
    };
    let confirmation_manager = Arc::new(ConfirmationManager::new(hitl_policy, event_tx));

    let permission_policy = PermissionPolicy::new();

    let config = AgentConfig {
        permission_checker: Some(Arc::new(permission_policy)),
        confirmation_manager: Some(confirmation_manager),
        ..Default::default()
    };

    // Don't approve - let it timeout
    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);
    let result = agent.execute(&[], "Echo", None).await.unwrap();

    // Should get timeout rejection response from LLM
    assert_eq!(result.text, "Timed out, I understand.");
}

#[tokio::test]
async fn test_agent_hitl_timeout_auto_approve() {
    use crate::hitl::{ConfirmationManager, ConfirmationPolicy, TimeoutAction};
    use tokio::sync::broadcast;

    let mock_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::tool_call_response(
            "tool-1",
            "bash",
            serde_json::json!({"command": "echo hello"}),
        ),
        MockLlmClient::text_response("Auto-approved and executed!"),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));

    // Create HITL with very short timeout and AutoApprove action
    let (event_tx, _event_rx) = broadcast::channel(100);
    let hitl_policy = ConfirmationPolicy {
        enabled: true,
        default_timeout_ms: 50, // Very short timeout
        timeout_action: TimeoutAction::AutoApprove,
        ..Default::default()
    };
    let confirmation_manager = Arc::new(ConfirmationManager::new(hitl_policy, event_tx));

    let permission_policy = PermissionPolicy::new();

    let config = AgentConfig {
        permission_checker: Some(Arc::new(permission_policy)),
        confirmation_manager: Some(confirmation_manager),
        ..Default::default()
    };

    // Don't approve - let it timeout and auto-approve
    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);
    let result = agent.execute(&[], "Echo", None).await.unwrap();

    // Should auto-approve on timeout and execute
    assert_eq!(result.text, "Auto-approved and executed!");
    assert_eq!(result.tool_calls_count, 1);
}

#[tokio::test]
async fn test_agent_hitl_confirmation_events() {
    use crate::hitl::{ConfirmationManager, ConfirmationPolicy};
    use tokio::sync::broadcast;

    let mock_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::tool_call_response(
            "tool-1",
            "bash",
            serde_json::json!({"command": "echo test"}),
        ),
        MockLlmClient::text_response("Done!"),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));

    // Create HITL confirmation manager
    let (event_tx, mut event_rx) = broadcast::channel(100);
    let hitl_policy = ConfirmationPolicy {
        enabled: true,
        default_timeout_ms: 5000, // Long enough timeout
        ..Default::default()
    };
    let confirmation_manager = Arc::new(ConfirmationManager::new(hitl_policy, event_tx));

    let permission_policy = PermissionPolicy::new();

    let config = AgentConfig {
        permission_checker: Some(Arc::new(permission_policy)),
        confirmation_manager: Some(confirmation_manager.clone()),
        ..Default::default()
    };

    // Spawn task to approve and collect events
    let cm_clone = confirmation_manager.clone();
    let event_handle = tokio::spawn(async move {
        let mut events = Vec::new();
        // Wait for ConfirmationRequired event
        while let Ok(event) = event_rx.recv().await {
            events.push(event.clone());
            if let AgentEvent::ConfirmationRequired { tool_id, .. } = event {
                // Approve it
                cm_clone.confirm(&tool_id, true, None).await.ok();
                // Wait for ConfirmationReceived
                if let Ok(recv_event) = event_rx.recv().await {
                    events.push(recv_event);
                }
                break;
            }
        }
        events
    });

    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);
    let _result = agent.execute(&[], "Echo", None).await.unwrap();

    // Check events
    let events = event_handle.await.unwrap();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::ConfirmationRequired { .. })),
        "Should have ConfirmationRequired event"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::ConfirmationReceived { approved: true, .. })),
        "Should have ConfirmationReceived event with approved=true"
    );
}

#[tokio::test]
async fn test_agent_hitl_disabled_auto_executes() {
    // When HITL is disabled, tools should execute automatically even with Ask permission
    use crate::hitl::{ConfirmationManager, ConfirmationPolicy};
    use tokio::sync::broadcast;

    let mock_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::tool_call_response(
            "tool-1",
            "bash",
            serde_json::json!({"command": "echo auto"}),
        ),
        MockLlmClient::text_response("Auto executed!"),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));

    // Create HITL with enabled=false
    let (event_tx, _event_rx) = broadcast::channel(100);
    let hitl_policy = ConfirmationPolicy {
        enabled: false, // HITL disabled
        ..Default::default()
    };
    let confirmation_manager = Arc::new(ConfirmationManager::new(hitl_policy, event_tx));

    let permission_policy = PermissionPolicy::new(); // Default is Ask

    let config = AgentConfig {
        permission_checker: Some(Arc::new(permission_policy)),
        confirmation_manager: Some(confirmation_manager),
        ..Default::default()
    };

    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);
    let result = agent.execute(&[], "Echo", None).await.unwrap();

    // Should execute without waiting for confirmation
    assert_eq!(result.text, "Auto executed!");
    assert_eq!(result.tool_calls_count, 1);
}

#[tokio::test]
async fn test_agent_hitl_with_permission_deny_skips_hitl() {
    // When permission is Deny, HITL should not be triggered
    use crate::hitl::{ConfirmationManager, ConfirmationPolicy};
    use tokio::sync::broadcast;

    let mock_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::tool_call_response(
            "tool-1",
            "bash",
            serde_json::json!({"command": "rm -rf /"}),
        ),
        MockLlmClient::text_response("Blocked by permission."),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));

    // Create HITL enabled
    let (event_tx, mut event_rx) = broadcast::channel(100);
    let hitl_policy = ConfirmationPolicy {
        enabled: true,
        ..Default::default()
    };
    let confirmation_manager = Arc::new(ConfirmationManager::new(hitl_policy, event_tx));

    // Permission policy denies rm commands
    let permission_policy = PermissionPolicy::new().deny("bash(rm:*)");

    let config = AgentConfig {
        permission_checker: Some(Arc::new(permission_policy)),
        confirmation_manager: Some(confirmation_manager),
        ..Default::default()
    };

    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);
    let result = agent.execute(&[], "Delete", None).await.unwrap();

    // Should be denied without HITL
    assert_eq!(result.text, "Blocked by permission.");

    // Should NOT have any ConfirmationRequired events
    let mut found_confirmation = false;
    while let Ok(event) = event_rx.try_recv() {
        if matches!(event, AgentEvent::ConfirmationRequired { .. }) {
            found_confirmation = true;
        }
    }
    assert!(
        !found_confirmation,
        "HITL should not be triggered when permission is Deny"
    );
}

#[tokio::test]
async fn test_agent_hitl_with_permission_allow_skips_hitl() {
    // When permission is Allow, HITL confirmation is skipped entirely.
    // PermissionPolicy is the declarative rule engine; Allow = execute directly.
    use crate::hitl::{ConfirmationManager, ConfirmationPolicy};
    use tokio::sync::broadcast;

    let mock_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::tool_call_response(
            "tool-1",
            "bash",
            serde_json::json!({"command": "echo hello"}),
        ),
        MockLlmClient::text_response("Allowed!"),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));

    // Create HITL enabled
    let (event_tx, mut event_rx) = broadcast::channel(100);
    let hitl_policy = ConfirmationPolicy {
        enabled: true,
        ..Default::default()
    };
    let confirmation_manager = Arc::new(ConfirmationManager::new(hitl_policy, event_tx));

    // Permission policy allows echo commands
    let permission_policy = PermissionPolicy::new().allow("bash(echo:*)");

    let config = AgentConfig {
        permission_checker: Some(Arc::new(permission_policy)),
        confirmation_manager: Some(confirmation_manager.clone()),
        ..Default::default()
    };

    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);
    let result = agent.execute(&[], "Echo", None).await.unwrap();

    // Should execute directly without HITL (permission Allow skips confirmation)
    assert_eq!(result.text, "Allowed!");

    // Should NOT have ConfirmationRequired event (Allow bypasses HITL)
    let mut found_confirmation = false;
    while let Ok(event) = event_rx.try_recv() {
        if matches!(event, AgentEvent::ConfirmationRequired { .. }) {
            found_confirmation = true;
        }
    }
    assert!(
        !found_confirmation,
        "Permission Allow should skip HITL confirmation"
    );
}

#[tokio::test]
async fn test_agent_hitl_multiple_tool_calls() {
    // Test multiple tool calls in sequence with HITL
    use crate::hitl::{ConfirmationManager, ConfirmationPolicy};
    use tokio::sync::broadcast;

    let mock_client = Arc::new(MockLlmClient::new(vec![
        // First response: two tool calls
        LlmResponse {
            message: Message {
                role: "assistant".to_string(),
                content: vec![
                    ContentBlock::ToolUse {
                        id: "tool-1".to_string(),
                        name: "bash".to_string(),
                        input: serde_json::json!({"command": "echo first"}),
                    },
                    ContentBlock::ToolUse {
                        id: "tool-2".to_string(),
                        name: "bash".to_string(),
                        input: serde_json::json!({"command": "echo second"}),
                    },
                ],
                reasoning_content: None,
            },
            usage: TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                cache_read_tokens: None,
                cache_write_tokens: None,
            },
            stop_reason: Some("tool_use".to_string()),
            meta: None,
        },
        MockLlmClient::text_response("Both executed!"),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));

    // Create HITL
    let (event_tx, _event_rx) = broadcast::channel(100);
    let hitl_policy = ConfirmationPolicy {
        enabled: true,
        default_timeout_ms: 5000,
        ..Default::default()
    };
    let confirmation_manager = Arc::new(ConfirmationManager::new(hitl_policy, event_tx));

    let permission_policy = PermissionPolicy::new(); // Default Ask

    let config = AgentConfig {
        permission_checker: Some(Arc::new(permission_policy)),
        confirmation_manager: Some(confirmation_manager.clone()),
        ..Default::default()
    };

    // Spawn task to approve both tools
    let cm_clone = confirmation_manager.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        cm_clone.confirm("tool-1", true, None).await.ok();
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        cm_clone.confirm("tool-2", true, None).await.ok();
    });

    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);
    let result = agent
        .execute_loop(
            &[],
            "run both commands now",
            AgentStyle::GeneralPurpose,
            None,
            None,
            &tokio_util::sync::CancellationToken::new(),
            true,
        )
        .await
        .unwrap();

    assert_eq!(result.text, "Both executed!");
    assert_eq!(result.tool_calls_count, 2);
}

#[tokio::test]
async fn test_agent_hitl_partial_approval() {
    // Test: first tool approved, second rejected
    use crate::hitl::{ConfirmationManager, ConfirmationPolicy};
    use tokio::sync::broadcast;

    let mock_client = Arc::new(MockLlmClient::new(vec![
        // First response: two tool calls
        LlmResponse {
            message: Message {
                role: "assistant".to_string(),
                content: vec![
                    ContentBlock::ToolUse {
                        id: "tool-1".to_string(),
                        name: "bash".to_string(),
                        input: serde_json::json!({"command": "echo safe"}),
                    },
                    ContentBlock::ToolUse {
                        id: "tool-2".to_string(),
                        name: "bash".to_string(),
                        input: serde_json::json!({"command": "rm -rf /"}),
                    },
                ],
                reasoning_content: None,
            },
            usage: TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                cache_read_tokens: None,
                cache_write_tokens: None,
            },
            stop_reason: Some("tool_use".to_string()),
            meta: None,
        },
        MockLlmClient::text_response("First worked, second rejected."),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));

    let (event_tx, _event_rx) = broadcast::channel(100);
    let hitl_policy = ConfirmationPolicy {
        enabled: true,
        default_timeout_ms: 5000,
        ..Default::default()
    };
    let confirmation_manager = Arc::new(ConfirmationManager::new(hitl_policy, event_tx));

    let permission_policy = PermissionPolicy::new();

    let config = AgentConfig {
        permission_checker: Some(Arc::new(permission_policy)),
        confirmation_manager: Some(confirmation_manager.clone()),
        ..Default::default()
    };

    // Approve first, reject second
    let cm_clone = confirmation_manager.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        cm_clone.confirm("tool-1", true, None).await.ok();
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        cm_clone
            .confirm("tool-2", false, Some("Dangerous".to_string()))
            .await
            .ok();
    });

    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);
    let result = agent.execute(&[], "Run both", None).await.unwrap();

    assert_eq!(result.text, "First worked, second rejected.");
    assert_eq!(result.tool_calls_count, 2);
}

#[tokio::test]
async fn test_agent_hitl_yolo_mode_auto_approves() {
    // YOLO mode: specific lanes auto-approve without confirmation
    use crate::hitl::{ConfirmationManager, ConfirmationPolicy};
    use crate::queue::SessionLane;
    use tokio::sync::broadcast;

    let mock_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::tool_call_response(
            "tool-1",
            "read", // Query lane tool
            serde_json::json!({"path": "/tmp/test.txt"}),
        ),
        MockLlmClient::text_response("File read!"),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));

    // YOLO mode for Query lane (read, glob, ls, grep)
    let (event_tx, mut event_rx) = broadcast::channel(100);
    let mut yolo_lanes = std::collections::HashSet::new();
    yolo_lanes.insert(SessionLane::Query);
    let hitl_policy = ConfirmationPolicy {
        enabled: true,
        yolo_lanes, // Auto-approve query operations
        ..Default::default()
    };
    let confirmation_manager = Arc::new(ConfirmationManager::new(hitl_policy, event_tx));

    let permission_policy = PermissionPolicy::new();

    let config = AgentConfig {
        permission_checker: Some(Arc::new(permission_policy)),
        confirmation_manager: Some(confirmation_manager),
        ..Default::default()
    };

    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);
    let result = agent.execute(&[], "Read file", None).await.unwrap();

    // Should auto-execute without confirmation (YOLO mode)
    assert_eq!(result.text, "File read!");

    // Should NOT have ConfirmationRequired for yolo lane
    let mut found_confirmation = false;
    while let Ok(event) = event_rx.try_recv() {
        if matches!(event, AgentEvent::ConfirmationRequired { .. }) {
            found_confirmation = true;
        }
    }
    assert!(
        !found_confirmation,
        "YOLO mode should not trigger confirmation"
    );
}

#[tokio::test]
async fn test_agent_config_with_all_options() {
    use crate::hitl::{ConfirmationManager, ConfirmationPolicy};
    use tokio::sync::broadcast;

    let (event_tx, _) = broadcast::channel(100);
    let hitl_policy = ConfirmationPolicy::default();
    let confirmation_manager = Arc::new(ConfirmationManager::new(hitl_policy, event_tx));

    let permission_policy = PermissionPolicy::new().allow("bash(*)");

    let config = AgentConfig {
        prompt_slots: SystemPromptSlots {
            extra: Some("Test system prompt".to_string()),
            ..Default::default()
        },
        tools: vec![],
        max_tool_rounds: 10,
        permission_checker: Some(Arc::new(permission_policy)),
        confirmation_manager: Some(confirmation_manager),
        context_providers: vec![],
        planning_mode: PlanningMode::default(),
        goal_tracking: false,
        hook_engine: None,
        skill_registry: None,
        ..AgentConfig::default()
    };

    assert!(config.prompt_slots.build().contains("Test system prompt"));
    assert_eq!(config.max_tool_rounds, 10);
    assert!(config.permission_checker.is_some());
    assert!(config.confirmation_manager.is_some());
    assert!(config.context_providers.is_empty());

    // Test Debug trait
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("AgentConfig"));
    assert!(debug_str.contains("permission_checker: true"));
    assert!(debug_str.contains("confirmation_manager: true"));
    assert!(debug_str.contains("context_providers: 0"));
}

// ========================================================================
// Context Provider Tests
// ========================================================================

/// Mock context provider for testing
struct MockContextProvider {
    name: String,
    items: Vec<ContextItem>,
    on_turn_calls: std::sync::Arc<tokio::sync::RwLock<Vec<(String, String, String)>>>,
}

impl MockContextProvider {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            items: Vec::new(),
            on_turn_calls: std::sync::Arc::new(tokio::sync::RwLock::new(Vec::new())),
        }
    }

    fn with_items(mut self, items: Vec<ContextItem>) -> Self {
        self.items = items;
        self
    }
}

#[async_trait::async_trait]
impl ContextProvider for MockContextProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn query(&self, _query: &ContextQuery) -> anyhow::Result<ContextResult> {
        let mut result = ContextResult::new(&self.name);
        for item in &self.items {
            result.add_item(item.clone());
        }
        Ok(result)
    }

    async fn on_turn_complete(
        &self,
        session_id: &str,
        prompt: &str,
        response: &str,
    ) -> anyhow::Result<()> {
        let mut calls = self.on_turn_calls.write().await;
        calls.push((
            session_id.to_string(),
            prompt.to_string(),
            response.to_string(),
        ));
        Ok(())
    }
}

#[tokio::test]
async fn test_agent_with_context_provider() {
    let mock_client = Arc::new(MockLlmClient::new(vec![MockLlmClient::text_response(
        "Response using context",
    )]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));

    let provider = MockContextProvider::new("test-provider").with_items(vec![ContextItem::new(
        "ctx-1",
        ContextType::Resource,
        "Relevant context here",
    )
    .with_source("test://docs/example")]);

    let config = AgentConfig {
        prompt_slots: SystemPromptSlots {
            extra: Some("You are helpful.".to_string()),
            ..Default::default()
        },
        context_providers: vec![Arc::new(provider)],
        ..Default::default()
    };

    let agent = AgentLoop::new(
        mock_client.clone(),
        tool_executor,
        test_tool_context(),
        config,
    );
    let result = agent
        .execute(&[], "verify context provider output", None)
        .await
        .unwrap();

    assert_eq!(result.text, "Response using context");
    assert_eq!(mock_client.call_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_agent_context_provider_events() {
    let mock_client = Arc::new(MockLlmClient::new(vec![MockLlmClient::text_response(
        "Answer",
    )]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));

    let provider = MockContextProvider::new("event-provider").with_items(vec![ContextItem::new(
        "item-1",
        ContextType::Memory,
        "Memory content",
    )
    .with_token_count(50)]);

    let config = AgentConfig {
        context_providers: vec![Arc::new(provider)],
        ..Default::default()
    };

    let (tx, mut rx) = mpsc::channel(100);
    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);
    let _result = agent.execute(&[], "Test prompt", Some(tx)).await.unwrap();

    // Collect events
    let mut events = Vec::new();
    while let Ok(event) = rx.try_recv() {
        events.push(event);
    }

    // Should have ContextResolving and ContextResolved events
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::ContextResolving { .. })),
        "Should have ContextResolving event"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::ContextResolved { .. })),
        "Should have ContextResolved event"
    );

    // Check context resolved values
    for event in &events {
        if let AgentEvent::ContextResolved {
            total_items,
            total_tokens,
        } = event
        {
            assert_eq!(*total_items, 1);
            assert_eq!(*total_tokens, 50);
        }
    }
}

#[tokio::test]
async fn test_agent_multiple_context_providers() {
    let mock_client = Arc::new(MockLlmClient::new(vec![MockLlmClient::text_response(
        "Combined response",
    )]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));

    let provider1 = MockContextProvider::new("provider-1").with_items(vec![ContextItem::new(
        "p1-1",
        ContextType::Resource,
        "Resource from P1",
    )
    .with_token_count(100)]);

    let provider2 = MockContextProvider::new("provider-2").with_items(vec![
        ContextItem::new("p2-1", ContextType::Memory, "Memory from P2").with_token_count(50),
        ContextItem::new("p2-2", ContextType::Skill, "Skill from P2").with_token_count(75),
    ]);

    let config = AgentConfig {
        prompt_slots: SystemPromptSlots {
            extra: Some("Base system prompt.".to_string()),
            ..Default::default()
        },
        context_providers: vec![Arc::new(provider1), Arc::new(provider2)],
        ..Default::default()
    };

    let (tx, mut rx) = mpsc::channel(100);
    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);
    let result = agent
        .execute(&[], "verify combined context", Some(tx))
        .await
        .unwrap();

    assert_eq!(result.text, "Combined response");

    // Check context resolved event has combined totals
    while let Ok(event) = rx.try_recv() {
        if let AgentEvent::ContextResolved {
            total_items,
            total_tokens,
        } = event
        {
            assert_eq!(total_items, 3); // 1 + 2
            assert_eq!(total_tokens, 225); // 100 + 50 + 75
        }
    }
}

#[tokio::test]
async fn test_agent_no_context_providers() {
    let mock_client = Arc::new(MockLlmClient::new(vec![MockLlmClient::text_response(
        "No context",
    )]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));

    // No context providers
    let config = AgentConfig::default();

    let (tx, mut rx) = mpsc::channel(100);
    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);
    let result = agent
        .execute(&[], "verify simple prompt", Some(tx))
        .await
        .unwrap();

    assert_eq!(result.text, "No context");

    // Should NOT have context events when no providers
    let mut events = Vec::new();
    while let Ok(event) = rx.try_recv() {
        events.push(event);
    }

    assert!(
        !events
            .iter()
            .any(|e| matches!(e, AgentEvent::ContextResolving { .. })),
        "Should NOT have ContextResolving event"
    );
}

#[tokio::test]
async fn test_agent_memory_recall_routes_through_context_assembly() {
    let mock_client = Arc::new(MockLlmClient::new(vec![MockLlmClient::text_response(
        "Memory-aware response",
    )]));

    let memory = crate::memory::AgentMemory::new(Arc::new(a3s_memory::InMemoryStore::new()));
    memory
        .remember(
            a3s_memory::MemoryItem::new(
                "verify focused regression tests caught context regressions.",
            )
            .with_importance(0.9),
        )
        .await
        .unwrap();

    let temp_dir = tempfile::tempdir().unwrap();
    let tool_executor = Arc::new(ToolExecutor::new(temp_dir.path().display().to_string()));
    let config = AgentConfig {
        memory: Some(Arc::new(memory)),
        ..Default::default()
    };

    let (tx, mut rx) = mpsc::channel(100);
    let agent = AgentLoop::new(
        mock_client,
        tool_executor,
        ToolContext::new(temp_dir.path().to_path_buf()),
        config,
    );
    let result = agent
        .execute(&[], "verify focused regression tests", Some(tx))
        .await
        .unwrap();

    assert_eq!(result.text, "Memory-aware response");

    let mut recalled = false;
    let mut resolved_items = None;
    while let Ok(event) = rx.try_recv() {
        match event {
            AgentEvent::MemoryRecalled { content, .. } => {
                recalled = content.contains("focused regression tests");
            }
            AgentEvent::ContextResolved { total_items, .. } => {
                resolved_items = Some(total_items);
            }
            _ => {}
        }
    }

    assert!(recalled);
    assert_eq!(resolved_items, Some(1));
}

#[tokio::test]
async fn test_agent_context_on_turn_complete() {
    let mock_client = Arc::new(MockLlmClient::new(vec![MockLlmClient::text_response(
        "Final response",
    )]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));

    let provider = Arc::new(MockContextProvider::new("memory-provider"));
    let on_turn_calls = provider.on_turn_calls.clone();

    let config = AgentConfig {
        context_providers: vec![provider],
        ..Default::default()
    };

    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);

    // Execute with session ID
    let result = agent
        .execute_with_session(&[], "verify user prompt", Some("sess-123"), None, None)
        .await
        .unwrap();

    assert_eq!(result.text, "Final response");

    // Check on_turn_complete was called
    let calls = on_turn_calls.read().await;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "sess-123");
    assert_eq!(calls[0].1, "verify user prompt");
    assert_eq!(calls[0].2, "Final response");
}

#[tokio::test]
async fn test_agent_context_on_turn_complete_no_session() {
    let mock_client = Arc::new(MockLlmClient::new(vec![MockLlmClient::text_response(
        "Response",
    )]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));

    let provider = Arc::new(MockContextProvider::new("memory-provider"));
    let on_turn_calls = provider.on_turn_calls.clone();

    let config = AgentConfig {
        context_providers: vec![provider],
        ..Default::default()
    };

    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);

    // Execute without session ID (uses execute() which passes None)
    let _result = agent.execute(&[], "Prompt", None).await.unwrap();

    // on_turn_complete should NOT be called when session_id is None
    let calls = on_turn_calls.read().await;
    assert!(calls.is_empty());
}

#[tokio::test]
async fn test_agent_build_augmented_system_prompt() {
    let mock_client = Arc::new(MockLlmClient::new(vec![MockLlmClient::text_response("OK")]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));

    let provider = MockContextProvider::new("test").with_items(vec![ContextItem::new(
        "doc-1",
        ContextType::Resource,
        "Auth uses JWT tokens.",
    )
    .with_source("viking://docs/auth")]);

    let config = AgentConfig {
        prompt_slots: SystemPromptSlots {
            extra: Some("You are helpful.".to_string()),
            ..Default::default()
        },
        context_providers: vec![Arc::new(provider)],
        ..Default::default()
    };

    let agent = AgentLoop::new(mock_client, tool_executor, test_tool_context(), config);

    // Test building augmented prompt
    let context_results = agent.resolve_context("test", None).await;
    let augmented = agent.build_augmented_system_prompt(&context_results);

    let augmented_str = augmented.unwrap();
    assert!(augmented_str.contains("You are helpful."));
    assert!(augmented_str.contains("<context source=\"viking://docs/auth\" type=\"Resource\">"));
    assert!(augmented_str.contains("Auth uses JWT tokens."));
}

// ========================================================================
// Agentic Loop Integration Tests
// ========================================================================

/// Helper: collect all events from a channel
async fn collect_events(mut rx: mpsc::Receiver<AgentEvent>) -> Vec<AgentEvent> {
    let mut events = Vec::new();
    while let Ok(event) = rx.try_recv() {
        events.push(event);
    }
    // Drain remaining
    while let Some(event) = rx.recv().await {
        events.push(event);
    }
    events
}

#[tokio::test]
async fn test_agent_multi_turn_tool_chain() {
    // LLM calls tool A → sees result → calls tool B → sees result → final answer
    let mock_client = Arc::new(MockLlmClient::new(vec![
        // Turn 1: call ls
        MockLlmClient::tool_call_response(
            "t1",
            "bash",
            serde_json::json!({"command": "echo step1"}),
        ),
        // Turn 2: call another tool based on first result
        MockLlmClient::tool_call_response(
            "t2",
            "bash",
            serde_json::json!({"command": "echo step2"}),
        ),
        // Turn 3: final answer
        MockLlmClient::text_response("Completed both steps: step1 then step2"),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let config = AgentConfig::default();

    let agent = AgentLoop::new(
        mock_client.clone(),
        tool_executor,
        test_tool_context(),
        config,
    );
    let result = agent.execute(&[], "Run two steps", None).await.unwrap();

    assert_eq!(result.text, "Completed both steps: step1 then step2");
    assert_eq!(result.tool_calls_count, 2);
    assert_eq!(mock_client.call_count.load(Ordering::SeqCst), 3);

    // Verify message history: user → assistant(tool_use) → user(tool_result) → assistant(tool_use) → user(tool_result) → assistant(text)
    assert_eq!(result.messages[0].role, "user");
    assert_eq!(result.messages[1].role, "assistant"); // tool call 1
    assert_eq!(result.messages[2].role, "user"); // tool result 1 (Anthropic convention)
    assert_eq!(result.messages[3].role, "assistant"); // tool call 2
    assert_eq!(result.messages[4].role, "user"); // tool result 2
    assert_eq!(result.messages[5].role, "assistant"); // final text
    assert_eq!(result.messages.len(), 6);
}

#[tokio::test]
async fn test_agent_conversation_history_preserved() {
    // Pass existing history, verify it's preserved in output
    let existing_history = vec![
        Message::user("What is Rust?"),
        Message {
            role: "assistant".to_string(),
            content: vec![ContentBlock::Text {
                text: "Rust is a systems programming language.".to_string(),
            }],
            reasoning_content: None,
        },
    ];

    let mock_client = Arc::new(MockLlmClient::new(vec![MockLlmClient::text_response(
        "Rust was created by Graydon Hoare at Mozilla.",
    )]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let agent = AgentLoop::new(
        mock_client.clone(),
        tool_executor,
        test_tool_context(),
        AgentConfig {
            prompt_slots: SystemPromptSlots {
                style: Some(AgentStyle::GeneralPurpose),
                ..Default::default()
            },
            ..Default::default()
        },
    );

    let result = agent
        .execute(&existing_history, "Who created it?", None)
        .await
        .unwrap();

    // History should contain: old user + old assistant + new user + new assistant
    assert_eq!(result.messages.len(), 4);
    assert_eq!(result.messages[0].text(), "What is Rust?");
    assert_eq!(
        result.messages[1].text(),
        "Rust is a systems programming language."
    );
    assert_eq!(result.messages[2].text(), "Who created it?");
    assert_eq!(
        result.messages[3].text(),
        "Rust was created by Graydon Hoare at Mozilla."
    );
}

#[tokio::test]
async fn test_agent_event_stream_completeness() {
    // Verify full event sequence for a single tool call loop
    let mock_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::tool_call_response("t1", "bash", serde_json::json!({"command": "echo hi"})),
        MockLlmClient::text_response("Done"),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let agent = AgentLoop::new(
        mock_client,
        tool_executor,
        test_tool_context(),
        AgentConfig {
            permission_checker: Some(Arc::new(PermissionPolicy::new().allow("bash(echo:*)"))),
            ..Default::default()
        },
    );

    let (tx, rx) = mpsc::channel(100);
    let result = agent.execute(&[], "Say hi", Some(tx)).await.unwrap();
    assert_eq!(result.text, "Done");

    let events = collect_events(rx).await;

    // Verify event sequence
    let event_types: Vec<&str> = events
        .iter()
        .map(|e| match e {
            AgentEvent::Start { .. } => "Start",
            AgentEvent::TurnStart { .. } => "TurnStart",
            AgentEvent::TurnEnd { .. } => "TurnEnd",
            AgentEvent::ToolEnd { .. } => "ToolEnd",
            AgentEvent::End { .. } => "End",
            _ => "Other",
        })
        .collect();

    // Mode/context events may precede Start; the execution lifecycle still
    // needs a Start before turns and an End as the final event.
    let start_index = event_types
        .iter()
        .position(|t| *t == "Start")
        .expect("Start event should be present");
    let first_turn_index = event_types
        .iter()
        .position(|t| *t == "TurnStart")
        .expect("TurnStart event should be present");
    assert!(start_index < first_turn_index);
    assert_eq!(event_types.last(), Some(&"End"));

    // Must have 2 TurnStarts (tool call turn + final answer turn)
    let turn_starts = event_types.iter().filter(|&&t| t == "TurnStart").count();
    assert_eq!(turn_starts, 2);

    // Must have 1 ToolEnd
    let tool_ends = event_types.iter().filter(|&&t| t == "ToolEnd").count();
    assert_eq!(tool_ends, 1);
}

#[tokio::test]
async fn test_agent_multiple_tools_single_turn() {
    // LLM returns 2 tool calls in one response
    let mock_client = Arc::new(MockLlmClient::new(vec![
        LlmResponse {
            message: Message {
                role: "assistant".to_string(),
                content: vec![
                    ContentBlock::ToolUse {
                        id: "t1".to_string(),
                        name: "bash".to_string(),
                        input: serde_json::json!({"command": "echo first"}),
                    },
                    ContentBlock::ToolUse {
                        id: "t2".to_string(),
                        name: "bash".to_string(),
                        input: serde_json::json!({"command": "echo second"}),
                    },
                ],
                reasoning_content: None,
            },
            usage: TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                cache_read_tokens: None,
                cache_write_tokens: None,
            },
            stop_reason: Some("tool_use".to_string()),
            meta: None,
        },
        MockLlmClient::text_response("Both commands ran"),
        MockLlmClient::text_response("Both commands ran"),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let agent = AgentLoop::new(
        mock_client.clone(),
        tool_executor,
        test_tool_context(),
        AgentConfig {
            prompt_slots: SystemPromptSlots {
                style: Some(AgentStyle::GeneralPurpose),
                ..Default::default()
            },
            ..Default::default()
        },
    );

    let result = agent
        .execute_loop(
            &[],
            "run both commands now",
            AgentStyle::GeneralPurpose,
            None,
            None,
            &tokio_util::sync::CancellationToken::new(),
            true,
        )
        .await
        .unwrap();

    assert_eq!(result.text, "Both commands ran");
    assert_eq!(result.tool_calls_count, 2);
    assert!(
        mock_client.call_count.load(Ordering::SeqCst) >= 2,
        "expected at least the tool-call turn and final response turn"
    );

    // Messages: user → assistant(2 tools) → user(tool_result) → user(tool_result) → assistant(text)
    assert_eq!(result.messages[0].role, "user");
    assert_eq!(result.messages[1].role, "assistant");
    assert_eq!(result.messages[2].role, "user"); // tool result 1
    assert_eq!(result.messages[3].role, "user"); // tool result 2
    assert_eq!(result.messages[4].role, "assistant");
}

#[tokio::test]
async fn test_agent_token_usage_accumulation() {
    // Verify usage sums across multiple turns
    let mock_client = Arc::new(MockLlmClient::new(vec![
        MockLlmClient::tool_call_response("t1", "bash", serde_json::json!({"command": "echo x"})),
        MockLlmClient::text_response("Done"),
    ]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let agent = AgentLoop::new(
        mock_client,
        tool_executor,
        test_tool_context(),
        AgentConfig::default(),
    );

    let result = agent.execute(&[], "test", None).await.unwrap();

    // Each mock response has prompt=10, completion=5, total=15
    // 2 LLM calls → 20 prompt, 10 completion, 30 total
    assert_eq!(result.usage.prompt_tokens, 20);
    assert_eq!(result.usage.completion_tokens, 10);
    assert_eq!(result.usage.total_tokens, 30);
}

#[tokio::test]
async fn test_agent_system_prompt_passed() {
    // Verify system prompt is used (MockLlmClient captures calls)
    let mock_client = Arc::new(MockLlmClient::new(vec![MockLlmClient::text_response(
        "I am a coding assistant.",
    )]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let config = AgentConfig {
        prompt_slots: SystemPromptSlots {
            extra: Some("You are a coding assistant.".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };

    let agent = AgentLoop::new(
        mock_client.clone(),
        tool_executor,
        test_tool_context(),
        config,
    );
    let result = agent.execute(&[], "What are you?", None).await.unwrap();

    assert_eq!(result.text, "I am a coding assistant.");
    assert_eq!(mock_client.call_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_agent_max_rounds_with_persistent_tool_calls() {
    // LLM keeps calling tools forever — should hit max_tool_rounds
    let mut responses = Vec::new();
    for i in 0..15 {
        responses.push(MockLlmClient::tool_call_response(
            &format!("t{}", i),
            "bash",
            serde_json::json!({"command": format!("echo round{}", i)}),
        ));
    }

    let mock_client = Arc::new(MockLlmClient::new(responses));
    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let config = AgentConfig {
        max_tool_rounds: 5,
        ..Default::default()
    };

    let agent = AgentLoop::new(
        mock_client.clone(),
        tool_executor,
        test_tool_context(),
        config,
    );
    let result = agent.execute(&[], "Loop forever", None).await;

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Max tool rounds (5) exceeded"));
}

#[tokio::test]
async fn test_agent_end_event_contains_final_text() {
    let mock_client = Arc::new(MockLlmClient::new(vec![MockLlmClient::text_response(
        "Final answer here",
    )]));

    let tool_executor = Arc::new(ToolExecutor::new("/tmp".to_string()));
    let agent = AgentLoop::new(
        mock_client,
        tool_executor,
        test_tool_context(),
        AgentConfig::default(),
    );

    let (tx, rx) = mpsc::channel(100);
    agent.execute(&[], "test", Some(tx)).await.unwrap();

    let events = collect_events(rx).await;
    let end_event = events.iter().find(|e| matches!(e, AgentEvent::End { .. }));
    assert!(end_event.is_some());

    if let AgentEvent::End { text, usage, .. } = end_event.unwrap() {
        assert_eq!(text, "Final answer here");
        assert_eq!(usage.total_tokens, 15);
    }
}

/// Regression: the parallel write fast path bypasses `ToolSafetyGate`, so it
/// may run only when the gate would unconditionally EXECUTE every call. Before
/// the fix it ignored the permission checker and skill restrictions entirely,
/// letting denied / ask / skill-restricted writes land ungated.
#[test]
fn parallel_write_batch_only_fast_paths_when_gate_would_execute() {
    use crate::llm::ToolCall;
    use crate::permissions::{PermissionChecker, PermissionDecision};
    use crate::skills::{Skill, SkillKind, SkillRegistry};

    struct Static(PermissionDecision);
    impl PermissionChecker for Static {
        fn check(&self, _tool: &str, _args: &serde_json::Value) -> PermissionDecision {
            self.0
        }
    }

    fn write_call(id: &str, path: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: "write_file".to_string(),
            args: serde_json::json!({ "path": path, "content": "x" }),
        }
    }

    fn loop_with(
        checker: Option<Arc<dyn PermissionChecker>>,
        skills: Option<Arc<SkillRegistry>>,
        enforce_active_skill_tool_restrictions: bool,
    ) -> AgentLoop {
        // `skill_registry` overrides the default builtins where needed.
        let config = AgentConfig {
            permission_checker: checker,
            skill_registry: skills,
            enforce_active_skill_tool_restrictions,
            ..Default::default()
        };
        AgentLoop::new(
            Arc::new(MockLlmClient::new(vec![])),
            Arc::new(ToolExecutor::new("/tmp".to_string())),
            test_tool_context(),
            config,
        )
    }

    let calls = vec![write_call("a", "a.txt"), write_call("b", "b.txt")];
    let allow = || Some(Arc::new(Static(PermissionDecision::Allow)) as Arc<dyn PermissionChecker>);

    // Explicit Allow + no restricting skills → fast path is taken.
    assert!(
        loop_with(allow(), None, false).can_run_parallel_write_batch(&calls),
        "explicit Allow with no restrictions → parallel write batch is allowed"
    );

    // No permission checker → gate resolves to Ask (a Deny without a confirmation
    // manager), so the fast path must be refused.
    assert!(
        !loop_with(None, None, false).can_run_parallel_write_batch(&calls),
        "missing checker resolves to Ask/Deny → fast path refused"
    );

    // Explicit Deny → refused.
    assert!(
        !loop_with(
            Some(Arc::new(Static(PermissionDecision::Deny))),
            None,
            false
        )
        .can_run_parallel_write_batch(&calls),
        "permission Deny → fast path refused"
    );

    // Ask → refused (sequential path would need a human round-trip).
    assert!(
        !loop_with(Some(Arc::new(Static(PermissionDecision::Ask))), None, false)
            .can_run_parallel_write_batch(&calls),
        "permission Ask → fast path refused"
    );

    // By default, active skill restrictions do not block ordinary session
    // tools before the permission/AHP/HITL chain.
    let restricted = SkillRegistry::new();
    restricted.register_unchecked(Arc::new(Skill {
        name: "read-only".to_string(),
        description: String::new(),
        allowed_tools: Some("read(*)".to_string()),
        disable_model_invocation: false,
        kind: SkillKind::Instruction,
        content: String::new(),
        tags: Vec::new(),
        version: None,
    }));
    let restricted = Arc::new(restricted);
    assert!(
        loop_with(allow(), Some(Arc::clone(&restricted)), false)
            .can_run_parallel_write_batch(&calls),
        "default active skill restriction mode → fast path follows permission allow"
    );

    // Compatibility mode preserves the legacy refusal.
    assert!(
        !loop_with(allow(), Some(restricted), true).can_run_parallel_write_batch(&calls),
        "legacy active skill restriction mode → fast path refused"
    );

    // Default builtins do not restrict → still fast-paths with Allow.
    assert!(
        loop_with(
            allow(),
            Some(Arc::new(SkillRegistry::with_builtins())),
            false
        )
        .can_run_parallel_write_batch(&calls),
        "non-restricting builtins → fast path still allowed"
    );
}
