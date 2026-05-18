//! AHP 2.4 Integration Tests
//!
//! Run with:
//! ```bash
//! cd crates/code
//! cargo test -p a3s-code-core --features ahp --test test_ahp_idle_with_llm -- --test-threads=1 --nocapture
//! ```

#![cfg(feature = "ahp")]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use a3s_ahp::transport::TransportLayer;
use a3s_code_core::ahp::{
    AhpHookExecutor, HeartbeatEvent, IdleDecision, IdleEvent, MemorySummary, SessionStats,
};
use a3s_code_core::hooks::{
    HookEvent, HookExecutor, HookResult, PostToolUseEvent, PreToolUseEvent, ToolResultData,
};
use async_trait::async_trait;

// ============================================================================
// Mock Transport for Testing
// ============================================================================

/// A mock transport that records sent requests and returns configurable responses
struct MockTransport {
    /// Queue of responses to return (FIFO)
    responses: Arc<Mutex<VecDeque<AhpResponse>>>,
    /// Record of all sent requests
    sent_requests: Arc<Mutex<Vec<AhpRequest>>>,
    /// Record of all sent notifications
    sent_notifications: Arc<Mutex<Vec<AhpNotification>>>,
}

impl MockTransport {
    fn new() -> Self {
        Self {
            responses: Arc::new(Mutex::new(VecDeque::new())),
            sent_requests: Arc::new(Mutex::new(Vec::new())),
            sent_notifications: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Add a response to be returned for the next request
    fn add_response(&self, response: AhpResponse) {
        self.responses.lock().unwrap().push_back(response);
    }

    /// Add an allow decision response
    fn add_allow_response(&self) {
        self.add_response(AhpResponse {
            jsonrpc: "2.0".to_string(),
            id: "1".to_string(),
            result: Some(serde_json::json!({
                "decision": "allow"
            })),
            error: None,
        });
    }

    /// Add a block decision response
    fn add_block_response(&self, reason: &str) {
        self.add_response(AhpResponse {
            jsonrpc: "2.0".to_string(),
            id: "1".to_string(),
            result: Some(serde_json::json!({
                "decision": "block",
                "reason": reason
            })),
            error: None,
        });
    }

    /// Add a defer decision response
    fn add_defer_response(&self, reason: &str, retry_after_ms: Option<u64>) {
        let mut result = serde_json::json!({
            "decision": "defer",
            "reason": reason
        });
        if let Some(ms) = retry_after_ms {
            result["retry_after_ms"] = serde_json::json!(ms);
        }
        self.add_response(AhpResponse {
            jsonrpc: "2.0".to_string(),
            id: "1".to_string(),
            result: Some(result),
            error: None,
        });
    }

    /// Get all sent requests
    fn get_sent_requests(&self) -> Vec<AhpRequest> {
        self.sent_requests.lock().unwrap().clone()
    }

    /// Get all sent notifications
    fn get_sent_notifications(&self) -> Vec<AhpNotification> {
        self.sent_notifications.lock().unwrap().clone()
    }
}

impl Clone for MockTransport {
    fn clone(&self) -> Self {
        Self {
            responses: Arc::clone(&self.responses),
            sent_requests: Arc::clone(&self.sent_requests),
            sent_notifications: Arc::clone(&self.sent_notifications),
        }
    }
}

#[async_trait]
impl TransportLayer for MockTransport {
    async fn send_request(&self, request: AhpRequest) -> a3s_ahp::Result<AhpResponse> {
        let request_id = request.id.clone();
        self.sent_requests.lock().unwrap().push(request);
        let mut response = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| a3s_ahp::AhpError::Transport("No response queued".to_string()))?;
        response.id = request_id;
        Ok(response)
    }

    async fn send_notification(&self, notification: AhpNotification) -> a3s_ahp::Result<()> {
        self.sent_notifications.lock().unwrap().push(notification);
        Ok(())
    }

    async fn close(&self) -> a3s_ahp::Result<()> {
        Ok(())
    }
}

// Types needed for the mock
use a3s_ahp::{AhpNotification, AhpRequest, AhpResponse};

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_idle_event_structure() {
    let idle_event = IdleEvent {
        idle_duration_ms: 10000,
        idle_reason: "no_activity".to_string(),
        last_event_type: Some("post_action".to_string()),
        suggested_action: Some("dream".to_string()),
    };

    assert_eq!(idle_event.idle_duration_ms, 10000);
    assert_eq!(idle_event.idle_reason, "no_activity");
    assert!(idle_event.suggested_action.is_some());
    assert_eq!(idle_event.suggested_action.as_deref(), Some("dream"));

    let json = serde_json::to_string(&idle_event).unwrap();
    assert!(json.contains("idle_duration_ms"));
    assert!(json.contains("no_activity"));
    assert!(json.contains("dream"));
}

#[test]
fn test_idle_decision_variants() {
    let allow = IdleDecision::Allow;
    let allow_json = serde_json::to_string(&allow).unwrap();
    assert!(allow_json.contains("allow"));
    assert!(!allow_json.contains("defer"));

    let defer = IdleDecision::Defer {
        reason: Some("busy".to_string()),
    };
    let defer_json = serde_json::to_string(&defer).unwrap();
    assert!(defer_json.contains("defer"));
    assert!(defer_json.contains("busy"));
}

#[test]
fn test_heartbeat_event_structure() {
    let heartbeat = HeartbeatEvent {
        uptime_ms: 60000,
        total_events_processed: 42,
        current_state: "active".to_string(),
        cpu_percent: None,
        memory_bytes: None,
        active_tools: None,
        pending_actions: None,
        queue_depth: None,
        tokens_used: None,
    };

    assert_eq!(heartbeat.uptime_ms, 60000);
    assert_eq!(heartbeat.total_events_processed, 42);
    assert_eq!(heartbeat.current_state, "active");

    let json = serde_json::to_string(&heartbeat).unwrap();
    assert!(json.contains("uptime_ms"));
    assert!(json.contains("total_events_processed"));
    assert!(json.contains("active"));
}

#[test]
fn test_idle_event_serialization_roundtrip() {
    let original = IdleEvent {
        idle_duration_ms: 5000,
        idle_reason: "waiting_for_input".to_string(),
        last_event_type: Some("pre_action".to_string()),
        suggested_action: Some("consolidate".to_string()),
    };

    let json = serde_json::to_string(&original).unwrap();
    let parsed: IdleEvent = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.idle_duration_ms, original.idle_duration_ms);
    assert_eq!(parsed.idle_reason, original.idle_reason);
    assert_eq!(parsed.last_event_type, original.last_event_type);
    assert_eq!(parsed.suggested_action, original.suggested_action);
}

#[test]
fn test_event_context_structure() {
    use a3s_ahp::EventContext;

    let context = EventContext {
        recent_facts: None,
        memory_summary: Some(MemorySummary {
            memory_type: "semantic".to_string(),
            total_items: 42,
            recent_topics: vec!["rust".to_string(), "async".to_string()],
        }),
        session_stats: Some(SessionStats {
            total_actions: 10,
            total_tokens: 5000,
            duration_ms: 60000,
            error_count: 0,
        }),
        current_task: Some("implementing idle detection".to_string()),
        capabilities: None,
    };

    let json = serde_json::to_string(&context).unwrap();
    assert!(json.contains("memory_summary"));
    assert!(json.contains("session_stats"));
    assert!(json.contains("rust"));
    assert!(json.contains("implementing idle detection"));
}

#[test]
fn test_mock_transport_basic() {
    let transport = MockTransport::new();

    // Add a response
    transport.add_allow_response();

    // Verify request is recorded
    let request = AhpRequest::new("test.method", serde_json::json!({}));
    let rt = tokio::runtime::Runtime::new().unwrap();
    let response = rt.block_on(transport.send_request(request.clone()));

    assert!(response.is_ok());
    let responses = transport.get_sent_requests();
    assert_eq!(responses.len(), 1);
    assert_eq!(responses[0].method, "test.method");
}

#[test]
fn test_mock_transport_notification() {
    let transport = MockTransport::new();

    let notification =
        AhpNotification::new("test.notification", serde_json::json!({"key": "value"}));

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(transport.send_notification(notification.clone()));

    assert!(result.is_ok());
    let notifications = transport.get_sent_notifications();
    assert_eq!(notifications.len(), 1);
    assert_eq!(notifications[0].method, "test.notification");
}

#[test]
fn test_mock_transport_empty_response_error() {
    let transport = MockTransport::new();

    let request = AhpRequest::new("test.method", serde_json::json!({}));

    let rt = tokio::runtime::Runtime::new().unwrap();
    let response = rt.block_on(transport.send_request(request));

    assert!(response.is_err());
}

/// Create an executor with mock transport for testing
/// Returns the executor and a clone of the mock transport (they share the same underlying state)
fn create_test_executor() -> (AhpHookExecutor, MockTransport) {
    use a3s_ahp::AhpClient;

    let mock_transport = MockTransport::new();
    // Clone the transport to get a separate Arc that shares the same inner data
    let transport_for_client: Arc<dyn TransportLayer> = Arc::new(mock_transport.clone());
    let client = AhpClient::new_for_testing(transport_for_client);
    let executor = AhpHookExecutor::new_for_testing(Arc::new(client), 10_000);

    (executor, mock_transport)
}

#[test]
fn test_executor_fire_pre_tool_use_allow() {
    let (executor, mock_transport) = create_test_executor();

    // Queue an allow response
    mock_transport.add_allow_response();

    let event = HookEvent::PreToolUse(PreToolUseEvent {
        session_id: "test-session".to_string(),
        tool: "bash".to_string(),
        args: serde_json::json!({"command": "ls"}),
        working_directory: "/test".to_string(),
        recent_tools: vec![],
    });

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(executor.fire(&event));

    // Should be allowed
    assert!(matches!(result, HookResult::Continue(None)));

    // Verify the request was sent
    let requests = mock_transport.get_sent_requests();
    assert_eq!(requests.len(), 1);
}

#[test]
fn test_executor_fire_pre_tool_use_block() {
    let (executor, mock_transport) = create_test_executor();

    // Queue a block response
    mock_transport.add_block_response("dangerous command");

    let event = HookEvent::PreToolUse(PreToolUseEvent {
        session_id: "test-session".to_string(),
        tool: "bash".to_string(),
        args: serde_json::json!({"command": "rm -rf /"}),
        working_directory: "/test".to_string(),
        recent_tools: vec![],
    });

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(executor.fire(&event));

    // Should be blocked
    assert!(matches!(result, HookResult::Block(reason) if reason.contains("dangerous")));
}

#[test]
fn test_executor_fire_pre_tool_use_defer() {
    let (executor, mock_transport) = create_test_executor();

    // Queue a defer response
    mock_transport.add_defer_response("rate limited", Some(5000));

    let event = HookEvent::PreToolUse(PreToolUseEvent {
        session_id: "test-session".to_string(),
        tool: "bash".to_string(),
        args: serde_json::json!({"command": "ls"}),
        working_directory: "/test".to_string(),
        recent_tools: vec![],
    });

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(executor.fire(&event));

    // Should be retried after 5000ms
    assert!(matches!(result, HookResult::Retry(ms) if ms == 5000));
}

#[test]
fn test_executor_non_blocking_event_fire_and_forget() {
    let (executor, mock_transport) = create_test_executor();

    // Queue responses (though they shouldn't be used for non-blocking)
    mock_transport.add_allow_response();

    let event = HookEvent::PostToolUse(PostToolUseEvent {
        session_id: "test-session".to_string(),
        tool: "bash".to_string(),
        args: serde_json::json!({"command": "ls"}),
        result: ToolResultData {
            success: true,
            output: "files".to_string(),
            exit_code: Some(0),
            duration_ms: 100,
        },
    });

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(executor.fire(&event));

    // Non-blocking events return Continue immediately
    assert!(matches!(result, HookResult::Continue(None)));

    // Give the async task a moment to complete
    std::thread::sleep(std::time::Duration::from_millis(50));

    // Should have sent a notification
    let notifications = mock_transport.get_sent_notifications();
    assert_eq!(notifications.len(), 1);
}

#[test]
fn test_executor_idle_threshold() {
    let (executor, _mock_transport) = create_test_executor();

    // Default threshold is 10 seconds
    assert_eq!(executor.idle_threshold(), 10_000);

    // Idle duration should be 0 immediately after creation
    let idle_duration = executor.get_idle_duration_ms();
    assert!(idle_duration < 100); // Should be very small
}

#[test]
fn test_executor_check_idle_below_threshold() {
    let (executor, _mock_transport) = create_test_executor();

    // With 10 second threshold and very recent creation, should not be idle
    let idle_event = executor.check_idle();
    assert!(idle_event.is_none());
}

#[test]
fn test_executor_record_error() {
    let (executor, _mock_transport) = create_test_executor();

    let initial_errors = executor.error_count_value();
    executor.record_error();
    assert_eq!(executor.error_count_value(), initial_errors + 1);
}

#[test]
fn test_executor_total_events() {
    let (executor, mock_transport) = create_test_executor();

    // Queue responses
    mock_transport.add_allow_response();

    let event = HookEvent::PreToolUse(PreToolUseEvent {
        session_id: "test-session".to_string(),
        tool: "bash".to_string(),
        args: serde_json::json!({"command": "ls"}),
        working_directory: "/test".to_string(),
        recent_tools: vec![],
    });

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(executor.fire(&event));

    assert_eq!(executor.total_events_count(), 1);
}

#[test]
fn test_executor_set_memory_summary() {
    use std::sync::Arc;

    let (executor, _mock_transport) = create_test_executor();
    let executor = Arc::new(executor);

    let summary = MemorySummary {
        memory_type: "three_tier".to_string(),
        total_items: 100,
        recent_topics: vec!["rust".to_string(), "testing".to_string()],
    };

    executor.set_memory_summary(summary);

    // Check idle event contains memory summary
    // Note: check_idle creates the event, we verify by checking it doesn't panic
    // The actual context population happens in build_context()
}

#[test]
fn test_executor_set_current_task() {
    use std::sync::Arc;

    let (executor, _mock_transport) = create_test_executor();
    let executor_ref = Arc::new(executor);

    executor_ref
        .clone()
        .set_current_task("implementing tests".to_string());

    // Verify the executor can create an idle event without panic
    let _idle_event = executor_ref.check_idle();
}

// ============================================================================
// Configuration Tests (using environment variables)
// ============================================================================

fn get_test_config() -> (String, String, String) {
    let api_key =
        std::env::var("MINIMAX_API_KEY").expect("MINIMAX_API_KEY environment variable not set");
    let base_url = std::env::var("MINIMAX_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1/".to_string());
    let model =
        std::env::var("MINIMAX_MODEL").unwrap_or_else(|_| "MiniMax-M2.7-highspeed".to_string());
    (api_key, base_url, model)
}

#[test]
#[ignore]
fn test_minmax_llm_basic_completion() {
    use a3s_code_core::llm::{LlmClient, Message, OpenAiClient};

    let (api_key, base_url, model) = get_test_config();

    let client = OpenAiClient::new(api_key, model).with_base_url(base_url);

    let messages = vec![Message::user(
        "Reply with exactly the word 'HELLO' in uppercase, nothing else.",
    )];

    let rt = tokio::runtime::Runtime::new().unwrap();
    let response = rt.block_on(client.complete(&messages, None, &[]));

    assert!(response.is_ok(), "LLM call failed: {:?}", response.err());
    let text = response.unwrap().text().trim().to_uppercase();
    assert_eq!(text, "HELLO", "Expected 'HELLO', got: {}", text);
}

#[test]
#[ignore]
fn test_minmax_llm_with_system_prompt() {
    use a3s_code_core::llm::{LlmClient, Message, OpenAiClient};

    let (api_key, base_url, model) = get_test_config();

    let client = OpenAiClient::new(api_key, model).with_base_url(base_url);

    let system = "You are a security analyzer. When given a command, respond with only 'SAFE' or 'DANGEROUS'.";

    let messages = vec![Message::user("ls -la")];

    let rt = tokio::runtime::Runtime::new().unwrap();
    let response = rt.block_on(client.complete(&messages, Some(system), &[]));

    assert!(response.is_ok(), "LLM call failed: {:?}", response.err());
    let text = response.unwrap().text().trim().to_uppercase();
    assert!(
        text == "SAFE" || text == "DANGEROUS",
        "Expected SAFE or DANGEROUS, got: {}",
        text
    );
}
