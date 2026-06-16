use super::*;
use crate::hitl::ConfirmationPolicy;
use crate::llm::{Message, TokenUsage};
use crate::permissions::PermissionPolicy;
use crate::prompts::PlanningMode;
use crate::queue::SessionQueueConfig;
use crate::run::RunRecord;
use crate::tools::ArtifactStore;
use crate::trace::TraceEvent;
use crate::verification::VerificationReport;
use tempfile::tempdir;

fn create_test_session_data() -> SessionData {
    SessionData {
        id: "test-session-1".to_string(),
        config: SessionConfig {
            name: "Test Session".to_string(),
            workspace: "/tmp/workspace".to_string(),
            system_prompt: Some("You are helpful.".to_string()),
            max_context_length: 200000,
            auto_compact: false,
            auto_compact_threshold: DEFAULT_AUTO_COMPACT_THRESHOLD,
            storage_type: crate::config::StorageBackend::File,
            queue_config: None,
            confirmation_policy: None,
            permission_policy: None,
            enforce_active_skill_tool_restrictions: false,
            max_parallel_tasks: None,
            auto_delegation: None,
            parent_id: None,
            security_config: None,
            hook_engine: None,
            planning_mode: PlanningMode::default(),
            goal_tracking: false,
        },
        state: SessionState::Active,
        messages: vec![
            Message::user("Hello"),
            Message {
                role: "assistant".to_string(),
                content: vec![crate::llm::ContentBlock::Text {
                    text: "Hi there!".to_string(),
                }],
                reasoning_content: None,
            },
        ],
        context_usage: ContextUsage {
            used_tokens: 100,
            max_tokens: 200000,
            percent: 0.0005,
            turns: 2,
        },
        total_usage: TokenUsage {
            prompt_tokens: 50,
            completion_tokens: 50,
            total_tokens: 100,
            cache_read_tokens: None,
            cache_write_tokens: None,
        },
        tool_names: vec!["bash".to_string(), "read".to_string()],
        thinking_enabled: false,
        thinking_budget: None,
        created_at: 1700000000,
        updated_at: 1700000100,
        llm_config: None,
        tasks: vec![],
        parent_id: None,
        tenant_id: None,
        principal: None,
        agent_template_id: None,
        correlation_id: None,
        total_cost: 0.0,
        model_name: None,
        cost_records: Vec::new(),
    }
}

fn create_test_verification_report() -> VerificationReport {
    VerificationReport::new(
        "program:test",
        vec![
            crate::verification::VerificationCheck::required("check:test", "test", "Run tests")
                .with_status(crate::verification::VerificationStatus::Passed),
        ],
    )
}

async fn create_test_run_records() -> Vec<RunRecord> {
    let runs = crate::run::InMemoryRunStore::new();
    let run = runs.create_run("session/a", "persist run").await;
    runs.record_event(
        &run.id,
        crate::agent::AgentEvent::Start {
            prompt: "persist run".to_string(),
        },
    )
    .await;
    runs.records().await
}

// ========================================================================
// FileSessionStore Tests
// ========================================================================

#[tokio::test]
async fn test_file_store_save_and_load() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();

    let session = create_test_session_data();

    // Save
    store.save(&session).await.unwrap();

    // Load
    let loaded = store.load(&session.id).await.unwrap();
    assert!(loaded.is_some());

    let loaded = loaded.unwrap();
    assert_eq!(loaded.id, session.id);
    assert_eq!(loaded.config.name, session.config.name);
    assert_eq!(loaded.messages.len(), 2);
    assert_eq!(loaded.state, SessionState::Active);
}

#[tokio::test]
async fn test_file_store_load_nonexistent() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();

    let loaded = store.load("nonexistent").await.unwrap();
    assert!(loaded.is_none());
}

#[tokio::test]
async fn test_file_store_delete() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();

    let session = create_test_session_data();
    store.save(&session).await.unwrap();

    // Verify exists
    assert!(store.exists(&session.id).await.unwrap());

    // Delete
    store.delete(&session.id).await.unwrap();

    // Verify gone
    assert!(!store.exists(&session.id).await.unwrap());
    assert!(store.load(&session.id).await.unwrap().is_none());
}

#[tokio::test]
async fn test_file_store_save_and_load_artifacts() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();
    let artifacts = ArtifactStore::new();
    artifacts.put(crate::tools::ToolArtifact {
        artifact_id: "tool-output:test:a".to_string(),
        artifact_uri: "a3s://tool-output/test/a".to_string(),
        tool_name: "test".to_string(),
        content: "artifact content".to_string(),
        original_bytes: 16,
        shown_bytes: 4,
    });

    store.save_artifacts("session/a", &artifacts).await.unwrap();
    let loaded = store
        .load_artifacts("session/a")
        .await
        .unwrap()
        .expect("artifacts");

    assert_eq!(loaded.len(), 1);
    assert_eq!(
        loaded
            .get("a3s://tool-output/test/a")
            .expect("artifact")
            .content,
        "artifact content"
    );
}

#[tokio::test]
async fn test_file_store_save_and_load_trace_events() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();
    let event = TraceEvent::tool_execution(
        "read",
        true,
        0,
        std::time::Duration::from_millis(9),
        12,
        Some(&serde_json::json!({
            "artifact": {
                "artifact_uri": "a3s://tool-output/read/abc"
            }
        })),
    );

    store
        .save_trace_events("session/a", std::slice::from_ref(&event))
        .await
        .unwrap();
    let loaded = store
        .load_trace_events("session/a")
        .await
        .unwrap()
        .expect("trace events");

    assert_eq!(loaded, vec![event]);
}

#[tokio::test]
async fn test_file_store_save_and_load_run_records() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();
    let records = create_test_run_records().await;

    store.save_run_records("session/a", &records).await.unwrap();
    let loaded = store
        .load_run_records("session/a")
        .await
        .unwrap()
        .expect("run records");

    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].snapshot.prompt, "persist run");
    assert_eq!(loaded[0].events.len(), 1);
}

#[tokio::test]
async fn test_file_store_save_and_load_verification_reports() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();
    let report = create_test_verification_report();

    store
        .save_verification_reports("session/a", std::slice::from_ref(&report))
        .await
        .unwrap();
    let loaded = store
        .load_verification_reports("session/a")
        .await
        .unwrap()
        .expect("verification reports");

    assert_eq!(loaded, vec![report]);
}

#[tokio::test]
async fn test_memory_store_save_load_and_delete_artifacts() {
    let store = MemorySessionStore::new();
    let session = create_test_session_data();
    store.save(&session).await.unwrap();
    let artifacts = ArtifactStore::new();
    artifacts.put(crate::tools::ToolArtifact {
        artifact_id: "tool-output:test:a".to_string(),
        artifact_uri: "a3s://tool-output/test/a".to_string(),
        tool_name: "test".to_string(),
        content: "artifact content".to_string(),
        original_bytes: 16,
        shown_bytes: 4,
    });

    store.save_artifacts(&session.id, &artifacts).await.unwrap();
    assert!(store
        .load_artifacts(&session.id)
        .await
        .unwrap()
        .expect("artifacts")
        .get("a3s://tool-output/test/a")
        .is_some());

    store.delete(&session.id).await.unwrap();
    assert!(store.load_artifacts(&session.id).await.unwrap().is_none());
}

#[tokio::test]
async fn test_memory_store_save_load_and_delete_trace_events() {
    let store = MemorySessionStore::new();
    let session = create_test_session_data();
    let event = TraceEvent::tool_execution(
        "grep",
        false,
        1,
        std::time::Duration::from_millis(2),
        24,
        None,
    );

    store.save(&session).await.unwrap();
    store
        .save_trace_events(&session.id, std::slice::from_ref(&event))
        .await
        .unwrap();
    let loaded = store
        .load_trace_events(&session.id)
        .await
        .unwrap()
        .expect("trace events");
    assert_eq!(loaded, vec![event]);

    store.delete(&session.id).await.unwrap();
    assert!(store
        .load_trace_events(&session.id)
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn test_memory_store_save_load_and_delete_run_records() {
    let store = MemorySessionStore::new();
    let session = create_test_session_data();
    let records = create_test_run_records().await;

    store.save(&session).await.unwrap();
    store.save_run_records(&session.id, &records).await.unwrap();
    let loaded = store
        .load_run_records(&session.id)
        .await
        .unwrap()
        .expect("run records");
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].events.len(), 1);

    store.delete(&session.id).await.unwrap();
    assert!(store.load_run_records(&session.id).await.unwrap().is_none());
}

#[tokio::test]
async fn test_memory_store_save_load_and_delete_verification_reports() {
    let store = MemorySessionStore::new();
    let session = create_test_session_data();
    let report = create_test_verification_report();

    store.save(&session).await.unwrap();
    store
        .save_verification_reports(&session.id, std::slice::from_ref(&report))
        .await
        .unwrap();
    let loaded = store
        .load_verification_reports(&session.id)
        .await
        .unwrap()
        .expect("verification reports");
    assert_eq!(loaded, vec![report]);

    store.delete(&session.id).await.unwrap();
    assert!(store
        .load_verification_reports(&session.id)
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn test_file_store_list() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();

    // Initially empty
    let list = store.list().await.unwrap();
    assert!(list.is_empty());

    // Add sessions
    for i in 1..=3 {
        let mut session = create_test_session_data();
        session.id = format!("session-{}", i);
        store.save(&session).await.unwrap();
    }

    // List should have 3 sessions
    let list = store.list().await.unwrap();
    assert_eq!(list.len(), 3);
    assert!(list.contains(&"session-1".to_string()));
    assert!(list.contains(&"session-2".to_string()));
    assert!(list.contains(&"session-3".to_string()));
}

#[tokio::test]
async fn test_file_store_overwrite() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();

    let mut session = create_test_session_data();
    store.save(&session).await.unwrap();

    // Modify and save again
    session.messages.push(Message::user("Another message"));
    session.updated_at = 1700000200;
    store.save(&session).await.unwrap();

    // Load and verify
    let loaded = store.load(&session.id).await.unwrap().unwrap();
    assert_eq!(loaded.messages.len(), 3);
    assert_eq!(loaded.updated_at, 1700000200);
}

#[tokio::test]
async fn test_file_store_path_traversal_prevention() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();

    // Attempt path traversal - should be sanitized
    let mut session = create_test_session_data();
    session.id = "../../../etc/passwd".to_string();
    store.save(&session).await.unwrap();

    // File should be in the store directory, not /etc/passwd
    let files: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(files.len(), 1);

    // Should still be loadable with sanitized ID
    let loaded = store.load(&session.id).await.unwrap();
    assert!(loaded.is_some());
}

#[tokio::test]
async fn test_file_store_with_policies() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();

    let mut session = create_test_session_data();
    session.config.confirmation_policy = Some(ConfirmationPolicy::enabled());
    session.config.permission_policy = Some(PermissionPolicy::new().allow("Bash(cargo:*)"));
    session.config.queue_config = Some(SessionQueueConfig::default());
    session.config.enforce_active_skill_tool_restrictions = true;

    store.save(&session).await.unwrap();

    let loaded = store.load(&session.id).await.unwrap().unwrap();
    assert!(loaded.config.confirmation_policy.is_some());
    assert!(loaded.config.permission_policy.is_some());
    assert!(loaded.config.queue_config.is_some());
    assert!(loaded.config.enforce_active_skill_tool_restrictions);
}

#[tokio::test]
async fn test_file_store_with_llm_config() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();

    let mut session = create_test_session_data();
    session.llm_config = Some(LlmConfigData {
        provider: "anthropic".to_string(),
        model: "claude-3-5-sonnet-20241022".to_string(),
        api_key: Some("secret".to_string()), // Should NOT be saved
        base_url: None,
    });

    store.save(&session).await.unwrap();

    let loaded = store.load(&session.id).await.unwrap().unwrap();
    let llm_config = loaded.llm_config.unwrap();
    assert_eq!(llm_config.provider, "anthropic");
    assert_eq!(llm_config.model, "claude-3-5-sonnet-20241022");
    // API key should not be persisted
    assert!(llm_config.api_key.is_none());
}

// ========================================================================
// MemorySessionStore Tests
// ========================================================================

#[tokio::test]
async fn test_memory_store_save_and_load() {
    let store = MemorySessionStore::new();
    let session = create_test_session_data();

    store.save(&session).await.unwrap();

    let loaded = store.load(&session.id).await.unwrap();
    assert!(loaded.is_some());
    assert_eq!(loaded.unwrap().id, session.id);
}

#[tokio::test]
async fn test_memory_store_delete() {
    let store = MemorySessionStore::new();
    let session = create_test_session_data();

    store.save(&session).await.unwrap();
    assert!(store.exists(&session.id).await.unwrap());

    store.delete(&session.id).await.unwrap();
    assert!(!store.exists(&session.id).await.unwrap());
}

#[tokio::test]
async fn test_memory_store_list() {
    let store = MemorySessionStore::new();

    for i in 1..=3 {
        let mut session = create_test_session_data();
        session.id = format!("session-{}", i);
        store.save(&session).await.unwrap();
    }

    let list = store.list().await.unwrap();
    assert_eq!(list.len(), 3);
}

// ========================================================================
// SessionData Tests
// ========================================================================

#[test]
fn test_session_data_serialization() {
    let session = create_test_session_data();
    let json = serde_json::to_string(&session).unwrap();
    let parsed: SessionData = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.id, session.id);
    assert_eq!(parsed.messages.len(), session.messages.len());
}

#[test]
fn test_tool_names_from_definitions() {
    let tools = vec![
        crate::llm::ToolDefinition {
            name: "bash".to_string(),
            description: "Execute bash".to_string(),
            parameters: serde_json::json!({}),
        },
        crate::llm::ToolDefinition {
            name: "read".to_string(),
            description: "Read file".to_string(),
            parameters: serde_json::json!({}),
        },
    ];

    let names = SessionData::tool_names_from_definitions(&tools);
    assert_eq!(names, vec!["bash", "read"]);
}

// ========================================================================
// Sanitization Tests
// ========================================================================

#[tokio::test]
async fn test_file_store_backslash_sanitization() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();

    let mut session = create_test_session_data();
    session.id = r"foo\bar\baz".to_string();
    store.save(&session).await.unwrap();

    let loaded = store.load(&session.id).await.unwrap();
    assert!(loaded.is_some());

    let loaded = loaded.unwrap();
    assert_eq!(loaded.id, session.id);

    // Verify the file on disk uses sanitized name
    let expected_path = dir.path().join("foo_bar_baz.json");
    assert!(expected_path.exists());
}

#[tokio::test]
async fn test_file_store_mixed_separator_sanitization() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();

    let mut session = create_test_session_data();
    session.id = r"foo/bar\baz..qux".to_string();
    store.save(&session).await.unwrap();

    let loaded = store.load(&session.id).await.unwrap();
    assert!(loaded.is_some());

    let loaded = loaded.unwrap();
    assert_eq!(loaded.id, session.id);

    // / -> _, \ -> _, .. -> _
    let expected_path = dir.path().join("foo_bar_baz_qux.json");
    assert!(expected_path.exists());
}

// ========================================================================
// Error Recovery Tests
// ========================================================================

#[tokio::test]
async fn test_file_store_corrupted_json_recovery() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();

    // Manually write invalid JSON to a session file
    let corrupted_path = dir.path().join("test-id.json");
    tokio::fs::write(&corrupted_path, b"not valid json {{{")
        .await
        .unwrap();

    // Loading should return an error, not panic
    let result = store.load("test-id").await;
    assert!(result.is_err());
}

// ========================================================================
// Exists Tests
// ========================================================================

#[tokio::test]
async fn test_file_store_exists() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();

    let session = create_test_session_data();

    // Not yet saved
    assert!(!store.exists(&session.id).await.unwrap());

    // Save and verify exists
    store.save(&session).await.unwrap();
    assert!(store.exists(&session.id).await.unwrap());

    // Delete and verify gone
    store.delete(&session.id).await.unwrap();
    assert!(!store.exists(&session.id).await.unwrap());
}

#[tokio::test]
async fn test_memory_store_exists() {
    let store = MemorySessionStore::new();

    // Unknown id
    assert!(!store.exists("unknown-id").await.unwrap());

    // Save and verify exists
    let session = create_test_session_data();
    store.save(&session).await.unwrap();
    assert!(store.exists(&session.id).await.unwrap());
}

#[tokio::test]
async fn test_file_store_health_check() {
    let dir = tempfile::tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();
    assert!(store.health_check().await.is_ok());
    assert_eq!(store.backend_name(), "file");
}

#[tokio::test]
async fn test_file_store_health_check_bad_dir() {
    let store = FileSessionStore {
        dir: std::path::PathBuf::from("/nonexistent/path/that/does/not/exist"),
    };
    assert!(store.health_check().await.is_err());
}

#[tokio::test]
async fn test_memory_store_health_check() {
    let store = MemorySessionStore::new();
    assert!(store.health_check().await.is_ok());
    assert_eq!(store.backend_name(), "memory");
}

// ========================================================================
// Session Resume Boundary Tests
// ========================================================================

#[tokio::test]
async fn test_file_store_load_empty_file() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();

    // Write an empty file — JSON parse must fail gracefully, not panic
    let empty_path = dir.path().join("empty-session.json");
    tokio::fs::write(&empty_path, b"").await.unwrap();

    let result = store.load("empty-session").await;
    assert!(
        result.is_err(),
        "Empty file must return error, not Ok(None)"
    );
}

#[tokio::test]
async fn test_file_store_load_partial_json() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();

    // Truncated JSON — simulates a crash mid-write
    let partial_path = dir.path().join("partial-session.json");
    tokio::fs::write(&partial_path, b"{\"id\":\"partial-session\",\"message")
        .await
        .unwrap();

    let result = store.load("partial-session").await;
    assert!(result.is_err(), "Partial JSON must return error");
}

#[tokio::test]
async fn test_file_store_concurrent_save() {
    let dir = tempdir().unwrap();
    let store = std::sync::Arc::new(FileSessionStore::new(dir.path()).await.unwrap());

    let session = create_test_session_data();
    let id = session.id.clone();

    // First save to create the file
    store.save(&session).await.unwrap();

    // Spawn multiple concurrent saves — last write wins, no corruption
    let mut handles = Vec::new();
    for _ in 0..5 {
        let s = store.clone();
        let sess = session.clone();
        handles.push(tokio::spawn(async move { s.save(&sess).await }));
    }
    for h in handles {
        h.await.unwrap().unwrap();
    }

    // File must be loadable after concurrent writes
    let loaded = store.load(&id).await.unwrap();
    assert!(loaded.is_some());
    assert_eq!(loaded.unwrap().id, id);
}

#[tokio::test]
async fn test_file_store_load_nonexistent_returns_none() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();

    let result = store.load("does-not-exist-at-all").await.unwrap();
    assert!(result.is_none(), "Missing session must return Ok(None)");
}

fn sample_checkpoint(run_id: &str) -> crate::loop_checkpoint::LoopCheckpoint {
    crate::loop_checkpoint::LoopCheckpoint {
        schema_version: crate::loop_checkpoint::LOOP_CHECKPOINT_SCHEMA_VERSION,
        run_id: run_id.to_string(),
        session_id: "s-1".to_string(),
        turn: 2,
        messages: vec![Message::user("hi")],
        total_usage: TokenUsage::default(),
        tool_calls_count: 1,
        verification_reports: Vec::new(),
        checkpoint_ms: 1_700_000_000_000,
    }
}

#[tokio::test]
async fn test_memory_store_delete_loop_checkpoint() {
    let store = MemorySessionStore::new();
    store
        .save_loop_checkpoint("run-x", &sample_checkpoint("run-x"))
        .await
        .unwrap();
    assert!(store.load_loop_checkpoint("run-x").await.unwrap().is_some());

    store.delete_loop_checkpoint("run-x").await.unwrap();
    assert!(
        store.load_loop_checkpoint("run-x").await.unwrap().is_none(),
        "checkpoint must be gone after delete"
    );

    // Deleting a non-existent checkpoint is a no-op success.
    store.delete_loop_checkpoint("never-existed").await.unwrap();
}

#[tokio::test]
async fn test_file_store_delete_loop_checkpoint() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();
    store
        .save_loop_checkpoint("run-y", &sample_checkpoint("run-y"))
        .await
        .unwrap();
    let loaded = store.load_loop_checkpoint("run-y").await.unwrap();
    assert_eq!(loaded.unwrap().run_id, "run-y");

    store.delete_loop_checkpoint("run-y").await.unwrap();
    assert!(store.load_loop_checkpoint("run-y").await.unwrap().is_none());

    // Idempotent on a missing file.
    store.delete_loop_checkpoint("run-y").await.unwrap();
}

#[tokio::test]
async fn test_file_store_checkpoint_write_is_atomic_no_temp_leftovers() {
    // The crash-atomic write uses a temp file + rename. After a normal
    // save, no `.tmp` files should be left behind in the checkpoint dir.
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();
    store
        .save_loop_checkpoint("run-z", &sample_checkpoint("run-z"))
        .await
        .unwrap();

    let ckpt_dir = dir.path().join("loop_checkpoints");
    let mut entries = tokio::fs::read_dir(&ckpt_dir).await.unwrap();
    let mut names = Vec::new();
    while let Some(e) = entries.next_entry().await.unwrap() {
        names.push(e.file_name().to_string_lossy().to_string());
    }
    assert!(
        names.iter().all(|n| !n.contains(".tmp")),
        "no temp files should remain after atomic write, got: {names:?}"
    );
    assert!(
        names.iter().any(|n| n == "run-z.json"),
        "the final checkpoint file must exist, got: {names:?}"
    );
}

fn sample_workflow_checkpoint(wf_id: &str) -> crate::orchestration::WorkflowCheckpoint {
    use crate::orchestration::{
        StepOutcome, WorkflowCheckpoint, WorkflowStepRecord, WORKFLOW_CHECKPOINT_SCHEMA_VERSION,
    };
    let outcome = |id: &str, structured| StepOutcome {
        task_id: id.to_string(),
        session_id: format!("task-run-{id}"),
        agent: "explore".to_string(),
        output: format!("out-{id}"),
        success: true,
        structured,
    };
    WorkflowCheckpoint {
        schema_version: WORKFLOW_CHECKPOINT_SCHEMA_VERSION,
        workflow_id: wf_id.to_string(),
        steps: vec![
            WorkflowStepRecord {
                task_id: "a".into(),
                outcome: outcome("a", None),
            },
            WorkflowStepRecord {
                task_id: "b".into(),
                outcome: outcome("b", Some(serde_json::json!({ "k": 1 }))),
            },
        ],
        checkpoint_ms: 1_700_000_000_000,
    }
}

#[tokio::test]
async fn test_file_store_workflow_checkpoint_roundtrip() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();
    let cp = sample_workflow_checkpoint("wf-1");
    store.save_workflow_checkpoint("wf-1", &cp).await.unwrap();

    let loaded = store
        .load_workflow_checkpoint("wf-1")
        .await
        .unwrap()
        .expect("present");
    assert_eq!(loaded, cp);
    assert_eq!(loaded.completed().len(), 2);

    store.delete_workflow_checkpoint("wf-1").await.unwrap();
    assert!(store
        .load_workflow_checkpoint("wf-1")
        .await
        .unwrap()
        .is_none());
    // Idempotent on a missing file.
    store.delete_workflow_checkpoint("wf-1").await.unwrap();
}

#[tokio::test]
async fn test_memory_store_rejects_future_workflow_checkpoint() {
    let store = MemorySessionStore::new();
    let mut future = sample_workflow_checkpoint("wf-future");
    future.schema_version = crate::orchestration::WORKFLOW_CHECKPOINT_SCHEMA_VERSION + 1;
    store
        .save_workflow_checkpoint("wf-future", &future)
        .await
        .unwrap();
    let err = store
        .load_workflow_checkpoint("wf-future")
        .await
        .unwrap_err();
    assert!(err.to_string().contains("schema version"), "got: {err}");

    store
        .save_workflow_checkpoint("wf-ok", &sample_workflow_checkpoint("wf-ok"))
        .await
        .unwrap();
    assert!(store
        .load_workflow_checkpoint("wf-ok")
        .await
        .unwrap()
        .is_some());
}

#[tokio::test]
async fn test_file_store_rejects_future_workflow_checkpoint() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();
    let mut future = sample_workflow_checkpoint("wf-f");
    future.schema_version = crate::orchestration::WORKFLOW_CHECKPOINT_SCHEMA_VERSION + 1;
    store
        .save_workflow_checkpoint("wf-f", &future)
        .await
        .unwrap();
    let err = store.load_workflow_checkpoint("wf-f").await.unwrap_err();
    assert!(err.to_string().contains("schema version"), "got: {err}");
}

#[tokio::test]
async fn test_file_store_workflow_checkpoint_atomic_no_temp_leftovers() {
    let dir = tempdir().unwrap();
    let store = FileSessionStore::new(dir.path()).await.unwrap();
    store
        .save_workflow_checkpoint("wf-z", &sample_workflow_checkpoint("wf-z"))
        .await
        .unwrap();

    let ckpt_dir = dir.path().join("workflow_checkpoints");
    let mut entries = tokio::fs::read_dir(&ckpt_dir).await.unwrap();
    let mut names = Vec::new();
    while let Some(e) = entries.next_entry().await.unwrap() {
        names.push(e.file_name().to_string_lossy().to_string());
    }
    assert!(
        names.iter().all(|n| !n.contains(".tmp")),
        "no temp leftovers after atomic write, got: {names:?}"
    );
    assert!(
        names.iter().any(|n| n == "wf-z.json"),
        "the final workflow checkpoint file must exist, got: {names:?}"
    );
}
