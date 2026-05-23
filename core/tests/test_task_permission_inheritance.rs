//! Integration test for task/parallel_task permission inheritance (Issue #28).
//!
//! Verifies that delegated child runs launched through the task tool inherit
//! the agent definition's permission policy into child AgentConfig, so tools
//! are not unexpectedly safe-denied with MissingConfirmationManager.
//!
//! Run with:
//!   cargo test --test test_task_permission_inheritance -- --nocapture
//!
//! For the full end-to-end LLM test (requires network):
//!   cargo test --test test_task_permission_inheritance -- --ignored --nocapture

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use a3s_code_core::config::CodeConfig;
use a3s_code_core::llm::create_client_with_config;
use a3s_code_core::permissions::{PermissionDecision, PermissionPolicy};
use a3s_code_core::subagent::{AgentRegistry, WorkerAgentSpec};
use a3s_code_core::tools::{TaskExecutor, TaskParams};

fn repo_config_path() -> PathBuf {
    std::env::var_os("A3S_CONFIG_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../..")
                .join(".a3s/config.acl")
        })
}

fn env_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

// ============================================================================
// Unit tests — no network required
// ============================================================================

/// The permission policy built from a worker spec with explicit allow rules
/// produces Allow decisions (not Ask).
#[test]
fn test_worker_spec_permissions_produce_allow_decisions() {
    let policy = PermissionPolicy::new().allow("read(*)").allow("write(*)");

    let spec =
        WorkerAgentSpec::custom("test-writer", "Write files").with_permissions(policy.clone());
    let def = spec.into_agent_definition();

    assert_eq!(
        def.permissions
            .check("read", &serde_json::json!({"file_path": "/tmp/x.txt"})),
        PermissionDecision::Allow,
        "read should be allowed by explicit allow rule"
    );
    assert_eq!(
        def.permissions.check(
            "write",
            &serde_json::json!({"file_path": "/tmp/x.txt", "content": "hi"})
        ),
        PermissionDecision::Allow,
        "write should be allowed by explicit allow rule"
    );
}

/// An implementer worker (which has general permissions including read/write/bash)
/// produces Allow decisions for its allowed tools.
#[test]
fn test_implementer_worker_permissions_allow_tools() {
    let spec = WorkerAgentSpec::implementer("impl-worker", "Implement features");
    let def = spec.into_agent_definition();

    assert!(
        !def.permissions.allow.is_empty(),
        "implementer should have non-empty allow rules"
    );

    assert_eq!(
        def.permissions
            .check("read", &serde_json::json!({"file_path": "src/main.rs"})),
        PermissionDecision::Allow,
        "implementer should be allowed to read"
    );
    assert_eq!(
        def.permissions.check(
            "write",
            &serde_json::json!({"file_path": "out.txt", "content": "x"})
        ),
        PermissionDecision::Allow,
        "implementer should be allowed to write"
    );
}

// ============================================================================
// Integration test — calls TaskExecutor directly with real LLM
// ============================================================================

/// Core reproduction of Issue #28: delegate a task to a worker agent with
/// explicit write permissions via TaskExecutor::execute and verify the child
/// run succeeds (write tool executes without MissingConfirmationManager denial).
///
/// This bypasses model non-determinism by calling TaskExecutor directly.
#[tokio::test]
#[ignore = "requires real provider credentials and network access"]
async fn test_task_executor_child_run_inherits_permissions() {
    let _guard = env_lock().lock().await;

    let config_path = repo_config_path();
    let config = CodeConfig::from_file(&config_path)
        .unwrap_or_else(|err| panic!("failed to load {}: {err}", config_path.display()));

    let llm_config = config
        .default_llm_config()
        .expect("default llm config should resolve");
    let llm_client = create_client_with_config(llm_config);

    let workspace = tempfile::tempdir().expect("temp workspace");

    // Register a custom worker with explicit read+write permissions.
    let registry = AgentRegistry::new();
    let writer_spec = WorkerAgentSpec::custom("file-writer", "Write files to workspace")
        .with_permissions(PermissionPolicy::new().allow("read(*)").allow("write(*)"))
        .with_prompt(
            "You are a file writer agent. Use the write tool to create files. \
             Do not ask for confirmation, just write the file immediately.",
        )
        .with_max_steps(3);
    registry.register(writer_spec.into_agent_definition());

    // Create TaskExecutor and execute a write task directly.
    let executor = TaskExecutor::new(
        Arc::new(registry),
        llm_client,
        workspace.path().to_string_lossy().to_string(),
    );

    let params = TaskParams {
        agent: "file-writer".to_string(),
        description: "Write hello.txt".to_string(),
        prompt: "Write a file called hello.txt with exactly this content: PERMISSION_OK"
            .to_string(),
        background: false,
        max_steps: Some(3),
    };

    let result = executor
        .execute(params, None, None)
        .await
        .expect("TaskExecutor::execute should not return Err");

    // The child run should succeed — no MissingConfirmationManager denial.
    assert!(
        result.success,
        "child run should succeed with inherited permissions. Output: {}",
        result.output
    );
    assert!(
        !result.output.contains("MissingConfirmationManager")
            && !result.output.contains("requires confirmation but no HITL"),
        "child run must not hit MissingConfirmationManager. Output: {}",
        result.output
    );

    // Verify the file was actually written.
    let hello_path = workspace.path().join("hello.txt");
    assert!(
        hello_path.exists(),
        "hello.txt should exist — write tool should have executed. Output: {}",
        result.output
    );
    let content = std::fs::read_to_string(&hello_path).unwrap();
    assert!(
        content.contains("PERMISSION_OK"),
        "hello.txt should contain PERMISSION_OK, got: {content:?}"
    );

    println!("Task output: {}", result.output);
    println!("File content: {content}");
}

/// Parallel task execution also inherits permissions correctly.
#[tokio::test]
#[ignore = "requires real provider credentials and network access"]
async fn test_parallel_task_executor_inherits_permissions() {
    let _guard = env_lock().lock().await;

    let config_path = repo_config_path();
    let config = CodeConfig::from_file(&config_path)
        .unwrap_or_else(|err| panic!("failed to load {}: {err}", config_path.display()));

    let llm_config = config
        .default_llm_config()
        .expect("default llm config should resolve");
    let llm_client = create_client_with_config(llm_config);

    let workspace = tempfile::tempdir().expect("temp workspace");

    // Register an implementer worker.
    let registry = AgentRegistry::new();
    let impl_spec = WorkerAgentSpec::implementer("impl-agent", "Implement tasks")
        .with_prompt("You write files. Use the write tool immediately without asking.")
        .with_max_steps(3);
    registry.register(impl_spec.into_agent_definition());

    let executor = Arc::new(TaskExecutor::new(
        Arc::new(registry),
        llm_client,
        workspace.path().to_string_lossy().to_string(),
    ));

    let tasks = vec![
        TaskParams {
            agent: "impl-agent".to_string(),
            description: "Write a.txt".to_string(),
            prompt: "Write a file called a.txt with exactly this content: FILE_A_OK".to_string(),
            background: false,
            max_steps: Some(3),
        },
        TaskParams {
            agent: "impl-agent".to_string(),
            description: "Write b.txt".to_string(),
            prompt: "Write a file called b.txt with exactly this content: FILE_B_OK".to_string(),
            background: false,
            max_steps: Some(3),
        },
    ];

    let results = executor.execute_parallel(tasks, None, None).await;

    assert_eq!(results.len(), 2, "should have 2 results");

    for result in &results {
        assert!(
            result.success,
            "parallel child run should succeed. Output: {}",
            result.output
        );
        assert!(
            !result.output.contains("MissingConfirmationManager")
                && !result.output.contains("requires confirmation but no HITL"),
            "parallel child run must not hit permission denial. Output: {}",
            result.output
        );
    }

    // Verify at least one file was written.
    let a_path = workspace.path().join("a.txt");
    let b_path = workspace.path().join("b.txt");

    assert!(
        a_path.exists() || b_path.exists(),
        "at least one output file should exist. Results: {:?}",
        results.iter().map(|r| &r.output).collect::<Vec<_>>()
    );

    if a_path.exists() {
        let content = std::fs::read_to_string(&a_path).unwrap();
        assert!(content.contains("FILE_A_OK"), "a.txt got: {content:?}");
    }
    if b_path.exists() {
        let content = std::fs::read_to_string(&b_path).unwrap();
        assert!(content.contains("FILE_B_OK"), "b.txt got: {content:?}");
    }
}
