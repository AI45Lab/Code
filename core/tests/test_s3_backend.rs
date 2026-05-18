//! End-to-end integration test for [`a3s_code_core::S3WorkspaceBackend`].
//!
//! Gated on `A3S_S3_TEST_ENDPOINT` so it does not run in default CI. To
//! exercise it locally, point at a MinIO / RustFS / real S3 endpoint:
//!
//! ```sh
//! export A3S_S3_TEST_ENDPOINT=http://127.0.0.1:9000
//! export A3S_S3_TEST_REGION=us-east-1
//! export A3S_S3_TEST_ACCESS_KEY_ID=minioadmin
//! export A3S_S3_TEST_SECRET_ACCESS_KEY=minioadmin
//! export A3S_S3_TEST_BUCKET=a3s-code-tests
//! export A3S_S3_TEST_FORCE_PATH_STYLE=true       # for MinIO/RustFS
//! cargo test -p a3s-code-core --features s3 --test test_s3_backend -- --ignored
//! ```
//!
//! The test uses a per-run UUID prefix so it never collides with other
//! sessions and cleans up its own keys on success.

#![cfg(feature = "s3")]

use a3s_code_core::tools::{ArtifactStoreLimits, ToolExecutor};
use a3s_code_core::{S3BackendConfig, S3WorkspaceBackend, WorkspaceServices};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

fn env_required(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

fn live_config() -> Option<S3BackendConfig> {
    let endpoint = env_required("A3S_S3_TEST_ENDPOINT")?;
    let bucket = env_required("A3S_S3_TEST_BUCKET")?;
    let access_key_id = env_required("A3S_S3_TEST_ACCESS_KEY_ID")?;
    let secret_access_key = env_required("A3S_S3_TEST_SECRET_ACCESS_KEY")?;
    let prefix = format!(
        "{}/{}",
        env_required("A3S_S3_TEST_PREFIX").unwrap_or_else(|| "a3s-code-tests".to_string()),
        Uuid::new_v4()
    );

    let mut cfg = S3BackendConfig::new(bucket, prefix, access_key_id, secret_access_key)
        .endpoint(endpoint)
        .request_timeout(Duration::from_secs(10));

    if let Some(region) = env_required("A3S_S3_TEST_REGION") {
        cfg = cfg.region(region);
    } else {
        cfg = cfg.region("us-east-1");
    }
    if let Some(force) = env_required("A3S_S3_TEST_FORCE_PATH_STYLE") {
        cfg = cfg.force_path_style(force.parse().unwrap_or(true));
    } else {
        cfg = cfg.force_path_style(true);
    }
    if let Some(token) = env_required("A3S_S3_TEST_SESSION_TOKEN") {
        cfg = cfg.session_token(token);
    }
    Some(cfg)
}

#[tokio::test]
#[ignore = "requires A3S_S3_TEST_ENDPOINT and friends"]
async fn s3_backend_roundtrips_via_session_executor() {
    let Some(cfg) = live_config() else {
        eprintln!("Skipping: A3S_S3_TEST_ENDPOINT not configured");
        return;
    };
    let prefix_for_cleanup = cfg.prefix.clone();
    let bucket_for_cleanup = cfg.bucket.clone();

    let backend = Arc::new(S3WorkspaceBackend::new(cfg));
    let services = WorkspaceServices::from_s3_backend(Arc::clone(&backend));

    // Capability gating must hide bash/git/grep/glob.
    let executor = ToolExecutor::new_with_workspace_services_and_artifact_limits(
        format!("s3://{}/{}", backend.bucket(), backend.prefix()),
        Arc::clone(&services),
        ArtifactStoreLimits::default(),
    );
    let definitions = executor.definitions();
    let names: Vec<&str> = definitions.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"read"), "read must be registered");
    assert!(names.contains(&"write"), "write must be registered");
    assert!(names.contains(&"edit"), "edit must be registered");
    assert!(names.contains(&"patch"), "patch must be registered");
    assert!(names.contains(&"ls"), "ls must be registered");
    assert!(
        !names.contains(&"bash"),
        "bash must NOT be registered for S3 backend"
    );
    assert!(
        !names.contains(&"grep"),
        "grep must NOT be registered for S3 backend"
    );
    assert!(
        !names.contains(&"glob"),
        "glob must NOT be registered for S3 backend"
    );
    assert!(
        !names.contains(&"git"),
        "git must NOT be registered for S3 backend"
    );

    // write
    let write = executor
        .execute(
            "write",
            &json!({ "file_path": "notes/hello.txt", "content": "one\ntwo\n" }),
        )
        .await
        .expect("write tool dispatched");
    assert_eq!(write.exit_code, 0, "{}", write.output);

    // read
    let read = executor
        .execute("read", &json!({ "file_path": "notes/hello.txt" }))
        .await
        .expect("read tool dispatched");
    assert_eq!(read.exit_code, 0, "{}", read.output);
    assert!(
        read.output.contains("one"),
        "read should return persisted content: {}",
        read.output
    );

    // ls — root should contain "notes" subdirectory
    let ls_root = executor
        .execute("ls", &json!({ "path": "." }))
        .await
        .expect("ls root");
    assert_eq!(ls_root.exit_code, 0, "{}", ls_root.output);
    assert!(
        ls_root.output.contains("notes"),
        "ls / should surface the notes/ prefix: {}",
        ls_root.output
    );

    // ls notes — should contain hello.txt
    let ls_notes = executor
        .execute("ls", &json!({ "path": "notes" }))
        .await
        .expect("ls notes");
    assert_eq!(ls_notes.exit_code, 0, "{}", ls_notes.output);
    assert!(
        ls_notes.output.contains("hello.txt"),
        "ls notes/ should surface hello.txt: {}",
        ls_notes.output
    );

    // edit — replace "one" with "uno"
    let edit = executor
        .execute(
            "edit",
            &json!({
                "file_path": "notes/hello.txt",
                "old_string": "one",
                "new_string": "uno"
            }),
        )
        .await
        .expect("edit tool dispatched");
    assert_eq!(edit.exit_code, 0, "{}", edit.output);

    let read_after_edit = executor
        .execute("read", &json!({ "file_path": "notes/hello.txt" }))
        .await
        .expect("read after edit");
    assert!(
        read_after_edit.output.contains("uno"),
        "edit should persist: {}",
        read_after_edit.output
    );

    // patch — apply a unified diff
    let patch = executor
        .execute(
            "patch",
            &json!({
                "file_path": "notes/hello.txt",
                "diff": "@@ -1,2 +1,2 @@\n uno\n-two\n+dos"
            }),
        )
        .await
        .expect("patch tool dispatched");
    assert_eq!(patch.exit_code, 0, "{}", patch.output);

    let final_content = executor
        .execute("read", &json!({ "file_path": "notes/hello.txt" }))
        .await
        .expect("read after patch")
        .output;
    assert!(
        final_content.contains("uno") && final_content.contains("dos"),
        "patch should produce uno/dos: {}",
        final_content
    );

    // Cleanup keys under our test prefix.
    cleanup_prefix(backend.client(), &bucket_for_cleanup, &prefix_for_cleanup).await;
}

async fn cleanup_prefix(client: &aws_sdk_s3::Client, bucket: &str, prefix: &str) {
    let prefix = if prefix.ends_with('/') {
        prefix.to_string()
    } else {
        format!("{prefix}/")
    };
    let mut continuation: Option<String> = None;
    loop {
        let mut req = client.list_objects_v2().bucket(bucket).prefix(&prefix);
        if let Some(ref token) = continuation {
            req = req.continuation_token(token);
        }
        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("cleanup list_objects_v2 failed: {e}");
                return;
            }
        };
        for obj in resp.contents() {
            if let Some(key) = obj.key() {
                let _ = client.delete_object().bucket(bucket).key(key).send().await;
            }
        }
        if resp.is_truncated().unwrap_or(false) {
            continuation = resp.next_continuation_token().map(|s| s.to_string());
            if continuation.is_none() {
                break;
            }
        } else {
            break;
        }
    }
}
