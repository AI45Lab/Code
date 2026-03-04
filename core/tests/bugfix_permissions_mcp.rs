//! Integration tests for Bug 1 and Bug 2 fixes (a3s-code v1.0.1)
//!
//! Bug 1: permissions.allow with non-empty values caused agent files to silently
//!         fail loading. Root causes:
//!         (a) PermissionRule didn't support plain-string YAML format
//!         (b) #[serde(skip)] on tool_name/arg_pattern made deserialized rules non-functional
//!
//! Bug 2: Sub-agent child sessions had no access to MCP tools because
//!         TaskExecutor didn't hold an McpManager reference.
//!
//! Run with:
//!   cargo test -p a3s-code-core --test bugfix_permissions_mcp

use a3s_code_core::permissions::{PermissionPolicy, PermissionRule};

// ============================================================================
// Bug 1(a): PermissionRule plain-string YAML deserialization
// ============================================================================

#[test]
fn bug1a_plain_string_deserialize_single_rule() {
    // Before fix: serde_yaml could NOT deserialize a plain string into PermissionRule
    let rule: PermissionRule = serde_yaml::from_str("read").unwrap();
    assert_eq!(rule.rule, "read");
    assert!(
        rule.matches("read", &serde_json::json!({})),
        "deserialized plain-string rule must match"
    );
}

#[test]
fn bug1a_plain_string_deserialize_with_pattern() {
    let rule: PermissionRule = serde_yaml::from_str("\"Bash(cargo:*)\"").unwrap();
    assert_eq!(rule.rule, "Bash(cargo:*)");
    assert!(rule.matches("Bash", &serde_json::json!({"command": "cargo build"})));
    assert!(!rule.matches("Bash", &serde_json::json!({"command": "npm install"})));
}

#[test]
fn bug1a_struct_form_still_works() {
    let rule: PermissionRule = serde_yaml::from_str("rule: read").unwrap();
    assert_eq!(rule.rule, "read");
    assert!(rule.matches("read", &serde_json::json!({})));
}

#[test]
fn bug1a_mixed_formats_in_policy() {
    let yaml = r#"
allow:
  - read
  - "Bash(cargo:*)"
  - rule: grep
deny:
  - write
  - rule: "Bash(rm:*)"
"#;
    let policy: PermissionPolicy = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(policy.allow.len(), 3);
    assert_eq!(policy.deny.len(), 2);
}

// ============================================================================
// Bug 1(b): Deserialized rules actually match (tool_name not None)
// ============================================================================

#[test]
fn bug1b_deserialized_rules_are_functional() {
    let yaml = r#"
allow:
  - read
  - grep
  - "Bash(cargo:*)"
deny:
  - write
"#;
    let policy: PermissionPolicy = serde_yaml::from_str(yaml).unwrap();

    // Before fix: all matches() returned false because tool_name was None
    assert!(
        policy.is_allowed("read", &serde_json::json!({})),
        "read should be allowed"
    );
    assert!(
        policy.is_allowed("grep", &serde_json::json!({})),
        "grep should be allowed"
    );
    assert!(
        policy.is_allowed("Bash", &serde_json::json!({"command": "cargo build"})),
        "Bash(cargo:*) should be allowed"
    );
    assert!(
        policy.is_denied("write", &serde_json::json!({})),
        "write should be denied"
    );
    assert!(
        !policy.is_allowed("edit", &serde_json::json!({})),
        "edit should not be explicitly allowed"
    );
}

#[test]
fn bug1b_mcp_tool_rules_match_after_deserialize() {
    let yaml = r#"
allow:
  - mcp__video_processor
  - mcp__longvt
"#;
    let policy: PermissionPolicy = serde_yaml::from_str(yaml).unwrap();

    assert!(
        policy.is_allowed("mcp__video_processor__analyze", &serde_json::json!({})),
        "mcp__video_processor prefix should match mcp__video_processor__analyze"
    );
    assert!(
        policy.is_allowed("mcp__longvt__process", &serde_json::json!({})),
        "mcp__longvt prefix should match mcp__longvt__process"
    );
    assert!(
        !policy.is_allowed("mcp__other__tool", &serde_json::json!({})),
        "mcp__other should NOT match"
    );
}

// ============================================================================
// Bug 1: Full agent YAML frontmatter with permissions
// ============================================================================

#[test]
fn bug1_scoring_agent_permissions_yaml() {
    // Simulates the exact frontmatter from the 11 scoring sub-agents
    let yaml = r#"
allow:
  - read
  - grep
  - mcp__video_processor
  - mcp__longvt
deny:
  - write
  - "Bash(rm:*)"
"#;
    let policy: PermissionPolicy = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(policy.allow.len(), 4);
    assert_eq!(policy.deny.len(), 2);

    // Every rule must be functional
    assert!(policy.is_allowed("read", &serde_json::json!({})));
    assert!(policy.is_allowed("grep", &serde_json::json!({})));
    assert!(policy.is_allowed("mcp__video_processor__analyze", &serde_json::json!({})));
    assert!(policy.is_allowed("mcp__longvt__process", &serde_json::json!({})));
    assert!(policy.is_denied("write", &serde_json::json!({})));
    assert!(policy.is_denied("Bash", &serde_json::json!({"command": "rm -rf /tmp"})));
    // Unlisted tool → not allowed (falls to default=Ask)
    assert!(!policy.is_allowed("Bash", &serde_json::json!({"command": "ls"})));
}

// ============================================================================
// Bug 2: TaskExecutor structural tests (MCP manager field)
// ============================================================================

#[test]
fn bug2_register_task_with_mcp_function_exists() {
    // Verify the new register_task_with_mcp is exported and compiles.
    // Full signature check isn't possible from integration tests because
    // AgentRegistry is pub(crate), but the function itself IS public.
    // The unit tests in subagent.rs and task.rs cover the internal wiring.
    //
    // Here we just verify the symbol exists and the module compiles.
    let _ = a3s_code_core::tools::register_task_with_mcp as *const ();
}

#[tokio::test]
async fn bug2_mcp_manager_tools_retrievable() {
    use a3s_code_core::mcp::manager::McpManager;

    let manager = McpManager::new();
    let tools = manager.get_all_tools().await;
    assert!(tools.is_empty(), "empty manager should return no tools");
}
