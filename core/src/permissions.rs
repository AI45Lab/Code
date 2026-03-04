//! Permission system for tool execution control
//!
//! Implements a declarative permission system similar to Claude Code's permissions.
//! Supports pattern matching with wildcards and three-tier evaluation:
//! 1. Deny rules - checked first, any match = immediate denial
//! 2. Allow rules - checked second, any match = auto-approval
//! 3. Ask rules - checked third, forces confirmation prompt
//! 4. Default behavior - falls back to HITL policy

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trait for checking tool execution permissions.
///
/// Implement this trait to provide custom permission logic.
/// The built-in `PermissionPolicy` implements this trait using
/// declarative allow/deny/ask rules with pattern matching.
pub trait PermissionChecker: Send + Sync {
    /// Check whether a tool invocation is allowed, denied, or requires confirmation.
    fn check(&self, tool_name: &str, args: &serde_json::Value) -> PermissionDecision;
}

/// Permission decision result
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionDecision {
    /// Automatically allow without user confirmation
    Allow,
    /// Deny execution
    Deny,
    /// Ask user for confirmation
    Ask,
}

/// A permission rule with pattern matching support
///
/// Format: `ToolName(pattern)` or `ToolName` (matches all)
///
/// Examples:
/// - `Bash(cargo:*)` - matches all cargo commands
/// - `Bash(npm run test:*)` - matches npm run test with any args
/// - `Read(src/**/*.rs)` - matches Rust files in src/
/// - `Grep(*)` - matches all grep invocations
/// - `mcp__pencil` - matches all pencil MCP tools
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PermissionRule {
    /// The original rule string
    pub rule: String,
    /// Parsed tool name
    #[serde(skip)]
    tool_name: Option<String>,
    /// Parsed argument pattern (None means match all)
    #[serde(skip)]
    arg_pattern: Option<String>,
}

impl PermissionRule {
    /// Create a new permission rule from a pattern string
    pub fn new(rule: &str) -> Self {
        let (tool_name, arg_pattern) = Self::parse_rule(rule);
        Self {
            rule: rule.to_string(),
            tool_name,
            arg_pattern,
        }
    }

    /// Parse rule string into tool name and argument pattern
    fn parse_rule(rule: &str) -> (Option<String>, Option<String>) {
        // Handle format: ToolName(pattern) or ToolName
        if let Some(paren_start) = rule.find('(') {
            if rule.ends_with(')') {
                let tool_name = rule[..paren_start].to_string();
                let pattern = rule[paren_start + 1..rule.len() - 1].to_string();
                return (Some(tool_name), Some(pattern));
            }
        }
        // No parentheses - tool name only, matches all args
        (Some(rule.to_string()), None)
    }

    /// Check if this rule matches a tool invocation
    pub fn matches(&self, tool_name: &str, args: &serde_json::Value) -> bool {
        // Check tool name
        let rule_tool = match &self.tool_name {
            Some(t) => t,
            None => return false,
        };

        if !self.matches_tool_name(rule_tool, tool_name) {
            return false;
        }

        // If no argument pattern, match all
        let pattern = match &self.arg_pattern {
            Some(p) => p,
            None => return true,
        };

        // Match against argument pattern
        self.matches_args(pattern, tool_name, args)
    }

    /// Check if tool names match (case-insensitive)
    fn matches_tool_name(&self, rule_tool: &str, actual_tool: &str) -> bool {
        // Handle MCP tools: mcp__server matches mcp__server__tool
        if rule_tool.starts_with("mcp__") && actual_tool.starts_with("mcp__") {
            // mcp__pencil matches mcp__pencil__batch_design
            if actual_tool.starts_with(rule_tool) {
                return true;
            }
        }
        rule_tool.eq_ignore_ascii_case(actual_tool)
    }

    /// Match argument pattern against tool arguments
    fn matches_args(&self, pattern: &str, tool_name: &str, args: &serde_json::Value) -> bool {
        // Handle wildcard pattern "*" - matches everything
        if pattern == "*" {
            return true;
        }

        // Build argument string based on tool type
        let arg_string = self.build_arg_string(tool_name, args);

        // Perform glob-style matching
        self.glob_match(pattern, &arg_string)
    }

    /// Build a string representation of arguments for matching
    fn build_arg_string(&self, tool_name: &str, args: &serde_json::Value) -> String {
        match tool_name.to_lowercase().as_str() {
            "bash" => {
                // For Bash, use the command field
                args.get("command")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            }
            "read" | "write" | "edit" => {
                // For file operations, use the file_path field
                args.get("file_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            }
            "glob" => {
                // For glob, use the pattern field
                args.get("pattern")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            }
            "grep" => {
                // For grep, combine pattern and path
                let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
                let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
                format!("{} {}", pattern, path)
            }
            "ls" => {
                // For ls, use the path field
                args.get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            }
            _ => {
                // For other tools, serialize the entire args
                serde_json::to_string(args).unwrap_or_default()
            }
        }
    }

    /// Perform glob-style pattern matching
    ///
    /// Supports:
    /// - `*` matches any sequence of characters (except /)
    /// - `**` matches any sequence including /
    /// - `:*` at the end matches any suffix (including empty)
    fn glob_match(&self, pattern: &str, text: &str) -> bool {
        // Handle special `:*` suffix (matches any args after the prefix)
        if let Some(prefix) = pattern.strip_suffix(":*") {
            return text.starts_with(prefix);
        }

        // Normalize Windows backslashes to forward slashes for consistent matching
        let text = text.replace('\\', "/");

        // Convert glob pattern to regex pattern
        let regex_pattern = Self::glob_to_regex(pattern);
        if let Ok(re) = regex::Regex::new(&regex_pattern) {
            re.is_match(&text)
        } else {
            // Fallback to simple prefix match if regex fails
            text.starts_with(pattern)
        }
    }

    /// Convert glob pattern to regex pattern
    fn glob_to_regex(pattern: &str) -> String {
        let mut regex = String::from("^");
        let chars: Vec<char> = pattern.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let c = chars[i];
            match c {
                '*' => {
                    // Check for ** (matches anything including /)
                    if i + 1 < chars.len() && chars[i + 1] == '*' {
                        // ** matches any path including /
                        // Skip optional following /
                        if i + 2 < chars.len() && chars[i + 2] == '/' {
                            regex.push_str(".*");
                            i += 3;
                        } else {
                            regex.push_str(".*");
                            i += 2;
                        }
                    } else {
                        // * matches anything except path separators
                        regex.push_str("[^/\\\\]*");
                        i += 1;
                    }
                }
                '?' => {
                    // ? matches any single character except path separators
                    regex.push_str("[^/\\\\]");
                    i += 1;
                }
                '.' | '+' | '^' | '$' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '\\' => {
                    // Escape regex special characters
                    regex.push('\\');
                    regex.push(c);
                    i += 1;
                }
                _ => {
                    regex.push(c);
                    i += 1;
                }
            }
        }

        regex.push('$');
        regex
    }
}

/// Permission policy configuration
///
/// Evaluation order:
/// 1. Deny rules - any match results in denial
/// 2. Allow rules - any match results in auto-approval
/// 3. Ask rules - any match requires user confirmation
/// 4. Default - falls back to default_decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionPolicy {
    /// Rules that always deny (checked first)
    #[serde(default)]
    pub deny: Vec<PermissionRule>,

    /// Rules that auto-approve without confirmation
    #[serde(default)]
    pub allow: Vec<PermissionRule>,

    /// Rules that always require confirmation
    #[serde(default)]
    pub ask: Vec<PermissionRule>,

    /// Default decision when no rules match
    #[serde(default = "default_decision")]
    pub default_decision: PermissionDecision,

    /// Whether the permission system is enabled
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_decision() -> PermissionDecision {
    PermissionDecision::Ask
}

fn default_enabled() -> bool {
    true
}

impl Default for PermissionPolicy {
    fn default() -> Self {
        Self {
            deny: Vec::new(),
            allow: Vec::new(),
            ask: Vec::new(),
            default_decision: PermissionDecision::Ask,
            enabled: true,
        }
    }
}

impl PermissionPolicy {
    /// Create a new permission policy
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a permissive policy that allows everything
    pub fn permissive() -> Self {
        Self {
            deny: Vec::new(),
            allow: Vec::new(),
            ask: Vec::new(),
            default_decision: PermissionDecision::Allow,
            enabled: true,
        }
    }

    /// Create a strict policy that asks for everything
    pub fn strict() -> Self {
        Self {
            deny: Vec::new(),
            allow: Vec::new(),
            ask: Vec::new(),
            default_decision: PermissionDecision::Ask,
            enabled: true,
        }
    }

    /// Add a deny rule
    pub fn deny(mut self, rule: &str) -> Self {
        self.deny.push(PermissionRule::new(rule));
        self
    }

    /// Add an allow rule
    pub fn allow(mut self, rule: &str) -> Self {
        self.allow.push(PermissionRule::new(rule));
        self
    }

    /// Add an ask rule
    pub fn ask(mut self, rule: &str) -> Self {
        self.ask.push(PermissionRule::new(rule));
        self
    }

    /// Add multiple deny rules
    pub fn deny_all(mut self, rules: &[&str]) -> Self {
        for rule in rules {
            self.deny.push(PermissionRule::new(rule));
        }
        self
    }

    /// Add multiple allow rules
    pub fn allow_all(mut self, rules: &[&str]) -> Self {
        for rule in rules {
            self.allow.push(PermissionRule::new(rule));
        }
        self
    }

    /// Add multiple ask rules
    pub fn ask_all(mut self, rules: &[&str]) -> Self {
        for rule in rules {
            self.ask.push(PermissionRule::new(rule));
        }
        self
    }

    /// Check permission for a tool invocation
    ///
    /// Returns the permission decision based on rule evaluation order:
    /// 1. Deny rules (any match = Deny)
    /// 2. Allow rules (any match = Allow)
    /// 3. Ask rules (any match = Ask)
    /// 4. Default decision
    pub fn check(&self, tool_name: &str, args: &serde_json::Value) -> PermissionDecision {
        if !self.enabled {
            return PermissionDecision::Allow;
        }

        // 1. Check deny rules first
        for rule in &self.deny {
            if rule.matches(tool_name, args) {
                return PermissionDecision::Deny;
            }
        }

        // 2. Check allow rules
        for rule in &self.allow {
            if rule.matches(tool_name, args) {
                return PermissionDecision::Allow;
            }
        }

        // 3. Check ask rules
        for rule in &self.ask {
            if rule.matches(tool_name, args) {
                return PermissionDecision::Ask;
            }
        }

        // 4. Fall back to default
        self.default_decision
    }

    /// Check if a tool invocation is allowed (Allow or not Deny)
    pub fn is_allowed(&self, tool_name: &str, args: &serde_json::Value) -> bool {
        matches!(self.check(tool_name, args), PermissionDecision::Allow)
    }

    /// Check if a tool invocation is denied
    pub fn is_denied(&self, tool_name: &str, args: &serde_json::Value) -> bool {
        matches!(self.check(tool_name, args), PermissionDecision::Deny)
    }

    /// Check if a tool invocation requires confirmation
    pub fn requires_confirmation(&self, tool_name: &str, args: &serde_json::Value) -> bool {
        matches!(self.check(tool_name, args), PermissionDecision::Ask)
    }

    /// Get matching rules for debugging/logging
    pub fn get_matching_rules(&self, tool_name: &str, args: &serde_json::Value) -> MatchingRules {
        let mut result = MatchingRules::default();

        for rule in &self.deny {
            if rule.matches(tool_name, args) {
                result.deny.push(rule.rule.clone());
            }
        }

        for rule in &self.allow {
            if rule.matches(tool_name, args) {
                result.allow.push(rule.rule.clone());
            }
        }

        for rule in &self.ask {
            if rule.matches(tool_name, args) {
                result.ask.push(rule.rule.clone());
            }
        }

        result
    }
}

impl PermissionChecker for PermissionPolicy {
    fn check(&self, tool_name: &str, args: &serde_json::Value) -> PermissionDecision {
        self.check(tool_name, args)
    }
}

/// Matching rules for debugging
#[derive(Debug, Default, Clone)]
pub struct MatchingRules {
    pub deny: Vec<String>,
    pub allow: Vec<String>,
    pub ask: Vec<String>,
}

impl MatchingRules {
    pub fn is_empty(&self) -> bool {
        self.deny.is_empty() && self.allow.is_empty() && self.ask.is_empty()
    }
}

/// Permission manager that handles per-session permissions
#[derive(Debug)]
pub struct PermissionManager {
    /// Global policy applied to all sessions
    global_policy: PermissionPolicy,
    /// Per-session policy overrides
    session_policies: HashMap<String, PermissionPolicy>,
}

impl Default for PermissionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PermissionManager {
    /// Create a new permission manager with default global policy
    pub fn new() -> Self {
        Self {
            global_policy: PermissionPolicy::default(),
            session_policies: HashMap::new(),
        }
    }

    /// Create with a custom global policy
    pub fn with_global_policy(policy: PermissionPolicy) -> Self {
        Self {
            global_policy: policy,
            session_policies: HashMap::new(),
        }
    }

    /// Set the global policy
    pub fn set_global_policy(&mut self, policy: PermissionPolicy) {
        self.global_policy = policy;
    }

    /// Get the global policy
    pub fn global_policy(&self) -> &PermissionPolicy {
        &self.global_policy
    }

    /// Set a session-specific policy
    pub fn set_session_policy(&mut self, session_id: &str, policy: PermissionPolicy) {
        self.session_policies.insert(session_id.to_string(), policy);
    }

    /// Remove a session-specific policy
    pub fn remove_session_policy(&mut self, session_id: &str) {
        self.session_policies.remove(session_id);
    }

    /// Get the effective policy for a session
    ///
    /// Session policy takes precedence over global policy for matching rules.
    /// If no session policy exists, uses global policy.
    pub fn get_effective_policy(&self, session_id: &str) -> &PermissionPolicy {
        self.session_policies
            .get(session_id)
            .unwrap_or(&self.global_policy)
    }

    /// Check permission for a tool invocation in a session
    pub fn check(
        &self,
        session_id: &str,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> PermissionDecision {
        // Get session policy or fall back to global
        let policy = self.get_effective_policy(session_id);

        // Session deny rules
        for rule in &policy.deny {
            if rule.matches(tool_name, args) {
                return PermissionDecision::Deny;
            }
        }

        // Global deny rules (if different policy)
        if !self.session_policies.contains_key(session_id) {
            // Already checked global
        } else {
            for rule in &self.global_policy.deny {
                if rule.matches(tool_name, args) {
                    return PermissionDecision::Deny;
                }
            }
        }

        // Session allow rules
        for rule in &policy.allow {
            if rule.matches(tool_name, args) {
                return PermissionDecision::Allow;
            }
        }

        // Session ask rules
        for rule in &policy.ask {
            if rule.matches(tool_name, args) {
                return PermissionDecision::Ask;
            }
        }

        // Fall back to policy default
        policy.default_decision
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ========================================================================
    // PermissionRule Tests
    // ========================================================================

    #[test]
    fn test_rule_parse_simple() {
        let rule = PermissionRule::new("Bash");
        assert_eq!(rule.tool_name, Some("Bash".to_string()));
        assert_eq!(rule.arg_pattern, None);
    }

    #[test]
    fn test_rule_parse_with_pattern() {
        let rule = PermissionRule::new("Bash(cargo:*)");
        assert_eq!(rule.tool_name, Some("Bash".to_string()));
        assert_eq!(rule.arg_pattern, Some("cargo:*".to_string()));
    }

    #[test]
    fn test_rule_parse_wildcard() {
        let rule = PermissionRule::new("Grep(*)");
        assert_eq!(rule.tool_name, Some("Grep".to_string()));
        assert_eq!(rule.arg_pattern, Some("*".to_string()));
    }

    #[test]
    fn test_rule_match_tool_only() {
        let rule = PermissionRule::new("Bash");
        assert!(rule.matches("Bash", &json!({"command": "ls -la"})));
        assert!(rule.matches("bash", &json!({"command": "echo hello"})));
        assert!(!rule.matches("Read", &json!({})));
    }

    #[test]
    fn test_rule_match_wildcard() {
        let rule = PermissionRule::new("Grep(*)");
        assert!(rule.matches("Grep", &json!({"pattern": "foo", "path": "/tmp"})));
        assert!(rule.matches("grep", &json!({"pattern": "bar"})));
    }

    #[test]
    fn test_rule_match_prefix_wildcard() {
        let rule = PermissionRule::new("Bash(cargo:*)");
        assert!(rule.matches("Bash", &json!({"command": "cargo build"})));
        assert!(rule.matches("Bash", &json!({"command": "cargo test --lib"})));
        assert!(rule.matches("Bash", &json!({"command": "cargo"})));
        assert!(!rule.matches("Bash", &json!({"command": "npm install"})));
    }

    #[test]
    fn test_rule_match_npm_commands() {
        let rule = PermissionRule::new("Bash(npm run:*)");
        assert!(rule.matches("Bash", &json!({"command": "npm run test"})));
        assert!(rule.matches("Bash", &json!({"command": "npm run build"})));
        assert!(!rule.matches("Bash", &json!({"command": "npm install"})));
    }

    #[test]
    fn test_rule_match_file_path() {
        let rule = PermissionRule::new("Read(src/*.rs)");
        assert!(rule.matches("Read", &json!({"file_path": "src/main.rs"})));
        assert!(rule.matches("Read", &json!({"file_path": "src/lib.rs"})));
        assert!(!rule.matches("Read", &json!({"file_path": "src/foo/bar.rs"})));
    }

    #[test]
    fn test_rule_match_recursive_glob() {
        let rule = PermissionRule::new("Read(src/**/*.rs)");
        assert!(rule.matches("Read", &json!({"file_path": "src/main.rs"})));
        assert!(rule.matches("Read", &json!({"file_path": "src/foo/bar.rs"})));
        assert!(rule.matches("Read", &json!({"file_path": "src/a/b/c.rs"})));
    }

    #[test]
    fn test_rule_match_mcp_tool() {
        let rule = PermissionRule::new("mcp__pencil");
        assert!(rule.matches("mcp__pencil", &json!({})));
        assert!(rule.matches("mcp__pencil__batch_design", &json!({})));
        assert!(rule.matches("mcp__pencil__batch_get", &json!({})));
        assert!(!rule.matches("mcp__other", &json!({})));
    }

    #[test]
    fn test_rule_case_insensitive() {
        let rule = PermissionRule::new("BASH(cargo:*)");
        assert!(rule.matches("Bash", &json!({"command": "cargo build"})));
        assert!(rule.matches("bash", &json!({"command": "cargo test"})));
        assert!(rule.matches("BASH", &json!({"command": "cargo check"})));
    }

    // ========================================================================
    // PermissionPolicy Tests
    // ========================================================================

    #[test]
    fn test_policy_default() {
        let policy = PermissionPolicy::default();
        assert!(policy.enabled);
        assert_eq!(policy.default_decision, PermissionDecision::Ask);
        assert!(policy.allow.is_empty());
        assert!(policy.deny.is_empty());
        assert!(policy.ask.is_empty());
    }

    #[test]
    fn test_policy_permissive() {
        let policy = PermissionPolicy::permissive();
        assert_eq!(policy.default_decision, PermissionDecision::Allow);
    }

    #[test]
    fn test_policy_strict() {
        let policy = PermissionPolicy::strict();
        assert_eq!(policy.default_decision, PermissionDecision::Ask);
    }

    #[test]
    fn test_policy_builder() {
        let policy = PermissionPolicy::new()
            .allow("Bash(cargo:*)")
            .allow("Grep(*)")
            .deny("Bash(rm -rf:*)")
            .ask("Write(*)");

        assert_eq!(policy.allow.len(), 2);
        assert_eq!(policy.deny.len(), 1);
        assert_eq!(policy.ask.len(), 1);
    }

    #[test]
    fn test_policy_check_allow() {
        let policy = PermissionPolicy::new().allow("Bash(cargo:*)");

        let decision = policy.check("Bash", &json!({"command": "cargo build"}));
        assert_eq!(decision, PermissionDecision::Allow);
    }

    #[test]
    fn test_policy_check_deny() {
        let policy = PermissionPolicy::new().deny("Bash(rm -rf:*)");

        let decision = policy.check("Bash", &json!({"command": "rm -rf /"}));
        assert_eq!(decision, PermissionDecision::Deny);
    }

    #[test]
    fn test_policy_check_ask() {
        let policy = PermissionPolicy::new().ask("Write(*)");

        let decision = policy.check("Write", &json!({"file_path": "/tmp/test.txt"}));
        assert_eq!(decision, PermissionDecision::Ask);
    }

    #[test]
    fn test_policy_check_default() {
        let policy = PermissionPolicy::new();

        let decision = policy.check("Unknown", &json!({}));
        assert_eq!(decision, PermissionDecision::Ask);
    }

    #[test]
    fn test_policy_deny_wins_over_allow() {
        let policy = PermissionPolicy::new().allow("Bash(*)").deny("Bash(rm:*)");

        // Allow matches, but deny also matches - deny wins
        let decision = policy.check("Bash", &json!({"command": "rm -rf /tmp"}));
        assert_eq!(decision, PermissionDecision::Deny);

        // Only allow matches
        let decision = policy.check("Bash", &json!({"command": "ls -la"}));
        assert_eq!(decision, PermissionDecision::Allow);
    }

    #[test]
    fn test_policy_allow_wins_over_ask() {
        let policy = PermissionPolicy::new()
            .allow("Bash(cargo:*)")
            .ask("Bash(*)");

        // Both match, but allow is checked before ask
        let decision = policy.check("Bash", &json!({"command": "cargo build"}));
        assert_eq!(decision, PermissionDecision::Allow);

        // Only ask matches
        let decision = policy.check("Bash", &json!({"command": "npm install"}));
        assert_eq!(decision, PermissionDecision::Ask);
    }

    #[test]
    fn test_policy_disabled() {
        let mut policy = PermissionPolicy::new().deny("Bash(rm:*)").ask("Bash(*)");
        policy.enabled = false;

        // When disabled, everything is allowed
        let decision = policy.check("Bash", &json!({"command": "rm -rf /"}));
        assert_eq!(decision, PermissionDecision::Allow);
    }

    #[test]
    fn test_policy_is_allowed() {
        let policy = PermissionPolicy::new().allow("Bash(cargo:*)");

        assert!(policy.is_allowed("Bash", &json!({"command": "cargo build"})));
        assert!(!policy.is_allowed("Bash", &json!({"command": "npm install"})));
    }

    #[test]
    fn test_policy_is_denied() {
        let policy = PermissionPolicy::new().deny("Bash(rm:*)");

        assert!(policy.is_denied("Bash", &json!({"command": "rm -rf /"})));
        assert!(!policy.is_denied("Bash", &json!({"command": "ls -la"})));
    }

    #[test]
    fn test_policy_requires_confirmation() {
        // Create a policy that explicitly allows Read but asks for Write
        let mut policy = PermissionPolicy::new().allow("Read(*)").ask("Write(*)");
        policy.default_decision = PermissionDecision::Deny; // Make default deny to test ask rule

        assert!(policy.requires_confirmation("Write", &json!({"file_path": "/tmp/test"})));
        assert!(!policy.requires_confirmation("Read", &json!({"file_path": "/tmp/test"})));
    }

    #[test]
    fn test_policy_matching_rules() {
        let policy = PermissionPolicy::new()
            .allow("Bash(cargo:*)")
            .deny("Bash(cargo fmt:*)")
            .ask("Bash(*)");

        let matching = policy.get_matching_rules("Bash", &json!({"command": "cargo fmt"}));
        assert_eq!(matching.deny.len(), 1);
        assert_eq!(matching.allow.len(), 1);
        assert_eq!(matching.ask.len(), 1);
    }

    #[test]
    fn test_policy_allow_all() {
        let policy =
            PermissionPolicy::new().allow_all(&["Bash(cargo:*)", "Bash(npm:*)", "Grep(*)"]);

        assert_eq!(policy.allow.len(), 3);
        assert!(policy.is_allowed("Bash", &json!({"command": "cargo build"})));
        assert!(policy.is_allowed("Bash", &json!({"command": "npm run test"})));
        assert!(policy.is_allowed("Grep", &json!({"pattern": "foo"})));
    }

    // ========================================================================
    // PermissionManager Tests
    // ========================================================================

    #[test]
    fn test_manager_default() {
        let manager = PermissionManager::new();
        assert_eq!(
            manager.global_policy().default_decision,
            PermissionDecision::Ask
        );
    }

    #[test]
    fn test_manager_with_global_policy() {
        let policy = PermissionPolicy::permissive();
        let manager = PermissionManager::with_global_policy(policy);
        assert_eq!(
            manager.global_policy().default_decision,
            PermissionDecision::Allow
        );
    }

    #[test]
    fn test_manager_session_policy() {
        let mut manager = PermissionManager::new();

        let session_policy = PermissionPolicy::new().allow("Bash(cargo:*)");
        manager.set_session_policy("session-1", session_policy);

        // Session 1 has custom policy
        let decision = manager.check("session-1", "Bash", &json!({"command": "cargo build"}));
        assert_eq!(decision, PermissionDecision::Allow);

        // Session 2 uses global policy
        let decision = manager.check("session-2", "Bash", &json!({"command": "cargo build"}));
        assert_eq!(decision, PermissionDecision::Ask);
    }

    #[test]
    fn test_manager_remove_session_policy() {
        let mut manager = PermissionManager::new();

        let session_policy = PermissionPolicy::permissive();
        manager.set_session_policy("session-1", session_policy);

        // Before removal
        let decision = manager.check("session-1", "Bash", &json!({"command": "anything"}));
        assert_eq!(decision, PermissionDecision::Allow);

        manager.remove_session_policy("session-1");

        // After removal - falls back to global
        let decision = manager.check("session-1", "Bash", &json!({"command": "anything"}));
        assert_eq!(decision, PermissionDecision::Ask);
    }

    #[test]
    fn test_manager_global_deny_overrides_session_allow() {
        let mut manager =
            PermissionManager::with_global_policy(PermissionPolicy::new().deny("Bash(rm:*)"));

        let session_policy = PermissionPolicy::new().allow("Bash(*)");
        manager.set_session_policy("session-1", session_policy);

        // Session allows all Bash, but global denies rm
        let decision = manager.check("session-1", "Bash", &json!({"command": "rm -rf /"}));
        assert_eq!(decision, PermissionDecision::Deny);

        // Other commands are allowed
        let decision = manager.check("session-1", "Bash", &json!({"command": "ls -la"}));
        assert_eq!(decision, PermissionDecision::Allow);
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[test]
    fn test_realistic_dev_policy() {
        let policy = PermissionPolicy::new()
            // Allow common dev commands
            .allow_all(&[
                "Bash(cargo:*)",
                "Bash(npm:*)",
                "Bash(pnpm:*)",
                "Bash(just:*)",
                "Bash(git status:*)",
                "Bash(git diff:*)",
                "Bash(echo:*)",
                "Grep(*)",
                "Glob(*)",
                "Ls(*)",
            ])
            // Deny dangerous commands
            .deny_all(&["Bash(rm -rf:*)", "Bash(sudo:*)", "Bash(curl | sh:*)"])
            // Always ask for writes
            .ask_all(&["Write(*)", "Edit(*)"]);

        // Allowed
        assert!(policy.is_allowed("Bash", &json!({"command": "cargo build"})));
        assert!(policy.is_allowed("Bash", &json!({"command": "npm run test"})));
        assert!(policy.is_allowed("Grep", &json!({"pattern": "TODO"})));

        // Denied
        assert!(policy.is_denied("Bash", &json!({"command": "rm -rf /"})));
        assert!(policy.is_denied("Bash", &json!({"command": "sudo apt install"})));

        // Ask
        assert!(policy.requires_confirmation("Write", &json!({"file_path": "/tmp/test.rs"})));
        assert!(policy.requires_confirmation("Edit", &json!({"file_path": "src/main.rs"})));
    }

    #[test]
    fn test_mcp_tool_permissions() {
        let policy = PermissionPolicy::new()
            .allow("mcp__pencil")
            .deny("mcp__dangerous");

        assert!(policy.is_allowed("mcp__pencil__batch_design", &json!({})));
        assert!(policy.is_allowed("mcp__pencil__batch_get", &json!({})));
        assert!(policy.is_denied("mcp__dangerous__execute", &json!({})));
    }

    #[test]
    fn test_serialization() {
        let policy = PermissionPolicy::new()
            .allow("Bash(cargo:*)")
            .deny("Bash(rm:*)");

        let json = serde_json::to_string(&policy).unwrap();
        let deserialized: PermissionPolicy = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.allow.len(), 1);
        assert_eq!(deserialized.deny.len(), 1);
    }

    #[test]
    fn test_matching_rules_is_empty() {
        let rules = MatchingRules {
            deny: vec![],
            allow: vec![],
            ask: vec![],
        };
        assert!(rules.is_empty());

        let rules = MatchingRules {
            deny: vec!["Bash".to_string()],
            allow: vec![],
            ask: vec![],
        };
        assert!(!rules.is_empty());

        let rules = MatchingRules {
            deny: vec![],
            allow: vec!["Read".to_string()],
            ask: vec![],
        };
        assert!(!rules.is_empty());

        let rules = MatchingRules {
            deny: vec![],
            allow: vec![],
            ask: vec!["Write".to_string()],
        };
        assert!(!rules.is_empty());
    }

    #[test]
    fn test_permission_manager_default() {
        let pm = PermissionManager::default();
        let policy = pm.global_policy();
        assert!(policy.allow.is_empty());
        assert!(policy.deny.is_empty());
        assert!(policy.ask.is_empty());
    }

    #[test]
    fn test_permission_manager_set_global_policy() {
        let mut pm = PermissionManager::new();
        let policy = PermissionPolicy::new().allow("Bash(*)");
        pm.set_global_policy(policy);
        assert_eq!(pm.global_policy().allow.len(), 1);
    }

    #[test]
    fn test_permission_manager_session_policy() {
        let mut pm = PermissionManager::new();
        let policy = PermissionPolicy::new().deny("Bash(rm:*)");
        pm.set_session_policy("s1", policy);

        let effective = pm.get_effective_policy("s1");
        assert_eq!(effective.deny.len(), 1);

        // Non-existent session falls back to global
        let global = pm.get_effective_policy("s2");
        assert!(global.deny.is_empty());
    }

    #[test]
    fn test_permission_manager_remove_session_policy() {
        let mut pm = PermissionManager::new();
        pm.set_session_policy("s1", PermissionPolicy::new().deny("Bash(*)"));
        assert_eq!(pm.get_effective_policy("s1").deny.len(), 1);

        pm.remove_session_policy("s1");
        assert!(pm.get_effective_policy("s1").deny.is_empty());
    }

    #[test]
    fn test_permission_manager_check_deny() {
        let mut pm = PermissionManager::new();
        pm.set_global_policy(PermissionPolicy::new().deny("Bash(rm:*)"));

        let decision = pm.check("s1", "Bash", &json!({"command": "rm -rf /"}));
        assert_eq!(decision, PermissionDecision::Deny);
    }

    #[test]
    fn test_permission_manager_check_allow() {
        let mut pm = PermissionManager::new();
        pm.set_global_policy(PermissionPolicy::new().allow("Bash(cargo:*)"));

        let decision = pm.check("s1", "Bash", &json!({"command": "cargo build"}));
        assert_eq!(decision, PermissionDecision::Allow);
    }

    #[test]
    fn test_permission_manager_check_session_override() {
        let mut pm = PermissionManager::new();
        pm.set_global_policy(PermissionPolicy::new().allow("Bash(*)"));
        pm.set_session_policy("s1", PermissionPolicy::new().deny("Bash(rm:*)"));

        // Session policy denies rm
        let decision = pm.check("s1", "Bash", &json!({"command": "rm -rf /"}));
        assert_eq!(decision, PermissionDecision::Deny);

        // Other session uses global (allow all)
        let decision = pm.check("s2", "Bash", &json!({"command": "rm -rf /"}));
        assert_eq!(decision, PermissionDecision::Allow);
    }

    #[test]
    fn test_permission_manager_with_global_policy() {
        let policy = PermissionPolicy::new().allow("Read(*)").deny("Write(*)");
        let pm = PermissionManager::with_global_policy(policy);
        assert_eq!(pm.global_policy().allow.len(), 1);
        assert_eq!(pm.global_policy().deny.len(), 1);
    }
}
