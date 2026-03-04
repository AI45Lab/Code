//! Hook Matchers
//!
//! Matchers filter which events trigger a hook based on patterns.

use super::events::HookEvent;
use serde::{Deserialize, Serialize};

/// Hook matcher for filtering events
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HookMatcher {
    /// Match specific tool name (exact match)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,

    /// Match file path pattern (glob)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path_pattern: Option<String>,

    /// Match command pattern (regex for Bash commands)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command_pattern: Option<String>,

    /// Match session ID (exact match)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,

    /// Match skill name (supports glob patterns)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skill: Option<String>,
}

impl HookMatcher {
    /// Create an empty matcher (matches all)
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a matcher for a specific tool
    pub fn tool(name: impl Into<String>) -> Self {
        Self {
            tool: Some(name.into()),
            ..Default::default()
        }
    }

    /// Create a matcher for a file path pattern
    pub fn path(pattern: impl Into<String>) -> Self {
        Self {
            path_pattern: Some(pattern.into()),
            ..Default::default()
        }
    }

    /// Create a matcher for a command pattern
    pub fn command(pattern: impl Into<String>) -> Self {
        Self {
            command_pattern: Some(pattern.into()),
            ..Default::default()
        }
    }

    /// Create a matcher for a specific session
    pub fn session(id: impl Into<String>) -> Self {
        Self {
            session_id: Some(id.into()),
            ..Default::default()
        }
    }

    /// Create a matcher for a specific skill (supports glob patterns)
    pub fn skill(name: impl Into<String>) -> Self {
        Self {
            skill: Some(name.into()),
            ..Default::default()
        }
    }

    /// Add tool filter
    pub fn with_tool(mut self, name: impl Into<String>) -> Self {
        self.tool = Some(name.into());
        self
    }

    /// Add path pattern filter
    pub fn with_path(mut self, pattern: impl Into<String>) -> Self {
        self.path_pattern = Some(pattern.into());
        self
    }

    /// Add command pattern filter
    pub fn with_command(mut self, pattern: impl Into<String>) -> Self {
        self.command_pattern = Some(pattern.into());
        self
    }

    /// Add session filter
    pub fn with_session(mut self, id: impl Into<String>) -> Self {
        self.session_id = Some(id.into());
        self
    }

    /// Add skill filter (supports glob patterns)
    pub fn with_skill(mut self, name: impl Into<String>) -> Self {
        self.skill = Some(name.into());
        self
    }

    /// Check if an event matches this matcher
    pub fn matches(&self, event: &HookEvent) -> bool {
        // Check session ID
        if let Some(ref session_id) = self.session_id {
            if event.session_id() != session_id {
                return false;
            }
        }

        // Check tool name
        if let Some(ref tool_pattern) = self.tool {
            if let Some(tool_name) = event.tool_name() {
                if tool_name != tool_pattern {
                    return false;
                }
            } else {
                // Event doesn't have a tool, but we're filtering by tool
                return false;
            }
        }

        // Check path pattern (in tool args)
        if let Some(ref path_pattern) = self.path_pattern {
            if !self.matches_path_pattern(event, path_pattern) {
                return false;
            }
        }

        // Check command pattern (in Bash args)
        if let Some(ref command_pattern) = self.command_pattern {
            if !self.matches_command_pattern(event, command_pattern) {
                return false;
            }
        }

        // Check skill name (supports glob patterns)
        if let Some(ref skill_pattern) = self.skill {
            if let Some(skill_name) = event.skill_name() {
                if !self.glob_match(skill_pattern, skill_name) {
                    return false;
                }
            } else {
                // Event doesn't have a skill, but we're filtering by skill
                return false;
            }
        }

        true
    }

    /// Check if event matches a path pattern (glob)
    fn matches_path_pattern(&self, event: &HookEvent, pattern: &str) -> bool {
        let args = match event.tool_args() {
            Some(args) => args,
            None => return false,
        };

        // Look for common path fields
        let path = args
            .get("file_path")
            .or_else(|| args.get("path"))
            .and_then(|v| v.as_str());

        match path {
            Some(p) => self.glob_match(pattern, p),
            None => false,
        }
    }

    /// Check if event matches a command pattern (regex)
    fn matches_command_pattern(&self, event: &HookEvent, pattern: &str) -> bool {
        // Only applies to Bash tool
        if event.tool_name() != Some("Bash") && event.tool_name() != Some("bash") {
            return false;
        }

        let args = match event.tool_args() {
            Some(args) => args,
            None => return false,
        };

        let command = args.get("command").and_then(|v| v.as_str());

        match command {
            Some(cmd) => {
                // Use regex matching
                if let Ok(re) = regex::Regex::new(pattern) {
                    re.is_match(cmd)
                } else {
                    // Fallback to contains if regex is invalid
                    cmd.contains(pattern)
                }
            }
            None => false,
        }
    }

    /// Simple glob matching (supports * and **)
    ///
    /// Matching rules:
    /// - `*` matches any characters in filename (excluding `/`)
    /// - `**` matches any path (including `/`)
    /// - `*.ext` matches any file ending with `.ext` (any depth)
    /// - `dir/**/*.ext` matches `.ext` files at any depth under dir
    fn glob_match(&self, pattern: &str, text: &str) -> bool {
        // Normalize Windows backslashes to forward slashes for consistent matching
        let text = text.replace('\\', "/");

        // Special handling: if pattern starts with * and has no /,
        // match file suffix. e.g., "*.rs" should match "src/main.rs"
        if pattern.starts_with('*') && !pattern.contains('/') {
            let suffix = &pattern[1..]; // Remove leading *
            return text.ends_with(suffix);
        }

        // Convert glob to regex
        let regex_pattern = pattern
            .replace('.', r"\.")
            .replace("**/", "__DOUBLE_STAR_SLASH__")
            .replace("**", "__DOUBLE_STAR__")
            .replace('*', "[^/]*")
            .replace("__DOUBLE_STAR_SLASH__", "(?:.*/)?") // **/ matches zero or more directories
            .replace("__DOUBLE_STAR__", ".*");

        let regex_pattern = format!("^{}$", regex_pattern);

        if let Ok(re) = regex::Regex::new(&regex_pattern) {
            re.is_match(&text)
        } else {
            text == pattern
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::events::PreToolUseEvent;

    fn make_pre_tool_event(session_id: &str, tool: &str, args: serde_json::Value) -> HookEvent {
        HookEvent::PreToolUse(PreToolUseEvent {
            session_id: session_id.to_string(),
            tool: tool.to_string(),
            args,
            working_directory: "/workspace".to_string(),
            recent_tools: vec![],
        })
    }

    #[test]
    fn test_empty_matcher_matches_all() {
        let matcher = HookMatcher::new();
        let event = make_pre_tool_event("s1", "Bash", serde_json::json!({}));
        assert!(matcher.matches(&event));
    }

    #[test]
    fn test_tool_matcher() {
        let matcher = HookMatcher::tool("Bash");

        let bash_event = make_pre_tool_event("s1", "Bash", serde_json::json!({}));
        let read_event = make_pre_tool_event("s1", "Read", serde_json::json!({}));

        assert!(matcher.matches(&bash_event));
        assert!(!matcher.matches(&read_event));
    }

    #[test]
    fn test_session_matcher() {
        let matcher = HookMatcher::session("session-1");

        let s1_event = make_pre_tool_event("session-1", "Bash", serde_json::json!({}));
        let s2_event = make_pre_tool_event("session-2", "Bash", serde_json::json!({}));

        assert!(matcher.matches(&s1_event));
        assert!(!matcher.matches(&s2_event));
    }

    #[test]
    fn test_path_pattern_matcher() {
        let matcher = HookMatcher::path("*.rs");

        let rs_event = make_pre_tool_event(
            "s1",
            "Write",
            serde_json::json!({"file_path": "src/main.rs"}),
        );
        let py_event = make_pre_tool_event(
            "s1",
            "Write",
            serde_json::json!({"file_path": "src/main.py"}),
        );

        assert!(matcher.matches(&rs_event));
        assert!(!matcher.matches(&py_event));
    }

    #[test]
    fn test_path_pattern_double_star() {
        let matcher = HookMatcher::path("src/**/*.rs");

        let nested_event = make_pre_tool_event(
            "s1",
            "Write",
            serde_json::json!({"file_path": "src/deep/nested/file.rs"}),
        );
        let root_event = make_pre_tool_event(
            "s1",
            "Write",
            serde_json::json!({"file_path": "src/file.rs"}),
        );

        assert!(matcher.matches(&nested_event));
        assert!(matcher.matches(&root_event));
    }

    #[test]
    fn test_command_pattern_matcher() {
        let matcher = HookMatcher::command(r"rm\s+-rf");

        let rm_event = make_pre_tool_event(
            "s1",
            "Bash",
            serde_json::json!({"command": "rm -rf /tmp/test"}),
        );
        let echo_event =
            make_pre_tool_event("s1", "Bash", serde_json::json!({"command": "echo hello"}));

        assert!(matcher.matches(&rm_event));
        assert!(!matcher.matches(&echo_event));
    }

    #[test]
    fn test_combined_matchers() {
        let matcher = HookMatcher::new().with_tool("Bash").with_command("rm");

        let bash_rm =
            make_pre_tool_event("s1", "Bash", serde_json::json!({"command": "rm file.txt"}));
        let bash_echo =
            make_pre_tool_event("s1", "Bash", serde_json::json!({"command": "echo hello"}));
        let read_event = make_pre_tool_event("s1", "Read", serde_json::json!({"path": "file.txt"}));

        assert!(matcher.matches(&bash_rm));
        assert!(!matcher.matches(&bash_echo)); // Bash but no rm
        assert!(!matcher.matches(&read_event)); // Not Bash
    }

    #[test]
    fn test_command_pattern_not_bash() {
        // Command pattern should only apply to Bash tool
        let matcher = HookMatcher::command("echo");

        let read_event = make_pre_tool_event("s1", "Read", serde_json::json!({"path": "echo.txt"}));

        assert!(!matcher.matches(&read_event));
    }

    #[test]
    fn test_builder_pattern() {
        let matcher = HookMatcher::tool("Write")
            .with_path("*.env")
            .with_session("secure-session");

        assert_eq!(matcher.tool, Some("Write".to_string()));
        assert_eq!(matcher.path_pattern, Some("*.env".to_string()));
        assert_eq!(matcher.session_id, Some("secure-session".to_string()));
    }

    #[test]
    fn test_matcher_serialization() {
        let matcher = HookMatcher::tool("Bash").with_command("rm.*");

        let json = serde_json::to_string(&matcher).unwrap();
        assert!(json.contains("Bash"));
        assert!(json.contains("rm.*"));

        let parsed: HookMatcher = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.tool, Some("Bash".to_string()));
        assert_eq!(parsed.command_pattern, Some("rm.*".to_string()));
    }

    #[test]
    fn test_path_with_alternative_field() {
        // Test that "path" field also works (not just "file_path")
        let matcher = HookMatcher::path("*.txt");

        let event = make_pre_tool_event("s1", "Read", serde_json::json!({"path": "readme.txt"}));

        assert!(matcher.matches(&event));
    }

    fn make_skill_load_event(skill_name: &str) -> HookEvent {
        HookEvent::SkillLoad(crate::hooks::events::SkillLoadEvent {
            skill_name: skill_name.to_string(),
            tool_names: vec!["tool1".to_string()],
            version: None,
            description: None,
            loaded_at: 0,
        })
    }

    fn make_skill_unload_event(skill_name: &str) -> HookEvent {
        HookEvent::SkillUnload(crate::hooks::events::SkillUnloadEvent {
            skill_name: skill_name.to_string(),
            tool_names: vec!["tool1".to_string()],
            duration_ms: 1000,
        })
    }

    #[test]
    fn test_skill_matcher() {
        let matcher = HookMatcher::skill("my-skill");

        let matching_event = make_skill_load_event("my-skill");
        let non_matching_event = make_skill_load_event("other-skill");

        assert!(matcher.matches(&matching_event));
        assert!(!matcher.matches(&non_matching_event));
    }

    #[test]
    fn test_skill_matcher_pattern() {
        // Test glob pattern matching for skill names
        let matcher = HookMatcher::skill("test-*");

        let test_skill = make_skill_load_event("test-skill");
        let test_other = make_skill_load_event("test-other");
        let no_match = make_skill_load_event("other-skill");

        assert!(matcher.matches(&test_skill));
        assert!(matcher.matches(&test_other));
        assert!(!matcher.matches(&no_match));
    }

    #[test]
    fn test_skill_matcher_unload_event() {
        let matcher = HookMatcher::skill("my-skill");

        let unload_event = make_skill_unload_event("my-skill");
        assert!(matcher.matches(&unload_event));

        let other_unload = make_skill_unload_event("other-skill");
        assert!(!matcher.matches(&other_unload));
    }

    #[test]
    fn test_skill_matcher_non_skill_event() {
        // Skill matcher should not match non-skill events
        let matcher = HookMatcher::skill("my-skill");

        let tool_event = make_pre_tool_event("s1", "Bash", serde_json::json!({}));
        assert!(!matcher.matches(&tool_event));
    }

    #[test]
    fn test_skill_matcher_with_builder() {
        let matcher = HookMatcher::new().with_skill("test-*");

        assert_eq!(matcher.skill, Some("test-*".to_string()));

        let event = make_skill_load_event("test-skill");
        assert!(matcher.matches(&event));
    }

    #[test]
    fn test_skill_matcher_serialization() {
        let matcher = HookMatcher::skill("my-skill");

        let json = serde_json::to_string(&matcher).unwrap();
        assert!(json.contains("my-skill"));
        assert!(json.contains("skill"));

        let parsed: HookMatcher = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.skill, Some("my-skill".to_string()));
    }
}
