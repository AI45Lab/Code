//! Subagent System
//!
//! Provides a system for delegating specialized tasks to focused child agents.
//! Each subagent runs in an isolated child session with restricted permissions.
//!
//! ## Architecture
//!
//! ```text
//! Parent Session
//!   └── Task Tool
//!         ├── AgentRegistry (lookup agent definitions)
//!         └── Child Session (isolated execution)
//!               ├── Restricted permissions
//!               ├── Optional model override
//!               └── Event forwarding to parent
//! ```
//!
//! ## Built-in Agents
//!
//! - `explore`: Fast codebase exploration (read-only)
//! - `general`: Multi-step task execution
//! - `plan`: Read-only planning mode
//! - `title`: Session title generation (hidden)
//! - `summary`: Session summarization (hidden)
//!
//! ## Loading Agents from Files
//!
//! Agents can be loaded from YAML or Markdown files:
//!
//! ### YAML Format
//! ```yaml
//! name: my-agent
//! description: Custom agent for specific tasks
//! mode: subagent
//! hidden: false
//! max_steps: 30
//! permissions:
//!   allow:
//!     - read
//!     - grep
//!   deny:
//!     - write
//! prompt: |
//!   You are a specialized agent...
//! ```
//!
//! ### Markdown Format
//! ```markdown
//! ---
//! name: my-agent
//! description: Custom agent
//! mode: subagent
//! max_steps: 30
//! ---
//! # System Prompt
//! You are a specialized agent...
//! ```

use crate::config::CodeConfig;
use crate::permissions::PermissionPolicy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;

use crate::error::{read_or_recover, write_or_recover};

/// Agent execution mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AgentMode {
    /// Primary agent (main conversation)
    #[default]
    Primary,
    /// Subagent (child session for delegated tasks)
    Subagent,
}

/// Model configuration for agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model identifier (e.g., "claude-3-5-sonnet-20241022")
    pub model: String,
    /// Optional provider override
    pub provider: Option<String>,
}

/// Agent definition
///
/// Defines the configuration and capabilities of an agent type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDefinition {
    /// Agent identifier (e.g., "explore", "plan", "general")
    pub name: String,
    /// Description of what the agent does
    pub description: String,
    /// Agent mode: "subagent" or "primary"
    #[serde(default)]
    pub mode: AgentMode,
    /// Whether this is a built-in agent
    #[serde(default)]
    pub native: bool,
    /// Whether to hide from UI
    #[serde(default)]
    pub hidden: bool,
    /// Permission rules for this agent
    #[serde(default)]
    pub permissions: PermissionPolicy,
    /// Optional model override
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelConfig>,
    /// System prompt for this agent
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Maximum execution steps (tool rounds)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_steps: Option<usize>,
    /// Whether this agent can spawn subagents (default: false)
    #[serde(default)]
    pub can_spawn_subagents: bool,
}

impl AgentDefinition {
    /// Create a new agent definition
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            mode: AgentMode::Subagent,
            native: false,
            hidden: false,
            permissions: PermissionPolicy::default(),
            model: None,
            prompt: None,
            max_steps: None,
            can_spawn_subagents: false,
        }
    }

    /// Set agent mode
    pub fn with_mode(mut self, mode: AgentMode) -> Self {
        self.mode = mode;
        self
    }

    /// Mark as native (built-in)
    pub fn native(mut self) -> Self {
        self.native = true;
        self
    }

    /// Mark as hidden from UI
    pub fn hidden(mut self) -> Self {
        self.hidden = true;
        self
    }

    /// Set permission policy
    pub fn with_permissions(mut self, permissions: PermissionPolicy) -> Self {
        self.permissions = permissions;
        self
    }

    /// Set model override
    pub fn with_model(mut self, model: ModelConfig) -> Self {
        self.model = Some(model);
        self
    }

    /// Set system prompt
    pub fn with_prompt(mut self, prompt: &str) -> Self {
        self.prompt = Some(prompt.to_string());
        self
    }

    /// Set maximum execution steps
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    /// Allow spawning subagents
    pub fn allow_subagents(mut self) -> Self {
        self.can_spawn_subagents = true;
        self
    }
}

/// Agent registry for managing agent definitions
///
/// Thread-safe registry that stores agent definitions and provides
/// lookup functionality.
pub struct AgentRegistry {
    agents: RwLock<HashMap<String, AgentDefinition>>,
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentRegistry {
    /// Create a new agent registry with built-in agents
    pub fn new() -> Self {
        let registry = Self {
            agents: RwLock::new(HashMap::new()),
        };

        // Register built-in agents
        for agent in builtin_agents() {
            registry.register(agent);
        }

        registry
    }

    /// Create a new agent registry with configuration
    ///
    /// Loads built-in agents first, then loads agents from configured directories.
    pub fn with_config(config: &CodeConfig) -> Self {
        let registry = Self::new();

        // Load agents from configured directories
        for dir in &config.agent_dirs {
            let agents = load_agents_from_dir(dir);
            for agent in agents {
                tracing::info!("Loaded agent '{}' from {}", agent.name, dir.display());
                registry.register(agent);
            }
        }

        registry
    }

    /// Register an agent definition
    pub fn register(&self, agent: AgentDefinition) {
        let mut agents = write_or_recover(&self.agents);
        tracing::debug!("Registering agent: {}", agent.name);
        agents.insert(agent.name.clone(), agent);
    }

    /// Unregister an agent by name
    ///
    /// Returns true if the agent was removed, false if not found.
    pub fn unregister(&self, name: &str) -> bool {
        let mut agents = write_or_recover(&self.agents);
        agents.remove(name).is_some()
    }

    /// Get an agent definition by name
    pub fn get(&self, name: &str) -> Option<AgentDefinition> {
        let agents = read_or_recover(&self.agents);
        agents.get(name).cloned()
    }

    /// List all registered agents
    pub fn list(&self) -> Vec<AgentDefinition> {
        let agents = read_or_recover(&self.agents);
        agents.values().cloned().collect()
    }

    /// List visible agents (not hidden)
    pub fn list_visible(&self) -> Vec<AgentDefinition> {
        let agents = read_or_recover(&self.agents);
        agents.values().filter(|a| !a.hidden).cloned().collect()
    }

    /// Check if an agent exists
    pub fn exists(&self, name: &str) -> bool {
        let agents = read_or_recover(&self.agents);
        agents.contains_key(name)
    }

    /// Get the number of registered agents
    pub fn len(&self) -> usize {
        let agents = read_or_recover(&self.agents);
        agents.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ============================================================================
// Agent File Loading
// ============================================================================

/// Parse an agent definition from YAML content
///
/// The YAML should contain fields matching AgentDefinition structure.
pub fn parse_agent_yaml(content: &str) -> anyhow::Result<AgentDefinition> {
    let agent: AgentDefinition = serde_yaml::from_str(content)
        .map_err(|e| anyhow::anyhow!("Failed to parse agent YAML: {}", e))?;

    if agent.name.is_empty() {
        return Err(anyhow::anyhow!("Agent name is required"));
    }

    Ok(agent)
}

/// Parse an agent definition from Markdown with YAML frontmatter
///
/// The frontmatter contains agent metadata, and the body becomes the prompt.
pub fn parse_agent_md(content: &str) -> anyhow::Result<AgentDefinition> {
    // Parse frontmatter (YAML between --- markers)
    let parts: Vec<&str> = content.splitn(3, "---").collect();

    if parts.len() < 3 {
        return Err(anyhow::anyhow!(
            "Invalid markdown format: missing YAML frontmatter"
        ));
    }

    let frontmatter = parts[1].trim();
    let body = parts[2].trim();

    // Parse the frontmatter as YAML
    let mut agent: AgentDefinition = serde_yaml::from_str(frontmatter)
        .map_err(|e| anyhow::anyhow!("Failed to parse agent frontmatter: {}", e))?;

    if agent.name.is_empty() {
        return Err(anyhow::anyhow!("Agent name is required"));
    }

    // Use body as prompt if not already set in frontmatter
    if agent.prompt.is_none() && !body.is_empty() {
        agent.prompt = Some(body.to_string());
    }

    Ok(agent)
}

/// Load all agent definitions from a directory
///
/// Scans for *.yaml and *.md files and parses them as agent definitions.
/// Invalid files are logged and skipped.
pub fn load_agents_from_dir(dir: &Path) -> Vec<AgentDefinition> {
    let mut agents = Vec::new();

    let Ok(entries) = std::fs::read_dir(dir) else {
        tracing::warn!("Failed to read agent directory: {}", dir.display());
        return agents;
    };

    for entry in entries.flatten() {
        let path = entry.path();

        // Skip non-files
        if !path.is_file() {
            continue;
        }

        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };

        // Read file content
        let Ok(content) = std::fs::read_to_string(&path) else {
            tracing::warn!("Failed to read agent file: {}", path.display());
            continue;
        };

        // Parse based on extension
        let result = match ext {
            "yaml" | "yml" => parse_agent_yaml(&content),
            "md" => parse_agent_md(&content),
            _ => continue,
        };

        match result {
            Ok(agent) => {
                tracing::debug!("Loaded agent '{}' from {}", agent.name, path.display());
                agents.push(agent);
            }
            Err(e) => {
                tracing::warn!("Failed to parse agent file {}: {}", path.display(), e);
            }
        }
    }

    agents
}

/// Create built-in agent definitions
pub fn builtin_agents() -> Vec<AgentDefinition> {
    vec![
        // Explore agent: Fast codebase exploration (read-only)
        AgentDefinition::new(
            "explore",
            "Fast codebase exploration agent. Use for searching files, reading code, \
             and understanding codebase structure. Read-only operations only.",
        )
        .native()
        .with_permissions(explore_permissions())
        .with_max_steps(20)
        .with_prompt(EXPLORE_PROMPT),
        // General agent: Multi-step task execution
        AgentDefinition::new(
            "general",
            "General-purpose agent for multi-step task execution. Can read, write, \
             and execute commands. Cannot spawn subagents.",
        )
        .native()
        .with_permissions(general_permissions())
        .with_max_steps(50),
        // Plan agent: Read-only planning mode
        AgentDefinition::new(
            "plan",
            "Planning agent for designing implementation approaches. Read-only access \
             to explore codebase and create plans.",
        )
        .native()
        .with_mode(AgentMode::Primary)
        .with_permissions(plan_permissions())
        .with_max_steps(30)
        .with_prompt(PLAN_PROMPT),
        // Title agent: Session title generation (hidden)
        AgentDefinition::new(
            "title",
            "Generate a concise title for the session based on conversation content.",
        )
        .native()
        .hidden()
        .with_mode(AgentMode::Primary)
        .with_permissions(PermissionPolicy::new())
        .with_max_steps(1)
        .with_prompt(TITLE_PROMPT),
        // Summary agent: Session summarization (hidden)
        AgentDefinition::new(
            "summary",
            "Summarize the session conversation for context compaction.",
        )
        .native()
        .hidden()
        .with_mode(AgentMode::Primary)
        .with_permissions(summary_permissions())
        .with_max_steps(5)
        .with_prompt(SUMMARY_PROMPT),
    ]
}

// ============================================================================
// Permission Policies for Built-in Agents
// ============================================================================

/// Permission policy for explore agent (read-only)
fn explore_permissions() -> PermissionPolicy {
    PermissionPolicy::new()
        .allow_all(&["read", "grep", "glob", "ls"])
        .deny_all(&["write", "edit", "task"])
        .allow("Bash(ls:*)")
        .allow("Bash(cat:*)")
        .allow("Bash(head:*)")
        .allow("Bash(tail:*)")
        .allow("Bash(find:*)")
        .allow("Bash(wc:*)")
        .deny("Bash(rm:*)")
        .deny("Bash(mv:*)")
        .deny("Bash(cp:*)")
}

/// Permission policy for general agent (full access except task)
fn general_permissions() -> PermissionPolicy {
    PermissionPolicy::new()
        .allow_all(&["read", "write", "edit", "grep", "glob", "ls", "bash"])
        .deny("task")
}

/// Permission policy for plan agent (read-only)
fn plan_permissions() -> PermissionPolicy {
    PermissionPolicy::new()
        .allow_all(&["read", "grep", "glob", "ls"])
        .deny_all(&["write", "edit", "bash", "task"])
}

/// Permission policy for summary agent (read-only)
fn summary_permissions() -> PermissionPolicy {
    PermissionPolicy::new()
        .allow("read")
        .deny_all(&["write", "edit", "bash", "grep", "glob", "ls", "task"])
}

// ============================================================================
// System Prompts for Built-in Agents
// ============================================================================

const EXPLORE_PROMPT: &str = crate::prompts::SUBAGENT_EXPLORE;

const PLAN_PROMPT: &str = crate::prompts::SUBAGENT_PLAN;

const TITLE_PROMPT: &str = crate::prompts::SUBAGENT_TITLE;

const SUMMARY_PROMPT: &str = crate::prompts::SUBAGENT_SUMMARY;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_definition_builder() {
        let agent = AgentDefinition::new("test", "Test agent")
            .native()
            .hidden()
            .with_max_steps(10);

        assert_eq!(agent.name, "test");
        assert_eq!(agent.description, "Test agent");
        assert!(agent.native);
        assert!(agent.hidden);
        assert_eq!(agent.max_steps, Some(10));
        assert!(!agent.can_spawn_subagents);
    }

    #[test]
    fn test_agent_registry_new() {
        let registry = AgentRegistry::new();

        // Should have built-in agents
        assert!(registry.exists("explore"));
        assert!(registry.exists("general"));
        assert!(registry.exists("plan"));
        assert!(registry.exists("title"));
        assert!(registry.exists("summary"));
        assert_eq!(registry.len(), 5);
    }

    #[test]
    fn test_agent_registry_get() {
        let registry = AgentRegistry::new();

        let explore = registry.get("explore").unwrap();
        assert_eq!(explore.name, "explore");
        assert!(explore.native);
        assert!(!explore.hidden);

        let title = registry.get("title").unwrap();
        assert!(title.hidden);

        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_agent_registry_register_unregister() {
        let registry = AgentRegistry::new();
        let initial_count = registry.len();

        // Register custom agent
        let custom = AgentDefinition::new("custom", "Custom agent");
        registry.register(custom);
        assert_eq!(registry.len(), initial_count + 1);
        assert!(registry.exists("custom"));

        // Unregister
        assert!(registry.unregister("custom"));
        assert_eq!(registry.len(), initial_count);
        assert!(!registry.exists("custom"));

        // Unregister non-existent
        assert!(!registry.unregister("nonexistent"));
    }

    #[test]
    fn test_agent_registry_list_visible() {
        let registry = AgentRegistry::new();

        let visible = registry.list_visible();
        let all = registry.list();

        // Hidden agents should not be in visible list
        assert!(visible.len() < all.len());
        assert!(visible.iter().all(|a| !a.hidden));
    }

    #[test]
    fn test_builtin_agents() {
        let agents = builtin_agents();

        // Check we have expected agents
        let names: Vec<&str> = agents.iter().map(|a| a.name.as_str()).collect();
        assert!(names.contains(&"explore"));
        assert!(names.contains(&"general"));
        assert!(names.contains(&"plan"));
        assert!(names.contains(&"title"));
        assert!(names.contains(&"summary"));

        // Check explore is read-only (has deny rules for write)
        let explore = agents.iter().find(|a| a.name == "explore").unwrap();
        assert!(!explore.permissions.deny.is_empty());

        // Check general cannot spawn subagents
        let general = agents.iter().find(|a| a.name == "general").unwrap();
        assert!(!general.can_spawn_subagents);
    }

    #[test]
    fn test_agent_mode_default() {
        let mode = AgentMode::default();
        assert_eq!(mode, AgentMode::Primary);
    }

    // ========================================================================
    // Agent File Loading Tests
    // ========================================================================

    #[test]
    fn test_parse_agent_yaml() {
        let yaml = r#"
name: test-agent
description: A test agent
mode: subagent
hidden: false
max_steps: 20
"#;
        let agent = parse_agent_yaml(yaml).unwrap();
        assert_eq!(agent.name, "test-agent");
        assert_eq!(agent.description, "A test agent");
        assert_eq!(agent.mode, AgentMode::Subagent);
        assert!(!agent.hidden);
        assert_eq!(agent.max_steps, Some(20));
    }

    #[test]
    fn test_parse_agent_yaml_with_permissions() {
        let yaml = r#"
name: restricted-agent
description: Agent with permissions
permissions:
  allow:
    - rule: read
    - rule: grep
  deny:
    - rule: write
"#;
        let agent = parse_agent_yaml(yaml).unwrap();
        assert_eq!(agent.name, "restricted-agent");
        assert_eq!(agent.permissions.allow.len(), 2);
        assert_eq!(agent.permissions.deny.len(), 1);
        // Verify that deserialized rules actually match (tool_name populated)
        assert!(agent.permissions.allow[0].matches("read", &serde_json::json!({})));
        assert!(agent.permissions.allow[1].matches("grep", &serde_json::json!({})));
        assert!(agent.permissions.deny[0].matches("write", &serde_json::json!({})));
    }

    #[test]
    fn test_parse_agent_yaml_with_plain_string_permissions() {
        // Users naturally write plain strings in allow/deny lists
        let yaml = r#"
name: plain-agent
description: Agent with plain string permissions
permissions:
  allow:
    - read
    - grep
    - "Bash(cargo:*)"
  deny:
    - write
"#;
        let agent = parse_agent_yaml(yaml).unwrap();
        assert_eq!(agent.name, "plain-agent");
        assert_eq!(agent.permissions.allow.len(), 3);
        assert_eq!(agent.permissions.deny.len(), 1);
        // Verify rules are functional
        assert!(agent.permissions.allow[0].matches("read", &serde_json::json!({})));
        assert!(agent.permissions.allow[1].matches("grep", &serde_json::json!({})));
        assert!(agent.permissions.allow[2].matches("Bash", &serde_json::json!({"command": "cargo build"})));
        assert!(agent.permissions.deny[0].matches("write", &serde_json::json!({})));
    }

    #[test]
    fn test_parse_agent_yaml_missing_name() {
        let yaml = r#"
description: Agent without name
"#;
        let result = parse_agent_yaml(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_agent_md() {
        let md = r#"---
name: md-agent
description: Agent from markdown
mode: subagent
max_steps: 15
---
# System Prompt

You are a helpful agent.
Do your best work.
"#;
        let agent = parse_agent_md(md).unwrap();
        assert_eq!(agent.name, "md-agent");
        assert_eq!(agent.description, "Agent from markdown");
        assert_eq!(agent.max_steps, Some(15));
        assert!(agent.prompt.is_some());
        assert!(agent.prompt.unwrap().contains("helpful agent"));
    }

    #[test]
    fn test_parse_agent_md_with_prompt_in_frontmatter() {
        let md = r#"---
name: prompt-agent
description: Agent with prompt in frontmatter
prompt: "Frontmatter prompt"
---
Body content that should be ignored
"#;
        let agent = parse_agent_md(md).unwrap();
        assert_eq!(agent.prompt.unwrap(), "Frontmatter prompt");
    }

    #[test]
    fn test_parse_agent_md_missing_frontmatter() {
        let md = "Just markdown without frontmatter";
        let result = parse_agent_md(md);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_agents_from_dir() {
        let temp_dir = tempfile::tempdir().unwrap();

        // Create a YAML agent file
        std::fs::write(
            temp_dir.path().join("agent1.yaml"),
            r#"
name: yaml-agent
description: Agent from YAML file
"#,
        )
        .unwrap();

        // Create a Markdown agent file
        std::fs::write(
            temp_dir.path().join("agent2.md"),
            r#"---
name: md-agent
description: Agent from Markdown file
---
System prompt here
"#,
        )
        .unwrap();

        // Create an invalid file (should be skipped)
        std::fs::write(temp_dir.path().join("invalid.yaml"), "not: valid: yaml: [").unwrap();

        // Create a non-agent file (should be skipped)
        std::fs::write(temp_dir.path().join("readme.txt"), "Just a text file").unwrap();

        let agents = load_agents_from_dir(temp_dir.path());
        assert_eq!(agents.len(), 2);

        let names: Vec<&str> = agents.iter().map(|a| a.name.as_str()).collect();
        assert!(names.contains(&"yaml-agent"));
        assert!(names.contains(&"md-agent"));
    }

    #[test]
    fn test_load_agents_from_nonexistent_dir() {
        let agents = load_agents_from_dir(std::path::Path::new("/nonexistent/dir"));
        assert!(agents.is_empty());
    }

    #[test]
    fn test_registry_with_config() {
        let temp_dir = tempfile::tempdir().unwrap();

        // Create an agent file
        std::fs::write(
            temp_dir.path().join("custom.yaml"),
            r#"
name: custom-agent
description: Custom agent from config
"#,
        )
        .unwrap();

        let config = CodeConfig::new().add_agent_dir(temp_dir.path());
        let registry = AgentRegistry::with_config(&config);

        // Should have built-in agents plus custom agent
        assert!(registry.exists("explore"));
        assert!(registry.exists("custom-agent"));
        assert_eq!(registry.len(), 6); // 5 built-in + 1 custom
    }

    #[test]
    fn test_agent_definition_with_model() {
        let model = ModelConfig {
            model: "claude-3-5-sonnet".to_string(),
            provider: Some("anthropic".to_string()),
        };
        let agent = AgentDefinition::new("test", "Test").with_model(model);
        assert!(agent.model.is_some());
        assert_eq!(agent.model.unwrap().provider, Some("anthropic".to_string()));
    }

    #[test]
    fn test_agent_definition_allow_subagents() {
        let agent = AgentDefinition::new("test", "Test").allow_subagents();
        assert!(agent.can_spawn_subagents);
    }

    #[test]
    fn test_agent_registry_default() {
        let registry = AgentRegistry::default();
        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 5);
    }

    #[test]
    fn test_agent_registry_is_empty() {
        let registry = AgentRegistry {
            agents: RwLock::new(HashMap::new()),
        };
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }
}
