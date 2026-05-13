use serde::{Deserialize, Serialize};

use super::{MatchingRules, PermissionChecker, PermissionDecision, PermissionRule};

/// Permission policy configuration
///
/// Evaluation order:
/// 1. Deny rules - any match results in denial
/// 2. Allow rules - any match results in auto-approval
/// 3. Ask rules - any match requires user confirmation
/// 4. Default - falls back to default_decision
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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

    /// Return true when this policy is still the implicit default Ask policy.
    pub fn is_default_policy(&self) -> bool {
        self == &Self::default()
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
