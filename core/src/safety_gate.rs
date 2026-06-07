//! Tool-use safety gate.
//!
//! This module concentrates the pre-execution decision chain for tool calls:
//! skill restrictions, hook blocks, permission policy, and HITL requirements.

use crate::agent::AgentConfig;
use crate::hitl::TimeoutAction;
use crate::permissions::PermissionDecision;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolGateApproval {
    PermissionAllow,
    ConfirmationNotRequired,
}

impl ToolGateApproval {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            ToolGateApproval::PermissionAllow => "permission_allow",
            ToolGateApproval::ConfirmationNotRequired => "confirmation_not_required",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolGateDenial {
    SkillRestriction,
    HookBlock,
    PermissionDeny,
    MissingConfirmationManager,
}

impl ToolGateDenial {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            ToolGateDenial::SkillRestriction => "skill_restriction",
            ToolGateDenial::HookBlock => "hook_block",
            ToolGateDenial::PermissionDeny => "permission_deny",
            ToolGateDenial::MissingConfirmationManager => "missing_confirmation_manager",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ToolGateDecision {
    Execute {
        reason: ToolGateApproval,
    },
    Confirm {
        timeout_ms: u64,
        timeout_action: TimeoutAction,
    },
    Deny {
        output: String,
        event_reason: String,
        reason: ToolGateDenial,
    },
}

pub(crate) struct ToolGateInput<'a> {
    pub(crate) tool_name: &'a str,
    pub(crate) args: &'a serde_json::Value,
    pub(crate) pre_tool_block: Option<String>,
}

pub(crate) struct ToolSafetyGate<'a> {
    config: &'a AgentConfig,
}

impl<'a> ToolSafetyGate<'a> {
    pub(crate) fn new(config: &'a AgentConfig) -> Self {
        Self { config }
    }

    pub(crate) async fn decide(&self, input: ToolGateInput<'_>) -> ToolGateDecision {
        if let Some(decision) = self.check_skill_restrictions(input.tool_name) {
            return decision;
        }

        if let Some(reason) = input.pre_tool_block {
            return ToolGateDecision::Deny {
                output: format!("Tool '{}' blocked by hook: {}", input.tool_name, reason),
                event_reason: reason,
                reason: ToolGateDenial::HookBlock,
            };
        }

        match self.permission_decision(input.tool_name, input.args) {
            PermissionDecision::Deny => ToolGateDecision::Deny {
                output: format!(
                    "Permission denied: Tool '{}' is blocked by permission policy.",
                    input.tool_name
                ),
                event_reason: "Blocked by deny rule in permission policy".to_string(),
                reason: ToolGateDenial::PermissionDeny,
            },
            PermissionDecision::Allow => ToolGateDecision::Execute {
                reason: ToolGateApproval::PermissionAllow,
            },
            PermissionDecision::Ask => self.confirmation_decision(input.tool_name).await,
        }
    }

    pub(crate) fn check_skill_restrictions(&self, tool_name: &str) -> Option<ToolGateDecision> {
        let registry = self.config.skill_registry.as_ref()?;
        let restricting_skills = registry.global_tool_restricting_skills();
        if restricting_skills.is_empty() {
            return None;
        }

        let allowed = restricting_skills
            .iter()
            .any(|skill| skill.is_tool_allowed(tool_name));
        if allowed {
            return None;
        }

        let msg = format!("Tool '{}' is not allowed by any active skill.", tool_name);
        Some(ToolGateDecision::Deny {
            output: msg.clone(),
            event_reason: msg,
            reason: ToolGateDenial::SkillRestriction,
        })
    }

    pub(crate) fn permission_decision(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> PermissionDecision {
        self.config
            .permission_checker
            .as_ref()
            .map(|checker| checker.check(tool_name, args))
            .unwrap_or(PermissionDecision::Ask)
    }

    async fn confirmation_decision(&self, tool_name: &str) -> ToolGateDecision {
        let Some(cm) = &self.config.confirmation_manager else {
            let msg = format!(
                "Tool '{}' requires confirmation but no HITL confirmation manager is configured. \
                 Configure a confirmation policy to enable tool execution.",
                tool_name
            );
            return ToolGateDecision::Deny {
                output: msg.clone(),
                event_reason: msg,
                reason: ToolGateDenial::MissingConfirmationManager,
            };
        };

        if !cm.requires_confirmation(tool_name).await {
            return ToolGateDecision::Execute {
                reason: ToolGateApproval::ConfirmationNotRequired,
            };
        }

        let policy = cm.policy().await;
        ToolGateDecision::Confirm {
            timeout_ms: policy.default_timeout_ms,
            timeout_action: policy.timeout_action,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hitl::{ConfirmationManager, ConfirmationPolicy};
    use crate::permissions::{PermissionChecker, PermissionDecision};
    use crate::queue::SessionLane;
    use crate::skills::{Skill, SkillKind, SkillRegistry};
    use serde_json::json;
    use std::sync::Arc;
    use tokio::sync::broadcast;

    struct StaticPermission(PermissionDecision);

    impl PermissionChecker for StaticPermission {
        fn check(&self, _tool_name: &str, _args: &serde_json::Value) -> PermissionDecision {
            self.0
        }
    }

    fn restricted_registry() -> Arc<SkillRegistry> {
        let registry = SkillRegistry::new();
        registry.register_unchecked(Arc::new(Skill {
            name: "read-only".to_string(),
            description: String::new(),
            allowed_tools: Some("read(*), grep(*)".to_string()),
            disable_model_invocation: false,
            kind: SkillKind::Instruction,
            content: String::new(),
            tags: Vec::new(),
            version: None,
        }));
        Arc::new(registry)
    }

    #[tokio::test]
    async fn skill_restriction_denies_before_permission_allow() {
        let config = AgentConfig {
            skill_registry: Some(restricted_registry()),
            permission_checker: Some(Arc::new(StaticPermission(PermissionDecision::Allow))),
            ..Default::default()
        };
        let gate = ToolSafetyGate::new(&config);

        let decision = gate
            .decide(ToolGateInput {
                tool_name: "write",
                args: &json!({"file_path": "x"}),
                pre_tool_block: None,
            })
            .await;

        assert!(matches!(
            decision,
            ToolGateDecision::Deny {
                reason: ToolGateDenial::SkillRestriction,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn builtin_skill_permissions_do_not_restrict_default_session_tools() {
        let config = AgentConfig {
            skill_registry: Some(Arc::new(SkillRegistry::with_builtins())),
            permission_checker: Some(Arc::new(StaticPermission(PermissionDecision::Allow))),
            ..Default::default()
        };
        let gate = ToolSafetyGate::new(&config);

        let decision = gate
            .decide(ToolGateInput {
                tool_name: "write",
                args: &json!({"file_path": "x"}),
                pre_tool_block: None,
            })
            .await;

        assert!(matches!(
            decision,
            ToolGateDecision::Execute {
                reason: ToolGateApproval::PermissionAllow,
            }
        ));
    }

    #[tokio::test]
    async fn hook_block_denies_before_permission_allow() {
        let config = AgentConfig {
            skill_registry: None,
            permission_checker: Some(Arc::new(StaticPermission(PermissionDecision::Allow))),
            ..Default::default()
        };
        let gate = ToolSafetyGate::new(&config);

        let decision = gate
            .decide(ToolGateInput {
                tool_name: "bash",
                args: &json!({"command": "echo ok"}),
                pre_tool_block: Some("blocked by policy".to_string()),
            })
            .await;

        assert!(matches!(
            decision,
            ToolGateDecision::Deny {
                reason: ToolGateDenial::HookBlock,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn ask_without_confirmation_manager_is_safe_deny() {
        let config = AgentConfig {
            skill_registry: None,
            permission_checker: None,
            confirmation_manager: None,
            ..Default::default()
        };
        let gate = ToolSafetyGate::new(&config);

        let decision = gate
            .decide(ToolGateInput {
                tool_name: "bash",
                args: &json!({"command": "echo ok"}),
                pre_tool_block: None,
            })
            .await;

        assert!(matches!(
            decision,
            ToolGateDecision::Deny {
                reason: ToolGateDenial::MissingConfirmationManager,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn ask_with_confirmation_manager_requests_confirmation() {
        let (event_tx, _) = broadcast::channel(8);
        let manager = Arc::new(ConfirmationManager::new(
            ConfirmationPolicy::enabled().with_timeout(1234, crate::hitl::TimeoutAction::Reject),
            event_tx,
        ));
        let config = AgentConfig {
            skill_registry: None,
            confirmation_manager: Some(manager),
            ..Default::default()
        };
        let gate = ToolSafetyGate::new(&config);

        let decision = gate
            .decide(ToolGateInput {
                tool_name: "bash",
                args: &json!({"command": "echo ok"}),
                pre_tool_block: None,
            })
            .await;

        assert_eq!(
            decision,
            ToolGateDecision::Confirm {
                timeout_ms: 1234,
                timeout_action: crate::hitl::TimeoutAction::Reject,
            }
        );
    }

    #[tokio::test]
    async fn yolo_lane_executes_without_confirmation() {
        let (event_tx, _) = broadcast::channel(8);
        let manager = Arc::new(ConfirmationManager::new(
            ConfirmationPolicy::enabled().with_yolo_lanes([SessionLane::Query]),
            event_tx,
        ));
        let config = AgentConfig {
            skill_registry: None,
            confirmation_manager: Some(manager),
            ..Default::default()
        };
        let gate = ToolSafetyGate::new(&config);

        let decision = gate
            .decide(ToolGateInput {
                tool_name: "read",
                args: &json!({"file_path": "README.md"}),
                pre_tool_block: None,
            })
            .await;

        assert_eq!(
            decision,
            ToolGateDecision::Execute {
                reason: ToolGateApproval::ConfirmationNotRequired,
            }
        );
    }
}
