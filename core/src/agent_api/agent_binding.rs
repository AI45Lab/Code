//! Agent definition to session option binding.
//!
//! This module owns the harness contract for launching a named/subagent-backed
//! session: definition-level permissions, model, step budget, and prompt are
//! applied without overwriting explicit host overrides.

use super::SessionOptions;
use crate::prompts::SystemPromptSlots;
use crate::subagent::AgentDefinition;
use std::sync::Arc;

pub(super) fn apply_agent_definition(
    mut opts: SessionOptions,
    def: &AgentDefinition,
) -> SessionOptions {
    apply_permissions(&mut opts, def);
    apply_step_budget(&mut opts, def);
    apply_model(&mut opts, def);
    apply_prompt(&mut opts, def);
    opts
}

fn apply_permissions(opts: &mut SessionOptions, def: &AgentDefinition) {
    if opts.permission_checker.is_none() && has_defined_permissions(def) {
        opts.permission_checker = Some(Arc::new(def.permissions.clone()));
    }
}

fn has_defined_permissions(def: &AgentDefinition) -> bool {
    !def.permissions.is_default_policy()
}

fn apply_step_budget(opts: &mut SessionOptions, def: &AgentDefinition) {
    if opts.max_tool_rounds.is_none() {
        opts.max_tool_rounds = def.max_steps;
    }
}

fn apply_model(opts: &mut SessionOptions, def: &AgentDefinition) {
    if opts.model.is_some() {
        return;
    }

    if let Some(model) = &def.model {
        let provider = model.provider.as_deref().unwrap_or("anthropic");
        opts.model = Some(format!("{}/{}", provider, model.model));
    }
}

fn apply_prompt(opts: &mut SessionOptions, def: &AgentDefinition) {
    let Some(prompt) = &def.prompt else {
        return;
    };

    let slots = opts
        .prompt_slots
        .get_or_insert_with(SystemPromptSlots::default);
    if slots.extra.is_none() {
        slots.extra = Some(prompt.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::permissions::PermissionPolicy;
    use crate::subagent::{AgentDefinition, ModelConfig};

    #[test]
    fn applies_agent_definition_defaults_without_host_overrides() {
        let permissions = PermissionPolicy::new().allow("read(*)").deny("bash(rm:*)");
        let def = AgentDefinition::new("reviewer", "Review code")
            .with_permissions(permissions)
            .with_model(ModelConfig {
                provider: Some("openai".to_string()),
                model: "gpt-4.1".to_string(),
            })
            .with_prompt("Focus on correctness.")
            .with_max_steps(7);

        let opts = apply_agent_definition(SessionOptions::new(), &def);

        assert!(opts.permission_checker.is_some());
        assert_eq!(opts.model.as_deref(), Some("openai/gpt-4.1"));
        assert_eq!(opts.max_tool_rounds, Some(7));
        assert_eq!(
            opts.prompt_slots
                .as_ref()
                .and_then(|slots| slots.extra.as_deref()),
            Some("Focus on correctness.")
        );
    }

    #[test]
    fn preserves_explicit_host_overrides() {
        let def = AgentDefinition::new("planner", "Plan work")
            .with_model(ModelConfig {
                provider: Some("anthropic".to_string()),
                model: "claude-sonnet".to_string(),
            })
            .with_prompt("Definition prompt.")
            .with_max_steps(3);
        let opts = SessionOptions::new()
            .with_model("openai/gpt-4o")
            .with_max_tool_rounds(12)
            .with_prompt_slots(SystemPromptSlots {
                role: Some("Host role".to_string()),
                extra: Some("Host prompt.".to_string()),
                ..SystemPromptSlots::default()
            });

        let opts = apply_agent_definition(opts, &def);

        assert_eq!(opts.model.as_deref(), Some("openai/gpt-4o"));
        assert_eq!(opts.max_tool_rounds, Some(12));
        let slots = opts.prompt_slots.unwrap();
        assert_eq!(slots.role.as_deref(), Some("Host role"));
        assert_eq!(slots.extra.as_deref(), Some("Host prompt."));
    }

    #[test]
    fn defaults_model_provider_when_definition_omits_one() {
        let def = AgentDefinition::new("general", "General work").with_model(ModelConfig {
            provider: None,
            model: "claude-opus".to_string(),
        });

        let opts = apply_agent_definition(SessionOptions::new(), &def);

        assert_eq!(opts.model.as_deref(), Some("anthropic/claude-opus"));
    }

    #[test]
    fn applies_default_allow_agent_permissions() {
        use crate::permissions::PermissionDecision;

        let permissions = PermissionPolicy {
            default_decision: PermissionDecision::Allow,
            ..PermissionPolicy::new()
        };
        let def = AgentDefinition::new("worker", "Allowed worker").with_permissions(permissions);

        let opts = apply_agent_definition(SessionOptions::new(), &def);

        let checker = opts
            .permission_checker
            .expect("non-default permission policy should be applied");
        assert_eq!(
            checker.check("bash", &serde_json::json!({"command": "echo ok"})),
            PermissionDecision::Allow
        );
    }
}
