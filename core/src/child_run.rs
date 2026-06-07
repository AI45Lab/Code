//! Parent-to-child capability inheritance for delegated task runs.
//!
//! When a parent session delegates work via `task`/`parallel_task`, certain
//! capabilities should propagate to the child run. This module makes that
//! contract explicit and documented.
//!
//! ## Inheritance Rules
//!
//! | Capability              | Inherits? | Rationale                                    |
//! |-------------------------|-----------|----------------------------------------------|
//! | security_provider       | Yes       | Taint tracking must be consistent             |
//! | hook_engine             | Yes       | Parent hooks observe child tool calls          |
//! | skill_registry          | Yes       | Skills are workspace-scoped                    |
//! | tool_timeout_ms         | Yes       | Safety limits should propagate                 |
//! | max_parallel_tasks      | Yes       | Parent fan-out limits should constrain children |
//! | max_execution_time_ms   | Yes       | Prevents runaway child runs                    |
//! | circuit_breaker_threshold | Yes     | LLM failure handling should be consistent      |
//! | confirmation_manager    | Depends   | Governed by ConfirmationInheritance            |
//! | workspace_services      | Yes       | Child tools must operate on the same workspace |
//! | budget_guard            | Yes       | One shared cost ledger spans the whole fan-out |
//! | memory                  | No        | Child has isolated context                     |
//! | queue_config            | No        | Child runs are synchronous within parent       |
//! | planning_mode           | No        | Child tasks are pre-planned by parent          |
//! | context_providers       | No        | Child has its own prompt context                |

use crate::agent::AgentConfig;
use crate::hitl::ConfirmationProvider;
use crate::hooks::HookExecutor;
use crate::security::SecurityProvider;
use crate::skills::SkillRegistry;
use std::sync::Arc;

/// Capabilities inherited from a parent session into a child run.
///
/// Fields that are `Some` are inherited from the parent. Fields that are `None`
/// use the child's own defaults (set by `AgentDefinition::apply_to`).
pub struct ChildRunContext {
    pub security_provider: Option<Arc<dyn SecurityProvider>>,
    pub hook_engine: Option<Arc<dyn HookExecutor>>,
    pub skill_registry: Option<Arc<SkillRegistry>>,
    pub tool_timeout_ms: Option<u64>,
    pub max_parallel_tasks: Option<usize>,
    pub max_execution_time_ms: Option<u64>,
    pub circuit_breaker_threshold: Option<u32>,
    pub confirmation_manager: Option<Arc<dyn ConfirmationProvider>>,
    pub workspace_services: Option<Arc<crate::workspace::WorkspaceServices>>,
    /// Shared budget/quota guard. When inherited, every child run feeds the same
    /// guard, so a single ledger can cap an entire delegated fan-out / workflow
    /// rather than each child counting independently.
    pub budget_guard: Option<Arc<dyn crate::budget::BudgetGuard>>,
}

impl ChildRunContext {
    /// Apply inherited capabilities to a child AgentConfig.
    ///
    /// Called after `AgentDefinition::apply_to()` so that parent capabilities
    /// fill remaining gaps without overriding agent-specific settings.
    pub(crate) fn apply_to(&self, config: &mut AgentConfig) {
        if config.security_provider.is_none() {
            config.security_provider = self.security_provider.clone();
        }
        if config.hook_engine.is_none() {
            config.hook_engine = self.hook_engine.clone();
        }
        if config.skill_registry.is_none() {
            config.skill_registry = self.skill_registry.clone();
        }
        if config.tool_timeout_ms.is_none() {
            config.tool_timeout_ms = self.tool_timeout_ms;
        }
        if let Some(max_parallel_tasks) = self.max_parallel_tasks {
            config.max_parallel_tasks = max_parallel_tasks.max(1);
        }
        if config.max_execution_time_ms.is_none() {
            config.max_execution_time_ms = self.max_execution_time_ms;
        }
        if let Some(threshold) = self.circuit_breaker_threshold {
            config.circuit_breaker_threshold = threshold;
        }
        if config.confirmation_manager.is_none() {
            config.confirmation_manager = self.confirmation_manager.clone();
        }
        if config.budget_guard.is_none() {
            config.budget_guard = self.budget_guard.clone();
        }
    }
}
