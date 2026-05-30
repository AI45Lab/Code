//! Budget / cost / quota contract for cluster-grade hosts.
//!
//! The framework does not enforce budgets itself — it only defines the
//! decision points and emits structured events. The host
//! implements [`BudgetGuard`] with whatever backend it likes
//! (per-tenant counters in Redis, per-day USD caps in Postgres, etc.)
//! and plugs it into [`SessionOptions::with_budget_guard`].
//!
//! Decision points wired today:
//!
//! 1. **Before each LLM call** — [`BudgetGuard::check_before_llm`].
//!    A `Deny` aborts the call; a `SoftLimit` lets the call proceed but
//!    triggers an [`AgentEvent::BudgetThresholdHit`] so in-session
//!    policy (hooks, custom prompts) can react.
//! 2. **After each LLM call** — [`BudgetGuard::record_after_llm`].
//!    The host updates its running spend total with the actual usage.
//! 3. **Before each tool call** — [`BudgetGuard::check_before_tool`].
//!    Same decision shape; useful for capping expensive tools per
//!    tenant.
//!
//! The default trait methods are no-ops returning [`BudgetDecision::Allow`]
//! so existing code is unaffected until a host plugs in a real impl.
//!
//! See [`AgentEvent::BudgetThresholdHit`](crate::agent::AgentEvent::BudgetThresholdHit)
//! for the event vocabulary triggered by `SoftLimit`.

use crate::llm::TokenUsage;
use async_trait::async_trait;

/// Outcome of a budget check.
///
/// The framework treats this purely as a decision — it never inspects
/// the carried strings except to forward them to [`AgentEvent`]s and to
/// the eventual error.
///
/// [`AgentEvent`]: crate::agent::AgentEvent
#[derive(Debug, Clone)]
pub enum BudgetDecision {
    /// Operation proceeds normally. No event is emitted.
    Allow,
    /// Operation proceeds, but the framework emits a
    /// [`AgentEvent::BudgetThresholdHit { kind: "soft", .. }`]
    /// event before continuing. In-session hooks can react (e.g. trigger
    /// auto-compact, swap to a cheaper model on next turn).
    ///
    /// [`AgentEvent::BudgetThresholdHit { kind: "soft", .. }`]: crate::agent::AgentEvent::BudgetThresholdHit
    SoftLimit {
        /// Logical resource label ("llm_tokens", "usd_cost", "wall_time", ...).
        resource: String,
        /// Current consumed amount (units depend on `resource`).
        consumed: f64,
        /// Threshold that was crossed.
        limit: f64,
        /// Optional human-readable explanation for logs / UI.
        message: Option<String>,
    },
    /// Operation is refused. The framework returns
    /// [`CodeError::BudgetExhausted`](crate::error::CodeError::BudgetExhausted)
    /// from the LLM / tool entry point. The session itself stays open —
    /// callers can re-try later or after the host has re-allocated
    /// budget.
    Deny {
        /// Logical resource label that exhausted.
        resource: String,
        /// Human-readable reason surfaced in the error and in any
        /// emitted `BudgetThresholdHit { kind: "hard", .. }` event.
        reason: String,
    },
}

/// Host-supplied budget / quota contract.
///
/// Implementations are typically wired up by a cluster control plane
/// to enforce cross-session, cross-tenant cost limits. The framework
/// itself ships only the no-op [`NoopBudgetGuard`].
///
/// All trait methods default to `Allow` / no-op so impls only need to
/// override what they actually want to govern.
#[async_trait]
pub trait BudgetGuard: Send + Sync {
    /// Called immediately before an LLM API call.
    ///
    /// `estimated_prompt_tokens` is a best-effort framework estimate
    /// from the message history at call time; impls that want precise
    /// accounting should use [`record_after_llm`](Self::record_after_llm)
    /// instead of trusting the estimate.
    async fn check_before_llm(
        &self,
        session_id: &str,
        estimated_prompt_tokens: usize,
    ) -> BudgetDecision {
        let _ = (session_id, estimated_prompt_tokens);
        BudgetDecision::Allow
    }

    /// Called after every successful LLM call with the actual usage
    /// reported by the provider. Lets the impl keep its running spend
    /// total in sync with reality.
    ///
    /// Failed LLM calls do not invoke this hook.
    async fn record_after_llm(&self, session_id: &str, usage: &TokenUsage) {
        let _ = (session_id, usage);
    }

    /// Called immediately before a tool invocation. The framework does
    /// not pass tool arguments — impls that need argument-aware caps
    /// must wrap the executor via a custom `ToolExecutor`.
    async fn check_before_tool(&self, session_id: &str, tool_name: &str) -> BudgetDecision {
        let _ = (session_id, tool_name);
        BudgetDecision::Allow
    }
}

/// Default implementation that always allows everything. Used when no
/// host-supplied guard is configured.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopBudgetGuard;

#[async_trait]
impl BudgetGuard for NoopBudgetGuard {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn noop_allows_everything() {
        let guard = NoopBudgetGuard;
        assert!(matches!(
            guard.check_before_llm("s", 1000).await,
            BudgetDecision::Allow
        ));
        assert!(matches!(
            guard.check_before_tool("s", "bash").await,
            BudgetDecision::Allow
        ));
        // record is just observable side-effect; ensure it doesn't panic.
        guard.record_after_llm("s", &TokenUsage::default()).await;
    }

    #[derive(Debug, Default)]
    struct CountingGuard {
        llm_checks: AtomicUsize,
        records: AtomicUsize,
    }

    #[async_trait]
    impl BudgetGuard for CountingGuard {
        async fn check_before_llm(&self, _: &str, _: usize) -> BudgetDecision {
            self.llm_checks.fetch_add(1, Ordering::SeqCst);
            BudgetDecision::Deny {
                resource: "llm_tokens".to_string(),
                reason: "budget exhausted in test".to_string(),
            }
        }
        async fn record_after_llm(&self, _: &str, _: &TokenUsage) {
            self.records.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn custom_guard_can_deny() {
        let guard: Arc<dyn BudgetGuard> = Arc::new(CountingGuard::default());
        let decision = guard.check_before_llm("s", 100).await;
        match decision {
            BudgetDecision::Deny { resource, .. } => assert_eq!(resource, "llm_tokens"),
            other => panic!("expected Deny, got {other:?}"),
        }
    }
}
