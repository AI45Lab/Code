//! A workflow-scoped, aggregating [`BudgetGuard`].
//!
//! [`BudgetGuard`] alone counts per child-loop (keyed by `session_id`); there is
//! no single ledger spanning a fan-out. [`WorkflowBudget`] closes that gap: it
//! wraps an optional inner guard, accumulates token spend from **every** child
//! step into one shared atomic counter, and refuses further LLM calls once an
//! optional hard `limit_tokens` is reached.
//!
//! It implements [`BudgetGuard`] itself, so it installs through the unchanged
//! seam — set it as the child runs' `budget_guard` (via the
//! [`ChildRunContext`](crate::child_run::ChildRunContext) the executor applies)
//! and the existing per-turn `check_before_llm` / `record_after_llm` decision
//! points in the agent loop feed this one ledger automatically.
//!
//! ## Honest limit
//!
//! `record_after_llm` runs *after* a call returns, while `check_before_llm` runs
//! before. Under a wide [`parallel`](super::Workflow::parallel) fan-out several
//! in-flight turns can therefore race a few calls past a hard cap before the
//! ledger catches up. This is a **soft cost ceiling**, not a hard per-token
//! guarantee — the same race the per-session [`BudgetGuard`] already documents.
//! The framework never force-kills a running fan-out; an exhausted budget
//! surfaces as a `Deny` on the *next* `check_before_llm`, which the agent loop
//! turns into a failed step the host can react to.

use crate::budget::{BudgetDecision, BudgetGuard};
use crate::llm::TokenUsage;
use async_trait::async_trait;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// An immutable view of a [`WorkflowBudget`] ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetSnapshot {
    /// Total tokens recorded across every child step so far.
    pub consumed_tokens: u64,
    /// The hard ceiling, if one was configured.
    pub limit_tokens: Option<u64>,
}

impl BudgetSnapshot {
    /// Tokens left before the cap, saturating at 0. `None` when uncapped.
    pub fn remaining_tokens(&self) -> Option<u64> {
        self.limit_tokens
            .map(|limit| limit.saturating_sub(self.consumed_tokens))
    }

    /// Whether a configured cap has been reached.
    pub fn is_exhausted(&self) -> bool {
        matches!(self.limit_tokens, Some(limit) if self.consumed_tokens >= limit)
    }
}

/// A shared, workflow-scoped token ledger that also acts as a [`BudgetGuard`].
pub struct WorkflowBudget {
    inner: Option<Arc<dyn BudgetGuard>>,
    consumed_tokens: AtomicU64,
    limit_tokens: Option<u64>,
}

impl WorkflowBudget {
    /// A ledger with an optional hard token ceiling and no inner guard.
    pub fn new(limit_tokens: Option<u64>) -> Self {
        Self {
            inner: None,
            consumed_tokens: AtomicU64::new(0),
            limit_tokens,
        }
    }

    /// Delegate `check`/`record` to `inner` in addition to maintaining the
    /// shared ledger — lets a host's existing per-tenant guard keep working
    /// while the workflow gets its aggregate cap.
    pub fn with_inner(mut self, inner: Arc<dyn BudgetGuard>) -> Self {
        self.inner = Some(inner);
        self
    }

    /// Total tokens recorded so far.
    pub fn consumed_tokens(&self) -> u64 {
        self.consumed_tokens.load(Ordering::SeqCst)
    }

    /// A point-in-time view of the ledger.
    pub fn snapshot(&self) -> BudgetSnapshot {
        BudgetSnapshot {
            consumed_tokens: self.consumed_tokens(),
            limit_tokens: self.limit_tokens,
        }
    }

    /// Whether the configured cap has been reached.
    pub fn is_exhausted(&self) -> bool {
        self.snapshot().is_exhausted()
    }
}

#[async_trait]
impl BudgetGuard for WorkflowBudget {
    async fn check_before_llm(
        &self,
        session_id: &str,
        estimated_prompt_tokens: usize,
    ) -> BudgetDecision {
        if self.is_exhausted() {
            return BudgetDecision::Deny {
                resource: "workflow_tokens".to_string(),
                reason: format!(
                    "workflow token budget exhausted ({} / {} tokens)",
                    self.consumed_tokens(),
                    self.limit_tokens.unwrap_or(0)
                ),
            };
        }
        match &self.inner {
            Some(inner) => {
                inner
                    .check_before_llm(session_id, estimated_prompt_tokens)
                    .await
            }
            None => BudgetDecision::Allow,
        }
    }

    async fn record_after_llm(&self, session_id: &str, usage: &TokenUsage) {
        self.consumed_tokens
            .fetch_add(usage.total_tokens as u64, Ordering::SeqCst);
        if let Some(inner) = &self.inner {
            inner.record_after_llm(session_id, usage).await;
        }
    }

    async fn check_before_tool(&self, session_id: &str, tool_name: &str) -> BudgetDecision {
        if self.is_exhausted() {
            return BudgetDecision::Deny {
                resource: "workflow_tokens".to_string(),
                reason: "workflow token budget exhausted".to_string(),
            };
        }
        match &self.inner {
            Some(inner) => inner.check_before_tool(session_id, tool_name).await,
            None => BudgetDecision::Allow,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn usage(total: usize) -> TokenUsage {
        TokenUsage {
            total_tokens: total,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn accumulates_and_caps() {
        let budget = WorkflowBudget::new(Some(100));
        assert!(matches!(
            budget.check_before_llm("s", 0).await,
            BudgetDecision::Allow
        ));

        budget.record_after_llm("a", &usage(60)).await;
        budget.record_after_llm("b", &usage(50)).await; // total 110 >= 100
        assert_eq!(budget.consumed_tokens(), 110);
        assert!(budget.is_exhausted());

        match budget.check_before_llm("s", 0).await {
            BudgetDecision::Deny { resource, .. } => assert_eq!(resource, "workflow_tokens"),
            other => panic!("expected Deny, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn uncapped_never_denies() {
        let budget = WorkflowBudget::new(None);
        budget.record_after_llm("a", &usage(1_000_000)).await;
        assert!(!budget.is_exhausted());
        assert!(matches!(
            budget.check_before_llm("s", 0).await,
            BudgetDecision::Allow
        ));
        assert_eq!(budget.snapshot().remaining_tokens(), None);
    }

    #[tokio::test]
    async fn snapshot_reports_remaining() {
        let budget = WorkflowBudget::new(Some(100));
        budget.record_after_llm("a", &usage(30)).await;
        let snap = budget.snapshot();
        assert_eq!(snap.consumed_tokens, 30);
        assert_eq!(snap.remaining_tokens(), Some(70));
        assert!(!snap.is_exhausted());
    }

    #[tokio::test]
    async fn delegates_to_inner_guard() {
        #[derive(Default)]
        struct Counting {
            checks: AtomicUsize,
            records: AtomicUsize,
        }
        #[async_trait]
        impl BudgetGuard for Counting {
            async fn check_before_llm(&self, _: &str, _: usize) -> BudgetDecision {
                self.checks.fetch_add(1, Ordering::SeqCst);
                BudgetDecision::Allow
            }
            async fn record_after_llm(&self, _: &str, _: &TokenUsage) {
                self.records.fetch_add(1, Ordering::SeqCst);
            }
        }

        let inner = Arc::new(Counting::default());
        let budget = WorkflowBudget::new(Some(1000)).with_inner(inner.clone());
        budget.check_before_llm("s", 0).await;
        budget.record_after_llm("s", &usage(10)).await;
        assert_eq!(inner.checks.load(Ordering::SeqCst), 1);
        assert_eq!(inner.records.load(Ordering::SeqCst), 1);
        assert_eq!(budget.consumed_tokens(), 10);
    }
}
