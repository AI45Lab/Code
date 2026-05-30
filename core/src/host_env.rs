//! Host-environment plumbing: ID generation and time.
//!
//! The framework relies on two ambient capabilities — fresh IDs and the
//! current time — at many call sites (`session_id`, `run_id`, event
//! timestamps, retry backoff). Defaulting both to `uuid::Uuid::new_v4()`
//! / `SystemTime::now()` is fine for production but blocks two
//! cluster-grade features:
//!
//! - **Deterministic replay** of a run on another node for failure
//!   investigation. With injectable [`IdGenerator`] / [`Clock`] impls a
//!   host can record the seed and replay it bit-identical elsewhere.
//! - **Time-bending tests** without monkey-patching `std::time`.
//!
//! Hosts plug a custom impl via
//! [`SessionOptions::with_host_env`](crate::agent_api::SessionOptions::with_host_env);
//! the framework uses [`SystemHostEnv`] (the wall-clock + random-UUID
//! default) when none is supplied — observably identical to pre-P2
//! behaviour.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Generator for unique identifiers used by the framework
/// (session_id, run_id, subagent task_id, …).
///
/// The contract is intentionally loose: implementations may produce
/// random, monotonic, or deterministic-by-seed IDs. The framework
/// treats output as opaque and only requires uniqueness within the
/// hosting process.
pub trait IdGenerator: Send + Sync + std::fmt::Debug {
    /// Return a fresh ID. May be called concurrently from many tasks.
    fn next_id(&self) -> String;
}

/// Source of the current time in Unix-epoch milliseconds.
///
/// Same uniqueness contract as [`IdGenerator`]: the framework treats
/// the value as opaque. Monotonicity is not required (NTP corrections
/// happen) but typical impls are at least non-decreasing.
pub trait Clock: Send + Sync + std::fmt::Debug {
    /// Current time, milliseconds since Unix epoch.
    fn now_ms(&self) -> u64;
}

/// Bundle of host-environment capabilities. Used as the single
/// `Option<Arc<HostEnv>>` slot on [`AgentConfig`](crate::agent::AgentConfig)
/// and [`SessionOptions`](crate::agent_api::SessionOptions) — avoids
/// growing two parallel `Arc<dyn …>` fields.
#[derive(Debug, Clone)]
pub struct HostEnv {
    pub id_generator: Arc<dyn IdGenerator>,
    pub clock: Arc<dyn Clock>,
}

impl HostEnv {
    /// Construct a host env from concrete components.
    pub fn new(id_generator: Arc<dyn IdGenerator>, clock: Arc<dyn Clock>) -> Self {
        Self {
            id_generator,
            clock,
        }
    }

    /// Default system-backed host env: random UUIDs + wall clock.
    /// Equivalent to pre-P2 behaviour.
    pub fn system() -> Self {
        Self {
            id_generator: Arc::new(SystemIdGenerator),
            clock: Arc::new(SystemClock),
        }
    }

    /// Shortcut for `self.id_generator.next_id()`.
    pub fn next_id(&self) -> String {
        self.id_generator.next_id()
    }

    /// Shortcut for `self.clock.now_ms()`.
    pub fn now_ms(&self) -> u64 {
        self.clock.now_ms()
    }
}

impl Default for HostEnv {
    fn default() -> Self {
        Self::system()
    }
}

// ============================================================================
// Default impls
// ============================================================================

/// UUID-v4 based ID generator — the framework default.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemIdGenerator;

impl IdGenerator for SystemIdGenerator {
    fn next_id(&self) -> String {
        uuid::Uuid::new_v4().to_string()
    }
}

/// Wall-clock time source — the framework default.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

// ============================================================================
// Deterministic helpers (cfg(test) + replay)
// ============================================================================

/// Deterministic ID generator that yields a configured prefix followed
/// by a monotonic counter (`<prefix>-0`, `<prefix>-1`, …).
///
/// Public so external host crates (e.g. replay tooling) can use it
/// without re-implementing the pattern.
#[derive(Debug, Default)]
pub struct SequentialIdGenerator {
    prefix: String,
    counter: std::sync::atomic::AtomicU64,
}

impl SequentialIdGenerator {
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            counter: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

impl IdGenerator for SequentialIdGenerator {
    fn next_id(&self) -> String {
        let n = self
            .counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if self.prefix.is_empty() {
            n.to_string()
        } else {
            format!("{}-{}", self.prefix, n)
        }
    }
}

/// Clock that returns a configured, atomically-updatable timestamp.
/// Useful for replay (advance to recorded value) and for tests that
/// need stable timestamps.
#[derive(Debug)]
pub struct FixedClock {
    now_ms: std::sync::atomic::AtomicU64,
}

impl FixedClock {
    pub fn new(now_ms: u64) -> Self {
        Self {
            now_ms: std::sync::atomic::AtomicU64::new(now_ms),
        }
    }

    /// Atomically set the clock to a new value. Returns the previous value.
    pub fn set(&self, now_ms: u64) -> u64 {
        self.now_ms
            .swap(now_ms, std::sync::atomic::Ordering::SeqCst)
    }

    /// Advance the clock by `delta_ms`.
    pub fn advance(&self, delta_ms: u64) {
        self.now_ms
            .fetch_add(delta_ms, std::sync::atomic::Ordering::SeqCst);
    }
}

impl Clock for FixedClock {
    fn now_ms(&self) -> u64 {
        self.now_ms.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_host_env_produces_nonempty_ids_and_increasing_time() {
        let env = HostEnv::system();
        let a = env.next_id();
        let b = env.next_id();
        assert!(!a.is_empty());
        assert!(!b.is_empty());
        assert_ne!(a, b);
        let t1 = env.now_ms();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let t2 = env.now_ms();
        assert!(t2 >= t1);
    }

    #[test]
    fn sequential_id_generator_is_deterministic() {
        let gen = SequentialIdGenerator::new("run");
        assert_eq!(gen.next_id(), "run-0");
        assert_eq!(gen.next_id(), "run-1");
        assert_eq!(gen.next_id(), "run-2");
    }

    #[test]
    fn fixed_clock_is_controllable() {
        let clock = FixedClock::new(1000);
        assert_eq!(clock.now_ms(), 1000);
        clock.advance(500);
        assert_eq!(clock.now_ms(), 1500);
        assert_eq!(clock.set(0), 1500);
        assert_eq!(clock.now_ms(), 0);
    }
}
