//! Durable serve layer for filesystem-first agents.
//!
//! Builds strictly on top of a3s-code's existing primitives — no new execution
//! machinery. Today it provides cron [`schedule`]s; the serve daemon (a session
//! registry persisted via `SessionStore`, graceful shutdown, rehydrate-on-boot)
//! and inbound channels attach here next. Gated behind the `serve` feature so
//! library-only embedders pay nothing.
//!
//! Invariant: every schedule/channel-triggered run is a FULL harness turn
//! (context, tool visibility, safety gate, verification) via `AgentSession::send`,
//! never a raw model call.

pub mod daemon;
pub mod schedule;
pub mod tools;

pub use daemon::serve_agent_dir;
pub use schedule::{ScheduleSink, ScheduledJob, Scheduler};
pub use tools::install_agent_dir_tools;
