//! Durable serve layer for filesystem-first agents.
//!
//! Builds strictly on top of a3s-code's existing primitives — no new execution
//! machinery. It provides cron [`schedule`]s and the serve [`daemon`], which
//! installs the agent dir's `tools/` (MCP + sandboxed scripts) into each schedule
//! session. When a [`SessionStore`](crate::store::SessionStore) is configured,
//! schedule sessions rehydrate from it on boot, so a daemon restart resumes the
//! accumulated context rather than starting cold. Gated behind the `serve`
//! feature so library-only embedders pay nothing.
//!
//! Invariant: every schedule-triggered run is a FULL harness turn (context, tool
//! visibility, safety gate, verification) via `AgentSession::send`, never a raw
//! model call.

pub mod daemon;
pub mod schedule;
pub mod tools;

pub use daemon::serve_agent_dir;
pub use schedule::{ScheduleSink, ScheduledJob, Scheduler};
pub use tools::install_agent_dir_tools;
