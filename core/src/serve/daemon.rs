//! The serve daemon: runs a filesystem-first agent's cron schedules as full,
//! durable harness turns until cancelled.
//!
//! Each schedule fires on its OWN session (stable id `schedule:<name>`), so a
//! schedule's repeated fires accumulate context/memory while distinct schedules
//! stay isolated. The agent dir's `instructions.md` (prompt slots), `skills/`
//! (`skill_dirs`), and `tools/` (MCP servers and sandboxed scripts) are injected
//! into every schedule session via [`SessionOptions`] / `install_agent_dir_tools`.
//!
//! Rehydrate-on-boot: when a [`SessionStore`](crate::store::SessionStore) is
//! configured, a schedule whose `schedule:<name>` session already exists in the
//! store is RESUMED (its conversation history is restored), so a daemon restart
//! keeps the accumulated context instead of starting cold. The current
//! `instructions.md`/`skills/`/`tools/` still win on resume (resume restores
//! history, not the prompt), so editing the agent dir takes effect on the next
//! boot. With no store configured, every boot starts a fresh session — same as
//! before. Two caveats: rehydrate restores conversation context, NOT missed
//! fires — a schedule that would have fired while the daemon was down is not
//! caught up; and resume loads the stored session via `block_in_place`, so the
//! daemon must run on a multi-threaded Tokio runtime (the default `#[tokio::main]`
//! and both bundled SDK runtimes are multi-threaded).
//!
//! Every triggered run is a FULL harness turn (`AgentSession::send`) — context,
//! tool visibility, safety gate, verification — never a raw model call.

use std::collections::HashMap;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use super::schedule::{ScheduleSink, Scheduler};
use crate::agent_api::{Agent, AgentSession, SessionOptions};
use crate::config::{AgentDir, ScheduleSpec};
use crate::error::Result;

/// Routes each schedule fire to that schedule's own [`AgentSession`] as a `send`
/// turn (context, tool visibility, safety gate, verification all stay on).
struct SessionScheduleSink {
    sessions: HashMap<String, Arc<AgentSession>>,
}

#[async_trait::async_trait]
impl ScheduleSink for SessionScheduleSink {
    async fn fire(&self, spec: &ScheduleSpec) {
        let Some(session) = self.sessions.get(&spec.name) else {
            tracing::warn!(schedule = %spec.name, "no session for schedule; skipping fire");
            return;
        };
        match session.send(&spec.prompt, None).await {
            Ok(_) => tracing::info!(schedule = %spec.name, "scheduled turn completed"),
            Err(e) => {
                tracing::warn!(schedule = %spec.name, error = %e, "scheduled turn failed")
            }
        }
    }
}

/// Serve an agent directory's schedules until `cancel` fires.
///
/// Builds one durable session per enabled schedule (`schedule:<name>`), injecting
/// the agent dir's `prompt_slots`, `skill_dirs`, and `tools/`. `extra` merges into
/// each schedule session's [`SessionOptions`] (model, `llm_client`,
/// `session_store`, …); a host-set `prompt_slots` is honored, but `session_id` is
/// always derived per schedule.
pub async fn serve_agent_dir(
    agent: &Agent,
    agent_dir: &AgentDir,
    workspace: impl Into<String> + Clone,
    extra: Option<SessionOptions>,
    cancel: CancellationToken,
) -> Result<()> {
    let extra = extra.unwrap_or_default();

    let mut sessions = HashMap::new();
    for spec in agent_dir.schedules.iter().filter(|s| s.enabled) {
        let session = build_session(
            agent,
            agent_dir,
            format!("schedule:{}", spec.name),
            workspace.clone(),
            &extra,
        )
        .await?;
        sessions.insert(spec.name.clone(), Arc::new(session));
    }

    let scheduler = Scheduler::new(agent_dir.schedules.clone())?;
    let sink: Arc<dyn ScheduleSink> = Arc::new(SessionScheduleSink { sessions });
    scheduler.run(sink, cancel).await;
    Ok(())
}

/// Build one durable serve session under the explicit `session_id`
/// (`schedule:<name>`), injecting the agent dir's prompt slots, skills, and
/// `tools/`.
///
/// Rehydrate-on-boot: if a `SessionStore` is configured (via `extra`) and already
/// holds `session_id`, RESUME it so prior context is restored; otherwise create a
/// fresh session. Resume re-applies the freshly loaded prompt slots / skills /
/// tools (it restores history, not the prompt) and uses the stored workspace; a
/// fresh session uses `workspace`.
///
/// The caller owns the id; a host-set `extra.session_id` is intentionally ignored
/// (it would collide every schedule onto one shared, store-clobbering id).
async fn build_session(
    agent: &Agent,
    agent_dir: &AgentDir,
    session_id: String,
    workspace: impl Into<String>,
    extra: &SessionOptions,
) -> Result<AgentSession> {
    let mut opts = extra.clone();
    if opts.prompt_slots.is_none() {
        opts.prompt_slots = Some(agent_dir.prompt_slots.clone());
    }
    opts.skill_dirs
        .extend(agent_dir.config.skill_dirs.iter().cloned());
    opts.session_id = Some(session_id.clone());

    // Resume only when the store already has this session; the borrow of
    // `opts.session_store` is released before `opts` is moved into the chosen
    // builder. `exists` returns the crate Result, so `?` propagates cleanly.
    let resume = match &opts.session_store {
        Some(store) => store.exists(&session_id).await?,
        None => false,
    };
    let session = if resume {
        agent.resume_session(&session_id, opts)?
    } else {
        agent.session(workspace, Some(opts))?
    };

    // Install the agent dir's tools/ (MCP servers + sandboxed scripts) into the
    // session, so a triggered turn can call them. Connection is fallible and
    // surfaces here (fail at startup, not at first call). Done for both fresh and
    // resumed sessions — tools are not persisted, they are re-installed each boot.
    super::tools::install_agent_dir_tools(&session, &agent_dir.tools).await?;
    Ok(session)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CodeConfig;
    use crate::prompts::SystemPromptSlots;

    fn test_agent_config() -> CodeConfig {
        let acl = r#"
default_model = "anthropic/claude-sonnet-4-20250514"
providers "anthropic" {
  api_key = "test-key"
  models "claude-sonnet-4-20250514" { name = "Claude Sonnet 4" }
}
"#;
        CodeConfig::from_acl(acl).unwrap()
    }

    fn agent_dir_with(schedules: Vec<ScheduleSpec>) -> AgentDir {
        AgentDir {
            dir: std::path::PathBuf::from("/tmp/serve-test-agent"),
            config: CodeConfig::default(),
            prompt_slots: SystemPromptSlots {
                role: Some("a scheduled test agent".to_string()),
                ..Default::default()
            },
            schedules,
            tools: vec![],
        }
    }

    #[tokio::test]
    async fn serve_with_no_schedules_returns_immediately() {
        let agent = Agent::from_config(test_agent_config()).await.unwrap();
        let dir = agent_dir_with(vec![]);
        let cancel = CancellationToken::new();
        // No schedules → no sessions, no jobs; returns Ok without blocking.
        serve_agent_dir(&agent, &dir, "/tmp/ws".to_string(), None, cancel)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn serve_builds_per_schedule_session_and_stops_on_cancel() {
        let agent = Agent::from_config(test_agent_config()).await.unwrap();
        let dir = agent_dir_with(vec![ScheduleSpec {
            name: "tick".to_string(),
            cron: "* * * * *".to_string(),
            prompt: "do the scheduled work".to_string(),
            enabled: true,
        }]);
        // Pre-cancel: the per-schedule session is created and the scheduler starts,
        // but the job loop returns on the cancelled token before any fire — so this
        // exercises session creation + wiring + graceful shutdown with no LLM call.
        let cancel = CancellationToken::new();
        cancel.cancel();
        serve_agent_dir(&agent, &dir, "/tmp/ws".to_string(), None, cancel)
            .await
            .unwrap();
    }

    fn tick_dir() -> AgentDir {
        agent_dir_with(vec![ScheduleSpec {
            name: "tick".to_string(),
            cron: "0 9 * * *".to_string(),
            prompt: "do the scheduled work".to_string(),
            enabled: true,
        }])
    }

    /// Rehydrate-on-boot: a schedule whose session already exists in the store is
    /// RESUMED, so a daemon restart keeps the accumulated conversation history.
    /// (multi_thread: `resume_session` loads via `block_in_place`.)
    #[tokio::test(flavor = "multi_thread")]
    async fn rehydrates_existing_schedule_session_from_store() {
        use crate::llm::Message;
        let store: Arc<dyn crate::store::SessionStore> =
            Arc::new(crate::store::MemorySessionStore::new());
        let agent = Agent::from_config(test_agent_config()).await.unwrap();

        // Simulate a prior daemon run: persist the schedule's session, then inject
        // a message into the stored snapshot (seeds via public API only — the
        // history field is private to agent_api).
        let seed_opts = SessionOptions::new()
            .with_session_store(store.clone())
            .with_session_id("schedule:tick");
        agent
            .session("/tmp/ws", Some(seed_opts))
            .unwrap()
            .save()
            .await
            .unwrap();
        let mut data = store.load("schedule:tick").await.unwrap().unwrap();
        data.messages = vec![Message::user("prior turn")];
        store.save(&data).await.unwrap();

        // A fresh daemon boot must RESUME this schedule, not start empty.
        let dir = tick_dir();
        let extra = SessionOptions::new().with_session_store(store.clone());
        let session = build_session(
            &agent,
            &dir,
            "schedule:tick".to_string(),
            "/tmp/ws".to_string(),
            &extra,
        )
        .await
        .unwrap();

        assert_eq!(session.session_id(), "schedule:tick");
        let history = session.history();
        assert_eq!(history.len(), 1, "prior context must rehydrate on boot");
        assert_eq!(history[0].text(), "prior turn");
    }

    /// No prior session in the store → a fresh, empty session (the resume branch
    /// is not taken). Also covers the no-store case implicitly (fresh either way).
    #[tokio::test(flavor = "multi_thread")]
    async fn fresh_schedule_session_when_store_has_no_prior() {
        let store: Arc<dyn crate::store::SessionStore> =
            Arc::new(crate::store::MemorySessionStore::new());
        let agent = Agent::from_config(test_agent_config()).await.unwrap();
        let dir = tick_dir();
        let extra = SessionOptions::new().with_session_store(store.clone());
        let session = build_session(
            &agent,
            &dir,
            "schedule:tick".to_string(),
            "/tmp/ws".to_string(),
            &extra,
        )
        .await
        .unwrap();
        assert_eq!(session.session_id(), "schedule:tick");
        assert!(
            session.history().is_empty(),
            "no prior session → fresh empty history"
        );
    }

    /// No store configured at all → fresh session, and the resume path (the only
    /// `block_in_place` caller) is never taken, so it works on a current-thread
    /// runtime (plain `#[tokio::test]`).
    #[tokio::test]
    async fn fresh_schedule_session_when_no_store_configured() {
        let agent = Agent::from_config(test_agent_config()).await.unwrap();
        let dir = tick_dir();
        let session = build_session(
            &agent,
            &dir,
            "schedule:tick".to_string(),
            "/tmp/ws".to_string(),
            &SessionOptions::new(),
        )
        .await
        .unwrap();
        assert_eq!(session.session_id(), "schedule:tick");
        assert!(session.history().is_empty());
    }
}
