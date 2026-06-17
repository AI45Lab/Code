//! The serve daemon: runs a filesystem-first agent's cron schedules as full,
//! durable harness turns until cancelled.
//!
//! Each schedule fires on its OWN session (stable id `schedule:<name>`), so a
//! schedule's repeated fires accumulate context/memory while distinct schedules
//! stay isolated. The agent dir's `instructions.md` (prompt slots) and `skills/`
//! (`skill_dirs`) are injected into every schedule session via [`SessionOptions`].
//!
//! Channels and full multi-session rehydration attach here next; the design keeps
//! every triggered run a FULL harness turn (`AgentSession::send`), never a raw
//! model call.

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
/// Builds one session per enabled schedule (stable id `schedule:<name>`),
/// injecting the agent dir's `prompt_slots` and `skill_dirs`. `extra` merges into
/// every schedule session's [`SessionOptions`] (model, `llm_client`,
/// `session_store`, …) — `prompt_slots`/`session_id` set there are NOT overridden,
/// so a host can pin them per schedule if it wants.
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
        let mut opts = extra.clone();
        if opts.prompt_slots.is_none() {
            opts.prompt_slots = Some(agent_dir.prompt_slots.clone());
        }
        opts.skill_dirs
            .extend(agent_dir.config.skill_dirs.iter().cloned());
        if opts.session_id.is_none() {
            opts.session_id = Some(format!("schedule:{}", spec.name));
        }
        let session = agent.session(workspace.clone(), Some(opts))?;
        // Install the agent dir's tools/ (e.g. MCP servers) into each schedule
        // session, so a scheduled turn can call them. Connection is fallible and
        // surfaces here (fail at startup, not at first call).
        super::tools::install_agent_dir_tools(&session, &agent_dir.tools).await?;
        sessions.insert(spec.name.clone(), Arc::new(session));
    }

    let scheduler = Scheduler::new(agent_dir.schedules.clone())?;
    let sink: Arc<dyn ScheduleSink> = Arc::new(SessionScheduleSink { sessions });
    scheduler.run(sink, cancel).await;
    Ok(())
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
            channels: vec![],
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
}
