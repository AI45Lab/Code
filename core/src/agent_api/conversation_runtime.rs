//! Conversation execution facade for a session.
//!
//! This module owns the public conversation contract: slash-command dispatch,
//! blocking sends, streaming sends, and attachment handling.
//! Lower-level runtime modules own run lifecycle and event forwarding.

use super::{
    command_runtime, runtime::BlockingRunContext, runtime::ConversationInput,
    runtime::StreamRunContext, AgentSession,
};
use crate::agent::{AgentEvent, AgentResult};
use crate::error::{CodeError, Result};
use crate::llm::{Attachment, Message};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

fn bail_if_closed(session: &AgentSession) -> Result<()> {
    if session.is_closed() {
        return Err(CodeError::SessionClosed {
            session_id: session.session_id.clone(),
        });
    }
    Ok(())
}

pub(super) async fn send(
    session: &AgentSession,
    prompt: &str,
    history: Option<&[Message]>,
) -> Result<AgentResult> {
    bail_if_closed(session)?;

    if let Some(result) = command_runtime::dispatch_blocking(session, prompt, history).await? {
        return Ok(result);
    }

    warn_deferred_init(session);
    let input = ConversationInput::from_history(session, history);
    let blocking_run = BlockingRunContext::start(session, prompt, input.persistence).await;
    blocking_run
        .execute_with_prompt(&input.messages, prompt, &session.session_id)
        .await
}

pub(super) async fn send_with_attachments(
    session: &AgentSession,
    prompt: &str,
    attachments: &[Attachment],
    history: Option<&[Message]>,
) -> Result<AgentResult> {
    bail_if_closed(session)?;

    // Build one user message containing text and images, then execute from the
    // resulting message list so the loop does not append a duplicate prompt.
    let input = ConversationInput::with_attachments(session, history, prompt, attachments);
    let blocking_run = BlockingRunContext::start(session, prompt, input.persistence).await;
    blocking_run
        .execute_from_messages(input.messages, &session.session_id)
        .await
}

pub(super) async fn stream_with_attachments(
    session: &AgentSession,
    prompt: &str,
    attachments: &[Attachment],
    history: Option<&[Message]>,
) -> Result<(mpsc::Receiver<AgentEvent>, JoinHandle<()>)> {
    bail_if_closed(session)?;

    let input = ConversationInput::with_attachments(session, history, prompt, attachments);
    let stream_run = StreamRunContext::start(session, prompt, input.persistence).await;
    Ok(stream_run.spawn_from_messages(input.messages))
}

pub(super) async fn stream(
    session: &AgentSession,
    prompt: &str,
    history: Option<&[Message]>,
) -> Result<(mpsc::Receiver<AgentEvent>, JoinHandle<()>)> {
    bail_if_closed(session)?;

    if let Some(stream) = command_runtime::dispatch_streaming(session, prompt).await {
        return Ok(stream);
    }

    let input = ConversationInput::from_history(session, history);
    let stream_run = StreamRunContext::start(session, prompt, input.persistence).await;
    Ok(stream_run.spawn_with_prompt(input.messages, prompt.to_string()))
}

/// Resume a previously-checkpointed run on this session (P3 cut 2).
///
/// Loads the latest [`LoopCheckpoint`](crate::loop_checkpoint::LoopCheckpoint)
/// for `checkpoint_run_id` from the session's `SessionStore` and replays
/// the agent loop from that boundary state. A **new** run id is
/// generated for the resumed work — the relationship between the old
/// and new run is metadata the host tracks externally.
///
/// Returns an error when the session has no store configured, or when
/// no checkpoint exists for `checkpoint_run_id`.
pub(super) async fn resume_run(
    session: &AgentSession,
    checkpoint_run_id: &str,
) -> Result<crate::agent::AgentResult> {
    bail_if_closed(session)?;

    let store = session.session_store.as_ref().ok_or_else(|| {
        CodeError::Session("resume_run requires a session_store on this session".to_string())
    })?;

    let checkpoint = store
        .load_loop_checkpoint(checkpoint_run_id)
        .await
        .map_err(|e| {
            CodeError::Session(format!(
                "load_loop_checkpoint('{checkpoint_run_id}') failed: {e}"
            ))
        })?
        .ok_or_else(|| {
            CodeError::Session(format!(
                "no loop checkpoint found for run '{checkpoint_run_id}'"
            ))
        })?;

    let persistence =
        Some(super::session_persistence::SessionPersistenceContext::from_session(session));
    let blocking_run = BlockingRunContext::start(
        session,
        &format!("<resume run={checkpoint_run_id} turn={}>", checkpoint.turn),
        persistence,
    )
    .await;
    // Seed the resumed run's loop state with the cumulative metrics from
    // the checkpoint so token usage and tool-call counts continue from
    // where the crashed/migrated run left off rather than re-starting at
    // zero (which would under-report the resumed AgentResult).
    let seed = crate::agent::ExecutionSeed {
        total_usage: checkpoint.total_usage.clone(),
        tool_calls_count: checkpoint.tool_calls_count,
        verification_reports: checkpoint.verification_reports.clone(),
    };
    blocking_run
        .execute_from_messages_seeded(checkpoint.messages, &session.session_id, Some(seed))
        .await
}

fn warn_deferred_init(session: &AgentSession) {
    if let Some(warning) = &session.init_warning {
        tracing::warn!(
            session_id = %session.session_id,
            "Session init warning: {}", warning
        );
    }
}
