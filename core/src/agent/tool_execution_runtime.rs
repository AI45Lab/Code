use super::tool_result_runtime::NormalizedToolResult;
use super::{AgentEvent, AgentLoop, ToolCommand};
use crate::tools::{ToolContext, ToolStreamEvent};
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

impl AgentLoop {
    pub(super) async fn execute_approved_tool_call(
        &self,
        event_tx: &Option<mpsc::Sender<AgentEvent>>,
        tool_id: &str,
        tool_name: &str,
        args: &Value,
    ) -> NormalizedToolResult {
        let stream_ctx = self.streaming_tool_context(event_tx, tool_id, tool_name);
        let normalized = NormalizedToolResult::from_execution(
            self.execute_tool_queued_or_direct(tool_name, args, &stream_ctx)
                .await,
        );
        self.track_tool_result(tool_name, args, normalized.exit_code);
        normalized
    }

    fn tool_context_for_plan(&self, session_id: Option<&str>) -> ToolContext {
        let mut ctx = self.tool_context.clone();
        if ctx.session_id.is_none() {
            if let Some(session_id) = session_id.filter(|id| !id.is_empty()) {
                ctx = ctx.with_session_id(session_id);
            }
        }
        ctx
    }

    pub(super) async fn execute_delegated_plan_tool(
        &self,
        tool_name: &str,
        args: &Value,
        session_id: Option<&str>,
        event_tx: &Option<mpsc::Sender<AgentEvent>>,
    ) -> (String, i32, bool, Option<Value>) {
        let call_id = format!("plan-{}-{}", tool_name, uuid::Uuid::new_v4());
        if let Some(tx) = event_tx {
            tx.send(AgentEvent::ToolStart {
                id: call_id.clone(),
                name: tool_name.to_string(),
            })
            .await
            .ok();
        }

        let ctx = self.tool_context_for_plan(session_id);
        let normalized = NormalizedToolResult::from_execution(
            self.execute_tool_timed(tool_name, args, &ctx).await,
        );

        if let Some(tx) = event_tx {
            tx.send(AgentEvent::ToolEnd {
                id: call_id,
                name: tool_name.to_string(),
                output: normalized.output.clone(),
                exit_code: normalized.exit_code,
                metadata: normalized.metadata.clone(),
                error_kind: normalized.error_kind.clone(),
            })
            .await
            .ok();
        }

        (
            normalized.output,
            normalized.exit_code,
            normalized.is_error,
            normalized.metadata,
        )
    }

    /// Execute a tool, applying the configured timeout if set.
    ///
    /// On timeout, returns an error describing which tool timed out and after
    /// how many milliseconds. The caller converts this to a tool-result error
    /// message that is fed back to the LLM.
    async fn execute_tool_timed(
        &self,
        name: &str,
        args: &serde_json::Value,
        ctx: &ToolContext,
    ) -> anyhow::Result<crate::tools::ToolResult> {
        let fut = self.tool_executor.execute_with_context(name, args, ctx);
        if let Some(timeout_ms) = self.config.tool_timeout_ms {
            match tokio::time::timeout(Duration::from_millis(timeout_ms), fut).await {
                Ok(result) => result,
                Err(_) => Err(anyhow::anyhow!(
                    "Tool '{}' timed out after {}ms",
                    name,
                    timeout_ms
                )),
            }
        } else {
            fut.await
        }
    }

    /// Execute a tool through the lane queue (if configured) or directly.
    async fn execute_tool_queued_or_direct(
        &self,
        name: &str,
        args: &serde_json::Value,
        ctx: &ToolContext,
    ) -> anyhow::Result<crate::tools::ToolResult> {
        self.execute_tool_queued_or_direct_inner(name, args, ctx)
            .await
    }

    /// Inner execution without task lifecycle wrapping.
    async fn execute_tool_queued_or_direct_inner(
        &self,
        name: &str,
        args: &serde_json::Value,
        ctx: &ToolContext,
    ) -> anyhow::Result<crate::tools::ToolResult> {
        if let Some(ref queue) = self.command_queue {
            let command = ToolCommand::new(
                Arc::clone(&self.tool_executor),
                name.to_string(),
                args.clone(),
                ctx.clone(),
                self.config.skill_registry.clone(),
                self.config.enforce_active_skill_tool_restrictions,
            );
            let rx = queue.submit_by_tool(name, Box::new(command)).await;
            match rx.await {
                Ok(Ok(value)) => {
                    let output = value["output"]
                        .as_str()
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "Queue result missing 'output' field for tool '{}'",
                                name
                            )
                        })?
                        .to_string();
                    let exit_code = value["exit_code"].as_i64().unwrap_or(0) as i32;
                    return Ok(crate::tools::ToolResult {
                        name: name.to_string(),
                        output,
                        exit_code,
                        metadata: None,
                        images: Vec::new(),
                        error_kind: None,
                    });
                }
                Ok(Err(e)) => {
                    tracing::warn!(
                        "Queue execution failed for tool '{}', falling back to direct: {}",
                        name,
                        e
                    );
                }
                Err(_) => {
                    tracing::warn!(
                        "Queue channel closed for tool '{}', falling back to direct",
                        name
                    );
                }
            }
        }
        self.execute_tool_timed(name, args, ctx).await
    }
    /// Create a tool context with streaming support.
    ///
    /// When `event_tx` is Some, spawns a forwarder task that converts
    /// `ToolStreamEvent::OutputDelta` into `AgentEvent::ToolOutputDelta`
    /// and sends them to the agent event channel.
    ///
    /// Returns the augmented `ToolContext`. The forwarder task runs until
    /// the tool-side sender is dropped (i.e., tool execution finishes).
    fn streaming_tool_context(
        &self,
        event_tx: &Option<mpsc::Sender<AgentEvent>>,
        tool_id: &str,
        tool_name: &str,
    ) -> ToolContext {
        let mut ctx = self.tool_context.clone();
        if let Some(agent_tx) = event_tx {
            let (tool_tx, mut tool_rx) = mpsc::channel::<ToolStreamEvent>(64);
            ctx.event_tx = Some(tool_tx);

            let agent_tx = agent_tx.clone();
            let tool_id = tool_id.to_string();
            let tool_name = tool_name.to_string();
            tokio::spawn(async move {
                while let Some(event) = tool_rx.recv().await {
                    match event {
                        ToolStreamEvent::OutputDelta(delta) => {
                            agent_tx
                                .send(AgentEvent::ToolOutputDelta {
                                    id: tool_id.clone(),
                                    name: tool_name.clone(),
                                    delta,
                                })
                                .await
                                .ok();
                        }
                    }
                }
            });
        }
        ctx
    }
}
