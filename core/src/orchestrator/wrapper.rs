//! SubAgent wrapper — executes a real AgentSession and forwards events to the
//! Orchestrator event bus, with pause/resume/cancel control signal support.

use crate::agent::AgentEvent;
use crate::error::Result;
use crate::orchestrator::{
    ControlSignal, OrchestratorEvent, SubAgentActivity, SubAgentConfig, SubAgentState,
};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, RwLock};

pub struct SubAgentWrapper {
    id: String,
    config: SubAgentConfig,
    /// Real agent for LLM execution; `None` → placeholder mode.
    agent: Option<Arc<crate::Agent>>,
    event_tx: broadcast::Sender<OrchestratorEvent>,
    control_rx: mpsc::Receiver<ControlSignal>,
    state: Arc<RwLock<SubAgentState>>,
    activity: Arc<RwLock<SubAgentActivity>>,
    /// Shared map of live sessions; wrapper registers its session here so
    /// `AgentOrchestrator::complete_external_task()` can reach it.
    session_registry:
        Arc<RwLock<std::collections::HashMap<String, Arc<crate::agent_api::AgentSession>>>>,
}

impl SubAgentWrapper {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: String,
        config: SubAgentConfig,
        agent: Option<Arc<crate::Agent>>,
        event_tx: broadcast::Sender<OrchestratorEvent>,
        control_rx: mpsc::Receiver<ControlSignal>,
        state: Arc<RwLock<SubAgentState>>,
        activity: Arc<RwLock<SubAgentActivity>>,
        session_registry: Arc<
            RwLock<std::collections::HashMap<String, Arc<crate::agent_api::AgentSession>>>,
        >,
    ) -> Self {
        Self {
            id,
            config,
            agent,
            event_tx,
            control_rx,
            state,
            activity,
            session_registry,
        }
    }

    /// Run the SubAgent.  Dispatches to real or placeholder execution.
    pub async fn execute(mut self) -> Result<String> {
        self.update_state(SubAgentState::Running).await;
        let start = std::time::Instant::now();

        let result = if let Some(agent) = self.agent.take() {
            self.execute_with_agent(agent).await
        } else {
            self.execute_placeholder().await
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        match &result {
            Ok(output) => {
                self.update_state(SubAgentState::Completed {
                    success: true,
                    output: output.clone(),
                })
                .await;
                let _ = self.event_tx.send(OrchestratorEvent::SubAgentCompleted {
                    id: self.id.clone(),
                    success: true,
                    output: output.clone(),
                    duration_ms,
                    token_usage: None,
                });
            }
            Err(e) => {
                let current = self.state.read().await.clone();
                if !matches!(current, SubAgentState::Cancelled) {
                    self.update_state(SubAgentState::Error {
                        message: e.to_string(),
                    })
                    .await;
                }
                let _ = self.event_tx.send(OrchestratorEvent::SubAgentCompleted {
                    id: self.id.clone(),
                    success: false,
                    output: e.to_string(),
                    duration_ms,
                    token_usage: None,
                });
            }
        }

        result
    }

    // -------------------------------------------------------------------------
    // Real execution via AgentSession
    // -------------------------------------------------------------------------

    async fn execute_with_agent(&mut self, agent: Arc<crate::Agent>) -> Result<String> {
        // Build an AgentRegistry from built-ins + extra agent_dirs.
        let registry = crate::AgentRegistry::new();
        for dir in &self.config.agent_dirs {
            let agents = crate::load_agents_from_dir(std::path::Path::new(dir));
            for def in agents {
                registry.register(def);
            }
        }

        // Build session options from SubAgentConfig fields.
        let mut opts = crate::SessionOptions::new();

        // Handle permissive mode with fine-grained deny control
        if self.config.permissive {
            // Build a permissive policy that still respects deny rules
            let mut policy = crate::permissions::PermissionPolicy::permissive();

            // Add deny rules from permissive_deny config
            for rule in &self.config.permissive_deny {
                policy = policy.deny(rule);
            }

            // If we have an agent definition, also add its deny rules
            if let Some(def) = registry.get(&self.config.agent_type) {
                for rule in &def.permissions.deny {
                    policy = policy.deny(&rule.rule);
                }
            }

            opts = opts.with_permission_checker(Arc::new(policy));
        }

        if let Some(steps) = self.config.max_steps {
            opts = opts.with_max_tool_rounds(steps);
        }
        if let Some(queue_cfg) = self.config.lane_config.clone() {
            opts = opts.with_queue_config(queue_cfg);
        }

        // Create session: use the named agent definition if found, otherwise
        // fall back to a plain session so unknown agent_types still work.
        let session = Arc::new(if let Some(def) = registry.get(&self.config.agent_type) {
            agent.session_for_agent(&self.config.workspace, &def, Some(opts))?
        } else {
            agent.session(&self.config.workspace, Some(opts))?
        });

        // Register session so complete_external_task() can reach it.
        self.session_registry
            .write()
            .await
            .insert(self.id.clone(), Arc::clone(&session));

        // Stream execution.
        let (mut rx, _task) = session.stream(&self.config.prompt, None).await?;

        let mut output = String::new();
        let mut step: usize = 0;

        loop {
            // Drain pending control signals before each event.
            while let Ok(signal) = self.control_rx.try_recv() {
                self.handle_control_signal(signal).await?;
            }

            // Abort if cancelled.
            if matches!(*self.state.read().await, SubAgentState::Cancelled) {
                // Drop rx to signal the background streaming task to stop.
                drop(rx);
                return Err(anyhow::anyhow!("Cancelled by orchestrator").into());
            }

            // Wait while paused (backpressure on rx naturally slows the agent).
            while matches!(*self.state.read().await, SubAgentState::Paused) {
                *self.activity.write().await = SubAgentActivity::WaitingForControl {
                    reason: "Paused by orchestrator".to_string(),
                };
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                while let Ok(signal) = self.control_rx.try_recv() {
                    self.handle_control_signal(signal).await?;
                }
                if matches!(*self.state.read().await, SubAgentState::Cancelled) {
                    drop(rx);
                    return Err(anyhow::anyhow!("Cancelled by orchestrator").into());
                }
            }

            // Consume the next agent event.
            match rx.recv().await {
                Some(AgentEvent::TurnStart { turn }) => {
                    *self.activity.write().await =
                        SubAgentActivity::RequestingLlm { message_count: 0 };
                    // Forward as internal event for observability
                    let _ = self
                        .event_tx
                        .send(OrchestratorEvent::SubAgentInternalEvent {
                            id: self.id.clone(),
                            event: AgentEvent::TurnStart { turn },
                        });
                }
                Some(AgentEvent::ToolStart { id, name }) => {
                    *self.activity.write().await = SubAgentActivity::CallingTool {
                        tool_name: name.clone(),
                        args: serde_json::Value::Null,
                    };
                    let _ = self.event_tx.send(OrchestratorEvent::ToolExecutionStarted {
                        id: self.id.clone(),
                        tool_id: id,
                        tool_name: name,
                        args: serde_json::Value::Null,
                    });
                }
                Some(AgentEvent::ToolEnd {
                    id,
                    name,
                    output: tool_out,
                    exit_code,
                    ..
                }) => {
                    step += 1;
                    *self.activity.write().await = SubAgentActivity::Idle;
                    let tool_start = std::time::Instant::now();
                    let _ = self
                        .event_tx
                        .send(OrchestratorEvent::ToolExecutionCompleted {
                            id: self.id.clone(),
                            tool_id: id,
                            tool_name: name,
                            result: tool_out,
                            exit_code,
                            duration_ms: tool_start.elapsed().as_millis() as u64,
                        });
                    let _ = self.event_tx.send(OrchestratorEvent::SubAgentProgress {
                        id: self.id.clone(),
                        step,
                        total_steps: self.config.max_steps.unwrap_or(0),
                        message: format!("Completed tool call {step}"),
                    });
                }
                Some(AgentEvent::TextDelta { text }) => {
                    output.push_str(&text);
                    // Forward as internal event for streaming observability
                    let _ = self
                        .event_tx
                        .send(OrchestratorEvent::SubAgentInternalEvent {
                            id: self.id.clone(),
                            event: AgentEvent::TextDelta { text },
                        });
                }
                Some(AgentEvent::ExternalTaskPending {
                    task_id,
                    session_id,
                    lane,
                    command_type,
                    payload,
                    timeout_ms,
                }) => {
                    let _ = self.event_tx.send(OrchestratorEvent::ExternalTaskPending {
                        id: self.id.clone(),
                        task_id,
                        lane,
                        command_type,
                        payload,
                        timeout_ms,
                    });
                    // session_id is informational; the orchestrator routes by subagent ID.
                    let _ = session_id;
                }
                Some(AgentEvent::ExternalTaskCompleted {
                    task_id,
                    session_id,
                    success,
                }) => {
                    let _ = self
                        .event_tx
                        .send(OrchestratorEvent::ExternalTaskCompleted {
                            id: self.id.clone(),
                            task_id,
                            success,
                        });
                    let _ = session_id;
                }
                Some(AgentEvent::End { text, .. }) => {
                    output = text;
                    break;
                }
                Some(AgentEvent::Error { message }) => {
                    return Err(anyhow::anyhow!("Agent error: {message}").into());
                }
                // Forward all other events as internal events for observability.
                Some(event) => {
                    let _ = self
                        .event_tx
                        .send(OrchestratorEvent::SubAgentInternalEvent {
                            id: self.id.clone(),
                            event,
                        });
                }
                None => break, // stream closed
            }
        }

        // Deregister so the Arc is dropped and the session can be freed.
        self.session_registry.write().await.remove(&self.id);

        Ok(output)
    }

    // -------------------------------------------------------------------------
    // Placeholder execution (backward compatibility when no agent is configured)
    // -------------------------------------------------------------------------

    async fn execute_placeholder(&mut self) -> Result<String> {
        for step in 1..=5 {
            while let Ok(signal) = self.control_rx.try_recv() {
                self.handle_control_signal(signal).await?;
            }

            if matches!(*self.state.read().await, SubAgentState::Cancelled) {
                return Err(anyhow::anyhow!("Cancelled by orchestrator").into());
            }

            while matches!(*self.state.read().await, SubAgentState::Paused) {
                *self.activity.write().await = SubAgentActivity::WaitingForControl {
                    reason: "Paused by orchestrator".to_string(),
                };
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                while let Ok(signal) = self.control_rx.try_recv() {
                    self.handle_control_signal(signal).await?;
                }
            }

            *self.activity.write().await = SubAgentActivity::CallingTool {
                tool_name: "read".to_string(),
                args: serde_json::json!({"path": "/tmp/file.txt"}),
            };

            let _ = self.event_tx.send(OrchestratorEvent::ToolExecutionStarted {
                id: self.id.clone(),
                tool_id: format!("tool-{step}"),
                tool_name: "read".to_string(),
                args: serde_json::json!({"path": "/tmp/file.txt"}),
            });

            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

            *self.activity.write().await = SubAgentActivity::RequestingLlm { message_count: 3 };

            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

            *self.activity.write().await = SubAgentActivity::Idle;

            let _ = self.event_tx.send(OrchestratorEvent::SubAgentProgress {
                id: self.id.clone(),
                step,
                total_steps: 5,
                message: format!("Step {step}/5 completed"),
            });
        }

        Ok(format!(
            "Placeholder result for SubAgent {} ({})",
            self.id, self.config.agent_type
        ))
    }

    // -------------------------------------------------------------------------
    // Control signal handling
    // -------------------------------------------------------------------------

    async fn handle_control_signal(&mut self, signal: ControlSignal) -> Result<()> {
        let _ = self
            .event_tx
            .send(OrchestratorEvent::ControlSignalReceived {
                id: self.id.clone(),
                signal: signal.clone(),
            });

        let result = match signal {
            ControlSignal::Pause => {
                self.update_state(SubAgentState::Paused).await;
                Ok(())
            }
            ControlSignal::Resume => {
                self.update_state(SubAgentState::Running).await;
                Ok(())
            }
            ControlSignal::Cancel => {
                self.update_state(SubAgentState::Cancelled).await;
                Err(anyhow::anyhow!("Cancelled by orchestrator").into())
            }
            ControlSignal::AdjustParams { max_steps, .. } => {
                if let Some(steps) = max_steps {
                    self.config.max_steps = Some(steps);
                }
                Ok(())
            }
            ControlSignal::InjectPrompt { ref prompt } => {
                // Append the injected prompt so the next LLM turn sees it.
                self.config.prompt.push('\n');
                self.config.prompt.push_str(prompt);
                Ok(())
            }
        };

        let _ = self.event_tx.send(OrchestratorEvent::ControlSignalApplied {
            id: self.id.clone(),
            signal,
            success: result.is_ok(),
            error: result.as_ref().err().map(|e| format!("{e}")),
        });

        result
    }

    async fn update_state(&self, new_state: SubAgentState) {
        let old_state = {
            let mut state = self.state.write().await;
            let old = state.clone();
            *state = new_state.clone();
            old
        };

        let _ = self.event_tx.send(OrchestratorEvent::SubAgentStateChanged {
            id: self.id.clone(),
            old_state,
            new_state,
        });
    }
}
