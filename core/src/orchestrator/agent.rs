//! Agent Orchestrator 核心实现

use crate::error::Result;
use crate::orchestrator::{
    AgentSlot, ControlSignal, OrchestratorConfig, OrchestratorEvent, SubAgentActivity,
    SubAgentConfig, SubAgentHandle, SubAgentInfo, SubAgentState,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};

/// Agent Orchestrator - 主子智能体协调器
///
/// 基于事件总线实现统一的监控和控制机制。
/// 默认使用内存事件通讯，支持用户自定义 NATS provider。
pub struct AgentOrchestrator {
    /// 配置
    config: OrchestratorConfig,

    /// Agent used to execute SubAgents (None = placeholder mode)
    agent: Option<Arc<crate::Agent>>,

    /// 事件广播通道
    event_tx: broadcast::Sender<OrchestratorEvent>,

    /// SubAgent 注册表
    subagents: Arc<RwLock<HashMap<String, SubAgentHandle>>>,

    /// Live session references keyed by subagent ID.
    ///
    /// Populated only for real-agent SubAgents (i.e., created via `from_agent()`).
    /// Used by `complete_external_task()` to route results back into the
    /// session's lane queue without exposing the session to the caller.
    sessions: Arc<RwLock<HashMap<String, Arc<crate::agent_api::AgentSession>>>>,

    /// 下一个 SubAgent ID
    next_id: Arc<RwLock<u64>>,
}

impl AgentOrchestrator {
    /// 创建新的 orchestrator（使用内存事件通讯）
    ///
    /// 这是默认的创建方式，适用于单进程场景。
    /// SubAgents 将以占位符模式运行，不执行实际的 LLM 操作。
    /// 要执行真实的 LLM 操作，请使用 `from_agent()`。
    pub fn new_memory() -> Self {
        Self::new(OrchestratorConfig::default())
    }

    /// 使用自定义配置创建 orchestrator（占位符模式）
    pub fn new(config: OrchestratorConfig) -> Self {
        let (event_tx, _) = broadcast::channel(config.event_buffer_size);

        Self {
            config,
            agent: None,
            event_tx,
            subagents: Arc::new(RwLock::new(HashMap::new())),
            sessions: Arc::new(RwLock::new(HashMap::new())),
            next_id: Arc::new(RwLock::new(1)),
        }
    }

    /// Create an orchestrator backed by a real Agent for LLM execution.
    ///
    /// SubAgents spawned by this orchestrator will run the actual agent
    /// definition (permissions, system prompt, model, max_steps) loaded from
    /// the agent's configuration and any extra `agent_dirs` provided in
    /// `SubAgentConfig`.
    pub fn from_agent(agent: Arc<crate::Agent>) -> Self {
        Self::from_agent_with_config(agent, OrchestratorConfig::default())
    }

    /// Create an orchestrator backed by a real Agent with custom config.
    pub fn from_agent_with_config(agent: Arc<crate::Agent>, config: OrchestratorConfig) -> Self {
        let (event_tx, _) = broadcast::channel(config.event_buffer_size);

        Self {
            config,
            agent: Some(agent),
            event_tx,
            subagents: Arc::new(RwLock::new(HashMap::new())),
            sessions: Arc::new(RwLock::new(HashMap::new())),
            next_id: Arc::new(RwLock::new(1)),
        }
    }

    /// 订阅所有 SubAgent 事件
    ///
    /// 返回一个接收器，可以接收所有 SubAgent 的事件。
    pub fn subscribe_all(&self) -> broadcast::Receiver<OrchestratorEvent> {
        self.event_tx.subscribe()
    }

    /// 订阅特定 SubAgent 的事件
    ///
    /// 返回一个过滤后的接收器，只接收指定 SubAgent 的事件。
    pub fn subscribe_subagent(&self, id: &str) -> SubAgentEventStream {
        let rx = self.event_tx.subscribe();
        SubAgentEventStream {
            rx,
            filter_id: id.to_string(),
        }
    }

    /// 启动新的 SubAgent
    ///
    /// 返回 SubAgent 句柄，可用于控制和查询状态。
    pub async fn spawn_subagent(&self, config: SubAgentConfig) -> Result<SubAgentHandle> {
        // 检查并发限制
        {
            let subagents = self.subagents.read().await;
            let active_count = subagents
                .values()
                .filter(|h| !h.state().is_terminal())
                .count();

            if active_count >= self.config.max_concurrent_subagents {
                return Err(anyhow::anyhow!(
                    "Maximum concurrent subagents ({}) reached",
                    self.config.max_concurrent_subagents
                )
                .into());
            }
        }

        // 生成 SubAgent ID
        let id = {
            let mut next_id = self.next_id.write().await;
            let id = format!("subagent-{}", *next_id);
            *next_id += 1;
            id
        };

        // 创建控制通道
        let (control_tx, control_rx) = tokio::sync::mpsc::channel(self.config.control_buffer_size);

        // 创建状态
        let state = Arc::new(RwLock::new(SubAgentState::Initializing));

        // 创建活动跟踪
        let activity = Arc::new(RwLock::new(SubAgentActivity::Idle));

        // 发布启动事件
        let _ = self.event_tx.send(OrchestratorEvent::SubAgentStarted {
            id: id.clone(),
            agent_type: config.agent_type.clone(),
            description: config.description.clone(),
            parent_id: config.parent_id.clone(),
            config: config.clone(),
        });

        // 创建 SubAgentWrapper 并启动执行
        let wrapper = crate::orchestrator::wrapper::SubAgentWrapper::new(
            id.clone(),
            config.clone(),
            self.agent.clone(),
            self.event_tx.clone(),
            control_rx,
            state.clone(),
            activity.clone(),
            Arc::clone(&self.sessions),
        );

        let task_handle = tokio::spawn(async move { wrapper.execute().await });

        // 创建句柄
        let handle = SubAgentHandle::new(
            id.clone(),
            config,
            control_tx,
            self.event_tx.clone(),
            state.clone(),
            activity.clone(),
            task_handle,
        );

        // 注册到 orchestrator
        self.subagents
            .write()
            .await
            .insert(id.clone(), handle.clone());

        Ok(handle)
    }

    /// Spawn a subagent from a unified `AgentSlot` declaration.
    ///
    /// Convenience wrapper around `spawn_subagent` that accepts the unified slot
    /// type.  The `role` field is ignored here — for team-based workflows use
    /// `run_team` instead.
    pub async fn spawn(&self, slot: AgentSlot) -> Result<SubAgentHandle> {
        self.spawn_subagent(SubAgentConfig::from(slot)).await
    }

    /// Run a goal through a Lead → Worker → Reviewer team built from `AgentSlot`s.
    ///
    /// Requires `from_agent()` mode — returns an error if no backing `Agent` is
    /// configured.  Each slot's `role` field determines its position in the team;
    /// slots without a role default to `Worker`.  Agent definitions are loaded
    /// from each slot's `agent_dirs` and looked up by `agent_type`.
    pub async fn run_team(
        &self,
        goal: impl Into<String>,
        workspace: impl Into<String>,
        slots: Vec<AgentSlot>,
    ) -> Result<crate::agent_teams::TeamRunResult> {
        let agent = self
            .agent
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("run_team requires a real Agent (use from_agent())"))?;

        let ws = workspace.into();
        let goal = goal.into();

        // Build a shared registry from all agent_dirs across every slot.
        let registry = crate::subagent::AgentRegistry::new();
        for slot in &slots {
            for dir in &slot.agent_dirs {
                for def in crate::subagent::load_agents_from_dir(std::path::Path::new(dir)) {
                    registry.register(def);
                }
            }
        }

        // Use wall-clock millis for a unique team name.
        let team_name = format!(
            "team-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        );

        let team = crate::agent_teams::AgentTeam::new(
            &team_name,
            crate::agent_teams::TeamConfig::default(),
        );
        let mut runner = crate::agent_teams::TeamRunner::new(team);

        for (i, slot) in slots.iter().enumerate() {
            let role = slot.role.unwrap_or(crate::agent_teams::TeamRole::Worker);
            let member_id = format!("{}-{}", role, i);
            runner.team_mut().add_member(&member_id, role);
            runner.bind_agent(&member_id, agent, &ws, &slot.agent_type, &registry)?;
        }

        runner.run_until_done(&goal).await
    }

    /// 发送控制信号到 SubAgent
    pub async fn send_control(&self, id: &str, signal: ControlSignal) -> Result<()> {
        let subagents = self.subagents.read().await;
        let handle = subagents
            .get(id)
            .ok_or_else(|| anyhow::anyhow!("SubAgent '{}' not found", id))?;

        handle.send_control(signal.clone()).await?;

        // 发布控制信号接收事件
        let _ = self
            .event_tx
            .send(OrchestratorEvent::ControlSignalReceived {
                id: id.to_string(),
                signal,
            });

        Ok(())
    }

    /// 暂停 SubAgent
    pub async fn pause_subagent(&self, id: &str) -> Result<()> {
        self.send_control(id, ControlSignal::Pause).await
    }

    /// 恢复 SubAgent
    pub async fn resume_subagent(&self, id: &str) -> Result<()> {
        self.send_control(id, ControlSignal::Resume).await
    }

    /// 取消 SubAgent
    pub async fn cancel_subagent(&self, id: &str) -> Result<()> {
        self.send_control(id, ControlSignal::Cancel).await
    }

    /// 调整 SubAgent 参数
    pub async fn adjust_subagent_params(
        &self,
        id: &str,
        max_steps: Option<usize>,
        timeout_ms: Option<u64>,
    ) -> Result<()> {
        self.send_control(
            id,
            ControlSignal::AdjustParams {
                max_steps,
                timeout_ms,
            },
        )
        .await
    }

    /// 获取 SubAgent 状态
    pub async fn get_subagent_state(&self, id: &str) -> Option<SubAgentState> {
        let subagents = self.subagents.read().await;
        subagents.get(id).map(|h| h.state())
    }

    /// 获取所有 SubAgent 的状态
    pub async fn get_all_states(&self) -> HashMap<String, SubAgentState> {
        let subagents = self.subagents.read().await;
        subagents
            .iter()
            .map(|(id, handle)| (id.clone(), handle.state()))
            .collect()
    }

    /// 获取活跃的 SubAgent 数量
    pub async fn active_count(&self) -> usize {
        let subagents = self.subagents.read().await;
        subagents
            .values()
            .filter(|h| !h.state().is_terminal())
            .count()
    }

    /// 等待所有 SubAgent 完成
    pub async fn wait_all(&self) -> Result<()> {
        loop {
            let active = self.active_count().await;
            if active == 0 {
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
        Ok(())
    }

    /// 获取所有 SubAgent 的信息列表
    pub async fn list_subagents(&self) -> Vec<SubAgentInfo> {
        let subagents = self.subagents.read().await;
        let mut infos = Vec::new();

        for (id, handle) in subagents.iter() {
            let state = handle.state_async().await;
            let activity = handle.activity().await;
            let config = handle.config();

            infos.push(SubAgentInfo {
                id: id.clone(),
                agent_type: config.agent_type.clone(),
                description: config.description.clone(),
                state: format!("{:?}", state),
                parent_id: config.parent_id.clone(),
                created_at: handle.created_at(),
                updated_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
                current_activity: Some(activity),
            });
        }

        infos
    }

    /// 获取特定 SubAgent 的详细信息
    pub async fn get_subagent_info(&self, id: &str) -> Option<SubAgentInfo> {
        let subagents = self.subagents.read().await;
        let handle = subagents.get(id)?;

        let state = handle.state_async().await;
        let activity = handle.activity().await;
        let config = handle.config();

        Some(SubAgentInfo {
            id: id.to_string(),
            agent_type: config.agent_type.clone(),
            description: config.description.clone(),
            state: format!("{:?}", state),
            parent_id: config.parent_id.clone(),
            created_at: handle.created_at(),
            updated_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            current_activity: Some(activity),
        })
    }

    /// 获取所有活跃 SubAgent 的当前活动
    pub async fn get_active_activities(&self) -> HashMap<String, SubAgentActivity> {
        let subagents = self.subagents.read().await;
        let mut activities = HashMap::new();

        for (id, handle) in subagents.iter() {
            if !handle.state().is_terminal() {
                let activity = handle.activity().await;
                activities.insert(id.clone(), activity);
            }
        }

        activities
    }

    /// 获取 SubAgent 句柄（用于直接控制）
    pub async fn get_handle(&self, id: &str) -> Option<SubAgentHandle> {
        let subagents = self.subagents.read().await;
        subagents.get(id).cloned()
    }

    /// Complete a pending external task for a SubAgent.
    ///
    /// Call this after processing an `OrchestratorEvent::ExternalTaskPending`
    /// event.  The `subagent_id` and `task_id` identify the waiting tool call;
    /// `result` is the outcome produced by the external worker.
    ///
    /// Returns `true` if the task was found and unblocked, `false` if the
    /// subagent or task ID was not found (e.g., already timed out).
    /// Return any external tasks currently waiting for the given SubAgent.
    ///
    /// Returns an empty list if the SubAgent does not exist or has no pending
    /// external tasks (e.g. when running with the default Internal lane mode).
    pub async fn pending_external_tasks_for(
        &self,
        subagent_id: &str,
    ) -> Vec<crate::queue::ExternalTask> {
        let sessions = self.sessions.read().await;
        match sessions.get(subagent_id) {
            Some(session) => session.pending_external_tasks().await,
            None => vec![],
        }
    }

    pub async fn complete_external_task(
        &self,
        subagent_id: &str,
        task_id: &str,
        result: crate::queue::ExternalTaskResult,
    ) -> bool {
        let sessions = self.sessions.read().await;
        match sessions.get(subagent_id) {
            Some(session) => session.complete_external_task(task_id, result).await,
            None => false,
        }
    }
}

impl std::fmt::Debug for AgentOrchestrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentOrchestrator")
            .field("event_buffer_size", &self.config.event_buffer_size)
            .field(
                "max_concurrent_subagents",
                &self.config.max_concurrent_subagents,
            )
            .finish()
    }
}

/// SubAgent 事件流（过滤特定 SubAgent 的事件）
pub struct SubAgentEventStream {
    rx: broadcast::Receiver<OrchestratorEvent>,
    filter_id: String,
}

impl SubAgentEventStream {
    /// 接收下一个事件
    pub async fn recv(&mut self) -> Option<OrchestratorEvent> {
        loop {
            match self.rx.recv().await {
                Ok(event) => {
                    if let Some(id) = event.subagent_id() {
                        if id == self.filter_id {
                            return Some(event);
                        }
                    }
                }
                Err(_) => return None,
            }
        }
    }
}
