//! SubAgent 句柄

use crate::error::Result;
use crate::orchestrator::{ControlSignal, SubAgentActivity, SubAgentConfig, SubAgentState};
use std::sync::Arc;
use tokio::sync::RwLock;

/// SubAgent 句柄
///
/// 用于控制和查询 SubAgent 的状态。
#[derive(Clone)]
pub struct SubAgentHandle {
    /// SubAgent ID
    pub id: String,

    /// SubAgent 配置
    pub(crate) config: SubAgentConfig,

    /// 创建时间（Unix 时间戳，毫秒）
    pub(crate) created_at: u64,

    /// 控制信号发送器
    control_tx: tokio::sync::mpsc::Sender<ControlSignal>,

    /// 事件广播发送器
    event_tx: tokio::sync::broadcast::Sender<OrchestratorEvent>,

    /// 状态
    state: Arc<RwLock<SubAgentState>>,

    /// 当前活动
    pub(crate) activity: Arc<RwLock<SubAgentActivity>>,

    /// 任务句柄（保持任务存活，不直接访问）
    #[allow(dead_code)]
    task_handle: Arc<tokio::task::JoinHandle<Result<String>>>,
}

impl SubAgentHandle {
    /// 创建新的句柄
    pub(crate) fn new(
        id: String,
        config: SubAgentConfig,
        control_tx: tokio::sync::mpsc::Sender<ControlSignal>,
        event_tx: tokio::sync::broadcast::Sender<OrchestratorEvent>,
        state: Arc<RwLock<SubAgentState>>,
        activity: Arc<RwLock<SubAgentActivity>>,
        task_handle: tokio::task::JoinHandle<Result<String>>,
    ) -> Self {
        Self {
            id,
            config,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            control_tx,
            event_tx,
            state,
            activity,
            task_handle: Arc::new(task_handle),
        }
    }

    /// 获取当前状态
    ///
    /// 注意：此方法使用非阻塞读取。如果状态锁当前被持有，返回 Initializing 状态。
    /// 在异步上下文中，建议使用 `state_async()` 以获得准确的状态。
    pub fn state(&self) -> SubAgentState {
        self.state
            .try_read()
            .map(|guard| guard.clone())
            .unwrap_or(SubAgentState::Initializing)
    }

    /// 异步获取当前状态
    pub async fn state_async(&self) -> SubAgentState {
        self.state.read().await.clone()
    }

    /// 获取当前活动
    pub async fn activity(&self) -> SubAgentActivity {
        self.activity.read().await.clone()
    }

    /// 获取配置
    pub fn config(&self) -> &SubAgentConfig {
        &self.config
    }

    /// 获取创建时间
    pub fn created_at(&self) -> u64 {
        self.created_at
    }

    /// 发送控制信号
    pub async fn send_control(&self, signal: ControlSignal) -> Result<()> {
        self.control_tx
            .send(signal)
            .await
            .map_err(|_| anyhow::anyhow!("Failed to send control signal: channel closed"))?;
        Ok(())
    }

    /// 暂停执行
    pub async fn pause(&self) -> Result<()> {
        self.send_control(ControlSignal::Pause).await
    }

    /// 恢复执行
    pub async fn resume(&self) -> Result<()> {
        self.send_control(ControlSignal::Resume).await
    }

    /// 取消执行
    pub async fn cancel(&self) -> Result<()> {
        self.send_control(ControlSignal::Cancel).await
    }

    /// 调整参数
    pub async fn adjust_params(
        &self,
        max_steps: Option<usize>,
        timeout_ms: Option<u64>,
    ) -> Result<()> {
        self.send_control(ControlSignal::AdjustParams {
            max_steps,
            timeout_ms,
        })
        .await
    }

    /// 注入新提示词
    pub async fn inject_prompt(&self, prompt: impl Into<String>) -> Result<()> {
        self.send_control(ControlSignal::InjectPrompt {
            prompt: prompt.into(),
        })
        .await
    }

    /// 等待完成
    pub async fn wait(&self) -> Result<String> {
        // Note: 这里需要克隆 Arc 来避免移动所有权问题
        // 实际使用中，应该只调用一次 wait()
        loop {
            let state = self.state_async().await;
            if state.is_terminal() {
                match state {
                    SubAgentState::Completed { output, .. } => return Ok(output),
                    SubAgentState::Cancelled => {
                        return Err(anyhow::anyhow!("SubAgent was cancelled").into())
                    }
                    SubAgentState::Error { message } => {
                        return Err(anyhow::anyhow!("SubAgent error: {}", message).into())
                    }
                    _ => unreachable!(),
                }
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
    }

    /// 是否已完成
    pub fn is_done(&self) -> bool {
        self.state().is_terminal()
    }

    /// 是否正在运行
    pub fn is_running(&self) -> bool {
        self.state().is_running()
    }

    /// 是否已暂停
    pub fn is_paused(&self) -> bool {
        self.state().is_paused()
    }

    /// Subscribe to events for this SubAgent.
    ///
    /// Returns a filtered event stream that only includes events for this SubAgent.
    pub fn events(&self) -> crate::orchestrator::SubAgentEventStream {
        let rx = self.event_tx.subscribe();
        crate::orchestrator::SubAgentEventStream {
            rx,
            filter_id: self.id.clone(),
        }
    }
}

impl std::fmt::Debug for SubAgentHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubAgentHandle")
            .field("id", &self.id)
            .field("state", &self.state())
            .finish()
    }
}
