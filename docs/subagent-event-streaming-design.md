# SubAgent Event Streaming - 设计方案

## 问题描述

当前SubAgent（通过task工具或AgentTeam创建）的内部事件不会传播到父session，导致：
- 无法监控SubAgent的工具调用
- 无法看到SubAgent的LLM响应流
- 调试和监控困难

## 当前架构

### Task工具创建的SubAgent
```rust
// task.rs:191
let (output, success) = match agent_loop.execute(&[], &params.prompt, None).await {
    //                                                                    ^^^^ event_tx = None
```

### AgentTeam创建的SubAgent
```rust
// agent_teams.rs:500
impl AgentExecutor for AgentSession {
    async fn execute(&self, prompt: &str) -> Result<String> {
        let result = self.send(prompt, None).await?;  // 没有事件流
        Ok(result.text)
    }
}
```

## 核心问题

1. **类型不匹配**：
   - `ToolContext.agent_event_tx`: `Option<broadcast::Sender<AgentEvent>>`
   - `AgentLoop::execute` 需要: `Option<mpsc::Sender<AgentEvent>>`

2. **事件隔离**：SubAgent的事件被丢弃，不会传播到父session

## 解决方案

### 方案1：在AgentLoop中添加broadcast支持（推荐）

**优点**：
- 统一的事件传播机制
- 支持多个监听者
- 最小化代码改动

**实现**：
1. 在 `AgentLoop::execute` 中添加 `broadcast_tx` 参数
2. 在事件发送时同时发送到 `mpsc` 和 `broadcast` channel
3. 修改 `TaskExecutor` 传递父session的 `broadcast::Sender`

### 方案2：添加事件转发层

**优点**：
- 不改变现有API
- 可以过滤/转换事件

**缺点**：
- 增加复杂度
- 性能开销

### 方案3：统一使用broadcast::Sender

**优点**：
- 简化架构
- 天然支持多播

**缺点**：
- 需要大量重构
- 破坏现有API

## 推荐实现：方案1

### 步骤1：修改AgentLoop支持broadcast

```rust
// agent.rs
pub async fn execute(
    &self,
    history: &[Message],
    prompt: &str,
    event_tx: Option<mpsc::Sender<AgentEvent>>,
) -> Result<AgentResult> {
    self.execute_with_broadcast(history, prompt, event_tx, None).await
}

pub async fn execute_with_broadcast(
    &self,
    history: &[Message],
    prompt: &str,
    event_tx: Option<mpsc::Sender<AgentEvent>>,
    broadcast_tx: Option<broadcast::Sender<AgentEvent>>,
) -> Result<AgentResult> {
    // 发送事件到两个channel
    fn send_event(
        event: &AgentEvent,
        mpsc_tx: &Option<mpsc::Sender<AgentEvent>>,
        broadcast_tx: &Option<broadcast::Sender<AgentEvent>>,
    ) {
        if let Some(tx) = mpsc_tx {
            let _ = tx.send(event.clone()).await;
        }
        if let Some(tx) = broadcast_tx {
            let _ = tx.send(event.clone());
        }
    }

    // ... 在所有事件发送处调用 send_event
}
```

### 步骤2：修改TaskExecutor传递broadcast_tx

```rust
// task.rs
let agent_loop = AgentLoop::new(...);

let (output, success) = match agent_loop.execute_with_broadcast(
    &[],
    &params.prompt,
    None,  // mpsc_tx
    event_tx.clone(),  // broadcast_tx from parent
).await {
    Ok(result) => (result.text, true),
    Err(e) => (format!("Task failed: {}", e), false),
};
```

### 步骤3：修改AgentTeam支持事件流

```rust
// agent_teams.rs
#[async_trait::async_trait]
pub trait AgentExecutor: Send + Sync {
    async fn execute(&self, prompt: &str) -> Result<String>;

    // 新增：支持事件流的执行
    async fn execute_with_events(
        &self,
        prompt: &str,
        event_tx: Option<broadcast::Sender<AgentEvent>>,
    ) -> Result<String> {
        // 默认实现：忽略事件
        self.execute(prompt).await
    }
}

// AgentSession实现
#[async_trait::async_trait]
impl AgentExecutor for AgentSession {
    async fn execute(&self, prompt: &str) -> Result<String> {
        let result = self.send(prompt, None).await?;
        Ok(result.text)
    }

    async fn execute_with_events(
        &self,
        prompt: &str,
        event_tx: Option<broadcast::Sender<AgentEvent>>,
    ) -> Result<String> {
        // 使用内部API传递事件流
        let result = self.send_with_broadcast(prompt, event_tx).await?;
        Ok(result.text)
    }
}
```

### 步骤4：在AgentSession中添加broadcast支持

```rust
// agent_api.rs
impl AgentSession {
    pub async fn send_with_broadcast(
        &self,
        prompt: &str,
        broadcast_tx: Option<broadcast::Sender<AgentEvent>>,
    ) -> Result<AgentResult> {
        // 创建内部mpsc channel
        let (mpsc_tx, mut mpsc_rx) = mpsc::channel(100);

        // 启动转发任务
        if let Some(btx) = broadcast_tx {
            tokio::spawn(async move {
                while let Some(event) = mpsc_rx.recv().await {
                    let _ = btx.send(event);
                }
            });
        }

        // 执行agent loop
        self.agent_loop.execute(&self.history, prompt, Some(mpsc_tx)).await
    }
}
```

## 测试计划

1. **单元测试**：验证事件正确传播
2. **集成测试**：
   - Task工具创建的SubAgent事件可见
   - AgentTeam创建的SubAgent事件可见
   - 嵌套SubAgent事件可见
3. **性能测试**：确保事件传播不影响性能

## 向后兼容性

- 现有API保持不变
- 新增可选的broadcast参数
- 默认行为不变（不传递事件）
