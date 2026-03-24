# A3S Code Advanced Developer Manual

For core developers, architects, and advanced users

## Table of Contents

1. Internal Architecture
2. Advanced Configuration
3. Advanced Tool Development
4. Advanced Skill Programming
5. Advanced Hook System
6. Security Hardening
7. Performance Optimization
8. Production Deployment
9. Integration
10. Troubleshooting

# Chapter 1: Internal Architecture

## 1.1 Runtime Architecture

A3S Code uses multi-threaded architecture:
- Main Thread: HTTP API / WebSocket / Scheduled Tasks
- Worker Thread Pool: Multiple Sessions
- I/O Thread Pool: LLM Client / Tool Execution / File I/O

## 1.2 AgentLoop State Machine

```rust
pub enum LoopState {
    Idle,
    Planning,
    Executing,
    WaitingForLLM,
    Compacting,
    Error,
    Completed,
}
```

## 1.3 Circuit Breaker

```rust
pub struct CircuitBreaker {
    failure_threshold: u32,  // Default: 3
    reset_timeout: Duration, // Default: 30s
}
```

# Chapter 2: Advanced Configuration

## 2.1 Queue System Configuration

```hcl
queue {
  control_max_concurrency = 2
  query_max_concurrency = 10
  execute_max_concurrency = 5
  generate_max_concurrency = 1
  enable_metrics = true
  enable_dlq = true
}
```

## 2.2 LLM Client Configuration

```rust
pub struct LlmClientConfig {
    pool_size: usize,             // Default: 10
    connection_timeout: Duration, // Default: 30s
    request_timeout: Duration,    // Default: 120s
}
```

## 2.3 Memory Limits

```rust
pub struct MemoryLimits {
    max_session_memory_mb: usize, // Default: 100
    max_message_history: usize,   // Default: 100
    max_context_tokens: usize,    // Default: 8000
}
```

# Chapter 3: Advanced Tool Development

## 3.1 Tool Lifecycle

Register -> Initialize -> Validate Input -> Pre-execute Hook -> Execute -> Post-execute Hook -> Cleanup

## 3.2 Advanced Tool Trait

```rust
#[async_trait]
pub trait AdvancedTool: Tool {
    async fn initialize(&mut self, config: &ToolConfig) -> Result<()>;
    fn validate_input(&self, input: &Value) -> Result<()>;
    fn pre_execute(&self, ctx: &Context) -> Result<PreExecuteAction>;
    async fn execute_async(&self, input: ToolInput) -> Result<ToolOutput>;
    fn post_execute(&self, output: &ToolOutput) -> Result<()>;
    async fn cleanup(&mut self) -> Result<()>;
}
```

## 3.3 Async Tool Example

```rust
use async_trait::async_trait;

pub struct AsyncWebTool {
    client: reqwest::Client,
    rate_limiter: RateLimiter,
}

#[async_trait]
impl Tool for AsyncWebTool {
    fn name(&self) -> &str { "async_web_fetch" }
    
    async fn execute(&self, input: ToolInput) -> Result<ToolOutput> {
        self.rate_limiter.acquire().await?;
        let response = self.client
            .get(input.get("url")?)
            .timeout(Duration::from_secs(30))
            .send().await?;
        Ok(ToolOutput::new(response.text().await?))
    }
}
```

# Chapter 4: Advanced Skill Programming

## 4.1 Skill Parsing Flow

1. Frontmatter parsing (YAML)
2. Content extraction (Markdown)
3. Template compilation
4. Permission validation
5. Injection into system prompt

## 4.2 Programmatic Skill Generation

```rust
pub struct SkillBuilder {
    name: String,
    description: String,
    allowed_tools: Vec<String>,
    content: String,
}

impl SkillBuilder {
    pub fn new(name: &str) -> Self;
    pub fn with_description(mut self, desc: &str) -> Self;
    pub fn with_tool(mut self, tool: &str) -> Self;
    pub fn build(self) -> Skill;
}
```

# Chapter 5: Advanced Hook System

## 5.1 Hook Chains

```rust
pub struct HookChain {
    hooks: Vec<(u32, Box<dyn HookHandler>)>,
}

impl HookChain {
    pub fn register(&mut self, priority: u32, hook: Box<dyn HookHandler>);
    pub async fn execute(&self, event: Event) -> HookResult;
}
```

## 5.2 Conditional Hooks

```rust
pub struct ConditionalHook {
    condition: Box<dyn Fn(&Event) -> bool>,
    hook: Box<dyn HookHandler>,
}
```

# Chapter 6: Security Hardening

## 6.1 AHP Integration

```rust
pub struct AHPConfig {
    enabled: bool,
    harness_endpoint: String,
    timeout: Duration,
    retry_policy: RetryPolicy,
}
```

## 6.2 Sandboxing

```rust
pub struct SandboxConfig {
    enabled: bool,
    chroot_path: Option<PathBuf>,
    network_access: bool,
    allowed_paths: Vec<PathBuf>,
    forbidden_commands: Vec<String>,
}
```

# Chapter 7: Performance Optimization

## 7.1 Token Usage Optimization

- Use compact context strategies
- Enable message summarization
- Set appropriate token limits

## 7.2 Caching Strategies

```rust
pub struct CacheConfig {
    enabled: bool,
    backend: CacheBackend,
    ttl: Duration,
    max_size: usize,
}

pub enum CacheBackend {
    InMemory,
    Redis(String),
    Disk(PathBuf),
}
```

# Chapter 8: Production Deployment

## 8.1 Docker Deployment

```dockerfile
FROM rust:1.75-slim as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl3
COPY --from=builder /app/target/release/a3s-code /usr/local/bin/
ENTRYPOINT ["a3s-code"]
```

## 8.2 Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: a3s-code
spec:
  replicas: 3
  selector:
    matchLabels:
      app: a3s-code
  template:
    metadata:
      labels:
        app: a3s-code
    spec:
      containers:
      - name: a3s-code
        image: a3s-lab/code:latest
        resources:
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

# Chapter 9: Integration

## 9.1 MCP Protocol

```rust
pub struct MCPConfig {
    enabled: bool,
    server_url: String,
    capabilities: Vec<MCPCapability>,
}
```

## 9.2 Custom Storage Backend

```rust
pub trait CustomStorage: Send + Sync {
    async fn save(&self, key: &str, value: &[u8]) -> Result<()>;
    async fn load(&self, key: &str) -> Result<Option<Vec<u8>>>;
    async fn delete(&self, key: &str) -> Result<()>;
}
```

# Chapter 10: Troubleshooting

## 10.1 Debug Mode

```bash
export RUST_LOG=debug
export A3S_DEBUG=1
```

## 10.2 Common Issues

| Issue | Solution |
|-------|----------|
| High memory usage | Reduce max_context_tokens |
| Slow LLM responses | Check connection pool size |
| Tool timeout | Increase timeout in config |

---

End of Advanced Developer Manual
