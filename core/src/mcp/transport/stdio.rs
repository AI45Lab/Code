//! Stdio Transport for MCP
//!
//! Implements MCP transport over standard input/output for local process communication.

use super::McpTransport;
use crate::mcp::protocol::{JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpNotification};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot, RwLock};

/// Default request timeout for MCP tool calls
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 60;

/// Stdio transport for MCP servers
pub struct StdioTransport {
    /// Child process
    child: RwLock<Option<Child>>,
    /// Stdin writer
    stdin_tx: mpsc::Sender<String>,
    /// Pending requests (id -> response sender)
    pending: Arc<RwLock<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>>,
    /// Notification receiver
    notification_rx: RwLock<Option<mpsc::Receiver<McpNotification>>>,
    /// Connected flag
    connected: AtomicBool,
    /// Per-request timeout in seconds
    request_timeout_secs: u64,
}

impl StdioTransport {
    /// Create a new stdio transport by spawning a process
    pub async fn spawn(
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
    ) -> Result<Self> {
        Self::spawn_with_timeout(command, args, env, DEFAULT_REQUEST_TIMEOUT_SECS).await
    }

    /// Create a new stdio transport with a custom request timeout
    pub async fn spawn_with_timeout(
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
        request_timeout_secs: u64,
    ) -> Result<Self> {
        // Spawn the process
        let mut cmd = Command::new(command);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        // Add environment variables
        for (key, value) in env {
            cmd.env(key, value);
        }

        let mut child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn MCP server: {} {:?}", command, args))?;

        let stdin = child.stdin.take().ok_or_else(|| anyhow!("No stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("No stdout"))?;

        // Create channels
        let (stdin_tx, mut stdin_rx) = mpsc::channel::<String>(100);
        let (notification_tx, notification_rx) = mpsc::channel::<McpNotification>(100);
        let pending: Arc<RwLock<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>> =
            Arc::new(RwLock::new(HashMap::new()));

        // Spawn stdin writer task
        let mut stdin_writer = stdin;
        tokio::spawn(async move {
            while let Some(msg) = stdin_rx.recv().await {
                if let Err(e) = stdin_writer.write_all(msg.as_bytes()).await {
                    tracing::error!("Failed to write to MCP stdin: {}", e);
                    break;
                }
                if let Err(e) = stdin_writer.flush().await {
                    tracing::error!("Failed to flush MCP stdin: {}", e);
                    break;
                }
            }
        });

        // Spawn stdout reader task
        let pending_clone = pending.clone();
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();

            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) => {
                        tracing::debug!("MCP stdout closed");
                        break;
                    }
                    Ok(_) => {
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }

                        // Try to parse as response
                        if let Ok(response) = serde_json::from_str::<JsonRpcResponse>(trimmed) {
                            if let Some(id) = response.id {
                                let mut pending = pending_clone.write().await;
                                if let Some(tx) = pending.remove(&id) {
                                    let _ = tx.send(response);
                                }
                            }
                            continue;
                        }

                        // Try to parse as notification
                        if let Ok(notification) =
                            serde_json::from_str::<JsonRpcNotification>(trimmed)
                        {
                            let mcp_notif = McpNotification::from_json_rpc(&notification);
                            let _ = notification_tx.send(mcp_notif).await;
                            continue;
                        }

                        tracing::warn!("Unknown MCP message: {}", trimmed);
                    }
                    Err(e) => {
                        tracing::error!("Failed to read MCP stdout: {}", e);
                        break;
                    }
                }
            }
        });

        Ok(Self {
            child: RwLock::new(Some(child)),
            stdin_tx,
            pending,
            notification_rx: RwLock::new(Some(notification_rx)),
            connected: AtomicBool::new(true),
            request_timeout_secs,
        })
    }
}

#[async_trait]
impl McpTransport for StdioTransport {
    async fn request(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
        if !self.connected.load(Ordering::SeqCst) {
            return Err(anyhow!("Transport not connected"));
        }

        // Create response channel
        let (tx, rx) = oneshot::channel();
        let request_id = request.id;

        // Register pending request
        {
            let mut pending = self.pending.write().await;
            pending.insert(request_id, tx);
        }

        // Serialize and send request
        let msg = serde_json::to_string(&request)? + "\n";
        self.stdin_tx
            .send(msg)
            .await
            .map_err(|_| anyhow!("Failed to send request"))?;

        // Wait for response with timeout
        let response = match tokio::time::timeout(
            std::time::Duration::from_secs(self.request_timeout_secs),
            rx,
        )
        .await
        {
            Ok(Ok(resp)) => resp,
            Ok(Err(_)) => {
                // Channel closed — clean up pending entry
                self.pending.write().await.remove(&request_id);
                return Err(anyhow!("Response channel closed"));
            }
            Err(_) => {
                // Timeout — clean up pending entry to prevent memory leak
                self.pending.write().await.remove(&request_id);
                return Err(anyhow!(
                    "MCP request timed out after {}s",
                    self.request_timeout_secs
                ));
            }
        };

        Ok(response)
    }

    async fn notify(&self, notification: JsonRpcNotification) -> Result<()> {
        if !self.connected.load(Ordering::SeqCst) {
            return Err(anyhow!("Transport not connected"));
        }

        let msg = serde_json::to_string(&notification)? + "\n";
        self.stdin_tx
            .send(msg)
            .await
            .map_err(|_| anyhow!("Failed to send notification"))?;

        Ok(())
    }

    fn notifications(&self) -> mpsc::Receiver<McpNotification> {
        // This is a bit awkward - we need to take ownership of the receiver
        // In practice, this should only be called once
        let mut rx_guard = self.notification_rx.blocking_write();
        rx_guard.take().unwrap_or_else(|| {
            let (_, rx) = mpsc::channel(1);
            rx
        })
    }

    async fn close(&self) -> Result<()> {
        self.connected.store(false, Ordering::SeqCst);

        // Kill the child process
        let mut child_guard = self.child.write().await;
        if let Some(mut child) = child_guard.take() {
            let _ = child.kill().await;
        }

        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stdio_transport_spawn_invalid_command() {
        let result = StdioTransport::spawn("nonexistent_command_12345", &[], &HashMap::new()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_stdio_transport_spawn_echo() {
        // Use a simple command that exists on most systems
        let result = StdioTransport::spawn("cat", &[], &HashMap::new()).await;

        if let Ok(transport) = result {
            assert!(transport.is_connected());
            transport.close().await.unwrap();
            assert!(!transport.is_connected());
        }
        // If cat doesn't exist, that's fine - skip the test
    }

    #[tokio::test]
    async fn test_stdio_transport_is_connected_initial() {
        let result = StdioTransport::spawn("cat", &[], &HashMap::new()).await;
        if let Ok(transport) = result {
            assert!(transport.is_connected());
            let _ = transport.close().await;
        }
    }

    #[tokio::test]
    async fn test_stdio_transport_close_disconnects() {
        let result = StdioTransport::spawn("cat", &[], &HashMap::new()).await;
        if let Ok(transport) = result {
            assert!(transport.is_connected());
            transport.close().await.unwrap();
            assert!(!transport.is_connected());
        }
    }

    #[tokio::test]
    async fn test_stdio_transport_spawn_with_args() {
        let args = vec!["--version".to_string()];
        let result = StdioTransport::spawn("cat", &args, &HashMap::new()).await;
        // May fail depending on system, but should not panic
        let _ = result;
    }

    #[tokio::test]
    async fn test_stdio_transport_spawn_with_env() {
        let mut env = HashMap::new();
        env.insert("TEST_VAR".to_string(), "test_value".to_string());
        let result = StdioTransport::spawn("cat", &[], &env).await;
        if let Ok(transport) = result {
            let _ = transport.close().await;
        }
    }

    #[tokio::test]
    async fn test_stdio_transport_double_close() {
        let result = StdioTransport::spawn("cat", &[], &HashMap::new()).await;
        if let Ok(transport) = result {
            transport.close().await.unwrap();
            // Second close should not panic
            let result = transport.close().await;
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_stdio_transport_request_after_close() {
        let result = StdioTransport::spawn("cat", &[], &HashMap::new()).await;
        if let Ok(transport) = result {
            transport.close().await.unwrap();

            let request = JsonRpcRequest::new(1, "test", None);
            let result = transport.request(request).await;
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("not connected"));
        }
    }

    #[tokio::test]
    async fn test_stdio_transport_notify_after_close() {
        let result = StdioTransport::spawn("cat", &[], &HashMap::new()).await;
        if let Ok(transport) = result {
            transport.close().await.unwrap();

            let notification = JsonRpcNotification::new("test", None);
            let result = transport.notify(notification).await;
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("not connected"));
        }
    }

    #[test]
    fn test_json_rpc_request_creation() {
        let request =
            JsonRpcRequest::new(1, "test_method", Some(serde_json::json!({"key": "value"})));
        assert_eq!(request.id, 1);
        assert_eq!(request.method, "test_method");
        assert!(request.params.is_some());
    }

    #[test]
    fn test_json_rpc_notification_creation() {
        let notification = JsonRpcNotification::new("test_notification", None);
        assert_eq!(notification.method, "test_notification");
        assert!(notification.params.is_none());
    }

    #[tokio::test]
    async fn test_stdio_transport_custom_timeout() {
        // Spawn with a very short timeout (1 second)
        let result = StdioTransport::spawn_with_timeout("cat", &[], &HashMap::new(), 1).await;
        if let Ok(transport) = result {
            assert_eq!(transport.request_timeout_secs, 1);
            let _ = transport.close().await;
        }
    }

    #[tokio::test]
    async fn test_stdio_transport_default_timeout() {
        let result = StdioTransport::spawn("cat", &[], &HashMap::new()).await;
        if let Ok(transport) = result {
            assert_eq!(transport.request_timeout_secs, DEFAULT_REQUEST_TIMEOUT_SECS);
            let _ = transport.close().await;
        }
    }
}
