//! MCP Protocol Type Definitions
//!
//! Defines the core types for the Model Context Protocol (MCP).
//! Based on the MCP specification: <https://spec.modelcontextprotocol.io/>

use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;

/// MCP protocol version
pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// JSON-RPC request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: u64,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl JsonRpcRequest {
    pub fn new(id: u64, method: &str, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.to_string(),
            params,
        }
    }
}

/// JSON-RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// JSON-RPC notification (no id)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcNotification {
    pub jsonrpc: String,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl JsonRpcNotification {
    pub fn new(method: &str, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params,
        }
    }
}

// ============================================================================
// MCP Initialize
// ============================================================================

/// Client capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClientCapabilities {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roots: Option<RootsCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling: Option<SamplingCapability>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RootsCapability {
    #[serde(default)]
    pub list_changed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SamplingCapability {}

/// Client info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    pub name: String,
    pub version: String,
}

/// Initialize request params
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    pub protocol_version: String,
    pub capabilities: ClientCapabilities,
    pub client_info: ClientInfo,
}

/// Server capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerCapabilities {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourcesCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PromptsCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logging: Option<LoggingCapability>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolsCapability {
    #[serde(default)]
    pub list_changed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourcesCapability {
    #[serde(default)]
    pub subscribe: bool,
    #[serde(default)]
    pub list_changed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptsCapability {
    #[serde(default)]
    pub list_changed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoggingCapability {}

/// Server info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

/// Initialize result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    pub protocol_version: String,
    pub capabilities: ServerCapabilities,
    pub server_info: ServerInfo,
}

// ============================================================================
// MCP Tools
// ============================================================================

/// MCP tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

/// List tools result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListToolsResult {
    pub tools: Vec<McpTool>,
}

/// Call tool params
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallToolParams {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<serde_json::Value>,
}

/// Tool content types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolContent {
    Text {
        text: String,
    },
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    Resource {
        resource: ResourceContent,
    },
}

/// Resource content
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceContent {
    pub uri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}

/// Call tool result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CallToolResult {
    pub content: Vec<ToolContent>,
    #[serde(default)]
    pub is_error: bool,
}

// ============================================================================
// MCP Resources
// ============================================================================

/// MCP resource definition
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpResource {
    pub uri: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// List resources result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListResourcesResult {
    pub resources: Vec<McpResource>,
}

/// Read resource params
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResourceParams {
    pub uri: String,
}

/// Read resource result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResourceResult {
    pub contents: Vec<ResourceContent>,
}

// ============================================================================
// MCP Prompts
// ============================================================================

/// MCP prompt definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPrompt {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Vec<PromptArgument>>,
}

/// Prompt argument
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptArgument {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub required: bool,
}

/// List prompts result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListPromptsResult {
    pub prompts: Vec<McpPrompt>,
}

// ============================================================================
// MCP Notifications
// ============================================================================

/// MCP notification types
#[derive(Debug, Clone)]
pub enum McpNotification {
    ToolsListChanged,
    ResourcesListChanged,
    PromptsListChanged,
    Progress {
        progress_token: String,
        progress: f64,
        total: Option<f64>,
    },
    Log {
        level: String,
        logger: Option<String>,
        data: serde_json::Value,
    },
    Unknown {
        method: String,
        params: Option<serde_json::Value>,
    },
}

impl McpNotification {
    pub fn from_json_rpc(notification: &JsonRpcNotification) -> Self {
        match notification.method.as_str() {
            "notifications/tools/list_changed" => McpNotification::ToolsListChanged,
            "notifications/resources/list_changed" => McpNotification::ResourcesListChanged,
            "notifications/prompts/list_changed" => McpNotification::PromptsListChanged,
            "notifications/progress" => {
                if let Some(params) = &notification.params {
                    let progress_token = params
                        .get("progressToken")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let progress = params
                        .get("progress")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let total = params.get("total").and_then(|v| v.as_f64());
                    McpNotification::Progress {
                        progress_token,
                        progress,
                        total,
                    }
                } else {
                    McpNotification::Unknown {
                        method: notification.method.clone(),
                        params: notification.params.clone(),
                    }
                }
            }
            "notifications/message" => {
                if let Some(params) = &notification.params {
                    let level = params
                        .get("level")
                        .and_then(|v| v.as_str())
                        .unwrap_or("info")
                        .to_string();
                    let logger = params
                        .get("logger")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let data = params
                        .get("data")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    McpNotification::Log {
                        level,
                        logger,
                        data,
                    }
                } else {
                    McpNotification::Unknown {
                        method: notification.method.clone(),
                        params: notification.params.clone(),
                    }
                }
            }
            _ => McpNotification::Unknown {
                method: notification.method.clone(),
                params: notification.params.clone(),
            },
        }
    }
}

// ============================================================================
// Configuration Types
// ============================================================================

/// MCP server configuration
#[derive(Debug, Clone, Serialize)]
pub struct McpServerConfig {
    /// Server name (used for tool prefix)
    pub name: String,
    /// Transport configuration
    pub transport: McpTransportConfig,
    /// Whether enabled
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Environment variables
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// OAuth configuration (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oauth: Option<OAuthConfig>,
    /// Per-tool execution timeout in seconds (default: 60)
    #[serde(default = "default_tool_timeout")]
    pub tool_timeout_secs: u64,
}

impl<'de> Deserialize<'de> for McpServerConfig {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        use serde_json::Value;

        let mut map = serde_json::Map::deserialize(deserializer)?;

        // Build transport from flat HCL fields (transport = "stdio", command = "...", args = [...])
        // or from a nested transport object ({ type = "stdio", command = "..." })
        let transport = if let Some(t) = map.remove("transport") {
            match &t {
                Value::String(kind) => {
                    // Flat HCL format: transport = "stdio", command = "...", args = [...]
                    match kind.as_str() {
                        "stdio" => {
                            let command = map
                                .remove("command")
                                .and_then(|v| v.as_str().map(String::from))
                                .ok_or_else(|| D::Error::missing_field("command"))?;
                            let args = map
                                .remove("args")
                                .and_then(|v| serde_json::from_value(v).ok())
                                .unwrap_or_default();
                            McpTransportConfig::Stdio { command, args }
                        }
                        "http" => {
                            let url = map
                                .remove("url")
                                .and_then(|v| v.as_str().map(String::from))
                                .ok_or_else(|| D::Error::missing_field("url"))?;
                            let headers = map
                                .remove("headers")
                                .and_then(|v| serde_json::from_value(v).ok())
                                .unwrap_or_default();
                            McpTransportConfig::Http { url, headers }
                        }
                        "streamable-http" | "streamable_http" => {
                            let url = map
                                .remove("url")
                                .and_then(|v| v.as_str().map(String::from))
                                .ok_or_else(|| D::Error::missing_field("url"))?;
                            let headers = map
                                .remove("headers")
                                .and_then(|v| serde_json::from_value(v).ok())
                                .unwrap_or_default();
                            McpTransportConfig::StreamableHttp { url, headers }
                        }
                        other => {
                            return Err(D::Error::unknown_variant(
                                other,
                                &["stdio", "http", "streamable-http"],
                            ));
                        }
                    }
                }
                // Nested object format: transport { type = "stdio", command = "..." }
                Value::Object(_) => serde_json::from_value(t).map_err(D::Error::custom)?,
                _ => return Err(D::Error::custom("transport must be a string or object")),
            }
        } else {
            return Err(D::Error::missing_field("transport"));
        };

        let name = map
            .remove("name")
            .and_then(|v| v.as_str().map(String::from))
            .ok_or_else(|| D::Error::missing_field("name"))?;
        let enabled = map
            .remove("enabled")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let env = map
            .remove("env")
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default();
        let oauth = map
            .remove("oauth")
            .and_then(|v| serde_json::from_value(v).ok());
        let tool_timeout_secs = map
            .remove("tool_timeout_secs")
            .or_else(|| map.remove("toolTimeoutSecs"))
            .and_then(|v| v.as_u64())
            .unwrap_or(60);

        Ok(McpServerConfig {
            name,
            transport,
            enabled,
            env,
            oauth,
            tool_timeout_secs,
        })
    }
}

#[allow(dead_code)] // used by serde default = "default_tool_timeout"
fn default_tool_timeout() -> u64 {
    60
}

#[allow(dead_code)]
fn default_true() -> bool {
    true
}

/// Transport configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum McpTransportConfig {
    /// Local process (stdio)
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
    },
    /// Remote HTTP + SSE (legacy, pre-2025-03-26)
    Http {
        url: String,
        #[serde(default)]
        headers: HashMap<String, String>,
    },
    /// Streamable HTTP (MCP 2025-03-26 spec)
    ///
    /// Single endpoint handles all communication.
    /// POST with `Accept: application/json, text/event-stream`.
    StreamableHttp {
        url: String,
        #[serde(default)]
        headers: HashMap<String, String>,
    },
}

/// OAuth configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthConfig {
    pub auth_url: String,
    pub token_url: String,
    pub client_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_secret: Option<String>,
    #[serde(default)]
    pub scopes: Vec<String>,
    pub redirect_uri: String,
    /// Static access token — if set, skips the OAuth exchange flow.
    /// Useful for long-lived tokens or service accounts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub access_token: Option<String>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_rpc_request_serialize() {
        let req = JsonRpcRequest::new(1, "initialize", Some(serde_json::json!({"test": true})));
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"id\":1"));
        assert!(json.contains("\"method\":\"initialize\""));
    }

    #[test]
    fn test_json_rpc_response_deserialize() {
        let json = r#"{"jsonrpc":"2.0","id":1,"result":{"success":true}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, Some(1));
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_json_rpc_error_deserialize() {
        let json =
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32600,"message":"Invalid Request"}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert!(resp.error.is_some());
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32600);
    }

    #[test]
    fn test_mcp_tool_deserialize() {
        let json = r#"{
            "name": "create_issue",
            "description": "Create a GitHub issue",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"}
                },
                "required": ["title"]
            }
        }"#;
        let tool: McpTool = serde_json::from_str(json).unwrap();
        assert_eq!(tool.name, "create_issue");
        assert!(tool.description.is_some());
    }

    #[test]
    fn test_tool_content_text() {
        let content = ToolContent::Text {
            text: "Hello".to_string(),
        };
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"text\":\"Hello\""));
    }

    #[test]
    fn test_mcp_transport_config_stdio() {
        let json = r#"{
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"]
        }"#;
        let config: McpTransportConfig = serde_json::from_str(json).unwrap();
        match config {
            McpTransportConfig::Stdio { command, args } => {
                assert_eq!(command, "npx");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected Stdio transport"),
        }
    }

    #[test]
    fn test_mcp_transport_config_http() {
        let json = r#"{
            "type": "http",
            "url": "https://mcp.example.com/api",
            "headers": {"Authorization": "Bearer token"}
        }"#;
        let config: McpTransportConfig = serde_json::from_str(json).unwrap();
        match config {
            McpTransportConfig::Http { url, headers } => {
                assert_eq!(url, "https://mcp.example.com/api");
                assert!(headers.contains_key("Authorization"));
            }
            _ => panic!("Expected Http transport"),
        }
    }

    #[test]
    fn test_mcp_notification_parse() {
        let notification = JsonRpcNotification::new("notifications/tools/list_changed", None);
        let mcp_notif = McpNotification::from_json_rpc(&notification);
        match mcp_notif {
            McpNotification::ToolsListChanged => {}
            _ => panic!("Expected ToolsListChanged"),
        }
    }
    #[test]
    fn test_json_rpc_request_new_with_params() {
        let req = JsonRpcRequest::new(1, "initialize", Some(serde_json::json!({"test": true})));
        assert_eq!(req.jsonrpc, "2.0");
        assert_eq!(req.id, 1);
        assert_eq!(req.method, "initialize");
        assert!(req.params.is_some());
    }

    #[test]
    fn test_json_rpc_request_new_without_params() {
        let req = JsonRpcRequest::new(2, "ping", None);
        assert_eq!(req.jsonrpc, "2.0");
        assert_eq!(req.id, 2);
        assert_eq!(req.method, "ping");
        assert!(req.params.is_none());
    }

    #[test]
    fn test_json_rpc_request_serialization() {
        let req = JsonRpcRequest::new(1, "test_method", Some(serde_json::json!({"key": "value"})));
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"id\":1"));
        assert!(json.contains("\"method\":\"test_method\""));
        assert!(json.contains("\"params\""));
    }

    #[test]
    fn test_json_rpc_response_with_result() {
        let resp = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: Some(1),
            result: Some(serde_json::json!({"success": true})),
            error: None,
        };
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_json_rpc_response_with_error() {
        let resp = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: Some(1),
            result: None,
            error: Some(JsonRpcError {
                code: -32600,
                message: "Invalid Request".to_string(),
                data: None,
            }),
        };
        assert!(resp.result.is_none());
        assert!(resp.error.is_some());
    }

    #[test]
    fn test_json_rpc_response_both_none() {
        let resp = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: Some(1),
            result: None,
            error: None,
        };
        assert!(resp.result.is_none());
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_json_rpc_response_serialization() {
        let resp = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: Some(1),
            result: Some(serde_json::json!({"data": "test"})),
            error: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"id\":1"));
        assert!(json.contains("\"result\""));
    }

    #[test]
    fn test_json_rpc_notification_new_with_params() {
        let notif =
            JsonRpcNotification::new("notification", Some(serde_json::json!({"msg": "hello"})));
        assert_eq!(notif.jsonrpc, "2.0");
        assert_eq!(notif.method, "notification");
        assert!(notif.params.is_some());
    }

    #[test]
    fn test_json_rpc_notification_new_without_params() {
        let notif = JsonRpcNotification::new("ping", None);
        assert_eq!(notif.jsonrpc, "2.0");
        assert_eq!(notif.method, "ping");
        assert!(notif.params.is_none());
    }

    #[test]
    fn test_json_rpc_notification_serialization() {
        let notif = JsonRpcNotification::new(
            "test_notification",
            Some(serde_json::json!({"key": "value"})),
        );
        let json = serde_json::to_string(&notif).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"method\":\"test_notification\""));
        assert!(!json.contains("\"id\""));
    }

    #[test]
    fn test_mcp_tool_serialize() {
        let tool = McpTool {
            name: "test_tool".to_string(),
            description: Some("A test tool".to_string()),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("\"name\":\"test_tool\""));
        assert!(json.contains("\"description\":\"A test tool\""));
    }

    #[test]
    fn test_mcp_tool_without_description() {
        let json = r#"{"name":"tool","inputSchema":{"type":"object"}}"#;
        let tool: McpTool = serde_json::from_str(json).unwrap();
        assert_eq!(tool.name, "tool");
        assert!(tool.description.is_none());
    }

    #[test]
    fn test_mcp_resource_serialize() {
        let resource = McpResource {
            uri: "file:///test.txt".to_string(),
            name: "test.txt".to_string(),
            description: Some("Test file".to_string()),
            mime_type: Some("text/plain".to_string()),
        };
        let json = serde_json::to_string(&resource).unwrap();
        assert!(json.contains("\"uri\":\"file:///test.txt\""));
        assert!(json.contains("\"name\":\"test.txt\""));
    }

    #[test]
    fn test_mcp_resource_deserialize() {
        let json = r#"{"uri":"file:///doc.md","name":"doc.md","mimeType":"text/markdown"}"#;
        let resource: McpResource = serde_json::from_str(json).unwrap();
        assert_eq!(resource.uri, "file:///doc.md");
        assert_eq!(resource.name, "doc.md");
        assert_eq!(resource.mime_type, Some("text/markdown".to_string()));
    }

    #[test]
    fn test_initialize_params_serialization() {
        let params = InitializeParams {
            protocol_version: PROTOCOL_VERSION.to_string(),
            capabilities: ClientCapabilities::default(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
        };
        let json = serde_json::to_string(&params).unwrap();
        assert!(json.contains("\"protocolVersion\""));
        assert!(json.contains("\"clientInfo\""));
    }

    #[test]
    fn test_initialize_result_serialization() {
        let result = InitializeResult {
            protocol_version: PROTOCOL_VERSION.to_string(),
            capabilities: ServerCapabilities::default(),
            server_info: ServerInfo {
                name: "test-server".to_string(),
                version: "1.0.0".to_string(),
            },
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"protocolVersion\""));
        assert!(json.contains("\"serverInfo\""));
    }

    #[test]
    fn test_call_tool_params_serialization() {
        let params = CallToolParams {
            name: "test_tool".to_string(),
            arguments: Some(serde_json::json!({"arg1": "value1"})),
        };
        let json = serde_json::to_string(&params).unwrap();
        assert!(json.contains("\"name\":\"test_tool\""));
        assert!(json.contains("\"arguments\""));
    }

    #[test]
    fn test_call_tool_params_without_arguments() {
        let params = CallToolParams {
            name: "simple_tool".to_string(),
            arguments: None,
        };
        let json = serde_json::to_string(&params).unwrap();
        assert!(json.contains("\"name\":\"simple_tool\""));
        assert!(!json.contains("\"arguments\""));
    }

    #[test]
    fn test_call_tool_result_serialization() {
        let result = CallToolResult {
            content: vec![ToolContent::Text {
                text: "Result".to_string(),
            }],
            is_error: false,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"content\""));
        assert!(json.contains("\"isError\":false"));
    }

    #[test]
    fn test_call_tool_result_error_flag() {
        let result = CallToolResult {
            content: vec![ToolContent::Text {
                text: "Error occurred".to_string(),
            }],
            is_error: true,
        };
        assert!(result.is_error);
    }

    #[test]
    fn test_call_tool_result_default() {
        let json = r#"{"content":[]}"#;
        let result: CallToolResult = serde_json::from_str(json).unwrap();
        assert!(!result.is_error);
    }

    #[test]
    fn test_read_resource_params_serialization() {
        let params = ReadResourceParams {
            uri: "file:///test.txt".to_string(),
        };
        let json = serde_json::to_string(&params).unwrap();
        assert!(json.contains("\"uri\":\"file:///test.txt\""));
    }

    #[test]
    fn test_read_resource_result_serialization() {
        let result = ReadResourceResult {
            contents: vec![ResourceContent {
                uri: "file:///test.txt".to_string(),
                mime_type: Some("text/plain".to_string()),
                text: Some("Hello".to_string()),
                blob: None,
            }],
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"contents\""));
        assert!(json.contains("\"uri\""));
    }

    #[test]
    fn test_list_tools_result_serialization() {
        let result = ListToolsResult {
            tools: vec![McpTool {
                name: "tool1".to_string(),
                description: None,
                input_schema: serde_json::json!({"type": "object"}),
            }],
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"tools\""));
    }

    #[test]
    fn test_list_resources_result_serialization() {
        let result = ListResourcesResult {
            resources: vec![McpResource {
                uri: "file:///test.txt".to_string(),
                name: "test.txt".to_string(),
                description: None,
                mime_type: None,
            }],
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"resources\""));
    }

    #[test]
    fn test_server_capabilities_default() {
        let caps = ServerCapabilities::default();
        assert!(caps.tools.is_none());
        assert!(caps.resources.is_none());
        assert!(caps.prompts.is_none());
        assert!(caps.logging.is_none());
    }

    #[test]
    fn test_server_capabilities_all_fields() {
        let caps = ServerCapabilities {
            tools: Some(ToolsCapability { list_changed: true }),
            resources: Some(ResourcesCapability {
                subscribe: true,
                list_changed: true,
            }),
            prompts: Some(PromptsCapability { list_changed: true }),
            logging: Some(LoggingCapability {}),
        };
        assert!(caps.tools.is_some());
        assert!(caps.resources.is_some());
        assert!(caps.prompts.is_some());
        assert!(caps.logging.is_some());
    }

    #[test]
    fn test_client_capabilities_default() {
        let caps = ClientCapabilities::default();
        assert!(caps.roots.is_none());
        assert!(caps.sampling.is_none());
    }

    #[test]
    fn test_client_capabilities_all_fields() {
        let caps = ClientCapabilities {
            roots: Some(RootsCapability { list_changed: true }),
            sampling: Some(SamplingCapability {}),
        };
        assert!(caps.roots.is_some());
        assert!(caps.sampling.is_some());
    }

    #[test]
    fn test_mcp_notification_tools_list_changed() {
        let notif = JsonRpcNotification::new("notifications/tools/list_changed", None);
        let mcp_notif = McpNotification::from_json_rpc(&notif);
        match mcp_notif {
            McpNotification::ToolsListChanged => {}
            _ => panic!("Expected ToolsListChanged"),
        }
    }

    #[test]
    fn test_mcp_notification_resources_list_changed() {
        let notif = JsonRpcNotification::new("notifications/resources/list_changed", None);
        let mcp_notif = McpNotification::from_json_rpc(&notif);
        match mcp_notif {
            McpNotification::ResourcesListChanged => {}
            _ => panic!("Expected ResourcesListChanged"),
        }
    }

    #[test]
    fn test_mcp_notification_prompts_list_changed() {
        let notif = JsonRpcNotification::new("notifications/prompts/list_changed", None);
        let mcp_notif = McpNotification::from_json_rpc(&notif);
        match mcp_notif {
            McpNotification::PromptsListChanged => {}
            _ => panic!("Expected PromptsListChanged"),
        }
    }

    #[test]
    fn test_mcp_notification_progress() {
        let notif = JsonRpcNotification::new(
            "notifications/progress",
            Some(serde_json::json!({
                "progressToken": "token-123",
                "progress": 50.0,
                "total": 100.0
            })),
        );
        let mcp_notif = McpNotification::from_json_rpc(&notif);
        match mcp_notif {
            McpNotification::Progress {
                progress_token,
                progress,
                total,
            } => {
                assert_eq!(progress_token, "token-123");
                assert_eq!(progress, 50.0);
                assert_eq!(total, Some(100.0));
            }
            _ => panic!("Expected Progress"),
        }
    }

    #[test]
    fn test_mcp_notification_log() {
        let notif = JsonRpcNotification::new(
            "notifications/message",
            Some(serde_json::json!({
                "level": "error",
                "logger": "test-logger",
                "data": {"message": "test"}
            })),
        );
        let mcp_notif = McpNotification::from_json_rpc(&notif);
        match mcp_notif {
            McpNotification::Log {
                level,
                logger,
                data,
            } => {
                assert_eq!(level, "error");
                assert_eq!(logger, Some("test-logger".to_string()));
                assert!(data.is_object());
            }
            _ => panic!("Expected Log"),
        }
    }

    #[test]
    fn test_mcp_notification_log_edge_case_no_logger() {
        let notif = JsonRpcNotification::new(
            "notifications/message",
            Some(serde_json::json!({
                "level": "info",
                "data": "simple message"
            })),
        );
        let mcp_notif = McpNotification::from_json_rpc(&notif);
        match mcp_notif {
            McpNotification::Log { level, logger, .. } => {
                assert_eq!(level, "info");
                assert!(logger.is_none());
            }
            _ => panic!("Expected Log"),
        }
    }

    #[test]
    fn test_mcp_notification_log_edge_case_default_level() {
        let notif = JsonRpcNotification::new(
            "notifications/message",
            Some(serde_json::json!({
                "data": "message"
            })),
        );
        let mcp_notif = McpNotification::from_json_rpc(&notif);
        match mcp_notif {
            McpNotification::Log { level, .. } => {
                assert_eq!(level, "info");
            }
            _ => panic!("Expected Log"),
        }
    }

    #[test]
    fn test_mcp_notification_unknown() {
        let notif = JsonRpcNotification::new(
            "unknown/notification",
            Some(serde_json::json!({"key": "value"})),
        );
        let mcp_notif = McpNotification::from_json_rpc(&notif);
        match mcp_notif {
            McpNotification::Unknown { method, params } => {
                assert_eq!(method, "unknown/notification");
                assert!(params.is_some());
            }
            _ => panic!("Expected Unknown"),
        }
    }

    #[test]
    fn test_tool_content_image() {
        let content = ToolContent::Image {
            data: "base64data".to_string(),
            mime_type: "image/png".to_string(),
        };
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("\"type\":\"image\""));
        assert!(json.contains("\"data\":\"base64data\""));
        assert!(json.contains("\"mimeType\":\"image/png\""));
    }

    #[test]
    fn test_tool_content_resource() {
        let content = ToolContent::Resource {
            resource: ResourceContent {
                uri: "file:///test.txt".to_string(),
                mime_type: Some("text/plain".to_string()),
                text: Some("content".to_string()),
                blob: None,
            },
        };
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("\"type\":\"resource\""));
        assert!(json.contains("\"uri\":\"file:///test.txt\""));
    }

    #[test]
    fn test_mcp_server_config_default() {
        let config = McpServerConfig {
            name: "test-server".to_string(),
            transport: McpTransportConfig::Stdio {
                command: "node".to_string(),
                args: vec!["server.js".to_string()],
            },
            enabled: true,
            env: HashMap::new(),
            oauth: None,
            tool_timeout_secs: 60,
        };
        assert!(config.enabled);
        assert!(config.oauth.is_none());
    }

    #[test]
    fn test_mcp_server_config_with_env() {
        let mut env = HashMap::new();
        env.insert("API_KEY".to_string(), "secret".to_string());
        let config = McpServerConfig {
            name: "test-server".to_string(),
            transport: McpTransportConfig::Stdio {
                command: "node".to_string(),
                args: vec![],
            },
            enabled: true,
            env,
            oauth: None,
            tool_timeout_secs: 60,
        };
        assert!(config.env.contains_key("API_KEY"));
    }

    #[test]
    fn test_mcp_server_config_with_oauth() {
        let config = McpServerConfig {
            name: "test-server".to_string(),
            transport: McpTransportConfig::Http {
                url: "https://api.example.com".to_string(),
                headers: HashMap::new(),
            },
            enabled: true,
            env: HashMap::new(),
            oauth: Some(OAuthConfig {
                auth_url: "https://auth.example.com".to_string(),
                token_url: "https://token.example.com".to_string(),
                client_id: "client-123".to_string(),
                client_secret: Some("secret".to_string()),
                scopes: vec!["read".to_string(), "write".to_string()],
                redirect_uri: "http://localhost:8080/callback".to_string(),
                access_token: None,
            }),
            tool_timeout_secs: 60,
        };
        assert!(config.oauth.is_some());
    }

    #[test]
    fn test_mcp_transport_config_stdio_variant() {
        let transport = McpTransportConfig::Stdio {
            command: "python".to_string(),
            args: vec!["-m".to_string(), "server".to_string()],
        };
        match transport {
            McpTransportConfig::Stdio { command, args } => {
                assert_eq!(command, "python");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected Stdio"),
        }
    }

    #[test]
    fn test_mcp_transport_config_http_variant() {
        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), "Bearer token".to_string());
        let transport = McpTransportConfig::Http {
            url: "https://mcp.example.com".to_string(),
            headers,
        };
        match transport {
            McpTransportConfig::Http { url, headers } => {
                assert_eq!(url, "https://mcp.example.com");
                assert!(headers.contains_key("Authorization"));
            }
            _ => panic!("Expected Http"),
        }
    }

    #[test]
    fn test_mcp_prompt_serialize() {
        let prompt = McpPrompt {
            name: "test_prompt".to_string(),
            description: Some("A test prompt".to_string()),
            arguments: Some(vec![PromptArgument {
                name: "arg1".to_string(),
                description: Some("First argument".to_string()),
                required: true,
            }]),
        };
        let json = serde_json::to_string(&prompt).unwrap();
        assert!(json.contains("\"name\":\"test_prompt\""));
        assert!(json.contains("\"arguments\""));
    }

    #[test]
    fn test_prompt_argument_default() {
        let json = r#"{"name":"arg"}"#;
        let arg: PromptArgument = serde_json::from_str(json).unwrap();
        assert_eq!(arg.name, "arg");
        assert!(!arg.required);
    }

    #[test]
    fn test_oauth_config_with_static_token() {
        let json = r#"{
            "auth_url": "https://auth.example.com/authorize",
            "token_url": "https://auth.example.com/token",
            "client_id": "my-client",
            "scopes": ["read", "write"],
            "redirect_uri": "http://localhost/callback",
            "access_token": "static-token-abc123"
        }"#;
        let config: OAuthConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.client_id, "my-client");
        assert_eq!(config.access_token, Some("static-token-abc123".to_string()));
    }

    #[test]
    fn test_oauth_config_without_static_token() {
        let json = r#"{
            "auth_url": "https://auth.example.com/authorize",
            "token_url": "https://auth.example.com/token",
            "client_id": "my-client",
            "scopes": [],
            "redirect_uri": "http://localhost/callback"
        }"#;
        let config: OAuthConfig = serde_json::from_str(json).unwrap();
        assert!(config.access_token.is_none());
    }

    #[test]
    fn test_oauth_config_static_token_not_serialized_when_absent() {
        let config = OAuthConfig {
            auth_url: "https://example.com/auth".to_string(),
            token_url: "https://example.com/token".to_string(),
            client_id: "client".to_string(),
            client_secret: None,
            scopes: vec![],
            redirect_uri: "http://localhost/cb".to_string(),
            access_token: None,
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.contains("access_token"));
    }
}
