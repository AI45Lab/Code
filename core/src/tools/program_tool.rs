//! Tool wrapper for programmatic tool calling.

use crate::program::ProgramCatalog;
use crate::text::truncate_utf8;
use crate::tools::types::{Tool, ToolContext, ToolOutput};
use crate::tools::ToolRegistry;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use rquickjs::function::{Async, Func};
use rquickjs::{async_with, AsyncContext, AsyncRuntime, CatchResultExt, Error as JsError, Promise};
use serde::Deserialize;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tokio::time::{timeout, Duration};

const DEFAULT_SCRIPT_TIMEOUT_MS: u64 = 30_000;
/// Scripts allowed to delegate (`task`/`parallel_task`) run child agents that
/// each take a full LLM turn, so they need a far more generous default timeout.
const DELEGATION_SCRIPT_TIMEOUT_MS: u64 = 600_000;
const DEFAULT_SCRIPT_MAX_TOOL_CALLS: usize = 20;
const DEFAULT_SCRIPT_MAX_OUTPUT_BYTES: usize = 64 * 1024;
const MAX_SCRIPT_SOURCE_BYTES: usize = 64 * 1024;

pub struct ProgramTool {
    registry: Arc<ToolRegistry>,
}

impl ProgramTool {
    pub fn new(registry: Arc<ToolRegistry>) -> Self {
        Self { registry }
    }

    pub fn with_catalog(registry: Arc<ToolRegistry>, _catalog: ProgramCatalog) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for ProgramTool {
    fn name(&self) -> &str {
        "program"
    }

    fn description(&self) -> &str {
        "Run a sandboxed JavaScript PTC script. The script defines async function run(ctx, inputs) and may call only allowed ctx tools."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "type": {
                    "type": "string",
                    "description": "Required. Program kind. Only \"script\" is supported.",
                    "enum": ["script"]
                },
                "inputs": {
                    "type": "object",
                    "description": "Optional. JSON inputs passed to the script as the second argument."
                },
                "language": {
                    "type": "string",
                    "description": "Script language. Only JavaScript is supported.",
                    "enum": ["javascript"]
                },
                "source": {
                    "type": "string",
                    "description": "Inline JavaScript source defining async function run(ctx, inputs)."
                },
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path to a .js or .mjs script defining async function run(ctx, inputs). Used when source is omitted."
                },
                "allowed_tools": {
                    "type": "array",
                    "description": "Tool names the script may call through ctx. Defaults to all registered tools except program.",
                    "items": { "type": "string" }
                },
                "limits": {
                    "type": "object",
                    "description": "Optional timeoutMs, maxToolCalls, and maxOutputBytes.",
                    "additionalProperties": false,
                    "properties": {
                        "timeoutMs": { "type": "integer", "minimum": 1 },
                        "maxToolCalls": { "type": "integer", "minimum": 1 },
                        "maxOutputBytes": { "type": "integer", "minimum": 1 }
                    }
                }
            },
            "required": ["type"]
        })
    }

    async fn execute(&self, args: &serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let Some(kind) = args.get("type").and_then(|value| value.as_str()) else {
            return Ok(ToolOutput::error("type parameter is required"));
        };
        if kind != "script" {
            return Ok(ToolOutput::error(format!(
                "Unsupported program type: {kind}. Only \"script\" is supported."
            )));
        }
        let inputs = args
            .get("inputs")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));

        execute_script_program(args, inputs, Arc::clone(&self.registry), ctx).await
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScriptLimits {
    timeout_ms: Option<u64>,
    max_tool_calls: Option<usize>,
    max_output_bytes: Option<usize>,
}

#[derive(Debug, Clone)]
struct ScriptCallRecord {
    tool_name: String,
    success: bool,
    exit_code: i32,
    output_bytes: usize,
    metadata: Option<serde_json::Value>,
}

async fn execute_script_program(
    args: &serde_json::Value,
    inputs: serde_json::Value,
    registry: Arc<ToolRegistry>,
    ctx: &ToolContext,
) -> Result<ToolOutput> {
    let language = args
        .get("language")
        .and_then(|value| value.as_str())
        .unwrap_or("javascript");
    if language != "javascript" {
        return Ok(ToolOutput::error(format!(
            "Unsupported script language: {language}"
        )));
    }

    let source = match load_script_source(args, ctx).await {
        Ok(source) => source,
        Err(message) => return Ok(ToolOutput::error(message)),
    };
    if source.len() > MAX_SCRIPT_SOURCE_BYTES {
        return Ok(ToolOutput::error(format!(
            "script source is too large: {} bytes exceeds {} bytes",
            source.len(),
            MAX_SCRIPT_SOURCE_BYTES
        )));
    }
    if let Err(message) = validate_script_source(&source) {
        return Ok(ToolOutput::error(message));
    }

    let allowed_tools = script_allowed_tools(args, &registry);
    let limits = script_limits(args);
    match run_quickjs_script(
        &source,
        inputs,
        registry,
        ctx.clone(),
        allowed_tools,
        limits,
    )
    .await
    {
        Ok(output) => Ok(output),
        Err(err) => Ok(ToolOutput::error(format!("program script failed: {err}"))),
    }
}

async fn load_script_source(
    args: &serde_json::Value,
    ctx: &ToolContext,
) -> std::result::Result<String, String> {
    if let Some(source) = args.get("source").and_then(|value| value.as_str()) {
        return Ok(source.to_string());
    }

    let Some(path) = args.get("path").and_then(|value| value.as_str()) else {
        return Err("program script requires either source or path".to_string());
    };
    if !(path.ends_with(".js") || path.ends_with(".mjs")) {
        return Err("program script path must point to a .js or .mjs file".to_string());
    }

    let workspace_path = ctx
        .resolve_workspace_path(path)
        .map_err(|err| format!("failed to resolve script path: {err}"))?;
    ctx.workspace_services
        .fs()
        .read_text(&workspace_path)
        .await
        .map_err(|err| format!("failed to read script path '{}': {err}", path))
}

fn script_allowed_tools(args: &serde_json::Value, registry: &ToolRegistry) -> HashSet<String> {
    let mut allowed = args
        .get("allowed_tools")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str())
                .map(ToString::to_string)
                .collect::<HashSet<_>>()
        })
        .unwrap_or_else(|| registry.list().into_iter().collect());

    allowed.remove("program");
    // `task`/`parallel_task` ARE allowed in PTC scripts now: host tool calls run
    // on the outer multi-threaded runtime (see execute_host_tool_json), so
    // `ctx.tool("parallel_task", …)` fans out child agents in parallel.
    allowed
}

fn script_limits(args: &serde_json::Value) -> ScriptLimits {
    args.get("limits")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or(ScriptLimits {
            timeout_ms: None,
            max_tool_calls: None,
            max_output_bytes: None,
        })
}

fn validate_script_source(source: &str) -> std::result::Result<(), String> {
    let forbidden = [
        ("import ", "imports are not allowed inside PTC scripts"),
        (
            "import(",
            "dynamic imports are not allowed inside PTC scripts",
        ),
        ("eval(", "eval is not allowed inside PTC scripts"),
        (
            "Function(",
            "Function constructor is not allowed inside PTC scripts",
        ),
        ("Worker(", "Worker is not allowed inside PTC scripts"),
        ("WebSocket", "WebSocket is not allowed inside PTC scripts"),
        (
            "fetch(",
            "fetch is not allowed inside PTC scripts; use ctx tools instead",
        ),
    ];

    for (needle, message) in forbidden {
        if source.contains(needle) {
            return Err(message.to_string());
        }
    }
    Ok(())
}

async fn run_quickjs_script(
    source: &str,
    inputs: serde_json::Value,
    registry: Arc<ToolRegistry>,
    ctx: ToolContext,
    allowed_tools: HashSet<String>,
    limits: ScriptLimits,
) -> Result<ToolOutput> {
    // A script that can delegate runs child agents (each a full LLM turn, often
    // 30s to several minutes), so the 30s default is far too short and silently
    // times out real workflows. Default delegation-capable scripts to a generous
    // timeout; pure compute/search scripts keep the short default. An explicit
    // limits.timeoutMs always wins.
    let delegating = allowed_tools.contains("parallel_task") || allowed_tools.contains("task");
    let timeout_ms = limits.timeout_ms.unwrap_or(if delegating {
        DELEGATION_SCRIPT_TIMEOUT_MS
    } else {
        DEFAULT_SCRIPT_TIMEOUT_MS
    });
    let max_tool_calls = limits
        .max_tool_calls
        .unwrap_or(DEFAULT_SCRIPT_MAX_TOOL_CALLS);
    let max_output_bytes = limits
        .max_output_bytes
        .unwrap_or(DEFAULT_SCRIPT_MAX_OUTPUT_BYTES);
    let executable_source = script_source_with_host_entrypoint(source)?;
    // Captured on the outer multi-threaded runtime (we're async here, before the
    // VM's nested single-thread runtime is built) so host tool calls fan out.
    let outer = tokio::runtime::Handle::current();
    let state = Arc::new(Mutex::new(ScriptVmState {
        registry,
        ctx,
        allowed_tools,
        max_tool_calls,
        max_output_bytes,
        tool_calls: 0,
        records: Vec::new(),
        outer,
    }));

    let vm_state = Arc::clone(&state);
    let result = timeout(
        Duration::from_millis(timeout_ms),
        tokio::task::spawn_blocking(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|err| anyhow!("failed to create program VM runtime: {err}"))?;
            runtime.block_on(run_embedded_script(
                executable_source,
                inputs,
                vm_state,
                timeout_ms,
            ))
        }),
    )
    .await;

    match result {
        Ok(Ok(Ok(result))) => {
            let records = state.lock().await.records.clone();
            let output = render_script_output(&result, &records, "");
            Ok(ToolOutput::success(output).with_metadata(serde_json::json!({
                "program": {
                    "name": "script",
                    "language": "javascript",
                    "runtime": "embedded-quickjs",
                    "success": true,
                    "tool_calls": records.iter().map(script_record_to_value).collect::<Vec<_>>(),
                },
                "script_result": result,
            })))
        }
        Ok(Ok(Err(err))) if is_quickjs_timeout(&err) => Ok(ToolOutput::error(format!(
            "program script timed out after {timeout_ms} ms"
        ))),
        Ok(Ok(Err(err))) => Ok(ToolOutput::error(format!("program script error:\n{err}"))),
        Ok(Err(err)) => Ok(ToolOutput::error(format!(
            "program VM thread failed: {err}"
        ))),
        Err(_) => Ok(ToolOutput::error(format!(
            "program script timed out after {timeout_ms} ms"
        ))),
    }
}

fn script_source_with_host_entrypoint(source: &str) -> Result<String> {
    let rewritten = if source.contains("export default async function run") {
        source.replacen("export default async function run", "async function run", 1)
    } else if source.contains("export default function run") {
        source.replacen("export default function run", "function run", 1)
    } else if source.contains("async function run") || source.contains("function run") {
        source.to_string()
    } else {
        return Err(anyhow!(
            "PTC script must define async function run(ctx, inputs)"
        ));
    };

    Ok(format!(
        r#"{rewritten}

globalThis.__a3sResultJson = (async () => JSON.stringify(await run(globalThis.__a3sCtx, globalThis.__a3sInputs)))();
"#
    ))
}

async fn run_embedded_script(
    source: String,
    inputs: serde_json::Value,
    state: Arc<Mutex<ScriptVmState>>,
    timeout_ms: u64,
) -> Result<serde_json::Value> {
    let runtime = AsyncRuntime::new()?;
    let started = Instant::now();
    runtime
        .set_interrupt_handler(Some(Box::new(move || {
            started.elapsed() >= Duration::from_millis(timeout_ms)
        })))
        .await;
    runtime.set_memory_limit(64 * 1024 * 1024).await;
    runtime.set_max_stack_size(512 * 1024).await;

    let context = AsyncContext::full(&runtime).await?;
    let inputs_json = serde_json::to_string(&inputs)?;
    let script = format!("{}\n{}", embedded_script_bootstrap(&inputs_json), source);
    let result_json = async_with!(context => |ctx| {
        let state = Arc::clone(&state);
        let host_tool = move |tool: String, args_json: String| {
            let state = Arc::clone(&state);
            async move { execute_host_tool_json(state, tool, args_json).await }
        };
        if let Err(err) = ctx.globals().set("__a3sHostTool", Func::from(Async(host_tool))) {
            return Err(format!("failed to install program host tool: {err}"));
        }
        let promise: Promise = match ctx.eval(script) {
            Ok(promise) => promise,
            Err(err) => return Err(format!("failed to evaluate program script: {err}")),
        };
        promise
            .into_future::<String>()
            .await
            .catch(&ctx)
            .map_err(|err| err.to_string())
    })
    .await
    .map_err(anyhow::Error::msg)?;

    serde_json::from_str(&result_json)
        .map_err(|err| anyhow!("program script returned invalid JSON: {err}"))
}

struct ScriptVmState {
    registry: Arc<ToolRegistry>,
    ctx: ToolContext,
    allowed_tools: HashSet<String>,
    max_tool_calls: usize,
    max_output_bytes: usize,
    tool_calls: usize,
    records: Vec<ScriptCallRecord>,
    /// Handle to the OUTER multi-threaded session runtime. The script VM runs on
    /// a nested single-thread runtime; host tool calls are dispatched here so
    /// delegation tools (`parallel_task`/`task`) can actually fan out children.
    outer: tokio::runtime::Handle,
}

fn embedded_script_bootstrap(inputs_json: &str) -> String {
    format!(
        r#"
const __a3sCallTool = async (tool, args = {{}}) => {{
  const response = await globalThis.__a3sHostTool(String(tool), JSON.stringify(args ?? {{}}));
  return JSON.parse(response);
}};

const __a3sCtx = Object.freeze({{
  tool: __a3sCallTool,
  readFile: (path) => __a3sCallTool("read", {{ file_path: path }}).then((r) => r.output),
  read: (path) => __a3sCallTool("read", {{ file_path: path }}),
  grep: (pattern, options = {{}}) => __a3sCallTool("grep", {{ pattern, ...options }}).then((r) => r.output),
  glob: (pattern, options = {{}}) => __a3sCallTool("glob", {{ pattern, ...options }}).then((r) => r.output),
  ls: (path = ".") => __a3sCallTool("ls", {{ path }}).then((r) => r.output),
  bash: (command) => __a3sCallTool("bash", {{ command }}).then((r) => r.output),
  git: (args = {{}}) => __a3sCallTool("git", args),
  webSearch: (params) => __a3sCallTool("web_search", params),
  verify: (args) => __a3sCallTool("bash", args),
}});

Object.defineProperty(globalThis, "__a3sCtx", {{ value: __a3sCtx, configurable: false }});
Object.defineProperty(globalThis, "__a3sInputs", {{ value: {inputs_json}, configurable: false }});
Object.defineProperty(globalThis, "fetch", {{ value: undefined, configurable: false, writable: false }});
Object.defineProperty(globalThis, "WebSocket", {{ value: undefined, configurable: false, writable: false }});
Object.defineProperty(globalThis, "Worker", {{ value: undefined, configurable: false, writable: false }});
"#
    )
}

async fn execute_host_tool_json(
    state: Arc<Mutex<ScriptVmState>>,
    tool: String,
    args_json: String,
) -> rquickjs::Result<String> {
    let args = serde_json::from_str(&args_json).map_err(|err| {
        JsError::new_from_js_message("string", "object", format!("invalid tool args JSON: {err}"))
    })?;
    let (registry, ctx, max_output_bytes, outer) = {
        let mut script = state.lock().await;
        if !script.allowed_tools.contains(&tool) {
            return Err(JsError::new_from_js_message(
                "tool",
                "allowed tool",
                format!("tool '{tool}' is not allowed for this PTC script"),
            ));
        }
        script.tool_calls += 1;
        if script.tool_calls > script.max_tool_calls {
            return Err(JsError::new_from_js_message(
                "tool call",
                "limited tool call",
                format!("PTC script exceeded maxToolCalls={}", script.max_tool_calls),
            ));
        }
        (
            Arc::clone(&script.registry),
            script.ctx.clone(),
            script.max_output_bytes,
            script.outer.clone(),
        )
    };

    // Run the tool on the OUTER multi-threaded runtime (not this nested
    // single-thread VM runtime) so delegation tools can spawn child agents that
    // actually run in parallel — `ctx.tool("parallel_task", …)` now fans out.
    let tool_for_spawn = tool.clone();
    let result = outer
        .spawn(async move {
            registry
                .execute_with_context(&tool_for_spawn, &args, &ctx)
                .await
        })
        .await
        .map_err(|err| JsError::new_from_js_message("tool", "spawn", err.to_string()))?
        .map_err(|err| JsError::new_from_js_message("tool", "result", err.to_string()))?;
    let mut output = result.output;
    if output.len() > max_output_bytes {
        output = truncate_utf8(&output, max_output_bytes).to_string();
    }
    let success = result.exit_code == 0;
    let metadata = result.metadata.clone();
    let exit_code = result.exit_code;
    let name = result.name;

    {
        let mut script = state.lock().await;
        script.records.push(ScriptCallRecord {
            tool_name: tool,
            success,
            exit_code,
            output_bytes: output.len(),
            metadata: metadata.clone(),
        });
    }

    serde_json::to_string(&serde_json::json!({
        "name": name,
        "output": output,
        "exitCode": exit_code,
        "metadata": metadata,
    }))
    .map_err(|err| JsError::new_from_js_message("tool result", "json", err.to_string()))
}

fn is_quickjs_timeout(err: &anyhow::Error) -> bool {
    let text = err.to_string();
    text.contains("interrupted") || text.contains("InternalError")
}

fn script_record_to_value(record: &ScriptCallRecord) -> serde_json::Value {
    serde_json::json!({
        "tool_name": record.tool_name,
        "success": record.success,
        "exit_code": record.exit_code,
        "output_bytes": record.output_bytes,
        "metadata": record.metadata,
    })
}

fn render_script_output(
    result: &serde_json::Value,
    records: &[ScriptCallRecord],
    stderr: &str,
) -> String {
    let mut output = String::from("Program script completed.");
    if let Some(summary) = result.get("summary").and_then(|value| value.as_str()) {
        output.push('\n');
        output.push_str(summary);
    }

    output.push_str(&format!("\n\nTool calls: {}", records.len()));
    for (index, record) in records.iter().enumerate() {
        output.push_str(&format!(
            "\n{}. {} ({}, exit_code={}, output_bytes={})",
            index + 1,
            record.tool_name,
            if record.success { "ok" } else { "failed" },
            record.exit_code,
            record.output_bytes
        ));
    }

    output.push_str("\n\nResult:\n");
    output.push_str(&serde_json::to_string_pretty(result).unwrap_or_else(|_| result.to_string()));

    if !stderr.is_empty() {
        output.push_str("\n\nstderr:\n");
        output.push_str(stderr);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::path::PathBuf;

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echo test tool"
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string" }
                }
            })
        }

        async fn execute(
            &self,
            args: &serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput> {
            let message = args
                .get("message")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            Ok(ToolOutput::success(format!("echo:{message}")))
        }
    }

    #[tokio::test]
    async fn program_tool_rejects_non_script_type() {
        let tool = ProgramTool::new(Arc::new(ToolRegistry::new(PathBuf::from("/tmp"))));
        let output = tool
            .execute(
                &serde_json::json!({ "type": "program_code_search" }),
                &ToolContext::new(PathBuf::from("/tmp")),
            )
            .await
            .unwrap();

        assert!(!output.success);
        assert!(output.content.contains("Only \"script\" is supported"));
    }

    #[tokio::test]
    async fn program_tool_rejects_missing_script_source_and_path() {
        let tool = ProgramTool::new(Arc::new(ToolRegistry::new(PathBuf::from("/tmp"))));
        let output = tool
            .execute(
                &serde_json::json!({ "type": "script" }),
                &ToolContext::new(PathBuf::from("/tmp")),
            )
            .await
            .unwrap();

        assert!(!output.success);
        assert!(output.content.contains("requires either source or path"));
    }

    #[tokio::test]
    async fn program_tool_rejects_unsupported_language() {
        let tool = ProgramTool::new(Arc::new(ToolRegistry::new(PathBuf::from("/tmp"))));
        let output = tool
            .execute(
                &serde_json::json!({
                    "type": "script",
                    "language": "typescript",
                    "source": "async function run() { return {}; }"
                }),
                &ToolContext::new(PathBuf::from("/tmp")),
            )
            .await
            .unwrap();

        assert!(!output.success);
        assert!(output.content.contains("Unsupported script language"));
    }

    #[tokio::test]
    async fn program_tool_rejects_unsupported_script_path() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("script.txt"), "async function run() {}").unwrap();
        let tool = ProgramTool::new(Arc::new(ToolRegistry::new(dir.path().to_path_buf())));
        let output = tool
            .execute(
                &serde_json::json!({
                    "type": "script",
                    "path": "script.txt"
                }),
                &ToolContext::new(dir.path().to_path_buf()),
            )
            .await
            .unwrap();

        assert!(!output.success);
        assert!(output.content.contains(".js or .mjs file"));
    }

    #[test]
    fn program_tool_default_allowed_tools_include_registry_tools_except_program() {
        let registry = ToolRegistry::new(PathBuf::from("/tmp"));
        registry.register(Arc::new(EchoTool));
        registry.register_builtin(Arc::new(ProgramTool::new(Arc::new(ToolRegistry::new(
            PathBuf::from("/tmp"),
        )))));

        let allowed = script_allowed_tools(&serde_json::json!({}), &registry);

        assert!(allowed.contains("echo"));
        assert!(!allowed.contains("program"));
    }

    #[test]
    fn program_tool_allows_delegation_tools_in_scripts() {
        // Delegation tools are allowed in PTC scripts again (host tool calls run
        // on the outer multi-threaded runtime, so they fan out). Only `program`
        // stays stripped (no nested PTC recursion).
        let registry = ToolRegistry::new(PathBuf::from("/tmp"));
        let args = serde_json::json!({
            "allowed_tools": ["parallel_task", "task", "program", "echo"]
        });
        let allowed = script_allowed_tools(&args, &registry);
        assert!(allowed.contains("parallel_task"));
        assert!(allowed.contains("task"));
        assert!(allowed.contains("echo"));
        assert!(!allowed.contains("program"));
    }

    #[tokio::test]
    async fn program_tool_source_uses_default_all_registered_tools() {
        let registry = Arc::new(ToolRegistry::new(PathBuf::from("/tmp")));
        registry.register(Arc::new(EchoTool));
        let tool = ProgramTool::new(Arc::clone(&registry));
        let output = tool
            .execute(
                &serde_json::json!({
                    "type": "script",
                    "source": r#"
                        async function run(ctx, inputs) {
                            const result = await ctx.tool("echo", { message: inputs.message });
                            return { summary: result.output, result };
                        }
                    "#,
                    "inputs": { "message": "hello" }
                }),
                &ToolContext::new(PathBuf::from("/tmp")),
            )
            .await
            .unwrap();

        assert!(output.success, "{}", output.content);
        assert!(output.content.contains("echo:hello"));
        let metadata = output.metadata.unwrap();
        assert_eq!(metadata["program"]["runtime"], "embedded-quickjs");
        assert_eq!(metadata["script_result"]["summary"], "echo:hello");
    }

    #[tokio::test]
    async fn program_tool_explicit_allowed_tools_restrict_default_tools() {
        let registry = Arc::new(ToolRegistry::new(PathBuf::from("/tmp")));
        registry.register(Arc::new(EchoTool));
        let tool = ProgramTool::new(Arc::clone(&registry));
        let output = tool
            .execute(
                &serde_json::json!({
                    "type": "script",
                    "source": r#"
                        async function run(ctx) {
                            await ctx.tool("echo", { message: "blocked" });
                            return {};
                        }
                    "#,
                    "allowed_tools": ["read"]
                }),
                &ToolContext::new(PathBuf::from("/tmp")),
            )
            .await
            .unwrap();

        assert!(!output.success);
        assert!(output.content.contains("tool 'echo' is not allowed"));
    }

    #[tokio::test]
    async fn program_tool_enforces_max_tool_calls() {
        let registry = Arc::new(ToolRegistry::new(PathBuf::from("/tmp")));
        registry.register(Arc::new(EchoTool));
        let tool = ProgramTool::new(Arc::clone(&registry));
        let output = tool
            .execute(
                &serde_json::json!({
                    "type": "script",
                    "source": r#"
                        async function run(ctx) {
                            await ctx.tool("echo", { message: "one" });
                            await ctx.tool("echo", { message: "two" });
                            return {};
                        }
                    "#,
                    "limits": { "maxToolCalls": 1 }
                }),
                &ToolContext::new(PathBuf::from("/tmp")),
            )
            .await
            .unwrap();

        assert!(!output.success);
        assert!(output.content.contains("exceeded maxToolCalls=1"));
    }

    #[test]
    fn program_tool_rejects_fetch_source_access() {
        let err =
            validate_script_source("export default async function run() { return fetch('/'); }")
                .unwrap_err();
        assert!(err.contains("fetch is not allowed"));
    }

    #[test]
    fn program_tool_accepts_plain_function_run_entrypoint() {
        let source = script_source_with_host_entrypoint(
            "async function run(ctx, inputs) { return { summary: inputs.message }; }",
        )
        .unwrap();

        assert!(source.contains("globalThis.__a3sResultJson"));
        assert!(source.contains("async function run"));
    }

    #[test]
    fn program_tool_renders_result_summary_and_tool_records() {
        let output = render_script_output(
            &serde_json::json!({ "summary": "done", "items": [1] }),
            &[ScriptCallRecord {
                tool_name: "echo".to_string(),
                success: true,
                exit_code: 0,
                output_bytes: 8,
                metadata: Some(serde_json::json!({ "kind": "test" })),
            }],
            "",
        );

        assert!(output.contains("Program script completed."));
        assert!(output.contains("done"));
        assert!(output.contains("echo (ok"));
        assert!(output.contains("\"items\""));
    }
}
