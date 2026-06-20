//! A `tools/` `kind = "script"` entry exposed as a model-visible tool.
//!
//! Thin facade over the existing [`ProgramTool`] QuickJS path: the script `path`,
//! `allowed_tools`, and sandbox `limits` are pinned by the [`ScriptToolSpec`]; the
//! model only supplies `inputs`. It adds NO new sandbox — execution, the frozen
//! `ctx`, the memory/stack/timeout caps, and the per-call tool-call/output limits
//! are all the program tool's.
//!
//! Safety boundary: the model's call to THIS tool is permission-gated like any
//! tool (the harness owns visibility and the gate). But the script's inner
//! `ctx.tool` calls go through `ToolRegistry::execute_with_context` directly —
//! they are bounded by the pinned `allowed_tools` list, the `maxToolCalls`
//! counter, and the QuickJS sandbox (no fs/net/proc/env), and they run against the
//! session workspace, but they are NOT re-evaluated against the session permission
//! policy / HITL. The allow-list is therefore the boundary for what a script may
//! reach, which is why the agent-dir loader fails it closed (empty by default).

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

use crate::config::ScriptToolSpec;
use crate::tools::types::{Tool, ToolContext, ToolOutput};
use crate::tools::{ProgramTool, ToolRegistry};

/// A named, pre-parameterized `program` script call.
pub struct AgentDirScriptTool {
    spec: ScriptToolSpec,
    program: ProgramTool,
}

impl AgentDirScriptTool {
    /// `registry` must be the session's registry so the script's `ctx.tool`
    /// calls resolve against the session's actual tools (and the allow-list).
    pub fn new(spec: ScriptToolSpec, registry: Arc<ToolRegistry>) -> Self {
        Self {
            spec,
            program: ProgramTool::new(registry),
        }
    }
}

#[async_trait]
impl Tool for AgentDirScriptTool {
    fn name(&self) -> &str {
        &self.spec.name
    }

    fn description(&self) -> &str {
        &self.spec.description
    }

    fn parameters(&self) -> serde_json::Value {
        // The model controls only `inputs`; path/allow-list/limits are pinned.
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "inputs": {
                    "type": "object",
                    "description": "JSON inputs passed to the script's async run(ctx, inputs)."
                }
            },
            "required": []
        })
    }

    async fn execute(&self, args: &serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let inputs = args.get("inputs").cloned().unwrap_or_else(|| json!({}));

        // Build exactly the args the `program` tool already accepts, with the
        // spec's path/allow-list/limits pinned. `limits` serializes to the
        // camelCase keys (timeoutMs/…) the program tool reads.
        let mut program_args = json!({
            "type": "script",
            "language": "javascript",
            "path": self.spec.path.to_string_lossy(),
            "limits": self.spec.limits,
            "inputs": inputs,
        });
        if let Some(allowed) = &self.spec.allowed_tools {
            program_args["allowed_tools"] = json!(allowed);
        }

        self.program.execute(&program_args, ctx).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ScriptToolLimits;
    use crate::tools::types::ToolOutput;
    use std::path::PathBuf;

    /// Minimal host tool the script calls through `ctx`.
    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "echo"
        }
        fn parameters(&self) -> serde_json::Value {
            json!({ "type": "object" })
        }
        async fn execute(
            &self,
            args: &serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput> {
            let msg = args.get("message").and_then(|v| v.as_str()).unwrap_or("");
            Ok(ToolOutput::success(format!("echo:{msg}")))
        }
    }

    fn spec(path: &str, allowed: Option<Vec<String>>) -> ScriptToolSpec {
        ScriptToolSpec {
            name: "echo-runner".to_string(),
            description: "runs echo".to_string(),
            path: PathBuf::from(path),
            allowed_tools: allowed,
            limits: ScriptToolLimits::default(),
        }
    }

    /// The spec's pinned `limits` actually reach the sandbox: a script that calls
    /// a tool twice under `maxToolCalls: 1` has its second call rejected (proving
    /// the wrapper forwards limits rather than silently dropping them).
    #[tokio::test]
    async fn script_tool_pinned_limits_are_enforced() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("twice.js"),
            r#"async function run(ctx) {
                   await ctx.tool("echo", { message: "1" });
                   await ctx.tool("echo", { message: "2" });
                   return { ok: true };
               }"#,
        )
        .unwrap();

        let registry = Arc::new(ToolRegistry::new(dir.path().to_path_buf()));
        registry.register(Arc::new(EchoTool));

        let mut s = spec("twice.js", Some(vec!["echo".to_string()]));
        s.limits.max_tool_calls = Some(1);
        let tool = AgentDirScriptTool::new(s, registry);

        let out = tool
            .execute(&json!({}), &ToolContext::new(dir.path().to_path_buf()))
            .await
            .unwrap();

        assert!(
            out.content.contains("maxToolCalls") || !out.success,
            "second tool call must be blocked by the pinned limit: {}",
            out.content
        );
    }

    /// `inputs` from the caller reach the script's `run(ctx, inputs)` second arg;
    /// a missing `inputs` defaults to `{}` (no panic).
    #[tokio::test]
    async fn script_tool_passes_inputs_and_defaults_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("echoback.js"),
            r#"async function run(ctx, inputs) {
                   return { got: inputs && inputs.name ? inputs.name : "DEFAULT" };
               }"#,
        )
        .unwrap();
        let registry = Arc::new(ToolRegistry::new(dir.path().to_path_buf()));
        let tool = AgentDirScriptTool::new(spec("echoback.js", Some(vec![])), registry);

        let with = tool
            .execute(
                &json!({ "inputs": { "name": "ada" } }),
                &ToolContext::new(dir.path().to_path_buf()),
            )
            .await
            .unwrap();
        assert!(
            with.content.contains("ada"),
            "inputs reach the script: {}",
            with.content
        );

        let without = tool
            .execute(&json!({}), &ToolContext::new(dir.path().to_path_buf()))
            .await
            .unwrap();
        assert!(
            without.content.contains("DEFAULT"),
            "missing inputs defaults to {{}}: {}",
            without.content
        );
    }

    /// The wrapper runs the pinned script through the QuickJS path and the script
    /// can call an allowed host tool via `ctx`.
    #[tokio::test]
    async fn script_tool_runs_pinned_script_and_calls_allowed_tool() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("echo.js"),
            r#"async function run(ctx, inputs) {
                   const r = await ctx.tool("echo", { message: inputs.message });
                   return { echoed: r.output };
               }"#,
        )
        .unwrap();

        let registry = Arc::new(ToolRegistry::new(dir.path().to_path_buf()));
        registry.register(Arc::new(EchoTool));

        let tool =
            AgentDirScriptTool::new(spec("echo.js", Some(vec!["echo".to_string()])), registry);
        assert_eq!(tool.name(), "echo-runner");

        let out = tool
            .execute(
                &json!({ "inputs": { "message": "hi" } }),
                &ToolContext::new(dir.path().to_path_buf()),
            )
            .await
            .unwrap();

        assert!(out.success, "script should succeed: {}", out.content);
        assert!(out.content.contains("echo:hi"), "got: {}", out.content);
    }

    /// The pinned allow-list is enforced: a tool not in `allowed_tools` is blocked
    /// even though it is registered.
    #[tokio::test]
    async fn script_tool_allow_list_blocks_unlisted_tool() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("echo.js"),
            r#"async function run(ctx, inputs) {
                   const r = await ctx.tool("echo", { message: "x" });
                   return { echoed: r.output };
               }"#,
        )
        .unwrap();

        let registry = Arc::new(ToolRegistry::new(dir.path().to_path_buf()));
        registry.register(Arc::new(EchoTool));

        // allowed_tools = [] → echo is NOT permitted.
        let tool = AgentDirScriptTool::new(spec("echo.js", Some(vec![])), registry);
        let out = tool
            .execute(&json!({}), &ToolContext::new(dir.path().to_path_buf()))
            .await
            .unwrap();

        // The ctx.tool("echo") call is rejected inside the sandbox; the script
        // throws and the program run surfaces a failure (not echo:x).
        assert!(
            !out.content.contains("echo:x"),
            "allow-list must block echo"
        );
    }
}
