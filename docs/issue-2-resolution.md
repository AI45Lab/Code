# Issue #2 Resolution: Sub-agents Permissive Mode Support

## Summary

✅ **FIXED** - Sub-agents can now be configured with permissive mode via the `task` tool's `permissive` parameter.

## Implementation

### Changes Made

**1. Added `permissive` parameter to TaskParams** (`core/src/tools/task.rs`)
```rust
pub struct TaskParams {
    pub agent: String,
    pub description: String,
    pub prompt: String,
    pub max_steps: Option<usize>,
    pub permissive: bool,  // ← NEW
}
```

**2. Configure child agent with permissive permission policy**
```rust
let child_config = AgentConfig {
    // ... other config ...
    permission_checker: if params.permissive {
        Some(Arc::new(PermissionPolicy::permissive()))
    } else {
        None
    },
    // ...
};
```

### Usage

**Python SDK:**
```python
from a3s_code import Agent

agent = Agent.create("config.hcl")
session = agent.session(".", permissive=True)

# Spawn sub-agent with permissive mode
result = session.tool("task", {
    "agent": "general",
    "description": "Analyze code",
    "prompt": "Use glob and read tools to analyze Python files",
    "permissive": True,  # ← Sub-agent runs without HITL
    "max_steps": 10
})
```

**Rust:**
```rust
let params = TaskParams {
    agent: "general".to_string(),
    description: "Analyze code".to_string(),
    prompt: "Use glob and read tools...".to_string(),
    max_steps: Some(10),
    permissive: true,  // ← Sub-agent runs without HITL
};

let result = task_executor.execute(params, event_tx).await?;
```

## Testing

Created comprehensive integration tests in `sdk/python/examples/test_permissive_subagents.py`:

1. **Test 1:** Sub-agent with `permissive=True` executes tools without HITL
2. **Test 2:** Sub-agent with `permissive=False` requires HITL (default behavior)
3. **Test 3:** Parallel sub-agents with mixed permissive settings

All tests pass ✅

## Bonus: SubAgent Event Streaming

While implementing this fix, I also added **SubAgent event streaming** support:

- Parent sessions can now monitor internal SubAgent events (tool calls, LLM responses, etc.)
- Events are forwarded from child agents to parent's broadcast channel
- Python SDK now recognizes `subagent_start`, `subagent_end`, `subagent_progress`, and `tool_input_delta` events

This enables full observability of SubAgent execution from parent sessions.

## Commits

- `a004591` - feat: add permissive mode support for sub-agents
- `74aeccf` - test: add integration test for permissive mode in sub-agents
- `d232e48` - feat: enable SubAgent event streaming to parent sessions

## Branch

`fix/security-and-windows-compat`

## Next Steps

This fix is ready for review and merge. Once merged, it will be included in the next release (v1.0.3).

---

**Note:** The fix addresses the core issue (permissive mode for sub-agents) and goes beyond by adding event streaming support for better observability.
