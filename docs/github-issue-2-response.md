## ✅ Issue Resolved

Sub-agents can now be configured with permissive mode via the `task` tool's `permissive` parameter.

### Implementation

Added `permissive: bool` parameter to `TaskParams` in `core/src/tools/task.rs`. When set to `true`, the child agent is configured with a permissive permission policy that bypasses HITL confirmation.

### Usage Example (Python SDK)

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

### Testing

Created comprehensive integration tests in `sdk/python/examples/test_permissive_subagents.py`:
- ✅ Sub-agent with `permissive=True` executes tools without HITL
- ✅ Sub-agent with `permissive=False` requires HITL (default behavior)
- ✅ Parallel sub-agents with mixed permissive settings

### Bonus: SubAgent Event Streaming

Also implemented SubAgent event streaming support:
- Parent sessions can now monitor internal SubAgent events (tool calls, LLM responses, etc.)
- Events are forwarded from child agents to parent's broadcast channel
- Python SDK now recognizes `subagent_start`, `subagent_end`, `subagent_progress`, and `tool_input_delta` events

### Commits

- `a004591` - feat: add permissive mode support for sub-agents
- `74aeccf` - test: add integration test for permissive mode in sub-agents
- `d232e48` - feat: enable SubAgent event streaming to parent sessions

### Branch

`fix/security-and-windows-compat`

This fix is ready for review and merge. It will be included in the next release (v1.0.3).
