# A3S Code — Python SDK

Native Python bindings for the A3S Code AI coding agent, built with PyO3.

## Installation

```bash
pip install a3s-code
```

## Quick Start

```python
from a3s_code import Agent

agent = Agent.create("agent.acl")
session = agent.session("/my-project")

result = session.send({"prompt": "What files handle authentication?"})
print(result.text)
```

## Slash Commands

Every session includes built-in slash commands dispatched before the LLM:

```python
# List all available commands
commands = session.list_commands()
for cmd in commands:
    print(f"/{cmd['name']:15s} {cmd['description']}")

# Built-in commands
result = session.send("/help")       # List all commands
result = session.send("/model")      # Show current model
result = session.send("/cost")       # Token usage and cost
result = session.send("/history")    # Conversation stats
```

### Custom Commands

```python
def my_handler(args: str, ctx: dict) -> str:
    return f"Model: {ctx['model']}, History: {ctx['history_len']} msgs, args: {args!r}"

session.register_command("status", "Show session info", my_handler)
result = session.send("/status hello")
```

## Full API

```python
from a3s_code import (
    Agent,
    ConfirmationPolicy,
    PermissionPolicy,
    SessionOptions,
    WorkerAgentSpec,
    DefaultSecurityProvider,
    FileMemoryStore,
    FileSessionStore,
    HttpTransport,
    LocalWorkspaceBackend,
)

agent = Agent.create("agent.acl")
session = agent.session("/my-project",
    model="openai/gpt-4o",
    builtin_skills=True,
    planning_mode="auto",  # "enabled" forces planning, "disabled" turns it off
)

# Send / Stream
result = session.send({"prompt": "Explain the auth module"})
for event in session.stream({"prompt": "Refactor auth"}):
    if event.event_type == "text_delta":
        print(event.text, end="", flush=True)

# Streams with no custom history update session history and verification evidence
# when the stream completes. Passing explicit history keeps the stream isolated.
# send(...) and stream(...) accept prompt strings or object-shaped requests
# with optional history and attachments.

# Planning events
# Prefer planning_mode="auto" | "enabled" | "disabled". The legacy planning
# bool still works: True forces planning, False disables it. In streaming mode,
# render task_updated as the current task list; step_start and step_end are
# per-step progress events.

# Run replay
runs = session.runs()
if runs:
    print(runs[-1]["id"], runs[-1]["status"])
    print(session.run_events(runs[-1]["id"]))
    print(session.active_tools())
    # Cancels only if that run is still active; stale IDs are ignored.
    session.cancel_run(runs[-1]["id"])

# Direct tools (bypass LLM)
opts = SessionOptions()
opts.workspace_backend = LocalWorkspaceBackend("/my-project")
session = agent.session("/my-project", opts)
session.write_file("notes.txt", "one\ntwo\n")
session.read_file("src/main.py")
session.ls()
session.edit_file("notes.txt", "one", "uno")
session.patch_file("notes.txt", "@@ -1,2 +1,2 @@\n uno\n-two\n+dos")
session.bash("pytest")
session.glob("**/*.py")
session.grep("TODO")
session.tool_names()
session.tool_definitions()

# Programmatic Tool Calling (embedded QuickJS)
program = session.program({
    "source": """
        export default async function run(ctx, inputs) {
          const hits = await ctx.grep(inputs.query, { glob: '*.py' });
          const files = await ctx.glob('src/**/*.py');
          return { hits, files: files.slice(0, 10) };
        }
    """,
    "inputs": {"query": "PermissionPolicy"},
    "allowed_tools": ["grep", "glob"],
    "limits": {"timeoutMs": 30000, "maxToolCalls": 20, "maxOutputBytes": 65536},
})
print(program.output)

# Delegation helpers (wrappers around task / parallel_task)
session.task({
    "agent": "explore",
    "description": "Find auth entry points",
    "prompt": "Inspect the repository and summarize the auth-related files.",
})
session.tasks([
    {"agent": "explore", "description": "Find tests", "prompt": "Locate auth tests."},
    {"agent": "verification", "description": "Check risk", "prompt": "Review auth edge cases."},
])

# Disposable worker agents (cattle mode)
opts = SessionOptions()
frontend = WorkerAgentSpec.implementer("frontend-cow", "Small verified frontend fixes")
frontend.model = "openai/gpt-4o"
frontend.max_steps = 24
frontend.prompt = "Keep patches focused and run the narrowest relevant check."
frontend.confirmation_inheritance = "auto_approve"  # child runs auto-approve Ask decisions
opts.add_worker_agent(frontend)
session = agent.session("/my-project", opts)
session.task({
    "agent": "frontend-cow",
    "description": "Fix admin chat loading state",
    "prompt": "Find and fix the loading-state regression, then summarize verification.",
})

# Confirmation inheritance controls how child runs resolve Ask decisions:
# - "auto_approve" (default): Child runs auto-approve all Ask decisions
# - "deny_on_ask": Child runs fail immediately when encountering an Ask
# - "inherit_parent": Child runs inherit the parent's confirmation policy
restricted = WorkerAgentSpec("restricted-writer", "Write files with parent confirmation", "implementer")
restricted.confirmation_inheritance = "inherit_parent"  # requires parent approval
opts.add_worker_agent(restricted)

# Object-shaped direct tools
session.git({"command": "status"})
session.git({"command": "worktree", "subcommand": "list"})
# git_command(...) and positional git(...) remain for compatibility.

# Live registration and top-level worker sessions are also supported.
session.register_worker_agent(WorkerAgentSpec.verifier("verify-cow", "Run focused checks"))
worker_session = agent.session_for_worker(
    "/my-project",
    WorkerAgentSpec.reviewer("review-cow", "Adversarial code review"),
)

# AHP-supervised background advice
opts = SessionOptions()
opts.ahp_transport = HttpTransport("http://localhost:8080/ahp")
session = agent.session("/my-project", opts)
# The AHP harness owns background advice, context supplements, and PTC proposals.
# Proposed scripts should be run explicitly with session.program(...) if approved.

# Slash commands
session.list_commands()
session.register_command("ping", "Pong!", lambda args, ctx: "pong")

# Memory
session.remember_success("task", ["tool"], "result")
session.recall_similar("auth", 5)

# Hooks
session.register_hook("audit", "pre_tool_use", handler_fn)

# MCP
session.add_mcp({
    "name": "github",
    "transport": {
        "type": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
    },
    "timeout_ms": 30_000,
})
session.mcps()
session.tool_names()
session.remove_mcp("github")

# Persistence
opts = SessionOptions()
opts.session_store = FileSessionStore('./sessions')
opts.session_id = 'my-session'
opts.auto_save = True
session2 = agent.session(".", opts)
resumed = agent.resume_session('my-session', opts)
```

## HITL Confirmations

Use `PermissionPolicy` to decide which tools ask, then `ConfirmationPolicy` to
control confirmation runtime behavior such as timeout and YOLO lanes. Invalid
permission decisions, timeout actions, and lane names are rejected when the
session is created so unsafe fallbacks do not silently change policy.

```python
opts = SessionOptions()
opts.permission_policy = PermissionPolicy(ask=["bash*"], default_decision="allow")
opts.confirmation_policy = ConfirmationPolicy(
    enabled=True,
    default_timeout_ms=30_000,
    timeout_action="reject",
    yolo_lanes=["query"],
)
session = agent.session(".", opts)

for pending in session.pending_confirmations():
    session.confirm_tool_use(pending["tool_id"], approved=True, reason="Reviewed")
```

For the streaming event-driven loop used by UIs, see
`examples/hitl_confirmation_loop.py`.

## Delegation

Routine multi-agent work uses the model-visible `task` and `parallel_task`
tools. Use `session.task(...)` and `session.tasks(...)` for SDK-native calls,
or `session.tool("task", {...})` when you need raw access.
The old standalone lifecycle control-plane API is intentionally removed from
the 2.0 SDK surface.

## License

MIT
