"""Real-provider smoke for the Python SDK surface hardened in 2.3.

The runner script rewrites .a3s/config.acl so OpenAI-compatible credentials
come from A3S_OPENAI_* environment variables. MINIMAX_* aliases are accepted
by the script before this file runs.
"""

from __future__ import annotations

import os
import tempfile
import time

from a3s_code import Agent, LocalWorkspaceBackend, PermissionPolicy, SessionOptions


RUN_FULL_AGENT_SMOKE = os.environ.get("A3S_CODE_SDK_REAL_AGENT_SMOKE") != "0"
RUN_CHILD_AGENT_SMOKE = os.environ.get("A3S_CODE_SDK_REAL_CHILD_AGENT_SMOKE") == "1"


def step(name, fn):
    print(f"[python-sdk-real] {name} ... ", end="", flush=True)
    started = time.time()
    try:
        value = fn()
        elapsed_ms = int((time.time() - started) * 1000)
        print(f"ok ({elapsed_ms}ms)", flush=True)
        return value
    except Exception:
        elapsed_ms = int((time.time() - started) * 1000)
        print(f"failed ({elapsed_ms}ms)", flush=True)
        raise


config_file = os.environ.get("A3S_CONFIG_FILE")
if not config_file:
    raise RuntimeError("A3S_CONFIG_FILE must point to the env-injected ACL config")

agent = step("Agent.create", lambda: Agent.create(config_file))

opts = SessionOptions()
opts.planning_mode = "disabled"
opts.permission_policy = PermissionPolicy(default_decision="allow")
opts.max_parse_retries = 1
opts.circuit_breaker_threshold = 1

workspace = os.environ.get("A3S_CODE_SDK_REAL_WORKSPACE") or tempfile.mkdtemp(
    prefix="a3s-code-python-sdk-real-"
)
opts.workspace_backend = LocalWorkspaceBackend(workspace)
print(f"[python-sdk-real] workspace={workspace}", flush=True)
session = agent.session(workspace, opts)

tool_names = step("tool_names", session.tool_names)
assert "program" in tool_names
assert "task" in tool_names
assert "parallel_task" in tool_names

tool_definitions = step("tool_definitions", session.tool_definitions)
assert isinstance(tool_definitions, list)
assert any(tool.get("name") == "program" for tool in tool_definitions)

write_result = step(
    "write_file",
    lambda: session.write_file("notes.txt", "one\ntwo\n"),
)
assert write_result.exit_code == 0, write_result.output
assert "one" in step("read_file", lambda: session.read_file("notes.txt"))
ls_result = step("ls", lambda: session.ls())
assert ls_result.exit_code == 0, ls_result.output
assert "notes.txt" in ls_result.output
edit_result = step(
    "edit_file",
    lambda: session.edit_file("notes.txt", "one", "uno"),
)
assert edit_result.exit_code == 0, edit_result.output
patch_result = step(
    "patch_file",
    lambda: session.patch_file("notes.txt", "@@ -1,2 +1,2 @@\n uno\n-two\n+dos"),
)
assert patch_result.exit_code == 0, patch_result.output

program_result = step(
    "program",
    lambda: session.program(
        {
        "source": """
            export default async function run(ctx, inputs) {
              const listing = await ctx.ls(".");
              return { marker: inputs.marker, listed: listing.length > 0 };
            }
        """,
        "inputs": {"marker": "python-sdk-program-ok"},
        "allowed_tools": ["ls"],
        }
    ),
)
assert program_result.exit_code == 0
assert "python-sdk-program-ok" in program_result.output

if RUN_FULL_AGENT_SMOKE:
    result = step(
        "send",
        lambda: session.send("Reply with exactly: PYTHON_SDK_REAL_OK"),
    )
    assert result.text.strip()

    runs = step("runs", session.runs)
    assert runs
    assert runs[0]["status"] == "completed"
    events = step("run_events", lambda: session.run_events(runs[0]["id"]))
    event_types = {event["event"]["type"] for event in events}
    assert "agent_start" in event_types
    assert "agent_end" in event_types

    delegated = step(
        "delegate_task",
        lambda: session.delegate_task(
            agent="explore",
            description="Python SDK delegated child smoke",
            prompt=(
                "Reply with exactly: PYTHON_SDK_DELEGATE_OK"
                if RUN_CHILD_AGENT_SMOKE
                else "Background smoke; no result is required."
            ),
            background=not RUN_CHILD_AGENT_SMOKE,
            max_steps=3,
        ),
    )
    assert delegated.exit_code == 0, delegated.output
    if RUN_CHILD_AGENT_SMOKE:
        assert delegated.output.strip()
    else:
        assert "Task started in background" in delegated.output
        print(
            "[python-sdk-real] synchronous child-agent delegate_task smoke skipped; "
            "set A3S_CODE_SDK_REAL_CHILD_AGENT_SMOKE=1 to enable",
            flush=True,
        )
else:
    print(
        "[python-sdk-real] full agent send/delegate_task smoke skipped by "
        "A3S_CODE_SDK_REAL_AGENT_SMOKE=0",
        flush=True,
    )

print("python sdk real config env integration ok")
