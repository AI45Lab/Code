"""Smoke test for the subagent task query API exposed in PR #4.

Verifies the three new Session methods are reachable from Python and
return the expected empty-state shapes for a fresh session. Mirrors the
Node SDK smoke test in sdk/node/test.mjs.

Run with: A3S_CONFIG_FILE not needed — uses inline ACL.
"""

from __future__ import annotations

import tempfile

from a3s_code import Agent, LocalWorkspaceBackend, PermissionPolicy, SessionOptions


INLINE_CONFIG = """
default_model = "anthropic/claude-sonnet-4-20250514"

providers "anthropic" {
  api_key = "test-key"
  models "claude-sonnet-4-20250514" {
    name = "Claude Sonnet 4"
  }
}
""".strip()


def main() -> None:
    workspace = tempfile.mkdtemp(prefix="a3s-code-python-subagent-")
    agent = Agent.create(INLINE_CONFIG)

    opts = SessionOptions()
    opts.permission_policy = PermissionPolicy(default_decision="allow")
    opts.workspace_backend = LocalWorkspaceBackend(workspace)

    session = agent.session(workspace, opts)

    tasks = session.subagent_tasks()
    assert isinstance(tasks, list), f"subagent_tasks() should return list, got {type(tasks)!r}"
    assert tasks == [], f"fresh session should have no subagent tasks, got {tasks!r}"

    pending = session.pending_subagent_tasks()
    assert isinstance(
        pending, list
    ), f"pending_subagent_tasks() should return list, got {type(pending)!r}"
    assert (
        pending == []
    ), f"fresh session should have no pending subagent tasks, got {pending!r}"

    missing = session.subagent_task("task-does-not-exist")
    assert missing is None, f"unknown subagent task id should return None, got {missing!r}"

    cancelled = session.cancel_subagent_task("task-does-not-exist")
    assert cancelled is False, (
        f"cancelling unknown subagent task id should return False, got {cancelled!r}"
    )

    session.close()
    print("python sdk subagent query api ok")


if __name__ == "__main__":
    main()
