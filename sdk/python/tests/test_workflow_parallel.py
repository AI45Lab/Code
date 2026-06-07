"""Smoke test for the budgeted workflow fan-out exposed on Session.

Verifies `session.workflow_parallel(...)` is reachable from Python and returns
the expected shape (outcomes + shared budget ledger snapshot) for the empty
fan-out, which takes no LLM path. Mirrors the Node SDK smoke in
sdk/node/test.mjs. Full orchestration behavior is covered by the core crate's
real-LLM integration tests.

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
    workspace = tempfile.mkdtemp(prefix="a3s-code-python-workflow-")
    agent = Agent.create(INLINE_CONFIG)

    opts = SessionOptions()
    opts.permission_policy = PermissionPolicy(default_decision="allow")
    opts.workspace_backend = LocalWorkspaceBackend(workspace)

    session = agent.session(workspace, opts)

    assert hasattr(
        session, "workflow_parallel"
    ), "Session should expose workflow_parallel"

    # An empty fan-out takes no LLM path: outcomes empty, ledger snapshot present.
    capped = session.workflow_parallel([], budget_tokens=50000)
    assert capped["outcomes"] == [], f"empty specs -> empty outcomes, got {capped!r}"
    assert (
        capped["budget"]["consumed_tokens"] == 0
    ), f"no spend yet, got {capped['budget']!r}"
    assert (
        capped["budget"]["limit_tokens"] == 50000
    ), f"limit reflected in the ledger, got {capped['budget']!r}"

    uncapped = session.workflow_parallel([])
    assert (
        uncapped["budget"]["limit_tokens"] is None
    ), f"uncapped -> None limit, got {uncapped['budget']!r}"

    session.close()
    print("python sdk workflow_parallel ok")


if __name__ == "__main__":
    main()
