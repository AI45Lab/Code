"""Smoke test for the budget overload on `Session.parallel(...)`.

Verifies that `parallel(specs)` returns the plain outcomes list and
`parallel(specs, budget_tokens=...)` returns the richer
`{"outcomes": [...], "budget": {...}}` shape — checked on the empty fan-out,
which takes no LLM path. Mirrors the Node SDK smoke in sdk/node/test.mjs. Full
orchestration behavior is covered by the core crate's real-LLM integration tests.

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
    workspace = tempfile.mkdtemp(prefix="a3s-code-python-parallel-budget-")
    agent = Agent.create(INLINE_CONFIG)

    opts = SessionOptions()
    opts.permission_policy = PermissionPolicy(default_decision="allow")
    opts.workspace_backend = LocalWorkspaceBackend(workspace)

    session = agent.session(workspace, opts)

    # No budget -> the plain list of outcome dicts (unchanged behavior).
    plain = session.parallel([])
    assert plain == [], f"no budget -> plain list, got {plain!r}"

    # With a budget -> {outcomes, budget}. Empty fan-out takes no LLM path.
    budgeted = session.parallel([], budget_tokens=50000)
    assert budgeted["outcomes"] == [], f"empty specs -> empty outcomes, got {budgeted!r}"
    assert (
        budgeted["budget"]["consumed_tokens"] == 0
    ), f"no spend yet, got {budgeted['budget']!r}"
    assert (
        budgeted["budget"]["limit_tokens"] == 50000
    ), f"limit reflected in the ledger, got {budgeted['budget']!r}"

    session.close()
    print("python sdk parallel budget overload ok")


if __name__ == "__main__":
    main()
