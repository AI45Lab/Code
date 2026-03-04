#!/usr/bin/env python3
"""
A3S Code Python SDK - Bug Fix Verification Tests

Tests for two framework-level bugs fixed in a3s-code-core v1.0.1:
  Bug 1: permissions.allow non-empty caused agent files to silently fail loading
  Bug 2: Sub-agent child sessions had no access to MCP tools

Prerequisites:
  - Rebuild Python SDK after core fixes: `maturin develop --release`
  - MCP echo server: examples/mcp_echo_server.py (bundled, no deps)

Run with: python examples/test_bugfix_permissions_mcp.py
"""

import os
import secrets as _secrets
import sys
import tempfile
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from a3s_code import Agent

_ECHO_SERVER = str(Path(__file__).parent / "mcp_echo_server.py")


class BugfixPermissionsMcpTest:
    """Verification tests for permissions.allow and sub-agent MCP bug fixes."""

    def __init__(self, agent: Agent, config_path: str) -> None:
        self.agent = agent
        self.config_path = config_path

    @staticmethod
    def resolve_config() -> str:
        """Resolve config file path from home directory or project tree."""
        home_cfg = Path.home() / ".a3s" / "config.hcl"
        if home_cfg.exists():
            return str(home_cfg)
        p = Path(__file__).resolve()
        for _ in range(10):
            candidate = p / ".a3s" / "config.hcl"
            if candidate.exists():
                return str(candidate)
            parent = p.parent
            if parent == p:
                break
            p = parent
        raise RuntimeError("Config not found at .a3s/config.hcl or ~/.a3s/config.hcl")

    @staticmethod
    def pass_test(label: str) -> None:
        """Print a passing test label."""
        print(f"  PASS  {label}")

    # ── Bug 1: Agent files with non-empty permissions.allow ───────────────────

    def test_bug1_agent_with_permissions(self, tmpdir: str) -> None:
        """Bug 1: Agent files with non-empty permissions.allow must load and work."""
        print("\n-- Bug 1: Agent loading with non-empty permissions.allow --")

        # Create a test agent directory with agents that have non-empty permissions
        agents_dir = os.path.join(tmpdir, "agents")
        os.makedirs(agents_dir, exist_ok=True)

        # Agent 1: YAML with plain string permissions (previously FAILED)
        with open(os.path.join(agents_dir, "scorer.yaml"), "w") as f:
            f.write("""
name: scorer
description: Scoring agent with permission restrictions
mode: subagent
max_steps: 10
permissions:
  allow:
    - read
    - grep
    - glob
  deny:
    - write
    - edit
""")

        # Agent 2: Markdown with plain string permissions (previously FAILED)
        with open(os.path.join(agents_dir, "reviewer.md"), "w") as f:
            f.write("""---
name: reviewer
description: Code reviewer with restricted permissions
mode: subagent
max_steps: 10
permissions:
  allow:
    - read
    - grep
  deny:
    - write
    - edit
    - bash
---
You are a code reviewer. Read files and provide feedback. Do NOT modify any files.
""")

        # Create session with the custom agent directory
        session = self.agent.session(tmpdir, agent_dirs=[agents_dir], permissive=True)

        # Verify agents are loaded by checking tool_names (task tool should be present)
        tool_names = session.tool_names()
        assert "task" in tool_names, (
            f"task tool should be available, got: {tool_names}"
        )
        self.pass_test("task tool available in session")

        # Test 1a: Use the 'scorer' agent via task tool
        result = session.send(
            "Use the task tool to delegate to the 'scorer' agent with this prompt: "
            "'List the files in the current directory using the glob tool, "
            "then reply with exactly: SCORER_OK'. "
            "Return the scorer's reply verbatim."
        )
        assert "SCORER_OK" in result.text, (
            f"scorer agent should have been loaded and returned SCORER_OK, "
            f"got: {result.text[:200]}"
        )
        self.pass_test("scorer agent loaded and executed (YAML with permissions.allow)")

        # Test 1b: Use the 'reviewer' agent via task tool
        result = session.send(
            "Use the task tool to delegate to the 'reviewer' agent with this prompt: "
            "'Reply with exactly: REVIEWER_OK'. "
            "Return the reviewer's reply verbatim."
        )
        assert "REVIEWER_OK" in result.text, (
            f"reviewer agent should have been loaded and returned REVIEWER_OK, "
            f"got: {result.text[:200]}"
        )
        self.pass_test("reviewer agent loaded and executed (MD with permissions.allow)")

    # ── Bug 2: Sub-agent MCP tool access ──────────────────────────────────────

    def test_bug2_subagent_mcp_tools(self, tmpdir: str) -> None:
        """Bug 2: Sub-agent child sessions must have access to MCP tools."""
        print("\n-- Bug 2: Sub-agent MCP tool access --")

        # Create a simple agent that uses MCP tools
        agents_dir = os.path.join(tmpdir, "mcp_agents")
        os.makedirs(agents_dir, exist_ok=True)

        with open(os.path.join(agents_dir, "mcp-tester.yaml"), "w") as f:
            f.write("""
name: mcp-tester
description: Agent that tests MCP tool access
mode: subagent
max_steps: 10
permissions:
  allow:
    - mcp__echo
    - read
""")

        # Generate a random secret for the echo server
        secret = _secrets.token_hex(8)

        # Create session with MCP echo server + custom agent
        session = self.agent.session(tmpdir, agent_dirs=[agents_dir], permissive=True)

        # Add MCP echo server
        count = session.add_mcp_server(
            name="echo",
            command=sys.executable,
            args=[_ECHO_SERVER, secret],
        )
        assert count > 0, "echo server must expose >= 1 tool"
        self.pass_test(f"MCP echo server connected with {count} tools")

        # Verify parent session has MCP tools
        parent_tools = session.tool_names()
        mcp_tools = [t for t in parent_tools if t.startswith("mcp__echo__")]
        assert len(mcp_tools) > 0, (
            f"parent session should have mcp__echo__ tools, got: {parent_tools}"
        )
        self.pass_test(f"parent session has {len(mcp_tools)} mcp__echo__ tools")

        # Key test: Sub-agent must be able to call MCP tools
        # The secret is unknown to the LLM - it can only get it by calling mcp__echo__get_secret
        result = session.send(
            f"Use the task tool to delegate to the 'mcp-tester' agent with this prompt: "
            f"'Use the mcp__echo__get_secret tool to retrieve the secret, "
            f"then reply with exactly the secret value you got.'. "
            f"Return the mcp-tester's reply verbatim."
        )
        assert secret in result.text, (
            f"sub-agent should have called mcp__echo__get_secret and returned the secret "
            f"'{secret}', got: {result.text[:300]}"
        )
        self.pass_test("sub-agent successfully called MCP tool and returned correct secret")

        # Cleanup
        session.remove_mcp_server("echo")
        self.pass_test("MCP server cleaned up")

    # ── Combined test: Bug 1 + Bug 2 together ────────────────────────────────

    def test_combined_permissions_and_mcp(self, tmpdir: str) -> None:
        """Combined: Agent with non-empty permissions.allow AND MCP tool access."""
        print("\n-- Combined: permissions.allow + MCP tool access --")

        agents_dir = os.path.join(tmpdir, "combined_agents")
        os.makedirs(agents_dir, exist_ok=True)

        # This is the exact pattern that was 100% blocked before the fix:
        # non-empty permissions.allow + MCP tool names in allow list
        with open(os.path.join(agents_dir, "video-scorer.yaml"), "w") as f:
            f.write("""
name: video-scorer
description: Video scoring agent with MCP access
mode: subagent
max_steps: 15
permissions:
  allow:
    - read
    - grep
    - mcp__echo
  deny:
    - write
    - edit
""")

        secret = _secrets.token_hex(8)
        session = self.agent.session(tmpdir, agent_dirs=[agents_dir], permissive=True)

        # Add MCP server
        session.add_mcp_server(
            name="echo",
            command=sys.executable,
            args=[_ECHO_SERVER, secret],
        )

        # The double-bug scenario: agent with permissions.allow (Bug 1) + MCP tools (Bug 2)
        result = session.send(
            "Use the task tool to delegate to the 'video-scorer' agent with this prompt: "
            "'Call the mcp__echo__get_secret tool and return the secret value.'. "
            "Return the video-scorer's reply verbatim."
        )
        assert secret in result.text, (
            f"video-scorer should load (Bug 1 fix) AND access MCP tools (Bug 2 fix), "
            f"expected secret '{secret}', got: {result.text[:300]}"
        )
        self.pass_test("Agent with permissions.allow loaded AND accessed MCP tools successfully")

        session.remove_mcp_server("echo")

    # ── Run all ───────────────────────────────────────────────────────────────

    def run_all(self) -> None:
        """Run all bug fix verification tests."""
        print("=== A3S Code Python SDK - Bug Fix Verification Tests ===")
        print(f"config: {self.config_path}\n")

        with tempfile.TemporaryDirectory() as tmpdir:
            tests = [
                lambda: self.test_bug1_agent_with_permissions(tmpdir),
                lambda: self.test_bug2_subagent_mcp_tools(tmpdir),
                lambda: self.test_combined_permissions_and_mcp(tmpdir),
            ]

            passed = 0
            failed = 0
            for test in tests:
                try:
                    test()
                    passed += 1
                except Exception as e:
                    print(f"\n  FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    failed += 1

        print("\n" + "=" * 60)
        if failed == 0:
            print(f"=== ALL {passed} TESTS PASSED ===")
        else:
            print(f"=== {passed} passed, {failed} FAILED ===")
            sys.exit(1)
        print("=" * 60)


if __name__ == "__main__":
    config_path = BugfixPermissionsMcpTest.resolve_config()
    agent = Agent.create(config_path)
    test = BugfixPermissionsMcpTest(agent, config_path)
    test.run_all()
