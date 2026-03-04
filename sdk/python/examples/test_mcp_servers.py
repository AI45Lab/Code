#!/usr/bin/env python3
"""
A3S Code Python SDK - Integration Tests

Tests all recently added/fixed features against the real kimi endpoint:
  1. task tool (delegate to general subagent, wait for result)
  2. parallel_task (fan-out to multiple subagents concurrently)
  3. tool_names() initial state on a fresh session
  4. mcp_status error field populated on failed connect
  5. MCP injection: add → status → LLM use → tool_names → remove
  6. refresh_mcp_tools (smoke test)

Run with: python examples/test_mcp_servers.py
"""

import secrets as _secrets
import sys
import tempfile
from pathlib import Path

# Ensure UTF-8 output on Windows (GBK default doesn't support ✓/──)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from a3s_code import Agent

# Path to the bundled minimal MCP echo server (no external deps required)
_ECHO_SERVER = str(Path(__file__).parent / "mcp_echo_server.py")


class McpServersTest:
    """Integration tests for MCP server functionality."""

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
        print(f"  ✓  {label}")

    # ── Test 1: task tool ─────────────────────────────────────────────────────

    def test_task_tool(self, tmpdir: str) -> None:
        """Test 1: task tool (subagent delegation)."""
        print("\n── Test 1: task tool (subagent delegation) ──")
        session = self.agent.session(tmpdir, permissive=True)
        result = session.send(
            "Use the task tool to delegate the following to the 'general' agent, "
            "then return its exact reply verbatim: "
            "'Reply with exactly the word: TASK_OK'"
        )
        assert "TASK_OK" in result.text, (
            f"task tool should have returned TASK_OK, got: {result.text}"
        )
        self.pass_test("task tool delegated and returned TASK_OK")

    # ── Test 2: parallel_task ─────────────────────────────────────────────────

    def test_parallel_task(self, tmpdir: str) -> None:
        """Test 2: parallel_task (concurrent fan-out)."""
        print("\n── Test 2: parallel_task (concurrent fan-out) ──")
        session = self.agent.session(tmpdir, permissive=True)
        result = session.send(
            "Use the parallel_task tool to run these three tasks concurrently "
            "using the 'general' agent, then list their outputs:\n"
            "1. Reply with exactly: PARALLEL_A\n"
            "2. Reply with exactly: PARALLEL_B\n"
            "3. Reply with exactly: PARALLEL_C"
        )
        for token in ("PARALLEL_A", "PARALLEL_B", "PARALLEL_C"):
            assert token in result.text, (
                f"expected {token} in result, got: {result.text}"
            )
        self.pass_test("parallel_task ran 3 subagents concurrently, all results returned")

    # ── Test 3: tool_names initial state ──────────────────────────────────────

    def test_tool_names(self, tmpdir: str) -> None:
        """Test 3: tool_names() initial state."""
        print("\n── Test 3: tool_names() initial state ──")
        session = self.agent.session(tmpdir, permissive=True)

        names = session.tool_names()
        assert len(names) > 0, "expected built-in tools to be present"
        self.pass_test(f"tool_names() returns {len(names)} built-in tools")

        assert not any(n.startswith("mcp__") for n in names), (
            f"expected no mcp__ tools on a fresh session (none configured globally), "
            f"got: {[n for n in names if n.startswith('mcp__')]}"
        )
        self.pass_test("no mcp__ tools on fresh session (none configured globally)")

    # ── Test 4: mcp_status error field ────────────────────────────────────────

    def test_mcp_status_error(self, tmpdir: str) -> None:
        """Test 4: mcp_status error field on failed connect."""
        print("\n── Test 4: mcp_status error field on failed connect ──")
        session = self.agent.session(tmpdir, permissive=True)

        err = None
        try:
            session.add_mcp_server(
                name="bad-server",
                command="nonexistent-mcp-binary-xyz",
                args=[],
            )
        except Exception as e:
            err = e
        assert err is not None, "add_mcp_server must raise for a nonexistent binary"

        # The config is registered before connection is attempted, so the server
        # must always appear in mcp_status — even on failure.
        status = session.mcp_status()
        assert "bad-server" in status, (
            f"bad-server must appear in mcp_status after failed connect, "
            f"got keys: {list(status.keys())}"
        )
        s = status["bad-server"]
        assert not s["connected"], "bad-server should not be connected"
        assert s.get("error"), (
            f"error field must be populated after failed connect, got: {s}"
        )
        self.pass_test(f"mcp_status.error captured: {s['error']}")

    # ── Test 5: MCP injection ─────────────────────────────────────────────────

    def test_mcp_injection(self, tmpdir: str) -> None:
        """Test 5: MCP injection (add -> status -> LLM use -> tool_names -> remove)."""
        print("\n── Test 5: MCP injection (add → status → LLM use → tool_names → remove) ──")
        session = self.agent.session(tmpdir, permissive=True)

        # Before add: no mcp__echo__ tools
        tools_before = session.tool_names()
        assert not any(n.startswith("mcp__echo__") for n in tools_before), (
            f"expected no mcp__echo__ tools before add_mcp_server, "
            f"got: {[n for n in tools_before if n.startswith('mcp__echo__')]}"
        )
        self.pass_test("no mcp__echo__ tools before add_mcp_server")

        # Generate a random secret unknown to the LLM — it can only be obtained by
        # actually calling mcp__echo__get_secret, preventing the LLM from faking it.
        secret = _secrets.token_hex(8)

        # Add the bundled echo server (uses the same Python interpreter — no external deps)
        count = session.add_mcp_server(
            name="echo",
            command=sys.executable,
            args=[_ECHO_SERVER, secret],
        )
        assert count > 0, "echo server must expose >= 1 tool"
        self.pass_test(f"add_mcp_server registered {count} tools")

        # Verify mcp_status: connected=true, tool_count=N, error=None
        status = session.mcp_status()
        assert "echo" in status, (
            f"echo must appear in mcp_status, got keys: {list(status.keys())}"
        )
        s = status["echo"]
        assert s["connected"], "echo server should be connected"
        assert s["tool_count"] == count, (
            f"mcp_status.tool_count should be {count}, got {s['tool_count']}"
        )
        assert s.get("error") is None, (
            f"no error expected for successful connect, got: {s.get('error')}"
        )
        self.pass_test(f"mcp_status: connected=true, tool_count={count}, error=None")

        # Verify tool_names reflects the injected tools
        tools_after = session.tool_names()
        mcp_tools = [n for n in tools_after if n.startswith("mcp__echo__")]
        assert len(mcp_tools) == count, (
            f"tool_names should show {count} mcp__echo__ tools, got {mcp_tools}"
        )
        self.pass_test(f"tool_names shows {len(mcp_tools)} mcp__echo__ tools")

        # LLM must actually call the MCP tool to retrieve the secret.
        # The secret value is NOT in the prompt — the LLM cannot fake this.
        result = session.send(
            "Use the mcp__echo__get_secret tool to retrieve the secret value, "
            "then tell me exactly what it returned."
        )
        assert secret in result.text, (
            f"LLM should have called mcp__echo__get_secret and returned the secret, "
            f"got: {result.text}"
        )
        self.pass_test("LLM used mcp__echo__get_secret tool and returned the correct secret")

        # Remove server: tools disappear
        session.remove_mcp_server("echo")
        tools_final = session.tool_names()
        assert not any(n.startswith("mcp__echo__") for n in tools_final), (
            "mcp__echo__ tools should be gone after remove_mcp_server"
        )
        self.pass_test("remove_mcp_server removed all mcp__echo__ tools")

    # ── Test 6: refresh_mcp_tools ─────────────────────────────────────────────

    def test_refresh_mcp_tools(self) -> None:
        """Test 6: refresh_mcp_tools (smoke test)."""
        print("\n── Test 6: refresh_mcp_tools (smoke test) ──")
        self.agent.refresh_mcp_tools()
        self.pass_test("refresh_mcp_tools completed without error")

    # ── Run all ───────────────────────────────────────────────────────────────

    def run_all(self) -> None:
        """Run all MCP integration tests."""
        print("=== A3S Code Python SDK - Integration Tests ===")
        print(f"config: {self.config_path}\n")

        with tempfile.TemporaryDirectory() as tmpdir:
            tests = [
                lambda: self.test_task_tool(tmpdir),
                lambda: self.test_parallel_task(tmpdir),
                lambda: self.test_tool_names(tmpdir),
                lambda: self.test_mcp_status_error(tmpdir),
                lambda: self.test_mcp_injection(tmpdir),
                lambda: self.test_refresh_mcp_tools(),
            ]

            passed = 0
            failed = 0
            for test in tests:
                try:
                    test()
                    passed += 1
                except Exception as e:
                    print(f"\n  FAILED: {e}")
                    failed += 1

        print("\n" + "=" * 48)
        if failed == 0:
            print(f"=== all {passed} tests passed ===")
        else:
            print(f"=== {passed} passed, {failed} failed ===")
            sys.exit(1)
        print("=" * 48)


if __name__ == "__main__":
    config_path = McpServersTest.resolve_config()
    agent = Agent.create(config_path)
    test = McpServersTest(agent, config_path)
    test.run_all()
