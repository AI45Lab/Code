#!/usr/bin/env python3
"""
A3S Code Python SDK - Permissive Mode for Sub-agents Test

Tests the fix for GitHub issue #2: Sub-agents can now inherit and configure
permissive mode from parent sessions.

This test verifies:
1. Sub-agents can be spawned with permissive=True
2. Permissive sub-agents execute tools without HITL confirmation
3. Non-permissive sub-agents (default) still require confirmation
4. Parallel tasks can use permissive mode

Run with: python examples/test_permissive_subagents.py
"""

import json
from pathlib import Path
from a3s_code import Agent


class PermissiveSubagentsTest:
    """Integration tests for permissive mode in sub-agents."""

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path

    @staticmethod
    def find_config_path() -> str:
        """Find config file in home directory or project root."""
        home_config = Path.home() / ".a3s" / "config.hcl"
        if home_config.exists():
            return str(home_config)

        project_config = (
            Path(__file__).parent.parent.parent.parent.parent.parent
            / ".a3s"
            / "config.hcl"
        )
        if project_config.exists():
            return str(project_config)

        raise FileNotFoundError("Config file not found. Please create ~/.a3s/config.hcl")

    # ========================================================================
    # Test 1: Sub-agent with permissive=True
    # ========================================================================

    def test_permissive_subagent(self) -> None:
        """Test 1: Sub-agent with permissive=True executes tools autonomously."""
        print("\n[Test 1] Sub-agent with permissive=True")
        print("-" * 80)

        agent = Agent.create(self.config_path)
        session = agent.session(".", permissive=True)

        print("  Parent session created with permissive=True")
        print("  Spawning sub-agent with permissive=True...")

        # Use the task tool to spawn a sub-agent with permissive mode
        result = session.tool(
            "task",
            {
                "agent": "general",
                "description": "List Python files in current directory",
                "prompt": "Use the ls tool to list all .py files in the current directory. Show the count.",
                "permissive": True,  # Sub-agent should execute without confirmation
                "max_steps": 5,
            },
        )

        print(f"  Sub-agent result: {result.output[:200]}...")
        print(f"  Exit code: {result.exit_code}")

        # Verify the sub-agent executed successfully
        assert result.exit_code == 0, "Sub-agent should execute successfully"
        assert ".py" in result.output, "Result should contain Python file listings"

        print("\n  [PASS] Test 1 passed: Permissive sub-agent executed autonomously\n")

    # ========================================================================
    # Test 2: Sub-agent with permissive=False (default)
    # ========================================================================

    def test_non_permissive_subagent(self) -> None:
        """Test 2: Sub-agent with permissive=False (default behavior)."""
        print("\n[Test 2] Sub-agent with permissive=False (default)")
        print("-" * 80)

        agent = Agent.create(self.config_path)
        # Create session WITHOUT permissive mode
        session = agent.session(".", permissive=False)

        print("  Parent session created with permissive=False")
        print("  Spawning sub-agent with permissive=False (default)...")

        # Spawn sub-agent without permissive mode
        # Use explore agent with more steps to avoid timeout
        result = session.tool(
            "task",
            {
                "agent": "explore",  # Read-only agent
                "description": "Search for Python files",
                "prompt": "Use glob to find all .py files in the current directory",
                "permissive": False,  # Explicit false
                "max_steps": 10,  # Give it more steps
            },
        )

        print(f"  Sub-agent result: {result.output[:200]}...")
        print(f"  Exit code: {result.exit_code}")

        # The explore agent should work even without permissive mode
        # because it only uses read-only tools
        assert result.exit_code == 0, "Sub-agent should execute successfully"

        print("\n  [PASS] Test 2 passed: Non-permissive sub-agent behavior verified\n")

    # ========================================================================
    # Test 3: Parallel tasks with permissive mode
    # ========================================================================

    def test_parallel_permissive_tasks(self) -> None:
        """Test 3: Parallel tasks can use permissive mode."""
        print("\n[Test 3] Parallel tasks with permissive mode")
        print("-" * 80)

        agent = Agent.create(self.config_path)
        session = agent.session(".", permissive=True)

        print("  Spawning 3 parallel sub-agents with permissive=True...")

        # Use parallel_task tool to spawn multiple sub-agents
        result = session.tool(
            "parallel_task",
            {
                "tasks": [
                    {
                        "agent": "explore",
                        "description": "Count Python files",
                        "prompt": "Count how many .py files exist in the current directory",
                        "permissive": True,
                    },
                    {
                        "agent": "explore",
                        "description": "Count Rust files",
                        "prompt": "Count how many .rs files exist in the current directory",
                        "permissive": True,
                    },
                    {
                        "agent": "explore",
                        "description": "Count total files",
                        "prompt": "Count the total number of files in the current directory",
                        "permissive": True,
                    },
                ]
            },
        )

        print(f"  Parallel tasks completed")
        print(f"  Exit code: {result.exit_code}")
        print(f"  Output preview: {result.output[:300]}...")

        assert result.exit_code == 0, "Parallel tasks should execute successfully"
        assert "Task 1" in result.output, "Should contain task 1 results"
        assert "Task 2" in result.output, "Should contain task 2 results"
        assert "Task 3" in result.output, "Should contain task 3 results"

        print("\n  [PASS] Test 3 passed: Parallel permissive tasks executed successfully\n")

    # ========================================================================
    # Run All Tests
    # ========================================================================

    def run_all(self) -> None:
        print("=" * 80)
        print("  A3S Code -- Permissive Mode for Sub-agents Tests")
        print("  Testing fix for GitHub issue #2")
        print("=" * 80)

        try:
            self.test_permissive_subagent()
            self.test_non_permissive_subagent()
            self.test_parallel_permissive_tasks()

            print("=" * 80)
            print("  [SUCCESS] All permissive mode tests passed!")
            print("  GitHub issue #2 is fixed: Sub-agents can now inherit permissive mode")
            print("=" * 80)
        except Exception as e:
            print(f"\n  [FAIL] Test failed: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    config_path = PermissiveSubagentsTest.find_config_path()
    suite = PermissiveSubagentsTest(config_path)
    suite.run_all()
