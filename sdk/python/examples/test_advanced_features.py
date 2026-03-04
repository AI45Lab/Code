#!/usr/bin/env python3
"""
A3S Code — Advanced Features Demo

Demonstrates: auto-compact, memory, security, hooks, planning,
permission policy, and resilience options.

Usage:
    cd crates/code/sdk/python
    python examples/test_advanced_features.py

Requires a valid LLM API key in ~/.a3s/config.hcl or $A3S_CONFIG.
"""

import asyncio
import os
import tempfile
import shutil
from pathlib import Path

import a3s_code


class AdvancedFeaturesTest:
    """Tests for advanced features: auto-compact, memory, security, hooks, etc."""

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path

    @staticmethod
    def find_config_path() -> str:
        if config := os.environ.get("A3S_CONFIG"):
            return config
        home_config = Path.home() / ".a3s" / "config.hcl"
        if home_config.exists():
            return str(home_config)
        raise RuntimeError("No config found. Set A3S_CONFIG or create ~/.a3s/config.hcl")

    # ========================================================================
    # Test 1: Auto-Compact
    # ========================================================================

    async def test_auto_compact(self) -> None:
        print("\n=== Test 1: Auto-Compact ===")
        agent = await a3s_code.Agent.create(self.config_path)

        opts = a3s_code.SessionOptions()
        opts.auto_compact = True
        opts.auto_compact_threshold = 0.8

        session = agent.session(tempfile.gettempdir(), options=opts)
        result = await session.send("Say hello in one sentence.")
        print(f"  auto-compact session created, response: {result.text[:60]}")

    # ========================================================================
    # Test 2: File Memory
    # ========================================================================

    async def test_memory(self) -> None:
        print("\n=== Test 2: File Memory ===")
        agent = await a3s_code.Agent.create(self.config_path)
        mem_dir = tempfile.mkdtemp(prefix="a3s-mem-")

        try:
            opts = a3s_code.SessionOptions()
            opts.memory_dir = mem_dir

            session = agent.session(tempfile.gettempdir(), options=opts)
            result = await session.send("Remember: my favorite language is Rust.")
            print(f"  memory session created, response: {result.text[:60]}")
        finally:
            shutil.rmtree(mem_dir, ignore_errors=True)

    # ========================================================================
    # Test 3: Default Security Provider
    # ========================================================================

    async def test_security(self) -> None:
        print("\n=== Test 3: Security Provider ===")
        agent = await a3s_code.Agent.create(self.config_path)

        opts = a3s_code.SessionOptions()
        opts.default_security = True

        session = agent.session(tempfile.gettempdir(), options=opts)
        result = await session.send("What is 2 + 2?")
        print(f"  security session created, response: {result.text[:60]}")

    # ========================================================================
    # Test 4: Hooks — intercept lifecycle events
    # ========================================================================

    async def test_hooks(self) -> None:
        print("\n=== Test 4: Hooks ===")
        agent = await a3s_code.Agent.create(self.config_path)
        session = agent.session(tempfile.gettempdir())

        tool_calls: list[str] = []

        def on_pre_tool_use(event: dict) -> None:
            tool_calls.append(event.get("tool_name", "unknown"))
            print(f"  -> pre_tool_use: {str(event)[:80]}")
            return None  # allow execution

        session.register_hook("log-tools", "pre_tool_use", on_pre_tool_use)
        print(f"  hook registered, hook count: {session.hook_count()}")

        session.unregister_hook("log-tools")
        print(f"  hook unregistered, hook count: {session.hook_count()}")

    # ========================================================================
    # Test 5: Planning Mode
    # ========================================================================

    async def test_planning(self) -> None:
        print("\n=== Test 5: Planning Mode ===")
        agent = await a3s_code.Agent.create(self.config_path)

        session = agent.session(
            tempfile.gettempdir(),
            planning=True,
            goal_tracking=True,
        )
        result = await session.send("List 3 benefits of Rust in one sentence each.")
        print(f"  planning session, response length: {len(result.text)}")

    # ========================================================================
    # Test 6: Permissive Policy
    # ========================================================================

    async def test_permissive(self) -> None:
        print("\n=== Test 6: Permissive Policy ===")
        agent = await a3s_code.Agent.create(self.config_path)

        session = agent.session(tempfile.gettempdir(), permissive=True)
        result = await session.send("What is the current directory?")
        print(f"  permissive session, response: {result.text[:60]}")

    # ========================================================================
    # Test 7: Resilience Options
    # ========================================================================

    async def test_resilience(self) -> None:
        print("\n=== Test 7: Resilience Options ===")
        agent = await a3s_code.Agent.create(self.config_path)

        session = agent.session(
            tempfile.gettempdir(),
            max_parse_retries=3,
            tool_timeout_ms=30000,
            circuit_breaker_threshold=5,
        )
        result = await session.send('Say "resilient" once.')
        print(f"  resilience session, response: {result.text[:60]}")

    # ========================================================================
    # Test 8: Combined — security + auto-compact + memory + hooks
    # ========================================================================

    async def test_combined(self) -> None:
        print("\n=== Test 8: Combined Features ===")
        agent = await a3s_code.Agent.create(self.config_path)
        mem_dir = tempfile.mkdtemp(prefix="a3s-combined-")

        try:
            opts = a3s_code.SessionOptions()
            opts.default_security = True
            opts.auto_compact = True
            opts.auto_compact_threshold = 0.9
            opts.memory_dir = mem_dir

            session = agent.session(tempfile.gettempdir(), options=opts, permissive=True)

            def on_post_tool_use(event: dict) -> None:
                print("  -> post_tool_use fired")
                return None

            session.register_hook("audit", "post_tool_use", on_post_tool_use)

            result = await session.send("What is 1 + 1?")
            print(f"  combined session, response: {result.text[:60]}")
        finally:
            shutil.rmtree(mem_dir, ignore_errors=True)

    # ========================================================================
    # Run All Tests
    # ========================================================================

    async def run_all(self) -> int:
        print("A3S Code -- Advanced Features Test")
        print("===================================")

        tests = [
            self.test_auto_compact,
            self.test_memory,
            self.test_security,
            self.test_hooks,
            self.test_planning,
            self.test_permissive,
            self.test_resilience,
            self.test_combined,
        ]

        passed = 0
        failed = 0

        for test in tests:
            try:
                await test()
                passed += 1
            except Exception as e:
                print(f"  {test.__name__} failed: {e}")
                failed += 1

        print(f"\n===================================")
        print(f"Results: {passed} passed, {failed} failed")
        return failed


if __name__ == "__main__":
    config_path = AdvancedFeaturesTest.find_config_path()
    suite = AdvancedFeaturesTest(config_path)
    failed = asyncio.run(suite.run_all())
    raise SystemExit(failed)
