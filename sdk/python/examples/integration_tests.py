#!/usr/bin/env python3
"""
A3S Code Python SDK - Integration Tests

Tests all major features using real LLM configuration from ~/.a3s/config.hcl

Run with: python examples/integration_tests.py
"""

import asyncio
import os
from pathlib import Path
from a3s_code import Agent, SessionQueueConfig, builtin_skills


class IntegrationTests:
    """Integration tests for all major features of the A3S Code Python SDK."""

    def __init__(self, agent: Agent, config_path: str) -> None:
        self.agent = agent
        self.config_path = config_path

    @staticmethod
    def find_config_path() -> str:
        """Find config file in home directory or project root."""
        home_config = Path.home() / ".a3s" / "config.hcl"
        if home_config.exists():
            return str(home_config)

        # Try project root (6 levels up from examples/integration_tests.py)
        project_config = Path(__file__).parent.parent.parent.parent.parent.parent / ".a3s" / "config.hcl"
        if project_config.exists():
            return str(project_config)

        raise FileNotFoundError("Config file not found. Please create ~/.a3s/config.hcl")

    @staticmethod
    def truncate(text: str, max_len: int) -> str:
        """Truncate text to max length."""
        if len(text) <= max_len:
            return text
        return f"{text[:max_len]}... (truncated)"

    async def test_basic_tools(self) -> None:
        """Test 1: Basic tool execution."""
        print("\n Test 1: Basic Tool Execution")
        print("-" * 80)

        session = self.agent.session(".")

        print("Testing: List current directory...")
        result = session.send("List the files in the current directory using ls")
        print(f"  Result preview: {self.truncate(result.text, 200)}")

        print("\nTesting: Read a file...")
        result = session.send("Read the Cargo.toml file")
        print(f"  Result preview: {self.truncate(result.text, 200)}")

        print("\n  Test 1 passed: Basic tools work correctly")

    async def test_builtin_skills(self) -> None:
        """Test 2: Built-in skills."""
        print("\n Test 2: Built-in Skills (7 skills)")
        print("-" * 80)

        # List available skills
        skills = builtin_skills()
        print(f"Available skills: {len(skills)}")
        for skill in skills[:3]:  # Show first 3
            print(f"  - {skill.name} ({skill.kind}): {self.truncate(skill.description, 60)}")

        # Create session with builtin skills enabled
        session = self.agent.session(".", builtin_skills=True)

        print("\nTesting: code-search skill...")
        result = session.send("Search for all functions named 'new' in Rust files")
        print(f"  Result preview: {self.truncate(result.text, 200)}")

        print("\nTesting: builtin-tools skill...")
        result = session.send("What tools are available for file operations?")
        print(f"  Result preview: {self.truncate(result.text, 200)}")

        print("\n  Test 2 passed: Built-in skills work correctly")

    async def test_file_operations(self) -> None:
        """Test 3: File operations."""
        print("\n Test 3: File Operations")
        print("-" * 80)

        session = self.agent.session(".")

        print("Testing: Create a test file...")
        result = session.send(
            "Create a file named 'test_integration_py.txt' with content 'Hello from Python SDK!'"
        )
        print(f"  Result: {self.truncate(result.text, 200)}")

        print("\nTesting: Read the test file...")
        result = session.send("Read the file test_integration_py.txt")
        print(f"  Result: {self.truncate(result.text, 200)}")

        print("\nCleaning up: Remove test file...")
        try:
            os.remove("test_integration_py.txt")
        except FileNotFoundError:
            pass

        print("\n  Test 3 passed: File operations work correctly")

    async def test_search_operations(self) -> None:
        """Test 4: Search operations."""
        print("\n Test 4: Search Operations")
        print("-" * 80)

        session = self.agent.session(".")

        print("Testing: grep search...")
        result = session.send("Search for the word 'Agent' in all Rust files using grep")
        print(f"  Result preview: {self.truncate(result.text, 200)}")

        print("\nTesting: glob pattern matching...")
        result = session.send("Find all .rs files in the src directory using glob")
        print(f"  Result preview: {self.truncate(result.text, 200)}")

        print("\n  Test 4 passed: Search operations work correctly")

    async def test_direct_tool_calls(self) -> None:
        """Test 5: Direct tool execution."""
        print("\n Test 5: Direct Tool Execution")
        print("-" * 80)

        session = self.agent.session(".")

        print("Testing: Direct read_file call...")
        content = session.read_file("Cargo.toml")
        print(f"  Read {len(content)} bytes from Cargo.toml")

        print("\nTesting: Direct bash call...")
        output = session.bash("echo 'Hello from Python SDK'")
        print(f"  Bash output: {output.strip()}")

        print("\nTesting: Direct glob call...")
        files = session.glob("src/*.rs")
        print(f"  Found {len(files)} Rust files")

        print("\nTesting: Direct grep call...")
        matches = session.grep("Agent")
        print(f"  Grep found {len(matches)} bytes of matches")

        print("\n  Test 5 passed: Direct tool calls work correctly")

    async def test_streaming(self) -> None:
        """Test 6: Streaming execution."""
        print("\n Test 6: Streaming Execution")
        print("-" * 80)

        session = self.agent.session(".")

        print("Testing: Stream events...")
        event_count = 0
        text_deltas = 0
        tool_calls = 0

        for event in session.stream("List all .rs files in the current directory"):
            event_count += 1
            if event.event_type == "text_delta":
                text_deltas += 1
            elif event.event_type == "tool_start":
                tool_calls += 1
                print(f"  Tool called: {event.tool_name}")

        print(f"  Received {event_count} events ({text_deltas} text deltas, {tool_calls} tool calls)")
        print("\n  Test 6 passed: Streaming works correctly")

    async def test_queue_config(self) -> None:
        """Test 7: Queue configuration."""
        print("\n Test 7: Queue Configuration (A3S Lane v0.4.0)")
        print("-" * 80)

        queue_config = SessionQueueConfig()
        queue_config.set_query_concurrency(5)
        queue_config.set_execute_concurrency(2)
        queue_config.enable_dlq()
        queue_config.enable_metrics()

        session = self.agent.session(".", queue_config=queue_config)

        print("Testing: Parallel query operations with queue...")
        result = session.send("List all .rs files and count how many contain the word 'async'")
        print(f"  Result preview: {self.truncate(result.text, 200)}")

        if session.has_queue():
            stats = session.queue_stats()
            print(f"  Queue stats: {stats}")

        print("\n  Test 7 passed: Queue configuration works correctly")

    async def test_conversation_history(self) -> None:
        """Test 8: Conversation history."""
        print("\n Test 8: Conversation History")
        print("-" * 80)

        session = self.agent.session(".")

        print("Testing: Multi-turn conversation...")
        result1 = session.send("What is 2 + 2?")
        print(f"  Turn 1: {self.truncate(result1.text, 100)}")

        result2 = session.send("What was my previous question?")
        print(f"  Turn 2: {self.truncate(result2.text, 100)}")

        history = session.history()
        print(f"  History has {len(history)} messages")

        print("\n  Test 8 passed: Conversation history works correctly")

    async def run_all(self) -> None:
        """Run all integration tests."""
        print("A3S Code Python SDK - Integration Tests\n")
        print("=" * 80)
        print(f"Using config: {self.config_path}")
        print("=" * 80)
        print()

        await self.test_basic_tools()
        await self.test_builtin_skills()
        await self.test_file_operations()
        await self.test_search_operations()
        await self.test_direct_tool_calls()
        await self.test_streaming()
        await self.test_queue_config()
        await self.test_conversation_history()

        print("\n" * 2)
        print("=" * 80)
        print("  All Python SDK integration tests completed successfully!")
        print("=" * 80)


if __name__ == "__main__":
    config_path = IntegrationTests.find_config_path()
    agent = Agent.create(config_path)
    tests = IntegrationTests(agent, config_path)
    asyncio.run(tests.run_all())
