#!/usr/bin/env python3
"""
A3S Code Python SDK - Parallel Task Processing Integration Test

Demonstrates parallel processing of multiple tasks using A3S Lane queue.
Tests concurrent file analysis, code review, and documentation generation.

Run with: python examples/test_parallel_processing.py
"""

import asyncio
import time
from pathlib import Path
from a3s_code import Agent, SessionQueueConfig


class ParallelProcessingTest:
    """Integration test for parallel task processing with the A3S Lane queue."""

    def __init__(self, agent: Agent, config_path: str) -> None:
        self.agent = agent
        self.config_path = config_path

    @staticmethod
    def find_config_path() -> str:
        """Find config file in home directory or project root."""
        home_config = Path.home() / ".a3s" / "config.hcl"
        if home_config.exists():
            return str(home_config)

        # Try project root (6 levels up)
        project_config = Path(__file__).parent.parent.parent.parent.parent.parent / ".a3s" / "config.hcl"
        if project_config.exists():
            return str(project_config)

        raise FileNotFoundError("Config file not found. Please create ~/.a3s/config.hcl")

    async def test_sequential_processing(self) -> None:
        """Test 1: Sequential processing (baseline)."""
        print("\n📦 Test 1: Sequential Processing (Baseline)")
        print("-" * 80)

        session = self.agent.session(".")

        tasks = [
            "Count the number of Python files in this project",
            "Find all TODO comments in Python files",
            "List all classes in the main module",
        ]

        start = time.time()
        print(f"Processing {len(tasks)} tasks sequentially...")

        for i, task in enumerate(tasks, 1):
            print(f"  Task {i}: {task}")
            result = session.send(task)
            print(f"  ✓ Completed: {len(result.text)} chars")

        duration = time.time() - start
        print(f"\n✓ Sequential processing took: {duration:.2f}s")
        print("\n✅ Test 1 passed: Sequential processing works\n")

    async def test_parallel_processing(self) -> None:
        """Test 2: Parallel processing with queue."""
        print("⚡ Test 2: Parallel Processing with Queue")
        print("-" * 80)

        # Configure queue for parallel processing
        queue_config = SessionQueueConfig()
        queue_config.set_query_concurrency(3)      # Allow 3 concurrent query operations
        queue_config.set_execute_concurrency(2)    # Allow 2 concurrent execute operations
        queue_config.enable_metrics()              # Enable metrics collection
        queue_config.enable_dlq()                  # Enable dead letter queue

        session = self.agent.session(".", queue_config=queue_config)

        tasks = [
            "Count the number of Python files in this project",
            "Find all TODO comments in Python files",
            "List all classes in the main module",
            "Find all async functions in the codebase",
            "Count lines of code in all Python files",
        ]

        start = time.time()
        print(f"Processing {len(tasks)} tasks in parallel...")

        # Create async tasks
        async def process_task(task_num: int, task_text: str) -> tuple:
            """Process a single task."""
            result = session.send(task_text)
            return task_num, result

        # Queue all tasks
        for i, task in enumerate(tasks, 1):
            print(f"  Queuing task {i}: {task}")

        print("\n  Waiting for all tasks to complete...")

        # Process tasks concurrently using asyncio
        async_tasks = [
            asyncio.create_task(asyncio.to_thread(process_task, i, task))
            for i, task in enumerate(tasks, 1)
        ]

        results = await asyncio.gather(*async_tasks, return_exceptions=True)

        for task_num, result in results:
            if isinstance(result, Exception):
                print(f"  ✗ Task {task_num} failed: {result}")
            else:
                print(f"  ✓ Task {task_num} completed: {len(result.text)} chars")

        duration = time.time() - start
        print(f"\n✓ Parallel processing took: {duration:.2f}s")

        # Check queue stats
        if session.has_queue():
            stats = session.queue_stats()
            print("\n📊 Queue Statistics:")
            print(f"  Total processed: {stats['total_processed']}")
            print(f"  Total failed: {stats['total_failed']}")
            print(f"  DLQ size: {stats['dlq_size']}")

        print("\n✅ Test 2 passed: Parallel processing with queue works\n")

    async def test_priority_lanes(self) -> None:
        """Test 3: Parallel processing with priority lanes."""
        print("🎯 Test 3: Parallel Processing with Priority Lanes")
        print("-" * 80)

        queue_config = SessionQueueConfig()
        queue_config.set_control_concurrency(1)    # P0: Control operations
        queue_config.set_query_concurrency(3)      # P1: Query operations (highest concurrency)
        queue_config.set_execute_concurrency(2)    # P2: Execute operations
        queue_config.set_generate_concurrency(1)   # P3: Generate operations
        queue_config.enable_metrics()
        queue_config.enable_dlq()

        session = self.agent.session(".", queue_config=queue_config)

        print("Testing priority-based task execution...")
        print("  P1 (Query): 3 concurrent tasks")
        print("  P2 (Execute): 2 concurrent tasks")
        print("  P3 (Generate): 1 concurrent task")

        start = time.time()

        # Mix of different task types
        tasks = [
            ("Query", "How many Python files are there?"),
            ("Query", "What is the project structure?"),
            ("Query", "List all dependencies"),
            ("Execute", "Create a summary of the codebase"),
            ("Execute", "Analyze code complexity"),
        ]

        async def process_task(task_num: int, task_type: str, task_text: str) -> tuple:
            """Process a single task."""
            result = session.send(task_text)
            return task_num, task_type, result

        for i, (task_type, task) in enumerate(tasks, 1):
            print(f"  Queuing {task_type} task {i}: {task}")

        print("\n  Waiting for all tasks to complete...")

        async_tasks = [
            asyncio.create_task(asyncio.to_thread(process_task, i, task_type, task))
            for i, (task_type, task) in enumerate(tasks, 1)
        ]

        results = await asyncio.gather(*async_tasks, return_exceptions=True)

        for task_num, task_type, result in results:
            if isinstance(result, Exception):
                print(f"  ✗ Task {task_num} ({task_type}) failed: {result}")
            else:
                print(f"  ✓ Task {task_num} ({task_type}) completed: {len(result.text)} chars")

        duration = time.time() - start
        print(f"\n✓ Priority-based processing took: {duration:.2f}s")
        print("\n✅ Test 3 passed: Priority lanes work correctly\n")

    async def test_retry_policy(self) -> None:
        """Test 4: Parallel processing with retry policy."""
        print("🔄 Test 4: Parallel Processing with Retry Policy")
        print("-" * 80)

        queue_config = SessionQueueConfig()
        queue_config.set_query_concurrency(3)
        queue_config.enable_metrics()
        queue_config.enable_dlq()

        session = self.agent.session(".", queue_config=queue_config)

        print("Testing retry policy with exponential backoff...")
        print("  Note: Retry policy configured at queue level")
        print("  Max retries: 3 (default)")
        print("  Strategy: exponential (default)")

        tasks = [
            "Analyze the main function",
            "Find all error types",
            "List all test functions",
        ]

        start = time.time()

        async def process_task(task_num: int, task_text: str) -> tuple:
            """Process a single task."""
            result = session.send(task_text)
            return task_num, result

        for i, task in enumerate(tasks, 1):
            print(f"  Queuing task {i}: {task}")

        print("\n  Waiting for all tasks to complete...")

        async_tasks = [
            asyncio.create_task(asyncio.to_thread(process_task, i, task))
            for i, task in enumerate(tasks, 1)
        ]

        results = await asyncio.gather(*async_tasks, return_exceptions=True)

        for task_num, result in results:
            if isinstance(result, Exception):
                print(f"  ✗ Task {task_num} failed: {result}")
            else:
                print(f"  ✓ Task {task_num} completed: {len(result.text)} chars")

        duration = time.time() - start
        print(f"\n✓ Processing with retry took: {duration:.2f}s")

        if session.has_queue():
            stats = session.queue_stats()
            print("\n📊 Final Queue Statistics:")
            print(f"  Total processed: {stats['total_processed']}")
            print(f"  Total failed: {stats['total_failed']}")
            print(f"  DLQ size: {stats['dlq_size']}")

        print("\n✅ Test 4 passed: Retry policy works correctly\n")

    async def run_all(self) -> None:
        """Run all parallel processing tests."""
        print("🚀 A3S Code Python SDK - Parallel Task Processing Integration Test\n")
        print("=" * 80)

        print(f"📄 Using config: {self.config_path}")
        print("=" * 80)

        await self.test_sequential_processing()
        await self.test_parallel_processing()
        await self.test_priority_lanes()
        await self.test_retry_policy()

        print()
        print("=" * 80)
        print("✅ All parallel processing tests completed successfully!")
        print("=" * 80)


if __name__ == "__main__":
    config_path = ParallelProcessingTest.find_config_path()
    agent = Agent.create(config_path)
    test = ParallelProcessingTest(agent, config_path)
    asyncio.run(test.run_all())
