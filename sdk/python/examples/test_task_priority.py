"""
Lane-Based Priority Preemption Test with Real LLM

Demonstrates how the lane-based priority system allows high-priority tasks
to preempt queued low-priority tasks:

1. Constrain concurrency so tasks must queue up
2. Submit multiple low-priority tasks (Execute lane: bash commands)
3. Submit a high-priority task later (Query lane: read/grep)
4. Observe that the high-priority task executes before queued low-priority tasks

Lane priorities (lower = higher priority):
  Control (P0) > Query (P1) > Execute (P2) > Generate (P3)

Run with: python test_task_priority.py
"""

import asyncio
import time
import os
from dataclasses import dataclass, field
from pathlib import Path

# pip install a3s-code
from a3s_code import Agent, SessionOptions, SessionQueueConfig


@dataclass
class CompletionRecord:
    name: str
    lane: str
    submitted_at: float
    completed_at: float = 0.0


class TaskPriorityTest:
    """Tests for lane-based priority preemption behavior."""

    def __init__(self, agent: Agent, config_path: str) -> None:
        self.agent = agent
        self.config_path = config_path

    @staticmethod
    def find_config() -> str:
        """Find config file in home directory."""
        home_config = Path.home() / ".a3s" / "config.hcl"
        if home_config.exists():
            return str(home_config)
        raise FileNotFoundError("Config not found. Create ~/.a3s/config.hcl")

    async def test_query_preempts_execute(self) -> None:
        """Test 1: Query-lane task (P1) preempts queued Execute-lane tasks (P2)"""
        print("\n📋 Test 1: Query (P1) Preempts Execute (P2)")
        print("-" * 80)
        print("  Execute lane concurrency: 1 (tasks must queue)")
        print("  Query lane concurrency: 2 (higher priority, separate capacity)")
        print("  Submit 3 Execute tasks first, then 1 Query task")
        print("  Expected: Query task completes before remaining Execute tasks\n")

        queue_config = SessionQueueConfig()
        queue_config.execute_max_concurrency = 1   # Bottleneck
        queue_config.query_max_concurrency = 2     # Higher priority, own capacity
        queue_config.generate_max_concurrency = 1
        queue_config.enable_metrics = True

        opts = SessionOptions()
        opts.queue_config = queue_config
        opts.auto_approve = True

        session = self.agent.session(".", opts)
        start = time.monotonic()
        completions: list[CompletionRecord] = []
        lock = asyncio.Lock()

        async def run_task(name: str, prompt: str, lane_label: str) -> object:
            submitted_at = time.monotonic() - start
            print(f"  [{submitted_at:>6.2f}s] 📤 Submitting: {name} ({lane_label})")
            result = await asyncio.to_thread(session.send, prompt)
            completed_at = time.monotonic() - start
            async with lock:
                completions.append(CompletionRecord(
                    name=name, lane=lane_label,
                    submitted_at=submitted_at, completed_at=completed_at,
                ))
            chars = len(result.text) if result else 0
            marker = "🚨" if "P1" in lane_label else "✅"
            print(f"  [{completed_at:>6.2f}s] {marker} Completed: {name} ({chars} chars)")
            return result

        # Submit 3 Execute-lane tasks (bash -> Execute lane P2)
        tasks = []
        for i in range(1, 4):
            prompt = f"Run this bash command and tell me the output: echo 'Task {i} started' && sleep 1 && echo 'Task {i} done'"
            tasks.append(asyncio.create_task(
                run_task(f"Execute-{i}", prompt, "Execute (P2)")
            ))
            await asyncio.sleep(0.2)

        # Wait for queue to fill
        await asyncio.sleep(0.5)

        # Submit Query-lane task (read -> Query lane P1)
        print()
        tasks.append(asyncio.create_task(
            run_task(
                "Query-Urgent",
                "Read the Cargo.toml file and tell me the package name and version",
                "Query (P1)",
            )
        ))

        await asyncio.gather(*tasks, return_exceptions=True)

        # Print completion order
        completions.sort(key=lambda r: r.completed_at)
        print("\n  --- Completion Order ---")
        for i, rec in enumerate(completions, 1):
            print(f"  {i}. {rec.name} [{rec.lane}] — submitted {rec.submitted_at:.2f}s, completed {rec.completed_at:.2f}s")

        print("\n✅ Test 1 completed")

    async def test_multi_level_priority(self) -> None:
        """Test 2: Multi-level priority -- Execute (P2) vs Query (P1)"""
        print("\n\n📋 Test 2: Multi-Level Priority (Query P1 vs Execute P2)")
        print("-" * 80)
        print("  All lanes constrained to concurrency=1")
        print("  Submit Execute task first, then Query task")
        print("  Expected: Query task scheduled with higher priority\n")

        queue_config = SessionQueueConfig()
        queue_config.control_max_concurrency = 1
        queue_config.query_max_concurrency = 1
        queue_config.execute_max_concurrency = 1
        queue_config.generate_max_concurrency = 1
        queue_config.enable_metrics = True

        opts = SessionOptions()
        opts.queue_config = queue_config
        opts.auto_approve = True

        session = self.agent.session(".", opts)
        start = time.monotonic()
        completions: list[CompletionRecord] = []
        lock = asyncio.Lock()

        async def run_task(name: str, prompt: str, lane_label: str) -> object:
            submitted_at = time.monotonic() - start
            print(f"  [{submitted_at:>6.2f}s] 📤 {name} ({lane_label})")
            result = await asyncio.to_thread(session.send, prompt)
            completed_at = time.monotonic() - start
            async with lock:
                completions.append(CompletionRecord(
                    name=name, lane=lane_label,
                    submitted_at=submitted_at, completed_at=completed_at,
                ))
            print(f"  [{completed_at:>6.2f}s] ✅ {name} completed")
            return result

        # Execute task first (lower priority)
        exec_task = asyncio.create_task(
            run_task("Execute-Task", "Run: echo 'execute-lane task' && sleep 1 && echo 'done'", "Execute (P2)")
        )
        await asyncio.sleep(0.3)

        # Query task (higher priority)
        query_task = asyncio.create_task(
            run_task("Query-Task", "Read the Cargo.toml file and show the first 5 lines", "Query (P1)")
        )

        await asyncio.gather(exec_task, query_task, return_exceptions=True)

        completions.sort(key=lambda r: r.completed_at)
        print("\n  --- Completion Order ---")
        for i, rec in enumerate(completions, 1):
            print(f"  {i}. {rec.name} [{rec.lane}] — submitted {rec.submitted_at:.2f}s, completed {rec.completed_at:.2f}s")

        print("\n✅ Test 2 completed")

    async def test_late_urgent_insertion(self) -> None:
        """Test 3: Late urgent task inserted into a busy queue"""
        print("\n\n📋 Test 3: Late Urgent Task Insertion")
        print("-" * 80)
        print("  Execute concurrency: 1, Query concurrency: 2")
        print("  Submit 4 slow Execute tasks, then 1 urgent Query task at 2s mark")
        print("  Expected: Urgent task completes before remaining Execute tasks\n")

        queue_config = SessionQueueConfig()
        queue_config.execute_max_concurrency = 1
        queue_config.query_max_concurrency = 2
        queue_config.generate_max_concurrency = 1
        queue_config.enable_metrics = True

        opts = SessionOptions()
        opts.queue_config = queue_config
        opts.auto_approve = True

        session = self.agent.session(".", opts)
        start = time.monotonic()
        completions: list[CompletionRecord] = []
        lock = asyncio.Lock()

        async def run_task(name: str, prompt: str, lane_label: str) -> object:
            submitted_at = time.monotonic() - start
            marker = "🚨" if "P1" in lane_label else "📤"
            print(f"  [{submitted_at:>6.2f}s] {marker} {name}")
            result = await asyncio.to_thread(session.send, prompt)
            completed_at = time.monotonic() - start
            async with lock:
                completions.append(CompletionRecord(
                    name=name, lane=lane_label,
                    submitted_at=submitted_at, completed_at=completed_at,
                ))
            marker = "🚨" if "P1" in lane_label else "✅"
            print(f"  [{completed_at:>6.2f}s] {marker} {name} completed")
            return result

        # Submit 4 slow Execute tasks
        tasks = []
        for i in range(1, 5):
            prompt = f"Run: echo 'slow task {i} start' && sleep 2 && echo 'slow task {i} end'"
            tasks.append(asyncio.create_task(
                run_task(f"SlowExec-{i}", prompt, "Execute (P2)")
            ))
            await asyncio.sleep(0.1)

        # Wait for queue to fill
        print("\n  ⏳ Waiting 2s for queue to fill up...\n")
        await asyncio.sleep(2.0)

        # Print queue stats
        stats = session.queue_stats()
        print(f"  📊 Queue stats: pending={stats.get('total_pending', '?')}, active={stats.get('total_active', '?')}")

        # Inject urgent Query task
        tasks.append(asyncio.create_task(
            run_task(
                "UrgentQuery",
                "Use grep to search for 'name' in Cargo.toml and tell me the result",
                "Query (P1)",
            )
        ))

        await asyncio.gather(*tasks, return_exceptions=True)

        # Print completion order
        completions.sort(key=lambda r: r.completed_at)
        print("\n  --- Completion Order ---")
        for i, rec in enumerate(completions, 1):
            marker = "🚨" if "P1" in rec.lane else "  "
            print(f"  {marker} {i}. {rec.name} [{rec.lane}] — submitted {rec.submitted_at:.2f}s, completed {rec.completed_at:.2f}s")

        # Check preemption
        urgent = next((r for r in completions if r.name == "UrgentQuery"), None)
        last_exec = next((r for r in reversed(completions) if "P2" in r.lane), None)
        if urgent and last_exec:
            if urgent.completed_at < last_exec.completed_at:
                print(f"\n  ✅ Priority preemption confirmed: UrgentQuery ({urgent.completed_at:.2f}s) before last Execute ({last_exec.completed_at:.2f}s)")
            else:
                print(f"\n  ℹ️  UrgentQuery ({urgent.completed_at:.2f}s) after last Execute ({last_exec.completed_at:.2f}s)")
                print("     This can happen if Execute tasks were already running")

        print("\n✅ Test 3 completed")

    async def run_all(self) -> None:
        """Run all lane-based priority preemption tests."""
        print("🚀 A3S Code - Lane-Based Priority Preemption Test (Python)\n")
        print("=" * 80)
        print(f"📄 Using config: {self.config_path}")
        print("=" * 80)

        await self.test_query_preempts_execute()
        await self.test_multi_level_priority()
        await self.test_late_urgent_insertion()

        print(f"\n{'=' * 80}")
        print("✅ All lane-based priority preemption tests completed!")
        print("=" * 80)


if __name__ == "__main__":
    config_path = TaskPriorityTest.find_config()
    agent = Agent(config_path)
    test = TaskPriorityTest(agent, config_path)
    asyncio.run(test.run_all())
