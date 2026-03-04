#!/usr/bin/env python3
"""
A3S Code Python SDK - External Task Handler Integration Test

Demonstrates the Multi-Machine External Task pattern:
1. Coordinator creates a session with Execute lane set to External mode
2. Agent streams a prompt that triggers bash/write/edit tool calls
3. ExternalTaskPending events fire — coordinator polls pending tasks
4. Coordinator processes tasks locally (simulating a remote worker)
5. Coordinator completes tasks via complete_external_task()

Run with: python examples/test_external_task_handler.py
"""

import subprocess
import time
import threading
from pathlib import Path
from a3s_code import Agent, SessionQueueConfig


class ExternalTaskHandlerTest:
    """Integration test for the external task handler pattern."""

    def __init__(self, agent: Agent, config_path: str) -> None:
        self.agent = agent
        self.config_path = config_path
        self._session = None

    @staticmethod
    def find_config_path() -> str:
        """Find config file."""
        config_path = Path.home() / ".a3s" / "config.hcl"
        if config_path.exists():
            return str(config_path)

        project_config = Path(__file__).parent.parent.parent.parent.parent.parent / ".a3s" / "config.hcl"
        if project_config.exists():
            return str(project_config)

        raise FileNotFoundError("Config file not found")

    @staticmethod
    def worker_execute_bash(command: str, working_dir: str = ".") -> dict:
        """Simulate a remote worker executing a bash command."""
        try:
            result = subprocess.run(
                ["sh", "-c", command],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "exit_code": result.returncode,
                "error": result.stderr if result.returncode != 0 else None,
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "exit_code": 1,
                "error": str(e),
            }

    def test_external_task_handler(self) -> None:
        """Test 1: External mode — Execute lane tasks routed to external handler."""
        print("\n📦 Test 1: External Task Handler (Execute Lane → External)")
        print("-" * 80)

        # 1. Create session with queue enabled
        qc = SessionQueueConfig()
        qc.with_lane_features()
        qc.set_timeout(60000)
        self._session = self.agent.session(".", queue_config=qc)

        # 2. Route Execute lane to External mode
        self._session.set_lane_handler("execute", mode="external", timeout_ms=60000)

        print("✓ Session created with Execute lane → External mode")
        print("  Query lane: Internal (read, glob, grep run locally)")
        print("  Execute lane: External (bash, write, edit → ExternalTask)")
        print()

        # 3. Stream a prompt that will trigger Execute-lane tools (bash)
        start = time.time()
        text_output = ""

        # Start a background thread to poll and process external tasks
        stop_polling = threading.Event()
        poll_results = {"count": 0}

        def poll_external_tasks() -> None:
            """Background poller that checks for pending external tasks."""
            while not stop_polling.is_set():
                try:
                    tasks = self._session.pending_external_tasks()
                    for task in tasks:
                        task_id = task["task_id"]
                        cmd_type = task["command_type"]
                        print(f"  📥 ExternalTaskPending: {task_id[:8]} ({cmd_type})")

                        # Execute the task (simulating a remote worker)
                        if cmd_type == "bash":
                            cmd = task.get("payload", {}).get("command", "echo 'no command'")
                            cwd = task.get("payload", {}).get("working_dir", ".")
                            result = self.worker_execute_bash(cmd, cwd)
                        else:
                            result = {
                                "success": True,
                                "output": f"External handler processed: {cmd_type}",
                                "exit_code": 0,
                                "error": None,
                            }

                        print(f"  🔧 Worker result: success={result['success']}, exit_code={result['exit_code']}")
                        output_preview = (result["output"] or "").strip()[:60]
                        if output_preview:
                            print(f"     Output: {output_preview}")

                        # Complete the external task
                        completed = self._session.complete_external_task(
                            task_id,
                            success=result["success"],
                            result={"output": result["output"], "exit_code": result["exit_code"]},
                            error=result["error"],
                        )

                        if completed:
                            poll_results["count"] += 1
                            print(f"  📤 Task {task_id[:8]} completed and returned to agent")
                except Exception as e:
                    # Session might be done
                    if "exhausted" not in str(e).lower():
                        print(f"  ⚠ Poll error: {e}")

                time.sleep(0.2)  # Poll every 200ms

        # Start the poller
        poller = threading.Thread(target=poll_external_tasks, daemon=True)
        poller.start()

        # 4. Stream the prompt — events arrive as the agent works
        for event in self._session.stream(
            "Run these bash commands and tell me the results:\n"
            "1. echo 'Hello from external worker'\n"
            "2. date '+%Y-%m-%d %H:%M:%S'\n"
            "3. uname -s"
        ):
            if event.event_type == "text_delta":
                text_output += event.text or ""
                print(event.text or "", end="", flush=True)
            elif event.event_type == "tool_start":
                print(f"\n  🔨 Tool: {event.tool_name}")
            elif event.event_type == "end":
                print()
                break
            elif event.event_type == "error":
                print(f"\n  ❌ Error: {event.error}")
                break

        # Stop the poller
        stop_polling.set()
        poller.join(timeout=2)

        duration = time.time() - start
        external_tasks_processed = poll_results["count"]

        print()
        print("-" * 80)
        print("📊 Results:")
        print(f"  Duration: {duration:.2f}s")
        print(f"  External tasks processed: {external_tasks_processed}")
        print(f"  Response length: {len(text_output)} chars")

    def test_hybrid_mode(self) -> None:
        """Test 2: Hybrid mode — local execution + external notification."""
        print("\n\n📦 Test 2: Hybrid Mode (Execute Lane → Hybrid)")
        print("-" * 80)

        # Switch to Hybrid mode
        self._session.set_lane_handler("execute", mode="hybrid", timeout_ms=60000)

        print("✓ Execute lane switched to Hybrid mode")
        print("  Tools execute locally AND emit ExternalTaskPending events")
        print()

        start = time.time()
        result = self._session.send("Run: echo 'hybrid mode test'")
        duration = time.time() - start

        print(f"✓ Completed in {duration:.2f}s")
        print(f"  Response: {result.text[:120]}")
        print(f"  Tool calls: {result.tool_calls_count}")

    def test_dynamic_lane_switching(self) -> None:
        """Test 3: Dynamic lane switching."""
        print("\n\n📦 Test 3: Dynamic Lane Switching")
        print("-" * 80)

        # Switch back to Internal
        self._session.set_lane_handler("execute", mode="internal", timeout_ms=60000)

        print("✓ Execute lane switched back to Internal mode")

        start = time.time()
        result = self._session.send("Run: echo 'back to internal mode'")
        duration = time.time() - start

        print(f"✓ Completed in {duration:.2f}s")
        print(f"  Response: {result.text[:120]}")

    def run_all(self) -> None:
        """Run all external task handler tests."""
        print("🚀 A3S Code - External Task Handler Integration Test\n")
        print("=" * 80)
        print(f"📄 Config: {self.config_path}")
        print("=" * 80)

        self.test_external_task_handler()
        self.test_hybrid_mode()
        self.test_dynamic_lane_switching()

        print("\n" + "=" * 80)
        print("✅ All external task handler tests completed!")
        print("=" * 80)


if __name__ == "__main__":
    config_path = ExternalTaskHandlerTest.find_config_path()
    agent = Agent.create(config_path)
    test = ExternalTaskHandlerTest(agent, config_path)
    test.run_all()
