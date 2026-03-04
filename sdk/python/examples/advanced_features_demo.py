#!/usr/bin/env python3
"""
A3S Code — Model Switching & Direct Tools Demo

Demonstrates:
1. Switching models per session (different providers)
2. Direct tool execution (bypass LLM)
3. Hooks registration and lifecycle events
4. Queue/lane configuration
5. Auto-compaction and resilience settings
6. Security provider

Usage:
    cd crates/code/sdk/python
    python examples/advanced_features_demo.py
"""

import os
import tempfile
from pathlib import Path
from a3s_code import Agent, SessionOptions, SessionQueueConfig, builtin_skills


class AdvancedFeaturesDemo:
    """Demonstrates advanced features of the A3S Code Python SDK."""

    def __init__(self, agent: Agent, config_path: str) -> None:
        self.agent = agent
        self.config_path = config_path

    @staticmethod
    def find_config_path() -> str:
        if env := os.environ.get("A3S_CONFIG"):
            return env
        home_config = Path.home() / ".a3s" / "config.hcl"
        if home_config.exists():
            return str(home_config)
        project_config = (
            Path(__file__).parent.parent.parent.parent.parent.parent
            / ".a3s" / "config.hcl"
        )
        if project_config.exists():
            return str(project_config)
        raise FileNotFoundError("Config not found. Create ~/.a3s/config.hcl or set A3S_CONFIG")

    @staticmethod
    def separator(title: str) -> None:
        print(f"\n{'=' * 72}")
        print(f"  {title}")
        print(f"{'=' * 72}\n")

    @staticmethod
    def truncate(text: str, max_len: int = 200) -> str:
        text = text.strip()
        return text if len(text) <= max_len else f"{text[:max_len]}..."

    # ========================================================================
    # Demo 1: Direct Tool Execution
    # ========================================================================

    def demo_direct_tools(self) -> None:
        self.separator("Demo 1: Direct Tool Execution (No LLM)")

        with tempfile.TemporaryDirectory() as workspace:
            session = self.agent.session(workspace, permissive=True)
            print(f"  Workspace: {workspace}\n")

            # Write a file
            print("  1. Write a file")
            r = session.tool("write", {"path": "hello.txt", "content": "Hello from Python!\nLine 2."})
            print(f"     exit={r.exit_code} output={self.truncate(r.output, 60)}")

            # Read it back
            print("  2. Read the file")
            content = session.read_file("hello.txt")
            print(f"     content: {self.truncate(content, 60)}")

            # Glob
            print("  3. Glob for *.txt")
            files = session.glob("*.txt")
            print(f"     matches: {files}")

            # Bash
            print("  4. Run bash command")
            output = session.bash("wc -l hello.txt")
            print(f"     output: {output.strip()}")

            # Grep
            print("  5. Grep for 'Python'")
            grep_out = session.grep("Python")
            print(f"     output: {self.truncate(grep_out, 60)}")

            # Edit
            print("  6. Edit the file")
            r = session.tool("edit", {
                "path": "hello.txt",
                "old_string": "Line 2.",
                "new_string": "Line 2 -- edited from Python!"
            })
            print(f"     exit={r.exit_code}")

            # Verify
            final_content = session.read_file("hello.txt")
            print(f"\n  Final content: {self.truncate(final_content, 80)}")

    # ========================================================================
    # Demo 2: Hooks
    # ========================================================================

    def demo_hooks(self) -> None:
        self.separator("Demo 2: Hooks (Lifecycle Event Interception)")

        with tempfile.TemporaryDirectory() as workspace:
            session = self.agent.session(workspace, permissive=True)

            # Register hooks
            session.register_hook("audit_tools", "pre_tool_use", config={"priority": 10})
            session.register_hook("audit_bash", "post_tool_use",
                                  matcher={"tool": "bash"}, config={"priority": 20})
            session.register_hook("log_gen", "generate_start")
            session.register_hook("log_err", "on_error")

            print(f"  Registered {session.hook_count()} hooks")
            print("  * audit_tools  (PreToolUse, priority=10)")
            print("  * audit_bash   (PostToolUse, bash only)")
            print("  * log_gen      (GenerateStart)")
            print("  * log_err      (OnError)\n")

            # Execute with hooks active
            for event in session.stream(
                "Create a file test.sh with `echo hello` and run it with bash."
            ):
                if event.event_type == "tool_start":
                    print(f"  [hook] PreToolUse -> {event.tool_name}")
                elif event.event_type == "tool_end":
                    if event.tool_name == "bash":
                        print(f"  [hook] PostToolUse(bash) -> exit={event.exit_code}")
                elif event.event_type == "end":
                    print(f"\n  Done ({event.total_tokens} tokens)")
                    break
                elif event.event_type == "error":
                    print(f"  Error: {event.error}")
                    break

            # Unregister
            removed = session.unregister_hook("audit_tools")
            print(f"\n  Unregistered audit_tools: {removed}")
            print(f"  Remaining hooks: {session.hook_count()}")

    # ========================================================================
    # Demo 3: Queue & Lanes
    # ========================================================================

    def demo_queue_lanes(self) -> None:
        self.separator("Demo 3: Queue & Lanes (Priority-Based Scheduling)")

        with tempfile.TemporaryDirectory() as workspace:
            # Create test files
            for i in range(1, 4):
                Path(workspace, f"file{i}.txt").write_text(f"Content of file {i}\n")

            qc = SessionQueueConfig()
            qc.with_lane_features()
            qc.set_query_concurrency(8)
            qc.set_execute_concurrency(2)

            session = self.agent.session(workspace, permissive=True, queue_config=qc)

            print(f"  Queue active: {session.has_queue()}")
            print(f"  Workspace: {workspace}\n")

            result = session.send(
                "Read all .txt files and create a combined.md with their contents."
            )

            print(f"  Tool calls: {result.tool_calls_count}")
            print(f"  Tokens:     {result.total_tokens}")

            # Queue stats
            stats = session.queue_stats()
            print(f"\n  Queue stats: {stats}")

            # Dead letters
            dlq = session.dead_letters()
            print(f"  Dead letters: {len(dlq)}")

    # ========================================================================
    # Demo 4: Security Provider
    # ========================================================================

    def demo_security(self) -> None:
        self.separator("Demo 4: Security Provider (Taint Tracking)")

        with tempfile.TemporaryDirectory() as workspace:
            # Create a file with sensitive data
            Path(workspace, "secrets.env").write_text(
                'DATABASE_URL=postgres://admin:p4ssw0rd@db.example.com/prod\n'
                'API_KEY=sk-abc123def456\n'
                'AWS_SECRET=AKIAIOSFODNN7EXAMPLE\n'
            )

            opts = SessionOptions()
            opts.default_security = True
            session = self.agent.session(workspace, options=opts, permissive=True)

            print("  Security: DefaultSecurityProvider enabled")
            print("  Features: taint tracking + output sanitization\n")

            result = session.send(
                "Read secrets.env and tell me what types of secrets are in it. "
                "Do NOT include the actual secret values in your response."
            )

            print(f"  Tool calls: {result.tool_calls_count}")
            print(f"  Tokens:     {result.total_tokens}")
            print(f"  Response:   {self.truncate(result.text, 300)}")

    # ========================================================================
    # Demo 5: Auto-Compaction & Resilience
    # ========================================================================

    def demo_resilience(self) -> None:
        self.separator("Demo 5: Auto-Compaction & Resilience")

        with tempfile.TemporaryDirectory() as workspace:
            opts = SessionOptions()
            opts.auto_compact = True
            opts.auto_compact_threshold = 0.7

            session = self.agent.session(
                workspace,
                options=opts,
                permissive=True,
                max_parse_retries=3,
                tool_timeout_ms=30_000,
                circuit_breaker_threshold=5,
            )

            print("  Auto-compact:    enabled (70% threshold)")
            print("  Parse retries:   3")
            print("  Tool timeout:    30s")
            print("  Circuit breaker: 5\n")

            result = session.send(
                "Create a CSV file with 15 rows of sample data (id, name, score), "
                "then create a report.md analyzing the data."
            )

            print(f"  Tool calls: {result.tool_calls_count}")
            print(f"  Tokens:     {result.total_tokens}")

            for name in ["data.csv", "report.md"]:
                p = Path(workspace) / name
                if p.exists():
                    print(f"  {name} ({p.stat().st_size} bytes)")
                else:
                    print(f"  {name} not found")

    # ========================================================================
    # Demo 6: Memory (Persistent)
    # ========================================================================

    def demo_memory(self) -> None:
        self.separator("Demo 6: Persistent Memory")

        with tempfile.TemporaryDirectory() as workspace:
            with tempfile.TemporaryDirectory() as memory_dir:
                opts = SessionOptions()
                opts.memory_dir = memory_dir

                session = self.agent.session(workspace, options=opts, permissive=True)

                print(f"  Memory dir: {memory_dir}\n")

                # Turn 1: teach the agent something
                print("  [Turn 1] Teaching the agent...")
                r1 = session.send(
                    "Remember this: The project deadline is March 15, 2026. "
                    "The tech lead is Alice and the PM is Bob."
                )
                print(f"    Tokens: {r1.total_tokens}")

                # Turn 2: ask about it (memory recall)
                print("  [Turn 2] Asking about remembered info...")
                r2 = session.send("Who is the tech lead and when is the deadline?")
                print(f"    Tokens: {r2.total_tokens}")
                print(f"    Answer: {self.truncate(r2.text, 150)}")

    # ========================================================================
    # Run All Demos
    # ========================================================================

    def run_all(self) -> None:
        print("+" + "=" * 70 + "+")
        print("|     A3S Code Python SDK -- Advanced Features Demo (Real LLM)        |")
        print("+" + "=" * 70 + "+")

        print(f"\n  Config: {self.config_path}")
        print("  Agent:  created\n")

        self.demo_direct_tools()
        self.demo_hooks()
        self.demo_queue_lanes()
        self.demo_security()
        self.demo_resilience()
        self.demo_memory()

        self.separator("All Demos Complete")


if __name__ == "__main__":
    config_path = AdvancedFeaturesDemo.find_config_path()
    agent = Agent.create(config_path)
    demo = AdvancedFeaturesDemo(agent, config_path)
    demo.run_all()
