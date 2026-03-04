#!/usr/bin/env python3
"""
A3S Code — Agentic Loop Demo (Real LLM)

Demonstrates the AgentLoop's autonomous problem-solving capabilities
using real LLM configuration. The agent reads files, writes code,
runs commands, and iterates until the task is complete.

## What This Example Shows

1. Autonomous File Operations — LLM reads, writes, and edits files
2. Streaming Events — Real-time visibility into the agent's actions
3. Planning Mode — Task decomposition with goal tracking
4. Multi-Turn Conversation — Context preserved across turns
5. Skills-Augmented Code Review — Built-in skills for better output
6. Resilient Session — Parse retries, tool timeout, circuit breaker

## Usage

    cd crates/code/sdk/python
    python examples/agentic_loop_demo.py

    # With custom config
    A3S_CONFIG=/path/to/config.hcl python examples/agentic_loop_demo.py

Requires a valid LLM API key in ~/.a3s/config.hcl or $A3S_CONFIG.
"""

import os
import tempfile
from pathlib import Path
from a3s_code import Agent, builtin_skills


class AgenticLoopDemo:
    """Demonstrates the agentic loop capabilities of the A3S Code Python SDK."""

    def __init__(self, agent: Agent, config_path: str) -> None:
        self.agent = agent
        self.config_path = config_path

    @staticmethod
    def find_config_path() -> str:
        """Find config file from env, home dir, or project root."""
        if env := os.environ.get("A3S_CONFIG"):
            return env

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

        raise FileNotFoundError(
            "Config not found. Create ~/.a3s/config.hcl or set A3S_CONFIG"
        )

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
    # Demo 1: Autonomous Multi-Step Coding
    # ========================================================================

    def demo_1_autonomous_coding(self) -> None:
        """Demo 1: Autonomous multi-step coding task.

        The LLM autonomously:
        1. Creates a Rust source file
        2. Reads it back to verify
        3. Edits it to add functionality
        4. Reads the final version to confirm
        """
        self.separator("Demo 1: Autonomous Multi-Step Coding")

        with tempfile.TemporaryDirectory() as workspace:
            session = self.agent.session(workspace, permissive=True)

            print(f"  Workspace: {workspace}")
            print("  Prompt:    Create a Rust file, then improve it\n")

            result = session.send(
                "Do the following steps:\n"
                "1. Create a file called `calculator.rs` with a basic Calculator struct "
                "that has add() and subtract() methods\n"
                "2. Read the file back to verify it was created correctly\n"
                "3. Edit the file to add multiply() and divide() methods "
                "(divide should handle division by zero)\n"
                "4. Read the final version and confirm all 4 methods exist"
            )

            print(f"  Tool calls: {result.tool_calls_count}")
            print(f"  Tokens:     {result.total_tokens} total")
            print(f"  Response:   {self.truncate(result.text)}")

            # Verify the file was actually created
            calc_path = Path(workspace) / "calculator.rs"
            if calc_path.exists():
                content = calc_path.read_text()
                has_add = "add" in content
                has_sub = "subtract" in content or "sub" in content
                has_mul = "multiply" in content or "mul" in content
                has_div = "divide" in content or "div" in content
                print(
                    f"\n  File verified: add={has_add} sub={has_sub} "
                    f"mul={has_mul} div={has_div}"
                )
            else:
                print("\n  calculator.rs not found (LLM may have used a different name)")

    # ========================================================================
    # Demo 2: Streaming Events
    # ========================================================================

    def demo_2_streaming_events(self) -> None:
        """Demo 2: Streaming -- watch the agent think and act in real time."""
        self.separator("Demo 2: Streaming Events (Real-Time Agent Visibility)")

        with tempfile.TemporaryDirectory() as workspace:
            # Pre-create a file for the agent to work with
            data_path = Path(workspace) / "data.json"
            data_path.write_text(
                '{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}'
            )

            session = self.agent.session(workspace, permissive=True)

            print(f"  Workspace: {workspace}")
            print("  Prompt:    Read data.json and create a summary\n")
            print("  --- Event Stream ---")

            tool_count = 0
            text_len = 0

            for event in session.stream(
                "Read the file data.json in the workspace, then create a file called "
                "summary.txt that lists each user's name and age in a human-readable format."
            ):
                if event.event_type == "agent_start":
                    print("  > Agent started")
                elif event.event_type == "turn_start":
                    print(f"  +-- Turn {event.turn}")
                elif event.event_type == "tool_start":
                    tool_count += 1
                    print(f"  |  Tool: {event.tool_name}...", end="", flush=True)
                elif event.event_type == "tool_end":
                    status = "ok" if event.exit_code == 0 else "fail"
                    print(f" {status} (exit={event.exit_code})")
                elif event.event_type == "text_delta":
                    text_len += len(event.text)
                elif event.event_type == "turn_end":
                    print(f"  +-- Turn {event.turn} done ({event.total_tokens} tokens)")
                elif event.event_type == "end":
                    print(f"\n  Agent finished")
                    print(
                        f"    Tools: {tool_count}, Response: {text_len} chars, "
                        f"Tokens: {event.total_tokens}"
                    )
                    break
                elif event.event_type == "error":
                    print(f"\n  Error: {event.error}")
                    break

            # Verify output
            summary_path = Path(workspace) / "summary.txt"
            if summary_path.exists():
                summary = summary_path.read_text()
                print(f"\n  summary.txt created ({len(summary)} bytes)")

    # ========================================================================
    # Demo 3: Planning Mode
    # ========================================================================

    def demo_3_planning_mode(self) -> None:
        """Demo 3: Planning mode -- decompose a complex task into steps."""
        self.separator("Demo 3: Planning Mode (Task Decomposition)")

        with tempfile.TemporaryDirectory() as workspace:
            session = self.agent.session(workspace, planning=True, goal_tracking=True, permissive=True)

            print(f"  Workspace: {workspace}")
            print("  Planning:  enabled")
            print("  Goal tracking: enabled\n")

            result = session.send(
                "Create a small project with these files:\n"
                "1. `lib.rs` -- a module with a `greet(name: &str) -> String` function\n"
                "2. `main.rs` -- imports from lib and calls greet(\"World\")\n"
                "3. `README.md` -- brief documentation explaining the project\n"
                "Then read all three files to verify they are correct."
            )

            print(f"  Tool calls: {result.tool_calls_count}")
            print(f"  Tokens:     {result.total_tokens} total")
            print(f"  Response:   {self.truncate(result.text)}")

            # Verify files
            for f in ["lib.rs", "main.rs", "README.md"]:
                path = Path(workspace) / f
                if path.exists():
                    size = path.stat().st_size
                    print(f"  {f} ({size} bytes)")
                else:
                    print(f"  {f} not found")

    # ========================================================================
    # Demo 4: Multi-Turn Conversation
    # ========================================================================

    def demo_4_multi_turn(self) -> None:
        """Demo 4: Multi-turn conversation -- context preserved across turns."""
        self.separator("Demo 4: Multi-Turn Conversation (Context Preservation)")

        with tempfile.TemporaryDirectory() as workspace:
            session = self.agent.session(workspace, permissive=True)

            print(f"  Workspace: {workspace}\n")

            # Turn 1
            print("  [Turn 1] Create a config file")
            r1 = session.send(
                "Create a file called `config.toml` with these settings:\n"
                '- server.host = "0.0.0.0"\n'
                "- server.port = 8080\n"
                '- database.url = "postgres://localhost/mydb"\n'
                "- database.pool_size = 5"
            )
            print(f"    Tools: {r1.tool_calls_count}, Tokens: {r1.total_tokens}")

            # Turn 2 -- LLM should remember the file from Turn 1
            print("\n  [Turn 2] Ask about the file (tests context memory)")
            r2 = session.send(
                "What port is the server configured to use? Read the config file to confirm."
            )
            print(f"    Tools: {r2.tool_calls_count}, Tokens: {r2.total_tokens}")
            print(f"    Answer: {self.truncate(r2.text, 120)}")

            # Turn 3 -- modify based on context
            print("\n  [Turn 3] Modify based on previous context")
            r3 = session.send(
                "Change the server port to 3000 and increase the pool_size to 10."
            )
            print(f"    Tools: {r3.tool_calls_count}, Tokens: {r3.total_tokens}")

            # Verify final state
            history = session.history()
            print(f"\n  History: {len(history)} messages across 3 turns")

            config_path = Path(workspace) / "config.toml"
            if config_path.exists():
                content = config_path.read_text()
                has_3000 = "3000" in content
                has_10 = "10" in content
                print(f"  config.toml: port=3000? {has_3000} pool=10? {has_10}")

    # ========================================================================
    # Demo 5: Skills-Augmented Code Assistance
    # ========================================================================

    def demo_5_skills_augmented(self) -> None:
        """Demo 5: Skills-augmented session for code review."""
        self.separator("Demo 5: Skills-Augmented Code Assistance")

        with tempfile.TemporaryDirectory() as workspace:
            # Pre-create a file with intentional issues
            app_path = Path(workspace) / "app.py"
            app_path.write_text(
                'import os\n'
                'import sys\n'
                '\n'
                'def process_data(data):\n'
                '    result = []\n'
                '    for i in range(len(data)):\n'
                '        if data[i] != None:\n'
                '            result.append(data[i] * 2)\n'
                '    return result\n'
                '\n'
                'def read_file(path):\n'
                '    f = open(path, \'r\')\n'
                '    content = f.read()\n'
                '    return content\n'
                '\n'
                'API_KEY = "sk-1234567890abcdef"\n'
            )

            # List available built-in skills
            skills = builtin_skills()
            session = self.agent.session(workspace, builtin_skills=True, permissive=True)

            print(f"  Workspace: {workspace}")
            print(f"  Skills:    built-in ({len(skills)} skills active)\n")

            result = session.send(
                "Review the file `app.py` in the workspace. Identify code quality issues "
                "and fix them. The review should cover:\n"
                "- Pythonic style (use `is not None`, enumerate, etc.)\n"
                "- Resource management (use context managers)\n"
                "- Security issues (hardcoded secrets)\n"
                "Apply the fixes by editing the file."
            )

            print(f"  Tool calls: {result.tool_calls_count}")
            print(f"  Tokens:     {result.total_tokens} total")
            print(f"  Response:   {self.truncate(result.text)}")

            # Check improvements
            if app_path.exists():
                content = app_path.read_text()
                has_ctx_mgr = "with open" in content
                has_enumerate = "enumerate" in content
                no_key = "sk-1234567890abcdef" not in content
                print(
                    f"\n  Improvements: context_manager={has_ctx_mgr} "
                    f"enumerate={has_enumerate} no_hardcoded_key={no_key}"
                )

    # ========================================================================
    # Demo 6: Resilient Session
    # ========================================================================

    def demo_6_resilient_session(self) -> None:
        """Demo 6: Resilient session with error recovery."""
        self.separator("Demo 6: Resilient Session (Error Recovery)")

        with tempfile.TemporaryDirectory() as workspace:
            session = self.agent.session(
                workspace,
                builtin_skills=True,
                permissive=True,
                max_parse_retries=2,
                tool_timeout_ms=120_000,
                circuit_breaker_threshold=3,
            )

            print(f"  Workspace:       {workspace}")
            print("  Parse retries:   2")
            print("  Tool timeout:    120s")
            print("  Circuit breaker: 3\n")

            result = session.send(
                "Create a file called `test.sh` with a bash script that prints 'hello' "
                "and the current date. Then execute it with bash and show me the output."
            )

            print(f"  Tool calls: {result.tool_calls_count}")
            print(f"  Tokens:     {result.total_tokens} total")
            print(f"  Response:   {self.truncate(result.text)}")

            if (Path(workspace) / "test.sh").exists():
                print("\n  test.sh created and executed successfully")

    # ========================================================================
    # Run All Demos
    # ========================================================================

    def run_all(self) -> None:
        print("+" + "=" * 70 + "+")
        print("|        A3S Code Python SDK -- Agentic Loop Demo (Real LLM)          |")
        print("+" + "=" * 70 + "+")

        print(f"\n  Config: {self.config_path}")
        print("  Agent:  created\n")

        self.demo_1_autonomous_coding()
        self.demo_2_streaming_events()
        self.demo_3_planning_mode()
        self.demo_4_multi_turn()
        self.demo_5_skills_augmented()
        self.demo_6_resilient_session()

        self.separator("All Demos Complete")


if __name__ == "__main__":
    config_path = AgenticLoopDemo.find_config_path()
    agent = Agent.create(config_path)
    demo = AgenticLoopDemo(agent, config_path)
    demo.run_all()
