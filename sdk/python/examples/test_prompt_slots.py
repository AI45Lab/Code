"""
System Prompt Slots — Customizing agent personality without overriding core behavior

Demonstrates the slot-based system prompt customization API:
1. Custom role (persona)
2. Custom guidelines (coding standards)
3. Custom response style
4. Extra freeform instructions
5. Verify core tool behavior is preserved

Run with: python test_prompt_slots.py
"""

import os
import tempfile
from pathlib import Path

from a3s_code import Agent, SessionOptions


class PromptSlotsTest:
    """Integration test for system prompt slot-based customization."""

    def __init__(self, agent: Agent, config_path: str) -> None:
        self.agent = agent
        self.config_path = config_path

    @staticmethod
    def find_config() -> str:
        """Find config file from environment or home directory."""
        if "A3S_CONFIG" in os.environ:
            return os.environ["A3S_CONFIG"]
        home_config = Path.home() / ".a3s" / "config.hcl"
        if home_config.exists():
            return str(home_config)
        raise FileNotFoundError("Config not found. Create ~/.a3s/config.hcl or set A3S_CONFIG")

    def test_custom_role(self, workspace: str) -> None:
        """Test 1: Custom role only."""
        print("═══ Test 1: Custom role ═══")
        opts = SessionOptions()
        opts.role = "You are a senior Rust developer who specializes in async programming."
        session = self.agent.session(workspace, options=opts)
        result = session.send("What is your area of expertise? Reply in one sentence.")
        print(f"Response: {result.text.strip()}\n")
        assert result.text, "should get a response"

    def test_role_guidelines_style(self, workspace: str) -> None:
        """Test 2: Role + guidelines + response style."""
        print("═══ Test 2: Role + guidelines + response style ═══")
        opts = SessionOptions()
        opts.role = "You are a Python code reviewer."
        opts.guidelines = "Always check for type hints. Flag any use of `eval()`."
        opts.response_style = "Reply in bullet points. Be concise."
        session = self.agent.session(workspace, options=opts)

        # Write a Python file for the agent to review
        Path(workspace, "app.py").write_text(
            'def add(a, b):\n'
            '    return eval(f"{a} + {b}")\n'
            '\n'
            'def greet(name):\n'
            '    return "Hello " + name\n'
        )

        result = session.send("Review the file app.py and list any issues you find.")
        print(f"Response:\n{result.text.strip()}\n")
        assert result.tool_calls_count > 0, "should have used read_file tool"

    def test_extra_instructions(self, workspace: str) -> None:
        """Test 3: Extra instructions only."""
        print("═══ Test 3: Extra instructions ═══")
        opts = SessionOptions()
        opts.extra = "Always end your response with '— A3S'"
        session = self.agent.session(workspace, options=opts)
        result = session.send("Say hello.")
        print(f"Response: {result.text.strip()}\n")

    def test_core_tool_behavior(self, workspace: str) -> None:
        """Test 4: Verify tools still work (core behavior preserved)."""
        print("═══ Test 4: Core tool behavior preserved ═══")
        opts = SessionOptions()
        opts.role = "You are a minimalist file manager."
        opts.guidelines = "Only create files when explicitly asked."
        session = self.agent.session(workspace, options=opts)
        result = session.send(
            "Create a file called test.txt with the content 'prompt slots work'. "
            "Then read it back."
        )
        print(f"Response: {result.text.strip()}\n")
        assert result.tool_calls_count > 0, "should have used tools"
        assert Path(workspace, "test.txt").exists(), "test.txt should exist"

    def run_all(self) -> None:
        """Run all prompt slots tests."""
        print(f"Config: {self.config_path}")
        print("Agent created ✓\n")

        with tempfile.TemporaryDirectory() as workspace:
            self.test_custom_role(workspace)
            self.test_role_guidelines_style(workspace)
            self.test_extra_instructions(workspace)
            self.test_core_tool_behavior(workspace)

        print("═══ All prompt slots tests passed ✓ ═══")


if __name__ == "__main__":
    config = PromptSlotsTest.find_config()
    agent = Agent(config)
    test = PromptSlotsTest(agent, config)
    test.run_all()
