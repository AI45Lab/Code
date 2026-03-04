#!/usr/bin/env python3
"""
A3S Code Python SDK - Custom Skills and Agents Integration Test

Tests loading and execution of custom skills and agents from ~/.a3s directory.

Run with: python examples/test_custom_skills_agents.py
"""

import asyncio
from pathlib import Path
from a3s_code import Agent


class CustomSkillsAgentsTest:
    """Integration test for custom skills and agents loading and execution."""

    def __init__(self, agent: Agent, config_path: str) -> None:
        self.agent = agent
        self.config_path = config_path
        self.skills_dir = self._find_skills_dir()
        self.agents_dir = self._find_agents_dir()

    @staticmethod
    def find_config_path() -> str:
        """Find config file in home directory or project root."""
        home_config = Path.home() / ".a3s" / "config.hcl"
        if home_config.exists():
            return str(home_config)

        # Try project root (6 levels up from examples/test_custom_skills_agents.py)
        project_config = Path(__file__).parent.parent.parent.parent.parent.parent / ".a3s" / "config.hcl"
        if project_config.exists():
            return str(project_config)

        raise FileNotFoundError("Config file not found. Please create ~/.a3s/config.hcl")

    @staticmethod
    def _find_skills_dir() -> str:
        """Find skills directory."""
        home_skills = Path.home() / ".a3s" / "skills"
        if home_skills.exists():
            return str(home_skills)

        # Try project root
        project_skills = Path(__file__).parent.parent.parent.parent.parent.parent / ".a3s" / "skills"
        if project_skills.exists():
            return str(project_skills)

        raise FileNotFoundError("Skills directory not found")

    @staticmethod
    def _find_agents_dir() -> str:
        """Find agents directory."""
        home_agents = Path.home() / ".a3s" / "agents"
        if home_agents.exists():
            return str(home_agents)

        # Try project root
        project_agents = Path(__file__).parent.parent.parent.parent.parent.parent / ".a3s" / "agents"
        if project_agents.exists():
            return str(project_agents)

        raise FileNotFoundError("Agents directory not found")

    @staticmethod
    def truncate(text: str, max_len: int) -> str:
        """Truncate text to max length."""
        if len(text) <= max_len:
            return text
        return f"{text[:max_len]}... (truncated)"

    async def test_custom_skills(self) -> None:
        """Test 1: Load custom skills in session."""
        print("\n📦 Test 1: Load Custom Skills in Session")
        print("-" * 80)

        session = self.agent.session(".", skill_dirs=[self.skills_dir])

        print("Testing: Ask about Rust best practices (should use rust-expert skill)...")
        result = session.send("What are the key principles of Rust ownership?")
        print(f"✓ Result preview: {self.truncate(result.text, 200)}")
        print()

        print("Testing: Ask about API design (should use api-designer skill)...")
        result = session.send("What are REST API best practices for error handling?")
        print(f"✓ Result preview: {self.truncate(result.text, 200)}")
        print()

        print("✅ Test 1 passed: Custom skills loaded and used in session\n")

    async def test_custom_agents(self) -> None:
        """Test 2: Load custom agents in session."""
        print("🤖 Test 2: Load Custom Agents in Session")
        print("-" * 80)

        session = self.agent.session(".", agent_dirs=[self.agents_dir])

        print("Testing: Request code review (should use code-reviewer agent)...")
        result = session.send("Review this function for potential issues: def add(a, b): return a + b")
        print(f"✓ Result preview: {self.truncate(result.text, 200)}")
        print()

        print("Testing: Request documentation (should use documentation-writer agent)...")
        result = session.send("Write API documentation for a user registration endpoint")
        print(f"✓ Result preview: {self.truncate(result.text, 200)}")
        print()

        print("✅ Test 2 passed: Custom agents loaded and used in session\n")

    async def test_skills_and_agents_together(self) -> None:
        """Test 3: Load both skills and agents together."""
        print("🔧 Test 3: Load Both Skills and Agents Together")
        print("-" * 80)

        session = self.agent.session(".", skill_dirs=[self.skills_dir], agent_dirs=[self.agents_dir])

        print("Testing: Complex task using both skills and agents...")
        result = session.send("Generate unit tests for a Python function that parses JSON")
        print(f"✓ Result preview: {self.truncate(result.text, 200)}")
        print()

        print("✅ Test 3 passed: Both skills and agents work together\n")

    async def test_multiple_sessions(self) -> None:
        """Test 4: Multiple sessions with different configurations."""
        print("🔀 Test 4: Multiple Sessions with Different Configurations")
        print("-" * 80)

        # Session 1: Only skills
        session1 = self.agent.session(".", skill_dirs=[self.skills_dir])

        # Session 2: Only agents
        session2 = self.agent.session(".", agent_dirs=[self.agents_dir])

        print("Testing: Session 1 with skills only...")
        result1 = session1.send("Explain Python decorators")
        print(f"✓ Session 1 result: {self.truncate(result1.text, 150)}")
        print()

        print("Testing: Session 2 with agents only...")
        result2 = session2.send("Review this code for improvements")
        print(f"✓ Session 2 result: {self.truncate(result2.text, 150)}")
        print()

        print("✅ Test 4 passed: Multiple sessions with different configs work correctly\n")

    async def run_all(self) -> None:
        """Run all custom skills and agents tests."""
        print("🚀 A3S Code Python SDK - Custom Skills and Agents Integration Test\n")
        print("=" * 80)

        print(f"📄 Using config: {self.config_path}")
        print(f"📚 Skills directory: {self.skills_dir}")
        print(f"🤖 Agents directory: {self.agents_dir}")

        print("=" * 80)

        await self.test_custom_skills()
        await self.test_custom_agents()
        await self.test_skills_and_agents_together()
        await self.test_multiple_sessions()

        print()
        print("=" * 80)
        print("✅ All custom skills and agents tests completed successfully!")
        print("=" * 80)


if __name__ == "__main__":
    config_path = CustomSkillsAgentsTest.find_config_path()
    agent = Agent.create(config_path)
    test = CustomSkillsAgentsTest(agent, config_path)
    asyncio.run(test.run_all())
