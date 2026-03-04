"""
Git Worktree Tool Test with Real LLM

Demonstrates the git_worktree builtin tool via the Python SDK:
1. Initialize a git repo in a temp directory
2. Direct tool calls: status, create, list, remove
3. LLM-driven: ask the agent to use git_worktree

Run with: python test_git_worktree.py
"""

import os
import subprocess
import tempfile
from pathlib import Path

from a3s_code import Agent


class GitWorktreeTest:
    """Integration test for the git_worktree builtin tool."""

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

    @staticmethod
    def init_git_repo(path: str) -> None:
        """Initialize a git repo with one commit."""
        subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, capture_output=True, check=True)
        Path(path, "README.md").write_text("# Test Repo\n")
        subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=path, capture_output=True, check=True)

    def test_worktree_status(self, session: object) -> None:
        """Test 1: Direct tool call — status."""
        print("═══ Test 1: git_worktree status ═══")
        result = session.tool("git_worktree", {"command": "status"})
        print(result.output)
        assert result.exit_code == 0, "status should succeed"
        print()

    def test_worktree_create(self, session: object, workspace: str) -> str:
        """Test 2: Direct tool call — create worktree. Returns the worktree path."""
        print("═══ Test 2: git_worktree create ═══")
        wt_path = os.path.join(workspace, "wt-feature-auth")
        result = session.tool("git_worktree", {
            "command": "create",
            "branch": "feature-auth",
            "path": wt_path,
        })
        print(result.output)
        assert result.exit_code == 0, f"create failed: {result.output}"
        assert os.path.exists(wt_path), "worktree directory should exist"
        print()
        return wt_path

    def test_worktree_list(self, session: object) -> None:
        """Test 3: Direct tool call — list."""
        print("═══ Test 3: git_worktree list ═══")
        result = session.tool("git_worktree", {"command": "list"})
        print(result.output)
        assert result.exit_code == 0
        assert "feature-auth" in result.output, "list should contain the new branch"
        print()

    def test_llm_driven_query(self, session: object) -> None:
        """Test 4: LLM-driven query."""
        print("═══ Test 4: LLM-driven worktree query ═══")
        llm_result = session.send(
            "Use the git_worktree tool with command 'list' to show me all worktrees. "
            "Just show the tool output, nothing else."
        )
        print(f"LLM response:\n{llm_result.text}")
        assert llm_result.tool_calls_count > 0, "LLM should have called git_worktree"
        print()

    def test_worktree_remove(self, session: object, wt_path: str) -> None:
        """Test 5: Direct tool call — remove."""
        print("═══ Test 5: git_worktree remove ═══")
        result = session.tool("git_worktree", {
            "command": "remove",
            "path": wt_path,
        })
        print(result.output)
        assert result.exit_code == 0, f"remove failed: {result.output}"
        assert not os.path.exists(wt_path), "worktree directory should be gone"
        print()

    def test_verify_cleanup(self, session: object) -> None:
        """Test 6: Verify cleanup."""
        print("═══ Test 6: Verify cleanup ═══")
        result = session.tool("git_worktree", {"command": "list"})
        print(result.output)
        assert "feature-auth" not in result.output
        print()

    def run_all(self) -> None:
        """Run all git_worktree tests."""
        print(f"Config: {self.config_path}")
        print("Agent created ✓")

        with tempfile.TemporaryDirectory() as workspace:
            self.init_git_repo(workspace)
            print(f"Git repo initialized at: {workspace}\n")

            session = self.agent.session(workspace)

            self.test_worktree_status(session)
            wt_path = self.test_worktree_create(session, workspace)
            self.test_worktree_list(session)
            self.test_llm_driven_query(session)
            self.test_worktree_remove(session, wt_path)
            self.test_verify_cleanup(session)

        print("═══ All git_worktree tests passed ✓ ═══")


if __name__ == "__main__":
    config = GitWorktreeTest.find_config()
    agent = Agent(config)
    test = GitWorktreeTest(agent, config)
    test.run_all()
