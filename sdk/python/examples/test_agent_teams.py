#!/usr/bin/env python3
"""
A3S Code Python SDK - Agent Teams Integration Test

Demonstrates multi-agent team coordination using TeamRunner:
  - Lead decomposes a goal into tasks via LLM JSON response
  - Workers concurrently claim and execute tasks
  - Reviewer approves or rejects completed work
  - Rejected tasks are re-queued for retry

Run with: python examples/test_agent_teams.py
"""

import time
from pathlib import Path
from a3s_code import Agent, Team, TeamRunner, TeamConfig, TeamTaskBoard


class AgentTeamsTest:
    """Integration tests for multi-agent team coordination."""

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path

    @staticmethod
    def find_config_path() -> str:
        """Find config file in home directory or project root."""
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

        raise FileNotFoundError("Config file not found. Please create ~/.a3s/config.hcl")

    @staticmethod
    def print_board(board: TeamTaskBoard) -> None:
        """Print a quick summary of the task board."""
        (open_c, prog, rev, done, rej) = board.stats()
        print(
            f"  Board: total={len(board)}, open={open_c}, "
            f"in_progress={prog}, in_review={rev}, done={done}, rejected={rej}"
        )

    # ========================================================================
    # Test 1: Manual Task Board Coordination
    # ========================================================================

    def test_manual_coordination(self) -> None:
        """Test 1: Manual task board API (no LLM)."""
        print("\n Test 1: Manual Task Board Coordination")
        print("-" * 80)

        config = TeamConfig(max_tasks=10, max_rounds=5, poll_interval_ms=50)
        team = Team("manual-test", config)
        team.add_member("lead", "lead")
        team.add_member("worker-1", "worker")
        team.add_member("reviewer", "reviewer")

        board = team.task_board()

        # Lead posts tasks
        id1 = board.post("Refactor auth to JWT", "lead")
        id2 = board.post("Write integration tests", "lead")
        id3 = board.post("Update API docs", "lead")
        assert id1 and id2 and id3, "Tasks should be posted"

        print(f"  Posted 3 tasks. Board state:")
        self.print_board(board)

        # Worker claims first task
        task = board.claim("worker-1")
        assert task is not None, "Worker should claim a task"
        assert task.status == "in_progress"
        print(f"  Worker claimed: [{task.id}] {task.description}")

        # Worker completes it
        board.complete(task.id, "Auth refactored to use JWT with RS256 keys")
        completed = board.get(task.id)
        assert completed.status == "in_review"
        assert completed.result == "Auth refactored to use JWT with RS256 keys"
        print(f"  Task submitted for review -> status: {completed.status}")

        # Reviewer rejects then approves
        board.reject(task.id)
        rejected = board.get(task.id)
        assert rejected.status == "rejected"
        print(f"  Reviewer rejected -> status: {rejected.status}")

        # Worker re-claims the rejected task
        retried = board.claim("worker-1")
        assert retried is not None and retried.id == task.id
        board.complete(task.id, "Auth refactored with proper token rotation")
        board.approve(task.id)
        approved = board.get(task.id)
        assert approved.status == "done"
        print(f"  After retry: reviewer approved -> status: {approved.status}")

        # Check by_status
        done_tasks = board.by_status("done")
        open_tasks = board.by_status("open")
        assert len(done_tasks) == 1
        assert len(open_tasks) == 2
        print(f"  Done: {len(done_tasks)}, Open: {len(open_tasks)}")

        print("\n  Test 1 passed: Manual task board coordination works\n")

    # ========================================================================
    # Test 2: Full TeamRunner Workflow
    # ========================================================================

    def test_team_runner_workflow(self) -> None:
        """Test 2: Full TeamRunner workflow with real LLM."""
        print("\n Test 2: TeamRunner -- Automated Lead -> Worker -> Reviewer Workflow")
        print("-" * 80)

        agent = Agent.create(self.config_path)

        config = TeamConfig(
            max_tasks=20,
            max_rounds=8,
            poll_interval_ms=100,
        )
        team = Team("code-review-team", config)
        team.add_member("lead", "lead")
        team.add_member("worker-1", "worker")
        team.add_member("reviewer", "reviewer")

        print(f"  Team created with {team.member_count()} members")

        runner = TeamRunner(team)
        runner.bind_session("lead", agent.session("."))
        runner.bind_session("worker-1", agent.session("."))
        runner.bind_session("reviewer", agent.session("."))

        goal = (
            "Audit this Python codebase for code quality issues. "
            "Identify and document the top 3 most important improvements."
        )
        print(f"  Goal: {goal}")
        print("  Running team workflow (Lead decomposes -> Workers execute -> Reviewer approves)...")

        start = time.time()
        result = runner.run_until_done(goal)
        elapsed = time.time() - start

        print(f"\n  Completed in {elapsed:.1f}s, {result.rounds} reviewer rounds")
        print(f"  Done tasks: {len(result.done_tasks)}")
        print(f"  Rejected tasks: {len(result.rejected_tasks)}")

        for task in result.done_tasks:
            snippet = (task.result or "")[:120].replace("\n", " ")
            print(f"\n  [{task.id[:8]}] {task.description}")
            print(f"    Result: {snippet}{'...' if len(task.result or '') > 120 else ''}")

        assert len(result.done_tasks) > 0, "At least one task should be completed"
        print("\n  Test 2 passed: TeamRunner executed end-to-end workflow\n")

    # ========================================================================
    # Test 3: TeamRunner Without Reviewer
    # ========================================================================

    def test_team_runner_no_reviewer(self) -> None:
        """Test 3: TeamRunner without a reviewer -- tasks reach InReview state."""
        print("\n Test 3: TeamRunner Without Reviewer (tasks reach InReview)")
        print("-" * 80)

        agent = Agent.create(self.config_path)

        team = Team("no-reviewer-team", TeamConfig(max_rounds=3, poll_interval_ms=50))
        team.add_member("lead", "lead")
        team.add_member("worker-1", "worker")
        # Note: no reviewer bound

        runner = TeamRunner(team)
        runner.bind_session("lead", agent.session("."))
        runner.bind_session("worker-1", agent.session("."))

        board = runner.task_board()
        result = runner.run_until_done("Count the number of Python files in this project")

        in_review = board.by_status("in_review")
        done = board.by_status("done")
        print(f"  InReview: {len(in_review)}, Done: {len(done)}")

        # Without reviewer, tasks stay InReview
        assert len(in_review) + len(done) > 0, "Tasks should have been executed"
        print("\n  Test 3 passed: Team runs correctly without a reviewer\n")

    # ========================================================================
    # Run All Tests
    # ========================================================================

    def run_all(self) -> None:
        print("=" * 80)
        print("  A3S Code -- Agent Teams Integration Tests")
        print("=" * 80)

        try:
            self.test_manual_coordination()
            self.test_team_runner_workflow()
            self.test_team_runner_no_reviewer()

            print("=" * 80)
            print("  All agent team tests passed!")
            print("=" * 80)
        except Exception as e:
            print(f"\n  Test failed: {e}")
            raise


if __name__ == "__main__":
    config_path = AgentTeamsTest.find_config_path()
    suite = AgentTeamsTest(config_path)
    suite.run_all()
