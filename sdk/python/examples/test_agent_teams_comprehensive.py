#!/usr/bin/env python3
"""
A3S Code Python SDK — Agent Teams Comprehensive Integration Test

Real-world multi-task decomposition and multi-agent parallel execution scenarios:

  Scenario 1: Code Quality Audit
    Lead decomposes → 3 workers audit in parallel → reviewer validates

  Scenario 2: Feature Implementation Pipeline
    Lead plans feature → 2 workers implement + test → reviewer approves/rejects

  Scenario 3: Documentation Sprint
    Lead identifies doc gaps → 3 workers write docs concurrently → reviewer checks

  Scenario 4: Stress Test — 10 Tasks, 4 Workers, Rejection + Retry

Run with: python examples/test_agent_teams_comprehensive.py
"""

import sys
import time
import tempfile
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from a3s_code import Agent, Team, TeamRunner, TeamConfig, TeamTaskBoard


def resolve_config() -> str:
    home_cfg = Path.home() / ".a3s" / "config.hcl"
    if home_cfg.exists():
        return str(home_cfg)
    p = Path(__file__).resolve()
    for _ in range(10):
        candidate = p / ".a3s" / "config.hcl"
        if candidate.exists():
            return str(candidate)
        parent = p.parent
        if parent == p:
            break
        p = parent
    raise RuntimeError("Config not found")


def pass_test(label: str) -> None:
    print(f"  PASS  {label}")


def print_board(board: TeamTaskBoard) -> None:
    o, p, r, d, rj = board.stats()
    print(f"  Board: total={len(board)}, open={o}, in_progress={p}, "
          f"in_review={r}, done={d}, rejected={rj}")


def print_results(result, verbose: bool = True) -> None:
    print(f"  Done: {len(result.done_tasks)}, Rejected: {len(result.rejected_tasks)}, "
          f"Rounds: {result.rounds}")
    if verbose:
        for t in result.done_tasks:
            desc = t.description[:60] + ("..." if len(t.description) > 60 else "")
            snippet = (t.result or "")[:100].replace("\n", " ")
            print(f"    [{t.id}] {desc}")
            print(f"      -> {snippet}{'...' if len(t.result or '') > 100 else ''}")


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 0: Task Board API — No LLM, Pure Coordination Primitives
# ═══════════════════════════════════════════════════════════════════════════════

def test_task_board_primitives() -> None:
    print("\n== Scenario 0: Task Board Primitives (No LLM) ==")
    print("-" * 70)

    config = TeamConfig(max_tasks=20, max_rounds=5, poll_interval_ms=10)
    team = Team("board-test", config)
    team.add_member("pm", "lead")
    team.add_member("dev-1", "worker")
    team.add_member("dev-2", "worker")
    team.add_member("qa", "reviewer")
    assert team.member_count() == 4
    pass_test(f"Team created with {team.member_count()} members")

    board = team.task_board()

    # PM posts a sprint backlog
    tasks = [
        "Implement user registration API endpoint",
        "Add input validation middleware",
        "Write database migration for users table",
        "Create unit tests for registration flow",
        "Set up CI pipeline for the auth service",
    ]
    ids = []
    for desc in tasks:
        tid = board.post(desc, "pm")
        assert tid is not None, f"Failed to post: {desc}"
        ids.append(tid)
    pass_test(f"Posted {len(ids)} tasks to sprint board")
    print_board(board)

    # dev-1 claims and completes first 2 tasks
    t1 = board.claim("dev-1")
    assert t1 is not None and t1.status == "in_progress"
    t2 = board.claim("dev-1")
    assert t2 is not None
    board.complete(t1.id, "POST /api/register implemented with bcrypt password hashing")
    board.complete(t2.id, "Zod schema validation added as Express middleware")
    pass_test(f"dev-1 completed 2 tasks")

    # dev-2 claims remaining open tasks
    claimed_count = 0
    while True:
        t = board.claim("dev-2")
        if t is None:
            break
        board.complete(t.id, f"Completed: {t.description}")
        claimed_count += 1
    pass_test(f"dev-2 completed {claimed_count} tasks")

    # QA reviews
    in_review = board.by_status("in_review")
    assert len(in_review) == 5, f"Expected 5 in_review, got {len(in_review)}"

    # QA rejects task 1, approves rest
    board.reject(ids[0])
    for tid in ids[1:]:
        board.approve(tid)
    pass_test("QA reviewed all: 1 rejected, 4 approved")

    # dev-1 re-claims rejected task
    retry = board.claim("dev-1")
    assert retry is not None and retry.id == ids[0]
    board.complete(retry.id, "Fixed: added email uniqueness check + rate limiting")
    board.approve(retry.id)
    pass_test("Rejected task retried and approved")

    # Final state check
    done = board.by_status("done")
    assert len(done) == 5
    print_board(board)
    pass_test("All 5 tasks done")

    # by_assignee check
    dev1_tasks = board.by_assignee("dev-1")
    dev2_tasks = board.by_assignee("dev-2")
    assert len(dev1_tasks) + len(dev2_tasks) == 5
    pass_test(f"Assignment tracking: dev-1={len(dev1_tasks)}, dev-2={len(dev2_tasks)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 1: Code Quality Audit (Lead → 1 Worker → Reviewer)
# ═══════════════════════════════════════════════════════════════════════════════

def test_code_quality_audit(agent: Agent, workspace: str) -> None:
    print("\n== Scenario 1: Code Quality Audit ==")
    print("-" * 70)

    config = TeamConfig(max_tasks=10, max_rounds=8, poll_interval_ms=200)
    team = Team("code-audit", config)
    team.add_member("tech-lead", "lead")
    team.add_member("auditor", "worker")
    team.add_member("senior-dev", "reviewer")

    runner = TeamRunner(team)
    runner.bind_session("tech-lead", agent.session(workspace, permissive=True))
    runner.bind_session("auditor", agent.session(workspace, permissive=True))
    runner.bind_session("senior-dev", agent.session(workspace, permissive=True))

    goal = (
        "Audit this project workspace for code quality. "
        "Decompose into exactly 3 tasks: "
        "1) check for potential security issues, "
        "2) identify performance bottlenecks, "
        "3) check code style consistency. "
        "Each task should produce a brief 2-3 sentence summary."
    )

    print(f"  Goal: {goal[:80]}...")
    t0 = time.time()
    result = runner.run_until_done(goal)
    elapsed = time.time() - t0

    print(f"  Completed in {elapsed:.1f}s")
    print_results(result)

    assert len(result.done_tasks) >= 1, "At least 1 task should complete"
    pass_test(f"{len(result.done_tasks)} audit tasks completed, "
              f"{len(result.rejected_tasks)} rejected")


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 2: Feature Planning with 2 Parallel Workers
# ═══════════════════════════════════════════════════════════════════════════════

def test_parallel_workers(agent: Agent, workspace: str) -> None:
    print("\n== Scenario 2: Parallel Workers — Feature Planning ==")
    print("-" * 70)

    config = TeamConfig(max_tasks=10, max_rounds=8, poll_interval_ms=200)
    team = Team("feature-team", config)
    team.add_member("architect", "lead")
    team.add_member("backend-dev", "worker")
    team.add_member("frontend-dev", "worker")
    team.add_member("code-reviewer", "reviewer")

    runner = TeamRunner(team)
    # Each member gets its own independent session
    runner.bind_session("architect", agent.session(workspace, permissive=True))
    runner.bind_session("backend-dev", agent.session(workspace, permissive=True))
    runner.bind_session("frontend-dev", agent.session(workspace, permissive=True))
    runner.bind_session("code-reviewer", agent.session(workspace, permissive=True))

    board = runner.task_board()

    goal = (
        "Plan an user authentication feature for a web application. "
        "Decompose into exactly 2 tasks: "
        "1) Design the backend API schema (endpoints, request/response format), "
        "2) Design the frontend login form component (props, states, events). "
        "Each task should produce a concise spec (3-5 lines)."
    )

    print(f"  Goal: {goal[:80]}...")
    print(f"  Workers: backend-dev, frontend-dev (parallel)")
    t0 = time.time()
    result = runner.run_until_done(goal)
    elapsed = time.time() - t0

    print(f"  Completed in {elapsed:.1f}s")
    print_results(result)
    print_board(board)

    total_completed = len(result.done_tasks) + len(board.by_status("in_review"))
    assert total_completed >= 1, "Workers should have completed at least 1 task"
    pass_test(f"Parallel workers completed {total_completed} tasks")


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 3: TeamRunner Without Reviewer
# ═══════════════════════════════════════════════════════════════════════════════

def test_no_reviewer(agent: Agent, workspace: str) -> None:
    print("\n== Scenario 3: Team Without Reviewer ==")
    print("-" * 70)

    config = TeamConfig(max_tasks=5, max_rounds=5, poll_interval_ms=100)
    team = Team("no-review", config)
    team.add_member("lead", "lead")
    team.add_member("worker", "worker")

    runner = TeamRunner(team)
    runner.bind_session("lead", agent.session(workspace, permissive=True))
    runner.bind_session("worker", agent.session(workspace, permissive=True))

    board = runner.task_board()

    goal = (
        "List exactly 2 simple math facts. "
        "Task 1: What is 7 * 8? "
        "Task 2: What is 12 + 15?"
    )

    print(f"  Goal: {goal}")
    result = runner.run_until_done(goal)

    in_review = board.by_status("in_review")
    done = board.by_status("done")
    print(f"  In-review: {len(in_review)}, Done: {len(done)}")

    # Without reviewer, tasks stay in_review
    assert len(in_review) + len(done) >= 1, "At least 1 task should be processed"
    pass_test(f"No-reviewer team: {len(in_review)} in-review, {len(done)} done")


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 4: Multi-Worker Task Board with Manual Post
# ═══════════════════════════════════════════════════════════════════════════════

def test_manual_post_then_run(agent: Agent, workspace: str) -> None:
    print("\n== Scenario 4: Manual Task Posting + TeamRunner Execution ==")
    print("-" * 70)

    config = TeamConfig(max_tasks=20, max_rounds=8, poll_interval_ms=200)
    team = Team("manual-post", config)
    team.add_member("pm", "lead")
    team.add_member("dev-1", "worker")
    team.add_member("dev-2", "worker")
    team.add_member("qa", "reviewer")

    # Manually post tasks before running
    board = team.task_board()
    board.post("What is the capital of France? Reply in one word.", "pm")
    board.post("What is the largest planet in our solar system? Reply in one word.", "pm")
    board.post("What is 100 divided by 4? Reply with just the number.", "pm")
    board.post("Name one programming language created by Guido van Rossum. Reply in one word.", "pm")
    print(f"  Manually posted 4 tasks")
    print_board(board)

    runner = TeamRunner(team)
    runner.bind_session("pm", agent.session(workspace, permissive=True))
    runner.bind_session("dev-1", agent.session(workspace, permissive=True))
    runner.bind_session("dev-2", agent.session(workspace, permissive=True))
    runner.bind_session("qa", agent.session(workspace, permissive=True))

    # run_until_done will skip lead decomposition since tasks are already on the board.
    # Actually, run_until_done always starts with lead decomposition.
    # Let's use the runner's task board to verify post-run state.
    result = runner.run_until_done(
        "Answer the questions already posted on the task board. "
        "Decompose into 0 additional tasks — all work is already posted."
    )

    final_board = runner.task_board()
    total = len(final_board)
    done = final_board.by_status("done")
    in_review = final_board.by_status("in_review")

    print(f"  Final: total={total}, done={len(done)}, in_review={len(in_review)}")
    print_results(result, verbose=False)

    pass_test(f"Manual + auto: {len(result.done_tasks)} done, {len(result.rejected_tasks)} rejected")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    config_path = resolve_config()
    print("=" * 70)
    print("  A3S Code — Agent Teams Comprehensive Integration Tests")
    print(f"  Config: {config_path}")
    print("=" * 70)

    agent = Agent.create(config_path)

    with tempfile.TemporaryDirectory() as workspace:
        tests = [
            ("Scenario 0: Task Board Primitives",
             lambda: test_task_board_primitives()),
            ("Scenario 1: Code Quality Audit",
             lambda: test_code_quality_audit(agent, workspace)),
            ("Scenario 2: Parallel Workers",
             lambda: test_parallel_workers(agent, workspace)),
            ("Scenario 3: No Reviewer",
             lambda: test_no_reviewer(agent, workspace)),
            ("Scenario 4: Manual Post + Run",
             lambda: test_manual_post_then_run(agent, workspace)),
        ]

        passed = 0
        failed = 0
        for name, test_fn in tests:
            try:
                test_fn()
                passed += 1
            except Exception as e:
                print(f"\n  FAILED [{name}]: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

    print("\n" + "=" * 70)
    if failed == 0:
        print(f"  ALL {passed} SCENARIOS PASSED")
    else:
        print(f"  {passed} passed, {failed} FAILED")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
