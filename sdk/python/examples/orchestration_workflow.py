#!/usr/bin/env python3
"""
Programmable orchestration via the A3S Code Python SDK.

Demonstrates the deterministic grammar exposed on a session:
  - session.parallel(specs)            — fan out steps, results in input order
  - session.pipeline(items, stages)    — per-item chains, no inter-stage barrier
  - session.parallel_resumable(...)     — journaled, resumable fan-out

Config is loaded from .a3s/config.acl at runtime (never hardcoded).

Run: cd crates/code/sdk/python && python examples/orchestration_workflow.py
"""

import sys
from pathlib import Path

from a3s_code import Agent, SessionOptions


def repo_config_path() -> Path:
    """Find .a3s/config.acl walking up from this file."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / ".a3s" / "config.acl"
        if candidate.exists():
            return candidate
    raise SystemExit("could not locate .a3s/config.acl above this example")


def main() -> int:
    config = str(repo_config_path())
    agent = Agent.create(config)
    session = agent.session(".", SessionOptions())

    # 1. parallel — fan out independent steps; outcomes come back in order.
    outcomes = session.parallel(
        [
            {
                "task_id": "langs",
                "agent": "general",
                "description": "list languages",
                "prompt": "Name three systems programming languages, comma-separated.",
                "max_steps": 2,
            },
            {
                "task_id": "verdict",
                "agent": "general",
                "description": "classify",
                "prompt": "Is Rust memory-safe without a GC? Answer yes or no.",
                "max_steps": 2,
                # Schema-validated structured output for this step.
                "output_schema": {
                    "type": "object",
                    "properties": {"memory_safe": {"type": "boolean"}},
                    "required": ["memory_safe"],
                },
            },
        ]
    )
    for o in outcomes:
        print(f"[parallel] {o['task_id']}: success={o['success']} structured={o.get('structured')}")

    # 2. pipeline — each item chains through stages; stage 2 builds on stage 1.
    #    Return None from a stage (or raise — caught and treated as None) to
    #    stop that item's chain.
    results = session.pipeline(
        ["the Rust programming language"],
        [
            lambda ctx: {
                "task_id": "summarize",
                "agent": "general",
                "description": "summarize",
                "prompt": f"In one sentence, what is {ctx['item']}?",
                "max_steps": 2,
            },
            lambda ctx: {
                "task_id": "classify",
                "agent": "general",
                "description": "classify",
                "prompt": "Reply with one word YES or NO: does this describe a "
                f"programming language?\n\n{ctx['previous']['output']}",
                "max_steps": 2,
            },
        ],
    )
    for r in results:
        print(f"[pipeline] final={None if r is None else r['output'][:60]!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
