#!/usr/bin/env python3
"""
Simple test to verify SubAgent event types are recognized.
"""

from pathlib import Path
from a3s_code import Agent


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


def test_subagent_events():
    """Test SubAgent event recognition."""
    print("=" * 80)
    print("  Testing SubAgent Event Types")
    print("=" * 80)

    config_path = find_config_path()
    agent = Agent.create(config_path)
    session = agent.session(".", permissive=True)

    print("\nExecuting task with SubAgent...")
    print("-" * 80)

    # Track events
    events = []
    unknown_count = 0
    subagent_start_count = 0
    subagent_end_count = 0

    # Use task tool directly with a simple query
    stream = session.stream(
        "List all Python files in the examples directory using the glob tool."
    )

    for event in stream:
        event_type = event.event_type
        events.append(event_type)

        if event_type == "unknown":
            unknown_count += 1
        elif event_type == "subagent_start":
            subagent_start_count += 1
            print(f"  [PASS] SubagentStart event recognized")
        elif event_type == "subagent_end":
            subagent_end_count += 1
            print(f"  [PASS] SubagentEnd event recognized")
        elif event_type in ["tool_start", "tool_end"]:
            tool_name = getattr(event, 'tool_name', 'unknown')
            print(f"  Event: {event_type} ({tool_name})")

    print("\n" + "=" * 80)
    print("  Results")
    print("=" * 80)
    print(f"  Total events: {len(events)}")
    print(f"  Unknown events: {unknown_count}")
    print(f"  SubagentStart events: {subagent_start_count}")
    print(f"  SubagentEnd events: {subagent_end_count}")

    print("\n  Event type distribution:")
    for event_type in sorted(set(events)):
        count = events.count(event_type)
        print(f"    - {event_type}: {count}")

    print("\n" + "=" * 80)
    if unknown_count == 0:
        print("  [PASS] All events recognized!")
    else:
        print(f"  [FAIL] {unknown_count} unknown events found")
    print("=" * 80)


if __name__ == "__main__":
    test_subagent_events()
