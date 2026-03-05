#!/usr/bin/env python3
"""
Test SubAgent event streaming by explicitly using task tool.
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
    """Test SubAgent event streaming."""
    print("=" * 80)
    print("  Testing SubAgent Event Streaming")
    print("=" * 80)

    config_path = find_config_path()
    agent = Agent.create(config_path)
    session = agent.session(".", permissive=True)

    print("\nSpawning SubAgent with task tool...")
    print("-" * 80)

    # Track events
    events = []
    subagent_events = []
    tool_events_in_subagent = []

    # Explicitly use task tool
    prompt = """Use the task tool to spawn a general agent with this exact configuration:
    - agent: "general"
    - description: "Count files"
    - prompt: "Use glob tool to find all .py files in current directory and count them"
    - permissive: true
    - max_steps: 3

    After spawning, wait for the result."""

    stream = session.stream(prompt)

    for event in stream:
        event_type = event.event_type
        events.append(event_type)

        if event_type == "subagent_start":
            subagent_events.append(event)
            agent_name = getattr(event, 'tool_name', 'unknown')
            task_id = getattr(event, 'tool_id', 'unknown')
            print(f"  [SubAgent] START - agent={agent_name}, task_id={task_id}")
        elif event_type == "subagent_end":
            subagent_events.append(event)
            success = getattr(event, 'exit_code', -1) == 0
            print(f"  [SubAgent] END - success={success}")
        elif event_type == "tool_start":
            tool_name = getattr(event, 'tool_name', 'unknown')
            if tool_name not in ["task", "parallel_task"]:
                tool_events_in_subagent.append(event)
                print(f"  [Tool] {tool_name} (from SubAgent)")
            else:
                print(f"  [Tool] {tool_name} (parent)")
        elif event_type == "unknown":
            print(f"  [UNKNOWN EVENT]")

    print("\n" + "=" * 80)
    print("  Results")
    print("=" * 80)
    print(f"  Total events: {len(events)}")
    print(f"  SubAgent lifecycle events: {len(subagent_events)}")
    print(f"  Tool calls from SubAgent: {len(tool_events_in_subagent)}")
    print(f"  Unknown events: {events.count('unknown')}")

    print("\n  Event type distribution:")
    for event_type in sorted(set(events)):
        count = events.count(event_type)
        print(f"    - {event_type}: {count}")

    print("\n" + "=" * 80)
    if len(subagent_events) >= 2:
        print("  [PASS] SubAgent lifecycle events detected")
    else:
        print("  [FAIL] SubAgent lifecycle events missing")

    if tool_events_in_subagent:
        print(f"  [PASS] SubAgent internal tool calls visible ({len(tool_events_in_subagent)} found)")
    else:
        print("  [INFO] No internal tool calls detected from SubAgent")

    if events.count('unknown') == 0:
        print("  [PASS] All event types recognized")
    else:
        print(f"  [INFO] {events.count('unknown')} unknown events")
    print("=" * 80)


if __name__ == "__main__":
    test_subagent_events()
