#!/usr/bin/env python3
"""
Test SubAgent Event Streaming

This test verifies whether SubAgent internal events (tool calls, LLM responses)
are propagated to the parent session's event stream.

Current behavior: Only SubagentStart and SubagentEnd events are emitted.
Expected behavior: All internal events should be visible to parent session.
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


def test_subagent_event_streaming():
    """Test if SubAgent events are visible in parent session stream."""
    print("=" * 80)
    print("  Testing SubAgent Event Streaming")
    print("=" * 80)

    config_path = find_config_path()
    agent = Agent.create(config_path)
    session = agent.session(".", permissive=True)

    print("\nSpawning SubAgent and monitoring events...")
    print("-" * 80)

    # Track events
    events_received = []
    subagent_internal_events = []

    # Stream the task execution
    stream = session.stream(
        "Use the task tool to spawn an explore agent. "
        "Ask it to count Python files in the current directory. "
        "Set permissive=True and max_steps=5."
    )

    for event in stream:
        event_type = event.event_type
        events_received.append(event_type)

        print(f"  Event: {event_type}")

        # Check for SubAgent-specific events
        if event_type == "subagent_start":
            agent_name = getattr(event, 'agent', 'unknown')
            print(f"    -> SubAgent started: {agent_name}")
        elif event_type == "subagent_end":
            success = getattr(event, 'success', False)
            print(f"    -> SubAgent ended: success={success}")
        elif event_type == "tool_start":
            tool_name = getattr(event, 'tool_name', 'unknown')
            print(f"    -> Tool call: {tool_name}")
            # Check if this is a tool call from within the SubAgent
            if tool_name not in ["task", "parallel_task"]:
                subagent_internal_events.append(event)
        elif event_type == "tool_end":
            print(f"    -> Tool result")
        elif event_type == "text_delta":
            text = getattr(event, 'text', '')
            if text.strip():
                print(f"    -> Text: {text[:50]}...")

    print("\n" + "=" * 80)
    print("  Event Summary")
    print("=" * 80)
    print(f"  Total events received: {len(events_received)}")
    print(f"  SubAgent internal events: {len(subagent_internal_events)}")

    print("\n  Event types:")
    for event_type in set(events_received):
        count = events_received.count(event_type)
        print(f"    - {event_type}: {count}")

    print("\n" + "=" * 80)
    if subagent_internal_events:
        print("  [SUCCESS] SubAgent internal events ARE visible!")
        print(f"  Found {len(subagent_internal_events)} internal tool calls from SubAgent")
    else:
        print("  [ISSUE] SubAgent internal events are NOT visible")
        print("  Only SubagentStart and SubagentEnd events are emitted")
        print("  Internal tool calls, LLM responses are hidden from parent session")
    print("=" * 80)


if __name__ == "__main__":
    test_subagent_event_streaming()
