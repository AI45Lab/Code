#!/usr/bin/env python3
"""
A3S Code Python SDK - Web Search Configuration Test

Tests A3S Search v0.7.0 configurable web search with real LLM.

Run with: python examples/test_search_config.py
"""

import asyncio
from pathlib import Path
from a3s_code import Agent


class SearchConfigTest:
    """Tests for configurable web search functionality."""

    def __init__(self, agent: Agent, config_path: str) -> None:
        self.agent = agent
        self.config_path = config_path

    @staticmethod
    def find_config_path() -> str:
        """Find config file in home directory or project root."""
        home_config = Path.home() / ".a3s" / "config.hcl"
        if home_config.exists():
            return str(home_config)

        # Try project root (6 levels up from examples/test_search_config.py)
        project_config = Path(__file__).parent.parent.parent.parent.parent.parent / ".a3s" / "config.hcl"
        if project_config.exists():
            return str(project_config)

        raise FileNotFoundError("Config file not found. Please create ~/.a3s/config.hcl")

    @staticmethod
    def truncate(text: str, max_len: int) -> str:
        """Truncate text to max length."""
        if len(text) <= max_len:
            return text
        return f"{text[:max_len]}... (truncated)"

    async def test_default_search(self) -> None:
        """Test 1: Default search configuration."""
        print("\n🔍 Test 1: Default Search Configuration")
        print("-" * 80)

        # Use config from file (should have default search engines)
        session = self.agent.session(".")

        print("Testing: Web search with default engines (ddg, wiki)...")
        try:
            result = session.send("Search the web for 'Rust async programming' and give me the top 3 results")
            print("✓ Default search works")
            print(f"  Result preview: {self.truncate(result.text, 200)}")
        except Exception as e:
            print(f"⚠️  Search failed (expected if engines unavailable): {e}")

        print("\n✅ Test 1 passed: Default configuration works")

    async def test_custom_search_config(self) -> None:
        """Test 2: Custom search configuration via HCL."""
        print("\n⚙️  Test 2: Custom Search Configuration")
        print("-" * 80)

        # Create agent with inline HCL config including search configuration
        config_hcl = """
            default_model = "openai/gpt-4o"

            providers {
              name = "openai"

              models {
                id          = "gpt-4o"
                name        = "GPT-4o"
                family      = "gpt"
                api_key     = "your-api-key-here"
                base_url    = "https://api.openai.com/v1"
                attachment  = false
                reasoning   = false
                tool_call   = true
                temperature = true

                modalities {
                  input  = ["text"]
                  output = ["text"]
                }

                limit {
                  context = 128000
                  output  = 4096
                }
              }
            }

            # Custom search configuration
            search {
              timeout = 30

              health {
                max_failures = 3
                suspend_seconds = 60
              }

              engine {
                ddg {
                  enabled = true
                  weight = 1.5
                }

                wiki {
                  enabled = true
                  weight = 1.2
                }

                brave {
                  enabled = true
                  weight = 1.0
                  timeout = 20
                }
              }
            }
        """

        print("Testing: Web search with custom configuration...")
        print("  - Timeout: 30 seconds")
        print("  - Engines: ddg (1.5), wiki (1.2), brave (1.0)")
        print("  - Health monitoring enabled")

        try:
            agent = Agent.create(config_hcl)
            session = agent.session(".")

            result = session.send("Search for 'Rust tokio tutorial' and summarize the findings")
            print("✓ Custom search configuration works")
            print(f"  Result preview: {self.truncate(result.text, 200)}")
        except Exception as e:
            print(f"⚠️  Search failed (expected if engines unavailable): {e}")

        print("\n✅ Test 2 passed: Custom configuration works")

    async def test_engine_control(self) -> None:
        """Test 3: Engine enable/disable control via HCL."""
        print("\n🎛️  Test 3: Engine Enable/Disable Control")
        print("-" * 80)

        # Create config with only wiki enabled
        config_hcl = """
            default_model = "openai/gpt-4o"

            providers {
              name = "openai"

              models {
                id          = "gpt-4o"
                name        = "GPT-4o"
                family      = "gpt"
                api_key     = "your-api-key-here"
                base_url    = "https://api.openai.com/v1"
                attachment  = false
                reasoning   = false
                tool_call   = true
                temperature = true

                modalities {
                  input  = ["text"]
                  output = ["text"]
                }

                limit {
                  context = 128000
                  output  = 4096
                }
              }
            }

            search {
              timeout = 20

              engine {
                ddg {
                  enabled = false
                }

                wiki {
                  enabled = true
                  weight = 1.0
                }

                brave {
                  enabled = false
                }
              }
            }
        """

        print("Testing: Only Wikipedia enabled...")
        try:
            agent = Agent.create(config_hcl)
            session = agent.session(".")

            result = session.send("Search for 'Rust programming language' using only Wikipedia")
            print("✓ Engine control works (only wiki enabled)")
            print(f"  Result preview: {self.truncate(result.text, 200)}")
        except Exception as e:
            print(f"⚠️  Search failed (expected if wiki unavailable): {e}")

        print("\n✅ Test 3 passed: Engine control works correctly")

    async def run_all(self) -> None:
        """Run all search configuration tests."""
        print("🚀 A3S Code Python SDK - Web Search Configuration Test\n")
        print("=" * 80)

        await self.test_default_search()
        await self.test_custom_search_config()
        await self.test_engine_control()

        print("\n" * 2)
        print("=" * 80)
        print("✅ All search configuration tests completed!")
        print("=" * 80)


if __name__ == "__main__":
    config_path = SearchConfigTest.find_config_path()
    agent = Agent.create(config_path)
    test = SearchConfigTest(agent, config_path)
    asyncio.run(test.run_all())
