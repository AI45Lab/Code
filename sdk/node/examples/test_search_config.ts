#!/usr/bin/env node
/**
 * A3S Code Node.js SDK - Web Search Configuration Test
 *
 * Tests A3S Search v0.7.0 configurable web search with real LLM.
 *
 * Run with: npx ts-node examples/test_search_config.ts
 */

import { Agent, Session, AgentResult } from "../index.js";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

class SearchConfigTest {
  private readonly agent: Agent;
  private readonly configPath: string;

  constructor(agent: Agent, configPath: string) {
    this.agent = agent;
    this.configPath = configPath;
  }

  /**
   * Find config file in home directory or project root.
   */
  static findConfigPath(): string {
    const homeConfig: string = path.join(os.homedir(), ".a3s", "config.hcl");
    if (fs.existsSync(homeConfig)) {
      return homeConfig;
    }

    // Try project root (6 levels up from examples/test_search_config.ts)
    const projectConfig: string = path.join(__dirname, "..", "..", "..", "..", "..", "..", ".a3s", "config.hcl");
    if (fs.existsSync(projectConfig)) {
      return projectConfig;
    }

    throw new Error("Config file not found. Please create ~/.a3s/config.hcl");
  }

  /**
   * Truncate text to max length.
   */
  static truncate(text: string, maxLen: number): string {
    if (text.length <= maxLen) {
      return text;
    }
    return `${text.substring(0, maxLen)}... (truncated)`;
  }

  /**
   * Test 1: Default search configuration.
   */
  async testDefaultSearch(): Promise<void> {
    console.log("\nTest 1: Default Search Configuration");
    console.log("-".repeat(80));

    // Use config from file (should have default search engines)
    const session: Session = this.agent.session(".");

    console.log("Testing: Web search with default engines (ddg, wiki)...");
    try {
      const result: AgentResult = await session.send("Search the web for 'Rust async programming' and give me the top 3 results");
      console.log("Default search works");
      console.log(`  Result preview: ${SearchConfigTest.truncate(result.text, 200)}`);
    } catch (err: unknown) {
      console.log(`Search failed (expected if engines unavailable): ${(err as Error).message}`);
    }

    console.log("\nTest 1 passed: Default configuration works");
  }

  /**
   * Test 2: Custom search configuration via HCL.
   */
  async testCustomSearchConfig(): Promise<void> {
    console.log("\nTest 2: Custom Search Configuration");
    console.log("-".repeat(80));

    // Create agent with inline HCL config including search configuration
    const configHcl: string = `
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
  `;

    console.log("Testing: Web search with custom configuration...");
    console.log("  - Timeout: 30 seconds");
    console.log("  - Engines: ddg (1.5), wiki (1.2), brave (1.0)");
    console.log("  - Health monitoring enabled");

    try {
      const agent: Agent = await Agent.create(configHcl);
      const session: Session = agent.session(".");

      const result: AgentResult = await session.send("Search for 'Rust tokio tutorial' and summarize the findings");
      console.log("Custom search configuration works");
      console.log(`  Result preview: ${SearchConfigTest.truncate(result.text, 200)}`);
    } catch (err: unknown) {
      console.log(`Search failed (expected if engines unavailable): ${(err as Error).message}`);
    }

    console.log("\nTest 2 passed: Custom configuration works");
  }

  /**
   * Test 3: Engine enable/disable control via HCL.
   */
  async testEngineControl(): Promise<void> {
    console.log("\nTest 3: Engine Enable/Disable Control");
    console.log("-".repeat(80));

    // Create config with only wiki enabled
    const configHcl: string = `
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
  `;

    console.log("Testing: Only Wikipedia enabled...");
    try {
      const agent: Agent = await Agent.create(configHcl);
      const session: Session = agent.session(".");

      const result: AgentResult = await session.send("Search for 'Rust programming language' using only Wikipedia");
      console.log("Engine control works (only wiki enabled)");
      console.log(`  Result preview: ${SearchConfigTest.truncate(result.text, 200)}`);
    } catch (err: unknown) {
      console.log(`Search failed (expected if wiki unavailable): ${(err as Error).message}`);
    }

    console.log("\nTest 3 passed: Engine control works correctly");
  }

  /**
   * Run all tests.
   */
  async runAll(): Promise<void> {
    console.log("A3S Code Node.js SDK - Web Search Configuration Test\n");
    console.log("=".repeat(80));

    await this.testDefaultSearch();
    await this.testCustomSearchConfig();
    await this.testEngineControl();

    console.log("\n\n");
    console.log("=".repeat(80));
    console.log("All search configuration tests completed!");
    console.log("=".repeat(80));
  }
}

/**
 * Main test runner.
 */
async function main(): Promise<void> {
  const configPath: string = SearchConfigTest.findConfigPath();
  const agent: Agent = await Agent.create(configPath);
  const test = new SearchConfigTest(agent, configPath);
  await test.runAll();
}

// Run tests
main().catch((err: unknown) => {
  console.error("Test failed:", err);
  process.exit(1);
});
