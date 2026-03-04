/**
 * System Prompt Slots -- Customizing agent personality without overriding core behavior
 *
 * Demonstrates the slot-based system prompt customization API:
 * 1. Custom role (persona)
 * 2. Custom guidelines (coding standards)
 * 3. Custom response style
 * 4. Extra freeform instructions
 * 5. Verify core tool behavior is preserved
 *
 * Run with: npx ts-node examples/test_prompt_slots.ts
 */

import { Agent, Session, AgentResult } from "../index.js";
import * as path from "path";
import * as os from "os";
import * as fs from "fs";

class PromptSlotsTest {
  private readonly agent: Agent;
  private readonly configPath: string;

  constructor(agent: Agent, configPath: string) {
    this.agent = agent;
    this.configPath = configPath;
  }

  static findConfig(): string {
    if (process.env.A3S_CONFIG) return process.env.A3S_CONFIG;
    const homeConfig: string = path.join(os.homedir(), ".a3s", "config.hcl");
    if (fs.existsSync(homeConfig)) return homeConfig;
    throw new Error("Config not found. Create ~/.a3s/config.hcl or set A3S_CONFIG");
  }

  static assert(condition: boolean, message: string): void {
    if (!condition) throw new Error(`Assertion failed: ${message}`);
  }

  async runAll(): Promise<void> {
    console.log(`Config: ${this.configPath}`);
    console.log("Agent created\n");

    const workspace: string = fs.mkdtempSync(path.join(os.tmpdir(), "a3s-slots-"));
    try {
      // --- Test 1: Custom role only ---
      console.log("=== Test 1: Custom role ===");
      let session: Session = this.agent.session(workspace, {
        role: "You are a senior Rust developer who specializes in async programming.",
      });
      let result: AgentResult = await session.send("What is your area of expertise? Reply in one sentence.");
      console.log(`Response: ${result.text.trim()}\n`);
      PromptSlotsTest.assert(!!result.text, "should get a response");

      // --- Test 2: Role + guidelines + response style ---
      console.log("=== Test 2: Role + guidelines + response style ===");
      session = this.agent.session(workspace, {
        role: "You are a Python code reviewer.",
        guidelines: "Always check for type hints. Flag any use of `eval()`.",
        responseStyle: "Reply in bullet points. Be concise.",
      });

      // Write a Python file for the agent to review
      fs.writeFileSync(
        path.join(workspace, "app.py"),
        'def add(a, b):\n' +
        '    return eval(f"{a} + {b}")\n' +
        '\n' +
        'def greet(name):\n' +
        '    return "Hello " + name\n'
      );

      result = await session.send("Review the file app.py and list any issues you find.");
      console.log(`Response:\n${result.text.trim()}\n`);
      PromptSlotsTest.assert(result.toolCallsCount > 0, "should have used read_file tool");

      // --- Test 3: Extra instructions only ---
      console.log("=== Test 3: Extra instructions ===");
      session = this.agent.session(workspace, {
        extra: "Always end your response with '-- A3S'",
      });
      result = await session.send("Say hello.");
      console.log(`Response: ${result.text.trim()}\n`);

      // --- Test 4: Verify tools still work (core behavior preserved) ---
      console.log("=== Test 4: Core tool behavior preserved ===");
      session = this.agent.session(workspace, {
        role: "You are a minimalist file manager.",
        guidelines: "Only create files when explicitly asked.",
      });
      result = await session.send(
        "Create a file called test.txt with the content 'prompt slots work'. " +
        "Then read it back."
      );
      console.log(`Response: ${result.text.trim()}\n`);
      PromptSlotsTest.assert(result.toolCallsCount > 0, "should have used tools");
      PromptSlotsTest.assert(fs.existsSync(path.join(workspace, "test.txt")), "test.txt should exist");

      console.log("=== All prompt slots tests passed ===");
    } finally {
      fs.rmSync(workspace, { recursive: true, force: true });
    }
  }
}

async function main(): Promise<void> {
  const configPath: string = PromptSlotsTest.findConfig();
  const agent: Agent = await Agent.create(configPath);
  const test = new PromptSlotsTest(agent, configPath);
  await test.runAll();
}

main().catch((err: unknown) => {
  console.error("Test failed:", (err as Error).message);
  process.exit(1);
});
