#!/usr/bin/env node
/**
 * A3S Code Node.js SDK - Integration Tests
 *
 * Tests all recently added/fixed features against the real kimi endpoint:
 *   1. task tool (delegate to general subagent, wait for result)
 *   2. parallel_task (fan-out to multiple subagents concurrently)
 *   3. toolNames() initial state on a fresh session
 *   4. mcpStatus error field populated on failed connect
 *   5. MCP injection: add -> status -> LLM use -> toolNames -> remove
 *   6. refreshMcpTools (smoke test)
 *
 * Run with: npx ts-node examples/test_mcp_servers.ts
 * Requires: ~/.a3s/config.hcl or repo .a3s/config.hcl
 */

import { Agent, Session, AgentResult } from "../index.js";
import * as crypto from "crypto";
import * as os from "os";
import * as path from "path";
import * as fs from "fs";

// Bundled minimal MCP echo server -- no external dependencies required
const ECHO_SERVER: string = path.join(__dirname, "mcp_echo_server.js");

class McpServersTest {
  private readonly agent: Agent;
  private readonly configPath: string;

  constructor(agent: Agent, configPath: string) {
    this.agent = agent;
    this.configPath = configPath;
  }

  static resolveConfig(): string {
    const homeConfig: string = path.join(os.homedir(), ".a3s", "config.hcl");
    if (fs.existsSync(homeConfig)) return homeConfig;
    let dir: string = __dirname;
    for (let i = 0; i < 10; i++) {
      const candidate: string = path.join(dir, ".a3s", "config.hcl");
      if (fs.existsSync(candidate)) return candidate;
      const parent: string = path.dirname(dir);
      if (parent === dir) break;
      dir = parent;
    }
    throw new Error("Config not found at .a3s/config.hcl or ~/.a3s/config.hcl");
  }

  static pass(label: string): void {
    console.log(`  ✓  ${label}`);
  }

  // -- Test 1: task tool ------------------------------------------------------------

  async testTaskTool(tmpdir: string): Promise<void> {
    console.log("\n-- Test 1: task tool (subagent delegation) --");
    const session: Session = this.agent.session(tmpdir, { permissive: true });
    const result: AgentResult = await session.send(
      "Use the task tool to delegate the following to the 'general' agent, " +
      "then return its exact reply verbatim: " +
      "'Reply with exactly the word: TASK_OK'"
    );
    if (!result.text.includes("TASK_OK")) {
      throw new Error(`task tool should return TASK_OK, got: ${result.text}`);
    }
    McpServersTest.pass("task tool delegated and returned TASK_OK");
  }

  // -- Test 2: parallel_task --------------------------------------------------------

  async testParallelTask(tmpdir: string): Promise<void> {
    console.log("\n-- Test 2: parallel_task (concurrent fan-out) --");
    const session: Session = this.agent.session(tmpdir, { permissive: true });
    const result: AgentResult = await session.send(
      "Use the parallel_task tool to run these three tasks concurrently " +
      "using the 'general' agent, then list their outputs:\n" +
      "1. Reply with exactly: PARALLEL_A\n" +
      "2. Reply with exactly: PARALLEL_B\n" +
      "3. Reply with exactly: PARALLEL_C"
    );
    for (const token of ["PARALLEL_A", "PARALLEL_B", "PARALLEL_C"]) {
      if (!result.text.includes(token)) {
        throw new Error(`expected ${token} in result, got: ${result.text}`);
      }
    }
    McpServersTest.pass("parallel_task ran 3 subagents concurrently, all results returned");
  }

  // -- Test 3: toolNames initial state ----------------------------------------------

  async testToolNames(tmpdir: string): Promise<void> {
    console.log("\n-- Test 3: toolNames() initial state --");
    const session: Session = this.agent.session(tmpdir, { permissive: true });

    const names: string[] = session.toolNames();
    if (names.length === 0) {
      throw new Error("expected built-in tools to be present");
    }
    McpServersTest.pass(`toolNames() returns ${names.length} built-in tools`);

    const mcpNames: string[] = names.filter((n: string) => n.startsWith("mcp__"));
    if (mcpNames.length > 0) {
      throw new Error(
        `expected no mcp__ tools on a fresh session (none configured globally), got: ${mcpNames}`
      );
    }
    McpServersTest.pass("no mcp__ tools on fresh session (none configured globally)");
  }

  // -- Test 4: mcpStatus error field ------------------------------------------------

  async testMcpStatusError(tmpdir: string): Promise<void> {
    console.log("\n-- Test 4: mcpStatus error field on failed connect --");
    const session: Session = this.agent.session(tmpdir, { permissive: true });

    let err: unknown = null;
    try {
      await session.addMcpServer(
        "bad-server", "stdio", "nonexistent-mcp-binary-xyz", [], undefined, undefined, undefined
      );
    } catch (e: unknown) {
      err = e;
    }
    if (!err) throw new Error("addMcpServer must throw for a nonexistent binary");

    // The config is registered before connection is attempted, so the server
    // must always appear in mcpStatus -- even on failure.
    const status: Array<{ name: string; connected: boolean; toolCount: number; error?: string }> =
      await session.mcpStatus();
    const badEntry = status.find((s) => s.name === "bad-server");
    if (!badEntry) {
      throw new Error(
        `bad-server must appear in mcpStatus after failed connect, ` +
        `got names: ${status.map((s) => s.name)}`
      );
    }
    if (badEntry.connected) {
      throw new Error("bad-server should not be connected");
    }
    if (!badEntry.error) {
      throw new Error(`error field must be populated after failed connect, got: ${JSON.stringify(badEntry)}`);
    }
    McpServersTest.pass(`mcpStatus.error captured: ${badEntry.error}`);
  }

  // -- Test 5: MCP injection --------------------------------------------------------

  async testMcpInjection(tmpdir: string): Promise<void> {
    console.log("\n-- Test 5: MCP injection (add -> status -> LLM use -> toolNames -> remove) --");
    const session: Session = this.agent.session(tmpdir, { permissive: true });

    // Before add: no mcp__echo__ tools
    const toolsBefore: string[] = session.toolNames();
    const echoToolsBefore: string[] = toolsBefore.filter((n: string) => n.startsWith("mcp__echo__"));
    if (echoToolsBefore.length > 0) {
      throw new Error(`expected no mcp__echo__ tools before addMcpServer, got: ${echoToolsBefore}`);
    }
    McpServersTest.pass("no mcp__echo__ tools before addMcpServer");

    // Generate a random secret unknown to the LLM -- it can only be obtained by
    // actually calling mcp__echo__get_secret, preventing the LLM from faking it.
    const secret: string = crypto.randomBytes(8).toString("hex");

    // Add the bundled echo server (uses the same Node.js binary -- no external deps)
    const count: number = await session.addMcpServer(
      "echo", "stdio",
      process.execPath, [ECHO_SERVER, secret],
      undefined, undefined, undefined
    );
    if (count === 0) throw new Error("echo server must expose >= 1 tool");
    McpServersTest.pass(`addMcpServer registered ${count} tools`);

    // Verify mcpStatus: connected=true, toolCount=N, error=null
    const mcpStat: Array<{ name: string; connected: boolean; toolCount: number; error?: string }> =
      await session.mcpStatus();
    const echoStat = mcpStat.find((s) => s.name === "echo");
    if (!echoStat) {
      throw new Error(`echo must appear in mcpStatus, got names: ${mcpStat.map((s) => s.name)}`);
    }
    if (!echoStat.connected) throw new Error("echo server should be connected");
    if (echoStat.toolCount !== count) {
      throw new Error(`mcpStatus.toolCount should be ${count}, got ${echoStat.toolCount}`);
    }
    if (echoStat.error) throw new Error(`no error expected, got: ${echoStat.error}`);
    McpServersTest.pass(`mcpStatus: connected=true, toolCount=${count}, error=null`);

    // Verify toolNames reflects the injected tools
    const toolsAfter: string[] = session.toolNames();
    const mcpTools: string[] = toolsAfter.filter((n: string) => n.startsWith("mcp__echo__"));
    if (mcpTools.length !== count) {
      throw new Error(`toolNames should show ${count} mcp__echo__ tools, got ${mcpTools.length}`);
    }
    McpServersTest.pass(`toolNames shows ${mcpTools.length} mcp__echo__ tools`);

    // LLM must actually call the MCP tool to retrieve the secret.
    // The secret value is NOT in the prompt -- the LLM cannot fake this.
    const mcpResult: AgentResult = await session.send(
      "Use the mcp__echo__get_secret tool to retrieve the secret value, " +
      "then tell me exactly what it returned."
    );
    if (!mcpResult.text.includes(secret)) {
      throw new Error(
        `LLM should have called mcp__echo__get_secret and returned the secret, ` +
        `got: ${mcpResult.text}`
      );
    }
    McpServersTest.pass("LLM used mcp__echo__get_secret tool and returned the correct secret");

    // Remove server: tools disappear
    await session.removeMcpServer("echo");
    const toolsFinal: string[] = session.toolNames();
    if (toolsFinal.some((n: string) => n.startsWith("mcp__echo__"))) {
      throw new Error("mcp__echo__ tools should be gone after removeMcpServer");
    }
    McpServersTest.pass("removeMcpServer removed all mcp__echo__ tools");
  }

  // -- Test 6: refreshMcpTools ------------------------------------------------------

  async testRefreshMcpTools(): Promise<void> {
    console.log("\n-- Test 6: refreshMcpTools (smoke test) --");
    await this.agent.refreshMcpTools();
    McpServersTest.pass("refreshMcpTools completed without error");
  }

  // -- Run all tests ----------------------------------------------------------------

  async runAll(): Promise<void> {
    console.log("=== A3S Code Node.js SDK - Integration Tests ===");
    console.log(`config: ${this.configPath}\n`);

    const tmpdir: string = fs.mkdtempSync(path.join(os.tmpdir(), "a3s-test-"));

    try {
      const tests: Array<() => Promise<void>> = [
        () => this.testTaskTool(tmpdir),
        () => this.testParallelTask(tmpdir),
        () => this.testToolNames(tmpdir),
        () => this.testMcpStatusError(tmpdir),
        () => this.testMcpInjection(tmpdir),
        () => this.testRefreshMcpTools(),
      ];

      let passed = 0;
      let failed = 0;

      for (const test of tests) {
        try {
          await test();
          passed++;
        } catch (err: unknown) {
          console.error(`\n  FAILED: ${(err as Error).message}`);
          failed++;
        }
      }

      console.log("\n" + "=".repeat(48));
      if (failed === 0) {
        console.log(`=== all ${passed} tests passed ===`);
      } else {
        console.log(`=== ${passed} passed, ${failed} failed ===`);
        process.exit(1);
      }
      console.log("=".repeat(48));
    } finally {
      fs.rmSync(tmpdir, { recursive: true, force: true });
    }
  }
}

async function main(): Promise<void> {
  const configPath: string = McpServersTest.resolveConfig();
  const agent: Agent = await Agent.create(configPath);
  const test = new McpServersTest(agent, configPath);
  await test.runAll();
}

main().catch((err: unknown) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
