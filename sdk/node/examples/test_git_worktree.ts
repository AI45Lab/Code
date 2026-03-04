/**
 * Git Worktree Tool Test with Real LLM
 *
 * Demonstrates the git_worktree builtin tool via the Node.js SDK:
 * 1. Initialize a git repo in a temp directory
 * 2. Direct tool calls: status, create, list, remove
 * 3. LLM-driven: ask the agent to use git_worktree
 *
 * Run with: npx ts-node examples/test_git_worktree.ts
 */

import { Agent, Session, AgentResult, ToolResult } from "../index.js";
import * as path from "path";
import * as os from "os";
import * as fs from "fs";
import { execSync } from "child_process";

class GitWorktreeTest {
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

  static initGitRepo(dir: string): void {
    execSync("git init", { cwd: dir, stdio: "pipe" });
    execSync('git config user.email "test@example.com"', { cwd: dir, stdio: "pipe" });
    execSync('git config user.name "Test User"', { cwd: dir, stdio: "pipe" });
    fs.writeFileSync(path.join(dir, "README.md"), "# Test Repo\n");
    execSync("git add .", { cwd: dir, stdio: "pipe" });
    execSync('git commit -m "Initial commit"', { cwd: dir, stdio: "pipe" });
  }

  static assert(condition: boolean, message: string): void {
    if (!condition) throw new Error(`Assertion failed: ${message}`);
  }

  async runAll(): Promise<void> {
    console.log(`Config: ${this.configPath}`);
    console.log("Agent created");

    // Create temp workspace
    const workspace: string = fs.mkdtempSync(path.join(os.tmpdir(), "a3s-wt-"));
    try {
      GitWorktreeTest.initGitRepo(workspace);
      console.log(`Git repo initialized at: ${workspace}\n`);

      const session: Session = this.agent.session(workspace);

      // --- Test 1: Direct tool call -- status ---
      console.log("=== Test 1: git_worktree status ===");
      let result: ToolResult = await session.tool("git_worktree", { command: "status" });
      console.log(result.output);
      GitWorktreeTest.assert(result.exitCode === 0, "status should succeed");
      console.log();

      // --- Test 2: Direct tool call -- create worktree ---
      console.log("=== Test 2: git_worktree create ===");
      const wtPath: string = path.join(workspace, "wt-feature-auth");
      result = await session.tool("git_worktree", {
        command: "create",
        branch: "feature-auth",
        path: wtPath,
      });
      console.log(result.output);
      GitWorktreeTest.assert(result.exitCode === 0, `create failed: ${result.output}`);
      GitWorktreeTest.assert(fs.existsSync(wtPath), "worktree directory should exist");
      console.log();

      // --- Test 3: Direct tool call -- list ---
      console.log("=== Test 3: git_worktree list ===");
      result = await session.tool("git_worktree", { command: "list" });
      console.log(result.output);
      GitWorktreeTest.assert(result.exitCode === 0, "list should succeed");
      GitWorktreeTest.assert(result.output.includes("feature-auth"), "list should contain the new branch");
      console.log();

      // --- Test 4: LLM-driven query ---
      console.log("=== Test 4: LLM-driven worktree query ===");
      const llmResult: AgentResult = await session.send(
        "Use the git_worktree tool with command 'list' to show me all worktrees. " +
        "Just show the tool output, nothing else."
      );
      console.log(`LLM response:\n${llmResult.text}`);
      GitWorktreeTest.assert(llmResult.toolCallsCount > 0, "LLM should have called git_worktree");
      console.log();

      // --- Test 5: Direct tool call -- remove ---
      console.log("=== Test 5: git_worktree remove ===");
      result = await session.tool("git_worktree", {
        command: "remove",
        path: wtPath,
      });
      console.log(result.output);
      GitWorktreeTest.assert(result.exitCode === 0, `remove failed: ${result.output}`);
      GitWorktreeTest.assert(!fs.existsSync(wtPath), "worktree directory should be gone");
      console.log();

      // --- Test 6: Verify cleanup ---
      console.log("=== Test 6: Verify cleanup ===");
      result = await session.tool("git_worktree", { command: "list" });
      console.log(result.output);
      GitWorktreeTest.assert(!result.output.includes("feature-auth"), "feature-auth should be removed");
      console.log();

      console.log("=== All git_worktree tests passed ===");
    } finally {
      // Cleanup
      fs.rmSync(workspace, { recursive: true, force: true });
    }
  }
}

async function main(): Promise<void> {
  const configPath: string = GitWorktreeTest.findConfig();
  const agent: Agent = await Agent.create(configPath);
  const test = new GitWorktreeTest(agent, configPath);
  await test.runAll();
}

main().catch((err: unknown) => {
  console.error("Test failed:", (err as Error).message);
  process.exit(1);
});
