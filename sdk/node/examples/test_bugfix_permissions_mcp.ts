#!/usr/bin/env npx tsx
/**
 * A3S Code Node.js SDK — Bug Fix Verification Tests (TypeScript)
 *
 * Tests for two framework-level bugs fixed in a3s-code-core v1.0.1:
 *   Bug 1: permissions.allow non-empty caused agent files to silently fail loading
 *   Bug 2: Sub-agent child sessions had no access to MCP tools
 *
 * Prerequisites:
 *   - Rebuild Node.js SDK after core fixes: `npx @napi-rs/cli build --platform --release`
 *   - MCP echo server: ../python/examples/mcp_echo_server.py (bundled, no deps)
 *
 * Run with: npx tsx examples/test_bugfix_permissions_mcp.ts
 */

import { Agent, Session, AgentResult } from '../index.js';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import * as crypto from 'crypto';
import { execSync } from 'child_process';

const ECHO_SERVER: string = path.join(
  __dirname, '..', '..', 'python', 'examples', 'mcp_echo_server.py'
);

class BugfixPermissionsMcpTest {
  private readonly agent: Agent;
  private readonly configPath: string;
  private readonly tmpdir: string;

  constructor(agent: Agent, configPath: string, tmpdir: string) {
    this.agent = agent;
    this.configPath = configPath;
    this.tmpdir = tmpdir;
  }

  // ── Static Helpers ───────────────────────────────────────────────────────

  static findConfig(): string {
    const homeConfig = path.join(os.homedir(), '.a3s', 'config.hcl');
    if (fs.existsSync(homeConfig)) return homeConfig;

    let p = path.resolve(__dirname);
    for (let i = 0; i < 10; i++) {
      const candidate = path.join(p, '.a3s', 'config.hcl');
      if (fs.existsSync(candidate)) return candidate;
      const parent = path.dirname(p);
      if (parent === p) break;
      p = parent;
    }
    throw new Error('Config not found');
  }

  static pass(label: string): void {
    console.log(`  PASS  ${label}`);
  }

  static findPython(): string {
    const candidates = ['python', 'python3', 'python3.14', 'python3.13'];
    for (const cmd of candidates) {
      try {
        execSync(`${cmd} --version`, { stdio: 'pipe' });
        return cmd;
      } catch {
        /* try next */
      }
    }
    throw new Error('Python not found (needed for MCP echo server)');
  }

  // ── Bug 1: Agent files with non-empty permissions.allow ─────────────────

  async testBug1(): Promise<void> {
    console.log('\n-- Bug 1: Agent loading with non-empty permissions.allow --');

    const agentsDir = path.join(this.tmpdir, 'agents');
    fs.mkdirSync(agentsDir, { recursive: true });

    // YAML agent with plain string permissions (previously FAILED to load)
    fs.writeFileSync(
      path.join(agentsDir, 'scorer.yaml'),
      `
name: scorer
description: Scoring agent with permission restrictions
mode: subagent
max_steps: 10
permissions:
  allow:
    - read
    - grep
    - glob
  deny:
    - write
    - edit
`
    );

    // Markdown agent with permissions (previously FAILED to load)
    fs.writeFileSync(
      path.join(agentsDir, 'reviewer.md'),
      `---
name: reviewer
description: Code reviewer with restricted permissions
mode: subagent
max_steps: 10
permissions:
  allow:
    - read
    - grep
  deny:
    - write
    - edit
    - bash
---
You are a code reviewer. Read files and provide feedback. Do NOT modify any files.
`
    );

    const session: Session = this.agent.session(this.tmpdir, {
      agentDirs: [agentsDir],
      permissive: true,
    });

    // Test 1a: scorer agent
    const result1: AgentResult = await session.send(
      "Use the task tool to delegate to the 'scorer' agent with this prompt: " +
      "'Reply with exactly: SCORER_OK'. Return the scorer's reply verbatim."
    );
    if (!result1.text.includes('SCORER_OK')) {
      throw new Error(
        `scorer should return SCORER_OK, got: ${result1.text.substring(0, 200)}`
      );
    }
    BugfixPermissionsMcpTest.pass('scorer agent loaded and executed (YAML with permissions.allow)');

    // Test 1b: reviewer agent
    const result2: AgentResult = await session.send(
      "Use the task tool to delegate to the 'reviewer' agent with this prompt: " +
      "'Reply with exactly: REVIEWER_OK'. Return the reviewer's reply verbatim."
    );
    if (!result2.text.includes('REVIEWER_OK')) {
      throw new Error(
        `reviewer should return REVIEWER_OK, got: ${result2.text.substring(0, 200)}`
      );
    }
    BugfixPermissionsMcpTest.pass('reviewer agent loaded and executed (MD with permissions.allow)');
  }

  // ── Bug 2: Sub-agent MCP tool access ──────────────────────────────────────

  async testBug2(): Promise<void> {
    console.log('\n-- Bug 2: Sub-agent MCP tool access --');

    const agentsDir = path.join(this.tmpdir, 'mcp_agents');
    fs.mkdirSync(agentsDir, { recursive: true });

    fs.writeFileSync(
      path.join(agentsDir, 'mcp-tester.yaml'),
      `
name: mcp-tester
description: Agent that tests MCP tool access
mode: subagent
max_steps: 10
permissions:
  allow:
    - mcp__echo
    - read
`
    );

    const secret: string = crypto.randomBytes(8).toString('hex');
    const python: string = BugfixPermissionsMcpTest.findPython();

    const session: Session = this.agent.session(this.tmpdir, {
      agentDirs: [agentsDir],
      permissive: true,
    });

    // Add MCP echo server
    const count: number = await session.addMcpServer(
      'echo',
      undefined, // transport defaults to stdio
      python,
      [ECHO_SERVER, secret]
    );
    if (count < 1) throw new Error('echo server must expose >= 1 tool');
    BugfixPermissionsMcpTest.pass(`MCP echo server connected with ${count} tools`);

    // Verify parent session has MCP tools
    const parentTools: string[] = session.toolNames();
    const mcpTools: string[] = parentTools.filter((t) => t.startsWith('mcp__echo__'));
    if (mcpTools.length === 0) {
      throw new Error(
        `parent session should have mcp__echo__ tools, got: ${parentTools.join(', ')}`
      );
    }
    BugfixPermissionsMcpTest.pass(`parent session has ${mcpTools.length} mcp__echo__ tools`);

    // Sub-agent must call MCP tool to get secret
    const result: AgentResult = await session.send(
      "Use the task tool to delegate to the 'mcp-tester' agent with this prompt: " +
      "'Use the mcp__echo__get_secret tool to retrieve the secret, " +
      "then reply with exactly the secret value you got.'. " +
      "Return the mcp-tester's reply verbatim."
    );
    if (!result.text.includes(secret)) {
      throw new Error(
        `sub-agent should have called mcp__echo__get_secret and returned '${secret}', ` +
        `got: ${result.text.substring(0, 300)}`
      );
    }
    BugfixPermissionsMcpTest.pass('sub-agent successfully called MCP tool and returned correct secret');

    await session.removeMcpServer('echo');
    BugfixPermissionsMcpTest.pass('MCP server cleaned up');
  }

  // ── Combined: Bug 1 + Bug 2 ──────────────────────────────────────────────

  async testCombined(): Promise<void> {
    console.log('\n-- Combined: permissions.allow + MCP tool access --');

    const agentsDir = path.join(this.tmpdir, 'combined_agents');
    fs.mkdirSync(agentsDir, { recursive: true });

    fs.writeFileSync(
      path.join(agentsDir, 'video-scorer.yaml'),
      `
name: video-scorer
description: Video scoring agent with MCP access
mode: subagent
max_steps: 15
permissions:
  allow:
    - read
    - grep
    - mcp__echo
  deny:
    - write
    - edit
`
    );

    const secret: string = crypto.randomBytes(8).toString('hex');
    const python: string = BugfixPermissionsMcpTest.findPython();

    const session: Session = this.agent.session(this.tmpdir, {
      agentDirs: [agentsDir],
      permissive: true,
    });
    await session.addMcpServer('echo', undefined, python, [ECHO_SERVER, secret]);

    const result: AgentResult = await session.send(
      "Use the task tool to delegate to the 'video-scorer' agent with this prompt: " +
      "'Call the mcp__echo__get_secret tool and return the secret value.'. " +
      "Return the video-scorer's reply verbatim."
    );
    if (!result.text.includes(secret)) {
      throw new Error(
        `video-scorer should load (Bug 1) AND access MCP tools (Bug 2), ` +
        `expected '${secret}', got: ${result.text.substring(0, 300)}`
      );
    }
    BugfixPermissionsMcpTest.pass('Agent with permissions.allow loaded AND accessed MCP tools');

    await session.removeMcpServer('echo');
  }

  // ── Run All ───────────────────────────────────────────────────────────────

  async runAll(): Promise<void> {
    console.log('=== A3S Code Node.js SDK - Bug Fix Verification Tests (TypeScript) ===');
    console.log(`config: ${this.configPath}\n`);

    const tests: Array<() => Promise<void>> = [
      () => this.testBug1(),
      () => this.testBug2(),
      () => this.testCombined(),
    ];

    let passed = 0;
    let failed = 0;

    for (const test of tests) {
      try {
        await test();
        passed++;
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        console.error(`\n  FAILED: ${msg}`);
        failed++;
      }
    }

    // Cleanup
    try {
      fs.rmSync(this.tmpdir, { recursive: true, force: true });
    } catch {}

    console.log('\n' + '='.repeat(60));
    if (failed === 0) {
      console.log(`=== ALL ${passed} TESTS PASSED ===`);
    } else {
      console.log(`=== ${passed} passed, ${failed} FAILED ===`);
      process.exit(1);
    }
    console.log('='.repeat(60));
  }
}

async function main(): Promise<void> {
  const configPath = BugfixPermissionsMcpTest.findConfig();
  const agent: Agent = await Agent.create(configPath);
  const tmpdir: string = fs.mkdtempSync(path.join(os.tmpdir(), 'a3s-bugfix-'));
  const test = new BugfixPermissionsMcpTest(agent, configPath, tmpdir);
  await test.runAll();
}

main().catch((e: unknown) => {
  console.error('Fatal:', e);
  process.exit(1);
});
