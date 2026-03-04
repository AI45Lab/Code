#!/usr/bin/env npx tsx
/**
 * A3S Code — Advanced Features Demo
 *
 * Demonstrates: auto-compact, memory, security, hooks, batch tool,
 * permission checker, planning, and context providers.
 *
 * ## Usage
 *
 *     cd crates/code/sdk/node
 *     npx tsx examples/test_advanced_features.ts
 *
 * Requires a valid LLM API key in ~/.a3s/config.hcl or $A3S_CONFIG.
 */

import { Agent, Session, AgentResult } from '../index.js';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

// ============================================================================
// AdvancedFeaturesTest
// ============================================================================

class AdvancedFeaturesTest {
  private readonly configPath: string;

  constructor(configPath: string) {
    this.configPath = configPath;
  }

  // --------------------------------------------------------------------------
  // Static helpers
  // --------------------------------------------------------------------------

  static findConfigPath(): string {
    if (process.env.A3S_CONFIG) return process.env.A3S_CONFIG;
    const homeConfig: string = path.join(os.homedir(), '.a3s', 'config.hcl');
    if (fs.existsSync(homeConfig)) return homeConfig;
    throw new Error('No config found. Set A3S_CONFIG or create ~/.a3s/config.hcl');
  }

  // --------------------------------------------------------------------------
  // Test 1: Auto-Compact
  // --------------------------------------------------------------------------

  async testAutoCompact(): Promise<void> {
    console.log('\n=== Test 1: Auto-Compact ===');
    const agent: Agent = await Agent.create(this.configPath);

    // Enable auto-compact at 80% context usage
    const session: Session = agent.session(os.tmpdir(), {
      autoCompact: true,
      autoCompactThreshold: 0.8,
    });

    const result: AgentResult = await session.send('Say hello in one sentence.');
    console.log('auto-compact session created, response:', result.text.slice(0, 60));
  }

  // --------------------------------------------------------------------------
  // Test 2: File Memory
  // --------------------------------------------------------------------------

  async testMemory(): Promise<void> {
    console.log('\n=== Test 2: File Memory ===');
    const agent: Agent = await Agent.create(this.configPath);
    const memDir: string = fs.mkdtempSync(path.join(os.tmpdir(), 'a3s-mem-'));

    const session: Session = agent.session(os.tmpdir(), {
      memoryDir: memDir,
    });

    const result: AgentResult = await session.send('Remember: my favorite language is Rust.');
    console.log('memory session created, response:', result.text.slice(0, 60));

    // Cleanup
    fs.rmSync(memDir, { recursive: true, force: true });
  }

  // --------------------------------------------------------------------------
  // Test 3: Default Security Provider
  // --------------------------------------------------------------------------

  async testSecurity(): Promise<void> {
    console.log('\n=== Test 3: Security Provider ===');
    const agent: Agent = await Agent.create(this.configPath);

    // Enable default security: taint-tracks input, sanitizes output
    const session: Session = agent.session(os.tmpdir(), {
      defaultSecurity: true,
    });

    const result: AgentResult = await session.send('What is 2 + 2?');
    console.log('security session created, response:', result.text.slice(0, 60));
  }

  // --------------------------------------------------------------------------
  // Test 4: Hooks — intercept lifecycle events
  // --------------------------------------------------------------------------

  async testHooks(): Promise<void> {
    console.log('\n=== Test 4: Hooks ===');
    const agent: Agent = await Agent.create(this.configPath);
    const session: Session = agent.session(os.tmpdir());

    const toolCalls: string[] = [];

    // Register a hook that fires before every tool use
    session.registerHook('log-tools', 'pre_tool_use', (event: Record<string, unknown>) => {
      toolCalls.push((event.toolName as string) || (event.tool_name as string) || 'unknown');
      console.log(`  -> pre_tool_use: ${JSON.stringify(event).slice(0, 80)}`);
      return null; // allow execution
    });

    console.log(`hook registered, hook count: ${session.hookCount()}`);

    // Cleanup
    session.unregisterHook('log-tools');
    console.log(`hook unregistered, hook count: ${session.hookCount()}`);
  }

  // --------------------------------------------------------------------------
  // Test 5: Planning Mode
  // --------------------------------------------------------------------------

  async testPlanning(): Promise<void> {
    console.log('\n=== Test 5: Planning Mode ===');
    const agent: Agent = await Agent.create(this.configPath);

    const session: Session = agent.session(os.tmpdir(), {
      planning: true,
      goalTracking: true,
    });

    const result: AgentResult = await session.send('List 3 benefits of Rust in one sentence each.');
    console.log('planning session, response length:', result.text.length);
  }

  // --------------------------------------------------------------------------
  // Test 6: Permissive Policy (no HITL confirmation)
  // --------------------------------------------------------------------------

  async testPermissive(): Promise<void> {
    console.log('\n=== Test 6: Permissive Policy ===');
    const agent: Agent = await Agent.create(this.configPath);

    const session: Session = agent.session(os.tmpdir(), {
      permissive: true,
    });

    const result: AgentResult = await session.send('What is the current directory?');
    console.log('permissive session, response:', result.text.slice(0, 60));
  }

  // --------------------------------------------------------------------------
  // Test 7: Resilience Options
  // --------------------------------------------------------------------------

  async testResilience(): Promise<void> {
    console.log('\n=== Test 7: Resilience Options ===');
    const agent: Agent = await Agent.create(this.configPath);

    const session: Session = agent.session(os.tmpdir(), {
      maxParseRetries: 3,
      toolTimeoutMs: 30000,
      circuitBreakerThreshold: 5,
    });

    const result: AgentResult = await session.send('Say "resilient" once.');
    console.log('resilience session, response:', result.text.slice(0, 60));
  }

  // --------------------------------------------------------------------------
  // Test 8: Combined — security + auto-compact + memory + hooks
  // --------------------------------------------------------------------------

  async testCombined(): Promise<void> {
    console.log('\n=== Test 8: Combined Features ===');
    const agent: Agent = await Agent.create(this.configPath);
    const memDir: string = fs.mkdtempSync(path.join(os.tmpdir(), 'a3s-combined-'));

    const session: Session = agent.session(os.tmpdir(), {
      defaultSecurity: true,
      autoCompact: true,
      autoCompactThreshold: 0.9,
      memoryDir: memDir,
      permissive: true,
    });

    session.registerHook('audit', 'post_tool_use', (event: Record<string, unknown>) => {
      console.log(`  -> post_tool_use fired`);
      return null;
    });

    const result: AgentResult = await session.send('What is 1 + 1?');
    console.log('combined session, response:', result.text.slice(0, 60));

    fs.rmSync(memDir, { recursive: true, force: true });
  }

  // --------------------------------------------------------------------------
  // Run All
  // --------------------------------------------------------------------------

  async runAll(): Promise<void> {
    const tests: Array<{ name: string; fn: () => Promise<void> }> = [
      { name: 'testAutoCompact', fn: () => this.testAutoCompact() },
      { name: 'testMemory', fn: () => this.testMemory() },
      { name: 'testSecurity', fn: () => this.testSecurity() },
      { name: 'testHooks', fn: () => this.testHooks() },
      { name: 'testPlanning', fn: () => this.testPlanning() },
      { name: 'testPermissive', fn: () => this.testPermissive() },
      { name: 'testResilience', fn: () => this.testResilience() },
      { name: 'testCombined', fn: () => this.testCombined() },
    ];

    let passed: number = 0;
    let failed: number = 0;

    for (const test of tests) {
      try {
        await test.fn();
        passed++;
      } catch (err) {
        const message: string = err instanceof Error ? err.message : String(err);
        console.error(`  ${test.name} failed:`, message);
        failed++;
      }
    }

    console.log(`\n===================================`);
    console.log(`Results: ${passed} passed, ${failed} failed`);
    process.exit(failed > 0 ? 1 : 0);
  }
}

// ============================================================================
// Main
// ============================================================================

async function main(): Promise<void> {
  console.log('A3S Code — Advanced Features Test');
  console.log('===================================');

  const configPath: string = AdvancedFeaturesTest.findConfigPath();
  const test = new AdvancedFeaturesTest(configPath);
  await test.runAll();
}

main().catch((err: unknown) => {
  console.error('Fatal:', err);
  process.exit(1);
});
