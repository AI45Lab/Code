#!/usr/bin/env npx tsx
/**
 * A3S Code — Advanced Features Demo (Real LLM)
 *
 * Demonstrates features NOT covered by agentic_loop_demo.js:
 * 1. Direct tool execution (no LLM)
 * 2. Hooks (lifecycle event interception)
 * 3. Queue & lanes (priority-based scheduling)
 * 4. Security provider (taint tracking)
 * 5. Auto-compaction & resilience
 * 6. Persistent memory
 *
 * Usage:
 *     cd crates/code/sdk/node
 *     npx tsx examples/advanced_features_demo.ts
 */

import {
  Agent,
  Session,
  SessionOptions,
  SessionQueueConfig,
  AgentEvent,
  AgentResult,
  ToolResult,
  builtinSkills,
} from '../index.js';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

// ============================================================================
// AdvancedFeaturesDemo
// ============================================================================

class AdvancedFeaturesDemo {
  private readonly agent: Agent;
  private readonly configPath: string;

  constructor(agent: Agent, configPath: string) {
    this.agent = agent;
    this.configPath = configPath;
  }

  // --------------------------------------------------------------------------
  // Static helpers
  // --------------------------------------------------------------------------

  static findConfigPath(): string {
    if (process.env.A3S_CONFIG) return process.env.A3S_CONFIG;
    const homeConfig: string = path.join(os.homedir(), '.a3s', 'config.hcl');
    if (fs.existsSync(homeConfig)) return homeConfig;
    const projectConfig: string = path.join(
      __dirname, '..', '..', '..', '..', '..', '..', '.a3s', 'config.hcl'
    );
    if (fs.existsSync(projectConfig)) return projectConfig;
    throw new Error('Config not found. Create ~/.a3s/config.hcl or set A3S_CONFIG');
  }

  private static separator(title: string): void {
    console.log(`\n${'='.repeat(72)}`);
    console.log(`  ${title}`);
    console.log(`${'='.repeat(72)}\n`);
  }

  private static truncate(text: string, max: number = 200): string {
    text = text.trim();
    return text.length <= max ? text : `${text.substring(0, max)}...`;
  }

  private static makeTempDir(): string {
    return fs.mkdtempSync(path.join(os.tmpdir(), 'a3s-adv-'));
  }

  private static cleanupDir(dir: string): void {
    try { fs.rmSync(dir, { recursive: true, force: true }); } catch (_) {}
  }

  // --------------------------------------------------------------------------
  // Demo 1: Direct Tool Execution
  // --------------------------------------------------------------------------

  async demo1DirectTools(): Promise<void> {
    AdvancedFeaturesDemo.separator('Demo 1: Direct Tool Execution (No LLM)');

    const workspace: string = AdvancedFeaturesDemo.makeTempDir();
    try {
      const session: Session = this.agent.session(workspace, { permissive: true });
      console.log(`  Workspace: ${workspace}\n`);

      // Write
      console.log('  1. Write a file');
      let r: ToolResult = await session.tool('write', { path: 'hello.txt', content: 'Hello from Node.js!\nLine 2.' });
      console.log(`     exit=${r.exitCode} output=${AdvancedFeaturesDemo.truncate(r.output, 60)}`);

      // Read
      console.log('  2. Read the file');
      const content: string = await session.readFile('hello.txt');
      console.log(`     content: ${AdvancedFeaturesDemo.truncate(content, 60)}`);

      // Glob
      console.log('  3. Glob for *.txt');
      const files: string[] = await session.glob('*.txt');
      console.log(`     matches: ${JSON.stringify(files)}`);

      // Bash
      console.log('  4. Run bash command');
      const bashOut: string = await session.bash('wc -l hello.txt');
      console.log(`     output: ${bashOut.trim()}`);

      // Grep
      console.log('  5. Grep for "Node"');
      const grepOut: string = await session.grep('Node');
      console.log(`     output: ${AdvancedFeaturesDemo.truncate(grepOut, 60)}`);

      // Edit
      console.log('  6. Edit the file');
      r = await session.tool('edit', {
        path: 'hello.txt',
        old_string: 'Line 2.',
        new_string: 'Line 2 — edited from Node.js!'
      });
      console.log(`     exit=${r.exitCode}`);

      // Verify
      const final_: string = await session.readFile('hello.txt');
      console.log(`\n  Final content: ${AdvancedFeaturesDemo.truncate(final_, 80)}`);
    } finally {
      AdvancedFeaturesDemo.cleanupDir(workspace);
    }
  }

  // --------------------------------------------------------------------------
  // Demo 2: Hooks
  // --------------------------------------------------------------------------

  async demo2Hooks(): Promise<void> {
    AdvancedFeaturesDemo.separator('Demo 2: Hooks (Lifecycle Event Interception)');

    const workspace: string = AdvancedFeaturesDemo.makeTempDir();
    try {
      const session: Session = this.agent.session(workspace, { permissive: true });

      // Register hooks
      session.registerHook('audit_tools', 'pre_tool_use', null, { priority: 10 });
      session.registerHook('audit_bash', 'post_tool_use', { tool: 'bash' }, { priority: 20 });
      session.registerHook('log_gen', 'generate_start');
      session.registerHook('log_err', 'on_error');

      console.log(`  Registered ${session.hookCount()} hooks`);
      console.log('  * audit_tools  (PreToolUse, priority=10)');
      console.log('  * audit_bash   (PostToolUse, bash only)');
      console.log('  * log_gen      (GenerateStart)');
      console.log('  * log_err      (OnError)\n');

      for await (const event of await session.stream(
        "Create a file test.sh with `echo hello` and run it with bash."
      )) {
        switch (event.type) {
          case 'tool_start':
            console.log(`  [hook] PreToolUse -> ${event.toolName}`);
            break;
          case 'tool_end':
            if (event.toolName === 'bash') {
              console.log(`  [hook] PostToolUse(bash) -> exit=${event.exitCode}`);
            }
            break;
          case 'end':
            console.log(`\n  Done (${event.totalTokens} tokens)`);
            break;
          case 'error':
            console.log(`  Error: ${event.error}`);
            break;
        }
        if (event.type === 'end' || event.type === 'error') break;
      }

      const removed: boolean = session.unregisterHook('audit_tools');
      console.log(`\n  Unregistered audit_tools: ${removed}`);
      console.log(`  Remaining hooks: ${session.hookCount()}`);
    } finally {
      AdvancedFeaturesDemo.cleanupDir(workspace);
    }
  }

  // --------------------------------------------------------------------------
  // Demo 3: Queue & Lanes
  // --------------------------------------------------------------------------

  async demo3QueueLanes(): Promise<void> {
    AdvancedFeaturesDemo.separator('Demo 3: Queue & Lanes (Priority-Based Scheduling)');

    const workspace: string = AdvancedFeaturesDemo.makeTempDir();
    try {
      for (let i = 1; i <= 3; i++) {
        fs.writeFileSync(path.join(workspace, `file${i}.txt`), `Content of file ${i}\n`);
      }

      const qc: SessionQueueConfig = new SessionQueueConfig();
      qc.withLaneFeatures();
      qc.setQueryConcurrency(8);
      qc.setExecuteConcurrency(2);

      const session: Session = this.agent.session(workspace, { permissive: true, queueConfig: qc });

      console.log(`  Queue active: ${session.hasQueue()}`);
      console.log(`  Workspace: ${workspace}\n`);

      const result: AgentResult = await session.send(
        'Read all .txt files and create a combined.md with their contents.'
      );

      console.log(`  Tool calls: ${result.toolCallsCount}`);
      console.log(`  Tokens:     ${result.totalTokens}`);

      const stats: Record<string, unknown> = session.queueStats();
      console.log(`\n  Queue stats: ${JSON.stringify(stats)}`);

      const dlq: Array<Record<string, unknown>> = session.deadLetters();
      console.log(`  Dead letters: ${dlq.length}`);
    } finally {
      AdvancedFeaturesDemo.cleanupDir(workspace);
    }
  }

  // --------------------------------------------------------------------------
  // Demo 4: Security Provider
  // --------------------------------------------------------------------------

  async demo4Security(): Promise<void> {
    AdvancedFeaturesDemo.separator('Demo 4: Security Provider (Taint Tracking)');

    const workspace: string = AdvancedFeaturesDemo.makeTempDir();
    try {
      fs.writeFileSync(
        path.join(workspace, 'secrets.env'),
        'DATABASE_URL=postgres://admin:p4ssw0rd@db.example.com/prod\n' +
        'API_KEY=sk-abc123def456\n' +
        'AWS_SECRET=AKIAIOSFODNN7EXAMPLE\n'
      );

      const session: Session = this.agent.session(workspace, { defaultSecurity: true, permissive: true });

      console.log('  Security: DefaultSecurityProvider enabled');
      console.log('  Features: taint tracking + output sanitization\n');

      const result: AgentResult = await session.send(
        'Read secrets.env and tell me what types of secrets are in it. ' +
        'Do NOT include the actual secret values in your response.'
      );

      console.log(`  Tool calls: ${result.toolCallsCount}`);
      console.log(`  Tokens:     ${result.totalTokens}`);
      console.log(`  Response:   ${AdvancedFeaturesDemo.truncate(result.text, 300)}`);
    } finally {
      AdvancedFeaturesDemo.cleanupDir(workspace);
    }
  }

  // --------------------------------------------------------------------------
  // Demo 5: Auto-Compaction & Resilience
  // --------------------------------------------------------------------------

  async demo5Resilience(): Promise<void> {
    AdvancedFeaturesDemo.separator('Demo 5: Auto-Compaction & Resilience');

    const workspace: string = AdvancedFeaturesDemo.makeTempDir();
    try {
      const session: Session = this.agent.session(workspace, {
        permissive: true,
        maxParseRetries: 3,
        toolTimeoutMs: 30_000,
        circuitBreakerThreshold: 5,
      });

      console.log('  Parse retries:   3');
      console.log('  Tool timeout:    30s');
      console.log('  Circuit breaker: 5\n');

      const result: AgentResult = await session.send(
        'Create a CSV file with 15 rows of sample data (id, name, score), ' +
        'then create a report.md analyzing the data.'
      );

      console.log(`  Tool calls: ${result.toolCallsCount}`);
      console.log(`  Tokens:     ${result.totalTokens}`);

      for (const name of ['data.csv', 'report.md']) {
        const p: string = path.join(workspace, name);
        if (fs.existsSync(p)) {
          console.log(`  ${name} (${fs.statSync(p).size} bytes)`);
        } else {
          console.log(`  ${name} not found`);
        }
      }
    } finally {
      AdvancedFeaturesDemo.cleanupDir(workspace);
    }
  }

  // --------------------------------------------------------------------------
  // Demo 6: Persistent Memory
  // --------------------------------------------------------------------------

  async demo6Memory(): Promise<void> {
    AdvancedFeaturesDemo.separator('Demo 6: Persistent Memory');

    const workspace: string = AdvancedFeaturesDemo.makeTempDir();
    const memoryDir: string = AdvancedFeaturesDemo.makeTempDir();
    try {
      const session: Session = this.agent.session(workspace, { memoryDir, permissive: true });

      console.log(`  Memory dir: ${memoryDir}\n`);

      // Turn 1: teach
      console.log('  [Turn 1] Teaching the agent...');
      const r1: AgentResult = await session.send(
        'Remember this: The project deadline is March 15, 2026. ' +
        'The tech lead is Alice and the PM is Bob.'
      );
      console.log(`    Tokens: ${r1.totalTokens}`);

      // Turn 2: recall
      console.log('  [Turn 2] Asking about remembered info...');
      const r2: AgentResult = await session.send('Who is the tech lead and when is the deadline?');
      console.log(`    Tokens: ${r2.totalTokens}`);
      console.log(`    Answer: ${AdvancedFeaturesDemo.truncate(r2.text, 150)}`);
    } finally {
      AdvancedFeaturesDemo.cleanupDir(workspace);
      AdvancedFeaturesDemo.cleanupDir(memoryDir);
    }
  }

  // --------------------------------------------------------------------------
  // Run All
  // --------------------------------------------------------------------------

  async runAll(): Promise<void> {
    await this.demo1DirectTools();
    await this.demo2Hooks();
    await this.demo3QueueLanes();
    await this.demo4Security();
    await this.demo5Resilience();
    await this.demo6Memory();

    AdvancedFeaturesDemo.separator('All Demos Complete');
  }
}

// ============================================================================
// Main
// ============================================================================

async function main(): Promise<void> {
  console.log('======================================================================');
  console.log('      A3S Code Node.js SDK — Advanced Features Demo (Real LLM)');
  console.log('======================================================================');

  const configPath: string = AdvancedFeaturesDemo.findConfigPath();
  console.log(`\n  Config: ${configPath}`);

  const agent: Agent = await Agent.create(configPath);
  console.log('  Agent:  created\n');

  const demo = new AdvancedFeaturesDemo(agent, configPath);
  await demo.runAll();
}

main().catch((err: unknown) => {
  console.error('Demo failed:', err);
  process.exit(1);
});
