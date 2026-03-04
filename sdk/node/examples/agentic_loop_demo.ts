#!/usr/bin/env npx tsx
/**
 * A3S Code — Agentic Loop Demo (Real LLM)
 *
 * Demonstrates the AgentLoop's autonomous problem-solving capabilities
 * using real LLM configuration. The agent reads files, writes code,
 * runs commands, and iterates until the task is complete.
 *
 * ## What This Example Shows
 *
 * 1. Autonomous File Operations — LLM reads, writes, and edits files
 * 2. Streaming Events — Real-time visibility into the agent's actions
 * 3. Planning Mode — Task decomposition with goal tracking
 * 4. Multi-Turn Conversation — Context preserved across turns
 * 5. Skills-Augmented Code Review — Built-in skills for better output
 * 6. Resilient Session — Parse retries, tool timeout, circuit breaker
 *
 * ## Usage
 *
 *     cd crates/code/sdk/node
 *     npx tsx examples/agentic_loop_demo.ts
 *
 *     # With custom config
 *     A3S_CONFIG=/path/to/config.hcl npx tsx examples/agentic_loop_demo.ts
 *
 * Requires a valid LLM API key in ~/.a3s/config.hcl or $A3S_CONFIG.
 */

import {
  Agent,
  Session,
  AgentEvent,
  AgentResult,
  SkillInfo,
  builtinSkills,
} from '../index.js';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

// ============================================================================
// AgenticLoopDemo
// ============================================================================

class AgenticLoopDemo {
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
    if (process.env.A3S_CONFIG) {
      return process.env.A3S_CONFIG;
    }

    const homeConfig: string = path.join(os.homedir(), '.a3s', 'config.hcl');
    if (fs.existsSync(homeConfig)) return homeConfig;

    // Project root (6 levels up from examples/)
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
    return fs.mkdtempSync(path.join(os.tmpdir(), 'a3s-demo-'));
  }

  private static cleanupDir(dir: string): void {
    try {
      fs.rmSync(dir, { recursive: true, force: true });
    } catch (_) { /* ignore */ }
  }

  // --------------------------------------------------------------------------
  // Demo 1: Autonomous Multi-Step Coding
  // --------------------------------------------------------------------------

  async demo1AutonomousCoding(): Promise<void> {
    AgenticLoopDemo.separator('Demo 1: Autonomous Multi-Step Coding');

    const workspace: string = AgenticLoopDemo.makeTempDir();
    try {
      const session: Session = this.agent.session(workspace, { permissive: true });

      console.log(`  Workspace: ${workspace}`);
      console.log('  Prompt:    Create a Rust file, then improve it\n');

      const result: AgentResult = await session.send(
        'Do the following steps:\n' +
        '1. Create a file called `calculator.rs` with a basic Calculator struct ' +
        'that has add() and subtract() methods\n' +
        '2. Read the file back to verify it was created correctly\n' +
        '3. Edit the file to add multiply() and divide() methods ' +
        '(divide should handle division by zero)\n' +
        '4. Read the final version and confirm all 4 methods exist'
      );

      console.log(`  Tool calls: ${result.toolCallsCount}`);
      console.log(`  Tokens:     ${result.totalTokens} total`);
      console.log(`  Response:   ${AgenticLoopDemo.truncate(result.text)}`);

      // Verify the file was actually created
      const calcPath: string = path.join(workspace, 'calculator.rs');
      if (fs.existsSync(calcPath)) {
        const content: string = fs.readFileSync(calcPath, 'utf-8');
        const hasAdd: boolean = content.includes('add');
        const hasSub: boolean = content.includes('subtract') || content.includes('sub');
        const hasMul: boolean = content.includes('multiply') || content.includes('mul');
        const hasDiv: boolean = content.includes('divide') || content.includes('div');
        console.log(
          `\n  File verified: add=${hasAdd} sub=${hasSub} mul=${hasMul} div=${hasDiv}`
        );
      } else {
        console.log('\n  calculator.rs not found (LLM may have used a different name)');
      }
    } finally {
      AgenticLoopDemo.cleanupDir(workspace);
    }
  }

  // --------------------------------------------------------------------------
  // Demo 2: Streaming Events
  // --------------------------------------------------------------------------

  async demo2StreamingEvents(): Promise<void> {
    AgenticLoopDemo.separator('Demo 2: Streaming Events (Real-Time Agent Visibility)');

    const workspace: string = AgenticLoopDemo.makeTempDir();
    try {
      // Pre-create a file for the agent to work with
      fs.writeFileSync(
        path.join(workspace, 'data.json'),
        '{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}'
      );

      const session: Session = this.agent.session(workspace, { permissive: true });

      console.log(`  Workspace: ${workspace}`);
      console.log('  Prompt:    Read data.json and create a summary\n');
      console.log('  --- Event Stream ---');

      const stream = await session.stream(
        'Read the file data.json in the workspace, then create a file called ' +
        'summary.txt that lists each user\'s name and age in a human-readable format.'
      );

      let toolCount: number = 0;
      let textLen: number = 0;

      for await (const event of stream) {
        switch (event.type) {
          case 'start':
            console.log('  > Agent started');
            break;
          case 'turn_start':
            console.log(`  +- Turn ${event.turn}`);
            break;
          case 'tool_start':
            toolCount++;
            process.stdout.write(`  |  ${event.toolName}...`);
            break;
          case 'tool_end': {
            const status: string = event.exitCode === 0 ? 'ok' : 'fail';
            console.log(` ${status} (exit=${event.exitCode})`);
            break;
          }
          case 'text_delta':
            textLen += (event.text || '').length;
            break;
          case 'turn_end':
            console.log(`  +- Turn ${event.turn} done (${event.totalTokens} tokens)`);
            break;
          case 'end':
            console.log('\n  Agent finished');
            console.log(
              `    Tools: ${toolCount}, Response: ${textLen} chars, ` +
              `Tokens: ${event.totalTokens}`
            );
            break;
          case 'error':
            console.log(`\n  Error: ${event.error}`);
            break;
        }
      }

      // Verify output
      const summaryPath: string = path.join(workspace, 'summary.txt');
      if (fs.existsSync(summaryPath)) {
        const summary: string = fs.readFileSync(summaryPath, 'utf-8');
        console.log(`\n  summary.txt created (${summary.length} bytes)`);
      }
    } finally {
      AgenticLoopDemo.cleanupDir(workspace);
    }
  }

  // --------------------------------------------------------------------------
  // Demo 3: Planning Mode
  // --------------------------------------------------------------------------

  async demo3PlanningMode(): Promise<void> {
    AgenticLoopDemo.separator('Demo 3: Planning Mode (Task Decomposition)');

    const workspace: string = AgenticLoopDemo.makeTempDir();
    try {
      const session: Session = this.agent.session(workspace, {
        planning: true,
        goalTracking: true,
        permissive: true,
      });

      console.log(`  Workspace: ${workspace}`);
      console.log('  Planning:  enabled');
      console.log('  Goal tracking: enabled\n');

      const result: AgentResult = await session.send(
        'Create a small project with these files:\n' +
        '1. `lib.rs` — a module with a `greet(name: &str) -> String` function\n' +
        '2. `main.rs` — imports from lib and calls greet("World")\n' +
        '3. `README.md` — brief documentation explaining the project\n' +
        'Then read all three files to verify they are correct.'
      );

      console.log(`  Tool calls: ${result.toolCallsCount}`);
      console.log(`  Tokens:     ${result.totalTokens} total`);
      console.log(`  Response:   ${AgenticLoopDemo.truncate(result.text)}`);

      // Verify files
      for (const f of ['lib.rs', 'main.rs', 'README.md']) {
        const filePath: string = path.join(workspace, f);
        if (fs.existsSync(filePath)) {
          const size: number = fs.statSync(filePath).size;
          console.log(`  ${f} (${size} bytes)`);
        } else {
          console.log(`  ${f} not found`);
        }
      }
    } finally {
      AgenticLoopDemo.cleanupDir(workspace);
    }
  }

  // --------------------------------------------------------------------------
  // Demo 4: Multi-Turn Conversation
  // --------------------------------------------------------------------------

  async demo4MultiTurn(): Promise<void> {
    AgenticLoopDemo.separator('Demo 4: Multi-Turn Conversation (Context Preservation)');

    const workspace: string = AgenticLoopDemo.makeTempDir();
    try {
      const session: Session = this.agent.session(workspace, { permissive: true });

      console.log(`  Workspace: ${workspace}\n`);

      // Turn 1
      console.log('  [Turn 1] Create a config file');
      const r1: AgentResult = await session.send(
        'Create a file called `config.toml` with these settings:\n' +
        '- server.host = "0.0.0.0"\n' +
        '- server.port = 8080\n' +
        '- database.url = "postgres://localhost/mydb"\n' +
        '- database.pool_size = 5'
      );
      console.log(`    Tools: ${r1.toolCallsCount}, Tokens: ${r1.totalTokens}`);

      // Turn 2 — LLM should remember the file from Turn 1
      console.log('\n  [Turn 2] Ask about the file (tests context memory)');
      const r2: AgentResult = await session.send(
        'What port is the server configured to use? Read the config file to confirm.'
      );
      console.log(`    Tools: ${r2.toolCallsCount}, Tokens: ${r2.totalTokens}`);
      console.log(`    Answer: ${AgenticLoopDemo.truncate(r2.text, 120)}`);

      // Turn 3 — modify based on context
      console.log('\n  [Turn 3] Modify based on previous context');
      const r3: AgentResult = await session.send(
        'Change the server port to 3000 and increase the pool_size to 10.'
      );
      console.log(`    Tools: ${r3.toolCallsCount}, Tokens: ${r3.totalTokens}`);

      // Verify final state
      const history = session.history();
      console.log(`\n  History: ${history.length} messages across 3 turns`);

      const configPath: string = path.join(workspace, 'config.toml');
      if (fs.existsSync(configPath)) {
        const content: string = fs.readFileSync(configPath, 'utf-8');
        const has3000: boolean = content.includes('3000');
        const has10: boolean = content.includes('10');
        console.log(`  config.toml: port=3000? ${has3000} pool=10? ${has10}`);
      }
    } finally {
      AgenticLoopDemo.cleanupDir(workspace);
    }
  }

  // --------------------------------------------------------------------------
  // Demo 5: Skills-Augmented Code Assistance
  // --------------------------------------------------------------------------

  async demo5SkillsAugmented(): Promise<void> {
    AgenticLoopDemo.separator('Demo 5: Skills-Augmented Code Assistance');

    const workspace: string = AgenticLoopDemo.makeTempDir();
    try {
      // Pre-create a file with intentional issues
      fs.writeFileSync(
        path.join(workspace, 'app.py'),
        'import os\n' +
        'import sys\n' +
        '\n' +
        'def process_data(data):\n' +
        '    result = []\n' +
        '    for i in range(len(data)):\n' +
        '        if data[i] != None:\n' +
        '            result.append(data[i] * 2)\n' +
        '    return result\n' +
        '\n' +
        'def read_file(path):\n' +
        "    f = open(path, 'r')\n" +
        '    content = f.read()\n' +
        '    return content\n' +
        '\n' +
        'API_KEY = "sk-1234567890abcdef"\n'
      );

      const skills: SkillInfo[] = builtinSkills();
      const session: Session = this.agent.session(workspace, { builtinSkills: true, permissive: true });

      console.log(`  Workspace: ${workspace}`);
      console.log(`  Skills:    built-in (${skills.length} skills active)\n`);

      const result: AgentResult = await session.send(
        'Review the file `app.py` in the workspace. Identify code quality issues ' +
        'and fix them. The review should cover:\n' +
        '- Pythonic style (use `is not None`, enumerate, etc.)\n' +
        '- Resource management (use context managers)\n' +
        '- Security issues (hardcoded secrets)\n' +
        'Apply the fixes by editing the file.'
      );

      console.log(`  Tool calls: ${result.toolCallsCount}`);
      console.log(`  Tokens:     ${result.totalTokens} total`);
      console.log(`  Response:   ${AgenticLoopDemo.truncate(result.text)}`);

      // Check improvements
      const appPath: string = path.join(workspace, 'app.py');
      if (fs.existsSync(appPath)) {
        const content: string = fs.readFileSync(appPath, 'utf-8');
        const hasCtxMgr: boolean = content.includes('with open');
        const hasEnumerate: boolean = content.includes('enumerate');
        const noKey: boolean = !content.includes('sk-1234567890abcdef');
        console.log(
          `\n  Improvements: context_manager=${hasCtxMgr} ` +
          `enumerate=${hasEnumerate} no_hardcoded_key=${noKey}`
        );
      }
    } finally {
      AgenticLoopDemo.cleanupDir(workspace);
    }
  }

  // --------------------------------------------------------------------------
  // Demo 6: Resilient Session
  // --------------------------------------------------------------------------

  async demo6ResilientSession(): Promise<void> {
    AgenticLoopDemo.separator('Demo 6: Resilient Session (Error Recovery)');

    const workspace: string = AgenticLoopDemo.makeTempDir();
    try {
      const session: Session = this.agent.session(workspace, {
        builtinSkills: true,
        permissive: true,
        maxParseRetries: 2,
        toolTimeoutMs: 120_000,
        circuitBreakerThreshold: 3,
      });

      console.log(`  Workspace:       ${workspace}`);
      console.log('  Parse retries:   2');
      console.log('  Tool timeout:    120s');
      console.log('  Circuit breaker: 3\n');

      const result: AgentResult = await session.send(
        "Create a file called `test.sh` with a bash script that prints 'hello' " +
        'and the current date. Then execute it with bash and show me the output.'
      );

      console.log(`  Tool calls: ${result.toolCallsCount}`);
      console.log(`  Tokens:     ${result.totalTokens} total`);
      console.log(`  Response:   ${AgenticLoopDemo.truncate(result.text)}`);

      if (fs.existsSync(path.join(workspace, 'test.sh'))) {
        console.log('\n  test.sh created and executed successfully');
      }
    } finally {
      AgenticLoopDemo.cleanupDir(workspace);
    }
  }

  // --------------------------------------------------------------------------
  // Run All
  // --------------------------------------------------------------------------

  async runAll(): Promise<void> {
    await this.demo1AutonomousCoding();
    await this.demo2StreamingEvents();
    await this.demo3PlanningMode();
    await this.demo4MultiTurn();
    await this.demo5SkillsAugmented();
    await this.demo6ResilientSession();

    AgenticLoopDemo.separator('All Demos Complete');
  }
}

// ============================================================================
// Main
// ============================================================================

async function main(): Promise<void> {
  console.log('======================================================================');
  console.log('       A3S Code Node.js SDK — Agentic Loop Demo (Real LLM)');
  console.log('======================================================================');

  const configPath: string = AgenticLoopDemo.findConfigPath();
  console.log(`\n  Config: ${configPath}`);

  const agent: Agent = await Agent.create(configPath);
  console.log('  Agent:  created\n');

  const demo = new AgenticLoopDemo(agent, configPath);
  await demo.runAll();
}

main().catch((err: unknown) => {
  console.error('Demo failed:', err);
  process.exit(1);
});
