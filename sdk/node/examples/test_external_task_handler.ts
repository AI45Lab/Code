#!/usr/bin/env npx tsx
/**
 * A3S Code Node.js SDK - External Task Handler Integration Test (TypeScript)
 *
 * Demonstrates the Multi-Machine External Task pattern:
 * 1. Coordinator creates a session with Execute lane set to External mode
 * 2. Agent sends a prompt that triggers bash/write/edit tool calls
 * 3. Coordinator polls pendingExternalTasks() in a background interval
 * 4. Coordinator processes tasks locally (simulating a remote worker)
 * 5. Coordinator completes tasks via completeExternalTask()
 *
 * Run with: npx tsx examples/test_external_task_handler.ts
 */

import { Agent, Session, AgentResult } from '../index.js';
import { execSync } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';

interface WorkerResult {
  success: boolean;
  output: string;
  exitCode: number;
  error: string | null;
}

interface ExternalTask {
  task_id: string;
  command_type: string;
  payload?: {
    command?: string;
    working_dir?: string;
  };
}

class ExternalTaskHandlerTest {
  private readonly agent: Agent;
  private readonly configPath: string;
  private session: Session | null = null;

  constructor(agent: Agent, configPath: string) {
    this.agent = agent;
    this.configPath = configPath;
  }

  // ── Static Helpers ───────────────────────────────────────────────────────

  static findConfig(): string {
    const homeConfig: string = path.join(os.homedir(), '.a3s', 'config.hcl');
    if (fs.existsSync(homeConfig)) return homeConfig;

    const projectConfig: string = path.resolve(__dirname, '..', '..', '..', '..', '..', '.a3s', 'config.hcl');
    if (fs.existsSync(projectConfig)) return projectConfig;

    throw new Error('Config file not found');
  }

  /**
   * Simulate a remote worker executing a bash command.
   */
  private static workerExecuteBash(command: string, workingDir: string = '.'): WorkerResult {
    try {
      const output: string = execSync(command, {
        cwd: workingDir,
        encoding: 'utf-8',
        timeout: 30000,
        stdio: ['pipe', 'pipe', 'pipe'],
      });
      return { success: true, output, exitCode: 0, error: null };
    } catch (e: unknown) {
      const err = e as { stdout?: string; status?: number; stderr?: string; message?: string };
      return {
        success: false,
        output: err.stdout || '',
        exitCode: err.status || 1,
        error: err.stderr || err.message || 'Unknown error',
      };
    }
  }

  // ── Instance Test Methods ────────────────────────────────────────────────

  async testExternalTaskHandler(): Promise<void> {
    console.log('\n📦 Test 1: External Task Handler (Execute Lane → External)');
    console.log('-'.repeat(80));

    // 1. Create session with queue enabled
    const session: Session = this.agent.session('.', {
      queueConfig: {
        enableAllFeatures: true,
        timeoutMs: 60000,
      },
    });

    // 2. Route Execute lane to External mode
    await session.setLaneHandler('execute', {
      mode: 'external',
      timeoutMs: 60000,
    });

    console.log('✓ Session created with Execute lane → External mode');
    console.log('  Query lane: Internal (read, glob, grep run locally)');
    console.log('  Execute lane: External (bash, write, edit → ExternalTask)');
    console.log();

    // 3. Start a background poller for external tasks
    const start: number = Date.now();
    let externalTasksProcessed: number = 0;
    let stopPolling: boolean = false;

    const pollInterval: ReturnType<typeof setInterval> = setInterval(async (): Promise<void> => {
      if (stopPolling) return;

      try {
        const tasks = await session.pendingExternalTasks() as unknown as ExternalTask[];
        if (!Array.isArray(tasks)) return;

        for (const task of tasks) {
          const taskId: string = task.task_id;
          const cmdType: string = task.command_type;
          console.log(`  📥 ExternalTaskPending: ${taskId.slice(0, 8)} (${cmdType})`);

          // Execute the task (simulating a remote worker)
          let result: WorkerResult;
          if (cmdType === 'bash') {
            const cmd: string = task.payload?.command || "echo 'no command'";
            const cwd: string = task.payload?.working_dir || '.';
            result = ExternalTaskHandlerTest.workerExecuteBash(cmd, cwd);
          } else {
            result = {
              success: true,
              output: `External handler processed: ${cmdType}`,
              exitCode: 0,
              error: null,
            };
          }

          console.log(`  🔧 Worker result: success=${result.success}, exit_code=${result.exitCode}`);
          const preview: string = (result.output || '').trim().slice(0, 60);
          if (preview) console.log(`     Output: ${preview}`);

          // Complete the external task
          const completed: boolean = await session.completeExternalTask(taskId, {
            success: result.success,
            result: { output: result.output, exit_code: result.exitCode },
            error: result.error,
          });

          if (completed) {
            externalTasksProcessed++;
            console.log(`  📤 Task ${taskId.slice(0, 8)} completed and returned to agent`);
          }
        }
      } catch (e: unknown) {
        // Session might be done
        const err = e as { message?: string };
        if (!err.message?.includes('exhausted')) {
          console.log(`  ⚠ Poll error: ${err.message}`);
        }
      }
    }, 200); // Poll every 200ms

    // 4. Send the prompt — agent will produce ExternalTask objects for bash calls
    //    Note: Node SDK stream() collects all events, so we use send() + polling
    let result: AgentResult;
    try {
      result = await session.send(
        "Run these bash commands and tell me the results:\n" +
        "1. echo 'Hello from external worker'\n" +
        "2. date '+%Y-%m-%d %H:%M:%S'\n" +
        "3. uname -s"
      );
    } finally {
      stopPolling = true;
      clearInterval(pollInterval);
    }

    const duration: number = (Date.now() - start) / 1000;

    console.log();
    console.log('-'.repeat(80));
    console.log('📊 Results:');
    console.log(`  Duration: ${duration.toFixed(2)}s`);
    console.log(`  External tasks processed: ${externalTasksProcessed}`);
    console.log(`  Response length: ${result.text.length} chars`);
    console.log(`  Tool calls: ${result.toolCallsCount}`);

    // Store session for subsequent tests
    this.session = session;
  }

  async testHybridMode(): Promise<void> {
    console.log('\n\n📦 Test 2: Hybrid Mode (Execute Lane → Hybrid)');
    console.log('-'.repeat(80));

    if (!this.session) throw new Error('testExternalTaskHandler must run first');

    // Switch to Hybrid mode
    await this.session.setLaneHandler('execute', {
      mode: 'hybrid',
      timeoutMs: 60000,
    });

    console.log('✓ Execute lane switched to Hybrid mode');
    console.log('  Tools execute locally AND emit ExternalTaskPending events');
    console.log();

    const start: number = Date.now();
    const result: AgentResult = await this.session.send("Run: echo 'hybrid mode test'");
    const duration: number = (Date.now() - start) / 1000;

    console.log(`✓ Completed in ${duration.toFixed(2)}s`);
    console.log(`  Response: ${result.text.slice(0, 120)}`);
    console.log(`  Tool calls: ${result.toolCallsCount}`);
  }

  async testDynamicLaneSwitching(): Promise<void> {
    console.log('\n\n📦 Test 3: Dynamic Lane Switching');
    console.log('-'.repeat(80));

    if (!this.session) throw new Error('testExternalTaskHandler must run first');

    // Switch back to Internal
    await this.session.setLaneHandler('execute', {
      mode: 'internal',
      timeoutMs: 60000,
    });

    console.log('✓ Execute lane switched back to Internal mode');

    const start: number = Date.now();
    const result: AgentResult = await this.session.send("Run: echo 'back to internal mode'");
    const duration: number = (Date.now() - start) / 1000;

    console.log(`✓ Completed in ${duration.toFixed(2)}s`);
    console.log(`  Response: ${result.text.slice(0, 120)}`);
  }

  // ── Run All ───────────────────────────────────────────────────────────────

  async runAll(): Promise<void> {
    console.log('🚀 A3S Code - External Task Handler Integration Test\n');
    console.log('='.repeat(80));
    console.log(`📄 Config: ${this.configPath}`);
    console.log('='.repeat(80));

    await this.testExternalTaskHandler();
    await this.testHybridMode();
    await this.testDynamicLaneSwitching();

    console.log('\n' + '='.repeat(80));
    console.log('✅ All external task handler tests completed!');
    console.log('='.repeat(80));
  }
}

async function main(): Promise<void> {
  const configPath = ExternalTaskHandlerTest.findConfig();
  const agent = await Agent.create(configPath);
  const test = new ExternalTaskHandlerTest(agent, configPath);
  await test.runAll();
}

main().catch(console.error);
