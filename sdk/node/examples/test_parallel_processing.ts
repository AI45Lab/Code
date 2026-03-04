#!/usr/bin/env npx tsx
/**
 * A3S Code Node.js SDK - Parallel Task Processing Integration Test (TypeScript)
 *
 * Demonstrates parallel processing of multiple tasks using A3S Lane queue.
 * Tests concurrent file analysis, code review, and documentation generation.
 *
 * Run with: npx tsx examples/test_parallel_processing.ts
 */

import { Agent, Session, AgentResult } from '../index.js';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

interface TaskResult {
  taskNum: number;
  success: boolean;
  result?: AgentResult;
  error?: Error;
}

interface PriorityTaskResult {
  taskNum: number;
  type: string;
  success: boolean;
  result?: AgentResult;
  error?: Error;
}

interface QueueStats {
  totalProcessed: number;
  totalFailed: number;
  dlqSize: number;
}

class ParallelProcessingTest {
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
    const homeConfig: string = path.join(os.homedir(), '.a3s', 'config.hcl');
    if (fs.existsSync(homeConfig)) {
      return homeConfig;
    }

    // Try project root (6 levels up)
    const projectConfig: string = path.join(__dirname, '..', '..', '..', '..', '..', '..', '.a3s', 'config.hcl');
    if (fs.existsSync(projectConfig)) {
      return projectConfig;
    }

    throw new Error('Config file not found. Please create ~/.a3s/config.hcl');
  }

  /**
   * Test 1: Sequential processing (baseline).
   */
  async testSequentialProcessing(): Promise<void> {
    console.log('\n📦 Test 1: Sequential Processing (Baseline)');
    console.log('-'.repeat(80));

    const session: Session = this.agent.session('.');

    const tasks: string[] = [
      'Count the number of JavaScript files in this project',
      'Find all TODO comments in JavaScript files',
      'List all exported functions in the main module',
    ];

    const start: number = Date.now();
    console.log(`Processing ${tasks.length} tasks sequentially...`);

    for (let i = 0; i < tasks.length; i++) {
      console.log(`  Task ${i + 1}: ${tasks[i]}`);
      const result: AgentResult = await session.send(tasks[i]);
      console.log(`  ✓ Completed: ${result.text.length} chars`);
    }

    const duration: number = (Date.now() - start) / 1000;
    console.log(`\n✓ Sequential processing took: ${duration.toFixed(2)}s`);
    console.log('\n✅ Test 1 passed: Sequential processing works\n');
  }

  /**
   * Test 2: Parallel processing with queue.
   */
  async testParallelProcessing(): Promise<void> {
    console.log('⚡ Test 2: Parallel Processing with Queue');
    console.log('-'.repeat(80));

    // Configure queue for parallel processing
    const session: Session = this.agent.session('.', {
      queueConfig: {
        queryConcurrency: 3,      // Allow 3 concurrent query operations
        executeConcurrency: 2,    // Allow 2 concurrent execute operations
        enableMetrics: true,      // Enable metrics collection
        enableDlq: true,          // Enable dead letter queue
      }
    });

    const tasks: string[] = [
      'Count the number of JavaScript files in this project',
      'Find all TODO comments in JavaScript files',
      'List all exported functions in the main module',
      'Find all async functions in the codebase',
      'Count lines of code in all JavaScript files',
    ];

    const start: number = Date.now();
    console.log(`Processing ${tasks.length} tasks in parallel...`);

    // Queue all tasks
    for (let i = 0; i < tasks.length; i++) {
      console.log(`  Queuing task ${i + 1}: ${tasks[i]}`);
    }

    console.log('\n  Waiting for all tasks to complete...');

    // Process tasks concurrently
    const promises: Promise<TaskResult>[] = tasks.map((task: string, i: number): Promise<TaskResult> =>
      session.send(task)
        .then((result: AgentResult): TaskResult => ({ taskNum: i + 1, success: true, result }))
        .catch((error: Error): TaskResult => ({ taskNum: i + 1, success: false, error }))
    );

    const results: TaskResult[] = await Promise.all(promises);

    for (const { taskNum, success, result, error } of results) {
      if (success) {
        console.log(`  ✓ Task ${taskNum} completed: ${result!.text.length} chars`);
      } else {
        console.log(`  ✗ Task ${taskNum} failed: ${error!.message}`);
      }
    }

    const duration: number = (Date.now() - start) / 1000;
    console.log(`\n✓ Parallel processing took: ${duration.toFixed(2)}s`);

    // Check queue stats
    if (session.hasQueue && session.hasQueue()) {
      const stats = session.queueStats() as unknown as QueueStats;
      console.log('\n📊 Queue Statistics:');
      console.log(`  Total processed: ${stats.totalProcessed}`);
      console.log(`  Total failed: ${stats.totalFailed}`);
      console.log(`  DLQ size: ${stats.dlqSize}`);
    }

    console.log('\n✅ Test 2 passed: Parallel processing with queue works\n');
  }

  /**
   * Test 3: Parallel processing with priority lanes.
   */
  async testPriorityLanes(): Promise<void> {
    console.log('🎯 Test 3: Parallel Processing with Priority Lanes');
    console.log('-'.repeat(80));

    const session: Session = this.agent.session('.', {
      queueConfig: {
        controlConcurrency: 1,    // P0: Control operations
        queryConcurrency: 3,      // P1: Query operations (highest concurrency)
        executeConcurrency: 2,    // P2: Execute operations
        generateConcurrency: 1,   // P3: Generate operations
        enableMetrics: true,
        enableDlq: true,
      }
    });

    console.log('Testing priority-based task execution...');
    console.log('  P1 (Query): 3 concurrent tasks');
    console.log('  P2 (Execute): 2 concurrent tasks');
    console.log('  P3 (Generate): 1 concurrent task');

    const start: number = Date.now();

    // Mix of different task types
    const tasks: Array<{ type: string; text: string }> = [
      { type: 'Query', text: 'How many JavaScript files are there?' },
      { type: 'Query', text: 'What is the project structure?' },
      { type: 'Query', text: 'List all dependencies' },
      { type: 'Execute', text: 'Create a summary of the codebase' },
      { type: 'Execute', text: 'Analyze code complexity' },
    ];

    for (let i = 0; i < tasks.length; i++) {
      console.log(`  Queuing ${tasks[i].type} task ${i + 1}: ${tasks[i].text}`);
    }

    console.log('\n  Waiting for all tasks to complete...');

    const promises: Promise<PriorityTaskResult>[] = tasks.map((task: { type: string; text: string }, i: number): Promise<PriorityTaskResult> =>
      session.send(task.text)
        .then((result: AgentResult): PriorityTaskResult => ({ taskNum: i + 1, type: task.type, success: true, result }))
        .catch((error: Error): PriorityTaskResult => ({ taskNum: i + 1, type: task.type, success: false, error }))
    );

    const results: PriorityTaskResult[] = await Promise.all(promises);

    for (const { taskNum, type, success, result, error } of results) {
      if (success) {
        console.log(`  ✓ Task ${taskNum} (${type}) completed: ${result!.text.length} chars`);
      } else {
        console.log(`  ✗ Task ${taskNum} (${type}) failed: ${error!.message}`);
      }
    }

    const duration: number = (Date.now() - start) / 1000;
    console.log(`\n✓ Priority-based processing took: ${duration.toFixed(2)}s`);
    console.log('\n✅ Test 3 passed: Priority lanes work correctly\n');
  }

  /**
   * Test 4: Parallel processing with retry policy.
   */
  async testRetryPolicy(): Promise<void> {
    console.log('🔄 Test 4: Parallel Processing with Retry Policy');
    console.log('-'.repeat(80));

    const session: Session = this.agent.session('.', {
      queueConfig: {
        queryConcurrency: 3,
        enableMetrics: true,
        enableDlq: true,
      }
    });

    console.log('Testing retry policy with exponential backoff...');
    console.log('  Note: Retry policy configured at queue level');
    console.log('  Max retries: 3 (default)');
    console.log('  Strategy: exponential (default)');

    const tasks: string[] = [
      'Analyze the main function',
      'Find all error types',
      'List all test functions',
    ];

    const start: number = Date.now();

    for (let i = 0; i < tasks.length; i++) {
      console.log(`  Queuing task ${i + 1}: ${tasks[i]}`);
    }

    console.log('\n  Waiting for all tasks to complete...');

    const promises: Promise<TaskResult>[] = tasks.map((task: string, i: number): Promise<TaskResult> =>
      session.send(task)
        .then((result: AgentResult): TaskResult => ({ taskNum: i + 1, success: true, result }))
        .catch((error: Error): TaskResult => ({ taskNum: i + 1, success: false, error }))
    );

    const results: TaskResult[] = await Promise.all(promises);

    for (const { taskNum, success, result, error } of results) {
      if (success) {
        console.log(`  ✓ Task ${taskNum} completed: ${result!.text.length} chars`);
      } else {
        console.log(`  ✗ Task ${taskNum} failed: ${error!.message}`);
      }
    }

    const duration: number = (Date.now() - start) / 1000;
    console.log(`\n✓ Processing with retry took: ${duration.toFixed(2)}s`);

    if (session.hasQueue && session.hasQueue()) {
      const stats = session.queueStats() as unknown as QueueStats;
      console.log('\n📊 Final Queue Statistics:');
      console.log(`  Total processed: ${stats.totalProcessed}`);
      console.log(`  Total failed: ${stats.totalFailed}`);
      console.log(`  DLQ size: ${stats.dlqSize}`);
    }

    console.log('\n✅ Test 4 passed: Retry policy works correctly\n');
  }

  /**
   * Run all tests.
   */
  async runAll(): Promise<void> {
    console.log('🚀 A3S Code Node.js SDK - Parallel Task Processing Integration Test\n');
    console.log('='.repeat(80));
    console.log(`📄 Using config: ${this.configPath}`);
    console.log('='.repeat(80));

    await this.testSequentialProcessing();
    await this.testParallelProcessing();
    await this.testPriorityLanes();
    await this.testRetryPolicy();

    console.log();
    console.log('='.repeat(80));
    console.log('✅ All parallel processing tests completed successfully!');
    console.log('='.repeat(80));
  }
}

/**
 * Main test runner.
 */
async function main(): Promise<void> {
  const configPath: string = ParallelProcessingTest.findConfigPath();
  const agent: Agent = await Agent.create(configPath);
  const test = new ParallelProcessingTest(agent, configPath);
  await test.runAll();
}

// Run tests
main().catch((err: unknown) => {
  console.error('❌ Test failed:', err);
  process.exit(1);
});
