#!/usr/bin/env npx tsx
/**
 * A3S Code Node.js SDK - Agent Teams Integration Test
 *
 * Demonstrates multi-agent team coordination using TeamRunner:
 *   - Lead decomposes a goal into tasks via LLM JSON response
 *   - Workers concurrently claim and execute tasks
 *   - Reviewer approves or rejects completed work
 *   - Rejected tasks are re-queued for retry
 *
 * Run with: npx tsx examples/test_agent_teams.ts
 */

import {
  Agent,
  Session,
  Team,
  TeamRunner,
  TeamTaskBoard,
  TeamTask,
  TeamRunResult,
  TeamConfig,
  BoardStats,
} from '../index.js';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

class AgentTeamsTest {
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
    if (fs.existsSync(homeConfig)) return homeConfig;

    const projectConfig: string = path.join(
      __dirname, '..', '..', '..', '..', '..', '..', '.a3s', 'config.hcl'
    );
    if (fs.existsSync(projectConfig)) return projectConfig;

    throw new Error('Config file not found. Please create ~/.a3s/config.hcl');
  }

  /**
   * Print a concise board summary.
   */
  static printBoard(board: TeamTaskBoard): void {
    const stats: BoardStats = board.stats();
    console.log(
      `  Board: total=${board.len}, open=${stats.open}, in_progress=${stats.inProgress}, ` +
      `in_review=${stats.inReview}, done=${stats.done}, rejected=${stats.rejected}`
    );
  }

  /**
   * Test 1: Manual task board API (no LLM required).
   */
  async testManualCoordination(): Promise<void> {
    console.log('\n Test 1: Manual Task Board Coordination');
    console.log('-'.repeat(80));

    const team: Team = new Team('manual-test');
    team.addMember('lead', 'lead');
    team.addMember('worker-1', 'worker');
    team.addMember('reviewer', 'reviewer');

    const board: TeamTaskBoard = team.taskBoard();

    // Lead posts tasks
    const id1: string | null = board.post('Refactor auth to JWT', 'lead');
    const id2: string | null = board.post('Write integration tests', 'lead');
    const id3: string | null = board.post('Update API docs', 'lead');
    if (!id1 || !id2 || !id3) throw new Error('Tasks should be posted');

    console.log('  Posted 3 tasks. Board state:');
    AgentTeamsTest.printBoard(board);

    // Worker claims first task
    const task: TeamTask | null = board.claim('worker-1');
    if (!task) throw new Error('Worker should claim a task');
    if (task.status !== 'in_progress') throw new Error(`Expected in_progress, got ${task.status}`);
    console.log(`  Worker claimed: [${task.id}] ${task.description}`);

    // Worker completes it
    board.complete(task.id, 'Auth refactored to use JWT with RS256 keys');
    const completed: TeamTask | null = board.get(task.id);
    if (!completed || completed.status !== 'in_review') throw new Error(`Expected in_review, got ${completed?.status}`);
    console.log(`  Task submitted for review -> status: ${completed.status}`);

    // Reviewer rejects
    board.reject(task.id);
    const rejected: TeamTask | null = board.get(task.id);
    if (!rejected || rejected.status !== 'rejected') throw new Error(`Expected rejected, got ${rejected?.status}`);
    console.log(`  Reviewer rejected -> status: ${rejected.status}`);

    // Worker re-claims and completes the rejected task
    const retried: TeamTask | null = board.claim('worker-1');
    if (!retried || retried.id !== task.id) throw new Error('Should reclaim rejected task');
    board.complete(task.id, 'Auth refactored with proper token rotation');
    board.approve(task.id);
    const approved: TeamTask | null = board.get(task.id);
    if (!approved || approved.status !== 'done') throw new Error(`Expected done, got ${approved?.status}`);
    console.log(`  After retry: reviewer approved -> status: ${approved.status}`);

    // Verify by_status
    const doneTasks: TeamTask[] = board.byStatus('done');
    const openTasks: TeamTask[] = board.byStatus('open');
    if (doneTasks.length !== 1) throw new Error('Expected 1 done task');
    if (openTasks.length !== 2) throw new Error('Expected 2 open tasks');
    console.log(`  Done: ${doneTasks.length}, Open: ${openTasks.length}`);

    console.log('\nTest 1 passed: Manual task board coordination works\n');
  }

  /**
   * Test 2: Full TeamRunner workflow with real LLM.
   */
  async testTeamRunnerWorkflow(): Promise<void> {
    console.log('\n Test 2: TeamRunner — Automated Lead -> Worker -> Reviewer Workflow');
    console.log('-'.repeat(80));

    const config: TeamConfig = {
      maxTasks: 20,
      channelBuffer: 128,
      maxRounds: 8,
      pollIntervalMs: 100,
    };
    const team: Team = new Team('code-review-team', config);
    team.addMember('lead', 'lead');
    team.addMember('worker-1', 'worker');
    team.addMember('reviewer', 'reviewer');

    console.log(`  Team created with ${team.memberCount} members`);

    const runner: TeamRunner = new TeamRunner(team);
    runner.bindSession('lead',     this.agent.session('.'));
    runner.bindSession('worker-1', this.agent.session('.'));
    runner.bindSession('reviewer', this.agent.session('.'));

    const goal: string =
      'Audit this JavaScript codebase for code quality issues. ' +
      'Identify and document the top 3 most important improvements.';
    console.log(`  Goal: ${goal}`);
    console.log('  Running team workflow (Lead decomposes -> Workers execute -> Reviewer approves)...');

    const start: number = Date.now();
    const result: TeamRunResult = await runner.runUntilDone(goal);
    const elapsed: string = ((Date.now() - start) / 1000).toFixed(1);

    console.log(`\n  Completed in ${elapsed}s, ${result.rounds} reviewer rounds`);
    console.log(`  Done tasks: ${result.doneTasks.length}`);
    console.log(`  Rejected tasks: ${result.rejectedTasks.length}`);

    for (const task of result.doneTasks) {
      const snippet: string = (task.result || '').slice(0, 120).replace(/\n/g, ' ');
      const ellipsis: string = (task.result || '').length > 120 ? '...' : '';
      console.log(`\n  [${task.id.slice(0, 8)}] ${task.description}`);
      console.log(`    Result: ${snippet}${ellipsis}`);
    }

    if (result.doneTasks.length === 0) throw new Error('At least one task should be completed');
    console.log('\nTest 2 passed: TeamRunner executed end-to-end workflow\n');
  }

  /**
   * Test 3: TeamRunner without a reviewer — tasks reach InReview state.
   */
  async testTeamRunnerNoReviewer(): Promise<void> {
    console.log('\n Test 3: TeamRunner Without Reviewer (tasks reach InReview)');
    console.log('-'.repeat(80));

    const config: TeamConfig = { maxRounds: 3, maxTasks: 10, channelBuffer: 128, pollIntervalMs: 50 };
    const team: Team = new Team('no-reviewer-team', config);
    team.addMember('lead', 'lead');
    team.addMember('worker-1', 'worker');
    // No reviewer bound

    const runner: TeamRunner = new TeamRunner(team);
    runner.bindSession('lead',     this.agent.session('.'));
    runner.bindSession('worker-1', this.agent.session('.'));

    const board: TeamTaskBoard = runner.taskBoard();
    const result: TeamRunResult = await runner.runUntilDone('Count the number of JavaScript files in this project');

    const inReview: TeamTask[] = board.byStatus('in_review');
    const done: TeamTask[] = board.byStatus('done');
    console.log(`  InReview: ${inReview.length}, Done: ${done.length}`);

    if (inReview.length + done.length === 0) throw new Error('Tasks should have been executed');
    console.log('\nTest 3 passed: Team runs correctly without a reviewer\n');
  }

  async runAll(): Promise<void> {
    console.log('='.repeat(80));
    console.log('  A3S Code — Agent Teams Integration Tests');
    console.log('='.repeat(80));

    await this.testManualCoordination();
    await this.testTeamRunnerWorkflow();
    await this.testTeamRunnerNoReviewer();

    console.log('='.repeat(80));
    console.log('  All agent team tests passed!');
    console.log('='.repeat(80));
  }
}

async function main(): Promise<void> {
  try {
    const configPath = AgentTeamsTest.findConfigPath();
    const agent = await Agent.create(configPath);
    const test = new AgentTeamsTest(agent, configPath);
    await test.runAll();
  } catch (err) {
    const message: string = err instanceof Error ? err.message : String(err);
    console.error(`\nTest failed: ${message}`);
    process.exit(1);
  }
}

main();
