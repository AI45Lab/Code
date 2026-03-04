#!/usr/bin/env npx tsx
/**
 * A3S Code Node.js SDK — Agent Teams Comprehensive Integration Test (TypeScript)
 *
 * Real-world multi-task decomposition and multi-agent parallel execution:
 *
 *   Scenario 0: Task Board Primitives (no LLM)
 *   Scenario 1: Code Quality Audit (Lead → Worker → Reviewer)
 *   Scenario 2: Parallel Workers — Feature Planning (2 workers)
 *   Scenario 3: Team Without Reviewer
 *   Scenario 4: Manual Task Posting + TeamRunner Execution
 *
 * Run with: npx tsx examples/test_agent_teams_comprehensive.ts
 */

import {
  Agent,
  Session,
  Team,
  TeamRunner,
  TeamTaskBoard,
  TeamTask,
  TeamRunResult,
  BoardStats,
  TeamConfig,
} from '../index.js';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

interface ScenarioEntry {
  name: string;
  fn: () => Promise<void> | void;
}

class AgentTeamsComprehensiveTest {
  private readonly agent: Agent;
  private readonly configPath: string;
  private readonly workspace: string;

  constructor(agent: Agent, configPath: string, workspace: string) {
    this.agent = agent;
    this.configPath = configPath;
    this.workspace = workspace;
  }

  // ── Static Helpers ───────────────────────────────────────────────────────

  static findConfig(): string {
    const homeConfig = path.join(os.homedir(), '.a3s', 'config.hcl');
    if (fs.existsSync(homeConfig)) return homeConfig;

    let p = path.resolve(__dirname);
    for (let i = 0; i < 10; i++) {
      const c = path.join(p, '.a3s', 'config.hcl');
      if (fs.existsSync(c)) return c;
      const parent = path.dirname(p);
      if (parent === p) break;
      p = parent;
    }
    throw new Error('Config not found');
  }

  static pass(label: string): void {
    console.log(`  PASS  ${label}`);
  }

  static printBoard(board: TeamTaskBoard): void {
    const s: BoardStats = board.stats();
    console.log(
      `  Board: total=${board.len}, open=${s.open}, in_progress=${s.inProgress}, ` +
      `in_review=${s.inReview}, done=${s.done}, rejected=${s.rejected}`
    );
  }

  static printResults(result: TeamRunResult, verbose = true): void {
    console.log(
      `  Done: ${result.doneTasks.length}, Rejected: ${result.rejectedTasks.length}, ` +
      `Rounds: ${result.rounds}`
    );
    if (verbose) {
      for (const t of result.doneTasks) {
        const desc = t.description.length > 60
          ? t.description.slice(0, 60) + '...'
          : t.description;
        const snippet = (t.result ?? '').replace(/\n/g, ' ').slice(0, 100);
        console.log(`    [${t.id}] ${desc}`);
        console.log(`      -> ${snippet}${(t.result ?? '').length > 100 ? '...' : ''}`);
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Scenario 0: Task Board Primitives (No LLM)
  // ═══════════════════════════════════════════════════════════════════════════

  testTaskBoardPrimitives(): void {
    console.log('\n== Scenario 0: Task Board Primitives (No LLM) ==');
    console.log('-'.repeat(70));

    const config: TeamConfig = { maxTasks: 20, maxRounds: 5, pollIntervalMs: 10 };
    const team = new Team('board-test', config);
    team.addMember('pm', 'lead');
    team.addMember('dev-1', 'worker');
    team.addMember('dev-2', 'worker');
    team.addMember('qa', 'reviewer');

    if (team.memberCount !== 4) {
      throw new Error(`Expected 4 members, got ${team.memberCount}`);
    }
    AgentTeamsComprehensiveTest.pass(`Team created with ${team.memberCount} members`);

    const board: TeamTaskBoard = team.taskBoard();

    // PM posts sprint backlog
    const tasks: string[] = [
      'Implement user registration API endpoint',
      'Add input validation middleware',
      'Write database migration for users table',
      'Create unit tests for registration flow',
      'Set up CI pipeline for auth service',
    ];

    const ids: string[] = [];
    for (const desc of tasks) {
      const tid = board.post(desc, 'pm');
      if (!tid) throw new Error(`Failed to post: ${desc}`);
      ids.push(tid);
    }
    AgentTeamsComprehensiveTest.pass(`Posted ${ids.length} tasks to sprint board`);
    AgentTeamsComprehensiveTest.printBoard(board);

    // dev-1 claims and completes first 2
    const t1: TeamTask | null = board.claim('dev-1');
    if (!t1 || t1.status !== 'in_progress') throw new Error('dev-1 should claim task');
    const t2: TeamTask | null = board.claim('dev-1');
    if (!t2) throw new Error('dev-1 should claim second task');
    board.complete(t1.id, 'POST /api/register with bcrypt hashing');
    board.complete(t2.id, 'Zod schema validation as Express middleware');
    AgentTeamsComprehensiveTest.pass('dev-1 completed 2 tasks');

    // dev-2 claims remaining
    let claimed = 0;
    while (true) {
      const t: TeamTask | null = board.claim('dev-2');
      if (!t) break;
      board.complete(t.id, `Completed: ${t.description}`);
      claimed++;
    }
    AgentTeamsComprehensiveTest.pass(`dev-2 completed ${claimed} tasks`);

    // QA reviews
    const inReview: TeamTask[] = board.byStatus('in_review');
    if (inReview.length !== 5) {
      throw new Error(`Expected 5 in_review, got ${inReview.length}`);
    }

    board.reject(ids[0]);
    for (let i = 1; i < ids.length; i++) board.approve(ids[i]);
    AgentTeamsComprehensiveTest.pass('QA reviewed all: 1 rejected, 4 approved');

    // Retry rejected task
    const retry: TeamTask | null = board.claim('dev-1');
    if (!retry || retry.id !== ids[0]) throw new Error('Should re-claim rejected task');
    board.complete(retry.id, 'Fixed: added email uniqueness + rate limiting');
    board.approve(retry.id);
    AgentTeamsComprehensiveTest.pass('Rejected task retried and approved');

    const done: TeamTask[] = board.byStatus('done');
    if (done.length !== 5) throw new Error(`Expected 5 done, got ${done.length}`);
    AgentTeamsComprehensiveTest.printBoard(board);
    AgentTeamsComprehensiveTest.pass('All 5 tasks done');

    const dev1Tasks: TeamTask[] = board.byAssignee('dev-1');
    const dev2Tasks: TeamTask[] = board.byAssignee('dev-2');
    if (dev1Tasks.length + dev2Tasks.length !== 5) throw new Error('Assignment mismatch');
    AgentTeamsComprehensiveTest.pass(`Assignment tracking: dev-1=${dev1Tasks.length}, dev-2=${dev2Tasks.length}`);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Scenario 1: Code Quality Audit
  // ═══════════════════════════════════════════════════════════════════════════

  async testCodeQualityAudit(): Promise<void> {
    console.log('\n== Scenario 1: Code Quality Audit ==');
    console.log('-'.repeat(70));

    const team = new Team('code-audit', { maxTasks: 10, maxRounds: 8, pollIntervalMs: 200 });
    team.addMember('tech-lead', 'lead');
    team.addMember('auditor', 'worker');
    team.addMember('senior-dev', 'reviewer');

    const runner = new TeamRunner(team);
    runner.bindSession('tech-lead', this.agent.session(this.workspace, { permissive: true }));
    runner.bindSession('auditor', this.agent.session(this.workspace, { permissive: true }));
    runner.bindSession('senior-dev', this.agent.session(this.workspace, { permissive: true }));

    const goal =
      'Audit this project workspace for code quality. ' +
      'Decompose into exactly 3 tasks: ' +
      '1) check for potential security issues, ' +
      '2) identify performance bottlenecks, ' +
      '3) check code style consistency. ' +
      'Each task should produce a brief 2-3 sentence summary.';

    console.log(`  Goal: ${goal.slice(0, 80)}...`);
    const t0 = Date.now();
    const result: TeamRunResult = await runner.runUntilDone(goal);
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

    console.log(`  Completed in ${elapsed}s`);
    AgentTeamsComprehensiveTest.printResults(result);

    if (result.doneTasks.length < 1) throw new Error('At least 1 task should complete');
    AgentTeamsComprehensiveTest.pass(`${result.doneTasks.length} audit tasks completed, ${result.rejectedTasks.length} rejected`);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Scenario 2: Parallel Workers — Feature Planning
  // ═══════════════════════════════════════════════════════════════════════════

  async testParallelWorkers(): Promise<void> {
    console.log('\n== Scenario 2: Parallel Workers — Feature Planning ==');
    console.log('-'.repeat(70));

    const team = new Team('feature-team', { maxTasks: 10, maxRounds: 8, pollIntervalMs: 200 });
    team.addMember('architect', 'lead');
    team.addMember('backend-dev', 'worker');
    team.addMember('frontend-dev', 'worker');
    team.addMember('code-reviewer', 'reviewer');

    const runner = new TeamRunner(team);
    runner.bindSession('architect', this.agent.session(this.workspace, { permissive: true }));
    runner.bindSession('backend-dev', this.agent.session(this.workspace, { permissive: true }));
    runner.bindSession('frontend-dev', this.agent.session(this.workspace, { permissive: true }));
    runner.bindSession('code-reviewer', this.agent.session(this.workspace, { permissive: true }));

    const board: TeamTaskBoard = runner.taskBoard();

    const goal =
      'Plan a user authentication feature for a web application. ' +
      'Decompose into exactly 2 tasks: ' +
      '1) Design the backend API schema (endpoints, request/response format), ' +
      '2) Design the frontend login form component (props, states, events). ' +
      'Each task should produce a concise spec (3-5 lines).';

    console.log(`  Goal: ${goal.slice(0, 80)}...`);
    console.log('  Workers: backend-dev, frontend-dev (parallel)');
    const t0 = Date.now();
    const result: TeamRunResult = await runner.runUntilDone(goal);
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

    console.log(`  Completed in ${elapsed}s`);
    AgentTeamsComprehensiveTest.printResults(result);
    AgentTeamsComprehensiveTest.printBoard(board);

    const reviewTasks: TeamTask[] = board.byStatus('in_review');
    const total = result.doneTasks.length + reviewTasks.length;
    if (total < 1) throw new Error('Workers should have completed at least 1 task');
    AgentTeamsComprehensiveTest.pass(`Parallel workers completed ${total} tasks`);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Scenario 3: Team Without Reviewer
  // ═══════════════════════════════════════════════════════════════════════════

  async testNoReviewer(): Promise<void> {
    console.log('\n== Scenario 3: Team Without Reviewer ==');
    console.log('-'.repeat(70));

    const team = new Team('no-review', { maxTasks: 5, maxRounds: 5, pollIntervalMs: 100 });
    team.addMember('lead', 'lead');
    team.addMember('worker', 'worker');

    const runner = new TeamRunner(team);
    runner.bindSession('lead', this.agent.session(this.workspace, { permissive: true }));
    runner.bindSession('worker', this.agent.session(this.workspace, { permissive: true }));

    const board: TeamTaskBoard = runner.taskBoard();

    const goal =
      'List exactly 2 simple math facts. ' +
      'Task 1: What is 7 * 8? ' +
      'Task 2: What is 12 + 15?';

    console.log(`  Goal: ${goal}`);
    const result: TeamRunResult = await runner.runUntilDone(goal);

    const inReview: TeamTask[] = board.byStatus('in_review');
    const done: TeamTask[] = board.byStatus('done');
    console.log(`  In-review: ${inReview.length}, Done: ${done.length}`);

    if (inReview.length + done.length < 1) throw new Error('At least 1 task should be processed');
    AgentTeamsComprehensiveTest.pass(`No-reviewer team: ${inReview.length} in-review, ${done.length} done`);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Scenario 4: Manual Task Posting + TeamRunner Execution
  // ═══════════════════════════════════════════════════════════════════════════

  async testManualPostThenRun(): Promise<void> {
    console.log('\n== Scenario 4: Manual Task Posting + TeamRunner Execution ==');
    console.log('-'.repeat(70));

    const team = new Team('manual-post', { maxTasks: 20, maxRounds: 8, pollIntervalMs: 200 });
    team.addMember('pm', 'lead');
    team.addMember('dev-1', 'worker');
    team.addMember('dev-2', 'worker');
    team.addMember('qa', 'reviewer');

    const board: TeamTaskBoard = team.taskBoard();
    board.post('What is the capital of France? Reply in one word.', 'pm');
    board.post('What is the largest planet in our solar system? Reply in one word.', 'pm');
    board.post('What is 100 divided by 4? Reply with just the number.', 'pm');
    board.post('Name one programming language created by Guido van Rossum. One word.', 'pm');
    console.log('  Manually posted 4 tasks');
    AgentTeamsComprehensiveTest.printBoard(board);

    const runner = new TeamRunner(team);
    runner.bindSession('pm', this.agent.session(this.workspace, { permissive: true }));
    runner.bindSession('dev-1', this.agent.session(this.workspace, { permissive: true }));
    runner.bindSession('dev-2', this.agent.session(this.workspace, { permissive: true }));
    runner.bindSession('qa', this.agent.session(this.workspace, { permissive: true }));

    const result: TeamRunResult = await runner.runUntilDone(
      'Answer the questions already posted on the task board. ' +
      'Decompose into 0 additional tasks — all work is already posted.'
    );

    const finalBoard: TeamTaskBoard = runner.taskBoard();
    const done: TeamTask[] = finalBoard.byStatus('done');
    const inReview: TeamTask[] = finalBoard.byStatus('in_review');
    console.log(`  Final: total=${finalBoard.len}, done=${done.length}, in_review=${inReview.length}`);
    AgentTeamsComprehensiveTest.printResults(result, false);

    AgentTeamsComprehensiveTest.pass(`Manual + auto: ${result.doneTasks.length} done, ${result.rejectedTasks.length} rejected`);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Run All
  // ═══════════════════════════════════════════════════════════════════════════

  async runAll(): Promise<void> {
    console.log('='.repeat(70));
    console.log('  A3S Code — Agent Teams Comprehensive Integration Tests (TypeScript)');
    console.log(`  Config: ${this.configPath}`);
    console.log('='.repeat(70));

    const scenarios: ScenarioEntry[] = [
      { name: 'Scenario 0: Task Board Primitives', fn: () => this.testTaskBoardPrimitives() },
      { name: 'Scenario 1: Code Quality Audit', fn: () => this.testCodeQualityAudit() },
      { name: 'Scenario 2: Parallel Workers', fn: () => this.testParallelWorkers() },
      { name: 'Scenario 3: No Reviewer', fn: () => this.testNoReviewer() },
      { name: 'Scenario 4: Manual Post + Run', fn: () => this.testManualPostThenRun() },
    ];

    let passed = 0;
    let failed = 0;

    for (const { name, fn } of scenarios) {
      try {
        await fn();
        passed++;
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        console.error(`\n  FAILED [${name}]: ${msg}`);
        failed++;
      }
    }

    try { fs.rmSync(this.workspace, { recursive: true, force: true }); } catch {}

    console.log('\n' + '='.repeat(70));
    if (failed === 0) {
      console.log(`  ALL ${passed} SCENARIOS PASSED`);
    } else {
      console.log(`  ${passed} passed, ${failed} FAILED`);
      process.exit(1);
    }
    console.log('='.repeat(70));
  }
}

async function main(): Promise<void> {
  const configPath = AgentTeamsComprehensiveTest.findConfig();
  const agent: Agent = await Agent.create(configPath);
  const workspace: string = fs.mkdtempSync(path.join(os.tmpdir(), 'a3s-teams-'));
  const test = new AgentTeamsComprehensiveTest(agent, configPath, workspace);
  await test.runAll();
}

main().catch((e: unknown) => {
  console.error('Fatal:', e);
  process.exit(1);
});
