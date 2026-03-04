/**
 * Lane-Based Priority Preemption Test with Real LLM
 *
 * Demonstrates how the lane-based priority system allows high-priority tasks
 * to preempt queued low-priority tasks:
 *
 * 1. Constrain concurrency so tasks must queue up
 * 2. Submit multiple low-priority tasks (Execute lane: bash commands)
 * 3. Submit a high-priority task later (Query lane: read/grep)
 * 4. Observe that the high-priority task executes before queued low-priority tasks
 *
 * Lane priorities (lower = higher priority):
 *   Control (P0) > Query (P1) > Execute (P2) > Generate (P3)
 *
 * Run with: npx ts-node examples/test_task_priority.ts
 */

import { Agent, Session, AgentResult, SessionQueueConfigOptions } from "../index.js";
import * as path from "path";
import * as os from "os";
import * as fs from "fs";

class CompletionRecord {
  name: string;
  lane: string;
  submittedAt: number;
  completedAt: number;

  constructor(name: string, lane: string, submittedAt: number, completedAt: number = 0) {
    this.name = name;
    this.lane = lane;
    this.submittedAt = submittedAt;
    this.completedAt = completedAt;
  }
}

class TaskPriorityTest {
  private readonly agent: Agent;
  private readonly configPath: string;

  constructor(agent: Agent, configPath: string) {
    this.agent = agent;
    this.configPath = configPath;
  }

  static findConfig(): string {
    const homeConfig: string = path.join(os.homedir(), ".a3s", "config.hcl");
    if (fs.existsSync(homeConfig)) return homeConfig;
    throw new Error("Config not found. Create ~/.a3s/config.hcl");
  }

  static elapsed(start: number): number {
    return (performance.now() - start) / 1000;
  }

  static printCompletionOrder(completions: CompletionRecord[]): CompletionRecord[] {
    const sorted: CompletionRecord[] = [...completions].sort(
      (a: CompletionRecord, b: CompletionRecord) => a.completedAt - b.completedAt
    );
    console.log("\n  --- Completion Order ---");
    sorted.forEach((rec: CompletionRecord, i: number) => {
      const marker: string = rec.lane.includes("P1") ? "[!]" : "  ";
      console.log(
        `  ${marker} ${i + 1}. ${rec.name} [${rec.lane}] -- submitted ${rec.submittedAt.toFixed(2)}s, completed ${rec.completedAt.toFixed(2)}s`
      );
    });
    return sorted;
  }

  // ============================================================================
  // Test 1: Query (P1) preempts Execute (P2)
  // ============================================================================
  async testQueryPreemptsExecute(): Promise<void> {
    console.log("\nTest 1: Query (P1) Preempts Execute (P2)");
    console.log("-".repeat(80));
    console.log("  Execute lane concurrency: 1 (tasks must queue)");
    console.log(
      "  Query lane concurrency: 2 (higher priority, separate capacity)"
    );
    console.log("  Submit 3 Execute tasks first, then 1 Query task");
    console.log(
      "  Expected: Query task completes before remaining Execute tasks\n"
    );

    const session: Session = this.agent.session(".", {
      queueConfig: {
        executeMaxConcurrency: 1, // Bottleneck
        queryMaxConcurrency: 2, // Higher priority, own capacity
        generateMaxConcurrency: 1,
        enableMetrics: true,
      } as SessionQueueConfigOptions,
      autoApprove: true,
    });

    const start: number = performance.now();
    const completions: CompletionRecord[] = [];

    async function runTask(name: string, prompt: string, laneLabel: string): Promise<AgentResult> {
      const submittedAt: number = TaskPriorityTest.elapsed(start);
      const marker: string = laneLabel.includes("P1") ? "[!]" : ">>";
      console.log(
        `  [${submittedAt.toFixed(2).padStart(6)}s] ${marker} Submitting: ${name} (${laneLabel})`
      );

      const result: AgentResult = await session.send(prompt);
      const completedAt: number = TaskPriorityTest.elapsed(start);
      completions.push(
        new CompletionRecord(name, laneLabel, submittedAt, completedAt)
      );

      const chars: number = result?.text?.length || 0;
      const doneMarker: string = laneLabel.includes("P1") ? "[!]" : "[ok]";
      console.log(
        `  [${completedAt.toFixed(2).padStart(6)}s] ${doneMarker} Completed: ${name} (${chars} chars)`
      );
      return result;
    }

    // Submit 3 Execute-lane tasks (bash -> Execute lane P2)
    const tasks: Promise<AgentResult>[] = [];
    for (let i = 1; i <= 3; i++) {
      const prompt: string = `Run this bash command and tell me the output: echo 'Task ${i} started' && sleep 1 && echo 'Task ${i} done'`;
      tasks.push(runTask(`Execute-${i}`, prompt, "Execute (P2)"));
      await new Promise<void>((r) => setTimeout(r, 200));
    }

    // Wait for queue to fill
    await new Promise<void>((r) => setTimeout(r, 500));

    // Submit Query-lane task (read -> Query lane P1)
    console.log();
    tasks.push(
      runTask(
        "Query-Urgent",
        "Read the Cargo.toml file and tell me the package name and version",
        "Query (P1)"
      )
    );

    await Promise.allSettled(tasks);
    TaskPriorityTest.printCompletionOrder(completions);
    console.log("\nTest 1 completed");
  }

  // ============================================================================
  // Test 2: Multi-level priority
  // ============================================================================
  async testMultiLevelPriority(): Promise<void> {
    console.log(
      "\n\nTest 2: Multi-Level Priority (Query P1 vs Execute P2)"
    );
    console.log("-".repeat(80));
    console.log("  All lanes constrained to concurrency=1");
    console.log("  Submit Execute task first, then Query task");
    console.log("  Expected: Query task scheduled with higher priority\n");

    const session: Session = this.agent.session(".", {
      queueConfig: {
        controlMaxConcurrency: 1,
        queryMaxConcurrency: 1,
        executeMaxConcurrency: 1,
        generateMaxConcurrency: 1,
        enableMetrics: true,
      } as SessionQueueConfigOptions,
      autoApprove: true,
    });

    const start: number = performance.now();
    const completions: CompletionRecord[] = [];

    async function runTask(name: string, prompt: string, laneLabel: string): Promise<AgentResult> {
      const submittedAt: number = TaskPriorityTest.elapsed(start);
      console.log(
        `  [${submittedAt.toFixed(2).padStart(6)}s] >> ${name} (${laneLabel})`
      );
      const result: AgentResult = await session.send(prompt);
      const completedAt: number = TaskPriorityTest.elapsed(start);
      completions.push(
        new CompletionRecord(name, laneLabel, submittedAt, completedAt)
      );
      console.log(
        `  [${completedAt.toFixed(2).padStart(6)}s] [ok] ${name} completed`
      );
      return result;
    }

    // Execute task first (lower priority)
    const execTask: Promise<AgentResult> = runTask(
      "Execute-Task",
      "Run: echo 'execute-lane task' && sleep 1 && echo 'done'",
      "Execute (P2)"
    );
    await new Promise<void>((r) => setTimeout(r, 300));

    // Query task (higher priority)
    const queryTask: Promise<AgentResult> = runTask(
      "Query-Task",
      "Read the Cargo.toml file and show the first 5 lines",
      "Query (P1)"
    );

    await Promise.allSettled([execTask, queryTask]);
    TaskPriorityTest.printCompletionOrder(completions);
    console.log("\nTest 2 completed");
  }

  // ============================================================================
  // Test 3: Late urgent task insertion
  // ============================================================================
  async testLateUrgentInsertion(): Promise<void> {
    console.log("\n\nTest 3: Late Urgent Task Insertion");
    console.log("-".repeat(80));
    console.log("  Execute concurrency: 1, Query concurrency: 2");
    console.log(
      "  Submit 4 slow Execute tasks, then 1 urgent Query task at 2s mark"
    );
    console.log(
      "  Expected: Urgent task completes before remaining Execute tasks\n"
    );

    const session: Session = this.agent.session(".", {
      queueConfig: {
        executeMaxConcurrency: 1,
        queryMaxConcurrency: 2,
        generateMaxConcurrency: 1,
        enableMetrics: true,
      } as SessionQueueConfigOptions,
      autoApprove: true,
    });

    const start: number = performance.now();
    const completions: CompletionRecord[] = [];

    async function runTask(name: string, prompt: string, laneLabel: string): Promise<AgentResult> {
      const submittedAt: number = TaskPriorityTest.elapsed(start);
      const marker: string = laneLabel.includes("P1") ? "[!]" : ">>";
      console.log(
        `  [${submittedAt.toFixed(2).padStart(6)}s] ${marker} ${name}`
      );
      const result: AgentResult = await session.send(prompt);
      const completedAt: number = TaskPriorityTest.elapsed(start);
      completions.push(
        new CompletionRecord(name, laneLabel, submittedAt, completedAt)
      );
      const doneMarker: string = laneLabel.includes("P1") ? "[!]" : "[ok]";
      console.log(
        `  [${completedAt.toFixed(2).padStart(6)}s] ${doneMarker} ${name} completed`
      );
      return result;
    }

    // Submit 4 slow Execute tasks
    const tasks: Promise<AgentResult>[] = [];
    for (let i = 1; i <= 4; i++) {
      const prompt: string = `Run: echo 'slow task ${i} start' && sleep 2 && echo 'slow task ${i} end'`;
      tasks.push(runTask(`SlowExec-${i}`, prompt, "Execute (P2)"));
      await new Promise<void>((r) => setTimeout(r, 100));
    }

    // Wait for queue to fill
    console.log("\n  Waiting 2s for queue to fill up...\n");
    await new Promise<void>((r) => setTimeout(r, 2000));

    // Print queue stats
    try {
      const stats: Record<string, unknown> = await session.queueStats();
      console.log(
        `  Queue stats: pending=${stats.totalPending}, active=${stats.totalActive}`
      );
    } catch {
      console.log("  Queue stats: (unavailable)");
    }

    // Inject urgent Query task
    tasks.push(
      runTask(
        "UrgentQuery",
        "Use grep to search for 'name' in Cargo.toml and tell me the result",
        "Query (P1)"
      )
    );

    await Promise.allSettled(tasks);

    // Print completion order
    const sorted: CompletionRecord[] = TaskPriorityTest.printCompletionOrder(completions);

    // Check preemption
    const urgent: CompletionRecord | undefined = sorted.find((r: CompletionRecord) => r.name === "UrgentQuery");
    const lastExec: CompletionRecord | undefined = [...sorted].reverse().find((r: CompletionRecord) => r.lane.includes("P2"));

    if (urgent && lastExec) {
      if (urgent.completedAt < lastExec.completedAt) {
        console.log(
          `\n  Priority preemption confirmed: UrgentQuery (${urgent.completedAt.toFixed(2)}s) before last Execute (${lastExec.completedAt.toFixed(2)}s)`
        );
      } else {
        console.log(
          `\n  UrgentQuery (${urgent.completedAt.toFixed(2)}s) after last Execute (${lastExec.completedAt.toFixed(2)}s)`
        );
        console.log(
          "     This can happen if Execute tasks were already running"
        );
      }
    }

    console.log("\nTest 3 completed");
  }

  // ============================================================================
  // Run all tests
  // ============================================================================
  async runAll(): Promise<void> {
    console.log("A3S Code - Lane-Based Priority Preemption Test (Node.js)\n");
    console.log("=".repeat(80));
    console.log(`Using config: ${this.configPath}`);
    console.log("=".repeat(80));

    await this.testQueryPreemptsExecute();
    await this.testMultiLevelPriority();
    await this.testLateUrgentInsertion();

    console.log(`\n${"=".repeat(80)}`);
    console.log("All lane-based priority preemption tests completed!");
    console.log("=".repeat(80));
  }
}

async function main(): Promise<void> {
  const configPath: string = TaskPriorityTest.findConfig();
  const agent: Agent = await Agent.create(configPath);
  const test = new TaskPriorityTest(agent, configPath);
  await test.runAll();
}

main().catch(console.error);
