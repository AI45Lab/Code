/**
 * Programmable orchestration via the A3S Code Node SDK.
 *
 *   session.parallel(specs)             — fan out steps, results in input order
 *   session.pipeline(items, stages)     — per-item chains, no inter-stage barrier
 *   session.parallelResumable(specs, id) — journaled, resumable fan-out
 *
 * Config is read from .a3s/config.acl at runtime (never hardcoded).
 * Run: node sdk/node/examples/orchestration/parallel-pipeline.mjs
 */
import { createRequire } from 'node:module';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { Agent } = require('@a3s-lab/code');

function repoConfigPath() {
  let dir = dirname(fileURLToPath(import.meta.url));
  for (let i = 0; i < 10; i++) {
    const candidate = join(dir, '.a3s', 'config.acl');
    if (existsSync(candidate)) return candidate;
    dir = dirname(dir);
  }
  throw new Error('could not locate .a3s/config.acl above this example');
}

async function main() {
  const agent = await Agent.create(repoConfigPath());
  const session = agent.session('.', {});

  // 1. parallel — independent steps; outcomes in input order.
  const outcomes = await session.parallel([
    { taskId: 'langs', agent: 'general', description: 'list', prompt: 'Name three systems languages.', maxSteps: 2 },
    { taskId: 'safe', agent: 'general', description: 'classify', prompt: 'Is Rust memory-safe without a GC? yes/no.', maxSteps: 2 },
  ]);
  for (const o of outcomes) console.log(`[parallel] ${o.taskId}: success=${o.success}`);

  // 2. pipeline — stage 2 builds on stage 1's output. Return null to stop a
  //    chain. A stage callback MUST NOT throw (return null on error).
  const results = await session.pipeline(
    ['the Rust programming language'],
    [
      (ctx) => ({ taskId: 'sum', agent: 'general', description: 'summarize', prompt: `In one sentence, what is ${ctx.item}?`, maxSteps: 2 }),
      (ctx) => ({ taskId: 'cls', agent: 'general', description: 'classify',
                  prompt: `Reply YES or NO: does this describe a programming language?\n\n${ctx.previous.output}`, maxSteps: 2 }),
    ],
  );
  for (const r of results) console.log(`[pipeline] final=${r === null ? null : JSON.stringify(r.output.slice(0, 60))}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
