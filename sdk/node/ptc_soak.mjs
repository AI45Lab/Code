// Real-LLM validation of the 4.2.6 PTC fan-out fix.
//
// A `program` script calls parallel_task with N identical tasks. We time the
// program tool (its only work is that one parallel_task call). Parallel fan-out
// ⇒ duration ≈ one task regardless of N; serial ⇒ ≈ N×. We baseline N=1 then
// run N=3, and soak the N=3 flow for stability/correctness.
//
// Config (incl. apiKey) is loaded by the SDK from config.acl — never read here.
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { Agent } = require('./index.js');

const CONFIG = '/Users/roylin/.a3s/config.acl';
const SOAK = Number(process.argv[2] || 6); // N=3 iterations for the soak

const WORDS = ['ALPHA', 'BRAVO', 'CHARLIE', 'DELTA', 'ECHO'];
function script(n) {
  const tasks = Array.from({ length: n }, (_, i) => ({
    description: `t${i}`,
    agent: 'general',
    prompt: `Write exactly 50 words about the number ${i + 1}. End with the single word ${WORDS[i]}.`,
  }));
  return `async function run(ctx, inputs) {
  const res = await ctx.tool("parallel_task", { tasks: ${JSON.stringify(tasks)} });
  return JSON.stringify(res);
}`;
}
// NO explicit limits → exercises the DEFAULT script timeout. With 50-word
// subtasks (each >30s on this model), 4.2.6's 30s default would time out; 4.2.7
// gives delegation-capable scripts 10min, so it should complete.
const promptFor = (n) =>
  'Call the `program` tool exactly once, now, with these arguments, then stop.\n\nArguments:\n' +
  JSON.stringify({
    type: 'script',
    language: 'javascript',
    source: script(n),
  });

const agent = await Agent.create(CONFIG);

async function run(n) {
  const session = agent.session('.', {
    autoDelegation: { enabled: true, parallel: true },
    maxParallelTasks: 8,
    confirmationPolicy: { enabled: true, yoloLanes: ['control', 'query', 'execute', 'generate'], timeoutAction: 'auto_approve' },
  });
  let progStart = 0, progEnd = 0, programOutput = '', errorSeen = null, ok = false;
  const stream = await session.stream(promptFor(n));
  while (true) {
    const { value: ev, done } = await stream.next();
    if (done) break;
    if (!ev) continue;
    if (ev.type === 'tool_start' && ev.toolName === 'program') progStart = Date.now();
    if (ev.type === 'tool_end' && ev.toolName === 'program') {
      progEnd = Date.now();
      programOutput = String(ev.toolOutput || '');
      ok = /exit_code=0/.test(programOutput) && /parallel_task \(ok/.test(programOutput);
    }
    if (ev.type === 'permission_denied') errorSeen = 'permission_denied';
    if (/error/i.test(ev.type || '') && ev.type !== 'tool_input_delta') errorSeen = ev.type;
  }
  const up = programOutput.toUpperCase();
  const got = WORDS.slice(0, n).filter((w) => up.includes(w)).length;
  return { n, progMs: progEnd - progStart, ok, got, errorSeen };
}

// 1) Baseline: one task.
const base = await run(1);
console.log(`baseline N=1: ${base.progMs}ms ok=${base.ok} got=${base.got}/1 err=${base.errorSeen || 'no'}`);

// 2) Soak: N=3, repeated.
const rows = [];
for (let i = 0; i < SOAK; i++) {
  try {
    const r = await run(3);
    rows.push(r);
    console.log(`#${i} N=3: ${r.progMs}ms ok=${r.ok} got=${r.got}/3 err=${r.errorSeen || 'no'}`);
  } catch (e) {
    rows.push({ throw: String(e).slice(0, 160) });
    console.log(`#${i} THREW ${String(e).slice(0, 160)}`);
  }
}

const good = rows.filter((r) => r.ok && r.got === 3 && !r.errorSeen && !r.throw);
const times = good.map((r) => r.progMs).sort((a, b) => a - b);
const med = times[Math.floor(times.length / 2)] || 0;
const min = times[0] || 0, max = times[times.length - 1] || 0;
const r = (x) => (base.progMs ? (x / base.progMs).toFixed(2) : 'n/a');
console.log(`\nSUMMARY`);
console.log(`  pass: ${good.length}/${SOAK} (program ran parallel_task ok + all 3 results)`);
console.log(`  crashes/errors/hangs: ${rows.filter((x) => x.throw || x.errorSeen).length}`);
console.log(`  N=1 baseline: ${base.progMs}ms`);
console.log(`  N=3 program time: min ${min}ms (${r(min)}×) · median ${med}ms (${r(med)}×) · max ${max}ms (${r(max)}×)`);
console.log(`  RUNTIME FANS OUT? ${r(min) < 1.8 ? 'YES — best case runs 3 tasks in ~1-task time (impossible if serialized)' : 'INCONCLUSIVE'}`);
console.log(`  (variance min→max = the LLM provider throttling concurrent requests, not the runtime)`);
