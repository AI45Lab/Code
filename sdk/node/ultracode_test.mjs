// Autonomous ultracode validation: give the model the REAL (strengthened)
// ULTRACODE_GUIDELINES via guidelines:string + a natural multi-part task, and
// verify it GENERATES a `program` workflow script that calls parallel_task and
// fans out — no hand-fed script. Live-logged to ./ultracode_test.log.
import { createRequire } from 'node:module';
import { appendFileSync } from 'node:fs';
const require = createRequire(import.meta.url);
const { Agent } = require('./index.js');

const CONFIG = '/Users/roylin/.a3s/config.acl';
const LOG = new URL('./ultracode_test.log', import.meta.url).pathname;
const ITERS = Number(process.argv[2] || 3);
const log = (s) => { appendFileSync(LOG, s + '\n'); console.log(s); };

// EXACT text from crates/cli/src/tui/panels/model.rs ULTRACODE_GUIDELINES.
const GUIDELINES = `[ultracode] Dynamic-workflow mode. Express ALL of your work as ONE generated, executable workflow SCRIPT. Do NOT call \`parallel_task\` or \`task\` directly at the top level — the script IS the workflow.
1. PLAN. Decompose the task into numbered steps; mark independent (concurrent) vs dependent (sequential).
2. WRITE + RUN THE SCRIPT by calling the \`program\` tool with a JavaScript \`source\` of this shape:
     async function run(ctx, inputs) {
       const results = await ctx.tool("parallel_task", { tasks: [
         { description: "step A", prompt: "..." },
         { description: "step B", prompt: "..." }
       ] });
       return results;
     }
   Put EVERY task/parallel_task call INSIDE the script; add further ctx.tool(...) calls for dependent steps and aggregate their outputs.
3. parallel_task inside the script fans out concurrent subagents on the multi-threaded runtime. After it returns, synthesize the results into your final answer.
4. Be exhaustive: pursue every thread to completion.`;

// Final validation: a GENERAL task + the EXACT per-turn nudge the cli's
// start_stream now appends for ultracode. This mirrors the shipped cli behavior
// (system guideline + turn nudge) — soaks whether the script form is now
// reliable for ordinary tasks the user would actually type.
const NUDGE =
  '\n\n[ultracode] Tackle this as a dynamic workflow. For the independent parts, ' +
  'call the `program` tool with a JavaScript script whose `async function ' +
  'run(ctx, inputs)` fans them out via `ctx.tool("parallel_task", { tasks: [...] })`, ' +
  'keeps all task/parallel_task delegation INSIDE the script, then aggregates and ' +
  'returns. After it runs, synthesize the results.';
const TASK =
  'Write about 50 words on each of these four independent topics: TCP slow-start, ' +
  'B-tree node splits, the TLS 1.3 handshake, and Bloom-filter false positives.' +
  NUDGE;

const agent = await Agent.create(CONFIG);

async function once(i) {
  const session = agent.session('.', {
    guidelines: GUIDELINES,
    autoDelegation: { enabled: true, parallel: true },
    maxParallelTasks: 8,
    confirmationPolicy: { enabled: true, yoloLanes: ['control', 'query', 'execute', 'generate'], timeoutAction: 'auto_approve' },
  });
  let inProgram = false, progArgs = '', scriptForm = false, directParallel = false, programOk = false, errorSeen = null;
  const t0 = Date.now();
  const stream = await session.stream(TASK);
  while (true) {
    const { value: ev, done } = await stream.next();
    if (done) break;
    if (!ev) continue;
    const ty = ev.type || '';
    if (ty === 'tool_start' && ev.toolName === 'program') inProgram = true;
    if (ty === 'tool_start' && ev.toolName === 'parallel_task') directParallel = true;
    if (ty === 'tool_input_delta' && inProgram) progArgs += (ev.text || '');
    if (ty === 'tool_end' && ev.toolName === 'program') {
      inProgram = false;
      scriptForm = /parallel_task/.test(progArgs);
      programOk = /exit_code=0/.test(String(ev.toolOutput || '')) && /parallel_task \(ok/.test(String(ev.toolOutput || ''));
    }
    if (ty === 'permission_denied') errorSeen = 'permission_denied:' + (ev.toolName || '');
  }
  log(`#${i} ${Date.now() - t0}ms scriptForm=${scriptForm} programOk=${programOk} directParallel=${directParallel} err=${errorSeen || 'no'}`);
  if (scriptForm && i === 0) {
    // Show the actual generated workflow script once, as evidence.
    const m = progArgs.match(/"source"\s*:\s*"((?:[^"\\]|\\.)*)"/);
    if (m) log('  --- generated workflow script ---\n' + JSON.parse('"' + m[1] + '"').split('\n').map((l) => '  | ' + l).join('\n'));
  }
  return { i, scriptForm, programOk, directParallel, errorSeen };
}

log(`\n=== autonomous ultracode test (strengthened guideline, ${ITERS} iters) ===`);
const rows = [];
for (let i = 0; i < ITERS; i++) {
  try { rows.push(await once(i)); }
  catch (e) { const m = String(e).slice(0, 160); rows.push({ throw: m }); log(`#${i} THREW ${m}`); }
}
const scriptAndRan = rows.filter((r) => r.scriptForm && r.programOk).length;
const anyFanout = rows.filter((r) => (r.scriptForm && r.programOk) || r.directParallel).length;
const errs = rows.filter((r) => r.throw || r.errorSeen).length;
log(`\nSUMMARY`);
log(`  generated a program WORKFLOW SCRIPT that fanned out (programOk): ${scriptAndRan}/${ITERS}`);
log(`  fanned out at all (script or direct): ${anyFanout}/${ITERS}`);
log(`  errors/crashes: ${errs}`);
log(`  VERDICT: ${scriptAndRan >= Math.ceil(ITERS / 2) ? 'EFFECTIVE — ultracode autonomously generates+runs dynamic workflow scripts' : 'STILL WEAK — strengthen further'}`);
