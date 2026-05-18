/**
 * Real-provider smoke for the Node SDK surface hardened in 2.3.
 *
 * The runner script rewrites .a3s/config.acl so OpenAI-compatible credentials
 * come from A3S_OPENAI_* environment variables. MINIMAX_* aliases are accepted
 * by the script before this file runs.
 */

import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const require = createRequire(import.meta.url);
const { Agent, LocalWorkspaceBackend } = require('@a3s-lab/code');
const timeoutMs = Number(process.env.A3S_CODE_SDK_REAL_TIMEOUT_MS || '180000');
const runFullAgentSmoke = process.env.A3S_CODE_SDK_REAL_AGENT_SMOKE !== '0';
const runChildAgentSmoke = process.env.A3S_CODE_SDK_REAL_CHILD_AGENT_SMOKE === '1';

async function step(name, fn) {
  process.stdout.write(`[node-sdk-real] ${name} ... `);
  const started = Date.now();
  try {
    const value = await withTimeout(Promise.resolve().then(fn), name);
    console.log(`ok (${Date.now() - started}ms)`);
    return value;
  } catch (error) {
    console.log(`failed (${Date.now() - started}ms)`);
    throw error;
  }
}

function withTimeout(promise, name) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => {
      reject(new Error(`${name} timed out after ${timeoutMs}ms`));
    }, timeoutMs);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

const configFile = process.env.A3S_CONFIG_FILE;
if (!configFile) {
  throw new Error('A3S_CONFIG_FILE must point to the env-injected ACL config');
}

const agent = await step('Agent.create', () => Agent.create(configFile));
const workspace = process.env.A3S_CODE_SDK_REAL_WORKSPACE || mkdtempSync(join(tmpdir(), 'a3s-code-node-sdk-real-'));
console.log(`[node-sdk-real] workspace=${workspace}`);
const session = agent.session(workspace, {
  planningMode: 'disabled',
  permissionPolicy: { defaultDecision: 'allow' },
  maxParseRetries: 1,
  circuitBreakerThreshold: 1,
  workspaceBackend: new LocalWorkspaceBackend(workspace),
});

const toolNames = await step('toolNames', () => session.toolNames());
assert.ok(toolNames.includes('program'), 'program tool should be registered');
assert.ok(toolNames.includes('task'), 'task tool should be registered');
assert.ok(toolNames.includes('parallel_task'), 'parallel_task tool should be registered');

const toolDefinitions = await step('toolDefinitions', () => session.toolDefinitions());
assert.ok(Array.isArray(toolDefinitions), 'toolDefinitions() should return an array');
assert.ok(
  toolDefinitions.some((tool) => tool.name === 'program'),
  'program schema should be visible through toolDefinitions()',
);

const writeResult = await step('writeFile', () => session.writeFile('notes.txt', 'one\ntwo\n'));
assert.equal(writeResult.exitCode, 0, writeResult.output);
assert.match(await step('readFile', () => session.readFile('notes.txt')), /one/);
const lsResult = await step('ls', () => session.ls());
assert.equal(lsResult.exitCode, 0, lsResult.output);
assert.match(lsResult.output, /notes\.txt/);
const editResult = await step('editFile', () => session.editFile('notes.txt', 'one', 'uno'));
assert.equal(editResult.exitCode, 0, editResult.output);
const patchResult = await step('patchFile', () =>
  session.patchFile('notes.txt', '@@ -1,2 +1,2 @@\n uno\n-two\n+dos'),
);
assert.equal(patchResult.exitCode, 0, patchResult.output);
assert.equal(readFileSync(join(workspace, 'notes.txt'), 'utf8'), 'uno\ndos\n');

const programResult = await step('program', () => session.program({
  source: `
    export default async function run(ctx, inputs) {
      const listing = await ctx.ls('.');
      return { marker: inputs.marker, listed: listing.length > 0 };
    }
  `,
  inputs: { marker: 'node-sdk-program-ok' },
  allowedTools: ['ls'],
}));
assert.equal(programResult.exitCode, 0);
assert.match(programResult.output, /node-sdk-program-ok/);

if (runFullAgentSmoke) {
  const result = await step('send', () =>
    session.send('Reply with exactly: NODE_SDK_REAL_OK'),
  );
  assert.ok(result.text.trim().length > 0, 'real LLM send should produce text');

  const [run] = await step('runs', () => session.runs());
  assert.ok(run, 'send() should record a run snapshot');
  assert.equal(run.status, 'completed');
  const events = await step('runEvents', () => session.runEvents(run.id));
  assert.ok(events.some((event) => event.event.type === 'agent_start'));
  assert.ok(events.some((event) => event.event.type === 'agent_end'));

  const delegated = await step('task', () => session.task({
    agent: 'explore',
    description: 'Node SDK delegated child smoke',
    prompt: runChildAgentSmoke
      ? 'Reply with exactly: NODE_SDK_DELEGATE_OK'
      : 'Background smoke; no result is required.',
    background: !runChildAgentSmoke,
    maxSteps: 3,
  }));
  assert.equal(delegated.exitCode, 0, delegated.output);
  if (runChildAgentSmoke) {
    assert.ok(delegated.output.trim().length > 0, 'task() should return child output');
  } else {
    assert.match(delegated.output, /Task started in background/);
    console.log('[node-sdk-real] synchronous child-agent task smoke skipped; set A3S_CODE_SDK_REAL_CHILD_AGENT_SMOKE=1 to enable');
  }
} else {
  console.log('[node-sdk-real] full agent send/task smoke skipped by A3S_CODE_SDK_REAL_AGENT_SMOKE=0');
}

console.log('node sdk real config env integration ok');
