import assert from 'node:assert/strict'
import mod from './index.js'
import os from 'node:os'
import path from 'node:path'
import fs from 'node:fs'

const requiredExports = [
  'Agent',
  'Session',
  'EventStream',
  'LocalWorkspaceBackend',
  'builtinSkills',
]

for (const name of requiredExports) {
  assert.equal(name in mod, true, `missing export: ${name}`)
}

assert.equal(typeof mod.Agent, 'function', 'Agent export should be a constructor')
assert.equal(typeof mod.builtinSkills, 'function', 'builtinSkills should be callable')

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'a3s-node-test-'))
const workspace = path.join(tmpRoot, 'workspace')
fs.mkdirSync(workspace, { recursive: true })
const canonicalWorkspace = fs.realpathSync(workspace)

const inlineConfig = `
default_model = "anthropic/claude-sonnet-4-20250514"

providers "anthropic" {
  api_key = "test-key"
  models "claude-sonnet-4-20250514" {
    name = "Claude Sonnet 4"
  }
}
`.trim()

const agent = await mod.Agent.create(inlineConfig)
const session = agent.session(workspace, {
  permissionPolicy: { defaultDecision: 'allow' },
  workspaceBackend: new mod.LocalWorkspaceBackend(workspace),
})

const write = await session.writeFile('notes.txt', 'one\ntwo\n')
assert.equal(write.exitCode, 0, write.output)

const read = await session.readFile('notes.txt')
assert.equal(read.includes('one'), true, 'readFile should read from workspace backend')

const listing = await session.ls()
assert.equal(listing.exitCode, 0, listing.output)
assert.equal(listing.output.includes('notes.txt'), true, 'ls should list workspace files')

const edit = await session.editFile('notes.txt', 'one', 'uno')
assert.equal(edit.exitCode, 0, edit.output)

const patch = await session.patchFile('notes.txt', '@@ -1,2 +1,2 @@\n uno\n-two\n+dos')
assert.equal(patch.exitCode, 0, patch.output)
assert.equal(fs.readFileSync(path.join(workspace, 'notes.txt'), 'utf8'), 'uno\ndos\n')

const commands = session.listCommands()
assert.equal(Array.isArray(commands), true, 'listCommands() should return an array')
assert.equal(commands.some((cmd) => cmd.name === 'help'), true, 'built-in /help should be registered')

session.registerCommand('status', 'Show session info', (args, ctx) => {
  return `args=${args};workspace=${ctx.workspace};tools=${ctx.toolNames.length}`
})

const updatedCommands = session.listCommands()
assert.equal(updatedCommands.some((cmd) => cmd.name === 'status'), true, 'custom /status should be registered')

const help = await session.send('/help')
assert.equal(help.text.includes('/help'), true, '/help should render command help text')

const model = await session.send('/model')
assert.equal(
  model.text.includes('Current model: anthropic/claude-sonnet-4-20250514'),
  true,
  '/model should report the active model'
)

const cost = await session.send('/cost')
assert.equal(cost.text.includes('Model:'), true, '/cost should include model info')
assert.equal(cost.text.includes('Tokens:'), true, '/cost should include token usage')

const history = await session.send('/history')
assert.equal(history.text.includes('Messages:'), true, '/history should include message count')
assert.equal(history.text.includes('Session:'), true, '/history should include session id')

const tools = await session.send('/tools')
assert.equal(tools.text.includes('Tools:'), true, '/tools should summarize registered tools')
assert.equal(tools.text.includes('Builtin'), true, '/tools should list builtin tools')

const result = await session.send('/status hello world')
assert.equal(result.text.includes('args=hello world;'), true, 'custom slash command should receive args')
assert.equal(
  result.text.includes(`workspace=${canonicalWorkspace};`),
  true,
  'custom slash command should receive workspace in context'
)
assert.match(result.text, /tools=\d+$/, 'custom slash command should receive toolNames in context')

// --- Subagent task query API (PR #3): three new Session methods ---
{
  const list = await session.subagentTasks()
  assert.ok(Array.isArray(list), 'subagentTasks() should resolve to an array')
  assert.equal(list.length, 0, 'fresh session should have no subagent tasks')

  const pending = await session.pendingSubagentTasks()
  assert.ok(Array.isArray(pending), 'pendingSubagentTasks() should resolve to an array')
  assert.equal(pending.length, 0, 'fresh session should have no pending subagent tasks')

  const missing = await session.subagentTask('task-does-not-exist')
  assert.equal(missing, null, 'unknown subagent task id should resolve to null')

  const cancelled = await session.cancelSubagentTask('task-does-not-exist')
  assert.equal(cancelled, false, 'cancelling an unknown subagent task id should resolve to false')
}

// --- Workflow facade: budgeted fan-out (offline shape check, no LLM) ---
{
  assert.equal(typeof session.workflowParallel, 'function', 'workflowParallel should be exposed')
  // An empty fan-out takes no LLM path: outcomes empty, ledger snapshot present.
  const capped = await session.workflowParallel([], 50000)
  assert.deepEqual(capped.outcomes, [], 'empty specs -> empty outcomes')
  assert.equal(capped.budget.consumedTokens, 0, 'no spend yet')
  assert.equal(capped.budget.limitTokens, 50000, 'limit reflected in the ledger snapshot')
  const uncapped = await session.workflowParallel([])
  assert.ok(uncapped.budget.limitTokens == null, 'uncapped -> no limit (null/undefined)')
}

session.close()

console.log('node sdk integration ok')
process.exit(0)
