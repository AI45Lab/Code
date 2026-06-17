// Serve layer (filesystem-first agents) — Node SDK tests.
//
// Verifies the napi serve binding added alongside the Rust serve daemon:
//   agent.serveAgentDir(dir, workspace[, options]) -> ServeHandle
//   handle.stop() / handle.isStopped()
//
// Unit (hermetic, inline ACL via temp file, no provider credentials): the
// ServeHandle lifecycle and exports.
// Integration (real provider, skipped without .a3s/config.acl): a real cron
// schedule firing full harness turns through the daemon without aborting the
// napi boundary (a panic in a sync #[napi] fn aborts the process), then a clean
// stop.
//
// Run with: node test_serve.mjs   (or `npm test`)

import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import mod from './index.js'

const { Agent, ServeHandle } = mod

const INLINE_CONFIG = `
default_model = "anthropic/claude-sonnet-4-20250514"
providers "anthropic" {
  api_key = "test-key"
  models "claude-sonnet-4-20250514" { name = "Claude Sonnet 4" }
}
`.trim()

function mkConfigFile() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'a3s-node-serve-cfg-'))
  const p = path.join(dir, 'agent.acl')
  fs.writeFileSync(p, INLINE_CONFIG)
  return p
}

function writeAgentDir({ withSchedule }) {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'a3s-node-serve-'))
  fs.writeFileSync(path.join(base, 'instructions.md'), 'You are a terse test agent. Answer in one word.')
  if (withSchedule) {
    fs.mkdirSync(path.join(base, 'schedules'))
    fs.writeFileSync(
      path.join(base, 'schedules', 'tick.md'),
      '---\ncron: "* * * * * *"\nname: tick\n---\nReply with exactly one word: PONG',
    )
  }
  return base
}

function repoConfig() {
  if (process.env.A3S_CONFIG_FILE && fs.existsSync(process.env.A3S_CONFIG_FILE)) {
    return process.env.A3S_CONFIG_FILE
  }
  let dir = path.dirname(new URL(import.meta.url).pathname)
  for (let i = 0; i < 8; i++) {
    const cand = path.join(dir, '.a3s', 'config.acl')
    if (fs.existsSync(cand)) return cand
    dir = path.dirname(dir)
  }
  return null
}

// ── Unit (hermetic): exports + ServeHandle lifecycle ────────────────────────
assert.equal(typeof ServeHandle, 'function', 'ServeHandle must be exported')
assert.equal(typeof Agent.prototype.serveAgentDir, 'function', 'Agent.serveAgentDir must exist')

{
  const agent = await Agent.create(mkConfigFile())
  const dir = writeAgentDir({ withSchedule: false })
  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'a3s-node-serve-ws-'))
  const handle = await agent.serveAgentDir(dir, ws)
  assert.equal(handle.isStopped(), false, 'handle should not be stopped before stop()')
  await handle.stop()
  assert.equal(handle.isStopped(), true, 'stop() must set isStopped() true')
  await handle.stop() // idempotent — must not throw
  assert.equal(handle.isStopped(), true)
  console.log('node sdk serve handle lifecycle ok')
}

// ── Integration (real provider, skipped without config) ─────────────────────
{
  const config = repoConfig()
  if (!config) {
    console.log('node sdk serve real-schedule SKIPPED (no .a3s/config.acl)')
  } else {
    const agent = await Agent.create(config)
    const dir = writeAgentDir({ withSchedule: true })
    const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'a3s-node-serve-real-ws-'))
    const handle = await agent.serveAgentDir(dir, ws)
    assert.equal(handle.isStopped(), false)
    // Let the every-second schedule fire real harness turns; surviving the
    // window (no napi abort) is the integration assertion.
    await new Promise((r) => setTimeout(r, 8000))
    await handle.stop()
    assert.equal(handle.isStopped(), true)
    console.log('node sdk serve real-schedule ok (daemon fired real turns, stopped clean)')
  }
}

console.log('all node serve tests passed')
