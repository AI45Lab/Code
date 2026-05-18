import assert from 'node:assert/strict'
import mod from './index.js'

assert.equal(typeof mod.builtinSkills, 'function')
assert.equal(typeof mod.formatVerificationSummary, 'function')
assert.equal(typeof mod.LocalWorkspaceBackend, 'function')

const skills = mod.builtinSkills()
assert.equal(Array.isArray(skills), true)
assert.equal(skills.length > 0, true)
assert.equal(typeof skills[0].name, 'string')

const summary = mod.formatVerificationSummary({
  status: 'skipped',
  report_count: 0,
  required_check_count: 0,
  pending_required_check_count: 0,
  failed_check_count: 0,
  residual_risk_count: 0,
})
assert.equal(summary, 'Verification skipped: no reports.')

console.log('node sdk helper smoke ok')
process.exit(0)
