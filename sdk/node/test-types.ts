// Compile-only check that both generated and hand-authored types resolve
// from the package entry. If the napi-rs build ever overwrites index.d.ts
// (i.e. the aggregator was lost), this file will fail to type-check.
//
// Run with: npx tsc --noEmit --module nodenext --moduleResolution nodenext test-types.ts

import type {
  // From generated.d.ts (napi-rs):
  Session,
  Agent,
  AgentEvent,
  ToolResult,
  // From extra-types.d.ts (hand-authored):
  ToolErrorKind,
  VerificationStatus,
  VerificationCheck,
  VerificationReport,
  ToolArtifact,
} from './index.js'

// Forced uses so unused-import lint stays quiet.
declare const _session: Session
declare const _agent: Agent
declare const _event: AgentEvent
declare const _result: ToolResult
declare const _err: ToolErrorKind
declare const _status: VerificationStatus
declare const _check: VerificationCheck
declare const _report: VerificationReport
declare const _artifact: ToolArtifact

// Exhaustive narrowing on the discriminated union — confirms the union
// shape survives a regenerate.
function _describe(err: ToolErrorKind): string {
  switch (err.type) {
    case 'version_conflict':
      return `${err.path} expected=${err.expected}`
    case 'remote_git_conflict':
      return err.message
    case 'not_found':
      return err.path
    case 'invalid_argument':
    case 'unsupported':
      return err.message
    case 'timeout':
      return `${err.op} after ${err.duration_ms}ms`
  }
}
