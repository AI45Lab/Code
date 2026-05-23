/**
 * Entry point for `@a3s-lab/code` type declarations.
 *
 * This file is hand-authored. The napi-rs build writes to `generated.d.ts`,
 * which this aggregator re-exports. Cross-boundary types that aren't
 * generated from Rust (because napi-derive would camelCase fields that
 * need to mirror the wire JSON, or because they describe discriminated
 * unions napi can't express) live in `extra-types.d.ts`.
 *
 * Edit those two files, not this one.
 */
export * from './generated'
export * from './extra-types'
