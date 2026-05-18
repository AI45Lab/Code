# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `S3WorkspaceBackend` — an S3-compatible workspace backend that lets
  built-in file tools (`read`, `write`, `edit`, `patch`, `ls`) operate
  directly against any S3-compatible endpoint (AWS S3, MinIO, RustFS, R2,
  Backblaze B2, ...). Gated behind the new `s3` Cargo feature.
- Added `S3BackendConfig` builder for configuring endpoint, region, static
  or session-token credentials, force-path-style, request timeout, and
  bucket prefix.
- Added `WorkspaceServices::s3()` factory and `WorkspaceServices::from_s3_backend()`
  helper. The factory installs a 60s default per-operation timeout and
  declines `bash`, `git`, `grep`, and `glob` capabilities — capability
  gating automatically hides those tools from the model so it cannot
  call operations the backend cannot service.
- Exposed `S3WorkspaceBackend` in the Node and Python SDKs alongside
  `LocalWorkspaceBackend`. Configuration uses the same option surface
  (`workspaceBackend` / `workspace_backend`).

### Changed

- Restructured `core/src/workspace.rs` into a `workspace/` module with
  `workspace/mod.rs` (abstract traits + `WorkspaceServices`),
  `workspace/local.rs` (`LocalWorkspaceBackend`), and `workspace/s3.rs`
  (`S3WorkspaceBackend`). No behavioural change for existing callers.

## [2.6.0] - 2026-05-18

### Added

- Added `WorkspaceServices` capability abstraction (`core/src/workspace.rs`)
  that lets the host supply file system, command runner, search, and Git
  providers behind the stable built-in tool contract. The default
  `LocalWorkspaceBackend` preserves existing local-filesystem behavior, while
  DFS, browser, container, and remote backends can be assembled via
  `WorkspaceServicesBuilder`.
- Added `SessionOptions::with_workspace_backend()` (alias
  `with_workspace_services`) so callers can opt-in to non-local workspaces
  without changing tool schemas.
- Added capability-driven tool gating: `bash`, `grep`, `glob`, and `git` are
  only registered when the workspace backend declares the matching capability,
  preventing models from invoking tools the backend cannot service.
- Added `Session::write_file`, `Session::ls`, `Session::edit_file`, and
  `Session::patch_file` direct-tool APIs in core, Node, and Python SDKs,
  alongside the existing `read_file` / `bash` / `glob` / `grep`.
- Added `LocalWorkspaceBackend` class to the Node and Python SDKs as the
  explicit typed form of the default backend and the option surface for future
  remote/browser/DFS workspaces.
- Added `workspace_services` to `ChildRunContext` so child runs inherit the
  parent's workspace backend.
- Added 17 unit + integration tests covering virtual path resolution, capability
  downgrade, contract-level tool routing for files / search / bash / git
  through pluggable backends, and session-level direct-tool dispatch.

### Changed

- Refactored built-in tools `read`, `write`, `edit`, `patch`, `ls`, `bash`,
  `grep`, `glob`, and `git` to route operations through `WorkspaceServices`
  instead of hard-coded local filesystem calls. Local behavior is unchanged.
- Centralized workspace-boundary path checks in
  `ToolContext::resolve_workspace_path`, removing duplicated canonicalization
  logic from `ToolExecutor::execute`.

### Fixed

- Removed two `clippy::useless_conversion` warnings in
  `core/tests/test_ahp_idle_with_llm.rs` so `cargo clippy --all-targets` is
  clean.

### Documentation

- Updated `README.md`, Node SDK README, and Python SDK README with workspace
  backend usage and the new direct-tool API surface.

## [2.5.0] - 2026-05-12

### Added

- Added `ConfirmationInheritance` enum for controlling how child runs resolve Ask
  decisions: `AutoApprove` (default), `DenyOnAsk`, and `InheritParent`.
- Added `confirmation_inheritance` field to `WorkerAgentSpec` in Node and Python
  SDKs, allowing fine-grained control over child run confirmation behavior.
- Added `ChildRunContext` for explicit parent capability inheritance, ensuring
  child runs properly inherit permission checkers and confirmation policies.
- Added comprehensive integration tests for task delegation with real LLM calls
  and mock LLM contract tests for permission and confirmation inheritance.
- Added SDK integration tests for `confirmation_inheritance` in both Node and
  Python SDKs with `.a3s/config.acl` configuration support.

### Fixed

- Fixed task delegation to properly inherit permission checker from agent
  definition in child runs (Issue #28).
- Fixed child runs to respect parent's confirmation policy when using
  `InheritParent` mode.

### Changed

- Unified `AgentDefinition` → `AgentConfig` conversion via `apply_to()` method
  for consistent configuration application.
- Refactored `ToolExecutor` to remove redundant `guard_policy` field, relying
  on `PermissionChecker` for all permission decisions.

### Documentation

- Updated Node and Python SDK READMEs with `confirmation_inheritance` examples
  and usage guidance.
- Updated English and Chinese documentation for teams and tasks with worker
  agent confirmation inheritance patterns.

## [2.4.0] - 2026-05-11

### Added

- Added `generate_object` built-in tool for structured JSON output with schema
  validation, automatic repair, and streaming partial objects. Works across all
  providers via tool-calling mode.
- Added `llm::structured` module with four output modes (tool, prompt, strict,
  json), robust JSON extraction from dirty LLM output, partial JSON parser for
  streaming, and a built-in JSON Schema validator supporting `anyOf`/`oneOf`,
  nullable types, `additionalProperties`, `pattern`, and numeric ranges.
- Added streaming partial object support: `generate_object` emits
  `tool_output_delta` events with progressively complete JSON snapshots.
- Added comprehensive documentation: structured output example (EN/CN), contract
  review tutorial (EN/CN), and 7 additional core mechanism tutorials (PTC,
  streaming, session persistence, skills, MCP, security/HITL, hooks, memory).

### Fixed

- Fixed Shiki build error in docs site caused by unsupported `acl` language
  identifier in code blocks (replaced with `text`).

## [2.3.0] - 2026-05-09

### Added

- Added compact, object-shaped SDK APIs for long-lived integrations:
  `send(...)`, `run(...)`, `stream(...)`, `task(...)`, `tasks(...)`,
  `git(...)`, `addMcp(...)` / `add_mcp(...)`, `removeMcp(...)` /
  `remove_mcp(...)`, and `mcps()`.
- Added live run/tool observability through active tool snapshots and richer
  run replay APIs across Rust, Node.js, and Python SDKs.
- Added a durable SDK API design contract under `manual/SDK_API_DESIGN.md`.
- Added Python SDK parity for worker agents, HITL confirmation policy/control,
  session-for-worker, live worker registration, and session close.

### Changed

- Split the large agent and session API implementation files into focused
  runtime modules for maintainability.
- Made AHP the single harness/advisory/control plane with richer event context,
  heartbeat state, runtime state snapshots, and decision mapping.
- Updated docs and examples to prefer short SDK method names while retaining
  long compatibility aliases.
- Re-exported `ActiveToolSnapshot` from the Rust core crate root.

### Removed

- Removed the obsolete sidecar/copilot/BTW/strategize/BTE mechanism and related
  prompts, docs, configs, and examples. Background advice, context supplements,
  and PTC proposals now belong to the caller or AHP harness.

---

## [2.0.0] - 2026-05-02

### Changed

- Promoted A3S Code package metadata to `2.0.0` across Rust core, Node.js SDK, and Python SDK.
- Standardized runtime configuration on ACL (`.acl`) and explicit `env(...)` credential injection.
- Reworked the public API surface around `Agent`, `AgentSession`, and 2.0-compatible session/control-plane primitives.

### Added

- Release-blocking real-provider integration test for `.a3s/config.acl` environment-variable injection.
- No-network integration coverage, script dry-run support, and literal-config extraction for MiniMax ACL `env(...)` resolution.
- Release validation scripts for local core tests, AHP feature tests, version consistency, patch hygiene, and real-provider ACL smoke tests.

### Removed

- Legacy HCL config artifacts and stale prompt tests that no longer match the 2.0 ACL runtime.

---

## [v1.8.6] - 2026-04-10

### Fixed

#### web_search Tool

- **Issue #25 Fix**: The `web_search` tool now returns an error when unknown parameters are passed (e.g., `engine` instead of `engines`). Previously, unknown parameters were silently ignored, causing confusion when users specified the wrong field name.

### Changed

- `engines` parameter type changed from `string` to `array` in schema to match actual API
- Updated a3s-search integration to v1.0.0

---

## [v1.8.5] - 2026-04-05

### Added

#### Git Built-in Tool

- **Built-in Git Client**: New `git` tool with auto-install support for Windows, macOS, and Linux. Downloads official pre-built git binaries to `~/.local/git/bin/` when git is not available - no package manager required.

  Full git operations: `status`, `log`, `branch`, `checkout`, `diff`, `stash`, `remote`, `worktree`

- **Git Convenience Methods**: Python SDK (`session.git(...)`) and Node SDK (`session.git(...)`) convenience methods for git operations.

#### System Prompt Updates

- Updated all system prompts to reference "A3S Code" instead of "Claude Code"
- Updated skill references to use `a3s-lab/code-skills`

### Removed

- **Document Parser**: Removed `composite_document_parser` and `document` modules and all related code. This feature was not fully implemented and has been removed to simplify the codebase.

- **Agentic Search/Parse Tools**: Removed `agentic_search` and `agentic_parse` built-in tools.

- **Git Worktree Tool**: Replaced by the new unified `git` tool with `worktree` subcommand.

### Changed

- **Tool Count**: Updated built-in tool count from 15 to 16 to reflect new git and box tools.
- **Documentation**: Updated all documentation to reflect new tool names and capabilities.

---

## [v1.6.0] - 2026-04-02

### Added

#### Document Parsing

- **XLSB (Excel Binary) Support**: Added calamine-based BIFF12 parsing for XLSB files with proper cell value extraction, supporting Float, Int, Bool, DateTime, DateTimeIso, and DurationIso types. Significantly improves table fidelity for .xlsb files.

- **HWPX Table Extraction**: Added structured table extraction from Korean HWPX documents. Parses `tbl/tr/tc` XML hierarchy and includes `structured_payload` for `tables[]` output.

- **Vision OCR Provider**: New OCR backend supporting OpenAI-compatible vision APIs for document OCR fallback.

  ```hcl
  document_parser {
    ocr {
      enabled  = true
      model    = "openai/gpt-4.1-mini"
      api_key  = "sk-..."
      base_url = "https://api.openai.com/v1"  # optional
      prompt   = "Extract all text from this document..."
      max_images = 8
      dpi     = 144
    }
  }
  ```

  Provider priority: External provider > Vision API (if model+api_key configured) > Builtin tesseract

#### Search Ranking

- **Tabular Query Intent Detection**: Automatically detects when queries relate to tables (keywords: table, column, row, spreadsheet, excel, csv, cell, data, record, etc.) and boosts table line matches by +10 keyword hits plus 1.3x relevance multiplier.

- **Heading Inheritance Boost**: When search matches appear under headings that also match the query, those matches receive a relevance boost (up to 1.3x). Looks backwards to find the closest preceding heading.

### Changed

#### Configuration

- `DocumentOcrConfig` extended with new fields:
  - `provider: Option<String>` - Backend selection ("vision" or "builtin")
  - `base_url: Option<String>` - Custom API endpoint
  - `api_key: Option<String>` - API authentication

#### Dependencies

- Added `calamine = "0.26"` for XLSB parsing
- Added `reqwest/blocking` feature for Vision API HTTP calls

### Fixed

- Test assertion: `paged_text_blocks_reflow_two_column_preserves_paragraph_breaks` - Corrected expected string "Parser metadata now tracks OCR" vs "Parser metadata now tracks OCR backend"

---

## [v1.5.8] - 2026-03-07

### Added

- Phase 1 structured result surfaces:
  - `structured_payload` exposed in `agentic_parse` output and metadata
  - Table payloads in stable machine-readable form
  - Page-level data in `agentic_parse` output and metadata
  - Stable `tables[]`, `pages[]`, `elements[]` outputs

- Phase 2 PDF extraction improvements:
  - lopdf position-aware text extraction
  - Reduced dependence on weak text fallbacks
  - Position-aware table detection

- `agentic_search` enhancements:
  - Chunk context consumption
  - Tabular content consumption
  - Page numbers and locators support

### Changed

- `ParsedDocument` extended with `tables: Vec<StructuredTable>` and `pages: Vec<PageInfo>`

### Fixed

- Windows shell compatibility improvements

---

## [v1.5.7] - 2026-02-28

### Added

- Runtime session header support for OpenAI configs
- Cross-platform environment variable expansion in tests

---

## [v1.5.6] - 2026-02-20

### Added

- Enhanced agent config, document parser, LLM, tools, and SDKs
- Host shell environment propagation to tool commands

---

## [v1.5.5] - 2026-02-10

### Added

- Zhipu AI client (`ZhipuClient` formerly `GlmClient`)
- Duplicate tool call circuit breaker
- Streaming fallback support
- `agentic_parse` skill

---

## [v1.5.4] - 2026-01-28

### Added

- Session-local skill registries

---

## [v1.5.3] - 2026-01-15

### Added

- Tool schema hardening
- Slash command output restoration
