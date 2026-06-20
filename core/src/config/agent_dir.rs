//! Filesystem-first agent directory convention (eve-style, harness-respecting).
//!
//! A single directory defines a durable agent by convention:
//!
//! ```text
//! agent/
//! ├── instructions.md   (required)  role/guidelines — injected as a prompt SLOT,
//! │                                 NOT a system-prompt override, so the harness
//! │                                 keeps BOUNDARIES, response-format, and
//! │                                 verification authoritative.
//! ├── agent.acl          (optional)  model/providers/queue (CodeConfig). Default if absent.
//! ├── skills/            (optional)  *.md skills, appended to CodeConfig.skill_dirs.
//! ├── schedules/         (optional)  *.md cron jobs (YAML frontmatter `cron:` + body=prompt).
//! └── tools/             (optional)  *.md tool specs: `kind: mcp` → MCP server,
//! │                                 `kind: script` → sandboxed QuickJS tool. Both
//! │                                 register into the session as ordinary tools.
//! ```
//!
//! [`AgentDir::load`] SYNTHESIZES existing config objects rather than adding a new
//! runtime: `instructions.md` → [`SystemPromptSlots`], `agent.acl` → [`CodeConfig`],
//! `skills/` → `skill_dirs`. Tool definition, visibility, and safety stay
//! harness-owned (the deliberate divergence from eve's user-defined-tools model).

use std::path::{Path, PathBuf};

use crate::config::CodeConfig;
use crate::error::{CodeError, Result};
use crate::mcp::McpServerConfig;
use crate::prompts::SystemPromptSlots;

/// A cron-triggered recurring turn, parsed from `schedules/<name>.md`.
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduleSpec {
    /// Schedule name (frontmatter `name`, else the file stem).
    pub name: String,
    /// Cron expression (validated/executed by the serve layer).
    pub cron: String,
    /// Markdown prompt sent into a turn on each fire (the file body).
    pub prompt: String,
    /// Whether the schedule is active (frontmatter `enabled`, default true).
    pub enabled: bool,
}

/// A tool definition parsed from `tools/<name>.md`, dispatched by `kind`.
///
/// Tool *definition* may come from the directory, but visibility and safety stay
/// harness-owned (the deliberate divergence from eve): an `mcp` spec is registered
/// through the normal [`add_mcp_server`](crate::AgentSession) path, so its tools
/// are namespaced `mcp__<server>__<tool>` and gated by the session's permission
/// policy like any other tool.
#[derive(Debug, Clone)]
pub enum ToolSpec {
    /// `kind = "mcp"` → an MCP server connected into the session, contributing its
    /// `list_tools()` as `mcp__<name>__*` tools.
    Mcp(McpServerConfig),
    /// `kind = "script"` → a sandboxed QuickJS tool over the existing `program`
    /// path. The model sees a named tool; the script `path`, allow-list, and
    /// limits are pinned by the spec.
    Script(ScriptToolSpec),
}

impl ToolSpec {
    /// The tool/server name (registry key; unique within `tools/`).
    pub fn name(&self) -> &str {
        match self {
            ToolSpec::Mcp(cfg) => &cfg.name,
            ToolSpec::Script(spec) => &spec.name,
        }
    }

    /// The spec kind discriminant (`mcp` or `script`).
    pub fn kind(&self) -> &str {
        match self {
            ToolSpec::Mcp(_) => "mcp",
            ToolSpec::Script(_) => "script",
        }
    }
}

/// A sandboxed QuickJS tool parsed from a `kind = "script"` file. Names a
/// workspace-relative `.js`/`.mjs` source and pins the sandbox allow-list +
/// limits; the model supplies only `inputs`. Executed via the existing `program`
/// tool path — no new sandbox. The model's call to it is permission-gated like any
/// tool; the script's inner `ctx.tool` calls are bounded by `allowed_tools` + the
/// sandbox (NOT the session permission policy), so the allow-list is the boundary.
#[derive(Debug, Clone)]
pub struct ScriptToolSpec {
    /// Model-visible tool name (registry key; unique within `tools/`).
    pub name: String,
    /// Model-facing description (frontmatter `description`, else the file body).
    pub description: String,
    /// Workspace-relative path to the `.js`/`.mjs` source.
    pub path: PathBuf,
    /// Tools the script may call through `ctx`. The agent-dir loader fails closed:
    /// an omitted list becomes `Some(vec![])` (the script may call NO tools), so a
    /// directory author must opt each tool in explicitly. `program` is always
    /// excluded (no script-launches-script). This allow-list — not the session
    /// permission policy — is what bounds a script's inner `ctx.tool` calls, so it
    /// is the security boundary for directory-authored scripts.
    pub allowed_tools: Option<Vec<String>>,
    /// Sandbox limits (timeout / tool-call / output caps); defaults apply when unset.
    pub limits: ScriptToolLimits,
}

/// Sandbox limits for a `kind = "script"` tool. Mirrors the three numeric fields
/// the `program` tool's `ScriptLimits` accepts and is serialized to it verbatim
/// (camelCase keys), so no new limit machinery is introduced.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScriptToolLimits {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_bytes: Option<usize>,
}

/// A loaded agent directory: synthesized [`CodeConfig`] + prompt slots + parsed
/// schedule + tool specs. Build a session from `config` + `prompt_slots`.
///
/// Distinct from [`CodeConfig::agent_dirs`](crate::config::CodeConfig) /
/// `register_agent_dir`, which scan a directory for **worker/subagent**
/// definitions. An `AgentDir` is the eve-style *primary* agent — the directory
/// that defines this agent's prompt, skills, schedules, and tools.
#[derive(Debug, Clone)]
pub struct AgentDir {
    pub dir: PathBuf,
    pub config: CodeConfig,
    pub prompt_slots: SystemPromptSlots,
    pub schedules: Vec<ScheduleSpec>,
    pub tools: Vec<ToolSpec>,
}

impl AgentDir {
    /// Load an agent directory by convention. `instructions.md` is required.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        if !dir.is_dir() {
            return Err(CodeError::Context(format!(
                "agent directory not found: {}",
                dir.display()
            )));
        }

        // instructions.md (required) → role SLOT. Using a slot (not a raw system
        // prompt) keeps the harness's BOUNDARIES/response-format/verification.
        let instructions = std::fs::read_to_string(dir.join("instructions.md")).map_err(|e| {
            CodeError::Context(format!(
                "agent dir {} is missing required instructions.md: {e}",
                dir.display()
            ))
        })?;
        let prompt_slots = SystemPromptSlots {
            role: Some(instructions.trim().to_string()),
            ..Default::default()
        };

        // agent.acl (optional) → CodeConfig, else default.
        let acl_path = dir.join("agent.acl");
        let mut config = if acl_path.is_file() {
            CodeConfig::from_file(&acl_path)?
        } else {
            CodeConfig::default()
        };

        // skills/ → appended to skill_dirs (existing *.md format, zero adaptation).
        let skills_dir = dir.join("skills");
        if skills_dir.is_dir() {
            config.skill_dirs.push(skills_dir);
        }

        let schedules = load_schedules(&dir.join("schedules"))?;
        let tools = load_tools(&dir.join("tools"))?;

        Ok(Self {
            dir,
            config,
            prompt_slots,
            schedules,
            tools,
        })
    }
}

/// Markdown files with a `<ext>` extension in `dir`, sorted by path. Returns an
/// empty list when `dir` does not exist.
fn md_files(dir: &Path, exts: &[&str]) -> Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| CodeError::Context(format!("read {}: {e}", dir.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .map(|e| exts.contains(&e))
                .unwrap_or(false)
        })
        .collect();
    entries.sort();
    Ok(entries)
}

fn load_schedules(dir: &Path) -> Result<Vec<ScheduleSpec>> {
    let mut out = Vec::new();
    for path in md_files(dir, &["md"])? {
        let content = std::fs::read_to_string(&path)
            .map_err(|e| CodeError::Context(format!("read {}: {e}", path.display())))?;
        let (front, body) = split_frontmatter(&content);
        let front = front.ok_or_else(|| {
            CodeError::Context(format!(
                "schedule {} has no YAML frontmatter (need `cron:`)",
                path.display()
            ))
        })?;
        let meta: ScheduleFront = serde_yaml::from_str(&front).map_err(|e| {
            CodeError::Context(format!("schedule {} frontmatter: {e}", path.display()))
        })?;
        out.push(ScheduleSpec {
            name: meta.name.unwrap_or_else(|| file_stem(&path)),
            cron: meta.cron,
            prompt: body.trim().to_string(),
            enabled: meta.enabled.unwrap_or(true),
        });
    }
    Ok(out)
}

/// Upper bounds for a `kind = "script"` tool's sandbox limits. A `tools/` file is
/// semi-trusted (the whole point of the guardrail), so an author cannot set an
/// effectively-unbounded `timeoutMs` that hangs the harness, nor a zero that makes
/// the tool silently non-functional. Generous ceilings; the program tool's own
/// defaults (30s / 20 calls / 64 KiB) apply when a field is unset.
const SCRIPT_MAX_TIMEOUT_MS: u64 = 600_000; // 10 minutes
const SCRIPT_MAX_TOOL_CALLS: usize = 1_000;
const SCRIPT_MAX_OUTPUT_BYTES: usize = 16 * 1024 * 1024; // 16 MiB

/// Reject zero or above-ceiling limits at load (fail closed). Unset fields keep
/// the program tool's defaults.
fn validate_script_limits(
    limits: ScriptToolLimits,
) -> std::result::Result<ScriptToolLimits, String> {
    fn check<T: PartialOrd + Copy + std::fmt::Display>(
        v: Option<T>,
        max: T,
        one: T,
        field: &str,
    ) -> std::result::Result<(), String> {
        if let Some(v) = v {
            if v < one || v > max {
                return Err(format!("limit {field}={v} is out of range [1, {max}]"));
            }
        }
        Ok(())
    }
    check(limits.timeout_ms, SCRIPT_MAX_TIMEOUT_MS, 1, "timeoutMs")?;
    check(
        limits.max_tool_calls,
        SCRIPT_MAX_TOOL_CALLS,
        1,
        "maxToolCalls",
    )?;
    check(
        limits.max_output_bytes,
        SCRIPT_MAX_OUTPUT_BYTES,
        1,
        "maxOutputBytes",
    )?;
    Ok(limits)
}

fn load_tools(dir: &Path) -> Result<Vec<ToolSpec>> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for path in md_files(dir, &["md"])? {
        let content = std::fs::read_to_string(&path)
            .map_err(|e| CodeError::Context(format!("read {}: {e}", path.display())))?;
        let (front, body) = split_frontmatter(&content);
        let front = front.ok_or_else(|| {
            CodeError::Context(format!(
                "tool {} has no YAML frontmatter (need `kind:`)",
                path.display()
            ))
        })?;
        let meta: ToolFront = serde_yaml::from_str(&front)
            .map_err(|e| CodeError::Context(format!("tool {} frontmatter: {e}", path.display())))?;
        let spec = match meta.kind.as_str() {
            "mcp" => {
                // The frontmatter's flat fields (transport/command/args/url/…) plus
                // `name` deserialize straight into McpServerConfig; the `kind` key is
                // ignored by its lenient Deserialize.
                let cfg: McpServerConfig = serde_yaml::from_str(&front).map_err(|e| {
                    CodeError::Context(format!(
                        "tool {} (kind=mcp) is not a valid MCP server config: {e}",
                        path.display()
                    ))
                })?;
                ToolSpec::Mcp(cfg)
            }
            "script" => {
                let meta: ScriptFront = serde_yaml::from_str(&front).map_err(|e| {
                    CodeError::Context(format!(
                        "tool {} (kind=script) frontmatter: {e}",
                        path.display()
                    ))
                })?;
                // Fail closed at load (not at first call), consistent with the
                // runtime guards the script runs under: a non-JS source, a path
                // that escapes the workspace, or an out-of-range sandbox limit are
                // all directory-load errors rather than first-call surprises.
                let p = meta.path.to_string_lossy();
                if !(p.ends_with(".js") || p.ends_with(".mjs")) {
                    return Err(CodeError::Context(format!(
                        "tool {} (kind=script) path `{p}` must point to a .js or .mjs file",
                        path.display()
                    )));
                }
                crate::workspace::validate_relative_pattern(&p, "script path").map_err(|e| {
                    CodeError::Context(format!("tool {} (kind=script): {e}", path.display()))
                })?;
                let limits =
                    validate_script_limits(meta.limits.unwrap_or_default()).map_err(|e| {
                        CodeError::Context(format!("tool {} (kind=script): {e}", path.display()))
                    })?;
                let description = meta
                    .description
                    .map(|d| d.trim().to_string())
                    .filter(|d| !d.is_empty())
                    .unwrap_or_else(|| body.trim().to_string());
                ToolSpec::Script(ScriptToolSpec {
                    name: meta.name.unwrap_or_else(|| file_stem(&path)),
                    description,
                    path: meta.path,
                    // Fail closed: a directory-authored script is semi-trusted and
                    // its inner `ctx.tool` calls are NOT re-checked by the session
                    // permission policy (only by this allow-list + the sandbox), so
                    // an omitted list grants NO tools rather than all of them. The
                    // author must opt each tool in explicitly.
                    allowed_tools: Some(meta.allowed_tools.unwrap_or_default()),
                    limits,
                })
            }
            other => {
                return Err(CodeError::Context(format!(
                    "tool {} has unsupported kind `{other}` (supported: `mcp`, `script`)",
                    path.display()
                )));
            }
        };
        if !seen.insert(spec.name().to_string()) {
            return Err(CodeError::Context(format!(
                "duplicate tool name `{}` in {}",
                spec.name(),
                path.display()
            )));
        }
        out.push(spec);
    }
    Ok(out)
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unnamed")
        .to_string()
}

/// Split a leading `---\n…\n---` YAML frontmatter block from the markdown body.
/// Returns `(None, whole)` when there is no frontmatter.
fn split_frontmatter(content: &str) -> (Option<String>, String) {
    let trimmed = content.trim_start();
    if let Some(rest) = trimmed.strip_prefix("---") {
        let rest = rest.trim_start_matches(['\r', '\n']);
        // Closing fence: a line that is exactly `---`.
        for marker in ["\n---\n", "\n---\r\n", "\n---"] {
            if let Some(end) = rest.find(marker) {
                let front = rest[..end].to_string();
                let body = rest[end + marker.len()..]
                    .trim_start_matches(['\r', '\n'])
                    .to_string();
                return (Some(front), body);
            }
        }
    }
    (None, content.to_string())
}

#[derive(serde::Deserialize)]
struct ScheduleFront {
    cron: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    enabled: Option<bool>,
}

#[derive(serde::Deserialize)]
struct ToolFront {
    kind: String,
}

/// Frontmatter for a `kind = "script"` tool. The `kind` key is ignored here
/// (already matched); unknown keys are tolerated like the other loaders.
#[derive(serde::Deserialize)]
struct ScriptFront {
    #[serde(default)]
    name: Option<String>,
    path: PathBuf,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    allowed_tools: Option<Vec<String>>,
    #[serde(default)]
    limits: Option<ScriptToolLimits>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a fixture agent dir under a unique temp path.
    fn fixture() -> PathBuf {
        let base = std::env::temp_dir().join(format!("a3s-agentdir-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("skills")).unwrap();
        std::fs::create_dir_all(base.join("schedules")).unwrap();
        std::fs::create_dir_all(base.join("tools")).unwrap();
        std::fs::write(
            base.join("instructions.md"),
            "You are a release-notes agent. Be terse and accurate.",
        )
        .unwrap();
        std::fs::write(
            base.join("skills/summarize.md"),
            "---\nname: summarize\ndescription: summarize text\n---\n# Summarize\n",
        )
        .unwrap();
        std::fs::write(
            base.join("schedules/daily.md"),
            "---\ncron: \"0 9 * * *\"\nname: daily-report\n---\nGenerate the daily report and post it.\n",
        )
        .unwrap();
        std::fs::write(
            base.join("tools/github.md"),
            "---\nkind: mcp\nname: github\ntransport: stdio\ncommand: echo\nargs: [\"hi\"]\n---\nGitHub MCP tools.\n",
        )
        .unwrap();
        std::fs::write(
            base.join("tools/search.md"),
            "---\nkind: script\nname: search-auth\npath: scripts/search.js\nallowed_tools: [grep, read]\nlimits:\n  timeoutMs: 30000\n  maxToolCalls: 10\n---\nFind auth-related files.\n",
        )
        .unwrap();
        base
    }

    #[test]
    fn loads_convention_into_slots_and_specs() {
        let dir = fixture();
        let agent = AgentDir::load(&dir).unwrap();

        // instructions.md → role SLOT (not a raw system-prompt override).
        assert_eq!(
            agent.prompt_slots.role.as_deref(),
            Some("You are a release-notes agent. Be terse and accurate.")
        );

        // skills/ → appended to skill_dirs.
        assert!(agent
            .config
            .skill_dirs
            .iter()
            .any(|p| p.ends_with("skills")));

        // schedules/*.md → parsed cron + body prompt.
        assert_eq!(agent.schedules.len(), 1);
        let s = &agent.schedules[0];
        assert_eq!(s.name, "daily-report");
        assert_eq!(s.cron, "0 9 * * *");
        assert_eq!(s.prompt, "Generate the daily report and post it.");
        assert!(s.enabled);

        // tools/*.md → parsed by kind (sorted by path: github.md, then search.md).
        assert_eq!(agent.tools.len(), 2);
        assert_eq!(agent.tools[0].kind(), "mcp");
        assert_eq!(agent.tools[0].name(), "github");

        // kind=script → ScriptToolSpec with pinned path, allow-list, limits; the
        // body becomes the model-facing description.
        assert_eq!(agent.tools[1].kind(), "script");
        assert_eq!(agent.tools[1].name(), "search-auth");
        let ToolSpec::Script(s) = &agent.tools[1] else {
            panic!("expected a script tool");
        };
        assert_eq!(s.path, PathBuf::from("scripts/search.js"));
        assert_eq!(s.description, "Find auth-related files.");
        assert_eq!(
            s.allowed_tools.as_deref(),
            Some(["grep".to_string(), "read".to_string()].as_slice())
        );
        assert_eq!(s.limits.timeout_ms, Some(30000));
        assert_eq!(s.limits.max_tool_calls, Some(10));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// One script tool per file, written under a unique temp dir, must fail to load.
    fn assert_script_tool_load_err(tag: &str, frontmatter: &str) {
        let base = std::env::temp_dir().join(format!("a3s-agentdir-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("tools")).unwrap();
        std::fs::write(base.join("instructions.md"), "role").unwrap();
        std::fs::write(base.join("tools/x.md"), frontmatter).unwrap();
        assert!(
            AgentDir::load(&base).is_err(),
            "expected load error for: {frontmatter}"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn script_tool_non_js_path_is_an_error() {
        // path must end .js/.mjs — fail closed at load, not at first call.
        assert_script_tool_load_err(
            "py",
            "---\nkind: script\nname: x\npath: scripts/run.py\n---\n",
        );
    }

    #[test]
    fn script_tool_escaping_path_is_an_error() {
        // Absolute and parent-traversal paths are rejected at load (fail closed),
        // matching the runtime workspace boundary.
        assert_script_tool_load_err(
            "abs",
            "---\nkind: script\nname: x\npath: /etc/evil.js\n---\n",
        );
        assert_script_tool_load_err(
            "dotdot",
            "---\nkind: script\nname: x\npath: ../../escape.js\n---\n",
        );
    }

    #[test]
    fn script_tool_out_of_range_limits_are_an_error() {
        // Zero disables the tool; u64::MAX disables the sandbox timeout. Both rejected.
        assert_script_tool_load_err(
            "zero",
            "---\nkind: script\nname: x\npath: a.js\nlimits:\n  timeoutMs: 0\n---\n",
        );
        assert_script_tool_load_err(
            "huge",
            "---\nkind: script\nname: x\npath: a.js\nlimits:\n  timeoutMs: 18446744073709551615\n---\n",
        );
        assert_script_tool_load_err(
            "calls",
            "---\nkind: script\nname: x\npath: a.js\nlimits:\n  maxToolCalls: 0\n---\n",
        );
    }

    #[test]
    fn unknown_tool_kind_is_an_error() {
        let base =
            std::env::temp_dir().join(format!("a3s-agentdir-toolkind-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("tools")).unwrap();
        std::fs::write(base.join("instructions.md"), "role").unwrap();
        std::fs::write(base.join("tools/x.md"), "---\nkind: wat\nname: x\n---\n").unwrap();
        assert!(AgentDir::load(&base).is_err());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn duplicate_tool_name_is_an_error() {
        let base =
            std::env::temp_dir().join(format!("a3s-agentdir-tooldup-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("tools")).unwrap();
        std::fs::write(base.join("instructions.md"), "role").unwrap();
        let spec = "---\nkind: mcp\nname: dup\ntransport: stdio\ncommand: echo\n---\n";
        std::fs::write(base.join("tools/a.md"), spec).unwrap();
        std::fs::write(base.join("tools/b.md"), spec).unwrap();
        assert!(AgentDir::load(&base).is_err());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn script_tool_accepts_mjs_and_frontmatter_description_wins_over_body() {
        let base = std::env::temp_dir().join(format!("a3s-agentdir-mjs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("tools")).unwrap();
        std::fs::write(base.join("instructions.md"), "role").unwrap();
        std::fs::write(
            base.join("tools/x.md"),
            "---\nkind: script\nname: x\npath: a.mjs\ndescription: from frontmatter\n---\nbody description\n",
        )
        .unwrap();

        let agent = AgentDir::load(&base).unwrap();
        let ToolSpec::Script(s) = &agent.tools[0] else {
            panic!("expected script tool");
        };
        assert_eq!(s.path, PathBuf::from("a.mjs"), ".mjs is accepted");
        assert_eq!(
            s.description, "from frontmatter",
            "frontmatter description takes precedence over the body"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn script_tool_omitted_allow_list_fails_closed_to_empty() {
        // A directory script with no `allowed_tools` must default to an EMPTY
        // allow-list (no tools), not "all tools" — its inner ctx.tool calls are
        // not re-checked by the session permission policy, so the allow-list is
        // the boundary and an omission must grant nothing.
        let base =
            std::env::temp_dir().join(format!("a3s-agentdir-noallow-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("tools")).unwrap();
        std::fs::write(base.join("instructions.md"), "role").unwrap();
        std::fs::write(
            base.join("tools/x.md"),
            "---\nkind: script\nname: x\npath: a.js\n---\n",
        )
        .unwrap();

        let agent = AgentDir::load(&base).unwrap();
        let ToolSpec::Script(s) = &agent.tools[0] else {
            panic!("expected script tool");
        };
        assert_eq!(
            s.allowed_tools.as_deref(),
            Some([].as_slice()),
            "omitted allowed_tools must fail closed to an empty list, not None/all"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn missing_instructions_is_an_error() {
        let base = std::env::temp_dir().join(format!("a3s-agentdir-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        assert!(AgentDir::load(&base).is_err());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn frontmatter_split_handles_no_frontmatter() {
        let (f, b) = split_frontmatter("no frontmatter here");
        assert!(f.is_none());
        assert_eq!(b, "no frontmatter here");
    }
}
