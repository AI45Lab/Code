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
//! ├── channels/          (optional)  *.{md,acl} inbound adapters — parsed here, served later.
//! └── tools/             (optional)  *.md tool specs (`kind: mcp`) → MCP servers
//! │                                 registered into the session (sandboxed-script
//! │                                 `kind` is the next increment).
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

/// An inbound channel adapter spec, parsed from `channels/<name>.{md,acl}`.
///
/// Parsed so the directory convention is complete; the serve layer does not yet
/// implement adapters (channels are design-only for now). `frontmatter` carries
/// the raw adapter options for whichever adapter eventually handles `kind`.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelSpec {
    /// Channel name (frontmatter `name`, else the file stem).
    pub name: String,
    /// Adapter kind: `http`, `slack`, `discord`, …
    pub kind: String,
    /// Raw frontmatter (YAML) for the adapter to interpret.
    pub frontmatter: String,
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
}

impl ToolSpec {
    /// The tool/server name (registry key; unique within `tools/`).
    pub fn name(&self) -> &str {
        match self {
            ToolSpec::Mcp(cfg) => &cfg.name,
        }
    }

    /// The spec kind discriminant (currently only `mcp`).
    pub fn kind(&self) -> &str {
        match self {
            ToolSpec::Mcp(_) => "mcp",
        }
    }
}

/// A loaded agent directory: synthesized [`CodeConfig`] + prompt slots + parsed
/// schedule/channel specs. Build a session from `config` + `prompt_slots`.
///
/// Distinct from [`CodeConfig::agent_dirs`](crate::config::CodeConfig) /
/// `register_agent_dir`, which scan a directory for **worker/subagent**
/// definitions. An `AgentDir` is the eve-style *primary* agent — the directory
/// that defines this agent's prompt, skills, schedules, and channels.
#[derive(Debug, Clone)]
pub struct AgentDir {
    pub dir: PathBuf,
    pub config: CodeConfig,
    pub prompt_slots: SystemPromptSlots,
    pub schedules: Vec<ScheduleSpec>,
    pub channels: Vec<ChannelSpec>,
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
        let channels = load_channels(&dir.join("channels"))?;
        let tools = load_tools(&dir.join("tools"))?;

        Ok(Self {
            dir,
            config,
            prompt_slots,
            schedules,
            channels,
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

fn load_channels(dir: &Path) -> Result<Vec<ChannelSpec>> {
    let mut out = Vec::new();
    for path in md_files(dir, &["md", "acl"])? {
        let content = std::fs::read_to_string(&path)
            .map_err(|e| CodeError::Context(format!("read {}: {e}", path.display())))?;
        let (front, _body) = split_frontmatter(&content);
        let front = front.ok_or_else(|| {
            CodeError::Context(format!(
                "channel {} has no frontmatter (need `kind:`)",
                path.display()
            ))
        })?;
        let meta: ChannelFront = serde_yaml::from_str(&front).map_err(|e| {
            CodeError::Context(format!("channel {} frontmatter: {e}", path.display()))
        })?;
        out.push(ChannelSpec {
            name: meta.name.unwrap_or_else(|| file_stem(&path)),
            kind: meta.kind,
            frontmatter: front,
        });
    }
    Ok(out)
}

fn load_tools(dir: &Path) -> Result<Vec<ToolSpec>> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for path in md_files(dir, &["md"])? {
        let content = std::fs::read_to_string(&path)
            .map_err(|e| CodeError::Context(format!("read {}: {e}", path.display())))?;
        let (front, _body) = split_frontmatter(&content);
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
            other => {
                return Err(CodeError::Context(format!(
                    "tool {} has unsupported kind `{other}` (supported: `mcp`)",
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
struct ChannelFront {
    kind: String,
    #[serde(default)]
    name: Option<String>,
}

#[derive(serde::Deserialize)]
struct ToolFront {
    kind: String,
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
        std::fs::create_dir_all(base.join("channels")).unwrap();
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
            base.join("channels/web.md"),
            "---\nkind: http\nport: 8787\n---\nInbound HTTP channel.\n",
        )
        .unwrap();
        std::fs::write(
            base.join("tools/github.md"),
            "---\nkind: mcp\nname: github\ntransport: stdio\ncommand: echo\nargs: [\"hi\"]\n---\nGitHub MCP tools.\n",
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

        // channels/*.md → parsed kind (adapters not yet implemented).
        assert_eq!(agent.channels.len(), 1);
        assert_eq!(agent.channels[0].kind, "http");

        // tools/*.md (kind=mcp) → parsed MCP server spec.
        assert_eq!(agent.tools.len(), 1);
        assert_eq!(agent.tools[0].kind(), "mcp");
        assert_eq!(agent.tools[0].name(), "github");

        let _ = std::fs::remove_dir_all(&dir);
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
