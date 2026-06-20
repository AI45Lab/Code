//! End-to-end integration test for the filesystem-first agent directory
//! convention: a single on-disk directory with EVERY supported sub-convention
//! (instructions, agent.acl, skills/, schedules/, tools/) loads into a
//! fully-populated [`AgentDir`]. Hermetic — no provider, no network.

use std::fs;
use std::path::{Path, PathBuf};

use a3s_code_core::config::{AgentDir, ToolSpec};

/// Write `content` to `dir/rel`, creating parent dirs.
fn write(dir: &Path, rel: &str, content: &str) {
    let path = dir.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, content).unwrap();
}

/// A realistic agent dir exercising every sub-convention at once.
fn build_full_agent_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();

    write(p, "instructions.md", "You are a release agent. Be terse.\n");

    // agent.acl → CodeConfig override (observable via default_model).
    write(
        p,
        "agent.acl",
        r#"
default_model = "anthropic/claude-sonnet-4-20250514"
providers "anthropic" {
  api_key = "test-key"
  models "claude-sonnet-4-20250514" { name = "Claude Sonnet 4" }
}
"#,
    );

    // skills/ → appended to skill_dirs.
    write(
        p,
        "skills/summarize.md",
        "---\nname: summarize\ndescription: summarize text\n---\n# Summarize\n",
    );

    // schedules/ → one enabled (named), one explicitly disabled.
    write(
        p,
        "schedules/daily.md",
        "---\ncron: \"0 9 * * *\"\nname: daily-report\n---\nGenerate the daily report.\n",
    );
    write(
        p,
        "schedules/paused.md",
        "---\ncron: \"*/5 * * * *\"\nenabled: false\n---\nThis one is paused.\n",
    );

    // tools/ → one mcp, one sandboxed script.
    write(
        p,
        "tools/github.md",
        "---\nkind: mcp\nname: github\ntransport: stdio\ncommand: echo\nargs: [\"hi\"]\n---\nGitHub MCP tools.\n",
    );
    write(
        p,
        "tools/search.md",
        "---\nkind: script\nname: search-auth\npath: scripts/search.js\nallowed_tools: [grep, read]\nlimits:\n  timeoutMs: 20000\n  maxToolCalls: 8\n---\nFind auth-related files.\n",
    );

    dir
}

#[test]
fn full_agent_dir_loads_every_convention() {
    let dir = build_full_agent_dir();
    let agent = AgentDir::load(dir.path()).expect("agent dir loads");

    // instructions.md → role slot (trimmed), NOT a system-prompt override.
    assert_eq!(
        agent.prompt_slots.role.as_deref(),
        Some("You are a release agent. Be terse.")
    );

    // agent.acl → CodeConfig override is applied.
    assert_eq!(
        agent.config.default_model.as_deref(),
        Some("anthropic/claude-sonnet-4-20250514"),
        "agent.acl default_model must override the default config"
    );

    // skills/ → appended to skill_dirs.
    assert!(
        agent
            .config
            .skill_dirs
            .iter()
            .any(|d| d.ends_with("skills")),
        "skills/ dir must be appended to skill_dirs"
    );

    // schedules/ → both parsed (sorted by path: daily, then paused), enabled flags
    // preserved; the serve layer is what skips disabled ones.
    assert_eq!(agent.schedules.len(), 2);
    let daily = agent
        .schedules
        .iter()
        .find(|s| s.name == "daily-report")
        .expect("named schedule");
    assert_eq!(daily.cron, "0 9 * * *");
    assert_eq!(daily.prompt, "Generate the daily report.");
    assert!(daily.enabled);
    let paused = agent
        .schedules
        .iter()
        .find(|s| s.name == "paused")
        .expect("file-stem-named schedule");
    assert!(!paused.enabled, "enabled: false must be honored");

    // tools/ → one mcp + one script (sorted by path: github, then search).
    assert_eq!(agent.tools.len(), 2);
    let gh = agent.tools.iter().find(|t| t.name() == "github").unwrap();
    assert_eq!(gh.kind(), "mcp");

    let search = agent
        .tools
        .iter()
        .find(|t| t.name() == "search-auth")
        .expect("script tool");
    assert_eq!(search.kind(), "script");
    let ToolSpec::Script(spec) = search else {
        panic!("expected a script tool spec");
    };
    assert_eq!(spec.path, PathBuf::from("scripts/search.js"));
    assert_eq!(spec.description, "Find auth-related files.");
    assert_eq!(
        spec.allowed_tools.as_deref(),
        Some(["grep".to_string(), "read".to_string()].as_slice())
    );
    assert_eq!(spec.limits.timeout_ms, Some(20000));
    assert_eq!(spec.limits.max_tool_calls, Some(8));
}

/// A minimal dir — only the required `instructions.md`, no optional subdirs — loads
/// with empty spec lists and the default config.
#[test]
fn minimal_agent_dir_loads_with_empty_specs() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "instructions.md", "Minimal agent.\n");

    let agent = AgentDir::load(dir.path()).expect("minimal dir loads");
    assert_eq!(agent.prompt_slots.role.as_deref(), Some("Minimal agent."));
    assert!(agent.schedules.is_empty());
    assert!(agent.tools.is_empty());
    // No agent.acl → default config (no default_model pinned).
    assert!(agent.config.default_model.is_none());
}

/// Loading a path that is not a directory, or a dir missing the required
/// instructions.md, is an error (fail closed).
#[test]
fn missing_dir_or_instructions_is_an_error() {
    // Not a directory.
    assert!(AgentDir::load("/nonexistent/a3s/agent/dir").is_err());

    // Directory exists but has no instructions.md.
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir_all(dir.path().join("skills")).unwrap();
    assert!(
        AgentDir::load(dir.path()).is_err(),
        "instructions.md is required"
    );
}
