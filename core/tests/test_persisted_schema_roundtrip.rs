//! Persisted-schema round-trip fuzz + cross-version compatibility (#31).
//!
//! Everything the framework writes through a [`SessionStore`] is JSON via
//! `serde_json`, and backward compatibility rests entirely on `#[serde(default)]`.
//! This suite guards three properties that unit tests per type don't cover
//! together:
//!
//! 1. **Round-trip stability** — for a large, deterministically-generated
//!    corpus of every persisted type, `to_string` is idempotent across a
//!    serialize → deserialize → re-serialize cycle. This catches dropped
//!    fields, mis-tagged enums, and the ambiguity hazard of the
//!    `#[serde(untagged)]` `ToolResultContentField`.
//! 2. **Backward compat** — payloads written by an *older* build (missing
//!    fields added in 3.3.0) still deserialize, falling back to defaults.
//! 3. **Forward compat** — payloads written by a *newer* build (carrying
//!    unknown extra fields) still deserialize. This is what makes rolling,
//!    mixed-version cluster upgrades safe; it would break the day someone
//!    adds `#[serde(deny_unknown_fields)]`.
//!
//! Plus the one explicit version contract: a [`LoopCheckpoint`] from a
//! *future*, incompatible schema version must be **rejected** on load.
//!
//! No `proptest`/`arbitrary` dependency — generation is a small seeded
//! xorshift so failures are reproducible from the printed seed.

use serde::de::DeserializeOwned;
use serde::Serialize;

use a3s_code_core::llm::{
    ContentBlock, ImageSource, Message, TokenUsage, ToolResultContent, ToolResultContentField,
};
use a3s_code_core::loop_checkpoint::{LoopCheckpoint, LOOP_CHECKPOINT_SCHEMA_VERSION};
use a3s_code_core::orchestration::{
    AgentStepSpec, StepOutcome, WorkflowCheckpoint, WorkflowStepRecord,
    WORKFLOW_CHECKPOINT_SCHEMA_VERSION,
};
use a3s_code_core::permissions::{PermissionPolicy, PermissionRule};
use a3s_code_core::planning::Task;
use a3s_code_core::run::{RunEventRecord, RunRecord, RunSnapshot, RunStatus};
use a3s_code_core::store::{
    ContextUsage, MemorySessionStore, SessionConfig, SessionData, SessionState, SessionStore,
};
use a3s_code_core::subagent_task_tracker::{
    SubagentProgressEntry, SubagentStatus, SubagentTaskSnapshot,
};
use a3s_code_core::trace::{TraceEvent, TraceEventKind, TRACE_EVENT_SCHEMA};
use a3s_code_core::verification::{
    VerificationCheck, VerificationReport, VerificationStatus, VERIFICATION_REPORT_SCHEMA,
};
use a3s_code_core::AgentEvent;

// ─────────────────────────────────────────────────────────────────────
// Deterministic generator (xorshift64*) — reproducible, dependency-free.
// ─────────────────────────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Avoid the all-zero state, which xorshift cannot escape.
        Rng(seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0x1234_5678))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            0
        } else {
            self.next() % n
        }
    }
    fn boolean(&mut self) -> bool {
        self.next() & 1 == 1
    }
    fn usize_below(&mut self, n: u64) -> usize {
        self.below(n) as usize
    }
    fn u64_small(&mut self) -> u64 {
        self.next() % 1_000_000_000
    }
    fn i32_small(&mut self) -> i32 {
        (self.next() % 512) as i32 - 256
    }
    /// Finite f64 only — serde_json renders NaN/Inf as `null`, which would
    /// not round-trip. We deliberately stay in the finite range so the
    /// stability property holds.
    fn f64_finite(&mut self) -> f64 {
        (self.below(1_000_000) as f64) / 100.0
    }
    /// A string drawn from a pool of round-trip hazards: empty, unicode,
    /// embedded quotes/newlines, and JSON-looking text (the last is the
    /// trap for the untagged `ToolResultContentField` — a string that
    /// *looks* like an array must still come back as `Text`, not `Blocks`).
    fn string(&mut self) -> String {
        const POOL: &[&str] = &[
            "",
            "ok",
            "hello world",
            "with \"double\" quotes",
            "line\nbreak\tand tab",
            "unicode ü ñ 日本語 🚀",
            "{\"looks\":\"like json\"}",
            "[1, 2, 3]",
            "null",
            "12345",
            "trailing space ",
            "ümlaut-prefixed",
        ];
        let base = POOL[self.usize_below(POOL.len() as u64)].to_string();
        // Occasionally suffix with the seed so distinct generations differ.
        if self.boolean() {
            format!("{base}#{}", self.below(1000))
        } else {
            base
        }
    }
    fn opt_string(&mut self) -> Option<String> {
        if self.boolean() {
            Some(self.string())
        } else {
            None
        }
    }
    fn opt_u64(&mut self) -> Option<u64> {
        if self.boolean() {
            Some(self.u64_small())
        } else {
            None
        }
    }
    fn opt_usize(&mut self) -> Option<usize> {
        if self.boolean() {
            Some(self.usize_below(1_000_000))
        } else {
            None
        }
    }
    fn string_vec(&mut self, max: u64) -> Vec<String> {
        (0..self.below(max + 1)).map(|_| self.string()).collect()
    }
    /// Small `serde_json::Value` whose re-serialization is order-stable
    /// (object keys are fixed and few).
    fn json_value(&mut self, depth: u8) -> serde_json::Value {
        use serde_json::Value;
        match self.below(if depth == 0 { 4 } else { 6 }) {
            0 => Value::Null,
            1 => Value::Bool(self.boolean()),
            2 => Value::from(self.i32_small()),
            3 => Value::from(self.string()),
            4 => {
                let n = self.below(3);
                Value::Array((0..n).map(|_| self.json_value(depth - 1)).collect())
            }
            _ => {
                let mut map = serde_json::Map::new();
                map.insert("k0".into(), self.json_value(depth - 1));
                if self.boolean() {
                    map.insert("k1".into(), Value::from(self.string()));
                }
                Value::Object(map)
            }
        }
    }
    /// Like [`json_value`], but never a top-level JSON `null`.
    ///
    /// For an `Option<serde_json::Value>` field, serde collapses
    /// `Some(Value::Null)` to `None` on deserialize (JSON `null` → `None`),
    /// so that one value cannot round-trip. The framework never stores
    /// `Some(Null)` in such fields (they hold `None` or a real object), so
    /// excluding it keeps the corpus representative. Nested nulls are fine.
    fn json_value_non_null(&mut self, depth: u8) -> serde_json::Value {
        let v = self.json_value(depth);
        if v.is_null() {
            serde_json::Value::Bool(true)
        } else {
            v
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Per-type generators.
// ─────────────────────────────────────────────────────────────────────

fn gen_token_usage(rng: &mut Rng) -> TokenUsage {
    TokenUsage {
        prompt_tokens: rng.usize_below(100_000),
        completion_tokens: rng.usize_below(100_000),
        total_tokens: rng.usize_below(200_000),
        cache_read_tokens: rng.opt_usize(),
        cache_write_tokens: rng.opt_usize(),
    }
}

fn gen_image_source(rng: &mut Rng) -> ImageSource {
    ImageSource {
        source_type: "base64".to_string(),
        media_type: "image/png".to_string(),
        data: rng.string(),
    }
}

fn gen_tool_result_content(rng: &mut Rng) -> ToolResultContent {
    if rng.boolean() {
        ToolResultContent::Text { text: rng.string() }
    } else {
        ToolResultContent::Image {
            source: gen_image_source(rng),
        }
    }
}

fn gen_tool_result_field(rng: &mut Rng) -> ToolResultContentField {
    // Exercise BOTH untagged arms, including Text holding JSON-looking
    // strings that must not be mis-parsed as the array arm.
    if rng.boolean() {
        ToolResultContentField::Text(rng.string())
    } else {
        let n = rng.below(3);
        ToolResultContentField::Blocks((0..n).map(|_| gen_tool_result_content(rng)).collect())
    }
}

fn gen_content_block(rng: &mut Rng) -> ContentBlock {
    match rng.below(4) {
        0 => ContentBlock::Text { text: rng.string() },
        1 => ContentBlock::Image {
            source: gen_image_source(rng),
        },
        2 => ContentBlock::ToolUse {
            id: rng.string(),
            name: rng.string(),
            input: rng.json_value(2),
        },
        _ => ContentBlock::ToolResult {
            tool_use_id: rng.string(),
            content: gen_tool_result_field(rng),
            is_error: if rng.boolean() {
                Some(rng.boolean())
            } else {
                None
            },
        },
    }
}

fn gen_message(rng: &mut Rng) -> Message {
    let roles = ["user", "assistant", "system", "tool"];
    let n = rng.below(4);
    Message {
        role: roles[rng.usize_below(roles.len() as u64)].to_string(),
        content: (0..n).map(|_| gen_content_block(rng)).collect(),
        reasoning_content: rng.opt_string(),
    }
}

fn gen_verification_status(rng: &mut Rng) -> VerificationStatus {
    match rng.below(4) {
        0 => VerificationStatus::Passed,
        1 => VerificationStatus::Failed,
        2 => VerificationStatus::NeedsReview,
        _ => VerificationStatus::Skipped,
    }
}

fn gen_verification_check(rng: &mut Rng) -> VerificationCheck {
    VerificationCheck {
        id: rng.string(),
        kind: rng.string(),
        description: rng.string(),
        status: gen_verification_status(rng),
        required: rng.boolean(),
        suggested_tools: rng.string_vec(3),
        evidence_uris: rng.string_vec(2),
        residual_risk: rng.opt_string(),
    }
}

fn gen_verification_report(rng: &mut Rng) -> VerificationReport {
    let n = rng.below(3);
    VerificationReport {
        schema: VERIFICATION_REPORT_SCHEMA.to_string(),
        subject: rng.string(),
        status: gen_verification_status(rng),
        checks: (0..n).map(|_| gen_verification_check(rng)).collect(),
        residual_risks: rng.string_vec(3),
    }
}

fn gen_trace_event(rng: &mut Rng) -> TraceEvent {
    TraceEvent {
        schema: TRACE_EVENT_SCHEMA.to_string(),
        kind: if rng.boolean() {
            TraceEventKind::ToolExecution
        } else {
            TraceEventKind::ProgramExecution
        },
        name: rng.string(),
        success: rng.boolean(),
        exit_code: rng.i32_small(),
        duration_ms: rng.u64_small(),
        output_bytes: rng.usize_below(1_000_000),
        metadata_keys: rng.string_vec(3),
        artifact_uris: rng.string_vec(2),
        details: if rng.boolean() {
            Some(rng.json_value_non_null(2))
        } else {
            None
        },
    }
}

fn gen_subagent_task(rng: &mut Rng) -> SubagentTaskSnapshot {
    let status = match rng.below(4) {
        0 => SubagentStatus::Running,
        1 => SubagentStatus::Completed,
        2 => SubagentStatus::Failed,
        _ => SubagentStatus::Cancelled,
    };
    let progress_n = rng.below(3);
    SubagentTaskSnapshot {
        task_id: rng.string(),
        parent_session_id: rng.string(),
        child_session_id: rng.string(),
        agent: rng.string(),
        description: rng.string(),
        status,
        started_ms: rng.u64_small(),
        updated_ms: rng.u64_small(),
        finished_ms: rng.opt_u64(),
        output: rng.opt_string(),
        success: if rng.boolean() {
            Some(rng.boolean())
        } else {
            None
        },
        progress: (0..progress_n)
            .map(|_| SubagentProgressEntry {
                timestamp_ms: rng.u64_small(),
                status: rng.string(),
                metadata: rng.json_value(2),
            })
            .collect(),
    }
}

/// Cheap-to-construct, representative `AgentEvent`s — including the three
/// variants added in 3.3.0 (`BudgetThresholdHit`, `PassivationRequested`,
/// `PeerInvocation`), since these are persisted inside `RunRecord.events`
/// and an old reader/writer mismatch around them is the live cross-version
/// risk.
fn gen_agent_event(rng: &mut Rng) -> AgentEvent {
    match rng.below(9) {
        0 => AgentEvent::Start {
            prompt: rng.string(),
        },
        1 => AgentEvent::TurnStart {
            turn: rng.usize_below(100),
        },
        2 => AgentEvent::ToolStart {
            id: rng.string(),
            name: rng.string(),
        },
        3 => AgentEvent::ToolEnd {
            id: rng.string(),
            name: rng.string(),
            output: rng.string(),
            exit_code: rng.i32_small(),
            metadata: if rng.boolean() {
                Some(rng.json_value_non_null(2))
            } else {
                None
            },
            error_kind: None,
        },
        4 => AgentEvent::Error {
            message: rng.string(),
        },
        5 => AgentEvent::TurnEnd {
            turn: rng.usize_below(100),
            usage: gen_token_usage(rng),
        },
        6 => AgentEvent::BudgetThresholdHit {
            resource: rng.string(),
            kind: rng.string(),
            consumed: rng.f64_finite(),
            limit: rng.f64_finite(),
            message: rng.opt_string(),
        },
        7 => AgentEvent::PassivationRequested {
            reason: rng.string(),
            deadline_ms: rng.opt_u64(),
        },
        _ => AgentEvent::PeerInvocation {
            from_session_id: rng.string(),
            from_tenant_id: rng.opt_string(),
            correlation_id: rng.opt_string(),
        },
    }
}

fn gen_run_status(rng: &mut Rng) -> RunStatus {
    match rng.below(7) {
        0 => RunStatus::Created,
        1 => RunStatus::Planning,
        2 => RunStatus::Executing,
        3 => RunStatus::Verifying,
        4 => RunStatus::Completed,
        5 => RunStatus::Failed,
        _ => RunStatus::Cancelled,
    }
}

fn gen_run_record(rng: &mut Rng) -> RunRecord {
    let event_n = rng.below(5);
    RunRecord {
        snapshot: RunSnapshot {
            id: rng.string(),
            session_id: rng.string(),
            status: gen_run_status(rng),
            prompt: rng.string(),
            created_at_ms: rng.u64_small(),
            updated_at_ms: rng.u64_small(),
            result_text: rng.opt_string(),
            error: rng.opt_string(),
            event_count: rng.usize_below(1000),
        },
        events: (0..event_n)
            .map(|i| RunEventRecord {
                sequence: i as usize,
                timestamp_ms: rng.u64_small(),
                event: gen_agent_event(rng),
            })
            .collect(),
    }
}

fn gen_loop_checkpoint(rng: &mut Rng) -> LoopCheckpoint {
    let msg_n = rng.below(4);
    let report_n = rng.below(2);
    LoopCheckpoint {
        // Generators stay at-or-below the current version; the future-version
        // rejection is tested separately.
        schema_version: rng.below(u64::from(LOOP_CHECKPOINT_SCHEMA_VERSION) + 1) as u32,
        run_id: rng.string(),
        session_id: rng.string(),
        turn: rng.usize_below(1000),
        messages: (0..msg_n).map(|_| gen_message(rng)).collect(),
        total_usage: gen_token_usage(rng),
        tool_calls_count: rng.usize_below(1000),
        verification_reports: (0..report_n)
            .map(|_| gen_verification_report(rng))
            .collect(),
        checkpoint_ms: rng.u64_small(),
    }
}

fn gen_permission_rule(rng: &mut Rng) -> PermissionRule {
    const RULES: &[&str] = &[
        "Bash",
        "Bash(cargo:*)",
        "Read(src/**/*.rs)",
        "Grep(*)",
        "mcp__pencil",
        "Write(/tmp/*)",
    ];
    PermissionRule::new(RULES[rng.usize_below(RULES.len() as u64)])
}

fn gen_permission_policy(rng: &mut Rng) -> PermissionPolicy {
    PermissionPolicy {
        deny: (0..rng.below(2))
            .map(|_| gen_permission_rule(rng))
            .collect(),
        allow: (0..rng.below(3))
            .map(|_| gen_permission_rule(rng))
            .collect(),
        ask: (0..rng.below(2))
            .map(|_| gen_permission_rule(rng))
            .collect(),
        ..Default::default()
    }
}

/// The largest persisted root type, and the carrier of the 3.3.0 identity
/// fields (tenant/principal/template/correlation). Exercises a nested
/// `PermissionPolicy` (itself holding the persisted untagged `PermissionRule`),
/// the `#[serde(alias = "todos")]` `tasks` field, and the `#[serde(skip)]`
/// `hook_engine` (safe: skipped in both directions).
fn gen_session_data(rng: &mut Rng) -> SessionData {
    let state = match rng.below(5) {
        0 => SessionState::Unknown,
        1 => SessionState::Active,
        2 => SessionState::Paused,
        3 => SessionState::Completed,
        _ => SessionState::Error,
    };
    let config = SessionConfig {
        name: rng.string(),
        workspace: rng.string(),
        system_prompt: rng.opt_string(),
        permission_policy: if rng.boolean() {
            Some(gen_permission_policy(rng))
        } else {
            None
        },
        ..Default::default()
    };
    SessionData {
        id: rng.string(),
        config,
        state,
        messages: (0..rng.below(3)).map(|_| gen_message(rng)).collect(),
        context_usage: ContextUsage::default(),
        total_usage: gen_token_usage(rng),
        total_cost: rng.f64_finite(),
        model_name: rng.opt_string(),
        cost_records: Vec::new(),
        tool_names: rng.string_vec(3),
        thinking_enabled: rng.boolean(),
        thinking_budget: rng.opt_usize(),
        created_at: rng.u64_small() as i64,
        updated_at: rng.u64_small() as i64,
        llm_config: None,
        tasks: (0..rng.below(3))
            .map(|i| Task::new(format!("t{i}"), rng.string()))
            .collect(),
        parent_id: rng.opt_string(),
        tenant_id: rng.opt_string(),
        principal: rng.opt_string(),
        agent_template_id: rng.opt_string(),
        correlation_id: rng.opt_string(),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Property helpers.
// ─────────────────────────────────────────────────────────────────────

/// Serialize → deserialize → re-serialize and assert the two JSON strings
/// are identical. Stronger than `PartialEq` (which several persisted types
/// don't derive) and it directly detects dropped fields and enum-tag drift.
fn assert_roundtrip_stable<T>(value: &T, label: &str)
where
    T: Serialize + DeserializeOwned,
{
    let s1 = serde_json::to_string(value).unwrap_or_else(|e| panic!("{label}: serialize: {e}"));
    let back: T = serde_json::from_str(&s1)
        .unwrap_or_else(|e| panic!("{label}: deserialize failed: {e}\n  json: {s1}"));
    let s2 = serde_json::to_string(&back).unwrap_or_else(|e| panic!("{label}: re-serialize: {e}"));
    assert_eq!(s1, s2, "{label}: round-trip not stable");
}

/// A newer writer added fields this build doesn't know. The old reader must
/// ignore them, not reject — the invariant that keeps rolling cluster
/// upgrades safe. Guards against a future `#[serde(deny_unknown_fields)]`.
fn assert_unknown_fields_ignored<T>(value: &T, label: &str)
where
    T: Serialize + DeserializeOwned,
{
    let mut v = serde_json::to_value(value).unwrap_or_else(|e| panic!("{label}: to_value: {e}"));
    let serde_json::Value::Object(map) = &mut v else {
        panic!("{label}: expected a JSON object at the persisted root");
    };
    map.insert(
        "__field_added_in_a_future_version__".into(),
        serde_json::json!({ "nested": true, "n": 7 }),
    );
    map.insert("__another_unknown__".into(), serde_json::Value::Bool(true));
    let s = serde_json::to_string(&v).unwrap();
    let _back: T = serde_json::from_str(&s).unwrap_or_else(|e| {
        panic!("{label}: unknown fields must be ignored, got: {e}\n  json: {s}")
    });
}

// ─────────────────────────────────────────────────────────────────────
// 1. Round-trip stability fuzz across the persisted graph.
// ─────────────────────────────────────────────────────────────────────

// Orchestration persisted types (#43): serializable, and WorkflowCheckpoint
// is explicitly designed for cross-node migration — so round-trip stability
// and forward/backward compat matter for exactly these.
fn gen_agent_step_spec(rng: &mut Rng) -> AgentStepSpec {
    AgentStepSpec {
        task_id: rng.string(),
        agent: rng.string(),
        description: rng.string(),
        prompt: rng.string(),
        max_steps: rng.opt_usize(),
        parent_session_id: rng.opt_string(),
        // Exercise BOTH arms — output_schema is skip_serializing_if=None.
        output_schema: if rng.boolean() {
            Some(rng.json_value_non_null(2))
        } else {
            None
        },
    }
}

fn gen_step_outcome(rng: &mut Rng) -> StepOutcome {
    StepOutcome {
        task_id: rng.string(),
        session_id: rng.string(),
        agent: rng.string(),
        output: rng.string(),
        success: rng.boolean(),
        structured: if rng.boolean() {
            Some(rng.json_value_non_null(2))
        } else {
            None
        },
    }
}

fn gen_workflow_checkpoint(rng: &mut Rng) -> WorkflowCheckpoint {
    let n = rng.below(4);
    WorkflowCheckpoint {
        // Stay at-or-below the current version (future versions are rejected).
        schema_version: rng.below(u64::from(WORKFLOW_CHECKPOINT_SCHEMA_VERSION) + 1) as u32,
        workflow_id: rng.string(),
        steps: (0..n)
            .map(|_| WorkflowStepRecord {
                task_id: rng.string(),
                outcome: gen_step_outcome(rng),
            })
            .collect(),
        checkpoint_ms: rng.u64_small(),
    }
}

#[test]
fn fuzz_roundtrip_all_persisted_types() {
    const SEEDS: u64 = 600;
    for seed in 0..SEEDS {
        let mut rng = Rng::new(seed);
        assert_roundtrip_stable(
            &gen_loop_checkpoint(&mut rng),
            &format!("LoopCheckpoint/{seed}"),
        );
        assert_roundtrip_stable(&gen_run_record(&mut rng), &format!("RunRecord/{seed}"));
        assert_roundtrip_stable(
            &gen_subagent_task(&mut rng),
            &format!("SubagentTask/{seed}"),
        );
        assert_roundtrip_stable(&gen_trace_event(&mut rng), &format!("TraceEvent/{seed}"));
        assert_roundtrip_stable(
            &gen_verification_report(&mut rng),
            &format!("VerificationReport/{seed}"),
        );
        assert_roundtrip_stable(&gen_message(&mut rng), &format!("Message/{seed}"));
        assert_roundtrip_stable(&gen_session_data(&mut rng), &format!("SessionData/{seed}"));
        assert_roundtrip_stable(
            &gen_agent_step_spec(&mut rng),
            &format!("AgentStepSpec/{seed}"),
        );
        assert_roundtrip_stable(&gen_step_outcome(&mut rng), &format!("StepOutcome/{seed}"));
        assert_roundtrip_stable(
            &gen_workflow_checkpoint(&mut rng),
            &format!("WorkflowCheckpoint/{seed}"),
        );
    }
}

/// Targeted regression for the `#[serde(untagged)]` `ToolResultContentField`:
/// a plain-text result whose text *looks* like a JSON array must round-trip
/// back as `Text`, never as the `Blocks` arm.
#[test]
fn untagged_tool_result_text_is_not_misparsed_as_blocks() {
    let json_looking =
        ToolResultContentField::Text("[{\"type\":\"text\",\"text\":\"x\"}]".to_string());
    let s = serde_json::to_string(&json_looking).unwrap();
    // A `Text` serializes as a bare JSON string, so the array brackets are
    // escaped inside it, not structural.
    assert!(
        s.starts_with('"'),
        "Text must serialize as a JSON string, got: {s}"
    );
    let back: ToolResultContentField = serde_json::from_str(&s).unwrap();
    assert!(
        matches!(back, ToolResultContentField::Text(_)),
        "JSON-looking text must deserialize back as Text, not Blocks"
    );
    assert_roundtrip_stable(&json_looking, "ToolResultContentField::Text(json-looking)");
}

/// `PermissionRule` (persisted under `SessionData.config.permission_policy`)
/// has a hand-written `Deserialize` over an internal `#[serde(untagged)]`
/// repr that accepts BOTH a bare string and a `{"rule": "..."}` object, with
/// the parsed `tool_name`/`arg_pattern` carried in `#[serde(skip)]` fields.
/// Both input forms must converge to the same rule, and the struct must
/// round-trip stably (Serialize emits the object form; the skipped parsed
/// fields are re-derived on load, which `Eq` checks).
#[test]
fn permission_rule_accepts_both_string_and_struct_forms() {
    let expected = PermissionRule::new("Bash(cargo:*)");
    let from_bare: PermissionRule =
        serde_json::from_str("\"Bash(cargo:*)\"").expect("bare-string form must parse");
    let from_struct: PermissionRule =
        serde_json::from_str(r#"{"rule":"Bash(cargo:*)"}"#).expect("struct form must parse");
    assert_eq!(
        from_bare, expected,
        "bare-string and constructed must match"
    );
    assert_eq!(
        from_struct, expected,
        "struct form and constructed must match"
    );
    assert_roundtrip_stable(&expected, "PermissionRule");
}

// ─────────────────────────────────────────────────────────────────────
// 2. Backward compatibility — older payloads (missing 3.3.0 fields) load.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn backward_compat_pre_v1_loop_checkpoint_loads_with_defaults() {
    // A pre-3.3.0 checkpoint: no `schema_version`, no `verification_reports`,
    // and a TokenUsage missing the cache fields entirely.
    let json = r#"{
        "run_id": "run-old",
        "session_id": "sess-old",
        "turn": 2,
        "messages": [],
        "total_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "tool_calls_count": 3,
        "checkpoint_ms": 1700000000000
    }"#;
    let cp: LoopCheckpoint = serde_json::from_str(json).expect("pre-v1 checkpoint must load");
    assert_eq!(
        cp.schema_version, 0,
        "missing version defaults to pre-v1 zero"
    );
    assert!(cp.verification_reports.is_empty());
    assert_eq!(cp.total_usage.cache_read_tokens, None);
    assert_eq!(cp.tool_calls_count, 3);
    // And a pre-v1 checkpoint is still loadable (older, not future).
    cp.ensure_loadable().expect("pre-v1 must be loadable");
}

#[test]
fn backward_compat_minimal_subagent_task_loads() {
    // An older writer omitted the optional terminal fields.
    let json = r#"{
        "task_id": "t1",
        "parent_session_id": "p1",
        "child_session_id": "c1",
        "agent": "explorer",
        "description": "scan",
        "status": "running",
        "started_ms": 1,
        "updated_ms": 2,
        "progress": []
    }"#;
    let task: SubagentTaskSnapshot = serde_json::from_str(json).expect("minimal task must load");
    assert_eq!(task.finished_ms, None);
    assert_eq!(task.output, None);
    assert_eq!(task.success, None);
    assert!(matches!(task.status, SubagentStatus::Running));
}

#[test]
fn backward_compat_session_data_without_identity_fields_loads() {
    // A pre-3.3.0 session payload predates the identity fields entirely.
    // It must load, with each identity field falling back to None.
    let mut rng = Rng::new(123);
    let mut sd = gen_session_data(&mut rng);
    sd.tenant_id = Some("acme".to_string());
    sd.principal = Some("svc-bot".to_string());
    sd.agent_template_id = Some("planner-v3".to_string());
    sd.correlation_id = Some("corr-1".to_string());

    let mut v = serde_json::to_value(&sd).unwrap();
    let obj = v.as_object_mut().expect("SessionData is a JSON object");
    for key in [
        "tenant_id",
        "principal",
        "agent_template_id",
        "correlation_id",
    ] {
        obj.remove(key);
    }
    let s = serde_json::to_string(&v).unwrap();

    let loaded: SessionData = serde_json::from_str(&s).expect("pre-3.3.0 session must load");
    assert_eq!(loaded.tenant_id, None);
    assert_eq!(loaded.principal, None);
    assert_eq!(loaded.agent_template_id, None);
    assert_eq!(loaded.correlation_id, None);
}

#[test]
fn backward_compat_session_data_accepts_todos_alias() {
    // Older payloads named the task list `todos`; the field carries
    // `#[serde(alias = "todos")]` so those still load into `tasks`.
    let mut rng = Rng::new(456);
    let mut sd = gen_session_data(&mut rng);
    sd.tasks = vec![Task::new("t-old", "legacy task")];

    let mut v = serde_json::to_value(&sd).unwrap();
    let obj = v.as_object_mut().unwrap();
    let tasks = obj.remove("tasks").expect("tasks always serializes");
    obj.insert("todos".to_string(), tasks);
    let s = serde_json::to_string(&v).unwrap();

    let loaded: SessionData = serde_json::from_str(&s).expect("legacy `todos` payload must load");
    assert_eq!(loaded.tasks.len(), 1);
    assert_eq!(loaded.tasks[0].content, "legacy task");
}

// ─────────────────────────────────────────────────────────────────────
// 3. Forward compatibility — newer payloads (unknown fields) load.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn forward_compat_unknown_fields_ignored_on_persisted_types() {
    let mut rng = Rng::new(7);
    assert_unknown_fields_ignored(&gen_loop_checkpoint(&mut rng), "LoopCheckpoint");
    assert_unknown_fields_ignored(&gen_subagent_task(&mut rng), "SubagentTaskSnapshot");
    assert_unknown_fields_ignored(&gen_trace_event(&mut rng), "TraceEvent");
    assert_unknown_fields_ignored(&gen_verification_report(&mut rng), "VerificationReport");
    assert_unknown_fields_ignored(&gen_run_record(&mut rng), "RunRecord");
    assert_unknown_fields_ignored(&gen_session_data(&mut rng), "SessionData");
    assert_unknown_fields_ignored(&gen_agent_step_spec(&mut rng), "AgentStepSpec");
    assert_unknown_fields_ignored(&gen_step_outcome(&mut rng), "StepOutcome");
    assert_unknown_fields_ignored(&gen_workflow_checkpoint(&mut rng), "WorkflowCheckpoint");
}

// ─────────────────────────────────────────────────────────────────────
// 4. The explicit version contract: future schema versions are rejected.
// ─────────────────────────────────────────────────────────────────────

fn sample_checkpoint(run_id: &str, version: u32) -> LoopCheckpoint {
    LoopCheckpoint {
        schema_version: version,
        run_id: run_id.to_string(),
        session_id: "s".to_string(),
        turn: 1,
        messages: vec![Message::user("hi")],
        total_usage: TokenUsage::default(),
        tool_calls_count: 0,
        verification_reports: Vec::new(),
        checkpoint_ms: 1,
    }
}

#[test]
fn ensure_loadable_rejects_only_future_versions() {
    // current and pre-v1 are fine
    sample_checkpoint("a", LOOP_CHECKPOINT_SCHEMA_VERSION)
        .ensure_loadable()
        .expect("current version loadable");
    sample_checkpoint("b", 0)
        .ensure_loadable()
        .expect("pre-v1 loadable");
    // a future version is refused with a descriptive error
    let err = sample_checkpoint("c", LOOP_CHECKPOINT_SCHEMA_VERSION + 1)
        .ensure_loadable()
        .unwrap_err();
    assert!(
        err.to_string().contains("schema version"),
        "expected a schema-version error, got: {err}"
    );
}

#[tokio::test]
async fn store_load_rejects_future_schema_version() {
    let store = MemorySessionStore::new();

    // A future-version checkpoint can be written (the writer was newer)...
    let future = sample_checkpoint("run-future", LOOP_CHECKPOINT_SCHEMA_VERSION + 1);
    store
        .save_loop_checkpoint("run-future", &future)
        .await
        .expect("save");
    // ...but this build must refuse to load it rather than silently
    // misinterpret a format it doesn't understand.
    let err = store
        .load_loop_checkpoint("run-future")
        .await
        .expect_err("future-version load must error");
    assert!(err.to_string().contains("schema version"), "got: {err}");

    // A current-version checkpoint round-trips through the store cleanly.
    let ok = sample_checkpoint("run-ok", LOOP_CHECKPOINT_SCHEMA_VERSION);
    store
        .save_loop_checkpoint("run-ok", &ok)
        .await
        .expect("save");
    let loaded = store
        .load_loop_checkpoint("run-ok")
        .await
        .expect("load ok")
        .expect("present");
    assert_eq!(loaded.run_id, "run-ok");
}
