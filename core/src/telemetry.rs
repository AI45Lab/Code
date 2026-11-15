//! Telemetry Module (Core)
//!
//! Provides centralized observability primitives for the A3S Code agent:
//! - Span name and attribute key constants
//! - LLM cost tracking (model, tokens, cost per call)
//! - Tool execution metrics (duration, success/failure)
//! - Span helper functions using `tracing` (no OTel dependency)
//!
//! For OpenTelemetry initialization (OTLP exporter, subscriber setup),
//! see the `telemetry_init` module in the `a3s-code` server crate.
//!
//! ## Span Hierarchy
//!
//! ```text
//! a3s.agent.execute
//!   +-- a3s.agent.context_resolve
//!   +-- a3s.agent.turn (repeated)
//!   |   +-- a3s.llm.completion
//!   |   +-- a3s.tool.execute (repeated)
//!   +-- a3s.agent.turn_notify
//! ```

use std::time::Instant;

// ============================================================================
// Constants
// ============================================================================

/// Service name for telemetry
pub const SERVICE_NAME: &str = "a3s-code";

// Span name constants
pub const SPAN_AGENT_EXECUTE: &str = "a3s.agent.execute";
pub const SPAN_AGENT_TURN: &str = "a3s.agent.turn";
pub const SPAN_LLM_COMPLETION: &str = "a3s.llm.completion";
pub const SPAN_TOOL_EXECUTE: &str = "a3s.tool.execute";
pub const SPAN_CONTEXT_RESOLVE: &str = "a3s.agent.context_resolve";

// Attribute key constants
pub const ATTR_SESSION_ID: &str = "a3s.session.id";
pub const ATTR_TURN_NUMBER: &str = "a3s.agent.turn_number";
pub const ATTR_MAX_TURNS: &str = "a3s.agent.max_turns";
pub const ATTR_TOOL_CALLS_COUNT: &str = "a3s.agent.tool_calls_count";

pub const ATTR_LLM_MODEL: &str = "a3s.llm.model";
pub const ATTR_LLM_PROVIDER: &str = "a3s.llm.provider";
pub const ATTR_LLM_STREAMING: &str = "a3s.llm.streaming";
pub const ATTR_LLM_PROMPT_TOKENS: &str = "a3s.llm.prompt_tokens";
pub const ATTR_LLM_COMPLETION_TOKENS: &str = "a3s.llm.completion_tokens";
pub const ATTR_LLM_TOTAL_TOKENS: &str = "a3s.llm.total_tokens";
pub const ATTR_LLM_STOP_REASON: &str = "a3s.llm.stop_reason";

pub const ATTR_TOOL_NAME: &str = "a3s.tool.name";
pub const ATTR_TOOL_ID: &str = "a3s.tool.id";
pub const ATTR_TOOL_EXIT_CODE: &str = "a3s.tool.exit_code";
pub const ATTR_TOOL_SUCCESS: &str = "a3s.tool.success";
pub const ATTR_TOOL_DURATION_MS: &str = "a3s.tool.duration_ms";
pub const ATTR_TOOL_PERMISSION: &str = "a3s.tool.permission";

pub const ATTR_CONTEXT_PROVIDERS: &str = "a3s.context.providers";
pub const ATTR_CONTEXT_ITEMS: &str = "a3s.context.items";
pub const ATTR_CONTEXT_TOKENS: &str = "a3s.context.tokens";

// ============================================================================
// Span Helpers
// ============================================================================

/// Record LLM token usage on the current span
pub fn record_llm_usage(
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
    stop_reason: Option<&str>,
) {
    let span = tracing::Span::current();
    span.record(ATTR_LLM_PROMPT_TOKENS, prompt_tokens as i64);
    span.record(ATTR_LLM_COMPLETION_TOKENS, completion_tokens as i64);
    span.record(ATTR_LLM_TOTAL_TOKENS, total_tokens as i64);
    if let Some(reason) = stop_reason {
        span.record(ATTR_LLM_STOP_REASON, reason);
    }
}

/// Record tool execution result on the current span
pub fn record_tool_result(exit_code: i32, duration: std::time::Duration) {
    let span = tracing::Span::current();
    span.record(ATTR_TOOL_EXIT_CODE, exit_code as i64);
    span.record(ATTR_TOOL_SUCCESS, exit_code == 0);
    span.record(ATTR_TOOL_DURATION_MS, duration.as_millis() as i64);
}

/// A guard that measures elapsed time and records it when dropped
pub struct TimedSpan {
    start: Instant,
    span_field: &'static str,
}

impl TimedSpan {
    pub fn new(span_field: &'static str) -> Self {
        Self {
            start: Instant::now(),
            span_field,
        }
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}

impl Drop for TimedSpan {
    fn drop(&mut self) {
        let elapsed_ms = self.elapsed_ms();
        let span = tracing::Span::current();
        span.record(self.span_field, elapsed_ms as i64);
    }
}

// ============================================================================
// LLM Cost Tracking
// ============================================================================

/// Cost record for a single LLM call
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LlmCostRecord {
    /// Model identifier
    pub model: String,
    /// Provider name
    pub provider: String,
    /// Input tokens
    pub prompt_tokens: usize,
    /// Output tokens
    pub completion_tokens: usize,
    /// Total tokens
    pub total_tokens: usize,
    /// Estimated cost in USD (if pricing is configured)
    pub cost_usd: Option<f64>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Session ID
    pub session_id: Option<String>,
}

/// Pricing table for LLM models (cost per 1M tokens)
#[derive(Debug, Clone)]
pub struct ModelPricing {
    /// Cost per 1M input tokens in USD
    pub input_per_million: f64,
    /// Cost per 1M output tokens in USD
    pub output_per_million: f64,
}

impl ModelPricing {
    pub fn new(input_per_million: f64, output_per_million: f64) -> Self {
        Self {
            input_per_million,
            output_per_million,
        }
    }

    /// Calculate cost for given token counts
    pub fn calculate_cost(&self, prompt_tokens: usize, completion_tokens: usize) -> f64 {
        let input_cost = (prompt_tokens as f64 / 1_000_000.0) * self.input_per_million;
        let output_cost = (completion_tokens as f64 / 1_000_000.0) * self.output_per_million;
        input_cost + output_cost
    }
}

/// Registry of known model pricing
pub fn default_model_pricing() -> std::collections::HashMap<String, ModelPricing> {
    let mut pricing = std::collections::HashMap::new();

    // Anthropic Claude models
    pricing.insert(
        "claude-sonnet-4-20250514".to_string(),
        ModelPricing::new(3.0, 15.0),
    );
    pricing.insert(
        "claude-3-5-sonnet-20241022".to_string(),
        ModelPricing::new(3.0, 15.0),
    );
    pricing.insert(
        "claude-3-haiku-20240307".to_string(),
        ModelPricing::new(0.25, 1.25),
    );
    pricing.insert(
        "claude-3-opus-20240229".to_string(),
        ModelPricing::new(15.0, 75.0),
    );

    // OpenAI models
    pricing.insert("gpt-4o".to_string(), ModelPricing::new(2.5, 10.0));
    pricing.insert("gpt-4o-mini".to_string(), ModelPricing::new(0.15, 0.6));
    pricing.insert("gpt-4-turbo".to_string(), ModelPricing::new(10.0, 30.0));

    // Every other model id and alias from the `llm-prices` table. Entries
    // above take precedence and are never overwritten.
    #[cfg(feature = "llm-prices")]
    for row in llm_prices::all() {
        if let (Some(input), Some(output)) = (row.input, row.output) {
            for id in std::iter::once(row.model).chain(row.aliases.iter().copied()) {
                pricing
                    .entry(id.to_string())
                    .or_insert_with(|| ModelPricing::new(input, output));
            }
        }
    }

    pricing
}

/// Look up pricing for a model id.
///
/// Checks [`default_model_pricing`] first. With the `llm-prices` feature, ids
/// that are not in the registry verbatim (provider prefixes such as
/// `anthropic/`, Bedrock ids, dated or undated Anthropic ids) are resolved
/// through `llm_prices::lookup`.
pub fn lookup_model_pricing(model: &str) -> Option<ModelPricing> {
    if let Some(pricing) = default_model_pricing().remove(model) {
        return Some(pricing);
    }
    #[cfg(feature = "llm-prices")]
    if let Some(row) = llm_prices::lookup(model) {
        if let (Some(input), Some(output)) = (row.input, row.output) {
            return Some(ModelPricing::new(input, output));
        }
    }
    None
}

// ============================================================================
// Tool Metrics
// ============================================================================

/// Per-session tool execution metrics collector
#[derive(Debug, Clone, Default)]
pub struct ToolMetrics {
    /// Per-tool statistics
    stats: std::collections::HashMap<String, ToolStats>,
    /// Total calls across all tools
    total_calls: u64,
    /// Total duration across all tools
    total_duration_ms: u64,
}

/// Statistics for a single tool
#[derive(Debug, Clone)]
pub struct ToolStats {
    pub tool_name: String,
    pub total_calls: u64,
    pub success_count: u64,
    pub failure_count: u64,
    pub total_duration_ms: u64,
    pub min_duration_ms: u64,
    pub max_duration_ms: u64,
    pub avg_duration_ms: u64,
    pub last_called_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl ToolMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a tool execution
    pub fn record(&mut self, tool_name: &str, success: bool, duration_ms: u64) {
        self.total_calls += 1;
        self.total_duration_ms += duration_ms;

        let entry = self
            .stats
            .entry(tool_name.to_string())
            .or_insert_with(|| ToolStats {
                tool_name: tool_name.to_string(),
                total_calls: 0,
                success_count: 0,
                failure_count: 0,
                total_duration_ms: 0,
                min_duration_ms: u64::MAX,
                max_duration_ms: 0,
                avg_duration_ms: 0,
                last_called_at: None,
            });

        entry.total_calls += 1;
        if success {
            entry.success_count += 1;
        } else {
            entry.failure_count += 1;
        }
        entry.total_duration_ms += duration_ms;
        entry.min_duration_ms = entry.min_duration_ms.min(duration_ms);
        entry.max_duration_ms = entry.max_duration_ms.max(duration_ms);
        entry.avg_duration_ms = entry.total_duration_ms / entry.total_calls;
        entry.last_called_at = Some(chrono::Utc::now());
    }

    /// Get all tool stats
    pub fn stats(&self) -> Vec<ToolStats> {
        self.stats.values().cloned().collect()
    }

    /// Get stats for a specific tool
    pub fn stats_for(&self, tool_name: &str) -> Vec<ToolStats> {
        self.stats
            .get(tool_name)
            .map(|s| vec![s.clone()])
            .unwrap_or_default()
    }

    /// Total calls across all tools
    pub fn total_calls(&self) -> u64 {
        self.total_calls
    }

    /// Total duration across all tools
    pub fn total_duration_ms(&self) -> u64 {
        self.total_duration_ms
    }
}

// ============================================================================
// Cost Aggregation
// ============================================================================

/// Aggregated cost summary
#[derive(Debug, Clone)]
pub struct CostSummary {
    pub total_cost_usd: f64,
    pub total_prompt_tokens: usize,
    pub total_completion_tokens: usize,
    pub total_tokens: usize,
    pub call_count: usize,
    pub by_model: Vec<ModelCostBreakdown>,
    pub by_day: Vec<DayCostBreakdown>,
}

/// Cost breakdown by model
#[derive(Debug, Clone)]
pub struct ModelCostBreakdown {
    pub model: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub cost_usd: f64,
    pub call_count: usize,
}

/// Cost breakdown by day
#[derive(Debug, Clone)]
pub struct DayCostBreakdown {
    pub date: String,
    pub cost_usd: f64,
    pub call_count: usize,
    pub total_tokens: usize,
}

/// Aggregate cost records with optional filters
pub fn aggregate_cost_records(
    records: &[LlmCostRecord],
    model_filter: Option<&str>,
    start_date: Option<&str>,
    end_date: Option<&str>,
) -> CostSummary {
    let filtered: Vec<&LlmCostRecord> = records
        .iter()
        .filter(|r| {
            if let Some(model) = model_filter {
                if r.model != model {
                    return false;
                }
            }
            if let Some(start) = start_date {
                let date_str = r.timestamp.format("%Y-%m-%d").to_string();
                if date_str.as_str() < start {
                    return false;
                }
            }
            if let Some(end) = end_date {
                let date_str = r.timestamp.format("%Y-%m-%d").to_string();
                if date_str.as_str() > end {
                    return false;
                }
            }
            true
        })
        .collect();

    let mut by_model_map: std::collections::HashMap<String, ModelCostBreakdown> =
        std::collections::HashMap::new();
    let mut by_day_map: std::collections::HashMap<String, DayCostBreakdown> =
        std::collections::HashMap::new();

    let mut total_cost_usd = 0.0;
    let mut total_prompt_tokens = 0usize;
    let mut total_completion_tokens = 0usize;
    let mut total_tokens = 0usize;

    for record in &filtered {
        let cost = record.cost_usd.unwrap_or(0.0);
        total_cost_usd += cost;
        total_prompt_tokens += record.prompt_tokens;
        total_completion_tokens += record.completion_tokens;
        total_tokens += record.total_tokens;

        // By model
        let model_entry =
            by_model_map
                .entry(record.model.clone())
                .or_insert_with(|| ModelCostBreakdown {
                    model: record.model.clone(),
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    total_tokens: 0,
                    cost_usd: 0.0,
                    call_count: 0,
                });
        model_entry.prompt_tokens += record.prompt_tokens;
        model_entry.completion_tokens += record.completion_tokens;
        model_entry.total_tokens += record.total_tokens;
        model_entry.cost_usd += cost;
        model_entry.call_count += 1;

        // By day
        let date_str = record.timestamp.format("%Y-%m-%d").to_string();
        let day_entry = by_day_map
            .entry(date_str.clone())
            .or_insert_with(|| DayCostBreakdown {
                date: date_str,
                cost_usd: 0.0,
                call_count: 0,
                total_tokens: 0,
            });
        day_entry.cost_usd += cost;
        day_entry.call_count += 1;
        day_entry.total_tokens += record.total_tokens;
    }

    let mut by_model: Vec<ModelCostBreakdown> = by_model_map.into_values().collect();
    by_model.sort_by(|a, b| {
        b.cost_usd
            .partial_cmp(&a.cost_usd)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut by_day: Vec<DayCostBreakdown> = by_day_map.into_values().collect();
    by_day.sort_by(|a, b| a.date.cmp(&b.date));

    CostSummary {
        total_cost_usd,
        total_prompt_tokens,
        total_completion_tokens,
        total_tokens,
        call_count: filtered.len(),
        by_model,
        by_day,
    }
}

/// Record LLM metrics via tracing events.
///
/// In the server crate (`a3s-code`), this is overridden by `telemetry_init::record_llm_metrics`
/// which writes to OpenTelemetry counters. In the core library, we emit a tracing event
/// so the data is available to any subscriber (including OTel if wired up externally).
pub fn record_llm_metrics(
    model: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
    cost_usd: f64,
    duration_secs: f64,
) {
    tracing::info!(
        model = model,
        prompt_tokens = prompt_tokens,
        completion_tokens = completion_tokens,
        total_tokens = prompt_tokens + completion_tokens,
        cost_usd = cost_usd,
        duration_secs = duration_secs,
        "llm.metrics"
    );
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_pricing_calculation() {
        let pricing = ModelPricing::new(3.0, 15.0); // Claude Sonnet pricing

        // 1000 input tokens + 500 output tokens
        let cost = pricing.calculate_cost(1000, 500);
        let expected = (1000.0 / 1_000_000.0) * 3.0 + (500.0 / 1_000_000.0) * 15.0;
        assert!((cost - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn test_model_pricing_zero_tokens() {
        let pricing = ModelPricing::new(3.0, 15.0);
        let cost = pricing.calculate_cost(0, 0);
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn test_model_pricing_large_tokens() {
        let pricing = ModelPricing::new(3.0, 15.0);
        // 1M input + 1M output
        let cost = pricing.calculate_cost(1_000_000, 1_000_000);
        assert!((cost - 18.0).abs() < f64::EPSILON); // $3 + $15
    }

    #[test]
    fn test_default_model_pricing_has_known_models() {
        let pricing = default_model_pricing();
        assert!(pricing.contains_key("claude-sonnet-4-20250514"));
        assert!(pricing.contains_key("claude-3-5-sonnet-20241022"));
        assert!(pricing.contains_key("claude-3-haiku-20240307"));
        assert!(pricing.contains_key("gpt-4o"));
        assert!(pricing.contains_key("gpt-4o-mini"));
    }

    #[test]
    fn test_llm_cost_record_serialize() {
        let record = LlmCostRecord {
            model: "claude-sonnet-4-20250514".to_string(),
            provider: "anthropic".to_string(),
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
            cost_usd: Some(0.0105),
            timestamp: chrono::Utc::now(),
            session_id: Some("sess-123".to_string()),
        };

        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("claude-sonnet-4-20250514"));
        assert!(json.contains("anthropic"));
        assert!(json.contains("1000"));

        // Deserialize back
        let deserialized: LlmCostRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.model, "claude-sonnet-4-20250514");
        assert_eq!(deserialized.prompt_tokens, 1000);
    }

    #[test]
    fn test_timed_span_elapsed() {
        let timer = TimedSpan::new(ATTR_TOOL_DURATION_MS);
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(timer.elapsed_ms() >= 10);
    }

    #[test]
    fn test_span_name_constants() {
        // Verify span names follow a3s.* convention
        assert!(SPAN_AGENT_EXECUTE.starts_with("a3s."));
        assert!(SPAN_AGENT_TURN.starts_with("a3s."));
        assert!(SPAN_LLM_COMPLETION.starts_with("a3s."));
        assert!(SPAN_TOOL_EXECUTE.starts_with("a3s."));
        assert!(SPAN_CONTEXT_RESOLVE.starts_with("a3s."));
    }

    #[test]
    fn test_attribute_key_constants() {
        // Verify attribute keys follow a3s.* convention
        assert!(ATTR_SESSION_ID.starts_with("a3s."));
        assert!(ATTR_LLM_MODEL.starts_with("a3s."));
        assert!(ATTR_TOOL_NAME.starts_with("a3s."));
        assert!(ATTR_CONTEXT_PROVIDERS.starts_with("a3s."));
    }

    #[test]
    fn test_record_llm_usage_does_not_panic() {
        // record_llm_usage should not panic even without an active span
        record_llm_usage(100, 50, 150, Some("end_turn"));
        record_llm_usage(0, 0, 0, None);
        record_llm_usage(1_000_000, 500_000, 1_500_000, Some("max_tokens"));
    }

    #[test]
    fn test_record_tool_result_does_not_panic() {
        // record_tool_result should not panic even without an active span
        record_tool_result(0, std::time::Duration::from_millis(100));
        record_tool_result(1, std::time::Duration::from_secs(0));
        record_tool_result(-1, std::time::Duration::from_secs(30));
    }

    #[test]
    fn test_timed_span_measures_duration() {
        let timer = TimedSpan::new(ATTR_TOOL_DURATION_MS);
        // Immediately check -- should be very small
        assert!(timer.elapsed_ms() < 1000);
    }

    #[test]
    fn test_model_pricing_registry_completeness() {
        let pricing = default_model_pricing();
        // Verify all expected providers are covered
        let anthropic_models: Vec<&str> = pricing
            .keys()
            .filter(|k| k.starts_with("claude"))
            .map(|k| k.as_str())
            .collect();
        assert!(
            anthropic_models.len() >= 3,
            "Expected at least 3 Anthropic models, got {}",
            anthropic_models.len()
        );

        let openai_models: Vec<&str> = pricing
            .keys()
            .filter(|k| k.starts_with("gpt"))
            .map(|k| k.as_str())
            .collect();
        assert!(
            openai_models.len() >= 2,
            "Expected at least 2 OpenAI models, got {}",
            openai_models.len()
        );
    }

    #[test]
    fn test_model_pricing_cost_ordering() {
        let pricing = default_model_pricing();
        // Haiku should be cheaper than Sonnet
        let haiku = pricing.get("claude-3-haiku-20240307").unwrap();
        let sonnet = pricing.get("claude-sonnet-4-20250514").unwrap();
        assert!(
            haiku.input_per_million < sonnet.input_per_million,
            "Haiku should be cheaper than Sonnet"
        );

        // GPT-4o-mini should be cheaper than GPT-4o
        let mini = pricing.get("gpt-4o-mini").unwrap();
        let full = pricing.get("gpt-4o").unwrap();
        assert!(
            mini.input_per_million < full.input_per_million,
            "GPT-4o-mini should be cheaper than GPT-4o"
        );
    }

    #[test]
    fn test_llm_cost_record_fields() {
        let record = LlmCostRecord {
            model: "gpt-4o".to_string(),
            provider: "openai".to_string(),
            prompt_tokens: 500,
            completion_tokens: 200,
            total_tokens: 700,
            cost_usd: None,
            timestamp: chrono::Utc::now(),
            session_id: None,
        };
        assert_eq!(
            record.total_tokens,
            record.prompt_tokens + record.completion_tokens
        );
        assert!(record.cost_usd.is_none());
        assert!(record.session_id.is_none());
    }

    #[test]
    fn test_attribute_keys_are_unique() {
        // All attribute keys should be distinct
        let keys = vec![
            ATTR_SESSION_ID,
            ATTR_TURN_NUMBER,
            ATTR_MAX_TURNS,
            ATTR_TOOL_CALLS_COUNT,
            ATTR_LLM_MODEL,
            ATTR_LLM_PROVIDER,
            ATTR_LLM_STREAMING,
            ATTR_LLM_PROMPT_TOKENS,
            ATTR_LLM_COMPLETION_TOKENS,
            ATTR_LLM_TOTAL_TOKENS,
            ATTR_LLM_STOP_REASON,
            ATTR_TOOL_NAME,
            ATTR_TOOL_ID,
            ATTR_TOOL_EXIT_CODE,
            ATTR_TOOL_SUCCESS,
            ATTR_TOOL_DURATION_MS,
            ATTR_TOOL_PERMISSION,
            ATTR_CONTEXT_PROVIDERS,
            ATTR_CONTEXT_ITEMS,
            ATTR_CONTEXT_TOKENS,
        ];
        let unique: std::collections::HashSet<&str> = keys.iter().copied().collect();
        assert_eq!(keys.len(), unique.len(), "Attribute keys must be unique");
    }

    #[test]
    fn test_model_pricing_new() {
        let pricing = ModelPricing::new(3.0, 15.0);
        assert_eq!(pricing.input_per_million, 3.0);
        assert_eq!(pricing.output_per_million, 15.0);
    }

    #[test]
    fn test_model_pricing_calculate_cost_zero() {
        let pricing = ModelPricing::new(3.0, 15.0);
        let cost = pricing.calculate_cost(0, 0);
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn test_model_pricing_calculate_cost_large() {
        let pricing = ModelPricing::new(3.0, 15.0);
        let cost = pricing.calculate_cost(1_000_000, 1_000_000);
        assert!((cost - 18.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_model_pricing_calculate_cost_fractional() {
        let pricing = ModelPricing::new(3.0, 15.0);
        let cost = pricing.calculate_cost(500, 250);
        let expected = (500.0 / 1_000_000.0) * 3.0 + (250.0 / 1_000_000.0) * 15.0;
        assert!((cost - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn test_model_pricing_clone() {
        let pricing = ModelPricing::new(3.0, 15.0);
        let cloned = pricing.clone();
        assert_eq!(cloned.input_per_million, 3.0);
        assert_eq!(cloned.output_per_million, 15.0);
    }

    #[test]
    fn test_model_pricing_debug() {
        let pricing = ModelPricing::new(3.0, 15.0);
        let debug_str = format!("{:?}", pricing);
        assert!(debug_str.contains("ModelPricing"));
        assert!(debug_str.contains("3.0"));
        assert!(debug_str.contains("15.0"));
    }

    #[test]
    fn test_lookup_model_pricing_registry_entry() {
        let pricing = lookup_model_pricing("claude-sonnet-4-20250514").unwrap();
        assert_eq!(pricing.input_per_million, 3.0);
        assert_eq!(pricing.output_per_million, 15.0);
        assert!(lookup_model_pricing("not-a-real-model").is_none());
    }

    #[cfg(feature = "llm-prices")]
    #[test]
    fn test_llm_prices_extends_registry() {
        let pricing = default_model_pricing();
        // Hand-written entries keep their values.
        let turbo = pricing.get("gpt-4-turbo").unwrap();
        assert_eq!(turbo.input_per_million, 10.0);
        assert_eq!(turbo.output_per_million, 30.0);
        // Newer models come from the llm-prices table.
        assert!(pricing.contains_key("claude-sonnet-4-6"));
        assert!(pricing.contains_key("gpt-5"));
        assert!(pricing.contains_key("gemini-2.5-pro"));
        // Provider prefixes and dated ids resolve through lookup.
        assert!(lookup_model_pricing("anthropic/claude-sonnet-4-6").is_some());
        assert!(lookup_model_pricing("claude-haiku-4-5-20251001").is_some());
    }

    #[test]
    fn test_default_model_pricing_all_positive() {
        let pricing = default_model_pricing();
        for (model, price) in pricing.iter() {
            assert!(
                price.input_per_million > 0.0,
                "Model {} has non-positive input cost",
                model
            );
            assert!(
                price.output_per_million > 0.0,
                "Model {} has non-positive output cost",
                model
            );
        }
    }

    #[test]
    fn test_default_model_pricing_output_greater_than_input() {
        let pricing = default_model_pricing();
        for (model, price) in pricing.iter() {
            assert!(
                price.output_per_million > price.input_per_million,
                "Model {} output cost should be greater than input cost",
                model
            );
        }
    }

    #[test]
    fn test_default_model_pricing_claude_sonnet_4() {
        let pricing = default_model_pricing();
        let sonnet = pricing.get("claude-sonnet-4-20250514").unwrap();
        assert_eq!(sonnet.input_per_million, 3.0);
        assert_eq!(sonnet.output_per_million, 15.0);
    }

    #[test]
    fn test_default_model_pricing_claude_haiku() {
        let pricing = default_model_pricing();
        let haiku = pricing.get("claude-3-haiku-20240307").unwrap();
        assert_eq!(haiku.input_per_million, 0.25);
        assert_eq!(haiku.output_per_million, 1.25);
    }

    #[test]
    fn test_default_model_pricing_gpt4o() {
        let pricing = default_model_pricing();
        let gpt4o = pricing.get("gpt-4o").unwrap();
        assert_eq!(gpt4o.input_per_million, 2.5);
        assert_eq!(gpt4o.output_per_million, 10.0);
    }

    #[test]
    fn test_llm_cost_record_with_cost() {
        let record = LlmCostRecord {
            model: "claude-sonnet-4-20250514".to_string(),
            provider: "anthropic".to_string(),
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
            cost_usd: Some(0.0105),
            timestamp: chrono::Utc::now(),
            session_id: Some("sess-123".to_string()),
        };
        assert_eq!(record.cost_usd, Some(0.0105));
    }

    #[test]
    fn test_llm_cost_record_without_cost() {
        let record = LlmCostRecord {
            model: "unknown-model".to_string(),
            provider: "unknown".to_string(),
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            cost_usd: None,
            timestamp: chrono::Utc::now(),
            session_id: None,
        };
        assert!(record.cost_usd.is_none());
    }

    #[test]
    fn test_llm_cost_record_with_session() {
        let record = LlmCostRecord {
            model: "gpt-4o".to_string(),
            provider: "openai".to_string(),
            prompt_tokens: 500,
            completion_tokens: 200,
            total_tokens: 700,
            cost_usd: Some(0.003),
            timestamp: chrono::Utc::now(),
            session_id: Some("session-abc".to_string()),
        };
        assert_eq!(record.session_id, Some("session-abc".to_string()));
    }

    #[test]
    fn test_llm_cost_record_without_session() {
        let record = LlmCostRecord {
            model: "gpt-4o-mini".to_string(),
            provider: "openai".to_string(),
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            cost_usd: Some(0.00006),
            timestamp: chrono::Utc::now(),
            session_id: None,
        };
        assert!(record.session_id.is_none());
    }

    #[test]
    fn test_llm_cost_record_serialization() {
        let record = LlmCostRecord {
            model: "claude-sonnet-4-20250514".to_string(),
            provider: "anthropic".to_string(),
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
            cost_usd: Some(0.0105),
            timestamp: chrono::Utc::now(),
            session_id: Some("sess-123".to_string()),
        };
        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("claude-sonnet-4-20250514"));
        assert!(json.contains("anthropic"));
        assert!(json.contains("1000"));
        assert!(json.contains("500"));
    }

    #[test]
    fn test_llm_cost_record_zero_tokens() {
        let record = LlmCostRecord {
            model: "test-model".to_string(),
            provider: "test".to_string(),
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            cost_usd: Some(0.0),
            timestamp: chrono::Utc::now(),
            session_id: None,
        };
        assert_eq!(record.prompt_tokens, 0);
        assert_eq!(record.completion_tokens, 0);
        assert_eq!(record.total_tokens, 0);
    }

    #[test]
    fn test_llm_cost_record_clone() {
        let record = LlmCostRecord {
            model: "gpt-4o".to_string(),
            provider: "openai".to_string(),
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            cost_usd: Some(0.001),
            timestamp: chrono::Utc::now(),
            session_id: Some("sess-xyz".to_string()),
        };
        let cloned = record.clone();
        assert_eq!(cloned.model, "gpt-4o");
        assert_eq!(cloned.provider, "openai");
        assert_eq!(cloned.prompt_tokens, 100);
    }

    #[test]
    fn test_timed_span_new() {
        let timer = TimedSpan::new(ATTR_TOOL_DURATION_MS);
        assert!(timer.elapsed_ms() < 100);
    }

    #[test]
    fn test_timed_span_elapsed_sleep() {
        let timer = TimedSpan::new(ATTR_TOOL_DURATION_MS);
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(timer.elapsed_ms() >= 10);
    }

    #[test]
    fn test_service_name_constant() {
        assert_eq!(SERVICE_NAME, "a3s-code");
    }

    #[test]
    fn test_span_constants() {
        assert_eq!(SPAN_AGENT_EXECUTE, "a3s.agent.execute");
        assert_eq!(SPAN_AGENT_TURN, "a3s.agent.turn");
        assert_eq!(SPAN_LLM_COMPLETION, "a3s.llm.completion");
        assert_eq!(SPAN_TOOL_EXECUTE, "a3s.tool.execute");
        assert_eq!(SPAN_CONTEXT_RESOLVE, "a3s.agent.context_resolve");
    }

    #[test]
    fn test_attribute_constants_session() {
        assert_eq!(ATTR_SESSION_ID, "a3s.session.id");
        assert_eq!(ATTR_TURN_NUMBER, "a3s.agent.turn_number");
        assert_eq!(ATTR_MAX_TURNS, "a3s.agent.max_turns");
        assert_eq!(ATTR_TOOL_CALLS_COUNT, "a3s.agent.tool_calls_count");
    }

    #[test]
    fn test_attribute_constants_llm() {
        assert_eq!(ATTR_LLM_MODEL, "a3s.llm.model");
        assert_eq!(ATTR_LLM_PROVIDER, "a3s.llm.provider");
        assert_eq!(ATTR_LLM_STREAMING, "a3s.llm.streaming");
        assert_eq!(ATTR_LLM_PROMPT_TOKENS, "a3s.llm.prompt_tokens");
        assert_eq!(ATTR_LLM_COMPLETION_TOKENS, "a3s.llm.completion_tokens");
        assert_eq!(ATTR_LLM_TOTAL_TOKENS, "a3s.llm.total_tokens");
        assert_eq!(ATTR_LLM_STOP_REASON, "a3s.llm.stop_reason");
    }

    #[test]
    fn test_attribute_constants_tool() {
        assert_eq!(ATTR_TOOL_NAME, "a3s.tool.name");
        assert_eq!(ATTR_TOOL_ID, "a3s.tool.id");
        assert_eq!(ATTR_TOOL_EXIT_CODE, "a3s.tool.exit_code");
        assert_eq!(ATTR_TOOL_SUCCESS, "a3s.tool.success");
        assert_eq!(ATTR_TOOL_DURATION_MS, "a3s.tool.duration_ms");
        assert_eq!(ATTR_TOOL_PERMISSION, "a3s.tool.permission");
    }

    #[test]
    fn test_attribute_constants_context() {
        assert_eq!(ATTR_CONTEXT_PROVIDERS, "a3s.context.providers");
        assert_eq!(ATTR_CONTEXT_ITEMS, "a3s.context.items");
        assert_eq!(ATTR_CONTEXT_TOKENS, "a3s.context.tokens");
    }

    #[test]
    fn test_record_llm_usage_basic() {
        record_llm_usage(100, 50, 150, Some("end_turn"));
    }

    #[test]
    fn test_record_llm_usage_no_stop_reason() {
        record_llm_usage(100, 50, 150, None);
    }

    #[test]
    fn test_record_tool_result_success() {
        record_tool_result(0, std::time::Duration::from_millis(100));
    }

    #[test]
    fn test_record_tool_result_failure() {
        record_tool_result(1, std::time::Duration::from_secs(5));
    }
}
