//! Context compaction logic
//!
//! Summarizes old conversation messages to reduce context size while
//! preserving key information. Supports both message-count and token-based
//! triggers, plus tool output pruning for large results.
//!
//! ## Auto-Compact Flow
//!
//! After each LLM turn, if `auto_compact` is enabled:
//! 1. Check if `context_usage.percent >= auto_compact_threshold`
//! 2. If yes, first prune large tool outputs (reclaim space cheaply)
//! 3. If still over threshold, summarize old messages via LLM

use crate::llm::{ContentBlock, LlmClient, Message, ToolResultContentField};
use anyhow::{Context, Result};
use std::sync::Arc;

/// Number of recent messages to keep intact during compaction
pub(crate) const KEEP_RECENT_MESSAGES: usize = 20;

/// Minimum message count before compaction is triggered
pub(crate) const MIN_MESSAGES_FOR_COMPACTION: usize = 30;

/// Number of initial messages to keep (usually system context)
pub(crate) const KEEP_INITIAL_MESSAGES: usize = 2;

/// Protect the first N tokens of tool outputs from pruning
const TOOL_OUTPUT_PROTECT_TOKENS: usize = 40_000;

/// Minimum token savings required to justify pruning a tool output
const MIN_PRUNE_SAVINGS_TOKENS: usize = 20_000;

/// Replacement text for pruned tool outputs
const PRUNED_MARKER: &str = "[output pruned — re-read file or re-run command if needed]";

/// Compact messages by summarizing old conversation turns.
///
/// Returns `Some(new_messages)` if compaction was performed, or `None` if
/// the message count is below the compaction threshold.
pub(crate) async fn compact_messages(
    session_id: &str,
    messages: &[Message],
    llm_client: &Arc<dyn LlmClient>,
) -> Result<Option<Vec<Message>>> {
    if messages.len() <= MIN_MESSAGES_FOR_COMPACTION {
        tracing::debug!(
            "Session {} has {} messages, no compaction needed (threshold: {})",
            session_id,
            messages.len(),
            MIN_MESSAGES_FOR_COMPACTION
        );
        return Ok(None);
    }

    tracing::info!(
        "Compacting session {} with {} messages",
        session_id,
        messages.len()
    );

    let total = messages.len();
    let summarize_start = KEEP_INITIAL_MESSAGES;
    let summarize_end = total.saturating_sub(KEEP_RECENT_MESSAGES);

    // If there's nothing to summarize, keep initial + recent messages
    if summarize_end <= summarize_start {
        tracing::debug!(
            "Not enough messages to summarize, keeping initial {} + last {}",
            KEEP_INITIAL_MESSAGES,
            KEEP_RECENT_MESSAGES
        );
        let mut result = messages[..KEEP_INITIAL_MESSAGES.min(total)].to_vec();
        let recent_start = total.saturating_sub(KEEP_RECENT_MESSAGES).max(KEEP_INITIAL_MESSAGES);
        result.extend_from_slice(&messages[recent_start..]);
        return Ok(Some(result));
    }

    let initial_messages = messages[..summarize_start].to_vec();
    let messages_to_summarize = &messages[summarize_start..summarize_end];
    let recent_messages = messages[summarize_end..].to_vec();

    tracing::debug!(
        "Compaction split: {} initial, {} to summarize, {} recent",
        initial_messages.len(),
        messages_to_summarize.len(),
        recent_messages.len()
    );

    // Build summarization prompt
    let conversation_text = messages_to_summarize
        .iter()
        .map(|msg| {
            let role = &msg.role;
            let text = msg.text();
            format!("{}: {}", role, text)
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let summarization_prompt = crate::prompts::render(
        crate::prompts::CONTEXT_COMPACT,
        &[("conversation", &conversation_text)],
    );

    // Call LLM to generate summary
    let summary_message = Message::user(&summarization_prompt);
    let response = llm_client
        .complete(&[summary_message], None, &[])
        .await
        .context("Failed to generate conversation summary")?;

    let summary_text = response.text();
    tracing::debug!("Generated summary: {} chars", summary_text.len());

    let summary_message = Message {
        role: "user".to_string(),
        content: vec![ContentBlock::Text {
            text: format!("{}{}", crate::prompts::CONTEXT_SUMMARY_PREFIX, summary_text),
        }],
        reasoning_content: None,
    };

    let mut new_messages = initial_messages;
    new_messages.push(summary_message);
    new_messages.extend(recent_messages);

    tracing::info!(
        "Compaction complete: {} messages -> {} messages",
        messages.len(),
        new_messages.len()
    );

    Ok(Some(new_messages))
}

/// Check if auto-compaction should be triggered based on token usage.
///
/// Returns `true` if `used_tokens / max_tokens >= threshold`.
pub(crate) fn should_auto_compact(used_tokens: usize, max_tokens: usize, threshold: f32) -> bool {
    if max_tokens == 0 {
        return false;
    }
    let usage_percent = used_tokens as f32 / max_tokens as f32;
    usage_percent >= threshold
}

/// Prune large tool outputs from messages to reclaim context space.
///
/// Iterates backward from recent messages, protecting the first
/// `TOOL_OUTPUT_PROTECT_TOKENS` worth of tool output tokens. Only prunes
/// outputs that would save at least `MIN_PRUNE_SAVINGS_TOKENS`.
///
/// Returns `Some(pruned_messages)` if any outputs were pruned, or `None`
/// if no pruning was needed.
pub(crate) fn prune_tool_outputs(messages: &[Message]) -> Option<Vec<Message>> {
    // First pass: estimate total tool output tokens (backward)
    let mut tool_outputs: Vec<(usize, usize, usize)> = Vec::new(); // (msg_idx, block_idx, token_count)

    for (msg_idx, msg) in messages.iter().enumerate() {
        for (block_idx, block) in msg.content.iter().enumerate() {
            if let ContentBlock::ToolResult { content, .. } = block {
                let text = content.as_text();
                let token_count = estimate_tokens(&text);
                if token_count > 0 {
                    tool_outputs.push((msg_idx, block_idx, token_count));
                }
            }
        }
    }

    if tool_outputs.is_empty() {
        return None;
    }

    // Calculate total tool output tokens
    let total_tool_tokens: usize = tool_outputs.iter().map(|(_, _, t)| *t).sum();

    // If total is small, no pruning needed
    if total_tool_tokens <= TOOL_OUTPUT_PROTECT_TOKENS {
        return None;
    }

    // Iterate from oldest to newest, protecting the most recent outputs
    // We prune old outputs first, keeping recent ones intact
    let mut protected_tokens = 0usize;
    let mut to_prune: Vec<(usize, usize)> = Vec::new();
    let mut savings = 0usize;

    // Walk backward (newest first) to protect recent outputs
    for &(msg_idx, block_idx, token_count) in tool_outputs.iter().rev() {
        if protected_tokens < TOOL_OUTPUT_PROTECT_TOKENS {
            protected_tokens += token_count;
        } else {
            to_prune.push((msg_idx, block_idx));
            savings += token_count;
        }
    }

    // Only prune if savings are significant
    if savings < MIN_PRUNE_SAVINGS_TOKENS {
        return None;
    }

    // Apply pruning
    let mut pruned = messages.to_vec();
    for (msg_idx, block_idx) in &to_prune {
        if let Some(msg) = pruned.get_mut(*msg_idx) {
            if let Some(ContentBlock::ToolResult { content, .. }) = msg.content.get_mut(*block_idx)
            {
                *content = ToolResultContentField::Text(PRUNED_MARKER.to_string());
            }
        }
    }

    tracing::info!(
        pruned_outputs = to_prune.len(),
        tokens_saved = savings,
        "Tool output pruning complete"
    );

    Some(pruned)
}

/// Rough token estimation (~4 chars per token for English/code)
fn estimate_tokens(text: &str) -> usize {
    text.len() / 4
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- should_auto_compact tests --

    #[test]
    fn test_should_auto_compact_below_threshold() {
        assert!(!should_auto_compact(50_000, 200_000, 0.80));
    }

    #[test]
    fn test_should_auto_compact_at_threshold() {
        assert!(should_auto_compact(160_000, 200_000, 0.80));
    }

    #[test]
    fn test_should_auto_compact_above_threshold() {
        assert!(should_auto_compact(190_000, 200_000, 0.80));
    }

    #[test]
    fn test_should_auto_compact_zero_max() {
        assert!(!should_auto_compact(100, 0, 0.80));
    }

    #[test]
    fn test_should_auto_compact_exact_boundary() {
        // 80% of 100_000 = 80_000
        assert!(should_auto_compact(80_000, 100_000, 0.80));
        assert!(!should_auto_compact(79_999, 100_000, 0.80));
    }

    #[test]
    fn test_should_auto_compact_custom_threshold() {
        assert!(should_auto_compact(95_000, 100_000, 0.95));
        assert!(!should_auto_compact(94_000, 100_000, 0.95));
    }

    // -- estimate_tokens tests --

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn test_estimate_tokens_short() {
        assert_eq!(estimate_tokens("hello world!"), 3); // 12 chars / 4
    }

    #[test]
    fn test_estimate_tokens_code() {
        let code = "fn main() {\n    println!(\"Hello, world!\");\n}";
        let tokens = estimate_tokens(code);
        assert!(tokens > 5 && tokens < 20);
    }

    // -- prune_tool_outputs tests --

    fn make_tool_result_msg(tool_id: &str, content: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: vec![ContentBlock::ToolResult {
                tool_use_id: tool_id.to_string(),
                content: ToolResultContentField::Text(content.to_string()),
                is_error: None,
            }],
            reasoning_content: None,
        }
    }

    fn make_text_msg(role: &str, text: &str) -> Message {
        Message {
            role: role.to_string(),
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
            reasoning_content: None,
        }
    }

    #[test]
    fn test_prune_no_tool_outputs() {
        let messages = vec![
            make_text_msg("user", "hello"),
            make_text_msg("assistant", "hi there"),
        ];
        assert!(prune_tool_outputs(&messages).is_none());
    }

    #[test]
    fn test_prune_small_tool_outputs() {
        let messages = vec![
            make_tool_result_msg("t1", "small output"),
            make_text_msg("assistant", "ok"),
        ];
        // Small output, no pruning needed
        assert!(prune_tool_outputs(&messages).is_none());
    }

    #[test]
    fn test_prune_large_tool_outputs() {
        // Create messages with large tool outputs that exceed protection threshold
        let large_content = "x".repeat(200_000); // ~50k tokens
        let large_content2 = "y".repeat(200_000); // ~50k tokens
        let small_recent = "z".repeat(40_000); // ~10k tokens (recent, protected)

        let messages = vec![
            make_tool_result_msg("t1", &large_content), // old, should be pruned
            make_text_msg("assistant", "processed t1"),
            make_tool_result_msg("t2", &large_content2), // old, should be pruned
            make_text_msg("assistant", "processed t2"),
            make_tool_result_msg("t3", &small_recent), // recent, protected
            make_text_msg("assistant", "done"),
        ];

        let result = prune_tool_outputs(&messages);
        assert!(result.is_some());

        let pruned = result.unwrap();
        // t1 and/or t2 should be pruned (oldest first)
        let t1_content = match &pruned[0].content[0] {
            ContentBlock::ToolResult { content, .. } => content.as_text(),
            _ => panic!("Expected ToolResult"),
        };
        assert_eq!(t1_content, PRUNED_MARKER);
    }

    #[test]
    fn test_prune_preserves_recent_outputs() {
        // Recent output alone fills the protection budget (~50k tokens)
        let large_old = "a".repeat(400_000); // ~100k tokens
        let recent = "b".repeat(200_000); // ~50k tokens (fills protection budget)

        let messages = vec![
            make_tool_result_msg("old", &large_old),
            make_text_msg("assistant", "ok"),
            make_tool_result_msg("recent", &recent),
            make_text_msg("assistant", "done"),
        ];

        let result = prune_tool_outputs(&messages);
        assert!(result.is_some());

        let pruned = result.unwrap();
        // Old should be pruned
        let old_content = match &pruned[0].content[0] {
            ContentBlock::ToolResult { content, .. } => content.as_text(),
            _ => panic!("Expected ToolResult"),
        };
        assert_eq!(old_content, PRUNED_MARKER);

        // Recent should be preserved
        let recent_content = match &pruned[2].content[0] {
            ContentBlock::ToolResult { content, .. } => content.as_text(),
            _ => panic!("Expected ToolResult"),
        };
        assert_ne!(recent_content, PRUNED_MARKER);
    }

    #[test]
    fn test_prune_marker_text() {
        assert!(PRUNED_MARKER.contains("pruned"));
    }

    // -- compact_messages tests (existing behavior preserved) --

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_constants() {
        assert!(KEEP_RECENT_MESSAGES > 0);
        assert!(MIN_MESSAGES_FOR_COMPACTION > KEEP_RECENT_MESSAGES);
        assert!(KEEP_INITIAL_MESSAGES > 0);
        assert!(TOOL_OUTPUT_PROTECT_TOKENS > 0);
        assert!(MIN_PRUNE_SAVINGS_TOKENS > 0);
    }
}
