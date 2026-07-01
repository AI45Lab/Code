//! Retry logic for LLM API calls
//!
//! Provides exponential backoff with jitter for transient HTTP errors.
//! Supports `Retry-After` header parsing for rate-limited responses.
//!
//! ## Retryable Status Codes
//!
//! - 429: Too Many Requests (rate limited)
//! - 500: Internal Server Error
//! - 502: Bad Gateway
//! - 503: Service Unavailable
//! - 529: Overloaded (Anthropic-specific)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use a3s_code::retry::RetryConfig;
//!
//! let config = RetryConfig::default(); // 3 retries, 1s base, 30s max
//! ```

use std::time::Duration;

use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

/// Configuration for API retry behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (0 = no retries)
    pub max_retries: u32,
    /// Base delay in milliseconds for exponential backoff
    pub base_delay_ms: u64,
    /// Maximum delay in milliseconds (cap for exponential growth)
    pub max_delay_ms: u64,
    /// HTTP status codes that trigger a retry
    pub retryable_status_codes: Vec<u16>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 1000,
            max_delay_ms: 30_000,
            retryable_status_codes: vec![429, 500, 502, 503, 529],
        }
    }
}

impl RetryConfig {
    /// Create a retry config with no retries (disabled)
    pub fn disabled() -> Self {
        Self {
            max_retries: 0,
            ..Default::default()
        }
    }

    /// Check if a given HTTP status code is retryable
    pub fn is_retryable_status(&self, status: StatusCode) -> bool {
        self.retryable_status_codes.contains(&status.as_u16())
    }

    /// Calculate the delay for a given attempt number (0-indexed)
    ///
    /// Uses exponential backoff: `base_delay * 2^attempt`, capped at `max_delay`.
    /// Adds jitter of ±25% to avoid thundering herd.
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let exp_delay = self.base_delay_ms.saturating_mul(1u64 << attempt.min(10));
        let capped = exp_delay.min(self.max_delay_ms);

        // Add jitter: ±25%
        let jitter_range = capped / 4;
        let jitter = if jitter_range > 0 {
            // Use system time nanos + attempt as entropy for non-deterministic jitter,
            // so different clients/attempts spread their retries to avoid thundering herd.
            let entropy = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos() as u64)
                .unwrap_or(0);
            let jitter_offset = (entropy ^ (attempt as u64).wrapping_mul(0x517cc1b727220a95))
                % (jitter_range * 2 + 1);
            capped - jitter_range + jitter_offset
        } else {
            capped
        };

        Duration::from_millis(jitter)
    }

    /// Parse `Retry-After` header value to get a delay duration.
    ///
    /// Supports:
    /// - Integer seconds (e.g., "5")
    /// - Decimal seconds (e.g., "1.5")
    ///
    /// Returns `None` if the header is missing or unparseable.
    pub fn parse_retry_after(header_value: Option<&str>) -> Option<Duration> {
        let value = header_value?.trim();
        // Try parsing as float seconds (covers both "5" and "1.5")
        if let Ok(seconds) = value.parse::<f64>() {
            if seconds > 0.0 && seconds <= 300.0 {
                return Some(Duration::from_secs_f64(seconds));
            }
        }
        None
    }
}

/// Outcome of a single HTTP attempt, used by the retry loop
#[derive(Debug)]
pub enum AttemptOutcome<T> {
    /// Request succeeded
    Success(T),
    /// Request failed with a retryable error
    Retryable {
        status: StatusCode,
        body: String,
        retry_after: Option<Duration>,
    },
    /// Request failed with a non-retryable error (bail immediately)
    Fatal(anyhow::Error),
}

/// Execute an async operation with retry logic.
///
/// The `operation` closure is called on each attempt and must return an `AttemptOutcome`.
/// On retryable failures, waits with exponential backoff before retrying.
/// On fatal failures or after exhausting retries, returns an error.
pub async fn with_retry<T, F, Fut>(config: &RetryConfig, operation: F) -> anyhow::Result<T>
where
    F: Fn(u32) -> Fut,
    Fut: std::future::Future<Output = AttemptOutcome<T>>,
{
    let mut last_status = None;
    let mut last_body = String::new();

    for attempt in 0..=config.max_retries {
        match operation(attempt).await {
            AttemptOutcome::Success(value) => {
                if attempt > 0 {
                    tracing::info!("LLM API request succeeded after {} retries", attempt);
                }
                return Ok(value);
            }
            AttemptOutcome::Fatal(err) => {
                return Err(err);
            }
            AttemptOutcome::Retryable {
                status,
                body,
                retry_after,
            } => {
                last_status = Some(status);
                last_body = body;

                if attempt < config.max_retries {
                    // Determine delay: prefer Retry-After header, fallback to exponential backoff
                    let delay = retry_after.unwrap_or_else(|| config.delay_for_attempt(attempt));

                    tracing::warn!(
                        "LLM API request failed with {} (attempt {}/{}), retrying in {:?}",
                        status,
                        attempt + 1,
                        config.max_retries + 1,
                        delay,
                    );

                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    // All retries exhausted
    let status = last_status.unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    anyhow::bail!(
        "LLM API request failed after {} attempts. Last status: {} Body: {}",
        config.max_retries + 1,
        status,
        last_body,
    )
}

/// Heuristic: is this a transient *network* error worth retrying — a timeout,
/// connection reset/refused/closed, broken pipe, DNS failure, or a request that
/// dropped mid-flight? These carry no HTTP status (so `is_retryable_status`
/// can't see them), yet Claude Code retries them just like 429/5xx. We only have
/// the error's rendered text (a `CodeError`/`anyhow::Error` chain) to classify.
pub fn is_transient_error<E: std::fmt::Display>(e: &E) -> bool {
    let m = e.to_string().to_lowercase();
    [
        "timed out",
        "timeout",
        "connection reset",
        "connection refused",
        "connection closed",
        "connection aborted",
        "connection error",
        "broken pipe",
        "reset by peer",
        "error sending request",
        "incomplete message",
        "unexpected eof",
        "dns error",
        "unreachable",
        "tls handshake",
        "request error",
        "body error",
        "decoding response",
        "channel closed",
        "stream closed",
    ]
    .iter()
    .any(|p| m.contains(p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn transient_error_classification() {
        let t = |s: &str| is_transient_error(&anyhow::anyhow!("{s}"));
        // Transient network errors → retry.
        assert!(t("error sending request for url: operation timed out"));
        assert!(t("connection reset by peer"));
        assert!(t("LLM error: connection closed before message completed"));
        assert!(t("tls handshake eof"));
        // Real application errors → do NOT retry.
        assert!(!t("invalid api key"));
        assert!(!t("model not found"));
        assert!(!t("context length exceeded"));
    }

    // ========================================================================
    // RetryConfig unit tests
    // ========================================================================

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.base_delay_ms, 1000);
        assert_eq!(config.max_delay_ms, 30_000);
        assert_eq!(config.retryable_status_codes, vec![429, 500, 502, 503, 529]);
    }

    #[test]
    fn test_retry_config_disabled() {
        let config = RetryConfig::disabled();
        assert_eq!(config.max_retries, 0);
    }

    #[test]
    fn test_is_retryable_status() {
        let config = RetryConfig::default();
        assert!(config.is_retryable_status(StatusCode::TOO_MANY_REQUESTS)); // 429
        assert!(config.is_retryable_status(StatusCode::INTERNAL_SERVER_ERROR)); // 500
        assert!(config.is_retryable_status(StatusCode::BAD_GATEWAY)); // 502
        assert!(config.is_retryable_status(StatusCode::SERVICE_UNAVAILABLE)); // 503
                                                                              // 529 is not a standard StatusCode, test via from_u16
        assert!(config.is_retryable_status(StatusCode::from_u16(529).unwrap()));

        // Non-retryable
        assert!(!config.is_retryable_status(StatusCode::OK)); // 200
        assert!(!config.is_retryable_status(StatusCode::BAD_REQUEST)); // 400
        assert!(!config.is_retryable_status(StatusCode::UNAUTHORIZED)); // 401
        assert!(!config.is_retryable_status(StatusCode::FORBIDDEN)); // 403
        assert!(!config.is_retryable_status(StatusCode::NOT_FOUND)); // 404
    }

    #[test]
    fn test_delay_for_attempt_exponential() {
        let config = RetryConfig {
            base_delay_ms: 1000,
            max_delay_ms: 60_000,
            ..Default::default()
        };

        // Attempt 0: ~1000ms (with jitter)
        let d0 = config.delay_for_attempt(0);
        assert!(d0.as_millis() >= 750 && d0.as_millis() <= 1250);

        // Attempt 1: ~2000ms (with jitter)
        let d1 = config.delay_for_attempt(1);
        assert!(d1.as_millis() >= 1500 && d1.as_millis() <= 2500);

        // Attempt 2: ~4000ms (with jitter)
        let d2 = config.delay_for_attempt(2);
        assert!(d2.as_millis() >= 3000 && d2.as_millis() <= 5000);
    }

    #[test]
    fn test_delay_capped_at_max() {
        let config = RetryConfig {
            base_delay_ms: 1000,
            max_delay_ms: 5000,
            ..Default::default()
        };

        // Attempt 10 would be 1024s without cap, should be capped at 5s
        let d = config.delay_for_attempt(10);
        assert!(d.as_millis() <= 6250); // 5000 + 25% jitter
    }

    #[test]
    fn test_delay_zero_base() {
        let config = RetryConfig {
            base_delay_ms: 0,
            max_delay_ms: 1000,
            ..Default::default()
        };
        let d = config.delay_for_attempt(0);
        assert_eq!(d.as_millis(), 0);
    }

    // ========================================================================
    // Retry-After header parsing
    // ========================================================================

    #[test]
    fn test_parse_retry_after_integer() {
        let d = RetryConfig::parse_retry_after(Some("5"));
        assert_eq!(d, Some(Duration::from_secs(5)));
    }

    #[test]
    fn test_parse_retry_after_decimal() {
        let d = RetryConfig::parse_retry_after(Some("1.5"));
        assert_eq!(d, Some(Duration::from_secs_f64(1.5)));
    }

    #[test]
    fn test_parse_retry_after_none() {
        assert_eq!(RetryConfig::parse_retry_after(None), None);
    }

    #[test]
    fn test_parse_retry_after_invalid() {
        assert_eq!(RetryConfig::parse_retry_after(Some("not-a-number")), None);
    }

    #[test]
    fn test_parse_retry_after_negative() {
        assert_eq!(RetryConfig::parse_retry_after(Some("-1")), None);
    }

    #[test]
    fn test_parse_retry_after_zero() {
        assert_eq!(RetryConfig::parse_retry_after(Some("0")), None);
    }

    #[test]
    fn test_parse_retry_after_too_large() {
        // > 300s should be rejected
        assert_eq!(RetryConfig::parse_retry_after(Some("301")), None);
    }

    #[test]
    fn test_parse_retry_after_with_whitespace() {
        let d = RetryConfig::parse_retry_after(Some("  3  "));
        assert_eq!(d, Some(Duration::from_secs(3)));
    }

    // ========================================================================
    // RetryConfig serialization
    // ========================================================================

    #[test]
    fn test_retry_config_serde_roundtrip() {
        let config = RetryConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RetryConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.max_retries, config.max_retries);
        assert_eq!(deserialized.base_delay_ms, config.base_delay_ms);
        assert_eq!(deserialized.max_delay_ms, config.max_delay_ms);
        assert_eq!(
            deserialized.retryable_status_codes,
            config.retryable_status_codes
        );
    }

    #[test]
    fn test_retry_config_deserialize_custom() {
        let json = r#"{"max_retries":5,"base_delay_ms":500,"max_delay_ms":10000,"retryable_status_codes":[429,503]}"#;
        let config: RetryConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.base_delay_ms, 500);
        assert_eq!(config.max_delay_ms, 10_000);
        assert_eq!(config.retryable_status_codes, vec![429, 503]);
    }

    // ========================================================================
    // with_retry integration tests
    // ========================================================================

    #[tokio::test]
    async fn test_with_retry_success_first_attempt() {
        let config = RetryConfig::default();
        let call_count = Arc::new(AtomicU32::new(0));
        let cc = call_count.clone();

        let result = with_retry(&config, |_attempt| {
            let cc = cc.clone();
            async move {
                cc.fetch_add(1, Ordering::SeqCst);
                AttemptOutcome::Success("ok")
            }
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "ok");
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_with_retry_success_after_retries() {
        let config = RetryConfig {
            max_retries: 3,
            base_delay_ms: 10, // Fast for tests
            max_delay_ms: 50,
            ..Default::default()
        };
        let call_count = Arc::new(AtomicU32::new(0));
        let cc = call_count.clone();

        let result = with_retry(&config, |attempt| {
            let cc = cc.clone();
            async move {
                cc.fetch_add(1, Ordering::SeqCst);
                if attempt < 2 {
                    AttemptOutcome::Retryable {
                        status: StatusCode::TOO_MANY_REQUESTS,
                        body: "rate limited".to_string(),
                        retry_after: None,
                    }
                } else {
                    AttemptOutcome::Success("recovered")
                }
            }
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "recovered");
        assert_eq!(call_count.load(Ordering::SeqCst), 3); // 2 failures + 1 success
    }

    #[tokio::test]
    async fn test_with_retry_all_retries_exhausted() {
        let config = RetryConfig {
            max_retries: 2,
            base_delay_ms: 10,
            max_delay_ms: 50,
            ..Default::default()
        };
        let call_count = Arc::new(AtomicU32::new(0));
        let cc = call_count.clone();

        let result: anyhow::Result<&str> = with_retry(&config, |_attempt| {
            let cc = cc.clone();
            async move {
                cc.fetch_add(1, Ordering::SeqCst);
                AttemptOutcome::Retryable {
                    status: StatusCode::SERVICE_UNAVAILABLE,
                    body: "service down".to_string(),
                    retry_after: None,
                }
            }
        })
        .await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("3 attempts")); // max_retries(2) + 1
        assert!(err.contains("503"));
        assert!(err.contains("service down"));
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_with_retry_fatal_error_no_retry() {
        let config = RetryConfig {
            max_retries: 3,
            base_delay_ms: 10,
            max_delay_ms: 50,
            ..Default::default()
        };
        let call_count = Arc::new(AtomicU32::new(0));
        let cc = call_count.clone();

        let result: anyhow::Result<&str> = with_retry(&config, |_attempt| {
            let cc = cc.clone();
            async move {
                cc.fetch_add(1, Ordering::SeqCst);
                AttemptOutcome::Fatal(anyhow::anyhow!("invalid API key"))
            }
        })
        .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid API key"));
        assert_eq!(call_count.load(Ordering::SeqCst), 1); // No retries for fatal
    }

    #[tokio::test]
    async fn test_with_retry_disabled() {
        let config = RetryConfig::disabled();
        let call_count = Arc::new(AtomicU32::new(0));
        let cc = call_count.clone();

        let result: anyhow::Result<&str> = with_retry(&config, |_attempt| {
            let cc = cc.clone();
            async move {
                cc.fetch_add(1, Ordering::SeqCst);
                AttemptOutcome::Retryable {
                    status: StatusCode::TOO_MANY_REQUESTS,
                    body: "rate limited".to_string(),
                    retry_after: None,
                }
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(call_count.load(Ordering::SeqCst), 1); // Only initial attempt, no retries
    }

    #[tokio::test]
    async fn test_with_retry_respects_retry_after_header() {
        let config = RetryConfig {
            max_retries: 1,
            base_delay_ms: 10,
            max_delay_ms: 50,
            ..Default::default()
        };
        let call_count = Arc::new(AtomicU32::new(0));
        let cc = call_count.clone();

        let start = tokio::time::Instant::now();
        let result = with_retry(&config, |attempt| {
            let cc = cc.clone();
            async move {
                cc.fetch_add(1, Ordering::SeqCst);
                if attempt == 0 {
                    AttemptOutcome::Retryable {
                        status: StatusCode::TOO_MANY_REQUESTS,
                        body: "rate limited".to_string(),
                        retry_after: Some(Duration::from_millis(100)),
                    }
                } else {
                    AttemptOutcome::Success("ok")
                }
            }
        })
        .await;

        assert!(result.is_ok());
        // Should have waited at least 100ms (the retry-after value)
        assert!(start.elapsed() >= Duration::from_millis(90));
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }
}
