//! Integration tests for web_search tool with headless engines
//!
//! Run basic tests: cargo test -p a3s-code-core --test test_web_search_headless
//! Run with actual browser: cargo test -p a3s-code-core --test test_web_search_headless -- --ignored

use a3s_code_core::config::{BrowserBackend, HeadlessConfig, SearchConfig};
use a3s_code_core::tools::{ToolContext, ToolExecutor};

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Helper to create a ToolContext with headless search config
fn make_context(headless: Option<HeadlessConfig>) -> ToolContext {
    let search_config = headless.map(|h| {
        Arc::new(SearchConfig {
            timeout: 30,
            health: None,
            engines: HashMap::new(),
            headless: Some(h),
        })
    });

    let mut context = ToolContext::new(PathBuf::from("/tmp")).with_session_id("test-session");
    context.search_config = search_config;
    context
}

#[tokio::test]
async fn test_web_search_tool_creation() {
    let executor = ToolExecutor::new("/tmp".to_string());
    let definitions = executor.definitions();

    assert!(
        definitions.iter().any(|t| t.name == "web_search"),
        "web_search tool should be registered"
    );
}

#[tokio::test]
async fn test_web_search_http_engine() {
    let executor = ToolExecutor::new("/tmp".to_string());

    // HTTP engine (duckduckgo) doesn't need headless config
    let args = serde_json::json!({
        "query": "test",
        "engines": "duckduckgo"
    });

    let result = executor.execute("web_search", &args).await;

    match result {
        Ok(output) => {
            println!("✅ DuckDuckGo search succeeded!");
            println!("Output length: {}", output.output.len());
            println!("Exit code: {}", output.exit_code);
        }
        Err(e) => {
            println!("⚠️  DuckDuckGo search failed: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_web_search_with_baidu_headless_engine() {
    let executor = ToolExecutor::new("/tmp".to_string());

    let headless_config = HeadlessConfig {
        backend: BrowserBackend::Chrome,
        max_tabs: 2,
        browser_path: None,
        launch_args: Vec::new(),
        proxy_url: None,
    };

    let context = make_context(Some(headless_config));

    let args = serde_json::json!({
        "query": "test query",
        "engines": "baidu"
    });

    // This will attempt to use baidu with headless browser
    let result = executor
        .execute_with_context("web_search", &args, &context)
        .await;

    match result {
        Ok(output) => {
            println!("✅ Baidu headless search succeeded!");
            println!("Output length: {}", output.output.len());
            println!("Exit code: {:?}", output.exit_code);
        }
        Err(e) => {
            // If a browser is not available, this will fail - which is expected in CI
            println!(
                "⚠️  Baidu headless search failed (browser may not be available): {:?}",
                e
            );
        }
    }
}

#[tokio::test]
#[ignore] // Requires a browser to be installed
async fn test_baidu_headless_search_actual() {
    let executor = ToolExecutor::new("/tmp".to_string());

    let headless_config = HeadlessConfig {
        backend: BrowserBackend::Chrome,
        max_tabs: 2,
        browser_path: None,
        launch_args: Vec::new(),
        proxy_url: None,
    };

    let context = make_context(Some(headless_config));

    let args = serde_json::json!({
        "query": "rust programming language",
        "engines": "baidu"
    });

    let result = executor
        .execute_with_context("web_search", &args, &context)
        .await
        .unwrap();

    assert_eq!(
        result.exit_code, 0,
        "Search should succeed, got exit_code={}, output={}",
        result.exit_code, result.output
    );
    assert!(
        !result.output.trim().is_empty(),
        "Search output should not be empty"
    );
    println!("Search output: {}", result.output);
}

#[tokio::test]
#[ignore] // Requires a browser to be installed
async fn test_google_headless_search_actual() {
    let executor = ToolExecutor::new("/tmp".to_string());

    let headless_config = HeadlessConfig {
        backend: BrowserBackend::Chrome,
        max_tabs: 2,
        browser_path: None,
        launch_args: Vec::new(),
        proxy_url: None,
    };

    let context = make_context(Some(headless_config));

    let args = serde_json::json!({
        "query": "rust programming language",
        "engines": "google"
    });

    let result = executor
        .execute_with_context("web_search", &args, &context)
        .await
        .unwrap();

    assert_eq!(
        result.exit_code, 0,
        "Search should succeed, got exit_code={}, output={}",
        result.exit_code, result.output
    );
    assert!(
        !result.output.trim().is_empty(),
        "Search output should not be empty"
    );
    println!("Search output: {}", result.output);
}
