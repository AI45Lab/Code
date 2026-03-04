//! Vector RAG Context Provider
//!
//! Implements `ContextProvider` using vector embeddings for semantic code search.
//! Indexes workspace files into a vector store and retrieves relevant context
//! via cosine similarity on each agent turn.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use a3s_code_core::context::{
//!     VectorContextProvider, VectorContextConfig,
//!     embedding::OpenAiEmbeddingProvider,
//!     vector_store::InMemoryVectorStore,
//! };
//!
//! let embedder = OpenAiEmbeddingProvider::new("sk-...", "text-embedding-3-small", 1536)?;
//! let store = InMemoryVectorStore::new();
//! let config = VectorContextConfig::new("/my-project");
//!
//! let provider = VectorContextProvider::new(config, embedder, store);
//! provider.index().await?; // Index workspace files
//!
//! // Then register as a context provider on the session
//! ```

use super::embedding::EmbeddingProvider;
use super::vector_store::{VectorEntry, VectorMetadata, VectorStore};
use super::{ContextItem, ContextProvider, ContextQuery, ContextResult, ContextType};
use anyhow::Result;
use async_trait::async_trait;
use ignore::WalkBuilder;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Configuration for the vector context provider
#[derive(Debug, Clone)]
pub struct VectorContextConfig {
    /// Root directory to index
    pub root_path: PathBuf,
    /// Include file patterns (glob syntax)
    pub include_patterns: Vec<String>,
    /// Exclude file patterns (glob syntax)
    pub exclude_patterns: Vec<String>,
    /// Maximum file size in bytes (default: 512KB)
    pub max_file_size: usize,
    /// Maximum chunk size in characters (default: 2000)
    pub max_chunk_chars: usize,
    /// Minimum relevance score threshold (default: 0.3)
    pub min_relevance: f32,
    /// Whether to auto-index on first query (default: true)
    pub auto_index: bool,
}

impl VectorContextConfig {
    /// Create a new config with sensible defaults
    pub fn new(root_path: impl Into<PathBuf>) -> Self {
        Self {
            root_path: root_path.into(),
            include_patterns: vec![
                "**/*.rs".to_string(),
                "**/*.py".to_string(),
                "**/*.ts".to_string(),
                "**/*.js".to_string(),
                "**/*.go".to_string(),
                "**/*.md".to_string(),
                "**/*.toml".to_string(),
                "**/*.yaml".to_string(),
                "**/*.yml".to_string(),
            ],
            exclude_patterns: vec![
                "**/target/**".to_string(),
                "**/node_modules/**".to_string(),
                "**/.git/**".to_string(),
                "**/dist/**".to_string(),
                "**/build/**".to_string(),
                "**/*.lock".to_string(),
            ],
            max_file_size: 512 * 1024,
            max_chunk_chars: 2000,
            min_relevance: 0.3,
            auto_index: true,
        }
    }

    /// Set include patterns
    pub fn with_include_patterns(mut self, patterns: Vec<String>) -> Self {
        self.include_patterns = patterns;
        self
    }

    /// Set exclude patterns
    pub fn with_exclude_patterns(mut self, patterns: Vec<String>) -> Self {
        self.exclude_patterns = patterns;
        self
    }

    /// Set max file size
    pub fn with_max_file_size(mut self, size: usize) -> Self {
        self.max_file_size = size;
        self
    }

    /// Set max chunk size in characters
    pub fn with_max_chunk_chars(mut self, chars: usize) -> Self {
        self.max_chunk_chars = chars;
        self
    }

    /// Set minimum relevance score
    pub fn with_min_relevance(mut self, score: f32) -> Self {
        self.min_relevance = score.clamp(0.0, 1.0);
        self
    }

    /// Enable/disable auto-indexing on first query
    pub fn with_auto_index(mut self, enabled: bool) -> Self {
        self.auto_index = enabled;
        self
    }
}

/// A chunk of code extracted from a file
#[derive(Debug, Clone)]
struct CodeChunk {
    /// Source file path
    path: PathBuf,
    /// Chunk content
    content: String,
    /// Chunk index within the file (0-based)
    chunk_index: usize,
}

/// Vector RAG context provider
///
/// Indexes workspace files into vector embeddings and retrieves
/// semantically relevant code context for each agent turn.
pub struct VectorContextProvider<E: EmbeddingProvider, S: VectorStore> {
    config: VectorContextConfig,
    embedder: Arc<E>,
    store: Arc<S>,
    indexed: RwLock<bool>,
}

impl<E: EmbeddingProvider, S: VectorStore> VectorContextProvider<E, S> {
    /// Create a new vector context provider
    pub fn new(config: VectorContextConfig, embedder: E, store: S) -> Self {
        Self {
            config,
            embedder: Arc::new(embedder),
            store: Arc::new(store),
            indexed: RwLock::new(false),
        }
    }

    /// Create with shared (Arc) embedder and store
    pub fn with_shared(config: VectorContextConfig, embedder: Arc<E>, store: Arc<S>) -> Self {
        Self {
            config,
            embedder,
            store,
            indexed: RwLock::new(false),
        }
    }

    /// Index all workspace files into the vector store
    pub async fn index(&self) -> Result<usize> {
        let chunks = self.collect_chunks().await?;
        let chunk_count = chunks.len();

        if chunks.is_empty() {
            tracing::debug!("No files to index for vector context");
            *self.indexed.write().await = true;
            return Ok(0);
        }

        tracing::info!(
            chunks = chunk_count,
            root = %self.config.root_path.display(),
            "Indexing workspace for vector context"
        );

        // Embed in batches of 32
        let batch_size = 32;
        let mut entries = Vec::with_capacity(chunk_count);

        for batch_start in (0..chunks.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(chunks.len());
            let batch_texts: Vec<&str> = chunks[batch_start..batch_end]
                .iter()
                .map(|c| c.content.as_str())
                .collect();

            let embeddings = self.embedder.embed_batch(&batch_texts).await?;

            for (i, embedding) in embeddings.into_iter().enumerate() {
                let chunk = &chunks[batch_start + i];
                let id = format!("{}#{}", chunk.path.display(), chunk.chunk_index);

                entries.push(VectorEntry {
                    id,
                    embedding,
                    metadata: VectorMetadata {
                        source: format!("file:{}", chunk.path.display()),
                        content_type: detect_content_type(&chunk.path),
                        content: chunk.content.clone(),
                        token_count: chunk.content.split_whitespace().count(),
                        extra: {
                            let mut m = std::collections::HashMap::new();
                            m.insert(
                                "path".to_string(),
                                serde_json::Value::String(chunk.path.to_string_lossy().to_string()),
                            );
                            m.insert(
                                "chunk_index".to_string(),
                                serde_json::Value::Number(chunk.chunk_index.into()),
                            );
                            m
                        },
                    },
                });
            }
        }

        self.store.insert_batch(entries).await?;
        *self.indexed.write().await = true;

        tracing::info!(chunks = chunk_count, "Vector context indexing complete");

        Ok(chunk_count)
    }

    /// Collect and chunk all matching files
    async fn collect_chunks(&self) -> Result<Vec<CodeChunk>> {
        let root = self.config.root_path.clone();
        let max_file_size = self.config.max_file_size;
        let max_chunk_chars = self.config.max_chunk_chars;
        let include = self.config.include_patterns.clone();
        let exclude = self.config.exclude_patterns.clone();

        // File I/O in blocking task
        tokio::task::spawn_blocking(move || {
            let mut chunks = Vec::new();

            let walker = WalkBuilder::new(&root)
                .hidden(false)
                .git_ignore(true)
                .build();

            for entry in walker {
                let entry = entry.map_err(|e| anyhow::anyhow!("Walk error: {}", e))?;
                let path = entry.path();

                if !path.is_file() {
                    continue;
                }

                let metadata = std::fs::metadata(path)
                    .map_err(|e| anyhow::anyhow!("Metadata error for {}: {}", path.display(), e))?;

                if metadata.len() > max_file_size as u64 {
                    continue;
                }

                if !matches_patterns(path, &include, true) {
                    continue;
                }

                if matches_patterns(path, &exclude, false) {
                    continue;
                }

                let content = match std::fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(_) => continue, // Skip binary files
                };

                if content.trim().is_empty() {
                    continue;
                }

                // Chunk the file
                let file_chunks = chunk_text(&content, max_chunk_chars);
                for (i, chunk_content) in file_chunks.into_iter().enumerate() {
                    chunks.push(CodeChunk {
                        path: path.to_path_buf(),
                        content: chunk_content,
                        chunk_index: i,
                    });
                }
            }

            Ok::<_, anyhow::Error>(chunks)
        })
        .await
        .map_err(|e| anyhow::anyhow!("Spawn blocking failed: {}", e))?
    }

    /// Ensure the store is indexed (auto-index on first query)
    async fn ensure_indexed(&self) -> Result<()> {
        if *self.indexed.read().await {
            return Ok(());
        }
        if self.config.auto_index {
            self.index().await?;
        }
        Ok(())
    }
}

#[async_trait]
impl<E: EmbeddingProvider + 'static, S: VectorStore + 'static> ContextProvider
    for VectorContextProvider<E, S>
{
    fn name(&self) -> &str {
        "vector-rag"
    }

    async fn query(&self, query: &ContextQuery) -> Result<ContextResult> {
        self.ensure_indexed().await?;

        // Embed the query
        let query_embedding = self.embedder.embed(&query.query).await?;

        // Search vector store
        let search_results = self
            .store
            .search(&query_embedding, query.max_results)
            .await?;

        // Convert to ContextItems, filtering by min relevance
        let mut result = ContextResult::new("vector-rag");
        let mut total_tokens = 0usize;

        for sr in search_results {
            if sr.score < self.config.min_relevance {
                continue;
            }

            if total_tokens >= query.max_tokens {
                result.truncated = true;
                break;
            }

            let content = match query.depth {
                super::ContextDepth::Abstract => {
                    sr.metadata.content.chars().take(500).collect::<String>()
                }
                super::ContextDepth::Overview => {
                    sr.metadata.content.chars().take(2000).collect::<String>()
                }
                super::ContextDepth::Full => sr.metadata.content.clone(),
            };

            let token_count = content.split_whitespace().count();
            total_tokens += token_count;

            result.add_item(
                ContextItem::new(sr.id, ContextType::Resource, content)
                    .with_token_count(token_count)
                    .with_relevance(sr.score)
                    .with_source(&sr.metadata.source)
                    .with_metadata("content_type", serde_json::json!(sr.metadata.content_type)),
            );
        }

        Ok(result)
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Check if a path matches any of the given glob patterns
fn matches_patterns(path: &Path, patterns: &[String], default_if_empty: bool) -> bool {
    if patterns.is_empty() {
        return default_if_empty;
    }
    let path_str = path.to_string_lossy().replace('\\', "/");
    patterns.iter().any(|pattern| {
        glob::Pattern::new(pattern)
            .map(|p| p.matches(&path_str))
            .unwrap_or(false)
    })
}

/// Chunk text into segments of approximately `max_chars` characters
///
/// Splits on double-newlines (paragraph boundaries) first, then on single
/// newlines if chunks are still too large. Preserves logical boundaries.
fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    if text.len() <= max_chars {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();

    // Split on double-newline (paragraph/function boundaries)
    for paragraph in text.split("\n\n") {
        if current.len() + paragraph.len() + 2 > max_chars && !current.is_empty() {
            chunks.push(current.trim().to_string());
            current = String::new();
        }

        if paragraph.len() > max_chars {
            // Large paragraph: split on single newlines
            if !current.is_empty() {
                chunks.push(current.trim().to_string());
                current = String::new();
            }
            for line in paragraph.split('\n') {
                if current.len() + line.len() + 1 > max_chars && !current.is_empty() {
                    chunks.push(current.trim().to_string());
                    current = String::new();
                }
                if !current.is_empty() {
                    current.push('\n');
                }
                current.push_str(line);
            }
        } else {
            if !current.is_empty() {
                current.push_str("\n\n");
            }
            current.push_str(paragraph);
        }
    }

    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }

    chunks
}

/// Detect content type from file extension
fn detect_content_type(path: &Path) -> String {
    match path.extension().and_then(|e| e.to_str()) {
        Some("rs") => "rust",
        Some("py") => "python",
        Some("ts") | Some("tsx") => "typescript",
        Some("js") | Some("jsx") => "javascript",
        Some("go") => "go",
        Some("md") | Some("mdx") => "markdown",
        Some("toml") | Some("yaml") | Some("yml") | Some("json") => "config",
        _ => "text",
    }
    .to_string()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::vector_store::InMemoryVectorStore;
    use anyhow::Result;
    use std::fs::{self, File};
    use std::io::Write;
    use tempfile::TempDir;

    /// Simple mock that returns deterministic embeddings based on text hash
    struct MockEmbeddingProvider {
        dim: usize,
    }

    impl MockEmbeddingProvider {
        fn new(dim: usize) -> Self {
            Self { dim }
        }
    }

    #[async_trait]
    impl EmbeddingProvider for MockEmbeddingProvider {
        fn name(&self) -> &str {
            "mock-embedding"
        }

        fn dimension(&self) -> usize {
            self.dim
        }

        async fn embed(&self, text: &str) -> Result<super::super::embedding::Embedding> {
            let mut embedding = vec![0.0f32; self.dim];
            for (i, byte) in text.bytes().enumerate() {
                embedding[i % self.dim] += (byte as f32) / 255.0;
            }
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut embedding {
                    *v /= norm;
                }
            }
            Ok(embedding)
        }
    }

    fn setup_test_workspace() -> TempDir {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        // Create some test files
        let mut f1 = File::create(root.join("main.rs")).unwrap();
        writeln!(f1, "fn main() {{\n    println!(\"Hello, world!\");\n}}").unwrap();

        let mut f2 = File::create(root.join("lib.rs")).unwrap();
        writeln!(
            f2,
            "pub mod auth;\npub mod database;\n\npub fn init() -> Result<()> {{\n    Ok(())\n}}"
        )
        .unwrap();

        let mut f3 = File::create(root.join("README.md")).unwrap();
        writeln!(
            f3,
            "# My Project\n\nA Rust project for testing vector RAG context."
        )
        .unwrap();

        fs::create_dir(root.join("src")).unwrap();
        let mut f4 = File::create(root.join("src/auth.rs")).unwrap();
        writeln!(
            f4,
            "use jwt::Token;\n\npub fn verify_token(token: &str) -> Result<Claims> {{\n    // JWT verification logic\n    todo!()\n}}"
        )
        .unwrap();

        dir
    }

    // -- VectorContextConfig tests --

    #[test]
    fn test_config_defaults() {
        let config = VectorContextConfig::new("/tmp/test");
        assert_eq!(config.root_path, PathBuf::from("/tmp/test"));
        assert!(!config.include_patterns.is_empty());
        assert!(!config.exclude_patterns.is_empty());
        assert_eq!(config.max_file_size, 512 * 1024);
        assert_eq!(config.max_chunk_chars, 2000);
        assert!((config.min_relevance - 0.3).abs() < f32::EPSILON);
        assert!(config.auto_index);
    }

    #[test]
    fn test_config_builders() {
        let config = VectorContextConfig::new("/tmp")
            .with_include_patterns(vec!["**/*.rs".to_string()])
            .with_exclude_patterns(vec!["**/test/**".to_string()])
            .with_max_file_size(1024)
            .with_max_chunk_chars(500)
            .with_min_relevance(0.5)
            .with_auto_index(false);

        assert_eq!(config.include_patterns, vec!["**/*.rs"]);
        assert_eq!(config.exclude_patterns, vec!["**/test/**"]);
        assert_eq!(config.max_file_size, 1024);
        assert_eq!(config.max_chunk_chars, 500);
        assert!((config.min_relevance - 0.5).abs() < f32::EPSILON);
        assert!(!config.auto_index);
    }

    #[test]
    fn test_config_min_relevance_clamping() {
        let c1 = VectorContextConfig::new("/tmp").with_min_relevance(1.5);
        assert!((c1.min_relevance - 1.0).abs() < f32::EPSILON);

        let c2 = VectorContextConfig::new("/tmp").with_min_relevance(-0.5);
        assert!(c2.min_relevance.abs() < f32::EPSILON);
    }

    // -- chunk_text tests --

    #[test]
    fn test_chunk_text_small() {
        let chunks = chunk_text("hello world", 100);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello world");
    }

    #[test]
    fn test_chunk_text_splits_on_double_newline() {
        let text = "paragraph one\n\nparagraph two\n\nparagraph three";
        let chunks = chunk_text(text, 20);
        assert!(chunks.len() >= 2);
        assert!(chunks[0].contains("paragraph one"));
    }

    #[test]
    fn test_chunk_text_large_paragraph() {
        let text = (0..100)
            .map(|i| format!("line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = chunk_text(&text, 50);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            // Each chunk should be roughly within limit (may exceed slightly due to line boundaries)
            assert!(chunk.len() < 200, "Chunk too large: {} chars", chunk.len());
        }
    }

    #[test]
    fn test_chunk_text_empty() {
        let chunks = chunk_text("", 100);
        assert_eq!(chunks.len(), 1);
    }

    // -- detect_content_type tests --

    #[test]
    fn test_detect_content_type() {
        assert_eq!(detect_content_type(Path::new("main.rs")), "rust");
        assert_eq!(detect_content_type(Path::new("app.py")), "python");
        assert_eq!(detect_content_type(Path::new("index.ts")), "typescript");
        assert_eq!(detect_content_type(Path::new("app.tsx")), "typescript");
        assert_eq!(detect_content_type(Path::new("main.go")), "go");
        assert_eq!(detect_content_type(Path::new("README.md")), "markdown");
        assert_eq!(detect_content_type(Path::new("Cargo.toml")), "config");
        assert_eq!(detect_content_type(Path::new("config.yaml")), "config");
        assert_eq!(detect_content_type(Path::new("unknown.xyz")), "text");
    }

    // -- matches_patterns tests --

    #[test]
    fn test_matches_patterns_empty_default_true() {
        assert!(matches_patterns(Path::new("test.rs"), &[], true));
    }

    #[test]
    fn test_matches_patterns_empty_default_false() {
        assert!(!matches_patterns(Path::new("test.rs"), &[], false));
    }

    // -- VectorContextProvider integration tests --

    #[tokio::test]
    async fn test_provider_index() {
        let dir = setup_test_workspace();
        let config = VectorContextConfig::new(dir.path()).with_auto_index(false);
        let embedder = MockEmbeddingProvider::new(8);
        let store = InMemoryVectorStore::new();

        let provider = VectorContextProvider::new(config, embedder, store);
        let count = provider.index().await.unwrap();
        assert!(count > 0, "Should have indexed some chunks");
    }

    #[tokio::test]
    async fn test_provider_query() {
        let dir = setup_test_workspace();
        let config = VectorContextConfig::new(dir.path())
            .with_min_relevance(0.0) // Accept all results for testing
            .with_auto_index(false);
        let provider = VectorContextProvider::new(
            config,
            MockEmbeddingProvider::new(8),
            InMemoryVectorStore::new(),
        );
        provider.index().await.unwrap();

        let query = ContextQuery::new("authentication JWT token");
        let result = ContextProvider::query(&provider, &query).await.unwrap();

        assert_eq!(result.provider, "vector-rag");
        assert!(!result.items.is_empty());
        // All items should be Resource type
        for item in &result.items {
            assert_eq!(item.context_type, ContextType::Resource);
            assert!(item.source.is_some());
        }
    }

    #[tokio::test]
    async fn test_provider_auto_index() {
        let dir = setup_test_workspace();
        let config = VectorContextConfig::new(dir.path())
            .with_min_relevance(0.0)
            .with_auto_index(true);
        let provider = VectorContextProvider::new(
            config,
            MockEmbeddingProvider::new(8),
            InMemoryVectorStore::new(),
        );

        // Should auto-index on first query
        let query = ContextQuery::new("hello");
        let result = ContextProvider::query(&provider, &query).await.unwrap();
        assert!(!result.items.is_empty());
    }

    #[tokio::test]
    async fn test_provider_empty_workspace() {
        let dir = TempDir::new().unwrap();
        let config = VectorContextConfig::new(dir.path()).with_auto_index(false);
        let provider = VectorContextProvider::new(
            config,
            MockEmbeddingProvider::new(8),
            InMemoryVectorStore::new(),
        );
        let count = provider.index().await.unwrap();
        assert_eq!(count, 0);

        let query = ContextQuery::new("anything");
        let result = ContextProvider::query(&provider, &query).await.unwrap();
        assert!(result.items.is_empty());
    }

    #[tokio::test]
    async fn test_provider_respects_max_results() {
        let dir = setup_test_workspace();
        let config = VectorContextConfig::new(dir.path())
            .with_min_relevance(0.0)
            .with_auto_index(false);
        let provider = VectorContextProvider::new(
            config,
            MockEmbeddingProvider::new(8),
            InMemoryVectorStore::new(),
        );
        provider.index().await.unwrap();

        let query = ContextQuery::new("test").with_max_results(1);
        let result = ContextProvider::query(&provider, &query).await.unwrap();
        assert!(result.items.len() <= 1);
    }

    #[tokio::test]
    async fn test_provider_respects_max_tokens() {
        let dir = setup_test_workspace();
        let config = VectorContextConfig::new(dir.path())
            .with_min_relevance(0.0)
            .with_auto_index(false);
        let provider = VectorContextProvider::new(
            config,
            MockEmbeddingProvider::new(8),
            InMemoryVectorStore::new(),
        );
        provider.index().await.unwrap();

        let query = ContextQuery::new("test").with_max_tokens(5);
        let result = ContextProvider::query(&provider, &query).await.unwrap();
        // Should truncate early due to token limit
        assert!(result.total_tokens <= 50); // Some slack for word counting
    }

    #[tokio::test]
    async fn test_provider_with_shared() {
        let dir = setup_test_workspace();
        let config = VectorContextConfig::new(dir.path())
            .with_min_relevance(0.0)
            .with_auto_index(false);
        let embedder = Arc::new(MockEmbeddingProvider::new(8));
        let store = Arc::new(InMemoryVectorStore::new());

        let provider =
            VectorContextProvider::with_shared(config, Arc::clone(&embedder), Arc::clone(&store));
        provider.index().await.unwrap();

        // Store should have entries accessible via the shared Arc
        assert!(!store.is_empty().await);
    }

    #[tokio::test]
    async fn test_provider_name() {
        let dir = TempDir::new().unwrap();
        let config = VectorContextConfig::new(dir.path());
        let provider = VectorContextProvider::new(
            config,
            MockEmbeddingProvider::new(4),
            InMemoryVectorStore::new(),
        );
        assert_eq!(ContextProvider::name(&provider), "vector-rag");
    }

    #[tokio::test]
    async fn test_provider_context_depth_abstract() {
        let dir = setup_test_workspace();
        let config = VectorContextConfig::new(dir.path())
            .with_min_relevance(0.0)
            .with_auto_index(false);
        let provider = VectorContextProvider::new(
            config,
            MockEmbeddingProvider::new(8),
            InMemoryVectorStore::new(),
        );
        provider.index().await.unwrap();

        let query = ContextQuery::new("test").with_depth(crate::context::ContextDepth::Abstract);
        let result = ContextProvider::query(&provider, &query).await.unwrap();
        for item in &result.items {
            assert!(item.content.len() <= 500);
        }
    }
}
