//! File System Context Provider
//!
//! Provides simple file-based RAG (Retrieval-Augmented Generation):
//! - Index files in a directory with glob pattern filtering
//! - Simple keyword-based search with relevance scoring
//! - Support for file size limits and exclusion patterns

use crate::context::{ContextItem, ContextProvider, ContextQuery, ContextResult, ContextType};
use async_trait::async_trait;
use ignore::WalkBuilder;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

/// File system context provider configuration
#[derive(Debug, Clone)]
pub struct FileSystemContextConfig {
    /// Root directory to index
    pub root_path: PathBuf,
    /// Include patterns (glob syntax: ["**/*.rs", "**/*.md"])
    pub include_patterns: Vec<String>,
    /// Exclude patterns (glob syntax: ["**/target/**", "**/node_modules/**"])
    pub exclude_patterns: Vec<String>,
    /// Maximum file size in bytes (default: 1MB)
    pub max_file_size: usize,
    /// Whether to enable cache (default: true)
    pub enable_cache: bool,
}

impl FileSystemContextConfig {
    /// Create a new config with default settings
    pub fn new(root_path: impl Into<PathBuf>) -> Self {
        Self {
            root_path: root_path.into(),
            include_patterns: vec!["**/*.rs".to_string(), "**/*.md".to_string()],
            exclude_patterns: vec![
                "**/target/**".to_string(),
                "**/node_modules/**".to_string(),
                "**/.git/**".to_string(),
            ],
            max_file_size: 1024 * 1024, // 1MB
            enable_cache: true,
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

    /// Enable/disable cache
    pub fn with_cache(mut self, enable: bool) -> Self {
        self.enable_cache = enable;
        self
    }
}

/// Indexed file entry
#[derive(Debug, Clone)]
struct IndexedFile {
    path: PathBuf,
    content: String,
    size: usize,
}

/// File system context provider
pub struct FileSystemContextProvider {
    config: FileSystemContextConfig,
    /// Cached indexed files
    cache: Arc<RwLock<HashMap<PathBuf, IndexedFile>>>,
}

impl FileSystemContextProvider {
    /// Create a new file system context provider
    pub fn new(config: FileSystemContextConfig) -> Self {
        Self {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Index files in the root directory
    async fn index_files(&self) -> anyhow::Result<Vec<IndexedFile>> {
        let mut files = Vec::new();

        let walker = WalkBuilder::new(&self.config.root_path)
            .hidden(false)
            .git_ignore(true)
            .build();

        for entry in walker {
            let entry = entry.map_err(|e| anyhow::anyhow!("Walk error: {}", e))?;
            let path = entry.path();

            if !path.is_file() {
                continue;
            }

            let metadata =
                fs::metadata(path).map_err(|e| anyhow::anyhow!("Metadata error: {}", e))?;
            if metadata.len() > self.config.max_file_size as u64 {
                continue;
            }

            if !self.matches_include_patterns(path) {
                continue;
            }

            if self.matches_exclude_patterns(path) {
                continue;
            }

            let content =
                fs::read_to_string(path).map_err(|e| anyhow::anyhow!("Read error: {}", e))?;

            files.push(IndexedFile {
                path: path.to_path_buf(),
                content,
                size: metadata.len() as usize,
            });
        }

        Ok(files)
    }

    fn matches_include_patterns(&self, path: &Path) -> bool {
        if self.config.include_patterns.is_empty() {
            return true;
        }

        // Normalize to forward slashes for consistent cross-platform glob matching
        let path_str = path.to_string_lossy().replace('\\', "/");
        self.config.include_patterns.iter().any(|pattern| {
            glob::Pattern::new(pattern)
                .map(|p| p.matches(&path_str))
                .unwrap_or(false)
        })
    }

    fn matches_exclude_patterns(&self, path: &Path) -> bool {
        // Normalize to forward slashes for consistent cross-platform glob matching
        let path_str = path.to_string_lossy().replace('\\', "/");
        self.config.exclude_patterns.iter().any(|pattern| {
            glob::Pattern::new(pattern)
                .map(|p| p.matches(&path_str))
                .unwrap_or(false)
        })
    }

    async fn search_simple(
        &self,
        query: &str,
        files: &[IndexedFile],
        max_results: usize,
    ) -> Vec<(IndexedFile, f32)> {
        let query_lower = query.to_lowercase();
        let keywords: Vec<&str> = query_lower.split_whitespace().collect();

        let mut results: Vec<(IndexedFile, f32)> = files
            .iter()
            .filter_map(|file| {
                let content_lower = file.content.to_lowercase();
                let mut score = 0.0;
                for keyword in &keywords {
                    let count = content_lower.matches(keyword).count();
                    score += count as f32;
                }

                if score > 0.0 {
                    let normalized_score = score / (file.content.len() as f32).sqrt();
                    Some((file.clone(), normalized_score))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(max_results);
        results
    }

    async fn update_cache(&self, files: Vec<IndexedFile>) {
        if !self.config.enable_cache {
            return;
        }

        let mut cache = self.cache.write().await;
        cache.clear();
        for file in files {
            cache.insert(file.path.clone(), file);
        }
    }

    async fn get_files(&self) -> anyhow::Result<Vec<IndexedFile>> {
        if self.config.enable_cache {
            let cache = self.cache.read().await;
            if !cache.is_empty() {
                return Ok(cache.values().cloned().collect());
            }
        }

        let files = self.index_files().await?;
        self.update_cache(files.clone()).await;
        Ok(files)
    }
}

#[async_trait]
impl ContextProvider for FileSystemContextProvider {
    fn name(&self) -> &str {
        "filesystem"
    }

    async fn query(&self, query: &ContextQuery) -> anyhow::Result<ContextResult> {
        let files = self.get_files().await?;
        let results = self
            .search_simple(&query.query, &files, query.max_results)
            .await;

        let items: Vec<ContextItem> = results
            .into_iter()
            .map(|(file, score)| {
                let content = match query.depth {
                    crate::context::ContextDepth::Abstract => {
                        file.content.chars().take(500).collect::<String>()
                    }
                    crate::context::ContextDepth::Overview => {
                        file.content.chars().take(2000).collect::<String>()
                    }
                    crate::context::ContextDepth::Full => file.content.clone(),
                };

                let token_count = content.split_whitespace().count();

                ContextItem {
                    id: file.path.to_string_lossy().to_string(),
                    context_type: ContextType::Resource,
                    content,
                    token_count,
                    relevance: score,
                    source: Some(format!("file:{}", file.path.display())),
                    metadata: {
                        let mut meta = HashMap::new();
                        meta.insert(
                            "path".to_string(),
                            serde_json::Value::String(file.path.to_string_lossy().to_string()),
                        );
                        meta.insert(
                            "size".to_string(),
                            serde_json::Value::Number(file.size.into()),
                        );
                        meta
                    },
                }
            })
            .collect();

        let total_tokens: usize = items.iter().map(|item| item.token_count).sum();
        let truncated = items.len() < files.len();

        Ok(ContextResult {
            items,
            total_tokens,
            provider: self.name().to_string(),
            truncated,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_files(dir: &Path) -> anyhow::Result<()> {
        let mut file1 = File::create(dir.join("test1.rs"))?;
        writeln!(file1, "fn main() {{\n    println!(\"Hello, world!\");\n}}")?;

        let mut file2 = File::create(dir.join("test2.md"))?;
        writeln!(
            file2,
            "# Test Document\n\nThis is a test document about Rust programming."
        )?;

        fs::create_dir(dir.join("subdir"))?;
        let mut file4 = File::create(dir.join("subdir/test4.rs"))?;
        writeln!(file4, "fn test() {{\n    // Test function\n}}")?;

        Ok(())
    }

    #[tokio::test]
    async fn test_index_files() {
        let temp_dir = TempDir::new().unwrap();
        create_test_files(temp_dir.path()).unwrap();

        let config = FileSystemContextConfig::new(temp_dir.path());
        let provider = FileSystemContextProvider::new(config);

        let files = provider.index_files().await.unwrap();
        assert!(files.len() >= 2);
    }

    #[tokio::test]
    async fn test_search_simple() {
        let temp_dir = TempDir::new().unwrap();
        create_test_files(temp_dir.path()).unwrap();

        let config = FileSystemContextConfig::new(temp_dir.path());
        let provider = FileSystemContextProvider::new(config);

        let query = ContextQuery::new("Rust programming");
        let result = provider.query(&query).await.unwrap();

        assert!(!result.items.is_empty());
        assert!(result
            .items
            .iter()
            .any(|item| item.content.contains("Rust")));
    }
}
