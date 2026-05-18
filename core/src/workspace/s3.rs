//! S3-compatible object-storage workspace backend.
//!
//! [`S3WorkspaceBackend`] implements [`WorkspaceFileSystem`] against any
//! S3-compatible endpoint (AWS S3, MinIO, Cloudflare R2, Backblaze B2, ...)
//! using the AWS Rust SDK. The backend deliberately does **not** implement
//! [`WorkspaceCommandRunner`], [`WorkspaceSearch`], or any of the git
//! provider traits — object storage cannot natively service those operations,
//! and capability gating prevents the corresponding tools (`bash`, `grep`,
//! `glob`, `git`) from being registered when the backend is in use.
//!
//! Path semantics are lexical (no host filesystem involved), inherited from
//! [`super::VirtualPathResolver`]: paths are relative, parent-directory
//! traversal is rejected, and absolute or Windows-style paths are refused.
//!
//! # Concurrency caveats
//!
//! S3 does not provide atomic rename or read-modify-write. Tools like `edit`
//! and `patch` perform a `read_text` then `write_text` — concurrent writers
//! to the same key will overwrite each other (last-writer-wins). Callers
//! that need stronger guarantees should partition workspaces per session.
//!
//! Available only when the `s3` feature is enabled.

use super::{
    WorkspaceDirEntry, WorkspaceFileSystem, WorkspaceFileType, WorkspacePath, WorkspaceWriteOutcome,
};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use aws_credential_types::Credentials;
use aws_sdk_s3::config::{BehaviorVersion, Region};
use aws_sdk_s3::error::SdkError;
use aws_sdk_s3::operation::list_objects_v2::ListObjectsV2Error;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::Client;
use std::sync::Arc;
use std::time::Duration;

const DEFAULT_REGION: &str = "us-east-1";

/// Configuration for an [`S3WorkspaceBackend`].
///
/// `endpoint` is optional: omit it to use the AWS default. Set it to point at
/// MinIO, RustFS, R2, or any other S3-compatible service.
///
/// `prefix` is the logical workspace root inside the bucket — every workspace
/// path becomes `<prefix>/<path>` when sent to S3. An empty prefix means the
/// bucket root itself acts as the workspace.
#[derive(Debug, Clone)]
pub struct S3BackendConfig {
    pub endpoint: Option<String>,
    pub region: Option<String>,
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
    pub bucket: String,
    pub prefix: String,
    /// `true` for MinIO / RustFS / most non-AWS endpoints, `false` for AWS S3.
    pub force_path_style: bool,
    /// Per-operation request timeout. Defaults to 30 seconds. Independent of
    /// the workspace-level `operation_timeout` set on [`super::WorkspaceServices`];
    /// whichever fires first wins.
    pub request_timeout: Option<Duration>,
}

impl S3BackendConfig {
    pub fn new(
        bucket: impl Into<String>,
        prefix: impl Into<String>,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: None,
            region: None,
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
            session_token: None,
            bucket: bucket.into(),
            prefix: prefix.into(),
            force_path_style: false,
            request_timeout: None,
        }
    }

    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    pub fn session_token(mut self, token: impl Into<String>) -> Self {
        self.session_token = Some(token.into());
        self
    }

    pub fn force_path_style(mut self, enabled: bool) -> Self {
        self.force_path_style = enabled;
        self
    }

    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = Some(timeout);
        self
    }
}

/// S3-compatible workspace backend.
///
/// Construct with [`Self::new`] for production, or [`Self::with_client`] for
/// tests that need to inject a pre-built [`aws_sdk_s3::Client`] (e.g. with a
/// mock HTTP layer).
#[derive(Debug, Clone)]
pub struct S3WorkspaceBackend {
    client: Client,
    bucket: String,
    /// Normalised prefix without trailing slash. Empty string means
    /// "bucket root is the workspace".
    prefix: String,
}

impl S3WorkspaceBackend {
    /// Build a backend from declarative configuration.
    pub fn new(config: S3BackendConfig) -> Self {
        let credentials = Credentials::new(
            config.access_key_id,
            config.secret_access_key,
            config.session_token,
            None,
            "a3s-code-static",
        );

        let mut builder = aws_sdk_s3::Config::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new(
                config.region.unwrap_or_else(|| DEFAULT_REGION.to_string()),
            ))
            .credentials_provider(credentials)
            .force_path_style(config.force_path_style);

        if let Some(endpoint) = config.endpoint {
            builder = builder.endpoint_url(endpoint);
        }

        let client = Client::from_conf(builder.build());
        Self::with_client(client, config.bucket, config.prefix)
    }

    /// Build a backend from a pre-configured S3 client. Intended for tests
    /// and advanced use cases (custom retries, signer overrides, http_client
    /// injection, etc.).
    pub fn with_client(
        client: Client,
        bucket: impl Into<String>,
        prefix: impl Into<String>,
    ) -> Self {
        Self {
            client,
            bucket: bucket.into(),
            prefix: normalize_prefix(&prefix.into()),
        }
    }

    /// The bucket this backend is bound to.
    pub fn bucket(&self) -> &str {
        &self.bucket
    }

    /// The workspace prefix inside the bucket (no leading or trailing slash).
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Underlying AWS SDK client — exposed for advanced workflows that need
    /// to perform out-of-band operations (e.g. presigned URLs, ACL changes).
    pub fn client(&self) -> &Client {
        &self.client
    }

    fn key_for(&self, path: &WorkspacePath) -> String {
        if path.is_root() {
            self.prefix.clone()
        } else if self.prefix.is_empty() {
            path.as_str().to_string()
        } else {
            format!("{}/{}", self.prefix, path.as_str())
        }
    }

    fn list_prefix_for(&self, path: &WorkspacePath) -> String {
        if path.is_root() {
            if self.prefix.is_empty() {
                String::new()
            } else {
                format!("{}/", self.prefix)
            }
        } else if self.prefix.is_empty() {
            format!("{}/", path.as_str())
        } else {
            format!("{}/{}/", self.prefix, path.as_str())
        }
    }
}

#[async_trait]
impl WorkspaceFileSystem for S3WorkspaceBackend {
    async fn read_text(&self, path: &WorkspacePath) -> Result<String> {
        let key = self.key_for(path);
        let resp = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(&key)
            .send()
            .await
            .map_err(|e| classify_get_error(&self.bucket, &key, e))?;

        let bytes = resp
            .body
            .collect()
            .await
            .map_err(|e| {
                anyhow!(
                    "Failed to read S3 object body s3://{}/{}: {}",
                    self.bucket,
                    key,
                    e
                )
            })?
            .into_bytes();

        String::from_utf8(bytes.to_vec()).map_err(|e| {
            anyhow!(
                "S3 object s3://{}/{} is not valid UTF-8: {}",
                self.bucket,
                key,
                e
            )
        })
    }

    async fn write_text(
        &self,
        path: &WorkspacePath,
        content: &str,
    ) -> Result<WorkspaceWriteOutcome> {
        let key = self.key_for(path);
        let body = ByteStream::from(content.as_bytes().to_vec());

        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(&key)
            .body(body)
            .content_type("text/plain; charset=utf-8")
            .send()
            .await
            .map_err(|e| {
                anyhow!(
                    "Failed to write S3 object s3://{}/{}: {}",
                    self.bucket,
                    key,
                    e
                )
            })?;

        Ok(WorkspaceWriteOutcome {
            bytes: content.len(),
            lines: content.lines().count(),
        })
    }

    async fn list_dir(&self, path: &WorkspacePath) -> Result<Vec<WorkspaceDirEntry>> {
        let prefix = self.list_prefix_for(path);
        let mut entries: Vec<WorkspaceDirEntry> = Vec::new();
        let mut continuation: Option<String> = None;

        loop {
            let mut req = self
                .client
                .list_objects_v2()
                .bucket(&self.bucket)
                .prefix(&prefix)
                .delimiter("/");
            if let Some(token) = continuation.as_ref() {
                req = req.continuation_token(token);
            }

            let resp = req
                .send()
                .await
                .map_err(|e| classify_list_error(&self.bucket, &prefix, e))?;

            // CommonPrefixes → directories
            for cp in resp.common_prefixes() {
                if let Some(p) = cp.prefix() {
                    // p looks like "<prefix><name>/"; extract <name>
                    if let Some(name) = strip_dir_name(p, &prefix) {
                        entries.push(WorkspaceDirEntry {
                            name,
                            kind: WorkspaceFileType::Directory,
                            size: 0,
                        });
                    }
                }
            }

            // Contents → files
            for obj in resp.contents() {
                let Some(key) = obj.key() else { continue };
                // Skip the prefix marker itself (key == prefix exactly).
                if key == prefix {
                    continue;
                }
                if let Some(name) = strip_file_name(key, &prefix) {
                    entries.push(WorkspaceDirEntry {
                        name,
                        kind: WorkspaceFileType::File,
                        size: obj.size().unwrap_or(0).max(0) as u64,
                    });
                }
            }

            if resp.is_truncated().unwrap_or(false) {
                continuation = resp.next_continuation_token().map(|s| s.to_string());
                if continuation.is_none() {
                    break;
                }
            } else {
                break;
            }
        }

        Ok(entries)
    }
}

fn normalize_prefix(prefix: &str) -> String {
    prefix
        .trim_start_matches('/')
        .trim_end_matches('/')
        .to_string()
}

fn strip_dir_name(common_prefix: &str, listing_prefix: &str) -> Option<String> {
    let remainder = common_prefix.strip_prefix(listing_prefix)?;
    let trimmed = remainder.trim_end_matches('/');
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn strip_file_name(key: &str, listing_prefix: &str) -> Option<String> {
    let remainder = key.strip_prefix(listing_prefix)?;
    if remainder.is_empty() || remainder.contains('/') {
        None
    } else {
        Some(remainder.to_string())
    }
}

fn classify_get_error<E>(bucket: &str, key: &str, error: SdkError<E>) -> anyhow::Error
where
    E: std::error::Error + Send + Sync + 'static,
{
    let raw = error
        .raw_response()
        .map(|r| r.status().as_u16())
        .unwrap_or_default();
    if raw == 404 {
        anyhow!("S3 object not found: s3://{}/{}", bucket, key)
    } else {
        anyhow!(
            "Failed to read S3 object s3://{}/{}: {}",
            bucket,
            key,
            error
        )
    }
}

fn classify_list_error(
    bucket: &str,
    prefix: &str,
    error: SdkError<ListObjectsV2Error>,
) -> anyhow::Error {
    anyhow!(
        "Failed to list S3 prefix s3://{}/{}: {}",
        bucket,
        prefix,
        error
    )
}

impl super::WorkspaceServices {
    /// Build a workspace whose files live in an S3-compatible bucket.
    ///
    /// The resulting [`WorkspaceServices`](super::WorkspaceServices) exposes
    /// only read / write / list capabilities (`read`, `write`, `edit`,
    /// `patch`, `ls`); `bash`, `git`, `grep`, and `glob` are intentionally
    /// not registered because object storage cannot service them. A 60s
    /// per-operation timeout is applied by default — override via
    /// [`super::WorkspaceServicesBuilder::operation_timeout`] when building
    /// manually.
    pub fn s3(config: S3BackendConfig) -> Arc<Self> {
        let backend = Arc::new(S3WorkspaceBackend::new(config));
        Self::from_s3_backend(backend)
    }

    /// Build a workspace from a pre-constructed [`S3WorkspaceBackend`].
    ///
    /// Useful when the caller has injected a custom AWS client (e.g. a mocked
    /// HTTP layer, alternative credential provider, or a wrapper that adds
    /// metrics / tracing).
    pub fn from_s3_backend(backend: Arc<S3WorkspaceBackend>) -> Arc<Self> {
        let workspace_ref = super::WorkspaceRef::new(
            format!("s3://{}/{}", backend.bucket(), backend.prefix()),
            format!("s3://{}/{}", backend.bucket(), backend.prefix()),
        );
        let fs: Arc<dyn WorkspaceFileSystem> = backend;
        Self::builder(workspace_ref, fs)
            .operation_timeout(Duration::from_secs(60))
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_for_root_uses_prefix_only() {
        let backend = make_backend("ws/u1/s1");
        let key = backend.key_for(&WorkspacePath::root());
        assert_eq!(key, "ws/u1/s1");
    }

    #[test]
    fn key_for_nested_path_joins_with_slash() {
        let backend = make_backend("ws/u1/s1");
        let key = backend.key_for(&WorkspacePath::from_normalized("src/main.rs"));
        assert_eq!(key, "ws/u1/s1/src/main.rs");
    }

    #[test]
    fn key_for_empty_prefix_uses_path_only() {
        let backend = make_backend("");
        assert_eq!(
            backend.key_for(&WorkspacePath::from_normalized("notes.txt")),
            "notes.txt"
        );
        assert_eq!(backend.key_for(&WorkspacePath::root()), "");
    }

    #[test]
    fn list_prefix_root_with_workspace_prefix() {
        let backend = make_backend("ws/u1/s1");
        assert_eq!(backend.list_prefix_for(&WorkspacePath::root()), "ws/u1/s1/");
    }

    #[test]
    fn list_prefix_root_with_empty_workspace_prefix() {
        let backend = make_backend("");
        assert_eq!(backend.list_prefix_for(&WorkspacePath::root()), "");
    }

    #[test]
    fn list_prefix_nested_path() {
        let backend = make_backend("ws/u1/s1");
        let path = WorkspacePath::from_normalized("src");
        assert_eq!(backend.list_prefix_for(&path), "ws/u1/s1/src/");
    }

    #[test]
    fn normalize_prefix_strips_slashes() {
        assert_eq!(normalize_prefix("/foo/bar/"), "foo/bar");
        assert_eq!(normalize_prefix("foo"), "foo");
        assert_eq!(normalize_prefix(""), "");
        assert_eq!(normalize_prefix("/"), "");
    }

    #[test]
    fn strip_dir_name_extracts_immediate_child() {
        assert_eq!(
            strip_dir_name("ws/u1/s1/src/", "ws/u1/s1/"),
            Some("src".to_string())
        );
        assert_eq!(strip_dir_name("ws/u1/s1/", "ws/u1/s1/"), None);
        assert_eq!(strip_dir_name("other/", "ws/u1/s1/"), None);
    }

    #[test]
    fn strip_file_name_rejects_nested_keys() {
        assert_eq!(
            strip_file_name("ws/u1/s1/notes.txt", "ws/u1/s1/"),
            Some("notes.txt".to_string())
        );
        // Nested key — should be claimed by a deeper LIST instead.
        assert_eq!(strip_file_name("ws/u1/s1/src/main.rs", "ws/u1/s1/"), None);
        assert_eq!(strip_file_name("other/notes.txt", "ws/u1/s1/"), None);
    }

    #[test]
    fn config_builder_sets_fields() {
        let cfg = S3BackendConfig::new("bucket", "prefix", "AK", "SK")
            .endpoint("https://minio.local:9000")
            .region("cn-east-1")
            .session_token("TOKEN")
            .force_path_style(true)
            .request_timeout(Duration::from_secs(5));
        assert_eq!(cfg.bucket, "bucket");
        assert_eq!(cfg.prefix, "prefix");
        assert_eq!(cfg.endpoint.as_deref(), Some("https://minio.local:9000"));
        assert_eq!(cfg.region.as_deref(), Some("cn-east-1"));
        assert_eq!(cfg.session_token.as_deref(), Some("TOKEN"));
        assert!(cfg.force_path_style);
        assert_eq!(cfg.request_timeout, Some(Duration::from_secs(5)));
    }

    #[test]
    fn services_s3_factory_disables_exec_search_and_git() {
        let cfg = S3BackendConfig::new("bucket", "ws", "AK", "SK");
        let services = super::super::WorkspaceServices::s3(cfg);
        let caps = services.capabilities();
        assert!(caps.read);
        assert!(caps.write);
        assert!(!caps.exec);
        assert!(!caps.search);
        assert!(!caps.git);
        assert!(services.command_runner().is_none());
        assert!(services.search().is_none());
        assert!(services.git().is_none());
        assert!(services.git_stash().is_none());
        assert!(services.git_worktree().is_none());
        assert_eq!(services.operation_timeout(), Some(Duration::from_secs(60)));
    }

    fn make_backend(prefix: &str) -> S3WorkspaceBackend {
        let cfg = S3BackendConfig::new("bucket", prefix, "AK", "SK");
        S3WorkspaceBackend::new(cfg)
    }
}
