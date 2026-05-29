use super::{SessionData, SessionStore};
use crate::loop_checkpoint::LoopCheckpoint;
use crate::run::RunRecord;
use crate::subagent_task_tracker::SubagentTaskSnapshot;
use crate::tools::ArtifactStore;
use crate::trace::TraceEvent;
use crate::verification::VerificationReport;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;

// ============================================================================
// File-based Session Store
// ============================================================================

/// File-based session store
///
/// Stores each session as a JSON file in a directory:
/// ```text
/// sessions/
///   session-1.json
///   session-2.json
/// ```
pub struct FileSessionStore {
    /// Directory to store session files
    pub(super) dir: PathBuf,
}

impl FileSessionStore {
    /// Create a new file session store
    ///
    /// Creates the directory if it doesn't exist.
    pub async fn new<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        fs::create_dir_all(&dir)
            .await
            .with_context(|| format!("Failed to create session directory: {}", dir.display()))?;

        Ok(Self { dir })
    }

    /// Get the file path for a session
    fn session_path(&self, id: &str) -> PathBuf {
        // Sanitize ID to prevent path traversal
        self.dir.join(format!("{}.json", safe_session_id(id)))
    }

    fn artifact_dir(&self, id: &str) -> PathBuf {
        self.dir.join("artifacts").join(safe_session_id(id))
    }

    fn trace_path(&self, id: &str) -> PathBuf {
        self.dir
            .join("traces")
            .join(format!("{}.json", safe_session_id(id)))
    }

    fn verification_path(&self, id: &str) -> PathBuf {
        self.dir
            .join("verification")
            .join(format!("{}.json", safe_session_id(id)))
    }

    fn runs_path(&self, id: &str) -> PathBuf {
        self.dir
            .join("runs")
            .join(format!("{}.json", safe_session_id(id)))
    }

    fn subagent_tasks_path(&self, id: &str) -> PathBuf {
        self.dir
            .join("subagent_tasks")
            .join(format!("{}.json", safe_session_id(id)))
    }

    fn loop_checkpoint_path(&self, run_id: &str) -> PathBuf {
        self.dir
            .join("loop_checkpoints")
            .join(format!("{}.json", safe_session_id(run_id)))
    }
}

fn safe_session_id(id: &str) -> String {
    id.replace(['/', '\\'], "_").replace("..", "_")
}

#[async_trait::async_trait]
impl SessionStore for FileSessionStore {
    async fn save(&self, session: &SessionData) -> Result<()> {
        let path = self.session_path(&session.id);

        // Serialize to JSON with pretty printing for readability
        let json = serde_json::to_string_pretty(session)
            .with_context(|| format!("Failed to serialize session: {}", session.id))?;

        // Write atomically: write to temp file with unique name, then rename
        // Use timestamp + process ID to ensure uniqueness for concurrent saves
        let unique_suffix = format!(
            "{}.{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            std::process::id()
        );
        let temp_path = path.with_extension(format!("json.{}.tmp", unique_suffix));

        let mut file = fs::File::create(&temp_path)
            .await
            .with_context(|| format!("Failed to create temp file: {}", temp_path.display()))?;

        file.write_all(json.as_bytes())
            .await
            .with_context(|| format!("Failed to write session data: {}", session.id))?;

        file.sync_all()
            .await
            .with_context(|| format!("Failed to sync session file: {}", session.id))?;

        // Rename temp file to final path (atomic on most filesystems)
        fs::rename(&temp_path, &path)
            .await
            .with_context(|| format!("Failed to rename session file: {}", session.id))?;

        tracing::debug!("Saved session {} to {}", session.id, path.display());
        Ok(())
    }

    async fn load(&self, id: &str) -> Result<Option<SessionData>> {
        let path = self.session_path(id);

        if !path.exists() {
            return Ok(None);
        }

        let json = fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read session file: {}", path.display()))?;

        let session: SessionData = serde_json::from_str(&json)
            .with_context(|| format!("Failed to parse session file: {}", path.display()))?;

        tracing::debug!("Loaded session {} from {}", id, path.display());
        Ok(Some(session))
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let path = self.session_path(id);

        if path.exists() {
            fs::remove_file(&path)
                .await
                .with_context(|| format!("Failed to delete session file: {}", path.display()))?;

            tracing::debug!("Deleted session {} from {}", id, path.display());
        }

        let artifact_dir = self.artifact_dir(id);
        if artifact_dir.exists() {
            fs::remove_dir_all(&artifact_dir).await.with_context(|| {
                format!(
                    "Failed to delete artifact directory for session {}: {}",
                    id,
                    artifact_dir.display()
                )
            })?;
        }

        let trace_path = self.trace_path(id);
        if trace_path.exists() {
            fs::remove_file(&trace_path).await.with_context(|| {
                format!(
                    "Failed to delete trace file for session {}: {}",
                    id,
                    trace_path.display()
                )
            })?;
        }

        let verification_path = self.verification_path(id);
        if verification_path.exists() {
            fs::remove_file(&verification_path).await.with_context(|| {
                format!(
                    "Failed to delete verification report file for session {}: {}",
                    id,
                    verification_path.display()
                )
            })?;
        }

        let runs_path = self.runs_path(id);
        if runs_path.exists() {
            fs::remove_file(&runs_path).await.with_context(|| {
                format!(
                    "Failed to delete run record file for session {}: {}",
                    id,
                    runs_path.display()
                )
            })?;
        }

        let subagent_tasks_path = self.subagent_tasks_path(id);
        if subagent_tasks_path.exists() {
            fs::remove_file(&subagent_tasks_path)
                .await
                .with_context(|| {
                    format!(
                        "Failed to delete subagent task file for session {}: {}",
                        id,
                        subagent_tasks_path.display()
                    )
                })?;
        }

        Ok(())
    }

    async fn list(&self) -> Result<Vec<String>> {
        let mut session_ids = Vec::new();

        let mut entries = fs::read_dir(&self.dir)
            .await
            .with_context(|| format!("Failed to read session directory: {}", self.dir.display()))?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            if path.extension().is_some_and(|ext| ext == "json") {
                if let Some(stem) = path.file_stem() {
                    if let Some(id) = stem.to_str() {
                        session_ids.push(id.to_string());
                    }
                }
            }
        }

        Ok(session_ids)
    }

    async fn exists(&self, id: &str) -> Result<bool> {
        let path = self.session_path(id);
        Ok(path.exists())
    }

    async fn save_artifacts(&self, id: &str, artifacts: &ArtifactStore) -> Result<()> {
        let artifact_dir = self.artifact_dir(id);
        artifacts.save_to_dir(&artifact_dir).with_context(|| {
            format!(
                "Failed to save artifacts for session {} to {}",
                id,
                artifact_dir.display()
            )
        })
    }

    async fn load_artifacts(&self, id: &str) -> Result<Option<ArtifactStore>> {
        let artifact_dir = self.artifact_dir(id);
        if !artifact_dir.exists() {
            return Ok(None);
        }

        let artifacts = ArtifactStore::load_from_dir(&artifact_dir).with_context(|| {
            format!(
                "Failed to load artifacts for session {} from {}",
                id,
                artifact_dir.display()
            )
        })?;
        Ok(Some(artifacts))
    }

    async fn save_trace_events(&self, id: &str, events: &[TraceEvent]) -> Result<()> {
        let path = self.trace_path(id);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await.with_context(|| {
                format!("Failed to create trace directory: {}", parent.display())
            })?;
        }

        let json = serde_json::to_string_pretty(events)
            .with_context(|| format!("Failed to serialize trace events for session {id}"))?;
        fs::write(&path, json)
            .await
            .with_context(|| format!("Failed to write trace events to {}", path.display()))?;
        Ok(())
    }

    async fn load_trace_events(&self, id: &str) -> Result<Option<Vec<TraceEvent>>> {
        let path = self.trace_path(id);
        if !path.exists() {
            return Ok(None);
        }

        let json = fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read trace events from {}", path.display()))?;
        let events = serde_json::from_str(&json)
            .with_context(|| format!("Failed to parse trace events from {}", path.display()))?;
        Ok(Some(events))
    }

    async fn save_run_records(&self, id: &str, records: &[RunRecord]) -> Result<()> {
        let path = self.runs_path(id);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .await
                .with_context(|| format!("Failed to create run directory: {}", parent.display()))?;
        }

        let json = serde_json::to_string_pretty(records)
            .with_context(|| format!("Failed to serialize run records for session {id}"))?;
        fs::write(&path, json)
            .await
            .with_context(|| format!("Failed to write run records to {}", path.display()))?;
        Ok(())
    }

    async fn load_run_records(&self, id: &str) -> Result<Option<Vec<RunRecord>>> {
        let path = self.runs_path(id);
        if !path.exists() {
            return Ok(None);
        }

        let json = fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read run records from {}", path.display()))?;
        let records = serde_json::from_str(&json)
            .with_context(|| format!("Failed to parse run records from {}", path.display()))?;
        Ok(Some(records))
    }

    async fn save_verification_reports(
        &self,
        id: &str,
        reports: &[VerificationReport],
    ) -> Result<()> {
        let path = self.verification_path(id);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await.with_context(|| {
                format!(
                    "Failed to create verification report directory: {}",
                    parent.display()
                )
            })?;
        }

        let json = serde_json::to_string_pretty(reports).with_context(|| {
            format!("Failed to serialize verification reports for session {id}")
        })?;
        fs::write(&path, json).await.with_context(|| {
            format!("Failed to write verification reports to {}", path.display())
        })?;
        Ok(())
    }

    async fn load_verification_reports(&self, id: &str) -> Result<Option<Vec<VerificationReport>>> {
        let path = self.verification_path(id);
        if !path.exists() {
            return Ok(None);
        }

        let json = fs::read_to_string(&path).await.with_context(|| {
            format!(
                "Failed to read verification reports from {}",
                path.display()
            )
        })?;
        let reports = serde_json::from_str(&json).with_context(|| {
            format!(
                "Failed to parse verification reports from {}",
                path.display()
            )
        })?;
        Ok(Some(reports))
    }

    async fn save_subagent_tasks(&self, id: &str, tasks: &[SubagentTaskSnapshot]) -> Result<()> {
        let path = self.subagent_tasks_path(id);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await.with_context(|| {
                format!(
                    "Failed to create subagent task directory: {}",
                    parent.display()
                )
            })?;
        }

        let json = serde_json::to_string_pretty(tasks)
            .with_context(|| format!("Failed to serialize subagent tasks for session {id}"))?;
        fs::write(&path, json)
            .await
            .with_context(|| format!("Failed to write subagent tasks to {}", path.display()))?;
        Ok(())
    }

    async fn load_subagent_tasks(&self, id: &str) -> Result<Option<Vec<SubagentTaskSnapshot>>> {
        let path = self.subagent_tasks_path(id);
        if !path.exists() {
            return Ok(None);
        }
        let json = fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read subagent tasks from {}", path.display()))?;
        let tasks = serde_json::from_str(&json)
            .with_context(|| format!("Failed to parse subagent tasks from {}", path.display()))?;
        Ok(Some(tasks))
    }

    async fn save_loop_checkpoint(&self, run_id: &str, checkpoint: &LoopCheckpoint) -> Result<()> {
        let path = self.loop_checkpoint_path(run_id);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await.with_context(|| {
                format!(
                    "Failed to create loop checkpoint directory: {}",
                    parent.display()
                )
            })?;
        }
        let json = serde_json::to_string_pretty(checkpoint)
            .with_context(|| format!("Failed to serialize loop checkpoint for run {run_id}"))?;

        // Crash-atomic write: a checkpoint exists precisely to survive a
        // process crash, so the write itself must be crash-safe. A plain
        // `fs::write` can leave a truncated JSON file if the process dies
        // mid-write — which `resume_run` would then fail to parse,
        // defeating the whole point. Write to a unique temp file, fsync,
        // then atomically rename over the target.
        let unique_suffix = format!(
            "{}.{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
            std::process::id()
        );
        let temp_path = path.with_extension(format!("json.{}.tmp", unique_suffix));
        let mut file = fs::File::create(&temp_path).await.with_context(|| {
            format!(
                "Failed to create checkpoint temp file: {}",
                temp_path.display()
            )
        })?;
        file.write_all(json.as_bytes())
            .await
            .with_context(|| format!("Failed to write loop checkpoint for run {run_id}"))?;
        file.sync_all()
            .await
            .with_context(|| format!("Failed to fsync loop checkpoint for run {run_id}"))?;
        fs::rename(&temp_path, &path).await.with_context(|| {
            format!(
                "Failed to rename loop checkpoint into place: {}",
                path.display()
            )
        })?;
        Ok(())
    }

    async fn load_loop_checkpoint(&self, run_id: &str) -> Result<Option<LoopCheckpoint>> {
        let path = self.loop_checkpoint_path(run_id);
        if !path.exists() {
            return Ok(None);
        }
        let json = fs::read_to_string(&path)
            .await
            .with_context(|| format!("Failed to read loop checkpoint from {}", path.display()))?;
        let checkpoint: LoopCheckpoint = serde_json::from_str(&json)
            .with_context(|| format!("Failed to parse loop checkpoint from {}", path.display()))?;
        // Refuse to hand back a checkpoint written by a future, incompatible
        // schema version — its field semantics may differ (see the contract
        // on LoopCheckpoint::ensure_loadable).
        checkpoint.ensure_loadable()?;
        Ok(Some(checkpoint))
    }

    async fn delete_loop_checkpoint(&self, run_id: &str) -> Result<()> {
        let path = self.loop_checkpoint_path(run_id);
        if path.exists() {
            fs::remove_file(&path).await.with_context(|| {
                format!(
                    "Failed to delete loop checkpoint for run {}: {}",
                    run_id,
                    path.display()
                )
            })?;
        }
        Ok(())
    }

    async fn health_check(&self) -> Result<()> {
        // Verify directory exists and is writable
        let probe = self.dir.join(".health_check");
        fs::write(&probe, b"ok")
            .await
            .with_context(|| format!("Store directory not writable: {}", self.dir.display()))?;
        let _ = fs::remove_file(&probe).await;
        Ok(())
    }

    fn backend_name(&self) -> &str {
        "file"
    }
}
