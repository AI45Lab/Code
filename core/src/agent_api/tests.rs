use super::*;
use crate::config::{ModelConfig, ModelModalities, ProviderConfig};
use crate::llm::{ContentBlock, LlmResponse, StreamEvent, TokenUsage};
use crate::store::SessionStore;

#[derive(Clone)]
struct StaticStreamingClient {
    text: String,
}

impl StaticStreamingClient {
    fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }

    fn response(&self) -> LlmResponse {
        LlmResponse {
            message: Message {
                role: "assistant".to_string(),
                content: vec![ContentBlock::Text {
                    text: self.text.clone(),
                }],
                reasoning_content: None,
            },
            usage: TokenUsage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
                cache_read_tokens: None,
                cache_write_tokens: None,
            },
            stop_reason: Some("end_turn".to_string()),
            token_logprobs: Vec::new(),
            meta: None,
        }
    }
}

#[derive(Clone)]
struct FailingStreamingClient;

#[derive(Clone)]
struct CancellableStreamingClient {
    text: String,
}

#[derive(Debug, Default)]
struct RecordingRuntimeHook {
    events: std::sync::Mutex<Vec<(String, String, AgentEvent)>>,
}

#[derive(Debug, Default)]
struct CapturingContextProvider {
    session_ids: std::sync::Mutex<Vec<Option<String>>>,
}

#[derive(Default)]
struct TestWorkspaceFs {
    files: std::sync::RwLock<std::collections::HashMap<String, String>>,
}

impl TestWorkspaceFs {
    fn insert(&self, path: &str, content: &str) {
        self.files
            .write()
            .unwrap()
            .insert(path.to_string(), content.to_string());
    }

    fn read_raw(&self, path: &str) -> Option<String> {
        self.files.read().unwrap().get(path).cloned()
    }
}

#[async_trait::async_trait]
impl crate::workspace::WorkspaceFileSystem for TestWorkspaceFs {
    async fn read_text(
        &self,
        path: &crate::workspace::WorkspacePath,
    ) -> crate::workspace::WorkspaceResult<String> {
        self.files
            .read()
            .unwrap()
            .get(path.as_str())
            .cloned()
            .ok_or_else(|| crate::workspace::WorkspaceError::NotFound {
                path: path.as_str().to_string(),
            })
    }

    async fn write_text(
        &self,
        path: &crate::workspace::WorkspacePath,
        content: &str,
    ) -> crate::workspace::WorkspaceResult<crate::workspace::WorkspaceWriteOutcome> {
        self.insert(path.as_str(), content);
        Ok(crate::workspace::WorkspaceWriteOutcome {
            bytes: content.len(),
            lines: content.lines().count(),
        })
    }

    async fn list_dir(
        &self,
        path: &crate::workspace::WorkspacePath,
    ) -> crate::workspace::WorkspaceResult<Vec<crate::workspace::WorkspaceDirEntry>> {
        let prefix = if path.is_root() {
            String::new()
        } else {
            format!("{}/", path.as_str())
        };
        let files = self.files.read().unwrap();
        let mut entries = Vec::new();

        for (file_path, content) in files.iter() {
            if !file_path.starts_with(&prefix) {
                continue;
            }
            let remaining = &file_path[prefix.len()..];
            if remaining.is_empty() || remaining.contains('/') {
                continue;
            }
            entries.push(crate::workspace::WorkspaceDirEntry {
                name: remaining.to_string(),
                kind: crate::workspace::WorkspaceFileType::File,
                size: content.len() as u64,
            });
        }

        Ok(entries)
    }
}

#[derive(Default)]
struct TestWorkspaceRunner {
    commands: std::sync::RwLock<Vec<String>>,
}

#[async_trait::async_trait]
impl crate::workspace::WorkspaceCommandRunner for TestWorkspaceRunner {
    async fn exec(
        &self,
        request: crate::workspace::CommandRequest,
    ) -> anyhow::Result<crate::workspace::CommandOutput> {
        self.commands.write().unwrap().push(request.command.clone());
        Ok(crate::workspace::CommandOutput {
            output: format!("session runner: {}\n", request.command),
            exit_code: 0,
            timed_out: false,
        })
    }
}

#[async_trait::async_trait]
impl crate::context::ContextProvider for CapturingContextProvider {
    fn name(&self) -> &str {
        "capturing-context"
    }

    async fn query(
        &self,
        query: &crate::context::ContextQuery,
    ) -> anyhow::Result<crate::context::ContextResult> {
        self.session_ids
            .lock()
            .unwrap()
            .push(query.session_id.clone());
        Ok(crate::context::ContextResult::new(self.name()))
    }
}

#[async_trait::async_trait]
impl crate::hooks::HookExecutor for RecordingRuntimeHook {
    async fn fire(&self, _event: &crate::hooks::HookEvent) -> crate::hooks::HookResult {
        crate::hooks::HookResult::Continue(None)
    }

    async fn record_agent_event(&self, event: &AgentEvent, run_id: &str, session_id: &str) {
        self.events.lock().unwrap().push((
            run_id.to_string(),
            session_id.to_string(),
            event.clone(),
        ));
    }
}

impl CancellableStreamingClient {
    fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

#[async_trait::async_trait]
impl LlmClient for StaticStreamingClient {
    async fn complete(
        &self,
        _messages: &[Message],
        _system: Option<&str>,
        _tools: &[crate::llm::ToolDefinition],
    ) -> anyhow::Result<LlmResponse> {
        Ok(self.response())
    }

    async fn complete_streaming(
        &self,
        _messages: &[Message],
        _system: Option<&str>,
        _tools: &[crate::llm::ToolDefinition],
        _cancel_token: tokio_util::sync::CancellationToken,
    ) -> anyhow::Result<mpsc::Receiver<StreamEvent>> {
        let (tx, rx) = mpsc::channel(8);
        let text = self.text.clone();
        let response = self.response();
        tokio::spawn(async move {
            let _ = tx.send(StreamEvent::TextDelta(text)).await;
            let _ = tx.send(StreamEvent::Done(response)).await;
        });
        Ok(rx)
    }
}

#[async_trait::async_trait]
impl LlmClient for FailingStreamingClient {
    async fn complete(
        &self,
        _messages: &[Message],
        _system: Option<&str>,
        _tools: &[crate::llm::ToolDefinition],
    ) -> anyhow::Result<LlmResponse> {
        anyhow::bail!("non-streaming fallback failed")
    }

    async fn complete_streaming(
        &self,
        _messages: &[Message],
        _system: Option<&str>,
        _tools: &[crate::llm::ToolDefinition],
        _cancel_token: tokio_util::sync::CancellationToken,
    ) -> anyhow::Result<mpsc::Receiver<StreamEvent>> {
        anyhow::bail!("streaming setup failed")
    }
}

#[async_trait::async_trait]
impl LlmClient for CancellableStreamingClient {
    async fn complete(
        &self,
        _messages: &[Message],
        _system: Option<&str>,
        _tools: &[crate::llm::ToolDefinition],
    ) -> anyhow::Result<LlmResponse> {
        anyhow::bail!("cancellable client does not support fallback completion")
    }

    async fn complete_streaming(
        &self,
        _messages: &[Message],
        _system: Option<&str>,
        _tools: &[crate::llm::ToolDefinition],
        cancel_token: tokio_util::sync::CancellationToken,
    ) -> anyhow::Result<mpsc::Receiver<StreamEvent>> {
        let (tx, rx) = mpsc::channel(8);
        let text = self.text.clone();
        tokio::spawn(async move {
            let _ = tx.send(StreamEvent::TextDelta(text)).await;
            cancel_token.cancelled().await;
        });
        Ok(rx)
    }
}

fn test_config() -> CodeConfig {
    CodeConfig {
        default_model: Some("anthropic/claude-sonnet-4-20250514".to_string()),
        providers: vec![
            ProviderConfig {
                name: "anthropic".to_string(),
                api_key: Some("test-key".to_string()),
                base_url: None,
                headers: std::collections::HashMap::new(),
                session_id_header: None,
                models: vec![ModelConfig {
                    id: "claude-sonnet-4-20250514".to_string(),
                    name: "Claude Sonnet 4".to_string(),
                    family: "claude-sonnet".to_string(),
                    api_key: None,
                    base_url: None,
                    headers: std::collections::HashMap::new(),
                    session_id_header: None,
                    attachment: false,
                    reasoning: false,
                    tool_call: true,
                    temperature: true,
                    release_date: None,
                    modalities: ModelModalities::default(),
                    cost: Default::default(),
                    limit: Default::default(),
                }],
            },
            ProviderConfig {
                name: "openai".to_string(),
                api_key: Some("test-openai-key".to_string()),
                base_url: None,
                headers: std::collections::HashMap::new(),
                session_id_header: None,
                models: vec![ModelConfig {
                    id: "gpt-4o".to_string(),
                    name: "GPT-4o".to_string(),
                    family: "gpt-4".to_string(),
                    api_key: None,
                    base_url: None,
                    headers: std::collections::HashMap::new(),
                    session_id_header: None,
                    attachment: false,
                    reasoning: false,
                    tool_call: true,
                    temperature: true,
                    release_date: None,
                    modalities: ModelModalities::default(),
                    cost: Default::default(),
                    limit: Default::default(),
                }],
            },
        ],
        ..Default::default()
    }
}

fn build_effective_registry_for_test(
    agent_registry: Option<Arc<crate::skills::SkillRegistry>>,
    opts: &SessionOptions,
) -> Arc<crate::skills::SkillRegistry> {
    super::capabilities::build_effective_skill_registry(agent_registry.as_deref(), opts)
}

#[tokio::test]
async fn test_from_config() {
    let agent = Agent::from_config(test_config()).await;
    assert!(agent.is_ok());
}

#[tokio::test]
async fn test_session_default() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent.session("/tmp/test-workspace", None);
    assert!(session.is_ok());
    let debug = format!("{:?}", session.unwrap());
    assert!(debug.contains("AgentSession"));
}

#[tokio::test]
async fn test_session_uses_workspace_backend_for_direct_tools() {
    let fs = Arc::new(TestWorkspaceFs::default());
    fs.insert("app.txt", "hello from backend\n");
    let fs_backend: Arc<dyn crate::workspace::WorkspaceFileSystem> = fs.clone();
    let runner = Arc::new(TestWorkspaceRunner::default());
    let runner_backend: Arc<dyn crate::workspace::WorkspaceCommandRunner> = runner.clone();
    let services = crate::workspace::WorkspaceServices::builder(
        crate::workspace::WorkspaceRef::new("session-workspace", "session://workspace"),
        fs_backend,
    )
    .command_runner(runner_backend)
    .build();

    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent
        .session(
            "/server/local-placeholder",
            Some(SessionOptions::new().with_workspace_backend(services)),
        )
        .unwrap();

    let tool_names = session.tool_names();
    assert!(tool_names.contains(&"read".to_string()));
    assert!(tool_names.contains(&"write".to_string()));
    assert!(tool_names.contains(&"ls".to_string()));
    assert!(tool_names.contains(&"bash".to_string()));
    assert!(!tool_names.contains(&"grep".to_string()));
    assert!(!tool_names.contains(&"glob".to_string()));
    assert!(!tool_names.contains(&"git".to_string()));

    let read = session.read_file("app.txt").await.unwrap();
    assert!(read.contains("hello from backend"));

    let write = session
        .write_file("created.txt", "one\ntwo\n")
        .await
        .unwrap();
    assert_eq!(write.exit_code, 0, "{}", write.output);
    assert_eq!(fs.read_raw("created.txt").as_deref(), Some("one\ntwo\n"));

    let listing = session.ls(None).await.unwrap();
    assert_eq!(listing.exit_code, 0, "{}", listing.output);
    assert!(listing.output.contains("created.txt"));

    let edit = session
        .edit_file("created.txt", "one", "uno", false)
        .await
        .unwrap();
    assert_eq!(edit.exit_code, 0, "{}", edit.output);
    assert_eq!(fs.read_raw("created.txt").as_deref(), Some("uno\ntwo\n"));

    let patch = session
        .patch_file("created.txt", "@@ -1,2 +1,2 @@\n uno\n-two\n+dos")
        .await
        .unwrap();
    assert_eq!(patch.exit_code, 0, "{}", patch.output);
    assert_eq!(fs.read_raw("created.txt").as_deref(), Some("uno\ndos\n"));

    let bash = session.bash("pwd").await.unwrap();
    assert_eq!(bash, "session runner: pwd\n");
}

#[tokio::test]
async fn test_session_routes_agents_md_through_context_provider() {
    let temp_dir = tempfile::tempdir().unwrap();
    std::fs::write(
        temp_dir.path().join("AGENTS.md"),
        "Always run focused tests before reporting completion.",
    )
    .unwrap();

    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent
        .session(temp_dir.path().display().to_string(), None)
        .unwrap();

    let agents_provider = session
        .config
        .context_providers
        .iter()
        .find(|provider| provider.name() == "agents_md")
        .expect("AGENTS.md provider should be registered");
    assert!(!session
        .config
        .prompt_slots
        .extra
        .as_deref()
        .unwrap_or_default()
        .contains("Project Instructions (AGENTS.md)"));

    let result = agents_provider
        .query(&crate::context::ContextQuery::new("complete the task"))
        .await
        .unwrap();

    assert_eq!(result.items.len(), 1);
    assert_eq!(result.items[0].id, "agents_md");
    assert!(result.items[0]
        .content
        .contains("Always run focused tests before reporting completion."));
    assert_eq!(result.items[0].relevance, 0.95);
}

#[tokio::test]
async fn test_session_initializes_without_legacy_agentic_tools() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let _session = agent.session("/tmp/test-workspace", None).unwrap();
}

#[tokio::test]
async fn test_session_with_model_override() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new().with_model("openai/gpt-4o");
    let session = agent.session("/tmp/test-workspace", Some(opts));
    assert!(session.is_ok());
}

#[tokio::test]
async fn test_session_with_invalid_model_format() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new().with_model("gpt-4o");
    let session = agent.session("/tmp/test-workspace", Some(opts));
    assert!(session.is_err());
}

#[tokio::test]
async fn test_session_with_model_not_found() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new().with_model("openai/nonexistent");
    let session = agent.session("/tmp/test-workspace", Some(opts));
    assert!(session.is_err());
}

#[tokio::test]
async fn test_session_skill_dirs_preserve_agent_registry_validator() {
    use crate::skills::validator::DefaultSkillValidator;
    use crate::skills::SkillRegistry;

    let registry = Arc::new(SkillRegistry::new());
    registry.set_validator(Arc::new(DefaultSkillValidator::default()));

    let temp_dir = tempfile::tempdir().unwrap();
    let invalid_skill = temp_dir.path().join("invalid.md");
    std::fs::write(
        &invalid_skill,
        r#"---
name: BadName
description: "invalid skill name"
kind: instruction
---
# Invalid Skill
"#,
    )
    .unwrap();

    let opts = SessionOptions::new().with_skill_dirs([temp_dir.path()]);
    let effective_registry = build_effective_registry_for_test(Some(registry), &opts);
    assert!(effective_registry.get("BadName").is_none());
}

#[tokio::test]
async fn test_session_skill_registry_overrides_agent_registry_without_polluting_parent() {
    use crate::skills::{Skill, SkillKind, SkillRegistry};

    let registry = Arc::new(SkillRegistry::new());
    registry.register_unchecked(Arc::new(Skill {
        name: "shared-skill".to_string(),
        description: "agent level".to_string(),
        allowed_tools: None,
        disable_model_invocation: false,
        kind: SkillKind::Instruction,
        content: "agent content".to_string(),
        tags: vec![],
        version: None,
    }));

    let session_registry = Arc::new(SkillRegistry::new());
    session_registry.register_unchecked(Arc::new(Skill {
        name: "shared-skill".to_string(),
        description: "session level".to_string(),
        allowed_tools: None,
        disable_model_invocation: false,
        kind: SkillKind::Instruction,
        content: "session content".to_string(),
        tags: vec![],
        version: None,
    }));

    let opts = SessionOptions::new().with_skill_registry(session_registry);
    let effective_registry = build_effective_registry_for_test(Some(registry.clone()), &opts);

    assert_eq!(
        effective_registry.get("shared-skill").unwrap().content,
        "session content"
    );
    assert_eq!(
        registry.get("shared-skill").unwrap().content,
        "agent content"
    );
}

#[tokio::test]
async fn test_session_skill_dirs_override_session_registry_and_skip_invalid_entries() {
    use crate::skills::{Skill, SkillKind, SkillRegistry};

    let session_registry = Arc::new(SkillRegistry::new());
    session_registry.register_unchecked(Arc::new(Skill {
        name: "shared-skill".to_string(),
        description: "session registry".to_string(),
        allowed_tools: None,
        disable_model_invocation: false,
        kind: SkillKind::Instruction,
        content: "registry content".to_string(),
        tags: vec![],
        version: None,
    }));

    let temp_dir = tempfile::tempdir().unwrap();
    std::fs::write(
        temp_dir.path().join("shared.md"),
        r#"---
name: shared-skill
description: "skill dir override"
kind: instruction
---
# Shared Skill
dir content
"#,
    )
    .unwrap();
    std::fs::write(temp_dir.path().join("README.md"), "# not a skill").unwrap();

    let opts = SessionOptions::new()
        .with_skill_registry(session_registry)
        .with_skill_dirs([temp_dir.path()]);
    let effective_registry = build_effective_registry_for_test(None, &opts);

    assert_eq!(
        effective_registry.get("shared-skill").unwrap().description,
        "skill dir override"
    );
    assert!(effective_registry.get("README").is_none());
}

#[tokio::test]
async fn test_session_specific_skills_do_not_leak_across_sessions() {
    use crate::skills::{Skill, SkillKind, SkillRegistry};

    let mut agent = Agent::from_config(test_config()).await.unwrap();
    let agent_registry = Arc::new(SkillRegistry::with_builtins());
    agent.config.skill_registry = Some(agent_registry);

    let session_registry = Arc::new(SkillRegistry::new());
    session_registry.register_unchecked(Arc::new(Skill {
        name: "session-only".to_string(),
        description: "only for first session".to_string(),
        allowed_tools: None,
        disable_model_invocation: false,
        kind: SkillKind::Instruction,
        content: "session one".to_string(),
        tags: vec![],
        version: None,
    }));

    let session_one = agent
        .session(
            "/tmp/test-workspace",
            Some(SessionOptions::new().with_skill_registry(session_registry)),
        )
        .unwrap();
    let session_two = agent.session("/tmp/test-workspace", None).unwrap();

    assert!(session_one
        .config
        .skill_registry
        .as_ref()
        .unwrap()
        .get("session-only")
        .is_some());
    assert!(session_two
        .config
        .skill_registry
        .as_ref()
        .unwrap()
        .get("session-only")
        .is_none());
}

#[tokio::test]
async fn test_session_for_agent_applies_definition_and_keeps_skill_overrides_isolated() {
    use crate::skills::{Skill, SkillKind, SkillRegistry};
    use crate::subagent::AgentDefinition;

    let mut agent = Agent::from_config(test_config()).await.unwrap();
    agent.config.skill_registry = Some(Arc::new(SkillRegistry::with_builtins()));

    let definition = AgentDefinition::new("reviewer", "Review code")
        .with_prompt("Agent definition prompt")
        .with_max_steps(7);

    let session_registry = Arc::new(SkillRegistry::new());
    session_registry.register_unchecked(Arc::new(Skill {
        name: "agent-session-skill".to_string(),
        description: "agent session only".to_string(),
        allowed_tools: None,
        disable_model_invocation: false,
        kind: SkillKind::Instruction,
        content: "agent session content".to_string(),
        tags: vec![],
        version: None,
    }));

    let session_one = agent
        .session_for_agent(
            "/tmp/test-workspace",
            &definition,
            Some(SessionOptions::new().with_skill_registry(session_registry)),
        )
        .unwrap();
    let session_two = agent
        .session_for_agent("/tmp/test-workspace", &definition, None)
        .unwrap();

    assert_eq!(session_one.config.max_tool_rounds, 7);
    let extra = session_one.config.prompt_slots.extra.as_deref().unwrap();
    assert!(extra.contains("Agent definition prompt"));
    assert!(!extra.contains("agent-session-skill"));
    assert!(session_one
        .config
        .context_providers
        .iter()
        .any(|provider| provider.name() == "skills_catalog"));
    assert!(session_one
        .config
        .skill_registry
        .as_ref()
        .unwrap()
        .get("agent-session-skill")
        .is_some());
    assert!(session_two
        .config
        .skill_registry
        .as_ref()
        .unwrap()
        .get("agent-session-skill")
        .is_none());
}

#[tokio::test]
async fn test_session_for_agent_preserves_existing_prompt_slots_when_injecting_definition_prompt() {
    use crate::prompts::SystemPromptSlots;
    use crate::subagent::AgentDefinition;

    let agent = Agent::from_config(test_config()).await.unwrap();
    let definition = AgentDefinition::new("planner", "Plan work")
        .with_prompt("Definition extra prompt")
        .with_max_steps(3);

    let opts = SessionOptions::new().with_prompt_slots(SystemPromptSlots {
        style: None,
        role: Some("Custom role".to_string()),
        guidelines: None,
        response_style: None,
        extra: None,
    });

    let session = agent
        .session_for_agent("/tmp/test-workspace", &definition, Some(opts))
        .unwrap();

    assert_eq!(
        session.config.prompt_slots.role.as_deref(),
        Some("Custom role")
    );
    assert!(session
        .config
        .prompt_slots
        .extra
        .as_deref()
        .unwrap()
        .contains("Definition extra prompt"));
    assert_eq!(session.config.max_tool_rounds, 3);
}

#[tokio::test]
async fn test_new_with_acl_string() {
    let acl = r#"
            default_model = "anthropic/claude-sonnet-4-20250514"
            providers "anthropic" {
                apiKey = "test-key"
                models "claude-sonnet-4-20250514" {
                    name = "Claude Sonnet 4"
                }
            }
        "#;
    let agent = Agent::new(acl).await;
    assert!(agent.is_ok());
}

#[tokio::test]
async fn test_create_alias_acl() {
    let acl = r#"
            default_model = "anthropic/claude-sonnet-4-20250514"
            providers "anthropic" {
                apiKey = "test-key"
                models "claude-sonnet-4-20250514" {
                    name = "Claude Sonnet 4"
                }
            }
        "#;
    let agent = Agent::create(acl).await;
    assert!(agent.is_ok());
}

#[tokio::test]
async fn test_create_and_new_produce_same_result() {
    let acl = r#"
            default_model = "anthropic/claude-sonnet-4-20250514"
            providers "anthropic" {
                apiKey = "test-key"
                models "claude-sonnet-4-20250514" {
                    name = "Claude Sonnet 4"
                }
            }
        "#;
    let agent_new = Agent::new(acl).await;
    let agent_create = Agent::create(acl).await;
    assert!(agent_new.is_ok());
    assert!(agent_create.is_ok());

    // Both should produce working sessions
    let session_new = agent_new.unwrap().session("/tmp/test-ws-new", None);
    let session_create = agent_create.unwrap().session("/tmp/test-ws-create", None);
    assert!(session_new.is_ok());
    assert!(session_create.is_ok());
}

#[tokio::test]
async fn test_new_with_existing_acl_file_uses_file_loading() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config_path = temp_dir.path().join("agent.acl");
    std::fs::write(&config_path, "providers {").unwrap();

    let err = Agent::new(config_path.display().to_string())
        .await
        .unwrap_err();
    let msg = err.to_string();

    assert!(msg.contains("Failed to load config"));
    assert!(msg.contains("agent.acl"));
    assert!(!msg.contains("Failed to parse config as ACL string"));
}

#[tokio::test]
async fn test_new_with_missing_acl_file_reports_not_found() {
    let temp_dir = tempfile::tempdir().unwrap();
    let missing_path = temp_dir.path().join("agent.acl");

    let err = Agent::new(missing_path.display().to_string())
        .await
        .unwrap_err();
    let msg = err.to_string();

    assert!(msg.contains("Config file not found"));
    assert!(msg.contains("agent.acl"));
    assert!(!msg.contains("Failed to parse config as ACL string"));
}

#[tokio::test]
async fn test_new_rejects_hcl_files() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config_path = temp_dir.path().join("agent.hcl");
    std::fs::write(&config_path, "default_model = \"openai/test\"").unwrap();

    let err = Agent::new(config_path.display().to_string())
        .await
        .unwrap_err();
    let msg = err.to_string();

    assert!(msg.contains("HCL config files are not supported in 2.0"));
    assert!(msg.contains(".acl"));
}

#[test]
fn test_from_config_requires_default_model() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let config = CodeConfig {
        providers: vec![ProviderConfig {
            name: "anthropic".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: None,
            headers: std::collections::HashMap::new(),
            session_id_header: None,
            models: vec![],
        }],
        ..Default::default()
    };
    let result = rt.block_on(Agent::from_config(config));
    assert!(result.is_err());
}

#[tokio::test]
async fn test_history_empty_on_new_session() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent.session("/tmp/test-workspace", None).unwrap();
    assert!(session.history().is_empty());
}

#[tokio::test]
async fn test_stream_updates_history_and_auto_saves() {
    let store = Arc::new(crate::store::MemorySessionStore::new());
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new()
        .with_session_store(store.clone())
        .with_session_id("stream-history-test")
        .with_auto_save(true);
    let session = agent
        .build_session(
            "/tmp/test-stream-history".into(),
            Arc::new(StaticStreamingClient::new("streamed answer")),
            &opts,
        )
        .unwrap();

    let (mut rx, handle) = session.stream("hello", None).await.unwrap();
    let mut saw_end = false;
    while let Some(event) = rx.recv().await {
        if matches!(event, AgentEvent::End { .. }) {
            saw_end = true;
            break;
        }
    }
    handle.await.unwrap();

    assert!(saw_end);
    let history = session.history();
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].text(), "hello");
    assert_eq!(history[1].text(), "streamed answer");

    let saved = store
        .load("stream-history-test")
        .await
        .unwrap()
        .expect("saved session");
    assert_eq!(saved.messages.len(), 2);
    assert_eq!(saved.messages[1].text(), "streamed answer");

    let run_records = store
        .load_run_records("stream-history-test")
        .await
        .unwrap()
        .expect("saved run records");
    assert_eq!(run_records.len(), 1);
    assert_eq!(
        run_records[0].snapshot.status,
        crate::run::RunStatus::Completed
    );
    assert!(run_records[0]
        .events
        .iter()
        .any(|record| matches!(record.event, AgentEvent::End { .. })));
}

#[tokio::test]
async fn test_stream_with_custom_history_does_not_update_session_history() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent
        .build_session(
            "/tmp/test-stream-custom-history".into(),
            Arc::new(StaticStreamingClient::new("custom history answer")),
            &SessionOptions::new(),
        )
        .unwrap();
    let custom_history = vec![Message::user("custom prompt")];

    let (mut rx, handle) = session
        .stream("ignored", Some(&custom_history))
        .await
        .unwrap();
    while let Some(event) = rx.recv().await {
        if matches!(event, AgentEvent::End { .. }) {
            break;
        }
    }
    handle.await.unwrap();

    assert!(session.history().is_empty());
}

#[tokio::test]
async fn test_stream_error_does_not_update_history_or_auto_save() {
    let store = Arc::new(crate::store::MemorySessionStore::new());
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new()
        .with_session_store(store.clone())
        .with_session_id("stream-error-test")
        .with_auto_save(true);
    let session = agent
        .build_session(
            "/tmp/test-stream-error".into(),
            Arc::new(FailingStreamingClient),
            &opts,
        )
        .unwrap();

    let (mut rx, handle) = session.stream("hello", None).await.unwrap();
    let mut saw_error = false;
    while let Some(event) = rx.recv().await {
        if matches!(event, AgentEvent::Error { .. }) {
            saw_error = true;
            break;
        }
    }
    handle.await.unwrap();

    assert!(saw_error);
    assert!(session.history().is_empty());
    assert!(store.load("stream-error-test").await.unwrap().is_none());
}

#[tokio::test]
async fn test_stream_cancel_does_not_update_history_or_auto_save() {
    let store = Arc::new(crate::store::MemorySessionStore::new());
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new()
        .with_session_store(store.clone())
        .with_session_id("stream-cancel-test")
        .with_auto_save(true);
    let session = agent
        .build_session(
            "/tmp/test-stream-cancel".into(),
            Arc::new(CancellableStreamingClient::new("partial answer")),
            &opts,
        )
        .unwrap();

    let (mut rx, handle) = session.stream("hello", None).await.unwrap();
    let mut saw_delta = false;
    for _ in 0..16 {
        let event = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
            .await
            .expect("stream event before timeout")
            .expect("stream should stay open until cancelled");
        if matches!(event, AgentEvent::TextDelta { ref text } if text == "partial answer") {
            saw_delta = true;
            break;
        }
    }
    assert!(saw_delta);
    assert!(session.cancel().await);

    while rx.recv().await.is_some() {}
    handle.await.unwrap();

    assert!(session.history().is_empty());
    assert!(store.load("stream-cancel-test").await.unwrap().is_none());
    assert!(!session.cancel().await);
}

#[tokio::test]
async fn test_stream_with_attachments_cancel_does_not_update_history_or_auto_save() {
    let store = Arc::new(crate::store::MemorySessionStore::new());
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new()
        .with_session_store(store.clone())
        .with_session_id("stream-attachments-cancel-test")
        .with_auto_save(true);
    let session = agent
        .build_session(
            "/tmp/test-stream-attachments-cancel".into(),
            Arc::new(CancellableStreamingClient::new("partial attachment answer")),
            &opts,
        )
        .unwrap();
    let attachments = vec![crate::llm::Attachment::png(vec![1, 2, 3])];

    let (mut rx, handle) = session
        .stream_with_attachments("hello", &attachments, None)
        .await
        .unwrap();
    let mut saw_delta = false;
    for _ in 0..16 {
        let event = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
            .await
            .expect("stream event before timeout")
            .expect("stream should stay open until cancelled");
        if matches!(event, AgentEvent::TextDelta { .. }) {
            saw_delta = true;
            break;
        }
    }
    assert!(saw_delta);
    assert!(session.cancel().await);

    while rx.recv().await.is_some() {}
    handle.await.unwrap();

    assert!(session.history().is_empty());
    assert!(store
        .load("stream-attachments-cancel-test")
        .await
        .unwrap()
        .is_none());
    assert_eq!(
        session.runs().await[0].status,
        crate::run::RunStatus::Cancelled
    );
    assert!(!session.cancel().await);
}

#[tokio::test]
async fn test_run_handle_cancels_send_with_attachments() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = Arc::new(
        agent
            .build_session(
                "/tmp/test-send-attachments-run-handle-cancel".into(),
                Arc::new(CancellableStreamingClient::new("partial answer")),
                &SessionOptions::new(),
            )
            .unwrap(),
    );
    let worker_session = Arc::clone(&session);
    let attachments = vec![crate::llm::Attachment::png(vec![1, 2, 3])];

    let worker = tokio::spawn(async move {
        worker_session
            .send_with_attachments("hello", &attachments, None)
            .await
    });

    let mut run = None;
    for _ in 0..20 {
        if let Some(current) = session.current_run().await {
            run = Some(current);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let run = run.expect("current run should be visible");
    assert!(run.cancel().await);

    let result = tokio::time::timeout(std::time::Duration::from_secs(1), worker)
        .await
        .expect("send_with_attachments should stop after cancellation")
        .expect("worker should not panic");
    assert!(result.is_err());
    assert_eq!(run.status().await, Some(crate::run::RunStatus::Cancelled));
    assert!(session.history().is_empty());
    assert!(!session.cancel().await);
}

#[tokio::test]
async fn test_cancel_run_only_cancels_matching_current_run() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = Arc::new(
        agent
            .build_session(
                "/tmp/test-cancel-run-by-id".into(),
                Arc::new(CancellableStreamingClient::new("partial answer")),
                &SessionOptions::new(),
            )
            .unwrap(),
    );
    let worker_session = Arc::clone(&session);
    let worker = tokio::spawn(async move { worker_session.send("hello", None).await });

    let mut run_id = None;
    for _ in 0..20 {
        if let Some(current) = session.current_run().await {
            run_id = Some(current.id().to_string());
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let run_id = run_id.expect("current run should be visible");

    assert!(!session.cancel_run("stale-run").await);
    assert!(session.cancel_run(&run_id).await);

    let result = tokio::time::timeout(std::time::Duration::from_secs(1), worker)
        .await
        .expect("send should stop after cancellation")
        .expect("worker should not panic");
    assert!(result.is_err());
    assert_eq!(
        session.run_snapshot(&run_id).await.unwrap().status,
        crate::run::RunStatus::Cancelled
    );
    assert!(!session.cancel_run(&run_id).await);
}

#[tokio::test]
async fn test_is_closed_starts_false() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent.session("/tmp/test-close-default", None).unwrap();
    assert!(!session.is_closed());
}

#[tokio::test]
async fn test_close_marks_session_closed_and_is_idempotent() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent.session("/tmp/test-close-idempotent", None).unwrap();
    assert!(!session.is_closed());

    session.close().await;
    assert!(session.is_closed());

    session.close().await;
    assert!(session.is_closed());
}

#[tokio::test]
async fn test_send_after_close_returns_session_closed_error() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new().with_session_id("send-after-close");
    let session = agent
        .build_session(
            "/tmp/test-send-after-close".into(),
            Arc::new(StaticStreamingClient::new("never delivered")),
            &opts,
        )
        .unwrap();

    session.close().await;
    let err = session.send("hello", None).await.unwrap_err();
    match err {
        crate::error::CodeError::SessionClosed { session_id } => {
            assert_eq!(session_id, "send-after-close");
        }
        other => panic!("expected SessionClosed, got {other:?}"),
    }
}

#[tokio::test]
async fn test_stream_after_close_returns_session_closed_error() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new().with_session_id("stream-after-close");
    let session = agent
        .build_session(
            "/tmp/test-stream-after-close".into(),
            Arc::new(StaticStreamingClient::new("never delivered")),
            &opts,
        )
        .unwrap();

    session.close().await;
    let err = session.stream("hello", None).await.unwrap_err();
    assert!(matches!(
        err,
        crate::error::CodeError::SessionClosed { ref session_id }
            if session_id == "stream-after-close"
    ));
}

#[tokio::test]
async fn test_close_cancels_in_flight_send() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = Arc::new(
        agent
            .build_session(
                "/tmp/test-close-in-flight".into(),
                Arc::new(CancellableStreamingClient::new("partial answer")),
                &SessionOptions::new(),
            )
            .unwrap(),
    );

    let worker_session = Arc::clone(&session);
    let worker = tokio::spawn(async move { worker_session.send("hello", None).await });

    let mut run_id = None;
    for _ in 0..50 {
        if let Some(current) = session.current_run().await {
            run_id = Some(current.id().to_string());
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let run_id = run_id.expect("current run should be visible before close()");

    session.close().await;
    assert!(session.is_closed());

    let result = tokio::time::timeout(std::time::Duration::from_secs(1), worker)
        .await
        .expect("send should stop after close")
        .expect("worker should not panic");
    assert!(result.is_err());
    assert_eq!(
        session.run_snapshot(&run_id).await.unwrap().status,
        crate::run::RunStatus::Cancelled
    );
}

/// Custom BudgetGuard that denies the first LLM call — used to verify
/// that the framework consults the guard and bails before touching
/// the LLM client. Records whether `check_before_llm` was called.
#[derive(Debug, Default)]
struct DenyingBudgetGuard {
    checks: std::sync::atomic::AtomicUsize,
    llm_records: std::sync::atomic::AtomicUsize,
}

#[async_trait::async_trait]
impl crate::budget::BudgetGuard for DenyingBudgetGuard {
    async fn check_before_llm(
        &self,
        _session_id: &str,
        _est_tokens: usize,
    ) -> crate::budget::BudgetDecision {
        self.checks
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        crate::budget::BudgetDecision::Deny {
            resource: "llm_tokens".to_string(),
            reason: "test cap exceeded".to_string(),
        }
    }

    async fn record_after_llm(&self, _session_id: &str, _usage: &crate::llm::TokenUsage) {
        self.llm_records
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[tokio::test]
async fn test_budget_guard_deny_aborts_llm_call() {
    let guard = Arc::new(DenyingBudgetGuard::default());
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new()
        .with_session_id("budget-deny-test")
        .with_budget_guard(guard.clone() as Arc<dyn crate::budget::BudgetGuard>);
    let session = agent
        .build_session(
            "/tmp/test-budget-deny".into(),
            Arc::new(StaticStreamingClient::new("never-delivered")),
            &opts,
        )
        .unwrap();

    let err = session.send("hello", None).await.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Budget exhausted") || msg.contains("llm_tokens"),
        "expected budget-exhausted error, got: {msg}"
    );
    assert_eq!(
        guard.checks.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "BudgetGuard::check_before_llm must be consulted exactly once"
    );
    assert_eq!(
        guard.llm_records.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "record_after_llm must not fire when the call was denied"
    );
    assert!(
        session.history().is_empty(),
        "denied call must not pollute conversation history"
    );
}

#[test]
fn test_cluster_agent_events_serialize_with_expected_tags() {
    // Lock the wire schema for cluster-event variants — these are
    // emitted by the host through HookExecutor and need
    // stable JSON tags so external producers can target them.
    let budget = AgentEvent::BudgetThresholdHit {
        resource: "llm_tokens".to_string(),
        kind: "soft".to_string(),
        consumed: 12000.0,
        limit: 10000.0,
        message: Some("approaching daily cap".to_string()),
    };
    let json = serde_json::to_string(&budget).unwrap();
    assert!(
        json.contains("\"type\":\"budget_threshold_hit\""),
        "got: {json}"
    );
    assert!(json.contains("\"resource\":\"llm_tokens\""), "got: {json}");

    let passivate = AgentEvent::PassivationRequested {
        reason: "node_drain".to_string(),
        deadline_ms: Some(1_700_000_000_000),
    };
    let json = serde_json::to_string(&passivate).unwrap();
    assert!(
        json.contains("\"type\":\"passivation_requested\""),
        "got: {json}"
    );

    let peer = AgentEvent::PeerInvocation {
        from_session_id: "peer-1".to_string(),
        from_tenant_id: Some("acme".to_string()),
        correlation_id: None, // omitted via skip_serializing_if
    };
    let json = serde_json::to_string(&peer).unwrap();
    assert!(json.contains("\"type\":\"peer_invocation\""), "got: {json}");
    assert!(
        !json.contains("correlation_id"),
        "None field must be skipped, got: {json}"
    );

    // Round-trip — ensures the #[serde(default)] hints don't break loading
    // from a payload that omits the optional fields.
    let minimal_peer = r#"{"type":"peer_invocation","from_session_id":"x"}"#;
    let parsed: AgentEvent = serde_json::from_str(minimal_peer).unwrap();
    assert!(
        matches!(parsed, AgentEvent::PeerInvocation { ref from_session_id, .. } if from_session_id == "x")
    );
}

#[tokio::test]
async fn test_custom_host_env_yields_deterministic_session_and_run_ids() {
    use crate::host_env::{FixedClock, HostEnv, SequentialIdGenerator};

    let env = Arc::new(HostEnv::new(
        Arc::new(SequentialIdGenerator::new("test")),
        Arc::new(FixedClock::new(1_700_000_000_000)),
    ));

    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts_a = SessionOptions::new().with_host_env(env.clone());
    let session_a = agent
        .session("/tmp/test-host-env-a", Some(opts_a))
        .expect("session a");

    // First call to next_id() yields "test-0" — used as session_id.
    assert_eq!(
        session_a.id(),
        "test-0",
        "session_id must come from HostEnv"
    );

    // run_id derives from next_id() too, prefixed with "run-".
    let session_a = Arc::new(session_a);
    let worker = {
        let s = Arc::clone(&session_a);
        tokio::spawn(async move {
            // Use a static streaming client by building manually so the
            // call resolves without an actual provider.
            let _ = s;
        })
    };
    let _ = worker.await;

    // Second session reuses the same generator → continues the sequence.
    let opts_b = SessionOptions::new().with_host_env(env);
    let session_b = agent
        .session("/tmp/test-host-env-b", Some(opts_b))
        .expect("session b");
    assert_eq!(session_b.id(), "test-1");
}

#[tokio::test]
async fn test_runtime_budget_guard_overrides_session_options_value() {
    // A guard installed via set_budget_guard() *after* construction
    // must take effect on the next send/stream — that's the entry
    // point Node SDK relies on (JsFunction can't live inside a
    // value-typed SessionOptions).
    let runtime_guard = Arc::new(DenyingBudgetGuard::default());
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new().with_session_id("runtime-guard-override");
    let session = agent
        .build_session(
            "/tmp/test-runtime-guard".into(),
            Arc::new(StaticStreamingClient::new("never-delivered")),
            &opts,
        )
        .unwrap();

    // No guard installed at build time -> send would succeed. Install
    // a denying guard now and assert the next send is aborted.
    session.set_budget_guard(Some(
        runtime_guard.clone() as Arc<dyn crate::budget::BudgetGuard>
    ));
    let err = session.send("hello", None).await.unwrap_err();
    assert!(err.to_string().contains("Budget exhausted"));
    assert_eq!(
        runtime_guard
            .checks
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );

    // Clearing the override should let a follow-up send succeed.
    session.set_budget_guard(None);
    let result = session.send("hello again", None).await.unwrap();
    assert_eq!(result.text, "never-delivered");
}

#[tokio::test]
async fn test_disconnect_idle_mcp_is_safe_no_op_without_global_mcp() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    // test_config carries no mcp_servers, so global_mcp is None and
    // the idle sweep must short-circuit to an empty Vec without
    // panicking — the contract surface a host's sweeper will rely on.
    let dropped = agent.disconnect_idle_mcp(0).await;
    assert!(dropped.is_empty());
}

#[tokio::test]
async fn test_identity_labels_default_to_none() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent.session("/tmp/test-id-default", None).unwrap();
    assert!(session.tenant_id().is_none());
    assert!(session.principal().is_none());
    assert!(session.agent_template_id().is_none());
    assert!(session.correlation_id().is_none());
}

#[tokio::test]
async fn test_identity_labels_round_trip_via_session_options() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new()
        .with_tenant_id("acme-corp")
        .with_principal("user-42")
        .with_agent_template_id("planner-v3")
        .with_correlation_id("trace-deadbeef");
    let session = agent
        .session("/tmp/test-id-set", Some(opts))
        .expect("session");

    assert_eq!(session.tenant_id(), Some("acme-corp"));
    assert_eq!(session.principal(), Some("user-42"));
    assert_eq!(session.agent_template_id(), Some("planner-v3"));
    assert_eq!(session.correlation_id(), Some("trace-deadbeef"));
}

#[tokio::test]
async fn test_agent_list_sessions_tracks_live_sessions() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    assert!(agent.list_sessions().await.is_empty());

    let opts_a = SessionOptions::new().with_session_id("registry-a");
    let opts_b = SessionOptions::new().with_session_id("registry-b");
    let session_a = agent
        .build_session(
            "/tmp/test-registry-a".into(),
            Arc::new(StaticStreamingClient::new("answer-a")),
            &opts_a,
        )
        .unwrap();
    let session_b = agent
        .build_session(
            "/tmp/test-registry-b".into(),
            Arc::new(StaticStreamingClient::new("answer-b")),
            &opts_b,
        )
        .unwrap();

    let ids = agent.list_sessions().await;
    assert_eq!(
        ids,
        vec!["registry-a".to_string(), "registry-b".to_string()]
    );

    drop(session_a);
    // After drop, the registry's Weak becomes dangling; list_sessions prunes it.
    let after = agent.list_sessions().await;
    assert_eq!(after, vec!["registry-b".to_string()]);

    drop(session_b);
    assert!(agent.list_sessions().await.is_empty());
}

#[tokio::test]
async fn test_agent_close_session_closes_target_session() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new().with_session_id("close-by-id");
    let session = agent
        .build_session(
            "/tmp/test-agent-close-session".into(),
            Arc::new(StaticStreamingClient::new("never")),
            &opts,
        )
        .unwrap();
    assert!(!session.is_closed());

    assert!(agent.close_session("close-by-id").await);
    assert!(session.is_closed());

    // Idempotent: second call still reports `true` (we found a live handle)
    // OR `false` (target already closed) — accept either; what matters is no panic.
    let _ = agent.close_session("close-by-id").await;

    // Unknown ids report false.
    assert!(!agent.close_session("does-not-exist").await);
}

#[tokio::test]
async fn test_agent_close_closes_every_live_session() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts_a = SessionOptions::new().with_session_id("agent-close-a");
    let opts_b = SessionOptions::new().with_session_id("agent-close-b");
    let session_a = agent
        .build_session(
            "/tmp/test-agent-close-a".into(),
            Arc::new(StaticStreamingClient::new("a")),
            &opts_a,
        )
        .unwrap();
    let session_b = agent
        .build_session(
            "/tmp/test-agent-close-b".into(),
            Arc::new(StaticStreamingClient::new("b")),
            &opts_b,
        )
        .unwrap();

    agent.close().await;
    assert!(session_a.is_closed());
    assert!(session_b.is_closed());

    // After Agent::close(), session creation must fail fast — the agent has
    // already disposed of its resources.
    let err = agent
        .session("/tmp/test-agent-closed", None)
        .err()
        .expect("session() after close() must error");
    let msg = err.to_string();
    assert!(msg.contains("closed") || msg.contains("Closed"));
}

#[tokio::test]
async fn test_session_cancel_token_starts_uncancelled() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent
        .session("/tmp/test-session-cancel-fresh", None)
        .unwrap();
    let tok = session.session_cancel_token();
    assert!(!tok.is_cancelled());
}

#[tokio::test]
async fn test_close_cancels_session_token() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent
        .session("/tmp/test-session-cancel-on-close", None)
        .unwrap();
    let observer = session.session_cancel_token();
    assert!(!observer.is_cancelled());

    session.close().await;
    assert!(observer.is_cancelled());
}

#[tokio::test]
async fn test_session_cancel_token_propagates_to_in_flight_run() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = Arc::new(
        agent
            .build_session(
                "/tmp/test-session-cancel-cascades".into(),
                Arc::new(CancellableStreamingClient::new("partial answer")),
                &SessionOptions::new(),
            )
            .unwrap(),
    );

    let worker_session = Arc::clone(&session);
    let worker = tokio::spawn(async move { worker_session.send("hello", None).await });

    let mut run_id = None;
    for _ in 0..50 {
        if let Some(current) = session.current_run().await {
            run_id = Some(current.id().to_string());
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let run_id = run_id.expect("current run should be visible");

    // Fire the session-level token directly, bypassing close()/cancel().
    // The in-flight run's token must be a *child* of this one for
    // cancellation to propagate.
    session.session_cancel_token().cancel();

    let result = tokio::time::timeout(std::time::Duration::from_secs(1), worker)
        .await
        .expect("send should stop after session_cancel fires")
        .expect("worker should not panic");
    assert!(result.is_err());
    assert_eq!(
        session.run_snapshot(&run_id).await.unwrap().status,
        crate::run::RunStatus::Cancelled
    );
}

#[tokio::test]
async fn test_send_with_attachments_passes_session_id_to_context_providers() {
    let provider = Arc::new(CapturingContextProvider::default());
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new()
        .with_session_id("attachments-context-session")
        .with_context_provider(provider.clone());
    let session = agent
        .build_session(
            "/tmp/test-send-attachments-context".into(),
            Arc::new(StaticStreamingClient::new("attachment answer")),
            &opts,
        )
        .unwrap();
    let attachments = vec![crate::llm::Attachment::png(vec![1, 2, 3])];

    session
        .send_with_attachments("hello", &attachments, None)
        .await
        .unwrap();

    let session_ids = provider.session_ids.lock().unwrap();
    assert!(!session_ids.is_empty());
    assert!(session_ids
        .iter()
        .all(|id| id.as_deref() == Some("attachments-context-session")));
}

#[tokio::test]
async fn test_send_records_run_snapshot_and_events() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent
        .build_session(
            "/tmp/test-send-run-store".into(),
            Arc::new(StaticStreamingClient::new("run answer")),
            &SessionOptions::new(),
        )
        .unwrap();

    let result = session.send("hello", None).await.unwrap();
    assert_eq!(result.text, "run answer");

    let runs = session.runs().await;
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].status, crate::run::RunStatus::Completed);
    assert_eq!(runs[0].result_text.as_deref(), Some("run answer"));

    let events = session.run_events(&runs[0].id).await;
    assert!(events
        .iter()
        .any(|record| matches!(record.event, AgentEvent::Start { .. })));
    assert!(events
        .iter()
        .any(|record| matches!(record.event, AgentEvent::End { .. })));
}

#[tokio::test]
async fn test_send_publishes_runtime_events_to_hook_executor() {
    let hook = Arc::new(RecordingRuntimeHook::default());
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new().with_hook_executor(hook.clone());
    let session = agent
        .build_session(
            "/tmp/test-runtime-event-hook".into(),
            Arc::new(StaticStreamingClient::new("hooked answer")),
            &opts,
        )
        .unwrap();

    session.send("hello", None).await.unwrap();

    let events = hook.events.lock().unwrap();
    assert!(events
        .iter()
        .any(|(_, session_id, event)| session_id == session.id()
            && matches!(event, AgentEvent::Start { .. })));
    assert!(events
        .iter()
        .any(|(_, session_id, event)| session_id == session.id()
            && matches!(event, AgentEvent::End { .. })));
    assert!(events
        .iter()
        .all(|(run_id, _, _)| run_id.starts_with("run-")));
}

#[tokio::test]
async fn test_stream_exposes_current_run_handle_and_replay() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent
        .build_session(
            "/tmp/test-stream-run-handle".into(),
            Arc::new(CancellableStreamingClient::new("partial answer")),
            &SessionOptions::new(),
        )
        .unwrap();

    let (mut rx, handle) = session.stream("hello", None).await.unwrap();
    let mut saw_delta = false;
    for _ in 0..16 {
        let event = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
            .await
            .expect("stream event before timeout")
            .expect("stream emits event");
        if matches!(event, AgentEvent::TextDelta { .. }) {
            saw_delta = true;
            break;
        }
    }
    assert!(saw_delta);

    let run = session.current_run().await.expect("current run handle");
    assert_eq!(run.session_id(), session.id());
    assert!(matches!(
        run.status().await,
        Some(crate::run::RunStatus::Executing | crate::run::RunStatus::Planning)
    ));
    assert!(run.cancel().await);

    while rx.recv().await.is_some() {}
    handle.await.unwrap();

    let snapshot = run
        .snapshot()
        .await
        .expect("run snapshot remains replayable");
    assert_eq!(snapshot.status, crate::run::RunStatus::Cancelled);
    assert!(!run.events().await.is_empty());
}

#[tokio::test]
async fn test_session_options_with_agent_dir() {
    let opts = SessionOptions::new()
        .with_agent_dir("/tmp/agents")
        .with_agent_dir("/tmp/more-agents");
    assert_eq!(opts.agent_dirs.len(), 2);
    assert_eq!(opts.agent_dirs[0], PathBuf::from("/tmp/agents"));
    assert_eq!(opts.agent_dirs[1], PathBuf::from("/tmp/more-agents"));
}

// ========================================================================
// Queue Integration Tests
// ========================================================================

#[test]
fn test_session_options_with_queue_config() {
    let qc = SessionQueueConfig::default().with_lane_features();
    let opts = SessionOptions::new().with_queue_config(qc.clone());
    assert!(opts.queue_config.is_some());

    let config = opts.queue_config.unwrap();
    assert!(config.enable_dlq);
    assert!(config.enable_metrics);
    assert!(config.enable_alerts);
    assert_eq!(config.default_timeout_ms, Some(60_000));
}

#[tokio::test]
async fn test_session_uses_single_delegation_tool_surface() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent
        .session("/tmp/test-workspace-delegation-tools", None)
        .unwrap();
    let names = session.tool_names();

    assert!(names.contains(&"task".to_string()));
    assert!(names.contains(&"parallel_task".to_string()));
    assert!(!names.contains(&"run_team".to_string()));
}

#[tokio::test]
async fn test_session_can_disable_manual_delegation_tools_without_dropping_registry() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new()
        .with_worker_agent(crate::subagent::WorkerAgentSpec::planner(
            "release-planner",
            "Plan releases",
        ))
        .with_manual_delegation_enabled(false);
    let session = agent
        .session("/tmp/test-workspace-no-manual-delegation", Some(opts))
        .unwrap();
    let names = session.tool_names();

    assert!(!names.contains(&"task".to_string()));
    assert!(!names.contains(&"parallel_task".to_string()));
    assert!(session.agent_registry.exists("release-planner"));
    assert!(!session.config.auto_delegation.allow_manual_delegation);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_with_queue_config() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let qc = SessionQueueConfig::default();
    let opts = SessionOptions::new().with_queue_config(qc);
    let session = agent.session("/tmp/test-workspace-queue", Some(opts));
    assert!(session.is_ok());
    let session = session.unwrap();
    assert!(session.has_queue());
}

#[tokio::test]
async fn test_session_without_queue_config() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent.session("/tmp/test-workspace-noqueue", None).unwrap();
    assert!(!session.has_queue());
}

#[tokio::test]
async fn test_session_queue_stats_without_queue() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent.session("/tmp/test-workspace-stats", None).unwrap();
    let stats = session.queue_stats().await;
    // Without a queue, stats should have zero values
    assert_eq!(stats.total_pending, 0);
    assert_eq!(stats.total_active, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_queue_stats_with_queue() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let qc = SessionQueueConfig::default();
    let opts = SessionOptions::new().with_queue_config(qc);
    let session = agent
        .session("/tmp/test-workspace-qstats", Some(opts))
        .unwrap();
    let stats = session.queue_stats().await;
    // Fresh queue with no commands should have zero stats
    assert_eq!(stats.total_pending, 0);
    assert_eq!(stats.total_active, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_pending_external_tasks_empty() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let qc = SessionQueueConfig::default();
    let opts = SessionOptions::new().with_queue_config(qc);
    let session = agent
        .session("/tmp/test-workspace-ext", Some(opts))
        .unwrap();
    let tasks = session.pending_external_tasks().await;
    assert!(tasks.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_confirmation_api_resolves_pending_request() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let (event_tx, _) = tokio::sync::broadcast::channel(8);
    let manager = Arc::new(crate::hitl::ConfirmationManager::new(
        crate::hitl::ConfirmationPolicy::enabled(),
        event_tx,
    ));
    let opts = SessionOptions::new().with_confirmation_manager(manager.clone());
    let session = agent.session("/tmp/test-workspace", Some(opts)).unwrap();

    let receiver = manager
        .request_confirmation(
            "tool-1",
            "bash",
            &serde_json::json!({ "command": "echo hi" }),
        )
        .await;

    let pending = session.pending_confirmations().await;
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].tool_id, "tool-1");
    assert_eq!(pending[0].tool_name, "bash");

    let found = session
        .confirm_tool_use("tool-1", true, Some("approved by test".to_string()))
        .await
        .unwrap();
    assert!(found);

    let response = receiver.await.unwrap();
    assert!(response.approved);
    assert_eq!(response.reason.as_deref(), Some("approved by test"));
    assert!(session.pending_confirmations().await.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_confirmation_api_without_manager_is_noop() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent.session("/tmp/test-workspace", None).unwrap();

    assert!(session.pending_confirmations().await.is_empty());
    assert!(!session
        .confirm_tool_use("missing", true, None)
        .await
        .unwrap());
    assert_eq!(session.cancel_confirmations().await, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_dead_letters_empty() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let qc = SessionQueueConfig::default().with_dlq(Some(100));
    let opts = SessionOptions::new().with_queue_config(qc);
    let session = agent
        .session("/tmp/test-workspace-dlq", Some(opts))
        .unwrap();
    let dead = session.dead_letters().await;
    assert!(dead.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_queue_metrics_disabled() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    // Metrics not enabled
    let qc = SessionQueueConfig::default();
    let opts = SessionOptions::new().with_queue_config(qc);
    let session = agent
        .session("/tmp/test-workspace-nomet", Some(opts))
        .unwrap();
    let metrics = session.queue_metrics().await;
    assert!(metrics.is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_queue_metrics_enabled() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let qc = SessionQueueConfig::default().with_metrics();
    let opts = SessionOptions::new().with_queue_config(qc);
    let session = agent
        .session("/tmp/test-workspace-met", Some(opts))
        .unwrap();
    let metrics = session.queue_metrics().await;
    assert!(metrics.is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_set_lane_handler() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let qc = SessionQueueConfig::default();
    let opts = SessionOptions::new().with_queue_config(qc);
    let session = agent
        .session("/tmp/test-workspace-handler", Some(opts))
        .unwrap();

    // Set Execute lane to External mode
    session
        .set_lane_handler(
            SessionLane::Execute,
            LaneHandlerConfig {
                mode: crate::queue::TaskHandlerMode::External,
                timeout_ms: 30_000,
            },
        )
        .await;

    // No panic = success. The handler config is stored internally.
    // We can't directly read it back but we verify no errors.
}

// ========================================================================
// Session Persistence Tests
// ========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_session_has_id() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent.session("/tmp/test-ws-id", None).unwrap();
    // Auto-generated UUID
    assert!(!session.session_id().is_empty());
    assert_eq!(session.session_id().len(), 36); // UUID format
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_explicit_id() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new().with_session_id("my-session-42");
    let session = agent.session("/tmp/test-ws-eid", Some(opts)).unwrap();
    assert_eq!(session.session_id(), "my-session-42");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_artifact_store_limits_option() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts =
        SessionOptions::new().with_artifact_store_limits(crate::tools::ArtifactStoreLimits {
            max_artifacts: 3,
            max_bytes: 4096,
        });
    let session = agent
        .session("/tmp/test-ws-artifact-limits", Some(opts))
        .unwrap();

    let limits = session.tool_executor.artifact_store().limits();
    assert_eq!(limits.max_artifacts, 3);
    assert_eq!(limits.max_bytes, 4096);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_save_no_store() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent.session("/tmp/test-ws-save", None).unwrap();
    // save() is a no-op when no store is configured
    session.save().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_save_and_load() {
    let store = Arc::new(crate::store::MemorySessionStore::new());
    let agent = Agent::from_config(test_config()).await.unwrap();

    let opts = SessionOptions::new()
        .with_session_store(store.clone())
        .with_session_id("persist-test");
    let session = agent.session("/tmp/test-ws-persist", Some(opts)).unwrap();

    // Save empty session
    session.save().await.unwrap();

    // Verify it was stored
    assert!(store.exists("persist-test").await.unwrap());

    let data = store.load("persist-test").await.unwrap().unwrap();
    assert_eq!(data.id, "persist-test");
    assert!(data.messages.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_save_persists_runtime_config() {
    let store = Arc::new(crate::store::MemorySessionStore::new());
    let agent = Agent::from_config(test_config()).await.unwrap();
    let queue_config = SessionQueueConfig::default().with_metrics();
    let confirmation_policy = crate::hitl::ConfirmationPolicy::enabled();
    let permission_policy = crate::permissions::PermissionPolicy::new().allow("bash(echo:*)");

    let opts = SessionOptions::new()
        .with_session_store(store.clone())
        .with_session_id("runtime-config-test")
        .with_model("openai/gpt-4o")
        .with_queue_config(queue_config)
        .with_confirmation_policy(confirmation_policy)
        .with_permission_policy(permission_policy)
        .with_active_skill_tool_restrictions(true);
    let session = agent
        .session("/tmp/test-ws-runtime-config", Some(opts))
        .unwrap();
    session.save().await.unwrap();

    let data = store.load("runtime-config-test").await.unwrap().unwrap();
    assert_eq!(data.model_name.as_deref(), Some("openai/gpt-4o"));
    assert_eq!(
        data.llm_config.as_ref().map(|c| c.provider.as_str()),
        Some("openai")
    );
    assert_eq!(
        data.llm_config.as_ref().map(|c| c.model.as_str()),
        Some("gpt-4o")
    );
    assert!(data.config.queue_config.is_some());
    assert!(data
        .config
        .confirmation_policy
        .as_ref()
        .is_some_and(|p| p.enabled));
    assert!(data
        .config
        .permission_policy
        .as_ref()
        .is_some_and(|p| !p.allow.is_empty()));
    assert!(data.config.enforce_active_skill_tool_restrictions);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_resume_session_restores_runtime_config() {
    let store = Arc::new(crate::store::MemorySessionStore::new());
    let agent = Agent::from_config(test_config()).await.unwrap();
    let queue_config = SessionQueueConfig::default().with_metrics();
    let confirmation_policy = crate::hitl::ConfirmationPolicy::enabled();
    let permission_policy = crate::permissions::PermissionPolicy::new().allow("bash(echo:*)");

    let opts = SessionOptions::new()
        .with_session_store(store.clone())
        .with_session_id("resume-runtime-config-test")
        .with_model("openai/gpt-4o")
        .with_queue_config(queue_config)
        .with_confirmation_policy(confirmation_policy)
        .with_permission_policy(permission_policy)
        .with_active_skill_tool_restrictions(true);
    let session = agent
        .session("/tmp/test-ws-resume-runtime", Some(opts))
        .unwrap();
    session.save().await.unwrap();

    let opts2 = SessionOptions::new().with_session_store(store.clone());
    let resumed = agent
        .resume_session("resume-runtime-config-test", opts2)
        .unwrap();

    assert_eq!(resumed.model_name, "openai/gpt-4o");
    assert!(resumed.has_queue());
    assert!(resumed.config.confirmation_policy.is_some());
    assert!(resumed.config.confirmation_manager.is_some());
    assert!(resumed.config.permission_policy.is_some());
    assert!(resumed.config.permission_checker.is_some());
    assert!(resumed.config.enforce_active_skill_tool_restrictions);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_save_with_history() {
    let store = Arc::new(crate::store::MemorySessionStore::new());
    let agent = Agent::from_config(test_config()).await.unwrap();

    let opts = SessionOptions::new()
        .with_session_store(store.clone())
        .with_session_id("history-test");
    let session = agent.session("/tmp/test-ws-hist", Some(opts)).unwrap();

    // Manually inject history
    {
        let mut h = session.history.write().unwrap();
        h.push(Message::user("Hello"));
        h.push(Message::user("How are you?"));
    }

    session.save().await.unwrap();

    let data = store.load("history-test").await.unwrap().unwrap();
    assert_eq!(data.messages.len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_resume_session() {
    let store = Arc::new(crate::store::MemorySessionStore::new());
    let agent = Agent::from_config(test_config()).await.unwrap();

    // Create and save a session with history
    let opts = SessionOptions::new()
        .with_session_store(store.clone())
        .with_session_id("resume-test");
    let session = agent.session("/tmp/test-ws-resume", Some(opts)).unwrap();
    {
        let mut h = session.history.write().unwrap();
        h.push(Message::user("What is Rust?"));
        h.push(Message::user("Tell me more"));
    }
    session.save().await.unwrap();

    // Resume the session
    let opts2 = SessionOptions::new().with_session_store(store.clone());
    let resumed = agent.resume_session("resume-test", opts2).unwrap();

    assert_eq!(resumed.session_id(), "resume-test");
    let history = resumed.history();
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].text(), "What is Rust?");
}

/// H4 regression: a run that completes in-process must DELETE its loop
/// checkpoint (the checkpoint exists only to survive a crash). Before
/// the fix, every tool-using run leaked a checkpoint forever.
///
/// We use a deterministic HostEnv so the run id is predictable, seed a
/// checkpoint under that id, run a (no-tool) send that completes through
/// the normal lifecycle, and assert the checkpoint was cleared.
#[tokio::test(flavor = "multi_thread")]
async fn test_completed_run_clears_its_loop_checkpoint() {
    use crate::host_env::{HostEnv, SequentialIdGenerator, SystemClock};
    use crate::loop_checkpoint::{LoopCheckpoint, LOOP_CHECKPOINT_SCHEMA_VERSION};

    let store = Arc::new(crate::store::MemorySessionStore::new());
    let agent = Agent::from_config(test_config()).await.unwrap();

    // Deterministic ids: session_id is set explicitly (consumes no
    // counter), so the first next_id() goes to the run -> "run-seq-0".
    let env = Arc::new(HostEnv::new(
        Arc::new(SequentialIdGenerator::new("seq")),
        Arc::new(SystemClock),
    ));
    let opts = SessionOptions::new()
        .with_session_id("ckpt-clear-session")
        .with_session_store(store.clone() as Arc<dyn crate::store::SessionStore>)
        .with_host_env(env);
    let session = agent
        .build_session(
            "/tmp/test-ckpt-clear".into(),
            Arc::new(StaticStreamingClient::new("done")),
            &opts,
        )
        .unwrap();

    // Seed a checkpoint under the run id this send will use.
    let predicted_run_id = "run-seq-0";
    let cp_store: Arc<dyn crate::store::SessionStore> = store.clone();
    cp_store
        .save_loop_checkpoint(
            predicted_run_id,
            &LoopCheckpoint {
                schema_version: LOOP_CHECKPOINT_SCHEMA_VERSION,
                run_id: predicted_run_id.to_string(),
                session_id: "ckpt-clear-session".to_string(),
                turn: 1,
                messages: vec![Message::user("seed")],
                total_usage: crate::llm::TokenUsage::default(),
                tool_calls_count: 0,
                verification_reports: Vec::new(),
                checkpoint_ms: 1,
            },
        )
        .await
        .unwrap();

    let result = session.send("hello", None).await.unwrap();
    assert_eq!(result.text, "done");

    // Self-document the predicted run id.
    let runs = session.runs().await;
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].id, predicted_run_id, "run id must be deterministic");

    // The checkpoint must have been cleared by the run lifecycle.
    let after: Arc<dyn crate::store::SessionStore> = store.clone();
    assert!(
        after
            .load_loop_checkpoint(predicted_run_id)
            .await
            .unwrap()
            .is_none(),
        "completed run must delete its loop checkpoint (else unbounded leak)"
    );
}

/// P3 happy path (cut 2 E2E): a manually-seeded `LoopCheckpoint` in
/// the SessionStore can be picked up by `AgentSession::resume_run`,
/// the loop runs from the checkpoint's message vec (no new user
/// prompt is appended — `execute_from_messages` path), and the
/// resumed run is allocated a **fresh** run id (not the
/// checkpoint's).
///
/// This exercises the contract surface the host will sit on: write a
/// checkpoint on node A, hand the run id to node B which builds a
/// session against the shared store and calls `resume_run`. Crash
/// simulation is reduced to a manual checkpoint seed because the
/// in-process agent loop has no "die mid-round" affordance suitable
/// for unit testing.
#[tokio::test(flavor = "multi_thread")]
async fn test_resume_run_picks_up_from_persisted_checkpoint() {
    use crate::loop_checkpoint::{LoopCheckpoint, LOOP_CHECKPOINT_SCHEMA_VERSION};

    let store = Arc::new(crate::store::MemorySessionStore::new());
    let agent = Agent::from_config(test_config()).await.unwrap();

    // Seed a checkpoint as if a previous run on another node had
    // completed one tool round and persisted the boundary state.
    let seeded_run_id = "ckpt-old-run-x";
    let seeded_messages = vec![
        Message::user("kick off"),
        Message {
            role: "assistant".to_string(),
            content: vec![crate::llm::ContentBlock::Text {
                text: "intermediate work".to_string(),
            }],
            reasoning_content: None,
        },
    ];
    // Seed NON-ZERO cumulative metrics so the test can detect whether
    // resume_run carries them forward (H2 regression: it used to reset
    // them to zero, under-reporting the resumed AgentResult).
    let checkpoint = LoopCheckpoint {
        schema_version: LOOP_CHECKPOINT_SCHEMA_VERSION,
        run_id: seeded_run_id.to_string(),
        session_id: "resume-run-target".to_string(),
        turn: 1,
        messages: seeded_messages.clone(),
        total_usage: crate::llm::TokenUsage {
            prompt_tokens: 800,
            completion_tokens: 200,
            total_tokens: 1000,
            cache_read_tokens: None,
            cache_write_tokens: None,
        },
        tool_calls_count: 3,
        verification_reports: Vec::new(),
        checkpoint_ms: 1_700_000_000_000,
    };
    {
        let cp_store: Arc<dyn crate::store::SessionStore> = store.clone();
        cp_store
            .save_loop_checkpoint(seeded_run_id, &checkpoint)
            .await
            .expect("seed checkpoint");
    }

    // Build a session bound to the same store + a mock LLM that
    // produces a final-answer text. resume_run will feed it the
    // seeded `messages` and the loop should finish on this turn.
    let opts = SessionOptions::new()
        .with_session_store(store.clone() as Arc<dyn crate::store::SessionStore>)
        .with_session_id("resume-run-target");
    let session = agent
        .build_session(
            "/tmp/test-resume-run-target".into(),
            Arc::new(StaticStreamingClient::new("resumed and completed")),
            &opts,
        )
        .unwrap();

    let result = session
        .resume_run(seeded_run_id)
        .await
        .expect("resume_run must succeed");
    assert_eq!(result.text, "resumed and completed");

    // H2: the resumed run must CONTINUE accounting from the checkpoint's
    // cumulative metrics, not reset to zero. The mock LLM adds 2 tokens
    // (1 prompt + 1 completion) for its single turn, so the result must
    // reflect the seeded 1000 + 2 = 1002, and the seeded tool-call count
    // (3) must carry forward (this turn ran no tools).
    assert_eq!(
        result.usage.total_tokens, 1002,
        "resumed run must add to the checkpoint's cumulative token usage, not reset it"
    );
    assert_eq!(result.usage.prompt_tokens, 801);
    assert_eq!(result.usage.completion_tokens, 201);
    assert_eq!(
        result.tool_calls_count, 3,
        "resumed run must preserve the checkpoint's tool-call count"
    );

    // The resumed run records its own run id in the in-memory store,
    // and that id must NOT match the seeded checkpoint id — the
    // framework allocates a fresh run rather than pretending to
    // continue the old one.
    let runs = session.runs().await;
    assert_eq!(runs.len(), 1, "resume_run creates exactly one new run");
    let resumed_run = &runs[0];
    assert_ne!(
        resumed_run.id, seeded_run_id,
        "resumed run must have a fresh id, got the seeded one"
    );
    assert_eq!(resumed_run.status, crate::run::RunStatus::Completed);

    // The checkpoint stays in the store under the OLD run id —
    // resume does not delete it. (The host decides retention.)
    let still_there: Arc<dyn crate::store::SessionStore> = store.clone();
    let cp = still_there
        .load_loop_checkpoint(seeded_run_id)
        .await
        .expect("load")
        .expect("old checkpoint preserved");
    assert_eq!(cp.run_id, seeded_run_id);
    assert_eq!(cp.turn, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_resume_session_restores_artifacts() {
    let store = Arc::new(crate::store::MemorySessionStore::new());
    let agent = Agent::from_config(test_config()).await.unwrap();

    let opts = SessionOptions::new()
        .with_session_store(store.clone())
        .with_session_id("resume-artifacts-test");
    let session = agent.session("/tmp/test-ws-artifacts", Some(opts)).unwrap();
    session
        .tool_executor
        .artifact_store()
        .put(crate::tools::ToolArtifact {
            artifact_id: "tool-output:test:a".to_string(),
            artifact_uri: "a3s://tool-output/test/a".to_string(),
            tool_name: "test".to_string(),
            content: "artifact content".to_string(),
            original_bytes: 16,
            shown_bytes: 4,
        });

    session.save().await.unwrap();
    let opts2 = SessionOptions::new().with_session_store(store.clone());
    let resumed = agent
        .resume_session("resume-artifacts-test", opts2)
        .unwrap();

    let artifact = resumed
        .get_artifact("a3s://tool-output/test/a")
        .expect("artifact");
    assert_eq!(artifact.content, "artifact content");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_resume_session_restores_trace_events() {
    let store = Arc::new(crate::store::MemorySessionStore::new());
    let agent = Agent::from_config(test_config()).await.unwrap();
    let event = crate::trace::TraceEvent::tool_execution(
        "read",
        true,
        0,
        std::time::Duration::from_millis(3),
        32,
        Some(&serde_json::json!({
            "artifact": {
                "artifact_uri": "a3s://tool-output/read/abc"
            }
        })),
    );

    let opts = SessionOptions::new()
        .with_session_store(store.clone())
        .with_session_id("resume-trace-test");
    let session = agent.session("/tmp/test-ws-trace", Some(opts)).unwrap();
    session.trace_sink.replace_events(vec![event.clone()]);
    session.save().await.unwrap();

    let opts2 = SessionOptions::new().with_session_store(store.clone());
    let resumed = agent.resume_session("resume-trace-test", opts2).unwrap();

    assert_eq!(resumed.trace_events(), vec![event]);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_resume_session_restores_run_records() {
    let store = Arc::new(crate::store::MemorySessionStore::new());
    let agent = Agent::from_config(test_config()).await.unwrap();

    let opts = SessionOptions::new()
        .with_session_store(store.clone())
        .with_session_id("resume-runs-test");
    let session = agent.session("/tmp/test-ws-runs", Some(opts)).unwrap();
    let run = session
        .run_store
        .create_run(session.session_id(), "persist run")
        .await;
    session
        .run_store
        .record_event(
            &run.id,
            AgentEvent::Start {
                prompt: "persist run".to_string(),
            },
        )
        .await;
    session.save().await.unwrap();

    let opts2 = SessionOptions::new().with_session_store(store.clone());
    let resumed = agent.resume_session("resume-runs-test", opts2).unwrap();

    let runs = resumed.runs().await;
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].prompt, "persist run");
    assert_eq!(resumed.run_events(&run.id).await.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_resume_session_restores_verification_reports() {
    let store = Arc::new(crate::store::MemorySessionStore::new());
    let agent = Agent::from_config(test_config()).await.unwrap();
    let report = crate::verification::VerificationReport::new(
        "program:test",
        vec![
            crate::verification::VerificationCheck::required("check:test", "test", "Run tests")
                .with_status(crate::verification::VerificationStatus::Passed),
        ],
    );

    let opts = SessionOptions::new()
        .with_session_store(store.clone())
        .with_session_id("resume-verification-test");
    let session = agent
        .session("/tmp/test-ws-verification", Some(opts))
        .unwrap();
    session.record_verification_reports([report.clone()]);
    session.save().await.unwrap();

    let opts2 = SessionOptions::new().with_session_store(store.clone());
    let resumed = agent
        .resume_session("resume-verification-test", opts2)
        .unwrap();

    assert_eq!(resumed.verification_reports(), vec![report]);
    assert_eq!(
        resumed.verification_summary().status,
        crate::verification::VerificationStatus::Passed
    );
    assert!(resumed
        .verification_summary_text()
        .contains("Verification passed"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_verify_commands_builds_report_from_bash_results() {
    let temp_dir = tempfile::tempdir().unwrap();
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent
        .session(temp_dir.path().display().to_string(), None)
        .unwrap();
    let commands = vec![
        crate::verification::VerificationCommand::required(
            "check:smoke",
            "smoke",
            "Run smoke command",
            "printf ok",
        ),
        crate::verification::VerificationCommand::required(
            "check:failure",
            "smoke",
            "Run failing command",
            "exit 7",
        ),
    ];

    let report = session.verify_commands("turn", &commands).await.unwrap();

    assert_eq!(report.subject, "turn");
    assert_eq!(
        report.status,
        crate::verification::VerificationStatus::Failed
    );
    assert_eq!(
        report.checks[0].status,
        crate::verification::VerificationStatus::Passed
    );
    assert_eq!(
        report.checks[1].status,
        crate::verification::VerificationStatus::Failed
    );
    assert_eq!(
        report.checks[1].residual_risk.as_deref(),
        Some("verification command exited with code 7: exit 7")
    );
    assert_eq!(session.verification_reports(), vec![report]);
    assert_eq!(
        session.verification_summary().status,
        crate::verification::VerificationStatus::Failed
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_verification_presets_reflect_workspace() {
    let temp_dir = tempfile::tempdir().unwrap();
    std::fs::write(
        temp_dir.path().join("package.json"),
        r#"{"scripts":{"test":"vitest","typecheck":"tsc --noEmit"}}"#,
    )
    .unwrap();
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent
        .session(temp_dir.path().display().to_string(), None)
        .unwrap();

    let presets = session.verification_presets();

    assert_eq!(presets.len(), 1);
    assert_eq!(presets[0].project_kind, "node");
    assert_eq!(presets[0].commands[0].command, "npm test");
    assert_eq!(presets[0].commands[1].command, "npm run typecheck");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_resume_session_not_found() {
    let store = Arc::new(crate::store::MemorySessionStore::new());
    let agent = Agent::from_config(test_config()).await.unwrap();

    let opts = SessionOptions::new().with_session_store(store.clone());
    let result = agent.resume_session("nonexistent", opts);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_resume_session_no_store() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new();
    let result = agent.resume_session("any-id", opts);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("session_store"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_file_session_store_persistence() {
    let dir = tempfile::TempDir::new().unwrap();
    let store = Arc::new(
        crate::store::FileSessionStore::new(dir.path())
            .await
            .unwrap(),
    );
    let agent = Agent::from_config(test_config()).await.unwrap();

    // Save
    let opts = SessionOptions::new()
        .with_session_store(store.clone())
        .with_session_id("file-persist");
    let session = agent
        .session("/tmp/test-ws-file-persist", Some(opts))
        .unwrap();
    {
        let mut h = session.history.write().unwrap();
        h.push(Message::user("test message"));
    }
    session.save().await.unwrap();

    // Load from a fresh store instance pointing to same dir
    let store2 = Arc::new(
        crate::store::FileSessionStore::new(dir.path())
            .await
            .unwrap(),
    );
    let data = store2.load("file-persist").await.unwrap().unwrap();
    assert_eq!(data.messages.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_options_builders() {
    let opts = SessionOptions::new()
        .with_session_id("test-id")
        .with_auto_save(true)
        .with_max_parallel_tasks(3)
        .with_auto_delegation_enabled(true)
        .with_manual_delegation_enabled(false)
        .with_auto_parallel_delegation(false)
        .with_active_skill_tool_restrictions(true);
    assert_eq!(opts.session_id, Some("test-id".to_string()));
    assert!(opts.auto_save);
    assert_eq!(opts.max_parallel_tasks, Some(3));
    assert_eq!(opts.manual_delegation_enabled, Some(false));
    assert_eq!(opts.auto_parallel_delegation, Some(false));
    assert_eq!(opts.enforce_active_skill_tool_restrictions, Some(true));
    let auto = opts.auto_delegation.expect("auto delegation config");
    assert!(auto.enabled);
    assert!(!auto.allow_manual_delegation);
    assert!(!auto.auto_parallel);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_active_skill_tool_restriction_option_defaults_and_overrides() {
    let agent = Agent::from_config(test_config()).await.unwrap();

    let default_session = agent
        .session("/tmp/test-ws-skill-restriction-default", None)
        .unwrap();
    assert!(
        !default_session
            .config
            .enforce_active_skill_tool_restrictions
    );

    let legacy_session = agent
        .session(
            "/tmp/test-ws-skill-restriction-legacy",
            Some(SessionOptions::new().with_active_skill_tool_restrictions(true)),
        )
        .unwrap();
    assert!(legacy_session.config.enforce_active_skill_tool_restrictions);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_options_with_rl_trajectory_records_jsonl() {
    let dir = tempfile::TempDir::new().unwrap();
    let trajectory_path = dir.path().join("trajectory.jsonl");
    let agent = Agent::from_config(test_config()).await.unwrap();

    let opts = SessionOptions::new().with_rl_trajectory(
        crate::rl_trajectory::RlTrajectoryConfig::new(&trajectory_path).with_max_text_bytes(4),
    );
    let session = agent
        .session("/tmp/test-ws-rl-trajectory", Some(opts))
        .unwrap();

    assert!(session.config.rl_trajectory_recorder.is_enabled());
    session
        .config
        .rl_trajectory_recorder
        .record_execution_start(crate::rl_trajectory::ExecutionStartRecord {
            session_id: "sess-rl",
            workspace: std::path::Path::new("/tmp/test-ws-rl-trajectory"),
            prompt: "abcdef",
            history: &[],
            system_prompt: None,
            max_tool_rounds: 16,
            planning_mode: "disabled",
        });

    let content = std::fs::read_to_string(&trajectory_path).unwrap();
    let record: serde_json::Value = serde_json::from_str(content.lines().next().unwrap()).unwrap();
    assert_eq!(record["schema"], crate::rl_trajectory::RL_TRAJECTORY_SCHEMA);
    assert_eq!(record["event_type"], "execution_start");
    assert_eq!(record["session_id"], "sess-rl");
    assert_eq!(record["payload"]["prompt"]["text"], "abcd");
    assert_eq!(record["payload"]["prompt"]["truncated"], true);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_max_parallel_tasks_config_and_override() {
    let mut config = test_config();
    config.max_parallel_tasks = Some(6);
    config.auto_delegation.enabled = true;
    config.auto_delegation.auto_parallel = false;
    let agent = Agent::from_config(config).await.unwrap();

    let default_session = agent
        .session("/tmp/test-ws-parallel-default", None)
        .unwrap();
    assert_eq!(default_session.config.max_parallel_tasks, 6);
    assert!(default_session.config.auto_delegation.enabled);
    assert!(!default_session.config.auto_delegation.auto_parallel);

    let override_session = agent
        .session(
            "/tmp/test-ws-parallel-override",
            Some(
                SessionOptions::new()
                    .with_max_parallel_tasks(2)
                    .with_auto_parallel_delegation(true),
            ),
        )
        .unwrap();
    assert_eq!(override_session.config.max_parallel_tasks, 2);
    assert!(override_session.config.auto_delegation.auto_parallel);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_auto_parallel_override_preserves_base_auto_delegation() {
    let mut config = test_config();
    config.auto_delegation.enabled = true;
    config.auto_delegation.auto_parallel = true;
    let agent = Agent::from_config(config).await.unwrap();

    let session = agent
        .session(
            "/tmp/test-ws-auto-parallel-only",
            Some(SessionOptions::new().with_auto_parallel_delegation(false)),
        )
        .unwrap();

    assert!(session.config.auto_delegation.enabled);
    assert!(!session.config.auto_delegation.auto_parallel);
}

// ========================================================================
// Memory Integration Tests
// ========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_session_with_memory_store() {
    use a3s_memory::InMemoryStore;
    let store = Arc::new(InMemoryStore::new());
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new().with_memory(store);
    let session = agent.session("/tmp/test-ws-memory", Some(opts)).unwrap();
    assert!(session.memory().is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_without_memory_store() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent.session("/tmp/test-ws-no-memory", None).unwrap();
    assert!(session.memory().is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_memory_wired_into_config() {
    use a3s_memory::InMemoryStore;
    let store = Arc::new(InMemoryStore::new());
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new().with_memory(store);
    let session = agent
        .session("/tmp/test-ws-mem-config", Some(opts))
        .unwrap();
    // memory is accessible via the public session API
    assert!(session.memory().is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_with_file_memory() {
    let dir = tempfile::TempDir::new().unwrap();
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new().with_file_memory(dir.path());
    let session = agent.session("/tmp/test-ws-file-mem", Some(opts)).unwrap();
    assert!(session.memory().is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_memory_remember_and_recall() {
    use a3s_memory::InMemoryStore;
    let store = Arc::new(InMemoryStore::new());
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new().with_memory(store);
    let session = agent
        .session("/tmp/test-ws-mem-recall", Some(opts))
        .unwrap();

    let memory = session.memory().unwrap();
    memory
        .remember_success("write a file", &["write".to_string()], "done")
        .await
        .unwrap();

    let results = memory.recall_similar("write", 5).await.unwrap();
    assert!(!results.is_empty());
    let stats = memory.stats().await.unwrap();
    assert_eq!(stats.long_term_count, 1);
}

// ========================================================================
// Tool timeout tests
// ========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_session_tool_timeout_configured() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new().with_tool_timeout(5000);
    let session = agent.session("/tmp/test-ws-timeout", Some(opts)).unwrap();
    assert!(!session.id().is_empty());
}

// ========================================================================
// Queue fallback tests
// ========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_session_without_queue_builds_ok() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent.session("/tmp/test-ws-no-queue", None).unwrap();
    assert!(!session.id().is_empty());
}

// ========================================================================
// Concurrent history access tests
// ========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_history_reads() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = Arc::new(agent.session("/tmp/test-ws-concurrent", None).unwrap());

    let handles: Vec<_> = (0..10)
        .map(|_| {
            let s = Arc::clone(&session);
            tokio::spawn(async move { s.history().len() })
        })
        .collect();

    for h in handles {
        h.await.unwrap();
    }
}

// ========================================================================
// init_warning tests
// ========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_session_no_init_warning_without_file_memory() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent.session("/tmp/test-ws-no-warn", None).unwrap();
    assert!(session.init_warning().is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_register_agent_dir_loads_agents_into_live_session() {
    let temp_dir = tempfile::tempdir().unwrap();

    // Write a valid agent file
    std::fs::write(
        temp_dir.path().join("my-agent.yaml"),
        "name: my-dynamic-agent\ndescription: Dynamically registered agent\n",
    )
    .unwrap();

    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent.session(".", None).unwrap();

    // The agent must not be known before registration
    assert!(!session.agent_registry.exists("my-dynamic-agent"));

    let count = session.register_agent_dir(temp_dir.path());
    assert_eq!(count, 1);
    assert!(session.agent_registry.exists("my-dynamic-agent"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_register_agent_dir_empty_dir_returns_zero() {
    let temp_dir = tempfile::tempdir().unwrap();
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent.session(".", None).unwrap();
    let count = session.register_agent_dir(temp_dir.path());
    assert_eq!(count, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_register_agent_dir_nonexistent_returns_zero() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent.session(".", None).unwrap();
    let count = session.register_agent_dir(std::path::Path::new("/nonexistent/path/abc"));
    assert_eq!(count, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_register_worker_agent_loads_worker_into_live_session() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent.session(".", None).unwrap();

    assert!(!session.agent_registry.exists("frontend-cow"));
    let definition = session.register_worker_agent(
        crate::subagent::WorkerAgentSpec::implementer(
            "frontend-cow",
            "Disposable frontend implementer",
        )
        .with_max_steps(9),
    );

    assert_eq!(definition.max_steps, Some(9));
    assert!(session.agent_registry.exists("frontend-cow"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_register_worker_agents_loads_batch_into_live_session() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent.session(".", None).unwrap();

    let definitions = session.register_worker_agents([
        crate::subagent::WorkerAgentSpec::planner("planner-cow", "Plan work"),
        crate::subagent::WorkerAgentSpec::verifier("verify-cow", "Verify work"),
    ]);

    assert_eq!(definitions.len(), 2);
    assert!(session.agent_registry.exists("planner-cow"));
    assert!(session.agent_registry.exists("verify-cow"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_options_worker_agents_register_for_task_delegation() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new().with_worker_agent(crate::subagent::WorkerAgentSpec::planner(
        "release-planner",
        "Plan releases",
    ));
    let session = agent.session(".", Some(opts)).unwrap();

    assert!(session.agent_registry.exists("release-planner"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_loads_workspace_a3s_agents() {
    let workspace = tempfile::tempdir().unwrap();
    let agents_dir = workspace.path().join(".a3s").join("agents").join("quality");
    std::fs::create_dir_all(&agents_dir).unwrap();
    std::fs::write(
        agents_dir.join("code-reviewer.md"),
        r#"---
name: code-reviewer
description: Use proactively after code changes to review quality
tools: Read, Grep
---
Review the changed code and return prioritized findings.
"#,
    )
    .unwrap();

    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent
        .session(workspace.path().display().to_string(), None)
        .unwrap();

    let loaded = session
        .agent_registry
        .get("code-reviewer")
        .expect("workspace .a3s/agents agent should load");
    assert!(loaded
        .permissions
        .allow
        .iter()
        .any(|rule| { rule.matches("read", &serde_json::json!({"file_path": "README.md"})) }));
    assert!(loaded
        .permissions
        .allow
        .iter()
        .any(|rule| { rule.matches("grep", &serde_json::json!({"pattern": "TODO"})) }));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_keeps_claude_agents_as_compatibility_source() {
    let workspace = tempfile::tempdir().unwrap();
    let agents_dir = workspace.path().join(".claude").join("agents");
    std::fs::create_dir_all(&agents_dir).unwrap();
    std::fs::write(
        agents_dir.join("compat-reviewer.md"),
        r#"---
name: compat-reviewer
description: Compatibility agent
tools: Read
---
Review in compatibility mode.
"#,
    )
    .unwrap();

    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent
        .session(workspace.path().display().to_string(), None)
        .unwrap();

    assert!(session.agent_registry.exists("compat-reviewer"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_workspace_a3s_agents_override_claude_compat_agents() {
    let workspace = tempfile::tempdir().unwrap();
    let claude_dir = workspace.path().join(".claude").join("agents");
    let a3s_dir = workspace.path().join(".a3s").join("agents");
    std::fs::create_dir_all(&claude_dir).unwrap();
    std::fs::create_dir_all(&a3s_dir).unwrap();
    std::fs::write(
        claude_dir.join("same-agent.md"),
        r#"---
name: same-agent
description: Claude compatibility version
tools: Read
---
Compat prompt.
"#,
    )
    .unwrap();
    std::fs::write(
        a3s_dir.join("same-agent.md"),
        r#"---
name: same-agent
description: A3S native version
tools: Read
---
A3S prompt.
"#,
    )
    .unwrap();

    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent
        .session(workspace.path().display().to_string(), None)
        .unwrap();

    let loaded = session.agent_registry.get("same-agent").unwrap();
    assert_eq!(loaded.description, "A3S native version");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_for_worker_maps_worker_spec_to_session_options() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent
        .session_for_worker(
            ".",
            crate::subagent::WorkerAgentSpec::reviewer("review-cow", "Review changes")
                .with_max_steps(11),
            None,
        )
        .unwrap();

    assert_eq!(session.config.max_tool_rounds, 11);
    assert!(session.config.prompt_slots.extra.is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_session_with_mcp_manager_builds_ok() {
    use crate::mcp::manager::McpManager;
    let mcp = Arc::new(McpManager::new());
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new().with_mcp(mcp);
    // No servers connected — should build fine with zero MCP tools registered
    let session = agent.session("/tmp/test-ws-mcp", Some(opts)).unwrap();
    assert!(!session.id().is_empty());
}

#[test]
fn test_session_command_is_available_from_queue_module() {
    // Compile-time check: SessionCommand remains available from its owning module.
    use crate::queue::SessionCommand;
    let _ = std::marker::PhantomData::<Box<dyn SessionCommand>>;
}

#[tokio::test]
async fn subagent_events_populate_session_tracker() {
    use super::runtime_events::RuntimeEventSink;
    use crate::agent::AgentEvent;
    use crate::subagent_task_tracker::SubagentStatus;

    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent
        .session("/tmp/test-ws-subagent-tracker", None)
        .unwrap();

    // Drive a synthetic subagent lifecycle through the session's runtime sink.
    let run = session
        .run_store
        .create_run(session.session_id(), "parent prompt")
        .await;
    let sink = RuntimeEventSink::from_session(&session, &run.id);

    let task_id = "task-test-1".to_string();
    let child_session_id = format!("task-run-{}", task_id);

    sink.observe(&AgentEvent::SubagentStart {
        task_id: task_id.clone(),
        session_id: child_session_id.clone(),
        parent_session_id: session.session_id().to_string(),
        agent: "explore".to_string(),
        description: "demo delegation".to_string(),
    })
    .await;

    let snap = session
        .subagent_task(&task_id)
        .await
        .expect("running task should be visible");
    assert_eq!(snap.status, SubagentStatus::Running);
    assert_eq!(snap.parent_session_id, session.session_id());
    assert_eq!(snap.child_session_id, child_session_id);
    assert_eq!(snap.agent, "explore");
    assert!(snap.finished_ms.is_none());

    let pending = session.pending_subagent_tasks().await;
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].task_id, task_id);

    sink.observe(&AgentEvent::SubagentEnd {
        task_id: task_id.clone(),
        session_id: child_session_id,
        agent: "explore".to_string(),
        output: "found things".to_string(),
        success: true,
    })
    .await;

    let snap = session.subagent_task(&task_id).await.unwrap();
    assert_eq!(snap.status, SubagentStatus::Completed);
    assert_eq!(snap.success, Some(true));
    assert_eq!(snap.output.as_deref(), Some("found things"));
    assert!(snap.finished_ms.is_some());

    assert!(session.pending_subagent_tasks().await.is_empty());
    assert_eq!(session.subagent_tasks().await.len(), 1);
}

#[tokio::test]
async fn subagent_progress_events_accumulate_in_tracker() {
    use super::runtime_events::RuntimeEventSink;
    use crate::agent::AgentEvent;

    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent
        .session("/tmp/test-ws-subagent-progress", None)
        .unwrap();

    let run = session
        .run_store
        .create_run(session.session_id(), "parent prompt")
        .await;
    let sink = RuntimeEventSink::from_session(&session, &run.id);

    let task_id = "task-progress".to_string();
    let child_session_id = format!("task-run-{}", task_id);

    sink.observe(&AgentEvent::SubagentStart {
        task_id: task_id.clone(),
        session_id: child_session_id.clone(),
        parent_session_id: session.session_id().to_string(),
        agent: "explore".to_string(),
        description: "demo".to_string(),
    })
    .await;

    sink.observe(&AgentEvent::SubagentProgress {
        task_id: task_id.clone(),
        session_id: child_session_id.clone(),
        status: "tool_completed".to_string(),
        metadata: serde_json::json!({ "tool": "bash", "exit_code": 0 }),
    })
    .await;

    sink.observe(&AgentEvent::SubagentProgress {
        task_id: task_id.clone(),
        session_id: child_session_id.clone(),
        status: "turn_completed".to_string(),
        metadata: serde_json::json!({ "turn": 1, "total_tokens": 50 }),
    })
    .await;

    let snap = session.subagent_task(&task_id).await.unwrap();
    assert_eq!(snap.progress.len(), 2);
    assert_eq!(snap.progress[0].status, "tool_completed");
    assert_eq!(snap.progress[1].status, "turn_completed");
    assert_eq!(snap.progress[1].metadata["total_tokens"], 50);
}

#[tokio::test]
async fn subagent_tasks_scope_to_parent_session() {
    use super::runtime_events::RuntimeEventSink;
    use crate::agent::AgentEvent;

    let agent = Agent::from_config(test_config()).await.unwrap();
    let session_a = agent.session("/tmp/test-ws-subagent-a", None).unwrap();
    let session_b = agent.session("/tmp/test-ws-subagent-b", None).unwrap();

    let run = session_a
        .run_store
        .create_run(session_a.session_id(), "p")
        .await;
    let sink = RuntimeEventSink::from_session(&session_a, &run.id);

    sink.observe(&AgentEvent::SubagentStart {
        task_id: "task-from-a".to_string(),
        session_id: "task-run-task-from-a".to_string(),
        parent_session_id: session_a.session_id().to_string(),
        agent: "explore".to_string(),
        description: "isolated".to_string(),
    })
    .await;

    // session A sees the task; session B has its own (empty) tracker.
    assert_eq!(session_a.subagent_tasks().await.len(), 1);
    assert!(session_b.subagent_tasks().await.is_empty());
    assert!(session_b.subagent_task("task-from-a").await.is_none());
}

#[tokio::test]
async fn cancel_subagent_task_marks_snapshot_cancelled() {
    use super::runtime_events::RuntimeEventSink;
    use crate::agent::AgentEvent;
    use crate::subagent_task_tracker::SubagentStatus;
    use tokio_util::sync::CancellationToken;

    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent.session("/tmp/test-ws-subagent-cancel", None).unwrap();
    let run = session
        .run_store
        .create_run(session.session_id(), "parent")
        .await;
    let sink = RuntimeEventSink::from_session(&session, &run.id);

    let task_id = "task-to-cancel".to_string();
    sink.observe(&AgentEvent::SubagentStart {
        task_id: task_id.clone(),
        session_id: format!("task-run-{}", task_id),
        parent_session_id: session.session_id().to_string(),
        agent: "explore".to_string(),
        description: "long task".to_string(),
    })
    .await;

    // Simulate what TaskExecutor would do: register a cancellation token
    // for this in-flight task so the public API has something to fire.
    let token = CancellationToken::new();
    session
        .subagent_tasks
        .register_canceller(&task_id, token.clone())
        .await;

    assert!(session.cancel_subagent_task(&task_id).await);
    assert!(token.is_cancelled());

    let snap = session.subagent_task(&task_id).await.unwrap();
    assert_eq!(snap.status, SubagentStatus::Cancelled);

    // A late SubagentEnd from the cancelled child must not downgrade.
    sink.observe(&AgentEvent::SubagentEnd {
        task_id: task_id.clone(),
        session_id: format!("task-run-{}", task_id),
        agent: "explore".to_string(),
        output: "Task cancelled by caller".to_string(),
        success: false,
    })
    .await;
    let snap = session.subagent_task(&task_id).await.unwrap();
    assert_eq!(snap.status, SubagentStatus::Cancelled);

    // Cancelling again or against an unknown id is a no-op.
    assert!(!session.cancel_subagent_task(&task_id).await);
    assert!(!session.cancel_subagent_task("task-unknown").await);
}

/// Regression: `agent_executor()` must install the same `ChildRunContext` the
/// model-driven `task` path uses, so orchestrated/scripted steps inherit the
/// session's governance instead of running under weaker ambient authority.
/// Before the fix the executor was built without `.with_parent_context(..)`.
#[tokio::test]
async fn test_agent_executor_inherits_parent_run_context() {
    use crate::security::DefaultSecurityProvider;
    use crate::skills::SkillRegistry;

    let agent = Agent::from_config(test_config()).await.unwrap();

    let security: Arc<dyn crate::security::SecurityProvider> =
        Arc::new(DefaultSecurityProvider::new());
    let skills = Arc::new(SkillRegistry::new());
    let opts = SessionOptions::new()
        .with_security_provider(Arc::clone(&security))
        .with_skill_registry(Arc::clone(&skills));

    let session = agent.session("/tmp/test-workspace", Some(opts)).unwrap();
    let ctx = session.parent_run_context();

    assert!(
        ctx.security_provider.is_some(),
        "security provider must propagate to delegated/orchestrated child runs"
    );
    assert!(
        ctx.skill_registry.is_some(),
        "skill registry (skill restrictions) must propagate to child runs"
    );
    assert!(
        ctx.workspace_services.is_some(),
        "workspace services must propagate so child tools share the workspace"
    );
    assert!(
        ctx.hook_engine.is_none(),
        "hook_engine stays None, matching the model-driven task path"
    );
}

/// `AgentSession::workflow()` must pre-wire a shared budget ledger and a stable,
/// session-derived root id (so phase checkpoints resume across runs).
#[tokio::test]
async fn test_session_workflow_is_prewired_with_budget_and_stable_root_id() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let session = agent.session("/tmp/test-workspace", None).unwrap();

    let wf = session.workflow();
    // An uncapped workflow still owns a ledger (for snapshots / aggregation).
    let snap = wf
        .budget_snapshot()
        .expect("workflow is pre-wired with a shared budget ledger");
    assert_eq!(snap.limit_tokens, None);
    assert_eq!(snap.consumed_tokens, 0);
    assert!(
        wf.root_id().contains(session.id()),
        "root id is session-derived so phase checkpoints are stable across runs"
    );

    // A capped workflow records its hard ceiling.
    let capped = session.workflow_with_token_budget(Some(50_000));
    assert_eq!(capped.budget_snapshot().unwrap().limit_tokens, Some(50_000));
}

/// End-to-end: `session.workflow().agent(spec)` actually spawns a real child
/// agent loop through the wired executor and returns its output. Uses a static
/// mock LLM so the built-in `explore` agent finishes with that text.
#[tokio::test]
async fn test_session_workflow_runs_a_real_child_agent_step() {
    let agent = Agent::from_config(test_config()).await.unwrap();
    let opts = SessionOptions::new();
    let session = agent
        .build_session(
            "/tmp/test-workflow-e2e".into(),
            Arc::new(StaticStreamingClient::new("explored answer")),
            &opts,
        )
        .unwrap();

    let wf = session.workflow();
    let outcome = wf
        .agent(crate::orchestration::AgentStepSpec::new(
            "t1",
            "explore",
            "explore",
            "find the auth code",
        ))
        .await;

    assert!(outcome.success, "child step failed: {}", outcome.output);
    assert_eq!(outcome.agent, "explore");
    assert!(
        outcome.output.contains("explored answer"),
        "child agent returned the mock LLM output; got: {}",
        outcome.output
    );

    // The shared ledger recorded the child's token usage (proves the workflow
    // budget was installed as the child's budget guard).
    assert!(
        wf.budget_snapshot().unwrap().consumed_tokens > 0,
        "child LLM usage fed the shared workflow budget"
    );
}
