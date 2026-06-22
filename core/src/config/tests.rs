use super::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[test]
fn test_config_default() {
    let config = CodeConfig::default();
    assert!(config.skill_dirs.is_empty());
    assert!(config.agent_dirs.is_empty());
    assert!(config.providers.is_empty());
    assert!(config.default_model.is_none());
    assert_eq!(config.storage_backend, StorageBackend::File);
    assert!(config.sessions_dir.is_none());
}

#[test]
fn test_storage_backend_default() {
    let backend = StorageBackend::default();
    assert_eq!(backend, StorageBackend::File);
}

#[test]
fn test_storage_backend_serde() {
    // Test serialization
    let memory = StorageBackend::Memory;
    let json = serde_json::to_string(&memory).unwrap();
    assert_eq!(json, "\"memory\"");

    let file = StorageBackend::File;
    let json = serde_json::to_string(&file).unwrap();
    assert_eq!(json, "\"file\"");

    // Test deserialization
    let memory: StorageBackend = serde_json::from_str("\"memory\"").unwrap();
    assert_eq!(memory, StorageBackend::Memory);

    let file: StorageBackend = serde_json::from_str("\"file\"").unwrap();
    assert_eq!(file, StorageBackend::File);
}

#[test]
fn test_config_with_storage_backend() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config_path = temp_dir.path().join("config.acl");

    std::fs::write(
        &config_path,
        r#"
            storage_backend = "memory"
            sessions_dir = "/tmp/sessions"
            max_parallel_tasks = 3
            auto_parallel = false

            auto_delegation {
              enabled = true
              auto_parallel = true
              allow_manual_delegation = false
              min_confidence = 0.8
              max_tasks = 2
            }
        "#,
    )
    .unwrap();

    let config = CodeConfig::from_file(&config_path).unwrap();
    assert_eq!(config.storage_backend, StorageBackend::Memory);
    assert_eq!(config.sessions_dir, Some(PathBuf::from("/tmp/sessions")));
    assert_eq!(config.max_parallel_tasks, Some(3));
    assert!(config.auto_delegation.enabled);
    assert!(!config.auto_delegation.auto_parallel);
    assert!(!config.auto_delegation.allow_manual_delegation);
    assert!((config.auto_delegation.min_confidence - 0.8).abs() < f32::EPSILON);
    assert_eq!(config.auto_delegation.max_tasks, 2);
}

#[test]
fn test_config_rejects_unlabeled_provider_blocks() {
    std::env::set_var("A3S_CODE_TEST_API_KEY", "sk-test");
    let err = CodeConfig::from_acl(
        r#"
            default_model = "openai/gpt-4.1"
            max_tool_rounds = 12
            skill_dirs = ["./skills"]

            providers {
              name     = "openai"
              api_key  = env("A3S_CODE_TEST_API_KEY")
              base_url = "https://api.openai.com/v1"

              models "gpt-4.1" {
                name      = "GPT 4.1"
                reasoning = true
                tool_call = false
              }
            }
        "#,
    )
    .unwrap_err();

    assert!(err.to_string().contains("providers block requires a label"));
}

#[test]
fn test_config_supports_acl_style_provider_labels() {
    let config = CodeConfig::from_acl(
        r#"
            default_model = "openai/gpt-4.1"

            providers "openai" {
              apiKey  = "sk-test"
              baseUrl = "https://api.openai.com/v1"

              models "gpt-4.1" {
                name     = "GPT 4.1"
                toolCall = true
              }
            }
        "#,
    )
    .unwrap();

    assert_eq!(config.default_model.as_deref(), Some("openai/gpt-4.1"));
    assert_eq!(config.providers[0].name, "openai");
    assert_eq!(config.providers[0].api_key.as_deref(), Some("sk-test"));
    assert_eq!(config.providers[0].models[0].id, "gpt-4.1");
    assert_eq!(config.providers[0].models[0].name, "GPT 4.1");
    assert!(config.providers[0].models[0].tool_call);
}

#[test]
fn test_config_parses_acl_model_token_limits() {
    let config = CodeConfig::from_acl(
        r#"
            default_model = "openai/glm-5.1"

            providers "openai" {
              api_key = "sk-test"
              base_url = "http://127.0.0.1:18051/v1"

              models "glm-5.1" {
                name = "GLM 5.1"
                limit = {
                  output = 4096
                  context = 32768
                }
              }

              models "flat-alias" {
                maxTokens = 8192
                contextTokens = 65536
              }
            }
        "#,
    )
    .unwrap();

    let flat = &config.providers[0].models[0].limit;
    assert_eq!(flat.output, 4096);
    assert_eq!(flat.context, 32768);

    let flat_alias = &config.providers[0].models[1].limit;
    assert_eq!(flat_alias.output, 8192);
    assert_eq!(flat_alias.context, 65536);
}

#[test]
fn test_config_parses_acl_model_token_limit_edges() {
    let config = CodeConfig::from_acl(
        r#"
            providers "openai" {
              api_key = "sk-test"

              models "partial-limit" {
                limit = {
                  context = 2048
                }
              }

              models "legacy-large-limit" {
                maxTokens = 5000000000
                contextTokens = -1
              }
            }
        "#,
    )
    .unwrap();

    let partial = &config.providers[0].models[0].limit;
    assert_eq!(partial.output, 0);
    assert_eq!(partial.context, 2048);

    let legacy_large = &config.providers[0].models[1].limit;
    assert_eq!(legacy_large.output, u32::MAX);
    assert_eq!(legacy_large.context, 0);
}

#[test]
fn test_acl_model_output_limit_flows_to_llm_config() {
    let config = CodeConfig::from_acl(
        r#"
            default_model = "openai/glm-5.1"

            providers "openai" {
              api_key = "sk-test"

              models "glm-5.1" {
                limit = {
                  output = 1234
                  context = 32768
                }
              }
            }
        "#,
    )
    .unwrap();

    let llm_config = config.default_llm_config().unwrap();
    assert_eq!(llm_config.max_tokens, Some(1234));
}

#[test]
fn test_config_builder() {
    let config = CodeConfig::new()
        .add_skill_dir("/tmp/skills")
        .add_agent_dir("/tmp/agents");

    assert_eq!(config.skill_dirs.len(), 1);
    assert_eq!(config.agent_dirs.len(), 1);
}

#[test]
fn test_find_provider() {
    let config = CodeConfig {
        providers: vec![
            ProviderConfig {
                name: "anthropic".to_string(),
                api_key: Some("key1".to_string()),
                base_url: None,
                headers: HashMap::new(),
                session_id_header: None,
                models: vec![],
            },
            ProviderConfig {
                name: "openai".to_string(),
                api_key: Some("key2".to_string()),
                base_url: None,
                headers: HashMap::new(),
                session_id_header: None,
                models: vec![],
            },
        ],
        ..Default::default()
    };

    assert!(config.find_provider("anthropic").is_some());
    assert!(config.find_provider("openai").is_some());
    assert!(config.find_provider("unknown").is_none());
}

#[test]
fn test_default_llm_config() {
    let config = CodeConfig {
        default_model: Some("anthropic/claude-sonnet-4".to_string()),
        providers: vec![ProviderConfig {
            name: "anthropic".to_string(),
            api_key: Some("test-api-key".to_string()),
            base_url: Some("https://api.anthropic.com".to_string()),
            headers: HashMap::new(),
            session_id_header: None,
            models: vec![ModelConfig {
                id: "claude-sonnet-4".to_string(),
                name: "Claude Sonnet 4".to_string(),
                family: "claude-sonnet".to_string(),
                api_key: None,
                base_url: None,
                headers: HashMap::new(),
                session_id_header: None,
                attachment: false,
                reasoning: false,
                tool_call: true,
                temperature: true,
                release_date: None,
                modalities: ModelModalities::default(),
                cost: ModelCost::default(),
                limit: ModelLimit::default(),
            }],
        }],
        ..Default::default()
    };

    let llm_config = config.default_llm_config().unwrap();
    assert_eq!(llm_config.provider, "anthropic");
    assert_eq!(llm_config.model, "claude-sonnet-4");
    assert_eq!(llm_config.api_key.expose(), "test-api-key");
    assert_eq!(
        llm_config.base_url,
        Some("https://api.anthropic.com".to_string())
    );
}

#[test]
fn test_model_api_key_override() {
    let provider = ProviderConfig {
        name: "openai".to_string(),
        api_key: Some("provider-key".to_string()),
        base_url: Some("https://api.openai.com".to_string()),
        headers: HashMap::new(),
        session_id_header: None,
        models: vec![
            ModelConfig {
                id: "gpt-4".to_string(),
                name: "GPT-4".to_string(),
                family: "gpt".to_string(),
                api_key: None, // Uses provider key
                base_url: None,
                headers: HashMap::new(),
                session_id_header: None,
                attachment: false,
                reasoning: false,
                tool_call: true,
                temperature: true,
                release_date: None,
                modalities: ModelModalities::default(),
                cost: ModelCost::default(),
                limit: ModelLimit::default(),
            },
            ModelConfig {
                id: "custom-model".to_string(),
                name: "Custom Model".to_string(),
                family: "custom".to_string(),
                api_key: Some("model-specific-key".to_string()), // Override
                base_url: Some("https://custom.api.com".to_string()), // Override
                headers: HashMap::new(),
                session_id_header: None,
                attachment: false,
                reasoning: false,
                tool_call: true,
                temperature: true,
                release_date: None,
                modalities: ModelModalities::default(),
                cost: ModelCost::default(),
                limit: ModelLimit::default(),
            },
        ],
    };

    // Model without override uses provider key
    let model1 = provider.find_model("gpt-4").unwrap();
    assert_eq!(provider.get_api_key(model1), Some("provider-key"));
    assert_eq!(
        provider.get_base_url(model1),
        Some("https://api.openai.com")
    );

    // Model with override uses its own key
    let model2 = provider.find_model("custom-model").unwrap();
    assert_eq!(provider.get_api_key(model2), Some("model-specific-key"));
    assert_eq!(
        provider.get_base_url(model2),
        Some("https://custom.api.com")
    );
}

#[test]
fn test_list_models() {
    let config = CodeConfig {
        providers: vec![
            ProviderConfig {
                name: "anthropic".to_string(),
                api_key: None,
                base_url: None,
                headers: HashMap::new(),
                session_id_header: None,
                models: vec![
                    ModelConfig {
                        id: "claude-1".to_string(),
                        name: "Claude 1".to_string(),
                        family: "claude".to_string(),
                        api_key: None,
                        base_url: None,
                        headers: HashMap::new(),
                        session_id_header: None,
                        attachment: false,
                        reasoning: false,
                        tool_call: true,
                        temperature: true,
                        release_date: None,
                        modalities: ModelModalities::default(),
                        cost: ModelCost::default(),
                        limit: ModelLimit::default(),
                    },
                    ModelConfig {
                        id: "claude-2".to_string(),
                        name: "Claude 2".to_string(),
                        family: "claude".to_string(),
                        api_key: None,
                        base_url: None,
                        headers: HashMap::new(),
                        session_id_header: None,
                        attachment: false,
                        reasoning: false,
                        tool_call: true,
                        temperature: true,
                        release_date: None,
                        modalities: ModelModalities::default(),
                        cost: ModelCost::default(),
                        limit: ModelLimit::default(),
                    },
                ],
            },
            ProviderConfig {
                name: "openai".to_string(),
                api_key: None,
                base_url: None,
                headers: HashMap::new(),
                session_id_header: None,
                models: vec![ModelConfig {
                    id: "gpt-4".to_string(),
                    name: "GPT-4".to_string(),
                    family: "gpt".to_string(),
                    api_key: None,
                    base_url: None,
                    headers: HashMap::new(),
                    session_id_header: None,
                    attachment: false,
                    reasoning: false,
                    tool_call: true,
                    temperature: true,
                    release_date: None,
                    modalities: ModelModalities::default(),
                    cost: ModelCost::default(),
                    limit: ModelLimit::default(),
                }],
            },
        ],
        ..Default::default()
    };

    let models = config.list_models();
    assert_eq!(models.len(), 3);
}

#[test]
fn test_config_from_file_not_found() {
    let result = CodeConfig::from_file(Path::new("/nonexistent/config.json"));
    assert!(result.is_err());
}

#[test]
fn test_config_has_directories() {
    let empty = CodeConfig::default();
    assert!(!empty.has_directories());

    let with_skills = CodeConfig::new().add_skill_dir("/tmp/skills");
    assert!(with_skills.has_directories());

    let with_agents = CodeConfig::new().add_agent_dir("/tmp/agents");
    assert!(with_agents.has_directories());
}

#[test]
fn test_config_has_providers() {
    let empty = CodeConfig::default();
    assert!(!empty.has_providers());

    let with_providers = CodeConfig {
        providers: vec![ProviderConfig {
            name: "test".to_string(),
            api_key: None,
            base_url: None,
            headers: HashMap::new(),
            session_id_header: None,
            models: vec![],
        }],
        ..Default::default()
    };
    assert!(with_providers.has_providers());
}

#[test]
fn test_storage_backend_equality() {
    assert_eq!(StorageBackend::Memory, StorageBackend::Memory);
    assert_eq!(StorageBackend::File, StorageBackend::File);
    assert_ne!(StorageBackend::Memory, StorageBackend::File);
}

#[test]
fn test_storage_backend_serde_custom() {
    let custom = StorageBackend::Custom;
    // Custom variant is now serializable
    let json = serde_json::to_string(&custom).unwrap();
    assert_eq!(json, "\"custom\"");

    // And deserializable
    let parsed: StorageBackend = serde_json::from_str("\"custom\"").unwrap();
    assert_eq!(parsed, StorageBackend::Custom);
}

#[test]
fn test_model_cost_default() {
    let cost = ModelCost::default();
    assert_eq!(cost.input, 0.0);
    assert_eq!(cost.output, 0.0);
    assert_eq!(cost.cache_read, 0.0);
    assert_eq!(cost.cache_write, 0.0);
}

#[test]
fn test_model_cost_serialization() {
    let cost = ModelCost {
        input: 3.0,
        output: 15.0,
        cache_read: 0.3,
        cache_write: 3.75,
    };
    let json = serde_json::to_string(&cost).unwrap();
    assert!(json.contains("\"input\":3"));
    assert!(json.contains("\"output\":15"));
}

#[test]
fn test_model_cost_deserialization_missing_fields() {
    let json = r#"{"input":3.0}"#;
    let cost: ModelCost = serde_json::from_str(json).unwrap();
    assert_eq!(cost.input, 3.0);
    assert_eq!(cost.output, 0.0);
    assert_eq!(cost.cache_read, 0.0);
    assert_eq!(cost.cache_write, 0.0);
}

#[test]
fn test_model_limit_default() {
    let limit = ModelLimit::default();
    assert_eq!(limit.context, 0);
    assert_eq!(limit.output, 0);
}

#[test]
fn test_model_limit_serialization() {
    let limit = ModelLimit {
        context: 200000,
        output: 8192,
    };
    let json = serde_json::to_string(&limit).unwrap();
    assert!(json.contains("\"context\":200000"));
    assert!(json.contains("\"output\":8192"));
}

#[test]
fn test_model_limit_deserialization_missing_fields() {
    let json = r#"{"context":100000}"#;
    let limit: ModelLimit = serde_json::from_str(json).unwrap();
    assert_eq!(limit.context, 100000);
    assert_eq!(limit.output, 0);
}

#[test]
fn test_model_modalities_default() {
    let modalities = ModelModalities::default();
    assert!(modalities.input.is_empty());
    assert!(modalities.output.is_empty());
}

#[test]
fn test_model_modalities_serialization() {
    let modalities = ModelModalities {
        input: vec!["text".to_string(), "image".to_string()],
        output: vec!["text".to_string()],
    };
    let json = serde_json::to_string(&modalities).unwrap();
    assert!(json.contains("\"input\""));
    assert!(json.contains("\"text\""));
}

#[test]
fn test_model_modalities_deserialization_missing_fields() {
    let json = r#"{"input":["text"]}"#;
    let modalities: ModelModalities = serde_json::from_str(json).unwrap();
    assert_eq!(modalities.input.len(), 1);
    assert!(modalities.output.is_empty());
}

#[test]
fn test_model_config_serialization() {
    let config = ModelConfig {
        id: "gpt-4o".to_string(),
        name: "GPT-4o".to_string(),
        family: "gpt-4".to_string(),
        api_key: Some("sk-test".to_string()),
        base_url: None,
        headers: HashMap::new(),
        session_id_header: None,
        attachment: true,
        reasoning: false,
        tool_call: true,
        temperature: true,
        release_date: Some("2024-05-13".to_string()),
        modalities: ModelModalities::default(),
        cost: ModelCost::default(),
        limit: ModelLimit::default(),
    };
    let json = serde_json::to_string(&config).unwrap();
    assert!(json.contains("\"id\":\"gpt-4o\""));
    assert!(json.contains("\"attachment\":true"));
}

#[test]
fn test_model_config_deserialization_with_defaults() {
    let json = r#"{"id":"test-model"}"#;
    let config: ModelConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.id, "test-model");
    assert_eq!(config.name, "");
    assert_eq!(config.family, "");
    assert!(config.api_key.is_none());
    assert!(!config.attachment);
    assert!(config.tool_call);
    assert!(config.temperature);
}

#[test]
fn test_model_config_all_optional_fields() {
    let json = r#"{
        "id": "claude-sonnet-4",
        "name": "Claude Sonnet 4",
        "family": "claude-sonnet",
        "apiKey": "sk-test",
        "baseUrl": "https://api.anthropic.com",
        "attachment": true,
        "reasoning": true,
        "toolCall": false,
        "temperature": false,
        "releaseDate": "2025-05-14"
    }"#;
    let config: ModelConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.id, "claude-sonnet-4");
    assert_eq!(config.name, "Claude Sonnet 4");
    assert_eq!(config.api_key, Some("sk-test".to_string()));
    assert_eq!(
        config.base_url,
        Some("https://api.anthropic.com".to_string())
    );
    assert!(config.attachment);
    assert!(config.reasoning);
    assert!(!config.tool_call);
    assert!(!config.temperature);
}

#[test]
fn test_provider_config_serialization() {
    let provider = ProviderConfig {
        name: "anthropic".to_string(),
        api_key: Some("sk-test".to_string()),
        base_url: Some("https://api.anthropic.com".to_string()),
        headers: HashMap::new(),
        session_id_header: None,
        models: vec![],
    };
    let json = serde_json::to_string(&provider).unwrap();
    assert!(json.contains("\"name\":\"anthropic\""));
    assert!(json.contains("\"apiKey\":\"sk-test\""));
}

#[test]
fn test_provider_config_deserialization_missing_optional() {
    let json = r#"{"name":"openai"}"#;
    let provider: ProviderConfig = serde_json::from_str(json).unwrap();
    assert_eq!(provider.name, "openai");
    assert!(provider.api_key.is_none());
    assert!(provider.base_url.is_none());
    assert!(provider.models.is_empty());
}

#[test]
fn test_provider_config_find_model() {
    let provider = ProviderConfig {
        name: "anthropic".to_string(),
        api_key: None,
        base_url: None,
        headers: HashMap::new(),
        session_id_header: None,
        models: vec![ModelConfig {
            id: "claude-sonnet-4".to_string(),
            name: "Claude Sonnet 4".to_string(),
            family: "claude-sonnet".to_string(),
            api_key: None,
            base_url: None,
            headers: HashMap::new(),
            session_id_header: None,
            attachment: false,
            reasoning: false,
            tool_call: true,
            temperature: true,
            release_date: None,
            modalities: ModelModalities::default(),
            cost: ModelCost::default(),
            limit: ModelLimit::default(),
        }],
    };

    let found = provider.find_model("claude-sonnet-4");
    assert!(found.is_some());
    assert_eq!(found.unwrap().id, "claude-sonnet-4");

    let not_found = provider.find_model("gpt-4o");
    assert!(not_found.is_none());
}

#[test]
fn test_provider_config_get_api_key() {
    let provider = ProviderConfig {
        name: "anthropic".to_string(),
        api_key: Some("provider-key".to_string()),
        base_url: None,
        headers: HashMap::new(),
        session_id_header: None,
        models: vec![],
    };

    let model_with_key = ModelConfig {
        id: "test".to_string(),
        name: "".to_string(),
        family: "".to_string(),
        api_key: Some("model-key".to_string()),
        base_url: None,
        headers: HashMap::new(),
        session_id_header: None,
        attachment: false,
        reasoning: false,
        tool_call: true,
        temperature: true,
        release_date: None,
        modalities: ModelModalities::default(),
        cost: ModelCost::default(),
        limit: ModelLimit::default(),
    };

    let model_without_key = ModelConfig {
        id: "test2".to_string(),
        name: "".to_string(),
        family: "".to_string(),
        api_key: None,
        base_url: None,
        headers: HashMap::new(),
        session_id_header: None,
        attachment: false,
        reasoning: false,
        tool_call: true,
        temperature: true,
        release_date: None,
        modalities: ModelModalities::default(),
        cost: ModelCost::default(),
        limit: ModelLimit::default(),
    };

    assert_eq!(provider.get_api_key(&model_with_key), Some("model-key"));
    assert_eq!(
        provider.get_api_key(&model_without_key),
        Some("provider-key")
    );
}

#[test]
fn test_provider_config_get_headers_and_session_id_header() {
    let mut provider_headers = HashMap::new();
    provider_headers.insert("X-Provider".to_string(), "provider".to_string());
    provider_headers.insert("X-Shared".to_string(), "provider".to_string());

    let mut model_headers = HashMap::new();
    model_headers.insert("X-Model".to_string(), "model".to_string());
    model_headers.insert("X-Shared".to_string(), "model".to_string());

    let provider = ProviderConfig {
        name: "openai".to_string(),
        api_key: Some("provider-key".to_string()),
        base_url: None,
        headers: provider_headers,
        session_id_header: Some("X-Session-Id".to_string()),
        models: vec![],
    };

    let model = ModelConfig {
        id: "gpt-4o".to_string(),
        name: "".to_string(),
        family: "".to_string(),
        api_key: None,
        base_url: None,
        headers: model_headers,
        session_id_header: Some("X-Model-Session".to_string()),
        attachment: false,
        reasoning: false,
        tool_call: true,
        temperature: true,
        release_date: None,
        modalities: ModelModalities::default(),
        cost: ModelCost::default(),
        limit: ModelLimit::default(),
    };

    let headers = provider.get_headers(&model);
    assert_eq!(headers.get("X-Provider"), Some(&"provider".to_string()));
    assert_eq!(headers.get("X-Model"), Some(&"model".to_string()));
    assert_eq!(headers.get("X-Shared"), Some(&"model".to_string()));
    assert_eq!(
        provider.get_session_id_header(&model),
        Some("X-Model-Session")
    );
}

#[test]
fn test_llm_config_includes_headers_and_runtime_session_header() {
    let mut provider_headers = HashMap::new();
    provider_headers.insert("X-Provider".to_string(), "provider".to_string());

    let config = CodeConfig {
        default_model: Some("openai/gpt-4o".to_string()),
        providers: vec![ProviderConfig {
            name: "openai".to_string(),
            api_key: Some("sk-test".to_string()),
            base_url: Some("https://api.example.com".to_string()),
            headers: provider_headers,
            session_id_header: Some("X-Session-Id".to_string()),
            models: vec![ModelConfig {
                id: "gpt-4o".to_string(),
                name: "".to_string(),
                family: "".to_string(),
                api_key: None,
                base_url: None,
                headers: HashMap::new(),
                session_id_header: None,
                attachment: false,
                reasoning: false,
                tool_call: true,
                temperature: true,
                release_date: None,
                modalities: ModelModalities::default(),
                cost: ModelCost::default(),
                limit: ModelLimit::default(),
            }],
        }],
        ..Default::default()
    };

    let llm_config = config.default_llm_config().unwrap();
    assert_eq!(
        llm_config.headers.get("X-Provider"),
        Some(&"provider".to_string())
    );
    assert_eq!(
        llm_config.session_id_header.as_deref(),
        Some("X-Session-Id")
    );
}

#[test]
fn test_code_config_default_provider_config() {
    let config = CodeConfig {
        default_model: Some("anthropic/claude-sonnet-4".to_string()),
        providers: vec![ProviderConfig {
            name: "anthropic".to_string(),
            api_key: Some("sk-test".to_string()),
            base_url: None,
            headers: HashMap::new(),
            session_id_header: None,
            models: vec![],
        }],
        ..Default::default()
    };

    let provider = config.default_provider_config();
    assert!(provider.is_some());
    assert_eq!(provider.unwrap().name, "anthropic");
}

#[test]
fn test_code_config_default_model_config() {
    let config = CodeConfig {
        default_model: Some("anthropic/claude-sonnet-4".to_string()),
        providers: vec![ProviderConfig {
            name: "anthropic".to_string(),
            api_key: Some("sk-test".to_string()),
            base_url: None,
            headers: HashMap::new(),
            session_id_header: None,
            models: vec![ModelConfig {
                id: "claude-sonnet-4".to_string(),
                name: "Claude Sonnet 4".to_string(),
                family: "claude-sonnet".to_string(),
                api_key: None,
                base_url: None,
                headers: HashMap::new(),
                session_id_header: None,
                attachment: false,
                reasoning: false,
                tool_call: true,
                temperature: true,
                release_date: None,
                modalities: ModelModalities::default(),
                cost: ModelCost::default(),
                limit: ModelLimit::default(),
            }],
        }],
        ..Default::default()
    };

    let result = config.default_model_config();
    assert!(result.is_some());
    let (provider, model) = result.unwrap();
    assert_eq!(provider.name, "anthropic");
    assert_eq!(model.id, "claude-sonnet-4");
}

#[test]
fn test_code_config_default_llm_config() {
    let config = CodeConfig {
        default_model: Some("anthropic/claude-sonnet-4".to_string()),
        providers: vec![ProviderConfig {
            name: "anthropic".to_string(),
            api_key: Some("sk-test".to_string()),
            base_url: Some("https://api.anthropic.com".to_string()),
            headers: HashMap::new(),
            session_id_header: None,
            models: vec![ModelConfig {
                id: "claude-sonnet-4".to_string(),
                name: "Claude Sonnet 4".to_string(),
                family: "claude-sonnet".to_string(),
                api_key: None,
                base_url: None,
                headers: HashMap::new(),
                session_id_header: None,
                attachment: false,
                reasoning: false,
                tool_call: true,
                temperature: true,
                release_date: None,
                modalities: ModelModalities::default(),
                cost: ModelCost::default(),
                limit: ModelLimit::default(),
            }],
        }],
        ..Default::default()
    };

    let llm_config = config.default_llm_config();
    assert!(llm_config.is_some());
}

#[test]
fn test_code_config_list_models() {
    let config = CodeConfig {
        providers: vec![
            ProviderConfig {
                name: "anthropic".to_string(),
                api_key: None,
                base_url: None,
                headers: HashMap::new(),
                session_id_header: None,
                models: vec![ModelConfig {
                    id: "claude-sonnet-4".to_string(),
                    name: "".to_string(),
                    family: "".to_string(),
                    api_key: None,
                    base_url: None,
                    headers: HashMap::new(),
                    session_id_header: None,
                    attachment: false,
                    reasoning: false,
                    tool_call: true,
                    temperature: true,
                    release_date: None,
                    modalities: ModelModalities::default(),
                    cost: ModelCost::default(),
                    limit: ModelLimit::default(),
                }],
            },
            ProviderConfig {
                name: "openai".to_string(),
                api_key: None,
                base_url: None,
                headers: HashMap::new(),
                session_id_header: None,
                models: vec![ModelConfig {
                    id: "gpt-4o".to_string(),
                    name: "".to_string(),
                    family: "".to_string(),
                    api_key: None,
                    base_url: None,
                    headers: HashMap::new(),
                    session_id_header: None,
                    attachment: false,
                    reasoning: false,
                    tool_call: true,
                    temperature: true,
                    release_date: None,
                    modalities: ModelModalities::default(),
                    cost: ModelCost::default(),
                    limit: ModelLimit::default(),
                }],
            },
        ],
        ..Default::default()
    };

    let models = config.list_models();
    assert_eq!(models.len(), 2);
}

#[test]
fn test_llm_config_specific_provider_model() {
    let model: ModelConfig = serde_json::from_value(serde_json::json!({
        "id": "claude-3",
        "name": "Claude 3"
    }))
    .unwrap();

    let config = CodeConfig {
        providers: vec![ProviderConfig {
            name: "anthropic".to_string(),
            api_key: Some("sk-test".to_string()),
            base_url: None,
            headers: HashMap::new(),
            session_id_header: None,
            models: vec![model],
        }],
        ..Default::default()
    };

    let llm = config.llm_config("anthropic", "claude-3");
    assert!(llm.is_some());
    let llm = llm.unwrap();
    assert_eq!(llm.provider, "anthropic");
    assert_eq!(llm.model, "claude-3");
}

#[test]
fn test_llm_config_missing_provider() {
    let config = CodeConfig::default();
    assert!(config.llm_config("nonexistent", "model").is_none());
}

#[test]
fn test_llm_config_missing_model() {
    let config = CodeConfig {
        providers: vec![ProviderConfig {
            name: "anthropic".to_string(),
            api_key: Some("sk-test".to_string()),
            base_url: None,
            headers: HashMap::new(),
            session_id_header: None,
            models: vec![],
        }],
        ..Default::default()
    };
    assert!(config.llm_config("anthropic", "nonexistent").is_none());
}

#[test]
fn test_document_parser_config_normalizes_nested_ocr_values() {
    let config = DocumentParserConfig {
        enabled: true,
        max_file_size_mb: 0,
        cache: Some(DocumentCacheConfig {
            enabled: true,
            directory: Some(PathBuf::from("/tmp/cache")),
        }),
        ocr: Some(DocumentOcrConfig {
            enabled: true,
            model: Some("openai/gpt-4.1-mini".to_string()),
            prompt: None,
            max_images: 0,
            dpi: 10,
            provider: None,
            base_url: None,
            api_key: None,
        }),
    }
    .normalized();

    assert_eq!(config.max_file_size_mb, 1);
    let cache = config.cache.unwrap();
    assert!(cache.enabled);
    assert_eq!(cache.directory, Some(PathBuf::from("/tmp/cache")));
    let ocr = config.ocr.unwrap();
    assert_eq!(ocr.max_images, 1);
    assert_eq!(ocr.dpi, 72);
}
