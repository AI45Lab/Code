use super::SessionOptions;
use crate::config::CodeConfig;
use crate::error::Result;
use crate::llm::LlmClient;
use anyhow::Context;
use std::sync::Arc;

pub(super) fn resolve_auto_delegation_config(
    code_config: &CodeConfig,
    opts: &SessionOptions,
) -> crate::config::AutoDelegationConfig {
    let mut auto_delegation = if let Some(config) = opts.auto_delegation.clone() {
        config
    } else {
        let mut config = code_config.auto_delegation.clone();
        if let Some(auto_parallel) = code_config.auto_parallel {
            config.auto_parallel = auto_parallel;
        }
        config
    };
    if let Some(enabled) = opts.manual_delegation_enabled {
        auto_delegation.allow_manual_delegation = enabled;
    }
    if let Some(auto_parallel) = opts.auto_parallel_delegation {
        auto_delegation.auto_parallel = auto_parallel;
    }

    auto_delegation
}

pub(super) fn resolve_session_llm_client(
    code_config: &CodeConfig,
    opts: &SessionOptions,
    session_id: Option<&str>,
) -> Result<Arc<dyn LlmClient>> {
    // A host-supplied client overrides the provider-string factory entirely:
    // the host owns the full Action-layer dependency (custom provider, replay
    // client, proxy/audit wrapper). Config-based model resolution is bypassed.
    if let Some(ref client) = opts.llm_client {
        return Ok(Arc::clone(client));
    }

    let model_ref = if let Some(ref model) = opts.model {
        model.as_str()
    } else {
        if opts.temperature.is_some() || opts.thinking_budget.is_some() {
            tracing::warn!(
                "temperature/thinking_budget set without model override - these will be ignored. \
                 Use with_model() to apply LLM parameter overrides."
            );
        }
        code_config
            .default_model
            .as_deref()
            .context("default_model must be set in 'provider/model' format")?
    };

    let (provider_name, model_id) = model_ref
        .split_once('/')
        .context("model format must be 'provider/model' (e.g., 'openai/gpt-4o')")?;

    let mut llm_config = code_config
        .llm_config(provider_name, model_id)
        .with_context(|| {
            format!("provider '{provider_name}' or model '{model_id}' not found in config")
        })?;

    if opts.model.is_some() {
        if let Some(temp) = opts.temperature {
            llm_config = llm_config.with_temperature(temp);
        }
        if let Some(budget) = opts.thinking_budget {
            llm_config = llm_config.with_thinking_budget(budget);
        }
    }

    if let Some(session_id) = session_id {
        llm_config = llm_config.with_session_id(session_id);
    }

    Ok(crate::llm::create_client_with_config(llm_config))
}

pub(super) struct ResolvedSessionMemory {
    pub(super) memory: Option<Arc<crate::memory::AgentMemory>>,
    pub(super) init_warning: Option<String>,
}

pub(super) fn resolve_session_memory(opts: &SessionOptions) -> ResolvedSessionMemory {
    let mut init_warning = None;
    let store = if let Some(ref store) = opts.memory_store {
        Some(Arc::clone(store))
    } else if let Some(ref dir) = opts.file_memory_dir {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                let dir = dir.clone();
                match tokio::task::block_in_place(|| {
                    handle.block_on(a3s_memory::FileMemoryStore::new(dir))
                }) {
                    Ok(store) => Some(Arc::new(store) as Arc<dyn a3s_memory::MemoryStore>),
                    Err(e) => {
                        let msg = format!("Failed to create file memory store: {}", e);
                        tracing::warn!("{}", msg);
                        init_warning = Some(msg);
                        None
                    }
                }
            }
            Err(_) => {
                let msg = "No async runtime available for file memory store - memory disabled"
                    .to_string();
                tracing::warn!("{}", msg);
                init_warning = Some(msg);
                None
            }
        }
    } else {
        None
    };

    ResolvedSessionMemory {
        memory: store.map(|s| Arc::new(crate::memory::AgentMemory::new(s))),
        init_warning,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{LlmResponse, Message, StreamEvent, ToolDefinition};
    // The LlmClient trait returns anyhow::Result; shadow super's crate::error::Result.
    use anyhow::Result;
    use async_trait::async_trait;
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    struct DummyClient;

    #[async_trait]
    impl LlmClient for DummyClient {
        async fn complete(
            &self,
            _: &[Message],
            _: Option<&str>,
            _: &[ToolDefinition],
        ) -> Result<LlmResponse> {
            anyhow::bail!("resolver short-circuits before the client is called")
        }
        async fn complete_streaming(
            &self,
            _: &[Message],
            _: Option<&str>,
            _: &[ToolDefinition],
            _: CancellationToken,
        ) -> Result<mpsc::Receiver<StreamEvent>> {
            anyhow::bail!("not used")
        }
    }

    // A default CodeConfig has no default_model, so the provider-string factory
    // path errors — proving the override is what makes resolution succeed.
    #[test]
    fn host_supplied_llm_client_overrides_factory() {
        let config = CodeConfig::default();
        let opts = SessionOptions::new().with_llm_client(Arc::new(DummyClient));
        assert!(
            resolve_session_llm_client(&config, &opts, None).is_ok(),
            "with_llm_client must bypass provider/model config resolution"
        );
    }

    #[test]
    fn without_llm_client_missing_default_model_errors() {
        let config = CodeConfig::default();
        let opts = SessionOptions::new();
        assert!(
            resolve_session_llm_client(&config, &opts, None).is_err(),
            "no host client + no default_model should error (control case)"
        );
    }
}

pub(super) fn resolve_session_store(
    code_config: &CodeConfig,
    opts: &SessionOptions,
) -> Option<Arc<dyn crate::store::SessionStore>> {
    if opts.session_store.is_some() {
        return opts.session_store.clone();
    }

    let dir = code_config.sessions_dir.as_ref()?;
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            let dir = dir.clone();
            match tokio::task::block_in_place(|| {
                handle.block_on(crate::store::FileSessionStore::new(dir))
            }) {
                Ok(store) => Some(Arc::new(store) as Arc<dyn crate::store::SessionStore>),
                Err(e) => {
                    tracing::warn!("Failed to create session store from sessions_dir: {}", e);
                    None
                }
            }
        }
        Err(_) => {
            tracing::warn!("No async runtime for sessions_dir store - persistence disabled");
            None
        }
    }
}
