use super::{context_perception, project_context, AgentEvent, AgentLoop};
use crate::context::{
    ContextAssembler, ContextAssembly, ContextItem, ContextQuery, ContextResult, ContextType,
};
use crate::hooks::{
    HookEvent, HookResult, IntentDetectionEvent, PreContextPerceptionEvent, PrePromptEvent,
};
use futures::future::join_all;
use tokio::sync::mpsc;

pub(super) struct TurnContext {
    pub(super) effective_prompt: String,
    pub(super) augmented_system: Option<String>,
}

impl AgentLoop {
    pub(super) async fn prepare_turn_context(
        &self,
        effective_system_prompt: &str,
        effective_prompt: &str,
        message_count: usize,
        session_id: Option<&str>,
        event_tx: &Option<mpsc::Sender<AgentEvent>>,
    ) -> TurnContext {
        let built_system_prompt = Some(effective_system_prompt.to_string());
        let effective_prompt = self
            .fire_pre_prompt(
                session_id.unwrap_or(""),
                effective_prompt,
                &built_system_prompt,
                message_count,
            )
            .await
            .unwrap_or_else(|| effective_prompt.to_string());

        if let Some(ref sp) = self.config.security_provider {
            sp.taint_input(&effective_prompt);
        }

        let mut context_results = self
            .resolve_prompt_context(&effective_prompt, session_id, event_tx)
            .await;
        self.recall_memory_context(&effective_prompt, &mut context_results, event_tx)
            .await;

        let context_assembly = self.assemble_context_results(&context_results);
        self.emit_context_resolved(&context_assembly, event_tx)
            .await;

        TurnContext {
            augmented_system: self.build_augmented_system_prompt_with_base(
                effective_system_prompt,
                &context_assembly,
            ),
            effective_prompt,
        }
    }

    async fn resolve_prompt_context(
        &self,
        effective_prompt: &str,
        session_id: Option<&str>,
        event_tx: &Option<mpsc::Sender<AgentEvent>>,
    ) -> Vec<ContextResult> {
        if self.config.context_providers.is_empty() {
            return Vec::new();
        }

        if let Some(tx) = event_tx {
            tx.send(AgentEvent::ContextResolving {
                providers: self
                    .config
                    .context_providers
                    .iter()
                    .map(|p| p.name().to_string())
                    .collect(),
            })
            .await
            .ok();
        }

        let workspace = self.tool_context.workspace.display().to_string();
        let session_id_str = session_id.unwrap_or("");
        let harness_intent = self
            .fire_intent_detection(effective_prompt, session_id_str, &workspace)
            .await;

        let perception_event = if let Some(detected) = harness_intent {
            tracing::info!(
                intent = %detected.detected_intent,
                confidence = %detected.confidence,
                "Intent detected from AHP harness"
            );
            Some(
                context_perception::build_pre_context_perception_from_intent(
                    detected,
                    effective_prompt,
                    session_id_str,
                    &workspace,
                ),
            )
        } else {
            tracing::debug!("No intent from harness, using local keyword detection");
            self.detect_context_perception_intent(effective_prompt, session_id_str, &workspace)
        };

        let Some(perception_event) = perception_event else {
            return self.resolve_context(effective_prompt, session_id).await;
        };

        tracing::info!(
            intent = %perception_event.intent,
            target_type = %perception_event.target_type,
            "Context perception intent detected, firing AHP hook"
        );

        match self.fire_pre_context_perception(&perception_event).await {
            HookResult::Continue(Some(modified_context)) => {
                #[cfg(feature = "ahp")]
                {
                    if let Ok(injected) =
                        serde_json::from_value::<crate::ahp::InjectedContext>(modified_context)
                    {
                        tracing::info!(
                            facts = injected.facts.len(),
                            "Using injected context from AHP harness"
                        );
                        self.apply_injected_context(injected)
                    } else {
                        tracing::warn!(
                            "Failed to parse injected context, falling back to providers"
                        );
                        self.resolve_context(effective_prompt, session_id).await
                    }
                }
                #[cfg(not(feature = "ahp"))]
                {
                    let _ = modified_context;
                    self.resolve_context(effective_prompt, session_id).await
                }
            }
            HookResult::Block(_) => {
                tracing::info!("AHP harness blocked context injection");
                Vec::new()
            }
            _ => self.resolve_context(effective_prompt, session_id).await,
        }
    }

    async fn recall_memory_context(
        &self,
        effective_prompt: &str,
        context_results: &mut Vec<ContextResult>,
        event_tx: &Option<mpsc::Sender<AgentEvent>>,
    ) {
        let Some(ref memory) = self.config.memory else {
            return;
        };

        match memory.recall_similar(effective_prompt, 5).await {
            Ok(items) if !items.is_empty() => {
                if let Some(tx) = event_tx {
                    for item in &items {
                        tx.send(AgentEvent::MemoryRecalled {
                            memory_id: item.id.clone(),
                            content: item.content.clone(),
                            relevance: item.relevance_score(),
                        })
                        .await
                        .ok();
                    }
                    tx.send(AgentEvent::MemoriesSearched {
                        query: Some(effective_prompt.to_string()),
                        tags: Vec::new(),
                        result_count: items.len(),
                    })
                    .await
                    .ok();
                }
                context_results.push(crate::memory::memory_items_to_context_result(
                    "memory", items,
                ));
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(error = %e, "Failed to recall memory context");
            }
        }
    }

    async fn emit_context_resolved(
        &self,
        context_assembly: &ContextAssembly,
        event_tx: &Option<mpsc::Sender<AgentEvent>>,
    ) {
        let Some(tx) = event_tx else {
            return;
        };

        let total_items = context_assembly.items.len();
        let total_tokens = context_assembly.total_tokens;

        tracing::info!(
            context_items = total_items,
            context_tokens = total_tokens,
            context_truncated = context_assembly.truncated,
            "Context resolution completed"
        );

        tx.send(AgentEvent::ContextResolved {
            total_items,
            total_tokens,
        })
        .await
        .ok();
    }

    /// Resolve context from all providers for a given prompt
    ///
    /// Returns aggregated context results from all configured providers.
    pub(super) async fn resolve_context(
        &self,
        prompt: &str,
        session_id: Option<&str>,
    ) -> Vec<ContextResult> {
        if self.config.context_providers.is_empty() {
            return Vec::new();
        }

        let query = ContextQuery::new(prompt).with_session_id(session_id.unwrap_or(""));

        let futures = self
            .config
            .context_providers
            .iter()
            .map(|p| p.query(&query));
        let outcomes = join_all(futures).await;

        outcomes
            .into_iter()
            .enumerate()
            .filter_map(|(i, r)| match r {
                Ok(result) if !result.is_empty() => Some(result),
                Ok(_) => None,
                Err(e) => {
                    tracing::warn!(
                        "Context provider '{}' failed: {}",
                        self.config.context_providers[i].name(),
                        e
                    );
                    None
                }
            })
            .collect()
    }

    /// Detect if context perception is needed based on user prompt.
    ///
    /// Returns `Some(PreContextPerceptionEvent)` if the prompt suggests the model
    /// needs workspace knowledge (finding files, understanding code, etc.).
    pub fn detect_context_perception_intent(
        &self,
        prompt: &str,
        session_id: &str,
        workspace: &str,
    ) -> Option<PreContextPerceptionEvent> {
        context_perception::detect_local_context_perception_intent(prompt, session_id, workspace)
    }

    /// Fire PreContextPerception hook and wait for harness decision.
    async fn fire_pre_context_perception(&self, event: &PreContextPerceptionEvent) -> HookResult {
        if let Some(he) = &self.config.hook_engine {
            let hook_event = HookEvent::PreContextPerception(event.clone());
            he.fire(&hook_event).await
        } else {
            HookResult::continue_()
        }
    }

    /// Fire IntentDetection hook and wait for harness decision.
    ///
    /// This is called on every prompt to detect user intent via the AHP harness.
    /// Returns the detected intent if the harness provides one, or None if blocked/failed.
    async fn fire_intent_detection(
        &self,
        prompt: &str,
        session_id: &str,
        workspace: &str,
    ) -> Option<context_perception::IntentDetectionResult> {
        let event = IntentDetectionEvent {
            session_id: session_id.to_string(),
            prompt: prompt.to_string(),
            workspace: workspace.to_string(),
            language_hint: context_perception::detect_language_hint(prompt),
        };

        let hook_result = if let Some(he) = &self.config.hook_engine {
            let hook_event = HookEvent::IntentDetection(event);
            he.fire(&hook_event).await
        } else {
            return None;
        };

        match hook_result {
            HookResult::Continue(Some(modified)) => {
                // Parse the intent detection result
                serde_json::from_value::<context_perception::IntentDetectionResult>(modified).ok()
            }
            HookResult::Block(_) => {
                // Harness blocked intent detection - use fallback
                tracing::info!("AHP harness blocked intent detection");
                None
            }
            _ => None,
        }
    }

    /// Apply injected context from AHP harness decision.
    #[cfg(feature = "ahp")]
    fn apply_injected_context(&self, injected: crate::ahp::InjectedContext) -> Vec<ContextResult> {
        context_perception::injected_context_to_results(injected)
    }

    /// Build augmented system prompt with context
    #[allow(dead_code)]
    pub(super) fn build_augmented_system_prompt(
        &self,
        context_results: &[ContextResult],
    ) -> Option<String> {
        let base = self.system_prompt();
        let context_assembly = self.assemble_context_results(context_results);
        self.build_augmented_system_prompt_with_base(&base, &context_assembly)
    }

    pub(super) fn assemble_context_results(
        &self,
        context_results: &[ContextResult],
    ) -> ContextAssembly {
        let mut results = context_results.to_vec();

        if self.config.prompt_slots.guidelines.is_none() {
            let project_hint = project_context::detect_project_hint(&self.tool_context.workspace);
            if !project_hint.is_empty() {
                let token_count = project_hint.split_whitespace().count().max(1);
                let mut result = ContextResult::new("project_hint");
                result.add_item(
                    ContextItem::new("project_hint", ContextType::Resource, project_hint)
                        .with_source("a3s://project-hint")
                        .with_provenance("workspace_marker")
                        .with_priority(0.65)
                        .with_trust(0.8)
                        .with_freshness(1.0)
                        .with_relevance(0.9)
                        .with_token_count(token_count),
                );
                results.push(result);
            }
        }

        ContextAssembler::with_default_budget().assemble(&results)
    }

    fn build_augmented_system_prompt_with_base(
        &self,
        base: &str,
        context_assembly: &ContextAssembly,
    ) -> Option<String> {
        let base = base.to_string();

        // MCP tool definitions are selected per turn by ToolSelector. Keep the
        // system prompt small instead of listing every external tool here.
        let has_mcp_tools = self
            .tool_executor
            .definitions()
            .iter()
            .any(|t| t.name.starts_with("mcp__"));

        let mcp_section = if has_mcp_tools {
            "## MCP Tools\n\nExternal MCP tools are available on demand when relevant to the current request.".to_string()
        } else {
            String::new()
        };

        // Always-on grounding facts the model cannot infer from tool output —
        // most importantly today's date (training cutoff is in the past).
        let env_section = render_env_block(&self.tool_context.workspace);

        let parts: Vec<&str> = [base.as_str(), env_section.as_str(), mcp_section.as_str()]
            .iter()
            .filter(|s| !s.is_empty())
            .copied()
            .collect();

        if context_assembly.is_empty() {
            return Some(parts.join("\n\n"));
        }

        let context_xml = context_assembly.to_xml();
        Some(format!("{}\n\n{}", parts.join("\n\n"), context_xml))
    }

    /// Fire PrePrompt hook event before prompt augmentation.
    /// Returns optional modified prompt text from the hook.
    async fn fire_pre_prompt(
        &self,
        session_id: &str,
        prompt: &str,
        system_prompt: &Option<String>,
        message_count: usize,
    ) -> Option<String> {
        if let Some(he) = &self.config.hook_engine {
            let event = HookEvent::PrePrompt(PrePromptEvent {
                session_id: session_id.to_string(),
                prompt: prompt.to_string(),
                system_prompt: system_prompt.clone(),
                message_count,
            });
            let result = he.fire(&event).await;
            if let HookResult::Continue(Some(modified)) = result {
                // Extract modified prompt from hook response
                if let Some(new_prompt) = modified.get("prompt").and_then(|v| v.as_str()) {
                    return Some(new_prompt.to_string());
                }
            }
        }
        None
    }
}

/// Render the always-on `<env>` grounding block.
///
/// Supplies the few facts the model cannot recover from tool output: today's
/// date (the training cutoff is in the past, so the model must not guess it),
/// the host platform, and the working directory. Computed fresh from live values
/// each turn — cheap (no shell-out), so it is built in code rather than a static
/// template that would serve stale data.
fn render_env_block(workspace: &std::path::Path) -> String {
    format!(
        "<env>\nWorking directory: {}\nPlatform: {} ({})\nToday's date: {}\n</env>",
        workspace.display(),
        std::env::consts::OS,
        std::env::consts::ARCH,
        chrono::Local::now().format("%Y-%m-%d"),
    )
}

#[cfg(test)]
mod tests {
    use super::render_env_block;

    #[test]
    fn env_block_contains_grounding_facts() {
        let block = render_env_block(std::path::Path::new("/tmp/demo-ws"));
        assert!(block.starts_with("<env>"), "block: {block}");
        assert!(block.trim_end().ends_with("</env>"));
        assert!(block.contains("Working directory: /tmp/demo-ws"));
        assert!(block.contains("Platform:"));
        assert!(block.contains(std::env::consts::OS));
        assert!(block.contains("Today's date:"));
    }

    #[test]
    fn env_block_date_is_iso_yyyy_mm_dd() {
        let block = render_env_block(std::path::Path::new("/tmp"));
        let line = block
            .lines()
            .find(|l| l.starts_with("Today's date:"))
            .expect("date line present");
        let date = line.trim_start_matches("Today's date:").trim();
        assert_eq!(date.len(), 10, "date not YYYY-MM-DD: {date}");
        assert_eq!(date.matches('-').count(), 2, "date not YYYY-MM-DD: {date}");
        assert!(date.chars().all(|c| c.is_ascii_digit() || c == '-'));
    }
}
