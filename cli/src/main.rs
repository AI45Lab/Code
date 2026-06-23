//! Codex-style terminal UI for the A3S Code agent.
//!
//! Built on the `a3s-tui` TEA framework: it drives an [`AgentSession`] via
//! `session.stream()` and renders the resulting [`AgentEvent`] stream as a live
//! chat transcript, mapping tool-confirmation events to an approve/deny modal.
//!
//! Streaming bridge: `session.stream()` yields a `tokio::mpsc` receiver. A
//! self-re-issuing "pump" command reads one event, turns it into a `Msg`, and
//! the update handler issues the next pump — feeding the async event stream into
//! the synchronous TEA update loop one event at a time.

use std::sync::Arc;
use std::time::Duration;

use a3s_code_core::{Agent, AgentEvent, AgentSession};
use a3s_tui::cmd::{self, Cmd};
use a3s_tui::components::modal::{Modal, ModalMsg};
use a3s_tui::components::textarea::TextareaMsg;
use a3s_tui::components::viewport::ViewportMsg;
use a3s_tui::components::{Spinner, StatusBar, Textarea, Viewport};
use a3s_tui::event::KeyEvent;
use a3s_tui::keymap::{KeyBinding, Keymap};
use a3s_tui::layout::{Constraint, Layout};
use a3s_tui::streaming::StreamingMarkdown;
use a3s_tui::style::{Color, Style};
use a3s_tui::{Event, KeyCode, KeyModifiers, Model, ProgramBuilder};
use tokio::sync::{mpsc, Mutex};

/// Shared, single-consumer receiver for the active agent run. Wrapped so the
/// pump command can own a clone; pumps run sequentially, so the mutex never
/// actually contends.
type SharedRx = Arc<Mutex<mpsc::Receiver<AgentEvent>>>;

#[derive(PartialEq)]
enum State {
    Idle,
    Streaming,
    Awaiting,
}

#[derive(Clone)]
#[allow(clippy::enum_variant_names)]
enum Action {
    ScrollUp,
    ScrollDown,
    ScrollTop,
    ScrollBottom,
}

enum Msg {
    Term(Event),
    // Boxed: AgentEvent is large; keeps the Msg enum small.
    Agent(Box<AgentEvent>),
    Submit(String),
    StreamStarted(SharedRx),
    StreamEnded,
    StreamError(String),
    SpinnerTick,
    ModalConfirm(usize),
    ModalDismiss,
    Resume,
    Interrupted,
    Quit,
}

impl From<Event> for Msg {
    fn from(event: Event) -> Self {
        match &event {
            Event::Key(KeyEvent {
                code: KeyCode::Char('c'),
                modifiers,
            }) if modifiers.contains(KeyModifiers::CONTROL) => Msg::Quit,
            _ => Msg::Term(event),
        }
    }
}

/// Read one event from the active run and turn it into a `Msg`.
fn pump(rx: SharedRx) -> Cmd<Msg> {
    cmd::cmd(move || async move {
        let mut guard = rx.lock().await;
        match guard.recv().await {
            Some(event) => Msg::Agent(Box::new(event)),
            None => Msg::StreamEnded,
        }
    })
}

fn spinner_tick() -> Cmd<Msg> {
    cmd::tick(Duration::from_millis(80), Msg::SpinnerTick)
}

struct App {
    session: Arc<AgentSession>,
    viewport: Viewport,
    textarea: Textarea,
    spinner: Spinner,
    streaming: StreamingMarkdown,
    /// Live reasoning ("thinking") text for the current turn, shown dimmed above
    /// the answer and cleared when the answer is finalized.
    thinking: String,
    state: State,
    messages: Vec<String>,
    rx: Option<SharedRx>,
    modal: Option<Modal>,
    pending_tool: Option<(String, String)>,
    width: u16,
    height: u16,
    keymap: Keymap<Action>,
}

impl Model for App {
    type Msg = Msg;

    fn init(&mut self) -> Option<Cmd<Msg>> {
        let welcome = Style::new().fg(Color::BrightBlack).italic().render(
            "  A3S Code — type a message and press Enter.\n  \
             Esc interrupt | Ctrl+C quit | PgUp/PgDn scroll | /help\n",
        );
        self.viewport.set_content(&welcome);
        None
    }

    fn update(&mut self, msg: Msg) -> Option<Cmd<Msg>> {
        match msg {
            Msg::Quit => return Some(cmd::quit()),

            Msg::Term(Event::Resize { width, height }) => {
                self.width = width;
                self.height = height;
                self.viewport.resize(width, height.saturating_sub(7));
                self.streaming = StreamingMarkdown::new((width as usize).saturating_sub(2));
                self.rebuild_viewport();
            }

            Msg::Term(Event::Key(key)) => {
                if self.state == State::Awaiting {
                    return self.handle_modal_key(&key);
                }
                if let Some(action) = self.keymap.resolve(&key) {
                    let m = match action {
                        Action::ScrollUp => ViewportMsg::PageUp,
                        Action::ScrollDown => ViewportMsg::PageDown,
                        Action::ScrollTop => ViewportMsg::Top,
                        Action::ScrollBottom => ViewportMsg::Bottom,
                    };
                    self.viewport.update(m);
                    return None;
                }
                if self.state == State::Streaming {
                    // Esc interrupts the in-progress run.
                    if key.code == KeyCode::Esc {
                        self.push_line(&Style::new().fg(Color::Yellow).render("  ⎋ interrupting…"));
                        let session = self.session.clone();
                        return Some(cmd::cmd(move || async move {
                            session.cancel().await;
                            Msg::Interrupted
                        }));
                    }
                    return None;
                }
                if let Some(TextareaMsg::Submit(text)) = self.textarea.handle_key(&key) {
                    return Some(cmd::msg(Msg::Submit(text)));
                }
            }

            Msg::Submit(text) => return self.on_submit(text),

            Msg::StreamStarted(rx) => {
                self.rx = Some(rx.clone());
                return Some(pump(rx));
            }

            Msg::StreamError(e) => {
                self.push_line(&Style::new().fg(Color::Red).render(&format!("  error: {e}")));
                self.finish();
            }

            Msg::Agent(event) => return self.on_agent_event(*event),

            Msg::StreamEnded => {
                if self.state == State::Streaming {
                    self.finalize_streaming();
                }
                self.finish();
            }

            Msg::SpinnerTick => {
                self.spinner.tick();
                if self.state == State::Streaming {
                    self.update_viewport_with_stream();
                    return Some(spinner_tick());
                }
            }

            Msg::ModalConfirm(idx) => {
                self.modal = None;
                let approved = idx == 0;
                self.state = State::Streaming;
                if let Some((tool_id, name)) = self.pending_tool.take() {
                    let verdict = if approved { "allowed" } else { "denied" };
                    let color = if approved { Color::Yellow } else { Color::Red };
                    self.push_line(
                        &Style::new()
                            .fg(color)
                            .render(&format!("  [{verdict}] {name}")),
                    );
                    let session = self.session.clone();
                    return Some(cmd::batch(vec![
                        cmd::cmd(move || async move {
                            let _ = session.confirm_tool_use(&tool_id, approved, None).await;
                            Msg::Resume
                        }),
                        spinner_tick(),
                    ]));
                }
            }

            Msg::ModalDismiss => return Some(cmd::msg(Msg::ModalConfirm(1))),

            Msg::Resume => {
                if let Some(rx) = self.rx.clone() {
                    return Some(pump(rx));
                }
            }

            _ => {}
        }
        None
    }

    fn view(&self) -> String {
        if self.state == State::Awaiting {
            if let Some(modal) = &self.modal {
                return modal.view(self.width, self.height);
            }
        }

        let status_text = match self.state {
            State::Streaming => format!(" {} working... (Esc interrupt)", self.spinner.view()),
            State::Idle => " a3s-code".to_string(),
            State::Awaiting => " awaiting approval...".to_string(),
        };
        let status = StatusBar::new()
            .left(&status_text)
            .right("Ctrl+C quit | PgUp/PgDn scroll ")
            .fg(Color::White)
            .bg(Color::BrightBlack)
            .view(self.width);

        let viewport_view = self.viewport.view();
        let separator = Style::new()
            .fg(Color::BrightBlack)
            .render(&"─".repeat(self.width as usize));
        let prompt = Style::new().fg(Color::BrightGreen).bold().render("❯ ");
        let input_view = format!("{}{}", prompt, self.textarea.view());

        Layout::vertical()
            .item(&status, Constraint::Fixed(1))
            .item(&viewport_view, Constraint::Fill)
            .item(&separator, Constraint::Fixed(1))
            .item(&input_view, Constraint::Fixed(3))
            .render(self.height)
    }
}

impl App {
    fn on_submit(&mut self, text: String) -> Option<Cmd<Msg>> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }
        match trimmed {
            "/exit" | "/quit" => return Some(cmd::quit()),
            "/clear" => {
                self.messages.clear();
                self.textarea.clear();
                self.rebuild_viewport();
                return None;
            }
            "/help" => {
                self.messages
                    .push(Style::new().fg(Color::BrightBlack).render(
                        "  commands: /clear reset · /exit quit\n  \
                     Enter send · Esc interrupt · Ctrl+C quit · PgUp/PgDn scroll",
                    ));
                self.textarea.clear();
                self.rebuild_viewport();
                return None;
            }
            _ => {}
        }

        self.messages.push(
            Style::new()
                .bold()
                .fg(Color::BrightGreen)
                .render(&format!("❯ {trimmed}")),
        );
        self.textarea.clear();
        self.streaming.clear();
        self.state = State::Streaming;
        self.spinner.start();
        self.rebuild_viewport();

        let session = self.session.clone();
        let prompt = trimmed.to_string();
        Some(cmd::batch(vec![
            cmd::cmd(move || async move {
                match session.stream(prompt.as_str(), None).await {
                    Ok((rx, _join)) => Msg::StreamStarted(Arc::new(Mutex::new(rx))),
                    Err(e) => Msg::StreamError(e.to_string()),
                }
            }),
            spinner_tick(),
        ]))
    }

    fn on_agent_event(&mut self, event: AgentEvent) -> Option<Cmd<Msg>> {
        match event {
            AgentEvent::TextDelta { text } => {
                self.streaming.push(&text);
                self.update_viewport_with_stream();
            }
            AgentEvent::ReasoningDelta { text } => {
                self.thinking.push_str(&text);
                self.update_viewport_with_stream();
            }
            AgentEvent::ToolStart { name, .. } => {
                self.finalize_streaming();
                self.push_line(&Style::new().fg(Color::Cyan).render(&format!("  ⚙ {name}")));
            }
            AgentEvent::ToolEnd {
                name,
                output,
                exit_code,
                metadata,
                ..
            } => {
                self.push_line(&render_tool_end(
                    &name,
                    exit_code,
                    &output,
                    metadata.as_ref(),
                ));
            }
            AgentEvent::ConfirmationRequired {
                tool_id,
                tool_name,
                args,
                ..
            } => {
                self.state = State::Awaiting;
                self.pending_tool = Some((tool_id, tool_name.clone()));
                let body = format!(
                    "Tool: {tool_name}\nArgs: {}",
                    truncate(&args.to_string(), 300)
                );
                self.modal = Some(
                    Modal::new()
                        .title("Approve tool call?")
                        .body(&body)
                        .options(vec!["Allow", "Deny"]),
                );
                return None; // wait for the user; do not pump
            }
            AgentEvent::End { text, usage, .. } => {
                if self.streaming.raw_content().trim().is_empty() && !text.is_empty() {
                    self.streaming.push(&text);
                }
                self.finalize_streaming();
                if usage.total_tokens > 0 {
                    self.push_line(&Style::new().fg(Color::BrightBlack).render(&format!(
                        "  ⏱ {} tokens (prompt {}, completion {})",
                        usage.total_tokens, usage.prompt_tokens, usage.completion_tokens
                    )));
                }
                self.finish();
                return None;
            }
            AgentEvent::Error { message } => {
                self.push_line(
                    &Style::new()
                        .fg(Color::Red)
                        .render(&format!("  error: {message}")),
                );
                self.finish();
                return None;
            }
            // TurnStart/TurnEnd, ToolInputDelta, planning, memory, subagent,
            // confirmation echoes, etc. — not surfaced in this MVP.
            _ => {}
        }
        // Keep draining the stream.
        self.rx.clone().map(pump)
    }

    fn finalize_streaming(&mut self) {
        let rendered = self.streaming.view();
        if !rendered.trim().is_empty() {
            self.messages.push(rendered);
        }
        self.streaming.clear();
        self.thinking.clear();
        self.rebuild_viewport();
    }

    fn finish(&mut self) {
        self.state = State::Idle;
        self.spinner.stop();
        self.rx = None;
        self.rebuild_viewport();
    }

    fn push_line(&mut self, line: &str) {
        self.messages.push(line.to_string());
        self.rebuild_viewport();
    }

    fn update_viewport_with_stream(&mut self) {
        let mut blocks: Vec<String> = self.messages.clone();
        if !self.thinking.trim().is_empty() {
            blocks.push(
                Style::new()
                    .fg(Color::BrightBlack)
                    .italic()
                    .render(&format!("💭 {}", self.thinking.trim())),
            );
        }
        let rendered = self.streaming.view();
        if !rendered.is_empty() {
            blocks.push(rendered);
        }
        self.viewport.set_content(&blocks.join("\n\n"));
    }

    fn rebuild_viewport(&mut self) {
        let full = self.messages.join("\n\n");
        self.viewport.set_content(&format!("{full}\n"));
    }

    fn handle_modal_key(&mut self, key: &KeyEvent) -> Option<Cmd<Msg>> {
        if let Some(modal) = &mut self.modal {
            match key.code {
                KeyCode::Up | KeyCode::Char('k') => {
                    modal.update(ModalMsg::Prev);
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    modal.update(ModalMsg::Next);
                }
                KeyCode::Enter => {
                    let idx = modal.confirm();
                    return Some(cmd::msg(Msg::ModalConfirm(idx)));
                }
                KeyCode::Esc => return Some(cmd::msg(Msg::ModalDismiss)),
                _ => {}
            }
        }
        None
    }
}

/// Headless probe of the same `session.stream()` / `AgentEvent` path the TUI
/// uses, auto-approving tool calls. Drives the integration without a TTY.
async fn run_smoke(session: Arc<AgentSession>) -> anyhow::Result<()> {
    let prompt = std::env::var("A3S_CODE_TUI_PROMPT")
        .unwrap_or_else(|_| "Reply with exactly one short sentence: what is 2 + 2?".to_string());
    eprintln!("[smoke] prompt: {prompt}");
    let (mut rx, _join) = session.stream(prompt.as_str(), None).await?;
    while let Some(event) = rx.recv().await {
        match event {
            AgentEvent::TextDelta { text } => print!("{text}"),
            AgentEvent::ToolStart { name, .. } => eprintln!("\n[tool start] {name}"),
            AgentEvent::ToolEnd {
                name, exit_code, ..
            } => eprintln!("[tool end] {name} (exit {exit_code})"),
            AgentEvent::ConfirmationRequired {
                tool_id, tool_name, ..
            } => {
                eprintln!("[confirm] auto-allowing {tool_name}");
                let _ = session.confirm_tool_use(&tool_id, true, None).await;
            }
            AgentEvent::End { .. } => {
                eprintln!("\n[end]");
                break;
            }
            AgentEvent::Error { message } => {
                eprintln!("\n[error] {message}");
                break;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Render a completed tool call. File-editing tools (`write`/`edit`) carry
/// `before`/`after`/`file_path` in their metadata — show those as a colored
/// diff; everything else shows a status line + a few lines of output.
fn render_tool_end(
    name: &str,
    exit_code: i32,
    output: &str,
    meta: Option<&serde_json::Value>,
) -> String {
    if let Some(meta) = meta {
        if let (Some(before), Some(after), Some(path)) = (
            meta.get("before").and_then(|v| v.as_str()),
            meta.get("after").and_then(|v| v.as_str()),
            meta.get("file_path").and_then(|v| v.as_str()),
        ) {
            return render_diff(path, before, after);
        }
    }
    let status = if exit_code == 0 { "✓" } else { "✗" };
    let head = output.lines().take(6).collect::<Vec<_>>().join("\n");
    Style::new()
        .fg(Color::BrightBlack)
        .render(&format!("  {status} {name}\n{head}"))
}

/// Render a unified-ish line diff (changed lines only) with +/- coloring.
fn render_diff(path: &str, before: &str, after: &str) -> String {
    use similar::{ChangeTag, TextDiff};
    const MAX_LINES: usize = 80;

    let diff = TextDiff::from_lines(before, after);
    let mut lines: Vec<String> = Vec::new();
    let (mut adds, mut dels) = (0usize, 0usize);
    for change in diff.iter_all_changes() {
        let raw = change.value();
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        match change.tag() {
            ChangeTag::Delete => {
                dels += 1;
                if lines.len() < MAX_LINES {
                    lines.push(Style::new().fg(Color::Red).render(&format!("  - {raw}")));
                }
            }
            ChangeTag::Insert => {
                adds += 1;
                if lines.len() < MAX_LINES {
                    lines.push(Style::new().fg(Color::Green).render(&format!("  + {raw}")));
                }
            }
            ChangeTag::Equal => {}
        }
    }
    if lines.len() >= MAX_LINES {
        lines.push(
            Style::new()
                .fg(Color::BrightBlack)
                .render("  … (diff truncated)"),
        );
    }
    let mut out = Style::new()
        .fg(Color::Cyan)
        .render(&format!("  ✎ {path}  (+{adds} -{dels})"));
    if !lines.is_empty() {
        out.push('\n');
        out.push_str(&lines.join("\n"));
    }
    out
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max).collect();
        format!("{head}…")
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config_path =
        std::env::var("A3S_CONFIG_FILE").unwrap_or_else(|_| ".a3s/config.acl".to_string());
    let agent = Agent::new(config_path.clone())
        .await
        .map_err(|e| anyhow::anyhow!("failed to load agent from {config_path}: {e}"))?;
    let workspace = std::env::current_dir()?.to_string_lossy().to_string();
    let session = Arc::new(agent.session(workspace, None)?);

    // Headless smoke mode: exercise the agent-stream integration (the hard part
    // the TUI depends on) without taking over the terminal. Useful for CI/probes
    // and for validating a model/config end-to-end.
    if std::env::var_os("A3S_CODE_TUI_SMOKE").is_some() {
        return run_smoke(session).await;
    }

    let (width, height) = a3s_tui::terminal::Terminal::size().unwrap_or((80, 24));
    let keymap = Keymap::new()
        .bind(
            KeyBinding::new(KeyCode::PageUp),
            Action::ScrollUp,
            "Scroll up",
        )
        .bind(
            KeyBinding::new(KeyCode::PageDown),
            Action::ScrollDown,
            "Scroll down",
        )
        .bind(
            KeyBinding::ctrl(KeyCode::Home),
            Action::ScrollTop,
            "Scroll to top",
        )
        .bind(
            KeyBinding::ctrl(KeyCode::End),
            Action::ScrollBottom,
            "Scroll to bottom",
        );

    let app = App {
        session,
        viewport: Viewport::new(width, height.saturating_sub(7)),
        textarea: Textarea::new()
            .with_height(3)
            .with_width(width)
            .with_submit_on_enter(true),
        spinner: Spinner::new().with_title(""),
        streaming: StreamingMarkdown::new((width as usize).saturating_sub(2)),
        thinking: String::new(),
        state: State::Idle,
        messages: Vec::new(),
        rx: None,
        modal: None,
        pending_tool: None,
        width,
        height,
        keymap,
    };

    ProgramBuilder::new(app)
        .with_alt_screen()
        .with_fps(30)
        .run()
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_metadata_renders_colored_diff() {
        let meta = serde_json::json!({
            "file_path": "src/x.rs",
            "before": "let a = 1;\nkeep;\n",
            "after": "let a = 2;\nkeep;\n",
        });
        let out = render_tool_end("edit", 0, "ok", Some(&meta));
        assert!(out.contains("src/x.rs"), "header has path");
        assert!(out.contains("+1") && out.contains("-1"), "add/del counts");
        assert!(out.contains("let a = 2;"), "shows inserted line");
        assert!(out.contains("let a = 1;"), "shows deleted line");
        assert!(!out.contains("keep;"), "unchanged lines are omitted");
    }

    #[test]
    fn non_edit_tool_renders_status_line() {
        let out = render_tool_end("bash", 0, "hello\nworld", None);
        assert!(out.contains("bash") && out.contains("hello"));
        assert!(!out.contains('✎'), "no diff marker for non-edit tools");
    }
}
