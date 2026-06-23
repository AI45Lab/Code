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

use a3s_code_core::{Agent, AgentEvent, AgentSession, SessionOptions};
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
    /// Submitted prompts, oldest first, for ↑/↓ recall.
    history: Vec<String>,
    /// Cursor into `history` while browsing; `None` means "fresh input".
    history_pos: Option<usize>,
    /// Model name reported by the provider (captured from the first turn).
    model: Option<String>,
    /// Cumulative tokens used this session.
    total_tokens: usize,
    /// Accumulated streamed JSON args of the in-progress tool call, so the
    /// result line can show what the tool actually did (command/path/pattern).
    tool_args: String,
    /// Live stdout of the in-progress tool (e.g. a running command), shown
    /// dimmed under the action and cleared when the tool completes.
    tool_output: String,
    /// When true, tool-confirmation prompts are auto-approved (Codex-style
    /// approval mode), toggled with `/auto`.
    auto_approve: bool,
    /// Working directory shown for context.
    cwd: String,
    width: u16,
    height: u16,
    keymap: Keymap<Action>,
}

impl Model for App {
    type Msg = Msg;

    fn init(&mut self) -> Option<Cmd<Msg>> {
        if self.messages.is_empty() {
            let welcome = Style::new()
                .fg(Color::BrightBlack)
                .italic()
                .render(&format!(
                    "  A3S Code — {}\n  Type a message and press Enter.\n  \
                 ↑/↓ history · Esc interrupt · /help · Ctrl+C quit\n",
                    self.cwd
                ));
            self.viewport.set_content(&welcome);
        } else {
            // Resumed session — show the prior conversation, scrolled to the end.
            self.rebuild_viewport();
            self.viewport.update(ViewportMsg::Bottom);
        }
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
                // ↑/↓ recall prompt history (single-line input only, so multi-line
                // editing keeps normal cursor movement).
                if matches!(key.code, KeyCode::Up | KeyCode::Down)
                    && !self.textarea.value().contains('\n')
                    && !self.history.is_empty()
                {
                    self.history_recall(key.code == KeyCode::Up);
                    return None;
                }
                if let Some(TextareaMsg::Submit(text)) = self.textarea.handle_key(&key) {
                    return Some(cmd::msg(Msg::Submit(text)));
                }
            }

            Msg::Term(Event::Mouse(m)) => {
                use a3s_tui::event::MouseEventKind;
                match m.kind {
                    MouseEventKind::ScrollUp => self.viewport.update(ViewportMsg::ScrollUp(3)),
                    MouseEventKind::ScrollDown => self.viewport.update(ViewportMsg::ScrollDown(3)),
                    _ => {}
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
        let mut right = String::new();
        if let Some(model) = &self.model {
            right.push_str(model);
            right.push_str(" · ");
        }
        if self.total_tokens > 0 {
            right.push_str(&format!("{} tok · ", self.total_tokens));
        }
        right.push_str("Ctrl+C quit ");
        let status = StatusBar::new()
            .left(&status_text)
            .right(&right)
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
                        "  commands: /clear reset · /auto toggle auto-approve · /exit quit\n  \
                     Enter send · ↑/↓ history · Esc interrupt · Ctrl+C quit · PgUp/PgDn scroll",
                    ));
                self.textarea.clear();
                self.rebuild_viewport();
                return None;
            }
            "/auto" => {
                self.auto_approve = !self.auto_approve;
                let state = if self.auto_approve { "on" } else { "off" };
                self.messages.push(
                    Style::new()
                        .fg(Color::Yellow)
                        .render(&format!("  ⚡ auto-approve: {state}")),
                );
                self.textarea.clear();
                self.rebuild_viewport();
                return None;
            }
            _ => {}
        }

        self.history.push(trimmed.to_string());
        self.history_pos = None;
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
                self.tool_args.clear();
                self.tool_output.clear();
                self.push_line(&Style::new().fg(Color::Cyan).render(&format!("  ⚙ {name}")));
            }
            AgentEvent::ToolInputDelta { delta } => {
                self.tool_args.push_str(&delta);
            }
            AgentEvent::ToolOutputDelta { delta, .. } => {
                self.tool_output.push_str(&delta);
                self.update_viewport_with_stream();
            }
            AgentEvent::ToolEnd {
                name,
                output,
                exit_code,
                metadata,
                ..
            } => {
                let args: Option<serde_json::Value> = serde_json::from_str(&self.tool_args).ok();
                self.push_line(&render_tool_end(
                    &name,
                    exit_code,
                    &output,
                    metadata.as_ref(),
                    args.as_ref(),
                    self.width as usize,
                ));
                self.tool_args.clear();
                self.tool_output.clear();
            }
            AgentEvent::SubagentStart {
                agent, description, ..
            } => {
                self.finalize_streaming();
                self.push_line(
                    &Style::new()
                        .fg(Color::Magenta)
                        .render(&format!("  ↳ subagent {agent}: {description}")),
                );
            }
            AgentEvent::SubagentEnd { agent, success, .. } => {
                let mark = if success { "✓" } else { "✗" };
                self.push_line(
                    &Style::new()
                        .fg(Color::Magenta)
                        .render(&format!("  ↳ {mark} subagent {agent} done")),
                );
            }
            AgentEvent::ConfirmationRequired {
                tool_id,
                tool_name,
                args,
                ..
            } => {
                if self.auto_approve {
                    self.push_line(
                        &Style::new()
                            .fg(Color::BrightBlack)
                            .render(&format!("  ⚡ auto-approved {tool_name}")),
                    );
                    let session = self.session.clone();
                    return Some(cmd::batch(vec![
                        cmd::cmd(move || async move {
                            let _ = session.confirm_tool_use(&tool_id, true, None).await;
                            Msg::Resume
                        }),
                        spinner_tick(),
                    ]));
                }
                self.state = State::Awaiting;
                self.pending_tool = Some((tool_id, tool_name.clone()));
                let pretty =
                    serde_json::to_string_pretty(&args).unwrap_or_else(|_| args.to_string());
                let body = format!("Tool: {tool_name}\n{}", truncate(&pretty, 400));
                self.modal = Some(
                    Modal::new()
                        .title("Approve tool call?")
                        .body(&body)
                        .options(vec!["Allow", "Deny"]),
                );
                return None; // wait for the user; do not pump
            }
            AgentEvent::End {
                text, usage, meta, ..
            } => {
                if self.streaming.raw_content().trim().is_empty() && !text.is_empty() {
                    self.streaming.push(&text);
                }
                self.finalize_streaming();
                self.total_tokens += usage.total_tokens;
                if self.model.is_none() {
                    self.model = meta.and_then(|m| m.response_model.or(m.request_model));
                }
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

    /// Move through prompt history and load the entry into the input. Going
    /// forward past the newest entry returns to a fresh, empty input.
    fn history_recall(&mut self, up: bool) {
        let pos = match (self.history_pos, up) {
            (None, true) => self.history.len().saturating_sub(1),
            (None, false) => return,
            (Some(i), true) => i.saturating_sub(1),
            (Some(i), false) => i + 1,
        };
        if pos >= self.history.len() {
            self.history_pos = None;
            self.textarea.clear();
        } else {
            self.history_pos = Some(pos);
            self.textarea.set_value(&self.history[pos]);
        }
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
        // Live stdout of the running tool — show the tail like a terminal.
        if !self.tool_output.trim().is_empty() {
            let tail: Vec<&str> = self.tool_output.lines().rev().take(12).collect();
            let tail = tail.into_iter().rev().collect::<Vec<_>>().join("\n");
            blocks.push(Style::new().fg(Color::BrightBlack).render(&tail));
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
    let (mut rx, join) = session.stream(prompt.as_str(), None).await?;
    while let Some(event) = rx.recv().await {
        match event {
            AgentEvent::TextDelta { text } => print!("{text}"),
            AgentEvent::ToolStart { name, .. } => eprintln!("\n[tool start] {name}"),
            AgentEvent::ToolEnd {
                name,
                exit_code,
                output,
                ..
            } => eprintln!(
                "[tool end] {name} (exit {exit_code}): {}",
                output.lines().take(2).collect::<Vec<_>>().join(" | ")
            ),
            AgentEvent::ConfirmationRequired {
                tool_id, tool_name, ..
            } => {
                eprintln!("[confirm] auto-allowing {tool_name}");
                let _ = session.confirm_tool_use(&tool_id, true, None).await;
            }
            AgentEvent::End { .. } => eprintln!("\n[end]"),
            AgentEvent::Error { message } => eprintln!("\n[error] {message}"),
            _ => {}
        }
    }
    // Let the stream task finish (incl. auto-save/persist) before we exit.
    let _ = join.await;
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
    args: Option<&serde_json::Value>,
    width: usize,
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
    // Show the tool's primary argument (command/path/pattern) so the action log
    // reads like Codex — "✓ bash — npm test" rather than just "✓ bash".
    let header = Style::new()
        .fg(Color::BrightBlack)
        .render(&match args.and_then(arg_summary) {
            Some(summary) => format!("  {status} {name} — {summary}"),
            None => format!("  {status} {name}"),
        });
    let head = output.lines().take(8).collect::<Vec<_>>().join("\n");
    if head.trim().is_empty() {
        return header;
    }
    // If the output is file/code content (read/edit on a known extension),
    // syntax-highlight it; otherwise show it dimmed.
    if exit_code == 0 {
        if let Some(lang) = args
            .and_then(|a| {
                a.get("file_path")
                    .or_else(|| a.get("path"))
                    .and_then(|v| v.as_str())
            })
            .and_then(lang_from_path)
        {
            let fenced = format!("```{lang}\n{head}\n```");
            let rendered = a3s_tui::markdown::Markdown::new()
                .with_width(width.saturating_sub(4).max(20))
                .render(&fenced);
            return format!("{header}\n{rendered}");
        }
    }
    format!(
        "{header}\n{}",
        Style::new().fg(Color::BrightBlack).render(&head)
    )
}

/// Map a file path to a syntect language token for fenced rendering.
fn lang_from_path(path: &str) -> Option<&'static str> {
    let ext = path.rsplit('.').next()?;
    Some(match ext {
        "rs" => "rust",
        "py" => "python",
        "js" | "mjs" | "cjs" => "javascript",
        "ts" | "tsx" => "typescript",
        "go" => "go",
        "json" => "json",
        "toml" => "toml",
        "yaml" | "yml" => "yaml",
        "md" => "markdown",
        "sh" | "bash" => "bash",
        "c" | "h" => "c",
        "cpp" | "cc" | "hpp" => "cpp",
        "java" => "java",
        "rb" => "ruby",
        "html" => "html",
        "css" => "css",
        "sql" => "sql",
        _ => return None,
    })
}

/// Extract a one-line summary of a tool's primary argument.
fn arg_summary(args: &serde_json::Value) -> Option<String> {
    for key in [
        "command",
        "file_path",
        "path",
        "pattern",
        "query",
        "url",
        "old_string",
    ] {
        if let Some(v) = args.get(key).and_then(|v| v.as_str()) {
            let v = v.replace('\n', " ");
            return Some(truncate(v.trim(), 120));
        }
    }
    None
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

    // Persistent, resumable session: stored under <cwd>/.a3s/tui-sessions and
    // keyed by a fixed id, so relaunching in the same directory continues the
    // conversation. Falls back to a fresh session when none exists yet.
    let store_dir = std::path::Path::new(&workspace).join(".a3s/tui-sessions");
    let store: Arc<dyn a3s_code_core::store::SessionStore> = Arc::new(
        a3s_code_core::store::FileSessionStore::new(&store_dir)
            .await
            .map_err(|e| anyhow::anyhow!("failed to open session store {store_dir:?}: {e}"))?,
    );
    const SESSION_ID: &str = "tui-default";
    let session = match agent.resume_session(
        SESSION_ID,
        SessionOptions::new()
            .with_session_store(store.clone())
            .with_auto_save(true),
    ) {
        Ok(s) => s,
        Err(_) => agent.session(
            workspace.clone(),
            Some(
                SessionOptions::new()
                    .with_session_store(store.clone())
                    .with_session_id(SESSION_ID)
                    .with_auto_save(true),
            ),
        )?,
    };

    // Seed the transcript with any resumed conversation (user + assistant text).
    let initial_messages: Vec<String> = session
        .history()
        .iter()
        .filter_map(|m| {
            let text = m.text();
            if text.trim().is_empty() {
                return None;
            }
            match m.role.as_str() {
                "user" => Some(
                    Style::new()
                        .bold()
                        .fg(Color::BrightGreen)
                        .render(&format!("❯ {}", text.trim())),
                ),
                "assistant" => Some(text),
                _ => None,
            }
        })
        .collect();

    let session = Arc::new(session);

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
        messages: initial_messages,
        rx: None,
        modal: None,
        pending_tool: None,
        history: Vec::new(),
        history_pos: None,
        model: None,
        total_tokens: 0,
        tool_args: String::new(),
        tool_output: String::new(),
        auto_approve: false,
        cwd: workspace,
        width,
        height,
        keymap,
    };

    ProgramBuilder::new(app)
        .with_alt_screen()
        .with_mouse_support()
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
        let out = render_tool_end("edit", 0, "ok", Some(&meta), None, 80);
        assert!(out.contains("src/x.rs"), "header has path");
        assert!(out.contains("+1") && out.contains("-1"), "add/del counts");
        assert!(out.contains("let a = 2;"), "shows inserted line");
        assert!(out.contains("let a = 1;"), "shows deleted line");
        assert!(!out.contains("keep;"), "unchanged lines are omitted");
    }

    #[test]
    fn non_edit_tool_renders_status_line() {
        let out = render_tool_end("bash", 0, "hello\nworld", None, None, 80);
        assert!(out.contains("bash") && out.contains("hello"));
        assert!(!out.contains('✎'), "no diff marker for non-edit tools");
    }

    #[test]
    fn tool_end_shows_primary_arg_summary() {
        let args = serde_json::json!({ "command": "npm test", "timeout": 60 });
        let out = render_tool_end("bash", 0, "ok\n", None, Some(&args), 80);
        assert!(out.contains("bash"));
        assert!(out.contains("npm test"), "shows the command argument");
    }

    #[test]
    fn arg_summary_extracts_known_keys() {
        assert_eq!(
            arg_summary(&serde_json::json!({ "command": "ls -la" })),
            Some("ls -la".to_string())
        );
        assert_eq!(
            arg_summary(&serde_json::json!({ "pattern": "TODO" })),
            Some("TODO".to_string())
        );
        assert_eq!(arg_summary(&serde_json::json!({ "unknown": "x" })), None);
    }
}
