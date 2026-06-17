# Inbound Channels — Design Doc (eve parity)

Status: DESIGN ONLY. No implementation. This document specifies the seam for
turning external inbound messages (HTTP, Slack, Discord) into full harness turns
on a filesystem-first agent directory. It is the channel companion to the cron
`schedule` layer that already ships under the `serve` feature.

Scope anchors (already in tree):

- `core/src/config/agent_dir.rs` — parses `channels/*.{md,acl}` into
  `ChannelSpec { name, kind, frontmatter }` and stores them on
  `AgentDir { channels: Vec<ChannelSpec>, .. }`. This is done; this doc does NOT
  change the parser.
- `core/src/serve/schedule.rs` — `ScheduleSink` trait + `Scheduler` (cron loop).
- `core/src/serve/daemon.rs` — `SessionScheduleSink` + `serve_agent_dir(..)`.
- `core/src/serve/mod.rs` — `serve` feature module, re-exports.

The channel layer MIRRORS the schedule layer. Where a schedule is a *time*
source that fires a fixed prompt, a channel is a *message* source that fires an
arriving message. Both converge on the SAME invariant and the SAME execution
primitive (`AgentSession::send`).

---

## 0. The core invariant (read this first)

> Every inbound channel message becomes a FULL harness turn via
> `AgentSession::send`, never a raw model call.

This is the entire reason the layer exists, and it is non-negotiable. An adapter
NEVER touches `LlmClient`, never assembles a prompt string and calls a provider,
never short-circuits tool/visibility/safety. The ONLY thing an adapter is
allowed to do with an inbound message is:

```rust
let result: AgentResult = session.send(&message.text, None).await?;
```

`AgentSession::send(&self, prompt: &str, history: Option<&[Message]>) -> Result<AgentResult>`
is what carries:

- **Context**: the agent dir's `instructions.md` (injected as a prompt SLOT via
  `SessionOptions::prompt_slots`, not a system-prompt override), the session's
  accumulated history (`history = None` => use+update the session's own history),
  and any `skill_dirs`/context providers configured on the session.
- **Tool visibility**: tools are harness-owned. The adapter does not declare,
  hide, or inject tools — this is the deliberate divergence from eve's
  user-defined-tools model already documented in `agent_dir.rs`.
- **Safety gate**: permission checker, security provider (taint/sanitization),
  HITL confirmation, and budget guard all run inside `send`.
- **Verification**: `AgentResult.verification_reports` is populated by the
  harness's completion-evidence path; the adapter may surface it but never
  replaces it.

A reviewer can enforce the invariant with one grep: an adapter module that
imports `crate::llm::LlmClient` (other than to *forward* a host-supplied one into
`SessionOptions`) is wrong. The only execution call in any adapter is
`session.send(..)`.

Inbound text is UNTRUSTED. It enters as the `prompt` argument of a turn, which
means the harness's existing prompt-injection / taint defenses (security
provider, BOUNDARIES) are exactly the defenses that apply. Adapters add transport
authentication on top (signature verification), they do not add or weaken
model-layer safety.

---

## 1. The `ChannelAdapter` trait seam

Lives in a new file `core/src/serve/channel.rs`, parallel to `schedule.rs`.

Two responsibilities, split into two traits to keep transport concerns out of
the session-routing core (single responsibility, mirrors how `ScheduleSink` is
separate from `Scheduler`):

### 1.1 `ChannelSink` — the harness-binding seam (core, stable)

The exact analogue of `ScheduleSink`. The serve daemon implements it; it is the
ONLY place a turn is fired. Adapters depend on this trait, never on
`AgentSession` directly — so the "must go through `send`" rule is structurally
enforced (an adapter literally has no other method to call).

```rust
/// An inbound message normalized from any transport. The adapter produces it;
/// the sink turns it into a harness turn.
#[derive(Debug, Clone)]
pub struct InboundMessage {
    /// Logical channel name (the `ChannelSpec.name`), e.g. "support-web".
    pub channel: String,
    /// Transport-stable conversation key used to derive the session id.
    /// HTTP: caller-supplied conversation id (or a per-connection uuid).
    /// Slack: channel_id + thread_ts. Discord: channel_id (+ thread id).
    pub conversation: String,
    /// The user's text — fired verbatim as the turn prompt.
    pub text: String,
    /// Opaque principal (Slack user id, Discord author id, HTTP auth subject).
    /// Forwarded into SessionOptions.principal; never interpreted by core.
    pub principal: Option<String>,
}

/// What to do when a channel message arrives. The serve daemon implements this
/// to drive the message into `AgentSession::send` — a FULL harness turn
/// (context, tool visibility, safety gate, verification), never a raw model call.
#[async_trait::async_trait]
pub trait ChannelSink: Send + Sync {
    /// Fire one inbound message as a harness turn and return the reply.
    /// Returns the reply so adapters can route it back (see §4).
    async fn deliver(&self, msg: InboundMessage) -> crate::error::Result<ChannelReply>;
}

/// The harness's answer, shaped for routing back to the transport.
#[derive(Debug, Clone)]
pub struct ChannelReply {
    /// AgentResult.text — the text to send back to the user.
    pub text: String,
    /// Echo of the conversation key, so the adapter knows where to route.
    pub conversation: String,
    /// True when verification is NeedsReview (adapter may flag the reply).
    pub needs_review: bool,
}
```

Note `deliver` RETURNS the reply rather than the sink pushing it. This keeps the
sink transport-agnostic (it knows nothing about Slack `chat.postMessage` or HTTP
response bodies) and lets each adapter route replies in its native idiom (§4).

### 1.2 `ChannelAdapter` — the transport seam (extension, per-protocol)

One impl per protocol. Owns the socket/listener, authenticates the transport,
normalizes wire payloads into `InboundMessage`, calls `sink.deliver(..)`, and
routes `ChannelReply` back. It runs until cancelled — exactly like a
`Scheduler::run` job loop.

```rust
#[async_trait::async_trait]
pub trait ChannelAdapter: Send + Sync {
    /// Stable kind string matched against `ChannelSpec.kind` ("http"/"slack"/"discord").
    fn kind(&self) -> &'static str;

    /// Bind the transport and serve inbound messages until `cancel` fires.
    /// Every accepted message is normalized and handed to `sink.deliver(..)`;
    /// the returned `ChannelReply` is routed back over the transport.
    async fn serve(
        &self,
        spec: &crate::config::ChannelSpec,
        sink: std::sync::Arc<dyn ChannelSink>,
        cancel: tokio_util::sync::CancellationToken,
    ) -> crate::error::Result<()>;
}
```

Adapter construction parses `ChannelSpec.frontmatter` (raw YAML) into a
per-adapter typed options struct (`HttpOptions`, `SlackOptions`, `DiscordOptions`)
with `serde`. The frontmatter is already captured by the parser; adapters own its
interpretation, consistent with the `agent_dir.rs` doc comment ("`frontmatter`
carries the raw adapter options for whichever adapter eventually handles `kind`").

Why two traits and not one: `ChannelSink` is **core** (per Rule 2 — it is the
harness-binding contract, non-replaceable). `ChannelAdapter` is an **extension** —
each transport is replaceable/addable without touching the sink. A host could
ship an SQS or WhatsApp adapter by implementing `ChannelAdapter` alone.

---

## 2. The three inbound adapters

All three are feature-gated sub-modules under `core/src/serve/channels/` and
share the §1 traits. Each is one file (one protocol per file). They differ ONLY
in transport + auth + reply routing; the body of each handler is the same three
steps: normalize -> `sink.deliver` -> route reply.

### 2.1 HTTP (`channels/http.rs`, kind = `"http"`)

- **Transport**: a small HTTP listener (server side via a gated `axum` dep — see
  §6; `reqwest`/`tokio` are already deps). One `POST /message` endpoint.
- **Frontmatter** (`HttpOptions`): `port: u16`, optional `path: String`
  (default `/message`), optional `auth_token: String` (bearer) or
  `auth_token_env: String`.
- **Auth**: constant-time bearer-token compare; 401 on mismatch. No token =>
  bind loopback only and log a warning (never expose an unauthenticated agent to
  `0.0.0.0`).
- **Request body**: `{ "conversation": "<id?>", "text": "..." }`. Missing
  `conversation` => generate a per-request uuid (stateless one-shot turn).
- **Reply routing**: synchronous — the `ChannelReply.text` is the HTTP 200 JSON
  body `{ "text": "...", "needs_review": bool }`. This is the simplest adapter
  and the reference impl for the trait.

### 2.2 Slack (`channels/slack.rs`, kind = `"slack"`)

- **Transport**: Slack Events API over HTTP (recommended for parity with eve and
  to avoid a socket-mode websocket dependency). The adapter exposes one webhook
  endpoint Slack POSTs events to.
- **Frontmatter** (`SlackOptions`): `signing_secret_env`, `bot_token_env`,
  optional `port`/`path`, optional `event_types` filter (default
  `app_mention` + `message.im`).
- **Auth**: verify the `X-Slack-Signature` HMAC-SHA256 over
  `v0:{X-Slack-Request-Timestamp}:{body}` using the signing secret; reject stale
  timestamps (replay guard). Respond to Slack's `url_verification` challenge.
- **Normalize**: `conversation = format!("{channel_id}:{thread_ts_or_ts}")` so a
  thread maps to one session; strip the bot mention from `text`; `principal =
  event.user`. ACK Slack within 3s (return 200 immediately) and run `deliver` on
  a spawned task — Slack retries on slow ACK, so the turn must not block the ACK.
- **Reply routing**: asynchronous — post `ChannelReply.text` back via
  `chat.postMessage` to the same channel/thread using the bot token. This is the
  first adapter where reply routing is decoupled from the inbound request, which
  is exactly why `deliver` returns the reply rather than writing a response body.

### 2.3 Discord (`channels/discord.rs`, kind = `"discord"`)

- **Transport**: Discord Gateway websocket (Discord has no inbound-webhook model
  for receiving messages; a bot must hold a gateway connection). The adapter
  maintains the gateway heartbeat and subscribes to `MESSAGE_CREATE`.
- **Frontmatter** (`DiscordOptions`): `bot_token_env`, optional
  `application_id`, optional `allowed_channels: Vec<String>`, intent flags.
- **Auth**: the bot token authenticates the gateway connection itself; inbound
  messages are trusted as gateway-delivered. Ignore messages authored by the bot
  (loop guard) and messages outside `allowed_channels`.
- **Normalize**: `conversation = channel_id` (or thread id when present);
  `text = message.content`; `principal = author.id`.
- **Reply routing**: asynchronous — `POST /channels/{id}/messages` with
  `ChannelReply.text`.

Adapter parity summary:

| kind    | transport            | inbound auth         | reply routing            | session key |
|---------|----------------------|----------------------|--------------------------|-------------|
| http    | HTTP listener        | bearer token         | sync HTTP 200 body       | conversation id / per-req uuid |
| slack   | Events API webhook   | HMAC signature       | async `chat.postMessage` | channel:thread_ts |
| discord | Gateway websocket    | bot-token connection | async REST `messages`    | channel/thread id |

---

## 3. Session id mapping (one session per conversation)

The schedule layer keys sessions by schedule name (`schedule:<name>`). Channels
key sessions by **conversation**, because that is what should accumulate history.

```text
session_id = format!("channel:{}:{}", msg.channel, msg.conversation)
```

- `msg.channel` is the `ChannelSpec.name` — isolates two different channels that
  happen to share a conversation id.
- `msg.conversation` is the transport-stable thread key from §2.

Properties this gives us, by mirroring the schedule design:

- **Continuity**: repeated messages in the same thread reuse the same
  `AgentSession`, so `send(text, None)` accumulates context across turns (the
  harness owns history; the adapter passes `None`).
- **Isolation**: distinct threads / channels get distinct sessions; no
  cross-talk.
- **Durability (later)**: because the id is stable and derived (not random), the
  same `SessionStore`-backed rehydrate-on-boot path the daemon doc mentions for
  schedules applies unchanged — a restarted daemon resumes a conversation's
  session by recomputing its id.

Session lifecycle: unlike schedules (fixed, known at boot), channel
conversations are dynamic and unbounded. The `SessionChannelSink` therefore
holds a `tokio::sync::Mutex<HashMap<String, Arc<AgentSession>>>` and lazily
creates a session on first message for a conversation (get-or-create under the
lock), reusing the agent dir's `prompt_slots` + `skill_dirs` exactly as
`serve_agent_dir` does for schedules. A retention/idle policy (LRU cap, idle
eviction via `session.close()`) is REQUIRED to avoid unbounded session growth —
this is the one place channels need machinery schedules do not. Recommend a
configurable `max_live_sessions` with LRU close; default conservative.

---

## 4. Reply routing

The harness produces `AgentResult.text`. The sink wraps it as
`ChannelReply { text, conversation, needs_review }` and returns it from
`deliver`. The ADAPTER routes it, because only the adapter knows the transport:

- **HTTP**: serialize the reply as the synchronous HTTP response body. The
  caller's `conversation` is echoed so a stateless client can correlate.
- **Slack/Discord**: the inbound request was already ACKed (Slack) or is a
  fire-and-forget gateway event (Discord); the adapter routes the reply
  out-of-band to `conversation` via the platform's send API, using credentials
  from the adapter's frontmatter (`bot_token_env`).

`needs_review` is derived from `AgentResult.verification_reports` (true when the
summary status is `NeedsReview`). Adapters MAY use it (e.g. prefix the Slack
reply with a warning, or set an HTTP header) but MUST NOT drop the reply — the
harness already decided the turn completed; surfacing review state is a
presentation concern.

Errors: when `deliver` returns `Err` (closed session, send failure), HTTP
returns 5xx; Slack/Discord log and post a terse "couldn't process that" to the
conversation. The error is never the raw `CodeError` (don't leak internals to an
untrusted channel).

---

## 5. Where adapters plug into `serve_agent_dir`

Today `serve_agent_dir` builds per-schedule sessions, then runs one `Scheduler`.
Channels attach in the SAME function as a second concurrent driver, joined under
the same `cancel` token. Sketch (matches existing signatures —
`Agent::session(workspace, Some(SessionOptions))`, `AgentSession::send`,
`async_trait`):

```rust
// in core/src/serve/daemon.rs, inside serve_agent_dir(..), after the schedule wiring:

// 1. One sink shared by every adapter; it owns lazy per-conversation sessions
//    and is the ONLY caller of AgentSession::send for channels.
let channel_sink: Arc<dyn ChannelSink> = Arc::new(SessionChannelSink::new(
    agent,                           // to build sessions on demand (Agent::session takes &self)
    agent_dir.clone(),               // prompt_slots + skill_dirs source
    workspace.clone().into(),
    extra.clone(),                   // merged SessionOptions, same rules as schedules
));

// 2. One adapter task per enabled channel, selected by kind.
let mut adapter_handles = Vec::new();
for spec in &agent_dir.channels {
    let adapter: Arc<dyn ChannelAdapter> = match spec.kind.as_str() {
        "http"    => Arc::new(HttpChannelAdapter::default()),
        "slack"   => Arc::new(SlackChannelAdapter::default()),
        "discord" => Arc::new(DiscordChannelAdapter::default()),
        other => { tracing::warn!(kind = %other, "unknown channel kind; skipping"); continue; }
    };
    let (spec, sink, cancel) = (spec.clone(), Arc::clone(&channel_sink), cancel.clone());
    adapter_handles.push(tokio::spawn(async move {
        if let Err(e) = adapter.serve(&spec, sink, cancel).await {
            tracing::warn!(channel = %spec.name, error = %e, "channel adapter stopped");
        }
    }));
}

// 3. Run schedules and channels concurrently; both stop on `cancel`.
//    (Today the fn ends `scheduler.run(sink, cancel).await; Ok(())`. The new form
//     joins the scheduler future with the adapter handles under the same token.)
```

Key points:

- `SessionChannelSink` is the `ChannelSink` impl and lives in `daemon.rs` next to
  `SessionScheduleSink`, for the same reason: it is the one place that holds
  `AgentSession`s and fires `send`. The dispatch-by-`kind` match is the channel
  analogue of `Scheduler::new(specs)`.
- Per-channel `SessionOptions` follow the EXACT merge rules already in
  `serve_agent_dir`: `prompt_slots` defaults to `agent_dir.prompt_slots` if the
  host didn't pin one; `skill_dirs` extends with `agent_dir.config.skill_dirs`;
  `session_id` is set by the sink per conversation (§3); `principal` is filled
  from `InboundMessage.principal`.
- No new execution machinery: the daemon stays a thin wiring layer over
  `Agent`/`AgentSession`, consistent with `serve/mod.rs`'s "builds strictly on
  top of existing primitives" promise.

---

## 6. Feature gating

Channels stay under the existing `serve` feature, but transport deps are split so
a host can take HTTP without dragging in Slack/Discord clients. Proposed
`Cargo.toml` additions (mirrors the existing `serve = ["dep:cron"]` shape):

```toml
[features]
# unchanged: cron schedules
serve = ["dep:cron"]

# Channel transports — each additive, each pulls only what it needs.
serve-channels = ["serve"]                                  # the ChannelAdapter/ChannelSink seam
serve-http     = ["serve-channels", "dep:axum"]             # HTTP inbound adapter
serve-slack    = ["serve-channels", "dep:axum", "dep:hmac", "dep:sha2"]  # Events API + signature
serve-discord  = ["serve-channels", "dep:tokio-tungstenite"]            # gateway websocket
```

- `core/src/serve/channel.rs` (the traits) is gated on `serve-channels`.
- `core/src/serve/channels/http.rs` on `serve-http`, `slack.rs` on
  `serve-slack`, `discord.rs` on `serve-discord`.
- The dispatch `match` in `serve_agent_dir` `#[cfg(..)]`s each arm so an
  unbuilt transport is a clean "unknown kind; skipping" rather than a build
  break.
- `serve/mod.rs` re-exports `ChannelSink`, `ChannelAdapter`, `InboundMessage`,
  `ChannelReply` under `#[cfg(feature = "serve-channels")]`, alongside the
  existing `ScheduleSink`/`Scheduler` exports.
- Library-only embedders with no `serve*` feature pay nothing — the parser
  (`ChannelSpec`) is always present (it is just data), but no transport,
  no server deps, no adapter code compiles.

---

## 7. Test plan (TDD, no live network)

Following the schedule layer's test style (sink counters, pre-cancelled tokens):

1. **Sink invariant** — test `SessionChannelSink` with a record/replay
   `LlmClient` injected via `SessionOptions.llm_client`, assert `deliver(msg)`
   produces a session whose id is `channel:<name>:<conversation>` and that the
   reply text equals the stubbed model text. Proves the message went through
   `send` (history grew, verification ran), not a raw call.
2. **Session reuse/isolation** — two `deliver`s with the same conversation reuse
   one session (history length grows); different conversations get distinct
   sessions.
3. **Adapter normalization (offline)** — feed each adapter a canned wire payload
   (a captured Slack event JSON, a Discord `MESSAGE_CREATE`, an HTTP body) and
   assert the produced `InboundMessage` fields (conversation key, stripped text,
   principal). No socket — the normalize function is pure and unit-testable.
4. **Auth** — Slack signature verify accepts a correctly-signed fixture and
   rejects a tampered body / stale timestamp; HTTP bearer compare rejects a bad
   token.
5. **Cancellation** — `adapter.serve(spec, sink, pre_cancelled_token)` returns
   `Ok(())` promptly without binding (mirrors
   `serve_builds_per_schedule_session_and_stops_on_cancel`).
6. **Unknown kind** — `serve_agent_dir` with a `ChannelSpec { kind: "telegram" }`
   logs and skips, does not error.

No test leaves a socket bound or a temp file behind (CLAUDE.md TDD rule).

---

## 8. What this design deliberately does NOT do

- No user-defined tools per channel (harness owns tools — the documented eve
  divergence).
- No bypass path: there is no API on `ChannelSink` other than `deliver`, and
  `deliver` only calls `send`. A raw-model fast path is structurally impossible.
- No new persistence model: durable rehydration rides the same stable-session-id
  + `SessionStore` mechanism the daemon already plans for schedules.
- No outbound-only "notification" channels — out of scope; this is the *inbound*
  message seam. (A schedule that posts to Slack is the outbound story and already
  works via a tool inside a scheduled turn.)
- No per-message model override — model is the session/agent-dir concern, not a
  transport concern.

---

## 9. Open design questions (flagged for review, not blocking the seam)

1. Reply for multi-turn tool-using sessions can be slow; Slack/Discord want a
   fast ACK + async reply (handled), but HTTP's synchronous reply could hit
   client timeouts. Option: HTTP gains an optional async mode (`202 Accepted` +
   callback URL). Deferred — sync is the correct reference default.
2. Session eviction policy numbers (`max_live_sessions`, idle TTL) — needs a
   real-workload default; the mechanism (LRU + `close()`) is settled, the
   constants are not.
3. Whether `InboundMessage` should carry attachments (Slack files, Discord
   embeds) to use `send_with_attachments` instead of `send`. The seam allows it
   (add a field + branch in the sink) but v1 is text-only for parity.
