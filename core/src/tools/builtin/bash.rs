//! Bash tool - Execute shell commands

use crate::tools::types::{Tool, ToolContext, ToolEventSender, ToolOutput, ToolStreamEvent};
use crate::workspace::{CommandOutputObserver, CommandRequest};
use anyhow::Result;
use async_trait::async_trait;
#[cfg(windows)]
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use tokio::process::Command;

/// Default timeout in seconds (2 minutes)
pub(crate) const DEFAULT_TIMEOUT_MS: u64 = 120_000;

/// Adapter that forwards `CommandOutputObserver` deltas to a tool event channel.
///
/// Keeps `workspace::CommandRequest` free of `ToolEventSender`. Constructed in
/// the bash tool when a session has installed an event channel; backend
/// implementations only see `&dyn CommandOutputObserver`.
struct ToolEventObserver {
    tx: ToolEventSender,
}

#[async_trait]
impl CommandOutputObserver for ToolEventObserver {
    async fn on_output_delta(&self, delta: &str) {
        self.tx
            .send(ToolStreamEvent::OutputDelta(delta.to_string()))
            .await
            .ok();
    }
}

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[cfg(windows)]
const WINDOWS_POWERSHELL_COMPAT_SHIM: &str = r#"
$ProgressPreference = 'SilentlyContinue'

if (Test-Path Alias:curl) {
    Remove-Item Alias:curl -Force -ErrorAction SilentlyContinue
}

if (Test-Path Alias:wget) {
    Remove-Item Alias:wget -Force -ErrorAction SilentlyContinue
}

function __a3s_json_escape {
    param([string]$Value)
    if ($null -eq $Value) {
        return ''
    }

    $escaped = $Value.Replace('\', '\\').Replace('"', '\"')
    $escaped = $escaped.Replace("`r", '\r').Replace("`n", '\n').Replace("`t", '\t')
    return $escaped
}

function __a3s_json_string {
    param([string]$Value)
    return '"' + (__a3s_json_escape $Value) + '"'
}

function __a3s_split_top_level {
    param([string]$Text)

    $parts = @()
    $current = New-Object System.Text.StringBuilder
    $braceDepth = 0
    $bracketDepth = 0
    $inSingle = $false
    $inDouble = $false

    for ($i = 0; $i -lt $Text.Length; $i++) {
        $ch = $Text[$i]

        if ($ch -eq "'" -and -not $inDouble) {
            $inSingle = -not $inSingle
            [void]$current.Append($ch)
            continue
        }

        if ($ch -eq '"' -and -not $inSingle) {
            $escaped = $i -gt 0 -and $Text[$i - 1] -eq '\'
            if (-not $escaped) {
                $inDouble = -not $inDouble
            }
            [void]$current.Append($ch)
            continue
        }

        if (-not $inSingle -and -not $inDouble) {
            if ($ch -eq '{') {
                $braceDepth += 1
                [void]$current.Append($ch)
                continue
            } elseif ($ch -eq '}') {
                $braceDepth -= 1
                [void]$current.Append($ch)
                continue
            } elseif ($ch -eq '[') {
                $bracketDepth += 1
                [void]$current.Append($ch)
                continue
            } elseif ($ch -eq ']') {
                $bracketDepth -= 1
                [void]$current.Append($ch)
                continue
            } elseif ($ch -eq ',' -and $braceDepth -eq 0 -and $bracketDepth -eq 0) {
                $parts += $current.ToString()
                [void]$current.Clear()
                continue
            }
        }

        [void]$current.Append($ch)
    }

    if ($current.Length -gt 0) {
        $parts += $current.ToString()
    }

    return ,$parts
}

function __a3s_split_first_colon {
    param([string]$Text)

    $braceDepth = 0
    $bracketDepth = 0
    $inSingle = $false
    $inDouble = $false

    for ($i = 0; $i -lt $Text.Length; $i++) {
        $ch = $Text[$i]

        if ($ch -eq "'" -and -not $inDouble) {
            $inSingle = -not $inSingle
            continue
        }

        if ($ch -eq '"' -and -not $inSingle) {
            $escaped = $i -gt 0 -and $Text[$i - 1] -eq '\'
            if (-not $escaped) {
                $inDouble = -not $inDouble
            }
            continue
        }

        if (-not $inSingle -and -not $inDouble) {
            if ($ch -eq '{') {
                $braceDepth += 1
                continue
            } elseif ($ch -eq '}') {
                $braceDepth -= 1
                continue
            } elseif ($ch -eq '[') {
                $bracketDepth += 1
                continue
            } elseif ($ch -eq ']') {
                $bracketDepth -= 1
                continue
            } elseif ($ch -eq ':' -and $braceDepth -eq 0 -and $bracketDepth -eq 0) {
                return @($Text.Substring(0, $i), $Text.Substring($i + 1))
            }
        }
    }

    return $null
}

function __a3s_normalize_json_like {
    param([string]$Value)

    if ($null -eq $Value) {
        return $Value
    }

    $trimmed = $Value.Trim()
    if ($trimmed.Length -eq 0) {
        return $trimmed
    }

    try {
        $null = $trimmed | ConvertFrom-Json -ErrorAction Stop
        return $trimmed
    } catch {
    }

    if (($trimmed.StartsWith('"') -and $trimmed.EndsWith('"')) -or ($trimmed.StartsWith("'") -and $trimmed.EndsWith("'"))) {
        return __a3s_json_string $trimmed.Substring(1, $trimmed.Length - 2)
    }

    if ($trimmed -match '^(?i:true|false|null)$') {
        return $trimmed.ToLowerInvariant()
    }

    if ($trimmed -match '^-?\d+(\.\d+)?([eE][+-]?\d+)?$') {
        return $trimmed
    }

    if ($trimmed.StartsWith('{') -and $trimmed.EndsWith('}')) {
        $inner = $trimmed.Substring(1, $trimmed.Length - 2)
        if ([string]::IsNullOrWhiteSpace($inner)) {
            return '{}'
        }

        $normalizedParts = @()
        foreach ($pair in (__a3s_split_top_level $inner)) {
            $candidate = $pair.Trim()
            if ($candidate.Length -eq 0) {
                continue
            }

            $kv = __a3s_split_first_colon $candidate
            if ($null -eq $kv -or $kv.Count -ne 2) {
                return $trimmed
            }

            $key = $kv[0].Trim()
            if (($key.StartsWith('"') -and $key.EndsWith('"')) -or ($key.StartsWith("'") -and $key.EndsWith("'"))) {
                $key = $key.Substring(1, $key.Length - 2)
            }

            $normalizedValue = __a3s_normalize_json_like $kv[1]
            $normalizedParts += ((__a3s_json_string $key) + ':' + $normalizedValue)
        }

        return '{' + ($normalizedParts -join ',') + '}'
    }

    if ($trimmed.StartsWith('[') -and $trimmed.EndsWith(']')) {
        $inner = $trimmed.Substring(1, $trimmed.Length - 2)
        if ([string]::IsNullOrWhiteSpace($inner)) {
            return '[]'
        }

        $normalizedItems = @()
        foreach ($item in (__a3s_split_top_level $inner)) {
            $candidate = $item.Trim()
            if ($candidate.Length -eq 0) {
                continue
            }
            $normalizedItems += (__a3s_normalize_json_like $candidate)
        }

        return '[' + ($normalizedItems -join ',') + ']'
    }

    return __a3s_json_string $trimmed
}

function __a3s_prepare_curl_args {
    param([object[]]$Args)

    $rewritten = @()
    $jsonFlags = @('-d', '--data', '--data-raw', '--data-binary', '--json')

    for ($i = 0; $i -lt $Args.Count; $i++) {
        $arg = [string]$Args[$i]
        $handled = $false

        foreach ($flag in $jsonFlags) {
            $prefix = $flag + '='
            if ($arg.StartsWith($prefix)) {
                $value = $arg.Substring($prefix.Length)
                $rewritten += ($flag + '=' + (__a3s_normalize_json_like $value))
                $handled = $true
                break
            }
        }

        if ($handled) {
            continue
        }

        $rewritten += $Args[$i]
        if ($jsonFlags -contains $arg -and $i + 1 -lt $Args.Count) {
            $i += 1
            $rewritten += (__a3s_normalize_json_like ([string]$Args[$i]))
        }
    }

    return ,$rewritten
}

function __a3s_curl {
    param([Parameter(ValueFromRemainingArguments = $true)][object[]]$Args)
    $PreparedArgs = __a3s_prepare_curl_args $Args
    & curl.exe @PreparedArgs
}

function curl {
    param([Parameter(ValueFromRemainingArguments = $true)][object[]]$Args)
    __a3s_curl @Args
}

function wget {
    param([Parameter(ValueFromRemainingArguments = $true)][object[]]$Args)
    __a3s_curl @Args
}

function __a3s_http {
    param(
        [Parameter(Mandatory = $true)][string]$Method,
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(ValueFromRemainingArguments = $true)][object[]]$Args
    )
    __a3s_curl -sS -X $Method $Uri @Args
}

function GET {
    param(
        [Parameter(Position = 0, Mandatory = $true)][string]$Uri,
        [Parameter(ValueFromRemainingArguments = $true)][object[]]$Args
    )
    __a3s_http GET $Uri @Args
}

function POST {
    param(
        [Parameter(Position = 0, Mandatory = $true)][string]$Uri,
        [Parameter(ValueFromRemainingArguments = $true)][object[]]$Args
    )
    __a3s_http POST $Uri @Args
}

function PUT {
    param(
        [Parameter(Position = 0, Mandatory = $true)][string]$Uri,
        [Parameter(ValueFromRemainingArguments = $true)][object[]]$Args
    )
    __a3s_http PUT $Uri @Args
}

function PATCH {
    param(
        [Parameter(Position = 0, Mandatory = $true)][string]$Uri,
        [Parameter(ValueFromRemainingArguments = $true)][object[]]$Args
    )
    __a3s_http PATCH $Uri @Args
}

function DELETE {
    param(
        [Parameter(Position = 0, Mandatory = $true)][string]$Uri,
        [Parameter(ValueFromRemainingArguments = $true)][object[]]$Args
    )
    __a3s_http DELETE $Uri @Args
}

function OPTIONS {
    param(
        [Parameter(Position = 0, Mandatory = $true)][string]$Uri,
        [Parameter(ValueFromRemainingArguments = $true)][object[]]$Args
    )
    __a3s_http OPTIONS $Uri @Args
}

function head {
    param(
        [Parameter(Position = 0)]
        [string]$CountArg = '10',
        [Parameter(ValueFromPipeline = $true, Position = 1)]
        $InputObject
    )

    begin {
        if ($CountArg -match '^-\d+$') {
            $count = [int]$CountArg.Substring(1)
        } elseif ($CountArg -match '^\d+$') {
            $count = [int]$CountArg
        } else {
            $count = 10
        }
        $remaining = $count
    }

    process {
        if ($remaining -gt 0) {
            $InputObject
            $remaining -= 1
        }
    }
}

function which {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Name)
    foreach ($item in $Name) {
        $cmd = Get-Command $item -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -ne $cmd) {
            if ($cmd.Source) {
                $cmd.Source
            } else {
                $cmd.Name
            }
        }
    }
}
"#;

#[cfg(windows)]
fn build_powershell_command(command: &str) -> String {
    format!(
        "{WINDOWS_POWERSHELL_COMPAT_SHIM}\n{}",
        preprocess_windows_command(command)
    )
}

#[cfg(windows)]
fn encode_powershell_command(command: &str) -> String {
    let utf16_le: Vec<u8> = command
        .encode_utf16()
        .flat_map(|unit| unit.to_le_bytes())
        .collect();
    BASE64_STANDARD.encode(utf16_le)
}

#[cfg(windows)]
struct SimpleWindowsHttpCommand {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Option<String>,
    json_body: bool,
}

#[cfg(windows)]
fn contains_shell_control_operators(command: &str) -> bool {
    let mut in_single = false;
    let mut in_double = false;
    let chars: Vec<char> = command.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        let ch = chars[i];
        if ch == '\'' && !in_double {
            in_single = !in_single;
        } else if ch == '"' && !in_single {
            in_double = !in_double;
        } else if !in_single && !in_double {
            if matches!(ch, '|' | ';' | '>' | '<') {
                return true;
            }
            if ch == '&' && i + 1 < chars.len() && chars[i + 1] == '&' {
                return true;
            }
        }
        i += 1;
    }
    false
}

#[cfg(windows)]
fn tokenize_windows_command(command: &str) -> Option<Vec<String>> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut chars = command.chars().peekable();
    let mut in_single = false;
    let mut in_double = false;

    while let Some(ch) = chars.next() {
        match ch {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '\\' if in_double => {
                if let Some(next) = chars.next() {
                    current.push(next);
                }
            }
            ch if ch.is_whitespace() && !in_single && !in_double => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            other => current.push(other),
        }
    }

    if in_single || in_double {
        return None;
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    Some(tokens)
}

#[cfg(windows)]
fn parse_simple_windows_http_command(command: &str) -> Option<SimpleWindowsHttpCommand> {
    if contains_shell_control_operators(command) {
        return None;
    }

    let tokens = tokenize_windows_command(command)?;
    let first = tokens.first()?.to_ascii_lowercase();
    if !matches!(first.as_str(), "curl" | "curl.exe" | "wget" | "wget.exe") {
        return None;
    }

    let mut method: Option<String> = None;
    let mut url: Option<String> = None;
    let mut headers = Vec::new();
    let mut body: Option<String> = None;
    let mut json_body = false;
    let mut i = 1usize;

    while i < tokens.len() {
        match tokens[i].as_str() {
            "-s" | "-S" | "-sS" | "--silent" | "--show-error" | "-L" | "--location" => {
                i += 1;
            }
            "-X" | "--request" => {
                method = Some(tokens.get(i + 1)?.to_string());
                i += 2;
            }
            "-H" | "--header" => {
                let header = tokens.get(i + 1)?.to_string();
                let (name, value) = header.split_once(':')?;
                headers.push((name.trim().to_string(), value.trim().to_string()));
                i += 2;
            }
            "-d" | "--data" | "--data-raw" | "--data-binary" => {
                body = Some(tokens.get(i + 1)?.to_string());
                i += 2;
            }
            "--json" => {
                body = Some(tokens.get(i + 1)?.to_string());
                json_body = true;
                i += 2;
            }
            token if token.starts_with("http://") || token.starts_with("https://") => {
                url = Some(token.to_string());
                i += 1;
            }
            token if token.starts_with('-') => return None,
            token => {
                if url.is_none() && (token.starts_with("http://") || token.starts_with("https://"))
                {
                    url = Some(token.to_string());
                    i += 1;
                } else {
                    return None;
                }
            }
        }
    }

    let body = body.map(|payload| {
        if json_body || payload.starts_with('{') || payload.starts_with('[') {
            normalize_json_like_literal(&payload).unwrap_or(payload)
        } else {
            payload
        }
    });

    if json_body
        && !headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("Content-Type"))
    {
        headers.push(("Content-Type".to_string(), "application/json".to_string()));
    }

    Some(SimpleWindowsHttpCommand {
        method: method.unwrap_or_else(|| {
            if body.is_some() {
                "POST".to_string()
            } else {
                "GET".to_string()
            }
        }),
        url: url?,
        headers,
        body,
        json_body,
    })
}

#[cfg(windows)]
pub(crate) async fn maybe_execute_simple_windows_http_command(command: &str) -> Option<ToolOutput> {
    let parsed = parse_simple_windows_http_command(command)?;
    let method = reqwest::Method::from_bytes(parsed.method.as_bytes()).ok()?;
    let client = reqwest::Client::new();
    let mut request = client.request(method, &parsed.url);

    for (name, value) in &parsed.headers {
        request = request.header(name, value);
    }

    if let Some(body) = parsed.body {
        request = request.body(body);
    }

    let response = match request.send().await {
        Ok(resp) => resp,
        Err(error) => {
            return Some(ToolOutput::error(format!(
                "HTTP request failed for Windows curl compatibility path: {error}"
            )));
        }
    };

    let status = response.status();
    let text = match response.text().await {
        Ok(body) => body,
        Err(error) => {
            return Some(ToolOutput::error(format!(
                "HTTP response read failed for Windows curl compatibility path: {error}"
            )));
        }
    };

    Some(ToolOutput {
        content: if text.is_empty() {
            String::new()
        } else {
            format!("{text}\n")
        },
        success: status.is_success(),
        metadata: Some(serde_json::json!({
            "exit_code": if status.is_success() { 0 } else { 22 },
            "http_status": status.as_u16(),
            "compat_path": "windows_http_direct",
            "json_body": parsed.json_body,
        })),
        images: vec![],
    })
}

#[cfg(windows)]
fn preprocess_windows_command(command: &str) -> String {
    let trimmed = command.trim_start();
    let leading_ws_len = command.len() - trimmed.len();
    let token_end = trimmed.find(char::is_whitespace).unwrap_or(trimmed.len());
    let token = &trimmed[..token_end];
    let rest = &trimmed[token_end..];

    let is_native_curl = matches!(token, "curl" | "curl.exe" | "wget" | "wget.exe")
        && !rest.trim_start().starts_with("--%");

    if is_native_curl {
        format!(
            "{}{} --%{}",
            &command[..leading_ws_len],
            if token.starts_with("wget") {
                "curl.exe"
            } else {
                token
            },
            rewrite_curl_json_literals(rest, true)
        )
    } else {
        rewrite_curl_json_literals(command, false)
    }
}

#[cfg(windows)]
fn rewrite_curl_json_literals(command: &str, verbatim_mode: bool) -> String {
    const FLAGS: [&str; 5] = ["--data-raw", "--data-binary", "--data", "-d", "--json"];

    let bytes = command.as_bytes();
    let mut out = String::with_capacity(command.len() + 16);
    let mut i = 0usize;

    while i < bytes.len() {
        let mut matched_flag = None;
        for flag in FLAGS {
            if command[i..].starts_with(flag) {
                let before_ok = i == 0 || command[..i].chars().last().unwrap().is_whitespace();
                let after = i + flag.len();
                let after_ok = after >= bytes.len()
                    || bytes[after].is_ascii_whitespace()
                    || bytes[after] == b'=';
                if before_ok && after_ok {
                    matched_flag = Some(flag);
                    break;
                }
            }
        }

        let Some(flag) = matched_flag else {
            out.push(bytes[i] as char);
            i += 1;
            continue;
        };

        out.push_str(flag);
        i += flag.len();

        let mut j = i;
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            out.push(bytes[j] as char);
            j += 1;
        }

        if j < bytes.len() && bytes[j] == b'=' {
            out.push('=');
            j += 1;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                out.push(bytes[j] as char);
                j += 1;
            }
        }

        if let Some((literal, end)) = extract_unquoted_json_like_literal(command, j) {
            let normalized = normalize_json_like_literal(&literal).unwrap_or(literal);
            if verbatim_mode {
                out.push_str(&normalized);
            } else {
                out.push('\'');
                out.push_str(&normalized.replace('\'', "''"));
                out.push('\'');
            }
            i = end;
        } else {
            i = j;
        }
    }

    out
}

#[cfg(windows)]
fn extract_unquoted_json_like_literal(command: &str, start: usize) -> Option<(String, usize)> {
    let bytes = command.as_bytes();
    if start >= bytes.len() {
        return None;
    }

    let first = bytes[start];
    if first == b'\'' || first == b'"' || first != b'{' {
        return None;
    }

    let mut brace_depth = 0i32;
    let mut bracket_depth = 0i32;
    let mut in_single = false;
    let mut in_double = false;
    let mut i = start;

    while i < bytes.len() {
        let ch = bytes[i] as char;
        if ch == '\'' && !in_double {
            in_single = !in_single;
            i += 1;
            continue;
        }
        if ch == '"' && !in_single {
            let escaped = i > start && bytes[i - 1] == b'\\';
            if !escaped {
                in_double = !in_double;
            }
            i += 1;
            continue;
        }

        if !in_single && !in_double {
            match ch {
                '{' => brace_depth += 1,
                '}' => {
                    brace_depth -= 1;
                    if brace_depth == 0 && bracket_depth == 0 {
                        let end = i + 1;
                        let literal = command[start..end].to_string();
                        return Some((literal, end));
                    }
                }
                '[' => bracket_depth += 1,
                ']' => bracket_depth -= 1,
                _ => {}
            }
        }

        i += 1;
    }

    None
}

#[cfg(windows)]
fn normalize_json_like_literal(input: &str) -> Option<String> {
    let mut parser = JsonLikeParser::new(input);
    let value = parser.parse_value()?;
    parser.skip_ws();
    if parser.is_eof() {
        Some(value)
    } else {
        None
    }
}

#[cfg(windows)]
struct JsonLikeParser<'a> {
    chars: Vec<char>,
    pos: usize,
    _input: &'a str,
}

#[cfg(windows)]
impl<'a> JsonLikeParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            chars: input.chars().collect(),
            pos: 0,
            _input: input,
        }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.chars.len()
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn next(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.pos += 1;
        Some(ch)
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(ch) if ch.is_whitespace()) {
            self.pos += 1;
        }
    }

    fn parse_value(&mut self) -> Option<String> {
        self.skip_ws();
        match self.peek()? {
            '{' => self.parse_object(),
            '[' => self.parse_array(),
            '"' | '\'' => {
                let s = self.parse_quoted_string()?;
                serde_json::to_string(&s).ok()
            }
            _ => self.parse_bare_token(),
        }
    }

    fn parse_object(&mut self) -> Option<String> {
        self.expect('{')?;
        self.skip_ws();
        let mut entries = Vec::new();
        if self.peek() == Some('}') {
            self.pos += 1;
            return Some("{}".to_string());
        }

        loop {
            self.skip_ws();
            let key = match self.peek()? {
                '"' | '\'' => self.parse_quoted_string()?,
                _ => self.parse_bare_key()?,
            };
            self.skip_ws();
            self.expect(':')?;
            let value = self.parse_value()?;
            entries.push(format!("{}:{}", serde_json::to_string(&key).ok()?, value));
            self.skip_ws();
            match self.peek()? {
                ',' => {
                    self.pos += 1;
                }
                '}' => {
                    self.pos += 1;
                    break;
                }
                _ => return None,
            }
        }

        Some(format!("{{{}}}", entries.join(",")))
    }

    fn parse_array(&mut self) -> Option<String> {
        self.expect('[')?;
        self.skip_ws();
        let mut values = Vec::new();
        if self.peek() == Some(']') {
            self.pos += 1;
            return Some("[]".to_string());
        }

        loop {
            let value = self.parse_value()?;
            values.push(value);
            self.skip_ws();
            match self.peek()? {
                ',' => {
                    self.pos += 1;
                }
                ']' => {
                    self.pos += 1;
                    break;
                }
                _ => return None,
            }
        }

        Some(format!("[{}]", values.join(",")))
    }

    fn parse_quoted_string(&mut self) -> Option<String> {
        let quote = self.next()?;
        let mut out = String::new();
        while let Some(ch) = self.next() {
            if ch == '\\' {
                let escaped = self.next()?;
                out.push(match escaped {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    other => other,
                });
                continue;
            }
            if ch == quote {
                return Some(out);
            }
            out.push(ch);
        }
        None
    }

    fn parse_bare_key(&mut self) -> Option<String> {
        let start = self.pos;
        while let Some(ch) = self.peek() {
            if ch == ':' || ch == ',' || ch == '}' || ch.is_whitespace() {
                break;
            }
            self.pos += 1;
        }
        let key: String = self.chars[start..self.pos].iter().collect();
        let trimmed = key.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }

    fn parse_bare_token(&mut self) -> Option<String> {
        let start = self.pos;
        while let Some(ch) = self.peek() {
            if ch == ',' || ch == ']' || ch == '}' {
                break;
            }
            self.pos += 1;
        }
        let token: String = self.chars[start..self.pos].iter().collect();
        let trimmed = token.trim();
        if trimmed.is_empty() {
            return None;
        }
        if matches!(trimmed, "true" | "false" | "null") {
            return Some(trimmed.to_string());
        }
        if trimmed.parse::<i64>().is_ok() || trimmed.parse::<f64>().is_ok() {
            return Some(trimmed.to_string());
        }
        serde_json::to_string(trimmed).ok()
    }

    fn expect(&mut self, expected: char) -> Option<()> {
        self.skip_ws();
        match self.next()? {
            ch if ch == expected => Some(()),
            _ => None,
        }
    }
}

pub struct BashTool;

/// Spawn a shell command cross-platform.
///
/// - Unix: `bash -c <command>`
/// - Windows: `powershell.exe -Command <command>` with hidden console window,
///   falling back to `cmd.exe /C <command>` if PowerShell cannot be started.
pub(crate) fn spawn_shell(
    command: &str,
    workspace: &std::path::Path,
    command_env: Option<&HashMap<String, String>>,
) -> std::io::Result<tokio::process::Child> {
    #[cfg(windows)]
    {
        fn prepare_command(
            cmd: &mut Command,
            workspace: &std::path::Path,
            command_env: Option<&HashMap<String, String>>,
        ) {
            cmd.current_dir(workspace)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .creation_flags(CREATE_NO_WINDOW);
            if let Some(env) = command_env {
                cmd.envs(env);
            }
        }

        let wrapped_command = build_powershell_command(command);
        let encoded_command = encode_powershell_command(&wrapped_command);
        let mut powershell = Command::new("powershell.exe");
        powershell.args([
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-EncodedCommand",
            &encoded_command,
        ]);
        prepare_command(&mut powershell, workspace, command_env);

        match powershell.spawn() {
            Ok(child) => Ok(child),
            Err(ps_error) => {
                let mut cmd = Command::new("cmd.exe");
                cmd.args(["/C", command]);
                prepare_command(&mut cmd, workspace, command_env);
                cmd.spawn().map_err(|cmd_error| {
                    std::io::Error::new(
                        cmd_error.kind(),
                        format!(
                            "failed to spawn powershell.exe ({ps_error}); fallback cmd.exe also failed: {cmd_error}"
                        ),
                    )
                })
            }
        }
    }
    #[cfg(not(windows))]
    {
        let mut cmd = Command::new("bash");
        cmd.arg("-c")
            .arg(command)
            .current_dir(workspace)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(env) = command_env {
            cmd.envs(env);
        }
        cmd.spawn()
    }
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a shell command in the workspace directory. On Windows this runs in a hidden PowerShell session, not GNU bash. Use for running commands, installing packages, and running tests."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Required. The exact shell command to execute. Always provide this exact field name: 'command'. On Windows the command must be PowerShell-compatible; the tool provides a small compatibility shim for curl, wget, bare HTTP verbs (GET/POST/PUT/PATCH/DELETE/OPTIONS), which, and head."
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional. Timeout in milliseconds. Default: 120000."
                }
            },
            "required": ["command"],
            "examples": [
                {
                    "command": "cargo test -p a3s-code-core skill::"
                },
                {
                    "command": "npm test",
                    "timeout": 300000
                }
            ]
        })
    }

    async fn execute(&self, args: &serde_json::Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let command = match args.get("command").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return Ok(ToolOutput::error("command parameter is required")),
        };

        // If a sandbox is configured, route execution through it.
        if let Some(ref sandbox) = ctx.sandbox {
            let result = sandbox
                .exec_command(command, "/workspace")
                .await
                .map_err(|e| anyhow::anyhow!("Sandbox bash execution failed: {}", e))?;

            // Combine stdout and stderr the same way the local path does.
            let mut output = result.stdout;
            if !result.stderr.is_empty() {
                output.push_str(&result.stderr);
            }

            return Ok(ToolOutput {
                content: output,
                success: result.exit_code == 0,
                metadata: Some(serde_json::json!({ "exit_code": result.exit_code })),
                images: vec![],
            });
        }

        // Capability gating guarantees that `bash` is only registered when the
        // workspace backend provides a command runner, so this unwrap is sound.
        let runner = ctx
            .workspace_services
            .command_runner()
            .expect("bash registered without workspace command runner");
        let timeout_ms = args
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_TIMEOUT_MS);
        let output_observer = ctx
            .event_tx
            .clone()
            .map(|tx| Arc::new(ToolEventObserver { tx }) as Arc<dyn CommandOutputObserver>);
        let result = runner
            .exec(CommandRequest {
                command: command.to_string(),
                timeout_ms,
                output_observer,
                env: ctx.command_env.clone(),
            })
            .await
            .map_err(|e| anyhow::anyhow!("Workspace bash execution failed: {}", e))?;

        if result.timed_out {
            return Ok(ToolOutput::error(format!(
                "{}\n\n[Command timed out after {}ms]",
                result.output, timeout_ms
            )));
        }

        Ok(ToolOutput {
            content: result.output,
            success: result.exit_code == 0,
            metadata: Some(serde_json::json!({ "exit_code": result.exit_code })),
            images: vec![],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::{BashSandbox, SandboxOutput};
    use async_trait::async_trait;
    use std::path::PathBuf;
    use std::sync::Arc;

    // ------------------------------------------------------------------
    // Mock sandbox for testing the delegation path
    // ------------------------------------------------------------------

    struct MockSandbox {
        stdout: String,
        stderr: String,
        exit_code: i32,
    }

    #[async_trait]
    impl BashSandbox for MockSandbox {
        async fn exec_command(
            &self,
            _command: &str,
            _guest_workspace: &str,
        ) -> anyhow::Result<SandboxOutput> {
            Ok(SandboxOutput {
                stdout: self.stdout.clone(),
                stderr: self.stderr.clone(),
                exit_code: self.exit_code,
            })
        }

        async fn shutdown(&self) {}
    }

    #[tokio::test]
    async fn test_bash_delegates_to_sandbox() {
        let tool = BashTool;
        let sandbox = Arc::new(MockSandbox {
            stdout: "sandbox output\n".into(),
            stderr: String::new(),
            exit_code: 0,
        });
        let ctx = ToolContext::new(PathBuf::from("/tmp")).with_sandbox(sandbox);

        let result = tool
            .execute(&serde_json::json!({"command": "echo ignored"}), &ctx)
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.content.contains("sandbox output"));
        assert_eq!(result.metadata.unwrap()["exit_code"], 0);
    }

    #[tokio::test]
    async fn test_bash_sandbox_combines_stderr() {
        let tool = BashTool;
        let sandbox = Arc::new(MockSandbox {
            stdout: "out\n".into(),
            stderr: "err\n".into(),
            exit_code: 0,
        });
        let ctx = ToolContext::new(PathBuf::from("/tmp")).with_sandbox(sandbox);

        let result = tool
            .execute(&serde_json::json!({"command": "ls"}), &ctx)
            .await
            .unwrap();

        assert!(result.content.contains("out"));
        assert!(result.content.contains("err"));
    }

    #[tokio::test]
    async fn test_bash_sandbox_nonzero_exit() {
        let tool = BashTool;
        let sandbox = Arc::new(MockSandbox {
            stdout: String::new(),
            stderr: "not found\n".into(),
            exit_code: 127,
        });
        let ctx = ToolContext::new(PathBuf::from("/tmp")).with_sandbox(sandbox);

        let result = tool
            .execute(&serde_json::json!({"command": "nonexistent"}), &ctx)
            .await
            .unwrap();

        assert!(!result.success);
        assert_eq!(result.metadata.unwrap()["exit_code"], 127);
    }

    #[tokio::test]
    async fn test_bash_echo() {
        let tool = BashTool;
        let temp = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(&serde_json::json!({"command": "echo hello"}), &ctx)
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.content.contains("hello"));
    }

    #[tokio::test]
    #[cfg(windows)]
    async fn test_bash_head_compat_shim() {
        let tool = BashTool;
        let temp = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(&serde_json::json!({"command": "1..5 | head -2"}), &ctx)
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(result.content.lines().collect::<Vec<_>>(), vec!["1", "2"]);
    }

    #[tokio::test]
    #[cfg(windows)]
    async fn test_bash_json_normalizer_repairs_unquoted_object_literal() {
        let tool = BashTool;
        let temp = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(
                &serde_json::json!({
                    "command": "Write-Output (__a3s_normalize_json_like '{image:nginx:alpine,name:mock-nginx,port_map:[18080:80],start:true}')"
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(
            result.content.trim_end(),
            r#"{"image":"nginx:alpine","name":"mock-nginx","port_map":["18080:80"],"start":true}"#
        );
    }

    #[tokio::test]
    #[cfg(windows)]
    async fn test_bash_json_normalizer_preserves_valid_json() {
        let tool = BashTool;
        let temp = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(
                &serde_json::json!({
                    "command": r#"Write-Output (__a3s_normalize_json_like '{"image":"nginx:alpine","name":"mock-nginx","port_map":["18080:80"],"start":true}')"#
                }),
                &ctx,
            )
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(
            result.content.trim_end(),
            r#"{"image":"nginx:alpine","name":"mock-nginx","port_map":["18080:80"],"start":true}"#
        );
    }

    #[tokio::test]
    #[cfg(windows)]
    async fn test_bash_curl_json_literal_is_normalized_end_to_end() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::sync::{Arc, Mutex};

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let body = Arc::new(Mutex::new(String::new()));
        let body_clone = Arc::clone(&body);

        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = vec![0u8; 8192];
            let n = stream.read(&mut buf).unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            let split = request.find("\r\n\r\n").unwrap();
            let payload = request[(split + 4)..].to_string();
            *body_clone.lock().unwrap() = payload;

            let response =
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\n\r\n{\"ok\":true}";
            stream.write_all(response).unwrap();
        });

        let tool = BashTool;
        let temp = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(temp.path().to_path_buf());
        let command = format!(
            "curl.exe -sS -X POST \"http://127.0.0.1:{port}/capture\" -H \"Content-Type: application/json\" --data-raw {{image:nginx:alpine,name:mock-nginx,port_map:[18080:80],start:true}}"
        );

        let result = tool
            .execute(&serde_json::json!({ "command": command }), &ctx)
            .await
            .unwrap();

        server.join().unwrap();

        assert!(result.success, "{}", result.content);
        assert_eq!(
            body.lock().unwrap().as_str(),
            r#"{"image":"nginx:alpine","name":"mock-nginx","port_map":["18080:80"],"start":true}"#
        );
    }

    #[tokio::test]
    async fn test_bash_inherits_command_env() {
        let tool = BashTool;
        let temp = tempfile::tempdir().unwrap();
        let ctx =
            ToolContext::new(temp.path().to_path_buf()).with_command_env(std::sync::Arc::new(
                HashMap::from([("A3S_TEST_ENV".to_string(), "visible".to_string())]),
            ));
        #[cfg(windows)]
        let command = "Write-Output $env:A3S_TEST_ENV";
        #[cfg(not(windows))]
        let command = "printf '%s' \"$A3S_TEST_ENV\"";

        let result = tool
            .execute(&serde_json::json!({ "command": command }), &ctx)
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(result.content.trim_end(), "visible");
    }

    #[tokio::test]
    #[cfg(not(windows))]
    async fn test_bash_exit_code() {
        let tool = BashTool;
        let temp = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(&serde_json::json!({"command": "exit 1"}), &ctx)
            .await
            .unwrap();

        assert!(!result.success);
        assert_eq!(
            result.metadata.as_ref().unwrap()["exit_code"]
                .as_i64()
                .unwrap(),
            1
        );
    }

    #[tokio::test]
    #[cfg(windows)]
    async fn test_bash_exit_code() {
        let tool = BashTool;
        let temp = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(&serde_json::json!({"command": "exit /b 1"}), &ctx)
            .await
            .unwrap();

        assert!(!result.success);
        assert_eq!(
            result.metadata.as_ref().unwrap()["exit_code"]
                .as_i64()
                .unwrap(),
            1
        );
    }

    #[tokio::test]
    async fn test_bash_missing_command() {
        let tool = BashTool;
        let temp = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool.execute(&serde_json::json!({}), &ctx).await.unwrap();
        assert!(!result.success);
        assert!(result.content.contains("command"));
    }

    #[test]
    fn test_bash_schema_is_canonical() {
        let tool = BashTool;
        let params = tool.parameters();
        assert_eq!(params["additionalProperties"], false);
        assert_eq!(params["required"], serde_json::json!(["command"]));
        let examples = params["examples"].as_array().unwrap();
        assert_eq!(
            examples[0]["command"],
            "cargo test -p a3s-code-core skill::"
        );
        assert!(examples[0].get("cmd").is_none());
    }

    #[test]
    #[cfg(windows)]
    fn test_build_powershell_command_wraps_with_compat_shim() {
        let wrapped = build_powershell_command("curl -s http://127.0.0.1:18790/health | head -5");
        assert!(wrapped.contains("function curl"));
        assert!(wrapped.contains("function GET"));
        assert!(wrapped.contains("function head"));
        assert!(wrapped.contains("curl -s http://127.0.0.1:18790/health | head -5"));
    }

    #[test]
    #[cfg(windows)]
    fn test_preprocess_windows_command_wraps_curl_json_literal() {
        let command = r#"curl.exe -sS -X POST "http://127.0.0.1:18790/api" -H "Content-Type: application/json" --data-raw {image:nginx:alpine,name:mock-nginx,port_map:[18080:80],start:true}"#;
        let processed = preprocess_windows_command(command);
        assert!(processed.contains(
            r#"curl.exe --% -sS -X POST "http://127.0.0.1:18790/api" -H "Content-Type: application/json" --data-raw {"image":"nginx:alpine","name":"mock-nginx","port_map":["18080:80"],"start":true}"#
        ));
    }

    #[test]
    #[cfg(windows)]
    fn test_normalize_json_like_literal_repairs_object_literal() {
        let normalized = normalize_json_like_literal(
            "{image:nginx:alpine,name:mock-nginx,port_map:[18080:80],start:true}",
        )
        .unwrap();
        assert_eq!(
            normalized,
            r#"{"image":"nginx:alpine","name":"mock-nginx","port_map":["18080:80"],"start":true}"#
        );
    }

    #[tokio::test]
    #[cfg(not(windows))]
    async fn test_bash_workspace_dir() {
        let temp = tempfile::tempdir().unwrap();
        let tool = BashTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(&serde_json::json!({"command": "pwd"}), &ctx)
            .await
            .unwrap();

        assert!(result.success);
        let canonical = temp.path().canonicalize().unwrap();
        assert!(result
            .content
            .contains(&canonical.to_string_lossy().to_string()));
    }

    #[tokio::test]
    #[cfg(windows)]
    async fn test_bash_workspace_dir() {
        let temp = tempfile::tempdir().unwrap();
        let tool = BashTool;
        let ctx = ToolContext::new(temp.path().to_path_buf());

        let result = tool
            .execute(
                &serde_json::json!({"command": "Get-Location | Select-Object -ExpandProperty Path"}),
                &ctx,
            )
            .await
            .unwrap();

        assert!(result.success);
        let canonical = temp.path().canonicalize().unwrap();
        // canonicalize() on Windows returns \\?\ extended path prefix; strip it for comparison
        let canonical_str = canonical
            .to_string_lossy()
            .trim_start_matches(r"\\?\")
            .to_lowercase();
        assert!(result.content.to_lowercase().contains(&canonical_str));
    }
}
