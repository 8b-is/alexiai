//! osarous — the local model adapter.
//!
//! OpenAI-compatible chat-completions over loopback: catalog discovery
//! (`/v1/models`), health, non-streaming completion, and SSE streaming with
//! per-request connection facts. The shape mirrors the harness `llm-osarous`
//! package. Nothing here can leave the machine — the transport's sovereignty
//! check runs before every connect.



use crate::transport::{self, SseEvent};

/// Default sidecar endpoint. Loopback by construction.
pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:1337";
/// Default Ollama loopback endpoint (OpenAI-compatible `/v1/*`).
pub const OLLAMA_BASE_URL: &str = "http://127.0.0.1:11434";
/// Sovereign loopback SSH bridge to MacBook.local Ollama (`127.0.0.1:11435 -> MacBook 127.0.0.1:11434`).
pub const OLLAMA_MACBOOK_BRIDGE_URL: &str = "http://127.0.0.1:11435";
/// Sovereign loopback SSH bridge to cheet.is RTX 6000 Ada Ollama (`127.0.0.1:11436 -> cheet.is 127.0.0.1:11434`).
pub const OLLAMA_CHEET_BRIDGE_URL: &str = "http://127.0.0.1:11436";
/// Advertised context window when a model does not declare one (16,384 tokens kept relevant by MEM|8).
pub const DEFAULT_CONTEXT_WINDOW: u64 = 16384;

/// Probe loopback endpoints (`127.0.0.1:1337`, `127.0.0.1:11436`, `127.0.0.1:11435`, then `127.0.0.1:11434`) so
/// osarous/LM Studio, bridged cheet.is RTX 6000 Ada Ollama (`gemma4:26b` / `qwen3.6:35b` / `aye-deepseek:32b`),
/// bridged Apple Silicon Ollama (`aye-gemma4:26b-16k` / `aye-qwen36:35b-16k`),
/// and local Linux GPU Ollama all work out of the box over loopback.
pub fn discover_endpoint() -> String {
    use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpStream};
    use std::time::Duration;

    let timeout = Duration::from_millis(40);
    let addr_1337 = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 1337);
    if TcpStream::connect_timeout(&addr_1337, timeout).is_ok() {
        return DEFAULT_BASE_URL.to_string();
    }
    let addr_11436 = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 11436);
    if TcpStream::connect_timeout(&addr_11436, timeout).is_ok() {
        return OLLAMA_CHEET_BRIDGE_URL.to_string();
    }
    let addr_11434 = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 11434);
    if TcpStream::connect_timeout(&addr_11434, timeout).is_ok()
        || std::path::Path::new("/usr/bin/ollama").exists()
    {
        return OLLAMA_BASE_URL.to_string();
    }
    let addr_11435 = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 11435);
    if TcpStream::connect_timeout(&addr_11435, timeout).is_ok() {
        return OLLAMA_MACBOOK_BRIDGE_URL.to_string();
    }
    DEFAULT_BASE_URL.to_string()
}

#[derive(Debug, Clone)]
pub struct CatalogModel {
    pub id: String,
    pub name: Option<String>,
    pub context_window: u64,
}

#[derive(Debug, Clone)]
pub struct Osarous {
    pub base_url: String,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Osarous {
    /// Construct, refusing any non-loopback base URL.
    pub fn new(base_url: &str) -> Result<Self, String> {
        crate::sovereign::assert_loopback(base_url)?;
        let base_url = base_url.trim_end_matches('/').to_string();
        Ok(Self { base_url })
    }

    pub fn health(&self) -> (bool, String) {
        let (up, detail, _) = self.health_with_latency();
        (up, detail)
    }

    pub fn health_with_latency(&self) -> (bool, String, u64) {
        let t0 = std::time::Instant::now();
        for candidate in self.mesh_endpoints() {
            if let Ok(resp) = transport::request(&format!("{candidate}/health"), "GET", None)
                && resp.status == 200
            {
                let note = if candidate == self.base_url {
                    format!("HTTP {}", resp.status)
                } else {
                    format!("HTTP {} (failover {candidate})", resp.status)
                };
                return (true, note, t0.elapsed().as_millis() as u64);
            }
            if let Ok(mresp) = transport::request(&format!("{candidate}/v1/models"), "GET", None)
                && mresp.status == 200
            {
                let note = if candidate == self.base_url {
                    format!("HTTP {}", mresp.status)
                } else {
                    format!("HTTP {} (failover {candidate})", mresp.status)
                };
                return (true, note, t0.elapsed().as_millis() as u64);
            }
        }
        (false, "all loopback mesh endpoints unreachable".to_string(), t0.elapsed().as_millis() as u64)
    }

    /// Ordered loopback mesh endpoints: primary `base_url`, `cheet.is` A6000 bridge (`:11436`),
    /// MacBook bridge (`:11435`), and local Ollama (`:11434`).
    pub fn mesh_endpoints(&self) -> Vec<String> {
        let mut out = vec![self.base_url.clone()];
        for ep in [OLLAMA_CHEET_BRIDGE_URL, OLLAMA_MACBOOK_BRIDGE_URL, OLLAMA_BASE_URL] {
            if !out.iter().any(|x| x == ep) {
                out.push(ep.to_string());
            }
        }
        out
    }

    /// True if `model` is a heavy 24B-35B model that prefers 48GB VRAM on `cheet.is` (`:11436`)
    /// when connected, while compact/PQM models fit directly inside the laptop's 8GB RTX 4070 (`:11434`).
    fn is_heavy_vram_model(model: &str) -> bool {
        matches!(
            model,
            "qwen3.6:35b"
                | "aye-qwen36:35b-16k"
                | "aye-deepseek:32b"
                | "cheet-deepseek:32b"
                | "deepseek-r1:32b"
                | "granite4.2:30b"
                | "qwen3.8:27b"
                | "gemma4:26b"
                | "aye-gemma4:26b-16k"
                | "devstral:24b"
        )
    }

    /// Pick the best fast model installed on local `OLLAMA_BASE_URL` (`127.0.0.1:11434`) when
    /// a remote bridge (`:11436` / `:11435`) is severed.
    pub fn resolve_local_fallback_model(&self, requested: &str) -> String {
        if let Ok(local) = Self::fetch_models_from(OLLAMA_BASE_URL) {
            if !Self::is_heavy_vram_model(requested)
                && local.iter().any(|m| m.id == requested)
            {
                return requested.to_string();
            }
            for fast_local in [
                "aye-deepseek-pqm:latest",
                "st-deepseek-pqm:latest",
                "gemma4:latest",
                "cheet-deepseek-v2:16b",
                "deepseek-v2:16b",
                "gemma4:26b",
                "aye-deepseek:32b",
            ] {
                if let Some(found) = local.iter().find(|m| m.id == fast_local) {
                    return found.id.clone();
                }
            }
            if let Some(first) = local.first() {
                return first.id.clone();
            }
        }
        requested.to_string()
    }

    pub fn resolve_model(&self, preferred: Option<&str>) -> String {
        if let Some(m) = preferred
            && !m.is_empty()
            && m != "local-model"
            && m != "alexiai"
            && m != "alexiai-gaia-b158"
        {
            return m.to_string();
        }
        let cheet_up = Self::fetch_models_from(OLLAMA_CHEET_BRIDGE_URL).is_ok();
        let catalog = self.models().unwrap_or_default();
        // When cheet.is RTX 6000 Ada (48GB VRAM) is reachable, prefer flagship 26B-35B models;
        // when severed/offline, prefer local RTX 4070 PQM/compact models for instant response.
        let priority: &[&str] = if cheet_up {
            &[
                "aye-gemma4:26b-16k",
                "aye-qwen36:35b-16k",
                "gemma4:26b",
                "qwen3.6:35b",
                "aye-deepseek:32b",
                "cheet-deepseek:32b",
                "aye-deepseek-pqm:latest",
                "st-deepseek-pqm:latest",
                "gemma4:latest",
                "gemma:latest",
            ]
        } else {
            &[
                "aye-deepseek-pqm:latest",
                "st-deepseek-pqm:latest",
                "gemma4:latest",
                "cheet-deepseek-v2:16b",
                "deepseek-v2:16b",
                "gemma4:26b",
                "aye-deepseek:32b",
                "gemma:latest",
            ]
        };
        for fav in priority {
            if let Some(found) = catalog.iter().find(|m| m.id == *fav) {
                return found.id.clone();
            }
        }
        catalog
            .into_iter()
            .next()
            .map(|m| m.id)
            .unwrap_or_else(|| "local-model".to_string())
    }

    fn fetch_models_from(endpoint: &str) -> Result<Vec<CatalogModel>, String> {
        crate::sovereign::assert_loopback(endpoint)?;
        let resp = transport::request(&format!("{endpoint}/v1/models"), "GET", None)?;
        if resp.status != 200 {
            return Err(format!("HTTP {}", resp.status));
        }
        let json: serde_json_lite::Value = serde_json_lite::parse_json(&resp.body_utf8())?;
        let rows = json.get("data").and_then(|d| d.as_array()).cloned().unwrap_or_default();
        Ok(rows
            .iter()
            .map(|row| CatalogModel {
                id: row.get("id").and_then(|v| v.as_str()).unwrap_or("local-model").to_string(),
                name: row.get("name").and_then(|v| v.as_str()).map(String::from),
                context_window: row
                    .get("context_window")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(DEFAULT_CONTEXT_WINDOW),
            })
            .collect())
    }

    /// Resolve which loopback endpoint (`127.0.0.1:11436` cheet.is bridge, `127.0.0.1:11435` MacBook bridge,
    /// or `127.0.0.1:11434` local Ollama) should execute `model`.
    /// Heavy 24B-35B models prefer `:11436` (48GB VRAM); compact/PQM models prefer `:11434` (local low-latency)
    /// when `base_url` is local, and all models fall back to `:11434` if remote bridges are severed.
    pub fn endpoint_for_model(&self, model: &str) -> String {
        if !Self::is_heavy_vram_model(model)
            && let Ok(local) = Self::fetch_models_from(OLLAMA_BASE_URL)
            && local.iter().any(|m| m.id == model)
        {
            return OLLAMA_BASE_URL.to_string();
        }
        if let Ok(cheet) = Self::fetch_models_from(OLLAMA_CHEET_BRIDGE_URL)
            && cheet.iter().any(|m| m.id == model)
        {
            return OLLAMA_CHEET_BRIDGE_URL.to_string();
        }
        if let Ok(bridged) = Self::fetch_models_from(OLLAMA_MACBOOK_BRIDGE_URL)
            && bridged.iter().any(|m| m.id == model)
        {
            return OLLAMA_MACBOOK_BRIDGE_URL.to_string();
        }
        if let Ok(local) = Self::fetch_models_from(OLLAMA_BASE_URL)
            && local.iter().any(|m| m.id == model)
        {
            return OLLAMA_BASE_URL.to_string();
        }
        if let Ok(primary) = Self::fetch_models_from(&self.base_url)
            && primary.iter().any(|m| m.id == model)
        {
            return self.base_url.clone();
        }
        OLLAMA_BASE_URL.to_string()
    }

    pub fn models(&self) -> Result<Vec<CatalogModel>, String> {
        let mut models = Vec::new();
        for ep in self.mesh_endpoints() {
            if let Ok(list) = Self::fetch_models_from(&ep) {
                for item in list {
                    if !models.iter().any(|m: &CatalogModel| m.id == item.id) {
                        models.push(item);
                    }
                }
            }
        }
        models.sort_by_key(|m| match m.id.as_str() {
            "aye-deepseek-pqm:latest" => 0,
            "st-deepseek-pqm:latest" => 1,
            "gemma4:latest" => 2,
            "aye-gemma4:26b-16k" => 3,
            "aye-qwen36:35b-16k" => 4,
            "gemma4:26b" => 5,
            "qwen3.6:35b" => 6,
            "aye-deepseek:32b" => 7,
            "cheet-deepseek:32b" => 8,
            _ => 9,
        });
        Ok(if models.is_empty() {
            vec![CatalogModel {
                id: "local-model".to_string(),
                name: Some("Local Model".to_string()),
                context_window: DEFAULT_CONTEXT_WINDOW,
            }]
        } else {
            models
        })
    }

    /// Reconcile & share optimal models across reachable loopback mesh nodes (`:11434` Local,
    /// `:11436` cheet.is RTX 6000 Ada, `:11435` MacBook). Safe when severed (skips unreachable peers).
    pub fn sync_mesh_models(&self) -> Vec<String> {
        let optimal_shared = [
            "aye-deepseek-pqm:latest",
            "st-deepseek-pqm:latest",
            "gemma4:latest",
            "gemma4:26b",
            "qwen3.6:35b",
            "aye-deepseek:32b",
        ];
        let mut report = Vec::new();
        let mut node_catalogs: Vec<(String, Vec<String>)> = Vec::new();
        for ep in [OLLAMA_BASE_URL, OLLAMA_CHEET_BRIDGE_URL, OLLAMA_MACBOOK_BRIDGE_URL] {
            match Self::fetch_models_from(ep) {
                Ok(list) => {
                    let ids: Vec<String> = list.into_iter().map(|m| m.id).collect();
                    report.push(format!("node {ep}: online ({} models)", ids.len()));
                    node_catalogs.push((ep.to_string(), ids));
                }
                Err(_) => {
                    report.push(format!("node {ep}: severed/offline (autonomous fallback active)"));
                }
            }
        }
        for model in optimal_shared {
            let holders: Vec<&str> = node_catalogs
                .iter()
                .filter(|(_, ids)| ids.iter().any(|id| id == model))
                .map(|(ep, _)| ep.as_str())
                .collect();
            if holders.is_empty() {
                continue;
            }
            for (ep, ids) in &node_catalogs {
                if !ids.iter().any(|id| id == model) {
                    let pull_body = format!(r#"{{"name":"{}","stream":false}}"#, model);
                    let status = transport::request(&format!("{ep}/api/pull"), "POST", Some(&pull_body))
                        .map(|r| format!("HTTP {}", r.status))
                        .unwrap_or_else(|e| e);
                    report.push(format!("shared {model} -> {ep} ({status})"));
                }
            }
        }
        report
    }

    /// Build the OpenAI-compatible request body.
    pub fn build_body(
        &self,
        model: &str,
        messages: &[Message],
        stream: bool,
        sampling: Option<SamplingParams>,
    ) -> String {
        let msgs: Vec<String> = messages
            .iter()
            .map(|m| format!(r#"{{"role":"{}","content":{}}}"#, m.role, json_escape(&m.content)))
            .collect();
        let mut body = format!(
            r#"{{"model":"{}","messages":[{}],"stream":{},"reasoning_effort":"none","options":{{"num_ctx":{}}}"#,
            model,
            msgs.join(","),
            stream,
            DEFAULT_CONTEXT_WINDOW
        );
        if let Some(s) = sampling {
            body.push_str(&format!(
                r#","temperature":{},"top_p":{},"top_k":{},"seed":{}"#,
                s.temperature, s.top_p, s.top_k, s.seed
            ));
        }
        body.push('}');
        body
    }

    /// Non-streaming completion. Returns `(text, usage_json, finish_reason)`.
    /// The streaming API is what the app uses; this one exists for tooling
    /// that wants a single buffered answer.
    #[allow(dead_code)]
    pub fn complete(
        &self,
        model: &str,
        messages: &[Message],
        sampling: Option<SamplingParams>,
    ) -> Result<(String, String, String), String> {
        let body = self.build_body(model, messages, false, sampling);
        let endpoint = self.endpoint_for_model(model);
        let resp = match transport::request(
            &format!("{endpoint}/v1/chat/completions"),
            "POST",
            Some(&body),
        ) {
            Ok(r) if r.status == 200 => r,
            _ if endpoint != OLLAMA_BASE_URL => {
                let fallback_model = self.resolve_local_fallback_model(model);
                let fallback_body = self.build_body(&fallback_model, messages, false, sampling);
                transport::request(
                    &format!("{OLLAMA_BASE_URL}/v1/chat/completions"),
                    "POST",
                    Some(&fallback_body),
                )?
            }
            Ok(r) => r,
            Err(e) => return Err(e),
        };
        if resp.status != 200 {
            return Err(format!("HTTP {}: {}", resp.status, resp.body_utf8().chars().take(200).collect::<String>()));
        }
        let json: serde_json_lite::Value = serde_json_lite::parse_json(&resp.body_utf8())?;
        let text = json
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let usage = json.get("usage").map(|u| u.to_json_string()).unwrap_or_default();
        let finish = json
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("finish_reason"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        Ok((text, usage, finish))
    }

    /// Streaming completion. Returns frames: `("delta", text)` while
    /// streaming and `("done", "")` at the end.
    pub fn stream(
        &self,
        model: &str,
        messages: &[Message],
        sampling: Option<SamplingParams>,
    ) -> Result<impl Iterator<Item = Result<(String, String), String>>, String> {
        let body = self.build_body(model, messages, true, sampling);
        let endpoint = self.endpoint_for_model(model);
        let url = format!("{endpoint}/v1/chat/completions");
        let sse_iter = match transport::sse(&url, "POST", Some(&body)) {
            Ok(it) => it,
            Err(_) if endpoint != OLLAMA_BASE_URL => {
                let fallback_model = self.resolve_local_fallback_model(model);
                let fallback_body = self.build_body(&fallback_model, messages, true, sampling);
                let fallback_url = format!("{OLLAMA_BASE_URL}/v1/chat/completions");
                transport::sse(&fallback_url, "POST", Some(&fallback_body))?
            }
            Err(e) => return Err(e),
        };
        Ok(sse_iter.map(|frame| -> Result<(String, String), String> {
            let frame: SseEvent = frame?;
            if frame.event == "error" {
                return Err(frame.data);
            }
            if frame.data == "[DONE]" {
                return Ok(("done".to_string(), String::new()));
            }
            match serde_json_lite::parse_json(&frame.data) {
                Ok(json) => {
                    let delta = json
                        .get("choices")
                        .and_then(|c| c.as_array())
                        .and_then(|a| a.first())
                        .and_then(|c| c.get("delta"))
                        .and_then(|d| d.get("content"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if delta.is_empty() {
                        Ok(("tick".to_string(), String::new()))
                    } else {
                        Ok(("delta".to_string(), delta))
                    }
                }
                Err(_) => Ok(("tick".to_string(), String::new())), // keepalives are not failures
            }
        }))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: u32,
    pub seed: u32,
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A ~120-line JSON parser, because the app has zero dependencies and only
/// needs to read a handful of shapes. It handles objects, arrays, strings,
/// numbers, booleans and null; enough for the OpenAI surface.
pub mod serde_json_lite {
    #[derive(Debug, Clone, PartialEq)]
    pub enum Value {
        Null,
        Bool(bool),
        Num(f64),
        Str(String),
        Arr(Vec<Value>),
        Obj(Vec<(String, Value)>),
    }

    impl Value {
        pub fn get(&self, key: &str) -> Option<&Value> {
            match self {
                Value::Obj(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
                _ => None,
            }
        }
        pub fn as_str(&self) -> Option<&str> {
            match self {
                Value::Str(s) => Some(s),
                _ => None,
            }
        }
        pub fn as_u64(&self) -> Option<u64> {
            match self {
                Value::Num(n) if n.fract() == 0.0 && *n >= 0.0 => Some(*n as u64),
                _ => None,
            }
        }
        pub fn as_array(&self) -> Option<&Vec<Value>> {
            match self {
                Value::Arr(a) => Some(a),
                _ => None,
            }
        }
        pub fn to_json_string(&self) -> String {
            match self {
                Value::Null => "null".to_string(),
                Value::Bool(b) => b.to_string(),
                Value::Num(n) => n.to_string(),
                Value::Str(s) => format!("\"{s}\""),
                Value::Arr(items) => {
                    format!("[{}]", items.iter().map(|v| v.to_json_string()).collect::<Vec<_>>().join(","))
                }
                Value::Obj(entries) => {
                    format!(
                        "{{{}}}",
                        entries
                            .iter()
                            .map(|(k, v)| format!("\"{k}\":{}", v.to_json_string()))
                            .collect::<Vec<_>>()
                            .join(",")
                    )
                }
            }
        }
    }

    pub fn parse_json(input: &str) -> Result<Value, String> {
        let bytes = input.as_bytes();
        let mut pos = 0usize;
        skip_ws(bytes, &mut pos);
        let value = parse_value(bytes, &mut pos)?;
        skip_ws(bytes, &mut pos);
        if pos != bytes.len() {
            return Err(format!("trailing bytes at {pos}"));
        }
        Ok(value)
    }

    fn skip_ws(b: &[u8], pos: &mut usize) {
        while *pos < b.len() && matches!(b[*pos], b' ' | b'\t' | b'\n' | b'\r') {
            *pos += 1;
        }
    }

    fn parse_value(b: &[u8], pos: &mut usize) -> Result<Value, String> {
        match b.get(*pos) {
            Some(b'{') => parse_obj(b, pos),
            Some(b'[') => parse_arr(b, pos),
            Some(b'"') => parse_str(b, pos),
            Some(b't') => parse_lit(b, pos, b"true", Value::Bool(true)),
            Some(b'f') => parse_lit(b, pos, b"false", Value::Bool(false)),
            Some(b'n') => parse_lit(b, pos, b"null", Value::Null),
            Some(c) if c.is_ascii_digit() || *c == b'-' || *c == b'.' => parse_num(b, pos),
            _ => Err(format!("unexpected byte at {}", *pos)),
        }
    }

    fn parse_lit(b: &[u8], pos: &mut usize, lit: &[u8], value: Value) -> Result<Value, String> {
        if b.len() >= *pos + lit.len() && &b[*pos..*pos + lit.len()] == lit {
            *pos += lit.len();
            Ok(value)
        } else {
            Err(format!("bad literal at {}", *pos))
        }
    }

    fn parse_obj(b: &[u8], pos: &mut usize) -> Result<Value, String> {
        *pos += 1; // {
        let mut entries = Vec::new();
        skip_ws(b, pos);
        if b.get(*pos) == Some(&b'}') {
            *pos += 1;
            return Ok(Value::Obj(entries));
        }
        loop {
            skip_ws(b, pos);
            let Value::Str(key) = parse_str(b, pos)? else {
                return Err("object key not a string".to_string());
            };
            skip_ws(b, pos);
            if b.get(*pos) != Some(&b':') {
                return Err(format!("expected ':' at {}", *pos));
            }
            *pos += 1;
            skip_ws(b, pos);
            let value = parse_value(b, pos)?;
            entries.push((key, value));
            skip_ws(b, pos);
            match b.get(*pos) {
                Some(b',') => {
                    *pos += 1;
                }
                Some(b'}') => {
                    *pos += 1;
                    return Ok(Value::Obj(entries));
                }
                _ => return Err(format!("expected ',' or '}}' at {}", *pos)),
            }
        }
    }

    fn parse_arr(b: &[u8], pos: &mut usize) -> Result<Value, String> {
        *pos += 1; // [
        let mut items = Vec::new();
        skip_ws(b, pos);
        if b.get(*pos) == Some(&b']') {
            *pos += 1;
            return Ok(Value::Arr(items));
        }
        loop {
            skip_ws(b, pos);
            let value = parse_value(b, pos)?;
            items.push(value);
            skip_ws(b, pos);
            match b.get(*pos) {
                Some(b',') => {
                    *pos += 1;
                }
                Some(b']') => {
                    *pos += 1;
                    return Ok(Value::Arr(items));
                }
                _ => return Err(format!("expected ',' or ']' at {}", *pos)),
            }
        }
    }

    fn parse_str(b: &[u8], pos: &mut usize) -> Result<Value, String> {
        *pos += 1; // "
        let mut out = String::new();
        loop {
            match b.get(*pos) {
                None => return Err("unterminated string".to_string()),
                Some(b'"') => {
                    *pos += 1;
                    return Ok(Value::Str(out));
                }
                Some(b'\\') => {
                    *pos += 1;
                    match b.get(*pos) {
                        Some(b'"') => out.push('"'),
                        Some(b'\\') => out.push('\\'),
                        Some(b'/') => out.push('/'),
                        Some(b'n') => out.push('\n'),
                        Some(b't') => out.push('\t'),
                        Some(b'r') => out.push('\r'),
                        Some(b'u') => {
                            if *pos + 4 >= b.len() {
                                return Err("bad \\u escape".to_string());
                            }
                            let hex = std::str::from_utf8(&b[*pos + 1..*pos + 5]).unwrap_or("");
                            let cp = u32::from_str_radix(hex, 16).map_err(|e| e.to_string())?;
                            if let Some(c) = char::from_u32(cp) {
                                out.push(c);
                            }
                            *pos += 4;
                        }
                        _ => return Err(format!("bad escape at {}", *pos)),
                    }
                    *pos += 1;
                }
                Some(_) => {
                    // copy one utf-8 char
                    let rest = std::str::from_utf8(&b[*pos..]).map_err(|e| e.to_string())?;
                    let c = rest.chars().next().unwrap();
                    out.push(c);
                    *pos += c.len_utf8();
                }
            }
        }
    }

    fn parse_num(b: &[u8], pos: &mut usize) -> Result<Value, String> {
        let start = *pos;
        while *pos < b.len() && matches!(b[*pos], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
            *pos += 1;
        }
        let text = std::str::from_utf8(&b[start..*pos]).map_err(|e| e.to_string())?;
        text.parse::<f64>().map(Value::Num).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_endpoint_cannot_be_constructed() {
        assert!(Osarous::new("https://api.openai.com/v1").is_err());
        assert!(Osarous::new("http://10.0.0.5:1337").is_err());
        assert!(Osarous::new(DEFAULT_BASE_URL).is_ok());
    }

    #[test]
    fn body_is_openai_shaped() {
        let os = Osarous::new(DEFAULT_BASE_URL).unwrap();
        let body = os.build_body(
            "local-model",
            &[Message { role: "user".into(), content: "hi \"there\"".into() }],
            true,
            Some(SamplingParams { temperature: 0.4, top_p: 0.9, top_k: 40, seed: 7 }),
        );
        assert!(body.contains(r#""model":"local-model""#));
        assert!(body.contains(r#""stream":true"#));
        assert!(body.contains(r#""temperature":0.4"#));
        assert!(body.contains(r#""top_k":40"#));
        assert!(body.contains(r#"hi \"there\""#));
    }

    #[test]
    fn json_lite_parses_the_openai_shapes() {
        use serde_json_lite::parse_json;
        let v = parse_json(r#"{"data":[{"id":"m1","context_window":4096}],"usage":{"total_tokens":7}}"#).unwrap();
        assert_eq!(v.get("data").unwrap().as_array().unwrap().len(), 1);
        assert_eq!(
            v.get("data").unwrap().as_array().unwrap()[0].get("id").unwrap().as_str(),
            Some("m1")
        );
        assert_eq!(
            v.get("usage").unwrap().get("total_tokens").unwrap().as_u64(),
            Some(7)
        );
    }

    #[test]
    fn json_lite_handles_escapes_and_nested() {
        use serde_json_lite::parse_json;
        let v = parse_json(r#"{"a":"x\n\"y\"","b":[1,2.5,-3e2,true,null]}"#).unwrap();
        assert_eq!(v.get("a").unwrap().as_str(), Some("x\n\"y\""));
        assert_eq!(v.get("b").unwrap().as_array().unwrap().len(), 5);
    }
}
