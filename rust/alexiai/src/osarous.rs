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
/// Advertised context window when a model does not declare one.
pub const DEFAULT_CONTEXT_WINDOW: u64 = 262144;

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
        match transport::request(&format!("{}/health", self.base_url), "GET", None) {
            Ok(resp) => (resp.status == 200, format!("HTTP {}", resp.status)),
            Err(e) => (false, e),
        }
    }

    pub fn models(&self) -> Result<Vec<CatalogModel>, String> {
        let resp = transport::request(&format!("{}/v1/models", self.base_url), "GET", None)?;
        if resp.status != 200 {
            return Err(format!("HTTP {}", resp.status));
        }
        let json: serde_json_lite::Value = serde_json_lite::parse_json(&resp.body_utf8())?;
        let rows = json.get("data").and_then(|d| d.as_array()).cloned().unwrap_or_default();
        let models: Vec<CatalogModel> = rows
            .iter()
            .map(|row| CatalogModel {
                id: row.get("id").and_then(|v| v.as_str()).unwrap_or("local-model").to_string(),
                name: row.get("name").and_then(|v| v.as_str()).map(String::from),
                context_window: row
                    .get("context_window")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(DEFAULT_CONTEXT_WINDOW),
            })
            .collect();
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
            r#"{{"model":"{}","messages":[{}],"stream":{}"#,
            model,
            msgs.join(","),
            stream
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
        let resp = transport::request(
            &format!("{}/v1/chat/completions", self.base_url),
            "POST",
            Some(&body),
        )?;
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
        let url = format!("{}/v1/chat/completions", self.base_url);
        Ok(transport::sse(&url, "POST", Some(&body))?.map(|frame| -> Result<(String, String), String> {
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
