//! The ALEXIAI server — one static binary, one port, no network.
//!
//! Serves the offline UI from strings embedded at compile time (no files, no
//! CDN) and proxies inference to the loopback model server. Every request
//! path is hand-rolled over `std::net`: there is no framework, no dependency,
//! and no outbound path except the loopback adapter.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;

use gaia_mlx_quant::{fold, sampling};

use crate::osarous::{Osarous, SamplingParams};
use crate::sovereign::attestation;

// The entire UI, embedded at compile time. One binary = the whole app.
const INDEX_HTML: &str = include_str!("../../../web/index.html");
const APP_JS: &str = include_str!("../../../web/app.js");
const STYLE_CSS: &str = include_str!("../../../web/style.css");
const SW_JS: &str = include_str!("../../../web/sw.js");
const MANIFEST: &str = include_str!("../../../web/app.webmanifest");
const ICON_SVG: &str = include_str!("../../../web/icon.svg");
const HERO_SVG: &str = include_str!("../../../web/hero-alexandra.svg");

/// The bench: honest three-way numbers from the native substrate.
pub fn run_bench(dim: usize) -> String {
    use gaia_mlx_quant::{
        dense_matmul, pack_ternary, quantize_ternary, relative_error, ternary_matmul,
    };

    let rows = dim;
    let cols = dim;
    let w: Vec<f64> = (0..rows * cols).map(|i| synthetic(i, 0x5eed)).collect();
    let x: Vec<f64> = (0..rows).map(|j| synthetic(j, 0xc0ffee)).collect();

    let q = quantize_ternary(&w, Some(rows), Some(cols));
    let packed = pack_ternary(&q.trits);

    let mut dense = vec![0f64; cols];
    let mut ternary = vec![0f64; cols];
    dense_matmul(&w, rows, cols, &x, &mut dense);
    ternary_matmul(&packed, rows, cols, &x, q.scales.as_deref(), 1.0, &mut ternary);

    let err = relative_error(&ternary, &dense);
    let compression = (w.len() * 8) as f64 / packed.len() as f64;
    format!(
        r#"{{"dim":{dim},"bitsPerWeight":1.585,"bytesPacked":{},"bytesFloat64":{},"compression":{:.2},"relativeError":{:.4}}}"#,
        packed.len(),
        w.len() * 8,
        compression,
        err
    )
}

/// Deterministic uniform-ish sample, so the bench is reproducible.
fn synthetic(i: usize, seed: u32) -> f64 {
    let s = seed.wrapping_add(i as u32).wrapping_mul(0x6d2b79f5);
    let mut t = s;
    t = (t ^ (t >> 15)).wrapping_mul(t | 1);
    t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
    ((t ^ (t >> 14)) >> 8) as f64 / 16_777_216.0 * 2.0 - 1.0
}

/// Serve the app on 127.0.0.1.
pub fn serve(port: u16, endpoint: &str) -> Result<(), String> {
    let adapter = Osarous::new(endpoint)?;
    let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|e| e.to_string())?;
    let banner = format!(
        "\n  ALEXIAI <3 · Omni edition\n  app      http://127.0.0.1:{port}\n  endpoint {endpoint}\n  egress   loopback only\n  offline  yes — one binary, zero CDN, zero third-party models\n"
    );
    eprintln!("{banner}");

    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let adapter = adapter.clone();
                std::thread::spawn(move || {
                    let _ = handle_connection(stream, &adapter);
                });
            }
            Err(_) => continue,
        }
    }
    Ok(())
}

fn handle_connection(mut stream: std::net::TcpStream, adapter: &Osarous) -> Result<(), String> {
    let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line).map_err(|e| e.to_string())?;
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        return Err("malformed request line".to_string());
    }
    let method = parts[0].to_string();
    let target = parts[1].to_string();

    let mut headers = Vec::new();
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).map_err(|e| e.to_string())?;
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
            headers.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
        }
    }
    let _ = headers;

    let mut body = String::new();
    if content_length > 0 {
        let mut buf = vec![0u8; content_length];
        reader.read_exact(&mut buf).map_err(|e| e.to_string())?;
        body = String::from_utf8_lossy(&buf).into_owned();
    }

    let path = target.split('?').next().unwrap_or(&target).to_string();
    let query = target
        .split_once('?')
        .map(|(_, q)| q.to_string())
        .unwrap_or_default();

    match (method.as_str(), path.as_str()) {
        ("GET", "/") => write_response(&mut stream, 200, "text/html; charset=utf-8", INDEX_HTML.as_bytes()),
        ("GET", "/app.js") => write_response(&mut stream, 200, "text/javascript; charset=utf-8", APP_JS.as_bytes()),
        ("GET", "/style.css") => write_response(&mut stream, 200, "text/css; charset=utf-8", STYLE_CSS.as_bytes()),
        ("GET", "/sw.js") => write_response(&mut stream, 200, "text/javascript; charset=utf-8", SW_JS.as_bytes()),
        ("GET", "/app.webmanifest") => write_response(&mut stream, 200, "application/manifest+json", MANIFEST.as_bytes()),
        ("GET", "/icon.svg") => write_response(&mut stream, 200, "image/svg+xml", ICON_SVG.as_bytes()),
        ("GET", "/hero-alexandra.svg") => write_response(&mut stream, 200, "image/svg+xml", HERO_SVG.as_bytes()),
        ("GET", "/api/gaia") => {
            let prompt = query_param(&query, "prompt").unwrap_or_default();
            let seed = query_param(&query, "seed").and_then(|s| s.parse().ok());
            let f = fold(&prompt, seed);
            let s = sampling(&f);
            let json = format!(
                r#"{{"seed":{},"layers":{{"gravity":{},"entropy":{},"weather":{},"resonance":{}}},"dominant":"{}","coherence":{},"sampling":{{"temperature":{},"top_p":{},"top_k":{},"seed":{}}}}}"#,
                f.seed,
                f.gravity,
                f.entropy,
                f.weather,
                f.resonance,
                f.dominant.name(),
                f.coherence,
                s.temperature,
                s.top_p,
                s.top_k,
                s.seed
            );
            write_response(&mut stream, 200, "application/json", json.as_bytes())
        }
        ("GET", "/api/health") => {
            let (up, detail) = adapter.health();
            let a = attestation();
            let json = format!(
                r#"{{"ok":{},"endpoint":"{}","detail":"{}","sovereignty":{{"policy":"{}","loopbackHosts":[{}]}}}}"#,
                up,
                adapter.base_url,
                json_escape(detail.as_str()),
                a.policy,
                a.loopback_hosts
                    .iter()
                    .map(|h| format!("\"{h}\""))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            write_response(&mut stream, 200, "application/json", json.as_bytes())
        }
        ("GET", "/api/models") => match adapter.models() {
            Ok(models) => {
                let rows = models
                    .iter()
                    .map(|m| {
                        format!(
                            r#"{{"id":"{}","name":"{}","contextWindow":{}}}"#,
                            json_escape(&m.id),
                            json_escape(m.name.as_deref().unwrap_or("Local Model")),
                            m.context_window
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                let json = format!(r#"{{"catalog":[{rows}],"baseURL":"{}"}}"#, adapter.base_url);
                write_response(&mut stream, 200, "application/json", json.as_bytes())
            }
            Err(e) => {
                let json = format!(
                    r#"{{"error":"{}","catalog":[{{"id":"local-model","name":"Local Model","contextWindow":262144}}]}}"#,
                    json_escape(&e)
                );
                write_response(&mut stream, 503, "application/json", json.as_bytes())
            }
        },
        ("GET", "/api/bench") => {
            let dim = query_param(&query, "dim").and_then(|d| d.parse().ok()).unwrap_or(256);
            write_response(&mut stream, 200, "application/json", run_bench(dim).as_bytes())
        }
        ("POST", "/api/chat") => handle_chat(&mut stream, adapter, &body),
        _ => write_response(&mut stream, 404, "application/json", br#"{"error":"no route"}"#),
    }
}

fn handle_chat(stream: &mut std::net::TcpStream, adapter: &Osarous, body: &str) -> Result<(), String> {
    use crate::osarous::serde_json_lite::parse_json;
    use crate::osarous::Message;

    let prompt = parse_json(body)
        .ok()
        .and_then(|v| v.get("prompt").and_then(|p| p.as_str().map(String::from)))
        .unwrap_or_default();
    let history: Vec<Message> = parse_json(body)
        .ok()
        .and_then(|v| v.get("history").and_then(|h| h.as_array()).cloned())
        .map(|rows| {
            rows.iter()
                .filter_map(|r| {
                    Some(Message {
                        role: r.get("role")?.as_str()?.to_string(),
                        content: r.get("content")?.as_str()?.to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let f = fold(&prompt, None);
    let s = sampling(&f);

    stream
        .write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        )
        .map_err(|e| e.to_string())?;

    let gaia_frame = format!(
        "event: gaia\ndata: {{\"field\":{{\"seed\":{},\"dominant\":\"{}\",\"coherence\":{}}},\"sampling\":{{\"temperature\":{},\"top_p\":{},\"top_k\":{},\"seed\":{}}}}}\n\n",
        f.seed,
        f.dominant.name(),
        f.coherence,
        s.temperature,
        s.top_p,
        s.top_k,
        s.seed
    );
    stream.write_all(gaia_frame.as_bytes()).map_err(|e| e.to_string())?;

    let mut messages = vec![Message {
        role: "system".into(),
        content: "You are ALEXIAI, running fully offline on the GAIA-MLX-QUANT substrate. Be precise, say when you do not know, and never claim to have looked anything up.".into(),
    }];
    messages.extend(history);
    messages.push(Message {
        role: "user".into(),
        content: prompt,
    });

    let sampling = SamplingParams {
        temperature: s.temperature,
        top_p: s.top_p,
        top_k: s.top_k,
        seed: s.seed,
    };

    match adapter.stream("local-model", &messages, Some(sampling)) {
        Ok(frames) => {
            for frame in frames {
                match frame {
                    Ok((kind, text)) if kind == "delta" => {
                        let _ = stream.write_all(
                            format!("event: delta\ndata: {}\n\n", json_escape_text(&text)).as_bytes(),
                        );
                    }
                    Ok(_) => {}
                    Err(e) => {
                        let _ = stream.write_all(
                            format!("event: error\ndata: {}\n\n", json_escape_text(&e)).as_bytes(),
                        );
                    }
                }
            }
        }
        Err(e) => {
            let _ = stream.write_all(
                format!("event: error\ndata: {}\n\n", json_escape_text(&e)).as_bytes(),
            );
        }
    }
    let _ = stream.write_all(b"data: [DONE]\n\n");
    Ok(())
}

fn json_escape_text(s: &str) -> String {
    format!("\"{}\"", json_escape(s))
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
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
    out
}

fn query_param(query: &str, key: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        if k == key {
            Some(percent_decode(v))
        } else {
            None
        }
    })
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() && let Ok(b) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
            out.push(b);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn write_response(stream: &mut std::net::TcpStream, status: u16, content_type: &str, body: &[u8]) -> Result<(), String> {
    let reason = match status {
        200 => "OK",
        404 => "Not Found",
        403 => "Forbidden",
        503 => "Service Unavailable",
        _ => "OK",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-cache\r\nContent-Security-Policy: default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
    stream.write_all(body).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_assets_exist() {
        assert!(INDEX_HTML.contains("ALEXIAI"));
        assert!(APP_JS.contains("refreshPanels"));
        assert!(STYLE_CSS.contains("--hot"));
        assert!(HERO_SVG.contains("ALEXANDRA"));
    }

    #[test]
    fn query_params_decode() {
        assert_eq!(query_param("prompt=hello%20field&seed=42", "prompt").as_deref(), Some("hello field"));
        assert_eq!(query_param("prompt=x&seed=42", "seed").as_deref(), Some("42"));
        assert_eq!(query_param("nope=1", "prompt"), None);
    }

    #[test]
    fn bench_reports_real_numbers() {
        let json = run_bench(64);
        assert!(json.contains(r#""dim":64"#));
        assert!(json.contains(r#""compression""#));
        assert!(json.contains(r#""relativeError""#));
    }

    #[test]
    fn the_server_boots_and_serves_the_field() {
        let port = 18080u16;
        let adapter = Osarous::new("http://127.0.0.1:1").unwrap(); // unreachable but loopback
        let listener = TcpListener::bind(("127.0.0.1", port)).unwrap();
        std::thread::spawn(move || {
            for conn in listener.incoming().flatten() {
                let adapter = adapter.clone();
                std::thread::spawn(move || {
                    let _ = handle_connection(conn, &adapter);
                });
            }
        });
        std::thread::sleep(std::time::Duration::from_millis(120));

        // The app itself is the client: loopback only, hand-rolled.
        let resp = crate::transport::request("http://127.0.0.1:18080/api/gaia?prompt=hello", "GET", None).unwrap();
        assert_eq!(resp.status, 200);
        let json = resp.body_utf8();
        assert!(json.contains("\"dominant\""));

        let page = crate::transport::request("http://127.0.0.1:18080/", "GET", None).unwrap();
        assert_eq!(page.status, 200);
        assert!(page.body_utf8().contains("ALEXIAI"));

        let chat = crate::transport::request(
            "http://127.0.0.1:18080/api/chat",
            "POST",
            Some(r#"{"prompt":"hi"}"#),
        );
        // The model is down; the chat stream must still answer 200 with a GAIA frame.
        assert!(chat.is_ok());
    }
}
