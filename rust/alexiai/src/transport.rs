//! A minimal, dependency-free HTTP/1.1 transport.
//!
//! The whole app talks to one process on one machine. It needs a request
//! loop, not an HTTP framework: `TcpStream` + hand-rolled parsing, with
//! `Content-Length` and chunked bodies both supported (local MLX servers
//! differ), and Server-Sent Events parsed frame by frame.
//!
//! Sovereignty is enforced here — every `connect` goes through the loopback
//! check in [`crate::sovereign`], so a remote endpoint is unreachable by
//! construction.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use crate::sovereign::assert_loopback;

/// A parsed HTTP response.
#[derive(Debug)]
pub struct Response {
    pub status: u16,
    pub body: Vec<u8>,
}

impl Response {
    pub fn body_utf8(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

/// A full HTTP request with the response body buffered.
pub fn request(url: &str, method: &str, body: Option<&str>) -> Result<Response, String> {
    request_with_headers(url, method, &[], body)
}

/// A full HTTP request with custom headers and the response body buffered.
pub fn request_with_headers(
    url: &str,
    method: &str,
    extra_headers: &[(&str, &str)],
    body: Option<&str>,
) -> Result<Response, String> {
    let (host, port) = assert_loopback(url)?; // the guard: before any socket
    let path = path_of(url);
    let mut stream = connect_raw(&host, port)?;
    write_head_with_headers(&mut stream, method, &path, extra_headers, body)?;

    // One reader for the whole exchange: a duplicated fd would lose bytes
    // buffered by a first reader.
    let mut reader = BufReader::new(stream);
    let status = read_status(&mut reader)?;
    let headers = read_headers(&mut reader)?;
    let body_bytes = read_body_from(&mut reader, &headers)?;
    Ok(Response { status, body: body_bytes })
}

fn connect_raw(host: &str, port: u16) -> Result<TcpStream, String> {
    let stream = TcpStream::connect((host, port)).map_err(|e| format!("connect {host}:{port}: {e}"))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(120)))
        .map_err(|e| e.to_string())?;
    Ok(stream)
}

fn path_of(url: &str) -> String {
    let after_scheme = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    let path = after_scheme.split('/').skip(1).collect::<Vec<_>>().join("/");
    let path = path.split('?').next().unwrap_or(&path);
    format!("/{path}")
}

fn write_head(stream: &mut TcpStream, method: &str, path: &str, body: Option<&str>) -> Result<(), String> {
    write_head_with_headers(stream, method, path, &[], body)
}

fn write_head_with_headers(
    stream: &mut TcpStream,
    method: &str,
    path: &str,
    extra_headers: &[(&str, &str)],
    body: Option<&str>,
) -> Result<(), String> {
    let mut head = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nAccept: */*\r\nConnection: close\r\n"
    );
    for (k, v) in extra_headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    if let Some(b) = body {
        head.push_str("Content-Type: application/json\r\n");
        head.push_str(&format!("Content-Length: {}\r\n", b.len()));
        head.push_str("\r\n");
        head.push_str(b);
    } else {
        head.push_str("\r\n");
    }
    stream.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())
}

fn read_status(reader: &mut impl BufRead) -> Result<u16, String> {
    let mut status_line = String::new();
    reader.read_line(&mut status_line).map_err(|e| e.to_string())?;
    status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("malformed status line: {status_line:?}"))
}

fn read_headers(reader: &mut impl BufRead) -> Result<Vec<(String, String)>, String> {
    let mut headers = Vec::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).map_err(|e| e.to_string())?;
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            headers.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
        }
    }
    Ok(headers)
}

fn is_chunked(headers: &[(String, String)]) -> bool {
    headers
        .iter()
        .find(|(k, _)| k == "transfer-encoding")
        .is_some_and(|(_, v)| v.to_ascii_lowercase().contains("chunked"))
}

fn read_body_from(reader: &mut impl BufRead, headers: &[(String, String)]) -> Result<Vec<u8>, String> {
    if is_chunked(headers) {
        return read_chunked(reader);
    }
    let len = headers
        .iter()
        .find(|(k, _)| k == "content-length")
        .and_then(|(_, v)| v.parse::<usize>().ok())
        .unwrap_or(0);
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf)
}

struct ChunkedReader<R> {
    inner: R,
    remaining: usize,
    eof: bool,
}

impl<R> ChunkedReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            remaining: 0,
            eof: false,
        }
    }
}

impl<R: BufRead> Read for ChunkedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.eof || buf.is_empty() {
            return Ok(0);
        }
        while self.remaining == 0 {
            let mut size_line = String::new();
            let n = self.inner.read_line(&mut size_line)?;
            if n == 0 {
                self.eof = true;
                return Ok(0);
            }
            let trimmed = size_line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let hex = trimmed.split(';').next().unwrap_or("0");
            let size = usize::from_str_radix(hex, 16)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            if size == 0 {
                let mut trailer = String::new();
                let _ = self.inner.read_line(&mut trailer);
                self.eof = true;
                return Ok(0);
            }
            self.remaining = size;
        }
        let max_read = buf.len().min(self.remaining);
        let n = self.inner.read(&mut buf[..max_read])?;
        if n == 0 {
            self.eof = true;
            return Ok(0);
        }
        self.remaining -= n;
        if self.remaining == 0 {
            let mut crlf = [0u8; 2];
            let _ = self.inner.read_exact(&mut crlf);
        }
        Ok(n)
    }
}

fn read_chunked(reader: &mut impl BufRead) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let mut chunked = ChunkedReader::new(reader);
    chunked.read_to_end(&mut out).map_err(|e| e.to_string())?;
    Ok(out)
}

/// An SSE frame.
#[derive(Debug, Clone)]
pub struct SseEvent {
    pub event: String,
    pub data: String,
}

/// Stream Server-Sent Events from an endpoint.
pub fn sse(url: &str, method: &str, body: Option<&str>) -> Result<SseStream, String> {
    let (host, port) = assert_loopback(url)?;
    let path = path_of(url);
    let mut stream = connect_raw(&host, port)?;
    write_head(&mut stream, method, &path, body)?;

    // Same single-reader rule as `request`: parse the head from the same
    // reader that will stream the body.
    let mut reader = BufReader::new(stream);
    let status = read_status(&mut reader)?;
    let headers = read_headers(&mut reader)?;
    if status != 200 {
        let err_body = read_body_from(&mut reader, &headers).unwrap_or_default();
        let detail = String::from_utf8_lossy(&err_body).trim().to_string();
        return Err(if detail.is_empty() {
            format!("HTTP {status}")
        } else {
            format!("HTTP {status}: {detail}")
        });
    }
    let reader: Box<dyn BufRead + Send> = if is_chunked(&headers) {
        Box::new(BufReader::new(ChunkedReader::new(reader)))
    } else {
        Box::new(reader)
    };
    Ok(SseStream { reader })
}

pub struct SseStream {
    reader: Box<dyn BufRead + Send>,
}

impl Iterator for SseStream {
    type Item = Result<SseEvent, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut event = "message".to_string();
        let mut data_lines: Vec<String> = Vec::new();
        let mut saw_any = false;

        loop {
            let mut line = String::new();
            match self.reader.read_line(&mut line) {
                Ok(0) => {
                    if saw_any {
                        return Some(Ok(SseEvent {
                            event,
                            data: data_lines.join("\n"),
                        }));
                    }
                    return None;
                }
                Ok(_) => {}
                Err(e) => return Some(Err(e.to_string())),
            }
            let line = line.trim_end().to_string();
            if line.is_empty() {
                if saw_any {
                    return Some(Ok(SseEvent {
                        event,
                        data: data_lines.join("\n"),
                    }));
                }
                continue; // leading blank line
            }
            saw_any = true;
            if line.strip_prefix(':').is_some() {
                continue; // SSE comment / keepalive
            }
            if let Some((field, value)) = line.split_once(':') {
                match field.trim() {
                    "event" => event = value.trim().to_string(),
                    "data" => data_lines.push(value.trim_start().to_string()),
                    _ => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn chunked_parse(input: &[u8]) -> Vec<u8> {
        let mut reader = BufReader::new(Cursor::new(input.to_vec()));
        read_chunked(&mut reader).unwrap()
    }

    #[test]
    fn chunked_bodies_parse() {
        let body = chunked_parse(b"4\r\nwiki\r\n5\r\npedia\r\n0\r\n\r\n");
        assert_eq!(String::from_utf8(body).unwrap(), "wikipedia");
    }

    #[test]
    fn empty_chunked_body_parses() {
        assert!(chunked_parse(b"0\r\n\r\n").is_empty());
    }

    #[test]
    fn chunked_sse_streams_events() {
        let raw = b"1b\r\ndata: {\"delta\":\"hello\"}\n\n\r\n\r\n10\r\ndata: [DONE]\n\n\r\n\r\n0\r\n\r\n";
        let reader: Box<dyn BufRead + Send> =
            Box::new(BufReader::new(ChunkedReader::new(Cursor::new(raw.to_vec()))));
        let mut stream = SseStream { reader };
        let first = stream.next().unwrap().unwrap();
        assert_eq!(first.data, "{\"delta\":\"hello\"}");
        let second = stream.next().unwrap().unwrap();
        assert_eq!(second.data, "[DONE]");
        assert!(stream.next().is_none());
    }
}
