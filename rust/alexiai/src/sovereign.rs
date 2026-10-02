//! Sovereignty — loopback or nothing, enforced before any socket opens.
//!
//! The guarantee is the same one the JS reference enforced: no model
//! inference ever leaves this machine. In the Rust app the guard sits
//! directly in front of `TcpStream::connect`: the host is validated against
//! the loopback vocabulary *before* a connection exists, so a non-local
//! endpoint is unreachable by construction, not by convention.

use std::net::IpAddr;

/// Thrown whenever a non-loopback endpoint is attempted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SovereigntyViolation {
    pub target: String,
    pub reason: String,
}

impl SovereigntyViolation {
    fn new(target: &str, reason: &str) -> Self {
        Self {
            target: target.to_string(),
            reason: reason.to_string(),
        }
    }
}

impl std::fmt::Display for SovereigntyViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sovereignty violation: {} — {}", self.target, self.reason)
    }
}

impl From<SovereigntyViolation> for String {
    fn from(v: SovereigntyViolation) -> Self {
        v.to_string()
    }
}

/// Loopback hosts by name.
const LOOPBACK_NAMES: &[&str] = &[
    "localhost",
    "localhost.localdomain",
    "ip6-localhost",
    "ip6-loopback",
];

/// Is this host (name or literal) loopback?
///
/// `0.0.0.0` is included: connecting to it on macOS and Linux resolves to
/// the local machine, so it is local by construction — the same vocabulary
/// the JS reference enforced.
pub fn is_loopback(host: &str) -> bool {
    let host = host.trim().trim_start_matches('[').trim_end_matches(']').to_ascii_lowercase();
    if LOOPBACK_NAMES.contains(&host.as_str()) || host == "0.0.0.0" {
        return true;
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        return ip.is_loopback();
    }
    false
}

/// Parse `scheme://host[:port][/path]` and assert the host is loopback.
/// No DNS is ever consulted — the check is lexical plus `IpAddr::is_loopback`.
pub fn assert_loopback(url: &str) -> Result<(String, u16), SovereigntyViolation> {
    let (host, port) = parse_host_port(url).ok_or_else(|| SovereigntyViolation::new(url, "not a parseable loopback URL"))?;
    if !is_loopback(&host) {
        return Err(SovereigntyViolation::new(url, &format!("{host} is not loopback")));
    }
    Ok((host, port))
}

/// Extract host and port from `scheme://host[:port][/anything]`.
fn parse_host_port(url: &str) -> Option<(String, u16)> {
    let rest = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    let authority = rest.split('/').next().unwrap_or(rest);
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) if !h.contains(']') && !h.is_empty() => (h.to_string(), p.parse().ok()?),
        Some((h, _)) => (h.to_string(), 80), // [::1] style: no port given
        None => (authority.to_string(), 80),
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    Some((host.to_string(), port))
}

/// The attestation, printed by `doctor` and echoed by `/api/health`.
#[derive(Clone, Debug)]
pub struct Attestation {
    pub policy: &'static str,
    pub loopback_hosts: Vec<&'static str>,
}

pub fn attestation() -> Attestation {
    Attestation {
        policy: "inference never leaves this machine",
        loopback_hosts: LOOPBACK_NAMES.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_names_and_literals_pass() {
        for host in ["localhost", "127.0.0.1", "127.1.2.3", "::1", "[::1]", "0.0.0.0"] {
            assert!(is_loopback(host), "{host} should be loopback");
        }
    }

    #[test]
    fn remote_hosts_fail() {
        for host in [
            "api.openai.com",
            "api.anthropic.com",
            "10.0.0.1",
            "192.168.1.1",
            "8.8.8.8",
            "169.254.169.254",
            "localhost.evil.com",
            "",
        ] {
            assert!(!is_loopback(host), "{host} must not be loopback");
        }
    }

    #[test]
    fn remote_urls_are_refused() {
        assert!(assert_loopback("https://api.openai.com/v1").is_err());
        assert!(assert_loopback("http://localhost.evil.com/v1").is_err());
        assert!(assert_loopback("http://127.0.0.1.evil.com:1337").is_err());
    }

    #[test]
    fn loopback_urls_parse() {
        assert_eq!(assert_loopback("http://127.0.0.1:1337").unwrap(), ("127.0.0.1".to_string(), 1337));
        assert_eq!(assert_loopback("http://localhost:8787/api").unwrap().1, 8787);
    }
}
