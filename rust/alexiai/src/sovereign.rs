//! Sovereignty — loopback or nothing, enforced before any socket opens.
//!
//! The guarantee is the same one the JS reference enforced: no model
//! inference ever leaves this machine. In the Rust app the guard sits
//! directly in front of `TcpStream::connect`: the host is validated against
//! the loopback vocabulary *before* a connection exists, so a non-local
//! endpoint is unreachable by construction, not by convention.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

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
        write!(
            f,
            "sovereignty violation: {} — {}",
            self.target, self.reason
        )
    }
}

impl From<SovereigntyViolation> for String {
    fn from(v: SovereigntyViolation) -> Self {
        v.to_string()
    }
}

/// The only hostname alias accepted; it is mapped without DNS or hosts-file lookup.
const LOOPBACK_NAMES: &[&str] = &["localhost"];

/// Resolve a loopback literal or the explicitly normalized localhost alias.
fn loopback_ip(host: &str) -> Option<IpAddr> {
    if host.eq_ignore_ascii_case("localhost") {
        return Some(IpAddr::V4(Ipv4Addr::LOCALHOST));
    }
    let host = if host.starts_with('[') {
        let inner = host.strip_prefix('[')?.strip_suffix(']')?;
        return inner
            .parse::<std::net::Ipv6Addr>()
            .ok()
            .filter(|ip| ip.is_loopback())
            .map(IpAddr::V6);
    } else {
        host
    };
    host.parse::<IpAddr>().ok().filter(|ip| ip.is_loopback())
}

/// Validate plain HTTP and return a literal socket address, never a DNS name.
pub fn assert_loopback(url: &str) -> Result<SocketAddr, SovereigntyViolation> {
    parse_loopback_url(url).ok_or_else(|| {
        SovereigntyViolation::new(
            url,
            "expected http://loopback[:port] (localhost is normalized to 127.0.0.1)",
        )
    })
}

fn parse_loopback_url(url: &str) -> Option<SocketAddr> {
    // This transport speaks HTTP, not TLS. Reject ambiguous authorities and
    // controls before any socket opens, including credentials and scoped IPv6.
    if url.chars().any(|c| c.is_whitespace() || c.is_control()) || url.contains(['\\', '#']) {
        return None;
    }
    let rest = url.strip_prefix("http://")?;
    let authority = rest.split(['/', '?']).next()?;
    let (host, port_text) = if authority.starts_with('[') {
        let end = authority.find(']')?;
        let host = &authority[..=end];
        let suffix = &authority[end + 1..];
        (
            host,
            if suffix.is_empty() {
                None
            } else {
                Some(suffix.strip_prefix(':')?)
            },
        )
    } else {
        match authority.split_once(':') {
            Some((host, port)) => (host, Some(port)),
            None => (authority, None),
        }
    };
    let port = match port_text {
        Some(text) if !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit()) => {
            text.parse::<u16>().ok()?
        }
        Some(_) => return None,
        None => 80,
    };
    if port == 0 {
        return None;
    }
    Some(SocketAddr::new(loopback_ip(host)?, port))
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
        for host in ["localhost", "127.0.0.1", "127.1.2.3", "::1", "[::1]"] {
            assert!(loopback_ip(host).is_some(), "{host} should be loopback");
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
            "localhost.localdomain",
            "ip6-localhost",
            "ip6-loopback",
            "0.0.0.0",
            "::",
            "[[::1]]",
            "[127.0.0.1]",
            "",
        ] {
            assert!(loopback_ip(host).is_none(), "{host} must not be loopback");
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
        for (url, address) in [
            ("http://127.0.0.1:1337", "127.0.0.1:1337"),
            ("http://LOCALHOST:8787/api", "127.0.0.1:8787"),
            ("http://127.1.2.3", "127.1.2.3:80"),
            ("http://[::1]:11434/api", "[::1]:11434"),
            ("http://[::1]", "[::1]:80"),
            ("http://[0:0:0:0:0:0:0:1]:65535", "[::1]:65535"),
        ] {
            assert_eq!(
                assert_loopback(url).unwrap(),
                address.parse::<SocketAddr>().unwrap()
            );
        }
    }

    #[test]
    fn ambiguous_and_non_http_urls_are_refused() {
        for url in [
            "https://127.0.0.1:443",
            "ftp://localhost",
            "localhost:8787",
            "http://0.0.0.0:80",
            "http://localhost.localdomain:80",
            "http://ip6-localhost:80",
            "http://ip6-loopback:80",
            "http://127.0.0.1@8.8.8.8",
            "http://user@127.0.0.1",
            "http://127.0.0.1:",
            "http://127.0.0.1:+80",
            "http://127.0.0.1:0",
            "http://127.0.0.1:65536",
            "http://127.0.0.1:80:90",
            "http://[::1]:invalid",
            "http://[::1]:",
            "http://[::1]suffix",
            "http://::1",
            "http://[::]",
            "http://[[::1]]",
            "http://[127.0.0.1]",
            "http://[::1%lo]:80",
            "http://[::ffff:127.0.0.1]",
            "http://127.0.0.1\\@8.8.8.8",
            "http://127.0.0.1/#fragment",
            "http://127.0.0.1/\r\nHost: remote",
            "http://127.0.0.1 /",
        ] {
            assert!(assert_loopback(url).is_err(), "accepted {url:?}");
        }
    }
}
