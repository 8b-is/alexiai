# Sovereignty — loopback or nothing

The guarantee, as an execution point.

## the rule

An endpoint is usable only if it resolves to loopback. There is no vendor
allowlist, no proxy escape hatch, no environment variable that widens it.

## the layers

1. **Constructor level** — `OsarousAdapter` throws `SovereigntyViolation` if
   its base URL is not loopback. A remote endpoint cannot even be constructed.
2. **Request level** — every send re-asserts the exact URL.
3. **Process level** — `installEgressGuard()` wraps `globalThis.fetch` and
   rejects non-loopback hosts *before a socket opens*. This is the backstop
   that catches transitive code and careless calls.

## what the tests try, and fail, to get past it

- `localhost.evil.com` — a name that merely *starts with* a loopback name
- `http://localhost@evil.com/v1` — userinfo smuggling
- `http://169.254.169.254/` — the link-local metadata endpoint
- `https://api.openai.com/v1/chat/completions` — the obvious one
- `file://` and `ftp://` — non-HTTP protocols

All rejected. The guard is idempotent and restorable; the attestation reports
honestly whether it is installed.

## the escape hatch

There is none. If a future feature genuinely needs remote egress, it is a new
module with its own review, not a widened guard. This file is part of that
review trail.

*the constellation · 0 + 1 · fine touch from within · vaked.dev*
