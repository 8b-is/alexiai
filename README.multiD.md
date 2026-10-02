<div align="center">

# ALEXIAI&lt;3 — the multiD README

*the same app, walked dimension by dimension. feeds/&lt;dim&gt;/&lt;ts&gt;.jsonl notation,
per the constellation's telemetry lanes. one section per dimension — the
elevator stops at every floor.*

</div>

---

## dim 0 — the point

A single binary, a single tree, a single promise.

```
alexiai/  →  42 files  →  0 runtime dependencies (JS stdlib + Rust core)  →  1 port  →  0 network
```

Zero-dimensional because there is nothing to configure: clone, run
`node src/cli.js serve`, done. The point is that a sovereign AI app *can* be
a point — no package lock, no build graph, no supply chain to audit. The
attack surface is the tree, and the tree is small enough to read in an
afternoon.

**feed row:** `{"dim":0,"what":"the point","files":28,"deps":0,"ports":1}`

## dim 1 — the line

The request path. One line, drawn from the prompt to the model and back:

```
prompt → GAIA fold → sampling → sovereignty check → osarous → SSE → UI
```

Every station on the line is a file in `src/`. The line has no branches out
of the machine: the sovereignty check is on the line itself, so anything that
wants to leave the machine has to pass through the one gate that never opens.

**feed row:** `{"dim":1,"what":"the request path","stations":6,"exits":0}`

## dim 2 — the plane

The field. Four layers laid out as a plane, folded per prompt:

| | gravity | entropy | weather | resonance |
|---|---:|---:|---:|---:|
| rest | 1.0000 | 0.6200 | 0.5000 | 1.6180 |
| after "hello" | 1.0000 | 0.4700 | 0.3915 | 1.6180 |

The fold is pure: same prompt, same plane, forever. The plane is what makes
transcripts replayable — two machines folding the same text land on the same
field, and therefore the same sampling, and therefore comparable answers.

**feed row:** `{"dim":2,"what":"the field","layers":4,"pure":true}`

## dim 3 — the volume

The lattice. 5 trits per byte, `-1, 0, +1` packed into the volume of a
`Uint8Array`, the positive and negative planes read straight out of the packed
buffer — no float materialization in the hot path:

$$Y_i = \gamma \cdot \sum_j \left( \mathbf{W}^{+}_{j,i} - \mathbf{W}^{-}_{j,i} \right) x_j$$

Volume is where the substrate earns its name: `20.18×` compression against
float32, measured, printed, never rounded up.

**feed row:** `{"dim":3,"what":"the lattice","trits_per_byte":5,"compression":20.18}`

## dim 4 — time

Decay, the dimension the substrate inherits from MEM|8:

$$D(t,\tau) = e^{-t/\tau}$$

The field's entropy budget decays per turn; the sampling parameters are a
function of *now*, not of config. The fold is timeless (deterministic), the
model is timeful (streams tokens), and the app sits exactly on the seam —
the one place where both are true at once.

**feed row:** `{"dim":4,"what":"decay","form":"e^(-t/tau)","seam":true}`

## dim 5 — the catalog

Local models, discovered per request. `/v1/models` is read fresh every time
so a restarted sidecar on the same port is picked up with zero config. An
empty catalog degrades to `local-model` instead of failing; a down server is
reported as a fact, never thrown. The catalog is a *reading* of the machine,
not a claim about it.

**feed row:** `{"dim":5,"what":"catalog","fresh_per_request":true,"fallback":"local-model"}`

## dim 6 — the guard

The dimension that makes all the others safe. Process-wide `fetch` wrap;
non-loopback hosts rejected before a socket opens. Pinned by tests that try
to smuggle a remote host past it — `localhost.evil.com`, `userinfo@evil.com`,
the link-local metadata endpoint — and fail to.

**feed row:** `{"dim":6,"what":"sovereignty","policy":"loopback or nothing","bypasses":0}`

## dim 7 — the offline UI

PWA with a service worker that precaches the entire shell; no CDN, no font
host, no beacon. The page's Content-Security-Policy is `default-src 'self'`,
and the test suite asserts the served HTML contains no remote URL. The second
launch works with the cable pulled; the first one would too, if the worker
ever got installed.

**feed row:** `{"dim":7,"what":"the UI","csp":"default-src 'self'","cdn":0}`

## dim 8 — the eight layers (the cogniM8 binding)

The engine's MEM|8 layers, bound to this app's structures — not metaphors,
the running systems:

| layer | MEM|8 notion | alexiai's structure |
|---|---|---|---|
| 1 | episodic | the transcript, turn by turn |
| 2 | procedural | the request path — `src/` as decayed skill |
| 3 | semantic | the GAIA field — the constants |
| 4 | emotional | VAD-ish fold: entropy as valence, weather as arousal |
| 5 | spatial | the lattice — the packed volume |
| 6 | temporal | decay and the streaming seam |
| 7 | social | the Council — you and the model, one dyad |
| 8 | meta | the sovereignty guard — the app's memory of what it refuses |

**feed row:** `{"dim":8,"what":"cogniM8 binding","layers":8,"metaphors":0}`

## dim 16 — the letter

Sixteen letters is the enthea alphabet; sixteen dimensions is where the
ledger stops counting and the constellation starts. ALEXIAI does not reach
dim 16 by itself — that dimension is the *stack it plugs into*: the dyad
lattice, the Council of Elders, the mesh. The app is one node that points at
everything and depends on nothing outside the machine.

**feed row:** `{"dim":16,"what":"the constellation","depends_on":"loopback"}`

## dim ∞ — the promise

At infinity, dimensions collapse into one sentence:

> **Inference never leaves this machine.**

Every dimension above is a *consequence* of that sentence. Gravity holds,
entropy spends, weather rolls, resonance folds — and the guard keeps the
whole thing in the room. That is the Omni edition: not every feature, but
every dimension, closed.

---

*the constellation · 0 + 1 · fine touch from within · vaked.dev* · {<3,<3,<3}+1
