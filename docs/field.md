# The field — GAIA, folded

GAIA ≡ planets ≡ deities. The world's constants are a *field*; a prompt is a
fold through it; the fold decides how the model samples.

## the four layers

| layer | rest | range | inhabitant | meaning |
|---|---|---|---|---|
| gravity | 1.0 | [0,1] | Ananke | cohesion — how tightly the answer must hold |
| entropy | 0.62 | [0,1] | Chaos | dispersal — how far the answer may wander |
| weather | 0.50 | [0,1] | Zephyrus | seed-rolled modulation between turns |
| resonance | φ ≈ 1.618 | constant | Aphrodite | the folding constant itself |

## the fold

For a prompt `p` and optional seed `s` (else `FNV-1a(p)`):

- `entropy += min(len(p)/4096, 0.35) - 0.15`, clamped — long prompts disperse.
- `weather += (rng(s) - 0.5) * 0.4` — the seed rolls the weather.
- dominant = the layer pushed furthest from rest.
- coherence = `1 - mean absolute deviation`, clamped.

Same `(p, s)` in ⇒ same fold out. There is no clock, no filesystem, no
network in the fold — a transcript is replayable on any machine.

## what the fold decides

```
temperature = 0.15 + entropy * 1.1        → higher dispersal, hotter sampling
top_p       = 0.50 + coherence * 0.48     → tighter when the field held
top_k       = 20 + (1 - coherence) * 60
seed        = s                           → reproducible sampling
```

The model never receives sampling parameters the user set in a config file;
it receives the field's opinion about how to answer this particular prompt.

*the constellation · 0 + 1 · fine touch from within · vaked.dev*
