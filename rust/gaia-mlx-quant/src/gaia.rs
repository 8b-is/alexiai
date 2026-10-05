//! GAIA — the field, in `core`.
//!
//! GAIA ≡ planets ≡ deities: the world's constants are a *field*, and every
//! prompt is folded through it before sampling is decided. The fold is pure:
//! same `(prompt, seed)` in, same fold out, on every machine, forever. There
//! is no clock, no filesystem, no network here — the module is `core`-only
//! and total.

/// Gravity at rest — the cohesion of the field.
pub const GRAVITY: f64 = 1.0;
/// Entropy at rest — the dispersal budget per turn.
pub const ENTROPY_REST: f64 = 0.62;
/// Weather at rest — seed-rolled modulation between turns.
pub const WEATHER_REST: f64 = 0.5;
/// The golden ratio, φ. The folding constant itself.
// Keep the shared four-lane literal stable without requiring newer core constants.
#[allow(clippy::approx_constant)]
pub const PHI: f64 = 1.618033988749894848204586834365638118;
/// Resonance — the folding constant.
pub const RESONANCE: f64 = PHI;

/// The four layers of the field. Order is load-bearing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    Gravity,
    Entropy,
    Weather,
    Resonance,
}

impl Layer {
    /// Every layer, in canonical order.
    pub const ALL: [Layer; 4] = [
        Layer::Gravity,
        Layer::Entropy,
        Layer::Weather,
        Layer::Resonance,
    ];

    /// The layer's name.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Layer::Gravity => "gravity",
            Layer::Entropy => "entropy",
            Layer::Weather => "weather",
            Layer::Resonance => "resonance",
        }
    }

    /// The deity that inhabits the layer.
    #[inline]
    pub const fn inhabitant(self) -> &'static str {
        match self {
            Layer::Gravity => "Ananke",
            Layer::Entropy => "Chaos",
            Layer::Weather => "Zephyrus",
            Layer::Resonance => "Aphrodite",
        }
    }

    /// The layer's rest value.
    #[inline]
    pub const fn rest(self) -> f64 {
        match self {
            Layer::Gravity => GRAVITY,
            Layer::Entropy => ENTROPY_REST,
            Layer::Weather => WEATHER_REST,
            Layer::Resonance => RESONANCE,
        }
    }
}

/// The result of folding a prompt through the field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fold {
    pub seed: u32,
    pub gravity: f64,
    pub entropy: f64,
    pub weather: f64,
    pub resonance: f64,
    pub dominant: Layer,
    pub coherence: f64,
}

impl Fold {
    /// The value of a layer after folding.
    #[inline]
    pub const fn get(self, layer: Layer) -> f64 {
        match layer {
            Layer::Gravity => self.gravity,
            Layer::Entropy => self.entropy,
            Layer::Weather => self.weather,
            Layer::Resonance => self.resonance,
        }
    }
}

/// FNV-1a over the prompt bytes — the derived seed space.
#[inline]
pub fn seed_from(prompt: &str) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for b in prompt.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    h
}

/// mulberry32 — a tiny deterministic PRNG. GAIA must be replayable, so the
/// same seed always yields the same weather.
#[inline]
fn mulberry32(seed: u32) -> impl FnMut() -> f64 {
    let mut a = seed;
    move || {
        a = a.wrapping_add(0x6d2b79f5);
        let mut t = a;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        ((t ^ (t >> 14)) >> 8) as f64 / 16_777_216.0
    }
}

#[inline]
fn clamp(v: f64, min: f64, max: f64) -> f64 {
    if v.is_nan() || v < min {
        min
    } else if v > max {
        max
    } else {
        v
    }
}

/// Fold a prompt through the field.
///
/// Pure and total: no allocation, no I/O, no environment access.
/// Same `(prompt, seed)` in ⇒ same [`Fold`] out.
pub fn fold(prompt: &str, seed: Option<u32>) -> Fold {
    let s = seed.unwrap_or_else(|| seed_from(prompt));
    let mut next = mulberry32(s);

    // Long prompts disperse; the seed rolls the weather.
    let entropy = clamp(
        ENTROPY_REST + (prompt.len() as f64 / 4096.0).min(0.35) - 0.15,
        0.0,
        1.0,
    );
    let weather = clamp(WEATHER_REST + (next() - 0.5) * 0.4, 0.0, 1.0);
    let gravity = GRAVITY;
    let resonance = RESONANCE;

    // Dominance: whichever layer the fold pushed furthest from rest.
    let mut dominant = Layer::Gravity;
    let mut best = f64::NEG_INFINITY;
    let layers = [gravity, entropy, weather, resonance];
    for (i, &v) in layers.iter().enumerate() {
        let rest = Layer::ALL[i].rest();
        let distance = (v - rest).abs();
        if distance > best {
            best = distance;
            dominant = Layer::ALL[i];
        }
    }

    // Coherence: how tightly the four layers agree after folding.
    let mean = (gravity + entropy + weather + resonance) / 4.0;
    let spread = ((gravity - mean).abs() + (entropy - mean).abs() + (weather - mean).abs()
        + (resonance - mean).abs())
        / 4.0;
    let coherence = clamp(1.0 - spread, 0.0, 1.0);

    Fold {
        seed: s,
        gravity,
        entropy,
        weather,
        resonance,
        dominant,
        coherence,
    }
}

/// The sampling parameters the field derives from a fold.
///
/// Higher entropy ⇒ hotter temperature. Higher coherence ⇒ tighter `top_p`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: u32,
    pub seed: u32,
}

/// Derive sampling from a fold.
pub fn sampling(f: &Fold) -> Sampling {
    Sampling {
        temperature: clamp(0.15 + f.entropy * 1.1, 0.05, 1.2),
        top_p: clamp(0.5 + f.coherence * 0.48, 0.5, 0.98),
        // round-half-up without std: the cast truncates toward zero, so the
        // +0.5 first makes it a correct round.
        top_k: (clamp(20.0 + (1.0 - f.coherence) * 60.0, 20.0, 80.0) + 0.5) as u32,
        seed: f.seed,
    }
}
