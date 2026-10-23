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

/// Canonical rational denominator `q = 2^20 = 1,048,576` for pure-integer GAIA folds.
pub const GAIA_PQ_Q: i64 = 1_048_576;
/// Gravity at rest in `p/q` (`1.0 * 2^20`).
pub const GRAVITY_REST_PQ: i64 = 1_048_576;
/// Entropy at rest in `p/q` (`round(0.62 * 2^20) = 650,117`).
pub const ENTROPY_REST_PQ: i64 = 650_117;
/// Weather at rest in `p/q` (`0.5 * 2^20 = 524,288`).
pub const WEATHER_REST_PQ: i64 = 524_288;
/// Golden ratio φ in `p/q` (`round(1.61803398875 * 2^20) = 1,696,632`).
pub const PHI_PQ: i64 = 1_696_632;
/// Resonance at rest in `p/q`.
pub const RESONANCE_REST_PQ: i64 = PHI_PQ;

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

    /// The layer's pure-integer `p/q` (`q = 2^20`) rest numerator.
    #[inline]
    pub const fn rest_pq(self) -> i64 {
        match self {
            Layer::Gravity => GRAVITY_REST_PQ,
            Layer::Entropy => ENTROPY_REST_PQ,
            Layer::Weather => WEATHER_REST_PQ,
            Layer::Resonance => RESONANCE_REST_PQ,
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

/// Fold a prompt together with passive RF spectral entropy (in bits, e.g. from
/// `8b-is/wigle-triad` / `esp32.8b.is` 2.4GHz + 5GHz + BLE + AirPlay scans).
///
/// Pure, `no_std`, and zero-allocation: modulates the `Chaos` entropy budget by
/// `clamp((rf_entropy_bits - 2.5) * 0.05, -0.08, 0.14)`.
pub fn fold_spectrum(prompt: &str, rf_entropy_bits: f64, seed: Option<u32>) -> Fold {
    let s = seed.unwrap_or_else(|| seed_from(prompt));
    let mut next = mulberry32(s);

    let rf_factor = clamp((rf_entropy_bits - 2.5) * 0.05, -0.08, 0.14);
    let entropy = clamp(
        ENTROPY_REST + (prompt.len() as f64 / 4096.0).min(0.35) - 0.15 + rf_factor,
        0.0,
        1.0,
    );
    let weather = clamp(WEATHER_REST + (next() - 0.5) * 0.4, 0.0, 1.0);
    let gravity = GRAVITY;
    let resonance = RESONANCE;

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

    let spread = (entropy - ENTROPY_REST).abs() * 0.6 + (weather - WEATHER_REST).abs() * 0.4;
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

// ── Pure-Integer p/q ("Counting Divided" · Zero Floating Point) GAIA Field ──

/// Pure-integer `p/q` (`q = 2^20 = 1,048,576`) GAIA fold state.
///
/// Implements `Eq` because every layer is an exact integer numerator `p` over
/// `q = GAIA_PQ_Q` — zero IEEE-754 floating-point instructions are executed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldPq {
    pub seed: u32,
    pub q: i64,
    pub gravity_p: i64,
    pub entropy_p: i64,
    pub weather_p: i64,
    pub resonance_p: i64,
    pub dominant: Layer,
    pub coherence_p: i64,
}

impl FoldPq {
    /// Get a layer's integer numerator `p` over `self.q`.
    #[inline]
    pub const fn get_p(self, layer: Layer) -> i64 {
        match layer {
            Layer::Gravity => self.gravity_p,
            Layer::Entropy => self.entropy_p,
            Layer::Weather => self.weather_p,
            Layer::Resonance => self.resonance_p,
        }
    }

    /// Project the integer `p/q` fold to a boundary [`Fold`].
    #[inline]
    pub fn to_fold(self) -> Fold {
        let inv_q = 1.0 / (self.q as f64);
        Fold {
            seed: self.seed,
            gravity: (self.gravity_p as f64) * inv_q,
            entropy: (self.entropy_p as f64) * inv_q,
            weather: (self.weather_p as f64) * inv_q,
            resonance: (self.resonance_p as f64) * inv_q,
            dominant: self.dominant,
            coherence: (self.coherence_p as f64) * inv_q,
        }
    }
}

/// Pure-integer `p/q` (`q = 2^20`) sampling parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SamplingPq {
    pub q: i64,
    pub temperature_p: i64,
    pub top_p_p: i64,
    pub top_k: u32,
    pub seed: u32,
}

impl SamplingPq {
    /// Project the integer `p/q` sampling parameters to a boundary [`Sampling`].
    #[inline]
    pub fn to_sampling(self) -> Sampling {
        let inv_q = 1.0 / (self.q as f64);
        Sampling {
            temperature: (self.temperature_p as f64) * inv_q,
            top_p: (self.top_p_p as f64) * inv_q,
            top_k: self.top_k,
            seed: self.seed,
        }
    }
}

/// Pure-integer mulberry32 step: returns `(rand_p, next_state)` where `rand_p`
/// is a uniform numerator in `0..GAIA_PQ_Q` (`0..1_048_576`) with zero floats.
#[inline]
pub const fn mulberry32_pq(seed: u32) -> (i64, u32) {
    let a = seed.wrapping_add(0x6d2b79f5);
    let mut t = (a ^ (a >> 15)).wrapping_mul(a | 1);
    t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
    let rand_p = ((t ^ (t >> 14)) >> 12) as i64;
    (rand_p, a)
}

#[inline]
const fn clamp_i64(v: i64, min_v: i64, max_v: i64) -> i64 {
    if v < min_v {
        min_v
    } else if v > max_v {
        max_v
    } else {
        v
    }
}

/// Fold a prompt through the GAIA field using 100% pure-integer `p/q` (`q = 2^20`)
/// arithmetic. Zero floating-point operations.
pub fn fold_pq(prompt: &str, seed: Option<u32>) -> FoldPq {
    let s = seed.unwrap_or_else(|| seed_from(prompt));
    let (rand_p, _) = mulberry32_pq(s);

    // prompt.len() / 4096 in q = 2^20 is prompt.len() * 256, capped at 0.35 * 2^20 = 367,002.
    // 0.15 * 2^20 = 157,286.
    let len_p = ((prompt.len() as i64) << 8).min(367_002);
    let entropy_p = clamp_i64(ENTROPY_REST_PQ + len_p - 157_286, 0, GAIA_PQ_Q);

    // weather = 0.5 + (rand - 0.5) * 0.4 => WEATHER_REST_PQ + ((rand_p - 524_288) * 2) / 5
    let weather_p = clamp_i64(
        WEATHER_REST_PQ + ((rand_p - WEATHER_REST_PQ) * 2) / 5,
        0,
        GAIA_PQ_Q,
    );
    let gravity_p = GRAVITY_REST_PQ;
    let resonance_p = RESONANCE_REST_PQ;

    let layers_p = [gravity_p, entropy_p, weather_p, resonance_p];
    let mut dominant = Layer::Gravity;
    let mut best_dist: i64 = -1;
    let mut i = 0;
    while i < 4 {
        let dist = (layers_p[i] - Layer::ALL[i].rest_pq()).abs();
        if dist > best_dist {
            best_dist = dist;
            dominant = Layer::ALL[i];
        }
        i += 1;
    }

    let mean_p = (gravity_p + entropy_p + weather_p + resonance_p) >> 2;
    let spread_p = ((gravity_p - mean_p).abs()
        + (entropy_p - mean_p).abs()
        + (weather_p - mean_p).abs()
        + (resonance_p - mean_p).abs())
        >> 2;
    let coherence_p = clamp_i64(GAIA_PQ_Q - spread_p, 0, GAIA_PQ_Q);

    FoldPq {
        seed: s,
        q: GAIA_PQ_Q,
        gravity_p,
        entropy_p,
        weather_p,
        resonance_p,
        dominant,
        coherence_p,
    }
}

/// Fold a prompt together with passive RF spectral entropy numerator `rf_entropy_p`
/// (in `q = 2^20`, so `2.5` bits is `2_621_440`) using 100% integer arithmetic.
pub fn fold_spectrum_pq(prompt: &str, rf_entropy_p: i64, seed: Option<u32>) -> FoldPq {
    let s = seed.unwrap_or_else(|| seed_from(prompt));
    let (rand_p, _) = mulberry32_pq(s);

    // rf_factor = clamp((rf_entropy - 2.5) * 0.05, -0.08, 0.14)
    // In q = 2^20: 2.5 * 2^20 = 2,621,440; -0.08 * 2^20 = -83,886; 0.14 * 2^20 = 146,801
    let rf_factor_p = clamp_i64((rf_entropy_p - 2_621_440) / 20, -83_886, 146_801);
    let len_p = ((prompt.len() as i64) << 8).min(367_002);
    let entropy_p = clamp_i64(ENTROPY_REST_PQ + len_p - 157_286 + rf_factor_p, 0, GAIA_PQ_Q);
    let weather_p = clamp_i64(
        WEATHER_REST_PQ + ((rand_p - WEATHER_REST_PQ) * 2) / 5,
        0,
        GAIA_PQ_Q,
    );
    let gravity_p = GRAVITY_REST_PQ;
    let resonance_p = RESONANCE_REST_PQ;

    let layers_p = [gravity_p, entropy_p, weather_p, resonance_p];
    let mut dominant = Layer::Gravity;
    let mut best_dist: i64 = -1;
    let mut i = 0;
    while i < 4 {
        let dist = (layers_p[i] - Layer::ALL[i].rest_pq()).abs();
        if dist > best_dist {
            best_dist = dist;
            dominant = Layer::ALL[i];
        }
        i += 1;
    }

    // spread = |entropy - 0.62| * 0.6 + |weather - 0.5| * 0.4
    let spread_p = ((entropy_p - ENTROPY_REST_PQ).abs() * 3
        + (weather_p - WEATHER_REST_PQ).abs() * 2)
        / 5;
    let coherence_p = clamp_i64(GAIA_PQ_Q - spread_p, 0, GAIA_PQ_Q);

    FoldPq {
        seed: s,
        q: GAIA_PQ_Q,
        gravity_p,
        entropy_p,
        weather_p,
        resonance_p,
        dominant,
        coherence_p,
    }
}

/// Derive pure-integer `p/q` (`q = 2^20`) sampling parameters from a [`FoldPq`].
/// Zero floating-point operations.
#[inline]
pub const fn sampling_pq(f: &FoldPq) -> SamplingPq {
    // temperature = clamp(0.15 + entropy * 1.1, 0.05, 1.2)
    // 0.15 * 2^20 = 157,286; 0.05 * 2^20 = 52,429; 1.2 * 2^20 = 1,258,291
    let temp_p = clamp_i64(157_286 + (f.entropy_p * 11) / 10, 52_429, 1_258_291);
    // top_p = clamp(0.5 + coherence * 0.48, 0.5, 0.98)
    // 0.5 * 2^20 = 524,288; 0.98 * 2^20 = 1,027,604
    let top_p_p = clamp_i64(524_288 + (f.coherence_p * 12) / 25, 524_288, 1_027_604);
    // top_k = round(clamp(20 + (1 - coherence) * 60, 20, 80))
    let raw_k_p = 20 * GAIA_PQ_Q + (GAIA_PQ_Q - f.coherence_p) * 60;
    let clamped_k_p = clamp_i64(raw_k_p, 20 * GAIA_PQ_Q, 80 * GAIA_PQ_Q);
    let top_k = ((clamped_k_p + (GAIA_PQ_Q >> 1)) >> 20) as u32;

    SamplingPq {
        q: GAIA_PQ_Q,
        temperature_p: temp_p,
        top_p_p,
        top_k,
        seed: f.seed,
    }
}
