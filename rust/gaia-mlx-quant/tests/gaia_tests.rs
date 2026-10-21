//! The field: determinism, purity, ranges.

use gaia_mlx_quant::{
    ENTROPY_REST, ENTROPY_REST_PQ, GAIA_PQ_Q, GRAVITY, GRAVITY_REST_PQ, Layer, PHI, PHI_PQ,
    RESONANCE, RESONANCE_REST_PQ, WEATHER_REST, WEATHER_REST_PQ, fold, fold_pq, fold_spectrum,
    fold_spectrum_pq, sampling, sampling_pq, seed_from,
};

#[test]
#[allow(clippy::approx_constant)]
fn rest_values_are_the_cosmology() {
    assert_eq!(GRAVITY, 1.0);
    assert_eq!(ENTROPY_REST, 0.62);
    assert_eq!(WEATHER_REST, 0.5);
    assert!((RESONANCE - PHI).abs() < f64::EPSILON);
    assert!((PHI - 1.618_033_988_749_895).abs() < 1e-15);
}

#[test]
fn fold_is_pure() {
    let a = fold("hello field", None);
    let b = fold("hello field", None);
    assert_eq!(a, b);
}

#[test]
fn explicit_seed_overrides_the_derived_one() {
    let a = fold("anything", Some(42));
    let b = fold("something else entirely", Some(42));
    assert_eq!(a.seed, 42);
    assert_eq!(b.seed, 42);
    // Weather is pure seed-rolled rng; entropy responds to length.
    assert_eq!(a.weather, b.weather);
    assert!(b.entropy > a.entropy);
}

#[test]
fn long_prompts_disperse() {
    let short = fold("hi", Some(11)).entropy;
    let long = fold(&"hi ".repeat(2000), Some(11)).entropy;
    assert!(long > short);
}

#[test]
fn layers_stay_in_range() {
    let long = fold(&"x".repeat(10000), Some(7));
    for layer in [Layer::Gravity, Layer::Entropy, Layer::Weather] {
        let v = long.get(layer);
        assert!((0.0..=1.0).contains(&v), "{:?} out of range: {v}", layer);
    }
    // Resonance is the golden ratio — above one by design.
    assert!(long.resonance > 1.6 && long.resonance < 1.62);
}

#[test]
fn coherence_is_a_probability() {
    for seed in [0u32, 1, 2, 99, 123456] {
        let f = fold("sample", Some(seed));
        assert!((0.0..=1.0).contains(&f.coherence));
    }
}

#[test]
fn sampling_stays_sane() {
    let params = sampling(&fold("question", Some(5)));
    assert!((0.05..=1.2).contains(&params.temperature));
    assert!((0.5..=0.98).contains(&params.top_p));
    assert!((20..=80).contains(&params.top_k));
    assert_eq!(params.seed, 5);
}

#[test]
fn seed_from_is_stable_and_sensitive() {
    assert_eq!(seed_from("alexiai"), seed_from("alexiai"));
    assert_ne!(seed_from("alexiai"), seed_from("alexia"));
}

#[test]
fn every_layer_has_a_name_and_an_inhabitant() {
    let names: Vec<&str> = Layer::ALL.iter().map(|l| l.name()).collect();
    assert_eq!(names, ["gravity", "entropy", "weather", "resonance"]);
    let gods: Vec<&str> = Layer::ALL.iter().map(|l| l.inhabitant()).collect();
    assert_eq!(gods, ["Ananke", "Chaos", "Zephyrus", "Aphrodite"]);
}

#[test]
fn fold_spectrum_modulates_entropy_with_rf_shannon_bits() {
    let prompt = "Analyze Shinjuku 2.4GHz Ch 6 saturation, BLE proximity & AirPlay RAOP surface";
    let low = fold_spectrum(prompt, 0.5, None);
    let high = fold_spectrum(prompt, 4.8, None);
    assert_eq!(low.seed, high.seed);
    assert_eq!(low.weather, high.weather);
    assert!(high.entropy > low.entropy);
    assert!((0.0..=1.0).contains(&low.coherence));
    assert!((0.0..=1.0).contains(&high.coherence));
}

#[test]
fn fold_pq_and_sampling_pq_operate_in_pure_integer_space() {
    assert_eq!(GAIA_PQ_Q, 1_048_576);
    assert_eq!(GRAVITY_REST_PQ, 1_048_576);
    assert_eq!(ENTROPY_REST_PQ, 650_117);
    assert_eq!(WEATHER_REST_PQ, 524_288);
    assert_eq!(RESONANCE_REST_PQ, PHI_PQ);
    assert_eq!(Layer::Gravity.rest_pq(), GRAVITY_REST_PQ);
    assert_eq!(Layer::Entropy.rest_pq(), ENTROPY_REST_PQ);
    assert_eq!(Layer::Weather.rest_pq(), WEATHER_REST_PQ);
    assert_eq!(Layer::Resonance.rest_pq(), RESONANCE_REST_PQ);

    let prompt = "Analyze Shinjuku 2.4GHz Ch 6 saturation, BLE proximity & AirPlay RAOP surface";
    let f_base = fold_pq(prompt, Some(42));
    assert_eq!(f_base, fold_pq(prompt, Some(42)));
    assert_eq!(f_base.get_p(Layer::Gravity), GRAVITY_REST_PQ);
    assert!((0..=GAIA_PQ_Q).contains(&f_base.entropy_p));
    assert!((0..=GAIA_PQ_Q).contains(&f_base.weather_p));
    assert!((0..=GAIA_PQ_Q).contains(&f_base.coherence_p));

    let low = fold_spectrum_pq(prompt, GAIA_PQ_Q / 2, Some(42));
    let high = fold_spectrum_pq(prompt, GAIA_PQ_Q * 4, Some(42));
    assert_eq!(low.seed, high.seed);
    assert_eq!(low.weather_p, high.weather_p);
    assert!(high.entropy_p > low.entropy_p);

    let s_pq = sampling_pq(&high);
    assert!((52_429..=1_258_291).contains(&s_pq.temperature_p));
    assert!((524_288..=1_027_604).contains(&s_pq.top_p_p));
    assert!((20..=80).contains(&s_pq.top_k));
    assert_eq!(s_pq.seed, 42);
}
