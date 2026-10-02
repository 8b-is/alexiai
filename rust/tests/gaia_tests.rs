//! The field: determinism, purity, ranges.

use gaia_mlx_quant::{
    ENTROPY_REST, GRAVITY, Layer, PHI, RESONANCE, WEATHER_REST, fold, sampling, seed_from,
};

#[test]
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
