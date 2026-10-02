//! alexiai — the Omni edition, one static binary.
//!
//! `serve` runs the app; `doctor` proves the guarantees; `models`, `chat`,
//! `gaia` and `bench` are the sovereign toolkit. Everything speaks loopback
//! or nothing.

mod osarous;
mod server;
mod sovereign;
mod transport;

use gaia_mlx_quant::{fold, sampling};

const USAGE: &str = "\
alexiai <3 — Omni edition (Rust everywhere)

  alexiai serve [--port 8787] [--endpoint URL]   start the offline app
  alexiai doctor [--endpoint URL]                field + sovereignty + health
  alexiai models [--endpoint URL]                list local models
  alexiai chat \"<prompt>\" [--endpoint URL]       one turn, streamed
  alexiai gaia [\"<text>\"]                        fold the field
  alexiai bench [--dim 256]                      ternary vs dense

  --endpoint must be loopback. That is not a bug, it is the product.
";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("serve");
    let rest = &args[1..];

    if matches!(command, "help" | "--help" | "-h") {
        print!("{USAGE}");
        return;
    }

    let result = match command {
        "serve" => cmd_serve(rest),
        "doctor" => cmd_doctor(rest),
        "models" => cmd_models(rest),
        "chat" => cmd_chat(rest),
        "gaia" => cmd_gaia(rest),
        "bench" => cmd_bench(rest),
        _ => {
            print!("{USAGE}");
            std::process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("\n  {e}\n");
        if e.contains("sovereignty violation") {
            eprintln!("  ALEXIAI does not talk to third-party model providers.");
            eprintln!("  Point --endpoint at a local MLX server, or run one first.\n");
        }
        std::process::exit(1);
    }
}

fn flag(rest: &[String], name: &str) -> Option<String> {
    rest.iter()
        .position(|a| a == &format!("--{name}"))
        .and_then(|i| rest.get(i + 1))
        .cloned()
}

fn endpoint(rest: &[String]) -> String {
    flag(rest, "endpoint").unwrap_or_else(|| osarous::DEFAULT_BASE_URL.to_string())
}

fn cmd_serve(rest: &[String]) -> Result<(), String> {
    let port: u16 = flag(rest, "port").and_then(|p| p.parse().ok()).unwrap_or(8787);
    let endpoint = endpoint(rest);
    server::serve(port, &endpoint)
}

fn cmd_doctor(rest: &[String]) -> Result<(), String> {
    let adapter = osarous::Osarous::new(&endpoint(rest))?;
    let f = fold("", None);
    println!("\n  GAIA field");
    println!("    seed {} · dominant {} · coherence {:.4}", f.seed, f.dominant.name(), f.coherence);
    for layer in gaia_mlx_quant::Layer::ALL {
        println!("    {:<10} {}", layer.name(), f.get(layer));
    }
    let a = sovereign::attestation();
    println!("\n  sovereignty");
    println!("    policy     {}", a.policy);
    println!("    loopback   {}", a.loopback_hosts.join(", "));
    let (up, detail) = adapter.health();
    println!("\n  local model");
    println!("    endpoint   {}", adapter.base_url);
    println!("    status     {} — {detail}", if up { "up" } else { "down" });
    match adapter.models() {
        Ok(models) => {
            for m in models {
                println!("    model      {} · {} ctx", m.id, m.context_window);
            }
        }
        Err(_) => println!("    model      (catalog unavailable while the endpoint is down)"),
    }
    let bench_json = server::run_bench(256);
    let compression = osarous::serde_json_lite::parse_json(&bench_json)
        .ok()
        .and_then(|v| v.get("compression").map(|c| c.to_json_string()))
        .unwrap_or_default();
    println!("\n  gaia-mlx-quant");
    println!("    ternary    1.5850 bits/weight · {compression}× vs f64");
    println!();
    Ok(())
}

fn cmd_models(rest: &[String]) -> Result<(), String> {
    let adapter = osarous::Osarous::new(&endpoint(rest))?;
    println!("\n  {}", adapter.base_url);
    for m in adapter.models()? {
        println!("\n  {}  {} ctx", m.id, m.context_window);
    }
    println!();
    Ok(())
}

fn cmd_chat(rest: &[String]) -> Result<(), String> {
    let prompt: String = rest
        .iter()
        .filter(|a| !a.starts_with("--"))
        .filter(|a| !rest.get(rest.iter().position(|x| x == *a).unwrap().saturating_sub(1)).map(|p| p == "--endpoint").unwrap_or(false))
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    if prompt.is_empty() {
        return Err("chat needs a prompt: alexiai chat \"hello\"".into());
    }
    let adapter = osarous::Osarous::new(&endpoint(rest))?;
    let f = fold(&prompt, None);
    let s = sampling(&f);
    eprintln!(
        "\n  field: dominant {} · coherence {:.3} · T={}\n",
        f.dominant.name(),
        f.coherence,
        s.temperature
    );
    let messages = vec![
        osarous::Message { role: "system".into(), content: "You are ALEXIAI, running fully offline. Be precise.".into() },
        osarous::Message { role: "user".into(), content: prompt },
    ];
    let params = osarous::SamplingParams {
        temperature: s.temperature,
        top_p: s.top_p,
        top_k: s.top_k,
        seed: s.seed,
    };
    print!("  ");
    for frame in adapter.stream("local-model", &messages, Some(params))? {
        match frame {
            Ok((kind, text)) if kind == "delta" => print!("{text}"),
            Ok(_) => {}
            Err(e) => eprintln!("\n  — {e}"),
        }
    }
    println!("\n");
    Ok(())
}

fn cmd_gaia(rest: &[String]) -> Result<(), String> {
    let text = rest.join(" ");
    let f = fold(&text, None);
    let s = sampling(&f);
    println!("\n  GAIA field");
    println!("    seed {} · dominant {} · coherence {:.4}", f.seed, f.dominant.name(), f.coherence);
    for layer in gaia_mlx_quant::Layer::ALL {
        println!("    {:<10} {}  ({})", layer.name(), f.get(layer), layer.inhabitant());
    }
    println!(
        "    sampling   T={} top_p={} top_k={} seed={}\n",
        s.temperature, s.top_p, s.top_k, s.seed
    );
    Ok(())
}

fn cmd_bench(rest: &[String]) -> Result<(), String> {
    let dim: usize = flag(rest, "dim").and_then(|d| d.parse().ok()).unwrap_or(256);
    println!("\n  {}\n", server::run_bench(dim));
    Ok(())
}
