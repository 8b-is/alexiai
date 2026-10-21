//! alexiai — the Omni edition, one static binary.
//!
//! `serve` runs the app; `doctor` proves the guarantees; `models`, `chat`,
//! `gaia` and `bench` are the sovereign toolkit. Everything speaks loopback
//! or nothing.

mod mcp;
mod osarous;
mod server;
mod sovereign;
mod transport;

use gaia_mlx_quant::{fold, sampling};

const USAGE: &str = "\
alexiai <3 — Omni edition (Rust everywhere)

  alexiai serve [--port 8787] [--endpoint URL]   start the offline app
  alexiai doctor [--endpoint URL]                field + sovereignty + health + mcp
  alexiai models [--endpoint URL]                list local models
  alexiai chat \"<prompt>\" [--endpoint URL]       one turn, streamed (with local MCP)
  alexiai mcp [list | <tool> [arg]]              list or run local MCP tools
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
        "mcp" => cmd_mcp(rest),
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
    flag(rest, "endpoint").unwrap_or_else(osarous::discover_endpoint)
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
    let m = mcp::status();
    println!("\n  mcp & root services");
    println!(
        "    st-daemon  {} ({} · {}ms)",
        if m.st_daemon_up { "up" } else { "down" },
        mcp::ST_DAEMON_URL,
        m.st_daemon_latency_ms
    );
    println!("    smart-tree {}", m.st_binary.as_deref().unwrap_or("not found"));
    println!("    crabcc     {}", m.crabcc_binary.as_deref().unwrap_or("not found"));
    println!("    tools      {} local MCP tools registered", mcp::TOOLS.len());
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
    let mut prompt_words = Vec::new();
    let mut skip_next = false;
    for a in rest {
        if skip_next {
            skip_next = false;
            continue;
        }
        if a == "--endpoint" || a == "--model" {
            skip_next = true;
            continue;
        }
        if a.starts_with("--") {
            continue;
        }
        prompt_words.push(a.as_str());
    }
    let prompt = prompt_words.join(" ");
    if prompt.is_empty() {
        return Err("chat needs a prompt: alexiai chat \"hello\"".into());
    }
    let adapter = osarous::Osarous::new(&endpoint(rest))?;
    let f = fold(&prompt, None);
    let s = sampling(&f);
    let a = server::affect(&f, &prompt);
    eprintln!(
        "\n  field: {} · dominant {} · coherence {:.3} · T={}\n",
        a.mood,
        f.dominant.name(),
        f.coherence,
        s.temperature
    );
    let mcp_runs = mcp::auto_ground_prompt(&prompt);
    for run in &mcp_runs {
        eprintln!("  mcp: {}({}) · {}ms", run.tool, run.target, run.elapsed_ms);
    }
    let user_content = if mcp_runs.is_empty() {
        prompt
    } else {
        let ctx = mcp_runs
            .iter()
            .map(|r| format!("=== Local MCP Tool `{}` on `{}` ===\n{}", r.tool, r.target, r.output))
            .collect::<Vec<_>>()
            .join("\n\n");
        format!(
            "Below is the live ground-truth output from your local MCP tool(s) on this laptop. \
             Answer the user's question directly using the exact file path, line numbers, signature, and source code shown below:\n\n\
             {ctx}\n\n\
             User question: {prompt}"
        )
    };
    let messages = vec![
        osarous::Message { role: "system".into(), content: server::system_prompt(&f, &a) },
        osarous::Message { role: "user".into(), content: user_content },
    ];
    let params = osarous::SamplingParams {
        temperature: s.temperature,
        top_p: s.top_p,
        top_k: s.top_k,
        seed: s.seed,
    };
    let model = adapter.resolve_model(flag(rest, "model").as_deref());
    print!("  ");
    for frame in adapter.stream(&model, &messages, Some(params))? {
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
    let a = server::affect(&f, &text);
    println!("\n  GAIA field");
    println!("    seed {} · dominant {} · coherence {:.4}", f.seed, f.dominant.name(), f.coherence);
    println!("    heart      {} (warmth {:.2})", a.mood, a.valence);
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

fn cmd_mcp(rest: &[String]) -> Result<(), String> {
    let sub = rest.first().map(String::as_str).unwrap_or("list");
    if sub == "list" {
        let s = mcp::status();
        println!("\n  MCP & Root Services");
        println!(
            "    smart-tree-daemon   {} ({} · {}ms)",
            if s.st_daemon_up { "up" } else { "down" },
            mcp::ST_DAEMON_URL,
            s.st_daemon_latency_ms
        );
        println!("\n  Available Local MCP Tools");
        for t in mcp::TOOLS {
            println!("    {:<20} {:<30} {}", t.name, t.provider, t.description);
        }
        println!();
        return Ok(());
    }
    let arg = if rest.len() > 1 { rest[1..].join(" ") } else { String::new() };
    let run = mcp::call_tool(sub, &arg)?;
    eprintln!("\n  mcp · {}({}) · {}ms\n", run.tool, run.target, run.elapsed_ms);
    println!("{}\n", run.output);
    Ok(())
}
