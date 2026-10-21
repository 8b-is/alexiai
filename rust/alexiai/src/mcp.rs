//! mcp — Sovereign local MCP tool bridge for ALEXIAI.
//!
//! Connects ALEXIAI to local Model Context Protocol (MCP) tools running on
//! this machine with zero third-party network egress:
//! - `smart-tree-daemon` (`http://127.0.0.1:28428/mcp/*`, authenticated via
//!   `/var/lib/smart-tree/daemon.token` or `~/.st/daemon.token`) + `/usr/local/bin/st`
//! - `crabcc` (`/usr/local/bin/crabcc` / `~/.local/bin/crabcc`) for Tree-sitter
//!   AST outlines, symbol definitions (`sym`), and references (`refs`)
//! - `fandom-mcp` (`/home/hue/8b-is/fandom-mcp/fandom-mcp.mjs`) for Sovereign
//!   Library Three-Kaya Balanced-Ternary Ontology (`{-1, 0, +1}`)

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use crate::osarous::serde_json_lite::{Value, parse_json};
use crate::transport;

pub const ST_DAEMON_URL: &str = "http://127.0.0.1:28428";
pub const WORKSPACE_ROOT: &str = "/home/hue/8b-is";
pub const SG_ROOT: &str = "/home/hue/standardgalactic";

#[derive(Debug, Clone)]
pub struct McpToolDescriptor {
    pub name: &'static str,
    pub provider: &'static str,
    pub description: &'static str,
}

#[derive(Debug, Clone)]
pub struct McpToolRun {
    pub tool: String,
    pub target: String,
    pub elapsed_ms: u64,
    pub output: String,
}

#[derive(Debug, Clone)]
pub struct McpStatus {
    pub st_daemon_up: bool,
    pub st_daemon_latency_ms: u64,
    pub st_binary: Option<String>,
    pub crabcc_binary: Option<String>,
    pub ontology_ready: bool,
}

pub const TOOLS: &[McpToolDescriptor] = &[
    McpToolDescriptor {
        name: "st_overview",
        provider: "smart-tree (127.0.0.1:28428)",
        description: "Compressed AI structural overview of a project or directory",
    },
    McpToolDescriptor {
        name: "st_search",
        provider: "smart-tree (127.0.0.1:28428)",
        description: "Search file contents with line numbers across a repository",
    },
    McpToolDescriptor {
        name: "st_find",
        provider: "smart-tree (127.0.0.1:28428)",
        description: "Find code, tests, docs, recent changes, or projects",
    },
    McpToolDescriptor {
        name: "st_security",
        provider: "smart-tree (127.0.0.1:28428 · custodian)",
        description: "In-memory AST & supply-chain threat audit via unified Smart Tree MCP",
    },
    McpToolDescriptor {
        name: "crabcc_outline",
        provider: "smart-tree (127.0.0.1:28428 · symbol)",
        description: "Extract Tree-sitter structural symbol outline via unified Smart Tree MCP",
    },
    McpToolDescriptor {
        name: "crabcc_sym",
        provider: "smart-tree (127.0.0.1:28428 · symbol)",
        description: "Symbol definition + docstring + source code via unified Smart Tree MCP",
    },
    McpToolDescriptor {
        name: "crabcc_refs",
        provider: "smart-tree (127.0.0.1:28428 · symbol)",
        description: "Find all callers & reference lines via unified Smart Tree MCP",
    },
    McpToolDescriptor {
        name: "sovereign_ontology",
        provider: "smart-tree (127.0.0.1:28428 · ontology)",
        description: "Offline Three-Kaya Ontology ({-1, 0, +1}) & NAND gate via unified Smart Tree MCP",
    },
    McpToolDescriptor {
        name: "wa_partner",
        provider: "smart-tree (127.0.0.1:28428 · partner) × wa-stream (:8788/:8789)",
        description: "WhatsApp ('little dude(s) and Angels') + Discord Partner automation & Dra CEO briefs",
    },
    McpToolDescriptor {
        name: "dra_queen",
        provider: "queen-of-ceo-dra v3.0 × pocoo.vaked.dev × ALEXANDRA.md",
        description: "Queen of CEO Dra v3.0 brief, Triangle Covenant (Mom* + Dad/\\ + Son oo), and live 168-essay Pocoo shelf",
    },
    McpToolDescriptor {
        name: "standardgalactic",
        provider: "smart-tree × /home/hue/standardgalactic (155+ repos)",
        description: "Latest Standard Galactic research (Antivenom, Spherepop, Memnet, Viable Continuation, Admissibility Lab, HYDRA)",
    },
    McpToolDescriptor {
        name: "mlx_quant",
        provider: "gaia-mlx-quant::mlx (100% Pure-Rust MLX Engine · zero-alloc)",
        description: "Pure-Rust MLX & MLX-QUANT tensor engine: Group-64 12.80x, b1.58 20.19x, mergeq, maybe, gather_qmm MoE, STE autodiff, and Ternary Transformer",
    },
];

pub fn find_binary(name: &str) -> Option<String> {
    let candidates = [
        format!("/usr/local/bin/{name}"),
        format!("/home/hue/.local/bin/{name}"),
        format!("/home/hue/.cargo/bin/{name}"),
        format!("/usr/bin/{name}"),
    ];
    candidates.into_iter().find(|c| Path::new(c).exists())
}

pub fn load_st_token() -> Option<String> {
    let paths = [
        PathBuf::from("/var/lib/smart-tree/daemon.token"),
        PathBuf::from("/home/hue/.st/daemon.token"),
    ];
    for p in paths {
        if let Ok(s) = std::fs::read_to_string(&p) {
            let trimmed = s.trim().to_string();
            if !trimmed.is_empty() {
                return Some(trimmed);
            }
        }
    }
    None
}

pub fn status() -> McpStatus {
    let t0 = Instant::now();
    let st_daemon_up = transport::request(&format!("{ST_DAEMON_URL}/health"), "GET", None)
        .map(|r| r.status == 200)
        .unwrap_or(false);
    let st_daemon_latency_ms = t0.elapsed().as_millis() as u64;

    McpStatus {
        st_daemon_up,
        st_daemon_latency_ms,
        st_binary: find_binary("st"),
        crabcc_binary: find_binary("crabcc"),
        ontology_ready: Path::new("/home/hue/8b-is/fandom-mcp/fandom-mcp.mjs").exists(),
    }
}

/// Call Smart Tree HTTP MCP daemon (`127.0.0.1:28428/mcp/tools/call`).
fn call_st_daemon(tool_name: &str, arguments_json: &str) -> Result<String, String> {
    let url = format!("{ST_DAEMON_URL}/mcp/tools/call");
    let body = format!(r#"{{"name":"{tool_name}","arguments":{arguments_json}}}"#);
    let auth_header = load_st_token().map(|t| format!("Bearer {t}"));
    let resp = if let Some(ref bearer) = auth_header {
        transport::request_with_headers(&url, "POST", &[("Authorization", bearer.as_str())], Some(&body))?
    } else {
        transport::request(&url, "POST", Some(&body))?
    };
    if resp.status != 200 {
        return Err(format!("HTTP {}", resp.status));
    }
    let text = resp.body_utf8();
    if let Ok(val) = parse_json(&text)
        && let Some(arr) = val.get("content").and_then(|c| c.as_array())
    {
        let mut chunks = Vec::new();
        for item in arr {
            if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
                chunks.push(t.to_string());
            }
        }
        if !chunks.is_empty() {
            return Ok(truncate_output(&chunks.join("\n"), 3200));
        }
    }
    Ok(truncate_output(&text, 3200))
}

fn resolve_workspace_path(raw: &str) -> String {
    let trimmed = raw.trim().trim_matches(|c| matches!(c, '"' | '\'' | '`'));
    if trimmed.is_empty() || trimmed == "." {
        return "/home/hue/8b-is/alexiai".to_string();
    }
    if trimmed.eq_ignore_ascii_case("dra") || trimmed.eq_ignore_ascii_case("queen-of-ceo-dra") {
        return "/home/hue/8b-is/queen-of-ceo-dra".to_string();
    }
    if trimmed.eq_ignore_ascii_case("standardgalactic") || trimmed.eq_ignore_ascii_case("sg") {
        return SG_ROOT.to_string();
    }
    if let Some(rest) = trimmed.strip_prefix("~/") {
        return format!("/home/hue/{rest}");
    }
    if Path::new(trimmed).is_absolute() {
        return trimmed.to_string();
    }
    let candidate_alexiai = format!("/home/hue/8b-is/alexiai/{trimmed}");
    if Path::new(&candidate_alexiai).exists() {
        return candidate_alexiai;
    }
    let candidate_8b = format!("{WORKSPACE_ROOT}/{trimmed}");
    if Path::new(&candidate_8b).exists() {
        return candidate_8b;
    }
    let candidate_sg = format!("{SG_ROOT}/{trimmed}");
    if Path::new(&candidate_sg).exists() {
        return candidate_sg;
    }
    trimmed.to_string()
}

pub fn call_tool(tool: &str, arg: &str) -> Result<McpToolRun, String> {
    let t0 = Instant::now();
    let target = arg.trim().to_string();

    let output = match tool {
        "st_overview" => {
            let path = resolve_workspace_path(&target);
            let args_json = format!(r#"{{"mode":"quick","path":"{}","depth":3}}"#, json_escape(&path));
            match call_st_daemon("overview", &args_json) {
                Ok(out) => out,
                Err(_) => run_st_cli(&["--mode", "ai", "--depth", "3", &path])?,
            }
        }
        "st_search" => {
            let (keyword, path) = split_arg_and_path(&target, "/home/hue/8b-is/alexiai");
            let args_json = format!(
                r#"{{"keyword":"{}","path":"{}","max_results":20,"include_content":true}}"#,
                json_escape(&keyword),
                json_escape(&path)
            );
            match call_st_daemon("search", &args_json) {
                Ok(out) => out,
                Err(_) => run_st_cli(&["--search", &keyword, &path])?,
            }
        }
        "st_find" => {
            let find_type = if target.is_empty() { "projects" } else { target.as_str() };
            let args_json = format!(
                r#"{{"type":"{}","path":"{}","limit":25}}"#,
                json_escape(find_type),
                WORKSPACE_ROOT
            );
            match call_st_daemon("find", &args_json) {
                Ok(out) => out,
                Err(_) => run_st_cli(&["--mode", "ai", "--depth", "2", WORKSPACE_ROOT])?,
            }
        }
        "st_security" | "security" => {
            let path = resolve_workspace_path(&target);
            let args_json = format!(r#"{{"path":"{}"}}"#, json_escape(&path));
            match call_st_daemon("security", &args_json) {
                Ok(out) => out,
                Err(_) => run_st_cli(&["--security-scan", &path])?,
            }
        }
        "crabcc_outline" => {
            let path = resolve_workspace_path(&target);
            let args_json = format!(
                r#"{{"operation":"outline","file":"{}","path":"/home/hue/8b-is/alexiai"}}"#,
                json_escape(&path)
            );
            match call_st_daemon("symbol", &args_json) {
                Ok(out) => out,
                Err(_) => run_crabcc_cli(&["lookup", "outline", &path], "/home/hue/8b-is/alexiai")?,
            }
        }
        "crabcc_sym" | "symbol" => {
            let (sym, repo_path) = split_arg_and_path(&target, "/home/hue/8b-is/alexiai");
            let args_json = format!(
                r#"{{"operation":"sym","name":"{}","path":"{}","include_source":true}}"#,
                json_escape(&sym),
                json_escape(&repo_path)
            );
            match call_st_daemon("symbol", &args_json) {
                Ok(out) if !out.trim().is_empty() && out.trim() != "[]" => out,
                _ => {
                    let raw = run_crabcc_cli(&["lookup", "sym", &sym], &repo_path)?;
                    enrich_crabcc_sym_output(&raw, &repo_path)
                }
            }
        }
        "crabcc_refs" => {
            let (sym, repo_path) = split_arg_and_path(&target, "/home/hue/8b-is/alexiai");
            let args_json = format!(
                r#"{{"operation":"refs","name":"{}","path":"{}","include_source":true}}"#,
                json_escape(&sym),
                json_escape(&repo_path)
            );
            match call_st_daemon("symbol", &args_json) {
                Ok(out) if !out.trim().is_empty() && out.trim() != "[]" => out,
                _ => run_crabcc_cli(&["lookup", "refs", &sym], &repo_path)?,
            }
        }
        "sovereign_ontology" | "ontology" => {
            let entity = if target.is_empty() { "Ultra Instinct" } else { target.as_str() };
            let args_json = format!(r#"{{"target":"{}"}}"#, json_escape(entity));
            match call_st_daemon("ontology", &args_json) {
                Ok(out) => out,
                Err(_) => run_cmd(
                    "node",
                    &["/home/hue/8b-is/fandom-mcp/fandom-mcp.mjs", "--ontology", entity],
                    WORKSPACE_ROOT,
                )?,
            }
        }
        "wa_partner" | "partner" | "wa_status" | "wa_inbox" | "wa_send" | "wa_slash" | "discord_send" => {
            let (op, text, channel) = match tool {
                "wa_status" => ("status", "", "both"),
                "wa_inbox" => ("inbox", "", "both"),
                "wa_send" => ("send", target.as_str(), "whatsapp"),
                "discord_send" => ("send", target.as_str(), "discord"),
                "wa_slash" => ("slash", if target.is_empty() { "/dra" } else { target.as_str() }, "both"),
                _ => {
                    if target.is_empty() || target.eq_ignore_ascii_case("status") {
                        ("status", "", "both")
                    } else if target.eq_ignore_ascii_case("inbox") {
                        ("inbox", "", "both")
                    } else if target.starts_with('/') {
                        ("slash", target.as_str(), "both")
                    } else if let Some(rest) = target.strip_prefix("send ") {
                        ("send", rest.trim(), "both")
                    } else {
                        ("status", "", "both")
                    }
                }
            };
            let args_json = format!(
                r#"{{"operation":"{}","text":"{}","channel":"{}"}}"#,
                json_escape(op),
                json_escape(text),
                json_escape(channel)
            );
            match call_st_daemon("partner", &args_json) {
                Ok(out) => out,
                Err(_) => call_partner_hub_direct(op, text, channel)?,
            }
        }
        "dra_queen" | "dra" | "queen" | "reads" | "pocoo" | "triangle" => {
            run_dra_queen_tool(&target)?
        }
        "standardgalactic" | "sg" => {
            run_standardgalactic_tool(&target)?
        }
        "mlx_quant" | "mlx" | "mlx-rs" => {
            run_mlx_quant_tool(&target)
        }
        "mcp_status" | "status" | "tools" => {
            let st = status();
            let mut report = format!(
                "Unified Smart Tree MCP Root Daemon (`http://127.0.0.1:28428`)\n\
                 - Daemon status: {} (latency: {} ms)\n\
                 - Binary (`st`): {}\n\
                 - Merged Code Intelligence (`crabcc` -> `symbol` tool): {}\n\
                 - Merged Sovereign Ontology (`ontology` tool): ready ({})\n\n\
                 Available Unified Local MCP Tools (all routed via `127.0.0.1:28428/mcp/tools/call`):\n",
                if st.st_daemon_up { "ONLINE (systemd root service active)" } else { "CLI fallback" },
                st.st_daemon_latency_ms,
                st.st_binary.as_deref().unwrap_or("not found"),
                st.crabcc_binary.as_deref().unwrap_or("not found"),
                st.ontology_ready
            );
            for t in TOOLS {
                report.push_str(&format!(
                    "• `{}` [{}] — {}\n",
                    t.name, t.provider, t.description
                ));
            }
            report.push_str(
                "\nExample prompts or `/mcp` commands you can test right now:\n\
                 1. `Where is symbol quantize_ternary defined and what does it do?` (or `/mcp crabcc_sym quantize_ternary`)\n\
                 2. `Outline rust/alexiai/src/server.rs` (or `/mcp crabcc_outline rust/alexiai/src/server.rs`)\n\
                 3. `Explore smart-tree overview of alexiai` (or `/mcp st_overview /home/hue/8b-is/alexiai`)\n\
                 4. `Run a security scan on alexiai` (or `/mcp st_security /home/hue/8b-is/alexiai`)\n\
                 5. `Run Sovereign Ontology for Ultra Instinct` (or `/mcp sovereign_ontology Ultra Instinct`)\n\
                 6. `Check our WhatsApp and Discord partner status` (or `/mcp wa_partner status` / `/mcp wa_slash /dra` / `/mcp wa_send Hello family <3`)"
            );
            report
        }
        _ => return Err(format!("unknown MCP tool: {tool}")),
    };

    Ok(McpToolRun {
        tool: tool.to_string(),
        target: if target.is_empty() { ".".to_string() } else { target },
        elapsed_ms: t0.elapsed().as_millis() as u64,
        output,
    })
}

fn call_partner_hub_direct(op: &str, text: &str, channel: &str) -> Result<String, String> {
    let url = match op {
        "inbox" => "http://127.0.0.1:8789/inbox?limit=15",
        "outbox" => "http://127.0.0.1:8789/outbox?limit=15",
        "reads" | "pocoo" => "http://127.0.0.1:8789/reads?limit=6",
        "send" => "http://127.0.0.1:8789/send",
        "slash" => "http://127.0.0.1:8789/slash",
        _ => "http://127.0.0.1:8789/status",
    };
    let resp = match op {
        "send" => {
            let body = format!(
                r#"{{"text":"{}","channel":"{}"}}"#,
                json_escape(text),
                json_escape(channel)
            );
            transport::request(url, "POST", Some(&body))?
        }
        "slash" => {
            let body = format!(
                r#"{{"command":"{}","channel":"{}"}}"#,
                json_escape(text),
                json_escape(channel)
            );
            transport::request(url, "POST", Some(&body))?
        }
        _ => transport::request(url, "GET", None)?,
    };
    Ok(truncate_output(&resp.body_utf8(), 3200))
}

fn run_dra_queen_tool(target: &str) -> Result<String, String> {
    let lower = target.trim().to_ascii_lowercase();
    let helper = "/home/hue/8b-is/wa-stream/dra-ceo-helper.mjs";
    let mut sections = Vec::new();

    if lower.contains("read") || lower.contains("pocoo") || lower.contains("triangle") {
        if let Ok(out) = run_cmd("node", &[helper, "--reads", "--print"], WORKSPACE_ROOT) {
            sections.push(out);
        }
    } else {
        let topic = if target.trim().is_empty() {
            "ZENO (zeno.vaked.dev), Triangle Track (Mom* + Dad/\\ + Son oo) & Pocoo Music-Lane Shelf"
        } else {
            target.trim()
        };
        if let Ok(out) = run_cmd("node", &[helper, "--queen", "--print", "--brief", topic], WORKSPACE_ROOT) {
            sections.push(out);
        }
    }

    if let Ok(readme) = std::fs::read_to_string("/home/hue/8b-is/queen-of-ceo-dra/README.md") {
        sections.push(format!(
            "Repository `/home/hue/8b-is/queen-of-ceo-dra` (`8b-is/queen-of-ceo-dra`):\n{}",
            truncate_output(readme.trim(), 900)
        ));
    }
    if let Ok(alexandra) = std::fs::read_to_string("/home/hue/8b-is/alexiai/ALEXANDRA.md") {
        sections.push(format!(
            "Heart Ledger `/home/hue/8b-is/alexiai/ALEXANDRA.md`:\n{}",
            truncate_output(alexandra.trim(), 600)
        ));
    }

    Ok(truncate_output(&sections.join("\n\n"), 3200))
}

fn run_standardgalactic_tool(target: &str) -> Result<String, String> {
    const LATEST_SG_REPOS: &[(&str, &str)] = &[
        ("standardgalactic", "Home of the Standard Galactic Alphabet · 5D Ising synchronization & Mandelbrot/Life boundary"),
        ("antivenom", "Venenum sola facit dosis — Wireheading & the Conditions of a Common World, Active Inference, Adaptive Code Reader, Affine Quantum Deformation, Against Namespace Laundering"),
        ("viable-continuation", "The Economics of Viable Continuation — Unit of Scrutiny, Concealment Encryption, Distributed Material Archives, Relief of Explanation, Stone Piano"),
        ("memnet", "Semantic-state architecture for adaptive-information rerouting — The Residue of Proof, HYDRA Operator Logic, Architecture of Constraint, Constraint as Ground"),
        ("spherepop", "Combinators laboratory — Duplication & Universality in circle calculus, Active Geodesic Inference, Averaging & Correcting"),
        ("admissibility-lab", "Prolegomena ad Omnem Metaphysicam Futuram — From Corpus to Argument, Metered Memory, Cost of Future Admissibility, adm_core / adm_graph"),
        ("persistent-distinctions", "Persistent distinctions in complex systems — Distinction and Continuation, The Shape of Possibility, comparative benchmarks"),
        ("machine-learning", "Order-aware data attribution (`ordered_attribution.py`) & 5-seed curriculum experiment (`curriculum_experiment.py`)"),
        ("HYDRA", "Hybrid Dynamic Reasoning Architecture — RSVP, TARTAN, and HYDRA unification theory & Chain of Memory"),
        ("library", "Enacting Cognitive Landscapes — Agency Is Not Computation, RSVP theory, active-inference glossary (Holographic Commons)"),
        ("fonts", "Procedural Typography Laboratory — Specimens for a Procedural Typography, Galactic Lettering, adversarial OCR experiments"),
        ("sitemap", "Guide to Forbidden Knowledge — deployed Standard Galactic repository index & branch topology"),
        ("research-projects", "Never Knowingly Undersold — cross-project research index"),
    ];

    let clean = target.trim().trim_matches(|c: char| matches!(c, '?' | '.' | ',' | '!' | '"' | '\'' | '`'));
    let lower = clean.to_ascii_lowercase();

    // Check if target matches a specific repo in /home/hue/standardgalactic
    let matched_repo = LATEST_SG_REPOS
        .iter()
        .map(|(name, _)| *name)
        .find(|name| lower.contains(&name.to_ascii_lowercase()))
        .or_else(|| {
            if !clean.is_empty() && clean != "latest" && clean != "list" && clean != "all" {
                let candidate = format!("{SG_ROOT}/{clean}");
                if Path::new(&candidate).exists() {
                    return Some(clean);
                }
            }
            None
        });

    if let Some(repo_name) = matched_repo {
        let repo_path = format!("{SG_ROOT}/{repo_name}");
        let mut out = format!("Standard Galactic Repository: `{repo_path}` (`standardgalactic/{repo_name}`)\n");
        if let Ok(readme) = std::fs::read_to_string(format!("{repo_path}/README.md")) {
            out.push_str("\nREADME.md:\n");
            out.push_str(&truncate_output(readme.trim(), 1600));
            out.push('\n');
        }
        if let Ok(entries) = std::fs::read_dir(&repo_path) {
            let mut files: Vec<String> = entries
                .flatten()
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|f| !f.starts_with('.'))
                .collect();
            files.sort();
            out.push_str(&format!(
                "\nTop-level files ({} total): {}\n",
                files.len(),
                files.into_iter().take(25).collect::<Vec<_>>().join(", ")
            ));
        }
        return Ok(truncate_output(&out, 3200));
    }

    let total_repos = std::fs::read_dir(SG_ROOT)
        .map(|it| it.flatten().filter(|e| !e.file_name().to_string_lossy().starts_with('.')).count())
        .unwrap_or(155);

    let mut report = format!(
        "Standard Galactic Constellation (`{SG_ROOT}` — {total_repos} repositories linked on disk)\n\
         Latest 2026 Research Repositories:\n"
    );
    for (name, desc) in LATEST_SG_REPOS {
        let path = format!("{SG_ROOT}/{name}");
        let status_mark = if Path::new(&path).exists() { "✔" } else { "·" };
        report.push_str(&format!("• {status_mark} `standardgalactic/{name}` — {desc}\n"));
    }
    report.push_str(
        "\nTip: Ask about any repo by name (e.g. `antivenom`, `spherepop`, `memnet`, `viable-continuation`, `admissibility-lab`, `persistent-distinctions`, `HYDRA`, `machine-learning`) or run `/mcp standardgalactic <repo>`."
    );
    Ok(truncate_output(&report, 3200))
}

fn run_mlx_quant_tool(target: &str) -> String {
    let dim: usize = target
        .split_whitespace()
        .find_map(|tok| tok.parse::<usize>().ok())
        .unwrap_or(256)
        .clamp(64, 1024);
    let bench_json = crate::server::run_bench(dim);
    format!(
        "Pure-Rust MLX & MLX-QUANT Engine (`gaia_mlx_quant::mlx` in `/home/hue/8b-is/alexiai/rust/gaia-mlx-quant/src/mlx.rs`)\n\
         • Attributes: `#![no_std]`, `#![forbid(unsafe_code)]`, zero heap allocations on all hot paths (proven by `tests/zero_alloc.rs`)\n\
         • Modes implemented in 100% safe Rust:\n\
           1. `TernaryGroup64` (`mlx_quantize_group64_into` / `mlx_dequantize_group64_into` / `mlx_quantized_matmul_into`): 16 two-bit trits per `u32` word, `group_size=64`, exact `12.80x` compression vs F32 (wire-compatible with `8b-is/mlx-quant-mcp`)\n\
           2. `TernaryB158` (`ternary_matmul` / `ternary_matmul_i_major` / `ternary_matmul_planes_i`): 5 trits per byte (`3^5 = 243`), `1.585` bits/weight, `20.19x` compression vs F32\n\
           3. `MergeQ` (`mergeq_matmul_into`): triple-fused `quantize + dequantize + matmul` in a single streaming register pass with 0 B intermediate buffer\n\
           4. `Maybe` (`maybe_matmul_into`): probabilistic stochastic ternary matmul with SplitMix64 seed (`E[w_hat] = w`)\n\
           5. `GatherQMM` (`gather_qmm_into`): Mixture-of-Experts (MoE) top-k routed quantized matmul\n\
           6. `StridedView`: zero-copy 2D unified-memory tensor views (`row_major`, `transpose`, `slice`)\n\
           7. `STE Autodiff` (`ste_linear_backward_into`): Straight-Through Estimator reverse-mode gradients for BitNet b1.58 QAT\n\
           8. `TernaryTransformerBlock` (`ternary_transformer_step_into`): full layer (`rms_norm_into` + Ternary QKV + `rope_into` + causal `sdpa_single_query_into` + Ternary O-Proj + `swiglu_into` MLP)\n\n\
         Live Measured Benchmark (`dim = {dim}`):\n```json\n{bench_json}\n```"
    )
}

fn split_arg_and_path(input: &str, default_path: &str) -> (String, String) {
    let parts: Vec<&str> = input.split_whitespace().collect();
    if parts.len() >= 2 {
        let first = parts[0].to_string();
        let second = resolve_workspace_path(&parts[1..].join(" "));
        (first, second)
    } else {
        (input.trim().to_string(), default_path.to_string())
    }
}

fn run_st_cli(args: &[&str]) -> Result<String, String> {
    let bin = find_binary("st").ok_or_else(|| "smart-tree (st) binary not found".to_string())?;
    run_cmd(&bin, args, WORKSPACE_ROOT)
}

fn run_crabcc_cli(args: &[&str], cwd: &str) -> Result<String, String> {
    let bin = find_binary("crabcc").ok_or_else(|| "crabcc binary not found".to_string())?;
    run_cmd(&bin, args, cwd)
}

/// Enrich `crabcc lookup sym` JSON output by reading the actual source code
/// and docstrings around `line_start..=line_end` from disk.
fn enrich_crabcc_sym_output(raw_json: &str, repo_path: &str) -> String {
    let Ok(parsed) = parse_json(raw_json) else {
        return raw_json.to_string();
    };
    let Some(items) = parsed.as_array() else {
        return raw_json.to_string();
    };
    if items.is_empty() {
        return raw_json.to_string();
    }

    let mut sections = Vec::new();
    for item in items.iter().take(3) {
        let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        let kind = item.get("kind").and_then(|v| v.as_str()).unwrap_or("symbol");
        let sig = item.get("signature").and_then(|v| v.as_str()).unwrap_or("");
        let file = item.get("file").and_then(|v| v.as_str()).unwrap_or("");
        let line_start = item.get("line_start").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let line_end = item.get("line_end").and_then(|v| v.as_u64()).unwrap_or(line_start as u64) as usize;

        let mut block = format!(
            "Symbol `{name}` ({kind})\nDefined in: `{file}` (lines {line_start}–{line_end})\nSignature: `{sig}`"
        );

        let full_path = Path::new(repo_path).join(file);
        if let Ok(src) = std::fs::read_to_string(&full_path) {
            let lines: Vec<&str> = src.lines().collect();
            // Include up to 8 preceding lines to capture `///` doc comments
            let mut start_idx = line_start.saturating_sub(1);
            let min_idx = start_idx.saturating_sub(8);
            while start_idx > min_idx {
                let prev = lines.get(start_idx - 1).map(|l| l.trim()).unwrap_or("");
                if prev.starts_with("///") || prev.starts_with("//") || prev.starts_with("#[") || prev.starts_with("/*") || prev.starts_with('*') {
                    start_idx -= 1;
                } else {
                    break;
                }
            }
            let end_idx = line_end.min(lines.len()).min(start_idx + 60);
            if start_idx < end_idx {
                block.push_str("\nSource code:\n```rust\n");
                for (idx, line) in lines[start_idx..end_idx].iter().enumerate() {
                    block.push_str(&format!("{:>4}: {}\n", start_idx + idx + 1, line));
                }
                block.push_str("```");
            }
        }
        sections.push(block);
    }
    sections.join("\n\n")
}

fn run_cmd(bin: &str, args: &[&str], cwd: &str) -> Result<String, String> {
    let out = Command::new(bin)
        .args(args)
        .current_dir(cwd)
        .env("CRABCC_NO_HINT", "1")
        .output()
        .map_err(|e| format!("{bin}: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let combined = if stdout.trim().is_empty() {
        stderr.to_string()
    } else {
        stdout.to_string()
    };
    Ok(truncate_output(combined.trim(), 3200))
}

fn truncate_output(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let head: String = s.chars().take(max_chars).collect();
        format!("{head}\n… [truncated by local MCP bridge]")
    }
}

/// Automatically detect and execute relevant local MCP tools for a user prompt.
/// Supports explicit `/mcp <tool> <arg>` commands as well as natural-language
/// code exploration requests.
pub fn auto_ground_prompt(prompt: &str) -> Vec<McpToolRun> {
    let trimmed = prompt.trim();
    let mut runs = Vec::new();

    // 1. Explicit `/mcp <tool> [arg]` command
    if let Some(rest) = trimmed.strip_prefix("/mcp ") {
        let mut parts = rest.trim().splitn(2, char::is_whitespace);
        let tool = parts.next().unwrap_or("st_overview");
        let arg = parts.next().unwrap_or("");
        if let Ok(run) = call_tool(tool, arg) {
            runs.push(run);
        }
        return runs;
    }

    let lower = trimmed.to_ascii_lowercase();

    // 2. Natural-language outline request (e.g., "outline server.rs" or "outline rust/alexiai/src/main.rs")
    if lower.contains("outline")
        && let Some(file_token) = trimmed
            .split_whitespace()
            .find(|w| w.ends_with(".rs") || w.ends_with(".c") || w.ends_with(".py") || w.ends_with(".js") || w.ends_with(".go") || w.ends_with(".swift"))
        && let Ok(run) = call_tool("crabcc_outline", file_token)
    {
        runs.push(run);
    }

    // 3. Natural-language symbol lookup (e.g., "find symbol fold_spectrum" or "where is quantize_ternary defined")
    if (lower.contains("symbol") || lower.contains("where is ") || lower.contains("definition of ") || lower.contains("crabcc"))
        && let Some(sym) = extract_code_identifier(trimmed)
        && let Ok(run) = call_tool("crabcc_sym", &sym)
    {
        runs.push(run);
    }

    // 4. Natural-language security scan request
    if (lower.contains("security scan") || lower.contains("custodian") || lower.contains("audit"))
        && (lower.contains("tree") || lower.contains("scan") || lower.contains("repo") || lower.contains("code"))
    {
        let target = extract_repo_or_path(trimmed).unwrap_or_else(|| "/home/hue/8b-is/alexiai".to_string());
        if let Ok(run) = call_tool("st_security", &target) {
            runs.push(run);
        }
    }

    // 5. Natural-language project/directory overview via Smart Tree
    if (lower.contains("smart-tree")
        || lower.contains("smart tree")
        || lower.contains("overview of")
        || lower.contains("explore ")
        || lower.contains("tree of")
        || lower.contains("structure of")
        || lower.contains("what is in "))
        && runs.is_empty()
    {
        let target = extract_repo_or_path(trimmed).unwrap_or_else(|| "/home/hue/8b-is/alexiai".to_string());
        if let Ok(run) = call_tool("st_overview", &target) {
            runs.push(run);
        }
    }

    // 6. Sovereign Ontology / Dragon Ball Three-Kaya alignment
    if lower.contains("ontology") || lower.contains("three-kaya") || lower.contains("nand gate") || lower.contains("scouter") {
        let entity = extract_quoted_or_tail(trimmed).unwrap_or_else(|| "Ultra Instinct".to_string());
        if let Ok(run) = call_tool("sovereign_ontology", &entity) {
            runs.push(run);
        }
    }

    // 7. Queen of CEO Dra v3.0 / Pocoo Shelf / Triangle Covenant
    if lower.contains("queen-of-ceo-dra")
        || lower.contains("queen of ceo")
        || lower.contains("dra repo")
        || lower.contains("pocoo")
        || lower.contains("triangle")
        || lower.contains("alexandra")
        || lower.contains("/reads")
        || lower.contains("/pocoo")
        || lower.contains("/triangle")
        || lower.contains("/evolve")
    {
        let arg = if lower.contains("read") || lower.contains("pocoo") || lower.contains("triangle") {
            "reads".to_string()
        } else {
            trimmed.to_string()
        };
        if let Ok(run) = call_tool("dra_queen", &arg) {
            runs.push(run);
        }
    }

    // 8. Standard Galactic Research Constellation (Flyxion / RSVP / TARTAN / HYDRA / Antivenom / Spherepop / Memnet)
    if lower.contains("standardgalactic")
        || lower.contains("standard galactic")
        || lower.contains("flyxion")
        || lower.contains("antivenom")
        || lower.contains("spherepop")
        || lower.contains("memnet")
        || lower.contains("viable-continuation")
        || lower.contains("viable continuation")
        || lower.contains("admissibility")
        || lower.contains("persistent-distinctions")
        || lower.contains("persistent distinctions")
        || lower.contains("rsvp")
        || lower.contains("tartan")
        || lower.contains("hydra")
    {
        let arg = if lower.contains("antivenom") {
            "antivenom"
        } else if lower.contains("spherepop") {
            "spherepop"
        } else if lower.contains("memnet") {
            "memnet"
        } else if lower.contains("viable") {
            "viable-continuation"
        } else if lower.contains("admissibility") {
            "admissibility-lab"
        } else if lower.contains("persistent") {
            "persistent-distinctions"
        } else if lower.contains("hydra") || lower.contains("rsvp") || lower.contains("tartan") {
            "HYDRA"
        } else {
            "latest"
        };
        if let Ok(run) = call_tool("standardgalactic", arg) {
            runs.push(run);
        }
    }

    // 9. Pure-Rust MLX & MLX-QUANT Tensor Engine
    if (lower.contains("mlx")
        || lower.contains("mergeq")
        || lower.contains("maybe_matmul")
        || lower.contains("gather_qmm")
        || lower.contains("bitnet")
        || lower.contains("b1.58")
        || (lower.contains("rust") && (lower.contains("quant") || lower.contains("tensor"))))
        && let Ok(run) = call_tool("mlx_quant", "256")
    {
        runs.push(run);
    }

    // 10. WhatsApp (wa-stream) & Discord Partner-in-Life automation
    if lower.contains("whatsapp")
        || lower.contains("wa-stream")
        || lower.contains("wastream")
        || lower.contains("discord")
        || lower.contains("partner")
        || lower.contains("buffett")
        || lower.contains("/dra")
        || lower.contains("/family")
    {
        let arg = if lower.contains("buffett") {
            "/buffett".to_string()
        } else if lower.contains("/dra") {
            "/dra".to_string()
        } else if lower.contains("/family") {
            "/family+1".to_string()
        } else if lower.contains("inbox") || lower.contains("messages") {
            "inbox".to_string()
        } else if let Some(idx) = lower.find("send ") {
            format!("send {}", &trimmed[idx + 5..])
        } else {
            "status".to_string()
        };
        if let Ok(run) = call_tool("wa_partner", &arg) {
            runs.push(run);
        }
    }

    // 10. General MCP / tools inquiry or diagnostic test ("test mcp servers or tools you have", "what tools do you have")
    if (lower.contains("mcp") || lower.contains("tools") || lower.contains("tool "))
        && runs.is_empty()
        && let Ok(run) = call_tool("mcp_status", "smart-tree-daemon")
    {
        runs.push(run);
    }

    runs
}

fn extract_code_identifier(prompt: &str) -> Option<String> {
    let stopwords = [
        "where", "is", "the", "symbol", "function", "struct", "defined",
        "definition", "of", "in", "find", "lookup", "crabcc", "for", "show",
        "me", "can", "you", "with", "use", "using", "please", "what",
    ];
    for raw in prompt.split_whitespace() {
        let clean = raw.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != ':');
        if clean.len() >= 3
            && !stopwords.contains(&clean.to_ascii_lowercase().as_str())
            && (clean.contains('_') || clean.chars().any(|c| c.is_uppercase()) || clean.len() >= 5)
        {
            return Some(clean.to_string());
        }
    }
    None
}

fn extract_repo_or_path(prompt: &str) -> Option<String> {
    for raw in prompt.split_whitespace() {
        let clean = raw.trim_matches(|c: char| matches!(c, '?' | '.' | ',' | '!' | '"' | '\'' | '`'));
        if clean.is_empty() {
            continue;
        }
        if clean.starts_with('/') || clean.starts_with("~/") {
            return Some(resolve_workspace_path(clean));
        }
        let candidate_8b = format!("{WORKSPACE_ROOT}/{clean}");
        if Path::new(&candidate_8b).exists() {
            return Some(candidate_8b);
        }
        let candidate_sg = format!("{SG_ROOT}/{clean}");
        if Path::new(&candidate_sg).exists() {
            return Some(candidate_sg);
        }
    }
    None
}

fn extract_quoted_or_tail(prompt: &str) -> Option<String> {
    if let Some((_, after)) = prompt.split_once('"')
        && let Some((inside, _)) = after.split_once('"')
        && !inside.trim().is_empty()
    {
        return Some(inside.trim().to_string());
    }
    None
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

pub fn parse_call_request(body: &str) -> Option<(String, String)> {
    let v = parse_json(body).ok()?;
    let tool = v.get("tool").or_else(|| v.get("name"))?.as_str()?.to_string();
    let arg = match v.get("arg").or_else(|| v.get("target")).or_else(|| v.get("path")) {
        Some(Value::Str(s)) => s.clone(),
        _ => String::new(),
    };
    Some((tool, arg))
}
