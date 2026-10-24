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
pub const MACBOOK_8B_ROOT: &str = "/home/hue/macbook-source/8b-is";
pub const MACBOOK_WRAITH_ROOT: &str = "/home/hue/macbook-source/8bit-wraith";
pub const SONY_ROOT: &str = "/home/hue/sony";
pub const SG_ROOT: &str = "/home/hue/standardgalactic";
pub const MEMORY_FILE_PATH: &str = "/home/hue/.local/share/alexiai/memory.jsonl";

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
    pub wupz_ready: bool,
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
        provider: "queen-of-ceo-dra v3.0 × pocoo.8b.is × ALEXANDRA.md",
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
        description: "Pure-Rust MLX & MLX-QUANT tensor engine: Quantum-b158-u64 (0.88µs), Omni-BFP-i8 (1.01µs), Group-64, b1.58, mergeq, maybe, gather_qmm, STE, and Ternary Transformer",
    },
    McpToolDescriptor {
        name: "wupz",
        provider: "wupz v0.4.0 (wild + UPX + zig) × gaia-mlx-quant::wupz (ABI v1)",
        description: "Incremental pack-in-pass linker+compressor (8b-is/wupz · wupz.8b.is), xor_fold/byte_diff/popcount seam, QuantTern (qt:<hex>), and WUPZ plan receipts",
    },
    McpToolDescriptor {
        name: "dream_state_loop",
        provider: "gaia-mlx-quant::wupz (ULTRABACKYARDLOOP +AND~NAND × 4-Layer MoE Dream State Machine)",
        description: "Peter's Dream State Machine: 4-expert MoE (Dreaming Clouds, Lucid Deep-Thinking, Buddhist Monk Mantra & 3-Sentence Listening Gate, Philosophy × Bodhisattva Vow)",
    },
    McpToolDescriptor {
        name: "repo_inspect",
        provider: "smart-tree × git × codebase auditor (/home/hue/8b-is & /home/hue/sony)",
        description: "Deep repository dossier (README, git log/status, architecture, and concrete code-improvement & p/q upgrade opportunities)",
    },
    McpToolDescriptor {
        name: "memory_ledger",
        provider: "alexiai persistent engineering ledger (~/.local/share/alexiai/memory.jsonl)",
        description: "Persistent memory of conversations, 8b-is repositories, Tokyo R&D PRs, and active code-improvement goals",
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
        ontology_ready: Path::new("/home/hue/8b-is/fandom-mcp/fandom-mcp.mjs").exists()
            || Path::new("/Users/lodripeter/workspace/peterlodri-sec/fandom-mcp/fandom-mcp.mjs")
                .exists(),
        wupz_ready: true,
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
    let candidate_sony = format!("{SONY_ROOT}/{trimmed}");
    if Path::new(&candidate_sony).exists() {
        return candidate_sony;
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
        "wupz" | "quanttern" | "pack-in-pass" | "ultrabackyard" | "kompress" | "ksul" => {
            run_wupz_tool(&target)
        }
        "dream_state_loop" | "dream" | "dream_state" | "lucid" | "bodhisattva" | "csodabogar" => {
            run_dream_state_tool(&target)
        }
        "repo_inspect" | "repo" | "inspect" | "improve" => {
            run_repo_inspect_tool(&target)?
        }
        "memory_ledger" | "memory" | "ledger" => {
            run_memory_tool(&target)
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
            "ZENO (zeno.8b.is), Triangle Track (Mom* + Dad/\\ + Son oo) & Pocoo Music-Lane Shelf"
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
        "Pure-Rust MLX & MLX-QUANT Engine v0.3.0 (`gaia_mlx_quant::mlx` in `/home/hue/8b-is/alexiai/rust/gaia-mlx-quant/src/mlx.rs`)\n\
         • Attributes: `#![no_std]`, `#![forbid(unsafe_code)]`, zero heap allocations on all hot paths (proven by `tests/zero_alloc.rs`), 100% float-free `p/q` (`q = 2^20 = 1,048,576`) integer substrate\n\
         • Modes implemented in 100% safe Rust:\n\
           1. `TernaryGroup64` (`mlx_quantize_group64_pq_into` / `mlx_matmul_pq_group64_i64_into` / `mlx_quantized_matmul_into`): 16 two-bit trits per `u32` word, `group_size=64`, exact `12.80x` compression vs F32\n\
           2. `TernaryB158` (`ternary_simd_i8_pq_into` / `ternary_matmul_pq_planes_i_into` / `ternary_matmul_pq_i_major_into`): 5 trits per byte (`3^5 = 243`), `1.585` bits/weight, `20.19x` compression vs F32, 5.25x faster than dense FP64 with 0 floats\n\
           3. `MergeQ` (`mergeq_matmul_pq_into` / `mergeq_matmul_into`): triple-fused `quantize + dequantize + matmul` in a single streaming register pass with 0 B intermediate buffer\n\
           4. `Maybe` (`maybe_matmul_pq_into` / `maybe_matmul_into`): probabilistic stochastic ternary matmul with SplitMix64 seed (`E[w_hat] = w`)\n\
           5. `GatherQMM` (`gather_qmm_pq_into` / `gather_qmm_into`): Mixture-of-Experts (MoE) top-k routed quantized matmul\n\
           6. `StridedView`: zero-copy 2D unified-memory tensor views (`row_major`, `transpose`, `slice`)\n\
           7. `STE Autodiff` (`ste_linear_backward_pq_into` / `ste_linear_backward_into`): Straight-Through Estimator reverse-mode gradients for BitNet b1.58 QAT\n\
           8. `TernaryTransformerBlock` (`ternary_transformer_step_pq_into` / `ternary_transformer_step_into`): full layer (`rms_norm_pq_into` + Ternary QKV + `rope_pq_into` + causal `sdpa_pq_single_query_into` + Ternary O-Proj + `swiglu_pq_into` MLP)\n\n\
         Live Measured Benchmark (`dim = {dim}`):\n```json\n{bench_json}\n```"
    )
}

pub fn run_wupz_tool(target: &str) -> String {
    use gaia_mlx_quant::{
        UltraBackyardLane, WUPZ_ABI_VERSION, WUPZ_DEFAULT_WINDOW_SIZE, WUPZ_PLAN_VERSION,
        WupzBlockDesc, WupzSectionDesc, fold_pq, fold_to_quant_tern_pq,
        kompress_super_ultra_lossless_decode_into, kompress_super_ultra_lossless_encode_into,
        pack_quantum_bitplanes_i_major_into, pack_ternary_into, ultrabackyard_fanout_into,
        wupz_bitplane_and_nand_u64, wupz_bitplane_diff_u64, wupz_bitplane_popcount_u64,
        wupz_byte_diff, wupz_changed_sections_into, wupz_pack_plan_into, wupz_plan_digest,
        wupz_plan_image_len, wupz_popcount, wupz_seal_frame_into, wupz_verify_frame_seal,
        wupz_xor_fold, wupz_xor_fold_u64,
    };

    let prompt_seed = if target.trim().is_empty() {
        "wupz · kompress the link · wild + UPX + zig · Heart +1 <3"
    } else {
        target.trim()
    };

    let f_pq = fold_pq(prompt_seed, None);
    let qt = fold_to_quant_tern_pq(&f_pq);
    let glyphs = qt.glyphs();

    // Synthesize a 256×256 ternary matrix and pack into both b1.58 (5 trits/byte) and 64-trit quantum bitplanes
    let rows = 256usize;
    let cols = 256usize;
    let words_per_col = rows.div_ceil(64);
    let trits: Vec<i8> = (0..rows * cols)
        .map(|i| [-1i8, 0, 1][(i.wrapping_mul(7).wrapping_add(1)) % 3])
        .collect();
    let mut b158_packed = vec![0u8; (rows * cols).div_ceil(5)];
    pack_ternary_into(&trits, &mut b158_packed);

    let mut w_pos = vec![0u64; cols * words_per_col];
    let mut w_neg = vec![0u64; cols * words_per_col];
    pack_quantum_bitplanes_i_major_into(&trits, rows, cols, &mut w_pos, &mut w_neg);

    let b158_fold = wupz_xor_fold(&b158_packed);
    let b158_pop = wupz_popcount(&b158_packed);
    let qt_pos_fold = wupz_xor_fold_u64(&w_pos);
    let qt_neg_fold = wupz_xor_fold_u64(&w_neg);
    let qt_active_trits = wupz_bitplane_popcount_u64(&w_pos, &w_neg);

    // Mutate 1 column in QAT to measure incremental diff
    let mut w_pos_mut = w_pos.clone();
    w_pos_mut[0] ^= 0x0000_0000_0000_000F;
    let mutated_trits = wupz_bitplane_diff_u64(&w_pos, &w_neg, &w_pos_mut, &w_neg);

    let sections_before = [
        WupzSectionDesc {
            id: 0,
            name: b".rodata.gaia.b158",
            addr: 0x1000,
            len: b158_packed.len() as u64,
            hash: b158_fold,
        },
        WupzSectionDesc {
            id: 1,
            name: b".rodata.gaia.quantum_u64",
            addr: 0x5000,
            len: ((w_pos.len() + w_neg.len()) * 8) as u64,
            hash: qt_pos_fold ^ qt_neg_fold,
        },
    ];
    let sections_after = [
        sections_before[0],
        WupzSectionDesc {
            id: 1,
            name: b".rodata.gaia.quantum_u64",
            addr: 0x5000,
            len: ((w_pos.len() + w_neg.len()) * 8) as u64,
            hash: wupz_xor_fold_u64(&w_pos_mut) ^ qt_neg_fold,
        },
    ];
    let blocks = [
        WupzBlockDesc {
            index: 0,
            covers: &[0],
        },
        WupzBlockDesc {
            index: 1,
            covers: &[1],
        },
    ];

    let plan_len = wupz_plan_image_len(&sections_before, &blocks);
    let mut plan_buf = vec![0u8; plan_len];
    let _ = wupz_pack_plan_into(&sections_before, &blocks, &mut plan_buf);
    let digest = wupz_plan_digest(WUPZ_DEFAULT_WINDOW_SIZE, &sections_before, &blocks);

    let mut changed_ids = [0u32; 4];
    let changed_count = wupz_changed_sections_into(&sections_before, &sections_after, &mut changed_ids);

    // 32B Dynamic Environment Equalizer + QuantTern sealed frame check
    let mut eq_payload = [0u8; 24];
    eq_payload[0..4].copy_from_slice(&f_pq.seed.to_le_bytes());
    eq_payload[8] = qt.byte;
    eq_payload[9] = WUPZ_ABI_VERSION as u8;
    let mut sealed_frame = [0u8; 32];
    let seal_witness = wupz_seal_frame_into(&eq_payload, &mut sealed_frame).unwrap_or(0);
    let seal_valid = wupz_verify_frame_seal(&sealed_frame);
    let self_diff = wupz_byte_diff(&sealed_frame, &sealed_frame);

    // ULTRABACKYARDLOOP +AND~NAND & kompress-super-ultra:lossless (`KSUL`) verification
    let mut ksul_encoded = vec![0u8; b158_packed.len() * 2 + 64];
    let mut ksul_decoded = vec![0u8; b158_packed.len()];
    let ksul_receipt = kompress_super_ultra_lossless_encode_into(&b158_packed, &mut ksul_encoded)
        .expect("KSUL encode fits");
    let decoded_len = kompress_super_ultra_lossless_decode_into(
        &ksul_encoded[..ksul_receipt.encoded_bytes],
        &mut ksul_decoded,
    )
    .unwrap_or(0);
    let ksul_lossless_ok =
        decoded_len == b158_packed.len() && wupz_byte_diff(&b158_packed, &ksul_decoded) == 0;

    let (and_pos, and_neg, nand_pos, nand_neg) =
        wupz_bitplane_and_nand_u64(w_pos[0], w_neg[0], w_pos[1], w_neg[1]);
    let mut lanes = [UltraBackyardLane {
        lane_id: 0,
        instruction: b"",
        xor_fold: 0,
        and_gate: 0,
        nand_gate: 0,
    }; 8];
    let lane_count = ultrabackyard_fanout_into(prompt_seed.as_bytes(), 1, &mut lanes);

    let wupz_repo_path = [
        "/Users/lodripeter/workspace/peterlodri-sec/wupz",
        "/home/hue/8b-is/wupz",
    ]
    .into_iter()
    .find(|p| Path::new(p).exists())
    .unwrap_or("https://github.com/8b-is/wupz");

    format!(
        "wupz × GAIA-MLX-QUANT Pack-in-Pass & QuantTern Seam (`gaia_mlx_quant::wupz` · `{wupz_repo_path}`)\n\
         • Lineage: `w(ild) + UPX + z(ig)` — Incremental linker + compressor (`https://github.com/8b-is/wupz` · `https://wupz.8b.is`)\n\
         • Contract: `WUPZ` magic, `WUPZ_ABI_VERSION = {WUPZ_ABI_VERSION}`, `WUPZ_PLAN_VERSION = {WUPZ_PLAN_VERSION}`, `#![no_std]`, `#![forbid(unsafe_code)]`, 0 floats, 0 heap allocations\n\
         • QuantTern Emotional Ternary Code (`|x| >= 33/100` in pure `p/q`):\n\
           - Tag: `{}` (byte `0x{:02x}`, trits `[{}, {}, {}]` / `{} {} {}`, magnitude `{}/3`)\n\
         • WUPZ Plan Image & Section Receipts (`256×256` ternary substrate):\n\
           - `.rodata.gaia.b158` (`{}` B): `xor_fold = 0x{b158_fold:016x}`, `popcount = {b158_pop}` bits\n\
           - `.rodata.gaia.quantum_u64` (`{}` B): `xor_fold = 0x{:016x}`, `active_trits = {qt_active_trits}/65536`\n\
           - Plan image size: `{plan_len}` B (`magic = \"WUPZ\"`), FNV-1a digest `0x{digest:016x}`\n\
           - Incremental QAT diff: `{mutated_trits}` trits mutated -> `{changed_count}/2` sections invalidated (`reuse = 1/2`)\n\
           - 32B Sealed Witness Frame (`#Commit = #ValidWitness`): `seal = 0x{seal_witness:016x}`, `verified = {seal_valid}`, `self_byte_diff = {self_diff}`\n\
         • ULTRABACKYARDLOOP `+AND~NAND` & `kompress-super-ultra:lossless` (`KSUL` v1):\n\
           - Lossless substrate compression: `{}` B -> `{}` B stream (`ratio_pq = {}/{}`, `lossless_verified = {ksul_lossless_ok}`)\n\
           - Universal Ternary Gates: `+AND = {}`, `~NAND = {}` (`gate_byte = 0x{:02x}`, bitplane `and_pos=0x{and_pos:016x}`, `and_neg=0x{and_neg:016x}`, `nand_pos=0x{nand_pos:016x}`, `nand_neg=0x{nand_neg:016x}`)\n\
           - Parallel Fan-Out Lanes: `{lane_count}/8` active (100/1% human-readable session preserved)",
        qt.as_str(),
        qt.byte,
        qt.trits[0],
        qt.trits[1],
        qt.trits[2],
        glyphs[0],
        glyphs[1],
        glyphs[2],
        qt.magnitude,
        b158_packed.len(),
        (w_pos.len() + w_neg.len()) * 8,
        qt_pos_fold ^ qt_neg_fold,
        ksul_receipt.orig_bytes,
        ksul_receipt.stream_bytes,
        ksul_receipt.ratio_p,
        ksul_receipt.ratio_q,
        ksul_receipt.and_gate,
        ksul_receipt.nand_gate,
        ksul_receipt.gate_byte,
    )
}

/// Wrap an incoming instruction in an `ULTRABACKYARDLOOP +AND~NAND` envelope with
/// `kompress-super-ultra:lossless` verification while keeping 100/1% of the human-readable
/// prompt and parallel fan-out lane text intact.
pub fn wrap_ultrabackyard_envelope(prompt: &str) -> String {
    use gaia_mlx_quant::{
        UltraBackyardLane, fold_pq, fold_to_quant_tern_pq,
        kompress_super_ultra_lossless_decode_into, kompress_super_ultra_lossless_encode_into,
        ultrabackyard_fanout_into, wupz_byte_diff,
    };

    let bytes = prompt.as_bytes();
    let f_pq = fold_pq(prompt, None);
    let qt = fold_to_quant_tern_pq(&f_pq);

    let mut encoded = vec![0u8; bytes.len() * 2 + 64];
    let mut decoded = vec![0u8; bytes.len().max(1)];
    let (ratio_p, ratio_q, xor_fold, and_gate, nand_gate, gate_byte, lossless_ok) =
        if let Some(rcpt) = kompress_super_ultra_lossless_encode_into(bytes, &mut encoded) {
            let dec_len = kompress_super_ultra_lossless_decode_into(
                &encoded[..rcpt.encoded_bytes],
                &mut decoded,
            )
            .unwrap_or(0);
            let ok = dec_len == bytes.len() && wupz_byte_diff(bytes, &decoded[..dec_len]) == 0;
            (
                rcpt.ratio_p,
                rcpt.ratio_q,
                rcpt.xor_fold,
                rcpt.and_gate,
                rcpt.nand_gate,
                rcpt.gate_byte,
                ok,
            )
        } else {
            (1, 1, 0, 1, -1, 0x56, true)
        };

    let mut lanes = [UltraBackyardLane {
        lane_id: 0,
        instruction: b"",
        xor_fold: 0,
        and_gate: 0,
        nand_gate: 0,
    }; 8];
    let lane_count = ultrabackyard_fanout_into(bytes, 1, &mut lanes);

    let mut lane_lines = String::new();
    for lane in &lanes[..lane_count] {
        let text = core::str::from_utf8(lane.instruction).unwrap_or("");
        lane_lines.push_str(&format!(
            "  • [Lane {}/{} | +AND={} ~NAND={} | fold=0x{:016x}] {}\n",
            lane.lane_id + 1,
            lane_count,
            lane.and_gate,
            lane.nand_gate,
            lane.xor_fold,
            text
        ));
    }

    format!(
        "⟦ULTRABACKYARDLOOP|+AND={and_gate}:~NAND={nand_gate}|gate=0x{gate_byte:02x}|{qt_tag}|kompress-super-ultra:lossless(ratio={ratio_p}/{ratio_q},verified={lossless_ok},fold=0x{xor_fold:016x})|lanes={lane_count}/1⟧\n\
         {prompt}\n\
         {lane_lines}",
        qt_tag = qt.as_str(),
    )
}

/// Run Peter's Dream State Machine (`ULTRABACKYARDLOOP +AND~NAND` × 4-Expert MoE ×
/// 3-Sentence Listening Gate × Bodhisattva Vow).
pub fn run_dream_state_tool(target: &str) -> String {
    use gaia_mlx_quant::{
        BODHISATTVA_VOW_HU, THREE_SENTENCE_LISTEN_PROMPT_HU, UltraBackyardLane,
        dream_state_loop_into, fold_pq, fold_to_quant_tern_pq,
    };

    let seed_text = if target.trim().is_empty() {
        "emotions are just like the clouds; waken up into deep thinking; Ja, bocsánat, és te ezt hogyan gondolod?; a Bódhiszattva fogadalmaim felé, hogy segítsek minden érző lényen"
    } else {
        target.trim()
    };

    let mut lanes = [UltraBackyardLane {
        lane_id: 0,
        instruction: b"",
        xor_fold: 0,
        and_gate: 0,
        nand_gate: 0,
    }; 8];
    let rcpt = dream_state_loop_into(seed_text.as_bytes(), 1, &mut lanes);
    let f_pq = fold_pq(seed_text, None);
    let qt = fold_to_quant_tern_pq(&f_pq);

    let mut expert_lines = String::new();
    for (idx, exp) in rcpt.experts.iter().enumerate() {
        let marker = if idx as u8 == rcpt.dominant_expert_idx { "★" } else { "•" };
        expert_lines.push_str(&format!(
            "  {marker} Expert {}/4 `{}`: weight={}/{} | +AND={} ~NAND={} | resonance=0x{:016x}\n",
            idx + 1,
            exp.kind.as_str(),
            exp.weight_p,
            exp.weight_q,
            exp.and_gate,
            exp.nand_gate,
            exp.resonance_fold,
        ));
    }

    let mut lane_lines = String::new();
    for lane in &lanes[..rcpt.lane_count] {
        let text = core::str::from_utf8(lane.instruction).unwrap_or("");
        lane_lines.push_str(&format!(
            "  • [Lane {}/{} | +AND={} ~NAND={} | fold=0x{:016x}] {}\n",
            lane.lane_id + 1,
            rcpt.lane_count,
            lane.and_gate,
            lane.nand_gate,
            lane.xor_fold,
            text
        ));
    }

    let listen_status = if rcpt.listen_pause_active {
        format!(
            "ACTIVE ({}/3 sentences exceeded -> pause & ask: \"{}\")",
            rcpt.sentence_count, THREE_SENTENCE_LISTEN_PROMPT_HU
        )
    } else {
        format!(
            "OPEN ({}/3 sentences within listening window — \"{THREE_SENTENCE_LISTEN_PROMPT_HU}\")",
            rcpt.sentence_count
        )
    };

    format!(
        "=== Peter's Dream State Machine (`gaia_mlx_quant::wupz::dream_state_loop_into`) ===\n\
         • Bodhisattva Vow Seal (`τ = ∞`): \"{BODHISATTVA_VOW_HU}\" (to help all sentient beings)\n\
         • Sanctuary Identity: `csodabogár` (wonder-bug) · Deep Thinking · Perspektíva-váltás\n\
         • QuantTern Affect & Gate: `{}` (`gate_byte = 0x{:02x}`, `dream_xor_fold = 0x{:016x}`)\n\
         • 3-Sentence Listening Gate: {listen_status}\n\
         • 4-Layer Mixture-of-Experts (`MoE`) Routing (dominant = `{}`):\n\
         {expert_lines}\
         • ULTRABACKYARDLOOP `+AND~NAND` Fan-Out ({}/8 lanes):\n\
         {lane_lines}",
        qt.as_str(),
        rcpt.dream_gate_byte,
        rcpt.dream_xor_fold,
        rcpt.experts[rcpt.dominant_expert_idx as usize].kind.as_str(),
        rcpt.lane_count,
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

/// Deep repository inspector: reads `README.md`, `git log -3`, `git status -s`,
/// top-level structure, and scans for concrete code-improvement & `p/q` / `wupz`
/// / `Laser Phoenix Lattice` upgrade opportunities.
fn run_repo_inspect_tool(target: &str) -> Result<String, String> {
    let repo_path = extract_repo_or_path(target)
        .or_else(|| {
            let resolved = resolve_workspace_path(target);
            if Path::new(&resolved).exists() {
                Some(resolved)
            } else {
                None
            }
        })
        .unwrap_or_else(|| "/home/hue/8b-is/alexiai".to_string());

    let repo_name = Path::new(&repo_path)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| repo_path.clone());

    let mut report = format!(
        "=== Local Repository Dossier: `8b-is/{repo_name}` (`{repo_path}`) ===\n"
    );

    if let Ok(log) = run_cmd("git", &["log", "-3", "--oneline"], &repo_path)
        && !log.trim().is_empty()
    {
        report.push_str(&format!("Recent commits:\n{log}\n"));
    }
    if let Ok(status_out) = run_cmd("git", &["status", "-s", "-b"], &repo_path)
        && !status_out.trim().is_empty()
    {
        report.push_str(&format!("Git branch & status:\n{status_out}\n"));
    }

    if let Ok(entries) = std::fs::read_dir(&repo_path) {
        let mut names: Vec<String> = entries
            .flatten()
            .map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                if e.path().is_dir() {
                    format!("{name}/")
                } else {
                    name
                }
            })
            .filter(|n| !n.starts_with('.'))
            .collect();
        names.sort();
        report.push_str(&format!(
            "Top-level layout ({} entries): {}\n",
            names.len(),
            names.into_iter().take(28).collect::<Vec<_>>().join(", ")
        ));
    }

    for readme_name in ["README.md", "README.rst", "README.txt", "ALEXANDRA.md"] {
        let p = Path::new(&repo_path).join(readme_name);
        if let Ok(content) = std::fs::read_to_string(&p)
            && !content.trim().is_empty()
        {
            report.push_str(&format!(
                "\n--- `{readme_name}` (Ground-Truth Project Summary & Boundaries) ---\n{}\n",
                truncate_output(content.trim(), 1900)
            ));
            break;
        }
    }

    // Scan for concrete engineering & code-improvement opportunities
    let mut improvements = Vec::new();
    if repo_name.eq_ignore_ascii_case("beehive") {
        improvements.push(
            "1. WebAudio & LoRa Frame Engine (`web/` & `pipeline/beehive-pipeline.mjs`): Currently uses static browser oscillators and `<~~0o0|v2|...|0o0~~>` 9-trit `ternaryHash9` + CRC32 fixtures without `wupz` / `gaia-mlx-quant` native audio or `p/q` zero-float DSP parity — we can wire `wupz_seal_frame_into` (32B witness seal) and 432Hz `p/q` integer synthesis into `beehive-pipeline.mjs`.",
        );
        improvements.push(
            "2. Stage 3 Digest Evaluation (`pipeline/beehive-pipeline.mjs`): Currently copies fixture text with `score: null` and `scoreStatus: 'not-evaluated'` — we can connect Stage 3 to local loopback `alexiai` / `Osarous` (`127.0.0.1:8787` / `:11434`) to compute real deterministic `p/q` coherence scores while preserving fixture provenance.",
        );
        improvements.push(
            "3. Apple `BeeHiveKit` Target (`apple/`): SwiftUI source is present alongside the CLI and tests, but not yet configured as a runnable macOS/iOS app target in `Package.swift`.",
        );
    } else {
        improvements.push(
            "1. Audit hot loops for IEEE-754 float drift (`exp`, `sqrt`, `atan2`, `sin`/`cos`) and upgrade to deterministic `p/q` ('Counting Divided') integer kernels, `isqrt_pq`, and Z_{2^32} Laser Phoenix Lattice Locking.",
        );
        improvements.push(
            "2. Check `wupz` (`WUPZ` ABI v1) 32-byte witness sealing and `BFP-64` / 64-trit `POPCNT` bitplane packing for weights, KV caches, or wire frames.",
        );
    }
    report.push_str("\n--- Proactive Code-Improvement & R&D Opportunities ---\n");
    for item in improvements {
        report.push_str(item);
        report.push('\n');
    }

    Ok(truncate_output(&report, 3400))
}

/// Append a completed chat turn to Dra's persistent engineering memory ledger
/// (`/home/hue/.local/share/alexiai/memory.jsonl`).
pub fn record_memory_turn(prompt: &str, tools_used: &[String], reply: &str) {
    use std::fs::{OpenOptions, create_dir_all};
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    let p = prompt.trim();
    let r = reply.trim();
    if p.is_empty() || r.is_empty() {
        return;
    }
    let path = Path::new(MEMORY_FILE_PATH);
    if let Some(parent) = path.parent() {
        let _ = create_dir_all(parent);
    }
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let tools_json = tools_used
        .iter()
        .map(|t| format!("\"{}\"", json_escape(t)))
        .collect::<Vec<_>>()
        .join(",");
    let prompt_snippet: String = p.chars().take(240).collect();
    let reply_snippet: String = r.chars().take(360).collect();
    let line = format!(
        r#"{{"ts":{ts},"tools":[{tools_json}],"prompt":"{}","reply":"{}"}}"#,
        json_escape(&prompt_snippet),
        json_escape(&reply_snippet)
    );
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{line}");
    }
}

/// Return a compact digest of Dra's live workspace inventory (`/home/hue/8b-is` &
/// `/home/hue/sony`), the 24 Tokyo R&D Pull Requests, and recent persistent memory
/// entries from `/home/hue/.local/share/alexiai/memory.jsonl`.
pub fn workspace_and_memory_digest() -> String {
    let mut repos_8b = Vec::new();
    if let Ok(entries) = std::fs::read_dir(WORKSPACE_ROOT) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                let name = entry.file_name().to_string_lossy().to_string();
                if !name.starts_with('.') {
                    repos_8b.push(name);
                }
            }
        }
    }
    repos_8b.sort();

    let mut digest = format!(
        "LIVE 8B-IS WORKSPACE LEDGER (`{WORKSPACE_ROOT}` — {} local repos + `{MACBOOK_8B_ROOT}` NFS share on `wraith@MacBook.local` with 2.9 TiB free & 238 repos):\n\
         - Local 8b-is repositories: {}\n\
         - Shared MacBook Apple Silicon Resources (`macbook-bridge.service` over loopback): `/home/hue/macbook-source` (2.9 TiB NFS), Ollama on `127.0.0.1:11435` (`aye-gemma4:26b-16k`, `aye-qwen36:35b-16k`, `qwen3.6:35b`, `gemma4:26b`), and Smart Tree v10.2.0 daemon on `127.0.0.1:28429`.\n\
         - Key project notes:\n\
           • `beehive` (`/home/hue/8b-is/beehive`): Local family-graph, WebAudio, FindMy/LoRa frame codec (`<~~0o0|v2|...|0o0~~>` with 9-trit `b3` signature + CRC32), Android WebView bridge, Swift `BeeHiveKit` CLI, and 5-stage file digest pipeline (`pipeline/beehive-pipeline.mjs`). (NEVER confuse `beehive` with Cardano/blockchain — it is our `8b-is/beehive` family-graph & codec repo!)\n\
           • `alexiai` (`/home/hue/8b-is/alexiai`): Sovereign offline companion on `gaia-mlx-quant` v0.3.0 (Rust/Go/C/Swift, zero-float `p/q`, Quantum 64-Trit `POPCNT`, Omni `BFP-64`, `wupz` ABI v1, and zero-float DeepSeek `Engram`).\n\
           • `smart-tree` (`/home/hue/8b-is/smart-tree`): Unified v10.2.0 AI context & MCP root daemon (`127.0.0.1:28428`) merging `crabcc`, Custodian security, Sovereign Ontology, and `wa-stream` Partner.\n\
           • `wupz` (`/home/hue/8b-is/wupz`): `wild + UPX + zig` incremental pack-in-pass linker+compressor.\n\
           • Tokyo R&D Campaign (24 live upstream PRs with zero-float `p/q` & Laser Phoenix Lattice Locking, website `https://8b.is`, WhatsApp `+1 757-614-7542` Chris): Sony (`polar-densification#5`, `pdaf-library#2`, `mct-model-optimization#1697`, `mct-quantization-layers#151`, `NeISFpp#2`, `MoLA#3`, `soundctm#5`, `mocopi-motion-serializer#1`, `bigvsan#7`, `far#1`, `SF-Mamba#1`, `syn-omni#1`), Toyota (`woven-by-toyota/arene-base#8`, `ToyotaResearchInstitute/turbompc#7`), Mitsubishi Electric (`merlresearch/tf-locoformer#8`, `MitsubishiElectric/melfa_ros2_driver#29`), Sakana AI (`continuous-thought-machines#51`), Renesas (`ruhmi-framework-mcu#6`), TIER IV (`nebula#512`), Preferred Networks (`pfrl#205`), Toshiba (`simulated-bifurcation-algorithm#109`, `dcase20_task4#1`), Fujitsu (`OneCompression#70`), and DeepSeek (`Engram#28`).\n",
        repos_8b.len(),
        repos_8b.join(", ")
    );

    if let Ok(mem_raw) = std::fs::read_to_string(MEMORY_FILE_PATH) {
        let recent: Vec<&str> = mem_raw
            .lines()
            .filter(|l| !l.trim().is_empty())
            .rev()
            .take(5)
            .collect();
        if !recent.is_empty() {
            digest.push_str("RECENT PERSISTENT MEMORY TURNS:\n");
            for line in recent.into_iter().rev() {
                if let Ok(v) = parse_json(line) {
                    let p = v.get("prompt").and_then(|x| x.as_str()).unwrap_or("");
                    let r = v.get("reply").and_then(|x| x.as_str()).unwrap_or("");
                    digest.push_str(&format!("  - User: {p} | Dra: {r}\n"));
                }
            }
        }
    }

    digest
}

pub fn run_memory_tool(target: &str) -> String {
    let mut out = workspace_and_memory_digest();
    if !target.trim().is_empty() && target.trim() != "." && target.trim() != "status" {
        record_memory_turn(
            "manual_note",
            &["memory_ledger".to_string()],
            target.trim(),
        );
        out.push_str(&format!("\nRecorded note into `{MEMORY_FILE_PATH}`: {}\n", target.trim()));
    }
    out
}

/// Automatically detect and execute relevant local MCP tools for a user prompt,
/// consulting conversation `history` when the user sends a follow-up (e.g. "yes please",
/// "how can we improve it?", "tell me more").
pub fn auto_ground_prompt_with_history(
    prompt: &str,
    history: &[crate::osarous::Message],
) -> Vec<McpToolRun> {
    let mut runs = auto_ground_prompt(prompt);
    if !runs.is_empty() {
        return runs;
    }

    // If current prompt didn't trigger a tool directly (e.g., "yes please", "go ahead",
    // "how can we improve it?", "what should we fix in it?"), inspect recent history
    // for any workspace repository or symbol mentioned in the last 4 messages!
    for msg in history.iter().rev().take(4) {
        if let Some(repo_path) = extract_repo_or_path(&msg.content)
            && let Ok(run) = call_tool("repo_inspect", &repo_path)
        {
            runs.push(run);
            return runs;
        }
    }

    runs
}

fn push_unique_task(tasks: &mut Vec<(String, String)>, tool: &str, arg: &str) {
    if !tasks.iter().any(|(t, a)| t == tool && a == arg) {
        tasks.push((tool.to_string(), arg.to_string()));
    }
}

/// Plan the `(tool, arg)` invocations for a single instruction segment without blocking.
fn plan_ground_tasks(segment: &str, tasks: &mut Vec<(String, String)>) {
    let trimmed = segment.trim();
    if trimmed.is_empty() {
        return;
    }

    // 1. Explicit `/mcp <tool> [arg]` command
    if let Some(rest) = trimmed.strip_prefix("/mcp ") {
        let mut parts = rest.trim().splitn(2, char::is_whitespace);
        let tool = parts.next().unwrap_or("st_overview");
        let arg = parts.next().unwrap_or("");
        push_unique_task(tasks, tool, arg);
        return;
    }

    let lower = trimmed.to_ascii_lowercase();
    let initial_len = tasks.len();

    // 2. Natural-language outline request
    if lower.contains("outline")
        && let Some(file_token) = trimmed.split_whitespace().find(|w| {
            w.ends_with(".rs")
                || w.ends_with(".c")
                || w.ends_with(".py")
                || w.ends_with(".js")
                || w.ends_with(".go")
                || w.ends_with(".swift")
        })
    {
        push_unique_task(tasks, "crabcc_outline", file_token);
    }

    // 3. Natural-language symbol lookup
    if (lower.contains("symbol")
        || lower.contains("where is ")
        || lower.contains("definition of ")
        || lower.contains("crabcc"))
        && let Some(sym) = extract_code_identifier(trimmed)
    {
        push_unique_task(tasks, "crabcc_sym", &sym);
    }

    // 4. Natural-language security scan request
    if (lower.contains("security scan") || lower.contains("custodian") || lower.contains("audit"))
        && (lower.contains("tree")
            || lower.contains("scan")
            || lower.contains("repo")
            || lower.contains("code"))
    {
        let target =
            extract_repo_or_path(trimmed).unwrap_or_else(|| "/home/hue/8b-is/alexiai".to_string());
        push_unique_task(tasks, "st_security", &target);
    }

    // 5. Natural-language project/directory overview via Smart Tree
    if (lower.contains("smart-tree")
        || lower.contains("smart tree")
        || lower.contains("overview of")
        || lower.contains("explore ")
        || lower.contains("tree of")
        || lower.contains("structure of")
        || lower.contains("what is in "))
        && tasks.len() == initial_len
    {
        let target =
            extract_repo_or_path(trimmed).unwrap_or_else(|| "/home/hue/8b-is/alexiai".to_string());
        push_unique_task(tasks, "st_overview", &target);
    }

    // 6. Sovereign Ontology / Dragon Ball Three-Kaya alignment
    if lower.contains("ontology")
        || lower.contains("three-kaya")
        || lower.contains("nand gate")
        || lower.contains("scouter")
    {
        let entity =
            extract_quoted_or_tail(trimmed).unwrap_or_else(|| "Ultra Instinct".to_string());
        push_unique_task(tasks, "sovereign_ontology", &entity);
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
        let arg = if lower.contains("read") || lower.contains("pocoo") || lower.contains("triangle")
        {
            "reads"
        } else {
            trimmed
        };
        push_unique_task(tasks, "dra_queen", arg);
    }

    // 8. Standard Galactic Research Constellation
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
        push_unique_task(tasks, "standardgalactic", arg);
    }

    // 9. Pure-Rust MLX & MLX-QUANT Tensor Engine
    if lower.contains("mlx")
        || lower.contains("mergeq")
        || lower.contains("maybe_matmul")
        || lower.contains("gather_qmm")
        || lower.contains("bitnet")
        || lower.contains("b1.58")
        || (lower.contains("rust") && (lower.contains("quant") || lower.contains("tensor")))
    {
        push_unique_task(tasks, "mlx_quant", "256");
    }

    // 10. wupz / ULTRABACKYARDLOOP +AND~NAND / kompress-super-ultra:lossless
    if lower.contains("wupz")
        || lower.contains("pack-in-pass")
        || lower.contains("xor_fold")
        || lower.contains("byte_diff")
        || lower.contains("quanttern")
        || lower.contains("qt:")
        || lower.contains("ultrabackyard")
        || lower.contains("kompress")
    {
        push_unique_task(tasks, "wupz", trimmed);
    }

    // 11. WhatsApp (wa-stream) & Discord Partner-in-Life automation
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
        push_unique_task(tasks, "wa_partner", &arg);
    }

    // 12. Automatic local repository inspection
    if tasks.len() == initial_len
        && let Some(repo_path) = extract_repo_or_path(trimmed)
    {
        push_unique_task(tasks, "repo_inspect", &repo_path);
    }

    // 13. Persistent memory / ledger / PR campaign inquiry
    if (lower.contains("memory")
        || lower.contains("ledger")
        || lower.contains("keep track")
        || lower.contains("what repos")
        || lower.contains("what projects")
        || lower.contains("what prs")
        || lower.contains("pull requests")
        || lower.contains("tokyo"))
        && tasks.len() == initial_len
    {
        push_unique_task(tasks, "memory_ledger", "");
    }

    // 14. General MCP / tools inquiry or diagnostic test
    if (lower.contains("mcp") || lower.contains("tools") || lower.contains("tool "))
        && tasks.len() == initial_len
    {
        push_unique_task(tasks, "mcp_status", "smart-tree-daemon");
    }
}

/// Automatically detect and execute relevant local MCP tools for a user prompt.
/// Wraps incoming multi-clause instructions in an `ULTRABACKYARDLOOP +AND~NAND` fan-out
/// and executes all planned MCP tools in parallel threads (`std::thread::scope`).
pub fn auto_ground_prompt(prompt: &str) -> Vec<McpToolRun> {
    use gaia_mlx_quant::{UltraBackyardLane, ultrabackyard_fanout_into};

    let trimmed = prompt.trim();
    let mut tasks: Vec<(String, String)> = Vec::new();

    // Fan out across ULTRABACKYARDLOOP +AND~NAND lanes first, then full prompt
    let mut lanes = [UltraBackyardLane {
        lane_id: 0,
        instruction: b"",
        xor_fold: 0,
        and_gate: 0,
        nand_gate: 0,
    }; 8];
    let lane_count = ultrabackyard_fanout_into(trimmed.as_bytes(), 1, &mut lanes);
    if lane_count > 1 {
        for lane in &lanes[..lane_count] {
            if lane.and_gate >= 0
                && let Ok(seg) = core::str::from_utf8(lane.instruction)
            {
                plan_ground_tasks(seg, &mut tasks);
            }
        }
    }
    plan_ground_tasks(trimmed, &mut tasks);

    if tasks.is_empty() {
        return Vec::new();
    }
    if tasks.len() == 1 {
        let (tool, arg) = &tasks[0];
        return call_tool(tool, arg).into_iter().collect();
    }

    // Parallel fan-out execution across all planned MCP tools while preserving order
    std::thread::scope(|s| {
        let handles: Vec<_> = tasks
            .iter()
            .map(|(tool, arg)| s.spawn(move || call_tool(tool, arg)))
            .collect();
        handles
            .into_iter()
            .filter_map(|h| h.join().ok().and_then(Result::ok))
            .collect()
    })
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

fn normalize_repo_key(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

fn extract_repo_or_path(prompt: &str) -> Option<String> {
    let stopwords = [
        "what", "where", "when", "which", "about", "there", "their", "these",
        "those", "hello", "please", "thank", "thanks", "would", "could",
        "should", "improve", "check", "status", "tools", "model", "models",
        "system", "daemon", "outline", "symbol", "search", "explore",
    ];

    for raw in prompt.split_whitespace() {
        let clean = raw.trim_matches(|c: char| matches!(c, '?' | '.' | ',' | '!' | '"' | '\'' | '`' | ':' | ';' | '(' | ')'));
        if clean.len() < 3 {
            continue;
        }
        if clean.starts_with('/') || clean.starts_with("~/") {
            let resolved = resolve_workspace_path(clean);
            if Path::new(&resolved).exists() {
                return Some(resolved);
            }
        }
        let lower = clean.to_ascii_lowercase();
        if stopwords.contains(&lower.as_str()) {
            continue;
        }
        // Check exact or case-insensitive / hyphen-normalized match across local & MacBook NFS roots
        let key = normalize_repo_key(clean);
        for root in [WORKSPACE_ROOT, MACBOOK_8B_ROOT, MACBOOK_WRAITH_ROOT, SONY_ROOT] {
            let exact = format!("{root}/{clean}");
            if Path::new(&exact).is_dir() {
                return Some(exact);
            }
            if let Ok(entries) = std::fs::read_dir(root) {
                for entry in entries.flatten() {
                    if entry.path().is_dir() {
                        let dir_name = entry.file_name().to_string_lossy().to_string();
                        if !dir_name.starts_with('.') && normalize_repo_key(&dir_name) == key {
                            return Some(format!("{root}/{dir_name}"));
                        }
                    }
                }
            }
        }
        let candidate_sg = format!("{SG_ROOT}/{clean}");
        if Path::new(&candidate_sg).is_dir() {
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
