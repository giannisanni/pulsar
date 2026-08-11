//! pulsar-serve: OpenAI-compatible chat completions over the pulsar
//! engine.
//!
//!   pulsar-serve -m model.gguf [--port 11435] [--host 127.0.0.1] [--ctx 8192]
//!
//! Endpoints: GET /v1/models, POST /v1/chat/completions (stream and
//! non-stream). One engine, one request at a time, prefill from position
//! zero per request - the ollama-style local single-user shape. The KV
//! cache is overwritten progressively, so no reset step is needed.
//! ponytail: hand-rolled HTTP/1.1 on TcpListener; an async framework
//! buys nothing for a sequential localhost server.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("pulsar-serve requires Linux + CUDA");
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
fn main() {
    if let Err(e) = run() {
        eprintln!("pulsar-serve: {e}");
        std::process::exit(1);
    }
}

/// Self-contained chat webui (vanilla JS, no build step). Served at `/`
/// alongside the API; same origin as `/v1/chat/completions`, so no CORS.
#[cfg(target_os = "linux")]
const WEBUI_HTML: &str = include_str!("../webui/index.html");

/// Inline SVG favicon (the pulsar dot) — keeps the browser console clean
/// instead of 404-ing on /favicon.ico.
#[cfg(target_os = "linux")]
const FAVICON_SVG: &str = include_str!("../webui/favicon.svg");

/// MCP client hub (rmcp). Gated to linux like the rest of the server; the
/// feature is opt-in via --webui-mcp-proxy, which enables the /mcp/* routes,
/// tool injection, and the webui MCP tab. Without the flag this module is
/// unused and behavior is unchanged.
#[cfg(target_os = "linux")]
mod mcp;

/// Multi-format tool-call parsers (generic JSON, Hy3 opensource, DeepSeek DSML).
/// Always compiled so unit tests run on any OS.
mod tool_calls;

/// Sanity bound only - NOT a capability limit. The reachable ceiling is
/// the checkpoint's own context_length (what the model was trained for)
/// narrowed by ctx_fit (what this machine's VRAM holds). A fixed number
/// here would cap a box that can do more: GLM-5.2 declares 1,048,576, and
/// on hardware with the VRAM for it that is a legitimate request.
const CTX_SANITY_MAX: u32 = 8_388_608;

/// Largest context whose KV complex still fits VRAM, projected from the
/// live cost per position (KV + rope tail + DSA indexer keys scale
/// linearly in ctx). Resizing past this re-execs into a failed load, and
/// since the re-exec REPLACES this process a failure leaves nothing
/// serving - hence a guard at all.
///
/// `reclaimable` is Stats.kv_headroom, computed by the engine because only
/// it knows placement. Two corrections are baked into that number: expert
/// TIERS count (they are sized from what KV leaves over, so free VRAM is
/// near zero at steady state and a resize rebuilds them smaller), and only
/// the cards that HOST KV count - summing every GPU let a Gqa model whose
/// KV lives on the primary borrow 28GiB of tier sitting on two other
/// cards, and the resize died at cudaMalloc with the server gone.
fn ctx_fit(ctx: u32, kv_bytes: usize, kv_compact: bool, reclaimable: usize) -> u32 {
    if ctx == 0 || kv_bytes == 0 {
        return CTX_SANITY_MAX;
    }
    // Project against the format the engine would ACTUALLY pick at the
    // larger size, not the one running now: below ~2GB it keeps exact f32,
    // above it switches to fp8 (~3.9x cheaper per position). Using the
    // current f32 cost refused resizes that demonstrably load - measured
    // from ctx 2048 (f32, 185KB/pos) it capped at 14k while 262144 (fp8,
    // 48KB/pos) was running minutes earlier.
    let mut per_pos = kv_bytes as f64 / ctx as f64;
    if !kv_compact {
        per_pos /= 3.9;
    }
    let room = kv_bytes as f64 + reclaimable as f64 * 0.85;
    ((room / per_pos) as u32).clamp(512, CTX_SANITY_MAX)
}

/// KV storage format this process was launched with; "auto" when unset,
/// which lets the engine's size-aware default pick (exact f32 while the
/// projection is small, fp8 once it would starve the expert cache).
#[cfg(target_os = "linux")]
fn kv_format() -> String {
    std::env::var("PULSAR_KV").unwrap_or_else(|_| "auto".into())
}

#[cfg(target_os = "linux")]
fn run() -> engine::Result {
    use std::io::{BufRead, BufReader, Read, Write};

    let mut model_path = None;
    let mut port = 11435u16;
    let mut host = String::from("127.0.0.1");
    let mut ctx = 8192u32;
    let mut prefix_file: Option<String> = None;
    let mut webui_mcp_proxy = false;
    let mut mcp_config: Option<String> = None;
    // Jinja chat encoding is **opt-in only** (`--jinja-chat` / PULSAR_JINJA_CHAT).
    // ChatMarkers stay the default so carefully-tuned families do not regress.
    // With Jinja on, resolution is GGUF embed → cache → HF → llama.cpp catalog
    // (network blocked only by PULSAR_OFFLINE).
    let mut jinja_chat = match std::env::var("PULSAR_JINJA_CHAT") {
        Ok(v) if v == "0"
            || v.is_empty()
            || v.eq_ignore_ascii_case("false")
            || v.eq_ignore_ascii_case("off") =>
        {
            false
        }
        Ok(_) => true,
        Err(_) => false,
    };
    let env_offline = std::env::var_os("PULSAR_OFFLINE").is_some();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut need = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match a.as_str() {
            "-m" => model_path = Some(need("-m")?),
            "--port" => port = need("--port")?.parse()?,
            "--host" => host = need("--host")?.to_string(),
            "--ctx" => ctx = need("--ctx")?.parse()?,
            "--prefix-file" => prefix_file = Some(need("--prefix-file")?),
            // MCP on/off switch (the whole feature). Optional path defaults to
            // ./mcp.json next to the server cwd.
            "--webui-mcp-proxy" => webui_mcp_proxy = true,
            "--mcp-config" => mcp_config = Some(need("--mcp-config")?),
            "--jinja-chat" => jinja_chat = true,
            other => return Err(format!("unknown arg {other}").into()),
        }
    }
    let model_path = model_path.ok_or("missing -m MODEL.gguf")?;
    let model_name = std::path::Path::new(&model_path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "pulsar".into());

    eprintln!("pulsar-serve: loading {model_path}");
    let model = engine::Model::load(std::path::Path::new(&model_path))?;
    let tok = {
        let (_, g) = engine::parse_header(std::path::Path::new(&model_path))?;
        tokenizer::Tokenizer::from_gguf(&g)?
    };
    // Resolve a chat template only when Jinja is requested. Default ChatMarkers
    // path never touches the network. With --jinja-chat: embed → cache → HF →
    // llama.cpp catalog (unless PULSAR_OFFLINE).
    let chat_template = if jinja_chat {
        let opts = tokenizer::ChatTemplateOptions {
            offline: env_offline,
            ..Default::default()
        };
        match tokenizer::get_chat_template_from_gguf(
            &model.gguf,
            Some(std::path::Path::new(&model_path)),
            None,
            &opts,
        ) {
            Ok(r) => {
                eprintln!(
                    "pulsar-serve: chat template from {} ({} bytes{})",
                    r.source,
                    r.template.len(),
                    r.model_id
                        .as_ref()
                        .map(|id| format!(", model_id={id}"))
                        .unwrap_or_default()
                );
                Some(r)
            }
            Err(e) => {
                eprintln!(
                    "pulsar-serve: chat template not resolved ({e}){}",
                    if env_offline {
                        " (PULSAR_OFFLINE: embed or local cache only)"
                    } else {
                        ""
                    }
                );
                None
            }
        }
    } else {
        None
    };
    let markers = match tokenizer::ChatMarkers::resolve(&tok) {
        Ok(m) => m,
        Err(e) => {
            // Last resort: if the operator opted into Jinja and we have a
            // template, serve with stops-only markers. Never auto-enable Jinja
            // without --jinja-chat.
            if jinja_chat {
                if chat_template.is_some() {
                    eprintln!(
                        "pulsar-serve: ChatMarkers unresolved ({e}); Jinja encoding with fallback markers"
                    );
                    tokenizer::ChatMarkers::jinja_fallback(&tok)?
                } else {
                    return Err(format!(
                        "ChatMarkers unresolved ({e}) and no Jinja template available \
                         (embed tokenizer.chat_template, warm the cache, or allow network)"
                    )
                    .into());
                }
            } else {
                return Err(format!(
                    "ChatMarkers unresolved ({e}); pass --jinja-chat if this model needs its HF/GGUF template"
                )
                .into());
            }
        }
    };
    if jinja_chat {
        if chat_template.is_some() {
            eprintln!("pulsar-serve: using Jinja chat template for /v1/chat/completions");
        } else {
            eprintln!(
                "pulsar-serve: --jinja-chat set but no template available; ChatMarkers encoding"
            );
            jinja_chat = false;
        }
    } else {
        eprintln!("pulsar-serve: ChatMarkers encoding (pass --jinja-chat to use GGUF/HF Jinja templates)");
    }
    // What the CHECKPOINT supports. Clients size their context control from
    // this; whether a given value fits is a separate, per-machine question
    // answered by ctx_fit.
    let ctx_model_max: u64 = model
        .gguf
        .metadata
        .iter()
        .find(|(k, _)| k.ends_with(".context_length"))
        .and_then(|(_, v)| v.as_u64())
        .unwrap_or(CTX_SANITY_MAX as u64)
        .min(CTX_SANITY_MAX as u64);
    let mut st = engine::State::new(&model, ctx)?;
    let default_temp = model
        .gguf
        .metadata
        .get("general.sampling.temp")
        .and_then(gguf::Value::as_f32)
        .unwrap_or(0.9);

    // MCP hub: only built when --webui-mcp-proxy is passed (the feature gate).
    // Without the flag this stays None, no /mcp/* routes match, no tools are
    // injected, the webui tab stays hidden — zero behavioral change.
    let mcp = if webui_mcp_proxy {
        let cfg_path =
            std::path::PathBuf::from(mcp_config.as_deref().unwrap_or("mcp.json"));
        let m = mcp::McpHub::new(Some(&cfg_path));
        m.connect_all();
        eprintln!(
            "pulsar-serve: --webui-mcp-proxy enabled ({} server(s) configured)",
            m.status_json()["servers"].as_array().map(|a| a.len()).unwrap_or(0)
        );
        Some(m)
    } else {
        None
    };

    // Host values a browser may legitimately reach us on, or None on a
    // network bind where any name is fair game (see allowed_hosts).
    let allowed_hosts = allowed_hosts(&host, port);

    let listener = std::net::TcpListener::bind((host.as_str(), port))?;
    eprintln!("pulsar-serve: listening on http://{host}:{port}  (web UI at /, API at /v1)");
    if allowed_hosts.is_none() {
        eprintln!(
            "pulsar-serve: reachable from other machines - Host allowlist off, and there is no \
             auth here. Keep it to a trusted network, or set PULSAR_ALLOWED_HOSTS=name:{port} to \
             pin the names it answers to."
        );
    }
    // Record this ctx as last-known-good: KV allocated and we are serving. A
    // resize that OOMs dies before reaching here, so PULSAR_CTX_STATE keeps the
    // previous value and the supervisor restarts at it (see launch script).
    if let Some(p) = std::env::var_os("PULSAR_CTX_STATE") {
        let _ = std::fs::write(&p, ctx.to_string());
    }

    let mut request_id = 0u64;
    // token ids fully forwarded into the engine (KV + recurrent state
    // consistent with them); the next request prefills only its suffix
    let mut hist: Vec<u32> = Vec::new();
    // --prefix-file: resume the persisted prefill state (dsv4). A stale
    // or mismatched file is skipped, never fatal.
    let prefix_path = prefix_file.as_ref().map(std::path::PathBuf::from);
    if let Some(pp) = &prefix_path {
        if pp.exists() {
            let t0 = std::time::Instant::now();
            match st.load_prefix(&model, pp) {
                Ok(h) => {
                    eprintln!(
                        "pulsar-serve: prefix restored from {} ({} tokens, {} ckpts, {:.1}s)",
                        pp.display(), h.len(), st.ckpt_count(), t0.elapsed().as_secs_f32()
                    );
                    hist = h;
                }
                Err(e) => eprintln!("pulsar-serve: prefix file skipped: {e}"),
            }
        }
    }
    let mut last_saved = hist.len();
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        // the accept loop is sequential: a half-open socket that never
        // sends its body would block EVERY later request forever (a
        // client retry storm during a restart left exactly that ghost)
        let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(30)));
        let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(120)));
        request_id += 1;
        let result = (|| -> engine::Result {
            let mut reader = BufReader::new(stream.try_clone()?);
            let mut request_line = String::new();
            reader.read_line(&mut request_line)?;
            let mut parts = request_line.split_whitespace();
            let method = parts.next().unwrap_or("").to_owned();
            let path = parts.next().unwrap_or("").to_owned();

            let mut content_length = 0usize;
            let mut origin: Option<String> = None;
            let mut host_hdr: Option<String> = None;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line)?;
                let line = line.trim();
                if line.is_empty() {
                    break;
                }
                let lower = line.to_ascii_lowercase();
                if let Some(v) = lower.strip_prefix("content-length:") {
                    content_length = v.trim().parse().unwrap_or(0);
                } else if let Some(v) = lower.strip_prefix("origin:") {
                    origin = Some(v.trim().to_string());
                } else if let Some(v) = lower.strip_prefix("host:") {
                    host_hdr = Some(v.trim().to_string());
                }
            }
            let mut body = vec![0u8; content_length];
            reader.read_exact(&mut body)?;

            // DNS-rebinding guard, checked on EVERY method. An attacker who
            // points evil.example at 127.0.0.1 makes the browser treat us as
            // same-origin: it sends Origin AND Host both reading
            // "evil.example", so an Origin==Host test passes and the page can
            // read our replies (/mcp/status returns server configs, /models/
            // list returns paths). Only the Host allowlist catches this.
            if !host_allowed(host_hdr.as_deref(), allowed_hosts.as_ref()) {
                eprintln!(
                    "pulsar-serve: rejected unrecognized Host {} on {path} (allowed: {}) - \
                     to reach this server from another machine start it with --host 0.0.0.0, \
                     or add the name via PULSAR_ALLOWED_HOSTS",
                    host_hdr.as_deref().unwrap_or("-"),
                    allowed_hosts.as_ref().map(|v| v.join(", ")).unwrap_or_default()
                );
                return respond_json(
                    &mut stream,
                    403,
                    &serde_json::json!({"error": {"message": "unrecognized Host header"}}),
                );
            }
            // CSRF guard. Every POST here mutates server state, and
            // /mcp/server spawns an arbitrary process (stdio transport) -
            // so a drive-by page was one `fetch` away from code execution:
            // a text/plain POST is a CORS "simple request" (no preflight),
            // so the browser sends it to 127.0.0.1 and the side effect
            // lands even though the attacker never reads the response.
            // Browsers always attach Origin to cross-site POSTs; non-browser
            // clients (curl, the OpenAI SDKs, Claude Code) send none. So:
            // no Origin -> allow, Origin matching Host -> allow (our own
            // web UI), anything else -> 403.
            if method == "POST" && !origin_ok(origin.as_deref(), host_hdr.as_deref()) {
                eprintln!(
                    "pulsar-serve: rejected cross-origin POST {path} (origin {}, host {})",
                    origin.as_deref().unwrap_or("-"),
                    host_hdr.as_deref().unwrap_or("-")
                );
                return respond_json(
                    &mut stream,
                    403,
                    &serde_json::json!({"error": {"message": "cross-origin request rejected"}}),
                );
            }

            match (method.as_str(), path.as_str()) {
                ("GET", "/") | ("GET", "/index.html") => {
                    respond_bytes(&mut stream, 200, "text/html; charset=utf-8", WEBUI_HTML.as_bytes())
                }
                ("GET", "/favicon.ico") | ("GET", "/favicon.svg") => {
                    respond_bytes(&mut stream, 200, "image/svg+xml", FAVICON_SVG.as_bytes())
                }
                ("GET", "/v1/models") => {
                    // `created` is required on the model object for the same
                    // reason it is on completions: strict clients deserialize
                    // into a struct with non-optional fields. The gguf's mtime
                    // is the honest answer to "when did this model appear".
                    let created = std::fs::metadata(&model_path)
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    let json = serde_json::json!({
                        "object": "list",
                        "data": [{"id": model_name, "object": "model", "created": created, "owned_by": "pulsar", "max_model_len": ctx}],
                    });
                    respond_json(&mut stream, 200, &json)
                }
                // Telemetry for the web dashboard (Runtime/tier bars, timing
                // breakdown, Brain heat). Cumulative counters; the UI diffs
                // successive snapshots for per-turn deltas.
                ("GET", "/stats") => {
                    let s = st.stats();
                    let meta = &model.gguf.metadata;
                    let find_u = |suf: &str| {
                        meta.iter()
                            .find(|(k, _)| k.ends_with(suf))
                            .and_then(|(_, v)| v.as_u64())
                    };
                    let tiers: Vec<_> = s
                        .tiers
                        .iter()
                        .map(|t| serde_json::json!({"dev": t.dev, "bytes": t.bytes, "hits": t.hits}))
                        .collect();
                    let gpus: Vec<_> = gpu_info()
                        .into_iter()
                        .enumerate()
                        .map(|(i, (name, total, used))| serde_json::json!({
                            "dev": i, "name": name, "vram_used": used, "vram_total": total,
                        }))
                        .collect();
                    // model residency: VRAM (resident tiers + slab cache), RAM (host
                    // cache), disk (the streamed remainder of the gguf on disk)
                    // A split model's -m points at shard 1, so metadata() on
                    // it reports 638MB for a 528GB model and the streamed
                    // remainder saturates to zero - the panel then shows only
                    // the caches, as if the whole model were resident.
                    let model_bytes = {
                        let p = std::path::Path::new(&model_path);
                        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                        match (split_shard(name), p.parent()) {
                            (Some(_), Some(d)) => dir_gguf_bytes(d, name),
                            _ => std::fs::metadata(p).map(|m| m.len()).unwrap_or(0),
                        }
                    };
                    let ram = s.host_used as u64;
                    // Only the routed experts stream; everything else was
                    // uploaded at load and never leaves VRAM. vram_resident
                    // counts the expert tiers and slab cache ONLY, so without
                    // this the resident remainder reads as "on disk" - 31GB of
                    // it on K3, whose attention and shared-expert stack dwarfs
                    // its expert cache.
                    let expert_pool: u64 = model
                        .gguf
                        .tensors
                        .iter()
                        .filter(|t| t.name.ends_with("_exps.weight"))
                        .filter_map(|t| t.byte_size())
                        .sum();
                    // Dense models (no MoE experts) are placed via per-layer card
                    // ownership - fully resident in VRAM, but not counted by the
                    // MoE tier residency counter, which made the whole model show
                    // as "disk". Attribute their weight to VRAM (they never stream).
                    let dense = find_u(".expert_count").unwrap_or(0) == 0;
                    let vram = if dense {
                        model_bytes.saturating_sub(ram)
                    } else {
                        // resident non-expert weights + whatever of the expert
                        // pool is cached in VRAM
                        model_bytes.saturating_sub(expert_pool) + s.vram_resident as u64
                    };
                    let disk = if dense {
                        0
                    } else {
                        expert_pool.saturating_sub(s.vram_resident as u64 + ram)
                    };
                    let json = serde_json::json!({
                        "model": model_name,
                        "ctx": s.ctx,
                        "cpu_lane": cpu_lane_on(),
                        "mtp": mtp_on(),
                        // reasoning controls the client can offer: the
                        // vocabulary is per chat style, so the UI builds
                        // its control from this instead of hardcoding one
                        // ctx_max: what the CHECKPOINT supports (clients size
                        // their context control from this, so it follows the
                        // model rather than a hardcoded number). ctx_fit: the
                        // largest ctx whose KV still fits VRAM, projected from
                        // the live KV cost per position - past it the resize
                        // would re-exec into a failed load.
                        "ctx_max": ctx_model_max,
                        "ctx_fit": ctx_fit(s.ctx, s.kv_bytes, s.kv_compact, s.kv_headroom),
                        "kv_bytes": s.kv_bytes,
                        "kv_headroom": s.kv_headroom,
                        "kv_format": kv_format(),
                        "kv_compact": s.kv_compact,
                        "kv_resolved": s.kv_resolved,
                        "reasoning": {
                            "capable": markers.reasoning_capable(),
                            "levels": markers.reasoning_levels(),
                            "default": markers.reasoning_default(),
                            "thinking": markers.opens_thinking(),
                        },
                        "n_layer": find_u(".block_count").unwrap_or(0),
                        "n_expert": find_u(".expert_count").unwrap_or(0),
                        "n_expert_used": find_u(".expert_used_count").unwrap_or(0),
                        "hardware": {
                            "gpu_count": s.gpu_count,
                            "cpu_name": cpu_name(),
                            "cores": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
                            "ram_total_kb": meminfo_kb("MemTotal"),
                            "ram_available_kb": meminfo_kb("MemAvailable"),
                            "gpus": gpus,
                        },
                        "model_bytes": model_bytes,
                        "residency": {"vram": vram, "ram": ram, "ram_budget": s.host_budget, "disk": disk},
                        "tiers": tiers,
                        "cpu_hits": s.cpu_hits,
                        "cache_hits": s.cache_hits,
                        "prof": {
                            "gpu_wait": s.prof_gpu_wait,
                            "resolve": s.prof_resolve,
                            "h2d": s.prof_h2d,
                            "fetch": s.prof_fetch,
                            "cpu": s.prof_cpu,
                            "tail": s.prof_tail,
                            "calls": s.prof_calls,
                        },
                    });
                    respond_json(&mut stream, 200, &json)
                }
                // Per-expert residency + routing heat for the Brain cortex.
                // Grouped by MoE layer into compact parallel arrays.
                ("GET", "/experts") => {
                    let n_expert = model
                        .gguf
                        .metadata
                        .iter()
                        .find(|(k, _)| k.ends_with(".expert_count"))
                        .and_then(|(_, v)| v.as_u64())
                        .unwrap_or(0) as usize;
                    let cells = st.expert_map(&model);
                    let layers: Vec<_> = if n_expert > 0 {
                        cells
                            .chunks(n_expert)
                            .map(|ch| {
                                let tier: Vec<u8> = ch.iter().map(|c| c.tier).collect();
                                let heat: Vec<u64> = ch.iter().map(|c| c.heat).collect();
                                serde_json::json!({"layer": ch[0].layer, "tier": tier, "heat": heat})
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };
                    let json = serde_json::json!({"n_expert": n_expert, "layers": layers});
                    respond_json(&mut stream, 200, &json)
                }
                // Expert topic-affinity atlas, built offline into a per-model
                // sidecar (<model>.atlas.json) by scripts/atlas_build.py.
                ("GET", "/atlas") => match std::fs::read(format!("{model_path}.atlas.json")) {
                    Ok(bytes) => respond_bytes(&mut stream, 200, "application/json", &bytes),
                    Err(_) => respond_json(
                        &mut stream,
                        404,
                        &serde_json::json!({"error": {"message": "no atlas built for this model"}}),
                    ),
                },
                // Available *.gguf models in the current model's directory.
                ("GET", "/models/list") => {
                    let dir = std::path::Path::new(&model_path)
                        .parent()
                        .map(|p| p.to_path_buf())
                        .unwrap_or_default();
                    let cur = std::path::Path::new(&model_path)
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("");
                    // A split model lives in its own directory, so the
                    // listing has to look one level down - and when the
                    // CURRENT model is one of those, one level up as well,
                    // or switching back to a top-level model is impossible.
                    let base = dir
                        .parent()
                        .filter(|p| has_gguf(p))
                        .map(|p| p.to_path_buf())
                        .unwrap_or_else(|| dir.clone());
                    let mut scan = vec![base.clone()];
                    scan.extend(
                        std::fs::read_dir(&base)
                            .into_iter()
                            .flatten()
                            .flatten()
                            .map(|e| e.path())
                            .filter(|p| p.is_dir()),
                    );
                    let cur_rel = std::path::Path::new(&model_path)
                        .strip_prefix(&base)
                        .map(|p| p.to_string_lossy().into_owned())
                        .unwrap_or_else(|_| cur.to_string());
                    let mut models: Vec<serde_json::Value> = Vec::new();
                    for d in scan {
                        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
                            let name = e.file_name().to_string_lossy().into_owned();
                            // skip non-servable: MTP draft sidecars, and full-
                            // precision source checkpoints (the -F16/-BF16 quant
                            // suffix, NOT "fromBF16" provenance on real quants)
                            if !name.ends_with(".gguf")
                                || name.contains("draft")
                                || name.ends_with("-F16.gguf")
                                || name.ends_with("-BF16.gguf")
                            {
                                continue;
                            }
                            // One entry per MODEL, not per file: a split gguf
                            // is served by opening shard 1, so the later
                            // shards are not separately loadable and listing
                            // them offers 93 broken choices.
                            let split = split_shard(&name);
                            if matches!(split, Some(n) if n != 1) {
                                continue;
                            }
                            let path = e.path();
                            let bytes = if split.is_some() {
                                // the whole model, not the 638MB first shard
                                dir_gguf_bytes(&d, &name)
                            } else {
                                e.metadata().map(|m| m.len()).unwrap_or(0)
                            };
                            let id = path
                                .strip_prefix(&base)
                                .map(|p| p.to_string_lossy().into_owned())
                                .unwrap_or_else(|_| name.clone());
                            models.push(serde_json::json!({
                                "id": id,
                                "label": pretty_model_name(&name),
                                "bytes": bytes,
                                "current": id == cur_rel,
                            }));
                        }
                    }
                    models.sort_by(|a, b| a["label"].as_str().cmp(&b["label"].as_str()));
                    respond_json(&mut stream, 200, &serde_json::json!({"models": models}))
                }
                // Switch model: validate, ack, then re-exec with the new -m.
                ("POST", "/models/load") => {
                    let req: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
                    let name = req["model"].as_str().unwrap_or("");
                    let dir = std::path::Path::new(&model_path)
                        .parent()
                        .map(|p| p.to_path_buf())
                        .unwrap_or_default();
                    // ids may carry one directory component (split models
                    // live in their own folder); resolve against the same
                    // base the listing used, and refuse traversal.
                    let base = dir
                        .parent()
                        .filter(|p| has_gguf(p))
                        .map(|p| p.to_path_buf())
                        .unwrap_or_else(|| dir.clone());
                    let target = base.join(name);
                    // Path::join with an ABSOLUTE name discards the base
                    // entirely, so the component checks alone would let
                    // "/x.gguf" escape; the pre-subdirectory version was
                    // safe only because it rejected every '/'. Verify
                    // containment after canonicalizing, which also
                    // settles symlinks.
                    let contained = || {
                        match (target.canonicalize(), base.canonicalize()) {
                            (Ok(t), Ok(b)) => t.starts_with(&b),
                            _ => false,
                        }
                    };
                    let ok = !name.is_empty()
                        && !std::path::Path::new(name).is_absolute()
                        && !name.contains("..")
                        && !name.contains('\\')
                        && name.matches('/').count() <= 1
                        && name.ends_with(".gguf")
                        && target.is_file()
                        && contained();
                    if !ok {
                        respond_json(&mut stream, 400, &serde_json::json!({"error": {"message": "invalid or unknown model"}}))
                    } else {
                        respond_json(&mut stream, 200, &serde_json::json!({"reloading": name}))?;
                        let _ = std::io::Write::flush(&mut stream);
                        eprintln!("pulsar-serve: switching model -> {name}, re-exec");
                        let err = reexec(&target, cpu_lane_on(), None, mtp_on(), None);
                        eprintln!("pulsar-serve: re-exec failed: {err}");
                        std::process::exit(1);
                    }
                }
                // Toggle the AVX2 CPU expert lane: re-exec the current model
                // with PULSAR_CPU set/unset (reload, same as a model switch).
                ("POST", "/cpu_lane") => {
                    let req: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
                    let enabled = req["enabled"].as_bool().unwrap_or(false);
                    respond_json(&mut stream, 200, &serde_json::json!({"reloading": true, "cpu_lane": enabled}))?;
                    let _ = std::io::Write::flush(&mut stream);
                    eprintln!("pulsar-serve: CPU lane -> {enabled}, re-exec");
                    let err = reexec(std::path::Path::new(&model_path), enabled, None, mtp_on(), None);
                    eprintln!("pulsar-serve: re-exec failed: {err}");
                    std::process::exit(1);
                }
                // Toggle MTP speculative decode: re-exec with PULSAR_MTP set/unset.
                // Models with no nextn/MTP block ignore it (engine logs a warning).
                ("POST", "/mtp") => {
                    let req: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
                    let enabled = req["enabled"].as_bool().unwrap_or(false);
                    respond_json(&mut stream, 200, &serde_json::json!({"reloading": true, "mtp": enabled}))?;
                    let _ = std::io::Write::flush(&mut stream);
                    eprintln!("pulsar-serve: MTP -> {enabled}, re-exec");
                    let err = reexec(std::path::Path::new(&model_path), cpu_lane_on(), None, enabled, None);
                    eprintln!("pulsar-serve: re-exec failed: {err}");
                    std::process::exit(1);
                }
                // KV storage format: re-exec with a new PULSAR_KV. Everything
                // downstream re-derives from it (KV bytes -> auto budget ->
                // expert cache -> tier build), same as a context resize.
                ("POST", "/kv") => {
                    let req: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
                    let want = req["format"].as_str().unwrap_or("auto").to_string();
                    const KV_FORMATS: [&str; 9] =
                        ["auto", "f32", "fp8", "fp16", "int8", "q8_0", "q4_0", "turbo4", "turbo8"];
                    if !KV_FORMATS.contains(&want.as_str()) {
                        respond_json(&mut stream, 400, &serde_json::json!({"error": {"message":
                            format!("unknown KV format {want} (one of {KV_FORMATS:?})")}}))
                    } else {
                        respond_json(&mut stream, 200, &serde_json::json!({"reloading": true, "kv": want}))?;
                        let _ = std::io::Write::flush(&mut stream);
                        eprintln!("pulsar-serve: PULSAR_KV -> {want}, re-exec");
                        let err = reexec(std::path::Path::new(&model_path), cpu_lane_on(), None, mtp_on(), Some(&want));
                        eprintln!("pulsar-serve: re-exec failed: {err}");
                        std::process::exit(1);
                    }
                }
                // Resize the context window: re-exec the current model with a new
                // --ctx (reallocates the KV cache, same as a model switch).
                ("POST", "/ctx") => {
                    let req: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
                    let n = req["ctx"].as_u64().unwrap_or(0) as u32;
                    let fs = st.stats();
                    let fit = ctx_fit(fs.ctx, fs.kv_bytes, fs.kv_compact, fs.kv_headroom);
                    if !(512..=ctx_model_max as u32).contains(&n) {
                        respond_json(&mut stream, 400, &serde_json::json!({"error": {"message":
                            format!("ctx out of range (512..{ctx_model_max}, the checkpoint's trained context)")}}))
                    } else if n > fit {
                        // Refuse rather than re-exec into an OOM: the re-exec
                        // replaces this process, so a failed load leaves the
                        // user with no server at all.
                        respond_json(&mut stream, 400, &serde_json::json!({"error": {"message":
                            format!("ctx {n} needs more KV than VRAM has free; largest that fits now is {fit}")}}))
                    } else {
                        respond_json(&mut stream, 200, &serde_json::json!({"reloading": true, "ctx": n}))?;
                        let _ = std::io::Write::flush(&mut stream);
                        eprintln!("pulsar-serve: ctx -> {n}, re-exec");
                        let err = reexec(std::path::Path::new(&model_path), cpu_lane_on(), Some(n), mtp_on(), None);
                        eprintln!("pulsar-serve: re-exec failed: {err}");
                        std::process::exit(1);
                    }
                }
                ("POST", "/v1/chat/completions") => handle_chat(
                    &mut stream,
                    &body,
                    &model,
                    &tok,
                    &markers,
                    chat_template.as_ref(),
                    jinja_chat,
                    &mut st,
                    &model_name,
                    default_temp,
                    request_id,
                    &mut hist,
                    mcp.as_ref(),
                ),
                // --- MCP management + tool surface (only meaningful with
                // --webui-mcp-proxy; 404 otherwise so the webui probe hides
                // the tab). ---
                ("GET", "/mcp/status") => match &mcp {
                    Some(m) => respond_json(&mut stream, 200, &m.status_json()),
                    None => respond_json(
                        &mut stream,
                        404,
                        &serde_json::json!({"error": {"message": "--webui-mcp-proxy not enabled"}}),
                    ),
                },
                ("GET", "/v1/tools") => match &mcp {
                    Some(m) => respond_json(
                        &mut stream,
                        200,
                        &serde_json::json!({"object": "list", "data": m.enabled_tools_as_openai()}),
                    ),
                    None => respond_json(
                        &mut stream,
                        404,
                        &serde_json::json!({"error": {"message": "--webui-mcp-proxy not enabled"}}),
                    ),
                },
                ("POST", "/mcp/server") => match &mcp {
                    Some(m) => {
                        let v: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
                        let name = v["name"].as_str().unwrap_or("").to_string();
                        let cfg = v
                            .get("config")
                            .and_then(|c| serde_json::from_value::<mcp::McpServerCfg>(c.clone()).ok());
                        match cfg {
                            Some(cfg) if !name.is_empty() => {
                                m.upsert_server(&name, cfg);
                                respond_json(&mut stream, 200, &m.status_json())
                            }
                            _ => respond_json(
                                &mut stream,
                                400,
                                &serde_json::json!({"error": {"message": "invalid server config"}}),
                            ),
                        }
                    }
                    None => respond_json(
                        &mut stream,
                        404,
                        &serde_json::json!({"error": {"message": "--webui-mcp-proxy not enabled"}}),
                    ),
                },
                ("POST", "/mcp/server/delete") => match &mcp {
                    Some(m) => {
                        let v: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
                        if let Some(name) = v["name"].as_str() {
                            m.remove_server(name);
                        }
                        respond_json(&mut stream, 200, &m.status_json())
                    }
                    None => respond_json(
                        &mut stream,
                        404,
                        &serde_json::json!({"error": {"message": "--webui-mcp-proxy not enabled"}}),
                    ),
                },
                ("POST", "/mcp/toggle") => match &mcp {
                    Some(m) => {
                        let v: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
                        if let (Some(tool), Some(disabled)) = (v["tool"].as_str(), v["disabled"].as_bool()) {
                            m.toggle(tool, disabled);
                        }
                        respond_json(&mut stream, 200, &serde_json::json!({"ok": true}))
                    }
                    None => respond_json(
                        &mut stream,
                        404,
                        &serde_json::json!({"error": {"message": "--webui-mcp-proxy not enabled"}}),
                    ),
                },
                _ => respond_json(
                    &mut stream,
                    404,
                    &serde_json::json!({"error": {"message": "not found"}}),
                ),
            }
        })();
        if let Err(e) = result {
            eprintln!("pulsar-serve: request failed: {e}");
            let _ = stream.write_all(
                b"HTTP/1.1 500 Internal Server Error\r\ncontent-length: 0\r\n\r\n",
            );
        }
        // persist the prefill investment once it has grown meaningfully;
        // a save costs seconds, a lost prefix costs a 20-40 min re-prefill
        if let Some(pp) = &prefix_path {
            if hist.len() >= last_saved + 2048 {
                let t0 = std::time::Instant::now();
                match st.save_prefix(&model, &hist, pp) {
                    Ok(()) => {
                        eprintln!(
                            "pulsar-serve: prefix saved ({} tokens, {:.1}s)",
                            hist.len(), t0.elapsed().as_secs_f32()
                        );
                        last_saved = hist.len();
                    }
                    Err(e) => eprintln!("pulsar-serve: prefix save failed: {e}"),
                }
            }
        }
    }
    Ok(())
}

/// True when the bind address keeps the server off the network entirely.
/// Only then is a Host allowlist both safe to impose and worth imposing.
#[cfg(target_os = "linux")]
fn is_loopback_bind(bind_host: &str) -> bool {
    matches!(bind_host, "localhost" | "::1" | "[::1]") || bind_host.starts_with("127.")
}

/// Host authorities a browser may legitimately reach this server on, or
/// `None` for "accept any".
///
/// The allowlist exists to stop DNS rebinding, and rebinding only buys an
/// attacker something when the service is otherwise unreachable - i.e. bound
/// to loopback. A `--host 0.0.0.0` (or LAN IP) bind is an explicit request to
/// be reachable from other machines under names we cannot enumerate, and
/// anyone who can rebind can already connect directly, so the list would cost
/// real usability and buy nothing. Setting PULSAR_ALLOWED_HOSTS forces the
/// strict path back on regardless of bind, for a public bind that still wants
/// a fixed set of names.
#[cfg(target_os = "linux")]
fn allowed_hosts(bind_host: &str, port: u16) -> Option<Vec<String>> {
    let explicit = std::env::var("PULSAR_ALLOWED_HOSTS")
        .ok()
        .filter(|s| !s.trim().is_empty());
    if explicit.is_none() && !is_loopback_bind(bind_host) {
        return None;
    }
    let mut v = vec![
        format!("127.0.0.1:{port}"),
        format!("localhost:{port}"),
        format!("[::1]:{port}"),
        format!("{bind_host}:{port}"),
    ];
    // Browsers omit the port from Host when it is the scheme default.
    if port == 80 {
        v.extend([
            "127.0.0.1".into(),
            "localhost".into(),
            "[::1]".into(),
            bind_host.to_string(),
        ]);
    }
    if let Some(extra) = explicit {
        v.extend(
            extra
                .split(',')
                .map(|s| s.trim().to_ascii_lowercase())
                .filter(|s| !s.is_empty()),
        );
    }
    // --host 127.0.0.1 (the default) repeats a loopback entry; keep the log
    // line readable.
    v.sort();
    v.dedup();
    Some(v)
}

/// True when the request's `Host` is one we answer to. A missing Host is
/// allowed (HTTP/1.0 clients omit it); browsers always send one, so a
/// rebinding attack can never present as absent.
#[cfg(target_os = "linux")]
fn host_allowed(host: Option<&str>, allowed: Option<&Vec<String>>) -> bool {
    let Some(allowed) = allowed else { return true };
    let Some(h) = host else { return true };
    let h = h.to_ascii_lowercase();
    allowed.contains(&h)
}

/// True when a POST may proceed: either it carries no `Origin` (every
/// non-browser client) or its Origin authority equals the `Host` we were
/// reached on (our own web UI). `Origin: null` - sandboxed iframe, file://
/// page - is never same-origin, so it falls through to false.
///
/// This runs only after host_allowed has confirmed the Host is one of ours,
/// which is what makes the equality test meaningful: without that, DNS
/// rebinding satisfies Origin==Host with the attacker's own domain on both
/// sides.
///
/// ponytail: authority string compare, no URL parser. A reverse proxy that
/// rewrites Host without rewriting Origin still 403s here; PULSAR_ALLOWED_HOSTS
/// does not cover that case, and the fix would be a matching origin allowlist.
#[cfg(target_os = "linux")]
fn origin_ok(origin: Option<&str>, host: Option<&str>) -> bool {
    let Some(origin) = origin else { return true };
    let authority = origin
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(origin);
    match host {
        Some(h) => !authority.is_empty() && authority == h,
        None => false,
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::{allowed_hosts, host_allowed, origin_ok};

    #[test]
    fn rebinding_guard() {
        // Default loopback bind: the allowlist is on, because that is the
        // case where rebinding reaches something the attacker otherwise
        // could not.
        let allowed = allowed_hosts("127.0.0.1", 11435);
        assert!(allowed.is_some());
        let a = allowed.as_ref();
        // the ways we are actually reached
        assert!(host_allowed(Some("127.0.0.1:11435"), a));
        assert!(host_allowed(Some("localhost:11435"), a));
        assert!(host_allowed(Some("LocalHost:11435"), a)); // Host is case-insensitive
        assert!(host_allowed(None, a)); // HTTP/1.0 client
        // DNS rebinding: attacker's domain resolved to 127.0.0.1. Origin and
        // Host agree, so origin_ok alone would have let this through.
        assert!(origin_ok(Some("http://evil.example:11435"), Some("evil.example:11435")));
        assert!(!host_allowed(Some("evil.example:11435"), a));
        // right name, wrong port is a different origin
        assert!(!host_allowed(Some("127.0.0.1:9999"), a));
    }

    #[test]
    fn network_bind_accepts_any_host() {
        // --host 0.0.0.0 is an explicit request to be reachable under names
        // we cannot enumerate (LAN IP, mDNS name, Tailscale name). Rebinding
        // buys an attacker nothing there - they can already connect - so the
        // allowlist would only break the feature. See issue #20.
        let wild = allowed_hosts("0.0.0.0", 11435);
        assert!(wild.is_none());
        assert!(host_allowed(Some("192.168.1.50:11435"), wild.as_ref()));
        assert!(host_allowed(Some("desktop.local:11435"), wild.as_ref()));
        // A specific LAN bind is equally a network bind.
        assert!(allowed_hosts("192.168.1.50", 11435).is_none());
        // ...but loopback in any spelling stays strict.
        assert!(allowed_hosts("localhost", 11435).is_some());
        assert!(allowed_hosts("127.0.0.53", 11435).is_some());
    }

    #[test]
    fn csrf_guard() {
        // non-browser clients send no Origin at all
        assert!(origin_ok(None, Some("127.0.0.1:8080")));
        // our own web UI, served from the same host:port
        assert!(origin_ok(Some("http://127.0.0.1:8080"), Some("127.0.0.1:8080")));
        assert!(origin_ok(Some("https://box.tail1234.ts.net"), Some("box.tail1234.ts.net")));
        // a drive-by page: this is the one that used to reach /mcp/server
        assert!(!origin_ok(Some("https://evil.example"), Some("127.0.0.1:8080")));
        // same host, different port is still a different origin
        assert!(!origin_ok(Some("http://127.0.0.1:9999"), Some("127.0.0.1:8080")));
        // sandboxed iframe / file:// page
        assert!(!origin_ok(Some("null"), Some("127.0.0.1:8080")));
        // malformed / absent Host cannot be matched against
        assert!(!origin_ok(Some("http://127.0.0.1:8080"), None));
        assert!(!origin_ok(Some("http://"), Some("127.0.0.1:8080")));
    }
}

#[cfg(target_os = "linux")]
fn respond_json(
    stream: &mut std::net::TcpStream,
    status: u16,
    json: &serde_json::Value,
) -> engine::Result {
    use std::io::Write;
    let body = json.to_string();
    let reason = if status == 200 { "OK" } else { "Error" };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )?;
    Ok(())
}

/// A /proc/meminfo field in kB (0 if unavailable, e.g. non-Linux). Used by
/// the /stats hardware panel.
fn meminfo_kb(key: &str) -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0)
}

/// CPU model name from /proc/cpuinfo (empty if unavailable). /stats hardware.
fn cpu_name() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|v| v.trim().to_string())
        })
        .unwrap_or_default()
}

/// Re-exec this process with a different `-m` model, keeping every other arg
/// (host/port/ctx). Same PID/systemd unit; the new image reloads the model.
/// Only returns (with an error) if exec fails. The model path is validated by
/// True when `d` directly contains at least one .gguf.
fn has_gguf(d: &std::path::Path) -> bool {
    std::fs::read_dir(d)
        .into_iter()
        .flatten()
        .flatten()
        .any(|e| e.file_name().to_string_lossy().ends_with(".gguf"))
}

/// `Some(n)` when `name` is shard n of a split gguf (`-00007-of-00094.gguf`).
fn split_shard(name: &str) -> Option<u32> {
    let stem = name.strip_suffix(".gguf")?;
    let (head, total) = stem.rsplit_once("-of-")?;
    if total.len() != 5 || !total.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let (_, n) = head.rsplit_once('-')?;
    if n.len() != 5 || !n.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    n.parse().ok()
}

/// Total bytes of every shard of the split model `first` belongs to.
fn dir_gguf_bytes(d: &std::path::Path, first: &str) -> u64 {
    let prefix = match first.rsplit_once("-of-") {
        Some((head, _)) => match head.rsplit_once('-') {
            Some((p, _)) => p.to_string(),
            None => return 0,
        },
        None => return 0,
    };
    std::fs::read_dir(d)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            n.starts_with(&prefix) && n.ends_with(".gguf") && split_shard(&n).is_some()
        })
        .filter_map(|e| e.metadata().ok().map(|m| m.len()))
        .sum()
}

/// Display name: drop the .gguf and the `-00001-of-00094` shard suffix.
fn pretty_model_name(name: &str) -> String {
    let stem = name.strip_suffix(".gguf").unwrap_or(name);
    match stem.rsplit_once("-of-") {
        Some((head, _)) => match head.rsplit_once('-') {
            Some((p, n)) if n.len() == 5 && n.bytes().all(|b| b.is_ascii_digit()) => p.to_string(),
            _ => stem.to_string(),
        },
        None => stem.to_string(),
    }
}

/// the caller to be a .gguf inside the current model's directory.
#[cfg(target_os = "linux")]
fn reexec(newmodel: &std::path::Path, cpu_lane: bool, new_ctx: Option<u32>, mtp: bool, kv: Option<&str>) -> std::io::Error {
    use std::os::unix::process::CommandExt;
    let args: Vec<String> = std::env::args().collect();
    let mut cmd = std::process::Command::new(&args[0]);
    let mut i = 1;
    let mut saw_ctx = false;
    while i < args.len() {
        if args[i] == "-m" || args[i] == "--model" {
            cmd.arg(&args[i]).arg(newmodel);
            i += 2;
        } else if args[i] == "--ctx" && i + 1 < args.len() {
            saw_ctx = true;
            cmd.arg("--ctx");
            match new_ctx {
                Some(c) => { cmd.arg(c.to_string()); }
                None => { cmd.arg(&args[i + 1]); }
            }
            i += 2;
        } else {
            cmd.arg(&args[i]);
            i += 1;
        }
    }
    if let (false, Some(c)) = (saw_ctx, new_ctx) {
        cmd.arg("--ctx").arg(c.to_string());
    }
    if cpu_lane {
        cmd.env("PULSAR_CPU", "1");
    } else {
        cmd.env_remove("PULSAR_CPU");
    }
    if mtp {
        cmd.env("PULSAR_MTP", "1");
    } else {
        cmd.env_remove("PULSAR_MTP");
    }
    // KV storage format. None = keep whatever this process was given;
    // Some("auto") = clear it so the engine's size-aware default decides.
    match kv {
        Some("auto") => { cmd.env_remove("PULSAR_KV"); }
        Some(v) => { cmd.env("PULSAR_KV", v); }
        None => {}
    }
    cmd.exec()
}

/// Whether the AVX2 CPU expert lane is enabled in this process.
fn cpu_lane_on() -> bool {
    std::env::var_os("PULSAR_CPU").is_some_and(|v| v != "0" && !v.is_empty())
}

/// Whether MTP speculative decode is enabled in this process.
fn mtp_on() -> bool {
    std::env::var_os("PULSAR_MTP").is_some_and(|v| v != "0" && !v.is_empty())
}

/// Per-GPU (name, vram_total_bytes, vram_used_bytes) for EVERY card, from one
/// nvidia-smi query so name/total/used stay consistent for the same device
/// (the engine's cudaMemGetInfo order can differ from nvidia-smi's, which
/// mismatched names against VRAM on multi-GPU boxes).
fn gpu_info() -> Vec<(String, u64, u64)> {
    std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter_map(|l| {
                    let mut p = l.split(',');
                    let name = p.next()?.trim().to_string();
                    let total: u64 = p.next()?.trim().parse().ok()?;
                    let used: u64 = p.next()?.trim().parse().ok()?;
                    Some((name, total * 1024 * 1024, used * 1024 * 1024)) // MiB -> bytes
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Raw bytes response with an explicit content-type (static assets, etc).
#[cfg(target_os = "linux")]
fn respond_bytes(
    stream: &mut std::net::TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> engine::Result {
    use std::io::Write;
    let reason = if status == 200 { "OK" } else { "Error" };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    Ok(())
}

/// content arrives as a plain string OR an array of typed blocks
/// (Claude Code / Anthropic-translated clients send
/// [{type:"text", text:...}, ...]); a string-only read silently
/// dropped the whole system prompt for those clients
#[cfg(target_os = "linux")]
fn message_text_of(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(blocks) => blocks
            .iter()
            .map(|b| {
                if let Some(t) = b["text"].as_str() {
                    t.to_string()
                } else if b["type"].as_str() == Some("tool_result") {
                    message_text_of(&b["content"])
                } else {
                    String::new()
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// Pick Jinja or ChatMarkers encoding for a message list.
#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)] // one call site; the request fields do not want a struct
fn encode_messages_auto(
    tok: &tokenizer::Tokenizer,
    m: &tokenizer::ChatMarkers,
    chat_template: Option<&tokenizer::ResolvedChatTemplate>,
    jinja_chat: bool,
    messages: &[serde_json::Value],
    tools: Option<&Vec<serde_json::Value>>,
    enable_thinking: Option<bool>,
    reasoning_effort: Option<&str>,
) -> Vec<u32> {
    if jinja_chat {
        if let Some(tmpl) = chat_template {
            match encode_messages_jinja(
                tok,
                tmpl,
                messages,
                tools,
                enable_thinking,
                reasoning_effort,
            ) {
                Ok(ids) => return ids,
                Err(e) => {
                    eprintln!(
                        "pulsar-serve: jinja chat template apply failed ({e}); falling back to ChatMarkers"
                    );
                }
            }
        }
    }
    encode_messages(tok, m, messages, tools)
}

/// Encode OpenAI messages via a resolved Jinja chat template, then
/// tokenize with special-token recognition.
#[cfg(target_os = "linux")]
fn encode_messages_jinja(
    tok: &tokenizer::Tokenizer,
    template: &tokenizer::ResolvedChatTemplate,
    messages: &[serde_json::Value],
    tools: Option<&Vec<serde_json::Value>>,
    enable_thinking: Option<bool>,
    reasoning_effort: Option<&str>,
) -> Result<Vec<u32>, String> {
    // DeepSeek V4 speaks DSML; Poolside Laguna speaks arg_key/arg_value XML.
    // Replaying the wrong dialect leaves the model stuck after MCP dispatch.
    let dsml = tool_calls::is_dsml_template(&template.template);
    let poolside = tool_calls::is_poolside_template(&template.template);
    let mut chat_msgs: Vec<tokenizer::ChatMessage> = Vec::new();
    for msg in messages {
        let role = msg["role"].as_str().unwrap_or("user").to_string();
        let mut content = message_text_of(&msg["content"]);
        if role == "assistant" {
            if let Some(calls) = msg["tool_calls"].as_array() {
                let pairs: Vec<(String, String)> = calls
                    .iter()
                    .filter_map(|c| {
                        let f = &c["function"];
                        let name = f["name"].as_str()?.to_string();
                        let args = f["arguments"].as_str().unwrap_or("{}").to_string();
                        Some((name, args))
                    })
                    .collect();
                if !pairs.is_empty() {
                    if !content.is_empty() {
                        content.push('\n');
                    }
                    if dsml {
                        content.push_str(&tool_calls::format_dsml_tool_calls(&pairs));
                    } else if poolside {
                        content.push_str(&tool_calls::format_poolside_tool_calls(&pairs));
                    } else {
                        content.push_str(&tool_calls::format_generic_tool_calls(&pairs));
                    }
                }
            }
        }
        if role == "tool" {
            if poolside && !dsml {
                // Laguna Jinja: role=tool → <tool_response>{{ content }}</tool_response>
                chat_msgs.push(tokenizer::ChatMessage {
                    role: "tool".into(),
                    content,
                });
            } else {
                let body = if dsml {
                    tool_calls::format_dsml_tool_result(&content)
                } else {
                    let id = msg["tool_call_id"].as_str().unwrap_or("");
                    tool_calls::format_generic_tool_result(id, &content)
                };
                // DeepSeek V4 examples feed tool results as a user turn.
                chat_msgs.push(tokenizer::ChatMessage {
                    role: "user".into(),
                    content: body,
                });
            }
            continue;
        }
        chat_msgs.push(tokenizer::ChatMessage { role, content });
    }

    let tools_json = tools.map(|t| {
        let schemas: Vec<&serde_json::Value> = t.iter().map(|f| &f["function"]).collect();
        serde_json::json!(schemas)
    });

    let mut extra = serde_json::Map::new();
    if let Some(b) = enable_thinking {
        extra.insert("enable_thinking".into(), serde_json::Value::Bool(b));
    }
    if let Some(e) = reasoning_effort {
        extra.insert(
            "reasoning_effort".into(),
            serde_json::Value::String(e.into()),
        );
    }
    let extra_v = if extra.is_empty() {
        None
    } else {
        Some(serde_json::Value::Object(extra))
    };

    let bos = tok.bos_id.and_then(|id| tok.token_str(id));
    let eos = tok.eos_id.and_then(|id| tok.token_str(id));
    let rendered = tokenizer::apply_chat_template_ex(
        &template.template,
        &chat_msgs,
        true,
        bos,
        eos,
        tools_json.as_ref(),
        extra_v.as_ref(),
    )
    .map_err(|e| e.to_string())?;
    if std::env::var_os("PULSAR_DEBUG_CHAT").is_some() {
        eprintln!("pulsar-serve: jinja prompt:\n{rendered}");
    }
    Ok(tok.encode_with_specials(&rendered))
}

/// Encode OpenAI messages as a Hy3 context: bos, system text, then per
/// turn user/assistant markers; past assistant turns carry empty think
/// tags and a trailing eos, exactly like the model's chat template.
#[cfg(target_os = "linux")]
fn encode_messages(
    tok: &tokenizer::Tokenizer,
    m: &tokenizer::ChatMarkers,
    messages: &[serde_json::Value],
    tools: Option<&Vec<serde_json::Value>>,
) -> Vec<u32> {
    let text_of = message_text_of;
    // Tool contract: schemas in the system context, calls as
    // <tool_call>{"name":...,"arguments":{...}}</tool_call> - the
    // template-agnostic convention most instruct models follow.
    let tool_text = tools.filter(|t| !t.is_empty()).map(|t| {
        let schemas: Vec<&serde_json::Value> = t.iter().map(|f| &f["function"]).collect();
        format!(
            "\n\n# Tools\n\nYou are a tool-using assistant. You MUST call a tool before answering whenever the question is about anything that can change over time (standings, rankings, news, weather, prices, current events, recent releases, live status) OR any specific external fact you are not 100% certain of from the conversation — even if you believe you already know it, because your training data may be stale. Do not ask for permission and do not mention that tools are available. Only answer directly with no tool call when the question is pure reasoning, math, or about something already settled in the conversation. To call one, output exactly:\n<tool_call>\n{{\"name\": \"<tool name>\", \"arguments\": <json arguments>}}\n</tool_call>\nYou may make multiple distinct calls in one reply, but do NOT repeat a call whose result is already in a <tool_result> block above — read that block and use it. When tool results arrive, you MUST base your answer entirely on their content — do not answer from your own knowledge for anything the tools surfaced. After all needed results arrive, stop calling tools and give the final answer immediately.\nAvailable tools (JSON Schema):\n{}",
            serde_json::to_string(&schemas).unwrap_or_default()
        )
    });
    let mut ids: Vec<u32> = m.prologue();
    ids.extend(m.prologue_effort(tok));
    let mut tools_injected = tool_text.is_none();
    // Styles that require a system block get one even when the caller sends
    // none. Harmony's channel list is not optional (gpt-oss never closes its
    // analysis channel without it) and the web UI sends no system turn, so
    // relying on the client to supply it means the model misbehaves by
    // default. Returns None for every other style.
    if !messages
        .iter()
        .any(|msg| msg["role"].as_str() == Some("system"))
    {
        if let Some(mut sys) = m.default_system() {
            if !tools_injected {
                sys.push_str(tool_text.as_deref().unwrap_or(""));
                tools_injected = true;
            }
            ids.extend(m.render_system(tok, &sys));
        }
    }
    for msg in messages {
        let role = msg["role"].as_str().unwrap_or("");
        let mut content = text_of(&msg["content"]);
        match role {
            "system" => {
                if !tools_injected {
                    content.push_str(tool_text.as_deref().unwrap_or(""));
                    tools_injected = true;
                }
                ids.extend(m.render_system(tok, &content));
            }
            "user" => {
                if !tools_injected {
                    // no system message: tools ride their own system turn
                    ids.extend(m.render_system(tok, tool_text.as_deref().unwrap_or("").trim_start()));
                    tools_injected = true;
                }
                ids.extend(m.render_user(tok, &content));
            }
            "assistant" => {
                // replay past tool calls in the dialect this family emits
                if let Some(calls) = msg["tool_calls"].as_array() {
                    let pairs: Vec<(String, String)> = calls
                        .iter()
                        .filter_map(|c| {
                            let f = &c["function"];
                            Some((
                                f["name"].as_str()?.to_string(),
                                f["arguments"].as_str().unwrap_or("{}").to_string(),
                            ))
                        })
                        .collect();
                    if !pairs.is_empty() {
                        // DeepSeek markers in vocab → DSML; Laguna → poolside XML
                        let dsml = tok.find_token("<｜User｜>").is_some()
                            || tok.find_token("<｜DSML｜tool_calls>").is_some();
                        if dsml {
                            if !content.is_empty() {
                                content.push('\n');
                            }
                            content.push_str(&tool_calls::format_dsml_tool_calls(&pairs));
                        } else if m.is_laguna() {
                            if !content.is_empty() {
                                content.push('\n');
                            }
                            content.push_str(&tool_calls::format_poolside_tool_calls(&pairs));
                        } else {
                            content.push_str(&tool_calls::format_generic_tool_calls(&pairs));
                        }
                    }
                }
                ids.extend(m.render_assistant_history(tok, &content));
            }
            "tool" => {
                let id = msg["tool_call_id"].as_str().unwrap_or("");
                let dsml = tok.find_token("<｜User｜>").is_some();
                let body = if dsml {
                    tool_calls::format_dsml_tool_result(&content)
                } else if m.is_laguna() {
                    tool_calls::format_poolside_tool_result(&content)
                } else {
                    tool_calls::format_generic_tool_result(id, &content)
                };
                ids.extend(m.render_user(tok, &body));
            }
            _ => {}
        }
    }
    ids.extend(m.open_assistant(tok));
    ids
}

/// Split generated text into (visible text, parsed tool calls).
/// Unclosed or unparseable blocks stay in the text untouched.
#[cfg(target_os = "linux")]
/// Split harmony channel output into (reasoning, reply).
///
/// gpt-oss answers on named channels: `analysis` carries its chain of
/// thought, `final` the actual reply, fenced by <|channel|>NAME<|message|>.
/// A client wants the reply in `content`, so everything that is not `final`
/// becomes reasoning, the way llama.cpp splits it. Text with no channel
/// markers passes through untouched, which is every other model here.
/// Split a reply that STARTS inside a reasoning block (GLM with thinking
/// on: `<think>` is the last prompt token, so the model emits reasoning
/// first and closes with `</think>`). Everything before the close is
/// reasoning, everything after is the reply. A reply that never closes hit
/// the token cap mid-thought - hand it back as reasoning rather than
/// returning an empty message.
fn split_open_think(s: &str) -> (String, String) {
    match s.split_once("</think>") {
        Some((think, rest)) => (think.trim().to_string(), rest.trim().to_string()),
        None => (s.trim().to_string(), String::new()),
    }
}

fn split_harmony(s: &str) -> (String, String) {
    if !s.contains("<|channel|>") {
        return (String::new(), s.to_string());
    }
    let mut reasoning = String::new();
    let mut reply = String::new();
    for seg in s.split("<|channel|>").skip(1) {
        // the name runs to the message marker, except the model sometimes
        // emits a bare ':' in its place, so accept either
        let (name, rest) = match seg.find("<|message|>") {
            Some(i) => (&seg[..i], &seg[i + "<|message|>".len()..]),
            None => match seg.find(':') {
                Some(i) => (&seg[..i], &seg[i + 1..]),
                None => (seg, ""),
            },
        };
        let mut body = rest;
        for end in ["<|end|>", "<|start|>", "<|return|>", "<|call|>"] {
            if let Some(i) = body.find(end) {
                body = &body[..i];
            }
        }
        if name.trim() == "final" {
            reply.push_str(body);
        } else {
            reasoning.push_str(body);
        }
    }
    // hit the token cap mid-reasoning and never reached `final`: show the
    // reasoning rather than hand back an empty reply
    if reply.trim().is_empty() {
        return (String::new(), reasoning.trim().to_string());
    }
    (reasoning.trim().to_string(), reply.trim().to_string())
}

fn extract_tool_calls(text: &str) -> (String, Vec<(String, String)>) {
    tool_calls::extract_tool_calls(text)
}

#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
/// Recompute the prefix-cache common-prefix length for a turn's prompt and, on
/// divergence, rewind the recurrent state to the nearest checkpoint. Same
/// logic as the initial prefix block in handle_chat, factored out so the
/// non-stream agentic loop can call it once per turn.
#[cfg(target_os = "linux")]
fn prefix_common(
    model: &engine::Model,
    st: &mut engine::State,
    hist: &mut Vec<u32>,
    cache_ok: bool,
    prompt: &[u32],
) -> engine::Result<usize> {
    if !cache_ok {
        return Ok(0);
    }
    let mut common = hist.iter().zip(prompt.iter()).take_while(|(a, b)| a == b).count();
    let recurrent = model.recurrent_state();
    if recurrent && (common < hist.len() || common == prompt.len()) {
        let target = common.min(prompt.len() - 1) as u32;
        match st.restore_nearest_ckpt(model, target)? {
            Some(c) => {
                hist.truncate(c as usize);
                common = c as usize;
            }
            None => common = 0,
        }
    } else if common == prompt.len() {
        common -= 1;
    }
    if common == 0 {
        hist.clear();
    }
    Ok(common)
}

#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)] // one call site; the request fields do not want a struct
fn handle_chat(
    stream: &mut std::net::TcpStream,
    body: &[u8],
    model: &engine::Model,
    tok: &tokenizer::Tokenizer,
    markers: &tokenizer::ChatMarkers,
    chat_template: Option<&tokenizer::ResolvedChatTemplate>,
    jinja_chat: bool,
    st: &mut engine::State,
    model_name: &str,
    default_temp: f32,
    request_id: u64,
    hist: &mut Vec<u32>,
    mcp: Option<&mcp::McpHub>,
) -> engine::Result {
    use std::io::Write;

    let req: serde_json::Value = serde_json::from_slice(body)?;
    let messages = req["messages"]
        .as_array()
        .ok_or("chat request needs a messages array")?;
    let temp = req["temperature"].as_f64().map(|v| v as f32).unwrap_or(default_temp);
    let top_p = req["top_p"].as_f64().map(|v| v as f32).unwrap_or(1.0);
    let min_p = req["min_p"].as_f64().map(|v| v as f32).unwrap_or(0.0);
    let seed = req["seed"].as_u64().unwrap_or(42);
    // MCP agentic loop is non-stream only. Force that when tools are enabled
    // so DSML/Hy3 tool markup is never streamed into the chat bubble before
    // extract_tool_calls can strip it.
    let want_stream = req["stream"].as_bool().unwrap_or(false);
    let streaming = want_stream
        && !mcp
            .map(|m| m.has_enabled_tools())
            .unwrap_or(false);

    // Per-request reasoning control, accepting both conventions clients
    // actually send: OpenAI's top-level `reasoning_effort` and the
    // vLLM/SGLang `chat_template_kwargs.enable_thinking`. "none"/"off"
    // disables; anything else is clamped to the style's own vocabulary
    // (GLM high|max, harmony low|medium|high) by set_reasoning. Markers
    // are cloned per request so one client's choice cannot leak into the
    // next - the server holds one engine but many callers.
    let mut req_markers = markers.clone();
    if let Some(e) = req["reasoning_effort"].as_str() {
        match e {
            "none" | "off" | "disabled" => req_markers.set_think(false),
            _ => {
                req_markers.set_think(true);
                req_markers.set_reasoning(e);
            }
        }
    }
    if let Some(b) = req["chat_template_kwargs"]["enable_thinking"].as_bool() {
        req_markers.set_think(b);
    }
    // Jinja templates carry their own defaults when the client omits the
    // kwarg. Laguna's official template defaults enable_thinking=true and
    // opens with `<assistant><think>` — align markers so stream/non-stream
    // split treats the reply as open-think (otherwise `</think>` leaks into
    // content: e.g. `NAT</think>NAT`). Web UI omits the field when the
    // thinking checkbox is on ("let checkpoint decide").
    let enable_thinking = req["chat_template_kwargs"]["enable_thinking"].as_bool();
    if jinja_chat && enable_thinking.is_none() && req_markers.is_laguna() {
        req_markers.set_think(true);
    }
    let markers = &req_markers;

    // Merge client-supplied tools with any enabled MCP tools (namespaced
    // `server__tool`). When MCP is off or has no enabled tools this is exactly
    // the client list (or None), so non-MCP requests are byte-identical to today.
    let mut tools_vec: Vec<serde_json::Value> =
        req["tools"].as_array().cloned().unwrap_or_default();
    if let Some(m) = mcp {
        if m.has_enabled_tools() {
            tools_vec.extend(m.enabled_tools_as_openai());
        }
    }
    let tools = if tools_vec.is_empty() { None } else { Some(tools_vec) };
    // When Jinja + Laguna and client omitted the kwarg, pass true so apply
    // matches opens_thinking / stream split (template default is true).
    let enable_thinking = enable_thinking.or_else(|| {
        if jinja_chat && markers.is_laguna() {
            Some(true)
        } else {
            None
        }
    });
    let reasoning_effort = req["reasoning_effort"].as_str().map(|s| s.to_string());
    let encode = |msgs: &[serde_json::Value]| {
        encode_messages_auto(
            tok,
            markers,
            chat_template,
            jinja_chat,
            msgs,
            tools.as_ref(),
            enable_thinking,
            reasoning_effort.as_deref(),
        )
    };
    let prompt = encode(messages);
    if std::env::var_os("PULSAR_DEBUG_IDS").is_some() {
        eprintln!("pulsar-serve: prompt ids {prompt:?}");
    }
    if prompt.len() as u32 + 2 >= st.ctx() {
        eprintln!(
            "pulsar-serve: req {request_id}: rejected, prompt {} tokens vs ctx {}",
            prompt.len(),
            st.ctx()
        );
        return respond_json(
            stream,
            400,
            &serde_json::json!({"error": {"message": format!("prompt exceeds context ({} tokens, ctx {})", prompt.len(), st.ctx())}}),
        );
    }
    // Default and ceiling are the remaining context, not a fixed number: a
    // reasoning model spends 1k+ tokens thinking before its final channel,
    // and a mid-thought cap hands the client a truncated monologue and no
    // answer. An explicit max_tokens is honoured but still clamped to what
    // the KV can hold.
    let room = (st.ctx() as usize).saturating_sub(prompt.len() + 2);
    let max_tokens = req["max_tokens"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or(room)
        .min(room);
    let mut sampler = engine::Sampler::new(temp, top_p, min_p, seed);
    let id = format!("chatcmpl-{request_id}");
    // `created` is a required member of both chat.completion and
    // chat.completion.chunk. Clients that deserialize into a struct with
    // non-optional fields reject the whole turn without it (issue #20/#21),
    // and lenient ones simply ignored that it was missing. Stamped once per
    // request and reused across every chunk of the stream, which is what
    // upstream does: the value marks when the completion started, not when
    // each chunk was flushed.
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Prefix cache: skip re-prefilling whatever the engine already holds.
    // Chat transcripts APPEND, so the common case reuses everything up to
    // the new turn (and the constant system prompt survives across
    // sessions while the server stays up). Recurrent-state families may
    // only extend the exact forwarded stream; pure-KV families can rewind
    // to the divergence and overwrite. Speculative modes rewrite KV in
    // ways this bookkeeping does not model - caching disables itself.
    let cache_ok = model.mtp_depth == 0
        && std::env::var_os("PULSAR_NGRAM").is_none()
        && std::env::var_os("PULSAR_NO_PREFIX_CACHE").is_none();
    let mut common = 0usize;
    if cache_ok {
        common = hist.iter().zip(prompt.iter()).take_while(|(a, b)| a == b).count();
        let recurrent = model.recurrent_state();
        if recurrent && (common < hist.len() || common == prompt.len()) {
            // recurrent state can only extend the exact stream: on a
            // divergence (or full replay) rewind to the nearest prefix
            // checkpoint instead of position 0
            let target = common.min(prompt.len() - 1) as u32;
            match st.restore_nearest_ckpt(model, target)? {
                Some(c) => {
                    eprintln!("pulsar-serve: {id}: rewound to checkpoint @{c}");
                    hist.truncate(c as usize);
                    common = c as usize;
                }
                None => common = 0,
            }
        } else if common == prompt.len() {
            // fully-cached prompt still needs one forward for logits
            common -= 1;
        }
    }
    if common == 0 {
        hist.clear(); // pos0 == 0 resets recurrent state in the engine
    } else {
        eprintln!("pulsar-serve: {id}: prefix cache hit, {common}/{} tokens reused", prompt.len());
    }
    let stop_seen = std::cell::Cell::new(None::<u32>);
    let tool_phase = std::cell::Cell::new(false);
    let mut emitted: Vec<u32> = Vec::new();

    if streaming {
        write!(
            stream,
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncache-control: no-cache\r\nconnection: close\r\n\r\n"
        )?;
        stream.flush()?;
        // First byte immediately: proxies buffer the downstream response
        // until real SSE data arrives, so a silent prefill looks like a
        // dead upstream. The role chunk is protocol-required anyway.
        let first = serde_json::json!({
            "id": id, "object": "chat.completion.chunk", "model": model_name, "created": created,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}],
            // prompt token count up front so the UI can show live context use
            // and prefill tok/s the moment the first decoded token lands.
            // completion/total are required members of the usage object -
            // clients that deserialize into a struct with non-optional fields
            // (grok-cli, the strict Go/Rust SDKs) hard-error on a partial one
            // rather than ignoring it, so send all three and let the zeros
            // stand for "nothing generated yet".
            "usage": {
                "prompt_tokens": prompt.len(),
                "completion_tokens": 0,
                "total_tokens": prompt.len(),
            },
        });
        write!(stream, "data: {first}\n\n")?;
        stream.flush()?;
        // Long prefills are silent for minutes; proxies kill idle reads.
        // A side thread drips SSE comments until the first token lands
        // (a comment between events is legal, clients ignore it).
        let ka_started = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ka_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        // set when a keepalive write fails = the client is gone; the
        // generate loop polls it and abandons the work
        let ka_dead = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ka_thread = {
            let started = ka_started.clone();
            let stop = ka_stop.clone();
            let dead = ka_dead.clone();
            let mut ks = stream.try_clone()?;
            std::thread::spawn(move || {
                use std::sync::atomic::Ordering;
                loop {
                    for _ in 0..15 {
                        std::thread::sleep(std::time::Duration::from_secs(1));
                        if stop.load(Ordering::Relaxed) || started.load(Ordering::Relaxed) {
                            return;
                        }
                    }
                    if ks.write_all(b": prefill keepalive\n\n").and_then(|_| ks.flush()).is_err() {
                        dead.store(true, Ordering::Relaxed);
                        return;
                    }
                }
            })
        };
        let mut bytes: Vec<u8> = Vec::new();
        let mut n_out = 0usize;
        let send_err = std::cell::Cell::new(false);
        // harmony channel state machine: the assistant turn is
        // <|channel|>analysis<|message|>...<|end|><|start|>assistant
        // <|channel|>final<|message|>... - header text (channel names,
        // roles) is swallowed, `final` bodies stream as content, every
        // other body as reasoning_content, mirroring the non-stream split
        let harmony = markers.is_harmony();
        let mut hdr = harmony; // generation opens on a channel header
        let mut hdr_buf: Vec<u8> = Vec::new();
        // GLM opens the think block in the PROMPT, so the stream begins
        // inside reasoning and the first </think> ends it.
        let open_think = markers.opens_thinking();
        let mut reasoning = open_think;
        let mut rbytes: Vec<u8> = Vec::new();
        engine::generate_cancellable(
            model,
            st,
            &prompt[common..],
            common as u32,
            &mut sampler,
            max_tokens,
            |t| {
                let s = markers.is_stop(t);
                if s {
                    stop_seen.set(Some(t));
                }
                s
            },
            |t| {
                ka_started.store(true, std::sync::atomic::Ordering::Relaxed);
                n_out += 1;
                emitted.push(t);
                if tool_phase.get() {
                    return; // buffering a tool call; nothing streams
                }
                {
                    let d = tok.decode(&[t]);
                    const FENCE: [&[u8]; 5] = [
                        b"<|channel|>",
                        b"<|message|>",
                        b"<|start|>",
                        b"<|end|>",
                        b"<|constrain|>",
                    ];
                    if harmony {
                        match d.as_slice() {
                            b"<|channel|>" | b"<|start|>" | b"<|end|>" => {
                                hdr = true;
                                hdr_buf.clear();
                            }
                            b"<|message|>" => {
                                hdr = false;
                                reasoning = !hdr_buf.windows(5).any(|w| w == b"final");
                            }
                            b"<|constrain|>" => {}
                            _ if hdr => hdr_buf.extend_from_slice(&d),
                            _ if reasoning => rbytes.extend_from_slice(&d),
                            _ => bytes.extend_from_slice(&d),
                        }
                    } else if open_think && d.as_slice() == b"</think>" {
                        reasoning = false; // close: the reply starts here
                    } else if open_think && reasoning {
                        rbytes.extend_from_slice(&d);
                    } else if !FENCE.contains(&d.as_slice()) {
                        bytes.extend_from_slice(&d);
                    }
                }
                let rvalid = match std::str::from_utf8(&rbytes) {
                    Ok(s) => s.len(),
                    Err(e) => e.valid_up_to(),
                };
                if rvalid > 0 && !send_err.get() {
                    let text = String::from_utf8_lossy(&rbytes[..rvalid]).into_owned();
                    rbytes.drain(..rvalid);
                    let chunk = serde_json::json!({
                        "id": id, "object": "chat.completion.chunk", "model": model_name, "created": created,
                        "choices": [{"index": 0, "delta": {"reasoning_content": text}, "finish_reason": null}],
                    });
                    if write!(stream, "data: {chunk}\n\n").and_then(|_| stream.flush()).is_err() {
                        send_err.set(true);
                    }
                }
                if let Some(p) = tool_calls::find_tool_open(&bytes) {
                    // stream the text before the call, then go silent
                    // (covers generic JSON, Hy3 opensource, DeepSeek DSML)
                    bytes.truncate(p);
                    tool_phase.set(true);
                }
                let mut valid = match std::str::from_utf8(&bytes) {
                    Ok(s) => s.len(),
                    Err(e) => e.valid_up_to(),
                };
                if !tool_phase.get() {
                    // hold back ONLY a tail that is itself a prefix of a
                    // tool-open marker, so ordinary text streams immediately
                    let hold = tool_calls::tool_open_holdback(&bytes);
                    valid = valid.min(bytes.len() - hold);
                }
                if valid > 0 && !send_err.get() {
                    let text = String::from_utf8_lossy(&bytes[..valid]).into_owned();
                    bytes.drain(..valid);
                    let chunk = serde_json::json!({
                        "id": id, "object": "chat.completion.chunk", "model": model_name, "created": created,
                        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}],
                    });
                    if write!(stream, "data: {chunk}\n\n").and_then(|_| stream.flush()).is_err() {
                        send_err.set(true);
                    }
                }
            },
            || {
                ka_dead.load(std::sync::atomic::Ordering::Relaxed) || send_err.get()
            },
        )?;
        ka_stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = ka_thread.join();
        // flush the marker holdback: without this a reply shorter than
        // the <tool_call> window streams as empty
        if !tool_phase.get() && !bytes.is_empty() && !send_err.get() {
            let text = String::from_utf8_lossy(&bytes).into_owned();
            let chunk = serde_json::json!({
                "id": id, "object": "chat.completion.chunk", "model": model_name, "created": created,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}],
            });
            let _ = write!(stream, "data: {chunk}\n\n").and_then(|_| stream.flush());
        }
        let full = String::from_utf8_lossy(&tok.decode(&emitted)).into_owned();
        let (_, calls) = extract_tool_calls(&full);
        for (ci, (name, args)) in calls.iter().enumerate() {
            let tc = serde_json::json!({
                "id": id, "object": "chat.completion.chunk", "model": model_name, "created": created,
                "choices": [{"index": 0, "delta": {"tool_calls": [{
                    "index": ci, "id": format!("call_{request_id}_{ci}"),
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }]}, "finish_reason": null}],
            });
            let _ = write!(stream, "data: {tc}\n\n");
        }
        let fin_reason = if calls.is_empty() { "stop" } else { "tool_calls" };
        let fin = serde_json::json!({
            "id": id, "object": "chat.completion.chunk", "model": model_name, "created": created,
            "choices": [{"index": 0, "delta": {}, "finish_reason": fin_reason}],
            "usage": {
                "prompt_tokens": prompt.len(),
                "completion_tokens": n_out,
                "total_tokens": prompt.len() + n_out,
            },
        });
        let _ = write!(stream, "data: {fin}\n\ndata: [DONE]\n\n");
        let _ = stream.flush();
        eprintln!("pulsar-serve: {id}: {} prompt + {n_out} completion tokens (streamed)", prompt.len());
        if cache_ok {
            *hist = prompt;
            hist.extend(&emitted);
            hist.extend(stop_seen.get());
        }
    } else {
        // ponytail: agentic loop is non-stream only. Each turn re-encodes the
        // full conversation — the prefix cache makes turn N+1 cheap because
        // hist already holds turn N's prompt+emitted, so only the suffix
        // re-prefills. When the model emits tool calls AND an MCP hub is
        // attached, the calls are dispatched, their results appended as
        // `tool` messages, and the loop regenerates. Without MCP (or once the
        // model stops calling tools) the loop degenerates to one turn and the
        // response is byte-identical to the pre-MCP server.
        const MAX_TURNS: usize = 8;
        let mut msgs: Vec<serde_json::Value> = messages.to_vec();
        let mut clean = String::new();
        let mut reasoning = String::new();
        let mut calls: Vec<(String, String)> = Vec::new();
        let mut n_out_total = 0usize;
        let mut prompt_len = prompt.len();
        let mut last_finish = "stop";
        let mut prev_calls: Vec<(String, String)> = Vec::new();
        let mut empty_nudge_used = false;
        for turn in 0..MAX_TURNS {
            let tp = encode(&msgs);
            if tp.len() as u32 + 2 >= st.ctx() {
                eprintln!(
                    "pulsar-serve: {id}: context exceeded after tool turn {turn} ({} tokens)",
                    tp.len()
                );
                break;
            }
            let common = prefix_common(model, st, hist, cache_ok, &tp)?;
            let room = (st.ctx() as usize).saturating_sub(tp.len() + 2);
            let max_t = max_tokens.min(room);
            let mut out: Vec<u8> = Vec::new();
            let mut emitted_t: Vec<u32> = Vec::new();
            engine::generate(
                model,
                st,
                &tp[common..],
                common as u32,
                &mut sampler,
                max_t,
                |t| {
                    let s = markers.is_stop(t);
                    if s {
                        stop_seen.set(Some(t));
                    }
                    s
                },
                |t| {
                    n_out_total += 1;
                    emitted_t.push(t);
                    out.extend_from_slice(&tok.decode(&[t]));
                },
            )?;
            prompt_len = tp.len();
            if cache_ok {
                *hist = tp;
                hist.extend(&emitted_t);
                hist.extend(stop_seen.get());
                stop_seen.set(None);
            }
            let full = String::from_utf8_lossy(&out).into_owned();
            let (c, this_calls) = extract_tool_calls(&full);
            if !this_calls.is_empty() {
                eprintln!(
                    "pulsar-serve: {id}: extracted {} tool call(s): {}",
                    this_calls.len(),
                    this_calls
                        .iter()
                        .map(|(n, _)| n.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            } else if full.contains("DSML")
                || full.contains("tool_calls:opensource")
                || full.contains("<tool_call>")
                || full.contains("<arg_key>")
            {
                eprintln!(
                    "pulsar-serve: {id}: tool-like markup present but parse yielded 0 calls ({} bytes)",
                    full.len()
                );
            }
            let (r, c2) = if markers.opens_thinking() {
                split_open_think(&c)
            } else {
                split_harmony(&c)
            };
            clean = c2;
            reasoning = r;
            calls = this_calls;
            if calls.is_empty() {
                // DeepSeek often ends a tool-only turn with empty clean text;
                // after MCP results it may EOS without a final answer if the
                // history dialect was wrong, or still need one nudge.
                if clean.trim().is_empty() && turn > 0 && !empty_nudge_used {
                    empty_nudge_used = true;
                    eprintln!(
                        "pulsar-serve: {id}: empty final after tools (turn {turn}); nudging once"
                    );
                    msgs.push(serde_json::json!({
                        "role": "user",
                        "content": "Based on the tool results above, give a concise final answer now. Do not call tools again.",
                    }));
                    continue;
                }
                last_finish = "stop";
                break;
            }
            // ponytail: collapse infinite-loop where the model repeats an
            // already-dispatched call verbatim instead of reading the
            // <tool_result> above. Break with the last text as the answer;
            // upgrade path: a tool-result-aware duplicate check (same name
            // + args, ignoring prior failed results) instead of exact match.
            if turn > 0 && calls.len() == prev_calls.len()
                && calls.iter().zip(prev_calls.iter()).all(|(a, b)| a == b)
            {
                eprintln!(
                    "pulsar-serve: {id}: tool loop stalled at turn {turn} (repeated call), forcing final answer");
                last_finish = "stop";
                break;
            }
            let Some(m) = mcp else {
                // calls but no hub: hand them to the client unchanged
                last_finish = "tool_calls";
                break;
            };
            let tool_calls_json: Vec<serde_json::Value> = calls
                .iter()
                .enumerate()
                .map(|(ci, (name, args))| serde_json::json!({
                    "id": format!("call_{request_id}_{turn}_{ci}"),
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }))
                .collect();
            msgs.push(serde_json::json!({
                "role": "assistant",
                "content": clean.clone(),
                "tool_calls": tool_calls_json,
            }));
            for (ci, (name, args)) in calls.iter().enumerate() {
                let call_id = format!("call_{request_id}_{turn}_{ci}");
                let result = m.dispatch_sync(name, args);
                eprintln!("pulsar-serve: {id}: mcp dispatch {name}");
                msgs.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result,
                }));
            }
            last_finish = "tool_calls";
            prev_calls = calls.to_vec();
        }
        // Never leave the web UI with (empty) after a successful tool loop:
        // prefer reasoning text, else surface the last tool payload.
        if clean.trim().is_empty() {
            if !reasoning.trim().is_empty() {
                clean = std::mem::take(&mut reasoning);
            } else {
                let tool_texts: Vec<String> = msgs
                    .iter()
                    .rev()
                    .filter(|m| m["role"].as_str() == Some("tool"))
                    .filter_map(|m| m["content"].as_str().map(str::to_owned))
                    .take(3)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect();
                if !tool_texts.is_empty() {
                    eprintln!(
                        "pulsar-serve: {id}: final content still empty; returning last tool result(s)"
                    );
                    clean = tool_texts.join("\n\n");
                }
            }
        }
        let mut message = serde_json::json!({"role": "assistant", "content": clean});
        if !reasoning.is_empty() {
            message["reasoning_content"] = serde_json::json!(reasoning);
        }
        // Final reply carries tool_calls only when the loop ended with
        // un-dispatched calls (no MCP hub). When the hub ran them, the
        // model's final turn is a plain answer.
        if last_finish == "tool_calls" && !calls.is_empty() {
            message["tool_calls"] = serde_json::json!(calls
                .iter()
                .enumerate()
                .map(|(ci, (name, args))| serde_json::json!({
                    "id": format!("call_{request_id}_{ci}"),
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }))
                .collect::<Vec<_>>());
        }
        let json = serde_json::json!({
            "id": id, "object": "chat.completion", "model": model_name, "created": created,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": last_finish,
            }],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": n_out_total,
                "total_tokens": prompt_len + n_out_total,
            },
        });
        eprintln!(
            "pulsar-serve: {id}: {prompt_len} prompt + {n_out_total} completion tokens (non-stream)"
        );
        respond_json(stream, 200, &json)?;
    }
    Ok(())
}
