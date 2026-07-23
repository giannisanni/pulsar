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

#[cfg(target_os = "linux")]
fn run() -> engine::Result {
    use std::io::{BufRead, BufReader, Read, Write};

    let mut model_path = None;
    let mut port = 11435u16;
    let mut host = String::from("127.0.0.1");
    let mut ctx = 8192u32;
    let mut prefix_file: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut need = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match a.as_str() {
            "-m" => model_path = Some(need("-m")?),
            "--port" => port = need("--port")?.parse()?,
            "--host" => host = need("--host")?.to_string(),
            "--ctx" => ctx = need("--ctx")?.parse()?,
            "--prefix-file" => prefix_file = Some(need("--prefix-file")?),
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
    let markers = tokenizer::ChatMarkers::resolve(&tok)?;
    let mut st = engine::State::new(&model, ctx)?;
    let default_temp = model
        .gguf
        .metadata
        .get("general.sampling.temp")
        .and_then(gguf::Value::as_f32)
        .unwrap_or(0.9);

    let listener = std::net::TcpListener::bind((host.as_str(), port))?;
    eprintln!("pulsar-serve: listening on http://{host}:{port}  (web UI at /, API at /v1)");

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
            loop {
                let mut line = String::new();
                reader.read_line(&mut line)?;
                let line = line.trim();
                if line.is_empty() {
                    break;
                }
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    content_length = v.trim().parse().unwrap_or(0);
                }
            }
            let mut body = vec![0u8; content_length];
            reader.read_exact(&mut body)?;

            match (method.as_str(), path.as_str()) {
                ("GET", "/") | ("GET", "/index.html") => {
                    respond_bytes(&mut stream, 200, "text/html; charset=utf-8", WEBUI_HTML.as_bytes())
                }
                ("GET", "/favicon.ico") | ("GET", "/favicon.svg") => {
                    respond_bytes(&mut stream, 200, "image/svg+xml", FAVICON_SVG.as_bytes())
                }
                ("GET", "/v1/models") => {
                    let json = serde_json::json!({
                        "object": "list",
                        "data": [{"id": model_name, "object": "model", "owned_by": "pulsar", "max_model_len": ctx}],
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
                    let gpu_names = gpu_names();
                    let gpus: Vec<_> = s
                        .gpus
                        .iter()
                        .enumerate()
                        .map(|(i, g)| serde_json::json!({
                            "dev": i,
                            "name": gpu_names.get(i).cloned().unwrap_or_default(),
                            "vram_used": g.vram_total.saturating_sub(g.vram_free),
                            "vram_total": g.vram_total,
                        }))
                        .collect();
                    // model residency: VRAM (resident tiers + slab cache), RAM (host
                    // cache), disk (the streamed remainder of the gguf on disk)
                    let model_bytes = std::fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0);
                    let vram = s.vram_resident as u64;
                    let ram = s.host_used as u64;
                    let disk = model_bytes.saturating_sub(vram + ram);
                    let json = serde_json::json!({
                        "model": model_name,
                        "ctx": s.ctx,
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
                    let mut models: Vec<_> = std::fs::read_dir(&dir)
                        .into_iter()
                        .flatten()
                        .flatten()
                        .filter_map(|e| {
                            let name = e.file_name().to_string_lossy().into_owned();
                            // skip non-models: MTP draft sidecars aren't standalone
                            if !name.ends_with(".gguf") || name.contains("draft") {
                                return None;
                            }
                            let bytes = e.metadata().map(|m| m.len()).unwrap_or(0);
                            Some(serde_json::json!({"id": name, "bytes": bytes, "current": name == cur}))
                        })
                        .collect();
                    models.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
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
                    let target = dir.join(name);
                    let ok = !name.is_empty()
                        && !name.contains('/')
                        && name.ends_with(".gguf")
                        && target.is_file();
                    if !ok {
                        respond_json(&mut stream, 400, &serde_json::json!({"error": {"message": "invalid or unknown model"}}))
                    } else {
                        respond_json(&mut stream, 200, &serde_json::json!({"reloading": name}))?;
                        let _ = std::io::Write::flush(&mut stream);
                        eprintln!("pulsar-serve: switching model -> {name}, re-exec");
                        let err = reexec_with_model(&target);
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
                    &mut st,
                    &model_name,
                    default_temp,
                    request_id,
                    &mut hist,
                ),
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
/// the caller to be a .gguf inside the current model's directory.
#[cfg(target_os = "linux")]
fn reexec_with_model(newmodel: &std::path::Path) -> std::io::Error {
    use std::os::unix::process::CommandExt;
    let args: Vec<String> = std::env::args().collect();
    let mut cmd = std::process::Command::new(&args[0]);
    let mut i = 1;
    while i < args.len() {
        if args[i] == "-m" || args[i] == "--model" {
            cmd.arg(&args[i]).arg(newmodel);
            i += 2;
        } else {
            cmd.arg(&args[i]);
            i += 1;
        }
    }
    cmd.exec()
}

/// GPU names by device index, from nvidia-smi (empty if unavailable). VRAM
/// used/total come from the engine's cudaMemGetInfo; nvidia-smi supplies names.
fn gpu_names() -> Vec<String> {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .map(|l| l.trim().to_string())
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
    // content arrives as a plain string OR an array of typed blocks
    // (Claude Code / Anthropic-translated clients send
    // [{type:"text", text:...}, ...]); a string-only read silently
    // dropped the whole system prompt for those clients
    fn text_of(content: &serde_json::Value) -> String {
        match content {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(blocks) => blocks
                .iter()
                .map(|b| {
                    if let Some(t) = b["text"].as_str() {
                        t.to_string()
                    } else if b["type"].as_str() == Some("tool_result") {
                        text_of(&b["content"])
                    } else {
                        String::new()
                    }
                })
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        }
    }
    // Tool contract: schemas in the system context, calls as
    // <tool_call>{"name":...,"arguments":{...}}</tool_call> - the
    // template-agnostic convention most instruct models follow.
    let tool_text = tools.filter(|t| !t.is_empty()).map(|t| {
        let schemas: Vec<&serde_json::Value> = t.iter().map(|f| &f["function"]).collect();
        format!(
            "\n\n# Tools\n\nYou may call tools. To call one, output exactly:\n<tool_call>\n{{\"name\": \"<tool name>\", \"arguments\": <json arguments>}}\n</tool_call>\nYou may make multiple calls in one reply. After tool results arrive, continue the task.\nAvailable tools (JSON Schema):\n{}",
            serde_json::to_string(&schemas).unwrap_or_default()
        )
    });
    let mut ids: Vec<u32> = m.prologue();
    let mut tools_injected = tool_text.is_none();
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
                // replay past tool calls in the same syntax the model emits
                if let Some(calls) = msg["tool_calls"].as_array() {
                    for c in calls {
                        let f = &c["function"];
                        content.push_str(&format!(
                            "\n<tool_call>\n{{\"name\": {}, \"arguments\": {}}}\n</tool_call>",
                            f["name"],
                            f["arguments"].as_str().unwrap_or("{}")
                        ));
                    }
                }
                ids.extend(m.render_assistant_history(tok, &content));
            }
            "tool" => {
                let id = msg["tool_call_id"].as_str().unwrap_or("");
                ids.extend(m.render_user(
                    tok,
                    &format!("<tool_result id=\"{id}\">\n{content}\n</tool_result>"),
                ));
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
fn extract_tool_calls(text: &str) -> (String, Vec<(String, String)>) {
    let mut clean = String::new();
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        let (before, after) = rest.split_at(start);
        clean.push_str(before);
        let body = &after["<tool_call>".len()..];
        let Some(end) = body.find("</tool_call>") else {
            clean.push_str(after);
            rest = "";
            break;
        };
        let block = body[..end].trim();
        match serde_json::from_str::<serde_json::Value>(block) {
            Ok(v) if v["name"].is_string() => {
                let args = if v["arguments"].is_null() {
                    "{}".to_string()
                } else {
                    v["arguments"].to_string()
                };
                calls.push((v["name"].as_str().unwrap_or("").to_string(), args));
            }
            _ => {
                clean.push_str(&after[..("<tool_call>".len() + end + "</tool_call>".len())]);
            }
        }
        rest = &body[end + "</tool_call>".len()..];
    }
    clean.push_str(rest);
    (clean.trim_end().to_string(), calls)
}

#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn handle_chat(
    stream: &mut std::net::TcpStream,
    body: &[u8],
    model: &engine::Model,
    tok: &tokenizer::Tokenizer,
    markers: &tokenizer::ChatMarkers,
    st: &mut engine::State,
    model_name: &str,
    default_temp: f32,
    request_id: u64,
    hist: &mut Vec<u32>,
) -> engine::Result {
    use std::io::Write;

    let req: serde_json::Value = serde_json::from_slice(body)?;
    let messages = req["messages"]
        .as_array()
        .ok_or("chat request needs a messages array")?;
    let temp = req["temperature"].as_f64().map(|v| v as f32).unwrap_or(default_temp);
    let top_p = req["top_p"].as_f64().map(|v| v as f32).unwrap_or(1.0);
    let min_p = req["min_p"].as_f64().map(|v| v as f32).unwrap_or(0.0);
    let max_tokens = req["max_tokens"].as_u64().unwrap_or(1024) as usize;
    let seed = req["seed"].as_u64().unwrap_or(42);
    let streaming = req["stream"].as_bool().unwrap_or(false);

    let tools = req["tools"].as_array().cloned();
    let prompt = encode_messages(tok, markers, messages, tools.as_ref());
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
    let mut sampler = engine::Sampler::new(temp, top_p, min_p, seed);
    let id = format!("chatcmpl-{request_id}");

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
            "id": id, "object": "chat.completion.chunk", "model": model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}],
            // prompt token count up front so the UI can show live context use
            // and prefill tok/s the moment the first decoded token lands
            "usage": {"prompt_tokens": prompt.len()},
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
                bytes.extend_from_slice(&tok.decode(&[t]));
                const MARK: &[u8] = b"<tool_call>";
                if let Some(p) = bytes.windows(MARK.len()).position(|w| w == MARK) {
                    // stream the text before the call, then go silent
                    bytes.truncate(p);
                    tool_phase.set(true);
                }
                let mut valid = match std::str::from_utf8(&bytes) {
                    Ok(s) => s.len(),
                    Err(e) => e.valid_up_to(),
                };
                if !tool_phase.get() {
                    // hold back a potential marker prefix split across tokens
                    valid = valid.min(bytes.len().saturating_sub(MARK.len() - 1));
                }
                if valid > 0 && !send_err.get() {
                    let text = String::from_utf8_lossy(&bytes[..valid]).into_owned();
                    bytes.drain(..valid);
                    let chunk = serde_json::json!({
                        "id": id, "object": "chat.completion.chunk", "model": model_name,
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
                "id": id, "object": "chat.completion.chunk", "model": model_name,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}],
            });
            let _ = write!(stream, "data: {chunk}\n\n").and_then(|_| stream.flush());
        }
        let full = String::from_utf8_lossy(&tok.decode(&emitted)).into_owned();
        let (_, calls) = extract_tool_calls(&full);
        for (ci, (name, args)) in calls.iter().enumerate() {
            let tc = serde_json::json!({
                "id": id, "object": "chat.completion.chunk", "model": model_name,
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
            "id": id, "object": "chat.completion.chunk", "model": model_name,
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
        let mut out: Vec<u8> = Vec::new();
        let mut n_out = 0usize;
        engine::generate(
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
                n_out += 1;
                emitted.push(t);
                out.extend_from_slice(&tok.decode(&[t]));
            },
        )?;
        let full = String::from_utf8_lossy(&out).into_owned();
        let (clean, calls) = extract_tool_calls(&full);
        let mut message = serde_json::json!({"role": "assistant", "content": clean});
        if !calls.is_empty() {
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
            "id": id, "object": "chat.completion", "model": model_name,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": if calls.is_empty() { "stop" } else { "tool_calls" },
            }],
            "usage": {
                "prompt_tokens": prompt.len(),
                "completion_tokens": n_out,
                "total_tokens": prompt.len() + n_out,
            },
        });
        eprintln!("pulsar-serve: {id}: {} prompt + {n_out} completion tokens", prompt.len());
        respond_json(stream, 200, &json)?;
        if cache_ok {
            *hist = prompt;
            hist.extend(&emitted);
            hist.extend(stop_seen.get());
        }
    }
    Ok(())
}
