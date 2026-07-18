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

#[cfg(target_os = "linux")]
fn run() -> engine::Result {
    use std::io::{BufRead, BufReader, Read, Write};

    let mut model_path = None;
    let mut port = 11435u16;
    let mut host = String::from("127.0.0.1");
    let mut ctx = 8192u32;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut need = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match a.as_str() {
            "-m" => model_path = Some(need("-m")?),
            "--port" => port = need("--port")?.parse()?,
            "--host" => host = need("--host")?.to_string(),
            "--ctx" => ctx = need("--ctx")?.parse()?,
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
    eprintln!("pulsar-serve: listening on http://{host}:{port}/v1");

    let mut request_id = 0u64;
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
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
                ("GET", "/v1/models") => {
                    let json = serde_json::json!({
                        "object": "list",
                        "data": [{"id": model_name, "object": "model", "owned_by": "pulsar"}],
                    });
                    respond_json(&mut stream, 200, &json)
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

/// Encode OpenAI messages as a Hy3 context: bos, system text, then per
/// turn user/assistant markers; past assistant turns carry empty think
/// tags and a trailing eos, exactly like the model's chat template.
#[cfg(target_os = "linux")]
fn encode_messages(
    tok: &tokenizer::Tokenizer,
    m: &tokenizer::ChatMarkers,
    messages: &[serde_json::Value],
) -> Vec<u32> {
    let mut ids: Vec<u32> = m.prologue();
    for msg in messages {
        let role = msg["role"].as_str().unwrap_or("");
        let content = msg["content"].as_str().unwrap_or("");
        match role {
            "system" => ids.extend(m.render_system(tok, content)),
            "user" => ids.extend(m.render_user(tok, content)),
            "assistant" => ids.extend(m.render_assistant_history(tok, content)),
            _ => {}
        }
    }
    ids.extend(m.open_assistant(tok));
    ids
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
) -> engine::Result {
    use std::io::Write;

    let req: serde_json::Value = serde_json::from_slice(body)?;
    let messages = req["messages"]
        .as_array()
        .ok_or("chat request needs a messages array")?;
    let temp = req["temperature"].as_f64().map(|v| v as f32).unwrap_or(default_temp);
    let top_p = req["top_p"].as_f64().map(|v| v as f32).unwrap_or(1.0);
    let max_tokens = req["max_tokens"].as_u64().unwrap_or(1024) as usize;
    let seed = req["seed"].as_u64().unwrap_or(42);
    let streaming = req["stream"].as_bool().unwrap_or(false);

    let prompt = encode_messages(tok, markers, messages);
    if std::env::var_os("PULSAR_DEBUG_IDS").is_some() {
        eprintln!("pulsar-serve: prompt ids {prompt:?}");
    }
    if prompt.len() as u32 + 2 >= st.ctx() {
        return respond_json(
            stream,
            400,
            &serde_json::json!({"error": {"message": "prompt exceeds context"}}),
        );
    }
    let mut sampler = engine::Sampler::new(temp, top_p, 0.0, seed);
    let id = format!("chatcmpl-{request_id}");

    if streaming {
        write!(
            stream,
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncache-control: no-cache\r\nconnection: close\r\n\r\n"
        )?;
        let mut bytes: Vec<u8> = Vec::new();
        let mut n_out = 0usize;
        let mut send_err = None;
        engine::generate(
            model,
            st,
            &prompt,
            0,
            &mut sampler,
            max_tokens,
            |t| markers.is_stop(t),
            |t| {
                n_out += 1;
                bytes.extend_from_slice(&tok.decode(&[t]));
                let valid = match std::str::from_utf8(&bytes) {
                    Ok(s) => s.len(),
                    Err(e) => e.valid_up_to(),
                };
                if valid > 0 && send_err.is_none() {
                    let text = String::from_utf8_lossy(&bytes[..valid]).into_owned();
                    bytes.drain(..valid);
                    let chunk = serde_json::json!({
                        "id": id, "object": "chat.completion.chunk", "model": model_name,
                        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}],
                    });
                    if write!(stream, "data: {chunk}\n\n").and_then(|_| stream.flush()).is_err() {
                        send_err = Some(());
                    }
                }
            },
        )?;
        let fin = serde_json::json!({
            "id": id, "object": "chat.completion.chunk", "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        });
        let _ = write!(stream, "data: {fin}\n\ndata: [DONE]\n\n");
        let _ = stream.flush();
        eprintln!("pulsar-serve: {id}: {} prompt + {n_out} completion tokens (streamed)", prompt.len());
    } else {
        let mut out: Vec<u8> = Vec::new();
        let mut n_out = 0usize;
        engine::generate(
            model,
            st,
            &prompt,
            0,
            &mut sampler,
            max_tokens,
            |t| markers.is_stop(t),
            |t| {
                n_out += 1;
                out.extend_from_slice(&tok.decode(&[t]));
            },
        )?;
        let json = serde_json::json!({
            "id": id, "object": "chat.completion", "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": String::from_utf8_lossy(&out)},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt.len(),
                "completion_tokens": n_out,
                "total_tokens": prompt.len() + n_out,
            },
        });
        eprintln!("pulsar-serve: {id}: {} prompt + {n_out} completion tokens", prompt.len());
        respond_json(stream, 200, &json)?;
    }
    Ok(())
}
