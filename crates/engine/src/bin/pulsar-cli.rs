//! pulsar-cli: Hy3 generation and diagnostics.
//!
//!   pulsar-cli -m model.gguf -p "text" -n 32 [--ctx 2048] [--no-bos]
//!   pulsar-cli -m model.gguf --chat [--system "..."] [--temp 0.9]
//!   pulsar-cli -m model.gguf --tokens 120000,16883,11 -n 32
//!
//! -p tokenizes raw text (BOS prepended unless --no-bos); --tokens feeds
//! exact ids, which is how A/B runs align with ds4 --dump-tokens output.
//! --chat is an interactive multi-turn loop with the KV cache retained
//! across turns; sampling defaults come from the gguf's
//! general.sampling.* metadata unless --temp/--top-p are given.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("pulsar-cli requires Linux + CUDA");
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
fn main() {
    if let Err(e) = run() {
        eprintln!("pulsar-cli: {e}");
        std::process::exit(1);
    }
}

/// Flush the longest valid UTF-8 prefix of `buf` to stdout, keeping any
/// incomplete trailing multi-byte sequence for the next token.
#[cfg(target_os = "linux")]
fn print_utf8_prefix(buf: &mut Vec<u8>) {
    use std::io::Write;
    let valid_len = match std::str::from_utf8(buf) {
        Ok(_) => buf.len(),
        Err(e) => e.valid_up_to(),
    };
    if valid_len > 0 {
        let out = std::io::stdout();
        let mut lock = out.lock();
        lock.write_all(&buf[..valid_len]).ok();
        lock.flush().ok();
        buf.drain(..valid_len);
    }
}

#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn run_chat(
    model: &engine::Model,
    tok: &tokenizer::Tokenizer,
    ctx: u32,
    system: Option<String>,
    temp: Option<f32>,
    top_p: Option<f32>,
    min_p: f32,
    seed: u64,
    max_tokens: usize,
) -> engine::Result {
    use std::io::BufRead;

    let markers = tokenizer::ChatMarkers::resolve(tok)?;
    // sampling defaults from the gguf's own metadata (Hy3 ships 0.9/1.0)
    let meta_f = |k: &str, d: f32| {
        model.gguf.metadata.get(k).and_then(gguf::Value::as_f32).unwrap_or(d)
    };
    let temp = temp.unwrap_or_else(|| meta_f("general.sampling.temp", 0.9));
    let top_p = top_p.unwrap_or_else(|| meta_f("general.sampling.top_p", 1.0));
    let mut sampler = engine::Sampler::new(temp, top_p, min_p, seed);

    let mut st = engine::State::new(model, ctx)?;
    let max_tokens = if max_tokens <= 16 { 1024 } else { max_tokens };
    eprintln!(
        "pulsar chat: temp {temp} top-p {top_p} seed {seed}; ctx {ctx}; empty line or Ctrl-D exits"
    );

    let stdin = std::io::stdin();
    let mut pos = 0u32;
    let mut first = true;
    loop {
        eprint!("\n> ");
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            break;
        }

        let mut ids = Vec::new();
        if first {
            ids.extend(markers.bos);
            if let Some(sys) = &system {
                ids.extend(markers.render_system(tok, sys));
            }
            first = false;
        }
        ids.extend(markers.render_user_turn(tok, line));
        if std::env::var_os("PULSAR_DEBUG_IDS").is_some() {
            eprintln!("pulsar chat: turn ids {ids:?}");
        }

        if pos + ids.len() as u32 + 2 >= ctx {
            eprintln!("pulsar chat: context full ({pos}/{ctx}), restart to continue");
            break;
        }

        let mut bytes = Vec::new();
        pos = engine::generate(
            model,
            &mut st,
            &ids,
            pos,
            &mut sampler,
            max_tokens,
            |id| {
                let stop = markers.is_stop(id);
                if stop && std::env::var_os("PULSAR_DEBUG_IDS").is_some() {
                    eprintln!("pulsar chat: stop token {id} (eos {}, eot {:?})", markers.eos, markers.eot);
                }
                stop
            },
            |id| {
                if std::env::var_os("PULSAR_DEBUG_IDS").is_some() {
                    eprint!("[{id}]");
                }
                bytes.extend_from_slice(&tok.decode(&[id]));
                print_utf8_prefix(&mut bytes);
            },
        )?;
        println!();
    }
    st.save_warm(model)?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn run() -> engine::Result {
    let mut model_path = None;
    let mut prompt = None;
    let mut tokens_arg = None;
    let mut n_predict = 16usize;
    let mut ctx = 2048u32;
    let mut bos: Option<bool> = None; // None = model default (add_bos KV)
    let mut dump_logits = None;
    let mut teacher_force = false;
    let mut decode_consistency = None;
    let mut chat = false;
    let mut system = None;
    let mut temp = None;
    let mut top_p = None;
    let mut min_p = 0.0f32;
    let mut seed = 42u64;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut need = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match a.as_str() {
            "-m" => model_path = Some(need("-m")?),
            "-p" => prompt = Some(need("-p")?),
            // long prompts exceed the OS single-arg limit (~128KB on Linux)
            "-f" | "--prompt-file" => {
                let path = need("--prompt-file")?;
                prompt = Some(std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?);
            }
            "--tokens" => tokens_arg = Some(need("--tokens")?),
            "-n" => n_predict = need("-n")?.parse()?,
            "--ctx" => ctx = need("--ctx")?.parse()?,
            "--no-bos" => bos = Some(false),
            "--bos" => bos = Some(true),
            "--dump-logits" => dump_logits = Some(need("--dump-logits")?),
            "--teacher-force" => teacher_force = true,
            "--decode-consistency" => decode_consistency = Some(need("--decode-consistency")?.parse::<usize>()?),
            "--chat" => chat = true,
            "--system" => system = Some(need("--system")?),
            "--temp" => temp = Some(need("--temp")?.parse::<f32>()?),
            "--top-p" => top_p = Some(need("--top-p")?.parse::<f32>()?),
            "--min-p" => min_p = need("--min-p")?.parse::<f32>()?,
            "--seed" => seed = need("--seed")?.parse::<u64>()?,
            other => return Err(format!("unknown arg {other}").into()),
        }
    }
    let model_path = model_path.ok_or("missing -m MODEL.gguf")?;

    eprintln!("pulsar: loading {model_path}");
    let t0 = std::time::Instant::now();
    let model = engine::Model::load(std::path::Path::new(&model_path))?;
    let tok = {
        let (_, g) = engine::parse_header(std::path::Path::new(&model_path))?;
        tokenizer::Tokenizer::from_gguf(&g)?
    };
    eprintln!(
        "pulsar: loaded in {:.1}s ({} layers, {} experts x top-{})",
        t0.elapsed().as_secs_f32(),
        model.shape.n_exec_layer,
        model.shape.n_expert,
        model.shape.n_expert_used
    );

    if chat {
        return run_chat(&model, &tok, ctx, system, temp, top_p, min_p, seed, n_predict);
    }

    let prompt_ids: Vec<u32> = match (tokens_arg, prompt) {
        (Some(t), _) => t.split(',').map(|s| s.trim().parse()).collect::<std::result::Result<_, _>>()?,
        (None, Some(p)) => {
            let mut ids = Vec::new();
            if bos.unwrap_or(tok.add_bos) {
                ids.push(tok.bos_id.ok_or("model has no BOS id")?);
            }
            ids.extend(tok.encode(&p));
            ids
        }
        (None, None) => return Err("need -p TEXT or --tokens IDS".into()),
    };
    eprintln!("pulsar: prompt ids {prompt_ids:?}");

    // Long prompts want one big prefill chunk (each chunk costs a full
    // expert-corpus pass) and can trade VRAM pool for it - the pool barely
    // hits during prefill anyway. Explicit env vars win.
    if prompt_ids.len() > 384 {
        if std::env::var_os("PULSAR_BATCH").is_none() {
            std::env::set_var("PULSAR_BATCH", prompt_ids.len().min(768).to_string());
        }
        if std::env::var_os("PULSAR_DEV_CACHE_GB").is_none() {
            std::env::set_var("PULSAR_DEV_CACHE_GB", "2");
        }
    }

    let mut st = engine::State::new(&model, ctx)?;

    if teacher_force {
        // Per-position top-5 (id, logit) along the given token sequence,
        // one JSON line per position, for cross-engine agreement checks.
        for (i, &id) in prompt_ids.iter().enumerate() {
            let l = model.forward_token(&mut st, id, i as u32, true)?.unwrap();
            let mut top: Vec<u32> = (0..l.len() as u32).collect();
            top.sort_by(|&a, &b| l[b as usize].total_cmp(&l[a as usize]));
            let entries: Vec<String> = top[..5]
                .iter()
                .map(|&t| format!("[{},{}]", t, l[t as usize]))
                .collect();
            println!("{{\"pos\":{},\"after\":{},\"top\":[{}]}}", i, id, entries.join(","));
        }
        return Ok(());
    }

    if let Some(nsteps) = decode_consistency {
        // Greedy-decode nsteps tokens through the incremental (n_tok=1)
        // path, then fresh-prefill the identical sequence batched and
        // compare the logits at the same position. Divergence here is the
        // reduction-order drift between the batch and decode matmul
        // kernels - the ds4 --decode-consistency analogue.
        let mut logits = None;
        let mut pos0 = 0u32;
        for chunk in prompt_ids.chunks(st.max_batch() as usize) {
            logits = model.forward_batch(&mut st, chunk, pos0, true)?;
            pos0 += chunk.len() as u32;
        }
        let mut seq = prompt_ids.clone();
        for _ in 0..nsteps.saturating_sub(1) {
            let next = engine::argmax(logits.as_ref().ok_or("no logits")?);
            seq.push(next);
            logits = model.forward_batch(&mut st, &[next], seq.len() as u32 - 1, true)?;
        }
        let decode_logits = logits.ok_or("no logits")?;
        let decode_argmax = engine::argmax(&decode_logits);

        drop(st); // free VRAM before the fresh state
        let mut st2 = engine::State::new(&model, ctx)?;
        let mut fresh = None;
        let mut pos0 = 0u32;
        for chunk in seq.chunks(st2.max_batch() as usize) {
            fresh = model.forward_batch(&mut st2, chunk, pos0, true)?;
            pos0 += chunk.len() as u32;
        }
        let fresh_logits = fresh.ok_or("no logits")?;
        let fresh_argmax = engine::argmax(&fresh_logits);

        let mut maxd = 0f32;
        let mut sum = 0f64;
        for (a, b) in decode_logits.iter().zip(&fresh_logits) {
            let d = (a - b).abs();
            maxd = maxd.max(d);
            sum += d as f64;
        }
        let gap = {
            let mut top = f32::NEG_INFINITY;
            let mut second = f32::NEG_INFINITY;
            for &v in &decode_logits {
                if v > top {
                    second = top;
                    top = v;
                } else if v > second {
                    second = v;
                }
            }
            top - second
        };
        println!(
            "decode-consistency after {} steps ({} total tokens):\n  max |dlogit| {maxd:.4}, mean {:.5}\n  argmax decode={decode_argmax} fresh-prefill={fresh_argmax} ({}), decode top1-top2 gap {gap:.4}",
            nsteps,
            seq.len(),
            sum / decode_logits.len() as f64,
            if decode_argmax == fresh_argmax { "MATCH" } else { "FLIP" },
        );
        return Ok(());
    }

    // DFlash speculative decode (qwen35moe + a matched block-diffusion
    // draft gguf): PULSAR_DFLASH=/path/to/draft.gguf, greedy one-shot
    if let (Ok(draft_path), None) = (std::env::var("PULSAR_DFLASH"), dump_logits.as_ref()) {
        let mut draft = engine::DraftModel::load(std::path::Path::new(&draft_path))?;
        eprintln!("pulsar: dflash draft loaded ({draft_path})");
        let mut generated: Vec<u32> = Vec::new();
        let mut t_first: Option<std::time::Instant> = None;
        let out = std::io::stdout();
        engine::generate_dflash(
            &model,
            &mut draft,
            &mut st,
            &prompt_ids,
            0,
            n_predict,
            |t| tok.is_eog(t),
            |t| {
                t_first.get_or_insert_with(std::time::Instant::now);
                generated.push(t);
                use std::io::Write;
                let mut o = out.lock();
                o.write_all(&tok.decode(&[t])).ok();
                o.flush().ok();
            },
        )?;
        println!();
        st.save_warm(&model)?;
        let dt = t_first.map(|t| t.elapsed().as_secs_f32()).unwrap_or(0.0);
        eprintln!(
            "pulsar: {} tokens in {:.2}s ({:.2} tok/s), dflash {}/{} drafts accepted ({:.0}%)\npulsar: ids {generated:?}",
            generated.len(),
            dt,
            generated.len() as f32 / dt.max(1e-6),
            st.mtp_accepted,
            st.mtp_drafted,
            100.0 * st.mtp_accepted as f64 / st.mtp_drafted.max(1) as f64
        );
        return Ok(());
    }

    // MTP speculative decode routes through engine::generate (the spec
    // loop lives there); greedy-only, so the one-shot default applies
    if (std::env::var("PULSAR_MTP").ok().as_deref() == Some("1")
        || std::env::var("PULSAR_NGRAM").is_ok())
        && dump_logits.is_none() {
        let mut generated: Vec<u32> = Vec::new();
        let mut t_first: Option<std::time::Instant> = None;
        let mut sampler = engine::Sampler::new(0.0, 1.0, 0.0, 1);
        let out = std::io::stdout();
        engine::generate(
            &model,
            &mut st,
            &prompt_ids,
            0,
            &mut sampler,
            n_predict,
            |t| tok.is_eog(t),
            |t| {
                t_first.get_or_insert_with(std::time::Instant::now);
                generated.push(t);
                use std::io::Write;
                let mut o = out.lock();
                o.write_all(&tok.decode(&[t])).ok();
                o.flush().ok();
            },
        )?;
        println!();
        if std::env::var_os("PULSAR_PROFILE").is_some() {
            eprintln!("pulsar: profile: {}", st.prof.report());
        }
        st.save_warm(&model)?;
        let dt = t_first.map(|t| t.elapsed().as_secs_f32()).unwrap_or(0.0);
        eprintln!(
            "pulsar: {} tokens in {:.2}s ({:.2} tok/s), mtp {}/{} drafts accepted ({:.0}%)\npulsar: ids {generated:?}",
            generated.len(),
            dt,
            generated.len() as f32 / dt.max(1e-6),
            st.mtp_accepted,
            st.mtp_drafted,
            100.0 * st.mtp_accepted as f64 / st.mtp_drafted.max(1) as f64
        );
        return Ok(());
    }

    let t1 = std::time::Instant::now();
    let mut logits = None;
    let mut pos0 = 0u32;
    for chunk in prompt_ids.chunks(st.max_batch() as usize) {
        let last = pos0 as usize + chunk.len() == prompt_ids.len();
        logits = model.forward_batch(&mut st, chunk, pos0, last)?;
        pos0 += chunk.len() as u32;
    }
    eprintln!(
        "pulsar: prefill {} tokens in {:.2}s",
        prompt_ids.len(),
        t1.elapsed().as_secs_f32()
    );

    if let Some(path) = dump_logits {
        let l = logits.as_ref().ok_or("no logits")?;
        let mut s = String::with_capacity(l.len() * 12);
        s.push('[');
        for (i, v) in l.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!("{v}"));
        }
        s.push(']');
        std::fs::write(&path, s)?;
        eprintln!("pulsar: wrote {} logits to {path}", l.len());
        return Ok(());
    }

    let mut pos = prompt_ids.len() as u32;
    let mut generated = Vec::new();
    let t2 = std::time::Instant::now();
    for _ in 0..n_predict {
        let l = logits.as_ref().ok_or("no logits")?;
        let next = engine::argmax(l);
        if tok.is_eog(next) {
            break;
        }
        generated.push(next);
        print!("{}", String::from_utf8_lossy(&tok.decode(&[next])));
        use std::io::Write;
        std::io::stdout().flush().ok();
        if pos >= ctx {
            break;
        }
        logits = model.forward_token(&mut st, next, pos, true)?;
        pos += 1;
    }
    println!();
    if std::env::var_os("PULSAR_PROFILE").is_some() {
        eprintln!("pulsar: profile: {}", st.prof.report());
    }
    st.save_warm(&model)?;
    let dt = t2.elapsed().as_secs_f32();
    let tier_note = {
        let hits: u64 = st.tiers.iter().map(|t| t.hits).sum();
        let mut s = if hits > 0 {
            format!(", tier {hits} resident slots")
        } else {
            String::new()
        };
        if st.cpu_hits > 0 {
            s += &format!(", cpu lane {} experts", st.cpu_hits);
        }
        if st.mtp_drafted > 0 {
            s += &format!(
                ", mtp {}/{} drafts accepted ({:.0}%)",
                st.mtp_accepted,
                st.mtp_drafted,
                100.0 * st.mtp_accepted as f64 / st.mtp_drafted as f64
            );
        }
        s
    };
    eprintln!(
        "pulsar: {} tokens in {:.2}s ({:.2} tok/s), vram cache {:.0}% hits, host cache {:.0}% of remainder{tier_note}\npulsar: ids {generated:?}",
        generated.len(),
        dt,
        generated.len() as f32 / dt.max(1e-6),
        100.0 * st.dev_cache.hits as f64 / (st.dev_cache.hits + st.dev_cache.misses).max(1) as f64,
        100.0 * st.store.hits as f64 / (st.store.hits + st.store.misses).max(1) as f64
    );
    Ok(())
}
