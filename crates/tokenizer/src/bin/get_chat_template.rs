//! Fetch a HuggingFace / llama.cpp chat template (Rust port of
//! llama.cpp's `scripts/get_chat_template.py`, extended for GGUF quants).
//!
//! Usage:
//!   get-chat-template MODEL_ID [VARIANT]
//!   get-chat-template path/to/model.gguf
//!   get-chat-template MODEL_ID --save out.jinja
//!   get-chat-template MODEL_ID --offline
//!
//! Examples:
//!   get-chat-template microsoft/Phi-3.5-mini-instruct
//!   get-chat-template CohereForAI/c4ai-command-r-plus tool_use
//!   get-chat-template ./Qwen2.5-7B-Instruct-Q4_K_M.gguf

fn main() {
    if let Err(e) = run() {
        eprintln!("get-chat-template: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "Usage: get-chat-template <MODEL_ID|MODEL.gguf> [VARIANT] [--save PATH] [--offline] [--meta]\n\
             \n\
             Resolves a Jinja chat template from, in order:\n\
               1. GGUF embedded tokenizer.chat_template (when given a .gguf)\n\
               2. Local cache ($PULSAR_TEMPLATE_CACHE)\n\
               3. HuggingFace tokenizer_config.json (base model for quants)\n\
               4. llama.cpp models/templates catalog on GitHub\n\
             \n\
             Gated HF models: set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN).\n\
             --meta   print source / model-id on stderr only; template on stdout\n\
             --offline  never hit the network (embedded + cache only)"
        );
        std::process::exit(if args.is_empty() { 1 } else { 0 });
    }

    let mut save: Option<String> = None;
    let mut offline = false;
    let mut meta = false;
    let mut positional = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--save" => {
                i += 1;
                save = Some(
                    args.get(i)
                        .cloned()
                        .ok_or("--save needs a path")?,
                );
            }
            "--offline" => offline = true,
            "--meta" => meta = true,
            other if other.starts_with('-') => {
                return Err(format!("unknown flag {other}").into());
            }
            other => positional.push(other.to_string()),
        }
        i += 1;
    }
    // Also allow env-style offline.
    if offline {
        std::env::set_var("PULSAR_OFFLINE", "1");
    }

    let spec = positional
        .first()
        .ok_or("missing MODEL_ID or MODEL.gguf")?
        .as_str();
    let variant = positional.get(1).map(|s| s.as_str());

    let mut opts = tokenizer::ChatTemplateOptions::default();
    opts.offline = offline;

    let resolved = tokenizer::get_chat_template_with_options(spec, variant, &opts)?;

    if meta {
        eprintln!("source:   {}", resolved.source);
        if let Some(id) = &resolved.model_id {
            eprintln!("model_id: {id}");
        }
        if let Some(v) = &resolved.variant {
            eprintln!("variant:  {v}");
        }
        eprintln!("bytes:    {}", resolved.template.len());
    }

    if let Some(path) = save {
        std::fs::write(&path, &resolved.template)?;
        eprintln!("wrote {} ({} bytes) from {}", path, resolved.template.len(), resolved.source);
    } else {
        print!("{}", resolved.template);
        if !resolved.template.ends_with('\n') {
            println!();
        }
    }
    Ok(())
}
