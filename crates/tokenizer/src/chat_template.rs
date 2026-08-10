//! Automatic chat-template discovery and application.
//!
//! Resolution order (first hit wins):
//! 1. Embedded `tokenizer.chat_template` in GGUF metadata
//! 2. Local disk cache (`$PULSAR_TEMPLATE_CACHE` or platform cache dir)
//! 3. HuggingFace `tokenizer_config.json` for the model (and, for quant
//!    GGUFs, the base model / org+basename candidates)
//! 4. llama.cpp's curated catalog at
//!    `https://github.com/ggml-org/llama.cpp/tree/master/models/templates`
//!
//! Based on llama.cpp's `scripts/get_chat_template.py`, extended so a
//! quantized GGUF without a template can still recover one from the base
//! model identity.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use gguf::Gguf;
use serde_json::Value as Json;

/// Where a resolved Jinja template came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatTemplateSource {
    /// `tokenizer.chat_template` inside the GGUF header.
    GgufEmbedded,
    /// Previously downloaded file under the template cache.
    Cache(PathBuf),
    /// HuggingFace `tokenizer_config.json` for this repo id.
    HuggingFace(String),
    /// Raw `.jinja` from the llama.cpp models/templates catalog.
    LlamaCppCatalog(String),
    /// Explicit path supplied by the caller.
    File(PathBuf),
}

impl std::fmt::Display for ChatTemplateSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GgufEmbedded => write!(f, "gguf:tokenizer.chat_template"),
            Self::Cache(p) => write!(f, "cache:{}", p.display()),
            Self::HuggingFace(id) => write!(f, "huggingface:{id}"),
            Self::LlamaCppCatalog(name) => write!(f, "llama.cpp/templates/{name}"),
            Self::File(p) => write!(f, "file:{}", p.display()),
        }
    }
}

/// A resolved Jinja chat template ready to apply or cache.
#[derive(Debug, Clone)]
pub struct ResolvedChatTemplate {
    pub template: String,
    pub source: ChatTemplateSource,
    /// HuggingFace-style `org/name` when known.
    pub model_id: Option<String>,
    /// Variant picked when the config shipped multiple templates.
    pub variant: Option<String>,
}

#[derive(Debug)]
pub enum ChatTemplateError {
    NotFound(String),
    Network(String),
    Parse(String),
    Io(String),
    Apply(String),
    InvalidModelId(String),
}

impl std::fmt::Display for ChatTemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(s) => write!(f, "chat template not found: {s}"),
            Self::Network(s) => write!(f, "chat template network error: {s}"),
            Self::Parse(s) => write!(f, "chat template parse error: {s}"),
            Self::Io(s) => write!(f, "chat template io error: {s}"),
            Self::Apply(s) => write!(f, "chat template apply error: {s}"),
            Self::InvalidModelId(s) => write!(f, "invalid model id: {s}"),
        }
    }
}

impl std::error::Error for ChatTemplateError {}

impl From<std::io::Error> for ChatTemplateError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Fetch the Jinja chat template for a HuggingFace model id, matching
/// llama.cpp's `get_chat_template(model_id, variant)`.
pub fn get_chat_template(
    model_id: &str,
    variant: Option<&str>,
) -> Result<ResolvedChatTemplate, ChatTemplateError> {
    get_chat_template_with_options(model_id, variant, &ChatTemplateOptions::default())
}

/// Options that control network lookup, cache, and quant base-model walks.
#[derive(Debug, Clone)]
pub struct ChatTemplateOptions {
    /// Also try the llama.cpp models/templates catalog.
    pub use_llama_cpp_catalog: bool,
    /// Read/write downloaded templates under the cache dir.
    pub use_cache: bool,
    /// Override cache root (else `$PULSAR_TEMPLATE_CACHE` / platform cache).
    pub cache_dir: Option<PathBuf>,
    /// Skip network (embedded + cache only).
    pub offline: bool,
    /// HuggingFace access token (`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` if None).
    pub hf_token: Option<String>,
    /// HTTP timeout.
    pub timeout: Duration,
}

impl Default for ChatTemplateOptions {
    fn default() -> Self {
        Self {
            use_llama_cpp_catalog: true,
            use_cache: true,
            cache_dir: None,
            offline: std::env::var_os("PULSAR_OFFLINE").is_some(),
            hf_token: None,
            timeout: Duration::from_secs(30),
        }
    }
}

/// Resolve a chat template from a HF id, free-form model name, or `.gguf` path.
pub fn get_chat_template_with_options(
    spec: &str,
    variant: Option<&str>,
    opts: &ChatTemplateOptions,
) -> Result<ResolvedChatTemplate, ChatTemplateError> {
    let path = Path::new(spec);
    if path.extension().and_then(|e| e.to_str()) == Some("gguf") && path.exists() {
        return resolve_from_gguf_path(path, variant, opts);
    }
    if path.is_file()
        && path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("jinja") || e.eq_ignore_ascii_case("txt"))
    {
        let template = fs::read_to_string(path)?;
        return Ok(ResolvedChatTemplate {
            template,
            source: ChatTemplateSource::File(path.to_path_buf()),
            model_id: None,
            variant: variant.map(str::to_owned),
        });
    }
    resolve_for_model_id(spec, variant, opts)
}

/// Resolve using GGUF metadata already loaded in-process (no re-parse).
pub fn get_chat_template_from_gguf(
    g: &Gguf,
    gguf_path: Option<&Path>,
    variant: Option<&str>,
    opts: &ChatTemplateOptions,
) -> Result<ResolvedChatTemplate, ChatTemplateError> {
    // 1. Embedded template in the GGUF itself.
    if let Some(t) = g.metadata.get("tokenizer.chat_template").and_then(|v| v.as_str()) {
        if !t.trim().is_empty() {
            return Ok(ResolvedChatTemplate {
                template: t.to_owned(),
                source: ChatTemplateSource::GgufEmbedded,
                model_id: primary_model_id(g, gguf_path),
                variant: variant.map(str::to_owned),
            });
        }
    }

    let candidates = model_id_candidates(g, gguf_path);
    if candidates.is_empty() {
        return Err(ChatTemplateError::NotFound(
            "no model identity in GGUF metadata and no usable filename".into(),
        ));
    }

    let mut errors = Vec::new();
    for id in &candidates {
        match resolve_for_model_id(id, variant, opts) {
            Ok(mut r) => {
                r.model_id = Some(id.clone());
                return Ok(r);
            }
            Err(e) => errors.push(format!("{id}: {e}")),
        }
    }
    Err(ChatTemplateError::NotFound(format!(
        "tried {}; {}",
        candidates.join(", "),
        errors.join("; ")
    )))
}

// ---------------------------------------------------------------------------
// Model identity / quant stripping
// ---------------------------------------------------------------------------

/// Strip common GGUF quant / precision / packaging suffixes so a filename
/// like `Qwen2.5-7B-Instruct-Q4_K_M` collapses toward the base HF name.
pub fn strip_quant_suffix(name: &str) -> String {
    let mut s = name.trim().to_string();
    // Drop extension if present.
    if let Some(stem) = Path::new(&s).file_stem().and_then(|x| x.to_str()) {
        if s.ends_with(".gguf") || s.ends_with(".GGUF") {
            s = stem.to_string();
        }
    }
    // Split GGUF shard suffixes: -00001-of-00015
    if let Some(i) = s.find("-0000") {
        if s[i..].contains("-of-") {
            s = s[..i].to_string();
        }
    }

    // Repeatedly peel trailing quant / dtype / packaging tokens.
    const SUFFIXES: &[&str] = &[
        // UD / imatrix recipes
        "UD-IQ1_S",
        "UD-IQ1_M",
        "UD-IQ2_XXS",
        "UD-IQ2_XS",
        "UD-IQ2_S",
        "UD-IQ2_M",
        "UD-IQ3_XXS",
        "UD-IQ3_XS",
        "UD-IQ3_S",
        "UD-IQ3_M",
        "UD-Q2_K_XL",
        "UD-Q3_K_XL",
        "UD-Q4_K_XL",
        "UD-Q5_K_XL",
        "UD-Q6_K_XL",
        "UD-Q8_K_XL",
        // standard K-quants + IQ
        "IQ1_S",
        "IQ1_M",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S",
        "IQ2_M",
        "IQ3_XXS",
        "IQ3_XS",
        "IQ3_S",
        "IQ3_M",
        "IQ4_XS",
        "IQ4_NL",
        "Q2_K_S",
        "Q2_K_M",
        "Q2_K_XL",
        "Q2_K",
        "Q3_K_S",
        "Q3_K_M",
        "Q3_K_L",
        "Q3_K_XL",
        "Q3_K",
        "Q4_K_S",
        "Q4_K_M",
        "Q4_K_XL",
        "Q4_K",
        "Q5_K_S",
        "Q5_K_M",
        "Q5_K_XL",
        "Q5_K",
        "Q6_K_XL",
        "Q6_K",
        "Q8_K_XL",
        "Q8_0",
        "Q8_1",
        "Q5_0",
        "Q5_1",
        "Q4_0",
        "Q4_1",
        "Q3_0",
        "TQ1_0",
        "TQ2_0",
        "MXFP4",
        "F32",
        "F16",
        "BF16",
        "FP16",
        "FP32",
        "GGUF",
        "gguf",
        "imatrix",
    ];

    loop {
        let before = s.clone();
        // Normalize separators at the end: -Q4_K_M / _Q4_K_M / .Q4_K_M
        for suf in SUFFIXES {
            for sep in ['-', '_', '.'] {
                let needle = format!("{sep}{suf}");
                if s.ends_with(&needle) {
                    s.truncate(s.len() - needle.len());
                }
                // case-insensitive peel for mixed-case names
                if s.len() >= needle.len() {
                    let tail = &s[s.len() - needle.len()..];
                    if tail.eq_ignore_ascii_case(&needle) {
                        s.truncate(s.len() - needle.len());
                    }
                }
            }
        }
        // Trailing "-v1" style quant tags sometimes appear as "Q4KM"
        s = s.trim_end_matches(['-', '_', '.']).to_string();
        if s == before {
            break;
        }
    }
    s
}

/// Build ordered HF model-id candidates from GGUF metadata + optional path.
pub fn model_id_candidates(g: &Gguf, gguf_path: Option<&Path>) -> Vec<String> {
    let mut out = Vec::new();
    let mut push = |s: String| {
        let t = s.trim().to_string();
        if t.is_empty() {
            return;
        }
        if !out.iter().any(|x| x == &t) {
            out.push(t);
        }
    };

    // Explicit base-model entries (quantized models often point here).
    if let Some(n) = meta_u64(g, "general.base_model.count") {
        for i in 0..n.min(16) {
            let name = meta_str(g, &format!("general.base_model.{i}.name"));
            let org = meta_str(g, &format!("general.base_model.{i}.organization"));
            let repo = meta_str(g, &format!("general.base_model.{i}.repo_url"));
            if let Some(r) = repo.as_deref().and_then(hf_id_from_url) {
                push(r);
            }
            match (org.as_deref(), name.as_deref()) {
                (Some(o), Some(n)) if !n.contains('/') => push(format!("{o}/{n}")),
                (_, Some(n)) if n.contains('/') => push(n.to_string()),
                (_, Some(n)) => push(n.to_string()),
                _ => {}
            }
        }
    }
    // Singular keys some converters write.
    for key in [
        "general.base_model.name",
        "general.base_model",
        "general.url",
        "general.source.url",
        "general.base_model.repo_url",
    ] {
        if let Some(v) = meta_str(g, key) {
            if let Some(id) = hf_id_from_url(&v) {
                push(id);
            } else if looks_like_hf_id(&v) {
                push(v);
            }
        }
    }

    let org = meta_str(g, "general.organization");
    let basename = meta_str(g, "general.basename");
    let finetune = meta_str(g, "general.finetune");
    let name = meta_str(g, "general.name");

    if let (Some(o), Some(b)) = (org.as_deref(), basename.as_deref()) {
        let mut id = format!("{o}/{b}");
        if let Some(f) = finetune.as_deref() {
            if !f.is_empty() && !b.ends_with(f) {
                id = format!("{o}/{b}-{f}");
                push(id.clone());
            }
        }
        push(format!("{o}/{b}"));
        // Also without org slash form when name already qualified.
        let _ = id;
    }

    if let Some(n) = name {
        if looks_like_hf_id(&n) {
            push(n.clone());
        }
        // quant-stripped general.name
        let stripped = strip_quant_suffix(&n);
        if looks_like_hf_id(&stripped) {
            push(stripped);
        } else if let Some(o) = org.as_deref() {
            if !stripped.is_empty() {
                push(format!("{o}/{stripped}"));
            }
        } else {
            push(stripped);
        }
    }

    if let Some(path) = gguf_path {
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            let stripped = strip_quant_suffix(stem);
            if looks_like_hf_id(&stripped) {
                push(stripped.clone());
            }
            // Unsloth / bartowski style: `ModelName-Q4_K_M` without org.
            if !stripped.is_empty() {
                push(stripped.clone());
            }
            // If org is known, prefer org/stripped.
            if let Some(o) = org.as_deref() {
                push(format!("{o}/{stripped}"));
            }
        }
    }

    // Architecture as a last-resort catalog key (qwen3moe, deepseek2, …).
    if let Some(arch) = g.architecture() {
        push(arch.to_string());
    }

    out
}

fn primary_model_id(g: &Gguf, gguf_path: Option<&Path>) -> Option<String> {
    model_id_candidates(g, gguf_path).into_iter().next()
}

fn meta_str(g: &Gguf, key: &str) -> Option<String> {
    g.metadata.get(key).and_then(|v| v.as_str()).map(str::to_owned)
}

fn meta_u64(g: &Gguf, key: &str) -> Option<u64> {
    g.metadata.get(key).and_then(|v| v.as_u64())
}

fn looks_like_hf_id(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() || s.contains(' ') {
        return false;
    }
    let mut parts = s.split('/');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(a), Some(b), None) => {
            !a.is_empty()
                && !b.is_empty()
                && a.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
                && b.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
        }
        _ => false,
    }
}

fn hf_id_from_url(url: &str) -> Option<String> {
    // https://huggingface.co/org/model or huggingface.co/org/model/tree/main
    let url = url.trim().trim_end_matches('/');
    let markers = ["huggingface.co/", "hf.co/"];
    for m in markers {
        if let Some(pos) = url.find(m) {
            let rest = &url[pos + m.len()..];
            let mut parts = rest.split('/').filter(|p| !p.is_empty());
            let org = parts.next()?;
            let name = parts.next()?;
            // skip non-model paths
            if matches!(org, "datasets" | "spaces" | "docs" | "api") {
                return None;
            }
            let id = format!("{org}/{name}");
            if looks_like_hf_id(&id) {
                return Some(id);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Resolution internals
// ---------------------------------------------------------------------------

fn resolve_from_gguf_path(
    path: &Path,
    variant: Option<&str>,
    opts: &ChatTemplateOptions,
) -> Result<ResolvedChatTemplate, ChatTemplateError> {
    let map = map_gguf_header(path)?;
    let g = Gguf::parse(&map).map_err(|e| ChatTemplateError::Parse(e.to_string()))?;
    get_chat_template_from_gguf(&g, Some(path), variant, opts)
}

/// Memory-map (or read) enough of the GGUF to parse the header + metadata.
fn map_gguf_header(path: &Path) -> Result<Vec<u8>, ChatTemplateError> {
    // Metadata for multi-hundred-GB models still fits in a few MB of header.
    // Read up to 64 MiB; if the header is larger, expand once.
    const FIRST: usize = 64 * 1024 * 1024;
    let mut f = fs::File::open(path)?;
    let mut buf = vec![0u8; FIRST];
    let n = f.read(&mut buf)?;
    buf.truncate(n);
    // If parse fails with truncated, caller surfaces the error; good enough
    // for tokenizer.chat_template + general.* which sit near the front.
    Ok(buf)
}

fn resolve_for_model_id(
    model_id: &str,
    variant: Option<&str>,
    opts: &ChatTemplateOptions,
) -> Result<ResolvedChatTemplate, ChatTemplateError> {
    let model_id = model_id.trim();
    if model_id.is_empty() {
        return Err(ChatTemplateError::InvalidModelId("empty".into()));
    }

    // Cache hit.
    if opts.use_cache {
        if let Some(cached) = read_cache(model_id, variant, opts) {
            return Ok(cached);
        }
    }

    // Prefer a variant-specific file from the llama.cpp catalog when a
    // variant is requested: HF tokenizer_config often only ships the default
    // string template, while the catalog has `*-tool_use.jinja` etc.
    if opts.use_llama_cpp_catalog && !opts.offline && variant.is_some() {
        if let Ok(r) = fetch_llama_cpp_catalog_template(model_id, variant, opts) {
            if opts.use_cache {
                let _ = write_cache(model_id, variant, &r.template, opts);
            }
            return Ok(r);
        }
    }

    // HuggingFace tokenizer_config.json
    if !opts.offline {
        match fetch_hf_chat_template(model_id, variant, opts) {
            Ok(mut r) => {
                if opts.use_cache {
                    if let Err(e) = write_cache(model_id, variant, &r.template, opts) {
                        eprintln!("pulsar chat-template: cache write failed: {e}");
                    }
                }
                r.model_id = Some(model_id.to_owned());
                return Ok(r);
            }
            Err(e) => {
                if !opts.use_llama_cpp_catalog {
                    return Err(e);
                }
            }
        }
    }

    // llama.cpp curated templates catalog (default / no-variant path)
    if opts.use_llama_cpp_catalog && !opts.offline {
        if let Ok(r) = fetch_llama_cpp_catalog_template(model_id, variant, opts) {
            if opts.use_cache {
                let _ = write_cache(model_id, variant, &r.template, opts);
            }
            return Ok(r);
        }
    }

    // Offline cache already tried; nothing else.
    Err(ChatTemplateError::NotFound(format!(
        "no chat template for '{model_id}' (HF + llama.cpp catalog)"
    )))
}

/// Parse `chat_template` out of a tokenizer_config.json body.
pub fn chat_template_from_tokenizer_config(
    config_str: &str,
    variant: Option<&str>,
) -> Result<(String, Option<String>), ChatTemplateError> {
    let cleaned = fix_broken_tokenizer_config(config_str);
    let config: Json = serde_json::from_str(&cleaned)
        .map_err(|e| ChatTemplateError::Parse(format!("tokenizer_config.json: {e}")))?;

    let chat_template = config
        .get("chat_template")
        .ok_or_else(|| ChatTemplateError::NotFound("no chat_template field".into()))?;

    match chat_template {
        Json::String(s) => Ok((s.clone(), None)),
        Json::Array(arr) => {
            let mut variants: Vec<(String, String)> = Vec::new();
            for ct in arr {
                let name = ct
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let tmpl = ct
                    .get("template")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        ChatTemplateError::Parse("chat_template entry missing template".into())
                    })?
                    .to_string();
                variants.push((name, tmpl));
            }
            let format_names = || {
                variants
                    .iter()
                    .map(|(n, _)| format!("\"{n}\""))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            let pick = match variant {
                Some(v) => v.to_string(),
                None => {
                    if variants.iter().any(|(n, _)| n == "default") {
                        "default".to_string()
                    } else {
                        return Err(ChatTemplateError::NotFound(format!(
                            "specify a chat template variant (one of {})",
                            format_names()
                        )));
                    }
                }
            };
            let tmpl = variants
                .iter()
                .find(|(n, _)| n == &pick)
                .map(|(_, t)| t.clone())
                .ok_or_else(|| {
                    ChatTemplateError::NotFound(format!(
                        "variant \"{pick}\" not found (found {})",
                        format_names()
                    ))
                })?;
            Ok((tmpl, Some(pick)))
        }
        _ => Err(ChatTemplateError::Parse(
            "chat_template is neither string nor array".into(),
        )),
    }
}

/// Fix the well-known broken Llama-3 tokenizer_config.json with an extra `}`.
fn fix_broken_tokenizer_config(s: &str) -> String {
    // Mirrors: re.sub(r'\}([\n\s]*\}[\n\s]*\],[\n\s]*"clean_up_tokenization_spaces")', r'\1', s)
    let re = regex_lite_fix(s);
    re.unwrap_or_else(|| s.to_string())
}

fn regex_lite_fix(s: &str) -> Option<String> {
    // Avoid a regex crate dependency: only rewrite the known Llama-3 pattern.
    let marker = "\"clean_up_tokenization_spaces\"";
    let pos = s.find(marker)?;
    // Walk back over whitespace and `],` then an extra `}`.
    let before = &s[..pos];
    let extra = before.rfind("}")?;
    // Ensure between extra `}` and marker we only have `} ] ,` whitespace.
    let mid = before[extra + 1..].trim_start();
    if !mid.starts_with('}') {
        return None;
    }
    let after_second = mid[1..].trim_start();
    if !after_second.starts_with(']') {
        return None;
    }
    // Drop the first of the two closing braces.
    let mut out = String::with_capacity(s.len());
    out.push_str(&s[..extra]);
    out.push_str(&s[extra + 1..]);
    Some(out)
}

fn fetch_hf_chat_template(
    model_id: &str,
    variant: Option<&str>,
    opts: &ChatTemplateOptions,
) -> Result<ResolvedChatTemplate, ChatTemplateError> {
    if !looks_like_hf_id(model_id) {
        // Still try — some single-segment names resolve under common orgs later.
        // HF requires org/name; reject pure junk.
        if !model_id.contains('/') {
            return Err(ChatTemplateError::InvalidModelId(format!(
                "{model_id} (expected org/name)"
            )));
        }
    }

    let url = format!("https://huggingface.co/{model_id}/resolve/main/tokenizer_config.json");
    let body = http_get(&url, opts)?;
    let (template, picked) = chat_template_from_tokenizer_config(&body, variant)?;
    Ok(ResolvedChatTemplate {
        template,
        source: ChatTemplateSource::HuggingFace(model_id.to_owned()),
        model_id: Some(model_id.to_owned()),
        variant: picked.or_else(|| variant.map(str::to_owned)),
    })
}

fn fetch_llama_cpp_catalog_template(
    model_id: &str,
    variant: Option<&str>,
    opts: &ChatTemplateOptions,
) -> Result<ResolvedChatTemplate, ChatTemplateError> {
    let names = catalog_candidate_filenames(model_id, variant);
    let mut last_err = ChatTemplateError::NotFound("empty catalog candidates".into());
    for name in names {
        let url = format!(
            "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/models/templates/{name}"
        );
        match http_get(&url, opts) {
            Ok(body) if body.contains("{{") || body.contains("{%") || body.contains("{%-") => {
                return Ok(ResolvedChatTemplate {
                    template: body,
                    source: ChatTemplateSource::LlamaCppCatalog(name),
                    model_id: Some(model_id.to_owned()),
                    variant: variant.map(str::to_owned),
                });
            }
            Ok(_) => {
                last_err = ChatTemplateError::Parse(format!("{name}: not a jinja template"));
            }
            Err(e) => last_err = e,
        }
    }
    Err(last_err)
}

/// Map a model id to likely filenames under llama.cpp `models/templates/`.
/// Catalog files look like `meta-llama-Llama-3.1-8B-Instruct.jinja` or
/// `Qwen-Qwen2.5-7B-Instruct.jinja` or `deepseek-ai-DeepSeek-V3.1.jinja`.
pub fn catalog_candidate_filenames(model_id: &str, variant: Option<&str>) -> Vec<String> {
    let mut out = Vec::new();
    let mut push = |s: String| {
        if !out.contains(&s) {
            out.push(s);
        }
    };

    let id = model_id.trim().trim_end_matches('/').to_string();
    let slash = id.replace('/', "-");
    let stripped = strip_quant_suffix(&slash);
    let bare = id.split('/').next_back().unwrap_or(&id).to_string();
    let bare_stripped = strip_quant_suffix(&bare);

    if let Some(v) = variant {
        push(format!("{slash}-{v}.jinja"));
        push(format!("{stripped}-{v}.jinja"));
    }
    push(format!("{slash}.jinja"));
    push(format!("{stripped}.jinja"));
    push(format!("{bare_stripped}.jinja"));
    push(format!("{bare}.jinja"));

    // org-name without fine-tune size variants: try progressively shorter.
    // e.g. Qwen-Qwen2.5-7B-Instruct → Qwen-Qwen2.5-7B → Qwen2.5-7B-Instruct
    for base in [&stripped, &bare_stripped] {
        let parts: Vec<&str> = base.split('-').collect();
        for len in (2..parts.len()).rev() {
            push(format!("{}.jinja", parts[..len].join("-")));
        }
    }

    // Architecture-ish short names already in the catalog (GLM-4.6.jinja, …).
    if !bare_stripped.is_empty() {
        push(format!("{bare_stripped}.jinja"));
    }

    out
}

// ---------------------------------------------------------------------------
// HTTP + cache
// ---------------------------------------------------------------------------

fn http_get(url: &str, opts: &ChatTemplateOptions) -> Result<String, ChatTemplateError> {
    let agent = ureq::AgentBuilder::new()
        .timeout(opts.timeout)
        .user_agent("pulsar-chat-template/0.1")
        .build();

    let mut req = agent.get(url);

    let token = opts
        .hf_token
        .clone()
        .or_else(|| std::env::var("HF_TOKEN").ok())
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok());
    if let Some(t) = token {
        if url.contains("huggingface.co") {
            req = req.set("Authorization", &format!("Bearer {t}"));
        }
    }

    let resp = req.call().map_err(|e| match &e {
        ureq::Error::Status(401, _) => ChatTemplateError::Network(
            "401 gated model — request access and set HF_TOKEN".into(),
        ),
        ureq::Error::Status(404, _) => ChatTemplateError::NotFound(format!("404 {url}")),
        other => ChatTemplateError::Network(other.to_string()),
    })?;

    let mut body = String::new();
    resp.into_reader()
        .take(8 * 1024 * 1024)
        .read_to_string(&mut body)
        .map_err(|e| ChatTemplateError::Io(e.to_string()))?;
    Ok(body)
}

fn cache_root(opts: &ChatTemplateOptions) -> PathBuf {
    if let Some(p) = &opts.cache_dir {
        return p.clone();
    }
    if let Ok(p) = std::env::var("PULSAR_TEMPLATE_CACHE") {
        return PathBuf::from(p);
    }
    if let Some(dirs) = home_cache_dir() {
        return dirs.join("pulsar").join("chat-templates");
    }
    std::env::temp_dir().join("pulsar-chat-templates")
}

fn home_cache_dir() -> Option<PathBuf> {
    // Prefer XDG / LOCALAPPDATA without pulling in the `dirs` crate.
    if let Ok(x) = std::env::var("XDG_CACHE_HOME") {
        return Some(PathBuf::from(x));
    }
    if let Ok(x) = std::env::var("LOCALAPPDATA") {
        return Some(PathBuf::from(x));
    }
    if let Ok(home) = std::env::var("HOME") {
        return Some(PathBuf::from(home).join(".cache"));
    }
    if let Ok(home) = std::env::var("USERPROFILE") {
        return Some(PathBuf::from(home).join("AppData").join("Local"));
    }
    None
}

fn cache_path(model_id: &str, variant: Option<&str>, opts: &ChatTemplateOptions) -> PathBuf {
    let safe = model_id.replace(['/', '\\', ':', ' '], "--");
    let name = match variant {
        Some(v) => format!("{safe}--{v}.jinja"),
        None => format!("{safe}.jinja"),
    };
    cache_root(opts).join(name)
}

fn read_cache(
    model_id: &str,
    variant: Option<&str>,
    opts: &ChatTemplateOptions,
) -> Option<ResolvedChatTemplate> {
    let path = cache_path(model_id, variant, opts);
    let template = fs::read_to_string(&path).ok()?;
    if template.trim().is_empty() {
        return None;
    }
    Some(ResolvedChatTemplate {
        template,
        source: ChatTemplateSource::Cache(path),
        model_id: Some(model_id.to_owned()),
        variant: variant.map(str::to_owned),
    })
}

fn write_cache(
    model_id: &str,
    variant: Option<&str>,
    template: &str,
    opts: &ChatTemplateOptions,
) -> Result<(), ChatTemplateError> {
    let path = cache_path(model_id, variant, opts);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, template)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Apply (Jinja) — use the template once we have it
// ---------------------------------------------------------------------------

/// One chat message for template rendering.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Render messages through a HuggingFace-style Jinja chat template.
///
/// Supports the common subset used by instruct models (`messages`,
/// `add_generation_prompt`, `bos_token`, `eos_token`, `tools`, and simple
/// filters). Full llama.cpp/minja parity is not the goal — on failure the
/// caller should fall back to `ChatMarkers`.
pub fn apply_chat_template(
    template: &str,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
    extra: Option<&Json>,
) -> Result<String, ChatTemplateError> {
    apply_chat_template_ex(
        template,
        messages,
        add_generation_prompt,
        bos_token,
        eos_token,
        None,
        extra,
    )
}

/// Same as [`apply_chat_template`] with optional tool schemas (JSON array).
pub fn apply_chat_template_ex(
    template: &str,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
    tools: Option<&Json>,
    extra: Option<&Json>,
) -> Result<String, ChatTemplateError> {
    // Strip llama.cpp / HF generation-tag extensions minijinja does not know.
    let tmpl = strip_generation_tags(template);

    let mut env = minijinja::Environment::new();
    env.add_filter("tojson", filter_tojson);
    env.add_function("raise_exception", raise_exception);

    env.add_template("chat", &tmpl)
        .map_err(|e| ChatTemplateError::Apply(format!("compile: {e}")))?;
    let t = env
        .get_template("chat")
        .map_err(|e| ChatTemplateError::Apply(e.to_string()))?;

    // Single JSON object context so chat_template_kwargs merge cleanly.
    let mut combined = serde_json::Map::new();
    combined.insert("messages".into(), messages_to_value(messages));
    combined.insert(
        "add_generation_prompt".into(),
        Json::Bool(add_generation_prompt),
    );
    combined.insert(
        "bos_token".into(),
        Json::String(bos_token.unwrap_or("").into()),
    );
    combined.insert(
        "eos_token".into(),
        Json::String(eos_token.unwrap_or("").into()),
    );
    combined.insert("tools".into(), tools.cloned().unwrap_or(Json::Null));
    if let Some(Json::Object(map)) = extra {
        for (k, v) in map {
            combined.insert(k.clone(), v.clone());
        }
    }

    t.render(Json::Object(combined))
        .map_err(|e| ChatTemplateError::Apply(e.to_string()))
}

fn messages_to_value(messages: &[ChatMessage]) -> Json {
    Json::Array(
        messages
            .iter()
            .map(|m| {
                Json::Object(
                    [
                        ("role".to_string(), Json::String(m.role.clone())),
                        ("content".to_string(), Json::String(m.content.clone())),
                    ]
                    .into_iter()
                    .collect(),
                )
            })
            .collect(),
    )
}

fn filter_tojson(value: minijinja::Value) -> Result<String, minijinja::Error> {
    // Prefer serde round-trip when the value is JSON-compatible.
    match serde_json::to_value(&value) {
        Ok(v) => serde_json::to_string(&v).map_err(|e| {
            minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string())
        }),
        Err(_) => Ok(value.to_string()),
    }
}

fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        msg,
    ))
}

/// Remove `{% generation %}` … `{% endgeneration %}` wrappers (llama.cpp).
fn strip_generation_tags(template: &str) -> String {
    let mut out = template.to_string();
    for tag in [
        "{% generation %}",
        "{%- generation %}",
        "{% endgeneration %}",
        "{%- endgeneration %}",
        "{% generation -%}",
        "{%- generation -%}",
        "{% endgeneration -%}",
        "{%- endgeneration -%}",
    ] {
        out = out.replace(tag, "");
    }
    out
}

// ---------------------------------------------------------------------------
// Convenience: resolve + encode via tokenizer
// ---------------------------------------------------------------------------

/// Try to resolve a chat template for this GGUF and apply it, returning
/// the rendered prompt text. Callers tokenize with `Tokenizer::encode`.
pub fn render_chat_prompt_from_gguf(
    g: &Gguf,
    gguf_path: Option<&Path>,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    opts: &ChatTemplateOptions,
) -> Result<(String, ResolvedChatTemplate), ChatTemplateError> {
    let resolved = get_chat_template_from_gguf(g, gguf_path, None, opts)?;
    let bos = g
        .metadata
        .get("tokenizer.ggml.bos_token_id")
        .and_then(|v| v.as_u64())
        .and_then(|_id| {
            // We only have the id here; token string needs the vocab. Leave empty;
            // most templates use special tokens as literal text already in-template.
            None::<String>
        });
    let text = apply_chat_template(
        &resolved.template,
        messages,
        add_generation_prompt,
        bos.as_deref(),
        None,
        None,
    )?;
    Ok((text, resolved))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_quant_q4_k_m() {
        assert_eq!(
            strip_quant_suffix("Qwen2.5-7B-Instruct-Q4_K_M"),
            "Qwen2.5-7B-Instruct"
        );
        assert_eq!(
            strip_quant_suffix("Meta-Llama-3.1-8B-Instruct-IQ2_XXS.gguf"),
            "Meta-Llama-3.1-8B-Instruct"
        );
        assert_eq!(
            strip_quant_suffix("model-UD-Q2_K_XL"),
            "model"
        );
    }

    #[test]
    fn strip_shard_suffix() {
        assert_eq!(
            strip_quant_suffix("DeepSeek-V3-Q4_K_M-00001-of-00015"),
            "DeepSeek-V3"
        );
    }

    #[test]
    fn catalog_names_for_hf_id() {
        let names = catalog_candidate_filenames("meta-llama/Llama-3.1-8B-Instruct", None);
        assert!(names.iter().any(|n| n == "meta-llama-Llama-3.1-8B-Instruct.jinja"));
        assert!(names.iter().any(|n| n.ends_with(".jinja")));
    }

    #[test]
    fn catalog_names_with_variant() {
        let names = catalog_candidate_filenames("CohereForAI/c4ai-command-r-plus", Some("tool_use"));
        assert!(names
            .iter()
            .any(|n| n == "CohereForAI-c4ai-command-r-plus-tool_use.jinja"));
    }

    #[test]
    fn parse_string_chat_template() {
        let cfg = r#"{"chat_template": "{{ bos_token }}{{ messages }}"}"#;
        let (t, v) = chat_template_from_tokenizer_config(cfg, None).unwrap();
        assert!(t.contains("bos_token"));
        assert!(v.is_none());
    }

    #[test]
    fn parse_variant_chat_template() {
        let cfg = r#"{
            "chat_template": [
                {"name": "default", "template": "DEFAULT"},
                {"name": "tool_use", "template": "TOOLS"}
            ]
        }"#;
        let (t, v) = chat_template_from_tokenizer_config(cfg, None).unwrap();
        assert_eq!(t, "DEFAULT");
        assert_eq!(v.as_deref(), Some("default"));
        let (t2, v2) = chat_template_from_tokenizer_config(cfg, Some("tool_use")).unwrap();
        assert_eq!(t2, "TOOLS");
        assert_eq!(v2.as_deref(), Some("tool_use"));
    }

    #[test]
    fn apply_simple_chatml() {
        let tmpl = r#"{% for message in messages %}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"#;
        let msgs = vec![
            ChatMessage {
                role: "user".into(),
                content: "Hi".into(),
            },
        ];
        let out = apply_chat_template(tmpl, &msgs, true, None, None, None).unwrap();
        assert!(out.contains("<|im_start|>user"));
        assert!(out.contains("Hi"));
        assert!(out.contains("<|im_start|>assistant"));
    }

    #[test]
    fn hf_id_from_repo_url() {
        assert_eq!(
            hf_id_from_url("https://huggingface.co/Qwen/Qwen2.5-7B-Instruct"),
            Some("Qwen/Qwen2.5-7B-Instruct".into())
        );
        assert_eq!(
            hf_id_from_url("https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/main"),
            Some("Qwen/Qwen2.5-7B-Instruct".into())
        );
    }

    #[test]
    fn looks_like_hf_id_ok() {
        assert!(looks_like_hf_id("Qwen/Qwen2.5-7B-Instruct"));
        assert!(!looks_like_hf_id("Qwen2.5 only"));
        assert!(!looks_like_hf_id("solo"));
    }
}
