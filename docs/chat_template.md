# Chat templates

Pulsar formats multi-turn chat in two ways:

1. **ChatMarkers** — hardcoded special-token layouts for known model families
2. **Jinja** — HuggingFace-style templates resolved at load time and rendered
   with minijinja

This document is the full reference: discovery, caching, how `pulsar-serve`
applies a template on every request, how to verify a converted GGUF, CLI
tooling, library API, env vars, limitations, and troubleshooting.

## Code map

| piece | path |
|---|---|
| resolve / fetch / apply | `crates/tokenizer/src/chat_template.rs` |
| special-token encode after Jinja | `Tokenizer::encode_with_specials` in `crates/tokenizer/src/lib.rs` |
| ChatMarkers (per-family render) | `ChatMarkers` in `crates/tokenizer/src/lib.rs` |
| CLI binary | `crates/tokenizer/src/bin/get_chat_template.rs` → `get-chat-template` |
| serve load + per-request encode | `crates/serve/src/main.rs` |
| CLI load log (discovery only) | `crates/engine/src/bin/pulsar-cli.rs` |

Upstream references:

- [llama.cpp `get_chat_template.py`](https://github.com/ggml-org/llama.cpp/blob/master/scripts/get_chat_template.py)
- [llama.cpp `models/templates`](https://github.com/ggml-org/llama.cpp/tree/master/models/templates)
- GGUF key `tokenizer.chat_template` ([gguf.md](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md))

---

## Two encoding paths

### 1. ChatMarkers (default for known families when Jinja is off)

`ChatMarkers::resolve` inspects the vocab for family markers and picks a
render style. Each style pushes special tokens **by id** and encodes only
ordinary text with BPE. Families currently recognized:

| style | markers / shape |
|---|---|
| Hy3 | `<｜hy_User:opensource｜>` / think tags |
| Kimi | `<\|im_user\|>` / `<\|im_middle\|>` / … |
| ChatML | `<\|im_start\|>` / `<\|im_end\|>` (Qwen and kin) |
| Gemma | `<start_of_turn>` / `<end_of_turn>` |
| MiniMax | `]~b]` / `[e~[` / `<mm:think>` |
| Inkling | `<\|message_user\|>` / `<\|message_model\|>` |
| DeepSeek | `<｜User｜>` / `<｜Assistant｜>` / think tags |
| GLM | `[gMASK]<sop>` / `<\|user\|>` / `<\|assistant\|>` |
| Laguna | `<assistant>` / `<think>` paired tags |
| Harmony | `<\|start\|>` / `<\|channel\|>` (gpt-oss) |
| Kimi K3 | XTML `<\|open\|>` / `<\|close\|>` / `<\|end_of_msg\|>` |

Thinking / reasoning effort is controlled on the markers object
(`set_think`, `set_reasoning`) from request fields `reasoning_effort` and
`chat_template_kwargs.enable_thinking`.

ChatMarkers layouts are bit-tuned for stop sets, empty-think openers, and
harmony channels. A downloaded Jinja template can diverge in whitespace or
optional blocks — that is why HF/catalog templates do **not** auto-enable
Jinja for known families (see [When serve uses Jinja](#when-serve-uses-jinja)).

If `ChatMarkers::resolve` fails but a Jinja template was found, serve
installs `ChatMarkers::jinja_fallback` (stops / eos only) and forces Jinja
encoding. Render methods on that fallback must not be used.

### 2. Jinja

A Jinja string is resolved (see [Resolution order](#resolution-order)),
rendered with **minijinja**, then tokenized with `encode_with_specials`
(longest-match special vocab entries, BPE on the gaps).

Apply is best-effort: templates that rely on full Jinja2 or llama.cpp
`{% generation %}` blocks may fail; serve logs the error and falls back to
ChatMarkers for that request.

**Context variables** passed into the template:

| name | meaning |
|---|---|
| `messages` | `[{role, content}, …]` |
| `add_generation_prompt` | always `true` for `/v1/chat/completions` |
| `bos_token` / `eos_token` | vocab strings when ids are known |
| `tools` | optional JSON array of tool function schemas |
| extras | `enable_thinking`, `reasoning_effort`, and other kwargs merged from the request |

**Filters / functions:** `tojson`, `raise_exception`.

**Preprocessing:** `{% generation %}` … `{% endgeneration %}` wrappers
(llama.cpp / minja) are stripped before compile; they are not executed.

**Stops:** generation stop detection still uses `ChatMarkers` /
`tokenizer.stop_ids`. The Jinja template only builds the **input** prompt.

---

## Resolution order

`get_chat_template` / `get_chat_template_from_gguf` /
`get_chat_template_with_options` try sources in order; **first success wins**.

### Spec kinds (`get_chat_template_with_options`)

| input | behavior |
|---|---|
| path ending in `.gguf` that exists | parse header → `get_chat_template_from_gguf` |
| path ending in `.jinja` / `.txt` that exists | read file as template (`source = File`) |
| anything else | treat as HuggingFace-style model id / name |

### GGUF path (`get_chat_template_from_gguf`)

1. **Embedded** — non-empty `tokenizer.chat_template` in GGUF metadata  
   → `ChatTemplateSource::GgufEmbedded`
2. Else build **model-id candidates** (see
   [Quantized GGUFs](#quantized-ggufs-and-base-model-walk)) and for each
   candidate run the remote resolution below.

### Remote / cache resolution (`resolve_for_model_id`)

1. **Disk cache** — previously downloaded `.jinja` under the cache root
2. If a **variant** is set (`tool_use`, …): **llama.cpp catalog first**  
   (HF often only has the default string; catalog has `*-tool_use.jinja`)
3. **HuggingFace** —  
   `https://huggingface.co/{id}/resolve/main/tokenizer_config.json`  
   (Bearer `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` when set; 30s timeout)
4. **llama.cpp catalog** — raw files under  
   `https://raw.githubusercontent.com/ggml-org/llama.cpp/master/models/templates/`

Successful HF or catalog fetches are written to the cache when caching is
enabled (default).

### `tokenizer_config.json` shapes

| shape | behavior |
|---|---|
| `"chat_template": "<jinja string>"` | use as-is |
| `"chat_template": [ { "name", "template" }, … ]` | pick `variant`, else `"default"`; error if neither exists |
| missing / wrong type | `NotFound` / `Parse` |

Broken Llama-3 configs with an extra `}` near
`clean_up_tokenization_spaces` are repaired the same way as llama.cpp’s
Python script.

### Catalog filename candidates

From `org/name` and optional variant, e.g.:

- `meta-llama-Llama-3.1-8B-Instruct.jinja`
- `CohereForAI-c4ai-command-r-plus-tool_use.jinja`
- progressive shorter peels of the bare name

### Sources (`ChatTemplateSource`)

| source | `Display` / log line |
|---|---|
| GGUF header | `gguf:tokenizer.chat_template` |
| Cache file | `cache:<path>` |
| HuggingFace | `huggingface:<org/name>` |
| llama.cpp catalog | `llama.cpp/templates/<file>.jinja` |
| Explicit path | `file:<path>` |

### Result type

```text
ResolvedChatTemplate {
  template: String,           // Jinja source
  source: ChatTemplateSource,
  model_id: Option<String>,   // org/name when known
  variant: Option<String>,    // e.g. tool_use
}
```

---

## Quantized GGUFs and base-model walk

Quant converters often drop or omit `tokenizer.chat_template`. Identity
candidates are collected (deduped, ordered) from:

| input | how |
|---|---|
| `general.base_model.count` + `general.base_model.{i}.{name,organization,repo_url}` | preferred base checkpoints; `repo_url` parsed for `org/name` |
| `general.base_model.name` / `general.base_model` / `general.url` / `general.source.url` | singular keys some writers use |
| `general.organization` + `general.basename` [+ `general.finetune`] | `org/basename` and `org/basename-finetune` |
| `general.name` | as-is if `org/name`, else quant-stripped; may combine with org |
| GGUF path stem | quant-stripped filename; with org if known |
| `general.architecture` | last-resort catalog key |

**Quant suffix peel** (`strip_quant_suffix`) repeatedly removes trailing
tokens such as:

`Q2_K` … `Q8_0`, `IQ2_XXS` … `IQ4_NL`, `UD-Q*_K_XL`, `F16` / `BF16` /
`FP16`, `GGUF`, `imatrix`, and split-shard `-00001-of-00015` tails.

Examples:

| input | stripped |
|---|---|
| `Qwen2.5-7B-Instruct-Q4_K_M` | `Qwen2.5-7B-Instruct` |
| `Meta-Llama-3.1-8B-Instruct-IQ2_XXS.gguf` | `Meta-Llama-3.1-8B-Instruct` |
| `DeepSeek-V3-Q4_K_M-00001-of-00015` | `DeepSeek-V3` |

Each candidate id is resolved independently until one yields a template.

---

## How `pulsar-serve` uses the template

### Startup (once per process)

```text
load GGUF + tokenizer
        │
        ▼
get_chat_template_from_gguf(model.gguf, path)
        │
        ├─ Ok(r)  → chat_template = Some(r)
        │           log: "chat template from {source} ({bytes}…)"
        └─ Err(e) → chat_template = None
                    log: "chat template not resolved; ChatMarkers only"
        │
        ▼
ChatMarkers::resolve(tok)
        │
        ├─ Ok(m)  → markers = m
        └─ Err + template present
                  → jinja_chat = true
                    markers = jinja_fallback(tok)   // stops only
        │
        ▼
if !jinja_chat && source == GgufEmbedded → jinja_chat = true
if jinja_chat && template present
        → log: "using Jinja chat template for /v1/chat/completions"
if jinja_chat && no template
        → log warning; jinja_chat = false
```

Flags at parse time:

| flag / env | effect |
|---|---|
| `PULSAR_JINJA_CHAT` non-empty and not `0` | `jinja_chat = true` before args |
| `--jinja-chat` | force on |
| `--no-jinja-chat` | force off (wins over auto-embedded) |

The resolved Jinja **string** stays in memory for the life of the process.
It is not re-fetched per request.

### Per request: `POST /v1/chat/completions`

```text
JSON body
  messages, tools?, temperature?, stream?,
  reasoning_effort?, chat_template_kwargs.enable_thinking?
        │
        ▼
clone markers; apply reasoning_effort / enable_thinking to ChatMarkers
merge client tools + MCP tools (if --webui-mcp-proxy)
        │
        ▼
encode_messages_auto(...)
        │
        ├─ jinja_chat && chat_template.is_some()?
        │     yes → encode_messages_jinja
        │             │
        │             ├─ Ok(ids) → prompt ids
        │             └─ Err(e)  → log; fall back to encode_messages (ChatMarkers)
        │     no  → encode_messages (ChatMarkers)
        │
        ▼
prefix-cache / prefill / generate
  stop = markers.is_stop(id)   // NOT from Jinja
        │
        ▼
SSE stream or JSON completion
```

**Also used for:** non-stream tool/agent loop re-encodes (same `encode`
closure), web UI chat (same HTTP API).

**Not used for:** raw non-chat paths, stop-id selection (always markers /
tokenizer).

### `encode_messages_jinja` steps

1. Flatten OpenAI messages to `ChatMessage { role, content }`  
   - content may be a string or an array of blocks (`type: text`, tool_result, …)  
   - assistant `tool_calls` appended as `<tool_call>…</tool_call>` text  
   - `role: tool` rewritten as a user turn with `<tool_result id="…">…`
2. Build optional `tools` JSON (function schemas only)
3. Merge extras: `enable_thinking`, `reasoning_effort`
4. `apply_chat_template_ex(template, messages, add_generation_prompt=true, bos, eos, tools, extras)`
5. If `PULSAR_DEBUG_CHAT` is set, log the rendered string
6. `tok.encode_with_specials(rendered)` → prompt token ids  
   If `PULSAR_DEBUG_IDS` is set, log the ids

### End-to-end picture (embedded GGUF template)

```text
Client:  { "messages": [{"role":"user","content":"Hello"}] }
            │
            ▼
     minijinja + tokenizer.chat_template from GGUF
     (markers, roles, optional think/tools blocks, assistant open)
            │
            ▼
     encode_with_specials → [u32, …]
            │
            ▼
     engine prefill + generate → stream / JSON response
```

### What uses what

| surface | template? |
|---|---|
| `/v1/chat/completions` | Yes — Jinja or ChatMarkers |
| Web UI chat | Yes — same endpoint |
| MCP agentic re-encode turns | Yes — same `encode` path |
| Stop / EOG detection | Markers / `stop_ids` only |
| `pulsar-cli --chat` | Discovery **logged**; encoding still ChatMarkers today |
| `get-chat-template` | Resolve / dump only (no inference) |

---

## When serve uses Jinja

| condition | behavior |
|---|---|
| GGUF embeds `tokenizer.chat_template` | Jinja **on** by default |
| `ChatMarkers::resolve` fails but a template was found | Jinja **on** (fallback markers for stops) |
| `--jinja-chat` or `PULSAR_JINJA_CHAT=1` | Jinja **on** if a template exists |
| Template only from HF / catalog (not embedded) | Jinja **off** unless forced (protects ChatMarkers parity) |
| `--no-jinja-chat` | force ChatMarkers even with an embedded template |
| Jinja apply error at request time | log + ChatMarkers for **that** request |

Startup log examples:

```text
pulsar-serve: chat template from gguf:tokenizer.chat_template (7646 bytes, model_id=…)
pulsar-serve: using Jinja chat template for /v1/chat/completions
```

```text
pulsar-serve: chat template from huggingface:Qwen/Qwen2.5-7B-Instruct (2507 bytes, model_id=…)
# (no "using Jinja" line unless --jinja-chat)
```

```text
pulsar-serve: chat template not resolved (…); ChatMarkers only
```

```text
pulsar-serve: jinja chat template apply failed (…); falling back to ChatMarkers
```

---

## How to check if a converted model has / uses a template

### 1. Inspect GGUF metadata (did convert embed one?)

```sh
python3 scripts/gguf_dump.py /path/to/model.gguf \
  | rg -i 'chat_template|general\.(name|basename|base_model|organization)'
```

| result | meaning |
|---|---|
| `tokenizer.chat_template = {%- …` (long Jinja) | **Embedded** — convert ships a template |
| key missing | not embedded; Pulsar may still fetch via base model / catalog |

Example (embedded):

```text
general.name = DeepSeek V4 Flash
tokenizer.chat_template = {%- if not add_generation_prompt is defined -%}
```

### 2. `get-chat-template --meta` (what would resolve?)

```sh
cargo build --release -p tokenizer --bin get-chat-template

# full resolution (may hit network)
./target/release/get-chat-template /path/to/model.gguf --meta

# embedded + cache only (proves convert baked it in)
./target/release/get-chat-template /path/to/model.gguf --offline --meta
```

| `source:` line | meaning |
|---|---|
| `gguf:tokenizer.chat_template` | from your convert |
| `cache:…` | previously downloaded |
| `huggingface:org/name` | fetched from HF (not in GGUF) |
| `llama.cpp/templates/….jinja` | from catalog (not in GGUF) |
| error | nothing embedded and nothing recoverable |

### 3. Serve / CLI load logs (what will inference use?)

```sh
./target/release/pulsar-serve -m /path/to/model.gguf
```

Look for `chat template from …` and optionally
`using Jinja chat template for /v1/chat/completions`.

### 4. Debug a live request (rendered text + ids)

```sh
PULSAR_DEBUG_CHAT=1 PULSAR_DEBUG_IDS=1 \
  ./target/release/pulsar-serve -m /path/to/model.gguf
```

Then call `/v1/chat/completions`. Logs show the Jinja string and token ids.

### Decision table

| question | how |
|---|---|
| Did convert embed a template? | `gguf_dump` or `get-chat-template --offline --meta` → `gguf:…` |
| Will serve **use** Jinja for this file? | load log `using Jinja…` (embedded auto-on) |
| Is encoding ChatMarkers instead? | no Jinja line, or `--no-jinja-chat`, or apply failure fallback |

---

## CLI: `get-chat-template`

No GPU required. Works from a HF id, free-form name, `.gguf` path, or local
`.jinja` / `.txt` file.

```sh
cargo build --release -p tokenizer --bin get-chat-template

# HuggingFace model id → template on stdout
./target/release/get-chat-template microsoft/Phi-3.5-mini-instruct

# variant (catalog preferred when named)
./target/release/get-chat-template CohereForAI/c4ai-command-r-plus tool_use

# quantized GGUF → base model walk
./target/release/get-chat-template ./Qwen2.5-7B-Instruct-Q4_K_M.gguf --meta

# write to file
./target/release/get-chat-template Qwen/Qwen2.5-7B-Instruct --save qwen.jinja

# offline: embedded + cache only
./target/release/get-chat-template ./model.gguf --offline --meta
```

| flag | meaning |
|---|---|
| `MODEL_ID \| MODEL.gguf [VARIANT]` | positional |
| `--save PATH` | write template to PATH instead of stdout |
| `--meta` | source / model_id / variant / bytes on **stderr**; template on **stdout** |
| `--offline` | set `PULSAR_OFFLINE` for this run |
| `-h` / `--help` | usage |

---

## Environment variables and serve flags

| var | default | meaning |
|---|---|---|
| `PULSAR_JINJA_CHAT` | unset | force Jinja encoding when a template is resolved (`0` / empty = off) |
| `PULSAR_TEMPLATE_CACHE` | platform cache | download cache root (see below) |
| `PULSAR_OFFLINE` | unset | skip HF and catalog HTTP (embedded + cache only) |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | unset | Bearer for gated HF `tokenizer_config.json` |
| `PULSAR_DEBUG_CHAT` | unset | log rendered Jinja prompt text each request |
| `PULSAR_DEBUG_IDS` | unset | log prompt token id sequences |

Platform cache default (if `PULSAR_TEMPLATE_CACHE` unset):

| OS | path |
|---|---|
| Linux | `$XDG_CACHE_HOME/pulsar/chat-templates` or `~/.cache/pulsar/chat-templates` |
| Windows | `%LOCALAPPDATA%\pulsar\chat-templates` |
| fallback | `$TMP/pulsar-chat-templates` |

Serve flags:

| flag | meaning |
|---|---|
| `--jinja-chat` | same as `PULSAR_JINJA_CHAT=1` |
| `--no-jinja-chat` | never use Jinja for encoding |

```sh
# force Jinja for HF/catalog-resolved templates
PULSAR_JINJA_CHAT=1 ./target/release/pulsar-serve -m model.gguf
# or
./target/release/pulsar-serve -m model.gguf --jinja-chat

# keep ChatMarkers even when GGUF embeds a template
./target/release/pulsar-serve -m model.gguf --no-jinja-chat
```

---

## Cache layout

Cached files are named from the model id with `/` → `--`:

```text
<cache_root>/
  Qwen--Qwen2.5-7B-Instruct.jinja
  CohereForAI--c4ai-command-r-plus--tool_use.jinja
```

Delete a file to force re-fetch. Air-gapped boxes can pre-seed this directory
or rely on embedded `tokenizer.chat_template` only.

---

## Request fields (OpenAI-compatible)

On `/v1/chat/completions`:

| field | ChatMarkers path | Jinja path |
|---|---|---|
| `messages` | required; roles system/user/assistant/tool | same; content string or content-block array |
| `tools` | injects system-side tool contract text | passed as `tools` into template |
| `reasoning_effort` | `none`/`off` → think off; else `set_reasoning` | merged into template kwargs |
| `chat_template_kwargs.enable_thinking` | `set_think(bool)` | `enable_thinking` in template kwargs |
| `stream` | SSE vs JSON body | same (after encode) |
| `temperature` / `top_p` / `min_p` / `seed` / `max_tokens` | sampling only | sampling only |

MCP tools (when `--webui-mcp-proxy` is on) are merged into `tools` before
encode; see `docs/mcp-server.md`.

---

## Library API (`tokenizer` crate)

```rust
use tokenizer::{
    get_chat_template, get_chat_template_from_gguf, get_chat_template_with_options,
    apply_chat_template, apply_chat_template_ex,
    ChatMessage, ChatTemplateOptions, ChatTemplateSource, ResolvedChatTemplate,
    ChatTemplateError,
};

// HF id, path, or .jinja file
let r = get_chat_template("Qwen/Qwen2.5-7B-Instruct", None)?;

// From an already-parsed GGUF (serve / cli load path)
let opts = ChatTemplateOptions::default();
let r = get_chat_template_from_gguf(&gguf, Some(path), None, &opts)?;

// Render + tokenize
let text = apply_chat_template(
    &r.template,
    &[ChatMessage { role: "user".into(), content: "Hi".into() }],
    true,   // add_generation_prompt
    None,   // bos_token
    None,   // eos_token
    None,   // extra kwargs JSON
)?;
let ids = tok.encode_with_specials(&text);
```

### `ChatTemplateOptions`

| field | default | meaning |
|---|---|---|
| `use_llama_cpp_catalog` | `true` | try GitHub catalog |
| `use_cache` | `true` | read/write cache dir |
| `cache_dir` | `None` → env / platform | override cache root |
| `offline` | `PULSAR_OFFLINE` set? | skip network |
| `hf_token` | `None` → env | HF Bearer |
| `timeout` | 30s | HTTP connect/read |

### Helpers

| function | use |
|---|---|
| `strip_quant_suffix(name)` | peel quant / shard suffixes |
| `model_id_candidates(gguf, path)` | ordered HF id guesses |
| `catalog_candidate_filenames(id, variant)` | catalog `.jinja` names |
| `chat_template_from_tokenizer_config(json, variant)` | parse HF config body |
| `render_chat_prompt_from_gguf(…)` | resolve + apply in one call |

### Errors (`ChatTemplateError`)

`NotFound`, `Network`, `Parse`, `Io`, `Apply`, `InvalidModelId`.

### Unit tests

`cargo test -p tokenizer --lib` covers quant strip, catalog names, config
variants, simple ChatML apply, and HF id URL parsing.

---

## `encode_with_specials`

After Jinja renders marker text (e.g. `<|im_start|>`, `<｜User｜>`), plain
BPE may split control strings into bytes. `encode_with_specials`:

1. Builds a list of vocab entries that look like control markers at
   tokenizer load (longest first)
2. Left-to-right longest match → push special id
3. Ordinary spans → normal `encode` (BPE)

Heuristic specials include strings starting with `<|`, `<｜`, `<start_`,
`]~`, `[e~`, `[gMASK]`, `<think>`, etc. Unusual marker text may still
BPE-split.

---

## Limitations

- minijinja is a **subset** of Jinja2. We register
  `minijinja-contrib` **pycompat** so common Python string methods
  (`.format`, `.strip`, `.startswith`, …) work — without that, templates
  like Hy3 fail with `string has no method named format` and serve falls
  back to ChatMarkers. Exotic filters, macros with non-JSON types, or
  helpers outside pycompat can still fail apply.
- `{% generation %}` blocks are stripped, not executed like llama.cpp/minja.
- Tool-call **emission** is multi-format: the MCP loop parses generic
  JSON `<tool_call>`, Hy3 `<tool_call:opensource>`, and DeepSeek DSML
  (`docs/mcp-server.md`). Replay into history still uses the generic
  form when re-encoding past assistant turns.
- Network fetches need outbound HTTPS; air-gapped boxes should rely on
  embedded templates, `--offline`, or a pre-seeded cache.
- `pulsar-cli --chat` currently logs discovery but still encodes with
  ChatMarkers (serve is the Jinja consumer).
- HF/catalog templates do not auto-enable Jinja on known families (avoids
  regressing carefully-tuned ChatMarkers).

---

## Troubleshooting

| symptom | check |
|---|---|
| `chat template not resolved` | `tokenizer.chat_template` missing? `general.name` / `base_model` / filename? network? `HF_TOKEN`? |
| `401 gated model` | accept license on HF; set `HF_TOKEN` |
| `using Jinja` never printed | template only from HF/catalog → pass `--jinja-chat`; or `--no-jinja-chat` forced off |
| Jinja apply fails every request | `PULSAR_DEBUG_CHAT=1`; dump with `get-chat-template`; try `--no-jinja-chat` |
| Wrong chat format / bad stops | embedded template vs ChatMarkers mismatch; try the other path |
| Stale template | delete under `PULSAR_TEMPLATE_CACHE` and re-fetch |
| Offline resolve fails | convert did not embed `tokenizer.chat_template`; re-convert with template or seed cache |

---

## Quick reference

```sh
# Build tools
cargo build --release -p tokenizer --bin get-chat-template
cargo build --release -p serve

# Does this GGUF embed a template?
python3 scripts/gguf_dump.py model.gguf | rg chat_template
./target/release/get-chat-template model.gguf --offline --meta

# Serve with embedded template (Jinja auto-on)
./target/release/pulsar-serve -m model.gguf --port 11435

# Force / block Jinja
./target/release/pulsar-serve -m model.gguf --jinja-chat
./target/release/pulsar-serve -m model.gguf --no-jinja-chat

# Debug one completion
PULSAR_DEBUG_CHAT=1 PULSAR_DEBUG_IDS=1 \
  ./target/release/pulsar-serve -m model.gguf
```

---

## Related

- README: Quick start, “Chat templates”, tuning knobs
- `docs/mcp-server.md` — tool injection on `/v1/chat/completions` (orthogonal
  to which encode path formats messages)
