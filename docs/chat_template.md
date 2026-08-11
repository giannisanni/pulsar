# Chat templates

Pulsar formats multi-turn chat in two ways:

1. **ChatMarkers** — hardcoded special-token layouts for known model families
2. **Jinja** — HuggingFace-style templates resolved at load time and rendered
   with minijinja

This document is the full reference: discovery, caching, how **`pulsar-serve`**
and **`pulsar-cli --chat`** apply templates, how to verify a converted GGUF,
the `get-chat-template` tool, library API, env vars, limitations, and
troubleshooting.

**Policy (both binaries):** ChatMarkers are the default and never phone
home. Jinja is **opt-in only** via `--jinja-chat` / `PULSAR_JINJA_CHAT=1`.
There is no `--fetch-template` and no `--no-jinja-chat`: with Jinja on,
resolution rolls over embed → cache → HF → llama.cpp catalog unless
`PULSAR_OFFLINE=1`.

## Code map

| piece | path |
|---|---|
| resolve / fetch / apply | `crates/tokenizer/src/chat_template.rs` |
| special-token encode after Jinja | `Tokenizer::encode_with_specials` in `crates/tokenizer/src/lib.rs` |
| ChatMarkers (per-family render) | `ChatMarkers` in `crates/tokenizer/src/lib.rs` |
| CLI binary | `crates/tokenizer/src/bin/get_chat_template.rs` → `get-chat-template` |
| serve load + per-request encode | `crates/serve/src/main.rs` |
| CLI chat (`--chat` / `--jinja-chat`) | `crates/engine/src/bin/pulsar-cli.rs` |

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
Jinja for known families (see [When Jinja is used](#when-jinja-is-used)).

If `ChatMarkers::resolve` fails and the operator passed `--jinja-chat` with
a resolved template, serve/CLI install `ChatMarkers::jinja_fallback`
(stops / eos only) so generation can still stop correctly. Render methods
on that fallback must not be used. Without `--jinja-chat`, resolve failure
is a hard error.

### 2. Jinja

A Jinja string is resolved (see [Resolution order](#resolution-order)),
rendered with **minijinja**, then tokenized with `encode_with_specials`
(longest-match special vocab entries, BPE on the gaps).

Apply is best-effort: templates that rely on full Jinja2 or llama.cpp
`{% generation %}` blocks may fail; serve/CLI log the error and fall back
to ChatMarkers for that request or turn.

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

## How serve and CLI use the template

### Policy (opt-in; shared)

| mode | encoding | network |
|---|---|---|
| default (no `--jinja-chat`) | ChatMarkers only | **none** (CLI may offline-peek for a load log) |
| `--jinja-chat` / `PULSAR_JINJA_CHAT=1` | Jinja if template resolves | embed → cache → HF → llama.cpp catalog |
| `--jinja-chat` + `PULSAR_OFFLINE=1` | Jinja offline only | embed + local cache only |

GGUF-embedded templates are **not** auto-enabled. Opt in with `--jinja-chat`
so carefully-tuned ChatMarkers stay the default for known families. There
is no separate fetch flag: Jinja on implies the full rollover path.

Applies to:

| binary | flag surface |
|---|---|
| `pulsar-serve` | `--jinja-chat` / `PULSAR_JINJA_CHAT` |
| `pulsar-cli` | `--chat --jinja-chat` / `PULSAR_JINJA_CHAT` (only affects `--chat`) |

### Jinja modes chart

```text
                         ┌──────────────────────────────────────┐
                         │  Opt into model HF/GGUF Jinja layout? │
                         └──────────────────┬───────────────────┘
                                            │
                     no                     │                    yes
            ┌───────────────────────────────┴───────────────────────────────┐
            ▼                                                               ▼
    DEFAULT — ChatMarkers                                      --jinja-chat
    (no flag / PULSAR_JINJA_CHAT unset)                        or PULSAR_JINJA_CHAT=1
            │                                                               │
            ▼                                                               ▼
    Hardcoded family markers                                      Resolve template
    Hy3 · ChatML · Laguna · GLM · …                               (first hit wins)
    Encoding: ChatMarkers                                                  │
    Network: NEVER                                                ┌────────┴────────┐
            │                                                     │                 │
            │                                          no PULSAR_OFFLINE     PULSAR_OFFLINE=1
            │                                                     │                 │
            │                                                     ▼                 ▼
            │                                            Full rollover        Offline only
            │                                            1. GGUF embed        1. GGUF embed
            │                                            2. local cache       2. local cache
            │                                            3. HuggingFace       (no HF / catalog)
            │                                            4. llama.cpp catalog
            │                                                     │                 │
            │                                                     └────────┬────────┘
            │                                                              │
            │                                               found? ────────┼─── missing?
            │                                                  yes         │      no
            │                                                   │          │      │
            │                                                   ▼          │      ▼
            │                                          Encoding: Jinja     │  warn + ChatMarkers
            │                                          minijinja →         │
            │                                          encode_with_specials│
            │                                                   │          │
            └───────────────────────┬───────────────────────────┘          │
                                    │                                      │
                                    ▼                                      │
                           Stop / EOG always from                          │
                           ChatMarkers / stop_ids  ◄───────────────────────┘
                           (Jinja builds the *prompt* only)
```

| Mode | How | Encoding | Network | Template sources |
|---|---|---|---|---|
| **ChatMarkers (default)** | no flag | hardcoded markers | none | n/a |
| **Jinja online** | `--jinja-chat` | Jinja if resolved | after local miss | embed → cache → HF → catalog |
| **Jinja offline** | `--jinja-chat` + `PULSAR_OFFLINE=1` | Jinja if resolved | blocked | embed → cache only |
| **Jinja requested, no template** | flag on, resolve fails | ChatMarkers fallback | attempted (unless offline) | — |
| **Apply error mid-request** | Jinja on, bad apply | ChatMarkers for that request/turn | already resolved | same template |

```sh
# A — ChatMarkers (default, no network)
./target/release/pulsar-serve -m model.gguf
./target/release/pulsar-cli -m model.gguf --chat

# B — Jinja online (embed → cache → HF → catalog)
./target/release/pulsar-serve -m model.gguf --jinja-chat
./target/release/pulsar-cli -m model.gguf --chat --jinja-chat

# C — Jinja offline (embed + cache only)
PULSAR_OFFLINE=1 ./target/release/pulsar-serve -m model.gguf --jinja-chat
PULSAR_OFFLINE=1 ./target/release/pulsar-cli -m model.gguf --chat --jinja-chat
```

**Thinking** (orthogonal, per request when Jinja is on):

```text
--jinja-chat on
      │
      ├─ chat_template_kwargs.enable_thinking: false  → template “thinking off” branch
      ├─ enable_thinking: true                        → template “thinking on” branch
      └─ omitted                                      → template default
            (Laguna official default = true → opens <assistant><think>)
```

Policy reminders:

| Rule | Meaning |
|---|---|
| No `--fetch-template` | Jinja on **is** the network rollover switch |
| No auto-Jinja from GGUF embed | Always opt-in |
| No `--no-jinja-chat` | Default is already ChatMarkers |
| Stops | Never from Jinja; always markers / `stop_ids` |
| CLI multi-turn Jinja | Full re-prefill each turn |
| CLI multi-turn ChatMarkers | Incremental KV across turns |

### Startup (once per process)

```text
load GGUF + tokenizer
        │
        ▼
if --jinja-chat:
    get_chat_template_from_gguf(
        offline = PULSAR_OFFLINE
        // else: embed → cache → HF → llama.cpp catalog
    )
    if template found → jinja_chat = true
    else             → warn; fall back to ChatMarkers
else:
    chat_template = None  // no network; no embedded auto-on
    // CLI may still offline-peek and log "available …; pass --jinja-chat"
    log: ChatMarkers encoding
        │
        ▼
ChatMarkers::resolve(tok)
  // stops for generate; full encode when Jinja off
  // on failure + jinja template → jinja_fallback (stops only)
```

The resolved Jinja **string** stays in memory for the life of the process.
It is not re-fetched per request / turn.

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
| `pulsar-cli --chat` | ChatMarkers default; `--jinja-chat` same resolve + apply path |
| `pulsar-cli -p` / `--tokens` | No chat formatting (raw prompt / ids) |
| `get-chat-template` | Resolve / dump only (no inference) |

### `pulsar-cli --chat` specifics

| mode | multi-turn behavior |
|---|---|
| ChatMarkers (default) | Incremental turns; KV retained across user turns |
| `--jinja-chat` | Full history re-rendered each turn (re-prefill from `pos=0`) so assistant history is template-faithful |
| Jinja apply error | log + ChatMarkers for **that** turn |

---

## When Jinja is used

| condition | behavior |
|---|---|
| default (no flags) | ChatMarkers; no network resolve |
| `--jinja-chat` | Jinja **on**; embed → cache → HF → llama.cpp catalog |
| `--jinja-chat` + `PULSAR_OFFLINE=1` | Jinja offline only (embed + cache) |
| Jinja apply error | log + ChatMarkers for **that** request (serve) or turn (CLI) |

Startup log examples:

```text
# default serve
pulsar-serve: ChatMarkers encoding (pass --jinja-chat to use GGUF/HF Jinja templates)
```

```text
# default CLI --chat (offline peek only)
pulsar: chat template available offline from gguf:tokenizer.chat_template (7646 bytes); pass --jinja-chat to use it
pulsar chat: … ChatMarkers encoding; empty line or Ctrl-D exits
```

```text
# --jinja-chat with embedded template
pulsar-serve: chat template from gguf:tokenizer.chat_template (7646 bytes, …)
pulsar-serve: using Jinja chat template for /v1/chat/completions
```

```text
# --jinja-chat, no embed — rolls over to HF/catalog
pulsar-serve: chat template from huggingface:Qwen/Qwen2.5-7B-Instruct (2507 bytes, …)
pulsar-serve: using Jinja chat template for /v1/chat/completions
```

```text
# CLI with Jinja
pulsar: chat template from gguf:tokenizer.chat_template (7646 bytes)
pulsar chat: … Jinja encoding; empty line or Ctrl-D exits
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
| key missing | not embedded; with `--jinja-chat` Pulsar may still resolve via base model / catalog |

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
# default — ChatMarkers (no Jinja)
./target/release/pulsar-serve -m /path/to/model.gguf
./target/release/pulsar-cli -m /path/to/model.gguf --chat

# opt-in Jinja
./target/release/pulsar-serve -m /path/to/model.gguf --jinja-chat
./target/release/pulsar-cli -m /path/to/model.gguf --chat --jinja-chat
```

With `--jinja-chat`, look for `chat template from …` and:
- serve: `using Jinja chat template for /v1/chat/completions`
- CLI: `Jinja encoding` in the chat banner

Without the flag, serve logs `ChatMarkers encoding`; CLI may offline-peek
with `available offline …; pass --jinja-chat to use it`.

### 4. Debug a live request (rendered text + ids)

```sh
PULSAR_DEBUG_CHAT=1 PULSAR_DEBUG_IDS=1 \
  ./target/release/pulsar-serve -m /path/to/model.gguf --jinja-chat

PULSAR_DEBUG_CHAT=1 PULSAR_DEBUG_IDS=1 \
  ./target/release/pulsar-cli -m /path/to/model.gguf --chat --jinja-chat
```

Serve: call `/v1/chat/completions`. CLI: type a chat turn. Logs show the
Jinja string and (with `PULSAR_DEBUG_IDS`) token ids.

### Decision table

| question | how |
|---|---|
| Did convert embed a template? | `gguf_dump` or `get-chat-template --offline --meta` → `gguf:…` |
| Will serve/CLI **use** Jinja for this file? | only with `--jinja-chat` and a resolved template (embed/cache/HF/catalog) |
| Is encoding ChatMarkers instead? | default, or no `--jinja-chat`, or apply failure fallback |

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

## Environment variables and flags

| var | default | meaning |
|---|---|---|
| `PULSAR_JINJA_CHAT` | unset | `1` = opt-in Jinja on serve and `pulsar-cli --chat` (may use network) |
| `PULSAR_TEMPLATE_CACHE` | platform cache | download cache root (see below) |
| `PULSAR_OFFLINE` | unset | with Jinja: embed + cache only (no HF/catalog) |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | unset | Bearer for gated HF `tokenizer_config.json` |
| `PULSAR_DEBUG_CHAT` | unset | log rendered Jinja prompt text |
| `PULSAR_DEBUG_IDS` | unset | log prompt token id sequences |

Platform cache default (if `PULSAR_TEMPLATE_CACHE` unset):

| OS | path |
|---|---|
| Linux | `$XDG_CACHE_HOME/pulsar/chat-templates` or `~/.cache/pulsar/chat-templates` |
| Windows | `%LOCALAPPDATA%\pulsar\chat-templates` |
| fallback | `$TMP/pulsar-chat-templates` |

Flags (serve and CLI share the same meaning):

| flag | meaning |
|---|---|
| `--jinja-chat` | opt-in Jinja; resolve embed → cache → HF → llama.cpp catalog |
| *(removed)* `--fetch-template` | not needed — Jinja on implies network rollover |
| *(removed)* `--no-jinja-chat` | not needed — default is already ChatMarkers |

```sh
# default: ChatMarkers, no network
./target/release/pulsar-serve -m model.gguf
./target/release/pulsar-cli -m model.gguf --chat

# Jinja: embed → cache → HF → catalog
./target/release/pulsar-serve -m model.gguf --jinja-chat
./target/release/pulsar-cli -m model.gguf --chat --jinja-chat

# Jinja offline only
PULSAR_OFFLINE=1 ./target/release/pulsar-serve -m model.gguf --jinja-chat
PULSAR_OFFLINE=1 ./target/release/pulsar-cli -m model.gguf --chat --jinja-chat
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
  back to ChatMarkers. Our `tojson` filter also accepts Python's
  `ensure_ascii=` kwarg. **GLM-5.2** ships
  [zai-org/GLM-5.2 `chat_template.jinja`](https://huggingface.co/zai-org/GLM-5.2/raw/main/chat_template.jinja)
  whose `tool_to_json` macro does
  `{{ v | tojson(ensure_ascii=False) }}` (around line 12); without the
  kwarg, apply fails with `too many arguments` and serve falls back to
  ChatMarkers. Fixture + unit test:
  `crates/tokenizer/tests/glm52_chat_template.jinja`. Exotic filters or
  helpers outside pycompat can still fail apply.
- `{% generation %}` blocks are stripped, not executed like llama.cpp/minja.
- Tool-call **emission** is multi-format: the MCP loop parses generic
  JSON `<tool_call>`, Hy3 `<tool_call:opensource>`, and DeepSeek DSML
  (`docs/mcp-server.md`). Replay into history still uses the generic
  form when re-encoding past assistant turns.
- Network fetches need outbound HTTPS; air-gapped boxes should rely on
  embedded templates, `PULSAR_OFFLINE` / `get-chat-template --offline`, or
  a pre-seeded cache.
- `pulsar-cli --chat --jinja-chat` uses the same opt-in resolve path as
  serve. Default `--chat` stays ChatMarkers. Jinja multi-turn re-prefills
  each turn so history matches the template (less incremental than
  ChatMarkers KV reuse).
- HF/catalog templates do not auto-enable Jinja on known families (avoids
  regressing carefully-tuned ChatMarkers).

---

## Troubleshooting

| symptom | check |
|---|---|
| `chat template not resolved` | `tokenizer.chat_template` missing? `general.name` / `base_model` / filename? network? `HF_TOKEN`? |
| `401 gated model` | accept license on HF; set `HF_TOKEN` |
| `using Jinja` never printed / still ChatMarkers | default is ChatMarkers; pass `--jinja-chat` (CLI also needs `--chat`) |
| `unknown arg --jinja-chat` | rebuild `pulsar-cli` / `pulsar-serve` from current main |
| `unknown arg --fetch-template` | flag removed; use `--jinja-chat` alone |
| Jinja apply fails every request | `PULSAR_DEBUG_CHAT=1`; dump with `get-chat-template`; omit `--jinja-chat` to stay on ChatMarkers |
| Wrong chat format / bad stops | embedded template vs ChatMarkers mismatch; try the other path |
| Stale template | delete under `PULSAR_TEMPLATE_CACHE` and re-fetch |
| Offline resolve fails | convert did not embed `tokenizer.chat_template`; re-convert with template or seed cache |

---

## Quick reference

```sh
# Build tools
cargo build --release -p tokenizer --bin get-chat-template
cargo build --release -p serve
cargo build --release -p engine --bin pulsar-cli

# Does this GGUF embed a template?
python3 scripts/gguf_dump.py model.gguf | rg chat_template
./target/release/get-chat-template model.gguf --offline --meta

# Default ChatMarkers (no network)
./target/release/pulsar-serve -m model.gguf --port 11435
./target/release/pulsar-cli -m model.gguf --chat

# Opt-in Jinja (embed → cache → HF → catalog)
./target/release/pulsar-serve -m model.gguf --jinja-chat
./target/release/pulsar-cli -m model.gguf --chat --jinja-chat

# Jinja offline only
PULSAR_OFFLINE=1 ./target/release/pulsar-serve -m model.gguf --jinja-chat
PULSAR_OFFLINE=1 ./target/release/pulsar-cli -m model.gguf --chat --jinja-chat

# Debug one completion
PULSAR_DEBUG_CHAT=1 PULSAR_DEBUG_IDS=1 \
  ./target/release/pulsar-serve -m model.gguf --jinja-chat
```

---

## Related

- README: Quick start, “Chat templates”, CLI flags, tuning knobs
- `docs/mcp-server.md` — tool injection on `/v1/chat/completions` (orthogonal
  to which encode path formats messages)
