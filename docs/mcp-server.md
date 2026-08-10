# pulsar-serve MCP support

`pulsar-serve` can act as an MCP (Model Context Protocol) **client**: it connects to one or more MCP servers, exposes their tools to the loaded model, executes the model's `<tool_call>` blocks against those servers, and feeds the results back into the conversation. The whole feature is opt-in, gated end-to-end by a single flag. Server-side, in Rust (the `rmcp` SDK pinned at 3.0.1), isolated behind a sync façade so the deliberately-sync HTTP server in `crates/serve/src/main.rs` stays sync everywhere except one leaf (`crates/serve/src/mcp.rs`).

This mirrors what llama.cpp's webui exposes behind `--ui-mcp-proxy`, but runs entirely on the server: pulsar-serve is vanilla Rust with no browser execution path, so the faithful analog is a server-side hub plus a vanilla-JS management surface in the web UI.

## Feature gate

| flag | default | meaning |
|---|---|---|
| `--webui-mcp-proxy` | off | **The** on/off switch. Presence enables the `/mcp/*` routes, injects enabled tools into `/v1/chat/completions`, and un-hides the MCP group in the web UI sidebar. Absence → every `/mcp/*` route returns 404, no tools are injected, the sidebar group stays hidden. Zero behavioral change when off. |
| `--mcp-config FILE` | `./mcp.json` | Path to the MCP config file. Created/rewritten on every web-ui edit; read at startup. |

Both flags are parsed in `main.rs` alongside the other `pulsar-serve` args (`-m`, `--port`). The hub is constructed only when `--webui-mcp-proxy` is set:

```rust
let mcp = if webui_mcp_proxy {
    let m = mcp::McpHub::new(mcp_config.as_deref());
    m.connect_all();
    Some(m)
} else { None };
```

When `mcp` is `None`, all five MCP routes fall through to a 404 (`match &mcp { Some(m) => …, None => respond_json(404, …) }`), so a `GET /mcp/status` probe from the web UI returns 404 and the sidebar group is never shown.

## Config file (`mcp.json`)

Claude-Code-compatible shape: `{ "mcpServers": { <name>: <server-cfg, …> } }`. Each entry is an untagged enum — the presence of `command` selects stdio, the presence of `url` selects remote streamable-http. `mcp.rs` derives `McpServerCfg`:

```jsonc
{
  "mcpServers": {
    // stdio: a child process speaking MCP over stdin/stdout
    "everything": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-everything"],
      "env": { "FOO": "bar" },          // optional, extra env for the child
      "allow": [],                       // optional tool allowlist
      "deny": [],                        // optional tool denylist
      "disabled": false,                 // optional, skip at connect time
      "timeout_s": 30                    // optional, per-call timeout (default 30s)
    },
    // remote: streamable-http (the verified path; sse folds into this since rmcp 0.11)
    "SearchTool": {
      "url": "http://127.0.0.1:8765/mcp",
      "transport": "http",               // optional, "http" (streamable); sse is accepted, routed to streamable
      "headers": {},                     // optional, e.g. { "Authorization": "Bearer …" }
      "allow": [],
      "deny": [],
      "disabled": false,
      "timeout_s": 30
    }
  }
}
```

All keys except the discriminating one (`command` / `url`) are optional and `#[serde(default)]`. The file is written atomically (tmp + rename) on every web-ui mutation (`upsert_server`, `remove_server`), so edits survive restarts and can also be hand-edited offline.

## HTTP routes

All five return a JSON error body `{"error":{"message":"--webui-mcp-proxy not enabled"}}` with status 404 when the flag is off.

| method | path | request body | response |
|---|---|---|---|
| `GET` | `/mcp/status` | — | `{ "servers": [ … ] }` (see below) |
| `GET` | `/v1/tools` | — | `{ "object": "list", "data": [ … OpenAI function specs … ] }` |
| `POST` | `/mcp/server` | `{ "name": "<server>", "config": <McpServerCfg> }` | refreshed `status_json()`; 400 on empty name / bad config |
| `POST` | `/mcp/server/delete` | `{ "name": "<server>" }` | refreshed `status_json()` |
| `POST` | `/mcp/toggle` | `{ "tool": "<namespaced>", "disabled": true\|false }` | `{ "ok": true }` |

`/mcp/status` per-server object:

```jsonc
{
  "name": "SearchTool",             // config key
  "transport": "http",             // "http" | "stdio"
  "connected": true,               // client handshake succeeded
  "disabled": false,               // cfg.disabled flag
  "error": null,                   // last connect error, if any
  "server_name": "SearXNG MCP",    // handshake-advertised name (null until init ok)
  "server_version": "1.4.0",       // handshake-advertised version
  "protocol_version": "2025-11-25",// negotiated MCP protocol version
  "connect_ms": 312,               // wall-clock ms of the last successful handshake
  "tools": [
    { "name": "search_searxng",
      "namespaced": "SearchTool__search_searxng",   // what the model sees
      "description": "…",
      "enabled": true }                            // false if toggled off or blocked by allow/deny
  ],
  "logs": [                        // rolling per-server connection log (newest last, capped 64)
    { "t": "14:08:21", "ok": true,  "msg": "connecting via http" },
    { "t": "14:08:21", "ok": true,  "msg": "handshake ok in 312ms — SearXNG MCP 1.4.0" },
    { "t": "14:08:21", "ok": true,  "msg": "listed 3 tools" }
  ],
  "config": { /* the raw McpServerCfg, so the edit form can repopulate every field */ }
}
```

## Web UI

The management surface lives in the left sidebar as its own group, between **Connection** and **Runtime**. On load, `index.html` does `fetch('/mcp/status')`; HTTP 200 un-hides the group, 404 (flag off) leaves it hidden — no HTML swapping, just a probe-to-show.

- **Server list**: one collapsible card per configured server, rendered inline (no popup). The card title shows the **handshake-advertised server name** (read from the MCP `initialize` result and falling back to the config key if the server advertises none), with transport, version, and protocol-version badges. Each card has an **on/off pill** toggle (same `.cpu-toggle` style as the CPU Lane / MTP toggles, re-upserts with the `disabled` flag flipped), plus **Edit** and **Remove**.
- **Connection log**: a collapsible `Connection log (N)` detail per card shows the per-server log (`logs` in `/mcp/status`), newest last, capped at 64 lines, with the last handshake latency (`connect_ms`) next to the header. Each connect cycle appends a `connecting` → `handshake ok in Nms — {name} {ver}` → `listed N tools` trace, or a `handshake failed in Nms: …` line on error.
- **Add / edit** (`<details>` form, expands inline): name, transport selector (`http` / `stdio`), and conditional fields — url + headers for http, command + args + env for stdio. Save → `POST /mcp/server`, which reconnects and re-renders status.
- **Per-tool toggle**: each server's tool list has enable/disable checkboxes → `POST /mcp/toggle`.
- When any tool is enabled, chat requests are forced to `stream:false` so the server runs the full non-stream agentic loop and returns one final `chat.completion`.

## How the agentic loop works

Non-stream branch of `handle_chat` only (`main.rs`, `MAX_TURNS = 8`). The stream branch stays single-turn for now.

1. Enabled MCP tools are merged into the request's `tools` array as OpenAI function specs, namespaced `server__tool`. The existing prompt builder (`encode_messages`) injects the `# Tools` schemas and already knows how to replay `tool_calls` / `tool` message roles.
2. Generate. The model emits tool-call markup; `extract_tool_calls` (in
   `tool_calls.rs`) returns `(clean_text, Vec<(name, args_json)>)`. Accepted
   formats:
   - **Generic JSON** (what the `# Tools` system prompt teaches):  
     `<tool_call>{"name":"SearchTool__search_searxng","arguments":{…}}</tool_call>`
   - **Hy3 opensource** (native Jinja / chat template):  
     `<tool_calls:opensource><tool_call:opensource>NAME<tool_sep:opensource>…arg_key/arg_value…</tool_call:opensource></tool_calls:opensource>`
   - **DeepSeek DSML** (fullwidth `｜`):  
     `<｜DSML｜tool_calls…><｜DSML｜invoke name="…"><｜DSML｜parameter name="k">v</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>`

   Bare or alias names (`search_searxng`, `web_search`) are rewritten to an
   enabled `server__tool` id by `McpHub::resolve_tool_name` when the match is
   unique (or when only one tool is enabled).

   **DeepSeek V4 / DSML history:** after dispatch, tool turns are re-fed as
   `<tool_result>…</tool_result>` user content and prior assistant calls are
   replayed as DSML (not Hermes JSON). Replaying the generic form left the
   model unable to continue and the web UI showed `(empty)`.
3. For each call, `mcp.dispatch_sync(name, args)`:
   - splits on the first `__` → `(server, tool)`,
   - checks `allow`/`deny` (deny wins; a non-empty `allow` not containing the tool denies it; otherwise permitted),
   - `tokio::time::timeout(timeout, client.call_tool(...))` on the private tokio runtime (default 30s, per-call),
   - returns the textual/JSON result, or an `error: …` string (timeout, call failure, malformed name, not permitted).
4. The assistant message (with `tool_calls`) plus one `tool` role per result is appended, messages are re-encoded, and the loop regenerates. Each turn after the first reuses the prefix-cache machinery, so only the suffix is re-prefilled.
5. Stops when the model emits no tool calls or `MAX_TURNS` is reached; the final assistant text is returned as one non-stream `chat.completion`.

`dispatch_sync` is a synchronous public method that internally does `self.rt.block_on(async { … })` on the hub's private multi-thread tokio runtime. Every public method on `McpHub` is similarly sync — that is the whole async↔sync bridge.

## Transports

- **streamable-http** (`url`): the primary, end-to-end-verified path. Uses `StreamableHttpClientTransport::from_uri(url)`; optional `headers` (e.g. auth bearer).
- **stdio** (`command` + `args`): spawns a child process; MCP runs over its stdin/stdout (`TokioChildProcess` + `ConfigureCommandExt`). Optional `env` augments the child environment.
- **sse**: accepted in config but routed to the streamable-http transport — `transport-sse-client` was removed in rmcp 0.11 and SSE now folds into streamable-http. Not separately verified live.

## Security / trust boundary

- **stdio spawns arbitrary processes on the serve host.** The web UI's add-server form can enter any `command`/`args`. This is acceptable for a local single-user server behind an opt-in flag, and is the intended UX (you are configuring your own tools). Do **not** expose `pulsar-serve --webui-mcp-proxy` to untrusted networks without your own authz in front — the `/mcp/*` routes are unauthenticated localhost-only like the rest of the server.
- **Unrecognized `Host` headers are rejected** (403) on every request, which is what stops DNS rebinding. An attacker who points `evil.example` at `127.0.0.1` gets a browser that considers us same-origin: it sends `Origin` *and* `Host` both reading `evil.example`, so an origin-equals-host test passes and the page can both POST to `/mcp/server` and read replies from `/mcp/status`. Only an allowlist catches that. Allowed by default: `127.0.0.1`, `localhost`, `[::1]` on the serving port, plus the `--host` value when it is not a wildcard. Reaching the server under any other name (a Tailscale hostname, a reverse proxy) needs `PULSAR_ALLOWED_HOSTS=name:port[,name:port]`; the rejection log names the allowed set.
- **Cross-origin POSTs are rejected** (403) for every route, `/mcp/*` included. Because `/mcp/server` spawns a process, a page you merely *visit* would otherwise have been one `fetch` away from running code on this host: a `text/plain` POST is a CORS "simple request", so the browser sends it to `127.0.0.1` without a preflight and the side effect lands even though the attacker never reads the response. The guard allows requests with no `Origin` header (curl, the OpenAI SDKs, Claude Code) and requests whose `Origin` matches the `Host` they arrived on (the web UI itself). It is not a substitute for authz on an untrusted network.
- **Per-call timeout** (default 30s) bounds a hung tool call; `allow`/`deny` per server bounds which tools the model may invoke.
- `McpHub` is a single process-global hub. pulsar-serve is sequential single-user localhost (one request at a time), so there is no per-request isolation; the ceiling is per-session hubs if multi-tenant ever matters (marked `ponytail:` in `mcp.rs`).

## Worked example (remote http)

Start an MCP server somewhere reachable (here, a SearXNG wrapper on `:8765`), then:

```bash
cat > mcp.json <<'EOF'
{ "mcpServers": { "SearchTool": { "url": "http://127.0.0.1:8765/mcp" } } }
EOF
./target/release/pulsar-serve -m model.gguf --port 11435 \
    --webui-mcp-proxy --mcp-config ./mcp.json
```

```bash
# tools are visible, namespaced:
curl -s http://127.0.0.1:11435/v1/tools | jq '.data[].function.name'
# "SearchTool__search_searxng"

# a prompt that triggers a lookup runs the loop and returns a grounded answer:
curl -s http://127.0.0.1:11435/v1/chat/completions -d '{
  "messages": [{"role":"user","content":"What does rust-lang.org say Rust is?"}],
  "stream": false
}' | jq -r '.choices[0].message.content'
```

Server log shows `mcp dispatch SearchTool__search_searxng` on the turn that triggered the call. Measured on Qwen3.6-35B-A3B against a local SearXNG MCP: full round-trip (prompt → tool_call → dispatch → grounded answer) in ~22s.

## Status

Done: feature gate (`--webui-mcp-proxy` / `--mcp-config`) · sync rmcp bridge (`McpHub`, private tokio runtime, `block_on`) · stdio + streamable-http transports · `connect_all` / `connect_one` / `dispatch_sync` / `status_json` / `toggle` / `upsert_server` / `remove_server` / atomic `save_config` · handshake identity auto-detect (server name/version + negotiated protocol version via `peer_info`, with `connect_ms`) · per-server rolling connection log (capped 64) surfaced as a `logs` array · `/mcp/status`, `/v1/tools`, `/mcp/server`, `/mcp/server/delete`, `/mcp/toggle` routes · non-stream agentic loop (MAX_TURNS=8, prefix-cache reuse, allow/deny, per-call timeout) · web UI sidebar (probe-to-show, collapsible cards with handshake-identified titles + connection log, on/off pill toggle matching CPU Lane / MTP, add/edit/remove, per-tool toggle) · end-to-end verification on a remote SearXNG MCP.

Not yet:

- Streamed multi-turn tool loop. The agentic loop is **non-stream only**; the stream branch is single-turn. Upgrade when streamed tool-calling is wanted.
- `sse` transport is accepted in config but routed through streamable-http; not separately verified live.
- stdio transport verified by code path only, not by a live `server-everything` round-trip on the reference box.
