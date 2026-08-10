//! Parse model-emitted tool-call blocks into (clean_text, Vec<(name, args_json)>).
//!
//! Models disagree on syntax. Pulsar historically taught a generic JSON form;
//! DeepSeek V4 / Hy3 (and their GGUF Jinja templates) emit native formats
//! instead. The agentic MCP loop must accept all of them or the raw markup
//! leaks into the user reply and no tool runs.
//!
//! Supported:
//! 1. Generic: `<tool_call>{"name":"…","arguments":{…}}</tool_call>`
//! 2. Hy3 opensource:
//!    `<tool_calls:opensource><tool_call:opensource>NAME<tool_sep:opensource>…`
//! 3. DeepSeek DSML — fullwidth `｜` (U+FF5C) **or** ASCII `|`:
//!    `<｜DSML｜tool_calls…>` / `<|DSML|tool_calls…>` with invoke + parameter

/// Byte markers that open a tool-call region (stream holdback / silence).
pub const TOOL_OPEN_MARKERS: &[&[u8]] = &[
    b"<tool_calls:opensource>",
    b"<tool_call:opensource>",
    b"<tool_call>",
    // DeepSeek DSML fullwidth: "<｜DSML｜"
    b"<\xef\xbd\x9cDSML\xef\xbd\x9c",
    // ASCII pipe variant some decoders emit
    b"<|DSML|",
];

/// Split assistant text into (visible content without tool markup, parsed calls).
pub fn extract_tool_calls(text: &str) -> (String, Vec<(String, String)>) {
    // Normalize DSML delimiters so fullwidth and ASCII pipes share one path.
    let norm = normalize_dsml_delims(text);
    let mut clean = String::new();
    let mut calls = Vec::new();
    let mut rest = norm.as_str();

    while !rest.is_empty() {
        let next = next_tool_region(rest);
        let Some((kind, start)) = next else {
            clean.push_str(rest);
            break;
        };
        clean.push_str(&rest[..start]);
        let after = &rest[start..];
        match kind {
            ToolKind::Generic => match take_generic(after) {
                Some((name, args, consumed)) => {
                    calls.push((name, args));
                    rest = &after[consumed..];
                }
                None => {
                    // skip one char to avoid infinite loop on garbage
                    let (skip, rem) = skip_unit(after);
                    clean.push_str(skip);
                    rest = rem;
                }
            },
            ToolKind::Hy3 => match take_hy3(after).or_else(|| {
                take_hy3_one(after).map(|(n, a, c)| (vec![(n, a)], c))
            }) {
                Some((batch, consumed)) => {
                    calls.extend(batch);
                    rest = &after[consumed..];
                }
                None => {
                    let (skip, rem) = skip_unit(after);
                    clean.push_str(skip);
                    rest = rem;
                }
            },
            ToolKind::Dsml => match take_dsml(after) {
                Some((batch, consumed)) => {
                    calls.extend(batch);
                    rest = &after[consumed..];
                }
                None => {
                    // Strip a whole DSML-looking region so markup never reaches the UI,
                    // even if invoke/parameter parse fails.
                    if let Some(n) = dsml_region_len(after) {
                        rest = &after[n..];
                    } else {
                        let (skip, rem) = skip_unit(after);
                        clean.push_str(skip);
                        rest = rem;
                    }
                }
            },
        }
    }

    // Safety net: any residual DSML / hy3 / generic tool markup must not leak.
    let clean = strip_residual_tool_markup(&clean);
    (clean.trim_end().to_string(), calls)
}

/// Earliest index in `bytes` where a tool-open marker begins, if any.
pub fn find_tool_open(bytes: &[u8]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for m in TOOL_OPEN_MARKERS {
        if let Some(p) = bytes.windows(m.len()).position(|w| w == *m) {
            best = Some(best.map_or(p, |b| b.min(p)));
        }
    }
    best
}

/// Length of a trailing partial prefix of any tool-open marker (stream holdback).
pub fn tool_open_holdback(bytes: &[u8]) -> usize {
    let mut hold = 0;
    for m in TOOL_OPEN_MARKERS {
        for k in (1..m.len().min(bytes.len() + 1)).rev() {
            if bytes.ends_with(&m[..k]) {
                hold = hold.max(k);
                break;
            }
        }
    }
    hold
}

// ---------------------------------------------------------------------------
// internals
// ---------------------------------------------------------------------------

/// Map fullwidth `｜` (U+FF5C) to ASCII `|` only inside DSML-looking tags so
/// we can match both tokenizer spellings with one set of constants.
fn normalize_dsml_delims(s: &str) -> String {
    // Cheap global replace is fine: DSML tags are the only place this char
    // is structural, and accidental fullwidth pipes in prose are rare.
    s.replace('｜', "|")
}

const DSML_OPEN_CALLS: &str = "<|DSML|tool_calls";
const DSML_CLOSE_CALLS: &str = "</|DSML|tool_calls>";
const DSML_OPEN_INVOKE: &str = "<|DSML|invoke";
const DSML_CLOSE_INVOKE: &str = "</|DSML|invoke>";
const DSML_OPEN_PARAM: &str = "<|DSML|parameter";
const DSML_CLOSE_PARAM: &str = "</|DSML|parameter>";

#[derive(Clone, Copy)]
enum ToolKind {
    Generic,
    Hy3,
    Dsml,
}

fn next_tool_region(s: &str) -> Option<(ToolKind, usize)> {
    let generic = s.find("<tool_call>").and_then(|i| {
        if s[i..].starts_with("<tool_call:opensource>")
            || s[i..].starts_with("<tool_calls:opensource>")
        {
            None
        } else {
            Some(i)
        }
    });
    // Also catch unclosed-looking open that is actually opensource plural
    let hy3 = s
        .find("<tool_calls:opensource>")
        .or_else(|| s.find("<tool_call:opensource>"));
    let dsml = find_dsml_open(s);
    [(ToolKind::Generic, generic), (ToolKind::Hy3, hy3), (ToolKind::Dsml, dsml)]
        .into_iter()
        .filter_map(|(k, o)| o.map(|i| (k, i)))
        .min_by_key(|(_, i)| *i)
}

fn find_dsml_open(s: &str) -> Option<usize> {
    let a = s.find(DSML_OPEN_CALLS);
    let b = s.find(DSML_OPEN_INVOKE);
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (Some(x), None) | (None, Some(x)) => Some(x),
        _ => None,
    }
}

fn skip_unit(s: &str) -> (&str, &str) {
    let mut it = s.chars();
    let ch = it.next();
    match ch {
        Some(c) => {
            let n = c.len_utf8();
            (&s[..n], &s[n..])
        }
        None => ("", s),
    }
}

/// How many bytes from `s` (which starts at a DSML open) form one DSML region
/// that should be dropped from the user-visible reply.
fn dsml_region_len(s: &str) -> Option<usize> {
    if s.starts_with(DSML_OPEN_CALLS) {
        // Prefer matching close; fall back to end of last invoke, else whole rest.
        if let Some(rel) = s.find(DSML_CLOSE_CALLS) {
            return Some(rel + DSML_CLOSE_CALLS.len());
        }
        // Unclosed container: consume through last </|DSML|invoke> if any
        if let Some(rel) = s.rfind(DSML_CLOSE_INVOKE) {
            return Some(rel + DSML_CLOSE_INVOKE.len());
        }
        return Some(s.len());
    }
    if s.starts_with(DSML_OPEN_INVOKE) {
        if let Some(rel) = s.find(DSML_CLOSE_INVOKE) {
            return Some(rel + DSML_CLOSE_INVOKE.len());
        }
        return Some(s.len());
    }
    None
}

fn strip_residual_tool_markup(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while !rest.is_empty() {
        let next = next_tool_region(rest);
        let Some((kind, start)) = next else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..start]);
        let after = &rest[start..];
        let n = match kind {
            ToolKind::Dsml => dsml_region_len(after).unwrap_or(1),
            ToolKind::Hy3 => {
                if after.starts_with("<tool_calls:opensource>") {
                    after
                        .find("</tool_calls:opensource>")
                        .map(|i| i + "</tool_calls:opensource>".len())
                        .unwrap_or(after.len())
                } else if after.starts_with("<tool_call:opensource>") {
                    after
                        .find("</tool_call:opensource>")
                        .map(|i| i + "</tool_call:opensource>".len())
                        .unwrap_or(after.len())
                } else {
                    1
                }
            }
            ToolKind::Generic => after
                .find("</tool_call>")
                .map(|i| i + "</tool_call>".len())
                .unwrap_or(after.len()),
        };
        rest = &after[n.min(after.len())..];
    }
    out
}

fn take_generic(s: &str) -> Option<(String, String, usize)> {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    if !s.starts_with(OPEN) {
        return None;
    }
    let body = &s[OPEN.len()..];
    let end = body.find(CLOSE)?;
    let block = body[..end].trim();
    let consumed = OPEN.len() + end + CLOSE.len();
    let v: serde_json::Value = serde_json::from_str(block).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let args = if v.get("arguments").map_or(true, |a| a.is_null()) {
        "{}".into()
    } else {
        v["arguments"].to_string()
    };
    Some((name, args, consumed))
}

fn take_hy3(s: &str) -> Option<(Vec<(String, String)>, usize)> {
    const OPEN: &str = "<tool_calls:opensource>";
    const CLOSE: &str = "</tool_calls:opensource>";
    if !s.starts_with(OPEN) {
        return None;
    }
    let body = &s[OPEN.len()..];
    let end = body.find(CLOSE)?;
    let inner = &body[..end];
    let mut calls = Vec::new();
    let mut rest = inner;
    while let Some(rel) = rest.find("<tool_call:opensource>") {
        rest = &rest[rel..];
        match take_hy3_one(rest) {
            Some((name, args, n)) => {
                calls.push((name, args));
                rest = &rest[n..];
            }
            None => break,
        }
    }
    let consumed = OPEN.len() + end + CLOSE.len();
    if calls.is_empty() {
        None
    } else {
        Some((calls, consumed))
    }
}

fn take_hy3_one(s: &str) -> Option<(String, String, usize)> {
    const OPEN: &str = "<tool_call:opensource>";
    const SEP: &str = "<tool_sep:opensource>";
    const CLOSE: &str = "</tool_call:opensource>";
    const KEY_O: &str = "<arg_key:opensource>";
    const KEY_C: &str = "</arg_key:opensource>";
    const VAL_O: &str = "<arg_value:opensource>";
    const VAL_C: &str = "</arg_value:opensource>";

    if !s.starts_with(OPEN) {
        return None;
    }
    let after_open = &s[OPEN.len()..];
    let sep = after_open.find(SEP)?;
    let name = after_open[..sep].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let after_sep = &after_open[sep + SEP.len()..];
    let close = after_sep.find(CLOSE)?;
    let args_body = &after_sep[..close];
    let mut map = serde_json::Map::new();
    let mut cursor = args_body;
    while let Some(k0) = cursor.find(KEY_O) {
        let after_k0 = &cursor[k0 + KEY_O.len()..];
        let Some(k1) = after_k0.find(KEY_C) else {
            break;
        };
        let key = after_k0[..k1].trim().to_string();
        let after_k1 = &after_k0[k1 + KEY_C.len()..];
        let Some(v0) = after_k1.find(VAL_O) else {
            break;
        };
        let after_v0 = &after_k1[v0 + VAL_O.len()..];
        let Some(v1) = after_v0.find(VAL_C) else {
            break;
        };
        let val = after_v0[..v1].to_string();
        map.insert(key, json_value_from_str(&val));
        cursor = &after_v0[v1 + VAL_C.len()..];
    }
    let args = serde_json::Value::Object(map).to_string();
    let consumed = OPEN.len() + sep + SEP.len() + close + CLOSE.len();
    Some((name, args, consumed))
}

fn take_dsml(s: &str) -> Option<(Vec<(String, String)>, usize)> {
    // Container: <|DSML|tool_calls…> … </|DSML|tool_calls>
    // Optional suffix after tool_calls (e.g. |_simple) before '>'.
    if s.starts_with(DSML_OPEN_CALLS) {
        let gt = s.find('>')?;
        let after = &s[gt + 1..];
        // Close: </|DSML|tool_calls> or </|DSML|tool_calls…>
        let (inner, consumed_total) = if let Some(ci) = after.find("</|DSML|tool_calls") {
            let close_end = after[ci..].find('>').map(|j| ci + j + 1)?;
            (after[..ci].to_string(), gt + 1 + close_end)
        } else {
            // Unclosed: parse invokes then consume region
            let region = dsml_region_len(s)?;
            let inner = s[gt + 1..region.min(s.len())].to_string();
            (inner, region)
        };
        let mut calls = Vec::new();
        let mut rest = inner.as_str();
        while let Some(rel) = rest.find(DSML_OPEN_INVOKE) {
            rest = &rest[rel..];
            match take_dsml_invoke(rest) {
                Some((name, args, n)) => {
                    calls.push((name, args));
                    rest = &rest[n..];
                }
                None => break,
            }
        }
        return if calls.is_empty() {
            None
        } else {
            Some((calls, consumed_total))
        };
    }

    if s.starts_with(DSML_OPEN_INVOKE) {
        let mut rest = s;
        let mut batch = Vec::new();
        let mut total = 0;
        while rest.starts_with(DSML_OPEN_INVOKE) {
            let (name, args, n) = take_dsml_invoke(rest)?;
            batch.push((name, args));
            total += n;
            rest = &rest[n..];
            let trimmed = rest.trim_start();
            total += rest.len() - trimmed.len();
            rest = trimmed;
        }
        return if batch.is_empty() {
            None
        } else {
            Some((batch, total))
        };
    }

    None
}

fn take_dsml_invoke(s: &str) -> Option<(String, String, usize)> {
    if !s.starts_with(DSML_OPEN_INVOKE) {
        return None;
    }
    let gt = s.find('>')?;
    let header = &s[..gt + 1];
    let name = attr_value(header, "name")?;
    let after = &s[gt + 1..];
    let end = after.find(DSML_CLOSE_INVOKE)?;
    let body = &after[..end];
    let mut map = serde_json::Map::new();
    let mut cursor = body;
    while let Some(rel) = cursor.find(DSML_OPEN_PARAM) {
        let from = &cursor[rel..];
        let pgt = match from.find('>') {
            Some(i) => i,
            None => break,
        };
        let pheader = &from[..=pgt];
        let Some(pkey) = attr_value(pheader, "name") else {
            cursor = &from[DSML_OPEN_PARAM.len()..];
            continue;
        };
        let after_h = &from[pgt + 1..];
        let Some(pend) = after_h.find(DSML_CLOSE_PARAM) else {
            break;
        };
        let pval = after_h[..pend].to_string();
        map.insert(pkey, json_value_from_str(&pval));
        cursor = &after_h[pend + DSML_CLOSE_PARAM.len()..];
    }
    let args = serde_json::Value::Object(map).to_string();
    let consumed = gt + 1 + end + DSML_CLOSE_INVOKE.len();
    Some((name, args, consumed))
}

fn attr_value(tag: &str, key: &str) -> Option<String> {
    let pat1 = format!("{key}=\"");
    let pat2 = format!("{key}='");
    if let Some(i) = tag.find(&pat1) {
        let rest = &tag[i + pat1.len()..];
        let end = rest.find('"')?;
        return Some(rest[..end].to_string());
    }
    if let Some(i) = tag.find(&pat2) {
        let rest = &tag[i + pat2.len()..];
        let end = rest.find('\'')?;
        return Some(rest[..end].to_string());
    }
    None
}

fn json_value_from_str(s: &str) -> serde_json::Value {
    let t = s.trim();
    if t.is_empty() {
        return serde_json::Value::String(String::new());
    }
    if t.eq_ignore_ascii_case("true") {
        return serde_json::Value::Bool(true);
    }
    if t.eq_ignore_ascii_case("false") {
        return serde_json::Value::Bool(false);
    }
    if let Ok(n) = t.parse::<i64>() {
        return serde_json::json!(n);
    }
    if let Ok(n) = t.parse::<f64>() {
        if t.contains('.') || t.contains('e') || t.contains('E') {
            return serde_json::json!(n);
        }
    }
    if (t.starts_with('{') && t.ends_with('}')) || (t.starts_with('[') && t.ends_with(']')) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(t) {
            return v;
        }
    }
    serde_json::Value::String(s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_json() {
        let t = r#"hello
<tool_call>
{"name": "SearchTool__search_searxng", "arguments": {"query": "rust"}}
</tool_call>
done"#;
        let (clean, calls) = extract_tool_calls(t);
        assert_eq!(clean, "hello\n\ndone");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "SearchTool__search_searxng");
    }

    #[test]
    fn hy3_opensource() {
        let t = r#"<tool_calls:opensource>
<tool_call:opensource>searxng__search_searxng<tool_sep:opensource>
<arg_key:opensource>query</arg_key:opensource>
<arg_value:opensource>Max Verstappen 2026 F1</arg_value:opensource>
<arg_key:opensource>limit</arg_key:opensource>
<arg_value:opensource>10</arg_value:opensource>
</tool_call:opensource>
</tool_calls:opensource>"#;
        let (clean, calls) = extract_tool_calls(t);
        assert!(clean.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "searxng__search_searxng");
        let v: serde_json::Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(v["query"], "Max Verstappen 2026 F1");
        assert_eq!(v["limit"], 10);
    }

    #[test]
    fn deepseek_dsml_simple_suffix() {
        let t = r#"<｜DSML｜tool_calls｜_simple>
<｜DSML｜invoke name="web_search">
<｜DSML｜parameter name="query" string="true">Vasco da Gama championship position 2024</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"#;
        let (clean, calls) = extract_tool_calls(t);
        assert!(clean.is_empty(), "clean={clean:?}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "web_search");
    }

    #[test]
    fn deepseek_dsml_plain_tool_calls() {
        // Exact form reported in the web UI for DeepSeek-V4-Flash
        let t = r#"<｜DSML｜tool_calls>
<｜DSML｜invoke name="search">
<｜DSML｜parameter name="query" string="true">Max Verstappen 2026 Formula 1 standings</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"#;
        let (clean, calls) = extract_tool_calls(t);
        assert!(
            clean.is_empty(),
            "DSML must not leak into content, clean={clean:?}"
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "search");
        let v: serde_json::Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(v["query"], "Max Verstappen 2026 Formula 1 standings");
    }

    #[test]
    fn deepseek_dsml_ascii_pipes() {
        let t = r#"<|DSML|tool_calls>
<|DSML|invoke name="search">
<|DSML|parameter name="query" string="true">hello</|DSML|parameter>
</|DSML|invoke>
</|DSML|tool_calls>"#;
        let (clean, calls) = extract_tool_calls(t);
        assert!(clean.is_empty());
        assert_eq!(calls[0].0, "search");
    }

    #[test]
    fn mixed_text_and_hy3() {
        let t = "Before.\n<tool_call:opensource>foo__bar<tool_sep:opensource>\n<arg_key:opensource>q</arg_key:opensource>\n<arg_value:opensource>x</arg_value:opensource>\n</tool_call:opensource>\nAfter.";
        let (clean, calls) = extract_tool_calls(t);
        assert!(clean.contains("Before"));
        assert!(clean.contains("After"));
        assert_eq!(calls[0].0, "foo__bar");
    }

    #[test]
    fn find_tool_open_dsml_utf8() {
        let s = "hi <｜DSML｜tool_calls>".as_bytes();
        assert!(find_tool_open(s).is_some());
    }
}
