//! Integration tests for multi-format tool-call parsing (no CUDA / linux).
// Only the parse helpers are exercised here; the formatters and holdback
// helper are used by the server binary, so they read as dead in this target.
#[allow(dead_code)]
#[path = "../src/tool_calls.rs"]
mod tool_calls;

// Re-export unit tests by invoking the public API with the same fixtures.
use tool_calls::extract_tool_calls;

#[test]
fn generic_json() {
    let t = r#"hello
<tool_call>
{"name": "SearchTool__search_searxng", "arguments": {"query": "rust"}}
</tool_call>
done"#;
    let (clean, calls) = extract_tool_calls(t);
    assert!(clean.contains("hello"));
    assert!(clean.contains("done"));
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "SearchTool__search_searxng");
    assert!(calls[0].1.contains("rust"));
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
fn deepseek_dsml() {
    let t = r#"<｜DSML｜tool_calls>
<｜DSML｜invoke name="search">
<｜DSML｜parameter name="query" string="true">Max Verstappen 2026 Formula 1 standings</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"#;
    let (clean, calls) = extract_tool_calls(t);
    assert!(clean.is_empty(), "DSML must not leak, clean={clean:?}");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "search");
    let v: serde_json::Value = serde_json::from_str(&calls[0].1).unwrap();
    assert_eq!(v["query"], "Max Verstappen 2026 Formula 1 standings");
}

#[test]
fn find_tool_open_markers() {
    assert!(tool_calls::find_tool_open(b"xx<tool_call>yy").is_some());
    assert!(tool_calls::find_tool_open(b"<tool_calls:opensource>").is_some());
    let dsml = "x<｜DSML｜tool_calls>".as_bytes();
    assert!(tool_calls::find_tool_open(dsml).is_some());
}

#[test]
fn poolside_laguna_xml() {
    let t = r#"I'll look that up.
<tool_call>SearchTool__search_searxng<arg_key>query</arg_key><arg_value>NATO members</arg_value><arg_key>limit</arg_key><arg_value>10</arg_value></tool_call>"#;
    let (clean, calls) = extract_tool_calls(t);
    assert_eq!(clean, "I'll look that up.");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "SearchTool__search_searxng");
    let v: serde_json::Value = serde_json::from_str(&calls[0].1).unwrap();
    assert_eq!(v["query"], "NATO members");
    assert_eq!(v["limit"], 10);
}

#[test]
fn poolside_format_roundtrip() {
    let calls = vec![(
        "terminal".into(),
        r#"{"cmd":"uname -a"}"#.into(),
    )];
    let s = tool_calls::format_poolside_tool_calls(&calls);
    assert!(s.contains("<tool_call>terminal"));
    assert!(s.contains("<arg_key>cmd</arg_key>"));
    let (clean, parsed) = extract_tool_calls(&s);
    assert!(clean.is_empty());
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].0, "terminal");
}
