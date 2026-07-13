//! Token-stream parity against ds4 on the production Hy3 vocab.
//! Gold vectors come from `ds4 --dump-tokens` with Hy3-ds4-IQ2XXS-AttnQ8.gguf.

use gguf::Gguf;
use tokenizer::Tokenizer;

fn hy3() -> Tokenizer {
    let head = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../gguf/tests/fixtures/hy3-header.bin"
    ))
    .expect("fixture missing: crates/gguf/tests/fixtures/hy3-header.bin");
    let g = Gguf::parse(&head).expect("parse");
    Tokenizer::from_gguf(&g).expect("tokenizer")
}

#[test]
fn matches_ds4_token_streams() {
    let t = hy3();
    assert_eq!(t.n_vocab(), 120832);
    assert_eq!(t.bos_id, Some(120000));
    assert_eq!(t.eos_id, Some(120025));

    let gold: &[(&str, &[u32])] = &[
        ("Hello, world!", &[16883, 11, 2385, 0]),
        ("int x;\nreturn y;\n\ndone", &[632, 1025, 401, 3589, 376, 983, 37470]),
        (
            "De kat zat op 12345 matten.",
            &[3864, 43204, 1575, 253, 4993, 206, 7827, 2445, 1672, 1645, 13],
        ),
        (
            "print(\"héllo\") 你好世界",
            &[3105, 1164, 71, 29609, 906, 5029, 206, 17687, 3042],
        ),
    ];
    for (text, ids) in gold {
        assert_eq!(t.encode(text), *ids, "encode mismatch for {text:?}");
        assert_eq!(t.decode(ids), text.as_bytes(), "decode mismatch for {text:?}");
    }
}

#[test]
fn chat_marker_tokens_resolve() {
    let t = hy3();
    assert!(t.find_token("<｜hy_User:opensource｜>").is_some());
    assert!(t.find_token("<｜hy_Assistant:opensource｜>").is_some());
    assert!(t.find_token("<think:opensource>").is_some());
    assert!(t.find_token("</think:opensource>").is_some());
}
