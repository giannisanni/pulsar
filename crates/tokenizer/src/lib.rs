//! GPT-2 style byte-level BPE tokenizer, built entirely from GGUF metadata
//! (`tokenizer.ggml.tokens` / `.merges` / special-token ids).
//!
//! The pre-tokenizer split matches ds4's `bpe_tokenize_text` (the path Hy3
//! takes there), because ds4 is pulsar's decode-parity reference: different
//! splits produce different merges, and therefore different token streams,
//! even when the text bytes are identical.

use std::collections::HashMap;

use gguf::{Gguf, Value};

#[derive(Debug)]
pub enum Error {
    MissingKey(&'static str),
    BadKey(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::MissingKey(k) => write!(f, "gguf metadata is missing {k}"),
            Error::BadKey(k) => write!(f, "gguf metadata key {k} has the wrong shape"),
        }
    }
}

impl std::error::Error for Error {}

pub struct Tokenizer {
    tokens: Vec<String>,
    token_to_id: HashMap<String, u32>,
    /// Keyed as "left right", value = merge priority (lower merges first).
    merge_rank: HashMap<String, u32>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub eot_id: Option<u32>,
    pre: Pre,
}

/// Pre-tokenizer split family, from `tokenizer.ggml.pre`. The split shape
/// determines the merges, so it must match the reference engine exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pre {
    /// ds4's JoyAI-style split (DeepSeek "joyai-llm", Hy3 "hunyuan-dense").
    JoyAi,
    /// ChatGLM4/GLM llama3-style split ("glm4").
    Glm4,
}

/// The special-token ids a chat loop needs, resolved from the vocab.
/// Hy3 layout (mirrors ds4's encode_chat_prompt): one turn is
/// `[bos] [system-text] user <text> assistant think_start think_end`,
/// and a finished assistant reply is followed by eos in the context.
#[derive(Debug, Clone, Copy)]
pub struct ChatMarkers {
    pub bos: u32,
    pub eos: u32,
    pub eot: Option<u32>,
    pub user: u32,
    pub assistant: u32,
    pub think_start: u32,
    pub think_end: u32,
}

impl ChatMarkers {
    pub fn resolve(t: &Tokenizer) -> Result<ChatMarkers, Error> {
        let find = |s: &'static str| t.find_token(s).ok_or(Error::MissingKey(s));
        Ok(ChatMarkers {
            bos: t.bos_id.ok_or(Error::MissingKey("bos_token_id"))?,
            eos: t.eos_id.ok_or(Error::MissingKey("eos_token_id"))?,
            eot: t.eot_id,
            user: find("<｜hy_User:opensource｜>")?,
            assistant: find("<｜hy_Assistant:opensource｜>")?,
            think_start: find("<think:opensource>")?,
            think_end: find("</think:opensource>")?,
        })
    }

    pub fn is_stop(&self, id: u32) -> bool {
        id == self.eos || Some(id) == self.eot
    }
}

/// GPT-2's byte<->unicode bijection: printable bytes map to themselves,
/// the rest to codepoints 256+n, so merges operate on valid UTF-8 without
/// losing byte identity.
fn gpt2_byte_to_char(b: u8) -> char {
    let printable = |x: u8| (33..=126).contains(&x) || (161..=172).contains(&x) || x >= 174;
    if printable(b) {
        return b as char;
    }
    let n = (0..b).filter(|&x| !printable(x)).count() as u32;
    char::from_u32(256 + n).unwrap()
}

fn string_array(g: &Gguf, key: &'static str) -> Result<Vec<String>, Error> {
    let Some(Value::Array(a)) = g.metadata.get(key) else {
        return Err(Error::MissingKey(key));
    };
    a.iter()
        .map(|v| v.as_str().map(str::to_owned).ok_or(Error::BadKey(key)))
        .collect()
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Self, Error> {
        let tokens = string_array(g, "tokenizer.ggml.tokens")?;
        let merges = string_array(g, "tokenizer.ggml.merges")?;

        let token_to_id = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        let merge_rank = merges
            .into_iter()
            .enumerate()
            .map(|(i, m)| (m, i as u32))
            .collect();

        let mut byte_to_char = ['\0'; 256];
        let mut char_to_byte = HashMap::with_capacity(256);
        for b in 0..=255u8 {
            let c = gpt2_byte_to_char(b);
            byte_to_char[b as usize] = c;
            char_to_byte.insert(c, b);
        }

        let id_key = |k| g.metadata.get(k).and_then(Value::as_u64).map(|v| v as u32);
        Ok(Tokenizer {
            tokens,
            token_to_id,
            merge_rank,
            byte_to_char,
            char_to_byte,
            bos_id: id_key("tokenizer.ggml.bos_token_id"),
            eos_id: id_key("tokenizer.ggml.eos_token_id"),
            eot_id: id_key("tokenizer.ggml.eot_token_id"),
            pre: match g.metadata.get("tokenizer.ggml.pre").and_then(Value::as_str) {
                Some("glm4") => Pre::Glm4,
                _ => Pre::JoyAi,
            },
        })
    }

    pub fn n_vocab(&self) -> usize {
        self.tokens.len()
    }

    /// The raw vocab string for an id (byte-encoded space for normal
    /// tokens, literal for control tokens).
    pub fn token_str(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    /// Exact vocab lookup, e.g. for chat marker tokens.
    pub fn find_token(&self, s: &str) -> Option<u32> {
        self.token_to_id.get(s).copied()
    }

    /// Encode plain text (no special-token recognition; chat markers are
    /// pushed by id, exactly as ds4 does).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let pieces = match self.pre {
            Pre::JoyAi => pretokenize(text.as_bytes()),
            Pre::Glm4 => pretokenize_glm4(text.as_bytes()),
        };
        for piece in pieces {
            self.bpe_piece(piece, &mut out);
        }
        out
    }

    /// Decode ids to bytes. Chars outside the byte map (control-token text)
    /// pass through as their UTF-8 bytes.
    pub fn decode(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            let Some(tok) = self.tokens.get(id as usize) else { continue };
            for c in tok.chars() {
                match self.char_to_byte.get(&c) {
                    Some(&b) => out.push(b),
                    None => {
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
        out
    }

    /// Byte-level BPE on one pre-tokenized piece.
    /// ponytail: O(n^2) merge scan with per-pair key allocation, exactly
    /// ds4's shape; pieces are words. Rank-heap it if prefill tokenization
    /// ever shows up in a profile.
    fn bpe_piece(&self, piece: &[u8], out: &mut Vec<u32>) {
        let encoded: String = piece.iter().map(|&b| self.byte_to_char[b as usize]).collect();
        let mut sym: Vec<String> = encoded.chars().map(String::from).collect();

        loop {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..sym.len().saturating_sub(1) {
                let key = format!("{} {}", sym[i], sym[i + 1]);
                if let Some(&rank) = self.merge_rank.get(&key) {
                    if best.map_or(true, |(_, r)| rank < r) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            let right = sym.remove(i + 1);
            sym[i].push_str(&right);
        }

        for s in &sym {
            if let Some(&id) = self.token_to_id.get(s) {
                out.push(id);
            } else {
                // unmergeable symbol: fall back to single byte-chars
                for c in s.chars() {
                    if let Some(&id) = self.token_to_id.get(c.to_string().as_str()) {
                        out.push(id);
                    }
                }
            }
        }
    }
}

/* ---- pre-tokenizer: port of ds4's JoyAI-style split -------------------- */

fn ascii_alpha(c: u8) -> bool {
    c.is_ascii_alphabetic()
}

fn ascii_digit(c: u8) -> bool {
    c.is_ascii_digit()
}

fn ascii_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

fn ascii_newline(c: u8) -> bool {
    c == b'\n' || c == b'\r'
}

fn punct_symbol(c: u8) -> bool {
    matches!(c, b'!'..=b'/' | b':'..=b'@' | b'['..=b'`' | b'{'..=b'~')
}

fn utf8_char_len(c: u8) -> usize {
    if c < 0x80 {
        1
    } else if c & 0xe0 == 0xc0 {
        2
    } else if c & 0xf0 == 0xe0 {
        3
    } else if c & 0xf8 == 0xf0 {
        4
    } else {
        1
    }
}

fn next_char(s: &[u8], pos: usize) -> usize {
    let n = utf8_char_len(s[pos]);
    if pos + n > s.len() {
        pos + 1
    } else {
        pos + n
    }
}

fn peek_codepoint(s: &[u8], pos: usize) -> u32 {
    let n = utf8_char_len(s[pos]);
    if pos + n > s.len() || n == 1 {
        return s[pos] as u32;
    }
    let cont = |i: usize| (s[pos + i] & 0x3f) as u32;
    match n {
        2 => ((s[pos] & 0x1f) as u32) << 6 | cont(1),
        3 => ((s[pos] & 0x0f) as u32) << 12 | cont(1) << 6 | cont(2),
        _ => ((s[pos] & 0x07) as u32) << 18 | cont(1) << 12 | cont(2) << 6 | cont(3),
    }
}

fn cjk_at(s: &[u8], pos: usize) -> bool {
    if s[pos] < 128 {
        return false;
    }
    let cp = peek_codepoint(s, pos);
    (0x4e00..=0x9fa5).contains(&cp) || (0x3040..=0x309f).contains(&cp) || (0x30a0..=0x30ff).contains(&cp)
}

/// ASCII letters, plus any non-ASCII char (CJK is carved out first by the
/// caller) - matching ds4's collapsed letter class.
fn letter_like_at(s: &[u8], pos: usize) -> bool {
    let c = s[pos];
    if c < 128 {
        ascii_alpha(c)
    } else {
        true
    }
}

fn consume_letters(s: &[u8], mut pos: usize) -> usize {
    while pos < s.len() && letter_like_at(s, pos) {
        pos = next_char(s, pos);
    }
    pos
}

/// Split text into BPE words. The split shape matters: it must match the
/// reference engine byte for byte.
fn pretokenize(s: &[u8]) -> Vec<&[u8]> {
    let len = s.len();
    let mut out = Vec::new();
    let mut pos = 0usize;

    while pos < len {
        let start = pos;
        let c = s[pos];

        if ascii_digit(c) {
            let mut n = 0;
            while pos < len && ascii_digit(s[pos]) && n < 3 {
                pos += 1;
                n += 1;
            }
        } else if cjk_at(s, pos) {
            loop {
                pos = next_char(s, pos);
                if pos >= len || !cjk_at(s, pos) {
                    break;
                }
            }
        } else if punct_symbol(c) && pos + 1 < len && ascii_alpha(s[pos + 1]) {
            pos += 1;
            while pos < len && ascii_alpha(s[pos]) {
                pos += 1;
            }
        } else if letter_like_at(s, pos) && !cjk_at(s, pos) {
            pos = consume_letters(s, pos);
        } else if !ascii_newline(c)
            && !punct_symbol(c)
            && pos + 1 < len
            && letter_like_at(s, pos + 1)
        {
            pos += 1;
            pos = consume_letters(s, pos);
        } else if c == b' ' && pos + 1 < len && punct_symbol(s[pos + 1]) {
            pos += 1;
            while pos < len && punct_symbol(s[pos]) {
                pos += 1;
            }
            while pos < len && ascii_newline(s[pos]) {
                pos += 1;
            }
        } else if punct_symbol(c) {
            while pos < len && punct_symbol(s[pos]) {
                pos += 1;
            }
            while pos < len && ascii_newline(s[pos]) {
                pos += 1;
            }
        } else if ascii_space(c) {
            let mut p = pos;
            let mut last_newline_end = 0usize;
            while p < len && ascii_space(s[p]) {
                let sc = s[p];
                p += 1;
                if ascii_newline(sc) {
                    last_newline_end = p;
                }
            }
            if last_newline_end != 0 {
                pos = last_newline_end;
            } else if p < len && p > pos + 1 && (letter_like_at(s, p) || punct_symbol(s[p])) {
                // a single leading space joins the following word:
                // "    int" splits as "   " + " int", not "    " + "int"
                pos = p - 1;
            } else {
                pos = p;
            }
        } else {
            pos = next_char(s, pos);
        }

        if pos == start {
            pos = next_char(s, pos);
        }
        out.push(&s[start..pos.min(len)]);
    }
    out
}

/* ---- glm4 pre-tokenizer: port of ds4's bpe_tokenize_text_glm4 ---------- */

#[derive(Clone, Copy)]
struct Glm4Char {
    cp: u32,
    next: usize,
    valid: bool,
    is_letter: bool,
    is_number: bool,
    is_whitespace: bool,
}

fn glm4_whitespace(cp: u32) -> bool {
    if cp < 128 {
        return ascii_space(cp as u8);
    }
    cp == 0x0085
        || cp == 0x00a0
        || cp == 0x1680
        || (0x2000..=0x200a).contains(&cp)
        || cp == 0x2028
        || cp == 0x2029
        || cp == 0x202f
        || cp == 0x205f
        || cp == 0x3000
}

fn glm4_number(cp: u32) -> bool {
    if cp < 128 {
        return cp.try_into().map(ascii_digit).unwrap_or(false);
    }
    const RANGES: &[(u32, u32)] = &[
        (0x0660, 0x0669), (0x06f0, 0x06f9), (0x07c0, 0x07c9), (0x0966, 0x096f),
        (0x09e6, 0x09ef), (0x0a66, 0x0a6f), (0x0ae6, 0x0aef), (0x0b66, 0x0b6f),
        (0x0be6, 0x0bef), (0x0c66, 0x0c6f), (0x0ce6, 0x0cef), (0x0d66, 0x0d6f),
        (0x0de6, 0x0def), (0x0e50, 0x0e59), (0x0ed0, 0x0ed9), (0x0f20, 0x0f29),
        (0x1040, 0x1049), (0x1090, 0x1099), (0x17e0, 0x17e9), (0x1810, 0x1819),
        (0xff10, 0xff19),
    ];
    RANGES.iter().any(|&(lo, hi)| (lo..=hi).contains(&cp))
}

fn glm4_punct_symbol(cp: u32) -> bool {
    if cp < 128 {
        return cp.try_into().map(punct_symbol).unwrap_or(false);
    }
    const RANGES: &[(u32, u32)] = &[
        (0x00a1, 0x00a9), (0x00ab, 0x00ac), (0x00ae, 0x00b1), (0x00b4, 0x00b4),
        (0x00b6, 0x00b8), (0x00bb, 0x00bb), (0x00bf, 0x00bf), (0x00d7, 0x00d7),
        (0x00f7, 0x00f7), (0x02c2, 0x02df), (0x02e5, 0x02eb), (0x02ed, 0x02ff),
        (0x0375, 0x037e), (0x0384, 0x0385), (0x0387, 0x0387), (0x055a, 0x055f),
        (0x0589, 0x058a), (0x05be, 0x05c0), (0x05c3, 0x05c3), (0x05c6, 0x05c7),
        (0x0609, 0x060a), (0x060c, 0x060d), (0x061b, 0x061b), (0x061e, 0x061f),
        (0x066a, 0x066a), (0x066d, 0x066d), (0x06d4, 0x06d4), (0x2000, 0x206f),
        (0x20a0, 0x20cf), (0x2100, 0x214f), (0x2190, 0x23ff), (0x2460, 0x24ff),
        (0x2500, 0x2775), (0x2794, 0x2bff), (0x2e00, 0x2e7f), (0x3000, 0x303f),
        (0xfd3e, 0xfd3f), (0xfe10, 0xfe6f), (0xff01, 0xff0f), (0xff1a, 0xff20),
        (0xff3b, 0xff40), (0xff5b, 0xff65), (0x1f000, 0x1faff),
    ];
    RANGES.iter().any(|&(lo, hi)| (lo..=hi).contains(&cp))
}

fn glm4_char_at(s: &[u8], pos: usize) -> Glm4Char {
    if pos >= s.len() {
        return Glm4Char { cp: 0, next: pos, valid: false, is_letter: false, is_number: false, is_whitespace: false };
    }
    let cp = peek_codepoint(s, pos);
    let next = next_char(s, pos);
    let is_whitespace = glm4_whitespace(cp);
    let is_number = glm4_number(cp);
    let is_letter = if cp < 128 {
        (cp as u8).is_ascii_alphabetic()
    } else {
        !is_whitespace && !is_number && !glm4_punct_symbol(cp)
    };
    Glm4Char { cp, next, valid: true, is_letter, is_number, is_whitespace }
}

fn ascii_lower(cp: u32) -> u32 {
    if (b'A' as u32..=b'Z' as u32).contains(&cp) {
        cp + 32
    } else {
        cp
    }
}

fn pretokenize_glm4(s: &[u8]) -> Vec<&[u8]> {
    let len = s.len();
    let mut out = Vec::new();
    let mut pos = 0usize;

    while pos < len {
        let start = pos;
        let cur = glm4_char_at(s, pos);
        if !cur.valid {
            break;
        }

        // english contractions: 's 't 'm 'd 're 've 'll
        if cur.cp == '\'' as u32 && cur.next < len {
            let next = glm4_char_at(s, cur.next);
            let n1 = ascii_lower(next.cp);
            if matches!(n1, 0x73 | 0x74 | 0x6d | 0x64) {
                pos = next.next;
                out.push(&s[start..pos]);
                continue;
            }
            if next.valid && next.next < len {
                let next2 = glm4_char_at(s, next.next);
                let n2 = ascii_lower(next2.cp);
                if (n1 == 0x72 && n2 == 0x65) || (n1 == 0x76 && n2 == 0x65) || (n1 == 0x6c && n2 == 0x6c) {
                    pos = next2.next;
                    out.push(&s[start..pos]);
                    continue;
                }
            }
        }

        // letter run (optionally led by one non-letter, non-newline char)
        if !(cur.cp == 0x0d || cur.cp == 0x0a || cur.is_number) {
            let next = glm4_char_at(s, cur.next);
            if cur.is_letter || next.is_letter {
                pos = cur.next;
                while pos < len {
                    let scan = glm4_char_at(s, pos);
                    if !scan.valid || !scan.is_letter {
                        break;
                    }
                    pos = scan.next;
                }
                out.push(&s[start..pos]);
                continue;
            }
        }

        // digits, max 3
        if cur.is_number {
            let mut ndigits = 0;
            while pos < len && ndigits < 3 {
                let scan = glm4_char_at(s, pos);
                if !scan.valid || !scan.is_number {
                    break;
                }
                pos = scan.next;
                ndigits += 1;
            }
            out.push(&s[start..pos]);
            continue;
        }

        // punct/symbol run (optionally led by one space), trailing newlines
        let (mut punct, punct_pos) = if cur.cp == ' ' as u32 {
            (glm4_char_at(s, cur.next), cur.next)
        } else {
            (cur, pos)
        };
        punct.valid = punct.valid && punct_pos < len;
        if punct.valid && !punct.is_whitespace && !punct.is_letter && !punct.is_number {
            pos = punct_pos;
            while pos < len {
                let scan = glm4_char_at(s, pos);
                if !scan.valid || scan.is_whitespace || scan.is_letter || scan.is_number {
                    break;
                }
                pos = scan.next;
            }
            while pos < len {
                let scan = glm4_char_at(s, pos);
                if !scan.valid || !(scan.cp == 0x0d || scan.cp == 0x0a) {
                    break;
                }
                pos = scan.next;
            }
            out.push(&s[start..pos]);
            continue;
        }

        // whitespace runs: keep through the last newline, or leave the
        // final ws char to join the next word
        if cur.is_whitespace {
            let mut p = pos;
            let mut last_newline_end = 0usize;
            let mut last_ws_start = pos;
            let mut nspace = 0;
            while p < len {
                let scan = glm4_char_at(s, p);
                if !scan.valid || !scan.is_whitespace {
                    break;
                }
                last_ws_start = p;
                if scan.cp == 0x0d || scan.cp == 0x0a {
                    last_newline_end = scan.next;
                }
                p = scan.next;
                nspace += 1;
            }
            if last_newline_end != 0 {
                pos = last_newline_end;
            } else if nspace > 1 && p < len {
                pos = last_ws_start;
            } else {
                pos = p;
            }
            out.push(&s[start..pos]);
            continue;
        }

        pos = cur.next;
        if pos == start {
            pos = next_char(s, pos);
        }
        out.push(&s[start..pos.min(len)]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_char_map_is_a_bijection() {
        let mut seen = std::collections::HashSet::new();
        for b in 0..=255u8 {
            assert!(seen.insert(gpt2_byte_to_char(b)));
        }
        // spot checks against the canonical GPT-2 table
        assert_eq!(gpt2_byte_to_char(b' '), '\u{120}'); // Ġ
        assert_eq!(gpt2_byte_to_char(b'\n'), '\u{10a}'); // Ċ
        assert_eq!(gpt2_byte_to_char(b'!'), '!');
    }

    #[test]
    fn pretokenize_splits_leading_space_runs() {
        let pieces: Vec<&[u8]> = pretokenize(b"    int x");
        assert_eq!(pieces, vec![&b"   "[..], &b" int"[..], &b" x"[..]]);
    }

    #[test]
    fn pretokenize_groups_digits_by_three() {
        let pieces: Vec<&[u8]> = pretokenize(b"12345");
        assert_eq!(pieces, vec![&b"123"[..], &b"45"[..]]);
    }

    #[test]
    fn pretokenize_keeps_newlines_with_punct() {
        let pieces: Vec<&[u8]> = pretokenize(b"x;\ny");
        assert_eq!(pieces, vec![&b"x"[..], &b";\n"[..], &b"y"[..]]);
    }

    #[test]
    fn glm4_splits_contractions() {
        let pieces: Vec<&[u8]> = pretokenize_glm4(b"I'll don't");
        assert_eq!(
            pieces,
            vec![&b"I"[..], &b"'ll"[..], &b" don"[..], &b"'t"[..]]
        );
    }

    #[test]
    fn glm4_groups_digits_by_three() {
        let pieces: Vec<&[u8]> = pretokenize_glm4(b"12345");
        assert_eq!(pieces, vec![&b"123"[..], &b"45"[..]]);
    }

    #[test]
    fn glm4_leading_space_joins_word_and_punct_keeps_newline() {
        let pieces: Vec<&[u8]> = pretokenize_glm4(b"a b;\nc");
        assert_eq!(
            pieces,
            vec![&b"a"[..], &b" b"[..], &b";\n"[..], &b"c"[..]]
        );
    }

    #[test]
    fn glm4_whitespace_run_leaves_last_for_next_word() {
        let pieces: Vec<&[u8]> = pretokenize_glm4(b"a    b");
        assert_eq!(pieces, vec![&b"a"[..], &b"   "[..], &b" b"[..]]);
    }
}
