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
        for piece in pretokenize(text.as_bytes()) {
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
}
