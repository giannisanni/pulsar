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
    /// Whether a raw prompt gets a leading bos (llama.cpp semantics:
    /// tokenizer.ggml.add_bos_token if present, else SPM yes / BPE no).
    pub add_bos: bool,
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
    KimiK2,
    MiniMax,
    /// Qwen2/Qwen3 split ("qwen2"): minimax minus its extras; standalone
    /// contractions, single-digit numbers.
    Qwen2,
    /// Gemma 4 raw SPM-style BPE (tokenizer.ggml.model == "gemma4"):
    /// spaces become U+2581, only newline runs pre-split, merges run on
    /// raw UTF-8 (no gpt2 byte map), unknown bytes fall back to <0xXX>.
    Gemma4,
}

/// The special-token ids a chat loop needs, resolved from the vocab.
/// Hy3 layout (mirrors ds4's encode_chat_prompt): one turn is
/// `[bos] [system-text] user <text> assistant think_start think_end`,
/// and a finished assistant reply is followed by eos in the context.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ChatStyle {
    /// bos + user-marker text assistant-marker <think></think>
    Hy3,
    /// <|im_user|>user<|im_middle|>text<|im_end|> ... (Kimi K2 family)
    Kimi,
    /// <|im_start|>user\ntext<|im_end|>\n ... (Qwen ChatML; no bos)
    ChatMl,
    /// <start_of_turn>user\ntext<end_of_turn>\n ... (Gemma; roles are
    /// "user"/"model", bos prepended)
    Gemma,
    /// ]~b]user\ntext[e~[ ... (MiniMax M3; roles are "user"/"ai" plain
    /// text after the ]~b] marker; ]~!b[ opens the conversation)
    MiniMax,
}

pub struct ChatMarkers {
    style: ChatStyle,
    /// None on ChatML models (qwen: add_bos_token=false, no bos id).
    pub bos: Option<u32>,
    pub eos: u32,
    pub eot: Option<u32>,
    user: u32,
    assistant: u32,
    /// Hy3: think_start/think_end. Kimi: <|im_middle|> / <|im_system|>.
    aux0: u32,
    aux1: u32,
}

impl ChatMarkers {
    pub fn resolve(t: &Tokenizer) -> Result<ChatMarkers, Error> {
        let find = |s: &'static str| t.find_token(s).ok_or(Error::MissingKey(s));
        if t.find_token("<|im_middle|>").is_some() {
            return Ok(ChatMarkers {
                style: ChatStyle::Kimi,
                bos: Some(t.bos_id.ok_or(Error::MissingKey("bos_token_id"))?),
                eos: t.eos_id.ok_or(Error::MissingKey("eos_token_id"))?,
                eot: t.find_token("<|im_end|>"),
                user: find("<|im_user|>")?,
                assistant: find("<|im_assistant|>")?,
                aux0: find("<|im_middle|>")?,
                aux1: find("<|im_system|>")?,
            });
        }
        if t.find_token("<start_of_turn>").is_some() {
            return Ok(ChatMarkers {
                style: ChatStyle::Gemma,
                bos: t.bos_id,
                eos: t.eos_id.ok_or(Error::MissingKey("eos_token_id"))?,
                eot: t.find_token("<end_of_turn>"),
                user: find("<start_of_turn>")?,
                assistant: find("<start_of_turn>")?,
                aux0: find("<end_of_turn>")?,
                aux1: find("<end_of_turn>")?,
            });
        }
        if t.find_token("]~b]").is_some() {
            // MiniMax M3: ]~!b[ (bod) opens, ]~b]<role>\n starts a turn,
            // [e~[ + \n ends it; roles are plain text ("system"/"user"/
            // "ai"). aux1 = <mm:think>: the assistant opener forces the
            // thinking-enabled prefix - the adaptive branch (no prefix)
            // greedily emits eos on the 2-bit quants.
            return Ok(ChatMarkers {
                style: ChatStyle::MiniMax,
                // the CLI seeds the context with bos - for M3 that seat
                // belongs to the beginning-of-dialog marker
                bos: t.find_token("]~!b["),
                eos: t.eos_id.ok_or(Error::MissingKey("eos_token_id"))?,
                eot: t.find_token("[e~["),
                user: find("]~b]")?,
                assistant: find("]~b]")?,
                aux0: find("[e~[")?,
                aux1: find("<mm:think>")?,
            });
        }
        if t.find_token("<|im_start|>").is_some() {
            // Qwen ChatML: <|im_start|> opens every turn, <|im_end|> closes
            return Ok(ChatMarkers {
                style: ChatStyle::ChatMl,
                bos: t.bos_id,
                eos: t.eos_id.ok_or(Error::MissingKey("eos_token_id"))?,
                eot: t.find_token("<|im_end|>"),
                user: find("<|im_start|>")?,
                assistant: find("<|im_start|>")?,
                aux0: find("<|im_end|>")?,
                aux1: find("<|im_end|>")?,
            });
        }
        Ok(ChatMarkers {
            style: ChatStyle::Hy3,
            bos: Some(t.bos_id.ok_or(Error::MissingKey("bos_token_id"))?),
            eos: t.eos_id.ok_or(Error::MissingKey("eos_token_id"))?,
            eot: t.eot_id,
            user: find("<｜hy_User:opensource｜>")?,
            assistant: find("<｜hy_Assistant:opensource｜>")?,
            aux0: find("<think:opensource>")?,
            aux1: find("</think:opensource>")?,
        })
    }

    /// System text ids for the first turn (Hy3: bare text after bos).
    pub fn render_system(&self, t: &Tokenizer, text: &str) -> Vec<u32> {
        match self.style {
            ChatStyle::Hy3 => t.encode(text),
            ChatStyle::ChatMl => {
                let mut v = vec![self.user];
                v.extend(t.encode(&format!("system\n{text}")));
                v.push(self.aux0);
                v.extend(t.encode("\n"));
                v
            }
            // gemma has no system role: it rides as a user-role turn
            ChatStyle::Gemma => {
                let mut v = vec![self.user];
                v.extend(t.encode(&format!("user\n{text}")));
                v.push(self.aux0);
                v.extend(t.encode("\n"));
                v
            }
            ChatStyle::Kimi => {
                let mut v = vec![self.aux1];
                v.extend(t.encode("system"));
                v.push(self.aux0);
                v.extend(t.encode(text));
                v.extend(self.eot);
                v
            }
            ChatStyle::MiniMax => {
                let mut v = vec![self.user];
                v.extend(t.encode(&format!("system\n{text}")));
                v.push(self.aux0);
                v.extend(t.encode("\n"));
                v
            }
        }
    }

    /// A user message (no assistant opener).
    pub fn render_user(&self, t: &Tokenizer, text: &str) -> Vec<u32> {
        match self.style {
            ChatStyle::Hy3 => {
                let mut v = vec![self.user];
                v.extend(t.encode(text));
                v
            }
            ChatStyle::Kimi => {
                let mut v = vec![self.user];
                v.extend(t.encode("user"));
                v.push(self.aux0);
                v.extend(t.encode(text));
                v.extend(self.eot);
                v
            }
            ChatStyle::ChatMl | ChatStyle::Gemma => {
                let mut v = vec![self.user];
                v.extend(t.encode(&format!("user\n{text}")));
                v.push(self.aux0);
                v.extend(t.encode("\n"));
                v
            }
            ChatStyle::MiniMax => {
                let mut v = vec![self.user];
                v.extend(t.encode(&format!("user\n{text}")));
                v.push(self.aux0);
                v.extend(t.encode("\n"));
                v
            }
        }
    }

    /// The assistant opener; generation starts right after it.
    pub fn open_assistant(&self, t: &Tokenizer) -> Vec<u32> {
        match self.style {
            ChatStyle::Hy3 => vec![self.assistant, self.aux0, self.aux1],
            ChatStyle::Kimi => {
                let mut v = vec![self.assistant];
                v.extend(t.encode("assistant"));
                v.push(self.aux0);
                // thinking off: close the think block immediately
                v.extend(t.encode("<think></think>"));
                v
            }
            ChatStyle::ChatMl => {
                let mut v = vec![self.assistant];
                v.extend(t.encode("assistant\n"));
                v
            }
            ChatStyle::Gemma => {
                let mut v = vec![self.assistant];
                v.extend(t.encode("model\n"));
                v
            }
            ChatStyle::MiniMax => {
                let mut v = vec![self.assistant];
                v.extend(t.encode("ai\n"));
                v.push(self.aux1); // <mm:think>: thinking-enabled prefix
                v
            }
        }
    }

    /// A completed assistant turn from history (opener + content + stop).
    pub fn render_assistant_history(&self, t: &Tokenizer, text: &str) -> Vec<u32> {
        let mut v = self.open_assistant(t);
        v.extend(t.encode(text));
        v.push(self.eot.unwrap_or(self.eos));
        if matches!(self.style, ChatStyle::ChatMl | ChatStyle::Gemma | ChatStyle::MiniMax) {
            v.extend(t.encode("\n"));
        }
        v
    }

    /// One user turn + assistant opener (generation starts after this).
    pub fn render_user_turn(&self, t: &Tokenizer, text: &str) -> Vec<u32> {
        let mut v = self.render_user(t, text);
        v.extend(self.open_assistant(t));
        v
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
            add_bos: match g.metadata.get("tokenizer.ggml.add_bos_token") {
                Some(Value::Bool(b)) => *b,
                _ => g.metadata.get("tokenizer.ggml.model").and_then(Value::as_str)
                    == Some("llama"),
            },
            pre: if g.metadata.get("tokenizer.ggml.model").and_then(Value::as_str)
                == Some("gemma4")
            {
                Pre::Gemma4
            } else {
                match g.metadata.get("tokenizer.ggml.pre").and_then(Value::as_str) {
                    Some("glm4") => Pre::Glm4,
                    Some("kimi-k2") => Pre::KimiK2,
                    Some("minimax-m2") => Pre::MiniMax,
                    Some("qwen2") => Pre::Qwen2,
                    _ => Pre::JoyAi,
                }
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
        if self.pre == Pre::Gemma4 {
            // SPM-style: normalize spaces to U+2581, split only on
            // newline runs, merge raw UTF-8 chars
            let norm = text.replace(' ', "\u{2581}");
            for (is_nl, piece) in split_newline_runs(&norm) {
                if is_nl {
                    // llama.cpp PR 21343: a pure-newline run that exists
                    // in the vocab is one token, skipping the merges
                    if let Some(&id) = self.token_to_id.get(piece) {
                        out.push(id);
                        continue;
                    }
                }
                self.bpe_raw_piece(piece, &mut out);
            }
            return out;
        }
        let pieces = match self.pre {
            Pre::JoyAi => pretokenize(text.as_bytes()),
            Pre::Glm4 => pretokenize_glm4(text.as_bytes()),
            Pre::KimiK2 => pretokenize_kimi_k2(text.as_bytes()),
            Pre::MiniMax => pretokenize_minimax(text.as_bytes()),
            Pre::Qwen2 => pretokenize_qwen2(text.as_bytes()),
            Pre::Gemma4 => unreachable!(),
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
        if self.pre == Pre::Gemma4 {
            for &id in ids {
                let Some(tok) = self.tokens.get(id as usize) else { continue };
                if let Some(b) = parse_byte_token(tok) {
                    out.push(b);
                    continue;
                }
                for c in tok.chars() {
                    if c == '\u{2581}' {
                        out.push(b' ');
                    } else {
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
            return out;
        }
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

    /// Raw-BPE merge on one piece (gemma4): symbols are the piece's
    /// unicode chars verbatim; unknown leftovers fall back to <0xXX>
    /// byte tokens (gemma's vocab has all 256).
    fn bpe_raw_piece(&self, piece: &str, out: &mut Vec<u32>) {
        let mut sym: Vec<String> = piece.chars().map(String::from).collect();
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
        for sm in &sym {
            if let Some(&id) = self.token_to_id.get(sm.as_str()) {
                out.push(id);
            } else {
                for b in sm.as_bytes() {
                    let hex = format!("<0x{b:02X}>");
                    if let Some(&id) = self.token_to_id.get(hex.as_str()) {
                        out.push(id);
                    }
                }
            }
        }
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

/// Split into alternating (is_newline_run, piece) segments.
fn split_newline_runs(s: &str) -> Vec<(bool, &str)> {
    let mut out = Vec::new();
    let b = s.as_bytes();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < b.len() {
        let nl = b[i] == b'\n';
        let run_start = i;
        while i < b.len() && (b[i] == b'\n') == nl {
            i += 1;
        }
        let _ = start;
        start = run_start;
        out.push((nl, &s[run_start..i]));
    }
    out
}

/// "<0xAB>" -> Some(0xAB)
fn parse_byte_token(t: &str) -> Option<u8> {
    let hex = t.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(hex, 16).ok()
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


/// Han (CJK ideograph) check for the kimi-k2 split (llama.cpp
/// unicode_cpt_is_han ranges).
fn kimi_is_han(cp: u32) -> bool {
    matches!(cp,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF
        | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F | 0x2B740..=0x2B81F
        | 0x2B820..=0x2CEAF | 0x2F800..=0x2FA1F)
}

/// kimi-k2 pre-tokenizer (K2 regex via llama.cpp's custom handler):
/// Han runs split alone; letter runs EXCLUDE Han, may take one leading
/// non-letter/non-number char and attach an English contraction; the
/// digit/punct/whitespace tail matches glm4 exactly.

/// minimax-m2/m3 pre-tokenizer: kimi-k2's letter/contraction/digit/ws
/// rules WITHOUT the Han split (Han joins letter runs), and punct runs
/// absorb trailing '/' as well as newlines.
fn pretokenize_minimax(s: &[u8]) -> Vec<&[u8]> {
    let len = s.len();
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < len {
        let start = pos;
        let cur = glm4_char_at(s, pos);
        if !cur.valid {
            break;
        }
        let leading_ok = !(cur.cp == 0x0d || cur.cp == 0x0a || cur.is_letter || cur.is_number);
        let next = glm4_char_at(s, cur.next);
        if cur.is_letter || (leading_ok && next.valid && next.is_letter) {
            pos = cur.next;
            while pos < len {
                let scan = glm4_char_at(s, pos);
                if !scan.valid || !scan.is_letter {
                    break;
                }
                pos = scan.next;
            }
            let ap = glm4_char_at(s, pos);
            if ap.valid && ap.cp == 0x27 && ap.next < len {
                let n1c = glm4_char_at(s, ap.next);
                let n1 = ascii_lower(n1c.cp);
                if matches!(n1, 0x73 | 0x74 | 0x6d | 0x64) {
                    pos = n1c.next;
                } else if n1c.valid && n1c.next < len {
                    let n2c = glm4_char_at(s, n1c.next);
                    let n2 = ascii_lower(n2c.cp);
                    if (n1 == 0x72 && n2 == 0x65) || (n1 == 0x76 && n2 == 0x65) || (n1 == 0x6c && n2 == 0x6c) {
                        pos = n2c.next;
                    }
                }
            }
            out.push(&s[start..pos]);
            continue;
        }
        if cur.is_number {
            let mut nd = 0;
            while pos < len && nd < 3 {
                let scan = glm4_char_at(s, pos);
                if !scan.valid || !scan.is_number {
                    break;
                }
                pos = scan.next;
                nd += 1;
            }
            out.push(&s[start..pos]);
            continue;
        }
        let (mut punct, punct_pos) = if cur.cp == 0x20 {
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
            // trailing newlines AND slashes
            while pos < len {
                let scan = glm4_char_at(s, pos);
                if !scan.valid || !(scan.cp == 0x0d || scan.cp == 0x0a || scan.cp == 0x2f) {
                    break;
                }
                pos = scan.next;
            }
            out.push(&s[start..pos]);
            continue;
        }
        if cur.is_whitespace {
            pos = glm4_whitespace_segment(s, pos, len);
            out.push(&s[start..pos]);
            continue;
        }
        pos = cur.next;
        out.push(&s[start..pos]);
    }
    out
}

/// qwen2 pre-tokenizer (llama.cpp LLAMA_VOCAB_PRE_TYPE_QWEN2 regex):
/// like minimax WITHOUT its extras, plus gpt2-style ordering: an English
/// contraction at the cursor is its own token (checked before the letter
/// run, never absorbed into it), digits split one per token, punct runs
/// absorb trailing newlines only.
fn pretokenize_qwen2(s: &[u8]) -> Vec<&[u8]> {
    let len = s.len();
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < len {
        let start = pos;
        let cur = glm4_char_at(s, pos);
        if !cur.valid {
            break;
        }
        // contraction alternative comes FIRST in the qwen2 regex
        if cur.cp == 0x27 && cur.next < len {
            let n1c = glm4_char_at(s, cur.next);
            let n1 = ascii_lower(n1c.cp);
            let mut end = 0usize;
            if matches!(n1, 0x73 | 0x74 | 0x6d | 0x64) {
                end = n1c.next;
            }
            if n1c.valid && n1c.next < len {
                let n2c = glm4_char_at(s, n1c.next);
                let n2 = ascii_lower(n2c.cp);
                if (n1 == 0x72 && n2 == 0x65) || (n1 == 0x76 && n2 == 0x65) || (n1 == 0x6c && n2 == 0x6c) {
                    end = n2c.next;
                }
            }
            if end > 0 {
                pos = end;
                out.push(&s[start..pos]);
                continue;
            }
        }
        let leading_ok = !(cur.cp == 0x0d || cur.cp == 0x0a || cur.is_letter || cur.is_number);
        let next = glm4_char_at(s, cur.next);
        if cur.is_letter || (leading_ok && next.valid && next.is_letter) {
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
        if cur.is_number {
            pos = cur.next; // \p{N}: one digit per token
            out.push(&s[start..pos]);
            continue;
        }
        let (mut punct, punct_pos) = if cur.cp == 0x20 {
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
        if cur.is_whitespace {
            pos = glm4_whitespace_segment(s, pos, len);
            out.push(&s[start..pos]);
            continue;
        }
        pos = cur.next;
        out.push(&s[start..pos]);
    }
    out
}

fn pretokenize_kimi_k2(s: &[u8]) -> Vec<&[u8]> {
    let len = s.len();
    let mut out = Vec::new();
    let mut pos = 0usize;

    while pos < len {
        let start = pos;
        let cur = glm4_char_at(s, pos);
        if !cur.valid {
            break;
        }

        // Pattern 1: Han run
        if kimi_is_han(cur.cp) {
            pos = cur.next;
            while pos < len {
                let scan = glm4_char_at(s, pos);
                if !scan.valid || !kimi_is_han(scan.cp) {
                    break;
                }
                pos = scan.next;
            }
            out.push(&s[start..pos]);
            continue;
        }

        // Patterns 2/3: letter run (non-Han), optional single leading
        // non-letter/non-number/non-newline char, contraction attached
        let cur_word_letter = cur.is_letter && !kimi_is_han(cur.cp);
        let leading_ok = !(cur.cp == 0x0d || cur.cp == 0x0a || cur.is_letter || cur.is_number);
        let next = glm4_char_at(s, cur.next);
        let next_word_letter = next.valid && next.is_letter && !kimi_is_han(next.cp);
        if cur_word_letter || (leading_ok && next_word_letter) {
            pos = cur.next;
            while pos < len {
                let scan = glm4_char_at(s, pos);
                if !scan.valid || !scan.is_letter || kimi_is_han(scan.cp) {
                    break;
                }
                pos = scan.next;
            }
            // optional contraction: 's 't 'm 'd 're 've 'll
            let ap = glm4_char_at(s, pos);
            if ap.valid && ap.cp == 0x27 && ap.next < len {
                let n1c = glm4_char_at(s, ap.next);
                let n1 = ascii_lower(n1c.cp);
                if matches!(n1, 0x73 | 0x74 | 0x6d | 0x64) {
                    pos = n1c.next;
                } else if n1c.valid && n1c.next < len {
                    let n2c = glm4_char_at(s, n1c.next);
                    let n2 = ascii_lower(n2c.cp);
                    if (n1 == 0x72 && n2 == 0x65) || (n1 == 0x76 && n2 == 0x65) || (n1 == 0x6c && n2 == 0x6c) {
                        pos = n2c.next;
                    }
                }
            }
            out.push(&s[start..pos]);
            continue;
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
        let (mut punct, punct_pos) = if cur.cp == 0x20 {
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

        // whitespace: same policy as glm4
        if cur.is_whitespace {
            pos = glm4_whitespace_segment(s, pos, len);
            out.push(&s[start..pos]);
            continue;
        }

        pos = cur.next;
        out.push(&s[start..pos]);
    }
    out
}


/// glm4/kimi shared whitespace policy: keep the run through its last
/// newline; otherwise leave the final ws char to join the next word.
fn glm4_whitespace_segment(s: &[u8], pos: usize, len: usize) -> usize {
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
        last_newline_end
    } else if nspace > 1 && p < len {
        last_ws_start
    } else {
        p
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
            pos = glm4_whitespace_segment(s, pos, len);
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
    fn kimi_k2_han_runs_and_inline_contractions() {
        let toks = pretokenize_kimi_k2("Hello\u{4f60}\u{597d}world don't 123".as_bytes());
        let strs: Vec<&str> = toks.iter().map(|t| std::str::from_utf8(t).unwrap()).collect();
        // Han run splits alone; contraction stays attached to its word
        assert_eq!(strs, vec!["Hello", "\u{4f60}\u{597d}", "world", " don't", " ", "123"]);
    }

    #[test]
    fn qwen2_standalone_contractions_and_single_digits() {
        let toks = pretokenize_qwen2(b"don't 12 x;\ny");
        let strs: Vec<&str> = toks.iter().map(|b| std::str::from_utf8(b).unwrap()).collect();
        // contraction is its OWN piece (gpt2 order), digits split singly,
        // punct absorbs the newline
        assert_eq!(strs, vec!["don", "'t", " ", "1", "2", " x", ";\n", "y"]);
    }

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
