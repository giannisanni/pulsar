//! GGUF reader. Parses the header, metadata KVs, and tensor table from a
//! byte slice (typically the head of an mmap). Tensor data is never read at
//! parse time: the engine maps or streams it later by (offset, size), which
//! is the whole point for models that dwarf RAM.

use std::collections::HashMap;
use std::fmt;

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
pub const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug)]
pub enum Error {
    BadMagic(u32),
    UnsupportedVersion(u32),
    Truncated { at: usize, need: usize, have: usize },
    BadUtf8 { at: usize },
    BadValueType(u32),
    BadTensorType { tensor: String, ty: u32 },
    TooMany { what: &'static str, count: u64 },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadMagic(m) => write!(f, "not a GGUF file (magic {m:#010x})"),
            Error::UnsupportedVersion(v) => write!(f, "unsupported GGUF version {v}"),
            Error::Truncated { at, need, have } => {
                write!(f, "truncated at byte {at}: need {need}, have {have}")
            }
            Error::BadUtf8 { at } => write!(f, "invalid utf-8 in string at byte {at}"),
            Error::BadValueType(t) => write!(f, "unknown metadata value type {t}"),
            Error::BadTensorType { tensor, ty } => {
                write!(f, "tensor {tensor}: unknown ggml type {ty}")
            }
            Error::TooMany { what, count } => write!(f, "implausible {what} count {count}"),
        }
    }
}

impl std::error::Error for Error {}

/// GGML tensor quantization/storage types (the subset a streaming MoE
/// engine meets in the wild, plus room to grow).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TensorType {
    F32,
    F16,
    Q4_0,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    IQ3S,
    IQ2S,
    IQ4XS,
    IQ4NL,
    BF16,
    /// Anything we do not model yet; the raw id is preserved.
    Other(u32),
}

impl TensorType {
    pub fn from_id(id: u32) -> Self {
        match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            8 => Self::Q8_0,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::IQ2XXS,
            17 => Self::IQ2XS,
            18 => Self::IQ3XXS,
            20 => Self::IQ4NL,
            21 => Self::IQ3S,
            22 => Self::IQ2S,
            23 => Self::IQ4XS,
            30 => Self::BF16,
            other => Self::Other(other),
        }
    }

    /// (block size in elements, bytes per block); None for Other.
    pub fn block_layout(self) -> Option<(u64, u64)> {
        Some(match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::BF16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q8_0 => (32, 34),
            Self::Q2K => (256, 84),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Q8K => (256, 292),
            Self::IQ2XXS => (256, 66),
            Self::IQ2XS => (256, 74),
            Self::IQ2S => (256, 82),
            Self::IQ3XXS => (256, 98),
            Self::IQ3S => (256, 110),
            Self::IQ4XS => (256, 136),
            Self::IQ4NL => (32, 18),
            Self::Other(_) => return None,
        })
    }

    /// Bytes a row of `n` elements occupies, if the layout is known.
    pub fn row_bytes(self, n: u64) -> Option<u64> {
        let (bs, bb) = self.block_layout()?;
        Some(n.div_ceil(bs) * bb)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I8(v) if v >= 0 => Some(v as u64),
            Value::I16(v) if v >= 0 => Some(v as u64),
            Value::I32(v) if v >= 0 => Some(v as u64),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            Value::F64(v) => Some(v as f32),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions in GGUF order (fastest-varying first).
    pub dims: Vec<u64>,
    pub ty: TensorType,
    /// Offset relative to the start of the data section.
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Total byte size, if the type layout is known. Row = dims[0].
    pub fn byte_size(&self) -> Option<u64> {
        let row = *self.dims.first()?;
        let rows: u64 = self.dims.iter().skip(1).product();
        Some(self.ty.row_bytes(row)? * rows.max(1))
    }
}

#[derive(Debug)]
pub struct Gguf {
    pub version: u32,
    pub metadata: HashMap<String, Value>,
    /// Tensor table in file order.
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    /// Absolute file offset where the tensor data section begins.
    pub data_offset: u64,
}

impl Gguf {
    /// Parse header + metadata + tensor table from the head of a GGUF file.
    /// `head` does not need to contain tensor data; it must merely cover the
    /// header (a few MB for big-vocab models).
    pub fn parse(head: &[u8]) -> Result<Self, Error> {
        let mut c = Cursor { buf: head, at: 0 };
        let magic = c.u32()?;
        if magic != GGUF_MAGIC {
            return Err(Error::BadMagic(magic));
        }
        let version = c.u32()?;
        if !(2..=3).contains(&version) {
            return Err(Error::UnsupportedVersion(version));
        }
        let tensor_count = c.u64()?;
        if tensor_count > 1 << 20 {
            return Err(Error::TooMany { what: "tensor", count: tensor_count });
        }
        let kv_count = c.u64()?;
        if kv_count > 1 << 20 {
            return Err(Error::TooMany { what: "metadata kv", count: kv_count });
        }

        let mut metadata = HashMap::with_capacity(kv_count as usize);
        for _ in 0..kv_count {
            let key = c.string()?;
            let ty = c.u32()?;
            let value = c.value(ty, 0)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = c.string()?;
            let n_dims = c.u32()?;
            if n_dims > 8 {
                return Err(Error::TooMany { what: "tensor dim", count: n_dims as u64 });
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(c.u64()?);
            }
            let ty_id = c.u32()?;
            let offset = c.u64()?;
            tensors.push(TensorInfo { name, dims, ty: TensorType::from_id(ty_id), offset });
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(Value::as_u64)
            .filter(|a| a.is_power_of_two())
            .unwrap_or(DEFAULT_ALIGNMENT);
        let data_offset = (c.at as u64).next_multiple_of(alignment);

        Ok(Gguf { version, metadata, tensors, alignment, data_offset })
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture")?.as_str()
    }

    /// Convenience: `<architecture>.<suffix>` metadata lookup.
    pub fn arch_meta(&self, suffix: &str) -> Option<&Value> {
        let arch = self.architecture()?;
        self.metadata.get(&format!("{arch}.{suffix}"))
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

struct Cursor<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.at.checked_add(n).ok_or(Error::Truncated {
            at: self.at,
            need: n,
            have: 0,
        })?;
        if end > self.buf.len() {
            return Err(Error::Truncated {
                at: self.at,
                need: n,
                have: self.buf.len() - self.at,
            });
        }
        let s = &self.buf[self.at..end];
        self.at = end;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String, Error> {
        let len = self.u64()? as usize;
        let at = self.at;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| Error::BadUtf8 { at })
    }

    fn value(&mut self, ty: u32, depth: u8) -> Result<Value, Error> {
        Ok(match ty {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            3 => Value::I16(i16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::String(self.string()?),
            9 => {
                if depth > 2 {
                    return Err(Error::BadValueType(ty));
                }
                let elem_ty = self.u32()?;
                let count = self.u64()?;
                if count > 1 << 26 {
                    return Err(Error::TooMany { what: "array element", count });
                }
                let mut v = Vec::with_capacity(count.min(1 << 16) as usize);
                for _ in 0..count {
                    v.push(self.value(elem_ty, depth + 1)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            other => return Err(Error::BadValueType(other)),
        })
    }
}
