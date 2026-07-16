//! pulsar-quant: rewrite a BF16/F16/F32 gguf (single or -00001-of-N split)
//! into a recipe-quantized single gguf.
//!
//!   pulsar-quant -i model-BF16-00001-of-00003.gguf -o out.gguf \
//!       --map "_exps.=q2_k" --default q8_0
//!
//! Recipe rules: `--map pat=type` (repeatable, comma-separable) matches by
//! SUBSTRING against the tensor name, first match wins; `--default` covers
//! unmatched 2D+ tensors. 1D tensors (norms, biases) always stay f32.
//! K-quant targets need row width % 256 == 0; offenders fall back to q8_0
//! (width % 32) or f16, with a warning - same guardrails llama.cpp applies.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;

use gguf::{Gguf, TensorType, Value};

fn parse_type(s: &str) -> Result<TensorType, String> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "q8_0" => TensorType::Q8_0,
        "q2_k" => TensorType::Q2K,
        "q3_k" => TensorType::Q3K,
        "q4_k" => TensorType::Q4K,
        "q5_k" => TensorType::Q5K,
        "q6_k" => TensorType::Q6K,
        "iq2_xxs" => TensorType::IQ2XXS,
        "f16" => TensorType::F16,
        "f32" => TensorType::F32,
        other => return Err(format!("unknown target type {other} (stage 1: q8_0 q2_k..q6_k f16 f32)")),
    })
}

fn parse_header(path: &std::path::Path) -> Result<Gguf, String> {
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut n = 32 << 20;
    loop {
        let mut head = vec![0u8; n];
        let got = {
            let mut read = 0;
            while read < head.len() {
                match f.read(&mut head[read..]) {
                    Ok(0) => break,
                    Ok(k) => read += k,
                    Err(e) => return Err(e.to_string()),
                }
            }
            read
        };
        match Gguf::parse(&head[..got]) {
            Ok(g) => return Ok(g),
            Err(gguf::Error::Truncated { .. }) if got == n => {
                n *= 2;
                f.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
            }
            Err(e) => return Err(format!("{}: {e:?}", path.display())),
        }
    }
}

/// Route a virtual offset to (file index, local offset).
struct VFile {
    files: Vec<(u64, File)>, // (base, file)
}

impl VFile {
    fn read_exact_at(&self, buf: &mut [u8], off: u64) -> std::io::Result<()> {
        let i = match self.files.binary_search_by(|(b, _)| b.cmp(&off)) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        self.files[i].1.read_exact_at(buf, off - self.files[i].0)
    }
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn value_type_id(v: &Value) -> u32 {
    match v {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::String(_) => 8,
        Value::Array(_) => 9,
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    }
}

fn write_value_payload(out: &mut Vec<u8>, v: &Value) -> Result<(), String> {
    match v {
        Value::U8(x) => out.push(*x),
        Value::I8(x) => out.push(*x as u8),
        Value::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::Bool(x) => out.push(*x as u8),
        Value::String(s) => write_string(out, s),
        Value::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::Array(items) => {
            let elem_ty = items.first().map(value_type_id).unwrap_or(4);
            if items.iter().any(|it| value_type_id(it) != elem_ty) {
                return Err("heterogeneous metadata array".into());
            }
            out.extend_from_slice(&elem_ty.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for it in items {
                write_value_payload(out, it)?;
            }
        }
    }
    Ok(())
}

struct OutTensor {
    name: String,
    dims: Vec<u64>,
    ty: TensorType,
    src_ty: TensorType,
    src_off: u64, // virtual absolute
    out_off: u64, // relative to output data section
}

fn main() {
    if let Err(e) = run() {
        eprintln!("pulsar-quant: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut input = None;
    let mut output = None;
    let mut maps: Vec<(String, TensorType)> = Vec::new();
    let mut default_ty = TensorType::Q8_0;
    let mut imatrix_path: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut need = |what: &str| args.next().ok_or(format!("{what} needs a value"));
        match a.as_str() {
            "-i" => input = Some(need("-i")?),
            "-o" => output = Some(need("-o")?),
            "--map" => {
                for part in need("--map")?.split(',') {
                    let (pat, ty) = part.split_once('=').ok_or(format!("bad --map entry {part}"))?;
                    maps.push((pat.to_string(), parse_type(ty)?));
                }
            }
            "--default" => default_ty = parse_type(&need("--default")?)?,
            "--imatrix" => imatrix_path = Some(need("--imatrix")?),
            other => return Err(format!("unknown arg {other}")),
        }
    }
    let input = std::path::PathBuf::from(input.ok_or("-i required")?);
    let output = std::path::PathBuf::from(output.ok_or("-o required")?);
    let imatrix = match &imatrix_path {
        Some(p) => Some(quant::iq::read_imatrix(std::path::Path::new(p))?),
        None => None,
    };

    // ---- read headers (single file or split), build virtual view
    let shard_paths = gguf::split_shards(&input).unwrap_or_else(|| vec![input.clone()]);
    let mut shards = Vec::new();
    let mut bases = Vec::new();
    let mut base = 0u64;
    for p in &shard_paths {
        let g = parse_header(p)?;
        bases.push(base);
        base += std::fs::metadata(p).map_err(|e| e.to_string())?.len();
        shards.push(g);
    }
    let merged = Gguf::merge_split(shards, &bases);
    let vfile = VFile {
        files: shard_paths
            .iter()
            .zip(&bases)
            .map(|(p, &b)| Ok((b, File::open(p).map_err(|e: std::io::Error| e.to_string())?)))
            .collect::<Result<_, String>>()?,
    };
    eprintln!(
        "pulsar-quant: {} tensors, arch {}, {} shard(s)",
        merged.tensors.len(),
        merged.architecture().unwrap_or("?"),
        shard_paths.len()
    );

    // ---- decide output types
    let pick = |name: &str, dims: &[u64]| -> TensorType {
        if dims.len() < 2 {
            return TensorType::F32;
        }
        let want = maps
            .iter()
            .find(|(pat, _)| name.contains(pat.as_str()))
            .map(|&(_, ty)| ty)
            .unwrap_or(default_ty);
        let row = dims[0];
        let ok = match want {
            TensorType::Q2K | TensorType::Q3K | TensorType::Q4K | TensorType::Q5K
            | TensorType::Q6K | TensorType::IQ2XXS => row % 256 == 0,
            TensorType::Q8_0 => row % 32 == 0,
            _ => true,
        };
        if ok {
            want
        } else if row % 32 == 0 {
            eprintln!("pulsar-quant: {name} width {row} not /256, falling back to q8_0");
            TensorType::Q8_0
        } else {
            eprintln!("pulsar-quant: {name} width {row} not /32, falling back to f16");
            TensorType::F16
        }
    };

    let mut out_tensors = Vec::with_capacity(merged.tensors.len());
    let mut out_off = 0u64;
    let align = merged.alignment.max(32);
    for t in &merged.tensors {
        match t.ty {
            TensorType::F32 | TensorType::F16 | TensorType::BF16 => {}
            other => return Err(format!("{}: source type {other:?} is not a float type", t.name)),
        }
        let mut ty = pick(&t.name, &t.dims);
        if ty == TensorType::IQ2XXS {
            let row = t.dims[0];
            let n_exp = t.dims.get(2).copied().unwrap_or(1);
            let ok = imatrix
                .as_ref()
                .and_then(|m| m.get(&t.name))
                .is_some_and(|e| e.len() as u64 == row || e.len() as u64 == row * n_exp);
            if !ok {
                eprintln!("pulsar-quant: {} has no usable imatrix entry, falling back to q2_k", t.name);
                ty = TensorType::Q2K;
            }
        }
        let row = *t.dims.first().unwrap_or(&1);
        let rows: u64 = t.dims.iter().skip(1).product::<u64>().max(1);
        let bytes = ty.row_bytes(row).ok_or("row_bytes")? * rows;
        out_tensors.push(OutTensor {
            name: t.name.clone(),
            dims: t.dims.clone(),
            ty,
            src_ty: t.ty,
            src_off: t.offset,
            out_off,
        });
        out_off = (out_off + bytes).next_multiple_of(align);
    }

    // ---- header
    let mut meta: Vec<(&String, &Value)> = merged
        .metadata
        .iter()
        .filter(|(k, _)| !k.starts_with("split."))
        .collect();
    meta.sort_by(|a, b| a.0.cmp(b.0));
    let mut head = Vec::with_capacity(16 << 20);
    head.extend_from_slice(&gguf::GGUF_MAGIC.to_le_bytes());
    head.extend_from_slice(&3u32.to_le_bytes());
    head.extend_from_slice(&(out_tensors.len() as u64).to_le_bytes());
    head.extend_from_slice(&(meta.len() as u64).to_le_bytes());
    for (k, v) in meta {
        write_string(&mut head, k);
        head.extend_from_slice(&value_type_id(v).to_le_bytes());
        write_value_payload(&mut head, v)?;
    }
    for t in &out_tensors {
        write_string(&mut head, &t.name);
        head.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            head.extend_from_slice(&d.to_le_bytes());
        }
        head.extend_from_slice(&t.ty.to_id().to_le_bytes());
        head.extend_from_slice(&t.out_off.to_le_bytes());
    }
    let data_start = (head.len() as u64).next_multiple_of(align);

    let f = File::create(&output).map_err(|e| e.to_string())?;
    let mut w = BufWriter::with_capacity(8 << 20, f);
    w.write_all(&head).map_err(|e| e.to_string())?;
    w.write_all(&vec![0u8; (data_start - head.len() as u64) as usize])
        .map_err(|e| e.to_string())?;

    // ---- data: read each tensor, encode rows in parallel, write
    let nthread = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let t0 = std::time::Instant::now();
    let mut written = 0u64;
    let mut by_type: HashMap<String, u64> = HashMap::new();
    for (i, t) in out_tensors.iter().enumerate() {
        let row = *t.dims.first().unwrap_or(&1) as usize;
        let rows = t.dims.iter().skip(1).product::<u64>().max(1) as usize;
        let src_row_bytes = t.src_ty.row_bytes(row as u64).unwrap() as usize;
        let out_row_bytes = t.ty.row_bytes(row as u64).unwrap() as usize;

        // read whole source tensor (largest single tensors in BF16 are a
        // few GB; fine on a 30GB box, shard streaming reuses this path)
        let mut src = vec![0u8; src_row_bytes * rows];
        vfile
            .read_exact_at(&mut src, t.src_off)
            .map_err(|e| format!("{}: read {e}", t.name))?;

        let chunk_rows = rows.div_ceil(nthread);
        let mut parts: Vec<Vec<u8>> = Vec::with_capacity(nthread);
        std::thread::scope(|s| -> Result<(), String> {
            let mut handles = Vec::new();
            for c in 0..nthread {
                let lo = c * chunk_rows;
                if lo >= rows {
                    break;
                }
                let hi = ((c + 1) * chunk_rows).min(rows);
                let src = &src[lo * src_row_bytes..hi * src_row_bytes];
                let (src_ty, out_ty) = (t.src_ty, t.ty);
                let entry = imatrix.as_ref().and_then(|m| m.get(&t.name));
                let ne1 = t.dims.get(1).copied().unwrap_or(1) as usize;
                handles.push(s.spawn(move || -> Result<Vec<u8>, String> {
                    let mut buf = Vec::with_capacity((hi - lo) * out_row_bytes);
                    let mut f32row = Vec::with_capacity(row);
                    for (k, r) in src.chunks_exact(src_row_bytes).enumerate() {
                        quant::row_to_f32(src_ty, r, &mut f32row)?;
                        let qw = entry.map(|e| {
                            if e.len() == row {
                                &e[..]
                            } else {
                                // per-expert imatrix: ne0 * n_expert values
                                let expert = (lo + k) / ne1.max(1);
                                &e[expert * row..(expert + 1) * row]
                            }
                        });
                        quant::quantize_row(out_ty, &f32row, qw, &mut buf)?;
                    }
                    Ok(buf)
                }));
            }
            for h in handles {
                parts.push(h.join().map_err(|_| "encode thread panicked")??);
            }
            Ok(())
        })?;

        let mut nbytes = 0u64;
        for p in &parts {
            w.write_all(p).map_err(|e| e.to_string())?;
            nbytes += p.len() as u64;
        }
        // pad to the next tensor's aligned offset
        let end = t.out_off + nbytes;
        let next = out_tensors
            .get(i + 1)
            .map(|n| n.out_off)
            .unwrap_or(end.next_multiple_of(align));
        w.write_all(&vec![0u8; (next - end) as usize]).map_err(|e| e.to_string())?;
        written += nbytes;
        *by_type.entry(format!("{:?}", t.ty)).or_default() += nbytes;
        if i % 50 == 0 || i + 1 == out_tensors.len() {
            eprintln!(
                "pulsar-quant: [{}/{}] {} ({:.1}GB written, {:.0}s)",
                i + 1,
                out_tensors.len(),
                t.name,
                written as f64 / 1e9,
                t0.elapsed().as_secs_f32()
            );
        }
    }
    w.flush().map_err(|e| e.to_string())?;
    let mut summary: Vec<_> = by_type.into_iter().collect();
    summary.sort_by(|a, b| b.1.cmp(&a.1));
    for (ty, b) in summary {
        eprintln!("pulsar-quant: {ty}: {:.2} GB", b as f64 / 1e9);
    }
    eprintln!(
        "pulsar-quant: wrote {} ({:.2} GB) in {:.0}s",
        output.display(),
        (data_start + out_off) as f64 / 1e9,
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}
