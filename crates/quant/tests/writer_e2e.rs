//! End-to-end: build a tiny BF16 gguf, run the pulsar-quant binary on it,
//! parse the result with the gguf crate, and check values round-trip.

use std::io::Write;

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

#[test]
fn quantize_tiny_model() {
    let dir = std::env::temp_dir().join(format!("pq-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("tiny-BF16.gguf");
    let output = dir.join("tiny-recipe.gguf");

    // source values: deterministic, smooth-ish
    let width = 512usize;
    let rows = 8usize;
    let vals: Vec<f32> = (0..width * rows)
        .map(|i| ((i as f32) * 0.37).sin() * 0.8)
        .collect();
    let bias: Vec<f32> = (0..width).map(|i| i as f32 * 0.001).collect();

    // ---- hand-build the input gguf (v3)
    let mut h = Vec::new();
    h.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    h.extend_from_slice(&3u32.to_le_bytes());
    h.extend_from_slice(&3u64.to_le_bytes()); // tensors
    h.extend_from_slice(&2u64.to_le_bytes()); // kvs
    put_str(&mut h, "general.architecture");
    h.extend_from_slice(&8u32.to_le_bytes());
    put_str(&mut h, "test");
    put_str(&mut h, "split.count"); // must be stripped by the writer
    h.extend_from_slice(&4u32.to_le_bytes());
    h.extend_from_slice(&1u32.to_le_bytes());

    let bf16_bytes = |v: &[f32]| -> Vec<u8> {
        v.iter().flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes()).collect()
    };
    let t_exps = bf16_bytes(&vals);
    let t_attn = bf16_bytes(&vals);
    let f32_bytes: Vec<u8> = bias.iter().flat_map(|x| x.to_le_bytes()).collect();

    // tensor table: offsets relative to data section, 32-aligned
    let mut data = Vec::new();
    let mut offs = Vec::new();
    for t in [&t_exps, &t_attn, &f32_bytes] {
        offs.push(data.len() as u64);
        data.extend_from_slice(t);
        while data.len() % 32 != 0 {
            data.push(0);
        }
    }
    let dims2 = [width as u64, rows as u64];
    for (i, (name, ty, dims)) in [
        ("blk.0.ffn_gate_exps.weight", 30u32, &dims2[..]),
        ("blk.0.attn_q.weight", 30u32, &dims2[..]),
        ("blk.0.attn_norm.weight", 0u32, &dims2[..1]),
    ]
    .iter()
    .enumerate()
    {
        put_str(&mut h, name);
        h.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in *dims {
            h.extend_from_slice(&d.to_le_bytes());
        }
        h.extend_from_slice(&ty.to_le_bytes());
        h.extend_from_slice(&offs[i].to_le_bytes());
    }
    while h.len() % 32 != 0 {
        h.push(0);
    }
    let mut f = std::fs::File::create(&input).unwrap();
    f.write_all(&h).unwrap();
    f.write_all(&data).unwrap();
    drop(f);

    // ---- run the binary
    let st = std::process::Command::new(env!("CARGO_BIN_EXE_pulsar-quant"))
        .args(["-i"])
        .arg(&input)
        .args(["-o"])
        .arg(&output)
        .args(["--map", "_exps.=q2_k", "--default", "q8_0"])
        .status()
        .unwrap();
    assert!(st.success());

    // ---- parse + verify
    let head = std::fs::read(&output).unwrap();
    let g = gguf::Gguf::parse(&head).unwrap();
    assert_eq!(g.architecture(), Some("test"));
    assert!(!g.metadata.contains_key("split.count"), "split.* must be stripped");
    let exps = g.tensor("blk.0.ffn_gate_exps.weight").unwrap();
    assert_eq!(exps.ty, gguf::TensorType::Q2K);
    let attn = g.tensor("blk.0.attn_q.weight").unwrap();
    assert_eq!(attn.ty, gguf::TensorType::Q8_0);
    let norm = g.tensor("blk.0.attn_norm.weight").unwrap();
    assert_eq!(norm.ty, gguf::TensorType::F32);

    // dequant the q8_0 tensor and compare
    let start = (g.data_offset + attn.offset) as usize;
    let nblocks = width * rows / 32;
    let mut dec = Vec::with_capacity(width * rows);
    for b in head[start..start + nblocks * 34].chunks_exact(34) {
        let hbits = u16::from_le_bytes([b[0], b[1]]);
        let d = quant::f16_to_f32(hbits);
        for i in 0..32 {
            dec.push(d * (b[2 + i] as i8) as f32);
        }
    }
    let rms: f32 = (vals.iter().map(|v| v * v).sum::<f32>() / vals.len() as f32).sqrt();
    let err: f32 = (vals
        .iter()
        .zip(&dec)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / vals.len() as f32)
        .sqrt();
    assert!(err / rms < 0.01, "q8_0 rel rmse {}", err / rms);

    // f32 pass-through must be exact
    let nstart = (g.data_offset + norm.offset) as usize;
    let back: Vec<f32> = head[nstart..nstart + width * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(back, bias);

    std::fs::remove_dir_all(&dir).ok();
}
