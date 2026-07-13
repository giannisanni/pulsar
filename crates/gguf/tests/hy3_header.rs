//! Parses the real header of the Hy3 295B ds4 quant (first 16MB of the
//! production gguf; tensor data absent by design - the parser must never
//! need it).

use gguf::{Gguf, TensorType};

fn fixture() -> Vec<u8> {
    std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/hy3-header.bin"
    ))
    .expect("fixture missing: crates/gguf/tests/fixtures/hy3-header.bin")
}

#[test]
fn parses_hy3_production_header() {
    let g = Gguf::parse(&fixture()).expect("parse");

    assert_eq!(g.architecture(), Some("hy-v3"));
    assert_eq!(g.arch_meta("block_count").and_then(|v| v.as_u64()), Some(81));
    assert_eq!(g.arch_meta("expert_count").and_then(|v| v.as_u64()), Some(192));
    assert_eq!(g.arch_meta("expert_used_count").and_then(|v| v.as_u64()), Some(8));

    // routed experts on the streaming path: uniform IQ2_XXS, layers 1..=79
    let gate1 = g.tensor("blk.1.ffn_gate_exps.weight").expect("blk.1 gate");
    assert_eq!(gate1.ty, TensorType::IQ2XXS);
    // the MTP draft layer rides at Q2_K (imatrix never covers it)
    let gate80 = g.tensor("blk.80.ffn_gate_exps.weight").expect("blk.80 gate");
    assert_eq!(gate80.ty, TensorType::Q2K);
    // resident set at Q8_0
    let q0 = g.tensor("blk.0.attn_q.weight").expect("attn_q");
    assert_eq!(q0.ty, TensorType::Q8_0);

    // every tensor the streaming engine will slab-address must have a
    // computable byte size
    for t in &g.tensors {
        assert!(
            t.byte_size().is_some(),
            "tensor {} has unmodeled type {:?}",
            t.name,
            t.ty
        );
    }

    // uniform expert slabs: every streamed layer's gate tensor is the same
    // shape and therefore the same slab size (the cache's core assumption)
    let slab = gate1.byte_size().unwrap();
    for il in 2..=79u32 {
        let t = g
            .tensor(&format!("blk.{il}.ffn_gate_exps.weight"))
            .expect("gate tensor");
        assert_eq!(t.byte_size().unwrap(), slab, "layer {il} slab mismatch");
    }

    assert!(g.data_offset % g.alignment == 0);
    assert!(g.tensors.len() > 900, "expected ~1k tensors, got {}", g.tensors.len());
}
