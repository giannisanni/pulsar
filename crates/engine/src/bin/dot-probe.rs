//! One-row GPU-vs-CPU dot arbiter (task #38): reads the slab + normed
//! dumps from PULSAR_CPU_VERIFY and prints the same dot three ways.
//! Usage: dot-probe <row.bin> <normed.bin> <n_embd> <row_bytes> <quant>

#[cfg(not(target_os = "linux"))]
fn main() {}

#[cfg(target_os = "linux")]
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let row = std::fs::read(&a[1]).expect("row file");
    let normed_raw = std::fs::read(&a[2]).expect("normed file");
    let ne: usize = a[3].parse().unwrap();
    let row_bytes: u64 = a[4].parse().unwrap();
    let quant: u32 = a[5].parse().unwrap();
    let normed: Vec<f32> = normed_raw
        .chunks_exact(4)
        .take(ne)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // CPU: host quantize + host dot
    let xq = quant::cpu_dot::quantize_row_q8_k(&normed);
    let cpu = match quant {
        q if q == kernels::QUANT_IQ2_XXS => quant::cpu_dot::vec_dot_iq2_xxs_q8_k(&row, &xq, ne),
        q if q == kernels::QUANT_Q4_K => quant::cpu_dot::vec_dot_q4_k_q8_k(&row, &xq, ne),
        _ => panic!("probe quant"),
    };

    // GPU: device quantize + matmul_kq single row
    let mut w = kernels::DeviceBuf::alloc(row_bytes as usize + 256).unwrap();
    w.write(0, &row[..row_bytes as usize]).unwrap();
    let x = kernels::DeviceBuf::from_f32(&normed).unwrap();
    let mut xqd = kernels::DeviceBuf::alloc(ne / 256 * 292 + 256).unwrap();
    kernels::quantize_q8_k(&mut xqd, &x, ne as u32, 1).unwrap();
    let mut out = kernels::DeviceBuf::alloc(16).unwrap();
    kernels::matmul_kq(&mut out, &w, &xqd, ne as u32, 1, 1, row_bytes, quant).unwrap();
    kernels::sync().unwrap();
    let gpu = out.read_f32(1).unwrap()[0];

    // GPU dot with HOST-quantized activations (isolates the quantizer)
    let mut xqh = kernels::DeviceBuf::alloc(ne / 256 * 292 + 256).unwrap();
    let mut hostq = Vec::new();
    for b in 0..ne / 256 {
        hostq.extend_from_slice(&xq.d[b].to_le_bytes());
        hostq.extend_from_slice(kernels::as_bytes(
            &xq.qs[b * 256..(b + 1) * 256].iter().map(|&v| v).collect::<Vec<i8>>(),
        ));
        for g in 0..16 {
            hostq.extend_from_slice(&(xq.bsums[b * 16 + g] as i16).to_le_bytes());
        }
    }
    xqh.write(0, &hostq).unwrap();
    let mut out2 = kernels::DeviceBuf::alloc(16).unwrap();
    kernels::matmul_kq(&mut out2, &w, &xqh, ne as u32, 1, 1, row_bytes, quant).unwrap();
    kernels::sync().unwrap();
    let gpu_hostq = out2.read_f32(1).unwrap()[0];

    println!("cpu={cpu:.6} gpu={gpu:.6} gpu(host-xq)={gpu_hostq:.6}");
}
