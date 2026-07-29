# Ornith-397B Q2_K CUDA crash — root cause and fix

**Date:** 2026-07-28
**Model:** Ornith-397B-Q2_K.gguf (`qwen35moe` arch → `Family::Qwen35`)
**Quant:** `pulsar-quant --map "_exps.=q2_k" --default q8_0` — experts Q2_K,
everything else Q8_0.
**Symptom:** `pulsar-cli: cuda kernel op failed: cudaDeviceSynchronize`
immediately after warm start, during the first prefill chunk. No output.

## Root cause

Two unrelated load-site bugs, both the same shape: a tensor was uploaded with
the generic `upload()` and then consumed by a kernel that reads **f32**.

`upload()` → `read_tensor_bytes()` has a branch
(`crates/engine/src/lib.rs:2518`):

```rust
if t.ty == TensorType::F16 && t.dims.len() >= 2 { /* requantize to q8_0 */ }
```

That branch exists for the **matmul** consumers (`matw` / `matmul_q8_0`),
which want q8_0 bytes. But three tensors in the Qwen3.5 GDN + MoE path are **not**
matmul-consumed — their kernels read the raw f32 — and two of them ship as F16
or Q8_0, so `upload()` either requantized (F16→Q8_0) or left raw q8_0 bytes,
then the f32 kernel read **past** the (4× smaller) allocation. That is an
out-of-bounds device read → NaN cascade → `cudaDeviceSynchronize` abort.

The three f32-reader tensors and what their kernels expect:

| tensor | ship type | ship shape | consumer kernel | wants |
|---|---|---|---|---|
| `ssm_conv1d.weight` | F16 | `[conv_dim, ssm_conv_k]` = `[12288, 4]` | `qwen35_conv_batch` | f32 `[conv_dim][ssm_conv_k]` |
| `ffn_gate_inp.weight` | Q8_0 | `[n_embd, n_expert]` = `[4096, 512]` | `matmul_f32` (router) | f32 |
| `ffn_gate_inp_shexp.weight` | Q8_0 | `[n_embd, 1]` = `[4096, 1]` | `matmul_f32` → `qwen35_row_sigmoid_scale` | f32 |

For context, the GDN `alpha_w` / `beta_w` weights had already been fixed the
same way in prior commits (`c636368`, `694ead0`) — they ship Q8_0 and are read
by `matmul_f32`. The three above were the remaining holes.

## Fix

Route all three through `upload_as_f32()`, which dequantizes every ship format
(F32 raw, F16→f32, Q8_0→f32, Q4_K→f32) into a proper f32 `DeviceBuf`.

`crates/engine/src/lib.rs`:

1. **GDN conv weight** (~line 3604):

   ```rust
   // conv kernel reads this as f32 [conv_dim][ssm_conv_k];
   // upload() would quantize the F16 2D tensor to q8_0 bytes,
   // which the conv kernel reads past (4x size mismatch) -> OOB.
   conv: upload_as_f32(&file, &gguf, &t("ssm_conv1d.weight"))?,
   ```

2. **MoE router weight** (`gate_inp`, ~line 3281) — collapsed the
   `dsv4_arch ? upload_f16_as_f32 : upload` fork to a single
   `upload_as_f32`; `upload_as_f32` already covers dsv4's F16, qwen35moe's
   Q8_0, and plain F32, so the `upload()` arm (raw q8_0 bytes → OOB) was just
   wrong for any non-f32 ship:

   ```rust
   // matmul_f32 wants f32 (router precision drives selection).
   // upload_as_f32 covers every ship format: dsv4's f16, qwen35moe's q8_0,
   // plain f32. The old upload() branch returned raw q8_0 bytes for qwen35moe,
   // which matmul_f32 read past (8MB read on a 2MB buffer).
   gate_inp: upload_as_f32(&file, &gguf, &t("ffn_gate_inp.weight"))?,
   ```

3. **Shared-expert router weight** (`shexp_gate`, ~line 3611):

   ```rust
   // matmul_f32 consumer (qwen35_row_sigmoid_scale path): q8_0 here would
   // be read as f32 and OOB.
   shexp_gate: if gguf.tensor(&t("ffn_gate_inp_shexp.weight")).is_some() {
       upload_as_f32(&file, &gguf, &t("ffn_gate_inp_shexp.weight"))?
   } else {
       DeviceBuf::alloc(4)?
   },
   ```

## Verification

```text
$ CUDA_VISIBLE_DEVICES=0 ./target/release/pulsar-cli \
    -m /home/cesar/models/Ornith-397B-Q2_K.gguf -p "The capital of France is" -n 15
pulsar: loading /home/cesar/models/Ornith-397B-Q2_K.gguf
pulsar: using CUDA device 0
pulsar: loaded in 7.8s (60 layers, 512 experts x top-10)
pulsar: prompt ids [760, 6511, 314, 9338, 369]
pulsar: auto budget: 10.9GiB VRAM free -> expert cache 8.2GiB, staging 2.0GiB, prefill chunk 256
pulsar: warm start: 1800 slabs in 2.5s
pulsar: prefill 5 tokens in 11.51s
 Paris.
A. 对
B. 错
答案:
pulsar: 15 tokens in 18.28s (0.82 tok/s), vram cache 33% hits, host cache 19% of remainder
```

" Paris." is the correct continuation. The Chinese A/B/答案 tail is the
model's own instruct template leaking through a one-shot prompt with no chat
template — unrelated to the crash. Sustained warm decode rate is not reported
here: 15 tokens off a cold census is not a bench, and the model sits at 397B
on a single GPU where the disk is the floor.

## The general rule

`upload()` is for **matmul** consumers that want q8_0 bytes (or raw bytes for
already-integer tensors). Any tensor whose kernel reads **f32** must go through
`upload_as_f32()`. The `read_tensor_bytes` F16-2D→Q8_0 branch is a matmul
optimization and is a trap for every f32 reader; the GDN and router weights are
the cases where it bit.
