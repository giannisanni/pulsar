/* Ported verbatim from the NeutronStar fork of ds4
 * (github.com/antirez/ds4), MIT License:
 *   Copyright (c) 2026 The ds4.c authors
 *   Copyright (c) 2023-2026 The ggml authors
 *
 * GQA full attention for Hy3-class models (hy_v3: 64 q heads / 8 kv heads,
 * head_dim 128, per-head QK-RMSNorm with weights, rotate-half RoPE).
 *
 * ds4's existing attention paths are all MLA-family (shared latent KV, no
 * per-kv-head cache); this file adds the grouped-query path: a standard
 * per-head KV cache [n_kv_head][cap][head_dim] and a decode/prefill
 * attention kernel where q head h attends kv head h / (n_head / n_kv_head).
 *
 * Everything here is validated by ds4_gpu_gqa_selftest(): a CPU reference
 * implementation runs the identical pipeline on random data and the kernels
 * must match it numerically. Run with `ds4 --gqa-selftest` (no model file
 * needed), including a multi-chunk phase for the online-softmax rescale.
 *
 * RoPE convention: HF Llama rotate-half. Pair (i, i + d/2), angle
 * pos * theta^(-2i/d). Hy3 rope_type is "default" (no yarn/ntk scaling). */

/* ---- kernels ---- */

/* Per-head RMSNorm with a [head_dim] weight vector, in place.
 * x: [rows][head_dim] where rows = n_tok * n_head (or n_kv_head). */
__global__ static void gqa_head_rms_norm_weight_kernel(
        float *x,
        const float *w,
        uint32_t rows,
        uint32_t head_dim,
        float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    float *v = x + (uint64_t)row * head_dim;
    __shared__ float red[256];
    float acc = 0.0f;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        acc += v[d] * v[d];
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2u; s != 0; s >>= 1u) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / (float)head_dim + eps);
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        v[d] = v[d] * inv * (w ? w[d] : 1.0f);
    }
}

/* Rotate-half RoPE over the full head, in place.
 * x: [n_tok][n_head][head_dim], token t is at position pos0 + t. */
__global__ static void gqa_rope_neox_kernel(
        float *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t rot_dim, /* partial rotary: rotate first rot_dim dims */
        uint32_t pos0,
        float theta,
        const float *factors) { /* [rot_dim/2] freq divisors, NULL = none */
    uint32_t half = rot_dim / 2u;
    uint64_t total = (uint64_t)n_tok * n_head * half;
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    uint32_t d = (uint32_t)(i % half);
    uint32_t rem = (uint32_t)(i / half);
    uint32_t h = rem % n_head;
    uint32_t t = rem / n_head;
    float *v = x + ((uint64_t)t * n_head + h) * head_dim;
    float freq = powf(theta, -2.0f * (float)d / (float)rot_dim);
    if (factors) freq /= factors[d];
    float ang = (float)(pos0 + t) * freq;
    float c = cosf(ang), s = sinf(ang);
    float a = v[d], b = v[d + half];
    v[d] = a * c - b * s;
    v[d + half] = a * s + b * c;
}

/* Append n_tok tokens of K and V into the per-head cache.
 * kv_in: [n_tok][n_kv_head][head_dim]; cache: [n_kv_head][cap][head_dim]. */
__global__ static void gqa_kv_cache_append_kernel(
        float *cache,
        const float *kv_in,
        uint32_t n_tok,
        uint32_t n_kv_head,
        uint32_t head_dim,
        uint32_t cap,
        uint32_t pos0) {
    uint64_t total = (uint64_t)n_tok * n_kv_head * head_dim;
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    uint32_t d = (uint32_t)(i % head_dim);
    uint32_t rem = (uint32_t)(i / head_dim);
    uint32_t h = rem % n_kv_head;
    uint32_t t = rem / n_kv_head;
    cache[((uint64_t)h * cap + pos0 + t) * head_dim + d] =
        kv_in[((uint64_t)t * n_kv_head + h) * head_dim + d];
}

/* Causal GQA attention over the cache, single-pass online softmax.
 * q: [n_tok][n_head][head_dim] (post norm+rope); token t sees cache rows
 * [0, pos0 + t]. out: [n_tok][n_head][head_dim]. One block per (t, h). */
/* ---- fp8 (OCP e4m3fn) KV cache, opt-in via PULSAR_KV=fp8 ----------------
 * Storage: per (kv_head, position) row = head_dim e4m3 bytes + one f32
 * scale (amax/448) at the row tail, stride head_dim+4. Scales factor out
 * of both the qk dot and the V accumulation, so decode is one LUT-free
 * bit unpack per element plus one multiply per row. Pure storage format:
 * no fp8 hardware needed, runs on Pascal. Lossy - stays opt-in so the
 * bit-exact default path keeps its guarantees. */

__device__ __host__ __forceinline__ static float e4m3_dec_common(uint8_t b) {
    const uint32_t e = (b >> 3) & 0xFu;
    const uint32_t m = b & 7u;
    float v;
    if (e == 0) {
        v = (float)m * 0.001953125f; /* m * 2^-9 (subnormals) */
    } else {
        uint32_t bits = ((e + 120u) << 23) | (m << 20);
        float f;
        memcpy(&f, &bits, 4);
        v = f;
    }
    return (b & 0x80u) ? -v : v;
}

__device__ __forceinline__ static uint8_t e4m3_enc(float x) {
    uint8_t s = 0;
    if (x < 0.0f) { s = 0x80; x = -x; }
    if (!(x > 0.0f)) return s;             /* zero and NaN encode as 0 */
    if (x >= 448.0f) return s | 0x7E;      /* clamp to max finite 448 */
    if (x < 0.0009765625f) {               /* below 2^-10: subnormal/zero */
        int m = __float2int_rn(x * 512.0f);
        return m >= 8 ? (s | 0x08) : (s | (uint8_t)m);
    }
    uint32_t u; memcpy(&u, &x, 4);
    int e = (int)(u >> 23) - 127;
    int q = __float2int_rn(ldexpf(x, 3 - e)); /* mantissa in [8,16] */
    if (q == 16) { q = 8; e += 1; }
    if (e < -6) {                          /* denormal band 2^-10 .. 2^-6 */
        int m = __float2int_rn(ldexpf(x, 9));
        return m >= 8 ? (s | 0x08) : (s | (uint8_t)m);
    }
    if (e > 8) return s | 0x7E;
    return s | (uint8_t)(((e + 7) << 3) | (q - 8));
}

/* ---- KV storage format enum + per-format row geometry/dequant helpers ----
 * KVF selects the decode-time KV layout. Row stride + dequant all vary:
 * F32/F16 are packed and scale-free; FP8/INT8 carry one f32 scale at the
 * row tail (stride = head_dim {e4m3|i8} + 4); Q8_0/Q4_0 carry a per-32-elem
 * f16 d inside each block (stride = head_dim/32 * {34|18}). The attention
 * kernel is templated on KVF so the per-row layout is compile-time. */
enum class Kvf : uint32_t { F32 = 0, FP8 = 1, FP16 = 2, INT8 = 3, Q8_0 = 4, Q4_0 = 5 };

__device__ __forceinline__ static uint64_t kv_row_stride(Kvf kvf, uint32_t head_dim) {
    switch (kvf) {
        case Kvf::FP8:
        case Kvf::INT8: return (uint64_t)head_dim + 4u;
        case Kvf::FP16: return (uint64_t)head_dim * 2u;
        case Kvf::Q8_0: return ((uint64_t)head_dim / 32u) * 34u;
        case Kvf::Q4_0: return ((uint64_t)head_dim / 32u) * 18u;
        default:       return (uint64_t)head_dim * 4u; /* F32 */
    }
}

/* qk dot for a single key row: dot = sum_d q[d] * dequant_k(d), then the
 * per-row / per-block scale folds out (applied once by the caller via
 * row_scale_out). The scale-free packed formats (f32, fp16) return 1.0. */
template <Kvf KVF>
__device__ __forceinline__ static float gqa_kdot(
        const void *row_void, const float *sq, uint32_t head_dim, float *row_scale_out) {
    float dot = 0.0f;
    float rs = 1.0f;
    if (KVF == Kvf::F32) {
        const float *krow = (const float *)row_void;
        for (uint32_t d = 0; d < head_dim; d++) dot += sq[d] * krow[d];
    } else if (KVF == Kvf::FP16) {
        const __half *krow = (const __half *)row_void;
        for (uint32_t d = 0; d < head_dim; d++) dot += sq[d] * __half2float(krow[d]);
    } else if (KVF == Kvf::FP8) {
        const uint8_t *krow = (const uint8_t *)row_void;
        for (uint32_t d = 0; d < head_dim; d++) dot += sq[d] * e4m3_dec_common(krow[d]);
        memcpy(&rs, krow + head_dim, 4);
    } else if (KVF == Kvf::INT8) {
        const int8_t *krow = (const int8_t *)row_void;
        for (uint32_t d = 0; d < head_dim; d++) dot += sq[d] * (float)krow[d];
        memcpy(&rs, (const uint8_t *)krow + head_dim, 4);
    } else if (KVF == Kvf::Q8_0) {
        const char *blocks = (const char *)row_void;
        const uint32_t n_blocks = head_dim / 32u;
        for (uint32_t b = 0; b < n_blocks; b++) {
            const char *bb = blocks + (uint64_t)b * 34u;
            float d = __half2float(*(const __half *)bb);
            const int8_t *qs = (const int8_t *)(bb + 2u);
            for (uint32_t i = 0; i < 32u; i++) dot += sq[b * 32u + i] * d * (float)qs[i];
        }
    } else { /* Q4_0 */
        const char *blocks = (const char *)row_void;
        const uint32_t n_blocks = head_dim / 32u;
        for (uint32_t b = 0; b < n_blocks; b++) {
            const char *bb = blocks + (uint64_t)b * 18u;
            float d = __half2float(*(const __half *)bb);
            const uint8_t *qs = (const uint8_t *)(bb + 2u);
            for (uint32_t i = 0; i < 16u; i++) {
                uint8_t byte = qs[i];
                dot += sq[b * 32u + i]      * d * ((float)(byte & 0xFu) - 8.0f);
                dot += sq[b * 32u + i + 16u] * d * ((float)(byte >> 4)   - 8.0f);
            }
        }
    }
    *row_scale_out = rs;
    return dot;
}

/* V accumulation against a weighted softmax output for output dim `d`:
 * out_acc(d) += w * dequant_v(d). The fp8/int8 per-row scale folds into
 * one packed weight `wv = w * row_scale`; q8_0/q4_0 fold their per-block d
 * inside the call. d is the per-thread output lane. */
template <Kvf KVF>
__device__ __forceinline__ static float gqa_vacc_lane(
        const void *row_void, uint32_t d, uint32_t head_dim) {
    if (KVF == Kvf::F32)  return ((const float *)row_void)[d];
    if (KVF == Kvf::FP16) return __half2float(((const __half *)row_void)[d]);
    if (KVF == Kvf::FP8)  return e4m3_dec_common(((const uint8_t *)row_void)[d]);
    if (KVF == Kvf::INT8) return (float)((const int8_t *)row_void)[d];
    const char *blocks = (const char *)row_void;
    if (KVF == Kvf::Q8_0) {
        const uint32_t b = d >> 5, i = d & 31u;
        const char *bb = blocks + (uint64_t)b * 34u;
        float dd = __half2float(*(const __half *)bb);
        return dd * (float)((const int8_t *)(bb + 2u))[i];
    }
    /* Q4_0 — append packs byte j as {elem[j] low, elem[j+16] high} (split-half,
     * one thread owns two nibbles 16 apart); index by i&15 for the byte and
     * (i>>4)<<2 for the nibble shift. The old i>>1 / (i&1)<<2 here assumed
     * sequential pairs and scrambled V. */
    const uint32_t b = d >> 5, i = d & 31u, bytei = i & 15u, sh = ((i >> 4) & 1u) << 2;
    const char *bb = blocks + (uint64_t)b * 18u;
    float dd = __half2float(*(const __half *)bb);
    uint8_t q = (((const uint8_t *)(bb + 2u))[bytei] >> sh) & 0xFu;
    return dd * ((float)q - 8.0f);
}

/* per-row V scale for the formats that carry one (fp8/int8), else 1.0. */
template <Kvf KVF>
__device__ __forceinline__ static float gqa_vrow_scale(const void *row_void, uint32_t head_dim) {
    if (KVF == Kvf::FP8 || KVF == Kvf::INT8) {
        float s;
        memcpy(&s, (const uint8_t *)row_void + (KVF == Kvf::FP8 ? head_dim : head_dim), 4);
        return s;
    }
    return 1.0f;
}

template <Kvf KVF>
__global__ static void gqa_attention_kernel(
        float *out,
        const float *q,
        const void *k_cache,
        const void *v_cache,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t n_kv_head,
        uint32_t head_dim,   /* <= 512: each thread owns up to 4 dims */
        uint32_t cap,
        uint32_t pos0,
        float scale,         /* qk scale (1/sqrt(head_dim) for vanilla) */
        uint32_t window,     /* sliding window; 0 = full causal */
        const float *rel,    /* inkling rel-pos bias [n_tok][n_head][rel_extent], NULL = none */
        uint32_t rel_extent) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tok || h >= n_head) return;
    uint32_t kvh = h / (n_head / n_kv_head);
    uint32_t n_rows = pos0 + t + 1u;
    uint32_t row0 = (window != 0u && n_rows > window) ? n_rows - window : 0u;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    /* rel bias row for this (token, head): score(q at pos0+t, key j) +=
     * rel_row[(pos0+t) - j] when the distance is inside the extent; the
     * bias rides OUTSIDE the qk scale (reference folds 1/head_dim into q) */
    const float *rel_row = rel ? rel + ((uint64_t)t * n_head + h) * rel_extent : NULL;
    const uint64_t kv_stride = kv_row_stride(KVF, head_dim);
    const char *kbase = (const char *)k_cache + (uint64_t)kvh * cap * kv_stride;
    const char *vbase = (const char *)v_cache + (uint64_t)kvh * cap * kv_stride;
    /* Online softmax, single pass over K and V.  128 threads: thread d owns
     * output dim d; rows are processed in chunks of 128 with each thread
     * computing one row's score, then everyone accumulates V against the
     * shared per-chunk weights.  Scores are computed exactly once (the old
     * two-pass version recomputed them per output dim: ~130x the FLOPs and,
     * past L1, ~head_dim re-reads of K per row at long context). */
    const uint32_t tid = threadIdx.x;
    __shared__ float sq[512];
    __shared__ float s_w[128];
    __shared__ float red[128];
    for (uint32_t d = tid; d < head_dim; d += 128u) sq[d] = qh[d];
    __syncthreads();

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f}; /* V accum, dims tid+k*128 */
    float den = 0.0f;   /* running softmax denominator (same in all threads) */
    float m = -INFINITY;/* running max */
    for (uint32_t base = row0; base < n_rows; base += 128u) {
        uint32_t chunk = n_rows - base < 128u ? n_rows - base : 128u;
        /* one score per thread */
        float s = -INFINITY;
        if (tid < chunk) {
            const char *krow = kbase + (uint64_t)(base + tid) * kv_stride;
            float ks;
            float dot = gqa_kdot<KVF>(krow, sq, head_dim, &ks);
            s = dot * ks * scale;
            if (rel_row) {
                const uint32_t dist = (pos0 + t) - (base + tid);
                if (dist < rel_extent) s += rel_row[dist];
            }
        }
        red[tid] = s;
        __syncthreads();
        for (uint32_t st = 64u; st != 0; st >>= 1u) {
            if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
            __syncthreads();
        }
        float m_new = fmaxf(m, red[0]);
        float corr = expf(m - m_new); /* expf(-inf) == 0, first chunk safe */
        #pragma unroll
        for (uint32_t k = 0; k < 4; k++) acc[k] *= corr;
        den *= corr;
        if (tid < chunk) s_w[tid] = expf(s - m_new);
        __syncthreads();
        for (uint32_t r = 0; r < chunk; r++) {
            float w = s_w[r];
            den += w;
            const char *vrow = vbase + (uint64_t)(base + r) * kv_stride;
            float wv = w * gqa_vrow_scale<KVF>(vrow, head_dim);
            for (uint32_t d = tid, k = 0; d < head_dim; d += 128u, k++) {
                acc[k] += wv * gqa_vacc_lane<KVF>(vrow, d, head_dim);
            }
        }
        m = m_new;
        __syncthreads();
    }
    for (uint32_t d = tid, k = 0; d < head_dim; d += 128u, k++) {
        out[((uint64_t)t * n_head + h) * head_dim + d] = acc[k] / den;
    }
}

/* ---- split-K decode attention -------------------------------------------
 * The kernel above gives one block per (token, head): at decode a 16-head
 * model occupies 16 SMs regardless of context depth and the position scan
 * serializes (~5.6us/row measured on Qwen3.6 = ~7GB/s of a 448GB/s card).
 * For decode-shaped calls the scan splits across blockIdx.z: each split
 * runs the same online softmax over its slice of rows and writes an
 * UNNORMALIZED partial {acc[hd], m, den}; a combine kernel rescales onto
 * the global max and normalizes. Engaged only when n_tok <= 4, the
 * visible rows >= 4096 and there is no rel bias, so short-context calls
 * (and their bit-exact gates) take the original kernel unchanged. */
#define GQA_SPLIT_MIN_ROWS 4096u
#define GQA_SPLIT_MAX_TOK  4u
#define GQA_SPLIT_ROWS     2048u
#define GQA_SPLIT_MAX      64u

template <Kvf KVF>
__global__ static void gqa_attention_split_kernel(
        float *partial,      /* [n_tok][n_head][n_split][hd+2]: acc, m, den */
        const float *q,
        const void *k_cache,
        const void *v_cache,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t n_kv_head,
        uint32_t head_dim,
        uint32_t cap,
        uint32_t pos0,
        float scale,
        uint32_t window,
        uint32_t n_split) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    uint32_t sp = blockIdx.z;
    if (t >= n_tok || h >= n_head) return;
    uint32_t kvh = h / (n_head / n_kv_head);
    uint32_t n_rows = pos0 + t + 1u;
    uint32_t row0 = (window != 0u && n_rows > window) ? n_rows - window : 0u;
    uint32_t span = (n_rows - row0 + n_split - 1u) / n_split;
    uint32_t lo = row0 + sp * span;
    uint32_t hi = lo + span < n_rows ? lo + span : n_rows;
    float *dst = partial + (((uint64_t)t * n_head + h) * n_split + sp) * ((uint64_t)head_dim + 2u);
    const uint32_t tid = threadIdx.x;
    if (lo >= hi) {
        for (uint32_t d = tid; d < head_dim; d += 128u) dst[d] = 0.0f;
        if (tid == 0) { dst[head_dim] = -INFINITY; dst[head_dim + 1u] = 0.0f; }
        return;
    }
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    const uint64_t kv_stride = kv_row_stride(KVF, head_dim);
    const char *kbase = (const char *)k_cache + (uint64_t)kvh * cap * kv_stride;
    const char *vbase = (const char *)v_cache + (uint64_t)kvh * cap * kv_stride;
    __shared__ float sq[512];
    __shared__ float s_w[128];
    __shared__ float red[128];
    for (uint32_t d = tid; d < head_dim; d += 128u) sq[d] = qh[d];
    __syncthreads();

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float den = 0.0f;
    float m = -INFINITY;
    for (uint32_t base = lo; base < hi; base += 128u) {
        uint32_t chunk = hi - base < 128u ? hi - base : 128u;
        float s = -INFINITY;
        if (tid < chunk) {
            const char *krow = kbase + (uint64_t)(base + tid) * kv_stride;
            float ks;
            float dot = gqa_kdot<KVF>(krow, sq, head_dim, &ks);
            s = dot * ks * scale;
        }
        red[tid] = s;
        __syncthreads();
        for (uint32_t st = 64u; st != 0; st >>= 1u) {
            if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
            __syncthreads();
        }
        float m_new = fmaxf(m, red[0]);
        float corr = expf(m - m_new);
        #pragma unroll
        for (uint32_t k = 0; k < 4; k++) acc[k] *= corr;
        den *= corr;
        if (tid < chunk) s_w[tid] = expf(s - m_new);
        __syncthreads();
        for (uint32_t r = 0; r < chunk; r++) {
            float w = s_w[r];
            den += w;
            const char *vrow = vbase + (uint64_t)(base + r) * kv_stride;
            float wv = w * gqa_vrow_scale<KVF>(vrow, head_dim);
            for (uint32_t d = tid, k = 0; d < head_dim; d += 128u, k++) {
                acc[k] += wv * gqa_vacc_lane<KVF>(vrow, d, head_dim);
            }
        }
        m = m_new;
        __syncthreads();
    }
    for (uint32_t d = tid, k = 0; d < head_dim; d += 128u, k++) dst[d] = acc[k];
    if (tid == 0) { dst[head_dim] = m; dst[head_dim + 1u] = den; }
}

__global__ static void gqa_attention_combine_kernel(
        float *out,
        const float *partial,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_split) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tok || h >= n_head) return;
    const uint64_t stride = (uint64_t)head_dim + 2u;
    const float *base = partial + ((uint64_t)t * n_head + h) * n_split * stride;
    float m = -INFINITY;
    for (uint32_t s = 0; s < n_split; s++) m = fmaxf(m, base[s * stride + head_dim]);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float den = 0.0f;
    for (uint32_t s = 0; s < n_split; s++) {
        const float ms = base[s * stride + head_dim];
        if (ms == -INFINITY) continue; /* empty split */
        const float w = expf(ms - m);
        den += base[s * stride + head_dim + 1u] * w;
        for (uint32_t d = tid, k = 0; d < head_dim; d += 128u, k++)
            acc[k] += base[s * stride + d] * w;
    }
    for (uint32_t d = tid, k = 0; d < head_dim; d += 128u, k++)
        out[((uint64_t)t * n_head + h) * head_dim + d] = acc[k] / den;
}

/* grow-only per-device scratch for the split partials (same pattern as
 * preq_scratch: single-stream engine, VRAM only dereferenceable on its
 * own device) */
static void *g_gqa_split_scratch[16];
static uint64_t g_gqa_split_cap[16];

static void *gqa_split_scratch(uint64_t bytes) {
    int dev = 0;
    (void)cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) return NULL;
    if (bytes <= g_gqa_split_cap[dev]) return g_gqa_split_scratch[dev];
    if (g_gqa_split_scratch[dev]) (void)cudaFree(g_gqa_split_scratch[dev]);
    g_gqa_split_scratch[dev] = NULL;
    g_gqa_split_cap[dev] = 0;
    if (!cuda_ok(cudaMalloc(&g_gqa_split_scratch[dev], bytes), "gqa split scratch")) return NULL;
    g_gqa_split_cap[dev] = bytes;
    return g_gqa_split_scratch[dev];
}

template <Kvf KVF>
static int gqa_attention_dispatch(
        float *out, const float *q, const void *kc, const void *vc,
        uint32_t n_tok, uint32_t n_head, uint32_t n_kv_head,
        uint32_t head_dim, uint32_t cap, uint32_t pos0,
        float scale, uint32_t window,
        const float *rel, uint32_t rel_extent) {
    uint32_t rows_eff = pos0 + n_tok;
    if (window != 0u && rows_eff > window) rows_eff = window;
    if (!rel && n_tok <= GQA_SPLIT_MAX_TOK && rows_eff >= GQA_SPLIT_MIN_ROWS) {
        uint32_t n_split = (rows_eff + GQA_SPLIT_ROWS - 1u) / GQA_SPLIT_ROWS;
        if (n_split > GQA_SPLIT_MAX) n_split = GQA_SPLIT_MAX;
        if (n_split >= 2u) {
            uint64_t bytes = (uint64_t)n_tok * n_head * n_split * ((uint64_t)head_dim + 2u) * 4u;
            float *partial = (float *)gqa_split_scratch(bytes);
            if (partial) { /* alloc failure falls through to the plain kernel */
                dim3 g1(n_tok, n_head, n_split);
                gqa_attention_split_kernel<KVF><<<g1, 128>>>(
                        partial, q, kc, vc, n_tok, n_head, n_kv_head,
                        head_dim, cap, pos0, scale, window, n_split);
                if (!cuda_ok(cudaGetLastError(), "gqa split launch")) return 0;
                dim3 g2(n_tok, n_head, 1);
                gqa_attention_combine_kernel<<<g2, 128>>>(out, partial, n_tok, n_head, head_dim, n_split);
                return cuda_ok(cudaGetLastError(), "gqa combine launch");
            }
        }
    }
    dim3 grid(n_tok, n_head, 1);
    gqa_attention_kernel<KVF><<<grid, 128>>>(
            out, q, kc, vc, n_tok, n_head, n_kv_head, head_dim, cap, pos0,
            scale, window, rel, rel_extent);
    return cuda_ok(cudaGetLastError(), "gqa attention launch");
}

/* ---- host wrappers ---- */

extern "C" int ds4_gpu_gqa_head_rms_norm_weight(
        ds4_gpu_tensor *x, const ds4_gpu_tensor *w, /* w NULL = unit weight */
        uint32_t rows, uint32_t head_dim, float eps) {
    if (!x || rows == 0 || head_dim == 0 || head_dim > 512u) return 0;
    gqa_head_rms_norm_weight_kernel<<<rows, 256>>>(
            (float *)x->ptr, w ? (const float *)w->ptr : NULL, rows, head_dim, eps);
    return cuda_ok(cudaGetLastError(), "gqa head rms norm launch");
}

extern "C" int ds4_gpu_gqa_rope(
        ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head,
        uint32_t head_dim, uint32_t rot_dim, uint32_t pos0, float theta,
        const ds4_gpu_tensor *factors) {
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0 || rot_dim == 0 ||
        rot_dim > head_dim || rot_dim % 2u) return 0;
    uint64_t total = (uint64_t)n_tok * n_head * (rot_dim / 2u);
    gqa_rope_neox_kernel<<<(uint32_t)((total + 255u) / 256u), 256>>>(
            (float *)x->ptr, n_tok, n_head, head_dim, rot_dim, pos0, theta,
            factors ? (const float *)factors->ptr : NULL);
    return cuda_ok(cudaGetLastError(), "gqa rope launch");
}

extern "C" int ds4_gpu_gqa_kv_cache_append(
        ds4_gpu_tensor *cache, const ds4_gpu_tensor *kv_in,
        uint32_t n_tok, uint32_t n_kv_head, uint32_t head_dim,
        uint32_t cap, uint32_t pos0) {
    if (!cache || !kv_in || n_tok == 0 || pos0 + n_tok > cap) return 0;
    uint64_t total = (uint64_t)n_tok * n_kv_head * head_dim;
    gqa_kv_cache_append_kernel<<<(uint32_t)((total + 255u) / 256u), 256>>>(
            (float *)cache->ptr, (const float *)kv_in->ptr,
            n_tok, n_kv_head, head_dim, cap, pos0);
    return cuda_ok(cudaGetLastError(), "gqa kv append launch");
}

extern "C" int ds4_gpu_gqa_attention(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k_cache, const ds4_gpu_tensor *v_cache,
        uint32_t n_tok, uint32_t n_head, uint32_t n_kv_head,
        uint32_t head_dim, uint32_t cap, uint32_t pos0,
        float scale, uint32_t window,
        const ds4_gpu_tensor *rel, uint32_t rel_extent) {
    if (!out || !q || !k_cache || !v_cache ||
        n_head == 0 || n_kv_head == 0 || n_head % n_kv_head != 0 ||
        head_dim == 0 || head_dim > 512u || pos0 + n_tok > cap ||
        (rel && rel_extent == 0)) {
        return 0;
    }
    return gqa_attention_dispatch<Kvf::F32>(
            (float *)out->ptr, (const float *)q->ptr,
            k_cache->ptr, v_cache->ptr,
            n_tok, n_head, n_kv_head, head_dim, cap, pos0, scale, window,
            rel ? (const float *)rel->ptr : NULL, rel_extent);
}

/* ---- fp8 KV variants ---------------------------------------------------- */

__global__ static void gqa_kv_append_fp8_kernel(
        uint8_t *cache, const float *kv, uint32_t n_tok, uint32_t n_kv_head,
        uint32_t head_dim, uint32_t cap, uint32_t pos0) {
    const uint32_t t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    if (t >= n_tok || h >= n_kv_head) return;
    const float *src = kv + ((uint64_t)t * n_kv_head + h) * head_dim;
    uint8_t *dst = cache + ((uint64_t)h * cap + pos0 + t) * ((uint64_t)head_dim + 4u);
    __shared__ float red[128];
    float amax = 0.0f;
    for (uint32_t d = tid; d < head_dim; d += 128u) amax = fmaxf(amax, fabsf(src[d]));
    red[tid] = amax;
    __syncthreads();
    for (uint32_t st = 64u; st != 0; st >>= 1u) {
        if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
        __syncthreads();
    }
    const float rmax = red[0];
    const float scale = rmax / 448.0f;
    const float inv = rmax > 0.0f ? 448.0f / rmax : 0.0f;
    for (uint32_t d = tid; d < head_dim; d += 128u) dst[d] = e4m3_enc(src[d] * inv);
    if (tid == 0) memcpy(dst + head_dim, &scale, 4);
}

extern "C" int ds4_gpu_gqa_kv_cache_append_fp8(
        ds4_gpu_tensor *cache, const ds4_gpu_tensor *kv,
        uint32_t n_tok, uint32_t n_kv_head, uint32_t head_dim,
        uint32_t cap, uint32_t pos0) {
    if (!cache || !kv || n_tok == 0 || n_kv_head == 0 || head_dim == 0 ||
        head_dim % 4u != 0 || pos0 + n_tok > cap) {
        return 0;
    }
    dim3 grid(n_tok, n_kv_head, 1);
    gqa_kv_append_fp8_kernel<<<grid, 128>>>(
            (uint8_t *)cache->ptr, (const float *)kv->ptr,
            n_tok, n_kv_head, head_dim, cap, pos0);
    return cuda_ok(cudaGetLastError(), "gqa kv append fp8 launch");
}

/* ---- fp16 (IEEE half) KV cache, PULSAR_KV=fp16 -------------------------
 * Storage: head_dim __half per row, stride head_dim*2. No per-row scale:
 * fp16 range (±65504) covers K/V post-rope amplitudes losslessly enough
 * for attention, and rounding is the only loss. ~2x smaller than f32. */
__global__ static void gqa_kv_append_fp16_kernel(
        __half *cache, const float *kv, uint32_t n_tok, uint32_t n_kv_head,
        uint32_t head_dim, uint32_t cap, uint32_t pos0) {
    const uint32_t t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    if (t >= n_tok || h >= n_kv_head) return;
    const float *src = kv + ((uint64_t)t * n_kv_head + h) * head_dim;
    __half *dst = cache + ((uint64_t)h * cap + pos0 + t) * head_dim;
    for (uint32_t d = tid; d < head_dim; d += 128u) dst[d] = __float2half(src[d]);
}

extern "C" int ds4_gpu_gqa_kv_cache_append_fp16(
        ds4_gpu_tensor *cache, const ds4_gpu_tensor *kv,
        uint32_t n_tok, uint32_t n_kv_head, uint32_t head_dim,
        uint32_t cap, uint32_t pos0) {
    if (!cache || !kv || n_tok == 0 || n_kv_head == 0 || head_dim == 0 ||
        head_dim % 2u != 0 || pos0 + n_tok > cap) {
        return 0;
    }
    dim3 grid(n_tok, n_kv_head, 1);
    gqa_kv_append_fp16_kernel<<<grid, 128>>>(
            (__half *)cache->ptr, (const float *)kv->ptr,
            n_tok, n_kv_head, head_dim, cap, pos0);
    return cuda_ok(cudaGetLastError(), "gqa kv append fp16 launch");
}

/* ---- int8 + per-row scale KV cache, PULSAR_KV=int8 ----------------------
 * Same layout/story as fp8 but signed int8 symmetric: row = head_dim
 * int8 + one f32 scale (amax/127) at the row tail, stride head_dim+4.
 * ~4x smaller than f32; per-row scale keeps the resolution the bit-exact
 * f32 path has via dot rescaling. */
__global__ static void gqa_kv_append_int8_kernel(
        int8_t *cache, const float *kv, uint32_t n_tok, uint32_t n_kv_head,
        uint32_t head_dim, uint32_t cap, uint32_t pos0) {
    const uint32_t t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    if (t >= n_tok || h >= n_kv_head) return;
    const float *src = kv + ((uint64_t)t * n_kv_head + h) * head_dim;
    int8_t *dst = cache + ((uint64_t)h * cap + pos0 + t) * ((uint64_t)head_dim + 4u);
    __shared__ float red[128];
    float amax = 0.0f;
    for (uint32_t d = tid; d < head_dim; d += 128u) amax = fmaxf(amax, fabsf(src[d]));
    red[tid] = amax;
    __syncthreads();
    for (uint32_t st = 64u; st != 0; st >>= 1u) {
        if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
        __syncthreads();
    }
    const float rmax = red[0];
    const float scale = rmax / 127.0f;
    const float inv = rmax > 0.0f ? 127.0f / rmax : 0.0f;
    for (uint32_t d = tid; d < head_dim; d += 128u) {
        float q = src[d] * inv;
        int qi = __float2int_rn(q);
        if (qi > 127) qi = 127;
        if (qi < -128) qi = -128;
        dst[d] = (int8_t)qi;
    }
    if (tid == 0) memcpy(dst + head_dim, &scale, 4);
}

extern "C" int ds4_gpu_gqa_kv_cache_append_int8(
        ds4_gpu_tensor *cache, const ds4_gpu_tensor *kv,
        uint32_t n_tok, uint32_t n_kv_head, uint32_t head_dim,
        uint32_t cap, uint32_t pos0) {
    if (!cache || !kv || n_tok == 0 || n_kv_head == 0 || head_dim == 0 ||
        head_dim % 4u != 0 || pos0 + n_tok > cap) {
        return 0;
    }
    dim3 grid(n_tok, n_kv_head, 1);
    gqa_kv_append_int8_kernel<<<grid, 128>>>(
            (int8_t *)cache->ptr, (const float *)kv->ptr,
            n_tok, n_kv_head, head_dim, cap, pos0);
    return cuda_ok(cudaGetLastError(), "gqa kv append int8 launch");
}

/* ---- q8_0 KV cache, PULSAR_KV=q8_0 -------------------------------------
 * 32-wide blocks: f16 d (2B) + 32 int8 (32B) = 34B/block. Row stride
 * head_dim/32*34. d = block amax/127; value = d * q[i]. Matches the ggml
 * block layout used elsewhere in pulsar (mla q8 dot). One CUDA block per
 * (t,h) row, sequential over the (<=16) 32-wide blocks; 32 threads/block
 * do the amax reduce then quantize. */
__global__ static void gqa_kv_append_q8_0_kernel(
        char *cache, const float *kv, uint32_t n_tok, uint32_t n_kv_head,
        uint32_t head_dim, uint32_t cap, uint32_t pos0) {
    const uint32_t t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    if (t >= n_tok || h >= n_kv_head || tid >= 32u) return;
    const uint32_t n_blocks = head_dim / 32u;
    const float *src = kv + ((uint64_t)t * n_kv_head + h) * head_dim;
    char *dst = cache + ((uint64_t)h * cap + pos0 + t) * ((uint64_t)n_blocks * 34u);
    __shared__ float red[32];
    for (uint32_t b = 0; b < n_blocks; b++) {
        const float *blk = src + b * 32u;
        char *bb = dst + (uint64_t)b * 34u;
        float amax = fabsf(blk[tid]);
        red[tid] = amax;
        __syncthreads();
        for (uint32_t st = 16u; st != 0; st >>= 1u)
            if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
        __syncthreads();
        const float rmax = red[0];
        const float d = rmax / 127.0f;
        const float inv = rmax > 0.0f ? 127.0f / rmax : 0.0f;
        if (tid == 0) *(__half *)bb = __float2half(d);
        float q = blk[tid] * inv;
        int qi = __float2int_rn(q);
        if (qi > 127) qi = 127;
        if (qi < -128) qi = -128;
        bb[2u + tid] = (int8_t)qi;
    }
}

extern "C" int ds4_gpu_gqa_kv_cache_append_q8_0(
        ds4_gpu_tensor *cache, const ds4_gpu_tensor *kv,
        uint32_t n_tok, uint32_t n_kv_head, uint32_t head_dim,
        uint32_t cap, uint32_t pos0) {
    if (!cache || !kv || n_tok == 0 || n_kv_head == 0 || head_dim == 0 ||
        head_dim % 32u != 0 || pos0 + n_tok > cap) {
        return 0;
    }
    dim3 grid(n_tok, n_kv_head, 1);
    gqa_kv_append_q8_0_kernel<<<grid, 32>>>(
            (char *)cache->ptr, (const float *)kv->ptr,
            n_tok, n_kv_head, head_dim, cap, pos0);
    return cuda_ok(cudaGetLastError(), "gqa kv append q8_0 launch");
}

/* ---- q4_0 KV cache, PULSAR_KV=q4_0 -------------------------------------
 * 32-wide blocks: f16 d (2B) + 16 bytes (4-bit quants, two elems/byte) =
 * 18B/block. Row stride head_dim/32*18. value = d * (q - 8); q in [0,15].
 * The tightest format here; inner-loop dequant unpacks two nibbles/byte.
 * 32 threads/block: first 16 own a byte each (two low/high nibbles). */
__global__ static void gqa_kv_append_q4_0_kernel(
        char *cache, const float *kv, uint32_t n_tok, uint32_t n_kv_head,
        uint32_t head_dim, uint32_t cap, uint32_t pos0) {
    const uint32_t t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    if (t >= n_tok || h >= n_kv_head || tid >= 32u) return;
    const uint32_t n_blocks = head_dim / 32u;
    const float *src = kv + ((uint64_t)t * n_kv_head + h) * head_dim;
    char *dst = cache + ((uint64_t)h * cap + pos0 + t) * ((uint64_t)n_blocks * 18u);
    __shared__ float red[32];
    for (uint32_t b = 0; b < n_blocks; b++) {
        const float *blk = src + b * 32u;
        char *bb = dst + (uint64_t)b * 18u;
        float amax = fabsf(blk[tid]);
        red[tid] = amax;
        __syncthreads();
        for (uint32_t st = 16u; st != 0; st >>= 1u)
            if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
        __syncthreads();
        const float rmax = red[0];
        /* q4_0: value = d*(q-8), q in [0,15]; symmetric around 0 via -8.
         * Map amax to the 8..15 swing (8.5 keeps midpoint resolution). */
        const float d = rmax / 8.5f;
        const float inv = rmax > 0.0f ? 8.5f / rmax : 0.0f;
        if (tid == 0) *(__half *)bb = __float2half(d);
        if (tid < 16u) {
            int q0 = __float2int_rn(blk[tid] * inv) + 8;
            int q1 = __float2int_rn(blk[tid + 16u] * inv) + 8;
            if (q0 > 15) q0 = 15; if (q0 < 0) q0 = 0;
            if (q1 > 15) q1 = 15; if (q1 < 0) q1 = 0;
            bb[2u + tid] = (uint8_t)(q0 | (q1 << 4));
        }
    }
}

extern "C" int ds4_gpu_gqa_kv_cache_append_q4_0(
        ds4_gpu_tensor *cache, const ds4_gpu_tensor *kv,
        uint32_t n_tok, uint32_t n_kv_head, uint32_t head_dim,
        uint32_t cap, uint32_t pos0) {
    if (!cache || !kv || n_tok == 0 || n_kv_head == 0 || head_dim == 0 ||
        head_dim % 32u != 0 || pos0 + n_tok > cap) {
        return 0;
    }
    dim3 grid(n_tok, n_kv_head, 1);
    gqa_kv_append_q4_0_kernel<<<grid, 32>>>(
            (char *)cache->ptr, (const float *)kv->ptr,
            n_tok, n_kv_head, head_dim, cap, pos0);
    return cuda_ok(cudaGetLastError(), "gqa kv append q4_0 launch");
}

extern "C" int ds4_gpu_gqa_attention_fp8(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k_cache, const ds4_gpu_tensor *v_cache,
        uint32_t n_tok, uint32_t n_head, uint32_t n_kv_head,
        uint32_t head_dim, uint32_t cap, uint32_t pos0,
        float scale, uint32_t window,
        const ds4_gpu_tensor *rel, uint32_t rel_extent) {
    if (!out || !q || !k_cache || !v_cache ||
        n_head == 0 || n_kv_head == 0 || n_head % n_kv_head != 0 ||
        head_dim == 0 || head_dim > 512u || head_dim % 4u != 0 ||
        pos0 + n_tok > cap || (rel && rel_extent == 0)) {
        return 0;
    }
    return gqa_attention_dispatch<Kvf::FP8>(
            (float *)out->ptr, (const float *)q->ptr,
            k_cache->ptr, v_cache->ptr,
            n_tok, n_head, n_kv_head, head_dim, cap, pos0, scale, window,
            rel ? (const float *)rel->ptr : NULL, rel_extent);
}

extern "C" int ds4_gpu_gqa_attention_fp16(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k_cache, const ds4_gpu_tensor *v_cache,
        uint32_t n_tok, uint32_t n_head, uint32_t n_kv_head,
        uint32_t head_dim, uint32_t cap, uint32_t pos0,
        float scale, uint32_t window,
        const ds4_gpu_tensor *rel, uint32_t rel_extent) {
    if (!out || !q || !k_cache || !v_cache ||
        n_head == 0 || n_kv_head == 0 || n_head % n_kv_head != 0 ||
        head_dim == 0 || head_dim > 512u || head_dim % 2u != 0 ||
        pos0 + n_tok > cap || (rel && rel_extent == 0)) {
        return 0;
    }
    return gqa_attention_dispatch<Kvf::FP16>(
            (float *)out->ptr, (const float *)q->ptr,
            k_cache->ptr, v_cache->ptr,
            n_tok, n_head, n_kv_head, head_dim, cap, pos0, scale, window,
            rel ? (const float *)rel->ptr : NULL, rel_extent);
}

extern "C" int ds4_gpu_gqa_attention_int8(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k_cache, const ds4_gpu_tensor *v_cache,
        uint32_t n_tok, uint32_t n_head, uint32_t n_kv_head,
        uint32_t head_dim, uint32_t cap, uint32_t pos0,
        float scale, uint32_t window,
        const ds4_gpu_tensor *rel, uint32_t rel_extent) {
    if (!out || !q || !k_cache || !v_cache ||
        n_head == 0 || n_kv_head == 0 || n_head % n_kv_head != 0 ||
        head_dim == 0 || head_dim > 512u || head_dim % 4u != 0 ||
        pos0 + n_tok > cap || (rel && rel_extent == 0)) {
        return 0;
    }
    return gqa_attention_dispatch<Kvf::INT8>(
            (float *)out->ptr, (const float *)q->ptr,
            k_cache->ptr, v_cache->ptr,
            n_tok, n_head, n_kv_head, head_dim, cap, pos0, scale, window,
            rel ? (const float *)rel->ptr : NULL, rel_extent);
}

extern "C" int ds4_gpu_gqa_attention_q8_0(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k_cache, const ds4_gpu_tensor *v_cache,
        uint32_t n_tok, uint32_t n_head, uint32_t n_kv_head,
        uint32_t head_dim, uint32_t cap, uint32_t pos0,
        float scale, uint32_t window,
        const ds4_gpu_tensor *rel, uint32_t rel_extent) {
    if (!out || !q || !k_cache || !v_cache ||
        n_head == 0 || n_kv_head == 0 || n_head % n_kv_head != 0 ||
        head_dim == 0 || head_dim > 512u || head_dim % 32u != 0 ||
        pos0 + n_tok > cap || (rel && rel_extent == 0)) {
        return 0;
    }
    return gqa_attention_dispatch<Kvf::Q8_0>(
            (float *)out->ptr, (const float *)q->ptr,
            k_cache->ptr, v_cache->ptr,
            n_tok, n_head, n_kv_head, head_dim, cap, pos0, scale, window,
            rel ? (const float *)rel->ptr : NULL, rel_extent);
}

extern "C" int ds4_gpu_gqa_attention_q4_0(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k_cache, const ds4_gpu_tensor *v_cache,
        uint32_t n_tok, uint32_t n_head, uint32_t n_kv_head,
        uint32_t head_dim, uint32_t cap, uint32_t pos0,
        float scale, uint32_t window,
        const ds4_gpu_tensor *rel, uint32_t rel_extent) {
    if (!out || !q || !k_cache || !v_cache ||
        n_head == 0 || n_kv_head == 0 || n_head % n_kv_head != 0 ||
        head_dim == 0 || head_dim > 512u || head_dim % 32u != 0 ||
        pos0 + n_tok > cap || (rel && rel_extent == 0)) {
        return 0;
    }
    return gqa_attention_dispatch<Kvf::Q4_0>(
            (float *)out->ptr, (const float *)q->ptr,
            k_cache->ptr, v_cache->ptr,
            n_tok, n_head, n_kv_head, head_dim, cap, pos0, scale, window,
            rel ? (const float *)rel->ptr : NULL, rel_extent);
}

/* ---- CPU reference + self-test ---- */

static void gqa_ref_head_rms_norm(float *x, const float *w,
                                  uint32_t rows, uint32_t hd, float eps) {
    for (uint32_t r = 0; r < rows; r++) {
        float *v = x + (uint64_t)r * hd;
        double ss = 0.0;
        for (uint32_t d = 0; d < hd; d++) ss += (double)v[d] * v[d];
        float inv = 1.0f / sqrtf((float)(ss / hd) + eps);
        for (uint32_t d = 0; d < hd; d++) v[d] = v[d] * inv * w[d];
    }
}

static void gqa_ref_rope(float *x, uint32_t n_tok, uint32_t n_head,
                         uint32_t hd, uint32_t pos0, float theta) {
    uint32_t half = hd / 2u;
    for (uint32_t t = 0; t < n_tok; t++)
    for (uint32_t h = 0; h < n_head; h++) {
        float *v = x + ((uint64_t)t * n_head + h) * hd;
        for (uint32_t d = 0; d < half; d++) {
            float freq = powf(theta, -2.0f * (float)d / (float)hd);
            float ang = (float)(pos0 + t) * freq;
            float c = cosf(ang), s = sinf(ang);
            float a = v[d], b = v[d + half];
            v[d] = a * c - b * s;
            v[d + half] = a * s + b * c;
        }
    }
}

static void gqa_ref_attention(float *out, const float *q,
                              const float *kc, const float *vc,
                              uint32_t n_tok, uint32_t n_head,
                              uint32_t n_kv_head, uint32_t hd,
                              uint32_t cap, uint32_t pos0,
                              float scale, uint32_t window,
                              const float *rel, uint32_t rel_ext) {
    uint32_t group = n_head / n_kv_head;
    float *scores = (float *)malloc((size_t)(pos0 + n_tok) * sizeof(float));
    for (uint32_t t = 0; t < n_tok; t++)
    for (uint32_t h = 0; h < n_head; h++) {
        uint32_t kvh = h / group;
        uint32_t n_rows = pos0 + t + 1u;
        uint32_t row0 = (window != 0u && n_rows > window) ? n_rows - window : 0u;
        const float *qh = q + ((uint64_t)t * n_head + h) * hd;
        const float *kh = kc + (uint64_t)kvh * cap * hd;
        const float *vh = vc + (uint64_t)kvh * cap * hd;
        const float *rr = rel ? rel + ((uint64_t)t * n_head + h) * rel_ext : NULL;
        float m = -INFINITY;
        for (uint32_t r = row0; r < n_rows; r++) {
            float dot = 0.0f;
            for (uint32_t d = 0; d < hd; d++) dot += qh[d] * kh[(uint64_t)r * hd + d];
            scores[r] = dot * scale;
            if (rr && (pos0 + t) - r < rel_ext) scores[r] += rr[(pos0 + t) - r];
            if (scores[r] > m) m = scores[r];
        }
        float den = 0.0f;
        for (uint32_t r = row0; r < n_rows; r++) den += expf(scores[r] - m);
        float *oh = out + ((uint64_t)t * n_head + h) * hd;
        for (uint32_t d = 0; d < hd; d++) {
            float acc = 0.0f;
            for (uint32_t r = row0; r < n_rows; r++) {
                acc += expf(scores[r] - m) / den * vh[(uint64_t)r * hd + d];
            }
            oh[d] = acc;
        }
    }
    free(scores);
}

static float gqa_max_abs_diff(const float *a, const float *b, uint64_t n) {
    float m = 0.0f;
    for (uint64_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static uint32_t gqa_test_rng_state = 0x1234567u;
static float gqa_test_randf(void) {
    gqa_test_rng_state = gqa_test_rng_state * 1664525u + 1013904223u;
    return ((float)(gqa_test_rng_state >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
}

/* Full-pipeline self-test on random data: qk-norm -> rope -> cache append
 * -> causal GQA attention, CPU reference vs kernels. Hy3 dims: 64/8 heads,
 * head_dim 128. Covers a 13-token prefill at pos0=0 and a 1-token decode at
 * pos0=13 against the populated cache. Returns 1 on pass. */
extern "C" int ds4_gpu_gqa_selftest(void) {
    const uint32_t n_head = 64, n_kv_head = 8, hd = 128, cap = 256;
    const float eps = 1e-5f, theta = 11158840.0f;
    /* {n_tok, pos0}: short prefill, decode, then a 200-token prefill and a
     * decode at row 215 to exercise the multi-chunk online-softmax path
     * (rows > 128 crosses a chunk boundary and rescales acc/den). */
    const uint32_t phases[4][2] = { {13u, 0u}, {1u, 13u}, {200u, 14u}, {1u, 214u} };
    int all_ok = 1;

    float *qn_w = (float *)malloc(hd * sizeof(float));
    float *kn_w = (float *)malloc(hd * sizeof(float));
    for (uint32_t d = 0; d < hd; d++) { qn_w[d] = 0.5f + 0.5f * fabsf(gqa_test_randf()); kn_w[d] = 0.5f + 0.5f * fabsf(gqa_test_randf()); }

    uint64_t kcache_elems = (uint64_t)n_kv_head * cap * hd;
    float *ref_kc = (float *)calloc(kcache_elems, sizeof(float));
    float *ref_vc = (float *)calloc(kcache_elems, sizeof(float));

    ds4_gpu_tensor *g_qnw = ds4_gpu_tensor_alloc(hd * sizeof(float));
    ds4_gpu_tensor *g_knw = ds4_gpu_tensor_alloc(hd * sizeof(float));
    ds4_gpu_tensor *g_kc = ds4_gpu_tensor_alloc(kcache_elems * sizeof(float));
    ds4_gpu_tensor *g_vc = ds4_gpu_tensor_alloc(kcache_elems * sizeof(float));
    if (!g_qnw || !g_knw || !g_kc || !g_vc) { fprintf(stderr, "gqa-selftest: alloc failed\n"); return 0; }
    ds4_gpu_tensor_write(g_qnw, 0, qn_w, hd * sizeof(float));
    ds4_gpu_tensor_write(g_knw, 0, kn_w, hd * sizeof(float));
    { float *z = (float *)calloc(kcache_elems, sizeof(float));
      ds4_gpu_tensor_write(g_kc, 0, z, kcache_elems * sizeof(float));
      ds4_gpu_tensor_write(g_vc, 0, z, kcache_elems * sizeof(float));
      free(z); }

    for (int phase = 0; phase < 4; phase++) {
        uint32_t n_tok = phases[phase][0], pos0 = phases[phase][1];
        uint64_t q_elems = (uint64_t)n_tok * n_head * hd;
        uint64_t kv_elems = (uint64_t)n_tok * n_kv_head * hd;
        float *q = (float *)malloc(q_elems * sizeof(float));
        float *k = (float *)malloc(kv_elems * sizeof(float));
        float *v = (float *)malloc(kv_elems * sizeof(float));
        for (uint64_t i = 0; i < q_elems; i++) q[i] = gqa_test_randf();
        for (uint64_t i = 0; i < kv_elems; i++) { k[i] = gqa_test_randf(); v[i] = gqa_test_randf(); }

        ds4_gpu_tensor *g_q = ds4_gpu_tensor_alloc(q_elems * sizeof(float));
        ds4_gpu_tensor *g_k = ds4_gpu_tensor_alloc(kv_elems * sizeof(float));
        ds4_gpu_tensor *g_v = ds4_gpu_tensor_alloc(kv_elems * sizeof(float));
        ds4_gpu_tensor *g_out = ds4_gpu_tensor_alloc(q_elems * sizeof(float));
        ds4_gpu_tensor_write(g_q, 0, q, q_elems * sizeof(float));
        ds4_gpu_tensor_write(g_k, 0, k, kv_elems * sizeof(float));
        ds4_gpu_tensor_write(g_v, 0, v, kv_elems * sizeof(float));

        /* reference pipeline */
        gqa_ref_head_rms_norm(q, qn_w, n_tok * n_head, hd, eps);
        gqa_ref_head_rms_norm(k, kn_w, n_tok * n_kv_head, hd, eps);
        gqa_ref_rope(q, n_tok, n_head, hd, pos0, theta);
        gqa_ref_rope(k, n_tok, n_kv_head, hd, pos0, theta);
        for (uint32_t t = 0; t < n_tok; t++)
        for (uint32_t h = 0; h < n_kv_head; h++)
        for (uint32_t d = 0; d < hd; d++) {
            ref_kc[((uint64_t)h * cap + pos0 + t) * hd + d] = k[((uint64_t)t * n_kv_head + h) * hd + d];
            ref_vc[((uint64_t)h * cap + pos0 + t) * hd + d] = v[((uint64_t)t * n_kv_head + h) * hd + d];
        }
        float *ref_out = (float *)malloc(q_elems * sizeof(float));
        gqa_ref_attention(ref_out, q, ref_kc, ref_vc, n_tok, n_head, n_kv_head, hd, cap, pos0,
                          1.0f / sqrtf((float)hd), 0u, NULL, 0u);

        /* kernel pipeline */
        int ok = ds4_gpu_gqa_head_rms_norm_weight(g_q, g_qnw, n_tok * n_head, hd, eps) &&
                 ds4_gpu_gqa_head_rms_norm_weight(g_k, g_knw, n_tok * n_kv_head, hd, eps) &&
                 ds4_gpu_gqa_rope(g_q, n_tok, n_head, hd, hd, pos0, theta, NULL) &&
                 ds4_gpu_gqa_rope(g_k, n_tok, n_kv_head, hd, hd, pos0, theta, NULL) &&
                 ds4_gpu_gqa_kv_cache_append(g_kc, g_k, n_tok, n_kv_head, hd, cap, pos0) &&
                 ds4_gpu_gqa_kv_cache_append(g_vc, g_v, n_tok, n_kv_head, hd, cap, pos0) &&
                 ds4_gpu_gqa_attention(g_out, g_q, g_kc, g_vc, n_tok, n_head, n_kv_head, hd, cap, pos0,
                                       1.0f / sqrtf((float)hd), 0u, NULL, 0u) &&
                 cuda_ok(cudaDeviceSynchronize(), "gqa selftest sync");
        float *gpu_out = (float *)malloc(q_elems * sizeof(float));
        if (ok) ok = ds4_gpu_tensor_read(g_out, 0, gpu_out, q_elems * sizeof(float)) != 0;
        float diff = ok ? gqa_max_abs_diff(ref_out, gpu_out, q_elems) : INFINITY;
        int pass = ok && diff < 2e-4f;
        fprintf(stderr, "gqa-selftest phase %d (n_tok=%u pos0=%u): %s (max diff %.2e)\n",
                phase, n_tok, pos0, pass ? "PASS" : "FAIL", diff);
        if (!pass) all_ok = 0;

        free(q); free(k); free(v); free(ref_out); free(gpu_out);
        ds4_gpu_tensor_free(g_q); ds4_gpu_tensor_free(g_k);
        ds4_gpu_tensor_free(g_v); ds4_gpu_tensor_free(g_out);
    }

    /* Phase 5 (inkling): rel-pos bias + sliding window over the populated
     * 215-row cache. rel_ext 64 < visible rows, so the out-of-band zero
     * path and the window clamp both execute; muP 1/head_dim scale. */
    {
        const uint32_t n_tok = 3, pos0 = 215, rel_ext = 64, window = 96;
        const float scale = 1.0f / (float)hd;
        uint64_t q_elems = (uint64_t)n_tok * n_head * hd;
        uint64_t kv_elems = (uint64_t)n_tok * n_kv_head * hd;
        uint64_t rel_elems = (uint64_t)n_tok * n_head * rel_ext;
        float *q = (float *)malloc(q_elems * sizeof(float));
        float *k = (float *)malloc(kv_elems * sizeof(float));
        float *v = (float *)malloc(kv_elems * sizeof(float));
        float *rel = (float *)malloc(rel_elems * sizeof(float));
        for (uint64_t i = 0; i < q_elems; i++) q[i] = gqa_test_randf();
        for (uint64_t i = 0; i < kv_elems; i++) { k[i] = gqa_test_randf(); v[i] = gqa_test_randf(); }
        for (uint64_t i = 0; i < rel_elems; i++) rel[i] = gqa_test_randf() * 2.0f;

        ds4_gpu_tensor *g_q = ds4_gpu_tensor_alloc(q_elems * sizeof(float));
        ds4_gpu_tensor *g_k = ds4_gpu_tensor_alloc(kv_elems * sizeof(float));
        ds4_gpu_tensor *g_v = ds4_gpu_tensor_alloc(kv_elems * sizeof(float));
        ds4_gpu_tensor *g_rel = ds4_gpu_tensor_alloc(rel_elems * sizeof(float));
        ds4_gpu_tensor *g_out = ds4_gpu_tensor_alloc(q_elems * sizeof(float));
        ds4_gpu_tensor_write(g_q, 0, q, q_elems * sizeof(float));
        ds4_gpu_tensor_write(g_k, 0, k, kv_elems * sizeof(float));
        ds4_gpu_tensor_write(g_v, 0, v, kv_elems * sizeof(float));
        ds4_gpu_tensor_write(g_rel, 0, rel, rel_elems * sizeof(float));

        /* no norm/rope: inkling feeds pre-normed, unrotated q/k */
        for (uint32_t t = 0; t < n_tok; t++)
        for (uint32_t h = 0; h < n_kv_head; h++)
        for (uint32_t d = 0; d < hd; d++) {
            ref_kc[((uint64_t)h * cap + pos0 + t) * hd + d] = k[((uint64_t)t * n_kv_head + h) * hd + d];
            ref_vc[((uint64_t)h * cap + pos0 + t) * hd + d] = v[((uint64_t)t * n_kv_head + h) * hd + d];
        }
        float *ref_out = (float *)malloc(q_elems * sizeof(float));
        gqa_ref_attention(ref_out, q, ref_kc, ref_vc, n_tok, n_head, n_kv_head, hd, cap, pos0,
                          scale, window, rel, rel_ext);

        int ok = ds4_gpu_gqa_kv_cache_append(g_kc, g_k, n_tok, n_kv_head, hd, cap, pos0) &&
                 ds4_gpu_gqa_kv_cache_append(g_vc, g_v, n_tok, n_kv_head, hd, cap, pos0) &&
                 ds4_gpu_gqa_attention(g_out, g_q, g_kc, g_vc, n_tok, n_head, n_kv_head, hd, cap, pos0,
                                       scale, window, g_rel, rel_ext) &&
                 cuda_ok(cudaDeviceSynchronize(), "gqa rel selftest sync");
        float *gpu_out = (float *)malloc(q_elems * sizeof(float));
        if (ok) ok = ds4_gpu_tensor_read(g_out, 0, gpu_out, q_elems * sizeof(float)) != 0;
        float diff = ok ? gqa_max_abs_diff(ref_out, gpu_out, q_elems) : INFINITY;
        int pass = ok && diff < 2e-4f;
        fprintf(stderr, "gqa-selftest phase 5 rel+window (n_tok=%u pos0=%u ext=%u win=%u): %s (max diff %.2e)\n",
                n_tok, pos0, rel_ext, window, pass ? "PASS" : "FAIL", diff);
        if (!pass) all_ok = 0;

        free(q); free(k); free(v); free(rel); free(ref_out); free(gpu_out);
        ds4_gpu_tensor_free(g_q); ds4_gpu_tensor_free(g_k); ds4_gpu_tensor_free(g_v);
        ds4_gpu_tensor_free(g_rel); ds4_gpu_tensor_free(g_out);
    }

    /* Phase 6 (fp8 KV): append+attention on the e4m3 cache. The GPU cache
     * bytes are read back and host-decoded into f32 caches, then the f32
     * reference runs on the DECODED values - so the attention math must
     * match to float tolerance, and the encoder is separately sanity
     * checked (round-trip relative error < 7%, e4m3's mantissa step). */
    {
        const uint32_t n_tok = 48, pos0 = 9, window = 0;
        const float scale = 1.0f / sqrtf((float)hd);
        const uint64_t stride = (uint64_t)hd + 4u;
        const uint64_t cache8 = (uint64_t)n_kv_head * cap * stride;
        uint64_t q_elems = (uint64_t)n_tok * n_head * hd;
        uint64_t kv_elems = (uint64_t)n_tok * n_kv_head * hd;
        float *q = (float *)malloc(q_elems * sizeof(float));
        float *k = (float *)malloc(kv_elems * sizeof(float));
        float *v = (float *)malloc(kv_elems * sizeof(float));
        for (uint64_t i = 0; i < q_elems; i++) q[i] = gqa_test_randf();
        for (uint64_t i = 0; i < kv_elems; i++) { k[i] = gqa_test_randf() * 3.0f; v[i] = gqa_test_randf() * 3.0f; }

        ds4_gpu_tensor *g_q = ds4_gpu_tensor_alloc(q_elems * sizeof(float));
        ds4_gpu_tensor *g_k = ds4_gpu_tensor_alloc(kv_elems * sizeof(float));
        ds4_gpu_tensor *g_v = ds4_gpu_tensor_alloc(kv_elems * sizeof(float));
        ds4_gpu_tensor *g_out = ds4_gpu_tensor_alloc(q_elems * sizeof(float));
        ds4_gpu_tensor *g_kc8 = ds4_gpu_tensor_alloc(cache8);
        ds4_gpu_tensor *g_vc8 = ds4_gpu_tensor_alloc(cache8);
        ds4_gpu_tensor_write(g_q, 0, q, q_elems * sizeof(float));
        ds4_gpu_tensor_write(g_k, 0, k, kv_elems * sizeof(float));
        ds4_gpu_tensor_write(g_v, 0, v, kv_elems * sizeof(float));
        { uint8_t *z = (uint8_t *)calloc(cache8, 1);
          ds4_gpu_tensor_write(g_kc8, 0, z, cache8);
          ds4_gpu_tensor_write(g_vc8, 0, z, cache8);
          free(z); }

        int ok = ds4_gpu_gqa_kv_cache_append_fp8(g_kc8, g_k, n_tok, n_kv_head, hd, cap, pos0) &&
                 ds4_gpu_gqa_kv_cache_append_fp8(g_vc8, g_v, n_tok, n_kv_head, hd, cap, pos0) &&
                 ds4_gpu_gqa_attention_fp8(g_out, g_q, g_kc8, g_vc8, n_tok, n_head, n_kv_head, hd,
                                           cap, pos0, scale, window, NULL, 0u) &&
                 cuda_ok(cudaDeviceSynchronize(), "gqa fp8 selftest sync");

        /* host-decode the GPU cache into f32 caches for the reference */
        uint8_t *kc8 = (uint8_t *)malloc(cache8);
        uint8_t *vc8 = (uint8_t *)malloc(cache8);
        if (ok) ok = ds4_gpu_tensor_read(g_kc8, 0, kc8, cache8) != 0 &&
                     ds4_gpu_tensor_read(g_vc8, 0, vc8, cache8) != 0;
        float enc_err = 0.0f;
        if (ok) {
            memset(ref_kc, 0, kcache_elems * sizeof(float));
            memset(ref_vc, 0, kcache_elems * sizeof(float));
            for (uint32_t h2 = 0; h2 < n_kv_head; h2++)
            for (uint32_t t2 = 0; t2 < n_tok; t2++) {
                const uint8_t *kr = kc8 + ((uint64_t)h2 * cap + pos0 + t2) * stride;
                const uint8_t *vr = vc8 + ((uint64_t)h2 * cap + pos0 + t2) * stride;
                float ksc, vsc;
                memcpy(&ksc, kr + hd, 4);
                memcpy(&vsc, vr + hd, 4);
                for (uint32_t d = 0; d < hd; d++) {
                    const float kd = e4m3_dec_common(kr[d]) * ksc;
                    const float vd = e4m3_dec_common(vr[d]) * vsc;
                    ref_kc[((uint64_t)h2 * cap + pos0 + t2) * hd + d] = kd;
                    ref_vc[((uint64_t)h2 * cap + pos0 + t2) * hd + d] = vd;
                    const float k0 = k[((uint64_t)t2 * n_kv_head + h2) * hd + d];
                    const float rel_e = fabsf(kd - k0) / fmaxf(fabsf(k0), 1e-3f);
                    enc_err = fmaxf(enc_err, rel_e);
                }
            }
        }
        float *ref_out = (float *)malloc(q_elems * sizeof(float));
        gqa_ref_attention(ref_out, q, ref_kc, ref_vc, n_tok, n_head, n_kv_head, hd, cap, pos0,
                          scale, window, NULL, 0u);
        float *gpu_out = (float *)malloc(q_elems * sizeof(float));
        if (ok) ok = ds4_gpu_tensor_read(g_out, 0, gpu_out, q_elems * sizeof(float)) != 0;
        float diff = ok ? gqa_max_abs_diff(ref_out, gpu_out, q_elems) : INFINITY;
        int pass = ok && diff < 5e-4f && enc_err < 0.07f;
        fprintf(stderr, "gqa-selftest phase 6 fp8-kv (n_tok=%u pos0=%u): %s (max diff %.2e, enc rel %.3f)\n",
                n_tok, pos0, pass ? "PASS" : "FAIL", diff, enc_err);
        if (!pass) all_ok = 0;

        free(q); free(k); free(v); free(kc8); free(vc8); free(ref_out); free(gpu_out);
        ds4_gpu_tensor_free(g_q); ds4_gpu_tensor_free(g_k); ds4_gpu_tensor_free(g_v);
        ds4_gpu_tensor_free(g_out); ds4_gpu_tensor_free(g_kc8); ds4_gpu_tensor_free(g_vc8);
    }

    /* Phase 6b/6c (q8_0 / q4_0 block KV): same shape as phase 6 but the
     * per-32-elem block layout (f16 d + 32 int8, or f16 d + 16 nibble
     * bytes). The GPU cache bytes are read back and host-decoded into f32
     * caches with an INDEPENDENT reference formula (the f16 bit-twiddle
     * below shares no code with the kernel's __half2float read path), then
     * gqa_ref_attention runs on the decoded values. A sub-tol diff proves
     * both that append wrote spec-correct block bytes and the attention
     * read path dequants them correctly. These formats previously shipped
     * without selftest coverage. */
    for (int blk_fmt = 0; blk_fmt < 2; blk_fmt++) {
        const int is_q8 = blk_fmt == 0;
        const uint32_t row_bytes = (hd / 32u) * (is_q8 ? 34u : 18u);
        const uint32_t n_tok = 48, pos0 = 9, window = 0;
        const float scale = 1.0f / sqrtf((float)hd);
        const uint64_t cache_b = (uint64_t)n_kv_head * cap * row_bytes;
        uint64_t q_elems = (uint64_t)n_tok * n_head * hd;
        uint64_t kv_elems = (uint64_t)n_tok * n_kv_head * hd;
        float *q = (float *)malloc(q_elems * sizeof(float));
        float *k = (float *)malloc(kv_elems * sizeof(float));
        float *v = (float *)malloc(kv_elems * sizeof(float));
        for (uint64_t i = 0; i < q_elems; i++) q[i] = gqa_test_randf();
        /* x3 like phase 6 so amax is non-trivial; block d = amax/{127,8.5}. */
        for (uint64_t i = 0; i < kv_elems; i++) { k[i] = gqa_test_randf() * 3.0f; v[i] = gqa_test_randf() * 3.0f; }

        ds4_gpu_tensor *g_q = ds4_gpu_tensor_alloc(q_elems * sizeof(float));
        ds4_gpu_tensor *g_k = ds4_gpu_tensor_alloc(kv_elems * sizeof(float));
        ds4_gpu_tensor *g_v = ds4_gpu_tensor_alloc(kv_elems * sizeof(float));
        ds4_gpu_tensor *g_out = ds4_gpu_tensor_alloc(q_elems * sizeof(float));
        ds4_gpu_tensor *g_kcb = ds4_gpu_tensor_alloc(cache_b);
        ds4_gpu_tensor *g_vcb = ds4_gpu_tensor_alloc(cache_b);
        ds4_gpu_tensor_write(g_q, 0, q, q_elems * sizeof(float));
        ds4_gpu_tensor_write(g_k, 0, k, kv_elems * sizeof(float));
        ds4_gpu_tensor_write(g_v, 0, v, kv_elems * sizeof(float));
        { uint8_t *z = (uint8_t *)calloc(cache_b, 1);
          ds4_gpu_tensor_write(g_kcb, 0, z, cache_b);
          ds4_gpu_tensor_write(g_vcb, 0, z, cache_b);
          free(z); }

        int ok = (is_q8
                  ? (ds4_gpu_gqa_kv_cache_append_q8_0(g_kcb, g_k, n_tok, n_kv_head, hd, cap, pos0) &&
                     ds4_gpu_gqa_kv_cache_append_q8_0(g_vcb, g_v, n_tok, n_kv_head, hd, cap, pos0) &&
                     ds4_gpu_gqa_attention_q8_0(g_out, g_q, g_kcb, g_vcb, n_tok, n_head, n_kv_head, hd,
                                               cap, pos0, scale, window, NULL, 0u))
                  : (ds4_gpu_gqa_kv_cache_append_q4_0(g_kcb, g_k, n_tok, n_kv_head, hd, cap, pos0) &&
                     ds4_gpu_gqa_kv_cache_append_q4_0(g_vcb, g_v, n_tok, n_kv_head, hd, cap, pos0) &&
                     ds4_gpu_gqa_attention_q4_0(g_out, g_q, g_kcb, g_vcb, n_tok, n_head, n_kv_head, hd,
                                               cap, pos0, scale, window, NULL, 0u))) &&
                 cuda_ok(cudaDeviceSynchronize(), "gqa block selftest sync");

        /* host-decode the GPU cache with a spec-reference formula (not the
         * kernel's own dequant) so a PASS is independent confirmation. */
        uint8_t *kcb = (uint8_t *)malloc(cache_b);
        uint8_t *vcb = (uint8_t *)malloc(cache_b);
        if (ok) ok = ds4_gpu_tensor_read(g_kcb, 0, kcb, cache_b) != 0 &&
                     ds4_gpu_tensor_read(g_vcb, 0, vcb, cache_b) != 0;
        /* IEEE-754 binary16 -> f32, pure host bit-twiddle. Independence
         * from __half2float is the point: a PASS/FAIL does not share code
         * with the kernel's read path. */
        auto f16_ref = [](uint16_t h) -> float {
            uint32_t sign = (uint32_t)(h >> 15) & 1u;
            uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
            uint32_t mant = (uint32_t)h & 0x3FFu;
            uint32_t bits;
            if (exp == 0u) {
                if (mant == 0u) {
                    bits = sign << 31;
                } else {
                    uint32_t e = 0u;
                    while ((mant & 0x400u) == 0u) { mant <<= 1; e++; }
                    mant &= 0x3FFu;
                    bits = (sign << 31) | ((127u - 14u - e) << 23) | (mant << 13);
                }
            } else if (exp == 31u) {
                bits = (sign << 31) | (0xFFu << 23) | (mant << 13);
            } else {
                bits = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
            }
            float f; memcpy(&f, &bits, 4); return f;
        };
        if (ok) {
            memset(ref_kc, 0, kcache_elems * sizeof(float));
            memset(ref_vc, 0, kcache_elems * sizeof(float));
            for (uint32_t h2 = 0; h2 < n_kv_head; h2++)
            for (uint32_t t2 = 0; t2 < n_tok; t2++) {
                const uint8_t *kr = kcb + ((uint64_t)h2 * cap + pos0 + t2) * row_bytes;
                const uint8_t *vr = vcb + ((uint64_t)h2 * cap + pos0 + t2) * row_bytes;
                const uint32_t nblk = hd / 32u;
                for (uint32_t b = 0; b < nblk; b++) {
                    uint16_t kbits = (uint16_t)kr[b * (is_q8 ? 34u : 18u)]
                                   | ((uint16_t)kr[b * (is_q8 ? 34u : 18u) + 1u] << 8);
                    uint16_t vbits = (uint16_t)vr[b * (is_q8 ? 34u : 18u)]
                                   | ((uint16_t)vr[b * (is_q8 ? 34u : 18u) + 1u] << 8);
                    float kd = f16_ref(kbits);
                    float vd = f16_ref(vbits);
                    for (uint32_t i = 0; i < 32u; i++) {
                        float kv, vv;
                        if (is_q8) {
                            kv = kd * (float)((int8_t)kr[b * 34u + 2u + i]);
                            vv = vd * (float)((int8_t)vr[b * 34u + 2u + i]);
                        } else {
                            uint32_t j = i & 15u;
                            uint8_t kb = kr[b * 18u + 2u + j];
                            uint8_t vb = vr[b * 18u + 2u + j];
                            int kq = (int)((kb >> (((i >> 4) & 1u) << 2)) & 0xFu) - 8;
                            int vq = (int)((vb >> (((i >> 4) & 1u) << 2)) & 0xFu) - 8;
                            kv = kd * (float)kq;
                            vv = vd * (float)vq;
                        }
                        uint32_t di = b * 32u + i;
                        ref_kc[((uint64_t)h2 * cap + pos0 + t2) * hd + di] = kv;
                        ref_vc[((uint64_t)h2 * cap + pos0 + t2) * hd + di] = vv;
                    }
                }
            }
        }
        float *ref_out = (float *)malloc(q_elems * sizeof(float));
        gqa_ref_attention(ref_out, q, ref_kc, ref_vc, n_tok, n_head, n_kv_head, hd, cap, pos0,
                          scale, window, NULL, 0u);
        float *gpu_out = (float *)malloc(q_elems * sizeof(float));
        if (ok) ok = ds4_gpu_tensor_read(g_out, 0, gpu_out, q_elems * sizeof(float)) != 0;
        float diff = ok ? gqa_max_abs_diff(ref_out, gpu_out, q_elems) : INFINITY;
        /* End-to-end gate: the host decode above shares no code with the
         * kernel's dequant, so a sub-tol diff proves both append (wrote
         * spec-correct block bytes) and the attention read path. Residual
         * is honest block-quant error (q8_0 step ~d/127, q4_0 ~d/8.5),
         * which the 1/sqrt(hd) qk scale leaves well under tol. */
        float tol = is_q8 ? 2e-2f : 0.5f;
        int pass = ok && diff < tol;
        fprintf(stderr, "gqa-selftest phase 6%c %s-kv (n_tok=%u pos0=%u): %s (max diff %.2e)\n",
                is_q8 ? 'b' : 'c', is_q8 ? "q8_0" : "q4_0",
                n_tok, pos0, pass ? "PASS" : "FAIL", diff);
        if (!pass) all_ok = 0;

        free(q); free(k); free(v); free(kcb); free(vcb); free(ref_out); free(gpu_out);
        ds4_gpu_tensor_free(g_q); ds4_gpu_tensor_free(g_k); ds4_gpu_tensor_free(g_v);
        ds4_gpu_tensor_free(g_out); ds4_gpu_tensor_free(g_kcb); ds4_gpu_tensor_free(g_vcb);
    }

    /* Phase 7 (split-K decode): a 7000-row cache forces the split path
     * (rows >= 4096, n_tok <= 4). Three sub-cases: f32 full-causal, f32
     * with a 5000-row sliding window (split ranges start at row0 > 0),
     * and fp8 KV. Reference = the same CPU function as every other
     * phase, so this also cross-checks split vs plain semantics. */
    {
        const uint32_t n_head2 = 16, n_kv2 = 2, hd2 = 256, cap2 = 8192;
        const uint32_t pos2 = 7000, n_tok2 = 2;
        const float scale2 = 1.0f / sqrtf((float)hd2);
        const uint64_t cache_elems2 = (uint64_t)n_kv2 * cap2 * hd2;
        const uint64_t stride8 = (uint64_t)hd2 + 4u;
        const uint64_t cache8_2 = (uint64_t)n_kv2 * cap2 * stride8;
        const uint64_t q_elems2 = (uint64_t)n_tok2 * n_head2 * hd2;
        const uint64_t fill_elems = (uint64_t)(pos2 + n_tok2) * n_kv2 * hd2;

        float *kc2 = (float *)calloc(cache_elems2, sizeof(float));
        float *vc2 = (float *)calloc(cache_elems2, sizeof(float));
        float *kin = (float *)malloc(fill_elems * sizeof(float));
        float *vin = (float *)malloc(fill_elems * sizeof(float));
        float *q2 = (float *)malloc(q_elems2 * sizeof(float));
        for (uint64_t i = 0; i < fill_elems; i++) { kin[i] = gqa_test_randf(); vin[i] = gqa_test_randf(); }
        for (uint64_t i = 0; i < q_elems2; i++) q2[i] = gqa_test_randf();
        for (uint32_t r = 0; r < pos2 + n_tok2; r++)
        for (uint32_t h2 = 0; h2 < n_kv2; h2++)
        for (uint32_t d = 0; d < hd2; d++) {
            kc2[((uint64_t)h2 * cap2 + r) * hd2 + d] = kin[((uint64_t)r * n_kv2 + h2) * hd2 + d];
            vc2[((uint64_t)h2 * cap2 + r) * hd2 + d] = vin[((uint64_t)r * n_kv2 + h2) * hd2 + d];
        }

        ds4_gpu_tensor *g_q2 = ds4_gpu_tensor_alloc(q_elems2 * sizeof(float));
        ds4_gpu_tensor *g_kc2 = ds4_gpu_tensor_alloc(cache_elems2 * sizeof(float));
        ds4_gpu_tensor *g_vc2 = ds4_gpu_tensor_alloc(cache_elems2 * sizeof(float));
        ds4_gpu_tensor *g_out2 = ds4_gpu_tensor_alloc(q_elems2 * sizeof(float));
        ds4_gpu_tensor_write(g_q2, 0, q2, q_elems2 * sizeof(float));
        ds4_gpu_tensor_write(g_kc2, 0, kc2, cache_elems2 * sizeof(float));
        ds4_gpu_tensor_write(g_vc2, 0, vc2, cache_elems2 * sizeof(float));

        float *ref_out2 = (float *)malloc(q_elems2 * sizeof(float));
        float *gpu_out2 = (float *)malloc(q_elems2 * sizeof(float));
        const uint32_t wins[2] = { 0u, 5000u };
        for (int wi = 0; wi < 2; wi++) {
            gqa_ref_attention(ref_out2, q2, kc2, vc2, n_tok2, n_head2, n_kv2, hd2, cap2, pos2,
                              scale2, wins[wi], NULL, 0u);
            int ok = ds4_gpu_gqa_attention(g_out2, g_q2, g_kc2, g_vc2, n_tok2, n_head2, n_kv2,
                                           hd2, cap2, pos2, scale2, wins[wi], NULL, 0u) &&
                     cuda_ok(cudaDeviceSynchronize(), "gqa split selftest sync");
            if (ok) ok = ds4_gpu_tensor_read(g_out2, 0, gpu_out2, q_elems2 * sizeof(float)) != 0;
            float diff = ok ? gqa_max_abs_diff(ref_out2, gpu_out2, q_elems2) : INFINITY;
            int pass = ok && diff < 5e-4f;
            fprintf(stderr, "gqa-selftest phase 7 split-k (rows=%u win=%u): %s (max diff %.2e)\n",
                    pos2 + n_tok2, wins[wi], pass ? "PASS" : "FAIL", diff);
            if (!pass) all_ok = 0;
        }

        /* fp8 KV split: append the same rows through the encoder, decode
         * the GPU cache bytes on the host for the reference (phase-6
         * recipe) */
        ds4_gpu_tensor *g_kin = ds4_gpu_tensor_alloc(fill_elems * sizeof(float));
        ds4_gpu_tensor *g_vin = ds4_gpu_tensor_alloc(fill_elems * sizeof(float));
        ds4_gpu_tensor *g_kc8 = ds4_gpu_tensor_alloc(cache8_2);
        ds4_gpu_tensor *g_vc8 = ds4_gpu_tensor_alloc(cache8_2);
        ds4_gpu_tensor_write(g_kin, 0, kin, fill_elems * sizeof(float));
        ds4_gpu_tensor_write(g_vin, 0, vin, fill_elems * sizeof(float));
        int ok8 = ds4_gpu_gqa_kv_cache_append_fp8(g_kc8, g_kin, pos2 + n_tok2, n_kv2, hd2, cap2, 0u) &&
                  ds4_gpu_gqa_kv_cache_append_fp8(g_vc8, g_vin, pos2 + n_tok2, n_kv2, hd2, cap2, 0u) &&
                  ds4_gpu_gqa_attention_fp8(g_out2, g_q2, g_kc8, g_vc8, n_tok2, n_head2, n_kv2,
                                            hd2, cap2, pos2, scale2, 0u, NULL, 0u) &&
                  cuda_ok(cudaDeviceSynchronize(), "gqa split fp8 selftest sync");
        uint8_t *kc8h = (uint8_t *)malloc(cache8_2);
        uint8_t *vc8h = (uint8_t *)malloc(cache8_2);
        if (ok8) ok8 = ds4_gpu_tensor_read(g_kc8, 0, kc8h, cache8_2) != 0 &&
                       ds4_gpu_tensor_read(g_vc8, 0, vc8h, cache8_2) != 0;
        if (ok8) {
            memset(kc2, 0, cache_elems2 * sizeof(float));
            memset(vc2, 0, cache_elems2 * sizeof(float));
            for (uint32_t h2 = 0; h2 < n_kv2; h2++)
            for (uint32_t r = 0; r < pos2 + n_tok2; r++) {
                const uint8_t *kr = kc8h + ((uint64_t)h2 * cap2 + r) * stride8;
                const uint8_t *vr = vc8h + ((uint64_t)h2 * cap2 + r) * stride8;
                float ksc, vsc;
                memcpy(&ksc, kr + hd2, 4);
                memcpy(&vsc, vr + hd2, 4);
                for (uint32_t d = 0; d < hd2; d++) {
                    kc2[((uint64_t)h2 * cap2 + r) * hd2 + d] = e4m3_dec_common(kr[d]) * ksc;
                    vc2[((uint64_t)h2 * cap2 + r) * hd2 + d] = e4m3_dec_common(vr[d]) * vsc;
                }
            }
        }
        gqa_ref_attention(ref_out2, q2, kc2, vc2, n_tok2, n_head2, n_kv2, hd2, cap2, pos2,
                          scale2, 0u, NULL, 0u);
        if (ok8) ok8 = ds4_gpu_tensor_read(g_out2, 0, gpu_out2, q_elems2 * sizeof(float)) != 0;
        float diff8 = ok8 ? gqa_max_abs_diff(ref_out2, gpu_out2, q_elems2) : INFINITY;
        int pass8 = ok8 && diff8 < 1e-3f;
        fprintf(stderr, "gqa-selftest phase 7 split-k fp8 (rows=%u): %s (max diff %.2e)\n",
                pos2 + n_tok2, pass8 ? "PASS" : "FAIL", diff8);
        if (!pass8) all_ok = 0;

        free(kc2); free(vc2); free(kin); free(vin); free(q2);
        free(ref_out2); free(gpu_out2); free(kc8h); free(vc8h);
        ds4_gpu_tensor_free(g_q2); ds4_gpu_tensor_free(g_kc2); ds4_gpu_tensor_free(g_vc2);
        ds4_gpu_tensor_free(g_out2); ds4_gpu_tensor_free(g_kin); ds4_gpu_tensor_free(g_vin);
        ds4_gpu_tensor_free(g_kc8); ds4_gpu_tensor_free(g_vc8);
    }

    free(qn_w); free(kn_w); free(ref_kc); free(ref_vc);
    ds4_gpu_tensor_free(g_qnw); ds4_gpu_tensor_free(g_knw);
    ds4_gpu_tensor_free(g_kc); ds4_gpu_tensor_free(g_vc);
    return all_ok;
}
