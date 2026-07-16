/* MLA (multi-head latent attention) kernels for the GLM-5.2 / DeepSeek
 * family, compact-KV path. Ported from the NeutronStar fork of ds4
 * (ds4_cuda_glm_attn.inc / ds4_cuda_glm_kv.inc / ds4_cuda_glm_indexer.inc,
 * themselves ports of metal/dsv4_misc.metal), MIT License:
 *   Copyright (c) 2026 The ds4.c authors
 *   Copyright (c) 2023-2026 The ggml authors
 *
 * pulsar contract changes vs ds4: weights are explicit device pointers
 * (no model map), caches are f32 only (the compact cache is ~186MB at ctx
 * 2048 for 79 layers - f16 would save nothing that matters yet), and the
 * DSA indexer is absent: attention runs over ALL visible rows via the
 * range-fill selection, which is exactly what ds4 does whenever
 * visible <= indexer_top_k (2048). Long-context sparse selection is an
 * optimization for later, not a correctness feature.
 *
 * The cache layout is the latent trick that makes 743B-class models
 * streamable: per position only kv_lora_dim (512) + qk_rope (64) floats,
 * shared across all 64 heads. Keys never materialize; scores fold through
 * q @ k_b once per token (qk_lowrank), and values project from the
 * attention-weighted latent through v_b afterwards.
 *
 * RoPE notes: q gets its rope tail at build time (query position); the
 * cached k_rope tail is stored UNROTATED and rotated lazily by row
 * position at attention time. YaRN parameters are carried but GLM-5.2
 * ships ext_factor 0 (yarn off). */

/* ---- device helpers ---------------------------------------------------- */

__device__ static float mla_rope_ramp(float low, float high, int i0) {
    const float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

__device__ static void mla_rope_corr_dims(
        uint32_t n_dims, uint32_t n_ctx_orig, float freq_base,
        float beta_fast, float beta_slow, float *c0, float *c1) {
    const float denom = 2.0f * logf(freq_base);
    const float f_fast = (float)n_dims *
        logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom;
    const float f_slow = (float)n_dims *
        logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom;
    *c0 = fmaxf(0.0f, floorf(f_fast));
    *c1 = fminf((float)n_dims - 1.0f, ceilf(f_slow));
}

__device__ static void mla_rope_yarn(
        float theta_extrap, float freq_scale, float c0, float c1, int i0,
        float ext_factor, float mscale, float *cos_theta, float *sin_theta) {
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float ramp_mix = mla_rope_ramp(c0, c1, i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

/* q8_0 row dot against an f32 vector (row = 34-byte blocks of 32). */
__device__ static float mla_q8_dot_row(
        const char *row, const float *x, uint32_t n_cols) {
    float acc = 0.0f;
    const uint32_t n_blocks = (n_cols + 31u) >> 5;
    for (uint32_t block = 0; block < n_blocks; block++) {
        const char *block_base = row + (uint64_t)block * 34u;
        const float d = __half2float(*(const __half *)block_base);
        const int8_t *qs = (const int8_t *)(block_base + 2u);
        const uint32_t base = block << 5;
        const uint32_t count = min(32u, n_cols - base);
        for (uint32_t qi = 0; qi < count; qi++) {
            acc += d * (float)qs[qi] * x[base + qi];
        }
    }
    return acc;
}

static uint64_t mla_q8_row_bytes(uint32_t n_cols) {
    return (((uint64_t)n_cols + 31u) / 32u) * 34u;
}

static int mla_smem_ok(const void *func, uint64_t smem_bytes, const char *what) {
    if (smem_bytes <= 48u * 1024u) return 1;
    if (smem_bytes > (uint64_t)INT_MAX) return 0;
    return cuda_ok(cudaFuncSetAttribute(
                           (const void *)func,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)smem_bytes),
                   what);
}

/* ---- rope tail: rotate the last rot_dim dims of each head vector ------- */

__global__ static void mla_rope_tail_kernel(
        float *x,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t rot_dim,
        uint32_t rot_offset,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float sin_sign /* -1 = inverse rotation (deepseek4 heads un-rope) */) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t half_rot = rot_dim / 2u;
    const uint32_t pairs = n_tokens * n_head * half_rot;
    if (gid >= pairs) return;
    const uint32_t pair = gid % half_rot;
    const uint32_t tmp = gid / half_rot;
    const uint32_t head = tmp % n_head;
    const uint32_t token = tmp / n_head;
    const uint32_t i = pair * 2u;

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        mla_rope_corr_dims(rot_dim, n_ctx_orig, freq_base,
                           beta_fast, beta_slow, &corr0, &corr1);
    }
    const float theta_extrap = (float)(pos0 + token) *
        powf(freq_base, -((float)i) / (float)rot_dim);
    float c, s;
    mla_rope_yarn(theta_extrap, freq_scale, corr0, corr1, (int)i,
                  ext_factor, attn_factor, &c, &s);
    s *= sin_sign;

    float *row = x + ((uint64_t)token * n_head + head) * head_dim + rot_offset;
    const float x0 = row[i];
    const float x1 = row[i + 1u];
    row[i] = x0 * c - x1 * s;
    row[i + 1u] = x0 * s + x1 * c;
}

extern "C" int pulsar_mla_rope_tail(
        void *x_dev,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t rot_dim,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    if (!x_dev || n_tokens == 0 || n_head == 0 || head_dim == 0 ||
        rot_dim == 0 || rot_dim > head_dim || (rot_dim & 1u) != 0) {
        return 0;
    }
    const uint32_t pairs = n_tokens * n_head * (rot_dim / 2u);
    mla_rope_tail_kernel<<<(pairs + 255u) / 256u, 256>>>(
            (float *)x_dev, n_tokens, n_head, head_dim, rot_dim,
            head_dim - rot_dim, pos0, n_ctx_orig,
            freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow, 1.0f);
    return cuda_ok(cudaGetLastError(), "mla rope tail launch");
}

/* deepseek4 rope tail: same rotation as mla_rope_tail plus an inverse
 * mode - V4 un-rotates the attention output heads before the grouped
 * output projection (the value rows carry their key-side rope). */
extern "C" int pulsar_dsv4_rope_tail(
        void *x_dev,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t rot_dim,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        uint32_t inverse) {
    if (!x_dev || n_tokens == 0 || n_head == 0 || head_dim == 0 ||
        rot_dim == 0 || rot_dim > head_dim || (rot_dim & 1u) != 0) {
        return 0;
    }
    const uint32_t pairs = n_tokens * n_head * (rot_dim / 2u);
    mla_rope_tail_kernel<<<(pairs + 255u) / 256u, 256>>>(
            (float *)x_dev, n_tokens, n_head, head_dim, rot_dim,
            head_dim - rot_dim, pos0, n_ctx_orig,
            freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow, inverse ? -1.0f : 1.0f);
    return cuda_ok(cudaGetLastError(), "dsv4 rope tail launch");
}

/* ---- kv lora rms norm: first kv_lora_dim of each kv_raw row ------------ */

__global__ static void mla_kv_lora_rms_norm_kernel(
        float *out,
        const float *kv_raw,
        const float *w,
        uint32_t n_tokens,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= n_tokens) return;
    const float *x = kv_raw + (uint64_t)row * kv_raw_dim;
    float *o = out + (uint64_t)row * kv_lora_dim;

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < kv_lora_dim; i += blockDim.x) {
        const float v = x[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / (float)kv_lora_dim + eps);
    for (uint32_t i = threadIdx.x; i < kv_lora_dim; i += blockDim.x) {
        o[i] = x[i] * inv * w[i];
    }
}

extern "C" int pulsar_mla_kv_lora_rms_norm(
        void *out_dev,
        const void *kv_raw_dev,
        const void *w_dev,
        uint32_t n_tokens,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        float eps) {
    if (!out_dev || !kv_raw_dev || !w_dev ||
        n_tokens == 0 || kv_raw_dim == 0 || kv_lora_dim == 0 ||
        kv_lora_dim > kv_raw_dim) {
        return 0;
    }
    mla_kv_lora_rms_norm_kernel<<<n_tokens, 256>>>(
            (float *)out_dev, (const float *)kv_raw_dev, (const float *)w_dev,
            n_tokens, kv_raw_dim, kv_lora_dim, eps);
    return cuda_ok(cudaGetLastError(), "mla kv lora rms norm launch");
}

/* ---- store compact kv: latent row + raw rope tail at pos --------------- */

__global__ static void mla_store_compact_kv_kernel(
        const float *kv_norm,
        const float *kv_raw,
        float *kv_lora_cache,
        float *k_rope_cache,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_cap,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        uint32_t qk_rope) {
    const uint32_t token = blockIdx.x;
    const uint32_t part = blockIdx.y;
    if (token >= n_tokens || part > 1u) return;
    const uint32_t pos = pos0 + token;
    if (pos >= cache_cap) return;

    if (part == 0u) {
        const float *src = kv_norm + (uint64_t)token * kv_lora_dim;
        for (uint32_t i = threadIdx.x; i < kv_lora_dim; i += blockDim.x) {
            kv_lora_cache[(uint64_t)pos * kv_lora_dim + i] = src[i];
        }
    } else {
        const float *src = kv_raw + (uint64_t)token * kv_raw_dim + kv_lora_dim;
        for (uint32_t i = threadIdx.x; i < qk_rope; i += blockDim.x) {
            k_rope_cache[(uint64_t)pos * qk_rope + i] = src[i];
        }
    }
}

extern "C" int pulsar_mla_store_compact_kv(
        void *kv_lora_cache_dev,
        void *k_rope_cache_dev,
        const void *kv_norm_dev,
        const void *kv_raw_dev,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_cap,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        uint32_t qk_rope) {
    if (!kv_lora_cache_dev || !k_rope_cache_dev || !kv_norm_dev || !kv_raw_dev ||
        n_tokens == 0 || cache_cap == 0 ||
        kv_lora_dim == 0 || qk_rope == 0 ||
        kv_lora_dim + qk_rope > kv_raw_dim ||
        pos0 > cache_cap || n_tokens > cache_cap - pos0) {
        return 0;
    }
    dim3 grid(n_tokens, 2, 1);
    mla_store_compact_kv_kernel<<<grid, 256>>>(
            (const float *)kv_norm_dev, (const float *)kv_raw_dev,
            (float *)kv_lora_cache_dev, (float *)k_rope_cache_dev,
            pos0, n_tokens, cache_cap, kv_raw_dim, kv_lora_dim, qk_rope);
    return cuda_ok(cudaGetLastError(), "mla store compact kv launch");
}

/* ---- selection range fill: token t sees rows [0, pos0+t] --------------- */

__global__ static void mla_fill_selected_range_kernel(
        uint32_t *selected, uint32_t n_tokens, uint32_t pos0,
        uint32_t n_selected, uint32_t pad_row) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = n_tokens * n_selected;
    if (gid >= total || n_selected == 0u) return;
    const uint32_t token = gid / n_selected;
    const uint32_t slot = gid - token * n_selected;
    const uint32_t visible = pos0 + token + 1u;
    selected[gid] = slot < visible ? slot : pad_row;
}

extern "C" int pulsar_mla_fill_selected_range(
        void *selected_dev,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_selected,
        uint32_t pad_row) {
    if (!selected_dev || n_tokens == 0 || n_selected == 0) return 0;
    const uint64_t total = (uint64_t)n_tokens * n_selected;
    if (total > UINT32_MAX) return 0;
    mla_fill_selected_range_kernel<<<(uint32_t)((total + 255u) / 256u), 256>>>(
            (uint32_t *)selected_dev, n_tokens, pos0, n_selected, pad_row);
    return cuda_ok(cudaGetLastError(), "mla fill selected range launch");
}

/* ---- qk lowrank: fold q_nope through k_b into latent space ------------- */

__global__ static void mla_qk_lowrank_kernel(
        float *qk_low,
        const float *q,
        const char *weight,        /* q8_0 [n_head][kv_lora_dim] rows of qk_nope */
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_dim,
        uint32_t row_bytes) {
    const uint32_t head = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (head >= n_head || token >= n_tokens) return;
    const float *qh = q + ((uint64_t)token * n_head + head) * qk_dim;
    float *out = qk_low + ((uint64_t)token * n_head + head) * kv_lora_dim;
    for (uint32_t j = threadIdx.x; j < kv_lora_dim; j += blockDim.x) {
        const char *row = weight + ((uint64_t)head * kv_lora_dim + j) * row_bytes;
        out[j] = mla_q8_dot_row(row, qh, qk_nope);
    }
}

extern "C" int pulsar_mla_qk_lowrank(
        void *qk_low_dev,
        const void *q_dev,
        const void *k_b_dev,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_dim) {
    if (!qk_low_dev || !q_dev || !k_b_dev ||
        n_tokens == 0 || n_head == 0 || kv_lora_dim == 0 ||
        qk_nope == 0 || qk_nope > qk_dim) {
        return 0;
    }
    dim3 grid(n_head, n_tokens, 1);
    mla_qk_lowrank_kernel<<<grid, 256>>>(
            (float *)qk_low_dev, (const float *)q_dev, (const char *)k_b_dev,
            n_tokens, n_head, kv_lora_dim, qk_nope, qk_dim,
            (uint32_t)mla_q8_row_bytes(qk_nope));
    return cuda_ok(cudaGetLastError(), "mla qk lowrank launch");
}

/* ---- fused attention over selected compact rows ------------------------
 * scores + softmax + latent accumulation + q8_0 v_b projection in one
 * launch. Cached k_rope is rotated lazily by row position here. */

__global__ static void mla_attention_kernel(
        float *heads,
        const float *q,
        const float *qk_low,
        const float *kv_lora_cache,
        const float *k_rope_cache,
        const char *value_weight,  /* q8_0 [n_head][value_dim] rows of kv_lora_dim */
        const uint32_t *selected,
        uint32_t n_tokens,
        uint32_t n_selected,
        uint32_t cache_cap,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope,
        uint32_t value_dim,
        uint32_t n_ctx_orig,
        uint32_t value_row_bytes,
        float scale,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint32_t head = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (head >= n_head || token >= n_tokens || n_selected == 0u) return;
    const uint32_t nth = blockDim.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t qk_dim = qk_nope + qk_rope;

    extern __shared__ float mla_attn_sm[];
    float *red = mla_attn_sm;
    float *scores = mla_attn_sm + 256u;
    float *lora_sum = scores + n_selected;

    const float *qh = q + ((uint64_t)token * n_head + head) * qk_dim;
    const float *low = qk_low + ((uint64_t)token * n_head + head) * kv_lora_dim;
    const uint32_t *token_selected = selected + (uint64_t)token * n_selected;

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        mla_rope_corr_dims(qk_rope, n_ctx_orig, freq_base,
                           beta_fast, beta_slow, &corr0, &corr1);
    }

    float local_max = -INFINITY;
    const float inv_ndims = -1.0f / (float)qk_rope;
    for (uint32_t s = tid; s < n_selected; s += nth) {
        const uint32_t row = token_selected[s];
        float score = -INFINITY;
        if (row < cache_cap) {
            float dotv = 0.0f;
            const uint64_t lora_base = (uint64_t)row * kv_lora_dim;
            for (uint32_t j = 0; j < kv_lora_dim; j++) {
                dotv += low[j] * kv_lora_cache[lora_base + j];
            }
            const uint64_t rope_base = (uint64_t)row * qk_rope;
            for (uint32_t r = 0; r < qk_rope; r += 2u) {
                const float theta = (float)row *
                    powf(freq_base, inv_ndims * (float)r);
                float c, sn;
                mla_rope_yarn(theta, freq_scale, corr0, corr1, (int)r,
                              ext_factor, attn_factor, &c, &sn);
                const float x0 = k_rope_cache[rope_base + r];
                const float x1 = k_rope_cache[rope_base + r + 1u];
                const float y0 = x0 * c - x1 * sn;
                const float y1 = x0 * sn + x1 * c;
                dotv += qh[qk_nope + r] * y0 + qh[qk_nope + r + 1u] * y1;
            }
            score = dotv * scale;
        }
        scores[s] = score;
        local_max = fmaxf(local_max, score);
    }
    red[tid] = local_max;
    __syncthreads();
    for (uint32_t step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] = fmaxf(red[tid], red[tid + step]);
        __syncthreads();
    }
    const float max_score = red[0];

    float local_sum = 0.0f;
    for (uint32_t s = tid; s < n_selected; s += nth) {
        const float w = expf(scores[s] - max_score);
        scores[s] = w;
        local_sum += w;
    }
    red[tid] = local_sum;
    __syncthreads();
    for (uint32_t step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        __syncthreads();
    }
    const float denom = fmaxf(red[0], 1.0e-20f);
    __syncthreads();

    for (uint32_t j = tid; j < kv_lora_dim; j += nth) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < n_selected; s++) {
            const uint32_t row = token_selected[s];
            if (row < cache_cap) {
                acc += scores[s] * kv_lora_cache[(uint64_t)row * kv_lora_dim + j];
            }
        }
        lora_sum[j] = acc / denom;
    }
    __syncthreads();

    float *out = heads + ((uint64_t)token * n_head + head) * value_dim;
    for (uint32_t d = tid; d < value_dim; d += nth) {
        const char *row =
            value_weight + ((uint64_t)head * value_dim + d) * value_row_bytes;
        out[d] = mla_q8_dot_row(row, lora_sum, kv_lora_dim);
    }
}

extern "C" int pulsar_mla_attention(
        void *heads_dev,
        const void *q_dev,
        const void *qk_low_dev,
        const void *kv_lora_cache_dev,
        const void *k_rope_cache_dev,
        const void *v_b_dev,
        const void *selected_dev,
        uint32_t n_tokens,
        uint32_t n_selected,
        uint32_t cache_cap,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope,
        uint32_t value_dim,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float kq_mult) {
    if (!heads_dev || !q_dev || !qk_low_dev || !kv_lora_cache_dev ||
        !k_rope_cache_dev || !v_b_dev || !selected_dev ||
        n_tokens == 0 || n_selected == 0 || cache_cap == 0 ||
        n_selected > cache_cap + 1u ||
        n_head == 0 || kv_lora_dim == 0 ||
        qk_nope == 0 || qk_rope == 0 || (qk_rope & 1u) != 0 ||
        value_dim == 0) {
        return 0;
    }
    const uint64_t smem =
        (256u + (uint64_t)n_selected + kv_lora_dim) * sizeof(float);
    if (!mla_smem_ok((const void *)mla_attention_kernel, smem,
                     "mla attention smem opt-in")) {
        return 0;
    }
    const uint32_t qk_dim = qk_nope + qk_rope;
    dim3 grid(n_head, n_tokens, 1);
    mla_attention_kernel<<<grid, 256, (size_t)smem>>>(
            (float *)heads_dev,
            (const float *)q_dev,
            (const float *)qk_low_dev,
            (const float *)kv_lora_cache_dev,
            (const float *)k_rope_cache_dev,
            (const char *)v_b_dev,
            (const uint32_t *)selected_dev,
            n_tokens, n_selected, cache_cap,
            n_head, kv_lora_dim, qk_nope, qk_rope, value_dim, n_ctx_orig,
            (uint32_t)mla_q8_row_bytes(kv_lora_dim),
            kq_mult / sqrtf((float)qk_dim),
            freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), "mla attention launch");
}
