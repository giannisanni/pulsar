/* Ported from the NeutronStar fork of ds4 (github.com/antirez/ds4),
 * MIT License:
 *   Copyright (c) 2026 The ds4.c authors
 *   Copyright (c) 2023-2026 The ggml authors
 *
 * GLM-5.2 DSA lightning indexer: a small per-layer scorer picks the
 * top-k KV rows each token attends to, lifting the ctx <= top_k limit.
 * scores[row] = sum_h relu(dot(q_h, k_row) * scale) * w_h  (relu AFTER
 * scaling, sum NOT rescaled - unlike the DeepSeek-V4 indexer).
 * Selected row lists feed pulsar_mla_attention unchanged. */

/* rope over the FIRST rot_dim dims in adjacent (i, i+1) pairs, with YaRN
 * corrections - the indexer flavor of the rope tail (rot_offset = 0).
 * Reuses mla_kernels.inc yarn helpers (included earlier in the .cu). */
__global__ static void idx_rope0_kernel(
        float *x, uint32_t n_tokens, uint32_t n_head, uint32_t head_dim,
        uint32_t rot_dim, uint32_t pos0, uint32_t n_ctx_orig,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow) {
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
    const float theta = (float)(pos0 + token) *
        powf(freq_base, -((float)i) / (float)rot_dim);
    float cos_t, sin_t;
    mla_rope_yarn(theta, freq_scale, corr0, corr1, (int)i,
                  ext_factor, attn_factor, &cos_t, &sin_t);
    float *v = x + ((uint64_t)token * n_head + head) * head_dim + i;
    const float x0 = v[0], x1 = v[1];
    v[0] = x0 * cos_t - x1 * sin_t;
    v[1] = x0 * sin_t + x1 * cos_t;
}

extern "C" int pulsar_idx_rope0(
        void *x_dev, uint32_t n_tokens, uint32_t n_head, uint32_t head_dim,
        uint32_t rot_dim, uint32_t pos0, uint32_t n_ctx_orig,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow) {
    if (!x_dev || n_tokens == 0 || n_head == 0 || rot_dim == 0 ||
        rot_dim > head_dim || (rot_dim & 1u)) {
        return 0;
    }
    const uint32_t pairs = n_tokens * n_head * (rot_dim / 2u);
    idx_rope0_kernel<<<(pairs + 255u) / 256u, 256>>>(
            (float *)x_dev, n_tokens, n_head, head_dim, rot_dim, pos0,
            n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), "idx rope0 launch");
}

/* LayerNorm (mean/var, weight + bias) over the raw indexer K row, partial
 * rope over the first rot_dim dims, stored to the f32 indexer key cache
 * at rows pos0..pos0+n_tokens. */
__global__ static void idx_store_k_kernel(
        const float *raw_k, const float *w, const float *b,
        float *key_cache, uint32_t pos0, uint32_t n_tokens,
        uint32_t cache_cap, uint32_t head_dim, uint32_t rot_dim,
        uint32_t n_ctx_orig, float eps, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    const uint32_t token = blockIdx.x;
    if (token >= n_tokens) return;
    const uint32_t pos = pos0 + token;
    if (pos >= cache_cap) return;
    const float *src = raw_k + (uint64_t)token * head_dim;
    __shared__ float partial[256];

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) sum += src[i];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    const float mean = partial[0] / (float)head_dim;
    __syncthreads();
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float d = src[i] - mean;
        ss += d * d;
    }
    partial[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / (float)head_dim + eps);

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        mla_rope_corr_dims(rot_dim, n_ctx_orig, freq_base,
                                beta_fast, beta_slow, &corr0, &corr1);
    }
    float *dst = key_cache + (uint64_t)pos * head_dim;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        if (i < rot_dim) {
            if ((i & 1u) != 0u) continue;
            const float theta = (float)pos * powf(freq_base, -1.0f / (float)rot_dim * (float)i);
            float cos_t, sin_t;
            mla_rope_yarn(theta, freq_scale, corr0, corr1, (int)i,
                          ext_factor, attn_factor, &cos_t, &sin_t);
            const float x0 = (src[i] - mean) * inv * w[i] + b[i];
            const float x1 = (src[i + 1u] - mean) * inv * w[i + 1u] + b[i + 1u];
            dst[i] = x0 * cos_t - x1 * sin_t;
            dst[i + 1u] = x0 * sin_t + x1 * cos_t;
        } else {
            dst[i] = (src[i] - mean) * inv * w[i] + b[i];
        }
    }
}

extern "C" int pulsar_idx_store_k(
        const void *raw_k_dev, const void *w_dev, const void *b_dev,
        void *key_cache_dev, uint32_t pos0, uint32_t n_tokens,
        uint32_t cache_cap, uint32_t head_dim, uint32_t rot_dim,
        uint32_t n_ctx_orig, float eps, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    if (!raw_k_dev || !w_dev || !b_dev || !key_cache_dev || n_tokens == 0 ||
        head_dim == 0 || rot_dim > head_dim || (rot_dim & 1u)) {
        return 0;
    }
    idx_store_k_kernel<<<n_tokens, 256>>>(
            (const float *)raw_k_dev, (const float *)w_dev, (const float *)b_dev,
            (float *)key_cache_dev, pos0, n_tokens, cache_cap, head_dim,
            rot_dim, n_ctx_orig, eps, freq_base, freq_scale, ext_factor,
            attn_factor, beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), "idx store k launch");
}

/* single-token scores over the whole visible cache */
__global__ static void idx_score_one_kernel(
        float *scores, const float *q, const float *weights,
        const float *key_cache, uint32_t n_rows, uint32_t n_head,
        uint32_t head_dim, float scale) {
    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;
    __shared__ float partial[256];
    float score = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + (uint64_t)h * head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += qh[d] * key_cache[(uint64_t)row * head_dim + d];
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) score += fmaxf(partial[0] * scale, 0.0f) * weights[h];
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[row] = score;
}

extern "C" int pulsar_idx_score_one(
        void *scores_dev, const void *q_dev, const void *weights_dev,
        const void *key_cache_dev, uint32_t n_rows, uint32_t n_head,
        uint32_t head_dim, float scale) {
    if (!scores_dev || !q_dev || !weights_dev || !key_cache_dev ||
        n_rows == 0 || n_head == 0 || head_dim == 0) {
        return 0;
    }
    idx_score_one_kernel<<<n_rows, 256>>>(
            (float *)scores_dev, (const float *)q_dev,
            (const float *)weights_dev, (const float *)key_cache_dev,
            n_rows, n_head, head_dim, scale);
    return cuda_ok(cudaGetLastError(), "idx score one launch");
}

/* bitonic top-k: deterministic tie-break by lower index */
__device__ __forceinline__ static bool idx_topk_better(
        float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

template <uint32_t SORT_N>
__global__ static void idx_topk_pow2_kernel(
        uint32_t *selected, const float *scores, uint32_t n_rows,
        uint32_t top_k) {
    const uint32_t tid = threadIdx.x;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < n_rows) {
            vals[i] = scores[i];
            idxs[i] = i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT32_MAX;
        }
    }
    __syncthreads();
    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                const uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i], bv = vals[other];
                    const uint32_t ai = idxs[i], bi = idxs[other];
                    const bool desc = (i & k) == 0u;
                    const bool swap = desc ? idx_topk_better(bv, bi, av, ai)
                                           : idx_topk_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv; idxs[i] = bi;
                        vals[other] = av; idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[i] = idxs[i];
    }
}

/* 8192-row variant with u16 indices: exactly fits the 48KB ceiling */
__global__ static void idx_topk_pow2_u16_kernel(
        uint32_t *selected, const float *scores, uint32_t n_rows,
        uint32_t top_k) {
    const uint32_t SORT_N = 8192;
    const uint32_t tid = threadIdx.x;
    __shared__ float vals[8192];
    __shared__ uint16_t idxs[8192];
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < n_rows) {
            vals[i] = scores[i];
            idxs[i] = (uint16_t)i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = 0xffffu;
        }
    }
    __syncthreads();
    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                const uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i], bv = vals[other];
                    const uint16_t ai = idxs[i], bi = idxs[other];
                    const bool desc = (i & k) == 0u;
                    const bool swap = desc ? idx_topk_better(bv, bi, av, ai)
                                           : idx_topk_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv; idxs[i] = bi;
                        vals[other] = av; idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[i] = idxs[i];
    }
}

extern "C" int pulsar_idx_topk(
        void *selected_dev, const void *scores_dev, uint32_t n_rows,
        uint32_t top_k) {
    if (!selected_dev || !scores_dev || n_rows == 0 || top_k == 0 ||
        top_k > n_rows) {
        return 0;
    }
    uint32_t sort_n = 1024;
    while (sort_n < n_rows) sort_n <<= 1u;
    uint32_t *sel = (uint32_t *)selected_dev;
    const float *sc = (const float *)scores_dev;
    switch (sort_n) {
    case 1024: idx_topk_pow2_kernel<1024><<<1, 1024>>>(sel, sc, n_rows, top_k); break;
    case 2048: idx_topk_pow2_kernel<2048><<<1, 1024>>>(sel, sc, n_rows, top_k); break;
    case 4096: idx_topk_pow2_kernel<4096><<<1, 1024>>>(sel, sc, n_rows, top_k); break;
    /* 8192 x (f32 + u16) = 48KB = the static shared-memory ceiling.
     * ponytail: ctx > 8192 needs a chunked/multi-block top-k; add it
     * when someone actually runs GLM past 8k context. */
    case 8192: idx_topk_pow2_u16_kernel<<<1, 1024>>>(sel, sc, n_rows, top_k); break;
    default:   return 0;
    }
    return cuda_ok(cudaGetLastError(), "idx topk launch");
}