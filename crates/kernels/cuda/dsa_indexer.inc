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
/* Index K cache is stored as f16 by default: the scores only pick top-k
 * rows (the reference DSA indexer runs FP8 in production), and f16 keys
 * halve the cache and feed the tensor-core batch scorer directly. FP8=1
 * (rides PULSAR_KV=fp8 with the MLA latent) halves it again: e4m3 bytes
 * + one f32 row scale at the tail, stride head_dim+4, matching the
 * production-DSA precision. Row stride: */
__device__ static inline uint64_t idx_row_stride(uint32_t fp8, uint32_t head_dim) {
    return fp8 ? (uint64_t)head_dim + 4u : (uint64_t)head_dim * 2u;
}

template <uint32_t FP8>
__global__ static void idx_store_k_kernel(
        const float *raw_k, const float *w, const float *b,
        uint8_t *key_cache, uint32_t pos0, uint32_t n_tokens,
        uint32_t cache_cap, uint32_t head_dim, uint32_t rot_dim,
        uint32_t n_ctx_orig, float eps, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    const uint32_t token = blockIdx.x;
    if (token >= n_tokens) return;
    const uint32_t pos = pos0 + token;
    if (pos >= cache_cap) return;
    const float *src = raw_k + (uint64_t)token * head_dim;
    __shared__ float partial[256];
    /* fp8 needs the whole normed row before it can pick the scale; the
     * wrapper bounds head_dim to this staging size on the fp8 path */
    __shared__ float rowv[512];

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
    uint8_t *dst8 = key_cache + (uint64_t)pos * idx_row_stride(FP8, head_dim);
    __half *dst16 = (__half *)dst8;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        if (i < rot_dim) {
            if ((i & 1u) != 0u) continue;
            const float theta = (float)pos * powf(freq_base, -1.0f / (float)rot_dim * (float)i);
            float cos_t, sin_t;
            mla_rope_yarn(theta, freq_scale, corr0, corr1, (int)i,
                          ext_factor, attn_factor, &cos_t, &sin_t);
            const float x0 = (src[i] - mean) * inv * w[i] + b[i];
            const float x1 = (src[i + 1u] - mean) * inv * w[i + 1u] + b[i + 1u];
            const float y0 = x0 * cos_t - x1 * sin_t;
            const float y1 = x0 * sin_t + x1 * cos_t;
            if (FP8) {
                rowv[i] = y0;
                rowv[i + 1u] = y1;
            } else {
                dst16[i] = __float2half(y0);
                dst16[i + 1u] = __float2half(y1);
            }
        } else {
            const float y = (src[i] - mean) * inv * w[i] + b[i];
            if (FP8) rowv[i] = y;
            else dst16[i] = __float2half(y);
        }
    }
    if (FP8) {
        __syncthreads();
        float amax = 0.0f;
        for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x)
            amax = fmaxf(amax, fabsf(rowv[i]));
        partial[threadIdx.x] = amax;
        __syncthreads();
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s)
                partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + s]);
            __syncthreads();
        }
        const float rmax = partial[0];
        const float scale8 = rmax / 448.0f;
        const float qinv = rmax > 0.0f ? 448.0f / rmax : 0.0f;
        for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x)
            dst8[i] = e4m3_enc(rowv[i] * qinv);
        if (threadIdx.x == 0u) memcpy(dst8 + head_dim, &scale8, 4);
    }
}

extern "C" int pulsar_idx_store_k(
        const void *raw_k_dev, const void *w_dev, const void *b_dev,
        void *key_cache_dev, uint32_t pos0, uint32_t n_tokens,
        uint32_t cache_cap, uint32_t head_dim, uint32_t rot_dim,
        uint32_t n_ctx_orig, float eps, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast, float beta_slow,
        uint32_t fp8) {
    if (!raw_k_dev || !w_dev || !b_dev || !key_cache_dev || n_tokens == 0 ||
        head_dim == 0 || rot_dim > head_dim || (rot_dim & 1u) ||
        (fp8 && head_dim > 512u)) {
        return 0;
    }
    if (fp8) {
        idx_store_k_kernel<1u><<<n_tokens, 256>>>(
                (const float *)raw_k_dev, (const float *)w_dev, (const float *)b_dev,
                (uint8_t *)key_cache_dev, pos0, n_tokens, cache_cap, head_dim,
                rot_dim, n_ctx_orig, eps, freq_base, freq_scale, ext_factor,
                attn_factor, beta_fast, beta_slow);
    } else {
        idx_store_k_kernel<0u><<<n_tokens, 256>>>(
                (const float *)raw_k_dev, (const float *)w_dev, (const float *)b_dev,
                (uint8_t *)key_cache_dev, pos0, n_tokens, cache_cap, head_dim,
                rot_dim, n_ctx_orig, eps, freq_base, freq_scale, ext_factor,
                attn_factor, beta_fast, beta_slow);
    }
    return cuda_ok(cudaGetLastError(), "idx store k launch");
}

/* single-token scores over the whole visible cache. The fp8 row scale is
 * uniform per row, so it applies to the dot BEFORE the relu (the relu is
 * the reason it cannot fold into the epilogue weight). */
template <uint32_t FP8>
__global__ static void idx_score_one_kernel(
        float *scores, const float *q, const float *weights,
        const uint8_t *key_cache, uint32_t n_rows, uint32_t n_head,
        uint32_t head_dim, float scale) {
    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;
    __shared__ float partial[256];
    const uint8_t *krow = key_cache + (uint64_t)row * idx_row_stride(FP8, head_dim);
    float rs = 1.0f;
    if (FP8) memcpy(&rs, krow + head_dim, 4);
    float score = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + (uint64_t)h * head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += qh[d] * (FP8 ? e4m3_dec_common(krow[d])
                                : __half2float(((const __half *)krow)[d]));
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) score += fmaxf(partial[0] * rs * scale, 0.0f) * weights[h];
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[row] = score;
}

extern "C" int pulsar_idx_score_one(
        void *scores_dev, const void *q_dev, const void *weights_dev,
        const void *key_cache_dev, uint32_t n_rows, uint32_t n_head,
        uint32_t head_dim, float scale, uint32_t fp8) {
    if (!scores_dev || !q_dev || !weights_dev || !key_cache_dev ||
        n_rows == 0 || n_head == 0 || head_dim == 0) {
        return 0;
    }
    if (fp8) {
        idx_score_one_kernel<1u><<<n_rows, 256>>>(
                (float *)scores_dev, (const float *)q_dev,
                (const float *)weights_dev, (const uint8_t *)key_cache_dev,
                n_rows, n_head, head_dim, scale);
    } else {
        idx_score_one_kernel<0u><<<n_rows, 256>>>(
                (float *)scores_dev, (const float *)q_dev,
                (const float *)weights_dev, (const uint8_t *)key_cache_dev,
                n_rows, n_head, head_dim, scale);
    }
    return cuda_ok(cudaGetLastError(), "idx score one launch");
}

/* batch scores with causal masking: token t sees rows < pos0 + t + 1 */
template <uint32_t FP8>
__global__ static void idx_scores_batch_kernel(
        float *scores, const float *q, const float *weights,
        const uint8_t *key_cache, uint32_t n_rows, uint32_t n_tokens,
        uint32_t pos0, uint32_t n_head, uint32_t head_dim, float scale) {
    const uint32_t row = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (row >= n_rows || token >= n_tokens) return;
    float *dst = scores + (uint64_t)token * n_rows + row;
    const uint32_t visible = min(pos0 + token + 1u, n_rows);
    if (row >= visible) {
        if (threadIdx.x == 0) *dst = -INFINITY;
        return;
    }
    __shared__ float partial[256];
    const uint8_t *krow = key_cache + (uint64_t)row * idx_row_stride(FP8, head_dim);
    float rs = 1.0f;
    if (FP8) memcpy(&rs, krow + head_dim, 4);
    float score = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + ((uint64_t)token * n_head + h) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += qh[d] * (FP8 ? e4m3_dec_common(krow[d])
                                : __half2float(((const __half *)krow)[d]));
        }
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
            __syncthreads();
        }
        /* weights are PER TOKEN ([n_tok][n_head], projected from each
         * token's residual) - indexing [h] alone scored every token with
         * token 0's weights (pre-existing bug, fixed with the wmma port) */
        if (threadIdx.x == 0) score += fmaxf(partial[0] * rs * scale, 0.0f) * weights[(uint64_t)token * n_head + h];
        __syncthreads();
    }
    if (threadIdx.x == 0) *dst = score;
}


/* ---- tensor-core batch scorer (sm_80+) ----------------------------------
 * Per head: S_h = Q_h (n_tokens x d) x K^T (d x n_rows) via
 * mma.m16n8k16 f16 with f32 accumulate; the relu(qk*scale)*w[token][h]
 * epilogue runs in registers between heads (the nonlinearity is why this
 * cannot be one big fused GEMM). Block = 4 warps = a 16-token x 32-row
 * tile; K is read straight from the f16 cache, Q from a pre-converted f16
 * scratch. Falls back to the scalar kernel below sm_80 / PULSAR_NO_MMA. */
template <uint32_t FP8>
__global__ static void idx_scores_batch_mma_kernel(
        float *scores, const __half *q16, const float *weights,
        const uint8_t *key_cache, uint32_t n_rows, uint32_t n_tokens,
        uint32_t pos0, uint32_t n_head, uint32_t head_dim, float scale) {
#if __CUDA_ARCH__ >= 800
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t gid = lane >> 2;      /* fragment group row */
    const uint32_t tig = lane & 3u;      /* thread-in-group */
    const uint32_t tile_r0 = blockIdx.x * 32u + warp * 8u;
    const uint32_t tile_t0 = blockIdx.y * 16u;

    __shared__ __half As[16][16 + 2];         /* Q tile: shared by all warps */
    __shared__ __half Bs[4][16][8 + 2];       /* K tile: one slice per warp */

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (uint32_t h = 0; h < n_head; h++) {
        float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
        for (uint32_t ks = 0; ks < head_dim; ks += 16u) {
            /* stage A cooperatively: 256 halfs over 128 threads */
            for (uint32_t e = threadIdx.x; e < 256u; e += blockDim.x) {
                const uint32_t t = e >> 4, k = e & 15u;
                const uint32_t tok = tile_t0 + t;
                As[t][k] = tok < n_tokens
                    ? q16[((uint64_t)tok * n_head + h) * head_dim + ks + k]
                    : __float2half(0.0f);
            }
            /* stage this warp's B slice: 128 halfs over 32 lanes. fp8
             * dequantizes here (scale applied per row) so the mma stays
             * pure f16; the scale read is an L2-broadcast 4B hit. */
            for (uint32_t e = lane; e < 128u; e += 32u) {
                const uint32_t k = e >> 3, r = e & 7u;
                const uint32_t row = tile_r0 + r;
                if (row >= n_rows) {
                    Bs[warp][k][r] = __float2half(0.0f);
                } else if (FP8) {
                    const uint8_t *krow =
                        key_cache + (uint64_t)row * idx_row_stride(1u, head_dim);
                    float rs;
                    memcpy(&rs, krow + head_dim, 4);
                    Bs[warp][k][r] = __float2half(e4m3_dec_common(krow[ks + k]) * rs);
                } else {
                    Bs[warp][k][r] =
                        ((const __half *)(key_cache + (uint64_t)row * idx_row_stride(0u, head_dim)))[ks + k];
                }
            }
            __syncthreads();
            uint32_t a0, a1, a2, a3, b0, b1;
            a0 = *(const uint32_t *)&As[gid][2u * tig];
            a1 = *(const uint32_t *)&As[gid + 8u][2u * tig];
            a2 = *(const uint32_t *)&As[gid][2u * tig + 8u];
            a3 = *(const uint32_t *)&As[gid + 8u][2u * tig + 8u];
            {
                const __half2 p0 = __halves2half2(Bs[warp][2u * tig][gid], Bs[warp][2u * tig + 1u][gid]);
                const __half2 p1 = __halves2half2(Bs[warp][2u * tig + 8u][gid], Bs[warp][2u * tig + 9u][gid]);
                b0 = *(const uint32_t *)&p0;
                b1 = *(const uint32_t *)&p1;
            }
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            __syncthreads();
        }
        /* epilogue: c row = token (gid / gid+8), col = row (2tig / 2tig+1) */
        const uint32_t tok_lo = tile_t0 + gid;
        const uint32_t tok_hi = tok_lo + 8u;
        const float w_lo = tok_lo < n_tokens ? weights[(uint64_t)tok_lo * n_head + h] : 0.0f;
        const float w_hi = tok_hi < n_tokens ? weights[(uint64_t)tok_hi * n_head + h] : 0.0f;
        acc0 += w_lo * fmaxf(c0 * scale, 0.0f);
        acc1 += w_lo * fmaxf(c1 * scale, 0.0f);
        acc2 += w_hi * fmaxf(c2 * scale, 0.0f);
        acc3 += w_hi * fmaxf(c3 * scale, 0.0f);
    }

    const uint32_t col0 = tile_r0 + 2u * tig;
    const uint32_t tok_lo = tile_t0 + gid;
    const uint32_t tok_hi = tok_lo + 8u;
    float accs[4] = {acc0, acc1, acc2, acc3};
    #pragma unroll
    for (uint32_t e = 0; e < 4u; e++) {
        const uint32_t tok = (e < 2u) ? tok_lo : tok_hi;
        const uint32_t row = col0 + (e & 1u);
        if (tok >= n_tokens || row >= n_rows) continue;
        const uint32_t visible = min(pos0 + tok + 1u, n_rows);
        scores[(uint64_t)tok * n_rows + row] =
            row < visible ? accs[e] : -INFINITY;
    }
#else
    (void)scores; (void)q16; (void)weights; (void)key_cache; (void)n_rows;
    (void)n_tokens; (void)pos0; (void)n_head; (void)head_dim; (void)scale;
#endif
}

__global__ static void idx_f32_to_f16_kernel(__half *dst, const float *src, uint64_t n) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (uint64_t)gridDim.x * blockDim.x) {
        dst[i] = __float2half(src[i]);
    }
}

extern "C" int pulsar_idx_scores_batch(
        void *scores_dev, const void *q_dev, const void *weights_dev,
        const void *key_cache_dev, void *q16_scratch_dev,
        uint32_t n_rows, uint32_t n_tokens,
        uint32_t pos0, uint32_t n_head, uint32_t head_dim, float scale,
        uint32_t fp8) {
    if (!scores_dev || !q_dev || !weights_dev || !key_cache_dev ||
        n_rows == 0 || n_tokens == 0 || n_head == 0 || head_dim == 0) {
        return 0;
    }
    if (q16_scratch_dev && head_dim % 16u == 0 &&
        pulsar_device_cc_major() >= 8 && !getenv("PULSAR_NO_MMA")) {
        const uint64_t nq = (uint64_t)n_tokens * n_head * head_dim;
        idx_f32_to_f16_kernel<<<(uint32_t)((nq + 255u) / 256u), 256>>>(
                (__half *)q16_scratch_dev, (const float *)q_dev, nq);
        dim3 grid((n_rows + 31u) / 32u, (n_tokens + 15u) / 16u, 1);
        if (fp8) {
            idx_scores_batch_mma_kernel<1u><<<grid, 128>>>(
                    (float *)scores_dev, (const __half *)q16_scratch_dev,
                    (const float *)weights_dev, (const uint8_t *)key_cache_dev,
                    n_rows, n_tokens, pos0, n_head, head_dim, scale);
        } else {
            idx_scores_batch_mma_kernel<0u><<<grid, 128>>>(
                    (float *)scores_dev, (const __half *)q16_scratch_dev,
                    (const float *)weights_dev, (const uint8_t *)key_cache_dev,
                    n_rows, n_tokens, pos0, n_head, head_dim, scale);
        }
        return cuda_ok(cudaGetLastError(), "idx scores batch mma launch");
    }
    dim3 grid(n_rows, n_tokens, 1);
    if (fp8) {
        idx_scores_batch_kernel<1u><<<grid, 256>>>(
                (float *)scores_dev, (const float *)q_dev,
                (const float *)weights_dev, (const uint8_t *)key_cache_dev,
                n_rows, n_tokens, pos0, n_head, head_dim, scale);
    } else {
        idx_scores_batch_kernel<0u><<<grid, 256>>>(
                (float *)scores_dev, (const float *)q_dev,
                (const float *)weights_dev, (const uint8_t *)key_cache_dev,
                n_rows, n_tokens, pos0, n_head, head_dim, scale);
    }
    return cuda_ok(cudaGetLastError(), "idx scores batch launch");
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
/* ---- indexer selftest: f16 store + scalar/mma batch vs host reference --- */
static int idx_selftest_one(uint32_t fp8) {
    const uint32_t hd = 128, n_head = 4, rows = 48, cap = 64, n_tok = 5, pos0 = 40;
    const float eps = 1e-6f;
    const float scale = 1.0f / sqrtf((float)(hd * n_head));
    int all_ok = 1;

    float *raw = (float *)malloc((uint64_t)rows * hd * sizeof(float));
    float *w = (float *)malloc(hd * sizeof(float));
    float *b = (float *)malloc(hd * sizeof(float));
    float *q = (float *)malloc((uint64_t)n_tok * n_head * hd * sizeof(float));
    float *wt = (float *)malloc((uint64_t)n_tok * n_head * sizeof(float));
    for (uint64_t i = 0; i < (uint64_t)rows * hd; i++) raw[i] = gqa_test_randf();
    for (uint32_t i = 0; i < hd; i++) { w[i] = 0.5f + 0.5f * fabsf(gqa_test_randf()); b[i] = 0.2f * gqa_test_randf(); }
    for (uint64_t i = 0; i < (uint64_t)n_tok * n_head * hd; i++) q[i] = gqa_test_randf();
    /* per-token head weights, deliberately distinct per token: catches the
     * weights[h]-for-every-token bug class */
    for (uint32_t t = 0; t < n_tok; t++)
        for (uint32_t h = 0; h < n_head; h++)
            wt[t * n_head + h] = 0.1f + 0.3f * (float)t + 0.2f * (float)h;

    ds4_gpu_tensor *g_raw = ds4_gpu_tensor_alloc((uint64_t)rows * hd * sizeof(float));
    ds4_gpu_tensor *g_w = ds4_gpu_tensor_alloc(hd * sizeof(float));
    ds4_gpu_tensor *g_b = ds4_gpu_tensor_alloc(hd * sizeof(float));
    ds4_gpu_tensor *g_q = ds4_gpu_tensor_alloc((uint64_t)n_tok * n_head * hd * sizeof(float));
    ds4_gpu_tensor *g_q16 = ds4_gpu_tensor_alloc((uint64_t)n_tok * n_head * hd * sizeof(__half));
    ds4_gpu_tensor *g_wt = ds4_gpu_tensor_alloc((uint64_t)n_tok * n_head * sizeof(float));
    const uint64_t kc_stride = fp8 ? (uint64_t)hd + 4u : (uint64_t)hd * sizeof(__half);
    ds4_gpu_tensor *g_kc = ds4_gpu_tensor_alloc((uint64_t)cap * kc_stride);
    ds4_gpu_tensor *g_sc = ds4_gpu_tensor_alloc((uint64_t)n_tok * rows * sizeof(float));
    if (!g_raw || !g_w || !g_b || !g_q || !g_q16 || !g_wt || !g_kc || !g_sc) {
        fprintf(stderr, "idx-selftest: alloc failed\n");
        return 0;
    }
    ds4_gpu_tensor_write(g_raw, 0, raw, (uint64_t)rows * hd * sizeof(float));
    ds4_gpu_tensor_write(g_w, 0, w, hd * sizeof(float));
    ds4_gpu_tensor_write(g_b, 0, b, hd * sizeof(float));
    ds4_gpu_tensor_write(g_q, 0, q, (uint64_t)n_tok * n_head * hd * sizeof(float));
    ds4_gpu_tensor_write(g_wt, 0, wt, (uint64_t)n_tok * n_head * sizeof(float));

    /* store in two appends (0..30, 30..48), rot_dim 0 = pure layernorm */
    int ok = pulsar_idx_store_k(g_raw->ptr, g_w->ptr, g_b->ptr, g_kc->ptr,
                                0, 30, cap, hd, 0, 0, eps, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, fp8) &&
             pulsar_idx_store_k((const char *)g_raw->ptr + (uint64_t)30 * hd * sizeof(float),
                                g_w->ptr, g_b->ptr, g_kc->ptr,
                                30, rows - 30, cap, hd, 0, 0, eps, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, fp8) &&
             cuda_ok(cudaDeviceSynchronize(), "idx store sync");

    /* host: expected layernorm+affine keys; check the decoded cache
     * matches. fp8 tolerance is the e4m3 mantissa bound (~6.25% of the
     * row amax); the score references below read the DECODED rows, so
     * the score comparisons stay tight in both formats. */
    uint8_t *kc_raw = (uint8_t *)malloc((uint64_t)cap * kc_stride);
    float *keys = (float *)malloc((uint64_t)rows * hd * sizeof(float));
    if (ok) ok = ds4_gpu_tensor_read(g_kc, 0, kc_raw, (uint64_t)cap * kc_stride) != 0;
    float store_err = 0.0f;
    if (ok) {
        for (uint32_t r = 0; r < rows; r++) {
            const float *src = raw + (uint64_t)r * hd;
            double mean = 0.0, ss = 0.0;
            for (uint32_t i = 0; i < hd; i++) mean += src[i];
            mean /= hd;
            for (uint32_t i = 0; i < hd; i++) { double d = src[i] - mean; ss += d * d; }
            const float inv = 1.0f / sqrtf((float)(ss / hd) + eps);
            const uint8_t *krow = kc_raw + (uint64_t)r * kc_stride;
            float rs = 1.0f;
            if (fp8) memcpy(&rs, krow + hd, 4);
            for (uint32_t i = 0; i < hd; i++) {
                const float expect = ((float)(src[i] - mean)) * inv * w[i] + b[i];
                const float got = fp8 ? e4m3_dec_common(krow[i]) * rs
                                      : __half2float(((const __half *)krow)[i]);
                keys[(uint64_t)r * hd + i] = got; /* decoded cache = score ref */
                store_err = fmaxf(store_err, fabsf(got - expect));
            }
        }
    }
    const float store_tol = fp8 ? 3e-1f : 2e-3f;

    /* reference scores from the decoded f16 cache, per-token weights */
    float *ref = (float *)malloc((uint64_t)n_tok * rows * sizeof(float));
    for (uint32_t t = 0; t < n_tok; t++) {
        const uint32_t visible = min(pos0 + t + 1u, rows);
        for (uint32_t r = 0; r < rows; r++) {
            if (r >= visible) { ref[(uint64_t)t * rows + r] = -INFINITY; continue; }
            float score = 0.0f;
            for (uint32_t h = 0; h < n_head; h++) {
                const float *qh = q + ((uint64_t)t * n_head + h) * hd;
                float dot = 0.0f;
                for (uint32_t d = 0; d < hd; d++) dot += qh[d] * keys[(uint64_t)r * hd + d];
                score += fmaxf(dot * scale, 0.0f) * wt[t * n_head + h];
            }
            ref[(uint64_t)t * rows + r] = score;
        }
    }

    float *got = (float *)malloc((uint64_t)n_tok * rows * sizeof(float));
    const char *names[2] = {"scalar", "mma-or-scalar"};
    for (int mode = 0; mode < 2; mode++) {
        int mok = ok && pulsar_idx_scores_batch(g_sc->ptr, g_q->ptr, g_wt->ptr, g_kc->ptr,
                                                mode == 0 ? NULL : g_q16->ptr,
                                                rows, n_tok, pos0, n_head, hd, scale, fp8) &&
                  cuda_ok(cudaDeviceSynchronize(), "idx batch sync");
        float diff = INFINITY;
        if (mok && ds4_gpu_tensor_read(g_sc, 0, got, (uint64_t)n_tok * rows * sizeof(float))) {
            diff = 0.0f;
            for (uint64_t i = 0; i < (uint64_t)n_tok * rows; i++) {
                if (isinf(ref[i]) && isinf(got[i])) continue;
                diff = fmaxf(diff, fabsf(ref[i] - got[i]));
            }
        }
        /* mode 1 runs mma on sm_80+ (f16 q adds rounding); scalar is tight */
        const float tol = mode == 0 ? 2e-3f : 3e-2f;
        const int pass = mok && diff < tol && store_err < store_tol;
        fprintf(stderr, "idx-selftest %s%s: %s (max diff %.2e, store err %.2e)\n",
                names[mode], fp8 ? "-fp8" : "", pass ? "PASS" : "FAIL", diff, store_err);
        if (!pass) all_ok = 0;
    }

    /* decode path: score_one for the last token */
    {
        const uint32_t t = n_tok - 1;
        const uint32_t visible = min(pos0 + t + 1u, rows);
        int mok = ok && pulsar_idx_score_one(g_sc->ptr,
                        (const char *)g_q->ptr + (uint64_t)t * n_head * hd * sizeof(float),
                        (const char *)g_wt->ptr + (uint64_t)t * n_head * sizeof(float),
                        g_kc->ptr, visible, n_head, hd, scale, fp8) &&
                  cuda_ok(cudaDeviceSynchronize(), "idx one sync");
        float diff = INFINITY;
        if (mok && ds4_gpu_tensor_read(g_sc, 0, got, visible * sizeof(float))) {
            diff = 0.0f;
            for (uint32_t r = 0; r < visible; r++)
                diff = fmaxf(diff, fabsf(ref[(uint64_t)t * rows + r] - got[r]));
        }
        const int pass = mok && diff < 2e-3f;
        fprintf(stderr, "idx-selftest score_one%s: %s (max diff %.2e)\n", fp8 ? "-fp8" : "", pass ? "PASS" : "FAIL", diff);
        if (!pass) all_ok = 0;
    }

    free(raw); free(w); free(b); free(q); free(wt); free(kc_raw); free(keys); free(ref); free(got);
    ds4_gpu_tensor_free(g_raw); ds4_gpu_tensor_free(g_w); ds4_gpu_tensor_free(g_b);
    ds4_gpu_tensor_free(g_q); ds4_gpu_tensor_free(g_q16); ds4_gpu_tensor_free(g_wt);
    ds4_gpu_tensor_free(g_kc); ds4_gpu_tensor_free(g_sc);
    return all_ok;
}

extern "C" int pulsar_idx_selftest(void) {
    return idx_selftest_one(0u) && idx_selftest_one(1u);
}
