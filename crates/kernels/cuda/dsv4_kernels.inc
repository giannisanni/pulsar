/* DeepSeek-V4-Flash (deepseek4) device kernels.
 *
 * Included by pulsar_kernels.cu AFTER gqa_kernels.inc: reuses its
 * e4m3_enc/e4m3_dec_common helpers and gqa_test_randf. Decode-only
 * shapes (n_tok == 1): the V4 state machines (raw SWA ring, streaming
 * compressor) are inherently sequential, so prefill loops tokens.
 *
 * Kernels:
 *  - hc_mix: hyper-connection stream combine (pre reduce, post expand,
 *    output-head merge - one kernel, coefficients by value)
 *  - dsv4_attention: sink-aware attention over raw ring + compressed
 *    rows, K == V (single 512-wide latent row per position)
 *  - dsv4_fp8_sim: ds4's e4m3 round-trip on the non-rope dims (64-wide
 *    blocks, power-of-2 scale, clamp +-448)
 *  - dsv4_f16_round: f16 storage round-trip (raw/comp cache pushes)
 */

/* Coefficients passed by value: out[j] = blk_c[j]*block[d] +
 * sum_src st_c[j*n_hc+src]*streams[src][d]. n_out<=4, n_hc<=4. */
typedef struct {
    float st_c[16];
    float blk_c[4];
} dsv4_hc_coef;

__global__ static void dsv4_hc_mix_kernel(
        float *out,
        const float *streams,
        const float *block, /* NULL = no block term */
        dsv4_hc_coef coef,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_out) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    float b = block ? block[d] : 0.0f;
    for (uint32_t j = 0; j < n_out; j++) {
        float acc = coef.blk_c[j] * b;
        for (uint32_t src = 0; src < n_hc; src++) {
            acc += coef.st_c[j * n_hc + src] * streams[(uint64_t)src * n_embd + d];
        }
        out[(uint64_t)j * n_embd + d] = acc;
    }
}

extern "C" int pulsar_dsv4_hc_mix(
        void *out_dev,
        const void *streams_dev,
        const void *block_dev, /* NULL ok */
        const float *st_c,     /* host [n_out*n_hc] */
        const float *blk_c,    /* host [n_out]; NULL = zeros */
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_out) {
    if (!out_dev || !streams_dev || !st_c || n_embd == 0 || n_hc == 0 ||
        n_hc > 4u || n_out == 0 || n_out > 4u) {
        return 0;
    }
    dsv4_hc_coef coef;
    memset(&coef, 0, sizeof(coef));
    memcpy(coef.st_c, st_c, (size_t)n_out * n_hc * sizeof(float));
    if (blk_c) memcpy(coef.blk_c, blk_c, (size_t)n_out * sizeof(float));
    dsv4_hc_mix_kernel<<<(n_embd + 255u) / 256u, 256>>>(
            (float *)out_dev, (const float *)streams_dev,
            (const float *)block_dev, coef, n_embd, n_hc, n_out);
    return cuda_ok(cudaGetLastError(), "dsv4 hc mix launch");
}

/* Sink-aware decode attention over raw SWA rows + compressed rows.
 * K and V are the SAME row. One block per head, 128 threads, online
 * softmax (gqa_attention_kernel's structure). The sink logit joins the
 * max and denominator with no value contribution. Raw rows live in a
 * ring (any order - softmax is permutation-invariant); comp rows are
 * append-ordered with an optional visibility mask. */
__global__ static void dsv4_attention_kernel(
        float *out,          /* [n_head][head_dim] */
        const float *q,      /* [n_head][head_dim] */
        const float *raw,    /* [raw_cap][head_dim] ring */
        uint32_t n_raw,
        const float *comp,   /* [>= n_comp][head_dim]; NULL if n_comp 0 */
        uint32_t n_comp,
        const uint8_t *allowed, /* [n_comp] 1 = visible; NULL = all */
        const float *sinks,  /* [n_head] */
        uint32_t head_dim,   /* <= 512 */
        float scale) {
    uint32_t h = blockIdx.x;
    const float *qh = q + (uint64_t)h * head_dim;
    const uint32_t tid = threadIdx.x;
    __shared__ float sq[512];
    __shared__ float s_w[128];
    __shared__ float red[128];
    for (uint32_t d = tid; d < head_dim; d += 128u) sq[d] = qh[d];
    __syncthreads();

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    /* sink as initial softmax state: m = sink, den = 1, no value */
    float m = sinks[h];
    float den = 1.0f;
    uint32_t n_total = n_raw + n_comp;
    for (uint32_t base = 0; base < n_total; base += 128u) {
        uint32_t chunk = n_total - base < 128u ? n_total - base : 128u;
        float s = -INFINITY;
        const float *srow = NULL;
        if (tid < chunk) {
            uint32_t r = base + tid;
            if (r < n_raw) {
                srow = raw + (uint64_t)r * head_dim;
            } else if (!allowed || allowed[r - n_raw]) {
                srow = comp + (uint64_t)(r - n_raw) * head_dim;
            }
            if (srow) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += sq[d] * srow[d];
                s = dot * scale;
            }
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
        if (tid < chunk) s_w[tid] = s == -INFINITY ? 0.0f : expf(s - m_new);
        __syncthreads();
        for (uint32_t r = 0; r < chunk; r++) {
            float w = s_w[r];
            if (w == 0.0f) continue;
            uint32_t row = base + r;
            const float *vrow = row < n_raw
                ? raw + (uint64_t)row * head_dim
                : comp + (uint64_t)(row - n_raw) * head_dim;
            den += w;
            for (uint32_t d = tid, k = 0; d < head_dim; d += 128u, k++) {
                acc[k] += w * vrow[d];
            }
        }
        m = m_new;
        __syncthreads();
    }
    for (uint32_t d = tid, k = 0; d < head_dim; d += 128u, k++) {
        out[(uint64_t)h * head_dim + d] = acc[k] / den;
    }
}

extern "C" int pulsar_dsv4_attention(
        void *out_dev,
        const void *q_dev,
        const void *raw_dev,
        uint32_t n_raw,
        const void *comp_dev,
        uint32_t n_comp,
        const void *allowed_dev, /* NULL = all comp rows visible */
        const void *sinks_dev,
        uint32_t n_head,
        uint32_t head_dim,
        float scale) {
    if (!out_dev || !q_dev || !raw_dev || !sinks_dev || n_head == 0 ||
        head_dim == 0 || head_dim > 512u || (n_comp > 0 && !comp_dev)) {
        return 0;
    }
    dsv4_attention_kernel<<<n_head, 128>>>(
            (float *)out_dev, (const float *)q_dev, (const float *)raw_dev,
            n_raw, (const float *)comp_dev, n_comp,
            (const uint8_t *)allowed_dev, (const float *)sinks_dev,
            head_dim, scale);
    return cuda_ok(cudaGetLastError(), "dsv4 attention launch");
}

/* ds4's dsv4_fp8_kv_quantize_row_inplace_cpu: the first head_dim-n_rot
 * dims of each row pass an e4m3fn round-trip in 64-wide blocks with a
 * power-of-2 scale from ceil(log2(amax/448)), clamped +-448. One warp
 * (32 threads) per 64-block. */
__global__ static void dsv4_fp8_sim_kernel(
        float *x,
        uint32_t n_rows,
        uint32_t head_dim,
        uint32_t n_nope) {
    uint32_t blk = blockIdx.x;           /* row * blocks_per_row + block */
    uint32_t blocks_per_row = n_nope / 64u;
    uint32_t row = blk / blocks_per_row;
    if (row >= n_rows) return;
    uint32_t off = (blk % blocks_per_row) * 64u;
    float *p = x + (uint64_t)row * head_dim + off;
    uint32_t lane = threadIdx.x;
    float a0 = fabsf(p[lane]);
    float a1 = fabsf(p[lane + 32u]);
    float amax = fmaxf(a0, a1);
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, o));
    }
    amax = __shfl_sync(0xffffffffu, amax, 0);
    if (amax < 1.0e-4f) amax = 1.0e-4f;
    const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
    #pragma unroll
    for (uint32_t k = 0; k < 2; k++) {
        uint32_t i = lane + k * 32u;
        float v = p[i] / scale;
        v = fminf(fmaxf(v, -448.0f), 448.0f);
        p[i] = e4m3_dec_common(e4m3_enc(v)) * scale;
    }
}

extern "C" int pulsar_dsv4_fp8_sim(
        void *x_dev, uint32_t n_rows, uint32_t head_dim, uint32_t n_rot) {
    if (!x_dev || n_rows == 0 || head_dim <= n_rot) return 0;
    uint32_t n_nope = head_dim - n_rot;
    if (n_nope % 64u != 0) return 0;
    uint32_t blocks = n_rows * (n_nope / 64u);
    dsv4_fp8_sim_kernel<<<blocks, 32>>>((float *)x_dev, n_rows, head_dim, n_nope);
    return cuda_ok(cudaGetLastError(), "dsv4 fp8 sim launch");
}

/* Round f32 values through f16 storage (ds4 caches rows as f16). */
__global__ static void dsv4_f16_round_kernel(float *x, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = __half2float(__float2half(x[i]));
}

extern "C" int pulsar_dsv4_f16_round(void *x_dev, uint32_t n) {
    if (!x_dev || n == 0) return 0;
    dsv4_f16_round_kernel<<<(n + 255u) / 256u, 256>>>((float *)x_dev, n);
    return cuda_ok(cudaGetLastError(), "dsv4 f16 round launch");
}

/* ---- selftest: CPU references vs the kernels --------------------------- */

static void dsv4_ref_attention(
        float *out, const float *q, const float *raw, uint32_t n_raw,
        const float *comp, uint32_t n_comp, const uint8_t *allowed,
        const float *sinks, uint32_t n_head, uint32_t head_dim, float scale) {
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + (uint64_t)h * head_dim;
        float m = sinks[h];
        uint32_t n_total = n_raw + n_comp;
        float *score = (float *)malloc(n_total * sizeof(float));
        for (uint32_t r = 0; r < n_total; r++) {
            const float *row;
            if (r < n_raw) {
                row = raw + (uint64_t)r * head_dim;
            } else if (allowed && !allowed[r - n_raw]) {
                score[r] = -INFINITY;
                continue;
            } else {
                row = comp + (uint64_t)(r - n_raw) * head_dim;
            }
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * row[d];
            score[r] = dot * scale;
            if (score[r] > m) m = score[r];
        }
        float *oh = out + (uint64_t)h * head_dim;
        memset(oh, 0, head_dim * sizeof(float));
        float den = expf(sinks[h] - m);
        for (uint32_t r = 0; r < n_total; r++) {
            if (score[r] == -INFINITY) continue;
            float w = expf(score[r] - m);
            const float *row = r < n_raw
                ? raw + (uint64_t)r * head_dim
                : comp + (uint64_t)(r - n_raw) * head_dim;
            den += w;
            for (uint32_t d = 0; d < head_dim; d++) oh[d] += w * row[d];
        }
        for (uint32_t d = 0; d < head_dim; d++) oh[d] /= den;
        free(score);
    }
}

/* Host e4m3fn round-trip via nearest-even value scan (e4m3_enc is
 * device-only: __float2int_rn). Byte values 0..126 decode to a
 * monotonically increasing positive ladder; even index parity == even
 * mantissa, so the tie rule below IS ties-to-even. */
static float dsv4_ref_e4m3_roundtrip(float v) {
    const float sign = v < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(v), 448.0f);
    int best = 0;
    float best_diff = fabsf(ax - e4m3_dec_common(0));
    for (int i = 1; i <= 126; i++) {
        const float d = fabsf(ax - e4m3_dec_common((uint8_t)i));
        if (d < best_diff || (d == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = d;
        }
    }
    return sign * e4m3_dec_common((uint8_t)best);
}

extern "C" int pulsar_dsv4_selftest(void) {
    const uint32_t n_head = 8, head_dim = 512, n_raw = 100, n_comp = 70;
    const float scale = 1.0f / sqrtf((float)head_dim);
    int ok = 1;

    /* attention with mask + sinks */
    {
        float *q = (float *)malloc((size_t)n_head * head_dim * 4);
        float *raw = (float *)malloc((size_t)n_raw * head_dim * 4);
        float *comp = (float *)malloc((size_t)n_comp * head_dim * 4);
        float *sinks = (float *)malloc(n_head * 4);
        uint8_t *allowed = (uint8_t *)malloc(n_comp);
        for (uint64_t i = 0; i < (uint64_t)n_head * head_dim; i++) q[i] = gqa_test_randf();
        for (uint64_t i = 0; i < (uint64_t)n_raw * head_dim; i++) raw[i] = gqa_test_randf();
        for (uint64_t i = 0; i < (uint64_t)n_comp * head_dim; i++) comp[i] = gqa_test_randf();
        for (uint32_t h = 0; h < n_head; h++) sinks[h] = gqa_test_randf() * 2.0f;
        for (uint32_t c = 0; c < n_comp; c++) allowed[c] = (c % 3u) != 0;

        float *ref = (float *)malloc((size_t)n_head * head_dim * 4);
        dsv4_ref_attention(ref, q, raw, n_raw, comp, n_comp, allowed, sinks,
                           n_head, head_dim, scale);

        void *dq, *draw, *dcomp, *dsinks, *dallowed, *dout;
        cudaMalloc(&dq, (size_t)n_head * head_dim * 4);
        cudaMalloc(&draw, (size_t)n_raw * head_dim * 4);
        cudaMalloc(&dcomp, (size_t)n_comp * head_dim * 4);
        cudaMalloc(&dsinks, n_head * 4);
        cudaMalloc(&dallowed, n_comp);
        cudaMalloc(&dout, (size_t)n_head * head_dim * 4);
        cudaMemcpy(dq, q, (size_t)n_head * head_dim * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(draw, raw, (size_t)n_raw * head_dim * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dcomp, comp, (size_t)n_comp * head_dim * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dsinks, sinks, n_head * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dallowed, allowed, n_comp, cudaMemcpyHostToDevice);
        ok &= pulsar_dsv4_attention(dout, dq, draw, n_raw, dcomp, n_comp,
                                    dallowed, dsinks, n_head, head_dim, scale);
        float *got = (float *)malloc((size_t)n_head * head_dim * 4);
        cudaMemcpy(got, dout, (size_t)n_head * head_dim * 4, cudaMemcpyDeviceToHost);
        float max_rel = 0.0f;
        for (uint64_t i = 0; i < (uint64_t)n_head * head_dim; i++) {
            float d = fabsf(got[i] - ref[i]);
            float r = d / fmaxf(fabsf(ref[i]), 1.0e-3f);
            if (r > max_rel) max_rel = r;
        }
        if (max_rel > 2.0e-4f) {
            fprintf(stderr, "dsv4 selftest: attention max rel err %.3e\n", max_rel);
            ok = 0;
        }
        /* no-mask, no-comp path */
        ok &= pulsar_dsv4_attention(dout, dq, draw, n_raw, NULL, 0,
                                    NULL, dsinks, n_head, head_dim, scale);
        dsv4_ref_attention(ref, q, raw, n_raw, NULL, 0, NULL, sinks,
                           n_head, head_dim, scale);
        cudaMemcpy(got, dout, (size_t)n_head * head_dim * 4, cudaMemcpyDeviceToHost);
        for (uint64_t i = 0; i < (uint64_t)n_head * head_dim; i++) {
            float d = fabsf(got[i] - ref[i]);
            float r = d / fmaxf(fabsf(ref[i]), 1.0e-3f);
            if (r > 2.0e-4f) { ok = 0; break; }
        }
        cudaFree(dq); cudaFree(draw); cudaFree(dcomp); cudaFree(dsinks);
        cudaFree(dallowed); cudaFree(dout);
        free(q); free(raw); free(comp); free(sinks); free(allowed);
        free(ref); free(got);
    }

    /* hc_mix: pre (n_out 1), post (n_out 4, block term) */
    {
        const uint32_t n_embd = 4096, n_hc = 4;
        float *streams = (float *)malloc((size_t)n_hc * n_embd * 4);
        float *block = (float *)malloc((size_t)n_embd * 4);
        for (uint64_t i = 0; i < (uint64_t)n_hc * n_embd; i++) streams[i] = gqa_test_randf();
        for (uint32_t i = 0; i < n_embd; i++) block[i] = gqa_test_randf();
        float st_c[16], blk_c[4];
        for (int i = 0; i < 16; i++) st_c[i] = gqa_test_randf();
        for (int i = 0; i < 4; i++) blk_c[i] = gqa_test_randf();

        void *dstreams, *dblock, *dout;
        cudaMalloc(&dstreams, (size_t)n_hc * n_embd * 4);
        cudaMalloc(&dblock, (size_t)n_embd * 4);
        cudaMalloc(&dout, (size_t)n_hc * n_embd * 4);
        cudaMemcpy(dstreams, streams, (size_t)n_hc * n_embd * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dblock, block, (size_t)n_embd * 4, cudaMemcpyHostToDevice);
        ok &= pulsar_dsv4_hc_mix(dout, dstreams, dblock, st_c, blk_c, n_embd, n_hc, 4);
        float *got = (float *)malloc((size_t)n_hc * n_embd * 4);
        cudaMemcpy(got, dout, (size_t)n_hc * n_embd * 4, cudaMemcpyDeviceToHost);
        for (uint32_t j = 0; j < n_hc && ok; j++) {
            for (uint32_t d = 0; d < n_embd; d++) {
                float want = blk_c[j] * block[d];
                for (uint32_t s = 0; s < n_hc; s++) {
                    want += st_c[j * n_hc + s] * streams[(uint64_t)s * n_embd + d];
                }
                if (fabsf(want - got[(uint64_t)j * n_embd + d]) > 1.0e-4f) {
                    fprintf(stderr, "dsv4 selftest: hc_mix mismatch\n");
                    ok = 0;
                    break;
                }
            }
        }
        cudaFree(dstreams); cudaFree(dblock); cudaFree(dout);
        free(streams); free(block); free(got);
    }

    /* fp8 sim vs CPU reference (same e4m3 helpers on both sides) */
    {
        const uint32_t rows = 3, hd = 512, n_rot = 64, n_nope = hd - n_rot;
        float *x = (float *)malloc((size_t)rows * hd * 4);
        for (uint64_t i = 0; i < (uint64_t)rows * hd; i++) x[i] = gqa_test_randf() * 8.0f;
        float *ref = (float *)malloc((size_t)rows * hd * 4);
        memcpy(ref, x, (size_t)rows * hd * 4);
        for (uint32_t r = 0; r < rows; r++) {
            float *p = ref + (uint64_t)r * hd;
            for (uint32_t off = 0; off < n_nope; off += 64u) {
                float amax = 0.0f;
                for (uint32_t i = 0; i < 64u; i++) amax = fmaxf(amax, fabsf(p[off + i]));
                if (amax < 1.0e-4f) amax = 1.0e-4f;
                float scale2 = exp2f(ceilf(log2f(amax / 448.0f)));
                for (uint32_t i = 0; i < 64u; i++) {
                    float v = p[off + i] / scale2;
                    v = fminf(fmaxf(v, -448.0f), 448.0f);
                    p[off + i] = dsv4_ref_e4m3_roundtrip(v) * scale2;
                }
            }
        }
        void *dx;
        cudaMalloc(&dx, (size_t)rows * hd * 4);
        cudaMemcpy(dx, x, (size_t)rows * hd * 4, cudaMemcpyHostToDevice);
        ok &= pulsar_dsv4_fp8_sim(dx, rows, hd, n_rot);
        float *got = (float *)malloc((size_t)rows * hd * 4);
        cudaMemcpy(got, dx, (size_t)rows * hd * 4, cudaMemcpyDeviceToHost);
        for (uint64_t i = 0; i < (uint64_t)rows * hd; i++) {
            if (got[i] != ref[i]) {
                fprintf(stderr, "dsv4 selftest: fp8 sim mismatch at %llu: %g vs %g\n",
                        (unsigned long long)i, got[i], ref[i]);
                ok = 0;
                break;
            }
        }
        cudaFree(dx);
        free(x); free(ref); free(got);
    }

    if (cudaDeviceSynchronize() != cudaSuccess) ok = 0;
    if (ok) fprintf(stderr, "dsv4 selftest: PASS\n");
    return ok;
}
