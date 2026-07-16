/* Qwen3.5/3.6 (qwen35moe) hybrid GDN kernels.
 *
 * Included by pulsar_kernels.cu after gqa_kernels.inc (reuses
 * gqa_test_randf). The Gated DeltaNet layers keep O(1) recurrent
 * state instead of KV: a depthwise conv window over the mixed qkv
 * projection and a per-head [128 x 128] delta-rule state. Decode-only
 * shapes (n_tok == 1); prefill loops tokens.
 *
 * Reference: llama.cpp delta-net-base.cpp build_delta_net_autoregressive
 * + qwen35moe.cpp build_layer_attn_linear.
 */

/* Depthwise causal conv (kernel size K, per channel) + SiLU over one
 * token. state [K-1][n_chan] rolls forward: window = [state | x].
 * kern layout matches gguf ssm_conv1d [K, n_chan]: taps contiguous
 * per channel, tap K-1 = current token. */
__global__ static void qwen35_conv_step_kernel(
        float *out,          /* [n_chan] */
        const float *x,      /* [n_chan] this token's qkv row */
        const float *kern,   /* [n_chan][K] */
        float *state,        /* [K-1][n_chan] rolling window */
        uint32_t n_chan,
        uint32_t k) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_chan) return;
    const float *kc = kern + (uint64_t)c * k;
    float acc = kc[k - 1] * x[c];
    for (uint32_t t = 0; t + 1 < k; t++) {
        acc += kc[t] * state[(uint64_t)t * n_chan + c];
    }
    /* roll: state rows shift left, x enters at the back */
    for (uint32_t t = 0; t + 2 < k; t++) {
        state[(uint64_t)t * n_chan + c] = state[(uint64_t)(t + 1) * n_chan + c];
    }
    state[(uint64_t)(k - 2) * n_chan + c] = x[c];
    out[c] = acc / (1.0f + expf(-acc)); /* silu */
}

extern "C" int pulsar_qwen35_conv_step(
        void *out_dev, const void *x_dev, const void *kern_dev,
        void *state_dev, uint32_t n_chan, uint32_t k) {
    if (!out_dev || !x_dev || !kern_dev || !state_dev || n_chan == 0 || k < 2) {
        return 0;
    }
    qwen35_conv_step_kernel<<<(n_chan + 255u) / 256u, 256>>>(
            (float *)out_dev, (const float *)x_dev, (const float *)kern_dev,
            (float *)state_dev, n_chan, k);
    return cuda_ok(cudaGetLastError(), "qwen35 conv step launch");
}

/* L2-normalize rows in place: x / sqrt(sum(x^2) + eps). NOT rms norm
 * (no 1/n, no weight) - ggml_l2_norm semantics. One block per row. */
__global__ static void qwen35_l2_norm_kernel(
        float *x, uint32_t rows, uint32_t dim, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    float *xr = x + (uint64_t)row * dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        sum += xr[i] * xr[i];
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) partial[threadIdx.x] += partial[threadIdx.x + st];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] + eps);
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        xr[i] *= scale;
    }
}

extern "C" int pulsar_qwen35_l2_norm(
        void *x_dev, uint32_t rows, uint32_t dim, float eps) {
    if (!x_dev || rows == 0 || dim == 0) return 0;
    qwen35_l2_norm_kernel<<<rows, 256>>>((float *)x_dev, rows, dim, eps);
    return cuda_ok(cudaGetLastError(), "qwen35 l2 norm launch");
}

/* Gated DeltaNet autoregressive step, one token, all heads in one
 * launch. Per v-head h with state S [dim x dim] (S[i][j]: i = k-dim,
 * j = v-dim), k/q from k-head h / (H_v / H_k):
 *   q_h *= 1/sqrt(dim)
 *   S   *= exp(g[h])
 *   sk_j = sum_i S[i][j] k_i
 *   d_j  = (v_j - sk_j) * beta[h]
 *   S[i][j] += k_i * d_j
 *   o_j  = sum_i S[i][j] q_i
 * One block per head, dim threads (thread j owns column j: S[i][j]
 * reads are coalesced across the block). dim <= 256. */
__global__ static void qwen35_gdn_step_kernel(
        float *out,          /* [H_v][dim] */
        float *state,        /* [H_v][dim][dim] in-place */
        const float *q,      /* [H_k][dim] (already conv+l2, pre-scale) */
        const float *k,      /* [H_k][dim] */
        const float *v,      /* [H_v][dim] */
        const float *g,      /* [H_v] log-decay (negative) */
        const float *beta,   /* [H_v] in (0,1) */
        uint32_t h_v,
        uint32_t h_k,
        uint32_t dim) {
    uint32_t h = blockIdx.x;
    uint32_t j = threadIdx.x;
    if (h >= h_v || j >= dim) return;
    /* ggml_repeat TILES along the head dim: v-head h reads k-head
     * h %% h_k (not the GQA-style block mapping h / (h_v/h_k)) */
    const uint32_t kh = h % h_k;
    const float *qh = q + (uint64_t)kh * dim;
    const float *kk = k + (uint64_t)kh * dim;
    float *S = state + (uint64_t)h * dim * dim;
    const float decay = expf(g[h]);
    const float qscale = rsqrtf((float)dim);

    __shared__ float sq[256];
    __shared__ float sk_sh[256];
    for (uint32_t i = j; i < dim; i += blockDim.x) {
        sq[i] = qh[i] * qscale;
        sk_sh[i] = kk[i];
    }
    __syncthreads();

    /* decay + sk in one pass down column j */
    float sk = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float s = S[(uint64_t)i * dim + j] * decay;
        S[(uint64_t)i * dim + j] = s;
        sk += s * sk_sh[i];
    }
    const float d = (v[(uint64_t)h * dim + j] - sk) * beta[h];
    /* rank-1 update + output in a second pass */
    float o = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float s = S[(uint64_t)i * dim + j] + sk_sh[i] * d;
        S[(uint64_t)i * dim + j] = s;
        o += s * sq[i];
    }
    out[(uint64_t)h * dim + j] = o;
}

extern "C" int pulsar_qwen35_gdn_step(
        void *out_dev, void *state_dev,
        const void *q_dev, const void *k_dev, const void *v_dev,
        const void *g_dev, const void *beta_dev,
        uint32_t h_v, uint32_t h_k, uint32_t dim) {
    if (!out_dev || !state_dev || !q_dev || !k_dev || !v_dev || !g_dev ||
        !beta_dev || h_v == 0 || h_k == 0 || h_v % h_k != 0 || dim == 0 ||
        dim > 256u) {
        return 0;
    }
    qwen35_gdn_step_kernel<<<h_v, dim>>>(
            (float *)out_dev, (float *)state_dev,
            (const float *)q_dev, (const float *)k_dev, (const float *)v_dev,
            (const float *)g_dev, (const float *)beta_dev, h_v, h_k, dim);
    return cuda_ok(cudaGetLastError(), "qwen35 gdn step launch");
}

/* Deinterleave the attention layers' fused q projection: input rows
 * are per-head [q dim | gate dim]; outputs are contiguous q and gate. */
__global__ static void qwen35_split_gate_kernel(
        float *q, float *gate, const float *fused,
        uint32_t n_head, uint32_t dim) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = n_head * dim;
    if (idx >= total) return;
    uint32_t h = idx / dim;
    uint32_t d = idx % dim;
    q[idx] = fused[(uint64_t)h * 2 * dim + d];
    gate[idx] = fused[(uint64_t)h * 2 * dim + dim + d];
}

extern "C" int pulsar_qwen35_split_gate(
        void *q_dev, void *gate_dev, const void *fused_dev,
        uint32_t n_head, uint32_t dim) {
    if (!q_dev || !gate_dev || !fused_dev || n_head == 0 || dim == 0) return 0;
    uint32_t total = n_head * dim;
    qwen35_split_gate_kernel<<<(total + 255u) / 256u, 256>>>(
            (float *)q_dev, (float *)gate_dev, (const float *)fused_dev,
            n_head, dim);
    return cuda_ok(cudaGetLastError(), "qwen35 split gate launch");
}

/* out *= sigmoid(gate) elementwise (attention output gating; also the
 * shared expert's scalar gate with gate broadcast handled host-side). */
__global__ static void qwen35_sigmoid_gate_kernel(
        float *x, const float *gate, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] *= 1.0f / (1.0f + expf(-gate[i]));
}

extern "C" int pulsar_qwen35_sigmoid_gate(
        void *x_dev, const void *gate_dev, uint32_t n) {
    if (!x_dev || !gate_dev || n == 0) return 0;
    qwen35_sigmoid_gate_kernel<<<(n + 255u) / 256u, 256>>>(
            (float *)x_dev, (const float *)gate_dev, n);
    return cuda_ok(cudaGetLastError(), "qwen35 sigmoid gate launch");
}

/* ---- selftest ----------------------------------------------------------- */

extern "C" int pulsar_qwen35_selftest(void) {
    int ok = 1;
    const uint32_t DIM = 128, HV = 8, HK = 4, NCH = 512, K = 4;

    /* conv step vs host reference, 3 sequential tokens */
    {
        float *kern = (float *)malloc((size_t)NCH * K * 4);
        float *state = (float *)calloc((size_t)(K - 1) * NCH, 4);
        float *state_ref = (float *)calloc((size_t)(K - 1) * NCH, 4);
        for (uint32_t i = 0; i < NCH * K; i++) kern[i] = gqa_test_randf();
        void *dkern, *dstate, *dx, *dout;
        cudaMalloc(&dkern, (size_t)NCH * K * 4);
        cudaMalloc(&dstate, (size_t)(K - 1) * NCH * 4);
        cudaMalloc(&dx, NCH * 4);
        cudaMalloc(&dout, NCH * 4);
        cudaMemcpy(dkern, kern, (size_t)NCH * K * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dstate, state, (size_t)(K - 1) * NCH * 4, cudaMemcpyHostToDevice);
        float *x = (float *)malloc(NCH * 4);
        float *got = (float *)malloc(NCH * 4);
        for (int step = 0; step < 3 && ok; step++) {
            for (uint32_t c = 0; c < NCH; c++) x[c] = gqa_test_randf();
            cudaMemcpy(dx, x, NCH * 4, cudaMemcpyHostToDevice);
            ok &= pulsar_qwen35_conv_step(dout, dx, dkern, dstate, NCH, K);
            cudaMemcpy(got, dout, NCH * 4, cudaMemcpyDeviceToHost);
            for (uint32_t c = 0; c < NCH && ok; c++) {
                float acc = kern[c * K + K - 1] * x[c];
                for (uint32_t t = 0; t + 1 < K; t++) {
                    acc += kern[c * K + t] * state_ref[t * NCH + c];
                }
                float want = acc / (1.0f + expf(-acc));
                if (fabsf(want - got[c]) > 1.0e-5f) {
                    fprintf(stderr, "qwen35 selftest: conv mismatch step %d ch %u\n", step, c);
                    ok = 0;
                }
            }
            for (uint32_t t = 0; t + 2 < K; t++) {
                memcpy(state_ref + t * NCH, state_ref + (t + 1) * NCH, NCH * 4);
            }
            memcpy(state_ref + (K - 2) * NCH, x, NCH * 4);
        }
        cudaFree(dkern); cudaFree(dstate); cudaFree(dx); cudaFree(dout);
        free(kern); free(state); free(state_ref); free(x); free(got);
    }

    /* gdn step vs host reference, 3 sequential tokens */
    {
        const uint64_t sbytes = (uint64_t)HV * DIM * DIM * 4;
        float *S_ref = (float *)calloc((size_t)HV * DIM * DIM, 4);
        float *q = (float *)malloc((size_t)HK * DIM * 4);
        float *k = (float *)malloc((size_t)HK * DIM * 4);
        float *v = (float *)malloc((size_t)HV * DIM * 4);
        float g[HV], beta[HV];
        void *dS, *dq, *dk, *dv, *dg, *dbeta, *dout;
        cudaMalloc(&dS, sbytes);
        cudaMemset(dS, 0, sbytes);
        cudaMalloc(&dq, (size_t)HK * DIM * 4);
        cudaMalloc(&dk, (size_t)HK * DIM * 4);
        cudaMalloc(&dv, (size_t)HV * DIM * 4);
        cudaMalloc(&dg, HV * 4);
        cudaMalloc(&dbeta, HV * 4);
        cudaMalloc(&dout, (size_t)HV * DIM * 4);
        float *got = (float *)malloc((size_t)HV * DIM * 4);
        for (int step = 0; step < 3 && ok; step++) {
            for (uint32_t i = 0; i < HK * DIM; i++) { q[i] = gqa_test_randf(); k[i] = gqa_test_randf(); }
            for (uint32_t i = 0; i < HV * DIM; i++) v[i] = gqa_test_randf();
            for (uint32_t h = 0; h < HV; h++) {
                g[h] = -fabsf(gqa_test_randf());          /* log-decay < 0 */
                beta[h] = 0.5f + 0.4f * gqa_test_randf(); /* ~(0,1) */
            }
            cudaMemcpy(dq, q, (size_t)HK * DIM * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(dk, k, (size_t)HK * DIM * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(dv, v, (size_t)HV * DIM * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(dg, g, HV * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(dbeta, beta, HV * 4, cudaMemcpyHostToDevice);
            ok &= pulsar_qwen35_gdn_step(dout, dS, dq, dk, dv, dg, dbeta, HV, HK, DIM);
            cudaMemcpy(got, dout, (size_t)HV * DIM * 4, cudaMemcpyDeviceToHost);
            for (uint32_t h = 0; h < HV && ok; h++) {
                const float *qh = q + (h % HK) * DIM;
                const float *kh = k + (h % HK) * DIM;
                float *S = S_ref + (uint64_t)h * DIM * DIM;
                const float decay = expf(g[h]);
                const float qs = 1.0f / sqrtf((float)DIM);
                for (uint32_t i = 0; i < DIM * DIM; i++) S[i] *= decay;
                for (uint32_t j = 0; j < DIM; j++) {
                    float sk = 0.0f;
                    for (uint32_t i = 0; i < DIM; i++) sk += S[i * DIM + j] * kh[i];
                    float d = (v[h * DIM + j] - sk) * beta[h];
                    float o = 0.0f;
                    for (uint32_t i = 0; i < DIM; i++) {
                        S[i * DIM + j] += kh[i] * d;
                        o += S[i * DIM + j] * qh[i] * qs;
                    }
                    float diff = fabsf(o - got[h * DIM + j]);
                    if (diff > 5.0e-4f * fmaxf(1.0f, fabsf(o))) {
                        fprintf(stderr,
                                "qwen35 selftest: gdn mismatch step %d h %u j %u: %g vs %g\n",
                                step, h, j, o, got[h * DIM + j]);
                        ok = 0;
                        break;
                    }
                }
            }
        }
        cudaFree(dS); cudaFree(dq); cudaFree(dk); cudaFree(dv);
        cudaFree(dg); cudaFree(dbeta); cudaFree(dout);
        free(S_ref); free(q); free(k); free(v); free(got);
    }

    /* l2 norm + split gate + sigmoid gate quick checks */
    {
        const uint32_t R = 4, D = 128;
        float *x = (float *)malloc((size_t)R * D * 4);
        for (uint32_t i = 0; i < R * D; i++) x[i] = gqa_test_randf() * 3.0f;
        void *dx;
        cudaMalloc(&dx, (size_t)R * D * 4);
        cudaMemcpy(dx, x, (size_t)R * D * 4, cudaMemcpyHostToDevice);
        ok &= pulsar_qwen35_l2_norm(dx, R, D, 1.0e-6f);
        float *got = (float *)malloc((size_t)R * D * 4);
        cudaMemcpy(got, dx, (size_t)R * D * 4, cudaMemcpyDeviceToHost);
        for (uint32_t r = 0; r < R && ok; r++) {
            float ss = 0.0f;
            for (uint32_t i = 0; i < D; i++) ss += x[r * D + i] * x[r * D + i];
            float sc = 1.0f / sqrtf(ss + 1.0e-6f);
            for (uint32_t i = 0; i < D; i++) {
                if (fabsf(x[r * D + i] * sc - got[r * D + i]) > 1.0e-5f) {
                    fprintf(stderr, "qwen35 selftest: l2 norm mismatch\n");
                    ok = 0;
                    break;
                }
            }
        }
        cudaFree(dx); free(x); free(got);

        const uint32_t H = 4, HD = 64;
        float *fused = (float *)malloc((size_t)H * 2 * HD * 4);
        for (uint32_t i = 0; i < H * 2 * HD; i++) fused[i] = (float)i;
        void *dfused, *dq2, *dgate;
        cudaMalloc(&dfused, (size_t)H * 2 * HD * 4);
        cudaMalloc(&dq2, (size_t)H * HD * 4);
        cudaMalloc(&dgate, (size_t)H * HD * 4);
        cudaMemcpy(dfused, fused, (size_t)H * 2 * HD * 4, cudaMemcpyHostToDevice);
        ok &= pulsar_qwen35_split_gate(dq2, dgate, dfused, H, HD);
        float *gq = (float *)malloc((size_t)H * HD * 4);
        float *gg = (float *)malloc((size_t)H * HD * 4);
        cudaMemcpy(gq, dq2, (size_t)H * HD * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(gg, dgate, (size_t)H * HD * 4, cudaMemcpyDeviceToHost);
        for (uint32_t h = 0; h < H && ok; h++) {
            for (uint32_t d = 0; d < HD; d++) {
                if (gq[h * HD + d] != fused[h * 2 * HD + d] ||
                    gg[h * HD + d] != fused[h * 2 * HD + HD + d]) {
                    fprintf(stderr, "qwen35 selftest: split gate mismatch\n");
                    ok = 0;
                    break;
                }
            }
        }
        ok &= pulsar_qwen35_sigmoid_gate(dq2, dgate, H * HD);
        float *gs = (float *)malloc((size_t)H * HD * 4);
        cudaMemcpy(gs, dq2, (size_t)H * HD * 4, cudaMemcpyDeviceToHost);
        for (uint32_t i = 0; i < H * HD && ok; i++) {
            float want = gq[i] / (1.0f + expf(-gg[i]));
            if (fabsf(want - gs[i]) > 1.0e-4f * fmaxf(1.0f, fabsf(want))) {
                fprintf(stderr, "qwen35 selftest: sigmoid gate mismatch\n");
                ok = 0;
            }
        }
        cudaFree(dfused); cudaFree(dq2); cudaFree(dgate);
        free(fused); free(gq); free(gg); free(gs);
    }

    if (cudaDeviceSynchronize() != cudaSuccess) ok = 0;
    if (ok) fprintf(stderr, "qwen35 selftest: PASS\n");
    return ok;
}
