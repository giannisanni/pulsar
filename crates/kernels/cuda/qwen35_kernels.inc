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


/* Batched variants for chain verify / chunked prefill: n_tok tokens
 * processed SEQUENTIALLY inside one launch (the recurrences are
 * per-token, but looping on-device removes 2*n_tok launch round-trips
 * per layer and keeps the working set in registers). */

/* conv+silu over n_tok tokens: thread owns a channel, window rides in
 * registers (k <= 8). x/out are [n_tok][n_chan]; state as in the
 * single-step kernel and left at the post-batch value. */
__global__ static void qwen35_conv_batch_kernel(
        float *out, const float *x, const float *kern, float *state,
        uint32_t n_chan, uint32_t k, uint32_t n_tok) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_chan) return;
    const float *kc = kern + (uint64_t)c * k;
    float win[8];
    for (uint32_t t = 0; t + 1 < k; t++) {
        win[t] = state[(uint64_t)t * n_chan + c];
    }
    for (uint32_t tok = 0; tok < n_tok; tok++) {
        const float cur = x[(uint64_t)tok * n_chan + c];
        float acc = kc[k - 1] * cur;
        for (uint32_t t = 0; t + 1 < k; t++) acc += kc[t] * win[t];
        out[(uint64_t)tok * n_chan + c] = acc / (1.0f + expf(-acc));
        for (uint32_t t = 0; t + 2 < k; t++) win[t] = win[t + 1];
        win[k - 2] = cur;
    }
    for (uint32_t t = 0; t + 1 < k; t++) {
        state[(uint64_t)t * n_chan + c] = win[t];
    }
}

extern "C" int pulsar_qwen35_conv_batch(
        void *out_dev, const void *x_dev, const void *kern_dev,
        void *state_dev, uint32_t n_chan, uint32_t k, uint32_t n_tok) {
    if (!out_dev || !x_dev || !kern_dev || !state_dev || n_chan == 0 ||
        k < 2 || k > 8 || n_tok == 0) {
        return 0;
    }
    qwen35_conv_batch_kernel<<<(n_chan + 255u) / 256u, 256>>>(
            (float *)out_dev, (const float *)x_dev, (const float *)kern_dev,
            (float *)state_dev, n_chan, k, n_tok);
    return cuda_ok(cudaGetLastError(), "qwen35 conv batch launch");
}

/* GDN delta rule over n_tok tokens: thread j holds its state COLUMN
 * S[:,j] in registers (dim <= 128 for the register budget) across the
 * whole batch; k/q broadcast through shared memory per step. Layouts:
 * q/k [n_tok][H_k][dim], v [n_tok][H_v][dim], g/beta [n_tok][H_v],
 * out [n_tok][H_v][dim]. State read once, written once. */
__global__ static void qwen35_gdn_batch_kernel(
        float *out, float *state,
        const float *q, const float *k, const float *v,
        const float *g, const float *beta,
        uint32_t h_v, uint32_t h_k, uint32_t dim, uint32_t n_tok) {
    uint32_t h = blockIdx.x;
    uint32_t j = threadIdx.x;
    if (h >= h_v || j >= dim) return;
    const uint32_t kh = h % h_k; /* ggml_repeat tiles */
    float *Sg = state + (uint64_t)h * dim * dim;
    const float qscale = rsqrtf((float)dim);

    float S[128];
    for (uint32_t i = 0; i < dim; i++) S[i] = Sg[(uint64_t)i * dim + j];

    __shared__ float sq[128];
    __shared__ float sk[128];
    for (uint32_t tok = 0; tok < n_tok; tok++) {
        const float *qh = q + ((uint64_t)tok * h_k + kh) * dim;
        const float *kk = k + ((uint64_t)tok * h_k + kh) * dim;
        __syncthreads();
        for (uint32_t i = j; i < dim; i += blockDim.x) {
            sq[i] = qh[i] * qscale;
            sk[i] = kk[i];
        }
        __syncthreads();
        const float decay = expf(g[(uint64_t)tok * h_v + h]);
        float dot_k = 0.0f;
        for (uint32_t i = 0; i < dim; i++) {
            S[i] *= decay;
            dot_k += S[i] * sk[i];
        }
        const float d =
            (v[((uint64_t)tok * h_v + h) * dim + j] - dot_k) *
            beta[(uint64_t)tok * h_v + h];
        float o = 0.0f;
        for (uint32_t i = 0; i < dim; i++) {
            S[i] += sk[i] * d;
            o += S[i] * sq[i];
        }
        out[((uint64_t)tok * h_v + h) * dim + j] = o;
    }
    for (uint32_t i = 0; i < dim; i++) Sg[(uint64_t)i * dim + j] = S[i];
}

extern "C" int pulsar_qwen35_gdn_batch(
        void *out_dev, void *state_dev,
        const void *q_dev, const void *k_dev, const void *v_dev,
        const void *g_dev, const void *beta_dev,
        uint32_t h_v, uint32_t h_k, uint32_t dim, uint32_t n_tok) {
    if (!out_dev || !state_dev || !q_dev || !k_dev || !v_dev || !g_dev ||
        !beta_dev || h_v == 0 || h_k == 0 || h_v % h_k != 0 || dim == 0 ||
        dim > 128u || n_tok == 0) {
        return 0;
    }
    qwen35_gdn_batch_kernel<<<h_v, dim>>>(
            (float *)out_dev, (float *)state_dev,
            (const float *)q_dev, (const float *)k_dev, (const float *)v_dev,
            (const float *)g_dev, (const float *)beta_dev, h_v, h_k, dim, n_tok);
    return cuda_ok(cudaGetLastError(), "qwen35 gdn batch launch");
}

/* Per-row scale: x[t][:] *= s[t] (batched shared-expert scalar gates). */
__global__ static void qwen35_row_scale_kernel(
        float *x, const float *s, uint32_t n_rows, uint32_t dim) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows * dim) return;
    x[idx] *= s[idx / dim];
}

extern "C" int pulsar_qwen35_row_scale(
        void *x_dev, const void *s_dev, uint32_t n_rows, uint32_t dim) {
    if (!x_dev || !s_dev || n_rows == 0 || dim == 0) return 0;
    uint32_t total = n_rows * dim;
    qwen35_row_scale_kernel<<<(total + 255u) / 256u, 256>>>(
            (float *)x_dev, (const float *)s_dev, n_rows, dim);
    return cuda_ok(cudaGetLastError(), "qwen35 row scale launch");
}

/* Non-causal draft attention (DFlash block diffusion): n_q query rows
 * attend over ALL n_kv rows. q [n_q][n_head][dim]; k/v contiguous rows
 * [n_kv][n_kv_head][dim]; GQA block mapping (flash_attn_ext semantics:
 * head h reads kv head h / (n_head/n_kv_head) - NOT the GDN tile map).
 * One block per (q row, head), online softmax over row chunks. */
__global__ static void qwen35_draft_attn_kernel(
        float *out, const float *q, const float *k, const float *v,
        uint32_t n_q, uint32_t n_kv, uint32_t n_head, uint32_t n_kv_head,
        uint32_t dim, float scale) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_q || h >= n_head) return;
    const uint32_t kvh = h / (n_head / n_kv_head);
    const float *qh = q + ((uint64_t)t * n_head + h) * dim;
    const uint32_t tid = threadIdx.x;
    __shared__ float sq[128];
    __shared__ float s_w[128];
    __shared__ float red[128];
    for (uint32_t d = tid; d < dim; d += 128u) sq[d] = qh[d];
    __syncthreads();
    float acc = 0.0f; /* dim <= 128: thread d owns output dim d */
    float den = 0.0f;
    float m = -INFINITY;
    const uint64_t kv_stride = (uint64_t)n_kv_head * dim;
    for (uint32_t base = 0; base < n_kv; base += 128u) {
        uint32_t chunk = n_kv - base < 128u ? n_kv - base : 128u;
        float s = -INFINITY;
        if (tid < chunk) {
            const float *krow = k + (uint64_t)(base + tid) * kv_stride + (uint64_t)kvh * dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < dim; d++) dot += sq[d] * krow[d];
            s = dot * scale;
        }
        red[tid] = s;
        __syncthreads();
        for (uint32_t st = 64u; st != 0; st >>= 1u) {
            if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
            __syncthreads();
        }
        float m_new = fmaxf(m, red[0]);
        float corr = expf(m - m_new);
        acc *= corr;
        den *= corr;
        if (tid < chunk) s_w[tid] = expf(s - m_new);
        __syncthreads();
        for (uint32_t r = 0; r < chunk; r++) {
            float wgt = s_w[r];
            den += wgt;
            if (tid < dim) {
                acc += wgt * v[(uint64_t)(base + r) * kv_stride + (uint64_t)kvh * dim + tid];
            }
        }
        m = m_new;
        __syncthreads();
    }
    if (tid < dim) {
        out[((uint64_t)t * n_head + h) * dim + tid] = acc / den;
    }
}

extern "C" int pulsar_qwen35_draft_attn(
        void *out_dev, const void *q_dev, const void *k_dev, const void *v_dev,
        uint32_t n_q, uint32_t n_kv, uint32_t n_head, uint32_t n_kv_head,
        uint32_t dim, float scale) {
    if (!out_dev || !q_dev || !k_dev || !v_dev || n_q == 0 || n_kv == 0 ||
        n_head == 0 || n_kv_head == 0 || n_head % n_kv_head != 0 ||
        dim == 0 || dim > 128u) {
        return 0;
    }
    dim3 grid(n_q, n_head, 1);
    qwen35_draft_attn_kernel<<<grid, 128>>>(
            (float *)out_dev, (const float *)q_dev, (const float *)k_dev,
            (const float *)v_dev, n_q, n_kv, n_head, n_kv_head, dim, scale);
    return cuda_ok(cudaGetLastError(), "qwen35 draft attn launch");
}

/* NEOX-paired YaRN rope (DFlash draft: trained with rope_scaling yarn
 * factor 64 / orig 4096 on theta 10M - plain rope smears its positions
 * 64x). Pair (i, i+half); cos/sin from the mla yarn helpers (ggml
 * semantics: pass attn_factor 1.0 and the kernel-internal
 * 1 + 0.1 ln(1/freq_scale) supplies the HF mscale). */
__global__ static void qwen35_rope_yarn_kernel(
        float *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
        uint32_t pos0, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast,
        float beta_slow, uint32_t n_ctx_orig) {
    const uint32_t half = head_dim / 2u;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pairs = n_tok * n_head * half;
    if (gid >= pairs) return;
    const uint32_t p = gid % half;
    const uint32_t tmp = gid / half;
    const uint32_t h = tmp % n_head;
    const uint32_t tok = tmp / n_head;
    float c0 = 0.0f, c1 = 0.0f;
    if (ext_factor != 0.0f) {
        mla_rope_corr_dims(head_dim, n_ctx_orig, freq_base,
                           beta_fast, beta_slow, &c0, &c1);
    }
    const float theta_extrap = (float)(pos0 + tok) *
        powf(freq_base, -2.0f * (float)p / (float)head_dim);
    float cs, sn;
    mla_rope_yarn(theta_extrap, freq_scale, c0, c1, (int)(2u * p),
                  ext_factor, attn_factor, &cs, &sn);
    float *row = x + ((uint64_t)tok * n_head + h) * head_dim;
    const float x0 = row[p];
    const float x1 = row[p + half];
    row[p] = x0 * cs - x1 * sn;
    row[p + half] = x0 * sn + x1 * cs;
}

extern "C" int pulsar_qwen35_rope_yarn(
        void *x_dev, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
        uint32_t pos0, float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow,
        uint32_t n_ctx_orig) {
    if (!x_dev || n_tok == 0 || n_head == 0 || head_dim == 0 ||
        (head_dim & 1u) != 0) {
        return 0;
    }
    const uint32_t pairs = n_tok * n_head * (head_dim / 2u);
    qwen35_rope_yarn_kernel<<<(pairs + 255u) / 256u, 256>>>(
            (float *)x_dev, n_tok, n_head, head_dim, pos0, freq_base,
            freq_scale, ext_factor, attn_factor, beta_fast, beta_slow,
            n_ctx_orig);
    return cuda_ok(cudaGetLastError(), "qwen35 rope yarn launch");
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

    /* batch kernels vs the single-step kernels (sequential equivalence) */
    {
        const uint32_t T = 5;
        /* conv */
        float *kern = (float *)malloc((size_t)NCH * K * 4);
        float *xs = (float *)malloc((size_t)T * NCH * 4);
        for (uint32_t i = 0; i < NCH * K; i++) kern[i] = gqa_test_randf();
        for (uint32_t i = 0; i < T * NCH; i++) xs[i] = gqa_test_randf();
        void *dkern, *dstate1, *dstate2, *dx, *dout1, *dout2;
        cudaMalloc(&dkern, (size_t)NCH * K * 4);
        cudaMalloc(&dstate1, (size_t)(K - 1) * NCH * 4);
        cudaMalloc(&dstate2, (size_t)(K - 1) * NCH * 4);
        cudaMalloc(&dx, (size_t)T * NCH * 4);
        cudaMalloc(&dout1, (size_t)T * NCH * 4);
        cudaMalloc(&dout2, (size_t)T * NCH * 4);
        cudaMemcpy(dkern, kern, (size_t)NCH * K * 4, cudaMemcpyHostToDevice);
        cudaMemset(dstate1, 0, (size_t)(K - 1) * NCH * 4);
        cudaMemset(dstate2, 0, (size_t)(K - 1) * NCH * 4);
        cudaMemcpy(dx, xs, (size_t)T * NCH * 4, cudaMemcpyHostToDevice);
        for (uint32_t t = 0; t < T; t++) {
            ok &= pulsar_qwen35_conv_step(
                    (float *)dout1 + (size_t)t * NCH,
                    (const float *)dx + (size_t)t * NCH, dkern, dstate1, NCH, K);
        }
        ok &= pulsar_qwen35_conv_batch(dout2, dx, dkern, dstate2, NCH, K, T);
        float *o1 = (float *)malloc((size_t)T * NCH * 4);
        float *o2 = (float *)malloc((size_t)T * NCH * 4);
        float *st1 = (float *)malloc((size_t)(K - 1) * NCH * 4);
        float *st2 = (float *)malloc((size_t)(K - 1) * NCH * 4);
        cudaMemcpy(o1, dout1, (size_t)T * NCH * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(o2, dout2, (size_t)T * NCH * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(st1, dstate1, (size_t)(K - 1) * NCH * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(st2, dstate2, (size_t)(K - 1) * NCH * 4, cudaMemcpyDeviceToHost);
        for (uint32_t i = 0; i < T * NCH && ok; i++) {
            if (o1[i] != o2[i]) { fprintf(stderr, "qwen35 selftest: conv batch out mismatch\n"); ok = 0; }
        }
        for (uint32_t i = 0; i < (K - 1) * NCH && ok; i++) {
            if (st1[i] != st2[i]) { fprintf(stderr, "qwen35 selftest: conv batch state mismatch\n"); ok = 0; }
        }
        cudaFree(dkern); cudaFree(dstate1); cudaFree(dstate2);
        cudaFree(dx); cudaFree(dout1); cudaFree(dout2);
        free(kern); free(xs); free(o1); free(o2); free(st1); free(st2);

        /* gdn batch vs single steps */
        const uint64_t sbytes = (uint64_t)HV * DIM * DIM * 4;
        float *q = (float *)malloc((size_t)T * HK * DIM * 4);
        float *kv = (float *)malloc((size_t)T * HK * DIM * 4);
        float *vv = (float *)malloc((size_t)T * HV * DIM * 4);
        float *gg = (float *)malloc((size_t)T * HV * 4);
        float *bb = (float *)malloc((size_t)T * HV * 4);
        for (uint32_t i = 0; i < T * HK * DIM; i++) { q[i] = gqa_test_randf(); kv[i] = gqa_test_randf(); }
        for (uint32_t i = 0; i < T * HV * DIM; i++) vv[i] = gqa_test_randf();
        for (uint32_t i = 0; i < T * HV; i++) { gg[i] = -fabsf(gqa_test_randf()); bb[i] = 0.5f + 0.4f * gqa_test_randf(); }
        void *dS1, *dS2, *dq, *dk, *dv, *dg, *db, *do1, *do2;
        cudaMalloc(&dS1, sbytes); cudaMemset(dS1, 0, sbytes);
        cudaMalloc(&dS2, sbytes); cudaMemset(dS2, 0, sbytes);
        cudaMalloc(&dq, (size_t)T * HK * DIM * 4);
        cudaMalloc(&dk, (size_t)T * HK * DIM * 4);
        cudaMalloc(&dv, (size_t)T * HV * DIM * 4);
        cudaMalloc(&dg, (size_t)T * HV * 4);
        cudaMalloc(&db, (size_t)T * HV * 4);
        cudaMalloc(&do1, (size_t)T * HV * DIM * 4);
        cudaMalloc(&do2, (size_t)T * HV * DIM * 4);
        cudaMemcpy(dq, q, (size_t)T * HK * DIM * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dk, kv, (size_t)T * HK * DIM * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dv, vv, (size_t)T * HV * DIM * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dg, gg, (size_t)T * HV * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(db, bb, (size_t)T * HV * 4, cudaMemcpyHostToDevice);
        for (uint32_t t = 0; t < T; t++) {
            ok &= pulsar_qwen35_gdn_step(
                    (float *)do1 + (size_t)t * HV * DIM, dS1,
                    (const float *)dq + (size_t)t * HK * DIM,
                    (const float *)dk + (size_t)t * HK * DIM,
                    (const float *)dv + (size_t)t * HV * DIM,
                    (const float *)dg + (size_t)t * HV,
                    (const float *)db + (size_t)t * HV, HV, HK, DIM);
        }
        ok &= pulsar_qwen35_gdn_batch(do2, dS2, dq, dk, dv, dg, db, HV, HK, DIM, T);
        float *go1 = (float *)malloc((size_t)T * HV * DIM * 4);
        float *go2 = (float *)malloc((size_t)T * HV * DIM * 4);
        float *gs1 = (float *)malloc(sbytes);
        float *gs2 = (float *)malloc(sbytes);
        cudaMemcpy(go1, do1, (size_t)T * HV * DIM * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(go2, do2, (size_t)T * HV * DIM * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(gs1, dS1, sbytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(gs2, dS2, sbytes, cudaMemcpyDeviceToHost);
        for (uint32_t i = 0; i < T * HV * DIM && ok; i++) {
            float diff = fabsf(go1[i] - go2[i]);
            if (diff > 1.0e-5f * fmaxf(1.0f, fabsf(go1[i]))) {
                fprintf(stderr, "qwen35 selftest: gdn batch out mismatch at %u\n", i);
                ok = 0;
            }
        }
        for (uint64_t i = 0; i < (uint64_t)HV * DIM * DIM && ok; i++) {
            float diff = fabsf(gs1[i] - gs2[i]);
            if (diff > 1.0e-5f * fmaxf(1.0f, fabsf(gs1[i]))) {
                fprintf(stderr, "qwen35 selftest: gdn batch state mismatch\n");
                ok = 0;
            }
        }
        cudaFree(dS1); cudaFree(dS2); cudaFree(dq); cudaFree(dk); cudaFree(dv);
        cudaFree(dg); cudaFree(db); cudaFree(do1); cudaFree(do2);
        free(q); free(kv); free(vv); free(gg); free(bb);
        free(go1); free(go2); free(gs1); free(gs2);
    }

    /* draft attention vs host reference (non-causal GQA) */
    {
        const uint32_t NQ = 3, NKV = 70, NH = 8, NKVH = 2, D = 128;
        const float scale = 1.0f / sqrtf((float)D);
        float *q = (float *)malloc((size_t)NQ * NH * D * 4);
        float *k = (float *)malloc((size_t)NKV * NKVH * D * 4);
        float *v = (float *)malloc((size_t)NKV * NKVH * D * 4);
        for (uint32_t i = 0; i < NQ * NH * D; i++) q[i] = gqa_test_randf();
        for (uint32_t i = 0; i < NKV * NKVH * D; i++) { k[i] = gqa_test_randf(); v[i] = gqa_test_randf(); }
        void *dq, *dk, *dv, *dout;
        cudaMalloc(&dq, (size_t)NQ * NH * D * 4);
        cudaMalloc(&dk, (size_t)NKV * NKVH * D * 4);
        cudaMalloc(&dv, (size_t)NKV * NKVH * D * 4);
        cudaMalloc(&dout, (size_t)NQ * NH * D * 4);
        cudaMemcpy(dq, q, (size_t)NQ * NH * D * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dk, k, (size_t)NKV * NKVH * D * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dv, v, (size_t)NKV * NKVH * D * 4, cudaMemcpyHostToDevice);
        ok &= pulsar_qwen35_draft_attn(dout, dq, dk, dv, NQ, NKV, NH, NKVH, D, scale);
        float *got = (float *)malloc((size_t)NQ * NH * D * 4);
        cudaMemcpy(got, dout, (size_t)NQ * NH * D * 4, cudaMemcpyDeviceToHost);
        for (uint32_t t = 0; t < NQ && ok; t++) {
            for (uint32_t h = 0; h < NH && ok; h++) {
                const uint32_t kvh = h / (NH / NKVH);
                const float *qh = q + ((size_t)t * NH + h) * D;
                float *sc = (float *)malloc(NKV * 4);
                float mx = -1.0e30f;
                for (uint32_t r = 0; r < NKV; r++) {
                    const float *kr = k + ((size_t)r * NKVH + kvh) * D;
                    float dot = 0.0f;
                    for (uint32_t d2 = 0; d2 < D; d2++) dot += qh[d2] * kr[d2];
                    sc[r] = dot * scale;
                    if (sc[r] > mx) mx = sc[r];
                }
                float den = 0.0f;
                for (uint32_t r = 0; r < NKV; r++) { sc[r] = expf(sc[r] - mx); den += sc[r]; }
                for (uint32_t d2 = 0; d2 < D && ok; d2++) {
                    float o = 0.0f;
                    for (uint32_t r = 0; r < NKV; r++) {
                        o += sc[r] * v[((size_t)r * NKVH + kvh) * D + d2];
                    }
                    o /= den;
                    float diff = fabsf(o - got[((size_t)t * NH + h) * D + d2]);
                    if (diff > 2.0e-4f * fmaxf(1.0f, fabsf(o))) {
                        fprintf(stderr, "qwen35 selftest: draft attn mismatch\n");
                        ok = 0;
                    }
                }
                free(sc);
            }
        }
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
        free(q); free(k); free(v); free(got);
    }

    if (cudaDeviceSynchronize() != cudaSuccess) ok = 0;
    if (ok) fprintf(stderr, "qwen35 selftest: PASS\n");
    return ok;
}
