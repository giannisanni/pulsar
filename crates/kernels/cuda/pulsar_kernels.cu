/* pulsar CUDA kernel library.
 *
 * gqa_kernels.inc and iq2_tables.inc are derived verbatim from the
 * NeutronStar fork of ds4 (github.com/antirez/ds4), MIT License:
 *   Copyright (c) 2026 The ds4.c authors
 *   Copyright (c) 2023-2026 The ggml authors
 * The MoE dequant functors below are ports of ds4's ds4_cuda_glm_moe.inc
 * (itself a port of metal/moe.metal), same license and attribution.
 * The shim below provides the minimal glue the .inc expects (a tensor is
 * a device pointer plus a byte count) so the kernels build standalone.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct ds4_gpu_tensor {
    void *ptr;
    uint64_t bytes;
} ds4_gpu_tensor;

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "pulsar-kernels: %s: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->bytes = bytes;
    if (!cuda_ok(cudaMalloc(&t->ptr, bytes), "cudaMalloc")) {
        free(t);
        return NULL;
    }
    return t;
}

static void ds4_gpu_tensor_free(ds4_gpu_tensor *t) {
    if (!t) return;
    if (t->ptr) (void)cudaFree(t->ptr);
    free(t);
}

static int ds4_gpu_tensor_write(ds4_gpu_tensor *t, uint64_t off,
                                const void *src, uint64_t bytes) {
    if (!t || off + bytes > t->bytes) return 0;
    return cuda_ok(cudaMemcpy((char *)t->ptr + off, src, bytes,
                              cudaMemcpyHostToDevice), "h2d");
}

static int ds4_gpu_tensor_read(const ds4_gpu_tensor *t, uint64_t off,
                               void *dst, uint64_t bytes) {
    if (!t || off + bytes > t->bytes) return 0;
    return cuda_ok(cudaMemcpy(dst, (const char *)t->ptr + off, bytes,
                              cudaMemcpyDeviceToHost), "d2h");
}

#include "gqa_kernels.inc"

static float f16_to_f32_host(uint16_t h) {
    /* scalar IEEE 754 half -> float, host side (no device intrinsics) */
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((man & 0x400) == 0) { man <<= 1; exp--; }
            man &= 0x3FF;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* ---- pulsar-native Q8_0 matmul ----------------------------------------
 * GGML q8_0 block: 32 int8 quants + one f16 scale (34 bytes). Weights are
 * row-major q8_0; activations are f32. One thread block per (row, token),
 * 256 threads reduce across in_dim. Correctness-first: tuning happens at
 * parity time, against measurements. */

typedef struct __align__(2) {
    uint16_t scale_f16;
    int8_t q[32];
} q8_0_block;

__device__ static float f16_to_f32(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

__global__ static void q8_0_matmul_kernel(
        float *out,                /* [n_tok][out_dim] */
        const q8_0_block *w,       /* [out_dim][in_dim/32] */
        const float *x,            /* [n_tok][in_dim] */
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t n_tok) {
    const uint32_t row = blockIdx.x;
    const uint32_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const uint32_t blocks = in_dim / 32u;
    const q8_0_block *wr = w + (uint64_t)row * blocks;
    const float *xt = x + (uint64_t)tok * in_dim;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        const q8_0_block *blk = &wr[b];
        float s = f16_to_f32(blk->scale_f16);
        float dot = 0.0f;
        const float *xb = xt + (uint64_t)b * 32u;
        for (int i = 0; i < 32; i++) dot += (float)blk->q[i] * xb[i];
        acc += s * dot;
    }
    __shared__ float red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2u; s != 0; s >>= 1u) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[(uint64_t)tok * out_dim + row] = red[0];
}

extern "C" int pulsar_q8_0_matmul(
        void *out_dev,
        const void *w_dev,
        const void *x_dev,
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t n_tok) {
    if (in_dim == 0 || in_dim % 32u != 0 || out_dim == 0 || n_tok == 0) return 0;
    dim3 grid(out_dim, n_tok, 1);
    q8_0_matmul_kernel<<<grid, 256>>>(
            (float *)out_dev, (const q8_0_block *)w_dev,
            (const float *)x_dev, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "q8_0 matmul launch");
}

/* CPU-reference selftest: quantize random weights to q8_0 on the host,
 * run both pipelines, compare. */
static uint16_t f32_to_f16_bits(float f) {
    /* scalar IEEE 754 float -> half (round-to-nearest-even), host side */
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t man = bits & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_man = man >> shift;
        uint32_t rem = man & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (half_man & 1u))) half_man++;
        return (uint16_t)(sign | half_man);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    uint32_t half_man = man >> 13;
    uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half_man & 1u))) {
        half_man++;
        if (half_man == 0x400u) { half_man = 0; exp++; if (exp >= 31) return (uint16_t)(sign | 0x7C00u); }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | half_man);
}

extern "C" int pulsar_q8_0_matmul_selftest(void) {
    const uint32_t in_dim = 4096, out_dim = 512, n_tok = 3;
    const uint32_t blocks = in_dim / 32u;
    q8_0_block *w = (q8_0_block *)malloc((uint64_t)out_dim * blocks * sizeof(*w));
    float *wf = (float *)malloc((uint64_t)out_dim * in_dim * sizeof(float));
    float *x = (float *)malloc((uint64_t)n_tok * in_dim * sizeof(float));
    float *ref = (float *)calloc((uint64_t)n_tok * out_dim, sizeof(float));
    float *gpu = (float *)malloc((uint64_t)n_tok * out_dim * sizeof(float));

    for (uint64_t i = 0; i < (uint64_t)n_tok * in_dim; i++) x[i] = gqa_test_randf();
    /* quantize: per 32-block, scale = amax/127, q = round(v/scale) */
    for (uint32_t r = 0; r < out_dim; r++) {
        for (uint32_t b = 0; b < blocks; b++) {
            float amax = 0.0f, vals[32];
            for (int i = 0; i < 32; i++) {
                vals[i] = gqa_test_randf();
                float a = fabsf(vals[i]);
                if (a > amax) amax = a;
            }
            float scale = amax / 127.0f;
            q8_0_block *blk = &w[(uint64_t)r * blocks + b];
            blk->scale_f16 = f32_to_f16_bits(scale);
            float s = f16_to_f32_host(blk->scale_f16);
            for (int i = 0; i < 32; i++) {
                int qi = scale > 0.0f ? (int)lrintf(vals[i] / scale) : 0;
                if (qi > 127) qi = 127;
                if (qi < -127) qi = -127;
                blk->q[i] = (int8_t)qi;
                wf[(uint64_t)r * in_dim + b * 32u + i] = s * (float)qi;
            }
        }
    }
    /* reference matmul on the dequantized weights */
    for (uint32_t t = 0; t < n_tok; t++)
        for (uint32_t r = 0; r < out_dim; r++) {
            double acc = 0.0;
            for (uint32_t i = 0; i < in_dim; i++)
                acc += (double)wf[(uint64_t)r * in_dim + i] * x[(uint64_t)t * in_dim + i];
            ref[(uint64_t)t * out_dim + r] = (float)acc;
        }

    void *w_dev = NULL, *x_dev = NULL, *out_dev = NULL;
    const uint64_t w_bytes = (uint64_t)out_dim * blocks * sizeof(*w);
    const uint64_t x_bytes = (uint64_t)n_tok * in_dim * sizeof(float);
    const uint64_t o_bytes = (uint64_t)n_tok * out_dim * sizeof(float);
    int ok = cuda_ok(cudaMalloc(&w_dev, w_bytes), "w alloc") &&
             cuda_ok(cudaMalloc(&x_dev, x_bytes), "x alloc") &&
             cuda_ok(cudaMalloc(&out_dev, o_bytes), "out alloc") &&
             cuda_ok(cudaMemcpy(w_dev, w, w_bytes, cudaMemcpyHostToDevice), "w h2d") &&
             cuda_ok(cudaMemcpy(x_dev, x, x_bytes, cudaMemcpyHostToDevice), "x h2d") &&
             pulsar_q8_0_matmul(out_dev, w_dev, x_dev, in_dim, out_dim, n_tok) &&
             cuda_ok(cudaDeviceSynchronize(), "sync") &&
             cuda_ok(cudaMemcpy(gpu, out_dev, o_bytes, cudaMemcpyDeviceToHost), "d2h");
    float maxd = 0.0f, maxref = 0.0f;
    if (ok) {
        for (uint64_t i = 0; i < (uint64_t)n_tok * out_dim; i++) {
            float d = fabsf(gpu[i] - ref[i]);
            if (d > maxd) maxd = d;
            float a = fabsf(ref[i]);
            if (a > maxref) maxref = a;
        }
        ok = maxd <= 1e-3f * (maxref > 1.0f ? maxref : 1.0f);
    }
    fprintf(stderr, "q8_0-matmul-selftest: %s (max abs diff %.2e, max |ref| %.2e)\n",
            ok ? "PASS" : "FAIL", (double)maxd, (double)maxref);
    if (w_dev) cudaFree(w_dev);
    if (x_dev) cudaFree(x_dev);
    if (out_dev) cudaFree(out_dev);
    free(w); free(wf); free(x); free(ref); free(gpu);
    return ok;
}


/* ---- sigmoid router + top-k select ------------------------------------
 * Warp-per-token select, derived from ds4's glm_router_select_kernel (the
 * Hy3 router mirrors GLM: probs = sigmoid(logits), selection score =
 * prob + bias, route weights = selected probs normalized * scale).
 * pulsar contract: bias is an explicit device pointer, not a model-map
 * offset. n_expert <= 256, k_used <= n_expert. */

__device__ __forceinline__ static float router_sigmoid(float x) {
    if (x >= 0.0f) {
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = expf(x);
    return e / (1.0f + e);
}

__device__ __forceinline__ static bool router_better(
        float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

__global__ static void router_select_kernel(
        int32_t *selected,         /* [n_tok][k_used] */
        float *weights,            /* [n_tok][k_used] */
        const float *logits,       /* [n_tok][n_expert] */
        const float *bias,         /* [n_expert] */
        uint32_t n_expert,
        uint32_t k_used,
        float weight_scale,
        uint32_t n_tok) {
    const uint32_t lane = threadIdx.x;
    const uint32_t token = blockIdx.x * blockDim.y + threadIdx.y;
    if (token >= n_tok || lane >= 32u) return;

    const float *log = logits + (uint64_t)token * n_expert;
    int32_t *sel = selected + (uint64_t)token * k_used;
    float *w = weights + (uint64_t)token * k_used;

    float local_prob[8];
    float local_score[8];
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        const uint32_t e = lane + j * 32u;
        if (e < n_expert) {
            const float p = router_sigmoid(log[e]);
            local_prob[j] = p;
            local_score[j] = p + bias[e];
        } else {
            local_prob[j] = 0.0f;
            local_score[j] = -INFINITY;
        }
    }
    __syncwarp();

    float sum = 0.0f;
    for (uint32_t k = 0; k < k_used; k++) {
        float best_score = -INFINITY;
        float best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            if (router_better(local_score[j], e, best_score, best_idx)) {
                best_score = local_score[j];
                best_prob = local_prob[j];
                best_idx = e;
            }
        }
        #pragma unroll
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
            const float other_score = __shfl_xor_sync(0xffffffffu, best_score, mask);
            const float other_prob = __shfl_xor_sync(0xffffffffu, best_prob, mask);
            const uint32_t other_idx = __shfl_xor_sync(0xffffffffu, best_idx, mask);
            if (router_better(other_score, other_idx, best_score, best_idx)) {
                best_score = other_score;
                best_prob = other_prob;
                best_idx = other_idx;
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            if (lane + j * 32u == best_idx) local_score[j] = -INFINITY;
        }
        if (lane == 0) {
            sel[k] = (int32_t)best_idx;
            w[k] = best_prob;
        }
        sum += best_prob;
    }

    if (lane == 0) {
        sum = fmaxf(sum, 6.103515625e-5f);
        for (uint32_t k = 0; k < k_used; k++) w[k] = w[k] / sum * weight_scale;
    }
}

extern "C" int pulsar_router_select(
        void *selected_dev,        /* int32 [n_tok][k_used] */
        void *weights_dev,         /* f32   [n_tok][k_used] */
        const void *logits_dev,    /* f32   [n_tok][n_expert] */
        const void *bias_dev,      /* f32   [n_expert] */
        uint32_t n_expert,
        uint32_t k_used,
        float weight_scale,
        uint32_t n_tok) {
    if (n_expert == 0 || n_expert > 256u || k_used == 0 || k_used > n_expert ||
        n_tok == 0) {
        return 0;
    }
    dim3 block(32, 4, 1);
    router_select_kernel<<<(n_tok + 3u) / 4u, block>>>(
            (int32_t *)selected_dev, (float *)weights_dev,
            (const float *)logits_dev, (const float *)bias_dev,
            n_expert, k_used, weight_scale, n_tok);
    return cuda_ok(cudaGetLastError(), "router select launch");
}

/* CPU-reference selftest across Hy3-like and GLM-like shapes. */
static int router_selftest_one(uint32_t n_expert, uint32_t k_used,
                               float scale, uint32_t n_tok) {
    float *logits = (float *)malloc((uint64_t)n_tok * n_expert * sizeof(float));
    float *bias = (float *)malloc((uint64_t)n_expert * sizeof(float));
    int32_t *sel_ref = (int32_t *)malloc((uint64_t)n_tok * k_used * sizeof(int32_t));
    float *w_ref = (float *)malloc((uint64_t)n_tok * k_used * sizeof(float));
    int32_t *sel_gpu = (int32_t *)malloc((uint64_t)n_tok * k_used * sizeof(int32_t));
    float *w_gpu = (float *)malloc((uint64_t)n_tok * k_used * sizeof(float));

    for (uint64_t i = 0; i < (uint64_t)n_tok * n_expert; i++)
        logits[i] = gqa_test_randf() * 4.0f;
    for (uint32_t e = 0; e < n_expert; e++) bias[e] = gqa_test_randf();

    for (uint32_t t = 0; t < n_tok; t++) {
        const float *log = logits + (uint64_t)t * n_expert;
        float prob[256], score[256];
        for (uint32_t e = 0; e < n_expert; e++) {
            prob[e] = 1.0f / (1.0f + expf(-log[e]));
            score[e] = prob[e] + bias[e];
        }
        float sum = 0.0f;
        for (uint32_t k = 0; k < k_used; k++) {
            uint32_t best = UINT32_MAX;
            for (uint32_t e = 0; e < n_expert; e++) {
                if (best == UINT32_MAX || score[e] > score[best]) best = e;
            }
            sel_ref[(uint64_t)t * k_used + k] = (int32_t)best;
            w_ref[(uint64_t)t * k_used + k] = prob[best];
            sum += prob[best];
            score[best] = -INFINITY;
        }
        sum = fmaxf(sum, 6.103515625e-5f);
        for (uint32_t k = 0; k < k_used; k++)
            w_ref[(uint64_t)t * k_used + k] =
                w_ref[(uint64_t)t * k_used + k] / sum * scale;
    }

    void *log_dev = NULL, *bias_dev = NULL, *sel_dev = NULL, *w_dev = NULL;
    const uint64_t log_bytes = (uint64_t)n_tok * n_expert * sizeof(float);
    const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
    const uint64_t sel_bytes = (uint64_t)n_tok * k_used * sizeof(int32_t);
    const uint64_t w_bytes = (uint64_t)n_tok * k_used * sizeof(float);
    int ok = cuda_ok(cudaMalloc(&log_dev, log_bytes), "logits alloc") &&
             cuda_ok(cudaMalloc(&bias_dev, bias_bytes), "bias alloc") &&
             cuda_ok(cudaMalloc(&sel_dev, sel_bytes), "sel alloc") &&
             cuda_ok(cudaMalloc(&w_dev, w_bytes), "w alloc") &&
             cuda_ok(cudaMemcpy(log_dev, logits, log_bytes, cudaMemcpyHostToDevice), "logits h2d") &&
             cuda_ok(cudaMemcpy(bias_dev, bias, bias_bytes, cudaMemcpyHostToDevice), "bias h2d") &&
             pulsar_router_select(sel_dev, w_dev, log_dev, bias_dev,
                                  n_expert, k_used, scale, n_tok) &&
             cuda_ok(cudaDeviceSynchronize(), "sync") &&
             cuda_ok(cudaMemcpy(sel_gpu, sel_dev, sel_bytes, cudaMemcpyDeviceToHost), "sel d2h") &&
             cuda_ok(cudaMemcpy(w_gpu, w_dev, w_bytes, cudaMemcpyDeviceToHost), "w d2h");
    float maxd = 0.0f;
    uint32_t idx_mismatch = 0;
    if (ok) {
        for (uint64_t i = 0; i < (uint64_t)n_tok * k_used; i++) {
            if (sel_gpu[i] != sel_ref[i]) idx_mismatch++;
            float d = fabsf(w_gpu[i] - w_ref[i]);
            if (d > maxd) maxd = d;
        }
        ok = idx_mismatch == 0 && maxd <= 1e-5f;
    }
    fprintf(stderr,
            "router-selftest n_expert=%u k=%u: %s (idx mismatches %u, max w diff %.2e)\n",
            n_expert, k_used, ok ? "PASS" : "FAIL", idx_mismatch, (double)maxd);
    if (log_dev) cudaFree(log_dev);
    if (bias_dev) cudaFree(bias_dev);
    if (sel_dev) cudaFree(sel_dev);
    if (w_dev) cudaFree(w_dev);
    free(logits); free(bias); free(sel_ref); free(w_ref); free(sel_gpu); free(w_gpu);
    return ok;
}

extern "C" int pulsar_router_selftest(void) {
    /* Hy3-like (64 experts, top-8), GLM-like (256, top-8), odd token count */
    return router_selftest_one(64, 8, 2.5f, 7) &&
           router_selftest_one(256, 8, 1.0f, 5) &&
           router_selftest_one(96, 6, 1.5f, 1);
}

/* ---- routed-expert MoE: IQ2_XXS / Q2_K dequant-dot kernels -------------
 * pair swiglu: mid[tok][slot][row] = silu(gate_row.x) * (up_row.x) * w
 * down:        out[tok][row]      = sum_slot down_row . mid[tok][slot]
 * pulsar contract: each (token, slot) carries explicit device pointers to
 * that expert's gate/up/down slabs (DESIGN-expert-store.md); a NULL slab
 * means "not routed" and contributes zero. One warp per output row. */

#define PULSAR_QK_K 256u

typedef struct {
    uint8_t scales[PULSAR_QK_K / 16];
    uint8_t qs[PULSAR_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t qs[PULSAR_QK_K / 8];
} block_iq2_xxs;

#include "iq2_tables.inc"

struct deq_q2_K {
    __device__ __forceinline__ static float value(const char *blocks, uint32_t k) {
        const block_q2_K *xb = (const block_q2_K *)blocks + (k / PULSAR_QK_K);
        const uint32_t idx = k & (PULSAR_QK_K - 1u);
        const uint32_t group = idx / 16u;
        const uint32_t l = idx & 15u;
        const uint32_t q_base = 32u * (group / 8u) + 16u * (group & 1u);
        const uint32_t shift = ((group / 2u) & 3u) * 2u;
        const uint32_t q = ((uint32_t)xb->qs[q_base + l] >> shift) & 0x03u;
        const uint32_t sc = (uint32_t)xb->scales[group];
        return f16_to_f32(xb->d) * (float)(sc & 0x0Fu) * (float)q -
               f16_to_f32(xb->dmin) * (float)(sc >> 4u);
    }
};

struct deq_iq2_xxs {
    __device__ __forceinline__ static float value(const char *blocks, uint32_t k) {
        const block_iq2_xxs *xb = (const block_iq2_xxs *)blocks + (k >> 8);
        const uint32_t within = k & 255u;
        const uint32_t ib32 = within >> 5;
        const uint32_t kk = within & 31u;
        const uint32_t l = kk >> 3;
        const uint32_t j = kk & 7u;
        const uint16_t *q2 = xb->qs + 4u * ib32;
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint32_t grid_idx = (aux0 >> (8u * l)) & 0xffu;
        const uint32_t sign_idx = (aux1 >> (7u * l)) & 127u;
        const uint64_t g = cuda_iq2xxs_grid[grid_idx];
        const int32_t gj = (int32_t)(uint32_t)(uint8_t)(g >> (8u * j));
        const uint8_t signs = cuda_ksigns_iq2xs[sign_idx];
        const float sign = (signs & (1u << j)) ? -1.0f : 1.0f;
        return 0.125f * f16_to_f32(xb->d) * (float)ls * (float)gj * sign;
    }
};

typedef struct {
    const void *gate;
    const void *up;
    const void *down;
} pulsar_expert_ptrs;

template <typename DEQ>
__global__ static void moe_pair_swiglu_kernel(
        float *mid,                     /* [n_tok][n_used][mid_dim] */
        const pulsar_expert_ptrs *ptrs, /* [n_tok][n_used] */
        const float *weights,           /* [n_tok][n_used] */
        const float *x,                 /* [n_tok][in_dim] */
        uint32_t in_dim,
        uint32_t mid_dim,
        uint32_t n_used,
        uint32_t n_tok,
        uint64_t row_bytes) {           /* gate and up share type+in_dim */
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t slot = blockIdx.y;
    const uint32_t token = blockIdx.z;
    if (row >= mid_dim || slot >= n_used || token >= n_tok) return;

    const uint64_t slot_off = (uint64_t)token * n_used + slot;
    const uint64_t mid_off = slot_off * mid_dim + row;
    const pulsar_expert_ptrs p = ptrs[slot_off];
    if (!p.gate || !p.up) {
        if (lane == 0) mid[mid_off] = 0.0f;
        return;
    }

    const char *gate_row = (const char *)p.gate + (uint64_t)row * row_bytes;
    const char *up_row = (const char *)p.up + (uint64_t)row * row_bytes;
    const float *token_x = x + (uint64_t)token * in_dim;

    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    for (uint32_t k = lane; k < in_dim; k += 32u) {
        const float xv = token_x[k];
        acc_gate += DEQ::value(gate_row, k) * xv;
        acc_up += DEQ::value(up_row, k) * xv;
    }
    #pragma unroll
    for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
        acc_gate += __shfl_xor_sync(0xffffffffu, acc_gate, mask);
        acc_up += __shfl_xor_sync(0xffffffffu, acc_up, mask);
    }
    if (lane == 0) {
        const float g = acc_gate;
        const float sw = g / (1.0f + expf(-g));
        mid[mid_off] = sw * acc_up * weights[slot_off];
    }
}

template <typename DEQ>
__global__ static void moe_down_kernel(
        float *out,                     /* [n_tok][out_dim] */
        const pulsar_expert_ptrs *ptrs, /* [n_tok][n_used] */
        const float *mid,               /* [n_tok][n_used][mid_dim] */
        uint32_t mid_dim,
        uint32_t out_dim,
        uint32_t n_used,
        uint32_t n_tok,
        uint64_t row_bytes) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t token = blockIdx.y;
    if (row >= out_dim || token >= n_tok) return;

    const uint64_t slot_base = (uint64_t)token * n_used;
    float acc = 0.0f;
    for (uint32_t slot = 0; slot < n_used; slot++) {
        const pulsar_expert_ptrs p = ptrs[slot_base + slot];
        if (!p.down) continue;
        const char *down_row = (const char *)p.down + (uint64_t)row * row_bytes;
        const float *slot_mid = mid + (slot_base + slot) * mid_dim;
        for (uint32_t k = lane; k < mid_dim; k += 32u) {
            acc += DEQ::value(down_row, k) * slot_mid[k];
        }
    }
    #pragma unroll
    for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
        acc += __shfl_xor_sync(0xffffffffu, acc, mask);
    }
    if (lane == 0) out[(uint64_t)token * out_dim + row] = acc;
}

enum {
    PULSAR_QUANT_Q2_K = 0,
    PULSAR_QUANT_IQ2_XXS = 1,
};

extern "C" int pulsar_moe_pair_swiglu(
        void *mid_dev,
        const void *ptrs_dev,
        const void *weights_dev,
        const void *x_dev,
        uint32_t in_dim,
        uint32_t mid_dim,
        uint32_t n_used,
        uint32_t n_tok,
        uint64_t row_bytes,
        uint32_t quant) {
    if (in_dim == 0 || in_dim % PULSAR_QK_K != 0 || mid_dim == 0 ||
        n_used == 0 || n_tok == 0 || row_bytes == 0) {
        return 0;
    }
    dim3 block(32, 4, 1);
    dim3 grid((mid_dim + 3u) / 4u, n_used, n_tok);
    switch (quant) {
    case PULSAR_QUANT_Q2_K:
        moe_pair_swiglu_kernel<deq_q2_K><<<grid, block>>>(
                (float *)mid_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const float *)weights_dev, (const float *)x_dev,
                in_dim, mid_dim, n_used, n_tok, row_bytes);
        break;
    case PULSAR_QUANT_IQ2_XXS:
        moe_pair_swiglu_kernel<deq_iq2_xxs><<<grid, block>>>(
                (float *)mid_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const float *)weights_dev, (const float *)x_dev,
                in_dim, mid_dim, n_used, n_tok, row_bytes);
        break;
    default:
        return 0;
    }
    return cuda_ok(cudaGetLastError(), "moe pair swiglu launch");
}

extern "C" int pulsar_moe_down(
        void *out_dev,
        const void *ptrs_dev,
        const void *mid_dev,
        uint32_t mid_dim,
        uint32_t out_dim,
        uint32_t n_used,
        uint32_t n_tok,
        uint64_t row_bytes,
        uint32_t quant) {
    if (mid_dim == 0 || mid_dim % PULSAR_QK_K != 0 || out_dim == 0 ||
        n_used == 0 || n_tok == 0 || row_bytes == 0) {
        return 0;
    }
    dim3 block(32, 4, 1);
    dim3 grid((out_dim + 3u) / 4u, n_tok, 1);
    switch (quant) {
    case PULSAR_QUANT_Q2_K:
        moe_down_kernel<deq_q2_K><<<grid, block>>>(
                (float *)out_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const float *)mid_dev, mid_dim, out_dim, n_used, n_tok,
                row_bytes);
        break;
    case PULSAR_QUANT_IQ2_XXS:
        moe_down_kernel<deq_iq2_xxs><<<grid, block>>>(
                (float *)out_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const float *)mid_dev, mid_dim, out_dim, n_used, n_tok,
                row_bytes);
        break;
    default:
        return 0;
    }
    return cuda_ok(cudaGetLastError(), "moe down launch");
}

/* ---- MoE selftest: random quantized slabs vs a host dequant reference -- */

static uint8_t test_randbyte(void) {
    return (uint8_t)lrintf((gqa_test_randf() * 0.5f + 0.5f) * 255.0f);
}

/* host mirrors of the device dequant functors; tables fetched from the
 * device so both sides read identical constants */
static uint8_t h_ksigns[128];
static uint64_t h_grid[256];

static float host_deq_q2_K(const char *blocks, uint32_t k) {
    const block_q2_K *xb = (const block_q2_K *)blocks + (k / PULSAR_QK_K);
    const uint32_t idx = k & (PULSAR_QK_K - 1u);
    const uint32_t group = idx / 16u;
    const uint32_t l = idx & 15u;
    const uint32_t q_base = 32u * (group / 8u) + 16u * (group & 1u);
    const uint32_t shift = ((group / 2u) & 3u) * 2u;
    const uint32_t q = ((uint32_t)xb->qs[q_base + l] >> shift) & 0x03u;
    const uint32_t sc = (uint32_t)xb->scales[group];
    return f16_to_f32_host(xb->d) * (float)(sc & 0x0Fu) * (float)q -
           f16_to_f32_host(xb->dmin) * (float)(sc >> 4u);
}

static float host_deq_iq2_xxs(const char *blocks, uint32_t k) {
    const block_iq2_xxs *xb = (const block_iq2_xxs *)blocks + (k >> 8);
    const uint32_t within = k & 255u;
    const uint32_t ib32 = within >> 5;
    const uint32_t kk = within & 31u;
    const uint32_t l = kk >> 3;
    const uint32_t j = kk & 7u;
    const uint16_t *q2 = xb->qs + 4u * ib32;
    const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
    const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
    const uint32_t ls = 2u * (aux1 >> 28) + 1u;
    const uint32_t grid_idx = (aux0 >> (8u * l)) & 0xffu;
    const uint32_t sign_idx = (aux1 >> (7u * l)) & 127u;
    const uint64_t g = h_grid[grid_idx];
    const int32_t gj = (int32_t)(uint32_t)(uint8_t)(g >> (8u * j));
    const uint8_t signs = h_ksigns[sign_idx];
    const float sign = (signs & (1u << j)) ? -1.0f : 1.0f;
    return 0.125f * f16_to_f32_host(xb->d) * (float)ls * (float)gj * sign;
}

static void fill_slab(char *slab, uint32_t n_rows, uint32_t n_el,
                      uint64_t row_bytes, uint32_t quant) {
    for (uint32_t r = 0; r < n_rows; r++) {
        char *row = slab + (uint64_t)r * row_bytes;
        for (uint64_t b = 0; b < row_bytes; b++) row[b] = (char)test_randbyte();
        /* overwrite scale halves with sane small values (random f16 bits
         * can be inf/nan) */
        for (uint32_t blk = 0; blk < n_el / PULSAR_QK_K; blk++) {
            if (quant == PULSAR_QUANT_Q2_K) {
                block_q2_K *q = (block_q2_K *)row + blk;
                q->d = f32_to_f16_bits(fabsf(gqa_test_randf()) * 0.05f + 0.001f);
                q->dmin = f32_to_f16_bits(fabsf(gqa_test_randf()) * 0.05f);
            } else {
                block_iq2_xxs *q = (block_iq2_xxs *)row + blk;
                q->d = f32_to_f16_bits(fabsf(gqa_test_randf()) * 0.1f + 0.001f);
            }
        }
    }
}

static int moe_selftest_one(uint32_t quant, const char *name) {
    const uint32_t in_dim = 512, mid_dim = 256, out_dim = 320;
    const uint32_t n_expert = 8, n_used = 4, n_tok = 3;
    const uint64_t pair_row_bytes = (uint64_t)(in_dim / PULSAR_QK_K) *
        (quant == PULSAR_QUANT_Q2_K ? sizeof(block_q2_K) : sizeof(block_iq2_xxs));
    const uint64_t down_row_bytes = (uint64_t)(mid_dim / PULSAR_QK_K) *
        (quant == PULSAR_QUANT_Q2_K ? sizeof(block_q2_K) : sizeof(block_iq2_xxs));
    const uint64_t gate_slab_bytes = (uint64_t)n_expert * mid_dim * pair_row_bytes;
    const uint64_t down_slab_bytes = (uint64_t)n_expert * out_dim * down_row_bytes;
    float (*deq)(const char *, uint32_t) =
        quant == PULSAR_QUANT_Q2_K ? host_deq_q2_K : host_deq_iq2_xxs;

    char *gate = (char *)malloc(gate_slab_bytes);
    char *up = (char *)malloc(gate_slab_bytes);
    char *down = (char *)malloc(down_slab_bytes);
    float *x = (float *)malloc((uint64_t)n_tok * in_dim * sizeof(float));
    float *w = (float *)malloc((uint64_t)n_tok * n_used * sizeof(float));
    int32_t *sel = (int32_t *)malloc((uint64_t)n_tok * n_used * sizeof(int32_t));
    float *mid_ref = (float *)calloc((uint64_t)n_tok * n_used * mid_dim, sizeof(float));
    float *out_ref = (float *)calloc((uint64_t)n_tok * out_dim, sizeof(float));
    float *mid_gpu = (float *)malloc((uint64_t)n_tok * n_used * mid_dim * sizeof(float));
    float *out_gpu = (float *)malloc((uint64_t)n_tok * out_dim * sizeof(float));

    fill_slab(gate, n_expert * mid_dim, in_dim, pair_row_bytes, quant);
    fill_slab(up, n_expert * mid_dim, in_dim, pair_row_bytes, quant);
    fill_slab(down, n_expert * out_dim, mid_dim, down_row_bytes, quant);
    for (uint64_t i = 0; i < (uint64_t)n_tok * in_dim; i++) x[i] = gqa_test_randf();
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t s = 0; s < n_used; s++) {
            sel[t * n_used + s] = (int32_t)(test_randbyte() % n_expert);
            w[t * n_used + s] = fabsf(gqa_test_randf()) + 0.1f;
        }
    }
    sel[1 * n_used + 2] = -1; /* one unrouted slot: NULL ptrs, zero output */

    void *gate_dev = NULL, *up_dev = NULL, *down_dev = NULL;
    void *x_dev = NULL, *w_dev = NULL, *ptrs_dev = NULL;
    void *mid_dev = NULL, *out_dev = NULL;
    pulsar_expert_ptrs ptrs[n_tok * n_used];
    int ok = cuda_ok(cudaMalloc(&gate_dev, gate_slab_bytes), "gate alloc") &&
             cuda_ok(cudaMalloc(&up_dev, gate_slab_bytes), "up alloc") &&
             cuda_ok(cudaMalloc(&down_dev, down_slab_bytes), "down alloc") &&
             cuda_ok(cudaMalloc(&x_dev, (uint64_t)n_tok * in_dim * sizeof(float)), "x alloc") &&
             cuda_ok(cudaMalloc(&w_dev, (uint64_t)n_tok * n_used * sizeof(float)), "w alloc") &&
             cuda_ok(cudaMalloc(&ptrs_dev, sizeof(ptrs)), "ptrs alloc") &&
             cuda_ok(cudaMalloc(&mid_dev, (uint64_t)n_tok * n_used * mid_dim * sizeof(float)), "mid alloc") &&
             cuda_ok(cudaMalloc(&out_dev, (uint64_t)n_tok * out_dim * sizeof(float)), "out alloc") &&
             cuda_ok(cudaMemcpy(gate_dev, gate, gate_slab_bytes, cudaMemcpyHostToDevice), "gate h2d") &&
             cuda_ok(cudaMemcpy(up_dev, up, gate_slab_bytes, cudaMemcpyHostToDevice), "up h2d") &&
             cuda_ok(cudaMemcpy(down_dev, down, down_slab_bytes, cudaMemcpyHostToDevice), "down h2d") &&
             cuda_ok(cudaMemcpy(x_dev, x, (uint64_t)n_tok * in_dim * sizeof(float), cudaMemcpyHostToDevice), "x h2d") &&
             cuda_ok(cudaMemcpy(w_dev, w, (uint64_t)n_tok * n_used * sizeof(float), cudaMemcpyHostToDevice), "w h2d");

    for (uint32_t i = 0; i < n_tok * n_used; i++) {
        const int32_t e = sel[i];
        if (e < 0) {
            ptrs[i].gate = ptrs[i].up = ptrs[i].down = NULL;
            continue;
        }
        ptrs[i].gate = (char *)gate_dev + (uint64_t)e * mid_dim * pair_row_bytes;
        ptrs[i].up = (char *)up_dev + (uint64_t)e * mid_dim * pair_row_bytes;
        ptrs[i].down = (char *)down_dev + (uint64_t)e * out_dim * down_row_bytes;
    }
    ok = ok && cuda_ok(cudaMemcpy(ptrs_dev, ptrs, sizeof(ptrs), cudaMemcpyHostToDevice), "ptrs h2d") &&
         pulsar_moe_pair_swiglu(mid_dev, ptrs_dev, w_dev, x_dev,
                                in_dim, mid_dim, n_used, n_tok,
                                pair_row_bytes, quant) &&
         pulsar_moe_down(out_dev, ptrs_dev, mid_dev, mid_dim, out_dim,
                         n_used, n_tok, down_row_bytes, quant) &&
         cuda_ok(cudaDeviceSynchronize(), "sync") &&
         cuda_ok(cudaMemcpy(mid_gpu, mid_dev, (uint64_t)n_tok * n_used * mid_dim * sizeof(float), cudaMemcpyDeviceToHost), "mid d2h") &&
         cuda_ok(cudaMemcpy(out_gpu, out_dev, (uint64_t)n_tok * out_dim * sizeof(float), cudaMemcpyDeviceToHost), "out d2h");

    /* host reference */
    for (uint32_t t = 0; t < n_tok && ok; t++) {
        for (uint32_t s = 0; s < n_used; s++) {
            const int32_t e = sel[t * n_used + s];
            if (e < 0) continue;
            const char *gs = gate + (uint64_t)e * mid_dim * pair_row_bytes;
            const char *us = up + (uint64_t)e * mid_dim * pair_row_bytes;
            for (uint32_t r = 0; r < mid_dim; r++) {
                double ag = 0.0, au = 0.0;
                const char *gr = gs + (uint64_t)r * pair_row_bytes;
                const char *ur = us + (uint64_t)r * pair_row_bytes;
                for (uint32_t k = 0; k < in_dim; k++) {
                    const double xv = x[(uint64_t)t * in_dim + k];
                    ag += (double)deq(gr, k) * xv;
                    au += (double)deq(ur, k) * xv;
                }
                const double sw = ag / (1.0 + exp(-ag));
                mid_ref[((uint64_t)t * n_used + s) * mid_dim + r] =
                    (float)(sw * au * w[t * n_used + s]);
            }
        }
        for (uint32_t r = 0; r < out_dim; r++) {
            double acc = 0.0;
            for (uint32_t s = 0; s < n_used; s++) {
                const int32_t e = sel[t * n_used + s];
                if (e < 0) continue;
                const char *dr = down + (uint64_t)e * out_dim * down_row_bytes +
                                 (uint64_t)r * down_row_bytes;
                const float *sm = mid_ref + ((uint64_t)t * n_used + s) * mid_dim;
                for (uint32_t k = 0; k < mid_dim; k++)
                    acc += (double)deq(dr, k) * sm[k];
            }
            out_ref[(uint64_t)t * out_dim + r] = (float)acc;
        }
    }

    float mid_maxd = 0.0f, mid_maxref = 0.0f, out_maxd = 0.0f, out_maxref = 0.0f;
    if (ok) {
        for (uint64_t i = 0; i < (uint64_t)n_tok * n_used * mid_dim; i++) {
            const float d = fabsf(mid_gpu[i] - mid_ref[i]);
            if (d > mid_maxd) mid_maxd = d;
            const float a = fabsf(mid_ref[i]);
            if (a > mid_maxref) mid_maxref = a;
        }
        for (uint64_t i = 0; i < (uint64_t)n_tok * out_dim; i++) {
            const float d = fabsf(out_gpu[i] - out_ref[i]);
            if (d > out_maxd) out_maxd = d;
            const float a = fabsf(out_ref[i]);
            if (a > out_maxref) out_maxref = a;
        }
        ok = mid_maxd <= 1e-3f * (mid_maxref > 1.0f ? mid_maxref : 1.0f) &&
             out_maxd <= 2e-3f * (out_maxref > 1.0f ? out_maxref : 1.0f);
    }
    fprintf(stderr, "moe-selftest %s: %s (mid max diff %.2e, out max diff %.2e, max |out ref| %.2e)\n",
            name, ok ? "PASS" : "FAIL", (double)mid_maxd, (double)out_maxd,
            (double)out_maxref);
    if (gate_dev) cudaFree(gate_dev);
    if (up_dev) cudaFree(up_dev);
    if (down_dev) cudaFree(down_dev);
    if (x_dev) cudaFree(x_dev);
    if (w_dev) cudaFree(w_dev);
    if (ptrs_dev) cudaFree(ptrs_dev);
    if (mid_dev) cudaFree(mid_dev);
    if (out_dev) cudaFree(out_dev);
    free(gate); free(up); free(down); free(x); free(w); free(sel);
    free(mid_ref); free(out_ref); free(mid_gpu); free(out_gpu);
    return ok;
}

extern "C" int pulsar_moe_selftest(void) {
    if (!cuda_ok(cudaMemcpyFromSymbol(h_ksigns, cuda_ksigns_iq2xs,
                                      sizeof(h_ksigns)), "ksigns fetch") ||
        !cuda_ok(cudaMemcpyFromSymbol(h_grid, cuda_iq2xxs_grid,
                                      sizeof(h_grid)), "grid fetch")) {
        return 0;
    }
    return moe_selftest_one(PULSAR_QUANT_Q2_K, "q2_K") &&
           moe_selftest_one(PULSAR_QUANT_IQ2_XXS, "iq2_xxs");
}

/* ---- forward-graph glue: rms-norm, f32 matmul, swiglu, add, embed ------
 * Verbatim ports of ds4's elementwise/reduction kernels; together with the
 * kernels above these cover every op in the Hy3 decode graph. */

__global__ static void rms_norm_weight_kernel(
        float *out, const float *x, const float *w,
        uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale * w[i];
    }
}

extern "C" int pulsar_rms_norm(
        void *out_dev, const void *x_dev, const void *w_dev,
        uint32_t n, uint32_t rows, float eps) {
    if (n == 0 || rows == 0) return 0;
    rms_norm_weight_kernel<<<rows, 256>>>(
            (float *)out_dev, (const float *)x_dev, (const float *)w_dev,
            n, rows, eps);
    return cuda_ok(cudaGetLastError(), "rms norm launch");
}

__global__ static void matmul_f32_kernel(
        float *out, const float *w, const float *x,
        uint64_t in_dim, uint64_t out_dim, uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    float sum = 0.0f;
    const float *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += wr[i] * xr[i];
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

extern "C" int pulsar_matmul_f32(
        void *out_dev, const void *w_dev, const void *x_dev,
        uint32_t in_dim, uint32_t out_dim, uint32_t n_tok) {
    if (in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    dim3 grid(out_dim, n_tok, 1);
    matmul_f32_kernel<<<grid, 256>>>(
            (float *)out_dev, (const float *)w_dev, (const float *)x_dev,
            in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul f32 launch");
}

__global__ static void swiglu_kernel(
        float *out, const float *gate, const float *up,
        uint32_t n, float clamp, float weight) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float u = up[i];
    if (clamp > 1.0e-6f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    float s = g / (1.0f + expf(-g));
    out[i] = s * u * weight;
}

extern "C" int pulsar_swiglu(
        void *out_dev, const void *gate_dev, const void *up_dev,
        uint32_t n, float clamp, float weight) {
    if (n == 0) return 0;
    swiglu_kernel<<<(n + 255u) / 256u, 256>>>(
            (float *)out_dev, (const float *)gate_dev, (const float *)up_dev,
            n, clamp, weight);
    return cuda_ok(cudaGetLastError(), "swiglu launch");
}

__global__ static void add_kernel(
        float *out, const float *a, const float *b, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i];
}

extern "C" int pulsar_add(
        void *out_dev, const void *a_dev, const void *b_dev, uint32_t n) {
    if (n == 0) return 0;
    add_kernel<<<(n + 255u) / 256u, 256>>>(
            (float *)out_dev, (const float *)a_dev, (const float *)b_dev, n);
    return cuda_ok(cudaGetLastError(), "add launch");
}

__global__ static void embed_tokens_q8_0_kernel(
        float *out,                /* [n_tok][n_embd] */
        const unsigned char *w,    /* q8_0 embedding matrix base */
        const int32_t *tokens,     /* [n_tok] */
        uint32_t n_embd,
        uint32_t n_vocab,
        uint32_t n_tok) {
    const uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (e >= n_embd || t >= n_tok) return;
    int32_t tok = tokens[t];
    if (tok < 0 || (uint32_t)tok >= n_vocab) tok = 0;
    const uint64_t row_bytes = (uint64_t)(n_embd / 32u) * 34u;
    const unsigned char *row = w + (uint64_t)(uint32_t)tok * row_bytes;
    const uint32_t blk = e >> 5;
    const uint32_t idx = e & 31u;
    const unsigned char *b = row + (uint64_t)blk * 34u;
    const float d = f16_to_f32(*(const uint16_t *)b);
    out[(uint64_t)t * n_embd + e] = d * (float)((const signed char *)(b + 2))[idx];
}

extern "C" int pulsar_embed_q8_0(
        void *out_dev, const void *w_dev, const void *tokens_dev,
        uint32_t n_embd, uint32_t n_vocab, uint32_t n_tok) {
    if (n_embd == 0 || n_embd % 32u != 0 || n_vocab == 0 || n_tok == 0) return 0;
    dim3 grid((n_embd + 255u) / 256u, n_tok, 1);
    embed_tokens_q8_0_kernel<<<grid, 256>>>(
            (float *)out_dev, (const unsigned char *)w_dev,
            (const int32_t *)tokens_dev, n_embd, n_vocab, n_tok);
    return cuda_ok(cudaGetLastError(), "embed q8_0 launch");
}

/* combined glue selftest vs CPU references */
extern "C" int pulsar_glue_selftest(void) {
    const uint32_t n = 512, rows = 3, n_vocab = 64;
    const float eps = 1e-5f;
    int ok = 1;
    float maxd;

    /* rms norm */
    {
        float *x = (float *)malloc(rows * n * sizeof(float));
        float *w = (float *)malloc(n * sizeof(float));
        float *ref = (float *)malloc(rows * n * sizeof(float));
        float *gpu = (float *)malloc(rows * n * sizeof(float));
        for (uint32_t i = 0; i < rows * n; i++) x[i] = gqa_test_randf();
        for (uint32_t i = 0; i < n; i++) w[i] = gqa_test_randf();
        for (uint32_t r = 0; r < rows; r++) {
            double sum = 0.0;
            for (uint32_t i = 0; i < n; i++) sum += (double)x[r * n + i] * x[r * n + i];
            float scale = (float)(1.0 / sqrt(sum / n + eps));
            for (uint32_t i = 0; i < n; i++) ref[r * n + i] = x[r * n + i] * scale * w[i];
        }
        void *x_d = NULL, *w_d = NULL, *o_d = NULL;
        ok = cuda_ok(cudaMalloc(&x_d, rows * n * 4), "x") &&
             cuda_ok(cudaMalloc(&w_d, n * 4), "w") &&
             cuda_ok(cudaMalloc(&o_d, rows * n * 4), "o") &&
             cuda_ok(cudaMemcpy(x_d, x, rows * n * 4, cudaMemcpyHostToDevice), "h2d") &&
             cuda_ok(cudaMemcpy(w_d, w, n * 4, cudaMemcpyHostToDevice), "h2d") &&
             pulsar_rms_norm(o_d, x_d, w_d, n, rows, eps) &&
             cuda_ok(cudaMemcpy(gpu, o_d, rows * n * 4, cudaMemcpyDeviceToHost), "d2h");
        maxd = 0.0f;
        if (ok) {
            for (uint32_t i = 0; i < rows * n; i++)
                maxd = fmaxf(maxd, fabsf(gpu[i] - ref[i]));
            ok = maxd <= 1e-5f;
        }
        fprintf(stderr, "glue-selftest rms_norm: %s (max diff %.2e)\n",
                ok ? "PASS" : "FAIL", (double)maxd);
        cudaFree(x_d); cudaFree(w_d); cudaFree(o_d);
        free(x); free(w); free(ref); free(gpu);
        if (!ok) return 0;
    }

    /* f32 matmul + swiglu + add, chained the way the dense FFN uses them */
    {
        const uint32_t in_dim = 512, out_dim = 128, n_tok = 2;
        float *w = (float *)malloc((uint64_t)out_dim * in_dim * 4);
        float *x = (float *)malloc((uint64_t)n_tok * in_dim * 4);
        float *ref_mm = (float *)malloc((uint64_t)n_tok * out_dim * 4);
        float *ref_sw = (float *)malloc((uint64_t)n_tok * out_dim * 4);
        float *ref_add = (float *)malloc((uint64_t)n_tok * out_dim * 4);
        float *gpu = (float *)malloc((uint64_t)n_tok * out_dim * 4);
        for (uint64_t i = 0; i < (uint64_t)out_dim * in_dim; i++) w[i] = gqa_test_randf();
        for (uint64_t i = 0; i < (uint64_t)n_tok * in_dim; i++) x[i] = gqa_test_randf();
        for (uint32_t t = 0; t < n_tok; t++)
            for (uint32_t r = 0; r < out_dim; r++) {
                double acc = 0.0;
                for (uint32_t i = 0; i < in_dim; i++)
                    acc += (double)w[(uint64_t)r * in_dim + i] * x[(uint64_t)t * in_dim + i];
                ref_mm[(uint64_t)t * out_dim + r] = (float)acc;
            }
        const uint32_t nel = n_tok * out_dim;
        for (uint32_t i = 0; i < nel; i++) {
            float g = ref_mm[i], u = ref_mm[i];
            float s = g / (1.0f + expf(-g));
            ref_sw[i] = s * u * 1.25f;
            ref_add[i] = ref_sw[i] + ref_mm[i];
        }
        void *w_d = NULL, *x_d = NULL, *mm_d = NULL, *sw_d = NULL, *add_d = NULL;
        ok = cuda_ok(cudaMalloc(&w_d, (uint64_t)out_dim * in_dim * 4), "w") &&
             cuda_ok(cudaMalloc(&x_d, (uint64_t)n_tok * in_dim * 4), "x") &&
             cuda_ok(cudaMalloc(&mm_d, nel * 4), "mm") &&
             cuda_ok(cudaMalloc(&sw_d, nel * 4), "sw") &&
             cuda_ok(cudaMalloc(&add_d, nel * 4), "add") &&
             cuda_ok(cudaMemcpy(w_d, w, (uint64_t)out_dim * in_dim * 4, cudaMemcpyHostToDevice), "h2d") &&
             cuda_ok(cudaMemcpy(x_d, x, (uint64_t)n_tok * in_dim * 4, cudaMemcpyHostToDevice), "h2d") &&
             pulsar_matmul_f32(mm_d, w_d, x_d, in_dim, out_dim, n_tok) &&
             pulsar_swiglu(sw_d, mm_d, mm_d, nel, 0.0f, 1.25f) &&
             pulsar_add(add_d, sw_d, mm_d, nel) &&
             cuda_ok(cudaMemcpy(gpu, add_d, nel * 4, cudaMemcpyDeviceToHost), "d2h");
        maxd = 0.0f;
        if (ok) {
            float maxref = 0.0f;
            for (uint32_t i = 0; i < nel; i++) {
                maxd = fmaxf(maxd, fabsf(gpu[i] - ref_add[i]));
                maxref = fmaxf(maxref, fabsf(ref_add[i]));
            }
            ok = maxd <= 1e-4f * fmaxf(maxref, 1.0f);
        }
        fprintf(stderr, "glue-selftest matmul_f32+swiglu+add: %s (max diff %.2e)\n",
                ok ? "PASS" : "FAIL", (double)maxd);
        cudaFree(w_d); cudaFree(x_d); cudaFree(mm_d); cudaFree(sw_d); cudaFree(add_d);
        free(w); free(x); free(ref_mm); free(ref_sw); free(ref_add); free(gpu);
        if (!ok) return 0;
    }

    /* q8_0 embedding lookup */
    {
        const uint32_t n_embd = 256, n_tok = 4;
        const uint64_t row_bytes = (uint64_t)(n_embd / 32u) * 34u;
        unsigned char *w = (unsigned char *)malloc((uint64_t)n_vocab * row_bytes);
        int32_t tokens[4] = {0, 5, 63, -1}; /* -1 clamps to 0 */
        float *ref = (float *)malloc((uint64_t)n_tok * n_embd * 4);
        float *gpu = (float *)malloc((uint64_t)n_tok * n_embd * 4);
        for (uint64_t i = 0; i < (uint64_t)n_vocab * row_bytes; i++)
            w[i] = test_randbyte();
        for (uint32_t v = 0; v < n_vocab; v++)
            for (uint32_t blk = 0; blk < n_embd / 32u; blk++) {
                uint16_t d = f32_to_f16_bits(gqa_test_randf() * 0.05f);
                memcpy(w + (uint64_t)v * row_bytes + (uint64_t)blk * 34u, &d, 2);
            }
        for (uint32_t t = 0; t < n_tok; t++) {
            int32_t tok = tokens[t];
            if (tok < 0 || (uint32_t)tok >= n_vocab) tok = 0;
            const unsigned char *row = w + (uint64_t)(uint32_t)tok * row_bytes;
            for (uint32_t e = 0; e < n_embd; e++) {
                const unsigned char *b = row + (uint64_t)(e >> 5) * 34u;
                uint16_t d16;
                memcpy(&d16, b, 2);
                ref[(uint64_t)t * n_embd + e] =
                    f16_to_f32_host(d16) * (float)((const signed char *)(b + 2))[e & 31u];
            }
        }
        void *w_d = NULL, *t_d = NULL, *o_d = NULL;
        ok = cuda_ok(cudaMalloc(&w_d, (uint64_t)n_vocab * row_bytes), "w") &&
             cuda_ok(cudaMalloc(&t_d, sizeof(tokens)), "t") &&
             cuda_ok(cudaMalloc(&o_d, (uint64_t)n_tok * n_embd * 4), "o") &&
             cuda_ok(cudaMemcpy(w_d, w, (uint64_t)n_vocab * row_bytes, cudaMemcpyHostToDevice), "h2d") &&
             cuda_ok(cudaMemcpy(t_d, tokens, sizeof(tokens), cudaMemcpyHostToDevice), "h2d") &&
             pulsar_embed_q8_0(o_d, w_d, t_d, n_embd, n_vocab, n_tok) &&
             cuda_ok(cudaMemcpy(gpu, o_d, (uint64_t)n_tok * n_embd * 4, cudaMemcpyDeviceToHost), "d2h");
        maxd = 0.0f;
        if (ok) {
            for (uint64_t i = 0; i < (uint64_t)n_tok * n_embd; i++)
                maxd = fmaxf(maxd, fabsf(gpu[i] - ref[i]));
            ok = maxd == 0.0f; /* pure lookup: bit-exact */
        }
        fprintf(stderr, "glue-selftest embed_q8_0: %s (max diff %.2e)\n",
                ok ? "PASS" : "FAIL", (double)maxd);
        cudaFree(w_d); cudaFree(t_d); cudaFree(o_d);
        free(w); free(ref); free(gpu);
    }
    return ok;
}

extern "C" int pulsar_gqa_selftest(void) { return ds4_gpu_gqa_selftest(); }
