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

/* ---- Q8_0 matmul, ds4 dp4a path ----------------------------------------
 * Activations are quantized to q8_0 first (quantize_q8_0_f32_kernel), then
 * the dot runs int8 x int8 via dp4a - exactly ds4's math, so decode logits
 * match the reference engine. Kernels are verbatim from ds4_cuda.cu. */

typedef struct __align__(2) {
    uint16_t scale_f16;
    int8_t q[32];
} q8_0_block;

__device__ static float f16_to_f32(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

__device__ static float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ static int32_t load_i8x4_i32_aligned(const int8_t *p) {
    return *(const int32_t *)p;
}

__device__ __forceinline__ static int32_t load_i8x4_i32_unaligned(const int8_t *p) {
    const uint8_t *u = (const uint8_t *)p;
    return (int32_t)((uint32_t)u[0] |
                     ((uint32_t)u[1] << 8) |
                     ((uint32_t)u[2] << 16) |
                     ((uint32_t)u[3] << 24));
}

__device__ __forceinline__ static int32_t dot_i8x32_dp4a(const int8_t *a, const int8_t *b) {
    int32_t dot = 0;
#pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        dot = __dp4a(load_i8x4_i32_unaligned(a + i), load_i8x4_i32_aligned(b + i), dot);
    }
    return dot;
}

__global__ static void quantize_q8_0_f32_kernel(
        int8_t *xq,
        float *xscale,
        const float *x,
        uint64_t in_dim,
        uint64_t blocks) {
    uint64_t b = blockIdx.x;
    uint64_t tok = blockIdx.y;
    if (b >= blocks) return;
    uint64_t i0 = b * 32;
    uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
    const float *xr = x + tok * in_dim + i0;

    float a = 0.0f;
    if (threadIdx.x < bn) a = fabsf(xr[threadIdx.x]);
    __shared__ float vals[32];
    vals[threadIdx.x] = a;
    __syncthreads();
    for (uint32_t stride = 16; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) vals[threadIdx.x] = fmaxf(vals[threadIdx.x], vals[threadIdx.x + stride]);
        __syncthreads();
    }
    const float d = vals[0] / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (threadIdx.x == 0) xscale[tok * blocks + b] = d;
    int8_t *dst = xq + (tok * blocks + b) * 32;
    if (threadIdx.x < bn) {
        int v = (int)lrintf(xr[threadIdx.x] * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        dst[threadIdx.x] = (int8_t)v;
    } else {
        dst[threadIdx.x] = 0;
    }
}

__global__ static void matmul_q8_0_preq_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        int dot = dot_i8x32_dp4a(qs, xqr + b * 32);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void matmul_q8_0_preq_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        int dot = dot_i8x32_dp4a(qs, xq + b * 32);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[row] = acc;
}

/* grow-only PER-DEVICE scratch for activation prequant: matmuls run on
 * whichever device is current (attn GPU vs expert GPU), and VRAM is only
 * dereferenceable on its own device without P2P.
 * ponytail: single scratch per device, single-stream engine; pool it
 * per-stream if pulsar ever runs concurrent graphs. */
#define PULSAR_MAX_DEVICES 16
static void *g_preq_scratch[PULSAR_MAX_DEVICES];
static uint64_t g_preq_scratch_cap[PULSAR_MAX_DEVICES];

static void *preq_scratch(uint64_t bytes) {
    int dev = 0;
    (void)cudaGetDevice(&dev);
    if (dev < 0 || dev >= PULSAR_MAX_DEVICES) return NULL;
    if (bytes <= g_preq_scratch_cap[dev]) return g_preq_scratch[dev];
    if (g_preq_scratch[dev]) (void)cudaFree(g_preq_scratch[dev]);
    g_preq_scratch[dev] = NULL;
    g_preq_scratch_cap[dev] = 0;
    if (!cuda_ok(cudaMalloc(&g_preq_scratch[dev], bytes), "preq scratch alloc")) return NULL;
    g_preq_scratch_cap[dev] = bytes;
    return g_preq_scratch[dev];
}

extern "C" int pulsar_q8_0_matmul(
        void *out_dev,
        const void *w_dev,
        const void *x_dev,
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t n_tok) {
    if (in_dim == 0 || in_dim % 32u != 0 || out_dim == 0 || n_tok == 0) return 0;
    const uint64_t blocks = in_dim / 32u;
    const uint64_t xq_bytes = (uint64_t)n_tok * blocks * 32u;
    const uint64_t scale_off = (xq_bytes + 15u) & ~15ull;
    void *tmp = preq_scratch(scale_off + (uint64_t)n_tok * blocks * sizeof(float));
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_off);

    dim3 qgrid((unsigned)blocks, n_tok, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, (const float *)x_dev, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "q8_0 prequant launch")) return 0;

    if (n_tok == 1) {
        matmul_q8_0_preq_warp8_kernel<<<(out_dim + 7u) / 8u, 256>>>(
                (float *)out_dev, (const unsigned char *)w_dev, xq, xscale,
                in_dim, out_dim, blocks);
    } else {
        dim3 grid(out_dim, n_tok, 1);
        matmul_q8_0_preq_kernel<<<grid, 256>>>(
                (float *)out_dev, (const unsigned char *)w_dev, xq, xscale,
                in_dim, out_dim, n_tok, blocks);
    }
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
    /* reference: mirror the GPU path exactly - quantize activations to
     * q8_0 per 32-block, integer dot, scale product */
    int8_t *xq = (int8_t *)malloc((uint64_t)n_tok * in_dim);
    float *xd = (float *)malloc((uint64_t)n_tok * blocks * sizeof(float));
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t b = 0; b < blocks; b++) {
            const float *xb = x + (uint64_t)t * in_dim + b * 32u;
            float amax = 0.0f;
            for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(xb[i]));
            const float d = amax / 127.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            xd[(uint64_t)t * blocks + b] = d;
            for (int i = 0; i < 32; i++) {
                int v = (int)lrintf(xb[i] * id);
                v = v > 127 ? 127 : (v < -128 ? -128 : v);
                xq[(uint64_t)t * in_dim + b * 32u + i] = (int8_t)v;
            }
        }
    }
    for (uint32_t t = 0; t < n_tok; t++)
        for (uint32_t r = 0; r < out_dim; r++) {
            float acc = 0.0f;
            for (uint32_t b = 0; b < blocks; b++) {
                const q8_0_block *blk = &w[(uint64_t)r * blocks + b];
                int32_t dot = 0;
                for (int i = 0; i < 32; i++)
                    dot += (int32_t)blk->q[i] * (int32_t)xq[(uint64_t)t * in_dim + b * 32u + i];
                acc += f16_to_f32_host(blk->scale_f16) * xd[(uint64_t)t * blocks + b] * (float)dot;
            }
            ref[(uint64_t)t * out_dim + r] = acc;
        }
    free(xq);
    free(xd);

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

/* Activations for expert dots are quantized to q8_K (ggml layout: f32
 * scale, 256 int8, 16 block sums), then the dots run integer dp4a -
 * ds4's exact math. Device functions verbatim from ds4_cuda.cu. */

typedef struct {
    float d;
    int8_t qs[PULSAR_QK_K];
    int16_t bsums[PULSAR_QK_K / 16];
} block_q8_K;

__global__ static void q8_K_quantize_kernel(block_q8_K *out, const float *x, uint32_t in_dim, uint32_t n_rows) {
    uint32_t b = blockIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= n_rows || b >= in_dim / PULSAR_QK_K) return;
    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * PULSAR_QK_K;
    block_q8_K *yb = out + (uint64_t)row * (in_dim / PULSAR_QK_K) + b;
    __shared__ float abs_part[256];
    __shared__ float val_part[256];
    __shared__ float maxv_s;
    __shared__ float iscale_s;
    uint32_t tid = threadIdx.x;
    float v = tid < PULSAR_QK_K ? xr[tid] : 0.0f;
    abs_part[tid] = tid < PULSAR_QK_K ? fabsf(v) : 0.0f;
    val_part[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
            abs_part[tid] = abs_part[tid + stride];
            val_part[tid] = val_part[tid + stride];
        }
        __syncthreads();
    }
    float amax = abs_part[0];
    if (amax == 0.0f) {
        if (tid == 0) yb->d = 0.0f;
        if (tid < PULSAR_QK_K) yb->qs[tid] = 0;
        if (tid < PULSAR_QK_K / 16) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0) {
        maxv_s = val_part[0];
        iscale_s = -127.0f / maxv_s;
    }
    __syncthreads();
    if (tid < PULSAR_QK_K) {
        int qv = (int)lrintf(iscale_s * xr[tid]);
        if (qv > 127) qv = 127;
        if (qv < -128) qv = -128;
        yb->qs[tid] = (int8_t)qv;
    }
    __syncthreads();
    if (tid < PULSAR_QK_K / 16) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0) yb->d = 1.0f / iscale_s;
}

extern "C" int pulsar_quantize_q8_K(
        void *out_dev, const void *x_dev, uint32_t in_dim, uint32_t n_rows) {
    if (in_dim == 0 || in_dim % PULSAR_QK_K != 0 || n_rows == 0) return 0;
    dim3 grid(in_dim / PULSAR_QK_K, n_rows, 1);
    q8_K_quantize_kernel<<<grid, 256>>>(
            (block_q8_K *)out_dev, (const float *)x_dev, in_dim, n_rows);
    return cuda_ok(cudaGetLastError(), "q8_K quantize launch");
}

__device__ __forceinline__ static uint32_t dev_unpack_iq2_signs(uint32_t v) {
    const uint32_t p = __popc(v) & 1u;
    const uint32_t s = v ^ (p << 7u);
    return s * 0x01010101u;
}

__device__ __forceinline__ static void dev_iq2_i8x8_lut(
        const uint64_t *grid,
        const uint8_t *signs,
        uint8_t grid_idx,
        uint32_t sign_idx,
        int32_t *w0,
        int32_t *w1) {
    const uint32_t s = dev_unpack_iq2_signs(signs[sign_idx]);
    const int32_t sm0 = __vcmpne4(s & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(s & 0x80402010u, 0);
    const uint64_t g = grid[grid_idx];
    *w0 = __vsub4((int32_t)(uint32_t)g ^ sm0, sm0);
    *w1 = __vsub4((int32_t)(uint32_t)(g >> 32) ^ sm1, sm1);
}

__device__ static float dev_dot_iq2_xxs_q8_K_block_lut(
        const block_iq2_xxs *x,
        const block_q8_K *y,
        const uint64_t *grid,
        const uint8_t *signs) {
    const float xd = f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < PULSAR_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)(aux0 & 0xffu),           (aux1 >> 0)  & 127u, &w[0], &w[1]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 8)  & 0xffu),   (aux1 >> 7)  & 127u, &w[2], &w[3]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 16) & 0xffu),   (aux1 >> 14) & 127u, &w[4], &w[5]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 24) & 0xffu),   (aux1 >> 21) & 127u, &w[6], &w[7]);
        int32_t sumi = 0;
        sumi = __dp4a(w[0], *(const int32_t *)(q8 + ib32 * 32u + 0),  sumi);
        sumi = __dp4a(w[1], *(const int32_t *)(q8 + ib32 * 32u + 4),  sumi);
        sumi = __dp4a(w[2], *(const int32_t *)(q8 + ib32 * 32u + 8),  sumi);
        sumi = __dp4a(w[3], *(const int32_t *)(q8 + ib32 * 32u + 12), sumi);
        sumi = __dp4a(w[4], *(const int32_t *)(q8 + ib32 * 32u + 16), sumi);
        sumi = __dp4a(w[5], *(const int32_t *)(q8 + ib32 * 32u + 20), sumi);
        sumi = __dp4a(w[6], *(const int32_t *)(q8 + ib32 * 32u + 24), sumi);
        sumi = __dp4a(w[7], *(const int32_t *)(q8 + ib32 * 32u + 28), sumi);
        bsum += sumi * ls;
    }
    return 0.125f * xd * y->d * (float)bsum;
}

__device__ __forceinline__ static int32_t dev_dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 16; i += 4) {
        const int32_t v = (*(const int32_t *)(q2 + i) >> shift) & 0x03030303;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ static float dev_dot_q2_K_q8_K_block(const block_q2_K *x, const block_q8_K *y) {
    const uint8_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    const uint8_t *sc = x->scales;
    int summs = 0;
    for (int j = 0; j < 16; j++) summs += y->bsums[j] * (sc[j] >> 4);
    const float dall = y->d * f16_to_f32(x->d);
    const float dmin = y->d * f16_to_f32(x->dmin);
    int isum = 0;
    int is = 0;
    for (int k = 0; k < (int)(PULSAR_QK_K / 128); k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            int d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2, q8, shift);
            d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)isum - dmin * (float)summs;
}

/* per-block dot functors for the templated MoE kernels */
struct dot_iq2_xxs {
    __device__ __forceinline__ static float block(const char *row, const block_q8_K *xq, uint32_t b) {
        return dev_dot_iq2_xxs_q8_K_block_lut(
                (const block_iq2_xxs *)row + b, xq + b,
                cuda_iq2xxs_grid, cuda_ksigns_iq2xs);
    }
};

struct dot_q2_K {
    __device__ __forceinline__ static float block(const char *row, const block_q8_K *xq, uint32_t b) {
        return dev_dot_q2_K_q8_K_block((const block_q2_K *)row + b, xq + b);
    }
};

typedef struct {
    const void *gate;
    const void *up;
    const void *down;
} pulsar_expert_ptrs;

template <typename DOT>
__global__ static void moe_pair_swiglu_kernel(
        float *mid,                     /* [n_tok][n_used][mid_dim] */
        const pulsar_expert_ptrs *ptrs, /* [n_tok][n_used] */
        const float *weights,           /* [n_tok][n_used] */
        const block_q8_K *xq,           /* [n_tok][in_dim/256] */
        uint32_t in_blocks,
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
    const block_q8_K *token_xq = xq + (uint64_t)token * in_blocks;

    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    for (uint32_t b = lane; b < in_blocks; b += 32u) {
        acc_gate += DOT::block(gate_row, token_xq, b);
        acc_up += DOT::block(up_row, token_xq, b);
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

template <typename DOT>
__global__ static void moe_down_kernel(
        float *out,                     /* [n_tok][out_dim] */
        const pulsar_expert_ptrs *ptrs, /* [n_tok][n_used] */
        const block_q8_K *midq,         /* [n_tok][n_used][mid_dim/256] */
        uint32_t mid_blocks,
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
        const block_q8_K *slot_midq = midq + (slot_base + slot) * mid_blocks;
        for (uint32_t b = lane; b < mid_blocks; b += 32u) {
            acc += DOT::block(down_row, slot_midq, b);
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
        const void *xq_dev,        /* q8_K [n_tok][in_dim/256] */
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
    const uint32_t in_blocks = in_dim / PULSAR_QK_K;
    dim3 block(32, 4, 1);
    dim3 grid((mid_dim + 3u) / 4u, n_used, n_tok);
    switch (quant) {
    case PULSAR_QUANT_Q2_K:
        moe_pair_swiglu_kernel<dot_q2_K><<<grid, block>>>(
                (float *)mid_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const float *)weights_dev, (const block_q8_K *)xq_dev,
                in_blocks, mid_dim, n_used, n_tok, row_bytes);
        break;
    case PULSAR_QUANT_IQ2_XXS:
        moe_pair_swiglu_kernel<dot_iq2_xxs><<<grid, block>>>(
                (float *)mid_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const float *)weights_dev, (const block_q8_K *)xq_dev,
                in_blocks, mid_dim, n_used, n_tok, row_bytes);
        break;
    default:
        return 0;
    }
    return cuda_ok(cudaGetLastError(), "moe pair swiglu launch");
}

extern "C" int pulsar_moe_down(
        void *out_dev,
        const void *ptrs_dev,
        const void *midq_dev,      /* q8_K [n_tok][n_used][mid_dim/256] */
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
    const uint32_t mid_blocks = mid_dim / PULSAR_QK_K;
    dim3 block(32, 4, 1);
    dim3 grid((out_dim + 3u) / 4u, n_tok, 1);
    switch (quant) {
    case PULSAR_QUANT_Q2_K:
        moe_down_kernel<dot_q2_K><<<grid, block>>>(
                (float *)out_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const block_q8_K *)midq_dev, mid_blocks, out_dim, n_used,
                n_tok, row_bytes);
        break;
    case PULSAR_QUANT_IQ2_XXS:
        moe_down_kernel<dot_iq2_xxs><<<grid, block>>>(
                (float *)out_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const block_q8_K *)midq_dev, mid_blocks, out_dim, n_used,
                n_tok, row_bytes);
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

/* host mirrors of the device quantizer and integer block dots; tables
 * fetched from the device so both sides read identical constants */
static uint8_t h_ksigns[128];
static uint64_t h_grid[256];

/* mirror of q8_K_quantize_kernel, incl. the first-max tiebreak */
static void host_quantize_q8_K(block_q8_K *out, const float *x,
                               uint32_t in_dim, uint32_t n_rows) {
    for (uint32_t row = 0; row < n_rows; row++) {
        for (uint32_t b = 0; b < in_dim / PULSAR_QK_K; b++) {
            const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * PULSAR_QK_K;
            block_q8_K *yb = out + (uint64_t)row * (in_dim / PULSAR_QK_K) + b;
            float amax = 0.0f, maxv = 0.0f;
            for (uint32_t i = 0; i < PULSAR_QK_K; i++) {
                const float a = fabsf(xr[i]);
                if (a > amax) { amax = a; maxv = xr[i]; }
            }
            if (amax == 0.0f) {
                memset(yb, 0, sizeof(*yb));
                continue;
            }
            const float iscale = -127.0f / maxv;
            for (uint32_t i = 0; i < PULSAR_QK_K; i++) {
                int qv = (int)lrintf(iscale * xr[i]);
                if (qv > 127) qv = 127;
                if (qv < -128) qv = -128;
                yb->qs[i] = (int8_t)qv;
            }
            for (uint32_t j = 0; j < PULSAR_QK_K / 16; j++) {
                int sum = 0;
                for (int i = 0; i < 16; i++) sum += yb->qs[j * 16 + i];
                yb->bsums[j] = (int16_t)sum;
            }
            yb->d = 1.0f / iscale;
        }
    }
}

static float host_dot_iq2_xxs_block(const char *row, const block_q8_K *xq, uint32_t b) {
    const block_iq2_xxs *xb = (const block_iq2_xxs *)row + b;
    const block_q8_K *y = xq + b;
    int64_t bsum = 0;
    for (uint32_t ib32 = 0; ib32 < PULSAR_QK_K / 32; ib32++) {
        const uint16_t *q2 = xb->qs + 4u * ib32;
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t sumi = 0;
        for (uint32_t kk = 0; kk < 32; kk++) {
            const uint32_t l = kk >> 3, j = kk & 7u;
            const uint8_t grid_idx = (uint8_t)((aux0 >> (8u * l)) & 0xffu);
            const uint32_t sign_idx = (aux1 >> (7u * l)) & 127u;
            int32_t w = (int32_t)(uint8_t)(h_grid[grid_idx] >> (8u * j));
            if (h_ksigns[sign_idx] & (1u << j)) w = -w;
            sumi += w * (int32_t)y->qs[ib32 * 32 + kk];
        }
        bsum += (int64_t)sumi * ls;
    }
    return 0.125f * f16_to_f32_host(xb->d) * y->d * (float)bsum;
}

static float host_dot_q2_K_block(const char *row, const block_q8_K *xq, uint32_t b) {
    const block_q2_K *xb = (const block_q2_K *)row + b;
    const block_q8_K *y = xq + b;
    const uint8_t *sc = xb->scales;
    int summs = 0;
    for (int j = 0; j < 16; j++) summs += y->bsums[j] * (sc[j] >> 4);
    int isum = 0;
    int is = 0;
    const uint8_t *q2 = xb->qs;
    const int8_t *q8 = y->qs;
    for (int k = 0; k < (int)(PULSAR_QK_K / 128); k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            for (int half = 0; half < 2; half++) {
                const int d = sc[is++] & 0x0f;
                int sum16 = 0;
                for (int i = 0; i < 16; i++)
                    sum16 += ((q2[half * 16 + i] >> shift) & 3) * (int)q8[half * 16 + i];
                isum += d * sum16;
            }
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return y->d * f16_to_f32_host(xb->d) * (float)isum -
           y->d * f16_to_f32_host(xb->dmin) * (float)summs;
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

/* GPU q8_K quantizer vs the host mirror. Not bit-exact: both pulsar and
 * ds4 build with --use_fast_math, so device division is approximate; allow
 * +-1 on quants at rounding boundaries and last-ulp scale drift, and check
 * bsums are self-consistent with the GPU quants. */
static int q8_K_quantize_selftest(void) {
    const uint32_t in_dim = 512, n_rows = 5;
    const uint32_t blocks = in_dim / PULSAR_QK_K;
    float *x = (float *)malloc((uint64_t)n_rows * in_dim * sizeof(float));
    block_q8_K *ref = (block_q8_K *)malloc((uint64_t)n_rows * blocks * sizeof(block_q8_K));
    block_q8_K *gpu = (block_q8_K *)malloc((uint64_t)n_rows * blocks * sizeof(block_q8_K));
    for (uint64_t i = 0; i < (uint64_t)n_rows * in_dim; i++) x[i] = gqa_test_randf() * 3.0f;
    host_quantize_q8_K(ref, x, in_dim, n_rows);

    void *x_dev = NULL, *q_dev = NULL;
    const uint64_t q_bytes = (uint64_t)n_rows * blocks * sizeof(block_q8_K);
    int ok = cuda_ok(cudaMalloc(&x_dev, (uint64_t)n_rows * in_dim * 4), "x alloc") &&
             cuda_ok(cudaMalloc(&q_dev, q_bytes), "q alloc") &&
             cuda_ok(cudaMemcpy(x_dev, x, (uint64_t)n_rows * in_dim * 4, cudaMemcpyHostToDevice), "x h2d") &&
             pulsar_quantize_q8_K(q_dev, x_dev, in_dim, n_rows) &&
             cuda_ok(cudaDeviceSynchronize(), "sync") &&
             cuda_ok(cudaMemcpy(gpu, q_dev, q_bytes, cudaMemcpyDeviceToHost), "q d2h");
    uint64_t q_off = 0;
    float d_maxrel = 0.0f;
    if (ok) {
        for (uint32_t bi = 0; bi < n_rows * blocks && ok; bi++) {
            const block_q8_K *r = &ref[bi], *g = &gpu[bi];
            const float dr = fabsf(g->d - r->d) / fmaxf(fabsf(r->d), 1e-30f);
            if (dr > d_maxrel) d_maxrel = dr;
            ok = dr <= 4e-7f;
            for (uint32_t i = 0; i < PULSAR_QK_K && ok; i++) {
                const int diff = abs((int)g->qs[i] - (int)r->qs[i]);
                if (diff > 0) q_off++;
                ok = diff <= 1;
            }
            for (uint32_t j = 0; j < PULSAR_QK_K / 16 && ok; j++) {
                int sum = 0;
                for (int i = 0; i < 16; i++) sum += g->qs[j * 16 + i];
                ok = g->bsums[j] == (int16_t)sum;
            }
        }
    }
    fprintf(stderr, "q8_K-quantize-selftest: %s (quants off-by-one %llu/%u, d max rel %.2e)\n",
            ok ? "PASS" : "FAIL", (unsigned long long)q_off,
            n_rows * blocks * PULSAR_QK_K, (double)d_maxrel);
    if (x_dev) cudaFree(x_dev);
    if (q_dev) cudaFree(q_dev);
    free(x); free(ref); free(gpu);
    return ok;
}

static int moe_selftest_one(uint32_t quant, const char *name) {
    const uint32_t in_dim = 512, mid_dim = 256, out_dim = 320;
    const uint32_t n_expert = 8, n_used = 4, n_tok = 3;
    const uint32_t in_blocks = in_dim / PULSAR_QK_K;
    const uint32_t mid_blocks = mid_dim / PULSAR_QK_K;
    const uint64_t pair_row_bytes = (uint64_t)in_blocks *
        (quant == PULSAR_QUANT_Q2_K ? sizeof(block_q2_K) : sizeof(block_iq2_xxs));
    const uint64_t down_row_bytes = (uint64_t)mid_blocks *
        (quant == PULSAR_QUANT_Q2_K ? sizeof(block_q2_K) : sizeof(block_iq2_xxs));
    const uint64_t gate_slab_bytes = (uint64_t)n_expert * mid_dim * pair_row_bytes;
    const uint64_t down_slab_bytes = (uint64_t)n_expert * out_dim * down_row_bytes;
    float (*dot)(const char *, const block_q8_K *, uint32_t) =
        quant == PULSAR_QUANT_Q2_K ? host_dot_q2_K_block : host_dot_iq2_xxs_block;

    char *gate = (char *)malloc(gate_slab_bytes);
    char *up = (char *)malloc(gate_slab_bytes);
    char *down = (char *)malloc(down_slab_bytes);
    float *x = (float *)malloc((uint64_t)n_tok * in_dim * sizeof(float));
    float *w = (float *)malloc((uint64_t)n_tok * n_used * sizeof(float));
    int32_t *sel = (int32_t *)malloc((uint64_t)n_tok * n_used * sizeof(int32_t));
    block_q8_K *xq = (block_q8_K *)malloc((uint64_t)n_tok * in_blocks * sizeof(block_q8_K));
    float *mid_host = (float *)malloc((uint64_t)n_tok * n_used * mid_dim * sizeof(float));
    block_q8_K *midq = (block_q8_K *)malloc((uint64_t)n_tok * n_used * mid_blocks * sizeof(block_q8_K));
    float *mid_ref = (float *)calloc((uint64_t)n_tok * n_used * mid_dim, sizeof(float));
    float *out_ref = (float *)calloc((uint64_t)n_tok * out_dim, sizeof(float));
    float *mid_gpu = (float *)malloc((uint64_t)n_tok * n_used * mid_dim * sizeof(float));
    float *out_gpu = (float *)malloc((uint64_t)n_tok * out_dim * sizeof(float));

    fill_slab(gate, n_expert * mid_dim, in_dim, pair_row_bytes, quant);
    fill_slab(up, n_expert * mid_dim, in_dim, pair_row_bytes, quant);
    fill_slab(down, n_expert * out_dim, mid_dim, down_row_bytes, quant);
    for (uint64_t i = 0; i < (uint64_t)n_tok * in_dim; i++) x[i] = gqa_test_randf();
    for (uint64_t i = 0; i < (uint64_t)n_tok * n_used * mid_dim; i++)
        mid_host[i] = gqa_test_randf();
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t s = 0; s < n_used; s++) {
            sel[t * n_used + s] = (int32_t)(test_randbyte() % n_expert);
            w[t * n_used + s] = fabsf(gqa_test_randf()) + 0.1f;
        }
    }
    sel[1 * n_used + 2] = -1; /* one unrouted slot: NULL ptrs, zero output */

    /* both sides consume the same host-quantized activations (the GPU
     * quantizer is proven bit-exact separately) */
    host_quantize_q8_K(xq, x, in_dim, n_tok);
    host_quantize_q8_K(midq, mid_host, mid_dim, n_tok * n_used);

    void *gate_dev = NULL, *up_dev = NULL, *down_dev = NULL;
    void *xq_dev = NULL, *midq_dev = NULL, *w_dev = NULL, *ptrs_dev = NULL;
    void *mid_dev = NULL, *out_dev = NULL;
    pulsar_expert_ptrs ptrs[n_tok * n_used];
    const uint64_t xq_bytes = (uint64_t)n_tok * in_blocks * sizeof(block_q8_K);
    const uint64_t midq_bytes = (uint64_t)n_tok * n_used * mid_blocks * sizeof(block_q8_K);
    int ok = cuda_ok(cudaMalloc(&gate_dev, gate_slab_bytes), "gate alloc") &&
             cuda_ok(cudaMalloc(&up_dev, gate_slab_bytes), "up alloc") &&
             cuda_ok(cudaMalloc(&down_dev, down_slab_bytes), "down alloc") &&
             cuda_ok(cudaMalloc(&xq_dev, xq_bytes), "xq alloc") &&
             cuda_ok(cudaMalloc(&midq_dev, midq_bytes), "midq alloc") &&
             cuda_ok(cudaMalloc(&w_dev, (uint64_t)n_tok * n_used * sizeof(float)), "w alloc") &&
             cuda_ok(cudaMalloc(&ptrs_dev, sizeof(ptrs)), "ptrs alloc") &&
             cuda_ok(cudaMalloc(&mid_dev, (uint64_t)n_tok * n_used * mid_dim * sizeof(float)), "mid alloc") &&
             cuda_ok(cudaMalloc(&out_dev, (uint64_t)n_tok * out_dim * sizeof(float)), "out alloc") &&
             cuda_ok(cudaMemcpy(gate_dev, gate, gate_slab_bytes, cudaMemcpyHostToDevice), "gate h2d") &&
             cuda_ok(cudaMemcpy(up_dev, up, gate_slab_bytes, cudaMemcpyHostToDevice), "up h2d") &&
             cuda_ok(cudaMemcpy(down_dev, down, down_slab_bytes, cudaMemcpyHostToDevice), "down h2d") &&
             cuda_ok(cudaMemcpy(xq_dev, xq, xq_bytes, cudaMemcpyHostToDevice), "xq h2d") &&
             cuda_ok(cudaMemcpy(midq_dev, midq, midq_bytes, cudaMemcpyHostToDevice), "midq h2d") &&
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
         pulsar_moe_pair_swiglu(mid_dev, ptrs_dev, w_dev, xq_dev,
                                in_dim, mid_dim, n_used, n_tok,
                                pair_row_bytes, quant) &&
         pulsar_moe_down(out_dev, ptrs_dev, midq_dev, mid_dim, out_dim,
                         n_used, n_tok, down_row_bytes, quant) &&
         cuda_ok(cudaDeviceSynchronize(), "sync") &&
         cuda_ok(cudaMemcpy(mid_gpu, mid_dev, (uint64_t)n_tok * n_used * mid_dim * sizeof(float), cudaMemcpyDeviceToHost), "mid d2h") &&
         cuda_ok(cudaMemcpy(out_gpu, out_dev, (uint64_t)n_tok * out_dim * sizeof(float), cudaMemcpyDeviceToHost), "out d2h");

    /* host reference: same integer block dots, f32 accumulation */
    for (uint32_t t = 0; t < n_tok && ok; t++) {
        for (uint32_t s = 0; s < n_used; s++) {
            const int32_t e = sel[t * n_used + s];
            if (e < 0) continue;
            const char *gs = gate + (uint64_t)e * mid_dim * pair_row_bytes;
            const char *us = up + (uint64_t)e * mid_dim * pair_row_bytes;
            const block_q8_K *txq = xq + (uint64_t)t * in_blocks;
            for (uint32_t r = 0; r < mid_dim; r++) {
                float ag = 0.0f, au = 0.0f;
                for (uint32_t b = 0; b < in_blocks; b++) {
                    ag += dot(gs + (uint64_t)r * pair_row_bytes, txq, b);
                    au += dot(us + (uint64_t)r * pair_row_bytes, txq, b);
                }
                const float sw = ag / (1.0f + expf(-ag));
                mid_ref[((uint64_t)t * n_used + s) * mid_dim + r] =
                    sw * au * w[t * n_used + s];
            }
        }
        for (uint32_t r = 0; r < out_dim; r++) {
            float acc = 0.0f;
            for (uint32_t s = 0; s < n_used; s++) {
                const int32_t e = sel[t * n_used + s];
                if (e < 0) continue;
                const char *dr = down + (uint64_t)e * out_dim * down_row_bytes +
                                 (uint64_t)r * down_row_bytes;
                const block_q8_K *smq = midq + ((uint64_t)t * n_used + s) * mid_blocks;
                for (uint32_t b = 0; b < mid_blocks; b++)
                    acc += dot(dr, smq, b);
            }
            out_ref[(uint64_t)t * out_dim + r] = acc;
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
             out_maxd <= 1e-3f * (out_maxref > 1.0f ? out_maxref : 1.0f);
    }
    fprintf(stderr, "moe-selftest %s: %s (mid max diff %.2e, out max diff %.2e, max |out ref| %.2e)\n",
            name, ok ? "PASS" : "FAIL", (double)mid_maxd, (double)out_maxd,
            (double)out_maxref);
    if (gate_dev) cudaFree(gate_dev);
    if (up_dev) cudaFree(up_dev);
    if (down_dev) cudaFree(down_dev);
    if (xq_dev) cudaFree(xq_dev);
    if (midq_dev) cudaFree(midq_dev);
    if (w_dev) cudaFree(w_dev);
    if (ptrs_dev) cudaFree(ptrs_dev);
    if (mid_dev) cudaFree(mid_dev);
    if (out_dev) cudaFree(out_dev);
    free(gate); free(up); free(down); free(x); free(w); free(sel);
    free(xq); free(mid_host); free(midq);
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
    return q8_K_quantize_selftest() &&
           moe_selftest_one(PULSAR_QUANT_Q2_K, "q2_K") &&
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

/* ---- raw-pointer wrappers over the gqa inc (its wrappers take shim
 * tensors; Rust passes device pointers) ---------------------------------- */

static ds4_gpu_tensor shim(const void *ptr) {
    ds4_gpu_tensor t;
    t.ptr = (void *)ptr;
    t.bytes = UINT64_MAX; /* the inc wrappers never consult bytes */
    return t;
}

extern "C" int pulsar_gqa_head_rms_norm(
        void *x, const void *w, uint32_t rows, uint32_t head_dim, float eps) {
    ds4_gpu_tensor xt = shim(x), wt = shim(w);
    return ds4_gpu_gqa_head_rms_norm_weight(&xt, &wt, rows, head_dim, eps);
}

extern "C" int pulsar_gqa_rope(
        void *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
        uint32_t pos0, float theta) {
    ds4_gpu_tensor xt = shim(x);
    return ds4_gpu_gqa_rope(&xt, n_tok, n_head, head_dim, pos0, theta);
}

extern "C" int pulsar_gqa_kv_append(
        void *cache, const void *kv, uint32_t n_tok, uint32_t n_kv_head,
        uint32_t head_dim, uint32_t cap, uint32_t pos0) {
    ds4_gpu_tensor ct = shim(cache), kt = shim(kv);
    return ds4_gpu_gqa_kv_cache_append(&ct, &kt, n_tok, n_kv_head, head_dim,
                                       cap, pos0);
}

extern "C" int pulsar_gqa_attention(
        void *out, const void *q, const void *k_cache, const void *v_cache,
        uint32_t n_tok, uint32_t n_head, uint32_t n_kv_head,
        uint32_t head_dim, uint32_t cap, uint32_t pos0) {
    ds4_gpu_tensor ot = shim(out), qt = shim(q), kt = shim(k_cache),
                   vt = shim(v_cache);
    return ds4_gpu_gqa_attention(&ot, &qt, &kt, &vt, n_tok, n_head,
                                 n_kv_head, head_dim, cap, pos0);
}

extern "C" int pulsar_gqa_selftest(void) { return ds4_gpu_gqa_selftest(); }

#include "mla_kernels.inc"

/* ---- MLA selftest: full compact-path chain vs a host reference --------- */

static float mla_host_q8_dot(const uint8_t *row, const float *x, uint32_t n) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < (n + 31u) / 32u; b++) {
        uint16_t d16;
        memcpy(&d16, row + (uint64_t)b * 34u, 2);
        const float d = f16_to_f32_host(d16);
        const int8_t *qs = (const int8_t *)(row + (uint64_t)b * 34u + 2u);
        const uint32_t base = b * 32u;
        const uint32_t count = n - base < 32u ? n - base : 32u;
        for (uint32_t i = 0; i < count; i++) acc += d * (float)qs[i] * x[base + i];
    }
    return acc;
}

static void mla_host_yarn(float theta_extrap, float freq_scale, float c0,
                          float c1, int i0, float ext, float mscale,
                          float *c, float *s) {
    const float interp = freq_scale * theta_extrap;
    float theta = interp;
    if (ext != 0.0f) {
        const float y = ((float)(i0 / 2) - c0) / fmaxf(0.001f, c1 - c0);
        const float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext;
        theta = interp * (1.0f - ramp) + theta_extrap * ramp;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *c = cosf(theta) * mscale;
    *s = sinf(theta) * mscale;
}

static void mla_host_corr(uint32_t n_dims, uint32_t n_ctx_orig, float fb,
                          float bfast, float bslow, float *c0, float *c1) {
    const float denom = 2.0f * logf(fb);
    *c0 = fmaxf(0.0f, floorf((float)n_dims *
            logf((float)n_ctx_orig / (bfast * 2.0f * (float)M_PI)) / denom));
    *c1 = fminf((float)n_dims - 1.0f, ceilf((float)n_dims *
            logf((float)n_ctx_orig / (bslow * 2.0f * (float)M_PI)) / denom));
}

static void mla_fill_q8(uint8_t *w, uint64_t rows, uint32_t cols) {
    const uint64_t row_bytes = mla_q8_row_bytes(cols);
    for (uint64_t r = 0; r < rows; r++) {
        uint8_t *row = w + r * row_bytes;
        for (uint32_t b = 0; b < (cols + 31u) / 32u; b++) {
            uint16_t d = f32_to_f16_bits(fabsf(gqa_test_randf()) * 0.05f + 0.001f);
            memcpy(row + (uint64_t)b * 34u, &d, 2);
            int8_t *qs = (int8_t *)(row + (uint64_t)b * 34u + 2u);
            for (int i = 0; i < 32; i++)
                qs[i] = (int8_t)((int)test_randbyte() - 128);
        }
    }
}

static int mla_selftest_one(float freq_scale, float ext_factor,
                            float beta_fast, float beta_slow,
                            const char *name) {
    const uint32_t n_head = 4, kv_lora = 64, qk_nope = 32, qk_rope = 8;
    const uint32_t qk_dim = qk_nope + qk_rope, value_dim = 16;
    const uint32_t kv_raw_dim = kv_lora + qk_rope;
    const uint32_t cache_cap = 16, n_prefill = 3;
    const uint32_t n_ctx_orig = 64;
    const float freq_base = 10000.0f, attn_factor = 1.0f, eps = 1e-5f;
    const float scale = 1.0f / sqrtf((float)qk_dim);

    const uint64_t kb_row = mla_q8_row_bytes(qk_nope);
    const uint64_t vb_row = mla_q8_row_bytes(kv_lora);
    const uint64_t kb_bytes = (uint64_t)n_head * kv_lora * kb_row;
    const uint64_t vb_bytes = (uint64_t)n_head * value_dim * vb_row;
    const uint32_t n_tok = n_prefill + 1; /* prefill batch + one decode */

    float *q = (float *)malloc((uint64_t)n_tok * n_head * qk_dim * 4);
    float *kv_raw = (float *)malloc((uint64_t)n_tok * kv_raw_dim * 4);
    float *w_norm = (float *)malloc(kv_lora * 4);
    uint8_t *k_b = (uint8_t *)malloc(kb_bytes);
    uint8_t *v_b = (uint8_t *)malloc(vb_bytes);
    for (uint64_t i = 0; i < (uint64_t)n_tok * n_head * qk_dim; i++)
        q[i] = gqa_test_randf();
    for (uint64_t i = 0; i < (uint64_t)n_tok * kv_raw_dim; i++)
        kv_raw[i] = gqa_test_randf();
    for (uint32_t i = 0; i < kv_lora; i++) w_norm[i] = gqa_test_randf();
    mla_fill_q8(k_b, (uint64_t)n_head * kv_lora, qk_nope);
    mla_fill_q8(v_b, (uint64_t)n_head * value_dim, kv_lora);

    /* ---- host reference over all n_tok positions ---- */
    float *h_kv_lora = (float *)calloc((uint64_t)cache_cap * kv_lora, 4);
    float *h_k_rope = (float *)calloc((uint64_t)cache_cap * qk_rope, 4);
    float *h_q_roped = (float *)malloc((uint64_t)n_tok * n_head * qk_dim * 4);
    float *h_heads = (float *)malloc((uint64_t)n_tok * n_head * value_dim * 4);
    memcpy(h_q_roped, q, (uint64_t)n_tok * n_head * qk_dim * 4);
    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f)
        mla_host_corr(qk_rope, n_ctx_orig, freq_base, beta_fast, beta_slow,
                      &corr0, &corr1);
    for (uint32_t t = 0; t < n_tok; t++) {
        const float *raw = kv_raw + (uint64_t)t * kv_raw_dim;
        double sum = 0.0;
        for (uint32_t i = 0; i < kv_lora; i++) sum += (double)raw[i] * raw[i];
        const float inv = 1.0f / sqrtf((float)(sum / kv_lora) + eps);
        for (uint32_t i = 0; i < kv_lora; i++)
            h_kv_lora[(uint64_t)t * kv_lora + i] = raw[i] * inv * w_norm[i];
        for (uint32_t i = 0; i < qk_rope; i++)
            h_k_rope[(uint64_t)t * qk_rope + i] = raw[kv_lora + i];
        for (uint32_t h = 0; h < n_head; h++) {
            float *row = h_q_roped + ((uint64_t)t * n_head + h) * qk_dim + qk_nope;
            for (uint32_t i = 0; i < qk_rope; i += 2) {
                const float theta = (float)t *
                    powf(freq_base, -((float)i) / (float)qk_rope);
                float c, s;
                mla_host_yarn(theta, freq_scale, corr0, corr1, (int)i,
                              ext_factor, attn_factor, &c, &s);
                const float x0 = row[i], x1 = row[i + 1];
                row[i] = x0 * c - x1 * s;
                row[i + 1] = x0 * s + x1 * c;
            }
        }
        for (uint32_t h = 0; h < n_head; h++) {
            const float *qh = h_q_roped + ((uint64_t)t * n_head + h) * qk_dim;
            float low[64];
            for (uint32_t j = 0; j < kv_lora; j++)
                low[j] = mla_host_q8_dot(
                        k_b + ((uint64_t)h * kv_lora + j) * kb_row, qh, qk_nope);
            float sc[16];
            float maxs = -INFINITY;
            for (uint32_t r = 0; r <= t; r++) {
                float dotv = 0.0f;
                for (uint32_t j = 0; j < kv_lora; j++)
                    dotv += low[j] * h_kv_lora[(uint64_t)r * kv_lora + j];
                for (uint32_t i = 0; i < qk_rope; i += 2) {
                    const float theta = (float)r *
                        powf(freq_base, -((float)i) / (float)qk_rope);
                    float c, s;
                    mla_host_yarn(theta, freq_scale, corr0, corr1, (int)i,
                                  ext_factor, attn_factor, &c, &s);
                    const float x0 = h_k_rope[(uint64_t)r * qk_rope + i];
                    const float x1 = h_k_rope[(uint64_t)r * qk_rope + i + 1];
                    dotv += qh[qk_nope + i] * (x0 * c - x1 * s) +
                            qh[qk_nope + i + 1] * (x0 * s + x1 * c);
                }
                sc[r] = dotv * scale;
                maxs = fmaxf(maxs, sc[r]);
            }
            float denom = 0.0f;
            for (uint32_t r = 0; r <= t; r++) {
                sc[r] = expf(sc[r] - maxs);
                denom += sc[r];
            }
            denom = fmaxf(denom, 1.0e-20f);
            float lora_sum[64];
            for (uint32_t j = 0; j < kv_lora; j++) {
                float acc = 0.0f;
                for (uint32_t r = 0; r <= t; r++)
                    acc += sc[r] * h_kv_lora[(uint64_t)r * kv_lora + j];
                lora_sum[j] = acc / denom;
            }
            float *out = h_heads + ((uint64_t)t * n_head + h) * value_dim;
            for (uint32_t d = 0; d < value_dim; d++)
                out[d] = mla_host_q8_dot(
                        v_b + ((uint64_t)h * value_dim + d) * vb_row,
                        lora_sum, kv_lora);
        }
    }

    /* ---- GPU: prefill batch of n_prefill, then one decode token ---- */
    void *q_d = NULL, *kvr_d = NULL, *wn_d = NULL, *kb_d = NULL, *vb_d = NULL;
    void *kvn_d = NULL, *lora_d = NULL, *rope_d = NULL, *sel_d = NULL;
    void *low_d = NULL, *heads_d = NULL;
    float *gpu_heads = (float *)malloc((uint64_t)n_tok * n_head * value_dim * 4);
    int ok = cuda_ok(cudaMalloc(&q_d, (uint64_t)n_tok * n_head * qk_dim * 4), "q") &&
             cuda_ok(cudaMalloc(&kvr_d, (uint64_t)n_tok * kv_raw_dim * 4), "kvr") &&
             cuda_ok(cudaMalloc(&wn_d, kv_lora * 4), "wn") &&
             cuda_ok(cudaMalloc(&kb_d, kb_bytes), "kb") &&
             cuda_ok(cudaMalloc(&vb_d, vb_bytes), "vb") &&
             cuda_ok(cudaMalloc(&kvn_d, (uint64_t)n_tok * kv_lora * 4), "kvn") &&
             cuda_ok(cudaMalloc(&lora_d, (uint64_t)cache_cap * kv_lora * 4), "lora") &&
             cuda_ok(cudaMalloc(&rope_d, (uint64_t)cache_cap * qk_rope * 4), "rope") &&
             cuda_ok(cudaMalloc(&sel_d, (uint64_t)n_tok * cache_cap * 4), "sel") &&
             cuda_ok(cudaMalloc(&low_d, (uint64_t)n_tok * n_head * kv_lora * 4), "low") &&
             cuda_ok(cudaMalloc(&heads_d, (uint64_t)n_tok * n_head * value_dim * 4), "heads") &&
             cuda_ok(cudaMemcpy(q_d, q, (uint64_t)n_tok * n_head * qk_dim * 4, cudaMemcpyHostToDevice), "q h2d") &&
             cuda_ok(cudaMemcpy(kvr_d, kv_raw, (uint64_t)n_tok * kv_raw_dim * 4, cudaMemcpyHostToDevice), "kvr h2d") &&
             cuda_ok(cudaMemcpy(wn_d, w_norm, kv_lora * 4, cudaMemcpyHostToDevice), "wn h2d") &&
             cuda_ok(cudaMemcpy(kb_d, k_b, kb_bytes, cudaMemcpyHostToDevice), "kb h2d") &&
             cuda_ok(cudaMemcpy(vb_d, v_b, vb_bytes, cudaMemcpyHostToDevice), "vb h2d");

    const struct { uint32_t pos0, n; } phase[2] = {
        {0, n_prefill}, {n_prefill, 1},
    };
    for (int p = 0; ok && p < 2; p++) {
        const uint32_t pos0 = phase[p].pos0, n = phase[p].n;
        const uint32_t n_selected = pos0 + n;
        float *qp = (float *)q_d + (uint64_t)pos0 * n_head * qk_dim;
        float *kvp = (float *)kvr_d + (uint64_t)pos0 * kv_raw_dim;
        float *kvnp = (float *)kvn_d + (uint64_t)pos0 * kv_lora;
        float *lowp = (float *)low_d + (uint64_t)pos0 * n_head * kv_lora;
        float *headsp = (float *)heads_d + (uint64_t)pos0 * n_head * value_dim;
        ok = pulsar_mla_rope_tail(qp, n, n_head, qk_dim, qk_rope, pos0,
                                  n_ctx_orig, freq_base, freq_scale,
                                  ext_factor, attn_factor, beta_fast, beta_slow) &&
             pulsar_mla_kv_lora_rms_norm(kvnp, kvp, wn_d, n, kv_raw_dim,
                                         kv_lora, eps) &&
             pulsar_mla_store_compact_kv(lora_d, rope_d, kvnp, kvp, pos0, n,
                                         cache_cap, kv_raw_dim, kv_lora, qk_rope) &&
             pulsar_mla_fill_selected_range(sel_d, n, pos0, n_selected,
                                            cache_cap) &&
             pulsar_mla_qk_lowrank(lowp, qp, kb_d, n, n_head, kv_lora,
                                   qk_nope, qk_dim) &&
             pulsar_mla_attention(headsp, qp, lowp, lora_d, rope_d, vb_d,
                                  sel_d, n, n_selected, cache_cap, n_head,
                                  kv_lora, qk_nope, qk_rope, value_dim,
                                  n_ctx_orig, freq_base, freq_scale,
                                  ext_factor, attn_factor, beta_fast,
                                  beta_slow);
    }
    ok = ok && cuda_ok(cudaDeviceSynchronize(), "sync") &&
         cuda_ok(cudaMemcpy(gpu_heads, heads_d,
                            (uint64_t)n_tok * n_head * value_dim * 4,
                            cudaMemcpyDeviceToHost), "heads d2h");

    float maxd = 0.0f, maxref = 0.0f;
    if (ok) {
        for (uint64_t i = 0; i < (uint64_t)n_tok * n_head * value_dim; i++) {
            maxd = fmaxf(maxd, fabsf(gpu_heads[i] - h_heads[i]));
            maxref = fmaxf(maxref, fabsf(h_heads[i]));
        }
        ok = maxd <= 2e-3f * fmaxf(maxref, 1.0f);
    }
    fprintf(stderr, "mla-selftest %s: %s (max diff %.2e, max |ref| %.2e)\n",
            name, ok ? "PASS" : "FAIL", (double)maxd, (double)maxref);
    if (q_d) cudaFree(q_d);
    if (kvr_d) cudaFree(kvr_d);
    if (wn_d) cudaFree(wn_d);
    if (kb_d) cudaFree(kb_d);
    if (vb_d) cudaFree(vb_d);
    if (kvn_d) cudaFree(kvn_d);
    if (lora_d) cudaFree(lora_d);
    if (rope_d) cudaFree(rope_d);
    if (sel_d) cudaFree(sel_d);
    if (low_d) cudaFree(low_d);
    if (heads_d) cudaFree(heads_d);
    free(q); free(kv_raw); free(w_norm); free(k_b); free(v_b);
    free(h_kv_lora); free(h_k_rope); free(h_q_roped); free(h_heads);
    free(gpu_heads);
    return ok;
}

extern "C" int pulsar_mla_selftest(void) {
    /* plain rope (GLM-5.2's live config) and a yarn config to exercise
     * the correction path */
    return mla_selftest_one(1.0f, 0.0f, 0.0f, 0.0f, "plain") &&
           mla_selftest_one(0.5f, 1.0f, 32.0f, 1.0f, "yarn");
}
