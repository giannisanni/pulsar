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

/* gelu tanh approximation (ggml GELU / gelu_pytorch_tanh) */
__device__ __forceinline__ static float pulsar_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456080286535588f *
                                    (x + 0.044715f * x * x * x)));
}

/* act_op for the gated-FFN kernels: 0 = silu (swiglu), 1 = gelu tanh */
__device__ __forceinline__ static float pulsar_gate_act(float g, uint32_t op) {
    return op ? pulsar_gelu(g) : g / (1.0f + expf(-g));
}

/* Gated-FFN combine: act(gate) * up by act_op.
 *   0 = silu (swiglu), 1 = gelu tanh, 2 = swiglu_oai (gpt-oss / MiniMax
 *   M3): gate one-side clamped to 7, up clamped to +-7, alpha-sharpened
 *   sigmoid (1.702), and the +1 on up - llama.cpp ggml_swiglu_oai,
 *   3 = deepseek4 clamped silu: gate one-side clamped to 10, up clamped
 *   to +-10 (DS4_SWIGLU_CLAMP_EXP, both Flash and Pro), plain silu. */
__device__ __forceinline__ static float pulsar_glu(float g, float u, uint32_t op) {
    if (op == 2u) {
        g = fminf(g, 7.0f);
        u = fminf(fmaxf(u, -7.0f), 7.0f);
        return g / (1.0f + expf(-1.702f * g)) * (u + 1.0f);
    }
    if (op == 3u) {
        g = fminf(g, 10.0f);
        u = fminf(fmaxf(u, -10.0f), 10.0f);
        return g / (1.0f + expf(-g)) * u;
    }
    return pulsar_gate_act(g, op) * u;
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

/* tiled prefill GEMM: the per-(row,token) kernel re-reads each weight row
 * from global once per token; here a warp stages k-slabs of its own row in
 * shared memory once and dots them against a 16-token tile, cutting weight
 * traffic 16x. Per-warp slab, so no cross-warp sync. Tile layout is the
 * substrate for the WMMA int8 variant (same slabs, mma.sync consumers). */
#define PULSAR_Q8_TILE_TOK 16u
#define PULSAR_Q8_SLAB_BLOCKS 32u /* 32 q8_0 blocks = 1088 B per warp */

__global__ static void matmul_q8_0_preq_tiled_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t row = (uint64_t)blockIdx.x * 8u + warp;
    const uint64_t t0 = (uint64_t)blockIdx.y * PULSAR_Q8_TILE_TOK;
    if (row >= out_dim || t0 >= n_tok) return;
    const uint32_t tn = n_tok - t0 < PULSAR_Q8_TILE_TOK
            ? (uint32_t)(n_tok - t0) : PULSAR_Q8_TILE_TOK;
    __shared__ unsigned char slab[8][PULSAR_Q8_SLAB_BLOCKS * 34u];
    const unsigned char *wr = w + row * blocks * 34u;
    float acc[PULSAR_Q8_TILE_TOK];
#pragma unroll
    for (uint32_t t = 0; t < PULSAR_Q8_TILE_TOK; t++) acc[t] = 0.0f;
    for (uint64_t b0 = 0; b0 < blocks; b0 += PULSAR_Q8_SLAB_BLOCKS) {
        const uint32_t bn = blocks - b0 < PULSAR_Q8_SLAB_BLOCKS
                ? (uint32_t)(blocks - b0) : PULSAR_Q8_SLAB_BLOCKS;
        const uint32_t slab_bytes = bn * 34u;
        for (uint32_t i = lane; i < slab_bytes; i += 32u) {
            slab[warp][i] = wr[b0 * 34u + i];
        }
        __syncwarp();
        for (uint32_t t = 0; t < tn; t++) {
            const int8_t *xt = xq + (t0 + t) * blocks * 32u;
            const float *xs = xscale + (t0 + t) * blocks;
            float a = 0.0f;
            for (uint32_t b = lane; b < bn; b += 32u) {
                const unsigned char *blk = slab[warp] + b * 34u;
                int dot = dot_i8x32_dp4a((const int8_t *)(blk + 2),
                                         xt + (b0 + b) * 32u);
                a += __half2float(*(const __half *)blk) * xs[b0 + b] * (float)dot;
            }
            acc[t] += a;
        }
        __syncwarp();
    }
    for (uint32_t t = 0; t < tn; t++) {
        float v = warp_sum_f32(acc[t]);
        if (lane == 0) out[(t0 + t) * out_dim + row] = v;
    }
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

/* tensor-core prefill GEMM (sm_80+): one mma.m16n8k32 s8s8s32 covers
 * exactly one q8_0 quant block (k=32), so the integer accumulator can be
 * rescaled by scale_w[row][blk] * scale_x[tok][blk] in registers using
 * the DOCUMENTED PTX fragment mapping (the opaque WMMA API can't do
 * per-block rescale). Warp tile: 16 rows x 8 tokens; block: 8 warps in a
 * 4x2 grid = 64 rows x 16 tokens. No smem: A/B reuse across the warp
 * grid rides L1/L2 (v1; smem staging is the next lever). */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ static void mma_s8_16x8x32(
        int32_t d[4], const int32_t a[4], const int32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(0), "r"(0), "r"(0), "r"(0));
}
#endif

__device__ __forceinline__ static float f16_bytes_to_f32(const unsigned char *p) {
    unsigned short h = (unsigned short)(p[0] | (p[1] << 8));
    return __half2float(__ushort_as_half(h));
}

__global__ static void matmul_q8_0_preq_mma_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint32_t out_dim,
        uint32_t n_tok,
        uint32_t blocks) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t warp = threadIdx.x >> 5u;   /* 0..7 */
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t g = lane >> 2u;             /* row group / B col 0..7 */
    const uint32_t tg = lane & 3u;             /* thread-in-group 0..3 */
    /* 4 row-warps x 2 token-warps */
    const uint32_t r0 = blockIdx.x * 64u + (warp >> 1u) * 16u;
    const uint32_t t0 = blockIdx.y * 16u + (warp & 1u) * 8u;
    if (r0 >= out_dim || t0 >= n_tok) return;

    const uint32_t row_a = r0 + g;       /* a0/a2 row */
    const uint32_t row_b = r0 + g + 8u;  /* a1/a3 row */
    const uint32_t tok_b0 = t0 + g;      /* B load column */
    const uint32_t col0 = t0 + tg * 2u;  /* C columns */
    const uint32_t col1 = col0 + 1u;
    const bool ra_ok = row_a < out_dim;
    const bool rb_ok = row_b < out_dim;
    const bool tb_ok = tok_b0 < n_tok;

    const unsigned char *wr_a = w + (uint64_t)row_a * blocks * 34u;
    const unsigned char *wr_b = w + (uint64_t)row_b * blocks * 34u;
    const int8_t *xr = xq + (uint64_t)tok_b0 * blocks * 32u;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint32_t blk = 0; blk < blocks; blk++) {
        int32_t a[4], b[2], d[4];
        const unsigned char *ba = wr_a + blk * 34u;
        const unsigned char *bb = wr_b + blk * 34u;
        a[0] = ra_ok ? load_i8x4_i32_unaligned((const int8_t *)(ba + 2u + tg * 4u)) : 0;
        a[1] = rb_ok ? load_i8x4_i32_unaligned((const int8_t *)(bb + 2u + tg * 4u)) : 0;
        a[2] = ra_ok ? load_i8x4_i32_unaligned((const int8_t *)(ba + 18u + tg * 4u)) : 0;
        a[3] = rb_ok ? load_i8x4_i32_unaligned((const int8_t *)(bb + 18u + tg * 4u)) : 0;
        b[0] = tb_ok ? *(const int32_t *)(xr + blk * 32u + tg * 4u) : 0;
        b[1] = tb_ok ? *(const int32_t *)(xr + blk * 32u + 16u + tg * 4u) : 0;
        mma_s8_16x8x32(d, a, b);
        const float wsa = ra_ok ? f16_bytes_to_f32(ba) : 0.0f;
        const float wsb = rb_ok ? f16_bytes_to_f32(bb) : 0.0f;
        const float xs0 = col0 < n_tok ? xscale[(uint64_t)col0 * blocks + blk] : 0.0f;
        const float xs1 = col1 < n_tok ? xscale[(uint64_t)col1 * blocks + blk] : 0.0f;
        acc0 += (float)d[0] * wsa * xs0;
        acc1 += (float)d[1] * wsa * xs1;
        acc2 += (float)d[2] * wsb * xs0;
        acc3 += (float)d[3] * wsb * xs1;
    }
    if (ra_ok && col0 < n_tok) out[(uint64_t)col0 * out_dim + row_a] = acc0;
    if (ra_ok && col1 < n_tok) out[(uint64_t)col1 * out_dim + row_a] = acc1;
    if (rb_ok && col0 < n_tok) out[(uint64_t)col0 * out_dim + row_b] = acc2;
    if (rb_ok && col1 < n_tok) out[(uint64_t)col1 * out_dim + row_b] = acc3;
#else
    (void)out; (void)w; (void)xq; (void)xscale;
    (void)out_dim; (void)n_tok; (void)blocks;
#endif
}

static int pulsar_device_cc_major(void) {
    static int cached[16] = {-1, -1, -1, -1, -1, -1, -1, -1,
                             -1, -1, -1, -1, -1, -1, -1, -1};
    int dev = 0;
    (void)cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) return 0;
    if (cached[dev] < 0) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
            major = 0;
        }
        cached[dev] = major;
    }
    return cached[dev];
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
    } else if (n_tok >= 16 && pulsar_device_cc_major() >= 8 &&
               !getenv("PULSAR_NO_MMA")) {
        dim3 grid((out_dim + 63u) / 64u, (n_tok + 15u) / 16u, 1);
        matmul_q8_0_preq_mma_kernel<<<grid, 256>>>(
                (float *)out_dev, (const unsigned char *)w_dev, xq, xscale,
                out_dim, n_tok, (uint32_t)blocks);
    } else if (n_tok >= 8) {
        dim3 grid((unsigned)((out_dim + 7u) / 8u),
                  (unsigned)((n_tok + PULSAR_Q8_TILE_TOK - 1u) / PULSAR_Q8_TILE_TOK), 1);
        matmul_q8_0_preq_tiled_kernel<<<grid, 256>>>(
                (float *)out_dev, (const unsigned char *)w_dev, xq, xscale,
                out_dim, n_tok, blocks);
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

static int q8_0_matmul_selftest_one(uint32_t n_tok) {
    /* n_tok 9 = tiled path incl. partial tile; 19/40 = tensor-core mma on
     * sm_80+ (partial + multi 16-token tiles), tiled elsewhere; in_dim
     * 4256 -> 133 blocks, a partial 5-block weight slab */
    const uint32_t in_dim = 4256, out_dim = 512;
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
    fprintf(stderr, "q8_0-matmul-selftest n_tok=%u: %s (max abs diff %.2e, max |ref| %.2e)\n",
            n_tok, ok ? "PASS" : "FAIL", (double)maxd, (double)maxref);
    if (w_dev) cudaFree(w_dev);
    if (x_dev) cudaFree(x_dev);
    if (out_dev) cudaFree(out_dev);
    free(w); free(wf); free(x); free(ref); free(gpu);
    return ok;
}

extern "C" int pulsar_q8_0_matmul_selftest(void) {
    return q8_0_matmul_selftest_one(9) &&
           q8_0_matmul_selftest_one(19) &&
           q8_0_matmul_selftest_one(40);
}

/* ---- sigmoid router + top-k select ------------------------------------
 * Warp-per-token select, derived from ds4's glm_router_select_kernel (the
 * Hy3 router mirrors GLM: probs = sigmoid(logits), selection score =
 * prob + bias, route weights = selected probs normalized * scale).
 * pulsar contract: bias is an explicit device pointer, not a model-map
 * offset. n_expert <= 512 (templated register tiling), k_used <= n_expert. */

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

/* softmax_mode 1 (qwen3moe): softmax(logits) then renormalize over the
 * selected k is algebraically softmax over just the selected logits, and
 * softmax is monotonic so top-k by prob == top-k by logit. So: select on
 * raw logits, then exp-normalize the k winners.
 *
 * softmax_mode 2 (inkling sink): the gate matmul emits n_expert routed
 * logits + n_shexp shared-expert logits per token. Selection = mode-0
 * (sigmoid + bias) over the routed rows only; weights = softmax of
 * logsigmoid over [k winners' raw logits ++ shared logits] * scale, and
 * the shared experts append as always-selected slots k_used..k_used+n_shexp
 * with ids n_expert+s (their gammas ride the same weights array). */
template <uint32_t J>
__global__ static void router_select_kernel(
        int32_t *selected,         /* [n_tok][k_used(+n_shexp)] */
        float *weights,            /* [n_tok][k_used(+n_shexp)] */
        const float *logits,       /* [n_tok][n_expert(+n_shexp)] */
        const float *bias,         /* [n_expert] */
        uint32_t n_expert,
        uint32_t k_used,
        float weight_scale,
        uint32_t n_tok,
        uint32_t softmax_mode,
        uint32_t n_shexp) {        /* mode 2 only; 0 otherwise */
    const uint32_t lane = threadIdx.x;
    const uint32_t token = blockIdx.x * blockDim.y + threadIdx.y;
    if (token >= n_tok || lane >= 32u) return;

    const uint32_t sink = softmax_mode == 2u ? n_shexp : 0u;
    const uint32_t n_logit = n_expert + sink;
    const uint32_t n_slot = k_used + sink;
    const float *log = logits + (uint64_t)token * n_logit;
    int32_t *sel = selected + (uint64_t)token * n_slot;
    float *w = weights + (uint64_t)token * n_slot;

    float local_prob[J];
    float local_score[J];
    #pragma unroll
    for (uint32_t j = 0; j < J; j++) {
        const uint32_t e = lane + j * 32u;
        if (e < n_expert) {
            const float raw = log[e];
            const float p = softmax_mode == 1u ? raw : router_sigmoid(raw);
            /* mode 2 weights use the raw winner logit, not its sigmoid */
            local_prob[j] = softmax_mode == 2u ? raw : p;
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
        for (uint32_t j = 0; j < J; j++) {
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
        for (uint32_t j = 0; j < J; j++) {
            if (lane + j * 32u == best_idx) local_score[j] = -INFINITY;
        }
        if (lane == 0) {
            sel[k] = (int32_t)best_idx;
            w[k] = best_prob;
        }
        sum += best_prob;
    }

    if (lane == 0) {
        if (softmax_mode == 2u) {
            /* append the always-on shared experts, then weight everyone by
             * softmax(logsigmoid(raw logit)) * scale (logsigmoid(x) =
             * -softplus(-x); softmax input <= 0, plain exp-sum is safe) */
            for (uint32_t s = 0; s < sink; s++) {
                sel[k_used + s] = (int32_t)(n_expert + s);
                w[k_used + s] = log[n_expert + s];
            }
            float m = -INFINITY;
            for (uint32_t k = 0; k < n_slot; k++) {
                const float x = w[k];
                w[k] = x < 0.0f ? x - log1pf(expf(x)) : -log1pf(expf(-x));
                m = fmaxf(m, w[k]);
            }
            float es = 0.0f;
            for (uint32_t k = 0; k < n_slot; k++) {
                w[k] = expf(w[k] - m);
                es += w[k];
            }
            for (uint32_t k = 0; k < n_slot; k++) w[k] = w[k] / es * weight_scale;
        } else if (softmax_mode == 1u) {
            float m = -INFINITY;
            for (uint32_t k = 0; k < k_used; k++) m = fmaxf(m, w[k]);
            float es = 0.0f;
            for (uint32_t k = 0; k < k_used; k++) {
                w[k] = expf(w[k] - m);
                es += w[k];
            }
            for (uint32_t k = 0; k < k_used; k++) w[k] = w[k] / es * weight_scale;
        } else {
            sum = fmaxf(sum, 6.103515625e-5f);
            for (uint32_t k = 0; k < k_used; k++) w[k] = w[k] / sum * weight_scale;
        }
    }
}

extern "C" int pulsar_router_select(
        void *selected_dev,        /* int32 [n_tok][k_used(+n_shexp)] */
        void *weights_dev,         /* f32   [n_tok][k_used(+n_shexp)] */
        const void *logits_dev,    /* f32   [n_tok][n_expert(+n_shexp)] */
        const void *bias_dev,      /* f32   [n_expert] */
        uint32_t n_expert,
        uint32_t k_used,
        float weight_scale,
        uint32_t n_tok,
        uint32_t softmax_mode,
        uint32_t n_shexp) {
    if (n_expert == 0 || k_used == 0 || k_used > n_expert || n_tok == 0 ||
        (softmax_mode != 2u && n_shexp != 0) ||
        n_expert + (softmax_mode == 2u ? n_shexp : 0u) > 512u) {
        return 0;
    }
    dim3 block(32, 4, 1);
    if (n_expert > 256u || (softmax_mode == 2u && n_expert + n_shexp > 256u)) {
        router_select_kernel<16><<<(n_tok + 3u) / 4u, block>>>(
                (int32_t *)selected_dev, (float *)weights_dev,
                (const float *)logits_dev, (const float *)bias_dev,
                n_expert, k_used, weight_scale, n_tok, softmax_mode, n_shexp);
        return cuda_ok(cudaGetLastError(), "router select launch");
    }
    router_select_kernel<8><<<(n_tok + 3u) / 4u, block>>>(
            (int32_t *)selected_dev, (float *)weights_dev,
            (const float *)logits_dev, (const float *)bias_dev,
            n_expert, k_used, weight_scale, n_tok, softmax_mode, n_shexp);
    return cuda_ok(cudaGetLastError(), "router select launch");
}

/* CPU-reference selftest across Hy3-like and GLM-like shapes. The softmax
 * reference is the llama.cpp order (full softmax over ALL experts, top-k,
 * renormalize the selected) - deliberately NOT the kernel's select-on-
 * logits algebra, so this also proves the equivalence the kernel relies on. */
static int router_selftest_one(uint32_t n_expert, uint32_t k_used,
                               float scale, uint32_t n_tok, uint32_t softmax) {
    float *logits = (float *)malloc((uint64_t)n_tok * n_expert * sizeof(float));
    float *bias = (float *)malloc((uint64_t)n_expert * sizeof(float));
    int32_t *sel_ref = (int32_t *)malloc((uint64_t)n_tok * k_used * sizeof(int32_t));
    float *w_ref = (float *)malloc((uint64_t)n_tok * k_used * sizeof(float));
    int32_t *sel_gpu = (int32_t *)malloc((uint64_t)n_tok * k_used * sizeof(int32_t));
    float *w_gpu = (float *)malloc((uint64_t)n_tok * k_used * sizeof(float));

    for (uint64_t i = 0; i < (uint64_t)n_tok * n_expert; i++)
        logits[i] = gqa_test_randf() * 4.0f;
    for (uint32_t e = 0; e < n_expert; e++)
        bias[e] = softmax ? 0.0f : gqa_test_randf();

    for (uint32_t t = 0; t < n_tok; t++) {
        const float *log = logits + (uint64_t)t * n_expert;
        float prob[512], score[512];
        float lmax = -INFINITY, lsum = 0.0f;
        if (softmax) {
            for (uint32_t e = 0; e < n_expert; e++) lmax = fmaxf(lmax, log[e]);
            for (uint32_t e = 0; e < n_expert; e++) lsum += expf(log[e] - lmax);
        }
        for (uint32_t e = 0; e < n_expert; e++) {
            prob[e] = softmax ? expf(log[e] - lmax) / lsum
                              : 1.0f / (1.0f + expf(-log[e]));
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
                                  n_expert, k_used, scale, n_tok, softmax, 0) &&
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
            "router-selftest n_expert=%u k=%u%s: %s (idx mismatches %u, max w diff %.2e)\n",
            n_expert, k_used, softmax ? " softmax" : "",
            ok ? "PASS" : "FAIL", idx_mismatch, (double)maxd);
    if (log_dev) cudaFree(log_dev);
    if (bias_dev) cudaFree(bias_dev);
    if (sel_dev) cudaFree(sel_dev);
    if (w_dev) cudaFree(w_dev);
    free(logits); free(bias); free(sel_ref); free(w_ref); free(sel_gpu); free(w_gpu);
    return ok;
}

/* mode 2 (inkling sink): CPU reference mirrors the llama.cpp graph -
 * select top-k by sigmoid+bias over routed rows, weights = softmax of
 * logsigmoid over [k winners' raw logits ++ shared logits] * scale. */
static int router_sink_selftest_one(uint32_t n_expert, uint32_t k_used,
                                    uint32_t n_shexp, float scale,
                                    uint32_t n_tok) {
    const uint32_t n_logit = n_expert + n_shexp;
    const uint32_t n_slot = k_used + n_shexp;
    float *logits = (float *)malloc((uint64_t)n_tok * n_logit * sizeof(float));
    float *bias = (float *)malloc((uint64_t)n_expert * sizeof(float));
    int32_t *sel_ref = (int32_t *)malloc((uint64_t)n_tok * n_slot * sizeof(int32_t));
    float *w_ref = (float *)malloc((uint64_t)n_tok * n_slot * sizeof(float));
    int32_t *sel_gpu = (int32_t *)malloc((uint64_t)n_tok * n_slot * sizeof(int32_t));
    float *w_gpu = (float *)malloc((uint64_t)n_tok * n_slot * sizeof(float));

    for (uint64_t i = 0; i < (uint64_t)n_tok * n_logit; i++)
        logits[i] = gqa_test_randf() * 4.0f;
    for (uint32_t e = 0; e < n_expert; e++) bias[e] = gqa_test_randf();

    for (uint32_t t = 0; t < n_tok; t++) {
        const float *log = logits + (uint64_t)t * n_logit;
        float score[512];
        for (uint32_t e = 0; e < n_expert; e++)
            score[e] = 1.0f / (1.0f + expf(-log[e])) + bias[e];
        float raw[64];
        for (uint32_t k = 0; k < k_used; k++) {
            uint32_t best = UINT32_MAX;
            for (uint32_t e = 0; e < n_expert; e++) {
                if (best == UINT32_MAX || score[e] > score[best]) best = e;
            }
            sel_ref[(uint64_t)t * n_slot + k] = (int32_t)best;
            raw[k] = log[best];
            score[best] = -INFINITY;
        }
        for (uint32_t s = 0; s < n_shexp; s++) {
            sel_ref[(uint64_t)t * n_slot + k_used + s] = (int32_t)(n_expert + s);
            raw[k_used + s] = log[n_expert + s];
        }
        float m = -INFINITY, es = 0.0f;
        for (uint32_t k = 0; k < n_slot; k++) {
            /* logsigmoid via double log1p for a tight reference */
            raw[k] = (float)(-log1p(exp(-(double)raw[k])));
            m = fmaxf(m, raw[k]);
        }
        for (uint32_t k = 0; k < n_slot; k++) es += expf(raw[k] - m);
        for (uint32_t k = 0; k < n_slot; k++)
            w_ref[(uint64_t)t * n_slot + k] = expf(raw[k] - m) / es * scale;
    }

    void *log_dev = NULL, *bias_dev = NULL, *sel_dev = NULL, *w_dev = NULL;
    const uint64_t log_bytes = (uint64_t)n_tok * n_logit * sizeof(float);
    const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
    const uint64_t sel_bytes = (uint64_t)n_tok * n_slot * sizeof(int32_t);
    const uint64_t w_bytes = (uint64_t)n_tok * n_slot * sizeof(float);
    int ok = cuda_ok(cudaMalloc(&log_dev, log_bytes), "logits alloc") &&
             cuda_ok(cudaMalloc(&bias_dev, bias_bytes), "bias alloc") &&
             cuda_ok(cudaMalloc(&sel_dev, sel_bytes), "sel alloc") &&
             cuda_ok(cudaMalloc(&w_dev, w_bytes), "w alloc") &&
             cuda_ok(cudaMemcpy(log_dev, logits, log_bytes, cudaMemcpyHostToDevice), "logits h2d") &&
             cuda_ok(cudaMemcpy(bias_dev, bias, bias_bytes, cudaMemcpyHostToDevice), "bias h2d") &&
             pulsar_router_select(sel_dev, w_dev, log_dev, bias_dev,
                                  n_expert, k_used, scale, n_tok, 2, n_shexp) &&
             cuda_ok(cudaDeviceSynchronize(), "sync") &&
             cuda_ok(cudaMemcpy(sel_gpu, sel_dev, sel_bytes, cudaMemcpyDeviceToHost), "sel d2h") &&
             cuda_ok(cudaMemcpy(w_gpu, w_dev, w_bytes, cudaMemcpyDeviceToHost), "w d2h");
    float maxd = 0.0f;
    uint32_t idx_mismatch = 0;
    if (ok) {
        for (uint64_t i = 0; i < (uint64_t)n_tok * n_slot; i++) {
            if (sel_gpu[i] != sel_ref[i]) idx_mismatch++;
            float d = fabsf(w_gpu[i] - w_ref[i]);
            if (d > maxd) maxd = d;
        }
        ok = idx_mismatch == 0 && maxd <= 1e-5f;
    }
    fprintf(stderr,
            "router-sink-selftest n_expert=%u k=%u shexp=%u: %s (idx mismatches %u, max w diff %.2e)\n",
            n_expert, k_used, n_shexp, ok ? "PASS" : "FAIL", idx_mismatch,
            (double)maxd);
    if (log_dev) cudaFree(log_dev);
    if (bias_dev) cudaFree(bias_dev);
    if (sel_dev) cudaFree(sel_dev);
    if (w_dev) cudaFree(w_dev);
    free(logits); free(bias); free(sel_ref); free(w_ref); free(sel_gpu); free(w_gpu);
    return ok;
}

extern "C" int pulsar_router_selftest(void) {
    /* Hy3-like (64 experts, top-8), GLM-like (256, top-8), odd token
     * count; qwen3moe-like softmax (128, top-8) + wide softmax (384);
     * inkling sink (256 routed + 2 shared, top-6) */
    return router_selftest_one(64, 8, 2.5f, 7, 0) &&
           router_selftest_one(256, 8, 1.0f, 5, 0) &&
           router_selftest_one(96, 6, 1.5f, 1, 0) &&
           router_selftest_one(128, 8, 1.0f, 6, 1) &&
           router_selftest_one(384, 8, 1.0f, 3, 1) &&
           router_sink_selftest_one(256, 6, 2, 8.0f, 5) &&
           router_sink_selftest_one(64, 4, 1, 1.0f, 3);
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

/* K-quants (ggml layouts, verbatim): unlock the AngelSlim official Hy3
 * ggufs whose experts are q4_K/q5_K/q6_K. */
typedef struct {
    uint8_t hmask[PULSAR_QK_K / 8]; /* high bit of each 3-bit quant */
    uint8_t qs[PULSAR_QK_K / 4];    /* low 2 bits */
    uint8_t scales[12];             /* 16x 6-bit scales */
    uint16_t d;
} block_q3_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12]; /* 8x (scale, min), 6 bits each */
    uint8_t qs[PULSAR_QK_K / 2];
} block_q4_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[PULSAR_QK_K / 8];
    uint8_t qs[PULSAR_QK_K / 2];
} block_q5_K;

typedef struct {
    uint8_t ql[PULSAR_QK_K / 2];
    uint8_t qh[PULSAR_QK_K / 4];
    int8_t scales[PULSAR_QK_K / 16];
    uint16_t d;
} block_q6_K;

/* iq4_xs (MiniMax M3 ffn_down_exps): 256-superblock, non-linear 4-bit.
 * 8 sub-blocks x 32, each with a 6-bit scale (4 low bits in scales_l,
 * 2 high bits in scales_h); quants index the fixed kvalues_iq4nl codebook. */
typedef struct {
    uint16_t d;
    uint16_t scales_h;
    uint8_t scales_l[PULSAR_QK_K / 64]; /* 4 bytes: 8x 4-bit low scale */
    uint8_t qs[PULSAR_QK_K / 2];        /* 128 bytes: 256x 4-bit */
} block_iq4_xs;                         /* 136 bytes */

/* ggml's get_scale_min_k4: 8 (scale, min) pairs packed 6-bit in 12 bytes */
__host__ __device__ static inline void k4_scale_min(
        int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = (uint8_t)((q[j + 4] & 0x0fu) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

#include "iq2_tables.inc"
#include "iq_extra_tables.inc"

typedef struct {
    uint16_t d;
    uint16_t qs[PULSAR_QK_K / 8]; /* 9-bit grid index + 7-bit sign index */
    uint8_t scales[PULSAR_QK_K / 32];
} block_iq2_xs;

typedef struct {
    uint16_t d;
    uint8_t qs[3 * PULSAR_QK_K / 8]; /* 256 grid bytes + 8 aux u32 */
} block_iq3_xxs;

typedef struct {
    uint16_t d;
    uint8_t qs[16]; /* 32 x 4-bit, offset -8 */
} block_q4_0;

typedef struct {
    uint16_t d;     /* delta */
    uint16_t m;     /* min */
    uint32_t qh;    /* high bit of each 5-bit quant */
    uint8_t qs[16]; /* low 4 bits */
} block_q5_1;

/* Activations for expert dots are quantized to q8_K (ggml layout: f32
 * scale, 256 int8, 16 block sums), then the dots run integer dp4a -
 * ds4's exact math. Device functions verbatim from ds4_cuda.cu. */

typedef struct {
    float d;
    int8_t qs[PULSAR_QK_K];
    int16_t bsums[PULSAR_QK_K / 16];
} block_q8_K;

__global__ static void q8_K_quantize_kernel(block_q8_K *out, const float *x, uint32_t in_dim, uint32_t n_rows) {
    /* tail block (in_dim not a 256-multiple, e.g. gemma4's 704-wide
     * experts): elements past in_dim quantize as zero, so integer dots
     * against ANY weight bytes there contribute exactly nothing */
    uint32_t b = blockIdx.x;
    uint32_t row = blockIdx.y;
    const uint32_t n_blocks = (in_dim + PULSAR_QK_K - 1u) / PULSAR_QK_K;
    if (row >= n_rows || b >= n_blocks) return;
    const uint32_t bn = min(PULSAR_QK_K, in_dim - b * PULSAR_QK_K);
    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * PULSAR_QK_K;
    block_q8_K *yb = out + (uint64_t)row * n_blocks + b;
    __shared__ float abs_part[256];
    __shared__ float val_part[256];
    __shared__ float maxv_s;
    __shared__ float iscale_s;
    uint32_t tid = threadIdx.x;
    float v = tid < bn ? xr[tid] : 0.0f;
    abs_part[tid] = tid < bn ? fabsf(v) : 0.0f;
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
        int qv = tid < bn ? (int)lrintf(iscale_s * xr[tid]) : 0;
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
    if (in_dim == 0 || n_rows == 0) return 0;
    dim3 grid((in_dim + PULSAR_QK_K - 1u) / PULSAR_QK_K, n_rows, 1);
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

/* K-quant dots vs q8_K activations. Integer accumulation via dp4a, float
 * scaling at the end - same shape as the ggml scalar references, so the
 * host mirrors in the selftests match to float rounding. */

/* q3_K: 16 6-bit scales (packed 12 bytes, value-32 signed), quants =
 * low 2 bits + hmask high bit, centered: q = lo2 - (hbit ? 0 : 4). */
__host__ __device__ static inline void k3_unpack_scales(const uint8_t *scales, int8_t *sc) {
    for (int j = 0; j < 16; j++) {
        uint8_t s;
        if (j < 8) {
            s = (scales[j] & 0x0fu) | (((scales[8 + j % 4] >> (2 * (j / 4))) & 3u) << 4);
        } else {
            s = (scales[j - 8] >> 4) | (((scales[8 + j % 4] >> (2 * (j / 4))) & 3u) << 4);
        }
        sc[j] = (int8_t)(s - 32);
    }
}

__device__ static float dev_dot_q3_K_q8_K_block(const block_q3_K *x, const block_q8_K *y) {
    const float d = f16_to_f32(x->d) * y->d;
    int8_t sc[16];
    k3_unpack_scales(x->scales, sc);
    const uint8_t *q3 = x->qs;
    const uint8_t *hm = x->hmask;
    const int8_t *q8 = y->qs;
    int isum = 0;
    uint32_t hbit = 1u;
    int is = 0;
    for (int k = 0; k < 2; k++) { /* 128 values per chunk */
        int shift = 0;
        for (int j = 0; j < 4; j++) { /* 4 x 32 per chunk */
            for (int half = 0; half < 2; half++) {
                int s16 = 0;
                for (int i = 0; i < 16; i++) {
                    const int l = half * 16 + i;
                    int q = (q3[l] >> shift) & 3;
                    if ((hm[l] & hbit) == 0u) q -= 4;
                    s16 += q * (int)q8[l];
                }
                isum += (int)sc[is++] * s16;
            }
            shift += 2;
            q8 += 32;
            hbit <<= 1u; /* hmask bit index runs 0..7 across BOTH chunks */
        }
        q3 += 32;
    }
    return d * (float)isum;
}
__device__ static float dev_dot_q4_K_q8_K_block(const block_q4_K *x, const block_q8_K *y) {
    const float d = f16_to_f32(x->d) * y->d;
    const float dmin = f16_to_f32(x->dmin) * y->d;
    const uint8_t *q4 = x->qs;
    const int8_t *q8 = y->qs;
    int isum = 0;
    int msum = 0;
    for (int j = 0; j < 4; j++) { /* 64 values per chunk */
        uint8_t sc1, m1, sc2, m2;
        k4_scale_min(2 * j, x->scales, &sc1, &m1);
        k4_scale_min(2 * j + 1, x->scales, &sc2, &m2);
        int s1 = 0, s2 = 0;
        #pragma unroll
        for (int i = 0; i < 32; i += 4) {
            const uint32_t v = *(const uint32_t *)(q4 + i);
            s1 = __dp4a((int)(v & 0x0f0f0f0fu), *(const int32_t *)(q8 + i), s1);
            s2 = __dp4a((int)((v >> 4) & 0x0f0f0f0fu), *(const int32_t *)(q8 + 32 + i), s2);
        }
        isum += (int)sc1 * s1 + (int)sc2 * s2;
        msum += (int)m1 * (y->bsums[4 * j] + y->bsums[4 * j + 1]) +
                (int)m2 * (y->bsums[4 * j + 2] + y->bsums[4 * j + 3]);
        q4 += 32;
        q8 += 64;
    }
    return d * (float)isum - dmin * (float)msum;
}

__device__ static float dev_dot_q5_K_q8_K_block(const block_q5_K *x, const block_q8_K *y) {
    const float d = f16_to_f32(x->d) * y->d;
    const float dmin = f16_to_f32(x->dmin) * y->d;
    const uint8_t *q5 = x->qs;
    const uint8_t *qh = x->qh;
    const int8_t *q8 = y->qs;
    int isum = 0;
    int msum = 0;
    for (int j = 0; j < 4; j++) {
        uint8_t sc1, m1, sc2, m2;
        k4_scale_min(2 * j, x->scales, &sc1, &m1);
        k4_scale_min(2 * j + 1, x->scales, &sc2, &m2);
        int s1 = 0, s2 = 0;
        #pragma unroll
        for (int i = 0; i < 32; i += 4) {
            const uint32_t v = *(const uint32_t *)(q5 + i);
            const uint32_t h = *(const uint32_t *)(qh + i);
            const uint32_t hb1 = ((h >> (2 * j)) & 0x01010101u) << 4;
            const uint32_t hb2 = ((h >> (2 * j + 1)) & 0x01010101u) << 4;
            s1 = __dp4a((int)((v & 0x0f0f0f0fu) | hb1), *(const int32_t *)(q8 + i), s1);
            s2 = __dp4a((int)(((v >> 4) & 0x0f0f0f0fu) | hb2), *(const int32_t *)(q8 + 32 + i), s2);
        }
        isum += (int)sc1 * s1 + (int)sc2 * s2;
        msum += (int)m1 * (y->bsums[4 * j] + y->bsums[4 * j + 1]) +
                (int)m2 * (y->bsums[4 * j + 2] + y->bsums[4 * j + 3]);
        q5 += 32;
        q8 += 64;
    }
    return d * (float)isum - dmin * (float)msum;
}

/* iq4_xs non-linear codebook (ggml kvalues_iq4nl) */
__device__ static const int8_t kd_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

__device__ static float dev_dot_iq4_xs_q8_K_block(const block_iq4_xs *x, const block_q8_K *y) {
    const uint8_t *qs = x->qs;
    const int8_t *q8 = y->qs;
    const uint16_t sh = x->scales_h;
    float sumf = 0.0f;
    int any = 0;
    for (int ib = 0; ib < 8; ib++) {
        const int ls = (int)(((x->scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xf) |
                             (((sh >> (2 * ib)) & 3) << 4)) - 32;
        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            const uint8_t b = qs[j];
            sumi += kd_iq4nl[b & 0xf] * (int)q8[j];
            sumi += kd_iq4nl[b >> 4] * (int)q8[j + 16];
        }
        /* tail guard: a zero-quantized activation block dots to 0; leaving
         * the super-scale x->d untouched keeps overread garbage from NaNs */
        if (sumi != 0) { sumf += (float)(ls * sumi); any = 1; }
        qs += 16;
        q8 += 32;
    }
    return any ? y->d * f16_to_f32(x->d) * sumf : 0.0f;
}

/* byte-assembled u32: block_q6_K is 210 bytes, so blocks after the first
 * sit 2-byte aligned and direct u32 loads fault */
__device__ __forceinline__ static uint32_t load_u32_bytes(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

__device__ static float dev_dot_q6_K_q8_K_block(const block_q6_K *x, const block_q8_K *y) {
    const float d = f16_to_f32(x->d) * y->d;
    const uint8_t *ql = x->ql;
    const uint8_t *qh = x->qh;
    const int8_t *sc = x->scales;
    const int8_t *q8 = y->qs;
    int isum = 0;
    for (int j = 0; j < 2; j++) { /* 128 values per chunk */
        int g[8] = {0, 0, 0, 0, 0, 0, 0, 0}; /* 8 x 16-value scale groups */
        #pragma unroll
        for (int i = 0; i < 32; i += 4) {
            const uint32_t lo0 = load_u32_bytes(ql + i);
            const uint32_t lo1 = load_u32_bytes(ql + 32 + i);
            const uint32_t h = load_u32_bytes(qh + i);
            const int32_t v0 = __vsub4((int)((lo0 & 0x0f0f0f0fu) | (((h >> 0) & 0x03030303u) << 4)), 0x20202020);
            const int32_t v1 = __vsub4((int)((lo1 & 0x0f0f0f0fu) | (((h >> 2) & 0x03030303u) << 4)), 0x20202020);
            const int32_t v2 = __vsub4((int)(((lo0 >> 4) & 0x0f0f0f0fu) | (((h >> 4) & 0x03030303u) << 4)), 0x20202020);
            const int32_t v3 = __vsub4((int)(((lo1 >> 4) & 0x0f0f0f0fu) | (((h >> 6) & 0x03030303u) << 4)), 0x20202020);
            const int sub = i >> 4; /* 16-value half within each 32-group */
            g[0 + sub] = __dp4a(v0, *(const int32_t *)(q8 + i), g[0 + sub]);
            g[2 + sub] = __dp4a(v1, *(const int32_t *)(q8 + 32 + i), g[2 + sub]);
            g[4 + sub] = __dp4a(v2, *(const int32_t *)(q8 + 64 + i), g[4 + sub]);
            g[6 + sub] = __dp4a(v3, *(const int32_t *)(q8 + 96 + i), g[6 + sub]);
        }
        for (int k = 0; k < 8; k++) isum += (int)sc[k] * g[k];
        sc += 8;
        ql += 64;
        qh += 32;
        q8 += 128;
    }
    return d * (float)isum;
}

/* iq2_xs: 32 groups of 8 values; qs[k]&511 -> grid row, qs[k]>>9 ->
 * ksigns (same table as iq2_xxs); 4-bit scales per 32-value pair. */
__device__ static float dev_dot_iq2_xs_q8_K_block(
        const block_iq2_xs *x, const block_q8_K *y,
        const uint64_t *grid, const uint8_t *signs) {
    const float xd = f16_to_f32(x->d);
    const int8_t *q8 = y->qs;
    float sumf = 0.0f;
    for (int g = 0; g < 8; g++) { /* 8 groups of 32 values */
        const int ls1 = 2 * (x->scales[g] & 0x0f) + 1;
        const int ls2 = 2 * (x->scales[g] >> 4) + 1;
        int s1 = 0, s2 = 0;
        for (int j = 0; j < 4; j++) {
            const uint16_t q = x->qs[g * 4 + j];
            int32_t w0, w1;
            const uint64_t gr = grid[q & 511];
            const uint32_t sgn = dev_unpack_iq2_signs(signs[q >> 9]);
            const int32_t sm0 = __vcmpne4(sgn & 0x08040201u, 0);
            const int32_t sm1 = __vcmpne4(sgn & 0x80402010u, 0);
            w0 = __vsub4((int32_t)(uint32_t)gr ^ sm0, sm0);
            w1 = __vsub4((int32_t)(uint32_t)(gr >> 32) ^ sm1, sm1);
            int acc = 0;
            acc = __dp4a(w0, *(const int32_t *)(q8 + (g * 32 + j * 8)), acc);
            acc = __dp4a(w1, *(const int32_t *)(q8 + (g * 32 + j * 8 + 4)), acc);
            if (j < 2) s1 += acc; else s2 += acc;
        }
        sumf += (float)(ls1 * s1 + ls2 * s2);
    }
    return 0.125f * xd * y->d * sumf;
}

/* iq3_xxs: first 64 qs bytes = 256 values via u32 grid rows of 4; the
 * trailing 8 u32 hold 7-bit sign indices (ksigns) + a 4-bit scale. */
__device__ static float dev_dot_iq3_xxs_q8_K_block(
        const block_iq3_xxs *x, const block_q8_K *y,
        const uint32_t *grid, const uint8_t *signs) {
    const float xd = f16_to_f32(x->d);
    const uint8_t *qg = x->qs;
    const int8_t *q8 = y->qs;
    float sumf = 0.0f;
    for (int g = 0; g < 8; g++) { /* 8 groups of 32 values */
        uint32_t aux;
        memcpy(&aux, x->qs + 64 + 4 * g, 4);
        const float db = xd * (0.5f + (float)(aux >> 28)) * 0.5f;
        int sumi = 0;
        for (int j = 0; j < 4; j++) { /* 4 sub-groups of 8 */
            const uint32_t sgn = dev_unpack_iq2_signs(signs[(aux >> (7 * j)) & 127]);
            const int32_t sm0 = __vcmpne4(sgn & 0x08040201u, 0);
            const int32_t sm1 = __vcmpne4(sgn & 0x80402010u, 0);
            uint32_t g0, g1;
            memcpy(&g0, &grid[qg[g * 4 * 2 + j * 2]], 4);
            memcpy(&g1, &grid[qg[g * 4 * 2 + j * 2 + 1]], 4);
            const int32_t w0 = __vsub4((int32_t)g0 ^ sm0, sm0);
            const int32_t w1 = __vsub4((int32_t)g1 ^ sm1, sm1);
            sumi = __dp4a(w0, *(const int32_t *)(q8 + g * 32 + j * 8), sumi);
            sumi = __dp4a(w1, *(const int32_t *)(q8 + g * 32 + j * 8 + 4), sumi);
        }
        sumf += db * (float)sumi;
    }
    return y->d * sumf;
}

/* q4_0: eight 32-element blocks per q8_K super-block; value = nib - 8,
 * folded via bsums (two 16-sums per q4_0 block). */
__device__ static float dev_dot_q4_0_q8_K_block(const char *row, const block_q8_K *y) {
    const block_q4_0 *xb = (const block_q4_0 *)row;
    const int8_t *q8 = y->qs;
    float sumf = 0.0f;
    for (int b = 0; b < 8; b++) {
        const block_q4_0 *x = xb + b;
        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 16; i += 4) {
            uint32_t v;
            memcpy(&v, x->qs + i, 4);
            sumi = __dp4a((int)(v & 0x0f0f0f0fu), *(const int32_t *)(q8 + b * 32 + i), sumi);
            sumi = __dp4a((int)((v >> 4) & 0x0f0f0f0fu), *(const int32_t *)(q8 + b * 32 + 16 + i), sumi);
        }
        const int bsum = y->bsums[2 * b] + y->bsums[2 * b + 1];
        sumf += f16_to_f32(x->d) * (float)(sumi - 8 * bsum);
    }
    return y->d * sumf;
}

/* q5_1 (Gemma 4 down_exps): eight 32-element blocks per q8_K super-block.
 * value = d * ((nib | qh_bit<<4)) + m; the +m term folds via bsums. */
__device__ static float dev_dot_q5_1_q8_K_block(const char *row, const block_q8_K *y) {
    const block_q5_1 *xb = (const block_q5_1 *)row;
    const int8_t *q8 = y->qs;
    float sumf = 0.0f;
    for (int b = 0; b < 8; b++) {
        const block_q5_1 *x = xb + b;
        uint32_t qh;
        memcpy(&qh, &x->qh, 4);
        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 16; i += 4) {
            uint32_t v;
            memcpy(&v, x->qs + i, 4);
            const uint32_t hlo = (((qh >> (i + 0)) & 1u) << 4) |
                                 (((qh >> (i + 1)) & 1u) << 12) |
                                 (((qh >> (i + 2)) & 1u) << 20) |
                                 (((qh >> (i + 3)) & 1u) << 28);
            const uint32_t hhi = (((qh >> (i + 16)) & 1u) << 4) |
                                 (((qh >> (i + 17)) & 1u) << 12) |
                                 (((qh >> (i + 18)) & 1u) << 20) |
                                 (((qh >> (i + 19)) & 1u) << 28);
            sumi = __dp4a((int)((v & 0x0f0f0f0fu) | hlo), *(const int32_t *)(q8 + b * 32 + i), sumi);
            sumi = __dp4a((int)(((v >> 4) & 0x0f0f0f0fu) | hhi), *(const int32_t *)(q8 + b * 32 + 16 + i), sumi);
        }
        const int bsum = y->bsums[2 * b] + y->bsums[2 * b + 1];
        /* tail guard: a zero-quantized activation block has sumi == 0 and
         * bsum == 0; skipping it keeps garbage weight scales (overread
         * past the row end) from minting inf*0 NaNs */
        if (sumi != 0 || bsum != 0) {
            sumf += f16_to_f32(x->d) * (float)sumi + f16_to_f32(x->m) * (float)bsum;
        }
    }
    return y->d * sumf;
}

/* q8_0 (Gemma 4 down_exps): eight 32-element blocks per q8_K super-block.
 * values are already int8 - no nibble unpack, no offset, no minima - just
 * dp4a and scale by the block's f16 d. (weight qs is 2-byte aligned, so
 * memcpy the 4-byte word like q4_0; the q8_K side is 4-aligned.) */
__device__ static float dev_dot_q8_0_q8_K_block(const char *row, const block_q8_K *y) {
    const q8_0_block *xb = (const q8_0_block *)row;
    const int8_t *q8 = y->qs;
    float sumf = 0.0f;
    for (int b = 0; b < 8; b++) {
        const q8_0_block *x = xb + b;
        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < 32; i += 4) {
            int32_t wv;
            memcpy(&wv, x->q + i, 4);
            sumi = __dp4a(wv, *(const int32_t *)(q8 + b * 32 + i), sumi);
        }
        /* tail guard (see q5_1): zero-quantized activation block dots to 0;
         * skipping keeps overread garbage f16 scales from minting NaNs */
        if (sumi != 0) sumf += f16_to_f32(x->scale_f16) * (float)sumi;
    }
    return y->d * sumf;
}

/* per-block dot functors for the templated MoE kernels */
struct dot_iq2_xxs {
    __device__ __forceinline__ static float block(const char *row, const block_q8_K *xq, uint32_t b) {
        return dev_dot_iq2_xxs_q8_K_block_lut(
                (const block_iq2_xxs *)row + b, xq + b,
                cuda_iq2xxs_grid, cuda_ksigns_iq2xs);
    }
};

struct dot_iq2_xs {
    __device__ __forceinline__ static float block(const char *row, const block_q8_K *xq, uint32_t b) {
        return dev_dot_iq2_xs_q8_K_block(
                (const block_iq2_xs *)row + b, xq + b,
                cuda_iq2xs_grid, cuda_ksigns_iq2xs);
    }
};

struct dot_iq3_xxs {
    __device__ __forceinline__ static float block(const char *row, const block_q8_K *xq, uint32_t b) {
        return dev_dot_iq3_xxs_q8_K_block(
                (const block_iq3_xxs *)row + b, xq + b,
                cuda_iq3xxs_grid, cuda_ksigns_iq2xs);
    }
};

struct dot_q4_0 {
    __device__ __forceinline__ static float block(const char *row, const block_q8_K *xq, uint32_t b) {
        /* 8 q4_0 blocks per 256-element q8_K block */
        return dev_dot_q4_0_q8_K_block(row + (uint64_t)b * 8u * sizeof(block_q4_0), xq + b);
    }
};

struct dot_q5_1 {
    __device__ __forceinline__ static float block(const char *row, const block_q8_K *xq, uint32_t b) {
        return dev_dot_q5_1_q8_K_block(row + (uint64_t)b * 8u * sizeof(block_q5_1), xq + b);
    }
};

struct dot_q8_0 {
    __device__ __forceinline__ static float block(const char *row, const block_q8_K *xq, uint32_t b) {
        /* 8 q8_0 blocks per 256-element q8_K block */
        return dev_dot_q8_0_q8_K_block(row + (uint64_t)b * 8u * sizeof(q8_0_block), xq + b);
    }
};

struct dot_iq4_xs {
    __device__ __forceinline__ static float block(const char *row, const block_q8_K *xq, uint32_t b) {
        return dev_dot_iq4_xs_q8_K_block((const block_iq4_xs *)row + b, xq + b);
    }
};

struct dot_q2_K {
    __device__ __forceinline__ static float block(const char *row, const block_q8_K *xq, uint32_t b) {
        return dev_dot_q2_K_q8_K_block((const block_q2_K *)row + b, xq + b);
    }
};

struct dot_q3_K {
    __device__ __forceinline__ static float block(const char *row, const block_q8_K *xq, uint32_t b) {
        return dev_dot_q3_K_q8_K_block((const block_q3_K *)row + b, xq + b);
    }
};

struct dot_q4_K {
    __device__ __forceinline__ static float block(const char *row, const block_q8_K *xq, uint32_t b) {
        return dev_dot_q4_K_q8_K_block((const block_q4_K *)row + b, xq + b);
    }
};

struct dot_q5_K {
    __device__ __forceinline__ static float block(const char *row, const block_q8_K *xq, uint32_t b) {
        return dev_dot_q5_K_q8_K_block((const block_q5_K *)row + b, xq + b);
    }
};

struct dot_q6_K {
    __device__ __forceinline__ static float block(const char *row, const block_q8_K *xq, uint32_t b) {
        return dev_dot_q6_K_q8_K_block((const block_q6_K *)row + b, xq + b);
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
        uint64_t row_bytes,             /* gate and up share type+in_dim */
        uint32_t act_op) {
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
        mid[mid_off] = pulsar_glu(acc_gate, acc_up, act_op) * weights[slot_off];
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
    PULSAR_QUANT_Q4_K = 2,
    PULSAR_QUANT_Q5_K = 3,
    PULSAR_QUANT_Q6_K = 4,
    PULSAR_QUANT_Q3_K = 5,
    PULSAR_QUANT_IQ2_XS = 6,
    PULSAR_QUANT_IQ3_XXS = 7,
    PULSAR_QUANT_Q4_0 = 8,
    PULSAR_QUANT_Q5_1 = 9,
    PULSAR_QUANT_Q8_0 = 10,
    PULSAR_QUANT_IQ4_XS = 11,
};

/* ---- grouped batch MoE: amortize weight reads across the prefill batch.
 * The plain kernels re-read each expert row once per (token, slot); in a
 * 256-token chunk an expert typically serves ~10 tokens, so rows are read
 * ~10x. Here tokens are grouped by expert (CSR: starts[], pairs[] packing
 * token*256+slot), each block stages its weight rows in shared memory
 * ONCE, and all of the group's tokens dot against the staged copy. Same
 * DOT templates, so every quant format inherits it. Down partials land in
 * mid-layout [token][slot][out_dim] and a deterministic slot-sum follows
 * (no atomics: prefill logits stay reproducible). */

#define PULSAR_GROUP_SMEM 49152 /* dynamic smem default ceiling */

template <typename DOT>
__global__ static void moe_pair_swiglu_grouped_kernel(
        float *mid,                     /* [n_tok][n_used][mid_dim] */
        const pulsar_expert_ptrs *gptrs, /* [n_group] */
        const uint32_t *starts,          /* [n_group+1] */
        const uint32_t *pairs,           /* token*16+slot (n_used <= 16) */
        const float *weights,            /* [n_tok][n_used] */
        const block_q8_K *xq,            /* [n_tok][in_blocks] */
        uint32_t in_blocks,
        uint32_t mid_dim,
        uint32_t n_used,
        uint64_t row_bytes,
        uint32_t act_op) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t group = blockIdx.y;
    const pulsar_expert_ptrs p = gptrs[group];
    if (!p.gate || !p.up) return;
    extern __shared__ char smem[];
    char *gate_s = smem + (uint64_t)threadIdx.y * 2u * row_bytes;
    char *up_s = gate_s + row_bytes;
    if (row < mid_dim) {
        const char *gate_g = (const char *)p.gate + (uint64_t)row * row_bytes;
        const char *up_g = (const char *)p.up + (uint64_t)row * row_bytes;
        for (uint32_t b = lane; b < row_bytes; b += 32u) {
            gate_s[b] = gate_g[b];
            up_s[b] = up_g[b];
        }
    }
    __syncwarp();
    if (row >= mid_dim) return;
    const uint32_t s0 = starts[group], s1 = starts[group + 1];
    for (uint32_t i = s0; i < s1; i++) {
        const uint32_t pr = pairs[i];
        const uint32_t token = pr >> 4;
        const uint32_t slot = pr & 0x0fu;
        const block_q8_K *txq = xq + (uint64_t)token * in_blocks;
        float ag = 0.0f, au = 0.0f;
        for (uint32_t b = lane; b < in_blocks; b += 32u) {
            ag += DOT::block(gate_s, txq, b);
            au += DOT::block(up_s, txq, b);
        }
        #pragma unroll
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
            ag += __shfl_xor_sync(0xffffffffu, ag, mask);
            au += __shfl_xor_sync(0xffffffffu, au, mask);
        }
        if (lane == 0) {
            mid[((uint64_t)token * n_used + slot) * mid_dim + row] =
                pulsar_glu(ag, au, act_op) * weights[(uint64_t)token * n_used + slot];
        }
    }
}

template <typename DOT>
__global__ static void moe_down_grouped_kernel(
        float *partial,                  /* [n_tok][n_used][out_dim] */
        const pulsar_expert_ptrs *gptrs,
        const uint32_t *starts,
        const uint32_t *pairs,
        const block_q8_K *midq,          /* [n_tok][n_used][mid_blocks] */
        uint32_t mid_blocks,
        uint32_t out_dim,
        uint32_t n_used,
        uint64_t row_bytes) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t group = blockIdx.y;
    const pulsar_expert_ptrs p = gptrs[group];
    if (!p.down) return;
    extern __shared__ char smem[];
    char *down_s = smem + (uint64_t)threadIdx.y * row_bytes;
    if (row < out_dim) {
        const char *down_g = (const char *)p.down + (uint64_t)row * row_bytes;
        for (uint32_t b = lane; b < row_bytes; b += 32u) {
            down_s[b] = down_g[b];
        }
    }
    __syncwarp();
    if (row >= out_dim) return;
    const uint32_t s0 = starts[group], s1 = starts[group + 1];
    for (uint32_t i = s0; i < s1; i++) {
        const uint32_t pr = pairs[i];
        const uint32_t token = pr >> 4;
        const uint32_t slot = pr & 0x0fu;
        const block_q8_K *smq = midq + ((uint64_t)token * n_used + slot) * mid_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < mid_blocks; b += 32u) {
            acc += DOT::block(down_s, smq, b);
        }
        #pragma unroll
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
            acc += __shfl_xor_sync(0xffffffffu, acc, mask);
        }
        if (lane == 0) {
            partial[((uint64_t)token * n_used + slot) * out_dim + row] = acc;
        }
    }
}

/* deterministic slot reduce: out[t][r] = sum_s partial[t][s][r]; slots
 * with NULL down never wrote - engine zeroes partial for those first */
__global__ static void moe_slot_sum_kernel(
        float *out, const float *partial, uint32_t out_dim, uint32_t n_used,
        uint32_t n_tok) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_tok * out_dim;
    if (gid >= total) return;
    const uint32_t token = (uint32_t)(gid / out_dim);
    const uint32_t row = (uint32_t)(gid - (uint64_t)token * out_dim);
    float acc = 0.0f;
    for (uint32_t s = 0; s < n_used; s++) {
        acc += partial[((uint64_t)token * n_used + s) * out_dim + row];
    }
    out[gid] = acc;
}

/* ---- mmq-style tensor-core MoE (sm_80+, phase 3) -----------------------
 * The grouped CSR already amortizes weight traffic across a prefill
 * chunk's tokens; here it also amortizes DEQUANT: each block unpacks one
 * superblock of its 16 weight rows to int8 in shared memory (plus a
 * per-16-chunk float scale), then every token tile consumes the unpacked
 * copy through mma.m16n8k16 s8s8s32, rescaling per chunk in registers
 * via the documented PTX fragment mapping. k=16 chunks make ONE uniform
 * inner loop legal for every quant format - per-format code is only the
 * 16-element unpacker. v1 formats: iq2_xxs (Hy3/GLM routed experts). */

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ static void mma_s8_16x8x16(
        int32_t d[4], const int32_t a[2], const int32_t b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(b),
          "r"(0), "r"(0), "r"(0), "r"(0));
}
#endif

/* unpacker functors: 16 elements (chunk c of a 256-superblock) of one
 * row into int8[16] + one float scale + one float min offset. The chunk
 * dot is scale * sumi - minoff * bsum (bsum = the activation chunk's
 * int sum, q8_K keeps one per 16 elements); HAS_MIN=false formats leave
 * minoff untouched and the harness compiles the bsum term out. */
struct unpack_iq2_xxs {
    static const bool HAS_MIN = false;
    static const uint32_t SB_BYTES = sizeof(block_iq2_xxs);
    __device__ __forceinline__ static void chunk16(
            const char *block, uint32_t c, int32_t out[4], float *scale, float *minoff) {
        (void)minoff;
        const block_iq2_xxs *x = (const block_iq2_xxs *)block;
        const uint16_t *q2 = x->qs + (c >> 1) * 4u;
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        const uint32_t h = c & 1u;
        dev_iq2_i8x8_lut(cuda_iq2xxs_grid, cuda_ksigns_iq2xs,
                         (uint8_t)((aux0 >> (16u * h)) & 0xffu),
                         (aux1 >> (14u * h)) & 127u, &out[0], &out[1]);
        dev_iq2_i8x8_lut(cuda_iq2xxs_grid, cuda_ksigns_iq2xs,
                         (uint8_t)((aux0 >> (16u * h + 8u)) & 0xffu),
                         (aux1 >> (14u * h + 7u)) & 127u, &out[2], &out[3]);
        const float ls = (float)(2u * (aux1 >> 28) + 1u);
        *scale = 0.125f * f16_to_f32(x->d) * ls;
    }
};

struct unpack_iq2_xs {
    static const bool HAS_MIN = false;
    static const uint32_t SB_BYTES = sizeof(block_iq2_xs);
    __device__ __forceinline__ static void chunk16(
            const char *block, uint32_t c, int32_t out[4], float *scale, float *minoff) {
        (void)minoff;
        const block_iq2_xs *x = (const block_iq2_xs *)block;
        const uint32_t g = c >> 1;
        const uint32_t h = c & 1u;
        #pragma unroll
        for (uint32_t j = 0; j < 2; j++) {
            const uint16_t q = x->qs[g * 4u + h * 2u + j];
            const uint64_t gr = cuda_iq2xs_grid[q & 511u];
            const uint32_t sgn = dev_unpack_iq2_signs(cuda_ksigns_iq2xs[q >> 9]);
            const int32_t sm0 = __vcmpne4(sgn & 0x08040201u, 0);
            const int32_t sm1 = __vcmpne4(sgn & 0x80402010u, 0);
            out[j * 2u] = __vsub4((int32_t)(uint32_t)gr ^ sm0, sm0);
            out[j * 2u + 1u] = __vsub4((int32_t)(uint32_t)(gr >> 32) ^ sm1, sm1);
        }
        const float ls = (float)(2u * ((x->scales[g] >> (4u * h)) & 0x0fu) + 1u);
        *scale = 0.125f * f16_to_f32(x->d) * ls;
    }
};

struct unpack_iq3_xxs {
    static const bool HAS_MIN = false;
    static const uint32_t SB_BYTES = sizeof(block_iq3_xxs);
    __device__ __forceinline__ static void chunk16(
            const char *block, uint32_t c, int32_t out[4], float *scale, float *minoff) {
        (void)minoff;
        const block_iq3_xxs *x = (const block_iq3_xxs *)block;
        const uint32_t g = c >> 1;
        const uint32_t h = c & 1u;
        uint32_t aux;
        memcpy(&aux, x->qs + 64u + 4u * g, 4u);
        #pragma unroll
        for (uint32_t j = 0; j < 2; j++) {
            const uint32_t sj = h * 2u + j; /* sub-group of 8 within g */
            const uint32_t sgn = dev_unpack_iq2_signs(
                    cuda_ksigns_iq2xs[(aux >> (7u * sj)) & 127u]);
            const int32_t sm0 = __vcmpne4(sgn & 0x08040201u, 0);
            const int32_t sm1 = __vcmpne4(sgn & 0x80402010u, 0);
            uint32_t g0, g1;
            memcpy(&g0, &cuda_iq3xxs_grid[x->qs[g * 8u + sj * 2u]], 4u);
            memcpy(&g1, &cuda_iq3xxs_grid[x->qs[g * 8u + sj * 2u + 1u]], 4u);
            out[j * 2u] = __vsub4((int32_t)g0 ^ sm0, sm0);
            out[j * 2u + 1u] = __vsub4((int32_t)g1 ^ sm1, sm1);
        }
        *scale = f16_to_f32(x->d) * (0.5f + (float)(aux >> 28)) * 0.5f;
    }
};

struct unpack_q4_0 {
    static const bool HAS_MIN = false;
    static const uint32_t SB_BYTES = 8u * sizeof(block_q4_0);
    __device__ __forceinline__ static void chunk16(
            const char *block, uint32_t c, int32_t out[4], float *scale, float *minoff) {
        (void)minoff;
        const block_q4_0 *x = (const block_q4_0 *)block + (c >> 1);
        const uint32_t shift = (c & 1u) * 4u; /* low / high nibbles */
        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            uint32_t v;
            memcpy(&v, x->qs + j * 4u, 4u);
            /* (nib - 8) per byte lane */
            out[j] = __vsub4((int32_t)((v >> shift) & 0x0f0f0f0fu), 0x08080808);
        }
        *scale = f16_to_f32(x->d);
    }
};

struct unpack_q8_0 {
    static const bool HAS_MIN = false;
    static const uint32_t SB_BYTES = 8u * sizeof(q8_0_block);
    __device__ __forceinline__ static void chunk16(
            const char *block, uint32_t c, int32_t out[4], float *scale, float *minoff) {
        (void)minoff;
        const q8_0_block *x = (const q8_0_block *)block + (c >> 1);
        const int8_t *q = x->q + (c & 1u) * 16u; /* 2-byte aligned */
        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) memcpy(&out[j], q + j * 4u, 4u);
        *scale = f16_to_f32(x->scale_f16);
    }
};

struct unpack_iq4_xs {
    static const bool HAS_MIN = false;
    static const uint32_t SB_BYTES = sizeof(block_iq4_xs);
    __device__ __forceinline__ static void chunk16(
            const char *block, uint32_t c, int32_t out[4], float *scale, float *minoff) {
        (void)minoff;
        const block_iq4_xs *x = (const block_iq4_xs *)block;
        const uint32_t s = c >> 1;
        const uint32_t shift = (c & 1u) * 4u;
        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            int8_t r[4];
            #pragma unroll
            for (uint32_t k = 0; k < 4; k++) {
                r[k] = kd_iq4nl[(x->qs[s * 16u + j * 4u + k] >> shift) & 0xfu];
            }
            memcpy(&out[j], r, 4u);
        }
        const int ls = (int)(((x->scales_l[s >> 1] >> (4u * (s & 1u))) & 0xfu) |
                             (((x->scales_h >> (2u * s)) & 3u) << 4u)) - 32;
        *scale = f16_to_f32(x->d) * (float)ls;
    }
};

/* minima formats: chunk dot = scale*sumi - minoff*bsum. q5_1 ADDS its
 * offset m (minoff = -m folds the sign). */
struct unpack_q5_1 {
    static const bool HAS_MIN = true;
    static const uint32_t SB_BYTES = 8u * sizeof(block_q5_1);
    __device__ __forceinline__ static void chunk16(
            const char *block, uint32_t c, int32_t out[4], float *scale, float *minoff) {
        const block_q5_1 *x = (const block_q5_1 *)block + (c >> 1);
        uint32_t qh;
        memcpy(&qh, &x->qh, 4u);
        const uint32_t hs = (c & 1u) * 16u; /* qh bit base for this half */
        const uint32_t shift = (c & 1u) * 4u;
        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            uint32_t v;
            memcpy(&v, x->qs + j * 4u, 4u);
            const uint32_t hb = (((qh >> (hs + j * 4u + 0u)) & 1u) << 4)
                              | (((qh >> (hs + j * 4u + 1u)) & 1u) << 12)
                              | (((qh >> (hs + j * 4u + 2u)) & 1u) << 20)
                              | (((qh >> (hs + j * 4u + 3u)) & 1u) << 28);
            out[j] = (int32_t)(((v >> shift) & 0x0f0f0f0fu) | hb);
        }
        *scale = f16_to_f32(x->d);
        *minoff = -f16_to_f32(x->m);
    }
};

struct unpack_q2_K {
    static const bool HAS_MIN = true;
    static const uint32_t SB_BYTES = sizeof(block_q2_K);
    __device__ __forceinline__ static void chunk16(
            const char *block, uint32_t c, int32_t out[4], float *scale, float *minoff) {
        const block_q2_K *x = (const block_q2_K *)block;
        const uint32_t boff = (c >> 3u) * 32u + (c & 1u) * 16u;
        const uint32_t shift = 2u * ((c >> 1u) & 3u);
        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            uint32_t v;
            memcpy(&v, x->qs + boff + j * 4u, 4u);
            out[j] = (int32_t)((v >> shift) & 0x03030303u);
        }
        const uint32_t sb = x->scales[c]; /* 16 sub-blocks == 16 chunks */
        *scale = f16_to_f32(x->d) * (float)(sb & 0xfu);
        *minoff = f16_to_f32(x->dmin) * (float)(sb >> 4);
    }
};

struct unpack_q3_K {
    static const bool HAS_MIN = false; /* centered by hmask, no minima */
    static const uint32_t SB_BYTES = sizeof(block_q3_K);
    __device__ __forceinline__ static void chunk16(
            const char *block, uint32_t c, int32_t out[4], float *scale, float *minoff) {
        (void)minoff;
        const block_q3_K *x = (const block_q3_K *)block;
        const uint32_t boff = (c >> 3u) * 32u + (c & 1u) * 16u;
        const uint32_t shift = 2u * ((c >> 1u) & 3u);
        const uint32_t hbit = c >> 1u;      /* hmask bit for this chunk */
        const uint32_t hoff = (c & 1u) * 16u;
        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            uint32_t v, h;
            memcpy(&v, x->qs + boff + j * 4u, 4u);
            memcpy(&h, x->hmask + hoff + j * 4u, 4u);
            /* q = lo2 - (hbit set ? 0 : 4) */
            const int32_t miss = __vcmpeq4((int32_t)((h >> hbit) & 0x01010101u), 0);
            out[j] = __vsub4((int32_t)((v >> shift) & 0x03030303u),
                             (int32_t)(0x04040404u & (uint32_t)miss));
        }
        int8_t sc[16];
        k3_unpack_scales(x->scales, sc);
        *scale = f16_to_f32(x->d) * (float)sc[c];
    }
};

struct unpack_q6_K {
    static const bool HAS_MIN = false; /* centered -32, signed scales */
    static const uint32_t SB_BYTES = sizeof(block_q6_K);
    __device__ __forceinline__ static void chunk16(
            const char *block, uint32_t c, int32_t out[4], float *scale, float *minoff) {
        (void)minoff;
        const block_q6_K *x = (const block_q6_K *)block;
        const uint32_t k = c >> 3u;       /* 128-element half */
        const uint32_t cc = c & 7u;       /* chunk within the half */
        const uint32_t qq = cc >> 1u;     /* quarter: q1/q2/q3/q4 region */
        const uint32_t loff = k * 64u + (qq & 1u) * 32u + (cc & 1u) * 16u;
        const uint32_t hoff = k * 32u + (cc & 1u) * 16u;
        const uint32_t lsh = (qq >> 1u) * 4u;
        const uint32_t hsh = qq * 2u;
        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            uint32_t v, h; /* 210-byte blocks: 2-aligned, memcpy only */
            memcpy(&v, x->ql + loff + j * 4u, 4u);
            memcpy(&h, x->qh + hoff + j * 4u, 4u);
            const uint32_t q = ((v >> lsh) & 0x0f0f0f0fu)
                             | (((h >> hsh) & 0x03030303u) << 4);
            out[j] = __vsub4((int32_t)q, 0x20202020);
        }
        *scale = f16_to_f32(x->d) * (float)x->scales[c];
    }
};

struct unpack_q4_K {
    static const bool HAS_MIN = true;
    static const uint32_t SB_BYTES = sizeof(block_q4_K);
    __device__ __forceinline__ static void chunk16(
            const char *block, uint32_t c, int32_t out[4], float *scale, float *minoff) {
        const block_q4_K *x = (const block_q4_K *)block;
        const uint32_t s = c >> 1u;                       /* sub-block 0..7 */
        const uint32_t boff = (s >> 1u) * 32u + (c & 1u) * 16u;
        const uint32_t shift = (s & 1u) * 4u;
        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            uint32_t v;
            memcpy(&v, x->qs + boff + j * 4u, 4u);
            out[j] = (int32_t)((v >> shift) & 0x0f0f0f0fu);
        }
        uint8_t sc, m;
        k4_scale_min(s, x->scales, &sc, &m);
        *scale = f16_to_f32(x->d) * (float)sc;
        *minoff = f16_to_f32(x->dmin) * (float)m;
    }
};

struct unpack_q5_K {
    static const bool HAS_MIN = true;
    static const uint32_t SB_BYTES = sizeof(block_q5_K);
    __device__ __forceinline__ static void chunk16(
            const char *block, uint32_t c, int32_t out[4], float *scale, float *minoff) {
        const block_q5_K *x = (const block_q5_K *)block;
        const uint32_t s = c >> 1u;
        const uint32_t boff = (s >> 1u) * 32u + (c & 1u) * 16u;
        const uint32_t shift = (s & 1u) * 4u;
        const uint32_t hoff = (c & 1u) * 16u; /* qh byte base (no j advance) */
        #pragma unroll
        for (uint32_t j = 0; j < 4; j++) {
            uint32_t v, h;
            memcpy(&v, x->qs + boff + j * 4u, 4u);
            memcpy(&h, x->qh + hoff + j * 4u, 4u);
            const uint32_t hb = ((h >> s) & 0x01010101u) << 4;
            out[j] = (int32_t)(((v >> shift) & 0x0f0f0f0fu) | hb);
        }
        uint8_t sc, m;
        k4_scale_min(s, x->scales, &sc, &m);
        *scale = f16_to_f32(x->d) * (float)sc;
        *minoff = f16_to_f32(x->dmin) * (float)m;
    }
};

#define PULSAR_MMA_TPW 4u /* token tiles per warp: 8 warps x 4 x 8 = 256 */

/* One block: 16 weight rows of one expert group vs all its tokens.
 * kind 0 = pair (gate+up staged interleaved, swiglu+route-weight at the
 * end), kind 1 = down (midq activations, partial write for slot-sum). */
template <int KIND, typename UNPACK>
__global__ static void moe_grouped_mma_kernel(
        float *out,                      /* pair: mid / down: partial */
        const pulsar_expert_ptrs *gptrs,
        const uint32_t *starts,
        const uint32_t *pairs,
        const float *weights,            /* [n_tok][n_used] (pair only) */
        const void *actq,                /* block_q8_K rows */
        uint32_t n_blocks,               /* superblocks along k */
        uint32_t n_rows,                 /* mid_dim (pair) / out_dim (down) */
        uint32_t n_used,
        uint64_t row_bytes,
        uint32_t act_op) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t group = blockIdx.y;
    const uint32_t r0 = blockIdx.x * 16u;
    const pulsar_expert_ptrs p = gptrs[group];
    if (KIND == 0 ? (!p.gate || !p.up) : !p.down) return;
    const uint32_t s0 = starts[group];
    const uint32_t len = starts[group + 1] - s0;
    if (len == 0 || r0 >= n_rows) return;

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t g = lane >> 2u;
    const uint32_t tg = lane & 3u;

    /* A rows for this block's tile, gate and up as two planes */
    __shared__ int8_t a_s[2][16][256];
    __shared__ float s_w[2][16][16];
    /* per-chunk min offsets (minima formats): dot = sc*sumi - mo*bsum.
     * dimensioned [2] unconditionally: the !HAS_MIN branches are dead
     * code but still type-checked, and [1]-dim indexing would be UB */
    __shared__ float s_m[2][16][16];
    const uint32_t planes = KIND == 0 ? 2u : 1u;

    const uint32_t n_tiles = (len + 7u) / 8u;
    const uint32_t tiles_per_round = 8u * PULSAR_MMA_TPW;

    for (uint32_t round = 0; round * tiles_per_round < n_tiles; round++) {
        float accg[PULSAR_MMA_TPW][4];
        float accu[PULSAR_MMA_TPW][4];
        #pragma unroll
        for (uint32_t t = 0; t < PULSAR_MMA_TPW; t++) {
            #pragma unroll
            for (uint32_t i = 0; i < 4; i++) { accg[t][i] = 0.0f; accu[t][i] = 0.0f; }
        }

        for (uint32_t sb = 0; sb < n_blocks; sb++) {
            __syncthreads();
            /* 256 threads unpack 16 rows x 16 chunks per plane */
            {
                const uint32_t r = threadIdx.x >> 4u;
                const uint32_t c = threadIdx.x & 15u;
                for (uint32_t pl = 0; pl < planes; pl++) {
                    const char *base = (const char *)(KIND == 0
                            ? (pl == 0 ? p.gate : p.up) : p.down);
                    int32_t w4[4] = {0, 0, 0, 0};
                    float sc = 0.0f;
                    float mo = 0.0f;
                    if (r0 + r < n_rows) {
                        UNPACK::chunk16(base + (uint64_t)(r0 + r) * row_bytes
                                                + (uint64_t)sb * UNPACK::SB_BYTES,
                                        c, w4, &sc, &mo);
                    }
                    *(int4 *)&a_s[pl][r][c * 16u] = *(const int4 *)w4;
                    s_w[pl][r][c] = sc;
                    if (UNPACK::HAS_MIN) s_m[pl][r][c] = mo;
                }
            }
            __syncthreads();

            for (uint32_t t = 0; t < PULSAR_MMA_TPW; t++) {
                const uint32_t tile = round * tiles_per_round + warp * PULSAR_MMA_TPW + t;
                if (tile >= n_tiles) break;
                /* lane j < 8 holds pair j of the tile; shfl distributes */
                uint32_t pr_own = 0xffffffffu;
                if (lane < 8u && tile * 8u + lane < len) {
                    pr_own = pairs[s0 + tile * 8u + lane];
                }
                const uint32_t pr_b = __shfl_sync(0xffffffffu, pr_own, (int)g);
                const uint32_t pr_c0 = __shfl_sync(0xffffffffu, pr_own, (int)(tg * 2u));
                const uint32_t pr_c1 = __shfl_sync(0xffffffffu, pr_own, (int)(tg * 2u + 1u));
                /* activation row: pair kernel reads token rows, down reads
                 * (token*n_used + slot) mid rows */
                const uint32_t arow_b = KIND == 0 ? (pr_b >> 4) : (pr_b >> 4) * n_used + (pr_b & 15u);
                const uint32_t arow_c0 = KIND == 0 ? (pr_c0 >> 4) : (pr_c0 >> 4) * n_used + (pr_c0 & 15u);
                const uint32_t arow_c1 = KIND == 0 ? (pr_c1 >> 4) : (pr_c1 >> 4) * n_used + (pr_c1 & 15u);
                const char *yb = pr_b != 0xffffffffu
                        ? (const char *)actq + ((uint64_t)arow_b * n_blocks + sb) * sizeof(block_q8_K)
                        : NULL;
                const char *yc0 = pr_c0 != 0xffffffffu
                        ? (const char *)actq + ((uint64_t)arow_c0 * n_blocks + sb) * sizeof(block_q8_K)
                        : NULL;
                const char *yc1 = pr_c1 != 0xffffffffu
                        ? (const char *)actq + ((uint64_t)arow_c1 * n_blocks + sb) * sizeof(block_q8_K)
                        : NULL;
                const float d0 = yc0 ? ((const block_q8_K *)yc0)->d : 0.0f;
                const float d1 = yc1 ? ((const block_q8_K *)yc1)->d : 0.0f;

                #pragma unroll
                for (uint32_t ch = 0; ch < 16u; ch++) {
                    const int32_t b = yb
                            ? *(const int32_t *)(yb + 4u + ch * 16u + tg * 4u)
                            : 0;
                    /* minima formats: chunk dot = sc*sumi - mo*bsum; the
                     * bsum rides the same activation scale d as sumi.
                     * q8_K bsums: one i16 per 16-element chunk at +260 */
                    float bs0 = 0.0f, bs1 = 0.0f;
                    if (UNPACK::HAS_MIN) {
                        bs0 = yc0 ? (float)((const int16_t *)(yc0 + 4u + 256u))[ch] : 0.0f;
                        bs1 = yc1 ? (float)((const int16_t *)(yc1 + 4u + 256u))[ch] : 0.0f;
                        accg[t][0] -= s_m[0][g][ch] * bs0 * d0;
                        accg[t][1] -= s_m[0][g][ch] * bs1 * d1;
                        accg[t][2] -= s_m[0][g + 8u][ch] * bs0 * d0;
                        accg[t][3] -= s_m[0][g + 8u][ch] * bs1 * d1;
                        if (KIND == 0) {
                            accu[t][0] -= s_m[1][g][ch] * bs0 * d0;
                            accu[t][1] -= s_m[1][g][ch] * bs1 * d1;
                            accu[t][2] -= s_m[1][g + 8u][ch] * bs0 * d0;
                            accu[t][3] -= s_m[1][g + 8u][ch] * bs1 * d1;
                        }
                    }
                    int32_t a[2], dsum[4];
                    a[0] = *(const int32_t *)&a_s[0][g][ch * 16u + tg * 4u];
                    a[1] = *(const int32_t *)&a_s[0][g + 8u][ch * 16u + tg * 4u];
                    mma_s8_16x8x16(dsum, a, b);
                    const float sg0 = s_w[0][g][ch];
                    const float sg8 = s_w[0][g + 8u][ch];
                    accg[t][0] += (float)dsum[0] * sg0 * d0;
                    accg[t][1] += (float)dsum[1] * sg0 * d1;
                    accg[t][2] += (float)dsum[2] * sg8 * d0;
                    accg[t][3] += (float)dsum[3] * sg8 * d1;
                    if (KIND == 0) {
                        a[0] = *(const int32_t *)&a_s[1][g][ch * 16u + tg * 4u];
                        a[1] = *(const int32_t *)&a_s[1][g + 8u][ch * 16u + tg * 4u];
                        mma_s8_16x8x16(dsum, a, b);
                        const float su0 = s_w[1][g][ch];
                        const float su8 = s_w[1][g + 8u][ch];
                        accu[t][0] += (float)dsum[0] * su0 * d0;
                        accu[t][1] += (float)dsum[1] * su0 * d1;
                        accu[t][2] += (float)dsum[2] * su8 * d0;
                        accu[t][3] += (float)dsum[3] * su8 * d1;
                    }
                }
            }
        }

        /* writeback this round's tiles */
        for (uint32_t t = 0; t < PULSAR_MMA_TPW; t++) {
            const uint32_t tile = round * tiles_per_round + warp * PULSAR_MMA_TPW + t;
            if (tile >= n_tiles) break;
            uint32_t pr_own = 0xffffffffu;
            if (lane < 8u && tile * 8u + lane < len) {
                pr_own = pairs[s0 + tile * 8u + lane];
            }
            #pragma unroll
            for (uint32_t half = 0; half < 2u; half++) {
                const uint32_t col = tg * 2u + half;
                const uint32_t pr = __shfl_sync(0xffffffffu, pr_own, (int)col);
                if (pr == 0xffffffffu) continue;
                const uint32_t token = pr >> 4;
                const uint32_t slot = pr & 15u;
                const uint64_t srow = (uint64_t)token * n_used + slot;
                #pragma unroll
                for (uint32_t rh = 0; rh < 2u; rh++) {
                    const uint32_t row = r0 + g + rh * 8u;
                    if (row >= n_rows) continue;
                    const uint32_t ci = rh * 2u + half;
                    if (KIND == 0) {
                        out[srow * n_rows + row] =
                            pulsar_glu(accg[t][ci], accu[t][ci], act_op) * weights[srow];
                    } else {
                        out[srow * n_rows + row] = accg[t][ci];
                    }
                }
            }
        }
    }
#else
    (void)out; (void)gptrs; (void)starts; (void)pairs; (void)weights;
    (void)actq; (void)n_blocks; (void)n_rows; (void)n_used; (void)row_bytes;
    (void)act_op;
#endif
}

#define PULSAR_GROUPED_DISPATCH(kern, ...)                                    \
    do {                                                                      \
        switch (quant) {                                                      \
        case PULSAR_QUANT_Q2_K:    kern<dot_q2_K><<<grid, block, shmem>>>(__VA_ARGS__); break; \
        case PULSAR_QUANT_IQ2_XXS: kern<dot_iq2_xxs><<<grid, block, shmem>>>(__VA_ARGS__); break; \
        case PULSAR_QUANT_Q3_K:    kern<dot_q3_K><<<grid, block, shmem>>>(__VA_ARGS__); break; \
        case PULSAR_QUANT_Q4_K:    kern<dot_q4_K><<<grid, block, shmem>>>(__VA_ARGS__); break; \
        case PULSAR_QUANT_Q5_K:    kern<dot_q5_K><<<grid, block, shmem>>>(__VA_ARGS__); break; \
        case PULSAR_QUANT_Q6_K:    kern<dot_q6_K><<<grid, block, shmem>>>(__VA_ARGS__); break; \
        case PULSAR_QUANT_IQ2_XS:  kern<dot_iq2_xs><<<grid, block, shmem>>>(__VA_ARGS__); break; \
        case PULSAR_QUANT_IQ3_XXS: kern<dot_iq3_xxs><<<grid, block, shmem>>>(__VA_ARGS__); break; \
        case PULSAR_QUANT_Q4_0:    kern<dot_q4_0><<<grid, block, shmem>>>(__VA_ARGS__); break; \
        case PULSAR_QUANT_Q5_1:    kern<dot_q5_1><<<grid, block, shmem>>>(__VA_ARGS__); break; \
        case PULSAR_QUANT_Q8_0:    kern<dot_q8_0><<<grid, block, shmem>>>(__VA_ARGS__); break; \
        case PULSAR_QUANT_IQ4_XS:  kern<dot_iq4_xs><<<grid, block, shmem>>>(__VA_ARGS__); break; \
        default: return 0;                                                    \
        }                                                                     \
    } while (0)

extern "C" int pulsar_moe_pair_swiglu_grouped(
        void *mid_dev, const void *gptrs_dev, const void *starts_dev,
        const void *pairs_dev, const void *weights_dev, const void *xq_dev,
        uint32_t in_dim, uint32_t mid_dim, uint32_t n_used, uint32_t n_group,
        uint64_t row_bytes, uint32_t quant, uint32_t act_op) {
    if (in_dim == 0 || in_dim % PULSAR_QK_K != 0 || mid_dim == 0 ||
        n_used == 0 || n_group == 0 || row_bytes == 0 ||
        2u * row_bytes * 4u > PULSAR_GROUP_SMEM) {
        return 0;
    }
    const uint32_t in_blocks = in_dim / PULSAR_QK_K;
    if (pulsar_device_cc_major() >= 8 && !getenv("PULSAR_NO_MMA")) {
        dim3 mgrid((mid_dim + 15u) / 16u, n_group, 1);
        switch (quant) {
#define PULSAR_MMA_PAIR(Q, U)                                                  \
        case Q:                                                                \
            moe_grouped_mma_kernel<0, U><<<mgrid, 256>>>(                      \
                    (float *)mid_dev, (const pulsar_expert_ptrs *)gptrs_dev,   \
                    (const uint32_t *)starts_dev, (const uint32_t *)pairs_dev, \
                    (const float *)weights_dev, xq_dev,                        \
                    in_blocks, mid_dim, n_used, row_bytes, act_op);            \
            return cuda_ok(cudaGetLastError(), "moe pair mma launch")
        PULSAR_MMA_PAIR(PULSAR_QUANT_IQ2_XXS, unpack_iq2_xxs);
        PULSAR_MMA_PAIR(PULSAR_QUANT_IQ2_XS, unpack_iq2_xs);
        PULSAR_MMA_PAIR(PULSAR_QUANT_IQ3_XXS, unpack_iq3_xxs);
        PULSAR_MMA_PAIR(PULSAR_QUANT_Q4_0, unpack_q4_0);
        PULSAR_MMA_PAIR(PULSAR_QUANT_Q8_0, unpack_q8_0);
        PULSAR_MMA_PAIR(PULSAR_QUANT_IQ4_XS, unpack_iq4_xs);
        PULSAR_MMA_PAIR(PULSAR_QUANT_Q5_1, unpack_q5_1);
        PULSAR_MMA_PAIR(PULSAR_QUANT_Q2_K, unpack_q2_K);
        PULSAR_MMA_PAIR(PULSAR_QUANT_Q3_K, unpack_q3_K);
        PULSAR_MMA_PAIR(PULSAR_QUANT_Q4_K, unpack_q4_K);
        PULSAR_MMA_PAIR(PULSAR_QUANT_Q5_K, unpack_q5_K);
        PULSAR_MMA_PAIR(PULSAR_QUANT_Q6_K, unpack_q6_K);
#undef PULSAR_MMA_PAIR
        default: break;
        }
    }
    dim3 block(32, 4, 1);
    dim3 grid((mid_dim + 3u) / 4u, n_group, 1);
    const uint32_t shmem = 2u * (uint32_t)row_bytes * 4u;
    PULSAR_GROUPED_DISPATCH(moe_pair_swiglu_grouped_kernel,
            (float *)mid_dev, (const pulsar_expert_ptrs *)gptrs_dev,
            (const uint32_t *)starts_dev, (const uint32_t *)pairs_dev,
            (const float *)weights_dev, (const block_q8_K *)xq_dev,
            in_blocks, mid_dim, n_used, row_bytes, act_op);
    return cuda_ok(cudaGetLastError(), "moe pair swiglu grouped launch");
}

extern "C" int pulsar_moe_down_grouped(
        void *partial_dev, const void *gptrs_dev, const void *starts_dev,
        const void *pairs_dev, const void *midq_dev,
        uint32_t mid_dim, uint32_t out_dim, uint32_t n_used, uint32_t n_group,
        uint64_t row_bytes, uint32_t quant) {
    if (mid_dim == 0 || out_dim == 0 ||
        n_used == 0 || n_group == 0 || row_bytes == 0 ||
        row_bytes * 4u > PULSAR_GROUP_SMEM ||
        /* smem row copies have no slack for the tail overread */
        mid_dim % PULSAR_QK_K != 0) {
        return 0;
    }
    const uint32_t mid_blocks = mid_dim / PULSAR_QK_K;
    if (pulsar_device_cc_major() >= 8 && !getenv("PULSAR_NO_MMA") &&
        mid_dim % PULSAR_QK_K == 0) {
        dim3 mgrid((out_dim + 15u) / 16u, n_group, 1);
        switch (quant) {
#define PULSAR_MMA_DOWN(Q, U)                                                  \
        case Q:                                                                \
            moe_grouped_mma_kernel<1, U><<<mgrid, 256>>>(                      \
                    (float *)partial_dev, (const pulsar_expert_ptrs *)gptrs_dev, \
                    (const uint32_t *)starts_dev, (const uint32_t *)pairs_dev, \
                    NULL, midq_dev, mid_blocks, out_dim, n_used, row_bytes, 0); \
            return cuda_ok(cudaGetLastError(), "moe down mma launch")
        PULSAR_MMA_DOWN(PULSAR_QUANT_IQ2_XXS, unpack_iq2_xxs);
        PULSAR_MMA_DOWN(PULSAR_QUANT_IQ2_XS, unpack_iq2_xs);
        PULSAR_MMA_DOWN(PULSAR_QUANT_IQ3_XXS, unpack_iq3_xxs);
        PULSAR_MMA_DOWN(PULSAR_QUANT_Q4_0, unpack_q4_0);
        PULSAR_MMA_DOWN(PULSAR_QUANT_Q8_0, unpack_q8_0);
        PULSAR_MMA_DOWN(PULSAR_QUANT_IQ4_XS, unpack_iq4_xs);
        PULSAR_MMA_DOWN(PULSAR_QUANT_Q5_1, unpack_q5_1);
        PULSAR_MMA_DOWN(PULSAR_QUANT_Q2_K, unpack_q2_K);
        PULSAR_MMA_DOWN(PULSAR_QUANT_Q3_K, unpack_q3_K);
        PULSAR_MMA_DOWN(PULSAR_QUANT_Q4_K, unpack_q4_K);
        PULSAR_MMA_DOWN(PULSAR_QUANT_Q5_K, unpack_q5_K);
        PULSAR_MMA_DOWN(PULSAR_QUANT_Q6_K, unpack_q6_K);
#undef PULSAR_MMA_DOWN
        default: break;
        }
    }
    dim3 block(32, 4, 1);
    dim3 grid((out_dim + 3u) / 4u, n_group, 1);
    const uint32_t shmem = (uint32_t)row_bytes * 4u;
    PULSAR_GROUPED_DISPATCH(moe_down_grouped_kernel,
            (float *)partial_dev, (const pulsar_expert_ptrs *)gptrs_dev,
            (const uint32_t *)starts_dev, (const uint32_t *)pairs_dev,
            (const block_q8_K *)midq_dev, mid_blocks, out_dim, n_used,
            row_bytes);
    return cuda_ok(cudaGetLastError(), "moe down grouped launch");
}

extern "C" int pulsar_moe_slot_sum(
        void *out_dev, const void *partial_dev, uint32_t out_dim,
        uint32_t n_used, uint32_t n_tok) {
    if (out_dim == 0 || n_used == 0 || n_tok == 0) return 0;
    const uint64_t total = (uint64_t)n_tok * out_dim;
    moe_slot_sum_kernel<<<(uint32_t)((total + 255u) / 256u), 256>>>(
            (float *)out_dev, (const float *)partial_dev, out_dim, n_used, n_tok);
    return cuda_ok(cudaGetLastError(), "moe slot sum launch");
}

/* Dense matmul over a K-quant weight matrix vs q8_K activations - the
 * lm-head of K-quant ggufs (AngelSlim Q4_K_M keeps output.weight q6_K).
 * Same warp-per-row shape as moe_down, single weight matrix. */
template <typename DOT>
__global__ static void matmul_kq_kernel(
        float *out,           /* [n_tok][out_dim] */
        const char *w,        /* [out_dim] rows of row_bytes */
        const block_q8_K *xq, /* [n_tok][in_blocks] */
        uint32_t in_blocks,
        uint32_t out_dim,
        uint32_t n_tok,
        uint64_t row_bytes) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t token = blockIdx.y;
    if (row >= out_dim || token >= n_tok) return;
    const char *wr = w + (uint64_t)row * row_bytes;
    const block_q8_K *txq = xq + (uint64_t)token * in_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < in_blocks; b += 32u) {
        acc += DOT::block(wr, txq, b);
    }
    #pragma unroll
    for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
        acc += __shfl_xor_sync(0xffffffffu, acc, mask);
    }
    if (lane == 0) out[(uint64_t)token * out_dim + row] = acc;
}

/* Token-tiled variant: one warp owns a row for ALL TT tokens, so the
 * row bytes are read once (L1-hot across the register token loop)
 * instead of once per token - on a 248k-row lm head the legacy grid
 * re-reads ~417MB per token (DFlash verify/draft: 6.7GB per call). */
template <typename DOT, int TT>
__global__ static void matmul_kq_tokens_kernel(
        float *out,
        const char *w,
        const block_q8_K *xq,
        uint32_t in_blocks,
        uint32_t out_dim,
        uint64_t row_bytes) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= out_dim) return;
    const char *wr = w + (uint64_t)row * row_bytes;
    float acc[TT];
    #pragma unroll
    for (int t = 0; t < TT; t++) acc[t] = 0.0f;
    for (uint32_t b = lane; b < in_blocks; b += 32u) {
        #pragma unroll
        for (int t = 0; t < TT; t++) {
            acc[t] += DOT::block(wr, xq + (uint64_t)t * in_blocks, b);
        }
    }
    #pragma unroll
    for (int t = 0; t < TT; t++) {
        float a = acc[t];
        #pragma unroll
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
            a += __shfl_xor_sync(0xffffffffu, a, mask);
        }
        if (lane == 0) out[(uint64_t)t * out_dim + row] = a;
    }
}

extern "C" int pulsar_matmul_kq(
        void *out_dev,
        const void *w_dev,
        const void *xq_dev,
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t n_tok,
        uint64_t row_bytes,
        uint32_t quant) {
    if (in_dim == 0 || in_dim % PULSAR_QK_K != 0 || out_dim == 0 || n_tok == 0 || row_bytes == 0) {
        return 0;
    }
    const uint32_t in_blocks = in_dim / PULSAR_QK_K;
    dim3 block(32, 4, 1);
    /* verify/draft-sized batches take the token-tiled kernel: one row
     * read serves all 16 tokens (identical math, per-token order) */
    if (n_tok == 16u) {
        dim3 tgrid((out_dim + 3u) / 4u, 1, 1);
        switch (quant) {
#define PULSAR_KQ16(Q, DOT)                                                    \
        case Q:                                                                \
            matmul_kq_tokens_kernel<DOT, 16><<<tgrid, block>>>(                \
                    (float *)out_dev, (const char *)w_dev,                     \
                    (const block_q8_K *)xq_dev, in_blocks, out_dim, row_bytes); \
            return cuda_ok(cudaGetLastError(), "matmul_kq16 launch")
        PULSAR_KQ16(PULSAR_QUANT_Q2_K, dot_q2_K);
        PULSAR_KQ16(PULSAR_QUANT_IQ2_XXS, dot_iq2_xxs);
        PULSAR_KQ16(PULSAR_QUANT_Q4_K, dot_q4_K);
        PULSAR_KQ16(PULSAR_QUANT_Q5_K, dot_q5_K);
        PULSAR_KQ16(PULSAR_QUANT_Q6_K, dot_q6_K);
        PULSAR_KQ16(PULSAR_QUANT_Q3_K, dot_q3_K);
#undef PULSAR_KQ16
        default: return 0;
        }
    }
    dim3 grid((out_dim + 3u) / 4u, n_tok, 1);
    switch (quant) {
    case PULSAR_QUANT_Q2_K:
        matmul_kq_kernel<dot_q2_K><<<grid, block>>>((float *)out_dev, (const char *)w_dev, (const block_q8_K *)xq_dev, in_blocks, out_dim, n_tok, row_bytes);
        break;
    case PULSAR_QUANT_IQ2_XXS:
        matmul_kq_kernel<dot_iq2_xxs><<<grid, block>>>((float *)out_dev, (const char *)w_dev, (const block_q8_K *)xq_dev, in_blocks, out_dim, n_tok, row_bytes);
        break;
    case PULSAR_QUANT_Q4_K:
        matmul_kq_kernel<dot_q4_K><<<grid, block>>>((float *)out_dev, (const char *)w_dev, (const block_q8_K *)xq_dev, in_blocks, out_dim, n_tok, row_bytes);
        break;
    case PULSAR_QUANT_Q5_K:
        matmul_kq_kernel<dot_q5_K><<<grid, block>>>((float *)out_dev, (const char *)w_dev, (const block_q8_K *)xq_dev, in_blocks, out_dim, n_tok, row_bytes);
        break;
    case PULSAR_QUANT_Q6_K:
        matmul_kq_kernel<dot_q6_K><<<grid, block>>>((float *)out_dev, (const char *)w_dev, (const block_q8_K *)xq_dev, in_blocks, out_dim, n_tok, row_bytes);
        break;
    case PULSAR_QUANT_Q3_K:
        matmul_kq_kernel<dot_q3_K><<<grid, block>>>((float *)out_dev, (const char *)w_dev, (const block_q8_K *)xq_dev, in_blocks, out_dim, n_tok, row_bytes);
        break;
    default:
        return 0;
    }
    return cuda_ok(cudaGetLastError(), "matmul_kq launch");
}

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
        uint32_t quant,
        uint32_t act_op) {
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
                in_blocks, mid_dim, n_used, n_tok, row_bytes, act_op);
        break;
    case PULSAR_QUANT_IQ2_XXS:
        moe_pair_swiglu_kernel<dot_iq2_xxs><<<grid, block>>>(
                (float *)mid_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const float *)weights_dev, (const block_q8_K *)xq_dev,
                in_blocks, mid_dim, n_used, n_tok, row_bytes, act_op);
        break;
    case PULSAR_QUANT_Q4_K:
        moe_pair_swiglu_kernel<dot_q4_K><<<grid, block>>>(
                (float *)mid_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const float *)weights_dev, (const block_q8_K *)xq_dev,
                in_blocks, mid_dim, n_used, n_tok, row_bytes, act_op);
        break;
    case PULSAR_QUANT_Q5_K:
        moe_pair_swiglu_kernel<dot_q5_K><<<grid, block>>>(
                (float *)mid_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const float *)weights_dev, (const block_q8_K *)xq_dev,
                in_blocks, mid_dim, n_used, n_tok, row_bytes, act_op);
        break;
    case PULSAR_QUANT_Q6_K:
        moe_pair_swiglu_kernel<dot_q6_K><<<grid, block>>>(
                (float *)mid_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const float *)weights_dev, (const block_q8_K *)xq_dev,
                in_blocks, mid_dim, n_used, n_tok, row_bytes, act_op);
        break;
    case PULSAR_QUANT_Q3_K:
        moe_pair_swiglu_kernel<dot_q3_K><<<grid, block>>>(
                (float *)mid_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const float *)weights_dev, (const block_q8_K *)xq_dev,
                in_blocks, mid_dim, n_used, n_tok, row_bytes, act_op);
        break;
    case PULSAR_QUANT_IQ2_XS:
        moe_pair_swiglu_kernel<dot_iq2_xs><<<grid, block>>>(
                (float *)mid_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const float *)weights_dev, (const block_q8_K *)xq_dev,
                in_blocks, mid_dim, n_used, n_tok, row_bytes, act_op);
        break;
    case PULSAR_QUANT_IQ3_XXS:
        moe_pair_swiglu_kernel<dot_iq3_xxs><<<grid, block>>>(
                (float *)mid_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const float *)weights_dev, (const block_q8_K *)xq_dev,
                in_blocks, mid_dim, n_used, n_tok, row_bytes, act_op);
        break;
    case PULSAR_QUANT_Q4_0:
        moe_pair_swiglu_kernel<dot_q4_0><<<grid, block>>>(
                (float *)mid_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const float *)weights_dev, (const block_q8_K *)xq_dev,
                in_blocks, mid_dim, n_used, n_tok, row_bytes, act_op);
        break;
    case PULSAR_QUANT_Q5_1:
        moe_pair_swiglu_kernel<dot_q5_1><<<grid, block>>>(
                (float *)mid_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const float *)weights_dev, (const block_q8_K *)xq_dev,
                in_blocks, mid_dim, n_used, n_tok, row_bytes, act_op);
        break;
    case PULSAR_QUANT_Q8_0:
        moe_pair_swiglu_kernel<dot_q8_0><<<grid, block>>>(
                (float *)mid_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const float *)weights_dev, (const block_q8_K *)xq_dev,
                in_blocks, mid_dim, n_used, n_tok, row_bytes, act_op);
        break;
    case PULSAR_QUANT_IQ4_XS:
        moe_pair_swiglu_kernel<dot_iq4_xs><<<grid, block>>>(
                (float *)mid_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const float *)weights_dev, (const block_q8_K *)xq_dev,
                in_blocks, mid_dim, n_used, n_tok, row_bytes, act_op);
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
    if (mid_dim == 0 || out_dim == 0 ||
        n_used == 0 || n_tok == 0 || row_bytes == 0) {
        return 0;
    }
    /* ceil: a partial tail superblock rides zero-quantized activations
     * (see q8_K_quantize_kernel); weight reads past the row end need
     * PULSAR_SLAB_SLACK bytes of slack after each expert slab */
    const uint32_t mid_blocks = (mid_dim + PULSAR_QK_K - 1u) / PULSAR_QK_K;
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
    case PULSAR_QUANT_Q4_K:
        moe_down_kernel<dot_q4_K><<<grid, block>>>(
                (float *)out_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const block_q8_K *)midq_dev, mid_blocks, out_dim, n_used,
                n_tok, row_bytes);
        break;
    case PULSAR_QUANT_Q5_K:
        moe_down_kernel<dot_q5_K><<<grid, block>>>(
                (float *)out_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const block_q8_K *)midq_dev, mid_blocks, out_dim, n_used,
                n_tok, row_bytes);
        break;
    case PULSAR_QUANT_Q6_K:
        moe_down_kernel<dot_q6_K><<<grid, block>>>(
                (float *)out_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const block_q8_K *)midq_dev, mid_blocks, out_dim, n_used,
                n_tok, row_bytes);
        break;
    case PULSAR_QUANT_Q3_K:
        moe_down_kernel<dot_q3_K><<<grid, block>>>(
                (float *)out_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const block_q8_K *)midq_dev, mid_blocks, out_dim, n_used,
                n_tok, row_bytes);
        break;
    case PULSAR_QUANT_IQ2_XS:
        moe_down_kernel<dot_iq2_xs><<<grid, block>>>(
                (float *)out_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const block_q8_K *)midq_dev, mid_blocks, out_dim, n_used,
                n_tok, row_bytes);
        break;
    case PULSAR_QUANT_IQ3_XXS:
        moe_down_kernel<dot_iq3_xxs><<<grid, block>>>(
                (float *)out_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const block_q8_K *)midq_dev, mid_blocks, out_dim, n_used,
                n_tok, row_bytes);
        break;
    case PULSAR_QUANT_Q4_0:
        moe_down_kernel<dot_q4_0><<<grid, block>>>(
                (float *)out_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const block_q8_K *)midq_dev, mid_blocks, out_dim, n_used,
                n_tok, row_bytes);
        break;
    case PULSAR_QUANT_Q5_1:
        moe_down_kernel<dot_q5_1><<<grid, block>>>(
                (float *)out_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const block_q8_K *)midq_dev, mid_blocks, out_dim, n_used,
                n_tok, row_bytes);
        break;
    case PULSAR_QUANT_Q8_0:
        moe_down_kernel<dot_q8_0><<<grid, block>>>(
                (float *)out_dev, (const pulsar_expert_ptrs *)ptrs_dev,
                (const block_q8_K *)midq_dev, mid_blocks, out_dim, n_used,
                n_tok, row_bytes);
        break;
    case PULSAR_QUANT_IQ4_XS:
        moe_down_kernel<dot_iq4_xs><<<grid, block>>>(
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
    const uint32_t n_blocks = (in_dim + PULSAR_QK_K - 1) / PULSAR_QK_K;
    for (uint32_t row = 0; row < n_rows; row++) {
        for (uint32_t b = 0; b < n_blocks; b++) {
            const uint32_t bn = in_dim - b * PULSAR_QK_K < PULSAR_QK_K
                    ? in_dim - b * PULSAR_QK_K : PULSAR_QK_K;
            const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * PULSAR_QK_K;
            block_q8_K *yb = out + (uint64_t)row * n_blocks + b;
            float amax = 0.0f, maxv = 0.0f;
            for (uint32_t i = 0; i < bn; i++) {
                const float a = fabsf(xr[i]);
                if (a > amax) { amax = a; maxv = xr[i]; }
            }
            if (amax == 0.0f) {
                memset(yb, 0, sizeof(*yb));
                continue;
            }
            const float iscale = -127.0f / maxv;
            for (uint32_t i = 0; i < PULSAR_QK_K; i++) {
                int qv = i < bn ? (int)lrintf(iscale * xr[i]) : 0;
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

/* host mirrors of the K-quant device dots: identical integer accumulation
 * order, scalar instead of dp4a */
static float host_dot_iq2_xs_block(const char *row, const block_q8_K *xq, uint32_t bi) {
    const block_iq2_xs *x = (const block_iq2_xs *)row + bi;
    const block_q8_K *y = xq + bi;
    const int8_t *q8 = y->qs;
    uint64_t grid_host[512];
    uint8_t signs_host[128];
    cudaMemcpyFromSymbol(grid_host, cuda_iq2xs_grid, sizeof(grid_host));
    cudaMemcpyFromSymbol(signs_host, cuda_ksigns_iq2xs, sizeof(signs_host));
    float sumf = 0.0f;
    for (int g = 0; g < 8; g++) {
        const int ls1 = 2 * (x->scales[g] & 0x0f) + 1;
        const int ls2 = 2 * (x->scales[g] >> 4) + 1;
        int s1 = 0, s2 = 0;
        for (int j = 0; j < 4; j++) {
            const uint16_t q = x->qs[g * 4 + j];
            const uint8_t *gr = (const uint8_t *)&grid_host[q & 511];
            const uint8_t sgn = signs_host[q >> 9];
            int acc = 0;
            for (int i = 0; i < 8; i++) {
                int w = (int8_t)gr[i];
                if (sgn & (1 << i)) w = -w;
                acc += w * (int)q8[g * 32 + j * 8 + i];
            }
            if (j < 2) s1 += acc; else s2 += acc;
        }
        sumf += (float)(ls1 * s1 + ls2 * s2);
    }
    return 0.125f * f16_to_f32_host(x->d) * y->d * sumf;
}

static float host_dot_iq3_xxs_block(const char *row, const block_q8_K *xq, uint32_t bi) {
    const block_iq3_xxs *x = (const block_iq3_xxs *)row + bi;
    const block_q8_K *y = xq + bi;
    const int8_t *q8 = y->qs;
    uint32_t grid_host[256];
    uint8_t signs_host[128];
    cudaMemcpyFromSymbol(grid_host, cuda_iq3xxs_grid, sizeof(grid_host));
    cudaMemcpyFromSymbol(signs_host, cuda_ksigns_iq2xs, sizeof(signs_host));
    float sumf = 0.0f;
    for (int g = 0; g < 8; g++) {
        uint32_t aux;
        memcpy(&aux, x->qs + 64 + 4 * g, 4);
        const float db = f16_to_f32_host(x->d) * (0.5f + (float)(aux >> 28)) * 0.5f;
        int sumi = 0;
        for (int j = 0; j < 4; j++) {
            const uint8_t sgn = signs_host[(aux >> (7 * j)) & 127];
            const uint8_t *g0 = (const uint8_t *)&grid_host[x->qs[g * 8 + j * 2]];
            const uint8_t *g1 = (const uint8_t *)&grid_host[x->qs[g * 8 + j * 2 + 1]];
            for (int i = 0; i < 4; i++) {
                int w = (int8_t)g0[i];
                if (sgn & (1 << i)) w = -w;
                sumi += w * (int)q8[g * 32 + j * 8 + i];
            }
            for (int i = 0; i < 4; i++) {
                int w = (int8_t)g1[i];
                if (sgn & (1 << (4 + i))) w = -w;
                sumi += w * (int)q8[g * 32 + j * 8 + 4 + i];
            }
        }
        sumf += db * (float)sumi;
    }
    return y->d * sumf;
}

static float host_dot_q4_0_block(const char *row, const block_q8_K *xq, uint32_t bi) {
    const block_q4_0 *xb = (const block_q4_0 *)(row + (uint64_t)bi * 8u * sizeof(block_q4_0));
    const block_q8_K *y = xq + bi;
    const int8_t *q8 = y->qs;
    float sumf = 0.0f;
    for (int b = 0; b < 8; b++) {
        const block_q4_0 *x = xb + b;
        int sumi = 0;
        for (int i = 0; i < 16; i++) {
            sumi += (int)(x->qs[i] & 0x0f) * (int)q8[b * 32 + i];
            sumi += (int)(x->qs[i] >> 4) * (int)q8[b * 32 + 16 + i];
        }
        const int bsum = y->bsums[2 * b] + y->bsums[2 * b + 1];
        if (sumi != 0 || bsum != 0) {
            sumf += f16_to_f32_host(x->d) * (float)(sumi - 8 * bsum);
        }
    }
    return y->d * sumf;
}

static float host_dot_q8_0_block(const char *row, const block_q8_K *xq, uint32_t bi) {
    const q8_0_block *xb = (const q8_0_block *)(row + (uint64_t)bi * 8u * sizeof(q8_0_block));
    const block_q8_K *y = xq + bi;
    const int8_t *q8 = y->qs;
    float sumf = 0.0f;
    for (int b = 0; b < 8; b++) {
        const q8_0_block *x = xb + b;
        int sumi = 0;
        for (int i = 0; i < 32; i++) sumi += (int)x->q[i] * (int)q8[b * 32 + i];
        if (sumi != 0) sumf += f16_to_f32_host(x->scale_f16) * (float)sumi;
    }
    return y->d * sumf;
}

static float host_dot_q3_K_block(const char *row, const block_q8_K *xq, uint32_t bi) {
    const block_q3_K *x = (const block_q3_K *)row + bi;
    const block_q8_K *y = xq + bi;
    int8_t sc[16];
    k3_unpack_scales(x->scales, sc);
    const uint8_t *q3 = x->qs;
    const uint8_t *hm = x->hmask;
    const int8_t *q8 = y->qs;
    int isum = 0;
    uint32_t hbit = 1u;
    int is = 0;
    for (int k = 0; k < 2; k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            for (int half = 0; half < 2; half++) {
                int s16 = 0;
                for (int i = 0; i < 16; i++) {
                    const int l = half * 16 + i;
                    int q = (q3[l] >> shift) & 3;
                    if ((hm[l] & hbit) == 0u) q -= 4;
                    s16 += q * (int)q8[l];
                }
                isum += (int)sc[is++] * s16;
            }
            shift += 2;
            q8 += 32;
            hbit <<= 1u;
        }
        q3 += 32;
    }
    return f16_to_f32_host(x->d) * y->d * (float)isum;
}

static const int8_t kh_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

static float host_dot_iq4_xs_block(const char *row, const block_q8_K *xq, uint32_t bi) {
    const block_iq4_xs *x = (const block_iq4_xs *)row + bi;
    const block_q8_K *y = xq + bi;
    const uint8_t *qs = x->qs;
    const int8_t *q8 = y->qs;
    const uint16_t sh = x->scales_h;
    float sumf = 0.0f;
    int any = 0;
    for (int ib = 0; ib < 8; ib++) {
        const int ls = (int)(((x->scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xf) |
                             (((sh >> (2 * ib)) & 3) << 4)) - 32;
        int sumi = 0;
        for (int j = 0; j < 16; j++) {
            const uint8_t b = qs[j];
            sumi += kh_iq4nl[b & 0xf] * (int)q8[j];
            sumi += kh_iq4nl[b >> 4] * (int)q8[j + 16];
        }
        if (sumi != 0) { sumf += (float)(ls * sumi); any = 1; }
        qs += 16;
        q8 += 32;
    }
    return any ? f16_to_f32_host(x->d) * y->d * sumf : 0.0f;
}

static float host_dot_q4_K_block(const char *row, const block_q8_K *xq, uint32_t bi) {
    const block_q4_K *x = (const block_q4_K *)row + bi;
    const block_q8_K *y = xq + bi;
    const uint8_t *q4 = x->qs;
    const int8_t *q8 = y->qs;
    int isum = 0, msum = 0;
    for (int j = 0; j < 4; j++) {
        uint8_t sc1, m1, sc2, m2;
        k4_scale_min(2 * j, x->scales, &sc1, &m1);
        k4_scale_min(2 * j + 1, x->scales, &sc2, &m2);
        int s1 = 0, s2 = 0;
        for (int i = 0; i < 32; i++) {
            s1 += (int)(q4[i] & 0x0f) * (int)q8[i];
            s2 += (int)(q4[i] >> 4) * (int)q8[32 + i];
        }
        isum += (int)sc1 * s1 + (int)sc2 * s2;
        msum += (int)m1 * (y->bsums[4 * j] + y->bsums[4 * j + 1]) +
                (int)m2 * (y->bsums[4 * j + 2] + y->bsums[4 * j + 3]);
        q4 += 32;
        q8 += 64;
    }
    return f16_to_f32_host(x->d) * y->d * (float)isum -
           f16_to_f32_host(x->dmin) * y->d * (float)msum;
}

static float host_dot_q5_K_block(const char *row, const block_q8_K *xq, uint32_t bi) {
    const block_q5_K *x = (const block_q5_K *)row + bi;
    const block_q8_K *y = xq + bi;
    const uint8_t *q5 = x->qs;
    const uint8_t *qh = x->qh;
    const int8_t *q8 = y->qs;
    int isum = 0, msum = 0;
    for (int j = 0; j < 4; j++) {
        uint8_t sc1, m1, sc2, m2;
        k4_scale_min(2 * j, x->scales, &sc1, &m1);
        k4_scale_min(2 * j + 1, x->scales, &sc2, &m2);
        int s1 = 0, s2 = 0;
        for (int i = 0; i < 32; i++) {
            const int h1 = (qh[i] >> (2 * j)) & 1;
            const int h2 = (qh[i] >> (2 * j + 1)) & 1;
            s1 += ((int)(q5[i] & 0x0f) | (h1 << 4)) * (int)q8[i];
            s2 += ((int)(q5[i] >> 4) | (h2 << 4)) * (int)q8[32 + i];
        }
        isum += (int)sc1 * s1 + (int)sc2 * s2;
        msum += (int)m1 * (y->bsums[4 * j] + y->bsums[4 * j + 1]) +
                (int)m2 * (y->bsums[4 * j + 2] + y->bsums[4 * j + 3]);
        q5 += 32;
        q8 += 64;
    }
    return f16_to_f32_host(x->d) * y->d * (float)isum -
           f16_to_f32_host(x->dmin) * y->d * (float)msum;
}

static float host_dot_q6_K_block(const char *row, const block_q8_K *xq, uint32_t bi) {
    const block_q6_K *x = (const block_q6_K *)row + bi;
    const block_q8_K *y = xq + bi;
    const uint8_t *ql = x->ql;
    const uint8_t *qh = x->qh;
    const int8_t *sc = x->scales;
    const int8_t *q8 = y->qs;
    int isum = 0;
    for (int j = 0; j < 2; j++) {
        int g[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < 32; i++) {
            const int sub = i >> 4;
            const int v0 = ((int)(ql[i] & 0x0f) | (((qh[i] >> 0) & 3) << 4)) - 32;
            const int v1 = ((int)(ql[32 + i] & 0x0f) | (((qh[i] >> 2) & 3) << 4)) - 32;
            const int v2 = ((int)(ql[i] >> 4) | (((qh[i] >> 4) & 3) << 4)) - 32;
            const int v3 = ((int)(ql[32 + i] >> 4) | (((qh[i] >> 6) & 3) << 4)) - 32;
            g[0 + sub] += v0 * (int)q8[i];
            g[2 + sub] += v1 * (int)q8[32 + i];
            g[4 + sub] += v2 * (int)q8[64 + i];
            g[6 + sub] += v3 * (int)q8[96 + i];
        }
        for (int k = 0; k < 8; k++) isum += (int)sc[k] * g[k];
        sc += 8;
        ql += 64;
        qh += 32;
        q8 += 128;
    }
    return f16_to_f32_host(x->d) * y->d * (float)isum;
}

static float host_dot_q5_1_block(const char *row, const block_q8_K *xq, uint32_t bi) {
    const block_q5_1 *xb = (const block_q5_1 *)(row + (uint64_t)bi * 8u * sizeof(block_q5_1));
    const block_q8_K *y = xq + bi;
    const int8_t *q8 = y->qs;
    float sumf = 0.0f;
    for (int b = 0; b < 8; b++) {
        const block_q5_1 *x = xb + b;
        uint32_t qh;
        memcpy(&qh, &x->qh, 4);
        int sumi = 0;
        for (int i = 0; i < 16; i++) {
            const int lo = (int)(x->qs[i] & 0x0f) | (int)(((qh >> i) & 1u) << 4);
            const int hi = (int)(x->qs[i] >> 4) | (int)(((qh >> (i + 16)) & 1u) << 4);
            sumi += lo * (int)q8[b * 32 + i];
            sumi += hi * (int)q8[b * 32 + 16 + i];
        }
        const int bsum = y->bsums[2 * b] + y->bsums[2 * b + 1];
        if (sumi != 0 || bsum != 0) {
            sumf += f16_to_f32_host(x->d) * (float)sumi + f16_to_f32_host(x->m) * (float)bsum;
        }
    }
    return y->d * sumf;
}

static void fill_slab(char *slab, uint32_t n_rows, uint32_t n_el,
                      uint64_t row_bytes, uint32_t quant) {
    for (uint32_t r = 0; r < n_rows; r++) {
        char *row = slab + (uint64_t)r * row_bytes;
        for (uint64_t b = 0; b < row_bytes; b++) row[b] = (char)test_randbyte();
        /* overwrite scale halves with sane small values (random f16 bits
         * can be inf/nan) */
        for (uint32_t blk = 0; blk < (n_el + PULSAR_QK_K - 1) / PULSAR_QK_K; blk++) {
            const uint16_t dv = f32_to_f16_bits(fabsf(gqa_test_randf()) * 0.05f + 0.001f);
            const uint16_t dm = f32_to_f16_bits(fabsf(gqa_test_randf()) * 0.05f);
            switch (quant) {
            case PULSAR_QUANT_Q2_K: {
                block_q2_K *q = (block_q2_K *)row + blk;
                q->d = dv;
                q->dmin = dm;
                break;
            }
            case PULSAR_QUANT_IQ2_XS: {
                block_iq2_xs *q = (block_iq2_xs *)row + blk;
                q->d = f32_to_f16_bits(fabsf(gqa_test_randf()) * 0.1f + 0.001f);
                break;
            }
            case PULSAR_QUANT_IQ3_XXS: {
                block_iq3_xxs *q = (block_iq3_xxs *)row + blk;
                q->d = f32_to_f16_bits(fabsf(gqa_test_randf()) * 0.1f + 0.001f);
                break;
            }
            case PULSAR_QUANT_Q4_0: {
                if (blk == 0) {
                    block_q4_0 *q = (block_q4_0 *)row;
                    for (uint32_t k = 0; k < n_el / 32u; k++)
                        q[k].d = f32_to_f16_bits(fabsf(gqa_test_randf()) * 0.05f + 0.001f);
                }
                break;
            }
            case PULSAR_QUANT_Q5_1: {
                if (blk == 0) {
                    block_q5_1 *q = (block_q5_1 *)row;
                    for (uint32_t k = 0; k < n_el / 32u; k++) {
                        q[k].d = f32_to_f16_bits(fabsf(gqa_test_randf()) * 0.05f + 0.001f);
                        q[k].m = f32_to_f16_bits(gqa_test_randf() * 0.05f);
                    }
                }
                break;
            }
            case PULSAR_QUANT_Q8_0: {
                if (blk == 0) {
                    q8_0_block *q = (q8_0_block *)row;
                    for (uint32_t k = 0; k < n_el / 32u; k++)
                        q[k].scale_f16 = f32_to_f16_bits(fabsf(gqa_test_randf()) * 0.05f + 0.001f);
                }
                break;
            }
            case PULSAR_QUANT_Q3_K: {
                block_q3_K *q = (block_q3_K *)row + blk;
                q->d = dv;
                break;
            }
            case PULSAR_QUANT_Q4_K: {
                block_q4_K *q = (block_q4_K *)row + blk;
                q->d = dv;
                q->dmin = dm;
                break;
            }
            case PULSAR_QUANT_Q5_K: {
                block_q5_K *q = (block_q5_K *)row + blk;
                q->d = dv;
                q->dmin = dm;
                break;
            }
            case PULSAR_QUANT_Q6_K: {
                block_q6_K *q = (block_q6_K *)row + blk;
                q->d = dv;
                break;
            }
            case PULSAR_QUANT_IQ4_XS: {
                block_iq4_xs *q = (block_iq4_xs *)row + blk;
                q->d = dv;
                break;
            }
            default: {
                block_iq2_xxs *q = (block_iq2_xxs *)row + blk;
                q->d = f32_to_f16_bits(fabsf(gqa_test_randf()) * 0.1f + 0.001f);
                break;
            }
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

static int moe_selftest_one2(uint32_t quant, const char *name, uint32_t mid_dim) {
    const uint32_t in_dim = 512, out_dim = 320;
    const uint32_t n_expert = 8, n_used = 4, n_tok = 3;
    const uint32_t in_blocks = in_dim / PULSAR_QK_K;
    const uint32_t mid_blocks = (mid_dim + PULSAR_QK_K - 1) / PULSAR_QK_K;
    uint64_t block_bytes;
    float (*dot)(const char *, const block_q8_K *, uint32_t);
    switch (quant) {
    case PULSAR_QUANT_Q2_K:   block_bytes = sizeof(block_q2_K);    dot = host_dot_q2_K_block;    break;
    case PULSAR_QUANT_Q3_K:   block_bytes = sizeof(block_q3_K);    dot = host_dot_q3_K_block;    break;
    case PULSAR_QUANT_IQ2_XS: block_bytes = sizeof(block_iq2_xs);  dot = host_dot_iq2_xs_block;  break;
    case PULSAR_QUANT_IQ3_XXS: block_bytes = sizeof(block_iq3_xxs); dot = host_dot_iq3_xxs_block; break;
    case PULSAR_QUANT_Q4_0:   block_bytes = 8 * sizeof(block_q4_0); dot = host_dot_q4_0_block;   break;
    case PULSAR_QUANT_Q5_1:   block_bytes = 8 * sizeof(block_q5_1); dot = host_dot_q5_1_block;   break;
    case PULSAR_QUANT_Q8_0:   block_bytes = 8 * sizeof(q8_0_block); dot = host_dot_q8_0_block;   break;
    case PULSAR_QUANT_IQ4_XS: block_bytes = sizeof(block_iq4_xs);   dot = host_dot_iq4_xs_block; break;
    case PULSAR_QUANT_Q4_K:   block_bytes = sizeof(block_q4_K);    dot = host_dot_q4_K_block;    break;
    case PULSAR_QUANT_Q5_K:   block_bytes = sizeof(block_q5_K);    dot = host_dot_q5_K_block;    break;
    case PULSAR_QUANT_Q6_K:   block_bytes = sizeof(block_q6_K);    dot = host_dot_q6_K_block;    break;
    default:                  block_bytes = sizeof(block_iq2_xxs); dot = host_dot_iq2_xxs_block; break;
    }
    /* sub-block quants (q4_0/q5_1) pack rows exactly (22 blocks for 704),
     * not per-superblock - mirror the real gguf row layout */
    const int subblock = quant == PULSAR_QUANT_Q4_0 || quant == PULSAR_QUANT_Q5_1 ||
                         quant == PULSAR_QUANT_Q8_0;
    const uint64_t pair_row_bytes = (uint64_t)in_blocks * block_bytes;
    const uint64_t down_row_bytes = subblock
            ? (uint64_t)(mid_dim / 32) * (block_bytes / 8)
            : (uint64_t)mid_blocks * block_bytes;
    const uint64_t gate_slab_bytes = (uint64_t)n_expert * mid_dim * pair_row_bytes;
    const uint64_t down_slab_bytes = (uint64_t)n_expert * out_dim * down_row_bytes;

    /* +256: host reference dots read phantom tail sub-blocks past the last
     * row (up to 68B for q8_0 at 704) - mirror the engine's SLAB_SLACK */
    char *gate = (char *)malloc(gate_slab_bytes + 256);
    char *up = (char *)malloc(gate_slab_bytes + 256);
    char *down = (char *)malloc(down_slab_bytes + 256);
    memset(gate + gate_slab_bytes, 0, 256);
    memset(up + gate_slab_bytes, 0, 256);
    memset(down + down_slab_bytes, 0, 256);
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
    /* +256: the device dots read the same phantom tail as the host ones */
    int ok = cuda_ok(cudaMalloc(&gate_dev, gate_slab_bytes + 256), "gate alloc") &&
             cuda_ok(cudaMalloc(&up_dev, gate_slab_bytes + 256), "up alloc") &&
             cuda_ok(cudaMalloc(&down_dev, down_slab_bytes + 256), "down alloc") &&
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
                                pair_row_bytes, quant, 0) &&
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

    /* grouped-path crosscheck (mma variant on iq2_xxs/sm_80+): CSR of the
     * same routing, compared against the (host-validated) plain output.
     * Tolerance is looser: the mma path reorders float accumulation. */
    if (ok && mid_dim % PULSAR_QK_K == 0) {
        uint32_t gsel[64];
        uint32_t n_group = 0;
        uint32_t starts_h[65];
        uint32_t pairs_h[64];
        pulsar_expert_ptrs gptrs_h[64];
        /* group by expert (matches the engine's pointer-keyed grouping) */
        for (uint32_t si = 0; si < n_tok * n_used; si++) {
            if (sel[si] < 0) continue;
            uint32_t gi = n_group;
            for (uint32_t k2 = 0; k2 < n_group; k2++) {
                if (gsel[k2] == (uint32_t)sel[si]) { gi = k2; break; }
            }
            if (gi == n_group) gsel[n_group++] = (uint32_t)sel[si];
        }
        uint32_t cursor = 0;
        for (uint32_t k2 = 0; k2 < n_group; k2++) {
            starts_h[k2] = cursor;
            const uint32_t e = gsel[k2];
            gptrs_h[k2].gate = (char *)gate_dev + (uint64_t)e * mid_dim * pair_row_bytes;
            gptrs_h[k2].up = (char *)up_dev + (uint64_t)e * mid_dim * pair_row_bytes;
            gptrs_h[k2].down = (char *)down_dev + (uint64_t)e * out_dim * down_row_bytes;
            for (uint32_t si = 0; si < n_tok * n_used; si++) {
                if (sel[si] == (int32_t)e) {
                    pairs_h[cursor++] = ((si / n_used) << 4) | (si % n_used);
                }
            }
        }
        starts_h[n_group] = cursor;
        void *gp_d = NULL, *st_d = NULL, *pr_d = NULL, *partial_d = NULL;
        float *mid2 = (float *)malloc((uint64_t)n_tok * n_used * mid_dim * 4);
        float *out2 = (float *)malloc((uint64_t)n_tok * out_dim * 4);
        const uint64_t partial_bytes = (uint64_t)n_tok * n_used * out_dim * 4;
        ok = cuda_ok(cudaMalloc(&gp_d, sizeof(gptrs_h)), "gp") &&
             cuda_ok(cudaMalloc(&st_d, sizeof(starts_h)), "st") &&
             cuda_ok(cudaMalloc(&pr_d, sizeof(pairs_h)), "pr") &&
             cuda_ok(cudaMalloc(&partial_d, partial_bytes), "partial") &&
             cuda_ok(cudaMemcpy(gp_d, gptrs_h, sizeof(gptrs_h), cudaMemcpyHostToDevice), "gp h2d") &&
             cuda_ok(cudaMemcpy(st_d, starts_h, sizeof(starts_h), cudaMemcpyHostToDevice), "st h2d") &&
             cuda_ok(cudaMemcpy(pr_d, pairs_h, sizeof(pairs_h), cudaMemcpyHostToDevice), "pr h2d") &&
             cuda_ok(cudaMemset(mid_dev, 0, (uint64_t)n_tok * n_used * mid_dim * 4), "mid clear") &&
             cuda_ok(cudaMemset(partial_d, 0, partial_bytes), "partial clear") &&
             pulsar_moe_pair_swiglu_grouped(mid_dev, gp_d, st_d, pr_d, w_dev, xq_dev,
                                            in_dim, mid_dim, n_used, n_group,
                                            pair_row_bytes, quant, 0) &&
             pulsar_moe_down_grouped(partial_d, gp_d, st_d, pr_d, midq_dev,
                                     mid_dim, out_dim, n_used, n_group,
                                     down_row_bytes, quant) &&
             pulsar_moe_slot_sum(out_dev, partial_d, out_dim, n_used, n_tok) &&
             cuda_ok(cudaDeviceSynchronize(), "gsync") &&
             cuda_ok(cudaMemcpy(mid2, mid_dev, (uint64_t)n_tok * n_used * mid_dim * 4, cudaMemcpyDeviceToHost), "mid2 d2h") &&
             cuda_ok(cudaMemcpy(out2, out_dev, (uint64_t)n_tok * out_dim * 4, cudaMemcpyDeviceToHost), "out2 d2h");
        float gmaxd = 0.0f;
        if (ok) {
            for (uint64_t i = 0; i < (uint64_t)n_tok * n_used * mid_dim; i++) {
                /* NULL-slot rows are zero in grouped, nonzero-garbage-free
                 * in plain (plain writes 0 too) - direct compare ok */
                const float d = fabsf(mid2[i] - mid_gpu[i]);
                if (d > gmaxd) gmaxd = d;
            }
            for (uint64_t i = 0; i < (uint64_t)n_tok * out_dim; i++) {
                const float d = fabsf(out2[i] - out_gpu[i]);
                if (d > gmaxd) gmaxd = d;
            }
            ok = gmaxd <= 5e-3f * (out_maxref > 1.0f ? out_maxref : 1.0f);
            fprintf(stderr, "moe-selftest %s grouped/mma crosscheck: %s (max diff vs plain %.2e)\n",
                    name, ok ? "PASS" : "FAIL", (double)gmaxd);
        }
        if (gp_d) cudaFree(gp_d);
        if (st_d) cudaFree(st_d);
        if (pr_d) cudaFree(pr_d);
        if (partial_d) cudaFree(partial_d);
        free(mid2); free(out2);
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

static int moe_selftest_one(uint32_t quant, const char *name) {
    return moe_selftest_one2(quant, name, 256);
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
           moe_selftest_one(PULSAR_QUANT_IQ2_XXS, "iq2_xxs") &&
           moe_selftest_one(PULSAR_QUANT_Q4_K, "q4_K") &&
           moe_selftest_one(PULSAR_QUANT_Q5_K, "q5_K") &&
           moe_selftest_one(PULSAR_QUANT_Q6_K, "q6_K") &&
           moe_selftest_one(PULSAR_QUANT_Q3_K, "q3_K") &&
           moe_selftest_one(PULSAR_QUANT_IQ2_XS, "iq2_xs") &&
           moe_selftest_one(PULSAR_QUANT_IQ3_XXS, "iq3_xxs") &&
           moe_selftest_one(PULSAR_QUANT_Q4_0, "q4_0") &&
           moe_selftest_one(PULSAR_QUANT_Q5_1, "q5_1") &&
           moe_selftest_one(PULSAR_QUANT_Q8_0, "q8_0") &&
           moe_selftest_one(PULSAR_QUANT_IQ4_XS, "iq4_xs") &&
           /* gemma4 shape: 704-wide experts, ceil-superblock tail */
           moe_selftest_one2(PULSAR_QUANT_Q5_1, "q5_1-704", 704) &&
           moe_selftest_one2(PULSAR_QUANT_Q8_0, "q8_0-704", 704) &&
           moe_selftest_one2(PULSAR_QUANT_IQ4_XS, "iq4_xs-704", 704);
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
        uint32_t n, float clamp, float weight, uint32_t act_op) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float u = up[i];
    if (clamp > 1.0e-6f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    out[i] = pulsar_glu(g, u, act_op) * weight;
}

extern "C" int pulsar_swiglu(
        void *out_dev, const void *gate_dev, const void *up_dev,
        uint32_t n, float clamp, float weight, uint32_t act_op) {
    if (n == 0) return 0;
    swiglu_kernel<<<(n + 255u) / 256u, 256>>>(
            (float *)out_dev, (const float *)gate_dev, (const float *)up_dev,
            n, clamp, weight, act_op);
    return cuda_ok(cudaGetLastError(), "swiglu launch");
}

/* ---- Inkling shortconv: y = x + causal depthwise conv1d over the last K
 * inputs. x: [n_tok][w] token-major; kern: [w][K] (gguf blk.*.shortconv_*
 * ne = [K, w], taps contiguous per channel, tap K-1 = current token);
 * state: [w][K-1] rolling window of the K-1 inputs BEFORE this chunk
 * (channel-major). out must not alias x. Chunked prefill carries state
 * across calls; the engine zeroes it at pos 0. */
__global__ static void sconv_kernel(
        float *out, const float *x, const float *kern, const float *state,
        uint32_t n_tok, uint32_t w, uint32_t K) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)n_tok * w) return;
    const uint32_t c = (uint32_t)(i % w);
    const uint32_t t = (uint32_t)(i / w);
    const uint32_t d = K - 1u;
    const float *kc = kern + (uint64_t)c * K;
    float acc = 0.0f;
    for (uint32_t j = 0; j < K; j++) {
        const int32_t src = (int32_t)(t + j) - (int32_t)d;
        const float v = src >= 0
                ? x[(uint64_t)src * w + c]
                : state[(uint64_t)c * d + (uint32_t)(src + (int32_t)d)];
        acc += kc[j] * v;
    }
    out[i] = x[i] + acc;
}

/* Roll the state window forward by n_tok columns: new state = last K-1
 * inputs of (old state ++ x). One thread per channel owns all K-1 columns
 * in registers, so the in-place shift has no cross-thread hazard. */
__global__ static void sconv_state_kernel(
        float *state, const float *x, uint32_t n_tok, uint32_t w, uint32_t K) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= w) return;
    const uint32_t d = K - 1u;
    float nv[7]; /* K <= 8 */
    for (uint32_t j = 0; j < d; j++) {
        const int32_t src = (int32_t)(n_tok + j) - (int32_t)d;
        nv[j] = src >= 0
                ? x[(uint64_t)src * w + c]
                : state[(uint64_t)c * d + (uint32_t)(src + (int32_t)d)];
    }
    for (uint32_t j = 0; j < d; j++) state[(uint64_t)c * d + j] = nv[j];
}

extern "C" int pulsar_sconv(
        void *out_dev, const void *x_dev, const void *kern_dev,
        void *state_dev, uint32_t n_tok, uint32_t w, uint32_t K) {
    if (n_tok == 0 || w == 0 || K < 2u || K > 8u || out_dev == x_dev) return 0;
    const uint64_t total = (uint64_t)n_tok * w;
    sconv_kernel<<<(uint32_t)((total + 255u) / 256u), 256>>>(
            (float *)out_dev, (const float *)x_dev, (const float *)kern_dev,
            (const float *)state_dev, n_tok, w, K);
    if (!cuda_ok(cudaGetLastError(), "sconv launch")) return 0;
    sconv_state_kernel<<<(w + 255u) / 256u, 256>>>(
            (float *)state_dev, (const float *)x_dev, n_tok, w, K);
    return cuda_ok(cudaGetLastError(), "sconv state launch");
}

/* Selftest: CPU reference + chunked-vs-single equivalence (state carry). */
extern "C" int pulsar_sconv_selftest(void) {
    const uint32_t n_tok = 11, w = 96, K = 4, d = K - 1;
    float *x = (float *)malloc((uint64_t)n_tok * w * 4);
    float *kern = (float *)malloc((uint64_t)w * K * 4);
    float *ref = (float *)malloc((uint64_t)n_tok * w * 4);
    float *got = (float *)malloc((uint64_t)n_tok * w * 4);
    float *st_ref = (float *)calloc((uint64_t)w * d, 4);
    for (uint64_t i = 0; i < (uint64_t)n_tok * w; i++) x[i] = gqa_test_randf();
    for (uint64_t i = 0; i < (uint64_t)w * K; i++) kern[i] = gqa_test_randf() * 0.5f;
    /* reference: zero history before t=0 */
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t c = 0; c < w; c++) {
            float acc = 0.0f;
            for (uint32_t j = 0; j < K; j++) {
                const int32_t src = (int32_t)(t + j) - (int32_t)d;
                acc += kern[(uint64_t)c * K + j] *
                       (src >= 0 ? x[(uint64_t)src * w + c] : 0.0f);
            }
            ref[(uint64_t)t * w + c] = x[(uint64_t)t * w + c] + acc;
        }
    }
    for (uint32_t c = 0; c < w; c++)
        for (uint32_t j = 0; j < d; j++)
            st_ref[(uint64_t)c * d + j] = x[(uint64_t)(n_tok - d + j) * w + c];

    void *x_d = NULL, *k_d = NULL, *o_d = NULL, *s_d = NULL;
    float *st_got = (float *)malloc((uint64_t)w * d * 4);
    int ok = cuda_ok(cudaMalloc(&x_d, (uint64_t)n_tok * w * 4), "x") &&
             cuda_ok(cudaMalloc(&k_d, (uint64_t)w * K * 4), "k") &&
             cuda_ok(cudaMalloc(&o_d, (uint64_t)n_tok * w * 4), "o") &&
             cuda_ok(cudaMalloc(&s_d, (uint64_t)w * d * 4), "s") &&
             cuda_ok(cudaMemcpy(x_d, x, (uint64_t)n_tok * w * 4, cudaMemcpyHostToDevice), "x h2d") &&
             cuda_ok(cudaMemcpy(k_d, kern, (uint64_t)w * K * 4, cudaMemcpyHostToDevice), "k h2d") &&
             cuda_ok(cudaMemset(s_d, 0, (uint64_t)w * d * 4), "s zero") &&
             /* chunked: 4 + 1 + 6 tokens, state carries */
             pulsar_sconv(o_d, x_d, k_d, s_d, 4, w, K) &&
             pulsar_sconv((char *)o_d + (uint64_t)4 * w * 4,
                          (const char *)x_d + (uint64_t)4 * w * 4, k_d, s_d, 1, w, K) &&
             pulsar_sconv((char *)o_d + (uint64_t)5 * w * 4,
                          (const char *)x_d + (uint64_t)5 * w * 4, k_d, s_d, 6, w, K) &&
             cuda_ok(cudaDeviceSynchronize(), "sync") &&
             cuda_ok(cudaMemcpy(got, o_d, (uint64_t)n_tok * w * 4, cudaMemcpyDeviceToHost), "o d2h") &&
             cuda_ok(cudaMemcpy(st_got, s_d, (uint64_t)w * d * 4, cudaMemcpyDeviceToHost), "s d2h");
    float maxd = 0.0f;
    if (ok) {
        for (uint64_t i = 0; i < (uint64_t)n_tok * w; i++)
            maxd = fmaxf(maxd, fabsf(got[i] - ref[i]));
        for (uint64_t i = 0; i < (uint64_t)w * d; i++)
            maxd = fmaxf(maxd, fabsf(st_got[i] - st_ref[i]));
        ok = maxd <= 1e-6f;
    }
    fprintf(stderr, "sconv-selftest: %s (max diff %.2e)\n",
            ok ? "PASS" : "FAIL", (double)maxd);
    if (x_d) cudaFree(x_d);
    if (k_d) cudaFree(k_d);
    if (o_d) cudaFree(o_d);
    if (s_d) cudaFree(s_d);
    free(x); free(kern); free(ref); free(got); free(st_ref); free(st_got);
    return ok;
}

/* Poison the padded tail of each logits row (inkling pads the vocab:
 * rows past unpadded_vocab_size hold garbage weights). */
__global__ static void fill_row_tail_kernel(
        float *x, uint32_t rows, uint32_t row_w, uint32_t keep, float v) {
    const uint32_t tail = row_w - keep;
    const uint64_t total = (uint64_t)rows * tail;
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const uint32_t r = (uint32_t)(i / tail);
    const uint32_t c = keep + (uint32_t)(i % tail);
    x[(uint64_t)r * row_w + c] = v;
}

extern "C" int pulsar_fill_row_tail(
        void *x_dev, uint32_t rows, uint32_t row_w, uint32_t keep, float v) {
    if (rows == 0 || keep >= row_w) return 1; /* nothing to poison */
    const uint64_t total = (uint64_t)rows * (row_w - keep);
    fill_row_tail_kernel<<<(uint32_t)((total + 255u) / 256u), 256>>>(
            (float *)x_dev, rows, row_w, keep, v);
    return cuda_ok(cudaGetLastError(), "fill row tail launch");
}

__global__ static void scale_kernel(float *x, uint32_t n, float c) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= c;
}

extern "C" int pulsar_scale(void *x_dev, uint32_t n, float c) {
    if (n == 0) return 0;
    scale_kernel<<<(n + 255u) / 256u, 256>>>((float *)x_dev, n, c);
    return cuda_ok(cudaGetLastError(), "scale launch");
}

/* logit softcap (gemma): x = cap * tanh(x / cap) */
__global__ static void softcap_kernel(float *x, uint32_t n, float cap) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}

extern "C" int pulsar_softcap(void *x_dev, uint32_t n, float cap) {
    if (n == 0) return 0;
    softcap_kernel<<<(n + 255u) / 256u, 256>>>((float *)x_dev, n, cap);
    return cuda_ok(cudaGetLastError(), "softcap launch");
}

/* fold a per-expert scale into selected route weights (gemma4
 * down_exps.scale): w[i] *= scale[sel[i]] */
__global__ static void router_scale_selected_kernel(
        float *w, const int32_t *sel, const float *scale, uint32_t n,
        uint32_t n_expert) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t e = sel[i];
    if (e >= 0 && (uint32_t)e < n_expert) w[i] *= scale[e];
}

extern "C" int pulsar_router_scale_selected(
        void *w_dev, const void *sel_dev, const void *scale_dev, uint32_t n,
        uint32_t n_expert) {
    if (n == 0 || n_expert == 0) return 0;
    router_scale_selected_kernel<<<(n + 255u) / 256u, 256>>>(
            (float *)w_dev, (const int32_t *)sel_dev,
            (const float *)scale_dev, n, n_expert);
    return cuda_ok(cudaGetLastError(), "router scale launch");
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
             pulsar_swiglu(sw_d, mm_d, mm_d, nel, 0.0f, 1.25f, 0) &&
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
    return ds4_gpu_gqa_head_rms_norm_weight(&xt, w ? &wt : NULL, rows, head_dim, eps);
}

extern "C" int pulsar_gqa_rope(
        void *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
        uint32_t rot_dim, uint32_t pos0, float theta, const void *factors) {
    ds4_gpu_tensor xt = shim(x), ft = shim(factors);
    return ds4_gpu_gqa_rope(&xt, n_tok, n_head, head_dim, rot_dim, pos0, theta,
                            factors ? &ft : NULL);
}

extern "C" int pulsar_gqa_kv_append(
        void *cache, const void *kv, uint32_t n_tok, uint32_t n_kv_head,
        uint32_t head_dim, uint32_t cap, uint32_t pos0, uint32_t kvq) {
    ds4_gpu_tensor ct = shim(cache), kt = shim(kv);
    if (kvq == 1)
        return ds4_gpu_gqa_kv_cache_append_fp8(&ct, &kt, n_tok, n_kv_head,
                                               head_dim, cap, pos0);
    return ds4_gpu_gqa_kv_cache_append(&ct, &kt, n_tok, n_kv_head, head_dim,
                                       cap, pos0);
}

extern "C" int pulsar_gqa_attention(
        void *out, const void *q, const void *k_cache, const void *v_cache,
        uint32_t n_tok, uint32_t n_head, uint32_t n_kv_head,
        uint32_t head_dim, uint32_t cap, uint32_t pos0,
        float scale, uint32_t window,
        const void *rel, uint32_t rel_extent, uint32_t kvq) {
    ds4_gpu_tensor ot = shim(out), qt = shim(q), kt = shim(k_cache),
                   vt = shim(v_cache);
    ds4_gpu_tensor rt = shim(rel);
    if (kvq == 1)
        return ds4_gpu_gqa_attention_fp8(&ot, &qt, &kt, &vt, n_tok, n_head,
                                         n_kv_head, head_dim, cap, pos0, scale,
                                         window, rel ? &rt : NULL, rel_extent);
    return ds4_gpu_gqa_attention(&ot, &qt, &kt, &vt, n_tok, n_head,
                                 n_kv_head, head_dim, cap, pos0, scale, window,
                                 rel ? &rt : NULL, rel_extent);
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
                                  beta_slow, 1.0f);
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
#include "dsa_indexer.inc"
#include "dsv4_kernels.inc"
#include "qwen35_kernels.inc"
