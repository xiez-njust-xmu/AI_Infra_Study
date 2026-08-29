#include "flash_attn.cuh"
#include "utils.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <vector>


// ──────────────────────────────────────────────────────────────────────
// Standalone main: generates random data, runs kernel, prints benchmark
// ──────────────────────────────────────────────────────────────────────
int main(int argc, char **argv)
{
    // ── Parse command-line or use defaults ─────────────────────────
    int B = 4, H = 8, N = 8192, d = 64;
    if (argc >= 5)
    {
        B = atoi(argv[1]);
        H = atoi(argv[2]);
        N = atoi(argv[3]);
        d = atoi(argv[4]);
    }
    float scale = 1.0f / sqrtf((float)d);

    printf("============================================\n");
    printf("FlashAttention V2+  (standalone harness)\n");
    printf("B=%d  H=%d  N=%d  d=%d  scale=%.6f\n", B, H, N, d, scale);
    printf("============================================\n");

    // Sanity: kernel is tuned for d=64
    if (d != 64)
        fprintf(stderr, "Warning: kernel is tuned for d=64, got d=%d\n", d);

    // ── Host tensors (host-side random data) ──────────────────────
    int qkv_elems = B * H * N * d;
    std::vector<float>  h_Q_f(qkv_elems), h_K_f(qkv_elems), h_V_f(qkv_elems);
    fill_random(h_Q_f.data(), qkv_elems);
    fill_random(h_K_f.data(), qkv_elems);
    fill_random(h_V_f.data(), qkv_elems);

    // Convert to bf16 for GPU
    std::vector<__nv_bfloat16> h_Q(qkv_elems), h_K(qkv_elems), h_V(qkv_elems);
    for (int i = 0; i < qkv_elems; i++)
    {
        h_Q[i] = __float2bfloat16(h_Q_f[i]);
        h_K[i] = __float2bfloat16(h_K_f[i]);
        h_V[i] = __float2bfloat16(h_V_f[i]);
    }

    // ── GPU memory & launch ───────────────────────────────────────
    __nv_bfloat16 *d_Q, *d_K, *d_V, *d_O;
    float *d_l, *d_m;
    CUDA_CHECK(cudaMalloc(&d_Q,  qkv_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_K,  qkv_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_V,  qkv_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_O,  qkv_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_l,  B * H * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m,  B * H * N * sizeof(float)));

    cudaMemcpy(d_Q, h_Q.data(), qkv_elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemset(d_O, 0, qkv_elems * sizeof(__nv_bfloat16));

    // ── Benchmark: warmup + timed ─────────────────────────────────
    const int warmup_iters = 3;
    const int bench_iters  = 20;

    for (int i = 0; i < warmup_iters; i++)
        flash_attn_naive_launch(d_Q, d_K, d_V, d_O, d_l, d_m, B, H, N, d, scale);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < bench_iters; i++)
        flash_attn_naive_launch(d_Q, d_K, d_V, d_O, d_l, d_m, B, H, N, d, scale);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_stop));
    float avg_ms = total_ms / bench_iters;

    printf("\n============================================\n");
    printf("Performance Benchmark (%d iters)\n", bench_iters);
    printf("============================================\n");
    printf("Avg latency      : %.3f ms\n", avg_ms);
    printf("============================================\n");

    // ── Cleanup ───────────────────────────────────────────────────
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_l)); CUDA_CHECK(cudaFree(d_m));

    return 0;
}
