#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace attention {

    void launchAttentionBaseline(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
        float* __restrict__ QK, float* __restrict__ QK_softmax, float* __restrict__ O,
        const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream = 0);

    void launchFlashAttentionKernel_v4(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
        float* __restrict__ O, const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream = 0);
    
    void launchFlashAttentionKernel_v5(const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
        half* __restrict__ O, const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream = 0);

    void launchFlashAttentionKernel_v6(const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
        half* __restrict__ O, const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream = 0);

} // namespace attention
