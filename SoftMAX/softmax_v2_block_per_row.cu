/*****************************************************************************
 * softmax_v2_block_per_row.cu — 一个 Block 协作处理一行 (Safe Softmax)
 *
 * 矩阵: 8192 × 8192, float
 * 输入: 模拟 QKV 矩阵 (N(0,5))
 *
 * 策略: 一个 Block 处理一行，多线程协作
 *   每个线程负责一段连续元素 (coalesced access)
 *
 *   Pass 1 (max-reduce): 线程内求局部 max → Shared Memory 树形规约 → 行 max
 *   Pass 2 (sum-reduce): 线程内 exp(x - max) 并累加局部 sum → 树形规约 → 行 sum
 *   Pass 3 (element-wise): 每个线程归一化自己那一段
 *****************************************************************************/

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>

// ============================================================================
// CUDA Kernel: 一个 Block 处理一行
// ============================================================================
__global__ void softmax_v2_kernel(const float* __restrict__ input,
                                  float*       __restrict__ output,
                                  int M, int N) {
    // 每个 Block 处理一行
    int row = blockIdx.x;
    if (row >= M) return;

    const float* row_in  = input  + row * N;
    float*       row_out = output + row * N;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // 每个线程处理连续的一段 (coalesced global memory access)
    int elems_per_thread = (N + num_threads - 1) / num_threads;
    int start = tid * elems_per_thread;
    int end   = min(start + elems_per_thread, N);

    // Shared Memory: [blockDim.x 用于 max 规约 | blockDim.x 用于 sum 规约]
    extern __shared__ float shared[];
    float* smax = shared;               // 前半段: max 规约
    float* ssum = shared + blockDim.x;  // 后半段: sum 规约

    // ========== Pass 1: max-reduce ==========
    float local_max = -INFINITY;
    for (int i = start; i < end; ++i) {
        local_max = fmaxf(local_max, row_in[i]);
    }
    smax[tid] = local_max;
    __syncthreads();

    // 树形规约: 求全局 max
    for (int s = num_threads >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }
    float row_max = smax[0];
    __syncthreads();

    // ========== Pass 2: exp + sum-reduce ==========
    float local_sum = 0.0f;
    for (int i = start; i < end; ++i) {
        float e = expf(row_in[i] - row_max);
        row_out[i] = e;         // 暂存 exp 结果
        local_sum += e;
    }
    ssum[tid] = local_sum;
    __syncthreads();

    // 树形规约: 求全局 sum
    for (int s = num_threads >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
        }
        __syncthreads();
    }
    float row_sum = ssum[0];
    __syncthreads();

    // ========== Pass 3: normalize (element-wise) ==========
    float inv_sum = 1.0f / row_sum;
    for (int i = start; i < end; ++i) {
        row_out[i] *= inv_sum;
    }
}

// ============================================================================
// CPU 参考 (double)
// ============================================================================
void softmax_reference_cpu(const float* input, float* output, int M, int N) {
    for (int row = 0; row < M; ++row) {
        const float* row_in = input + row * N;
        double max_val = -INFINITY;
        for (int col = 0; col < N; ++col) {
            if (row_in[col] > max_val) max_val = row_in[col];
        }
        double sum = 0.0;
        for (int col = 0; col < N; ++col) {
            sum += exp(static_cast<double>(row_in[col]) - max_val);
        }
        double inv_sum = 1.0 / sum;
        for (int col = 0; col < N; ++col) {
            output[row * N + col] = static_cast<float>(
                exp(static_cast<double>(row_in[col]) - max_val) * inv_sum);
        }
    }
}

// ============================================================================
int main() {
    const int M = 8192;
    const int N = 8192;
    const size_t numel = static_cast<size_t>(M) * N;
    const size_t bytes = numel * sizeof(float);

    int dev_id = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev_id);

    std::cout << "Softmax v2 — 一个 Block 一行 (CUDA, Shared Memory 规约)\n";
    std::cout << "矩阵: " << M << " x " << N
              << "  (" << bytes / (1024.0 * 1024.0) << " MB)\n";
    std::cout << "Device: " << prop.name << "\n\n";

    // ---------- 模拟 QKV 输入 ----------
    std::vector<float> h_input(numel);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 5.0f);
    for (size_t i = 0; i < numel; ++i) {
        h_input[i] = dist(rng);
    }
    float in_min = *std::min_element(h_input.begin(), h_input.end());
    float in_max = *std::max_element(h_input.begin(), h_input.end());
    std::cout << "输入范围: [" << in_min << ", " << in_max << "]\n\n";

    // ---------- 分配 device 内存 ----------
    float *d_input, *d_output;
    cudaMalloc(&d_input,  bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    // ---------- CPU 参考 ----------
    std::vector<float> h_ref(numel);
    auto t0 = std::chrono::high_resolution_clock::now();
    softmax_reference_cpu(h_input.data(), h_ref.data(), M, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ref_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---------- CUDA kernel ----------
    const int BLOCK = 256;
    // 每个 block 处理一行，共 M 个 block
    const int GRID = M;

    // Shared Memory 大小: 2 × blockDim.x × sizeof(float)
    size_t shared_bytes = 2 * BLOCK * sizeof(float);

    std::vector<float> h_output(numel);

    // warm-up
    softmax_v2_kernel<<<GRID, BLOCK, shared_bytes>>>(d_input, d_output, M, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int ITERS = 10;
    cudaEventRecord(start);
    for (int it = 0; it < ITERS; ++it) {
        softmax_v2_kernel<<<GRID, BLOCK, shared_bytes>>>(d_input, d_output, M, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_ms_total;
    cudaEventElapsedTime(&kernel_ms_total, start, stop);
    double kernel_ms = kernel_ms_total / static_cast<double>(ITERS);

    cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    // ---------- 验证 ----------
    double max_err = 0.0;
    for (size_t i = 0; i < numel; ++i) {
        double rel = fabs(static_cast<double>(h_output[i]) - static_cast<double>(h_ref[i]))
                   / fmax(1.0, static_cast<double>(h_ref[i]));
        if (rel > max_err) max_err = rel;
    }
    double max_sum_dev = 0.0;
    for (int r = 0; r < M; ++r) {
        double sum = 0.0;
        for (int c = 0; c < N; ++c) sum += h_output[r * N + c];
        double dev = fabs(sum - 1.0);
        if (dev > max_sum_dev) max_sum_dev = dev;
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Grid/Block/Shared:  " << GRID << " / " << BLOCK
              << " / " << (shared_bytes / 1024) << " KB\n";
    std::cout << "迭代次数:          " << ITERS << "\n";
    std::cout << "CUDA kernel 平均:  " << kernel_ms << " ms\n";
    std::cout << "CUDA 吞吐:         " << (2.0 * bytes / kernel_ms / 1e6) << " GB/s\n";
    std::cout << "CPU 参考耗时:      " << ref_ms << " ms\n";
    std::cout << std::scientific << std::setprecision(2);
    std::cout << "最大相对误差:      " << max_err << "\n";
    std::cout << "各行和=1 最大偏差:  " << max_sum_dev << "\n";

    bool pass = (max_err < 1e-5) && (max_sum_dev < 1e-5);
    std::cout << "\n>>> " << (pass ? "PASS" : "FAIL") << " <<<\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);

    return pass ? 0 : 1;
}