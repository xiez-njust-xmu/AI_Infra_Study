/*****************************************************************************
 * online_softmax_v4_onepass.cu — 寄存器缓存消除第二遍读取 (One-Pass)
 *
 * 矩阵: 8192 × 8192, float
 * 输入: 模拟 QKV 矩阵 (N(0,5))
 *
 * 真正的一遍 (One-Pass) Softmax:
 *
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │  全局内存: 只读一次, 只写一次                                   │
 *   │                                                               │
 *   │   Step 1: 读 input ──→ float4 buf[8] (寄存器)                 │
 *   │   Step 2: 从 buf 在线更新 max+sum (纯片上)                     │
 *   │   Step 3: 从 buf 归一化 + 写 output (纯片上 → 一次全局写)     │
 *   │                                                               │
 *   │  全局事务: 1 次读 + 1 次写 = 2 次 (理论下限)                  │
 *   └─────────────────────────────────────────────────────────────────┘
 *
 *  对比传统三遍 (v1 原始):
 *     读 input → 读 input → 写 output → 读 output → 写 output = 5 次
 *  对比两遍融合 (原系列 v5):
 *     读 input → 写 output = 2 次 (与 v5 相同, 但计算方式不同)
 *
 *  关键: 数据一次性加载到寄存器后, 所有计算都在片上完成,
 *        不需要中间结果写回全局内存再读回
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
// 设备工具: Warp 内合并两个 Online 状态 (max, sum)
// ============================================================================
__device__ void warp_merge_online(float& max_val, float& sum) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_max = __shfl_xor_sync(0xFFFFFFFF, max_val, offset);
        float other_sum = __shfl_xor_sync(0xFFFFFFFF, sum, offset);

        float new_max = fmaxf(max_val, other_max);
        sum = sum * expf(max_val - new_max)
            + other_sum * expf(other_max - new_max);
        max_val = new_max;
    }
}

// ============================================================================
// CUDA Kernel: One-Pass Online Softmax (寄存器全程驻留)
// ============================================================================
__global__ void online_softmax_v4_kernel(const float* __restrict__ input,
                                         float*       __restrict__ output,
                                         int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;

    // 全局内存指针 (每行偏移)
    const float* row_in  = input  + row * N;
    float*       row_out = output + row * N;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int warp_id = tid / 32;
    int lane   = tid & 31;
    int num_warps = (num_threads + 31) / 32;

    // 每线程连续段, float4 对齐
    int elems_per_thread = (N + num_threads - 1) / num_threads;
    int start = tid * elems_per_thread;

    // 寄存器缓冲: 整个线程的数据驻留在此
    float4 buf[8];  // 8 × float4 = 32 元素 = 32 寄存器

    // ==================== Step 1: 唯一一次全局读 ====================
    // float4 向量化, 128-bit 单次事务, coalesced
    #pragma unroll
    for (int i = 0; i < elems_per_thread / 4; ++i) {
        buf[i] = reinterpret_cast<const float4*>(row_in + start)[i];
    }

    // ==================== Step 2: 纯片上 Online max + sum ====================
    // 所有操作在寄存器上完成, 无全局访存
    float max_val = -INFINITY;
    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < elems_per_thread / 4; ++i) {
        float x0 = buf[i].x, x1 = buf[i].y, x2 = buf[i].z, x3 = buf[i].w;

        float old_max = max_val;
        if (x0 > max_val) max_val = x0;
        sum = sum * expf(old_max - max_val) + expf(x0 - max_val);

        old_max = max_val;
        if (x1 > max_val) max_val = x1;
        sum = sum * expf(old_max - max_val) + expf(x1 - max_val);

        old_max = max_val;
        if (x2 > max_val) max_val = x2;
        sum = sum * expf(old_max - max_val) + expf(x2 - max_val);

        old_max = max_val;
        if (x3 > max_val) max_val = x3;
        sum = sum * expf(old_max - max_val) + expf(x3 - max_val);
    }

    // --- 跨线程合并: 所有线程的 (max, sum) 合并为全局 (row_max, row_sum) ---
    // 一级: Warp 内 shuffle
    warp_merge_online(max_val, sum);

    // 二级: Warp 间 (Shared Memory)
    extern __shared__ float shared[];
    float* s_max = shared;
    float* s_sum = shared + num_warps;

    if (lane == 0) {
        s_max[warp_id] = max_val;
        s_sum[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        if (lane < num_warps) {
            max_val = s_max[lane];
            sum     = s_sum[lane];
        } else {
            max_val = -INFINITY;
            sum     = 0.0f;
        }
        warp_merge_online(max_val, sum);
        if (lane == 0) {
            s_max[0] = max_val;
            s_sum[0] = sum;
        }
    }
    __syncthreads();

    float row_max = s_max[0];
    float row_sum = s_sum[0];
    __syncthreads();

    // ==================== Step 3: 归一化 + 唯一一次全局写 ====================
    // 从寄存器 buf 读取, 归一化后直接写 output
    // 没有"先写 exp 再读回来归一化"的中间步骤
    float inv_sum = 1.0f / row_sum;
    #pragma unroll
    for (int i = 0; i < elems_per_thread / 4; ++i) {
        float4 v;
        v.x = expf(buf[i].x - row_max) * inv_sum;
        v.y = expf(buf[i].y - row_max) * inv_sum;
        v.z = expf(buf[i].z - row_max) * inv_sum;
        v.w = expf(buf[i].w - row_max) * inv_sum;
        reinterpret_cast<float4*>(row_out + start)[i] = v;
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

    std::cout << "Online Softmax v4 — One-Pass (寄存器缓存, 1 读 + 1 写)\n";
    std::cout << "矩阵: " << M << " x " << N
              << "  (" << bytes / (1024.0 * 1024.0) << " MB)\n";
    std::cout << "Device: " << prop.name << "\n\n";

    // ---------- 输入 ----------
    std::vector<float> h_input(numel);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 5.0f);
    for (size_t i = 0; i < numel; ++i) h_input[i] = dist(rng);
    float in_min = *std::min_element(h_input.begin(), h_input.end());
    float in_max = *std::max_element(h_input.begin(), h_input.end());
    std::cout << "输入范围: [" << in_min << ", " << in_max << "]\n\n";

    // ---------- 分配 device ----------
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
    const int GRID  = M;
    int num_warps = (BLOCK + 31) / 32;
    size_t shared_bytes = 2 * num_warps * sizeof(float);

    std::vector<float> h_output(numel);

    online_softmax_v4_kernel<<<GRID, BLOCK, shared_bytes>>>(d_input, d_output, M, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int ITERS = 10;
    cudaEventRecord(start);
    for (int it = 0; it < ITERS; ++it) {
        online_softmax_v4_kernel<<<GRID, BLOCK, shared_bytes>>>(d_input, d_output, M, N);
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
    std::cout << "每线程元素:        " << (N / BLOCK) << "  (8 × float4)\n";
    std::cout << "全局事务:          1 读 + 1 写 (理论下限)\n";
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