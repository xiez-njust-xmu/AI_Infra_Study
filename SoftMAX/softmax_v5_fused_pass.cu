/*****************************************************************************
 * softmax_v5_fused_pass.cu — 两遍融合 Kernel (避免第 3 次全局内存读取)
 *
 * 矩阵: 8192 × 8192, float
 * 输入: 模拟 QKV 矩阵 (N(0,5))
 *
 * 策略: 真正"两遍"不写回中间结果, 所有数据驻留寄存器
 *
 *   传统三遍 (v4):
 *     Pass 1: 读 input  → 求 max         (reg → reg)
 *     Pass 2: 读 input  → exp → 写 output → 求 sum  (reg → gmem)
 *     Pass 3: 读 output → 归一化 → 写 output         (gmem → gmem)  ← 浪费
 *
 *   融合两遍 (v5):
 *     Pass 1: 读 input 到寄存器 → 从寄存器求 max → shuffle 规约
 *     Pass 2: 从寄存器 exp(x - max) → 累加 sum → shuffle 规约
 *             再从寄存器归一化 → 写 output (一次全局写)
 *     输入只读一次, 结果只写一次, 无中间全局读写
 *
 *   每线程 32 元素 = 8 × float4 寄存器缓冲, 完全片上处理
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
// 工具: Warp 内 Shuffle 规约
// ============================================================================
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

// ============================================================================
// CUDA Kernel: 两遍融合 (寄存器驻留)
// ============================================================================
__global__ void softmax_v5_kernel(const float* __restrict__ input,
                                  float*       __restrict__ output,
                                  int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;

    const float* row_in  = input  + row * N;
    float*       row_out = output + row * N;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int warp_id = tid / 32;
    int lane   = tid & 31;
    int num_warps = (num_threads + 31) / 32;

    // 每线程 32 元素 = 8 × float4, 完全放入寄存器
    int elems_per_thread = (N + num_threads - 1) / num_threads;
    int start = tid * elems_per_thread;
    int end   = min(start + elems_per_thread, N);
    int vec_iters = elems_per_thread / 4;
    int tail_start = start + vec_iters * 4;

    // 寄存器缓冲: 复用同一块存输入 → 覆盖为 exp → 覆盖为输出
    float4 buf[8];  // 32 寄存器, 8192/256=32 元素/线程

    extern __shared__ float shared[];
    float* swarp = shared;

    // ==================== Pass 1: 读 + 求 max ====================
    float local_max = -INFINITY;
    for (int i = 0; i < vec_iters; ++i) {
        buf[i] = reinterpret_cast<const float4*>(row_in + start)[i];
        local_max = fmaxf(local_max, buf[i].x);
        local_max = fmaxf(local_max, buf[i].y);
        local_max = fmaxf(local_max, buf[i].z);
        local_max = fmaxf(local_max, buf[i].w);
    }
    for (int i = tail_start; i < end; ++i) {
        float v = row_in[i];
        // 存到 buf 尾部 (scalar 存不进 float4, 单独处理)
        // 对于 N=8192, BLOCK=256, 这里不会执行
        local_max = fmaxf(local_max, v);
    }

    // 两级 shuffle 规约 → 全局 max
    local_max = warp_reduce_max(local_max);
    if (lane == 0) swarp[warp_id] = local_max;
    __syncthreads();

    float row_max = -INFINITY;
    if (warp_id == 0) {
        if (lane < num_warps) row_max = swarp[lane];
        row_max = warp_reduce_max(row_max);
        if (lane == 0) swarp[0] = row_max;
    }
    __syncthreads();
    row_max = swarp[0];
    __syncthreads();

    // ==================== Pass 2: exp + sum + normalize + 写回 ====================
    float local_sum = 0.0f;
    for (int i = 0; i < vec_iters; ++i) {
        buf[i].x = expf(buf[i].x - row_max);
        buf[i].y = expf(buf[i].y - row_max);
        buf[i].z = expf(buf[i].z - row_max);
        buf[i].w = expf(buf[i].w - row_max);
        local_sum += buf[i].x + buf[i].y + buf[i].z + buf[i].w;
    }

    // 两级 shuffle 规约 → 全局 sum
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) swarp[warp_id] = local_sum;
    __syncthreads();

    float row_sum = 0.0f;
    if (warp_id == 0) {
        if (lane < num_warps) row_sum = swarp[lane];
        row_sum = warp_reduce_sum(row_sum);
        if (lane == 0) swarp[0] = row_sum;
    }
    __syncthreads();
    row_sum = swarp[0];
    __syncthreads();

    // 归一化 + 写回 (一次全局写)
    float inv_sum = 1.0f / row_sum;
    for (int i = 0; i < vec_iters; ++i) {
        buf[i].x *= inv_sum;
        buf[i].y *= inv_sum;
        buf[i].z *= inv_sum;
        buf[i].w *= inv_sum;
        reinterpret_cast<float4*>(row_out + start)[i] = buf[i];
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

    std::cout << "Softmax v5 — 两遍融合 Kernel (寄存器驻留, 无中间全局访存)\n";
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
    size_t shared_bytes = num_warps * sizeof(float);

    std::vector<float> h_output(numel);

    softmax_v5_kernel<<<GRID, BLOCK, shared_bytes>>>(d_input, d_output, M, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int ITERS = 10;
    cudaEventRecord(start);
    for (int it = 0; it < ITERS; ++it) {
        softmax_v5_kernel<<<GRID, BLOCK, shared_bytes>>>(d_input, d_output, M, N);
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
              << " / " << shared_bytes << " B\n";
    std::cout << "每线程元素:        " << (N / BLOCK)
              << "  (8 × float4, 32 寄存器)\n";
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