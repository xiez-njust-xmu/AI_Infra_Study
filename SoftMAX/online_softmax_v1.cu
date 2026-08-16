/*****************************************************************************
 * softmax_v1_online.cu — 单线程 Online Softmax (CUDA)
 *
 * 矩阵: 8192 × 8192, float
 * 输入: 模拟 QKV 矩阵 (N(0,5))
 *
 * 策略: 一个线程处理一行, Online 算法一遍扫描 max+sum
 *   float4 向量化加载到寄存器, 在线更新, 再归一化写回
 *
 *  Online 核心 (每元素):
 *     old_max = max_val
 *     if x > max_val:  max_val = x
 *     sum = sum * exp(old_max - new_max) + exp(x - new_max)
 *
 *  全局访存: 1 次读 + 1 次写 (同 v5 两遍融合)
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
// CUDA Kernel: 单线程一行, Online Softmax
// ============================================================================
__global__ void softmax_v1_online_kernel(const float* __restrict__ input,
                                         float*       __restrict__ output,
                                         int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const float* row_in  = input  + row * N;
    float*       row_out = output + row * N;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // 每线程连续处理一段 (coalesced access)
    int elems_per_thread = (N + num_threads - 1) / num_threads;
    int start = tid * elems_per_thread;
    int end   = min(start + elems_per_thread, N);
    int vec_iters = elems_per_thread / 4;
    int tail_start = start + vec_iters * 4;

    // 寄存器缓冲: 存储原始输入值
    float4 buf[8];

    // ==================== 加载输入到寄存器 ====================
    for (int i = 0; i < vec_iters; ++i) {
        buf[i] = reinterpret_cast<const float4*>(row_in + start)[i];
    }

    // ==================== 一遍扫描: Online max + sum ====================
    float max_val = -INFINITY;
    float sum = 0.0f;

    for (int i = 0; i < vec_iters; ++i) {
        float x0 = buf[i].x, x1 = buf[i].y, x2 = buf[i].z, x3 = buf[i].w;

        // --- x0 ---
        float old_max = max_val;
        if (x0 > max_val) max_val = x0;
        sum = sum * expf(old_max - max_val) + expf(x0 - max_val);

        // --- x1 ---
        old_max = max_val;
        if (x1 > max_val) max_val = x1;
        sum = sum * expf(old_max - max_val) + expf(x1 - max_val);

        // --- x2 ---
        old_max = max_val;
        if (x2 > max_val) max_val = x2;
        sum = sum * expf(old_max - max_val) + expf(x2 - max_val);

        // --- x3 ---
        old_max = max_val;
        if (x3 > max_val) max_val = x3;
        sum = sum * expf(old_max - max_val) + expf(x3 - max_val);
    }

    // 尾部非对齐元素 (N=8192, BLOCK=256 时不会执行)
    #pragma unroll
    for (int i = tail_start; i < end; ++i) {
        float x = row_in[i];
        float old_max = max_val;
        if (x > max_val) max_val = x;
        sum = sum * expf(old_max - max_val) + expf(x - max_val);
    }

    // ==================== 归一化 + 写回 ====================
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < vec_iters; ++i) {
        float4 v;
        v.x = expf(buf[i].x - max_val) * inv_sum;
        v.y = expf(buf[i].y - max_val) * inv_sum;
        v.z = expf(buf[i].z - max_val) * inv_sum;
        v.w = expf(buf[i].w - max_val) * inv_sum;
        reinterpret_cast<float4*>(row_out + start)[i] = v;
    }

    #pragma unroll
    for (int i = tail_start; i < end; ++i) {
        row_out[i] = expf(row_in[i] - max_val) * inv_sum;
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

    std::cout << "Softmax v1 Online — 单线程一行, Online 一遍扫描 max+sum\n";
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
    const int GRID  = (M + BLOCK - 1) / BLOCK;

    std::vector<float> h_output(numel);

    softmax_v1_online_kernel<<<GRID, BLOCK>>>(d_input, d_output, M, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int ITERS = 10;
    cudaEventRecord(start);
    for (int it = 0; it < ITERS; ++it) {
        softmax_v1_online_kernel<<<GRID, BLOCK>>>(d_input, d_output, M, N);
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
    std::cout << "Grid/Block:         " << GRID << " / " << BLOCK << "\n";
    std::cout << "每线程元素:        " << (N / BLOCK) << "  (8 × float4)\n";
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