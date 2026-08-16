/*****************************************************************************
 * online_softmax_v2_block_shuffle.cu — Block 级并行 + Warp Shuffle 合并规约
 *
 * 矩阵: 8192 × 8192, float
 * 输入: 模拟 QKV 矩阵 (N(0,5))
 *
 * 策略: 一个 Block 一行, 多线程协作, Online Softmax
 *
 *  每个线程:
 *     加载自己的段 (float4 向量化) → 寄存器
 *     在线更新 local max + local sum (一遍扫描)
 *
 *  跨线程合并 (Warp Shuffle 两级规约):
 *     merge(max_a, sum_a, max_b, sum_b) → (new_max, new_sum)
 *       new_max = max(max_a, max_b)
 *       new_sum = sum_a * exp(max_a - new_max) + sum_b * exp(max_b - new_max)
 *
 *  一级: Warp 内 shuffle 逐对合并
 *  二级: Warp 结果 → Shared Memory → 首 Warp 再 shuffle 合并
 *
 *  归一化: 每个线程从寄存器 exp(x - final_max) / final_sum → 写回
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
// CUDA Kernel: Block 并行 Online Softmax
// ============================================================================
__global__ void online_softmax_v2_kernel(const float* __restrict__ input,
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

    // 每线程一段连续元素 (coalesced)
    int elems_per_thread = (N + num_threads - 1) / num_threads;
    int start = tid * elems_per_thread;
    int end   = min(start + elems_per_thread, N);
    int vec_iters = elems_per_thread / 4;
    int tail_start = start + vec_iters * 4;

    // 寄存器缓冲: float4 × 8 (32 元素)
    float4 buf[8];

    // ==================== 加载输入到寄存器 ====================
    #pragma unroll
    for (int i = 0; i < vec_iters; ++i) {
        buf[i] = reinterpret_cast<const float4*>(row_in + start)[i];
    }

    // ==================== 每线程内 Online max+sum ====================
    float max_val = -INFINITY;
    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < vec_iters; ++i) {
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

    // ==================== 一级: Warp 内 shuffle 合并 ====================
    warp_merge_online(max_val, sum);  // 所有 lane 都得到该 Warp 的合并结果

    // ==================== 二级: Warp 间合并 ====================
    extern __shared__ float shared[];
    float* s_max = shared;               // num_warps 个 max
    float* s_sum = shared + num_warps;   // num_warps 个 sum

    if (lane == 0) {
        s_max[warp_id] = max_val;
        s_sum[warp_id] = sum;
    }
    __syncthreads();

    // 首 Warp 从 Shared Memory 加载, 再 shuffle 合并
    if (warp_id == 0) {
        if (lane < num_warps) {
            max_val = s_max[lane];
            sum     = s_sum[lane];
        } else {
            max_val = -INFINITY;
            sum     = 0.0f;
        }
        warp_merge_online(max_val, sum);  // lane 0 持有全局结果
        if (lane == 0) {
            s_max[0] = max_val;
            s_sum[0] = sum;
        }
    }
    __syncthreads();

    float row_max = s_max[0];
    float row_sum = s_sum[0];
    __syncthreads();

    // ==================== 归一化 + 写回 ====================
    float inv_sum = 1.0f / row_sum;
    #pragma unroll
    for (int i = 0; i < vec_iters; ++i) {
        float4 v;
        v.x = expf(buf[i].x - row_max) * inv_sum;
        v.y = expf(buf[i].y - row_max) * inv_sum;
        v.z = expf(buf[i].z - row_max) * inv_sum;
        v.w = expf(buf[i].w - row_max) * inv_sum;
        reinterpret_cast<float4*>(row_out + start)[i] = v;
    }
    #pragma unroll
    for (int i = tail_start; i < end; ++i) {
        row_out[i] = expf(row_in[i] - row_max) * inv_sum;
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

    std::cout << "Online Softmax v2 — Block 并行 + Warp Shuffle 合并规约\n";
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
    size_t shared_bytes = 2 * num_warps * sizeof(float);  // max + sum

    std::vector<float> h_output(numel);

    online_softmax_v2_kernel<<<GRID, BLOCK, shared_bytes>>>(d_input, d_output, M, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int ITERS = 10;
    cudaEventRecord(start);
    for (int it = 0; it < ITERS; ++it) {
        online_softmax_v2_kernel<<<GRID, BLOCK, shared_bytes>>>(d_input, d_output, M, N);
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