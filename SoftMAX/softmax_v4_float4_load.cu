/*****************************************************************************
 * softmax_v4_float4_load.cu — 向量化加载 float4
 *
 * 矩阵: 8192 × 8192, float
 * 输入: 模拟 QKV 矩阵 (N(0,5))
 *
 * 策略: 继承 v3 (Warp Shuffle 两级规约) + float4 向量化访存
 *
 *   每线程一次读 4 个 float (128-bit 单次事务), 减少指令数并提升带宽利用率
 *   8192 / 256 = 32 元素/线程 = 8 次 float4 访存, 完美对齐
 *
 *   Pass 1 (max-reduce): float4 加载 → 4 路 fmaxf → 两级 shuffle
 *   Pass 2 (sum-reduce): float4 加载 → 4 路 expf → 累加 → 两级 shuffle
 *   Pass 3 (normalize):  float4 加载 → 4 路乘 inv_sum → float4 写回
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
// CUDA Kernel: float4 向量化访存
// ============================================================================
__global__ void softmax_v4_kernel(const float* __restrict__ input,
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

    // 每个线程处理连续元素, float4 对齐
    int elems_per_thread = (N + num_threads - 1) / num_threads;
    int start = tid * elems_per_thread;
    int end   = min(start + elems_per_thread, N);

    // float4 迭代次数 (每线程固定处理 4 的倍数)
    int vec_iters = elems_per_thread / 4;
    int tail_start = start + vec_iters * 4;

    extern __shared__ float shared[];
    float* swarp = shared;

    // ========== Pass 1: max-reduce (float4) ==========
    float local_max = -INFINITY;
    for (int i = 0; i < vec_iters; ++i) {
        float4 v = reinterpret_cast<const float4*>(row_in + start)[i];
        local_max = fmaxf(local_max, v.x);
        local_max = fmaxf(local_max, v.y);
        local_max = fmaxf(local_max, v.z);
        local_max = fmaxf(local_max, v.w);
    }
    // 尾部非对齐元素 (N=8192, BLOCK=256 时不会执行)
    for (int i = tail_start; i < end; ++i) {
        local_max = fmaxf(local_max, row_in[i]);
    }

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

    // ========== Pass 2: exp + sum-reduce (float4) ==========
    float local_sum = 0.0f;
    for (int i = 0; i < vec_iters; ++i) {
        float4 v = reinterpret_cast<const float4*>(row_in + start)[i];
        float4 e;
        e.x = expf(v.x - row_max);
        e.y = expf(v.y - row_max);
        e.z = expf(v.z - row_max);
        e.w = expf(v.w - row_max);
        reinterpret_cast<float4*>(row_out + start)[i] = e;
        local_sum += e.x + e.y + e.z + e.w;
    }
    for (int i = tail_start; i < end; ++i) {
        float e = expf(row_in[i] - row_max);
        row_out[i] = e;
        local_sum += e;
    }

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

    // ========== Pass 3: normalize (float4) ==========
    float inv_sum = 1.0f / row_sum;
    for (int i = 0; i < vec_iters; ++i) {
        float4 v = reinterpret_cast<float4*>(row_out + start)[i];
        v.x *= inv_sum;
        v.y *= inv_sum;
        v.z *= inv_sum;
        v.w *= inv_sum;
        reinterpret_cast<float4*>(row_out + start)[i] = v;
    }
    for (int i = tail_start; i < end; ++i) {
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

    std::cout << "Softmax v4 — float4 向量化访存\n";
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

    softmax_v4_kernel<<<GRID, BLOCK, shared_bytes>>>(d_input, d_output, M, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int ITERS = 10;
    cudaEventRecord(start);
    for (int it = 0; it < ITERS; ++it) {
        softmax_v4_kernel<<<GRID, BLOCK, shared_bytes>>>(d_input, d_output, M, N);
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
    std::cout << "每线程元素:        " << (N / BLOCK) << "  ("
              << (N / BLOCK / 4) << " × float4)\n";
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