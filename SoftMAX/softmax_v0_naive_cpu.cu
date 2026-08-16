/*****************************************************************************
 * softmax_v0_naive_cpu.cu — 朴素三遍扫描 CPU 实现
 *
 * 矩阵: 8192 × 8192, float
 * 输入: 模拟 QKV 矩阵 (N(0,5), 值可能大于 1)
 *
 * 算法 (每行):
 *   Pass 1: 找最大值     max_val = max(x_j)
 *   Pass 2: exp(x_j - max_val) 并累加
 *   Pass 3: 除以行和
 *****************************************************************************/

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>

// ============================================================================
// 三遍扫描 CPU Softmax
// ============================================================================
void softmax_naive_cpu(const float* input, float* output, int M, int N) {
    for (int row = 0; row < M; ++row) {
        const float* row_in  = input  + row * N;
        float*       row_out = output + row * N;

        float max_val = -INFINITY;
        for (int col = 0; col < N; ++col) {
            if (row_in[col] > max_val) max_val = row_in[col];
        }

        float sum = 0.0f;
        for (int col = 0; col < N; ++col) {
            float e = expf(row_in[col] - max_val);
            row_out[col] = e;
            sum += e;
        }

        float inv_sum = 1.0f / sum;
        for (int col = 0; col < N; ++col) {
            row_out[col] *= inv_sum;
        }
    }
}

// ============================================================================
// 参考实现 (double, 一次遍历)
// ============================================================================
void softmax_reference(const float* input, float* output, int M, int N) {
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

    std::cout << "Softmax v0 — 朴素三遍扫描 CPU\n";
    std::cout << "矩阵: " << M << " x " << N
              << "  (" << bytes / (1024.0 * 1024.0) << " MB)\n\n";

    // ---------- 模拟 QKV 输入: N(0, 5) ----------
    std::vector<float> h_input(numel);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 5.0f);
    for (size_t i = 0; i < numel; ++i) {
        h_input[i] = dist(rng);
    }
    // 检查实际范围
    float in_min = *std::min_element(h_input.begin(), h_input.end());
    float in_max = *std::max_element(h_input.begin(), h_input.end());
    std::cout << "输入范围: [" << in_min << ", " << in_max << "]\n\n";

    std::vector<float> h_output_cpu(numel);
    std::vector<float> h_output_ref(numel);

    // ---------- 参考 ----------
    auto t0 = std::chrono::high_resolution_clock::now();
    softmax_reference(h_input.data(), h_output_ref.data(), M, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ref_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---------- 朴素三遍 CPU ----------
    auto t2 = std::chrono::high_resolution_clock::now();
    softmax_naive_cpu(h_input.data(), h_output_cpu.data(), M, N);
    auto t3 = std::chrono::high_resolution_clock::now();
    double naive_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // ---------- 验证 ----------
    double max_err = 0.0;
    for (size_t i = 0; i < numel; ++i) {
        double rel = fabs(static_cast<double>(h_output_cpu[i]) - static_cast<double>(h_output_ref[i]))
                   / fmax(1.0, static_cast<double>(h_output_ref[i]));
        if (rel > max_err) max_err = rel;
    }
    double max_sum_dev = 0.0;
    for (int r = 0; r < M; ++r) {
        double sum = 0.0;
        for (int c = 0; c < N; ++c) sum += h_output_cpu[r * N + c];
        double dev = fabs(sum - 1.0);
        if (dev > max_sum_dev) max_sum_dev = dev;
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CPU 参考耗时:     " << ref_ms << " ms\n";
    std::cout << "CPU 朴素三遍耗时: " << naive_ms << " ms\n";
    std::cout << "吞吐:             " << (bytes / naive_ms / 1e6) << " GB/s\n";
    std::cout << std::scientific << std::setprecision(2);
    std::cout << "最大相对误差:     " << max_err << "\n";
    std::cout << "各行和=1 最大偏差: " << max_sum_dev << "\n";

    bool pass = (max_err < 1e-5f) && (max_sum_dev < 1e-5f);
    std::cout << "\n>>> " << (pass ? "PASS" : "FAIL") << " <<<\n";
    return pass ? 0 : 1;
}