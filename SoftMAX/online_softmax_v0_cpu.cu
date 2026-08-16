/*****************************************************************************
 * softmax_v0_online_cpu.cu — Online Softmax CPU 实现 (一遍扫描)
 *
 * 矩阵: 8192 × 8192, float
 * 输入: 模拟 QKV 矩阵 (N(0,5))
 *
 * 算法: 一遍扫描同时维护 max 和 sum
 *   当出现更大的 max 时, 用乘法修正已累加的 sum:
 *     sum = sum * exp(old_max - new_max) + exp(x - new_max)
 *   避免了三遍扫描中"先求 max 再重算 exp"的开销
 *
 *   本质上仍是两遍 (第一遍在线更新 max+sum, 第二遍 normalize),
 *   但避免了第一遍中求 max 的单独遍历
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
// Online Softmax (一遍扫描 max+sum)
// ============================================================================
void softmax_online_cpu(const float* input, float* output, int M, int N) {
    for (int row = 0; row < M; ++row) {
        const float* row_in = input + row * N;

        // ---------- 一遍扫描: 同时维护 max 和 sum ----------
        float max_val = -INFINITY;
        float sum = 0.0f;
        for (int col = 0; col < N; ++col) {
            float x = row_in[col];
            float old_max = max_val;
            if (x > max_val) {
                max_val = x;
            }
            // 乘法修正: 旧 sum 缩放到新 max 尺度, 再加当前 exp
            sum = sum * expf(old_max - max_val) + expf(x - max_val);
        }

        // ---------- 归一化 ----------
        float inv_sum = 1.0f / sum;
        for (int col = 0; col < N; ++col) {
            output[row * N + col] = expf(row_in[col] - max_val) * inv_sum;
        }
    }
}

// ============================================================================
// 参考实现 (double, 三遍, 用于验证)
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
// 对比: 朴素三遍 CPU (float, 三遍)
// ============================================================================
void softmax_naive_cpu(const float* input, float* output, int M, int N) {
    for (int row = 0; row < M; ++row) {
        const float* row_in = input + row * N;
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
int main() {
    const int M = 8192;
    const int N = 8192;
    const size_t numel = static_cast<size_t>(M) * N;
    const size_t bytes = numel * sizeof(float);

    std::cout << "Softmax Online (CPU) — 一遍扫描 max+sum\n";
    std::cout << "矩阵: " << M << " x " << N
              << "  (" << bytes / (1024.0 * 1024.0) << " MB)\n\n";

    // ---------- 模拟 QKV 输入 ----------
    std::vector<float> h_input(numel);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 5.0f);
    for (size_t i = 0; i < numel; ++i) h_input[i] = dist(rng);
    float in_min = *std::min_element(h_input.begin(), h_input.end());
    float in_max = *std::max_element(h_input.begin(), h_input.end());
    std::cout << "输入范围: [" << in_min << ", " << in_max << "]\n\n";

    std::vector<float> h_online(numel);
    std::vector<float> h_naive(numel);
    std::vector<float> h_ref(numel);

    // ---------- 参考 (double 三遍) ----------
    auto t0 = std::chrono::high_resolution_clock::now();
    softmax_reference(h_input.data(), h_ref.data(), M, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ref_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---------- 朴素三遍 (float) ----------
    auto t2 = std::chrono::high_resolution_clock::now();
    softmax_naive_cpu(h_input.data(), h_naive.data(), M, N);
    auto t3 = std::chrono::high_resolution_clock::now();
    double naive_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // ---------- Online (一遍扫描) ----------
    auto t4 = std::chrono::high_resolution_clock::now();
    softmax_online_cpu(h_input.data(), h_online.data(), M, N);
    auto t5 = std::chrono::high_resolution_clock::now();
    double online_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

    // ---------- 验证 ----------
    auto check = [&](const float* result, const char* name) {
        double max_err = 0.0;
        for (size_t i = 0; i < numel; ++i) {
            double rel = fabs(static_cast<double>(result[i]) - static_cast<double>(h_ref[i]))
                       / fmax(1.0, static_cast<double>(h_ref[i]));
            if (rel > max_err) max_err = rel;
        }
        double max_sum_dev = 0.0;
        for (int r = 0; r < M; ++r) {
            double sum = 0.0;
            for (int c = 0; c < N; ++c) sum += result[r * N + c];
            double dev = fabs(sum - 1.0);
            if (dev > max_sum_dev) max_sum_dev = dev;
        }
        printf("  %-20s  相对误差: %8.2e  行和偏差: %8.2e\n", name, max_err, max_sum_dev);
    };

    // ---------- 打印 ----------
    printf("  方法                    耗时 (ms)       吞吐 (GB/s)\n");
    printf("  ---------------------------------------------------\n");
    printf("  %-20s  %10.3f      %8.2f\n", "参考 (double)", ref_ms, bytes / ref_ms / 1e6);
    printf("  %-20s  %10.3f      %8.2f\n", "朴素三遍 (float)", naive_ms, bytes / naive_ms / 1e6);
    printf("  %-20s  %10.3f      %8.2f\n", "Online (float)", online_ms, bytes / online_ms / 1e6);
    printf("\n");

    printf("精度对比 (vs double 参考):\n");
    check(h_naive.data(), "朴素三遍 (float)");
    check(h_online.data(), "Online (float)");

    // 额外对比: online vs naive 的直接差异
    double online_vs_naive = 0.0;
    for (size_t i = 0; i < numel; ++i) {
        double diff = fabs(static_cast<double>(h_online[i]) - static_cast<double>(h_naive[i]));
        if (diff > online_vs_naive) online_vs_naive = diff;
    }
    printf("\n  Online vs 朴素 最大绝对差: %8.2e\n", online_vs_naive);

    bool pass = true;
    for (size_t i = 0; i < numel; ++i) {
        double rel = fabs(static_cast<double>(h_online[i]) - static_cast<double>(h_ref[i]))
                   / fmax(1.0, static_cast<double>(h_ref[i]));
        if (rel >= 1e-5f) { pass = false; break; }
    }
    printf("\n>>> %s <<<\n", pass ? "PASS" : "FAIL");

    return pass ? 0 : 1;
}