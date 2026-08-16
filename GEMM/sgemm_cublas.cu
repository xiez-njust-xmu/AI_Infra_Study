/**
 * sgemm_cublas.cu
 *
 * 使用 cuBLAS 库的 cublasSgemm 做标准参考实现。
 * 用于与手写 kernel 做性能对比和正确性校验。
 *
 * 注意: cuBLAS 使用列主序 (column-major)，
 *       而我们的手写 kernel 使用行主序 (row-major)。
 *       这里用 cublasSgemm 计算 C = A × B (行主序语义)，
 *       需要将矩阵转置后传入: CublasOperation_t 用 CUBLAS_OP_N。
 */

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA 错误检查
#define CHECK_CUDA(err)                                                      \
    do {                                                                     \
        cudaError_t _err = (err);                                            \
        if (_err != cudaSuccess) {                                           \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__     \
                      << " - " << cudaGetErrorString(_err) << std::endl;     \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// cuBLAS 错误检查
#define CHECK_CUBLAS(err)                                                    \
    do {                                                                     \
        cublasStatus_t _err = (err);                                         \
        if (_err != CUBLAS_STATUS_SUCCESS) {                                 \
            std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__   \
                      << " - code " << _err << std::endl;                    \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

int main()
{
    // ---- 矩阵尺寸 ----
    const int M = 10240;
    const int N = 10240;
    const int K = 10240;

    size_t size_A = static_cast<size_t>(M) * K;
    size_t size_B = static_cast<size_t>(K) * N;
    size_t size_C = static_cast<size_t>(M) * N;

    std::cout << "===== cuBLAS SGEMM Reference =====" << std::endl;
    std::cout << "M=" << M << ", N=" << N << ", K=" << K << std::endl;

    // ---- 分配 Host 内存并初始化 ----
    float* h_A = new float[size_A];
    float* h_B = new float[size_B];
    float* h_C = new float[size_C];

    srand(42);
    for (size_t i = 0; i < size_A; i++)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (size_t i = 0; i < size_B; i++)
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // ---- 分配 Device 显存 ----
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, size_B * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, size_C * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice));

    // ---- 创建 cuBLAS handle ----
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // ---- 预热 ----
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,  // 不转置
                             M, N, K,                   // m, n, k
                             &alpha,
                             d_A, M,    // A (lda = M)
                             d_B, K,    // B (ldb = K)
                             &beta,
                             d_C, M));  // C (ldc = M)
    CHECK_CUDA(cudaDeviceSynchronize());

    // ---- 计时 ----
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    constexpr int NUM_ITER = 10;
    CHECK_CUDA(cudaEventRecord(start));

    for (int iter = 0; iter < NUM_ITER; iter++)
    {
        CHECK_CUBLAS(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 M, N, K,
                                 &alpha,
                                 d_A, M,
                                 d_B, K,
                                 &beta,
                                 d_C, M));
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float gpu_time_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_time_ms, start, stop));
    float avg_time_ms = gpu_time_ms / NUM_ITER;

    // ---- 结果回传 ----
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 性能输出 ----
    double flops = 2.0 * M * N * K;
    double tflops = flops / (avg_time_ms * 1e-3) / 1e12;

    std::cout << "\n===== cuBLAS Performance =====" << std::endl;
    std::cout << "GPU kernel time (avg of " << NUM_ITER << "): "
              << avg_time_ms << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;

    // ---- 打印前几个结果做采样 ----
    std::cout << "\nSample results (first 5x5):" << std::endl;
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
            printf("%8.4f ", h_C[i * N + j]);
        std::cout << std::endl;
    }

    // ---- 清理 ----
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}