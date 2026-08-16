#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// 朴素SGEMM GPU核函数
__global__ void sgemm_naive(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CPU串行SGEMM 用于结果校验
void cpu_sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// CUDA错误检查封装
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at line " << __LINE__ \
                  << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

int main() {
    // 矩阵尺寸，可自行修改
    const int M = 10240;
    const int N = 10240;
    const int K = 10240;

    // 1. 分配CPU内存并初始化随机矩阵
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C_gpu = new float[M * N];
    float* h_C_cpu = new float[M * N];

    srand(42);
    for (int i = 0; i < M * K; i++)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; i++)
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // 2. 分配GPU显存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * N * sizeof(float)));

    // 数据拷贝 Host -> Device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // 3. CUDA事件创建，用于精确计时GPU核执行时间
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // 线程块配置 32x32 线程
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // 启动计时
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    // 调用朴素SGEMM核函数
    sgemm_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA_ERROR(cudaGetLastError()); // 检查核启动错误

    // 结束计时并同步等待GPU计算完成
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    // 计算GPU运行耗时 ms
    float gpu_time_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time_ms, start, stop));

    // 4. 结果回传 Device -> Host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // 5. CPU计算参考结果校验
    //cpu_sgemm(h_A, h_B, h_C_cpu, M, N, K);

    // 计算最大误差
    // float max_err = 0.0f;
    // for (int i = 0; i < M * N; i++) {
    //     float diff = fabs(h_C_gpu[i] - h_C_cpu[i]);
    //     if (diff > max_err) max_err = diff;
    // }

    // 输出结果
    std::cout << "===== Naive SGEMM CUDA Timing Result =====" << std::endl;
    std::cout << "Matrix size M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "GPU kernel execution time: " << gpu_time_ms << " ms" << std::endl;
    //std::cout << "Max absolute error between GPU & CPU: " << max_err << std::endl;

    // 释放资源
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C_gpu;
    delete[] h_C_cpu;

    return 0;
}