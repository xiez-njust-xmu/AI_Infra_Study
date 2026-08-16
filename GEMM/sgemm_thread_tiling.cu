#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cstdio>


template<int BM, int BK>
__device__ void load_tile_A(const float* A, float As[BM][BK], int by, int bk, int tid, int M, int K)
{
    int base_row = by * BM;
    int base_col = bk;
    constexpr int load_elem = BM * BK;
    for(int i = tid; i < load_elem; i += 256)
    {
        int r = i / BK;
        int c = i % BK;
        int g_row = base_row + r;
        int g_col = base_col + c;
        As[r][c] = ((g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f);
    }
}

template<int BN, int BK>
__device__ void load_tile_B(const float* B, float Bs[BK][BN], int bx, int bk, int tid, int K, int N)
{
    int base_row = bk;
    int base_col = bx * BN;
    constexpr int load_elem = BK * BN;
    for(int i = tid; i < load_elem; i += 256)
    {
        int r = i / BN;
        int c = i % BN;
        int g_row = base_row + r;
        int g_col = base_col + c;
        Bs[r][c] = ((g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f);
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_thread_tiling(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // 每个线程在 C 中负责 TM×TN 的子块
    const int thread_num = BM / TM * BN / TN;  // = 256
    int tid = threadIdx.x;
    int thread_row = (tid / (BN / TN)) * TM;
    int thread_col = (tid % (BN / TN)) * TN;

    // 寄存器存储
    float a_frag[TM];
    float b_frag[TN];
    float c_frag[TM][TN] = {0.0f};

    int by = blockIdx.y, bx = blockIdx.x;

    for (int bk = 0; bk < K; bk += BK) {
        // 协作加载 A、B 到 Shared Memory（省略边界检查）
        load_tile_A<BM,BK>(A, As, by, bk, tid, M, K);
        load_tile_B<BN,BK>(B, Bs, bx, bk, tid, K, N);
        __syncthreads();

        // 外积累加
        for (int k = 0; k < BK; k++) {
            // 从 Shared Memory 加载到寄存器
            for (int i = 0; i < TM; i++)
                a_frag[i] = As[thread_row + i][k];
            for (int j = 0; j < TN; j++)
                b_frag[j] = Bs[k][thread_col + j];

            // 寄存器上做外积
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    c_frag[i][j] += a_frag[i] * b_frag[j];
        }
        __syncthreads();
    }

    // 写回 Global Memory
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            C[(by * BM + thread_row + i) * N + bx * BN + thread_col + j] = c_frag[i][j];
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

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr int BK = 8;
    constexpr int BLOCK_SIZE = 256;

    // 1. 分配CPU内存并初始化随机矩阵
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C_gpu = new float[M * N];

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
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((N + BN - 1) / BN,
                 (M + BM - 1) / BM);

    // 启动计时
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    // 调用朴素SGEMM核函数
    sgemm_thread_tiling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N ,K);
    CHECK_CUDA_ERROR(cudaGetLastError()); // 检查核启动错误

    // 结束计时并同步等待GPU计算完成
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    // 计算GPU运行耗时 ms
    float gpu_time_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time_ms, start, stop));

    // 4. 结果回传 Device -> Host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // 输出结果
    std::cout << "===== Naive SGEMM CUDA Timing Result =====" << std::endl;
    std::cout << "Matrix size M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "GPU kernel execution time: " << gpu_time_ms << " ms" << std::endl;

    // 释放资源
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C_gpu;
    return 0;
}