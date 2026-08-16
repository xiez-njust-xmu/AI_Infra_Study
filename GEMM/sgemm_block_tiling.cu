#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

template <int BM, int BN, int BK, int BLOCK_SIZE>
__global__ void sgemm_block_tiling(float* A, float* B, float* C,
                                       int M, int K, int N)
{
    // 共享内存: A-tile [BM][BK], B-tile [BK][BN]
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // 当前线程块在全局矩阵左上角坐标
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int tid = threadIdx.x;

    // --------------------------
    // 1. 配置加载A的线程拆分，适配 As[BM][BK]
    // --------------------------
    constexpr int A_LOAD_X = BK;
    constexpr int A_LOAD_Y = BLOCK_SIZE / A_LOAD_X;
    int a_x = tid % A_LOAD_X;
    int a_y = tid / A_LOAD_X;

    // --------------------------
    // 2. 配置加载B的线程拆分，适配 Bs[BK][BN]，y范围严格 0~BK-1
    // --------------------------
    constexpr int B_LOAD_Y = BK;
    constexpr int B_LOAD_X = BLOCK_SIZE / B_LOAD_Y;
    int b_x = tid % B_LOAD_X;
    int b_y = tid / B_LOAD_X;

    // --------------------------
    // 3. 计算输出C块的线程划分
    // BLOCK_SIZE=1024, C tile划分 32*32线程
    // --------------------------
    constexpr int C_THREAD_X = 32;
    constexpr int C_THREAD_Y = BLOCK_SIZE / C_THREAD_X;
    int c_x = tid % C_THREAD_X;
    int c_y = tid / C_THREAD_X;

    // 每个线程负责 Tm × Tn 个结果元素
    constexpr int Tm = BM / C_THREAD_Y;
    constexpr int Tn = BN / C_THREAD_X;

    // 寄存器缓存输出
    float Ct[Tm][Tn] = {0.0f};

    // K方向主循环，步长BK
    for(int k0 = 0; k0 < K; k0 += BK)
    {
        // 分批次加载 A -> shared As
        #pragma unroll
        for(int i = a_y; i < BM; i += A_LOAD_Y)
        {
            int global_r = block_row + i;
            int global_c = k0 + a_x;
            As[i][a_x] = (global_r < M && global_c < K) ? A[global_r * K + global_c] : 0.0f;
        }

        // 分批次加载 B -> shared Bs
        #pragma unroll
        for(int j = b_x; j < BN; j += B_LOAD_X)
        {
            int global_r = k0 + b_y;
            int global_c = block_col + j;
            Bs[b_y][j] = (global_r < K && global_c < N) ? B[global_r * N + global_c] : 0.0f;
        }

        // 等待全部线程完成共享内存加载
        __syncthreads();

        // 外积累加 As * Bs 到寄存器Ct
        #pragma unroll
        for(int p = 0; p < BK; p++)
        {
            #pragma unroll
            for(int ti = 0; ti < Tm; ti++)
            {
                int row_idx = c_y + ti * C_THREAD_Y;
                #pragma unroll
                for(int tj = 0; tj < Tn; tj++)
                {
                    int col_idx = c_x + tj * C_THREAD_X;
                    Ct[ti][tj] += As[row_idx][p] * Bs[p][col_idx];
                }
            }
        }

        // 此处不需要多余syncthreads，下一迭代会覆盖共享内存
    }

    // 将寄存器Ct写回全局内存C
    #pragma unroll
    for(int ti = 0; ti < Tm; ti++)
    {
        int r = block_row + c_y + ti * C_THREAD_Y;
        #pragma unroll
        for(int tj = 0; tj < Tn; tj++)
        {
            int c = block_col + c_x + tj * C_THREAD_X;
            if(r < M && c < N)
            {
                C[r * N + c] = Ct[ti][tj];
            }
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

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int BLOCK_SIZE = 1024;

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
    sgemm_block_tiling<BM, BN, BK, BLOCK_SIZE><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
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