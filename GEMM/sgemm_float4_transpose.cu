/**
 * sgemm_float4_transpose.cu
 *
 * 核心优化点:
 *   1. float4 向量化访存 — 每次加载/写入 4 个 float，减少指令数、提高带宽利用率
 *   2. A 在 Shared Memory 中转置存储 (As_T[BK][BM] 而非 As[BM][BK])
 *      — 内层循环读取 As_T[k][row] 是连续地址，消除 bank conflict
 *   3. 坐标映射随 float4 的变化:
 *      - 原来每个线程加载 1 个 float，现每个线程加载 1 个 float4 (4 floats)
 *      - 加载 A 的线程数从 BM*BK 降到 BM*BK/4
 *      - 加载 B 的线程数从 BK*BN 降到 BK*BN/4
 *      - 要求 K、N 是 4 的倍数以保证 float4 16 字节对齐
 */

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// =========================================================================
//  Kernel: float4 向量化 + A 转置共享存储
// =========================================================================
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__(BM / TM * BN / TN)
sgemm_float4_transpose(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C,
                       int M, int N, int K)
{
    // ---- 共享内存 ----
    // A 转置存储: 内层循环 k 维在外，读 As_T[k][row] 是连续地址
    __shared__ float As_T[BK][BM];
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int by  = blockIdx.y;
    const int bx  = blockIdx.x;

    // 每个线程在 C-tile 中负责 TM × TN 子块
    const int thread_row = (tid / (BN / TN)) * TM;   // 行起始 (0, TM, 2TM, ...)
    const int thread_col = (tid % (BN / TN)) * TN;   // 列起始 (0, TN, 2TN, ...)

    // 寄存器片段
    float a_frag[TM];
    float b_frag[TN];
    float c_frag[TM][TN] = {0.0f};

    const int block_row = by * BM;
    const int block_col = bx * BN;

    // ================================================================
    //  K 方向主循环，步长 BK
    // ================================================================
    for (int bk = 0; bk < K; bk += BK)
    {
        // ------------------------------------------------------------
        //  1. float4 加载 A tile → 转置存 As_T[BK][BM]
        //
        //  坐标映射:
        //    总需加载 BM*BK 个 float = BM*BK/4 个 float4
        //    每个线程 (tid) 负责 1 个 float4
        //    r  = tid / (BK/4)     → 0..BM-1, 该 tile 中的行
        //    c4 = tid % (BK/4)     → 0..BK/4-1, 该行中第几个 float4 组
        //    全局: g_row = block_row + r
        //          g_col = bk + c4 * 4
        //    转置存: As_T[c4*4 + 0..3][r] = v.x, v.y, v.z, v.w
        // ------------------------------------------------------------
        {
            constexpr int F4_PER_ROW = BK / 4;          // 每行 float4 组数
            int r  = tid / F4_PER_ROW;                  // tile 内行
            int c4 = tid % F4_PER_ROW;                  // 该行第几个 float4
            int g_row = block_row + r;
            int g_col = bk + c4 * 4;

            if (g_row < M && g_col + 3 < K)
            {
                // float4 向量化加载（需 16 字节对齐）
                float4 v = reinterpret_cast<const float4*>(
                    &A[static_cast<size_t>(g_row) * K + g_col])[0];
                As_T[c4 * 4 + 0][r] = v.x;
                As_T[c4 * 4 + 1][r] = v.y;
                As_T[c4 * 4 + 2][r] = v.z;
                As_T[c4 * 4 + 3][r] = v.w;
            }
            else if (g_row < M)
            {
                // 边界: K 方向不足 4 个元素，逐元素加载
                #pragma unroll
                for (int d = 0; d < 4; d++)
                {
                    As_T[c4 * 4 + d][r] = (g_col + d < K)
                        ? A[static_cast<size_t>(g_row) * K + g_col + d]
                        : 0.0f;
                }
            }
            else
            {
                // 行越界，填 0
                As_T[c4 * 4 + 0][r] = 0.0f;
                As_T[c4 * 4 + 1][r] = 0.0f;
                As_T[c4 * 4 + 2][r] = 0.0f;
                As_T[c4 * 4 + 3][r] = 0.0f;
            }
        }

        // ------------------------------------------------------------
        //  2. float4 加载 B tile → Bs[BK][BN]
        //
        //  坐标映射:
        //    总需加载 BK*BN 个 float = BK*BN/4 个 float4
        //    每个线程 1 个 float4
        //    r  = (tid * 4) / BN      → 0..BK-1
        //    c  = (tid * 4) % BN      → 0..BN-1, 4 对齐
        //    全局: g_row = bk + r
        //          g_col = block_col + c
        //    存:  Bs[r][c + 0..3] = v.x, v.y, v.z, v.w  (连续，无冲突)
        // ------------------------------------------------------------
        {
            int r = (tid * 4) / BN;            // 0..BK-1
            int c = (tid * 4) % BN;            // 0, 4, 8, ..., BN-4
            int g_row = bk + r;
            int g_col = block_col + c;

            if (g_row < K && g_col + 3 < N)
            {
                float4 v = reinterpret_cast<const float4*>(
                    &B[static_cast<size_t>(g_row) * N + g_col])[0];
                Bs[r][c + 0] = v.x;
                Bs[r][c + 1] = v.y;
                Bs[r][c + 2] = v.z;
                Bs[r][c + 3] = v.w;
            }
            else if (g_row < K)
            {
                #pragma unroll
                for (int d = 0; d < 4; d++)
                {
                    Bs[r][c + d] = (g_col + d < N)
                        ? B[static_cast<size_t>(g_row) * N + g_col + d]
                        : 0.0f;
                }
            }
            else
            {
                Bs[r][c + 0] = 0.0f;
                Bs[r][c + 1] = 0.0f;
                Bs[r][c + 2] = 0.0f;
                Bs[r][c + 3] = 0.0f;
            }
        }

        __syncthreads();

        // ------------------------------------------------------------
        //  3. 外积累加: C_frag += As_T × Bs
        //
        //  因为 A 转置存储，内层循环读 As_T[k][row] 是连续地址：
        //    - 同一 warp 内相邻线程的 thread_row 连续 → 读同一行不同列
        //    - 合并访问，无 bank conflict
        //  同样 Bs[k][col] 也是连续读取，无 bank conflict
        // ------------------------------------------------------------
        #pragma unroll
        for (int k = 0; k < BK; k++)
        {
            // 从 As_T 加载 a_frag — 连续读同一行 k 的不同列
            #pragma unroll
            for (int i = 0; i < TM; i++)
                a_frag[i] = As_T[k][thread_row + i];

            // 从 Bs 加载 b_frag — 连续读同一行 k 的不同列
            #pragma unroll
            for (int j = 0; j < TN; j++)
                b_frag[j] = Bs[k][thread_col + j];

            // 寄存器外积
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    c_frag[i][j] += a_frag[i] * b_frag[j];
        }

        __syncthreads();
    }

    // ----------------------------------------------------------------
    //  4. 写回结果 C  — 使用 float4 向量化写回
    //
    //  每个线程的 TM×TN 子块中，每行 TN=8 个元素，拆成 2 个 float4
    // ----------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < TM; i++)
    {
        int r = block_row + thread_row + i;
        if (r >= M) continue;

        int c = block_col + thread_col;
        size_t idx = static_cast<size_t>(r) * N + c;

        if (c + 7 < N)
        {
            // float4 向量化写: 每行 2 个 float4
            float4 v0 = make_float4(c_frag[i][0], c_frag[i][1],
                                    c_frag[i][2], c_frag[i][3]);
            float4 v1 = make_float4(c_frag[i][4], c_frag[i][5],
                                    c_frag[i][6], c_frag[i][7]);
            reinterpret_cast<float4*>(&C[idx])[0] = v0;
            reinterpret_cast<float4*>(&C[idx])[1] = v1;
        }
        else
        {
            // 边界: N 方向不足 8 个元素，逐元素写
            #pragma unroll
            for (int j = 0; j < TN; j++)
            {
                if (c + j < N)
                    C[idx + j] = c_frag[i][j];
            }
        }
    }
}

// =========================================================================
//  CPU 参考实现 (结果校验)
// =========================================================================
void cpu_sgemm(const float* A, const float* B, float* C,
               int M, int N, int K)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[static_cast<size_t>(i) * K + k]
                     * B[static_cast<size_t>(k) * N + j];
            C[static_cast<size_t>(i) * N + j] = sum;
        }
}

// =========================================================================
//  CUDA 错误检查封装
// =========================================================================
#define CHECK_CUDA(err)                                                      \
    do {                                                                     \
        cudaError_t _err = (err);                                            \
        if (_err != cudaSuccess) {                                           \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__     \
                      << " - " << cudaGetErrorString(_err) << std::endl;     \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// =========================================================================
//  main
// =========================================================================
int main()
{
    // ---- 矩阵尺寸 ----
    // 注意: float4 需要 K 和 N 是 4 的倍数以保证 16 字节对齐
    const int M = 10240;
    const int N = 10240;
    const int K = 10240;

    // ---- 调优参数 ----
    constexpr int BM = 128;          // C-tile 行 (A 的行)
    constexpr int BN = 128;          // C-tile 列 (B 的列)
    constexpr int BK = 8;            // K 分块大小
    constexpr int TM = 8;            // 每线程计算 C 的行数
    constexpr int TN = 8;            // 每线程计算 C 的列数
    constexpr int BLOCK_SIZE = BM / TM * BN / TN;  // = 256

    // ---- 检查对齐条件 ----
    if (K % 4 != 0 || N % 4 != 0)
    {
        std::cerr << "Error: K and N must be multiples of 4 for float4 alignment.\n";
        return 1;
    }

    std::cout << "===== SGEMM float4 + Transpose =====" << std::endl;
    std::cout << "M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "BM=" << BM << ", BN=" << BN << ", BK=" << BK
              << ", TM=" << TM << ", TN=" << TN << std::endl;

    // ---- 分配 Host 内存 ----
    size_t size_A = static_cast<size_t>(M) * K;
    size_t size_B = static_cast<size_t>(K) * N;
    size_t size_C = static_cast<size_t>(M) * N;

    float* h_A = new float[size_A];
    float* h_B = new float[size_B];
    float* h_C_gpu = new float[size_C];
    float* h_C_cpu = new float[size_C];

    // 随机初始化
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

    // ---- 预热 ----
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_float4_transpose<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // ---- 计时 ----
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    constexpr int NUM_ITER = 10;
    CHECK_CUDA(cudaEventRecord(start));

    for (int iter = 0; iter < NUM_ITER; iter++)
    {
        sgemm_float4_transpose<BM, BN, BK, TM, TN>
            <<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float gpu_time_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_time_ms, start, stop));
    float avg_time_ms = gpu_time_ms / NUM_ITER;

    // ---- 结果回传 ----
    CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, size_C * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // ---- 计算结果校验 ----
    // 取小规模子矩阵做 CPU 校验，避免全量 O(N³) 太慢
    constexpr int CHECK_M = 256, CHECK_N = 256;
    std::cout << "\nVerifying with " << CHECK_M << "×" << CHECK_N
              << " submatrix ..." << std::endl;

    cpu_sgemm(h_A, h_B, h_C_cpu, CHECK_M, CHECK_N, K);

    float max_err = 0.0f;
    int err_count = 0;
    for (int i = 0; i < CHECK_M; i++)
    {
        for (int j = 0; j < CHECK_N; j++)
        {
            float diff = fabs(h_C_gpu[static_cast<size_t>(i) * N + j]
                            - h_C_cpu[static_cast<size_t>(i) * N + j]);
            if (diff > max_err) max_err = diff;
            if (diff > 1e-3f) err_count++;
        }
    }

    // ---- 输出 ----
    double flops = 2.0 * M * N * K;
    double tflops = flops / (avg_time_ms * 1e-3) / 1e12;

    std::cout << "\n===== Result =====" << std::endl;
    std::cout << "GPU kernel time (avg of " << NUM_ITER << "): "
              << avg_time_ms << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;
    std::cout << "Max error (submatrix): " << max_err << std::endl;
    if (err_count > 0)
        std::cout << "WARNING: " << err_count << " elements exceed 1e-3 error threshold"
                  << std::endl;
    else
        std::cout << "✓ Correctness check passed" << std::endl;

    // ---- 清理 ----
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C_gpu;
    delete[] h_C_cpu;

    return 0;
}