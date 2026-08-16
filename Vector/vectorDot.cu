#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// 错误检查宏
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 内核：每个block计算一部分点积，结果存储在全局数组 partial_sums 中
__global__ void dot_product_kernel(const float *A, const float *B, float *partial_sums, int N) {
    extern __shared__ float sdata[];  // 共享内存，大小由启动时动态指定

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程加载一个元素对，计算乘积，存入共享内存
    float product = 0.0f;
    if (i < N) {
        product = A[i] * B[i];
    }
    sdata[tid] = product;

    __syncthreads();

    // 块内规约（树形加法）
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // 每个block的结果由线程0写入全局 partial_sums
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// 主机端计算点积的封装函数，返回结果
float dot_product(const float *A, const float *B, int N) {
    float *d_A, *d_B, *d_partial;
    float *h_partial;
    int block_size = 256;          // 每个block的线程数
    int grid_size = (N + block_size - 1) / block_size;  // block数量

    // 分配设备内存
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, grid_size * sizeof(float)));

    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice));

    // 启动内核，动态分配共享内存大小
    size_t shared_mem_size = block_size * sizeof(float);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 计时开始
    cudaEventRecord(start);

    dot_product_kernel<<<grid_size, block_size, shared_mem_size>>>(d_A, d_B, d_partial, N);
    // 计时结束
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算耗时
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Kernel time: %.4f ms\n", milliseconds);

    CUDA_CHECK(cudaDeviceSynchronize());

    // 将每个block的部分和拷贝回主机
    h_partial = (float*)malloc(grid_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, grid_size * sizeof(float), cudaMemcpyDeviceToHost));

    // 主机端求和得到最终结果
    float result = 0.0f;
    for (int i = 0; i < grid_size; i++) {
        result += h_partial[i];
    }

    // 释放资源
    free(h_partial);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_partial));

    return result;
}

// ---------- 主函数测试 ----------
int main() {
    const int N = 1<<28;  // 向量长度
    float *h_A, *h_B;

    // 分配主机内存
    h_A = (float*)malloc(N * sizeof(float));
    h_B = (float*)malloc(N * sizeof(float));

    // 初始化数据：随机值
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)(rand() % 100) / 10.0f;
        h_B[i] = (float)(rand() % 100) / 10.0f;
    }

    // ---------- 测试点积 ----------
    float result = dot_product(h_A, h_B, N);
    // 主机端验证
    float expected = 0.0f;
    for (int i = 0; i < N; i++) {
        expected += h_A[i] * h_B[i];
    }
    printf("Dot Product: GPU result = %f, CPU result = %f, diff = %e\n", result, expected, fabs(result - expected));

    // 释放主机内存
    free(h_A);
    free(h_B);
    return 0;
}