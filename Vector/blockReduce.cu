#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// 错误检查宏
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, code:%d, reason:%s\n", \
               __FILE__, __LINE__, error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// Warp 内归约求和（使用 shuffle 指令）
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// 每个 Block 归约内核
__global__ void block_reduce_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 每个线程加载一个元素（越界补 0）
    float val = (tid < N) ? input[tid] : 0.0f;

    // 第一步：Warp 内归约
    val = warp_reduce_sum(val);

    // 第二步：每个 Warp 的 lane 0 将结果写入共享内存
    __shared__ float warp_sums[32];  // 一个 Block 最多 1024 线程 → 最多 32 个 Warp
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;

    if (laneId == 0) {
        warp_sums[warpId] = val;
    }
    __syncthreads();

    // 第三步：第一个 Warp 对 warp_sums 做最终归约
    int numWarps = blockDim.x / 32;
    val = (threadIdx.x < numWarps) ? warp_sums[threadIdx.x] : 0.0f;

    if (warpId == 0) {
        val = warp_reduce_sum(val);
    }

    // lane 0 写出本 Block 的归约结果
    if (threadIdx.x == 0) {
        output[blockIdx.x] = val;
    }
}

int main() {
    // 数据规模
    const int N = 1 << 20;  // 约 2.68 亿个元素
    size_t bytes_input = N * sizeof(float);

    // 生成随机输入数据
    float* h_input = (float*)malloc(bytes_input);
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)(rand() % 100) / 10.0f;  // 0.0 ~ 9.9
    }

    // CPU 参考求和
    float cpu_sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        cpu_sum += h_input[i];
    }
    printf("CPU sum = %f\n", cpu_sum);

    // 分配设备内存
    float *d_input, *d_output;
    CHECK(cudaMalloc(&d_input, bytes_input));

    // 配置内核启动参数
    int blockSize = 256;                // 可调整，建议为 32 的倍数
    int gridSize = (N + blockSize - 1) / blockSize;
    size_t bytes_output = gridSize * sizeof(float);
    CHECK(cudaMalloc(&d_output, bytes_output));

    // 拷贝数据到设备
    CHECK(cudaMemcpy(d_input, h_input, bytes_input, cudaMemcpyHostToDevice));

    // 启动内核
    block_reduce_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // 将每个 Block 的和拷贝回主机
    float* h_output = (float*)malloc(bytes_output);
    CHECK(cudaMemcpy(h_output, d_output, bytes_output, cudaMemcpyDeviceToHost));

    // GPU 最终求和（所有 Block 的和）
    float gpu_sum = 0.0f;
    for (int i = 0; i < gridSize; ++i) {
        gpu_sum += h_output[i];
    }
    printf("GPU sum = %f\n", gpu_sum);

    // 验证结果
    float diff = fabs(cpu_sum - gpu_sum);
    printf("Difference = %e\n", diff);
    if (diff < 1e-5) {
        printf("Test PASSED\n");
    } else {
        printf("Test FAILED\n");
    }

    // 释放资源
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}