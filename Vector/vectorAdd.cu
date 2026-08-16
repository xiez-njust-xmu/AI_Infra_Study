#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void vectorAddStride(const float* A, const float* B,
                                float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查：防止越界访问
    if (idx < N) {
        C[idx] = A[idx] * 2.5f + B[idx];
    }
}

void vectorAddCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}


int main() {
    int N = 1 << 28;  // 约 100 万个元素
    size_t size = N * sizeof(float);

    // 1. 分配主机内存
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // 初始化输入数据
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }

    // 2. 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 3. 将输入数据从 Host 拷贝到 Device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 4. 配置并启动 Kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 计时开始
    cudaEventRecord(start);

    vectorAddStride<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // 计时结束
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算耗时
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Kernel time: %.4f ms\n", milliseconds);


    // 5. 将结果从 Device 拷回 Host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 6. 验证结果
    for (int i = 0; i < N; i++) {
        if (fabsf(h_C[i] - (h_A[i] + h_B[i])) > 1e-5f) {
            printf("Verification FAILED at index %d!\n", i);
            break;
        }
    }
    printf("Verification PASSED!\n");

    // 7. 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
