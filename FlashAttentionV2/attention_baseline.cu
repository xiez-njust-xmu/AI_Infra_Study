#include "utils.h"
#include "attention.cuh"
#include <assert.h>
#include <cstdio>

void timingAttn(const float *Q, const float *K, const float *V, const int batch_size, const int num_head,
    const int N, const int M, const int d, float *QK, float *QK_softmax, float *O)
{
    constexpr int REPEAT_NUM = 1;
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < REPEAT_NUM; ++i)
    {
        attention::launchAttentionBaseline(Q, K, V, QK, QK_softmax, O, batch_size, num_head, N, M, d);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("alogrithm: attention baseline bz(%d) nh(%d) N(%d) M(%d) d(%d), elapsed_time: %g ms\n", 
        batch_size, num_head, N, M, d, elapsed_time / REPEAT_NUM);
}
int main(int argc, char *argv[])
{
    constexpr int batch_size = 32;
    constexpr int num_head = 8;
    constexpr int N = 1024;
    constexpr int M = 1024;
    constexpr int d = 128;

    float *Q = new float[batch_size * num_head * N * d];
    float *K = new float[batch_size * num_head * M * d];
    float *V = new float[batch_size * num_head * M * d];
    float *O = new float[batch_size * num_head * N * d];

    // srand(1024);
    // for (int i = 0; i < batch_size * num_head * N * d; ++i)
    // {
    //     Q[i] = (rand() / (RAND_MAX + 1.0f)) * 1.0f - 0.5f;
    //     O[i] = 0.0f;
    // }

    // for (int i = 0; i < batch_size * num_head * M * d; ++i)
    // {
    //     K[i] = (rand() / (RAND_MAX + 1.0f)) * 1.0f - 0.5f;
    //     V[i] = (rand() / (RAND_MAX + 1.0f)) * 1.0f - 0.5f;
    // }

    // for (int i = 0; i < batch_size * num_head * N * d; ++i)
    // {
    //     Q[i] = i % 1003 - 500.0f;
    //     O[i] = 0.0f;
    // }

    // for (int i = 0; i < batch_size * num_head * M * d; ++i)
    // {
    //     K[i] = i % 2157 - 1218.1f;
    //     V[i] = i % 191 - 100.9f;
    // }

     // 初始化Q矩阵
    for (size_t i = 0; i < batch_size * num_head * N * d; ++i)
    {
        Q[i] = static_cast<float>(static_cast<int>(i * 41 % 2001) * 0.01f - 10.0f);
        O[i] = 0.0f;
    }

    // 初始化K矩阵（使用不同周期）
    for (size_t i = 0; i < batch_size * num_head * M * d; ++i)
    {
        K[i] = static_cast<float>((static_cast<int>(i % 211) - 105) * 0.095f);         // 211是质数
        V[i] = static_cast<float>(static_cast<int>(i * 53 % 1999) * 0.01f - 10.0f); // 503是质数
    }

    printMatrix(Q, (char *)("Matrix Q: "), N, d, 32, 32, 28, 24);
    printMatrix(K, (char *)("Matrix K: "), M, d, 32, 32, 28, 24);
    printMatrix(V, (char *)("Matrix V: "), M, d, 32, 32, 28, 24);

    float *d_Q;
    float *d_K;
    float *d_V;
    float *d_QK;
    float *d_QK_softmax;
    float *d_O;
    
    size_t mem_size = sizeof(float) * (batch_size * num_head * (N + M) * d * 2 + batch_size * num_head * N * M * 2);
    printf("requested global memory: %g GB \n", mem_size / 1024.0f / 1024.0f / 1024.0f);

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_Q, mem_size));
    d_K = d_Q + batch_size * num_head * N * d;
    d_V = d_K + batch_size * num_head * M * d;
    d_QK = d_V + batch_size * num_head * M * d;
    d_QK_softmax = d_QK + batch_size * num_head * N * M;
    d_O = d_QK_softmax + batch_size * num_head * N * M;

    CHECK_CUDA_ERROR(cudaMemcpy(d_Q, Q, sizeof(float) * batch_size * num_head * N * d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_K, K, sizeof(float) * batch_size * num_head * M * d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_V, V, sizeof(float) * batch_size * num_head * M * d, cudaMemcpyHostToDevice));

    timingAttn(d_Q, d_K, d_V, batch_size, num_head, N, M, d, d_QK, d_QK_softmax, d_O);

    CHECK_CUDA_ERROR(cudaMemcpy(O, d_O, sizeof(float) * batch_size * num_head * N * d, cudaMemcpyDeviceToHost));
    printMatrix(O, (char *)("Matrix output: "), N, d, 32, 32, 28, 24);

    CHECK_CUDA_ERROR(cudaFree(d_Q));
    delete [] Q;
    delete [] K;
    delete [] V;
    delete [] O;

    return 0;
}
