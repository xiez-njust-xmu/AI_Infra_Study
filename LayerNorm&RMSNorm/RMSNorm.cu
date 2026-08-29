#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdio>

__inline__ __device__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffff;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

__device__ float block_reduce_sum(float v, float* smem) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    v = warp_reduce_sum(v);

    if (lane == 0) {
        smem[warp_id] = v;
    }
    __syncthreads();

    float out = 0.0f;
    if (warp_id == 0) {
        out = (tid < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
        out = warp_reduce_sum(out);
    }

    if (tid == 0) {
        smem[0] = out;
    }
    __syncthreads();
    return smem[0];
}

__global__ void rmsnorm_kernel(const float* x,
                               const float* weight,
                               float* y,
                               int rows,
                               int hidden,
                               float eps) {
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    const float* row_x = x + row * hidden;
    float* row_y = y + row * hidden;

    // 求平方和。
    float local_sq_sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float v = row_x[i];
        local_sq_sum += v * v;
    }

    float sq_sum = block_reduce_sum(local_sq_sum, smem);
    float inv_rms = rsqrtf(sq_sum / hidden + eps);

    // 写出结果。
    for (int i = tid; i < hidden; i += blockDim.x) {
        row_y[i] = row_x[i] * inv_rms * weight[i];
    }
}

// CPU参考实现 RMSNorm
void rmsnorm_cpu(const float* x,
                 const float* weight,
                 float* y,
                 int rows,
                 int hidden,
                 float eps)
{
    for(int r = 0; r < rows; r++){
        const float* rx = x + r * hidden;
        float* ry = y + r * hidden;

        float sq_sum = 0.0f;
        for(int i = 0; i < hidden; i++){
            float v = rx[i];
            sq_sum += v * v;
        }
        float inv_rms = rsqrtf(sq_sum / hidden + eps);

        for(int i = 0; i < hidden; i++){
            ry[i] = rx[i] * inv_rms * weight[i];
        }
    }
}

float check_error(const float* a, const float* b, int n)
{
    float max_err = 0.0f;
    for(int i = 0; i < n; i++){
        float err = fabs(a[i] - b[i]);
        if(err > max_err) max_err = err;
    }
    return max_err;
}

int main()
{
    const int rows = 128;
    const int hidden = 1024;
    const float eps = 1e-5f;
    const int block_size = 128;
    int smem_size = (block_size / 32) * sizeof(float);

    int total_elem = rows * hidden;

    std::vector<float> h_x(total_elem);
    std::vector<float> h_weight(hidden);
    std::vector<float> h_y_cpu(total_elem);
    std::vector<float> h_y_gpu(total_elem);

    srand(42);
    for(int i = 0; i < total_elem; i++){
        h_x[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
    for(int i = 0; i < hidden; i++){
        h_weight[i] = (rand() / (float)RAND_MAX) * 1.0f;
    }

    // CPU计时
    auto cpu_start = std::chrono::high_resolution_clock::now();
    rmsnorm_cpu(h_x.data(), h_weight.data(), h_y_cpu.data(), rows, hidden, eps);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double,std::milli>(cpu_end - cpu_start).count();

    // GPU内存
    float *d_x, *d_weight, *d_y;
    cudaMalloc(&d_x,      total_elem * sizeof(float));
    cudaMalloc(&d_weight, hidden * sizeof(float));
    cudaMalloc(&d_y,      total_elem * sizeof(float));

    cudaMemcpy(d_x,      h_x.data(),     total_elem*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), hidden*sizeof(float),    cudaMemcpyHostToDevice);

    dim3 grid(rows);
    dim3 block(block_size);
    // warmup
    rmsnorm_kernel<<<grid, block, smem_size>>>(d_x, d_weight, d_y, rows, hidden, eps);
    cudaDeviceSynchronize();

    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);

    cudaEventRecord(gpu_start);
    rmsnorm_kernel<<<grid, block, smem_size>>>(d_x, d_weight, d_y, rows, hidden, eps);
    cudaEventRecord(gpu_end);
    cudaDeviceSynchronize();

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_end);

    cudaMemcpy(h_y_gpu.data(), d_y, total_elem*sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = check_error(h_y_cpu.data(), h_y_gpu.data(), total_elem);

    std::cout << "rows=" << rows << ", hidden=" << hidden << "\n";
    std::cout << "CPU time:      " << cpu_ms << " ms\n";
    std::cout << "GPU kernel time:" << gpu_ms << " ms\n";
    std::cout << "Max absolute error: " << max_err << "\n";
    if(max_err < 1e-3f){
        std::cout << "Result PASS\n";
    }else{
        std::cout << "Result WARNING: large error\n";
    }

    cudaFree(d_x);
    cudaFree(d_weight);
    cudaFree(d_y);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_end);

    return 0;
}
