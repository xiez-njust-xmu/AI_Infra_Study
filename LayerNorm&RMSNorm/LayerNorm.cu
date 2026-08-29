#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdio>

// -------------------- GPU 规约函数（你原来的代码） --------------------
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

__global__ void layernorm_kernel(const float* x,
                                 const float* gamma,
                                 const float* beta,
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

    // 第一次规约：求 sum。
    float local_sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        local_sum += row_x[i];
    }
    float sum = block_reduce_sum(local_sum, smem);
    float mean = sum / hidden;

    // 第二次规约：求方差。
    float local_var = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float diff = row_x[i] - mean;
        local_var += diff * diff;
    }
    float var_sum = block_reduce_sum(local_var, smem);
    float inv_std = rsqrtf(var_sum / hidden + eps);

    // 写出归一化结果。
    for (int i = tid; i < hidden; i += blockDim.x) {
        float norm = (row_x[i] - mean) * inv_std;
        row_y[i] = norm * gamma[i] + beta[i];
    }
}

// -------------------- CPU 参考实现 --------------------
void layernorm_cpu(const float* x,
                   const float* gamma,
                   const float* beta,
                   float* y,
                   int rows,
                   int hidden,
                   float eps)
{
    for(int r = 0; r < rows; r++){
        const float* rx = x + r * hidden;
        float* ry = y + r * hidden;

        // mean
        float sum = 0.0f;
        for(int i = 0; i < hidden; i++) sum += rx[i];
        float mean = sum / hidden;

        // var
        float var_sum = 0.0f;
        for(int i = 0; i < hidden; i++){
            float diff = rx[i] - mean;
            var_sum += diff * diff;
        }
        float inv_std = rsqrtf(var_sum / hidden + eps);

        for(int i = 0; i < hidden; i++){
            float norm = (rx[i] - mean) * inv_std;
            ry[i] = norm * gamma[i] + beta[i];
        }
    }
}

// -------------------- 误差检查工具 --------------------
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
    // ========== 超参，可以自行修改 ==========
    const int rows = 128;
    const int hidden = 1024;
    const float eps = 1e-5f;
    const int block_size = 128;   // 线程块大小，必须是32倍数
    // 共享内存大小: block_size / 32 个float
    int smem_size = (block_size / 32) * sizeof(float);

    int total_elem = rows * hidden;

    // CPU host内存
    std::vector<float> h_x(total_elem);
    std::vector<float> h_gamma(hidden);
    std::vector<float> h_beta(hidden);
    std::vector<float> h_y_cpu(total_elem);
    std::vector<float> h_y_gpu(total_elem);

    // 初始化随机数据
    srand(42);
    for(int i = 0; i < total_elem; i++){
        h_x[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
    for(int i = 0; i < hidden; i++){
        h_gamma[i] = (rand() / (float)RAND_MAX) * 1.0f;
        h_beta[i]  = (rand() / (float)RAND_MAX) * 0.5f;
    }

    // ========== CPU 计算 + 计时 ==========
    auto cpu_start = std::chrono::high_resolution_clock::now();
    layernorm_cpu(h_x.data(), h_gamma.data(), h_beta.data(), h_y_cpu.data(), rows, hidden, eps);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double,std::milli>(cpu_end - cpu_start).count();

    // ========== GPU 内存分配 ==========
    float *d_x, *d_gamma, *d_beta, *d_y;
    cudaMalloc(&d_x,      total_elem * sizeof(float));
    cudaMalloc(&d_gamma,  hidden * sizeof(float));
    cudaMalloc(&d_beta,   hidden * sizeof(float));
    cudaMalloc(&d_y,      total_elem * sizeof(float));

    cudaMemcpy(d_x,     h_x.data(),     total_elem*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma.data(), hidden*sizeof(float),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta,  h_beta.data(),  hidden*sizeof(float),    cudaMemcpyHostToDevice);

    // ========== GPU warm‑up，消除冷启动误差 ==========
    dim3 grid(rows);
    dim3 block(block_size);
    layernorm_kernel<<<grid, block, smem_size>>>(d_x, d_gamma, d_beta, d_y, rows, hidden, eps);
    cudaDeviceSynchronize();

    // ========== GPU 正式计时 ==========
    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);

    cudaEventRecord(gpu_start);
    layernorm_kernel<<<grid, block, smem_size>>>(d_x, d_gamma, d_beta, d_y, rows, hidden, eps);
    cudaEventRecord(gpu_end);
    cudaDeviceSynchronize();

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_end);

    // 回传结果
    cudaMemcpy(h_y_gpu.data(), d_y, total_elem*sizeof(float), cudaMemcpyDeviceToHost);

    // ========== 结果校验 ==========
    float max_err = check_error(h_y_cpu.data(), h_y_gpu.data(), total_elem);

    // ========== 打印结果 ==========
    std::cout << "rows=" << rows << ", hidden=" << hidden << "\n";
    std::cout << "CPU time: " << cpu_ms << " ms\n";
    std::cout << "GPU kernel time: " << gpu_ms << " ms\n";
    std::cout << "Max absolute error: " << max_err << "\n";
    if(max_err < 1e-3f){
        std::cout << "Result PASS\n";
    }else{
        std::cout << "Result WARNING: large error\n";
    }

    // 释放资源
    cudaFree(d_x);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_y);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_end);

    return 0;
}
