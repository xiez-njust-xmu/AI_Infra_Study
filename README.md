# AI_Infra 学习笔记

记录学习 AI Infra（AI 基础设施 / GPU 高性能计算）的过程。

## 目录结构

| 目录 | 内容 | 状态 |
|------|------|------|
| [Vector](Vector/) | CUDA 向量基础：vectorAdd、vectorDot、vectorReduce、blockReduce | ✅ |
| [SoftMAX](SoftMAX/) | Softmax 优化系列：naive → thread-per-row → block-per-row → warp shuffle → float4 → onepass | ✅ |
| [GEMM](GEMM/) | 矩阵乘优化系列：naive → thread tiling → block tiling → float4/transpose → cuBLAS 对比 | ✅ |
| [FlashAttentionV1](FlashAttentionV1/) | FlashAttention V1 实现：baseline → v1 → v2 → v3（分块 + online softmax + 融合 kernel，含实现思路笔记） | ✅ |
| [FlashAttentionV2](FlashAttentionV2/) | FlashAttention V2 实现：v1 → v6（去掉因果 mask 分支、寄存器分块 matmul 优化等） | ✅ |
| [FlashAttentionV2_Pytorch](FlashAttentionV2_Pytorch/) | FA2 逐步调优（v1→v10，基于 Nsight Compute）+ pytorch 对标与正确性验证脚本 | ✅ |

## 学习路径

1. **Vector** —— 入门：CUDA 线程、block、grid 的基础用法，以及规约（reduce）
2. **SoftMAX** —— 进阶：内存访问优化、warp shuffle、float4 向量化、online softmax 单遍算法
3. **GEMM** —— 深入：分块 tiling、访存优化、cuBLAS 性能对标
4. **FlashAttention V1** —— 可并行矩阵乘分片 + online softmax 融合，理解 FlashAttention 核心思路
5. **FlashAttention V2** —— 对比 V1 的进一步优化（不存储 N×N 中间矩阵、减少寄存器依赖）
6. **FlashAttention V2 调优（PyTorch 对照）** —— 基于 Nsight Compute 逐步优化（bank conflict、寄存器、双缓冲等），并用 `verify_output.py` 和 PyTorch 交互验证正确性

## 编译运行

每个子目录都是独立的 CMake 项目。`Vector`/`SoftMAX`/`GEMM` 会扫描目录下所有 `.cu` 文件；`FlashAttention*` 采用「静态库 attention_lib + 各 flash_attention 可执行文件」的结构：

```bash
# 简单算子目录
cd Vector && cmake -B build && cmake --build build

# FlashAttention 系列
cd FlashAttentionV1 && cmake -B build && cmake --build build
```

> 备注：CMake 默认架构为 sm_89（RTX 4090 / Ada Lovelace）。GEMM / SoftMAX 预留了 sm_75 兼容；
> 如有需要可在构建时显式指定：`cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89`

### FlashAttentionV2_Pytorch 正确性验证

```bash
# 1. 编译 CUDA 程序
# 2. 运行并输出结果文件（如 result.out）
./flash_attention2_v1 result.out
# 3. 与 PyTorch 的 SDPA 结果对比校验
python3 verify_output.py result.out
```

## 硬件环境

- GPU: NVIDIA RTX 4090（Ada Lovelace, sm_89）
- 编译: CMake + CUDA Toolkit
- (FlashAttentionV2_Pytorch 原参考来源为 A10G 环境，注意默认参数 M=N=8192, d=32)

## TODO

- [ ] 用 Tensor Core（fp16）做 FlashAttention 的 matmul，支持多 head
- [x] FlashAttention
- [ ] 更多算子优化（LayerNorm、Matmul + 融合）
