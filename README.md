# AI_Infra 学习笔记

记录学习 AI Infra（AI 基础设施 / GPU 高性能计算）的过程。

## 目录结构

| 目录 | 内容 | 状态 |
|------|------|------|
| [Vector](Vector/) | CUDA 向量基础：vectorAdd、vectorDot、vectorReduce、blockReduce | ✅ |
| [SoftMAX](SoftMAX/) | Softmax 优化系列：naive → thread-per-row → block-per-row → warp shuffle → float4 → onepass | ✅ |
| [GEMM](GEMM/) | 矩阵乘优化系列：naive → thread tiling → block tiling → float4/transpose → cuBLAS 对比 | ✅ |

## 学习路径

1. **Vector** —— 入门：线程、block、grid 的基础用法，以及规约（reduce）
2. **SoftMAX** —— 进阶：内存访问优化、warp shuffle、float4 向量化、online softmax 单遍算法
3. **GEMM** —— 深入：分块 tiling、访存优化、cuBLAS 性能对标

## 编译运行

每个子目录都是独立的 CMake 项目，会扫描目录下所有 `.cu` 文件并生成对应可执行文件：

```bash
cd Vector && cmake -B build && cmake --build build
```

> 备注：GEMM / SoftMAX 目录的 CMake 默认架构为 sm_75（预留兼容），Vector 为 sm_89（RTX 4090 / Ada Lovelace）。
> 如有需要，可在构建时显式指定：`cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89`

## 硬件环境

- GPU: NVIDIA RTX 4090（Ada Lovelace, sm_89）
- 编译: CMake + CUDA Toolkit

## TODO

- [ ] FlashAttention
- [ ] 更多算子优化（LayerNorm、Matmul + 融合）
