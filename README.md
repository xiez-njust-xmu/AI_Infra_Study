# AI_Infra 学习笔记

记录学习 AI Infra（AI 基础设施 / GPU 高性能计算）的过程。

## 目录结构

| 目录 | 内容 | 状态 |
|------|------|------|
| [Vector](Vector/) | CUDA 向量基础：vectorAdd、vectorDot、vectorReduce、blockReduce | ✅ |
| [SoftMAX](SoftMAX/) | Softmax 优化系列：naive → thread-per-row → block-per-row → warp shuffle → float4 → onepass | ✅ |
| [GEMM](GEMM/) | 矩阵乘优化系列：naive → thread tiling → block tiling → float4/transpose → cuBLAS 对比 | ✅ |
| [FlashAttentionV1](FlashAttentionV1/) | FlashAttention V1 实现：baseline → v1 → v2 → v3（分块 + online softmax + 融合 kernel，含实现思路笔记） | ✅ |
| [FlashAttentionV2](FlashAttentionV2/) | FlashAttention V2 实现：flash_attention2_v1/v2/v3 可执行 + baseline（kernels v4–v6，去掉因果 mask 分支、寄存器分块 matmul 优化等） | ✅ |
| [FlashAttentionV2_Pytorch](FlashAttentionV2_Pytorch/) | FA2 的 PyTorch 侧工具：SDPA / flash-attn 库 benchmark（flash-attn_benchmark、pytorch_benchmark、pytorch_test）+ verify_output.py 正确性校验 | ✅ |
| [LayerNorm&RMSNorm](LayerNorm&RMSNorm/) | LayerNorm / RMSNorm 手写 kernel，以及第三方参考实现：RMSNorm（H100/B200）、torchnorm | ✅ |
| [Python](Python/) | PyTorch 层学习：Transformer（MHA/RoPE/SwiGLU/Decode Block）、模型组织、训练循环、混合精度、分布通信（NCCL）、性能分析 | ✅ |

## 学习路径

1. **Vector** —— 入门：CUDA 线程、block、grid 的基础用法，以及规约（reduce）
2. **SoftMAX** —— 进阶：内存访问优化、warp shuffle、float4 向量化、online softmax 单遍算法
3. **GEMM** —— 深入：分块 tiling、访存优化、cuBLAS 性能对标
4. **FlashAttention V1** —— 可并行矩阵乘分片 + online softmax 融合，理解 FlashAttention 核心思路
5. **FlashAttention V2** —— 对比 V1 的进一步优化（不存储 N×N 中间矩阵、减少寄存器依赖）
6. **FlashAttention V2（PyTorch 对照）** —— 用 PyTorch 的 SDPA / flash-attn 库做 benchmark 与正确性校验（`verify_output.py`）
7. **LayerNorm / RMSNorm** —— 归一化算子手写 kernel，并对照 H100/B200 上的生产级参考实现（INT21 PTX Kernel Factory、torchnorm）
8. **Python（PyTorch 框架层）** —— 用 PyTorch 串起模型/算子全貌：Transformer 解码块、RoPE、SwiGLU、混合精度训练、NCCL 分布式通信、torch.profiler 性能分析

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

`Python/` 与 `FlashAttentionV2_Pytorch/` 下的 `.py` 文件无需编译，直接运行即可（需要装了 CUDA 版 PyTorch 的环境；`flash-attn_benchmark.py` 额外需要 `flash_attn` 库）：

```bash
cd Python && python "Transformer.py"

# FlashAttentionV2_Pytorch 对比基准
cd FlashAttentionV2_Pytorch && python pytorch_benchmark.py
```

## 硬件环境

- GPU: NVIDIA RTX 4090（Ada Lovelace, sm_89）
- 编译: CMake + CUDA Toolkit
- (FlashAttentionV2_Pytorch 的 PyTorch 脚本默认参数 M=N=8192, d=32；`LayerNorm&RMSNorm` 下的 H100/B200 参考实现需对应架构)

## TODO

- [ ] FlashAttention backward
- [x] FlashAttention
- [x] LayerNorm / RMSNorm 手写 kernel
- [ ] 更多算子融合优化（Matmul + 融合、KV cache / PagedAttention 等）
