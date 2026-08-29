# FlashAttention V2-PLUS

面向 **NVIDIA DGX Spark**（GB10 / Blackwell，计算能力 `sm_121`）的 **FlashAttention 前向注意力 CUDA 内核**。

这是一个从零手写、针对 Blackwell 架构调优的 FlashAttention **forward** 实现。核心目标是在单台（DGX Spark）上，用 Tensor Core + TMA + 生产者/消费者 warp specialization，把注意力计算做得比朴素的 `Q·Kᵀ → softmax → ·V` 更快、更省显存（不物化完整的 `N×N` 注意力矩阵）。

## 硬件目标

| 项目 | 值 |
| --- | --- |
| 目标设备 | NVIDIA DGX Spark（GB10，Grace Blackwell） |
| CUDA 架构 | `sm_121`（compute capability 12.1） |
| 计算精度 | BF16（`__nv_bfloat16`） |
| 头维度 `d` | **64**（kernel 针对此值调优） |

> 注意：`CMakeLists.txt` 中 `CMAKE_CUDA_ARCHITECTURES` 被硬编码为 `121`，仅在 Blackwell（`sm_121`）上编译运行。

## 核心技术

- **Tensor Core MMA**：`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`，配合 `ldmatrix` 从共享内存载入操作数。
- **TMA（Tensor Memory Accelerator）**：通过 `cuTensorMapEncodeTiled` + `cp.async.bulk.tensor` 做全局内存 → 共享内存的异步块拷贝，并使用 128B swizzle 布局。
- **Warp Specialization（生产者 / 消费者）**：每块 288 线程 = 1 个生产者 warp（负责 TMA 搬运）+ 8 个消费者 warp（负责计算）。
- **mbarrier（异步屏障）**：用 `mbarrier` 的 phase 计数同步生产者与消费者，避免 `__syncthreads` 的阻塞等待。
- **K/V 双缓冲（2-stage）**：两个 K/V 共享内存 stage 交替使用，让搬运和计算重叠。
- **Online Softmax**：在寄存器中维护 running max（`m_i`）与 running sum（`l_i`），逐 KV 块增量更新，避免保存完整 softmax 矩阵。
- **Swizzled Shared Memory**：通过 swizzle 地址计算减少 bank conflict。

## 项目结构

| 文件 | 说明 |
| --- | --- |
| `flash_attn_mma.cu` | 核心 kernel 实现与 launch 函数（MMA + TMA + mbarrier） |
| `flash_attn.cuh` | kernel 接口声明（统一的 launch 签名，便于切换/对比版本） |
| `utils.cuh` | 错误检查宏、`GpuTimer`、数组比较、随机初始化等工具 |
| `main.cu` | 独立 benchmark harness：生成随机数据 → 运行 kernel → 打印耗时 |
| `pytorch_benchmark.py` | PyTorch `scaled_dot_product_attention` 参考基准（BF16/F16） |
| `pytorch_benchmark_new.py` | 另一个 PyTorch SDPA 参考基准（F32） |
| `CMakeLists.txt` | 构建配置 |

## 构建

依赖：CMake ≥ 3.18、CUDA Toolkit（含 cuBLAS），目标设备为 `sm_121`。

```bash
cmake -S . -B build
cmake --build build
```

产物：`build/flash_attention_v2_plus`

## 运行

```bash
./build/flash_attention_v2_plus [B H N d]
# 缺省：B=4  H=8  N=8192  d=64
```

程序会打印配置与多次迭代（3 次 warmup + 20 次计时）的平均延迟（ms）。

### 参数说明

| 参数 | 含义 | 默认 |
| --- | --- | --- |
| `B` | batch size | 4 |
| `H` | attention 头数 | 8 |
| `N` | 序列长度 | 8192 |
| `d` | 头维度（**需为 64**） | 64 |

- 输入 `Q/K/V` 与输出 `O` 的形状均为 `[B, H, N, d]`（BF16，连续存储）。
- `scale` = `1 / sqrt(d)`，自动计算。
- 还会输出 `l`（logsumexp）与 `m`（row max）两个中间量，可用于 backward。

## 参考基准

在 DGX Spark 上与 PyTorch 内置的 Flash SDPA 对比：

```bash
python pytorch_benchmark.py
```

## 当前限制

- 仅实现 **forward**（前向），backward 尚未提供。
- 头维度 `d` 固定为 **64**；其他 `d` 会给出警告且未按此调优。
- 无 causal / mask 支持。
- 仅 BF16，未支持 FP16 / FP32 计算路径。