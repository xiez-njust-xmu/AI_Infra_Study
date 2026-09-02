import torch
import numpy as np

a = torch.tensor([1.0, 2.0, 3.0])          # 从 Python 列表创建
zeros = torch.zeros(3, 4)                    # 3x4 全零矩阵
ones = torch.ones(2, 3, 4)                   # 2x3x4 全一张量
rand_normal = torch.randn(3, 4)              # 标准正态分布
seq = torch.arange(0, 10, 2)                 # tensor([0, 2, 4, 6, 8])

x = torch.randn(3, 4)
y = torch.zeros_like(x)                      # 和 x 形状一样的全零 Tensor

t = torch.from_numpy(np.array([1.0, 2.0]))   # 从 NumPy 转换（共享内存）


import torch

x = torch.arange(12)             # 一维，12 个元素
print(x)
a = x.view(3, 4)                 # 变成 3x4（要求内存连续）
b = x.reshape(3, 4)              # 和 view 类似，但内存不连续时也能用
c = x.view(-1, 4)                # -1 自动推断为 3

# permute：交换维度，Transformer 中常用
t = torch.randn(2, 8, 12, 64)              # (batch, seq_len, heads, dim)
t_permuted = t.permute(0, 2, 1, 3)         # (batch, heads, seq_len, dim)

# squeeze/unsqueeze：去掉或插入大小为 1 的维度
e = torch.randn(1, 3, 1, 4)
print(e.squeeze().shape)                    # torch.Size([3, 4])
f = torch.randn(3, 4)
print(f.unsqueeze(0).shape)                 # torch.Size([1, 3, 4])


print(torch.cuda.is_available())            # GPU 是否可用

x = torch.randn(3, 4)
x_gpu = x.to('cuda')                        # CPU → GPU（推荐写法）
x_gpu = x.cuda()                             # 等价简写
x_cpu = x_gpu.cpu()                          # GPU → CPU

# 直接在 GPU 上创建（避免多余的搬运）
x_gpu = torch.randn(3, 4, device='cuda')


import torch

x_fp32 = torch.randn(1000, 1000, dtype=torch.float32)   # 4 bytes/element
x_bf16 = torch.randn(1000, 1000, dtype=torch.bfloat16)  # 2 bytes/element
print(f"fp32: {x_fp32.nelement() * x_fp32.element_size() / 1024:.0f} KB")  # 3906 KB
print(f"bf16: {x_bf16.nelement() * x_bf16.element_size() / 1024:.0f} KB")  # 1953 KB

x = torch.randn(3, 4)                     # 默认 fp32
x_half = x.half()                          # 转 fp16
x_bf16 = x.to(torch.bfloat16)             # 转 bf16


import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)  # 追踪所有操作
y = x * 3
z = y.sum()        # z = 3*2 + 3*3 = 15
z.backward()       # 反向传播
print(x.grad)      # tensor([3., 3.])，dz/dx = 3


import torch

x = torch.tensor([1.0], requires_grad=True)
(x * 2).sum().backward()
print(x.grad)         # tensor([2.])
(x * 3).sum().backward()
print(x.grad)         # tensor([5.]) ← 2+3，梯度被累加！
x.grad.zero_()        # 手动清零
(x * 4).sum().backward()
print(x.grad)         # tensor([4.]) ← 清零后正确
# 实际训练中用 optimizer.zero_grad() 一次性清零所有参数的梯度


import torch

w = torch.tensor([0.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
x_train = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_train = torch.tensor([3.0, 5.0, 7.0, 9.0])

for epoch in range(100):
    loss = ((w * x_train + b - y_train) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= 0.01 * w.grad
        b -= 0.01 * b.grad
    w.grad.zero_()
    b.grad.zero_()

print(f"学到的模型: y = {w.item():.2f}x + {b.item():.2f}")
# 输出接近 y = 2.00x + 1.00
