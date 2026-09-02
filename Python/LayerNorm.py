import torch
import torch.nn as nn

class ManualLayerNorm(nn.Module):
    """
    手动实现的 LayerNorm，与 nn.LayerNorm 行为一致。
    目的是让你清楚每一步在做什么。
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数 gamma，初始化为 1
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        # 可学习的偏移参数 beta，初始化为 0
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 的形状: (..., normalized_shape)
        # 沿最后一个维度计算均值
        mean = x.mean(dim=-1, keepdim=True)
        # 沿最后一个维度计算方差（无偏估计不重要，这里用有偏）
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 标准化：减均值，除标准差
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # 仿射变换：用可学习参数恢复表达能力
        return self.gamma * x_norm + self.beta

# 验证：与 PyTorch 官方实现对比
d_model = 768
manual_ln = ManualLayerNorm(d_model)
official_ln = nn.LayerNorm(d_model)

# 让两者使用相同的参数
with torch.no_grad():
    manual_ln.gamma.copy_(official_ln.weight)
    manual_ln.beta.copy_(official_ln.bias)

x = torch.randn(2, 10, d_model)
out_manual = manual_ln(x)
out_official = official_ln(x)
print(f"最大差异: {(out_manual - out_official).abs().max().item():.2e}")
# 输出约为 1e-7 量级，来自浮点精度差异
