import torch
import torch.nn as nn

class ResidualSubLayer(nn.Module):
    """
    通用的残差子层包装器。
    将任意子层（Attention 或 FFN）包装为：output = x + SubLayer(LayerNorm(x))
    """

    def __init__(self, d_model: int, sublayer: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm: 先归一化，再送入子层
        normed = self.norm(x)
        # 子层计算
        sub_output = self.sublayer(normed)
        # Dropout 正则化
        sub_output = self.dropout(sub_output)
        # 残差连接：直接加上原始输入
        return x + sub_output

class SimpleFeedForward(nn.Module):
    """简单的两层 FFN，用于演示"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.activation(self.w1(x)))

# 使用示例
d_model = 512
d_ff = 2048
ffn = SimpleFeedForward(d_model, d_ff)
residual_ffn = ResidualSubLayer(d_model, ffn)

# 模拟输入：batch_size=2, seq_len=10, d_model=512
x = torch.randn(2, 10, d_model)
output = residual_ffn(x)
print(output.shape)  # torch.Size([2, 10, 512])

# 验证：输出与输入的差值就是子层学到的"残差"
residual = output - x
print(f"残差的均值: {residual.mean().item():.6f}")
print(f"残差的标准差: {residual.std().item():.6f}")
