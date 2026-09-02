import torch
import torch.nn as nn

class ManualRMSNorm(nn.Module):
    """
    手动实现的 RMSNorm。
    相比 LayerNorm，去掉了均值计算和 beta 参数。
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 只有缩放参数 gamma（通常称为 weight），没有偏移 beta
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算均方根 RMS(x) = sqrt(mean(x^2))
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # 归一化
        x_norm = x / rms
        # 缩放
        return self.weight * x_norm

# 使用示例
d_model = 4096
rms_norm = ManualRMSNorm(d_model)

x = torch.randn(2, 128, d_model)
output = rms_norm(x)
print(output.shape)  # torch.Size([2, 128, 4096])

# 验证归一化效果：输出的 RMS 应接近 1（受 weight 参数影响）
rms_after = torch.sqrt(output.pow(2).mean(dim=-1))
print(f"归一化后 RMS 的均值: {rms_after.mean().item():.4f}")
# 初始化时 weight 全为 1，所以 RMS 约为 1.0
