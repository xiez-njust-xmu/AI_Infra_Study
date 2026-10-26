import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    """SwiGLU 前馈网络：门控投影 + 内容投影 -> 逐元素相乘 -> 降维"""

    def __init__(self, d_model: int, d_ff: int):
        """
        Args:
            d_model: 模型隐藏维度，如 4096
            d_ff: FFN 中间维度，SwiGLU 通常使用 (8/3) * d_model，如 11008
        """
        super().__init__()

        # 门控投影：(d_model, d_ff)
        # 输出经过 Swish 激活后，作为门控信号控制信息通过量
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)

        # 内容投影：(d_model, d_ff)
        # 生成候选通过的信息内容
        self.w_up = nn.Linear(d_model, d_ff, bias=False)

        # 降维投影：(d_ff, d_model)
        # 将门控筛选后的高维信息压缩回原始维度
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量，形状 (batch_size, seq_len, d_model)
        Returns:
            输出张量，形状 (batch_size, seq_len, d_model)
        """
        # Step 1: 门控路径 — 投影到高维空间后经 Swish 激活
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        gate = F.silu(self.w_gate(x))  # silu 即 Swish(x) = x * sigmoid(x)

        # Step 2: 内容路径 — 投影到高维空间（不经过激活函数）
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        up = self.w_up(x)

        # Step 3: 门控乘法 — 门控信号与内容逐元素相乘
        # 门控值大的维度保留信息，门控值小的维度抑制信息
        # (batch_size, seq_len, d_ff)
        hidden = gate * up

        # Step 4: 降维 — 压缩回原始维度
        # (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        output = self.w_down(hidden)

        return output

# 使用示例：LLaMA-2-7B 配置
d_model = 4096
d_ff = 11008  # (8/3) * 4096，取整到 256 的倍数

swiglu_ffn = SwiGLUFFN(d_model, d_ff)
x = torch.randn(2, 128, d_model)  # batch=2, seq_len=128
out = swiglu_ffn(x)  # 输出形状: (2, 128, 4096)
print(f"输入形状: {x.shape}, 输出形状: {out.shape}")
# 参数量: 4096*11008 * 2 (gate + up) + 11008*4096 (down) = 135,266,304 (约 135M)
print(f"参数量: {sum(p.numel() for p in swiglu_ffn.parameters()):,}")
