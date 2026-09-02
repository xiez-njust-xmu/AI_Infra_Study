import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """RMSNorm: 只做均方根归一化，省去减均值步骤"""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

def precompute_rope_freqs(d_k: int, max_seq_len: int, theta: float = 10000.0):
    """预计算 RoPE 的旋转频率"""
    # 频率: theta_i = 1 / (theta ^ (2i / d_k)), i = 0, 1, ..., d_k/2 - 1
    freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
    # 位置索引: 0, 1, ..., max_seq_len - 1
    positions = torch.arange(max_seq_len).float()
    # 外积: (max_seq_len, d_k // 2)
    angles = torch.outer(positions, freqs)
    # 返回 cos 和 sin，形状均为 (max_seq_len, d_k // 2)
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    对 Q 或 K 施加旋转位置编码
    x: (batch, seq_len, num_heads, d_k)
    cos, sin: (seq_len, d_k // 2)
    """
    seq_len = x.shape[1]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)  # (1, seq, 1, d_k//2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)

    # 将 x 的最后一维拆成两半
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    # 旋转公式：(x1 * cos - x2 * sin, x1 * sin + x2 * cos)
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated

class MaskedMultiHeadAttention(nn.Module):
    """带因果掩码的多头注意力"""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor,
                rope_sin: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # 线性投影
        Q = self.W_Q(x)  # (batch, seq, d_model)
        K = self.W_K(x)
        V = self.W_V(x)

        # 切分为多头: (batch, seq, num_heads, d_k)
        Q = Q.view(batch, seq_len, self.num_heads, self.d_k)
        K = K.view(batch, seq_len, self.num_heads, self.d_k)
        V = V.view(batch, seq_len, self.num_heads, self.d_k)

        # 施加 RoPE（仅对 Q 和 K）
        Q = apply_rope(Q, rope_cos, rope_sin)
        K = apply_rope(K, rope_cos, rope_sin)

        # 转置为 (batch, num_heads, seq, d_k) 便于批量矩阵乘
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 计算注意力分数: (batch, num_heads, seq, seq)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 因果掩码
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device),
            diagonal=1
        )
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Softmax + 加权求和
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)  # (batch, heads, seq, d_k)

        # 合并多头: (batch, seq, d_model)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        # 输出投影
        return self.W_O(output)

class SwiGLUFFN(nn.Module):
    """SwiGLU 前馈网络：三个线性层 + 门控激活"""

    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.W_gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.W_up = nn.Linear(d_model, ffn_dim, bias=False)
        self.W_down = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate 路径: 通过 SiLU (= Swish with beta=1) 激活
        gate = F.silu(self.W_gate(x))    # (batch, seq, ffn_dim)
        # up 路径: 线性变换，不加激活
        up = self.W_up(x)                # (batch, seq, ffn_dim)
        # 门控相乘 + 降维
        return self.W_down(gate * up)    # (batch, seq, d_model)

class TransformerDecoderBlock(nn.Module):
    """完整的 Pre-Norm Transformer Decoder Block"""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int,
                 norm_eps: float = 1e-6):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        self.attn = MaskedMultiHeadAttention(d_model, num_heads)
        self.norm2 = RMSNorm(d_model, eps=norm_eps)
        self.ffn = SwiGLUFFN(d_model, ffn_dim)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor,
                rope_sin: torch.Tensor) -> torch.Tensor:
        # 第一个子层: RMSNorm -> Masked MHA -> 残差
        h = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        # 第二个子层: RMSNorm -> SwiGLU FFN -> 残差
        out = h + self.ffn(self.norm2(h))
        return out

# ---------- 验证：用 LLaMA-2-7B 配置实例化 ----------
if __name__ == "__main__":
    d_model = 4096
    num_heads = 32
    ffn_dim = 11008
    max_seq_len = 4096
    d_k = d_model // num_heads  # 128

    block = TransformerDecoderBlock(d_model, num_heads, ffn_dim)

    # 预计算 RoPE
    rope_cos, rope_sin = precompute_rope_freqs(d_k, max_seq_len)

    # 模拟输入: batch=2, seq_len=128
    x = torch.randn(2, 128, d_model)
    output = block(x, rope_cos, rope_sin)

    print(f"Input shape:  {x.shape}")       # (2, 128, 4096)
    print(f"Output shape: {output.shape}")   # (2, 128, 4096)

    # 统计参数量
    total_params = sum(p.numel() for p in block.parameters())
    print(f"Single block params: {total_params:,}")  # ~201M
