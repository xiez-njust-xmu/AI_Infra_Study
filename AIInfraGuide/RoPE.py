import numpy as np

def precompute_freqs(dim, max_seq_len, base=10000.0):
    """
    预计算 RoPE 所需的频率参数和三角函数值。

    参数:
        dim: 每个注意力头的维度 (d_k)
        max_seq_len: 最大序列长度
        base: 频率基数，默认 10000

    返回:
        cos_cached: (max_seq_len, dim/2) 的余弦值
        sin_cached: (max_seq_len, dim/2) 的正弦值
    """
    # 频率参数: theta_i = 1 / base^(2i / dim), i = 0, 1, ..., dim/2 - 1
    freq_indices = np.arange(0, dim, 2, dtype=np.float32)  # [0, 2, 4, ..., dim-2]
    freqs = 1.0 / (base ** (freq_indices / dim))            # (dim/2,)

    # 位置序列
    positions = np.arange(max_seq_len, dtype=np.float32)     # [0, 1, 2, ..., max_seq_len-1]

    # 外积得到每个 (position, freq_pair) 的角度
    angles = np.outer(positions, freqs)  # (max_seq_len, dim/2)

    cos_cached = np.cos(angles)
    sin_cached = np.sin(angles)

    return cos_cached, sin_cached

def apply_rope(q, cos, sin):
    """
    对查询或键向量施加 RoPE 旋转。

    参数:
        q: (seq_len, dim) 的向量，dim 必须是偶数
        cos: (seq_len, dim/2) 的预计算余弦值
        sin: (seq_len, dim/2) 的预计算正弦值

    返回:
        q_rotated: 旋转后的向量，形状与输入相同
    """
    dim = q.shape[-1]
    assert dim % 2 == 0, "维度必须是偶数"

    # 将向量拆分为偶数和奇数维度
    q_even = q[:, 0::2]   # (seq_len, dim/2) -- 取 q_0, q_2, q_4, ...
    q_odd  = q[:, 1::2]   # (seq_len, dim/2) -- 取 q_1, q_3, q_5, ...

    # 对每个二维子空间施加旋转
    q_even_rotated = q_even * cos - q_odd * sin
    q_odd_rotated  = q_even * sin + q_odd * cos

    # 重新交错合并
    q_rotated = np.zeros_like(q)
    q_rotated[:, 0::2] = q_even_rotated
    q_rotated[:, 1::2] = q_odd_rotated

    return q_rotated

# 使用示例
dim = 128        # 头维度
seq_len = 64     # 序列长度

# 预计算三角函数值（只需在初始化时做一次）
cos_cached, sin_cached = precompute_freqs(dim, seq_len)

# 模拟 Q 和 K 向量
q = np.random.randn(seq_len, dim).astype(np.float32)
k = np.random.randn(seq_len, dim).astype(np.float32)

# 施加 RoPE
q_rotated = apply_rope(q, cos_cached, sin_cached)
k_rotated = apply_rope(k, cos_cached, sin_cached)

# 验证：旋转后的内积只依赖相对位置
# 取位置 10 的 Q 和位置 15 的 K
dot1 = np.dot(q_rotated[10], k_rotated[15])

# 取位置 20 的 Q 和位置 25 的 K（相对位置差同样是 5）
# 注意：这里 Q 和 K 的原始值不同，所以内积值不同
# 但如果给定相同的 q, k 向量，放在不同绝对位置上，
# 只要相对位置差相同，内积就相同
dot2 = np.dot(q_rotated[20], k_rotated[25])
print(f"dot(q_rotated[10], k_rotated[15]) = {dot1:.4f}")
print(f"dot(q_rotated[10], k_rotated[15]) = {dot2:.4f}")