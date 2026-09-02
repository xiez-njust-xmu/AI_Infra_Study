import numpy as np
import matplotlib.pyplot as plt

def sinusoidal_position_encoding(max_len, d_model):
    """
    生成 Sinusoidal 位置编码矩阵。

    参数:
        max_len: 最大序列长度
        d_model: 模型隐藏维度

    返回:
        PE: 形状为 (max_len, d_model) 的位置编码矩阵
    """
    PE = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]          # (max_len, 1)
    dim_pair_index = np.arange(0, d_model, 2)[np.newaxis, :]  # (1, d_model/2)

    # 计算频率：omega_i = 1 / 10000^(2i/d_model)
    # 等价于 exp(-2i/d_model * ln(10000))
    div_term = np.exp(dim_pair_index * -(np.log(10000.0) / d_model))

    # 偶数维度用 sin，奇数维度用 cos
    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)

    return PE

# 生成位置编码
max_len = 128
d_model = 64
PE = sinusoidal_position_encoding(max_len, d_model)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 热力图：展示完整编码矩阵
im = axes[0].imshow(PE, aspect='auto', cmap='RdBu_r', interpolation='nearest')
axes[0].set_xlabel('Dimension Index')
axes[0].set_ylabel('Position')
axes[0].set_title('Sinusoidal Position Encoding Matrix')
plt.colorbar(im, ax=axes[0])

# 曲线图：展示几个选定维度的编码值随位置的变化
dims_to_plot = [0, 1, 10, 11, 30, 31]
for dim in dims_to_plot:
    label = f'dim {dim} ({"sin" if dim % 2 == 0 else "cos"}, freq_idx={dim // 2})'
    axes[1].plot(PE[:, dim], label=label, alpha=0.8)
axes[1].set_xlabel('Position')
axes[1].set_ylabel('Encoding Value')
axes[1].set_title('Encoding Values at Selected Dimensions')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sinusoidal_pe.png', dpi=150)
plt.show()
