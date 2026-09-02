import torch
import time
# 从 flash_attn 库导入函数
from flash_attn import flash_attn_func

B = 4
H = 8
N = 8192
d = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # flash_attn_func 期望的输入形状是 (B, N, H, d)
    # 因此我们直接按此形状创建，或者从原形状 permute 过来
    Q = torch.randn((B, N, H, d), dtype=torch.bfloat16, device=device)
    K = torch.randn((B, N, H, d), dtype=torch.bfloat16, device=device)
    V = torch.randn((B, N, H, d), dtype=torch.bfloat16, device=device)

    # 预热（warmup）
    for _ in range(20):
        _ = flash_attn_func(Q, K, V, causal=False)
    torch.cuda.synchronize()

    niters = 20
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(niters):
        # 调用 flash_attn_func
        O = flash_attn_func(Q, K, V, causal=False)
    torch.cuda.synchronize()
    end = time.time()

    avg_ms = (end - start) / niters * 1000
    print(f"Average time per iter (flash_attn lib): {avg_ms:.3f} ms")
    print(f"Output shape: {O.shape}")  # 输出形状同样是 (B, N, H, d)

if __name__ == "__main__":
    main()