import torch
import torch.nn.functional as F
import time

B = 4
H = 8
N = 8192
d = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    torch.backends.cuda.enable_flash_sdp(True)

    Q = torch.randn((B, H, N, d), dtype=torch.float16, device=device)
    K = torch.randn((B, H, N, d), dtype=torch.float16, device=device)
    V = torch.randn((B, H, N, d), dtype=torch.float16, device=device)

    for _ in range(10):
        _ = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()

    niters = 20
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(niters):
        O = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()
    end = time.time()

    avg_ms = (end - start) / niters * 1000
    print(f"Average time per iter: {avg_ms:.3f} ms")

if __name__ == "__main__":
    main()
