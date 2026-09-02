import torch
import torch.nn as nn

assert torch.cuda.is_available(), "需要 GPU"
device = torch.device('cuda')
criterion = nn.CrossEntropyLoss()
data = torch.randn(256, 1024, device=device)
labels = torch.randint(0, 10, (256,), device=device)

def make_model():
    return nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(),
                         nn.Linear(2048, 1024), nn.ReLU(),
                         nn.Linear(1024, 10)).to(device)

# fp32 训练
model_fp32, opt_fp32 = make_model(), None
opt_fp32 = torch.optim.AdamW(model_fp32.parameters(), lr=1e-3)
torch.cuda.reset_peak_memory_stats()
criterion(model_fp32(data), labels).backward()
opt_fp32.step(); opt_fp32.zero_grad()
fp32_peak = torch.cuda.max_memory_allocated() / 1024**2

# bf16 混合精度训练
model_bf16, opt_bf16 = make_model(), None
opt_bf16 = torch.optim.AdamW(model_bf16.parameters(), lr=1e-3)
torch.cuda.reset_peak_memory_stats()
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    loss = criterion(model_bf16(data), labels)
loss.backward()
opt_bf16.step(); opt_bf16.zero_grad()
bf16_peak = torch.cuda.max_memory_allocated() / 1024**2

print(f"FP32: {fp32_peak:.1f} MB | BF16: {bf16_peak:.1f} MB | 节省: {(1-bf16_peak/fp32_peak)*100:.1f}%")
