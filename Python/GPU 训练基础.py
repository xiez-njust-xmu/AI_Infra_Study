import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.Linear(100, 10).to(device)
x = torch.randn(32, 100).to(device)     # 模型和数据必须在同一设备上
y = model(x)


import torch
import torch.nn as nn

device = torch.device('cuda')
model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# # BF16 混合精度（推荐，Ampere+ GPU，不需要 GradScaler）
# for images, labels in train_loader:
#     images, labels = images.to(device), labels.to(device)
#     with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#         loss = nn.functional.cross_entropy(model(images), labels)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()

# # FP16 混合精度（需要 GradScaler 防止梯度下溢）
# scaler = torch.cuda.amp.GradScaler()
# for images, labels in train_loader:
#     images, labels = images.to(device), labels.to(device)
#     with torch.autocast(device_type='cuda', dtype=torch.float16):
#         loss = nn.functional.cross_entropy(model(images), labels)
#     scaler.scale(loss).backward()
#     scaler.step(optimizer)
#     scaler.update()
#     optimizer.zero_grad()


import torch

print(f"已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"峰值:   {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print(torch.cuda.memory_summary(abbreviated=True))

# 分析某段代码的显存
torch.cuda.reset_peak_memory_stats()
# ... 运行目标代码 ...
print(f"峰值: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
