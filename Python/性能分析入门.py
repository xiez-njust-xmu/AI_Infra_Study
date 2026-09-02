import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = torch.nn.Linear(1024, 1024).cuda()
x = torch.randn(64, 1024).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, profile_memory=True) as prof:
    with record_function("forward_pass"):
        output = model(x)
    with record_function("backward_pass"):
        output.sum().backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))



import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(),
                      nn.Linear(4096, 4096), nn.ReLU(),
                      nn.Linear(4096, 10)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
data = torch.randn(128, 1024, device=device)
labels = torch.randint(0, 10, (128,), device=device)

# warmup（避免首次调用的初始化开销）
for _ in range(3):
    criterion(model(data), labels).backward()
    optimizer.step(); optimizer.zero_grad()

# profiler 分析
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, profile_memory=True) as prof:
    with record_function("data_to_gpu"):
        d = torch.randn(128, 1024).to(device)
        l = torch.randint(0, 10, (128,)).to(device)
    with record_function("forward"):
        loss = criterion(model(d), l)
    with record_function("backward"):
        loss.backward()
    with record_function("optimizer_step"):
        optimizer.step(); optimizer.zero_grad()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
prof.export_chrome_trace("trace.json")   # 可在 chrome://tracing 中可视化
