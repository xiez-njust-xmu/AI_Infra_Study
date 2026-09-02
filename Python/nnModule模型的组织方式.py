import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

model = SimpleModel(784, 256, 10)
output = model(torch.randn(32, 784))    # 用 model(x)，不要直接调用 forward
print(output.shape)                       # torch.Size([32, 10])


import torch.nn as nn

model = nn.Linear(10, 5)
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")   # weight: [5, 10], bias: [5]

# 统计参数量（AI Infra 中非常常用）
total_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {total_params}")       # 55

# state_dict 用于保存和加载模型
print(model.state_dict().keys())       # odict_keys(['weight', 'bias'])


import torch
import torch.nn as nn

# nn.Linear：大模型中用最多的层（Attention Q/K/V 投影、FFN）
linear = nn.Linear(512, 256)

# nn.Embedding：将 token id 映射为稠密向量（大模型第一层）
embedding = nn.Embedding(num_embeddings=50000, embedding_dim=768)
print(embedding(torch.tensor([0, 42])).shape)   # torch.Size([2, 768])

# nn.LayerNorm：Transformer 每个子层后都有，稳定训练
layer_norm = nn.LayerNorm(768)

# nn.Dropout：训练时随机丢弃神经元，model.eval() 后自动关闭
dropout = nn.Dropout(p=0.1)


import torch
import torch.nn as nn

class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

model = TwoLayerMLP(784, 256, 10)
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")  # 203,530
print(model(torch.randn(64, 784)).shape)  # torch.Size([64, 10])
