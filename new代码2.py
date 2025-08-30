import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import numpy as np

# 加载酶数据集：
dataset = TUDataset(root='./data/ENZYMES/', name='ENZYMES')

# 将数据集划分为训练集和测试集：
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义GCN模型：
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)  # 增加第三层卷积层

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

# 初始化模型：
in_channels = dataset.num_node_features
hidden_channels = 64
out_channels = dataset.num_classes

# 创建模型实例：
model = GCN(in_channels, hidden_channels, out_channels)

# 使用Adam优化器，并应用学习率调度器：
optimizer = Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

# 训练模型：
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 测试模型：
def test():
    model.eval()
    correct = 0
    total = 0
    for data in test_loader:
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # 获取预测结果
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)
    accuracy = correct / total
    return accuracy

# 训练和评估：
epochs = 100
for epoch in range(epochs):
    loss = train()
    accuracy = test()
    scheduler.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy*100:.2f}%")
