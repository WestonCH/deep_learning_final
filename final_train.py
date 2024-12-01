import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, Dataset

# 加载数据
data = pd.read_csv('./dataset/merged_data_6w.csv', parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

# 按时间窗口聚合交通流量
data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
data['time_window'] = data['tpep_pickup_datetime'].dt.floor('5T')  # 每5分钟聚合一次
aggregated_data = data.groupby('time_window').size().reset_index(name='traffic_flow')

# 添加时间特征
aggregated_data['hour'] = aggregated_data['time_window'].dt.hour
aggregated_data['weekday'] = aggregated_data['time_window'].dt.weekday
aggregated_data['is_peak_hour'] = aggregated_data['hour'].apply(lambda x: 1 if 7 <= x <= 9 or 17 <= x <= 19 else 0)

# 平滑处理流量
aggregated_data['traffic_flow'] = aggregated_data['traffic_flow'].rolling(window=3, center=True).mean().fillna(method='bfill').fillna(method='ffill')

# 数据标准化
scaler = MinMaxScaler()
aggregated_data[['traffic_flow']] = scaler.fit_transform(aggregated_data[['traffic_flow']])

# 构造时间序列特征
time_steps = 12  # 使用过去12个时间步预测下一个时间步
X, y = [], []
for i in range(time_steps, len(aggregated_data)):
    features = aggregated_data[['traffic_flow', 'hour', 'weekday', 'is_peak_hour']].iloc[i-time_steps:i].values
    X.append(features)
    y.append(aggregated_data['traffic_flow'].iloc[i])

X = np.array(X)
y = np.array(y)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PyTorch 数据集定义
class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 创建 DataLoader
batch_size = 32
train_dataset = TrafficDataset(X_train, y_train)
test_dataset = TrafficDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

import torch.nn as nn

# 模型定义
class TrafficPredictionWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1, num_heads=4):
        super(TrafficPredictionWithAttention, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=3, batch_first=True, dropout=0.3)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.gru(x)  # GRU 层输出
        x, _ = self.attention(x, x, x)  # 自注意力机制
        x = self.fc(x[:, -1, :])  # 取最后一个时间步
        return x
    
# 模型初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = X_train.shape[2]
model = TrafficPredictionWithAttention(input_size=input_size, hidden_size=128, output_size=1, num_heads=4)
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

# 早停机制
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# 训练循环
early_stopping = EarlyStopping(patience=10)
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 验证集评估
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(test_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

# 预测
model.eval()
y_pred, y_true = [], []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs).squeeze()
        y_pred.extend(outputs.cpu().numpy())
        y_true.extend(targets.cpu().numpy())

# 反归一化流量数据
y_pred = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
y_true = scaler.inverse_transform(np.array(y_true).reshape(-1, 1)).flatten()

# 计算评估指标
rmse = root_mean_squared_error(y_true, y_pred, squared=False)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print(f"RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

# 绘制对比图
plt.figure(figsize=(12, 6))
plt.plot(y_true[:100], label='True Values', color='blue', alpha=0.6)
plt.plot(y_pred[:100], label='Predicted Values', color='orange', alpha=0.6)
plt.xlabel('Sample Index')
plt.ylabel('Traffic Flow')
plt.title('True vs Predicted Traffic Flow')
plt.legend()
plt.grid()
plt.savefig('traffic_flow_prediction_1.png')
plt.close()