import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tushare as ts
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset
from tqdm import tqdm
from evaluate import  evaluation,diff_evaluation

timestep = 5 # 时间步长
batch_size = 32  # 批次大小
input_dim = 3  # 每个步长对应的特征数量
hidden_dim = 256  # 隐层大小
output_dim = 1  # 由于是回归任务，最终输出层大小为1
num_layers = 16 # LSTM的层数
epochs = 50
best_loss = 0
model_name = 'LSTMwithAttention'
save_path = './{}.pth'.format(model_name)
device="cuda:0" if torch.cuda.is_available() else "CPU"
# # 1.加载时间序列数据
df = pd.read_csv(r"C:\Users\Misery\Desktop\predict\seasonal\8+background.csv", index_col=0)
# 2.数据处理
data=np.array(df)
print(data.shape)

# 形成训练数据
def split_data(data, timestep, input_dim):
    dataX = []  # 保存X
    dataY = []  # 保存Y

    # 将整个窗口的数据保存到X中，将未来一个时间段的数据保存到Y中
    for index in range(len(data) - timestep):
        dataX.append(data[index: index + timestep][:, :])
        dataY.append(data[index + timestep][0])

    dataX = np.array(dataX)
    dataY = np.array(dataY)
    print(dataX.shape)
    print(dataY.shape)
    # 获取训练集大小
    train_size = int(np.round(0.8 * dataX.shape[0]))
    addnum=int(np.round(0.2 * dataX.shape[0]))
    # 划分训练集、测试集
    x_train = dataX[: train_size, :].reshape(-1, timestep, input_dim)
    y_train = dataY[: train_size].reshape(-1, 1)

    x_test = dataX[train_size:train_size+addnum, :].reshape(-1, timestep, input_dim)
    y_test = dataY[train_size:train_size+addnum].reshape(-1, 1)
    return [x_train, y_train, x_test, y_test]

# 3.获取训练数据   x_train
x_train, y_train, x_test, y_test = split_data(data, timestep, input_dim)

# 4.将数据转为tensor
x_train_tensor = torch.from_numpy(x_train).to(torch.float32).to(device)
y_train_tensor = torch.from_numpy(y_train).to(torch.float32).to(device)
x_test_tensor = torch.from_numpy(x_test).to(torch.float32).to(device)
y_test_tensor = torch.from_numpy(y_test).to(torch.float32).to(device)
print(y_test_tensor)
# 5.形成训练数据集
train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)

# 6.将数据加载成迭代器
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size,
                                           True)

test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size,
                                          False)

# 7.定义LSTM网络
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim  # 隐层大小
        self.num_layers = num_layers  # LSTM层数
        # embed_dim为每个时间步对应的特征数
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=1,dropout=0.5)
        # input_dim为特征维度，就是每个时间点对应的特征数量
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, query, key, value):
#         print(query.shape) # torch.Size([16, 1, 4]) batch_size, time_step, input_dim
        attention_output, attn_output_weights = self.attention(query, key, value)
#         print(attention_output.shape) # torch.Size([16, 1, 4]) batch_size, time_step, input_dim
        output, (h_n, c_n) = self.lstm(attention_output)
#         print(output.shape) # torch.Size([16, 1, 64]) batch_size, time_step, hidden_dim
        batch_size, timestep, hidden_dim = output.shape
        output = output.reshape(-1, hidden_dim)
        output = self.fc(output)
        output = output.reshape(timestep, batch_size, -1)
        return output[-1]


model = LSTM(input_dim, hidden_dim, num_layers, output_dim).to(device)  # 定义LSTM网络
loss_function = nn.MSELoss()  # 定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 定义优化器

# 8.模型训练
for epoch in range(epochs):
    model.train()
    running_loss = 0
    train_bar = tqdm(train_loader)  # 形成进度条
    for data in train_bar:
        x_train, y_train = data  # 解包迭代器中的X和Y
        optimizer.zero_grad()
        y_train_pred = model(x_train, x_train, x_train)
        loss = loss_function(y_train_pred, y_train.reshape(-1, 1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)

    # 模型验证
    model.eval()
    test_loss = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            x_test, y_test = data
            y_test_pred = model(x_test, x_test, x_test)
            test_loss = loss_function(y_test_pred, y_test.reshape(-1, 1))

    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), save_path)

print('Finished Training')
model.to("cpu")
x_train_tensor = x_train_tensor.to("cpu")
y_train_tensor= y_train_tensor.to("cpu")
x_test_tensor= x_test_tensor.to("cpu")
y_test_tensor =y_test_tensor.to("cpu")
print(len(x_test_tensor),len(y_test_tensor))

size=200

orginal_y_test_pred=model(x_test_tensor,x_test_tensor,x_test_tensor).cpu().detach().numpy()[: size]
orginal_y_test_tensor=y_test_tensor.cpu().detach().numpy().reshape(-1, 1)[: size]

mae, rmse, mape, r_2=evaluation(orginal_y_test_pred,orginal_y_test_tensor)
print(f"{save_path}:MAE={mae},RMSE={rmse},MAPE={mape},R_2={r_2}")
mae_, rmse_, mape_, r_2_=diff_evaluation(orginal_y_test_pred,orginal_y_test_tensor)
print(f"Diff_1:{save_path}:MAE={mae_},RMSE={rmse_},MAPE={mape_},R_2={r_2_}")
