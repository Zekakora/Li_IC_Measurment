# 模型设置版块（模型选择和超参数设置）
# 超参数存储值为窗口大小,输出长度, 均值,方差,模型选择
from component.ic.ic_getin import *
import numpy as np
import math
from math import sqrt
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt

import re

# 定义一个函数，用于从文件名中提取数字部分
def extract_number(filename):
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        return -1  # 如果找不到数字，则返回一个标记值



##随机种子设定
def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


##generator构造滑动窗口，txt_prepare或者excel_prepare构造样本
def generator(data, lookback, shuffle=False, step=1):
    batch_size = len(data)
    max_index = len(data) - 1
    min_index = 0
    i = min_index + lookback
    if shuffle:
        rows = np.random.randint(
            min_index + lookback, max_index, size=batch_size)
    else:
        if i + batch_size >= max_index:
            i = min_index + lookback
        rows = np.arange(i, min(i + batch_size, max_index))
        i += len(rows)
    samples = np.zeros((len(rows),
                        lookback // step - step,
                        data.shape[-1]
                        ))
    for j, row in enumerate(rows):
        indices = range(rows[j] - lookback, rows[j], step)
        samples[j] = np.array(data[indices][1:, :])
        samples[j][0, :] -= data[indices][0, 0]
    return samples


def txt_prepare(ic_data, v_data, window_size):
    rows = v_data
    lookback_size = window_size + 1  # window size
    step_size = 1  # sampling step
    # get mean and std
    entire_v = []
    for line in rows[0:]:
        line = line.strip()  # 去除行首尾的空白字符
        columns = line.split(',')
        columns = np.array(columns)  # 假设以逗号为分隔符
        columns = np.delete(columns, -1, axis=0)
        for column in columns:
            entire_v.append(eval(column))
    entire_total = np.array(entire_v)
    mean = entire_total.mean(axis=0)
    entire_total -= mean
    std = entire_total.std(axis=0)
    # get sequence
    data_train_temp = []
    target_train_ic = []
    for i, row in enumerate(rows[0:]):  # 0列为换行符
        v = []
        row = row.strip()  # 去除行首尾的空白字符
        columns = row.split(',')
        columns = np.array(columns)  # 逗号为分隔符
        columns = np.delete(columns, -1, axis=0)
        for column in columns:
            v.append(eval(column))
        v = np.array(v)
        v = v.reshape(len(v), 1)
        ic = ic_data[i]
        # standarisation
        temp_train = v - mean
        temp_train = temp_train / std
        batch_size_train = len(temp_train)
        (train_gen) = generator(temp_train,
                                lookback=lookback_size,
                                shuffle=False,
                                step=step_size)
        data_train_temp.append(train_gen)
        A = np.tile(ic, [len(train_gen), 1])  # type: ignore
        target_train_ic.append(A)
    train_gen_final = np.concatenate(data_train_temp, axis=0)
    train_target_ic_final = np.concatenate(target_train_ic, axis=0)
    return train_gen_final, train_target_ic_final, mean, std


def excel_prepare(ic_data, v_data, window_size):
    rows = v_data
    lookback_size = window_size + 1  # window size
    step_size = 1  # sampling step
    # get mean and std
    entire_v = []
    for line in range(len(rows)):
        columns = [value for value in rows.iloc[line, :] if pd.notna(value)]
        for column in columns:
            entire_v.append(column)
    entire_total = np.array(entire_v)
    mean = entire_total.mean(axis=0)
    entire_total -= mean
    std = entire_total.std(axis=0)
    # get sequence
    data_train_temp = []
    target_train_ic = []
    for i in range(len(rows)):  # 0列为换行符
        row = rows.iloc[i, :]
        v = []
        columns = [value for value in row if pd.notna(value)]
        for column in columns:
            v.append(column)
        v = np.array(v)
        v = v.reshape(len(v), 1)
        ic = ic_data.iloc[i, :]
        # standarisation
        temp_train = v - mean
        temp_train = temp_train / std
        batch_size_train = len(temp_train)
        (train_gen) = generator(temp_train,
                                lookback=lookback_size,
                                shuffle=False,
                                step=step_size)
        data_train_temp.append(train_gen)
        A = np.tile(ic, [len(train_gen), 1])  # type: ignore
        target_train_ic.append(A)
    train_gen_final = np.concatenate(data_train_temp, axis=0)
    train_target_ic_final = np.concatenate(target_train_ic, axis=0)
    return train_gen_final, train_target_ic_final, mean, std


# txt_prepare_test或者excel_prepare_test构造测试集样本
def txt_prepare_test(v_data, window_size, v_mean, v_std):
    v = pd.eval(v_data)
    lookback_size = window_size + 1  # window size
    step_size = 1  # sampling step
    mean = v_mean
    std = v_std
    # get sequence
    data_train_temp = []
    v = np.array(v)
    v = v.reshape(len(v), 1)
    # standarisation
    temp_train = v - mean
    temp_train = temp_train / std
    batch_size_train = len(temp_train)
    (train_gen) = generator(temp_train,
                            lookback=lookback_size,
                            shuffle=False,
                            step=step_size)
    data_train_temp.append(train_gen)
    train_gen_final = np.concatenate(data_train_temp, axis=0)
    return train_gen_final


def excel_prepare_test(v_data, window_size, v_mean, v_std):
    v = v_data
    lookback_size = window_size + 1  # window size
    step_size = 1  # sampling step
    mean = v_mean
    std = v_std
    # get sequence
    data_train_temp = []
    v = np.array(v)
    v = v.reshape(len(v), 1)
    # standarisation
    temp_train = v - mean
    temp_train = temp_train / std
    batch_size_train = len(temp_train)
    (train_gen) = generator(temp_train,
                            lookback=lookback_size,
                            shuffle=False,
                            step=step_size)
    data_train_temp.append(train_gen)
    train_gen_final = np.concatenate(data_train_temp, axis=0)
    return train_gen_final


'''
    模型代码部分
'''


##ResNet模型构建部分，ResNet(window_size,num_classes)num_classes为输出长度
###一维卷积
def conv3x1(in_planes, out_planes, stride=1):
    "3x1 convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

    ###基础残差模块


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 维度不变
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes, track_running_stats = True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # downsample 进行调整
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

    ###ResNet主结构


class ResNet(nn.Module):
    def __init__(self, window_size=32, num_classes=160):
        super(ResNet, self).__init__()
        self.inplanes = 16
        # print(bottleneck)
        # 三个单元
        n = 1
        block = BasicBlock
        self.conv1 = nn.Conv1d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes, track_running_stats = True)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool1d(math.ceil(window_size / 4))  # 全局平均
        self.maxpool = nn.MaxPool1d(math.ceil(window_size / 4))
        # 输入avgpool为64*8
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        # 进行权重初始化kaiming正态分布
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = (m.kernel_size * m.out_channels)[0]
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # x= self.maxpool(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x= self.dropout(x)
        x = self.fc(x)
        return x


##LSTM,RNN,GRU模型构建部分,Net(window_size,num_classes,model)    model为LSTM,RNN,GRU
##因为LSTM,RNN,GRU都是同类循环网络,构造是一样的
class Net(nn.Module):  # 三者都相同，所以直接使用mode进行区别就行
    def __init__(self, input_size, n_class=160, mode='LSTM'):
        super(Net, self).__init__()
        hidden_dim = 32
        self.hidden_dim = hidden_dim
        num_layers = 2
        if mode == 'LSTM':
            self.cell = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim * 2, num_layers=num_layers,
                                batch_first=True)
        if mode == 'GRU':
            self.cell = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim * 2, num_layers=num_layers,
                               batch_first=True)
        elif mode == 'RNN':
            self.cell = nn.RNN(input_size=hidden_dim, hidden_size=hidden_dim * 2, num_layers=num_layers,
                               batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, n_class * 2)
        self.linear_in = nn.Linear(input_size, hidden_dim)
        self.linear_out = nn.Linear(n_class * 2, n_class)
        self.dropout = nn.Dropout(p=0.2)
        self.bn = nn.BatchNorm1d(hidden_dim * 2)

    def forward(self, x):  # x shape: (batch_size, seq_len, input_size)
        out = self.linear_in(x)
        out, _ = self.cell(out)
        out = out.reshape(-1, self.hidden_dim * 2)
        # out = self.linear(out)
        # out=self.bn(out)
        out = F.relu(out)
        # out = out.reshape(-1,32, self.hidden_dim)
        out = self.linear(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear_out(out)

        return out


##CNN模型构建代码,CNN(window_size,num_classes)
class CNN(nn.Module):
    def __init__(self, window_size, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=10, stride=1)
        self.fc1 = nn.Linear(int(32 * window_size / 30), 160)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(160, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 32)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


'''
    模型代码结束
'''


# 训练集loss函数
def trainloss(model, train_loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    train_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    if epoch < 360:
        scheduler.step()
    train_loss /= len(train_loader.dataset)
    return train_loss


# 验证集loss函数
def valiloss(model, val_loader, criterion, device):
    model.eval()
    vali_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            vali_loss += criterion(output, target).item() * data.size(0)
    vali_loss /= len(val_loader.dataset)
    return vali_loss


# 定义模型训练函数
def train(input, output, model_select, best_model_path, window_size,
          epoch_num, batch_size, train_ratio, num_classes, device):
    total_loss, total_validloss = [], []  # 用于记录每个epoch的loss和validloss画图
    # 数据分割整形
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=1 - train_ratio, shuffle=True)
    # print('train size: {}'.format(len(X_train)))
    # print('valid size: {}'.format(len(X_test)))
    X_train, X_test = X_train.reshape(-1, 1, window_size), X_test.reshape(-1, 1, window_size)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    seed = 10000
    lr = 0.01
    MAX = np.array(float('inf'))
    validloss_list = [torch.from_numpy(MAX)]
    setup_seed(seed)
    # 选择模型
    if model_select == 'ResNet':
        model = ResNet(window_size=window_size, num_classes=num_classes)
    elif model_select == 'LSTM' or model_select == 'RNN' or model_select == 'GRU':
        model = Net(input_size=window_size, n_class=num_classes, mode=model_select)
    elif model_select == 'CNN':
        model = CNN(window_size=window_size, num_classes=num_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.65)
    criterion = nn.MSELoss()
    for epoch in range(epoch_num):
        train_loss = trainloss(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        valid_loss = valiloss(model, test_loader, criterion, device)
        print('epoch:{:<4d} | loss:{:<7f} | validloss:{:<7f} (RMSE)'
              .format(epoch, sqrt(train_loss), sqrt(valid_loss)))  # RMSE指标
        if (valid_loss < validloss_list[-1]):
            # save best data for repetition
            doc = best_model_path + '/' + str(epoch) + '_' \
                  + str(sqrt(valid_loss)) + '.pth'
            torch.save(model.state_dict(), doc)
            # 删除之前的模型
            files = [f for f in os.listdir(best_model_path) if f.endswith('.pth')]
            # best_pth_file = max(files, key=lambda x: int(x[:4].split('_')[0]))
            best_pth_file = max(files, key=lambda x: extract_number(x))
            for file in files:
                if file != best_pth_file:
                    os.remove(best_model_path + '/' + file)
            validloss_list.append(valid_loss)
        if (valid_loss < 90) and (valid_loss > validloss_list[-1]):
            break
        total_loss.append(train_loss)
        total_validloss.append(valid_loss)
    return total_loss, total_validloss


'''
    训练模型部分构建函数结束，以下为模型训练函数，用户点击模型训练按钮后，调用以下函数
'''


##模型选择和超参数设置,用户点击模型训练所需的变量设置
# data_formate,v_path,ic_path为数据读取部分的变量,其他为模型训练部分的变量
def model_train(model_select, best_model_path, best_model_name, parameter_path, window_size,
                epoch_num, batch_size, train_ratio, data_formate, v_data, ic_data, num_classes):
    # 构造样本部分进行模型训练
    if data_formate == 'TXT':
        input, output, v_mean, v_std = txt_prepare(ic_data, v_data, window_size)
    elif data_formate == 'Excel':
        input, output, v_mean, v_std = excel_prepare(ic_data, v_data, window_size)
    # 打乱样本集，平衡分布
    index = np.arange(input.shape[0])
    np.random.shuffle(index)
    input = (input[index, :, :])
    output = (output[index, :])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    total_loss, total_validloss = train(input=input, output=output, model_select=model_select,
                                        best_model_path=best_model_path, window_size=window_size,
                                        epoch_num=epoch_num, batch_size=batch_size, device=device,
                                        train_ratio=train_ratio, num_classes=num_classes)


    # plt.figure(dpi=150)
    # plt.plot(range(1, len(total_loss) + 1), total_loss, 'bo', label='trainloss')
    # plt.plot(range(1, len(total_validloss) + 1), total_validloss, 'r', label='validloss')
    # plt.title('loss_figure')
    # plt.ylabel('loss')
    # plt.xlabel('epoch_num')
    # plt.legend()
    # plt.show()
    # 保存超参数，超参数存储值为窗口大小,输出长度, 均值,方差,模型选择
    parameter = open(parameter_path + '/' + best_model_name + '_parameter.txt', "a+")
    parameter.write(
        str(window_size) + ',' + str(num_classes) + ',' + str(v_mean) + ',' + str(v_std) + ',' + model_select)
    parameter.close

    print("写入完成")
    # # 重命名最优模型为用户定义名称
    # files = [f for f in os.listdir(best_model_path) if f.endswith('.pth')]
    # best_pth_file = max(files, key=lambda x: int(x[:4].split('_')[0]))
    # os.rename(best_pth_file, best_model_name + '.pth')
    #
    # print("写入完成2")

    return total_loss, total_validloss

'''
    构建模型训练函数
'''


# 模型测试函数
def model_test(test_model_path, test_parameter_path, data_formate, v_data, ic_data):
    # 数据加载部分的操作
    parameter = pd.read_table(test_parameter_path, delimiter=',', header=None)

    # 读取所存储参数赋值为变量
    window_size = pd.eval(parameter.iloc[0, 0])
    num_classes = pd.eval(parameter.iloc[0, 1])
    v_mean = pd.eval(parameter.iloc[0, 2])
    v_std = pd.eval(parameter.iloc[0, 3])
    model_select = parameter.iloc[0, 4]
    # 选取模型
    if model_select == 'ResNet':
        model = ResNet(window_size=window_size, num_classes=num_classes)
    elif model_select == 'LSTM' or model_select == 'RNN' or model_select == 'GRU':
        model = Net(input_size=window_size, n_class=num_classes, mode=model_select)
    elif model_select == 'CNN':
        model = CNN(window_size=window_size, num_classes=num_classes)
    # 模型权重加载
    model.load_state_dict(torch.load(test_model_path))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    predict = []
    with torch.no_grad():
        for cycle in range(len(v_data)):
            # 构造滑动窗口
            if data_formate == 'TXT':
                input = txt_prepare_test(v_data[cycle], window_size, v_mean, v_std)
            elif data_formate == 'Excel':
                v_temp = [value for value in v_data.iloc[cycle, :] if pd.notna(value)]
                input = excel_prepare_test(v_temp, window_size, v_mean, v_std)
            X = np.reshape(input, (-1, 1, window_size)).astype(np.float32)
            X = torch.from_numpy(X).to(device)
            output = model(X)
            output = output.reshape(-1, num_classes)
            output = output.mean(axis=0)
            predict.append(np.ceil(output.cpu().detach().numpy()))
    ground = np.array(ic_data) / 3600  # 转换单位为Ah/V 这是IC曲线的常用单位
    predict = np.array(predict) / 3600
    MSE = np.mean(np.square(predict - ground), axis=1)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(predict - ground), axis=1)
    R2 = 1 - np.sum(np.square(predict - ground), axis=1) \
         / np.sum(np.square(ground - np.mean(ground, axis=1).reshape(-1, 1)), axis=1)
    print('------------------------------------------------------------------')
    print('RMSE: average:{:.4f},worst:{:.4f},best:{:.4f}'.format(np.mean(RMSE), np.max(RMSE), np.min(RMSE)))
    print('MSE: average:{:.4f},worst:{:.4f},best:{:.4f}'.format(np.mean(MSE), np.max(MSE), np.min(MSE)))
    print('MAE: average:{:.4f},worst:{:.4f},best:{:.4f}'.format(np.mean(MAE), np.max(MAE), np.min(MAE)))
    print('R2: average:{:.4f},worst:{:.4f},best:{:.4f}'.format(np.mean(R2), np.min(R2), np.max(R2)))
    print('------------------------------------------------------------------')

    # 假设您的结果已经计算完毕并存储在相应的变量中

    # 建立一个空字典来存放结果
    results = {}

    results['RMSE'] = 'Average: {:>10.4f}\tWorst: {:>10.4f}\tBest: {:>10.4f}'.format(np.mean(RMSE), np.max(RMSE),
                                                                                     np.min(RMSE))
    results['MSE'] = 'Average: {:>10.4f}\tWorst: {:>10.4f}\tBest: {:>10.4f}'.format(np.mean(MSE), np.max(MSE),
                                                                                    np.min(MSE))
    results['MAE'] = 'Average: {:>10.4f}\tWorst: {:>10.4f}\tBest: {:>10.4f}'.format(np.mean(MAE), np.max(MAE),
                                                                                    np.min(MAE))
    results['R2'] = 'Average: {:>10.4f}\tWorst: {:>10.4f}\tBest: {:>10.4f}'.format(np.mean(R2), np.min(R2), np.max(R2))

    for key, value in results.items():
        print(value)

    return results, MAE, RMSE, num_classes, ground, predict
    # MAE_worst = np.argmax(MAE)
    # MAE_best = np.argmin(MAE)
    # v_index = np.linspace(2.01, 3.60, num_classes)
    # # MAE_worst
    # plt.figure(dpi=150)
    # plt.plot(v_index, ground[MAE_worst, :], 'bo', label='ground')
    # plt.plot(v_index, predict[MAE_worst, :], 'r', label='pred')
    # plt.title('MAE_worst_predict')
    # plt.ylabel('Incremental Capacity(Ah/V)')
    # plt.xlabel('voltage (V)')
    # plt.legend()
    # plt.show()
    # # plt.clf()
    #
    # # MAE_best
    # plt.figure(dpi=150)
    # plt.plot(v_index, ground[MAE_best, :], 'bo', label='ground')
    # plt.plot(v_index, predict[MAE_best, :], 'r', label='pred')
    # plt.title('MAE_best_predict')
    # plt.ylabel('Incremental Capacity(Ah/V)')
    # plt.xlabel('voltage (V)')
    # plt.legend()
    # plt.show()
    # # plt.clf()
    # # RMSE_figure
    # RMSE_worst = np.argmax(RMSE)
    # RMSE_best = np.argmin(RMSE)
    # v_index = np.linspace(2.01, 3.60, num_classes)
    # # RMSE_worst
    # plt.figure(dpi=150)
    # plt.plot(v_index, ground[RMSE_worst, :], 'bo', label='ground')
    # plt.plot(v_index, predict[RMSE_worst, :], 'r', label='pred')
    # plt.title('RMSE_worst_predict')
    # plt.ylabel('Incremental Capacity(Ah/V)')
    # plt.xlabel('voltage (V)')
    # plt.legend()
    # plt.show()
    # # plt.clf()
    #
    # # RMSE_best
    # plt.figure(dpi=150)
    # plt.plot(v_index, ground[RMSE_best, :], 'bo', label='ground')
    # plt.plot(v_index, predict[RMSE_best, :], 'r', label='pred')
    # plt.title('RMSE_best_predict')
    # plt.ylabel('Incremental Capacity(Ah/V)')
    # plt.xlabel('voltage (V)')
    # plt.legend()
    # plt.show()
    # # plt.clf()


# 模型预测函数
def model_predict(test_model_path, test_parameter_path, data_formate, v_data, output_path):
    parameter = pd.read_table(test_parameter_path, delimiter=',', header=None)
    # 读取所存储参数赋值为变量
    window_size = pd.eval(parameter.iloc[0, 0])
    num_classes = pd.eval(parameter.iloc[0, 1])
    v_mean = pd.eval(parameter.iloc[0, 2])
    v_std = pd.eval(parameter.iloc[0, 3])
    model_select = parameter.iloc[0, 4]
    # 选取模型
    if model_select == 'ResNet':
        model = ResNet(window_size=window_size, num_classes=num_classes)
    elif model_select == 'LSTM' or model_select == 'RNN' or model_select == 'GRU':
        model = Net(input_size=window_size, n_class=num_classes, mode=model_select)
    elif model_select == 'CNN':
        model = CNN(window_size=window_size, num_classes=num_classes)
    # 模型权重加载
    model.load_state_dict(torch.load(test_model_path))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    predict = []
    with torch.no_grad():
        for cycle in range(len(v_data)):
            # 构造滑动窗口
            if data_formate == 'TXT':
                input = txt_prepare_test(v_data[cycle], window_size, v_mean, v_std)
            elif data_formate == 'Excel':
                v_temp = [value for value in v_data.iloc[cycle, :] if pd.notna(value)]
                input = excel_prepare_test(v_temp, window_size, v_mean, v_std)
            X = np.reshape(input, (-1, 1, window_size)).astype(np.float32)
            X = torch.from_numpy(X).to(device)
            output = model(X)
            output = output.reshape(-1, num_classes)
            output = output.mean(axis=0)
            predict.append(np.ceil(output.cpu().detach().numpy()))
    predict = np.array(predict) / 3600
    # 存储预测得到的数据
    if data_formate == 'TXT':
        np.savetxt(output_path + '/predict.txt', predict, delimiter=",")
    elif data_formate == 'Excel':
        output_path = output_path + '/predict.xlsx'
        print(output_path)
        pd.DataFrame(predict).to_excel(output_path, index=False, header=False)
    print("finish")

'''
# #模型训练操作流程
'''


# #先是数据加载板块
# data_formate='excel'#用户选择的数据格式
# v_path='D:/v.xlsx'#这个是Q_V数据路径
# ic_path='D:/ic.xlsx'#这个是IC数据路径
# v_data,ic_data,num_classes,degration =ic_getin(data_formate,v_path,ic_path)#这是数据加载部分的结果
# #degration需要一个小输出框表示（名字是老化机制)
# print(degration)
# #然后才是模型选择和模型参数设置的变量
# model_select='ResNet'   #ResNet,LSTM,RNN,GRU,or CNN
# best_model_path='./'  #最优模型保存路径
# best_model_name='best_model' #最优模型名称
# parameter_path='./'   #超参数保存路径
# window_size=30        #窗口大小
# epoch_num= 2      #训练轮数
# batch_size=512         #批次大小
# train_ratio=0.7       #训练集比例
# #点击模型训练按钮后进行操作
# model_train(model_select=model_select,best_model_path=best_model_path, best_model_name=best_model_name,
#             parameter_path=parameter_path,window_size=window_size,epoch_num=epoch_num,batch_size=batch_size,
#             train_ratio=train_ratio,data_formate=data_formate,v_data=v_data,ic_data=ic_data,num_classes=num_classes)


def train_model_wrapper(model_select, best_model_path, best_model_name, parameter_path,
                        window_size, epoch_num, batch_size, train_ratio, data_formate,
                        v_data, ic_data, num_classes):
    # 在这里可以进行一些预处理或其他操作
    print("over")
    total_loss, total_vaildloss = None, None
    # 调用 model_train 函数
    total_loss, total_vaildloss = model_train(model_select=model_select, best_model_path=best_model_path,
                best_model_name=best_model_name, parameter_path=parameter_path,
                window_size=window_size, epoch_num=epoch_num, batch_size=batch_size,
                train_ratio=train_ratio, data_formate=data_formate, v_data=v_data,
                ic_data=ic_data, num_classes=num_classes)
    # print("我已完成")

    return total_loss, total_vaildloss

'''
    模型测试部分,衡量模型性能
'''


# 数据加载

# data_formate='txt'
# v_path='D:/Desktop/deal_data/6C-60per_3C/CH30/v.txt'#这个是Q_V数据路径
# ic_path='D:/Desktop/deal_data/6C-60per_3C/CH30/ic.txt'#这个是IC数据路径
# v_data, ic_data, num_classes ,degration= ic_getin(data_formate, v_path, ic_path)  # 这是数据加载部分的结果
##degration需要一个小输出框表示（名字是老化机制)
# print(degration)
# #然后才是模型测试需要导入的两个变量
# test_model_path='./best_model.pth'
# test_parameter_path='./best_model_parameter.txt'
# #再进行模型验证按钮
#
# model_test(test_model_path=test_model_path, test_parameter_path=test_parameter_path, data_formate=data_formate,
#             v_data=v_data, ic_data=ic_data )

def test_model_wrapper(test_model_path, test_parameter_path, data_formate, v_data, ic_data):
    lables, MAE, RMSE, num_classes, ground, predict =  model_test(test_model_path=test_model_path, test_parameter_path=test_parameter_path,
               data_formate=data_formate, v_data=v_data, ic_data=ic_data)

    return lables, MAE, RMSE, num_classes, ground, predict

'''
    模型预测部分,预测未知数据
'''


# #数据导入
# data_formate='excel'
# v_path='D:/Desktop/电气电子/method_code/6C-60per_3C/CH30/v.xlsx'#这个是Q_V数据路径
# v_data = ic_getin_predict(data_formate, v_path)
# #然后是模型预测需要导入的模型和参数路径还有保存预测数据的地址
# test_model_path='./best_model.pth'
# test_parameter_path='./best_model_parameter.txt'
# output_path='./'#这个是预测结果保存路径
# #点击模型预测按钮执行以下
# model_predict(test_model_path=test_model_path, test_parameter_path=test_parameter_path, data_formate=data_formate,
#               v_data=v_data,output_path=output_path)

def predict_model_wrapper(test_model_path, test_parameter_path, data_formate, v_data, output_path):
    model_predict(test_model_path=test_model_path, test_parameter_path=test_parameter_path,
                  data_formate=data_formate, v_data=v_data, output_path=output_path)
