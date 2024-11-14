#模型设置版块（模型选择和超参数设置）
#超参数存储值为窗口大小,输出长度, 均值,方差,模型选择,标称容量,终止容量
#这里引入ic_model.py中的一些可以共用的函数
from component.ic.ic_model import (setup_seed, txt_prepare, excel_prepare, txt_prepare_test,
                                   excel_prepare_test, ResNet, CNN, valiloss)
from sklearn.model_selection import train_test_split
from math import sqrt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import pandas as pd
import torch.nn.functional as F


##LSTM,RNN,GRU模型构建部分,Net(window_size,num_classes,model)    model为LSTM,RNN,GRU
##因为LSTM,RNN,GRU都是同类循环网络,构造是一样的
##为了更好适应SOH预测任务，重新设计了网络结构
class soh_Net(nn.Module):  # 三者都相同，所以直接使用mode进行区别就行
    def __init__(self, input_size, n_class=160, mode='LSTM'):
        super(soh_Net, self).__init__()
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
        self.linear = nn.Linear(hidden_dim * 2, n_class * 128)
        self.linear_in = nn.Linear(input_size, hidden_dim)
        self.linear_out = nn.Linear(n_class * 128, n_class)
        self.dropout = nn.Dropout(p=0.5)  #SOH
        self.bn = nn.BatchNorm1d(hidden_dim * 2)

    def forward(self, x):  # x shape: (batch_size, seq_len, input_size)
        out = self.linear_in(x)
        out, _ = self.cell(out)
        out = out.reshape(-1, self.hidden_dim * 2)
        #out = self.linear(out)
        #out=self.bn(out)#
        out = F.relu(out)
        #out = out.reshape(-1,32, self.hidden_dim)
        out = self.linear(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear_out(out)

        return out


#定义训练loss函数
def soh_trainloss(model, train_loader, criterion, optimizer, scheduler, device, epoch):
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
    scheduler.step()
    train_loss /= len(train_loader.dataset)
    return train_loss


#定义模型训练函数
def soh_train(input, output, model_select, best_model_path, window_size,
              epoch_num, batch_size, train_ratio, num_classes, device):
    total_loss, total_validloss = [], []  #用于记录每个epoch的loss和validloss画图
    #数据分割整形
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=1 - train_ratio, shuffle=True)
    #print('train size: {}'.format(len(X_train)))
    #print('valid size: {}'.format(len(X_test)))
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
        model = soh_Net(input_size=window_size, n_class=num_classes, mode=model_select)
    elif model_select == 'CNN':
        model = CNN(window_size=window_size, num_classes=num_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.65)
    criterion = nn.MSELoss()
    for epoch in range(epoch_num):
        train_loss = soh_trainloss(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        valid_loss = valiloss(model, test_loader, criterion, device)
        print('epoch:{:<4d} | loss:{:<7f} | validloss:{:<7f} (RMSE)'
              .format(epoch, sqrt(train_loss), sqrt(valid_loss)))  # RMSE指标
        if (valid_loss < validloss_list[-1]):
            # save best data for repetition
            doc = best_model_path + '/' + str(epoch) + '_' \
                  + str(sqrt(valid_loss)) + '.pth'
            torch.save(model.state_dict(), doc)
            #删除之前的模型
            files = [f for f in os.listdir(best_model_path) if f.endswith('.pth')]
            best_pth_file = max(files, key=lambda x: int(x[:4].split('_')[0]))
            for file in files:
                if file != best_pth_file:
                    os.remove(best_model_path + '/' + file)
            validloss_list.append(valid_loss)
        total_loss.append(train_loss)
        total_validloss.append(valid_loss)
    return total_loss, total_validloss


##模型选择和超参数设置,用户点击模型训练所需的变量设置
#data_formate,v_path,capacity_path为数据读取部分的变量,其他为模型训练部分的变量
def soh_model_train(model_select, best_model_path, best_model_name, parameter_path, window_size,
                    epoch_num, batch_size, train_ratio, data_formate, v_data, soh_data, num_classes,nominal_capacity, eol):
    #构造样本部分进行模型训练
    if data_formate == 'Txt':
        input, output, v_mean, v_std = txt_prepare(soh_data, v_data, window_size)
    elif data_formate == 'Excel':
        input, output, v_mean, v_std = excel_prepare(soh_data, v_data, window_size)
    #打乱样本集，平衡分布
    index = np.arange(input.shape[0])
    np.random.shuffle(index)
    input = (input[index, :, :])
    output = (output[index, :])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    total_loss, total_validloss = soh_train(input=input, output=output, model_select=model_select,
                                            best_model_path=best_model_path, window_size=window_size,
                                            epoch_num=epoch_num, batch_size=batch_size, device=device,
                                            train_ratio=train_ratio, num_classes=num_classes)
    #重命名最优模型为用户定义名称
    # files = [f for f in os.listdir(best_model_path) if f.endswith('.pth')]
    # best_pth_file = max(files, key=lambda x: int(x[:4].split('_')[0]))
    # os.rename(best_pth_file, best_model_name + '.pth')

    #
    # plt.figure(dpi=150)
    # plt.plot(range(1, len(total_loss)+1), total_loss, 'bo', label='trainloss')
    # plt.plot(range(1, len(total_validloss)+1), total_validloss, 'r', label='validloss')
    # plt.title('loss_figure')
    # plt.ylabel('loss')
    # plt.xlabel('epoch_num')
    # plt.legend()
    # plt.show()
    #保存超参数，超参数存储值为窗口大小,输出长度, 均值,方差,模型选择
    parameter = open(parameter_path + '/' + best_model_name + '_parameter.txt', "a+")
    parameter.write(
        str(window_size) + ',' + str(num_classes) + ',' + str(v_mean) + ',' + str(v_std) + ',' + model_select + ','
        + str(nominal_capacity) + ',' + str(eol))
    parameter.close

    return total_loss, total_validloss


#模型测试函数
def soh_model_test(test_model_path, test_parameter_path, data_formate, v_data, soh_data):
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
        model = soh_Net(input_size=window_size, n_class=num_classes, mode=model_select)
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
    ground = np.array(soh_data)
    predict = np.array(predict)
    MSE = np.mean(np.square(predict - ground))
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(predict - ground))
    R2 = 1 - np.sum(np.square(predict - ground)) \
         / np.sum(np.square(ground - np.mean(ground)))
    print('------------------------------------------------------------------')
    print('RMSE: {:.4f},MSE:{:.4f},MAE:{:.4f},R2:{:.4f}'.format(RMSE, MSE, MAE, R2))
    print('------------------------------------------------------------------')

    results = {}
    #
    # results['RMSE'] = 'Average：{:>10.3f}\tWorst：{:>10.3f}\t\tBest：{:>10.3f}'.format(np.mean(RMSE), np.max(RMSE),
    #                                                                                  np.min(RMSE))
    # results['MSE'] = 'Average：{:>10.3f}\tWorst：{:>10.3f}\t\tBest：{:>10.3f}'.format(np.mean(MSE), np.max(MSE),
    #                                                                                 np.min(MSE))
    # results['MAE'] = 'Average：{:>10.3f}\tWorst：{:>10.3f}\t\tBest：{:>10.3f}'.format(np.mean(MAE), np.max(MAE),
    #                                                                                 np.min(MAE))
    # results['R2'] = 'Average：{:>10.3f}\tWorst：{:>10.3f}\t\tBest：{:>10.3f}'.format(np.mean(R2), np.min(R2), np.max(R2))

    # for key, value in results.items():
    #     print(value)
    MAE = round(MAE,3)
    RMSE = round(RMSE, 3)
    R2 = round(R2, 3)

    return results, MAE, RMSE, R2, num_classes, ground, predict
    # 直接绘图预测值和实际值
    # cycle_index = range(len(ground))
    # plt.figure(dpi=150)
    # ground_color=np.array([171,58,41])/255
    # predict_color=np.array([19,103,158])/255
    # plt.plot(cycle_index, ground, color=ground_color, label='ground',linewidth=3)
    # plt.scatter(cycle_index, predict, color=predict_color, label='pred',s=15)
    # point_x = cycle_index[int(len(cycle_index)/2)]
    # point_y = predict[int(len(predict)/2)]
    # plt.annotate(f'R²: {R2:.2f}',
    #              xy=(point_x, point_y),
    #              xytext=(point_x-20, point_y-20),
    #              textcoords='data',
    #              fontsize=12,  # 可以调整字体大小
    #              arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=.2'))
    # plt.title('ground vs predict')
    # plt.ylabel('SOH (%)')
    # plt.xlabel('cycle index')
    # plt.legend()
    # plt.show()
    # # 映射图
    # plt.figure(dpi=150)
    # color1=np.array([98,190,166])/255
    # plt.scatter(ground, predict, color=color1,s=70)#映射点
    # # 计算拟合直线（线性回归）
    # coefficients = np.polyfit(ground.ravel(), predict.ravel(), 1)
    # polynomial = np.poly1d(coefficients)
    # y_fit = polynomial(ground)
    # # 绘制拟合直线
    # plt.plot(ground, y_fit, color=color1,linewidth=3)  # 拟合直线通常用红色显示
    # lims = [
    #     min(min(ground), min(predict)),
    #     max(max(ground), max(predict)),
    # ]
    # color2=np.array([231,98,84])/255
    # plt.plot(lims, lims,  alpha=0.75, zorder=0,color=color2,linewidth=3)#45度线
    # plt.xlim(lims)
    # plt.ylim(lims)
    # plt.title('True vs Estimate SOH')
    # plt.ylabel('Estimate SOH (%)')
    # plt.xlabel('True SOH (%)')
    # plt.show()


#模型预测函数
def soh_model_predict(test_model_path, test_parameter_path, data_formate, v_data, output_path):
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
        model = soh_Net(input_size=window_size, n_class=num_classes, mode=model_select)
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
    predict = np.array(predict)
    #存储预测得到的数据
    if data_formate == 'TXT':
        np.savetxt(output_path + '/predict.txt', predict, delimiter=",")
    elif data_formate == 'Excel':
        pd.DataFrame(predict).to_excel(output_path + '/predict.xlsx', index=False, header=False)


# data_formate='excel'#用户选择的数据格式
# v_path='D:/Desktop/deal_data/6C-60per_3C/CH29/v.xlsx'#这个是Q_V数据路径
# capacity_path='D:/Desktop/deal_data/6C-60per_3C/CH29/capacity.xlsx'#这个是IC数据路径
# nominal_capacity=1.1#这个是电池的标称容量
# eol=nominal_capacity*80/100#80才是用户输入的数字，输入后我们需要将其转变为百分比小数
# v_data,soh_data,num_classes=soh_getin(data_formate,v_path,capacity_path,nominal_capacity,eol)
# model_select='ResNet'   #ResNet,LSTM,RNN,GRU,or CNN
# best_model_path='./'  #最优模型保存路径
# best_model_name='best_model' #最优模型名称
# parameter_path='./'   #超参数保存路径
# window_size=30        #窗口大小
# epoch_num= 220          #训练轮数
# batch_size=128         #批次大小
# train_ratio=0.7       #训练集比例
# #点击模型训练按钮后进行操作
# soh_model_train(model_select=model_select,best_model_path=best_model_path, best_model_name=best_model_name,
#             parameter_path=parameter_path,window_size=window_size,epoch_num=epoch_num,batch_size=batch_size,
#             train_ratio=train_ratio,data_formate=data_formate,v_data=v_data,soh_data=soh_data,num_classes=num_classes)

#数据加载
# data_formate='excel'
# v_path='D:/Desktop/deal_data/6C-60per_3C/CH30/v.xlsx'#这个是Q_V数据路径
# capacity_path='D:/Desktop/deal_data/6C-60per_3C/CH30/capacity.xlsx'#这个是IC数据路径
# nominal_capacity=1.1#这个是电池的标称容量
# eol=nominal_capacity*80/100#80才是用户输入的数字，输入后我们需要将其转变为百分比小数
# v_data,soh_data,num_classes=soh_getin(data_formate,v_path,capacity_path,nominal_capacity,eol)  # 这是数据加载部分的结果
# #然后才是模型测试需要导入的两个变量
# test_model_path='./best_model.pth'
# test_parameter_path='./best_model_parameter.txt'
# #再进行模型验证按钮
# soh_model_test(test_model_path=test_model_path, test_parameter_path=test_parameter_path, data_formate=data_formate,
#             v_data=v_data, soh_data=soh_data)


'''
    模型预测部分,预测未知数据
'''


#数据导入
# data_formate='txt'
# v_path='D:/Desktop/deal_data/6C-60per_3C/CH30/v.txt'#这个是Q_V数据路径
# v_data = soh_getin_predict(data_formate, v_path)
# #然后是模型预测需要导入的模型和参数路径还有保存预测数据的地址
# test_model_path='./best_model.pth'
# test_parameter_path='./best_model_parameter.txt'
# output_path='./'#这个是预测结果保存路径
# #点击模型预测按钮执行以下
# soh_model_predict(test_model_path=test_model_path, test_parameter_path=test_parameter_path, data_formate=data_formate,
#               v_data=v_data,output_path=output_path)


def train_model_wrapper(model_select, best_model_path, best_model_name, parameter_path,
                        window_size, epoch_num, batch_size, train_ratio, data_formate,
                        v_data, ic_data, num_classes, nominal_capacity, eol):
    # 在这里可以进行一些预处理或其他操作
    print("over")
    total_loss, total_vaildloss = None, None
    # 调用 model_train 函数
    total_loss, total_vaildloss = soh_model_train(model_select=model_select, best_model_path=best_model_path,
                                                  best_model_name=best_model_name, parameter_path=parameter_path,
                                                  window_size=window_size, epoch_num=epoch_num, batch_size=batch_size,
                                                  train_ratio=train_ratio, data_formate=data_formate, v_data=v_data,
                                                  soh_data=ic_data, num_classes=num_classes,
                                                  nominal_capacity=nominal_capacity,eol=eol)

    # print("我已完成")

    return total_loss, total_vaildloss


def test_model_wrapper(test_model_path, test_parameter_path, data_formate, v_data, ic_data):
    lables, MAE, RMSE, R2, num_classes, ground, predict = soh_model_test(test_model_path=test_model_path,
                                                                     test_parameter_path=test_parameter_path,
                                                                     data_formate=data_formate, v_data=v_data,
                                                                     soh_data=ic_data)

    # soh_model_test(test_model_path=test_model_path, test_parameter_path=test_parameter_path, data_formate=data_formate,
    #             v_data=v_data, soh_data=soh_data)

    return lables, MAE, RMSE, R2, num_classes, ground, predict


def predict_model_wrapper(test_model_path, test_parameter_path, data_formate, v_data, output_path):
    soh_model_predict(test_model_path=test_model_path, test_parameter_path=test_parameter_path,
                      data_formate=data_formate, v_data=v_data, output_path=output_path)
