##LSTM,RNN,GRU模型构建部分,Net(window_size,num_classes,model)    model为LSTM,RNN,GRU
##因为LSTM,RNN,GRU都是同类循环网络,构造是一样的
class Net(nn.Module):  # 三者都相同，所以直接使用mode进行区别就行
    def __init__(self, input_size,  n_class=160, mode='LSTM'):
        super(Net, self).__init__()
        hidden_dim = 32
        self.hidden_dim = hidden_dim
        num_layers=2
        if mode == 'LSTM':
            self.cell = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim*2, num_layers=num_layers, batch_first=True)
        if mode == 'GRU':
            self.cell = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim*2, num_layers=num_layers, batch_first=True)
        elif mode == 'RNN':
            self.cell = nn.RNN(input_size=hidden_dim, hidden_size=hidden_dim*2, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim*2, n_class*2)
        self.linear_in=nn.Linear(input_size,hidden_dim)
        self.linear_out = nn.Linear(n_class*2, n_class)
        self.dropout = nn.Dropout(p=0.2)
        self.bn= nn.BatchNorm1d(hidden_dim*2)

    def forward(self, x):  # x shape: (batch_size, seq_len, input_size)
        out = self.linear_in(x)
        out, _ = self.cell(out)
        out = out.reshape(-1, self.hidden_dim*2)
        #out = self.linear(out)
        #out=self.bn(out)
        out = F.relu(out)
        #out = out.reshape(-1,32, self.hidden_dim)
        out = self.linear(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear_out(out)

        return out

##CNN模型构建代码,CNN(window_size,num_classes)
class CNN(nn.Module):
    def __init__(self,window_size,num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=10, stride=1)
        self.fc1 = nn.Linear(int(32*window_size/30) , 160)
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
        x = x.view(-1, 32 )
        x = self.fc1(x)
        x = self.fc2(x)
        return x