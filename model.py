import torch.nn as nn
import torch.nn.functional as F
input_number=100
class Fuel_Cell_Net_16800(nn.Module):
    def __init__(self):
        super(Fuel_Cell_Net_16800,self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 100, 10)
        self.pool1 = nn.MaxPool1d(3, 2)
        self.conv2 = nn.Conv1d(64, 32, 5, 10)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(32 * 42, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 32 * 42)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class Fuel_Cell_Net_40000(nn.Module):
    def __init__(self):
        super(Fuel_Cell_Net_40000,self).__init__()
        self.conv1=nn.Conv1d(1,64,100,10)
        self.pool1 = nn.MaxPool1d(3, 2)
        self.conv2 = nn.Conv1d(64, 32, 5,10)
        self.pool2=nn.MaxPool1d(2,2)
        self.fc1=nn.Linear(32*100,512)
        self.fc2=nn.Linear(512,64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x=self.pool2(x)
        x=x.view(-1,32*100)
        x=F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
class Fuel_Cell_Net_10001(nn.Module):
    def __init__(self):
        super(Fuel_Cell_Net_10001,self).__init__()
        self.conv1=nn.Conv1d(1,32,101,10)
        self.pool1 = nn.MaxPool1d(3, 2)
        self.conv2 = nn.Conv1d(32, 16, 5,10)
        self.pool2=nn.MaxPool1d(2,2)
        self.fc1=nn.Linear(16*25,256)
        self.fc2=nn.Linear(256,64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x=self.pool2(x)
        x=x.view(-1,16*25)
        x=F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
