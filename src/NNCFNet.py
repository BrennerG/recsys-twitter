import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import torch.optim as optim

# Define Neural Network
class Net(nn.Module):

    def __init__(self, users:int, items:int, k:int):
        super(Net, self).__init__()
        self.dense1 = nn.Linear(users, k)
        self.dense2 = nn.Linear(items, k)
        self.fc1 = nn.Linear(2*k, k)
        self.fc2 = nn.Linear(k, math.floor(k/2))
        self.fc3 = nn.Linear(math.floor(k/2), 1)

    def forward(self, users, items):
        users = F.relu(self.dense1(users))
        items = F.relu(self.dense2(items))
        # concat users and items into 1 vector
        x = torch.cat((users, items), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        sigfried = nn.Sigmoid()
        x = sigfried(self.fc3(x))
        return x
