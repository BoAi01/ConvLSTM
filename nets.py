import torch.nn as nn
import torch.nn.functional as F

from convlstm import ConvLSTM


class Net04(nn.Module):
    def __init__(self):
        super(Net04, self).__init__()
        global NUM_FRAMES
        # self.conv1 = ConvLSTM(input_channels=1, hidden_channels=[64, 32, 32], kernel_size=3, step=5,
        #           effective_step=[4]).cuda()

        self.maxpool = nn.MaxPool3d((1, 5, 5), stride=(1, 3, 3))
        self.BD = nn.BatchNorm2d(16)
        self.drop = nn.Dropout(p=0.3)
        self.conv1 = ConvLSTM(input_size=(36, 111), input_dim=3, hidden_dim=[64, 32, 32, 16],
                              kernel_size=(3, 3), num_layers=4, batch_first=True)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(3744, 120)
        self.fc2 = nn.Linear(120, 40)
        self.fc3 = nn.Linear(40, 10)

    def forward(self, x):
        # print (x.shape)
        x = self.maxpool(x)
        # x = np.moveaxis(x,1,2)
        # print (x.shape)

        x = self.conv1(x)
        x = self.BD(x[0][0])

        # x = x[0][0]
        # print (x.shape)
        x = F.relu(F.max_pool2d(x, 2))
        # print (x.shape)
        x = self.BD(self.conv2(x))
        x = F.relu(F.max_pool2d(x, 2))

        x = x.view(1, -1)
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))
        x = self.fc3(x).reshape(-1, 1, 2)
        return x
