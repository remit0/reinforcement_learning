import torch.nn as nn
import torch.nn.functional as F


def conv2d_size_out(size, kernel_size=5, stride=2):
    return (size - (kernel_size - 1) - 1) // stride + 1


class BaseCNN(nn.Module):

    def __init__(self, h, w):
        super(BaseCNN, self).__init__()
        self.h = h
        self.w = w
        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class Net(nn.Module):

    def __init__(self, base_net, output_size, use_softmax):
        super(Net, self).__init__()
        self.base_net = base_net
        self.use_softmax = use_softmax
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(base_net.w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(base_net.h)))
        linear_input_size = convw * convh * 32
        self.fc = nn.Linear(linear_input_size, output_size)

    def forward(self, x):
        x = self.base_net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.use_softmax:
            x = F.softmax(x)
        return x


def create_networks():
    base_net = BaseCNN(80, 80)
    policy_net = Net(base_net, 2, use_softmax=True)
    value_net = Net(base_net, 1, use_softmax=False)
    return policy_net, value_net
