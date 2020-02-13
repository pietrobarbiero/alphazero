import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib
import numpy as np
from game import Game
from abc import ABC

matplotlib.use("Agg")


class AlphaNet(nn.Module):

    def __init__(self, game: Game):
        super().__init__()

        self.game_state_shape = game.game_state_shape
        self.n_conv_blocks = game.n_conv_blocks
        self.n_res_blocks = game.n_res_blocks
        self.game_dim = game.game_dim
        self.action_size = game.action_size

        for block in range(self.n_res_blocks):
            self.conv = ConvBlock(self.action_size, self.game_state_shape)
        for block in range(self.n_res_blocks):
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock(self.game_dim, self.action_size)

    def forward(self, s):
        s = self.conv(s)
        for block in range(self.n_res_blocks):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s


class ConvBlock(nn.Module):
    def __init__(self, action_size, reshape_size):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.action_size = action_size
        self.reshape_size = reshape_size

    def forward(self, s):
        s = s.view(*self.reshape_size)
        s = F.relu(self.bn1(self.conv1(s)))
        return s


class ResBlock(nn.Module):

    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class OutBlock(nn.Module):

    def __init__(self, game_dim, action_size):
        super(OutBlock, self).__init__()
        self.game_dim = game_dim
        self.action_size = action_size

        self.conv = nn.Conv2d(128, 3, kernel_size=1)  # value head
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3 * self.game_dim, 32)
        self.fc2 = nn.Linear(32, 1)

        self.conv1 = nn.Conv2d(128, 32, kernel_size=1)  # policy head
        self.bn1 = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(self.game_dim * 32, self.action_size)

    def forward(self, s):
        v = F.relu(self.bn(self.conv(s)))  # value head
        v = v.view(-1, 3 * self.game_dim)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))

        p = F.relu(self.bn1(self.conv1(s)))  # policy head
        p = p.view(-1, self.game_dim * 32)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v


class AlphaLoss(torch.nn.Module):

    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy *
                                  (1e-8 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error


class AlphaDataset(Dataset):

    def __init__(self, dataset):  # dataset = np.array of (s, p, v)
        self.X = dataset[:, 0]
        self.y_p, self.y_v = dataset[:, 1], dataset[:, 2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return np.int64(self.X[idx].transpose(2, 0, 1)), self.y_p[idx], self.y_v[idx]
