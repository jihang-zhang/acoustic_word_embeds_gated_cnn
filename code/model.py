import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, pooling=True, initialization=True):
        super(BasicBlock, self).__init__()
        self.initialization = initialization
        self.pooling = pooling

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

        if pooling:
            self.pool = nn.MaxPool1d(3)

        if initialization:
            self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        if self.pooling:
            out = self.pool(out)
        return out

class GatingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, pooling=True, initialization=True, activation=None):
        super(GatingBlock, self).__init__()

        self.activation = activation

        self.block1 = BasicBlock(in_channels, out_channels, kernel_size, stride, padding, pooling, initialization)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.block2 = BasicBlock(in_channels, out_channels, kernel_size, stride, padding, pooling, initialization)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out1 = self.block1(x)
        out1 = F.sigmoid(self.bn1(out1))
        out2 = self.block2(x)
        out2 = self.bn2(out2)
        if self.activation != None:
            out2 = self.activation(out2)
        return out1 * out2

class LinearGatingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None):
        super(LinearGatingBlock, self).__init__()

        self.activation = activation

        self.fc1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.fc2 = nn.Linear(in_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out1 = self.fc1(x)
        out1 = F.sigmoid(self.bn1(out1))
        out2 = self.fc2(x)
        out2 = self.bn2(out2)
        if self.activation != None:
            out2 = self.activation(out2)
        return out1 * out2

class GatedCNN(nn.Module):
    def __init__(self, out_dims, initialization=True, activation=None):
        super(GatedCNN, self).__init__()

        self.activation = activation

        self.gb1 = GatingBlock(39, 512, 9, initialization=initialization, activation=activation)

        self.bottleneck = nn.Sequential(GatingBlock(512, 128, 3, padding=1, pooling=False, initialization=False, activation=activation),
                                        GatingBlock(128, 128, 9, padding=4, pooling=False, initialization=False, activation=activation),
                                        GatingBlock(128, 512, 3, padding=1, pooling=False, initialization=False, activation=activation))

        self.pool = nn.MaxPool1d(16)
        self.lgb = LinearGatingBlock(2048, 1024, activation=activation)
        self.fc = nn.Linear(1024, out_dims)
        self.softmax = torch.nn.Softmax(dim=1)

        if initialization:
            self.init_weights()

    def init_weights(self):
        self.fc.bias.data.fill_(0)
        nn.init.kaiming_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.gb1(x)
        out = self.bottleneck(out)
        out = self.pool(out)
        out = out.view(-1, 2048)
        out = self.lgb(out)
        out = self.fc(out)
        return out

    def get_embeds(self, out):
        return self.softmax(out)

class Siamese(nn.Module):
    def __init__(self, submod, out_dims, margin=0.4, initialization=True, activation=None):
        super(Siamese, self).__init__()

        self.activation = activation
        self.margin = margin

        self.gc = submod(out_dims, initialization, activation)

    def forward_impl(self, achor, same, diff):
        out_a = self.gc(achor)
        out_s = self.gc(same)
        out_d = self.gc(diff)
        return out_a, out_s, out_d

    def forward(self, x):
        return self.gc(x)

    def loss(self, achor, same, diff):
        out_a, out_s, out_d = self.forward_impl(achor, same, diff)
        return torch.mean(F.relu(self.margin + F.cosine_similarity(out_a, out_d) - F.cosine_similarity(out_a, out_s)))

    def get_embeds(self, x):
        return self.gc(x)
