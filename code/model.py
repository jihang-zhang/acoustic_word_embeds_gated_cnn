import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialization=True):
        super(BasicBlock, self).__init__()
        self.initialization = initialization
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

        if initialization:
            self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)

class GatingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialization=True, activation=None):
        super(GatingBlock, self).__init__()

        self.activation = activation

        self.block1 = BasicBlock(in_channels, out_channels, kernel_size, stride, padding, initialization)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.block2 = BasicBlock(in_channels, out_channels, kernel_size, stride, padding, initialization)
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
        self.pool1 = nn.MaxPool1d(3)

        self.bottleneck = nn.Sequential(GatingBlock(512, 128, 3, padding=1, initialization=False, activation=activation),
                                        GatingBlock(128, 128, 9, padding=4, initialization=False, activation=activation),
                                        GatingBlock(128, 512, 3, padding=1, initialization=False, activation=activation))

        self.pool2 = nn.MaxPool1d(16)
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
        out = self.pool1(out)
        out = self.bottleneck(out)
        out = self.pool2(out)
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

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, activation=None, downsample=None):
        super(Bottleneck, self).__init__()
        self.activation = activation
        self.stride = stride

        self.gb1 = GatingBlock(inplanes, planes, 3, padding=1, initialization=False, activation=activation)
        self.gb2 = GatingBlock(planes, planes, 9, stride=stride, padding=4, initialization=False, activation=activation)
        self.gb3 = GatingBlock(planes, planes * self.expansion, 3, padding=1, initialization=False, activation=activation)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.gb1(x)
        out = self.gb2(out)
        out = self.gb3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out

class GatedResNet(nn.Module):
    def __init__(self, out_dims, block, layers, initialization=True, activation=None):
        self.inplanes = 256
        super(GatedResNet, self).__init__()
        self.activation = activation

        self.gb1 = GatingBlock(39, 256, 9, initialization=initialization, activation=activation)
        self.pool1 = nn.MaxPool1d(3)
        self.layer1 = self._make_layer(block, 64, layers[0], activation)
        self.layer2 = self._make_layer(block, 128, layers[1], activation)
        self.layer3 = self._make_layer(block, 256, layers[2], activation)
        self.layer4 = self._make_layer(block, 512, layers[3], activation)
        self.pool2 = nn.MaxPool1d(64)
        self.lgb = LinearGatingBlock(2048, 1024, activation=activation)
        self.fc = nn.Linear(1024, out_dims)
        self.softmax = torch.nn.Softmax(dim=1)

        if initialization:
            self.init_weights()

    def init_weights(self):
        self.fc.bias.data.fill_(0)
        nn.init.kaiming_uniform_(self.fc.weight)

    def _make_layer(self, block, planes, blocks, activation, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, activation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride, activation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.gb1(x)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool2(x)
        x = x.view(-1, 2048)
        x = self.lgb(x)
        x = self.fc(x)
        return x

    def get_embeds(self, out):
        return self.softmax(out)

class biLSTM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, nout, ninp, nhid, nlayers, dropout=0.4):
        super(biLSTM, self).__init__()
        self.lstm = nn.LSTM(ninp, nhid, nlayers, batch_first=True, dropout=dropout, bidirectional=True)
        self.decoder = nn.Linear(nhid*2, nout)

        self.nhid = nhid
        self.nlayers = nlayers
        self.ninp = ninp
        self.nout = nout

        self.init_weights()

    def forward(self, input, hidden):
        self.lstm.flatten_parameters()
        output, h = self.lstm(input, hidden)
        decoded = self.decoder(h[0][-2:, :, :].transpose(0, 1).contiguous().view(-1, self.nhid*2))
        return decoded, h

    def init_weights(self):
        initrange = 0.5
        self.lstm.named_parameters()
        for name, val in self.lstm.named_parameters():
            if name.find('bias') == -1:
                # getattr(self.lstm, name).data.uniform_(-initrange, initrange)
                getattr(self.lstm, name).data.normal_(0, math.sqrt(2. / (self.ninp + self.nhid)))
            else:
                getattr(self.lstm, name).data.fill_(0)

        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, math.sqrt(2. / (self.nhid*2 + self.nout))) #bidirectional

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers*2, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers*2, bsz, self.nhid).zero_()))

class CrossView(nn.Module):
    def __init__(self, submod, out_dims, margin=0.4, initialization=True, activation=None):
        super(CrossView, self).__init__()

        self.activation = activation
        self.margin = margin

        # Initialize constituent networks
        self.gc = submod(out_dims, initialization, activation)
        self.lstm = biLSTM(out_dims, 26, 512, 2)

        # Initialize optimizers
        # self.optimizer_gc = optim.Adam(self.gc.parameters())
        # self.optimizer_lstm = optim.Adam(self.lstm.parameters())

    def forward(self, achor, same, diff, hidden, batch_size):
        # Checkpoint
        # state_dict = self.lstm.state_dict()

        out_a, hidden = self.lstm(achor, hidden)

        out_s = self.gc(same)
        out_d = self.gc(diff)

        # Restore checkpoint to establish weight-sharing
#         self.lstm_d.load_state_dict(state_dict)
#         hidden_d = self.lstm_d.init_hidden(batch_size)
#         output_d, hidden_d = self.lstm_s(diff, hidden_d)

        return out_a, out_s, out_d, hidden

    def loss(self, achor, same, diff, hidden, batch_size):
        out_a, out_s, out_d, hidden = self.forward(achor, same, diff, hidden, batch_size)
        return torch.mean(F.relu(self.margin + F.cosine_similarity(out_a, out_d) - F.cosine_similarity(out_a, out_s))), hidden

    def get_embeds(self, x):
        return self.gc(x)
