import torch.nn as nn, torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, ch_in, ch_out, ch_mid=None):
        super().__init__()
        if ch_mid is None:
            ch_mid = ch_out
        self.layers = nn.Sequential(
            nn.Conv2d(ch_in, ch_mid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch_mid),
            nn.ReLU(),
            nn.Conv2d(ch_mid, ch_out, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class CNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg['ch_in']
        fts, ch_out = cfg['init_filters'], cfg['ch_out']
        n_layers = cfg['n_layers']

        self.bn_in = nn.BatchNorm2d(C) # input normalization for the lazy
        self.conv_in = nn.Conv2d(C, fts, 3, 1, 1, bias=False)
        self.bn0 = nn.BatchNorm2d(fts)
        
        self.conv_layers = nn.ModuleList([
            DoubleConv(2**k * fts, 2**(k+1) * fts) for k in range(n_layers)
        ])

        self.conv_out = nn.Conv2d(2**n_layers * fts, ch_out, 3, 1, 1)

    def forward(self, x):
        x = self.bn_in(x)
        x = F.relu(self.bn0(self.conv_in(x)))
        for layer in self.conv_layers:
            x = layer(x)
        return self.conv_out(x)
