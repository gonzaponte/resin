import torch
import torch.nn as nn
import torch.nn.functional as fnal

_nn_registry = {}
def register(c):
    _nn_registry[c.__name__] = c
    return c


@register
class NN_linear_v0(nn.Module):
    def __init__(self, nsipms, drop=False):
        super().__init__()
        self.layer1 = nn.Linear(nsipms, 64, dtype=torch.float64)
        self.layer2 = nn.Linear(64    ,  2, dtype=torch.float64)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x =            self.layer2(x)
        return x

@register
class NN_linear_v1(nn.Module):
    def __init__(self, nsipms):
        super().__init__()
        nreco    = 2
        self.layer1 = nn.Linear(nsipms, 512, dtype=torch.float64)
        self.layer2 = nn.Linear(   512,  64, dtype=torch.float64)
        self.layer3 = nn.Linear(    64,   2, dtype=torch.float64)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x =            self.layer3(x)
        return x

@register
class NN_conv_v0(nn.Module):
    def __init__(self, nsipms):
        super().__init__()
        nreco    = 2
        size     = 2**6
        self.layer1 = nn.Conv2d     (     1, size*1, 4, padding=1, dtype=torch.float64)
        self.norm1  = nn.BatchNorm2d(size*1,                       dtype=torch.float64)
        self.layer2 = nn.Conv2d     (size*1, size*2, 3, padding=1, dtype=torch.float64)
        self.norm2  = nn.BatchNorm2d(size*2,                       dtype=torch.float64)
        self.layer3 = nn.Conv2d     (size*2, size*4, 2, padding=1, dtype=torch.float64)
        self.norm3  = nn.BatchNorm2d(size*4,                       dtype=torch.float64)
        self.layer4 = nn.Linear     (size*4,  nreco,               dtype=torch.float64)

        self.pool   = nn.MaxPool2d(2, 4)
        self.drop   = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x.reshape(len(x), 1, nsipms_side, nsipms_side)
        x = self.pool(self.norm1(fnal.leaky_relu(self.layer1(x))))
        x = self.pool(self.norm2(fnal.leaky_relu(self.layer2(x))))
        x = self.pool(self.norm3(fnal.leaky_relu(self.layer3(x))))
        x = x.flatten(start_dim=1)
        x =                                      self.layer4(x)
        return x


@register
class NN_conv_v1(nn.Module):
    def __init__(self, nsipms):
        super().__init__()
        nreco    = 2
        size     = 2**6
        self.layer1 = nn.Conv2d     (     1, size*1, 4, padding=2, dtype=torch.float64)
        self.norm1  = nn.BatchNorm2d(size*1,                       dtype=torch.float64)
        self.layer2 = nn.Conv2d     (size*1, size*2, 4, padding=2, dtype=torch.float64)
        self.norm2  = nn.BatchNorm2d(size*2,                       dtype=torch.float64)
        self.layer3 = nn.Linear     (size*2,  nreco,               dtype=torch.float64)

        self.pool   = nn.MaxPool2d(2, 4)

    def forward(self, x):
        x = x.reshape(len(x), 1, nsipms_side, nsipms_side)
        x = self.pool(self.norm1(fnal.leaky_relu(self.layer1(x))))
        x = self.pool(self.norm2(fnal.leaky_relu(self.layer2(x))))
        x = x.flatten(start_dim=1)
        x =                                      self.layer3(x)
        return x

@register
class NN_linear_v2(nn.Module):
    def __init__(self, nsipms):
        super().__init__()
        self.layer1 = nn.Linear(nsipms, 128, dtype=torch.float64)
        self.layer2 = nn.Linear(   128,  64, dtype=torch.float64)
        self.layer3 = nn.Linear(    64,   2, dtype=torch.float64)

    def forward(self, x):
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer1(x))
        x =            self.layer3(x)
        return x

@register
class NN_conv_v2(nn.Module):
    def __init__(self, nsipms):
        super().__init__()
        self.layer1 = nn.Conv2d(     1,  16, kernel_size=3, stride=1, padding=1, dtype=torch.float64)
        self.layer2 = nn.Conv2d(    16,  32, kernel_size=3, stride=1, padding=1, dtype=torch.float64)
        self.layer3 = nn.Conv2d(    32,  64, kernel_size=3, stride=1, padding=1, dtype=torch.float64)
        self.layer4 = nn.Linear(2*2*64, 128,                                     dtype=torch.float64)
        self.layer5 = nn.Linear(   128,  64,                                     dtype=torch.float64)
        self.layer6 = nn.Linear(    64,   2,                                     dtype=torch.float64)
        self.pool   = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = x.reshape(len(x), 1, nsipms_side, nsipms_side)
        x = self.pool(torch.relu(self.layer1(x)))
        x = self.pool(torch.relu(self.layer2(x)))
        x = self.pool(torch.relu(self.layer3(x)))
        x = x.view(-1, 2*2*64)
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        x =            self.layer6(x)
        return x


def to_arch(name : str) -> nn.Module:
    print("Enteting")
    try:
        return _nn_registry["NN_" + name]
    except KeyError:
        options = "\n- " + "\n- ".join(
            map(lambda x: x[3:],
                filter(lambda x: x[0].startswith("NN_", ),
                       _nn_registry.items())))
        print(f"No architecture named >{name}<. Available options are: {options}")
        raise
