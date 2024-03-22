from torch import nn
from torch.nn import Module, Conv2d, ConvTranspose2d, BatchNorm2d, ReLU


class Input_Block(Module):
    def __init__(self, features=3):
        super(Input_Block, self).__init__()
        self.conv = Conv2d(in_channels=features, out_channels=4,
                           stride=1, padding=1, kernel_size=(3, 3))
        self.bn = BatchNorm2d(4)
        self.relu = ReLU()

    def _init_weights(self, module):
        if isinstance(module, (Conv2d, BatchNorm2d)):
            if module.bias.data is not None:
                module.bias.data.zero_()
            else:
                nn.init.kaiming_normal_(
                    module.weight.data, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Output_Block(Module):
    def __init__(self, features):
        super(Output_Block, self).__init__()
        self.conv = ConvTranspose2d(
            in_channels=features, out_channels=3, stride=1, padding=1, kernel_size=(3, 3))
        self.bn = BatchNorm2d(3)
        self.relu = ReLU()

    def _init_weights(self, module):
        if isinstance(module, (Conv2d, BatchNorm2d)):
            if module.bias.data is not None:
                module.bias.data.zero_()
            else:
                nn.init.kaiming_normal_(
                    module.weight.data, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Enc_Block(Module):
    def __init__(self, features):
        super(Enc_Block, self).__init__()
        self.conv = Conv2d(in_channels=features, out_channels=2 *
                           features, stride=2, padding=1, kernel_size=(3, 3))
        self.bn = BatchNorm2d(2*features)
        self.relu = ReLU()

    def _init_weights(self, module):
        if isinstance(module, (Conv2d, BatchNorm2d)):
            if module.bias.data is not None:
                module.bias.data.zero_()
            else:
                nn.init.kaiming_normal_(
                    module.weight.data, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Dec_Block(Module):
    def __init__(self, features):
        super(Dec_Block, self).__init__()
        self.dconv = ConvTranspose2d(in_channels=features, out_channels=features //
                                     2, stride=2, padding=1, output_padding=1, kernel_size=(3, 3))
        self.bn = BatchNorm2d(features//2)
        self.relu = ReLU()

    def _init_weights(self, module):
        if isinstance(module, (ConvTranspose2d, BatchNorm2d)):
            if module.bias.data is not None:
                module.bias.data.zero_()
            else:
                nn.init.kaiming_normal_(
                    module.weightt.data, model='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.dconv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
