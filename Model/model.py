import torch
from torch.nn import Conv2d, Module
from Model.unet import Unet


class CombinedModel(Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=4,
                            kernel_size=(3, 3), padding=1, stride=1)
        self.unet = Unet(filters=4)
        self.output = Conv2d(in_channels=4, out_channels=1,
                             kernel_size=(3, 3), padding=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.unet(x)
        x = self.output(x)

        return x
