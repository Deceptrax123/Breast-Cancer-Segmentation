import torch
from Model.SegNet.blocks import Enc_Block, Dec_Block, Input_Block, Output_Block
from torch.nn import Module
import torch.nn.functional as f
from torchsummary import summary


class SegNetModel(Module):
    def __init__(self, initial_filters):
        super(SegNetModel, self).__init__()

        self.inp = Input_Block()

        self.enc1 = Enc_Block(initial_filters)
        self.enc2 = Enc_Block(initial_filters*2)
        self.enc3 = Enc_Block(initial_filters*4)
        self.enc4 = Enc_Block(initial_filters*8)
        self.enc5 = Enc_Block(initial_filters*16)
        self.enc6 = Enc_Block(initial_filters*32)
        self.enc7 = Enc_Block(initial_filters*64)

        self.dec1 = Dec_Block(initial_filters*128)
        self.dec2 = Dec_Block(initial_filters*64)
        self.dec3 = Dec_Block(initial_filters*32)
        self.dec4 = Dec_Block(initial_filters*16)
        self.dec5 = Dec_Block(initial_filters*8)
        self.dec6 = Dec_Block(initial_filters*4)
        self.dec7 = Dec_Block(initial_filters*2)

        self.classifier = Output_Block(initial_filters)

    def forward(self, x):
        x0 = self.inp(x)

        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)

        x = self.dec1(x7)
        x = torch.add(x, x6)

        x = self.dec2(x)
        x = torch.add(x, x5)

        x = self.dec3(x)
        x = torch.add(x, x4)

        x = self.dec4(x)
        x = torch.add(x, x3)

        x = self.dec5(x)
        x = torch.add(x, x2)

        x = self.dec6(x)
        x = torch.add(x, x1)

        x = self.dec7(x)
        x = torch.add(x, x0)

        x = self.classifier(x)

        return x, f.sigmoid(x)
