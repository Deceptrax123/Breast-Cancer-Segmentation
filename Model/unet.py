import torch
from torch import nn
from Model.blocks import Unet_Decoding_Block, Unet_Encoding_Block
from torch.nn import Module, ConvTranspose2d, Conv2d, MaxPool2d, Dropout2d
from torchsummary import summary


class Unet(Module):
    def __init__(self, filters):
        super(Unet, self).__init__()

        # Downsampling Blocks
        self.down1 = Unet_Encoding_Block(filters)
        self.down2 = Unet_Encoding_Block(filters*2)
        self.down3 = Unet_Encoding_Block(filters*4)
        self.down4 = Unet_Encoding_Block(filters*8)
        self.down5 = Unet_Encoding_Block(filters*16)
        self.down6 = Unet_Encoding_Block(filters*32)
        self.down7 = Unet_Encoding_Block(filters*64)

        # Bottleneck
        self.emb1 = Conv2d(in_channels=filters*128, out_channels=filters *
                           128, stride=1, padding=1, kernel_size=(3, 3))
        self.emb2 = Conv2d(in_channels=filters*128, out_channels=filters *
                           128, stride=1, padding=1, kernel_size=(3, 3))

        # Upsampling Blocks
        self.up1 = Unet_Decoding_Block(filters*128)
        self.up2 = Unet_Decoding_Block(filters*64)
        self.up3 = Unet_Decoding_Block(filters*32)
        self.up4 = Unet_Decoding_Block(filters*16)
        self.up5 = Unet_Decoding_Block(filters*8)
        self.up6 = Unet_Decoding_Block(filters*4)
        self.up7 = Unet_Decoding_Block(filters*2)

        # maxpooling
        self.max1 = MaxPool2d(kernel_size=(2, 2), stride=2)
        self.max2 = MaxPool2d(kernel_size=(2, 2), stride=2)
        self.max3 = MaxPool2d(kernel_size=(2, 2), stride=2)
        self.max4 = MaxPool2d(kernel_size=(2, 2), stride=2)
        self.max5 = MaxPool2d(kernel_size=(2, 2), stride=2)
        self.max6 = MaxPool2d(kernel_size=(2, 2), stride=2)
        self.max7 = MaxPool2d(kernel_size=(2, 2), stride=2)

        # Deconvolution Blocks
        self.dconv1 = ConvTranspose2d(in_channels=filters*128, out_channels=filters*128,
                                      stride=2, padding=1, output_padding=1, kernel_size=(3, 3))
        self.dconv2 = ConvTranspose2d(in_channels=filters*64, out_channels=filters*64,
                                      stride=2, padding=1, output_padding=1, kernel_size=(3, 3))
        self.dconv3 = ConvTranspose2d(in_channels=filters*32, out_channels=filters*32,
                                      stride=2, padding=1, kernel_size=(3, 3), output_padding=1)
        self.dconv4 = ConvTranspose2d(in_channels=filters*16, out_channels=filters*16,
                                      stride=2, padding=1, output_padding=1, kernel_size=(3, 3))
        self.dconv5 = ConvTranspose2d(in_channels=filters*8, out_channels=filters*8,
                                      stride=2, padding=1, output_padding=1, kernel_size=(3, 3))
        self.dconv6 = ConvTranspose2d(in_channels=filters*4, out_channels=filters*4,
                                      stride=2, padding=1, kernel_size=(3, 3), output_padding=1)
        self.dconv7 = ConvTranspose2d(in_channels=filters*2, out_channels=filters*2,
                                      stride=2, padding=1, kernel_size=(3, 3), output_padding=1)

        # Dropouts to prevent overfitting
        self.dp1 = Dropout2d()
        self.dp2 = Dropout2d()
        self.dp3 = Dropout2d()
        self.dp4 = Dropout2d()
        self.dp5 = Dropout2d()
        self.dp6 = Dropout2d()

    def forward(self, x):
        # Down sampling
        x1 = self.down1(x)
        x1_max = self.max1(x1)

        x2 = self.down2(x1_max)
        x2_max = self.max2(x2)

        x3 = self.down3(x2_max)
        x3_max = self.max3(x3)

        x4 = self.down4(x3_max)
        x4_max = self.max4(x4)

        x5 = self.down5(x4_max)
        x5_max = self.max5(x5)

        x6 = self.down6(x5_max)
        x6_max = self.max6(x6)

        x7 = self.down7(x6_max)
        x7_max = self.max7(x7)

        x8 = self.emb1(x7_max)
        x9 = self.emb2(x8)

        # Upsampling
        x10 = self.dconv1(x9)
        xcat1 = torch.add(x10, x7)
        xu1 = self.up1(xcat1)

        x11 = self.dconv2(xu1)
        xcat2 = torch.add(x11, x6)
        xu2 = self.up2(xcat2)

        x12 = self.dconv3(xu2)
        xcat3 = torch.add(x12, x5)
        xu3 = self.up3(xcat3)

        x13 = self.dconv4(xu3)
        xcat4 = torch.add(x13, x4)
        xu4 = self.up4(xcat4)

        x14 = self.dconv5(xu4)
        xcat5 = torch.add(x14, x3)
        xu5 = self.up5(xcat5)

        x15 = self.dconv6(xu5)
        xcat6 = torch.add(x15, x2)
        xu6 = self.up6(xcat6)

        x16 = self.dconv7(xu6)
        xcat7 = torch.add(x16, x1)

        output = self.up7(xcat7)

        return output
