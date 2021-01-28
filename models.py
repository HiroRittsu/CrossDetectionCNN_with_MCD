import torch
import torch.nn as nn
from torch.nn import init


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mp_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        x = self.mp_conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, mode='nearest'):
        super(Up, self).__init__()
        if mode is not None:
            self.up = nn.Upsample(scale_factor=2, mode=mode)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        """diff_x = x1.size()[2] - x2.size()[2]
        diff_y = x1.size()[3] - x2.size()[3]
        x2 = f.pad(x2, [diff_x // 2, diff_x // 2, diff_y // 2, diff_y // 2])"""
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch, sig):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.act = nn.Sigmoid()
        self.sig = sig

    def forward(self, x):
        x = self.conv(x)
        if self.sig:
            x = self.act(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_channels):
        super(Encoder, self).__init__()
        self.inc = InConv(n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(256, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)

    def forward(self, x1, x2):
        x2_t1 = self.inc(x1)
        x2_t2 = self.inc(x2)

        x3_t1 = self.down1(x2_t1)
        x3_t2 = self.down1(x2_t2)

        x4 = torch.cat([x3_t1, x3_t2], dim=1)

        x5 = self.down2(x4)
        x6 = self.down3(x5)
        x7 = self.down4(x6)

        x8 = self.up1(x7, x6)
        x9 = self.up2(x8, x5)

        return (x2_t1, x3_t1), (x2_t2, x3_t2), x9


class Decoder(nn.Module):
    def __init__(self, n_classes, sig=False):
        super(Decoder, self).__init__()

        self.up3 = Up(256, 64)
        self.up4 = Up(128, 32)
        self.out = OutConv(32, n_classes, sig=sig)

    def forward(self, skip, x):
        x2, x3 = skip
        x9 = x

        x10 = self.up3(x9, x3)
        x11 = self.up4(x10, x2)

        out = self.out(x11)

        return out
