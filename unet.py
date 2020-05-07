from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

backbone = [1, 1, 1, 1]
encoder_channels = np.array([64, 128, 256, 512, 1024])*2
decoder_channels = np.array([512, 256, 128, 64])*2
stride = 2 # for 4 times

class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        # x: [B, C, H]
        s = F.adaptive_avg_pool1d(x, 1)  # [B, C, 1]
        s = self.conv1(s)  # [B, C//reduction, 1]
        s = F.relu(s, inplace=True)
        s = self.conv2(s)  # [B, C, 1]
        x = x + torch.sigmoid(s)
        return x


class ConvBR1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, dilation=1, stride=1, groups=1,
                 is_activation=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                              stride=stride, groups=groups, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.is_activation = is_activation

        if is_activation:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.is_activation:
            x = self.relu(x)
        return x


class SENextBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=32, reduction=16, pool=None, is_shortcut=False):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = ConvBR1d(in_channels, mid_channels, 1, 0, 1, )
        self.conv2 = ConvBR1d(mid_channels, mid_channels, 3, 1, 1, groups=groups)
        self.conv3 = ConvBR1d(mid_channels, out_channels, 1, 0, 1, is_activation=False)
        self.se = SEModule(out_channels, reduction)
        self.stride = stride
        self.is_shortcut = is_shortcut
        if is_shortcut:
            self.shortcut = ConvBR1d(in_channels, out_channels, 1, 0, 1, is_activation=False)
        if stride > 1:
            if pool == 'max':
                self.pool = nn.MaxPool1d(stride, stride)
            elif pool == 'avg':
                self.pool = nn.AvgPool1d(stride, stride)

    def forward(self, x):
        s = self.conv1(x)
        s = self.conv2(s)
        if self.stride > 1:
            s = self.pool(s)
        s = self.conv3(s)
        s = self.se(s)

        if self.is_shortcut:
            if self.stride > 1:
                x = F.avg_pool1d(x, self.stride, self.stride)  # avg
            x = self.shortcut(x)

        x = x + s
        x = F.relu(x, inplace=True)

        return x


class Encoder(nn.Module):
    def __init__(self, num_features=1):
        super().__init__()
        self.block0 = nn.Sequential(
            ConvBR1d(num_features, encoder_channels[0], kernel_size=5, stride=1, padding=2),
            ConvBR1d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),
            ConvBR1d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),
        )
        self.block1 = nn.Sequential(
            SENextBottleneck(encoder_channels[0], encoder_channels[1], stride=stride, is_shortcut=True, pool='max'),
            *[SENextBottleneck(encoder_channels[1], encoder_channels[1], stride=1, is_shortcut=False) for i in
              range(backbone[0])]
        )
        self.block2 = nn.Sequential(
            SENextBottleneck(encoder_channels[1], encoder_channels[2], stride=stride, is_shortcut=True, pool='max'),
            *[SENextBottleneck(encoder_channels[2], encoder_channels[2], stride=1, is_shortcut=False) for i in
              range(backbone[1])]
        )
        self.block3 = nn.Sequential(
            SENextBottleneck(encoder_channels[2], encoder_channels[3], stride=stride, is_shortcut=True, pool='max'),
            *[SENextBottleneck(encoder_channels[3], encoder_channels[3], stride=1, is_shortcut=False) for i in
              range(backbone[2])]
        )
        self.block4 = nn.Sequential(
            SENextBottleneck(encoder_channels[3], encoder_channels[4], stride=stride, is_shortcut=True, pool='avg'),
            *[SENextBottleneck(encoder_channels[4], encoder_channels[4], stride=1, is_shortcut=False) for i in
              range(backbone[3])]
        )

    def forward(self, x):
        x0 = self.block0(x)  # [B, 64, L]
        x1 = self.block1(x0)  # [B, 256, L//2]
        x2 = self.block2(x1)  # [B, 512, L//4]
        x3 = self.block3(x2)  # [B, 1024, L//8]
        x4 = self.block4(x3)  # [B, 2048, L//16]

        return [x0, x1, x2, x3, x4]


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv1d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBR1d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = ConvBR1d(out_channels, out_channels, kernel_size=3, padding=1)
        # att
        # self.att1 = SCSEModule(in_channels + skip_channels)
        # self.att2 = SCSEModule(out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=stride, mode="linear", align_corners=True)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            # x = self.att1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.att2(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block4 = DecoderBlock(encoder_channels[-1], encoder_channels[-2], decoder_channels[0])
        self.block3 = DecoderBlock(decoder_channels[0], encoder_channels[-3], decoder_channels[1])
        self.block2 = DecoderBlock(decoder_channels[1], encoder_channels[-4], decoder_channels[2])
        self.block1 = DecoderBlock(decoder_channels[2], encoder_channels[-5], decoder_channels[3])

    def forward(self, xs):
        x = self.block4(xs[4], xs[3])
        x = self.block3(x, xs[2])
        x = self.block2(x, xs[1])
        x = self.block1(x, xs[0])

        return x


class Unet(nn.Module):
    def __init__(self, num_features=1, num_classes=11):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.encoder = Encoder(num_features=num_features)
        self.decoder = Decoder()
        self.segmentation_head = nn.Conv1d(decoder_channels[-1], num_classes, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        features = self.encoder(x)
        x = self.decoder(features)
        x = self.segmentation_head(x)
        x = x.permute(0, 2, 1)

        return x



