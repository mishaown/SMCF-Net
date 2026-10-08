import torch
import torch.nn as nn
import torch.nn.functional as F
from . import MobileNetV2


class CMA(nn.Module):
    """Coordinate attention variant — channel + spatial attention."""
    def __init__(self, inp, oup, reduction=1):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, 1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, 1)
        self.conv3 = nn.Conv2d(mip, oup, 1)
        self.relu = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(inp, oup, 1) if inp != oup else nn.Identity()

    def forward(self, x):
        identity = self.proj(x)
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)
        return identity * x_w * x_h


class AFCFBlock(nn.Module):
    """AFCF fusion block for two adjacent feature levels."""
    def __init__(self, channel):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.conv_down = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.conv_cat = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1), nn.BatchNorm2d(channel))
        self.se = CMA(channel * 2, channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_high, x_low):
        # x_high has 2x resolution of x_low
        x_low_up = F.interpolate(x_low, scale_factor=2, mode='bilinear', align_corners=True)
        x_low_up = self.conv_up(x_low_up)
        feat = x_high + x_low_up
        feat = self.conv_cat(feat)
        feat_se = self.se(torch.cat([feat, x_high], 1))
        return feat_se + x_high


class AFCFBlock2(nn.Module):
    """AFCF fusion block for three adjacent feature levels."""
    def __init__(self, channel):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.conv_down = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.conv_cat = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1), nn.BatchNorm2d(channel))
        self.se = CMA(channel * 2, channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_high, x_mid, x_low):
        # x_high: 4x resolution of x_low, x_mid: 2x resolution of x_low
        x_high_down = self.conv_down(x_high)
        x_low_up = F.interpolate(x_low, scale_factor=2, mode='bilinear', align_corners=True)
        x_low_up = self.conv_up(x_low_up)
        feat = x_high_down + x_mid + x_low_up
        feat = self.conv_cat(feat)
        feat_se = self.se(torch.cat([feat, x_mid], 1))
        return feat_se + x_mid


class FeatureFusion(nn.Module):
    """Multi-level feature fusion."""
    def __init__(self, channel):
        super().__init__()
        self.afcf1 = AFCFBlock(channel)
        self.afcf2_1 = AFCFBlock2(channel)
        self.afcf2_2 = AFCFBlock2(channel)
        self.afcf2_3 = AFCFBlock2(channel)
        self.afcf3 = AFCFBlock(channel)

    def forward(self, x0, x1, x2, x3, x4):
        C1 = self.afcf1(x0, x1)
        C2 = self.afcf2_1(x0, x1, x2)
        C3 = self.afcf2_2(x1, x2, x3)
        C4 = self.afcf2_3(x2, x3, x4)
        C5 = self.afcf3(x3, x4)
        return C1, C2, C3, C4, C5


class DecoderBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1), nn.BatchNorm2d(channel))
        self.se = CMA(channel * 5, channel * 5)

    def forward(self, x_curr, x_high_list):
        # Upsample and fuse multi-level features
        h, w = x_curr.shape[-2:]
        upsampled = []
        for feat in x_high_list:
            if feat.shape[-2:] != (h, w):
                feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)
            upsampled.append(feat)
        concat = torch.cat(upsampled, 1)
        concat = self.se(concat)
        out = self.conv_cat(concat[:, :self.conv_cat[0].out_channels])
        return out


class AFCFNet(nn.Module):
    """AFCFNet adapted for CFB-Net framework.

    AFCFNet: Adaptive Frequency and Cross-scale Fusion Network for
    3D Change Detection (TGRS 2023)
    https://github.com/wm-Githuber/AFCF3D-Net

    Note: Adapted from 3D to work with standard 2D change detection.
    """
    def __init__(self, input_nc=3, output_nc=1, pretrained=True):
        super().__init__()
        self.backbone = MobileNetV2.mobilenet_v2(pretrained=pretrained)
        channels = [16, 24, 32, 96, 320]
        channel = 64
        self.en_d = 32
        self.mid_d = self.en_d * 2

        # Reduce backbone channels
        self.reduction0 = nn.Sequential(
            nn.Conv2d(channels[0], channel, 1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.reduction1 = nn.Sequential(
            nn.Conv2d(channels[1], channel, 1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.reduction2 = nn.Sequential(
            nn.Conv2d(channels[2], channel, 1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.reduction3 = nn.Sequential(
            nn.Conv2d(channels[3], channel, 1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.reduction4 = nn.Sequential(
            nn.Conv2d(channels[4], channel, 1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))

        self.feature_fusion = FeatureFusion(channel)

        # Decoder
        self.conv_upsample = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.conv_downsample = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))

        self.decoder3 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1))
        self.decoder2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1))
        self.decoder1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1))

        self.out_conv = nn.Sequential(
            nn.Conv2d(channel, channel * 2, 1), nn.BatchNorm2d(channel * 2), nn.ReLU(inplace=True),
            nn.Conv2d(channel * 2, channel, 1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True),
            nn.Conv2d(channel, output_nc, 1))
        self.aux2 = nn.Conv2d(channel, output_nc, 1)
        self.aux3 = nn.Conv2d(channel, output_nc, 1)
        self.aux4 = nn.Conv2d(channel, output_nc, 1)

    def forward(self, x1, x2):
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone(x2)

        # Compute difference features at each level
        d0 = torch.abs(x1_1 - x2_1)
        d1 = torch.abs(x1_2 - x2_2)
        d2 = torch.abs(x1_3 - x2_3)
        d3 = torch.abs(x1_4 - x2_4)
        d4 = torch.abs(x1_5 - x2_5)

        d0 = self.reduction0(d0)
        d1 = self.reduction1(d1)
        d2 = self.reduction2(d2)
        d3 = self.reduction3(d3)
        d4 = self.reduction4(d4)

        C1, C2, C3, C4, C5 = self.feature_fusion(d0, d1, d2, d3, d4)

        # Decode from deep to shallow
        size = C1.shape[2:]
        C5_up = F.interpolate(C5, size=size, mode='bilinear', align_corners=True)
        C4_up = F.interpolate(C4, size=size, mode='bilinear', align_corners=True)
        C3_up = F.interpolate(C3, size=size, mode='bilinear', align_corners=True)
        C2_up = F.interpolate(C2, size=size, mode='bilinear', align_corners=True)

        out = C1 + C2_up + C3_up + C4_up + C5_up
        out = self.out_conv(out)
        m1 = torch.sigmoid(F.interpolate(out, scale_factor=2, mode='bilinear'))

        # Multi-scale outputs
        m2 = torch.sigmoid(F.interpolate(self.aux2(C2), size=m1.shape[2:], mode='bilinear'))
        m3 = torch.sigmoid(F.interpolate(self.aux3(C3), size=m1.shape[2:], mode='bilinear'))
        m4 = torch.sigmoid(F.interpolate(self.aux4(C4), size=m1.shape[2:], mode='bilinear'))

        return m1, m2, m3, m4
