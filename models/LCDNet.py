import torch
import torch.nn as nn
import torch.nn.functional as F
from . import MobileNetV2


class GMM(nn.Module):
    """Gated Mixing Module — feature gating mechanism."""
    def __init__(self, num_channels, epsilon=1e-5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon

    def forward(self, x):
        embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
        norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        gate = 1. + torch.tanh(embedding * norm + self.beta)
        return x * gate


class ChannelExchange(nn.Module):
    """Channel exchange between bi-temporal features."""
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, x1, x2):
        N, C, H, W = x1.shape
        exchange_mask = torch.arange(C, device=x1.device) % self.p == 0
        exchange_mask = exchange_mask.unsqueeze(0).expand((N, -1))
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[:, ~exchange_mask[0]] = x1[:, ~exchange_mask[0]]
        out_x2[:, ~exchange_mask[0]] = x2[:, ~exchange_mask[0]]
        out_x1[:, exchange_mask[0]] = x2[:, exchange_mask[0]]
        out_x2[:, exchange_mask[0]] = x1[:, exchange_mask[0]]
        return out_x1, out_x2


class SqueezeDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels), nn.GELU())
        self.double_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.BatchNorm2d(out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        self.acfun = nn.GELU()
        self.gmm = GMM(out_channels)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.gmm(x)
        block_x = self.double_conv(x)
        return self.acfun(x + block_x)


class FFM(nn.Module):
    """Feature Fusion Module."""
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = self.relu(self.conv1(x1))
        x2 = self.relu(self.conv1(x2))
        x = x1 * x2
        x = self.relu(x)
        x = x + x2
        x = x * x1
        return self.relu(x)


class LCDNet(nn.Module):
    """LCD-Net adapted for CFB-Net framework.

    LCD-Net: Lightweight Change Detection with Channel Exchange and Gating (JSTARS 2025)
    https://github.com/WenyuLiu6/LCD-Net
    """
    def __init__(self, input_nc=3, output_nc=1, pretrained=True):
        super().__init__()

        mob = MobileNetV2.mobilenet_v2(pretrained=pretrained)
        features = mob.features
        self.inc = features[:2]       # 16
        self.down1 = features[2:4]    # 24
        self.down2 = features[4:7]    # 32
        self.down3 = features[7:14]   # 96
        self.down4 = features[14:18]  # 320

        self.cont1 = SqueezeDoubleConv(24, 64)
        self.cont2 = SqueezeDoubleConv(32, 64)
        self.cont3 = SqueezeDoubleConv(96, 64)
        self.cont4 = SqueezeDoubleConv(320, 64)

        total_ch = 16 + 24 + 32 + 96 + 320
        self.decoder = nn.Sequential(SqueezeDoubleConv(total_ch, 64), nn.Conv2d(64, 1, 1))
        self.decoder_4 = SqueezeDoubleConv(64 * 2 + 1, 64)
        self.decoder_3 = SqueezeDoubleConv(64 * 3 + 1, 64)
        self.decoder_2 = SqueezeDoubleConv(64 * 3 + 1, 64)
        self.decoder_1 = SqueezeDoubleConv(64 * 3 + 1, 64)
        self.decoder_final = nn.Sequential(SqueezeDoubleConv(64, 64), nn.Conv2d(64, 1, 1))

        self.chan = ChannelExchange()
        self.ffm = FFM(total_ch, total_ch)
        self.aux3 = nn.Conv2d(64, output_nc, 1)
        self.aux4 = nn.Conv2d(64, output_nc, 1)

    def forward(self, x1, x2):
        size = x1.size()[2:]

        layer1_pre = self.inc(x1)
        layer2_pre = self.inc(x2)
        layer1_A = self.down1(layer1_pre)
        layer1_B = self.down1(layer2_pre)
        layer2_A = self.down2(layer1_A)
        layer2_B = self.down2(layer1_B)
        layer2_A, layer2_B = self.chan(layer2_A, layer2_B)
        layer3_A = self.down3(layer2_A)
        layer3_B = self.down3(layer2_B)
        layer3_A, layer3_B = self.chan(layer3_A, layer3_B)
        layer4_A = self.down4(layer3_A)
        layer4_B = self.down4(layer3_B)
        layer4_A, layer4_B = self.chan(layer4_A, layer4_B)

        ref_size = layer1_A.size()[2:]

        # Collect multi-level features (including inc level)
        layer0_As = F.interpolate(layer1_pre, ref_size, mode='bilinear', align_corners=True)
        layer1_As = F.interpolate(layer1_A, ref_size, mode='bilinear', align_corners=True)
        layer2_As = F.interpolate(layer2_A, ref_size, mode='bilinear', align_corners=True)
        layer3_As = F.interpolate(layer3_A, ref_size, mode='bilinear', align_corners=True)
        layer4_As = F.interpolate(layer4_A, ref_size, mode='bilinear', align_corners=True)
        layer0_Bs = F.interpolate(layer2_pre, ref_size, mode='bilinear', align_corners=True)
        layer1_Bs = F.interpolate(layer1_B, ref_size, mode='bilinear', align_corners=True)
        layer2_Bs = F.interpolate(layer2_B, ref_size, mode='bilinear', align_corners=True)
        layer3_Bs = F.interpolate(layer3_B, ref_size, mode='bilinear', align_corners=True)
        layer4_Bs = F.interpolate(layer4_B, ref_size, mode='bilinear', align_corners=True)

        layer_As = torch.cat([layer0_As, layer1_As, layer2_As, layer3_As, layer4_As], 1)
        layer_Bs = torch.cat([layer0_Bs, layer1_Bs, layer2_Bs, layer3_Bs, layer4_Bs], 1)
        layer_ss = self.ffm(layer_As, layer_Bs)

        layer1_A = self.cont1(layer1_A)
        layer2_A = self.cont2(layer2_A)
        layer3_A = self.cont3(layer3_A)
        layer4_A = self.cont4(layer4_A)
        layer1_B = self.cont1(layer1_B)
        layer2_B = self.cont2(layer2_B)
        layer3_B = self.cont3(layer3_B)
        layer4_B = self.cont4(layer4_B)

        layer1 = torch.cat((layer1_B, layer1_A), 1)
        layer2 = torch.cat((layer2_B, layer2_A), 1)
        layer3 = torch.cat((layer3_B, layer3_A), 1)
        layer4 = torch.cat((layer4_B, layer4_A), 1)

        layer4_1 = F.interpolate(layer4, ref_size, mode='bilinear', align_corners=True)
        layer3_1 = F.interpolate(layer3, ref_size, mode='bilinear', align_corners=True)
        layer2_1 = F.interpolate(layer2, ref_size, mode='bilinear', align_corners=True)
        layer1_1 = F.interpolate(layer1, ref_size, mode='bilinear', align_corners=True)

        change_map = self.decoder(layer_ss)
        change_map = F.interpolate(change_map, size, mode='bilinear', align_corners=True)
        change_map1 = F.interpolate(change_map, ref_size, mode='bilinear', align_corners=True)

        layer4_1 = torch.cat([layer4_1, change_map1], 1)
        layer4_1 = self.decoder_4(layer4_1)
        layer3_1 = torch.cat([layer4_1, layer3_1, change_map1], 1)
        layer3_1 = self.decoder_3(layer3_1)
        layer2_1 = torch.cat([layer3_1, layer2_1, change_map1], 1)
        layer2_1 = self.decoder_2(layer2_1)
        layer1_1 = torch.cat([layer2_1, layer1_1, change_map1], 1)
        layer1_1 = self.decoder_1(layer1_1)
        final_map = self.decoder_final(layer1_1)
        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True)

        m1 = torch.sigmoid(final_map)
        m2 = torch.sigmoid(change_map)
        m3 = torch.sigmoid(F.interpolate(
            self.aux3(layer3_1), size, mode='bilinear', align_corners=True))
        m4 = torch.sigmoid(F.interpolate(
            self.aux4(layer4_1), size, mode='bilinear', align_corners=True))

        return m1, m2, m3, m4
