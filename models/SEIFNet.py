import torch
import torch.nn as nn
import torch.nn.functional as F
from . import ResNet


# ==================== Attention Modules ====================

class SEModule(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], 1)))


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        self.se = SEModule(channel, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.sa(self.se(x))


# ==================== Difference Module ====================

class CoDEM(nn.Module):
    """Contrastive Difference Enhancement Module."""
    def __init__(self, channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, 1, 1),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.cbam = CBAM(channel)

    def forward(self, x1, x2):
        diff = torch.abs(x1 - x2)
        concat = torch.cat([x1, x2], 1)
        concat = self.conv1(concat)
        concat = self.conv2(concat)
        out = diff + concat
        return self.cbam(out)


# ==================== Cross-level Fusion ====================

class ACFF(nn.Module):
    """Adaptive Cross-scale Feature Fusion."""
    def __init__(self, channel_L, channel_H):
        super().__init__()
        self.conv_low = nn.Sequential(
            nn.Conv2d(channel_L, channel_L, 3, 1, 1),
            nn.BatchNorm2d(channel_L), nn.ReLU(inplace=True))
        self.conv_high = nn.Sequential(
            nn.Conv2d(channel_H, channel_L, 1), nn.BatchNorm2d(channel_L))
        self.cbam = CBAM(channel_L)
        self.fuse = nn.Sequential(
            nn.Conv2d(channel_L * 2, channel_L, 3, 1, 1),
            nn.BatchNorm2d(channel_L), nn.ReLU(inplace=True))

    def forward(self, f_low, f_high):
        f_high_up = F.interpolate(f_high, size=f_low.shape[-2:],
                                   mode='bilinear', align_corners=True)
        f_low_t = self.conv_low(f_low)
        f_high_t = self.conv_high(f_high_up)
        flow = f_low_t * f_high_t
        flow = self.cbam(flow)
        return self.fuse(torch.cat([f_low, flow * f_high_t], 1))


# ==================== Supervised Attention ====================

class SupervisedAttentionModule(nn.Module):
    def __init__(self, mid_d):
        super().__init__()
        self.cls = nn.Conv2d(mid_d, 1, 1)
        self.conv_context = nn.Sequential(
            nn.Conv2d(2, mid_d, 1), nn.BatchNorm2d(mid_d), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_d, mid_d, 3, 1, 1),
            nn.BatchNorm2d(mid_d), nn.ReLU(inplace=True))

    def forward(self, x):
        mask = self.cls(x)
        mask_f = torch.sigmoid(mask)
        mask_b = 1 - mask_f
        context = self.conv_context(torch.cat([mask_f, mask_b], 1))
        x_out = self.conv2(x.mul(context))
        return x_out, mask


# ==================== Main SEIFNet ====================

class SEIFNet(nn.Module):
    """SEIFNet for CFB-Net framework.

    SEIFNet: A Network Combining Semantic Flow and Edge-Aware
    Refinement for Highly Efficient Change Detection (TGRS 2024)
    https://github.com/lixinghua5540/SEIFNet

    Uses ResNet18 backbone (original).
    """
    def __init__(self, input_nc=3, output_nc=1, backbone='resnet18', pretrained=True):
        super().__init__()
        if backbone == 'resnet18':
            self.backbone = ResNet.resnet18(pretrained=pretrained)
            channels = self.backbone.channels  # [64, 64, 128, 256, 512]
        elif backbone == 'resnet34':
            self.backbone = ResNet.resnet34(pretrained=pretrained)
            channels = self.backbone.channels
        elif backbone == 'resnet50':
            self.backbone = ResNet.resnet50(pretrained=pretrained)
            channels = self.backbone.channels  # [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # SEIFNet uses last 4 stages: c1, c2, c3, c4
        stage_dims = channels[1:]  # [64, 128, 256, 512] for r18

        # Diff modules
        self.diff1 = CoDEM(stage_dims[0])
        self.diff2 = CoDEM(stage_dims[1])
        self.diff3 = CoDEM(stage_dims[2])
        self.diff4 = CoDEM(stage_dims[3])

        # Cross-level fusion
        self.acff3 = ACFF(stage_dims[2], stage_dims[3])
        self.acff2 = ACFF(stage_dims[1], stage_dims[2])
        self.acff1 = ACFF(stage_dims[0], stage_dims[1])

        # Supervised attention
        self.sam_p4 = SupervisedAttentionModule(stage_dims[3])
        self.sam_p3 = SupervisedAttentionModule(stage_dims[2])
        self.sam_p2 = SupervisedAttentionModule(stage_dims[1])
        self.sam_p1 = SupervisedAttentionModule(stage_dims[0])

        # Output fusion
        self.conv4 = nn.Conv2d(stage_dims[3], 64, 1)
        self.conv3 = nn.Conv2d(stage_dims[2], 64, 1)
        self.conv2 = nn.Conv2d(stage_dims[1], 64, 1)
        self.conv_final = nn.Conv2d(64, output_nc, 1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        H, W = x1.size(2), x1.size(3)

        # ResNet forward (5 features: c0,c1,c2,c3,c4)
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)

        # Use c1~c4 (skip c0 which is before layer1)
        x1_feats = f1[1:]  # [64, 128, 256, 512]
        x2_feats = f2[1:]

        # Compute differences
        d1 = self.diff1(x1_feats[0], x2_feats[0])
        d2 = self.diff2(x1_feats[1], x2_feats[1])
        d3 = self.diff3(x1_feats[2], x2_feats[2])
        d4 = self.diff4(x1_feats[3], x2_feats[3])

        # Top-down refinement
        p4, mask_p4 = self.sam_p4(d4)
        acff_43 = self.acff3(d3, p4)
        p3, mask_p3 = self.sam_p3(acff_43)
        acff_32 = self.acff2(d2, p3)
        p2, mask_p2 = self.sam_p2(acff_32)
        acff_21 = self.acff1(d1, p2)
        p1, mask_p1 = self.sam_p1(acff_21)

        # Multi-scale fusion — all features at 32x32 due to dilation, upsample 2x to 64x64
        p4_up = self.conv4(self.upsample2(p4))
        p3_up = self.conv3(self.upsample2(p3))
        p2_up = self.conv2(self.upsample2(p2))
        p = p1 + p2_up + p3_up + p4_up
        p_up = self.upsample4(p)
        out = self.conv_final(p_up)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)

        m1 = torch.sigmoid(out)

        # Multi-scale outputs
        m2 = torch.sigmoid(F.interpolate(mask_p1, size=(H, W),
                                          mode='bilinear', align_corners=True))
        m3 = torch.sigmoid(F.interpolate(mask_p2, size=(H, W),
                                          mode='bilinear', align_corners=True))
        m4 = torch.sigmoid(F.interpolate(mask_p3, size=(H, W),
                                          mode='bilinear', align_corners=True))

        return m1, m2, m3, m4
