import torch
import torch.nn as nn
import torch.nn.functional as F
from . import LWGANet_backbone


# ==================== Neighbor Feature Aggregation ====================

class FeatureFusionModule(nn.Module):
    def __init__(self, fuse_d, id_d, out_d):
        super().__init__()
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(fuse_d, out_d, 3, 1, 1), nn.BatchNorm2d(out_d), nn.ReLU(inplace=True),
            nn.Conv2d(out_d, out_d, 3, 1, 1), nn.BatchNorm2d(out_d))
        self.conv_identity = nn.Conv2d(id_d, out_d, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, c_fuse, c):
        return self.relu(self.conv_fuse(c_fuse) + self.conv_identity(c))


class NeighborFeatureAggregation(nn.Module):
    """Sliding-window neighbor feature aggregation across 4 scales."""
    def __init__(self, in_d, out_d=64):
        super().__init__()
        self.in_d = in_d  # [c0, c1, c2, c3] from LWGANet
        self.mid_d = out_d // 2
        self.out_d = out_d

        self.conv_scale2_c0 = nn.Sequential(
            nn.Conv2d(in_d[0], self.mid_d, 3, 1, 1), nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale2_c1 = nn.Sequential(
            nn.Conv2d(in_d[1], self.mid_d, 3, 1, 1), nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_aggregation_s2 = FeatureFusionModule(self.mid_d * 2, in_d[0], out_d)

        self.conv_scale3_c0 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_d[0], self.mid_d, 3, 1, 1), nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale3_c1 = nn.Sequential(
            nn.Conv2d(in_d[1], self.mid_d, 3, 1, 1), nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale3_c2 = nn.Sequential(
            nn.Conv2d(in_d[2], self.mid_d, 3, 1, 1), nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_aggregation_s3 = FeatureFusionModule(self.mid_d * 3, in_d[1], out_d)

        self.conv_scale4_c1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_d[1], self.mid_d, 3, 1, 1), nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale4_c2 = nn.Sequential(
            nn.Conv2d(in_d[2], self.mid_d, 3, 1, 1), nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale4_c3 = nn.Sequential(
            nn.Conv2d(in_d[3], self.mid_d, 3, 1, 1), nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_aggregation_s4 = FeatureFusionModule(self.mid_d * 3, in_d[2], out_d)

        self.conv_scale5_c2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_d[2], self.mid_d, 3, 1, 1), nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale5_c3 = nn.Sequential(
            nn.Conv2d(in_d[3], self.mid_d, 3, 1, 1), nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_aggregation_s5 = FeatureFusionModule(self.mid_d * 2, in_d[3], out_d)

    def forward(self, c0, c1, c2, c3):
        c0_s0 = self.conv_scale2_c0(c0)
        c1_s0 = F.interpolate(self.conv_scale2_c1(c1), scale_factor=(2, 2), mode='bilinear')
        s0 = self.conv_aggregation_s2(torch.cat([c0_s0, c1_s0], 1), c0)

        c0_s1 = self.conv_scale3_c0(c0)
        c1_s1 = self.conv_scale3_c1(c1)
        c2_s1 = F.interpolate(self.conv_scale3_c2(c2), scale_factor=(2, 2), mode='bilinear')
        s1 = self.conv_aggregation_s3(torch.cat([c0_s1, c1_s1, c2_s1], 1), c1)

        c1_s2 = self.conv_scale4_c1(c1)
        c2_s2 = self.conv_scale4_c2(c2)
        c3_s2 = F.interpolate(self.conv_scale4_c3(c3), scale_factor=(2, 2), mode='bilinear')
        s2 = self.conv_aggregation_s4(torch.cat([c1_s2, c2_s2, c3_s2], 1), c2)

        c2_s3 = self.conv_scale5_c2(c2)
        c3_s3 = self.conv_scale5_c3(c3)
        s3 = self.conv_aggregation_s5(torch.cat([c2_s3, c3_s3], 1), c3)

        return s0, s1, s2, s3


# ==================== Temporal Fusion ====================

class TemporalFeatureFusionModule(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv_branch1 = nn.Sequential(
            nn.Conv2d(in_d, in_d, 3, 1, 7, dilation=7), nn.BatchNorm2d(in_d))
        self.conv_branch2 = nn.Conv2d(in_d, in_d, 1)
        self.conv_branch2_f = nn.Sequential(
            nn.Conv2d(in_d, in_d, 3, 1, 5, dilation=5), nn.BatchNorm2d(in_d))
        self.conv_branch3 = nn.Conv2d(in_d, in_d, 1)
        self.conv_branch3_f = nn.Sequential(
            nn.Conv2d(in_d, in_d, 3, 1, 3, dilation=3), nn.BatchNorm2d(in_d))
        self.conv_branch4 = nn.Conv2d(in_d, in_d, 1)
        self.conv_branch4_f = nn.Sequential(
            nn.Conv2d(in_d, out_d, 3, 1, 1, dilation=1), nn.BatchNorm2d(out_d))
        self.conv_branch5 = nn.Conv2d(in_d, out_d, 1)

    def forward(self, x1, x2):
        x = torch.abs(x1 - x2)
        x1b = self.conv_branch1(x)
        x2b = self.relu(self.conv_branch2(x) + x1b)
        x2b = self.conv_branch2_f(x2b)
        x3b = self.relu(self.conv_branch3(x) + x2b)
        x3b = self.conv_branch3_f(x3b)
        x4b = self.relu(self.conv_branch4(x) + x3b)
        x4b = self.conv_branch4_f(x4b)
        return self.relu(self.conv_branch5(x) + x4b)


class TemporalFusionModule(nn.Module):
    def __init__(self, in_d=32, out_d=32):
        super().__init__()
        self.tffm_x0 = TemporalFeatureFusionModule(in_d, out_d)
        self.tffm_x1 = TemporalFeatureFusionModule(in_d, out_d)
        self.tffm_x2 = TemporalFeatureFusionModule(in_d, out_d)
        self.tffm_x3 = TemporalFeatureFusionModule(in_d, out_d)

    def forward(self, x1_0, x1_1, x1_2, x1_3, x2_0, x2_1, x2_2, x2_3):
        return (self.tffm_x0(x1_0, x2_0), self.tffm_x1(x1_1, x2_1),
                self.tffm_x2(x1_2, x2_2), self.tffm_x3(x1_3, x2_3))


# ==================== Decoder ====================

class SupervisedAttentionModule(nn.Module):
    def __init__(self, mid_d):
        super().__init__()
        self.cls = nn.Conv2d(mid_d, 1, 1)
        self.conv_context = nn.Sequential(
            nn.Conv2d(2, mid_d, 1), nn.BatchNorm2d(mid_d), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_d, mid_d, 3, 1, 1), nn.BatchNorm2d(mid_d), nn.ReLU(inplace=True))

    def forward(self, x):
        mask = self.cls(x)
        mask_f = torch.sigmoid(mask)
        mask_b = 1 - mask_f
        context = self.conv_context(torch.cat([mask_f, mask_b], 1))
        return self.conv2(x.mul(context)), mask


class Decoder(nn.Module):
    def __init__(self, mid_d=64):
        super().__init__()
        self.sam_p4 = SupervisedAttentionModule(mid_d)
        self.sam_p3 = SupervisedAttentionModule(mid_d)
        self.sam_p2 = SupervisedAttentionModule(mid_d)
        self.conv_p3 = nn.Sequential(
            nn.Conv2d(mid_d, mid_d, 3, 1, 1), nn.BatchNorm2d(mid_d), nn.ReLU(inplace=True))
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(mid_d, mid_d, 3, 1, 1), nn.BatchNorm2d(mid_d), nn.ReLU(inplace=True))
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(mid_d, mid_d, 3, 1, 1), nn.BatchNorm2d(mid_d), nn.ReLU(inplace=True))
        self.cls = nn.Conv2d(mid_d, 1, 1)

    def forward(self, d0, d1, d2, d3):
        p3, mask_p3 = self.sam_p4(d3)
        p2 = self.conv_p3(d2 + F.interpolate(p3, scale_factor=2, mode='bilinear'))
        p2, mask_p2 = self.sam_p3(p2)
        p1 = self.conv_p2(d1 + F.interpolate(p2, scale_factor=2, mode='bilinear'))
        p1, mask_p1 = self.sam_p2(p1)
        p0 = self.conv_p1(d0 + F.interpolate(p1, scale_factor=2, mode='bilinear'))
        mask_p0 = self.cls(p0)
        return mask_p0, mask_p1, mask_p2, mask_p3


# ==================== Main Model ====================

class LWGANet_CD(nn.Module):
    """LWGANet Change Detection for CFB-Net framework.

    LWGANet: Lightweight Global-Aware Network for Change Detection (AAAI 2026)
    https://github.com/AeroVILab-AHU/LWGANet

    Uses full LWGANet-L0 backbone with multi-range attention
    (PA, LA, MRA, GA12/D_GA/GA).
    """
    def __init__(self, input_nc=3, output_nc=1, variant='L0', pretrained=True):
        super().__init__()
        if variant == 'L0':
            self.backbone = LWGANet_backbone.lwganet_l0(pretrained=pretrained)
            # channels = [32, 64, 128, 256]
        elif variant == 'L1':
            self.backbone = LWGANet_backbone.lwganet_l1(pretrained=pretrained)
            # channels = [64, 128, 256, 512]
        elif variant == 'L2':
            self.backbone = LWGANet_backbone.lwganet_l2(pretrained=pretrained)
            # channels = [96, 192, 384, 768]
        else:
            raise ValueError(f"Unknown variant: {variant}")

        channels = self.backbone.channels  # 4 channels from LWGANet
        self.en_d = 32
        self.mid_d = self.en_d * 2  # 64

        self.swa = NeighborFeatureAggregation(channels, self.mid_d)
        self.tfm = TemporalFusionModule(self.mid_d, self.mid_d)
        self.decoder = Decoder(self.mid_d)

    def forward(self, x1, x2):
        # LWGANet backbone returns 4 feature maps
        x1_feats = self.backbone(x1)
        x2_feats = self.backbone(x2)

        # SWA aggregation
        x1_0, x1_1, x1_2, x1_3 = self.swa(*x1_feats)
        x2_0, x2_1, x2_2, x2_3 = self.swa(*x2_feats)

        # Temporal fusion
        c0, c1, c2, c3 = self.tfm(x1_0, x1_1, x1_2, x1_3,
                                    x2_0, x2_1, x2_2, x2_3)

        # Decoder
        mask_p0, mask_p1, mask_p2, mask_p3 = self.decoder(c0, c1, c2, c3)

        mask_p0 = F.interpolate(mask_p0, scale_factor=4, mode='bilinear')
        mask_p0 = torch.sigmoid(mask_p0)
        mask_p1 = F.interpolate(mask_p1, scale_factor=8, mode='bilinear')
        mask_p1 = torch.sigmoid(mask_p1)
        mask_p2 = F.interpolate(mask_p2, scale_factor=16, mode='bilinear')
        mask_p2 = torch.sigmoid(mask_p2)
        mask_p3 = F.interpolate(mask_p3, scale_factor=32, mode='bilinear')
        mask_p3 = torch.sigmoid(mask_p3)

        return mask_p0, mask_p1, mask_p2, mask_p3
