import torch
import torch.nn as nn
import torch.nn.functional as F
from . import ShuffleNetV2


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
    """Sliding-window neighbor feature aggregation across 4 scales.

    Takes c2,c3,c4,c5 from ShuffleNetV2 backbone.
    """
    def __init__(self, in_d, out_d=64):
        super().__init__()
        self.in_d = in_d  # [c2, c3, c4, c5]
        self.mid_d = out_d // 2
        self.out_d = out_d

        # scale 2 (c2 native scale)
        self.conv_scale2_c2 = nn.Sequential(
            nn.Conv2d(in_d[0], self.mid_d, 3, 1, 1), nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale2_c3 = nn.Sequential(
            nn.Conv2d(in_d[1], self.mid_d, 3, 1, 1), nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_aggregation_s2 = FeatureFusionModule(self.mid_d * 2, in_d[0], out_d)

        # scale 3 (c3 native scale)
        self.conv_scale3_c2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_d[0], self.mid_d, 3, 1, 1), nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale3_c3 = nn.Sequential(
            nn.Conv2d(in_d[1], self.mid_d, 3, 1, 1), nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale3_c4 = nn.Sequential(
            nn.Conv2d(in_d[2], self.mid_d, 3, 1, 1), nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_aggregation_s3 = FeatureFusionModule(self.mid_d * 3, in_d[1], out_d)

        # scale 4 (c4 native scale)
        self.conv_scale4_c3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_d[1], self.mid_d, 3, 1, 1), nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale4_c4 = nn.Sequential(
            nn.Conv2d(in_d[2], self.mid_d, 3, 1, 1), nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale4_c5 = nn.Sequential(
            nn.Conv2d(in_d[3], self.mid_d, 3, 1, 1), nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_aggregation_s4 = FeatureFusionModule(self.mid_d * 3, in_d[2], out_d)

        # scale 5 (c5 native scale)
        self.conv_scale5_c4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_d[2], self.mid_d, 3, 1, 1), nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_scale5_c5 = nn.Sequential(
            nn.Conv2d(in_d[3], self.mid_d, 3, 1, 1), nn.BatchNorm2d(self.mid_d), nn.ReLU(inplace=True))
        self.conv_aggregation_s5 = FeatureFusionModule(self.mid_d * 2, in_d[3], out_d)

    def forward(self, c2, c3, c4, c5):
        c2_s2 = self.conv_scale2_c2(c2)
        c3_s2 = F.interpolate(self.conv_scale2_c3(c3), scale_factor=(2, 2), mode='bilinear')
        s2 = self.conv_aggregation_s2(torch.cat([c2_s2, c3_s2], 1), c2)

        c2_s3 = self.conv_scale3_c2(c2)
        c3_s3 = self.conv_scale3_c3(c3)
        c4_s3 = F.interpolate(self.conv_scale3_c4(c4), scale_factor=(2, 2), mode='bilinear')
        s3 = self.conv_aggregation_s3(torch.cat([c2_s3, c3_s3, c4_s3], 1), c3)

        c3_s4 = self.conv_scale4_c3(c3)
        c4_s4 = self.conv_scale4_c4(c4)
        c5_s4 = F.interpolate(self.conv_scale4_c5(c5), scale_factor=(2, 2), mode='bilinear')
        s4 = self.conv_aggregation_s4(torch.cat([c3_s4, c4_s4, c5_s4], 1), c4)

        c4_s5 = self.conv_scale5_c4(c4)
        c5_s5 = self.conv_scale5_c5(c5)
        s5 = self.conv_aggregation_s5(torch.cat([c4_s5, c5_s5], 1), c5)

        return s2, s3, s4, s5


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
        self.tffm_x2 = TemporalFeatureFusionModule(in_d, out_d)
        self.tffm_x3 = TemporalFeatureFusionModule(in_d, out_d)
        self.tffm_x4 = TemporalFeatureFusionModule(in_d, out_d)
        self.tffm_x5 = TemporalFeatureFusionModule(in_d, out_d)

    def forward(self, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):
        return (self.tffm_x2(x1_2, x2_2), self.tffm_x3(x1_3, x2_3),
                self.tffm_x4(x1_4, x2_4), self.tffm_x5(x1_5, x2_5))


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
        self.sam_p5 = SupervisedAttentionModule(mid_d)
        self.sam_p4 = SupervisedAttentionModule(mid_d)
        self.sam_p3 = SupervisedAttentionModule(mid_d)
        self.conv_p4 = nn.Sequential(
            nn.Conv2d(mid_d, mid_d, 3, 1, 1), nn.BatchNorm2d(mid_d), nn.ReLU(inplace=True))
        self.conv_p3 = nn.Sequential(
            nn.Conv2d(mid_d, mid_d, 3, 1, 1), nn.BatchNorm2d(mid_d), nn.ReLU(inplace=True))
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(mid_d, mid_d, 3, 1, 1), nn.BatchNorm2d(mid_d), nn.ReLU(inplace=True))
        self.cls = nn.Conv2d(mid_d, 1, 1)

    def forward(self, d2, d3, d4, d5):
        p5, mask_p5 = self.sam_p5(d5)
        p4 = self.conv_p4(d4 + F.interpolate(p5, scale_factor=2, mode='bilinear'))
        p4, mask_p4 = self.sam_p4(p4)
        p3 = self.conv_p3(d3 + F.interpolate(p4, scale_factor=2, mode='bilinear'))
        p3, mask_p3 = self.sam_p3(p3)
        p2 = self.conv_p2(d2 + F.interpolate(p3, scale_factor=2, mode='bilinear'))
        mask_p2 = self.cls(p2)
        return mask_p2, mask_p3, mask_p4, mask_p5


# ==================== Main Model ====================

class ELWCDNet(nn.Module):
    """ELW_CDNet for CFB-Net framework.

    ELW_CDNet: Lightweight Change Detection Network with
    Edge-aware Learning (GRSL 2023)
    https://github.com/dyl96/ELW_CDNet

    Uses ShuffleNetV2 backbone (original).
    """
    def __init__(self, input_nc=3, output_nc=1, model_size='1.0x', pretrained=True):
        super().__init__()
        self.backbone = ShuffleNetV2.ShuffleNetV2(model_size=model_size, pretrain=pretrained)
        channels = self.backbone.channels  # e.g. [24, 24, 116, 232, 464] for 1.0x
        # SWA uses c2,c3,c4,c5 (skip c0,c1)
        swa_channels = channels[2:]  # [116, 232, 464] + first is actually c1
        # Actually channels = [c0, c1, c2, c3, c4], SWA uses [c1, c2, c3, c4]
        # Wait - ELWCDNet original uses [24, 24, 116, 232, 464] and SWA takes last 4
        self.en_d = 32
        self.mid_d = self.en_d * 2  # 64
        self.swa = NeighborFeatureAggregation(channels[1:], self.mid_d)  # [c1,c2,c3,c4]
        self.tfm = TemporalFusionModule(self.mid_d, self.mid_d)
        self.decoder = Decoder(self.mid_d)

    def forward(self, x1, x2):
        # ShuffleNetV2 returns c0,c1,c2,c3,c4
        _, x1_1, x1_2, x1_3, x1_4 = self.backbone(x1)
        _, x2_1, x2_2, x2_3, x2_4 = self.backbone(x2)

        # SWA aggregation (c1~c4)
        x1_1, x1_2, x1_3, x1_4 = self.swa(x1_1, x1_2, x1_3, x1_4)
        x2_1, x2_2, x2_3, x2_4 = self.swa(x2_1, x2_2, x2_3, x2_4)

        # Temporal fusion
        c1, c2, c3, c4 = self.tfm(x1_1, x1_2, x1_3, x1_4,
                                    x2_1, x2_2, x2_3, x2_4)

        # Decoder
        mask_p1, mask_p2, mask_p3, mask_p4 = self.decoder(c1, c2, c3, c4)

        mask_p1 = F.interpolate(mask_p1, scale_factor=4, mode='bilinear')
        mask_p1 = torch.sigmoid(mask_p1)
        mask_p2 = F.interpolate(mask_p2, scale_factor=8, mode='bilinear')
        mask_p2 = torch.sigmoid(mask_p2)
        mask_p3 = F.interpolate(mask_p3, scale_factor=16, mode='bilinear')
        mask_p3 = torch.sigmoid(mask_p3)
        mask_p4 = F.interpolate(mask_p4, scale_factor=32, mode='bilinear')
        mask_p4 = torch.sigmoid(mask_p4)

        return mask_p1, mask_p2, mask_p3, mask_p4
