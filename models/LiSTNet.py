import torch
import torch.nn as nn
import torch.nn.functional as F
from . import ViTAEv2


# ==================== LS3DAM ====================

class LS3DAM(nn.Module):
    """Lightweight Spatio-Channel 3D Attention Module.

    Combines spatial and channel attention efficiently for SAR flood mapping.
    Inspired by SimAM/ECA-Net: uses 3-branch lightweight attention.
    """
    def __init__(self, channel, reduction=4):
        super().__init__()
        mid_ch = max(1, channel // reduction)
        # Channel attention (SE-like, lightweight)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, mid_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, channel, 1),
            nn.Sigmoid())
        # Spatial attention (lightweight)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channel, 1, 3, 1, 1),
            nn.Sigmoid())
        # Cross-dimensional fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(channel, channel, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True))

    def forward(self, x):
        ca = self.channel_att(x)
        sa = self.spatial_att(x)
        out = x * ca * sa
        return self.fuse(out) + x


# ==================== GNM ====================

class GraphNeighborModule(nn.Module):
    """Graph Neighbor Module (GNM).

    Fuses features from adjacent backbone stages using residual
    learning. For each stage i, aggregates features from i-1, i, i+1
    via maxpool/conv/upsample + concat + 3x3 conv + residual.

    As described in LiST-Net Eq.(5)-(6).
    """
    def __init__(self, channels):
        """
        Args:
            channels: list of 4 channel dimensions [c1, c2, c3, c4]
        """
        super().__init__()
        self.channels = channels  # [c1, c2, c3, c4]

        # For stage 1 (uses c1, c2): c2 upsampled to c1 resolution
        self.gnm1_up = nn.Sequential(
            nn.Conv2d(channels[1], channels[0], 3, 1, 1),
            nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
        self.gnm1_fuse = nn.Sequential(
            nn.Conv2d(channels[0] * 2, channels[0], 3, 1, 1),
            nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
        self.gnm1_res = nn.Conv2d(channels[0], channels[0], 1)

        # For stage 2 (uses c1, c2, c3)
        self.gnm2_down = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(channels[0], channels[1], 3, 1, 1),
            nn.BatchNorm2d(channels[1]), nn.ReLU(inplace=True))
        self.gnm2_self = nn.Sequential(
            nn.Conv2d(channels[1], channels[1], 3, 1, 1),
            nn.BatchNorm2d(channels[1]), nn.ReLU(inplace=True))
        self.gnm2_up = nn.Sequential(
            nn.Conv2d(channels[2], channels[1], 3, 1, 1),
            nn.BatchNorm2d(channels[1]), nn.ReLU(inplace=True))
        self.gnm2_fuse = nn.Sequential(
            nn.Conv2d(channels[1] * 3, channels[1], 3, 1, 1),
            nn.BatchNorm2d(channels[1]), nn.ReLU(inplace=True))
        self.gnm2_res = nn.Conv2d(channels[1], channels[1], 1)

        # For stage 3 (uses c2, c3, c4)
        self.gnm3_down = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(channels[1], channels[2], 3, 1, 1),
            nn.BatchNorm2d(channels[2]), nn.ReLU(inplace=True))
        self.gnm3_self = nn.Sequential(
            nn.Conv2d(channels[2], channels[2], 3, 1, 1),
            nn.BatchNorm2d(channels[2]), nn.ReLU(inplace=True))
        self.gnm3_up = nn.Sequential(
            nn.Conv2d(channels[3], channels[2], 3, 1, 1),
            nn.BatchNorm2d(channels[2]), nn.ReLU(inplace=True))
        self.gnm3_fuse = nn.Sequential(
            nn.Conv2d(channels[2] * 3, channels[2], 3, 1, 1),
            nn.BatchNorm2d(channels[2]), nn.ReLU(inplace=True))
        self.gnm3_res = nn.Conv2d(channels[2], channels[2], 1)

        # For stage 4 (uses c3, c4): c3 downsampled to c4 resolution
        self.gnm4_down = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(channels[2], channels[3], 3, 1, 1),
            nn.BatchNorm2d(channels[3]), nn.ReLU(inplace=True))
        self.gnm4_self = nn.Sequential(
            nn.Conv2d(channels[3], channels[3], 3, 1, 1),
            nn.BatchNorm2d(channels[3]), nn.ReLU(inplace=True))
        self.gnm4_fuse = nn.Sequential(
            nn.Conv2d(channels[3] * 2, channels[3], 3, 1, 1),
            nn.BatchNorm2d(channels[3]), nn.ReLU(inplace=True))
        self.gnm4_res = nn.Conv2d(channels[3], channels[3], 1)

    def forward(self, feats):
        c1, c2, c3, c4 = feats

        # Stage 1: c2 upsampled → concat with c1
        c2_up = F.interpolate(self.gnm1_up(c2), size=c1.shape[-2:],
                               mode='bilinear', align_corners=True)
        o1 = self.gnm1_fuse(torch.cat([c1, c2_up], 1))
        o1 = o1 + self.gnm1_res(c1)

        # Stage 2: c1 down + c2 + c3 up
        c2d = self.gnm2_down(c1)
        c2s = self.gnm2_self(c2)
        c3u = F.interpolate(self.gnm2_up(c3), size=c2.shape[-2:],
                             mode='bilinear', align_corners=True)
        o2 = self.gnm2_fuse(torch.cat([c2d, c2s, c3u], 1))
        o2 = o2 + self.gnm2_res(c2)

        # Stage 3: c2 down + c3 + c4 up
        c3d = self.gnm3_down(c2)
        c3s = self.gnm3_self(c3)
        c4u = F.interpolate(self.gnm3_up(c4), size=c3.shape[-2:],
                             mode='bilinear', align_corners=True)
        o3 = self.gnm3_fuse(torch.cat([c3d, c3s, c4u], 1))
        o3 = o3 + self.gnm3_res(c3)

        # Stage 4: c3 down + c4
        c4d = self.gnm4_down(c3)
        c4s = self.gnm4_self(c4)
        o4 = self.gnm4_fuse(torch.cat([c4d, c4s], 1))
        o4 = o4 + self.gnm4_res(c4)

        return o1, o2, o3, o4


# ==================== DIA ====================

class ChannelWidthAggregation(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Aggregate height while preserving channel and width: B,C,1,W.
        avg = torch.mean(x, dim=2, keepdim=True)
        maxv, _ = torch.max(x, dim=2, keepdim=True)
        return self.conv(torch.cat([avg, maxv], dim=1))
class DIA(nn.Module):
    """Dimension-wise Interactive Attention (DIA).

    Replaces standard self-attention with three linear-complexity branches:
    - A1: Channel-Channel attention (Q·K^T softmax) → O(C²)
    - A2: Channel-Width attention → O(C·W)
    - A3: Channel-Height attention → O(C·H)

    Total complexity: O(C² + C·W + C·H) << O((HW)²)

    Hyperparameters: γ1:γ2:γ3 = 2:1:1 (from paper Section IV-B)
    """
    def __init__(self, channel, gamma=(2.0, 1.0, 1.0)):
        super().__init__()
        self.gamma1, self.gamma2, self.gamma3 = gamma
        self.scale = nn.Parameter(torch.ones(1) * (channel ** -0.5))

        # A1: Channel-Channel attention
        self.q_proj = nn.Conv2d(channel, channel, 1)
        self.k_proj = nn.Conv2d(channel, channel, 1)
        self.v_proj = nn.Conv2d(channel, channel, 1)

        # A2: Channel-Width attention
        self.cw_agg = ChannelWidthAggregation(channel)

        # A3: Channel-Height attention
        self.ch_agg = ChannelWidthAggregation(channel)

        self.proj_out = nn.Conv2d(channel, channel, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        # A1: Channel-Channel attention
        Q = self.q_proj(x).view(B, C, -1)          # B, C, HW
        K = self.k_proj(x).view(B, C, -1)          # B, C, HW
        V = self.v_proj(x).view(B, C, -1)          # B, C, HW

        attn1 = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # B, C, C
        attn1 = F.softmax(attn1, dim=-1)
        a1 = torch.bmm(attn1, V).view(B, C, H, W)  # B, C, H, W

        # A2: Channel-Width attention (aggregate over height)
        a2 = self.cw_agg(x)                         # B, C, 1, W
        a2 = F.interpolate(a2, size=(H, W), mode='bilinear', align_corners=True)

        # A3: Channel-Height attention (aggregate over width)
        a3 = self.ch_agg(x.transpose(2, 3)).transpose(2, 3)  # B, C, H, 1
        a3 = F.interpolate(a3, size=(H, W), mode='bilinear', align_corners=True)

        # Weighted fusion
        out = self.gamma1 * (a1 + identity) + \
              self.gamma2 * (a2 * x) + \
              self.gamma3 * (a3 * x)

        return self.proj_out(out) + identity


# ==================== DPEM ====================

class DPEM(nn.Module):
    """Detail-Preserving Enhancement Module (DPEM).

    Two branches:
    - Detail-pooling: Concat(f_t1, f_t2) → 1×1 Conv → LS3DAM
    - Difference-enhancement: |LS3DAM(f_t1) - LS3DAM(f_t2)|
    - Final fusion: element-wise addition

    As described in LiST-Net Eq.(8)-(10).
    """
    def __init__(self, channel):
        super().__init__()
        self.ls3dam1 = LS3DAM(channel)
        self.ls3dam2 = LS3DAM(channel)
        self.ls3dam3 = LS3DAM(channel)

        # Detail-pooling branch
        self.detail_conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.detail_fuse = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True))

        # Difference-enhancement branch
        self.diff_conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.diff_conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True))

        self.out_conv = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True))

    def forward(self, f_t1, f_t2):
        # Detail-pooling branch
        f1 = self.detail_conv1(f_t1)
        f2 = self.detail_conv1(f_t2)
        f_cat = self.detail_fuse(torch.cat([f1, f2], 1))
        f_aggr = self.ls3dam1(f_cat)

        # Difference-enhancement branch
        d1 = self.ls3dam2(self.diff_conv1(f_t1))
        d2 = self.ls3dam3(self.diff_conv2(f_t2))
        f_diff = torch.abs(d1 - d2)

        # Fusion
        out = self.out_conv(torch.cat([f_aggr, f_diff], 1))
        return out


# ==================== ASLM ====================

class ASLM(nn.Module):
    """Attentive Supervised Learning Module (ASLM).

    Pixel mask gate that selectively passes change water information
    while suppressing noise.

    As described in LiST-Net Eq.(11):
    - M = Sigmoid(Conv1×1(d̄))
    - M_r = InvertLUT(M) = 1 - M
    - G = Conv1×1(Concat(M, M_r))
    - d̄_r = Conv3×3(d̄ ⊗ G)
    """
    def __init__(self, channel):
        super().__init__()
        self.conv_mask = nn.Conv2d(channel, 1, 1)
        self.conv_gate = nn.Sequential(
            nn.Conv2d(2, channel, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True))

    def forward(self, x):
        M = torch.sigmoid(self.conv_mask(x))       # Change mask
        M_r = 1.0 - M                                # InvertLUT: reverse mask
        G = self.conv_gate(torch.cat([M, M_r], 1))  # Pixel mask gate
        out = self.conv_out(x * G)                   # Gated feature
        return out, M


# ==================== Main LiST-Net ====================

class LiSTNet(nn.Module):
    """LiST-Net for CFB-Net framework.

    LiST-Net: Enhanced Flood Mapping With Lightweight SAR Transformer
    Network and Dimension-Wise Attention (TGRS 2024)

    Architecture:
    - Siamese encoder with Graph Neighbor Module (GNM)
    - Dimension-wise Interactive Attention (DIA) — O(C²+CW+CH)
    - Detail-Preserving Enhancement Module (DPEM) — two-branch
    - Attentive Supervised Learning Module (ASLM) — pixel mask gate decoder

    Note: Uses ViTAEv2-S backbone exactly as in the original paper.
    ViTAEv2 channels: [64, 128, 256, 512] across 4 stages.

    Paper: 7.34M params, 11.78G FLOPs
    """
    def __init__(self, input_nc=3, output_nc=1, pretrained=True):
        super().__init__()
        # ViTAEv2-S backbone (as in the original LiST-Net paper)
        self.backbone = ViTAEv2.vitaev2_s(pretrained=pretrained, img_size=256)
        backbone_channels = self.backbone.channels  # [64, 128, 256, 512]

        # GNM operates on 4 ViTAEv2 stages
        gnm_channels = backbone_channels  # [64, 128, 256, 512]
        self.gnm = GraphNeighborModule(gnm_channels)

        # DIA at each of the 4 GNM output levels
        self.dia1 = DIA(gnm_channels[0])
        self.dia2 = DIA(gnm_channels[1])
        self.dia3 = DIA(gnm_channels[2])
        self.dia4 = DIA(gnm_channels[3])

        # DPEM at each level
        self.dpem1 = DPEM(gnm_channels[0])
        self.dpem2 = DPEM(gnm_channels[1])
        self.dpem3 = DPEM(gnm_channels[2])
        self.dpem4 = DPEM(gnm_channels[3])

        # ASLM decoder stages
        self.aslm4 = ASLM(gnm_channels[3])
        self.aslm3 = ASLM(gnm_channels[2])
        self.aslm2 = ASLM(gnm_channels[1])
        self.aslm1 = ASLM(gnm_channels[0])

        # Decoder fusion convolutions
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(gnm_channels[2] + gnm_channels[3], gnm_channels[2], 3, 1, 1),
            nn.BatchNorm2d(gnm_channels[2]), nn.ReLU(inplace=True))
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(gnm_channels[1] + gnm_channels[2], gnm_channels[1], 3, 1, 1),
            nn.BatchNorm2d(gnm_channels[1]), nn.ReLU(inplace=True))
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(gnm_channels[0] + gnm_channels[1], gnm_channels[0], 3, 1, 1),
            nn.BatchNorm2d(gnm_channels[0]), nn.ReLU(inplace=True))

        # Final output
        self.cls = nn.Conv2d(gnm_channels[0], output_nc, 1)

        # Auxiliary outputs for multi-scale
        self.aux3 = nn.Conv2d(gnm_channels[1], output_nc, 1)
        self.aux4 = nn.Conv2d(gnm_channels[2], output_nc, 1)
        self.aux5 = nn.Conv2d(gnm_channels[3], output_nc, 1)

    def _encode(self, x):
        """Siamese encoder with GNM + DIA."""
        # ViTAEv2 backbone: 4 stage outputs [64, 128, 256, 512]
        c1, c2, c3, c4 = self.backbone(x)
        # GNM on 4 levels
        o1, o2, o3, o4 = self.gnm([c1, c2, c3, c4])
        # DIA at each level
        o1 = self.dia1(o1)
        o2 = self.dia2(o2)
        o3 = self.dia3(o3)
        o4 = self.dia4(o4)
        return o1, o2, o3, o4

    def forward(self, x1, x2):
        H, W = x1.size(2), x1.size(3)

        # Siamese encoder
        o1_t1, o2_t1, o3_t1, o4_t1 = self._encode(x1)
        o1_t2, o2_t2, o3_t2, o4_t2 = self._encode(x2)

        # DPEM at each level
        de1 = self.dpem1(o1_t1, o1_t2)
        de2 = self.dpem2(o2_t1, o2_t2)
        de3 = self.dpem3(o3_t1, o3_t2)
        de4 = self.dpem4(o4_t1, o4_t2)

        # ASLM decoder (top-down)
        # Level 4 (deepest)
        aslm4_out, mask4 = self.aslm4(de4)
        # Level 3
        de4_up = F.interpolate(aslm4_out, size=de3.shape[-2:],
                                mode='bilinear', align_corners=True)
        de3_cat = self.dec_conv3(torch.cat([de3, de4_up], 1))
        aslm3_out, mask3 = self.aslm3(de3_cat)
        # Level 2
        de3_up = F.interpolate(aslm3_out, size=de2.shape[-2:],
                                mode='bilinear', align_corners=True)
        de2_cat = self.dec_conv2(torch.cat([de2, de3_up], 1))
        aslm2_out, mask2 = self.aslm2(de2_cat)
        # Level 1 (shallowest)
        de2_up = F.interpolate(aslm2_out, size=de1.shape[-2:],
                                mode='bilinear', align_corners=True)
        de1_cat = self.dec_conv1(torch.cat([de1, de2_up], 1))
        aslm1_out, mask1 = self.aslm1(de1_cat)

        # Main output
        out = self.cls(aslm1_out)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        m1 = torch.sigmoid(out)

        # Multi-scale outputs
        m2 = torch.sigmoid(F.interpolate(mask1, size=(H, W),
                                          mode='bilinear', align_corners=True))
        m3 = torch.sigmoid(F.interpolate(mask2, size=(H, W),
                                          mode='bilinear', align_corners=True))
        m4 = torch.sigmoid(F.interpolate(mask4, size=(H, W),
                                          mode='bilinear', align_corners=True))

        return m1, m2, m3, m4
