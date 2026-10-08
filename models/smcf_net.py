"""Segmentation-only SMCF-Net with explicit, separately trained empirical controls."""
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass
import timm

VARIANTS = ('parallel', 'no_temporal', 'attention_only', 'memory_only', 'sequential',
            'deepest_parallel', 'concat_only', 'difference_only')

@dataclass
class SMCFNetModelConfig:
    in_channels: int = 3
    num_classes: int = 2
    encoder: str = 'efficientnetv2_rw_t'
    pretrained: bool = True
    hidden_dim: int = 64
    lstm_hidden: int = 128
    dropout: float = .1
    attention_reduction: int = 4

class ConvBlock(nn.Module):
    """Standard convolution block with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell for spatial-temporal modeling.

    Preserves spatial structure through convolutional gates rather than
    fully-connected operations, essential for dense prediction tasks.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        # Single convolution for all four gates (input, forget, output, cell)
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding,
            bias=True
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through ConvLSTM cell.

        Args:
            x: Input tensor of shape (B, C, H, W)
            state: Tuple of (hidden_state, cell_state) or None for initialization

        Returns:
            h_new: New hidden state (B, hidden_dim, H, W)
            (h_new, c_new): Tuple of new hidden and cell states
        """
        B, _, H, W = x.shape

        if state is None:
            h = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        # Split into four gates
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        o = torch.sigmoid(o)  # Output gate
        g = torch.tanh(g)     # Cell gate

        # Update cell and hidden states
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, (h_new, c_new)

class CrossTemporalAttentionModule(nn.Module):
    """
    Cross-Temporal Attention Module (CTAM).

    Computes change-aware features through:
    1. Cross-temporal spatial attention (query from t0, key-value from t1)
    2. Channel difference attention for adaptive re-weighting
    3. Learnable gamma parameter for stable training

    Complexity: O(C²/r) where r is the reduction ratio
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.channels = channels
        mid_ch = max(channels // reduction, 16)

        # Cross-temporal attention projections
        self.query = nn.Conv2d(channels, mid_ch, 1)
        self.key = nn.Conv2d(channels, mid_ch, 1)
        self.value = nn.Conv2d(channels, channels, 1)

        # Channel difference attention
        self.diff_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, channels, 1),
            nn.Sigmoid()
        )

        # Learnable fusion parameter (initialized at zero for training stability)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat_t0: torch.Tensor, feat_t1: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-temporal attention.

        Args:
            feat_t0: Pre-flood features (B, C, H, W)
            feat_t1: Post-flood features (B, C, H, W)

        Returns:
            Change-aware features (B, C, H, W)
        """
        B, C, H, W = feat_t0.shape

        # Cross-temporal spatial attention
        Q = self.query(feat_t0).view(B, -1, H * W)      # Query from pre-flood
        K = self.key(feat_t1).view(B, -1, H * W)        # Key from post-flood
        V = self.value(feat_t1).view(B, C, H * W)       # Value from post-flood

        # Compute attention weights
        attn = torch.bmm(Q.transpose(1, 2), K)
        attn = F.softmax(attn / (C ** 0.5), dim=-1)

        # Apply attention to values
        attended = torch.bmm(V, attn.transpose(1, 2))
        attended = attended.view(B, C, H, W)

        # Channel difference attention
        diff = torch.abs(feat_t1 - feat_t0)
        diff_weight = self.diff_attn(diff)
        weighted_diff = diff * diff_weight

        # Fuse with learnable gamma
        change_feat = feat_t0 + self.gamma * attended + weighted_diff

        return change_feat

class SpatialTemporalSequenceModule(nn.Module):
    """
    Spatial-Temporal Sequence Module (STSM).

    Wraps ConvLSTM for sequential bi-temporal processing.
    Processes pre-flood features first, then post-flood features
    using the propagated hidden and cell states.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.convlstm = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.hidden_dim = hidden_dim

    def forward(self, feat_t0: torch.Tensor, feat_t1: torch.Tensor) -> torch.Tensor:
        """
        Sequential temporal processing: t0 → t1.

        Args:
            feat_t0: Pre-flood features (B, C, H, W)
            feat_t1: Post-flood features (B, C, H, W)

        Returns:
            Temporally-modeled features (B, hidden_dim, H, W)
        """
        # Process pre-flood features (initialize states)
        h1, (h1_out, c1_out) = self.convlstm(feat_t0, None)

        # Process post-flood features with propagated states
        h2, (h2_out, c2_out) = self.convlstm(feat_t1, (h1_out, c1_out))

        return h2

class MultiScaleDifferenceAggregationModule(nn.Module):
    """
    Multi-Scale Difference Aggregation Module (MSDAM).

    Dual-path architecture:
    1. Concatenation path: Learns complex feature interactions
    2. Absolute difference path: Captures magnitude-based changes

    Asymmetric channel allocation (2:1 ratio) balances interaction
    capacity with explicit difference preservation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Concatenation path (processes joint features)
        self.concat_conv = nn.Sequential(
            ConvBlock(in_channels * 2, out_channels),
            ConvBlock(out_channels, out_channels)
        )

        # Absolute difference path
        self.diff_conv = nn.Sequential(
            ConvBlock(in_channels, out_channels // 2),
        )

        # Feature fusion (2:1 asymmetric ratio)
        self.fusion = nn.Sequential(
            ConvBlock(out_channels + out_channels // 2, out_channels),
            nn.Conv2d(out_channels, out_channels, 1)
        )

    def forward(self, feat_t0: torch.Tensor, feat_t1: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale difference features.

        Args:
            feat_t0: Pre-flood features (B, C, H, W)
            feat_t1: Post-flood features (B, C, H, W)

        Returns:
            Difference-aggregated features (B, out_channels, H, W)
        """
        # Concatenation path
        concat = torch.cat([feat_t0, feat_t1], dim=1)
        concat_feats = self.concat_conv(concat)

        # Absolute difference path
        abs_diff = torch.abs(feat_t1 - feat_t0)
        diff_feats = self.diff_conv(abs_diff)

        # Fuse with asymmetric channel allocation
        fused = torch.cat([concat_feats, diff_feats], dim=1)

        return self.fusion(fused)

class ProgressiveUpsamplingDecoder(nn.Module):
    """
    Progressive Upsampling Decoder (PUD).

    Lightweight FPN-style decoder with:
    1. Additive skip connections (parameter-efficient)
    2. Progressive coarse-to-fine refinement
    3. Unified channel dimension across all scales
    """

    def __init__(
        self,
        encoder_channels: List[int],
        hidden_dim: int = 64,
        num_classes: int = 2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Lateral connections (project to unified channel dimension)
        self.laterals = nn.ModuleList([
            nn.Conv2d(ch, hidden_dim, 1) for ch in encoder_channels
        ])

        # Decoder blocks for progressive refinement
        self.decoder_blocks = nn.ModuleList([
            ConvBlock(hidden_dim, hidden_dim) for _ in range(len(encoder_channels))
        ])

        # Final classifier
        self.classifier = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, num_classes, 1)
        )

    def forward(
        self,
        features: List[torch.Tensor],
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Progressive decoding from coarse to fine.

        Args:
            features: List of multi-scale features [scale1, scale2, scale3, scale4]
            target_size: Output spatial dimensions (H, W)

        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        x = None

        # Process from coarsest to finest (reverse order)
        for i in range(len(features) - 1, -1, -1):
            lateral = self.laterals[i](features[i])

            if x is not None:
                # Upsample and add (additive skip connection)
                x = F.interpolate(
                    x, size=lateral.shape[-2:],
                    mode='bilinear', align_corners=False
                )
                x = x + lateral
            else:
                x = lateral

            x = self.decoder_blocks[i](x)

        # Upsample to target resolution
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        return self.classifier(x)

class SinglePathAggregation(nn.Module):
    def __init__(self, channels: int, hidden: int, difference: bool):
        super().__init__()
        self.difference = difference
        self.branch = nn.Sequential(ConvBlock(channels if difference else channels * 2, hidden),
                                    ConvBlock(hidden, hidden))
        self.fusion = nn.Sequential(ConvBlock(hidden, hidden), nn.Conv2d(hidden, hidden, 1))

    def forward(self, pre, post):
        x = torch.abs(post - pre) if self.difference else torch.cat((pre, post), dim=1)
        return self.fusion(self.branch(x))

class SMCFNet(nn.Module):
    """Paired encoder, cross-temporal attention, ConvLSTM, difference aggregation and decoder."""
    def __init__(self, config: SMCFNetModelConfig, variant='parallel'):
        super().__init__()
        if variant not in VARIANTS: raise ValueError(f'Unknown variant: {variant}')
        self.config, self.variant = config, variant
        self.encoder = self._build_encoder()
        with torch.no_grad():
            self.encoder_channels = [f.shape[1] for f in self.encoder(torch.zeros(1,config.in_channels,64,64))]
        channels = self.encoder_channels
        self.active_scales = (3,) if variant=='deepest_parallel' else (0,1,2,3)
        self.ctam_modules = nn.ModuleList([
            CrossTemporalAttentionModule(ch,config.attention_reduction)
            if variant not in ('no_temporal','memory_only') and i in self.active_scales else None
            for i,ch in enumerate(channels)])
        self.stsm_modules = nn.ModuleList([
            SpatialTemporalSequenceModule(ch,config.lstm_hidden)
            if variant not in ('no_temporal','attention_only') and i in self.active_scales else None
            for i,ch in enumerate(channels)])
        self.stsm_projections = nn.ModuleList([
            nn.Conv2d(config.lstm_hidden,ch,1) if module is not None else None
            for ch,module in zip(channels,self.stsm_modules)])
        self.msdam_modules = nn.ModuleList([
            SinglePathAggregation(ch,config.hidden_dim,variant=='difference_only')
            if variant in ('concat_only','difference_only') else MultiScaleDifferenceAggregationModule(ch,config.hidden_dim)
            for ch in channels])
        self.decoder = ProgressiveUpsamplingDecoder([config.hidden_dim]*len(channels),config.hidden_dim,config.num_classes)
        self.dropout = nn.Dropout2d(config.dropout)

    def _build_encoder(self):
        return timm.create_model(self.config.encoder,pretrained=self.config.pretrained,features_only=True,
                                 out_indices=[1,2,3,4],in_chans=self.config.in_channels)

    def encode(self,x):
        return self.encoder(x)

    def forward(self, pre, post, return_features=False):
            left, right = self.encode(pre), self.encode(post)
            combined = []
            for i, (f0, f1) in enumerate(zip(left, right)):
                a0, a1 = f0, f1
                if self.ctam_modules[i] is not None:
                    a0 = self.ctam_modules[i](f0, f1)
                    a1 = self.ctam_modules[i](f1, f0)
                if self.stsm_modules[i] is not None:
                    t0, t1 = (a0, a1) if self.variant == 'sequential' else (f0, f1)
                    memory = self.stsm_projections[i](self.stsm_modules[i](t0, t1))
                    a0, a1 = a0 + memory, a1 + memory
                combined.append(self.dropout(self.msdam_modules[i](a0, a1)))
            output = {'logits': self.decoder(combined, pre.shape[-2:])}
            if return_features:
                output['features'] = combined
            return output

    def predict(self, pre, post, threshold=.5):
            self.eval()
            with torch.no_grad():
                output = self(pre, post)
                probabilities = torch.softmax(output['logits'], dim=1)
                return {'probabilities': probabilities,
                        'predictions': (probabilities[:, 1] >= threshold).long()}

    def get_module_parameters(self):
            return {name: sum(p.numel() for p in module.parameters())
                    for name, module in self.named_children()}
