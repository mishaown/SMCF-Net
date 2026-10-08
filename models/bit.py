"""Local BIT baseline: shared ResNet18 encoder and paired-token transformer."""
import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Union
from dataclasses import dataclass
import timm
@dataclass
class BaselineConfig:
    """Configuration for baseline models."""
    in_channels: int = 3
    num_classes: int = 2
    encoder: str = "resnet18"
    pretrained: bool = True
    hidden_dim: int = 64
    img_size: int = 256

class TransformerEncoder(nn.Module):
    """Simplified transformer encoder for BIT."""
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, depth: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, num_heads, batch_first=True),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, int(dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Linear(int(dim * mlp_ratio), dim)
                )
            ]))

    def forward(self, x):
        for norm1, attn, norm2, mlp in self.layers:
            x = x + attn(norm1(x), norm1(x), norm1(x))[0]
            x = x + mlp(norm2(x))
        return x

class BIT(nn.Module):
    """
    BIT: Bitemporal Image Transformer for Change Detection.

    Simplified implementation using ResNet backbone and transformer.
    Reference: Chen et al., "Remote Sensing Image Change Detection With Transformers", IEEE TGRS 2022
    """
    def __init__(self, config: Union[BaselineConfig, dict]):
        super().__init__()
        if isinstance(config, dict):
            config = BaselineConfig(**{k: v for k, v in config.items() if k in BaselineConfig.__dataclass_fields__})
        self.config = config

        # CNN backbone (shared)
        self.encoder = timm.create_model(
            config.encoder,
            pretrained=config.pretrained,
            features_only=True,
            out_indices=[1, 2, 3, 4],
            in_chans=config.in_channels
        )

        # Get feature dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, config.in_channels, 64, 64)
            feats = self.encoder(dummy)
            self.feat_channels = [f.shape[1] for f in feats]

        # Token projection
        self.token_dim = 256
        self.proj = nn.Conv2d(self.feat_channels[-1], self.token_dim, 1)

        # Transformer encoder
        self.transformer = TransformerEncoder(self.token_dim, num_heads=8, depth=2)

        # Difference computation
        self.diff_conv = nn.Sequential(
            nn.Conv2d(self.token_dim * 2, self.token_dim, 1),
            nn.BatchNorm2d(self.token_dim),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(self.token_dim, config.hidden_dim * 2, 3, padding=1),
            nn.BatchNorm2d(config.hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.hidden_dim * 2, config.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.hidden_dim, config.num_classes, 1)
        )

    def forward(self, img_t0: torch.Tensor, img_t1: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = img_t0.shape

        # Extract features
        feats_t0 = self.encoder(img_t0)[-1]  # Deepest features
        feats_t1 = self.encoder(img_t1)[-1]

        # Project to tokens
        tokens_t0 = self.proj(feats_t0)  # [B, token_dim, h, w]
        tokens_t1 = self.proj(feats_t1)

        _, _, h, w = tokens_t0.shape

        # Reshape to sequence
        tokens_t0 = tokens_t0.flatten(2).transpose(1, 2)  # [B, h*w, token_dim]
        tokens_t1 = tokens_t1.flatten(2).transpose(1, 2)

        # Concatenate and apply transformer
        tokens = torch.cat([tokens_t0, tokens_t1], dim=1)  # [B, 2*h*w, token_dim]
        tokens = self.transformer(tokens)

        # Split back
        tokens_t0 = tokens[:, :h*w, :].transpose(1, 2).view(B, self.token_dim, h, w)
        tokens_t1 = tokens[:, h*w:, :].transpose(1, 2).view(B, self.token_dim, h, w)

        # Compute difference
        diff = self.diff_conv(torch.cat([tokens_t0, tokens_t1], dim=1))

        # Decode
        logits = self.decoder(diff)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

        return {'logits': logits}
