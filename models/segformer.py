"""Six-channel early-fusion SegFormer-B0 baseline, initialized from scratch.

MiT uses overlapping patches, spatial-reduction attention and depthwise Mix-FFN.
Architecture reference: https://arxiv.org/abs/2105.15203 . This standalone
implementation does not load old EfficientNet-fallback checkpoints.
"""
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import DropPath


class MixBlock(nn.Module):
    def __init__(self, channels, heads, reduction, drop_path):
        super().__init__()
        self.channels, self.heads, self.reduction = channels, heads, reduction
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.q = nn.Linear(channels, channels)
        self.kv = nn.Linear(channels, channels*2)
        self.proj = nn.Linear(channels, channels)
        if reduction > 1:
            self.sr = nn.Conv2d(channels, channels, reduction, stride=reduction)
            self.sr_norm = nn.LayerNorm(channels, eps=1e-6)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Linear(channels, channels*4)
        self.depthwise = nn.Conv2d(channels*4, channels*4, 3, padding=1, groups=channels*4)
        self.contract = nn.Linear(channels*4, channels)
        self.drop_path = DropPath(drop_path)

    def forward(self, tokens, height, width):
        batch, length, channels = tokens.shape
        normalized = self.norm1(tokens)
        q = self.q(normalized).reshape(batch,length,self.heads,channels//self.heads).transpose(1,2)
        reduced = normalized
        if self.reduction > 1:
            reduced = self.sr(normalized.transpose(1,2).reshape(batch,channels,height,width))
            reduced = self.sr_norm(reduced.flatten(2).transpose(1,2))
        kv = self.kv(reduced).reshape(batch,-1,2,self.heads,channels//self.heads).permute(2,0,3,1,4)
        weights = (q @ kv[0].transpose(-2,-1)) * (channels//self.heads)**-.5
        attended = (weights.softmax(dim=-1) @ kv[1]).transpose(1,2).reshape(batch,length,channels)
        tokens = tokens + self.drop_path(self.proj(attended))
        mixed = self.expand(self.norm2(tokens)).transpose(1,2).reshape(batch,channels*4,height,width)
        mixed = F.gelu(self.depthwise(mixed)).flatten(2).transpose(1,2)
        return tokens + self.drop_path(self.contract(mixed))


class MiTStage(nn.Module):
    def __init__(self, input_channels, channels, heads, reduction, first, rates):
        super().__init__()
        self.patch = nn.Conv2d(input_channels, channels, 7 if first else 3,
                               stride=4 if first else 2, padding=3 if first else 1)
        self.patch_norm = nn.LayerNorm(channels, eps=1e-6)
        self.blocks = nn.ModuleList([MixBlock(channels,heads,reduction,rate) for rate in rates])
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, image):
        image = self.patch(image)
        batch, channels, height, width = image.shape
        tokens = self.patch_norm(image.flatten(2).transpose(1,2))
        for block in self.blocks: tokens = block(tokens,height,width)
        return self.norm(tokens).transpose(1,2).reshape(batch,channels,height,width)


class SegFormerCD(nn.Module):
    def __init__(self):
        super().__init__()
        channels, heads, reductions = [32,64,160,256], [1,2,5,8], [8,4,2,1]
        rates = torch.linspace(0,.1,8).tolist()
        self.encoder = nn.ModuleList([
            MiTStage(6 if i==0 else channels[i-1],ch,heads[i],reductions[i],i==0,rates[2*i:2*i+2])
            for i,ch in enumerate(channels)])
        self.project = nn.ModuleList([nn.Conv2d(ch,256,1) for ch in channels])
        self.fuse = nn.Sequential(nn.Conv2d(1024,256,1,bias=False),nn.BatchNorm2d(256),nn.ReLU())
        self.dropout = nn.Dropout2d(.1)
        self.classifier = nn.Conv2d(256,2,1)
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module,nn.Linear):
            nn.init.trunc_normal_(module.weight,std=.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Conv2d):
            nn.init.kaiming_normal_(module.weight,mode='fan_out',nonlinearity='relu')
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module,(nn.LayerNorm,nn.BatchNorm2d)):
            nn.init.ones_(module.weight); nn.init.zeros_(module.bias)

    def forward(self, pre, post):
        image = torch.cat((pre,post),dim=1)
        features = []
        for stage in self.encoder:
            image = stage(image)
            features.append(image)
        resolution = features[0].shape[-2:]
        projected = [F.interpolate(layer(feature),size=resolution,mode='bilinear',align_corners=False)
                     for layer,feature in zip(self.project,features)]
        logits = self.classifier(self.dropout(self.fuse(torch.cat(projected,dim=1))))
        return {'logits':F.interpolate(logits,size=pre.shape[-2:],mode='bilinear',align_corners=False)}
