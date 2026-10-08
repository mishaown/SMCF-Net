"""SMCF-Net construction and foreground probability conversion."""
import torch
from .smcf_net import SMCFNet, SMCFNetModelConfig


def build_model(name, config, variant='parallel'):
    if name != 'smcf_net':
        raise ValueError(f'Unsupported model: {name}')
    return SMCFNet(SMCFNetModelConfig(**config), variant)


def flood_probabilities(name, output):
    if name != 'smcf_net':
        raise ValueError(f'Unsupported model: {name}')
    return [torch.softmax(output['logits'], dim=1)[:, 1:2]]
