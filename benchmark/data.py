"""Manifest-based loader shared by every dataset and model."""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import tifffile
from torch.utils.data import Dataset


CFB_MEAN_BGR = np.asarray([0.406, 0.456, 0.485], dtype=np.float32)
CFB_STD_BGR = np.asarray([0.225, 0.224, 0.229], dtype=np.float32)


def _three_channels(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.shape[2] == 2:
        return np.concatenate([image, image.mean(axis=2, keepdims=True)], axis=2)
    return image[:, :, :3]


def _read_sar(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        image = tifffile.imread(path)
        if image.ndim == 3 and image.shape[0] <= 4 and image.shape[-1] > 4:
            image = np.moveaxis(image, 0, -1)
        if np.issubdtype(image.dtype, np.floating):
            # Floating SAR rasters store calibrated backscatter in
            # dB with occasional NaN no-data pixels. Use one declared, fixed
            # range for both dates rather than per-image normalization.
            image = np.nan_to_num(image, nan=-50.0, neginf=-50.0, posinf=10.0)
            image = np.clip(image, -50.0, 10.0)
            image = ((image + 50.0) / 60.0 * 255.0).astype(np.uint8)
        return image
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _read_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        mask = tifffile.imread(path)
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        return mask.astype(np.int16)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 127).astype(np.int16)


class PairedTransform:
    def __init__(
        self,
        size: int = 256,
        train: bool = False,
        random_exchange: bool = False,
    ) -> None:
        self.size = size
        self.train = train
        self.random_exchange = random_exchange

    def __call__(
        self, pre: np.ndarray, post: np.ndarray, mask: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pre = cv2.resize(pre, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        post = cv2.resize(post, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        if self.train and random.random() < 0.5:
            margin = int(7.0 / 224.0 * self.size)
            crop_x = random.randint(0, margin)
            crop_y = random.randint(0, margin)
            pre = cv2.resize(pre[crop_y : self.size - crop_y, crop_x : self.size - crop_x], (self.size, self.size))
            post = cv2.resize(post[crop_y : self.size - crop_y, crop_x : self.size - crop_x], (self.size, self.size))
            mask = cv2.resize(
                mask[crop_y : self.size - crop_y, crop_x : self.size - crop_x],
                (self.size, self.size),
                interpolation=cv2.INTER_NEAREST,
            )
        if self.train and random.random() < 0.5:
            pre, post, mask = cv2.flip(pre, 0), cv2.flip(post, 0), cv2.flip(mask, 0)
        if self.train and random.random() < 0.5:
            pre, post, mask = cv2.flip(pre, 1), cv2.flip(post, 1), cv2.flip(mask, 1)
        if self.train and self.random_exchange and random.random() < 0.5:
            pre, post = post, pre

        pre = pre.astype(np.float32) / 255.0
        post = post.astype(np.float32) / 255.0
        pre = (pre - CFB_MEAN_BGR) / CFB_STD_BGR
        post = (post - CFB_MEAN_BGR) / CFB_STD_BGR

        pre = pre[:, :, ::-1].copy()
        post = post[:, :, ::-1].copy()

        pre_tensor = torch.from_numpy(pre.transpose(2, 0, 1)).float()
        post_tensor = torch.from_numpy(post.transpose(2, 0, 1)).float()
        valid_tensor = torch.from_numpy((mask >= 0).astype(np.float32))[None]
        mask_tensor = torch.from_numpy((mask == 1).astype(np.float32))[None]
        return pre_tensor, post_tensor, mask_tensor, valid_tensor


def resolve_data_path(repository_root: Path, relative: str) -> Path:
    """SMCF_NET_DATA_ROOT names the existing data/ directory, not datasets."""
    parts = relative.replace('\\', '/').split('/')
    external = os.environ.get('SMCF_NET_DATA_ROOT') or os.environ.get('FLOODSENSE_DATA_ROOT')
    if external and parts[0] == 'data':
        if any(part in ('', '.', '..') for part in parts):
            raise ValueError(f'Unsafe manifest path: {relative}')
        return Path(external).expanduser().resolve().joinpath(*parts[1:])
    return repository_root / relative


class ManifestDataset(Dataset):
    def __init__(self, repository_root: Path, manifest: Path, transform: PairedTransform):
        self.repository_root = repository_root.resolve()
        self.transform = transform
        with manifest.open(newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        required = {"id", "pre", "post", "mask"}
        if not self.rows or not required.issubset(self.rows[0]):
            raise ValueError(f"invalid or empty manifest: {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        pre = _read_sar(resolve_data_path(self.repository_root, row["pre"]))
        post = _read_sar(resolve_data_path(self.repository_root, row["post"]))
        mask = _read_mask(resolve_data_path(self.repository_root, row["mask"]))
        pre, post = _three_channels(pre), _three_channels(post)
        pre_tensor, post_tensor, mask_tensor, valid_tensor = self.transform(pre, post, mask)
        return {
            "pre": pre_tensor,
            "post": post_tensor,
            "mask": mask_tensor,
            "valid": valid_tensor,
            "id": row["id"],
        }
