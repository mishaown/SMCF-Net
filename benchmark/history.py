"""Durable experiment-history serialization."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


HISTORY_METRICS = ("f1", "iou", "precision", "recall", "accuracy", "kappa", "tp", "fp", "tn", "fn")


def write_history(output_dir: Path, history: list[dict[str, Any]]) -> None:
    """Persist nested JSON plus a flat CSV suitable for plotting and auditing."""
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    fieldnames = ["epoch", "step", "lr", "train_loss", "val_loss"] + [
        f"{split}_{metric}"
        for split in ("train", "val")
        for metric in HISTORY_METRICS
    ]
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            row = {
                key: record[key]
                for key in ("epoch", "step", "lr", "train_loss", "val_loss")
            }
            row.update(
                {
                    f"{split}_{metric}": record[split][metric]
                    for split in ("train", "val")
                    for metric in HISTORY_METRICS
                }
            )
            writer.writerow(row)
