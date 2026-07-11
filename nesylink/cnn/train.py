from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CNN_DIR = Path(__file__).resolve().parent

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit("PyTorch is required for CNN training. Install torch before running python -m nesylink.cnn.train.") from exc

from nesylink.cnn.components import (
    CLASS_TO_ID,
    COMPONENT_CLASSES,
    DYNAMIC_CLASSES,
    dynamic_heatmap_targets,
    dynamic_targets_from_room_json,
    static_labels_from_room_json,
)
from nesylink.cnn.model import TinyHybridCNN, component_boxes_from_hybrid_output


@dataclass(frozen=True)
class Sample:
    image_path: Path
    json_path: Path


class NesyLinkCnnDataset(Dataset):
    def __init__(self, samples: list[Sample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        payload = json.loads(sample.json_path.read_text(encoding="utf-8"))

        image = Image.open(sample.image_path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        if array.shape != (128, 160, 3):
            raise ValueError(f"{sample.image_path} must be 160x128 RGB, got {array.shape}")

        tile_target = static_labels_from_room_json(payload)
        dynamic_targets = dynamic_targets_from_room_json(payload)
        heatmap, box, mask = dynamic_heatmap_targets(dynamic_targets)

        return {
            "image": torch.from_numpy(array).permute(2, 0, 1).contiguous(),
            "tile_target": torch.from_numpy(tile_target).long(),
            "dynamic_heatmap": torch.from_numpy(heatmap).float(),
            "dynamic_box": torch.from_numpy(box).float(),
            "dynamic_mask": torch.from_numpy(mask).float(),
            "image_path": str(sample.image_path),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyHybridCNN on generated NesyLink scene PNG/JSON pairs.")
    parser.add_argument("--data-dir", type=Path, default=CNN_DIR / "generated" / "train")
    parser.add_argument(
        "--extra-data-dir",
        type=Path,
        action="append",
        default=[],
        help="Additional PNG/JSON directories to include in the training pool.",
    )
    parser.add_argument(
        "--extra-repeat",
        type=int,
        default=1,
        help="Repeat samples from --extra-data-dir this many times to upweight targeted data.",
    )
    parser.add_argument("--pattern", default="*.json", help="JSON glob inside --data-dir, e.g. batch30_seed*.json")
    parser.add_argument("--out", type=Path, default=CNN_DIR / "checkpoints" / "tiny_hybrid_cnn_exit_split.pt")
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Optional checkpoint/weights to fine-tune from.")
    parser.add_argument("--preview-out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--floor-weight", type=float, default=0.20)
    parser.add_argument("--heatmap-pos-weight", type=float, default=80.0)
    parser.add_argument("--heatmap-loss-weight", type=float, default=1.0)
    parser.add_argument("--box-loss-weight", type=float, default=0.10)
    parser.add_argument("--threshold", type=float, default=0.50, help="Detection threshold for optional preview image.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)

    if args.extra_repeat < 1:
        raise SystemExit("--extra-repeat must be >= 1")
    data_dirs = [args.data_dir, *args.extra_data_dir]
    samples = discover_samples(
        args.data_dir,
        args.pattern,
        extra_data_dirs=args.extra_data_dir,
        extra_repeat=args.extra_repeat,
        limit=args.limit,
    )
    if len(samples) < 2:
        joined_dirs = ", ".join(str(path) for path in data_dirs)
        raise SystemExit(f"Need at least 2 PNG/JSON pairs, found {len(samples)} in {joined_dirs}")

    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)
    print(f"device={device}")
    print(f"samples total={len(samples)} train={len(train_samples)} val={len(val_samples)}")
    print_dataset_summary(samples)

    train_loader = DataLoader(
        NesyLinkCnnDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        NesyLinkCnnDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = TinyHybridCNN().to(device)
    if args.init_checkpoint is not None:
        load_initial_checkpoint(model, args.init_checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tile_weights = build_tile_weights(args.floor_weight, device)
    heatmap_pos_weight = torch.full((len(DYNAMIC_CLASSES), 1, 1), args.heatmap_pos_weight, device=device)

    best_val_loss = float("inf")
    best_epoch = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            tile_weights,
            heatmap_pos_weight,
            optimizer=optimizer,
            heatmap_loss_weight=args.heatmap_loss_weight,
            box_loss_weight=args.box_loss_weight,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            tile_weights,
            heatmap_pos_weight,
            optimizer=None,
            heatmap_loss_weight=args.heatmap_loss_weight,
            box_loss_weight=args.box_loss_weight,
        )

        print(format_epoch(epoch, train_metrics, val_metrics))
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            save_checkpoint(args.out, model, optimizer, epoch, args, train_metrics, val_metrics)
            save_weights_checkpoint(args.out.with_suffix(".weights.pt"), model)

    print(f"saved best checkpoint {args.out} epoch={best_epoch} val_loss={best_val_loss:.4f}")
    if args.preview_out is not None:
        save_preview(model, val_samples[0], device, args.preview_out, args.threshold)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def load_initial_checkpoint(model: nn.Module, path: Path, device: torch.device) -> None:
    if not path.exists():
        raise SystemExit(f"--init-checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    print(f"loaded init checkpoint {path}")


def discover_samples(
    data_dir: Path,
    pattern: str,
    *,
    extra_data_dirs: list[Path],
    extra_repeat: int,
    limit: int | None,
) -> list[Sample]:
    samples: list[Sample] = []
    samples.extend(discover_samples_in_dir(data_dir, pattern))
    extra_samples: list[Sample] = []
    for extra_dir in extra_data_dirs:
        extra_samples.extend(discover_samples_in_dir(extra_dir, pattern))
    for _ in range(extra_repeat):
        samples.extend(extra_samples)
    if limit is not None:
        samples = samples[: max(0, limit)]
    return samples


def discover_samples_in_dir(data_dir: Path, pattern: str) -> list[Sample]:
    samples: list[Sample] = []
    for json_path in sorted(data_dir.glob(pattern)):
        image_path = json_path.with_suffix(".png")
        if image_path.exists():
            samples.append(Sample(image_path=image_path, json_path=json_path))
    return samples


def split_samples(samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    val_count = int(round(len(shuffled) * val_ratio))
    val_count = min(max(1, val_count), len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def print_dataset_summary(samples: list[Sample]) -> None:
    tile_counts = np.zeros(len(COMPONENT_CLASSES), dtype=np.int64)
    dynamic_counts = {kind: 0 for kind in DYNAMIC_CLASSES}
    for sample in samples:
        payload = json.loads(sample.json_path.read_text(encoding="utf-8"))
        labels = static_labels_from_room_json(payload)
        unique, counts = np.unique(labels, return_counts=True)
        for class_id, count in zip(unique, counts, strict=True):
            tile_counts[int(class_id)] += int(count)
        for target in dynamic_targets_from_room_json(payload):
            if target.kind in dynamic_counts:
                dynamic_counts[target.kind] += 1

    nonzero_tiles = [
        f"{name}={int(tile_counts[class_id])}"
        for name, class_id in CLASS_TO_ID.items()
        if tile_counts[class_id] > 0
    ]
    dynamic_summary = [f"{name}={count}" for name, count in dynamic_counts.items()]
    print("tile targets " + " ".join(nonzero_tiles))
    print("dynamic targets " + " ".join(dynamic_summary))


def build_tile_weights(floor_weight: float, device: torch.device) -> torch.Tensor:
    weights = torch.ones(len(COMPONENT_CLASSES), dtype=torch.float32, device=device)
    weights[CLASS_TO_ID["floor"]] = float(floor_weight)
    return weights


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tile_weights: torch.Tensor,
    heatmap_pos_weight: torch.Tensor,
    *,
    optimizer: torch.optim.Optimizer | None,
    heatmap_loss_weight: float,
    box_loss_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = empty_totals()

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(training):
            output = model(batch["image"])
            losses = compute_losses(
                output,
                batch,
                tile_weights,
                heatmap_pos_weight,
                heatmap_loss_weight=heatmap_loss_weight,
                box_loss_weight=box_loss_weight,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                optimizer.step()

        update_totals(totals, losses, output, batch)

    return finalize_totals(totals)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def compute_losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    tile_weights: torch.Tensor,
    heatmap_pos_weight: torch.Tensor,
    *,
    heatmap_loss_weight: float,
    box_loss_weight: float,
) -> dict[str, torch.Tensor]:
    tile_loss = F.cross_entropy(output["tile_logits"], batch["tile_target"], weight=tile_weights)
    heatmap_loss = F.binary_cross_entropy_with_logits(
        output["dynamic_heatmap_logits"],
        batch["dynamic_heatmap"],
        pos_weight=heatmap_pos_weight,
    )

    decoded_box = decode_box_regression(output["dynamic_box"], stride=TinyHybridCNN.dynamic_stride)
    expanded_mask = batch["dynamic_mask"].repeat_interleave(4, dim=1)
    raw_box_loss = F.smooth_l1_loss(decoded_box, batch["dynamic_box"], reduction="none")
    box_loss = (raw_box_loss * expanded_mask).sum() / expanded_mask.sum().clamp_min(1.0)

    loss = tile_loss + heatmap_loss * heatmap_loss_weight + box_loss * box_loss_weight
    return {
        "loss": loss,
        "tile_loss": tile_loss,
        "heatmap_loss": heatmap_loss,
        "box_loss": box_loss,
    }


def decode_box_regression(raw: torch.Tensor, *, stride: int) -> torch.Tensor:
    decoded = torch.empty_like(raw)
    num_dynamic_classes = raw.shape[1] // 4
    for class_index in range(num_dynamic_classes):
        base = class_index * 4
        decoded[:, base + 0] = torch.sigmoid(raw[:, base + 0]) * stride
        decoded[:, base + 1] = torch.sigmoid(raw[:, base + 1]) * stride
        decoded[:, base + 2] = 8.0 + torch.sigmoid(raw[:, base + 2]) * 24.0
        decoded[:, base + 3] = 8.0 + torch.sigmoid(raw[:, base + 3]) * 24.0
    return decoded


def empty_totals() -> dict[str, float]:
    return {
        "samples": 0.0,
        "loss": 0.0,
        "tile_loss": 0.0,
        "heatmap_loss": 0.0,
        "box_loss": 0.0,
        "tile_correct": 0.0,
        "tile_count": 0.0,
        "object_tile_correct": 0.0,
        "object_tile_count": 0.0,
        "dynamic_pos_score": 0.0,
        "dynamic_pos_count": 0.0,
        "dynamic_neg_score": 0.0,
        "dynamic_neg_count": 0.0,
    }


@torch.no_grad()
def update_totals(
    totals: dict[str, float],
    losses: dict[str, torch.Tensor],
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> None:
    batch_size = float(batch["image"].shape[0])
    totals["samples"] += batch_size
    for name in ("loss", "tile_loss", "heatmap_loss", "box_loss"):
        totals[name] += float(losses[name].detach()) * batch_size

    tile_pred = output["tile_logits"].argmax(dim=1)
    tile_target = batch["tile_target"]
    totals["tile_correct"] += float((tile_pred == tile_target).sum())
    totals["tile_count"] += float(tile_target.numel())

    object_mask = tile_target != CLASS_TO_ID["floor"]
    object_count = float(object_mask.sum())
    if object_count > 0:
        totals["object_tile_correct"] += float(((tile_pred == tile_target) & object_mask).sum())
        totals["object_tile_count"] += object_count

    scores = torch.sigmoid(output["dynamic_heatmap_logits"])
    pos_mask = batch["dynamic_heatmap"] > 0.5
    neg_mask = ~pos_mask
    pos_count = float(pos_mask.sum())
    neg_count = float(neg_mask.sum())
    if pos_count > 0:
        totals["dynamic_pos_score"] += float(scores[pos_mask].sum())
        totals["dynamic_pos_count"] += pos_count
    if neg_count > 0:
        totals["dynamic_neg_score"] += float(scores[neg_mask].sum())
        totals["dynamic_neg_count"] += neg_count


def finalize_totals(totals: dict[str, float]) -> dict[str, float]:
    samples = max(1.0, totals["samples"])
    tile_count = max(1.0, totals["tile_count"])
    object_tile_count = max(1.0, totals["object_tile_count"])
    dynamic_pos_count = max(1.0, totals["dynamic_pos_count"])
    dynamic_neg_count = max(1.0, totals["dynamic_neg_count"])
    return {
        "loss": totals["loss"] / samples,
        "tile_loss": totals["tile_loss"] / samples,
        "heatmap_loss": totals["heatmap_loss"] / samples,
        "box_loss": totals["box_loss"] / samples,
        "tile_acc": totals["tile_correct"] / tile_count,
        "object_tile_acc": totals["object_tile_correct"] / object_tile_count,
        "dynamic_pos_score": totals["dynamic_pos_score"] / dynamic_pos_count,
        "dynamic_neg_score": totals["dynamic_neg_score"] / dynamic_neg_count,
    }


def format_epoch(epoch: int, train: dict[str, float], val: dict[str, float]) -> str:
    return (
        f"epoch {epoch:03d} "
        f"train loss={train['loss']:.4f} tile={train['tile_loss']:.4f} heat={train['heatmap_loss']:.4f} "
        f"box={train['box_loss']:.4f} tile_acc={train['tile_acc']:.3f} obj_acc={train['object_tile_acc']:.3f} "
        f"dyn+={train['dynamic_pos_score']:.3f} dyn-={train['dynamic_neg_score']:.3f} | "
        f"val loss={val['loss']:.4f} tile={val['tile_loss']:.4f} heat={val['heatmap_loss']:.4f} "
        f"box={val['box_loss']:.4f} tile_acc={val['tile_acc']:.3f} obj_acc={val['object_tile_acc']:.3f} "
        f"dyn+={val['dynamic_pos_score']:.3f} dyn-={val['dynamic_neg_score']:.3f}"
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "component_classes": COMPONENT_CLASSES,
            "dynamic_classes": DYNAMIC_CLASSES,
            "args": vars(args),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        path,
    )


def save_weights_checkpoint(path: Path, model: nn.Module) -> None:
    torch.save(model.state_dict(), path)


@torch.no_grad()
def save_preview(model: nn.Module, sample: Sample, device: torch.device, out_path: Path, threshold: float) -> None:
    model.eval()
    image = Image.open(sample.image_path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    output = model(tensor)
    boxes = component_boxes_from_hybrid_output(output, tile_min_score=threshold, dynamic_min_score=threshold)[0]

    from nesylink.cnn.components import draw_component_boxes

    out = draw_component_boxes(image, boxes, labels=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    print(f"saved preview {out_path} boxes={len(boxes)} source={sample.image_path}")


if __name__ == "__main__":
    main()
