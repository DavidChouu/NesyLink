from __future__ import annotations

import torch
from torch import nn

from .components import CLASS_TO_ID, COMPONENT_CLASSES, ComponentBox, component_boxes_from_class_grid


DYNAMIC_CLASSES = ("player", "monster")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyTileCNN(nn.Module):
    """Small CNN that predicts one component class for each 16x16 tile.

    Input shape:
        (B, 3, 128, 160)

    Output shape:
        (B, num_classes, 8, 10)

    Non-floor tiles can be decoded into component boxes.
    """

    def __init__(self, num_classes: int = len(COMPONENT_CLASSES)):
        super().__init__()
        self.num_classes = int(num_classes)
        self.features = nn.Sequential(
            ConvBlock(3, 16, stride=2),   # 64 x 80
            ConvBlock(16, 32, stride=2),  # 32 x 40
            ConvBlock(32, 48, stride=2),  # 16 x 20
            ConvBlock(48, 64, stride=2),  # 8 x 10
            ConvBlock(64, 64, stride=1),
        )
        self.head = nn.Conv2d(64, self.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[-2:] != (128, 160):
            raise ValueError(f"TinyTileCNN expects input shape (B, 3, 128, 160), got {tuple(x.shape)}")
        return self.head(self.features(x))


class TinyHybridCNN(nn.Module):
    """Small hybrid detector for grid-aligned and pixel-moving components.

    Static objects such as walls/chests/traps/exits are naturally tile-aligned, so
    `tile_logits` predicts one class per 16x16 tile.

    Player and monsters can move at pixel resolution, so the dynamic head predicts
    heatmaps on a stride-4 feature map plus class-specific box regressions.

    Input:
        (B, 3, 128, 160)

    Output:
        {
            "tile_logits": (B, num_tile_classes, 8, 10),
            "dynamic_heatmap_logits": (B, 2, 32, 40),
            "dynamic_box": (B, 8, 32, 40),
        }
    """

    dynamic_stride = 4

    def __init__(
        self,
        num_tile_classes: int = len(COMPONENT_CLASSES),
        num_dynamic_classes: int = len(DYNAMIC_CLASSES),
    ):
        super().__init__()
        self.num_tile_classes = int(num_tile_classes)
        self.num_dynamic_classes = int(num_dynamic_classes)

        self.stage1 = ConvBlock(3, 16, stride=2)    # 64 x 80
        self.stage2 = ConvBlock(16, 32, stride=2)   # 32 x 40
        self.stage3 = ConvBlock(32, 48, stride=2)   # 16 x 20
        self.stage4 = ConvBlock(48, 64, stride=2)   # 8 x 10
        self.tile_refine = ConvBlock(64, 64, stride=1)

        self.tile_head = nn.Conv2d(64, self.num_tile_classes, kernel_size=1)
        self.dynamic_refine = nn.Sequential(
            ConvBlock(32, 32, stride=1),
            ConvBlock(32, 32, stride=1),
        )
        self.dynamic_heatmap_head = nn.Conv2d(32, self.num_dynamic_classes, kernel_size=1)
        self.dynamic_box_head = nn.Conv2d(32, self.num_dynamic_classes * 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[-2:] != (128, 160):
            raise ValueError(f"TinyHybridCNN expects input shape (B, 3, 128, 160), got {tuple(x.shape)}")
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        tile_features = self.tile_refine(x4)
        dynamic_features = self.dynamic_refine(x2)
        return {
            "tile_logits": self.tile_head(tile_features),
            "dynamic_heatmap_logits": self.dynamic_heatmap_head(dynamic_features),
            "dynamic_box": self.dynamic_box_head(dynamic_features),
        }


@torch.no_grad()
def component_boxes_from_logits(
    logits: torch.Tensor,
    *,
    min_score: float = 0.50,
    suppressed_classes: tuple[str, ...] = (),
) -> list[list[ComponentBox]]:
    """Decode model logits into connected component boxes for each batch item."""
    logits = suppress_tile_classes(logits, suppressed_classes)
    probs = torch.softmax(logits, dim=1)
    scores, class_ids = probs.max(dim=1)
    class_ids = class_ids.cpu().numpy()
    scores = scores.cpu().numpy()

    batch_boxes: list[list[ComponentBox]] = []
    for class_grid, score_grid in zip(class_ids, scores, strict=True):
        masked_grid = class_grid.copy()
        masked_grid[score_grid < min_score] = 0
        batch_boxes.append(component_boxes_from_class_grid(masked_grid, score_grid=score_grid))
    return batch_boxes


def suppress_tile_classes(logits: torch.Tensor, class_names: tuple[str, ...]) -> torch.Tensor:
    if not class_names:
        return logits
    adjusted = logits.clone()
    min_value = torch.finfo(adjusted.dtype).min
    for class_name in class_names:
        class_id = CLASS_TO_ID.get(class_name)
        if class_id is not None and class_id < adjusted.shape[1]:
            adjusted[:, class_id, :, :] = min_value
    return adjusted


@torch.no_grad()
def component_boxes_from_hybrid_output(
    output: dict[str, torch.Tensor],
    *,
    tile_min_score: float = 0.50,
    dynamic_min_score: float = 0.50,
    dynamic_top_k: int = 8,
) -> list[list[ComponentBox]]:
    """Decode TinyHybridCNN output into boxes for every batch item."""
    tile_boxes = component_boxes_from_logits(
        output["tile_logits"],
        min_score=tile_min_score,
        suppressed_classes=DYNAMIC_CLASSES,
    )
    dynamic_boxes = dynamic_boxes_from_output(
        output["dynamic_heatmap_logits"],
        output["dynamic_box"],
        min_score=dynamic_min_score,
        top_k=dynamic_top_k,
    )
    dynamic_boxes = [dedupe_dynamic_boxes(boxes) for boxes in dynamic_boxes]
    return [static + dynamic for static, dynamic in zip(tile_boxes, dynamic_boxes, strict=True)]


def dedupe_dynamic_boxes(boxes: list[ComponentBox]) -> list[ComponentBox]:
    players = [box for box in boxes if box.kind == "player"]
    monsters = [box for box in boxes if box.kind == "monster"]
    result: list[ComponentBox] = []
    if players:
        result.append(max(players, key=lambda box: box.score))
    result.extend(non_max_suppression(monsters, iou_threshold=0.30))
    return result


def non_max_suppression(boxes: list[ComponentBox], *, iou_threshold: float) -> list[ComponentBox]:
    kept: list[ComponentBox] = []
    for box in sorted(boxes, key=lambda item: item.score, reverse=True):
        if all(box_iou(box.bbox_px, kept_box.bbox_px) < iou_threshold for kept_box in kept):
            kept.append(box)
    return kept


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection == 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


@torch.no_grad()
def dynamic_boxes_from_output(
    heatmap_logits: torch.Tensor,
    box_regression: torch.Tensor,
    *,
    min_score: float = 0.50,
    top_k: int = 8,
) -> list[list[ComponentBox]]:
    """Decode player/monster heatmaps and box regressions into pixel boxes.

    `box_regression` uses 4 channels per dynamic class:
    dx, dy, width, height. The current decoder is deliberately simple:

    - dx/dy are sigmoid offsets inside the stride-4 heatmap cell.
    - width/height are sigmoid-scaled into roughly 8..32 pixels.
    """
    if heatmap_logits.ndim != 4:
        raise ValueError("dynamic heatmap must have shape (B, C, H, W)")
    if box_regression.ndim != 4:
        raise ValueError("dynamic box regression must have shape (B, C*4, H, W)")

    heatmaps = torch.sigmoid(heatmap_logits).cpu()
    boxes_raw = box_regression.cpu()
    batch_size, num_classes, height, width = heatmaps.shape
    stride = TinyHybridCNN.dynamic_stride
    decoded: list[list[ComponentBox]] = []

    for batch_index in range(batch_size):
        batch_boxes: list[ComponentBox] = []
        for class_index, kind in enumerate(DYNAMIC_CLASSES[:num_classes]):
            scores = heatmaps[batch_index, class_index]
            flat_scores = scores.flatten()
            count = min(top_k, flat_scores.numel())
            top_scores, top_indices = torch.topk(flat_scores, k=count)
            for score_tensor, flat_index_tensor in zip(top_scores, top_indices, strict=True):
                score = float(score_tensor)
                if score < min_score:
                    continue
                flat_index = int(flat_index_tensor)
                y = flat_index // width
                x = flat_index % width

                reg_start = class_index * 4
                reg = boxes_raw[batch_index, reg_start : reg_start + 4, y, x]
                offset_x = float(torch.sigmoid(reg[0]) * stride)
                offset_y = float(torch.sigmoid(reg[1]) * stride)
                box_w = float(8.0 + torch.sigmoid(reg[2]) * 24.0)
                box_h = float(8.0 + torch.sigmoid(reg[3]) * 24.0)

                center_x = x * stride + offset_x
                center_y = y * stride + offset_y
                left = max(0, int(round(center_x - box_w * 0.5)))
                top = max(0, int(round(center_y - box_h * 0.5)))
                right = min(160, int(round(center_x + box_w * 0.5)))
                bottom = min(128, int(round(center_y + box_h * 0.5)))
                tile = (min(9, max(0, int(center_x // 16))), min(7, max(0, int(center_y // 16))))
                batch_boxes.append(ComponentBox(kind=kind, tiles=(tile,), bbox_px=(left, top, right, bottom), score=score))
        decoded.append(batch_boxes)
    return decoded
