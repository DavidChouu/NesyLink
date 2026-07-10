from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesylink.cnn.components import draw_component_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TinyHybridCNN and draw predicted component boxes.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from nesylink.cnn.model import TinyHybridCNN, component_boxes_from_hybrid_output
    except ImportError as exc:
        raise SystemExit("PyTorch is required for CNN inference. Install torch before running python -m nesylink.cnn.infer_boxes.") from exc

    image = Image.open(args.image).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.shape != (128, 160, 3):
        raise ValueError(f"expected image shape (128, 160, 3), got {array.shape}")
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(args.device)

    model = TinyHybridCNN().to(args.device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    except Exception:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        output = model(tensor)
        boxes = component_boxes_from_hybrid_output(output, tile_min_score=args.threshold, dynamic_min_score=args.threshold)[0]

    out = draw_component_boxes(image, boxes, labels=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"saved {args.out} boxes={len(boxes)}")


if __name__ == "__main__":
    main()
