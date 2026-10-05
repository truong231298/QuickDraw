import argparse
import json
from pathlib import Path

import torch

from quickdraw_ai.models.cnn import SketchCNN


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained checkpoint to ONNX")
    parser.add_argument("--checkpoint", default="models/quickdraw.pt")
    parser.add_argument("--output", default="models/quickdraw.onnx")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SketchCNN(len(payload["classes"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, 1, 28, 28), args.output,
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17)
    metadata_path = Path(args.output).with_suffix(".json")
    metadata_path.write_text(json.dumps({"classes": payload["classes"], "image_size": 28}, indent=2),
                             encoding="utf-8")
    print(f"Exported {args.output} and {metadata_path}")


if __name__ == "__main__":
    main()
