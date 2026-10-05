from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image, ImageOps

from quickdraw_ai.models.cnn import SketchCNN


def prepare_image(image: Image.Image) -> torch.Tensor:
    """Crop, center, and scale a drawing to the model's 1×28×28 input."""
    image = ImageOps.grayscale(image)
    array = np.asarray(image, dtype=np.float32) / 255.0
    # White-background drawings are common in image files; flip to the dataset's
    # black-background, white-stroke convention when the border is mostly bright.
    border = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]))
    if float(np.median(border)) > 0.5:
        array = 1.0 - array

    # Scale based on the drawing bounds rather than the full camera/photo/canvas.
    # This makes a small mouse drawing occupy a similar fraction of the input as
    # the centered Quick, Draw! training bitmaps.
    foreground = np.argwhere(array > 0.12)
    if foreground.size == 0:
        return torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    top, left = foreground.min(axis=0)
    bottom, right = foreground.max(axis=0) + 1
    cropped = Image.fromarray(np.uint8(np.clip(array[top:bottom, left:right] * 255, 0, 255)))
    cropped.thumbnail((24, 24), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(cropped, ((28 - cropped.width) // 2, (28 - cropped.height) // 2))
    normalized = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(normalized[None, None, ...])


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str | None = None):
        checkpoint = Path(checkpoint)
        self.is_onnx = checkpoint.suffix.lower() == ".onnx"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.is_onnx:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("Install onnxruntime to use ONNX checkpoints") from exc
            metadata = checkpoint.with_suffix(".json")
            if not metadata.is_file():
                raise FileNotFoundError(f"ONNX class metadata is missing: {metadata}")
            self.classes = json.loads(metadata.read_text(encoding="utf-8"))["classes"]
            providers = ["CPUExecutionProvider"]
            if self.device.type == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            self.session = ort.InferenceSession(str(checkpoint), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            return

        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.classes: list[str] = payload["classes"]
        self.model = SketchCNN(len(self.classes)).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def predict(self, image: Image.Image, top_k: int = 3) -> list[dict[str, float | str]]:
        tensor = prepare_image(image)
        if self.is_onnx:
            logits = torch.from_numpy(self.session.run(None, {self.input_name: tensor.numpy()})[0])
        else:
            with torch.inference_mode():
                logits = self.model(tensor.to(self.device)).cpu()
        scores, indices = torch.softmax(logits, dim=1).topk(min(top_k, len(self.classes)))
        return [{"class": self.classes[int(i)], "confidence": float(score)}
                for score, i in zip(scores[0], indices[0])]
