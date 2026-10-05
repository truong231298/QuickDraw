import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from quickdraw_ai.data.dataset import QuickDrawDataset
from quickdraw_ai.models.cnn import SketchCNN


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a QuickDraw checkpoint on a held-out split")
    parser.add_argument("--data", default="data")
    parser.add_argument("--checkpoint", default="models/quickdraw.pt")
    parser.add_argument("--output", default="reports/evaluation")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-per-class", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = payload["classes"]
    dataset = QuickDrawDataset(args.data, classes, args.max_per_class)
    _, val_idx = train_test_split(range(len(dataset)), test_size=0.1, random_state=args.seed,
                                  stratify=dataset.labels)
    loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SketchCNN(len(classes)).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    # Warm the execution path before timing so one-time dispatch/setup costs do
    # not dominate the steady-state latency estimate.
    warmup_images, _ = next(iter(loader))
    with torch.inference_mode():
        for _ in range(5):
            model(warmup_images.to(device))
        single_image = warmup_images[:1].to(device)
        for _ in range(5):
            model(single_image)
    if device.type == "cuda":
        torch.cuda.synchronize()
    single_elapsed = 0.0
    with torch.inference_mode():
        for _ in range(50):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(single_image)
            if device.type == "cuda":
                torch.cuda.synchronize()
            single_elapsed += time.perf_counter() - start

    y_true: list[int] = []
    y_pred: list[int] = []
    top3_hits = 0
    elapsed = 0.0
    samples = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            pred = logits.argmax(1).cpu().tolist()
            top3 = logits.topk(min(3, len(classes)), dim=1).indices.cpu().numpy()
            target = labels.numpy()
            y_true.extend(target.tolist())
            y_pred.extend(pred)
            top3_hits += int(np.any(top3 == target[:, None], axis=1).sum())
            samples += len(target)

    report = classification_report(y_true, y_pred, labels=list(range(len(classes))),
                                   target_names=classes, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    result = {
        "checkpoint": str(Path(args.checkpoint)),
        "validation_samples": samples,
        "accuracy": accuracy_score(y_true, y_pred),
        "top3_accuracy": top3_hits / samples,
        "inference_latency_ms_per_image": single_elapsed / 50 * 1000,
        "batched_average_ms_per_image": elapsed / samples * 1000,
        "throughput_images_per_second": samples / elapsed if elapsed else None,
        "model_size_mb": Path(args.checkpoint).stat().st_size / (1024 * 1024),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {name: report[name] for name in classes},
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output / "per_class.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["class", "precision", "recall", "f1-score", "support"])
        writer.writeheader()
        for name in classes:
            writer.writerow({"class": name, **report[name]})
    np.savetxt(output / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    print(json.dumps({key: value for key, value in result.items() if key != "per_class"}, indent=2))
    print(f"reports saved to {output}")


if __name__ == "__main__":
    main()
