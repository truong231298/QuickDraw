import argparse

import cv2
import numpy as np
from PIL import Image

from quickdraw_ai.inference.predictor import Predictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a sketch and classify it")
    parser.add_argument("--model", default="models/quickdraw.pt")
    parser.add_argument("--size", type=int, default=560)
    parser.add_argument("--brush", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()
    predictor = Predictor(args.model)
    canvas = np.zeros((args.size, args.size), dtype=np.uint8)
    drawing = False
    previous: tuple[int, int] | None = None
    predictions: list[dict[str, float | str]] = []

    def mouse(event: int, x: int, y: int, flags: int, _: object) -> None:
        nonlocal drawing, previous
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            previous = (x, y)
            cv2.circle(canvas, (x, y), args.brush // 2, 255, -1)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            if previous is not None:
                cv2.line(canvas, previous, (x, y), 255, args.brush, cv2.LINE_AA)
            previous = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            previous = None

    window = "QuickDraw AI Canvas"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, mouse)
    try:
        while True:
            display = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            cv2.putText(display, "Draw | P predict | C clear | Q quit", (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 80), 1, cv2.LINE_AA)
            if predictions:
                for index, result in enumerate(predictions):
                    text = f"{result['class']}: {result['confidence']:.1%}"
                    cv2.putText(display, text, (12, 58 + index * 28), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, display)
            key = cv2.waitKey(16) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                canvas.fill(0)
                predictions = []
            if key == ord("p"):
                if float(canvas.max()) / 255 < args.threshold or np.count_nonzero(canvas) < 20:
                    predictions = []
                    continue
                predictions = predictor.predict(Image.fromarray(canvas), top_k=3)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
