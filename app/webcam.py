import argparse

import cv2
from PIL import Image

from quickdraw_ai.inference.predictor import Predictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a drawing shown to the webcam")
    parser.add_argument("--model", default="models/quickdraw.pt")
    args = parser.parse_args()
    predictor = Predictor(args.model)
    # Open the default camera. The try/finally block below always releases it,
    # including when the user closes the preview window.
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            # Keep a centered square as the sketch area so the classifier sees
            # a predictable region instead of the full webcam frame.
            h, w = frame.shape[:2]
            side = int(min(h, w) * 0.7)
            x, y = (w - side) // 2, (h - side) // 2
            crop = frame[y:y + side, x:x + side]

            # Shared preprocessing handles grayscale conversion, background
            # polarity, cropping to the sketch bounds, and 28×28 resizing.
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            results = predictor.predict(Image.fromarray(rgb), 3)

            # Draw the inference area and ranked class probabilities over the
            # live preview; press q to exit.
            cv2.rectangle(frame, (x, y), (x + side, y + side), (0, 220, 0), 2)
            for row, result in enumerate(results):
                label = f"{result['class']}: {result['confidence']:.2f}"
                cv2.putText(frame, label, (20, 35 + row * 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 220, 0), 2, cv2.LINE_AA)
            cv2.imshow("QuickDraw AI - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
