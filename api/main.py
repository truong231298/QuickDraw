import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from quickdraw_ai.inference.predictor import Predictor

app = FastAPI(title="QuickDraw AI", description="Sketch classification inference API", version="0.1.0")
_predictor: Predictor | None = None
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        checkpoint = os.getenv("QUICKDRAW_MODEL", "models/quickdraw.pt")
        if not Path(checkpoint).is_file():
            raise HTTPException(status_code=503, detail=f"Model checkpoint not found: {checkpoint}")
        _predictor = Predictor(checkpoint)
    return _predictor


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "loaded" if _predictor is not None else "not_loaded"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3) -> dict:
    if not 1 <= top_k <= 20:
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 20")
    if file.content_type not in {"image/png", "image/jpeg", "image/webp", "image/bmp"}:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPEG, WebP, or BMP image")
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image must be 10 MiB or smaller")
        image = Image.open(BytesIO(content))
        image.load()
        results = get_predictor().predict(image, top_k)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
    return {"prediction": results[0]["class"], "confidence": results[0]["confidence"], "top_k": results}
