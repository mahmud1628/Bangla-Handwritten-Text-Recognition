from __future__ import annotations

import base64
import io
import json
import zipfile
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError

from box_detector import crop_boxes, detect_handwritten_boxes

app = FastAPI(
    title="Handwritten Box Detection API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def parse_uploaded_image(file: UploadFile) -> Image.Image:
    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        content = await file.read()
        return Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded image: {exc}") from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/detect-boxes")
async def detect_boxes(file: UploadFile = File(...)):
    pil_image = await parse_uploaded_image(file)

    try:
        boxes, img_rgb = detect_handwritten_boxes(pil_image)
        crops = crop_boxes(img_rgb, boxes, padding=10)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Box detection failed: {exc}") from exc

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("boxes.json", json.dumps(boxes, indent=2))

        for i, crop in enumerate(crops, start=1):
            img_bytes = BytesIO()
            crop.save(img_bytes, format="PNG")
            zf.writestr(f"box_{i}.png", img_bytes.getvalue())

    zip_buffer.seek(0)
    filename = "detected_boxes.zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/detect-boxes-json")
async def detect_boxes_json(file: UploadFile = File(...)) -> dict:
    pil_image = await parse_uploaded_image(file)

    try:
        boxes, img_rgb = detect_handwritten_boxes(pil_image)
        crops = crop_boxes(img_rgb, boxes, padding=10)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Box detection failed: {exc}") from exc

    crop_images = []
    for i, crop in enumerate(crops, start=1):
        img_bytes = BytesIO()
        crop.save(img_bytes, format="PNG")
        encoded = base64.b64encode(img_bytes.getvalue()).decode("ascii")
        crop_images.append({"id": i, "image": f"data:image/png;base64,{encoded}"})

    return {
        "count": len(crop_images),
        "boxes": boxes,
        "crops": crop_images,
    }
