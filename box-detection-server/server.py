from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import os
import zipfile
from io import BytesIO
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError

from box_detector import crop_boxes, detect_handwritten_boxes

TEXT_RECOGNIZER_URL = os.getenv("TEXT_RECOGNIZER_URL", "http://127.0.0.1:8000/recognize")


def get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


TEXT_RECOGNIZER_TIMEOUT_SECONDS = get_int_env("TEXT_RECOGNIZER_TIMEOUT_SECONDS", 300)
OCR_MAX_WORKERS = max(1, get_int_env("OCR_MAX_WORKERS", 4))

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


def recognize_text_from_crop(crop: Image.Image) -> dict[str, Any]:
    img_bytes = BytesIO()
    crop.save(img_bytes, format="PNG")

    boundary = "----BoxDetectionBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"crop.png\"\r\n"
        "Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + img_bytes.getvalue() + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib_request.Request(
        TEXT_RECOGNIZER_URL,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=TEXT_RECOGNIZER_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return {
                "line_count": payload.get("line_count", 0),
                "word_count": payload.get("word_count", 0),
                "full_text": payload.get("full_text", ""),
                "error": None,
            }
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return {
            "line_count": 0,
            "word_count": 0,
            "full_text": "",
            "error": f"Recognizer HTTP {exc.code}: {detail or exc.reason}",
        }
    except Exception as exc:
        return {
            "line_count": 0,
            "word_count": 0,
            "full_text": "",
            "error": str(exc),
        }


def attach_recognition_results(boxes: list[dict], crops: list[Image.Image]) -> list[dict]:
    if not boxes:
        return []

    max_workers = min(OCR_MAX_WORKERS, len(crops))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        recognitions = list(executor.map(recognize_text_from_crop, crops))

    enriched_boxes = []
    for box, recognition in zip(boxes, recognitions):
        enriched_box = dict(box)
        enriched_box["recognition"] = recognition
        enriched_boxes.append(enriched_box)
    return enriched_boxes


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/detect-boxes")
async def detect_boxes(file: UploadFile = File(...)):
    pil_image = await parse_uploaded_image(file)

    try:
        boxes, img_rgb = detect_handwritten_boxes(pil_image)
        crops = crop_boxes(img_rgb, boxes, outer_padding=0, inner_margin=20)
        boxes_with_text = attach_recognition_results(boxes, crops)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Box detection failed: {exc}") from exc

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("boxes.json", json.dumps(boxes_with_text, indent=2))

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
        crops = crop_boxes(img_rgb, boxes, outer_padding=0, inner_margin=20)
        boxes_with_text = attach_recognition_results(boxes, crops)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Box detection failed: {exc}") from exc

    crop_images = []
    for i, crop in enumerate(crops, start=1):
        img_bytes = BytesIO()
        crop.save(img_bytes, format="PNG")
        encoded = base64.b64encode(img_bytes.getvalue()).decode("ascii")
        recognition = boxes_with_text[i - 1].get("recognition", {})
        crop_images.append(
            {
                "id": i,
                "image": f"data:image/png;base64,{encoded}",
                "recognition": recognition,
            }
        )

    return {
        "count": len(crop_images),
        "boxes": boxes_with_text,
        "crops": crop_images,
    }
