from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

COLORS: Sequence[Tuple[int, int, int]] = [
    (230, 57, 70),
    (42, 157, 143),
    (233, 196, 106),
    (38, 70, 83),
    (244, 162, 97),
]


def detect_handwritten_boxes(pil_image: Image.Image) -> tuple[list[dict], np.ndarray]:
    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_h, img_w = img_bgr.shape[:2]
    image_area = img_h * img_w

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    masks = []
    masks.append(cv2.inRange(hsv, np.array([35, 40, 40]), np.array([90, 255, 255])))
    masks.append(cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255])))
    masks.append(cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255])))
    masks.append(cv2.inRange(hsv, np.array([100, 50, 50]), np.array([135, 255, 255])))
    masks.append(cv2.inRange(hsv, np.array([85, 50, 50]), np.array([100, 255, 255])))

    color_mask = masks[0]
    for mask in masks[1:]:
        color_mask = cv2.bitwise_or(color_mask, mask)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25,
        C=10,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    color_dilated = cv2.dilate(color_mask, kernel, iterations=2)
    binary_dilated = cv2.dilate(binary, kernel, iterations=1)

    results = []
    for mask_name, mask in [("color", color_dilated), ("binary", binary_dilated)]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < image_area * 0.003 or area > image_area * 0.65:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 80:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area < 1:
                continue

            solidity = area / hull_area
            approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
            nv = len(approx)
            x, y, bw, bh = cv2.boundingRect(approx)
            aspect = max(bw, bh) / (min(bw, bh) + 1e-5)

            if mask_name == "color":
                if nv < 4 or nv > 12:
                    continue
                if aspect > 8:
                    continue
                hull_extent = hull_area / (bw * bh + 1e-5)
                if hull_extent < 0.50:
                    continue
            else:
                if solidity < 0.50:
                    continue
                if nv < 4 or nv > 10:
                    continue
                if aspect > 8:
                    continue
                extent = area / (bw * bh + 1e-5)
                if extent < 0.20:
                    continue

            vertex_score = max(0.0, 1.0 - abs(nv - 4) * 0.10)
            solidity_score = min(solidity / 0.5, 1.0)
            confidence = vertex_score * 0.5 + solidity_score * 0.5

            results.append(
                {
                    "bbox": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
                    "area": float(area),
                    "num_vertices": nv,
                    "solidity": round(float(solidity), 3),
                    "confidence": round(float(confidence), 3),
                    "source": mask_name,
                }
            )

    results.sort(key=lambda b: (b["confidence"], b["area"]), reverse=True)
    kept = []
    for box in results:
        bx, by, bw2, bh2 = box["bbox"].values()
        skip = False
        for kb in kept:
            kx, ky, kw, kh = kb["bbox"].values()
            ix = max(0, min(bx + bw2, kx + kw) - max(bx, kx))
            iy = max(0, min(by + bh2, ky + kh) - max(by, ky))
            inter = ix * iy
            union = bw2 * bh2 + kw * kh - inter
            if union > 0 and inter / union > 0.40:
                skip = True
                break
        if not skip:
            kept.append(box)

    for i, box in enumerate(kept):
        box["id"] = i + 1

    return kept, img_rgb


def annotate_image(img_rgb: np.ndarray, boxes: List[Dict]) -> Image.Image:
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    for box in boxes:
        color = COLORS[(box["id"] - 1) % len(COLORS)]
        x, y, bw, bh = box["bbox"].values()
        pad = 6
        draw.rectangle([x - pad, y - pad, x + bw + pad, y + bh + pad], outline=color, width=5)
        label = f"  Box {box['id']}  conf={box['confidence']:.2f}  "
        lw = len(label) * 9
        draw.rectangle([x - pad, y - pad - 28, x - pad + lw, y - pad], fill=color)
        draw.text((x - pad + 4, y - pad - 24), label, fill="white")

    return pil_img


def crop_boxes(img_rgb: np.ndarray, boxes: List[Dict], padding: int = 10) -> list[Image.Image]:
    pil_img = Image.fromarray(img_rgb)
    img_w, img_h = pil_img.size
    crops = []

    for box in boxes:
        x, y, bw, bh = box["bbox"].values()
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + bw + padding)
        y2 = min(img_h, y + bh + padding)
        crop = pil_img.crop((x1, y1, x2, y2))
        crops.append(crop)

    return crops
