# Box Detection Server

FastAPI server for detecting handwritten rectangular boxes in an image using OpenCV.

## Features

- Reuses notebook detection logic in modular Python package (`box_detector`).
- Exposes an HTTP API endpoint that accepts an image upload.
- Returns detected box crops as PNG images in a ZIP file.
- Includes detection metadata in `boxes.json` inside the ZIP.

## Project Structure

- `box_detector/detector.py`: Core reusable detection and cropping logic.
- `server.py`: FastAPI app and API endpoints.
- `requirements.txt`: Runtime dependencies.

## Setup

```bash
cd box-detection-server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

## API

### `POST /detect-boxes`

- Form field: `file` (image)
- Response: `application/zip`
- ZIP contents:
  - `boxes.json`
  - `box_1.png`, `box_2.png`, ...

### `POST /detect-boxes-json`

- Form field: `file` (image)
- Response: JSON
- JSON fields:
  - `count`: detected box count
  - `boxes`: detection metadata list
  - `crops`: list of `{ id, image }` where `image` is a base64 PNG data URL

### `GET /health`

Returns service health status.

## Example cURL

```bash
curl -X POST "http://localhost:8001/detect-boxes" \
  -H "accept: application/zip" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg" \
  --output detected_boxes.zip
```
