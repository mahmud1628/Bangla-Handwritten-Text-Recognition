# Box Detection Server

FastAPI server for detecting handwritten rectangular boxes in an image using OpenCV.

## Features

- Reuses notebook detection logic in modular Python package (`box_detector`).
- Exposes an HTTP API endpoint that accepts an image upload.
- Returns detected box crops as PNG images in a ZIP file.
- Includes detection metadata in `boxes.json` inside the ZIP.
- Calls text recognition API for each detected crop and includes recognized text in responses.

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

## OCR Integration

This server calls the text-recognition API after box detection for each cropped box.

Set recognizer URL via environment variable:

```bash
export TEXT_RECOGNIZER_URL=http://127.0.0.1:8000/recognize
```

If not set, default is `http://127.0.0.1:8000/recognize`.

Optional OCR tuning:

```bash
export TEXT_RECOGNIZER_TIMEOUT_SECONDS=300
export OCR_MAX_WORKERS=4
```

- `TEXT_RECOGNIZER_TIMEOUT_SECONDS`: timeout per OCR request for one box.
- `OCR_MAX_WORKERS`: parallel OCR calls for detected boxes.

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
  - `crops`: list of `{ id, image, recognition }` where:
    - `image` is a base64 PNG data URL
    - `recognition` includes `line_count`, `word_count`, `full_text`, and `error`

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
