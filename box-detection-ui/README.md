# Box Detection UI

React + Vite frontend for the handwritten box detection backend.

## Prerequisites

- Node.js 18+
- npm
- Running backend from `box-detection-server`

## Install

```bash
cd box-detection-ui
npm install
```

## Configure Backend URL

Create `.env` file in this folder if your backend is not on `http://127.0.0.1:8001`:

```bash
VITE_BOX_API_URL=http://127.0.0.1:8001
```

## Run

```bash
npm run dev
```

## Build

```bash
npm run build
```

## API Used

This UI calls:

- `POST /detect-boxes-json`

Request:

- FormData field `file` containing the uploaded image.

Response expected:

- `count`: number of detected boxes
- `boxes`: metadata for each box
- `crops`: array of `{ id, image }` where `image` is a base64 PNG data URL
