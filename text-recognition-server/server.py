from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

from htr_pipeline.pipeline import init_pipeline_components, recognize_document


class RecognitionResponse(BaseModel):
    line_count: int
    word_count: int
    full_text: str


app = FastAPI(
    title="Bangla Handwritten Text Recognition API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    # Load models and processors once when the server starts.
    init_pipeline_components()


@app.post("/recognize", response_model=RecognitionResponse)
async def recognize(file: UploadFile = File(...)):
    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        content = await file.read()
        image = Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded image: {exc}") from exc

    try:
        result = recognize_document(image)
        return RecognitionResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {exc}") from exc
