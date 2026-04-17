import shutil
import tempfile
import traceback

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from .modeling import load_recognition_components
from .segmentation import segment_image


_COMPONENTS = None


def init_pipeline_components():
    """Load heavy inference components once and reuse across requests."""
    global _COMPONENTS
    if _COMPONENTS is None:
        _COMPONENTS = load_recognition_components()
    return _COMPONENTS


def _get_components():
    if _COMPONENTS is None:
        return init_pipeline_components()
    return _COMPONENTS


def recognize_word(image_path, num_beams=3):
    try:
        components = _get_components()
        image = Image.open(image_path).convert("RGB")
        image_inputs = components["image_processor"](image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(components["device"])

        with torch.no_grad():
            generated_ids = components["recog_model"].generate(
                pixel_values=pixel_values,
                max_length=128,
                num_beams=int(num_beams),
                bos_token_id=components["text_processor"].bos_token_id,
                eos_token_id=components["text_processor"].eos_token_id,
                pad_token_id=components["text_processor"].pad_token_id,
            )

        return components["text_processor"].decode(generated_ids[0].cpu().numpy())
    except Exception as exc:
        return f"[Error: {str(exc)}]"


def recognize_document(image, num_beams=3):
    """Server-friendly full document recognition that returns structured output."""
    temp_dir = None
    try:
        _get_components()

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        temp_dir = tempfile.mkdtemp(prefix="bengali_htr_")

        preprocessed_gray = preprocess_input_image(image)
        image_path = tempfile.NamedTemporaryFile(dir=temp_dir, suffix=".jpg", delete=False).name
        cv2.imwrite(image_path, preprocessed_gray)

        word_segments = segment_image(image_path, temp_dir)
        if not word_segments:
            raise ValueError(
                "No text detected in the image. Check model files, YOLOv5 setup, and image quality."
            )

        full_text_lines = []
        total_words = sum(len(words) for words in word_segments.values())

        for line_id in sorted(word_segments.keys(), key=lambda x: int(x.split("_")[1])):
            word_images = word_segments[line_id]
            line_words = [recognize_word(word_img, num_beams=num_beams) for word_img in word_images]
            full_text_lines.append(" ".join(line_words))

        full_text = "\n".join(full_text_lines)
        return {
            "line_count": len(word_segments),
            "word_count": total_words,
            "full_text": full_text,
        }
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def preprocess_input_image(image):
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = image
    else:
        raise ValueError("Unsupported input image type")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    brightened = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)

    thresh = cv2.adaptiveThreshold(
        brightened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=10,
    )

    return cv2.fastNlMeansDenoising(thresh, h=15)


def process_full_document(image, num_beams=3, progress=gr.Progress()):
    temp_dir = None
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        progress(0.1, desc="Preprocessing uploaded image...")
        temp_dir = tempfile.mkdtemp(prefix="bengali_htr_")

        original_image_path = tempfile.NamedTemporaryFile(
            dir=temp_dir, suffix=".jpg", delete=False
        ).name
        image.save(original_image_path)

        preprocessed_gray = preprocess_input_image(image)
        image_path = tempfile.NamedTemporaryFile(dir=temp_dir, suffix=".jpg", delete=False).name
        cv2.imwrite(image_path, preprocessed_gray)

        preview_image = cv2.cvtColor(preprocessed_gray, cv2.COLOR_GRAY2RGB)

        progress(0.2, desc="Segmenting image (detecting lines and words)...")
        word_segments = segment_image(image_path, temp_dir)

        if not word_segments:
            error_msg = (
                "No text detected in the image.\n\n"
                "Possible issues:\n"
                "1. YOLO models (line_model_best.pt, word_model_best.pt) are missing\n"
                "2. YOLOv5 is not installed or not found\n"
                "3. Image quality is too low\n"
                "4. Text is too skewed or unclear\n\n"
                "Check the console or terminal for detailed error messages."
            )
            return error_msg, "Check terminal output for details", preview_image

        progress(0.5, desc="Recognizing text from words...")

        full_text_lines = []
        detailed_output = []

        total_words = sum(len(words) for words in word_segments.values())
        processed_words = 0

        for line_id in sorted(word_segments.keys(), key=lambda x: int(x.split("_")[1])):
            word_images = word_segments[line_id]
            line_words = []

            for word_img in word_images:
                predicted_text = recognize_word(word_img, num_beams=num_beams)
                line_words.append(predicted_text)

                processed_words += 1
                progress(
                    0.5 + 0.4 * (processed_words / total_words),
                    desc=f"Recognizing text: {processed_words}/{total_words} words",
                )

            line_text = " ".join(line_words)
            full_text_lines.append(line_text)
            detailed_output.append(f"{line_id}: {line_text}")

        progress(0.95, desc="Finalizing...")

        full_text = "\n".join(full_text_lines)
        detailed_text = "\n\n".join(detailed_output)

        progress(1.0, desc="Complete")

        stats = f"\n\nStatistics:\n- Lines: {len(word_segments)}\n- Total words: {total_words}"
        return full_text, detailed_text + stats, preview_image

    except Exception as exc:
        error_msg = f"Error: {str(exc)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, error_msg, None
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
