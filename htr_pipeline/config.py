from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = str(PROJECT_ROOT / "best_model.pt")
VOCAB_FILE = str(PROJECT_ROOT / "bn_grapheme_1296_from_bengali.ai.buet.txt")

LINE_MODEL_PATH = str(PROJECT_ROOT / "line_model_best.pt")
WORD_MODEL_PATH = str(PROJECT_ROOT / "word_model_best.pt")
YOLO_PATH = str(PROJECT_ROOT / "yolov5")
