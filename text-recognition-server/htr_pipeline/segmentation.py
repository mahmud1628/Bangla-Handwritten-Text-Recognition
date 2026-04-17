import os
import shutil
import subprocess

import cv2

from .config import LINE_MODEL_PATH, WORD_MODEL_PATH, YOLO_PATH


def sort_by_number(item_name):
    parts = item_name.replace(".jpg", "").replace(".png", "").replace(".txt", "").split("_")
    return tuple(int(p) for p in parts if p.isdigit())


class LineSort:
    def __init__(self, txt_files, txt_loc, sort_label, flag):
        self.txt_files = txt_files
        self.txt_loc = txt_loc
        self.sort_label = sort_label
        self.flag = flag
        self.read_file()

    def read_file(self):
        files = self.txt_files
        os.makedirs(self.sort_label, exist_ok=True)
        for file_name in files:
            txt_file = []
            file_loc = self.txt_loc + file_name
            with open(file_loc, "r", encoding="utf-8", errors="ignore") as lines:
                for line in lines:
                    token = line.split()
                    if len(token) >= 6:
                        txt_file.append(token)

            print(f"Processing file {file_name}: Number of detections: {len(txt_file)}")

            if self.flag == 0:
                sorted_txt_file = sorted(txt_file, key=lambda x: float(x[2]))
            else:
                sorted_txt_file = sorted(txt_file, key=lambda x: float(x[1]))

            self.file_write(sorted_txt_file, file_name)

    def file_write(self, txt_file, file_name):
        loc = self.sort_label + file_name
        with open(loc, "w", encoding="utf-8") as file:
            for line in txt_file:
                file.write(" ".join(line) + "\n")


def sort_detection_label(txt_loc, sort_label, flag):
    txt_files = os.listdir(txt_loc)
    LineSort(txt_files, txt_loc, sort_label, flag)


def word_segmentation(line_images_dir, word_labels_dir, output_dir):
    line_img = os.listdir(line_images_dir)
    word_label = os.listdir(word_labels_dir)
    print(f"Line images: {line_img}")
    print(f"Word labels: {word_label}")

    word_count = 0

    for label_name in word_label:
        for image_name in line_img:
            fn_i = label_name.split(".")
            fn_j = image_name.split(".")
            if fn_i[0] == fn_j[0]:
                target_dir = os.path.join(output_dir, fn_i[0])
                os.makedirs(target_dir, exist_ok=True)

                img = cv2.imread(os.path.join(line_images_dir, image_name))
                dh, dw, _ = img.shape
                with open(os.path.join(word_labels_dir, label_name), "r", encoding="utf-8") as txt_lb:
                    txt_lb_data = txt_lb.readlines()
                img_lb = fn_i[0]

                k = 1
                for dt in txt_lb_data:
                    parts = dt.strip().split()
                    if len(parts) < 5:
                        continue

                    _, x, y, w, h = map(float, parts[:5])

                    margin = 0.01
                    left = max(0, int((x - (w / 2 + margin)) * dw))
                    right = min(dw, int((x + (w / 2 + margin)) * dw))
                    top = max(0, int((y - (h / 2 + margin)) * dh))
                    bottom = min(dh, int((y + (h / 2 + margin)) * dh))

                    if right > left and bottom > top:
                        crop = img[top:bottom, left:right]
                        if crop.size > 0:
                            cv2.imwrite(f"{target_dir}/{img_lb}_{k}.jpg", crop)
                            word_count += 1
                            k += 1

    return word_count


def run_yolo_detection(image_path, model_path, output_dir, conf=0.4, is_word_detection=False):
    yolo_path = YOLO_PATH

    if not os.path.exists(yolo_path):
        print(f"ERROR: YOLOv5 not found at {yolo_path}")
        print("Please clone YOLOv5: git clone https://github.com/ultralytics/yolov5")
        return None

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return None

    print(f"Running YOLO detection with model: {os.path.basename(model_path)}")
    detect_script = os.path.join(yolo_path, "detect.py")

    if is_word_detection:
        cmd = [
            "python",
            detect_script,
            "--weights",
            model_path,
            "--img",
            "640",
            "--conf-thres",
            "0.20",
            "--iou-thres",
            "0.3",
            "--source",
            image_path,
            "--save-conf",
            "--save-txt",
            "--agnostic-nms",
            "--augment",
            "--max-det",
            "150",
            "--project",
            output_dir,
            "--name",
            "detect",
            "--exist-ok",
            "--device",
            "cpu",
        ]
    else:
        cmd = [
            "python",
            detect_script,
            "--weights",
            model_path,
            "--img",
            "640",
            "--conf-thres",
            str(conf),
            "--source",
            image_path,
            "--save-conf",
            "--save-txt",
            "--project",
            output_dir,
            "--name",
            "detect",
            "--exist-ok",
            "--device",
            "cpu",
        ]

    result = subprocess.run(cmd, capture_output=False)
    detect_dir = os.path.join(output_dir, "detect")

    if result.returncode != 0:
        print(f"YOLO detection failed with exit code: {result.returncode}")
        return None

    labels_dir = os.path.join(detect_dir, "labels")
    if os.path.exists(labels_dir):
        num_detections = len([f for f in os.listdir(labels_dir) if f.endswith(".txt")])
        print(f"Detected {num_detections} objects")
    else:
        print("No labels directory found")

    return detect_dir


def crop_detections(image_path, label_path, output_dir, sort_axis="x"):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    os.makedirs(output_dir, exist_ok=True)

    cropped_images = []
    detections = []

    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8", errors="ignore") as file:
            lines = file.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            x_center, y_center, w, h = map(float, parts[1:5])
            margin = 0.01

            left = max(0, int((x_center - (w / 2 + margin)) * width))
            right = min(width, int((x_center + (w / 2 + margin)) * width))
            top = max(0, int((y_center - (h / 2 + margin)) * height))
            bottom = min(height, int((y_center + (h / 2 + margin)) * height))

            if right > left and bottom > top:
                detections.append(
                    {
                        "x_center": x_center * width,
                        "y_center": y_center * height,
                        "bbox": (left, top, right, bottom),
                    }
                )

        if sort_axis == "y":
            detections.sort(key=lambda d: d["y_center"])
        else:
            detections.sort(key=lambda d: d["x_center"])

        for idx, detection in enumerate(detections):
            left, top, right, bottom = detection["bbox"]
            cropped = img[top:bottom, left:right]

            if cropped.size > 0:
                output_path = os.path.join(output_dir, f"crop_{idx:03d}.jpg")
                cv2.imwrite(output_path, cropped)
                cropped_images.append(output_path)

    return cropped_images


def segment_image(image_path, temp_dir):
    print("\n" + "=" * 70)
    print("STARTING IMAGE SEGMENTATION")
    print("=" * 70)

    if not os.path.exists(LINE_MODEL_PATH):
        print("\nERROR: Line detection model not found")
        print(f"Expected location: {LINE_MODEL_PATH}")
        return {}

    if not os.path.exists(WORD_MODEL_PATH):
        print("\nERROR: Word detection model not found")
        print(f"Expected location: {WORD_MODEL_PATH}")
        return {}

    print("\nConfiguration:")
    print(f"Input image: {image_path}")
    print(f"Line model: {LINE_MODEL_PATH} ({os.path.getsize(LINE_MODEL_PATH) / (1024 ** 2):.1f} MB)")
    print(f"Word model: {WORD_MODEL_PATH} ({os.path.getsize(WORD_MODEL_PATH) / (1024 ** 2):.1f} MB)")

    print("\nStep 1: Line Detection")
    print("-" * 70)

    line_output = run_yolo_detection(image_path, LINE_MODEL_PATH, temp_dir, conf=0.3, is_word_detection=False)

    if line_output is None:
        print("Line detection failed")
        return {}

    line_images_dir = os.path.join(temp_dir, "final_line_segmentation")
    os.makedirs(line_images_dir, exist_ok=True)

    labels_dir = os.path.join(line_output, "labels")
    if not os.path.exists(labels_dir):
        print(f"No labels directory found at: {labels_dir}")
        return {}

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    label_file = os.path.join(labels_dir, f"{image_name}.txt")
    line_images = crop_detections(image_path, label_file, line_images_dir, sort_axis="y")

    print(f"Found {len(line_images)} lines")

    if len(line_images) == 0:
        print("No lines detected")
        return {}

    print("\nStep 2: Word Detection (All Lines)")
    print("-" * 70)
    word_detect_output = run_yolo_detection(
        line_images_dir,
        WORD_MODEL_PATH,
        temp_dir,
        conf=0.10,
        is_word_detection=True,
    )

    if word_detect_output is None:
        print("Word detection failed")
        return {}

    print("\nStep 3: Sorting Word Labels")
    print("-" * 70)

    word_labels_dir = os.path.join(word_detect_output, "labels")
    sorted_word_labels_dir = os.path.join(temp_dir, "sorted_Word_detection")

    if os.path.exists(sorted_word_labels_dir):
        shutil.rmtree(sorted_word_labels_dir)

    sort_detection_label(word_labels_dir + "/", sorted_word_labels_dir + "/", 1)
    print("Word labels sorted by x-axis")

    print("\nStep 4: Word Segmentation")
    print("-" * 70)

    final_word_dir = os.path.join(temp_dir, "final_word_segmentation")
    os.makedirs(final_word_dir, exist_ok=True)
    word_count = word_segmentation(line_images_dir + "/", sorted_word_labels_dir + "/", final_word_dir)

    print(f"Segmented {word_count} total words")

    print("\nStep 5: Organizing Results")
    print("-" * 70)

    word_segments = {}

    if os.path.exists(final_word_dir):
        word_dirs = [
            d
            for d in os.listdir(final_word_dir)
            if os.path.isdir(os.path.join(final_word_dir, d))
        ]
        word_dirs.sort(key=sort_by_number)

        for idx, word_dir in enumerate(word_dirs):
            line_id = f"line_{idx + 1}"
            word_dir_path = os.path.join(final_word_dir, word_dir)
            word_images = [
                os.path.join(word_dir_path, f)
                for f in os.listdir(word_dir_path)
                if f.endswith(".jpg")
            ]
            word_images.sort(key=sort_by_number)

            word_segments[line_id] = word_images
            print(f"Line {idx + 1}: {len(word_images)} words")

    total_words = sum(len(words) for words in word_segments.values())
    print(f"\nSegmentation complete: {total_words} total words from {len(word_segments)} lines")

    return word_segments
