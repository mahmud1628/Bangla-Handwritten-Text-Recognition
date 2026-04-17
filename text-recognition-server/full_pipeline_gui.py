"""
Bengali Handwritten Text Recognition - Full Pipeline GUI
Complete end-to-end system: Image → Line/Word Segmentation → Text Recognition
"""

import os
import sys
import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
import numpy as np
import cv2
from transformers import ViTModel, GPT2LMHeadModel, AutoImageProcessor, GPT2Config
import unicodedata
from collections import defaultdict
import shutil
import tempfile
from pathlib import Path
import traceback

# ============================================================================
# Configuration
# ============================================================================

BASE_PATH = '/home/mahmud1628/Documents/3-2/CSE 330/Project/GraDeT-HTR'
CHECKPOINT_PATH = '/home/mahmud1628/Documents/3-2/CSE 330/Project/best_model_new.pt'
VOCAB_FILE = os.path.join(BASE_PATH, 'tokenization', 'bn_grapheme_1296_from_bengali.ai.buet.txt')

# YOLO models for line and word segmentation
LINE_MODEL_PATH = '/home/mahmud1628/Documents/3-2/CSE 330/Project/line_model_best.pt'
WORD_MODEL_PATH = '/home/mahmud1628/Documents/3-2/CSE 330/Project/word_model_best.pt'

sys.path.insert(0, BASE_PATH)

# ============================================================================
# Tokenizer Classes (same as inference_gui.py)
# ============================================================================

class TrieTokenizer:
    def __init__(self, vocab, separator=""):
        self.vocab = vocab
        self.separator = separator
        self.trie = self._make_trie()
        
    def _make_trie(self):
        trie = {}
        for token in self.vocab:
            self._add_token(trie, token)
        return trie
    
    def _add_token(self, trie, token):
        node = trie
        for char in token:
            if char not in node:
                node[char] = {}
            node = node[char]
        node[''] = token
    
    def tokenize(self, text):
        tokens = []
        i = 0
        while i < len(text):
            token, length = self._get_next_token(text, i)
            if token:
                tokens.append(token)
                i += length
            else:
                i += 1
        return tokens
    
    def _get_next_token(self, text, start):
        node = self.trie
        last_token = None
        last_length = 0
        
        for i, char in enumerate(text[start:]):
            if char not in node:
                break
            node = node[char]
            if '' in node:
                last_token = node['']
                last_length = i + 1
        
        return last_token, last_length


class GraphemeTokenizer:
    def __init__(self, tokenizer_class, max_len=64, separator="", blank_token="_", 
                 oov_token="▁", normalize_unicode=False, normalization_mode="NFKC",
                 normalizer="unicode", printer=print, bos_token="_", eos_token="_",
                 add_bos_token=True, add_eos_token=True):
        self.vocab = list(dict.fromkeys([oov_token, blank_token, bos_token, eos_token]))
        self.max_len = max_len
        self.oov_token = oov_token
        self.blank_token = blank_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.tokenizer_class = tokenizer_class
        self.separator = separator
        self.tokenizer = self.tokenizer_class([(idx) for idx in self.vocab], separator=self.separator)
        self.word2index = {token: idx for idx, token in enumerate(self.vocab)}
        self.normalize_unicode = normalize_unicode
        self.normalization_mode = normalization_mode
        self.print = printer
        self.out_of_vocabulary_info = defaultdict(set)
        self.frequency_counter = defaultdict(int)
        self._set_normalizer(normalizer)
        self.pad_token_id = self.word2index[self.blank_token]
        self.bos_token_id = self.word2index[self.bos_token]
        self.eos_token_id = self.word2index[self.eos_token]

    def tokenize(self, text, padding=False, normalize_unicode=None, normalization_mode=None):
        if isinstance(text, list):
            return [self.tokenize(_text, padding, normalize_unicode, normalization_mode) for _text in text]
        
        normalize_unicode = self.normalize_unicode if normalize_unicode is None else normalize_unicode
        if normalize_unicode:
            text = self._unicode_normalizer(text, normalization_mode)
        
        tokens = self.tokenizer.tokenize(text)
        
        if self.add_bos_token and self.add_eos_token:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        elif self.add_bos_token:
            tokens = [self.bos_token] + tokens
        elif self.add_eos_token:
            tokens = tokens + [self.eos_token]
        
        tokens = tokens[:self.max_len]
        n_tokens = len(tokens)
        
        if padding:
            tokens = tokens + [self.blank_token] * (self.max_len - n_tokens)
        
        tokens_id = [self.word2index.get(token, self.word2index[self.oov_token]) for token in tokens]
        attention_mask = [1] * n_tokens + [0] * (len(tokens) - n_tokens)
        
        return {'tokens': tokens, 'input_ids': tokens_id, 'token_len': n_tokens, 'attention_mask': attention_mask}

    def add_tokens(self, vocab, normalize_unicode=None, reset_oov=False):
        normalize_unicode = self.normalize_unicode if normalize_unicode is None else normalize_unicode
        vocab = self._validate_tokens(vocab, normalize_unicode)
        self.vocab = self.vocab + vocab
        self.tokenizer = self.tokenizer_class([(v) for v in self.vocab], separator=self.separator)
        self.word2index = {token: idx for idx, token in enumerate(self.vocab)}
        self.bos_token_id = self.word2index[self.bos_token]
        self.eos_token_id = self.word2index[self.eos_token]
        if reset_oov:
            self.reset_out_of_vocabulary_info(keys=vocab)

    def _validate_tokens(self, vocab, normalize_unicode=False):
        if normalize_unicode:
            vocab = list(map(self._unicode_normalizer, vocab))
        vocab = sorted(list(set(vocab)))
        vocab = [v for v in vocab if v not in self.vocab]
        return vocab

    def _set_normalizer(self, type="unicode"):
        if type == "unicode":
            self.normalizer = lambda text, mode: unicodedata.normalize(mode, text)
        else:
            self.normalizer = lambda text, mode: text

    def _unicode_normalizer(self, text, mode=None):
        mode = self.normalization_mode if mode is None else mode
        text = self.normalizer(text, mode)
        text = text.replace("\u200c", "").replace("\u200d", "")
        return text

    def ids_to_token(self, ids):
        if not ids:
            raise ValueError("ids must be non-empty")
        if not isinstance(ids[0], list):
            token_list = [self.vocab[idx] for idx in ids 
                         if self.vocab[idx] not in [self.blank_token, self.bos_token, self.eos_token]]
            return token_list
        if isinstance(ids[0], list):
            return list(map(self.ids_to_token, ids))

    def ids_to_text(self, ids):
        if not ids:
            raise ValueError("ids must be non-empty")
        tokens = self.ids_to_token(ids)
        if not isinstance(tokens[0], list):
            return "".join(tokens)
        if isinstance(tokens[0], list):
            return list(map("".join, tokens))

    def reset_out_of_vocabulary_info(self, keys=None):
        if isinstance(keys, list):
            for k in keys:
                self.out_of_vocabulary_info.pop(k, None)
            return
        if isinstance(keys, str):
            if keys.lower() == "all":
                self.out_of_vocabulary_info = defaultdict(set)
            return


class BnGraphemizerProcessor:
    def __init__(self, grapheme_file, model_max_length=128, normalize_unicode=True,
                 normalization_mode='NFKC', normalizer="unicode", blank_token="_",
                 bos_token="<s>", eos_token="</s>", add_bos_token=True, add_eos_token=True):
        self.grapheme_file = grapheme_file
        self.model_max_length = model_max_length
        self.blank_token = blank_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.list_of_graphemes = self._load_graphemes()
        self.bn_graphmemizer = self._initialize_graphemizer()
        self.pad_token_id = self.bn_graphmemizer.pad_token_id
        self.bos_token_id = self.bn_graphmemizer.bos_token_id
        self.eos_token_id = self.bn_graphmemizer.eos_token_id
        self.vocab = self.bn_graphmemizer.vocab

    def _load_graphemes(self):
        with open(self.grapheme_file, 'r', encoding='utf-8') as f:
            graphemes = sorted(list(set([line.rstrip('\n\r') for line in f.readlines() if line.strip()])))
        return graphemes

    def _initialize_graphemizer(self):
        graphemizer = GraphemeTokenizer(
            tokenizer_class=TrieTokenizer, max_len=self.model_max_length,
            blank_token=self.blank_token, bos_token=self.bos_token, eos_token=self.eos_token,
            add_bos_token=True, add_eos_token=True
        )
        graphemizer.add_tokens(self.list_of_graphemes, reset_oov=True)
        return graphemizer

    def __call__(self, texts, padding=False):
        bng_text_inputs = self.bn_graphmemizer.tokenize(texts, padding=padding)
        bng_inputs = self._get_tokenized_inputs(bng_text_inputs)
        bng_input_ids = torch.Tensor(bng_inputs['input_ids']).long()
        bng_attention_mask = torch.Tensor(bng_inputs['attention_mask']).long()
        if bng_input_ids.ndim == 1:
            bng_input_ids = bng_input_ids.unsqueeze(0)
        if bng_attention_mask.ndim == 1:
            bng_attention_mask = bng_attention_mask.unsqueeze(0)
        return {'input_ids': bng_input_ids, 'attention_mask': bng_attention_mask}

    def _get_tokenized_inputs(self, inputs):
        if not isinstance(inputs, list):
            return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}
        input_ids, attention_mask = [], []
        for input in inputs:
            if isinstance(input, list):
                input = self._get_tokenized_inputs(input)
            input_ids.append(input['input_ids'])
            attention_mask.append(input['attention_mask'])
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def decode(self, input_ids):
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().numpy() if input_ids.is_cuda else input_ids.numpy()
        if isinstance(input_ids, np.ndarray):
            input_ids = input_ids.tolist()
        if isinstance(input_ids, list):
            if len(input_ids) == 0:
                return ""
            if isinstance(input_ids[0], list):
                return [self.decode(ids) for ids in input_ids]
            else:
                token_list = self.bn_graphmemizer.ids_to_token(input_ids)
                return ''.join(token_list)


# ============================================================================
# Model Definition
# ============================================================================

class ViTGPT2EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, max_length=128):
        super().__init__()
        
        self.encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
        gpt2_config.add_cross_attention = True
        gpt2_config.is_decoder = True
        
        self.decoder = GPT2LMHeadModel(gpt2_config)
        pretrained_state = GPT2LMHeadModel.from_pretrained('openai-community/gpt2').state_dict()
        self.decoder.load_state_dict(pretrained_state, strict=False)
        self.decoder.resize_token_embeddings(vocab_size)
        
        vit_hidden_size = self.encoder.config.hidden_size
        gpt2_hidden_size = self.decoder.config.n_embd
        
        if vit_hidden_size != gpt2_hidden_size:
            self.encoder_projection = nn.Linear(vit_hidden_size, gpt2_hidden_size)
        else:
            self.encoder_projection = nn.Identity()
        
        self.vocab_size = vocab_size
        self.max_length = max_length
    
    @torch.no_grad()
    def generate(self, pixel_values, max_length=128, num_beams=1, bos_token_id=1, eos_token_id=2, pad_token_id=0):
        encoder_outputs = self.encoder(pixel_values=pixel_values)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        encoder_hidden_states = self.encoder_projection(encoder_hidden_states)
        
        encoder_attention_mask = torch.ones(
            encoder_hidden_states.shape[:2],
            dtype=torch.long,
            device=encoder_hidden_states.device
        )
        
        batch_size = pixel_values.shape[0]
        input_ids = torch.full(
            (batch_size, 1), 
            bos_token_id, 
            dtype=torch.long, 
            device=pixel_values.device
        )
        
        generated = self.decoder.generate(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            early_stopping=True
        )
        
        return generated


# ============================================================================
# Segmentation Functions
# ============================================================================

def sort_by_number(item_name):
    """Sort helper for underscore-separated numbers"""
    parts = item_name.replace('.jpg', '').replace('.png', '').replace('.txt', '').split('_')
    return tuple(int(p) for p in parts if p.isdigit())


class Line_sort:
    """Sort and filter YOLO detection labels - from notebook"""
    def __init__(self, txt_files, txt_loc, sort_label, flag):
        self.txt_files = txt_files
        self.txt_loc = txt_loc
        self.sort_label = sort_label
        self.flag = flag
        self.read_file()

    def read_file(self):
        files = self.txt_files
        os.makedirs(self.sort_label, exist_ok=True)
        for file in files:
            txt_file = []
            file_loc = self.txt_loc + file
            with open(file_loc, 'r', encoding='utf-8', errors='ignore') as lines:
                for line in lines:
                    token = line.split()
                    if len(token) >= 6:  # Ensure valid YOLO format with confidence
                        txt_file.append(token)

            print(f"Processing file {file}: Number of detections: {len(txt_file)}")

            if self.flag == 0:  # Line detection - sort by y-axis
                sorted_txt_file = sorted(txt_file, key=lambda x: float(x[2]))
            else:  # Word detection - sort by x-axis
                sorted_txt_file = sorted(txt_file, key=lambda x: float(x[1]))

            self.file_write(sorted_txt_file, file)

    def file_write(self, txt_file, file_name):
        loc = self.sort_label + file_name
        with open(loc, 'w') as f:
            for line in txt_file:
                f.write(' '.join(line) + '\n')


def sort_detection_label(txt_loc, sort_label, flag):
    """Sort detection labels - from notebook"""
    txt_files = os.listdir(txt_loc)
    obj = Line_sort(txt_files, txt_loc, sort_label, flag)


def word_segmentation(line_images_dir, word_labels_dir, output_dir):
    """Segment words from line images using sorted labels - from notebook"""
    line_img = os.listdir(line_images_dir)
    word_label = os.listdir(word_labels_dir)
    print(f"Line images: {line_img}")
    print(f"Word labels: {word_label}")

    word_count = 0
    
    for i in word_label:
        for j in line_img:
            fn_i = i.split(".")
            fn_j = j.split(".")
            if fn_i[0] == fn_j[0]:
                dir = os.path.join(output_dir, fn_i[0])
                os.makedirs(dir, exist_ok=True)

                img = cv2.imread(os.path.join(line_images_dir, j))
                dh, dw, _ = img.shape
                txt_lb = open(os.path.join(word_labels_dir, i), 'r')
                txt_lb_data = txt_lb.readlines()
                txt_lb.close()
                img_lb = fn_i[0]

                k = 1
                for dt in txt_lb_data:
                    parts = dt.strip().split()
                    if len(parts) < 5:
                        continue

                    _, x, y, w, h = map(float, parts[:5])

                    # Add small margin for better word capture
                    margin = 0.01
                    l = max(0, int((x - (w/2 + margin)) * dw))
                    r = min(dw, int((x + (w/2 + margin)) * dw))
                    t = max(0, int((y - (h/2 + margin)) * dh))
                    b = min(dh, int((y + (h/2 + margin)) * dh))

                    # Ensure valid crop
                    if r > l and b > t:
                        crop = img[t:b, l:r]
                        if crop.size > 0:
                            cv2.imwrite("{}/{}_{}.jpg".format(dir, img_lb, k), crop)
                            word_count += 1
                            k += 1
    
    return word_count


def run_yolo_detection(image_path, model_path, output_dir, conf=0.4, is_word_detection=False):
    """Run YOLO detection on an image"""
    yolo_path = '/home/mahmud1628/Documents/3-2/CSE 330/Project/yolov5'
    
    # Check if YOLOv5 exists
    if not os.path.exists(yolo_path):
        print(f"❌ ERROR: YOLOv5 not found at {yolo_path}")
        print("Please clone YOLOv5: git clone https://github.com/ultralytics/yolov5")
        return None
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model not found at {model_path}")
        return None
    
    print(f"🔍 Running YOLO detection with model: {os.path.basename(model_path)}")
    
    # Use shlex.quote to properly escape paths with spaces
    import shlex
    detect_script = os.path.join(yolo_path, 'detect.py')
    
    # Build command with proper quoting for paths with spaces
    if is_word_detection:
        # Word detection settings: using conf=0.10 as a balance
        # (0.001 was too permissive and detected noise as words)
        cmd = [
            'python',
            detect_script,
            '--weights', model_path,
            '--img', '640',
            '--conf-thres', '0.20',
            '--iou-thres', '0.3',
            '--source', image_path,
            '--save-conf',
            '--save-txt',
            '--agnostic-nms',
            '--augment',
            '--max-det', '150',
            '--project', output_dir,
            '--name', 'detect',
            '--exist-ok',
            '--device', 'cpu'
        ]
    else:
        # Line detection settings
        cmd = [
            'python',
            detect_script,
            '--weights', model_path,
            '--img', '640',
            '--conf-thres', str(conf),
            '--source', image_path,
            '--save-conf',
            '--save-txt',
            '--project', output_dir,
            '--name', 'detect',
            '--exist-ok',
            '--device', 'cpu'
        ]
    
    # Run with subprocess for better path handling
    import subprocess
    print(f"   Running detection...")
    result = subprocess.run(cmd, capture_output=False)
    
    detect_dir = os.path.join(output_dir, 'detect')
    
    if result.returncode != 0:
        print(f"❌ YOLO detection failed with exit code: {result.returncode}")
        return None
    
    # Check if any detections were made
    labels_dir = os.path.join(detect_dir, 'labels')
    if os.path.exists(labels_dir):
        num_detections = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
        print(f"   ✓ Detected {num_detections} objects")
    else:
        print(f"   ⚠️ No labels directory found")
    
    return detect_dir


def crop_detections(image_path, label_path, output_dir, sort_axis='x'):
    """Crop detected regions from image based on YOLO labels
    
    Args:
        image_path: Path to input image
        label_path: Path to YOLO label file
        output_dir: Directory to save cropped images
        sort_axis: 'x' for left-to-right (words), 'y' for top-to-bottom (lines)
    """
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    os.makedirs(output_dir, exist_ok=True)
    
    cropped_images = []
    detections = []
    
    if os.path.exists(label_path):
        with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Parse all detections and store with coordinates for sorting
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            x_center, y_center, w, h = map(float, parts[1:5])
            
            # Add margin for better word capture (matching notebook)
            margin = 0.01
            
            # Convert to pixel coordinates with margin
            l = max(0, int((x_center - (w/2 + margin)) * width))
            r = min(width, int((x_center + (w/2 + margin)) * width))
            t = max(0, int((y_center - (h/2 + margin)) * height))
            b = min(height, int((y_center + (h/2 + margin)) * height))
            
            # Validate dimensions before storing
            if r > l and b > t:
                # Store detection with coordinates
                detections.append({
                    'x_center': x_center * width,
                    'y_center': y_center * height,
                    'bbox': (l, t, r, b)
                })
        
        # Sort detections based on specified axis
        if sort_axis == 'y':
            # Sort by y-coordinate (top to bottom for lines)
            detections.sort(key=lambda d: d['y_center'])
        else:
            # Sort by x-coordinate (left to right for words)
            detections.sort(key=lambda d: d['x_center'])
        
        # Crop images in sorted order
        for idx, detection in enumerate(detections):
            l, t, r, b = detection['bbox']
            cropped = img[t:b, l:r]
            
            # Ensure valid crop (matching notebook validation)
            if cropped.size > 0:
                output_path = os.path.join(output_dir, f'crop_{idx:03d}.jpg')
                cv2.imwrite(output_path, cropped)
                cropped_images.append(output_path)
    
    return cropped_images  # Already sorted by specified axis


def segment_image(image_path, temp_dir):
    """
    Segment image into lines and words - EXACT notebook workflow
    Returns dictionary of line_id -> list of word image paths
    """
    print("\n" + "="*70)
    print("🔍 STARTING IMAGE SEGMENTATION (Notebook Workflow)")
    print("="*70)
    
    # Check if models exist before processing
    if not os.path.exists(LINE_MODEL_PATH):
        print(f"\n❌ ERROR: Line detection model not found!")
        print(f"   Expected location: {LINE_MODEL_PATH}")
        return {}
    
    if not os.path.exists(WORD_MODEL_PATH):
        print(f"\n❌ ERROR: Word detection model not found!")
        print(f"   Expected location: {WORD_MODEL_PATH}")
        return {}
    
    print(f"\n📋 Configuration:")
    print(f"   Input image: {image_path}")
    print(f"   Line model: {LINE_MODEL_PATH} ({os.path.getsize(LINE_MODEL_PATH)/(1024**2):.1f} MB)")
    print(f"   Word model: {WORD_MODEL_PATH} ({os.path.getsize(WORD_MODEL_PATH)/(1024**2):.1f} MB)")
    
    print("\n📍 Step 1: Line Detection")
    print("-" * 70)
    
    # Step 1: Detect lines
    line_output = run_yolo_detection(image_path, LINE_MODEL_PATH, temp_dir, conf=0.3, is_word_detection=False)
    
    if line_output is None:
        print("❌ Line detection failed!")
        return {}
    
    line_images_dir = os.path.join(temp_dir, 'final_line_segmentation')
    os.makedirs(line_images_dir, exist_ok=True)
    
    # Get detected lines
    labels_dir = os.path.join(line_output, 'labels')
    if not os.path.exists(labels_dir):
        print(f"❌ No labels directory found at: {labels_dir}")
        return {}
    
    # Crop line images
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    label_file = os.path.join(labels_dir, f'{image_name}.txt')
    
    # Sort lines by y-coordinate (top to bottom)
    line_images = crop_detections(image_path, label_file, line_images_dir, sort_axis='y')
    
    print(f"✓ Found {len(line_images)} lines")
    
    if len(line_images) == 0:
        print("❌ No lines detected!")
        return {}
    
    # Step 2: Detect words in ALL lines at once (like notebook)
    print("\n📍 Step 2: Word Detection (All Lines)")
    print("-" * 70)
    print("   Using balanced settings:")
    print("   - conf-thres: 0.10 (balanced threshold)")
    print("   - iou-thres: 0.3")
    print("   - agnostic-nms: enabled")
    print("   - augment: enabled")
    print("   - max-det: 150")
    
    # Run word detection on the line images directory (all at once)
    word_detect_output = run_yolo_detection(line_images_dir, WORD_MODEL_PATH, temp_dir, conf=0.10, is_word_detection=True)
    
    if word_detect_output is None:
        print("❌ Word detection failed!")
        return {}
    
    # Step 3: Sort the word labels (CRITICAL - from notebook)
    print("\n📍 Step 3: Sorting Word Labels")
    print("-" * 70)
    
    word_labels_dir = os.path.join(word_detect_output, 'labels')
    sorted_word_labels_dir = os.path.join(temp_dir, 'sorted_Word_detection')
    
    if os.path.exists(sorted_word_labels_dir):
        shutil.rmtree(sorted_word_labels_dir)
    
    # Sort word labels by x-axis (flag=1)
    flag = 1  # Word detection - sort by x-axis
    sort_detection_label(word_labels_dir + '/', sorted_word_labels_dir + '/', flag)
    
    print("✓ Word labels sorted by x-axis (left to right)")
    
    # Step 4: Segment words using sorted labels (EXACT notebook function)
    print("\n📍 Step 4: Word Segmentation")
    print("-" * 70)
    
    final_word_dir = os.path.join(temp_dir, 'final_word_segmentation')
    os.makedirs(final_word_dir, exist_ok=True)
    
    word_count = word_segmentation(line_images_dir + '/', sorted_word_labels_dir + '/', final_word_dir)
    
    print(f"✓ Segmented {word_count} total words")
    
    # Step 5: Organize results into line -> words dictionary
    print("\n📍 Step 5: Organizing Results")
    print("-" * 70)
    
    word_segments = {}
    
    # List all word directories
    if os.path.exists(final_word_dir):
        word_dirs = [d for d in os.listdir(final_word_dir) if os.path.isdir(os.path.join(final_word_dir, d))]
        word_dirs.sort(key=sort_by_number)
        
        for idx, word_dir in enumerate(word_dirs):
            line_id = f"line_{idx + 1}"
            word_dir_path = os.path.join(final_word_dir, word_dir)
            
            # Get all word images in this directory
            word_images = [os.path.join(word_dir_path, f) for f in os.listdir(word_dir_path) if f.endswith('.jpg')]
            word_images.sort(key=sort_by_number)
            
            word_segments[line_id] = word_images
            print(f"  Line {idx + 1}: {len(word_images)} words")
    
    total_words = sum(len(words) for words in word_segments.values())
    print(f"\n✓ Segmentation complete: {total_words} total words from {len(word_segments)} lines")
    
    return word_segments


# ============================================================================
# Initialize Models
# ============================================================================

print("🚀 Initializing models...")

# Initialize text recognition model
text_processor = BnGraphemizerProcessor(VOCAB_FILE, model_max_length=128)
image_processor = AutoImageProcessor.from_pretrained(
    'google/vit-base-patch16-224',
    size={'height': 224, 'width': 224},
    use_fast=True
)

device = torch.device('cpu')
print(f"📱 Using device: {device}")

recog_model = ViTGPT2EncoderDecoder(vocab_size=len(text_processor.vocab), max_length=128)

try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
except TypeError:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

if isinstance(checkpoint, dict):
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

recog_model.load_state_dict(state_dict, strict=False)
recog_model = recog_model.to(device)
recog_model.eval()

print("✅ Models loaded successfully!")


# ============================================================================
# Main Processing Pipeline
# ============================================================================

def recognize_word(image_path, num_beams=3):
    """Recognize text from a single word image"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_inputs = image_processor(image, return_tensors="pt")
        pixel_values = image_inputs['pixel_values'].to(device)
        
        with torch.no_grad():
            generated_ids = recog_model.generate(
                pixel_values=pixel_values,
                max_length=128,
                num_beams=int(num_beams),
                bos_token_id=text_processor.bos_token_id,
                eos_token_id=text_processor.eos_token_id,
                pad_token_id=text_processor.pad_token_id
            )
        
        predicted_text = text_processor.decode(generated_ids[0].cpu().numpy())
        return predicted_text
    
    except Exception as e:
        return f"[Error: {str(e)}]"


def process_full_document(image, num_beams=3, progress=gr.Progress()):
    """
    Full pipeline: Image → Segmentation → Recognition → Full Text
    """
    try:
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        progress(0.1, desc="Saving uploaded image...")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix='bengali_htr_')
        
        # Save uploaded image
        image_path = os.path.join(temp_dir, 'input_image.jpg')
        image.save(image_path)
        
        progress(0.2, desc="Segmenting image (detecting lines and words)...")
        
        # Segment image into words
        word_segments = segment_image(image_path, temp_dir)
        
        if not word_segments:
            error_msg = (
                "❌ No text detected in the image.\n\n"
                "Possible issues:\n"
                "1. YOLO models (line_model_best.pt, word_model_best.pt) are missing\n"
                "2. YOLOv5 is not installed or not found\n"
                "3. Image quality is too low\n"
                "4. Text is too skewed or unclear\n\n"
                "Check the console/terminal for detailed error messages."
            )
            return error_msg, "Check terminal output for details"
        
        # Recognize text from each word
        progress(0.5, desc="Recognizing text from words...")
        
        full_text_lines = []
        detailed_output = []
        
        total_words = sum(len(words) for words in word_segments.values())
        processed_words = 0
        
        for line_id in sorted(word_segments.keys(), key=lambda x: int(x.split('_')[1])):
            word_images = word_segments[line_id]
            line_words = []
            
            for word_img in word_images:
                predicted_text = recognize_word(word_img, num_beams=num_beams)
                line_words.append(predicted_text)
                
                processed_words += 1
                progress(0.5 + 0.4 * (processed_words / total_words), 
                        desc=f"Recognizing text: {processed_words}/{total_words} words")
            
            line_text = ' '.join(line_words)
            full_text_lines.append(line_text)
            detailed_output.append(f"{line_id}: {line_text}")
        
        progress(0.95, desc="Finalizing...")
        
        # Combine all lines
        full_text = '\n'.join(full_text_lines)
        detailed_text = '\n\n'.join(detailed_output)
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        progress(1.0, desc="Complete!")
        
        stats = f"\n\n📊 Statistics:\n- Lines: {len(word_segments)}\n- Total words: {total_words}"
        
        return full_text, detailed_text + stats
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, error_msg


# ============================================================================
# Gradio Interface
# ============================================================================

# Replace the create_interface function (around line 587-682)

def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="Bengali HTR - Full Pipeline") as demo:
        gr.Markdown(
            """
            # 🇧🇩 Bengali Handwritten Text Recognition - Full Pipeline
            ### Complete System: Image Segmentation → Text Recognition
            
            Upload an image of Bengali handwritten text to get the full transcription.
            The system will automatically:
            1. 📄 Detect and segment lines
            2. ✂️ Detect and segment words
            3. 🔤 Recognize text from each word
            4. 📝 Combine into full document text
            """
        )
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="Upload Handwritten Document Image",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=400
                )
                
                num_beams_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Recognition Quality (Beam Search Width)",
                    info="Higher = better quality but slower (recommended: 3)"
                )
                
                process_btn = gr.Button(
                    "🚀 Process Document", 
                    variant="primary", 
                    size="lg"
                )
            
            with gr.Column():
                output_text = gr.Textbox(
                    label="📝 Recognized Text (Full Document)",
                    lines=10,
                    placeholder="The recognized text will appear here..."
                )
                
                detailed_output = gr.Textbox(
                    label="🔍 Detailed Output (Line by Line)",
                    lines=8,
                    placeholder="Detailed line-by-line output with statistics..."
                )
        
        gr.Markdown(
            """
            ### 💡 Tips for Best Results:
            - ✅ Use clear, well-lit images with good contrast
            - ✅ Ensure text is horizontal (not too skewed)
            - ✅ Higher quality images = better recognition
            - ✅ Works with full pages, paragraphs, or sentences
            - ⚙️ Processing time: ~10-30 seconds depending on image size
            
            ### 📋 Supported:
            - Image formats: JPG, PNG, JPEG
            - Content: Bengali handwritten text (lines, paragraphs, full pages)
            """
        )
        
        # Connect the process function
        process_btn.click(
            fn=process_full_document,
            inputs=[image_input, num_beams_slider],
            outputs=[output_text, detailed_output]
        )
    
    return demo


# ============================================================================
# Launch App
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 Starting Bengali HTR Full Pipeline GUI...")
    print("="*70 + "\n")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft()
    )