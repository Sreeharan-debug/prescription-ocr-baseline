import easyocr
import pytesseract
from paddleocr import PaddleOCR
import cv2
import json
import difflib
import numpy as np
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

print("Files in this folder:", os.listdir())

# -----------------------------
# LOAD MEDICINE DATABASE
# -----------------------------
with open("medicine_db.json") as f:
    medicine_db = json.load(f)

medicine_list = list(medicine_db.keys())

# -----------------------------
# INITIALISE OCR ENGINES & PATHS
# -----------------------------
# Check standard Windows paths for Tesseract binary if not on PATH
def check_tesseract():
    default_tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for p in default_tesseract_paths:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            break
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

TESSERACT_AVAILABLE = check_tesseract()

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

easy_reader = easyocr.Reader(['en'], verbose=False)
paddle_reader = PaddleOCR(lang='en')

# -----------------------------
# LOAD IMAGE
# -----------------------------
image = cv2.imread("test.jpeg")

if image is None:
    print("Error: Image not found")
    exit()

# -----------------------------
# PREPROCESSING
# -----------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# improve contrast
gray = cv2.convertScaleAbs(gray, alpha=1.6, beta=0)

# noise reduction
gray = cv2.GaussianBlur(gray, (3, 3), 0)

# sharpening
kernel = np.array([
    [0, -1,  0],
    [-1,  5, -1],
    [0, -1,  0]
])

gray = cv2.filter2D(gray, -1, kernel)

# adaptive threshold
thresh = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

# repair broken strokes
kernel = np.ones((2, 2), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# resize for OCR clarity
thresh = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

cv2.imwrite("processed_output.png", thresh)

# -----------------------------------------------------
# ENSEMBLE OCR — runs all three engines in parallel
# -----------------------------------------------------
def run_easyocr(img):
    """Return (tokens, boxes) from EasyOCR."""
    try:
        results = easy_reader.readtext(img)
        tokens = [text for (_, text, _) in results]
        boxes  = [(bbox, text) for (bbox, text, _) in results]
        return tokens, boxes
    except Exception as e:
        print(f"Warning: EasyOCR error: {e}")
        return [], []


def run_paddleocr(img):
    """Return (tokens, boxes) from PaddleOCR."""
    try:
        if img is None:
            return [], []
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        results = paddle_reader.predict(img)
        tokens, boxes = [], []
        for r in results:
            if isinstance(r, dict):
                texts = r.get("rec_texts", [])
                polys = r.get("rec_polys", r.get("rec_boxes", []))
                for text in texts:
                    tokens.append(text)
                for poly, text in zip(polys, texts):
                    boxes.append((poly, text))
            elif isinstance(r, list):
                for line in r:
                    if isinstance(line, list) and len(line) == 2:
                        bbox, text_tuple = line
                        text = text_tuple[0] if isinstance(text_tuple, (list, tuple)) else str(text_tuple)
                        tokens.append(text)
                        boxes.append((bbox, text))
        return tokens, boxes
    except Exception as e:
        print(f"Warning: PaddleOCR error: {e}")
        return [], []


def run_tesseract(img):
    """Return (tokens, []) from Tesseract (no bbox data used for highlighting)."""
    if not TESSERACT_AVAILABLE:
        return [], []
    try:
        text = pytesseract.image_to_string(img, config='--psm 6')
        tokens = [t for t in text.split() if t.strip()]
        return tokens, []
    except Exception as e:
        print(f"Warning: Tesseract OCR error: {e}")
        return [], []


def merge_tokens(token_lists):
    """
    Voting / deduplication merge.

    Strategy:
      1. Normalise every token (lowercase, strip punctuation).
      2. Keep the token if it appears in at least ONE engine's output
         (union merge) — maximises recall for the medicine matcher.
      3. Preserve insertion order; skip near-duplicates using difflib.
    """
    seen = []
    merged = []

    for tokens in token_lists:
        for raw in tokens:
            norm = re.sub(r'[^a-z0-9]', '', raw.lower())
            if not norm:
                continue
            # skip if a very similar token is already present
            close = difflib.get_close_matches(norm, seen, n=1, cutoff=0.85)
            if not close:
                seen.append(norm)
                merged.append(raw)   # keep the original casing/spacing

    return merged


def extract_text_ensemble(preprocessed_img, original_img):
    """
    Run EasyOCR, PaddleOCR, and Tesseract simultaneously on the
    preprocessed and original images to maximize recall.
    Returns (merged_text_string, easy_boxes).
    """
    futures = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures['easy_orig']   = executor.submit(run_easyocr,    original_img)
        futures['easy_proc']   = executor.submit(run_easyocr,    preprocessed_img)
        futures['paddle_orig'] = executor.submit(run_paddleocr,  original_img)
        futures['paddle_proc'] = executor.submit(run_paddleocr,  preprocessed_img)
        futures['tesseract']   = executor.submit(run_tesseract,  preprocessed_img)

    easy_tokens_orig,   easy_boxes_orig   = futures['easy_orig'].result()
    easy_tokens_proc,   easy_boxes_proc   = futures['easy_proc'].result()
    paddle_tokens_orig, paddle_boxes_orig = futures['paddle_orig'].result()
    paddle_tokens_proc, paddle_boxes_proc = futures['paddle_proc'].result()
    tesseract_tokens,   _                 = futures['tesseract'].result()

    all_token_lists = [
        easy_tokens_orig,
        easy_tokens_proc,
        paddle_tokens_orig,
        paddle_tokens_proc,
        tesseract_tokens,
    ]

    merged_tokens = merge_tokens(all_token_lists)
    merged_text   = " ".join(merged_tokens)

    # Collect bounding boxes from EasyOCR and PaddleOCR for downstream use
    all_boxes = easy_boxes_orig + easy_boxes_proc + paddle_boxes_orig + paddle_boxes_proc

    return merged_text, all_boxes


# -----------------------------
# RUN ENSEMBLE
# -----------------------------
print("\nRunning ensemble OCR (EasyOCR + PaddleOCR + Tesseract)...\n")

detected_text, easy_boxes = extract_text_ensemble(thresh, image)

print("Merged Extracted Text:\n")
print(detected_text)

# -----------------------------
# MEDICINE DETECTION
# -----------------------------
def detect_medicine(text, medicine_db):
    words = text.lower().split()
    for word in words:
        clean_word = re.sub(r'\d+', '', word)
        clean_word = clean_word.strip().replace(",", "").replace(".", "")

        # 1. match generic names
        match = difflib.get_close_matches(clean_word, medicine_db.keys(), n=1, cutoff=0.6)
        if match:
            return match[0]

        # 2. match brand names
        for med, info in medicine_db.items():
            brands = info.get("brands", [])
            brand_match = difflib.get_close_matches(clean_word, brands, n=1, cutoff=0.6)
            if brand_match:
                return med

    return None

detected_medicine = detect_medicine(detected_text, medicine_db)

# -----------------------------
# DOSAGE EXTRACTION
# -----------------------------
dosage_pattern = r'\b\d\s*-\s*\d\s*-\s*\d\b'
dosage_match = re.search(dosage_pattern, detected_text)

dosage = dosage_match.group() if dosage_match else "Not detected"

# -----------------------------
# STRENGTH EXTRACTION
# -----------------------------
strength_pattern = r'\d{2,4}\s?mg'
strength_matches = re.findall(strength_pattern, detected_text.lower())

strength = max(strength_matches, key=len) if strength_matches else "Not detected"

# -----------------------------
# FINAL OUTPUT
# -----------------------------
print("\n------ INTERPRETED PRESCRIPTION ------")

if detected_medicine:

    info = medicine_db[detected_medicine]

    print("Medicine:", detected_medicine.capitalize())
    print("Strength:", strength)
    print("Dosage:  ", dosage)

    print("\nUses:")
    print(info["uses"])

    print("\nSide Effects:")
    print(info["side_effects"])

else:
    print("No medicine detected")

print("\nProcessed image saved as: processed_output.png")
