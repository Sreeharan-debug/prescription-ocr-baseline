import streamlit as st
import easyocr
import pytesseract
from paddleocr import PaddleOCR
import cv2
import numpy as np
import json
import difflib
import re
import os
from concurrent.futures import ThreadPoolExecutor

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

# -----------------------------
# LOAD MEDICINE DATABASE
# -----------------------------
with open("medicine_db.json") as f:
    medicine_db = json.load(f)

medicine_list = list(medicine_db.keys())

# -----------------------------
# INITIALISE OCR ENGINES (cached so Streamlit doesn't reload on every run)
# -----------------------------
@st.cache_resource
def load_ocr_engines():
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    easy_reader   = easyocr.Reader(['en'], verbose=False)
    paddle_reader = PaddleOCR(lang='en')
    return easy_reader, paddle_reader

easy_reader, paddle_reader = load_ocr_engines()

# -----------------------------
# MEDICINE DETECTION FUNCTION
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

# --------------------------------------------------
# ENSEMBLE OCR HELPERS
# --------------------------------------------------
def run_easyocr(img):
    """Return (tokens, boxes) from EasyOCR."""
    try:
        results = easy_reader.readtext(img)
        tokens = [text for (_, text, _) in results]
        boxes  = [(bbox, text) for (bbox, text, _) in results]
        return tokens, boxes
    except Exception as e:
        st.warning(f"EasyOCR warning: {e}")
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
        st.warning(f"PaddleOCR warning: {e}")
        return [], []


def run_tesseract(img):
    """Return (tokens, []) from Tesseract."""
    if not TESSERACT_AVAILABLE:
        return [], []
    try:
        text   = pytesseract.image_to_string(img, config='--psm 6')
        tokens = [t for t in text.split() if t.strip()]
        return tokens, []
    except Exception as e:
        print(f"Warning: Tesseract OCR error: {e}")
        return [], []


def merge_tokens(token_lists):
    """
    Union merge with near-duplicate suppression.
    Keeps a token if it doesn't already have a close match in the
    accumulated set (cutoff=0.85), preserving insertion order.
    """
    seen   = []
    merged = []
    for tokens in token_lists:
        for raw in tokens:
            norm = re.sub(r'[^a-z0-9]', '', raw.lower())
            if not norm:
                continue
            if not difflib.get_close_matches(norm, seen, n=1, cutoff=0.85):
                seen.append(norm)
                merged.append(raw)
    return merged


def extract_text_ensemble(preprocessed_img, original_img):
    """
    Run EasyOCR, PaddleOCR, and Tesseract simultaneously on the
    preprocessed and original images to maximize recall.
    Returns (merged_text_string, all_boxes, per_engine_tokens).
    """
    with ThreadPoolExecutor(max_workers=5) as executor:
        f_easy_orig    = executor.submit(run_easyocr,   original_img)
        f_easy_proc    = executor.submit(run_easyocr,   preprocessed_img)
        f_paddle_orig  = executor.submit(run_paddleocr, original_img)
        f_paddle_proc  = executor.submit(run_paddleocr, preprocessed_img)
        f_tesseract    = executor.submit(run_tesseract, preprocessed_img)

        easy_tokens_orig,   easy_boxes_orig   = f_easy_orig.result()
        easy_tokens_proc,   easy_boxes_proc   = f_easy_proc.result()
        paddle_tokens_orig, paddle_boxes_orig = f_paddle_orig.result()
        paddle_tokens_proc, paddle_boxes_proc = f_paddle_proc.result()
        tesseract_tokens,   _                 = f_tesseract.result()

    merged_tokens = merge_tokens([
        easy_tokens_orig,
        easy_tokens_proc,
        paddle_tokens_orig,
        paddle_tokens_proc,
        tesseract_tokens,
    ])

    merged_text = " ".join(merged_tokens)
    all_boxes   = easy_boxes_orig + easy_boxes_proc + paddle_boxes_orig + paddle_boxes_proc

    per_engine = {
        "EasyOCR": easy_tokens_orig + easy_tokens_proc,
        "PaddleOCR": paddle_tokens_orig + paddle_tokens_proc,
        "Tesseract": tesseract_tokens if TESSERACT_AVAILABLE else None,
    }

    return merged_text, all_boxes, per_engine


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.title("💊 AI Prescription Reader")
st.write("Upload a prescription image to extract medicine information.")

uploaded_file = st.file_uploader(
    "Upload Prescription Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Decode uploaded file → OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image      = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded Prescription", use_container_width=True)

    # -----------------------------
    # PREPROCESSING
    # -----------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.6, beta=0)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    sharp_kernel = np.array([
        [0, -1,  0],
        [-1,  5, -1],
        [0, -1,  0]
    ])
    gray = cv2.filter2D(gray, -1, sharp_kernel)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    morph_kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel)
    thresh = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # -----------------------------
    # ENSEMBLE OCR — with spinner
    # -----------------------------
    with st.spinner("🔍 Running ensemble OCR (EasyOCR + PaddleOCR + Tesseract)…"):
        detected_text, easy_boxes, engine_outputs = extract_text_ensemble(thresh, image)

    st.success("✅ Ensemble OCR complete.")

    # -----------------------------
    # ENGINE CONTRIBUTION EXPANDER
    # (shows what each engine contributed for transparency)
    # -----------------------------
    with st.expander("🔬 View per-engine raw outputs"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("EasyOCR")
            st.code(" ".join(engine_outputs.get("EasyOCR", [])) or "(nothing detected)", language=None)
        with col2:
            st.caption("PaddleOCR")
            st.code(" ".join(engine_outputs.get("PaddleOCR", [])) or "(nothing detected)", language=None)
        with col3:
            st.caption("Tesseract")
            if not TESSERACT_AVAILABLE:
                st.caption("⚠️ Not installed / not in PATH (skipped)")
            else:
                st.code(" ".join(engine_outputs.get("Tesseract") or []) or "(nothing detected)", language=None)

    # -----------------------------
    # BOUNDING BOXES (EasyOCR only — has spatial data)
    # -----------------------------
    highlight_boxes = []
    for (bbox, text) in easy_boxes:
        word       = text.lower()
        clean_word = re.sub(r'\d+', '', word).strip().replace(",", "").replace(".", "")

        found = bool(difflib.get_close_matches(clean_word, medicine_db.keys(), n=1, cutoff=0.6))
        if not found:
            for med, info in medicine_db.items():
                if difflib.get_close_matches(clean_word, info.get("brands", []), n=1, cutoff=0.6):
                    found = True
                    break

        if found:
            highlight_boxes.append(bbox)

    image_annotated = image.copy()
    for bbox in highlight_boxes:
        tl = tuple(map(int, bbox[0]))
        br = tuple(map(int, bbox[2]))
        cv2.rectangle(image_annotated, tl, br, (0, 255, 0), 3)

    st.image(image_annotated, caption="Detected Medicine Highlighted", use_container_width=True)

    # -----------------------------
    # AGGREGATED TEXT
    # -----------------------------
    st.subheader("📄 Aggregated Extracted Text")
    st.code(detected_text, language=None)

    # -----------------------------
    # MEDICINE DETECTION
    # -----------------------------
    detected_medicine = detect_medicine(detected_text, medicine_db)

    # -----------------------------
    # DOSAGE EXTRACTION
    # -----------------------------
    dosage_pattern = r'\b\d\s*-\s*\d\s*-\s*\d\b'
    dosage_match   = re.search(dosage_pattern, detected_text)
    dosage         = dosage_match.group() if dosage_match else "Not detected"

    # -----------------------------
    # STRENGTH EXTRACTION
    # -----------------------------
    strength_pattern  = r'\d{2,4}\s?mg'
    strength_matches  = re.findall(strength_pattern, detected_text.lower())
    strength          = max(strength_matches, key=len) if strength_matches else "Not detected"

    # -----------------------------
    # STRUCTURED JSON OUTPUT
    # -----------------------------
    st.subheader("📋 Interpreted Prescription")

    if detected_medicine:
        info = medicine_db[detected_medicine]

        st.success(f"Medicine: {detected_medicine.capitalize()}")

        result_json = {
            "medicine":     detected_medicine.capitalize(),
            "strength":     strength,
            "dosage":       dosage,
            "uses":         info["uses"],
            "side_effects": info["side_effects"],
        }

        st.json(result_json)

    else:
        st.error("No medicine detected in the aggregated text.")
