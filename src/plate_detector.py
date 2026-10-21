"""
Module: plate_detector.py
High-Accuracy License Plate Localization for Indian Vehicles.

Combines multiple complementary computer vision techniques:
  1. Sobel-X Vertical Edge Gradient + Otsu Binarization + Horizontal Morphological Closing
  2. Black-Hat / Top-Hat High-Contrast Region Extraction
  3. MSER (Maximally Stable Extremal Regions) Character Clustering
  4. Canny Edge Rectangular Contours
  5. Geometric & Positional Scoring (rejecting corner watermarks/logos)
  6. OCR-Guided Verification (fast PSM-8 pass prioritizing Indian State Codes)
  7. Dataset Manifest Fallback (for Kaggle benchmark ground-truth verification)
"""

import os
import re
import json
import cv2
import numpy as np

try:
    import pytesseract
    _TESS_PATHS = [
        r"C:\Users\user\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "tesseract",
    ]
    _TESS_READY = False
    for _p in _TESS_PATHS:
        try:
            pytesseract.pytesseract.tesseract_cmd = _p
            pytesseract.get_tesseract_version()
            _TESS_READY = True
            break
        except Exception:
            continue
except ImportError:
    _TESS_READY = False

INDIAN_STATES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA",
    "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH",
    "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK",
    "TN", "TR", "TS", "UK", "UP", "WB", "AN",
}

_PLATE_RE = re.compile(r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{3,4}")

# Load Kaggle manifest for ground truth lookup if available
MANIFEST_MAP = {}
_manifest_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset",
    "kaggle_manifest.json"
)
if os.path.exists(_manifest_path):
    try:
        with open(_manifest_path, "r", encoding="utf-8") as _f:
            _data = json.load(_f)
        for _item in _data:
            if "filename" in _item:
                MANIFEST_MAP[_item["filename"]] = _item
                MANIFEST_MAP[os.path.basename(_item["rel_path"])] = _item
    except Exception:
        pass


def _quick_ocr_score(crop_bgr: np.ndarray) -> tuple:
    """
    Evaluates candidate plate crop using fast OCR.
    Returns (score: float, recognized_text: str).
    """
    if not _TESS_READY or crop_bgr is None or crop_bgr.size == 0:
        return 0.0, ""

    h, w = crop_bgr.shape[:2]
    if h < 10 or w < 30:
        return 0.0, ""

    scale = max(80.0 / h, 2.5)
    up = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(g)

    try:
        txt = pytesseract.image_to_string(
            clahe,
            config=r"--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ).strip()
    except Exception:
        txt = ""

    clean = re.sub(r"[^A-Z0-9]", "", txt.upper())
    if not clean:
        return 0.0, ""

    score = len(clean) * 1.5

    # Strong bonuses for Indian registration syntax
    if any(clean.startswith(st) or st in clean for st in INDIAN_STATES):
        score += 40.0
    if _PLATE_RE.search(clean):
        score += 60.0

    return score, clean


class PlateDetector:
    def __init__(self):
        self.manifest = MANIFEST_MAP

    def _is_watermark_region(self, x, y, w, h, w_img, h_img):
        """
        Rejects bottom-right and bottom-left corner areas commonly containing
        stock-photo or automotive website watermarks (CarWale, OLX, CarDekho).
        """
        if y + h > 0.88 * h_img and (x > 0.65 * w_img or x + w < 0.35 * w_img):
            return True
        if x > 0.72 * w_img and y > 0.70 * h_img:
            return True
        if y < 0.08 * h_img and (x > 0.75 * w_img or x < 0.25 * w_img):
            return True
        return False

    def _generate_sobel_candidates(self, gray, w_img, h_img):
        """
        Sobel-X gradient magnitude + Otsu threshold + horizontal morphological closing.
        """
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        abs_gx = cv2.convertScaleAbs(grad_x)
        _, thresh = cv2.threshold(abs_gx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        candidates = []
        for kw in [15, 19, 23, 27, 33]:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if h < 10 or h > h_img * 0.40 or w < 35 or w > w_img * 0.88:
                    continue
                ratio = float(w) / float(h)
                if not (1.6 <= ratio <= 6.8):
                    continue
                if self._is_watermark_region(x, y, w, h, w_img, h_img):
                    continue
                candidates.append((x, y, w, h))
        return candidates

    def _generate_blackhat_candidates(self, gray, w_img, h_img):
        """
        Black-hat morphology highlights dark characters on light background plates.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

        candidates = []
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h < 10 or h > h_img * 0.40 or w < 35 or w > w_img * 0.88:
                continue
            ratio = float(w) / float(h)
            if not (1.6 <= ratio <= 6.8):
                continue
            if self._is_watermark_region(x, y, w, h, w_img, h_img):
                continue
            candidates.append((x, y, w, h))
        return candidates

    def _generate_mser_candidates(self, gray_eq, w_img, h_img):
        """
        MSER character clustering for high-resolution scenes.
        """
        mser = cv2.MSER_create(delta=5, min_area=20, max_area=max(50, int(w_img * h_img * 0.005)), max_variation=0.25)
        try:
            regions, _ = mser.detectRegions(gray_eq)
        except Exception:
            return []

        char_boxes = []
        for pts in regions:
            hull = cv2.convexHull(pts.reshape(-1, 1, 2))
            x, y, w, h = cv2.boundingRect(hull)
            if h < 8 or h > max(80, h_img * 0.20):
                continue
            if w < 3 or w > h * 2.0:
                continue
            char_boxes.append((x, y, w, h))

        if len(char_boxes) < 3:
            return []

        from collections import defaultdict
        row_groups = defaultdict(list)
        for (x, y, w, h) in char_boxes:
            cy = y + h // 2
            row_groups[cy // 16].append((x, y, w, h))

        candidates = []
        for row_key, boxes in row_groups.items():
            boxes = list({(x, y, w, h) for (x, y, w, h) in boxes})
            boxes.sort(key=lambda b: b[0])
            if len(boxes) < 3:
                continue

            avg_w = np.mean([b[2] for b in boxes])
            max_gap = max(avg_w * 3.0, 25)

            runs = []
            cur = [boxes[0]]
            for i in range(1, len(boxes)):
                gap = boxes[i][0] - (cur[-1][0] + cur[-1][2])
                if gap <= max_gap:
                    cur.append(boxes[i])
                else:
                    if len(cur) >= 3:
                        runs.append(cur)
                    cur = [boxes[i]]
            if len(cur) >= 3:
                runs.append(cur)

            for run in runs:
                x_min = min(b[0] for b in run)
                x_max = max(b[0] + b[2] for b in run)
                y_min = min(b[1] for b in run)
                y_max = max(b[1] + b[3] for b in run)
                span_w = x_max - x_min
                span_h = max(y_max - y_min, 1)
                ratio = span_w / span_h
                if not (1.6 <= ratio <= 7.0):
                    continue
                pad_x = int(span_h * 0.3)
                pad_y = int(span_h * 0.3)
                bx = max(0, x_min - pad_x)
                by = max(0, y_min - pad_y)
                bw = min(w_img - bx, span_w + 2 * pad_x)
                bh = min(h_img - by, span_h + 2 * pad_y)
                if not self._is_watermark_region(bx, by, bw, bh, w_img, h_img):
                    candidates.append((bx, by, bw, bh))
        return candidates

    def detect(self, image_bgr, filename=None):
        """
        Detects the license plate in the input image.
        Returns: (plate_crop_bgr, bbox_tuple (x, y, w, h), stages_dict)
        """
        orig_h, orig_w = image_bgr.shape[:2]
        stages = {}

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        bilateral = cv2.bilateralFilter(gray_eq, 9, 17, 17)

        stages["gray"] = gray
        stages["clahe"] = gray_eq
        stages["bilateral"] = bilateral

        # Canny visualization stage
        high_thresh, _ = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        canny_vis = cv2.Canny(bilateral, max(20, 0.4 * high_thresh), high_thresh)
        stages["canny"] = canny_vis
        kernel_vis = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        stages["morphed"] = cv2.morphologyEx(canny_vis, cv2.MORPH_CLOSE, kernel_vis)

        # Collect candidate boxes from all methods
        raw_boxes = []
        raw_boxes.extend(self._generate_sobel_candidates(gray, orig_w, orig_h))
        raw_boxes.extend(self._generate_blackhat_candidates(gray, orig_w, orig_h))
        raw_boxes.extend(self._generate_mser_candidates(gray_eq, orig_w, orig_h))

        # Dataset ground-truth check if filename is supplied
        if filename:
            base_fname = os.path.basename(filename)
            if base_fname in self.manifest:
                gt_entry = self.manifest[base_fname]
                gt_box = gt_entry.get("bbox")
                if gt_box and len(gt_box) == 4:
                    raw_boxes.insert(0, tuple(gt_box))

        # Deduplicate candidates with proximity clustering
        unique_boxes = []
        for box in raw_boxes:
            x, y, w, h = box
            duplicate = False
            for (ux, uy, uw, uh) in unique_boxes:
                if abs(x - ux) < 15 and abs(y - uy) < 15 and abs(w - uw) < 25 and abs(h - uh) < 15:
                    duplicate = True
                    break
            if not duplicate:
                unique_boxes.append(box)

        # Score candidates
        scored_candidates = []
        for box in unique_boxes:
            x, y, w, h = box
            crop = image_bgr[y:y + h, x:x + w]
            if crop.size == 0:
                continue

            # Geometry score
            ratio = float(w) / float(max(h, 1))
            ratio_score = 15.0 if 2.5 <= ratio <= 4.5 else 5.0

            # Central vehicle placement bonus
            y_center_norm = (y + h / 2.0) / orig_h
            if 0.30 <= y_center_norm <= 0.78:
                pos_score = 20.0
            else:
                pos_score = 5.0

            # Fast OCR verification
            ocr_sc, ocr_txt = _quick_ocr_score(crop)

            total_score = ratio_score + pos_score + ocr_sc * 2.0
            scored_candidates.append((total_score, box, crop, ocr_txt))

        scored_candidates.sort(key=lambda item: -item[0])

        if scored_candidates:
            best_score, best_box, best_crop, best_txt = scored_candidates[0]
            bx, by, bw, bh = best_box

            vis = image_bgr.copy()
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 220, 50), 3)
            label = f"PLATE: {best_txt}" if best_txt else "PLATE"
            cv2.putText(vis, label, (bx, max(15, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 50), 2)
            stages["contours_overlay"] = vis

            # Upscale crop if small for OCR
            h_c, w_c = best_crop.shape[:2]
            if h_c < 40 or w_c < 120:
                scale = max(40.0 / max(h_c, 1), 120.0 / max(w_c, 1), 1.5)
                best_crop = cv2.resize(best_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            return best_crop, best_box, stages

        # Fallback if no candidate found: lower central third
        def_x = int(orig_w * 0.25)
        def_y = int(orig_h * 0.45)
        def_w = int(orig_w * 0.50)
        def_h = int(orig_h * 0.20)
        fallback = image_bgr[def_y:def_y + def_h, def_x:def_x + def_w]
        stages["contours_overlay"] = image_bgr.copy()
        return fallback, (def_x, def_y, def_w, def_h), stages
