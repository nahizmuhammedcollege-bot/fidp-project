"""
Module: plate_ocr.py
OCR Engine with multi-pass preprocessing for Indian license plates.
"""

import os
import re
import cv2
import numpy as np
import pytesseract

POSSIBLE_TESSERACT_PATHS = [
    r"C:\Users\user\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "tesseract",
]

TESSERACT_AVAILABLE = False
for _p in POSSIBLE_TESSERACT_PATHS:
    if os.path.exists(_p) or _p == "tesseract":
        try:
            pytesseract.pytesseract.tesseract_cmd = _p
            pytesseract.get_tesseract_version()
            TESSERACT_AVAILABLE = True
            break
        except Exception:
            continue

INDIAN_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA",
    "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH",
    "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK",
    "TN", "TR", "TS", "UK", "UP", "WB", "AN",
}

DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B", "7": "T"}
LETTER_TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8", "T": "7", "J": "1"}

OCR_CONFIGS = [
    r"--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    r"--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    r"--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    r"--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
]


class PlateOCR:
    def __init__(self):
        self.tesseract_ready = TESSERACT_AVAILABLE

    def _clean(self, text: str) -> str:
        """Strip non-alphanumeric and uppercase."""
        cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        if cleaned.startswith("IND") and len(cleaned) > 7:
            cleaned = cleaned[3:]
        return cleaned

    def _contains_state(self, s: str) -> bool:
        return any(st in s for st in INDIAN_STATE_CODES)

    def _score_candidate(self, raw: str) -> float:
        """Score a raw OCR result — higher is better."""
        c = self._clean(raw)
        if len(c) < 4:
            return 0.0
        score = len(c) * 2.0
        # Check if starts with or contains state code
        for st in INDIAN_STATE_CODES:
            if c.startswith(st):
                score += 50.0
                break
            elif st in c:
                score += 30.0
                break
        m = re.search(r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{3,4}", c)
        if m:
            score += 60.0
        return float(score)

    def _image_variants(self, plate_bgr):
        """
        Generates preprocessed variants for OCR:
        1. CLAHE enhanced grayscale (primary)
        2. High-contrast grayscale
        3. Otsu binarization
        4. Adaptive threshold
        5. Otsu inverted
        """
        h, w = plate_bgr.shape[:2]
        target_h = max(100, int(h * 2.5))
        scale = target_h / float(max(h, 1))
        new_w = int(w * scale)
        plate_up = cv2.resize(plate_bgr, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(plate_up, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(gray)

        variants = {}
        variants["clahe"] = clahe
        variants["gray"] = gray

        # Otsu binarization
        _, v_otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Check border polarity
        border_mean = (np.mean(v_otsu[:6, :]) + np.mean(v_otsu[-6:, :]) +
                       np.mean(v_otsu[:, :6]) + np.mean(v_otsu[:, -6:])) / 4.0
        if border_mean < 128:
            v_otsu = cv2.bitwise_not(v_otsu)
        variants["otsu"] = v_otsu

        # Adaptive Gaussian
        v_adap = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 8)
        if np.mean(v_adap[:6, :]) < 128:
            v_adap = cv2.bitwise_not(v_adap)
        variants["adaptive"] = v_adap

        # Add white padding to each variant
        padded_variants = {}
        for k, v in variants.items():
            padded_variants[k] = cv2.copyMakeBorder(v, 15, 15, 20, 20, cv2.BORDER_CONSTANT, value=255)

        return padded_variants, plate_up

    def _anchor_and_extract(self, raw: str) -> str:
        """
        Find a known state code in the raw OCR string and extract the
        plate substring starting there.
        """
        c = self._clean(raw)
        if len(c) < 5:
            return c

        for st in sorted(INDIAN_STATE_CODES, key=len, reverse=True):
            idx = c.find(st)
            if idx != -1:
                sub = c[idx: idx + 10]
                if len(sub) >= 7:
                    return sub

        m = re.search(r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{3,4}", c)
        if m:
            return m.group(0)

        return c

    def _apply_hsrp_heuristics(self, raw: str):
        """
        Applies positional rules of the Indian HSRP format:
          [2 letters] [1-2 digits] [1-2 letters] [4 digits]
        """
        plate = self._anchor_and_extract(raw)
        if len(plate) < 5:
            return plate, 0.35

        chars = list(plate)
        n = len(chars)

        # State Code: Positions 0-1 must be letters
        for i in range(min(2, n)):
            if chars[i].isdigit() and chars[i] in DIGIT_TO_LETTER:
                chars[i] = DIGIT_TO_LETTER[chars[i]]

        # District Code: Positions 2-3 must be digits
        district_end = 4 if n >= 10 else 3
        for i in range(2, min(district_end, n)):
            if chars[i].isalpha() and chars[i] in LETTER_TO_DIGIT:
                chars[i] = LETTER_TO_DIGIT[chars[i]]

        # Registration Number: Last 4 must be digits
        last_digits_start = max(district_end, n - 4)
        for i in range(last_digits_start, n):
            if chars[i].isalpha() and chars[i] in LETTER_TO_DIGIT:
                chars[i] = LETTER_TO_DIGIT[chars[i]]

        # Series: Middle characters must be letters
        for i in range(district_end, last_digits_start):
            if chars[i].isdigit() and chars[i] in DIGIT_TO_LETTER:
                chars[i] = DIGIT_TO_LETTER[chars[i]]

        corrected = "".join(chars)

        # Format spacing
        if len(corrected) == 10:
            formatted = f"{corrected[:2]} {corrected[2:4]} {corrected[4:6]} {corrected[6:10]}"
        elif len(corrected) == 9:
            formatted = f"{corrected[:2]} {corrected[2:3]} {corrected[3:5]} {corrected[5:9]}"
        else:
            formatted = corrected

        conf = 0.50
        if corrected[:2] in INDIAN_STATE_CODES:
            conf += 0.25
        if re.match(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$", corrected):
            conf += 0.20
        if 8 <= len(corrected) <= 10:
            conf += 0.04

        return formatted, min(conf, 0.99)

    def recognize(self, plate_bgr):
        """
        Runs multi-pass OCR on a plate image.
        Returns dict with formatted_text, clean_text, confidence, binary_image, resized_plate.
        """
        variants, resized_plate = self._image_variants(plate_bgr)

        best_raw = ""
        best_score = -1.0
        best_img = list(variants.values())[0]

        if self.tesseract_ready:
            # Check fast priority configs on clahe and otsu first
            priority_keys = ["clahe", "otsu", "gray", "adaptive"]
            for var_key in priority_keys:
                if var_key not in variants:
                    continue
                v_img = variants[var_key]
                for cfg in OCR_CONFIGS:
                    try:
                        text = pytesseract.image_to_string(v_img, config=cfg)
                        sc = self._score_candidate(text)
                        if sc > best_score:
                            best_score = sc
                            best_raw = text
                            best_img = v_img
                            # If high score (>90) found, early exit
                            if best_score >= 100:
                                break
                    except Exception:
                        continue
                if best_score >= 100:
                    break

        formatted, conf = self._apply_hsrp_heuristics(best_raw)
        clean = formatted.replace(" ", "")

        return {
            "formatted_text": formatted,
            "clean_text": clean,
            "raw_text": best_raw.strip(),
            "confidence": round(conf, 3),
            "binary_image": best_img,
            "resized_plate": resized_plate,
        }
