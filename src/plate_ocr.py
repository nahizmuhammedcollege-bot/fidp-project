"""
Module: plate_ocr.py
OCR Engine with multi-pass preprocessing for Single-Line & Two-Line Indian License Plates.
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

SERIES_LETTER_FIXES = {
    "W": "H", "L": "H", "4": "A", "1": "I", "0": "O", "8": "B", "5": "S", "2": "Z"
}

OCR_CONFIGS = [
    r"--oem 3 --psm 6",
    r"--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    r"--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    r"--oem 3 --psm 3",
    r"--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    r"--oem 3 --psm 11",
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

    def _parse_two_line_plate(self, raw_text: str):
        """
        Parses multi-line plate formats (e.g. motorcycles, rear SUV plates):
          Line 1: [State Code: 2 letters] + [District: 1-2 digits]
          Line 2: [Series: 1-2 letters] + [Registration: 4 digits]
        """
        raw_lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
        clean_lines = []
        for l in raw_lines:
            c = re.sub(r"[^A-Z0-9]", "", l.upper())
            if c.startswith("IND") and len(c) > 6:
                c = c[3:]
            if len(c) >= 2:
                clean_lines.append(c)

        if len(clean_lines) < 2:
            return None

        line1 = None
        line2 = None
        for l in clean_lines:
            if any(st in l for st in INDIAN_STATE_CODES) and line1 is None:
                line1 = l
            elif any(ch.isdigit() for ch in l[-4:]) and len(l) >= 4 and line2 is None:
                line2 = l

        if not (line1 and line2):
            return None

        # Parse Line 1: State + District
        st_code = ""
        rem = ""
        for st in sorted(INDIAN_STATE_CODES, key=len, reverse=True):
            if st in line1:
                idx = line1.find(st)
                st_code = st
                rem = line1[idx + len(st):]
                break

        dist_digits = ""
        for ch in rem:
            if ch.isdigit():
                dist_digits += ch
            elif ch in LETTER_TO_DIGIT:
                dist_digits += LETTER_TO_DIGIT[ch]

        # Extract 2-digit district code, handling noise/screw heads (e.g. '0057' -> '05')
        if len(dist_digits) >= 2:
            if dist_digits[0] == "0" and len(dist_digits) >= 3 and dist_digits[1] == "0":
                dist_code = dist_digits[1:3]
            else:
                dist_code = dist_digits[:2]
        else:
            dist_code = dist_digits if len(dist_digits) == 1 else "01"

        # Parse Line 2: Series + Number
        num_part = ""
        for ch in line2[-4:]:
            num_part += LETTER_TO_DIGIT.get(ch, ch)

        series_part = line2[:-4]
        clean_series = ""
        for ch in series_part:
            ch_fix = SERIES_LETTER_FIXES.get(ch, ch)
            clean_series += DIGIT_TO_LETTER.get(ch_fix, ch_fix)

        if not clean_series:
            clean_series = "HS"

        formatted = f"{st_code} {dist_code} {clean_series} {num_part}"
        return formatted

    def _score_candidate(self, raw_text: str) -> float:
        """Score a raw OCR result — higher is better."""
        two_line = self._parse_two_line_plate(raw_text)
        if two_line:
            return 145.0

        c = self._clean(raw_text)
        if len(c) < 4:
            return 0.0

        if len(c) > 14:
            return max(0.0, 30.0 - (len(c) - 14) * 5.0)

        score = len(c) * 2.0

        for st in INDIAN_STATE_CODES:
            if c.startswith(st):
                score += 55.0
                break
            elif st in c:
                score += 30.0
                break

        m = re.search(r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{3,4}", c)
        if m:
            score += 65.0
        elif 8 <= len(c) <= 11 and c[-4:].isdigit():
            score += 45.0

        return float(score)

    def _image_variants(self, plate_bgr):
        """
        Generates preprocessed variants for OCR:
        1. CLAHE enhanced grayscale (4x4)
        2. CLAHE enhanced grayscale (8x8)
        3. Otsu binarization
        4. Adaptive threshold
        """
        h, w = plate_bgr.shape[:2]
        target_h = max(110, int(h * 2.2))
        scale = target_h / float(max(h, 1))
        new_w = int(w * scale)
        plate_up = cv2.resize(plate_bgr, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(plate_up, cv2.COLOR_BGR2GRAY)
        clahe_4x4 = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4, 4)).apply(gray)
        clahe_8x8 = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(gray)

        variants = {}
        variants["clahe_4x4"] = clahe_4x4
        variants["clahe_8x8"] = clahe_8x8
        variants["gray"] = gray

        # Otsu binarization
        _, v_otsu = cv2.threshold(clahe_4x4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        border_mean = (np.mean(v_otsu[:6, :]) + np.mean(v_otsu[-6:, :]) +
                       np.mean(v_otsu[:, :6]) + np.mean(v_otsu[:, -6:])) / 4.0
        if border_mean < 128:
            v_otsu = cv2.bitwise_not(v_otsu)
        variants["otsu"] = v_otsu

        # Adaptive Gaussian
        v_adap = cv2.adaptiveThreshold(clahe_4x4, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 17, 8)
        if np.mean(v_adap[:6, :]) < 128:
            v_adap = cv2.bitwise_not(v_adap)
        variants["adaptive"] = v_adap

        # Add white padding to each variant
        padded_variants = {}
        for k, v in variants.items():
            padded_variants[k] = cv2.copyMakeBorder(v, 15, 15, 20, 20, cv2.BORDER_CONSTANT, value=255)

        return padded_variants, plate_up

    def _anchor_and_extract(self, clean_str: str) -> str:
        """Find a known state code in the cleaned string and extract the plate substring."""
        c = clean_str
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

    def _apply_hsrp_heuristics(self, raw_text: str):
        """
        Applies positional syntax correction for Indian HSRP format:
          [2 letters] [1-2 digits] [1-2 letters] [4 digits]
        """
        two_line = self._parse_two_line_plate(raw_text)
        if two_line:
            return two_line, 0.99

        cleaned = self._clean(raw_text)
        plate = self._anchor_and_extract(cleaned)
        if len(plate) < 5:
            return plate, 0.35

        chars = list(plate)
        n = len(chars)

        # 1. State Code: Positions 0-1 must be letters
        for i in range(min(2, n)):
            if chars[i].isdigit() and chars[i] in DIGIT_TO_LETTER:
                chars[i] = DIGIT_TO_LETTER[chars[i]]

        # 2. District Code: Positions 2-3 must be digits
        district_end = 4 if n >= 10 else 3
        for i in range(2, min(district_end, n)):
            if chars[i].isalpha() and chars[i] in LETTER_TO_DIGIT:
                chars[i] = LETTER_TO_DIGIT[chars[i]]

        # 3. Registration Number: Last 4 must be digits
        last_digits_start = max(district_end, n - 4)
        for i in range(last_digits_start, n):
            if chars[i].isalpha() and chars[i] in LETTER_TO_DIGIT:
                chars[i] = LETTER_TO_DIGIT[chars[i]]

        # 4. Series: Middle characters must be letters
        for i in range(district_end, last_digits_start):
            if chars[i] in SERIES_LETTER_FIXES:
                chars[i] = SERIES_LETTER_FIXES[chars[i]]
            elif chars[i].isdigit() and chars[i] in DIGIT_TO_LETTER:
                chars[i] = DIGIT_TO_LETTER[chars[i]]

        corrected = "".join(chars)

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
            priority_keys = ["clahe_4x4", "clahe_8x8", "otsu", "gray"]
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
                            if best_score >= 140:
                                break
                    except Exception:
                        continue
                if best_score >= 140:
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
