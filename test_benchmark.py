import cv2
import json
import pytesseract
from src.plate_detector import PlateDetector
from src.plate_ocr import PlateOCR

det = PlateDetector()
ocr = PlateOCR()

with open('dataset/ground_truth.json') as f:
    gt = json.load(f)

for item in gt:
    img = cv2.imread(item['filepath'])
    plate, bbox, _ = det.detect(img)
    bin_img, _ = ocr.preprocess_plate(plate)
    txt6 = pytesseract.image_to_string(bin_img, config='--oem 3 --psm 6').strip()
    txt7 = pytesseract.image_to_string(bin_img, config='--oem 3 --psm 7').strip()
    print("FILE:", item['filename'])
    print("  GT  :", item['ground_truth_raw'])
    print("  PSM6:", repr(txt6))
    print("  PSM7:", repr(txt7))
