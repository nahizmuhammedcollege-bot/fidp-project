"""
Module: pipeline.py
End-to-End Automatic Number Plate Recognition (ANPR) System Pipeline.
"""

import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.plate_detector import PlateDetector
from src.plate_ocr import PlateOCR
from src.vehicle_logger import VehicleLogger

class ANPRPipeline:
    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.detector = PlateDetector()
        self.ocr = PlateOCR()
        self.logger = VehicleLogger(output_dir=self.output_dir)

    def draw_annotation(self, image_bgr, bbox, plate_text, confidence, latency_ms, status=None):
        """
        Draws a high-visibility, professional bounding box and HUD text badge.
        """
        annotated = image_bgr.copy()
        x, y, w, h = bbox

        # Main bounding box
        box_color = (0, 220, 50) if status != "MISMATCH" else (0, 80, 240)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 3)

        # Corner accents
        length = int(min(w, h) * 0.3)
        thick = 4
        # Top-left
        cv2.line(annotated, (x, y), (x + length, y), box_color, thick)
        cv2.line(annotated, (x, y), (x, y + length), box_color, thick)
        # Top-right
        cv2.line(annotated, (x + w, y), (x + w - length, y), box_color, thick)
        cv2.line(annotated, (x + w, y), (x + w, y + length), box_color, thick)
        # Bottom-left
        cv2.line(annotated, (x, y + h), (x + length, y + h), box_color, thick)
        cv2.line(annotated, (x, y + h), (x, y + h - length), box_color, thick)
        # Bottom-right
        cv2.line(annotated, (x + w, y + h), (x + w - length, y + h), box_color, thick)
        cv2.line(annotated, (x + w, y + h), (x + w, y + h - length), box_color, thick)

        # Label Badge background above plate
        label = f"{plate_text}  [{int(confidence * 100)}%]"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.75
        (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, 2)

        badge_y1 = max(0, y - lh - 18)
        badge_y2 = y - 2
        badge_x1 = max(0, x)
        badge_x2 = min(annotated.shape[1], x + lw + 24)

        # Badge rectangle
        cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (20, 22, 25), -1)
        cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), box_color, 2)

        # Badge text
        cv2.putText(annotated, label, (badge_x1 + 10, badge_y2 - 8), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

        # HUD Watermark in top left corner
        hud_bg = (15, 18, 22)
        cv2.rectangle(annotated, (15, 15), (380, 75), hud_bg, -1)
        cv2.rectangle(annotated, (15, 15), (380, 75), (0, 180, 240), 1)

        cv2.putText(annotated, "ANPR REAL-TIME INFERENCE", (25, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 1, cv2.LINE_AA)
        status_txt = f"Latency: {latency_ms:.1f}ms | Status: {status or 'DETECTED'}"
        cv2.putText(annotated, status_txt, (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

        return annotated

    def save_pipeline_stages_plot(self, original_bgr, stages, plate_crop, ocr_res, save_path):
        """
        Generates a 6-panel composite scientific figure detailing all processing phases.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
        fig.patch.set_facecolor('#0d1b2a')

        titles = [
            "(1) Original Vehicle Scene",
            "(2) Contrast Normalization (CLAHE)",
            "(3) Bilateral Edge-Preserving Filter",
            "(4) Canny Edges & Morphological Closing",
            "(5) Bounding Box Localization",
            f"(6) Segmented Plate & OCR: '{ocr_res['formatted_text']}'"
        ]

        images = [
            cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB),
            stages.get("clahe", original_bgr),
            stages.get("bilateral", original_bgr),
            stages.get("morphed", original_bgr),
            cv2.cvtColor(stages.get("contours_overlay", original_bgr), cv2.COLOR_BGR2RGB),
            ocr_res.get("binary_image", plate_crop)
        ]

        cmaps = [None, 'gray', 'gray', 'gray', None, 'gray']

        for ax, title, img, cmap in zip(axes.flat, titles, images, cmaps):
            ax.set_facecolor('#1b263b')
            ax.set_title(title, color='#e0e1dd', fontsize=11, fontweight='bold', pad=10)
            if cmap:
                ax.imshow(img, cmap=cmap)
            else:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#415a77')
                spine.set_linewidth(1.5)

        plt.suptitle("AUTOMATIC NUMBER PLATE RECOGNITION (ANPR) - MULTI-STAGE PIPELINE",
                     color='#00b4d8', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)

    def process(self, image_input, filename="sample.jpg", ground_truth=None,
                vehicle_type="Unknown", condition="Normal", save_visuals=True):
        """
        Executes end-to-end recognition on an image.
        Args:
            image_input: path (str) or numpy BGR array
        """
        if isinstance(image_input, str):
            filename = os.path.basename(image_input)
            image_bgr = cv2.imread(image_input)
            if image_bgr is None:
                raise ValueError(f"Could not load image at {image_input}")
        else:
            image_bgr = image_input

        t_start = time.perf_counter()

        # Step 1: Detect Plate
        t_det_start = time.perf_counter()
        plate_crop, bbox, stages = self.detector.detect(image_bgr, filename=filename)
        t_det = (time.perf_counter() - t_det_start) * 1000.0

        # Step 2: OCR Recognition
        t_ocr_start = time.perf_counter()
        ocr_result = self.ocr.recognize(plate_crop)
        t_ocr = (time.perf_counter() - t_ocr_start) * 1000.0

        t_total = (time.perf_counter() - t_start) * 1000.0

        # Step 3: Logging & Status Evaluation
        status = self.logger.log_entry(
            image_file=filename,
            detected_plate=ocr_result["formatted_text"],
            ground_truth=ground_truth,
            latency_ms=t_total,
            confidence=ocr_result["confidence"],
            vehicle_type=vehicle_type,
            condition=condition
        )

        # Step 4: Annotation
        annotated = self.draw_annotation(
            image_bgr=image_bgr,
            bbox=bbox,
            plate_text=ocr_result["formatted_text"],
            confidence=ocr_result["confidence"],
            latency_ms=t_total,
            status=status
        )

        annotated_path = None
        stages_plot_path = None

        if save_visuals:
            annotated_path = os.path.join(self.output_dir, f"detected_{filename}")
            cv2.imwrite(annotated_path, annotated)

            stages_plot_path = os.path.join(self.output_dir, f"pipeline_stages_{filename}.png")
            self.save_pipeline_stages_plot(image_bgr, stages, plate_crop, ocr_result, stages_plot_path)

        return {
            "filename": filename,
            "detected_text": ocr_result["formatted_text"],
            "clean_text": ocr_result["clean_text"],
            "confidence": ocr_result["confidence"],
            "bbox": bbox,
            "status": status,
            "latency": {
                "detection_ms": round(t_det, 2),
                "ocr_ms": round(t_ocr, 2),
                "total_ms": round(t_total, 2)
            },
            "annotated_image": annotated,
            "plate_crop": plate_crop,
            "annotated_path": annotated_path,
            "stages_plot_path": stages_plot_path
        }
