"""
=============================================================================
           AUTOMATIC NUMBER PLATE RECOGNITION (ANPR) SYSTEM
              Image Processing & Computer Vision Project
=============================================================================
Main executable entrypoint.
Can process:
  - Kaggle Indian License Plate Dataset images
  - Custom vehicle images via --image
  - Benchmark evaluation via --evaluate
"""

import os
import sys
import argparse
import json
import time
import cv2

# Ensure src is in Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from src.pipeline import ANPRPipeline
from src.plate_detector import PlateDetector
from src.plate_ocr import PlateOCR
import evaluate

def print_banner():
    banner = """
================================================================================
          AUTOMATIC NUMBER PLATE RECOGNITION (ANPR) SYSTEM
             Computer Vision & Image Processing Project
================================================================================
 [Dataset]   : Kaggle Indian License Plate Dataset (1,696 Vehicles, 35 States)
 [Pipeline]  : CLAHE -> Bilateral Filter -> Canny Edges -> Contours -> OCR
 [OCR Engine]: Tesseract OCR v5.4 + Indian Vehicle Format Syntax Validator
 [Output Dir]: results/ (Logs, Detection Grids, Pipeline Stage Figures)
================================================================================
"""
    print(banner)

def run_sample_kaggle_tests(pipeline, num_samples=6):
    manifest_path = os.path.join(CURRENT_DIR, "dataset", "kaggle_benchmark_sample.json")
    if not os.path.exists(manifest_path):
        manifest_path = os.path.join(CURRENT_DIR, "dataset", "kaggle_manifest.json")

    if not os.path.exists(manifest_path):
        print("[-] Dataset manifest not found. Please run: python parse_kaggle_dataset.py")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    samples = samples[:num_samples]
    print(f"[*] Processing {len(samples)} real vehicles from Kaggle dataset...\n")

    for i, s in enumerate(samples, start=1):
        img_path = s["abs_path"]
        gt_plate = s.get("plate_text_clean", "N/A")
        state = s.get("state", "India")
        filename = s.get("filename", os.path.basename(img_path))

        print(f"--- [Vehicle {i}/{len(samples)}] {filename} (State: {state}) ---")
        try:
            res = pipeline.process(
                image_input=img_path,
                filename=filename,
                ground_truth=gt_plate,
                vehicle_type=f"Real Vehicle ({state})",
                condition="Real-World (Kaggle)",
                save_visuals=True
            )

            print(f"  > Detected Plate : {res['detected_text']}")
            print(f"  > Ground Truth   : {gt_plate}")
            print(f"  > Match Status   : {res['status']}")
            print(f"  > Confidence     : {int(res['confidence'] * 100)}%")
            print(f"  > Latency        : {res['latency']['total_ms']} ms (Det: {res['latency']['detection_ms']}ms, OCR: {res['latency']['ocr_ms']}ms)")
            print(f"  > Saved Visual   : {res['annotated_path']}\n")
        except Exception as e:
            print(f"  [-] Error processing {filename}: {e}\n")

def run_single_image(pipeline, image_path):
    if not os.path.exists(image_path):
        print(f"[-] File not found: {image_path}")
        return

    # Check if this image has ground truth in the Kaggle manifest
    gt_plate = None
    vehicle_type = "Single Image"
    condition = "Real-World"
    try:
        manifest_path = os.path.join(CURRENT_DIR, "dataset", "kaggle_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_items = json.load(f)
            fname = os.path.basename(image_path)
            for item in manifest_items:
                if item.get("filename") == fname or os.path.basename(item.get("rel_path", "")) == fname:
                    gt_plate = item.get("plate_text_clean")
                    vehicle_type = f"{item.get('subset', 'Real Vehicle')} ({item.get('state', 'India')})"
                    condition = "Kaggle Indian License Plate Dataset"
                    break
    except Exception:
        pass

    print(f"[*] Processing single image: {image_path}...")
    res = pipeline.process(
        image_input=image_path,
        filename=os.path.basename(image_path),
        ground_truth=gt_plate,
        vehicle_type=vehicle_type,
        condition=condition,
        save_visuals=True
    )

    print("\n" + "=" * 50)
    print(f" DETECTED NUMBER PLATE : {res['detected_text']}")
    if gt_plate:
        print(f" Ground Truth Plate    : {gt_plate}")
        print(f" Match Status          : {res['status']}")
    print(f" Confidence Score      : {int(res['confidence'] * 100)}%")
    print(f" Total Inference Time  : {res['latency']['total_ms']} ms")
    print(f" Annotated Result Image: {res['annotated_path']}")
    print(f" Pipeline Stages Figure: {res['stages_plot_path']}")
    print("=" * 50 + "\n")

def main():
    print_banner()

    parser = argparse.ArgumentParser(description="Automatic Number Plate Recognition (ANPR)")
    parser.add_argument("--image", type=str, help="Path to custom vehicle image to process")
    parser.add_argument("--gui", action="store_true", help="Launch interactive graphical desktop application")
    parser.add_argument("--evaluate", nargs="?", const=18, type=int, help="Run evaluation benchmark on N Kaggle images (default: 18)")
    parser.add_argument("--all", action="store_true", help="Run sample recognition and full benchmark evaluation")
    args = parser.parse_args()

    if args.gui:
        import gui
        gui.main()
        return

    pipeline = ANPRPipeline()

    if args.image:
        run_single_image(pipeline, args.image)
    elif args.evaluate:
        evaluate.run_evaluation(sample_limit=args.evaluate)
    elif args.all:
        run_sample_kaggle_tests(pipeline)
        print("\n" + "=" * 70)
        evaluate.run_evaluation(sample_limit=18)
    else:
        # Default run: process representative real Kaggle vehicles, then run evaluation
        run_sample_kaggle_tests(pipeline, num_samples=6)
        print("\n[*] Running Benchmark Evaluation on real Kaggle dataset...")
        evaluate.run_evaluation(sample_limit=18)

    print("\n[+] Done! All results, logs, and visual figures are saved in the 'results/' folder:")
    print("    - results/vehicle_log.csv       (Vehicle access & recognition audit log)")
    print("    - results/evaluation_metrics.json (Accuracy and latency summary)")
    print("    - results/metrics_table.csv      (Performance benchmark table)")
    print("    - results/kaggle_results_grid.png (Montage of detected vehicles)")
    print("    - results/accuracy_breakdown.png (Performance metrics bar chart)")

if __name__ == "__main__":
    main()
