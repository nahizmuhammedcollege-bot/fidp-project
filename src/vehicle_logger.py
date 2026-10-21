"""
Module: vehicle_logger.py
Real-Time Vehicle Access and Detection Logger (CSV & JSON).
"""

import os
import csv
import json
from datetime import datetime

class VehicleLogger:
    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.csv_path = os.path.join(self.output_dir, "vehicle_log.csv")
        self.json_path = os.path.join(self.output_dir, "vehicle_log.json")

        self.headers = [
            "Timestamp", "Image_File", "Detected_Plate", "Ground_Truth",
            "Match_Status", "Confidence", "Latency_ms", "Vehicle_Type", "Condition"
        ]
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_entry(self, image_file, detected_plate, ground_truth=None, latency_ms=0.0,
                  confidence=0.0, vehicle_type="Unknown", condition="Normal"):
        """
        Logs a single detection event.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        status = "UNVERIFIED"
        if ground_truth:
            clean_det = detected_plate.replace(" ", "").upper()
            clean_gt = ground_truth.replace(" ", "").upper()
            if clean_det == clean_gt:
                status = "EXACT MATCH"
            elif clean_det in clean_gt or clean_gt in clean_det:
                status = "PARTIAL MATCH"
            else:
                status = "MISMATCH"

        row = [
            timestamp,
            image_file,
            detected_plate,
            ground_truth or "N/A",
            status,
            f"{confidence:.2f}",
            f"{latency_ms:.1f}",
            vehicle_type,
            condition
        ]

        # Append to CSV
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Update JSON history
        entries = []
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    entries = json.load(f)
            except Exception:
                entries = []

        entries.append({
            "timestamp": timestamp,
            "image_file": image_file,
            "detected_plate": detected_plate,
            "ground_truth": ground_truth,
            "status": status,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "vehicle_type": vehicle_type,
            "condition": condition
        })

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)

        return status
