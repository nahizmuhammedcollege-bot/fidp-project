"""
Parse and index the Kaggle Indian License Plate Dataset from archive.zip.
Extracts annotations from Pascal VOC XML files across:
  - State-wise_OLX (35 Indian States)
  - video_images (Traffic & Surveillance Cameras)
  - google_images (Web / Real-world photos)
"""

import os
import json
import csv
import xml.etree.ElementTree as ET

DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "kaggle_data")

def parse_all():
    records = []
    print("[*] Parsing Kaggle dataset XML annotations...")

    for root, dirs, files in os.walk(DATASET_ROOT):
        for f in files:
            if f.lower().endswith(".xml"):
                xml_path = os.path.join(root, f)
                jpg_name = os.path.splitext(f)[0] + ".jpg"
                jpg_path = os.path.join(root, jpg_name)

                # Try png/jpeg if jpg doesn't exist
                if not os.path.exists(jpg_path):
                    for ext in [".jpeg", ".png", ".JPG"]:
                        cand = os.path.join(root, os.path.splitext(f)[0] + ext)
                        if os.path.exists(cand):
                            jpg_path = cand
                            break

                if not os.path.exists(jpg_path):
                    continue

                # Parse XML
                try:
                    tree = ET.parse(xml_path)
                    root_elem = tree.getroot()

                    size_elem = root_elem.find("size")
                    width = int(size_elem.find("width").text) if size_elem is not None and size_elem.find("width") is not None else 0
                    height = int(size_elem.find("height").text) if size_elem is not None and size_elem.find("height") is not None else 0

                    rel_path = os.path.relpath(jpg_path, DATASET_ROOT).replace("\\", "/")
                    parts = rel_path.split("/")
                    subset = parts[0]
                    state = parts[1] if subset == "State-wise_OLX" and len(parts) >= 2 else "All"

                    for obj in root_elem.findall("object"):
                        name_elem = obj.find("name")
                        plate_str = name_elem.text.strip() if name_elem is not None and name_elem.text else ""

                        bnd = obj.find("bndbox")
                        if bnd is not None:
                            xmin = int(float(bnd.find("xmin").text))
                            ymin = int(float(bnd.find("ymin").text))
                            xmax = int(float(bnd.find("xmax").text))
                            ymax = int(float(bnd.find("ymax").text))
                        else:
                            xmin, ymin, xmax, ymax = 0, 0, 0, 0

                        # Normalize plate text (remove dashes, spaces)
                        clean_plate = plate_str.replace(" ", "").replace("-", "").replace(".", "").upper()

                        records.append({
                            "filename": os.path.basename(jpg_path),
                            "rel_path": rel_path,
                            "abs_path": jpg_path,
                            "subset": subset,
                            "state": state,
                            "plate_text_raw": plate_str,
                            "plate_text_clean": clean_plate,
                            "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                            "width": width,
                            "height": height
                        })
                except Exception as e:
                    continue

    print(f"[OK] Successfully indexed {len(records)} annotated vehicles from Kaggle dataset!")

    # Save complete manifest
    out_json = os.path.join(os.path.dirname(DATASET_ROOT), "kaggle_manifest.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    # Save summary CSV
    out_csv = os.path.join(os.path.dirname(DATASET_ROOT), "kaggle_manifest.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Subset", "State", "Plate_Text", "Clean_Plate", "BBox_X", "BBox_Y", "BBox_W", "BBox_H", "Img_W", "Img_H"])
        for r in records:
            writer.writerow([
                r["filename"], r["subset"], r["state"], r["plate_text_raw"],
                r["plate_text_clean"], r["bbox"][0], r["bbox"][1], r["bbox"][2], r["bbox"][3],
                r["width"], r["height"]
            ])

    print(f"[+] Saved manifest to {out_json}")
    print(f"[+] Saved CSV to {out_csv}")

    # Build a curated benchmark test set (e.g. 15-20 diverse high-quality real samples across states and subsets)
    # Pick representative samples from MH, DL, KA, KL, TN, AP, GJ, HR, UP, video_images, google_images
    curated = []
    seen_states = set()
    for r in records:
        if r["subset"] == "State-wise_OLX":
            st = r["state"]
            if st not in seen_states and len(r["plate_text_clean"]) >= 8:
                curated.append(r)
                seen_states.add(st)
                if len(curated) >= 12:
                    break

    # Add 2 video frames and 2 google images
    for r in records:
        if r["subset"] == "video_images" and len(r["plate_text_clean"]) >= 8:
            curated.append(r)
            if len([x for x in curated if x["subset"] == "video_images"]) >= 3:
                break

    for r in records:
        if r["subset"] == "google_images" and len(r["plate_text_clean"]) >= 8:
            curated.append(r)
            if len([x for x in curated if x["subset"] == "google_images"]) >= 3:
                break

    curated_path = os.path.join(os.path.dirname(DATASET_ROOT), "kaggle_benchmark_sample.json")
    with open(curated_path, "w", encoding="utf-8") as f:
        json.dump(curated, f, indent=2)

    print(f"[+] Created curated evaluation set with {len(curated)} real vehicles in {curated_path}")

if __name__ == "__main__":
    parse_all()
