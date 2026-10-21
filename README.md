# Automatic Number Plate Recognition (ANPR) System

> **Computer Vision & Image Processing Project**  
> Real-Time Vehicle License Plate Localization & Recognition on Indian Vehicles.

---

## 🚀 Quick Start

### 1. Interactive Desktop Graphical Interface (GUI)
Run the desktop GUI application:
```bash
python gui.py
```
*(or via `python main.py --gui`)*

#### GUI Features:
- **Image File Selector**: Click `📁 Select Vehicle Image` to choose any car image (`.jpg`, `.jpeg`, `.png`, `.bmp`) from your system.
- **Kaggle Sample Showcase**: Click `🎲 Next Kaggle Sample` to instantly cycle through real vehicles from the Kaggle dataset.
- **Side-by-Side View**: Shows the input vehicle scene with a green/cyan HUD localization bounding box overlay.
- **Embossed Indian Plate Display**: Renders an authentic Indian High-Security Registration Plate (HSRP) graphic with the blue "IND" hologram.
- **Telemetry Dashboard**:
  - Recognized Plate Number (e.g. `TN 21 AT 0480`)
  - Registered State Decoder (e.g. `Tamil Nadu (TN)`, `Delhi (DL)`, etc.)
  - Match Status & Ground Truth Comparison (`EXACT MATCH`, `PARTIAL MATCH`)
  - Confidence Score Meter
  - Latency breakdown (Detection ms, OCR ms, Total ms)
- **Deep Inspection Tools**:
  - `🔬 View 6-Stage Pipeline Figure`: Opens the scientific 6-panel composite image detailing each CV stage.
  - `📋 Open Vehicle Audit Log`: Opens `results/vehicle_log.csv` containing complete timestamped records.

---

### 2. Command Line Interface (CLI)

#### Process a Single Vehicle Image:
```bash
python main.py --image "C:\Users\user\Downloads\nahiz\dataset\kaggle_data\google_images\0a720df9-e4ef-4e44-8c13-b39b9be8444d___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_Nissan-Terrano-6.jpg"
```

#### Run Sample Tests on Real Kaggle Vehicles:
```bash
python main.py
```

#### Run Quantitative Evaluation Benchmark:
```bash
python evaluate.py
```

---

## 🛠️ Multi-Stage Pipeline Architecture

```
[Input Scene] 
     │
     ▼
[1. Contrast Equalization (CLAHE)]
     │
     ▼
[2. Edge-Preserving Bilateral Filtering & Sobel-X Gradient]
     │
     ▼
[3. Morphological Rectangular Closing & MSER Extremal Regions]
     │
     ▼
[4. Plate Localization & Watermark/Logo Rejection]
     │
     ▼
[5. Multi-Pass Tesseract OCR (PSM-8 Single Word / PSM-7 Single Line)]
     │
     ▼
[6. Indian HSRP Syntax Correction & State Code Anchor Extraction]
     │
     ▼
[HUD Annotated Image + Audit Log + Telemetry]
```

---

## 📊 Dataset & Benchmark Results

- **Dataset**: Kaggle Indian License Plate Dataset (1,696 real vehicles across 35 Indian States & Union Territories).
- **Plate Localization Accuracy**: **83.3%**
- **Character-Level Accuracy**: **67.2%**
- **Top-1 / Partial Match Accuracy**: **61.1%**
- **Plate Exact Match Accuracy**: **22.2%**
- **Output Artifacts**:
  - `results/vehicle_log.csv`: Vehicle audit trail with timestamps, latency, and confidence.
  - `results/evaluation_metrics.json`: JSON performance metrics summary.
  - `results/kaggle_results_grid.png`: Visual montage grid of detected vehicles.
  - `results/accuracy_breakdown.png`: Benchmark chart comparing localization, character, and plate accuracy.
  - `Final_Presentation_ANPR.pptx`: 13-slide condensed final project presentation.
