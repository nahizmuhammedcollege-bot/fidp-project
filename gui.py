"""
=============================================================================
         AUTOMATIC NUMBER PLATE RECOGNITION (ANPR) SYSTEM - GUI
              Image Processing & Computer Vision Final Project
=============================================================================
Modern Desktop Application for Real-Time License Plate Localization & Recognition.
Features:
  - Interactive Image File Selector (.jpg, .jpeg, .png, .bmp)
  - Quick-Load Showcase for Kaggle Indian License Plate Dataset samples
  - Side-by-Side Vehicle HUD Preview & Segmented Plate Inspection
  - Realistic Indian HSRP Plate Graphic Rendering
  - State Decoder (decodes 35+ Indian States & Union Territories)
  - Real-Time Inference Telemetry (Detection time, OCR time, Confidence Gauge)
  - One-Click 6-Stage Scientific Pipeline Inspection Plot Viewer
  - Vehicle Audit Log Quick Access
"""

import os
import sys
import json
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk

# Enable Windows High DPI scaling for crisp rendering
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# Ensure project root is in sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.pipeline import ANPRPipeline

# ─── Color Palette & Design Tokens ───────────────────────────────────────────
BG_DARK = "#0b1120"        # Deep slate background
CARD_BG = "#151f32"        # Panel card surface
CARD_BORDER = "#25334d"    # Panel border
ACCENT_BLUE = "#0284c7"    # Tech blue primary
ACCENT_CYAN = "#06b6d4"    # Highlighting cyan
ACCENT_GREEN = "#10b981"   # Success green
ACCENT_AMBER = "#f59e0b"   # Warning amber
TEXT_MAIN = "#f8fafc"      # Pure white / light slate
TEXT_MUTED = "#94a3b8"     # Secondary labels
PLATE_YELLOW = "#fef08a"   # Indian commercial plate yellow
PLATE_WHITE = "#ffffff"    # Indian private plate white

INDIAN_STATE_NAMES = {
    "AN": "Andaman & Nicobar",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Daman & Diu",
    "DL": "Delhi (NCT)",
    "DN": "Dadra & Nagar Haveli",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HP": "Himachal Pradesh",
    "HR": "Haryana",
    "JH": "Jharkhand",
    "JK": "Jammu & Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "TS": "Telangana",
    "UK": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
}


class ANPRApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("ANPR - Automatic Number Plate Recognition System")
        self.geometry("1240x820")
        self.minsize(1080, 720)
        self.configure(bg=BG_DARK)

        # Application state
        self.pipeline = ANPRPipeline()
        self.current_image_path = None
        self.current_result = None
        self.is_processing = False

        # Load Kaggle manifest for quick-samples
        self.manifest_samples = self._load_manifest_samples()
        self.sample_index = 0

        # Build UI layout
        self._build_header()
        self._build_main_layout()
        self._build_status_bar()

        # Try auto-loading a default sample image if present
        self._load_initial_sample()

    # ─── Manifest & Sample Helpers ───────────────────────────────────────────
    def _load_manifest_samples(self):
        manifest_path = os.path.join(PROJECT_DIR, "dataset", "kaggle_manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return [d for d in data if os.path.exists(d.get("abs_path", ""))]
            except Exception:
                pass
        return []

    def _load_initial_sample(self):
        # Look for the Nissan Terrano image or first manifest sample
        sample_path = os.path.join(
            PROJECT_DIR, "dataset", "kaggle_data", "google_images",
            "0a720df9-e4ef-4e44-8c13-b39b9be8444d___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_Nissan-Terrano-6.jpg"
        )
        if os.path.exists(sample_path):
            self.load_image(sample_path, auto_process=True)
        elif self.manifest_samples:
            self.load_image(self.manifest_samples[0]["abs_path"], auto_process=True)

    # ─── UI Component Builders ───────────────────────────────────────────────
    def _build_header(self):
        header = tk.Frame(self, bg=CARD_BG, height=72, highlightbackground=CARD_BORDER, highlightthickness=1)
        header.pack(side=tk.TOP, fill=tk.X)

        # Title & Subtitle container
        title_box = tk.Frame(header, bg=CARD_BG)
        title_box.pack(side=tk.LEFT, padx=24, pady=10)

        title_lbl = tk.Label(
            title_box,
            text="AUTOMATIC NUMBER PLATE RECOGNITION",
            font=("Segoe UI", 15, "bold"),
            fg=TEXT_MAIN,
            bg=CARD_BG,
        )
        title_lbl.pack(anchor="w")

        sub_lbl = tk.Label(
            title_box,
            text="Edge-Preserving Filtering · Morphological Localization · Multi-Pass Tesseract OCR · Indian HSRP Validator",
            font=("Segoe UI", 9),
            fg=ACCENT_CYAN,
            bg=CARD_BG,
        )
        sub_lbl.pack(anchor="w")

        # Header Action Buttons
        btn_box = tk.Frame(header, bg=CARD_BG)
        btn_box.pack(side=tk.RIGHT, padx=20, pady=12)

        self.btn_browse = self._create_button(
            btn_box,
            text="📁 Select Vehicle Image",
            bg=ACCENT_BLUE,
            command=self.on_browse_file,
            width=20
        )
        self.btn_browse.pack(side=tk.LEFT, padx=6)

        if self.manifest_samples:
            self.btn_sample = self._create_button(
                btn_box,
                text="🎲 Next Kaggle Sample",
                bg="#334155",
                command=self.on_next_sample,
                width=19
            )
            self.btn_sample.pack(side=tk.LEFT, padx=6)

        self.btn_run = self._create_button(
            btn_box,
            text="⚡ Run Recognition",
            bg=ACCENT_GREEN,
            command=self.on_run_inference,
            width=18
        )
        self.btn_run.pack(side=tk.LEFT, padx=6)

    def _create_button(self, parent, text, bg, command, width=15):
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 10, "bold"),
            bg=bg,
            fg="#ffffff",
            activebackground=ACCENT_CYAN,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            bd=0,
            padx=14,
            pady=8,
            cursor="hand2",
            width=width
        )
        # Subtle hover effect
        btn.bind("<Enter>", lambda e, b=btn, c=bg: b.configure(bg=self._lighten_color(c)))
        btn.bind("<Leave>", lambda e, b=btn, c=bg: b.configure(bg=c))
        return btn

    def _lighten_color(self, hex_color):
        try:
            hex_color = hex_color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            r = min(255, int(r * 1.15) + 15)
            g = min(255, int(g * 1.15) + 15)
            b = min(255, int(b * 1.15) + 15)
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return hex_color

    def _build_main_layout(self):
        main_frame = tk.Frame(self, bg=BG_DARK)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=16, pady=14)

        # ── Left: Image Preview & Detection Canvas ───────────────────────────
        left_card = tk.Frame(main_frame, bg=CARD_BG, highlightbackground=CARD_BORDER, highlightthickness=1)
        left_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        left_header = tk.Frame(left_card, bg=CARD_BG)
        left_header.pack(fill=tk.X, padx=16, pady=(12, 6))

        tk.Label(
            left_header,
            text="VEHICLE SCENE & LOCALIZATION OVERLAY",
            font=("Segoe UI", 11, "bold"),
            fg=TEXT_MAIN,
            bg=CARD_BG
        ).pack(side=tk.LEFT)

        self.lbl_image_info = tk.Label(
            left_header,
            text="No image loaded",
            font=("Segoe UI", 9),
            fg=TEXT_MUTED,
            bg=CARD_BG
        )
        self.lbl_image_info.pack(side=tk.RIGHT)

        # Canvas for image display
        self.canvas = tk.Canvas(left_card, bg="#090d16", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # ── Right: Telemetry & Plate Dashboard ───────────────────────────────
        right_card = tk.Frame(main_frame, bg=CARD_BG, width=440, highlightbackground=CARD_BORDER, highlightthickness=1)
        right_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(6, 0))
        right_card.pack_propagate(False)

        right_header = tk.Frame(right_card, bg=CARD_BG)
        right_header.pack(fill=tk.X, padx=16, pady=(12, 6))

        tk.Label(
            right_header,
            text="RECOGNITION DASHBOARD",
            font=("Segoe UI", 11, "bold"),
            fg=TEXT_MAIN,
            bg=CARD_BG
        ).pack(side=tk.LEFT)

        # ── Card 1: License Plate Graphic ────────────────────────────────────
        plate_card = tk.Frame(right_card, bg="#111827", highlightbackground=CARD_BORDER, highlightthickness=1)
        plate_card.pack(fill=tk.X, padx=16, pady=8)

        tk.Label(
            plate_card,
            text="EXTRACTED NUMBER PLATE",
            font=("Segoe UI", 8, "bold"),
            fg=ACCENT_CYAN,
            bg="#111827"
        ).pack(anchor="w", padx=12, pady=(10, 4))

        # Realistic Indian HSRP License Plate Canvas
        self.plate_canvas = tk.Canvas(plate_card, width=370, height=86, bg="#111827", highlightthickness=0)
        self.plate_canvas.pack(padx=12, pady=(2, 12))
        self._draw_plate_graphic("WAITING FOR IMAGE", is_placeholder=True)

        # ── Card 2: Recognition Telemetry Metrics ─────────────────────────────
        metrics_card = tk.Frame(right_card, bg="#111827", highlightbackground=CARD_BORDER, highlightthickness=1)
        metrics_card.pack(fill=tk.X, padx=16, pady=8)

        tk.Label(
            metrics_card,
            text="TELEMETRY & VERIFICATION",
            font=("Segoe UI", 8, "bold"),
            fg=ACCENT_CYAN,
            bg="#111827"
        ).pack(anchor="w", padx=12, pady=(10, 6))

        # Grid of metrics
        self.val_number = self._add_metric_row(metrics_card, "Detected Plate:", "—", val_fg="#38bdf8")
        self.val_state = self._add_metric_row(metrics_card, "Registered State:", "—", val_fg="#f8fafc")
        self.val_status = self._add_metric_row(metrics_card, "Match Status:", "—", val_fg=ACCENT_GREEN)
        self.val_conf = self._add_metric_row(metrics_card, "Confidence Score:", "—", val_fg="#fbbf24")
        self.val_time = self._add_metric_row(metrics_card, "Total Latency:", "—", val_fg="#f8fafc")
        self.val_split = self._add_metric_row(metrics_card, "Det / OCR Latency:", "—", val_fg=TEXT_MUTED)

        # Visual confidence meter
        meter_frame = tk.Frame(metrics_card, bg="#111827")
        meter_frame.pack(fill=tk.X, padx=12, pady=(8, 12))

        tk.Label(meter_frame, text="Confidence Level", font=("Segoe UI", 8), fg=TEXT_MUTED, bg="#111827").pack(anchor="w")
        self.conf_bar = ttk.Progressbar(meter_frame, orient="horizontal", length=350, mode="determinate")
        self.conf_bar.pack(fill=tk.X, pady=(2, 0))

        # ── Card 3: Deep Inspection & Scientific Actions ─────────────────────
        actions_card = tk.Frame(right_card, bg="#111827", highlightbackground=CARD_BORDER, highlightthickness=1)
        actions_card.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)

        tk.Label(
            actions_card,
            text="SCIENTIFIC INSPECTION & LOGS",
            font=("Segoe UI", 8, "bold"),
            fg=ACCENT_CYAN,
            bg="#111827"
        ).pack(anchor="w", padx=12, pady=(10, 8))

        self.btn_stages = self._create_button(
            actions_card,
            text="🔬 View 6-Stage Pipeline Figure",
            bg="#1e293b",
            command=self.on_view_stages,
            width=28
        )
        self.btn_stages.pack(fill=tk.X, padx=12, pady=4)

        self.btn_logs = self._create_button(
            actions_card,
            text="📋 Open Vehicle Audit Log (CSV)",
            bg="#1e293b",
            command=self.on_open_logs,
            width=28
        )
        self.btn_logs.pack(fill=tk.X, padx=12, pady=4)

        self.btn_results_folder = self._create_button(
            actions_card,
            text="📂 Open Results Directory",
            bg="#1e293b",
            command=self.on_open_results_folder,
            width=28
        )
        self.btn_results_folder.pack(fill=tk.X, padx=12, pady=4)

    def _add_metric_row(self, parent, label_text, default_val="—", val_fg="#f8fafc"):
        row = tk.Frame(parent, bg="#111827")
        row.pack(fill=tk.X, padx=12, pady=3)

        lbl = tk.Label(row, text=label_text, font=("Segoe UI", 9), fg=TEXT_MUTED, bg="#111827")
        lbl.pack(side=tk.LEFT)

        val = tk.Label(row, text=default_val, font=("Segoe UI", 9, "bold"), fg=val_fg, bg="#111827")
        val.pack(side=tk.RIGHT)
        return val

    def _build_status_bar(self):
        status_frame = tk.Frame(self, bg="#060911", height=30)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_status = tk.Label(
            status_frame,
            text="Ready. Select an image or click 'Next Kaggle Sample' to test real vehicles.",
            font=("Segoe UI", 9),
            fg=TEXT_MUTED,
            bg="#060911"
        )
        self.lbl_status.pack(side=tk.LEFT, padx=16, pady=4)

        self.lbl_dataset_info = tk.Label(
            status_frame,
            text=f"Kaggle Dataset: {len(self.manifest_samples)} Indexed Vehicles",
            font=("Segoe UI", 9),
            fg=ACCENT_CYAN,
            bg="#060911"
        )
        self.lbl_dataset_info.pack(side=tk.RIGHT, padx=16, pady=4)

    # ─── License Plate Graphic Renderer ──────────────────────────────────────
    def _draw_plate_graphic(self, text, is_placeholder=False):
        c = self.plate_canvas
        c.delete("all")
        w, h = 370, 84

        if is_placeholder:
            # Subtle placeholder box
            c.create_rectangle(6, 6, w - 6, h - 6, fill="#1f2937", outline="#374151", width=2)
            c.create_text(w / 2, h / 2, text=text, font=("Segoe UI", 11, "bold"), fill="#6b7280")
            return

        # Authentic Indian HSRP License Plate:
        # White background with black rounded rectangle border
        bg_color = "#ffffff"
        c.create_rectangle(6, 6, w - 6, h - 6, fill=bg_color, outline="#111827", width=4)
        c.create_rectangle(10, 10, w - 10, h - 10, fill=bg_color, outline="#111827", width=1)

        # Left blue strip: IND Hologram Section
        c.create_rectangle(11, 11, 52, h - 10, fill="#1d4ed8", outline="#1d4ed8")
        # Ashoka Chakra circle simulation
        c.create_oval(26, 20, 38, 32, outline="#facc15", width=2)
        c.create_text(32, 54, text="IND", font=("Arial", 9, "bold"), fill="#ffffff")

        # Main Embossed License Plate Text
        font_size = 20 if len(text) <= 12 else 17
        c.create_text(
            (w + 52) / 2, h / 2,
            text=text,
            font=("Consolas", font_size, "bold"),
            fill="#09090b"
        )

    # ─── Event Handlers ──────────────────────────────────────────────────────
    def on_browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Vehicle Image",
            initialdir=os.path.join(PROJECT_DIR, "dataset"),
            filetypes=[
                ("Vehicle Images", "*.jpg *.jpeg *.png *.bmp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.load_image(file_path, auto_process=True)

    def on_next_sample(self):
        if not self.manifest_samples:
            messagebox.showinfo("Dataset", "Kaggle manifest not loaded.")
            return

        sample = self.manifest_samples[self.sample_index]
        self.sample_index = (self.sample_index + 1) % len(self.manifest_samples)
        self.load_image(sample["abs_path"], auto_process=True)

    def load_image(self, file_path, auto_process=False):
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"Image file not found:\n{file_path}")
            return

        self.current_image_path = file_path
        fname = os.path.basename(file_path)
        self.lbl_image_info.configure(text=f"{fname} ({os.path.getsize(file_path) // 1024} KB)")

        # Render preview of loaded image immediately
        img_bgr = cv2.imread(file_path)
        if img_bgr is not None:
            self._display_cv_image(img_bgr)

        # Clear telemetry for fresh run
        self._clear_telemetry()
        self._set_status(f"Loaded: {fname}. Ready to recognize.")

        if auto_process:
            self.on_run_inference()

    def _clear_telemetry(self):
        self.val_number.configure(text="—", fg="#38bdf8")
        self.val_state.configure(text="—")
        self.val_status.configure(text="—", fg=TEXT_MUTED)
        self.val_conf.configure(text="—")
        self.val_time.configure(text="—")
        self.val_split.configure(text="—")
        self.conf_bar["value"] = 0
        self._draw_plate_graphic("INSPECTING...", is_placeholder=True)

    def on_run_inference(self):
        if not self.current_image_path or self.is_processing:
            return

        self.is_processing = True
        self.btn_run.configure(state=tk.DISABLED, bg="#475569")
        self.btn_browse.configure(state=tk.DISABLED)
        if hasattr(self, "btn_sample"):
            self.btn_sample.configure(state=tk.DISABLED)

        self._set_status("Processing: Localizing plate & running multi-pass OCR...")

        # Run in background thread to keep GUI responsive
        threading.Thread(target=self._inference_worker, daemon=True).start()

    def _inference_worker(self):
        img_path = self.current_image_path
        filename = os.path.basename(img_path)

        # Check Kaggle dataset manifest for ground truth
        gt_plate = None
        vehicle_type = "Single Vehicle"
        condition = "Real-World"
        manifest_path = os.path.join(PROJECT_DIR, "dataset", "kaggle_manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                for item in manifest:
                    if item.get("filename") == filename or os.path.basename(item.get("rel_path", "")) == filename:
                        gt_plate = item.get("plate_text_clean")
                        vehicle_type = f"{item.get('subset', 'Real')} ({item.get('state', 'India')})"
                        condition = "Kaggle Indian License Plate Dataset"
                        break
            except Exception:
                pass

        try:
            res = self.pipeline.process(
                image_input=img_path,
                filename=filename,
                ground_truth=gt_plate,
                vehicle_type=vehicle_type,
                condition=condition,
                save_visuals=True
            )
            self.current_result = res
            self.after(0, self._on_inference_complete, res, gt_plate)
        except Exception as e:
            self.after(0, self._on_inference_error, str(e))

    def _on_inference_complete(self, res, gt_plate):
        self.is_processing = False
        self.btn_run.configure(state=tk.NORMAL, bg=ACCENT_GREEN)
        self.btn_browse.configure(state=tk.NORMAL)
        if hasattr(self, "btn_sample"):
            self.btn_sample.configure(state=tk.NORMAL)

        # 1. Update Canvas with annotated image (with HUD & bounding box)
        if res.get("annotated_image") is not None:
            self._display_cv_image(res["annotated_image"])

        # 2. Update License Plate Graphic
        plate_text = res.get("detected_text", "UNKNOWN")
        self._draw_plate_graphic(plate_text, is_placeholder=False)

        # 3. Decode State Code
        clean_text = res.get("clean_text", "")
        state_prefix = clean_text[:2] if len(clean_text) >= 2 else ""
        state_name = INDIAN_STATE_NAMES.get(state_prefix, "Unknown / General")
        if state_prefix in INDIAN_STATE_NAMES:
            state_display = f"{state_name} ({state_prefix})"
        else:
            state_display = "Standard Format"

        # 4. Telemetry values
        self.val_number.configure(text=plate_text, fg="#38bdf8")
        self.val_state.configure(text=state_display)

        # Status badge
        status = res.get("status", "DETECTED")
        if status == "EXACT MATCH":
            status_color = ACCENT_GREEN
            status_text = f"✔ {status} ({gt_plate})"
        elif status == "PARTIAL MATCH":
            status_color = ACCENT_AMBER
            status_text = f"~ {status} (GT: {gt_plate})"
        elif gt_plate:
            status_color = "#ef4444"
            status_text = f"✖ {status} (GT: {gt_plate})"
        else:
            status_color = ACCENT_CYAN
            status_text = "✔ DETECTED & VALIDATED"

        self.val_status.configure(text=status_text, fg=status_color)

        conf = res.get("confidence", 0.0)
        self.val_conf.configure(text=f"{int(conf * 100)}%")
        self.conf_bar["value"] = int(conf * 100)

        latency = res.get("latency", {})
        total_ms = latency.get("total_ms", 0)
        det_ms = latency.get("detection_ms", 0)
        ocr_ms = latency.get("ocr_ms", 0)

        self.val_time.configure(text=f"{total_ms:.1f} ms")
        self.val_split.configure(text=f"Det: {det_ms:.1f} ms | OCR: {ocr_ms:.1f} ms")

        self._set_status(f"Recognition finished in {total_ms:.1f} ms. License Plate: '{plate_text}'.")

    def _on_inference_error(self, err_msg):
        self.is_processing = False
        self.btn_run.configure(state=tk.NORMAL, bg=ACCENT_GREEN)
        self.btn_browse.configure(state=tk.NORMAL)
        if hasattr(self, "btn_sample"):
            self.btn_sample.configure(state=tk.NORMAL)

        self._set_status(f"Error during recognition: {err_msg}")
        messagebox.showerror("Recognition Error", f"An error occurred:\n{err_msg}")

    # ─── Image Rendering on Canvas ───────────────────────────────────────────
    def _display_cv_image(self, cv_bgr):
        self._last_cv_image = cv_bgr
        self._render_current_canvas_image()

    def _on_canvas_resize(self, event):
        self._render_current_canvas_image()

    def _render_current_canvas_image(self):
        if not hasattr(self, "_last_cv_image") or self._last_cv_image is None:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 50 or ch < 50:
            return

        # Convert BGR to RGB
        rgb = cv2.cvtColor(self._last_cv_image, cv2.COLOR_BGR2RGB)
        ih, iw = rgb.shape[:2]

        # Calculate aspect ratio scaling
        scale = min(cw / float(iw), ch / float(ih))
        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))

        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        self._tk_img = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        # Center the image in the canvas
        cx = cw // 2
        cy = ch // 2
        self.canvas.create_image(cx, cy, anchor=tk.CENTER, image=self._tk_img)

    def _set_status(self, text):
        self.lbl_status.configure(text=text)

    # ─── Additional Action Handlers ──────────────────────────────────────────
    def on_view_stages(self):
        if self.current_result and self.current_result.get("stages_plot_path"):
            path = self.current_result["stages_plot_path"]
            if os.path.exists(path):
                self._open_file_externally(path)
                return
        messagebox.showinfo("Pipeline Stages", "Please run recognition on an image first to view the 6-stage figure.")

    def on_open_logs(self):
        log_path = os.path.join(PROJECT_DIR, "results", "vehicle_log.csv")
        if os.path.exists(log_path):
            self._open_file_externally(log_path)
        else:
            messagebox.showinfo("Audit Log", "Vehicle log has not been created yet. Run recognition on an image first.")

    def on_open_results_folder(self):
        results_dir = os.path.join(PROJECT_DIR, "results")
        os.makedirs(results_dir, exist_ok=True)
        self._open_file_externally(results_dir)

    def _open_file_externally(self, path):
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open {path}:\n{e}")


def main():
    app = ANPRApp()
    app.mainloop()


if __name__ == "__main__":
    main()
