"""
Generate a diverse, realistic dataset of vehicle images with Indian High-Security
Registration Plates (HSRP) for Automatic Number Plate Recognition (ANPR) testing.
"""

import os
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

# Select font
FONT_PATH = "C:/Windows/Fonts/arialbd.ttf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "C:/Windows/Fonts/arial.ttf"

def draw_hsrp_plate(plate_text, width=320, height=80, plate_type="private"):
    """
    Renders an authentic Indian High Security Registration Plate (HSRP).
    - Private: White background, black text
    - Commercial: Yellow background, black text
    """
    img = Image.new("RGB", (width, height), color=(255, 255, 255) if plate_type == "private" else (245, 200, 20))
    draw = ImageDraw.Draw(img)

    # Outer border
    draw.rectangle([(2, 2), (width - 3, height - 3)], outline=(20, 20, 20), width=3)
    draw.rectangle([(5, 5), (width - 6, height - 6)], outline=(70, 70, 70), width=1)

    # Blue IND strip on left side
    ind_width = 32
    draw.rectangle([(5, 5), (ind_width, height - 6)], fill=(0, 60, 160))
    
    # IND text and circular emblem
    try:
        small_font = ImageFont.truetype(FONT_PATH, 11)
    except:
        small_font = ImageFont.load_default()

    # Draw emblem circle
    draw.ellipse([(10, 14), (26, 30)], outline=(220, 220, 220), width=2)
    draw.ellipse([(16, 20), (20, 24)], fill=(220, 220, 220))
    draw.text((8, 40), "IND", fill=(255, 255, 255), font=small_font)

    # Main Registration Characters
    try:
        plate_font = ImageFont.truetype(FONT_PATH, 42)
    except:
        plate_font = ImageFont.load_default()

    # Calculate text positioning
    text_bbox = draw.textbbox((0, 0), plate_text, font=plate_font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    # Center text in the remaining plate area
    available_w = width - ind_width
    text_x = ind_width + (available_w - text_w) // 2 + 2
    text_y = (height - text_h) // 2 - 4

    draw.text((text_x, text_y), plate_text, fill=(15, 15, 15), font=plate_font)

    # Screws / mounting rivets
    screw_color = (130, 130, 130)
    draw.ellipse([(45, 10), (51, 16)], fill=screw_color)
    draw.ellipse([(width - 25, 10), (width - 19, 16)], fill=screw_color)
    draw.ellipse([(45, height - 16), (51, height - 10)], fill=screw_color)
    draw.ellipse([(width - 25, height - 16), (width - 19, height - 10)], fill=screw_color)

    return np.array(img)

def create_vehicle_scene(
    plate_text,
    plate_type="private",
    body_color=(200, 200, 200),
    condition="daylight",
    angle=0.0,
    vehicle_type="Sedan"
):
    """
    Synthesizes a realistic vehicle scene with bumper, lights, shadow, and plate.
    """
    width, height = 960, 640
    scene = np.zeros((height, width, 3), dtype=np.uint8)

    # Background / Road / Environment
    if condition == "night":
        scene[:400] = [15, 18, 25]     # Dark night sky / environment
        scene[400:] = [25, 25, 28]     # Asphalt road
    else:
        # Daylight gradient sky / scenery
        for y in range(400):
            factor = y / 400.0
            scene[y, :] = [int(190 - 40 * factor), int(170 - 30 * factor), int(150 - 20 * factor)]
        # Road
        scene[400:] = [65, 68, 72]

    # Road lane markings
    if condition != "night":
        cv2.line(scene, (200, 640), (380, 400), (220, 220, 220), 4)
        cv2.line(scene, (760, 640), (580, 400), (220, 220, 220), 4)

    # Vehicle shadow on the asphalt
    cv2.ellipse(scene, (480, 520), (360, 45), 0, 0, 360, (18, 18, 20), -1)

    # Vehicle Body (Curved chassis / bumper)
    chassis_pts = np.array([
        [180, 490], [170, 320], [210, 230], [290, 170],
        [670, 170], [750, 230], [790, 320], [780, 490],
        [720, 510], [480, 520], [240, 510]
    ], np.int32)
    cv2.fillPoly(scene, [chassis_pts], body_color)

    # Rear window / windshield
    window_pts = np.array([
        [300, 180], [660, 180], [720, 250], [240, 250]
    ], np.int32)
    win_color = (40, 45, 50) if condition != "night" else (15, 20, 25)
    cv2.fillPoly(scene, [window_pts], win_color)
    # Windshield tint reflection
    cv2.line(scene, (320, 190), (450, 245), (120, 130, 140), 2)

    # Trunk seam line
    cv2.line(scene, (210, 330), (750, 330), tuple(max(0, c - 45) for c in body_color), 3)

    # Taillights
    light_l = np.array([[190, 300], [270, 305], [260, 345], [185, 340]], np.int32)
    light_r = np.array([[690, 305], [770, 300], [775, 340], [700, 345]], np.int32)

    if condition == "night":
        # Glowing bright red taillights with halo
        cv2.fillPoly(scene, [light_l], (30, 30, 240))
        cv2.fillPoly(scene, [light_r], (30, 30, 240))
        # Halo glow
        overlay = scene.copy()
        cv2.circle(overlay, (225, 320), 70, (20, 20, 180), -1)
        cv2.circle(overlay, (735, 320), 70, (20, 20, 180), -1)
        cv2.addWeighted(overlay, 0.35, scene, 0.65, 0, scene)
    else:
        cv2.fillPoly(scene, [light_l], (35, 35, 190))
        cv2.fillPoly(scene, [light_r], (35, 35, 190))
        # Inner white reverse/turn lamp
        cv2.fillPoly(scene, [np.array([[240, 312], [265, 315], [258, 338], [236, 335]], np.int32)], (220, 220, 220))
        cv2.fillPoly(scene, [np.array([[695, 315], [720, 312], [724, 335], [702, 338]], np.int32)], (220, 220, 220))

    # Bumper contour & lower diffuser
    diffuser_pts = np.array([
        [270, 485], [690, 485], [670, 510], [290, 510]
    ], np.int32)
    cv2.fillPoly(scene, [diffuser_pts], (30, 30, 32))

    # Exhaust pipes
    cv2.circle(scene, (310, 502), 9, (80, 80, 85), -1)
    cv2.circle(scene, (310, 502), 6, (20, 20, 22), -1)
    cv2.circle(scene, (650, 502), 9, (80, 80, 85), -1)
    cv2.circle(scene, (650, 502), 6, (20, 20, 22), -1)

    # Brand emblem above plate
    cv2.circle(scene, (480, 310), 16, (210, 215, 220), 2)
    cv2.circle(scene, (480, 310), 10, (180, 185, 190), 2)

    # Plate Cavity / Recess
    cavity_x, cavity_y, cavity_w, cavity_h = 320, 365, 320, 96
    cv2.rectangle(scene, (cavity_x - 12, cavity_y - 10), (cavity_x + cavity_w + 12, cavity_y + cavity_h + 10), (35, 36, 40), -1)
    cv2.rectangle(scene, (cavity_x - 14, cavity_y - 12), (cavity_x + cavity_w + 14, cavity_y + cavity_h + 12), (60, 62, 65), 1)

    # Plate LED lights
    cv2.rectangle(scene, (cavity_x + 60, cavity_y - 8), (cavity_x + 90, cavity_y - 4), (220, 230, 240), -1)
    cv2.rectangle(scene, (cavity_x + cavity_w - 90, cavity_y - 8), (cavity_x + cavity_w - 60, cavity_y - 4), (220, 230, 240), -1)

    # Render Number Plate
    plate_img = draw_hsrp_plate(plate_text, width=cavity_w, height=cavity_h, plate_type=plate_type)

    # Apply Angle / Perspective Transformation if angled
    px, py = cavity_x, cavity_y
    if abs(angle) > 0.01:
        # Warp the plate slightly
        src_pts = np.float32([[0, 0], [cavity_w, 0], [cavity_w, cavity_h], [0, cavity_h]])
        # skew right side down and inwards
        offset_y = int(cavity_h * np.sin(np.radians(angle)))
        offset_x = int(cavity_w * 0.08)
        dst_pts = np.float32([
            [offset_x, 0],
            [cavity_w - offset_x, offset_y],
            [cavity_w, cavity_h + offset_y],
            [0, cavity_h]
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_plate = cv2.warpPerspective(plate_img, M, (cavity_w + 20, cavity_h + abs(offset_y) + 10), borderValue=(35, 36, 40))
        # Place into scene
        ph, pw = warped_plate.shape[:2]
        scene[py:py + ph, px:px + pw] = warped_plate
        bbox = [px, py, pw, ph]
    else:
        scene[py:py + cavity_h, px:px + cavity_w] = plate_img
        bbox = [px, py, cavity_w, cavity_h]

    # Environmental Adjustments
    if condition == "night":
        # Plate illumination glow
        glow = scene.copy()
        cv2.ellipse(glow, (480, 410), (180, 60), 0, 0, 360, (230, 235, 245), -1)
        cv2.addWeighted(glow, 0.25, scene, 0.75, 0, scene)
        # Global darkening
        scene = cv2.convertScaleAbs(scene, alpha=0.6, beta=-15)
    elif condition == "shadow":
        # Deep shadow across half the vehicle
        mask = np.zeros((height, width), dtype=np.float32)
        cv2.rectangle(mask, (0, 0), (520, height), 0.55, -1)
        cv2.rectangle(mask, (520, 0), (width, height), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        for c in range(3):
            scene[:, :, c] = np.clip(scene[:, :, c] * mask, 0, 255).astype(np.uint8)
    elif condition == "rain":
        # Rain streaks & surface reflections
        noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        scene = cv2.add(scene, noise)
        # Wet ground blur
        scene[420:] = cv2.GaussianBlur(scene[420:], (5, 5), 0)

    return scene, bbox

def generate_all():
    print("[*] Generating comprehensive vehicle dataset...")

    test_samples = [
        {
            "filename": "car_daylight_01.jpg",
            "plate_text": "MH 12 AB 1234",
            "clean_text": "MH12AB1234",
            "plate_type": "private",
            "body_color": (220, 220, 225),  # White Sedan
            "condition": "daylight",
            "angle": 0.0,
            "vehicle_type": "Sedan (Private)",
            "state": "Maharashtra (Pune)"
        },
        {
            "filename": "car_daylight_02.jpg",
            "plate_text": "DL 8C AA 1111",
            "clean_text": "DL8CAA1111",
            "plate_type": "private",
            "body_color": (160, 165, 170),  # Silver SUV
            "condition": "daylight",
            "angle": 0.0,
            "vehicle_type": "SUV (Private)",
            "state": "Delhi"
        },
        {
            "filename": "car_night_03.jpg",
            "plate_text": "KA 05 MJ 5050",
            "clean_text": "KA05MJ5050",
            "plate_type": "private",
            "body_color": (35, 40, 50),     # Dark Blue Hatchback
            "condition": "night",
            "angle": 0.0,
            "vehicle_type": "Hatchback (Night)",
            "state": "Karnataka (Bangalore)"
        },
        {
            "filename": "car_angled_04.jpg",
            "plate_text": "KL 07 CD 4567",
            "clean_text": "KL07CD4567",
            "plate_type": "private",
            "body_color": (40, 35, 180),    # Red Sedan
            "condition": "angled",
            "angle": 12.0,
            "vehicle_type": "Sedan (Angled)",
            "state": "Kerala (Kochi)"
        },
        {
            "filename": "car_commercial_05.jpg",
            "plate_text": "HR 26 DQ 5551",
            "clean_text": "HR26DQ5551",
            "plate_type": "commercial",
            "body_color": (210, 210, 210),  # Yellow plate commercial cab
            "condition": "daylight",
            "angle": 0.0,
            "vehicle_type": "Commercial Taxi",
            "state": "Haryana (Gurgaon)"
        },
        {
            "filename": "car_shadow_06.jpg",
            "plate_text": "TN 09 BK 2020",
            "clean_text": "TN09BK2020",
            "plate_type": "private",
            "body_color": (150, 80, 40),    # Blue-Bronze SUV
            "condition": "shadow",
            "angle": 0.0,
            "vehicle_type": "Compact SUV (Shadow)",
            "state": "Tamil Nadu (Chennai)"
        },
        {
            "filename": "car_toll_07.jpg",
            "plate_text": "TS 08 UB 9999",
            "clean_text": "TS08UB9999",
            "plate_type": "commercial",
            "body_color": (230, 230, 235),  # White Commercial Van
            "condition": "daylight",
            "angle": 0.0,
            "vehicle_type": "Commercial Van (Toll)",
            "state": "Telangana (Hyderabad)"
        },
        {
            "filename": "car_rain_08.jpg",
            "plate_text": "UP 32 EF 7890",
            "clean_text": "UP32EF7890",
            "plate_type": "private",
            "body_color": (25, 25, 30),     # Black Luxury Sedan
            "condition": "rain",
            "angle": 0.0,
            "vehicle_type": "Sedan (Wet/Rain)",
            "state": "Uttar Pradesh (Lucknow)"
        }
    ]

    annotations = []

    for item in test_samples:
        filepath = os.path.join(DATASET_DIR, item["filename"])
        img, bbox = create_vehicle_scene(
            plate_text=item["plate_text"],
            plate_type=item["plate_type"],
            body_color=item["body_color"],
            condition=item["condition"],
            angle=item["angle"],
            vehicle_type=item["vehicle_type"]
        )
        cv2.imwrite(filepath, img)

        annotations.append({
            "filename": item["filename"],
            "filepath": filepath,
            "ground_truth_raw": item["plate_text"],
            "ground_truth_clean": item["clean_text"],
            "plate_type": item["plate_type"],
            "vehicle_type": item["vehicle_type"],
            "condition": item["condition"],
            "state": item["state"],
            "bbox_ground_truth": bbox
        })
        print(f" [OK] Created {item['filename']} -> Plate: {item['plate_text']} ({item['condition']})")

    # Save Ground Truth JSON
    gt_path = os.path.join(DATASET_DIR, "ground_truth.json")
    with open(gt_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"\n[+] Generated {len(annotations)} test scenes and annotations in {DATASET_DIR}")
    print(f"[+] Ground truth saved to {gt_path}")

if __name__ == "__main__":
    generate_all()
