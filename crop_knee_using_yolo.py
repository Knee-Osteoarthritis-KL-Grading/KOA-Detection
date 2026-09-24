"""
crop_knee_using_yolo.py

Author: Kalhara

Purpose:
Detect the knee using a trained YOLO11n model and crop
the detected knee bounding box from unseen X-ray images.
"""

import os
import cv2
from ultralytics import YOLO

# =====================================================
# PROJECT PATHS
# =====================================================

PROJECT_DIR = r"D:\Knee_YOLO_Project"

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "Results",
    "YOLO11n_50",
    "weights",
    "best.pt"
)

INPUT_FOLDER = os.path.join(
    PROJECT_DIR,
    "Unknown_Images"
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_DIR,
    "Cropped_Knees"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =====================================================
# SETTINGS
# =====================================================

CONFIDENCE_THRESHOLD = 0.25

# =====================================================
# LOAD MODEL
# =====================================================

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.\n")

# =====================================================
# PROCESS IMAGES
# =====================================================

processed = 0
cropped = 0
failed = 0

# Store failed image names
failed_images = []

image_list = sorted(os.listdir(INPUT_FOLDER))

for image_name in image_list:

    if not image_name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        continue

    processed += 1

    image_path = os.path.join(INPUT_FOLDER, image_name)

    image = cv2.imread(image_path)

    if image is None:
        print(f"[FAILED] Unable to read: {image_name}")
        failed += 1
        failed_images.append(image_name)
        continue

    # Run YOLO prediction
    results = model(image, conf=CONFIDENCE_THRESHOLD, verbose=False)

    # No detection
    if len(results[0].boxes) == 0:
        print(f"[FAILED] No knee detected: {image_name}")
        failed += 1
        failed_images.append(image_name)
        continue

    # Highest-confidence detection
    boxes = results[0].boxes
    best_index = boxes.conf.argmax()
    best_box = boxes[best_index]

    x1, y1, x2, y2 = (
        best_box.xyxy[0]
        .cpu()
        .numpy()
        .astype(int)
    )

    # Keep coordinates inside image
    h, w = image.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # Crop image
    cropped_image = image[y1:y2, x1:x2]

    # Save crop
    save_path = os.path.join(
        OUTPUT_FOLDER,
        image_name
    )

    cv2.imwrite(save_path, cropped_image)

    confidence = float(best_box.conf)

    print(
        f"[OK] {image_name:<20} "
        f"Confidence: {confidence:.3f}"
    )

    cropped += 1

# =====================================================
# SUMMARY
# =====================================================

print("\n======================================")
print("Cropping Completed Successfully")
print("======================================")
print(f"Images Processed : {processed}")
print(f"Cropped Images   : {cropped}")
print(f"Failed Detections: {failed}")
print(f"Output Folder    : {OUTPUT_FOLDER}")
print("======================================")

# =====================================================
# FAILED IMAGE LIST
# =====================================================

if failed_images:
    print("\nFailed Image Names")
    print("------------------")
    for img in failed_images:
        print(img)
else:
    print("\nNo failed images. All images were processed successfully.")
    