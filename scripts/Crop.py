import os
import csv

import cv2
import numpy as np
from ultralytics import YOLO


# =====================================================
# PATHS  
# =====================================================

MODEL_PATH = r"F:\KOA_Project\best.pt"

INPUT_FOLDERS = {
    "left":  r"F:\KOA_Project\split\left",
    "right": r"F:\KOA_Project\split\right",
}

OUTPUT_ROOT = r"F:\KOA_Project\cropped"



# =====================================================
# SETTINGS
# =====================================================

CONFIDENCE_THRESHOLD = 0.25
PADDING = 0                # px added around the box; try 10-20 if tight
LIMIT = None               # small int to test; None = all
RESUME = True              # skip crops that already exist

VALID_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


# =====================================================
# HELPER: prepare image for YOLO
# =====================================================

def load_original_and_detect(path):
    """
    Returns (original, detect_bgr8):
      original    - image as stored (keeps bit depth/channels) for cropping
      detect_bgr8 - 8-bit, 3-channel BGR copy for YOLO detection
    Returns (None, None) if the file can't be read.
    """
    original = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if original is None:
        return None, None

    det = original

    # Bring to 8-bit (contrast-stretch 16-bit / other depths).
    if det.dtype != np.uint8:
        det = cv2.normalize(det, None, 0, 255, cv2.NORM_MINMAX)
        det = det.astype(np.uint8)

    # Force 3 channels.
    if det.ndim == 2:
        det = cv2.cvtColor(det, cv2.COLOR_GRAY2BGR)
    elif det.shape[2] == 4:
        det = cv2.cvtColor(det, cv2.COLOR_BGRA2BGR)

    return original, det


# =====================================================
# LOAD MODEL
# =====================================================

model = YOLO(MODEL_PATH)
print("YOLO model loaded successfully.\n")


# =====================================================
# PROCESS EACH FOLDER
# =====================================================

grand_processed = 0
grand_cropped = 0
grand_failed = 0

for side, input_folder in INPUT_FOLDERS.items():

    output_folder = os.path.join(OUTPUT_ROOT, side)
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n=== {side.upper()}  ({input_folder}) ===")

    processed = 0
    cropped = 0
    failed = 0
    failed_images = []

    image_list = sorted(os.listdir(input_folder))
    image_list = [n for n in image_list
                  if n.lower().endswith(VALID_EXT)]
    if LIMIT is not None:
        image_list = image_list[:LIMIT]

    for image_name in image_list:

        processed += 1
        image_path = os.path.join(input_folder, image_name)
        save_path = os.path.join(output_folder, image_name)

        if (RESUME and os.path.exists(save_path)
                and os.path.getsize(save_path) > 0):
            cropped += 1
            continue

        image, detect_img = load_original_and_detect(image_path)
        if image is None:
            print(f"[FAILED] Unable to read: {image_name}")
            failed += 1
            failed_images.append(image_name)
            continue

        results = model(detect_img, conf=CONFIDENCE_THRESHOLD,
                        verbose=False)

        if len(results[0].boxes) == 0:
            print(f"[FAILED] No knee detected: {image_name}")
            failed += 1
            failed_images.append(image_name)
            continue

        # Highest-confidence detection
        boxes = results[0].boxes
        best_index = int(boxes.conf.argmax())
        best_box = boxes[best_index]

        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)

        # Pad + clamp to image bounds (coords map 1:1 to original)
        h, w = image.shape[:2]
        x1 = max(0, x1 - PADDING)
        y1 = max(0, y1 - PADDING)
        x2 = min(w, x2 + PADDING)
        y2 = min(h, y2 + PADDING)

        # Crop from the ORIGINAL
        cropped_image = image[y1:y2, x1:x2]

        if save_path.lower().endswith((".tif", ".tiff")):
            cv2.imwrite(save_path, cropped_image,
                        [cv2.IMWRITE_TIFF_COMPRESSION, 5])
        else:
            cv2.imwrite(save_path, cropped_image)

        confidence = float(best_box.conf)
        print(f"[OK] {image_name:<22} conf: {confidence:.3f}")
        cropped += 1

    if failed_images:
        log_path = os.path.join(output_folder, "_failed.csv")
        with open(log_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["failed_image"])
            for name in failed_images:
                writer.writerow([name])
        print(f"  Failed list -> {log_path}")

    print(f"  {side}: processed={processed}  "
          f"cropped={cropped}  failed={failed}")

    grand_processed += processed
    grand_cropped += cropped
    grand_failed += failed


# =====================================================
# SUMMARY
# =====================================================

print("\n======================================")
print("CROPPING COMPLETED")
print("======================================")
print(f"Total processed : {grand_processed}")
print(f"Total cropped   : {grand_cropped}")
print(f"Total failed    : {grand_failed}")
print(f"Output root     : {OUTPUT_ROOT}")
print("======================================")
