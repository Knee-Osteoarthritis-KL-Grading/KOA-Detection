from pathlib import Path
import csv

import cv2
from tqdm import tqdm


# ============================================================
# CONFIG  
# ============================================================

INPUT_DIR = Path(r"F:\KOA_Project\Data (labels)")

OUTPUT_ROOT = Path(r"F:\KOA_Project\split")
LEFT_DIR  = OUTPUT_ROOT / "left"    # patient's LEFT knee
RIGHT_DIR = OUTPUT_ROOT / "right"   # patient's RIGHT knee

LIMIT = None

RESUME = True

SAVE_PREVIEWS = True
PREVIEW_COUNT = 20
PREVIEW_DIR = OUTPUT_ROOT / "_previews"

# NHANES / standard radiographic convention:
#   viewer's LEFT half of the film = patient's RIGHT knee.
# If your preview spot-check shows the opposite, flip this to False.
IMAGE_LEFT_IS_PATIENT_RIGHT = True

MANIFEST_CSV = OUTPUT_ROOT / "split_manifest.csv"

VALID_SUFFIXES = (".tif", ".tiff")


# ============================================================
# SETUP
# ============================================================

for d in (LEFT_DIR, RIGHT_DIR):
    d.mkdir(parents=True, exist_ok=True)
if SAVE_PREVIEWS:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(
    f for f in INPUT_DIR.iterdir()
    if f.is_file() and f.suffix.lower() in VALID_SUFFIXES
)
if LIMIT is not None:
    files = files[:LIMIT]

print(f"Found {len(files)} bilateral images to process "
      f"(LIMIT={LIMIT}).")


def to_preview_bgr(im):
    
    if im.dtype != "uint8":
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX)
        im = im.astype("uint8")
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return im


# ============================================================
# PROCESS
# ============================================================

successful = 0
skipped = 0
errors = 0
manifest_rows = []

for i, file in enumerate(
    tqdm(files, desc="Splitting", unit="img")
):
    base = file.stem
    ext = file.suffix

    left_out  = LEFT_DIR  / f"{base}_L{ext}"   # patient LEFT
    right_out = RIGHT_DIR / f"{base}_R{ext}"   # patient RIGHT

    if (RESUME and left_out.exists() and right_out.exists() and left_out.stat().st_size > 0 and right_out.stat().st_size > 0):    
        skipped += 1
        continue

    img = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
    if img is None:
        tqdm.write(f"Cannot read: {file.name}")
        errors += 1
        continue

    height, width = img.shape[:2]
    mid_x = width // 2

    left_half  = img[:, :mid_x]
    right_half = img[:, mid_x:]

    # Map image halves -> patient sides based on the convention flag.
    if IMAGE_LEFT_IS_PATIENT_RIGHT:
        patient_right, patient_left = left_half, right_half
    else:
        patient_left, patient_right = left_half, right_half

    tiff_opts = [cv2.IMWRITE_TIFF_COMPRESSION, 5]  # 5 = LZW, lossless
    ok_l = cv2.imwrite(str(left_out),  patient_left,  tiff_opts)
    ok_r = cv2.imwrite(str(right_out), patient_right, tiff_opts)

    if ok_l and ok_r:
        successful += 1
        manifest_rows.append({
            "source": file.name,
            "width": width,
            "height": height,
            "mid_x": mid_x,
            "left_output": left_out.name,
            "right_output": right_out.name,
            "image_left_is_patient_right": IMAGE_LEFT_IS_PATIENT_RIGHT,
        })
    else:
        tqdm.write(f"Write error: {file.name}")
        errors += 1
        continue

    # Annotated preview for the first PREVIEW_COUNT images.
    if SAVE_PREVIEWS and i < PREVIEW_COUNT:
        prev = to_preview_bgr(img.copy())
        cv2.line(prev, (mid_x, 0), (mid_x, height),
                 (0, 0, 255), 2)
        left_label  = "PATIENT R" if IMAGE_LEFT_IS_PATIENT_RIGHT else "PATIENT L"
        right_label = "PATIENT L" if IMAGE_LEFT_IS_PATIENT_RIGHT else "PATIENT R"
        cv2.putText(prev, left_label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(prev, right_label, (mid_x + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite(str(PREVIEW_DIR / f"{base}_preview.jpg"), prev)


# ============================================================
# MANIFEST + SUMMARY
# ============================================================

if manifest_rows:
    with open(MANIFEST_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

print("\n======================================")
print("SPLITTING COMPLETED")
print("======================================")
print(f"Processed        : {len(files)}")
print(f"Successfully split: {successful}")
print(f"Skipped (resume) : {skipped}")
print(f"Errors           : {errors}")
print(f"\nLeft knees  -> {LEFT_DIR}")
print(f"Right knees -> {RIGHT_DIR}")
if SAVE_PREVIEWS:
    print(f"Previews    -> {PREVIEW_DIR}  (check these before full run)")
if manifest_rows:
    print(f"Manifest    -> {MANIFEST_CSV}")
