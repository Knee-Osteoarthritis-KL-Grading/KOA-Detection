import os
import csv
import shutil

try:
    from tqdm import tqdm
except ImportError:            # tqdm optional; plain loop if absent
    def tqdm(x, **k):
        return x


# =====================================================
# CONFIG  
# =====================================================

# Source crops (from the YOLO cropping step)
CROPPED_LEFT  = r"F:\KOA_Project\cropped\left"
CROPPED_RIGHT = r"F:\KOA_Project\cropped\right"

# Where the 9 downloaded files (txt lists + manifest) live
LISTS_DIR   = r"F:\KOA_Project"
MANIFEST    = os.path.join(LISTS_DIR, "frozen_split_manifest.csv")
FROZEN_TEST = os.path.join(LISTS_DIR, "frozen_test.txt")

# Where the batch_<N> folders get created 
OUTPUT_ROOT = r"F:\KOA_Project"

BATCHES = [500, 1000, 1500, 2000, 3000, 4686]

DUPLICATE_TEST_PER_BATCH = True


# =====================================================
# LOAD GRADES
# =====================================================

grade_of = {}
with open(MANIFEST, newline="") as fh:
    for row in csv.DictReader(fh):
        grade_of[row["image"]] = int(row["grade"])


def read_list(path):
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def src_path(fname):
    """cropped\\right for _R, cropped\\left for _L."""
    side = fname.rsplit("_", 1)[1][0].upper()
    folder = CROPPED_RIGHT if side == "R" else CROPPED_LEFT
    return os.path.join(folder, fname)


def copy_group(files, dest_split_dir, desc):
    for g in range(5):
        os.makedirs(os.path.join(dest_split_dir, f"Grade {g}"), exist_ok=True)

    copied = skipped = missing = 0
    for f in tqdm(files, desc=desc, unit="img"):
        g = grade_of.get(f)
        if g is None:
            missing += 1
            continue
        src = src_path(f)
        if not os.path.exists(src):
            missing += 1
            continue
        dst = os.path.join(dest_split_dir, f"Grade {g}", f)
        if os.path.exists(dst):
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
    return copied, skipped, missing


# =====================================================
# BUILD
# =====================================================

print("Building ImageFolder trees...\n")
test_files = read_list(FROZEN_TEST)

# Optional single shared test folder
if not DUPLICATE_TEST_PER_BATCH:
    dest = os.path.join(OUTPUT_ROOT, "frozen_test")
    c, s, m = copy_group(test_files, dest, "frozen_test")
    print(f"frozen_test: copied={c} skipped={s} missing={m}\n")

for b in BATCHES:
    print(f"=== batch_{b} ===")
    train_files = read_list(os.path.join(LISTS_DIR, f"batch_{b}_train.txt"))
    batch_dir = os.path.join(OUTPUT_ROOT, f"batch_{b}")

    c, s, m = copy_group(train_files,
                         os.path.join(batch_dir, "train"),
                         f"batch_{b} train")
    print(f"  train: copied={c} skipped={s} missing={m}")

    if DUPLICATE_TEST_PER_BATCH:
        c, s, m = copy_group(test_files,
                             os.path.join(batch_dir, "test"),
                             f"batch_{b} test")
        print(f"  test : copied={c} skipped={s} missing={m}")
    print()

print("Done.")
if not DUPLICATE_TEST_PER_BATCH:
    print(f"Shared test set -> {os.path.join(OUTPUT_ROOT, 'frozen_test')}")
