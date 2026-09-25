import os
import shutil
from collections import Counter

import pandas as pd
from tqdm import tqdm

# --- Configuration ---
CSV_FILE = 'xr.csv'             # Path to your CSV (use a full path if not alongside this script)
SOURCE_FOLDER = 'cropped_1000'  # Folder containing the cropped _L / _R images
OUTPUT_BASE = 'train_1000'      # Root folder for the organized copies

# NHANES codes 8 and 9 are "uninterpretable / missing", NOT grades 0-4.
# They are routed here and kept out of train/0..4 so they never reach training.
INVALID_CODES = {'8', '9'}
INVALID_DIR = '_invalid_89'


def load_labels(csv_path):
    print(f"Loading labels from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['SEQN'] = df['SEQN'].astype(str)

    label_map = {}
    for _, row in df.iterrows():
        seqn = row['SEQN']
        val_r = str(row['XRPKLR']).split('.')[0]   # 1.0 -> "1"
        val_l = str(row['XRPKLL']).split('.')[0]
        label_map[seqn] = {'R': val_r, 'L': val_l}
    return label_map


def copy_and_organize():
    labels = load_labels(CSV_FILE)
    if not labels:
        return

    if not os.path.exists(SOURCE_FOLDER):
        print(f"Source folder '{SOURCE_FOLDER}' does not exist.")
        return

    all_files = [f for f in os.listdir(SOURCE_FOLDER) if f.lower().endswith('.tiff')]
    print(f"Found {len(all_files)} images. Starting copy process...")

    copied_count = 0
    skipped_count = 0
    dist = Counter()   # per-destination tally for the final report

    for filename in tqdm(all_files, desc="Copying", unit="img"):
        try:
            if 'K' not in filename:
                skipped_count += 1
                continue
            seqn = filename.split('K')[0]

            name_no_ext = os.path.splitext(filename)[0]
            if name_no_ext.endswith('_R'):
                side = 'R'
            elif name_no_ext.endswith('_L'):
                side = 'L'
            else:
                skipped_count += 1
                continue

            # Decide destination
            if seqn not in labels:
                dest_subfolder = 'unknown_seqn'
            else:
                val = labels[seqn][side]
                if val.lower() == 'nan':
                    dest_subfolder = 'uncategorized'
                elif val in INVALID_CODES:
                    dest_subfolder = INVALID_DIR      # 8/9 -> excluded, not a grade
                else:
                    dest_subfolder = f"Grade {val}"            # "0".."4"

            target_dir = os.path.join(OUTPUT_BASE, dest_subfolder)
            os.makedirs(target_dir, exist_ok=True)

            shutil.copy2(os.path.join(SOURCE_FOLDER, filename),
                         os.path.join(target_dir, filename))
            copied_count += 1
            dist[dest_subfolder] += 1

        except Exception as e:
            tqdm.write(f"Error copying {filename}: {e}")

    # --- Report ---
    print("\nProcessing Complete.")
    print(f"Successfully copied: {copied_count}")
    print(f"Skipped (no K, no _L/_R): {skipped_count}")

    print("\nClass distribution:")
    grade_total = 0
    for g in ['0', '1', '2', '3', '4']:
        print(f"  KL {g}          : {dist.get(g, 0)}")
        grade_total += dist.get(g, 0)
    print(f"  --------------------")
    print(f"  Gradeable 0-4 : {grade_total}")
    for extra in [INVALID_DIR, 'uncategorized', 'unknown_seqn']:
        if dist.get(extra):
            print(f"  {extra:<13}: {dist[extra]}  (excluded from training)")


if __name__ == "__main__":
    copy_and_organize()
