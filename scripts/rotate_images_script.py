from pathlib import Path
import csv
import shutil

import cv2
import easyocr
import numpy as np
from tqdm import tqdm


# ============================================================
# FOLDERS
# ============================================================

INPUT_DIR = Path("Rotate images")

OUTPUT_DIR = Path("rotation_fixed")

UNCERTAIN_DIR = Path("rotation_uncertain")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNCERTAIN_DIR.mkdir(parents=True, exist_ok=True)


LOG_FILE = OUTPUT_DIR / "rotation_log.csv"


# ============================================================
# OCR
# ============================================================

reader = easyocr.Reader(
    ["en"],
    gpu=False
)


# ============================================================
# PREPARE IMAGE FOR OCR
# ============================================================

def prepare_for_ocr(img):

    if img.ndim == 3:

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = img


    # Convert TIFF to OCR-friendly 8-bit image.
    #
    # This image is used ONLY for OCR.
    # The original TIFF is rotated and saved.

    low, high = np.percentile(
        gray,
        (1, 99)
    )


    if high > low:

        gray = np.clip(
            gray,
            low,
            high
        )

        gray = (
            (gray - low)
            / (high - low)
            * 255
        ).astype(np.uint8)


    else:

        gray = cv2.normalize(
            gray,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)


    return gray


# ============================================================
# DETECT R/L MARKERS AND SCORE ORIENTATION
# ============================================================

def orientation_score(img):

    gray = prepare_for_ocr(img)

    height, width = gray.shape[:2]


    results = reader.readtext(

        gray,

        detail=1,

        paragraph=False

    )


    score = 0

    detected_markers = []


    for box, text, confidence in results:


        marker = str(
            text
        ).upper().strip()


        # Ignore weak OCR detections

        if confidence < 0.20:

            continue


        # Accept only isolated R and L markers

        if marker not in ("R", "L"):

            continue


        center_x = sum(

            point[0]

            for point in box

        ) / len(box)


        center_y = sum(

            point[1]

            for point in box

        ) / len(box)


        x_fraction = center_x / width

        y_fraction = center_y / height


        detected_markers.append({

            "marker":
                marker,

            "confidence":
                round(
                    float(confidence),
                    3
                ),

            "x":
                round(
                    float(x_fraction),
                    3
                ),

            "y":
                round(
                    float(y_fraction),
                    3
                )

        })


        # ----------------------------------------------------
        # EXPECTED FINAL ORIENTATION
        #
        # IMAGE LEFT  = RIGHT knee
        # IMAGE RIGHT = LEFT knee
        # ----------------------------------------------------


        if marker == "R":


            if x_fraction < 0.5:

                score += 10


            else:

                score -= 10



        elif marker == "L":


            if x_fraction >= 0.5:

                score += 10


            else:

                score -= 10



        # Small confidence bonus

        score += confidence


    return score, detected_markers


# ============================================================
# FIND TIFF FILES
# ============================================================

files = sorted([

    file

    for file in INPUT_DIR.iterdir()

    if file.is_file()

    and file.suffix.lower()
    in (".tif", ".tiff")

])


print(
    f"Found {len(files)} images needing rotation."
)


# ============================================================
# PROCESS IMAGES
# ============================================================

rows = []


for file in tqdm(

    files,

    desc="Rotating images",

    unit="img"

):


    img = cv2.imread(

        str(file),

        cv2.IMREAD_UNCHANGED

    )


    # ========================================================
    # READ ERROR
    # ========================================================

    if img is None:


        rows.append({

            "filename":
                file.name,

            "action":
                "READ_ERROR",

            "cw_score":
                "",

            "ccw_score":
                "",

            "cw_markers":
                "",

            "ccw_markers":
                ""

        })


        continue


    # ========================================================
    # CREATE BOTH ROTATIONS
    # ========================================================

    clockwise = cv2.rotate(

        img,

        cv2.ROTATE_90_CLOCKWISE

    )


    counterclockwise = cv2.rotate(

        img,

        cv2.ROTATE_90_COUNTERCLOCKWISE

    )


    # ========================================================
    # SCORE BOTH ORIENTATIONS
    # ========================================================

    cw_score, cw_markers = orientation_score(

        clockwise

    )


    ccw_score, ccw_markers = orientation_score(

        counterclockwise

    )


    # ========================================================
    # SCORE DIFFERENCE
    # ========================================================

    score_difference = abs(

        cw_score - ccw_score

    )


    # ========================================================
    # SELECT CLOCKWISE
    # ========================================================

    if (

        cw_score > ccw_score

        and score_difference >= 5

    ):


        selected = clockwise

        action = "CW"



    # ========================================================
    # SELECT COUNTER-CLOCKWISE
    # ========================================================

    elif (

        ccw_score > cw_score

        and score_difference >= 5

    ):


        selected = counterclockwise

        action = "CCW"



    # ========================================================
    # UNCERTAIN
    # ========================================================

    else:


        # Copy ORIGINAL unrotated TIFF
        # into manual-review folder.

        uncertain_path = (

            UNCERTAIN_DIR

            / file.name

        )


        shutil.copy2(

            file,

            uncertain_path

        )


        rows.append({

            "filename":
                file.name,

            "action":
                "UNCERTAIN",

            "cw_score":
                round(
                    cw_score,
                    3
                ),

            "ccw_score":
                round(
                    ccw_score,
                    3
                ),

            "cw_markers":
                str(
                    cw_markers
                ),

            "ccw_markers":
                str(
                    ccw_markers
                )

        })


        continue


    # ========================================================
    # SAVE CONFIDENT ROTATION
    # ========================================================

    output_path = (

        OUTPUT_DIR

        / file.name

    )


    success = cv2.imwrite(

        str(output_path),

        selected,

        [
            cv2.IMWRITE_TIFF_COMPRESSION,
            5
        ]

    )


    rows.append({

        "filename":
            file.name,

        "action":

            action

            if success

            else "WRITE_ERROR",

        "cw_score":
            round(
                cw_score,
                3
            ),

        "ccw_score":
            round(
                ccw_score,
                3
            ),

        "cw_markers":
            str(
                cw_markers
            ),

        "ccw_markers":
            str(
                ccw_markers
            )

    })


# ============================================================
# SAVE LOG
# ============================================================

fieldnames = [

    "filename",

    "action",

    "cw_score",

    "ccw_score",

    "cw_markers",

    "ccw_markers"

]


with open(

    LOG_FILE,

    "w",

    newline="",

    encoding="utf-8"

) as csv_file:


    writer = csv.DictWriter(

        csv_file,

        fieldnames=fieldnames

    )


    writer.writeheader()

    writer.writerows(rows)


# ============================================================
# SUMMARY
# ============================================================

cw_count = sum(

    row["action"] == "CW"

    for row in rows

)


ccw_count = sum(

    row["action"] == "CCW"

    for row in rows

)


uncertain_count = sum(

    row["action"] == "UNCERTAIN"

    for row in rows

)


error_count = sum(

    row["action"] in (

        "READ_ERROR",

        "WRITE_ERROR"

    )

    for row in rows

)


print("\n======================================")

print("ROTATION COMPLETED")

print("======================================")


print(f"Input images : {len(files)}")

print(f"Rotated CW   : {cw_count}")

print(f"Rotated CCW  : {ccw_count}")

print(f"Uncertain    : {uncertain_count}")

print(f"Errors       : {error_count}")


print("\nCorrected images saved to:")

print(OUTPUT_DIR)


print("\nUncertain images saved to:")

print(UNCERTAIN_DIR)


print("\nRotation log saved to:")

print(LOG_FILE)