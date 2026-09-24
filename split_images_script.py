from pathlib import Path

import cv2
from tqdm import tqdm


# ============================================================
# FOLDERS
# ============================================================

INPUT_DIR = Path("rotation_fixed")

LEFT_DIR = Path("split_test/left")

RIGHT_DIR = Path("split_test/right")


LEFT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RIGHT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIND TIFF IMAGES
# ============================================================

files = sorted([

    file

    for file in INPUT_DIR.iterdir()

    if file.is_file()

    and file.suffix.lower()
    in (".tif", ".tiff")

])


print(f"Found {len(files)} bilateral images.")


# ============================================================
# PROCESS IMAGES
# ============================================================

successful = 0

errors = 0


for file in tqdm(

    files,

    desc="Splitting images",

    unit="img"

):


    # --------------------------------------------------------
    # READ ORIGINAL TIFF
    # --------------------------------------------------------

    img = cv2.imread(

        str(file),

        cv2.IMREAD_UNCHANGED

    )


    if img is None:

        tqdm.write(
            f"Cannot read: {file.name}"
        )

        errors += 1

        continue


    height, width = img.shape[:2]


    # --------------------------------------------------------
    # FIND IMAGE CENTER
    # --------------------------------------------------------

    mid_x = width // 2


    # --------------------------------------------------------
    # NHANES BILATERAL KNEE CONVENTION
    #
    # IMAGE LEFT  = PATIENT'S RIGHT KNEE
    # IMAGE RIGHT = PATIENT'S LEFT KNEE
    # --------------------------------------------------------

    right_knee = img[:, :mid_x]

    left_knee = img[:, mid_x:]


    # --------------------------------------------------------
    # CREATE OUTPUT FILENAMES
    # --------------------------------------------------------

    extension = file.suffix

    base_name = file.stem


    left_output = (

        LEFT_DIR

        / f"{base_name}_L{extension}"

    )


    right_output = (

        RIGHT_DIR

        / f"{base_name}_R{extension}"

    )


    # --------------------------------------------------------
    # SAVE TIFFS
    # --------------------------------------------------------

    left_saved = cv2.imwrite(

        str(left_output),

        left_knee,

        [
            cv2.IMWRITE_TIFF_COMPRESSION,
            5
        ]

    )


    right_saved = cv2.imwrite(

        str(right_output),

        right_knee,

        [
            cv2.IMWRITE_TIFF_COMPRESSION,
            5
        ]

    )


    if left_saved and right_saved:

        successful += 1


    else:

        tqdm.write(
            f"Write error: {file.name}"
        )

        errors += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n======================================")

print("SPLITTING COMPLETED")

print("======================================")


print(f"Input bilateral images : {len(files)}")

print(f"Successfully split     : {successful}")

print(f"Errors                 : {errors}")


print("\nLeft knee images:")

print(LEFT_DIR)


print("\nRight knee images:")

print(RIGHT_DIR)


print("\nExpected output:")

print(f"Left images  : {successful}")

print(f"Right images : {successful}")

print(f"Total knees  : {successful * 2}")