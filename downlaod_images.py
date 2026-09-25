import requests
from bs4 import BeautifulSoup
import concurrent.futures
from pathlib import Path
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("D:\Knee_OA_Project_raw_images\data")

URL = "https://ftp.cdc.gov/pub/NHANES/XRays/Nhanes3/"

MAX_WORKERS = 10

REQUEST_TIMEOUT = 120


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GET FILE LIST
# ============================================================

def get_file_list():

    print("Fetching file list...")

    response = requests.get(
        URL,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    files = []

    for link in soup.find_all("a"):

        href = link.get("href")

        if not href:
            continue

        if href == "../":
            continue

        if href.startswith("?"):
            continue

        filename = href.split("/")[-1]

        # Keep knee X-ray files
        if "K" in filename.upper():
            files.append(filename)

    # Remove duplicates
    files = sorted(set(files))

    return files


# ============================================================
# DOWNLOAD ONE FILE
# ============================================================

def download_file(filename):

    output_path = OUTPUT_DIR / filename

    try:

        # --------------------------------------------
        # Skip images that already exist
        # --------------------------------------------

        if output_path.exists():

            return "SKIPPED"


        # --------------------------------------------
        # Download original file
        # --------------------------------------------

        response = requests.get(
            URL + filename,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()


        # --------------------------------------------
        # Save EXACT downloaded bytes
        #
        # NO:
        # resizing
        # normalization
        # TIFF recompression
        # bit-depth conversion
        # image decoding / encoding
        # --------------------------------------------

        with open(output_path, "wb") as file:

            file.write(response.content)


        return "DOWNLOADED"


    except Exception as error:

        return f"ERROR: {filename}: {error}"


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    files = get_file_list()

    total_files = len(files)

    print(f"Found {total_files} files.")

    print("Downloading original images...")

    print("Existing images will be skipped.")


    downloaded = 0

    skipped = 0

    errors = []


    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:


        future_to_filename = {

            executor.submit(
                download_file,
                filename
            ): filename

            for filename in files

        }


        for future in tqdm(

            concurrent.futures.as_completed(
                future_to_filename
            ),

            total=total_files,

            unit="img"

        ):

            result = future.result()


            if result == "DOWNLOADED":

                downloaded += 1


            elif result == "SKIPPED":

                skipped += 1


            else:

                errors.append(result)

                tqdm.write(result)


    print("\n========================================")

    print("DOWNLOAD COMPLETED")

    print("========================================")

    print(f"Total files : {total_files}")

    print(f"Downloaded  : {downloaded}")

    print(f"Skipped     : {skipped}")

    print(f"Errors      : {len(errors)}")


    if errors:

        error_file = Path(
            "download_errors.txt"
        )

        with open(
            error_file,
            "w",
            encoding="utf-8"
        ) as file:

            for error in errors:

                file.write(error + "\n")

        print(
            f"\nErrors saved to: {error_file}"
        )


    print(
        f"\nOriginal files saved to: {OUTPUT_DIR}"
    )