import re
import shutil
from pathlib import Path

# ---- settings -----------------------------------------------------------
SOURCE_DIR = r"F:\KOA_Project\cropped\new cropped"   # folder to scan
LEFT_DIR   = r"F:\KOA_Project\cropped\new cropped\new_left"    # where _L images go
RIGHT_DIR  = r"F:\KOA_Project\cropped\new cropped\new_right"   # where _R images go
MODE       = "move"     # "move" or "copy"
DRY_RUN    = True       # True = just print what would happen, change nothing
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# -------------------------------------------------------------------------

# Matches "_L" or "_R" immediately before the file extension, any case.
SUFFIX_RE = re.compile(r"_([LR])$", re.IGNORECASE)


def side_for(path: Path) -> str | None:
    """Return 'left', 'right', or None based on the filename stem."""
    match = SUFFIX_RE.search(path.stem)
    if not match:
        return None
    return "left" if match.group(1).upper() == "L" else "right"


def main() -> None:
    source = Path(SOURCE_DIR)
    if not source.is_dir():
        raise SystemExit(f"Source folder not found: {source}")

    left_dir = Path(LEFT_DIR)
    right_dir = Path(RIGHT_DIR)

    counts = {"left": 0, "right": 0}
    skipped = []

    for path in source.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue

        side = side_for(path)
        if side is None:
            skipped.append(path.name)
            continue

        target_dir = left_dir if side == "left" else right_dir
        target = target_dir / path.name

        if DRY_RUN:
            label = "would COPY" if MODE == "copy" else "would MOVE"
            print(f"[DRY-RUN] {label} {path.name} -> {side}/")
        else:
            action = "COPY" if MODE == "copy" else "MOVE"
            print(f"[{action}] {path.name} -> {side}/")

        if not DRY_RUN:
            target_dir.mkdir(parents=True, exist_ok=True)
            if MODE == "copy":
                shutil.copy2(path, target)
            else:
                shutil.move(str(path), str(target))

        counts[side] += 1

    print("\n--- summary ---")
    print(f"left : {counts['left']}")
    print(f"right: {counts['right']}")
    if skipped:
        print(f"skipped (no _L/_R suffix): {len(skipped)}")
        for name in skipped:
            print(f"   {name}")
    if DRY_RUN:
        print("\nDRY_RUN is on — nothing was changed. "
              "Set DRY_RUN = False to apply.")


if __name__ == "__main__":
    main()
