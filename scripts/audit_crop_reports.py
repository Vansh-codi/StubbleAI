from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "crop_reports_des"

EXPECTED_STATES = {
    "assam": "Assam",
    "bihar": "Bihar",
    "haryana": "Haryana",
    "madhyapradesh": "Madhya Pradesh",
    "punjab": "Punjab",
    "uttarpradesh": "Uttar Pradesh",
}
EXPECTED_CROPS = {"rice", "wheat", "maize"}

def normalize(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())

def main():
    print("STUBBLEAI V3 — DES CROP REPORT AUDIT")
    print("=" * 55)
    print("Folder:", DATA_DIR)

    if not DATA_DIR.exists():
        print("ERROR: Crop report folder does not exist.")
        sys.exit(1)

    files = sorted(p for p in DATA_DIR.iterdir() if p.is_file())
    excel_files = [p for p in files if p.suffix.lower() in {".xls", ".xlsx"}]
    print(f"\nFiles found: {len(files)}")
    print(f"Excel reports: {len(excel_files)}")

    seen = set()
    issues = []

    for path in excel_files:
        stem = normalize(path.stem)
        state = next((s for key, s in EXPECTED_STATES.items() if key in stem), None)
        crop = next((c for c in EXPECTED_CROPS if c in stem), None)

        if not state:
            issues.append(f"Unrecognized state in filename: {path.name}")
        if not crop:
            issues.append(f"Unrecognized crop in filename: {path.name}")

        if state and crop:
            pair = (state, crop)
            if pair in seen:
                issues.append(f"Duplicate state/crop report: {state} / {crop}")
            seen.add(pair)

        print(f"\n{path.name}")
        print(f"  State: {state or 'UNRECOGNIZED'}")
        print(f"  Crop:  {crop or 'UNRECOGNIZED'}")
        print(f"  Size:  {path.stat().st_size:,} bytes")

    missing = [
        (state, crop)
        for state in EXPECTED_STATES.values()
        for crop in sorted(EXPECTED_CROPS)
        if (state, crop) not in seen
    ]

    print("\n" + "=" * 55)
    print("EXPECTED COMBINATIONS:", len(EXPECTED_STATES) * len(EXPECTED_CROPS))
    print("RECOGNIZED COMBINATIONS:", len(seen))
    print("MISSING COMBINATIONS:", len(missing))

    for state, crop in missing:
        issues.append(f"Missing report: {state} / {crop}")

    if issues:
        print("\nISSUES TO REVIEW:")
        for issue in issues:
            print(" -", issue)
    else:
        print("\nFilename-level checks passed.")

    print("\nNOTE: This is only a filename and inventory audit.")
    print("It has NOT yet checked workbook contents, years, seasons,")
    print("district coverage, units, or duplicate crop records.")

if __name__ == "__main__":
    main()
