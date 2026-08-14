"""
Converts a public image-based ASL dataset into the landmark CSV format
used by train.py — fully automated, no camera or manual signing needed.

Expected input folder structure (this matches the common Kaggle
"ASL Alphabet" layout):

    dataset_root/
        A/
            img1.jpg
            img2.jpg
            ...
        B/
            ...
        0/
            ...
        9/
            ...

Each subfolder name is used as the label (must be a single character
A-Z or 0-9; anything else, like "space"/"del"/"nothing" folders, is
skipped automatically).

Usage:
    python convert_dataset.py /path/to/dataset_root
    python convert_dataset.py /path/to/dataset_root --max-per-class 500
    python convert_dataset.py /path/to/dataset_root --append

By default this OVERWRITES data/landmarks.csv. Use --append to add to
an existing file instead (e.g. to combine with a small self-recorded
set later).
"""
import argparse
import csv
import os
import sys

import cv2
import mediapipe as mp

VALID_LABELS = set(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")


def normalize(landmarks):
    pts = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
    wx, wy, wz = pts[0]
    pts = [(x - wx, y - wy, z - wz) for x, y, z in pts]
    mx, my, mz = pts[9]
    scale = (mx**2 + my**2 + mz**2) ** 0.5 or 1.0
    pts = [(x / scale, y / scale, z / scale) for x, y, z in pts]
    flat = []
    for x, y, z in pts:
        flat.extend([x, y, z])
    return flat


def main():
    parser = argparse.ArgumentParser(description="Convert an image ASL dataset to landmark CSV.")
    parser.add_argument("dataset_root", help="Path to the dataset folder (one subfolder per label)")
    parser.add_argument("--max-per-class", type=int, default=None,
                         help="Cap number of images processed per class (default: use all)")
    parser.add_argument("--append", action="store_true",
                         help="Append to existing landmarks.csv instead of overwriting")
    parser.add_argument("--output", default=CSV_PATH,
                        help="Output CSV path (default: backend/data/landmarks.csv)")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5,
                         help="MediaPipe hand detection confidence threshold (default 0.5, "
                              "lower than live-app default since dataset images are static/cropped)")
    args = parser.parse_args()

    root = args.dataset_root
    if not os.path.isdir(root):
        print(f"Error: {root} is not a directory")
        sys.exit(1)

    subfolders = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )
    label_folders = [d for d in subfolders if d.upper() in VALID_LABELS]
    skipped_folders = [d for d in subfolders if d.upper() not in VALID_LABELS]

    if not label_folders:
        print("No folders matching A-Z or 0-9 found. Check your dataset structure.")
        sys.exit(1)

    if skipped_folders:
        print(f"Skipping non A-Z/0-9 folders: {skipped_folders}")

    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mode = "a" if args.append else "w"
    write_header = not (args.append and os.path.exists(output_path))

    mp_hands = mp.solutions.hands
    total_ok = 0
    total_skipped = 0

    with open(output_path, mode, newline="") as csv_file, \
         mp_hands.Hands(
             static_image_mode=True,
             max_num_hands=1,
             min_detection_confidence=args.min_detection_confidence,
         ) as hands:

        writer = csv.writer(csv_file)
        if write_header:
            header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
            writer.writerow(header)

        for folder in label_folders:
            label = folder.upper()
            folder_path = os.path.join(root, folder)
            files = [
                f for f in sorted(os.listdir(folder_path))
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS
            ]
            if args.max_per_class:
                files = files[: args.max_per_class]

            class_ok = 0
            class_skipped = 0
            for fname in files:
                img_path = os.path.join(folder_path, fname)
                image = cv2.imread(img_path)
                if image is None:
                    class_skipped += 1
                    continue
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                if not results.multi_hand_landmarks:
                    class_skipped += 1
                    continue
                row = [label] + normalize(results.multi_hand_landmarks[0])
                writer.writerow(row)
                class_ok += 1

            total_ok += class_ok
            total_skipped += class_skipped
            print(f"{label}: {class_ok} converted, {class_skipped} skipped (no hand detected / unreadable)")

    print()
    print(f"Done. {total_ok} samples written to {output_path} ({total_skipped} images skipped total).")
    print(f"Import into SQL Server: python import_landmarks_to_sql.py --csv \"{output_path}\"")


if __name__ == "__main__":
    main()
