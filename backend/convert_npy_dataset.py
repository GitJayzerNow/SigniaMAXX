"""
Converts an .npy-format image dataset into the landmark CSV format used
by train.py. Supports two cases:

1. X.npy + Y.npy (one-hot labels)
2. X.npy ONLY -- labels inferred from contiguous per-class blocks.
   This matches the well-documented layout of the ardamavi
   "Sign Language Digits Dataset" (2062 samples, 64x64 grayscale):
   the 2062 images are stored as 10 contiguous blocks, one per digit,
   in the order 9,0,1,2,3,4,5,6,7,8, with these per-class counts:
   9:204, 0:205, 1:206, 2:206, 3:206, 4:207, 5:207, 6:207, 7:206, 8:208

   If your X.npy doesn't match this exact shape (2062, 64, 64), the
   block-count defaults below won't apply -- pass --block-sizes and
   --block-labels yourself, or provide a Y.npy if you have one.

Usage:
    # With Y.npy:
    python convert_npy_dataset.py X.npy --y-path Y.npy

    # Without Y.npy (ardamavi digits dataset, auto-detected by shape):
    python convert_npy_dataset.py X.npy

    # Without Y.npy, custom block layout:
    python convert_npy_dataset.py X.npy \
        --block-labels 9,0,1,2,3,4,5,6,7,8 \
        --block-sizes 204,205,206,206,206,207,207,207,206,208

    python convert_npy_dataset.py X.npy --append
"""
import argparse
import csv
import os

import cv2
import numpy as np
import mediapipe as mp

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")

# Known layout for the ardamavi Sign Language Digits Dataset (2062, 64, 64)
ARDAMAVI_BLOCK_LABELS = ["9", "0", "1", "2", "3", "4", "5", "6", "7", "8"]
ARDAMAVI_BLOCK_SIZES = [204, 205, 206, 206, 206, 207, 207, 207, 206, 208]


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


def to_uint8_bgr(img):
    img = np.array(img)
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img = cv2.cvtColor(img.squeeze(-1), cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def build_labels_from_blocks(n_samples, block_labels, block_sizes):
    if sum(block_sizes) != n_samples:
        raise ValueError(
            f"Block sizes sum to {sum(block_sizes)} but X has {n_samples} samples. "
            "Pass --block-sizes matching your actual dataset, or provide --y-path."
        )
    labels = []
    for label, size in zip(block_labels, block_sizes):
        labels.extend([label] * size)
    return labels


def main():
    parser = argparse.ArgumentParser(description="Convert an .npy image dataset to landmark CSV.")
    parser.add_argument("x_path", help="Path to X.npy (images)")
    parser.add_argument("--y-path", type=str, default=None,
                         help="Path to Y.npy (one-hot labels), if you have one")
    parser.add_argument("--label-order", type=str, default="9,0,1,2,3,4,5,6,7,8",
                         help="Used only with --y-path: maps one-hot column index -> label")
    parser.add_argument("--block-labels", type=str, default=None,
                         help="Used only WITHOUT --y-path: comma-separated label per contiguous block, "
                              "e.g. 9,0,1,2,3,4,5,6,7,8")
    parser.add_argument("--block-sizes", type=str, default=None,
                         help="Used only WITHOUT --y-path: comma-separated sample count per block, "
                              "must sum to len(X)")
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--min-detection-confidence", type=float, default=0.4)
    args = parser.parse_args()

    X = np.load(args.x_path)
    print(f"Loaded X: {X.shape}, dtype {X.dtype}")

    if args.y_path:
        Y = np.load(args.y_path)
        print(f"Loaded Y: {Y.shape}")
        label_order = [s.strip() for s in args.label_order.split(",")]
        class_idx = np.argmax(Y, axis=1) if Y.ndim == 2 else Y.astype(int)
        labels = [label_order[i] for i in class_idx]
    else:
        if args.block_labels and args.block_sizes:
            block_labels = [s.strip() for s in args.block_labels.split(",")]
            block_sizes = [int(s.strip()) for s in args.block_sizes.split(",")]
        elif X.shape == (2062, 64, 64):
            print("No --y-path given. X.npy shape matches the known ardamavi "
                  "Sign-Language-Digits-Dataset layout -- using its documented "
                  "block order/sizes automatically.")
            block_labels = ARDAMAVI_BLOCK_LABELS
            block_sizes = ARDAMAVI_BLOCK_SIZES
        else:
            raise SystemExit(
                "No --y-path given and X.npy shape doesn't match the known "
                f"ardamavi layout (got {X.shape}, expected (2062, 64, 64)).\n"
                "Provide --y-path Y.npy, OR --block-labels and --block-sizes "
                "matching how your dataset is actually ordered."
            )
        labels = build_labels_from_blocks(len(X), block_labels, block_sizes)

    os.makedirs(DATA_DIR, exist_ok=True)
    mode = "a" if args.append else "w"
    write_header = not (args.append and os.path.exists(CSV_PATH))

    mp_hands = mp.solutions.hands
    per_class_count = {}
    total_ok = 0
    total_skipped = 0

    with open(CSV_PATH, mode, newline="") as csv_file, \
         mp_hands.Hands(
             static_image_mode=True,
             max_num_hands=1,
             min_detection_confidence=args.min_detection_confidence,
         ) as hands:

        writer = csv.writer(csv_file)
        if write_header:
            header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
            writer.writerow(header)

        for i in range(len(X)):
            label = labels[i]
            if args.max_per_class and per_class_count.get(label, 0) >= args.max_per_class:
                continue

            img = to_uint8_bgr(X[i])
            h, w = img.shape[:2]
            if max(h, w) < 300:
                scale = 300 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if not results.multi_hand_landmarks:
                total_skipped += 1
                continue

            row = [label] + normalize(results.multi_hand_landmarks[0])
            writer.writerow(row)
            per_class_count[label] = per_class_count.get(label, 0) + 1
            total_ok += 1

    print()
    for label in sorted(per_class_count.keys()):
        print(f"{label}: {per_class_count[label]} converted")
    print(f"\nDone. {total_ok} samples written to {CSV_PATH} ({total_skipped} images skipped, no hand detected).")
    print("Next: python train.py")


if __name__ == "__main__":
    main()
