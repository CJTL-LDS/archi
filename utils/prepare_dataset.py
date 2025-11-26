"""Prepare YOLO detection dataset from custom annotation format.

Source data layout (current project):
    png/               -> original images (PNG files). Filenames match label names.
    png_annt/          -> annotation files named <class>.txt with ONE line: `class_id x_center y_center width height`
    labels.txt         -> ordered list of class names (one per line) defining class indices.

This script will create YOLO-compliant structure:
    dataset/
        images/train/*.png
        images/val/*.png
        labels/train/*.txt  (per-image label files; empty if no boxes)
        labels/val/*.txt
    dataset.yaml

Assumptions / Notes:
1. Each annotation file currently contains a single bounding box line. It is mapped to the image whose filename matches the class name.
2. If multiple images exist for the same class (e.g. duplicated copies), only the matching annotation file is reused; others get an empty label file unless you create additional annotations manually.
3. Trailing spaces in names (e.g. "赛龙舟 ") are trimmed for mapping but the original filenames are preserved.
4. Bounding box coordinates in annotation files are already normalized (0-1). Script does basic validation.

Adjust logic if your annotation format evolves.
"""

from __future__ import annotations

import os
import shutil
import random
from typing import Dict, List

RANDOM_SEED = 42
# TRAIN_SPLIT = 0.8  # Not used anymore, we use all images for both train and val

LABELS_FILE = "labels.txt"
IMAGES_SRC_DIR = "png"
ANNOTATIONS_SRC_DIR = "png_annt"
OUTPUT_ROOT = "dataset"


def load_class_names(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Class labels file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        names = [line.rstrip("\n") for line in f if line.strip()]
    return names


def build_annotation_mapping(annotations_dir: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for fname in os.listdir(annotations_dir):
        if not fname.endswith(".txt"):
            continue
        class_name = fname[:-4].strip()  # remove .txt and trim whitespace
        mapping[class_name] = os.path.join(annotations_dir, fname)
    return mapping


def read_annotation_line(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
    if not line:
        raise ValueError(f"Annotation file {path} is empty.")
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(
            f"Annotation file {path} expected 5 fields 'class x y w h' but got {len(parts)}: {line}")
    # Basic numeric validation
    try:
        _ = int(parts[0])
        for v in parts[1:]:
            float(v)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Invalid numeric values in {path}: {line}") from e
    return line


def prepare_structure(root: str) -> None:
    if os.path.exists(root):
        shutil.rmtree(root)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(root, sub), exist_ok=True)


def main() -> None:
    random.seed(RANDOM_SEED)

    class_names = load_class_names(LABELS_FILE)
    class_index_map = {name.strip(): idx for idx, name in enumerate(class_names)}
    annotation_map = build_annotation_mapping(ANNOTATIONS_SRC_DIR)

    # Check if png folder exists
    if not os.path.exists(IMAGES_SRC_DIR):
         # Try to find it in raw/png if not in root
         if os.path.exists(os.path.join("raw", "png")):
             global IMAGES_SRC_DIR
             IMAGES_SRC_DIR = os.path.join("raw", "png")
         else:
             # Try to find it in dataset/images/train (if already built?) No, source is png
             pass

    if not os.path.exists(IMAGES_SRC_DIR):
        raise FileNotFoundError(f"Images directory '{IMAGES_SRC_DIR}' not found.")

    image_files = [f for f in os.listdir(IMAGES_SRC_DIR) if f.lower().endswith(".png")]
    if not image_files:
        raise FileNotFoundError(f"No PNG images found in {IMAGES_SRC_DIR}")

    # random.shuffle(image_files)
    # split_idx = int(len(image_files) * TRAIN_SPLIT)
    # train_images = set(image_files[:split_idx])
    # val_images = set(image_files[split_idx:])
    
    # FIX: Use ALL images for training because we only have 1 image per class.
    # If we split, the validation set will contain classes never seen in training.
    train_images = set(image_files)
    val_images = set(image_files) # Use same for val to verify overfitting/convergence

    prepare_structure(OUTPUT_ROOT)

    # Process each image
    for img_name in image_files:
        src_img_path = os.path.join(IMAGES_SRC_DIR, img_name)
        base_name = os.path.splitext(img_name)[0]
        trimmed_base = base_name.strip()

        # Determine destination split (both in this case)
        splits = []
        if img_name in train_images: splits.append("train")
        if img_name in val_images: splits.append("val")
        
        for split in splits:
            dst_img_path = os.path.join(OUTPUT_ROOT, "images", split, img_name)
            shutil.copy2(src_img_path, dst_img_path)

            # Matching annotation
            label_file_name = f"{base_name}.txt"  # keep original (including spaces) to match image
            dst_label_path = os.path.join(OUTPUT_ROOT, "labels", split, label_file_name)

            annotation_src_path = annotation_map.get(trimmed_base)
            if annotation_src_path:
                line = read_annotation_line(annotation_src_path)
                # Ensure class id matches mapping (optionally override)
                parts = line.split()
                orig_class_id = int(parts[0])
                expected_class_id = class_index_map.get(trimmed_base)
                if expected_class_id is None:
                    # raise KeyError(f"Class '{trimmed_base}' not found in labels.txt")
                    print(f"Warning: Class '{trimmed_base}' not found in labels.txt. Skipping annotation.")
                    continue
                
                if orig_class_id != expected_class_id:
                    # Replace with expected to be consistent
                    parts[0] = str(expected_class_id)
                    line = " ".join(parts)
                with open(dst_label_path, "w", encoding="utf-8") as f:
                    f.write(line + "\n")
            else:
                # No annotation found: create empty label file (image treated as background / no objects)
                with open(dst_label_path, "w", encoding="utf-8") as f:
                    f.write("")

    # Create dataset.yaml
    yaml_path = "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated YOLO dataset configuration\n")
        f.write(f"path: {os.path.abspath(OUTPUT_ROOT)}\n") # Use absolute path to be safe
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for idx, name in enumerate(class_names):
            f.write(f"  {idx}: {name.strip()}\n")

    print("Dataset preparation complete.")
    print(f"Total images: {len(image_files)} | Train: {len(train_images)} | Val: {len(val_images)}")
    print("Generated dataset.yaml and populated dataset/ directory.")


if __name__ == "__main__":
    main()
