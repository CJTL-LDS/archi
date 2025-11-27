"""Augment the custom PNG dataset and rebuild a YOLO-formatted dataset directory.

Source folders expected next to this script:
    png/        -> original PNG images (one per cultural heritage item)
    png_annt/   -> YOLO txt annotations named after the class (normalized xywh)
    labels.txt  -> class name list; line number equals class id

The script will:
1. Generate a richer set of variants (original, horizontal flip, light/dark brightness).
2. Recompute bounding boxes for geometric transforms.
3. Regenerate the YOLO directory layout under `dataset/`.
4. Write an updated `dataset.yaml` pointing at the new dataset.

Usage (PowerShell):
    uv run python augment_dataset.py

You can customise behaviour with CLI flags:
    --output dataset_aug        # change output folder
    --train-split 0.75          # ratio of samples kept in the training split
    --copies 3                  # number of extra brightness jitters for each base image

Only Pillow is required (already used elsewhere in the project).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

from PIL import Image, ImageEnhance

# Types --------------------------------------------------------------------
NormalizedBox = Tuple[int, float, float, float, float]  # cls, x, y, w, h
AugmentFn = Callable[[Image.Image, Sequence[NormalizedBox]], Tuple[Image.Image, List[NormalizedBox]]]

# Constants ----------------------------------------------------------------
LABELS_FILE = "../labels.txt"
IMAGES_SRC_DIR = Path("../raw/hide")
ANNOTATIONS_SRC_DIR = Path("../raw/hid_annt")
DEFAULT_OUTPUT = Path("../dataset3")
RANDOM_SEED = 42

# Utility helpers ----------------------------------------------------------

def load_class_names(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Class labels file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_annotation_mapping(annotations_dir: Path) -> dict[str, Path]:
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotations_dir}")
    mapping: dict[str, Path] = {}
    for txt in annotations_dir.glob("*.txt"):
        mapping[txt.stem.strip()] = txt
    return mapping


def parse_boxes(path: Path) -> List[NormalizedBox]:
    if not path.exists():
        return []
    boxes: List[NormalizedBox] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) != 5:
                raise ValueError(f"Annotation {path} must have 5 fields per line, got: {raw}")
            cls = int(float(parts[0]))
            x, y, w, h = (float(p) for p in parts[1:])
            boxes.append((cls, x, y, w, h))
    return boxes


def ensure_class_ids(boxes: Sequence[NormalizedBox], class_idx_map: dict[str, int], name: str) -> List[NormalizedBox]:
    """Force class ids to match labels.txt ordering."""
    expected = class_idx_map.get(name)
    if expected is None:
        raise KeyError(f"Class '{name}' missing from labels.txt")
    patched: List[NormalizedBox] = []
    for _, x, y, w, h in boxes:
        patched.append((expected, x, y, w, h))
    return patched


def slugify(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        cleaned = "sample"
    # Replace spaces and forbidden characters for Windows paths
    translation = str.maketrans({" ": "_", "/": "_", "\\": "_", ":": "-"})
    cleaned = cleaned.translate(translation)
    cleaned = cleaned.replace("__", "_")
    return cleaned


def format_boxes(boxes: Sequence[NormalizedBox]) -> str:
    return "\n".join(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for cls, x, y, w, h in boxes)


def prepare_output_dir(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (root / sub).mkdir(parents=True, exist_ok=True)


# Augmentations ------------------------------------------------------------

def identity(image: Image.Image, boxes: Sequence[NormalizedBox]) -> Tuple[Image.Image, List[NormalizedBox]]:
    return image.copy(), list(boxes)


def hflip(image: Image.Image, boxes: Sequence[NormalizedBox]) -> Tuple[Image.Image, List[NormalizedBox]]:
    flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
    updated = [(cls, 1.0 - x, y, w, h) for cls, x, y, w, h in boxes]
    return flipped, updated


def vflip(image: Image.Image, boxes: Sequence[NormalizedBox]) -> Tuple[Image.Image, List[NormalizedBox]]:
    flipped = image.transpose(Image.FLIP_TOP_BOTTOM)
    updated = [(cls, x, 1.0 - y, w, h) for cls, x, y, w, h in boxes]
    return flipped, updated


def brightness_jitter(factor: float) -> AugmentFn:
    def apply(image: Image.Image, boxes: Sequence[NormalizedBox]) -> Tuple[Image.Image, List[NormalizedBox]]:
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor), list(boxes)
    return apply


def color_jitter(factor: float) -> AugmentFn:
    def apply(image: Image.Image, boxes: Sequence[NormalizedBox]) -> Tuple[Image.Image, List[NormalizedBox]]:
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor), list(boxes)
    return apply


@dataclass
class Variant:
    name: str
    apply: AugmentFn
    split: str  # "train" or "val"


def build_variant_plan(copies: int) -> List[Variant]:
    """Construct the augmentation plan.

    Every base image yields:
        - original (train)
        - horizontal flip (train)
        - vertical flip (train)
        - `copies` brightness jitters (train)
        - one darker colour jitter reserved for val split
    """

    variants: List[Variant] = [
        Variant("orig", identity, "train"),
        Variant("hflip", hflip, "train"),
        Variant("vflip", vflip, "train"),
    ]

    # train brightness variations
    for idx in range(copies):
        factor = 1.0 + 0.2 * math.sin((idx + 1) * math.pi / (copies + 1))
        variants.append(Variant(f"bright{idx+1}", brightness_jitter(factor), "train"))

    # validation colour shift
    variants.append(Variant("valcolor", color_jitter(0.8), "val"))
    return variants


# Main pipeline ------------------------------------------------------------

def augment_dataset(output_dir: Path, copies: int) -> None:
    random.seed(RANDOM_SEED)

    class_names = load_class_names(Path(LABELS_FILE))
    class_index_map = {name: idx for idx, name in enumerate(class_names)}
    annotation_map = build_annotation_mapping(ANNOTATIONS_SRC_DIR)

    if not IMAGES_SRC_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGES_SRC_DIR}")

    image_files = sorted([p for p in IMAGES_SRC_DIR.glob("*.png") if p.is_file()])
    if not image_files:
        raise FileNotFoundError(f"No PNG files found in {IMAGES_SRC_DIR}")

    prepare_output_dir(output_dir)
    variants = build_variant_plan(max(0, copies))

    used_names: set[str] = set()
    per_class_val_written: set[int] = set()

    for img_path in image_files:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            base = img_path.stem
            trimmed = base.strip()
            slug = slugify(trimmed)

            annotation_path = annotation_map.get(trimmed)
            boxes = parse_boxes(annotation_path) if annotation_path else []
            # if boxes:
            #     boxes = ensure_class_ids(boxes, class_index_map, trimmed)

            for variant in variants:
                variant_img, variant_boxes = variant.apply(img, boxes)
                target_split = variant.split

                # ensure at least one sample per class in val split
                if target_split == "val" and not variant_boxes:
                    target_split = "train"
                elif target_split == "val":
                    # if val already contains this class, keep remaining in train to avoid tiny val sets
                    val_classes = {cls for cls, *_ in variant_boxes}
                    if per_class_val_written.issuperset(val_classes):
                        target_split = "train"
                    else:
                        per_class_val_written.update(val_classes)

                filename = make_unique_filename(slug, variant.name, used_names)
                save_image_and_label(
                    variant_img,
                    variant_boxes,
                    output_dir,
                    target_split,
                    filename,
                )

    # Fallback: if some classes never made it to val, copy a training sample over
    if per_class_val_written != set(range(len(class_names))):
        mirror_missing_validation(output_dir, class_names, per_class_val_written)

    write_dataset_yaml(output_dir, class_names)


def make_unique_filename(slug: str, suffix: str, used: set[str]) -> str:
    candidate = f"{slug}_{suffix}" if suffix else slug
    base_candidate = candidate
    counter = 1
    while candidate in used:
        candidate = f"{base_candidate}_{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def save_image_and_label(image: Image.Image, boxes: Sequence[NormalizedBox], root: Path, split: str, stem: str) -> None:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_path = image_dir / f"{stem}.png"
    label_path = label_dir / f"{stem}.txt"

    image.save(image_path, format="PNG")
    if boxes:
        label_path.write_text(format_boxes(boxes) + "\n", encoding="utf-8")
    else:
        label_path.write_text("", encoding="utf-8")


def mirror_missing_validation(root: Path, class_names: Sequence[str], seen: set[int]) -> None:
    """If some classes never reached the val split, duplicate a train sample."""
    missing = [idx for idx in range(len(class_names)) if idx not in seen]
    if not missing:
        return

    train_labels = list((root / "labels" / "train").glob("*.txt"))
    if not train_labels:
        return

    for class_idx in missing:
        for label_path in train_labels:
            lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            classes_in_file = {int(line.split()[0]) for line in lines}
            if class_idx in classes_in_file:
                stem = label_path.stem
                src_img = root / "images" / "train" / f"{stem}.png"
                dst_img = root / "images" / "val" / f"{stem}_copy.png"
                dst_lbl = root / "labels" / "val" / f"{stem}_copy.txt"
                if src_img.exists():
                    shutil.copy2(src_img, dst_img)
                    shutil.copy2(label_path, dst_lbl)
                    seen.add(class_idx)
                    break


def write_dataset_yaml(root: Path, class_names: Sequence[str]) -> None:
    yaml_path = Path("dataset.yaml")
    with yaml_path.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated YOLO dataset configuration\n")
        f.write(f"path: {root.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for idx, name in enumerate(class_names):
            f.write(f"  {idx}: {name}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment dataset and rebuild YOLO directory structure")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output dataset directory (default: dataset)")
    parser.add_argument("--copies", type=int, default=2, help="Number of extra brightness jitter copies per image")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    augment_dataset(args.output, max(0, args.copies))
    print(f"Augmented dataset written to: {args.output}")
    print("Updated dataset.yaml to reference the new dataset.")


if __name__ == "__main__":
    main()
