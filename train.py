"""
train YOLO11 detection model on the prepared custom dataset.

directory structure under `dataset/`:
    dataset/
        images/train/*.png
        images/val/*.png
        labels/train/*.txt
        labels/val/*.txt
    dataset.yaml
"""

from ultralytics import YOLO

import os
import argparse


def main(args) -> None:

    dataset_yaml = args.dataset
    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"{dataset_yaml} not found. Ensure dataset builder script has generated it.")

    weight_path = args.weight
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Pretrained weights not found at {weight_path}. Update `YOLO_WEIGHTS` or train from scratch.")

    model = YOLO(weight_path)
    model.train(
        data=dataset_yaml,
        epochs=int(args.epochs),
        batch=args.batch_size,
        imgsz=args.img_size,
        device=os.environ.get("YOLO_DEVICE") if os.environ.get("YOLO_DEVICE") else 0 if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu",
        resume=bool(int(os.environ.get("YOLO_RESUME", "0"))),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="./dataset2.yaml", help="要用的dataset的配置文件（path）")
    parser.add_argument("--weight", default="./yolo11n.pt", help="要训练的初始权重（path）")
    parser.add_argument("--epochs", default=500, help="训练的轮数")
    parser.add_argument("--batch_size", default=4, help="batch size")
    parser.add_argument("--img_size", default=640, help="训练图片大小（一般不用改）")

    args = parser.parse_args()

    main(args)