from ultralytics import YOLO

import os
import argparse


def main(args) -> None:
    weights_path = args.weight
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Error: Weights file not found at {weights_path}")
        
    print(f"Loading model from {weights_path}...")
    model = YOLO(weights_path)

    source_dir = args.test_dir
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Error: Source directory '{source_dir}' not found.")

    print(f"Running inference on images in '{source_dir}'...")
    
    conf_thres = float(args.confidence)
    results = model.predict(
        source=source_dir, 
        save=True, 
        conf=conf_thres
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_dir", default="./raw/hide", help="测试的数据集")
    parser.add_argument("--weight", default="./runs/detect/train12/weights/best.pt", help="测试用的权重")
    parser.add_argument("--confidence", default=0.2, help="置信度阈值")

    args = parser.parse_args()

    main(args)
