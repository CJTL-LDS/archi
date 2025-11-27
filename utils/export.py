import os
from pathlib import Path

from ultralytics import YOLO


def str_to_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def main() -> None:
    weight_path = Path(os.environ.get("YOLO_WEIGHTS", "../runs/detect/train12/weights/best.pt"))
    if not weight_path.exists():
        raise FileNotFoundError(f"Weights not found at {weight_path}")

    project_dir = Path("../exports")
    project_dir.mkdir(parents=True, exist_ok=True)

    export_name = "train9"
    imgsz = int(os.environ.get("YOLO_IMGSZ", "640"))

    export_kwargs = {
        "format": "onnx",
        "imgsz": imgsz,
        "project": str(project_dir),
        "name": export_name,
    }

    
    export_kwargs["opset"] = 11
    if "YOLO_ONNX_DYNAMIC" in os.environ:
        export_kwargs["dynamic"] = str_to_bool(os.environ["YOLO_ONNX_DYNAMIC"])

    model = YOLO(str(weight_path))
    onnx_path = model.export(**export_kwargs)
    print(f"Exported ONNX model to: {onnx_path}")


if __name__ == "__main__":
    main()
