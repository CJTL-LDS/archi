from pathlib import Path

labels_file = Path("../labels.txt")
yaml_file = Path("../dataset3.yaml")

with labels_file.open("r", encoding="utf-8") as f:
    names = [line.strip() for line in f if line.strip()]

with yaml_file.open("w", encoding="utf-8") as f:
    f.write("# Auto-generated YOLO dataset configuration\n")
    f.write("path: dataset3\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("names:\n")
    for idx, name in enumerate(names):
        f.write(f"  {idx}: {name}\n")

print(f"Created {yaml_file}")
