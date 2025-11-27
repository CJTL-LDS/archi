import os
import shutil
from pathlib import Path

def merge_dataset3_train():
    # Define paths
    project_root = Path(__file__).parent.parent
    dataset3_dir = project_root / "dataset3"
    dataset2_aug_dir = project_root / "dataset2_aug"
    
    # Source directories
    src_images_dir = dataset3_dir / "images" / "train"
    src_labels_dir = dataset3_dir / "labels" / "train"
    
    # Destination directories
    dst_images_dir = dataset2_aug_dir / "images" / "train"
    dst_labels_dir = dataset2_aug_dir / "labels" / "train"
    
    # Ensure destination directories exist
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    
    print(f"Merging training data from {dataset3_dir} to {dataset2_aug_dir}...")
    
    # Copy images
    if src_images_dir.exists():
        for img_file in src_images_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                shutil.copy2(img_file, dst_images_dir / img_file.name)
                
                # Copy corresponding label if it exists
                label_name = img_file.stem + ".txt"
                src_label = src_labels_dir / label_name
                if src_label.exists():
                    shutil.copy2(src_label, dst_labels_dir / label_name)
                
                count += 1
            
    print(f"Successfully merged {count} images and labels from dataset3/train to dataset2_aug/train.")

if __name__ == "__main__":
    merge_dataset3_train()
