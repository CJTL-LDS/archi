import os
import shutil
from pathlib import Path

def merge_dataset3_originals():
    # Define paths
    project_root = Path(__file__).parent.parent
    dataset3_dir = project_root / "dataset3"
    dataset2_aug_dir = project_root / "dataset2_aug"
    
    # Source directories to scan
    src_images_dirs = [
        dataset3_dir / "images" / "train",
        dataset3_dir / "images" / "val"
    ]
    
    # Destination directories
    dst_images_dir = dataset2_aug_dir / "images" / "val"
    dst_labels_dir = dataset2_aug_dir / "labels" / "val"
    
    # Ensure destination directories exist
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    
    print(f"Scanning for original images in {dataset3_dir}...")
    
    for src_img_dir in src_images_dirs:
        if not src_img_dir.exists():
            continue
            
        # Corresponding label dir (assuming standard YOLO structure)
        # dataset3/images/train -> dataset3/labels/train
        src_lbl_dir = dataset3_dir / "labels" / src_img_dir.name
        
        for img_file in src_img_dir.glob("*_orig.*"):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
                
            # Copy image
            shutil.copy2(img_file, dst_images_dir / img_file.name)
            
            # Find and copy label
            label_name = img_file.stem + ".txt"
            src_label = src_lbl_dir / label_name
            
            if src_label.exists():
                shutil.copy2(src_label, dst_labels_dir / label_name)
            else:
                print(f"Warning: Label not found for {img_file.name}")
                
            count += 1
            
    print(f"Successfully merged {count} original images and labels from dataset3 to dataset2_aug/val.")

if __name__ == "__main__":
    merge_dataset3_originals()
