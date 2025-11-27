import os
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
import yaml
from tqdm import tqdm

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_yaml(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

def augment_scale_down(img, bboxes, scale_range=(0.3, 0.7)):
    # Zoom out: Resize image down and place on gray canvas of original size
    scale_factor = random.uniform(*scale_range)
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    resized_img = cv2.resize(img, (new_w, new_h))
    
    # Create canvas
    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
    
    # Random placement on canvas
    max_dx = w - new_w
    max_dy = h - new_h
    dx = random.randint(0, max_dx)
    dy = random.randint(0, max_dy)
    
    canvas[dy:dy+new_h, dx:dx+new_w] = resized_img
    
    new_bboxes = []
    for cls, xc, yc, wn, hn in bboxes:
        xc_new = (xc * w * scale_factor + dx) / w
        yc_new = (yc * h * scale_factor + dy) / h
        wn_new = wn * scale_factor
        hn_new = hn * scale_factor
        
        new_bboxes.append([cls, xc_new, yc_new, wn_new, hn_new])
        
    return canvas, new_bboxes

def augment_scale_up(img, bboxes, scale_range=(1.5, 2.0)):
    # Zoom in: Resize image up and crop a random area to original size
    scale_factor = random.uniform(*scale_range)
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    resized_img = cv2.resize(img, (new_w, new_h))
    
    # Random crop
    max_dx = new_w - w
    max_dy = new_h - h
    dx = random.randint(0, max_dx)
    dy = random.randint(0, max_dy)
    
    cropped_img = resized_img[dy:dy+h, dx:dx+w]
    
    new_bboxes = []
    for cls, xc, yc, wn, hn in bboxes:
        # Scaled pixel coords
        x1_s = (xc - wn/2) * w * scale_factor
        x2_s = (xc + wn/2) * w * scale_factor
        y1_s = (yc - hn/2) * h * scale_factor
        y2_s = (yc + hn/2) * h * scale_factor
        
        # Intersect with crop window [dx, dy, dx+w, dy+h]
        c_x1, c_y1 = dx, dy
        c_x2, c_y2 = dx + w, dy + h
        
        i_x1 = max(x1_s, c_x1)
        i_y1 = max(y1_s, c_y1)
        i_x2 = min(x2_s, c_x2)
        i_y2 = min(y2_s, c_y2)
        
        if i_x2 > i_x1 and i_y2 > i_y1:
            # Box exists in crop
            box_w = i_x2 - i_x1
            box_h = i_y2 - i_y1
            
            # Filter tiny boxes (e.g. < 5 pixels)
            if box_w < 5 or box_h < 5:
                continue

            box_cx = i_x1 + box_w / 2 - dx
            box_cy = i_y1 + box_h / 2 - dy
            
            xc_new = box_cx / w
            yc_new = box_cy / h
            wn_new = box_w / w
            hn_new = box_h / h
            
            new_bboxes.append([cls, xc_new, yc_new, wn_new, hn_new])
            
    return cropped_img, new_bboxes

def augment_rotate(img, bboxes, angle_range=(-30, 30)):
    angle = random.uniform(*angle_range)
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    # New bounding dimensions
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    # Adjust translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    
    rotated_img = cv2.warpAffine(img, M, (nW, nH), borderValue=(114, 114, 114))
    
    new_bboxes = []
    for cls, xc, yc, wn, hn in bboxes:
        # Denormalize
        cx_box = xc * w
        cy_box = yc * h
        w_box = wn * w
        h_box = hn * h
        
        x1 = cx_box - w_box / 2
        y1 = cy_box - h_box / 2
        x2 = cx_box + w_box / 2
        y2 = cy_box + h_box / 2
        
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ])
        
        ones = np.ones(shape=(len(corners), 1))
        corners_ones = np.hstack([corners, ones])
        
        transformed_corners = M.dot(corners_ones.T).T
        
        x_coords = transformed_corners[:, 0]
        y_coords = transformed_corners[:, 1]
        
        new_x1 = np.min(x_coords)
        new_y1 = np.min(y_coords)
        new_x2 = np.max(x_coords)
        new_y2 = np.max(y_coords)
        
        # Clip
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(nW, new_x2)
        new_y2 = min(nH, new_y2)
        
        if new_x2 <= new_x1 or new_y2 <= new_y1:
            continue
            
        new_w_box = new_x2 - new_x1
        new_h_box = new_y2 - new_y1
        new_cx_box = new_x1 + new_w_box / 2
        new_cy_box = new_y1 + new_h_box / 2
        
        new_bboxes.append([
            cls,
            new_cx_box / nW,
            new_cy_box / nH,
            new_w_box / nW,
            new_h_box / nH
        ])
        
    return rotated_img, new_bboxes

def main():
    source_root = Path("dataset2")
    target_root = Path("dataset2_aug")
    
    if target_root.exists():
        shutil.rmtree(target_root)
    
    # Copy structure
    shutil.copytree(source_root, target_root)
    
    train_img_dir = target_root / "images" / "train"
    train_lbl_dir = target_root / "labels" / "train"
    
    images = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))
    
    print(f"Augmenting {len(images)} training images...")
    
    for img_path in tqdm(images):
        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            
        bboxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                bboxes.append([int(parts[0])] + [float(x) for x in parts[1:]])
        
        # Generate multiple augmented versions
        
        # 1. Scale Down (2 versions)
        for i in range(2):
            aug_img, aug_lbls = augment_scale_down(img, bboxes, scale_range=(0.3, 0.7))
            save_name = f"{img_path.stem}_down_{i}.jpg"
            cv2.imwrite(str(train_img_dir / save_name), aug_img)
            with open(train_lbl_dir / f"{img_path.stem}_down_{i}.txt", 'w') as f:
                for b in aug_lbls:
                    f.write(f"{int(b[0])} {' '.join(f'{x:.6f}' for x in b[1:])}\n")

        # 2. Scale Up (2 versions)
        for i in range(2):
            aug_img, aug_lbls = augment_scale_up(img, bboxes, scale_range=(1.5, 2.0))
            save_name = f"{img_path.stem}_up_{i}.jpg"
            cv2.imwrite(str(train_img_dir / save_name), aug_img)
            with open(train_lbl_dir / f"{img_path.stem}_up_{i}.txt", 'w') as f:
                for b in aug_lbls:
                    f.write(f"{int(b[0])} {' '.join(f'{x:.6f}' for x in b[1:])}\n")
                    
        # 3. Rotate (2 versions)
        for i in range(2):
            aug_img, aug_lbls = augment_rotate(img, bboxes, angle_range=(-30, 30))
            save_name = f"{img_path.stem}_rot_{i}.jpg"
            cv2.imwrite(str(train_img_dir / save_name), aug_img)
            with open(train_lbl_dir / f"{img_path.stem}_rot_{i}.txt", 'w') as f:
                for b in aug_lbls:
                    f.write(f"{int(b[0])} {' '.join(f'{x:.6f}' for x in b[1:])}\n")

    # Update dataset2.yaml
    orig_yaml = load_yaml("dataset2.yaml")
    orig_yaml['path'] = str(target_root.as_posix())
    save_yaml(orig_yaml, "dataset2_aug.yaml")
    
    print(f"Augmentation complete. New dataset at {target_root}")
    print(f"New config saved to dataset2_aug.yaml")

if __name__ == "__main__":
    main()
