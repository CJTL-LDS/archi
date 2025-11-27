from ultralytics import YOLO
import os

# Path to the ONNX model
model_path = r'../runs/detect/train7/weights/best.onnx'

# Paths to the images
# image_dir = r'raw/pictures'
# Pick the first two images found in the directory to ensure they exist
# all_files = os.listdir(image_dir)
# image_files = [os.path.join(image_dir, f) for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:2]

image_files = [
    r'../dataset/images/val/太极拳_valcolor.png',
    r'../dataset/images/val/泥塑_valcolor.png',
    r'../dataset/images/val/舞被狮_valcolor.png'
]

if not image_files:
    print(f"No images found")
    exit(1)

print(f"Testing with images: {image_files}")

# Check if model exists
if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    exit(1)

# Load the model
print(f"Loading model from {model_path}...")
# Note: When loading a non-pt model (like onnx), we instantiate YOLO with the file path.
# Ultralytics handles the backend (onnxruntime) automatically.
model = YOLO(model_path, task='detect')

import cv2
import numpy as np

# Function to load image with Chinese path
def imread_chinese(file_path):
    try:
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Run inference
print("Running inference...")
# Run inference on images one by one to avoid batch size issues if model has fixed batch size
results = []
for img_path in image_files:
    print(f"Processing {img_path}...")
    
    # Load image manually to handle Chinese paths
    img = imread_chinese(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        continue
        
    print(f"Image shape: {img.shape}")

    # Run inference on a single image
    # We wrap it in a list to get a list of results, though model(img_path) returns a list of 1 result usually
    # Lower confidence to see if anything is detected
    res = model(img, conf=0.05) 
    
    # Manually set the path attribute so result.save() might work or for our reference
    for r in res:
        r.path = img_path
        
    results.extend(res)

# Process results
for i, result in enumerate(results):
    img_path = image_files[i]
    print(f"\nResult for image: {img_path}")
    
    # Print detected boxes
    if len(result.boxes) == 0:
        print("  No objects detected.")
    else:
        for box in result.boxes:
            class_id = int(box.cls)
            conf = float(box.conf)
            # Get class name if available, otherwise use ID
            class_name = result.names[class_id] if result.names else str(class_id)
            print(f"  Class: {class_name} ({conf:.2f})")
    
    # Save result image
    # result.save() saves to runs/detect/predict... by default, 
    # but we can also plot and save manually or let it save to default location.
    # Let's use the built-in save method which saves to the project run directory.
    saved_path = result.save(filename=f'inference_result_{i}.jpg')
    print(f"  Saved visualization to {saved_path}")

print("\nDone.")
