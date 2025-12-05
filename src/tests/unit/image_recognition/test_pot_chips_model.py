from ultralytics import YOLO
import os
import random
from pathlib import Path

# Get project root (3 levels up from tests/unit/image_recognition/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '../../..')

# Load the pot chips model
model_path = os.path.join(project_root, 'Image_Recognition', 'Models', 'pot_chips_model.pt')
model = YOLO(model_path)

print(f"\n{'='*80}")
print("POT CHIPS MODEL TEST")
print(f"{'='*80}\n")
print(f"Model: {model_path}")

# Test on a few chip images from CVDataset
dataset_root = Path(project_root) / "CVDataset"

# Get some random chip images
test_images = []
for color in ["BlackChips", "BlueChips", "RedChips"]:
    for count in ["1Chip", "2Chips", "3Chips", "4Chips", "5Chips"]:
        folder = dataset_root / color / count
        if folder.exists():
            images = list(folder.glob("*.jpg"))[:2]  # Take 2 from each
            test_images.extend(images)

# Shuffle and take 10 random images
random.shuffle(test_images)
test_images = test_images[:10]

print(f"\nTesting on {len(test_images)} random chip images:\n")

for idx, image_path in enumerate(test_images, 1):
    print(f"\n[{idx}] Testing: {image_path.parent.name}/{image_path.name}")

    # Run inference
    results = model(str(image_path), conf=0.25, verbose=False)

    # Extract detections
    detected_objects = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = result.names[class_id]
            confidence = float(box.conf[0])
            detected_objects.append((label, confidence))

    if detected_objects:
        print(f"  ✓ Detected {len(detected_objects)} object(s):")
        for label, conf in detected_objects:
            print(f"    - {label} (confidence: {conf:.2f})")
    else:
        print(f"  ✗ NO DETECTIONS")

print(f"\n{'='*80}")
print("TEST COMPLETE")
print(f"{'='*80}\n")

# Also show what classes the model knows
print("Model Classes:")
print(model.names)
