from ultralytics import YOLO
import os
import random
from pathlib import Path

# Get project root (3 levels up from tests/unit/image_recognition/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '../../..')

# Load the player action model
model_path = os.path.join(project_root, 'Image_Recognition', 'Models', 'player_action_model.pt')
model = YOLO(model_path)

print(f"\n{'='*80}")
print("PLAYER ACTION MODEL TEST")
print(f"{'='*80}\n")
print(f"Model: {model_path}")

# Test on images from FoldedCardsDataset, HandDataset, and ChipDataset
dataset_root = Path(project_root) / "CVDataset"

test_images = []

# Get some folded card images
folded_cards_folder = dataset_root / "FoldedCardsDataset"
if folded_cards_folder.exists():
    images = list(folded_cards_folder.glob("*.jpg"))[:5]
    test_images.extend([("FOLD", img) for img in images])

# Get some hand images
hand_folder = dataset_root / "HandDataset"
if hand_folder.exists():
    images = list(hand_folder.glob("*.jpg"))[:5]
    test_images.extend([("CHECK", img) for img in images])

# Get some chip images (from action zone)
chip_folder = dataset_root / "ChipDataset"
if chip_folder.exists():
    images = list(chip_folder.glob("*.jpg"))[:5]
    test_images.extend([("BET/RAISE", img) for img in images])

# Shuffle
random.shuffle(test_images)

print(f"\nTesting on {len(test_images)} images from different action datasets:\n")

for idx, (expected_action, image_path) in enumerate(test_images, 1):
    print(f"\n[{idx}] Testing: {image_path.parent.name}/{image_path.name}")
    print(f"  Expected Action: {expected_action}")

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
