from ultralytics import YOLO
import os
import glob

# Get the directory where THIS SCRIPT is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the PROJECT ROOT (one level up from Image_Recognition)
project_root = os.path.dirname(script_dir)

# Load the model using absolute path (inside Image_Recognition)
model_path = os.path.join(script_dir, 'Models', 'chip_processing_model.pt')
model = YOLO(model_path)

# Outputs folder is at project root level
outputs_dir = os.path.join(project_root, 'Outputs')

if not os.path.exists(outputs_dir):
    print(f"ERROR: Outputs directory not found: {outputs_dir}")
    exit(1)

available_images = glob.glob(os.path.join(outputs_dir, '*.jpg'))

if not available_images:
    print(f"ERROR: No .jpg images found in {outputs_dir}")
    print(f"Please add a test image to the Outputs folder")
    exit(1)

print(f"Available images in {outputs_dir}:")
for i, img in enumerate(available_images, 1):
    print(f"  {i}. {os.path.basename(img)}")

# Use the first available image
image_path = available_images[0]
print(f"\nUsing image: {os.path.basename(image_path)}")

# Run inference on the test chip image
results = model(image_path)

# Extract unique chip labels
detected_chips = set()
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        chip_label = result.names[class_id]
        detected_chips.add(chip_label)

# Print results
print(f"\nDetected chips: {len(detected_chips)}")
for chip in detected_chips:
    print(f"  - {chip}")