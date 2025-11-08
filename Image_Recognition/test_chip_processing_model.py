from ultralytics import YOLO
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model using absolute path
model_path = os.path.join(script_dir, 'Models', 'chip_processing_model.pt')
model = YOLO(model_path)

# Run inference on the test chip image
image_path = os.path.join(script_dir, 'Outputs', 'test_env_three.jpg')  # Update with your test chip image
results = model(image_path)

# Extract unique chip labels
detected_chips = set()
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        chip_label = result.names[class_id]
        detected_chips.add(chip_label)

# Print unique chips
for chip in detected_chips:
    print(chip)