from ultralytics import YOLO
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model using absolute path
model_path = os.path.join(script_dir, 'Models', 'chip_amount_processing_model.pt')
model = YOLO(model_path)

# Run inference on the test chip image
image_path = os.path.join(script_dir, 'Outputs', 'test_chips.jpg')  # Update with your test chip image if needed
results = model(image_path)

# Extract unique chip amount labels
detected_chip_amounts = set()
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        chip_amount_label = result.names[class_id]
        detected_chip_amounts.add(chip_amount_label)

# Print unique chip amounts
for chip_amount in detected_chip_amounts:
    print(chip_amount)