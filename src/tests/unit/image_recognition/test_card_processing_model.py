from ultralytics import YOLO
import os

# Get project root (3 levels up from tests/unit/image_recognition/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '../../..')

# Load the model using absolute path from project root
model_path = os.path.join(project_root, 'Image_Recognition', 'Models', 'card_processing_model.pt')
model = YOLO(model_path)

# Run inference on the test card image
image_path = os.path.join(project_root, 'Outputs', 'test_two_cards.jpg')
results = model(image_path)

# Extract unique card labels
detected_cards = set()
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        card_label = result.names[class_id]
        detected_cards.add(card_label)

# Print unique cards
for card in detected_cards:
    print(card)