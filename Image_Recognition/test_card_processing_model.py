from ultralytics import YOLO
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model using absolute path
model_path = os.path.join(script_dir, 'Models', 'card_processing_model.pt')
model = YOLO(model_path)

# Run inference on the test card image
image_path = os.path.join(script_dir, 'Outputs', 'test_two_cards.jpg')
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