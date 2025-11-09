from ultralytics import YOLO
import cv2
import os
import numpy as np

# Try to import MediaPipe, handle different versions
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        MEDIAPIPE_AVAILABLE = True
    else:
        # Newer MediaPipe API
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        MEDIAPIPE_AVAILABLE = True
        MEDIAPIPE_NEW_API = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_NEW_API = False
    print("[WARNING] MediaPipe not available. Hand detection will be disabled.")

class ActionAnalyzer:
    def __init__(self):
        # Get the directory where this script is located
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load YOLO models
        chip_model_path = os.path.join(self.script_dir, 'Models', 'chip_processing_model.pt')
        chip_amount_model_path = os.path.join(self.script_dir, 'Models', 'chip_amount_processing_model.pt')
        
        self.chip_model = YOLO(chip_model_path)
        self.chip_amount_model = YOLO(chip_amount_model_path)
        
        # Initialize MediaPipe Hand Detection if available
        self.hands = None
        if MEDIAPIPE_AVAILABLE:
            try:
                if not MEDIAPIPE_NEW_API:
                    # Old API
                    self.hands = mp_hands.Hands(
                        static_image_mode=True,
                        max_num_hands=2,
                        min_detection_confidence=0.5
                    )
                else:
                    # For newer API, we'll skip hand detection for now
                    print("[WARNING] MediaPipe new API detected. Hand detection simplified.")
                    self.hands = None
            except Exception as e:
                print(f"[WARNING] Could not initialize MediaPipe hands: {e}")
                self.hands = None
    
    def check_chips(self, image_path):
        """Check if chips are present in the image"""
        results = self.chip_model(image_path, verbose=False)
        
        detected_chips = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                chip_label = result.names[class_id]
                detected_chips.append(chip_label)
        
        return len(detected_chips) > 0, detected_chips
    
    def check_chip_amounts(self, image_path):
        """Check chip amounts in the image"""
        results = self.chip_amount_model(image_path, verbose=False)
        
        detected_amounts = set()
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                amount_label = result.names[class_id]
                detected_amounts.add(amount_label)
        
        return detected_amounts
    
    def check_folded_cards(self, image_path):
        """Check for folded cards (blue/white rectangles or flat cards)"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return False
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for blue and white
        # Blue range
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # White range
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(blue_mask, white_mask)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for rectangular shapes (folded cards)
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Check if contour is card-like (rectangular, certain size)
            area = cv2.contourArea(contour)
            if area > 500 and 0.5 < aspect_ratio < 2.0:  # Adjust thresholds as needed
                return True
        
        return False
    
    def check_hand_present(self, image_path):
        """Check if a hand is present using MediaPipe or simple skin detection"""
        if not MEDIAPIPE_AVAILABLE or self.hands is None:
            # Fallback: Use simple skin color detection
            return self.check_hand_simple(image_path)
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image with MediaPipe
            results = self.hands.process(image_rgb)
            
            # Check if hands are detected
            if results.multi_hand_landmarks:
                return True
            
            return False
        except Exception as e:
            print(f"[WARNING] MediaPipe hand detection failed: {e}")
            return self.check_hand_simple(image_path)
    
    def check_hand_simple(self, image_path):
        """Simple hand detection using skin color detection"""
        image = cv2.imread(image_path)
        if image is None:
            return False
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin color
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if any significant skin-colored region exists
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Threshold for hand size
                return True
        
        return False
    
    def analyze_action(self, image_path):
        """
        Main analysis function that checks for poker actions in order:
        1. Chips (betting/raising)
        2. Folded cards
        3. Hand tapping (checking)
        
        Returns: dict with action type and details
        """
        result = {
            'action': None,
            'details': {}
        }
        
        # Step 1: Check for chips
        chips_present, chip_labels = self.check_chips(image_path)
        
        if chips_present:
            # If chips detected, analyze chip amounts
            chip_amounts = self.check_chip_amounts(image_path)
            result['action'] = 'BET/RAISE'
            result['details']['chips_detected'] = chip_labels
            result['details']['chip_amounts'] = list(chip_amounts) if chip_amounts else []
            return result
        
        # Step 2: Check for folded cards
        folded_cards = self.check_folded_cards(image_path)
        
        if folded_cards:
            result['action'] = 'FOLD'
            result['details']['folded_cards_detected'] = True
            return result
        
        # Step 3: Check for hand (checking motion)
        hand_present = self.check_hand_present(image_path)
        
        if hand_present:
            result['action'] = 'CHECK'
            result['details']['hand_detected'] = True
            return result
        
        # No action detected
        result['action'] = 'NO_ACTION'
        return result
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'hands') and self.hands is not None:
            try:
                self.hands.close()
            except:
                pass


# Example usage
if __name__ == "__main__":
    analyzer = ActionAnalyzer()
    
    # Test with an image
    test_image = os.path.join(os.path.dirname(__file__), 'Outputs', 'test_env_three.jpg')
    
    if os.path.exists(test_image):
        result = analyzer.analyze_action(test_image)
        
        print(f"Action detected: {result['action']}")
        print(f"Details: {result['details']}")
    else:
        print(f"Test image not found: {test_image}")