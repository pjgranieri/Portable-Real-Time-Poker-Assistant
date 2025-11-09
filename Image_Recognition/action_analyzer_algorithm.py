from ultralytics import YOLO
import cv2
import os
import numpy as np

# Try to import MediaPipe, handle different versions
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    
    # Check MediaPipe version to determine API
    try:
        # MediaPipe 0.10+ uses new API
        MEDIAPIPE_NEW_API = hasattr(mp.solutions.hands, 'Hands')
    except:
        MEDIAPIPE_NEW_API = False
        
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_NEW_API = False
    print("[WARNING] MediaPipe not available. Hand detection will be disabled.")

class ActionAnalyzer:
    def __init__(self, confidence_threshold=0.75):
        self.confidence_threshold = confidence_threshold
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
        """Check for chips - with enhanced filtering for false positives"""
        if self.chip_model is None:
            print("[MOCK] No chip model - simulating 2 chips detected")
            return True, ['Chip', 'Chip']
        
        try:
            results = self.chip_model(image_path, conf=self.confidence_threshold, verbose=False)
            
            chip_labels = []
            chip_boxes = []
            
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    
                    # FILTER 1: Only accept "Chip" class
                    if 'chip' not in class_name.lower():
                        continue
                    
                    # FILTER 2: Check aspect ratio (chips should be roughly circular/square)
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    aspect_ratio = width / height if height > 0 else 0
                    
                    # Chips should have aspect ratio between 0.7 and 1.4 (roughly square/circular)
                    if aspect_ratio < 0.7 or aspect_ratio > 1.4:
                        print(f"[FILTER] Ignoring detection - bad aspect ratio: {aspect_ratio:.2f}")
                        continue
                    
                    # FILTER 3: Size check - chips shouldn't be too large or too small
                    area = width * height
                    image = cv2.imread(image_path)
                    if image is not None:
                        img_area = image.shape[0] * image.shape[1]
                        relative_area = area / img_area
                        
                        # ADJUSTED: Chip can be 0.5% to 70% of image area
                        if relative_area < 0.005 or relative_area > 0.70:
                            print(f"[FILTER] Ignoring detection - bad size: {relative_area*100:.2f}% of image")
                            continue
                    
                    # FILTER 4: Color check - chips should have consistent colors (not skin-like)
                    if not self._verify_chip_color(image_path, bbox):
                        print(f"[FILTER] Ignoring detection - skin-like color detected")
                        continue
                    
                    chip_labels.append(class_name)
                    chip_boxes.append(bbox)
            
            print(f"[CHIP DETECTION] Found {len(chip_labels)} valid chips after filtering")
            return len(chip_labels) > 0, chip_labels
            
        except Exception as e:
            print(f"[ERROR] Chip detection failed: {e}")
            return False, []
    
    def _verify_chip_color(self, image_path, bbox):
        """Verify the detected region has chip-like colors, not skin tones"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return True  # Can't verify, assume valid
            
            # Extract the region
            x1, y1, x2, y2 = map(int, bbox)
            region = image[y1:y2, x1:x2]
            
            if region.size == 0:
                return True
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Define skin tone range in HSV
            # Skin tones typically: H=0-25, S=30-170, V=80-255
            lower_skin = np.array([0, 30, 80], dtype=np.uint8)
            upper_skin = np.array([25, 170, 255], dtype=np.uint8)
            
            # Check how much of the region is skin-colored
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_percentage = np.count_nonzero(skin_mask) / skin_mask.size
            
            # If more than 60% is skin-colored, it's probably not a chip
            if skin_percentage > 0.6:
                print(f"[COLOR CHECK] {skin_percentage*100:.1f}% skin-tone pixels detected")
                return False
            
            return True
            
        except Exception as e:
            print(f"[WARNING] Color verification failed: {e}")
            return True  # If verification fails, assume valid
    
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