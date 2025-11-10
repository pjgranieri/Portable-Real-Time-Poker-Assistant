from ultralytics import YOLO
import cv2
import os
import numpy as np

# Try to import MediaPipe, handle different versions
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    mp_hands = mp.solutions.hands
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[WARNING] MediaPipe not available. Hand detection will be disabled.")

class ActionAnalyzer:
    def __init__(self, confidence_threshold=0.9):
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
                self.hands = mp_hands.Hands(
                    static_image_mode=True,
                    max_num_hands=2,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7
                )
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
            
            chip_detections = []  # Store (confidence, bbox, class_name)
            
            image = cv2.imread(image_path)
            if image is None:
                return False, []
            
            img_height, img_width = image.shape[:2]
            img_area = img_height * img_width
            
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
                    
                    # TIGHTENED: Chips should have aspect ratio between 0.8 and 1.25
                    if aspect_ratio < 0.8 or aspect_ratio > 1.25:
                        print(f"[FILTER] Ignoring detection - bad aspect ratio: {aspect_ratio:.2f}")
                        continue
                    
                    # FILTER 3: Size check - chips shouldn't be too large or too small
                    area = width * height
                    relative_area = area / img_area
                    
                    # TIGHTENED: Chip can be 1% to 40% of image area
                    if relative_area < 0.01 or relative_area > 0.40:
                        print(f"[FILTER] Ignoring detection - bad size: {relative_area*100:.2f}% of image")
                        continue
                    
                    # FILTER 4: Position check - chips usually in center/lower area
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    norm_x = center_x / img_width
                    
                    # NEW: Chips typically in middle 70% horizontally
                    if norm_x < 0.15 or norm_x > 0.85:
                        print(f"[FILTER] Ignoring detection - too far horizontally: {norm_x:.2f}")
                        continue
                    
                    # FILTER 5: Enhanced color check - chips should have consistent colors (not skin-like)
                    if not self._verify_chip_color_enhanced(image, bbox):
                        print(f"[FILTER] Ignoring detection - failed color verification")
                        continue
                    
                    # FILTER 6: Texture analysis
                    if not self._verify_chip_texture(image, bbox):
                        print(f"[FILTER] Ignoring detection - failed texture verification")
                        continue
                    
                    chip_detections.append((confidence, bbox, class_name))
            
            # FILTER 7: Non-maximum suppression
            chip_detections = self._non_maximum_suppression(chip_detections, iou_threshold=0.5)
            
            chip_labels = [det[2] for det in chip_detections]
            
            print(f"[CHIP DETECTION] Found {len(chip_labels)} valid chips after filtering")
            return len(chip_labels) > 0, chip_labels
            
        except Exception as e:
            print(f"[ERROR] Chip detection failed: {e}")
            return False, []
    
    def _verify_chip_color_enhanced(self, image, bbox):
        """Enhanced color verification to avoid skin tones and ensure chip-like colors"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            region = image[y1:y2, x1:x2]
            
            if region.size == 0:
                return False
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Check 1: Reject skin tones (dual range)
            lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
            
            lower_skin2 = np.array([0, 10, 60], dtype=np.uint8)
            upper_skin2 = np.array([25, 170, 255], dtype=np.uint8)
            skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            
            skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
            skin_percentage = np.count_nonzero(skin_mask) / skin_mask.size
            
            # TIGHTENED: If more than 40% is skin-colored, reject
            if skin_percentage > 0.40:
                print(f"[COLOR CHECK] {skin_percentage*100:.1f}% skin-tone pixels detected")
                return False
            
            # Check 2: Color variance (chips should have relatively solid colors)
            hsv_std = np.std(hsv, axis=(0, 1))
            if hsv_std[0] > 40:  # Hue variance
                return False
            
            # Check 3: Brightness consistency
            v_channel = hsv[:, :, 2]
            v_std = np.std(v_channel)
            if v_std > 60:
                return False
            
            return True
            
        except Exception as e:
            print(f"[WARNING] Color verification failed: {e}")
            return False
    
    def _verify_chip_texture(self, image, bbox):
        """Verify chip texture - chips have distinctive circular patterns/edges"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            region = image[y1:y2, x1:x2]
            
            if region.size == 0:
                return False
            
            # Convert to grayscale
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            edge_percentage = np.count_nonzero(edges) / edges.size
            
            # Chips typically have 5-50% edge pixels
            if edge_percentage < 0.05 or edge_percentage > 0.50:
                return False
            
            # Check for circular shapes
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=50,
                param2=30,
                minRadius=int(min(gray.shape) * 0.2),
                maxRadius=int(max(gray.shape) * 0.6)
            )
            
            # If we detect circular patterns or reasonable edge percentage, accept
            if circles is not None or (0.08 < edge_percentage < 0.35):
                return True
            
            return False
            
        except Exception as e:
            print(f"[WARNING] Texture verification failed: {e}")
            return True
    
    def _non_maximum_suppression(self, detections, iou_threshold=0.5):
        """Remove overlapping detections"""
        if len(detections) == 0:
            return []
        
        detections = sorted(detections, key=lambda x: x[0], reverse=True)
        keep = []
        
        while len(detections) > 0:
            best = detections[0]
            keep.append(best)
            detections = detections[1:]
            
            filtered = []
            for det in detections:
                iou = self._calculate_iou(best[1], det[1])
                if iou < iou_threshold:
                    filtered.append(det)
            
            detections = filtered
        
        return keep
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def check_chip_amounts(self, image_path):
        """Check chip amounts in the image"""
        try:
            results = self.chip_amount_model(image_path, verbose=False)
            
            detected_amounts = set()
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    amount_label = result.names[class_id]
                    detected_amounts.add(amount_label)
            
            return detected_amounts
        except Exception as e:
            print(f"[ERROR] Chip amount detection failed: {e}")
            return set()
    
    def check_folded_cards(self, image_path):
        """Check for folded cards (blue/white rectangles or flat cards)"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            img_height, img_width = image.shape[:2]
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Enhanced blue detection for card backs (multiple ranges)
            lower_blue1 = np.array([90, 50, 50])
            upper_blue1 = np.array([130, 255, 255])
            blue_mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
            
            lower_blue2 = np.array([100, 30, 30])
            upper_blue2 = np.array([140, 255, 200])
            blue_mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
            
            blue_mask = cv2.bitwise_or(blue_mask1, blue_mask2)
            
            # White range
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 40, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Combine masks
            combined_mask = cv2.bitwise_or(blue_mask, white_mask)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check for rectangular shapes (folded cards)
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Cards should be 1-30% of image
                relative_area = area / (img_height * img_width)
                if relative_area < 0.01 or relative_area > 0.30:
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Playing cards: aspect ratio 0.5-1.5 (allows some rotation)
                if not (0.5 < aspect_ratio < 1.5):
                    continue
                
                # Check rectangularity
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                
                # Rectangles have 4 corners (allow 4-6 for some imperfection)
                if len(approx) >= 4 and len(approx) <= 6:
                    return True
            
            return False
            
        except Exception as e:
            print(f"[ERROR] Fold detection failed: {e}")
            return False
    
    def check_hand_present(self, image_path):
        """Check if a hand is present using MediaPipe or simple skin detection"""
        if not MEDIAPIPE_AVAILABLE or self.hands is None:
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
                # Additional verification: check if hand landmarks are reasonable
                for hand_landmarks in results.multi_hand_landmarks:
                    if len(hand_landmarks.landmark) == 21:
                        if self._verify_hand_landmarks(hand_landmarks, image.shape):
                            return True
                return True  # If landmarks exist but verification failed, still possible
            
            return False
        except Exception as e:
            print(f"[WARNING] MediaPipe hand detection failed: {e}")
            return self.check_hand_simple(image_path)
    
    def _verify_hand_landmarks(self, hand_landmarks, img_shape):
        """Verify that hand landmarks are in reasonable positions"""
        try:
            img_height, img_width = img_shape[:2]
            
            # Get key landmarks
            wrist = hand_landmarks.landmark[0]
            middle_tip = hand_landmarks.landmark[12]
            
            # Convert to pixel coordinates
            wrist_pos = (wrist.x * img_width, wrist.y * img_height)
            middle_pos = (middle_tip.x * img_width, middle_tip.y * img_height)
            
            # Calculate hand length
            hand_length = np.sqrt(
                (middle_pos[0] - wrist_pos[0])**2 + 
                (middle_pos[1] - wrist_pos[1])**2
            )
            
            # Hand should be 5-50% of image diagonal
            img_diagonal = np.sqrt(img_width**2 + img_height**2)
            relative_size = hand_length / img_diagonal
            
            if relative_size < 0.05 or relative_size > 0.50:
                return False
            
            return True
            
        except Exception as e:
            return True
    
    def check_hand_simple(self, image_path):
        """Simple hand detection using skin color detection"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            img_height, img_width = image.shape[:2]
            img_area = img_height * img_width
            
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define skin color range in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create mask for skin color
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check if any significant skin-colored region exists
            for contour in contours:
                area = cv2.contourArea(contour)
                relative_area = area / img_area
                
                # Hand should be 2-40% of image
                if 0.02 < relative_area < 0.40:
                    # Additional check: hand-like shape
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Hands are not too circular (0.15-0.75)
                    if 0.15 < circularity < 0.75:
                        return True
            
            return False
            
        except Exception as e:
            print(f"[ERROR] Simple hand detection failed: {e}")
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