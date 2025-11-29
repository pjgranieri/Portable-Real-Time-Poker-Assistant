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
    def __init__(self, confidence_threshold=0.50):
        self.confidence_threshold = confidence_threshold
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load YOLO models
        # NEW: Use player_action_model.pt for detecting chips/folded cards in player action zone
        player_action_model_path = os.path.join(self.script_dir, 'Models', 'player_action_model.pt')
        chip_amount_model_path = os.path.join(self.script_dir, 'Models', 'chip_amount_processing_model.pt')

        self.chip_model = YOLO(player_action_model_path)  # Updated to use player_action_model

        # Load chip amount model (optional, only if file exists)
        if os.path.exists(chip_amount_model_path):
            self.chip_amount_model = YOLO(chip_amount_model_path)
        else:
            self.chip_amount_model = None
            print("[WARNING] chip_amount_processing_model.pt not found. Chip amount detection disabled.")
        
        # CONFIDENCE THRESHOLDS
        self.fold_confidence_threshold = 0.20  # Lowered to improve detection
        self.check_confidence_threshold = 0.40
        
        # ROI CROPPING - Use 65% to include more of the table
        self.table_crop_factor = 0.35  # Start from 35% down (include bottom 65%)
        
        # Initialize MediaPipe
        self.hands = None
        if MEDIAPIPE_AVAILABLE:
            try:
                self.hands = mp_hands.Hands(
                    static_image_mode=True,
                    max_num_hands=2,
                    min_detection_confidence=0.7,  # Increased from 0.3 to reduce false positives
                    min_tracking_confidence=0.7    # Increased from 0.3 to reduce false positives
                )
                print("[INFO] MediaPipe Hands initialized successfully")
            except Exception as e:
                print(f"[WARNING] Could not initialize MediaPipe hands: {e}")
                self.hands = None
    
    def _verify_hand_landmarks(self, hand_landmarks, image_shape):
        """Verify that hand landmarks are valid and visible"""
        if hand_landmarks is None or len(hand_landmarks.landmark) != 21:
            return False
        
        height, width = image_shape[:2]
        
        # Count how many landmarks are in bounds
        valid_landmarks = 0
        for landmark in hand_landmarks.landmark:
            x, y = landmark.x * width, landmark.y * height
            if 0 <= x <= width and 0 <= y <= height:
                valid_landmarks += 1
        
        # At least 15 out of 21 landmarks should be visible
        return valid_landmarks >= 15
    
    def _crop_to_table_region(self, image):
        """Crop image to focus on table area - bottom 65% of image"""
        h, w = image.shape[:2]
        
        # Crop to bottom 65% of image (start from 35% down)
        crop_top = int(h * self.table_crop_factor)
        
        cropped = image[crop_top:h, 0:w]
        
        return cropped
    
    def _detect_chips_by_color(self, image_path):
        """
        Fallback chip detection using color analysis
        VERY STRICT - only detect obvious circular RED poker chips
        """
        try:
            full_image = cv2.imread(image_path)
            if full_image is None:
                return 0
            
            # Crop to table region
            image = self._crop_to_table_region(full_image)
            
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Detect ONLY bright RED chips (very strict saturation/value)
            lower_red1 = np.array([0, 120, 100])  # High saturation (was 50)
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 120, 100])  # High saturation (was 50)
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            # Exclude skin tones (hands)
            lower_skin = np.array([0, 10, 50])
            upper_skin = np.array([30, 255, 255])
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Exclude blue (card backs)
            lower_blue = np.array([85, 30, 30])
            upper_blue = np.array([135, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Exclude white/light colors (cards, table reflections)
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 50, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Remove ALL exclusions from red mask
            red_mask = cv2.bitwise_and(red_mask, cv2.bitwise_not(skin_mask))
            red_mask = cv2.bitwise_and(red_mask, cv2.bitwise_not(blue_mask))
            red_mask = cv2.bitwise_and(red_mask, cv2.bitwise_not(white_mask))
            
            # Aggressive noise removal
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            chip_count = 0
            img_height, img_width = image.shape[:2]
            img_area = img_height * img_width
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Chip must be substantial size (200-8000 pixels)
                if not (200 < area < 8000):
                    continue
                
                # Check circularity (must be VERY circular)
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * np.pi * area / (perimeter ** 2)
                
                # VERY STRICT: Must be almost perfectly circular (0.75+)
                if circularity < 0.75:
                    continue
                
                # Check aspect ratio (must be nearly 1:1)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # VERY STRICT: Must be nearly square
                if not (0.85 < aspect_ratio < 1.15):
                    continue
                
                # Additional check: region must be PREDOMINANTLY red
                region_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(region_mask, [contour], -1, 255, -1)
                
                red_pixels = cv2.bitwise_and(red_mask, region_mask)
                red_ratio = np.sum(red_pixels > 0) / np.sum(region_mask > 0) if np.sum(region_mask > 0) > 0 else 0
                
                # At least 70% of region must be red (was 60%)
                if red_ratio < 0.70:
                    continue
                
                # Check if chip is in reasonable location (not at extreme edges)
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Reject chips at very edge of image (likely artifacts)
                edge_margin = 0.05
                if (center_x < img_width * edge_margin or 
                    center_x > img_width * (1 - edge_margin) or
                    center_y < img_height * edge_margin or 
                    center_y > img_height * (1 - edge_margin)):
                    continue
                
                chip_count += 1
            
            print(f"[COLOR CHIP DETECTION] Found {chip_count} red chips (very strict)")
            return chip_count
            
        except Exception as e:
            print(f"[ERROR] Color chip detection failed: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def check_chips(self, image_path):
        """Check for chips - ONLY use YOLO, disable unreliable color fallback"""
        if self.chip_model is None:
            return False, []
        
        try:
            # Read image and crop to table region
            full_image = cv2.imread(image_path)
            if full_image is None:
                print(f"[ERROR] Could not read image: {image_path}")
                return False, []
            
            # CROP TO TABLE
            image = self._crop_to_table_region(full_image)
            
            # Save cropped image temporarily for YOLO
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, image)
                temp_path = tmp.name
            
            try:
                # Run YOLO on CROPPED image
                results = self.chip_model(temp_path, conf=0.35, verbose=False)
                
                detected_chips = []
                raw_detections = []
                
                # Collect all detections
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        chip_label = result.names[class_id]
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].cpu().numpy()
                        
                        raw_detections.append((confidence, chip_label, bbox))
                
                print(f"[CHIP DETECTION] Raw YOLO detections: {len(raw_detections)}")
                
                # Strict filtering
                for confidence, chip_label, bbox in raw_detections:
                    is_valid_chip = self._verify_is_chip(image, bbox)
                    
                    if is_valid_chip:
                        detected_chips.append((confidence, chip_label, bbox))
                    else:
                        print(f"  [FILTER] Rejected {chip_label} - failed validation")
                
                # Apply NMS
                detected_chips = self._non_maximum_suppression(detected_chips, iou_threshold=0.4)
                
                chip_labels = [label for _, label, _ in detected_chips]
                
                print(f"[CHIP DETECTION] Found {len(chip_labels)} valid chips after filtering")
                
                # DISABLE COLOR FALLBACK - too unreliable
                # Only use YOLO results
                
                return len(chip_labels) > 0, chip_labels
            
            finally:
                # Clean up temp file
                os.unlink(temp_path)
        
        except Exception as e:
            print(f"[ERROR] Chip detection failed: {e}")
            import traceback
            traceback.print_exc()
            return False, []
    
    def _verify_is_chip(self, image, bbox):
        """Verify detection is actually a chip, not a card or hand"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure valid coordinates
            h, w = image.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return False
            
            region = image[y1:y2, x1:x2]
            region_h, region_w = region.shape[:2]
            
            # Chips should be roughly circular (aspect ratio close to 1:1)
            # Widened tolerance to allow for chips at angles or multiple chips
            aspect_ratio = region_w / region_h if region_h > 0 else 0
            if not (0.5 < aspect_ratio < 2.5):  # More permissive for real-world chip images
                print(f"  [FILTER] Rejected - bad aspect ratio: {aspect_ratio:.2f}")
                return False
            
            # Convert to HSV
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Reject if mostly blue/white (likely a card)
            lower_blue = np.array([80, 30, 30])
            upper_blue = np.array([140, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            blue_ratio = np.sum(blue_mask > 0) / blue_mask.size
            
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 50, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            white_ratio = np.sum(white_mask > 0) / white_mask.size
            
            if blue_ratio > 0.4 or white_ratio > 0.5:
                print(f"  [FILTER] Rejected - looks like card (blue:{blue_ratio:.2f}, white:{white_ratio:.2f})")
                return False
            
            # Reject if mostly skin tone (likely a hand)
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 255, 255])
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
            
            if skin_ratio > 0.6:
                print(f"  [FILTER] Rejected - looks like hand (skin:{skin_ratio:.2f})")
                return False
            
            return True
            
        except Exception as e:
            print(f"  [FILTER] Exception in verification: {e}")
            return False
    
    def analyze_action(self, image_path):
        """
        Main analysis function that checks for poker actions in order:
        1. Folded cards (FOLD) - Check FIRST to prevent confusion with hands
        2. Chips (BET/RAISE)
        3. Hand (CHECK) - Check LAST as it's most prone to false positives

        Returns: dict with action type and details
        """
        result = {
            'action': None,
            'details': {}
        }

        # Step 1: Check for folded cards FIRST
        # This prevents MediaPipe from detecting hands in folded card images
        folded_cards, fold_confidence = self.check_folded_cards(image_path)

        if folded_cards:
            result['action'] = 'FOLD'
            result['details']['folded_cards_detected'] = True
            result['details']['fold_confidence'] = fold_confidence
            return result

        # Step 2: Check for chips SECOND
        chips_present, chip_labels = self.check_chips(image_path)

        if chips_present:
            # If chips detected, analyze chip amounts
            chip_amounts = self.check_chip_amounts(image_path)
            result['action'] = 'BET/RAISE'
            result['details']['chips_detected'] = chip_labels
            result['details']['chip_amounts'] = list(chip_amounts) if chip_amounts else []
            return result

        # Step 3: Check for hand LAST
        # MediaPipe is prone to false positives, so check this last
        hand_present, hand_confidence = self.check_hand_present(image_path)

        if hand_present:
            result['action'] = 'CHECK'
            result['details']['hand_detected'] = True
            result['details']['hand_confidence'] = hand_confidence
            return result
        
        # Step 4: Uncertain states
        if hand_confidence > 0.20:
            result['action'] = 'UNCERTAIN_CHECK'
            result['details']['hand_confidence'] = hand_confidence
            result['details']['message'] = f"Possible hand detected but confidence too low ({hand_confidence:.1%} < {self.check_confidence_threshold:.1%})"
            return result
        
        if fold_confidence > 0.25:
            result['action'] = 'UNCERTAIN_FOLD'
            result['details']['fold_confidence'] = fold_confidence
            result['details']['message'] = f"Possible fold detected but confidence too low ({fold_confidence:.1%} < {self.fold_confidence_threshold:.1%})"
            return result
        
        # No action detected
        result['action'] = 'NO_ACTION'
        return result
    
    def check_folded_cards(self, image_path):
        """IMPROVED fold detection - EXCLUDE SKIN TONES (hands)"""
        try:
            full_image = cv2.imread(image_path)
            if full_image is None:
                return False, 0.0
            
            # CROP TO TABLE REGION
            image = self._crop_to_table_region(full_image)
            
            img_height, img_width = image.shape[:2]
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # CREATE SKIN MASK to exclude hands
            lower_skin = np.array([0, 15, 50])
            upper_skin = np.array([25, 255, 255])
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Dilate skin mask to be more aggressive
            kernel_skin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            skin_mask = cv2.dilate(skin_mask, kernel_skin, iterations=2)
            
            # Method 1: Blue detection BUT exclude vertical structures AND skin
            lower_blue = np.array([70, 20, 20])
            upper_blue = np.array([150, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # EXCLUDE skin regions from blue mask
            blue_mask = cv2.bitwise_and(blue_mask, cv2.bitwise_not(skin_mask))
            
            # Find contours in blue mask
            contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter out vertical contours (blinds)
            blue_mask_filtered = np.zeros_like(blue_mask)
            for contour in contours_blue:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Keep only horizontal-ish rectangles (cards), reject vertical ones (blinds)
                if aspect_ratio > 0.3:  # Not too vertical
                    cv2.drawContours(blue_mask_filtered, [contour], -1, 255, -1)
            
            blue_mask = blue_mask_filtered
            
            # Method 2: White/light detection (also exclude skin)
            lower_white = np.array([0, 0, 140])
            upper_white = np.array([180, 60, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # EXCLUDE skin regions from white mask
            white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(skin_mask))
            
            # Combine (blue cards and white cards)
            combined_mask = cv2.bitwise_or(blue_mask, white_mask)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_confidence = 0.0
            
            # Check for rectangular shapes
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Cards should be 0.3-40% of CROPPED image
                relative_area = area / (img_height * img_width)
                if relative_area < 0.003 or relative_area > 0.40:
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # REJECT VERTICAL SHAPES (blinds have aspect < 0.3)
                if aspect_ratio < 0.3:
                    continue
                
                # Playing cards: aspect ratio 0.4-2.5
                if not (0.4 < aspect_ratio < 2.5):
                    continue
                
                # Check rectangularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                
                # Rectangles have 4-10 corners
                if len(approx) >= 4 and len(approx) <= 10:
                    # Additional check: region should NOT be mostly skin-colored
                    region_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.drawContours(region_mask, [contour], -1, 255, -1)
                    
                    # Check overlap with skin mask
                    overlap = cv2.bitwise_and(region_mask, skin_mask)
                    overlap_ratio = np.sum(overlap > 0) / np.sum(region_mask > 0) if np.sum(region_mask > 0) > 0 else 0
                    
                    # Reject if more than 30% of region is skin
                    if overlap_ratio > 0.30:
                        continue
                    
                    # Calculate confidence
                    rectangularity_score = 1.0 if len(approx) == 4 else 0.6
                    
                    # Aspect ratio score
                    target_ratio = 0.714
                    aspect_diff = abs(aspect_ratio - target_ratio)
                    aspect_score = max(0, 1.0 - (aspect_diff / 1.5))
                    
                    # Size score
                    size_score = min(relative_area / 0.10, 1.0)
                    
                    confidence = (
                        (rectangularity_score * 0.35) +
                        (aspect_score * 0.35) +
                        (size_score * 0.30)
                    )
                    
                    best_confidence = max(best_confidence, confidence)
            
            print(f"[FOLD DETECTION] Confidence: {best_confidence:.2%}")
            return best_confidence >= self.fold_confidence_threshold, best_confidence
            
        except Exception as e:
            print(f"[ERROR] Fold detection failed: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0
    
    def check_chip_amounts(self, image_path):
        """Check chip amounts in the image"""
        if self.chip_amount_model is None:
            return set()

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
    
    def check_hand_present(self, image_path):
        """Check if a hand is present using MediaPipe with confidence score"""
        if not MEDIAPIPE_AVAILABLE or self.hands is None:
            return self.check_hand_simple(image_path)
        
        try:
            # Load and crop image
            full_image = cv2.imread(image_path)
            if full_image is None:
                return False, 0.0
            
            # CROP TO TABLE
            image = self._crop_to_table_region(full_image)
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image with MediaPipe
            results = self.hands.process(image_rgb)
            
            # Check if hands are detected
            if results.multi_hand_landmarks:
                best_confidence = 0.0
                
                # Get confidence scores from each hand
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if self._verify_hand_landmarks(hand_landmarks, image.shape):
                        # If we have multi_handedness, use that confidence
                        if hasattr(results, 'multi_handedness') and results.multi_handedness:
                            if idx < len(results.multi_handedness):
                                hand_confidence = results.multi_handedness[idx].classification[0].score
                                best_confidence = max(best_confidence, hand_confidence)
                            else:
                                best_confidence = max(best_confidence, 0.80)  # High confidence
                        else:
                            best_confidence = max(best_confidence, 0.80)  # Landmarks exist
                
                print(f"[HAND DETECTION] MediaPipe confidence: {best_confidence:.2%}")
                return best_confidence >= self.check_confidence_threshold, best_confidence
            
            print(f"[HAND DETECTION] No hand detected by MediaPipe")
            return False, 0.0
            
        except Exception as e:
            print(f"[WARNING] MediaPipe hand detection failed: {e}")
            return self.check_hand_simple(image_path)
    
    def check_hand_simple(self, image_path):
        """Simple hand detection - CROP TO TABLE FIRST"""
        try:
            full_image = cv2.imread(image_path)
            if full_image is None:
                return False, 0.0
            
            # CROP TO TABLE
            image = self._crop_to_table_region(full_image)
            
            img_height, img_width = image.shape[:2]
            img_area = img_height * img_width
            
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define skin color range in HSV (BROADER RANGE)
            lower_skin = np.array([0, 15, 50], dtype=np.uint8)
            upper_skin = np.array([25, 255, 255], dtype=np.uint8)
            
            # Create mask for skin color
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_confidence = 0.0
            
            # Check if any significant skin-colored region exists
            for contour in contours:
                area = cv2.contourArea(contour)
                relative_area = area / img_area
                
                # Hand should be 1-50% of image (MORE LENIENT)
                if 0.01 < relative_area < 0.50:
                    # Additional check: hand-like shape
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Hands are not too circular (0.1-0.8) - MORE LENIENT
                    if 0.1 < circularity < 0.8:
                        # Calculate confidence
                        size_score = min(relative_area / 0.15, 1.0)  # Optimal at 15%
                        shape_score = max(0, 1.0 - abs(circularity - 0.40) / 0.60)  # More lenient
                        
                        confidence = (size_score * 0.5) + (shape_score * 0.5)
                        best_confidence = max(best_confidence, confidence)
            
            print(f"[HAND DETECTION] Simple method confidence: {best_confidence:.2%}")
            return best_confidence >= self.check_confidence_threshold, best_confidence
            
        except Exception as e:
            print(f"[ERROR] Simple hand detection failed: {e}")
            return False, 0.0
    
    def _non_maximum_suppression(self, detections, iou_threshold=0.4):
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if len(detections) == 0:
            return []
        
        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x[0], reverse=True)
        
        keep = []
        
        while len(detections) > 0:
            # Take the detection with highest confidence
            best = detections[0]
            keep.append(best)
            detections = detections[1:]
            
            # Remove all detections that overlap significantly with best
            filtered = []
            for det in detections:
                iou = self._calculate_iou(best[2], det[2])
                if iou < iou_threshold:
                    filtered.append(det)
            
            detections = filtered
        
        return keep
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # Get coordinates
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height
        
        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        # Calculate IoU
        if union_area == 0:
            return 0
        
        iou = inter_area / union_area
        return iou
    
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