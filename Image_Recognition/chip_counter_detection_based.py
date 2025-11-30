"""
Detection-Based Chip Counter - Better alternative to height-based counting.

Instead of measuring bounding box height, this counts actual chip detections
and analyzes the chip region for patterns.

Strategies:
1. Use YOLO to detect chip regions
2. Analyze detected region for color (red vs blue)
3. Return color-specific value ($5 red, $10 blue)
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path


class DetectionBasedChipCounter:
    """Count chips using YOLO detection + color analysis."""

    def __init__(self, model_path=None):
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'Models', 'pot_chips_model.pt')

        self.model = YOLO(model_path)
        print(f"[DETECTION COUNTER] Loaded model: {model_path}")
        print(f"[DETECTION COUNTER] Model classes: {self.model.names}")

        # Chip values by color
        self.chip_values = {
            'red': 5,
            'blue': 10
        }

        # Color thresholds in HSV
        self.color_ranges = {
            'red': [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([180, 255, 255]))
            ],
            'blue': [
                (np.array([100, 100, 100]), np.array([130, 255, 255]))
            ]
        }

        print("[CHIP DETECTION COUNTER] Initialized with color detection")
        print("[CHIP DETECTION COUNTER] Red = $5, Blue = $10")
        print("[CHIP DETECTION COUNTER] Hardcap: 1 chip at a time")

    def detect_chip_color_and_value(self, image_path, conf_threshold=0.25):
        """
        Detect chip using YOLO, then analyze color
        
        Process:
        1. Use YOLO to find chip region
        2. Crop to chip bounding box
        3. Analyze cropped region for color (red vs blue)
        4. Return color and corresponding value
        
        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold for YOLO detection
        
        Returns:
            dict: {
                'color': 'red' | 'blue' | None,
                'value': int (5 or 10),
                'confidence': float (0-1),
                'chip_count': int (0 or 1),
                'details': str
            }
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return {
                'color': None,
                'value': 0,
                'confidence': 0.0,
                'chip_count': 0,
                'details': 'Failed to load image'
            }
        
        # STEP 1: Use YOLO to detect chip
        results = self.model(image_path, conf=conf_threshold, verbose=False)
        
        # Find the best chip detection
        best_detection = None
        best_conf = 0
        
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Check if it's a poker chip
                if self.model.names[class_id] == 'Poker-Chips' and conf > best_conf:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    best_detection = {
                        'bbox': (x1, y1, x2, y2),
                        'conf': conf
                    }
                    best_conf = conf
        
        if best_detection is None:
            return {
                'color': None,
                'value': 0,
                'confidence': 0.0,
                'chip_count': 0,
                'details': 'No chip detected by YOLO'
            }
        
        # STEP 2: Crop to detected chip region
        x1, y1, x2, y2 = best_detection['bbox']
        chip_region = img[y1:y2, x1:x2]
        
        if chip_region.size == 0:
            return {
                'color': None,
                'value': 0,
                'confidence': best_detection['conf'],
                'chip_count': 0,
                'details': 'Empty chip region'
            }
        
        # STEP 3: Analyze color in the detected region
        color_result = self._analyze_chip_color(chip_region)
        
        if color_result['color'] is None:
            return {
                'color': None,
                'value': 0,
                'confidence': best_detection['conf'],
                'chip_count': 1,
                'details': f"YOLO detected chip but color unclear: {color_result['details']}"
            }
        
        # STEP 4: Return result with color-specific value
        detected_color = color_result['color']
        chip_value = self.chip_values[detected_color]
        
        return {
            'color': detected_color,
            'value': chip_value,
            'confidence': best_detection['conf'],
            'chip_count': 1,
            'details': f"YOLO detected chip (conf={best_detection['conf']:.2f}), color={detected_color}, value=${chip_value}"
        }
    
    def _analyze_chip_color(self, chip_region):
        """
        Analyze color of detected chip region
        
        Args:
            chip_region: Cropped image of chip (from YOLO bbox)
        
        Returns:
            dict: {'color': 'red'|'blue'|None, 'details': str, 'scores': dict}
        """
        # Convert to HSV
        hsv = cv2.cvtColor(chip_region, cv2.COLOR_BGR2HSV)
        
        # Score each color
        color_scores = {}
        
        for color_name, hsv_ranges in self.color_ranges.items():
            total_pixels = 0
            
            for lower_hsv, upper_hsv in hsv_ranges:
                # Create mask for this color range
                mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
                
                # Count pixels matching this color
                pixel_count = np.count_nonzero(mask)
                total_pixels += pixel_count
            
            # Calculate percentage of pixels matching this color
            total_area = chip_region.shape[0] * chip_region.shape[1]
            percentage = (total_pixels / total_area) * 100 if total_area > 0 else 0
            
            color_scores[color_name] = percentage
        
        # Find dominant color
        best_color = max(color_scores.items(), key=lambda x: x[1])
        
        # Require at least 10% of pixels to match a color
        MIN_THRESHOLD = 10.0
        
        if best_color[1] >= MIN_THRESHOLD:
            return {
                'color': best_color[0],
                'details': f"{best_color[0]} ({best_color[1]:.1f}% match)",
                'scores': color_scores
            }
        
        return {
            'color': None,
            'details': f"No strong color match (red={color_scores['red']:.1f}%, blue={color_scores['blue']:.1f}%)",
            'scores': color_scores
        }

    def _analyze_chip_region(self, image, bbox):
        """
        Analyze the chip region to estimate stack height.

        Strategy: Look for color transitions (chips have edges between them)
        More transitions = more chips stacked
        """
        x1, y1, x2, y2 = bbox

        # Extract region
        region = image[y1:y2, x1:x2]
        if region.size == 0:
            return 1

        h, w = region.shape[:2]

        # Analyze vertical color profile (look for chip edges)
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region

        # Get vertical profile (average intensity per row)
        vertical_profile = np.mean(gray, axis=1)

        # Find peaks and valleys (chip edges cause intensity changes)
        # Use gradient to find transitions
        gradient = np.abs(np.gradient(vertical_profile))

        # Count significant transitions
        threshold = np.mean(gradient) + np.std(gradient)
        transitions = np.sum(gradient > threshold)

        # More transitions likely means more chips
        # Rough heuristic: 2-3 transitions per chip
        estimated_chips = max(1, transitions // 2)

        return min(estimated_chips, 5)  # Cap at 5

    def count_chips(self, image_path, conf_threshold=0.25):
        """
        Count chips using multiple strategies.

        Returns: dict with count and confidence info
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'count': 0, 'method': 'error', 'confidence': 0.0}

        # Run YOLO detection with low threshold to catch all chips
        results = self.model(image_path, conf=0.15, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                area = (x2 - x1) * (y2 - y1)

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'conf': conf,
                    'area': area,
                    'width': x2 - x1,
                    'height': y2 - y1
                })

        if not detections:
            return {'count': 0, 'method': 'no_detections', 'confidence': 0.0}

        # Strategy 1: Count detections (works if model detects each chip separately)
        num_detections = len(detections)

        # Strategy 2: Use largest detection and analyze its area
        largest = max(detections, key=lambda x: x['area'])

        # Strategy 3: Use aspect ratio (most reliable signal)
        aspect_ratio = largest['height'] / largest['width'] if largest['width'] > 0 else 0

        # Use original height-based thresholds (they worked well!)
        # These are proven from the height-based calibration
        if aspect_ratio < 0.85:
            ratio_hint = 1
        elif aspect_ratio < 0.89:
            ratio_hint = 2
        elif aspect_ratio < 0.93:
            ratio_hint = 3
        elif aspect_ratio < 0.985:
            ratio_hint = 4
        else:
            ratio_hint = 5

        # Strategy 4: Analyze chip region for stacking
        visual_count = self._analyze_chip_region(image, largest['bbox'])

        # Combine strategies with weighted voting
        estimates = {
            'num_detections': num_detections,
            'aspect_ratio': ratio_hint,
            'visual_analysis': visual_count
        }

        # If model detected multiple separate chips, trust that
        if num_detections > 1:
            final_count = min(num_detections, 5)
            method = 'detection_count'
        else:
            # Weighted combination works best
            # Give heavy weight to aspect ratio, but let visual analysis contribute
            weighted_avg = ratio_hint * 0.75 + visual_count * 0.25
            final_count = round(weighted_avg)
            method = 'aspect_ratio_weighted'

        # Cap at 5
        final_count = min(max(final_count, 1), 5)

        return {
            'count': final_count,
            'method': method,
            'confidence': largest['conf'],
            'estimates': estimates,
            'num_detections': num_detections
        }


def count_pot_chips_detection_based(image_path):
    """
    Convenience function for detection-based chip counting.

    Args:
        image_path: Path to image

    Returns:
        int: Number of chips detected
    """
    counter = DetectionBasedChipCounter()
    result = counter.count_chips(image_path)
    return result['count']


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Test on single image
        image_path = sys.argv[1]

        counter = DetectionBasedChipCounter()
        
        # Test color detection
        print(f"\n{'='*60}")
        print(f"CHIP COLOR DETECTION TEST")
        print(f"{'='*60}")
        result = counter.detect_chip_color_and_value(image_path)
        print(f"Image: {image_path}")
        print(f"Color: {result['color']}")
        print(f"Value: ${result['value']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Details: {result['details']}")
        print(f"{'='*60}\n")
        
        # Test chip counting
        result = counter.count_chips(image_path)
        print(f"\n{'='*60}")
        print(f"DETECTION-BASED CHIP COUNTING")
        print(f"{'='*60}")
        print(f"Image: {image_path}")
        print(f"Final Count: {result['count']} chips")
        print(f"Method: {result['method']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Estimates breakdown:")
        for method, count in result['estimates'].items():
            print(f"  - {method}: {count}")
        print(f"{'='*60}\n")
    else:
        print("Usage:")
        print("  python chip_counter_detection_based.py <image_path>")
        print("\nExample:")
        print("  python chip_counter_detection_based.py CVDataset/RedChips/1Chip/image.jpg")
