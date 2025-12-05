"""
Detection-Based Chip Counter - Better alternative to height-based counting.

Instead of measuring bounding box height, this counts actual chip detections
and analyzes the chip region for patterns.

Strategies:
1. Count number of chip detections (multiple chips = multiple bboxes)
2. Analyze bounding box area (more chips = larger area)
3. Analyze pixel patterns within chip region (stack height via pixels)
4. Combine multiple signals for robustness
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path


class DetectionBasedChipCounter:
    """Count chips using detection count + area + visual analysis."""

    def __init__(self, model_path=None):
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'Models', 'pot_chips_model.pt')

        self.model = YOLO(model_path)
        print(f"[DETECTION COUNTER] Loaded model: {model_path}")
        print(f"[DETECTION COUNTER] Model classes: {self.model.names}")

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
        print("  python chip_counter_detection_based.py CVDataset/RedChips/3Chips/image.jpg")
