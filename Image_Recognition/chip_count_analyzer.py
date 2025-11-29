"""
Two-stage chip detection and counting system.

Stage 1: YOLO detects chip regions (WHERE chips are)
Stage 2: HSV color analysis counts chips in each region (HOW MANY chips)

This combines the strengths of both approaches:
- YOLO is excellent at locating chips
- HSV analysis is better at counting stacked chips
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os


class ChipCountAnalyzer:
    """Analyzes chip regions to count individual chips."""

    def __init__(self, yolo_model_path=None):
        """
        Initialize the chip count analyzer.

        Args:
            yolo_model_path: Path to YOLO chip detection model
        """
        if yolo_model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            yolo_model_path = os.path.join(script_dir, 'Models', 'chip_processing_model.pt')

        self.yolo_model = YOLO(yolo_model_path)

    def detect_and_count_chips(self, image_path, color='red', conf_threshold=0.35):
        """
        Detect chip regions with YOLO, then count chips in each region.

        Args:
            image_path: Path to image file
            color: Chip color to count ('red', 'blue', 'black')
            conf_threshold: YOLO confidence threshold

        Returns:
            int: Total number of chips detected across all regions
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return 0

        # Stage 1: Detect chip regions with YOLO
        chip_regions = self._detect_chip_regions(image_path, conf_threshold)

        if len(chip_regions) == 0:
            print(f"[CHIP COUNT] No chip regions detected by YOLO")
            return 0

        print(f"[CHIP COUNT] YOLO found {len(chip_regions)} chip region(s)")

        # Stage 2: Count chips in each detected region
        total_chips = 0
        hsv_range = self._get_hsv_range_for_color(color)

        for idx, (x1, y1, x2, y2, confidence) in enumerate(chip_regions):
            # Extract region
            region = image[y1:y2, x1:x2]

            # Count chips in this region
            chip_count = self._count_chips_in_region(region, hsv_range)

            print(f"  Region {idx+1}: {chip_count} chip(s) (YOLO conf: {confidence:.2f})")
            total_chips += chip_count

        print(f"[CHIP COUNT] Total: {total_chips} chip(s)")
        return total_chips

    def _detect_chip_regions(self, image_path, conf_threshold=0.35):
        """
        Use YOLO to detect chip regions.

        Returns:
            List of (x1, y1, x2, y2, confidence) tuples
        """
        results = self.yolo_model(image_path, conf=conf_threshold, verbose=False)

        chip_regions = []

        for result in results:
            for box in result.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0])

                chip_regions.append((x1, y1, x2, y2, confidence))

        return chip_regions

    def _count_chips_in_region(self, region, hsv_range):
        """
        Count individual chips in a detected region using HSV analysis.

        This works by:
        1. Creating a color mask for the chip color
        2. Finding circular contours (individual chips)
        3. Counting valid chip-shaped objects

        Args:
            region: Cropped image region containing chips
            hsv_range: (lower_hsv, upper_hsv) tuple for color filtering

        Returns:
            int: Number of chips detected in this region
        """
        if region is None or region.size == 0:
            return 0

        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Create color mask
        mask = cv2.inRange(hsv, hsv_range[0], hsv_range[1])

        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        chip_count = 0
        region_area = region.shape[0] * region.shape[1]

        for contour in contours:
            area = cv2.contourArea(contour)

            # Chip must be a reasonable size relative to the region
            # (between 5% and 95% of region area)
            relative_area = area / region_area
            if not (0.05 < relative_area < 0.95):
                continue

            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)

            # Must be reasonably circular (chips are round)
            if circularity > 0.5:  # Lowered threshold for stacked chips
                chip_count += 1

        # If we found 0 chips but there's significant color area,
        # assume at least 1 chip (might be a single chip without clear contour)
        if chip_count == 0:
            color_ratio = np.sum(mask > 0) / mask.size
            if color_ratio > 0.3:  # More than 30% is the target color
                chip_count = 1

        return chip_count

    def _get_hsv_range_for_color(self, color):
        """
        Get HSV range for a specific chip color.

        Args:
            color: 'red', 'blue', 'black', 'green', or 'white'

        Returns:
            tuple: (lower_hsv, upper_hsv)
        """
        hsv_ranges = {
            'red': ((0, 0, 47), (77, 194, 255)),      # MODERATE: 80% coverage (measured)
            'blue': ((26, 0, 59), (120, 165, 255)),   # MODERATE: 80% coverage (measured)
            'black': ((23, 0, 55), (77, 169, 255)),   # MODERATE: 80% coverage (measured)
            'green': ((40, 100, 100), (80, 255, 255)),
            'white': ((0, 0, 200), (180, 30, 255)),
        }

        if color.lower() not in hsv_ranges:
            raise ValueError(f"Unknown color: {color}. Must be one of {list(hsv_ranges.keys())}")

        return hsv_ranges[color.lower()]


def count_chips_in_image(image_path, color='red'):
    """
    Convenience function to count chips in an image.

    Args:
        image_path: Path to image
        color: Chip color ('red', 'blue', 'black')

    Returns:
        int: Number of chips detected
    """
    analyzer = ChipCountAnalyzer()
    return analyzer.detect_and_count_chips(image_path, color=color)


if __name__ == "__main__":
    # Test the analyzer
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        color = sys.argv[2] if len(sys.argv) > 2 else 'red'

        count = count_chips_in_image(image_path, color)
        print(f"\n{'='*60}")
        print(f"Final Count: {count} {color} chip(s)")
        print(f"{'='*60}")
    else:
        print("Usage: python chip_count_analyzer.py <image_path> [color]")
        print("Example: python chip_count_analyzer.py test.jpg red")
