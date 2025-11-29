"""
Chip Pot Analyzer - Detects and counts chips in the pot.

Combines:
1. pot_chips_model.pt - YOLO model to detect WHERE chips are in the pot
2. Height-based counting - Analyzes bounding box aspect ratio to determine HOW MANY chips

This is the production-ready chip counting system.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path


class ChipPotAnalyzer:
    """Detects chips in pot and counts quantity using height-based analysis."""

    def __init__(self, pot_model_path=None):
        """
        Initialize the chip pot analyzer.

        Args:
            pot_model_path: Path to pot_chips_model.pt (optional)
        """
        if pot_model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            pot_model_path = os.path.join(script_dir, 'Models', 'pot_chips_model.pt')

        # Load pot chips detection model
        self.pot_model = YOLO(pot_model_path)

        # Height thresholds (calibrated from dataset)
        # These are aspect ratios (height/width) of bounding box
        # Based on calibration: 1chip≈0.83, 2chips≈0.87, 3chips≈0.91, 4chips≈0.95, 5chips≈1.02
        # Using midpoints between means for better classification
        self.height_thresholds = {
            1: (0.00, 0.85),   # 1 chip: below 0.85
            2: (0.85, 0.89),   # 2 chips: 0.85-0.89
            3: (0.89, 0.93),   # 3 chips: 0.89-0.93
            4: (0.93, 0.985),  # 4 chips: 0.93-0.985
            5: (0.985, 10.00), # 5+ chips: above 0.985
        }

    def detect_and_count_chips(self, image_path, color='red', conf_threshold=0.25):
        """
        Detect chips in pot and count quantity.

        Args:
            image_path: Path to image
            color: Chip color (optional, for logging)
            conf_threshold: YOLO confidence threshold

        Returns:
            int: Number of chips detected (1-5, or 0 if no chips found)
        """
        # Try multiple confidence levels if first attempt fails
        confidence_levels = [conf_threshold, 0.15, 0.1, 0.05]

        chip_regions = []
        used_conf = conf_threshold

        for conf_level in confidence_levels:
            # Run YOLO detection on pot
            results = self.pot_model(image_path, conf=conf_level, verbose=False)

            chip_regions = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])

                    width = x2 - x1
                    height = y2 - y1

                    chip_regions.append((x1, y1, x2, y2, width, height, conf))

            # If we found chips, stop trying lower confidence levels
            if chip_regions:
                used_conf = conf_level
                break

        if not chip_regions:
            print(f"[POT CHIP COUNT] No chips detected by pot model even at conf=0.05")
            return 0  # No chips in pot

        # Take largest detection (most likely the chip stack)
        largest = max(chip_regions, key=lambda x: x[4] * x[5])
        x1, y1, x2, y2, width, height, conf = largest

        # Calculate aspect ratio (height / width)
        aspect_ratio = height / width if width > 0 else 0

        # Determine chip count from aspect ratio
        chip_count = self._height_to_count(aspect_ratio)

        print(f"[POT CHIP COUNT] BBox: {width}x{height}, Aspect: {aspect_ratio:.2f}, Count: {chip_count}, Conf: {conf:.2f} (threshold: {used_conf})")

        return chip_count

    def _height_to_count(self, aspect_ratio):
        """
        Map aspect ratio to chip count.

        Args:
            aspect_ratio: Height/width ratio of bounding box

        Returns:
            int: Chip count (1-5)
        """
        for count, (min_ratio, max_ratio) in self.height_thresholds.items():
            if min_ratio <= aspect_ratio < max_ratio:
                return count
        return 5  # Default to 5 if very tall

    def detect_pot_chips_detailed(self, image_path, conf_threshold=0.25):
        """
        Detect all chip stacks in pot with detailed information.

        Returns:
            list: List of dicts with {bbox, count, confidence, aspect_ratio}
        """
        results = self.pot_model(image_path, conf=conf_threshold, verbose=False)

        chip_stacks = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])

                width = x2 - x1
                height = y2 - y1
                aspect_ratio = height / width if width > 0 else 0

                chip_count = self._height_to_count(aspect_ratio)

                chip_stacks.append({
                    'bbox': (x1, y1, x2, y2),
                    'count': chip_count,
                    'confidence': conf,
                    'aspect_ratio': aspect_ratio,
                    'size': (width, height)
                })

        return chip_stacks


def count_pot_chips(image_path, color='red'):
    """
    Convenience function to count chips in pot.

    Args:
        image_path: Path to image
        color: Chip color (optional)

    Returns:
        int: Number of chips
    """
    analyzer = ChipPotAnalyzer()
    return analyzer.detect_and_count_chips(image_path, color=color)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Test on single image
        image_path = sys.argv[1]
        color = sys.argv[2] if len(sys.argv) > 2 else 'red'

        analyzer = ChipPotAnalyzer()
        count = analyzer.detect_and_count_chips(image_path, color)

        print(f"\n{'='*60}")
        print(f"Final Pot Chip Count: {count} chip(s)")
        print(f"{'='*60}")
    else:
        print("Usage:")
        print("  python chip_pot_analyzer.py <image_path> [color]")
        print("\nExample:")
        print("  python chip_pot_analyzer.py CVDataset/RedChips/3Chips/image.jpg red")
