"""
Chip Stack Counter - Uses HEIGHT of detected chips to count quantity.

Brilliant insight: Stacked chips are TALLER in the bounding box!
- 1 chip = short height
- 5 chips = tall height

This actually works for top-down 2D images.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path


class ChipHeightAnalyzer:
    """Count chips by measuring stack height from YOLO bounding box."""

    def __init__(self, yolo_model_path=None):
        """Initialize analyzer."""
        if yolo_model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            yolo_model_path = os.path.join(script_dir, 'Models', 'chip_processing_model.pt')

        self.yolo_model = YOLO(yolo_model_path)

        # Height thresholds (calibrated from dataset)
        # These are aspect ratios (height/width) of bounding box
        # Based on calibration: 1chip≈0.83, 2chips≈0.87, 3chips≈0.91, 4chips≈0.95, 5chips≈1.02
        # Using midpoints between means for better classification
        self.height_thresholds = {
            1: (0.00, 0.85),   # 1 chip: below 0.85
            2: (0.85, 0.89),   # 2 chips: 0.85-0.89 (midpoint 0.83-0.87 = 0.85, midpoint 0.87-0.91 = 0.89)
            3: (0.89, 0.93),   # 3 chips: 0.89-0.93 (midpoint 0.91-0.95 = 0.93)
            4: (0.93, 0.985),  # 4 chips: 0.93-0.985 (midpoint 0.95-1.02 = 0.985)
            5: (0.985, 10.00), # 5+ chips: above 0.985
        }

    def detect_and_count_chips(self, image_path, color='red', conf_threshold=0.25):
        """
        Detect chips and count by measuring height.

        Args:
            image_path: Path to image
            color: Chip color (for filtering, not used in height method)
            conf_threshold: YOLO confidence threshold

        Returns:
            int: Number of chips (1-5)
        """
        # Try multiple confidence levels if first attempt fails
        confidence_levels = [conf_threshold, 0.15, 0.1, 0.05]

        chip_regions = []
        used_conf = conf_threshold

        for conf_level in confidence_levels:
            # Run YOLO detection
            results = self.yolo_model(image_path, conf=conf_level, verbose=False)

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
            print(f"[CHIP COUNT] No chips detected by YOLO even at conf=0.05")
            return 1  # Default to 1 chip if nothing detected (better than 0)

        # Take largest detection (most confident)
        largest = max(chip_regions, key=lambda x: x[4] * x[5])
        x1, y1, x2, y2, width, height, conf = largest

        # Calculate aspect ratio (height / width)
        aspect_ratio = height / width if width > 0 else 0

        # Determine chip count from aspect ratio
        chip_count = self._height_to_count(aspect_ratio)

        print(f"[CHIP COUNT] BBox: {width}x{height}, Aspect: {aspect_ratio:.2f}, Count: {chip_count}, Conf: {conf:.2f} (threshold: {used_conf})")

        return chip_count

    def _height_to_count(self, aspect_ratio):
        """Map aspect ratio to chip count."""
        for count, (min_ratio, max_ratio) in self.height_thresholds.items():
            if min_ratio <= aspect_ratio < max_ratio:
                return count
        return 5  # Default to 5 if very tall

    def calibrate_from_dataset(self, dataset_root='CVDataset', conf_threshold=0.25):
        """
        Auto-calibrate height thresholds from dataset.

        Returns:
            dict: Calibrated thresholds
        """
        dataset_root = Path(dataset_root)

        # Collect aspect ratios for each chip count
        aspect_ratios = {1: [], 2: [], 3: [], 4: [], 5: []}

        for color_folder in ['BlackChips', 'BlueChips', 'RedChips']:
            color_path = dataset_root / color_folder
            if not color_path.exists():
                continue

            for chip_count in [1, 2, 3, 4, 5]:
                folder_name = f"{chip_count}Chip" if chip_count == 1 else f"{chip_count}Chips"
                chip_folder = color_path / folder_name

                if not chip_folder.exists():
                    continue

                # Sample 10 images per category
                images = list(chip_folder.glob("*.jpg"))[:10]

                for img_path in images:
                    results = self.yolo_model(str(img_path), conf=conf_threshold, verbose=False)

                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            width = x2 - x1
                            height = y2 - y1
                            aspect_ratio = height / width if width > 0 else 0

                            if aspect_ratio > 0:
                                aspect_ratios[chip_count].append(aspect_ratio)
                                break  # Only take first detection

        # Calculate thresholds from data
        print(f"\n{'='*70}")
        print("CALIBRATION RESULTS")
        print(f"{'='*70}\n")

        calibrated_thresholds = {}

        for chip_count in [1, 2, 3, 4, 5]:
            ratios = aspect_ratios[chip_count]
            if ratios:
                mean = np.mean(ratios)
                std = np.std(ratios)
                min_val = np.min(ratios)
                max_val = np.max(ratios)

                print(f"{chip_count} chip(s): mean={mean:.2f}, std={std:.2f}, range=[{min_val:.2f}, {max_val:.2f}], n={len(ratios)}")

                # Set threshold as mean ± std
                lower = max(0.0, mean - std)
                upper = mean + std

                calibrated_thresholds[chip_count] = (lower, upper)

        print(f"\n{'='*70}")
        print("SUGGESTED THRESHOLDS")
        print(f"{'='*70}\n")

        # Adjust boundaries to not overlap
        sorted_counts = sorted(calibrated_thresholds.keys())
        final_thresholds = {}

        for i, count in enumerate(sorted_counts):
            if i == 0:
                # First threshold starts at 0
                final_thresholds[count] = (0.0, calibrated_thresholds[count][1])
            elif i == len(sorted_counts) - 1:
                # Last threshold goes to infinity
                final_thresholds[count] = (calibrated_thresholds[count][0], 10.0)
            else:
                # Middle thresholds
                final_thresholds[count] = calibrated_thresholds[count]

        print("self.height_thresholds = {")
        for count in sorted(final_thresholds.keys()):
            lower, upper = final_thresholds[count]
            print(f"    {count}: ({lower:.2f}, {upper:.2f}),")
        print("}")

        return final_thresholds


def count_chips_in_image(image_path, color='red'):
    """Convenience function."""
    analyzer = ChipHeightAnalyzer()
    return analyzer.detect_and_count_chips(image_path, color=color)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == '--calibrate':
            # Run calibration
            analyzer = ChipHeightAnalyzer()
            analyzer.calibrate_from_dataset()
        else:
            # Test on single image
            image_path = sys.argv[1]
            color = sys.argv[2] if len(sys.argv) > 2 else 'red'

            count = count_chips_in_image(image_path, color)
            print(f"\n{'='*60}")
            print(f"Final Count: {count} chip(s)")
            print(f"{'='*60}")
    else:
        print("Usage:")
        print("  python chip_height_analyzer.py <image_path> [color]")
        print("  python chip_height_analyzer.py --calibrate")
