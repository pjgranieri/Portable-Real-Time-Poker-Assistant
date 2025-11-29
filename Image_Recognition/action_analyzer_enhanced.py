"""
Enhanced Action Analyzer - Improved version of simple analyzer.

Improvements over simple analyzer:
1. Multi-threshold detection (tries lower thresholds if nothing found)
2. Image preprocessing (brightness/contrast adjustment)
3. Multiple detection passes with different settings
4. Better handling of edge cases

Goal: Push accuracy above 90%+
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os


class EnhancedActionAnalyzer:
    """Enhanced action analyzer with multiple detection strategies."""

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # Load player action model
        player_action_model_path = os.path.join(self.script_dir, 'Models', 'player_action_model.pt')
        self.model = YOLO(player_action_model_path)

        # Multi-level confidence thresholds (try in order)
        self.fold_thresholds = [0.20, 0.15, 0.10]
        self.chip_thresholds = [0.25, 0.20, 0.15]

        print("[ENHANCED ANALYZER] Initialized")
        print(f"[ENHANCED ANALYZER] Model classes: {self.model.names}")

    def _preprocess_image(self, image_path):
        """
        Apply image preprocessing to improve detection.

        Returns list of image variations to try:
        - Original
        - Brightness adjusted (+20%)
        - Contrast enhanced
        """
        variations = []

        # Read original
        img = cv2.imread(image_path)
        if img is None:
            return [image_path]

        # Save original path
        variations.append(image_path)

        # Create temp directory for variations
        temp_dir = os.path.join(self.script_dir, 'temp_variations')
        os.makedirs(temp_dir, exist_ok=True)

        filename = os.path.basename(image_path)

        # Variation 1: Brightness boost
        bright = cv2.convertScaleAbs(img, alpha=1.0, beta=30)
        bright_path = os.path.join(temp_dir, f"bright_{filename}")
        cv2.imwrite(bright_path, bright)
        variations.append(bright_path)

        # Variation 2: Contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        enhanced_path = os.path.join(temp_dir, f"enhanced_{filename}")
        cv2.imwrite(enhanced_path, enhanced)
        variations.append(enhanced_path)

        return variations

    def _cleanup_temp_files(self, variations):
        """Clean up temporary preprocessed images."""
        for path in variations[1:]:  # Skip original
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass

    def _detect_with_multiple_thresholds(self, image_path, thresholds):
        """
        Try detection with multiple confidence thresholds.

        Returns: (detected, confidence, detections_list)
        """
        for threshold in thresholds:
            results = self.model(image_path, conf=threshold, verbose=False)

            for r in results:
                if len(r.boxes) > 0:
                    # Found something at this threshold
                    detections = []
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        label = r.names[class_id]
                        confidence = float(box.conf[0])
                        detections.append((label, confidence))

                    if detections:
                        max_conf = max(d[1] for d in detections)
                        return True, max_conf, detections

        return False, 0.0, []

    def analyze_action(self, image_path):
        """
        Enhanced action analysis with multiple strategies.

        Strategy:
        1. Try original image with standard thresholds
        2. If nothing found, try preprocessed variations
        3. Try lower thresholds as fallback
        4. Return best detection found

        Returns: dict with action type and details
        """
        result = {
            'action': None,
            'details': {}
        }

        # Generate image variations
        variations = self._preprocess_image(image_path)

        best_fold_conf = 0.0
        best_chip_conf = 0.0
        fold_detections = []
        chip_detections = []

        # Try each image variation
        for variant_idx, variant_path in enumerate(variations):
            variant_name = "original" if variant_idx == 0 else ("bright" if variant_idx == 1 else "enhanced")

            # Try fold detection with multiple thresholds
            fold_found, fold_conf, fold_dets = self._detect_with_multiple_thresholds(
                variant_path, self.fold_thresholds
            )

            if fold_found and fold_conf > best_fold_conf:
                best_fold_conf = fold_conf
                fold_detections = fold_dets
                result['details']['fold_variant'] = variant_name

            # Try chip detection with multiple thresholds
            chip_found, chip_conf, chip_dets = self._detect_with_multiple_thresholds(
                variant_path, self.chip_thresholds
            )

            if chip_found and chip_conf > best_chip_conf:
                best_chip_conf = chip_conf
                chip_detections = chip_dets
                result['details']['chip_variant'] = variant_name

        # Clean up temp files
        self._cleanup_temp_files(variations)

        # Parse detections to get best result
        folded_cards_detected = any(label == 'Folded-Cards' for label, _ in fold_detections)
        chips_detected = any(label == 'Poker-Chips' for label, _ in chip_detections)

        # Decision logic: FOLD takes priority, then CHIPS, then CHECK
        if folded_cards_detected and best_fold_conf >= min(self.fold_thresholds):
            result['action'] = 'FOLD'
            result['details']['folded_cards_detected'] = True
            result['details']['fold_confidence'] = best_fold_conf
            result['details']['method'] = 'enhanced YOLO (multi-threshold)'
            return result

        if chips_detected and best_chip_conf >= min(self.chip_thresholds):
            result['action'] = 'BET/RAISE'
            result['details']['chips_detected'] = True
            result['details']['chip_confidence'] = best_chip_conf
            result['details']['chip_count'] = len([d for d in chip_detections if d[0] == 'Poker-Chips'])
            result['details']['method'] = 'enhanced YOLO (multi-threshold)'
            return result

        # Default to CHECK if nothing detected
        result['action'] = 'CHECK'
        result['details']['method'] = 'default (no detections after enhancement)'
        result['details']['note'] = 'Model has no Hand class - CHECK is inferred'

        return result


def analyze_action_enhanced(image_path):
    """
    Convenience function for enhanced action analysis.

    Args:
        image_path: Path to image

    Returns:
        dict: {'action': str, 'details': dict}
    """
    analyzer = EnhancedActionAnalyzer()
    return analyzer.analyze_action(image_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Test on single image
        image_path = sys.argv[1]

        analyzer = EnhancedActionAnalyzer()
        result = analyzer.analyze_action(image_path)

        print(f"\n{'='*60}")
        print(f"ENHANCED ACTION ANALYSIS")
        print(f"{'='*60}")
        print(f"Image: {image_path}")
        print(f"Action: {result['action']}")
        print(f"Details: {result['details']}")
        print(f"{'='*60}\n")
    else:
        print("Usage:")
        print("  python action_analyzer_enhanced.py <image_path>")
        print("\nExample:")
        print("  python action_analyzer_enhanced.py CVDataset/ChipDataset/image.jpg")
