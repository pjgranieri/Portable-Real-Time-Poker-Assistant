"""
Simple Action Analyzer - Barebones version that trusts the YOLO models directly.

This version removes all the complex filtering and heuristics from action_analyzer_algorithm.py
and instead relies purely on the trained YOLO models.

Purpose: Test if the heavy filtering was causing low accuracy.
"""

from ultralytics import YOLO
import cv2
import os


class SimpleActionAnalyzer:
    """Simplified action analyzer that trusts YOLO model detections."""

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # Load player action model (detects Folded-Cards and Poker-Chips)
        player_action_model_path = os.path.join(self.script_dir, 'Models', 'player_action_model.pt')
        self.model = YOLO(player_action_model_path)

        # Detection thresholds
        self.fold_confidence = 0.20
        self.chip_confidence = 0.25

        print("[SIMPLE ANALYZER] Initialized")
        print(f"[SIMPLE ANALYZER] Model classes: {self.model.names}")

    def analyze_action(self, image_path):
        """
        Analyze player action using YOLO model directly.

        Logic:
        1. Run YOLO on image
        2. If Folded-Cards detected → FOLD
        3. Else if Poker-Chips detected → BET/RAISE
        4. Else → CHECK (default when nothing detected)

        Returns: dict with action type and details
        """
        result = {
            'action': None,
            'details': {}
        }

        # Run YOLO detection
        results = self.model(image_path, conf=0.20, verbose=False)

        # Parse detections
        folded_cards_detected = False
        chips_detected = False
        max_fold_conf = 0.0
        max_chip_conf = 0.0
        chip_count = 0

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                label = r.names[class_id]
                confidence = float(box.conf[0])

                if label == 'Folded-Cards':
                    folded_cards_detected = True
                    max_fold_conf = max(max_fold_conf, confidence)
                elif label == 'Poker-Chips':
                    chips_detected = True
                    chip_count += 1
                    max_chip_conf = max(max_chip_conf, confidence)

        # Decision logic: FOLD takes priority, then CHIPS, then CHECK
        if folded_cards_detected and max_fold_conf >= self.fold_confidence:
            result['action'] = 'FOLD'
            result['details']['folded_cards_detected'] = True
            result['details']['fold_confidence'] = max_fold_conf
            result['details']['method'] = 'YOLO detection'
            return result

        if chips_detected and max_chip_conf >= self.chip_confidence:
            result['action'] = 'BET/RAISE'
            result['details']['chips_detected'] = True
            result['details']['chip_confidence'] = max_chip_conf
            result['details']['chip_count'] = chip_count
            result['details']['method'] = 'YOLO detection'
            return result

        # Default to CHECK if nothing detected
        # Note: Model doesn't have a "Hand" class, so we can't detect CHECKs via YOLO
        result['action'] = 'CHECK'
        result['details']['method'] = 'default (no detections)'
        result['details']['note'] = 'Model has no Hand class - CHECK is inferred'

        return result


def analyze_action_simple(image_path):
    """
    Convenience function for simple action analysis.

    Args:
        image_path: Path to image

    Returns:
        dict: {'action': str, 'details': dict}
    """
    analyzer = SimpleActionAnalyzer()
    return analyzer.analyze_action(image_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Test on single image
        image_path = sys.argv[1]

        analyzer = SimpleActionAnalyzer()
        result = analyzer.analyze_action(image_path)

        print(f"\n{'='*60}")
        print(f"SIMPLE ACTION ANALYSIS")
        print(f"{'='*60}")
        print(f"Image: {image_path}")
        print(f"Action: {result['action']}")
        print(f"Details: {result['details']}")
        print(f"{'='*60}\n")
    else:
        print("Usage:")
        print("  python action_analyzer_simple.py <image_path>")
        print("\nExample:")
        print("  python action_analyzer_simple.py CVDataset/ChipDataset/image.jpg")
