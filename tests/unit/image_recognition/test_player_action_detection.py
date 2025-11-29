"""
Comprehensive unit tests for player action detection.

Tests the player_action_model.pt YOLO model paired with action_analyzer_algorithm:
1. player_action_model detects folded cards OR chips in player action zone
2. MediaPipe detects hands for CHECK action
3. Combined logic determines player action

This tests across different action types:
- FoldedCardsDataset → FOLD action
- HandDataset → CHECK action
- ChipDataset → BET/RAISE action

Run with:
    pytest tests/unit/image_recognition/test_player_action_detection.py -v

Run with detailed report:
    pytest tests/unit/image_recognition/test_player_action_detection.py -v --tb=short

Run in parallel:
    pytest tests/unit/image_recognition/test_player_action_detection.py -n auto
"""

import pytest
import os
from pathlib import Path
from collections import defaultdict

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from Image_Recognition.action_analyzer_algorithm import ActionAnalyzer


# Dataset configuration
DATASET_ROOT = Path(__file__).parent.parent.parent.parent / "CVDataset"

# Action datasets and expected actions
ACTION_DATASETS = {
    "FoldedCardsDataset": "FOLD",
    "HandDataset": "CHECK",
    "ChipDataset": "BET/RAISE",
}


def discover_test_images():
    """
    Dynamically discover all test images in action datasets.

    Returns:
        list: List of tuples (dataset_name, expected_action, image_path)
    """
    test_cases = []

    for dataset_name, expected_action in ACTION_DATASETS.items():
        dataset_path = DATASET_ROOT / dataset_name

        if not dataset_path.exists():
            print(f"Warning: Dataset not found: {dataset_path}")
            continue

        # Get all image files recursively
        image_files = list(dataset_path.glob("**/*.jpg")) + \
                     list(dataset_path.glob("**/*.png")) + \
                     list(dataset_path.glob("*.jpg")) + \
                     list(dataset_path.glob("*.png"))

        for image_path in image_files:
            test_cases.append((dataset_name, expected_action, str(image_path)))

    return test_cases


# Generate all test cases
TEST_IMAGES = discover_test_images()


@pytest.fixture(scope="module")
def action_analyzer():
    """Create an ActionAnalyzer instance for all tests."""
    return ActionAnalyzer()


@pytest.mark.parametrize("dataset_name,expected_action,image_path", TEST_IMAGES)
def test_player_action_accuracy(action_analyzer, dataset_name, expected_action, image_path):
    """
    Test that the action analyzer correctly identifies player actions.

    Uses player_action_model.pt + MediaPipe to detect:
    - FOLD: Folded cards detected
    - CHECK: Hand detected (tapping table)
    - BET/RAISE: Chips detected in action zone

    Args:
        action_analyzer: ActionAnalyzer fixture
        dataset_name: Name of the dataset folder
        expected_action: Expected action (FOLD, CHECK, or BET/RAISE)
        image_path: Path to the test image
    """
    # Verify image exists
    assert os.path.exists(image_path), f"Image not found: {image_path}"

    # Analyze action
    result = action_analyzer.analyze_action(image_path)
    detected_action = result['action']

    # Check if detected action matches expected
    # Note: Some actions might be detected as UNCERTAIN variants
    if expected_action == "FOLD":
        valid_actions = ["FOLD", "UNCERTAIN_FOLD"]
    elif expected_action == "CHECK":
        valid_actions = ["CHECK", "UNCERTAIN_CHECK"]
    elif expected_action == "BET/RAISE":
        valid_actions = ["BET/RAISE"]
    else:
        valid_actions = [expected_action]

    # Assert correct action
    assert detected_action in valid_actions, (
        f"Player action mismatch for {Path(image_path).name}:\n"
        f"  Expected: {expected_action}\n"
        f"  Detected: {detected_action}\n"
        f"  Dataset: {dataset_name}\n"
        f"  Details: {result['details']}\n"
        f"  Image: {image_path}"
    )


class TestPlayerActionDetectionAccuracy:
    """Test suite for overall player action detection accuracy metrics."""

    @pytest.fixture(scope="class")
    def detection_results(self):
        """
        Run detection on all images and collect results.

        Returns:
            dict: Results organized by dataset and action
        """
        results = defaultdict(list)
        analyzer = ActionAnalyzer()

        print(f"\n[TEST] Running player action detection on {len(TEST_IMAGES)} images...")

        for dataset_name, expected_action, image_path in TEST_IMAGES:
            # Skip if image doesn't exist
            if not os.path.exists(image_path):
                continue

            # Analyze action
            result = analyzer.analyze_action(image_path)
            detected_action = result['action']

            # Determine if correct (including uncertain variants)
            if expected_action == "FOLD":
                is_correct = detected_action in ["FOLD", "UNCERTAIN_FOLD"]
            elif expected_action == "CHECK":
                is_correct = detected_action in ["CHECK", "UNCERTAIN_CHECK"]
            elif expected_action == "BET/RAISE":
                is_correct = detected_action == "BET/RAISE"
            else:
                is_correct = detected_action == expected_action

            # Store result
            results[dataset_name].append({
                'expected': expected_action,
                'detected': detected_action,
                'correct': is_correct,
                'image': Path(image_path).name,
                'details': result['details']
            })

        return results

    def test_overall_accuracy(self, detection_results):
        """Test that overall player action detection accuracy is acceptable."""
        total_images = 0
        correct_detections = 0

        for dataset_name in detection_results:
            for result in detection_results[dataset_name]:
                total_images += 1
                if result['correct']:
                    correct_detections += 1

        accuracy = (correct_detections / total_images * 100) if total_images > 0 else 0

        print(f"\n{'='*60}")
        print(f"OVERALL PLAYER ACTION DETECTION ACCURACY")
        print(f"{'='*60}")
        print(f"Total images tested: {total_images}")
        print(f"Correct detections: {correct_detections}")
        print(f"Incorrect detections: {total_images - correct_detections}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"{'='*60}\n")

        # Assert at least 70% accuracy
        assert accuracy >= 70.0, f"Overall accuracy {accuracy:.2f}% is below 70% threshold"

    def test_accuracy_by_action(self, detection_results):
        """Test accuracy broken down by action type."""
        print(f"\n{'='*60}")
        print(f"PLAYER ACTION ACCURACY BY ACTION TYPE")
        print(f"{'='*60}")

        for dataset_name in ACTION_DATASETS.keys():
            if dataset_name not in detection_results:
                continue

            expected_action = ACTION_DATASETS[dataset_name]
            results = detection_results[dataset_name]

            total = len(results)
            correct = sum(1 for r in results if r['correct'])
            accuracy = (correct / total * 100) if total > 0 else 0

            print(f"{expected_action:12s}: {correct:3d}/{total:3d} correct ({accuracy:5.2f}%)")

            # Assert at least 60% accuracy per action type
            assert accuracy >= 60.0, f"{expected_action} accuracy {accuracy:.2f}% is below 60% threshold"

        print(f"{'='*60}\n")

    def test_confusion_matrix(self, detection_results):
        """Show confusion matrix of detected vs expected actions."""
        print(f"\n{'='*60}")
        print(f"PLAYER ACTION CONFUSION MATRIX")
        print(f"{'='*60}")

        # Collect all action types
        all_actions = set()
        for dataset_name in detection_results:
            for result in detection_results[dataset_name]:
                all_actions.add(result['expected'])
                all_actions.add(result['detected'])

        all_actions = sorted(all_actions)

        # Build confusion matrix
        confusion = defaultdict(lambda: defaultdict(int))

        for dataset_name in detection_results:
            for result in detection_results[dataset_name]:
                expected = result['expected']
                detected = result['detected']
                confusion[expected][detected] += 1

        # Print header
        print(f"\n{'Expected':<15} | {'Detected Action Counts':^60}")
        print(f"{'':<15} | {' | '.join([f'{a:^12}' for a in all_actions])}")
        print("-" * 80)

        # Print rows
        for expected in sorted(confusion.keys()):
            counts = [str(confusion[expected][detected]) if confusion[expected][detected] > 0 else "-"
                     for detected in all_actions]
            print(f"{expected:<15} | {' | '.join([f'{c:^12}' for c in counts])}")

        print(f"{'='*60}\n")

    def test_failure_analysis(self, detection_results):
        """Print detailed analysis of failed detections."""
        print(f"\n{'='*60}")
        print(f"PLAYER ACTION FAILURE ANALYSIS")
        print(f"{'='*60}")

        failures_by_type = defaultdict(int)

        for dataset_name in detection_results:
            for result in detection_results[dataset_name]:
                if not result['correct']:
                    error_type = f"Expected {result['expected']}, got {result['detected']}"
                    failures_by_type[error_type] += 1

        if not failures_by_type:
            print("No failures detected!")
        else:
            print("Common failure patterns:")
            for error_type, count in sorted(failures_by_type.items(), key=lambda x: -x[1]):
                print(f"  {error_type}: {count} occurrences")

        print(f"{'='*60}\n")


@pytest.mark.slow
def test_action_detection_speed_benchmark():
    """Benchmark the speed of player action detection."""
    import time

    if not TEST_IMAGES:
        pytest.skip("No test images available")

    # Test on first 10 images
    sample_images = TEST_IMAGES[:10]
    analyzer = ActionAnalyzer()

    start_time = time.time()

    for dataset_name, expected_action, image_path in sample_images:
        if not os.path.exists(image_path):
            continue

        analyzer.analyze_action(image_path)

    end_time = time.time()
    avg_time = (end_time - start_time) / len(sample_images)

    print(f"\n{'='*60}")
    print(f"PLAYER ACTION DETECTION PERFORMANCE BENCHMARK")
    print(f"{'='*60}")
    print(f"Average detection time: {avg_time*1000:.2f}ms per image")
    print(f"Images tested: {len(sample_images)}")
    print(f"Method: player_action_model.pt + MediaPipe + action logic")
    print(f"{'='*60}\n")

    # Assert reasonable performance
    assert avg_time < 3.0, f"Average detection time {avg_time:.3f}s exceeds 3.0s threshold"


if __name__ == "__main__":
    # Allow running directly for quick testing
    pytest.main([__file__, "-v", "--tb=short"])
