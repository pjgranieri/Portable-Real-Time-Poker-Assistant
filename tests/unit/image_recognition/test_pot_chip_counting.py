"""
Comprehensive unit tests for pot chip detection and counting.

Tests the pot_chips_model.pt YOLO model paired with height-based chip counting:
1. pot_chips_model detects chip regions in pot (WHERE chips are)
2. Bounding box HEIGHT determines chip count (HOW MANY chips stacked)

This tests across all colors (Black, Blue, Red) and quantities (1-5 chips).

Run with:
    pytest tests/unit/image_recognition/test_pot_chip_counting.py -v

Run with detailed report:
    pytest tests/unit/image_recognition/test_pot_chip_counting.py -v --tb=short

Run in parallel:
    pytest tests/unit/image_recognition/test_pot_chip_counting.py -n auto
"""

import pytest
import cv2
import os
from pathlib import Path
from collections import defaultdict

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from Image_Recognition.chip_pot_analyzer import ChipPotAnalyzer


# Dataset configuration
DATASET_ROOT = Path(__file__).parent.parent.parent.parent / "CVDataset"
CHIP_COLORS = ["BlackChips", "BlueChips", "RedChips"]
CHIP_COUNTS = [1, 2, 3, 4, 5]


def get_color_name(color_folder):
    """
    Convert folder name to color string.

    Args:
        color_folder: One of "BlackChips", "BlueChips", "RedChips"

    Returns:
        str: 'black', 'blue', or 'red'
    """
    color_map = {
        "BlackChips": "black",
        "BlueChips": "blue",
        "RedChips": "red",
    }
    return color_map[color_folder]


def discover_test_images():
    """
    Dynamically discover all test images in the CVDataset directory.

    Yields:
        tuple: (color_name, expected_count, image_path)
    """
    test_cases = []

    for color in CHIP_COLORS:
        for count in CHIP_COUNTS:
            folder_name = f"{count}Chip" if count == 1 else f"{count}Chips"
            folder_path = DATASET_ROOT / color / folder_name

            if not folder_path.exists():
                print(f"Warning: Folder not found: {folder_path}")
                continue

            # Get all image files
            image_files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))

            for image_path in image_files:
                test_cases.append((color, count, str(image_path)))

    return test_cases


# Generate all test cases
TEST_IMAGES = discover_test_images()


@pytest.fixture(scope="module")
def pot_analyzer():
    """Create a ChipPotAnalyzer instance for all tests."""
    return ChipPotAnalyzer()


@pytest.mark.parametrize("color,expected_count,image_path", TEST_IMAGES)
def test_pot_chip_count_accuracy(pot_analyzer, color, expected_count, image_path):
    """
    Test that the pot chip analyzer correctly identifies the number of chips.

    Uses pot_chips_model.pt to detect chip regions, then height-based counting
    to determine quantity.

    Args:
        pot_analyzer: ChipPotAnalyzer fixture
        color: Chip color name (e.g., "BlackChips")
        expected_count: Expected number of chips in the image
        image_path: Path to the test image
    """
    # Verify image exists
    assert os.path.exists(image_path), f"Image not found: {image_path}"

    # Get color string ('black', 'blue', 'red')
    color_str = get_color_name(color)

    # Detect and count chips in pot
    detected_count = pot_analyzer.detect_and_count_chips(
        image_path,
        color=color_str,
        conf_threshold=0.25
    )

    # Assert correct count (allow ±1 chip error)
    error_margin = abs(detected_count - expected_count)
    assert error_margin <= 1, (
        f"Pot chip count mismatch for {Path(image_path).name}:\n"
        f"  Expected: {expected_count} chips\n"
        f"  Detected: {detected_count} chips\n"
        f"  Error: {error_margin} chips (allowed: ±1)\n"
        f"  Color: {color} ({color_str})\n"
        f"  Image: {image_path}"
    )


class TestPotChipDetectionAccuracy:
    """Test suite for overall pot chip detection accuracy metrics."""

    @pytest.fixture(scope="class")
    def detection_results(self):
        """
        Run detection on all images and collect results.

        Returns:
            dict: Results organized by color and chip count
        """
        results = defaultdict(lambda: defaultdict(list))
        analyzer = ChipPotAnalyzer()

        print(f"\n[TEST] Running pot chip detection on {len(TEST_IMAGES)} images...")

        for color, expected_count, image_path in TEST_IMAGES:
            # Skip if image doesn't exist
            if not os.path.exists(image_path):
                continue

            # Get color string
            color_str = get_color_name(color)

            # Detect chips in pot
            detected_count = analyzer.detect_and_count_chips(
                image_path,
                color=color_str,
                conf_threshold=0.25
            )

            # Store result (±1 chip margin allowed)
            error_margin = abs(detected_count - expected_count)
            results[color][expected_count].append({
                'expected': expected_count,
                'detected': detected_count,
                'correct': error_margin <= 1,  # Allow ±1 chip error
                'error_margin': error_margin,
                'image': Path(image_path).name
            })

        return results

    def test_overall_accuracy(self, detection_results):
        """Test that overall pot chip counting accuracy is acceptable."""
        total_images = 0
        correct_detections = 0

        for color in detection_results:
            for count in detection_results[color]:
                for result in detection_results[color][count]:
                    total_images += 1
                    if result['correct']:
                        correct_detections += 1

        accuracy = (correct_detections / total_images * 100) if total_images > 0 else 0

        print(f"\n{'='*60}")
        print(f"OVERALL POT CHIP DETECTION ACCURACY (±1 chip margin)")
        print(f"{'='*60}")
        print(f"Total images tested: {total_images}")
        print(f"Correct detections: {correct_detections}")
        print(f"Incorrect detections: {total_images - correct_detections}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"{'='*60}\n")

        # Assert at least 70% accuracy
        assert accuracy >= 70.0, f"Overall accuracy {accuracy:.2f}% is below 70% threshold"

    def test_accuracy_by_color(self, detection_results):
        """Test accuracy broken down by chip color."""
        print(f"\n{'='*60}")
        print(f"POT CHIP ACCURACY BY COLOR")
        print(f"{'='*60}")

        for color in CHIP_COLORS:
            if color not in detection_results:
                continue

            total = 0
            correct = 0

            for count in detection_results[color]:
                for result in detection_results[color][count]:
                    total += 1
                    if result['correct']:
                        correct += 1

            accuracy = (correct / total * 100) if total > 0 else 0

            print(f"{color:12s}: {correct:3d}/{total:3d} correct ({accuracy:5.2f}%)")

            # Assert at least 60% accuracy per color
            assert accuracy >= 60.0, f"{color} accuracy {accuracy:.2f}% is below 60% threshold"

        print(f"{'='*60}\n")

    def test_accuracy_by_chip_count(self, detection_results):
        """Test accuracy broken down by number of chips."""
        print(f"\n{'='*60}")
        print(f"POT CHIP ACCURACY BY CHIP COUNT")
        print(f"{'='*60}")

        for chip_count in CHIP_COUNTS:
            total = 0
            correct = 0

            for color in detection_results:
                if chip_count in detection_results[color]:
                    for result in detection_results[color][chip_count]:
                        total += 1
                        if result['correct']:
                            correct += 1

            accuracy = (correct / total * 100) if total > 0 else 0

            print(f"{chip_count} chip{'s' if chip_count > 1 else ' ':1s}: {correct:3d}/{total:3d} correct ({accuracy:5.2f}%)")

            # Assert at least 60% accuracy per chip count
            assert accuracy >= 60.0, f"{chip_count} chips accuracy {accuracy:.2f}% is below 60% threshold"

        print(f"{'='*60}\n")

    def test_failure_analysis(self, detection_results):
        """Print detailed analysis of failed detections."""
        print(f"\n{'='*60}")
        print(f"POT CHIP DETECTION FAILURE ANALYSIS")
        print(f"{'='*60}")

        failures_by_type = defaultdict(int)

        for color in detection_results:
            for count in detection_results[color]:
                for result in detection_results[color][count]:
                    if not result['correct']:
                        error_margin = result.get('error_margin', abs(result['detected'] - result['expected']))
                        error_type = f"Expected {result['expected']}, got {result['detected']} (error: {error_margin} chips)"
                        failures_by_type[error_type] += 1

        if not failures_by_type:
            print("No failures detected!")
        else:
            print("Common failure patterns:")
            for error_type, count in sorted(failures_by_type.items(), key=lambda x: -x[1]):
                print(f"  {error_type}: {count} occurrences")

        print(f"{'='*60}\n")


@pytest.mark.slow
def test_pot_detection_speed_benchmark():
    """Benchmark the speed of pot chip detection."""
    import time

    if not TEST_IMAGES:
        pytest.skip("No test images available")

    # Test on first 10 images
    sample_images = TEST_IMAGES[:10]
    analyzer = ChipPotAnalyzer()

    start_time = time.time()

    for color, expected_count, image_path in sample_images:
        if not os.path.exists(image_path):
            continue

        color_str = get_color_name(color)
        analyzer.detect_and_count_chips(image_path, color=color_str, conf_threshold=0.25)

    end_time = time.time()
    avg_time = (end_time - start_time) / len(sample_images)

    print(f"\n{'='*60}")
    print(f"POT CHIP DETECTION PERFORMANCE BENCHMARK")
    print(f"{'='*60}")
    print(f"Average detection time: {avg_time*1000:.2f}ms per image")
    print(f"Images tested: {len(sample_images)}")
    print(f"Method: pot_chips_model.pt + height-based counting")
    print(f"{'='*60}\n")

    # Assert reasonable performance
    assert avg_time < 2.0, f"Average detection time {avg_time:.3f}s exceeds 2.0s threshold"


if __name__ == "__main__":
    # Allow running directly for quick testing
    pytest.main([__file__, "-v", "--tb=short"])
