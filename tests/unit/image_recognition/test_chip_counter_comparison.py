"""
Compare height-based vs detection-based chip counting.
"""

import sys
import os
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '../../..')
sys.path.insert(0, project_root)

from Image_Recognition.chip_pot_analyzer import ChipPotAnalyzer
from Image_Recognition.chip_counter_detection_based import DetectionBasedChipCounter

# Get dataset root
dataset_root = Path(project_root) / "CVDataset"

# Initialize both counters
height_counter = ChipPotAnalyzer()
detection_counter = DetectionBasedChipCounter()

# Chip datasets
colors = ['BlackChips', 'BlueChips', 'RedChips']
counts = [1, 2, 3, 4, 5]

# Results tracking
results_height = {'correct': 0, 'total': 0, 'by_count': {}}
results_detection = {'correct': 0, 'total': 0, 'by_count': {}}

print(f"\n{'='*80}")
print("CHIP COUNTING METHOD COMPARISON")
print(f"{'='*80}\n")

for chip_count in counts:
    results_height['by_count'][chip_count] = {'correct': 0, 'total': 0}
    results_detection['by_count'][chip_count] = {'correct': 0, 'total': 0}

    for color in colors:
        folder_name = f"{chip_count}Chip" if chip_count == 1 else f"{chip_count}Chips"
        folder = dataset_root / color / folder_name

        if not folder.exists():
            continue

        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))

        for img_path in images[:20]:  # Test 20 from each
            # Test height-based
            height_result = height_counter.detect_and_count_chips(str(img_path))
            height_correct = abs(height_result - chip_count) <= 1  # ±1 margin

            # Test detection-based
            detection_result = detection_counter.count_chips(str(img_path))
            detection_count = detection_result['count']
            detection_correct = abs(detection_count - chip_count) <= 1  # ±1 margin

            # Track results
            results_height['total'] += 1
            results_detection['total'] += 1
            results_height['by_count'][chip_count]['total'] += 1
            results_detection['by_count'][chip_count]['total'] += 1

            if height_correct:
                results_height['correct'] += 1
                results_height['by_count'][chip_count]['correct'] += 1

            if detection_correct:
                results_detection['correct'] += 1
                results_detection['by_count'][chip_count]['correct'] += 1

# Print results
print(f"{'='*80}")
print("OVERALL RESULTS")
print(f"{'='*80}")

height_acc = (results_height['correct'] / results_height['total'] * 100) if results_height['total'] > 0 else 0
detection_acc = (results_detection['correct'] / results_detection['total'] * 100) if results_detection['total'] > 0 else 0

print(f"Height-based:    {results_height['correct']:3d}/{results_height['total']:3d} correct ({height_acc:5.1f}%)")
print(f"Detection-based: {results_detection['correct']:3d}/{results_detection['total']:3d} correct ({detection_acc:5.1f}%)")
print(f"\nImprovement: {detection_acc - height_acc:+.1f} percentage points")

# By chip count
print(f"\n{'='*80}")
print("RESULTS BY CHIP COUNT (±1 margin)")
print(f"{'='*80}")
print(f"\n{'Chips':>8} | {'Height-Based':>15} | {'Detection-Based':>17} | {'Improvement':>12}")
print(f"{'-'*80}")

for chip_count in counts:
    h_stats = results_height['by_count'][chip_count]
    d_stats = results_detection['by_count'][chip_count]

    h_acc = (h_stats['correct'] / h_stats['total'] * 100) if h_stats['total'] > 0 else 0
    d_acc = (d_stats['correct'] / d_stats['total'] * 100) if d_stats['total'] > 0 else 0
    improvement = d_acc - h_acc

    print(f"{chip_count:3d} chip{'s' if chip_count > 1 else ' '} | "
          f"{h_stats['correct']:3d}/{h_stats['total']:3d} ({h_acc:5.1f}%) | "
          f"{d_stats['correct']:3d}/{d_stats['total']:3d} ({d_acc:5.1f}%) | "
          f"{improvement:+6.1f}%")

print(f"{'='*80}\n")

if detection_acc > height_acc:
    print(f"✅ Detection-based method is BETTER by {detection_acc - height_acc:.1f} percentage points!")
elif detection_acc < height_acc:
    print(f"❌ Height-based method is better by {height_acc - detection_acc:.1f} percentage points")
else:
    print("⚖️  Both methods perform equally")
