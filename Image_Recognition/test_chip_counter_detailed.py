"""
Detailed analysis of detection-based chip counter performance by chip count.
"""

from chip_counter_detection_based import DetectionBasedChipCounter
from pathlib import Path
import os
from collections import defaultdict

# Get dataset root
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_root = Path(script_dir).parent / "CVDataset"

# Initialize counter
counter = DetectionBasedChipCounter()

# Chip datasets
colors = ['BlackChips', 'BlueChips', 'RedChips']
counts = [1, 2, 3, 4, 5]

# Results tracking
results_by_count = {}
confusion_matrix = defaultdict(lambda: defaultdict(int))

print(f"\n{'='*80}")
print("DETECTION-BASED CHIP COUNTER - DETAILED ANALYSIS")
print(f"{'='*80}\n")

for expected_count in counts:
    results_by_count[expected_count] = {
        'exact_correct': 0,
        'within_1': 0,
        'total': 0,
        'predictions': defaultdict(int)
    }

    print(f"Testing {expected_count}-chip images:")
    print("-" * 60)

    for color in colors:
        folder_name = f"{expected_count}Chip" if expected_count == 1 else f"{expected_count}Chips"
        folder = dataset_root / color / folder_name

        if not folder.exists():
            continue

        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))

        for img_path in images[:20]:  # Test 20 from each color
            result = counter.count_chips(str(img_path))
            detected_count = result['count']
            method = result['method']

            # Track results
            results_by_count[expected_count]['total'] += 1
            results_by_count[expected_count]['predictions'][detected_count] += 1
            confusion_matrix[expected_count][detected_count] += 1

            # Check accuracy
            if detected_count == expected_count:
                results_by_count[expected_count]['exact_correct'] += 1
                results_by_count[expected_count]['within_1'] += 1
            elif abs(detected_count - expected_count) <= 1:
                results_by_count[expected_count]['within_1'] += 1

    # Print summary for this chip count
    stats = results_by_count[expected_count]
    total = stats['total']
    exact = stats['exact_correct']
    within_1 = stats['within_1']

    exact_acc = (exact / total * 100) if total > 0 else 0
    within_1_acc = (within_1 / total * 100) if total > 0 else 0

    print(f"  Total images: {total}")
    print(f"  Exact match: {exact}/{total} ({exact_acc:.1f}%)")
    print(f"  Within ±1:   {within_1}/{total} ({within_1_acc:.1f}%)")
    print(f"  Predictions breakdown:")
    for pred_count in sorted(stats['predictions'].keys()):
        count = stats['predictions'][pred_count]
        pct = (count / total * 100) if total > 0 else 0
        marker = "✓" if pred_count == expected_count else ("~" if abs(pred_count - expected_count) <= 1 else "✗")
        print(f"    {marker} Detected {pred_count} chips: {count:3d} ({pct:5.1f}%)")
    print()

# Overall summary
print(f"\n{'='*80}")
print("OVERALL SUMMARY")
print(f"{'='*80}\n")

total_images = sum(stats['total'] for stats in results_by_count.values())
total_exact = sum(stats['exact_correct'] for stats in results_by_count.values())
total_within_1 = sum(stats['within_1'] for stats in results_by_count.values())

overall_exact_acc = (total_exact / total_images * 100) if total_images > 0 else 0
overall_within_1_acc = (total_within_1 / total_images * 100) if total_images > 0 else 0

print(f"Total images tested: {total_images}")
print(f"Exact matches: {total_exact}/{total_images} ({overall_exact_acc:.1f}%)")
print(f"Within ±1 chip: {total_within_1}/{total_images} ({overall_within_1_acc:.1f}%)")

# Accuracy by chip count
print(f"\n{'='*80}")
print("ACCURACY BY CHIP COUNT")
print(f"{'='*80}\n")

print(f"{'Chips':>6} | {'Exact Match':>15} | {'Within ±1':>15} | {'Status':>10}")
print("-" * 80)

for chip_count in counts:
    stats = results_by_count[chip_count]
    total = stats['total']
    exact = stats['exact_correct']
    within_1 = stats['within_1']

    exact_acc = (exact / total * 100) if total > 0 else 0
    within_1_acc = (within_1 / total * 100) if total > 0 else 0

    status = "✅" if within_1_acc >= 80 else ("⚠️" if within_1_acc >= 50 else "❌")

    print(f"{chip_count:3d}    | {exact:3d}/{total:3d} ({exact_acc:5.1f}%) | "
          f"{within_1:3d}/{total:3d} ({within_1_acc:5.1f}%) | {status}")

# Confusion matrix
print(f"\n{'='*80}")
print("CONFUSION MATRIX")
print(f"{'='*80}\n")

print(f"{'Expected':>8} |", end="")
for detected in counts:
    print(f" {detected} chips |", end="")
print()
print("-" * 80)

for expected in counts:
    print(f"{expected:3d} chips |", end="")
    for detected in counts:
        count = confusion_matrix[expected][detected]
        print(f"{count:7d} |", end="")
    print()

print(f"\n{'='*80}")
print("KEY INSIGHTS")
print(f"{'='*80}\n")

# Find best and worst performing
best = max(counts, key=lambda c: results_by_count[c]['within_1'] / results_by_count[c]['total'])
worst = min(counts, key=lambda c: results_by_count[c]['within_1'] / results_by_count[c]['total'])

best_acc = (results_by_count[best]['within_1'] / results_by_count[best]['total'] * 100)
worst_acc = (results_by_count[worst]['within_1'] / results_by_count[worst]['total'] * 100)

print(f"✅ Best performance: {best}-chip detection ({best_acc:.1f}% within ±1)")
print(f"❌ Worst performance: {worst}-chip detection ({worst_acc:.1f}% within ±1)")

# Common misclassifications
print(f"\nCommon misclassifications:")
for expected in counts:
    wrong_preds = [(det, confusion_matrix[expected][det])
                   for det in counts
                   if det != expected and confusion_matrix[expected][det] > 0]

    if wrong_preds:
        wrong_preds.sort(key=lambda x: -x[1])
        most_common = wrong_preds[0]
        print(f"  {expected} chips → {most_common[0]} chips: {most_common[1]} times")

print()
