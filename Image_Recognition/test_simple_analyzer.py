"""
Test the simple action analyzer to see if removing filters improves accuracy.
"""

from action_analyzer_simple import SimpleActionAnalyzer
from pathlib import Path
import os

# Get dataset root
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_root = Path(script_dir).parent / "CVDataset"

# Initialize analyzer
analyzer = SimpleActionAnalyzer()

# Test datasets
test_sets = {
    'FOLD': dataset_root / "FoldedCardsDataset",
    'CHECK': dataset_root / "HandDataset",
    'BET/RAISE': dataset_root / "ChipDataset"
}

# Results tracking
total_correct = 0
total_tested = 0
results_by_action = {}

print(f"\n{'='*80}")
print("SIMPLE ACTION ANALYZER TEST")
print(f"{'='*80}\n")

for expected_action, folder in test_sets.items():
    if not folder.exists():
        print(f"⚠️  Folder not found: {folder}")
        continue

    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))

    if not images:
        print(f"⚠️  No images found in {folder}")
        continue

    correct = 0
    tested = 0

    print(f"\nTesting {expected_action} ({len(images)} images):")
    print("-" * 60)

    # Test first 20 images from each dataset
    for img_path in images[:20]:
        result = analyzer.analyze_action(str(img_path))
        detected_action = result['action']

        tested += 1
        total_tested += 1

        # Check if correct
        is_correct = False
        if expected_action == 'FOLD' and detected_action in ['FOLD', 'UNCERTAIN_FOLD']:
            is_correct = True
        elif expected_action == 'CHECK' and detected_action in ['CHECK', 'UNCERTAIN_CHECK']:
            is_correct = True
        elif expected_action == 'BET/RAISE' and detected_action == 'BET/RAISE':
            is_correct = True

        if is_correct:
            correct += 1
            total_correct += 1
            status = "✓"
        else:
            status = "✗"

        # Print every 5th result to avoid clutter
        if tested % 5 == 0 or not is_correct:
            print(f"  [{tested:2d}] {status} Expected: {expected_action:10s} | Detected: {detected_action:15s} | {img_path.name[:30]}")

    accuracy = (correct / tested * 100) if tested > 0 else 0
    print(f"\n  {expected_action}: {correct}/{tested} correct ({accuracy:.1f}%)")

    results_by_action[expected_action] = {
        'correct': correct,
        'total': tested,
        'accuracy': accuracy
    }

# Overall summary
overall_accuracy = (total_correct / total_tested * 100) if total_tested > 0 else 0

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
for action, stats in results_by_action.items():
    print(f"{action:10s}: {stats['correct']:3d}/{stats['total']:3d} correct ({stats['accuracy']:5.1f}%)")
print(f"{'='*80}")
print(f"OVERALL   : {total_correct:3d}/{total_tested:3d} correct ({overall_accuracy:5.1f}%)")
print(f"{'='*80}\n")

# Comparison with complex analyzer
print("NOTE: This is the SIMPLE analyzer (no filters).")
print("Compare this to the complex analyzer results to see if filters are the issue.\n")
