"""
Test enhanced analyzer vs simple analyzer.
"""

import sys
import os
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '../../..')
sys.path.insert(0, project_root)

from Image_Recognition.action_analyzer_enhanced import EnhancedActionAnalyzer

# Get dataset root
dataset_root = Path(project_root) / "CVDataset"

# Initialize analyzer
analyzer = EnhancedActionAnalyzer()

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
confusion_matrix = {
    'FOLD': {'FOLD': 0, 'CHECK': 0, 'BET/RAISE': 0},
    'CHECK': {'FOLD': 0, 'CHECK': 0, 'BET/RAISE': 0},
    'BET/RAISE': {'FOLD': 0, 'CHECK': 0, 'BET/RAISE': 0}
}

print(f"\n{'='*80}")
print("ENHANCED ACTION ANALYZER - FULL DATASET TEST")
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

    print(f"Testing {expected_action} ({len(images)} images)...")

    for img_path in images:
        result = analyzer.analyze_action(str(img_path))
        detected_action = result['action']

        tested += 1
        total_tested += 1

        # Normalize detected action
        if detected_action in ['UNCERTAIN_FOLD']:
            detected_action = 'FOLD'
        elif detected_action in ['UNCERTAIN_CHECK']:
            detected_action = 'CHECK'

        # Update confusion matrix
        if detected_action in confusion_matrix[expected_action]:
            confusion_matrix[expected_action][detected_action] += 1

        # Check if correct
        if detected_action == expected_action:
            correct += 1
            total_correct += 1

        # Print progress every 50 images
        if tested % 50 == 0:
            print(f"  Progress: {tested}/{len(images)} images tested...")

    accuracy = (correct / tested * 100) if tested > 0 else 0
    print(f"  ✓ {expected_action}: {correct}/{tested} correct ({accuracy:.1f}%)\n")

    results_by_action[expected_action] = {
        'correct': correct,
        'total': tested,
        'accuracy': accuracy
    }

# Overall summary
overall_accuracy = (total_correct / total_tested * 100) if total_tested > 0 else 0

print(f"\n{'='*80}")
print("RESULTS SUMMARY")
print(f"{'='*80}")
for action, stats in results_by_action.items():
    print(f"{action:10s}: {stats['correct']:3d}/{stats['total']:3d} correct ({stats['accuracy']:5.1f}%)")
print(f"{'='*80}")
print(f"OVERALL   : {total_correct:3d}/{total_tested:3d} correct ({overall_accuracy:5.1f}%)")
print(f"{'='*80}\n")

# Confusion matrix
print(f"{'='*80}")
print("CONFUSION MATRIX")
print(f"{'='*80}")
print(f"\n{'Expected':<15} | {'BET/RAISE':>10} | {'CHECK':>10} | {'FOLD':>10}")
print(f"{'-'*80}")
for expected in ['BET/RAISE', 'CHECK', 'FOLD']:
    if expected in confusion_matrix:
        print(f"{expected:<15} | {confusion_matrix[expected].get('BET/RAISE', 0):>10} | "
              f"{confusion_matrix[expected].get('CHECK', 0):>10} | "
              f"{confusion_matrix[expected].get('FOLD', 0):>10}")
print(f"{'='*80}\n")

print("COMPARISON:")
print("Simple Analyzer:   90.0% (416/462)")
print(f"Enhanced Analyzer: {overall_accuracy:.1f}% ({total_correct}/{total_tested})")
