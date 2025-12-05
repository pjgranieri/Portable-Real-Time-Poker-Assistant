"""
Test Enhanced Action Analyzer on SampleGameDataset.

This test evaluates the enhanced action analyzer across different betting cycles,
players, and action types using the structured SampleGameDataset.

Expected mappings:
- RedChips folder → BET/RAISE
- BlueChips folder → BET/RAISE
- Fold folder → FOLD
- Check folder → CHECK

Betting cycles tested: FlopBetting, TurnBetting, RiverBetting (PreFlopBetting excluded)
Players tested: P1, P2, P3
"""

import sys
import os
from pathlib import Path

# Add project root to path (2 levels up from tests/visual/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '../..')
sys.path.insert(0, project_root)

from Image_Recognition.action_analyzer_enhanced import EnhancedActionAnalyzer


def test_betting_cycle(analyzer, betting_cycle_path, cycle_name):
    """Test a single betting cycle across all players and actions."""

    # Action folder mapping: folder_name → expected_action
    action_mapping = {
        'RedChips': 'BET/RAISE',
        'BlueChips': 'BET/RAISE',
        'Fold': 'FOLD',
        'Check': 'CHECK'
    }

    # Results tracking for this cycle
    cycle_results = {
        'total_correct': 0,
        'total_tested': 0,
        'by_player': {},
        'by_action': {},
        'confusion_matrix': {
            'BET/RAISE': {'BET/RAISE': 0, 'CHECK': 0, 'FOLD': 0},
            'CHECK': {'BET/RAISE': 0, 'CHECK': 0, 'FOLD': 0},
            'FOLD': {'BET/RAISE': 0, 'CHECK': 0, 'FOLD': 0}
        }
    }

    print(f"\n{'='*80}")
    print(f"TESTING: {cycle_name}")
    print(f"{'='*80}\n")

    # Test each player
    for player in ['P1', 'P2', 'P3']:
        player_path = betting_cycle_path / player

        if not player_path.exists():
            print(f"⚠️  Player folder not found: {player_path}")
            continue

        player_results = {
            'correct': 0,
            'total': 0,
            'by_action': {}
        }

        print(f"\n{player}:")
        print(f"{'-'*40}")

        # Test each action type
        for action_folder, expected_action in action_mapping.items():
            action_path = player_path / action_folder

            if not action_path.exists():
                continue

            # Get all images in this action folder
            images = list(action_path.glob("*.jpg")) + list(action_path.glob("*.png"))

            if not images:
                continue

            # Test each image
            action_correct = 0
            action_tested = len(images)

            for img_path in images:
                result = analyzer.analyze_action(str(img_path))
                detected_action = result['action']

                # Normalize detected action
                if detected_action in ['UNCERTAIN_FOLD']:
                    detected_action = 'FOLD'
                elif detected_action in ['UNCERTAIN_CHECK']:
                    detected_action = 'CHECK'

                # Update confusion matrix
                if detected_action in cycle_results['confusion_matrix'][expected_action]:
                    cycle_results['confusion_matrix'][expected_action][detected_action] += 1

                # Check if correct
                if detected_action == expected_action:
                    action_correct += 1
                    player_results['correct'] += 1
                    cycle_results['total_correct'] += 1

                player_results['total'] += 1
                cycle_results['total_tested'] += 1

            # Calculate accuracy for this action
            action_accuracy = (action_correct / action_tested * 100) if action_tested > 0 else 0

            # Store results
            player_results['by_action'][expected_action] = {
                'correct': action_correct,
                'total': action_tested,
                'accuracy': action_accuracy
            }

            # Update cycle-level action tracking
            if expected_action not in cycle_results['by_action']:
                cycle_results['by_action'][expected_action] = {
                    'correct': 0,
                    'total': 0
                }
            cycle_results['by_action'][expected_action]['correct'] += action_correct
            cycle_results['by_action'][expected_action]['total'] += action_tested

            # Print action results
            print(f"  {action_folder:10s} ({expected_action:10s}): {action_correct:3d}/{action_tested:3d} ({action_accuracy:5.1f}%)")

        # Calculate player accuracy
        player_accuracy = (player_results['correct'] / player_results['total'] * 100) if player_results['total'] > 0 else 0
        print(f"  {'-'*40}")
        print(f"  {player} Total: {player_results['correct']:3d}/{player_results['total']:3d} ({player_accuracy:5.1f}%)")

        # Store player results
        cycle_results['by_player'][player] = player_results

    return cycle_results


def print_cycle_summary(cycle_name, cycle_results):
    """Print summary for a betting cycle."""

    cycle_accuracy = (cycle_results['total_correct'] / cycle_results['total_tested'] * 100) if cycle_results['total_tested'] > 0 else 0

    print(f"\n{'='*80}")
    print(f"{cycle_name} SUMMARY")
    print(f"{'='*80}")

    # Per-action summary
    print(f"\nBy Action Type:")
    print(f"{'-'*40}")
    for action, stats in cycle_results['by_action'].items():
        action_acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {action:10s}: {stats['correct']:3d}/{stats['total']:3d} ({action_acc:5.1f}%)")

    # Per-player summary
    print(f"\nBy Player:")
    print(f"{'-'*40}")
    for player, stats in cycle_results['by_player'].items():
        player_acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {player}: {stats['correct']:3d}/{stats['total']:3d} ({player_acc:5.1f}%)")

    print(f"\n{cycle_name} Overall: {cycle_results['total_correct']:3d}/{cycle_results['total_tested']:3d} ({cycle_accuracy:5.1f}%)")
    print(f"{'='*80}")

    # Confusion matrix
    print(f"\n{cycle_name} Confusion Matrix:")
    print(f"{'-'*80}")
    print(f"{'Expected':<15} | {'BET/RAISE':>10} | {'CHECK':>10} | {'FOLD':>10}")
    print(f"{'-'*80}")
    for expected in ['BET/RAISE', 'CHECK', 'FOLD']:
        if expected in cycle_results['confusion_matrix']:
            print(f"{expected:<15} | {cycle_results['confusion_matrix'][expected].get('BET/RAISE', 0):>10} | "
                  f"{cycle_results['confusion_matrix'][expected].get('CHECK', 0):>10} | "
                  f"{cycle_results['confusion_matrix'][expected].get('FOLD', 0):>10}")
    print(f"{'='*80}\n")


def main():
    """Run the full test suite on SampleGameDataset."""

    # Get dataset root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = Path(script_dir).parent / "SampleGameDataset" / "BettingCycles"

    if not dataset_root.exists():
        print(f"❌ Error: Dataset not found at {dataset_root}")
        return

    # Initialize analyzer
    print("Initializing Enhanced Action Analyzer...")
    analyzer = EnhancedActionAnalyzer()

    # Betting cycles to test (excluding PreFlopBetting as requested)
    betting_cycles = ['FlopBetting', 'TurnBetting', 'RiverBetting']

    # Overall results tracking
    overall_results = {
        'total_correct': 0,
        'total_tested': 0,
        'by_cycle': {},
        'overall_confusion_matrix': {
            'BET/RAISE': {'BET/RAISE': 0, 'CHECK': 0, 'FOLD': 0},
            'CHECK': {'BET/RAISE': 0, 'CHECK': 0, 'FOLD': 0},
            'FOLD': {'BET/RAISE': 0, 'CHECK': 0, 'FOLD': 0}
        }
    }

    print(f"\n{'='*80}")
    print("ENHANCED ACTION ANALYZER - SAMPLE GAME DATASET TEST")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_root}")
    print(f"Betting Cycles: {', '.join(betting_cycles)}")
    print(f"Players: P1, P2, P3")
    print(f"Actions: RedChips, BlueChips, Fold, Check")
    print(f"{'='*80}")

    # Test each betting cycle
    for cycle_name in betting_cycles:
        cycle_path = dataset_root / cycle_name

        if not cycle_path.exists():
            print(f"⚠️  Betting cycle not found: {cycle_path}")
            continue

        # Test this cycle
        cycle_results = test_betting_cycle(analyzer, cycle_path, cycle_name)

        # Print cycle summary
        print_cycle_summary(cycle_name, cycle_results)

        # Update overall results
        overall_results['total_correct'] += cycle_results['total_correct']
        overall_results['total_tested'] += cycle_results['total_tested']
        overall_results['by_cycle'][cycle_name] = {
            'correct': cycle_results['total_correct'],
            'total': cycle_results['total_tested'],
            'accuracy': (cycle_results['total_correct'] / cycle_results['total_tested'] * 100) if cycle_results['total_tested'] > 0 else 0
        }

        # Merge confusion matrices
        for expected in ['BET/RAISE', 'CHECK', 'FOLD']:
            for detected in ['BET/RAISE', 'CHECK', 'FOLD']:
                overall_results['overall_confusion_matrix'][expected][detected] += \
                    cycle_results['confusion_matrix'][expected][detected]

    # Print overall summary
    overall_accuracy = (overall_results['total_correct'] / overall_results['total_tested'] * 100) if overall_results['total_tested'] > 0 else 0

    print(f"\n{'='*80}")
    print("OVERALL RESULTS - ALL BETTING CYCLES")
    print(f"{'='*80}")

    print(f"\nBy Betting Cycle:")
    print(f"{'-'*40}")
    for cycle_name, stats in overall_results['by_cycle'].items():
        print(f"  {cycle_name:15s}: {stats['correct']:3d}/{stats['total']:3d} ({stats['accuracy']:5.1f}%)")

    print(f"\n{'='*80}")
    print(f"OVERALL ACCURACY: {overall_results['total_correct']:3d}/{overall_results['total_tested']:3d} ({overall_accuracy:5.1f}%)")
    print(f"{'='*80}")

    # Overall confusion matrix
    print(f"\nOverall Confusion Matrix:")
    print(f"{'-'*80}")
    print(f"{'Expected':<15} | {'BET/RAISE':>10} | {'CHECK':>10} | {'FOLD':>10}")
    print(f"{'-'*80}")
    for expected in ['BET/RAISE', 'CHECK', 'FOLD']:
        print(f"{expected:<15} | {overall_results['overall_confusion_matrix'][expected]['BET/RAISE']:>10} | "
              f"{overall_results['overall_confusion_matrix'][expected]['CHECK']:>10} | "
              f"{overall_results['overall_confusion_matrix'][expected]['FOLD']:>10}")
    print(f"{'='*80}\n")

    # Performance assessment
    print("PERFORMANCE ASSESSMENT:")
    print(f"{'-'*80}")
    if overall_accuracy >= 90:
        print(f"✓ EXCELLENT: {overall_accuracy:.1f}% accuracy (Target: >90%)")
    elif overall_accuracy >= 80:
        print(f"○ GOOD: {overall_accuracy:.1f}% accuracy (Target: >90%)")
    elif overall_accuracy >= 70:
        print(f"△ FAIR: {overall_accuracy:.1f}% accuracy (Target: >90%)")
    else:
        print(f"✗ NEEDS IMPROVEMENT: {overall_accuracy:.1f}% accuracy (Target: >90%)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
