#!/usr/bin/env python3
"""
Test chip color detection accuracy for DetectionBasedChipCounter
Tests red ($5) and blue ($10) chip detection on dataset images
Provides statistics on detection accuracy
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Image_Recognition.chip_counter_detection_based import DetectionBasedChipCounter


class ChipColorDetectionTest:
    """Test chip color detection on dataset images"""
    
    def __init__(self):
        self.chip_counter = DetectionBasedChipCounter()
        
        # Expected chip values
        self.expected_values = {
            'red': 5,
            'blue': 10
        }
        
        # Statistics
        self.stats = {
            'red': {'total': 0, 'correct': 0, 'incorrect': 0, 'not_detected': 0},
            'blue': {'total': 0, 'correct': 0, 'incorrect': 0, 'not_detected': 0}
        }
        
        # Test directories
        self.test_dirs = [
            'CVDataset/RedChips/1Chip',
            'CVDataset/BlueChips/1Chip',
            'SampleGameDataset/PotAmounts/Red',
            'SampleGameDataset/PotAmounts/Blue'
        ]
    
    def test_image(self, image_path, expected_color):
        """
        Test a single image for chip color detection
        
        Args:
            image_path: Path to test image
            expected_color: Expected color ('red' or 'blue')
        
        Returns:
            bool: True if detection was correct
        """
        expected_value = self.expected_values[expected_color]
        
        # Analyze image
        result = self.chip_counter.detect_chip_color_and_value(str(image_path))
        
        detected_color = result.get('color', None)
        detected_value = result.get('value', 0)
        confidence = result.get('confidence', 0)
        chip_count = result.get('chip_count', 0)
        
        # Check if chip was detected
        if detected_color is None or chip_count == 0:
            print(f"  ❌ FAILED: No chip detected")
            print(f"     Result: {result}")
            self.stats[expected_color]['not_detected'] += 1
            return False
        
        # Check if color and value match
        if detected_color == expected_color and detected_value == expected_value:
            print(f"  ✅ SUCCESS: Detected {detected_color} chip (${detected_value})")
            print(f"     Confidence: {confidence:.2%}, Chip count: {chip_count}")
            self.stats[expected_color]['correct'] += 1
            return True
        else:
            print(f"  ❌ FAILED: Detected {detected_color} chip (${detected_value}), expected {expected_color} (${expected_value})")
            print(f"     Confidence: {confidence:.2%}, Chip count: {chip_count}")
            self.stats[expected_color]['incorrect'] += 1
            return False
    
    def test_directory(self, dir_path):
        """
        Test all images in a directory
        
        Args:
            dir_path: Path to directory containing chip images
        """
        # Get color from directory path
        path_parts = Path(dir_path).parts
        if 'RedChips' in path_parts or 'Red' in path_parts:
            color = 'red'
        elif 'BlueChips' in path_parts or 'Blue' in path_parts:
            color = 'blue'
        else:
            print(f"⚠️  Cannot determine color from path: {dir_path}")
            return
        
        # Check if directory exists
        if not os.path.exists(dir_path):
            print(f"⚠️  Directory not found: {dir_path}")
            return
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(dir_path).glob(f'*{ext}'))
            image_files.extend(Path(dir_path).glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"⚠️  No images found in {dir_path}")
            return
        
        print(f"\n{'='*70}")
        print(f"Testing {color.upper()} chips from: {dir_path}")
        print(f"Expected value: ${self.expected_values[color]}")
        print(f"Total images: {len(image_files)}")
        print(f"{'='*70}")
        
        # Test each image
        for i, image_path in enumerate(sorted(image_files), 1):
            print(f"\n[{i}/{len(image_files)}] Testing: {image_path.name}")
            self.stats[color]['total'] += 1
            self.test_image(image_path, color)
    
    def run_all_tests(self):
        """Run tests on all directories"""
        print("\n" + "="*70)
        print(" CHIP COLOR DETECTION TEST")
        print(" Testing DetectionBasedChipCounter chip detection")
        print(" Red chips = $5, Blue chips = $10")
        print("="*70)
        
        # Test each directory
        for dir_path in self.test_dirs:
            self.test_directory(dir_path)
        
        # Print final statistics
        self.print_statistics()
    
    def print_statistics(self):
        """Print final statistics"""
        print("\n" + "="*70)
        print(" FINAL STATISTICS")
        print("="*70)
        
        total_all = 0
        correct_all = 0
        
        for color in ['red', 'blue']:
            stats = self.stats[color]
            total = stats['total']
            correct = stats['correct']
            incorrect = stats['incorrect']
            not_detected = stats['not_detected']
            
            if total == 0:
                accuracy = 0.0
            else:
                accuracy = (correct / total) * 100
            
            print(f"\n{color.upper()} CHIPS (${self.expected_values[color]}):")
            print(f"  Total images:     {total}")
            print(f"  Correctly detected: {correct} ({correct}/{total})")
            print(f"  Incorrectly detected: {incorrect} ({incorrect}/{total})")
            print(f"  Not detected:     {not_detected} ({not_detected}/{total})")
            print(f"  Accuracy:         {accuracy:.1f}%")
            
            total_all += total
            correct_all += correct
        
        print(f"\n{'='*70}")
        print(f"OVERALL ACCURACY:")
        if total_all > 0:
            overall_accuracy = (correct_all / total_all) * 100
            print(f"  {correct_all}/{total_all} = {overall_accuracy:.1f}%")
        else:
            print(f"  No images tested")
        print(f"{'='*70}\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test chip color detection accuracy'
    )
    parser.add_argument(
        '--dir',
        type=str,
        help='Test a specific directory only (e.g., CVDataset/RedChips/1Chip)'
    )
    
    args = parser.parse_args()
    
    tester = ChipColorDetectionTest()
    
    if args.dir:
        # Test specific directory
        tester.test_directory(args.dir)
        tester.print_statistics()
    else:
        # Test all directories
        tester.run_all_tests()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()