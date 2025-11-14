#!/usr/bin/env python3
"""
Test CV detection accuracy on labeled test images
Tests fold detection, check detection, and chip counting
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Image_Recognition.action_analyzer_algorithm import ActionAnalyzer

class CVDetectionTester:
    def __init__(self):
        # Initialize action analyzer with same thresholds as test_game
        self.action_analyzer = ActionAnalyzer(confidence_threshold=0.5)
        
        # Test data directory
        self.test_dir = os.path.join('Outputs', 'Test_Outputs')
        
        # Results tracking
        self.results = {
            'folds': {'correct': 0, 'total': 0, 'details': []},
            'checks': {'correct': 0, 'total': 0, 'details': []},
            'chips': {'correct': 0, 'total': 0, 'details': []},
            'chip_amounts': {'correct': 0, 'total': 0, 'details': []}
        }
    
    def detect_chips(self, image_path):
        """Detect chips using ActionAnalyzer (uses cropped ROI internally)"""
        chips_present, chip_labels = self.action_analyzer.check_chips(image_path)
        chip_count = len(chip_labels)
        bet_amount = chip_count * 5
        return chip_count, bet_amount
    
    def test_fold_detection(self):
        """Test fold detection on labeled fold images"""
        print("\n" + "="*60)
        print("TESTING FOLD DETECTION")
        print("="*60)
        
        fold_images = [
            'fold.jpg',
            'test_fold.jpg',
            'test_fold_three.jpg',
            'test_fold_four.jpg'
        ]
        
        for img_name in fold_images:
            img_path = os.path.join(self.test_dir, img_name)
            
            if not os.path.exists(img_path):
                print(f"⚠️  SKIP: {img_name} not found")
                continue
            
            self.results['folds']['total'] += 1
            
            print(f"\n📸 Testing: {img_name}")
            result = self.action_analyzer.analyze_action(img_path)
            action = result['action']
            
            is_correct = action == 'FOLD'
            
            if is_correct:
                self.results['folds']['correct'] += 1
                print(f"   ✅ PASS - Detected: {action}")
            else:
                print(f"   ❌ FAIL - Expected: FOLD, Got: {action}")
                print(f"   Details: {result['details']}")
            
            self.results['folds']['details'].append({
                'image': img_name,
                'expected': 'FOLD',
                'detected': action,
                'correct': is_correct
            })
    
    def test_check_detection(self):
        """Test check/hand detection on labeled check images"""
        print("\n" + "="*60)
        print("TESTING CHECK DETECTION")
        print("="*60)
        
        check_images = [
            'test_check_one.jpg',
            'test_check_two.jpg',
            'test_check_three.jpg',
            'test_check_four.jpg'
        ]
        
        for img_name in check_images:
            img_path = os.path.join(self.test_dir, img_name)
            
            if not os.path.exists(img_path):
                print(f"⚠️  SKIP: {img_name} not found")
                continue
            
            self.results['checks']['total'] += 1
            
            print(f"\n📸 Testing: {img_name}")
            result = self.action_analyzer.analyze_action(img_path)
            action = result['action']
            
            is_correct = action == 'CHECK'
            
            if is_correct:
                self.results['checks']['correct'] += 1
                print(f"   ✅ PASS - Detected: {action}")
            else:
                print(f"   ❌ FAIL - Expected: CHECK, Got: {action}")
                print(f"   Details: {result['details']}")
            
            self.results['checks']['details'].append({
                'image': img_name,
                'expected': 'CHECK',
                'detected': action,
                'correct': is_correct
            })
    
    def test_chip_detection(self):
        """Test chip/bet detection on labeled chip images"""
        print("\n" + "="*60)
        print("TESTING CHIP/BET DETECTION")
        print("="*60)
        
        chip_images = [
            'test_chips_one.jpg',
            'test_chips_two.jpg',
            'test_chips_three.jpg',
            'test_chips_four.jpg',
            'test_chips_five.jpg',
            'test_chips_six.jpg',
            'test_chips_seven.jpg',
            'test_chips_eight.jpg',
            'test_chips_nine.jpg',
            'test_chips_ten.jpg'
        ]
        
        for img_name in chip_images:
            img_path = os.path.join(self.test_dir, img_name)
            
            if not os.path.exists(img_path):
                print(f"⚠️  SKIP: {img_name} not found")
                continue
            
            self.results['chips']['total'] += 1
            
            print(f"\n📸 Testing: {img_name}")
            
            # Check if chips are detected (either through action analyzer or chip model)
            result = self.action_analyzer.analyze_action(img_path)
            action = result['action']
            
            chip_count, bet_amount = self.detect_chips(img_path)
            
            # Consider it correct if either method detects chips
            is_correct = (action == 'BET/RAISE') or (chip_count > 0)
            
            if is_correct:
                self.results['chips']['correct'] += 1
                print(f"   ✅ PASS - Detected: {action} ({chip_count} chips, ${bet_amount})")
            else:
                print(f"   ❌ FAIL - Expected: BET/RAISE, Got: {action}")
                print(f"   Chip count: {chip_count}, Amount: ${bet_amount}")
                print(f"   Details: {result['details']}")
            
            self.results['chips']['details'].append({
                'image': img_name,
                'expected': 'BET/RAISE',
                'detected': action,
                'chip_count': chip_count,
                'bet_amount': bet_amount,
                'correct': is_correct
            })
    
    def test_chip_amounts(self):
        """Test chip amount counting accuracy"""
        print("\n" + "="*60)
        print("TESTING CHIP AMOUNT COUNTING")
        print("="*60)
        
        amount_tests = [
            ('test_one_chip.jpg', 1, 5),
            ('test_two_chips.jpg', 2, 10),
            ('test_four_chips.jpg', 4, 20),
            ('test_five_chips.jpg', 5, 25)
        ]
        
        for img_name, expected_count, expected_amount in amount_tests:
            img_path = os.path.join(self.test_dir, img_name)
            
            if not os.path.exists(img_path):
                print(f"⚠️  SKIP: {img_name} not found")
                continue
            
            self.results['chip_amounts']['total'] += 1
            
            print(f"\n📸 Testing: {img_name}")
            print(f"   Expected: {expected_count} chips = ${expected_amount}")
            
            chip_count, bet_amount = self.detect_chips(img_path)
            
            is_correct = (chip_count == expected_count) and (bet_amount == expected_amount)
            
            if is_correct:
                self.results['chip_amounts']['correct'] += 1
                print(f"   ✅ PASS - Detected: {chip_count} chips = ${bet_amount}")
            else:
                print(f"   ❌ FAIL - Detected: {chip_count} chips = ${bet_amount}")
            
            self.results['chip_amounts']['details'].append({
                'image': img_name,
                'expected_count': expected_count,
                'expected_amount': expected_amount,
                'detected_count': chip_count,
                'detected_amount': bet_amount,
                'correct': is_correct
            })
    
    def print_summary(self):
        """Print summary of test results"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        total_tests = 0
        total_correct = 0
        
        # Folds
        fold_total = self.results['folds']['total']
        fold_correct = self.results['folds']['correct']
        if fold_total > 0:
            fold_rate = (fold_correct / fold_total) * 100
            print(f"\n📁 FOLD DETECTION:")
            print(f"   {fold_correct}/{fold_total} Correct ({fold_rate:.1f}%)")
            total_tests += fold_total
            total_correct += fold_correct
        
        # Checks
        check_total = self.results['checks']['total']
        check_correct = self.results['checks']['correct']
        if check_total > 0:
            check_rate = (check_correct / check_total) * 100
            print(f"\n✋ CHECK DETECTION:")
            print(f"   {check_correct}/{check_total} Correct ({check_rate:.1f}%)")
            total_tests += check_total
            total_correct += check_correct
        
        # Chips
        chip_total = self.results['chips']['total']
        chip_correct = self.results['chips']['correct']
        if chip_total > 0:
            chip_rate = (chip_correct / chip_total) * 100
            print(f"\n🪙 CHIP/BET DETECTION:")
            print(f"   {chip_correct}/{chip_total} Correct ({chip_rate:.1f}%)")
            total_tests += chip_total
            total_correct += chip_correct
        
        # Chip amounts
        amount_total = self.results['chip_amounts']['total']
        amount_correct = self.results['chip_amounts']['correct']
        if amount_total > 0:
            amount_rate = (amount_correct / amount_total) * 100
            print(f"\n💰 CHIP AMOUNT COUNTING:")
            print(f"   {amount_correct}/{amount_total} Correct ({amount_rate:.1f}%)")
            total_tests += amount_total
            total_correct += amount_correct
        
        # Overall
        if total_tests > 0:
            overall_rate = (total_correct / total_tests) * 100
            print(f"\n" + "="*60)
            print(f"OVERALL ACCURACY:")
            print(f"   {total_correct}/{total_tests} Correct ({overall_rate:.1f}%)")
            print("="*60)
        
        # Failed tests details
        print("\n" + "="*60)
        print("FAILED TESTS DETAILS:")
        print("="*60)
        
        has_failures = False
        
        for category, data in self.results.items():
            failures = [d for d in data['details'] if not d['correct']]
            if failures:
                has_failures = True
                print(f"\n{category.upper()}:")
                for failure in failures:
                    print(f"   ❌ {failure['image']}")
                    if 'expected' in failure:
                        print(f"      Expected: {failure['expected']}, Got: {failure['detected']}")
                    if 'expected_count' in failure:
                        print(f"      Expected: {failure['expected_count']} chips (${failure['expected_amount']})")
                        print(f"      Detected: {failure['detected_count']} chips (${failure['detected_amount']})")
        
        if not has_failures:
            print("   ✅ No failures - All tests passed!")
    
    def run_all_tests(self):
        """Run all detection tests"""
        print("\n" + "="*60)
        print("CV DETECTION TESTING")
        print("="*60)
        print(f"Test directory: {self.test_dir}")
        
        if not os.path.exists(self.test_dir):
            print(f"\n❌ ERROR: Test directory not found: {self.test_dir}")
            return
        
        # Run tests
        self.test_fold_detection()
        self.test_check_detection()
        self.test_chip_detection()
        self.test_chip_amounts()
        
        # Print summary
        self.print_summary()


if __name__ == "__main__":
    try:
        tester = CVDetectionTester()
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\nTesting interrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()