#!/usr/bin/env python3
"""
Test script for MediaPipe hand detection (checking action)
Tests the check_hand_present() method from ActionAnalyzer
"""

import os
import sys
import cv2

# Add the parent directory to path so we can import from Image_Recognition
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now try to import - handle both being in root or Image_Recognition folder
try:
    from Image_Recognition.action_analyzer_algorithm import ActionAnalyzer
except ImportError:
    # We're likely in the Image_Recognition folder, try direct import
    from action_analyzer_algorithm import ActionAnalyzer

def main():
    print("="*60)
    print(" MEDIAPIPE HAND DETECTION TEST")
    print("="*60)
    
    # Initialize analyzer
    print("\n[1] Initializing ActionAnalyzer...")
    analyzer = ActionAnalyzer()
    
    # Path to test image - adjust based on current location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if we're in Image_Recognition folder or project root
    if os.path.basename(current_dir) == 'Image_Recognition':
        # Running from Image_Recognition folder
        test_image_path = os.path.join(current_dir, 'Outputs', 'test_checking.jpg')
    else:
        # Running from project root
        test_image_path = os.path.join(current_dir, 'Image_Recognition', 'Outputs', 'test_check_two.jpg')
    
    print(f"\n[2] Testing image: {test_image_path}")
    
    # Check if image exists
    if not os.path.exists(test_image_path):
        print(f"\n[ERROR] Image not found: {test_image_path}")
        print("\nPlease ensure test_checking.jpg exists in Image_Recognition/Outputs/")
        return
    
    # Display image info
    image = cv2.imread(test_image_path)
    if image is not None:
        height, width = image.shape[:2]
        print(f"   Image size: {width}x{height}")
    else:
        print(f"\n[ERROR] Could not read image: {test_image_path}")
        return
    
    print("\n[3] Running hand detection tests...")
    print("-" * 60)
    
    # Test 1: MediaPipe hand detection
    print("\n  Test 1: MediaPipe Hand Detection")
    hand_present = analyzer.check_hand_present(test_image_path)
    print(f"    Result: {'✓ Hand detected' if hand_present else '✗ No hand detected'}")
    
    # Test 2: Simple skin color detection (fallback)
    print("\n  Test 2: Simple Skin Detection (Fallback)")
    hand_simple = analyzer.check_hand_simple(test_image_path)
    print(f"    Result: {'✓ Skin detected' if hand_simple else '✗ No skin detected'}")
    
    # Test 3: Full action analysis
    print("\n  Test 3: Full Action Analysis")
    result = analyzer.analyze_action(test_image_path)
    print(f"    Action: {result['action']}")
    print(f"    Details: {result['details']}")
    
    # Summary
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    
    if result['action'] == 'CHECK':
        print("✓ SUCCESS: Checking action detected!")
        print(f"  - Hand detected: {hand_present}")
        print(f"  - Skin detected: {hand_simple}")
    elif result['action'] == 'BET/RAISE':
        print("⚠ WARNING: Detected BET/RAISE instead of CHECK")
        print(f"  - Chips found: {result['details'].get('chips_detected', [])}")
    elif result['action'] == 'FOLD':
        print("⚠ WARNING: Detected FOLD instead of CHECK")
        print(f"  - Folded cards detected")
    else:
        print("✗ FAILED: No action detected")
        print("  - Make sure image shows a hand/checking gesture")
        print("  - Try adjusting lighting/hand position")
    
    print("="*60)

if __name__ == "__main__":
    main()