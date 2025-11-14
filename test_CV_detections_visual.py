#!/usr/bin/env python3
"""
Visual debugging version of CV detection tests
Saves annotated images showing what was detected
"""

import os
import sys
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Image_Recognition.action_analyzer_algorithm import ActionAnalyzer
from ultralytics import YOLO

class VisualCVTester:
    def __init__(self):
        self.action_analyzer = ActionAnalyzer(confidence_threshold=0.5)
        
        self.test_dir = os.path.join('Outputs', 'Test_Outputs')
        self.output_dir = os.path.join('Outputs', 'Test_Debug_Visuals')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def visualize_detection(self, image_path, image_name, expected_action):
        """Create annotated image showing what was detected - VISUALIZE CROPPED ROI"""
        # Read image
        full_img = cv2.imread(image_path)
        if full_img is None:
            print(f"Failed to read: {image_path}")
            return False
        
        # Crop to ROI (same as action analyzer does)
        img = self.action_analyzer._crop_to_table_region(full_img)
        
        h, w = img.shape[:2]
        
        # Create overlay for annotations
        overlay = img.copy()
        
        # Run action analyzer
        result = self.action_analyzer.analyze_action(image_path)
        detected_action = result['action']
        
        # Check for chip detection
        chips_present, chip_labels = self.action_analyzer.check_chips(image_path)
        chip_count = len(chip_labels)
        
        # Check for fold detection (draw contours if detected)
        fold_detected, fold_conf = self.action_analyzer.check_folded_cards(image_path)
        
        if fold_detected or fold_conf > 0.25:
            # Show fold detection areas in BLUE
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(img_gray, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                relative_area = area / (h * w)
                if 0.003 < relative_area < 0.40:
                    cv2.drawContours(overlay, [contour], -1, (255, 0, 0), 2)
        
        # Check for hand detection (draw skin regions in RED)
        hand_detected, hand_conf = self.action_analyzer.check_hand_present(image_path)
        
        if hand_detected or hand_conf > 0.20:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 15, 50])
            upper_skin = np.array([25, 255, 255])
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Find skin contours
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                relative_area = area / (h * w)
                if 0.01 < relative_area < 0.50:
                    cv2.drawContours(overlay, [contour], -1, (0, 0, 255), 2)
        
        # Blend overlay with original
        alpha = 0.6
        annotated = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        # Add text header
        header_height = 150
        header = np.zeros((header_height, w, 3), dtype=np.uint8)
        
        # Expected vs Detected
        cv2.putText(header, f'Expected: {expected_action}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(header, f'Detected: {detected_action}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                   (0, 255, 0) if detected_action == expected_action else (0, 0, 255), 2)
        
        # Confidence scores
        cv2.putText(header, f'Chips: {chip_count} | Fold: {fold_conf:.1%} | Hand: {hand_conf:.1%}',
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Legend
        cv2.putText(header, f'BLUE=Fold  RED=Hand', (10, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # ROI indicator
        cv2.putText(header, f'[CROPPED TO TABLE ROI - Bottom 60%]', (10, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        
        # Combine header and annotated image
        final = np.vstack([header, annotated])
        
        # Save
        output_path = os.path.join(self.output_dir, f'debug_{image_name}')
        cv2.imwrite(output_path, final)
        print(f"   💾 Saved debug image: {output_path}")
        
        return detected_action == expected_action
    
    def run_visual_tests(self):
        """Run tests with visual output"""
        print("\n" + "="*60)
        print("VISUAL CV DETECTION TESTING")
        print("="*60)
        print(f"Test directory: {self.test_dir}")
        print(f"Output directory: {self.output_dir}")
        
        if not os.path.exists(self.test_dir):
            print(f"\n❌ ERROR: Test directory not found")
            return
        
        test_cases = [
            # Folds
            ('fold.jpg', 'FOLD'),
            ('test_fold.jpg', 'FOLD'),
            ('test_fold_three.jpg', 'FOLD'),
            ('test_fold_four.jpg', 'FOLD'),
            # Checks
            ('test_check_one.jpg', 'CHECK'),
            ('test_check_two.jpg', 'CHECK'),
            ('test_check_three.jpg', 'CHECK'),
            ('test_check_four.jpg', 'CHECK'),
            # Chips
            ('test_chips_one.jpg', 'BET/RAISE'),
            ('test_chips_two.jpg', 'BET/RAISE'),
            ('test_chips_six.jpg', 'BET/RAISE'),
            ('test_chips_seven.jpg', 'BET/RAISE'),
        ]
        
        correct = 0
        total = 0
        
        for img_name, expected in test_cases:
            img_path = os.path.join(self.test_dir, img_name)
            
            if not os.path.exists(img_path):
                print(f"⚠️  SKIP: {img_name}")
                continue
            
            total += 1
            print(f"\n📸 Testing: {img_name} (expect: {expected})")
            
            is_correct = self.visualize_detection(img_path, img_name, expected)
            if is_correct:
                correct += 1
                print(f"   ✅ PASS")
            else:
                print(f"   ❌ FAIL")
        
        print(f"\n{'='*60}")
        print(f"RESULTS: {correct}/{total} correct ({correct/total*100:.1f}%)")
        print(f"Debug images saved to: {self.output_dir}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        tester = VisualCVTester()
        tester.run_visual_tests()
    except KeyboardInterrupt:
        print("\n\nTesting interrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()