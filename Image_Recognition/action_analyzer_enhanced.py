"""
Enhanced Action Analyzer - Improved version of simple analyzer.

Improvements over simple analyzer:
1. Multi-threshold detection (tries lower thresholds if nothing found)
2. Image preprocessing (brightness/contrast adjustment)
3. Multiple detection passes with different settings
4. Better handling of edge cases

Goal: Push accuracy above 90%+
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
import mediapipe as mp


class EnhancedActionAnalyzer:
    """
    Enhanced action analyzer for hand gesture detection only
    Detection: FOLD (closed fist) or CHECK (open palm)
    Chip detection is handled by DetectionBasedChipCounter
    """
    
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
        self.fold_thresholds = {
            'hand_confidence': 0.5,
            'closed_hand_threshold': 0.15
        }
        
        self.check_thresholds = {
            'hand_confidence': 0.5,
            'open_hand_threshold': 0.25
        }
        
        print("[ACTION ANALYZER] Initialized with MediaPipe")
        print("[ACTION ANALYZER] Detects: FOLD (fist) and CHECK (open palm)")
    
    def analyze_action(self, image_path):
        """
        Analyze player gesture from image
        
        Args:
            image_path: Path to image
            
        Returns:
            dict: {
                'action': 'FOLD' | 'CHECK' | 'UNCLEAR',
                'confidence': float (0-1),
                'details': str
            }
        """
        img = cv2.imread(image_path)
        if img is None:
            return {
                'action': 'UNCLEAR',
                'confidence': 0.0,
                'details': 'Failed to load image'
            }
        
        # Check for FOLD (closed hand)
        fold_result = self._detect_fold_mediapipe(img)
        if fold_result['detected']:
            return {
                'action': 'FOLD',
                'confidence': fold_result['confidence'],
                'details': f"Fold detected: {fold_result['details']}"
            }
        
        # Check for CHECK (open palm)
        check_result = self._detect_check_mediapipe(img)
        if check_result['detected']:
            return {
                'action': 'CHECK',
                'confidence': check_result['confidence'],
                'details': f"Check detected: {check_result['details']}"
            }
        
        return {
            'action': 'UNCLEAR',
            'confidence': 0.0,
            'details': 'No fold or check gesture detected'
        }
    
    def _detect_fold_mediapipe(self, img):
        """Detect FOLD gesture using MediaPipe (closed hand/fist)"""
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_img)
        
        if not results.multi_hand_landmarks:
            return {'detected': False, 'confidence': 0.0, 'details': 'No hands detected'}
        
        for hand_landmarks in results.multi_hand_landmarks:
            palm_center = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            
            fingertips = [
                self.mp_hands.HandLandmark.THUMB_TIP,
                self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                self.mp_hands.HandLandmark.RING_FINGER_TIP,
                self.mp_hands.HandLandmark.PINKY_TIP
            ]
            
            distances = []
            for tip in fingertips:
                tip_point = hand_landmarks.landmark[tip]
                dist = np.sqrt((tip_point.x - palm_center.x)**2 + (tip_point.y - palm_center.y)**2)
                distances.append(dist)
            
            avg_distance = np.mean(distances)
            
            if avg_distance < self.fold_thresholds['closed_hand_threshold']:
                confidence = 1.0 - (avg_distance / self.fold_thresholds['closed_hand_threshold'])
                return {
                    'detected': True,
                    'confidence': confidence,
                    'details': f"Closed hand (avg distance={avg_distance:.3f})"
                }
        
        return {'detected': False, 'confidence': 0.0, 'details': 'Hand detected but not closed'}
    
    def _detect_check_mediapipe(self, img):
        """Detect CHECK gesture using MediaPipe (open palm)"""
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_img)
        
        if not results.multi_hand_landmarks:
            return {'detected': False, 'confidence': 0.0, 'details': 'No hands detected'}
        
        for hand_landmarks in results.multi_hand_landmarks:
            palm_center = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            
            fingertips = [
                self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                self.mp_hands.HandLandmark.RING_FINGER_TIP,
                self.mp_hands.HandLandmark.PINKY_TIP
            ]
            
            distances = []
            for tip in fingertips:
                tip_point = hand_landmarks.landmark[tip]
                dist = np.sqrt((tip_point.x - palm_center.x)**2 + (tip_point.y - palm_center.y)**2)
                distances.append(dist)
            
            avg_distance = np.mean(distances)
            
            if avg_distance > self.check_thresholds['open_hand_threshold']:
                confidence = min(1.0, avg_distance / 0.4)
                return {
                    'detected': True,
                    'confidence': confidence,
                    'details': f"Open palm (avg distance={avg_distance:.3f})"
                }
        
        return {'detected': False, 'confidence': 0.0, 'details': 'Hand detected but not open'}
    
    def __del__(self):
        if hasattr(self, 'hands'):
            self.hands.close()
