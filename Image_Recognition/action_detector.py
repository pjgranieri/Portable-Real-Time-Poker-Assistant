# detects whether a player has folded, checked, or called/raised
import cv2
import os
import time
import mediapipe as mp
from pathlib import Path
from Image_Recognition.chip_analyzer import analyze_chips

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

OUTPUTS_DIR = Path(__file__).parent / "Outputs"

def detect_blue_rectangles(image):
    """Detect folded cards (blue rectangles) in the image"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for blue color (playing cards back)
    lower_blue = (90, 50, 50)
    upper_blue = (130, 255, 255)
    
    # Create mask for blue regions
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for rectangular shapes (cards)
    card_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area threshold
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:  # Rectangle
                card_contours.append(contour)
    
    # If we found 2 rectangles close together, likely folded cards
    if len(card_contours) >= 2:
        return True, len(card_contours)
    
    return False, 0

def detect_hand_check(image):
    """Detect hand gesture (check) using MediaPipe"""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    
    if results.multi_hand_landmarks:
        # Hand detected - likely a check gesture
        num_hands = len(results.multi_hand_landmarks)
        print(f"  🤚 Hand detected: {num_hands} hand(s) visible")
        return True, num_hands
    
    return False, 0

def get_latest_image(outputs_dir=OUTPUTS_DIR):
    """Get the most recent image from Outputs directory"""
    if not outputs_dir.exists():
        return None
    
    # Get all jpg files
    image_files = list(outputs_dir.glob("*.jpg"))
    
    if not image_files:
        return None
    
    # Sort by modification time, get newest
    latest_image = max(image_files, key=lambda p: p.stat().st_mtime)
    return str(latest_image)

def detect_action(crop_mode=None, timeout=30):
    """
    Detect player action with hierarchical fallback:
    1. Check for folded cards (blue rectangles)
    2. Check for hand gesture (check)
    3. Analyze chips (call/raise)
    
    Args:
        crop_mode: Current crop mode (CropLeft, CropMiddle, CropRight)
        timeout: Max seconds to wait for action detection
    
    Returns:
        tuple: (action, value) where action is 'fold', 'check', 'call', or 'raise'
               value is the bet amount (0 for fold/check)
    """
    start_time = time.time()
    last_checked_file = None
    
    print(f"[ACTION_DETECTOR] Monitoring for action (crop: {crop_mode})...")
    
    while time.time() - start_time < timeout:
        # Get latest image
        latest_image_path = get_latest_image()
        
        if latest_image_path and latest_image_path != last_checked_file:
            print(f"[ACTION_DETECTOR] Processing: {os.path.basename(latest_image_path)}")
            last_checked_file = latest_image_path
            
            # Load image
            image = cv2.imread(latest_image_path)
            if image is None:
                print("  ⚠️  Failed to load image")
                time.sleep(0.5)
                continue
            
            # 1. Check for folded cards (highest priority)
            folded, num_cards = detect_blue_rectangles(image)
            if folded:
                print(f"  ✅ FOLD detected ({num_cards} cards face down)")
                return ("fold", 0)
            
            # 2. Check for hand gesture (check)
            hand_check, num_hands = detect_hand_check(image)
            if hand_check:
                print(f"  ✅ CHECK detected (hand gesture)")
                return ("check", 0)
            
            # 3. Analyze chips (call/raise)
            print("  🔍 No fold/check detected, analyzing chips...")
            chip_result = analyze_chips(image, crop_mode)
            
            if chip_result:
                action, value = chip_result
                print(f"  ✅ {action.upper()} detected: ${value}")
                return (action, value)
            else:
                print("  ⏳ No chips detected yet, waiting...")
        
        time.sleep(0.5)  # Poll every 500ms
    
    print("[ACTION_DETECTOR] ⚠️  Timeout - no action detected")
    return ("check", 0)  # Default to check on timeout

def detect_player_from_crop(latest_image_path):
    """Extract player info from image filename"""
    filename = os.path.basename(latest_image_path)
    
    if "_LEFT_" in filename:
        return "PlayerThree", "CropLeft"
    elif "_MIDDLE_" in filename:
        return "PlayerTwo", "CropMiddle"
    elif "_RIGHT_" in filename:
        return "PlayerOne", "CropRight"
    elif "_CARDS_" in filename:
        return "CommunityCards", "CropCards"
    else:
        return "Unknown", "NoCrop"