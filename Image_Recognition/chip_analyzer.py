# Analyzes chips to determine call or raise amount
import cv2
import numpy as np

def detect_chip_color(image, hsv_range):
    """Detect chips of a specific color"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_range[0], hsv_range[1])
    
    # Find circular contours (chips)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    chip_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Minimum chip size
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.7:  # Reasonably circular
                    chip_count += 1
    
    return chip_count

def analyze_chips(image, crop_mode=None):
    """
    Analyze chips in the image to determine bet amount
    
    Chip color denominations (example):
    - White: $1
    - Red: $5
    - Blue: $10
    - Green: $25
    - Black: $100
    
    Args:
        image: OpenCV image
        crop_mode: Current crop region
    
    Returns:
        tuple: (action, value) or None if no chips detected
    """
    print("[CHIP_ANALYZER] Analyzing chips...")
    
    # Define HSV ranges for different chip colors
    chip_colors = {
        "white": ((0, 0, 200), (180, 30, 255), 1),      # $1
        "red": ((0, 100, 100), (10, 255, 255), 5),      # $5
        "blue": ((100, 100, 100), (130, 255, 255), 10), # $10
        "green": ((40, 100, 100), (80, 255, 255), 25),  # $25
        "black": ((0, 0, 0), (180, 255, 50), 100)       # $100
    }
    
    total_value = 0
    chip_details = {}
    
    for color, (lower, upper, value) in chip_colors.items():
        count = detect_chip_color(image, (lower, upper))
        if count > 0:
            chip_details[color] = count
            total_value += count * value
            print(f"  {color.capitalize()}: {count} chips (${count * value})")
    
    if total_value == 0:
        print("  No chips detected")
        return None
    
    print(f"  Total: ${total_value}")
    
    # Determine if call or raise
    # This would need context from the game state (current call value)
    # For now, return as call with the total value
    return ("call", total_value)

def estimate_chip_value_ml(image):
    """
    Use ML model to estimate chip value (future enhancement)
    Could use YOLO or custom CNN to detect and classify chips
    """
    # Placeholder for ML-based chip detection
    pass