# Analyzes cards for community cards and showdown hands
import cv2
import numpy as np
from pathlib import Path

# Card rank and suit detection patterns
RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
SUITS = ['H', 'D', 'C', 'S']  # Hearts, Diamonds, Clubs, Spades

def preprocess_card_image(image):
    """Preprocess card image for OCR/template matching"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get white card regions
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    return gray, thresh

def detect_cards_in_image(image):
    """Detect individual card regions in the image"""
    gray, thresh = preprocess_card_image(image)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    card_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Minimum card size
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w if w > 0 else 0
            
            # Cards have aspect ratio around 1.4
            if 1.2 < aspect_ratio < 1.6:
                card_regions.append((x, y, w, h))
    
    # Sort cards left to right
    card_regions.sort(key=lambda c: c[0])
    
    return card_regions

def recognize_card(card_image):
    """
    Recognize rank and suit of a single card
    This is a placeholder - real implementation would use:
    - Template matching
    - OCR (Tesseract)
    - ML model (CNN)
    """
    # Placeholder - return random card for now
    # In real implementation:
    # 1. Crop top-left corner of card
    # 2. Use template matching or OCR to identify rank
    # 3. Use color detection or template matching for suit
    
    rank = "A"  # Would be detected
    suit = "H"  # Would be detected
    
    return f"{rank}{suit}"

def analyze_community_cards(image):
    """
    Analyze community cards from CropCards region
    
    Returns:
        list: List of card strings like ["AH", "KD", "7C"]
    """
    print("[CARD_ANALYZER] Analyzing community cards...")
    
    card_regions = detect_cards_in_image(image)
    print(f"  Found {len(card_regions)} card regions")
    
    cards = []
    for i, (x, y, w, h) in enumerate(card_regions):
        card_img = image[y:y+h, x:x+w]
        card = recognize_card(card_img)
        cards.append(card)
        print(f"  Card {i+1}: {card}")
    
    return cards

def analyze_player_hand(image, player_name):
    """
    Analyze a player's hand during showdown
    
    Args:
        image: Image of player's hand region
        player_name: Name of the player
    
    Returns:
        list: List of 2 card strings like ["AH", "KD"]
    """
    print(f"[CARD_ANALYZER] Analyzing {player_name}'s hand...")
    
    card_regions = detect_cards_in_image(image)
    
    if len(card_regions) < 2:
        print(f"  ⚠️  Only found {len(card_regions)} cards")
        return []
    
    # Take first 2 cards
    cards = []
    for i in range(min(2, len(card_regions))):
        x, y, w, h = card_regions[i]
        card_img = image[y:y+h, x:x+w]
        card = recognize_card(card_img)
        cards.append(card)
        print(f"  Card {i+1}: {card}")
    
    return cards

def wait_for_cards(expected_count, crop_mode="CropCards", timeout=30):
    """
    Wait for expected number of cards to be detected
    
    Args:
        expected_count: Number of cards expected (3 for flop, 1 for turn/river)
        crop_mode: Expected crop mode
        timeout: Max seconds to wait
    
    Returns:
        list: Detected cards
    """
    from Image_Recognition.action_detector import get_latest_image
    import time
    
    start_time = time.time()
    last_checked_file = None
    
    print(f"[CARD_ANALYZER] Waiting for {expected_count} card(s)...")
    
    while time.time() - start_time < timeout:
        latest_image_path = get_latest_image()
        
        if latest_image_path and latest_image_path != last_checked_file:
            # Check if this is a community cards image
            if "_CARDS_" not in latest_image_path:
                time.sleep(0.5)
                continue
            
            last_checked_file = latest_image_path
            print(f"[CARD_ANALYZER] Processing: {latest_image_path}")
            
            image = cv2.imread(latest_image_path)
            if image is None:
                time.sleep(0.5)
                continue
            
            cards = analyze_community_cards(image)
            
            if len(cards) >= expected_count:
                print(f"  ✅ Detected {len(cards)} cards")
                return cards[:expected_count]
            else:
                print(f"  ⏳ Found {len(cards)}/{expected_count} cards, waiting...")
        
        time.sleep(0.5)
    
    print("[CARD_ANALYZER] ⚠️  Timeout - returning partial results")
    return []