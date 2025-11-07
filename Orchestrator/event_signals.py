# Handles calls with CV and ML
from enum import Enum
import requests
import os

class SignalType(Enum):
    GAME_START = 0
    HOLE_CARDS = 1
    COMMUNITY_CARDS = 2
    ACTION = 3

# Server configuration
SERVER_URL = os.getenv('SERVER_URL', 'http://localhost:3000')
API_KEY = os.getenv('API_KEY', 'ewfgjiohewiuhwe8934yt83gigiuewhui83h8ge84849g4h489g')

def set_crop_mode(NoCrop=False, CropLeft=False, CropMiddle=False, CropRight=False, CropCards=False):
    """Set the cropping mode on the server"""
    try:
        response = requests.post(
            f'{SERVER_URL}/api/set-crop-mode',
            headers={'x-api-key': API_KEY},
            json={
                'NoCrop': NoCrop,
                'CropLeft': CropLeft,
                'CropMiddle': CropMiddle,
                'CropRight': CropRight,
                'CropCards': CropCards
            }
        )
        response.raise_for_status()
        print(f"Crop mode set: {response.json()['mode']}")
        return True
    except Exception as e:
        print(f"Failed to set crop mode: {e}")
        return False

def get_crop_mode():
    """Get current cropping mode from server"""
    try:
        response = requests.get(
            f'{SERVER_URL}/api/get-crop-mode',
            headers={'x-api-key': API_KEY}
        )
        response.raise_for_status()
        return response.json()['mode']
    except Exception as e:
        print(f"Failed to get crop mode: {e}")
        return None

def wait_for_signal(signal_type, count=None):
    """
    Wait for event from CV/Server
    Calls the appropriate CV module to analyze images
    """
    if signal_type == SignalType.GAME_START:
        input("Press Enter to start new game...")
        return "soft"
    
    elif signal_type == SignalType.HOLE_CARDS:
        # Wait for CV to detect 2 cards in full frame
        from Image_Recognition.card_analyzer import wait_for_cards
        print("[SIGNAL] Waiting for hole cards...")
        cards = wait_for_cards(expected_count=2, crop_mode="NoCrop")
        return cards if cards else ["AS", "KH"]  # Fallback
    
    elif signal_type == SignalType.COMMUNITY_CARDS:
        # Wait for CV to detect cards in community card crop region
        from Image_Recognition.card_analyzer import wait_for_cards
        print(f"[SIGNAL] Waiting for {count} community card(s)...")
        cards = wait_for_cards(expected_count=count, crop_mode="CropCards")
        
        # Fallback for testing
        if not cards:
            if count == 3:
                return ["7C", "8H", "JD"]
            elif count == 1:
                return ["QC"]
        return cards
    
    elif signal_type == SignalType.ACTION:
        # Should not be called directly - use player_manager.get_action()
        print("[SIGNAL] ACTION signal called directly - use player_manager.get_action()")
        return ("check", 0)