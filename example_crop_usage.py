#!/usr/bin/env python3
"""
Example: How to use the cropping system with existing CV models

This demonstrates the integration between:
1. test_multi_cv_game.py - Game orchestrator with cropping
2. action_analyzer_enhanced.py - Enhanced action detection
3. chip_pot_analyzer.py - Chip counting

Run this to test cropping + CV analysis on a sample image.
"""

import cv2
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Image_Recognition.action_analyzer_enhanced import EnhancedActionAnalyzer
from Image_Recognition.chip_pot_analyzer import ChipPotAnalyzer

# Crop zones (same as test_multi_cv_game.py)
CROP_ZONES = {
    'player1': (30, 180, 200, 200),
    'player2': (220, 180, 200, 200),
    'player3': (410, 180, 200, 200),
    'pot': (180, 50, 280, 170),
    'coach_cards': (220, 300, 200, 150),
    'community_cards': (170, 120, 300, 180),
}


def crop_region(image_path, zone_name):
    """Crop a specific region from the image"""
    if zone_name not in CROP_ZONES:
        raise ValueError(f"Unknown zone: {zone_name}")

    x, y, w, h = CROP_ZONES[zone_name]
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Crop the region
    cropped = img[y:y+h, x:x+w]

    # Save to temp file
    temp_path = f"temp_{zone_name}.jpg"
    cv2.imwrite(temp_path, cropped)

    print(f"✅ Cropped {zone_name} zone ({w}x{h}) → {temp_path}")
    return temp_path


def example_player_action_detection(full_image_path):
    """
    Example: Detect a player's action

    Process:
    1. Crop to player's region
    2. Analyze action (fold/check/bet)
    3. If bet detected, crop pot and count chips
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Player Action Detection")
    print("="*60)

    # Step 1: Crop to player 1's action zone
    player1_crop = crop_region(full_image_path, 'player1')

    # Step 2: Analyze action
    print("\n🔍 Analyzing Player 1's action...")
    analyzer = EnhancedActionAnalyzer()
    result = analyzer.analyze_action(player1_crop)

    print(f"\n📊 Result:")
    print(f"   Action: {result['action']}")
    print(f"   Details: {result['details']}")

    # Step 3: If bet/raise, analyze pot chips
    if result['action'] == 'BET/RAISE':
        print("\n💰 Bet/Raise detected! Analyzing pot chips...")
        pot_crop = crop_region(full_image_path, 'pot')

        chip_analyzer = ChipPotAnalyzer()
        chip_count = chip_analyzer.detect_and_count_chips(pot_crop)

        bet_amount = chip_count * 5
        print(f"\n📊 Pot Analysis:")
        print(f"   Chips: {chip_count}")
        print(f"   Bet Amount: ${bet_amount}")

    print("="*60)


def example_game_flow_simulation(full_image_path):
    """
    Example: Simulate game flow with cropping

    Shows how different game states use different crop zones
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Game Flow Simulation")
    print("="*60)

    # Game State: WAIT_FOR_HOLE_CARDS
    print("\n🎴 Game State: WAIT_FOR_HOLE_CARDS")
    print("   → Cropping: coach_cards zone")
    coach_crop = crop_region(full_image_path, 'coach_cards')
    print("   → Would detect cards here...")

    # Game State: WAIT_FOR_FLOP
    print("\n🎴 Game State: WAIT_FOR_FLOP")
    print("   → Cropping: community_cards zone")
    community_crop = crop_region(full_image_path, 'community_cards')
    print("   → Would detect 3 flop cards here...")

    # Game State: PRE_FLOP_BETTING (Player 1's turn)
    print("\n🎰 Game State: PRE_FLOP_BETTING (Player 1's turn)")
    print("   → Cropping: player1 zone")
    player1_crop = crop_region(full_image_path, 'player1')
    print("   → Would analyze Player 1's action here...")

    # Game State: PRE_FLOP_BETTING (Player 2's turn)
    print("\n🎰 Game State: PRE_FLOP_BETTING (Player 2's turn)")
    print("   → Cropping: player2 zone")
    player2_crop = crop_region(full_image_path, 'player2')
    print("   → Would analyze Player 2's action here...")

    # Game State: After bet/raise detected
    print("\n💰 Detected BET/RAISE → Wait 5s → Analyze pot")
    print("   → Cropping: pot zone")
    pot_crop = crop_region(full_image_path, 'pot')
    print("   → Would count chips in pot here...")

    print("\n" + "="*60)
    print("✅ Game flow simulation complete!")
    print("="*60)


def main():
    # You can replace this with any 640x480 image from your ESP32
    # For testing, use any image from CVDataset
    default_image = "CVDataset/ChipDataset/ESP32S3_OV5640_100988 2.jpg"

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        if os.path.exists(default_image):
            image_path = default_image
        else:
            print("❌ Error: No image provided and default not found")
            print("\nUsage:")
            print("   python example_crop_usage.py <image_path>")
            print("\nExample:")
            print("   python example_crop_usage.py CVDataset/ChipDataset/sample.jpg")
            return

    print("\n" + "="*60)
    print("CROPPING + CV ANALYSIS EXAMPLES")
    print("="*60)
    print(f"Using image: {image_path}")

    # Run examples
    try:
        example_player_action_detection(image_path)
        example_game_flow_simulation(image_path)

        print("\n✅ All examples completed!")
        print("\n💡 Next steps:")
        print("   1. Test with your own ESP32 images")
        print("   2. Adjust CROP_ZONES if needed")
        print("   3. Use test_crop_zones.py to visualize zones")
        print("   4. Run test_multi_cv_game.py with --live flag")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
