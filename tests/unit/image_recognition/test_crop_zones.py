#!/usr/bin/env python3
"""
Utility script to test and visualize crop zones on actual images
This helps verify that the cropping regions are correct before running the game

Usage:
    python test_crop_zones.py <image_path>
    python test_crop_zones.py <image_path> --zone player1
    python test_crop_zones.py <image_path> --all  # Show all zones
    python test_crop_zones.py <image_path> --save # Save cropped regions
"""

import cv2
import numpy as np
import os
import sys
import argparse

# Cropping regions (same as in test_multi_cv_game.py)
CROP_ZONES = {
    # Player action zones (for detecting fold/check/bet)
    'player1': (30, 180, 200, 200),      # Left circle - Player 1 actions
    'player2': (220, 180, 200, 200),     # Middle circle - Player 2 actions
    'player3': (410, 180, 200, 200),     # Right circle - Player 3 actions

    # Pot zone (for counting chips in pot)
    'pot': (180, 50, 280, 170),          # Center pot area

    # Card zones (for card detection)
    'coach_cards': (220, 300, 200, 150), # Coach's hole cards (bottom center)
    'community_cards': (170, 120, 300, 180),  # Flop/Turn/River (center table)

    # Showdown card zones (when opponents reveal cards)
    'player1_cards': (30, 180, 200, 100),    # Player 1 cards
    'player2_cards': (220, 180, 200, 100),   # Player 2 cards
    'player3_cards': (410, 180, 200, 100),   # Player 3 cards
}

# Colors for visualization (BGR format)
ZONE_COLORS = {
    'player1': (0, 255, 0),           # Green
    'player2': (255, 0, 0),           # Blue
    'player3': (0, 0, 255),           # Red
    'pot': (0, 255, 255),             # Yellow
    'coach_cards': (255, 255, 0),     # Cyan
    'community_cards': (255, 0, 255), # Magenta
    'player1_cards': (128, 255, 128), # Light Green
    'player2_cards': (255, 128, 128), # Light Blue
    'player3_cards': (128, 128, 255), # Light Red
}


def draw_zone_on_image(img, zone_name, zone_coords, color, thickness=2):
    """
    Draw a rectangle on the image to show the crop zone

    Args:
        img: Image to draw on
        zone_name: Name of the zone
        zone_coords: (x, y, width, height)
        color: BGR color tuple
        thickness: Line thickness
    """
    x, y, w, h = zone_coords

    # Draw rectangle
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    # Add label
    label = zone_name.upper()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # Get text size for background
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    # Draw background for text
    cv2.rectangle(img, (x, y - text_height - 5), (x + text_width, y), color, -1)

    # Draw text
    cv2.putText(img, label, (x, y - 5), font, font_scale, (0, 0, 0), font_thickness)


def visualize_zones(image_path, zones_to_show=None, save=False):
    """
    Visualize crop zones on an image

    Args:
        image_path: Path to image
        zones_to_show: List of zone names to show (None = all zones)
        save: Whether to save the visualization
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not read image: {image_path}")
        return

    height, width = img.shape[:2]
    print(f"📸 Image loaded: {width}x{height}")

    # Determine which zones to show
    if zones_to_show is None:
        zones_to_show = list(CROP_ZONES.keys())

    # Draw zones
    visualization = img.copy()
    for zone_name in zones_to_show:
        if zone_name not in CROP_ZONES:
            print(f"⚠️  Warning: Unknown zone '{zone_name}', skipping")
            continue

        zone_coords = CROP_ZONES[zone_name]
        color = ZONE_COLORS.get(zone_name, (255, 255, 255))

        draw_zone_on_image(visualization, zone_name, zone_coords, color)
        print(f"✅ Drew zone: {zone_name} at {zone_coords}")

    # Display
    window_name = f"Crop Zones - {os.path.basename(image_path)}"
    cv2.imshow(window_name, visualization)
    print(f"\n👁️  Displaying visualization. Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save if requested
    if save:
        output_path = image_path.replace('.jpg', '_zones.jpg')
        cv2.imwrite(output_path, visualization)
        print(f"💾 Saved visualization to: {output_path}")


def crop_and_save_zone(image_path, zone_name, output_dir='cropped_zones'):
    """
    Crop a specific zone and save it

    Args:
        image_path: Path to image
        zone_name: Name of zone to crop
        output_dir: Directory to save cropped images
    """
    if zone_name not in CROP_ZONES:
        print(f"❌ Error: Unknown zone '{zone_name}'")
        return

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not read image: {image_path}")
        return

    # Get zone coordinates
    x, y, w, h = CROP_ZONES[zone_name]

    # Crop
    cropped = img[y:y+h, x:x+w]

    # Save
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(image_path).replace('.jpg', '')
    output_path = os.path.join(output_dir, f"{base_name}_{zone_name}.jpg")
    cv2.imwrite(output_path, cropped)

    print(f"✅ Cropped {zone_name} zone: {w}x{h}")
    print(f"💾 Saved to: {output_path}")


def crop_all_zones(image_path, output_dir='cropped_zones'):
    """
    Crop all zones and save them

    Args:
        image_path: Path to image
        output_dir: Directory to save cropped images
    """
    print(f"\n{'='*60}")
    print(f"Cropping all zones from: {os.path.basename(image_path)}")
    print(f"{'='*60}\n")

    for zone_name in CROP_ZONES.keys():
        crop_and_save_zone(image_path, zone_name, output_dir)
        print()

    print(f"{'='*60}")
    print(f"All zones saved to: {output_dir}/")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Test and visualize crop zones on images')
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('--zone', help='Show only this zone (e.g., player1, pot)')
    parser.add_argument('--all', action='store_true', help='Show all zones with labels')
    parser.add_argument('--save', action='store_true', help='Save cropped regions to files')
    parser.add_argument('--list', action='store_true', help='List all available zones')

    args = parser.parse_args()

    # List zones
    if args.list:
        print("\n📋 Available crop zones:")
        print(f"{'='*60}")
        for zone_name, coords in CROP_ZONES.items():
            x, y, w, h = coords
            print(f"  {zone_name:20s} → x={x:3d}, y={y:3d}, w={w:3d}, h={h:3d}")
        print(f"{'='*60}\n")
        return

    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image not found: {args.image_path}")
        return

    # Save cropped zones
    if args.save:
        if args.zone:
            crop_and_save_zone(args.image_path, args.zone)
        else:
            crop_all_zones(args.image_path)
    else:
        # Visualize zones
        zones_to_show = [args.zone] if args.zone else None
        visualize_zones(args.image_path, zones_to_show, save=args.save)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\n" + "="*60)
        print("CROP ZONE TESTING UTILITY")
        print("="*60)
        print("\nUsage Examples:")
        print("  python test_crop_zones.py image.jpg")
        print("  python test_crop_zones.py image.jpg --zone player1")
        print("  python test_crop_zones.py image.jpg --all --save")
        print("  python test_crop_zones.py --list")
        print("\nOptions:")
        print("  --zone NAME   Show only one zone (player1, player2, player3, pot, etc.)")
        print("  --all         Show all zones with labels")
        print("  --save        Save cropped regions to files")
        print("  --list        List all available zones")
        print("\n" + "="*60 + "\n")
    else:
        main()
