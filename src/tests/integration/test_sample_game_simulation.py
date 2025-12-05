#!/usr/bin/env python3
"""
Sample Game Simulation using Real Dataset Images

This script simulates a full poker game using the images from SampleGameDataset.
It follows a predefined game flow to test CV detection with real images.

Game Flow:
1. PreFlop: Detect hole cards
2. PreFlop Betting: Everyone calls to 2 chips (SB adds 1, BB checks)
3. Flop: Detect flop cards, everyone checks except P3 who folds
4. Turn: Detect turn card, BB raises blue chip, rest call
5. River: Detect river card, SB folds, BB raises, others fold

Run with: python test_sample_game_simulation.py
"""

import sys
import os
import time
import glob
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Orchestrator.config import Player, GameState
from Image_Recognition.action_analyzer_enhanced import EnhancedActionAnalyzer
from Image_Recognition.chip_pot_analyzer import ChipPotAnalyzer
from ultralytics import YOLO

# Cropping regions (640x480)
CROP_ZONES = {
    'player1': (30, 180, 200, 200),
    'player2': (220, 180, 200, 200),
    'player3': (410, 180, 200, 200),
    'pot': (180, 50, 280, 170),
    'coach_cards': (220, 300, 200, 150),
    'community_cards': (170, 120, 300, 180),
}


class SampleGameSimulator:
    """Simulates a full game using real dataset images"""

    def __init__(self, dataset_path='SampleGameDataset'):
        self.dataset_path = dataset_path
        self.base_path = os.path.join(os.path.dirname(__file__), dataset_path)

        # Load CV models
        print("\n" + "="*70)
        print("LOADING CV MODELS")
        print("="*70)

        model_path = os.path.join('Image_Recognition', 'Models', 'card_processing_model.pt')
        self.card_model = YOLO(model_path)
        print("✅ Card detection model loaded")

        self.action_analyzer = EnhancedActionAnalyzer()
        print("✅ Enhanced action analyzer loaded")

        self.chip_pot_analyzer = ChipPotAnalyzer()
        print("✅ Chip pot analyzer loaded")

        # Game state
        self.pot = 0
        self.players_in = [Player.PlayerCoach, Player.PlayerOne, Player.PlayerTwo, Player.PlayerThree]
        self.folded = []

        print("\n" + "="*70)
        print("SAMPLE GAME SIMULATION - Using Real Dataset Images")
        print("="*70)
        print(f"Dataset path: {self.base_path}")
        print(f"Players: {len(self.players_in)}")
        print("="*70 + "\n")

    def get_random_image(self, subfolder):
        """Get a random image from a dataset subfolder"""
        folder_path = os.path.join(self.base_path, subfolder)
        images = glob.glob(os.path.join(folder_path, "*.jpg"))

        if not images:
            raise FileNotFoundError(f"No images found in {folder_path}")

        selected = random.choice(images)
        print(f"  📁 Using: {os.path.relpath(selected, self.base_path)}")
        return selected

    def crop_and_analyze_action(self, image_path, player_zone):
        """Crop player zone and analyze action"""
        import cv2

        # Read and crop
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        x, y, w, h = CROP_ZONES[player_zone]
        cropped = img[y:y+h, x:x+w]

        # Save temp crop
        temp_path = f"temp_crop_{player_zone}.jpg"
        cv2.imwrite(temp_path, cropped)

        # Analyze action
        result = self.action_analyzer.analyze_action(temp_path)

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return result

    def crop_and_analyze_pot(self, image_path):
        """Crop pot zone and count chips"""
        import cv2

        # Read and crop
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        x, y, w, h = CROP_ZONES['pot']
        cropped = img[y:y+h, x:x+w]

        # Save temp crop
        temp_path = f"temp_crop_pot.jpg"
        cv2.imwrite(temp_path, cropped)

        # Count chips
        chip_count = self.chip_pot_analyzer.detect_and_count_chips(temp_path)

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return chip_count

    def crop_and_detect_cards(self, image_path, zone_key):
        """Crop card zone and detect cards"""
        import cv2

        # Read and crop
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        x, y, w, h = CROP_ZONES[zone_key]
        cropped = img[y:y+h, x:x+w]

        # Save temp crop
        temp_path = f"temp_crop_cards.jpg"
        cv2.imwrite(temp_path, cropped)

        # Detect cards
        results = self.card_model(temp_path, conf=0.40, verbose=False)

        detected_cards = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                card_label = result.names[class_id]
                confidence = float(box.conf[0])
                detected_cards.append((card_label, confidence))

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return detected_cards

    def print_pot_status(self):
        """Print current pot and players"""
        active = [p for p in self.players_in if p not in self.folded]
        print(f"\n💰 POT: ${self.pot} | Active Players: {[p.name for p in active]}")

    def step_1_hole_cards(self):
        """Step 1: Detect coach's hole cards"""
        print("\n" + "="*70)
        print("STEP 1: HOLE CARDS")
        print("="*70)

        image_path = self.get_random_image("Cards/HoleCards")

        print("\n🔍 Analyzing coach's hole cards...")
        cards = self.crop_and_detect_cards(image_path, 'coach_cards')

        print("\n📊 DETECTED CARDS:")
        for card, conf in cards[:2]:
            print(f"   {card} (confidence: {conf:.2%})")

        self.print_pot_status()
        input("\n⏸️  Press Enter to continue to PreFlop Betting...")

    def step_2_preflop_betting(self):
        """Step 2: PreFlop betting - everyone calls to 2 chips"""
        print("\n" + "="*70)
        print("STEP 2: PRE-FLOP BETTING")
        print("="*70)
        print("\nBetting Order: P2 (SB) → P3 (BB) → P1 → P2 → P3")

        # P2 (Small Blind): Posts 1 chip
        print("\n" + "-"*70)
        print("🎲 PLAYER 2 (Small Blind): Post 1 chip")
        print("-"*70)
        image_path = self.get_random_image("PotAmounts/Red/1Chip")
        print("  🔍 Analyzing pot...")
        chip_count = self.crop_and_analyze_pot(image_path)
        print(f"  💰 Detected: {chip_count} chips = ${chip_count * 5}")
        self.pot += 5
        self.print_pot_status()
        time.sleep(1)

        # P3 (Big Blind): Posts 2 chips total
        print("\n" + "-"*70)
        print("🎲 PLAYER 3 (Big Blind): Post 2 chips")
        print("-"*70)
        image_path = self.get_random_image("PotAmounts/Red/2Chips")
        print("  🔍 Analyzing pot...")
        chip_count = self.crop_and_analyze_pot(image_path)
        print(f"  💰 Detected: {chip_count} chips = ${chip_count * 5}")
        self.pot += 5  # Adding 1 more chip to make 2 total
        self.print_pot_status()
        time.sleep(1)

        # P1: Calls 2 chips
        print("\n" + "-"*70)
        print("🎲 PLAYER 1: Call 2 chips")
        print("-"*70)
        image_path = self.get_random_image("BettingCycles/PreFlopBetting/P1")
        print("  🔍 Analyzing P1's action...")
        result = self.crop_and_analyze_action(image_path, 'player1')
        print(f"  📊 Action detected: {result['action']}")

        # Show pot after P1 calls
        pot_image = self.get_random_image("PotAmounts/Red/2Chips")
        chip_count = self.crop_and_analyze_pot(pot_image)
        print(f"  💰 Pot now has: {chip_count} chips = ${chip_count * 5}")
        self.pot += 10
        self.print_pot_status()
        time.sleep(1)

        # P2: Calls (adds 1 more to match 2)
        print("\n" + "-"*70)
        print("🎲 PLAYER 2: Call (add 1 chip to match 2)")
        print("-"*70)
        image_path = self.get_random_image("BettingCycles/PreFlopBetting/P2")
        print("  🔍 Analyzing P2's action...")
        result = self.crop_and_analyze_action(image_path, 'player2')
        print(f"  📊 Action detected: {result['action']}")
        self.pot += 5
        self.print_pot_status()
        time.sleep(1)

        # P3: Checks
        print("\n" + "-"*70)
        print("🎲 PLAYER 3: Check")
        print("-"*70)
        image_path = self.get_random_image("BettingCycles/PreFlopBetting/P3")
        print("  🔍 Analyzing P3's action...")
        result = self.crop_and_analyze_action(image_path, 'player3')
        print(f"  📊 Action detected: {result['action']}")
        self.print_pot_status()

        input("\n⏸️  Press Enter to continue to Flop...")

    def step_3_flop(self):
        """Step 3: Detect flop cards"""
        print("\n" + "="*70)
        print("STEP 3: FLOP")
        print("="*70)

        image_path = self.get_random_image("Cards/FlopCards")

        print("\n🔍 Analyzing flop cards...")
        cards = self.crop_and_detect_cards(image_path, 'community_cards')

        print("\n📊 DETECTED FLOP CARDS:")
        for card, conf in cards[:3]:
            print(f"   {card} (confidence: {conf:.2%})")

        self.print_pot_status()
        input("\n⏸️  Press Enter to continue to Flop Betting...")

    def step_4_flop_betting(self):
        """Step 4: Flop betting - everyone checks except P3 folds"""
        print("\n" + "="*70)
        print("STEP 4: FLOP BETTING")
        print("="*70)
        print("\nBetting Order: P2 → P3 → P1")

        # P2: Check
        print("\n" + "-"*70)
        print("🎲 PLAYER 2: Check")
        print("-"*70)
        image_path = self.get_random_image("BettingCycles/FlopBetting/P2/Check")
        print("  🔍 Analyzing P2's action...")
        result = self.crop_and_analyze_action(image_path, 'player2')
        print(f"  📊 Action detected: {result['action']}")
        time.sleep(1)

        # P3: Fold
        print("\n" + "-"*70)
        print("🎲 PLAYER 3: Fold")
        print("-"*70)
        image_path = self.get_random_image("BettingCycles/FlopBetting/P3/Fold")
        print("  🔍 Analyzing P3's action...")
        result = self.crop_and_analyze_action(image_path, 'player3')
        print(f"  📊 Action detected: {result['action']}")
        self.folded.append(Player.PlayerThree)
        time.sleep(1)

        # P1: Check
        print("\n" + "-"*70)
        print("🎲 PLAYER 1: Check")
        print("-"*70)
        image_path = self.get_random_image("BettingCycles/FlopBetting/P1/Check")
        print("  🔍 Analyzing P1's action...")
        result = self.crop_and_analyze_action(image_path, 'player1')
        print(f"  📊 Action detected: {result['action']}")

        self.print_pot_status()
        input("\n⏸️  Press Enter to continue to Turn...")

    def step_5_turn(self):
        """Step 5: Detect turn card"""
        print("\n" + "="*70)
        print("STEP 5: TURN")
        print("="*70)

        image_path = self.get_random_image("Cards/TurnCard")

        print("\n🔍 Analyzing turn card...")
        cards = self.crop_and_detect_cards(image_path, 'community_cards')

        print("\n📊 DETECTED TURN CARD:")
        if cards:
            card, conf = cards[0]
            print(f"   {card} (confidence: {conf:.2%})")

        self.print_pot_status()
        input("\n⏸️  Press Enter to continue to Turn Betting...")

    def step_6_turn_betting(self):
        """Step 6: Turn betting - P2 raises blue, P1 calls blue"""
        print("\n" + "="*70)
        print("STEP 6: TURN BETTING")
        print("="*70)
        print("\nBetting Order: P2 → P1 (P3 is folded)")

        # P2: Raise (blue chip)
        print("\n" + "-"*70)
        print("🎲 PLAYER 2: Raise (blue chip)")
        print("-"*70)
        image_path = self.get_random_image("BettingCycles/TurnBetting/P2/BlueChips")
        print("  🔍 Analyzing P2's action...")
        result = self.crop_and_analyze_action(image_path, 'player2')
        print(f"  📊 Action detected: {result['action']}")

        # Wait 5s and check pot
        print("  ⏱️  Waiting 5 seconds before analyzing pot...")
        time.sleep(5)

        pot_image = self.get_random_image("PotAmounts/Blue")
        print("  🔍 Analyzing pot...")
        chip_count = self.crop_and_analyze_pot(pot_image)
        print(f"  💰 Detected: {chip_count} chips = ${chip_count * 5}")
        self.pot += 20  # Blue chip value
        self.print_pot_status()
        time.sleep(1)

        # P1: Call (blue chip)
        print("\n" + "-"*70)
        print("🎲 PLAYER 1: Call (blue chip)")
        print("-"*70)
        image_path = self.get_random_image("BettingCycles/TurnBetting/P1/BlueChips")
        print("  🔍 Analyzing P1's action...")
        result = self.crop_and_analyze_action(image_path, 'player1')
        print(f"  📊 Action detected: {result['action']}")

        # Wait 5s and check pot
        print("  ⏱️  Waiting 5 seconds before analyzing pot...")
        time.sleep(5)

        pot_image = self.get_random_image("PotAmounts/Blue")
        print("  🔍 Analyzing pot...")
        chip_count = self.crop_and_analyze_pot(pot_image)
        print(f"  💰 Detected: {chip_count} chips = ${chip_count * 5}")
        self.pot += 20
        self.print_pot_status()

        input("\n⏸️  Press Enter to continue to River...")

    def step_7_river(self):
        """Step 7: Detect river card"""
        print("\n" + "="*70)
        print("STEP 7: RIVER")
        print("="*70)

        image_path = self.get_random_image("Cards/RiverCard")

        print("\n🔍 Analyzing river card...")
        cards = self.crop_and_detect_cards(image_path, 'community_cards')

        print("\n📊 DETECTED RIVER CARD:")
        if cards:
            card, conf = cards[0]
            print(f"   {card} (confidence: {conf:.2%})")

        self.print_pot_status()
        input("\n⏸️  Press Enter to continue to River Betting...")

    def step_8_river_betting(self):
        """Step 8: River betting - P2 folds, P1 wins"""
        print("\n" + "="*70)
        print("STEP 8: RIVER BETTING")
        print("="*70)
        print("\nBetting Order: P2 → P1")

        # P2: Fold
        print("\n" + "-"*70)
        print("🎲 PLAYER 2: Fold")
        print("-"*70)
        image_path = self.get_random_image("BettingCycles/RiverBetting/P2/Fold")
        print("  🔍 Analyzing P2's action...")
        result = self.crop_and_analyze_action(image_path, 'player2')
        print(f"  📊 Action detected: {result['action']}")
        self.folded.append(Player.PlayerTwo)
        time.sleep(1)

        # P1: Wins (everyone else folded)
        print("\n" + "="*70)
        print("🏆 PLAYER 1 WINS!")
        print(f"💰 Pot: ${self.pot}")
        print("="*70)

        self.print_pot_status()

    def run(self):
        """Run the full game simulation"""
        try:
            self.step_1_hole_cards()
            self.step_2_preflop_betting()
            self.step_3_flop()
            self.step_4_flop_betting()
            self.step_5_turn()
            self.step_6_turn_betting()
            self.step_7_river()
            self.step_8_river_betting()

            print("\n" + "="*70)
            print("✅ GAME SIMULATION COMPLETE!")
            print("="*70)
            print("\n📊 Summary:")
            print(f"   Final Pot: ${self.pot}")
            print(f"   Winner: Player 1")
            print(f"   Folded: {[p.name for p in self.folded]}")
            print("="*70 + "\n")

        except Exception as e:
            print(f"\n❌ Error during simulation: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SAMPLE GAME SIMULATION")
    print("Testing CV Detection with Real Dataset Images")
    print("="*70)

    simulator = SampleGameSimulator()
    simulator.run()
