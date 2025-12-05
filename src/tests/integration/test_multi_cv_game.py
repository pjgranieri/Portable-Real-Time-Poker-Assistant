#!/usr/bin/env python3
"""
Multi-Player Computer Vision Integration Test for poker game orchestrator
Uses ML model for PlayerCoach and CV for opponent inputs
Supports up to 4 players with regional detection
Run with: python test_multi_cv_game.py
"""

import sys
import os
import time
import cv2
import numpy as np
import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Orchestrator.config import Player, GameState
from Orchestrator.player_manager import PlayerManager
from Orchestrator.card_manager import CardManager
from Orchestrator.betting_cycle import run_betting_cycle
from Orchestrator.ml_json_input import MLJSONGenerator
from Orchestrator.input_interface import InputInterface

# Import CV modules
from ultralytics import YOLO
from Image_Recognition.action_analyzer_enhanced import EnhancedActionAnalyzer
from Image_Recognition.chip_pot_analyzer import ChipPotAnalyzer

# Import ML module for coach actions
try:
    from Orchestrator.ml_module import get_action as ml_get_action, reset_game as ml_reset_game
    ML_ENABLED = True
    print("[OK] ML Model loaded successfully!")
except Exception as e:
    print(f"[WARNING] ML Model not available: {e}")
    print("   Install missing libraries: pip install torch torchvision treys")
    ML_ENABLED = False


# Cropping regions for each zone (x, y, width, height)
# Based on 640x480 resolution from ESP32 camera
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


class MultiPlayerCVInterface:
    """Computer Vision interface for multi-player poker with regional detection"""

    def __init__(self, use_placeholder=True):
        # Load card detection model
        model_path = os.path.join('Image_Recognition', 'Models', 'card_processing_model.pt')
        self.card_model = YOLO(model_path)

        # Load enhanced action analyzer
        self.action_analyzer = EnhancedActionAnalyzer()

        # Load chip pot analyzer
        self.chip_pot_analyzer = ChipPotAnalyzer()

        # Confidence thresholds
        self.card_confidence_threshold = 0.40

        # Placeholder mode (for testing without real images)
        self.use_placeholder = use_placeholder

        print(f"[INFO] Multi-Player CV Interface initialized")
        print(f"[INFO] Placeholder mode: {use_placeholder}")
        print(f"[INFO] Crop zones configured:")
        for zone, coords in CROP_ZONES.items():
            print(f"  {zone}: x={coords[0]}, y={coords[1]}, w={coords[2]}, h={coords[3]}")

    def crop_image_region(self, image_path, zone_key):
        """
        Crop image to specific zone for analysis

        Args:
            image_path: Path to full image
            zone_key: Key in CROP_ZONES dict (e.g., 'player1', 'pot')

        Returns:
            Path to cropped image (temporary file)
        """
        if zone_key not in CROP_ZONES:
            raise ValueError(f"Unknown zone: {zone_key}")

        x, y, w, h = CROP_ZONES[zone_key]

        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Crop region
        cropped = img[y:y+h, x:x+w]

        # Save to temp file
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp_crops')
        os.makedirs(temp_dir, exist_ok=True)

        filename = os.path.basename(image_path)
        temp_path = os.path.join(temp_dir, f"{zone_key}_{filename}")
        cv2.imwrite(temp_path, cropped)

        print(f"  [CROP] Created {zone_key} crop: {temp_path}")
        return temp_path

    def detect_cards(self, image_path):
        """Detect cards in image"""
        results = self.card_model(image_path, conf=self.card_confidence_threshold, verbose=False)

        detected_cards_raw = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                card_label = result.names[class_id]
                confidence = float(box.conf[0])
                detected_cards_raw.append(card_label)
                print(f"  [CV] Detected: {card_label} (confidence: {confidence:.2%})")

        # Remove duplicates while preserving order
        unique_cards = []
        for card in detected_cards_raw:
            if card not in unique_cards:
                unique_cards.append(card)

        print(f"  [CV] Returning cards (VALUE+SUIT format): {unique_cards}")
        return unique_cards

    def get_latest_image_from_server(self, timeout=60):
        """
        Get latest image from server (placeholder for now)
        TODO: Implement actual server fetch when ready

        Returns:
            Path to latest image file
        """
        # For now, return a placeholder path
        # In production, this would fetch from:
        # response = requests.get('http://20.246.97.176:3000/api/latest-image', timeout=timeout)
        return "placeholder_full_image.jpg"

    def get_hole_cards(self):
        """Get coach's hole cards from CV with cropping"""
        if self.use_placeholder:
            # Return placeholder cards for testing
            print("[PLACEHOLDER] Using placeholder hole cards: ['AS', 'KH']")
            return ['AS', 'KH']

        print(f"\n{'='*60}")
        print("📸 DETECTING COACH'S HOLE CARDS")
        print(f"   Crop zone: coach_cards {CROP_ZONES['coach_cards']}")
        print(f"{'='*60}")

        # Get latest image from server
        full_image_path = self.get_latest_image_from_server()

        try:
            # Crop to coach's card zone
            cropped_image_path = self.crop_image_region(full_image_path, 'coach_cards')

            # Detect cards in cropped region
            print("  [CV] Analyzing coach's card region...")
            cards = self.detect_cards(cropped_image_path)

            if len(cards) >= 2:
                print(f"  ✅ Detected coach's cards: {cards[:2]}")
                return cards[:2]
            else:
                print(f"  ⚠️  Only found {len(cards)} cards, expected 2")
                print("  [FALLBACK] Using placeholder cards")
                return ['AS', 'KH']

        except Exception as e:
            print(f"  [ERROR] Failed to detect hole cards: {e}")
            print("  [FALLBACK] Using placeholder cards")
            return ['AS', 'KH']

    def get_community_cards(self, count, card_type):
        """Get community cards from CV with cropping"""
        if self.use_placeholder:
            # Return placeholder community cards
            placeholder_cards = {
                'flop': ['QD', '10C', '9S'],
                'turn': ['7H'],
                'river': ['2D']
            }
            cards = placeholder_cards.get(card_type, ['AC'] * count)
            print(f"[PLACEHOLDER] Using placeholder {card_type} cards: {cards}")
            return cards

        print(f"\n{'='*60}")
        print(f"📸 DETECTING {card_type.upper()} CARDS ({count} expected)")
        print(f"   Crop zone: community_cards {CROP_ZONES['community_cards']}")
        print(f"{'='*60}")

        # Get latest image from server
        full_image_path = self.get_latest_image_from_server()

        try:
            # Crop to community card zone
            cropped_image_path = self.crop_image_region(full_image_path, 'community_cards')

            # Detect cards in cropped region
            print(f"  [CV] Analyzing community card region...")
            cards = self.detect_cards(cropped_image_path)

            if len(cards) >= count:
                print(f"  ✅ Detected {card_type} cards: {cards[:count]}")
                return cards[:count]
            else:
                print(f"  ⚠️  Only found {len(cards)} cards, expected {count}")
                print("  [FALLBACK] Using placeholder cards")
                placeholder_cards = {
                    'flop': ['QD', '10C', '9S'],
                    'turn': ['7H'],
                    'river': ['2D']
                }
                return placeholder_cards.get(card_type, ['AC'] * count)

        except Exception as e:
            print(f"  [ERROR] Failed to detect {card_type} cards: {e}")
            print("  [FALLBACK] Using placeholder cards")
            placeholder_cards = {
                'flop': ['QD', '10C', '9S'],
                'turn': ['7H'],
                'river': ['2D']
            }
            return placeholder_cards.get(card_type, ['AC'] * count)

    def get_opponent_action(self, player_enum, call_value, bankroll):
        """
        Get opponent action using CV with regional cropping

        Args:
            player_enum: Player enum (Player.PlayerOne, etc.)
            call_value: Current amount to call
            bankroll: Player's available bankroll

        Returns:
            Tuple of (action, value)
        """
        if self.use_placeholder:
            # Placeholder action for testing
            return self.generate_placeholder_action(player_enum, call_value, bankroll)

        # Real CV detection with regional cropping
        return self.detect_player_action(player_enum, call_value, bankroll)

    def generate_placeholder_action(self, player_enum, call_value, bankroll):
        """Generate a placeholder action for testing"""
        import random

        # Weighted random action (more realistic distribution)
        actions_weights = {
            'fold': 0.25,
            'check': 0.15,
            'call': 0.35,
            'raise': 0.25
        }

        action = random.choices(
            list(actions_weights.keys()),
            weights=list(actions_weights.values())
        )[0]

        if action == 'fold':
            value = 0
        elif action == 'check':
            if call_value > 0:
                action = 'fold'  # Can't check with bet
            value = 0
        elif action == 'call':
            if call_value == 0:
                action = 'check'
                value = 0
            else:
                value = call_value
        else:  # raise
            min_raise = call_value + 5 if call_value > 0 else 5
            max_raise = min(bankroll, call_value + 30)
            value = random.randrange(min_raise, max_raise + 1, 5) if max_raise >= min_raise else min_raise
            if value > bankroll:
                value = bankroll

        print(f"[PLACEHOLDER] {player_enum.name} action: {action} ${value}")
        time.sleep(1)  # Simulate detection delay
        return (action, value)

    def detect_player_action(self, player_enum, call_value, bankroll):
        """
        Detect player action using CV with regional cropping

        Process:
        1. Get full image from server
        2. Crop to player's region
        3. Analyze action (fold/check/bet detection)
        4. If bet/raise detected, wait 5s and analyze pot chips

        Args:
            player_enum: Player enum
            call_value: Current amount to call
            bankroll: Player's available bankroll

        Returns:
            Tuple of (action, value)
        """
        # Map player enum to crop zone
        zone_mapping = {
            Player.PlayerOne: 'player1',
            Player.PlayerTwo: 'player2',
            Player.PlayerThree: 'player3'
        }

        zone_key = zone_mapping.get(player_enum)
        if not zone_key:
            raise ValueError(f"Unknown player: {player_enum}")

        print(f"\n{'='*60}")
        print(f"=� DETECTING {player_enum.name}'s ACTION")
        print(f"   Crop zone: {zone_key} {CROP_ZONES[zone_key]}")
        print(f"   Call value: ${call_value}")
        print(f"   Bankroll: ${bankroll}")
        print(f"{'='*60}")

        # Get latest image from server
        full_image_path = self.get_latest_image_from_server(timeout=60)

        # Crop to player's region
        try:
            cropped_image_path = self.crop_image_region(full_image_path, zone_key)
        except Exception as e:
            print(f"  [ERROR] Failed to crop image: {e}")
            print(f"  [FALLBACK] Using placeholder action")
            return self.generate_placeholder_action(player_enum, call_value, bankroll)

        # STEP 1: Analyze action using Enhanced Action Analyzer
        print(f"  [CV] Analyzing action in {zone_key} region...")
        result = self.action_analyzer.analyze_action(cropped_image_path)
        action_type = result['action']
        details = result['details']

        print(f"  [CV] Enhanced analyzer result: {action_type}")
        print(f"  [CV] Details: {details}")

        # STEP 2: If BET/RAISE detected, wait and analyze pot chips
        if action_type == 'BET/RAISE':
            print(f"  [CV] Bet/Raise detected - waiting 5 seconds before analyzing pot...")
            time.sleep(5)

            # Crop pot region
            try:
                pot_image_path = self.crop_image_region(full_image_path, 'pot')
            except Exception as e:
                print(f"  [ERROR] Failed to crop pot region: {e}")
                # Default to minimum bet
                bet_amount = call_value + 5 if call_value > 0 else 5
                print(f"  [FALLBACK] Using minimum bet: ${bet_amount}")
                return ('raise', min(bet_amount, bankroll))

            # Count chips in pot using Chip Pot Analyzer
            print(f"  [CV] Analyzing pot chips...")
            chip_count = self.chip_pot_analyzer.detect_and_count_chips(pot_image_path)
            bet_amount = chip_count * 5  # Each chip = $5

            print(f"  [CV] Pot analysis: {chip_count} chips = ${bet_amount}")

            # Determine if it's a call or raise
            if bet_amount > call_value:
                if bet_amount > bankroll:
                    bet_amount = bankroll
                print(f"   Action: RAISE to ${bet_amount}")
                return ('raise', bet_amount)
            elif bet_amount == call_value and call_value > 0:
                print(f"   Action: CALL ${call_value}")
                return ('call', call_value)
            else:
                print(f"  �  Detected ${bet_amount} but need ${call_value} to call")
                print(f"  � Treating as RAISE to ${bet_amount}")
                return ('raise', bet_amount)

        # STEP 3: Handle FOLD
        if action_type == 'FOLD':
            print(f"   Action: FOLD")
            return ('fold', 0)

        # STEP 4: Handle CHECK
        if action_type == 'CHECK':
            if call_value == 0:
                print(f"   Action: CHECK")
                return ('check', 0)
            else:
                print(f"  �  Cannot check when facing a bet - interpreting as FOLD")
                return ('fold', 0)

        # Fallback for NO_ACTION or unclear
        print(f"  �  Unclear action, defaulting to CHECK/FOLD")
        if call_value == 0:
            return ('check', 0)
        else:
            return ('fold', 0)


class TestMultiPlayerCVGame:
    """Test version with CV integration for multi-player game"""

    def __init__(self, num_players=4, use_placeholder=True):
        self.state = GameState.WAIT_FOR_GAME_START
        self.players = PlayerManager()
        self.cards = CardManager()
        self.community_pot = 0
        self.call_value = 0
        self.ml_generator = MLJSONGenerator()
        self.input_interface = InputInterface()
        self.cv_interface = MultiPlayerCVInterface(use_placeholder=use_placeholder)
        self.num_players = num_players

        # Override player manager methods with test versions
        self.players.get_action = self.mock_get_action
        self.players.evaluate_with_ml = self.mock_evaluate_with_ml
        self.players.read_showdown_hands = self.mock_read_showdown_hands

    def post_coach_action_to_server(self, action, value):
        """Post coach's action to server for ESP32 to display"""
        try:
            response = requests.post(
                'http://20.246.97.176:3000/api/coach-action',
                headers={
                    'Content-Type': 'application/json',
                    'X-API-Key': 'ewfgjiohewiuhwe8934yt83gigiuewhui83h8ge84849g4h489g'
                },
                json={
                    'action': action,
                    'value': value
                },
                timeout=5
            )
            if response.status_code == 200:
                print(f"   Coach action sent to server: {action} ${value if value > 0 else ''}")
        except Exception as e:
            print(f"  �  Failed to send action to server: {e}")

    def post_winner_to_server(self, winner_name, amount):
        """Post winner to server for ESP32 to display"""
        try:
            response = requests.post(
                'http://20.246.97.176:3000/api/winner',
                headers={
                    'Content-Type': 'application/json',
                    'X-API-Key': 'ewfgjiohewiuhwe8934yt83gigiuewhui83h8ge84849g4h489g'
                },
                json={
                    'winner': winner_name,
                    'amount': amount
                },
                timeout=5
            )
            if response.status_code == 200:
                print(f"   Winner sent to server: {winner_name} wins ${amount}")
        except Exception as e:
            print(f"  �  Failed to send winner to server: {e}")

    def reset_game_on_server(self):
        """Reset game state on server (clears winner flag)"""
        try:
            response = requests.post(
                'http://20.246.97.176:3000/api/reset-game',
                headers={
                    'Content-Type': 'application/json',
                    'X-API-Key': 'ewfgjiohewiuhwe8934yt83gigiuewhui83h8ge84849g4h489g'
                },
                timeout=5
            )
            if response.status_code == 200:
                print(f"   Game state reset on server")
        except Exception as e:
            print(f"  �  Failed to reset game on server: {e}")

    def mock_get_action(self, player_enum, crop_region, call_value, min_raise_total=None):
        """Get player action with server posting for coach"""
        player_data = self.players.get(player_enum)

        # If this is the coach and ML is enabled, use ML model
        if player_enum == Player.PlayerCoach and ML_ENABLED:
            try:
                json_payload = self.ml_generator.generate_json_for_coach_action(
                    game_state=self.state,
                    cards=self.cards,
                    players=self.players,
                    community_pot=self.community_pot,
                    call_value=call_value
                )

                action, value = ml_get_action(json_payload)

                # POST ACTION TO SERVER FOR ESP32
                self.post_coach_action_to_server(action, value)

                return (action, value)
            except Exception as e:
                print(f"[ERROR] ML model failed: {e}")
                print("[INFO] Falling back to manual input for coach")

        if player_enum == Player.PlayerCoach:
            # Manual input for coach
            print(f"\n{'='*60}")
            print(f"{player_enum.name}'s Turn (Manual Input)")
            print(f"  Bankroll: ${player_data['bankroll']}")
            print(f"  Call value: ${call_value}")
            print(f"{'='*60}")

            while True:
                action = input(f"Action (fold/check/call/raise): ").strip().lower()

                if action == 'fold':
                    self.post_coach_action_to_server('fold', 0)
                    return ('fold', 0)
                elif action == 'check':
                    if call_value == 0:
                        self.post_coach_action_to_server('check', 0)
                        return ('check', 0)
                    else:
                        print(f"Cannot check - must call ${call_value}")
                        continue
                elif action == 'call':
                    if call_value == 0:
                        print("Nothing to call - use 'check' instead")
                        continue
                    self.post_coach_action_to_server('call', call_value)
                    return ('call', call_value)
                elif action == 'raise':
                    try:
                        amount = int(input(f"Raise to (total): $"))

                        if call_value == 0:
                            min_raise = 5
                        else:
                            min_raise = call_value + 5

                        if amount < min_raise:
                            print(f"Raise must be at least ${min_raise}")
                            continue
                        if amount > player_data['bankroll']:
                            print(f"Cannot raise to ${amount} - only have ${player_data['bankroll']}")
                            continue

                        self.post_coach_action_to_server('raise', amount)
                        return ('raise', amount)
                    except ValueError:
                        print("Invalid amount")
                        continue
                else:
                    print("Invalid action. Use: fold, check, call, or raise")
                    continue

        # CV for opponents
        return self.cv_interface.get_opponent_action(
            player_enum,
            call_value,
            player_data["bankroll"]
        )

    def mock_read_showdown_hands(self, remaining_players):
        """Use CV to read opponent hands at showdown"""
        player_hands = {}

        print(f"\n{'='*60}")
        print("SHOWDOWN - Detecting player hands")
        print(f"{'='*60}")

        for player in remaining_players:
            if player == Player.PlayerCoach:
                continue  # Already have coach's cards

            print(f"\nDetecting {player.name}'s hand...")

            if self.cv_interface.use_placeholder:
                # Placeholder cards
                import random
                suits = ['H', 'D', 'C', 'S']
                values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
                card1 = random.choice(values) + random.choice(suits)
                card2 = random.choice(values) + random.choice(suits)
                player_hands[player] = [card1, card2]
                print(f"[PLACEHOLDER] {player.name}'s cards: {[card1, card2]}")
            else:
                # Real CV detection - would need server integration
                print(f"[WARNING] Real CV detection requires server integration")
                card1 = self.input_interface.get_card("Card 1")
                card2 = self.input_interface.get_card("Card 2")
                player_hands[player] = [card1, card2]

        return player_hands

    def mock_evaluate_with_ml(self, community_cards, hole_cards, remaining_players):
        """Manual input for winner selection (fallback)"""
        showdown_hands = self.players.read_showdown_hands(remaining_players)
        all_hands = {**hole_cards, **showdown_hands}

        return self.input_interface.get_winner_selection(
            remaining_players,
            community_cards,
            all_hands
        )

    def check_early_winner(self):
        """Check if only one player remains and award pot immediately"""
        remaining = self.players.get_active_players()
        if len(remaining) == 1:
            winner = remaining[0]
            self.players.award_pot(winner, self.community_pot)
            print(f"\n{'='*60}")
            print(f" {winner.name} wins ${self.community_pot} (all others folded)")
            print(f"{'='*60}")

            # POST WINNER TO SERVER
            self.post_winner_to_server(winner.name, self.community_pot)

            self.print_bankrolls()
            return True
        return False

    def print_bankrolls(self):
        """Print current bankrolls"""
        print(f"\n{'='*60}")
        print(" CURRENT BANKROLLS")
        print(f"{'='*60}")

        active_players = [Player.PlayerCoach, Player.PlayerOne, Player.PlayerTwo, Player.PlayerThree][:self.num_players]

        for player in active_players:
            bankroll = self.players.get(player)["bankroll"]
            folded = " (FOLDED)" if self.players.get(player)["folded"] else ""
            print(f"  {player.name}: ${bankroll}{folded}")
        print(f"  Community Pot: ${self.community_pot}")
        print(f"{'='*60}")

    def run(self):
        """Main game loop"""
        print("\n" + "="*60)
        print(" MULTI-PLAYER POKER GAME - CV INTEGRATION TEST")
        print("="*60)
        print(f" Number of players: {self.num_players}")
        if ML_ENABLED:
            print(" Coach: ML Model")
        else:
            print(" Coach: Manual Input (ML disabled)")
        print(" Opponents: Computer Vision Detection")
        if self.cv_interface.use_placeholder:
            print(" Mode: PLACEHOLDER (Testing)")
        else:
            print(" Mode: LIVE CV (Real detection)")
        print("="*60)

        while True:
            if self.state == GameState.WAIT_FOR_GAME_START:
                self.wait_for_game_start()
            elif self.state == GameState.WAIT_FOR_HOLE_CARDS:
                self.wait_for_hole_cards()
            elif self.state == GameState.PRE_FLOP_BETTING:
                self.pre_flop_betting()
            elif self.state == GameState.WAIT_FOR_FLOP:
                self.wait_for_flop()
            elif self.state == GameState.POST_FLOP_BETTING:
                self.post_flop_betting()
            elif self.state == GameState.WAIT_FOR_TURN_CARD:
                self.wait_for_turn_card()
            elif self.state == GameState.TURN_BETTING:
                self.turn_betting()
            elif self.state == GameState.WAIT_FOR_RIVER_CARD:
                self.wait_for_river_card()
            elif self.state == GameState.RIVER_BETTING:
                self.river_betting()
            elif self.state == GameState.SHOWDOWN:
                self.showdown()

    # === STATE METHODS ===
    def wait_for_game_start(self):
        """Wait for game start"""
        self.input_interface.wait_for_game_start(self.ml_generator.hand_id + 1)

        # Reset server state (clears winner needsDisplay flag)
        self.reset_game_on_server()

        # Initialize/reset all values
        self.players.initialize_bankrolls()
        self.cards.reset()
        self.community_pot = 0
        self.call_value = 0

        # Rotate blinds for new hand (except first hand)
        if self.ml_generator.hand_id > 0:
            self.players.rotate_blinds()

        # Increment hand_id and reset ML model
        self.ml_generator.increment_hand()
        if ML_ENABLED:
            try:
                ml_reset_game()
            except:
                pass

        print(f"\n Game initialized. All players start with $175. (Hand #{self.ml_generator.hand_id})")
        print(f" Small Blind: {self.players.small_blind.name}, Big Blind: {self.players.big_blind.name}")
        self.state = GameState.WAIT_FOR_HOLE_CARDS

    def wait_for_hole_cards(self):
        """Wait for hole cards using CV"""
        hole_cards = self.cv_interface.get_hole_cards()
        self.cards.set_hole_cards(Player.PlayerCoach, hole_cards)
        print(f" Your hole cards: {hole_cards}")

        self.state = GameState.PRE_FLOP_BETTING

    def pre_flop_betting(self):
        """Pre-flop betting cycle"""
        print(f"\n{'='*60}")
        print(" PRE-FLOP BETTING")
        print(f"   Small Blind: {self.players.small_blind.name}")
        print(f"   Big Blind: {self.players.big_blind.name}")
        print(f"{'='*60}")

        self.community_pot, self.call_value = run_betting_cycle(
            self.players, self.community_pot, self.call_value,
            game_state=self.state, cards=self.cards, ml_generator=self.ml_generator,
            is_preflop=True
        )

        self.call_value = 0
        self.print_bankrolls()

        if self.check_early_winner():
            self.state = GameState.WAIT_FOR_GAME_START
        else:
            self.state = GameState.WAIT_FOR_FLOP

    def wait_for_flop(self):
        """Wait for flop using CV"""
        flop_cards = self.cv_interface.get_community_cards(3, "flop")
        self.cards.add_community_cards(flop_cards)
        print(f" Flop: {flop_cards}")

        self.state = GameState.POST_FLOP_BETTING

    def post_flop_betting(self):
        """Post-flop betting cycle"""
        print(f"\n{'='*60}")
        print("POST-FLOP BETTING")
        print(f"{'='*60}")

        self.community_pot, self.call_value = run_betting_cycle(
            self.players, self.community_pot, self.call_value,
            game_state=self.state, cards=self.cards, ml_generator=self.ml_generator
        )

        self.call_value = 0
        self.print_bankrolls()

        if self.check_early_winner():
            self.state = GameState.WAIT_FOR_GAME_START
        else:
            self.state = GameState.WAIT_FOR_TURN_CARD

    def wait_for_turn_card(self):
        """Wait for turn card using CV"""
        turn_card = self.cv_interface.get_community_cards(1, "turn")
        self.cards.add_community_cards(turn_card)
        print(f"Turn: {turn_card}")

        self.state = GameState.TURN_BETTING

    def turn_betting(self):
        """Turn betting cycle"""
        print(f"\n{'='*60}")
        print("TURN BETTING")
        print(f"{'='*60}")

        self.community_pot, self.call_value = run_betting_cycle(
            self.players, self.community_pot, self.call_value,
            game_state=self.state, cards=self.cards, ml_generator=self.ml_generator
        )

        self.call_value = 0
        self.print_bankrolls()

        if self.check_early_winner():
            self.state = GameState.WAIT_FOR_GAME_START
        else:
            self.state = GameState.WAIT_FOR_RIVER_CARD

    def wait_for_river_card(self):
        """Wait for river card using CV"""
        river_card = self.cv_interface.get_community_cards(1, "river")
        self.cards.add_community_cards(river_card)
        print(f"River: {river_card}")

        self.state = GameState.RIVER_BETTING

    def river_betting(self):
        """River betting cycle"""
        print(f"\n{'='*60}")
        print("RIVER BETTING (FINAL)")
        print(f"{'='*60}")

        self.community_pot, self.call_value = run_betting_cycle(
            self.players, self.community_pot, self.call_value,
            game_state=self.state, cards=self.cards, ml_generator=self.ml_generator
        )

        self.print_bankrolls()

        remaining_count = self.players.get_remaining_count()

        if remaining_count == 1:
            if self.check_early_winner():
                self.state = GameState.WAIT_FOR_GAME_START
        elif remaining_count >= 2:
            self.state = GameState.SHOWDOWN
        else:
            print("ERROR: No players remaining!")
            self.state = GameState.WAIT_FOR_GAME_START

    def showdown(self):
        """Showdown - determine winner"""
        remaining = self.players.get_active_players()

        print(f"\n{'='*60}")
        print(f"SHOWDOWN - {len(remaining)} players remaining")
        print(f"{'='*60}")

        winner = self.players.evaluate_with_ml(
            self.cards.community_cards,
            self.cards.hole_cards,
            remaining
        )

        self.players.award_pot(winner, self.community_pot)

        print(f"\n{'='*60}")
        print(f"{winner.name} wins ${self.community_pot}!")
        print(f"{'='*60}")

        # POST WINNER TO SERVER
        self.post_winner_to_server(winner.name, self.community_pot)

        self.print_bankrolls()
        self.state = GameState.WAIT_FOR_GAME_START


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Multi-player CV Poker Game Test')
    parser.add_argument('--players', type=int, default=4, choices=[2, 3, 4],
                        help='Number of players (2-4)')
    parser.add_argument('--live', action='store_true',
                        help='Use live CV detection instead of placeholders')

    args = parser.parse_args()

    try:
        game = TestMultiPlayerCVGame(
            num_players=args.players,
            use_placeholder=not args.live
        )
        game.run()
    except KeyboardInterrupt:
        print("\n\nGame ended")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
