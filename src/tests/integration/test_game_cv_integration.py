#!/usr/bin/env python3
"""
Computer Vision Integration Test for poker game orchestrator
Uses ML model for PlayerCoach and CV for opponent inputs
Run with: python test_game_integration.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Orchestrator.config import Player, GameState
from Orchestrator.player_manager import PlayerManager
from Orchestrator.card_manager import CardManager
from Orchestrator.betting_cycle import run_betting_cycle
from Orchestrator.ml_json_input import MLJSONGenerator
from Orchestrator.input_interface import InputInterface
from Orchestrator.card_converter import CardConverter

# Import CV modules
from ultralytics import YOLO
from Image_Recognition.action_analyzer_algorithm import ActionAnalyzer

# Import ML module for coach actions
try:
    from Orchestrator.ml_module import get_action as ml_get_action, reset_game as ml_reset_game
    ML_ENABLED = True
    print("[OK] ML Model loaded successfully!")
except Exception as e:
    print(f"[WARNING] ML Model not available: {e}")
    print("   Install missing libraries: pip install torch torchvision treys")
    ML_ENABLED = False


class CVInputInterface:
    """Computer Vision-based input interface"""
    
    def __init__(self):
        # Load card detection model
        model_path = os.path.join('Image_Recognition', 'Models', 'card_processing_model.pt')
        self.card_model = YOLO(model_path)
        
        # Load chip detection model
        chip_model_path = os.path.join('Image_Recognition', 'Models', 'chip_processing_model.pt')
        self.chip_model = YOLO(chip_model_path)
        
        # Load action analyzer
        self.action_analyzer = ActionAnalyzer()
        
        # Test image paths
        self.two_cards_image = os.path.join('Image_Recognition', 'Outputs', 'test_pocket_aces.jpg')
        self.multiple_cards_image = os.path.join('Image_Recognition', 'Outputs', 'test_multiple_cards.jpg')
        self.action_image = os.path.join('Image_Recognition', 'Outputs', 'test_chips.jpg')
        self.chip_image = os.path.join('Image_Recognition', 'Outputs', 'test_chips.jpg')
    
    def detect_cards(self, image_path):
        """
        Detect cards in image and return as list in VALUE+SUIT format
        YOLO returns: "AC", "KD", "10S" (VALUE+SUIT)
        CardManager expects: "AC", "KD", "10S" (VALUE+SUIT) - same format!
        ml_json_input.py will handle the conversion to SUIT+VALUE
        """
        results = self.card_model(image_path, verbose=False)
        
        detected_cards_raw = []
        
        # Collect all raw detections
        for result in results:
            for box in result.boxes:  
                class_id = int(box.cls[0])
                card_label = result.names[class_id]  # e.g., "AC", "10S", "KH"
                detected_cards_raw.append(card_label)
                print(f"  [CV] Detected: {card_label}")
        
        # Remove duplicates while preserving order
        unique_cards = []
        for card in detected_cards_raw:
            if card not in unique_cards:
                unique_cards.append(card)
        
        # Return cards in VALUE+SUIT format (same as YOLO detected them)
        # DO NOT CONVERT - let ml_json_input.py handle conversion
        print(f"  [CV] Returning cards (VALUE+SUIT format): {unique_cards}")
        return unique_cards
    
    def detect_chip_count(self, image_path):
        """
        Detect number of chips in image
        Returns total bet amount (chips * $5)
        """
        results = self.chip_model(image_path, verbose=False)
        
        chip_count = 0
        for result in results:
            chip_count = len(result.boxes)
        
        print(f"  [CV] Detected {chip_count} chips")
        
        # Each chip is worth $5
        bet_amount = chip_count * 5
        return bet_amount
    
    def get_hole_cards(self):
        """Get hole cards from CV - returns cards in SUIT+VALUE format"""
        print(f"\n{'='*60}")
        print("DETECTING HOLE CARDS...")
        print(f"{'='*60}")
        
        cards = self.detect_cards(self.two_cards_image)
        
        if len(cards) < 2:
            print(f"[WARNING] Only detected {len(cards)} cards, expected 2")
            # Fallback to manual input
            manual = InputInterface()
            return manual.get_cards(2, "hole")
        
        final_cards = cards[:2]
        print(f"Final hole cards (SUIT+VALUE format): {final_cards}")
        return final_cards
    
    def get_community_cards(self, count, card_type):
        """
        Get community cards from CV - returns cards in SUIT+VALUE format
        Returns cards based on what stage we're in:
        - Flop: first 3 cards
        - Turn: 4th card
        - River: 5th card
        """
        print(f"\n{'='*60}")
        print(f"DETECTING {card_type.upper()} CARDS...")
        print(f"{'='*60}")
        
        all_cards = self.detect_cards(self.multiple_cards_image)
        
        if card_type == "flop":
            # Return first 3 cards
            cards = all_cards[:3]
            if len(cards) < 3:
                print(f"[WARNING] Only detected {len(cards)} unique cards for flop, expected 3")
        elif card_type == "turn":
            # Return 4th card
            if len(all_cards) >= 4:
                cards = [all_cards[3]]
            else:
                print(f"[WARNING] Not enough cards detected for turn")
                cards = []
        elif card_type == "river":
            # Return 5th card
            if len(all_cards) >= 5:
                cards = [all_cards[4]]
            else:
                print(f"[WARNING] Not enough cards detected for river")
                cards = []
        else:
            cards = all_cards[:count]
        
        if len(cards) < count:
            print(f"[WARNING] Only detected {len(cards)} cards, expected {count}")
            # Fallback to manual input
            manual = InputInterface()
            return manual.get_cards(count, card_type)
        
        print(f"Final {card_type} cards (SUIT+VALUE format): {cards}")
        return cards
    
    def get_opponent_action(self, player_name, call_value, bankroll, min_raise_total=None):
        """
        Get opponent action using CV action analyzer
        Returns a VALID action that will pass betting_cycle validation
        """
        print(f"\n{'='*60}")
        print(f"DETECTING {player_name}'s ACTION...")
        print(f"{'='*60}")
        
        result = self.action_analyzer.analyze_action(self.action_image)
        
        action_type = result['action']
        details = result['details']
        
        print(f"Detected action: {action_type}")
        print(f"Details: {details}")
        
        # Detect chip amount
        bet_amount = self.detect_chip_count(self.chip_image)
        print(f"  [CV] Detected {len(details.get('chips_detected', []))} chips")
        print(f"  → Detected ${bet_amount} in chips")
        print(f"  → Current call value: ${call_value}")
        
        # SIMPLE LOGIC:
        # If detected amount > call_value → RAISE
        # If detected amount == call_value → CALL
        # If detected amount is 0 and call_value is 0 → CHECK
        # If action is FOLD → FOLD
        
        if action_type == 'FOLD':
            print(f"  → Action: FOLD")
            return ('fold', 0)
        
        if bet_amount == 0:
            # No chips detected
            if call_value == 0:
                print(f"  → Action: CHECK (no chips, no call needed)")
                return ('check', 0)
            else:
                print(f"  → Action: FOLD (no chips detected, can't call ${call_value})")
                return ('fold', 0)
        
        # Chips detected
        if bet_amount > call_value:
            # More chips than needed to call → RAISE
            if bet_amount > bankroll:
                bet_amount = bankroll
            print(f"  → Action: RAISE to ${bet_amount}")
            return ('raise', bet_amount)
        elif bet_amount == call_value and call_value > 0:
            # Exact match → CALL
            print(f"  → Action: CALL ${call_value}")
            return ('call', call_value)
        else:
            # bet_amount < call_value
            print(f"  → Detected ${bet_amount} but need ${call_value} to call")
            print(f"  → Action: FOLD (insufficient chips)")
            return ('fold', 0)


class TestGameIntegration:
    """Test version with CV integration for opponent inputs"""
    
    def __init__(self):
        self.state = GameState.WAIT_FOR_GAME_START
        self.players = PlayerManager()
        self.cards = CardManager()
        self.community_pot = 0
        self.call_value = 0
        self.ml_generator = MLJSONGenerator()
        self.input_interface = InputInterface()
        self.cv_interface = CVInputInterface()
        
        # Override player manager methods with test versions
        self.players.get_action = self.mock_get_action
        self.players.evaluate_with_ml = self.mock_evaluate_with_ml
        self.players.read_showdown_hands = self.mock_read_showdown_hands
        
    def mock_get_action(self, player_enum, crop_region, call_value, min_raise_total=None):
        """
        Get player action:
        - PlayerCoach: Use ML model (if enabled), otherwise manual
        - Others: Use CV action detection
        """
        player_data = self.players.get(player_enum)
        
        # If this is the coach and ML is enabled, use ML model
        if player_enum == Player.PlayerCoach and ML_ENABLED:
            try:
                # Generate JSON for ML model
                json_payload = self.ml_generator.generate_json_for_coach_action(
                    game_state=self.state,
                    cards=self.cards,
                    players=self.players,
                    community_pot=self.community_pot,
                    call_value=call_value
                )
                
                # Get ML prediction
                action, value = ml_get_action(json_payload)
                return (action, value)
            except Exception as e:
                print(f"[ERROR] ML model failed: {e}")
                print("[INFO] Falling back to manual input for coach")
                # Fall through to manual input
        
        if player_enum == Player.PlayerCoach:
            # Use manual input for coach
            print(f"\n{'='*60}")
            print(f"{player_enum.name}'s Turn (Manual Input)")
            print(f"  Bankroll: ${player_data['bankroll']}")
            print(f"  Call value: ${call_value}")
            print(f"{'='*60}")
            
            # Get manual action - ONLY fold, check, call, raise
            while True:
                action = input(f"Action (fold/check/call/raise): ").strip().lower()
                
                if action == 'fold':
                    return ('fold', 0)
                elif action == 'check':
                    if call_value == 0:
                        return ('check', 0)
                    else:
                        print(f"Cannot check - must call ${call_value}")
                        continue
                elif action == 'call':
                    if call_value == 0:
                        print("Nothing to call - use 'check' instead")
                        continue
                    return ('call', call_value)
                elif action == 'raise':
                    try:
                        amount = int(input(f"Raise to (total): $"))
                        
                        # Calculate minimum raise
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
                        return ('raise', amount)
                    except ValueError:
                        print("Invalid amount")
                        continue
                else:
                    print("Invalid action. Use: fold, check, call, or raise")
                    continue
        
        # Otherwise use CV for opponent
        return self.cv_interface.get_opponent_action(
            player_enum.name,
            call_value,
            player_data["bankroll"],
            min_raise_total=min_raise_total
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
            cards = self.cv_interface.detect_cards(self.cv_interface.two_cards_image)
            
            if len(cards) >= 2:
                player_hands[player] = cards[:2]
                print(f"{player.name}'s cards: {cards[:2]}")
            else:
                print(f"[WARNING] Could not detect {player.name}'s cards, using manual input")
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
            self.print_bankrolls()
            return True
        return False
    
    def print_bankrolls(self):
        """Print current bankrolls"""
        print(f"\n{'='*60}")
        print(" CURRENT BANKROLLS")
        print(f"{'='*60}")
        for player in [Player.PlayerCoach, Player.PlayerOne]:
            bankroll = self.players.get(player)["bankroll"]
            folded = " (FOLDED)" if self.players.get(player)["folded"] else ""
            print(f"  {player.name}: ${bankroll}{folded}")
        print(f"  Community Pot: ${self.community_pot}")
        print(f"{'='*60}")
    
    def run(self):
        """Main game loop"""
        print("\n" + "="*60)
        print(" POKER GAME - COMPUTER VISION INTEGRATION TEST")
        print("="*60)
        if ML_ENABLED:
            print(" Coach: ML Model")
        else:
            print(" Coach: Manual Input (ML disabled)")
        print(" Opponent: Computer Vision Detection")
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
        
        # Initialize/reset all values
        self.players.initialize_bankrolls()
        self.community_pot = 0
        self.call_value = 0
        
        # Increment hand_id and reset ML model
        self.ml_generator.increment_hand()
        if ML_ENABLED:
            try:
                ml_reset_game()
            except:
                pass
        
        print(f"\n Game initialized. All players start with $175. (Hand #{self.ml_generator.hand_id})")
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
        
        self.print_bankrolls()
        self.state = GameState.WAIT_FOR_GAME_START


if __name__ == "__main__":
    try:
        game = TestGameIntegration()
        game.run()
    except KeyboardInterrupt:
        print("\n\nGame ended")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()