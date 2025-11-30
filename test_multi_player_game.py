#!/usr/bin/env python3
"""
Test stub for poker game orchestrator
Uses ML model for PlayerCoach and manual inputs for opponent
Run with: python test_game.py
"""

from Orchestrator.config import Player, GameState
from Orchestrator.player_manager import PlayerManager
from Orchestrator.card_manager import CardManager
from Orchestrator.betting_cycle import run_betting_cycle
from Orchestrator.ml_json_input import MLJSONGenerator
from Orchestrator.input_interface import InputInterface

# Import ML module for coach actions
try:
    from Orchestrator.ml_module import get_action as ml_get_action, reset_game as ml_reset_game
    ML_ENABLED = True
    print("[OK] ML Model loaded successfully!")
except Exception as e:
    print(f"[WARNING] ML Model not available: {e}")
    print("   Falling back to manual input for all players")
    ML_ENABLED = False

class TestGameOrchestrator:
    """Test version of PokerGameOrchestrator with ML for coach, manual for opponent"""
    
    def __init__(self):
        self.state = GameState.WAIT_FOR_GAME_START
        self.players = PlayerManager()
        self.cards = CardManager()
        self.community_pot = 0
        self.call_value = 0
        self.ml_generator = MLJSONGenerator()
        self.input_interface = InputInterface()
        
        # Override player manager methods with test versions
        self.players.get_action = self.mock_get_action
        self.players.evaluate_with_ml = self.mock_evaluate_with_ml
        self.players.read_showdown_hands = self.mock_read_showdown_hands
        
    def mock_get_action(self, player_enum, crop_region, call_value, min_raise_total=None):
        """
        Get player action:
        - PlayerCoach: Use ML model (if enabled)
        - Others: Manual input
        """
        # If this is the coach and ML is enabled, use ML model
        if player_enum == Player.PlayerCoach and ML_ENABLED:
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
        
        # Otherwise use manual input
        player_data = self.players.get(player_enum)
        return self.input_interface.get_action(
            player_enum.name,
            call_value,
            player_data["bankroll"],
            min_raise_total=min_raise_total
        )
    
    def mock_read_showdown_hands(self, remaining_players):
        """Manual input for player hands at showdown"""
        player_hands = {}
        
        print(f"\n{'='*60}")
        print("SHOWDOWN - Enter each player's hand")
        print(f"{'='*60}")
        
        for player in remaining_players:
            if player == Player.PlayerCoach:
                continue  # Already have coach's cards
            
            print(f"\n{player.name}'s hand:")
            card1 = self.input_interface.get_card("Card 1")
            card2 = self.input_interface.get_card("Card 2")
            player_hands[player] = [card1, card2]
        
        return player_hands
    
    def mock_evaluate_with_ml(self, community_cards, hole_cards, remaining_players):
        """Manual input for winner selection"""
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
        for player in [Player.PlayerCoach, Player.PlayerOne, Player.PlayerTwo, Player.PlayerThree]:
            bankroll = self.players.get(player)["bankroll"]
            folded = " (FOLDED)" if self.players.get(player)["folded"] else ""
            print(f"  {player.name}: ${bankroll}{folded}")
        print(f"  Community Pot: ${self.community_pot}")
        print(f"{'='*60}")
    
    def run(self):
        """Main game loop"""
        print("\n" + "="*60)
        print(" POKER GAME ORCHESTRATOR - TEST MODE")
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
        self.cards.reset()  # Reset hole cards and community cards
        self.community_pot = 0
        self.call_value = 0

        # Rotate blinds for new hand (except first hand)
        if self.ml_generator.hand_id > 0:
            self.players.rotate_blinds()

        # Increment hand_id and reset ML model
        self.ml_generator.increment_hand()
        if ML_ENABLED:
            ml_reset_game()

        print(f"\n Game initialized. All players start with $175. (Hand #{self.ml_generator.hand_id})")
        print(f" Small Blind: {self.players.small_blind.name}, Big Blind: {self.players.big_blind.name}")
        self.state = GameState.WAIT_FOR_HOLE_CARDS
    
    def wait_for_hole_cards(self):
        """Wait for hole cards"""
        hole_cards = self.input_interface.get_cards(2, "hole")
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
        """Wait for flop"""
        flop_cards = self.input_interface.get_cards(3, "flop")
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
        """Wait for turn card"""
        turn_card = self.input_interface.get_cards(1, "turn")
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
        """Wait for river card"""
        river_card = self.input_interface.get_cards(1, "river")
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
        game = TestGameOrchestrator()
        game.run()
    except KeyboardInterrupt:
        print("\n\nGame ended")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
