#!/usr/bin/env python3
"""
Production Poker Game Orchestrator
Integrates with CV and ML modules for real gameplay
"""

from Orchestrator.config import Player, GameState
from Orchestrator.player_manager import PlayerManager
from Orchestrator.card_manager import CardManager
from Orchestrator.betting_cycle import run_betting_cycle
from Orchestrator.ml_json_input import MLJSONGenerator
from Orchestrator.event_signals import wait_for_signal, SignalType, set_crop_mode
import json

class PokerGameOrchestrator:
    """Production game orchestrator with CV and ML integration"""
    
    def __init__(self):
        self.state = GameState.WAIT_FOR_GAME_START
        self.players = PlayerManager()
        self.cards = CardManager()
        self.community_pot = 0
        self.call_value = 0
        self.ml_generator = MLJSONGenerator()
        
        # Override get_action for coach to use ML
        self.original_get_action = self.players.get_action
        self.players.get_action = self.get_action_with_ml_integration
        
    def get_action_with_ml_integration(self, player_enum, crop_region, call_value):
        """Get action from CV (for opponents) or ML (for coach)"""
        
        if player_enum == Player.PlayerCoach:
            # COACH: Use ML model for decision
            # Generate JSON payload
            json_payload = self.ml_generator.generate_json_for_coach_action(
                game_state=self.state,
                cards=self.cards,
                players=self.players,
                community_pot=self.community_pot,
                call_value=call_value
            )
            
            print(f"\n{'='*60}")
            print("ML MODEL INPUT (Coach's Turn)")
            print(f"{'='*60}")
            print(json_payload)
            print(f"{'='*60}\n")
            
            # Send JSON to ML model and get decision
            try:
                # TODO: Replace this import with your actual ML module
                # from ML_Model.predictor import get_poker_action
                # data = json.loads(json_payload)
                # action, value = get_poker_action(data)
                # print(f"ML Decision: {action} {value if value > 0 else ''}")
                # return action, value
                
                # PLACEHOLDER: Manual input until ML is connected
                print("ML model not connected - using manual input")
                from Orchestrator.input_interface import InputInterface
                interface = InputInterface()
                return interface.get_action(
                    player_enum.name,
                    call_value,
                    self.players.get(player_enum)["bankroll"]
                )
                
            except Exception as e:
                print(f"ML model error: {e}")
                print("Falling back to manual input...")
                from Orchestrator.input_interface import InputInterface
                interface = InputInterface()
                return interface.get_action(
                    player_enum.name,
                    call_value,
                    self.players.get(player_enum)["bankroll"]
                )
        
        else:
            # OPPONENT: Use CV for detection
            self.players.set_crop_for_player(player_enum)
            
            print(f"\n🎥 Waiting for {player_enum.name} action in {crop_region} region...")
            
            try:
                # TODO: Replace with actual CV action detection
                # from Image_Recognition.action_detector import detect_action
                # action, value = detect_action(crop_mode=crop_region, timeout=30)
                # print(f"CV Detected: {action} {value if value > 0 else ''}")
                # return action, value
                
                # PLACEHOLDER: Manual input until CV is connected
                print("CV not connected - using manual input")
                from Orchestrator.input_interface import InputInterface
                interface = InputInterface()
                return interface.get_action(
                    player_enum.name,
                    call_value,
                    self.players.get(player_enum)["bankroll"]
                )
                
            except Exception as e:
                print(f"CV error: {e}")
                print("Falling back to manual input...")
                from Orchestrator.input_interface import InputInterface
                interface = InputInterface()
                return interface.get_action(
                    player_enum.name,
                    call_value,
                    self.players.get(player_enum)["bankroll"]
                )
    
    def check_early_winner(self):
        """Check if only one player remains and award pot immediately"""
        remaining = self.players.get_active_players()
        if len(remaining) == 1:
            winner = remaining[0]
            self.players.award_pot(winner, self.community_pot)
            print(f"\n{'='*60}")
            print(f"🎉 {winner.name} wins ${self.community_pot} (all others folded)")
            print(f"{'='*60}")
            self.print_bankrolls()
            return True
        return False
    
    def print_bankrolls(self):
        """Print current bankrolls"""
        print(f"\n{'='*60}")
        print("CURRENT BANKROLLS")
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
        print("POKER GAME ORCHESTRATOR - PRODUCTION MODE")
        print("="*60)
        print("Integration Points:")
        print("   • CV Card Detection: Image_Recognition/card_detector.py")
        print("   • CV Action Detection: Image_Recognition/action_detector.py")
        print("   • ML Decision Making: ML_Model/predictor.py")
        print("="*60 + "\n")
        
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
        """Wait for game start signal"""
        print(f"\n{'='*60}")
        print(f"WAITING FOR NEW GAME (Hand #{self.ml_generator.hand_id + 1})")
        print(f"{'='*60}")
        
        try:
            # TODO: Replace with CV game start detection
            # reset_type = wait_for_signal(SignalType.GAME_START)
            # set_crop_mode(NoCrop=True)
            
            # PLACEHOLDER: Manual trigger
            input("Press Enter to start new game...")
            
        except Exception as e:
            print(f"Error detecting game start: {e}")
            input("Press Enter to continue anyway...")
        
        # Initialize/reset all values
        self.players.initialize_bankrolls()
        self.community_pot = 0
        self.call_value = 0
        
        # Increment hand_id
        self.ml_generator.increment_hand()
        
        print(f"\nGame initialized. All players start with $175. (Hand #{self.ml_generator.hand_id})")
        self.state = GameState.WAIT_FOR_HOLE_CARDS
    
    def wait_for_hole_cards(self):
        """Wait for hole cards"""
        print(f"\n{'='*60}")
        print("Waiting for hole cards...")
        print(f"{'='*60}")
        
        try:
            # TODO: Replace with CV card detection
            # set_crop_mode(NoCrop=True)
            # hole_cards = wait_for_signal(SignalType.HOLE_CARDS)
            
            # PLACEHOLDER: Manual input
            from Orchestrator.input_interface import InputInterface
            interface = InputInterface()
            hole_cards = interface.get_cards(2, "hole")
            
        except Exception as e:
            print(f"Error detecting hole cards: {e}")
            from Orchestrator.input_interface import InputInterface
            interface = InputInterface()
            hole_cards = interface.get_cards(2, "hole")
        
        self.cards.set_hole_cards(Player.PlayerCoach, hole_cards)
        print(f"Your hole cards: {hole_cards}")
        
        self.state = GameState.PRE_FLOP_BETTING
    
    def pre_flop_betting(self):
        """Pre-flop betting cycle"""
        print(f"\n{'='*60}")
        print("PRE-FLOP BETTING")
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
        print(f"\n{'='*60}")
        print("Waiting for flop...")
        print(f"{'='*60}")
        
        try:
            # TODO: Replace with CV card detection
            # set_crop_mode(CropCards=True)
            # flop_cards = wait_for_signal(SignalType.COMMUNITY_CARDS, count=3)
            
            # PLACEHOLDER: Manual input
            from Orchestrator.input_interface import InputInterface
            interface = InputInterface()
            flop_cards = interface.get_cards(3, "flop")
            
        except Exception as e:
            print(f"Error detecting flop: {e}")
            from Orchestrator.input_interface import InputInterface
            interface = InputInterface()
            flop_cards = interface.get_cards(3, "flop")
        
        self.cards.add_community_cards(flop_cards)
        print(f"Flop: {flop_cards}")
        
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
        print(f"\n{'='*60}")
        print("Waiting for turn...")
        print(f"{'='*60}")
        
        try:
            # TODO: Replace with CV card detection
            # set_crop_mode(CropCards=True)
            # turn_card = wait_for_signal(SignalType.COMMUNITY_CARDS, count=1)
            
            # PLACEHOLDER: Manual input
            from Orchestrator.input_interface import InputInterface
            interface = InputInterface()
            turn_card = interface.get_cards(1, "turn")
            
        except Exception as e:
            print(f"Error detecting turn: {e}")
            from Orchestrator.input_interface import InputInterface
            interface = InputInterface()
            turn_card = interface.get_cards(1, "turn")
        
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
        print(f"\n{'='*60}")
        print("Waiting for river...")
        print(f"{'='*60}")
        
        try:
            # TODO: Replace with CV card detection
            # set_crop_mode(CropCards=True)
            # river_card = wait_for_signal(SignalType.COMMUNITY_CARDS, count=1)
            
            # PLACEHOLDER: Manual input
            from Orchestrator.input_interface import InputInterface
            interface = InputInterface()
            river_card = interface.get_cards(1, "river")
            
        except Exception as e:
            print(f"Error detecting river: {e}")
            from Orchestrator.input_interface import InputInterface
            interface = InputInterface()
            river_card = interface.get_cards(1, "river")
        
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
        
        try:
            # TODO: Replace with ML hand evaluation
            # winner = self.players.evaluate_with_ml(
            #     self.cards.community_cards,
            #     self.cards.hole_cards,
            #     remaining
            # )
            
            # PLACEHOLDER: Manual winner selection
            from Orchestrator.input_interface import InputInterface
            interface = InputInterface()
            
            # Read opponent hands first
            showdown_hands = {}
            for player in remaining:
                if player == Player.PlayerCoach:
                    continue
                print(f"\n{player.name}'s hand:")
                card1 = interface.get_card("Card 1")
                card2 = interface.get_card("Card 2")
                showdown_hands[player] = [card1, card2]
            
            all_hands = {**self.cards.hole_cards, **showdown_hands}
            winner = interface.get_winner_selection(remaining, self.cards.community_cards, all_hands)
            
        except Exception as e:
            print(f"Error in showdown: {e}")
            winner = remaining[0]
        
        self.players.award_pot(winner, self.community_pot)
        
        print(f"\n{'='*60}")
        print(f"{winner.name} wins ${self.community_pot}!")
        print(f"{'='*60}")
        
        self.print_bankrolls()
        self.state = GameState.WAIT_FOR_GAME_START


if __name__ == "__main__":
    try:
        game = PokerGameOrchestrator()
        game.run()
    except KeyboardInterrupt:
        print("\n\nGame ended.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()