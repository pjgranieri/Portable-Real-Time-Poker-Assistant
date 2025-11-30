# FSM logic for the game controller
from Orchestrator.betting_cycle import run_betting_cycle
from Orchestrator.player_manager import PlayerManager
from Orchestrator.card_manager import CardManager
from Orchestrator.event_signals import wait_for_signal, SignalType, set_crop_mode
from Orchestrator.config import GameState, Player
from Orchestrator.ml_json_input import MLJSONGenerator

class PokerGameOrchestrator:
    def __init__(self):
        self.state = GameState.WAIT_FOR_GAME_START
        self.players = PlayerManager()
        self.cards = CardManager()
        self.community_pot = 0
        self.call_value = 0
        self.ml_generator = MLJSONGenerator()  # Initialize ML JSON generator

    def run(self):
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

    def check_early_winner(self):
        """Check if only one player remains and award pot immediately"""
        remaining = self.players.get_active_players()
        if len(remaining) == 1:
            winner = remaining[0]
            self.players.award_pot(winner, self.community_pot)
            print(f"[EARLY WIN] {winner.name} wins {self.community_pot} chips (all others folded)")
            
            # Print final bankrolls
            print("\n[STATE] Final bankrolls:")
            for player in Player:
                bankroll = self.players.get(player)["bankroll"]
                print(f"  {player.name}: {bankroll} chips")
            
            # Increment hand counter for next game
            self.ml_generator.increment_hand()
            
            return True
        return False

    # === STATE METHODS ===
    def wait_for_game_start(self):
        """Wait for game start indicator (blank screen, red paper, etc.)"""
        print("[STATE] Waiting for game start...")
        reset_type = wait_for_signal(SignalType.GAME_START)

        # Initialize/reset all values
        self.players.initialize_bankrolls()
        self.community_pot = 0
        self.call_value = 0

        # Rotate blinds for new hand (except first hand)
        if self.ml_generator.hand_id > 0:
            self.players.rotate_blinds()

        # Set server to NoCrop mode
        set_crop_mode(NoCrop=True)

        print(f"[STATE] Game initialized. All players start with 175 chips. Hand #{self.ml_generator.hand_id + 1}")
        print(f"[STATE] Small Blind: {self.players.small_blind.name}, Big Blind: {self.players.big_blind.name}")
        self.state = GameState.WAIT_FOR_HOLE_CARDS

    def wait_for_hole_cards(self):
        """Wait for two hole cards to be shown to camera (full frame)"""
        print("[STATE] Waiting for hole cards... (Full camera view)")
        
        # Ensure NoCrop mode is active
        set_crop_mode(NoCrop=True)
        
        hole_cards = wait_for_signal(SignalType.HOLE_CARDS)
        self.cards.set_hole_cards(Player.PlayerCoach, hole_cards)
        print(f"[STATE] Hole cards received: {hole_cards}")
        
        # Increment hand_id when new game starts (cards are dealt)
        self.ml_generator.increment_hand()
        
        self.state = GameState.PRE_FLOP_BETTING

    def pre_flop_betting(self):
        """Pre-flop betting cycle starting at small blind"""
        print("[STATE] Pre-flop betting...")
        self.community_pot, self.call_value = run_betting_cycle(
            self.players, self.community_pot, self.call_value,
            game_state=self.state, cards=self.cards, ml_generator=self.ml_generator
        )
        
        # Check if only one player remains
        if self.check_early_winner():
            self.state = GameState.WAIT_FOR_GAME_START
        else:
            self.state = GameState.WAIT_FOR_FLOP

    def wait_for_flop(self):
        """Wait for 3 community cards (crop community card region)"""
        print("[STATE] Waiting for flop... (Cropping community cards)")
        
        # Set server to CropCards mode
        set_crop_mode(CropCards=True)
        
        flop_cards = wait_for_signal(SignalType.COMMUNITY_CARDS, count=3)
        self.cards.add_community_cards(flop_cards)
        print(f"[STATE] Flop cards: {flop_cards}")
        
        self.state = GameState.POST_FLOP_BETTING

    def post_flop_betting(self):
        """Post-flop betting cycle"""
        print("[STATE] Post-flop betting...")
        self.community_pot, self.call_value = run_betting_cycle(
            self.players, self.community_pot, self.call_value,
            game_state=self.state, cards=self.cards, ml_generator=self.ml_generator
        )
        
        # Check if only one player remains
        if self.check_early_winner():
            self.state = GameState.WAIT_FOR_GAME_START
        else:
            self.state = GameState.WAIT_FOR_TURN_CARD

    def wait_for_turn_card(self):
        """Wait for 4th community card"""
        print("[STATE] Waiting for turn card... (Cropping community cards)")
        
        # Set server to CropCards mode
        set_crop_mode(CropCards=True)
        
        turn_card = wait_for_signal(SignalType.COMMUNITY_CARDS, count=1)
        self.cards.add_community_cards(turn_card)
        print(f"[STATE] Turn card: {turn_card}")
        
        self.state = GameState.TURN_BETTING

    def turn_betting(self):
        """Turn betting cycle"""
        print("[STATE] Turn betting...")
        self.community_pot, self.call_value = run_betting_cycle(
            self.players, self.community_pot, self.call_value,
            game_state=self.state, cards=self.cards, ml_generator=self.ml_generator
        )
        
        # Check if only one player remains
        if self.check_early_winner():
            self.state = GameState.WAIT_FOR_GAME_START
        else:
            self.state = GameState.WAIT_FOR_RIVER_CARD

    def wait_for_river_card(self):
        """Wait for 5th community card"""
        print("[STATE] Waiting for river card... (Cropping community cards)")
        
        # Set server to CropCards mode
        set_crop_mode(CropCards=True)
        
        river_card = wait_for_signal(SignalType.COMMUNITY_CARDS, count=1)
        self.cards.add_community_cards(river_card)
        print(f"[STATE] River card: {river_card}")
        
        self.state = GameState.RIVER_BETTING

    def river_betting(self):
        """River betting cycle (final betting round)"""
        print("[STATE] River betting...")
        self.community_pot, self.call_value = run_betting_cycle(
            self.players, self.community_pot, self.call_value,
            game_state=self.state, cards=self.cards, ml_generator=self.ml_generator
        )
        
        # After river betting, check if only one player remains
        remaining_count = self.players.get_remaining_count()
        
        if remaining_count == 1:
            # Only one player left - they win automatically
            if self.check_early_winner():
                self.state = GameState.WAIT_FOR_GAME_START
        elif remaining_count >= 2:
            # 2 or more players remain - showdown needed
            self.state = GameState.SHOWDOWN
        else:
            # This shouldn't happen, but handle it
            print("[ERROR] No players remaining!")
            self.state = GameState.WAIT_FOR_GAME_START

    def showdown(self):
        """Determine winner via hand evaluation (only called when 2+ players remain after river)"""
        print("[STATE] Showdown - evaluating hands...")
        remaining = self.players.get_active_players()
        
        print(f"[SHOWDOWN] {len(remaining)} players remaining:")
        for player in remaining:
            print(f"  - {player.name}")
        
        # Read each remaining player's hand for ML evaluation
        print("[STATE] Reading player hands for evaluation...")
        winner = self.players.evaluate_with_ml(
            self.cards.community_cards, 
            self.cards.hole_cards,
            remaining
        )
        
        self.players.award_pot(winner, self.community_pot)
        print(f"[SHOWDOWN] {winner.name} wins {self.community_pot} chips with best hand!")
        
        # Print final bankrolls
        print("\n[STATE] Final bankrolls:")
        for player in Player:
            bankroll = self.players.get(player)["bankroll"]
            print(f"  {player.name}: {bankroll} chips")
        
        # Increment hand counter for next game
        self.ml_generator.increment_hand()
        
        self.state = GameState.WAIT_FOR_GAME_START
