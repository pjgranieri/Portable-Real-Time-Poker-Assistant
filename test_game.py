#!/usr/bin/env python3
"""
Test stub for poker game orchestrator
Simulates CV and ML modules with manual terminal inputs
Run with: python test_game.py
"""

from Orchestrator.config import Player, GameState
from Orchestrator.player_manager import PlayerManager
from Orchestrator.card_manager import CardManager
from Orchestrator.betting_cycle import run_betting_cycle

class TestGameOrchestrator:
    """Test version of PokerGameOrchestrator with manual inputs"""
    
    def __init__(self):
        self.state = GameState.WAIT_FOR_GAME_START
        self.players = PlayerManager()
        self.cards = CardManager()
        self.community_pot = 0
        self.call_value = 0
        
        # Override player manager methods with test versions
        self.players.get_action = self.mock_get_action
        self.players.evaluate_with_ml = self.mock_evaluate_with_ml
        self.players.read_showdown_hands = self.mock_read_showdown_hands
        
    def mock_get_action(self, player_enum, crop_region, call_value):
        """Manual input for player actions"""
        print(f"\n{'='*60}")
        print(f"🎮 {player_enum.name}'s turn (Region: {crop_region if crop_region else 'YOU'})")
        print(f"   Current bankroll: ${self.players.get(player_enum)['bankroll']}")
        print(f"   Current call value: ${call_value}")
        
        # Show valid actions based on call_value
        if call_value == 0:
            print(f"   Valid actions: check, raise, fold")
        else:
            print(f"   Valid actions: call (${call_value}), raise, fold")
        print(f"{'='*60}")
        
        while True:
            action_input = input("Enter action: ").strip().lower()
            
            if action_input == "fold":
                return ("fold", 0)
            elif action_input == "check":
                if call_value == 0:
                    return ("check", 0)
                else:
                    print(f"  ❌ Cannot check - must call ${call_value} or fold")
            elif action_input == "call":
                if call_value > 0:
                    return ("call", call_value)
                else:
                    print("  ❌ Nothing to call - use 'check' instead")
            elif action_input == "raise":
                try:
                    amount = int(input("  Enter raise amount: $").strip())
                    if amount > 0:
                        return ("raise", amount)
                    else:
                        print("  ❌ Raise amount must be positive")
                except ValueError:
                    print("  ❌ Invalid amount, try again")
            else:
                print("  ❌ Invalid action. Use: fold, check, call, or raise")
    
    def mock_read_showdown_hands(self, remaining_players):
        """Manual input for player hands at showdown"""
        player_hands = {}
        
        print(f"\n{'='*60}")
        print("🃏 SHOWDOWN - Enter each player's hand")
        print(f"{'='*60}")
        
        for player in remaining_players:
            if player == Player.PlayerCoach:
                continue  # Already have coach's cards
            
            print(f"\n{player.name}'s hand:")
            card1 = input("  Card 1 (e.g., AH): ").strip().upper()
            card2 = input("  Card 2 (e.g., KD): ").strip().upper()
            player_hands[player] = [card1, card2]
        
        return player_hands
    
    def mock_evaluate_with_ml(self, community_cards, hole_cards, remaining_players):
        """Manual input for winner selection"""
        showdown_hands = self.players.read_showdown_hands(remaining_players)
        all_hands = {**hole_cards, **showdown_hands}
        
        print(f"\n{'='*60}")
        print("🏆 HAND EVALUATION")
        print(f"{'='*60}")
        print(f"Community cards: {community_cards}")
        
        for i, player in enumerate(remaining_players, 1):
            cards = all_hands.get(player, [])
            print(f"{i}. {player.name}: {cards}")
        
        while True:
            try:
                winner_num = int(input(f"\nSelect winner (1-{len(remaining_players)}): ").strip())
                if 1 <= winner_num <= len(remaining_players):
                    return remaining_players[winner_num - 1]
                else:
                    print(f"  ❌ Enter a number between 1 and {len(remaining_players)}")
            except ValueError:
                print("  ❌ Invalid input")
    
    def mock_wait_for_cards(self, count, card_type):
        """Manual input for cards"""
        print(f"\n{'='*60}")
        print(f"🃏 Enter {count} {card_type} card(s)")
        print(f"{'='*60}")
        
        cards = []
        for i in range(count):
            while True:
                card = input(f"  Card {i+1} (e.g., 7C): ").strip().upper()
                if len(card) >= 2:
                    cards.append(card)
                    break
                else:
                    print("  ❌ Invalid format (e.g., AH, 10D, KC)")
        
        return cards
    
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
        print("💰 CURRENT BANKROLLS")
        print(f"{'='*60}")
        for player in Player:
            bankroll = self.players.get(player)["bankroll"]
            folded = " (FOLDED)" if self.players.get(player)["folded"] else ""
            print(f"  {player.name}: ${bankroll}{folded}")
        print(f"  Community Pot: ${self.community_pot}")
        print(f"{'='*60}")
    
    def run(self):
        """Main game loop"""
        print("\n" + "="*60)
        print("🎰 POKER GAME ORCHESTRATOR - TEST MODE")
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
        print(f"\n{'='*60}")
        print("🎮 NEW GAME")
        print(f"{'='*60}")
        input("Press Enter to start new game...")
        
        # Initialize/reset all values
        self.players.initialize_bankrolls()
        self.community_pot = 0
        self.call_value = 0
        
        print("\n✅ Game initialized. All players start with $100.")
        self.state = GameState.WAIT_FOR_HOLE_CARDS
    
    def wait_for_hole_cards(self):
        """Wait for hole cards"""
        hole_cards = self.mock_wait_for_cards(2, "hole")
        self.cards.set_hole_cards(Player.PlayerCoach, hole_cards)
        print(f"✅ Your hole cards: {hole_cards}")
        
        self.state = GameState.PRE_FLOP_BETTING
    
    def pre_flop_betting(self):
        """Pre-flop betting cycle"""
        print(f"\n{'='*60}")
        print("💵 PRE-FLOP BETTING")
        print(f"   Small Blind: {self.players.small_blind.name}")
        print(f"   Big Blind: {self.players.big_blind.name}")
        print(f"{'='*60}")
        
        self.community_pot, self.call_value = run_betting_cycle(
            self.players, self.community_pot, self.call_value, is_preflop=True
        )
        
        # Reset call value for next round
        self.call_value = 0
        
        self.print_bankrolls()
        
        if self.check_early_winner():
            self.state = GameState.WAIT_FOR_GAME_START
        else:
            self.state = GameState.WAIT_FOR_FLOP
    
    def wait_for_flop(self):
        """Wait for flop"""
        flop_cards = self.mock_wait_for_cards(3, "flop")
        self.cards.add_community_cards(flop_cards)
        print(f"✅ Flop: {flop_cards}")
        
        self.state = GameState.POST_FLOP_BETTING
    
    def post_flop_betting(self):
        """Post-flop betting cycle"""
        print(f"\n{'='*60}")
        print("💵 POST-FLOP BETTING")
        print(f"{'='*60}")
        
        self.community_pot, self.call_value = run_betting_cycle(
            self.players, self.community_pot, self.call_value
        )
        
        # Reset call value for next round
        self.call_value = 0
        
        self.print_bankrolls()
        
        if self.check_early_winner():
            self.state = GameState.WAIT_FOR_GAME_START
        else:
            self.state = GameState.WAIT_FOR_TURN_CARD
    
    def wait_for_turn_card(self):
        """Wait for turn card"""
        turn_card = self.mock_wait_for_cards(1, "turn")
        self.cards.add_community_cards(turn_card)
        print(f"✅ Turn: {turn_card}")
        
        self.state = GameState.TURN_BETTING
    
    def turn_betting(self):
        """Turn betting cycle"""
        print(f"\n{'='*60}")
        print("💵 TURN BETTING")
        print(f"{'='*60}")
        
        self.community_pot, self.call_value = run_betting_cycle(
            self.players, self.community_pot, self.call_value
        )
        
        # Reset call value for next round
        self.call_value = 0
        
        self.print_bankrolls()
        
        if self.check_early_winner():
            self.state = GameState.WAIT_FOR_GAME_START
        else:
            self.state = GameState.WAIT_FOR_RIVER_CARD
    
    def wait_for_river_card(self):
        """Wait for river card"""
        river_card = self.mock_wait_for_cards(1, "river")
        self.cards.add_community_cards(river_card)
        print(f"✅ River: {river_card}")
        
        self.state = GameState.RIVER_BETTING
    
    def river_betting(self):
        """River betting cycle"""
        print(f"\n{'='*60}")
        print("💵 RIVER BETTING (FINAL)")
        print(f"{'='*60}")
        
        self.community_pot, self.call_value = run_betting_cycle(
            self.players, self.community_pot, self.call_value
        )
        
        self.print_bankrolls()
        
        remaining_count = self.players.get_remaining_count()
        
        if remaining_count == 1:
            if self.check_early_winner():
                self.state = GameState.WAIT_FOR_GAME_START
        elif remaining_count >= 2:
            self.state = GameState.SHOWDOWN
        else:
            print("❌ ERROR: No players remaining!")
            self.state = GameState.WAIT_FOR_GAME_START
    
    def showdown(self):
        """Showdown - determine winner"""
        remaining = self.players.get_active_players()
        
        print(f"\n{'='*60}")
        print(f"🏆 SHOWDOWN - {len(remaining)} players remaining")
        print(f"{'='*60}")
        
        winner = self.players.evaluate_with_ml(
            self.cards.community_cards,
            self.cards.hole_cards,
            remaining
        )
        
        self.players.award_pot(winner, self.community_pot)
        
        print(f"\n{'='*60}")
        print(f"🎉 {winner.name} wins ${self.community_pot}!")
        print(f"{'='*60}")
        
        self.print_bankrolls()
        
        self.state = GameState.WAIT_FOR_GAME_START


if __name__ == "__main__":
    try:
        game = TestGameOrchestrator()
        game.run()
    except KeyboardInterrupt:
        print("\n\n👋 Game ended. Thanks for playing!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
