"""
ML JSON Input Generator
Formats game state into JSON for ML model consumption
"""
from Orchestrator.config import Player, GameState
from Orchestrator.card_converter import CardConverter
import json

class MLJSONGenerator:
    def __init__(self):
        self.hand_id = 0  # Game counter (increments each new game)
        self.last_action = ""  # Track last action taken THIS ROUND
        self.first_to_act = True  # Track if anyone has acted yet THIS ROUND
        self.action_history = []  # Full action history for entire hand: [{player, street, action, amount}, ...]
        self.current_street = "preflop"  # Track current betting round
    
    def increment_hand(self):
        """Increment hand counter for new game"""
        self.hand_id += 1
        self.reset_hand()
    
    def reset_hand(self):
        """Reset all tracking for new hand"""
        self.action_history = []
        self.current_street = "preflop"
        self.reset_round()
    
    def reset_round(self):
        """Reset round-specific tracking (called at start of each betting round)"""
        self.last_action = ""
        self.first_to_act = True
    
    def record_action(self, player_name, action, amount=0):
        """
        Record an action in the full hand history
        
        Args:
            player_name: "PlayerCoach" or "PlayerOne" 
            action: "fold", "check", "call", "raise"
            amount: chips involved (for raise, this is the total raise-to amount)
        """
        # Simplify action text - remove player name, just keep action
        action_lower = action.lower()
        if "fold" in action_lower:
            action_type = "fold"
        elif "check" in action_lower:
            action_type = "check"
        elif "call" in action_lower:
            action_type = "call"
        elif "raise" in action_lower or "bet" in action_lower:
            action_type = "raise"
        else:
            action_type = action
        
        # Add to history
        self.action_history.append({
            'player': player_name,
            'street': self.current_street,
            'action': action_type,
            'amount': amount
        })
        
        # Update last action for this round
        self.last_action = action_type
        self.first_to_act = False
    
    def set_street(self, game_state):
        """Update current street and reset round tracking"""
        new_street = self.get_round_name(game_state)
        if new_street != self.current_street:
            self.current_street = new_street
            self.reset_round()
    
    def get_round_name(self, game_state):
        """Convert GameState to round name"""
        if game_state == GameState.PRE_FLOP_BETTING:
            return "preflop"
        elif game_state in [GameState.WAIT_FOR_FLOP, GameState.POST_FLOP_BETTING]:
            return "flop"
        elif game_state in [GameState.WAIT_FOR_TURN_CARD, GameState.TURN_BETTING]:
            return "turn"
        elif game_state in [GameState.WAIT_FOR_RIVER_CARD, GameState.RIVER_BETTING]:
            return "river"
        elif game_state == GameState.SHOWDOWN:
            return "showdown"
        else:
            return ""
    
    def generate_json_for_coach_action(self, game_state, cards, players, community_pot, call_value):
        """
        Generate JSON payload for ML model when coach needs to make a decision

        Args:
            game_state: Current GameState enum
            cards: CardManager instance
            players: PlayerManager instance
            community_pot: Current pot size
            call_value: Amount to call/raise

        Returns:
            JSON string ready to send to ML model (cards in SUIT|VALUE format)
        """
        # Update street tracking
        self.set_street(game_state)
        
        # Get coach hole cards and convert to ML format
        coach_cards = cards.hole_cards.get(Player.PlayerCoach, ["", ""])
        hole1 = CardConverter.convert_to_ml_format(coach_cards[0]) if len(coach_cards) > 0 else ""
        hole2 = CardConverter.convert_to_ml_format(coach_cards[1]) if len(coach_cards) > 1 else ""

        # Get community cards and convert to ML format
        community = cards.community_cards
        flop1 = CardConverter.convert_to_ml_format(community[0]) if len(community) > 0 else ""
        flop2 = CardConverter.convert_to_ml_format(community[1]) if len(community) > 1 else ""
        flop3 = CardConverter.convert_to_ml_format(community[2]) if len(community) > 2 else ""
        turn = CardConverter.convert_to_ml_format(community[3]) if len(community) > 3 else ""
        river = CardConverter.convert_to_ml_format(community[4]) if len(community) > 4 else ""

        # Get bankrolls for all players
        coach_bankroll = players.get(Player.PlayerCoach)["bankroll"]
        p1_bankroll = players.get(Player.PlayerOne)["bankroll"]
        p2_bankroll = players.get(Player.PlayerTwo)["bankroll"]
        p3_bankroll = players.get(Player.PlayerThree)["bankroll"]

        # Calculate dealer position (player before small blind)
        small_blind_index = players.small_blind_index()
        dealer_position = (small_blind_index - 1) % 4

        # Count players remaining (not folded)
        players_remaining = len(players.get_active_players())

        # Action is blank if coach is first to act THIS ROUND
        action = "" if self.first_to_act else self.last_action

        # Build JSON payload
        
        # Build JSON payload with action history
        payload = {
            "hand_id": self.hand_id,
            "player_id": Player.PlayerCoach.value,  # Always 0 for coach
            "round": self.get_round_name(game_state),
            "hole1": hole1,
            "hole2": hole2,
            "flop1": flop1,
            "flop2": flop2,
            "flop3": flop3,
            "turn": turn,
            "river": river,
            "stack_bb": coach_bankroll,
            "p1_stack_bb": p1_bankroll,
            "p2_stack_bb": p2_bankroll,
            "p3_stack_bb": p3_bankroll,
            "dealer_position": dealer_position,
            "players_remaining": players_remaining,
            "to_call_bb": call_value,
            "pot_bb": community_pot,
            "action": action,
            "action_history": self.action_history,  # Pass full action history
            "final_pot_bb": ""  # Only filled at showdown
        }

        return json.dumps(payload, indent=2)
    
    def generate_json_for_showdown(self, game_state, cards, players, community_pot, remaining_players):
        """
        Generate JSON payload for showdown (includes all remaining players)

        Args:
            game_state: Current GameState enum (should be SHOWDOWN)
            cards: CardManager instance
            players: PlayerManager instance
            community_pot: Final pot size
            remaining_players: List of Player enums still in hand

        Returns:
            List of JSON strings (one per remaining player)
        """
        payloads = []

        for player in remaining_players:
            # Get player hole cards and convert to ML format
            player_cards = cards.hole_cards.get(player, ["", ""])
            hole1 = CardConverter.convert_to_ml_format(player_cards[0]) if len(player_cards) > 0 else ""
            hole2 = CardConverter.convert_to_ml_format(player_cards[1]) if len(player_cards) > 1 else ""

            # Get community cards and convert to ML format
            community = cards.community_cards
            flop1 = CardConverter.convert_to_ml_format(community[0]) if len(community) > 0 else ""
            flop2 = CardConverter.convert_to_ml_format(community[1]) if len(community) > 1 else ""
            flop3 = CardConverter.convert_to_ml_format(community[2]) if len(community) > 2 else ""
            turn = CardConverter.convert_to_ml_format(community[3]) if len(community) > 3 else ""
            river = CardConverter.convert_to_ml_format(community[4]) if len(community) > 4 else ""

            # Get bankrolls for all players
            player_bankroll = players.get(player)["bankroll"]
            p1_bankroll = players.get(Player.PlayerOne)["bankroll"]
            p2_bankroll = players.get(Player.PlayerTwo)["bankroll"]
            p3_bankroll = players.get(Player.PlayerThree)["bankroll"]

            # Calculate dealer position (player before small blind)
            small_blind_index = players.small_blind_index()
            dealer_position = (small_blind_index - 1) % 4

            # Count players remaining (not folded)
            players_remaining_count = len(players.get_active_players())

            # Build JSON payload
            payload = {
                "hand_id": self.hand_id,
                "player_id": player.value,
                "round": "showdown",
                "hole1": hole1,
                "hole2": hole2,
                "flop1": flop1,
                "flop2": flop2,
                "flop3": flop3,
                "turn": turn,
                "river": river,
                "stack_bb": player_bankroll,
                "p1_stack_bb": p1_bankroll,
                "p2_stack_bb": p2_bankroll,
                "p3_stack_bb": p3_bankroll,
                "dealer_position": dealer_position,
                "players_remaining": players_remaining_count,
                "to_call_bb": 0,  # No more betting at showdown
                "pot_bb": community_pot,
                "action": "showdown",
                "final_pot_bb": community_pot
            }

            payloads.append(json.dumps(payload, indent=2))

        return payloads
    
    def print_json(self, json_string):
        """Pretty print JSON for debugging"""
        data = json.loads(json_string)
        print(json.dumps(data, indent=2))