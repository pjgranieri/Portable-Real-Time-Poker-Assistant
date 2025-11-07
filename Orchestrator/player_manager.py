# Handles Player bankroll, actions, player states
from Orchestrator.config import Player
from Orchestrator.event_signals import set_crop_mode

class PlayerManager:
    def __init__(self):
        self.players = {
            Player.PlayerCoach: {"bankroll": 175, "call": True, "folded": False},
            Player.PlayerOne: {"bankroll": 175, "call": True, "folded": False},
            Player.PlayerTwo: {"bankroll": 175, "call": True, "folded": False},
            Player.PlayerThree: {"bankroll": 175, "call": True, "folded": False}
        }
        self.small_blind = Player.PlayerCoach
        self.big_blind = Player.PlayerOne

    def initialize_bankrolls(self):
        """Reset all player states for new game"""
        for p in self.players:
            self.players[p]["bankroll"] = 175
            self.players[p]["folded"] = False
            self.players[p]["call"] = True

    def get(self, player_enum):
        return self.players[player_enum]

    def small_blind_index(self):
        return self.small_blind.value

    def get_crop_region(self, player_enum):
        """Returns the crop region for a given player"""
        if player_enum == Player.PlayerCoach:
            return None
        elif player_enum == Player.PlayerOne:
            return "CropRight"
        elif player_enum == Player.PlayerTwo:
            return "CropMiddle"
        elif player_enum == Player.PlayerThree:
            return "CropLeft"

    def set_crop_for_player(self, player_enum):
        """Sets the crop region on the server for a specific player"""
        crop_region = self.get_crop_region(player_enum)
        
        if crop_region == "CropRight":
            set_crop_mode(CropRight=True)
        elif crop_region == "CropMiddle":
            set_crop_mode(CropMiddle=True)
        elif crop_region == "CropLeft":
            set_crop_mode(CropLeft=True)
        else:
            print(f"⚠️  No crop region for {player_enum.name}")

    def get_action(self, player_enum, crop_region, call_value, min_raise_total=None):
        """Get action from CV module for a specific player"""
    
        # If this is the coach, generate ML JSON input
        if player_enum == Player.PlayerCoach:
            from Orchestrator.ml_json_input import MLJSONGenerator
            ml_gen = MLJSONGenerator()  # Should be stored in GameController
            
            # Generate JSON with current game state
            json_payload = ml_gen.generate_json_for_coach_action(
                game_state=...,  # Pass from game_controller
                cards=...,  # Pass CardManager instance
                players=self,
                community_pot=...,  # Pass from game_controller
                call_value=call_value
            )
            
            # Send to ML model or print for debugging
            print("[ML INPUT]")
            ml_gen.print_json(json_payload)
            
            # TODO: Send json_payload to ML model API/module
            # ml_action, ml_value = send_to_ml_model(json_payload)
        
        from Image_Recognition.action_detector import detect_action
        
        # Set the crop region on the server
        self.set_crop_for_player(player_enum)
        
        print(f"Waiting for {player_enum.name} action in {crop_region} region...")
        
        # Call CV action detector
        # Note: min_raise_total is provided by the caller for validation/display
        action, value = detect_action(crop_mode=crop_region, timeout=30)

        return action, value

    def get_active_players(self):
        """Returns list of players who haven't folded"""
        return [p for p in self.players if not self.players[p]["folded"]]

    def get_remaining_count(self):
        """Returns count of players who haven't folded"""
        return len(self.get_active_players())

    def award_pot(self, player_enum, pot):
        self.players[player_enum]["bankroll"] += pot

    def read_showdown_hands(self, remaining_players):
        """Read each remaining player's hand for showdown"""
        from Image_Recognition.card_analyzer import analyze_player_hand
        import cv2
        from Image_Recognition.action_detector import get_latest_image
        
        player_hands = {}
        
        for player in remaining_players:
            if player == Player.PlayerCoach:
                continue  # Already have coach's cards
            
            crop_region = self.get_crop_region(player)
            self.set_crop_for_player(player)
            
            print(f"[SHOWDOWN] Reading {player.name}'s hand...")
            
            # Wait for image with player's cards
            import time
            time.sleep(2)  # Give time for image to be captured
            
            latest_image = get_latest_image()
            if latest_image:
                image = cv2.imread(latest_image)
                cards = analyze_player_hand(image, player.name)
                player_hands[player] = cards
        
        return player_hands

    def evaluate_with_ml(self, community_cards, hole_cards, remaining_players):
        """Send to ML module to determine winner among remaining players"""
        
        # Read each player's hand for showdown
        showdown_hands = self.read_showdown_hands(remaining_players)
        
        # Merge with known hole cards
        all_hands = {**hole_cards, **showdown_hands}
        
        print(f"[ML] Evaluating hands for {len(remaining_players)} players...")
        for player in remaining_players:
            cards = all_hands.get(player, [])
            print(f"  {player.name}: {cards}")
        print(f"  Community: {community_cards}")
        
        # Placeholder - would call ML evaluator
        # from ml_module.evaluator import evaluate_winner
        # return evaluate_winner(remaining_players, all_hands, community_cards)
        
        return remaining_players[0]  # Placeholder
