"""
Input Interface - Standardized interface for card and action inputs
Can be used for manual testing or replaced with CV/ML modules
"""

from Orchestrator.card_converter import CardConverter, validate_card

class InputInterface:
    """Standardized interface for getting user inputs"""
    
    @staticmethod
    def get_card(prompt="Enter card", allow_empty=False):
        """
        Get a single card from user input with validation
        
        Args:
            prompt: Display prompt
            allow_empty: If True, allows empty input
        
        Returns:
            Valid card string in VALUE|SUIT format (e.g., "AH", "10D")
        """
        while True:
            card = input(f"  {prompt} (e.g., AH, 10D, KS): ").strip().upper()
            
            if allow_empty and card == "":
                return ""
            
            if validate_card(card):
                return card
            else:
                print(f"     Invalid card format. Use format like: AH, 10D, KS, 7C")
                print(f"     Valid values: A, 2-10, J, Q, K")
                print(f"     Valid suits: C (Clubs), D (Diamonds), H (Hearts), S (Spades)")
    
    @staticmethod
    def get_cards(count, card_type="card"):
        """
        Get multiple cards from user input with validation
        
        Args:
            count: Number of cards to get
            card_type: Type of cards (for display purposes)
        
        Returns:
            List of valid card strings
        """
        print(f"\n{'='*60}")
        print(f"Enter {count} {card_type}(s)")
        print(f"{'='*60}")
        
        cards = []
        for i in range(count):
            card = InputInterface.get_card(f"Card {i+1}")
            cards.append(card)
        
        return cards
    
    @staticmethod
    def get_action(player_name, call_value, current_bankroll, min_raise_total=None, max_retries=3):
        """
        Get player action from input with validation
        
        Args:
            player_name: Name of player
            call_value: Amount needed to call
            current_bankroll: Player's current bankroll
        
        Returns:
            Tuple of (action, value) where action is "fold", "check", "call", "raise"
        """
        print(f"\n{'='*60}")
        print(f"{player_name}'s turn")
        print(f"   Current bankroll: ${current_bankroll}")
        print(f"   Amount needed to call: ${call_value}")
        
        # Show valid actions based on call_value
        if call_value == 0:
            print(f"   Valid actions: check, raise, fold")
        else:
            print(f"   Valid actions: call (${call_value}), raise, fold")

        # If caller provided a stricter minimum total raise, show it
        if min_raise_total is not None:
            print(f"   Minimum legal total raise: ${min_raise_total}")
        print(f"{'='*60}")
        
        attempts = 0
        while True:
            action_input = input("Enter action: ").strip().lower()
            
            if action_input == "fold":
                return ("fold", 0)
            
            elif action_input == "check":
                if call_value == 0:
                    return ("check", 0)
                else:
                    print(f"  Cannot check - must call ${call_value} or fold")
            
            elif action_input == "call":
                if call_value > 0:
                    if call_value <= current_bankroll:
                        return ("call", call_value)
                    else:
                        print(f"  Not enough chips to call (need ${call_value}, have ${current_bankroll})")
                else:
                    print("  Nothing to call - use 'check' instead")
            
            elif action_input == "raise":
                # If player doesn't have enough to even call, they cannot raise
                if current_bankroll <= call_value:
                    print(f"  Not enough chips to raise (have ${current_bankroll})")
                    # If they can still call, force call; otherwise force fold
                    if call_value <= current_bankroll:
                        print("  Falling back to CALL")
                        return ("call", call_value)
                    else:
                        print("  Falling back to FOLD")
                        return ("fold", 0)

                # Determine the minimum total raise to enforce
                effective_min = call_value + 1
                if min_raise_total is not None:
                    effective_min = max(effective_min, min_raise_total)

                try:
                    amount = int(input(f"  Enter TOTAL raise amount (min ${effective_min}, max ${current_bankroll}): $").strip())
                except ValueError:
                    print("  Invalid amount, try again")
                    attempts += 1
                    if attempts >= max_retries:
                        # Fallback: call if possible, else fold
                        if call_value <= current_bankroll:
                            print("  Too many invalid attempts - auto CALL")
                            return ("call", call_value)
                        else:
                            print("  Too many invalid attempts - auto FOLD")
                            return ("fold", 0)
                    continue

                # Validate amount
                if amount < effective_min:
                    print(f"  Raise must be at least ${effective_min} total")
                elif amount > current_bankroll:
                    print(f"  Cannot raise more than your bankroll (${current_bankroll})")
                else:
                    return ("raise", amount)
            
            else:
                print("  Invalid action. Use: fold, check, call, or raise")
    
    @staticmethod
    def get_winner_selection(remaining_players, community_cards, hole_cards):
        """
        Get winner selection for showdown
        
        Args:
            remaining_players: List of Player enums
            community_cards: List of community cards
            hole_cards: Dict of player hole cards
        
        Returns:
            Player enum of winner
        """
        print(f"\n{'='*60}")
        print("HAND EVALUATION")
        print(f"{'='*60}")
        print(f"Community cards: {community_cards}")
        
        for i, player in enumerate(remaining_players, 1):
            cards = hole_cards.get(player, [])
            print(f"{i}. {player.name}: {cards}")
        
        while True:
            try:
                winner_num = int(input(f"\nSelect winner (1-{len(remaining_players)}): ").strip())
                if 1 <= winner_num <= len(remaining_players):
                    return remaining_players[winner_num - 1]
                else:
                    print(f"  Enter a number between 1 and {len(remaining_players)}")
            except ValueError:
                print("  Invalid input")
    
    @staticmethod
    def wait_for_game_start(hand_number):
        """Wait for game start signal"""
        print(f"\n{'='*60}")
        print(f"NEW GAME (Hand #{hand_number})")
        print(f"{'='*60}")
        input("Press Enter to start new game...")