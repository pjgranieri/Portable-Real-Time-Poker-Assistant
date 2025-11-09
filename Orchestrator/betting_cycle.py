# Handles betting cycles
from Orchestrator.config import Player

def run_betting_cycle(players, community_pot, call_value=0, game_state=None, cards=None, ml_generator=None, is_preflop=False):
    """
    Runs a complete betting cycle starting from small blind.
    Players 2 and 3 always fold (only Coach and Player1 play).
    Continues until everyone has either folded or matched the current bet.
    
    Args:
        players: PlayerManager instance
        community_pot: Current pot value
        call_value: Current bet to call (should be 0 for new rounds)
        game_state: Current GameState enum (for ML JSON generation)
        cards: CardManager instance (for ML JSON generation)
        ml_generator: MLJSONGenerator instance (for tracking actions)
        is_preflop: If True, post blinds automatically
    """
    start_index = players.small_blind_index()
    players_acted = {p: False for p in Player}  # Track who has acted this round
    amount_paid_this_round = {p: 0 for p in Player}  # Track how much each player has paid
    has_called = {p: 0 for p in Player}  # Track each player's committed bet amount
    last_raiser = None
    
    # Reset round tracking in ml_generator at start of NEW betting round
    if ml_generator:
        ml_generator.reset_round()
    
    # Players 2 and 3 always fold immediately
    players.get(Player.PlayerTwo)["folded"] = True
    players.get(Player.PlayerThree)["folded"] = True
    players_acted[Player.PlayerTwo] = True
    players_acted[Player.PlayerThree] = True
    
    # Handle blinds for pre-flop
    if is_preflop:
        small_blind_player = list(Player)[start_index]
        big_blind_player = list(Player)[(start_index + 1) % 4]
        
        # Skip if players 2 or 3 are blinds (should rotate between coach and player1)
        if small_blind_player in [Player.PlayerTwo, Player.PlayerThree]:
            small_blind_player = Player.PlayerCoach
        if big_blind_player in [Player.PlayerTwo, Player.PlayerThree]:
            big_blind_player = Player.PlayerOne
        
        # Small blind posts 5
        small_blind_amount = 5
        players.get(small_blind_player)["bankroll"] -= small_blind_amount
        community_pot += small_blind_amount
        amount_paid_this_round[small_blind_player] = small_blind_amount
        has_called[small_blind_player] = small_blind_amount
        print(f"  {small_blind_player.name} posts small blind: ${small_blind_amount}")
        
        # Big blind posts 10
        big_blind_amount = 10
        players.get(big_blind_player)["bankroll"] -= big_blind_amount
        community_pot += big_blind_amount
        amount_paid_this_round[big_blind_player] = big_blind_amount
        has_called[big_blind_player] = big_blind_amount
        call_value = big_blind_amount
        print(f"  {big_blind_player.name} posts big blind: ${big_blind_amount}")
        print(f"  Current call value: ${call_value}\n")
        
        # DON'T update ml_generator with blind posts - they're not "actions"
        # This keeps first_to_act = True until actual betting starts
        
        # Blinds have NOT acted yet - they need to respond to any raises
        # Start action after big blind
        if big_blind_player == Player.PlayerCoach:
            start_index = Player.PlayerOne.value
        else:
            start_index = Player.PlayerCoach.value
    
    cycle_position = 0
    
    while True:
        current_player_index = (start_index + cycle_position) % 4
        current_player = list(Player)[current_player_index]
        player_data = players.get(current_player)
        
        # Skip folded players
        if player_data["folded"]:
            cycle_position += 1
            continue
        
        # Check if only one player remains (early win condition)
        active_players = [p for p in Player if not players.get(p)["folded"]]
        if len(active_players) == 1:
            print(f"\n  Only {active_players[0].name} remains - betting round ends")
            break
        
        # Check if betting round is complete
        all_acted = all(players_acted[p] for p in active_players)
        
        if all_acted:
            # Everyone has acted and matched bets
            break
        
        # Calculate how much this player needs to call
        amount_already_paid = amount_paid_this_round[current_player]
        amount_needed_to_call = call_value - amount_already_paid
        
        # Get crop region for this player
        crop_region = players.get_crop_region(current_player)
        
        # Generate ML JSON if this is the coach
        if current_player == Player.PlayerCoach and ml_generator and game_state and cards:
            json_payload = ml_generator.generate_json_for_coach_action(
                game_state=game_state,
                cards=cards,
                players=players,
                community_pot=community_pot,
                call_value=amount_needed_to_call  # Use the actual amount needed
            )
            print(f"\n{'='*60}")
            print("ML MODEL INPUT (Coach's Turn)")
            print(f"{'='*60}")
            print(json_payload)
            print(f"{'='*60}")
            print("\nNOTE: To send this JSON to your ML model:")
            print("  1. In orchestrator.py, replace mock with: from ml_module import get_action")
            print("  2. Parse JSON: import json; data = json.loads(json_payload)")
            print("  3. Call ML: action, value = get_action(data)")
            print("  4. This is done in orchestrator.py around line 50\n")
        
        # Compute the current minimum legal total raise for informing callers
        highest_has_called = max(has_called.values()) if has_called else 0
        min_raise_total = highest_has_called + 1

        # Ask for action (pass min_raise_total so InputInterface can validate)
        action, value = players.get_action(current_player, crop_region, amount_needed_to_call, min_raise_total)
        
        # Update last action in ML generator (simplified) - NOW we track actions
        if ml_generator:
            if action == "raise":
                ml_generator.set_last_action("raise")
            elif action == "call":
                ml_generator.set_last_action("call")
            elif action == "check":
                ml_generator.set_last_action("check")
            elif action == "fold":
                ml_generator.set_last_action("fold")
        
        if action == "fold":
            player_data["folded"] = True
            players_acted[current_player] = True
            print(f"  {current_player.name} folds")
            cycle_position += 1
            
        elif action == "check":
            # Can only check if amount_needed_to_call is 0
            if amount_needed_to_call == 0:
                players_acted[current_player] = True
                print(f"  {current_player.name} checks")
                cycle_position += 1
            else:
                print(f"  Cannot check - must call ${amount_needed_to_call} or fold")
                continue  # Ask again
            
        elif action == "call":
            # Can only call if there's something to call
            if amount_needed_to_call > 0:
                player_data["bankroll"] -= amount_needed_to_call
                community_pot += amount_needed_to_call
                amount_paid_this_round[current_player] += amount_needed_to_call
                has_called[current_player] = amount_paid_this_round[current_player]
                players_acted[current_player] = True
                print(f"  {current_player.name} calls ${amount_needed_to_call} (paid ${amount_paid_this_round[current_player]} total this round)")
                cycle_position += 1
            else:
                print(f"  Nothing to call - use 'check' instead")
                continue  # Ask again
            
        elif action == "raise":
            requested_total = value
            highest_has_called = max(has_called.values())
            min_raise_total = highest_has_called + 1
            if requested_total < min_raise_total:
                total_raise = min_raise_total
            else:
                total_raise = requested_total
            amount_to_pay = total_raise - amount_already_paid
            if amount_to_pay > player_data["bankroll"]:
                total_raise = amount_already_paid + player_data["bankroll"]
                amount_to_pay = player_data["bankroll"]
            if total_raise <= highest_has_called and amount_to_pay <= 0:
                to_call = highest_has_called - amount_already_paid
                if to_call > 0 and player_data["bankroll"] >= to_call:
                    player_data["bankroll"] -= to_call
                    community_pot += to_call
                    amount_paid_this_round[current_player] += to_call
                    has_called[current_player] = amount_paid_this_round[current_player]
                    players_acted[current_player] = True
                    print(f"  {current_player.name} calls ${to_call}")
                else:
                    players_acted[current_player] = True
                    print(f"  {current_player.name} checks")
                cycle_position += 1
                continue
            player_data["bankroll"] -= amount_to_pay
            community_pot += amount_to_pay
            amount_paid_this_round[current_player] = total_raise
            has_called[current_player] = total_raise
            call_value = total_raise
            last_raiser = current_player
            players_acted[current_player] = True
            print(f"  {current_player.name} raises to ${total_raise} (pays ${amount_to_pay})")
            for p in Player:
                if p != current_player and not players.get(p)["folded"]:
                    players_acted[p] = False
            cycle_position += 1
    
    return community_pot, call_value
