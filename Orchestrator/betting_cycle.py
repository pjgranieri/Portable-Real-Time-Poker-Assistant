# Handles betting cycles
from Orchestrator.config import Player

def run_betting_cycle(players, community_pot, call_value=0, is_preflop=False):
    """
    Runs a complete betting cycle starting from small blind.
    All 4 players (including coach) must act.
    Continues until everyone has either folded or matched the current bet.
    
    Args:
        players: PlayerManager instance
        community_pot: Current pot value
        call_value: Current bet to call (should be 0 for new rounds)
        is_preflop: If True, post blinds automatically
    """
    start_index = players.small_blind_index()
    players_acted = {p: False for p in Player}  # Track who has acted this round
    last_raiser = None
    
    # Handle blinds for pre-flop
    if is_preflop:
        small_blind_player = list(Player)[start_index]
        big_blind_player = list(Player)[(start_index + 1) % 4]
        
        # Small blind posts 5
        small_blind_amount = 5
        players.get(small_blind_player)["bankroll"] -= small_blind_amount
        community_pot += small_blind_amount
        print(f"  💰 {small_blind_player.name} posts small blind: ${small_blind_amount}")
        
        # Big blind posts 10
        big_blind_amount = 10
        players.get(big_blind_player)["bankroll"] -= big_blind_amount
        community_pot += big_blind_amount
        call_value = big_blind_amount
        print(f"  💰 {big_blind_player.name} posts big blind: ${big_blind_amount}")
        print(f"  Current call value: ${call_value}\n")
        
        # Blinds have NOT acted yet - they need to respond to any raises
        # Start action after big blind (player 2 positions from small blind)
        start_index = (start_index + 2) % 4
    
    cycle_position = 0
    
    while True:
        current_player_index = (start_index + cycle_position) % 4
        current_player = list(Player)[current_player_index]
        player_data = players.get(current_player)
        
        # Skip folded players
        if player_data["folded"]:
            cycle_position += 1
            continue
        
        # Check if betting round is complete
        active_players = [p for p in Player if not players.get(p)["folded"]]
        all_acted = all(players_acted[p] for p in active_players)
        
        if all_acted:
            # Everyone has acted and matched bets
            break
        
        # Get crop region for this player
        crop_region = players.get_crop_region(current_player)
        
        # Ask for action
        action, value = players.get_action(current_player, crop_region, call_value)
        
        if action == "fold":
            player_data["folded"] = True
            players_acted[current_player] = True
            cycle_position += 1
            
        elif action == "check":
            # Can only check if call_value is 0
            if call_value == 0:
                players_acted[current_player] = True
                cycle_position += 1
            else:
                print(f"  ⚠️  Cannot check - must call ${call_value} or fold")
                continue  # Ask again
            
        elif action == "call":
            # Can only call if there's something to call
            if call_value > 0:
                # For blinds, they need to call the difference
                if is_preflop:
                    small_blind_player = list(Player)[players.small_blind_index()]
                    big_blind_player = list(Player)[(players.small_blind_index() + 1) % 4]
                    
                    if current_player == small_blind_player:
                        # Small blind already paid 5, only pay difference
                        amount_to_call = call_value - 5
                    elif current_player == big_blind_player:
                        # Big blind already paid 10, only pay difference
                        amount_to_call = call_value - 10
                    else:
                        amount_to_call = call_value
                else:
                    amount_to_call = call_value
                
                player_data["bankroll"] -= amount_to_call
                community_pot += amount_to_call
                players_acted[current_player] = True
                cycle_position += 1
            else:
                print(f"  ⚠️  Nothing to call - use 'check' instead")
                continue  # Ask again
            
        elif action == "raise":
            # For blinds raising, add their blind to the raise
            if is_preflop:
                small_blind_player = list(Player)[players.small_blind_index()]
                big_blind_player = list(Player)[(players.small_blind_index() + 1) % 4]
                
                if current_player == small_blind_player:
                    # Small blind already paid 5, total raise is value + 5
                    total_raise = value
                    amount_to_pay = value - 5
                elif current_player == big_blind_player:
                    # Big blind already paid 10, total raise is value + 10
                    total_raise = value
                    amount_to_pay = value - 10
                else:
                    total_raise = value
                    amount_to_pay = value
            else:
                total_raise = value
                amount_to_pay = value
            
            player_data["bankroll"] -= amount_to_pay
            community_pot += amount_to_pay
            call_value = total_raise
            last_raiser = current_player
            players_acted[current_player] = True
            
            # Reset all other players' acted status (they need to respond to raise)
            for p in Player:
                if p != current_player:
                    players_acted[p] = False
            
            cycle_position += 1
    
    return community_pot, call_value
