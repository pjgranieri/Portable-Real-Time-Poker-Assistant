"""
ML Module Integration
Connects the poker orchestrator to the LiveHandTracker ML model
"""
import json
import sys
import os


# Add ML Model directory to path
ml_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ML Model Work", "Model")
sys.path.insert(0, ml_model_path)

from LiveHandTracker import main as get_ml_prediction, reset_hand

def get_action(json_payload, ml_generator=None):
    """
    Get ML model's recommended action from JSON payload
    
    Args:
        json_payload (str): JSON string from MLJSONGenerator
        ml_generator: MLJSONGenerator instance to record action (optional)
        
    Returns:
        tuple: (action, value) where:
            - action: str - "fold", "check", "call", or "raise"
            - value: float - amount to raise (0 for fold/check/call)
    """
    # Parse JSON
    data = json.loads(json_payload)
    
    # align BTN & preflop handling
    if data.get('player_id', 0) == 0:
        data['player_id'] = 1
    round_name = (data.get('round') or '').lower()
    is_first_to_act = not bool(data.get('action'))
    if round_name == 'preflop' and is_first_to_act:
        if float(data.get('pot_bb', 0) or 0) <= 0:
            data['pot_bb'] = 15.0
        # Only zero to_call if truly opening (no completion needed)
        if float(data.get('to_call_bb', 0) or 0) <= 1e-6:
            data['to_call_bb'] = 0.0

    # Get ML prediction
    result = get_ml_prediction(data)
    
    # Extract action
    ml_action = result['action']  # 4-class model: "raise", "call", "fold", "check"
    confidence = result['confidence']
    probabilities = result.get('probabilities', {})
    
    # Print ML decision for visibility
    print(f"\n{'='*60}")
    print(f"[ML MODEL DECISION]")
    print(f"{'='*60}")
    print(f"   Action: {ml_action.upper()}")
    print(f"   Confidence: {confidence:.1%}")
    if probabilities:
        print(f"   Probabilities: {probabilities}")
    print(f"{'='*60}\n")
    
    # Convert ML action to orchestrator format
    # LiveHandTracker now returns 4 classes: fold, check, call, raise
    # Orchestrator expects: (action: str, value: float)
    
    final_action = None
    final_value = 0
    
    if ml_action == "fold":
        final_action = "fold"
        final_value = 0
    
    elif ml_action == "check":
        final_action = "check"
        final_value = 0
    
    elif ml_action == "call":
        final_action = "call"
        final_value = 0
    
    elif ml_action == "raise":
        # Calculate raise size using context-aware heuristics
        # IMPROVEMENTS:
        # 1. Dynamic sizing based on raise confidence (hand strength proxy)
        # 2. SPR-aware adjustments (stack-to-pot ratio)
        # 3. Position-aware preflop opens
        # 4. Street-specific logic (preflop vs postflop vs river)
        # 5. Situation-aware (opening, 3-betting, check-raising, etc.)
        # 6. Proper minimum raise calculations (previous bet size, not just +1)
        
        pot_chips = float(data.get('pot_bb', 0) or 0)
        to_call_chips = float(data.get('to_call_bb', 0) or 0)
        stack_chips = float(data.get('stack_bb', 0) or 0)
        opp_stack = float(data.get('opp_stack_bb', 0) or 0)
        street = (data.get('round') or 'preflop').lower()
        
        # Get hand strength info from probabilities if available
        raise_confidence = float(probabilities.get('raise', '0.0'))
        
        # Calculate effective stacks (SPR = Stack-to-Pot Ratio)
        effective_stack = min(stack_chips, opp_stack) if opp_stack > 0 else stack_chips
        pot_for_spr = pot_chips if pot_chips > 0 else 1.0
        spr = effective_stack / pot_for_spr
        
        # Determine raise sizing based on street and situation
        if street == 'preflop':
            if to_call_chips > 0:
                # Facing a raise: 3-betting
                # Use dynamic sizing based on position and raise confidence
                if raise_confidence > 0.70:
                    # Very confident (premium hands) - larger 3-bet for value
                    multiplier = 3.5 if to_call_chips <= 3.0 else 3.0
                elif raise_confidence > 0.50:
                    # Moderately confident - standard 3-bet
                    multiplier = 3.0
                else:
                    # Less confident (bluffs/marginal) - smaller 3-bet
                    multiplier = 2.5
                
                additional = to_call_chips * multiplier
                
                # Adjust for stack depth
                if spr < 5:  # Short stacked
                    additional = min(additional, effective_stack * 0.4)
                elif spr > 20:  # Deep stacked
                    additional = max(additional, to_call_chips * 2.5)
            else:
                # Opening raise preflop
                # Standard open: 2.5-3.5 BB based on position
                player_id = data.get('player_id', 1)
                dealer_pos = data.get('dealer_position', 0)
                num_players = data.get('players_remaining', 2)
                
                # BTN/Late position: larger opens (3-3.5x)
                # Early/Blinds: smaller opens (2.5-3x)
                seats_from_btn = (player_id - dealer_pos) % num_players
                if seats_from_btn == 0:  # BTN
                    base_open = 3.0
                elif seats_from_btn in [1, 2]:  # Blinds
                    base_open = 2.75
                else:  # Early
                    base_open = 2.5
                
                # Adjust for confidence
                if raise_confidence > 0.70:
                    base_open += 0.5  # Premium hands
                
                additional = base_open
        else:
            # Postflop raise sizing
            pot_before_call = pot_chips - to_call_chips if to_call_chips > 0 else pot_chips
            
            if to_call_chips > 0:
                # Facing a bet: raising (check-raise or re-raise)
                bet_size = to_call_chips
                bet_frac = bet_size / pot_before_call if pot_before_call > 0 else 1.0
                
                if raise_confidence > 0.70:
                    # Very confident (strong hands/bluffs) - larger raise
                    # Aim for 2.5-3x their bet
                    raise_multiplier = 2.8
                elif raise_confidence > 0.50:
                    # Moderate confidence - standard raise
                    raise_multiplier = 2.5
                else:
                    # Less confident - minimum raise
                    raise_multiplier = 2.0
                
                # Calculate raise amount (total chips committed)
                additional = bet_size * raise_multiplier
                
                # Adjust for board texture if we have action history
                action_hist = data.get('action_history', [])
                if len(action_hist) > 3:  # Active betting
                    # On wet boards or multi-street action, use larger sizing
                    additional *= 1.1
                
                # Stack depth consideration
                if spr < 3:  # Very short stacked - just shove
                    additional = effective_stack
                elif spr < 6:  # Short stacked - commit more
                    additional = min(additional, effective_stack * 0.6)
            else:
                # First to act: betting (donk bet or continuation bet)
                # Use pot-fraction sizing based on confidence
                if raise_confidence > 0.75:
                    # Very strong - large value bet or bluff
                    pot_fraction = 0.75  # 75% pot
                elif raise_confidence > 0.60:
                    # Strong - standard bet
                    pot_fraction = 0.66  # 2/3 pot
                elif raise_confidence > 0.45:
                    # Medium - smaller bet
                    pot_fraction = 0.50  # 1/2 pot
                else:
                    # Weak/blocking bet
                    pot_fraction = 0.33  # 1/3 pot
                
                additional = pot_before_call * pot_fraction
                
                # Minimum bet sizing (at least 1 BB)
                additional = max(1.0, additional)
                
                # On river with strong confidence, bet bigger for value
                if street == 'river' and raise_confidence > 0.70:
                    additional = pot_before_call * 0.85
                
                # Stack depth adjustment
                if spr < 3:  # Short - overbet or shove
                    additional = effective_stack
                elif spr > 15:  # Deep - can use more varied sizing
                    if raise_confidence > 0.80:
                        additional = pot_before_call * 1.0  # Pot-sized bet
        
        # Calculate total raise amount (what we're raising TO, not BY)
        requested_total = to_call_chips + additional
        
        # Ensure minimum valid raise (at least to_call + 1 BB)
        min_raise = to_call_chips + max(1.0, to_call_chips)  # Minimum raise = previous bet size
        if requested_total < min_raise:
            requested_total = min_raise
        
        # Cap at effective stack size (can't bet more than we have)
        if requested_total > effective_stack:
            requested_total = effective_stack
        
        # Convert to integer for orchestrator (round to nearest chip)
        final_action = "raise"
        final_value = round(requested_total)
    
    else:
        # Unknown action - default to check/call
        print(f"WARNING: Unknown ML action '{ml_action}', defaulting to call")
        final_action = "call"
        final_value = 0
    
    # Record action in ml_generator if provided
    if ml_generator is not None:
        ml_generator.record_action("PlayerCoach", final_action, final_value)
    
    return (final_action, final_value)


def reset_game():
    """Call this when starting a new hand to reset tracker state"""
    reset_hand()


if __name__ == "__main__":
    # Test the integration
    test_payload = {
        "hand_id": 1,
        "player_id": 1,
        "round": "preflop",
        "hole1": "HA",
        "hole2": "DK",
        "flop1": "",
        "flop2": "",
        "flop3": "",
        "turn": "",
        "river": "",
        "stack_bb": 170,
        "opp_stack_bb": 165,
        "to_call_bb": 5,
        "pot_bb": 15,
        "action": "",
        "final_pot_bb": ""
    }
    
    print("Testing ML module integration...")
    action, value = get_action(json.dumps(test_payload))
    print(f"\nResult: action='{action}', value={value}")
