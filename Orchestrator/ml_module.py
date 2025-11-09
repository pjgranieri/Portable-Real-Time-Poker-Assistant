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

def get_action(json_payload):
    """
    Get ML model's recommended action from JSON payload
    
    Args:
        json_payload (str): JSON string from MLJSONGenerator
        
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
    ml_action = result['action']  # e.g., "raise_m", "call", "fold", "check"
    confidence = result['confidence']
    
    # Print ML decision for visibility
    print(f"\n{'='*60}")
    print(f"[ML MODEL DECISION]")
    print(f"{'='*60}")
    print(f"   Action: {ml_action.upper()}")
    print(f"   Confidence: {confidence:.1%}")
    print(f"{'='*60}\n")
    
    # Convert ML action to orchestrator format
    # LiveHandTracker returns: fold, check, call, raise_s, raise_m, raise_l
    # Orchestrator expects: (action: str, value: float)
    
    if ml_action == "fold":
        return ("fold", 0)
    
    elif ml_action == "check":
        return ("check", 0)
    
    elif ml_action == "call":
        return ("call", 0)
    
    elif ml_action.startswith("raise_"):
        pot_chips = float(data.get('pot_bb', 0) or 0)
        to_call_chips = float(data.get('to_call_bb', 0) or 0)
        stack_chips = float(data.get('stack_bb', 0) or 0)
        opp_stack = float(data.get('opp_stack_bb', 0) or 0)

        multiplier = 1.0
        if ml_action == "raise_s":
            multiplier = 0.5
        elif ml_action == "raise_m":
            multiplier = 1.0
        elif ml_action == "raise_l":
            multiplier = 2.0

        additional = int(max(1.0, (pot_chips * multiplier) if pot_chips > 0 else (stack_chips * 0.2)))
        requested_total = int(to_call_chips + additional)

        # avoid raise loop
        min_total = int(to_call_chips + 1)
        if requested_total < min_total:
            requested_total = min_total

        # all-in cap
        hard_cap = int(max(1.0, min(stack_chips, max(1.0, opp_stack)))) if stack_chips > 0 else requested_total
        requested_total = int(min(requested_total, hard_cap))

        return ("raise", int(requested_total))
    
    else:
        # Unknown action - default to check/call
        print(f"WARNING: Unknown ML action '{ml_action}', defaulting to call")
        return ("call", 0)


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
