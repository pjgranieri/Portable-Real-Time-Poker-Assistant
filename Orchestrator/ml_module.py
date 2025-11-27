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
        # (Raise sizing logic will be tuned after model training completes)
        
        pot_chips = float(data.get('pot_bb', 0) or 0)
        to_call_chips = float(data.get('to_call_bb', 0) or 0)
        stack_chips = float(data.get('stack_bb', 0) or 0)
        opp_stack = float(data.get('opp_stack_bb', 0) or 0)
        street = (data.get('round') or 'preflop').lower()

        # Default raise sizing (can be tuned based on model behavior)
        # For now, use standard 2.5x-3x raises preflop, 0.5-0.75 pot postflop
        if street == 'preflop':
            # Preflop: 2.5-3x the big blind or to_call amount
            if to_call_chips > 0:
                # Facing a raise: 3x their raise
                additional = int(max(2.0, to_call_chips * 2.0))
            else:
                # Opening: 2.5-3x BB
                additional = int(max(2.0, 2.5))
        else:
            # Postflop: 0.5-0.75 pot sizing
            pot_before_call = pot_chips - to_call_chips if to_call_chips > 0 else pot_chips
            additional = int(max(1.0, pot_before_call * 0.66))
        
        requested_total = int(to_call_chips + additional)

        # Ensure minimum valid raise (at least to_call + 1)
        min_total = int(to_call_chips + 1)
        if requested_total < min_total:
            requested_total = min_total

        # Cap at effective stack size
        hard_cap = int(max(1.0, min(stack_chips, max(1.0, opp_stack)))) if stack_chips > 0 else requested_total
        requested_total = int(min(requested_total, hard_cap))

        final_action = "raise"
        final_value = int(requested_total)
    
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
