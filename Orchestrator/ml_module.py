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
        # Bucket- and confidence-aware raise sizing, still conservatively capped.
        pot_chips = float(data.get('pot_bb', 0) or 0)
        to_call_chips = float(data.get('to_call_bb', 0) or 0)
        stack_chips = float(data.get('stack_bb', 0) or 0)
        opp_stack = float(data.get('opp_stack_bb', 0) or 0)
        street = (data.get('round') or 'preflop').lower()

        raise_confidence = float(probabilities.get('raise', '0.0'))

        # Hand strength bucket, if present in payload; default to medium.
        hand_bucket = float(data.get('hand_bucket', 1.5) or 1.5)
        is_weak = hand_bucket < 1.5
        is_medium = 1.5 <= hand_bucket < 3.0
        is_strong = hand_bucket >= 3.0

        effective_stack = min(stack_chips, opp_stack) if opp_stack > 0 else stack_chips

        # Cap overall commitment by street to keep things sane.
        if street == 'preflop':
            max_commit = min(effective_stack, 8.0)  # at most 8BB preflop
        elif street in ('flop', 'turn'):
            max_commit = min(effective_stack, 12.0)  # at most 12BB mid-street
        else:  # river
            max_commit = min(effective_stack, 16.0)  # at most 16BB on river

        pot_before_call = pot_chips
        if street == 'preflop':
            # Preflop: standard open/3-bet sizes by bucket.
            if to_call_chips > 0:
                # Facing a raise: aim for ~2.2–3.0x their bet, bucket based.
                if is_strong:
                    base_mult = 3.0
                elif is_medium:
                    base_mult = 2.6
                else:
                    base_mult = 2.2
                base = to_call_chips * base_mult
                target_total = min(base, max_commit)
            else:
                # Opening: 2.2–3.8BB based on bucket and confidence.
                if is_strong:
                    base_open = 3.2
                elif is_medium:
                    base_open = 2.8
                else:
                    base_open = 2.4

                # Nudge slightly by confidence.
                if raise_confidence < 0.4:
                    base_open -= 0.3
                elif raise_confidence > 0.8:
                    base_open += 0.3

                base_open = max(2.0, min(base_open, 3.8))
                target_total = min(base_open, max_commit)
        else:
            # Postflop: pot-fraction bets/raises by bucket, capped.
            if to_call_chips > 0:
                # Raising a bet: around 1.8–2.4x their bet.
                if is_strong:
                    mult = 2.4
                elif is_medium:
                    mult = 2.1
                else:
                    mult = 1.8

                # Very low confidence → shrink slightly.
                if raise_confidence < 0.4:
                    mult -= 0.2

                base = to_call_chips * mult
                target_total = min(base, max_commit)
            else:
                # First to act: keep existing pot control style but let bucket
                # pull the fraction up/down within a sane band.
                if pot_before_call <= 0:
                    pot_before_call = 1.0

                # Start from confidence-driven baseline.
                if raise_confidence > 0.75:
                    frac = 0.55
                elif raise_confidence > 0.5:
                    frac = 0.48
                else:
                    frac = 0.40

                # Bump by bucket tier.
                if is_strong:
                    frac += 0.15  # strong hands closer to 70% pot
                elif is_medium:
                    frac += 0.05

                frac = max(0.35, min(frac, 0.70))
                base = pot_before_call * frac
                target_total = min(base, max_commit)

        # Never go below a legal minimum raise over current bet.
        additional = max(target_total - to_call_chips, 1.0)

        # Calculate total raise amount (what we're raising TO, not BY)
        requested_total = to_call_chips + additional

        # Ensure minimum valid raise (at least to_call + 1 BB)
        min_raise = to_call_chips + max(1.0, to_call_chips)
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
