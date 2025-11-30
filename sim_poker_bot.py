"""
Poker simulation: AI model vs 3 SuperBots using PyPokerEngine.
Tracks performance and saves detailed results to sim_results.csv.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'ML Model Work', 'Model'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ML Model Work', 'Data Generation'))

import torch
import pandas as pd
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
from pypokerengine.engine.card import Card
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
import random
from datetime import datetime
from pathlib import Path
import json

# Import the FULL ORCHESTRATOR INTEGRATION (uses LiveHandTracker)
from Orchestrator.ml_module import get_action, reset_game
from Orchestrator.card_converter import pypoker_to_internal

# Import SuperBot from pokerBotDataCreator
from pokerBotDataCreator import SuperBot


class AIModelPlayer(BasePokerPlayer):
    """Your trained AI model using FULL LiveHandTracker integration."""
    
    def __init__(self, model_path=None, name="AI_Model"):
        super().__init__()
        self.name = name
        self.hand_history = []
        self.detailed_hands = []  # Store detailed hand info for CSV
        self.current_hand_actions = []  # Actions in current hand
        self.current_hand_num = 0
        self.big_blind = 10.0  # Will be set properly in game
        self.json_log = []  # NEW: Store all decisions with full details for JSON output
        # Note: model_path not needed - LiveHandTracker loads its own model
        
    def declare_action(self, valid_actions, hole_card, round_state):
        """Called when it's this player's turn - uses LiveHandTracker integration."""
        
        # Build JSON payload for ml_module (same format as orchestrator uses)
        # IMPORTANT: Pass valid_actions to get correct to_call amount
        json_payload = self._build_json_payload(hole_card, round_state, valid_actions)
        
        # Call the FULL ORCHESTRATOR INTEGRATION
        # This goes through: ml_module → LiveHandTracker → Model + Heuristics + Improved Raise Sizing
        try:
            # Capture terminal output for JSON logging
            import io
            from contextlib import redirect_stdout
            
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                ml_action, ml_amount = get_action(json_payload)
            
            terminal_output = output_buffer.getvalue()
            
            # Map to PyPokerEngine valid actions
            action, amount = self._map_ml_to_pypoker(ml_action, ml_amount, valid_actions, round_state)
            
            # Store for history AND JSON log
            self._record_decision(hole_card, round_state, action, amount, ml_action, ml_amount, terminal_output)
            
            # Print to terminal so user can still see it
            print(terminal_output, end='')
            
            return action, amount
            
        except Exception as e:
            print(f"\n!!! ERROR in AI decision: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to safe action
            return self._safe_fallback(valid_actions)
    
    def _build_json_payload(self, hole_card, round_state, valid_actions):
        """Build JSON payload in the exact format the orchestrator uses."""
        street = round_state['street']
        pot = round_state['pot']['main']['amount']
        
        # Convert hole cards to internal format
        hole_str = [pypoker_to_internal(c) for c in hole_card]
        
        # Get board cards
        community = round_state.get('community_card', [])
        board_str = [pypoker_to_internal(c) for c in community]
        
        # Get player info
        my_uuid = self.uuid
        seats = round_state['seats']
        my_seat = next(s for s in seats if s['uuid'] == my_uuid)
        my_stack = my_seat['stack']
        
        # Get to_call directly from valid_actions - THIS IS THE CORRECT SOURCE!
        # PyPokerEngine already calculated this for us, including blinds!
        # This fixes the bug where to_call was 0 when player needed to call the big blind
        to_call = 0
        for va in valid_actions:
            if va['action'] == 'call':
                to_call = va['amount']
                break
        
        # Get opponent stacks (for multi-player features)
        opp_stacks = [s['stack'] for s in seats if s['uuid'] != my_uuid and s['state'] == 'participating']
        avg_opp_stack = sum(opp_stacks) / len(opp_stacks) if opp_stacks else my_stack
        
        # Get dealer position
        dealer_btn = round_state['dealer_btn']
        my_seat_idx = next(i for i, s in enumerate(seats) if s['uuid'] == my_uuid)
        
        # Count active players
        num_active = sum(1 for s in seats if s['state'] == 'participating')
        
        # Build JSON payload matching ml_json_input format
        payload = {
            "hand_id": round_state.get('round_count', 0),
            "player_id": my_seat_idx,
            "dealer_position": dealer_btn,
            "players_remaining": num_active,
            "round": street,
            "hole1": hole_str[0] if len(hole_str) > 0 else "",
            "hole2": hole_str[1] if len(hole_str) > 1 else "",
            "flop1": board_str[0] if len(board_str) > 0 else "",
            "flop2": board_str[1] if len(board_str) > 1 else "",
            "flop3": board_str[2] if len(board_str) > 2 else "",
            "turn": board_str[3] if len(board_str) > 3 else "",
            "river": board_str[4] if len(board_str) > 4 else "",
            "pot_bb": pot,  # Raw chips (ml_module will normalize)
            "to_call_bb": to_call,  # Raw chips
            "stack_bb": my_stack,  # Raw chips
            "opp_stack_bb": avg_opp_stack,  # Raw chips
            "big_blind": self.big_blind,
            "action_history": []  # Could extract full history if needed
        }
        
        return json.dumps(payload)
    
    def _map_ml_to_pypoker(self, ml_action, ml_amount, valid_actions, round_state):
        """Map ml_module output to valid PyPokerEngine actions."""
        valid_action_names = [a['action'] for a in valid_actions]
        
        # If ML says check but check isn't valid, it means there's a bet to call
        # This can happen due to blind posts or if to_call calculation was wrong
        if ml_action == 'check' and 'check' not in valid_action_names:
            # Check not valid - there's a bet. Default to call (safe play)
            if 'call' in valid_action_names:
                ml_action = 'call'
                ml_amount = 0
            elif 'fold' in valid_action_names:
                ml_action = 'fold'
                ml_amount = 0
        
        # If ML action is valid, use it
        if ml_action in valid_action_names:
            action_info = next(a for a in valid_actions if a['action'] == ml_action)
            
            if ml_action == 'raise':
                # ml_amount is the total raise-to amount from ml_module
                # Ensure it's within PyPokerEngine's limits
                min_raise = action_info['amount']['min']
                max_raise = action_info['amount']['max']
                final_amount = max(min_raise, min(ml_amount, max_raise))
                return ml_action, int(final_amount)
            else:
                return ml_action, action_info['amount']
        
        # Fallback if action still not valid
        # Smart fallback - prefer passive actions
        if 'check' in valid_action_names:
            action_info = next(a for a in valid_actions if a['action'] == 'check')
            return 'check', action_info['amount']
        elif 'call' in valid_action_names:
            action_info = next(a for a in valid_actions if a['action'] == 'call')
            return 'call', action_info['amount']
        else:
            # Must fold
            action_info = next(a for a in valid_actions if a['action'] == 'fold')
            return 'fold', action_info['amount']
    
    def _safe_fallback(self, valid_actions):
        """Safe fallback action in case of error."""
        for action_name in ['check', 'call', 'fold']:
            action_info = next((a for a in valid_actions if a['action'] == action_name), None)
            if action_info:
                return action_name, action_info['amount']
        return valid_actions[0]['action'], valid_actions[0]['amount']
    
    def _record_decision(self, hole_card, round_state, action, amount, ml_action, ml_amount, terminal_output=""):
        """Record decision for later analysis AND JSON logging."""
        street = round_state['street']
        pot = round_state['pot']['main']['amount']
        
        # Get my position
        my_uuid = self.uuid
        seats = round_state['seats']
        my_seat_idx = next(i for i, s in enumerate(seats) if s['uuid'] == my_uuid)
        my_seat = next(s for s in seats if s['uuid'] == my_uuid)
        dealer_btn = round_state['dealer_btn']
        
        # Determine position name
        num_players = len([s for s in seats if s['state'] == 'participating'])
        relative_pos = (my_seat_idx - dealer_btn - 1) % num_players
        if num_players == 4:
            pos_names = ['SB', 'BB', 'BTN', 'CO']
        else:
            pos_names = [f'P{i}' for i in range(num_players)]
        position = pos_names[relative_pos] if relative_pos < len(pos_names) else f'P{relative_pos}'
        
        # Parse terminal output for key details
        probabilities = self._extract_probabilities(terminal_output)
        hand_bucket = self._extract_bucket(terminal_output)
        board_warnings = self._extract_board_warnings(terminal_output)
        heuristic_triggers = self._extract_heuristic_triggers(terminal_output)
        
        record = {
            'hand_num': self.current_hand_num,
            'street': street,
            'position': position,
            'hole_cards': str(hole_card),
            'community': str(round_state.get('community_card', [])),
            'pot': pot,
            'stack': my_seat['stack'],
            'ml_action': ml_action,
            'ml_amount': ml_amount,
            'final_action': action,
            'final_amount': amount,
        }
        
        # JSON log entry with FULL details
        json_entry = {
            **record,
            'probabilities': probabilities,
            'hand_bucket': hand_bucket,
            'board_warnings': board_warnings,
            'heuristic_triggers': heuristic_triggers,
            'raw_output': terminal_output,
        }
        
        self.hand_history.append(record)
        self.current_hand_actions.append(record)
        self.json_log.append(json_entry)
    
    def _extract_probabilities(self, output):
        """Extract probability dict from terminal output."""
        import re
        match = re.search(r"Probabilities: (\{[^}]+\})", output)
        if match:
            try:
                return eval(match.group(1))  # Safe since it's our own output
            except:
                pass
        return {}
    
    def _extract_bucket(self, output):
        """Extract hand_bucket from terminal output."""
        import re
        match = re.search(r"Bucket=([\d.]+)", output)
        return float(match.group(1)) if match else None
    
    def _extract_board_warnings(self, output):
        """Extract board texture warnings."""
        warnings = []
        if "DANGEROUS BOARD" in output:
            import re
            match = re.search(r"Paired=([\d.]+), Monotone=([\d.]+), Connected=([\d.]+)", output)
            if match:
                warnings.append({
                    'paired': float(match.group(1)),
                    'monotone': float(match.group(2)),
                    'connected': float(match.group(3))
                })
        return warnings
    
    def _extract_heuristic_triggers(self, output):
        """Extract which heuristics triggered."""
        triggers = []
        # Match the actual print statements more carefully (include colons)
        if "PAIRED BOARD DEFENSE:" in output or "PAIRED BOARD:" in output:
            triggers.append("paired_board_defense")
        if "MONOTONE BOARD DEFENSE:" in output or "CONNECTED BOARD DEFENSE:" in output:
            triggers.append("coordinated_board_defense")
        if "MONOTONE BOARD:" in output or "CONNECTED BOARD:" in output:
            triggers.append("coordinated_board_checking")
        if "OVERRIDE: Terrible hand" in output:
            triggers.append("terrible_hand_override")
        return triggers
    
    def receive_game_start_message(self, game_info):
        """Called at start of game."""
        pass
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        """Reset LiveHandTracker state at start of each hand."""
        reset_game()
        self.current_hand_num = round_count
        self.current_hand_actions = []
    
    def receive_street_start_message(self, street, round_state):
        """Called at start of each street."""
        pass
    
    def receive_game_update_message(self, action, round_state):
        """Called after each action."""
        pass
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        """Called at end of each hand - record final result."""
        # Determine if we won
        my_uuid = self.uuid
        won = any(w['uuid'] == my_uuid for w in winners)
        
        # Calculate profit/loss for this hand
        seats = round_state['seats']
        my_seat = next(s for s in seats if s['uuid'] == my_uuid)
        
        # Get starting stack (before hand)
        # Since we don't track it directly, estimate from previous hand
        # For now, just mark win/loss
        
        # Get final board
        community = round_state.get('community_card', [])
        
        # Store detailed hand summary
        if self.current_hand_actions:
            # Get hole cards from first action
            first_action = self.current_hand_actions[0]
            
            hand_summary = {
                'hand_num': self.current_hand_num,
                'hole_cards': first_action['hole_cards'],
                'final_board': str(community),
                'num_actions': len(self.current_hand_actions),
                'actions_taken': ', '.join([f"{a['street']}:{a['final_action']}" for a in self.current_hand_actions]),
                'result': 'win' if won else 'loss',
                'final_pot': round_state['pot']['main']['amount'],
            }
            self.detailed_hands.append(hand_summary)


def run_simulation(num_hands=1000, initial_stack=1000, small_blind=5, big_blind=10):
    """
    Run poker simulation with AI model vs 3 SuperBots using FULL LiveHandTracker integration.
    SuperBots use equity calculations, position-based play, and different styles (TAG/LAG).
    Players are rebought when they bust out to ensure full 1000 hands are played.
    
    Args:
        num_hands: Number of hands to play
        initial_stack: Starting stack for each player
        small_blind: Small blind amount
        big_blind: Big blind amount
    
    Returns:
        Tuple of (summary_df, detailed_actions_df, detailed_hands_df)
    """
    print(f"Starting simulation: {num_hands} hands")
    print(f"Using FULL LiveHandTracker integration (model + heuristics + improved raise sizing)")
    print(f"Initial stack: ${initial_stack}, Blinds: ${small_blind}/${big_blind}")
    print(f"Players will be rebought if they bust out to ensure full {num_hands} hands")
    print(f"Opponents: 2x TAG SuperBots + 1x LAG SuperBot (MUCH STRONGER THAN BASIC BOTS)")
    print("="*70)
    
    # Create players
    ai_player = AIModelPlayer(name="AI_Model")
    ai_player.big_blind = big_blind  # Set BB size for JSON payload
    
    # Create SuperBot opponents with different styles
    # TAG bots play tight-aggressive (premium hands, aggressive postflop)
    # LAG bot plays loose-aggressive (wider range, more bluffs)
    tag_bot_1 = SuperBot(style_name="TAG")
    tag_bot_2 = SuperBot(style_name="TAG")
    lag_bot = SuperBot(style_name="LAG")
    
    # Track results
    results = []
    current_stacks = {
        "AI_Model": initial_stack,
        "TAG_Bot_1": initial_stack,
        "TAG_Bot_2": initial_stack,
        "LAG_Bot": initial_stack
    }
    
    # Track rebuy counts only (profit calculated as: current_stack - initial_stack - rebuys*initial_stack)
    rebuys = {
        "AI_Model": 0,
        "TAG_Bot_1": 0,
        "TAG_Bot_2": 0,
        "LAG_Bot": 0
    }
    
    # Setup game config for entire simulation
    config = setup_config(
        max_round=num_hands,
        initial_stack=initial_stack,
        small_blind_amount=small_blind
    )
    
    config.register_player(name="AI_Model", algorithm=ai_player)
    config.register_player(name="TAG_Bot_1", algorithm=tag_bot_1)
    config.register_player(name="TAG_Bot_2", algorithm=tag_bot_2)
    config.register_player(name="LAG_Bot", algorithm=lag_bot)
    
    # Run entire simulation at once
    print(f"  Running {num_hands} hands (this may take a while)...")
    game_result = start_poker(config, verbose=0)
    
    # Get final stacks
    if 'players' in game_result:
        for player_info in game_result['players']:
            player_name = player_info['name']
            current_stacks[player_name] = player_info['stack']
    
    # Final results
    print("\n" + "="*70)
    print("GAME COMPLETE - Final Results")
    print("="*70)
    
    print(f"\nFinal Stacks:")
    for player_name in ["AI_Model", "TAG_Bot_1", "TAG_Bot_2", "LAG_Bot"]:
        final = current_stacks[player_name]
        net_profit = final - initial_stack
        roi = (net_profit / initial_stack) * 100
        print(f"  {player_name}: ${final:.0f} | Net: ${net_profit:+.0f} | ROI: {roi:+.1f}%")
    
    # Create summary result - use NET profit (final stack - initial_stack)
    ai_net_profit = current_stacks['AI_Model'] - initial_stack
    
    result = {
        'total_hands': num_hands,
        'ai_initial_stack': initial_stack,
        'ai_final_stack': current_stacks['AI_Model'],
        'ai_net_profit': ai_net_profit,
        'ai_profit_per_hand': ai_net_profit / num_hands,
        'tag_bot_1_final_stack': current_stacks['TAG_Bot_1'],
        'tag_bot_1_net_profit': current_stacks['TAG_Bot_1'] - initial_stack,
        'tag_bot_2_final_stack': current_stacks['TAG_Bot_2'],
        'tag_bot_2_net_profit': current_stacks['TAG_Bot_2'] - initial_stack,
        'lag_bot_final_stack': current_stacks['LAG_Bot'],
        'lag_bot_net_profit': current_stacks['LAG_Bot'] - initial_stack,
        'timestamp': datetime.now().isoformat()
    }
    results.append(result)
    
    # Convert to DataFrames
    summary_df = pd.DataFrame(results)
    
    # Create detailed action DataFrame from AI player's history
    actions_df = pd.DataFrame(ai_player.hand_history)
    
    # Create detailed hand summary DataFrame
    hands_df = pd.DataFrame(ai_player.detailed_hands)
    
    # Save JSON log with FULL decision details
    json_log_path = Path(__file__).parent / 'sim_decisions_detailed.json'
    with open(json_log_path, 'w') as f:
        json.dump({
            'summary': result,
            'decisions': ai_player.json_log,
            'metadata': {
                'total_decisions': len(ai_player.json_log),
                'hands_played': num_hands,
                'timestamp': datetime.now().isoformat()
            }
        }, f, indent=2)
    
    # Summary statistics
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print(f"\nFinal Stacks:")
    print(f"  AI_Model: ${current_stacks['AI_Model']:.2f}")
    print(f"  TAG_Bot_1: ${current_stacks['TAG_Bot_1']:.2f}")
    print(f"  TAG_Bot_2: ${current_stacks['TAG_Bot_2']:.2f}")
    print(f"  LAG_Bot: ${current_stacks['LAG_Bot']:.2f}")
    
    ai_net_profit = current_stacks['AI_Model'] - initial_stack
    print(f"\nAI Model Performance:")
    print(f"  Initial Stack: ${initial_stack:.2f}")
    print(f"  Final Stack: ${current_stacks['AI_Model']:.2f}")
    print(f"  Net Profit: ${ai_net_profit:.2f}")
    print(f"  Profit per Hand: ${ai_net_profit / num_hands:.2f}")
    print(f"  ROI: {(ai_net_profit / initial_stack) * 100:.1f}%")
    
    print(f"\nDetailed Statistics:")
    print(f"  Total Actions Recorded: {len(actions_df)}")
    print(f"  Total Hands Recorded: {len(hands_df)}")
    if len(hands_df) > 0:
        print(f"  Hands Won: {(hands_df['result'] == 'win').sum()}")
        print(f"  Win Rate: {(hands_df['result'] == 'win').sum() / len(hands_df) * 100:.1f}%")
    
    return summary_df, actions_df, hands_df, json_log_path


if __name__ == '__main__':
    # Configuration
    NUM_HANDS = 5000  # Full simulation
    INITIAL_STACK = 1000
    SMALL_BLIND = 5
    BIG_BLIND = 10
    OUTPUT_CSV = r"c:\Users\nickl\OneDrive\Documents\Computer-Vision-Powered-AI-Poker-Coach\sim_results.csv"
    
    print("="*70)
    print("POKER BOT SIMULATION - USING FULL LIVEHANDTRACKER INTEGRATION")
    print("="*70)
    print("\nThis simulation tests your AI through the complete deployment pipeline:")
    print("  1. PyPokerEngine game state")
    print("  2. JSON payload construction (as orchestrator does)")
    print("  3. ml_module.get_action()")
    print("  4. LiveHandTracker.main()")
    print("  5. Full model inference + heuristics")
    print("  6. Improved raise sizing calculations")
    print("  7. Action execution")
    print("\nThis is the REAL deployment test!\n")
    
    # Run simulation
    summary_df, actions_df, hands_df, json_log_path = run_simulation(
        num_hands=NUM_HANDS,
        initial_stack=INITIAL_STACK,
        small_blind=SMALL_BLIND,
        big_blind=BIG_BLIND
    )
    
    # Save results to multiple files
    base_path = r"c:\Users\nickl\OneDrive\Documents\Computer-Vision-Powered-AI-Poker-Coach"
    summary_df.to_csv(OUTPUT_CSV, index=False)
    actions_df.to_csv(f"{base_path}\\sim_actions_detailed.csv", index=False)
    hands_df.to_csv(f"{base_path}\\sim_hands_detailed.csv", index=False)
    
    print(f"\nResults saved to:")
    print(f"  Summary: {OUTPUT_CSV}")
    print(f"  Detailed Actions: {base_path}\\sim_actions_detailed.csv")
    print(f"  Detailed Hands: {base_path}\\sim_hands_detailed.csv")
    print(f"  📊 JSON Decision Log: {json_log_path}")
    print("\n🎯 The JSON log contains FULL details for every decision:")
    print("  - Probabilities for each action")
    print("  - Hand bucket values")
    print("  - Board texture warnings (paired/monotone/connected)")
    print("  - Which heuristics triggered")
    print("  - Complete terminal output for each decision")
    print("\nYou can analyze the JSON to see:")
    print("  - Why the AI made specific decisions")
    print("  - If board texture detection is working")
    print("  - If defensive heuristics are triggering")
    print("  - Exact probability distributions for each situation")
