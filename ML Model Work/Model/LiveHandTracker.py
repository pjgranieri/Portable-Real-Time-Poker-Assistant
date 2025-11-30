import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any

from treys_helpers import evaluate_hand_features

# Import model definition
try:
    from CoachMLP import PokerMLP, Cfg
except ImportError:
    print("Error: Could not import PokerMLP or Cfg from CoachMLP.py")
    class Cfg:
        classes = ["fold", "check", "call", "raise"]
        fold_thresh = 0.55
        fold_margin = 0.05
        call_logit_bias = -0.15
    class PokerMLP(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.net = nn.Linear(10, 6)
        def forward(self, *args, **kwargs):
            pass

# Basic model vocab
STREET_VOCAB = ["preflop", "flop", "turn", "river"]
POS_VOCAB = ["Early", "Late", "Blinds"]
CLASSES = Cfg.classes

# Position aliases from seats to buckets
POS_ALIASES = {
    "SB": "Blinds", "BB": "Blinds",
    "UTG": "Early", "UTG1": "Early", "UTG+1": "Early",
    "MP": "Early", "MP1": "Early", "MP+1": "Early",
    "CO": "Late", "BTN": "Late",
    "Early": "Early", "Late": "Late", "Blinds": "Blinds"
}

# Model artifact paths
THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS_FILE)))
MODEL_DIR = os.path.join(REPO_ROOT, "runs", "poker_mlp_v1")
MODEL_PATH = os.path.join(MODEL_DIR, "model.best.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
META_PATH = os.path.join(MODEL_DIR, "meta.json")

# Card encoding tables
RANKS = "23456789TJQKA"
SUITS = "cdhs"
CARD2IDX = {f"{r}{s}": i for i, (r, s) in enumerate((r, s) for r in RANKS for s in SUITS)}
NUM_CARDS = 52

def card_one_hot(card: str) -> np.ndarray:
    v = np.zeros(NUM_CARDS, dtype=np.float32)
    if not isinstance(card, str) or len(card) != 2:
        return v
    
    c = card.strip().upper()
    if not c or len(c) != 2:
        return v
    
    # Try format 1: rank+suit (e.g., 'As', '2c')
    if c[0] in RANKS and c[1].lower() in SUITS:
        card_key = c[0] + c[1].lower()
    # Try format 2: suit+rank (e.g., 'SA', 'C2', 'HA', 'DK')
    elif c[0] in 'CDHS' and c[1] in RANKS:
        card_key = c[1] + c[0].lower()
    else:
        return v
    
    if card_key in CARD2IDX:
        v[CARD2IDX[card_key]] = 1.0
    return v

def one_hot(value: str, vocab: List[str]) -> np.ndarray:
    v = np.zeros(len(vocab), dtype=np.float32)
    if value in vocab:
        v[vocab.index(value)] = 1.0
    return v

def canonicalize_position(pos: str) -> str:
    if not pos or not isinstance(pos, str):
        raise ValueError(f"Invalid position: {pos}")
    pos_clean = pos.strip().capitalize()
    result = POS_ALIASES.get(pos_clean, POS_ALIASES.get(pos.strip()))
    if result is None:
        raise ValueError(f"Unknown position: {pos}")
    return result


class LiveHandTracker:
    def __init__(self, my_position: str, big_blind_size: float = 1.0):
        self.bb_size = big_blind_size
        self.my_pos = canonicalize_position(my_position)
        
        # Live hand state
        self.hole_cards = []
        self.board_cards = []
        self.current_street = "preflop"
        self.cv_data = {}
        self.action_history = []
        self.my_player_name = "PlayerCoach"
        self.was_pfr_live = False
        
        # Load metadata, model, and scaler
        try:
            import json
            with open(META_PATH, 'r') as f:
                meta = json.load(f)
            
            self.static_dim = meta['static_dim']
            self.numeric_cols_order = meta['numeric_cols']
            self.numeric_dim = len(self.numeric_cols_order)
            self.binary_cols = meta.get('binary_cols', [])
            self.n_scaled_cols_meta = meta.get('n_scaled_cols', None)
            self.learned_temperature = meta.get('temperature', 1.0)
            
            # Load policy hyperparameters
            policy = meta.get('policy', {})
            try:
                Cfg.fold_thresh = policy.get('fold_thresh', getattr(Cfg, 'fold_thresh', 0.0))
                Cfg.fold_margin = policy.get('fold_margin', getattr(Cfg, 'fold_margin', 0.0))
                Cfg.call_logit_bias = policy.get('call_logit_bias', getattr(Cfg, 'call_logit_bias', 0.0))
                self.class_logit_bias = policy.get('class_logit_bias', {})
            except Exception:
                pass

            print(f"Loading model with static_dim={self.static_dim}, numeric_dim={self.numeric_dim}")
            
            # Load model
            self.model = PokerMLP(
                static_dim=self.static_dim, 
                numeric_dim=self.numeric_dim, 
                num_classes=len(CLASSES)
            )
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            self.model.eval()
            
            # Load scaler
            self.scaler = joblib.load(SCALER_PATH)
            if hasattr(self.scaler, 'n_features_in_'):
                self.n_scaled_cols = int(self.scaler.n_features_in_)
            else:
                if self.n_scaled_cols_meta is not None:
                    self.n_scaled_cols = int(self.n_scaled_cols_meta)
                elif self.binary_cols:
                    self.n_scaled_cols = self.numeric_dim - len(self.binary_cols)
                else:
                    self.n_scaled_cols = max(0, self.numeric_dim - 5)
            self.n_binary_cols = max(0, self.numeric_dim - self.n_scaled_cols)
            
            print(f"Successfully loaded model, scaler, and metadata.")
            print(f"Learned Temperature: {self.learned_temperature:.3f}")
            print(f"Numeric columns expected: {self.numeric_cols_order}")
            print(f"Scaled cols: {self.n_scaled_cols} | Binary cols: {self.n_binary_cols}")

        except Exception as e:
            print(f"FATAL ERROR: Could not load model artifacts: {e}")
            raise

    def reset_hand(self):
        self.hole_cards = []
        self.board_cards = []
        self.current_street = "preflop"
        self.cv_data = {}
        self.action_history = []
        self.was_pfr_live = False

    def update_state_from_cv(self, raw_cv_data: Dict[str, Any]):
        self.hole_cards = raw_cv_data.get('hole_cards', [])
        self.board_cards = raw_cv_data.get('board_cards', [])
        self.cv_data = raw_cv_data
        self.action_history = raw_cv_data.get('action_sequence', [])
        
        # Multiplayer features from JSON
        players_remaining = raw_cv_data.get('players_remaining', 2)
        self.cv_data['num_active_players'] = players_remaining
        self.cv_data['multiway_pot_flag'] = 1 if players_remaining >= 3 else 0
        
        # Relative position: 0 = first, 1 = last
        dealer_pos = raw_cv_data.get('dealer_position', 0)
        player_id = raw_cv_data.get('player_id', 0)
        if players_remaining > 1:
            seats_from_btn = (player_id - dealer_pos) % players_remaining
            self.cv_data['position_relative'] = seats_from_btn / (players_remaining - 1)
        else:
            self.cv_data['position_relative'] = 0.5
        
        # Callers and raisers on current street
        current_street_actions = [a for a in self.action_history 
                                  if a.get('street', 'preflop') == self.current_street]
        self.cv_data['num_callers_this_street'] = sum(1 for a in current_street_actions 
                                                        if a.get('action') == 'call')
        self.cv_data['num_raisers_this_street'] = sum(1 for a in current_street_actions 
                                                        if a.get('action') == 'raise')
        
        # Fallback average opponent stack
        if 'avg_opp_stack_bb' not in self.cv_data:
            self.cv_data['avg_opp_stack_bb'] = raw_cv_data.get('opp_stack_chips', 100.0)
    
    def _calculate_was_pfr(self) -> int:
        return 1 if self.was_pfr_live else 0

    def _calculate_raises_preflop(self) -> int:
        pre_hist = [a for a in self.action_history if a.get('street', 'preflop') == 'preflop']
        return sum(1 for a in pre_hist if a.get('action') == 'raise')

    def _calculate_pot_type(self) -> str:
        raises_preflop = self._calculate_raises_preflop()
        if raises_preflop == 0:
            return 'limped'
        elif raises_preflop == 1:
            return 'single_raised'
        elif raises_preflop == 2:
            return 'three_bet'
        else:
            return 'other'

    def _calculate_raises_this_street(self) -> int:
        street_hist = [a for a in self.action_history if a.get('street') == self.current_street]
        return sum(1 for a in street_hist if a.get('action') == 'raise')

    def _calculate_in_position(self) -> int:
        is_btn = (self.my_pos in ["BTN", "Late"] or self.my_pos == "SB")
        
        if self.current_street == 'preflop':
            return 0 if is_btn else 1
        else:
            return 1 if is_btn else 0

    def _calculate_board_texture(self) -> str:
        feats = evaluate_hand_features(self.hole_cards, self.board_cards)
        has_draw = feats["has_flush_draw"] or feats["has_straight_draw"]
        if has_draw or feats["hand_bucket"] >= 2.0:
            return "wet"
        return "dry"

    def get_static_vector(self) -> torch.Tensor:
        # One-hot encode cards, street, and position
        h1 = self.hole_cards[0] if len(self.hole_cards) > 0 else ""
        h2 = self.hole_cards[1] if len(self.hole_cards) > 1 else ""
        
        f1 = self.board_cards[0] if len(self.board_cards) > 0 else ""
        f2 = self.board_cards[1] if len(self.board_cards) > 1 else ""
        f3 = self.board_cards[2] if len(self.board_cards) > 2 else ""
        t = self.board_cards[3] if len(self.board_cards) > 3 else ""
        r = self.board_cards[4] if len(self.board_cards) > 4 else ""

        cards_vec = np.concatenate([
            card_one_hot(h1), card_one_hot(h2),
            card_one_hot(f1), card_one_hot(f2), card_one_hot(f3),
            card_one_hot(t), card_one_hot(r)
        ])
        
        street_vec = one_hot(self.current_street, STREET_VOCAB)
        pos_vec = one_hot(self.my_pos, POS_VOCAB)
        
        static_features = np.concatenate([cards_vec, street_vec, pos_vec])
        return torch.tensor(static_features, dtype=torch.float32).unsqueeze(0)

    def get_numeric_vector(self) -> np.ndarray:
        # Raw CV values
        pot_chips = self.cv_data.get('pot_chips', 0.0)
        to_call_chips = self.cv_data.get('to_call_chips', 0.0)
        stack_chips = self.cv_data.get('my_stack_chips', 0.0)
        opp_stack_chips = self.cv_data.get('opp_stack_chips', 0.0)
        last_bet_chips = self.cv_data.get('last_bet_size_chips', 0.0)

        # Base money features (in big blinds)
        pot_bb = pot_chips / self.bb_size
        to_call_bb = to_call_chips / self.bb_size
        stack_bb = stack_chips / self.bb_size
        opp_stack_bb = opp_stack_chips / self.bb_size
        last_bet_bb = last_bet_chips / self.bb_size

        # Bet sizing as fraction of pot
        pot_before_bet = pot_chips - last_bet_chips
        pot_bb_before = pot_before_bet / self.bb_size if self.bb_size > 0 else 0.0
        raise_size_bb = last_bet_bb if to_call_bb > 1e-9 else max(0.0, last_bet_bb - to_call_bb)
        bet_frac_of_pot = raise_size_bb / pot_bb_before if pot_bb_before > 1e-9 else 0.0
        
        # Position and game state features
        street_index = STREET_VOCAB.index(self.current_street)
        in_position = self._calculate_in_position()
        was_pfr = self._calculate_was_pfr()
        raises_this_street = self._calculate_raises_this_street()

        # Board texture using Treys
        board_texture_str = self._calculate_board_texture()
        board_texture_dry = 1.0 if board_texture_str == "dry" else 0.0
        board_texture_wet = 1.0 if board_texture_str == "wet" else 0.0

        # Treys-based hand strength features
        treys_feats = evaluate_hand_features(self.hole_cards, self.board_cards)
        hand_bucket = treys_feats["hand_bucket"]
        has_flush_draw = treys_feats["has_flush_draw"]
        has_straight_draw = treys_feats["has_straight_draw"]
        has_combo_draw = treys_feats["has_combo_draw"]
        is_missed_draw_river = treys_feats["is_missed_draw_river"]
        
        # Pot type flags
        pot_type_str = self._calculate_pot_type()
        pot_type_limped = 1.0 if pot_type_str == "limped" else 0.0
        pot_type_single_raised = 1.0 if pot_type_str == "single_raised" else 0.0
        pot_type_three_bet = 1.0 if pot_type_str == "three_bet" else 0.0
        
        # Multi-player features from JSON
        num_active_players = self.cv_data.get('num_active_players', 2)
        num_players_to_act = 0  # Not available in current JSON format, default to 0
        multiway_pot_flag = self.cv_data.get('multiway_pot_flag', 0)
        position_relative = self.cv_data.get('position_relative', 0.5)
        num_callers_this_street = self.cv_data.get('num_callers_this_street', 0)
        num_raisers_this_street = self.cv_data.get('num_raisers_this_street', 0)
        avg_opp_stack_bb = self.cv_data.get('avg_opp_stack_bb', opp_stack_bb)
        
        # Assemble feature map for model
        feature_map = {
            # Core money features (6)
            "pot_bb": pot_bb,
            "to_call_bb": to_call_bb,
            "stack_bb": stack_bb,
            "opp_stack_bb": opp_stack_bb,
            "raise_to_bb": last_bet_bb,
            "bet_frac_of_pot": bet_frac_of_pot,
            # Position & context (4)
            "in_position": in_position,
            "was_pfr": was_pfr,
            "street_index": street_index,
            "raises_this_street": raises_this_street,
            # Categorical one-hots (5)
            "board_texture_dry": board_texture_dry,
            "board_texture_wet": board_texture_wet,
            "pot_type_limped": pot_type_limped,
            "pot_type_single_raised": pot_type_single_raised,
            "pot_type_three_bet": pot_type_three_bet,
            # Treys features (5)
            "hand_bucket": hand_bucket,
            "has_flush_draw": has_flush_draw,
            "has_straight_draw": has_straight_draw,
            "has_combo_draw": has_combo_draw,
            "is_missed_draw_river": is_missed_draw_river,
            # Multi-player features (7)
            "num_active_players": num_active_players,
            "num_players_to_act": num_players_to_act,
            "multiway_pot_flag": multiway_pot_flag,
            "position_relative": position_relative,
            "num_callers_this_street": num_callers_this_street,
            "num_raisers_this_street": num_raisers_this_street,
            "avg_opp_stack_bb": avg_opp_stack_bb,
        }
        
        numeric_feature_vector = []
        for col_name in self.numeric_cols_order:
            numeric_feature_vector.append(feature_map.get(col_name, 0.0))

        return np.array(numeric_feature_vector, dtype=np.float32).reshape(1, -1)

    def predict_action(self) -> Tuple[str, Dict[str, float]]:
        """Get model prediction with Treys-based aggression adjustments"""
        print(f"--- Predicting for {self.my_pos} on {self.current_street} ---")
        
        # Get features
        numeric_vec_unscaled = self.get_numeric_vector()
        
        # DEBUG: Show key features
        try:
            idx_bucket = self.numeric_cols_order.index('hand_bucket')
            idx_pot = self.numeric_cols_order.index('pot_bb')
            idx_to_call = self.numeric_cols_order.index('to_call_bb')
            idx_stack = self.numeric_cols_order.index('stack_bb')
            print(f"DEBUG: pot_bb={numeric_vec_unscaled[0, idx_pot]:.1f}, to_call={numeric_vec_unscaled[0, idx_to_call]:.1f}, stack={numeric_vec_unscaled[0, idx_stack]:.1f}, hand_bucket={numeric_vec_unscaled[0, idx_bucket]:.2f}")
        except:
            pass
        
        # Apply scaler
        try:
            scaled_part = numeric_vec_unscaled[:, :self.n_scaled_cols]
            binary_part = numeric_vec_unscaled[:, self.n_scaled_cols:]
            scaled_part_transformed = self.scaler.transform(scaled_part)
            numeric_vec_scaled = np.concatenate([scaled_part_transformed, binary_part], axis=1)
        except Exception as e:
            print(f"Error applying scaler: {e}")
            return "ERROR", {}

        static_vec = self.get_static_vector()
        numeric_tensor = torch.tensor(numeric_vec_scaled, dtype=torch.float32)
        
        # Forward pass through model
        with torch.no_grad():
            logits = self.model(static_vec, numeric_tensor)
        
        # Determine legal actions and create action mask
        try:
            idx_to_call = self.numeric_cols_order.index('to_call_bb')
            to_call_bb_val = float(numeric_vec_unscaled[0, idx_to_call])
        except Exception:
            to_call_bb_val = 0.0
        
        # Mask illegal actions before temperature scaling
        facing_bet = to_call_bb_val > 0.01
        
        # Create mask for legal moves [fold, check, call, raise]
        action_mask = torch.ones(4, dtype=torch.bool)
        if facing_bet:
            # Facing bet: disable check
            action_mask[CLASSES.index('check')] = False
        else:
            # No bet: disable fold and call
            action_mask[CLASSES.index('fold')] = False
            action_mask[CLASSES.index('call')] = False
        
        # Apply mask by heavily penalizing illegal logits
        logits_masked = logits.clone()
        # Use -100 instead of -1e9 to avoid numerical instability
        logits_masked[0, ~action_mask] = -100.0
        
        # Allowed actions list from mask
        if facing_bet:
            allowed_actions = ['fold', 'call', 'raise']
        else:
            allowed_actions = ['check', 'raise']
        
        # Logit biases after masking
        call_idx = CLASSES.index("call")
        logits_masked[0, call_idx] = logits_masked[0, call_idx] + Cfg.call_logit_bias
        
        if hasattr(self, 'class_logit_bias') and isinstance(self.class_logit_bias, dict):
            for cls_name, bias in self.class_logit_bias.items():
                if cls_name in CLASSES and isinstance(bias, (int, float)):
                    idx = CLASSES.index(cls_name)
                    logits_masked[0, idx] = logits_masked[0, idx] + float(bias)

        # Temperature scaling and softmax
        probs = torch.softmax(logits_masked / self.learned_temperature, dim=1).squeeze()

        # Live Treys features for debug and heuristics
        treys_feats_live = evaluate_hand_features(self.hole_cards, self.board_cards)
        hand_bucket = treys_feats_live["hand_bucket"]
        has_flush_draw = treys_feats_live["has_flush_draw"]
        has_straight_draw = treys_feats_live["has_straight_draw"]
        is_missed_draw_river = treys_feats_live.get("is_missed_draw_river", 0.0)
        board_has_pair = treys_feats_live.get("board_has_pair", 0.0)
        board_is_monotone = treys_feats_live.get("board_is_monotone", 0.0)
        board_is_connected = treys_feats_live.get("board_is_connected", 0.0)
        
        # DEBUG: Hand evaluation
        print(f"DEBUG: Hand={self.hole_cards}, Board={self.board_cards}, Bucket={hand_bucket:.2f}, FD={has_flush_draw}, SD={has_straight_draw}")
        if board_has_pair > 0.5 or board_is_monotone > 0.5 or board_is_connected > 0.5:
            print(f"DANGEROUS BOARD: Paired={board_has_pair}, Monotone={board_is_monotone}, Connected={board_is_connected}")

        # === Active heuristics ===
        probs_adjusted = probs.clone()
        
        # Street and pot info for heuristics
        try:
            idx_street = self.numeric_cols_order.index('street_index')
            idx_pot = self.numeric_cols_order.index('pot_bb')
            idx_to_call = self.numeric_cols_order.index('to_call_bb')
            idx_raises = self.numeric_cols_order.index('raises_this_street')
            idx_num_raisers = self.numeric_cols_order.index('num_raisers_this_street')
            idx_multiway = self.numeric_cols_order.index('multiway_pot_flag')
            street_idx = int(numeric_vec_unscaled[0, idx_street])
            pot_bb_val = float(numeric_vec_unscaled[0, idx_pot])
            to_call_bb_val = float(numeric_vec_unscaled[0, idx_to_call])
            raises_this_street = float(numeric_vec_unscaled[0, idx_raises])
            num_raisers_this_street = float(numeric_vec_unscaled[0, idx_num_raisers])
            multiway_pot_flag = float(numeric_vec_unscaled[0, idx_multiway])
        except (ValueError, IndexError):
            street_idx = 0
            pot_bb_val = 0.0
            to_call_bb_val = 0.0
            raises_this_street = 0.0
            num_raisers_this_street = 0.0
            multiway_pot_flag = 0.0

        # PRE-FLOP AGGRESSION, RANGE & SHORT-STACK LOGIC (STRONG OVERRIDES)
        if street_idx == 0:
            strong_preflop = hand_bucket >= 3.0
            premium_preflop = hand_bucket >= 3.5
            # Buckets 2.0–3.0: good-but-not-nutty value hands (pairs, big broadways)
            value_preflop = 2.0 <= hand_bucket < 3.0
            # Buckets ≥1.0: playable speculative hands (suited connectors/gappers, weak Ax/Kx)
            playable_spec = hand_bucket >= 1.0

            short_stack = False
            num_active_players = self.cv_data.get('num_active_players', 2)
            multiway_pot_flag = self.cv_data.get('multiway_pot_flag', 0)
            try:
                idx_stack = self.numeric_cols_order.index('stack_bb')
                stack_bb_val = float(numeric_vec_unscaled[0, idx_stack])
                short_stack = stack_bb_val <= 10.0
            except (ValueError, IndexError):
                stack_bb_val = 0.0

            # Treat blinds as "facing a small bet" preflop; still allow strong opens
            # FORCE-RAISE with strong/value hands when cost is small (<= 2BB)
            if to_call_bb_val <= 2.0 and (strong_preflop or value_preflop):
                p_fold = probs_adjusted[CLASSES.index('fold')]
                p_call = probs_adjusted[CLASSES.index('call')]
                p_check = probs_adjusted[CLASSES.index('check')]
                # Collapse most non-raise mass into raise
                new_raise = 0.75
                probs_adjusted[CLASSES.index('raise')] = new_raise
                # Leftover distributed between fold/call/check but small
                leftover = 1.0 - new_raise
                probs_adjusted[CLASSES.index('call')] = min(p_call, leftover * 0.5)
                probs_adjusted[CLASSES.index('check')] = min(p_check, leftover * 0.3)
                probs_adjusted[CLASSES.index('fold')] = max(0.0, 1.0 - probs_adjusted.sum())

            # Late-position / blinds widen range aggressively for playable speculative hands
            if self.my_pos in ("Late", "Blinds") and playable_spec and to_call_bb_val <= 1.5:
                # Cap fold probability so we don't just muck everything reasonable
                max_fold = 0.25
                cur_fold = probs_adjusted[CLASSES.index('fold')].item()
                if cur_fold > max_fold:
                    reduce = cur_fold - max_fold
                    probs_adjusted[CLASSES.index('fold')] = max_fold
                    probs_adjusted[CLASSES.index('call')] += reduce * 0.8
                    probs_adjusted[CLASSES.index('raise')] += reduce * 0.2

                # In heads-up/single-raised pots, give these hands real raising weight
                if num_active_players <= 3 and multiway_pot_flag == 0 and to_call_bb_val <= 1.0:
                    p_call = probs_adjusted[CLASSES.index('call')]
                    # Shift half of call into raise when we actually like the hand
                    shift = p_call * 0.5
                    probs_adjusted[CLASSES.index('raise')] += shift
                    probs_adjusted[CLASSES.index('call')] = p_call - shift

            # Short-stack shove/fold bias
            if short_stack:
                # Ultra short-stack zone (true push/fold): be willing to jam
                # a lot of hands instead of nit-folding.
                if stack_bb_val <= 4.0 and to_call_bb_val > 0:
                    if hand_bucket >= 1.0:
                        # Treat as shove/fold: push most non-fold mass into raise.
                        p_fold = probs_adjusted[CLASSES.index('fold')].item()
                        p_call = probs_adjusted[CLASSES.index('call')].item()
                        p_check = probs_adjusted[CLASSES.index('check')].item()
                        total_non_fold = p_call + p_check
                        if total_non_fold > 0.0:
                            shift = total_non_fold * 0.75
                            probs_adjusted[CLASSES.index('raise')] += shift
                            probs_adjusted[CLASSES.index('call')] = p_call * 0.25
                            probs_adjusted[CLASSES.index('check')] = p_check * 0.25
                    else:
                        # True trash in all-in zone: still allow folding, but
                        # don't over-kill speculative calls.
                        if hand_bucket < 0.5 and to_call_bb_val > 0.5:
                            p_call = probs_adjusted[CLASSES.index('call')].item()
                            if p_call > 0.10:
                                shift = p_call * 0.60
                                probs_adjusted[CLASSES.index('fold')] += shift
                                probs_adjusted[CLASSES.index('call')] = p_call - shift
                else:
                    # Non-ultra-short stacks: keep original behavior but only
                    # punish true trash, not all weak hands.
                    if strong_preflop and to_call_bb_val <= stack_bb_val:
                        p_call = probs_adjusted[CLASSES.index('call')].item()
                        p_check = probs_adjusted[CLASSES.index('check')].item()
                        boost = 0.80 * (p_call + p_check)
                        probs_adjusted[CLASSES.index('raise')] += boost
                        probs_adjusted[CLASSES.index('call')] = p_call * 0.20
                        probs_adjusted[CLASSES.index('check')] = p_check * 0.20
                    elif hand_bucket < 0.5 and to_call_bb_val > 0.5:
                        p_call = probs_adjusted[CLASSES.index('call')].item()
                        if p_call > 0.05:
                            shift = p_call * 0.85
                            probs_adjusted[CLASSES.index('fold')] += shift
                            probs_adjusted[CLASSES.index('call')] = p_call - shift
        
        # Pocket pair defense: don't fold to single blind
        if street_idx == 0 and len(self.hole_cards) == 2 and to_call_bb_val <= 1.5:
            ranks_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
            try:
                card1, card2 = self.hole_cards[0], self.hole_cards[1]
                rank1 = ranks_map.get(card1[1] if card1[0] in 'CDHS' else card1[0], 0)
                rank2 = ranks_map.get(card2[1] if card2[0] in 'CDHS' else card2[0], 0)
                
                is_pocket_pair = (rank1 == rank2)
                
                if is_pocket_pair:
                    p_fold = probs_adjusted[CLASSES.index('fold')].item()
                    if p_fold > 0.30:
                        # Shift fold to call for pocket pairs
                        shift = p_fold * 0.80
                        probs_adjusted[CLASSES.index('call')] += shift
                        probs_adjusted[CLASSES.index('fold')] = p_fold * 0.20
                        print(f"POCKET PAIR DEFENSE: Calling with {card1}{card2} (never fold pairs to single blind)")
            except (IndexError, KeyError):
                pass
        
        # Bluff control: kill bluffs only with pure air
        if to_call_bb_val <= 1e-6 and street_idx >= 1:  # First to act postflop
            has_draw = bool(treys_feats_live.get("has_flush_draw", 0.0) or treys_feats_live.get("has_straight_draw", 0.0))
            if hand_bucket <= 0.0 and not has_draw:  # Complete air, no equity
                p_raise = probs_adjusted[CLASSES.index('raise')].item()
                if p_raise > 0.30:
                    shift = p_raise * 0.70
                    probs_adjusted[CLASSES.index('check')] += shift
                    probs_adjusted[CLASSES.index('raise')] = p_raise * 0.30
                    print(f"BLUFF CONTROL: Checking with pure air (bucket={hand_bucket:.1f})")
        
        # Value extraction: bet/raise strong postflop hands
        if street_idx >= 1 and hand_bucket >= 3.0:  # Postflop with strong+ hand
            if to_call_bb_val <= 1e-6:  # First to act - bet instead of checking
                p_check = probs_adjusted[CLASSES.index('check')].item()
                if p_check > 0.50:  # Checking too much
                    shift = p_check * 0.60  # Shift 60% of checks to raises
                    probs_adjusted[CLASSES.index('raise')] += shift
                    probs_adjusted[CLASSES.index('check')] = p_check * 0.40
                    print(f"VALUE BET: Strong hand (bucket={hand_bucket:.1f}) - betting")
            else:  # Facing a bet - raise for value
                p_call = probs_adjusted[CLASSES.index('call')].item()
                p_raise = probs_adjusted[CLASSES.index('raise')].item()
                # If mostly calling/checking, shift to raising
                if p_call > 0.40 and p_raise < 0.30:  # Passive with strong hand
                    shift = p_call * 0.50  # Shift 50% of calls to raises
                    probs_adjusted[CLASSES.index('raise')] += shift
                    probs_adjusted[CLASSES.index('call')] = p_call * 0.50
                    print(f"VALUE RAISE: Strong hand (bucket={hand_bucket:.1f}) - raising for value")
        
        # River protection: avoid hero calls with marginal hands
        if street_idx == 3 and to_call_bb_val > 0:  # River facing bet
            if hand_bucket < 2.0:  # Marginal or worse
                try:
                    idx_bet_frac = self.numeric_cols_order.index('bet_frac_of_pot')
                    bet_frac = float(numeric_vec_unscaled[0, idx_bet_frac])
                    
                    if bet_frac > 0.50:  # Facing big bet
                        p_call = probs_adjusted[CLASSES.index('call')].item()
                        p_raise = probs_adjusted[CLASSES.index('raise')].item()
                        if p_call > 0.20 or p_raise > 0.05:
                            # Shift call/raise to fold
                            shift = (p_call * 0.80) + (p_raise * 0.95)
                            probs_adjusted[CLASSES.index('fold')] += shift
                            probs_adjusted[CLASSES.index('call')] = p_call * 0.20
                            probs_adjusted[CLASSES.index('raise')] = p_raise * 0.05
                            print(f"RIVER PROTECTION: Folding marginal hand (bucket={hand_bucket:.1f})")
                except (ValueError, IndexError):
                    pass

        # Flop/turn discipline vs big bets and scary boards
        if street_idx in (1, 2) and to_call_bb_val > 0:
            try:
                idx_bet_frac = self.numeric_cols_order.index('bet_frac_of_pot')
                bet_frac = float(numeric_vec_unscaled[0, idx_bet_frac])
            except (ValueError, IndexError):
                bet_frac = 0.0

            has_draw = bool(treys_feats_live.get("has_flush_draw", 0.0) or treys_feats_live.get("has_straight_draw", 0.0))
            weak_hand = hand_bucket < 2.0
            # Treat anything below ~3.5 as non-nut for pot-control on scary boards
            medium_hand = hand_bucket < 3.5
            dangerous_board = (board_has_pair > 0.5) or (board_is_monotone > 0.5) or (board_is_connected > 0.5)

            # Weak no-draw hands vs medium+ bets: fold earlier
            if weak_hand and not has_draw and bet_frac >= 0.5:
                p_call = probs_adjusted[CLASSES.index('call')].item()
                p_raise = probs_adjusted[CLASSES.index('raise')].item()
                if p_call + p_raise > 0.05:
                    shift = 0.85 * (p_call + p_raise)
                    probs_adjusted[CLASSES.index('fold')] += shift
                    probs_adjusted[CLASSES.index('call')] = p_call * 0.15
                    probs_adjusted[CLASSES.index('raise')] = p_raise * 0.15

            # Medium hands on scary boards: call/fold, avoid raising
            if medium_hand and dangerous_board and bet_frac >= 0.5:
                p_raise = probs_adjusted[CLASSES.index('raise')].item()
                p_call = probs_adjusted[CLASSES.index('call')].item()
                if p_raise > 0.01:
                    # No raising here: all raise mass goes to call/fold
                    shift_to_call = p_raise * (0.60 if bet_frac < 0.9 else 0.40)
                    shift_to_fold = p_raise - shift_to_call
                    probs_adjusted[CLASSES.index('call')] += shift_to_call
                    probs_adjusted[CLASSES.index('fold')] += shift_to_fold
                    probs_adjusted[CLASSES.index('raise')] = 0.0

        # Protect strong draws when price is reasonable
        if street_idx in (1, 2) and to_call_bb_val > 0:
            has_flush_draw_live = treys_feats_live.get("has_flush_draw", 0.0) > 0.5
            has_straight_draw_live = treys_feats_live.get("has_straight_draw", 0.0) > 0.5
            if (has_flush_draw_live or has_straight_draw_live) and hand_bucket >= 0.5:
                try:
                    idx_bet_frac = self.numeric_cols_order.index('bet_frac_of_pot')
                    bet_frac = float(numeric_vec_unscaled[0, idx_bet_frac])
                except (ValueError, IndexError):
                    bet_frac = 0.0

                # Reasonable price: cap fold, favour call/raise
                if bet_frac <= 0.7:
                    p_fold = probs_adjusted[CLASSES.index('fold')].item()
                    if p_fold > 0.25:
                        shift = p_fold - 0.25
                        probs_adjusted[CLASSES.index('call')] += shift * 0.8
                        probs_adjusted[CLASSES.index('raise')] += shift * 0.2
                        probs_adjusted[CLASSES.index('fold')] = 0.25

        # Protect made flushes vs normal bets
        board_is_monotone = treys_feats_live.get("board_is_monotone", 0.0)
        made_flush_candidate = street_idx >= 1 and hand_bucket >= 3.0 and board_is_monotone > 0.5
        if made_flush_candidate and to_call_bb_val > 0:
            try:
                idx_bet_frac = self.numeric_cols_order.index('bet_frac_of_pot')
                bet_frac = float(numeric_vec_unscaled[0, idx_bet_frac])
            except (ValueError, IndexError):
                bet_frac = 0.0
            if bet_frac <= 1.0:
                p_fold = probs_adjusted[CLASSES.index('fold')].item()
                if p_fold > 0.15:
                    shift = p_fold - 0.15
                    # Prefer calling over folding when holding strong flush-type hand
                    probs_adjusted[CLASSES.index('call')] += shift
                    probs_adjusted[CLASSES.index('fold')] = 0.15
        
        # Use adjusted postflop/river probabilities
        probs = probs_adjusted

        # Preflop limp leniency for small-completion bucket-1 hands
        try:
            idx_street_pf = self.numeric_cols_order.index('street_index')
            idx_to_call_pf = self.numeric_cols_order.index('to_call_bb')
            street_idx_pf = int(numeric_vec_unscaled[0, idx_street_pf])
            to_call_bb_pf = float(numeric_vec_unscaled[0, idx_to_call_pf])
        except (ValueError, IndexError):
            street_idx_pf = 0
            to_call_bb_pf = 0.0

        if street_idx_pf == 0 and 1.0 <= hand_bucket < 2.0 and facing_bet:
            # Facing a small preflop completion (anything up to 1BB)
            if to_call_bb_pf <= 1.1:
                p_fold = probs[CLASSES.index('fold')].item()
                p_call = probs[CLASSES.index('call')].item()
                # If there is some call weight, force a strong preference for call
                if p_call >= 0.20 and p_fold > 0.0:
                    shift = p_fold * 0.90
                    probs[CLASSES.index('call')] += shift
                    probs[CLASSES.index('fold')] = p_fold * 0.10
                    print(
                        f"PREFLOP LIMP LENIENCY: bucket={hand_bucket:.1f}, "
                        f"small to_call={to_call_bb_pf:.2f}, forcing call "
                        f"(p_fold={p_fold:.2f}, p_call={p_call:.2f})"
                    )

        # Raise-war control: clamp further aggression after multiple raises
        try:
            idx_bet_frac = self.numeric_cols_order.index('bet_frac_of_pot')
            bet_frac = float(numeric_vec_unscaled[0, idx_bet_frac])
        except (ValueError, IndexError):
            bet_frac = 0.0

        dangerous_board_live = (board_has_pair > 0.5) or (board_is_monotone > 0.5) or (board_is_connected > 0.5)
        weak_or_medium = hand_bucket < 3.5

        # Wet-board stab after flop check-through (turn/river, no bet)
        if street_idx in (2, 3) and to_call_bb_val <= 0.01:
            flop_actions = [a for a in self.action_history if a.get('street') == 'flop']
            flop_had_bet_or_raise = any(a.get('action') in ('bet', 'raise') for a in flop_actions)
            flop_all_checks_or_folds = all(a.get('action') in ('check', 'fold') for a in flop_actions) if flop_actions else False

            wet_scary_board = (board_has_pair > 0.5) or (board_is_monotone > 0.5) or (board_is_connected > 0.5)

            if wet_scary_board and not flop_had_bet_or_raise and flop_all_checks_or_folds:
                p_check = probs_adjusted[CLASSES.index('check')].item()
                p_raise = probs_adjusted[CLASSES.index('raise')].item()
                if p_check > 0.20:
                    shift = p_check * 0.40
                    probs_adjusted[CLASSES.index('raise')] = p_raise + shift
                    probs_adjusted[CLASSES.index('check')] = p_check - shift
                    print("WET-BOARD STAB: turn/river bluff after flop check-through on scary board")

        if street_idx >= 1 and to_call_bb_val > 0 and raises_this_street >= 2:
            # Two or more raises already this street: avoid raise wars.
            # For non-nut hands on scary or big pots, convert most raise mass
            # into calls or folds.
            p_raise = probs_adjusted[CLASSES.index('raise')].item()
            if p_raise > 0.0:
                p_call = probs_adjusted[CLASSES.index('call')].item()
                p_fold = probs_adjusted[CLASSES.index('fold')].item()

                big_pot = pot_bb_val >= 25.0
                super_big_pot = pot_bb_val >= 40.0

                if weak_or_medium and (dangerous_board_live or big_pot or multiway_pot_flag > 0.5):
                    # Scary/huge pots with non-nuts: no more raising
                    shift_to_call = p_raise * 0.60
                    shift_to_fold = p_raise * 0.40
                    probs_adjusted[CLASSES.index('call')] = p_call + shift_to_call
                    probs_adjusted[CLASSES.index('fold')] = p_fold + shift_to_fold
                    probs_adjusted[CLASSES.index('raise')] = 0.0
                elif hand_bucket >= 3.5 and (dangerous_board_live or super_big_pot):
                    # Even strong hands: call more on nightmare/huge pots
                    shift_to_call = p_raise * 0.70
                    shift_to_fold = p_raise * 0.30 if bet_frac >= 1.2 else 0.10
                    probs_adjusted[CLASSES.index('call')] = p_call + shift_to_call
                    probs_adjusted[CLASSES.index('fold')] = p_fold + shift_to_fold
                    probs_adjusted[CLASSES.index('raise')] = 0.0

        # Extra cap on river raises in big pots
        if street_idx == 3 and to_call_bb_val > 0 and pot_bb_val >= 35.0:
            p_raise = probs_adjusted[CLASSES.index('raise')].item()
            if p_raise > 0.10:
                # Convert most raise mass to call with non-nuts on scary boards
                if hand_bucket < 3.5 or dangerous_board_live:
                    shift = p_raise * 0.85
                    probs_adjusted[CLASSES.index('call')] += shift
                    probs_adjusted[CLASSES.index('raise')] = p_raise * 0.15

        # River bluff suppression: avoid spewing in no-bet spots
        if street_idx == 3 and to_call_bb_val <= 0.01:
            p_check = probs_adjusted[CLASSES.index('check')].item()
            p_raise = probs_adjusted[CLASSES.index('raise')].item()
            if p_raise > 0.0:
                nightmare_board = dangerous_board_live or bet_frac >= 0.75

                if hand_bucket < 2.0 and nightmare_board:
                    # Weak/marginal on nightmare boards: almost never bluff
                    shift = p_raise * 0.90
                    probs_adjusted[CLASSES.index('check')] = p_check + shift
                    probs_adjusted[CLASSES.index('raise')] = p_raise * 0.10
                elif hand_bucket < 1.0:
                    # Pure trash on river: strongly prefer check
                    shift = p_raise * 0.80
                    probs_adjusted[CLASSES.index('check')] = p_check + shift
                    probs_adjusted[CLASSES.index('raise')] = p_raise * 0.20

        # Missed river draws: fold/call only, never raise
        if street_idx == 3 and to_call_bb_val > 0 and is_missed_draw_river:
            p_raise = probs_adjusted[CLASSES.index('raise')].item()
            p_call = probs_adjusted[CLASSES.index('call')].item()
            p_fold = probs_adjusted[CLASSES.index('fold')].item()

            if p_raise > 0.0:
                # Remove raise; split into fold/call with fold preference.
                shift_to_fold = p_raise * 0.70
                shift_to_call = p_raise * 0.30
                probs_adjusted[CLASSES.index('fold')] = p_fold + shift_to_fold
                probs_adjusted[CLASSES.index('call')] = p_call + shift_to_call
                probs_adjusted[CLASSES.index('raise')] = 0.0

        # TREYS-based dynamic aggression block removed (was disabled)
        
        # Emergency override: true garbage vs real bets folds more
        try:
            hand_bucket = treys_feats_live["hand_bucket"]
            idx_to_call = self.numeric_cols_order.index('to_call_bb')
            to_call_val = float(numeric_vec_unscaled[0, idx_to_call])
            idx_pot_em = self.numeric_cols_order.index('pot_bb')
            pot_bb_em = float(numeric_vec_unscaled[0, idx_pot_em])
            idx_street_em = self.numeric_cols_order.index('street_index')
            street_idx_em = int(numeric_vec_unscaled[0, idx_street_em])

            # Overcards: any card T or higher in our hand
            ranks_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                         'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
            hero_ranks = []
            for c in self.hole_cards:
                r = c[1] if c and c[0] in 'CDHS' and len(c) >= 2 else (c[0] if c else '')
                hero_ranks.append(ranks_map.get(r, 0))
            has_overcard = any(r >= 10 for r in hero_ranks)

            has_fd = treys_feats_live.get("has_flush_draw", 0.0) > 0.5
            has_sd = treys_feats_live.get("has_straight_draw", 0.0) > 0.5

            facing_real_bet = to_call_val > 0.0
            if hand_bucket <= 0.0 and facing_real_bet and not has_fd and not has_sd and not has_overcard and to_call_val > 0.5:
                print(
                    f"OVERRIDE: Pure garbage (bucket={hand_bucket:.2f}) vs bet "
                    f"(to_call={to_call_val:.2f}BB, pot={pot_bb_em:.2f}BB, street={street_idx_em}) - fold bias"
                )
                probs_override = probs.clone()
                probs_override[CLASSES.index('fold')] = 0.85
                probs_override[CLASSES.index('call')] = 0.10
                probs_override[CLASSES.index('raise')] = 0.05
                total = sum(probs_override[CLASSES.index(a)].item() for a in allowed_actions)
                if total > 1e-9:
                    for a in allowed_actions:
                        probs_override[CLASSES.index(a)] /= total
                probs = probs_override

            # Turn weak-no-draw safety: bias hard toward folding
            if street_idx_em == 2 and facing_real_bet:
                has_flush_draw_live = treys_feats_live.get("has_flush_draw", 0.0) > 0.5
                has_straight_draw_live = treys_feats_live.get("has_straight_draw", 0.0) > 0.5
                if hand_bucket <= 1.5 and (not has_flush_draw_live) and (not has_straight_draw_live) and to_call_val >= 0.5:
                    print(
                        f"OVERRIDE: Turn weak no-draw hand (bucket={hand_bucket:.2f}) "
                        f"vs bet (to_call={to_call_val:.2f}BB, pot={pot_bb_em:.2f}BB) - biasing to fold"
                    )
                    probs_override = probs.clone()
                    p_fold = probs_override[CLASSES.index('fold')]
                    p_call = probs_override[CLASSES.index('call')]
                    p_raise = probs_override[CLASSES.index('raise')]
                    extra_to_fold = 0.7 * (p_call + p_raise)
                    probs_override[CLASSES.index('fold')] = min(1.0, p_fold + extra_to_fold)
                    probs_override[CLASSES.index('call')] = max(0.0, 0.3 * p_call)
                    probs_override[CLASSES.index('raise')] = max(0.0, 0.3 * p_raise)
                    total = sum(probs_override[CLASSES.index(a)].item() for a in allowed_actions)
                    if total > 1e-9:
                        for a in allowed_actions:
                            probs_override[CLASSES.index(a)] /= total
                    probs = probs_override
        except Exception as e:
            print(f"DEBUG: Override exception ({e})")
        
        # Final sanity clamp: tiny raise never beats near-certain passive
        try:
            idx_check = CLASSES.index('check')
            idx_raise = CLASSES.index('raise')
            idx_call = CLASSES.index('call')
            # No-bet: if check crushes raise, force check
            if to_call_bb_val <= 0.01 and 'check' in allowed_actions:
                p_check = probs[idx_check].item()
                p_raise = probs[idx_raise].item()
                if p_check >= 0.95 and p_raise <= 0.05:
                    action_name = 'check'
                else:
                    action_name = None
            # Facing bet: if call crushes raise, force call
            elif to_call_bb_val > 0.01 and 'call' in allowed_actions:
                p_call = probs[idx_call].item()
                p_raise = probs[idx_raise].item()
                if p_call >= 0.95 and p_raise <= 0.05:
                    action_name = 'call'
                else:
                    action_name = None
            else:
                action_name = None
        except Exception:
            action_name = None

        # Choose best action if sanity clamp didn't already decide
        if not action_name:
            best_name = None
            best_prob = -1.0
            for name in allowed_actions:
                idx = CLASSES.index(name)
                val = probs[idx].item()
                if val > best_prob:
                    best_prob = val
                    best_name = name
            action_name = best_name or CLASSES[int(torch.argmax(probs).item())]

        # Remap illegal 'check' when facing bet
        if to_call_bb_val > 1e-6 and action_name == 'check':
            p_call = probs[CLASSES.index('call')].item()
            p_fold = probs[CLASSES.index('fold')].item()
            action_name = 'call' if p_call >= p_fold else 'fold'
            print(f"DEBUG: Remapped check to {action_name}")

        prob_dict = {name: f"{p.item():.4f}" for name, p in zip(CLASSES, probs)}
        print(f"Probs: {prob_dict} ==> {action_name}")
        
        return action_name, prob_dict


# Global tracker instance
_tracker = None
_bb_normalizer = None

def main(cv_json: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point for orchestrator integration"""
    global _tracker, _bb_normalizer
    
    player_id = cv_json.get('player_id', 1)
    dealer_position = cv_json.get('dealer_position', 0)
    players_remaining = cv_json.get('players_remaining', 2)
    
    # Calculate correct position based on dealer button and player count
    position = _calculate_position(player_id, dealer_position, players_remaining)
    
    # Get or calculate BB size from JSON
    bb_size = float(cv_json.get('big_blind', 10.0))  # Default BB = 10 chips
    
    if _tracker is None:
        _tracker = LiveHandTracker(my_position=position, big_blind_size=bb_size)
    else:
        # Update position if it changed (e.g., different seat in new hand)
        _tracker.my_pos = position
        _tracker.bb_size = bb_size
    
    # Extract game state
    street = cv_json.get('round', 'preflop').lower()
    
    board_cards = []
    for card_key in ['flop1', 'flop2', 'flop3', 'turn', 'river']:
        card = cv_json.get(card_key, '').strip()
        if card:
            board_cards.append(card)
    
    hole_cards = []
    for card_key in ['hole1', 'hole2']:
        card = cv_json.get(card_key, '').strip()
        if card:
            hole_cards.append(card)
    
    # JSON contains RAW chip values - normalize to BB units
    pot_chips = float(cv_json.get('pot_bb', 0.0))  # Misleading name - actually raw chips
    to_call_chips = float(cv_json.get('to_call_bb', 0.0))
    stack_chips = float(cv_json.get('stack_bb', 0.0))
    
    # DEBUG: Show raw values from JSON before normalization
    print(f"DEBUG JSON: pot_chips={pot_chips}, to_call_chips={to_call_chips}, stack_chips={stack_chips}, bb_size={bb_size}")
    
    # Normalize to BB
    pot_bb = pot_chips / bb_size
    to_call_bb = to_call_chips / bb_size
    stack_bb = stack_chips / bb_size
    
    print(f"DEBUG NORMALIZED: pot_bb={pot_bb:.2f}, to_call_bb={to_call_bb:.2f}, stack_bb={stack_bb:.2f}")
    
    # Calculate average opponent stack from all player stacks (raw chips)
    p1_stack_chips = float(cv_json.get('p1_stack_bb', 0.0))
    p2_stack_chips = float(cv_json.get('p2_stack_bb', 0.0))
    p3_stack_chips = float(cv_json.get('p3_stack_bb', 0.0))
    all_stack_chips = [s for s in [p1_stack_chips, p2_stack_chips, p3_stack_chips] if s > 0]
    avg_stack_chips = np.mean(all_stack_chips) if all_stack_chips else stack_chips
    opp_stack_bb = avg_stack_chips / bb_size
    
    tracker_data = {
        'hole_cards': hole_cards,
        'board_cards': board_cards,
        'my_stack_chips': stack_bb,  # Now in BB units
        'opp_stack_chips': opp_stack_bb,  # Now in BB units
        'pot_chips': pot_bb,  # Now in BB units
        'to_call_chips': to_call_bb,  # Now in BB units
        'last_bet_size_chips': to_call_bb,  # Now in BB units
        'action_sequence': cv_json.get('action_history', []),
        'my_player_name': 'PlayerCoach',
        # Add multiplayer features from JSON
        'players_remaining': players_remaining,
        'dealer_position': dealer_position,
        'player_id': player_id,
        'p1_stack_bb': p1_stack_chips / bb_size,  # Normalize to BB
        'p2_stack_bb': p2_stack_chips / bb_size,  # Normalize to BB
        'p3_stack_bb': p3_stack_chips / bb_size,  # Normalize to BB
        'avg_opp_stack_bb': opp_stack_bb  # Already normalized
    }
    
    _tracker.bb_size = 1.0
    _tracker.current_street = street
    _tracker.update_state_from_cv(tracker_data)
    
    action, prob_dict = _tracker.predict_action()
    
    probs_float = {k: float(v) if isinstance(v, (int, float)) else float(v) for k, v in prob_dict.items()}
    confidence = max(probs_float.values())
    
    return {
        'action': action,
        'confidence': confidence,
        'probabilities': prob_dict
    }

def reset_hand():
    global _tracker
    if _tracker:
        _tracker.reset_hand()

def _calculate_position(player_id: int, dealer_position: int, num_players: int) -> str:
    """
    Calculate relative position based on dealer button.
    
    Args:
        player_id: 0-3 (0 = Coach, 1-3 = Opponents)
        dealer_position: 0-3 (which seat is dealer)
        num_players: 2-4 (number of active players)
    
    Returns:
        Position string from POS_VOCAB: 'Early', 'Late', 'Blinds'
    """
    # Calculate seats from button (BTN = 0)
    seats_from_btn = (player_id - dealer_position) % num_players
    
    if num_players == 2:
        # Heads-up: both players in blinds
        return 'Blinds'
    elif num_players == 3:
        # 3-way: BTN (Late), SB (Blinds), BB (Blinds)
        if seats_from_btn == 0:
            return 'Late'    # BTN
        else:
            return 'Blinds'  # SB or BB
    elif num_players == 4:
        # 4-way: BTN (Late), SB (Blinds), BB (Blinds), UTG (Early)
        if seats_from_btn == 0:
            return 'Late'    # BTN
        elif seats_from_btn in [1, 2]:
            return 'Blinds'  # SB or BB
        else:
            return 'Early'   # UTG
    else:
        # Default to Blinds for safety (always valid in POS_VOCAB)
        return 'Blinds'

def new_session(position: str, big_blind: float):
    global _tracker, _bb_normalizer
    _tracker = LiveHandTracker(my_position=position, big_blind_size=big_blind)
    _bb_normalizer = None

if __name__ == "__main__":
    print("LiveHandTracker - Clean Version")
    print("=" * 80)
    
    cv_data = {
        "hand_id": 1,
        "player_id": 1,
        "round": "preflop",
        "hole1": "HA",
        "hole2": "DK",
        "stack_bb": 170,
        "opp_stack_bb": 165,
        "to_call_bb": 5,
        "pot_bb": 10
    }
    
    print("\nPreflop: HA DK (A♥ K♦)")
    result = main(cv_data)
    print(f"Action: {result['action']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Probabilities: {result['probabilities']}")
