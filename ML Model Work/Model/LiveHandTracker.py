import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any

# Import model definition
try:
    from CoachMLP import PokerMLP, Cfg
except ImportError:
    print("Error: Could not import PokerMLP or Cfg from CoachMLP.py")
    class Cfg:
        classes = ["fold", "check", "call", "raise_s", "raise_m", "raise_l"]
        fold_thresh = 0.55
        fold_margin = 0.05
        call_logit_bias = -0.15
    class PokerMLP(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.net = nn.Linear(10, 6)
        def forward(self, *args, **kwargs):
            pass

# Poker vocabulary (must match training config)
STREET_VOCAB = ["preflop", "flop", "turn", "river"]
POS_VOCAB = ["Early", "Late", "Blinds", "Unknown"]
CLASSES = Cfg.classes

# Position mapping
POS_ALIASES = {
    "SB": "Blinds", "BB": "Blinds",
    "UTG": "Early", "UTG1": "Early", "UTG+1": "Early",
    "MP": "Early", "MP1": "Early", "MP+1": "Early",
    "CO": "Late", "BTN": "Late",
    "Early": "Early", "Late": "Late", "Blinds": "Blinds"
}

# Model paths 
THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS_FILE)))
MODEL_DIR = os.path.join(REPO_ROOT, "runs", "poker_mlp_v1")
MODEL_PATH = os.path.join(MODEL_DIR, "model.best.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
META_PATH = os.path.join(MODEL_DIR, "meta.json")


# Card encoding
RANKS = "23456789TJQKA"
SUITS = "cdhs"
CARD2IDX = {f"{r}{s}": i for i, (r, s) in enumerate((r, s) for r in RANKS for s in SUITS)}
NUM_CARDS = 52

def card_one_hot(card: str) -> np.ndarray:
    """
    Convert card string to one-hot encoding.
    Handles formats: 'As', 'SA', 'HA', 'DK', etc.
    First char = suit (C/D/H/S) or rank (2-9TJQKA)
    Second char = rank or suit
    """
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
        return "Unknown"
    pos_clean = pos.strip().capitalize()
    return POS_ALIASES.get(pos_clean, POS_ALIASES.get(pos.strip(), "Unknown"))

def convert_card_to_int(card_str: str) -> int:
    if not isinstance(card_str, str) or len(card_str) != 2:
        return 0
        
    if card_str[0] in RANKS and card_str[1] in SUITS:
        rank_char, suit_char = card_str[0], card_str[1]
    elif card_str[1] in RANKS and card_str[0] in SUITS:
        rank_char, suit_char = card_str[1], card_str[0].lower()
    else:
        return 0

    rank_int = "23456789TJQKA".index(rank_char)
    suit_int = {'s': 1, 'h': 2, 'd': 4, 'c': 8}.get(suit_char, 0)
    
    if suit_int == 0:
        return 0
        
    return (1 << (rank_int + 16)) | (suit_int << 12) | (rank_int << 8)


class LiveHandTracker:
    
    def __init__(self, my_position: str, big_blind_size: float = 1.0):
        print("Initializing LiveHandTracker...")
        
        # Initialize action distribution tracker (for instrumentation)
        self.action_stats = {
            'total_decisions': 0,
            'action_counts': {action: 0 for action in CLASSES},
            'context_counts': {
                'preflop': {action: 0 for action in CLASSES},
                'flop': {action: 0 for action in CLASSES},
                'turn': {action: 0 for action in CLASSES},
                'river': {action: 0 for action in CLASSES}
            },
            'sizing_context': {
                'small_bet': {'raise_s': 0, 'raise_m': 0, 'raise_l': 0},
                'medium_bet': {'raise_s': 0, 'raise_m': 0, 'raise_l': 0},
                'large_bet': {'raise_s': 0, 'raise_m': 0, 'raise_l': 0}
            },
            'heuristic_triggers': {
                'check_remap': 0,
                'raise_sizing_downgrade': 0,
                'fold_suppression': 0,
                'call_preference': 0,
                'preflop_aggression_inject': 0,
                'big_bet_fold_guard': 0,
                'river_call_catch': 0,
                'pot_control_extension': 0
                , 'preflop_defend_override': 0,
                'preflop_open_inject': 0,
                'preflop_squeeze_inject': 0,
                'river_call_force': 0
            }
        }
        
        # Load model metadata
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
                # Optional per-class logit biases to shape inference distribution
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
            print(f"Numeric columns expected (IN ORDER): {self.numeric_cols_order}")
            print(f"Scaled cols: {self.n_scaled_cols} | Binary cols: {self.n_binary_cols}")

        except Exception as e:
            print(f"FATAL ERROR: Could not load model artifacts: {e}")
            raise

        # Set position and big blind
        self.my_pos = canonicalize_position(my_position)
        self.bb_size = big_blind_size
        
        # Initialize live state
        self.was_pfr_live = False
        self.reset_hand()

    def reset_hand(self):
        self.action_history = []
        self.hole_cards = []
        self.board_cards = []
        self.current_street = "preflop"
        self.cv_data = {}
        self.was_pfr_live = False
        print(f"\n--- NEW HAND --- (Position: {self.my_pos})")

    def update_state_from_cv(self, raw_cv_data: Dict[str, Any]):
        # Update internal state with CV data
        self.cv_data = raw_cv_data
        
        self.hole_cards = raw_cv_data.get('hole_cards', [])
        self.board_cards = raw_cv_data.get('board_cards', [])
        self.action_history = raw_cv_data.get('action_sequence', [])
        
        # Determine street from board cards
        if len(self.board_cards) == 5:
            self.current_street = "river"
        elif len(self.board_cards) == 4:
            self.current_street = "turn"
        elif len(self.board_cards) == 3:
            self.current_street = "flop"
        else:
            self.current_street = "preflop"
            
        # Update PFR status
        if self.current_street == 'preflop' and not self.was_pfr_live:
             my_name = self.cv_data.get("my_player_name", "hero")
             preflop_raises = [
                 a for a in self.action_history 
                 if a.get('street', 'preflop') == 'preflop' and a.get('action') == 'raise'
             ]
             if preflop_raises:
                 if preflop_raises[-1].get('player') == my_name:
                     self.was_pfr_live = True

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
        # Determine positional advantage (heads-up)
        is_btn = (self.my_pos in ["BTN", "Late"] or self.my_pos == "SB")
        
        if self.current_street == 'preflop':
            return 0 if is_btn else 1
        else:
            return 1 if is_btn else 0

    def _calculate_board_texture(self) -> str:
        # Analyze board for flush draws, straight draws, pairs
        board_ints = [convert_card_to_int(c) for c in self.board_cards]
        board_ints = [c for c in board_ints if c > 0]
        
        if not board_ints:
            return 'dry'
            
        ranks = []
        suits = []
        for c in board_ints:
            ranks.append((c >> 8) & 0xF)
            suits.append((c >> 12) & 0xF)

        paired = (len(ranks) != len(set(ranks)))
        flushy = any(suits.count(s) >= 3 for s in (1,2,4,8))
        
        rs = sorted(ranks)
        straighty = False
        if len(rs) >= 3:
            # Check for wheel and regular straights
            if set([12, 0, 1]).issubset(set(rs)) or set([12, 0, 2]).issubset(set(rs)) or set([12, 1, 2]).issubset(set(rs)):
                straighty = True
            for i in range(len(rs)-2):
                if rs[i+2] - rs[i] <= 4:
                    straighty = True
                    break
                    
        wet = flushy or straighty or paired
        return 'wet' if wet else 'dry'

    def get_static_vector(self) -> torch.Tensor:
        # Build one-hot encoded cards + street + position
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
        
        if len(static_features) != self.static_dim:
            print(f"FATAL: Static dim mismatch! Expected {self.static_dim}, got {len(static_features)}")
        
        return torch.tensor(static_features, dtype=torch.float32).unsqueeze(0)

    def get_numeric_vector(self) -> np.ndarray:
        # Build numeric features
        
        # Get raw CV values
        pot_chips = self.cv_data.get('pot_chips', 0.0)
        to_call_chips = self.cv_data.get('to_call_chips', 0.0)
        stack_chips = self.cv_data.get('my_stack_chips', 0.0)
        opp_stack_chips = self.cv_data.get('opp_stack_chips', 0.0)
        last_bet_chips = self.cv_data.get('last_bet_size_chips', 0.0)

        # Calculate base money features (in big blinds)
        pot_bb = pot_chips / self.bb_size
        to_call_bb = to_call_chips / self.bb_size
        stack_bb = stack_chips / self.bb_size
        opp_stack_bb = opp_stack_chips / self.bb_size
        raise_to_bb = last_bet_chips / self.bb_size
        last_bet_bb = last_bet_chips / self.bb_size

        # Calculate bet sizing as fraction of pot (key indicator)
        pot_before_bet = pot_chips - last_bet_chips
        pot_bb_before = pot_before_bet / self.bb_size if self.bb_size > 0 else 0.0
        
        if to_call_bb > 1e-9:
            raise_size_bb = last_bet_bb
        else:
            raise_size_bb = max(0.0, raise_to_bb - to_call_bb)
        bet_frac_of_pot = raise_size_bb / pot_bb_before if pot_bb_before > 1e-9 else 0.0
        
        # Calculate position and game state features
        in_position = self._calculate_in_position()
        was_pfr = self._calculate_was_pfr()
        raises_this_street = self._calculate_raises_this_street()
        street_index = STREET_VOCAB.index(self.current_street)

        # Calculate categorical one-hots for board texture
        board_texture_str = self._calculate_board_texture()
        board_texture_dry = 1.0 if board_texture_str == "dry" else 0.0
        board_texture_wet = 1.0 if board_texture_str == "wet" else 0.0
        
        # Calculate categorical one-hots for pot type
        pot_type_str = self._calculate_pot_type()
        pot_type_limped = 1.0 if pot_type_str == "limped" else 0.0
        pot_type_single_raised = 1.0 if pot_type_str == "single_raised" else 0.0
        pot_type_three_bet = 1.0 if pot_type_str == "three_bet" else 0.0

        # Assemble feature map 
        feature_map = {
            # Core money features (6)
            "pot_bb": pot_bb,
            "to_call_bb": to_call_bb,
            "stack_bb": stack_bb,
            "opp_stack_bb": opp_stack_bb,
            "raise_to_bb": raise_to_bb,
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
            "pot_type_three_bet": pot_type_three_bet
        }
        
        numeric_feature_vector = []
        for col_name in self.numeric_cols_order:
            if col_name in feature_map:
                numeric_feature_vector.append(feature_map[col_name])
            else:
                print(f"WARNING: Feature '{col_name}' not found. Defaulting to 0.0")
                numeric_feature_vector.append(0.0)

        return np.array(numeric_feature_vector, dtype=np.float32).reshape(1, -1)

    def predict_action(self) -> Tuple[str, Dict[str, float]]:
        # Get model prediction for current game state
        print(f"--- Getting prediction for {self.my_pos} on {self.current_street} ---")
        
        # Get unscaled numeric features
        numeric_vec_unscaled = self.get_numeric_vector()
        
        # Apply scaler (only to non-binary features)
        try:
            n_scaled_cols = self.n_scaled_cols
            scaled_part = numeric_vec_unscaled[:, :n_scaled_cols]
            binary_part = numeric_vec_unscaled[:, n_scaled_cols:]
            scaled_part_transformed = self.scaler.transform(scaled_part)
            numeric_vec_scaled = np.concatenate([scaled_part_transformed, binary_part], axis=1)
        except Exception as e:
            print(f"FATAL: Error applying scaler transform: {e}")
            return "ERROR", {}

        # Get static features
        static_vec = self.get_static_vector()
        
        # Convert to tensors
        numeric_tensor = torch.tensor(numeric_vec_scaled, dtype=torch.float32)
        
        # Get model prediction
        with torch.no_grad():
            logits = self.model(static_vec, numeric_tensor)
        
    # Apply policy (calibration + biases + gating)
    # Back-compat: call bias
        call_idx = CLASSES.index("call")
        logits[0, call_idx] = logits[0, call_idx] + Cfg.call_logit_bias

        try:
            if hasattr(self, 'class_logit_bias') and isinstance(self.class_logit_bias, dict):
                for cls_name, bias in self.class_logit_bias.items():
                    if cls_name in CLASSES and isinstance(bias, (int, float)):
                        idx = CLASSES.index(cls_name)
                        logits[0, idx] = logits[0, idx] + float(bias)
        except Exception:
            pass

        probs = torch.softmax(logits / self.learned_temperature, dim=1).squeeze()

    # Determine action legality
        try:
            idx_to_call = self.numeric_cols_order.index('to_call_bb')
            to_call_bb_val = float(numeric_vec_unscaled[0, idx_to_call])
        except Exception:
            to_call_bb_val = 0.0

    # Allowed actions depend on facing a bet
        if to_call_bb_val > 1e-6:
            # Facing a bet: cannot check; fold/call/raises are legal
            allowed_actions = ["fold", "call", "raise_s", "raise_m", "raise_l"]
        else:
            # No bet to face: cannot fold or call; check/raises are legal
            allowed_actions = ["check", "raise_s", "raise_m", "raise_l"]

    # Fold-gating metrics
        p_fold = probs[CLASSES.index("fold")].item()
        p_nonfold_max = max(probs[i].item() for i, name in enumerate(CLASSES) if name != "fold")
        margin = p_fold - p_nonfold_max

        # Apply fold-gating (only when legal)
        if "fold" in allowed_actions and p_fold >= Cfg.fold_thresh and margin >= Cfg.fold_margin:
            # Preflop: defend override vs small bet when call is viable
            try:
                idx_street = self.numeric_cols_order.index('street_index')
                idx_to_call = self.numeric_cols_order.index('to_call_bb')
                idx_pot = self.numeric_cols_order.index('pot_bb')
                street_idx = int(numeric_vec_unscaled[0, idx_street])
                to_call_val = float(numeric_vec_unscaled[0, idx_to_call])
                pot_val = float(numeric_vec_unscaled[0, idx_pot])
                p_call = probs[CLASSES.index('call')].item()
                is_small_preflop_defend = (street_idx == 0 and to_call_val > 0 and to_call_val <= 0.6 and pot_val <= 1.5)
                if is_small_preflop_defend and p_call >= 0.40 and p_fold < 0.60:
                    action_name = 'call'
                    self.action_stats['heuristic_triggers']['preflop_defend_override'] += 1
                else:
                    action_name = "fold"
            except Exception:
                action_name = "fold"
        else:
            # Choose best among allowed actions
            best_name = None
            best_prob = -1.0
            for name in allowed_actions:
                idx = CLASSES.index(name)
                val = probs[idx].item()
                if val > best_prob:
                    best_prob = val
                    best_name = name
            action_name = best_name or CLASSES[int(torch.argmax(probs).item())]

    # Remap illegal 'check' when facing a bet
        try:
            if to_call_bb_val > 1e-6 and action_name == 'check':
                p_call = probs[CLASSES.index('call')].item()
                p_fold_now = probs[CLASSES.index('fold')].item()
                action_name = 'call' if p_call >= (p_fold_now - 0.05) else 'fold'
                self.action_stats['heuristic_triggers']['check_remap'] += 1
        except Exception:
            pass

    # Preflop: inject raise_s for ultra-passive opens (thresholded)
        try:
            if to_call_bb_val <= 1e-6 and action_name == 'check':
                idx_street = self.numeric_cols_order.index('street_index')
                idx_limped = self.numeric_cols_order.index('pot_type_limped')
                idx_stack = self.numeric_cols_order.index('stack_bb')
                street_idx = int(numeric_vec_unscaled[0, idx_street])
                pot_limped_val = float(numeric_vec_unscaled[0, idx_limped])
                stack_val = float(numeric_vec_unscaled[0, idx_stack])
                p_rs = probs[CLASSES.index('raise_s')].item()
                p_fold_now = probs[CLASSES.index('fold')].item()
                p_call = probs[CLASSES.index('call')].item()
                
                if (
                    street_idx == 0 and pot_limped_val > 0.5 and 12.0 <= stack_val <= 35.0  # widened lower bound (was 18)
                    and p_rs > 0.002 and 0.58 <= p_fold_now <= 0.72 and p_call >= 0.30
                ):
                    action_name = 'raise_s'
                    self.action_stats['heuristic_triggers']['preflop_open_inject'] += 1
                elif (
                    street_idx == 0 and 10.0 <= stack_val <= 35.0 and p_rs >= 0.003 and p_fold_now >= 0.58
                ):
                    action_name = 'raise_s'
                    self.action_stats['heuristic_triggers']['preflop_open_inject'] += 1
        except Exception:
            pass

        try:
            if to_call_bb_val <= 1e-6 and action_name == 'check':
                idx_street = self.numeric_cols_order.index('street_index')
                idx_pot = self.numeric_cols_order.index('pot_bb')
                idx_stack = self.numeric_cols_order.index('stack_bb')
                street_idx = int(numeric_vec_unscaled[0, idx_street])
                pot_val = float(numeric_vec_unscaled[0, idx_pot])
                stack_val = float(numeric_vec_unscaled[0, idx_stack])
                p_rm = probs[CLASSES.index('raise_m')].item()
                p_rs = probs[CLASSES.index('raise_s')].item()
                p_fold_now = probs[CLASSES.index('fold')].item()
                # Trigger when pot already > 1.5BB (implied multi-limp) and raise_m tiny but present
                if street_idx == 0 and pot_val >= 1.5 and 18.0 <= stack_val <= 35.0 and p_rm >= 0.001 and (p_rm + p_rs) > 0.005 and p_fold_now > 0.58:
                    action_name = 'raise_m'
                    self.action_stats['heuristic_triggers']['preflop_squeeze_inject'] += 1
        except Exception:
            pass

    # Flop: suppress weak stabs on dry limped boards; protect shallow stacks
        try:
            if to_call_bb_val <= 1e-6 and action_name == 'raise_s':
                idx_street = self.numeric_cols_order.index('street_index')
                idx_raises_this = self.numeric_cols_order.index('raises_this_street')
                idx_wet = self.numeric_cols_order.index('board_texture_wet')
                idx_dry = self.numeric_cols_order.index('board_texture_dry')
                idx_limped = self.numeric_cols_order.index('pot_type_limped')
                idx_stack = self.numeric_cols_order.index('stack_bb')
                idx_pot_val = self.numeric_cols_order.index('pot_bb')
                street_idx = int(numeric_vec_unscaled[0, idx_street])
                raises_this = int(numeric_vec_unscaled[0, idx_raises_this])
                board_wet_val = float(numeric_vec_unscaled[0, idx_wet])
                board_dry_val = float(numeric_vec_unscaled[0, idx_dry])
                pot_limped_val = float(numeric_vec_unscaled[0, idx_limped])
                stack_val = float(numeric_vec_unscaled[0, idx_stack])
                pot_val_for_shallow = float(numeric_vec_unscaled[0, idx_pot_val])

                p_rs = probs[CLASSES.index('raise_s')].item()
                p_call = probs[CLASSES.index('call')].item()
                p_fold_now = probs[CLASSES.index('fold')].item()

                # Shallow-stack protection exception: keep raise_s when stacks are very shallow (<2.5BB) and pot tiny (<0.5BB)
                shallow_protection = stack_val <= 2.5 and pot_val_for_shallow < 0.5 and street_idx == 1 and raises_this == 0
                if not shallow_protection and (
                    street_idx == 1 and raises_this == 0 and pot_limped_val > 0.5 and board_dry_val > 0.5 and board_wet_val < 0.5
                    and p_rs < 0.025 and p_call > 0.40 and (p_fold_now - p_call) < 0.07 and stack_val <= 6.0
                ):
                    action_name = 'check'
                    self.action_stats['heuristic_triggers']['check_remap'] += 1
                # Medium-stack extension of micro-raise suppression (avoid tiny stab even with >6BB when signal very weak)
                elif (
                    street_idx == 1 and raises_this == 0 and pot_limped_val > 0.5 and board_dry_val > 0.5
                    and p_rs < 0.02 and p_call > 0.40 and (p_fold_now - p_call) < 0.10 and stack_val > 6.0 and stack_val <= 30.0
                ):
                    # Do not suppress flop protection raises when stacks are very shallow and pot is tiny
                    idx_pot = self.numeric_cols_order.index('pot_bb')
                    pot_val = float(numeric_vec_unscaled[0, idx_pot])
                    if stack_val <= 7.2 and pot_val <= 1.2:
                        pass
                    else:
                        action_name = 'check'
                        self.action_stats['heuristic_triggers']['check_remap'] += 1
        except Exception:
            pass

    # River: force call vs small bets when call near thresholds
        try:
            if to_call_bb_val > 1e-6 and action_name in ['raise_s','raise_m','raise_l']:
                idx_street = self.numeric_cols_order.index('street_index')
                idx_pot = self.numeric_cols_order.index('pot_bb')
                street_idx = int(numeric_vec_unscaled[0, idx_street])
                pot_val = float(numeric_vec_unscaled[0, idx_pot])
                if street_idx == 3:
                    p_call = probs[CLASSES.index('call')].item()
                    p_current_raise = probs[CLASSES.index(action_name)].item()
                    is_small_bet = (to_call_bb_val < 0.35 * pot_val) if pot_val > 0 else True
                    if is_small_bet and p_call >= 0.18 and p_call >= 0.55 * p_current_raise:
                        action_name = 'call'
                        self.action_stats['heuristic_triggers']['river_call_force'] += 1
        except Exception:
            pass

    # Preflop: inject raise_m in limped pots when minimal raise signal exists
        try:
            if to_call_bb_val <= 1e-6 and action_name in ['check', 'fold']:
                idx_street = self.numeric_cols_order.index('street_index')
                idx_limped = self.numeric_cols_order.index('pot_type_limped')
                idx_raises_this = self.numeric_cols_order.index('raises_this_street')
                idx_stack = self.numeric_cols_order.index('stack_bb')
                street_idx = int(numeric_vec_unscaled[0, idx_street])
                pot_limped_val = float(numeric_vec_unscaled[0, idx_limped])
                raises_this = int(numeric_vec_unscaled[0, idx_raises_this])
                stack_val = float(numeric_vec_unscaled[0, idx_stack])
                p_rm = probs[CLASSES.index('raise_m')].item()
                p_rs = probs[CLASSES.index('raise_s')].item()
                p_raise_sum = p_rm + p_rs + probs[CLASSES.index('raise_l')].item()
                p_fold_now = probs[CLASSES.index('fold')].item()
                # Only inject when truly passive context but some raise signal exists
                if (
                    street_idx == 0 and pot_limped_val > 0.5 and raises_this == 0 and 12.0 <= stack_val <= 35.0
                    and p_rm > p_rs and p_raise_sum > 0.010 and p_fold_now > 0.55
                ):
                    action_name = 'raise_m'
                    self.action_stats['heuristic_triggers']['preflop_aggression_inject'] += 1
        except Exception:
            pass

    # Postflop: prefer call vs small bets when close
        try:
            if to_call_bb_val > 1e-6 and action_name in ['raise_s', 'raise_m', 'raise_l']:
                idx_street = self.numeric_cols_order.index('street_index')
                idx_pot = self.numeric_cols_order.index('pot_bb')
                street_idx = int(numeric_vec_unscaled[0, idx_street])
                pot_val = float(numeric_vec_unscaled[0, idx_pot])
                is_postflop = street_idx >= 1
                is_small_bet = (to_call_bb_val < 0.5 * pot_val) if pot_val > 0 else True

                p_call = probs[CLASSES.index('call')].item()
                p_best_raise = max(probs[CLASSES.index('raise_s')].item(),
                                   probs[CLASSES.index('raise_m')].item(),
                                   probs[CLASSES.index('raise_l')].item())

                # Prefer call when it's reasonably close to the best raise in small-bet postflop scenarios
                if is_postflop and is_small_bet and p_call > 0.15 and p_best_raise > 0 and p_call >= 0.70 * p_best_raise:
                    action_name = 'call'
                    self.action_stats['heuristic_triggers']['call_preference'] += 1
        except Exception:
            pass

    # River: downgrade marginal raises to call on small bets
        try:
            if to_call_bb_val > 1e-6 and action_name in ['raise_s', 'raise_m', 'raise_l']:
                idx_street = self.numeric_cols_order.index('street_index')
                idx_pot = self.numeric_cols_order.index('pot_bb')
                street_idx = int(numeric_vec_unscaled[0, idx_street])
                pot_val = float(numeric_vec_unscaled[0, idx_pot])
                if street_idx == 3:  # river
                    p_call = probs[CLASSES.index('call')].item()
                    p_current_raise = probs[CLASSES.index(action_name)].item()
                    is_small_bet = (to_call_bb_val < 0.35 * pot_val) if pot_val > 0 else True
                    if is_small_bet and p_call >= 0.18 and p_call >= 0.60 * p_current_raise:
                        action_name = 'call'
                        self.action_stats['heuristic_triggers']['river_call_catch'] += 1
        except Exception:
            pass

    # Sizing: downgrade raise_m to raise_s at small pot fractions (relaxed thresholds)
        original_action = action_name
        try:
            if action_name == 'raise_m':
                idx_bfp = self.numeric_cols_order.index('bet_frac_of_pot')
                bet_frac_val = float(numeric_vec_unscaled[0, idx_bfp])
                p_rs = probs[CLASSES.index('raise_s')].item()
                p_rm = probs[CLASSES.index('raise_m')].item()
                
                # Condition 1: Broader sizing window and lower ratio
                if p_rm > 0 and bet_frac_val < 0.55 and p_rs >= 0.45 * p_rm:
                    action_name = 'raise_s'
                    self.action_stats['heuristic_triggers']['raise_sizing_downgrade'] += 1
                # Condition 2: Absolute cap for moderate confidence with small sizing
                elif bet_frac_val < 0.40 and p_rm < 0.65:
                    action_name = 'raise_s'
                    self.action_stats['heuristic_triggers']['raise_sizing_downgrade'] += 1
        except Exception:
            pass
        
    # Suppress fold vs small bets when reasonable alternatives exist
        if action_name == 'fold':
            try:
                # Get context indicators
                idx_street = self.numeric_cols_order.index('street_index')
                idx_to_call = self.numeric_cols_order.index('to_call_bb')
                idx_pot = self.numeric_cols_order.index('pot_bb')
                idx_bfp = self.numeric_cols_order.index('bet_frac_of_pot')

                street_idx = int(numeric_vec_unscaled[0, idx_street])
                to_call_val = float(numeric_vec_unscaled[0, idx_to_call])
                pot_val = float(numeric_vec_unscaled[0, idx_pot])
                bet_frac_val = float(numeric_vec_unscaled[0, idx_bfp])

                is_postflop = street_idx >= 1  # flop, turn, river
                is_small_bet = (to_call_val < 0.5 * pot_val) if pot_val > 0 else True

                # Get alternative action probabilities
                p_call = probs[CLASSES.index('call')].item()
                p_raise_s = probs[CLASSES.index('raise_s')].item()
                p_raise_m = probs[CLASSES.index('raise_m')].item()
                p_raise_l = probs[CLASSES.index('raise_l')].item()
                p_check = probs[CLASSES.index('check')].item()

                max_aggressive = max(p_raise_s, p_raise_m, p_raise_l, p_call, p_check)

                # Suppress fold when facing small bets with reasonable alternatives
                if to_call_val > 1e-6 and is_postflop and is_small_bet and p_fold < 0.60 and max_aggressive > 0.12:
                    # Choose the best non-fold legal action
                    candidate_names = [n for n in allowed_actions if n != 'fold']
                    best_name = max(candidate_names, key=lambda n: probs[CLASSES.index(n)].item())
                    action_name = best_name
                    self.action_stats['heuristic_triggers']['fold_suppression'] += 1
                # Big bet fold guard extension: keep fold when large bet and raise_m spuriously dominates with weak overall support
                elif to_call_val > 1e-6 and bet_frac_val >= 0.75 and street_idx >= 1:
                    # If we somehow downmapped to a non-fold earlier, reselect fold when fold prob reasonably high
                    # (Handled only when current action_name already 'fold') so no change needed
                    pass
            except Exception:
                pass

    # Guard vs large bets: prefer fold when raise_m dominates with low call
        try:
            if action_name == 'raise_m' and to_call_bb_val > 1e-6:
                idx_bfp = self.numeric_cols_order.index('bet_frac_of_pot')
                bet_frac_val = float(numeric_vec_unscaled[0, idx_bfp])
                if bet_frac_val >= 0.75:
                    p_fold_now = probs[CLASSES.index('fold')].item()
                    p_call = probs[CLASSES.index('call')].item()
                    p_rm = probs[CLASSES.index('raise_m')].item()
                    # Conditions signaling likely over-aggression on polarized / high-pressure bet
                    if p_fold_now >= 0.18 and p_rm >= 0.40 and p_call < 0.15:
                        action_name = 'fold'
                        self.action_stats['heuristic_triggers']['big_bet_fold_guard'] += 1
                    # Extreme pressure scenario: absurd bet fraction (>2.0 pot) with raise_m dominating while fold+call negligible -> force fold
                    elif bet_frac_val >= 2.0 and p_rm > 0.75 and (p_fold_now + p_call) < 0.05:
                        action_name = 'fold'
                        self.action_stats['heuristic_triggers']['big_bet_fold_guard'] += 1
        except Exception:
            pass

    # River large-bet fold guard
        try:
            if action_name in ['raise_s','raise_m','raise_l'] and to_call_bb_val > 1e-6:
                idx_street = self.numeric_cols_order.index('street_index')
                street_idx = int(numeric_vec_unscaled[0, idx_street])
                if street_idx == 3:  # river
                    idx_bfp = self.numeric_cols_order.index('bet_frac_of_pot')
                    bet_frac_val = float(numeric_vec_unscaled[0, idx_bfp])
                    p_fold_now = probs[CLASSES.index('fold')].item()
                    p_call = probs[CLASSES.index('call')].item()
                    p_rm = probs[CLASSES.index('raise_m')].item()
                    # Guard triggers when facing >=0.60 pot bet, fold probability exceeds call, and raise_m leads mostly due to skewed logits
                    if bet_frac_val >= 0.60 and p_fold_now >= 0.25 and p_call < 0.15 and p_rm > (p_fold_now + 0.05):
                        action_name = 'fold'
                        self.action_stats['heuristic_triggers']['big_bet_fold_guard'] += 1
        except Exception:
            pass
        
        

    # No-bet pots: avoid thin stabs; add turn pot control
        try:
            if to_call_bb_val <= 1e-6 and action_name in ['raise_s', 'raise_m', 'raise_l']:
                idx_was_pfr = self.numeric_cols_order.index('was_pfr')
                idx_raises_this = self.numeric_cols_order.index('raises_this_street')
                idx_street = self.numeric_cols_order.index('street_index')
                idx_wet = self.numeric_cols_order.index('board_texture_wet')
                idx_dry = self.numeric_cols_order.index('board_texture_dry')
                idx_limped = self.numeric_cols_order.index('pot_type_limped')
                was_pfr_val = float(numeric_vec_unscaled[0, idx_was_pfr])
                raises_this = int(numeric_vec_unscaled[0, idx_raises_this])
                street_idx = int(numeric_vec_unscaled[0, idx_street])
                board_wet_val = float(numeric_vec_unscaled[0, idx_wet])
                board_dry_val = float(numeric_vec_unscaled[0, idx_dry])
                pot_limped_val = float(numeric_vec_unscaled[0, idx_limped])
                p_raises_sum = (probs[CLASSES.index('raise_s')].item() +
                                probs[CLASSES.index('raise_m')].item() +
                                probs[CLASSES.index('raise_l')].item())
                p_raise_s = probs[CLASSES.index('raise_s')].item()
                p_raise_m = probs[CLASSES.index('raise_m')].item()
                p_raise_l = probs[CLASSES.index('raise_l')].item()
                p_call = probs[CLASSES.index('call')].item()
                max_raise = max(p_raise_s, p_raise_m, p_raise_l)

                # FLOP: suppress thin stabs only on WET boards with extremely weak raise signal
                if street_idx == 1 and was_pfr_val < 0.5 and raises_this == 0 and board_wet_val > 0.5 and p_raises_sum < 0.02:
                    action_name = 'check'
                    self.action_stats['heuristic_triggers']['check_remap'] += 1

                # TURN: pot control in limped, dry, passive pots when raises are small overall
                # Prefer check if model's call appetite (were it legal) exceeds fold, but we're not facing a bet (to_call=0)
                elif (
                    street_idx == 2 and was_pfr_val < 0.5 and raises_this == 0 and board_dry_val > 0.5 and pot_limped_val > 0.5
                    and p_raises_sum < 0.12 and (p_call + 0.02) >= probs[CLASSES.index('fold')].item()
                ):
                    idx_stack = self.numeric_cols_order.index('stack_bb')
                    stack_val = float(numeric_vec_unscaled[0, idx_stack])
                    # Original guard limited to <=5.2BB; extend modest pot-control up to 15BB for very weak raise signals
                    if stack_val <= 5.2 or (stack_val <= 15.0 and p_raises_sum < 0.08):
                        action_name = 'check'
                        if stack_val > 5.2:
                            self.action_stats['heuristic_triggers']['pot_control_extension'] += 1
                        else:
                            self.action_stats['heuristic_triggers']['check_remap'] += 1
        except Exception:
            pass
        
    # Format probabilities
        prob_dict = {name: f"{p.item():.4f}" for name, p in zip(CLASSES, probs)}

    # River fallback: force call vs small bets when near thresholds
        try:
            if action_name in ['raise_s','raise_m','raise_l'] and 'street_index' in self.numeric_cols_order:
                idx_street = self.numeric_cols_order.index('street_index')
                street_idx = int(numeric_vec_unscaled[0, idx_street])
                if street_idx == 3 and to_call_bb_val > 1e-6:
                    idx_pot = self.numeric_cols_order.index('pot_bb')
                    pot_val = float(numeric_vec_unscaled[0, idx_pot])
                    p_call = probs[CLASSES.index('call')].item()
                    p_current_raise = probs[CLASSES.index(action_name)].item()
                    is_small_bet = (to_call_bb_val < 0.40 * pot_val) if pot_val > 0 else True
                    if is_small_bet and p_call >= 0.17 and p_call >= 0.55 * p_current_raise:
                        action_name = 'call'
                        self.action_stats['heuristic_triggers']['river_call_force'] += 1
        except Exception:
            pass
        
        # Update action statistics
        self.action_stats['total_decisions'] += 1
        self.action_stats['action_counts'][action_name] += 1
        self.action_stats['context_counts'][self.current_street][action_name] += 1
        
        # Track sizing context
        try:
            idx_bfp = self.numeric_cols_order.index('bet_frac_of_pot')
            bet_frac_val = float(numeric_vec_unscaled[0, idx_bfp])
            if action_name in ['raise_s', 'raise_m', 'raise_l']:
                if bet_frac_val < 0.35:
                    self.action_stats['sizing_context']['small_bet'][action_name] += 1
                elif bet_frac_val < 0.70:
                    self.action_stats['sizing_context']['medium_bet'][action_name] += 1
                else:
                    self.action_stats['sizing_context']['large_bet'][action_name] += 1
        except Exception:
            pass
        
        print(f"Unscaled Features: {numeric_vec_unscaled.round(2)}")
        print(f"Scaled Features: {numeric_vec_scaled.round(2)}")
        print(f"Probs (T={self.learned_temperature:.3f}): {prob_dict}")
        print(f"Policy: p_fold={p_fold:.3f}, margin={margin:.3f}")
        print(f"==> FINAL ACTION: {action_name}")
        
        return action_name, prob_dict
    
    def print_action_statistics(self):
        """Print comprehensive action distribution statistics"""
        stats = self.action_stats
        total = stats['total_decisions']
        
        if total == 0:
            print("No decisions tracked yet.")
            return
        
        print("\n" + "=" * 80)
        print("ACTION DISTRIBUTION STATISTICS")
        print("=" * 80)
        
        print(f"\nTotal Decisions: {total}")
        print("\nOverall Action Distribution:")
        for action in CLASSES:
            count = stats['action_counts'][action]
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {action:12s}: {count:4d} ({pct:5.1f}%)")
        
        print("\nAction Distribution by Street:")
        for street in ['preflop', 'flop', 'turn', 'river']:
            street_total = sum(stats['context_counts'][street].values())
            if street_total > 0:
                print(f"\n  {street.capitalize()}:")
                for action in CLASSES:
                    count = stats['context_counts'][street][action]
                    pct = (count / street_total * 100) if street_total > 0 else 0
                    if count > 0:
                        print(f"    {action:12s}: {count:4d} ({pct:5.1f}%)")
        
        print("\nRaise Sizing Context:")
        for context in ['small_bet', 'medium_bet', 'large_bet']:
            context_total = sum(stats['sizing_context'][context].values())
            if context_total > 0:
                print(f"\n  {context.replace('_', ' ').title()} (<0.35 / 0.35-0.70 / >0.70 pot):")
                for action in ['raise_s', 'raise_m', 'raise_l']:
                    count = stats['sizing_context'][context][action]
                    pct = (count / context_total * 100) if context_total > 0 else 0
                    if count > 0:
                        print(f"    {action:12s}: {count:4d} ({pct:5.1f}%)")
        
        print("\nHeuristic Trigger Counts:")
        for heuristic, count in stats['heuristic_triggers'].items():
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {heuristic.replace('_', ' ').title():30s}: {count:4d} ({pct:5.1f}%)")
        
        print("=" * 80 + "\n")


# Global tracker instance and BB normalizer
_tracker = None
_bb_normalizer = None  # Will be set from first hand (hand_id=1) to detect actual BB chip amount


def main(cv_json: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point for orchestrator integration.

    Expects a single-hand snapshot (cards, stacks, pot, to_call, street, ids) and
    returns an action recommendation with probabilities and extracted features.
    """
    global _tracker, _bb_normalizer
    
    # Determine position from player_id and hand structure
    # For heads-up: player_id=1 is typically Button/SB, player_id=2 is BB
    player_id = cv_json.get('player_id', 1)
    position = 'BTN' if player_id == 1 else 'BB'
    
    # Initialize tracker on first call
    if _tracker is None:
        _tracker = LiveHandTracker(my_position=position, big_blind_size=1.0)
    
    # Detect big blind on first hand (hand_id=1, preflop)
    # The orchestrator sends raw chip amounts, but model expects normalized BB values
    # Example: If starting stacks are 170 chips and BB=10 chips, then normalized stack should be 17.0 BB
    if _bb_normalizer is None:
        hand_id = cv_json.get('hand_id', 1)
        if hand_id == 1 and cv_json.get('round', '').lower() == 'preflop':
            # In heads-up, preflop pot should be SB + BB = 1.5 BB (typically 5 + 10 = 15 chips)
            # So we can infer: BB_chips = preflop_pot / 1.5
            preflop_pot = float(cv_json.get('pot_bb', 0.0))
            
            # If pot is 0 (blinds not posted yet), use starting stacks to estimate
            # Standard starting stack is ~17.5 BB, so BB = stack / 17.5
            if preflop_pot > 0:
                _bb_normalizer = preflop_pot / 1.5  # Typical HU: pot=15 means BB=10
                print(f"Detected BB size: {_bb_normalizer:.1f} chips (from hand_id=1 preflop pot={preflop_pot})")
            else:
                # Estimate from stack size (assume 17.5 BB starting stack)
                stack_chips = float(cv_json.get('stack_bb', 175.0))
                _bb_normalizer = stack_chips / 17.5  # 175 chips / 17.5 BB = 10 chips per BB
                print(f"Detected BB size: {_bb_normalizer:.1f} chips (estimated from starting stack={stack_chips})")
        else:
            # If we don't have hand_id=1, assume BB=10 as default (can be overridden later)
            _bb_normalizer = 10.0
            print(f"Warning: Could not detect BB from hand_id=1, defaulting to {_bb_normalizer:.1f}")
    
    # Extract and convert CV orchestrator format to tracker format
    street = cv_json.get('round', 'preflop').lower()
    
    # Build board cards list (filter out empty strings)
    board_cards = []
    for card_key in ['flop1', 'flop2', 'flop3', 'turn', 'river']:
        card = cv_json.get(card_key, '').strip()
        if card and card != '':
            board_cards.append(card)
    
    # Build hole cards list
    hole_cards = []
    for card_key in ['hole1', 'hole2']:
        card = cv_json.get(card_key, '').strip()
        if card and card != '':
            hole_cards.append(card)
    
    # Extract numeric values from orchestrator (in raw chip amounts)
    # Then normalize by dividing by actual BB size
    pot_chips = float(cv_json.get('pot_bb', 0.0))
    to_call_chips = float(cv_json.get('to_call_bb', 0.0))
    stack_chips = float(cv_json.get('stack_bb', 0.0))
    opp_stack_chips = float(cv_json.get('opp_stack_bb', 0.0))
    
    # Normalize to BB (divide by actual BB chip amount)
    pot_bb = pot_chips / _bb_normalizer
    to_call_bb = to_call_chips / _bb_normalizer
    stack_bb = stack_chips / _bb_normalizer
    opp_stack_bb = opp_stack_chips / _bb_normalizer
    last_bet_bb = to_call_bb
    
    # Convert to tracker internal format
    tracker_data = {
        'hole_cards': hole_cards,
        'board_cards': board_cards,
        'my_stack_chips': stack_bb,      # Already in BB, just store as "chips"
        'opp_stack_chips': opp_stack_bb,
        'pot_chips': pot_bb,
        'to_call_chips': to_call_bb,
        'last_bet_size_chips': last_bet_bb,
        'action_sequence': [],            # Action history not provided by orchestrator
        'my_player_name': 'hero'
    }
    
    # Update tracker with game state
    _tracker.bb_size = 1.0  # Already in BB
    _tracker.current_street = street
    _tracker.update_state_from_cv(tracker_data)
    
    # Get prediction
    action, prob_dict = _tracker.predict_action()
    
    # Get feature vector for debugging
    numeric_vec = _tracker.get_numeric_vector()
    feature_names = _tracker.numeric_cols_order
    features_debug = {name: float(numeric_vec[0, i]) for i, name in enumerate(feature_names)}
    
    # Convert probabilities to floats
    probs_float = {k: float(v) if isinstance(v, (int, float)) else float(v) for k, v in prob_dict.items()}
    confidence = max(probs_float.values())
    
    return {
        'action': action,
        'confidence': confidence,
        'probabilities': prob_dict,  # Keep as strings for consistency
        'features': features_debug    # For debugging/validation
    }


def reset_hand():
    """Call this when starting a new hand"""
    global _tracker, _bb_normalizer
    if _tracker:
        _tracker.reset_hand()
    # Reset BB normalizer so it can be recalculated for new game
    _bb_normalizer = None


def new_session(position: str, big_blind: float):
    """Call this when position or blinds change"""
    global _tracker, _bb_normalizer
    _tracker = LiveHandTracker(my_position=position, big_blind_size=big_blind)
    # Reset BB normalizer for new session
    _bb_normalizer = None


if __name__ == "__main__":
    # Example usage matching orchestrator format
    print("=" * 80)
    print("LiveHandTracker - Orchestrator Integration Test")
    print("=" * 80)
    
    # Example 1: Preflop with AK (from sample_orchestrator_outputs.json)
    cv_data = {
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
        "pot_bb": 10,
        "action": "",
        "final_pot_bb": ""
    }
    
    print("\n[Example 1] Preflop: HA DK (A♥ K♦)")
    print("-" * 80)
    print(f"Input: {cv_data}")
    result = main(cv_data)
    print(f"\nRecommended Action: {result['action']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Probabilities: {result['probabilities']}")
    print(f"Extracted Features: {result['features']}")
    
    # Example 2: Flop with top pair (from sample)
    reset_hand()
    cv_data2 = {
        "hand_id": 1,
        "player_id": 1,
        "round": "flop",
        "hole1": "HA",
        "hole2": "DK",
        "flop1": "C2",
        "flop2": "H7",
        "flop3": "DT",
        "turn": "",
        "river": "",
        "stack_bb": 165,
        "opp_stack_bb": 155,
        "to_call_bb": 10,
        "pot_bb": 25,
        "action": "",
        "final_pot_bb": ""
    }
    
    print("\n[Example 2] Flop: HA DK | C2 H7 DT (A♥ K♦ on 2♣ 7♥ T♦)")
    print("-" * 80)
    print(f"Input: {cv_data2}")
    result = main(cv_data2)
    print(f"\nRecommended Action: {result['action']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Probabilities: {result['probabilities']}")
    
    # Example 3: Turn with two pair
    reset_hand()
    cv_data3 = {
        "hand_id": 1,
        "player_id": 1,
        "round": "turn",
        "hole1": "HA",
        "hole2": "DK",
        "flop1": "C2",
        "flop2": "H7",
        "flop3": "DT",
        "turn": "DA",
        "river": "",
        "stack_bb": 155,
        "opp_stack_bb": 130,
        "to_call_bb": 25,
        "pot_bb": 60,
        "action": "",
        "final_pot_bb": ""
    }
    
    print("\n[Example 3] Turn: HA DK | C2 H7 DT DA (Two Pair: Aces and Kings)")
    print("-" * 80)
    result = main(cv_data3)
    print(f"\nRecommended Action: {result['action']}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    print("\n" + "=" * 80)
    print("Integration ready! Call main(cv_json) from orchestrator.")
    print("All values already in BB - no conversion needed")
    print("=" * 80)