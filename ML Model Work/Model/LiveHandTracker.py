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

# Poker vocabulary
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
    """Convert card string to one-hot encoding. Handles formats: 'As', 'SA', 'HA', 'DK', etc."""
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


class LiveHandTracker:
    def __init__(self, my_position: str, big_blind_size: float = 1.0):
        self.bb_size = big_blind_size
        self.my_pos = canonicalize_position(my_position)
        
        # Game state
        self.hole_cards = []
        self.board_cards = []
        self.current_street = "preflop"
        self.cv_data = {}
        self.action_history = []
        self.my_player_name = "PlayerCoach"
        self.was_pfr_live = False
        
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
        """Update internal state from CV data"""
        self.hole_cards = raw_cv_data.get('hole_cards', [])
        self.board_cards = raw_cv_data.get('board_cards', [])
        self.cv_data = raw_cv_data
        self.action_history = raw_cv_data.get('action_sequence', [])
    
    def _calculate_was_pfr(self) -> int:
        """Check if player was preflop raiser"""
        return 1 if self.was_pfr_live else 0

    def _calculate_raises_preflop(self) -> int:
        """Count number of raises preflop"""
        pre_hist = [a for a in self.action_history if a.get('street', 'preflop') == 'preflop']
        return sum(1 for a in pre_hist if a.get('action') == 'raise')

    def _calculate_pot_type(self) -> str:
        """Determine pot type based on preflop action"""
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
        """Count raises on current street"""
        street_hist = [a for a in self.action_history if a.get('street') == self.current_street]
        return sum(1 for a in street_hist if a.get('action') == 'raise')

    def _calculate_in_position(self) -> int:
        """Determine positional advantage (heads-up)"""
        is_btn = (self.my_pos in ["BTN", "Late"] or self.my_pos == "SB")
        
        if self.current_street == 'preflop':
            return 0 if is_btn else 1
        else:
            return 1 if is_btn else 0

    def _calculate_board_texture(self) -> str:
        """Calculate board texture using Treys features"""
        feats = evaluate_hand_features(self.hole_cards, self.board_cards)
        has_draw = feats["has_flush_draw"] or feats["has_straight_draw"]
        if has_draw or feats["hand_bucket"] >= 2.0:
            return "wet"
        return "dry"

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
        return torch.tensor(static_features, dtype=torch.float32).unsqueeze(0)

    def get_numeric_vector(self) -> np.ndarray:
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
        last_bet_bb = last_bet_chips / self.bb_size

        # Calculate bet sizing as fraction of pot
        pot_before_bet = pot_chips - last_bet_chips
        pot_bb_before = pot_before_bet / self.bb_size if self.bb_size > 0 else 0.0
        raise_size_bb = last_bet_bb if to_call_bb > 1e-9 else max(0.0, last_bet_bb - to_call_bb)
        bet_frac_of_pot = raise_size_bb / pot_bb_before if pot_bb_before > 1e-9 else 0.0
        
        # Calculate position and game state features
        street_index = STREET_VOCAB.index(self.current_street)
        in_position = self._calculate_in_position()
        was_pfr = self._calculate_was_pfr()
        raises_this_street = self._calculate_raises_this_street()

        # Calculate board texture using Treys
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
        
        # Calculate pot type
        pot_type_str = self._calculate_pot_type()
        pot_type_limped = 1.0 if pot_type_str == "limped" else 0.0
        pot_type_single_raised = 1.0 if pot_type_str == "single_raised" else 0.0
        pot_type_three_bet = 1.0 if pot_type_str == "three_bet" else 0.0
        
        # Multi-player features (from CV/orchestrator JSON)
        # These will be provided by orchestrator after CV module update
        num_active_players = self.cv_data.get('num_active_players', 2)
        num_players_to_act = self.cv_data.get('num_players_to_act', 0)
        multiway_pot_flag = self.cv_data.get('multiway_pot_flag', 0)
        position_relative = self.cv_data.get('position_relative', 0.5)
        num_callers_this_street = self.cv_data.get('num_callers_this_street', 0)
        num_raisers_this_street = self.cv_data.get('num_raisers_this_street', 0)
        avg_opp_stack_bb = self.cv_data.get('avg_opp_stack_bb', opp_stack_bb)
        
        # Assemble feature map (MUST match training exactly)
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
        
        # Get model prediction
        with torch.no_grad():
            logits = self.model(static_vec, numeric_tensor)
        
        # Apply logit biases
        call_idx = CLASSES.index("call")
        logits[0, call_idx] = logits[0, call_idx] + Cfg.call_logit_bias
        
        if hasattr(self, 'class_logit_bias') and isinstance(self.class_logit_bias, dict):
            for cls_name, bias in self.class_logit_bias.items():
                if cls_name in CLASSES and isinstance(bias, (int, float)):
                    idx = CLASSES.index(cls_name)
                    logits[0, idx] = logits[0, idx] + float(bias)

        probs = torch.softmax(logits / self.learned_temperature, dim=1).squeeze()

        # Determine legal actions
        try:
            idx_to_call = self.numeric_cols_order.index('to_call_bb')
            to_call_bb_val = float(numeric_vec_unscaled[0, idx_to_call])
        except Exception:
            to_call_bb_val = 0.0

        if to_call_bb_val > 1e-6:
            allowed_actions = ["fold", "call", "raise"]
        else:
            allowed_actions = ["check", "raise"]
            probs_adjusted = probs.clone()
            for illegal_action in ["fold", "call"]:
                if illegal_action in CLASSES:
                    probs_adjusted[CLASSES.index(illegal_action)] = 0.0
            legal_sum = sum(probs_adjusted[CLASSES.index(a)].item() for a in allowed_actions)
            if legal_sum > 1e-9:
                for a in allowed_actions:
                    probs_adjusted[CLASSES.index(a)] /= legal_sum
            probs = probs_adjusted

        # === TREYS-BASED DYNAMIC AGGRESSION ===
        try:
            idx_street = self.numeric_cols_order.index('street_index')
            idx_pot = self.numeric_cols_order.index('pot_bb')
            street_idx = int(numeric_vec_unscaled[0, idx_street])
            pot_bb = float(numeric_vec_unscaled[0, idx_pot])
            
            # Recalculate Treys features live
            treys_feats_live = evaluate_hand_features(self.hole_cards, self.board_cards)
            hand_bucket = treys_feats_live["hand_bucket"]
            has_flush_draw = treys_feats_live["has_flush_draw"]
            has_straight_draw = treys_feats_live["has_straight_draw"]
            
            probs_adjusted = probs.clone()
            
            # === PREFLOP AGGRESSION ===
            if street_idx == 0 and len(self.hole_cards) == 2:
                ranks_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
                try:
                    card1, card2 = self.hole_cards[0], self.hole_cards[1]
                    rank1 = ranks_map.get(card1[1] if card1[0] in 'CDHS' else card1[0], 0)
                    rank2 = ranks_map.get(card2[1] if card2[0] in 'CDHS' else card2[0], 0)
                    
                    is_pocket_pair = (rank1 == rank2)
                    is_high_pair = is_pocket_pair and rank1 >= 10  # TT+
                    is_premium_pair = is_pocket_pair and rank1 >= 12  # QQ+
                    is_broadway = (rank1 >= 10 and rank2 >= 10)
                    
                    # ULTRA AGGRESSIVE with premium hands preflop
                    if to_call_bb_val <= 1e-6:  # First to act
                        p_check = probs_adjusted[CLASSES.index('check')].item()
                        
                        if is_premium_pair:  # QQ, KK, AA
                            if p_check > 0.05:
                                boost = p_check * 0.95
                                probs_adjusted[CLASSES.index('raise')] += boost
                                probs_adjusted[CLASSES.index('check')] -= boost
                                print(f"DEBUG: ULTRA AGGRO PREFLOP - premium pair {card1}{card2}")
                        elif is_high_pair:  # TT, JJ
                            if p_check > 0.10:
                                boost = p_check * 0.90
                                probs_adjusted[CLASSES.index('raise')] += boost
                                probs_adjusted[CLASSES.index('check')] -= boost
                                print(f"DEBUG: AGGRO PREFLOP - high pair {card1}{card2}")
                        elif (rank1 == 14 or rank2 == 14) and max(rank1, rank2) >= 11:  # AK, AQ, AJ
                            if p_check > 0.15:
                                boost = p_check * 0.85
                                probs_adjusted[CLASSES.index('raise')] += boost
                                probs_adjusted[CLASSES.index('check')] -= boost
                                print(f"DEBUG: AGGRO PREFLOP - ace high {card1}{card2}")
                        elif is_broadway:
                            if p_check > 0.25:
                                boost = p_check * 0.60
                                probs_adjusted[CLASSES.index('raise')] += boost
                                probs_adjusted[CLASSES.index('check')] -= boost
                    
                    # AGGRESSIVE 3-bet with premium hands
                    elif to_call_bb_val > 0 and to_call_bb_val <= 3.5:
                        p_call = probs_adjusted[CLASSES.index('call')].item()
                        
                        if is_premium_pair:  # QQ, KK, AA
                            if p_call > 0.30:
                                boost = p_call * 0.70
                                probs_adjusted[CLASSES.index('raise')] += boost
                                probs_adjusted[CLASSES.index('call')] -= boost
                                print(f"DEBUG: ULTRA AGGRO PREFLOP - 3bet premium {card1}{card2}")
                        elif is_high_pair or (rank1 == 14 and rank2 >= 13):  # TT+, AK
                            if p_call > 0.40:
                                boost = p_call * 0.50
                                probs_adjusted[CLASSES.index('raise')] += boost
                                probs_adjusted[CLASSES.index('call')] -= boost
                                print(f"DEBUG: AGGRO PREFLOP - 3bet strong {card1}{card2}")
                except:
                    pass
            
            # === POSTFLOP AGGRESSION - First to Act ===
            if to_call_bb_val <= 1e-6 and street_idx >= 1:
                p_check = probs_adjusted[CLASSES.index('check')].item()
                
                if hand_bucket >= 3.5:  # Nutted
                    if p_check > 0.10:
                        boost = p_check * 0.95
                        probs_adjusted[CLASSES.index('raise')] += boost
                        probs_adjusted[CLASSES.index('check')] -= boost
                        print(f"DEBUG: ULTRA AGGRO - nutted (bucket={hand_bucket:.1f})")
                elif hand_bucket >= 2.5:  # Strong
                    if p_check > 0.30:
                        boost = p_check * 0.80
                        probs_adjusted[CLASSES.index('raise')] += boost
                        probs_adjusted[CLASSES.index('check')] -= boost
                        print(f"DEBUG: AGGRO - strong (bucket={hand_bucket:.1f})")
                elif hand_bucket >= 1.5:  # Medium
                    if pot_bb <= 12.0 and p_check > 0.50:
                        boost = p_check * 0.50
                        probs_adjusted[CLASSES.index('raise')] += boost
                        probs_adjusted[CLASSES.index('check')] -= boost
                        print(f"DEBUG: AGGRO - medium (bucket={hand_bucket:.1f})")
                elif hand_bucket >= 0.5:  # Draws
                    if (has_flush_draw or has_straight_draw) and pot_bb >= 1.5 and p_check > 0.50:
                        boost = p_check * 0.55
                        probs_adjusted[CLASSES.index('raise')] += boost
                        probs_adjusted[CLASSES.index('check')] -= boost
                        print(f"DEBUG: AGGRO - semi-bluff (bucket={hand_bucket:.1f})")
            
            # === POSTFLOP AGGRESSION - Facing Bets ===
            if to_call_bb_val > 0 and street_idx >= 1:
                try:
                    idx_bet_frac = self.numeric_cols_order.index('bet_frac_of_pot')
                    bet_frac = float(numeric_vec_unscaled[0, idx_bet_frac])
                except:
                    bet_frac = 0.0
                
                if bet_frac <= 0.75:
                    p_fold = probs_adjusted[CLASSES.index('fold')].item()
                    p_call = probs_adjusted[CLASSES.index('call')].item()
                    
                    if hand_bucket >= 3.5:  # Nutted
                        if p_fold > 0.01:
                            boost_from_fold = p_fold * 0.98
                            probs_adjusted[CLASSES.index('raise')] += boost_from_fold * 0.8
                            probs_adjusted[CLASSES.index('call')] += boost_from_fold * 0.2
                            probs_adjusted[CLASSES.index('fold')] -= boost_from_fold
                        if p_call > 0.30:
                            boost = p_call * 0.70
                            probs_adjusted[CLASSES.index('raise')] += boost
                            probs_adjusted[CLASSES.index('call')] -= boost
                            print(f"DEBUG: ULTRA AGGRO - nutted defend (bucket={hand_bucket:.1f})")
                    elif hand_bucket >= 2.5:  # Strong
                        if p_fold > 0.10 and bet_frac <= 0.70:
                            boost = p_fold * 0.85
                            probs_adjusted[CLASSES.index('call')] += boost * 0.6
                            probs_adjusted[CLASSES.index('raise')] += boost * 0.4
                            probs_adjusted[CLASSES.index('fold')] -= boost
                            print(f"DEBUG: AGGRO - strong defend (bucket={hand_bucket:.1f})")
                    elif hand_bucket >= 1.5:  # Medium
                        if p_fold > p_call and bet_frac <= 0.60:
                            boost = (p_fold - p_call) * 0.65
                            probs_adjusted[CLASSES.index('call')] += boost
                            probs_adjusted[CLASSES.index('fold')] -= boost
                            print(f"DEBUG: AGGRO - medium defend (bucket={hand_bucket:.1f})")
            
            # === DEFENSIVE FOLDING HEURISTICS ===
            # Increase fold probability when facing raises with weak hands
            # Only trigger when facing actual aggression (not just blinds preflop)
            facing_real_bet = False
            if to_call_bb_val > 0:
                if street_idx == 0:  # Preflop
                    # Only consider it a real bet if it's bigger than just calling the BB
                    facing_real_bet = to_call_bb_val > 2.0  # More than 2BB (3-bet+)
                else:  # Postflop
                    # Any bet postflop is real aggression
                    facing_real_bet = to_call_bb_val > 0.5
            
            if facing_real_bet:
                try:
                    idx_stack = self.numeric_cols_order.index('stack_bb')
                    stack_bb = float(numeric_vec_unscaled[0, idx_stack])
                    
                    p_fold = probs_adjusted[CLASSES.index('fold')].item()
                    p_call = probs_adjusted[CLASSES.index('call')].item()
                    p_raise = probs_adjusted[CLASSES.index('raise')].item()
                    
                    should_fold = False
                    fold_reason = ""
                    
                    # 1. WEAK HAND FACING BIG RAISE
                    if hand_bucket < 1.5 and to_call_bb_val > pot_bb * 0.6:
                        if not (has_flush_draw or has_straight_draw):
                            # Weak hand, big raise, no draws -> FOLD
                            should_fold = True
                            fold_reason = f"weak hand (bucket={hand_bucket:.1f}) facing big raise ({to_call_bb_val:.1f}BB into {pot_bb:.1f}BB pot)"
                    
                    # 2. MEDIUM HAND FACING HUGE RAISE (more lenient)
                    elif hand_bucket >= 1.5 and hand_bucket < 2.0 and to_call_bb_val > pot_bb * 1.5:
                        # Medium hand facing massive overbet
                        should_fold = True
                        fold_reason = f"medium hand (bucket={hand_bucket:.1f}) facing huge overbet ({to_call_bb_val:.1f}BB into {pot_bb:.1f}BB pot)"
                    
                    # 3. FACING RAISE WHEN SHORT STACKED (more lenient)
                    elif stack_bb < 15.0 and to_call_bb_val > stack_bb * 0.4:
                        # Very short stack, big commitment needed
                        if hand_bucket < 1.5:
                            should_fold = True
                            fold_reason = f"short stack ({stack_bb:.1f}BB) with weak hand (bucket={hand_bucket:.1f})"
                    
                    # 4. FACING AGGRESSION ON RIVER WITH WEAK HAND
                    elif street_idx == 3 and hand_bucket < 1.5:  # River
                        if to_call_bb_val > pot_bb * 0.5:
                            # Weak hand facing river bet
                            should_fold = True
                            fold_reason = f"weak hand (bucket={hand_bucket:.1f}) on river facing {to_call_bb_val:.1f}BB bet"
                    
                    # 5. MISSED DRAWS ON RIVER
                    is_missed_draw = treys_feats_live.get("is_missed_draw_river", 0)
                    if street_idx == 3 and is_missed_draw:  # River with missed draw
                        if to_call_bb_val > pot_bb * 0.4:
                            # Missed draw, facing bet
                            should_fold = True
                            fold_reason = "missed draw on river facing bet"
                    
                    # Apply fold boost if criteria met
                    if should_fold:
                        # Shift probability from call/raise to fold (but not as extreme)
                        fold_boost = p_call * 0.50 + p_raise * 0.40
                        
                        probs_adjusted[CLASSES.index('fold')] += fold_boost
                        probs_adjusted[CLASSES.index('call')] = p_call * 0.50
                        probs_adjusted[CLASSES.index('raise')] = p_raise * 0.60
                        
                        print(f"DEBUG: DEFENSIVE FOLD - {fold_reason}")
                
                except Exception as e:
                    print(f"DEBUG: Defensive fold exception ({e})")
            
            # Renormalize
            total = sum(probs_adjusted[CLASSES.index(a)].item() for a in allowed_actions)
            if total > 1e-9:
                for a in allowed_actions:
                    probs_adjusted[CLASSES.index(a)] /= total
            
            probs = probs_adjusted
        except Exception as e:
            print(f"DEBUG: Aggression exception ({e})")
        
        # Choose best action
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
    position = 'BTN' if player_id == 1 else 'BB'
    
    if _tracker is None:
        _tracker = LiveHandTracker(my_position=position, big_blind_size=1.0)
    
    # Detect BB on first hand
    if _bb_normalizer is None:
        hand_id = cv_json.get('hand_id', 1)
        if hand_id == 1 and cv_json.get('round', '').lower() == 'preflop':
            preflop_pot = float(cv_json.get('pot_bb', 0.0))
            if preflop_pot > 0:
                _bb_normalizer = preflop_pot / 1.5
            else:
                stack_chips = float(cv_json.get('stack_bb', 175.0))
                _bb_normalizer = stack_chips / 17.5
        else:
            _bb_normalizer = 10.0
    
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
    
    # Normalize to BB
    pot_bb = float(cv_json.get('pot_bb', 0.0)) / _bb_normalizer
    to_call_bb = float(cv_json.get('to_call_bb', 0.0)) / _bb_normalizer
    stack_bb = float(cv_json.get('stack_bb', 0.0)) / _bb_normalizer
    opp_stack_bb = float(cv_json.get('opp_stack_bb', 0.0)) / _bb_normalizer
    
    tracker_data = {
        'hole_cards': hole_cards,
        'board_cards': board_cards,
        'my_stack_chips': stack_bb,
        'opp_stack_chips': opp_stack_bb,
        'pot_chips': pot_bb,
        'to_call_chips': to_call_bb,
        'last_bet_size_chips': to_call_bb,
        'action_sequence': cv_json.get('action_history', []),
        'my_player_name': 'PlayerCoach'
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
