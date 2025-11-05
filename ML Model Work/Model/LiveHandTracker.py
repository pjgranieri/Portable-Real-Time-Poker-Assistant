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
MODEL_DIR = "./runs/poker_mlp_v1"
MODEL_PATH = os.path.join(MODEL_DIR, "model.best.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
META_PATH = os.path.join(MODEL_DIR, "meta.json")


# Card encoding
RANKS = "23456789TJQKA"
SUITS = "cdhs"
CARD2IDX = {f"{r}{s}": i for i, (r, s) in enumerate((r, s) for r in RANKS for s in SUITS)}
NUM_CARDS = 52

def card_one_hot(card: str) -> np.ndarray:
    v = np.zeros(NUM_CARDS, dtype=np.float32)
    if isinstance(card, str):
        c = card.strip()
        if len(c) == 2:
            # Handle both 'SA' and 'As' formats
            if c[0] in RANKS and c[1] in SUITS:
                card_key = c
            elif c[1] in RANKS and c[0] in SUITS:
                card_key = c[1] + c[0].lower()
            else:
                card_key = None
            
            if card_key and card_key in CARD2IDX:
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
        # Build numeric features (must match training order)
        
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
        
        # Calculate bet sizing as fraction of pot (key indicator)
        pot_before_bet = pot_chips - last_bet_chips
        pot_bb_before = pot_before_bet / self.bb_size if self.bb_size > 0 else 0.0
        raise_size_bb = raise_to_bb - to_call_bb
        bet_frac_of_pot = raise_size_bb / pot_bb_before if pot_bb_before > 1e-9 else 0.0
        
        # Calculate position & game state features
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

        # Assemble feature map (only the 15 features model uses)
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
        call_idx = CLASSES.index("call")
        logits[0, call_idx] = logits[0, call_idx] + Cfg.call_logit_bias

        probs = torch.softmax(logits / self.learned_temperature, dim=1).squeeze()
        
        p_fold = probs[CLASSES.index("fold")].item()
        p_nonfold_max = max(probs[i].item() for i, name in enumerate(CLASSES) if name != "fold")
        margin = p_fold - p_nonfold_max
        
        # Apply fold-gating policy
        if p_fold >= Cfg.fold_thresh and margin >= Cfg.fold_margin:
            action_index = CLASSES.index("fold")
        else:
            action_index = torch.argmax(probs).item()
            
        action_name = CLASSES[action_index]
        
        # Format probabilities
        prob_dict = {name: f"{p.item():.4f}" for name, p in zip(CLASSES, probs)}
        
        print(f"Unscaled Features: {numeric_vec_unscaled.round(2)}")
        print(f"Scaled Features: {numeric_vec_scaled.round(2)}")
        print(f"Probs (T={self.learned_temperature:.3f}): {prob_dict}")
        print(f"Policy: p_fold={p_fold:.3f}, margin={margin:.3f}")
        print(f"==> FINAL ACTION: {action_name}")
        
        return action_name, prob_dict


# Global tracker instance
_tracker = None


def main(cv_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for CV system integration.
    Call this function with CV JSON data to get poker action recommendation.
    
    CV JSON format:
    {
        "position": "BTN",              # Your seat (BTN, SB, BB, CO, etc.)
        "big_blind": 1.0,               # Big blind size
        "hole_cards": ["As", "Kh"],     # Your 2 cards
        "board_cards": ["Qh", "Jc"],    # Community cards (0-5 cards)
        "pot": 15.5,                    # Pot size in chips
        "to_call": 5.0,                 # Amount to call
        "your_stack": 100.0,            # Your stack
        "opponent_stack": 85.0,         # Opponent's stack
        "last_bet": 10.0,               # Last bet/raise size
        "action_history": [...]         # Optional action sequence
    }
    
    Returns:
    {
        "action": "raise_m",            # Recommended action
        "confidence": 0.85,             # Confidence (0-1)
        "probabilities": {              # All action probabilities
            "fold": 0.05,
            "check": 0.10,
            "call": 0.15,
            "raise_s": 0.20,
            "raise_m": 0.35,
            "raise_l": 0.15
        }
    }
    """
    global _tracker
    
    # Initialize tracker on first call or if position/blinds changed
    position = cv_json.get('position', 'BTN')
    bb_size = cv_json.get('big_blind', 1.0)
    
    if _tracker is None:
        _tracker = LiveHandTracker(my_position=position, big_blind_size=bb_size)
    
    # Convert CV JSON to tracker format
    tracker_data = {
        'hole_cards': cv_json.get('hole_cards', []),
        'board_cards': cv_json.get('board_cards', []),
        'my_stack_chips': float(cv_json.get('your_stack', 0.0)),
        'opp_stack_chips': float(cv_json.get('opponent_stack', 0.0)),
        'pot_chips': float(cv_json.get('pot', 0.0)),
        'to_call_chips': float(cv_json.get('to_call', 0.0)),
        'last_bet_size_chips': float(cv_json.get('last_bet', 0.0)),
        'action_sequence': cv_json.get('action_history', []),
        'my_player_name': cv_json.get('player_name', 'hero')
    }
    
    # Update tracker with game state
    _tracker.update_state_from_cv(tracker_data)
    
    # Get prediction
    action, prob_dict = _tracker.predict_action()
    
    # Convert probabilities to floats
    probs_float = {k: float(v) for k, v in prob_dict.items()}
    confidence = max(probs_float.values())
    
    return {
        'action': action,
        'confidence': confidence,
        'probabilities': probs_float
    }


def reset_hand():
    """Call this when starting a new hand"""
    global _tracker
    if _tracker:
        _tracker.reset_hand()


def new_session(position: str, big_blind: float):
    """Call this when position or blinds change"""
    global _tracker
    _tracker = LiveHandTracker(my_position=position, big_blind_size=big_blind)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("LiveHandTracker - Example Usage")
    print("=" * 60)
    
    # Example 1: Preflop with pocket aces
    cv_data = {
        "position": "BTN",
        "big_blind": 1.0,
        "hole_cards": ["As", "Ah"],
        "board_cards": [],
        "pot": 1.5,
        "to_call": 1.0,
        "your_stack": 100.0,
        "opponent_stack": 100.0,
        "last_bet": 1.0
    }
    
    print("\n[Example 1] Preflop with AA on BTN")
    print("-" * 60)
    result = main(cv_data)
    print(f"Recommended Action: {result['action']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Probabilities: {result['probabilities']}")
    
    # Example 2: Flop with top pair
    reset_hand()
    cv_data2 = {
        "position": "BTN",
        "big_blind": 1.0,
        "hole_cards": ["As", "Kh"],
        "board_cards": ["Ah", "Qc", "7d"],
        "pot": 10.0,
        "to_call": 5.0,
        "your_stack": 95.0,
        "opponent_stack": 95.0,
        "last_bet": 5.0
    }
    
    print("\n[Example 2] Flop with top pair + top kicker")
    print("-" * 60)
    result = main(cv_data2)
    print(f"Recommended Action: {result['action']}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    print("\n" + "=" * 60)
    print("Integration ready! Call main(cv_json) from your CV code.")
    print("=" * 60)