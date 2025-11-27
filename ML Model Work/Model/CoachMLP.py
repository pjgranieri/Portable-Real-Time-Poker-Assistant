"""
PokerMLP Training with Action Masking

Key Features:
- Action Masking: Enforces legal move constraints during training
  * check only legal when to_call == 0 (no bet to face)
  * call only legal when to_call > 0 (bet to face)
  * fold and raise always legal
- Label Smoothing: Reduces overconfidence (default 0.15)
- Hand Bucket Diversity Regularization: Ensures weak hands fold more than strong hands
- Temperature Calibration: Post-training probability calibration
- Multi-player Support: 2-4 player games with proper position encoding

The action masking prevents the model from learning illegal moves by setting
illegal action logits to -inf before softmax, ensuring zero probability.
"""

import os, glob, json, random, time, joblib, gc, csv, math
from functools import partial
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

class Cfg:
    seed = 42

    bot_glob  = ""
    real_glob = ""
    gto_glob  = r"C:\Users\nickl\OneDrive\Documents\Computer-Vision-Powered-AI-Poker-Coach\ML Model Work\Model\Poker Data Final (Multi-Player)\data(gto_v2)_*.csv"
    gto_v3_glob = ""
    # Multi-player synthetic data
    multiplayer_glob = r"C:\Users\nickl\OneDrive\Documents\Computer-Vision-Powered-AI-Poker-Coach\ML Model Work\Model\Poker Data Final (Multi-Player)\data_*player_synthetic_aggressive.csv"

    use_focal = False  # Enable focal loss to handle class imbalance
    focal_gamma = 1.2  # Reduced from 2.0 to reduce over-confidence
    label_smoothing = 0.15  # Increased from 0.1 to soften targets and reduce overconfidence

    batch_size   = 1024
    lr           = 1e-3
    lr_min       = 1e-8
    weight_decay = 1e-4
    max_epochs   = 100   
    val_split    = 0.10 
    num_workers  = 8     
    patience     = 8

    streets   = ["preflop", "flop", "turn", "river"]
    positions = ["Early", "Late", "Blinds"]

    # Updated to 4 classes: merged raise_s/raise_m/raise_l into single 'raise'
    classes = ["fold", "check", "call", "raise"]

    weight_bot_preflop    = 1.0
    weight_real_preflop   = 1.0
    weight_bot_postflop   = 1.0 
    weight_real_postflop  = 1.0 
    base_weight           = 1.0 

    out_dir = os.getenv("OUT_DIR", "./runs/poker_mlp_v1")

    fold_thresh     = 0.00
    fold_margin     = 0.00
    call_logit_bias = 0.00

    focal_gamma = 1.8

    fold_weight_multiplier = 1.0 
    call_weight_multiplier = 1.0 
COLUMN_NAMES = {
    "source": "source",            
    "street": "round",              
    "hole1": "hole1",
    "hole2": "hole2",
    "board": ["flop1", "flop2", "flop3", "turn", "river"],

    "numeric_pref": [
        "pot_bb",
        "to_call_bb",
        "stack_bb",
        "opp_stack_bb",
        "raise_to_bb",
        "bet_frac_of_pot",
        "in_position",
        "was_pfr",
        "street_index",
        "raises_this_street",
        # Hand strength features (added from Treys evaluator)
        "hand_bucket",
        "has_flush_draw",
        "has_straight_draw",
        "has_combo_draw",
        "is_missed_draw_river",
        # Multi-player features (added for 3-4 player support)
        "num_active_players",
        "num_players_to_act",
        "multiway_pot_flag",
        "position_relative",
        # "num_callers_this_street",  # Removed: all zeros in training data
        "num_raisers_this_street",
        "avg_opp_stack_bb",
    ],

    "categorical_pref": [
        "pot_type",
        "board_texture",
    ],

    "position": "pos",
    "action": "action",

    "whos_turn": None,

    "weight_cols": [
    ],

    # Columns that are outcome
    "leakage_cols": [
        "result_bb",
        "won_flag",
        "final_pot_bb",
        "source",
        "style",
        "player_id"
    ]
}

# Utils
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

RANKS = "23456789TJQKA"
SUITS = "cdhs"
CARD2IDX = {f"{r}{s}": i for i, (r, s) in enumerate((r, s) for r in RANKS for s in SUITS)}
NUM_CARDS = 52

def card_one_hot(card: str) -> np.ndarray:
    v = np.zeros(NUM_CARDS, dtype=np.float32)
    if isinstance(card, str):
        c = card.strip()
        if len(c) == 2 and c[0] in RANKS and c[1] in SUITS:
            v[CARD2IDX[c]] = 1.0
    return v

def one_hot(value: str, vocab: List[str]) -> np.ndarray:
    v = np.zeros(len(vocab), dtype=np.float32)
    if value in vocab:
        v[vocab.index(value)] = 1.0
    return v

def safe_float(x) -> float:
    try:
        return float(x)
    except:
        return 0.0

def sanitize_numeric_df(df: pd.DataFrame, cols, clip_abs=1e6, tag="(unspecified)"):
    """Ensure all numerics are finite float32 and reasonably bounded."""
    bad_counts = {}
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")
        bad_counts[c] = {
            "nan": int(df[c].isna().sum()),
            "+inf": int(np.isposinf(df[c]).sum()),
            "-inf": int(np.isneginf(df[c]).sum()),
        }
        if c in ["stack_bb", "opp_stack_bb"]:
            df[c] = df[c].replace(np.inf, 1000.0)
            df[c] = df[c].replace(-np.inf, 0.0)
        else:
            df[c] = df[c].replace(np.inf, clip_abs)
            df[c] = df[c].replace(-np.inf, -clip_abs)
        df[c] = df[c].fillna(0.0)
        df[c] = df[c].clip(lower=-clip_abs, upper=clip_abs).astype(np.float32)

    print(f"[sanitize_numeric_df] Dataset {tag}:")
    for c, cnts in bad_counts.items():
        if any(v > 0 for v in cnts.values()):
            print(f"   - {c}: NaN={cnts['nan']} +Inf={cnts['+inf']} -Inf={cnts['-inf']}")
    return df

# Dataset
class PokerDataset(Dataset):
    def __init__(self, paths: List[str]):
        rows = []
        total_files = len(paths)
        print(f"\n[PokerDataset] Loading {total_files} CSV files...")
        
        for file_idx, p in enumerate(paths, 1):
            try:
                # Progress indicator every 50 files
                if file_idx % 50 == 0 or file_idx == total_files:
                    print(f"  Loaded {file_idx}/{total_files} files ({file_idx*100//total_files}%)...")
                
                sniffer = csv.Sniffer()
                with open(p, 'r', encoding='utf-8') as f:
                    try:
                        dialect = sniffer.sniff(f.read(1024 * 5))
                        delimiter = dialect.delimiter
                    except csv.Error:
                        delimiter = ','

                #  Read/ normalize
                df = pd.read_csv(p, delimiter=delimiter, encoding="utf-8-sig", low_memory=False)
                df.columns = (
                    df.columns.astype(str)
                      .str.replace("\ufeff", "", regex=False)
                      .str.strip()
                      .str.lower()
                      .str.replace(r"\s+", "_", regex=True)
                )

                if "round" in df.columns and df["round"].dtype != object:
                    df["round"] = df["round"].map({0: "preflop", 1: "flop", 2: "turn", 3: "river"})

                if "pos" not in df.columns:
                    for alt in ("position", "hero_pos", "player_pos", "seat_pos"):
                        if alt in df.columns:
                            df["pos"] = df[alt]
                            break
                if "pos" not in df.columns:
                    raise ValueError(f"Missing position column in {p}")

                POS_ALIASES = {
                    "sb": "Blinds", "small_blind": "Blinds", "bb": "Blinds", "big_blind": "Blinds",
                    "utg": "Early", "utg1": "Early", "mp": "Early", "mp1": "Early", "ep": "Early", "middle": "Early",
                    "co": "Late", "cutoff": "Late", "btn": "Late", "button": "Late",
                    "early": "Early", "late": "Late", "blinds": "Blinds",
                }
                def _canon_pos(x):
                    if x is None or (isinstance(x, float) and np.isnan(x)):
                        raise ValueError(f"Invalid position value: {x} in {p}")
                    s = str(x).strip().lower()
                    if s in ("early", "late", "blinds"):
                        return s.capitalize()
                    if s in POS_ALIASES:
                        return POS_ALIASES[s]
                    raise ValueError(f"Unknown position: {x} in {p}")
                df["pos"] = df["pos"].map(_canon_pos)

                if "in_position" in df.columns:
                    def _to01(v):
                        if v is None: return 0
                        if isinstance(v, str):
                            vv = v.strip().lower()
                            if vv in ("1","true","t","yes","y"): return 1
                            if vv in ("0","false","f","no","n",""): return 0
                        try:
                            return int(float(v) != 0.0)
                        except Exception:
                            return 0
                    df["in_position"] = df["in_position"].map(_to01).astype(np.float32)

                # REMOVED: progress_in_hand computation (redundant with street_index)


                expanded_cat_cols = []
                for cat in COLUMN_NAMES.get("categorical_pref", []):
                    if cat in df.columns:
                        df[cat] = df[cat].astype(str).str.strip().str.lower()
                        dummies = pd.get_dummies(df[cat], prefix=cat, dummy_na=False)
                        for c in dummies.columns:
                            df[c] = dummies[c].astype(np.float32)
                        expanded_cat_cols.extend(list(dummies.columns))

            except Exception as e:
                print(f"[WARN] Skip {p}: {e}")
                continue

            # Ensure expected label 
            street_col_name = COLUMN_NAMES["street"]
            if street_col_name in df.columns:
                df[street_col_name] = df[street_col_name].astype(str).str.lower()
            else:
                print(f"[WARN] Missing '{street_col_name}' in {p}, skipping file.")
                continue

            pos_col_name = COLUMN_NAMES["position"]
            if pos_col_name in df.columns:
                # Keep proper case for 3-position system (Early, Late, Blinds)
                # Don't use .upper() as it would break the vocab matching
                df[pos_col_name] = df[pos_col_name].astype(str)
            else:
                raise ValueError(f"Missing '{pos_col_name}' column in {p}")

            # Card columns
            for k in ["hole1", "hole2"]:
                col = COLUMN_NAMES[k]
                if col in df.columns:
                    df[col] = df[col].fillna("").astype(str)
                    df[col] = df[col].replace("nan", "")
                else:
                    df[col] = ""
            for b_col in COLUMN_NAMES["board"]:
                if b_col in df.columns:
                    df[b_col] = df[b_col].fillna("").astype(str)
                    df[b_col] = df[b_col].replace("nan", "")
                else:
                        df[b_col] = ""

            # fill missing
            # Now ensure ALL numeric_pref exist and are numeric
            for num_col in COLUMN_NAMES["numeric_pref"]:
                if num_col not in df.columns:
                    df[num_col] = 0.0
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0.0)


            # Action label
            action_col_name = COLUMN_NAMES["action"]
            if action_col_name not in df.columns:
                print(f"[WARN] {p} has no '{action_col_name}', skipping.")
                continue
            df[action_col_name] = df[action_col_name].astype(str).str.lower().str.strip()

            # Infer simple source tag from filename 
            fname_lower = os.path.basename(p).lower()
            src = "bot" if "bot" in fname_lower else ("real" if "real" in fname_lower else "unknown")
            df["__source"] = src

            # Attach list of expanded categorical columns 
            if expanded_cat_cols:
                df["__expanded_cat_cols"] = ",".join(expanded_cat_cols)
            else:
                df["__expanded_cat_cols"] = ""

            rows.append(df)

        if not rows:
            raise RuntimeError("No CSVs loaded. Check your paths and column names.")

        # Concat all files
        self.df = pd.concat(rows, ignore_index=True)

        # Filter known streets only
        street_col = COLUMN_NAMES["street"]
        self.df = self.df[self.df[street_col].isin(Cfg.streets)].reset_index(drop=True)

        all_expanded = set()
        if "__expanded_cat_cols" in self.df.columns:
            for s in self.df["__expanded_cat_cols"].tolist():
                if s:
                    all_expanded.update(s.split(","))
            self.df.drop(columns=["__expanded_cat_cols"], inplace=True, errors="ignore")
        expanded_cat_cols = sorted(list(all_expanded))

        self.street_vocab = Cfg.streets
        self.pos_vocab    = Cfg.positions  

        # Vectorize labels (simplified for 4-class model)
        print("[PokerDataset] Vectorizing labels...")
        act_col = self.df[COLUMN_NAMES["action"]].astype(str).str.lower().str.strip()
        default_choice = Cfg.classes.index("check")
        
        # Map all action strings to class indices
        conditions = [
            act_col == "fold",
            act_col == "check",
            act_col == "call",
            act_col.isin(["raise", "bet", "allin", "raise_s", "raise_m", "raise_l"])  # All raise variants → 'raise'
        ]
        choices = [
            Cfg.classes.index("fold"),
            Cfg.classes.index("check"),
            Cfg.classes.index("call"),
            Cfg.classes.index("raise")
        ]
        
        self.df["__label"] = np.select(conditions, choices, default=default_choice).astype(np.int64)
        print("[PokerDataset] Labels vectorized.")

        have_nums = [c for c in COLUMN_NAMES["numeric_pref"] if c in self.df.columns]
        
        self.numeric_cols = have_nums
        self.binary_cols = expanded_cat_cols

        # Sanitize numerics
        self.df = sanitize_numeric_df(self.df, self.numeric_cols, clip_abs=1_000_000, tag="PokerDataset")

        # Encode static features
        print("[PokerDataset] Encoding static features (cards, street, pos)...")
        enc = [self._row_static_encoding(row) for _, row in self.df.iterrows()]
        self.static_mat = np.stack([e[0] for e in enc], axis=0).astype(np.float16)
        self.card_dim   = enc[0][1]
        self.static_dim = self.static_mat.shape[1]
        print(f"[PokerDataset] Static features encoded. Shape: {self.static_mat.shape}")

        # Sample weights
        print("[PokerDataset] Vectorizing sample weights (non-leaky)...")
        w = np.ones(len(self.df), dtype=np.float32)
        for wc in COLUMN_NAMES.get("weight_cols", []):
            if wc in self.df.columns:
                try:
                    w *= self.df[wc].astype(float).values
                except Exception:
                    pass
        # Clip 
        w = np.clip(w, 0.25, 4.0).astype(np.float32)
        self.sample_w = w
        print("[PokerDataset] Sample weights vectorized.")

        # Final tensors
        print("[PokerDataset] Extracting data to NumPy arrays...")
        self.y = self.df["__label"].values.astype(np.int64)

        scaled_numerics = self.df[self.numeric_cols].values.astype(np.float16)
        
        # DEPLOYMENT FIX: Ensure all expected binary columns exist
        # Model expects specific one-hot encoded categorical features
        expected_binary_cols = [
            "board_texture_dry",
            "board_texture_wet",
            "pot_type_3bet",
            "pot_type_4bet",
            "pot_type_limped",
            "pot_type_single_raised",
            "pot_type_three_bet",
            "pot_type_unopened"
        ]
        
        # Add missing binary columns with zeros
        for col in expected_binary_cols:
            if col not in self.df.columns:
                self.df[col] = 0.0
        
        # Update binary_cols to include all expected columns
        self.binary_cols = expected_binary_cols
        
        if self.binary_cols:
            binary_features = self.df[self.binary_cols].values.astype(np.float16)
            binary_features = np.nan_to_num(binary_features, nan=0.0).astype(np.float16)
        else:
            binary_features = np.empty((len(self.df), 0), dtype=np.float16)
        
        self.numeric_mat = np.concatenate([scaled_numerics, binary_features], axis=1)

        print(f"[PokerDataset] Numeric matrix created. Shape: {self.numeric_mat.shape}")
        print(f"  - Scaled features: {len(self.numeric_cols)} ({self.numeric_cols[:5]}...)")
        print(f"  - Binary features (unscaled): {len(self.binary_cols)}")
        
        # Print dataset summary
        print(f"\n{'='*80}")
        print(f"DATASET SUMMARY")
        print(f"{'='*80}")
        print(f"Total examples: {len(self.y):,}")
        print(f"Static features: {self.static_dim} dims")
        print(f"Numeric features: {self.numeric_mat.shape[1]} dims")
        print(f"  - Continuous (scaled): {len(self.numeric_cols)}")
        print(f"  - Binary (unscaled): {len(self.binary_cols)}")
        print(f"\nClass distribution:")
        unique, counts = np.unique(self.y, return_counts=True)
        for cls_idx, count in zip(unique, counts):
            cls_name = Cfg.classes[cls_idx] if cls_idx < len(Cfg.classes) else f"unknown_{cls_idx}"
            pct = count * 100.0 / len(self.y)
            print(f"  {cls_name:10s}: {count:7,} ({pct:5.2f}%)")
        print(f"{'='*80}\n")
        
        print("[PokerDataset] Deleting DataFrame to free memory...")
        del self.df
        gc.collect()
        print("[PokerDataset] Dataset ready for training!\n")

        # Scaler will be set during fit_scaler
        self.scaler = None

    def _encode_cards(self, row) -> np.ndarray:
        h1 = row.get(COLUMN_NAMES["hole1"], "")
        h2 = row.get(COLUMN_NAMES["hole2"], "")
        boards = [row.get(b, "") for b in COLUMN_NAMES["board"]]
        vecs = [card_one_hot(h1), card_one_hot(h2)]
        for c in boards:
            vecs.append(card_one_hot(c))
        return np.concatenate(vecs, axis=0).astype(np.float32)

    def _row_static_encoding(self, row) -> Tuple[np.ndarray, int]:
        cards = self._encode_cards(row)
        street_oh = one_hot(row.get(COLUMN_NAMES["street"], ""), self.street_vocab)
        pos_val = row.get(COLUMN_NAMES["position"])
        if pos_val not in self.pos_vocab:
            raise ValueError(f"Invalid position '{pos_val}' not in vocabulary {self.pos_vocab}")
        pos_oh = one_hot(pos_val, self.pos_vocab)
        static = np.concatenate([cards, street_oh, pos_oh], axis=0).astype(np.float32)
        return static, cards.shape[0]

    def fit_scaler(self, idxs: np.ndarray):
        if idxs.size == 0:
            print("[fit_scaler] Warning: No training indices provided for scaler fitting. Scaler will not be fitted.")
            self.scaler = None
            return
        n_scaled_cols = len(self.numeric_cols)
        X = self.numeric_mat[idxs, :n_scaled_cols].astype(np.float32)
        n_nonfinite = int(np.sum(~np.isfinite(X)))
        if n_nonfinite > 0:
            print(f"[fit_scaler] Warning: found {n_nonfinite} non-finite numeric values in training slice. Fixing...")
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

        if X.shape[0] == 0:
            print("[fit_scaler] Warning: Training data for scaler is empty after filtering. Scaler will not be fitted.")
            self.scaler = None
            return

        print("[fit_scaler] Numeric column stats (train split):")
        for j, cname in enumerate(self.numeric_cols):
            col = X[:, j]
            try:
                p1 = float(np.percentile(col, 1))
                p99 = float(np.percentile(col, 99))
                mean_val = col.mean()
                std_val = col.std()
                print(f"   {cname:>24} | mean={mean_val:8.4f} std={std_val:8.4f} p1={p1:8.4f} p99={p99:8.4f}")
            except Exception:
                print(f"   {cname:>24} | (stats unavailable)")

        # Fit scaler
        if np.all(np.var(X, axis=0) < 1e-9):
            print("[fit_scaler] Warning: Data has zero variance. Using identity scaler.")
            from sklearn.base import BaseEstimator, TransformerMixin
            class IdentityScaler(BaseEstimator, TransformerMixin):
                def __init__(self, n_features):
                    self.n_features_in_ = int(n_features)
                def fit(self, X, y=None): return self
                def transform(self, X): return X
            self.scaler = IdentityScaler(n_features=X.shape[1])
        else:
            try:
                self.scaler = StandardScaler().fit(X)
                print("[fit_scaler] Scaler fitted.")
            except ValueError as e:
                print(f"[fit_scaler] Error fitting StandardScaler: {e}. Using identity scaler.")
                from sklearn.base import BaseEstimator, TransformerMixin
                class IdentityScaler(BaseEstimator, TransformerMixin):
                    def __init__(self, n_features):
                        self.n_features_in_ = int(n_features)
                    def fit(self, X, y=None): return self
                    def transform(self, X): return X
                self.scaler = IdentityScaler(n_features=X.shape[1])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        static = self.static_mat[idx].astype(np.float32)
        numeric = self.numeric_mat[idx].astype(np.float32)
        y = self.y[idx]
        w = self.sample_w[idx]
        return static, numeric, y, w


# ----------------------------
# Model
# ----------------------------
class PokerMLP(nn.Module):
    def __init__(self, static_dim: int, numeric_dim: int, num_classes: int, deep=False):
        super().__init__()
        
        if deep:
            # DEEPER ARCHITECTURE: More layers for complex pattern learning
            # Static features: cards, position, action history
            self.static_net = nn.Sequential(
                nn.Linear(static_dim, 512),      # 512 neurons (2x original)
                nn.ReLU(),
                nn.BatchNorm1d(512),             # Batch norm helps deep networks
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.15),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            # Numeric features: pot, stack, SPR, bet sizing
            self.numeric_net = nn.Sequential(
                nn.Linear(numeric_dim, 128),     # 128 neurons (2x original)
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.15),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU()
            )
            
            # Classifier: combines static + numeric features
            self.classifier = nn.Sequential(
                nn.Linear(160, 256),             # 160 = 128 (static) + 32 (numeric)
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.15),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, num_classes)
            )
        else:
            # ORIGINAL ARCHITECTURE: Proven to work well
            self.static_net = nn.Sequential(
                nn.Linear(static_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            self.numeric_net = nn.Sequential(
                nn.Linear(numeric_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU()
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(160, 128),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(128, num_classes)
            )

    def forward(self, static_x, numeric_x):
        static_features = self.static_net(static_x)
        numeric_features = self.numeric_net(numeric_x)
        x = torch.cat([static_features, numeric_features], dim=1)
        return self.classifier(x)

# --- Focal loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, reduction='mean'):
        super().__init__()
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        logp = torch.log_softmax(logits, dim=1)
        p = logp.exp()
        idx = torch.arange(logits.size(0), device=logits.device)
        target_clamped = torch.clamp(target, 0, len(self.alpha) - 1)
        pt = p[idx, target_clamped]
        at = self.alpha[target_clamped]
        epsilon = 1e-9
        logpt = logp[idx, target_clamped]
        loss = -at * (1 - pt + epsilon).pow(self.gamma) * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss 

def collate_with_scaler(batch, scaler: Optional[StandardScaler], n_scaled_cols: int = None, n_binary_cols: int = None):
    statics, numerics, ys, ws = zip(*batch)
    statics = torch.from_numpy(np.stack(statics, axis=0)).float()
    numerics_np = np.stack(numerics, axis=0).astype(np.float32)
    
    if n_scaled_cols is not None and n_binary_cols is not None and n_binary_cols > 0:
        scaled_part = numerics_np[:, :n_scaled_cols]
        binary_part = numerics_np[:, n_scaled_cols:]
        
        if scaler is not None and hasattr(scaler, 'transform'):
            try:
                scaled_part = scaler.transform(scaled_part)
            except Exception as e:
                print(f"[collate_with_scaler] Error applying scaler transform: {e}. Using raw numerics.")
        
        # Recombine: scaled features + unscaled binary features
        numerics_np = np.concatenate([scaled_part, binary_part], axis=1)
    else:
        # No binary columns, scale everything (backward compatible)
        if scaler is not None and hasattr(scaler, 'transform'):
            try:
                numerics_np = scaler.transform(numerics_np)
            except Exception as e:
                print(f"[collate_with_scaler] Error applying scaler transform: {e}. Using raw numerics.")
    
    numerics = torch.from_numpy(numerics_np).float()
    y = torch.tensor(ys, dtype=torch.long)
    w = torch.tensor(ws, dtype=torch.float32)
    return statics, numerics, y, w


def create_action_mask(to_call_bb, num_classes=4):
  
    batch_size = to_call_bb.size(0)
    mask = torch.ones((batch_size, num_classes), dtype=torch.bool, device=to_call_bb.device)
    
    facing_bet = to_call_bb > 0.01  # Small epsilon for floating point
    
    # fold (idx 0): only legal when facing bet
    mask[:, 0] = facing_bet
    # check (idx 1): only legal when NOT facing bet
    mask[:, 1] = ~facing_bet
    # call (idx 2): only legal when facing bet
    mask[:, 2] = facing_bet
    # raise (idx 3): always legal
    
    return mask


def apply_action_mask(logits, mask):
    """
    Apply action legality mask to logits by setting illegal actions to very negative value.
    
    Args:
        logits: Raw model outputs (batch_size, num_classes)
        mask: Boolean mask (batch_size, num_classes) where True = legal
    
    Returns:
        Masked logits with illegal actions set to -100 (avoids numerical instability)
    """
    masked_logits = logits.clone()
    # Use -100 instead of -1e9 to avoid overflow in log_softmax
    # After softmax, exp(-100) ≈ 3.7e-44 which is effectively zero
    masked_logits[~mask] = -100.0
    return masked_logits


def apply_label_smoothing(y_true, num_classes, smoothing=0.0):
    """
    Apply label smoothing to target labels.
    
    Args:
        y_true: Long tensor of class indices (batch_size,)
        num_classes: Number of classes
        smoothing: Smoothing parameter (0.0 = no smoothing, 0.1 = 10% smoothing)
    
    Returns:
        Smoothed probability distribution (batch_size, num_classes)
    
    Formula: smooth_target = (1 - smoothing) * one_hot + smoothing / num_classes
    """
    if smoothing == 0.0:
        return torch.nn.functional.one_hot(y_true, num_classes).float()
    
    # Use one_hot which creates a proper tensor, then apply smoothing
    one_hot = torch.nn.functional.one_hot(y_true, num_classes).float()
    smooth_dist = one_hot * (1.0 - smoothing) + smoothing / num_classes
    return smooth_dist


@torch.no_grad()
def eval_loop(model, loader, device, loss_fn, use_label_smoothing=False, label_smoothing=0.0, num_classes=4, 
              scaler=None, to_call_idx=1, use_action_masking=False):
    """Evaluation loop with optional label smoothing and action masking to match training loss calculation."""
    model.eval()
    total, correct = 0, 0
    losses = []
    all_y, all_p = [], []
    illegal_predictions = 0
    
    for statics, numerics, y, w in loader:
        statics, numerics = statics.to(device), numerics.to(device)
        y, w = y.to(device), w.to(device)
        logits = model(statics, numerics)
        
        # Apply action masking if requested
        if use_action_masking and scaler is not None:
            # Extract to_call_bb from scaled numerics
            to_call_scaled = numerics[:, to_call_idx]
            to_call_bb = to_call_scaled * scaler.scale_[to_call_idx] + scaler.mean_[to_call_idx]
            
            # Create and apply mask
            mask = create_action_mask(to_call_bb, num_classes)
            logits = apply_action_mask(logits, mask)
            
            # Track illegal predictions (before masking)
            pred_before_mask = model(statics, numerics).argmax(1)
            illegal_predictions += (~mask[torch.arange(mask.size(0)), pred_before_mask]).sum().item()
 
        # Apply label smoothing if requested (to match training)
        if use_label_smoothing and label_smoothing > 0:
            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            smooth_targets = apply_label_smoothing(y, num_classes, label_smoothing)
            loss_vec = -(smooth_targets * log_probs).sum(dim=1)
        else:
            # Standard cross-entropy (raw loss)
            loss_vec = loss_fn(logits, y)
        
        loss = (loss_vec * w).mean()
        losses.append(loss.item())
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        all_y.append(y.cpu().numpy())
        all_p.append(pred.cpu().numpy())
    
    acc = correct / max(total, 1)
    mean_loss = float(np.mean(losses)) if losses else 0.0
    y_concat = np.concatenate(all_y) if all_y else np.array([], dtype=np.int64)
    p_concat = np.concatenate(all_p) if all_p else np.array([], dtype=np.int64)
    
    if use_action_masking and total > 0:
        illegal_rate = illegal_predictions / total
        print(f"   [eval] Illegal action rate (before masking): {illegal_rate:.2%}")
    
    return mean_loss, acc, y_concat, p_concat


@torch.no_grad()
def collect_logits_labels(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for statics, numerics, y, w in loader:
        statics, numerics = statics.to(device), numerics.to(device)
        logits = model(statics, numerics)
        all_logits.append(logits.cpu())
        all_y.append(y.clone())
    num_classes = len(Cfg.classes)
    logits_cat = torch.cat(all_logits, 0) if all_logits else torch.empty((0, num_classes))
    y_cat = torch.cat(all_y, 0) if all_y else torch.empty((0,), dtype=torch.long)
    return logits_cat, y_cat


@torch.no_grad()
def predict_with_policy(model, loader, device, dataset=None, T_val=1.0, fold_thresh=0, fold_margin=0.00, call_bias_logit=0.0):
    """Applies calibration, action masking, fold-gating, and call bias to get policy predictions."""
    model.eval()
    all_true, all_pred = [], []
    try:
        fold_idx = Cfg.classes.index("fold")
        call_idx = Cfg.classes.index("call")
        check_idx = Cfg.classes.index("check")
        raise_idx = Cfg.classes.index("raise")
    except ValueError:
        print("[predict_with_policy] Error: Required classes not found in Cfg.classes. Check config.")
        raise ValueError("Configuration error: Missing required classes")

    nonfold_idx = [i for i in range(len(Cfg.classes)) if i != fold_idx]
    if not nonfold_idx:
         raise ValueError("Configuration error: Only 'fold' class defined?")

    # Get to_call_bb index for action masking
    to_call_idx = None
    if dataset is not None and hasattr(dataset, 'numeric_cols') and hasattr(dataset, 'scaler'):
        try:
            to_call_idx = dataset.numeric_cols.index("to_call_bb")
        except (ValueError, AttributeError):
            print("[predict_with_policy] Warning: Cannot find 'to_call_bb' in dataset. Action masking disabled.")

    for statics, numerics, y, w in loader:
        statics, numerics = statics.to(device), numerics.to(device)
        logits = model(statics, numerics)

        # Apply action masking BEFORE any other transformations
        if to_call_idx is not None and dataset is not None:
            # Unscale to_call_bb to get original values
            to_call_scaled = numerics[:, to_call_idx].cpu().numpy()
            to_call_bb = to_call_scaled * dataset.scaler.scale_[to_call_idx] + dataset.scaler.mean_[to_call_idx]
            to_call_bb = torch.from_numpy(to_call_bb).to(device)
            
            # Create action mask
            mask = create_action_mask(to_call_bb, len(Cfg.classes))
            logits = apply_action_mask(logits, mask)

        if call_bias_logit != 0.0:
            logits[:, call_idx] = logits[:, call_idx] + call_bias_logit

        probs = torch.softmax(logits / max(T_val, 1e-6), dim=1)

        pfold = probs[:, fold_idx]
        if probs.shape[1] > 1:
             p_nonfold, _ = probs[:, nonfold_idx].max(dim=1)
             margin = pfold - p_nonfold
        else:
             p_nonfold = torch.zeros_like(pfold)
             margin = pfold

        pred = probs.argmax(dim=1)

        fold_mask = (pfold >= fold_thresh) & (margin >= fold_margin)
        pred = torch.where(fold_mask, torch.full_like(pred, fold_idx), pred)

        all_true.append(y.cpu())
        all_pred.append(pred.cpu())

    true_cat = torch.cat(all_true, 0).numpy() if all_true else np.array([], dtype=np.int64)
    pred_cat = torch.cat(all_pred, 0).numpy() if all_pred else np.array([], dtype=np.int64)
    return true_cat, pred_cat
def train():
    set_seed(Cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(Cfg.out_dir, exist_ok=True)

    bot_files = []
    gto_files = glob.glob(Cfg.gto_glob) if Cfg.gto_glob else []
    gto_v3_files = []
    multiplayer_files = glob.glob(Cfg.multiplayer_glob) if Cfg.multiplayer_glob else []
    
    # Combine all data sources
    paths = gto_files + multiplayer_files
    if not paths:
        raise SystemExit("No CSV files found. Check Cfg.gto_glob and Cfg.multiplayer_glob paths.")
    
    print(f"\n{'='*80}")
    print(f"LOADING MULTI-PLAYER DATASET")
    print(f"{'='*80}")
    print(f"Total files: {len(paths)} files")
    print(f"  - Augmented heads-up (GTO v2): {len(gto_files)} files")
    print(f"  - Multi-player synthetic: {len(multiplayer_files)} files")
    print(f"Position encoding: 3 categories (Early, Late, Blinds)")
    print(f"Multi-player features: 6 new features added")
    print(f"  - num_active_players, num_players_to_act, multiway_pot_flag")
    print(f"  - position_relative, num_raisers_this_street, avg_opp_stack_bb")
    print(f"  - (num_callers_this_street removed: all zeros in training data)")
    print(f"{'='*80}\n")

    ds = PokerDataset(paths)
    if len(ds) == 0:
         raise SystemExit("Dataset is empty after loading and filtering. Check paths and data.")

    n = len(ds)
    idx = np.arange(n)
    np.random.default_rng(Cfg.seed).shuffle(idx)
    val_n = int(n * Cfg.val_split)
    val_n = min(val_n, n)
    if n > 0 and val_n == n:
        val_n = n - 1

    val_idx, train_idx = idx[:val_n], idx[val_n:]
    if train_idx.size == 0 and n > 0:
        print("[WARN] Training set split is empty. Check val_split and dataset size.")
        train_idx = idx
    if val_idx.size == 0 and n > 0:
        print("[WARN] Validation set split is empty. Evaluation will be skipped.")


    ds.fit_scaler(train_idx)

    train_subset = Subset(ds, train_idx) if train_idx.size > 0 else None
    val_subset   = Subset(ds, val_idx) if val_idx.size > 0 else None

    if train_subset is None or len(train_subset) == 0:
         print("[WARN] Training subset is empty. Training loop will be skipped.")
         train_loader = []
    else:
        collate_train = partial(collate_with_scaler, scaler=ds.scaler, 
                               n_scaled_cols=len(ds.numeric_cols), 
                               n_binary_cols=len(ds.binary_cols))
        pin = torch.cuda.is_available()
        train_loader = DataLoader(train_subset, batch_size=Cfg.batch_size, shuffle=True,
                                  num_workers=0,
                                  pin_memory=pin,
                                  collate_fn=collate_train, drop_last=False)
        try:
             s0, n0, y0, w0 = next(iter(DataLoader(train_subset, batch_size=min(32, len(train_subset)), shuffle=True, num_workers=0, collate_fn=collate_train)))
             print(f"[sanity] static {tuple(s0.shape)} | numeric {tuple(n0.shape)} | y {tuple(y0.shape)} | w {tuple(w0.shape)}")
        except StopIteration:
              print("[sanity] Could not get sanity batch (train subset might be too small).")


    if val_subset is None or len(val_subset) == 0:
         print("[WARN] Validation subset is empty. Evaluation will be skipped.")
         val_loader = []
    else:
        collate_val   = partial(collate_with_scaler, scaler=ds.scaler,
                               n_scaled_cols=len(ds.numeric_cols),
                               n_binary_cols=len(ds.binary_cols))
        pin = torch.cuda.is_available()
        val_loader   = DataLoader(val_subset, batch_size=Cfg.batch_size, shuffle=False,
                                  num_workers=0,
                                  pin_memory=pin,
                                  collate_fn=collate_val, drop_last=False)

    if train_idx.size > 0:
         train_y = ds.y[train_idx]
         train_counts = np.bincount(train_y, minlength=len(Cfg.classes))
    else:
         train_counts = np.zeros(len(Cfg.classes), dtype=int)

    if val_idx.size > 0:
         val_y = ds.y[val_idx]
         val_counts = np.bincount(val_y, minlength=len(Cfg.classes))
    else:
         val_counts = np.zeros(len(Cfg.classes), dtype=int)

    print("======== DATASET SUMMARY ========")
    print(f"Total rows: {len(ds)}")
    print(f"Train rows: {len(train_idx)} | Val rows: {len(val_idx)}")
    total_numeric_dim = ds.numeric_mat.shape[1]
    print(f"Static dim: {ds.static_dim} | Numeric dim (scaled+binary): {total_numeric_dim} | Classes: {Cfg.classes}")
    print("Numeric columns (scaled):", ds.numeric_cols)
    print("Binary columns (unscaled):", ds.binary_cols)
    print("Train class counts:", dict(zip(Cfg.classes, train_counts.tolist())))
    print("Val class counts:",   dict(zip(Cfg.classes, val_counts.tolist())))
    print("=================================")

    # Model / optim / loss / scheduler
    model = PokerMLP(static_dim=ds.static_dim, numeric_dim=ds.numeric_mat.shape[1], num_classes=len(Cfg.classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=Cfg.lr, weight_decay=Cfg.weight_decay)
    
    # Use ReduceLROnPlateau - automatically reduces LR when validation loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, 
        mode='min',           # Minimize validation loss
        factor=0.5,           # Reduce LR by half
        patience=3,           # Wait 3 epochs before reducing
        min_lr=Cfg.lr_min,    # Don't go below 1e-6
        verbose=True          # Print when LR changes
    )
    
    print(f"[Scheduler] Using ReduceLROnPlateau: initial_lr={Cfg.lr:.1e}, factor=0.5, patience=3, min_lr={Cfg.lr_min:.1e}")
    print(f"[Scheduler] This will automatically reduce LR when validation loss plateaus")
    print(f"[Scheduler] Expected LR schedule: 5e-4 → 2.5e-4 → 1.25e-4 → 6.25e-5 → ... → 1e-6")
    
    # Choose loss function based on config
    if Cfg.use_focal:
        # Use focal loss WITHOUT class weights - let FocalLoss handle imbalance naturally
        # Previous attempts: 2.5x call weight caused 44% fold→call errors, 1.5x caused model to predict call even in no-bet situations
        # Focal loss with gamma=2.0 should be enough to handle minority classes without artificial weights
        alpha = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)  # [fold, check, call, raise] - all equal
        criterion = FocalLoss(alpha=alpha, gamma=Cfg.focal_gamma, reduction='none')
        print(f"[Loss] Using FocalLoss with gamma={Cfg.focal_gamma}, alpha={alpha.cpu().tolist()}")
        print(f"[Loss] NO class weights - letting FocalLoss gamma={Cfg.focal_gamma} handle imbalance naturally")
    else:
        # NOTE: PyTorch CrossEntropyLoss doesn't support label_smoothing with reduction='none'
        # So we implement label smoothing manually in the loss computation
        criterion = nn.CrossEntropyLoss(reduction="none")
        print(f"[Loss] Using CrossEntropyLoss with label_smoothing={Cfg.label_smoothing}")

    # Hand bucket fold regularization helper
    def compute_hand_bucket_fold_loss(model, statics, numerics, scaler_obj, hand_bucket_idx=10, to_call_idx=1):
        """
        Penalize model when fold probability doesn't vary with hand_bucket.
        Training data shows bucket=0 folds 48% vs bucket=4 folds 0% when facing bet.
        """
        # Only apply when facing a bet (to_call_bb > 0)
        # Check if facing bet (numerics are already scaled)
        to_call_scaled = numerics[:, to_call_idx]
        # Inverse transform to check original values
        to_call_vals = to_call_scaled * scaler_obj.scale_[to_call_idx] + scaler_obj.mean_[to_call_idx]
        facing_bet_mask = to_call_vals > 0.5
        
        if facing_bet_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        # Get only hands facing a bet
        statics_bet = statics[facing_bet_mask]
        numerics_bet = numerics[facing_bet_mask]
        
        # Create versions with weak (0) and strong (4) hands
        numerics_weak = numerics_bet.clone()
        numerics_strong = numerics_bet.clone()
        
        # Scale bucket values: 0 and 4
        weak_scaled = (0 - scaler_obj.mean_[hand_bucket_idx]) / scaler_obj.scale_[hand_bucket_idx]
        strong_scaled = (4 - scaler_obj.mean_[hand_bucket_idx]) / scaler_obj.scale_[hand_bucket_idx]
        
        numerics_weak[:, hand_bucket_idx] = weak_scaled
        numerics_strong[:, hand_bucket_idx] = strong_scaled
        
        # Get predictions
        logits_weak = model(statics_bet, numerics_weak)
        logits_strong = model(statics_bet, numerics_strong)
        
        probs_weak = torch.softmax(logits_weak, dim=1)
        probs_strong = torch.softmax(logits_strong, dim=1)
        
        # Fold probability (class 0)
        fold_weak = probs_weak[:, 0]
        fold_strong = probs_strong[:, 0]
        
        # Weak hands should fold ~40% more than strong hands
        fold_diff = fold_weak - fold_strong
        target_diff = torch.full_like(fold_diff, 0.40)
        
        # Smooth L1 loss (Huber)
        return torch.nn.functional.smooth_l1_loss(fold_diff, target_diff)
    
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": [], "diversity_loss": []}
    best_val = float("inf")
    patience, bad = Cfg.patience, 0
    
    # Enable hand_bucket regularization (set to 0.0 to disable)
    diversity_weight = 0.15  # Start conservative, can increase to 0.3 if needed
    print(f"\n[Regularization] Hand bucket fold diversity: weight={diversity_weight}")
    print(f"[Regularization] Target: weak hands (bucket=0) fold 40% more than strong (bucket=4)")
    
    # Enable action masking to prevent illegal moves
    use_action_masking = True
    to_call_idx = COLUMN_NAMES["numeric_pref"].index("to_call_bb")
    print(f"\n[Action Masking] ENABLED - Model will learn legal action constraints")
    print(f"[Action Masking] Rules: check only when to_call=0, call only when to_call>0")
    print(f"[Action Masking] Feature index: to_call_bb at position {to_call_idx}")

    if not train_loader:
         print("Skipping training loop as train_loader is empty.")
    else:
        for epoch in range(1, Cfg.max_epochs + 1):
            t0 = time.time()
            model.train()
            losses = []
            
            current_lr = opt.param_groups[0]['lr']
            diversity_losses = []
            
            for step, (statics, numerics, y, w) in enumerate(train_loader, start=1):
                statics, numerics, y, w = statics.to(device), numerics.to(device), y.to(device), w.to(device)
                logits = model(statics, numerics)
                
                # Get action mask for legal moves (but don't apply to logits yet)
                mask = None
                if use_action_masking:
                    # Extract to_call_bb from scaled numerics (inverse transform)
                    to_call_scaled = numerics[:, to_call_idx]
                    to_call_bb = to_call_scaled * ds.scaler.scale_[to_call_idx] + ds.scaler.mean_[to_call_idx]
                    
                    # Create mask
                    mask = create_action_mask(to_call_bb, len(Cfg.classes))
                
                # Apply label smoothing if not using focal loss
                if not Cfg.use_focal and Cfg.label_smoothing > 0:
                    # Use KL divergence with smoothed labels
                    # Apply masking to logits BEFORE softmax to get proper probabilities
                    if mask is not None:
                        logits_masked = apply_action_mask(logits, mask)
                    else:
                        logits_masked = logits
                    
                    log_probs = torch.nn.functional.log_softmax(logits_masked, dim=1)
                    smooth_targets = apply_label_smoothing(y, len(Cfg.classes), Cfg.label_smoothing)
                    # KL divergence: -sum(target * log_pred)
                    loss_vec = -(smooth_targets * log_probs).sum(dim=1)
                else:
                    # Standard cross-entropy (for focal loss or no smoothing)
                    # Apply masking for focal loss too
                    if mask is not None:
                        logits_masked = apply_action_mask(logits, mask)
                    else:
                        logits_masked = logits
                    loss_vec = criterion(logits_masked, y)
                
                ce_loss = (loss_vec * w).mean()
                
                # Hand bucket fold diversity regularization
                diversity_loss = compute_hand_bucket_fold_loss(model, statics, numerics, ds.scaler)
                diversity_losses.append(diversity_loss.item())
                
                # Combined loss
                loss = ce_loss + diversity_weight * diversity_loss

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                losses.append(loss.item())

                if step % 100 == 0:
                    avg_loss_so_far = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
                    avg_div_loss = np.mean(diversity_losses[-100:]) if len(diversity_losses) >= 100 else np.mean(diversity_losses)
                    print(f"   [epoch {epoch:02d}] step {step:05d} | loss {loss.item():.4f} | avg_loss {avg_loss_so_far:.4f} | div_loss {avg_div_loss:.4f} | lr {current_lr:.2e}")

            train_loss = float(np.mean(losses)) if losses else 0.0
            avg_diversity = float(np.mean(diversity_losses)) if diversity_losses else 0.0
            history["diversity_loss"].append(avg_diversity)

            if not val_loader:
                 print(f"Epoch {epoch:02d} | train {train_loss:.4f} | VAL SKIPPED | {time.time() - t0:.1f}s")
                 val_loss, val_acc = 0.0, 0.0
            else:
                # Calculate validation loss WITH same settings as training for fair comparison
                val_loss, val_acc, _, _ = eval_loop(
                    model, val_loader, device, criterion,
                    use_label_smoothing=(not Cfg.use_focal and Cfg.label_smoothing > 0),
                    label_smoothing=Cfg.label_smoothing,
                    num_classes=len(Cfg.classes),
                    scaler=ds.scaler,
                    to_call_idx=to_call_idx,
                    use_action_masking=use_action_masking
                )
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["lr"].append(current_lr)
                dt = time.time() - t0
                
                scheduler.step(val_loss)
                
                print(f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.3f} | div {avg_diversity:.4f} | lr {current_lr:.2e} | {dt:.1f}s")

                if val_loss < best_val - 1e-5:
                    best_val, bad = val_loss, 0
                    torch.save(model.state_dict(), os.path.join(Cfg.out_dir, "model.best.pt"))
                    print("   ↳ saved model.best.pt")
                else:
                    bad += 1
                    print(f"   ↳ no improvement ({bad}/{patience})")
                    if bad >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

    # Save model architecture configuration
    config = {
        'model_type': 'PokerMLP',
        'architecture': {
            'static_dim': ds.static_dim,
            'numeric_dim': ds.numeric_mat.shape[1],
            'num_classes': len(Cfg.classes),
            'version': 'v2_attention'  # Track architecture version
        },
        'training': {
            'batch_size': Cfg.batch_size,
            'learning_rate': Cfg.lr,
            'weight_decay': Cfg.weight_decay,
            'use_focal': Cfg.use_focal,
            'focal_gamma': Cfg.focal_gamma if Cfg.use_focal else None
        }
    }
    
    with open(os.path.join(Cfg.out_dir, 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
        
    # ----- Final evaluation on best model -----
    # Only proceed if a best model was saved (i.e., validation happened)
    best_model_path = os.path.join(Cfg.out_dir, "model.best.pt")
    if os.path.exists(best_model_path) and val_loader:
        print("Loading best checkpoint for final reports...")
        # Add weights_only=True for security
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

        print("Calculating temperature scaling...")
        logits_val, y_val_t = collect_logits_labels(model, val_loader, device)

        if logits_val.numel() > 0 and y_val_t.numel() > 0:
            try:
                T = torch.ones(1, requires_grad=True, device=logits_val.device)
                optT = torch.optim.Adam([T], lr=0.01)
                ce_mean = nn.CrossEntropyLoss(reduction="mean")

                for _ in range(100):
                    optT.zero_grad()
                    loss = ce_mean(logits_val / (T + 1e-6), y_val_t.to(logits_val.device))
                    loss.backward()
                    optT.step()
                    with torch.no_grad():
                        T.clamp_(min=1e-3)

                T_val = float(T.detach().cpu().item())
                print(f"[Calibration] Learned temperature T = {T_val:.3f}")
            except Exception as e:
                print(f"[Calibration] Failed to learn temperature: {e}. Using T=1.0")
                T_val = 1.0
        else:
            print("[Calibration] No validation data available for temperature scaling. Using T=1.0")
            T_val = 1.0


        print("Applying policy for final report (with action masking)...")
        y_true, y_pred = predict_with_policy(
            model, val_loader, device,
            dataset=ds,
            T_val=T_val,
            fold_thresh=Cfg.fold_thresh,
            fold_margin=Cfg.fold_margin,
            call_bias_logit=Cfg.call_logit_bias,
        )

        os.makedirs(Cfg.out_dir, exist_ok=True)

        if history["train_loss"] and history["val_loss"]:
            plt.figure()
            plt.plot(history["train_loss"], label="train_loss")
            plt.plot(history["val_loss"], label="val_loss")
            plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss Curves")
            plt.savefig(os.path.join(Cfg.out_dir, "loss_curves.png"), dpi=150); plt.close()

        if history["val_acc"]:
            plt.figure()
            plt.plot(history["val_acc"], label="val_acc")
            plt.legend(); plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Validation Accuracy")
            plt.savefig(os.path.join(Cfg.out_dir, "val_acc.png"), dpi=150); plt.close()

        if y_true.size > 0 and y_pred.size > 0:
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(Cfg.classes))))
            fig = plt.figure(figsize=(7, 6))
            plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.colorbar()
            plt.xticks(range(len(Cfg.classes)), Cfg.classes, rotation=45, ha="right")
            plt.yticks(range(len(Cfg.classes)), Cfg.classes)
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if cm[i, j] > thresh else "black")
            plt.title("Confusion Matrix (with policy)"); plt.xlabel("Predicted label"); plt.ylabel("True label")
            plt.tight_layout()
            fig.savefig(os.path.join(Cfg.out_dir, "confusion_matrix.png"), dpi=150); plt.close(fig)

            report = classification_report(y_true, y_pred, target_names=Cfg.classes, digits=3, zero_division=0)
            with open(os.path.join(Cfg.out_dir, "classification_report.txt"), "w") as f:
                f.write(report)
            print("\n--- Final Classification Report (with policy) ---")
            print(report)
        else:
             print("Skipping confusion matrix and classification report (no predictions made).")

    else:
        print("Skipping final evaluation as best model was not saved or validation was skipped.")
        T_val = 1.0

    if train_loader:
        torch.save(model.state_dict(), os.path.join(Cfg.out_dir, "model.pt"))

    full_numeric_cols = (ds.numeric_cols + ds.binary_cols) if ds else []
    meta = {
        "numeric_cols": full_numeric_cols,
        "binary_cols": ds.binary_cols if ds else [],
        "street_vocab": Cfg.streets,
        "pos_vocab": Cfg.positions,
        "classes": Cfg.classes,
        # Removed raise_small_max_bb and raise_medium_max_bb (no longer used with 4-class model)
        "static_dim": ds.static_dim if ds else 0,
        "card_dim": ds.card_dim if ds else 0,
        "numeric_dim": (ds.numeric_mat.shape[1] if ds else 0),
        "n_scaled_cols": (len(ds.numeric_cols) if ds else 0),
        "temperature": T_val,
        "policy": {
            "fold_thresh": Cfg.fold_thresh,
            "fold_margin": Cfg.fold_margin,
            "call_logit_bias": Cfg.call_logit_bias
        },
        "loss": {
            "type": "focal" if Cfg.use_focal else "crossentropy",
            "focal_gamma": Cfg.focal_gamma if Cfg.use_focal else None,
            "class_weights": [1.0, 1.0, 1.0, 1.0] if Cfg.use_focal else None,
            "label_smoothing": Cfg.label_smoothing
        }
    }
    with open(os.path.join(Cfg.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if ds and ds.scaler is not None:
        scaler_path = os.path.join(Cfg.out_dir, "scaler.joblib")
        try:
             joblib.dump(ds.scaler, scaler_path)
             print(f"Scaler saved to: {scaler_path}")
        except Exception as e:
             print(f"Error saving scaler: {e}")

    print(f"Artifacts potentially saved to: {os.path.abspath(Cfg.out_dir)}")


if __name__ == "__main__":
    train()