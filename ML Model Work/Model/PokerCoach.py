import os, json, argparse
import numpy as np
import torch
import torch.nn as nn

# Try to load a saved scaler
try:
    import joblib
    HAVE_JOBLIB = True
except Exception:
    HAVE_JOBLIB = False

# -----------------------
# Cards & encoders (must match training)
# -----------------------
RANKS = "23456789TJQKA"
SUITS = "cdhs"
CARD2IDX = {f"{r}{s}": i for i, (r, s) in enumerate((rr, ss) for rr in RANKS for ss in SUITS)}
NUM_CARDS = 52

def card_one_hot(card: str) -> np.ndarray:
    v = np.zeros(NUM_CARDS, dtype=np.float32)
    if isinstance(card, str):
        c = card.strip()
        if len(c) == 2 and c[0] in RANKS and c[1] in SUITS:
            v[CARD2IDX[c]] = 1.0
    return v

def one_hot(value: str, vocab) -> np.ndarray:
    v = np.zeros(len(vocab), dtype=np.float32)
    if value in vocab:
        v[vocab.index(value)] = 1.0
    return v

# -----------------------
# Model (must match training)
# -----------------------
class PokerMLP(nn.Module):
    def __init__(self, static_dim: int, numeric_dim: int, num_classes: int):
        super().__init__()
        in_dim = static_dim + numeric_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(512, 256),    nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(256, 128),    nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, static_x, numeric_x):
        x = torch.cat([static_x, numeric_x], dim=1)
        return self.net(x)

def softmax_np(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-9)

# -----------------------
# Feature builder (same schema as training)
# -----------------------
def build_features(example: dict, meta: dict):
    """
    Expected keys in `example`:
      round: 'preflop'|'flop'|'turn'|'river'
      position: 'SB'|'BB'|'UTG'|'MP'|'CO'|'BTN' (optional)
      hole1, hole2: like 'As','Kd'
      flop1, flop2, flop3, turn, river: '' if not revealed yet
      numeric fields: exactly those in meta["numeric_cols"]
    """
    street_vocab = meta["street_vocab"]
    pos_vocab    = meta["pos_vocab"]
    numeric_cols = meta["numeric_cols"]

    # --- cards to 7×52 one-hots ---
    h1 = example.get("hole1",""); h2 = example.get("hole2","")
    f1 = example.get("flop1",""); f2 = example.get("flop2",""); f3 = example.get("flop3","")
    t  = example.get("turn","");  r  = example.get("river","")

    cards_vec = np.concatenate([
        card_one_hot(h1), card_one_hot(h2),
        card_one_hot(f1), card_one_hot(f2), card_one_hot(f3),
        card_one_hot(t),  card_one_hot(r)
    ]).astype(np.float32)  # 7*52 = 364

    street_oh = one_hot(str(example.get("round","")).lower(), street_vocab)
    pos_oh    = one_hot(str(example.get("position","")), pos_vocab)

    static_vec = np.concatenate([cards_vec, street_oh, pos_oh], axis=0).astype(np.float32)

    # --- numerics (same order as training) ---
    num_vals = []
    for c in numeric_cols:
        v = example.get(c, 0.0)
        try:
            v = float(v)
        except:
            v = 0.0
        if not np.isfinite(v):
            v = 0.0
        num_vals.append(v)
    numeric_vec = np.array(num_vals, dtype=np.float32)

    return static_vec, numeric_vec

# -----------------------
# Artifacts & prediction
# -----------------------
def load_artifacts(run_dir):
    # meta
    with open(os.path.join(run_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    static_dim = meta["static_dim"]
    numeric_dim = len(meta["numeric_cols"])
    num_classes = len(meta["classes"])

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PokerMLP(static_dim, numeric_dim, num_classes).to(device)
    ckpt = os.path.join(run_dir, "model.best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(run_dir, "model.pt")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # scaler
    scaler = None
    spath = os.path.join(run_dir, "scaler.pkl")
    if HAVE_JOBLIB and os.path.exists(spath):
        scaler = joblib.load(spath)
    else:
        class _Identity:
            def transform(self, X): return X
        scaler = _Identity()
        if not HAVE_JOBLIB:
            print("[warn] joblib not installed; using identity scaler.")
        elif not os.path.exists(spath):
            print("[warn] scaler.pkl not found; using identity scaler.")

    return model, meta, scaler, device

def predict_one(example: dict, run_dir="./runs/poker_mlp_v1"):
    model, meta, scaler, device = load_artifacts(run_dir)
    static_vec, numeric_vec = build_features(example, meta)
    # scale numerics
    numeric_vec = scaler.transform(numeric_vec.reshape(1, -1)).astype(np.float32)

    static_t  = torch.from_numpy(static_vec.reshape(1, -1)).to(device)
    numeric_t = torch.from_numpy(numeric_vec).to(device)

    with torch.no_grad():
        logits = model(static_t.float(), numeric_t.float()).cpu().numpy()[0]
        probs  = softmax_np(logits)

    classes = meta["classes"]
    pred_idx = int(np.argmax(probs))
    return classes[pred_idx], {cls: float(p) for cls, p in zip(classes, probs)}

# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Predict poker action with trained MLP")
    ap.add_argument("--run_dir", default="./runs/poker_mlp_v1", help="Folder with model.pt/meta.json/scaler.pkl")
    ap.add_argument("--example_json", help="Path to JSON with a single state")
    ap.add_argument("--print_probs", action="store_true", help="Print class probabilities")
    args = ap.parse_args()

    if args.example_json:
        with open(args.example_json, "r") as f:
            ex = json.load(f)
    else:
        # Demo example — edit to your liking
        ex = {
            "round": "flop",
            "position": "BTN",
            "hole1": "As", "hole2": "Td",
            "flop1": "Qs", "flop2": "7h", "flop3": "2s",
            "turn": "", "river": "",
            "pot_bb": 6.0, "to_call_bb": 2.0,
            "hero_stack_bb": 80.0, "opp_stack_bb": 100.0,
            "effective_stack_bb": 80.0, "spr": 13.0,
            "raise_to_bb": 0.0, "bet_frac_of_pot": 0.0, "bet_frac_of_pot_clipped": 0.0,
            "is_allin": 0.0
        }

    action, probs = predict_one(ex, args.run_dir)
    print(f"Predicted action: {action}")
    if args.print_probs:
        pretty = {k: float(f"{v:.4f}") for k, v in probs.items()}
        print("Class probabilities:", pretty)

if __name__ == "__main__":
    main()
