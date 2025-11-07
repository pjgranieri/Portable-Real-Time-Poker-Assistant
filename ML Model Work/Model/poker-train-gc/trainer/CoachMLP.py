# train_poker_mlp.py — schema-adapted + hardened + verbose

import os, glob, json, random, time
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


# ----------------------------
# Config
# ----------------------------
class Cfg:
    seed = 42

    # Update these paths to your actual CSVs
    bot_glob  = os.getenv("BOT_GLOB",  r"/mnt/data/bot/data(bot)_*.csv")
    real_glob = os.getenv("REAL_GLOB", r"/mnt/data/real/data(real)_*.csv")

    # Training
    batch_size   = 512
    lr           = 3e-4
    weight_decay = 1e-4
    max_epochs   = 25
    val_split    = 0.15
    num_workers  = 0  # set to 4 if your machine can handle it

    # Streets / positions (observed)
    streets   = ["preflop", "flop", "turn", "river"]
    positions = ["SB", "BB", "UTG", "MP", "CO", "BTN"]

    # Action classes (fixed order)
    classes = ["fold", "check", "call", "raise_s", "raise_m", "raise_l"]

    # Raise buckets (in big blinds)
    raise_small_max_bb  = 4.0     # (0, 4] -> small
    raise_medium_max_bb = 12.0    # (4, 12] -> medium; else large

    # Weighting: rely on bot for preflop, real for postflop
    weight_bot_preflop    = 1.5
    weight_real_postflop  = 1.5
    base_weight           = 1.0

    # Card encoding: 7 slots × 52 one-hot
    out_dir = os.getenv("OUT_DIR",   "./runs/poker_mlp_v1")


# Map your column names here (as detected)
COLUMN_NAMES = {
    "source": None,  # inferred from filename
    "street": "round",
    "hole1": "hole1",
    "hole2": "hole2",
    "board": ["flop1", "flop2", "flop3", "turn", "river"],
    # numerics already normalized to BB in both datasets
    "numeric_pref": [
        "pot_bb", "to_call_bb", "stack_bb", "hero_stack_bb", "opp_stack_bb",
        "effective_stack_bb", "spr", "raise_to_bb", "bet_frac_of_pot",
        "bet_frac_of_pot_clipped", "is_allin"
    ],
    "position": "position",
    "action": "action",          # raw action
    "whos_turn": "whos_turn",    # if present; harmless if missing
}


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


RANKS = "23456789TJQKA"
SUITS = "cdhs"
# Build 52-card index mapping
CARD2IDX = {}
_idx = 0
for r in RANKS:
    for s in SUITS:
        CARD2IDX[f"{r}{s}"] = _idx
        _idx += 1
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
        # force numeric, coerce errors to NaN
        df[c] = pd.to_numeric(df[c], errors="coerce")
        # count issues before fixing
        bad_counts[c] = {
            "nan": int(df[c].isna().sum()),
            "+inf": int(np.isposinf(df[c]).sum()),
            "-inf": int(np.isneginf(df[c]).sum()),
        }
        # fix
        s = df[c].replace([np.inf, -np.inf], np.nan)
        s = s.clip(lower=-clip_abs, upper=clip_abs)
        s = s.fillna(0.0).astype(np.float32)
        df[c] = s

    print(f"[sanitize_numeric_df] Dataset {tag}:")
    for c, cnts in bad_counts.items():
        if any(v > 0 for v in cnts.values()):
            print(f"  - {c}: NaN={cnts['nan']} +Inf={cnts['+inf']} -Inf={cnts['-inf']}")
    return df


# ----------------------------
# Dataset
# ----------------------------
class PokerDataset(Dataset):
    """
    Schema matches your CSVs:
      - street: round
      - cards: hole1, hole2, flop1..river
      - numeric: *_bb, spr, raise_to_bb, bet_frac_of_pot(_clipped), is_allin
      - position: position
      - action: fold/call/check/raise (we bucket raises)
      - source: inferred from filename ('bot' or 'real')
    """
    def __init__(self, paths: List[str]):
        rows = []
        for p in paths:
            try:
                df = pd.read_csv(p)
            except Exception as e:
                print(f"[WARN] Skip {p}: {e}")
                continue

            # normalize basic types
            if COLUMN_NAMES["street"] in df.columns:
                df[COLUMN_NAMES["street"]] = df[COLUMN_NAMES["street"]].astype(str).str.lower()
            if COLUMN_NAMES["position"] in df.columns:
                df[COLUMN_NAMES["position"]] = df[COLUMN_NAMES["position"]].astype(str)

            for k in ["hole1", "hole2"]:
                if COLUMN_NAMES[k] in df.columns:
                    df[COLUMN_NAMES[k]] = df[COLUMN_NAMES[k]].astype(str)
            for b in COLUMN_NAMES["board"]:
                if b in df.columns:
                    df[b] = df[b].astype(str)
                else:
                    df[b] = ""

            # ensure numeric cols exist; fill missing with 0
            for col in COLUMN_NAMES["numeric_pref"]:
                if col not in df.columns:
                    df[col] = 0.0
                df[col] = df[col].apply(safe_float)

            # action
            if COLUMN_NAMES["action"] not in df.columns:
                print(f"[WARN] {p} has no '{COLUMN_NAMES['action']}', skipping.")
                continue
            df[COLUMN_NAMES["action"]] = df[COLUMN_NAMES["action"]].astype(str).str.lower()

            # infer source from filename
            src = "bot" if "bot" in os.path.basename(p).lower() else ("real" if "real" in os.path.basename(p).lower() else "unknown")
            df["__source"] = src

            rows.append(df)

        if not rows:
            raise RuntimeError("No CSVs loaded. Check your paths.")
        self.df = pd.concat(rows, ignore_index=True)

        # Keep only known streets
        self.df = self.df[self.df[COLUMN_NAMES["street"]].isin(Cfg.streets)].reset_index(drop=True)

        # Feature vocab
        self.street_vocab = Cfg.streets
        self.pos_vocab    = Cfg.positions

        # Labels via bucketing
        self.df["__label"] = self.df.apply(self._derive_label, axis=1).astype(int)

        # Numeric columns actually present
        exist_numeric = [c for c in COLUMN_NAMES["numeric_pref"] if c in self.df.columns]
        # Prefer hero_stack_bb over stack_bb if both exist
        if "hero_stack_bb" in exist_numeric and "stack_bb" in exist_numeric:
            exist_numeric.remove("stack_bb")
        self.numeric_cols = exist_numeric

        # Sanitize numerics before scaler
        self.df = sanitize_numeric_df(self.df, self.numeric_cols, clip_abs=1_000_000, tag="PokerDataset")

        # Pre-encode static (cards + categorical)
        enc = [self._row_static_encoding(row) for _, row in self.df.iterrows()]
        # Store static as float16 to halve RAM; cast later to float32 for the model
        self.static_mat = np.stack([e[0] for e in enc], axis=0).astype(np.float16)
        self.card_dim   = enc[0][1]
        self.static_dim = self.static_mat.shape[1]

        # Labels and weights
        self.y = self.df["__label"].values.astype(np.int64)
        self.sample_w = self.df.apply(self._row_weight, axis=1).values.astype(np.float32)
        self.df["__label"] = self.df.apply(self._derive_label, axis=1).astype(int)
        fb = int(getattr(self, "_fallback_count", 0))
        if fb > 0:
            print(f"[label-fallback] used {fb} times (examples: {getattr(self, '_fallback_examples', [])})")
        else:
            print("[label-fallback] 0 rows (great).")

        # Scaler set in fit_scaler()
        self.scaler: Optional[StandardScaler] = None

    def _derive_label(self, row) -> int:

        if not hasattr(self, "_fallback_count"):
            self._fallback_count = 0
            self._fallback_examples = []

        act = str(row[COLUMN_NAMES["action"]]).lower().strip()

        if act == "fold":
            return Cfg.classes.index("fold")
        if act == "check":
            return Cfg.classes.index("check")
        if act == "call":
            return Cfg.classes.index("call")

        # treat raise/bet/allin as raises with bucketing
        if act in ("raise", "bet", "allin"):
            size = row.get("raise_to_bb", np.nan)
            if not np.isfinite(size) or size <= 0:
                # fallback proxy using pot fraction (roughly preflop pot ≈ 2.5bb)
                frac = float(row.get("bet_frac_of_pot_clipped", 0.0) or 0.0)
                size = 2.5 * frac
            size = float(np.clip(size, 0.0, 1e6))
            if size <= Cfg.raise_small_max_bb:
                return Cfg.classes.index("raise_s")
            if size <= Cfg.raise_medium_max_bb:
                return Cfg.classes.index("raise_m")
            return Cfg.classes.index("raise_l")

        self._fallback_count += 1
        if len(self._fallback_examples) < 5:
            self._fallback_examples.append(act)
        return Cfg.classes.index("check")


    def _row_weight(self, row) -> float:
        w = Cfg.base_weight
        if row[COLUMN_NAMES["street"]] == "preflop" and row["__source"] == "bot":
            w *= Cfg.weight_bot_preflop
        if row[COLUMN_NAMES["street"]] in ["flop", "turn", "river"] and row["__source"] == "real":
            w *= Cfg.weight_real_postflop
        return w

    def _encode_cards(self, row) -> np.ndarray:
        h1 = row[COLUMN_NAMES["hole1"]]
        h2 = row[COLUMN_NAMES["hole2"]]
        boards = [row[b] for b in COLUMN_NAMES["board"]]

        vecs = [card_one_hot(h1), card_one_hot(h2)]
        for c in boards:
            vecs.append(card_one_hot(c))
        return np.concatenate(vecs, axis=0).astype(np.float32)  # 7*52

    def _row_static_encoding(self, row) -> Tuple[np.ndarray, int]:
        cards = self._encode_cards(row)
        street_oh = one_hot(row[COLUMN_NAMES["street"]], self.street_vocab)
        pos_oh    = one_hot(row.get(COLUMN_NAMES["position"], ""), self.pos_vocab)
        static = np.concatenate([cards, street_oh, pos_oh], axis=0).astype(np.float32)
        return static, cards.shape[0]

    def fit_scaler(self, idxs: np.ndarray):
        sub = self.df.iloc[idxs]
        X = sub[self.numeric_cols].values.astype(np.float32)

        # Diagnostics before fit
        n_nonfinite = int(np.sum(~np.isfinite(X)))
        if n_nonfinite > 0:
            print(f"[fit_scaler] Warning: found {n_nonfinite} non-finite numeric values in training slice. Fixing...")
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

        # Column-wise stats (first and last percentile)
        print("[fit_scaler] Numeric column stats (train split):")
        for j, cname in enumerate(self.numeric_cols):
            col = X[:, j]
            p1 = float(np.percentile(col, 1))
            p99 = float(np.percentile(col, 99))
            print(f"  {cname:>24} | mean={col.mean():8.4f} std={col.std():8.4f} p1={p1:8.4f} p99={p99:8.4f}")

        self.scaler = StandardScaler().fit(X)
        print("[fit_scaler] Scaler fitted.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # cast back to float32 at tensor creation
        static = self.static_mat[idx].astype(np.float32)
        numeric = self.df.iloc[idx][self.numeric_cols].values.astype(np.float32)
        y = self.y[idx]
        w = self.sample_w[idx]
        return static, numeric, y, w


# ----------------------------
# Model
# ----------------------------
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


# ----------------------------
# Train / Eval
# ----------------------------
def collate_with_scaler(batch, scaler: Optional[StandardScaler]):
    statics, numerics, ys, ws = zip(*batch)
    statics = torch.from_numpy(np.stack(statics, axis=0)).float()
    numerics_np = np.stack(numerics, axis=0).astype(np.float32)
    if scaler is not None:
        numerics_np = scaler.transform(numerics_np)
    numerics = torch.from_numpy(numerics_np).float()
    y = torch.tensor(ys, dtype=torch.long)
    w = torch.tensor(ws, dtype=torch.float32)
    return statics, numerics, y, w


@torch.no_grad()
def eval_loop(model, loader, device):
    ce = nn.CrossEntropyLoss(reduction="none")
    model.eval()
    total, correct = 0, 0
    losses = []
    all_y, all_p = [], []
    for statics, numerics, y, w in loader:
        statics = statics.to(device)
        numerics = numerics.to(device)
        y = y.to(device)
        w = w.to(device)

        logits = model(statics, numerics)
        loss_vec = ce(logits, y)
        loss = (loss_vec * w).mean()
        losses.append(loss.item())

        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        all_y.append(y.cpu().numpy())
        all_p.append(pred.cpu().numpy())
    acc = correct / max(total, 1)
    return float(np.mean(losses)), acc, np.concatenate(all_y), np.concatenate(all_p)


def train():
    set_seed(Cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure output dir exists early (for best-model saves during training)
    os.makedirs(Cfg.out_dir, exist_ok=True)

    # expand globs
    bot_files  = glob.glob(Cfg.bot_glob)
    real_files = glob.glob(Cfg.real_glob)
    paths = bot_files + real_files
    if not paths:
        raise SystemExit("No CSVs found. Update Cfg.bot_glob / Cfg.real_glob.")
    print(f"[paths] bot files: {len(bot_files)}, real files: {len(real_files)}, total: {len(paths)}")

    # Build dataset
    ds = PokerDataset(paths)
    n = len(ds)
    idx = np.arange(n)
    np.random.default_rng(Cfg.seed).shuffle(idx)
    val_n = int(n * Cfg.val_split)
    val_idx, train_idx = idx[:val_n], idx[val_n:]

    # Fit scaler on train slice (with stats prints)
    ds.fit_scaler(train_idx)

    # Subsets + loaders
    train_subset = Subset(ds, train_idx)
    val_subset   = Subset(ds, val_idx)

    print("======== DATASET SUMMARY ========")
    print(f"Total rows: {len(ds)}")
    print(f"Train rows: {len(train_idx)} | Val rows: {len(val_idx)}")
    print(f"Static dim: {ds.static_dim} | Numeric dim: {len(ds.numeric_cols)} | Classes: {Cfg.classes}")
    print("Numeric columns:", ds.numeric_cols)
    

    # Class distribution on train/val
    train_y = ds.y[train_idx]
    val_y   = ds.y[val_idx]
    train_counts = np.bincount(train_y, minlength=len(Cfg.classes))
    val_counts   = np.bincount(val_y,   minlength=len(Cfg.classes))
    print("Train class counts:", dict(zip(Cfg.classes, train_counts.tolist())))
    print("Val class counts:",   dict(zip(Cfg.classes, val_counts.tolist())))
    print("=================================")

    # custom collate to apply scaler
    def collate_train(batch): return collate_with_scaler(batch, ds.scaler)
    def collate_val(batch):   return collate_with_scaler(batch, ds.scaler)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_subset, batch_size=Cfg.batch_size, shuffle=True,
                              num_workers=max(0, Cfg.num_workers), pin_memory=pin,
                              collate_fn=collate_train, drop_last=False)
    val_loader   = DataLoader(val_subset,   batch_size=Cfg.batch_size, shuffle=False,
                              num_workers=max(0, Cfg.num_workers), pin_memory=pin,
                              collate_fn=collate_val, drop_last=False)

    # One-time sanity batch print
    s0, n0, y0, w0 = next(iter(DataLoader(train_subset, batch_size=32, shuffle=True,
                                          num_workers=0, collate_fn=collate_train)))
    print(f"[sanity] static {tuple(s0.shape)} | numeric {tuple(n0.shape)} | y {tuple(y0.shape)} | w {tuple(w0.shape)}")

    # Model / optim
    model = PokerMLP(static_dim=ds.static_dim, numeric_dim=len(ds.numeric_cols), num_classes=len(Cfg.classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=Cfg.lr, weight_decay=Cfg.weight_decay)
    ce = nn.CrossEntropyLoss(reduction="none")

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    # Early stopping
    best_val = float("inf")
    patience, bad = 4, 0

    # Train loop
    for epoch in range(1, Cfg.max_epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for step, (statics, numerics, y, w) in enumerate(train_loader, start=1):
            statics, numerics, y, w = statics.to(device), numerics.to(device), y.to(device), w.to(device)
            logits = model(statics, numerics)
            loss = (ce(logits, y) * w).mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(loss.item())

            if step % 200 == 0:
                print(f"  [epoch {epoch:02d}] step {step:05d} | batch_loss {loss.item():.4f}")

        train_loss = float(np.mean(losses))
        val_loss, val_acc, _, _ = eval_loop(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        dt = time.time() - t0
        print(f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.3f} | {dt:.1f}s")

        if val_loss < best_val - 1e-4:
            best_val, bad = val_loss, 0
            torch.save(model.state_dict(), os.path.join(Cfg.out_dir, "model.best.pt"))
            print("  ↳ saved model.best.pt")
        else:
            bad += 1
            print(f"  ↳ no improvement ({bad}/{patience})")
            if bad >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # ----- Final evaluation on best model -----
    # Reload the best checkpoint before producing artifacts
    if os.path.exists(os.path.join(Cfg.out_dir, "model.best.pt")):
        model.load_state_dict(torch.load(os.path.join(Cfg.out_dir, "model.best.pt"), map_location=device))
        print("Loaded best checkpoint for final reports.")

    _, _, y_true, y_pred = eval_loop(model, val_loader, device)

    # ----- Plots -----
    # Ensure dir still exists
    os.makedirs(Cfg.out_dir, exist_ok=True)

    # Loss curves
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss Curves")
    plt.savefig(os.path.join(Cfg.out_dir, "loss_curves.png"), dpi=150); plt.close()

    # Val accuracy
    plt.figure()
    plt.plot(history["val_acc"], label="val_acc")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Validation Accuracy")
    plt.savefig(os.path.join(Cfg.out_dir, "val_acc.png"), dpi=150); plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(Cfg.classes))))
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.xticks(range(len(Cfg.classes)), Cfg.classes, rotation=45, ha="right")
    plt.yticks(range(len(Cfg.classes)), Cfg.classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.title("Confusion Matrix"); plt.xlabel("Pred"); plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(os.path.join(Cfg.out_dir, "confusion_matrix.png"), dpi=150); plt.close(fig)

    # Text report
    report = classification_report(y_true, y_pred, target_names=Cfg.classes, digits=3)
    with open(os.path.join(Cfg.out_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    print(report)

    # Save final model + meta (also keep best.pt)
    torch.save(model.state_dict(), os.path.join(Cfg.out_dir, "model.pt"))
    meta = {
        "numeric_cols": ds.numeric_cols,
        "street_vocab": ds.street_vocab,
        "pos_vocab": ds.pos_vocab,
        "classes": Cfg.classes,
        "raise_small_max_bb": Cfg.raise_small_max_bb,
        "raise_medium_max_bb": Cfg.raise_medium_max_bb,
        "static_dim": ds.static_dim,
        "card_dim": ds.card_dim,
    }
    with open(os.path.join(Cfg.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Artifacts saved to: {os.path.abspath(Cfg.out_dir)}")


if __name__ == "__main__":
    train()
