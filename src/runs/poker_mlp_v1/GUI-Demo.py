# app.py — Heads-up GUI (You vs Model), "pure model" version
# - No inference-time guardrails: the model's logits decide the action
# - Enforces legality only (can't fold with 0 to call, can't check if you owe)
# - Preflop: BIG BLIND acts first; Postflop: SMALL BLIND acts first
# - SB must complete preflop (via per-street contributions/current_bet)
# - Robust street closure + auto-advance

import os, json, time, random, csv
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

try:
    import joblib
    HAVE_JOBLIB = True
except Exception:
    HAVE_JOBLIB = False

# ---------- Safe rerun helper ----------
def do_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# ------------------ Model & encoders (match training) ------------------
RANKS = "23456789TJQKA"
SUITS = "cdhs"
ALL_CARDS = [r + s for r in RANKS for s in SUITS]
CARD2IDX = {c: i for i, c in enumerate(ALL_CARDS)}
NUM_CARDS = 52

def card_one_hot(card: str) -> np.ndarray:
    v = np.zeros(NUM_CARDS, dtype=np.float32)
    if isinstance(card, str) and len(card)==2 and card[0] in RANKS and card[1] in SUITS:
        v[CARD2IDX[card]] = 1.0
    return v

def one_hot(value: str, vocab) -> np.ndarray:
    v = np.zeros(len(vocab), dtype=np.float32)
    if value in vocab:
        v[vocab.index(value)] = 1.0
    return v

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

# ------------------ Artifacts loader ------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts(run_dir):
    meta_path = os.path.abspath(os.path.join(run_dir or ".", "meta.json"))
    if not os.path.isfile(meta_path):
        st.error(f"meta.json not found at:\n{meta_path}\nSet 'Run dir' to the folder with meta.json + model.pt.")
        st.stop()
    with open(meta_path, "r") as f:
        meta = json.load(f)

    static_dim = meta["static_dim"]
    numeric_dim = len(meta["numeric_cols"])
    num_classes = len(meta["classes"])
    classes = meta["classes"]

    device = torch.device("cpu")
    model = PokerMLP(static_dim, numeric_dim, num_classes).to(device)
    model_best = os.path.abspath(os.path.join(run_dir or ".", "model.best.pt"))
    model_pt   = os.path.abspath(os.path.join(run_dir or ".", "model.pt"))
    ckpt = model_best if os.path.isfile(model_best) else model_pt
    if not os.path.isfile(ckpt):
        st.error(f"No model checkpoint found.\nTried:\n{model_best}\n{model_pt}")
        st.stop()
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    spath = os.path.abspath(os.path.join(run_dir or ".", "scaler.pkl"))
    if HAVE_JOBLIB and os.path.exists(spath):
        scaler = joblib.load(spath)
    else:
        class _Identity:
            def transform(self, X): return X
        scaler = _Identity()
        if not os.path.exists(spath):
            st.warning("scaler.pkl not found — using identity scaler (still works).")
        elif not HAVE_JOBLIB:
            st.warning("joblib not installed — using identity scaler.")

    return model, meta, scaler, device, classes

def build_features(example: dict, meta: dict, scaler):
    street_vocab = meta["street_vocab"]
    pos_vocab    = meta["pos_vocab"]
    numeric_cols = meta["numeric_cols"]

    h1 = example.get("hole1",""); h2 = example.get("hole2","")
    f1 = example.get("flop1",""); f2 = example.get("flop2",""); f3 = example.get("flop3","")
    t  = example.get("turn","");  r  = example.get("river","")

    cards_vec = np.concatenate([
        card_one_hot(h1), card_one_hot(h2),
        card_one_hot(f1), card_one_hot(f2), card_one_hot(f3),
        card_one_hot(t),  card_one_hot(r)
    ]).astype(np.float32)

    street_oh = one_hot(str(example.get("round","")).lower(), street_vocab)
    pos_oh    = one_hot(str(example.get("position","")), pos_vocab)
    static_vec = np.concatenate([cards_vec, street_oh, pos_oh], axis=0).astype(np.float32)

    nums = []
    for c in numeric_cols:
        v = example.get(c, 0.0)
        try: v = float(v)
        except: v = 0.0
        if not np.isfinite(v): v = 0.0
        nums.append(v)
    numeric_vec = np.array(nums, dtype=np.float32)
    numeric_vec = scaler.transform(numeric_vec.reshape(1, -1)).astype(np.float32)
    return static_vec, numeric_vec

# ------------------ Dealing helpers ------------------
def deal_deck():
    deck = ALL_CARDS[:]
    random.shuffle(deck)
    return deck

def deal_hand(deck):
    hero_h1 = deck.pop(); vill_h1 = deck.pop()
    hero_h2 = deck.pop(); vill_h2 = deck.pop()
    return (hero_h1, hero_h2), (vill_h1, vill_h2)

def burn(deck):
    if deck: deck.pop()

def deal_flop(deck):
    burn(deck); return deck.pop(), deck.pop(), deck.pop()

def deal_turn(deck):
    burn(deck); return deck.pop()

def deal_river(deck):
    burn(deck); return deck.pop()

def choose_raise_size_bucket(cls_name, pot_bb):
    if cls_name == "raise_s": return max(1.5, 0.5 * pot_bb)
    if cls_name == "raise_m": return max(2.5, 0.75 * pot_bb)
    if cls_name == "raise_l": return max(3.5, 1.25 * pot_bb)
    return 0.0

def normalize_user_action(s):
    s = (s or "").strip().lower()
    if s in {"f","fold"}: return ("fold", 0.0)
    if s in {"k","check"}: return ("check", 0.0)
    if s in {"c","call"}: return ("call", 0.0)
    if s.startswith("bet"):
        parts = s.split()
        if len(parts)==2:
            try: return ("bet", float(parts[1]))
            except: pass
        return ("bet", 0.0)
    if s.startswith("raise"):
        parts = s.split()
        if len(parts)==2:
            try: return ("raise", float(parts[1]))
            except: pass
        return ("raise", 0.0)
    return (s, 0.0)

# ------------------ CSV logging ------------------
def ensure_csv(path, header):
    new = not os.path.exists(path)
    if new:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)

def append_csv(path, row, header):
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writerow(row)

# ------------------ Session state ------------------
st.set_page_config(page_title="Poker Coach Live Demo", layout="wide")
st.title("🃏 Poker Coach — Live GUI Demo (You vs Model)")

with st.sidebar:
    st.header("Settings")
    default_run_dir = APP_DIR
    run_dir = st.text_input("Run dir (folder with model/meta/scaler)", default_run_dir)
    log_csv = st.text_input("Log CSV", os.path.join(APP_DIR, "gui_demo_log.csv"))
    tick = st.slider("Tick seconds (pace)", 1, 20, 6, 1)
    sb = st.number_input("Small blind (bb)", value=0.5, step=0.5)
    bb = st.number_input("Big blind (bb)", value=1.0, step=0.5)
    stack_bb = st.number_input("Starting stacks (bb)", value=100.0, step=10.0)
    new_hand_btn = st.button("🆕 Start New Hand")

model, meta, scaler, device, classes = load_artifacts(run_dir)
csv_header = [
    "round","position","whos_turn",
    "hole1","hole2","flop1","flop2","flop3","turn","river",
    "pot_bb","to_call_bb","hero_stack_bb","opp_stack_bb","effective_stack_bb","spr",
    "raise_to_bb","bet_frac_of_pot","bet_frac_of_pot_clipped","is_allin","action",
    *[f"p_{c}" for c in classes], "model_class"
]
ensure_csv(log_csv, csv_header)

EPS = 1e-6

if "rng" not in st.session_state: st.session_state.rng = random.Random(42)
if "deck" not in st.session_state: st.session_state.deck = []
if "hand_no" not in st.session_state: st.session_state.hand_no = 0
if "state" not in st.session_state: st.session_state.state = {}

# ------ Low-level betting helpers that read/write session state directly ------
def init_contributions(preflop=False, hero_pos="SB"):
    if preflop:
        if hero_pos == "SB":
            return {"hero": float(sb), "villain": float(bb)}
        else:
            return {"hero": float(bb), "villain": float(sb)}
    else:
        return {"hero": 0.0, "villain": 0.0}

def current_bet_from_contrib(contrib):
    return max(contrib["hero"], contrib["villain"])

def to_call_for(actor):
    S = st.session_state.state
    return max(0.0, current_bet_from_contrib(S["contrib"]) - S["contrib"][actor])

def start_new_street(street):
    S = st.session_state.state
    if street == "preflop":
        S["contrib"] = init_contributions(preflop=True, hero_pos=S["hero_pos"])
        S["turn_to"] = "hero" if S["hero_pos"] == "BB" else "villain"  # BB acts first preflop
    else:
        S["contrib"] = init_contributions(preflop=False)
        S["turn_to"] = "hero" if S["hero_pos"] == "SB" else "villain"  # SB acts first postflop
    S["raised"] = False
    S["to_call_bb"] = to_call_for(S["turn_to"])

def init_hand():
    st.session_state.hand_no += 1

    model_is_sb = (st.session_state.hand_no % 2 == 1)  # model SB on odd hands
    hero_pos = "SB" if model_is_sb else "BB"  # model is "hero"
    opp_pos  = "BB" if model_is_sb else "SB"

    deck = deal_deck()
    (hero_h1, hero_h2), (vill_h1, vill_h2) = deal_hand(deck)

    hero_stack = float(stack_bb)
    vill_stack = float(stack_bb)

    # Post blinds (deduct stacks, add to pot)
    hero_post = float(sb if model_is_sb else bb)
    vill_post = float(bb if model_is_sb else sb)
    hero_stack -= hero_post
    vill_stack -= vill_post
    pot = float(sb + bb)

    st.session_state.deck = deck
    st.session_state.state = {
        "hero_h1": hero_h1, "hero_h2": hero_h2,
        "vill_h1": vill_h1, "vill_h2": vill_h2,
        "flop1":"", "flop2":"", "flop3":"", "turn":"", "river":"",
        "pot_bb": pot,
        "hero_stack_bb": hero_stack, "vill_stack_bb": vill_stack,
        "street": "preflop",
        "hero_pos": hero_pos,
        "opp_pos": opp_pos,
        "history": [],
        "raised": False,
        "contrib": {"hero":0.0, "villain":0.0},  # set properly below
        "turn_to": "hero"  # set properly below
    }
    # Now that session_state.state exists, we can call:
    start_new_street("preflop")

if new_hand_btn or not st.session_state.state:
    init_hand()

# Bind S only after initialization
S = st.session_state.state
deck = st.session_state.deck

def update_to_call():
    S = st.session_state.state
    S["to_call_bb"] = to_call_for(S["turn_to"])

# ------------------ Layout ------------------
c1, c2, c3 = st.columns([1.4,1.2,1.0])
with c1:
    st.subheader("Board")
    st.write(f"**Street:** {S['street'].upper()}")
    st.write(f"{S['flop1'] or '??'} {S['flop2'] or '??'} {S['flop3'] or '??'}  |  {S['turn'] or '??'}  |  {S['river'] or '??'}")

    st.subheader("Hole Cards")
    st.write(f"**Model ({S['hero_pos']}):** {S['hero_h1']} {S['hero_h2']}")
    st.write(f"**You ({S['opp_pos']}):** {S['vill_h1']} {S['vill_h2']}")

with c2:
    st.subheader("Stacks / Pot")
    st.metric("Pot (bb)", f"{S['pot_bb']:.2f}")
    st.metric("To Call (bb)", f"{S['to_call_bb']:.2f}")
    st.metric("Model stack (bb)", f"{S['hero_stack_bb']:.1f}")
    st.metric("You stack (bb)", f"{S['vill_stack_bb']:.1f}")

with c3:
    st.subheader("Turn")
    st.write("Acting:", "**MODEL**" if S["turn_to"]=="hero" else "**YOU**")
    st.write("Model Seat:", S["hero_pos"])
    st.write("Your Seat:", S["opp_pos"])

st.subheader("Round Status (debug)")
st.write(f"turn_to={S['turn_to']}, street={S['street']}, raised={S.get('raised', False)}")
st.write(f"contrib={S['contrib']} current_bet={max(S['contrib'].values()):.3f} to_call={S['to_call_bb']:.3f}")
st.write(f"history (last 6) = {S['history'][-6:]}")

st.divider()

# ------------------ Logging helper ------------------
def append_row(whos_turn, action, amount_bb, probs=None, pred_cls=None):
    S = st.session_state.state
    row = {
        "round": S["street"], "position": S["hero_pos"] if whos_turn=="hero" else S["opp_pos"],
        "whos_turn": whos_turn,
        "hole1": S["hero_h1"] if whos_turn=="hero" else S["vill_h1"],
        "hole2": S["hero_h2"] if whos_turn=="hero" else S["vill_h2"],
        "flop1": S["flop1"], "flop2": S["flop2"], "flop3": S["flop3"],
        "turn": S["turn"], "river": S["river"],
        "pot_bb": S["pot_bb"], "to_call_bb": S["to_call_bb"],
        "hero_stack_bb": S["hero_stack_bb"], "opp_stack_bb": S["vill_stack_bb"],
        "effective_stack_bb": min(S["hero_stack_bb"], S["vill_stack_bb"]),
        "spr": (min(S["hero_stack_bb"], S["vill_stack_bb"]) / max(S["pot_bb"], 1e-9)),
        "raise_to_bb": amount_bb if action in ("bet","raise") else 0.0,
        "bet_frac_of_pot": (amount_bb / max(S["pot_bb"], 1e-9)) if action in ("bet","raise") else 0.0,
        "bet_frac_of_pot_clipped": min((amount_bb / max(S["pot_bb"], 1e-9)) if action in ("bet","raise") else 0.0, 2.5463),
        "is_allin": 1.0 if min(S["hero_stack_bb"], S["vill_stack_bb"]) <= 0 else 0.0,
        "action": action if action!="call" else ("check" if S["to_call_bb"]<=EPS else "call"),
    }
    if probs is not None and pred_cls is not None:
        for i, cls in enumerate(classes):
            row[f"p_{cls}"] = float(probs[i])
        row["model_class"] = pred_cls
    append_csv(log_csv, row, csv_header)

# ------------------ Street closure + transitions ------------------
def street_is_closed():
    S = st.session_state.state
    owes_hero = to_call_for("hero") > EPS
    owes_vill = to_call_for("villain") > EPS
    if owes_hero or owes_vill:
        return False

    if S.get("raised", False):
        return len(S["history"]) >= 1 and S["history"][-1] == "call"

    if len(S["history"]) >= 2:
        a, b = S["history"][-2], S["history"][-1]
        if a == "check" and b == "check":
            return True
        if S["street"] == "preflop" and {a, b} == {"call", "check"}:
            return True
    return False

def next_street():
    S = st.session_state.state
    if S["street"] == "preflop":
        f1, f2, f3 = deal_flop(deck)
        S["flop1"], S["flop2"], S["flop3"] = f1, f2, f3
        S["street"] = "flop"
        start_new_street("postflop")
    elif S["street"] == "flop":
        S["turn"] = deal_turn(deck)
        S["street"] = "turn"
        start_new_street("postflop")
    elif S["street"] == "turn":
        S["river"] = deal_river(deck)
        S["street"] = "river"
        start_new_street("postflop")
    else:
        st.info("Showdown (demo). Starting a new hand.")
        init_hand()

def advance_if_closed():
    if street_is_closed():
        next_street()
        do_rerun()

# ------------------ Chip movement helpers using contributions ------------------
def put_money(actor, amount):
    S = st.session_state.state
    amount = max(0.0, float(amount))
    if actor == "hero":
        invest = min(amount, S["hero_stack_bb"])
        S["hero_stack_bb"] -= invest
        S["pot_bb"] += invest
        S["contrib"]["hero"] += invest
        return invest
    else:
        invest = min(amount, S["vill_stack_bb"])
        S["vill_stack_bb"] -= invest
        S["pot_bb"] += invest
        S["contrib"]["villain"] += invest
        return invest

def set_turn(next_actor):
    S = st.session_state.state
    S["turn_to"] = next_actor
    update_to_call()

# ------------------ Model act (pure) ------------------
def model_act():
    S = st.session_state.state
    ex = {
        "round": S["street"], "position": S["hero_pos"],
        "hole1": S["hero_h1"], "hole2": S["hero_h2"],
        "flop1": S["flop1"], "flop2": S["flop2"], "flop3": S["flop3"],
        "turn": S["turn"], "river": S["river"],
    }
    for col in meta["numeric_cols"]:
        if   col=="pot_bb": ex[col]=S["pot_bb"]
        elif col=="to_call_bb": ex[col]=to_call_for("hero")
        elif col in ("hero_stack_bb","stack_bb"): ex[col]=S["hero_stack_bb"]
        elif col=="opp_stack_bb": ex[col]=S["vill_stack_bb"]
        elif col=="effective_stack_bb": ex[col]=min(S["hero_stack_bb"], S["vill_stack_bb"])
        elif col=="spr": ex[col]=(min(S["hero_stack_bb"], S["vill_stack_bb"])/max(S["pot_bb"],1e-9))
        elif col=="raise_to_bb": ex[col]=0.0
        elif col=="bet_frac_of_pot": ex[col]=0.0
        elif col=="bet_frac_of_pot_clipped": ex[col]=0.0
        elif col=="is_allin": ex[col]=1.0 if min(S["hero_stack_bb"], S["vill_stack_bb"])<=0 else 0.0
        else: ex[col]=0.0

    static_vec, numeric_vec = build_features(ex, meta, scaler)
    static_t = torch.from_numpy(static_vec.reshape(1,-1))
    numeric_t = torch.from_numpy(numeric_vec)

    with torch.no_grad():
        logits = model(static_t.float(), numeric_t.float()).cpu().numpy()[0]
        probs = softmax_np(logits)
    pred_idx = int(np.argmax(probs)); pred_cls = classes[pred_idx]

    # Enforce legality ONLY (no strategy overrides)
    owe = to_call_for("hero")
    if pred_cls == "fold" and owe <= EPS:
        pred_cls = "check_call"  # can't fold with 0 to call

    # Execute action
    if pred_cls == "check_call":
        if owe <= EPS:
            append_row("hero", "check", 0.0, probs, pred_cls)
            S["history"].append("check")
        else:
            invest = put_money("hero", owe)
            append_row("hero", "call", invest, probs, pred_cls)
            S["history"].append("call")
        set_turn("villain")

    elif pred_cls in ("raise_s","raise_m","raise_l"):
        target = choose_raise_size_bucket(pred_cls, S["pot_bb"])
        curr = current_bet_from_contrib(S["contrib"])
        raise_to = max(curr + 2.0, target)  # simple sizing rule to exceed current bet
        invest = put_money("hero", raise_to - S["contrib"]["hero"])
        S["raised"] = True
        append_row("hero", "raise", invest, probs, pred_cls)
        S["history"].append("raise")
        set_turn("villain")

    else:  # 'fold'
        if owe > EPS:
            append_row("hero", "fold", 0.0, probs, pred_cls)
            st.error("MODEL folds. You win. New hand.")
            init_hand()
            do_rerun()
            return
        else:
            append_row("hero", "check", 0.0, probs, pred_cls)
            S["history"].append("check")
            set_turn("villain")

    time.sleep(tick)
    advance_if_closed()

# ------------------ User action ------------------
def user_action(raw):
    S = st.session_state.state
    a, amt = normalize_user_action(raw)
    owe = to_call_for("villain")

    if a == "fold":
        if owe <= EPS:
            st.warning("Can't fold a free check. Treated as check.")
            a, amt = "check", 0.0
        else:
            append_row("villain", "fold", 0.0)
            st.error("You folded. Model wins. New hand.")
            init_hand()
            do_rerun()
            return

    if a in ("check","call"):
        if a == "check" and owe > EPS:
            a = "call"
        if a == "call":
            invest = put_money("villain", owe)
            append_row("villain", "call", invest)
            S["history"].append("call")
        else:
            append_row("villain", "check", 0.0)
            S["history"].append("check")
        set_turn("hero")

    elif a in ("bet","raise"):
        size = max(0.0, float(amt))
        if size <= 0.0: size = 2.5
        curr = current_bet_from_contrib(S["contrib"])
        raise_to = max(curr + 2.0, size)
        invest = put_money("villain", raise_to - S["contrib"]["villain"])
        S["raised"] = True
        append_row("villain", a, invest)
        S["history"].append("raise" if a=="raise" else "bet")
        set_turn("hero")

    else:
        append_row("villain", "check", 0.0)
        S["history"].append("check")
        set_turn("hero")

    time.sleep(tick)
    advance_if_closed()

# ------------------ UI controls ------------------
c_left, c_right = st.columns(2)

with c_left:
    st.subheader("Your Action")
    help_txt = "fold / check / call / bet X / raise X  (X in bb, e.g., raise 6)"
    user_in = st.text_input("Action", value="", help=help_txt, placeholder="e.g., raise 6")
    go = st.button("Submit Action")
    if go and S["turn_to"] == "villain":
        user_action(user_in)

with c_right:
    st.subheader("Model Action")
    if st.button("Make Model Act") and S["turn_to"] == "hero":
        model_act()

st.divider()
st.caption("Preflop: BB acts first; Postflop: SB acts first. Street closes on check/check, limp+check (preflop), or raise+call. SB must complete preflop; no free check.")

# Optional manual advance (debug)
if st.button("Force Advance (debug)"):
    next_street()
    do_rerun()
