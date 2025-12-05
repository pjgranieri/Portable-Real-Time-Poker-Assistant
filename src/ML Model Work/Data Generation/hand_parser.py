# Parser + Tracer for Absolute Poker logs
# - only keep rows for seats that SHOW hole cards

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import re, os, csv, argparse, glob

EPS = 1e-9

#helpers
def parse_cards(run: str) -> List[str]:
    """Split 'JdKsAc' -> ['Jd','Ks','Ac']; '5d' -> ['5d']"""
    run = run.strip()
    return [run[i:i+2] for i in range(0, len(run), 2) if run[i:i+2]]

def parse_float_list(s: str) -> List[float]:
    s = s.strip()
    if not s or not s.startswith("["):
        return []
    return [float(x.strip()) for x in s.strip("[]").split(",") if x.strip()]

def parse_str_list(s: str) -> List[str]:
    s = s.strip()
    inner = s[1:-1] if (s.startswith("[") and s.endswith("]")) else s
    parts = []
    for tok in inner.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if (tok.startswith("'") and tok.endswith("'")) or (tok.startswith('"') and tok.endswith('"')):
            tok = tok[1:-1]
        parts.append(tok)
    return parts

def safe_div(a: float, b: float, eps: float = EPS) -> float:
    return a / (b if abs(b) > eps else eps)


class Tracer:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.events: List[Dict[str, Any]] = []   # structured logs
        self.unknown: List[str] = []             # raw lines 
        self.counts: Dict[str, int] = {}         # verb 

    def log(self, **kv):
        if self.enabled:
            self.events.append(kv)

    def bump(self, verb: str):
        self.counts[verb] = self.counts.get(verb, 0) + 1

    def log_unknown(self, raw: str):
        self.unknown.append(raw)

    def dump_text(self) -> str:
        if not self.enabled:
            return "(trace disabled)"
        lines = []
        for i, e in enumerate(self.events, 1):
            lines.append(
                f"[{i:03d}] raw='{e.get('raw','')}'  →  ev={e.get('event','?')} "
                f"seat={e.get('seat','-')}  street={e.get('street_name','?')}  "
                f"pot: {e.get('pot_before',0):.2f}→{e.get('pot_after',0):.2f}  "
                f"curr_to: {e.get('curr_to_before',0):.2f}→{e.get('curr_to_after',0):.2f}  "
                f"to_call(hero): {e.get('to_call_hero','-')}"
                f"{'  board=' + ''.join(e['board']) if e.get('board') else ''}"
            )
        if self.unknown:
            lines.append("\nUNKNOWN LINES:")
            lines += [f"  - {u}" for u in self.unknown]
        lines.append("\nACTION COUNTS: " + ", ".join(f"{k}:{v}" for k,v in sorted(self.counts.items())))
        return "\n".join(lines)

@dataclass
class PlayerState:
    seat: int
    name: str
    stack_bb: float                 # uncommitted stack left
    street_commit_bb: float = 0.0   # this street only
    total_commit_bb: float = 0.0    # across all streets
    hole: Optional[List[str]] = None
    in_hand: bool = True

@dataclass
class HandState:
    bb: float
    sb: float
    min_bet_cash: float
    players_by_pos: List[int]              # simple seat order proxy
    seat_to_index: Dict[int, int]
    player_states: Dict[int, PlayerState]
    dealer_seat: Optional[int] = None

    # dynamic
    pot_bb: float = 0.0
    board: List[str] = field(default_factory=list)
    curr_bet_bb: float = 0.0        
    raises_this_street: int = 0
    action_history_this_street: List[Tuple[int,str,float]] = field(default_factory=list)

    def reset_street(self):
        self.curr_bet_bb = 0.0
        self.action_history_this_street.clear()
        self.raises_this_street = 0
        for p in self.player_states.values():
            p.street_commit_bb = 0.0

    def to_call_bb(self, seat: int) -> float:
        p = self.player_states[seat]
        return max(0.0, self.curr_bet_bb - p.street_commit_bb)

def resolve_seat_token(st: HandState, token_num: int) -> int:

    if token_num in st.player_states:
        return token_num
    if 1 <= token_num <= len(st.players_by_pos):
        return st.players_by_pos[token_num - 1]
    raise KeyError(f"Unknown seat token 'p{token_num}'")

def street_from_board(board_cnt: int) -> Tuple[str, int]:
    if board_cnt == 0: return "preflop", 0
    if board_cnt == 3: return "flop", 1
    if board_cnt == 4: return "turn", 2
    if board_cnt == 5: return "river", 3
    if board_cnt in (1, 2): return "flop", 1  
    raise ValueError(f"Unexpected board card count: {board_cnt}")

def initialize_preflop_forces(st: HandState, antes_list: List[float], blinds_list: List[float], tracer: Tracer | None):

    order = st.players_by_pos
    if len(order) < 2:
        return
    if antes_list and any(float(x) > 0 for x in antes_list):
        for i, seat in enumerate(order):
            a_cash = float(antes_list[i]) if i < len(antes_list) else 0.0
            if a_cash > 0:
                a_bb = a_cash / st.bb
                ps = st.player_states[seat]
                a_bb = min(a_bb, ps.stack_bb)
                ps.stack_bb -= a_bb
                ps.total_commit_bb += a_bb
                st.pot_bb += a_bb
        if tracer and tracer.enabled:
            tracer.log(raw="<init_antes>", event="init_antes", street_name="preflop",
                       pot_before=0.0, pot_after=st.pot_bb,
                       curr_to_before=0.0, curr_to_after=st.curr_bet_bb, board=[])

    # SB/BB
    sb = st.sb / st.bb
    bb = 1.0
    sb_seat = order[0]
    bb_seat = order[1]
    for seat_, post_ in [(sb_seat, sb), (bb_seat, bb)]:
        ps = st.player_states[seat_]
        post_ = min(post_, ps.stack_bb)
        ps.stack_bb -= post_
        ps.street_commit_bb += post_
        ps.total_commit_bb += post_
        st.pot_bb += post_
    st.curr_bet_bb = bb

    # Optional straddle
    if len(blinds_list) >= 3 and float(blinds_list[2]) > 0 and len(order) >= 3:
        straddle_cash = float(blinds_list[2])
        straddle_to_bb = straddle_cash / st.bb
        straddle_seat = order[2]
        ps = st.player_states[straddle_seat]
        add = max(0.0, min(straddle_to_bb - ps.street_commit_bb, ps.stack_bb))
        if add > 0:
            ps.stack_bb -= add
            ps.street_commit_bb += add
            ps.total_commit_bb += add
            st.pot_bb += add
            st.curr_bet_bb = max(st.curr_bet_bb, straddle_to_bb)

    if tracer and tracer.enabled:
        tracer.log(raw="<init_blinds>", event="init_blinds", street_name="preflop",
                   seat=[sb_seat, bb_seat], pot_before=0.0, pot_after=st.pot_bb,
                   curr_to_before=0.0, curr_to_after=st.curr_bet_bb, board=[])

def safe_add_to_player(ps: PlayerState, pot_bb: float, add_bb: float) -> Tuple[float, float]:
    """Add chips from a player's stack into the pot, clamped to available stack. Returns (actual_add, new_pot)."""
    add = max(0.0, min(add_bb, ps.stack_bb))
    if add > 0:
        ps.stack_bb -= add
        ps.street_commit_bb += add
        ps.total_commit_bb += add
        pot_bb += add
    return add, pot_bb

def compute_effective_stack(st: HandState, seat: int) -> float:
    hero = st.player_states[seat]
    if not hero.in_hand:
        return 0.0
    hero_uncomm = max(0.0, hero.stack_bb)
    opp_uncomms = [max(0.0, p.stack_bb) for s,p in st.player_states.items() if s != seat and p.in_hand]
    opp_max = max(opp_uncomms) if opp_uncomms else 0.0
    return min(hero_uncomm, opp_max)

def row_is_valid(r: Dict[str, Any]) -> bool:
    if r["stack_bb"] is not None and r["stack_bb"] < -1e-6: return False
    if r["effective_stack_bb"] is not None and r["effective_stack_bb"] < -1e-6: return False
    if (r["effective_stack_bb"] is not None and r["stack_bb"] is not None
        and r["effective_stack_bb"] - r["stack_bb"] > 1e-6): return False
    if r["spr"] is not None and r["spr"] < -1e-6: return False
    if r["action"] == "raise":
        if r["raise_to_bb"] is None or r["raise_to_bb"] <= 0: return False
        # if bet_frac exploded but not an all-in, reject
        if r.get("bet_frac_of_pot", 0.0) > 50.0 and not r.get("is_allin", 0): return False
    return True

HDR_PAT = re.compile(r"^\[(\d+)\]$")
KV_PAT  = re.compile(r"^(\w+)\s*=\s*(.+)$")

def parse_hand_block_from_text(block_text: str) -> List[str]:
    lines = [ln.rstrip() for ln in block_text.splitlines() if ln.strip()]
    return lines

def build_rows_from_hand(block_lines: List[str], tracer: Tracer | None = None) -> List[Dict[str, Any]]:
    rows_raw: List[Dict[str, Any]] = []
    if tracer is None:
        tracer = Tracer(False)

    kv: Dict[str, str] = {}
    for ln in block_lines:
        m = KV_PAT.match(ln)
        if m:
            k, v = m.group(1), m.group(2)
            kv[k] = v

    if "blinds_or_straddles" not in kv or "starting_stacks" not in kv or "players" not in kv or "seats" not in kv:
        return rows_raw

    blinds_list = parse_float_list(kv["blinds_or_straddles"])
    if len(blinds_list) < 2:
        return rows_raw

    sb_cash, bb_cash = blinds_list[0], blinds_list[1]
    min_bet_cash = float(kv.get("min_bet", bb_cash))

    seats = [int(x) for x in kv["seats"].strip("[]").split(",")]
    players = parse_str_list(kv["players"])
    stacks_cash = parse_float_list(kv["starting_stacks"])
    if not (len(seats) == len(players) == len(stacks_cash)):
        return rows_raw

    seat_to_name = {s:n for s,n in zip(seats, players)}
    seat_to_stack_bb = {s: stacks_cash[i]/bb_cash for i,s in enumerate(seats)}

    players_by_pos = seats[:]  #listed order
    seat_to_index = {s:i for i,s in enumerate(players_by_pos)}

    # 2) initial state
    ps = { s: PlayerState(seat=s, name=seat_to_name[s], stack_bb=seat_to_stack_bb[s], in_hand=True)
           for s in seats }
    st = HandState(
        bb=bb_cash, sb=sb_cash, min_bet_cash=min_bet_cash,
        players_by_pos=players_by_pos,
        seat_to_index=seat_to_index,
        player_states=ps
    )

    # Post antes
    antes_list = parse_float_list(kv.get("antes","[]"))
    initialize_preflop_forces(st, antes_list, blinds_list, tracer)

    actions = parse_str_list(kv.get("actions", "[]"))
    actions = [a.replace("\\'", "'") for a in actions]

    showed: Dict[int, List[str]] = {}
    first_preflop_raiser: Optional[int] = None
    max_pot_bb_seen: float = st.pot_bb

    for raw in actions:
        pot_before = st.pot_bb
        curr_to_before = st.curr_bet_bb
        street_name, street_idx = street_from_board(len(st.board))

        # Deal hole
        if raw.startswith('d dh '):
            tracer.bump('dh')
            tracer.log(raw=raw, event='deal_hole', street_name=street_name,
                       pot_before=pot_before, pot_after=pot_before,
                       curr_to_before=curr_to_before, curr_to_after=curr_to_before,
                       board=st.board[:])
            continue

        # Deal board
        if raw.startswith('d db '):
            parts = raw.split()
            if len(parts) >= 3:
                board_run = parts[2]
                st.board.extend(parse_cards(board_run))
                st.reset_street()
                street_name2, _ = street_from_board(len(st.board))
                tracer.bump('db')
                tracer.log(raw=raw, event='deal_board', street_name=street_name2,
                           pot_before=pot_before, pot_after=st.pot_bb,
                           curr_to_before=curr_to_before, curr_to_after=st.curr_bet_bb,
                           board=st.board[:])
            else:
                tracer.log_unknown(raw)
            continue

        # Player action
        m = re.match(r"^p(\d+)\s+(\w+)(?:\s+(.+))?$", raw)
        if not m:
            tracer.log_unknown(raw)
            continue

        seat_token_num = int(m.group(1))
        try:
            seat = resolve_seat_token(st, seat_token_num)
        except KeyError:
            tracer.log_unknown(raw)
            tracer.log(raw=raw, event="warn_bad_seat", street_name=street_name, seat_token=seat_token_num,
                       pot_before=pot_before, pot_after=st.pot_bb,
                       curr_to_before=curr_to_before, curr_to_after=st.curr_bet_bb,
                       board=st.board[:])
            continue

        verb = m.group(2)
        amt_tok = m.group(3)
        amt_cash = None
        if amt_tok and amt_tok != '????':
            try:
                amt_cash = float(amt_tok)
            except ValueError:
                pass

        tracer.bump(verb)

        # Showdown reveal
        if verb == 'sm':
            if amt_tok:
                hole_cards = parse_cards(amt_tok)
                showed[seat] = hole_cards
                ps[seat].hole = hole_cards
            tracer.log(raw=raw, event='show', seat=seat, street_name=street_name,
                       pot_before=pot_before, pot_after=st.pot_bb,
                       curr_to_before=curr_to_before, curr_to_after=st.curr_bet_bb,
                       board=st.board[:])
            continue

        # numbers  before mutating state
        to_call = st.to_call_bb(seat)

        action = None
        raise_to_bb = 0.0
        bet_frac = 0.0
        add = 0.0

        if verb == 'f':
            action = 'fold'
            ps[seat].in_hand = False

        elif verb == 'cc':
            if to_call <= EPS:
                action = 'check'
            else:
                action = 'call'
                add_needed = to_call
                add, st.pot_bb = safe_add_to_player(ps[seat], st.pot_bb, add_needed)

        elif verb == 'cbr':
            action = 'raise'
            if amt_cash is None:
                tracer.log_unknown(raw); continue
            raise_to_bb = amt_cash / st.bb
   
            target_commit = max(raise_to_bb, st.curr_bet_bb)
            max_commit = ps[seat].street_commit_bb + ps[seat].stack_bb
            target_commit = min(target_commit, max_commit)
            add_needed = max(0.0, target_commit - ps[seat].street_commit_bb)
            add, st.pot_bb = safe_add_to_player(ps[seat], st.pot_bb, add_needed)
            # Update current raise
            st.curr_bet_bb = max(st.curr_bet_bb, ps[seat].street_commit_bb)
            # Count raises for this street
            st.raises_this_street += 1
            # bet fraction 
            bet_frac = add / max(pot_before, EPS)

            if street_name == "preflop" and first_preflop_raiser is None:
                first_preflop_raiser = seat

        else:
            tracer.log_unknown(raw)
            continue

        max_pot_bb_seen = max(max_pot_bb_seen, st.pot_bb)

        tracer.log(raw=raw, event=action, seat=seat, street_name=street_name,
                   pot_before=pot_before, pot_after=st.pot_bb,
                   curr_to_before=curr_to_before, curr_to_after=st.curr_bet_bb,
                   to_call_hero=to_call, board=st.board[:])

        seen_board = st.board[:]
        eff = compute_effective_stack(st, seat)
        spr_val = safe_div(eff, pot_before)
        is_allin = int(abs(ps[seat].stack_bb) <= EPS)

        row = {
            "hand_id": int(kv.get("hand", "0")),
            "player_id": ps[seat].name,
            "round": street_name,
            "hole1": (ps[seat].hole[0] if ps[seat].hole else None),
            "hole2": (ps[seat].hole[1] if ps[seat].hole else None),
            "flop1": (seen_board[0] if len(seen_board) >= 1 else None),
            "flop2": (seen_board[1] if len(seen_board) >= 2 else None),
            "flop3": (seen_board[2] if len(seen_board) >= 3 else None),
            "turn":  (seen_board[3] if len(seen_board) >= 4 else None),
            "river": (seen_board[4] if len(seen_board) >= 5 else None),

            "stack_bb": ps[seat].stack_bb,
            "pot_bb": pot_before,
            "to_call_bb": to_call,
            "action": action,

            "result_bb": None,
            "won_flag": None,

            "final_pot_bb": None,   # fill later
            "step": None,           # fill later
            "source": "handhq",
            "style": "unknown",

            "opp_stack_bb": eff,         
            "effective_stack_bb": eff,
            "spr": spr_val,

            "pos": st.seat_to_index.get(seat, 0),
            "in_position": 0,               # TBD

            "was_pfr": 0,                   # fill later
            "pot_type": "cash",
            "raises_preflop": 0,            # fill later
            "raises_this_street": st.raises_this_street,
            "street_index": street_idx,
            "board_texture": "unknown",

            "raise_to_bb": raise_to_bb if action == "raise" else 0.0,
            "bet_frac_of_pot": bet_frac if action == "raise" else 0.0,

            "is_allin": is_allin,
            "bet_frac_of_pot_clipped": (min(bet_frac, 5.0) if action == "raise" else 0.0),

            "per_hand_rows": None,                  # fill later
            "row_weight_decisionshare": None,       # fill later
            "row_weight_from_result": 0.0,
            "sample_weight": 1.0,

            "seat": seat,  # internal
        }

        # Validate row
        if row_is_valid(row):
            rows_raw.append(row)
        else:
            tracer.log(raw=raw, event="drop_row_invalid", seat=seat, street_name=street_name,
                       pot_before=pot_before, pot_after=st.pot_bb,
                       curr_to_before=curr_to_before, curr_to_after=st.curr_bet_bb,
                       board=st.board[:])

    # keep rows only for seats that actually showed hole cards
    keep_seats = set(showed.keys())
    rows = [r for r in rows_raw if r["seat"] in keep_seats]

    # Fill missing hole cards
    for r in rows:
        if r["seat"] in showed and (r["hole1"] is None or r["hole2"] is None):
            h = showed[r["seat"]]
            if len(h) == 2:
                r["hole1"], r["hole2"] = h[0], h[1]
    rows = [r for r in rows if r.get("hole1") and r.get("hole2")]

    if not rows:
        return rows

    hand_id = rows[0]["hand_id"]
    for i, r in enumerate(rows, start=1):
        r["step"] = i

    preflop_raises = sum(1 for r in rows if r["round"] == "preflop" and r["action"] == "raise")
    for r in rows:
        r["raises_preflop"] = int(preflop_raises)
        r["was_pfr"] = int(r["seat"] == first_preflop_raiser) if first_preflop_raiser is not None else 0
        r["final_pot_bb"] = max_pot_bb_seen

    per_hand_rows = len(rows)
    for r in rows:
        r["per_hand_rows"] = per_hand_rows
        r["row_weight_decisionshare"] = 1.0 / per_hand_rows

    #enforce bot column order
    for r in rows:
        r.pop("seat", None)

    bot_cols = [
        "hand_id","player_id","round","hole1","hole2","flop1","flop2","flop3","turn","river",
        "stack_bb","pot_bb","to_call_bb","action","result_bb","won_flag","final_pot_bb",
        "step","source","style",
        "opp_stack_bb","effective_stack_bb","spr",
        "pos","in_position","was_pfr","pot_type","raises_preflop","raises_this_street","street_index","board_texture",
        "raise_to_bb","bet_frac_of_pot",
        "per_hand_rows","row_weight_decisionshare","row_weight_from_result","sample_weight",
        # extras (safe to keep; drop at training time if you want)
        "is_allin","bet_frac_of_pot_clipped"
    ]

    for r in rows:
        for c in bot_cols:
            if c not in r:
                r[c] = None

    rows = [{c: r[c] for c in bot_cols} for r in rows]
    return rows

def iter_hand_blocks_from_text(text: str):
 
    header_idxs = [m.start() for m in re.finditer(r"(?m)^\[(\d+)\]\s*$", text)]
    if header_idxs:
        header_idxs.append(len(text))
        for i in range(len(header_idxs) - 1):
            chunk = text[header_idxs[i]:header_idxs[i+1]].strip()
            if chunk:
                yield chunk
        return
    for chunk in re.split(r"(?:\r?\n){2,}", text):
        ch = chunk.strip()
        if ch:
            yield ch

def iter_hand_blocks_from_file(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    for blk in iter_hand_blocks_from_text(text):
        if ("seats" in blk and "players" in blk and "starting_stacks" in blk and "blinds_or_straddles" in blk):
            yield blk

def collect_rows_from_path(in_path: str, limit_rows: Optional[int] = None,
                           file_glob: str = "*.txt", tracer_enabled: bool = False):

    paths = []
    if os.path.isdir(in_path):
        patterns = [file_glob, "*.log", "*.txt"]
        seen = set()
        for pat in patterns:
            for p in glob.glob(os.path.join(in_path, "**", pat), recursive=True):
                if p not in seen:
                    seen.add(p)
                    paths.append(p)
        paths.sort()
    else:
        paths = [in_path]

    all_rows = []
    for p in paths:
        try:
            for block in iter_hand_blocks_from_file(p):
                lines = parse_hand_block_from_text(block)
                tracer = Tracer(enabled=tracer_enabled)
                rows = build_rows_from_hand(lines, tracer=tracer)
                if rows:
                    all_rows.extend(rows)
                if limit_rows is not None and len(all_rows) >= limit_rows:
                    return all_rows[:limit_rows]
        except Exception:
            continue
    return all_rows

def write_rows_to_csv(rows, out_csv_path: str):
    bot_cols = [
        "hand_id","player_id","round","hole1","hole2","flop1","flop2","flop3","turn","river",
        "stack_bb","pot_bb","to_call_bb","action","result_bb","won_flag","final_pot_bb",
        "step","source","style",
        "opp_stack_bb","effective_stack_bb","spr",
        "pos","in_position","was_pfr","pot_type","raises_preflop","raises_this_street","street_index","board_texture",
        "raise_to_bb","bet_frac_of_pot",
        "per_hand_rows","row_weight_decisionshare","row_weight_from_result","sample_weight",
        "is_allin","bet_frac_of_pot_clipped"
    ]
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=bot_cols)
        w.writeheader()
        for r in rows or []:
            rr = {c: r.get(c, None) for c in bot_cols}
            w.writerow(rr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse poker logs into CSV.")
    parser.add_argument("--input", "-i", required=True,
                        help="Path.")
    parser.add_argument("--out", "-o", required=True,
                        help="Output CSV path.")
    parser.add_argument("--limit", "-n", type=int, default=None,
                        help="Max number of rows to save.")
    parser.add_argument("--glob", default="*.txt",
                        help="Glob pattern when input is a directory (default: *.txt).")
    parser.add_argument("--trace", action="store_true",
                        help="Enable per-hand tracer.")
    args = parser.parse_args()

    rows = collect_rows_from_path(
        in_path=args.input,
        limit_rows=args.limit,
        file_glob=args.glob,
        tracer_enabled=args.trace
    )
    write_rows_to_csv(rows, args.out)
    print(f"Saved {len(rows)} rows → {args.out}")
