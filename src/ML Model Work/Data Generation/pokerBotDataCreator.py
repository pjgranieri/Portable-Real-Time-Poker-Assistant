import csv
import random
import time
import os
from collections import OrderedDict, defaultdict

from treys import Evaluator, Card, Deck
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer


# Tunables
TARGET_TOTAL_HANDS     = 1200     # generate at least this many hands overall
ROUNDS_PER_SESSION     = 300      # hands per short match
SMALL_BLIND_CHIPS      = 5        # -> BB = 10 chips (1000/10 = 100bb)
INITIAL_STACK_CHIPS    = 1000     # ~100bb per player per session
FLUSH_EVERY_N_HANDS    = 200      # periodic flush to CSV 
LRU_MAXSIZE            = 4096     # equity cache size


# ML labels

ROWS = []
HAND_BUFFER = defaultdict(list)   # hand_id 
GLOBAL_STEP = 0                   # action order across all sessions
_writer = None

# Helpers

def _reset_globals():
        global ROWS, HAND_BUFFER, GLOBAL_STEP, _writer
        ROWS = []
        HAND_BUFFER = defaultdict(list)
        GLOBAL_STEP = 0
        _writer = None
        
def hand_to_key(hole_cards_str):
    #Converts ['S2','HA'] to 'A2s' / 'A2o' / '22'
    rank_order = "23456789TJQKA"
    r1, r2 = hole_cards_str[0][1], hole_cards_str[1][1]
    s1, s2 = hole_cards_str[0][0], hole_cards_str[1][0]
    if rank_order.index(r2) > rank_order.index(r1):
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    if r1 == r2:
        return r1 + r2
    return (r1 + r2 + "s") if s1 == s2 else (r1 + r2 + "o")

def convert_card(card_str):
    
    #Converts 'SA','HA','D9' (PPE style) or 'As','Td' to treys int.

    if len(card_str) != 2:
        if len(card_str) >= 2:
            r = card_str[0] if card_str[0] in "23456789TJQKA" else card_str[1]
            s = card_str[1] if r == card_str[0] else card_str[0]
            return Card.new(r + s.lower())
        raise ValueError(f"Unexpected card format: {card_str}")
    suit, rank = card_str[0], card_str[1]
    return Card.new(rank + suit.lower())


# Preflop charts & styles 
HAND_PERCENTILES = {
    "AA": 1, "KK": 1, "QQ": 1, "JJ": 1, "AKs": 1, "TT": 5, "AQs": 5, "AJs": 5, "KQs": 5, "AKo": 5,
    "99": 10, "ATs": 10, "KJs": 10, "QJs": 10, "JTs": 10, "AQo": 10, "88": 15, "KTs": 15, "Q9s": 15,
    "T9s": 15, "98s": 15, "AJo": 15, "KQo": 15, "77": 20, "A9s": 20, "A8s": 20, "A7s": 20, "A6s": 20,
    "A5s": 20, "A4s": 20, "A3s": 20, "A2s": 20, "K9s": 20, "J9s": 20, "T8s": 20, "87s": 20, "76s": 20,
    "65s": 20, "any_other": 75,
    # extra offsuit coverage
    "ATo": 22, "KJo": 28, "QJo": 30, "KTo": 35, "QTo": 38, "JTo": 32,
    "T9o": 40, "98o": 48, "A9o": 35, "K9o": 42, "Q9o": 48
}

STYLE = {
    "TAG":     {"raise_bump": -3, "call_bump": -5, "bb_defend_bump": -5,
                "semi_bluff_bump": -0.05, "cbet_bump": -0.10, "threebet_top_pct": 10},
    "LAG":     {"raise_bump": +4, "call_bump": +8, "bb_defend_bump": +10,
                "semi_bluff_bump": +0.07, "cbet_bump": +0.10, "threebet_top_pct": 18},
    # Optional extra styles for more diversity later:
    "NIT":     {"raise_bump": -8, "call_bump": -10, "bb_defend_bump": -10,
                "semi_bluff_bump": -0.10, "cbet_bump": -0.12, "threebet_top_pct": 6},
    "STATION": {"raise_bump": -2, "call_bump": +14, "bb_defend_bump": +16,
                "semi_bluff_bump": -0.05, "cbet_bump": -0.05, "threebet_top_pct": 8},
    "WHALE":   {"raise_bump": +8, "call_bump": +16, "bb_defend_bump": +18,
                "semi_bluff_bump": +0.12, "cbet_bump": +0.15, "threebet_top_pct": 22},
}

# LRU cache for equity

class LRUCache:
    def __init__(self, maxsize=LRU_MAXSIZE):
        self.maxsize = maxsize
        self._od = OrderedDict()
    def get(self, key):
        if key in self._od:
            self._od.move_to_end(key)
            return self._od[key]
        return None
    def put(self, key, value):
        self._od[key] = value
        self._od.move_to_end(key)
        if len(self._od) > self.maxsize:
            self._od.popitem(last=False)

# SuperBot
class SuperBot(BasePokerPlayer):
    RANKS = "23456789TJQKA"

    def __init__(self, style_name="TAG"):
        self.evaluator = Evaluator()
        self.hand_id = 0
        self.hand_offset = 0     # used to make hand_id unique 
        self.big_blind_amount = 0
        self.style_name = style_name
        self.style = STYLE[style_name]
        self.was_pfr = False
        self.raised_this_street = False
        self.start_stacks = {}

        self._full_deck = Deck.GetFullDeck()
        self._equity_cache = LRUCache(maxsize=LRU_MAXSIZE)

    # ---------- Position / seats ----------
    def _preflop_pos(self, round_state):
        seats = round_state['seats']
        n = len(seats)
        btn = round_state['dealer_btn']
        order = [(btn + 1 + i) % n for i in range(n)]
        labels_9max = ['UTG','UTG+1','MP','MP+1','HJ','CO','BTN','SB','BB']
        if n >= 9: labels = labels_9max
        elif n >= 6: labels = ['UTG','MP','CO','BTN','SB','BB'][-n:]
        elif n == 5: labels = ['MP','CO','BTN','SB','BB']
        elif n == 4: labels = ['CO','BTN','SB','BB']
        elif n == 3: labels = ['BTN','SB','BB']
        else: labels = ['SB','BB']
        me = next(i for i, s in enumerate(seats) if s['uuid'] == self.uuid)
        idx = order.index(me)
        return labels[idx]

    def _my_and_opp_stacks(self, round_state):
        seats = round_state['seats']
        me = next(s for s in seats if s['uuid'] == self.uuid)
        opp = next(s for s in seats if s['uuid'] != self.uuid)
        return me['stack'], opp['stack'], opp['name']

    def _in_position_hu(self, street, round_state):
        seats = round_state['seats']
        n = len(seats)
        btn = round_state['dealer_btn']
        me_index = next(i for i,s in enumerate(seats) if s['uuid']==self.uuid)
        is_btn = (me_index == btn)
        if n == 2:
            if street == 'preflop':
                return 0 if is_btn else 1
            else:
                return 1 if is_btn else 0
        return 1 if is_btn else 0

    # Range estimation
    def _estimate_opponent_range(self, round_state):
        history = round_state['action_histories'].get('preflop', [])
        n_raises = sum(1 for a in history if a.get('action') == 'RAISE')
        n_calls  = sum(1 for a in history if a.get('action') == 'CALL')
        if n_raises >= 2: cutoff = 12
        elif n_raises == 1: cutoff = 20
        elif n_calls >= 2: cutoff = 40
        else: cutoff = 60
        return [h for h,p in HAND_PERCENTILES.items() if h != "any_other" and p <= cutoff]

    # Combo matching
    def _combo_matches_key(self, c1, c2, key):
        r1 = Card.get_rank_int(c1); r2 = Card.get_rank_int(c2)
        s1 = Card.get_suit_int(c1); s2 = Card.get_suit_int(c2)
        k1 = self.RANKS[r1]; k2 = self.RANKS[r2]
        if r2 > r1: k1, k2, s1, s2 = k2, k1, s2, s1
        if k1 == k2: return key == (k1 + k2)
        need_s = (len(key) == 3 and key[2] == 's')
        need_o = (len(key) == 3 and key[2] == 'o')
        if (k1 + k2) not in (key[:2], key[:2][::-1]): return False
        suited = (s1 == s2)
        return (need_s and suited) or (need_o and not suited) or (not need_s and not need_o)

    # Board texture
    def _board_texture(self, board_ints):
        if not board_ints:
            return ('dry', False)
        ranks = [Card.get_rank_int(c) for c in board_ints]
        suits = [Card.get_suit_int(c) for c in board_ints]
        paired = (len(ranks) != len(set(ranks)))
        flushy = any(suits.count(s) >= 3 for s in (1,2,3,4))
        rs = sorted(ranks)
        straighty = False
        if len(rs) >= 3:
            for i in range(len(rs)-2):
                if rs[i+2] - rs[i] <= 4:
                    straighty = True; break
        wet = flushy or straighty or paired
        return ('wet' if wet else 'dry', paired)

    # EV helpers / bet sizing 
    def _ev_call_positive(self, equity, pot, call_amt):
        return equity * (pot + call_amt) - (1 - equity) * call_amt > 0

    def _mix(self, p): return random.random() < p

    def _open_size_bb(self, pos):
        return {'BTN': 2.2, 'CO': 2.5, 'MP': 2.7, 'UTG': 3.0}.get(pos, 2.5)

    def _threebet_size(self, call_amount, in_position):
        mult = 3.0 if in_position else 3.5
        return int(max(self.big_blind_amount * 2, call_amount * mult))

    def _fourbet_size(self, call_amount):
        return int(max(self.big_blind_amount * 5, call_amount * 2.2))

    # equitys-
    def _build_live_deck(self, known):
        return [c for c in self._full_deck if c not in known]

    def _build_opp_combos(self, live, opp_keys):
        if not opp_keys: return []
        keyset = set(opp_keys)
        combos = []
        L = len(live)
        for i in range(L-1):
            ci = live[i]
            for j in range(i+1, L):
                cj = live[j]
                for key in keyset:
                    if self._combo_matches_key(ci, cj, key):
                        combos.append((ci, cj)); break
        return combos

    def _equity_cache_key(self, my_hole, board, opp_keys, sims_hint):
        return (tuple(sorted(my_hole)), tuple(sorted(board)), tuple(sorted(opp_keys)), sims_hint, len(board))

    def _calculate_equity_fast(self, my_hole, board, opp_keys, num_sims=200, early_eps=0.02):
        cached = self._equity_cache.get(self._equity_cache_key(my_hole, board, opp_keys, num_sims//50))
        if cached is not None: return cached

        known = set(my_hole + board)
        live = self._build_live_deck(known)
        opp_combos = self._build_opp_combos(live, opp_keys)
        if not opp_combos:
            self._equity_cache.put(self._equity_cache_key(my_hole, board, opp_keys, num_sims//50), 0.0)
            return 0.0

        wins = 0.0; iters = 0; need = 5 - len(board)

        def confident(p, n):
            if n < 80: return False
            se = (p*(1-p)/n) ** 0.5
            return se < early_eps

        for _ in range(num_sims):
            opp = random.choice(opp_combos); opp_set = set(opp)
            drawn = set()
            while len(drawn) < need:
                c = random.choice(live)
                if (c not in drawn) and (c not in opp_set):
                    drawn.add(c)
            full_board = board + list(drawn)

            my_rank  = self.evaluator.evaluate(my_hole, full_board)
            opp_rank = self.evaluator.evaluate(list(opp), full_board)
            if my_rank < opp_rank: wins += 1.0
            elif my_rank == opp_rank: wins += 0.5
            iters += 1
            p = wins / iters
            if confident(p, iters): break

        result = wins / iters if iters else 0.0
        self._equity_cache.put(self._equity_cache_key(my_hole, board, opp_keys, num_sims//50), result)
        return result

    # ---------- PPE callbacks ----------
    def receive_game_start_message(self, game_info):
        self.big_blind_amount = game_info["rule"]["small_blind_amount"] * 2

    def receive_round_start_message(self, round_count, hole_card, seats):
        # Make hand_id uinque
        self.hand_id = self.hand_offset + round_count
        self.was_pfr = False
        self.raised_this_street = False
        self.start_stacks = {s['name']: s['stack'] for s in seats}

    def receive_street_start_message(self, street, round_state):
        self.raised_this_street = False

    def declare_action(self, valid_actions, hole_card_str, round_state):
        global GLOBAL_STEP

        # Common vars
        my_hole_ints = [convert_card(c) for c in hole_card_str]
        board_ints   = [convert_card(c) for c in round_state['community_card']]
        pot_size = round_state['pot']['main']['amount']
        call_action  = next((a for a in valid_actions if a['action'] == 'call'), None)
        raise_action = next((a for a in valid_actions if a['action'] == 'raise'), None)
        call_amount  = call_action['amount'] if call_action else 0
        action, amount = 'fold', 0

        street = round_state['street'].lower()
        equity_val = None

        # Preflop logic 
        if not board_ints and street == 'preflop':
            if call_action and call_amount == 0:
                action, amount = 'call', 0
            else:
                pos = self._preflop_pos(round_state)
                hand_key   = hand_to_key(hole_card_str)
                percentile = HAND_PERCENTILES.get(hand_key, HAND_PERCENTILES["any_other"])

                pre_hist  = round_state['action_histories'].get('preflop', [])
                n_raises  = sum(1 for a in pre_hist if a.get('action') == 'RAISE')

                if pos == 'BB' and call_action:
                    if call_amount == 0:
                        action, amount = 'call', 0
                    else:
                        price_bb = call_amount / self.big_blind_amount
                        defend_cut = 70 if price_bb <= 1.0 else (55 if price_bb <= 1.5 else 40)
                        defend_cut += self.style["bb_defend_bump"]
                        defend_cut = max(20, min(85, defend_cut))
                        if percentile <= defend_cut:
                            if raise_action and percentile <= self.style["threebet_top_pct"] and self._mix(0.35):
                                target = self._threebet_size(call_amount, in_position=False)
                                target = int(target * random.uniform(0.93, 1.07))
                                amt = min(raise_action['amount']['max'], max(raise_action['amount']['min'], target))
                                action, amount = 'raise', amt
                                self.was_pfr = (n_raises == 0)
                            else:
                                action, amount = 'call', call_amount
                        else:
                            action, amount = 'fold', 0

                elif pos == 'SB' and call_action:
                    sb_complete = (call_amount == max(1, self.big_blind_amount // 2))
                    if sb_complete and n_raises == 0:
                        if raise_action and percentile <= 12 and self._mix(0.40):
                            target = int(self.big_blind_amount * 3)
                            target = int(target * random.uniform(0.93, 1.07))
                            amt = min(raise_action['amount']['max'], max(raise_action['amount']['min'], target))
                            action, amount = 'raise', amt
                            self.was_pfr = True
                        else:
                            action, amount = 'call', call_amount
                    else:
                        if percentile <= 18 and raise_action:
                            target = self._threebet_size(call_amount, in_position=False)
                            target = int(target * random.uniform(0.93, 1.07))
                            amt = min(raise_action['amount']['max'], max(raise_action['amount']['min'], target))
                            action, amount = 'raise', amt
                            self.was_pfr = (n_raises == 0)
                        elif percentile <= 45:
                            action, amount = 'call', call_amount
                        else:
                            action, amount = 'fold', 0

                else:
                    if pos in ('UTG','UTG+1','MP'):
                        range_cfg = {"raise": 12, "call": 22}
                    elif pos in ('HJ',):
                        range_cfg = {"raise": 16, "call": 28}
                    elif pos == 'CO':
                        range_cfg = {"raise": 22, "call": 38}
                    elif pos == 'BTN':
                        range_cfg = {"raise": 32, "call": 48}
                    else:
                        range_cfg = {"raise": 0, "call": 0}

                    range_cfg["raise"] = max(1, min(80, range_cfg["raise"] + self.style["raise_bump"]))
                    range_cfg["call"]  = max(1, min(80, range_cfg["call"]  + self.style["call_bump"]))

                    if n_raises == 0:
                        if raise_action and percentile <= range_cfg["raise"]:
                            open_bb = self._open_size_bb(pos)
                            target = int(open_bb * self.big_blind_amount)
                            target = int(target * random.uniform(0.93, 1.07))
                            amt = min(raise_action['amount']['max'], max(raise_action['amount']['min'], target))
                            if self._mix(0.80) or not call_action:
                                action, amount = 'raise', amt
                                self.was_pfr = True
                            else:
                                action, amount = 'call', call_amount
                        elif call_action and percentile <= range_cfg["call"]:
                            action, amount = 'call', call_amount
                        else:
                            action, amount = 'fold', 0
                    elif n_raises == 1:
                        top_pct = self.style["threebet_top_pct"]
                        if raise_action and percentile <= top_pct:
                            target = self._threebet_size(call_amount, in_position=True)
                            target = int(target * random.uniform(0.93, 1.07))
                            amt = min(raise_action['amount']['max'], max(raise_action['amount']['min'], target))
                            action, amount = 'raise', amt
                        elif call_action and percentile <= 38:
                            action, amount = 'call', call_amount
                        else:
                            action, amount = 'fold', 0
                    else:
                        if raise_action and percentile <= 5:
                            target = self._fourbet_size(call_amount)
                            target = int(target * random.uniform(0.93, 1.07))
                            amt = min(raise_action['amount']['max'], max(raise_action['amount']['min'], target))
                            action, amount = 'raise', amt
                        elif call_action and percentile <= 12:
                            action, amount = 'call', call_amount
                        else:
                            action, amount = 'fold', 0

        #  Postflop logic 
        else:
            if call_action and call_amount == 0 and not raise_action:
                equity_val = None
            else:
                opp_range = self._estimate_opponent_range(round_state)
                board_tex, _ = self._board_texture(board_ints)
                sims = 150 if (board_tex == 'dry' and len(board_ints) <= 3) else 200
                equity_val = self._calculate_equity_fast(
                    my_hole_ints, board_ints, opp_range, num_sims=sims, early_eps=0.02
                )

            board_tex, _ = self._board_texture(board_ints)
            is_dry = (board_tex == 'dry')

            # Baseline bet fractions 
            if raise_action:
                if is_dry:
                    bet_frac_value = 0.33 * random.uniform(0.92, 1.08)
                    bet_frac_bluff = 0.33 * random.uniform(0.92, 1.08)
                else:
                    bet_frac_value = 0.60 * random.uniform(0.92, 1.08)
                    bet_frac_bluff = 0.50 * random.uniform(0.92, 1.08)
            else:
                bet_frac_value = bet_frac_bluff = 0.0

            ev_ok = (self._ev_call_positive(equity_val, pot_size, call_amount)
                     if (call_amount > 0 and equity_val is not None) else True)

            if street == 'flop':
                value_thresh = 0.62
                semi_bluff_prob = 0.25
                is_check_to_me = (call_action is not None and call_amount == 0)
                cbet_freq = 0.65 + self.style["cbet_bump"]
                if raise_action and self.was_pfr and is_dry and is_check_to_me and self._mix(max(0.05, min(0.95, cbet_freq))):
                    bet_size = int(pot_size * bet_frac_value)
                    amt = min(raise_action['amount']['max'], max(raise_action['amount']['min'], bet_size))
                    action, amount = 'raise', amt
            elif street == 'turn':
                value_thresh = 0.66
                semi_bluff_prob = 0.18
            else:
                value_thresh = 0.70
                semi_bluff_prob = 0.10

            semi_bluff_prob = max(0.0, min(0.95, semi_bluff_prob + self.style["semi_bluff_bump"]))

            if action == 'fold':
                if raise_action and (equity_val is not None) and equity_val > value_thresh:
                    bet_size = int(pot_size * bet_frac_value)
                    amt = min(raise_action['amount']['max'], max(raise_action['amount']['min'], bet_size))
                    action, amount = 'raise', amt
                elif raise_action and (equity_val is not None) and 0.33 < equity_val < 0.50 and self._mix(semi_bluff_prob):
                    bet_size = int(pot_size * bet_frac_bluff)
                    amt = min(raise_action['amount']['max'], max(raise_action['amount']['min'], bet_size))
                    action, amount = 'raise', amt
                elif call_action and ev_ok:
                    action, amount = 'call', call_amount
                else:
                    action, amount = ('call', 0) if (call_action and call_amount == 0) else ('fold', 0)

        # monster exception
        if action == 'raise':
            eq = equity_val if equity_val is not None else (1.0 if street == 'preflop' else 0.0)
            if self.raised_this_street and (street == 'preflop' or eq < 0.80):
                if call_action is not None:
                    action, amount = 'call', call_amount
                else:
                    action, amount = ('call', 0) if (call_action and call_amount == 0) else ('fold', 0)
            else:
                self.raised_this_street = True

        # Validity guard 
        if action not in [a['action'] for a in valid_actions]:
            action, amount = 'fold', 0

        # Logging 
  
        my_stack, opp_stack, _ = self._my_and_opp_stacks(round_state)
        eff_stack_chips = min(my_stack, opp_stack)
        eff_stack_bb = eff_stack_chips / self.big_blind_amount if self.big_blind_amount else 0.0
        pot_bb = pot_size / self.big_blind_amount if self.big_blind_amount else 0.0
        spr = (eff_stack_bb / pot_bb) if pot_bb > 1e-9 else 0.0

        pre_hist = round_state['action_histories'].get('preflop', [])
        raises_preflop = sum(1 for a in pre_hist if a.get('action') == 'RAISE')
        street_hist = round_state['action_histories'].get(round_state['street'].lower(), [])
        raises_this_street = sum(1 for a in street_hist if a.get('action') == 'RAISE')

        pot_type = ('limped' if raises_preflop == 0 else
                    'single_raised' if raises_preflop == 1 else
                    'three_bet' if raises_preflop == 2 else 'four_bet_plus')

        street_index = {'preflop':0, 'flop':1, 'turn':2, 'river':3}.get(street, -1)
        pos_label = self._preflop_pos(round_state)
        in_pos = self._in_position_hu(street, round_state)

        # no leakage of future cards
        board_padded = round_state['community_card'] + [""] * (5 - len(round_state['community_card']))
        if street == 'preflop':
            board_padded = ["", "", "", "", ""]
        elif street == 'flop':
            board_padded[3] = ""; board_padded[4] = ""
        elif street == 'turn':
            board_padded[4] = ""
        # river

        # check vs call label
        is_check = (call_action is not None and call_amount == 0)
        logged_action = 'check' if (action == 'call' and is_check) else action

        # clamp raise amount
        raise_to_bb = (amount / self.big_blind_amount) if (action == 'raise' and self.big_blind_amount) else 0.0
        added = max(0, amount - call_amount) if (action == 'raise') else 0
        bet_frac_of_pot = (added / pot_size) if (action == 'raise' and pot_size > 1e-9) else 0.0

        GLOBAL_STEP += 1
        row = {
      
            "hand_id": self.hand_id,
            "player_id": next(s['name'] for s in round_state['seats'] if s['uuid']==self.uuid),
            "round": round_state['street'],
            "hole1": hole_card_str[0], "hole2": hole_card_str[1],
            "flop1": board_padded[0],  "flop2": board_padded[1], "flop3": board_padded[2],
            "turn":  board_padded[3],  "river": board_padded[4],
            "stack_bb": my_stack / self.big_blind_amount if self.big_blind_amount else 0.0,
            "pot_bb":   pot_bb,
            "to_call_bb": (call_amount / self.big_blind_amount) if (call_action and self.big_blind_amount) else 0.0,
            "action": logged_action,
            "result_bb": None, "won_flag": None, "final_pot_bb": None,

           #training features
            "step": GLOBAL_STEP,
            "source": "sim",
            "style": self.style_name,

            "opp_stack_bb": opp_stack / self.big_blind_amount if self.big_blind_amount else 0.0,
            "effective_stack_bb": eff_stack_bb,
            "spr": spr,

            "pos": pos_label,
            "in_position": in_pos,
            "was_pfr": 1 if self.was_pfr else 0,
            "pot_type": pot_type,
            "raises_preflop": raises_preflop,
            "raises_this_street": raises_this_street,
            "street_index": street_index,
            "board_texture": self._board_texture(board_ints)[0],

            "raise_to_bb": raise_to_bb,
            "bet_frac_of_pot": bet_frac_of_pot,

            # weights and labels to be filled in later
            "per_hand_rows": None,
            "row_weight_decisionshare": None,
            "row_weight_from_result": None,
            "sample_weight": None,
        }
        HAND_BUFFER[self.hand_id].append(row)
        return action, amount

    def receive_game_update_message(self, action, round_state): pass

    def receive_round_result_message(self, winners, hand_info, round_state):

        final_pot_bb = round_state['pot']['main']['amount'] / self.big_blind_amount if self.big_blind_amount else 0.0
        end_stacks = {s['name']: s['stack'] for s in round_state['seats']}

        # result in BB 
        result_bb_by_name = {}
        for name, start_chips in self.start_stacks.items():
            end_chips = end_stacks.get(name, start_chips)
            result_bb_by_name[name] = (end_chips - start_chips) / self.big_blind_amount if self.big_blind_amount else 0.0

        rows = HAND_BUFFER.pop(self.hand_id, [])
        rows.sort(key=lambda r: r.get("step", 0))

        # decision counts
        counts = defaultdict(int)
        for r in rows:
            counts[(r["hand_id"], r["player_id"])] += 1

        # labels and weights
        for r in rows:
            rb = result_bb_by_name.get(r["player_id"], 0.0)
            r["result_bb"] = rb
            r["won_flag"] = 1 if rb > 0 else 0
            r["final_pot_bb"] = final_pot_bb

            per_rows = counts[(r["hand_id"], r["player_id"])]
            r["per_hand_rows"] = per_rows
            r["row_weight_decisionshare"] = 1.0 / max(1, per_rows)

            CLIP_UP = 10.0
            w_reward = max(-CLIP_UP, min(CLIP_UP, rb))
            w_reward = w_reward if rb >= 0 else 0.5 * w_reward  # slightly downweight losses
            r["row_weight_from_result"] = w_reward

            r["sample_weight"] = r["row_weight_decisionshare"] * (abs(w_reward) + 1.0)

            ROWS.append(r)

        # periodic flush
        global _writer
        if (FLUSH_EVERY_N_HANDS is not None) and (_writer is not None):
            if (self.hand_id % FLUSH_EVERY_N_HANDS) == 0:
                _writer.writerows(ROWS)
                ROWS.clear()
    

# write 500 separate CSVs into "Poker Data (Bot)"

if __name__ == "__main__":
    out_dir = "Poker Data (Bot)"
    os.makedirs(out_dir, exist_ok=True)

    for i in range(1, 501):
        _reset_globals()

        OUTFILE = os.path.join(out_dir, f"data(bot)_{i:03d}.csv")
        f = open(OUTFILE, "w", newline="")

        header = [
            # original 17
            "hand_id","player_id","round","hole1","hole2","flop1","flop2","flop3","turn","river",
            "stack_bb","pot_bb","to_call_bb","action","result_bb","won_flag","final_pot_bb",
            # additions
            "step","source","style",
            "opp_stack_bb","effective_stack_bb","spr",
            "pos","in_position","was_pfr","pot_type","raises_preflop","raises_this_street","street_index","board_texture",
            "raise_to_bb","bet_frac_of_pot",
            "per_hand_rows","row_weight_decisionshare","row_weight_from_result","sample_weight",
        ]
        _writer = csv.DictWriter(f, fieldnames=header)
        _writer.writeheader()

        # Create persistent bot instances 
        tag = SuperBot(style_name="TAG")
        lag = SuperBot(style_name="LAG")

        print(f"Starting multi-session simulation for file {i}/500 -> {OUTFILE}")
        t0 = time.perf_counter()

        sessions_needed = max(1, (TARGET_TOTAL_HANDS + ROUNDS_PER_SESSION - 1) // ROUNDS_PER_SESSION)
        for s in range(sessions_needed):
            # keep hand_id unique 
            tag.hand_offset = s * (ROUNDS_PER_SESSION + 5)
            lag.hand_offset = s * (ROUNDS_PER_SESSION + 5)

            config = setup_config(max_round=ROUNDS_PER_SESSION,
                                  initial_stack=INITIAL_STACK_CHIPS,
                                  small_blind_amount=SMALL_BLIND_CHIPS)
            config.register_player(name="tag_bot", algorithm=tag)
            config.register_player(name="lag_bot", algorithm=lag)
            start_poker(config, verbose=0)

        # Final flush
        if ROWS:
            _writer.writerows(ROWS)
            ROWS.clear()
        f.close()

        elapsed = time.perf_counter() - t0
        print(f"File {i}/500 complete in {elapsed:.2f}s — saved to {OUTFILE}")

