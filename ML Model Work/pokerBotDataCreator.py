import csv
import random
import itertools
from treys import Evaluator, Card, Deck
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer

# ----------------------------
# Helper Functions
# ----------------------------
def hand_to_key(hole_cards_str):
    """Converts ['S2', 'HA'] to a key like 'A2s' or 'A2o'."""
    rank_order = "23456789TJQKA"
    r1, r2 = hole_cards_str[0][1], hole_cards_str[1][1]
    s1, s2 = hole_cards_str[0][0], hole_cards_str[1][0]
    # normalize so r1 is the higher rank
    if rank_order.index(r2) > rank_order.index(r1):
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    if r1 == r2:
        return r1 + r2
    return (r1 + r2 + "s") if s1 == s2 else (r1 + r2 + "o")

def convert_card(card_str):
    """Converts 'SA' to 'As' for the treys library."""
    return card_str[1] + card_str[0].lower()

# ----------------------------
# Preflop Charts & Playstyles
# ----------------------------
HAND_PERCENTILES = {
    "AA": 1, "KK": 1, "QQ": 1, "JJ": 1, "AKs": 1, "TT": 5, "AQs": 5, "AJs": 5, "KQs": 5, "AKo": 5,
    "99": 10, "ATs": 10, "KJs": 10, "QJs": 10, "JTs": 10, "AQo": 10, "88": 15, "KTs": 15, "Q9s": 15,
    "T9s": 15, "98s": 15, "AJo": 15, "KQo": 15, "77": 20, "A9s": 20, "A8s": 20, "A7s": 20, "A6s": 20,
    "A5s": 20, "A4s": 20, "A3s": 20, "A2s": 20, "K9s": 20, "J9s": 20, "T8s": 20, "87s": 20, "76s": 20,
    "65s": 20, "any_other": 75
}
# Expand to include playable offsuit broadways / gappers so they’re not all "trash"
HAND_PERCENTILES.update({
    "ATo": 22, "KJo": 28, "QJo": 30, "KTo": 35, "QTo": 38, "JTo": 32,
    "T9o": 40, "98o": 48, "A9o": 35, "K9o": 42, "Q9o": 48
})

# Style knobs for TAG vs LAG
STYLE = {
    "TAG": {
        "raise_bump": -3,      # tighter opens/raises
        "call_bump":  -5,      # tighter flats
        "bb_defend_bump": -5,  # tighter BB defence
        "semi_bluff_bump": -0.05,  # lower semi-bluff frequency
        "cbet_bump": -0.10,        # lower c-bet frequency
        "threebet_top_pct": 10
    },
    "LAG": {
        "raise_bump": +4,      # looser opens/raises
        "call_bump":  +8,      # looser flats
        "bb_defend_bump": +10, # wider BB defence
        "semi_bluff_bump": +0.07,  # higher semi-bluff frequency
        "cbet_bump": +0.10,        # higher c-bet frequency
        "threebet_top_pct": 18
    }
}

# ----------------------------
# The SuperBot AI Class
# ----------------------------
class SuperBot(BasePokerPlayer):
    def __init__(self, style_name="TAG"):
        self.evaluator = Evaluator()
        self.hand_id = 0
        self.big_blind_amount = 0
        self.style = STYLE[style_name]
        self.was_pfr = False  # preflop aggressor flag for c-bet logic

    # 1) Position/blind logic -------------------------------------------------
    def _preflop_pos(self, round_state):
        """Return one of UTG/MP/CO/BTN/SB/BB (for 6-9 max), or BTN/SB/BB (3-max), SB/BB for HU."""
        seats = round_state['seats']
        n = len(seats)
        btn = round_state['dealer_btn']
        order = [(btn + 1 + i) % n for i in range(n)]  # UTG is first to act preflop

        labels_9max = ['UTG','UTG+1','MP','MP+1','HJ','CO','BTN','SB','BB']
        if n >= 9:
            labels = labels_9max
        elif n >= 6:
            labels = ['UTG','MP','CO','BTN','SB','BB'][-n:]
        elif n == 5:
            labels = ['MP','CO','BTN','SB','BB']
        elif n == 4:
            labels = ['CO','BTN','SB','BB']
        elif n == 3:
            labels = ['BTN','SB','BB']
        else:
            labels = ['SB','BB']  # heads-up

        me = next(i for i, s in enumerate(seats) if s['uuid'] == self.uuid)
        idx = order.index(me)
        return labels[idx]

    # 6) Stronger opponent-range estimator -----------------------------------
    def _estimate_opponent_range(self, round_state):
        """Tighten/widen opponent range based on preflop action history counts."""
        history = round_state['action_histories'].get('preflop', [])
        n_raises = sum(1 for a in history if a.get('action') == 'RAISE')
        n_calls  = sum(1 for a in history if a.get('action') == 'CALL')
        if n_raises >= 2:
            cutoff = 12
        elif n_raises == 1:
            cutoff = 20
        elif n_calls >= 2:
            cutoff = 40
        else:
            cutoff = 60
        return [hand for hand, p in HAND_PERCENTILES.items() if hand != "any_other" and p <= cutoff]

    # 2) Fair combo sampling from live deck -----------------------------------
    def _sample_hand_from_key(self, deck_cards, hand_key):
        """
        Sample an opponent combo from remaining deck that matches a key like 'AKs','AJo','TT'.
        deck_cards: list[int] (treys int cards)
        """
        r1, r2 = hand_key[0], hand_key[1]
        need_suited  = (len(hand_key) == 3 and hand_key[2] == 's')
        need_offsuit = (len(hand_key) == 3 and hand_key[2] == 'o')

        # convert deck ints -> strings ('As','Td',...)
        deck_str = [Card.int_to_str(ci) for ci in deck_cards]

        combos = []
        for c1, c2 in itertools.combinations(deck_str, 2):
            rank1, suit1 = c1[0], c1[1]
            rank2, suit2 = c2[0], c2[1]
            # pocket pair
            if r1 == r2:
                if rank1 == rank2 == r1:
                    combos.append((Card.new(c1), Card.new(c2)))
            else:
                # need the two ranks
                if {rank1, rank2} == {r1, r2}:
                    suited = (suit1 == suit2)
                    if (need_suited and suited) or (need_offsuit and not suited) or (not need_suited and not need_offsuit):
                        combos.append((Card.new(c1), Card.new(c2)))
        return random.choice(combos) if combos else None

    # 4) Board-aware heuristics + EV call check -------------------------------
    def _board_texture(self, board_cards_str):
        """
        board_cards_str: list like ['As','Td','7c'] (0-5 cards).
        Returns ('dry'|'wet', paired:bool)
        """
        if not board_cards_str:
            return ('dry', False)
        ranks = [c[0] for c in board_cards_str]
        suits = [c[1] for c in board_cards_str]
        paired = (len(ranks) != len(set(ranks)))
        flushy = any(suits.count(s) >= 3 for s in 'shdc')

        order = "A23456789TJQKA"  # duplicate 'A' to allow simple wrap proxy
        # guard against unexpected chars
        idxs = []
        for r in ranks:
            try:
                idxs.append(order.index(r))
            except ValueError:
                pass
        idxs.sort()

        straighty = False
        if len(idxs) >= 3:
            for j in range(len(idxs) - 2):
                if idxs[j+2] - idxs[j] <= 4:
                    straighty = True
                    break

        wet = flushy or straighty or paired
        return ('wet' if wet else 'dry', paired)

    def _ev_call_positive(self, equity, pot, call_amt):
        # EV(call) = equity*(pot+call) - (1-equity)*call
        return equity * (pot + call_amt) - (1 - equity) * call_amt > 0

    # 3) Mixed strategies helper ----------------------------------------------
    def _mix(self, p):
        return random.random() < p

    # 5) Bet sizing helpers (open/3b/4b scaffolding) --------------------------
    def _open_size_bb(self, pos):
        return {'BTN': 2.2, 'CO': 2.5, 'MP': 2.7, 'UTG': 3.0}.get(pos, 2.5)

    def _threebet_size(self, call_amount, in_position):
        # Roughly 3x IP, 3.5x OOP vs the call-to amount
        mult = 3.0 if in_position else 3.5
        return int(max(self.big_blind_amount * 2, call_amount * mult))

    def _fourbet_size(self, call_amount):
        # ~2.2x the 3-bet size baseline
        return int(max(self.big_blind_amount * 5, call_amount * 2.2))

    # Equity calc with fair sampling (pass lists to treys evaluator) ----------
    def _calculate_equity(self, my_hole_cards_str, community_cards_str, opp_range, num_sims=300):
        my_cards = [Card.new(c) for c in my_hole_cards_str]           # list[int]
        board    = [Card.new(c) for c in community_cards_str]         # list[int]
        wins = 0.0
        iters = 0

        for _ in range(num_sims):
            deck = Deck()

            # remove my cards and current board from deck
            for card in my_cards + board:
                if card in deck.cards:
                    deck.cards.remove(card)

            # build sampleable opp combos from keys (respecting blockers)
            opp_combos = []
            for hand_key in opp_range:
                sample = self._sample_hand_from_key(deck.cards, hand_key)
                if sample:
                    opp_combos.append(sample)
            if not opp_combos:
                continue

            opp_hand = random.choice(opp_combos)          # tuple(int,int)
            opp_hand_list = [opp_hand[0], opp_hand[1]]    # ensure list[int] for treys

            # remove chosen opp cards
            for c in opp_hand_list:
                if c in deck.cards:
                    deck.cards.remove(c)

            remaining_board = deck.draw(5 - len(board))   # list[int]
            # ---- Critical: pass lists to treys evaluator ----
            my_rank  = self.evaluator.evaluate(list(my_cards), list(board + remaining_board))
            opp_rank = self.evaluator.evaluate(opp_hand_list, list(board + remaining_board))

            if my_rank < opp_rank:
                wins += 1.0
            elif my_rank == opp_rank:
                wins += 0.5
            iters += 1

        return (wins / iters) if iters > 0 else 0.0

    def receive_game_start_message(self, game_info):
        self.big_blind_amount = game_info["rule"]["small_blind_amount"] * 2

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.hand_id = round_count
        self.was_pfr = False  # reset per hand

    def declare_action(self, valid_actions, hole_card_str, round_state):
        # Prepare common vars
        my_hole_cards_treys   = [convert_card(c) for c in hole_card_str]
        community_cards_treys = [convert_card(c) for c in round_state['community_card']]
        pot_size = round_state['pot']['main']['amount']
        call_action  = next((a for a in valid_actions if a['action'] == 'call'), None)
        raise_action = next((a for a in valid_actions if a['action'] == 'raise'), None)
        call_amount  = call_action['amount'] if call_action else 0
        action, amount = 'fold', 0

        # convenience
        street = round_state['street'].lower()

        # ---------------- Preflop block ----------------
        if not community_cards_treys and street == 'preflop':
            # free check (posting BB, etc.)
            if call_action and call_amount == 0:
                action, amount = 'call', 0
            else:
                pos = self._preflop_pos(round_state)
                hand_key   = hand_to_key(hole_card_str)
                percentile = HAND_PERCENTILES.get(hand_key, HAND_PERCENTILES["any_other"])

                # detect prior raises in history for 3-bet/4-bet scaffolding
                pre_hist  = round_state['action_histories'].get('preflop', [])
                n_raises  = sum(1 for a in pre_hist if a.get('action') == 'RAISE')

                # --- Blind-specific logic ---
                if pos == 'BB' and call_action:
                    if call_amount == 0:
                        action, amount = 'call', 0  # free check
                    else:
                        price_bb = call_amount / self.big_blind_amount
                        defend_cut = 70 if price_bb <= 1.0 else (55 if price_bb <= 1.5 else 40)
                        defend_cut += self.style["bb_defend_bump"]
                        defend_cut = max(20, min(85, defend_cut))
                        if percentile <= defend_cut:
                            # occasionally 3-bet top end
                            if raise_action and percentile <= self.style["threebet_top_pct"] and self._mix(0.35):
                                target = self._threebet_size(call_amount, in_position=False)
                                amt = min(raise_action['amount']['max'], max(raise_action['amount']['min'], target))
                                action, amount = 'raise', amt
                                self.was_pfr = (n_raises == 0)  # if we are first raiser
                            else:
                                action, amount = 'call', call_amount
                        else:
                            action, amount = 'fold', 0

                elif pos == 'SB' and call_action:
                    # typical SB complete price is 0.5bb (assuming no raise yet)
                    sb_complete = (call_amount == max(1, self.big_blind_amount // 2))
                    if sb_complete and n_raises == 0:
                        # mix completes; iso-raise small top range
                        if raise_action and percentile <= 12 and self._mix(0.40):
                            target = int(self.big_blind_amount * 3)
                            amt = min(raise_action['amount']['max'], max(raise_action['amount']['min'], target))
                            action, amount = 'raise', amt
                            self.was_pfr = True
                        else:
                            action, amount = 'call', call_amount
                    else:
                        # facing a raise in SB
                        if percentile <= 18 and raise_action:
                            target = self._threebet_size(call_amount, in_position=False)
                            amt = min(raise_action['amount']['max'], max(raise_action['amount']['min'], target))
                            action, amount = 'raise', amt
                            self.was_pfr = (n_raises == 0)
                        elif percentile <= 45:
                            action, amount = 'call', call_amount
                        else:
                            action, amount = 'fold', 0

                else:
                    # Non-blind positions by label
                    if pos in ('UTG','UTG+1','MP'):
                        range_cfg = {"raise": 12, "call": 22}
                    elif pos in ('HJ',):
                        range_cfg = {"raise": 16, "call": 28}
                    elif pos == 'CO':
                        range_cfg = {"raise": 22, "call": 38}
                    elif pos == 'BTN':
                        range_cfg = {"raise": 32, "call": 48}
                    else:
                        range_cfg = {"raise": 0, "call": 0}  # SB/BB handled above

                    # Style bumps
                    range_cfg["raise"] = max(1, min(80, range_cfg["raise"] + self.style["raise_bump"]))
                    range_cfg["call"]  = max(1, min(80, range_cfg["call"]  + self.style["call_bump"]))

                    # If no raises yet, choose open size; else vs raise use 3-bet/flat logic
                    if n_raises == 0:
                        if raise_action and percentile <= range_cfg["raise"]:
                            open_bb = self._open_size_bb(pos)
                            target = int(open_bb * self.big_blind_amount)
                            amt = min(raise_action['amount']['max'], max(raise_action['amount']['min'], target))
                            # Mixed: mostly raise, sometimes flat if possible
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
                        # facing a single open
                        top_pct = self.style["threebet_top_pct"]
                        if raise_action and percentile <= top_pct:
                            target = self._threebet_size(call_amount, in_position=True)
                            amt = min(raise_action['amount']['max'], max(raise_action['amount']['min'], target))
                            action, amount = 'raise', amt
                            # still not the original PFR, but we're aggressing now
                        elif call_action and percentile <= 38:
                            action, amount = 'call', call_amount
                        else:
                            action, amount = 'fold', 0
                    else:
                        # multi-raise pot (3-bet already out there) -> 4-bet or fold mostly
                        if raise_action and percentile <= 5:
                            target = self._fourbet_size(call_amount)
                            amt = min(raise_action['amount']['max'], max(raise_action['amount']['min'], target))
                            action, amount = 'raise', amt
                        elif call_action and percentile <= 12:
                            action, amount = 'call', call_amount
                        else:
                            action, amount = 'fold', 0

        # ---------------- Postflop block ----------------
        else:
            opp_range = self._estimate_opponent_range(round_state)

            equity = self._calculate_equity(
                my_hole_cards_treys, community_cards_treys, opp_range, num_sims=300
            )

            board_tex, paired = self._board_texture(community_cards_treys)
            is_dry = (board_tex == 'dry')

            # Baseline bet fractions (value/bluff)
            if raise_action:
                if is_dry:
                    bet_frac_value = 0.33
                    bet_frac_bluff = 0.33
                else:
                    bet_frac_value = 0.60
                    bet_frac_bluff = 0.50
            else:
                bet_frac_value = bet_frac_bluff = 0.0  # can't raise

            # EV-based call check
            pot_odds = call_amount / (pot_size + call_amount) if call_amount > 0 else 0.0
            ev_ok = self._ev_call_positive(equity, pot_size, call_amount) if call_amount > 0 else True

            # Per-street thresholds
            if street == 'flop':
                value_thresh = 0.62
                semi_bluff_prob = 0.25
                # PFR c-bet module: if we were the preflop raiser and it’s checked to us on a dry flop
                is_check_to_me = (call_action is not None and call_amount == 0)
                cbet_freq = 0.65 + self.style["cbet_bump"]
                if raise_action and self.was_pfr and is_check_to_me and is_dry and self._mix(max(0.05, min(0.95, cbet_freq))):
                    bet_size = int(pot_size * 0.33)
                    amt = min(raise_action['amount']['max'], max(raise_action['amount']['min'], bet_size))
                    action, amount = 'raise', amt
                # If c-bet path didn’t trigger, continue below
            elif street == 'turn':
                value_thresh = 0.66
                semi_bluff_prob = 0.18
            else:  # river
                value_thresh = 0.70
                semi_bluff_prob = 0.10

            semi_bluff_prob = max(0.0, min(0.95, semi_bluff_prob + self.style["semi_bluff_bump"]))

            # If not already chosen by c-bet path:
            if action == 'fold':
                # Value raise when strong enough for this street
                if raise_action and equity > value_thresh:
                    bet_size = int(pot_size * bet_frac_value)
                    amt = min(raise_action['amount']['max'], max(raise_action['amount']['min'], bet_size))
                    action, amount = 'raise', amt
                # Semi-bluff occasionally on mid equity (mix)
                elif raise_action and 0.33 < equity < 0.50 and self._mix(semi_bluff_prob):
                    bet_size = int(pot_size * bet_frac_bluff)
                    amt = min(raise_action['amount']['max'], max(raise_action['amount']['min'], bet_size))
                    action, amount = 'raise', amt
                # Otherwise call if EV(call) positive (or check if free)
                elif call_action and ev_ok:
                    action, amount = 'call', call_amount
                else:
                    action, amount = ('call', 0) if (call_action and call_amount == 0) else ('fold', 0)

        # Validity guard
        if action not in [a['action'] for a in valid_actions]:
            action, amount = 'fold', 0

        # Logging (same schema as before)
        my_seat_data = next((s for s in round_state['seats'] if s['uuid'] == self.uuid), {})
        board_padded = round_state['community_card'] + [""] * (5 - len(round_state['community_card']))
        writer.writerow({
            "hand_id": self.hand_id,
            "player_id": my_seat_data.get('name'),
            "round": round_state['street'],
            "hole1": hole_card_str[0],
            "hole2": hole_card_str[1],
            "flop1": board_padded[0],
            "flop2": board_padded[1],
            "flop3": board_padded[2],
            "turn":  board_padded[3],
            "river": board_padded[4],
            "stack_bb": my_seat_data.get('stack', 0) / self.big_blind_amount,
            "pot_bb":   pot_size / self.big_blind_amount,
            "to_call_bb": (call_amount / self.big_blind_amount) if call_action else 0.0,  # actual cost to continue
            "action": action
        })
        return action, amount

    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass


# ----------------------------
# CSV Setup and Simulation Run
# ----------------------------
OUTFILE = "superbot_training_data_patched.csv"
f = open(OUTFILE, "w", newline="")
header = ["hand_id", "player_id", "round", "hole1", "hole2",
          "flop1", "flop2", "flop3", "turn", "river",
          "stack_bb", "pot_bb", "to_call_bb", "action"]
writer = csv.DictWriter(f, fieldnames=header)
writer.writeheader()

config = setup_config(max_round=1000, initial_stack=1000, small_blind_amount=5)
config.register_player(name="tag_bot", algorithm=SuperBot(style_name="TAG"))
config.register_player(name="lag_bot", algorithm=SuperBot(style_name="LAG"))

print("Starting simulation.")
import time
t0 = time.perf_counter()

game_result = start_poker(config, verbose=0)
print(f"Elapsed: {time.perf_counter() - t0:.2f}s")
f.close()
print(f"Simulation complete. Saved data for {config.max_round} hands to {OUTFILE}")
