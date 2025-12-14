from typing import List, Dict, Any

try:
    from treys import Card, Evaluator
except ImportError:  # Fallback so tests that don't touch this still run
    Card = None
    Evaluator = None


_evaluator = Evaluator() if Evaluator is not None else None


def _card_str_to_treys(card: str) -> int:
    """Convert a card like 'HA', 'DK', 'C2' to a Treys int.

    Accepts both rank-first and suit-first encodings used in the project.
    Returns 0 if invalid or if Treys is unavailable.
    """
    if Card is None or not isinstance(card, str) or len(card) != 2:
        return 0

    c = card.strip().upper()
    if not c or len(c) != 2:
        return 0

    ranks = "23456789TJQKA"
    suits = "CDHS"

    # Format 1: rank + suit ("AH")
    if c[0] in ranks and c[1] in suits:
        rank, suit = c[0], c[1]
    # Format 2: suit + rank ("HA")
    elif c[0] in suits and c[1] in ranks:
        rank, suit = c[1], c[0]
    else:
        return 0

    try:
        # Treys expects lowercase suit letters
        return Card.new(rank + suit.lower())
    except Exception:
        return 0


def _evaluate_preflop_hand(hole_cards: List[str]) -> int:
    """
    Evaluate preflop hand strength and return bucket 0-4.
    
    Buckets:
    0 = Trash (offsuit low cards, big gaps)
    1 = Weak (small pairs, weak suited connectors, weak Ax)
    2 = Medium (medium pairs, good suited connectors, good broadways)
    3 = Strong (high pairs, premium suited cards, AK/AQ)
    4 = Premium (AA, KK, QQ, AKs)
    """
    if len(hole_cards) < 2:
        return 0
    
    ranks = "23456789TJQKA"
    
    # Parse cards
    c1, c2 = hole_cards[0].upper(), hole_cards[1].upper()
    
    # Extract rank and suit
    if c1[0] in ranks and c1[1] in "CDHS":
        r1, s1 = c1[0], c1[1]
    elif c1[1] in ranks and c1[0] in "CDHS":
        r1, s1 = c1[1], c1[0]
    else:
        return 0
    
    if c2[0] in ranks and c2[1] in "CDHS":
        r2, s2 = c2[0], c2[1]
    elif c2[1] in ranks and c2[0] in "CDHS":
        r2, s2 = c2[1], c2[0]
    else:
        return 0
    
    # Get numeric rank values
    rank1 = ranks.index(r1)
    rank2 = ranks.index(r2)
    high_rank = max(rank1, rank2)
    low_rank = min(rank1, rank2)
    gap = high_rank - low_rank
    
    is_suited = (s1 == s2)
    is_pair = (rank1 == rank2)
    
    # Pair evaluation - ALL POCKET PAIRS ARE PLAYABLE!
    if is_pair:
        if high_rank >= 10:  # QQ+ (Q=10, K=11, A=12)
            return 4  # Premium
        elif high_rank >= 7:  # 99, TT, JJ (9=7, T=8, J=9)
            return 3  # Strong
        elif high_rank >= 4:  # 66, 77, 88 (6=4, 7=5, 8=6)
            return 2  # Medium
        else:  # 22-55 - RAISED FROM 1 to 2 (small pairs are valuable for set mining)
            return 2  # Medium (was 1)
    
    # Two high cards (both T or higher)
    if low_rank >= 8:  # Both T+ (T=8, J=9, Q=10, K=11, A=12)
        if high_rank == 12 and low_rank >= 10:  # AK, AQ
            return 4 if is_suited else 3  # Premium suited, strong offsuit
        elif high_rank == 12:  # AJ, AT
            return 3 if is_suited else 2
        elif high_rank == 11:  # KQ, KJ, KT
            return 3 if is_suited else 2
        else:  # QJ, QT, JT
            return 2 if is_suited else 1
    
    # Ace with medium/low card
    if high_rank == 12:  # Ax
        if low_rank >= 6:  # A9-AT (already handled above)
            return 2 if is_suited else 1
        elif low_rank >= 3:  # A5-A8
            return 1 if is_suited else 0
        else:  # A2-A4
            return 1 if is_suited else 0  # Wheel potential
    
    # Suited connectors and gappers
    if is_suited:
        if gap == 0:  # Suited connectors
            if low_rank >= 7:  # 89s, 9Ts (already handled)
                return 2
            elif low_rank >= 4:  # 56s, 67s, 78s
                return 2
            else:  # 23s, 34s, 45s
                return 1
        elif gap == 1:  # One gap suited (86s, 97s, etc)
            if low_rank >= 6:
                return 2
            else:
                return 1
        elif gap == 2:  # Two gap suited (75s, 86s, etc)
            if low_rank >= 5:
                return 1
            else:
                return 0
    
    # Offsuit connectors
    if gap == 0 and not is_suited:
        if low_rank >= 7:  # 89o, 9To
            return 1
        else:
            return 0
    
    # Everything else is trash
    return 0


def evaluate_hand_features(hole_cards: List[str], board_cards: List[str]) -> Dict[str, Any]:
    """Return coarse hand / draw features using Treys.

    This is intentionally lightweight: we compute a single hand rank and
    bucket it, plus simple draw / missed-draw booleans.
    """
    if _evaluator is None or len(hole_cards) < 2:
        return {
            "hand_bucket": 0,
            "has_flush_draw": 0.0,
            "has_straight_draw": 0.0,
            "has_combo_draw": 0.0,
            "is_missed_draw_river": 0.0,
        }

    hero = [_card_str_to_treys(c) for c in hole_cards[:2]]
    board = [_card_str_to_treys(c) for c in board_cards]
    hero = [c for c in hero if c]
    board = [c for c in board if c]

    if len(hero) < 2:
        # Invalid hole cards
        return {
            "hand_bucket": 0,
            "has_flush_draw": 0.0,
            "has_straight_draw": 0.0,
            "has_combo_draw": 0.0,
            "is_missed_draw_river": 0.0,
        }
    
    if not board:
        # Preflop - evaluate hole card strength
        bucket = _evaluate_preflop_hand(hole_cards)
        return {
            "hand_bucket": float(bucket),
            "has_flush_draw": 0.0,
            "has_straight_draw": 0.0,
            "has_combo_draw": 0.0,
            "is_missed_draw_river": 0.0,
        }

    try:
        rank = _evaluator.evaluate(board, hero)
    except Exception:
        rank = 7462  # Worst

    # Map Treys rank (1 best .. 7462 worst) into 0..4 bucket
    # 0 = air, 1 = weak, 2 = medium, 3 = strong, 4 = nutted
    max_rank = 7462.0
    strength = 1.0 - (rank / max_rank)
    if strength < 0.25:
        bucket = 0
    elif strength < 0.45:
        bucket = 1
    elif strength < 0.70:
        bucket = 2
    elif strength < 0.88:
        bucket = 3
    else:
        bucket = 4

    # Very simple draw detection based on ranks/suits of hero+board
    # We keep this coarse to stay fast and implementation-light.
    all_cards = hero + board
    suits_count = {}
    ranks_seen = set()
    for c in all_cards:
        # Card.to_string(c) -> like 'Ah'. We guard in case of absence.
        try:
            s = Card.int_to_str(c)
            rank_char, suit_char = s[0], s[1]
        except Exception:
            continue
        suits_count[suit_char] = suits_count.get(suit_char, 0) + 1
        ranks = "23456789TJQKA"
        if rank_char in ranks:
            ranks_seen.add(ranks.index(rank_char))

    has_flush_draw = any(v >= 4 for v in suits_count.values())

    sorted_ranks = sorted(ranks_seen)
    has_straight_draw = False
    for i in range(len(sorted_ranks) - 2):
        if sorted_ranks[i + 2] - sorted_ranks[i] <= 4:
            has_straight_draw = True
            break

    has_combo_draw = has_flush_draw and has_straight_draw

    # Missed draw on river: river present, weak bucket, and at least one draw earlier.
    is_missed_draw_river = 0.0
    if len(board) == 5:
        if bucket <= 1 and (has_flush_draw or has_straight_draw):
            is_missed_draw_river = 1.0

    # PAIRED BOARD DETECTION: Check if board has a pair (or better)
    # This is CRITICAL for defense - paired boards are much more dangerous
    board_has_pair = False
    board_is_monotone = False  # All same suit (very dangerous for flush)
    board_is_connected = False  # Connected ranks (dangerous for straights)
    
    if len(board) >= 2:
        board_ranks = []
        board_suits = []
        
        for c in board:
            try:
                s = Card.int_to_str(c)
                rank_char = s[0]
                suit_char = s[1]
                ranks = "23456789TJQKA"
                if rank_char in ranks:
                    board_ranks.append(ranks.index(rank_char))
                    board_suits.append(suit_char)
            except Exception:
                continue
        
        # Check for paired board
        from collections import Counter
        rank_counts = Counter(board_ranks)
        board_has_pair = any(count >= 2 for count in rank_counts.values())
        
        # Check for monotone board (3+ cards of same suit on flop)
        if len(board) >= 3:
            suit_counts = Counter(board_suits)
            board_is_monotone = any(count >= 3 for count in suit_counts.values())
        
        # Check for connected board (ranks within 4 of each other = straight possible)
        # Tightened criteria: need ranks within 3 gap (like 987, JT9, 654)
        # This prevents false positives like 642 being marked as "connected"
        if len(board) >= 3:
            sorted_ranks = sorted(set(board_ranks))
            if len(sorted_ranks) >= 3:
                # Check if we have 3+ ranks within a 4-card span (more restrictive)
                for i in range(len(sorted_ranks) - 2):
                    if sorted_ranks[i+2] - sorted_ranks[i] <= 3:  # Changed from <= 4
                        board_is_connected = True
                        break

    # DANGEROUS BOARD PENALTIES: Downgrade bucket on paired/monotone/connected boards
    # These boards are MUCH more dangerous - opponents likely have made hands
    adjusted_bucket = bucket
    
    # Get hero ranks for checking trips/straights/flushes/overpairs
    hero_ranks = []
    hero_suits = []
    hero_rank_counts = {}
    for c in hero:
        try:
            s = Card.int_to_str(c)
            rank_char = s[0]
            suit_char = s[1]
            ranks = "23456789TJQKA"
            if rank_char in ranks:
                rank_idx = ranks.index(rank_char)
                hero_ranks.append(rank_idx)
                hero_suits.append(suit_char)
                hero_rank_counts[rank_idx] = hero_rank_counts.get(rank_idx, 0) + 1
        except Exception:
            continue
    
    # Check if hero has a pocket pair (both cards same rank)
    has_pocket_pair = any(count >= 2 for count in hero_rank_counts.values())
    
    # If hero has pocket pair, check if it's an overpair (higher than all board ranks)
    is_overpair = False
    if has_pocket_pair and len(board) >= 3:
        pocket_rank = [r for r, c in hero_rank_counts.items() if c >= 2][0]
        max_board_rank = max(board_ranks) if board_ranks else -1
        is_overpair = pocket_rank > max_board_rank
    
    # PAIRED BOARD PENALTY (but NOT if hero has overpair or strong hand)
    # Only penalize WEAK hands (bucket 1-2) that don't have trips
    if board_has_pair and len(board) >= 3 and not is_overpair and adjusted_bucket < 3.0:
        from collections import Counter
        board_rank_counts = Counter(board_ranks)
        paired_ranks = [rank for rank, count in board_rank_counts.items() if count >= 2]
        
        # Check if hero has trips+
        has_trips = any(rank in hero_ranks for rank in paired_ranks)
        
        if not has_trips:
            # Only downgrade weak pairs (bucket 1-2), not strong hands
            if 1.0 <= adjusted_bucket < 3.0:
                adjusted_bucket = max(0.0, adjusted_bucket - 0.5)  # Slight downgrade
    
    # MONOTONE BOARD PENALTY (3+ same suit = flush danger)
    # Only penalize WEAK hands (bucket 1-2) that don't have the flush
    if board_is_monotone and len(board) >= 3 and adjusted_bucket < 3.0:
        from collections import Counter
        board_suit_counts = Counter(board_suits)
        
        # Find the dominant suit
        for suit, count in board_suit_counts.items():
            if count >= 3:
                # Check if hero has 2 of that suit (making flush)
                hero_flush_count = hero_suits.count(suit)
                if hero_flush_count < 2:  # Don't have flush
                    # Only downgrade weak pairs (bucket 1-2), not strong hands
                    if 1.0 <= adjusted_bucket < 3.0:
                        adjusted_bucket = max(0.0, adjusted_bucket - 0.5)  # Slight downgrade
                break
    
    # CONNECTED BOARD PENALTY (straight possible)
    # Only penalize WEAK hands (bucket 1-2) on connected boards
    # Don't penalize overpairs or strong hands that likely have the straight
    if board_is_connected and len(board) >= 3 and not is_overpair and adjusted_bucket < 3.0:
        # Only downgrade weak pairs (bucket 1-2), not strong hands
        if 1.0 <= adjusted_bucket < 3.0:
            adjusted_bucket = max(0.0, adjusted_bucket - 0.5)  # Slight downgrade
    
    return {
        "hand_bucket": float(adjusted_bucket),
        "has_flush_draw": 1.0 if has_flush_draw else 0.0,
        "has_straight_draw": 1.0 if has_straight_draw else 0.0,
        "has_combo_draw": 1.0 if has_combo_draw else 0.0,
        "is_missed_draw_river": is_missed_draw_river,
        "board_has_pair": 1.0 if board_has_pair else 0.0,
        "board_is_monotone": 1.0 if board_is_monotone else 0.0,
        "board_is_connected": 1.0 if board_is_connected else 0.0,
    }
