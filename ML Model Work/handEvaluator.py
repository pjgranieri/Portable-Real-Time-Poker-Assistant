from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import List, Tuple


@dataclass
class Card:
    value: int   # 2–14 (where 11=J, 12=Q, 13=K, 14=A)
    suit: str    # "hearts", "diamonds", "clubs", "spades"


class HandRank(Enum):
    HIGH_CARD = 1
    ONE_PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10


@dataclass
class EvaluatedHand:
    rank: HandRank
    highCard: int
    description: str
    kickers: List[int] = None


class HandEvaluator:

    @staticmethod
    def evaluateHand(cards: List[Card]) -> EvaluatedHand:
        if len(cards) < 5:
            raise ValueError("At least 5 cards required")
        all_combos = combinations(cards, 5)
        evaluated = [HandEvaluator.evaluateFiveCardHand(list(c)) for c in all_combos]
        return max(evaluated, key=lambda h: (
            h.rank.value, h.highCard, h.kickers or []))

    @staticmethod
    def evaluateFiveCardHand(cards: List[Card]) -> EvaluatedHand:
        values = sorted([c.value for c in cards], reverse=True)
        suits = [c.suit for c in cards]
        value_counts = {v: values.count(v) for v in set(values)}
        distinct_values = sorted(set(values), reverse=True)
        is_straight = HandEvaluator.isStraight(distinct_values)

        suit_groups = {}
        for c in cards:
            suit_groups.setdefault(c.suit, []).append(c)
        flush_group = next((g for g in suit_groups.values() if len(g) >= 5), None)
        is_flush = flush_group is not None
        flush_cards = sorted(flush_group, key=lambda x: x.value, reverse=True) if flush_group else []

        # Straight Flush / Royal Flush
        if is_flush and is_straight:
            flush_values = sorted(set([c.value for c in flush_cards]), reverse=True)
            sf_high = HandEvaluator.straightHighCard(flush_values)
            if sf_high == 14:
                return EvaluatedHand(HandRank.ROYAL_FLUSH, 14, "Royal Flush")
            return EvaluatedHand(HandRank.STRAIGHT_FLUSH, sf_high,
                                 f"Straight Flush to {HandEvaluator.rankToSymbol(sf_high)}")

        # Four of a Kind
        fours = [v for v, c in value_counts.items() if c >= 4]
        if fours:
            four = max(fours)
            kicker = max(v for v in values if v != four)
            return EvaluatedHand(HandRank.FOUR_OF_A_KIND, four,
                                 f"Four of a Kind ({HandEvaluator.rankToSymbol(four)}s)",
                                 [kicker])

        # Full House
        threes = [v for v, c in value_counts.items() if c >= 3]
        pairs = [v for v, c in value_counts.items() if c >= 2]
        if threes and (len(pairs) >= 2 or (pairs and pairs[0] != threes[0])):
            three = max(threes)
            pair = max([p for p in pairs if p != three], default=three)
            return EvaluatedHand(HandRank.FULL_HOUSE, three,
                                 f"Full House: {HandEvaluator.rankToSymbol(three)}s over {HandEvaluator.rankToSymbol(pair)}s")

        # Flush
        if is_flush:
            high = flush_cards[0].value
            kickers = [c.value for c in flush_cards[1:]]
            return EvaluatedHand(HandRank.FLUSH, high,
                                 f"Flush, high card {HandEvaluator.rankToSymbol(high)}",
                                 kickers)

        # Straight
        if is_straight:
            high = HandEvaluator.straightHighCard(distinct_values)
            return EvaluatedHand(HandRank.STRAIGHT, high,
                                 f"Straight to {HandEvaluator.rankToSymbol(high)}")

        # Three of a Kind
        if threes:
            three = max(threes)
            kickers = [v for v in values if v != three][:2]
            return EvaluatedHand(HandRank.THREE_OF_A_KIND, three,
                                 f"Three of a Kind ({HandEvaluator.rankToSymbol(three)}s)", kickers)

        # Two Pair
        if len(pairs) >= 2:
            top_two = sorted(pairs, reverse=True)[:2]
            kicker = max(v for v in values if v not in top_two)
            return EvaluatedHand(HandRank.TWO_PAIR, top_two[0],
                                 f"Two Pair: {HandEvaluator.rankToSymbol(top_two[0])}s and {HandEvaluator.rankToSymbol(top_two[1])}s",
                                 [kicker])

        # One Pair
        if pairs:
            pair = max(pairs)
            kickers = [v for v in values if v != pair][:3]
            return EvaluatedHand(HandRank.ONE_PAIR, pair,
                                 f"One Pair of {HandEvaluator.rankToSymbol(pair)}s", kickers)

        # High Card
        return EvaluatedHand(HandRank.HIGH_CARD, values[0],
                             f"High Card {HandEvaluator.rankToSymbol(values[0])}",
                             values[1:])

    @staticmethod
    def isStraight(sortedValues: List[int]) -> bool:
        if len(sortedValues) < 5:
            return False
        unique = sorted(set(sortedValues), reverse=True)
        for i in range(len(unique) - 4):
            if all(unique[i + j] == unique[i] - j for j in range(5)):
                return True
        return set([14, 5, 4, 3, 2]).issubset(unique)

    @staticmethod
    def straightHighCard(values: List[int]) -> int:
        unique = sorted(set(values), reverse=True)
        for i in range(len(unique) - 4):
            if all(unique[i + j] == unique[i] - j for j in range(5)):
                return unique[i]
        return 5 if set([14, 5, 4, 3, 2]).issubset(unique) else -1

    @staticmethod
    def rankToSymbol(value: int) -> str:
        return {11: "J", 12: "Q", 13: "K", 14: "A"}.get(value, str(value))
