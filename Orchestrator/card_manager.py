# Handles hole cards, as well as community cards

class CardManager:
    def __init__(self):
        self.hole_cards = {}
        self.community_cards = []

    def set_hole_cards(self, player, cards):
        self.hole_cards[player] = cards

    def add_community_cards(self, cards):
        """Add cards to community cards (don't replace existing ones)"""
        # Extend the list instead of replacing it
        self.community_cards.extend(cards)
