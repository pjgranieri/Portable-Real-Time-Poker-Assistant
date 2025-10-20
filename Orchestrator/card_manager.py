# Handles hole cards, as well as community cards

# card_manager.py
class CardManager:
    def __init__(self):
        self.hole_cards = {}
        self.community_cards = []

    def set_hole_cards(self, player, cards):
        self.hole_cards[player] = cards

    def add_community_cards(self, cards):
        self.community_cards = cards
