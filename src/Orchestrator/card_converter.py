"""
Card Converter - Converts card notation to ML-compatible format
Converts from VALUE|SUIT (e.g., "AH", "10D") to SUIT|VALUE (e.g., "HA", "DT")
"""

class CardConverter:
    """Converts card notation between formats"""
    
    VALID_VALUES = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    VALID_SUITS = ['C', 'D', 'H', 'S']
    
    @staticmethod
    def convert_to_ml_format(card):
        """
        Convert card from VALUE|SUIT to SUIT|VALUE format
        
        Args:
            card: String like "AH", "10D", "KS"
        
        Returns:
            String like "HA", "DT", "SK" or "" if invalid
        
        Examples:
            "3D" -> "D3"
            "10D" -> "DT"
            "QD" -> "DQ"
            "AH" -> "HA"
        """
        if not card or len(card) < 2:
            return ""
        
        card = card.upper().strip()
        
        # Extract suit (last character)
        suit = card[-1]
        
        # Extract value (everything before suit)
        value = card[:-1]
        
        # Convert "10" to "T"
        if value == "10":
            value = "T"
        
        # Validate
        if suit not in CardConverter.VALID_SUITS:
            print(f"Invalid suit: {suit} (must be C, D, H, or S)")
            return ""
        
        if value not in CardConverter.VALID_VALUES:
            print(f"Invalid value: {value} (must be A, 2-9, T, J, Q, or K)")
            return ""
        
        # Return SUIT|VALUE format
        return f"{suit}{value}"
    
    @staticmethod
    def convert_from_ml_format(card):
        """
        Convert card from SUIT|VALUE to VALUE|SUIT format
        
        Args:
            card: String like "HA", "DT", "SK"
        
        Returns:
            String like "AH", "10D", "KS" or "" if invalid
        
        Examples:
            "D3" -> "3D"
            "DT" -> "10D"
            "DQ" -> "QD"
            "HA" -> "AH"
        """
        if not card or len(card) < 2:
            return ""
        
        card = card.upper().strip()
        
        # Extract suit (first character)
        suit = card[0]
        
        # Extract value (everything after suit)
        value = card[1:]
        
        # Convert "T" to "10"
        if value == "T":
            value = "10"
        
        # Validate
        if suit not in CardConverter.VALID_SUITS:
            print(f"Invalid suit: {suit} (must be C, D, H, or S)")
            return ""
        
        if value not in CardConverter.VALID_VALUES and value != "10":
            print(f"Invalid value: {value} (must be A, 2-10, J, Q, or K)")
            return ""
        
        # Return VALUE|SUIT format
        return f"{value}{suit}"
    
    @staticmethod
    def validate_card(card):
        """
        Validate card input (in VALUE|SUIT format)
        
        Args:
            card: String like "AH", "10D", "KS"
        
        Returns:
            Boolean indicating if card is valid
        """
        if not card or len(card) < 2:
            return False
        
        card = card.upper().strip()
        
        # Extract suit (last character)
        suit = card[-1]
        
        # Extract value (everything before suit)
        value = card[:-1]
        
        # Check if 10 should be converted to T
        if value == "10":
            value = "T"
        
        # Validate
        return suit in CardConverter.VALID_SUITS and value in CardConverter.VALID_VALUES
    
    @staticmethod
    def is_valid_card(card):
        """Alias for validate_card"""
        return CardConverter.validate_card(card)


# Convenience functions
def convert_card(card):
    """Convert card to ML format"""
    return CardConverter.convert_to_ml_format(card)

def validate_card(card):
    """Validate card"""
    return CardConverter.validate_card(card)

def pypoker_to_internal(card_str):
    """
    Convert PyPokerEngine card format to internal format.
    PyPokerEngine uses format like: 'HA', 'D2', 'SK', 'CT'
    Internal format is same: 'HA', 'D2', 'SK', 'CT'
    
    Args:
        card_str: Card string from PyPokerEngine (e.g., 'HA', 'D2')
    
    Returns:
        Card string in internal format
    """
    if not card_str or len(card_str) != 2:
        return ""
    
    # PyPokerEngine already uses SUIT+RANK format, which matches internal
    # Just need to validate and return
    suit = card_str[0].upper()
    rank = card_str[1].upper()
    
    # Validate
    if suit not in ['C', 'D', 'H', 'S']:
        return ""
    if rank not in ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']:
        return ""
    
    return suit + rank