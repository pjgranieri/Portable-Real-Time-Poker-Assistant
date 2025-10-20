# Handles configuration of player constants, enums etc...
from enum import Enum

class Player(Enum):
    PlayerCoach = 0
    PlayerOne = 1
    PlayerTwo = 2
    PlayerThree = 3

class GameState(Enum):
    WAIT_FOR_GAME_START = 0
    WAIT_FOR_HOLE_CARDS = 1
    PRE_FLOP_BETTING = 2
    WAIT_FOR_FLOP = 3
    POST_FLOP_BETTING = 4
    WAIT_FOR_TURN_CARD = 5
    TURN_BETTING = 6
    WAIT_FOR_RIVER_CARD = 7
    RIVER_BETTING = 8
    SHOWDOWN = 9

class CropRegion(Enum):
    NO_CROP = 0
    CROP_LEFT = 1
    CROP_MIDDLE = 2
    CROP_RIGHT = 3
    CROP_CARDS = 4
