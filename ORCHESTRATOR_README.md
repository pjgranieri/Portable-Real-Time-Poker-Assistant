# Poker Game Orchestrator Documentation

## Overview
This system implements a complete Texas Hold'em poker game orchestrator designed for heads-up play (Coach vs Player1) with integrated ML decision-making and computer vision support. The system uses a modular architecture with clear separation between testing and production environments.

## Architecture

### Components

#### 1. **Orchestrator** (`Orchestrator/`)
Core game logic and state management.

##### Configuration & State Management

- **`config.py`**: Enums and constants
  - `Player`: PlayerCoach (value=0), PlayerOne (value=1), PlayerTwo (value=2, disabled), PlayerThree (value=3, disabled)
  - `GameState`: Game flow states (WAIT_FOR_GAME_START, WAIT_FOR_HOLE_CARDS, PRE_FLOP_BETTING, etc.)
  - Note: Players 2 and 3 are disabled for current implementation (always folded)

- **`player_manager.py`**: Player state management
  - Bankroll tracking (starts at $175 per player)
  - Blind rotation (alternates between Coach and Player1)
  - Action detection via CV (production) or manual input (testing)
  - Hand evaluation via ML
  - **Players 2 & 3**: Always marked as folded, not included in active play

- **`card_manager.py`**: Card tracking
  - Hole cards storage per player
  - Community cards management (flop, turn, river)
  - Properly appends cards (doesn't overwrite on turn/river)

##### Game Logic

- **`betting_cycle.py`**: Betting round logic
  - Handles all betting rounds (pre-flop, flop, turn, river)
  - Automatic blind posting for pre-flop ($5 small blind, $10 big blind)
  - Validates actions (check/call/raise/fold)
  - Tracks amounts paid per round
  - Manages raise cycles (resets all players' "acted" status)
  - **Players 2 & 3**: Auto-fold at start of each cycle
  - Outputs ML JSON when Coach's turn arrives

- **`ml_json_input.py`**: ML model integration
  - Generates JSON payloads for ML model consumption
  - Tracks game state (hand_id, round, pot, stacks, actions)
  - Converts cards to ML format (SUIT|VALUE, e.g., "HA" for Ace of Hearts)
  - **Action tracking**: 
    - Simplified to just "fold", "call", "raise", "check"
    - Blank ("") if coach is first to act
  - **Only Player1 tracked**: opp_stack_bb and action fields only reference Player1

- **`card_converter.py`**: Card format conversion
  - Converts between VALUE|SUIT (input) ↔ SUIT|VALUE (ML format)
  - Examples:
    - "AH" → "HA" (Ace of Hearts)
    - "10D" → "DT" (Ten of Diamonds)
    - "3D" → "D3" (Three of Diamonds)
  - Validates card inputs (rejects invalid cards like "G104")
  - Handles "T" notation for 10s in ML format

##### User Interface & Integration

- **`input_interface.py`**: Standardized input interface
  - `get_card()`: Single card input with validation
  - `get_cards()`: Multiple card input (flop, turn, river)
  - `get_action()`: Player action input with validation
  - `get_winner_selection()`: Showdown winner selection
  - Easy to replace with CV/ML modules for production

- **`event_signals.py`**: External communication
  - Server crop mode control
  - CV module integration points
  - Signal waiting/polling

##### Execution Modes

- **`orchestrator.py`**: **Production mode** - Real gameplay
  - Integrates with CV for opponent actions and card detection
  - Integrates with ML for coach decision-making
  - Outputs JSON to ML model when coach must act
  - Graceful fallback to manual input if CV/ML unavailable
  - Clear TODO comments for integration points
  - Run with: `python -m Orchestrator.orchestrator`

- **`test_game.py`**: **Test mode** - Manual testing (located in root directory)
  - Simulates CV/ML with terminal inputs
  - Uses `InputInterface` for all user interactions
  - Overrides `player_manager` methods with mock versions
  - Run with: `python test_game.py`

- **`game_controller.py`**: Legacy FSM implementation (deprecated, use `orchestrator.py` instead)

#### 2. **Image Recognition** (`Image_Recognition/`)
Computer vision modules for game detection (to be integrated).

- **`action_detector.py`**: Hierarchical action detection
  1. Check for folded cards (blue rectangles)
  2. Check for hand gesture (check via skin color detection)
  3. Analyze chips (call/raise amount)
  - Polls `Outputs/` folder for new cropped images
  - Returns: `(action, value)` tuple

- **`card_analyzer.py`**: Card recognition
  - Detects community cards (flop, turn, river)
  - Reads player hands at showdown
  - Uses contour detection + template matching/OCR

- **`chip_analyzer.py`**: Chip value detection
  - Detects chip colors (white=$1, red=$5, blue=$10, green=$25, black=$100)
  - Counts chips via circular contour detection
  - Returns total bet value

#### 3. **ML Model** (to be integrated)
Machine learning module for coach decision-making.

Expected interface:
```python
def get_poker_action(json_data: dict) -> tuple:
    """
    Args:
        json_data: Parsed JSON from ml_json_input.py
    Returns:
        (action, value) where action is "fold", "check", "call", "raise"
        and value is raise amount (0 for fold/check/call)
    """
```

## Game Flow

### State Machine

```
WAIT_FOR_GAME_START
  ↓ (Initialize bankrolls=$175, rotate blinds between Coach/Player1)
  ↓ (Increment hand_id counter)
WAIT_FOR_HOLE_CARDS (Full camera view or manual input)
  ↓ (Detect 2 cards for coach)
PRE_FLOP_BETTING (Action zone cropping)
  ↓ (Auto-post blinds: SB=$5, BB=$10, betting cycle, ML JSON output for coach)
  ↓ (Check early win condition)
WAIT_FOR_FLOP (Community cards crop or manual input)
  ↓ (Detect 3 cards, append to community_cards)
POST_FLOP_BETTING (Action zone cropping)
  ↓ (Betting cycle, reset call_value=0, ML JSON output for coach)
  ↓ (Check early win condition)
WAIT_FOR_TURN_CARD (Community cards crop or manual input)
  ↓ (Detect 1 card, append to community_cards)
TURN_BETTING (Action zone cropping)
  ↓ (Betting cycle, ML JSON output for coach)
  ↓ (Check early win condition)
WAIT_FOR_RIVER_CARD (Community cards crop or manual input)
  ↓ (Detect 1 card, append to community_cards)
RIVER_BETTING (Action zone cropping)
  ↓ (Betting cycle, ML JSON output for coach)
SHOWDOWN (if 2+ players remain)
  ↓ (Read hands, evaluate winner, award pot)
→ WAIT_FOR_GAME_START (loop to next hand)
```

### Betting Cycle Logic

**Pre-flop:**
1. Small blind auto-posts $5 (deducted from bankroll)
2. Big blind auto-posts $10 (deducted from bankroll, call_value = $10)
3. Action starts after big blind (alternates between Coach/Player1)
4. Players act in turn: call $10, raise, or fold
5. Blinds get chance to act (call difference to match raise, re-raise, or fold)
6. If raise: everyone who hasn't folded must respond (call new amount, re-raise, or fold)
7. Round ends when all active players have acted and matched bets
8. **ML JSON**: Generated when coach must act, includes current game state

**Post-flop/Turn/River:**
1. Call value resets to $0
2. Action starts at small blind
3. Players can check (if call_value=$0), bet, or fold
4. If bet: others must call, raise, or fold
5. Round ends when all active players have acted and matched bets
6. **ML JSON**: Generated when coach must act

**Amount Tracking:**
- Tracks `amount_paid_this_round` per player
- `amount_needed_to_call = call_value - amount_already_paid`
- Example: Small blind raising to $20 (already paid $5, pays $15 more)

**Action Validation:**
- Cannot check if call_value > 0
- Cannot call if call_value = 0 (must check)
- Raise must be > current call_value
- Cannot raise more than bankroll

### ML JSON Format

Generated when coach must act:

```json
{
  "hand_id": 2,
  "player_id": 0,
  "round": "preflop",
  "hole1": "HA",
  "hole2": "DK",
  "flop1": "S7",
  "flop2": "C8",
  "flop3": "D9",
  "turn": "HT",
  "river": "",
  "stack_bb": 165,
  "opp_stack_bb": 165,
  "to_call_bb": 10,
  "pot_bb": 20,
  "action": "raise",
  "final_pot_bb": ""
}
```

**Field Notes:**
- Cards in SUIT|VALUE format (e.g., "HA" = Ace of Hearts, "DT" = Ten of Diamonds)
- `action`: Last opponent action ("fold", "call", "raise", "check", or "" if coach is first)
- `opp_stack_bb`: Only tracks Player1's stack
- `to_call_bb`: Actual amount coach needs to pay to call (not total call_value)
- `final_pot_bb`: Only filled at showdown

### Card Format Conversion

**Input Format (VALUE|SUIT):**
- User/CV enters: "AH", "10D", "KS", "7C"
- Valid values: A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K
- Valid suits: C (Clubs), D (Diamonds), H (Hearts), S (Spades)

**ML Format (SUIT|VALUE):**
- ML receives: "HA", "DT", "SK", "C7"
- 10 becomes "T": "10D" → "DT"
- Conversion handled automatically by `card_converter.py`

### Early Win Condition
If only 1 player remains after any betting round:
- Immediately award pot to remaining player
- Skip remaining rounds
- Return to WAIT_FOR_GAME_START

### Showdown
If 2+ players remain after river betting:
1. Read each player's hand via CV (or manual input in test mode)
2. Evaluate hands with ML module (or manual selection in test mode)
3. Award pot to winner
4. Display final bankrolls

## Current Configuration

### Active Players
- **PlayerCoach** (Player 0): User-controlled, ML-assisted
- **PlayerOne** (Player 1): Opponent, CV-detected or manual input

### Disabled Players
- **PlayerTwo** (Player 2): Always folded, not in play
- **PlayerThree** (Player 3): Always folded, not in play

### Blind Rotation
Blinds alternate between Coach and Player1:
- **Game 1**: Coach=SB ($5), Player1=BB ($10)
- **Game 2**: Player1=SB ($5), Coach=BB ($10)
- **Game 3**: Coach=SB ($5), Player1=BB ($10)
- And so on...

### Starting Bankrolls
- All players start each hand with $175
- Blinds are deducted before action begins

## Testing

### Test Mode (`test_game.py`)

Run with:
```bash
python test_game.py
```

**Test Flow:**
1. Press Enter to start new game (Hand #1, #2, etc.)
2. Enter coach's hole cards (e.g., "AH", "KD")
3. Blinds auto-post (watch bankrolls update)
4. Each player prompted for action in order
5. Enter actions: `fold`, `check`, `call`, `raise`
   - If raise: enter total raise amount (e.g., 20)
6. Enter community cards when prompted:
   - Flop: 3 cards
   - Turn: 1 card
   - River: 1 card
7. At showdown: enter opponent's hand and select winner
8. Game loops to next hand (hand_id increments)

**Input Validation:**
- Cards validated (rejects "G104", "100D", etc.)
- Actions validated (can't check when must call)
- Raise amounts validated (must be > call_value, ≤ bankroll)

### Test Scenarios

**Scenario 1: Everyone Calls**
```
Hand #1: Coach=SB, Player1=BB
- Coach: posts $5 (bankroll: $170)
- Player1: posts $10 (bankroll: $165)
- Coach: call $5 (pays $5 more, total $10, bankroll: $165)
- Player1: check (already at $10)
→ Pot: $20, proceed to flop
```

**Scenario 2: Raise and Responses**
```
Hand #1: Coach=SB, Player1=BB
- Coach: posts $5
- Player1: posts $10
- Coach: raise 20 (pays $15 more, bankroll: $160)
- Player1: call 10 (pays $10 more to match $20, bankroll: $155)
→ Pot: $40, proceed to flop
```

**Scenario 3: Early Win**
```
Hand #1: Coach=SB, Player1=BB
- Coach: posts $5
- Player1: posts $10
- Coach: fold
→ Player1 wins $15 immediately, return to game start
```

**Scenario 4: ML JSON Output**
```
Pre-flop betting:
- Coach: posts $5
- Player1: posts $10
- Coach's turn (ML JSON displayed):
  {
    "hand_id": 1,
    "action": "",  // Empty, coach is first after blinds
    "to_call_bb": 5,
    ...
  }
- Coach: call
- Player1's turn
- Player1: raise 25
- Coach's turn (ML JSON displayed):
  {
    "hand_id": 1,
    "action": "raise",  // Player1 just raised
    "to_call_bb": 20,  // Need $20 more to match $25
    ...
  }
```

## Integration Guide

### 1. Integrate ML Model

In `orchestrator.py`, replace placeholder at line ~50:

```python
# Current (placeholder):
from Orchestrator.input_interface import InputInterface
interface = InputInterface()
return interface.get_action(...)

# Replace with:
from ML_Model.predictor import get_poker_action
data = json.loads(json_payload)
action, value = get_poker_action(data)
print(f"✅ ML Decision: {action} {value if value > 0 else ''}")
return action, value
```

### 2. Integrate CV for Card Detection

In `orchestrator.py`, replace placeholders in card detection methods:

```python
# Current (placeholder):
from Orchestrator.input_interface import InputInterface
interface = InputInterface()
hole_cards = interface.get_cards(2, "hole")

# Replace with:
from Image_Recognition.card_detector import detect_cards
set_crop_mode(NoCrop=True)
hole_cards = detect_cards(count=2)
print(f"✅ CV Detected: {hole_cards}")
```

### 3. Integrate CV for Action Detection

In `orchestrator.py`, replace placeholder at line ~90:

```python
# Current (placeholder):
from Orchestrator.input_interface import InputInterface
interface = InputInterface()
return interface.get_action(...)

# Replace with:
from Image_Recognition.action_detector import detect_action
action, value = detect_action(crop_mode=crop_region, timeout=30)
print(f"✅ CV Detected: {action} {value if value > 0 else ''}")
return action, value
```

### 4. Server Integration (Optional)

If using camera crop server:

```bash
cd Communication/Server
npm install
node server.js
```

Set environment variables:
```bash
export SERVER_URL=http://localhost:3000
export API_KEY=your_api_key_here
```

The `event_signals.py` module will handle crop mode switching automatically.

## Module Dependencies

```bash
pip install opencv-python numpy requests mediapipe
```

## File Structure

```
Computer-Vision-Powered-AI-Poker-Coach/
├── Orchestrator/
│   ├── config.py                 # Enums and constants
│   ├── player_manager.py         # Player state management
│   ├── card_manager.py           # Card tracking
│   ├── betting_cycle.py          # Betting logic
│   ├── ml_json_input.py          # ML JSON generation
│   ├── card_converter.py         # Card format conversion
│   ├── input_interface.py        # Standardized input interface
│   ├── orchestrator.py           # Production mode (CV/ML integration)
│   ├── event_signals.py          # External communication
│   └── game_controller.py        # Legacy FSM (deprecated)
├── test_game.py                  # Test mode (manual inputs)
├── Image_Recognition/            # CV modules (to be integrated)
│   ├── action_detector.py
│   ├── card_analyzer.py
│   └── chip_analyzer.py
├── ML_Model/                     # ML module (to be integrated)
│   └── predictor.py
└── Communication/Server/         # Camera crop server (optional)
    └── server.js
```

## Key Features

✅ **Heads-up play** (Coach vs Player1 only)  
✅ **Automatic blind posting** (pre-flop, $5/$10)  
✅ **Blind rotation** (alternates between Coach and Player1)  
✅ **Proper raise handling** (tracks amounts paid per round)  
✅ **Call value tracking** per betting round  
✅ **Action validation** (can't check when must call, etc.)  
✅ **Early win detection** (1 player remaining)  
✅ **Showdown logic** (2+ players)  
✅ **Bankroll tracking** (updated after each action)  
✅ **Hand counter** (hand_id increments each game)  
✅ **ML JSON generation** (automatic when coach must act)  
✅ **Card format conversion** (VALUE|SUIT ↔ SUIT|VALUE)  
✅ **Input validation** (rejects invalid cards/actions)  
✅ **Modular architecture** (easy CV/ML integration)  
✅ **Test mode** (manual inputs for development)  
✅ **Production mode** (CV/ML integration with fallback)  

## Future Enhancements

- [ ] 4-player support (enable Players 2 & 3)
- [ ] Pot-limit / No-limit betting rules
- [ ] All-in detection and side pots
- [ ] Multi-table support
- [ ] Hand history logging (JSON/CSV export)
- [ ] Replay functionality
- [ ] Real-time hand strength prediction
- [ ] Pre-flop range analysis
- [ ] Post-game statistics dashboard

## Troubleshooting

### Common Issues

**Issue**: Cards overwriting on turn/river  
**Solution**: Fixed in `card_manager.py` - now uses `extend()` instead of assignment

**Issue**: ML JSON shows wrong call amount  
**Solution**: Fixed in `betting_cycle.py` - now passes `amount_needed_to_call` not `call_value`

**Issue**: Hand_id skipping numbers  
**Solution**: Fixed - `increment_hand()` only called once in `wait_for_game_start()`

**Issue**: Action field not blank when coach first to act  
**Solution**: Fixed in `ml_json_input.py` - tracks `first_to_act` flag per round

**Issue**: Players 2 & 3 affecting game  
**Solution**: Fixed in `betting_cycle.py` - auto-folded at start, blinds skip them

## Support

For questions or issues:
1. Check `test_game.py` for expected behavior
2. Review `orchestrator.py` for integration points
3. Examine `ml_json_input.py` for JSON format
4. Inspect `betting_cycle.py` for game logic
