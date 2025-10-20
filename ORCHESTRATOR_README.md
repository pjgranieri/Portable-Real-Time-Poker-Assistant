# Poker Game Orchestrator Documentation

## Overview
This system implements a complete Texas Hold'em poker game orchestrator with computer vision integration and ML-based hand evaluation.

## Architecture

### Components

#### 1. **Orchestrator** (`Orchestrator/`)
Core game logic and state management.

- **`config.py`**: Enums and constants
  - `Player`: PlayerCoach, PlayerOne, PlayerTwo, PlayerThree
  - `GameState`: Game flow states (WAIT_FOR_GAME_START, PRE_FLOP_BETTING, etc.)
  - `CropRegion`: Camera crop regions for each player

- **`game_controller.py`**: Main FSM (Finite State Machine)
  - Orchestrates entire game flow
  - Manages state transitions
  - Integrates CV and ML modules

- **`betting_cycle.py`**: Betting round logic
  - Handles all betting rounds (pre-flop, flop, turn, river)
  - Automatic blind posting for pre-flop
  - Validates actions (check/call/raise/fold)
  - Tracks who has acted and manages raise cycles

- **`player_manager.py`**: Player state management
  - Bankroll tracking
  - Action detection via CV
  - Hand evaluation via ML
  - Crop region assignment

- **`card_manager.py`**: Card tracking
  - Hole cards storage
  - Community cards management

- **`event_signals.py`**: External communication
  - Server crop mode control
  - CV module integration
  - Signal waiting/polling

#### 2. **Image Recognition** (`Image_Recognition/`)
Computer vision modules for game detection.

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

#### 3. **Communication** (`Communication/Server/`)
Node.js server for camera control.

- **`server.js`**: HTTP API for crop mode control
  - Endpoints: `/api/set-crop-mode`, `/api/get-crop-mode`
  - Controls camera cropping regions
  - Saves cropped images to `Image_Recognition/Outputs/`

## Game Flow

### State Machine

```
WAIT_FOR_GAME_START
  ↓ (Initialize bankrolls, blinds)
WAIT_FOR_HOLE_CARDS (Full camera view)
  ↓ (Detect 2 cards for coach)
PRE_FLOP_BETTING (Action zone cropping)
  ↓ (Auto-post blinds, betting cycle)
WAIT_FOR_FLOP (Community cards crop)
  ↓ (Detect 3 cards)
POST_FLOP_BETTING (Action zone cropping)
  ↓ (Betting cycle)
WAIT_FOR_TURN_CARD (Community cards crop)
  ↓ (Detect 1 card)
TURN_BETTING (Action zone cropping)
  ↓ (Betting cycle)
WAIT_FOR_RIVER_CARD (Community cards crop)
  ↓ (Detect 1 card)
RIVER_BETTING (Action zone cropping)
  ↓ (Betting cycle)
SHOWDOWN (if 2+ players remain)
  ↓ (Read hands, evaluate winner)
→ WAIT_FOR_GAME_START (loop)
```

### Betting Cycle Logic

**Pre-flop:**
1. Small blind auto-posts $5
2. Big blind auto-posts $10 (call value = $10)
3. Action starts after big blind (PlayerThree or PlayerCoach)
4. Players act in turn: call $10, raise, or fold
5. Blinds get chance to act (call difference, raise, or fold)
6. If raise: everyone must respond (call new amount, re-raise, or fold)
7. Round ends when all active players have acted and matched bets

**Post-flop/Turn/River:**
1. Call value resets to $0
2. Action starts at small blind
3. Players can check (if $0), bet, or fold
4. If bet: others must call, raise, or fold
5. Round ends when all active players have acted and matched bets

**Blind Raising:**
- Small blind raising to $20: Already paid $5, pays $15 more, call value = $20
- Big blind raising to $30: Already paid $10, pays $20 more, call value = $30

### Early Win Condition
If only 1 player remains after any betting round:
- Immediately award pot to remaining player
- Skip remaining rounds
- Return to WAIT_FOR_GAME_START

### Showdown
If 2+ players remain after river betting:
1. Read each player's hand via CV
2. Evaluate hands with ML module
3. Award pot to winner

## Crop Regions

| Player | Region | Server Variable |
|--------|--------|-----------------|
| PlayerOne (Small Blind) | Right | `CropRight` |
| PlayerTwo (Big Blind) | Middle | `CropMiddle` |
| PlayerThree | Left | `CropLeft` |
| PlayerCoach (User) | Full/None | `NoCrop` |
| Community Cards | Cards Area | `CropCards` |

## Testing

### Running Tests
```bash
./run_test.sh
```

### Test Flow
1. Press Enter to start game
2. Enter your hole cards (e.g., "AH", "KD")
3. Blinds are auto-posted
4. Each player prompted for action in order
5. Enter actions: `fold`, `check`, `call`, `raise`
6. Enter community cards when prompted
7. At showdown: enter each player's hand and select winner
8. Game loops to next round

### Test Scenarios

**Scenario 1: Everyone Calls**
- Small Blind: posts $5
- Big Blind: posts $10
- PlayerThree: call $10
- PlayerCoach: call $10
- Small Blind: call $5 (to match $10)
- Big Blind: check (already at $10)
- → Flop

**Scenario 2: Raise and Responses**
- Small Blind: posts $5
- Big Blind: posts $10
- PlayerThree: raise $20
- PlayerCoach: call $20
- Small Blind: fold
- Big Blind: call $10 (to match $20)
- → Flop

**Scenario 3: Early Win**
- Small Blind: posts $5
- Big Blind: posts $10
- PlayerThree: fold
- PlayerCoach: fold
- Small Blind: fold
- Big Blind wins $15

## Integration with Real System

### CV Module Integration
Replace test stubs in `player_manager.py`:
```python
def get_action(self, player_enum, crop_region, call_value):
    from Image_Recognition.action_detector import detect_action
    
    self.set_crop_for_player(player_enum)
    action, value = detect_action(crop_mode=crop_region, timeout=30)
    
    return action, value
```

### ML Module Integration
Replace test stub in `player_manager.py`:
```python
def evaluate_with_ml(self, community_cards, hole_cards, remaining_players):
    from ml_module.evaluator import evaluate_winner
    
    showdown_hands = self.read_showdown_hands(remaining_players)
    all_hands = {**hole_cards, **showdown_hands}
    
    return evaluate_winner(remaining_players, all_hands, community_cards)
```

### Server Integration
Start the Node.js server:
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

## Key Features

✅ **Automatic blind posting** (pre-flop)
✅ **Proper raise handling** (includes blind amounts)
✅ **Call value tracking** per betting round
✅ **Action validation** (can't check when must call, etc.)
✅ **Early win detection** (1 player remaining)
✅ **Showdown logic** (2+ players)
✅ **Bankroll tracking** (updated after each action)
✅ **State persistence** across rounds
✅ **Hierarchical CV detection** (fold → check → chips)
✅ **Crop region management** (auto-switching per player)

## Dependencies

```bash
pip install opencv-python numpy requests mediapipe
```

## Future Enhancements

- [ ] Pot-limit / No-limit betting rules
- [ ] All-in detection and side pots
- [ ] Multi-table support
- [ ] Hand history logging
- [ ] Replay functionality
- [ ] Real-time hand strength prediction
- [ ] Adaptive AI opponents
