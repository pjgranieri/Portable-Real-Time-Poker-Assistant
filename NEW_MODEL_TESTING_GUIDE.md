# New Model Testing Guide

This guide explains the new testing infrastructure for your two new YOLO models.

## Overview

You have two new specialized models:

1. **player_action_model.pt** - Detects chips OR folded cards in player action zones
2. **pot_chips_model.pt** - Detects chips in the pot area

Each model has dedicated testing infrastructure and integration code.

---

## Part 1: Player Action Detection

### Model: `player_action_model.pt`

**Purpose**: Detect player actions (FOLD, CHECK, BET/RAISE) in the player action zone.

### Integration File: `action_analyzer_algorithm.py`

**Updated** to use `player_action_model.pt` instead of the old `chip_processing_model.pt`.

**Location**: [Image_Recognition/action_analyzer_algorithm.py:25](Image_Recognition/action_analyzer_algorithm.py#L25)

**Change**:
```python
# OLD:
self.chip_model = YOLO('chip_processing_model.pt')

# NEW:
self.chip_model = YOLO('player_action_model.pt')
```

### Test File: `test_player_action_detection.py`

**Location**: [tests/unit/image_recognition/test_player_action_detection.py](tests/unit/image_recognition/test_player_action_detection.py)

**Test Datasets**:
```
CVDataset/
├── FoldedCardsDataset/   → Expected: FOLD action
├── HandDataset/          → Expected: CHECK action
└── ChipDataset/          → Expected: BET/RAISE action
```

**How it works**:
1. Loads images from each dataset
2. Runs `ActionAnalyzer.analyze_action()` which uses:
   - `player_action_model.pt` to detect chips/folded cards
   - MediaPipe to detect hands
   - Combined logic to determine final action
3. Compares detected action to expected action

**Run Tests**:
```bash
# Activate venv
source venv/bin/activate

# Run all tests
pytest tests/unit/image_recognition/test_player_action_detection.py -v

# Run with detailed output
pytest tests/unit/image_recognition/test_player_action_detection.py -v --tb=short

# Run in parallel
pytest tests/unit/image_recognition/test_player_action_detection.py -n auto
```

**Expected Output**:
```
OVERALL PLAYER ACTION DETECTION ACCURACY
=========================================================
Total images tested: XXX
Correct detections: XXX
Accuracy: XX.XX%
=========================================================

PLAYER ACTION ACCURACY BY ACTION TYPE
=========================================================
BET/RAISE   : XXX/XXX correct (XX.XX%)
CHECK       : XXX/XXX correct (XX.XX%)
FOLD        : XXX/XXX correct (XX.XX%)
=========================================================
```

---

## Part 2: Pot Chip Counting

### Model: `pot_chips_model.pt`

**Purpose**: Detect and count chips in the pot.

### Integration File: `chip_pot_analyzer.py` (NEW)

**Location**: [Image_Recognition/chip_pot_analyzer.py](Image_Recognition/chip_pot_analyzer.py)

**Purpose**: Combines pot chip detection with height-based counting.

**How it works**:
```python
from Image_Recognition.chip_pot_analyzer import ChipPotAnalyzer

analyzer = ChipPotAnalyzer()
chip_count = analyzer.detect_and_count_chips('image.jpg', color='red')
print(f"Chips in pot: {chip_count}")
```

**Two-Stage Process**:
1. **Stage 1**: `pot_chips_model.pt` detects WHERE chips are (bounding box)
2. **Stage 2**: Height analysis determines HOW MANY chips (aspect ratio)

### Test File: `test_pot_chip_counting.py`

**Location**: [tests/unit/image_recognition/test_pot_chip_counting.py](tests/unit/image_recognition/test_pot_chip_counting.py)

**Test Datasets**:
```
CVDataset/
├── BlackChips/
│   ├── 1Chip/     → Expected: 1 chip
│   ├── 2Chips/    → Expected: 2 chips
│   ├── 3Chips/    → Expected: 3 chips
│   ├── 4Chips/    → Expected: 4 chips
│   └── 5Chips/    → Expected: 5 chips
├── BlueChips/
│   └── (same structure)
└── RedChips/
    └── (same structure)
```

**Run Tests**:
```bash
# Activate venv
source venv/bin/activate

# Run all tests
pytest tests/unit/image_recognition/test_pot_chip_counting.py -v

# Run with detailed output
pytest tests/unit/image_recognition/test_pot_chip_counting.py -v --tb=short

# Run in parallel
pytest tests/unit/image_recognition/test_pot_chip_counting.py -n auto
```

**Expected Output**:
```
OVERALL POT CHIP DETECTION ACCURACY
=========================================================
Total images tested: XXXX
Correct detections: XXX
Accuracy: XX.XX%
=========================================================

POT CHIP ACCURACY BY COLOR
=========================================================
BlackChips  : XXX/XXX correct (XX.XX%)
BlueChips   : XXX/XXX correct (XX.XX%)
RedChips    : XXX/XXX correct (XX.XX%)
=========================================================

POT CHIP ACCURACY BY CHIP COUNT
=========================================================
1 chip  : XXX/XXX correct (XX.XX%)
2 chips : XXX/XXX correct (XX.XX%)
3 chips : XXX/XXX correct (XX.XX%)
4 chips : XXX/XXX correct (XX.XX%)
5 chips : XXX/XXX correct (XX.XX%)
=========================================================
```

---

## Files Created/Modified

### New Files Created

1. **[Image_Recognition/chip_pot_analyzer.py](Image_Recognition/chip_pot_analyzer.py)**
   - Production-ready pot chip detection and counting
   - Replaces the old chip_count_analyzer.py and chip_height_analyzer.py for pot scenarios
   - Uses `pot_chips_model.pt` + height-based counting

2. **[tests/unit/image_recognition/test_player_action_detection.py](tests/unit/image_recognition/test_player_action_detection.py)**
   - Comprehensive tests for player action detection
   - Tests FOLD, CHECK, BET/RAISE actions
   - Accuracy metrics by action type
   - Confusion matrix

3. **[tests/unit/image_recognition/test_pot_chip_counting.py](tests/unit/image_recognition/test_pot_chip_counting.py)**
   - Comprehensive tests for pot chip counting
   - Tests all colors and chip counts
   - Accuracy metrics by color and count
   - Failure analysis

### Modified Files

4. **[Image_Recognition/action_analyzer_algorithm.py](Image_Recognition/action_analyzer_algorithm.py)**
   - Updated line 25: Now uses `player_action_model.pt` instead of `chip_processing_model.pt`
   - No other changes to logic

---

## Quick Start

### Test Player Actions
```bash
source venv/bin/activate
pytest tests/unit/image_recognition/test_player_action_detection.py -v -s
```

### Test Pot Chip Counting
```bash
source venv/bin/activate
pytest tests/unit/image_recognition/test_pot_chip_counting.py -v -s
```

### Test Both Together
```bash
source venv/bin/activate
pytest tests/unit/image_recognition/test_player_action_detection.py tests/unit/image_recognition/test_pot_chip_counting.py -v
```

---

## Model Requirements

Make sure these model files exist:

```
Image_Recognition/Models/
├── player_action_model.pt       ← NEW (for player actions)
├── pot_chips_model.pt            ← NEW (for pot chips)
└── chip_amount_processing_model.pt  ← Existing (for chip amounts)
```

If models are missing, tests will fail with:
```
FileNotFoundError: [Errno 2] No such file or directory: '.../player_action_model.pt'
```

---

## Understanding the Results

### Player Action Detection

**Good Results**:
- Overall accuracy ≥70%
- FOLD accuracy ≥60%
- CHECK accuracy ≥60%
- BET/RAISE accuracy ≥60%

**Common Issues**:
- Hands mistaken for folded cards → Adjust MediaPipe confidence
- Chips not detected → Check YOLO confidence threshold
- Folded cards missed → Check fold_confidence_threshold

### Pot Chip Counting

**Good Results**:
- Overall accuracy ≥70%
- Each color accuracy ≥60%
- Each chip count accuracy ≥60%

**Common Issues**:
- 2-4 chips often confused → Aspect ratios very similar (fundamental limitation)
- Black chips low accuracy → YOLO struggles with black chips
- High chip counts (5+) underestimated → Aspect ratio threshold needs tuning

---

## Cleanup Notes

The following files are now **OBSOLETE** for pot chip detection:

- `chip_count_analyzer.py` (HSV-based, doesn't work)
- `chip_height_analyzer.py` (works but replaced by chip_pot_analyzer.py)

**Recommendation**: Keep `chip_height_analyzer.py` for reference but use `chip_pot_analyzer.py` in production.

---

## Integration with Your System

### For Player Action Detection

```python
from Image_Recognition.action_analyzer_algorithm import ActionAnalyzer

analyzer = ActionAnalyzer()
result = analyzer.analyze_action('path/to/player_image.jpg')

print(f"Action: {result['action']}")
print(f"Details: {result['details']}")

# Possible actions:
# - 'FOLD': Player folded
# - 'CHECK': Player checking (hand tap)
# - 'BET/RAISE': Player betting/raising
# - 'UNCERTAIN_FOLD': Possible fold but low confidence
# - 'UNCERTAIN_CHECK': Possible check but low confidence
# - 'NO_ACTION': No action detected
```

### For Pot Chip Counting

```python
from Image_Recognition.chip_pot_analyzer import ChipPotAnalyzer

analyzer = ChipPotAnalyzer()
chip_count = analyzer.detect_and_count_chips('path/to/pot_image.jpg', color='red')

print(f"Chips in pot: {chip_count}")
# Returns: 0-5 (0 means no chips detected)
```

---

## Performance Benchmarks

Both test suites include performance benchmarks:

```bash
# Test speed
pytest tests/unit/image_recognition/test_player_action_detection.py::test_action_detection_speed_benchmark -v -s
pytest tests/unit/image_recognition/test_pot_chip_counting.py::test_pot_detection_speed_benchmark -v -s
```

**Expected Performance**:
- Player Action Detection: <3.0 seconds per image
- Pot Chip Counting: <2.0 seconds per image

---

## Troubleshooting

### Issue: Tests not running

**Solution**: Install pytest
```bash
source venv/bin/activate
pip install pytest pytest-cov pytest-xdist
```

### Issue: Model not found

**Solution**: Verify model files exist
```bash
ls Image_Recognition/Models/player_action_model.pt
ls Image_Recognition/Models/pot_chips_model.pt
```

### Issue: Low accuracy

**Solutions**:
1. Check if datasets are labeled correctly
2. Adjust confidence thresholds in code
3. Retrain models with more data
4. Review failure analysis in test output

### Issue: MediaPipe not working

**Solution**: Install mediapipe
```bash
source venv/bin/activate
pip install mediapipe
```

---

## Next Steps

1. **Run the tests** to establish baseline accuracy
2. **Review failure analysis** to identify problem areas
3. **Tune thresholds** if needed (confidence, aspect ratios)
4. **Retrain models** if accuracy is too low
5. **Integrate** into your main poker system

Good luck! 🎰
