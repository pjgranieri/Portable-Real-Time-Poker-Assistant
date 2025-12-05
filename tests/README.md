# Testing Guide - AI Poker Coach

This directory contains the comprehensive test suite for the Computer Vision Powered AI Poker Coach project.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install pytest, pytest-cov, and pytest-xdist for running tests.

### 2. Run All Tests

```bash
# Run all tests with verbose output
pytest

# Run all tests in parallel (faster)
pytest -n auto

# Run with coverage report
pytest --cov=. --cov-report=html
```

### 3. Run Specific Tests

```bash
# Run only chip detection tests
pytest tests/unit/image_recognition/test_chip_detection.py -v

# Run a specific test function
pytest tests/unit/image_recognition/test_chip_detection.py::test_chip_count_accuracy -v

# Run tests for a specific color and count
pytest tests/unit/image_recognition/test_chip_detection.py -k "RedChips and 3"
```

## Test Organization

```
tests/
├── README.md                          # This file
├── conftest.py                        # Shared fixtures and configuration
├── __init__.py
├── unit/                              # Unit tests (fast, isolated)
│   ├── __init__.py
│   └── image_recognition/
│       ├── __init__.py
│       └── test_chip_detection.py     # Chip detection accuracy tests
├── integration/                       # Integration tests (slower)
│   └── (future tests)
└── fixtures/                          # Test data and utilities
    └── (future fixtures)
```

## Chip Detection Tests

### Overview

The chip detection tests (`test_chip_detection.py`) validate the **two-stage chip detection system**:

**Stage 1: YOLO Detection** - Finds WHERE chips are located (bounding boxes)
**Stage 2: HSV Counting** - Counts HOW MANY chips are in each detected region

This approach combines the strengths of both methods:
- YOLO excels at locating chip regions
- HSV analysis excels at counting stacked chips

### Test Coverage

- **3 colors**: Black, Blue, Red
- **5 quantities**: 1, 2, 3, 4, 5 chips
- **~591 total images** from the CVDataset

### Test Structure

**Parameterized Tests**: Uses `pytest.mark.parametrize` to automatically test all images
- Each image is tested individually with the two-stage approach
- Clear failure messages show which specific images fail
- Shows which stage failed (YOLO detection or HSV counting)

**Accuracy Tests**: Multiple test methods analyze detection accuracy:
- Overall accuracy across all images
- Accuracy by color (Black/Blue/Red)
- Accuracy by chip count (1-5)
- Detailed failure analysis

### Running Chip Detection Tests

```bash
# Run all chip detection tests
pytest tests/unit/image_recognition/test_chip_detection.py -v

# Run with short traceback for easier reading
pytest tests/unit/image_recognition/test_chip_detection.py -v --tb=short

# Run in parallel for speed (uses all CPU cores)
pytest tests/unit/image_recognition/test_chip_detection.py -n auto

# Run and see detailed accuracy report
pytest tests/unit/image_recognition/test_chip_detection.py -v -s

# Run only the overall accuracy test
pytest tests/unit/image_recognition/test_chip_detection.py::TestChipDetectionAccuracy::test_overall_accuracy -v

# Skip slow performance tests
pytest tests/unit/image_recognition/test_chip_detection.py -v -m "not slow"
```

### Understanding Test Output

When you run the chip detection tests, you'll see output like:

```
OVERALL CHIP DETECTION ACCURACY
============================================================
Total images tested: 591
Correct detections: 523
Incorrect detections: 68
Accuracy: 88.49%
============================================================

ACCURACY BY COLOR
============================================================
BlackChips  : 165/171 correct (96.49%)
BlueChips   : 201/218 correct (92.20%)
RedChips    : 157/202 correct (77.72%)
============================================================

ACCURACY BY CHIP COUNT
============================================================
1 chip : 110/114 correct (96.49%)
2 chips: 105/110 correct (95.45%)
3 chips: 112/118 correct (94.92%)
4 chips: 115/123 correct (93.50%)
5 chips: 81/126 correct (64.29%)
============================================================
```

This shows:
- Overall accuracy percentage
- Which colors are detected most accurately
- Which chip counts are hardest to detect
- Common failure patterns

### Adjusting Accuracy Thresholds

The tests include accuracy thresholds that cause test failures if accuracy drops below acceptable levels:

```python
# In test_chip_detection.py

# Overall accuracy threshold (line ~152)
assert accuracy >= 70.0, f"Overall accuracy {accuracy:.2f}% is below 70% threshold"

# Per-color accuracy threshold (line ~176)
assert accuracy >= 60.0, f"{color} accuracy {accuracy:.2f}% is below 60% threshold"

# Per-chip-count threshold (line ~199)
assert accuracy >= 60.0, f"{chip_count} chips accuracy {accuracy:.2f}% is below 60% threshold"
```

You can adjust these thresholds based on your requirements.

## Common Commands

### Running Tests

```bash
# All tests
pytest

# All tests, parallel
pytest -n auto

# Specific file
pytest tests/unit/image_recognition/test_chip_detection.py

# Specific test class
pytest tests/unit/image_recognition/test_chip_detection.py::TestChipDetectionAccuracy

# Specific test function
pytest tests/unit/image_recognition/test_chip_detection.py::test_chip_count_accuracy

# Tests matching a keyword
pytest -k "chip"
pytest -k "BlackChips"
pytest -k "accuracy"
```

### Filtering Tests

```bash
# Run only unit tests
pytest -m unit

# Run only CV tests
pytest -m cv

# Run only chip detection tests
pytest -m chip_detection

# Exclude slow tests
pytest -m "not slow"

# Run unit tests but not slow ones
pytest -m "unit and not slow"
```

### Output Options

```bash
# Verbose output
pytest -v

# Extra verbose (show test names and results)
pytest -vv

# Show print statements (don't capture stdout)
pytest -s

# Show short traceback
pytest --tb=short

# Show no traceback
pytest --tb=no

# Show summary of all test outcomes
pytest -ra
```

### Coverage Reports

```bash
# Generate coverage report
pytest --cov=. --cov-report=term

# Generate HTML coverage report (opens in browser)
pytest --cov=. --cov-report=html
open htmlcov/index.html

# Generate coverage for specific module
pytest --cov=Image_Recognition tests/unit/image_recognition/

# Show missing line numbers
pytest --cov=. --cov-report=term-missing
```

### Performance

```bash
# Show 10 slowest tests
pytest --durations=10

# Run tests in parallel (faster)
pytest -n auto

# Run tests in parallel with 4 workers
pytest -n 4
```

## Writing New Tests

### Example: Simple Unit Test

```python
import pytest
from your_module import your_function

def test_your_function():
    """Test that your_function works correctly."""
    result = your_function(input_value)
    assert result == expected_value
```

### Example: Parameterized Test

```python
import pytest

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_multiply_by_two(input, expected):
    """Test multiplication by 2."""
    assert input * 2 == expected
```

### Example: Using Fixtures

```python
import pytest

@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    return {"key": "value"}

def test_with_fixture(sample_data):
    """Test using the fixture."""
    assert sample_data["key"] == "value"
```

## Continuous Integration

The test suite is designed to be run in CI/CD pipelines. Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.12
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Troubleshooting

### Import Errors

If you get import errors when running tests:

```bash
# Make sure you're in the project root
cd /path/to/Computer-Vision-Powered-AI-Poker-Coach

# Run pytest from the root
pytest
```

### Image Loading Errors

If tests fail with "Failed to load image" errors:

```bash
# Verify the CVDataset exists
ls -la CVDataset/

# Check that images exist in the expected locations
ls -la CVDataset/RedChips/1Chip/
```

### Test Discovery Issues

If pytest doesn't find your tests:

```bash
# List all collected tests without running
pytest --collect-only

# Check pytest configuration
pytest --version
pytest --markers
```

## Best Practices

1. **Write descriptive test names**: `test_chip_detector_handles_5_red_chips` is better than `test_1`

2. **Use fixtures for setup**: Avoid duplicating setup code across tests

3. **Test one thing per test**: Each test should verify a single behavior

4. **Use parametrize for similar tests**: Don't copy-paste tests with different inputs

5. **Add helpful assertion messages**: Make failures easy to debug

6. **Keep tests fast**: Unit tests should run in milliseconds, not seconds

7. **Use markers**: Organize tests with markers like `@pytest.mark.slow`

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [pytest-xdist documentation](https://pytest-xdist.readthedocs.io/)
