"""
Shared pytest fixtures and configuration for all tests.

This file is automatically loaded by pytest and provides fixtures
that can be used across all test files.
"""

import pytest
import cv2
import numpy as np
from pathlib import Path


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def dataset_root(project_root):
    """Return the CVDataset directory."""
    return project_root / "CVDataset"


@pytest.fixture
def sample_image():
    """Create a simple test image for basic testing."""
    # Create a 640x480 blank image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_red_chip_image():
    """Create a test image with a red circle (simulating a red chip)."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a red circle
    cv2.circle(image, (320, 240), 50, (0, 0, 255), -1)
    return image


@pytest.fixture
def sample_blue_chip_image():
    """Create a test image with a blue circle (simulating a blue chip)."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a blue circle
    cv2.circle(image, (320, 240), 50, (255, 0, 0), -1)
    return image


@pytest.fixture
def sample_black_chip_image():
    """Create a test image with a black circle (simulating a black chip)."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Fill with white background
    image[:] = (255, 255, 255)
    # Draw a black circle
    cv2.circle(image, (320, 240), 50, (0, 0, 0), -1)
    return image


@pytest.fixture
def sample_multiple_chips_image():
    """Create a test image with multiple red chips."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw 3 red circles
    positions = [(200, 240), (320, 240), (440, 240)]
    for pos in positions:
        cv2.circle(image, pos, 40, (0, 0, 255), -1)
    return image


@pytest.fixture(scope="session")
def hsv_ranges():
    """
    Provide HSV color ranges for chip detection.

    Returns:
        dict: Mapping of color names to HSV ranges
    """
    return {
        "black": ((0, 0, 0), (180, 255, 50)),
        "red": ((0, 100, 100), (10, 255, 255)),
        "blue": ((100, 100, 100), (130, 255, 255)),
        "green": ((40, 100, 100), (80, 255, 255)),
        "white": ((0, 0, 200), (180, 30, 255)),
    }


def pytest_configure(config):
    """
    Configure pytest with custom settings.

    This function is called before test collection begins.
    """
    # Add custom markers
    config.addinivalue_line(
        "markers", "chip_detection: mark test as chip detection test"
    )
    config.addinivalue_line(
        "markers", "card_detection: mark test as card detection test"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test items during collection.

    Add markers to tests based on their location.
    """
    for item in items:
        # Add markers based on test file location
        if "chip_detection" in item.nodeid:
            item.add_marker(pytest.mark.chip_detection)
            item.add_marker(pytest.mark.cv)
        elif "card_detection" in item.nodeid:
            item.add_marker(pytest.mark.card_detection)
            item.add_marker(pytest.mark.cv)

        # Add unit or integration marker based on path
        if "/unit/" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in item.nodeid:
            item.add_marker(pytest.mark.integration)


def pytest_report_header(config):
    """Add custom header to pytest report."""
    return [
        "Computer Vision Powered AI Poker Coach - Test Suite",
        f"Project: {Path.cwd().name}",
    ]
