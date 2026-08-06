"""
Test suite for utils.py

This module contains tests to verify the mathematical logic, spatial distance
calculations, and image masking functionalities used in the ecosystem graph.
"""

import math
import numpy as np
import pytest
from utils import (
    haversine_distance_km,
    calculate_interaction_likelihood,
    get_interaction_parameters,
    make_circular
)


def test_haversine_distance_zero() -> None:
    """Test that the distance between the exact same coordinates is 0.0."""
    loc = (43.6532, -79.3832)
    assert haversine_distance_km(loc, loc) == 0.0


def test_haversine_distance_known_points() -> None:
    """Test the Haversine formula against a known real-world distance."""
    # New York City to Los Angeles is approximately 3935.7 km
    nyc = (40.7128, -74.0060)
    la = (34.0522, -118.2437)
    dist = haversine_distance_km(nyc, la)
    # Use math.isclose to account for minor floating point rounding differences
    assert math.isclose(dist, 3935.7, rel_tol=0.01)


def test_calculate_interaction_likelihood_zero() -> None:
    """Test that zero observations correctly return a 0.0 likelihood without crashing."""
    assert calculate_interaction_likelihood(0, 0, 0) == 0.0


def test_calculate_interaction_likelihood_perfect_overlap() -> None:
    """Test the Jaccard index when two species are always seen together."""
    # 5 total A, 5 total B, and they co-occurred 5 times.
    # Math: 5 / (5 + 5 - 5) = 1.0
    assert calculate_interaction_likelihood(5, 5, 5) == 1.0


def test_calculate_interaction_likelihood_partial_overlap() -> None:
    """Test the Jaccard index for partial overlaps."""
    # 10 sightings of A, 10 sightings of B, 5 co-occurrences.
    # Math: 5 / (10 + 10 - 5) = 5 / 15 = 0.3333...
    result = calculate_interaction_likelihood(5, 10, 10)
    assert math.isclose(result, 0.3333333, rel_tol=0.0001)


def test_get_interaction_parameters_empty() -> None:
    """Test parameter extraction when lists are empty."""
    co_occur, obs_a, obs_b = get_interaction_parameters([], [], 1.0)
    assert co_occur == 0
    assert obs_a == 0
    assert obs_b == 0


def test_get_interaction_parameters_different_years() -> None:
    """Test that sightings at the same location but DIFFERENT years do not count as an interaction."""
    locs_a = [("2023-05-10", 43.6, -79.3)]
    locs_b = [("2024-05-10", 43.6, -79.3)]

    co_occur, obs_a, obs_b = get_interaction_parameters(locs_a, locs_b, 1.0)
    assert co_occur == 0
    assert obs_a == 1
    assert obs_b == 1


def test_get_interaction_parameters_within_distance() -> None:
    """Test that sightings in the same year and within 1km are counted."""
    # These coordinates are extremely close to each other
    locs_a = [("2023-05-10", 43.6000, -79.3000)]
    locs_b = [("2023-06-12", 43.6001, -79.3000)]

    co_occur, _, _ = get_interaction_parameters(locs_a, locs_b, 1.0)
    assert co_occur == 1


def test_get_interaction_parameters_outside_distance() -> None:
    """Test that sightings in the same year but further than 1km are NOT counted."""
    locs_a = [("2023-05-10", 43.6000, -79.3000)]
    locs_b = [("2023-06-12", 44.6000, -80.3000)]  # Too far away

    co_occur, _, _ = get_interaction_parameters(locs_a, locs_b, 1.0)
    assert co_occur == 0


def test_make_circular_adds_alpha_channel() -> None:
    """Test that the numpy masking function correctly adds a transparency channel."""
    # Create a dummy 10x10 RGB image (all white pixels)
    dummy_img = np.ones((10, 10, 3), dtype=np.uint8) * 255
    circ_img = make_circular(dummy_img)

    # It should now be RGBA (4 channels)
    assert circ_img.shape == (10, 10, 4)

    # Check that a corner pixel (0,0) was made completely transparent (alpha = 0)
    assert circ_img[0, 0, 3] == 0

    # Check that a center pixel (5,5) remained fully visible (alpha = 255)
    assert circ_img[5, 5, 3] == 255


if __name__ == '__main__':
    pytest.main(['test_utils.py'])
