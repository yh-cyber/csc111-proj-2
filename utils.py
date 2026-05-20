""" CSC111 Project 2

Module Description
===============================
This file contains all the methods used to calculate probability of two species
interacting and to calculate distance between two locations (longitude and latitude)

Copyright and Usage Information
===============================

This file is provided solely for the personal and private use of students
taking CSC111 at the University of Toronto St. George campus. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited. For more information on copyright for CSC111 materials,
please consult our Course Syllabus.

This file is Copyright (c) 2026 by Nabiha Tariq, Yusyra Hossain, Eleanor Neal, Ruoshui Deng
"""
import math
import numpy as np
import pandas as pd


def calc_jaccard_index(co_occurrences: int, obs_a: int, obs_b: int) -> float:
    """
    Calculates the likelihood of interaction between two species using the Jaccard Index. Returns a float between 0.0
    (never interact) and 1.0 (always interact).

    We say that two species are more likely to interact when they are closer to each other.

    Instance Attributes:
        co_occurrences: The number of times the two species were observed
                        within your proximity/time thresholds.
        obs_a: Total number of observations for Species A in that season.
        obs_b: Total number of observations for Species B in that season.

    >>> calc_jaccard_index(5, 10, 10)
    0.33
    >>> calc_jaccard_index(0, 0, 0)
    0.0
    >>> calc_jaccard_index(10, 10, 10)
    1.0
    """
    if obs_a == 0 and obs_b == 0:
        return 0.0
    else:
        total_unique_observations = (obs_a + obs_b) - co_occurrences
        likelihood = co_occurrences / total_unique_observations
        return round(likelihood, 2)


def get_interaction_parameters(locs_a: list[tuple[str, float, float]],
                               locs_b: list[tuple[str, float, float]],
                               max_distance_km: float) -> tuple[int, int, int]:
    """Finds the co-occurrences, obs_A, and obs_B for two species in a specific season.

    >>> l_a = [('2026-05-10', 0.0, 0.0), ('2026-05-11', 1.0, 1.0)]
    >>> l_b = [('2026-06-12', 0.0, 0.5)]
    >>> get_interaction_parameters(l_a, l_b, 100.0)
    (1, 2, 1)
    >>> get_interaction_parameters([], [('2026-01-01', 0.0, 0.0)], 50.0)
    (0, 0, 1)
    """
    obs_a_count = len(locs_a)
    obs_b_count = len(locs_b)

    if obs_a_count == 0 or obs_b_count == 0:
        return 0, obs_a_count, obs_b_count

    # Count how many sightings of A have at least one B nearby in the same year
    a_near_b = 0
    for loc_a in locs_a:
        if _has_nearby_sighting(loc_a, locs_b, max_distance_km):
            a_near_b += 1

    # Count how many sightings of B have at least one A nearby in the same year
    b_near_a = 0
    for loc_b in locs_b:
        if _has_nearby_sighting(loc_b, locs_a, max_distance_km):
            b_near_a += 1

    co_occurrences = min(a_near_b, b_near_a)

    return co_occurrences, obs_a_count, obs_b_count


def _has_nearby_sighting(target_loc: tuple[str, float, float],
                         comparison_locs: list[tuple[str, float, float]],
                         max_distance_km: float) -> bool:
    """Return whether target_loc is within max_distance_km of any location
    in comparison_locs during the same year.

    >>> target = ('2026-05-10', 0.0, 0.0)
    >>> comps = [('2026-06-12', 0.0, 0.5)]
    >>> _has_nearby_sighting(target, comps, 100.0)
    True
    >>> _has_nearby_sighting(target, comps, 10.0)
    False
    >>> _has_nearby_sighting(('2025-05-10', 0.0, 0.0), comps, 100.0)
    False
    """
    target_date, target_lat, target_lon = target_loc

    for comp_date, comp_lat, comp_lon in comparison_locs:
        if target_date[:4] == comp_date[:4]:
            dist = haversine_distance_km((target_lat, target_lon), (comp_lat, comp_lon))
            if dist <= max_distance_km:
                return True

    return False


def haversine_distance_km(loc1: tuple[float, float], loc2: tuple[float, float]) -> float:
    """Calculates distance in kilometers between two (lat, lon) points using the Haversine formula.

    Preconditions:
        - loc1 and loc2 are floats representing (latitude, longitude)

    >>> haversine_distance_km((43.66, -79.39), (43.66, -79.39))
    0.0
    >>> round(haversine_distance_km((0.0, 0.0), (0.0, 1.0)), 2)
    111.19
    """
    # create latitude and longtidue coordinates from the tuples
    lat1, lon1 = loc1
    lat2, lon2 = loc2

    # account for earth's curvature
    radius_km = 6371.0

    # convert into radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    # difference in latitudes
    d_phi = math.radians(lat2 - lat1)

    # difference in longitudes
    d_lambda = math.radians(lon2 - lon1)

    # apply haversine formula
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return radius_km * c


def make_circular(img: np.ndarray) -> np.ndarray:
    """Return a circular image

    Preconditions:
        - img is a np.ndarray which is an array that holds all the pixels
    """

    # img.shape returns (height, width, channels) but we only need height and width so we split
    h, w = img.shape[:2]

    # find center which is half the width and height and is a corrdinate
    center = (int(w / 2), int(h / 2))

    # find the radius which is the minimum of center tuple
    radius = min(center[0], center[1])

    # create a coordinate grid using numpy (lowercase to satisfy PythonTA)
    y, x = np.ogrid[:h, :w]

    # find distance from center using distance formula
    dist_from_center = (x - center[0]) ** 2 + (y - center[1]) ** 2

    # true/false variable to ensure that distance is less than diameter
    mask = dist_from_center <= radius ** 2

    # ensure that img.shape[2] has rgb colours
    if img.shape[2] == 3:
        # creates a grid of 255's. 255 means it's visible and 0 means it's invisible (the pixel)
        alpha = np.ones((h, w), dtype=np.uint8) * 255

        # stack the alpha onto the image to become rgba
        img = np.dstack((img, alpha))

    # apply mask (outside circle = transparent) and the 3 is the alpha stack we added previously

    img[~mask, 3] = 0

    return img


def get_all_species(data_file: str) -> list[str]:
    """Returns a list of every species in the dataset, used for the species
    dropdown menu in entities.py

    Preconditions:
        - data_file is a valid file path to a csv file in the format written by
          the clean_data function
    """

    df = pd.read_csv(data_file)
    return list(df['common_name'].unique())


if __name__ == '__main__':
    import doctest
    doctest.testmod()
#     import python_ta
#
#     python_ta.check_all(config={
#         'extra-imports': ['numpy', 'pandas', 'math'],
#         'allowed-io': ['get_all_species'],
#         'max-line-length': 120,
#         'max-messages': 10,
#         'typecheck': False,
#         'disable': ['E9999']
#     })
