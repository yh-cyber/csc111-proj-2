"""CSC111 Project 2

Module Description
==================
This module contains the handling data functions used to clean and manipulate a csv data file.

Copyright and Usage Information
===============================

This file is provided solely for the personal and private use of students
taking CSC111 at the University of Toronto St. George campus. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited. For more information on copyright for CSC111 materials,
please consult our Course Syllabus.

This file is Copyright (c) 2026 by Nabiha Tariq, Yusyra Hossain, Eleanor Neal, Ruoshui Deng
"""

import pandas as pd

from entities import Graph, Observation


def clean_data(data_file: str) -> None:
    """
    Creates a datamap using pandas library to clean and filter data based on conditions

    Preconditions:
        - data_file is a csv. file with the expected columns and data
    """

    df = pd.read_csv(data_file)

    # Adding errors="ignore" prevents the KeyError if the columns are already gone
    df_new = df.drop(
        columns=["id", "uuid", "observed_on_string", "time_observed_at", "time_zone", "user_id", "user_login",
                 "user_name", "created_at", "updated_at", "quality_grade", "url", "sound_url", "tag_list",
                 "description", "num_identification_agreements", "num_identification_disagreements",
                 "captive_cultivated", "oauth_application_id", "private_place_guess", "private_latitude",
                 "private_longitude", "public_positional_accuracy", "geoprivacy", "taxon_geoprivacy",
                 "coordinates_obscured", "positioning_method", "positioning_device", "scientific_name",
                 "iconic_taxon_name", "taxon_id", "species_guess"], errors="ignore")

    # Safely filter by accuracy only if the column actually exists in the file
    if "positional_accuracy" in df_new.columns:
        df_new = df_new[df_new["positional_accuracy"] <= 1000]

    df_new.to_csv("new_file.csv", index=False)


def data_handle(file: str) -> list[Observation]:
    """
    Creates a list of Observations (instances of the class Observation) by grouping data using pandas library and
    error handling

    Preconditions:
        - file is a csv. file with the expected columns and data
    """
    df = pd.read_csv(file)
    grouped = df.groupby("common_name")

    observations = []

    for name, group in grouped:
        obs = _process_species_group(str(name), group)
        observations.append(obs)

    return observations


def _process_species_group(name: str, group: pd.DataFrame) -> Observation:
    """Helper method to process a single species group into an Observation object
    to reduce local variable complexity.
    """
    valid_images = group["image_url"].dropna()

    # Condensing the image check into one line saves a local variable
    img_val = str(valid_images.iloc[0]) if not valid_images.empty else ""

    obs = Observation(name, img_val)

    dates = list(group["observed_on"])
    lats = list(group["latitude"])
    longs = list(group["longitude"])

    for x in range(len(dates)):
        date_str = str(dates[x])
        # Condensing the string split and int conversion saves another variable
        curr_month = int(date_str.split("-")[1])

        if curr_month in [12, 1, 2]:
            season = 1
        elif curr_month in [3, 4, 5]:
            season = 2
        elif curr_month in [6, 7, 8]:
            season = 3
        else:
            season = 4

        if season not in obs.season_to_loc:
            obs.season_to_loc[season] = []

        # Appending and casting to float directly saves two more variables
        obs.season_to_loc[season].append((date_str, float(lats[x]), float(longs[x])))

    return obs


def observations_to_graph(observations: list[Observation]) -> tuple[Graph, Graph, Graph, Graph]:
    """Builds 4 graphs from a list of Observation objects (one for each season), by calling the graph building
    method. See Graph.build_from_observations for more details.

    Preconditions:
        - each Observation in the list has a valid species name and season_to_loc mapping
        - the list of observations is not empty
    """
    summer = Graph()
    spring = Graph()
    fall = Graph()
    winter = Graph()
    summer.build_from_observations(observations, 3)
    spring.build_from_observations(observations, 2)
    fall.build_from_observations(observations, 4)
    winter.build_from_observations(observations, 1)
    return summer, spring, fall, winter


# if __name__ == '__main__':
#     import python_ta
#
#     python_ta.check_all(config={
#         'extra-imports': ['pandas', 'entities'],
#
#         'allowed-io': ['clean_data', 'data_handle'],
#         'max-line-length': 120,
#         'output-format': 'txt'
#     })
