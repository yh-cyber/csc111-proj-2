""" CSC111 Project 2

Module Description
==================
File that contains:
    - Runner class: Methods for running and creating interactive visualization such as:
        - Creating circular images
        - Hovering over image to display information me
        - Popup handling
        - User input
        - Creating the weighted graphs depending on season

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
import os
import urllib.request
import urllib.error
from io import BytesIO
import tkinter as tk
from tkinter import ttk
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mplcursors import cursor

import data_manipulation
import entities
from utils import make_circular


class Runner:
    """Class with methods for running the visualization

    Instance Attributes:
       - data_file: File where the original data is stored in a csv

    Representation Invariants:
       - data_file is a valid filepath that links to a csv file of the right format
    """
    data_file: str
    _observations: list[entities.Observation]
    _graphs: dict[str, entities.Graph]
    species_input: entities.SpeciesSearchDropdown | None
    season_input: ttk.Combobox | None
    status_label: ttk.Label | None

    def __init__(self, data_file: str) -> None:
        self.data_file = data_file
        self._observations = []
        self._graphs = {}
        self.species_input = None
        self.season_input = None
        self.status_label = None

    @staticmethod
    def create_species_graph(target_species: str, current_graph: entities.Graph) -> nx.Graph:
        """
        Creates a networkx species graph using thresholds stated below to display only a portion of the entire graph
        using networkx.
        """
        species_graph = nx.Graph()

        # Inlined the image lookup to save a local variable
        img = current_graph.get_vertex(target_species).image
        species_graph.add_node(target_species, image=img, name=target_species, level=0)

        queue = [(target_species, 0)]
        visited = {target_species}

        # Inlined max_nodes (50) here to save local variables
        while queue and len(visited) < 50:
            current, level = queue.pop(0)
            current_v = current_graph.get_vertex(current)

            neighbours = current_graph.get_neighbours(current)
            # Inlined top_k (4) to save a local variable
            neighbours = sorted(neighbours, key=current_v.get_prob, reverse=True)[:4]

            for neighbour in neighbours:
                prob = current_v.get_prob(neighbour)

                # Inlined min_prob (0.02) to save a local variable
                if prob < 0.02:
                    continue

                if neighbour not in species_graph:
                    neighbour_v = current_graph.get_vertex(neighbour)
                    species_graph.add_node(neighbour, image=neighbour_v.image, name=neighbour, level=level + 1)

                display_label = f"{prob * 100:.2f}%"
                thickness = math.sqrt(current_v.get_weight(neighbour)) * 1.5

                species_graph.add_edge(current, neighbour, weight=thickness, label=display_label)

                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append((neighbour, level + 1))

        return species_graph

    @staticmethod
    def _fetch_image(img_data: Any) -> np.ndarray:
        """Helper method to safely download or load an image, preventing deep nesting.

        Preconditions:
            - img_data is a string representing a URL/file path, or a pandas Series containing one
        """
        if hasattr(img_data, "iloc"):
            img_data = img_data.iloc[0]

        if not isinstance(img_data, str):
            return np.full((100, 100, 4), [150, 150, 150, 255], dtype=np.uint8)

        if img_data.startswith("http"):
            try:
                req = urllib.request.Request(img_data, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    img_data_raw = Image.open(BytesIO(response.read())).convert('RGBA')
                    return np.array(img_data_raw)
            except (urllib.error.URLError, OSError):
                pass
        else:
            try:
                return plt.imread(img_data)
            except (FileNotFoundError, OSError):
                pass

        # Fallback if both tries fail
        return np.full((100, 100, 4), [150, 150, 150, 255], dtype=np.uint8)

    @staticmethod
    def _draw_edges(species_graph: nx.Graph, pos: dict, ax: Any) -> None:
        """Helper method to draw graph edges.

        Preconditions:
            - all('weight' in species_graph.edges[u, v] for u, v in species_graph.edges())
            - all(u in pos and v in pos for u, v in species_graph.edges())
        """
        for u, v in species_graph.edges():
            single_weight = float(species_graph.edges[u, v]['weight'])
            nx.draw_networkx_edges(species_graph, pos, edgelist=[(u, v)], width=single_weight, ax=ax)

    @staticmethod
    def _get_node_info(species_graph: nx.Graph) -> dict[str, str]:
        """Helper method to generate formatted interaction probabilities for the hover tool.

        Preconditions:
            - all('label' in species_graph.edges[u, v] for u, v in species_graph.edges())
        """
        node_info = {}
        for node in species_graph.nodes():
            neighbours = []
            for nbr in species_graph.neighbors(node):
                label = species_graph.edges[node, nbr].get("label", "")
                neighbours.append(f"{nbr}: {label}")
            node_info[node] = "\n".join(neighbours)
        return node_info

    @staticmethod
    def _draw_images(species_graph: nx.Graph, pos: dict, ax: Any, node_info: dict) -> list[tuple]:
        """Helper method to draw circular node images and collect positions.

        Preconditions:
            - all('image' in species_graph.nodes[node] for node in species_graph.nodes())
            - all(node in pos and node in node_info for node in species_graph.nodes())
        """
        node_points = []
        for n in species_graph.nodes():
            x, y = pos[n]
            node_points.append((x, y, n, node_info[n]))

            raw_img = species_graph.nodes[n]['image']
            img = make_circular(Runner._fetch_image(raw_img))

            ab = AnnotationBbox(OffsetImage(img, zoom=0.1), (x, y), frameon=False)
            ax.add_artist(ab)

        return node_points

    @staticmethod
    def _setup_hover(ax: Any, node_points: list[tuple]) -> None:
        """Helper method to establish the interactive cursor.

        Preconditions:
            - all(len(point) == 4 for point in node_points)
        """
        ax.set_axis_off()
        x_vals = [p[0] for p in node_points]
        y_vals = [p[1] for p in node_points]

        scatter = ax.scatter(x_vals, y_vals, s=100, alpha=0)
        crs = cursor(scatter, hover=True)

        crs.connect(
            "add",
            lambda sel: (
                sel.annotation.set_text(
                    f"{node_points[sel.index][2]}\nLikelihood of Interactions:\n{node_points[sel.index][3]}"
                ),
                sel.annotation.get_bbox_patch().set_alpha(1),
            )
        )

    def display_window(self) -> None:
        """
        Display the window that lets you create a graph using tkinter and adding the use input spaces and buttons.
        """
        root = tk.Tk()  # Fixed the Tk() call!
        root.title('Species Interaction Visualizer')

        # center window in middle of screen
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.geometry(f"+{screen_width // 3}+{screen_height // 3}")

        frm = ttk.Frame(root)
        frm.grid(padx=20, pady=10)

        # Add text boxes and dropdown menus for inputs
        ttk.Label(frm, text="Welcome to the species interaction visualizer!",
                  font=("TkDefaultFont", 14)).grid(column=0, row=0, pady=10, columnspan=2)

        # Using type ignore to suppress PyCharm's outdated internal Tkinter typing
        self.species_input = entities.SpeciesSearchDropdown(c=1, r=1, window=frm)  # type: ignore

        ttk.Label(frm, text="Input season: ").grid(column=0, row=2, padx=5, sticky="E")
        self.season_input = ttk.Combobox(frm, values=['Spring', 'Summer', 'Fall', 'Winter'])
        self.season_input.grid(column=1, row=2, padx=5, pady=5, sticky="W")

        # Buttons
        ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=5, padx=5, pady=5)
        ttk.Button(frm, text="Initialize Graph with current settings", command=self._initialize_graph_vals).grid(
            column=0, row=5, padx=5, pady=5)

        self.status_label = ttk.Label(frm, text="", foreground="red")
        self.status_label.grid(column=0, row=4, columnspan=2)
        root.mainloop()

    def _initialize_graph_vals(self) -> None:
        """
        Save the values in the textboxes, and draw and display the graph from them.
        """
        if self.season_input is None or self.species_input is None or self.status_label is None:
            return

        # Grab values as local variables
        season_input_text = self.season_input.get()
        target_species = self.species_input.species

        # Check if the season is valid
        if season_input_text not in self._graphs:
            self.status_label.config(text='Not a valid season.')
            return

        current_graph = self._graphs[season_input_text]

        if not target_species:
            self.status_label.config(text="Please select a species first.")
            return

        if not current_graph.is_vertex(target_species):
            self.status_label.config(text=f"Species '{target_species}' not found in {season_input_text}.")
            return

        self.status_label.config(text="")

        species_graph = self.create_species_graph(target_species, current_graph)
        pos = nx.spring_layout(species_graph, k=10)
        _, ax = plt.subplots()

        Runner._draw_edges(species_graph, pos, ax)
        node_info = Runner._get_node_info(species_graph)
        node_points = Runner._draw_images(species_graph, pos, ax, node_info)
        Runner._setup_hover(ax, node_points)

        plt.show()

    def run(self) -> None:
        """
        Run and create a graph based on user input
        """
        if "new_file.csv" not in os.listdir():
            data_manipulation.clean_data(self.data_file)
        self._observations = data_manipulation.data_handle("new_file.csv")

        summer, spring, fall, winter = data_manipulation.observations_to_graph(self._observations)
        self._graphs = {
            'Spring': spring,
            'Summer': summer,
            'Fall': fall,
            'Winter': winter
        }

        self.display_window()


# if __name__ == '__main__':
#     import python_ta
#
#     python_ta.check_all(config={
#         'extra-imports': ['math', 'os', 'urllib.request', 'urllib.error', 'io', 'tkinter', 'matplotlib.pyplot',
#                           'networkx', 'numpy', 'PIL', 'matplotlib.offsetbox', 'mplcursors',
#                           'data_manipulation', 'entities', 'utils'],
#         'allowed-io': ['run', '_initialize_graph_vals'],
#         'max-line-length': 120,
#         'max-messages': 10,
#         'typecheck': False,
#         'disable': ['E9999']
#     })
