"""Contains the RoutePruner class, which aims to reduce the number of candidate
routes. It selects the N most promising candidates according to the user
provided preferences."""

import math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from networkx import DiGraph

from routing.containers.routes import RouteConfig, Route
from routing.containers.pruning import BBox


class RoutePruner:
    """Class which reduces the number of candidate routes, selecting the most
    promising candidates according to the supplied user preferences."""

    def __init__(self, graph: DiGraph, config: RouteConfig):
        """Class which reduces the number of candidate routes, selecting the
        most promising candidates according to the supplied user preferences.

        Args:
            graph (DiGraph): _description_
            config (RouteConfig): _description_
        """
        self.graph = graph
        self.config = config

        # Create container objects
        self.candidates: List[Route] = []

    def fetch_node_coords(self, node_id: int) -> Tuple[int, int]:
        """Convenience function, retrieves the latitude and longitude for a
        single node in a graph."""
        node = self.graph.nodes[node_id]
        lat = node["lat"]
        lon = node["lon"]
        return lat, lon

    # Route Pruning ###########################################################
    def _get_bounding_box(self) -> BBox:
        """_summary_

        Returns:
            BBox: The physical boundaries of the box which covers all nodes
              visited by candidate routes
        """
        all_nodes = set()
        for route in self.candidates:
            terminal_node = route.route[-1]
            all_nodes.add(terminal_node)

        all_lats = set()
        all_lons = set()
        for node in all_nodes:
            lat, lon = self.fetch_node_coords(node)
            all_lats.add(lat)
            all_lons.add(lon)

        bbox = BBox(
            min_lat=min(all_lats),
            min_lon=min(all_lons),
            max_lat=max(all_lats),
            max_lon=max(all_lons),
        )

        return bbox  # type: ignore

    def _generate_grid(self, bbox: BBox) -> Dict[Tuple[int, int], BBox]:
        """Subdivide a bounding box into a grid of smaller bounding boxes

        Args:
            bbox (BBox): The bounding box for all candidate routes

        Returns:
            Dict[Tuple[int, int], BBox]: A dictionary mapping grid identifiers
              (row & column numbers) to bounding boxes
        """

        lat_bins = np.linspace(
            bbox.min_lat, bbox.max_lat, int(self.config.max_distance * 2)
        )

        lon_bins = np.linspace(
            bbox.min_lon, bbox.max_lon, int(self.config.max_distance * 2)
        )

        last_lat = None
        last_lon = None
        grid = {}
        for lat_inx, lat in enumerate(lat_bins):
            if lat_inx == 0:
                last_lat = lat
                continue

            for lon_inx, lon in enumerate(lon_bins):
                if lon_inx == 0:
                    last_lon = lon
                    continue

                bbox = BBox(
                    min_lat=last_lat,  # type: ignore
                    min_lon=last_lon,  # type: ignore
                    max_lat=lat,
                    max_lon=lon,
                )

                grid[(lat_inx, lon_inx)] = bbox

                last_lon = lon

            last_lat = lat

        return grid

    def _calculate_terminal_grid_square(
        self,
        route: Route,
        tagged_grid: Dict[Tuple[int, int], BBox],
    ) -> Tuple[int, int]:
        """Calculate which grid square a route ends in

        Args:
            route (Route): A candidate route
            tagged_grid (Dict[Tuple[int, int], BBox]): A mapping of grid
              IDs to bounding boxes

        Raises:
            KeyError: If the terminal grid square cannot be located, an
              error will be raised

        Returns:
            Tuple[int, int]: The grid ID for the terminal node of the provided
              route
        """
        # Find last location of the route
        terminal_node = route.route[-1]
        terminal_lat, terminal_lon = self.fetch_node_coords(terminal_node)

        excluded_lats = set()
        for (lat_inx, lon_inx), bbox in tagged_grid.items():
            # Find the correct column
            if lat_inx in excluded_lats:
                continue
            min_lat = bbox.min_lat
            max_lat = bbox.max_lat
            if min_lat <= terminal_lat <= max_lat:
                # Find the correct row
                min_lon = bbox.min_lon
                max_lon = bbox.max_lon
                if min_lon <= terminal_lon <= max_lon:
                    return (lat_inx, lon_inx)
            else:
                excluded_lats.add(lat_inx)
                continue

        raise KeyError(
            "Unable to locate a terminal grid square for node: "
            f"{terminal_node}"
        )

    def _tag_routes_with_terminal_grid_square(
        self,
        tagged_grid: Dict[Tuple[int, int], BBox],
    ):
        """Tag every candidate route with the grid ID in which they end

        Args:
            routes (List[Route]): A list of candidate routes
            tagged_grid (Dict[Tuple[int, int], BBox]): A grid of bounding
              boxes
        """
        # NOTE: Apply _calculate_terminal_grid_square to each node in the graph
        # NOTE: Only apply this to nodes which have been visited during the
        #       last iteration (collect this as an attribute of self)

        for route in self.candidates:
            terminal_square = self._calculate_terminal_grid_square(
                route, tagged_grid
            )

            route.terminal_square = terminal_square

    def _calculate_route_ratio(self, route: Route) -> float:
        """Calculate the ratio/gradient of a route, hillier routes will have
        a higher ratio. This calculation also accounts for the elevation
        gain which will be accrued as a result of returning from the current
        end point back to the start point (in a straight line).

        Args:
            route (Route): A candidate route

        Returns:
            float: The ratio of the route
        """
        gain = route.elevation_gain
        pot_gain = route.elevation_gain_potential

        ratio = (gain + pot_gain) / self.config.max_distance

        return ratio

    def prune_routes(self, routes: List[Route]) -> List[Route]:
        """Reduce the total number of candidate routes, returning only the
        routes which are most likely to satisfy the requirement for the
        hilliest/flattest route.

        Args:
            routes (List[Route]): A list of candidate routes

        Returns:
            List[Route]: A smaller list of candidate routes
        """

        self.candidates = routes

        # Skip ahead if the current list is sufficiently small
        if len(routes) < self.config.max_candidates:
            return routes

        # Subdivide graph into smaller squares, calculate which square each
        # route ends in
        cur_bbox = self._get_bounding_box()
        tagged_grid = self._generate_grid(cur_bbox)
        self._tag_routes_with_terminal_grid_square(tagged_grid)

        # Group routes by terminal grid square, select top N. Sort by current
        # elevation loss/gain, and factor in delta back to start (express both
        # as a ratio)
        grouped_routes = defaultdict(list)
        for route in routes:
            terminal_square = route.terminal_square
            route_ratio = self._calculate_route_ratio(route)
            route.ratio = route_ratio
            grouped_routes[terminal_square].append(route)

        # Evenly distribute max candidates between squares
        routes_per_square = math.ceil(
            self.config.max_candidates / len(grouped_routes.keys())
        )

        # Take the N most promising routes for each grid square
        pruned_routes = []
        for terminal_square, routes in grouped_routes.items():
            routes = sorted(
                routes,
                key=lambda x: x.ratio,
                reverse=self.config.route_mode == "hilly",
            )
            pruned_routes += routes[:routes_per_square]

        # Final sort spans output from each grid square
        pruned_routes = sorted(
            pruned_routes,
            key=lambda x: x.ratio,
            reverse=self.config.route_mode == "hilly",
        )

        return pruned_routes
