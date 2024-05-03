"""Primary class which handles the creation of circular routes according
to the user provided configuration."""

from typing import List, Tuple

import tqdm

from routing.containers.routes import Route, RouteConfig
from routing.graph_utils.selector.selector import Selector
from routing.graph_utils.condenser.condenser import GraphCondenser
from routing.route_maker.zimmer import Zimmer
from routing.route_pruner.route_pruner import RoutePruner

# TODO: Make this more configurable


class RouteMaker:
    """Main route finder class for the application, given a graph containing
    enriched OSM/Defra data create a circular route based on a starting
    point & a max distance.
    """

    def __init__(
        self,
        config: RouteConfig,
    ):
        """Create a route finder based on user preferences.

        Args:
            config (RouteConfig): The user-generated route configuration
        """

        self.config = config

        # Fetch networkx graph from enriched parquet dataset
        selector = Selector(self.config)
        start_node, full_graph = selector.create_graph()
        self.start_node = start_node

        # Create a condensed representation of the selected graph
        condenser = GraphCondenser(full_graph, start_node)
        graph = condenser.condense_graph()
        self.graph = graph
        del selector, condenser

        # Set up for route generation
        self.zimmer = Zimmer(self.graph, self.config)
        self.candidates = self._create_seed_route(self.start_node)
        self.completed_routes: List[Route] = []
        self.pruner = RoutePruner(
            self.graph,
            config,
        )

        # Debugging
        self.last_candidates: List[Route] = []

    def fetch_node_coords(self, node_id: int) -> Tuple[int, int]:
        """Convenience function, retrieves the latitude and longitude for a
        single node in a graph.

        Args:
            node_id (int): The ID of the node to fetch lat/lon for
        """
        node = self.graph.nodes[node_id]
        lat = node["lat"]
        lon = node["lon"]
        return lat, lon

    # Route Seeding ###########################################################

    def _create_seed_route(self, start_node: int) -> List[Route]:
        """Based on the user provided start point, generate a seed route
        starting at the closest available node.

        Returns:
            Route: A candidate route of zero distance, starting at the node
              closest to the specified start point.
        """

        seed = [
            Route(
                route_id="seed",
                current_position=start_node,
                route=[start_node],
                visited={start_node},
            )
        ]

        return seed

    # Route Finding ###########################################################

    def _remove_start_node_from_visited(self, route: Route) -> Route:
        """In order to be able to complete a circular route, the starting node
        must be removed from the list of visited nodes. This must be done once
        the algorithm is on its second iteration, at which point there will
        be an intermediate node which prevents an immediate return to the
        starting point.

        Args:
            route (Route): A candidate route

        Returns:
            Route: A copy of the input route, with the first visited
              node removed from the 'visited' key
        """

        start_pos = route.route[0]
        route.visited.remove(start_pos)
        return route

    def _update_progress_bar(self, pbar: tqdm.tqdm):
        """Update the progress bar with relevant metrics which enable the
        user to track the calculation as it progresses

        Args:
            pbar (tqdm.tqdm): Handle for the progress bar
        """
        n_candidates = len(self.candidates)
        n_valid = len(self.completed_routes)

        if n_candidates:
            iter_dist = sum(route.distance for route in self.candidates)
            iter_dist += sum(route.distance for route in self.completed_routes)
            avg_distance = iter_dist / (n_candidates + n_valid)
        else:
            avg_distance = self.config.max_distance

        pbar.update(1)
        pbar.set_description(
            (
                f"{n_candidates:,.0f} cands | {n_valid:,.0f} valid | "
                f"{avg_distance:,.2f} avg dist"
            )
        )

    def _generate_route_id(self, cand_inx: int, step_inx: int) -> str:
        """Generate a unique identifier for a route"""
        return f"{cand_inx}_{step_inx}"

    def find_routes(self) -> List[Route]:
        """Main user-facing function for this class. Generates a list of
        circular routes according to the user's preferences.

        Returns:
            List[Route]: A list of completed routes
        """

        # Recursively check for circular routes
        pbar = tqdm.tqdm()
        iters = 0
        while self.candidates:
            # For each potential candidate route
            new_candidates = []
            for cand_inx, candidate in enumerate(self.candidates):
                # Check which nodes can be reached
                possible_steps = self.zimmer.generate_possible_steps(candidate)

                # For each node which can be reached
                for step_inx, possible_step in enumerate(possible_steps):
                    # Step to the next node and validate the resulting route
                    new_id = self._generate_route_id(cand_inx, step_inx)
                    candidate_status, new_candidate = (
                        self.zimmer.step_to_next_node(
                            candidate, possible_step, new_id
                        )
                    )

                    # Make sure the route can get back to the starting node
                    if iters == 2:
                        new_candidate = self._remove_start_node_from_visited(
                            new_candidate
                        )

                    # Check whether the route is still valid
                    if candidate_status == "complete":
                        self.completed_routes.append(new_candidate)
                    elif candidate_status == "valid":
                        new_candidates.append(new_candidate)

            # Make sure the total number of routes stays below the configured
            # limit
            new_candidates = self.pruner.prune_routes(new_candidates)
            self.last_candidates = self.candidates
            self.candidates = new_candidates

            # Update the progress bar
            iters += 1
            self._update_progress_bar(pbar)

        self.completed_routes = sorted(
            self.completed_routes,
            key=lambda x: x.ratio,
            reverse=self.config.route_mode == "hilly",
        )

        return self.completed_routes
