# TODO: Implement class which returns materially different routes from a
#       list of valid routes, use difflib and iterate through the sorted
#       list. Skip any items with a similarity score which is above a defined
#       threshold when compared to any already selected route.
# from difflib import SequenceMatcher
from thefuzz import fuzz
from typing import List
from routing.containers.routes import Route


class RouteSelector:
    """Takes the top N routes from a pre-sorted list of candidates, ensuring
    that each route is sufficiently different to all of the routes which
    preceeded it."""

    def __init__(
        self, routes: List[Route], num_routes_to_select: int, threshold: float
    ):
        """Create a route selector with the provided parameters

        Args:
            routes (List[Route]): A list of valid route, sorted according to
              their desired elevation profile
            num_routes_to_select (int): How many distinct routes should be
              pulled from the provided list
            threshold (float): How similar can each route be to the next.
              Set to 0 to allow absolutely no overlap, set to 1 to allow
              even completely identical routes.
        """
        self.routes = routes
        self.n_routes = num_routes_to_select
        self.threshold = threshold

        self.selected_routes: List[Route] = []

    @staticmethod
    def get_similarity(route: str, selected_route: str):
        # return SequenceMatcher(a=route, b=selected_route).ratio()
        ratio = fuzz.ratio(route, selected_route)
        ratio /= 100
        return ratio

    def select_routes(self):
        for route in self.routes:
            is_selectable = True
            route_str = " ".join(str(node) for node in route.route)

            for selected_route in self.selected_routes:
                selected_str = " ".join(
                    str(node) for node in selected_route.route
                )
                route_diff = self.get_similarity(route_str, selected_str)
                if route_diff > self.threshold:
                    is_selectable = False

            if is_selectable:
                if len(self.selected_routes) < self.n_routes:
                    self.selected_routes.append(route)
                if len(self.selected_routes) == self.n_routes:
                    return self.selected_routes

        return self.selected_routes
