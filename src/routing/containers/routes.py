"""Container objects which are used to represent various structures used
during the route finding process"""

from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional


@dataclass
class StepMetrics:
    """Container for metrics calculated when stepping from the end of one
    route to a neighbouring node.

    Args:
        distance (float): The distance change
        elevation_gain (float): The elevation increase
        elevation_loss (float): The elevation los
    """

    distance: float
    elevation_gain: float
    elevation_loss: float


@dataclass
class Route:
    """Container for the information required to represent a single route

    Args:
        route (List[int]): A list of the node IDs which are crossed as part of
          this route, in the order that they are crossed
        visited (Set[int]): A set of all the unique nodes which are visited as
          part of this route
        distance (float): The total distance of the route
        elevation_gain (float): The elevation gain for this route
        elevation_loss (float): The elevation loss for this route
        elevation_gain_potential (float): The elevation gain required in order
          to get back to the route's starting point
        elevation_loss_potential (float): The elevation loss required in order
          to get back to the route's starting point
        ratio (float): The ratio of elevation gained to distance travelled
        terminal_square (Optional[Tuple[int, int]]): The grid square in which
          this route terminates, used only while pruning a list of candidate
          routes"""

    route: List[int]
    route_id: str

    current_position: int
    visited: Set[int]
    terminal_square: Optional[Tuple[int, int]] = None

    distance: float = 0.0
    elevation_gain: float = 0.0
    elevation_loss: float = 0.0
    elevation_gain_potential: float = 0.0
    elevation_loss_potential: float = 0.0
    ratio: float = 0.0


@dataclass
class RouteConfig:
    """Contains user configuration options for route calculation

    Args:
        start_lat (float): Latitude for the route start point
        start_lon (float): Longitude for the route start point
        target_distance (float): Target distance for the route (in km)
        tolerance (float): How far above/below the target distance a
          completed route can be while still being considered valid
        route_mode (str): Set to 'hilly' to generate the hilliest possible
          route, or 'flat' for the flattest possible route
        max_candidates (int): The maximum number of candidate routes which
          should be held in memory. Lower this to increase calculation speed,
          increase it to potentially increase the quality of routes generated.
        terrain_types (List[str]): The different types of terrain which should
          be considered for this route. If not provided, all terrain types
          will be considered. Defaults to None. Possible options are:

          - footway
          - living_street
          - path
          - pedestrian
          - primary
          - primary_link
          - residential
          - secondary
          - secondary_link
          - service
          - steps
          - tertiary
          - tertiary_link
          - track
          - unclassified
    """

    start_lat: float
    start_lon: float

    target_distance: float
    tolerance: float

    route_mode: str
    max_candidates: int

    terrain_types: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate min_distance and max_distance based on user provided
        target_distance and tolerance"""

        self.min_distance = self.target_distance / (1 + self.tolerance)
        self.max_distance = self.target_distance * (1 + self.tolerance)
