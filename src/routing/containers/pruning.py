from dataclasses import dataclass
from bng_latlon import WGS84toOSGB36


@dataclass
class BBox:
    """Contains information about the physical boundaries of one or more
    routes

    Args:
        min_lat (float): Minimum latitude
        min_lon (float): Minimum longitude
        max_lat (float): Maximum latitude
        max_lon (float): Maximum longitude"""

    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

    def __post_init__(self):
        min_easting, min_northing = WGS84toOSGB36(self.min_lat, self.min_lon)
        max_easting, max_northing = WGS84toOSGB36(self.max_lat, self.max_lon)
        self.min_easting_ptn = int(min_easting) // 1000
        self.min_northing_ptn = int(min_northing) // 1000
        self.max_easting_ptn = int(max_easting) // 1000
        self.max_northing_ptn = int(max_northing) // 1000
