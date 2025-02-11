"""In the absence of a web frontend, this script is used to trigger library
calls and generate/plot routes"""

from routing.containers.routes import RouteConfig
from routing.route_maker.route_maker import RouteMaker
from routing.route_maker.route_selector import RouteSelector
from routing.plotting.plotting import (
    plot_route,
    plot_elevation_profile,
    generate_filename,
)

"""
Plan:
 Phase 1 - Get something tangible
 * Finish off refactor, clear out archived code once satisfied
 * Improve static plotting utilities
   * Look in to using Folium / dash-leaflet
   * Folium has a ColorLine option which could do this
 * Use plots to validate against Strava
   * Elevation profiles in particular should be very helpful
 * Build out a basic webapp as a PoC
 * Set up function to spit out .gpx files

 Phase 2 - Foundations for scalable webapp
 * Look in to options for improving cassandra cluster performance locally,
   suspect memory causing bottleneck at present
 * Check whether there is any way to use partitions with graphframes
 * Swap over to using Docker for pySpark
 * Sort out your requirements files! There should not be missing dependencies

 Phase 3 - Set up the data preparation pipeline
  Single k8s stack with all required components
  * Cassandra database
  * pySpark cluster
  * airflow
    * triggers ingestion & processing when new files land
  * custom ingestion script
    https://docs.datastax.com/en/dsbulk/docs/overview/dsbulk-about.html
    Loads data from lidar files into the database
  * custom tagging script
    * triggers tagging of nodes/edges with elevation data once ingestion
      completes
  * custom parsing script
    osm to networkx, one-off run, arguably could be left out of stack

 Phase 4 - Full build
  * Set up a separate k8s stack for the webapp (or add to existing? check
    suggested design patterns)
  * Move things over to the cloud
"""

# TODO: Edge 4331 to 4336, does it appear in the full graph at the point
#       it is generated by the selector? Need to step through the process...

config = RouteConfig(
    start_lat=50.969540,
    start_lon=-1.383318,
    target_distance=10,
    route_mode="flat",
    max_candidates=2048,
    tolerance=0.1,
)

maker = RouteMaker(config)
routes = maker.find_routes()

selector = RouteSelector(routes, num_routes_to_select=25, threshold=0.85)
selected_routes = selector.select_routes()


for route in selected_routes[:10]:
    route_plot = plot_route(maker.graph, route)
    profile_plot = plot_elevation_profile(maker.graph, route)
    fname = generate_filename(route)
    route_plot.write_html(
        f"/home/ross/repos/rex_run_planner/plots/{fname}_route.html"
    )
    profile_plot.write_html(
        f"/home/ross/repos/rex_run_planner/plots/{fname}_profile.html"
    )
