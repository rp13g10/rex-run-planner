from dash import Dash, html, dcc, callback, Output, Input, State, no_update

from routing.containers.routes import RouteConfig
from routing.route_maker.route_maker import RouteMaker
from routing.plotting.plotting import (
    plot_route,
    plot_elevation_profile,
)

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Basic Webapp", style={"textAlign": "center"}),
        html.Div("Select Latitude"),
        dcc.Input(type="number", value=50.969540, id="route-lat"),
        html.Div("Select Longitude"),
        dcc.Input(type="number", value=-1.383318, id="route-lon"),
        html.Div("Set target distance"),
        dcc.Input(type="number", value=10, id="route-dist"),
        html.Div("Set mode"),
        dcc.Dropdown(["hilly", "flat"], value="hilly", id="route-mode"),
        html.Button("Calculate", id="route-calculate"),
        dcc.Graph(id="route-plot"),
        dcc.Graph(id="route-profile"),
    ]
)


@callback(
    [Output("route-plot", "figure"), Output("route-profile", "figure")],
    Input("route-calculate", "n_clicks"),
    [
        State("route-lat", "value"),
        State("route-lon", "value"),
        State("route-dist", "value"),
        State("route-mode", "value"),
    ],
)
def calculate_and_render_route(
    n_clicks, route_lat, route_lon, route_dist, route_mode
):
    if not n_clicks:
        return no_update, no_update

    config = RouteConfig(
        start_lat=route_lat,
        start_lon=route_lon,
        target_distance=route_dist,
        route_mode=route_mode,
        max_candidates=2048,
        tolerance=0.1,
    )

    maker = RouteMaker(config)
    routes = maker.find_routes()

    route = routes[0]

    route_plot = plot_route(maker.full_graph, route)
    profile_plot = plot_elevation_profile(maker.full_graph, route)

    return route_plot, profile_plot


if __name__ == "__main__":
    app.run(debug=True)
