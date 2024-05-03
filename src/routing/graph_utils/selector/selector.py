"""Contains the Selector class, which retrieves a graph containing all nodes
which can be reached from the requested route start point without going over
the max configured distance."""

from typing import Dict, List, Tuple

import graph_tool.all as gt
import numpy as np
from geopy import distance, point
from pyarrow.parquet import ParquetDataset
import networkx as nx

from routing.containers.pruning import BBox
from routing.containers.routes import RouteConfig
from routing.graph_utils import find_nearest_node, remove_isolates

# TODO: Remove hard-coded data directories


class Selector:
    """Retrieves a graph containing all nodes which can be reached from the
    requested route start point without going over the max configured distance.
    """

    def __init__(
        self,
        config: RouteConfig,
    ):
        """Create an instance of the graph enricher class based on the
        contents of the networkx graph specified by `source_path`

        Args:
            config (RouteConfig): A configuration file detailing the route
              requested by the user.
        """

        # Store down core attributes
        self.config = config
        self.bbox = self.get_bounding_box_for_route()

    def get_bounding_box_for_route(self) -> BBox:
        """Generate a square bounding box which contains a circle with diameter
        equal to the max requested distance.

        Returns:
            BBox: A bounding box for the entire route
        """
        start_point = point.Point(self.config.start_lat, self.config.start_lon)

        dist_to_corner = (self.config.max_distance / 2) * (2**0.5)

        nw_corner = distance.distance(kilometers=dist_to_corner).destination(
            point=start_point, bearing=315
        )

        se_corner = distance.distance(kilometers=dist_to_corner).destination(
            point=start_point, bearing=135
        )

        bbox = BBox(
            min_lat=se_corner.latitude,
            min_lon=nw_corner.longitude,
            max_lat=nw_corner.latitude,
            max_lon=se_corner.longitude,
        )

        return bbox

    def retrieve_nodes_for_bounding_box(self) -> List[Dict]:
        """For the provided bounding box, fetch a list of dictionaries from
        the enriched parquet dataset. Each entry in the list represents one
        node in the graph.

        Returns:
            List[Dict]: A list of node metadata
        """
        nodes_dataset = ParquetDataset(
            "/home/ross/repos/refinement/data/enriched_nodes",
            filters=[
                ("easting_ptn", ">=", self.bbox.min_easting_ptn),
                ("easting_ptn", "<=", self.bbox.max_easting_ptn),
                ("northing_ptn", ">=", self.bbox.min_northing_ptn),
                ("northing_ptn", "<=", self.bbox.max_northing_ptn),
            ],
        )

        node_cols = ["id", "lat", "lon", "elevation"]

        nodes_dict = nodes_dataset.read(columns=node_cols).to_pydict()

        nodes_list = [
            {node_cols[i]: vals[i] for i in range(len(node_cols))}
            for vals in zip(*[nodes_dict.get(col) for col in node_cols])
        ]

        return nodes_list

    def retrieve_edges_for_bounding_box(self) -> List[Tuple]:
        """For the provided bounding box, fetch a list of tuples from the
        enriched parquet dataset. Each entry in the list represents one edge
        in the graph.

        Returns:
            List[Tuple]: A list of edges & the corresponding metadata
        """
        edges_dataset = ParquetDataset(
            "/home/ross/repos/refinement/data/enriched_edges",
            filters=[
                ("easting_ptn", ">=", self.bbox.min_easting_ptn),
                ("easting_ptn", "<=", self.bbox.max_easting_ptn),
                ("northing_ptn", ">=", self.bbox.min_northing_ptn),
                ("northing_ptn", "<=", self.bbox.max_northing_ptn),
            ],
        )

        edge_cols = [
            "src",
            "dst",
            "distance",
            "elevation_gain",
            "elevation_loss",
            "type",
        ]

        edges_dict = edges_dataset.read(columns=edge_cols).to_pydict()

        edges_list = list(zip(*[edges_dict.get(col) for col in edge_cols]))

        return edges_list

    def fetch_coarse_subgraph(self) -> gt.Graph:
        """Fetch a graph which covers roughly the right area, by filtering with
        a square bounding box with edges the same length as the max requested
        route distance.

        Returns:
            gt.Graph: A graph-tool graph object.
        """
        nodes_list = self.retrieve_nodes_for_bounding_box()
        edges_list = self.retrieve_edges_for_bounding_box()

        graph = gt.Graph()

        graph.add_edge_list(
            edges_list,
            hashed=False,
            eprops=[
                ("distance", "double"),
                ("elevation_gain", "double"),
                ("elevation_loss", "double"),
                ("type", "string"),
            ],
        )

        graph.vertex_properties["lat"] = graph.new_vertex_property("double")
        graph.vertex_properties["lon"] = graph.new_vertex_property("double")
        graph.vertex_properties["elevation"] = graph.new_vertex_property(
            "double"
        )

        for node_dict in nodes_list:
            node_id = node_dict["id"]
            try:
                _ = graph.vertex(node_id, add_missing=False)
            except ValueError:
                continue
            for prop in ["lat", "lon", "elevation"]:
                graph.vertex_properties[prop][node_id] = node_dict[prop]

        return graph

    def fetch_node_coords(
        self, graph: gt.Graph, node_id: int
    ) -> Tuple[int, int]:
        """Convenience function, retrieves the latitude and longitude for a
        single node in a graph."""
        lat = graph.vertex_properties["lat"][node_id]
        lon = graph.vertex_properties["lon"][node_id]
        return lat, lon

    def tag_distances_to_start(
        self, graph: gt.Graph, start_node: np.int64
    ) -> gt.Graph:
        """Tags each node in the graph with the distance & elevation which
        must be travelled in order to get back to the start point."""

        # NOTE: Dijkstra's algorithm finds the distance from the start, to
        #       every other node in the graph. We need the distance from every
        #       node back to the start, so must invert all of the edges. This
        #       is a directed graph, so the two problems may not be equivalent
        inverse_graph = graph.copy()
        new_edges = [
            (end, start, dist)
            for start, end, dist in inverse_graph.iter_edges(
                eprops=[inverse_graph.edge_properties["distance"]]
            )
        ]
        inverse_graph.clear_edges()
        inverse_graph.add_edge_list(new_edges, eprops=[("distance", "float")])

        # Calculate distances back to start using the inverted graph
        source = inverse_graph.vertex(start_node)
        distance = inverse_graph.edge_properties["distance"]
        dist_i, _ = gt.dijkstra_search(
            inverse_graph, weight=distance, source=source
        )

        # Apply the calculated distances back to the original graph
        dist = graph.new_vertex_property("double")
        dist.a = dist_i.a
        graph.vertex_properties["dist_to_start"] = dist

        return graph

    def generate_fine_subgraph(self, graph: gt.Graph) -> gt.Graph:
        """After tagging each node with its distance from the start point, use
        this information to remove all nodes which cannot be reached without
        the route going over the maximum configured distance.

        Returns:
            Graph: A fully trimmed graph where all nodes are a useful distance
              from the start point.
        """

        filter_prop = graph.new_vertex_property("bool")
        search_radius = self.config.max_distance / 2
        filter_vals = (
            graph.vertex_properties["dist_to_start"].get_array()
            <= search_radius
        )
        filter_prop.a = filter_vals

        graph.set_vertex_filter(prop=filter_prop)

        graph.purge_vertices()
        graph.purge_edges()

        return graph

    def convert_graph_from_gt_to_nx(self, graph: gt.Graph) -> nx.DiGraph:
        """Until downstream components have been updated to work with
        graph-tool, convert the output to networkx for compatibility."""

        node_props = ["lat", "lon", "elevation", "dist_to_start"]
        node_array = graph.get_vertices(
            vprops=[
                getattr(graph.vertex_properties, prop) for prop in node_props
            ]
        )
        node_list = [
            (
                int(node[0]),
                {prop: node[inx] for inx, prop in enumerate(node_props, 1)},
            )
            for node in node_array
        ]

        edge_props = ["distance", "elevation_gain", "elevation_loss"]
        edge_array = graph.get_edges(
            eprops=[
                getattr(graph.edge_properties, prop) for prop in edge_props
            ]
        )
        type_list = graph.edge_properties["type"]
        edge_list = [
            (
                int(edge[0]),
                int(edge[1]),
                {
                    prop: edge[inx] if prop != "type" else type_
                    for inx, prop in enumerate(edge_props + ["type"], 2)
                },
            )
            for edge, type_ in zip(edge_array, type_list)
        ]

        graph.clear()

        nx_graph = nx.DiGraph()
        nx_graph.add_nodes_from(node_list)
        nx_graph.add_edges_from(edge_list)

        return nx_graph

    def create_graph(self) -> Tuple[int, nx.DiGraph]:
        """Based on the provided route configuration, return a networkx graph
        which represents the road/path network in the area which can be reached
        from the start point without going over the max configured distance.

        Returns:
            Tuple[int, nx.DiGraph]: The ID for the starting node in the graph,
              and the graph itself
        """
        graph = self.fetch_coarse_subgraph()

        graph = remove_isolates(graph)

        start_node = find_nearest_node(
            graph, self.config.start_lat, self.config.start_lon
        )
        graph = self.tag_distances_to_start(graph, start_node)
        graph = self.generate_fine_subgraph(graph)
        start_node = find_nearest_node(
            graph, self.config.start_lat, self.config.start_lon
        )
        start_node = int(start_node)
        nx_graph = self.convert_graph_from_gt_to_nx(graph)

        return start_node, nx_graph
