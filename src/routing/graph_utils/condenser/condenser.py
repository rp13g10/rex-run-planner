"""Contains the GraphCondenser class, which is used to minimise the size of
graph used to generate routes. This is achieved by removing any nodes which
represent a turn in a road rather than a junction, and removing any
dead ends."""

import itertools
from copy import deepcopy
from typing import List, Tuple, Set, Any, Dict

import networkx as nx
from networkx import DiGraph
from networkx.exception import NetworkXNoPath


class GraphCondenser:
    """Minimises the size the provided graph. This is achieved by removing any
    nodes which represent a turn in a road rather than a junction, and removing
    any dead ends.
    """

    def __init__(self, graph: DiGraph, start_node: int):
        """Create an instance of the graph condenser using the provided graph.
        This will create a copy of the provided graph, but the graph will
        not be modified until the `condense_graph` method is called.

        Args:
            graph (DiGraph): The network graph to be condensed
            start_node (int): The start point for the route to be generated,
              required so that we don't remove it as a result of condensing
              the graph.
        """

        # Store down user preferences
        self.graph = deepcopy(graph)
        self.start_node = start_node
        self.max_condense_passes = 5
        self._condense_passes = 0

    def _remove_isolates(self):
        """Remove any nodes from the graph which are not connected to another
        node."""
        isolates = set(nx.isolates(self.graph))
        self.graph.remove_nodes_from(isolates)

    def _generate_node_sets(self) -> Tuple[Set[int], Set[int]]:
        """Generate two sets of nodes; those which form part of the chains that
        represent road geometry, and those which represent junctions.

        Returns:
            Tuple[Set[int], Set[int]]: A tuple containing a set of chain nodes,
              and a set of other nodes
        """

        # NOTE: We cannot simply require that the degree be 4, as this is a
        #       directed graph. A node with 4 one-way exit points should not
        #       be considered part of a chain. This prevents us from using the
        #       built in graph.degree attribute.
        node_list = list(self.graph.nodes)
        degree_list = [
            len(set(nx.all_neighbors(self.graph, node))) for node in node_list
        ]

        chains_nodes = {
            node
            for node, degree in zip(node_list, degree_list)
            if degree == 2 and node != self.start_node
        }
        other_nodes = {
            node
            for node, degree in zip(node_list, degree_list)
            if degree != 2 or node == self.start_node
        }
        return chains_nodes, other_nodes

    def _generate_chain_subgraphs(
        self, chains_nodes: Set[int]
    ) -> List[DiGraph]:
        """Given a set containing all nodes which form part of a chain, return
        a list of graphs where each graph contains a single chain.

        Args:
            chains_nodes (Set[int]): Every node in the internal graph which
              forms part of a chain

        Returns:
            List[DiGraph]: A list of graphs, where each graph represents a
              single chain in the internal graph
        """

        # Single graph, contains all chains
        chains_graph = self.graph.subgraph(chains_nodes)

        # List of lists of node IDs, each list represents one chain
        chains_list = nx.weakly_connected_components(chains_graph)

        # List of graphs, each graph represents on chain
        chain_graphs = [chains_graph.subgraph(chain) for chain in chains_list]

        return chain_graphs

    def _generate_chain_endpoints(
        self, chain_graph: DiGraph
    ) -> Tuple[int, int, int, int]:
        """For a given chain, work out where it starts and ends. For the
        identified start and end points, work out which junctions it connects
        to.

        Args:
            chain_graph (DiGraph): A directed graph representing a single chain

        Returns:
            Tuple[int, int, int, int]: First node in the chain, last node in
              the chain, connection point for first node, connection point for
              last node
        """

        chain_nodes = set(chain_graph.nodes)
        if len(chain_nodes) == 1:

            # Start and end will be the same
            chain_start = chain_end = list(chain_graph.nodes)[0]

            # Single node will necessarily have 2 connections, as this is a
            # requirement for any nodes in a chain
            connections = list(
                set(nx.all_neighbors(self.graph, chain_start)) - chain_nodes
            )

            # Arbitrarily define one as the starting connection, and the other
            # as the ending connection.
            edge_start = connections[0]
            edge_end = connections[1]

        else:

            # Identify the nodes which connect to only one other node in the
            # chain as the end points
            node_list = list(chain_graph.nodes)
            degree_list = [
                len(set(nx.all_neighbors(chain_graph, node)))
                for node in node_list
            ]

            end_points = [
                node
                for node, degree in zip(node_list, degree_list)
                if degree == 1
            ]

            # Arbitrarily define one as the start of the chain, and the other
            # as the end
            chain_start = end_points[0]
            chain_end = end_points[1]

            # Identify the nodes which connect to the start and end nodes in
            # the chain
            edge_start = (
                set(nx.all_neighbors(self.graph, chain_start)) - chain_nodes
            ).pop()
            edge_end = (
                set(nx.all_neighbors(self.graph, chain_end)) - chain_nodes
            ).pop()

        return chain_start, chain_end, edge_start, edge_end

    def _generate_hyperedge_metrics(
        self, start: int, end: int, vias: List[int]
    ) -> Dict[str, Any]:
        """The new edge to be added must retain the pre-calculated distance
        and elevation metrics. To do this, we must sum the metrics for each
        link in the chain.

        Args:
            start (int): The starting point for the chain
            end (int): The ending point for the chain
            vias (List[int]): The mid-points in the chain

        Raises:
            NetworkXNoPath: If no route between `start` and `end` exists, an
              exception will be raised.

        Returns:
            Dict[str, Any]: A dictionary containing distance, elevation_gain,
              elevation_loss and type keys.
        """
        edges = itertools.pairwise([start] + vias + [end])

        metrics = {
            "distance": 0.0,
            "elevation_gain": 0.0,
            "elevation_loss": 0.0,
            "type": "composite",
        }
        for edge in edges:
            for metric in ["distance", "elevation_gain", "elevation_loss"]:
                try:
                    metrics[metric] += self.graph[edge[0]][edge[1]][metric]
                except KeyError as exc:
                    # This may occur when a bi-directional chain is connected
                    # to the main graph by a one-directional edge. Note that
                    # chains of length 1 are treated as bi-directional when
                    # asking networkx to calculate the shortest path.
                    raise NetworkXNoPath from exc

        via_coords = [
            (
                self.graph.nodes[node]["lat"],
                self.graph.nodes[node]["lon"],
                self.graph.nodes[node]["elevation"],
            )
            for node in vias
        ]
        metrics["via"] = via_coords

        return metrics

    def _remove_dead_ends(self, condensed_graph: DiGraph) -> DiGraph:
        """Remove any dead ends from the condensed graph. This is achieved by
        removing any nodes which connect to one one other node.

        TODO: Confirm whether multiple passes are required when running this
              over a condensed graph. Suspect it probably isn't.

        Args:
            condensed_graph (DiGraph): The condensed graph to be processed

        Returns:
            DiGraph: A reference to `condensed_graph` with no dead ends
        """

        # Identify any dead end nodes
        node_list = list(condensed_graph.nodes)
        degree_list = [
            len(set(nx.all_neighbors(condensed_graph, node)))
            for node in node_list
        ]
        dead_end_nodes = {
            node
            for node, degree in zip(node_list, degree_list)
            if degree == 1 and node != self.start_node
        }

        # Terminate early if none are present
        if not dead_end_nodes:
            return condensed_graph

        # Remove dead end nodes, make a second pass
        condensed_graph.remove_nodes_from(dead_end_nodes)

        self._condense_passes += 1

        if self._condense_passes == self.max_condense_passes:
            return condensed_graph
        return self._remove_dead_ends(condensed_graph)

    def condense_graph(self) -> DiGraph:
        """Replace the internal graph with a condensed representation of
        itself. Any node chains which represent the geometry of a road will
        be replaced with a single edge going from one junction to the next.
        This will allow the route finding algorithm to generate new routes
        more quickly.

        Returns:
            DiGraph: A condensed representation of the internal graph
        """
        # Remove any orphaned nodes
        self._remove_isolates()

        # Work out which nodes form part of a chain
        chains_nodes, other_nodes = self._generate_node_sets()

        # Create one graph for each chain
        chain_subgraphs = self._generate_chain_subgraphs(chains_nodes)

        # Generate new hyperedges based on these chains
        new_edges = []
        for chain_graph in chain_subgraphs:

            # Figure out the start & end points for the chain, and which
            # junctions it connects to
            chain_start, chain_end, edge_start, edge_end = (
                self._generate_chain_endpoints(chain_graph)
            )

            # If possible, generate a new edge going from start to end across
            # the entire chain
            try:
                vias_se = nx.shortest_path(chain_graph, chain_start, chain_end)
                metrics_se = self._generate_hyperedge_metrics(
                    edge_start, edge_end, vias_se  # type: ignore
                )
                new_edges.append((edge_start, edge_end, metrics_se))
            except NetworkXNoPath:
                pass

            # If possible, generate a new edge going from end to start across
            # the entire chain
            try:
                vias_es = nx.shortest_path(chain_graph, chain_end, chain_start)
                metrics_es = self._generate_hyperedge_metrics(
                    edge_end, edge_start, vias_es  # type: ignore
                )
                new_edges.append((edge_end, edge_start, metrics_es))
            except NetworkXNoPath:
                pass

        # Create a new graph with all chains represented by these new
        # hyperedges
        condensed_graph = self.graph.subgraph(other_nodes).copy()
        condensed_graph.add_edges_from(new_edges)

        return condensed_graph
