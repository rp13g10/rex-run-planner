"""Contains the GraphEnricher class, which will be executed if this script
is executed directly."""

from itertools import repeat
from math import ceil
from typing import List, Set, Tuple, Union

import networkx as nx
from networkx import Graph
from networkx.exception import NetworkXError
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from refinement.containers import ChainMetrics
from refinement.graph_utils.splitter import (
    GraphSplitter,
)


class GraphCondenser:
    """Class which enriches the data which is provided by Open Street Maps.
    Unused data is stripped out, and elevation data is added for both nodes and
    edges. The graph itself is condensed, with nodes that lead to dead ends
    or only represent a bend in the route being removed.
    """

    def __init__(self, graph: Graph, cb_nodes: Set[int]):
        """Create an instance of the graph enricher class based on the
        contents of the networkx graph specified by `source_path`

        Args:
            source_path (str): The location of the networkx graph to be
              enriched. The graph must have been saved to json format.
            dist_mode (str, optional): The preferred output mode for distances
              which are saved to node edges. Returns kilometers if set to
              metric, miles if set to imperial. Defaults to "metric".
            elevation_interval (int, optional): When calculating elevation
              changes across an edge, values will be estimated by taking
              checkpoints at regular checkpoints. Smaller values will result in
              more accurate elevation data, but may slow down the calculation.
              Defaults to 10.
            max_condense_passes (int, optional): When condensing the graph, new
              dead ends may be created on each pass (i.e. if one dead end
              splits into two, pass 1 removes the 2 dead ends, pass 2 removes
              the one they split from). Use this to set the maximum number of
              passes which will be performed.
        """

        # TODO: Update this docstring

        # Store down user preferences
        self.graph = graph
        self.max_condense_passes = 5
        self.cb_nodes = cb_nodes

        # Create container objects
        self.nodes_to_condense = set()
        self.nodes_to_remove = set()

    def _remove_isolates(self):
        """Remove any nodes from the graph which are not connected to another
        node."""
        isolates = set(nx.isolates(self.graph))
        self.graph.remove_nodes_from(isolates)

    def _remove_dead_ends(self):
        """Remove any nodes which have been flagged for removal on account of
        them representing a dead end."""
        self.graph.remove_nodes_from(self.nodes_to_remove)

    def _get_node_degree(self, node_id) -> int:
        edges = self.graph.edges(node_id)
        node_degree = len(edges)
        return node_degree

    def _refresh_node_lists(self):
        """Check the graph for any nodes which can be condensed, or removed
        entirely."""
        self.nodes_to_condense = set()
        self.nodes_to_remove = set()

        for node_id in self.graph.nodes:
            node_degree = self._get_node_degree(node_id)

            if node_degree >= 3:
                # Node is a junction, must be retained
                continue
            elif node_id in self.cb_nodes:
                continue
            elif node_degree == 2:
                # Node represents a bend in a straight line, can safely be
                # condensed
                self.nodes_to_condense.add(node_id)
            elif node_degree == 1:
                # Node is a dead end, can safely be removed
                self.nodes_to_remove.add(node_id)
            # Node is an orphan, will be caught by remove_isolates
            continue

    def _update_node_lists(self, applied_chain: List[int]):
        start_id = applied_chain[0]
        # removed_id = applied_chain[1]
        end_id = applied_chain[2]

        # self.nodes_to_condense.remove(removed_id)

        end_degree = self._get_node_degree(end_id)

        if start_id in self.nodes_to_condense:
            start_degree = self._get_node_degree(start_id)
            if start_degree != 2:
                self.nodes_to_condense.remove(start_id)
        if end_id in self.nodes_to_condense:
            end_degree = self._get_node_degree(end_id)
            if end_degree != 2:
                self.nodes_to_condense.remove(end_id)

    def _generate_node_chain(
        self, node_id: int
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        """For a node with order 2, generate a chain which represents the
        traversal of both edges. For example if node_id is B and node B is
        connected to nodes A and C, the resultant chain would be [A, B, C].

        Args:
            node_id (int): The node which forms the middle of the chain

        Returns:
            Tuple[List[int], List[Tuple[int, int]]]: A list of the node IDs
              which are crossed as part of this chain, and a list of the
              edges which are traversed as part of this journey.
        """
        node_edges = list(self.graph.edges(node_id))
        node_chain = [node_edges[0][1], node_edges[0][0], node_edges[1][1]]

        return node_chain, node_edges

    def _calculate_chain_metrics(
        self, chain: List[int]
    ) -> Union[ChainMetrics, None]:
        """For a chain of 3 nodes, sum up the metrics for each of the edges
        which it is comprised of.

        Args:
            chain (List[int]): A list of 3 node IDs

        Returns:
            ChainMetrics: A container for the calculated metrics
        """
        try:
            # Fetch the two edges for the chain
            edge_1 = self.graph[chain[0]][chain[1]]
            edge_2 = self.graph[chain[1]][chain[2]]
        except KeyError:
            # Edge does not exist in this direction
            return None

        # Retrieve data for edge 1
        gain_1 = edge_1["elevation_gain"]
        loss_1 = edge_1["elevation_loss"]
        dist_1 = edge_1["distance"]
        via_1 = edge_1.get("via", [])

        # Retrieve data for edge 2
        gain_2 = edge_2["elevation_gain"]
        loss_2 = edge_2["elevation_loss"]
        dist_2 = edge_2["distance"]
        via_2 = edge_2.get("via", [])

        # Calculate whole-chain metrics
        metrics = ChainMetrics(
            start=chain[0],
            end=chain[-1],
            gain=gain_1 + gain_2,
            loss=loss_1 + loss_2,
            dist=dist_1 + dist_2,
            vias=via_1 + [chain[1]] + via_2,
        )

        return metrics

    def _add_edge_from_chain_metrics(self, metrics: Union[ChainMetrics, None]):
        """Once metrics have been calculated for a node chain, use them to
        create a new edge which skips over the middle node. The ID of this
        middle node will be recorded within the `via` attribute of the new
        edge.

        Args:
            metrics (Union[ChainMetrics, None]): Container for calculated
              metrics for this chain
        """
        if metrics:
            self.graph.add_edge(
                metrics.start,
                metrics.end,
                via=metrics.vias,
                elevation_gain=metrics.gain,
                elevation_loss=metrics.loss,
                distance=metrics.dist,
            )

    def _remove_original_edges(self, node_edges: List[Tuple[int, int]]):
        """Once a new edge has been created based on a node chain, the
        original edges can be removed.

        Args:
            node_edges (List[Tuple[int, int]]): A list of node edges to be
              removed from the graph.
        """
        # Remove original edges
        for start, end in node_edges:
            try:
                self.graph.remove_edge(start, end)
            except NetworkXError:
                pass
            try:
                self.graph.remove_edge(end, start)
            except NetworkXError:
                pass

    def condense_graph(self, _iter: int = 0):
        """Disconnect all nodes from the graph which contribute only
        geometrical information (i.e. they form corners along paths/roads but
        do not represent junctions). Update the edges in the graph to skip over
        these nodes, instead going direct from one junction to the next.

        Args:
            _iter (int, optional): The number of times this function has been
              called. This is incremented automatically and should not be
              configured by the user. Defaults to 0.
        """
        self._remove_isolates()
        self._refresh_node_lists()

        # Early stopping condition
        if not self.nodes_to_condense:
            return

        iters = 0
        while self.nodes_to_condense:
            node_id = self.nodes_to_condense.pop()

            node_chain, node_edges = self._generate_node_chain(node_id)

            se_metrics = self._calculate_chain_metrics(node_chain)
            es_metrics = self._calculate_chain_metrics(node_chain[::-1])

            self._add_edge_from_chain_metrics(se_metrics)
            self._add_edge_from_chain_metrics(es_metrics)

            self._remove_original_edges(node_edges)

            # TODO: Update node lists based on knowledge of last node
            #       processed, rather than redoing the entire thing
            # self._refresh_node_lists()
            self._update_node_lists(node_chain)

            iters += 1

        if not self.cb_nodes:
            # Dead end on a subgraph might not actually be a dead end
            self._remove_dead_ends()

        if _iter < self.max_condense_passes:
            self.condense_graph(_iter=_iter + 1)


def _condense_subgraph(subgraph: Graph, edge_nodes: Set[int]) -> Graph:
    """Condense a single subgraph and return it

    Args:
        subgraph (Graph): The subgraph to be condensed
        edge_nodes (Set[int]): A set of nodes which are at the very edge of
          the subgraph. These will be excluded from any checks for dead-end
          edges as they may continue into adjacent grid squares.

    Returns:
        Graph: A condensed representation of the provided graph
    """
    condenser = GraphCondenser(subgraph, edge_nodes)
    condenser.condense_graph()

    return condenser.graph


def _condense_subgraph_star(args):
    """Wrapper function, allows use of _condense_subgraph with process_map
    by expanding out all provided arguments"""
    return _condense_subgraph(*args)


def condense_graph(graph: Graph) -> Graph:
    """For a given graph, minimise its size in-memory by removing any nodes
    which do not correspond to a junction. Paths/roads will be represented by
    a single edge, rather than a chain of edges. The intermediate nodes
    traversed by each edge will instead be recorded in the 'via' attribute
    of each new edge.

    Args:
        graph (Graph): The graph to be condensed

    Returns:
        Graph: A condensed version of the input graph
    """

    # Split the graph across a grid
    no_subgraphs = ceil(len(graph.nodes) / 10000)

    splitter = GraphSplitter(graph, no_subgraphs=no_subgraphs)
    splitter.explode_graph()
    cb_nodes = splitter.edge_nodes

    # Condense each grid separately
    map_args = zip(splitter.subgraphs.values(), repeat(cb_nodes))
    new_subgraphs = process_map(
        _condense_subgraph_star,
        map_args,
        desc="Condensing subgraphs",
        tqdm_class=tqdm,
        total=len(splitter.grid),
        # max_workers=8,
    )

    # Re-combine the condensed subgraphs
    for new_subgraph in new_subgraphs:
        subgraph_id = new_subgraph.graph["grid_square"]
        splitter.subgraphs[subgraph_id] = new_subgraph
    splitter.rebuild_graph()

    # Perform a mop-up to pick up any edges which spanned more than one
    # subgraph
    graph = _condense_subgraph(splitter.graph, set())

    return graph
