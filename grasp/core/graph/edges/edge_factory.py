from abc import ABC
from typing import Any, Optional

from grasp.core.graph.nodes.base_node import BaseNode, NodeType
from grasp.core.graph.nodes.node_utils import get_node
from grasp.core.graph.nodes.special_node import SpecialNode
from grasp.logger.logger_config import logger
from grasp.utils import constants


class BaseEdge(ABC):
    """
    Data structure representing an edge in the graph.
    Stores source and target nodes, as well as any condition and path_map information.

    As of now normal edge and conditional edge are handled with this class, but they can be redesigned based on the requirement.
    May need to support multiple target node (one to many) in future.
    """

    def __init__(self, source: Optional[BaseNode], target: Optional[BaseNode]):
        """
        Create a BaseEdge object using source and sink node.

        Args:
            source(BaseNode): source node object of type BaseNode
            target(BaseNode): target node object of type BaseNode
        """
        self.source = source
        self.target = target
        # Condition string or method for conditional edges
        self.condition: Optional[str] = None
        # Mapping from condition outcome to node name
        self.path_map: dict = {}
        # Original edge configuration dictionary
        self.edge_config: dict = {}

    def get_source(self) -> Optional[BaseNode]:
        """
        Get source node object.

        Returns:
            BaseNode: the source node object
        """
        return self.source

    def get_target(self) -> Optional[BaseNode]:
        """
        Get target or sink node object.

        Returns:
            BaseNode: the sink node object
        """
        return self.target

    def set_condition(self, condition: Optional[str]):
        """
        Set condition string which represent a class(new) or a method(old).

        This is only set for conditional edge having path map, but target in null

        Args:
            condition(str): condition string with full class path

        Returns:
            None
        """
        self.condition = condition

    def get_condition(self) -> Optional[str]:
        """
        Get condition string which represents full class path(new) or a method(old).

        Call this method only if the edge is conditional edge with path map, but target is null

        Returns:
            str: condition string with full class path or method path(old format)
        """
        return self.condition

    def set_path_map(self, path_map: dict):
        """
        Set path map, which represents return value of the condition to node name.

        Key of the path map is the possible return value
        Value of the path map is the node name including START/END

        Args:
            path_map(dict): dictionary of return values and corresponding node name

        Returns:
            None
        """
        self.path_map = path_map

    def get_path_map(self) -> dict:
        """
        Get path map, which represents return value of the condition to node name.

        Returns:
             dict: path map dictionary
        """
        return self.path_map

    def set_edge_config(self, edge_config: dict):
        """
        Set edge config dictionary which represent this edge.

        Args:
            edge_config(dict): dictionary representing the edge

        Returns:
            None
        """
        self.edge_config = edge_config

    def get_edge_config(self):
        """
        Get edge config dictionary which represents this edge.

        Returns:
            dict: edge config dictionary
        """
        return self.edge_config


class EdgeFactory:
    """
    EdgeFactory creates BaseEdge objects based on edge configurations and node objects.
    It resolves subgraph placeholders by using subgraph-specific entry and exit nodes.
    Additionally, it updates condition path_map values if they refer to subgraphs.
    """

    def __init__(
        self,
        edges_config: list[dict[str, Any]],
        nodes: dict[str, BaseNode],
        subgraphs: dict[str, Any],
    ):
        """
        Initialize the EdgeFactory.

        Args:
            edges_config: List of edge configuration dictionaries.
            nodes: Dictionary mapping node names to BaseNode objects.
            subgraphs: Dictionary mapping subgraph names to their GraphConfig objects.
        """
        self.edges = []  # Final list of BaseEdge objects
        self.subgraphs = subgraphs
        self.nodes = nodes

        # Process top-level (parent) edges
        for edge_cfg in edges_config:
            src_node_name: Optional[str] = edge_cfg.get("from")
            tgt_node_name: Optional[str] = edge_cfg.get("to")
            condition: Optional[str] = edge_cfg.get("condition")
            path_map: Optional[dict[Any, Any]] = edge_cfg.get("path_map")

            # Resolve source and target nodes.
            # For subgraph references, use the subgraph's exit for source and entry for target.
            src_node = self._resolve_source(src_node_name)
            tgt_node = self._resolve_target(tgt_node_name)

            # Update path_map values if they refer to subgraphs.
            if path_map:
                new_pm = {}
                for k, v in path_map.items():
                    new_pm[k] = self._resolve_mapping_value(v)
                path_map = new_pm

            self.update_edge_config(edge_cfg, src_node, tgt_node, condition, path_map)

            edge_obj = BaseEdge(src_node, tgt_node)
            edge_obj.set_condition(condition)
            edge_obj.set_path_map(path_map or {})
            edge_obj.set_edge_config(edge_cfg)
            self.edges.append(edge_obj)
            if tgt_node_name:
                logger.info(f"Edge created from {src_node_name} to {tgt_node_name}")
            elif path_map:
                logger.info(f"Edge created from {src_node_name} with path_map: {path_map}")

        # Process internal subgraph edges.
        # Instead of scanning the raw configuration, we assume subgraphs have already computed their edges.
        # We inline all edges from each subgraph except those with "from" equal to "START" or "to" equal to "END".
        for sg_name, sg in self.subgraphs.items():
            for e in sg.edges:
                # Skip subgraph placeholder edges.
                if (
                    e.edge_config.get("from") == constants.START
                    or e.edge_config.get("to") == constants.END
                ):
                    continue
                self.edges.append(e)
            logger.info(f"Subgraph {sg_name} edges inlined.")

    def update_edge_config(
        self,
        edge_cfg: dict,
        src_node: Optional[BaseNode],
        tgt_node: Optional[BaseNode],
        condition: Optional[str],
        path_map: Optional[dict],
    ):
        """
        Updates the edge configuration dictionary with resolved source and target nodes.

        Args:
            edge_cfg: The edge configuration dictionary to update.
            src_node: The resolved source BaseNode object.
            tgt_node: The resolved target BaseNode object.
            condition: The condition string for the edge.
            path_map: The path map dictionary for the edge.

        Returns:
            None
        """
        if not src_node:
            raise RuntimeError("Source node cannot be None for edge configuration.")
        edge_cfg["from"] = src_node.name
        edge_cfg["to"] = tgt_node.name if tgt_node else None
        if condition:
            edge_cfg["condition"] = condition
        if path_map:
            edge_cfg["path_map"] = path_map

    def get_edges(self) -> list[BaseEdge]:
        """
        Returns the final list of BaseEdge objects.
        """
        return self.edges

    def _resolve_source(self, node_name: Optional[str]) -> Optional[BaseNode]:
        """
        Resolves the source node for an edge.
        If the node is a subgraph placeholder, returns its exit node.

        Args:
            node_name: The node name or subgraph name.

        Returns:
            The corresponding BaseNode.
        """
        if node_name in self.subgraphs:
            subgraph = self.subgraphs[node_name]
            exit_node_name = self._find_subgraph_exit(subgraph)
            return self._get_node(node_name, exit_node_name)
        return self._get_node(node_name)

    def _resolve_target(self, node_name: Optional[str]) -> Optional[BaseNode]:
        """
        Resolves the target node for an edge.
        If the node is a subgraph placeholder, returns its entry node.

        Args:
            node_name: The node name or subgraph name.

        Returns:
            The corresponding BaseNode.
        """
        if node_name in self.subgraphs:
            subgraph = self.subgraphs[node_name]
            entry_node_name = self._find_subgraph_entry(subgraph)
            return self._get_node(node_name, entry_node_name)
        return self._get_node(node_name)

    def _find_subgraph_entry(self, subgraph) -> Any:
        """
        Finds the entry node of a subgraph by scanning for the edge with 'from' == "START".

        Args:
            subgraph: A GraphConfig instance representing the subgraph.

        Returns:
            The name of the entry node.
        """
        for edge in subgraph.graph_config["edges"]:
            if edge["from"] == constants.START:
                return edge["to"]
        raise RuntimeError(f"Subgraph {subgraph.parent_node} has no valid entry node.")

    def _find_subgraph_exit(self, subgraph) -> Any:
        """
        Finds the exit node of a subgraph by scanning for the edge with 'to' == "END".

        Args:
            subgraph: A GraphConfig instance representing the subgraph.

        Returns:
            The name of the exit node.
        """
        for edge in subgraph.graph_config["edges"]:
            if edge.get("to") == constants.END:
                return edge["from"]
        raise RuntimeError(f"Subgraph {subgraph.parent_node} has no valid exit node.")

    def _resolve_mapping_value(self, value: str) -> str:
        """
        Resolves a mapping value in a condition path_map.
        If the value corresponds to a subgraph, update it to the subgraph's entry node.

        Args:
            value: The original mapping value.

        Returns:
            The resolved mapping value.
        """
        if value in self.subgraphs:
            subgraph = self.subgraphs[value]
            entry_node = self._find_subgraph_entry(subgraph)
            return f"{entry_node}"
        return value

    def _get_node(
        self, node_name: Optional[str], sub_node: Optional[str] = None
    ) -> Optional[BaseNode]:
        """
        Retrieves the BaseNode instance for a given node name.
        If the node is a subgraph placeholder (stored as a dictionary),
        sub_node must be provided to fetch the correct internal node.

        Args:
            node_name: The node name or subgraph name.
            sub_node: The internal node name within the subgraph.

        Returns:
            The corresponding BaseNode object.
        """
        if node_name is None:
            return None
        # First, try to retrieve the node by its name.
        node = self.nodes.get(node_name)
        if node is None and sub_node is not None:
            node = self.nodes.get(sub_node)
        if node is None:
            if node_name in SpecialNode.SPECIAL_NODES:
                node = get_node(node_name, {"node_type": NodeType.SPECIAL.value})
            else:
                raise RuntimeError(
                    f"Node {node_name} not found in graph or as a special node in edge configuration."
                )
        if not node.is_valid():
            raise RuntimeError(f"Node {node_name} is idle, can't be used for edge creation.")
        return node
