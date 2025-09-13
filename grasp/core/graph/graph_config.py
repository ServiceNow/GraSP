from copy import deepcopy
from typing import Any, Optional, Union, cast

from datasets import Dataset, IterableDataset  # type: ignore[import-untyped]

from grasp.core.graph.edges.edge_factory import EdgeFactory
from grasp.core.graph.nodes import node_utils
from grasp.core.graph.nodes.base_node import BaseNode
from grasp.logger.logger_config import logger
from grasp.utils import constants, utils


class GraphConfig:
    """
    The purpose of this class is to store the graph configuration and initialize the nodes and edges.
    Processes the YAML configuration, including subgraph inlining, state variable extraction, and node overrides.
    """

    def __init__(
        self,
        config: Union[str, dict],
        dataset: Union[list[dict], IterableDataset],
        output_transform_args: dict,
        parent_node: str = "",
        override_config=None,  # New parameter for overrides
    ) -> None:
        """
        Initialize a GraphConfig.

        Args:
            config: YAML file path or configuration dictionary.
            dataset: Dataset used to extract state variables.
            parent_node: If part of a subgraph, the parent's name is used as a prefix.
            override_config: Additional node configuration overrides (default is empty for main graph).
        """
        if override_config is None:
            override_config = {}
        self.nodes: dict[str, BaseNode] = {}
        self.edges: list[Any] = []
        self.sub_graphs: dict[str, "GraphConfig"] = {}
        self.parent_node: str = parent_node
        self.state_variables: set = set()
        self.graph_state_config: dict[str, Any] = {}
        self.pattern_to_extract_variables = r"(?<!\{)\{([^{}]+)\}(?!\})"

        if isinstance(config, str):
            config = utils.load_yaml_file(filepath=config)

        # Ensure config is a dictionary after potential YAML load
        if not isinstance(config, dict):
            raise TypeError("config must be a dict after loading YAML or when provided directly")

        # Help mypy understand the type of config
        config = cast(dict[str, Any], config)

        if "graph_config" not in config:
            raise ValueError("graph_config key is required in configuration")

        config = self._validate_and_update_config(config, output_transform_args)

        self.config = config

        self.graph_config = deepcopy(config["graph_config"])
        self.schema_config = config.get("schema_config")
        # Normalize output_config to a dict to satisfy mypy and avoid Optional access
        output_cfg = config.get("output_config") or {}
        if not isinstance(output_cfg, dict):
            output_cfg = {}
        self.oasst_mapper = output_cfg.get("oasst_mapper")
        self.data_config = config.get("data_config")

        # Apply overrides before processing subgraphs
        self._apply_node_config_overrides(override_config)

        # Process subgraphs (load subgraph GraphConfig objects)
        self._process_subgraphs()

        # If this is a subgraph (i.e. part of a parent), ensure it has an explicit exit.
        if self.parent_node:
            self._ensure_connector_exit()

        # Initialize nodes and edges based on updated graph_config.
        self._initialize_graph()

        # Extract state variables from the dataset
        self._extract_input_data_keys(dataset)

        # Merge state variables from subgraphs.
        for subgraph in self.sub_graphs.values():
            self.state_variables.update(subgraph.state_variables)

        # Validate unique state variables.
        if len(self.state_variables) != len(set(self.state_variables)):
            raise ValueError(f"State variables are not unique: {self.state_variables}")

    def _validate_and_update_config(self, config: dict, transform_args: dict) -> dict:
        """
        Validates and updates the graph configuration based on the provided arguments.

        `oasst_mapper` and `data_quality` if already present in the config, it will take precedence over the transform_args.
        If `oasst_mapper` or `data_quality` is not present in the config, it will be added based on the transform_args.

        Args:
            config: The graph configuration dictionary.
            transform_args: The arguments containing oasst and quality args.
        """
        post_generation_tasks = utils.load_yaml_file(constants.GRASP_CONFIG).get(
            "post_generation_tasks", {}
        )

        # Ensure output_config exists and is a dict before writing to it
        output_cfg = config.setdefault("output_config", {})
        if not isinstance(output_cfg, dict):
            output_cfg = {}
            config["output_config"] = output_cfg

        # Check and update for oasst
        if transform_args.get("oasst", False) and "oasst_mapper" not in output_cfg:
            output_cfg["oasst_mapper"] = post_generation_tasks["oasst_mapper"]

        # Check and update for data_quality
        if transform_args.get("quality", False) and "data_quality" not in output_cfg:
            output_cfg["data_quality"] = post_generation_tasks["data_quality"]

        return config

    def _apply_node_config_overrides(self, override_config: dict[str, Any]) -> None:
        """
        Applies node-specific overrides before processing subgraphs.

        Args:
            override_config: Dictionary containing overrides for subgraph nodes.
        """
        for node_name, config_updates in override_config.items():
            if node_name in self.graph_config.get("nodes", {}):
                # Perform a deep merge of the base config with the overrides
                merged_config = self._recur_merge_dicts(
                    self.graph_config["nodes"][node_name], config_updates
                )
                # If there is a prompt_placeholder_map in the override, update the prompt accordingly.
                if "prompt_placeholder_map" in config_updates:
                    merged_config = self._apply_prompt_placeholder_map(
                        merged_config, config_updates["prompt_placeholder_map"]
                    )
                self.graph_config["nodes"][node_name] = merged_config

    def _recur_merge_dicts(self, base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively merges two dictionaries to preserve nested configurations.

        Args:
            base: Original dictionary.
            updates: Dictionary with override values.

        Returns:
            Merged dictionary with overrides applied.
        """
        merged = deepcopy(base)
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._recur_merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _apply_prompt_placeholder_map(
        self, node_config: dict[str, Any], placeholder_map: dict[str, str]
    ) -> dict[str, Any]:
        """
        Updates the placeholders in the prompt using the provided placeholder map.

        Args:
            node_config: The node configuration dictionary.
            placeholder_map: Mapping of old placeholder (without braces) to new placeholder (without braces).

        Returns:
            The updated node configuration with placeholders replaced in all prompt strings.
        """
        if "prompt" not in node_config:
            return node_config

        updated_prompts = []
        for prompt_entry in node_config["prompt"]:
            updated_entry = {}
            for role, text in prompt_entry.items():
                for old_key, new_key in placeholder_map.items():
                    # Replace occurrences of the placeholder in curly braces
                    text = text.replace(f"{{{old_key}}}", f"{{{new_key}}}")
                updated_entry[role] = text
            updated_prompts.append(updated_entry)
        node_config["prompt"] = updated_prompts
        return node_config

    def _process_subgraphs(self) -> None:
        """
        Processes subgraphs by identifying nodes of type 'subgraph'.
        Loads subgraph configurations and applies any node_config_map overrides.
        """
        updated_nodes = {}

        for node_name, node_config in deepcopy(self.graph_config.get("nodes", {})).items():
            if node_config.get("node_type") == "subgraph":
                # Extract the override configuration from node_config_map (if any)
                override_config = node_config.get("node_config_map", {})

                # Load the subgraph config and pass the overrides.
                subgraph_path = utils.get_file_in_task_dir(
                    node_config.get("subgraph"), "graph_config.yaml"
                )
                full_prefix = f"{self.parent_node}.{node_name}" if self.parent_node else node_name
                self.sub_graphs[full_prefix] = GraphConfig(
                    config=subgraph_path,
                    dataset=[],
                    parent_node=full_prefix,
                    output_transform_args={},
                    override_config=override_config,
                )

            # Update node names: if part of a parent subgraph, prepend parent's name.
            if self.parent_node:
                updated_nodes[f"{self.parent_node}.{node_name}"] = node_config
            else:
                updated_nodes[node_name] = node_config

        self.graph_config["nodes"] = updated_nodes
        self._update_edges_for_subgraph()

    def _update_edges_for_subgraph(self) -> None:
        """
        Updates edge definitions for nodes within a subgraph by applying the parent prefix.
        For each edge, update the 'from' and 'to' fields and the mapped values in path_map,
        leaving keys unchanged.
        """
        if not self.parent_node:
            return

        updated_edges = []
        for edge in deepcopy(self.graph_config.get("edges", [])):
            # Prefix the "from" field if it's not "START"
            edge["from"] = (
                f"{self.parent_node}.{edge['from']}" if edge["from"] != "START" else "START"
            )

            # Prefix the "to" field if present and not "END"
            if "to" in edge:
                edge["to"] = f"{self.parent_node}.{edge['to']}" if edge["to"] != "END" else "END"
            else:
                edge["to"] = None

            # Update the path_map values (only update values that are not START/END)
            if "path_map" in edge:
                edge["path_map"] = {
                    key: value if value in ["START", "END"] else f"{self.parent_node}.{value}"
                    for key, value in edge["path_map"].items()
                }
            updated_edges.append(edge)

        self.graph_config["edges"] = updated_edges

    def _ensure_connector_exit(self) -> None:
        """
        Ensures that the subgraph has a clear exit node.
        If no edge with "to" == "END" exists, or if there are multiple "to" == "END" edges, create a connector node.
        """
        end_nodes = [
            edge["from"] for edge in self.graph_config.get("edges", []) if edge["to"] == "END"
        ]
        if len(end_nodes) == 1:
            return

        connector_name = (
            f"{self.parent_node}.connector_node" if self.parent_node else "connector_node"
        )
        self.graph_config["nodes"][connector_name] = {"node_type": "connector"}

        if len(end_nodes) == 0:
            for edge in self.graph_config.get("edges", []):
                if "path_map" in edge:
                    new_pm = {}
                    for k, v in edge["path_map"].items():
                        new_value = connector_name if v == "END" else v
                        new_pm[k] = new_value
                    edge["path_map"] = new_pm
        else:
            for edge in self.graph_config.get("edges", []):
                if edge["to"] == "END":
                    edge["to"] = connector_name

        self.graph_config["edges"].append({"from": connector_name, "to": "END"})

    def _initialize_graph(self) -> None:
        """
        Initializes the graph nodes and edges.
        Creates node objects and collects state variables.
        """
        self.graph_state_config = self.graph_config.get("graph_state", {})

        for node_name, node_config in self.graph_config.get("nodes", {}).items():
            if node_name in self.sub_graphs:
                self.nodes.update(self.sub_graphs[node_name].get_nodes())
            else:
                self.nodes[node_name] = node_utils.get_node(node_name, node_config)
            logger.info(f"Node {node_name} initialized")
            self._process_variables(node_config, [constants.GRAPH_OUTPUT_KEY])
            # This extracts variables form prompt, which is mostly dataset variables
            # commenting it now, but can be re-added in future, if we find any other scenario
            self._extract_variables_from_prompts(node_config)

        self._initialize_edges()

    def _initialize_edges(self) -> None:
        """
        Creates edge objects from the configuration using the EdgeFactory.
        """
        edge_config = self.graph_config.get("edges", [])
        self.edges = EdgeFactory(edge_config, self.nodes, self.sub_graphs).get_edges()

    def _process_variables(self, node_config: dict[str, Any], keys: list[str]) -> None:
        """
        Extracts and validates state variables from a node configuration.

        Args:
            node_config: Node configuration dictionary.
            keys: List of keys to extract state variables from.
        """
        for key in keys:
            if key not in node_config:
                continue

            variable = node_config[key]
            if isinstance(variable, str):
                if variable in self.state_variables:
                    logger.warning(f"Duplicate output variable: {variable}")
                    continue
                self.state_variables.add(variable)
            elif isinstance(variable, list) and all(isinstance(var, str) for var in variable):
                if any(var in self.state_variables for var in variable):
                    raise ValueError(f"Duplicate output variable in list: {variable}")
                self.state_variables.update(variable)
            else:
                raise ValueError(f"Invalid variable format: {variable}")

    def _extract_variables_from_prompts(self, node_config: dict[str, Any]) -> None:
        """
        Extracts variables from the prompt templates in the node configuration.

        Args:
            node_config: Node configuration dictionary.
        """
        if "prompt" not in node_config:
            return

        for prompt in node_config["prompt"]:
            for content in prompt.values():
                if isinstance(content, str):
                    self.state_variables.update(
                        utils.extract_pattern(content, self.pattern_to_extract_variables)
                    )
                elif isinstance(content, list):
                    for c in content:
                        if isinstance(c, str):
                            self.state_variables.update(
                                utils.extract_pattern(c, self.pattern_to_extract_variables)
                            )
                        elif isinstance(c, dict):
                            for sub_content in c.values():
                                if isinstance(sub_content, str):
                                    self.state_variables.update(
                                        utils.extract_pattern(
                                            sub_content,
                                            self.pattern_to_extract_variables,
                                        )
                                    )
                                else:
                                    raise ValueError(f"Invalid prompt content: {sub_content}")
                else:
                    raise ValueError(f"Invalid prompt content: {content}")

    def _extract_input_data_keys(self, data: Union[list[dict], Dataset, IterableDataset]) -> None:
        """
        Extracts keys from the dataset to be used as state variables.

        Args:
            data: Dataset, list of dicts, or IterableDataset.
        """
        try:
            if isinstance(data, list) and data:
                self.state_variables.update(data[0].keys())
            elif isinstance(data, IterableDataset):
                peek_record = list(data.take(1))[0]
                self.state_variables.update(peek_record.keys())
            elif hasattr(data, "column_names"):
                self.state_variables.update(data.column_names)
        except Exception as e:
            raise ValueError(f"Error extracting keys from dataset: {e}")

    def get_nodes(self) -> dict[str, BaseNode]:
        """
        Returns all initialized nodes.

        Returns:
            Dictionary mapping node names to BaseNode objects.
        """
        return self.nodes

    def get_edges(self) -> list:
        """
        Returns all initialized edges.

        Returns:
            List of edge objects.
        """
        return self.edges

    def get_graph_state_config(self) -> dict:
        """
        Returns the graph state configuration.

        Returns:
            Dictionary containing graph state configuration.
        """
        return self.graph_state_config

    def get_node(self, node_name: str) -> Optional[BaseNode]:
        """
        Retrieves a node by its name.

        Args:
            node_name: The name of the node.

        Returns:
            The corresponding BaseNode object.
        """
        return self.nodes.get(node_name)

    def get_node_config(self, node_name: str) -> dict[str, Any]:
        """
        Retrieves the configuration dictionary for a specific node.

        Args:
            node_name: The name of the node.

        Returns:
            The node configuration dictionary.
        """
        return node_utils.get_node_config(node_name, self.graph_config)

    def get_subgraph_nodes(self, node_name: str) -> dict[str, BaseNode]:
        """
        Retrieves nodes for a given subgraph.

        Args:
            node_name: The subgraph's name.

        Returns:
            Dictionary of nodes within the subgraph.
        """
        if node_name not in self.sub_graphs:
            raise ValueError(f"Subgraph {node_name} not found.")
        return self.sub_graphs[node_name].get_nodes()
