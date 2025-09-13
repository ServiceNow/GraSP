from inspect import isclass
from typing import Any

from grasp.core.graph.nodes.base_node import BaseNode
from grasp.utils import utils


class LambdaNode(BaseNode):
    REQUIRED_KEYS: list[str] = ["lambda"]

    def __init__(self, node_name: str, config: dict):
        """
        LambdaNode constructor.

        Args:
            node_name: Name of the node, defined as key under "nodes" in the YAML file.
            config: Node configuration defined in YAML file as the node value.
        """
        super().__init__(node_name, config)

        # specific variable for this node type
        self.func_to_execute = utils.get_func_from_str(self.node_config["lambda"])
        if isclass(self.func_to_execute):
            self.func_to_execute = self.func_to_execute.apply

    def to_backend(self) -> Any:
        """
        Convert the Node object to backend platform specific Runnable object.

        Returns:
             Any: platform specific runnable object like Runnable in LangGraph.
        """
        return utils.backend_factory.create_lambda_runnable(self.func_to_execute, self.node_config)

    def validate_node(self):
        """
        Override the method to add required validation for this Node type.
        It throws Exception.

        Returns:
            None
        """
        self.validate_config_keys(self.REQUIRED_KEYS, self.node_type, self.node_config)
