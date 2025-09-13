from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional

from grasp.utils import utils


class NodeType(Enum):
    """
    Types of Node Supported in GraSP.

    """

    LLM = "llm"
    MULTI_LLM = "multi_llm"
    AGENT = "agent"
    LAMBDA = "lambda"
    WEIGHTED_SAMPLER = "weighted_sampler"
    UNKNOWN = "unknown"
    SPECIAL = "special"
    CONNECTOR = "connector"


class NodeState(Enum):
    """
    by default node is ACTIVE, but it can be disabled with idle key.

    """

    ACTIVE = "active"
    IDLE = "idle"


class BaseNode(ABC):
    """
    Node structure holds the node configuration, GraSP uses this Node in the platform.

    """

    def __init__(self, name, config: Optional[dict[str, Any]] = None):
        """
        Node constructor to build the Node object
        :param name: node name
        :param config: node configuration
        """
        # name of the node defined in yaml file
        self.name: str = name
        self.node_type: str
        self.node_state: str

        if self.name is None:
            raise ValueError("Node name is required")

        if config is not None:
            # node type
            self.node_type = config.get("node_type", NodeType.UNKNOWN.value)
            # node state
            self.node_state = config.get("node_state", NodeState.ACTIVE.value)
            # store the node config from graph yaml file for this specific node
            self.node_config = config
            if self.node_config is None:
                raise ValueError("Node configuration is required.")
        else:
            self.node_type = NodeType.UNKNOWN.value
            self.node_state = NodeState.IDLE.value
            # Ensure node_config is always a dict to simplify downstream typing
            self.node_config = {}
        # stores node specific state variables, this should be passed to langgraph
        self.state_variables: List[str] = []

        # throws ValueError
        self.validate_node()

    def get_node_state(self) -> str:
        """
        Get the node_state.

        Which can be "active" by default or "idle" if not used int the graph.
        User might disable a node during experiment, by setting "node_state" to "idle" in the YAML file.

        Returns:
            NodeState: state of the node
        """
        return self.node_state

    def get_node_type(self) -> str:
        """
        Get the node_type.

        Which can be llm, lambda, multi_llm, special, weighted_sampler, or unknown.
        llm type node talks to single model and response is assigned to output after postprocessing.
        multi_llm type node talks to multiple models.
        weighted_sampler type node is used to select random sample data from the static list of samples.
        special type of node is used to denote START or END node.

        Returns:
            NodeType: type of the node
        """
        return self.node_type

    def is_special_type(self) -> bool:
        """
        Checks if the node is special type.

        Returns:
            bool: True if the node is special type.
        """
        return bool(self.node_type == NodeType.SPECIAL.value)

    def get_node_config(self) -> dict[str, Any]:
        """
        Get the node configuration as dictionary type.

        This is the raw data defined in the YAML file.
        Returns:
             dict: node configuration as defined in the YAML file.
        """
        return self.node_config

    def get_name(self) -> str:
        """
        Get the name of the node, which is the key name given for each node in the nodes definition.

        Returns:
             str: name of the node as defined in the YAML file.
        """
        return self.name

    def get_state_variables(self) -> list[str]:
        """
        If there are variables needs to be injected from node level to graph state, this variable is used.

        Before the graph is built, these variables has to be fed into the GraphConfig object.

        Returns:
            list[str]: list of variable names which will be injected into graph as state variables.
        """
        return self.state_variables

    def is_active(self) -> bool:
        """
        Checks if the node is active.

        Returns:
             bool: True if the node is active.
        """
        return self.get_node_state() != NodeState.IDLE.value

    def is_valid(self) -> bool:
        """
        Checks if the node is valid.
        This method differentiate between a valid node and an active node.
        Like we can have nodes used for other purpose, which are not valid.

        Returns:
             bool: True if the node is valid.
        """
        return self.is_active()

    def validate_node(self):
        """
        Validates the node property keys and other behavior.

        It throws Exception.
        Returns:
            None
        """
        pass

    def validate_config_keys(self, required_keys: list[str], config_name: str, node_config: dict):
        utils.validate_required_keys(required_keys, node_config, config_name)

    @abstractmethod
    def to_backend(self) -> Any:
        """
        Implement get runnable object specific to backend platform like LangGraph.

        It converts Node to platform(LangGraph) specific object
        Returns:
            Any: runnable node specific to the backend platform
        """
        pass
