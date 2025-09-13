from inspect import isclass
from typing import Any, Optional

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from grasp.core.graph.edges.edge_factory import BaseEdge, EdgeFactory
from grasp.core.graph.graph_config import GraphConfig
from grasp.core.graph.nodes.base_node import BaseNode
from grasp.logger.logger_config import logger
from grasp.utils.utils import backend_factory, get_func_from_str


class LangGraphBuilder:
    """
    LangGraph specific builder class to create StateGraph workflow object
    In future, if we want to use any other library, create interface for this and use in BaseTaskExecutor
    """

    # LangGraph specific string to Start/End point translation map
    SPECIAL_NODES_MAP = {
        "START": START,
        "END": END,
    }

    def __init__(self, graph_config: GraphConfig):
        """
        LangGraphBuilder constructor.

        Args:
            graph_config(GraphConfig): GraphConfig object created out of graph yaml file.
        """
        self.graph_config = graph_config
        self.workflow = None

    # this should be overriden from base abstract method in future, so keep return type as Any
    def build(self) -> Any:
        """
        Build the LangGraph's StateGraph workflow object from GraphConfig object.

        Returns:
            updated workflow object.
        """

        # get all state variables from nodes and inject into self.graph_config
        self._update_state_variables()

        workflow = backend_factory.build_workflow(self.graph_config)

        self.add_nodes(workflow)
        self.add_edges(workflow)
        # self.set_entry_point(workflow)

        self.workflow = workflow
        return workflow

    def _update_state_variables(self):
        """
        Add state variables from node to graph_config so that we can build the State Graph(workflow).

        Returns:
            None
        """
        nodes = self.graph_config.get_nodes()
        for node_name, node in nodes.items():
            if node.is_active():
                # add the node state variables, if they dont exists in the state var list
                for var in node.get_state_variables():
                    if var not in self.graph_config.state_variables:
                        self.graph_config.state_variables.add(var)

    def add_nodes(self, workflow: StateGraph):
        """
        Add nodes into workflow graph.

        Args:
            workflow: nodes to be added into this workflow

        Returns:
            None
        """
        nodes = self.graph_config.get_nodes()
        for node_name, node in nodes.items():
            if node.is_active():
                # add the node runnable(lang graph object) into workflow
                workflow.add_node(node_name, node.to_backend())
        logger.info("Completed adding nodes into the workflow graph.")

    def convert_to_graph(self, node: Optional[BaseNode]):
        """
        Convert Special node to START or END object.

        Args:
            node(BaseNode): SpecialNode object

        Returns:
            BaseNode|START|END: Returns START or END object if they are Start or End. Else returns node name.
        """
        if node is None:
            return None
        name = node.get_name()
        return self.SPECIAL_NODES_MAP[name] if node.is_special_type() else name

    def add_edges(self, workflow: StateGraph):
        """
        Add edges into workflow graph.

        Args:
            workflow(StateGraph): StateGraph workflow object.

        Returns:
             None
        """
        errors = []
        # if edges are already built, just assign the edges list
        # else create from edge factory
        edges = self.graph_config.get_edges()
        if (
            edges is not None
            and len(edges) > 0
            and isinstance(self.graph_config.get_edges()[0], BaseEdge)
        ):
            edges = self.graph_config.get_edges()
        else:
            edges = EdgeFactory(
                edges, self.graph_config.get_nodes(), self.graph_config.sub_graphs
            ).get_edges()

        for edge in edges:
            source = edge.get_source()
            target = edge.get_target()
            condition = edge.get_condition()
            path_map = edge.get_path_map()

            if source is not None and not source.is_valid():
                errors.append(f"Invalid source node: {source}")

            if condition and path_map:
                # convert the condition to method object
                condition = get_func_from_str(condition)
                if isclass(condition):
                    condition = condition.apply

                # START/END string mapping to START/END langgraph object in path_map
                path_map = {
                    self.get_node(key): self.get_node(value) for key, value in path_map.items()
                }

                workflow.add_conditional_edges(self.convert_to_graph(source), condition, path_map)
                logger.info(
                    f"Completed adding conditional edge into the workflow graph. Source node: {self.convert_to_graph(source)}"
                )
            elif target:
                workflow.add_edge(self.convert_to_graph(source), self.convert_to_graph(target))
                logger.info(
                    "Completed adding simple edge into the workflow graph. "
                    + f"Source node: {self.convert_to_graph(source)}, Destination node: {self.convert_to_graph(target)}"
                )
            else:
                errors.append(f"Invalid edge configuration: {edge.get_edge_config()}")

        if errors:
            raise ValueError("Errors in edge configurations:\n" + "\n".join(errors))

    # use START node to connect, instead this API
    # @deprecated("Create an edge from START to first node")
    """
    def set_entry_point(self, workflow: StateGraph):
        entry_point = self.graph_config.get_entry_point()
        workflow.set_entry_point(entry_point)
    """

    @staticmethod
    def get_node(value: str) -> str:
        return LangGraphBuilder.SPECIAL_NODES_MAP.get(value, value)

    def compile(self) -> CompiledStateGraph:
        """
        Compile the raw StateGraph to Compiled Graph.

        Returns:
            CompiledStateGraph: compiled graph object
        """
        if self.workflow is None:
            raise RuntimeError("The build() method must be called before compiling the graph.")
        return self.workflow.compile()
