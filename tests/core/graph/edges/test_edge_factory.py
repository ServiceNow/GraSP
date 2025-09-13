import os
import sys

# Add project root to sys.path for relative imports to work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from types import SimpleNamespace

import pytest

from grasp.core.graph.edges.edge_factory import BaseEdge, EdgeFactory
from grasp.core.graph.nodes.base_node import BaseNode, NodeType
from grasp.core.graph.nodes.special_node import SpecialNode


class DummyNode(BaseNode):
    def __init__(self, name):
        super().__init__(name=name, config={"node_type": NodeType.LAMBDA})
        self._valid = True

    def is_valid(self):
        return self._valid

    def to_backend(self):
        return {}  # required abstract method


@pytest.fixture
def dummy_nodes():
    return {
        "node_a": DummyNode("node_a"),
        "node_b": DummyNode("node_b"),
        "subgraph.entry": DummyNode("subgraph.entry"),
        "subgraph.exit": DummyNode("subgraph.exit"),
    }


@pytest.fixture
def dummy_subgraph():
    graph_config = {
        "edges": [
            {"from": "START", "to": "subgraph.entry"},
            {"from": "subgraph.entry", "to": "subgraph.exit"},
            {"from": "subgraph.exit", "to": "END"},
        ]
    }
    return SimpleNamespace(
        graph_config=graph_config,
        parent_node="subgraph",
        edges=[],
    )


# -------------------- Tests --------------------


def test_base_edge_properties(dummy_nodes):
    """
    Unit test for verifying the correct behavior of BaseEdge class methods.

    This test checks that the BaseEdge object correctly stores and returns:
    - the source and target node references,
    - a condition string used for conditional edge evaluation,
    - a path map dictionary mapping condition outcomes to target node names,
    - and the original edge configuration dictionary.

    The test uses two dummy nodes ('node_a' and 'node_b') to simulate a basic edge and
    ensures that all internal setters and getters of BaseEdge function as expected.

    Assertions:
        - The source node name should be 'node_a'.
        - The target node name should be 'node_b'.
        - The condition should match the test string.
        - The path map should be stored and returned accurately.
        - The edge configuration should match the assigned dictionary.
    """
    edge = BaseEdge(dummy_nodes["node_a"], dummy_nodes["node_b"])
    edge.set_condition("some.condition.func")
    edge.set_path_map({"yes": "node_b"})
    edge.set_edge_config({"from": "node_a", "to": "node_b"})

    assert edge.get_source().name == "node_a"
    assert edge.get_target().name == "node_b"
    assert edge.get_condition() == "some.condition.func"
    assert edge.get_path_map() == {"yes": "node_b"}
    assert edge.get_edge_config() == {"from": "node_a", "to": "node_b"}


def test_edge_factory_creates_direct_edge(dummy_nodes):
    """
    Test that EdgeFactory correctly creates a direct edge between two regular nodes.

    This verifies the most basic usage of the EdgeFactory, where no subgraphs or conditions are involved.
    The created edge should have:
    - source: 'node_a'
    - target: 'node_b'

    Assertions:
        - Only one edge is created.
        - The edge source is 'node_a'.
        - The edge target is 'node_b'.
    """
    edges_config = [{"from": "node_a", "to": "node_b"}]
    factory = EdgeFactory(edges_config, dummy_nodes, subgraphs={})
    edges = factory.get_edges()

    assert len(edges) == 1
    assert edges[0].get_source().name == "node_a"
    assert edges[0].get_target().name == "node_b"


def test_edge_factory_with_subgraph_resolution(dummy_nodes, dummy_subgraph):
    """
    Test that a target node referencing a subgraph is resolved to the subgraph's entry node.

    This ensures the EdgeFactory identifies subgraph targets and replaces them internally with
    the correct entry node for correct edge execution behavior.

    Setup:
        - 'subgraph' is a placeholder node representing a subgraph.
        - 'subgraph.entry' is the actual entry node inside the subgraph.

    Assertions:
        - The target node of the created edge should resolve to 'subgraph.entry'.
    """
    dummy_nodes["subgraph"] = DummyNode("subgraph")
    dummy_nodes["subgraph.entry"] = DummyNode("subgraph.entry")
    dummy_nodes["subgraph.exit"] = DummyNode("subgraph.exit")

    edges_config = [{"from": "node_a", "to": "subgraph"}]
    factory = EdgeFactory(edges_config, dummy_nodes, subgraphs={"subgraph": dummy_subgraph})
    edges = factory.get_edges()

    assert len(edges) == 1
    assert edges[0].get_target().name == "subgraph"


def test_edge_factory_path_map_resolution(dummy_nodes, dummy_subgraph):
    """
    Test that path_map entries referencing subgraphs are correctly resolved to subgraph entry nodes.

    This ensures that conditional edges with subgraph targets in their path_map are internally
    resolved to point to the correct subgraph entry nodes.

    Setup:
        - 'loop' path points to the subgraph (should resolve to subgraph.entry)
        - 'done' path points to node_b (regular node)

    Assertions:
        - path_map['loop'] resolves to 'subgraph.entry'
        - path_map['done'] remains as 'node_b'
        - The condition string is preserved.
    """
    dummy_nodes["subgraph"] = DummyNode("subgraph")
    dummy_nodes["subgraph.entry"] = DummyNode("subgraph.entry")
    dummy_nodes["subgraph.exit"] = DummyNode("subgraph.exit")

    edges_config = [
        {
            "from": "subgraph",
            "condition": "some.condition.check",
            "path_map": {"loop": "subgraph", "done": "node_b"},
        }
    ]

    factory = EdgeFactory(edges_config, dummy_nodes, subgraphs={"subgraph": dummy_subgraph})
    edge = factory.get_edges()[0]

    assert edge.get_condition() == "some.condition.check"
    assert edge.get_path_map()["loop"] == "subgraph.entry"
    assert edge.get_path_map()["done"] == "node_b"


def test_edge_factory_raises_on_invalid_node():
    """
    Test that a RuntimeError is raised when a node referenced in the edge config does not exist.

    This validates EdgeFactory's ability to catch incorrect node references during edge creation.

    Setup:
        - Edge configuration refers to a non-existent node 'non_existent'.

    Expectation:
        - A RuntimeError is raised with a message indicating the missing node.
    """
    edges_config = [{"from": "non_existent", "to": "another"}]
    with pytest.raises(RuntimeError, match="Node non_existent not found"):
        EdgeFactory(edges_config, nodes={}, subgraphs={})


def test_resolve_subgraph_entry_and_exit(dummy_subgraph):
    """
    Test that EdgeFactory correctly identifies the entry and exit nodes of a subgraph.

    This ensures that subgraph traversal uses the correct internal starting and ending nodes.

    Setup:
        - dummy_subgraph has edges START -> subgraph.entry and subgraph.exit -> END

    Assertions:
        - Entry node resolved as 'subgraph.entry'
        - Exit node resolved as 'subgraph.exit'
    """
    factory = EdgeFactory([], {}, subgraphs={})
    assert factory._find_subgraph_entry(dummy_subgraph) == "subgraph.entry"
    assert factory._find_subgraph_exit(dummy_subgraph) == "subgraph.exit"


def test_invalid_subgraph_entry_raises():
    """
    Test that EdgeFactory raises a RuntimeError when a subgraph lacks a valid START->entry edge.

    This validates the error handling for malformed or incomplete subgraph configurations.

    Setup:
        - subgraph has no edge starting from START.

    Expectation:
        - RuntimeError indicating the subgraph has no valid entry node.
    """
    subgraph = SimpleNamespace(
        graph_config={"edges": [{"from": "x", "to": "y"}]},
        parent_node="faulty_subgraph",
        edges=[],
    )
    factory = EdgeFactory([], {}, {})
    with pytest.raises(RuntimeError, match="has no valid entry"):
        factory._find_subgraph_entry(subgraph)


def test_invalid_subgraph_exit_raises():
    """
    Test that EdgeFactory raises a RuntimeError when a subgraph lacks a valid exit->END edge.

    This test ensures graceful failure when no proper terminal node exists in the subgraph.

    Setup:
        - subgraph has no edge pointing to END.

    Expectation:
        - RuntimeError indicating the subgraph has no valid exit node.
    """
    subgraph = SimpleNamespace(
        graph_config={"edges": [{"from": "a", "to": "b"}]},
        parent_node="bad_subgraph",
        edges=[],
    )
    factory = EdgeFactory([], {}, {})
    with pytest.raises(RuntimeError, match="has no valid exit"):
        factory._find_subgraph_exit(subgraph)


def test_conditional_edge_without_target(dummy_nodes):
    """
    Test creation of a conditional edge with no direct target, relying solely on a path_map.

    This simulates the case where the flow is determined dynamically by a condition function,
    and each possible outcome leads to a different node via the path_map.

    Assertions:
        - The edge has a source but no direct target.
        - The path_map correctly maps the condition outcome to the node name.
    """
    edge_config = {
        "from": "node_a",
        "condition": "some.condition.func",
        "path_map": {"yes": "node_b"},
    }
    factory = EdgeFactory([edge_config], dummy_nodes, subgraphs={})
    edge = factory.get_edges()[0]

    assert edge.get_source().name == "node_a"
    assert edge.get_target() is None
    assert edge.get_path_map()["yes"] == "node_b"


def test_node_marked_invalid_raises(dummy_nodes):
    """
    Test that EdgeFactory raises an error when trying to use an invalid (idle) node.

    Simulates a scenario where a node is marked as unusable (e.g., disabled in the graph editor),
    and ensures such nodes are not silently accepted in edge construction.

    Expectation:
        - RuntimeError is raised with a message indicating the node is idle.
    """
    dummy_nodes["node_a"]._valid = False
    with pytest.raises(RuntimeError, match="idle"):
        EdgeFactory([{"from": "node_a", "to": "node_b"}], dummy_nodes, {})


def test_empty_edge_config(dummy_nodes):
    """
    Test that EdgeFactory handles an empty edge configuration list gracefully.

    Ensures that the factory does not error or produce any edges when no configs are provided.

    Assertion:
        - The list of generated edges is empty.
    """
    factory = EdgeFactory([], dummy_nodes, {})
    assert factory.get_edges() == []


def test_target_and_path_map_both_subgraph(dummy_nodes, dummy_subgraph):
    """
    Test edge creation where both the direct target and path_map values refer to a subgraph.

    Verifies that:
    - The direct target resolves to the subgraph’s entry node.
    - The path_map entries referring to the same subgraph are also resolved to its entry.

    Assertions:
        - edge.get_target() is resolved to the subgraph’s entry node.
        - path_map "yes" → subgraph.entry, "no" → node_b
    """
    dummy_nodes["subgraph"] = DummyNode("subgraph")
    dummy_nodes["subgraph.entry"] = DummyNode("subgraph.entry")
    dummy_nodes["subgraph.exit"] = DummyNode("subgraph.exit")

    edge_config = {
        "from": "node_a",
        "to": "subgraph",
        "path_map": {"yes": "subgraph", "no": "node_b"},
        "condition": "check.something",
    }
    factory = EdgeFactory([edge_config], dummy_nodes, {"subgraph": dummy_subgraph})
    edge = factory.get_edges()[0]

    assert edge.get_target().name == "subgraph"
    assert edge.get_path_map()["yes"] == "subgraph.entry"
    assert edge.get_path_map()["no"] == "node_b"


def test_special_node_resolution(monkeypatch):
    """
    Test fallback logic for resolving special nodes like 'START' or 'END' using SpecialNode.

    This verifies that if a node name is not found in the nodes dictionary but is considered
    special (e.g., 'END'), the EdgeFactory can still construct a valid BaseNode for it using
    the special node mechanism.

    Assertions:
        - The resolved node is a valid BaseNode instance.
        - The node's name matches the special identifier ('END').
    """
    monkeypatch.setattr(SpecialNode, "SPECIAL_NODES", ["START", "END"])

    factory = EdgeFactory([], {}, {})
    node = factory._get_node("END")

    assert isinstance(node, BaseNode)
    assert node.name == "END"


def test_condition_without_path_map(dummy_nodes):
    """
    Test that an edge with a condition but no path_map is handled correctly.

    This simulates a legacy or simplified conditional edge that only stores a condition,
    without defining branching logic.

    Assertions:
        - The condition is set correctly.
        - The path_map is initialized as an empty dictionary.
    """
    edge_config = {
        "from": "node_a",
        "condition": "some.condition.func",
    }
    factory = EdgeFactory([edge_config], dummy_nodes, {})
    edge = factory.get_edges()[0]

    assert edge.get_condition() == "some.condition.func"
    assert edge.get_path_map() == {}


def test_start_end_as_normal_nodes(dummy_nodes):
    """
    Test that 'START' and 'END' can be treated as regular node names if defined in the node dictionary.

    Normally these are reserved keywords, but users may define them explicitly as functional nodes.

    Assertions:
        - Edges correctly resolve 'START' and 'END' from the node dictionary, not as special nodes.
    """
    dummy_nodes["START"] = DummyNode("START")
    dummy_nodes["END"] = DummyNode("END")

    edge_config = [{"from": "START", "to": "END"}]
    factory = EdgeFactory(edge_config, dummy_nodes, {})

    assert factory.get_edges()[0].get_source().name == "START"
    assert factory.get_edges()[0].get_target().name == "END"


def test_none_target_with_path_map(dummy_nodes):
    """
    Test that an edge with a None target but a valid path_map is handled properly.

    This is common in conditional branches where the actual flow is determined by evaluating
    the path_map based on runtime conditions.

    Assertions:
        - The edge target is None (intentionally).
        - The path_map remains valid and resolvable.
    """
    edge_config = {
        "from": "node_a",
        "to": None,
        "condition": "condition.path",
        "path_map": {"yes": "node_b"},
    }
    factory = EdgeFactory([edge_config], dummy_nodes, {})
    edge = factory.get_edges()[0]

    assert edge.get_target() is None


def test_node_vs_subgraph_name_conflict(dummy_nodes):
    """
    Ensures node-subgraph name conflicts prioritize node resolution.

    If both a node and a subgraph share the same name, this test ensures that
    the `EdgeFactory` resolves to the node (from the `nodes` dictionary), not
    mistakenly as a subgraph placeholder.

    Assertions:
        - The source node in the created edge should be the actual node named "conflict",
          not the subgraph's exit node.
    """
    dummy_nodes["conflict"] = DummyNode("conflict")
    dummy_nodes["node_b"] = DummyNode("node_b")

    subgraph = SimpleNamespace(
        graph_config={
            "edges": [
                {"from": "START", "to": "subgraph.entry"},
                {"from": "subgraph.entry", "to": "subgraph.exit"},
                {"from": "subgraph.exit", "to": "END"},
            ]
        },
        parent_node="conflict",
        edges=[],
    )

    edge_config = [{"from": "conflict", "to": "node_b"}]
    factory = EdgeFactory(edge_config, dummy_nodes, {"conflict": subgraph})
    edge = factory.get_edges()[0]

    assert edge.get_source().name == "conflict"


def test_multiple_edges_from_same_node(dummy_nodes, dummy_subgraph):
    """
    Verifies that the EdgeFactory correctly handles multiple outgoing edges from the same node.

    This test includes one direct edge and one edge to a subgraph from the same source node,
    ensuring that multiple paths are properly instantiated.

    Assertions:
        - The factory creates two edges.
        - Both edges originate from "node_a".
    """
    dummy_nodes["subgraph"] = DummyNode("subgraph")
    dummy_nodes["subgraph.entry"] = DummyNode("subgraph.entry")
    dummy_nodes["subgraph.exit"] = DummyNode("subgraph.exit")

    edges_config = [
        {"from": "node_a", "to": "node_b"},
        {"from": "node_a", "to": "subgraph"},
    ]

    factory = EdgeFactory(edges_config, dummy_nodes, {"subgraph": dummy_subgraph})
    assert len(factory.get_edges()) == 2


def test_empty_subgraph_is_ignored(dummy_nodes):
    """
    Ensures that subgraphs with no internal inlinable edges do not break or contribute extra edges.

    This test validates behavior when a subgraph contains only the START → entry edge and nothing else.
    Only the parent edge using the subgraph should be created.

    Assertions:
        - Only one edge exists, linking the parent node to the subgraph.
        - No internal subgraph edges are inlined into the main graph.
    """
    dummy_nodes["subgraph"] = DummyNode("subgraph")
    dummy_nodes["subgraph.entry"] = DummyNode("subgraph.entry")
    dummy_nodes["subgraph.exit"] = DummyNode("subgraph.exit")

    empty_subgraph = SimpleNamespace(
        graph_config={"edges": [{"from": "START", "to": "subgraph.entry"}]},
        parent_node="subgraph",
        edges=[],
    )

    edge_config = {"from": "node_a", "to": "subgraph"}
    factory = EdgeFactory([edge_config], dummy_nodes, {"subgraph": empty_subgraph})

    assert len(factory.get_edges()) == 1


def test_path_map_invalid_target_raises(dummy_nodes):
    """
    Validates that path_map resolution raises an error if the target node does not exist.

    This ensures internal path_map logic correctly detects missing nodes
    and fails fast, preventing malformed graph execution.

    Assertions:
        - Calling _get_node on an invalid path_map target raises RuntimeError.
    """

    factory = EdgeFactory([], dummy_nodes, {})
    with pytest.raises(RuntimeError, match="not found"):
        factory._get_node("non_existent")
