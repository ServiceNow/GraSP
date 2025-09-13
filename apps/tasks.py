import os
import yaml
from pathlib import Path

try:
    import streamlit as st
    from streamlit_flow import streamlit_flow
    from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
    from streamlit_flow.state import StreamlitFlowState
    from streamlit_flow.layouts import (
        TreeLayout,
        RadialLayout,
        LayeredLayout,
        ForceLayout,
        StressLayout,
        RandomLayout,
    )
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "GraSP UI requires the optional 'ui' dependencies. "
        "Install them with: pip install 'grasp[ui]'"
    )

TASKS_DIR = Path("tasks")


class GraphConfigVisualizer:
    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        self.nodes = {}
        self._load_yaml()

    def _load_yaml(self):
        try:
            with open(self.yaml_path, "r") as f:
                self.config = yaml.safe_load(f)
            self.nodes = self.config.get("graph_config", {}).get("nodes", {})
        except Exception as e:
            self.config = {}
            self.nodes = {}

    def get_node_ids(self):
        return list(self.nodes.keys())

    def get_node_details(self, node_id):
        raw = self.nodes.get(node_id, {})

        return {
            "node_type": raw.get("node_type"),
            "output_key": raw.get("output_key"),
            "pre_process": raw.get("pre_process"),
            "post_process": raw.get("post_process"),
            "model": raw.get("model"),
            "attributes": raw.get("attributes") or raw.get("sampling_attributes"),
            "prompts": raw.get("prompt", []),
            "raw": raw,
        }


def parse_graph_config_from_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
        graph_config = config.get("graph_config", {})

    # Extract nodes with specific details
    nodes = graph_config.get("nodes", {})

    node_list = []
    for node_id, node_data in nodes.items():
        node_info = {
            "name": node_id,
            "node_type": node_data.get("node_type"),
            "model_name": node_data.get("model", {}).get("name")
            if node_data.get("model")
            else None,
            "pre_process": node_data.get("pre_process"),
            "post_process": node_data.get("post_process"),
        }
        node_list.append(node_info)

    # Extract edges (same as before)
    raw_edges = graph_config.get("edges", [])
    edge_list = []
    for edge in raw_edges:
        from_node = edge["from"]
        to_node = edge.get("to")
        condition = edge.get("condition")
        path_map = edge.get("path_map")

        if to_node:
            edge_list.append({"from": from_node, "to": to_node})
        elif path_map:
            for label, target in path_map.items():
                edge_list.append(
                    {"from": from_node, "to": target, "label": "conditional"}
                )

    return node_list, edge_list


def create_graph(nodes_list, edges_list):
    nodes = []
    edges = []
    style_llm = {"backgroundColor": "#52c2fa", "color": "#041a75"}
    style_weighted_sampler = {"backgroundColor": "#ebd22f", "color": "#0d0800"}
    style_lamda = {"backgroundColor": "#dcb1fa", "color": "#5e059c"}
    style_multillm = {"backgroundColor": "#c99999", "color": "#381515"}
    style_agent = {"backgroundColor": "#e37a30", "color": "#401b01"}
    nodes.append(
        StreamlitFlowNode(
            id="START",
            pos=(0, 0),
            data={
                "content": "__start__",
            },
            node_type="input",
            source_position="right",
            draggable=True,
            style={
                "color": "white",
                "backgroundColor": "#00c04b",
                "border": "2px solid white",
            },
        )
    )

    for node in nodes_list:
        style = {}
        if node["node_type"] == "llm":
            style = style_llm
        elif node["node_type"] == "weighted_sampler":
            style = style_weighted_sampler
        elif node["node_type"] == "lambda":
            style = style_lamda
        elif node["node_type"] == "multi_llm":
            style = style_multillm
        elif node["node_type"] == "agent":
            style = style_agent
        else:
            st.warning(f"Not implemented for node type : {node['node_type']}")
            return [], []

        nodes.append(
            StreamlitFlowNode(
                id=node["name"],
                pos=(0, 0),
                data={
                    "content": node["name"],
                    "node_type": node["node_type"],
                    "model_name": node["model_name"],
                    "pre_process": node["pre_process"],
                    "post_process": node["post_process"],
                },
                node_type="default",
                source_position="right",
                target_position="left",
                draggable=True,
                style=style,
            )
        )

    nodes.append(
        StreamlitFlowNode(
            id="END",
            pos=(0, 0),
            data={
                "content": "__end__",
            },
            node_type="output",
            target_position="left",
            draggable=True,
            style={
                "color": "white",
                "backgroundColor": "#d95050",
                "border": "2px solid white",
            },
        )
    )

    for edge in edges_list:
        from_node = edge["from"]
        to_node = edge["to"]
        id = f"{from_node}-{to_node}"
        label = ""
        if "label" in edge and edge["label"]:
            label = edge["label"]
        edges.append(
            StreamlitFlowEdge(
                id=id,
                source=from_node,
                target=to_node,
                edge_type="default",
                label=label,
                label_visibility=True if label else False,
                label_show_bg=True,
                label_bg_style={"fill": "gray"},
                animated=False,
                marker_end={"type": "arrowclosed", "width": 25, "height": 25},
            )
        )

    return nodes, edges


# Get top-level task folders
def get_task_list():
    return [
        f for f in os.listdir(TASKS_DIR) if os.path.isdir(os.path.join(TASKS_DIR, f))
    ]


# Recursively find all folders with both required files
def find_valid_subfolders(task_path):
    valid = []
    for dirpath, _, filenames in os.walk(task_path):
        if "graph_config.yaml" in filenames and "task_executor.py" in filenames:
            valid.append(dirpath)
    return valid


def read_file(file_path):
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


def show_task_list():
    st.markdown("# Tasks 🗒️")
    tasks = get_task_list()

    if tasks:
        for task in tasks:
            if st.button(f"📂 {task}", key=f"task_{task}"):
                st.session_state["selected_task"] = task
                st.session_state["selected_subtask"] = None
                st.rerun()
    else:
        st.info("No tasks created yet.")


def show_subtask_selector(task_name, subtasks):
    st.header(f"Task: {task_name}")
    if st.button("🔙 Back to Task List"):
        st.session_state["selected_task"] = None
        st.rerun()

    st.markdown("### Choose a subtask:")
    for subtask_path in subtasks:
        subtask_name = os.path.relpath(subtask_path, os.path.join(TASKS_DIR, task_name))
        if st.button(f"📄 {subtask_name}", key=subtask_path):
            st.session_state["selected_subtask"] = subtask_path
            st.rerun()


def show_subtask_details(subtask_path):
    st.header(f"Subtask: {os.path.relpath(subtask_path, TASKS_DIR)}")

    if st.button("🔙 Back"):
        st.session_state["selected_subtask"] = None
        st.session_state["selected_task"] = (
            None  # This line ensures clean back navigation
        )
        st.rerun()

    graph_config_path = os.path.join(subtask_path, "graph_config.yaml")
    task_executor_path = os.path.join(subtask_path, "task_executor.py")
    visualizer = GraphConfigVisualizer(graph_config_path)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Graph Visualization",
            "Configuration Overview",
            "Task Executor Overview",
            "Node Details",
        ]
    )

    with tab1:
        st.subheader("Graph Visualization")
        nodes_list, edges_list = parse_graph_config_from_yaml(graph_config_path)
        nodes, edges = create_graph(nodes_list, edges_list)

        if st.session_state["static_flow_state1"] is None:
            st.session_state.static_flow_state1 = StreamlitFlowState(nodes, edges)

        streamlit_flow(
            "static_flow1",
            st.session_state.static_flow_state1,
            fit_view=True,
            show_minimap=True,
            show_controls=True,
            pan_on_drag=True,
            allow_zoom=True,
            hide_watermark=True,
            layout=StressLayout(),
        )

        st.markdown("### Legend")
        st.markdown("- 🟩 **Green**: START node")
        st.markdown("- 🟥 **Red**: END node")
        st.markdown("- 🟦 **Blue Box**: LLM node")
        st.markdown("- 🟨 **Yellow Box**: Sampler node")
        st.markdown("- 🟪 **Purple Box**: Lamda node")
        st.markdown("- 🟫 **Brown Box**: Multi-LLM node")
        st.markdown("- 🟧 **Orange Box**: Agent node")

        st.markdown("**Edges:**")
        st.markdown("- Solid line: Direct flow")
        st.markdown("- Labelled line: Conditional path")

    with tab2:
        st.subheader("graph_config.yaml")

        raw_content = read_file(graph_config_path)

        try:
            parsed_yaml = yaml.safe_load(raw_content)
        except Exception as e:
            st.error(f"Failed to parse YAML: {e}")
            st.code(raw_content, language="yaml")
            st.stop()

        # Expander 1 - Data Config
        if "data_config" in parsed_yaml:
            with st.expander("Data Configuration", expanded=False):
                st.code(
                    yaml.dump(
                        {"data_config": parsed_yaml["data_config"]}, sort_keys=False
                    ),
                    language="yaml",
                )

        # Expander 2 - Output Config
        if "output_config" in parsed_yaml:
            with st.expander("Output Configuration", expanded=False):
                st.code(
                    yaml.dump(
                        {"output_config": parsed_yaml["output_config"]}, sort_keys=False
                    ),
                    language="yaml",
                )

        # Expander 3 - Full YAML
        with st.expander("Complete Configuration", expanded=True):
            st.code(raw_content, language="yaml")

    with tab3:
        st.subheader("task_executor.py")
        content = read_file(task_executor_path)
        st.code(content, language="python")

    with tab4:
        st.subheader("Node Configuration Details")
        node_ids = visualizer.get_node_ids()

        if node_ids:
            selected_node = st.selectbox("Select a node to view details:", node_ids)

            if selected_node:
                node_info = visualizer.get_node_details(selected_node)

                with st.expander("Basic Information", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Node Type:** {node_info['node_type']}")
                        if node_info["output_key"]:
                            st.write(f"**Output Key:** {node_info['output_key']}")
                    with col2:
                        if node_info["pre_process"]:
                            st.write(f"**Pre-processor:** {node_info['pre_process']}")
                        if node_info["post_process"]:
                            st.write(f"**Post-processor:** {node_info['post_process']}")

                if node_info["model"]:
                    with st.expander("Model Configuration", expanded=True):
                        st.json(node_info["model"])

                if node_info["attributes"]:
                    with st.expander("Sampling Attributes", expanded=True):
                        for attr_name, attr_config in node_info["attributes"].items():
                            st.subheader(f"Attribute: {attr_name}")
                            if (
                                isinstance(attr_config, dict)
                                and "values" in attr_config
                            ):
                                st.write("Values:")
                                st.write(attr_config["values"])

                if node_info["prompts"]:
                    with st.expander("Prompts", expanded=True):
                        for i, prompt in enumerate(node_info["prompts"], 1):
                            st.subheader(f"Prompt {i}")
                            for role, content in prompt.items():
                                st.markdown(f"**{role.title()}:**")
                                st.text_area(
                                    "",
                                    value=content,
                                    height=100,
                                    key=f"{selected_node}_prompt_{i}_{role}",
                                    disabled=True,
                                )

                with st.expander("Raw Configuration"):
                    st.json(node_info["raw"])
        else:
            st.info("No nodes available in the current configuration.")


def main():
    st.set_page_config(page_title="GraSP UI", layout="wide")
    if "selected_task" not in st.session_state:
        st.session_state["selected_task"] = None
    if "selected_subtask" not in st.session_state:
        st.session_state["selected_subtask"] = None
    if "static_flow_state1" not in st.session_state:
        st.session_state["static_flow_state1"] = None

    if st.session_state["selected_subtask"]:
        show_subtask_details(st.session_state["selected_subtask"])

    elif st.session_state["selected_task"]:
        task_path = os.path.join(TASKS_DIR, st.session_state["selected_task"])
        valid_subtasks = find_valid_subfolders(task_path)

        if not valid_subtasks:
            st.warning("No valid subfolders with both files found in this task.")
            if st.button("🔙 Back to Task List"):
                st.session_state["selected_task"] = None
                st.session_state["selected_subtask"] = None
                st.session_state["static_flow_state1"] = None
                st.rerun()
        elif len(valid_subtasks) == 1:
            # Skip selection, show the only subtask directly
            st.session_state["selected_subtask"] = valid_subtasks[0]
            st.session_state["static_flow_state1"] = None
            st.rerun()
        else:
            show_subtask_selector(st.session_state["selected_task"], valid_subtasks)
            st.session_state["static_flow_state1"] = None

    else:
        show_task_list()


if __name__ == "__main__":
    main()
