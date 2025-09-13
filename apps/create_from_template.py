import yaml
from pathlib import Path
from datasets import load_dataset, get_dataset_config_names

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

import os
import time
import json
import csv
import re
import pandas as pd

st.set_page_config(page_title="GraSP UI", layout="wide")
TASKS_DIR = Path("tasks")

TASKS_DIR.mkdir(exist_ok=True)
if "selected_task1" not in st.session_state:
    st.session_state["selected_task1"] = None
if "selected_subtask1" not in st.session_state:
    st.session_state["selected_subtask1"] = None
if "draft_data_config1" not in st.session_state:
    st.session_state["draft_data_config1"] = {}
if "draft_output_config1" not in st.session_state:
    st.session_state["draft_output_config1"] = {}
if "draft_graph_config1" not in st.session_state:
    st.session_state["draft_graph_config1"] = {}
if "current_task1" not in st.session_state:
    st.session_state["current_task1"] = ""
if "nodes1" not in st.session_state:
    st.session_state.nodes1 = []
if "edges1" not in st.session_state:
    st.session_state.edges1 = []
if "parsed_edges1" not in st.session_state:
    st.session_state.parsed_edges1 = []
if "static_flow_state2" not in st.session_state:
    st.session_state["static_flow_state2"] = None
if "draft_yaml1" not in st.session_state:
    st.session_state.draft_yaml1 = None
if "draft_executor1" not in st.session_state:
    st.session_state.draft_executor1 = ""
if "load_yaml" not in st.session_state:
    st.session_state["load_yaml"] = None
if "active_models" not in st.session_state:
    st.session_state["active_models"] = []


def process_graph_config(draft_graph_config):
    nodes = []
    edges = []

    # 1. Process Nodes
    for idx, (node_name, node_info) in enumerate(draft_graph_config["nodes"].items()):
        node_type = node_info.get("node_type", "unknown")
        node_id = f"{node_type}_{idx + 1}"  # index+1

        node_data = {}
        for k, v in node_info.items():
            if k in ["pre_process", "post_process"] and isinstance(v, str):
                # Only take the function/class name
                node_data[k] = v.split(".")[-1]
            elif k != "node_type":
                node_data[k] = v
        node_data["label"] = node_name  # Save node name as label

        node = {
            "id": node_id,
            "type": node_type,
            "data": node_data,
        }
        nodes.append(node)

    # Helper to map node names to (label(id))
    node_name_to_label_id = {
        node["data"]["label"]: f"{node['data']['label']}({node['id']})"
        for node in nodes
    }

    # 2. Process Edges
    draft_edges = draft_graph_config.get("edges", [])

    for edge in draft_edges:
        src = edge["from"]
        tgt = edge.get("to")

        # Skip the first edge from START
        if src == "START":
            continue

        # Skip self-looping end edge (from node to itself)
        if src == tgt:
            continue

        edge_info = {
            "source": node_name_to_label_id.get(src, src),
            "target": node_name_to_label_id.get(tgt, tgt),
        }

        if "condition" in edge:
            # Only keep the function name, not full path
            condition_func_name = edge["condition"].split(".")[-1]

            edge_info["is_conditional"] = True
            edge_info["condition"] = condition_func_name
            # Update path_map with formatted node IDs
            raw_path_map = edge.get("path_map", {})
            formatted_path_map = {
                cond: node_name_to_label_id.get(node_name, node_name)
                for cond, node_name in raw_path_map.items()
            }
            edge_info["path_map"] = formatted_path_map

        edges.append(edge_info)

    return nodes, edges


def read_yaml_file(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
        if config is not None:
            graph_config = config.get("graph_config", {})
            st.session_state.draft_graph_config1 = graph_config

            nodes, edges = process_graph_config(st.session_state.draft_graph_config1)
            st.session_state.nodes1 = nodes
            st.session_state.edges1 = edges

            data_config = config.get("data_config", {})
            st.session_state.draft_data_config1 = data_config

            output_config = config.get("output_config", {})
            st.session_state.draft_output_config1 = output_config

            st.session_state.load_yaml = True


def get_first_record(file_path, file_format, encoding):
    try:
        if file_format == "json":
            with open(file_path, "r", encoding=encoding) as f:
                data = json.load(f)
                return data[0] if isinstance(data, list) else data

        elif file_format == "jsonl":
            with open(file_path, "r", encoding=encoding) as f:
                first_line = f.readline()
                return json.loads(first_line)

        elif file_format == "csv":
            with open(file_path, "r", encoding=encoding) as f:
                reader = csv.DictReader(f)
                return next(reader)

        elif file_format == "parquet":
            df = pd.read_parquet(file_path)
            return df.iloc[0].to_dict()

        else:
            return {"error": f"Unsupported format: {file_format}"}

    except Exception as e:
        return {"error": str(e)}


def setup_task():
    st.subheader("Task Setup")
    task_name = st.text_input("Enter Task Name")
    if st.button("Create"):
        if task_name:
            st.session_state["current_task1"] = task_name
            task_folder = TASKS_DIR / task_name
            if not task_folder.exists():
                task_folder.mkdir(parents=True)
                (task_folder / "graph_config.yaml").write_text("# Graph config\n")
                (task_folder / "task_executor.py").write_text("# Task executor\n")
                st.success(f"Created folder: {task_folder}")
            else:
                st.warning("Task already exists.")
        else:
            st.warning("Please enter a task name.")


@st.fragment
def load_hf_dataset():
    source = st.session_state.draft_data_config1.get("source") or {}

    # Properly convert list -> comma-separated string for prefill
    prefill_split = (
        ", ".join(source.get("split", []))
        if isinstance(source.get("split", []), list)
        else source.get("split", "")
    )

    repo_id = st.text_input(
        "repo_id",
        placeholder="e.g., username/dataset-name",
        value=source.get("repo_id", ""),
    )
    config_name = st.text_input(
        "config_name", placeholder="e.g., default", value=source.get("config_name", "")
    )
    split_input = st.text_input(
        "split (comma-separated)",
        placeholder="e.g., train, test, validation",
        value=prefill_split,
    )
    token = st.text_input(
        "token (optional)", placeholder="hf_token", value=source.get("token", "")
    )
    streaming = st.checkbox(
        "Streaming? (optional)", value=source.get("streaming", False)
    )
    shard = st.text_input("shard (optional)", value=source.get("shard", ""))

    # After taking user input, split properly
    split_list = [s.strip() for s in split_input.split(",") if s.strip()]

    load = st.button("Save & Load Dataset")
    if load:
        progress_bar = st.progress(0)
        status_text = st.empty()

        for percent in range(0, 70, 10):
            progress_bar.progress(percent)
            status_text.text(f"Loading dataset... {percent}%")
            time.sleep(0.1)

        try:
            dataset = load_dataset(
                path=repo_id,
                name=config_name if config_name != "default" else None,
                split=split_list[0] if split_list else None,
                token=token or None,
                streaming=streaming,
            )

            for percent in range(70, 101, 10):
                progress_bar.progress(percent)
                status_text.text(f"Finishing up... {percent}%")
                time.sleep(0.05)

            st.success("Dataset loaded!")
            st.session_state["draft_data_config1"].update(
                {
                    "source": {
                        "type": "hf",
                        "repo_id": repo_id,
                        "config_name": config_name,
                        "split": split_list,
                        "token": token,
                        "streaming": streaming,
                        "shard": shard,
                    }
                }
            )
            st.json(dataset[0])

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error loading dataset: {e}")


@st.fragment
def load_disk_dataset():
    with st.form("disk_data"):
        disk_data = {
            "type": "disk",
            "file_path": st.text_input(
                "File Path",
                value=(st.session_state.draft_data_config1.get("source") or {}).get(
                    "file_path", ""
                ),
            ),
            "file_format": st.text_input(
                "File Format (e.g., csv, json)",
                value=(st.session_state.draft_data_config1.get("source") or {}).get(
                    "file_format", ""
                ),
            ),
            "encoding": st.text_input(
                "Encoding",
                value=(st.session_state.draft_data_config1.get("source") or {}).get(
                    "encoding", ""
                ),
            ),
        }
        load = st.form_submit_button("Save & Load Dataset")
        if load:
            progress_bar = st.progress(0)
            status_text = st.empty()

            for percent in range(0, 70, 10):
                progress_bar.progress(percent)
                status_text.text(f"Loading dataset... {percent}%")
                time.sleep(0.1)
            try:
                first_record = get_first_record(
                    disk_data["file_path"],
                    disk_data["file_format"],
                    disk_data["encoding"],
                )
                for percent in range(70, 101, 10):
                    progress_bar.progress(percent)
                    status_text.text(f"Finishing up... {percent}%")
                    time.sleep(0.05)
                st.success("Dataset loaded!")
                st.session_state["draft_data_config1"].update(
                    {
                        "source": {
                            "type": "disk",
                            "file_path": disk_data["file_path"],
                            "file_format": disk_data["file_format"],
                            "encoding": disk_data["encoding"],
                        }
                    }
                )
                st.json(first_record)
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Error loading dataset: {e}")


def dataset_source():
    with st.container(border=True):
        data_config = st.checkbox("Data Source?", value=True)
        if data_config:
            st.subheader("📂 Data Source")

            # Safely get the current source type
            source_type_existing = (
                st.session_state.get("draft_data_config1", {})
                .get("source", {})
                .get("type", "")
            )

            # Set index based on existing source_type
            source_options = ["hf", "disk"]
            if source_type_existing in source_options:
                index = source_options.index(source_type_existing)
            else:
                index = 0  # Default to "hf"

            source_type = st.selectbox(
                "Select Source Type", source_options, index=index
            )

            if source_type == "hf":
                load_hf_dataset()
            else:
                load_disk_dataset()


@st.fragment
def transformation_form():
    with st.container(border=True):
        transformation = st.checkbox("Transformation?", value=True)
        if transformation:
            # Check if existing transformations are present
            existing_transformations = (
                st.session_state.get("draft_data_config1", {})
                .get("source", {})
                .get("transformations", [])
            )
            existing_count = len(existing_transformations)

            # If transformations exist, prefill the number
            transformations_count = st.number_input(
                "No. of Transformations",
                min_value=1,
                step=1,
                value=existing_count if existing_count > 0 else 1,
            )

            transformations = []

            for i in range(int(transformations_count)):
                # Prefill transform_path and params if available
                if i < existing_count:
                    existing = existing_transformations[i]
                    transform_path_prefill = existing["transform"].split(".")[
                        -1
                    ]  # Take only the class name
                    params_prefill = json.dumps(existing.get("params", {}), indent=2)
                else:
                    transform_path_prefill = ""
                    params_prefill = "{}"

                transform_path = st.text_input(
                    f"#{i + 1} Transform class (e.g., ClassName)",
                    value=transform_path_prefill,
                    key=f"transform_path_{i}",
                )

                params = st.text_area(
                    f"#{i + 1} Transform parameters (Dict)",
                    value=params_prefill,
                    key=f"params_{i}",
                )

                if transform_path and params:
                    try:
                        params_dict = json.loads(params)
                        transformations.append(
                            {
                                "transform": f"processors.data_transform.{transform_path}",
                                "params": params_dict,
                            }
                        )
                    except json.JSONDecodeError:
                        st.error(
                            f"Invalid JSON in parameters for transformation #{i + 1}"
                        )

            transform_submit = st.button("Save Mappings")
            if transform_submit:
                st.session_state["draft_data_config1"]["source"].update(
                    {"transformations": transformations}
                )
                st.success("Transformations saved")


@st.fragment
def sink_hf_form(sink_type):
    with st.form("sink_hf_form"):
        # Prefill if available
        sink_data = st.session_state.get("draft_data_config1", {}).get("sink", {})

        repo_id = st.text_input(
            "repo_id",
            value=sink_data.get("repo_id", ""),
            placeholder="e.g., username/dataset-name",
        )
        config_name = st.text_input(
            "config_name",
            placeholder="e.g., config-name",
            value=sink_data.get("config_name", ""),
        )
        split = st.text_input(
            "split",
            value=sink_data.get("split", ""),
            placeholder="e.g., train, test, validation",
        )
        private = st.checkbox("private", value=sink_data.get("private", False))
        hf_token = st.text_input(
            "hf_token (optional)",
            placeholder="hf-token",
            value=sink_data.get("token", ""),
        )

        if st.form_submit_button("Save HF Sink"):
            st.session_state["draft_data_config1"].update(
                {
                    "sink": {
                        "type": sink_type,
                        "repo_id": repo_id,
                        "config_name": config_name,
                        "split": split,
                        "token": hf_token,
                        "private": private,
                    }
                }
            )
            st.success(
                f"HF Sink Configured with repo_id: {repo_id}, split: {split}, private: {private}"
            )


@st.fragment
def sink_disk_form(sink_type):
    with st.form("sink_disk_form"):
        # Prefill if available
        sink_data = st.session_state.get("draft_data_config1", {}).get("sink", {})

        file_path = st.text_input(
            "file_path",
            value=sink_data.get("file_path", ""),
            placeholder="e.g., /path/to/output/file.json",
        )
        encoding = st.text_input("encoding", value=sink_data.get("encoding", "utf-8"))

        if st.form_submit_button("Save Disk Sink"):
            st.session_state["draft_data_config1"].update(
                {
                    "sink": {
                        "type": sink_type,
                        "file_path": file_path,
                        "encoding": encoding,
                    }
                }
            )
            st.success(
                f"Disk Sink Configured with file_path: {file_path}, encoding: {encoding}"
            )


def dataset_sink():
    with st.container(border=True):
        dataset_sink = st.checkbox("Dataset Sink?", value=True)
        if dataset_sink:
            st.subheader("📂 Data Sink (Optional)")

            # Safely get existing sink type
            sink_data = st.session_state.get("draft_data_config1", {}).get("sink", {})
            sink_type_existing = sink_data.get("type", "")

            sink_options = ["hf", "json", "jsonl", "csv", "parquet"]
            if sink_type_existing in sink_options:
                index = sink_options.index(sink_type_existing)
            else:
                index = 0  # Default to "hf"

            sink_type = st.selectbox("Select sink Type", sink_options, index=index)

            if sink_type == "hf":
                sink_hf_form(sink_type)
            else:
                sink_disk_form(sink_type)


@st.fragment
def output_config():
    with st.container(border=True):
        output_config_enabled = st.checkbox("Output Config?", value=True)

        if output_config_enabled:
            st.subheader("Output Config")

            # Safely get existing data
            draft = st.session_state.get("draft_output_config1", {})
            generator_existing = draft.get("generator", "")
            output_map_existing = draft.get("output_map", {})
            oasst_mapper_existing = draft.get("oasst_mapper", {})

            # Prefill Generator Function
            generator_display = (
                draft.get("generator", "").split(".")[-1]
                if draft.get("generator")
                else ""
            )
            generator_function = st.text_input(
                "Generator Function", value=generator_display
            )

            # Prefill Output Map
            st.markdown("### Output Map")
            existing_fields = list(output_map_existing.keys())
            output_mapping_count = st.number_input(
                "Add field renames",
                min_value=1,
                step=1,
                value=len(existing_fields) if existing_fields else 1,
                key="mapping_count",
            )

            mappings = {}
            for i in range(int(output_mapping_count)):
                field_default = existing_fields[i] if i < len(existing_fields) else ""
                field_data = (
                    output_map_existing.get(field_default, {}) if field_default else {}
                )

                col1, col2 = st.columns([1, 3])
                with col1:
                    field_name = st.text_input(
                        f"Field Name #{i + 1}",
                        value=field_default,
                        key=f"field_name_{i}",
                    )
                with col2:
                    from_field = st.text_input(
                        f"From #{i + 1}",
                        value=field_data.get("from", ""),
                        key=f"from_{i}",
                    )
                    transform = st.text_input(
                        f"Transform #{i + 1}",
                        value=field_data.get("transform", ""),
                        key=f"transform_{i}",
                    )
                    value = st.text_input(
                        f"Value #{i + 1}",
                        value=json.dumps(field_data.get("value", ""))
                        if "value" in field_data
                        else "",
                        key=f"value_{i}",
                    )

                if field_name:
                    # Note: Only add keys that have non-empty values
                    mappings[field_name] = {}
                    if from_field:
                        mappings[field_name]["from"] = from_field
                    if transform:
                        mappings[field_name]["transform"] = transform
                    if value:
                        try:
                            parsed_value = json.loads(value)
                        except json.JSONDecodeError:
                            parsed_value = value
                        mappings[field_name]["value"] = parsed_value

            # Prefill OASST Mapper
            st.markdown("### OASST Mapper")
            required = st.checkbox(
                "Required?",
                value=oasst_mapper_existing.get("required", "").lower()
                in ["yes", "true"],
                key="required_checkbox",
            )
            oasst_type = st.selectbox(
                "Type",
                options=["sft", "dpo"],
                index=0 if oasst_mapper_existing.get("type", "sft") == "sft" else 1,
                key="oasst_type",
            )
            intermediate_writing = st.checkbox(
                "Intermediate Writing?",
                value=oasst_mapper_existing.get("intermediate_writing", "").lower()
                in ["yes", "true"],
                key="intermediate_checkbox",
            )

            # Save Button
            save = st.button("Save Output Config")
            if save:
                st.session_state["draft_output_config1"].update(
                    {
                        "generator": f"tasks.{st.session_state.get('current_task1', '')}.task_executor.{generator_function}",
                        "output_map": mappings,
                        "oasst_mapper": {
                            "required": "yes" if required else "no",
                            "type": oasst_type,
                            "intermediate_writing": "yes"
                            if intermediate_writing
                            else "no",
                        },
                    }
                )
                st.success("Output Config saved")


def show_graph_builder():
    st.subheader("Create Graph")

    # Split into two columns
    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("---")
        st.subheader("Current Nodes and Edges")

        if st.button("Show Nodes and Edges"):
            # Display all nodes
            st.markdown("**Nodes:**")
            for i, node in enumerate(st.session_state.nodes1):
                with st.expander(f"{node['data']['label']}: {node['id']}"):
                    st.json(node)

            # Display all edges
            if "edges1" in st.session_state and st.session_state.edges1:
                st.markdown("**Edges:**")
                for i, edge in enumerate(st.session_state.edges1):
                    markdown = ""
                    if "is_conditional" in edge and edge["is_conditional"]:
                        markdown = f"idx={i} | {edge['source']} (conditional)"
                    else:
                        markdown = f"idx={i} | {edge['source']} ----→ {edge['target']}"
                    with st.expander(markdown):
                        st.json(edge)

    with col1:
        # Node palette and configuration
        st.markdown("---")
        st.subheader("Add Node")
        node_type = st.selectbox(
            "Node Type", ["llm", "weighted_sampler", "lambda", "multi_llm", "agent"]
        )

        node_label = st.text_input("Node Label")

        if st.button("Add Node"):
            if add_node(node_type, node_label):
                st.info(f"Node added: {node_label}")

        # Node configuration
        if st.session_state.nodes1:
            st.subheader("Configure Node")
            selected_node = st.selectbox(
                "Select Node",
                [
                    node["data"]["label"] + "(" + node["id"] + ")"
                    for node in st.session_state.nodes1
                ],
            )
            node_id = selected_node.split("(")[-1].rstrip(")")
            configure_node(node_id)

        # Edge creation
        if len(st.session_state.nodes1) >= 2:
            st.markdown("---")
            st.subheader("Add Edge")
            source = st.selectbox(
                "From",
                [
                    node["data"]["label"] + "(" + node["id"] + ")"
                    for node in st.session_state.nodes1
                ],
                key="edge_source",
            )

            is_conditional = st.checkbox("Conditional Edge")

            if is_conditional:
                condition = st.text_input("Condition Function")
                num_paths = st.number_input("Number of Paths", min_value=1, value=2)

                paths = {}
                for i in range(num_paths):
                    col1, col2 = st.columns(2)
                    with col1:
                        condition_value = st.text_input(
                            f"Condition {i + 1}", key=f"cond_{i}"
                        )
                    with col2:
                        target_node = st.selectbox(
                            "Target",
                            ["END"]
                            + [
                                node["data"]["label"] + "(" + node["id"] + ")"
                                for node in st.session_state.nodes1
                            ],
                            key=f"target_{i}",
                        )
                    paths[condition_value] = target_node

            else:
                target = st.selectbox(
                    "To",
                    [
                        node["data"]["label"] + "(" + node["id"] + ")"
                        for node in st.session_state.nodes1
                    ],
                    key="edge_target",
                )

            if st.button("Add Edge"):
                if not is_conditional:
                    add_edge(
                        source,
                        target,
                        is_conditional,
                        condition if is_conditional else None,
                        paths if is_conditional else None,
                    )
                else:
                    add_edge_conditional(
                        source,
                        is_conditional,
                        condition if is_conditional else None,
                        paths if is_conditional else None,
                    )

            if len(st.session_state.edges1) >= 1:
                edge_to_remove = st.number_input(
                    "Edge Index to Remove",
                    min_value=0,
                    max_value=len(st.session_state.edges1) - 1,
                    step=1,
                )
                if st.button("Remove Edge"):
                    delete_edge(edge_to_remove)


@st.dialog("Add Node Failed")
def failed_node_message(item, reason: str = "None"):
    st.error(f"Node {item} failed. Reason: {reason}")


def check_node_exists(node_label):
    for node in st.session_state.nodes:
        label = node["data"]["label"]
        if label == node_label:
            return True
    return False


def add_node(node_type: str, label: str):
    if check_node_exists(label):
        failed_node_message(label, "duplicate node found.")
        return False
    node_id = f"{node_type}_{len(st.session_state.nodes1) + 1}"
    st.session_state.nodes1.append(
        {"id": node_id, "type": node_type, "data": {"label": label}}
    )
    return True


def delete_edge(index: int):
    if "edges1" in st.session_state and 0 <= index < len(st.session_state.edges1):
        del st.session_state.edges1[index]
        st.rerun()


def add_edge(
    source: str,
    target: str,
    is_conditional: bool,
    condition: str = None,
    paths: dict[str, str] = None,
):
    edge = {
        "source": source,
        "target": target,
    }

    if is_conditional:
        edge.update(
            {
                "is_conditional": is_conditional,
                "condition": condition,
                "path_map": paths,
            }
        )

    st.session_state.edges1.append(edge)


def add_edge_conditional(
    source: str,
    is_conditional: bool,
    condition: str = None,
    paths: dict[str, str] = None,
):
    edge = {
        "source": source,
    }

    if is_conditional:
        edge.update(
            {
                "is_conditional": is_conditional,
                "condition": condition,
                "path_map": paths,
            }
        )

    st.session_state.edges1.append(edge)


def configure_node(node_id: str):
    node = next(node for node in st.session_state.nodes1 if node["id"] == node_id)

    if node["type"] == "llm":
        configure_llm_node(node)
    elif node["type"] == "weighted_sampler":
        configure_sampler_node(node)
    elif node["type"] == "lambda":
        configure_lambda_node(node)
    elif node["type"] == "multi_llm":
        configure_multi_llm_node(node)
    elif node["type"] == "agent":
        configure_agent_node(node)


def configure_agent_node(node: dict):
    data = node.get("data", {})
    model_name = data.get("model", "")
    parameters = data.get("parameters", {})
    temperature = float(parameters.get("temperature", 0.7))
    max_tokens = int(parameters.get("max_tokens", 500))

    pre_process = data.get("pre_process", "")
    post_process = data.get("post_process", "")
    chat_history = data.get("chat_history", False)
    default_messages = data.get("prompt", [])
    default_tools = data.get("tools", [])

    # UI Components
    model = st.selectbox(
        "Model",
        st.session_state.active_models,
        index=st.session_state.active_models.index(model_name)
        if model_name in st.session_state.active_models
        else 0,
    )

    temperature = st.slider("Temperature", 0.0, 1.0, temperature)
    max_tokens = st.number_input("Max Tokens", 100, 2000, max_tokens)
    output_keys = st.text_input(
        "Output Keys",
        placeholder="e.g. ai_answer",
        value=",".join(data.get("output_keys", []))
        if isinstance(data.get("output_keys", []), list)
        else data.get("output_keys", ""),
    )
    output_keys = [key.strip() for key in output_keys.split(",") if key.strip()]
    # Tools Input
    tools_input = st.text_area(
        "Tools (comma-separated paths)",
        value=", ".join(default_tools),
        placeholder="eg. tasks.agent_task.tools.func_name, tasks.agent_task.tools_from_module, tasks.agent_task.tools_from_class",
        help="Provide import paths for tools the agent will use",
    )
    tools = [tool.strip() for tool in tools_input.split(",") if tool.strip()]

    pre_process = st.text_input(
        "Pre-Process (optional)",
        placeholder="eg. CritiqueAnsNodePreProcessor",
        help="write the name of the pre_processor function",
        value=pre_process,
    )

    post_process = st.text_input(
        "Post-Process (optional)",
        placeholder="eg. CritiqueAnsNodePostProcessor",
        help="write the name of the post_processor function",
        value=post_process,
    )

    chat_history = st.checkbox("Enable Chat History", value=chat_history)

    # Prompt Messages
    st.subheader("Prompt Messages")
    num_messages = st.number_input(
        "Number of Messages", 1, 5, max(1, len(default_messages))
    )

    messages = []
    for i in range(int(num_messages)):
        default_message = default_messages[i] if i < len(default_messages) else {}
        default_role = list(default_message.keys())[0] if default_message else "system"
        default_content = (
            default_message.get(default_role, "") if default_message else ""
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            role = st.selectbox(
                f"Role {i + 1}",
                ["system", "user", "assistant"],
                index=["system", "user", "assistant"].index(default_role),
                key=f"agent_role_{i}",
            )
        with col2:
            content = st.text_area(
                f"Content {i + 1}", value=default_content, key=f"agent_content_{i}"
            )
        messages.append({role: content})

    # Save
    if st.button("Save"):
        node["data"].update(
            {
                "node_type": "agent",
                "model": model,
                "parameters": {"temperature": temperature, "max_tokens": max_tokens},
                "prompt": messages,
                "pre_process": pre_process,
                "post_process": post_process,
                "chat_history": chat_history,
                "tools": tools,
                "output_keys": output_keys,
            }
        )
        st.info("Saved Successfully")

    # Delete
    match = re.search(r"(\d+)$", node["id"])
    index = int(match.group(1)) - 1
    if index >= len(st.session_state.nodes1):
        index = index - 1
    if st.button("Delete Node"):
        del st.session_state["nodes1"][index]
        st.rerun()


def configure_llm_node(node: dict):
    data = node.get("data", {})
    model_data = data.get("model", {})
    params = model_data.get("parameters", {})

    model = st.selectbox(
        "Model",
        st.session_state.active_models,
        index=st.session_state.active_models.index(model_data.get("name", "gpt4"))
        if model_data.get("name", "gpt4") in st.session_state.active_models
        else 1,
    )
    output_keys = st.text_input(
        "Output Keys",
        placeholder="e.g. ai_answer",
        value=",".join(data.get("output_keys", []))
        if isinstance(data.get("output_keys", []), list)
        else data.get("output_keys", ""),
    )
    output_keys = [key.strip() for key in output_keys.split(",") if key.strip()]

    temperature = st.slider(
        "Temperature", 0.0, 1.0, float(params.get("temperature", 0.7))
    )
    max_tokens = st.number_input(
        "Max Tokens", 100, 2000, int(params.get("max_tokens", 500))
    )
    chat_history = st.checkbox(
        "Enable Chat History", value=data.get("chat_history", False)
    )
    pre_process = st.text_input(
        "Pre-Process (optional)",
        placeholder="eg. NodeNamePreProcessor",
        help="write the name of the pre_processor function",
        value=data.get("pre_process", ""),
    )
    post_process = st.text_input(
        "Post-Process (optional)",
        placeholder="eg. NodeNamePostProcessor",
        help="write the name of the post_processor function",
        value=data.get("post_process", ""),
    )

    st.subheader("Prompt Messages")
    default_messages = data.get("prompt", [])
    num_messages = st.number_input(
        "Number of Messages", 1, 5, max(1, len(default_messages))
    )

    messages = []
    for i in range(int(num_messages)):
        default_message = default_messages[i] if i < len(default_messages) else {}
        default_role = list(default_message.keys())[0] if default_message else "system"
        default_content = (
            default_message.get(default_role, "") if default_message else ""
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            role = st.selectbox(
                f"Role {i + 1}",
                ["system", "user", "assistant"],
                index=["system", "user", "assistant"].index(default_role),
                key=f"role_{i}",
            )
        with col2:
            content = st.text_area(
                f"Content {i + 1}", value=default_content, key=f"content_{i}"
            )
        messages.append({role: content})

    if st.button("Save"):
        node["data"].update(
            {
                "node_type": "llm",
                "output_keys": output_keys,
                "model": {
                    "name": model,
                    "parameters": {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                },
                "prompt": messages,
                "pre_process": pre_process,
                "post_process": post_process,
                "chat_history": chat_history,
            }
        )
        st.info("Saved Successfully")

    match = re.search(r"(\d+)$", node["id"])
    index = int(match.group(1)) - 1
    if index >= len(st.session_state.nodes1):
        index = index - 1
    if st.button("Delete Node"):
        del st.session_state["nodes1"][index]
        st.rerun()


def configure_sampler_node(node: dict):
    st.subheader("Attributes")
    data = node.get("data", {})
    existing_attrs = data.get("attributes", {})
    num_attrs = st.number_input(
        "Number of Attributes",
        1,
        20,
        value=len(existing_attrs) if existing_attrs else 1,
    )

    attributes = {}
    attr_keys = list(existing_attrs.keys())

    for i in range(int(num_attrs)):
        default_name = attr_keys[i] if i < len(attr_keys) else ""
        default_values = (
            existing_attrs.get(default_name, {}).get("values", [])
            if default_name
            else []
        )
        default_weights = (
            existing_attrs.get(default_name, {}).get("weights", [])
            if default_name
            else []
        )

        attr_name = st.text_input(
            f"Attribute {i + 1} Name", value=default_name, key=f"attr_name_{i}"
        )
        values = st.text_input(
            f"Values (comma-separated)",
            value=", ".join(map(str, default_values)),
            key=f"values_{i}",
        )
        weights = st.text_input(
            f"Weights (comma-separated, optional)",
            value=", ".join(map(str, default_weights)) if default_weights else "",
            key=f"weights_{i}",
        )

        if attr_name:
            attributes[attr_name] = {
                "values": [v.strip() for v in values.split(",")],
                "weights": [float(w.strip()) for w in weights.split(",")]
                if weights
                else None,
            }

    if st.button("Save"):
        node["data"].update(
            {
                "node_type": "weighted_sampler",
                "attributes": attributes,
            }
        )
        st.info("Saved Successfully")

    match = re.search(r"(\d+)$", node["id"])
    index = int(match.group(1)) - 1
    if index >= len(st.session_state.nodes1):
        index = index - 1
    if st.button("Delete Node"):
        del st.session_state["nodes1"][index]
        st.rerun()


def configure_lambda_node(node: dict):
    data = node.get("data", {})
    default_lambda = data.get("lambda", "")
    default_node_state = data.get("node_state", "")

    function_path = st.text_input(
        "Function Path", value=default_lambda, help="e.g., path.to.module.function_name"
    )
    node_state = st.text_input("Node State", value=default_node_state)
    output_keys = st.text_input(
        "Output Keys",
        placeholder="e.g. ai_answer",
        value=",".join(data.get("output_keys", []))
        if isinstance(data.get("output_keys", []), list)
        else data.get("output_keys", ""),
    )
    output_keys = [key.strip() for key in output_keys.split(",") if key.strip()]

    if st.button("Save"):
        node["data"].update(
            {
                "node_type": "lambda",
                "lambda": function_path,
                "node_state": node_state,
                "output_keys": output_keys,
            }
        )
        st.info("Saved Successfully")

    match = re.search(r"(\d+)$", node["id"])
    index = int(match.group(1)) - 1
    if index >= len(st.session_state.nodes1):
        index = index - 1
    if st.button("Delete Node"):
        del st.session_state["nodes1"][index]
        st.rerun()


def configure_multi_llm_node(node: dict):
    data = node.get("data", {})
    existing_prompt = data.get("prompt", [])
    pre_process_multi_llm = data.get("pre_process", "")
    post_process_multi_llm = data.get(
        "multi_llm_post_process", data.get("post_process", "")
    )
    output_keys = st.text_input(
        "Output Keys",
        placeholder="e.g. ai_answer",
        value=",".join(data.get("output_keys", []))
        if isinstance(data.get("output_keys", []), list)
        else data.get("output_keys", ""),
    )
    output_keys = [key.strip() for key in output_keys.split(",") if key.strip()]
    existing_models = data.get("models", {})

    st.subheader("Prompt Messages")
    num_messages = st.number_input(
        "Number of Messages", 1, 5, max(1, len(existing_prompt))
    )

    messages = []
    for i in range(num_messages):
        col1, col2 = st.columns([1, 3])
        default_role = (
            list(existing_prompt[i].keys())[0] if i < len(existing_prompt) else "user"
        )
        default_content = (
            list(existing_prompt[i].values())[0] if i < len(existing_prompt) else ""
        )
        with col1:
            role = st.selectbox(
                f"Role {i + 1}",
                ["system", "user", "assistant"],
                key=f"role_{i}",
                index=["system", "user", "assistant"].index(default_role),
            )
        with col2:
            content = st.text_area(
                f"Content {i + 1}", value=default_content, key=f"content_{i}"
            )
        messages.append({role: content})

    pre_process_multi_llm = st.text_input(
        "Pre-Process MultiLLM (optional)",
        placeholder="eg. generate_samples_pre_process",
        help="write the name of the pre_processor function",
        value=pre_process_multi_llm,
    )
    post_process_multi_llm = st.text_input(
        "Post-Process MultiLLM (optional)",
        placeholder="eg. generate_samples_post_process",
        help="write the name of the post_processor function",
        value=post_process_multi_llm,
    )

    st.subheader("Models")
    num_models = st.number_input("Number of Models", 1, 5, max(1, len(existing_models)))

    models = {}
    model_keys = list(existing_models.keys())

    for i in range(num_models):
        st.markdown(f"**Model {i + 1}**")
        default_model_key = model_keys[i] if i < len(model_keys) else ""
        default_model_data = existing_models.get(default_model_key, {})
        default_name = default_model_data.get("name", "")
        default_temperature = default_model_data.get("parameters", {}).get(
            "temperature", 0.7
        )
        default_max_tokens = default_model_data.get("parameters", {}).get(
            "max_tokens", 500
        )

        model_name = st.text_input(
            "Name", key=f"model_name_{i}", value=default_model_key
        )
        model_type = st.selectbox(
            "Type",
            st.session_state.active_models,
            key=f"model_type_{i}",
            index=st.session_state.active_models.index(default_name)
            if default_name in st.session_state.active_models
            else 0,
        )
        temperature = st.slider(
            "Temperature", 0.0, 1.0, default_temperature, key=f"temp_{i}"
        )
        max_tokens = st.number_input(
            "Max Tokens", 100, 2000, default_max_tokens, key=f"tokens_{i}"
        )

        if model_name:
            models[model_name] = {
                "name": model_type,
                "parameters": {"temperature": temperature, "max_tokens": max_tokens},
            }

    if st.button("Save"):
        node["data"].update(
            {
                "node_type": "multi_llm",
                "prompt": messages,
                "models": models,
                "pre_process": pre_process_multi_llm,
                "multi_llm_post_process": post_process_multi_llm,
                "output_keys": output_keys,
            }
        )
        st.info("Saved Successfully")

    match = re.search(r"(\d+)$", node["id"])
    index = int(match.group(1)) - 1
    if index >= len(st.session_state.nodes1):
        index = index - 1
    if st.button("Delete Node"):
        del st.session_state["nodes1"][index]
        st.rerun()


def parse_edges(raw_edges):
    edge_list = []
    for edge in raw_edges:
        from_node = edge["source"]
        to_node = edge.get("target")
        condition = edge.get("condition")
        path_map = edge.get("path_map")
        is_conditional = edge.get("is_conditional")

        if to_node:
            edge_list.append({"source": from_node, "target": to_node})
        elif path_map:
            for label, target in path_map.items():
                edge_list.append(
                    {
                        "source": from_node,
                        "target": target,
                        "label": "conditional",
                        "condition": condition,
                        "is_conditional": is_conditional,
                    }
                )

    st.session_state.parsed_edges1 = edge_list


def create_graph(nodes_list, edges_list):
    nodes = []
    edges = []
    style_llm = {"backgroundColor": "#52c2fa", "color": "#041a75"}
    style_weighted_sampler = {"backgroundColor": "#ebd22f", "color": "#0d0800"}
    style_lambda = {"backgroundColor": "#dcb1fa", "color": "#5e059c"}
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
        if node["type"] == "llm":
            style = style_llm
        elif node["type"] == "weighted_sampler":
            style = style_weighted_sampler
        elif node["type"] == "lambda":
            style = style_lambda
        elif node["type"] == "multi_llm":
            style = style_multillm
        elif node["type"] == "agent":
            style = style_agent
        else:
            st.warning(f"Not implemented for node type : {node['type']}")
        id = node["data"]["label"] + "(" + node["id"] + ")"
        nodes.append(
            StreamlitFlowNode(
                id=id,
                pos=(0, 0),
                data={
                    "content": node["data"]["label"],
                    "node_type": node["type"],
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

    first_node_label = nodes_list[0]["data"]["label"]
    first_node_id = nodes_list[0]["id"]
    edges.append(
        StreamlitFlowEdge(
            id=f"START-{first_node_label}({first_node_id})",
            source="START",
            target=f"{first_node_label}({first_node_id})",
            edge_type="default",
            animated=False,
            marker_end={"type": "arrowclosed", "width": 25, "height": 25},
        )
    )

    for edge in edges_list:
        from_node = edge["source"]
        to_node = edge.get("target")
        id = f"{from_node}-{to_node}"
        is_conditional = edge.get("is_conditional")
        label = ""
        if is_conditional:
            label = "(conditional)"
        edges.append(
            StreamlitFlowEdge(
                id=id,
                source=from_node,
                target=to_node,
                edge_type="default",
                label=label,
                label_show_bg=True,
                label_bg_style={"fill": "gray"},
                animated=False,
                marker_end={"type": "arrowclosed", "width": 25, "height": 25},
            )
        )
    last_node_id = nodes_list[-1]["id"]
    last_node_label = nodes_list[-1]["data"]["label"]
    edges.append(
        StreamlitFlowEdge(
            id=f"{last_node_label}({last_node_id})-END",
            source=f"{last_node_label}({last_node_id})",
            target="END",
            edge_type="default",
            animated=False,
            marker_end={"type": "arrowclosed", "width": 25, "height": 25},
        )
    )

    return nodes, edges


@st.fragment
def show_graph():
    if st.button("Refresh Graph", key="refresh_graph1"):
        del st.session_state.static_flow_state2
        st.rerun()
    parse_edges(st.session_state.edges1)
    nodes, edges = create_graph(st.session_state.nodes1, st.session_state.parsed_edges1)
    if st.session_state["static_flow_state2"] is None:
        st.session_state.static_flow_state2 = StreamlitFlowState(nodes, edges)

    streamlit_flow(
        "static_flow2",
        st.session_state.static_flow_state2,
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
    st.markdown("- 🟪 **Purple Box**: Lambda node")
    st.markdown("- 🟫 **Brown Box**: Multi-LLM node")
    st.markdown("- 🟧 **Orange Box**: Agent node")

    st.markdown("**Edges:**")
    st.markdown("- Solid line: Direct flow")
    st.markdown("- Labelled line: Conditional path")


def strip_node_id(name):
    """Removes ID like '(llm_1)' from 'paraphrase_answer(llm_1)'."""
    return re.split(r"\(", name)[0]


@st.fragment
def publish(yaml_str, executor_code):
    if st.button("Publish"):
        task_name = st.session_state.get("current_task1")
        if task_name:
            task_folder = Path("tasks") / task_name
            yaml_path = task_folder / "graph_config.yaml"
            yaml_path.write_text(yaml_str)
            task_path = Path("tasks") / task_name / "task_executor.py"
            task_path.write_text(executor_code)
            st.success(f"Published to {yaml_path} and {task_path}")
        else:
            st.warning("No task selected. Please create a task first.")


@st.fragment
def generate_yaml_and_executor():
    st.markdown("---")
    if st.button("Generate Yaml and Task Executor file"):
        # ----------- YAML Generation ------------
        data_config = st.session_state.get("draft_data_config1", {})
        output_config = st.session_state.get("draft_output_config1", {})
        current_task = st.session_state.get("current_task1", "example_task")

        yaml_file = {
            "data_config": data_config,
            "graph_config": {"nodes": {}, "edges": []},
            "output_config": output_config,
        }

        nodes = st.session_state.get("nodes1", [])
        edges = st.session_state.get("edges1", [])

        for node in nodes:
            node_type = node.get("type")
            data = node.get("data", {})
            label = data.get("label")
            output_keys = data.get("output_keys", [])
            prompts_list = data.get("prompt", [])
            for p in prompts_list:
                if p.get("system"):
                    p["system"] = " |\n" + p.get("system")
                if p.get("user"):
                    p["user"] = " |\n" + p.get("user")
                if p.get("assistant"):
                    p["assistant"] = " |\n" + p.get("assistant")
            if not label:
                continue

            node_entry = {
                "node_type": node_type,
                "output_keys": output_keys,
                "prompt": data.get("prompt", []),
            }

            if node_type in ["llm", "multi_llm"]:
                if node_type == "llm":
                    node_entry["model"] = {
                        "name": data.get("model", ""),
                        "parameters": data.get("parameters", {}),
                    }
                    if data.get("chat_history") == True:
                        node_entry["chat_history"] = data.get("chat_history", False)

                elif node_type == "multi_llm":
                    node_entry["models"] = data.get("models", [])

                # Add pre_process if present
                if data.get("pre_process"):
                    node_entry["pre_process"] = (
                        f"tasks.{current_task}.task_executor.{data.get('pre_process')}"
                    )

                # Add post_process or multi_llm_post_process accordingly
                if data.get("post_process"):
                    key = (
                        "multi_llm_post_process"
                        if node_type == "multi_llm"
                        else "post_process"
                    )
                    node_entry[key] = (
                        f"tasks.{current_task}.task_executor.{data.get('post_process')}"
                    )

            elif node_type == "weighted_sampler":
                node_entry = {
                    "node_type": node_type,
                    "attributes": data.get("attributes", {}),
                }
            elif node_type == "lambda":
                node_entry = {
                    "node_type": node_type,
                    "lambda": data.get("lambda", ""),
                    "node_state": data.get("node_state", ""),
                }
            elif node_type == "agent":
                node_entry["model"] = {
                    "name": data.get("model", ""),
                    "parameters": data.get("parameters", {}),
                }
                node_entry["chat_history"] = data.get("chat_history", False)
                if data.get("tools"):
                    node_entry["tools"] = data.get("tools", [])
                if data.get("pre_process"):
                    node_entry["pre_process"] = (
                        f"tasks.{current_task}.task_executor.{data.get('pre_process')}"
                    )
                if data.get("post_process"):
                    node_entry["post_process"] = (
                        f"tasks.{current_task}.task_executor.{data.get('post_process')}"
                    )

            yaml_file["graph_config"]["nodes"].update({label: node_entry})

        # ----- Build graph_config.edges -----
        all_edges = []

        if nodes:
            # Add START edge to first node
            first_node_label = nodes[0]["data"]["label"]
            all_edges.append({"from": "START", "to": first_node_label})

        for edge in edges:
            if edge.get("is_conditional"):
                path_map = {
                    k: strip_node_id(v) for k, v in edge.get("path_map", {}).items()
                }
                all_edges.append(
                    {
                        "from": strip_node_id(edge.get("source")),
                        "condition": f"tasks.{current_task}.task_executor.{edge.get('condition', '')}",
                        "path_map": path_map,
                    }
                )
            else:
                all_edges.append(
                    {
                        "from": strip_node_id(edge.get("source")),
                        "to": strip_node_id(edge.get("target")),
                    }
                )

        # Add END edge logic
        if len(nodes) == 1:
            # Only one node, connect to END
            only_node_label = nodes[0]["data"]["label"]
            all_edges.append({"from": only_node_label, "to": "END"})
        # Add END edge if last edge is not conditional
        elif edges:
            last_edge = edges[-1]
            if not last_edge.get("is_conditional"):
                last_target = strip_node_id(last_edge.get("target"))
                all_edges.append({"from": last_target, "to": "END"})

        yaml_file["graph_config"]["edges"] = all_edges

        yaml_str = yaml.dump(yaml_file, sort_keys=False)
        st.session_state.draft_yaml1 = yaml_str

        # ---------- Task Executor Generation ------------
        imports = [
            "from core.graph.functions.node_processor import NodePreProcessor, NodePostProcessor",
            "from core.graph.functions.edge_condition import EdgeCondition",
            "from processors.output_record_generator import BaseOutputGenerator",
            "from core.graph.grasp_state import GraspState",
            "from core.base_task_executor import BaseTaskExecutor",
            "from utils import utils, constants",
        ]

        classes = []
        for node in nodes:
            data = node.get("data", {})
            pre = data.get("pre_process")
            post = data.get("post_process")

            if pre and not any(f"class {pre}(" in c for c in classes):
                classes.append(f"""
            class {pre}(NodePreProcessor):
                def apply(self, state:GraspState) -> GraspState:
                    return state
            """)

            if post and not any(f"class {post}(" in c for c in classes):
                classes.append(f"""
            class {post}(NodePostProcessorWithState):
                def apply(self, resp:GraspMessage, state:GraspState) -> GraspState:
                    return resp
            """)

        for edge in edges:
            condition = edge.get("condition")
            if condition and not any(f"class {condition}(" in c for c in classes):
                classes.append(f"""
            class {condition}(EdgeCondition):
                def apply(state:GraspState) -> str:
                    return "END"
            """)

        generator = output_config.get("generator")
        if generator:
            class_name = generator.split(".")[-1]
            if not any(f"class {class_name}(" in c for c in classes):
                classes.append(f"""
            class {class_name}(BaseOutputGenerator):
                def map_function(data: Any, state: GraspState):
                    return None
            """)

        executor_code = "\n\n".join(imports + classes)
        st.session_state.draft_executor1 = executor_code

        # ---------- Tabs Display ----------
        tab1, tab2 = st.tabs(["YAML", "Task Executor"])
        with tab1:
            st.code(yaml_str, language="yaml")
        with tab2:
            st.code(executor_code, language="python")

    st.markdown("---")
    if st.session_state.draft_yaml1 and st.session_state.draft_executor1:
        publish(st.session_state.draft_yaml1, st.session_state.draft_executor1)


def get_task_list():
    return [
        f for f in os.listdir(TASKS_DIR) if os.path.isdir(os.path.join(TASKS_DIR, f))
    ]


def find_valid_subfolders(task_path):
    valid = []
    for dirpath, _, filenames in os.walk(task_path):
        if "graph_config.yaml" in filenames and "task_executor.py" in filenames:
            valid.append(dirpath)
    return valid


def show_task_list():
    st.header("Select a Task")
    tasks = get_task_list()

    if tasks:
        for task in tasks:
            if st.button(f"📂 {task}", key=f"task_{task}"):
                st.session_state["selected_task1"] = task
                st.session_state["selected_subtask1"] = None
                st.rerun()
    else:
        st.info("No tasks created yet.")


def show_subtask_selector(task_name, subtasks):
    st.header(f"Task: {task_name}")
    if st.button("🔙 Back to Task List"):
        st.session_state["selected_task1"] = None
        st.rerun()

    st.markdown("### Choose a subtask:")
    for subtask_path in subtasks:
        subtask_name = os.path.relpath(subtask_path, os.path.join(TASKS_DIR, task_name))
        if st.button(f"📄 {subtask_name}", key=subtask_path):
            st.session_state["selected_subtask1"] = subtask_path
            st.rerun()


def show_subtask_details(subtask_path):
    st.header(f"Subtask: {os.path.relpath(subtask_path, TASKS_DIR)}")

    if st.button("🔙 Back"):
        st.session_state["selected_subtask1"] = None
        st.session_state["selected_task1"] = None
        st.session_state.load_yaml = False
        st.rerun()

    graph_config_path = os.path.join(subtask_path, "graph_config.yaml")

    if not st.session_state.load_yaml:
        read_yaml_file(graph_config_path)

    setup_task()
    dataset_source()
    transformation_form()
    dataset_sink()

    show_graph_builder()
    st.markdown("---")
    if len(st.session_state.nodes1) >= 1:
        show_graph()
    output_config()
    generate_yaml_and_executor()


def main():
    if st.session_state["selected_subtask1"]:
        show_subtask_details(st.session_state["selected_subtask1"])

    elif st.session_state["selected_task1"]:
        task_path = os.path.join(TASKS_DIR, st.session_state["selected_task1"])
        valid_subtasks = find_valid_subfolders(task_path)

        if not valid_subtasks:
            st.warning("No valid subfolders with both files found in this task.")
            if st.button("🔙 Back to Task List"):
                st.session_state["selected_task1"] = None
                st.session_state["selected_subtask1"] = None
                st.session_state["static_flow_state2"] = None
                st.rerun()
        elif len(valid_subtasks) == 1:
            # Skip selection, show the only subtask directly
            st.session_state["selected_subtask1"] = valid_subtasks[0]
            st.session_state["static_flow_state2"] = None
            st.rerun()
        else:
            show_subtask_selector(st.session_state["selected_task1"], valid_subtasks)
            st.session_state["static_flow_state2"] = None

    else:
        show_task_list()


if __name__ == "__main__":
    main()
