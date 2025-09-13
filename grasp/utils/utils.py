import importlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union, cast

import yaml  # type: ignore[import-untyped]
from datasets import IterableDataset  # type: ignore[import-untyped]
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    AIMessagePromptTemplate,
    BaseMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from grasp.core.dataset.dataset_config import DataSourceConfig
from grasp.core.dataset.file_handler import FileHandler
from grasp.core.dataset.huggingface_handler import HuggingFaceHandler
from grasp.data_mapper.helper import JSONEncoder
from grasp.logger.logger_config import logger
from grasp.utils import constants


def load_model_config() -> Any:
    """
    Load model configurations from both models.yaml and environment variables.

    The function:
    1. Loads base model configurations from models.yaml
    2. Loads sensitive data (URL and auth token/API key) from environment variables
    3. Combines them, with environment variables taking precedence for sensitive fields

    Environment variable naming convention:
    - GRASP_{MODEL_NAME}_URL: URL for the model (can be a single URL or a pipe-separated list of URLs)
      For multiple URLs, separate them with the LIST_SEPARATOR (|)
      Example: "http://url1.com|http://url2.com|http://url3.com"
    - GRASP_{MODEL_NAME}_TOKEN: Authentication token or API key for the model

    Returns:
        Dict containing combined model configurations
    """
    from grasp.utils.dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv(dotenv_path=".env", override=True)

    # Load base configurations from models.yaml
    base_configs = load_yaml_file(constants.MODEL_CONFIG_YAML)

    # For each model, look for corresponding environment variables
    for model_name, config in base_configs.items():
        # Convert model name to uppercase for environment variable lookup
        env_prefix = f"GRASP_{get_env_name(model_name)}"

        # Look for URL in environment variables
        if url := os.environ.get(f"{env_prefix}_URL"):
            # Check if it contains the list separator (indicating a list of URLs)
            if constants.LIST_SEPARATOR in url:
                # Split by the separator and filter out any empty strings
                url_list = [u for u in url.split(constants.LIST_SEPARATOR) if u]
                config["url"] = url_list
            else:
                # It's a single URL string
                config["url"] = url

        # Look for auth token/API key in environment variables
        if token := os.environ.get(f"{env_prefix}_TOKEN"):
            # Determine whether to use auth_token or api_key based on model_type
            model_type = config.get("model_type", "").lower()

            # OpenAI models use api_key, others use auth_token
            if model_type == "azure_openai":
                config["api_key"] = token
            else:
                config["auth_token"] = token

    return base_configs


def get_updated_model_config(model_config: dict) -> dict:
    """
    Updates a model config dictionary with default values from the grasp config file.

    Args:
        model_config: Dictionary containing model configuration parameters

    Returns:
        Updated dictionary containing model configuration parameters
    """
    config = load_yaml_file(constants.GRASP_CONFIG)
    default_model_config = config.get("model_config", {})
    if "ssl_verify" not in model_config:
        model_config["ssl_verify"] = default_model_config.get("ssl_verify", True)
    if "ssl_cert" not in model_config:
        ssl_cert = default_model_config.get("ssl_cert", None)
        ssl_cert = ssl_cert if ssl_cert and ssl_cert != "None" else None
        model_config["ssl_cert"] = ssl_cert
    return model_config


def get_payload(inference_server: str, key="default") -> Any:
    """Get the payload for the specific inference server using the key.

    JSON format:
    triton -> default/api_based_key -> payload_json/test_payload/response_key
    tgi -> None
    vllm -> None

    Args:
        inference_server: inference server type, as of now we support
            only triton, may extend in future
        key: value to fetch for this payload key, this key defines
            input/output/test payload for an api

    Returns:
        json containing keys payload_json, test_payload, response_key in
        triton flow this returns None if inference type is not defined
        in the configuration file like tgi or vllm
    """
    payload_cfg = {}
    if Path(constants.PAYLOAD_CFG_FILE).exists():
        payload_cfg.update(load_json_file(constants.PAYLOAD_CFG_FILE))
    # get inference server specific config, it can be None for tgi and vllm for now
    inference_server_cfg = payload_cfg.get(inference_server)

    # return the value, if not found return default for a valid inference server(triton), else return none
    return (
        inference_server_cfg.get(key, inference_server_cfg.get("default"))
        if inference_server_cfg
        else None
    )


def get_env_name(name):
    """
    Returns the name normalized for environment variable usage:
    - All whitespace sequences replaced with a single underscore
    - Result is uppercased
    """
    return re.sub(r"\s+", "_", name).upper()


def load_yaml_file(filepath: str) -> Any:
    with open(filepath) as f:
        return yaml.safe_load(f)


def load_json_file(filepath: str) -> Any:
    with open(filepath) as f:
        return json.load(f)


def load_jsonl_file(filepath: str) -> Any:
    with open(filepath, "r") as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]


def extract_and_load_json(text: str) -> Any:
    try:
        match = re.search(r"[\[\n\r\s]*{.+}[\n\r\s\]]*", text, re.DOTALL)
        if not match:
            return None
        json_str = match.group(0)
        return json.loads(json_str)
    except Exception:
        return None


def load_json(text) -> Any:
    object = None
    try:
        object = json.loads(text)
    except Exception:
        pass
    return object


def append_to_json_file(filepath: str, data: list[dict]):
    existing_data = []
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            existing_data = json.load(f)
    existing_data.extend(data)
    with open(filepath, "w") as outfile:
        json.dump(
            existing_data,
            outfile,
            indent=4,
            ensure_ascii=False,
            cls=JSONEncoder,
        )


def append_to_jsonl_file(filepath: str, data: list[dict]):
    with open(filepath, "a") as outfile:
        for entry in data:
            json.dump(
                entry,
                outfile,
                ensure_ascii=False,
                cls=JSONEncoder,
            )
            outfile.write("\n")


def save_json_file(filepath: str, data: list[dict]):
    with open(filepath, "w") as outfile:
        json.dump(
            data,
            outfile,
            indent=4,
            ensure_ascii=False,
            cls=JSONEncoder,
        )


def save_jsonl_file(filepath: str, data: list[dict]):
    with open(filepath, "w") as outfile:
        for entry in data:
            json.dump(
                entry,
                outfile,
                ensure_ascii=False,
                cls=JSONEncoder,
            )
            outfile.write("\n")


def delete_file(filepath: str):
    if os.path.exists(filepath):
        os.remove(filepath)


def _normalize_task_path(task: str) -> str:
    """
    Helper function to normalize task paths.

    Handles both module-style paths and filesystem paths correctly:
    - Module paths: Convert dots to path separators
    - Filesystem paths: Use as-is
    """
    # Check if task is already a filesystem path
    if os.sep in task or (os.altsep and os.altsep in task) or os.path.isabs(task):
        # This is a filesystem path - use it directly
        return task
    else:
        # This is a module-style dotted path - convert dots to path separators
        return task.replace(".", os.sep)


def get_file_in_task_dir(task: str, file: str):
    task_dir = _normalize_task_path(task)
    return os.path.join(task_dir, file)


def get_file_in_dir(dot_walk_path: str, file: str):
    dir_path = "/".join(dot_walk_path.split("."))
    return f"{constants.ROOT_DIR}/{dir_path}/{file}"


def get_dot_walk_path(path: str):
    return ".".join(path.split("/"))


def validate_required_keys(
    required_keys: list[str],
    config: dict,
    config_name: str,
):
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Required keys {missing_keys} are missing in {config_name} configuration")


def get_func_from_str(func_str: str) -> Callable[..., Any]:
    assert not func_str.startswith(".")
    objects = func_str.split(".")
    module = ".".join(objects[:-1])
    method_name = objects[-1]
    obj = getattr(importlib.import_module(module), method_name)
    if not callable(obj):
        raise TypeError(f"Resolved object '{func_str}' is not callable")
    return cast(Callable[..., Any], obj)


def flatten_dict(d, parent_key="", sep="_"):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def deep_update(target: dict, src: dict):
    """
    Recursively update target with src, merging nested dictionaries.

    Args:
        target: The dictionary to update
        src: The dictionary with values to update from

    Examples:
        >>> deep_update({"a": 1, "b": {"c": 2}}, {"b": {"d": 3}})
        {'a': 1, 'b': {'c': 2, 'd': 3}}
    """
    for key, val in src.items():
        if key in target and isinstance(target[key], dict) and isinstance(val, dict):
            # both sides are dicts → recursively merge
            deep_update(target[key], val)
        else:
            # otherwise overwrite or insert
            target[key] = val


def convert_messages_from_chat_format_to_langchain(
    messages: list[dict],
) -> list[BaseMessagePromptTemplate]:
    langchain_messages: list[BaseMessagePromptTemplate] = []
    for message in messages:
        role, content = message["role"], message["content"]
        if role == "user":
            langchain_messages.append(HumanMessagePromptTemplate.from_template(content))
        elif role == "assistant":
            langchain_messages.append(AIMessagePromptTemplate.from_template(content))
        elif role == "system":
            langchain_messages.append(SystemMessagePromptTemplate.from_template(content))
    return langchain_messages


def convert_messages_from_config_to_chat_format(
    messages: list[dict],
) -> list[dict]:
    chat_messages = []
    for message in messages:
        # There will be only key-value pair in the message
        role = next(iter(message))
        chat_messages.append({"role": role, "content": message[role]})
    return chat_messages


def convert_messages_from_langchain_to_chat_format(
    messages: Sequence[BaseMessage],
) -> list[dict]:
    chat_messages = []
    for message in messages:
        if isinstance(message, HumanMessage):
            chat_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            chat_messages.append({"role": "assistant", "content": message.content})
        elif isinstance(message, SystemMessage):
            chat_messages.append({"role": "system", "content": message.content})
    return chat_messages


def extract_pattern(string: str, pattern: str) -> list[str]:
    """
    Extract content matching a given pattern.

    Args:
        string (str): The input string to search.
        pattern (str): The regex pattern to match content.

    Returns:
        list[str]: A list of content matching the pattern, excluding empty matches.
    """
    matches = re.findall(pattern, string)
    return [match for match in matches if match]


def get_models_used(task: str):
    """
    Get model config which are being used in this task
    """
    task = _normalize_task_path(task)
    # Use load_model_config instead of directly loading models.yaml
    all_model_config = load_model_config()
    grasp_yaml = load_yaml_file(os.path.join(task, "graph_config.yaml"))
    nodes = grasp_yaml.get("graph_config", {}).get("nodes", {})
    models_used = set()
    for k, n in nodes.items():
        node_type = n.get("node_type")
        node_state = n.get("node_state")
        if node_state and node_state == "idle":
            continue
        if node_type == "multi_llm":
            models = n.get("models", [])
            for key, model_config in models.items():
                name = model_config.get("name")
                models_used.add(name)
        elif node_type == "llm" or node_type == "agent":
            model = n.get("model")
            name = model.get("name")
            models_used.add(name)
    used_model_config = {}
    for model in models_used:
        used_model_config[model] = all_model_config.get(model)

    return used_model_config


def get_graph_factory(backend: str):
    if backend == "langgraph":
        from grasp.core.graph.langgraph.langgraph_factory import LangGraphFactory

        return LangGraphFactory()
    else:
        raise ValueError(f"{backend} is not a supported backend.")


def get_graph_properties(task_name: Optional[str] = None) -> Any:
    task = task_name or current_task
    if not task:
        logger.error("Current task name is not initialized.")
        return {}

    formatted_task_name = _normalize_task_path(task)
    path = os.path.join(f"{formatted_task_name}/graph_config.yaml")
    yaml_config = load_yaml_file(path)
    return yaml_config.get("graph_config", {}).get("graph_properties", {})


def get_graph_property(key: str, default_value: Any, task_name: Optional[str] = None) -> Any:
    """
    Get the graph property value
    If task_name is None, returns the property value from current task
    """
    props = get_graph_properties(task_name)
    return props.get(key, default_value)


def get_dataset(datasrc: dict) -> Union[list[dict[str, Any]], IterableDataset]:
    """
    Get dataset from HuggingFace dataset or Disk based on yaml configuration(dict).
    """
    ds_type = datasrc.get("type")
    ds_cfg = DataSourceConfig.from_dict(datasrc)
    if ds_type == "hf":
        hf = HuggingFaceHandler(ds_cfg)
        dataset = hf.read()
    elif ds_type == "disk":
        fh = FileHandler(ds_cfg)
        dataset = fh.read()
    else:
        raise ValueError(f"{ds_type} is not supported.")
    return dataset


# Fetch next record from the data source, also maintain sequential call to keep the moving pointer
# this is used in weighted sampler, to read a column values from datasource dynamically in a sequence

# sampler_cache stores the dataset and pointer to read next record
# key is the datasource dictionary in string format to make it unique
# {key: (dataset, pointer)}
sampler_cache: dict[
    str,
    tuple[Union[IterableDataset, list[dict[str, Any]]], int],
] = {}
cache_lock = threading.Lock()


def fetch_next_record(datasrc: dict, column: str):
    key = str(datasrc)
    record: Any = None
    with cache_lock:
        if key in sampler_cache:
            dataset, pointer = sampler_cache[key]
            # during resume, we store only pointer not dataset object
            if dataset is None:
                logger.info(f"Loading dataset again for resume function: {datasrc}")
                dataset = get_dataset(datasrc)
            # read the record
            logger.info(f"Reading {pointer}th index record for sampler.")
            if isinstance(dataset, IterableDataset):
                # NOTE: Avoid using stream in sampler, it will be slow
                index = 0
                for data in dataset:
                    if pointer == index:
                        record = data.get(column)
                        break
                    index += 1
            else:
                # if pointer reached the end, reset to start
                # this is only for full dataset or disk based, cant check with stream
                if pointer >= len(dataset):
                    logger.info(f"Index {pointer} is out of range, resetting to 0.")
                    pointer = 0
                record = dataset[pointer][column]
            # point to next record
            pointer = pointer + 1
        else:
            dataset = get_dataset(datasrc)

            # read first record
            if isinstance(dataset, IterableDataset):
                for data in dataset:
                    record = data.get(column)
                    break
            else:
                record = dataset[0][column]
            # store next record pointer
            pointer = 1
            logger.info("Reading first record(index=0) for sampler.")
        # save the dataset and updated next pointer
        sampler_cache[key] = (dataset, pointer)
    # task is complete, unlock the threads and return
    return record


def get_class_from(cpath: str) -> Any:
    """
    Get the class from the given path.
    If the class is not found in the internal module, it will try to import from the original path.

    Args:
        cpath (str): The path to the class in the format 'module.submodule.ClassName'.
    Returns:
        Any: The class object.
    """
    internal = "grasp.internal." + ".".join(cpath.split(".")[1:])
    try:
        class_name = getattr(
            importlib.import_module(".".join(internal.split(".")[:-1])),
            internal.split(".")[-1],
        )
    except ModuleNotFoundError:
        class_name = getattr(
            importlib.import_module(".".join(cpath.split(".")[:-1])),
            cpath.split(".")[-1],
        )
    return class_name


# backend factory object : singleton
# delayed backend factory initialization to avoid circular imports
_backend_factory = None
_current_task = None


def get_backend_factory():
    """
    Get backend factory instance, creating it if necessary.
    This delays the import to avoid circular dependency issues.
    """
    global _backend_factory
    if _backend_factory is None:
        _backend_factory = get_graph_factory(constants.BACKEND)
    return _backend_factory


class BackendFactoryProxy:
    """
    Proxy object that behaves like the original backend_factory
    but delays initialization to avoid circular imports.
    """

    def __getattr__(self, name):
        return getattr(get_backend_factory(), name)

    def __call__(self, *args, **kwargs):
        return get_backend_factory()(*args, **kwargs)


# Create the proxy instance
backend_factory = BackendFactoryProxy()

# store the current task to access it later to fetch properties
current_task: Optional[str] = None
