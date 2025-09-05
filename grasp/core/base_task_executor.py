import copy
import hashlib
import json
import ast
import types
from abc import ABC
from datetime import datetime
from typing import Any, Union, Optional
import datasets
import numpy as np
from PIL import Image
import os
from langgraph.graph import StateGraph

from grasp.tools.toolkits.data_quality.processor import DataQuality
from grasp.core.dataset.dataset_config import (
    DataSourceConfig,
    OutputConfig,
    DataSourceType,
    OutputType,
)
from grasp.core.dataset.dataset_processor import DatasetProcessor
from grasp.core.dataset.file_handler import FileHandler
from grasp.core.dataset.huggingface_handler import HuggingFaceHandler
from grasp.core.graph.graph_config import GraphConfig
from grasp.core.graph.grasp_state import GraspState
from grasp.core.graph.langgraph.graph_builder import LangGraphBuilder
from grasp.logger.logger_config import logger
from grasp.processors.output_record_generator import BaseOutputGenerator
from grasp.utils import utils, constants


class BaseTaskExecutor(ABC):
    def __init__(self, args: Any, graph_config_dict: Optional[dict] = None):
        self.args = args
        self.task_name = args.task
        logger.info(f"Loading graph config for task {self.task_name}")
        self.output_dir = args.output_dir
        self.source_config: Optional[DataSourceConfig] = None
        self.output_config: Optional[OutputConfig] = None

        config_file_path = utils.get_file_in_task_dir(
            self.task_name, "graph_config.yaml"
        )
        self.config = graph_config_dict or utils.load_yaml_file(
            filepath=config_file_path
        )

        data_config = self.config.get("data_config", {})
        config_resumable = data_config.get("resumable", False)
        self.id_column = data_config.get("id_column", {})

        self.resumable = self._configure_resume_behavior(args, config_resumable)

        self.dataset = self.init_dataset()
        output_transform_args = {"oasst": args.oasst, "quality": args.quality}
        self.graph_config = GraphConfig(
            utils.get_file_in_task_dir(self.task_name, "graph_config.yaml"),
            self.dataset,
            output_transform_args,
        )
        self.output_generator: Optional[BaseOutputGenerator] = (
            self._init_output_generator()
        )

    @staticmethod
    def _configure_resume_behavior(args: Any, config_resumable: bool) -> bool:
        """
        Configure resumable behavior based on command-line arguments and configuration

        Args:
            args: Command-line arguments
            config_resumable: Whether resumable is enabled in the configuration

        Returns:
            bool: Whether resumable execution is enabled
        """
        if hasattr(args, "resume") and args.resume is not None:
            resumable = (
                args.resume
                if isinstance(args.resume, bool)
                else ast.literal_eval(str(args.resume))
            )
            logger.info(
                f"Resumable execution {'enabled' if resumable else 'disabled'} by command-line argument"
            )
            return resumable
        else:
            if config_resumable:
                logger.info("Resumable execution enabled by configuration")
            return config_resumable

    def _fetch_variable_value(self, value, config):
        """
        It updates direct string value, dictionary or list having value starts with $

        Args:
            value: to be parsed and replace $ path from config
            config: complete config dictionary
        """
        # process a string value if it starts with $, else just return the value
        if isinstance(value, str) and value.startswith("$"):
            json_node_keys = value[1:].split(".")
            json_node = config
            # recursively reach to the leave node and get the value
            for key in json_node_keys:
                index = -1
                # if it has subscript operator, json_node should be an array
                if "[" in key and key.endswith("]"):
                    key, index_str = key.split("[")
                    try:
                        index = int(index_str.rstrip("]"))
                    except ValueError:
                        logger.error(f"Invalid index: {index_str}")
                        raise
                try:
                    json_node = json_node[key]
                except KeyError:
                    logger.error(f"Key '{key}' not found in config path: {value}")
                    raise
                if index >= 0:
                    json_node = json_node[index]
            return json_node
        elif isinstance(value, dict):
            # update each value of the dictionary and return
            return {k: self._fetch_variable_value(v, config) for k, v in value.items()}
        elif isinstance(value, list):
            # update each value of the list
            return [self._fetch_variable_value(v, config) for v in value]
        else:
            # just return the value as it is
            return value

    def _process_static_variables(self, output_config: dict, config: dict) -> dict:
        """
        Process the variable with $, they are the static values from the config(dict path)
        For example: $data_config.source.repo_id
        It should be path from root, it also supports subscript operator for array
        """
        output_map = output_config.get("output_map")
        if output_map is None:
            return output_config
        for k, v in output_map.items():
            value = v.get("value")
            # replace the final value
            if value:
                output_map[k]["value"] = self._fetch_variable_value(value, config)
        return output_config

    def _init_output_generator(self) -> Optional[BaseOutputGenerator]:
        """
        Check if there's an 'output_config' block in the top-level config (graph_config.yaml).
        If present, check if it has 'generator'. If present, try to import & instantiate it.
        Otherwise, return None.

        Returns:
            Optional[BaseOutputGenerator]: The output generator object
        """
        config = self.graph_config.config
        output_config = config.get("output_config", {})
        if not output_config:
            return None
        output_config = self._process_static_variables(output_config, config)

        if "output_map" in output_config and not output_config.get("generator"):
            logger.info("Using default output generator with output_map")
            return BaseOutputGenerator(output_config)

        gen_class_str = output_config.get("generator", "")
        if not gen_class_str:
            return None

        try:
            generator_cls = utils.get_func_from_str(gen_class_str)
            output_generator = generator_cls(output_config)
            logger.info(f"Initialized output generator: {gen_class_str}")
            return output_generator
        except Exception as e:
            logger.error(
                f"Could not initialize output generator '{gen_class_str}': {e}"
            )
            return None

    # Initialize and return the langgraph StateGraph object for the task
    def init_graph(self) -> StateGraph:
        graph_builder = LangGraphBuilder(self.graph_config)
        return graph_builder.build()

    # Initialize and return the dataset for the task
    def init_dataset(
        self,
    ) -> Union[list[dict], datasets.Dataset, datasets.IterableDataset]:
        data_config = self.config.get("data_config", {})

        # Configure output
        self._configure_sink(data_config)

        # Configure and load source data
        data = self._load_source_data(data_config)

        # Infer features for IterableDataset if they're missing/unknown
        if isinstance(data, datasets.IterableDataset):
            features = self._get_or_infer_features(data)
            return self.assign_ids(data, features=features)
        else:
            return self.assign_ids(data)

    def _get_or_infer_features(
        self, dataset: datasets.IterableDataset
    ) -> datasets.Features:
        """Get existing features or infer them if missing/unknown."""
        features = dataset.features or datasets.Features()

        # Only infer if features are empty (Unknown case)
        if len(features) == 0:
            logger.info("Features are Unknown/empty, inferring from sample...")
            features = self._infer_features_from_sample(dataset)
        else:
            logger.info("Using existing dataset features")
            features = features.copy()

        return features

    def _infer_features_from_sample(
        self, dataset: datasets.IterableDataset
    ) -> datasets.Features:
        """Infer dataset features by sampling the first record - only called when needed."""
        features = datasets.Features()

        try:
            sample_record = next(iter(dataset.take(1)))
            for field_name, field_value in sample_record.items():
                features[field_name] = self._infer_field_type(field_name, field_value)
        except (StopIteration, Exception) as e:
            logger.warning(f"Could not sample a record to determine features: {e}")

        return features

    def _infer_field_type(
        self, field_name: str, field_value: Any
    ) -> datasets.Features.type:
        """Infer the appropriate datasets feature type for a field value."""
        if isinstance(field_value, str):
            return datasets.Value("string")
        elif isinstance(field_value, bool):
            return datasets.Value("bool")
        elif isinstance(field_value, int):
            return datasets.Value("int32")
        elif isinstance(field_value, float):
            return datasets.Value("float32")
        elif isinstance(field_value, Image.Image):
            return datasets.Image()
        elif isinstance(field_value, dict):
            # Check for special cases like Audio, Image, etc.
            if isinstance(field_value.get("array"), np.ndarray) and isinstance(
                field_value.get("sampling_rate"), (int, float)
            ):
                # HuggingFace audio dict
                return datasets.Audio()
            elif (
                "path" in field_value
                and "bytes" in field_value
                and ("audio" in field_name.lower() or "image" in field_name.lower())
            ):
                return (
                    datasets.Audio()
                    if "audio" in field_name.lower()
                    else datasets.Image()
                )
            elif field_value:
                return datasets.Features(
                    {
                        key: self._infer_field_type(f"{field_name}.{key}", val)
                        for key, val in field_value.items()
                    }
                )
            else:
                # Empty dictionary
                return datasets.Features({})
        elif isinstance(field_value, (list, tuple)):
            if field_value:  # Non-empty list
                first_item = field_value[0]
                if isinstance(first_item, dict):
                    # List of dictionaries
                    return datasets.Sequence(
                        datasets.Features(
                            {
                                key: self._infer_field_type(
                                    f"{field_name}[].{key}", value
                                )
                                for key, value in first_item.items()
                            }
                        )
                    )
                elif isinstance(first_item, (list, tuple)):
                    # Potential multidimensional array
                    array = np.array(field_value)
                    shape = array.shape
                    dtype = str(array.dtype)
                    if array.ndim == 2:
                        return datasets.Array2D(shape=shape, dtype=dtype)
                    elif array.ndim == 3:
                        return datasets.Array3D(shape=shape, dtype=dtype)
                    elif array.ndim == 4:
                        return datasets.Array4D(shape=shape, dtype=dtype)
                    elif array.ndim == 5:
                        return datasets.Array5D(shape=shape, dtype=dtype)
                    else:
                        return datasets.Sequence(
                            self._infer_field_type(f"{field_name}[]", first_item)
                        )
                else:
                    # List of primitives
                    return datasets.Sequence(
                        self._infer_field_type(f"{field_name}[]", first_item)
                    )
            else:
                # Empty list - default to sequence of strings
                return datasets.Sequence(datasets.Value("string"))
        elif hasattr(field_value, "item") and isinstance(
            field_value.item(), (int, float, bool, str)
        ):
            return self._infer_field_type(field_name, field_value.item())
        elif isinstance(field_value, types.NoneType):
            return datasets.Value("null")
        else:
            logger.warning(
                f"Unsupported field type {type(field_value)} for field {field_name}. Defaulting to string."
            )
            return datasets.Value("string")

    def _configure_sink(self, data_config: dict) -> None:
        """Configure the sink settings from data config"""
        sink_config = data_config.get("sink")
        if sink_config:
            self.output_config = OutputConfig.from_dict(sink_config)

    def _load_source_data(
        self, data_config: dict
    ) -> Union[list[dict], datasets.Dataset, datasets.IterableDataset]:
        """Load data from the configured source"""
        source_config = data_config.get("source")
        if not source_config:
            logger.info("No data source configured. Generating empty dataset with IDs.")
            return self._generate_empty_dataset()

        self.source_config = DataSourceConfig.from_dict(source_config)
        reader = self._get_data_reader()
        full_data = self._read_data(reader)

        # Apply transformations to the dataset
        full_data = self.apply_transforms(self.source_config, full_data)

        if isinstance(full_data, list):
            assert len(full_data) > 0, "No data found in the dataset"
        elif not isinstance(full_data, datasets.IterableDataset):
            raise ValueError(
                f"Unsupported data format: {type(full_data)}. Expected list or IterableDataset."
            )

        return full_data

    def _generate_empty_dataset(self) -> list[dict]:
        """Generate empty dataset with specified number of records"""
        num_records = self.args.num_records
        logger.info(f"Generating {num_records} empty records")
        return [{} for _ in range(num_records)]

    def _get_data_reader(self) -> Union[HuggingFaceHandler, FileHandler]:
        """Get appropriate data reader based on source type"""
        if self.source_config.type == DataSourceType.HUGGINGFACE:
            return HuggingFaceHandler(self.source_config)
        elif self.source_config.type == DataSourceType.DISK_FILE:
            return FileHandler(self.source_config)
        else:
            raise ValueError(f"Unsupported data source type: {self.source_config.type}")

    def _read_data(
        self, reader
    ) -> Union[list[dict], datasets.Dataset, datasets.IterableDataset]:
        """Read data from the configured source using the provided reader"""
        try:
            if self.source_config.shard is None:
                return reader.read()
            else:
                full_data = []
                shard_files = reader.get_files()
                for shard_path in shard_files:
                    data = reader.read(shard_path)
                    full_data.extend(data)
                return full_data
        except Exception as e:
            logger.error(f"Error reading data: {str(e)}")
            raise RuntimeError(f"Failed to read data: {str(e)}") from e

    def apply_transforms(
        self,
        source_config: DataSourceConfig,
        data: Union[list[dict[str, Any]], datasets.IterableDataset],
    ) -> Union[list[dict[str, Any]], datasets.IterableDataset]:
        """
        Apply each transformation in source_config.transformations
        (the default_transformations from config are applied first)
          - If `data` is a list of dicts, run transform(list, params) in‐memory.
          - If `data` is an IterableDataset, apply each transform one record at a time.
        """
        config = utils.load_yaml_file(constants.GRASP_CONFIG)
        default_cfgs = (config or {}).get("default_transformations", [])
        custom_cfgs = [
            cfg.model_dump() for cfg in (source_config.transformations or [])
        ]
        all_transforms = default_cfgs + custom_cfgs

        if not all_transforms:
            return data

        logger.info(
            f"Applying {len(all_transforms)} transforms in order (Default: {len(default_cfgs)}, Custom: {len(custom_cfgs)})"
        )

        if isinstance(data, list):
            return self._apply_transform_sequence(all_transforms, data)

        elif isinstance(data, datasets.IterableDataset):
            return self._apply_transforms_iterable(all_transforms, data)

        else:
            raise TypeError(f"Unsupported dataset type: {type(data)}")

    def _get_transform_instances(
        self, transform_cfgs: list[dict[str, Any]]
    ) -> list[tuple[Any, dict[str, Any]]]:
        """
        Get instances of transformation functions based on the provided configuration.
        """
        instances = []
        for cfg in transform_cfgs:
            if "transform" not in cfg:
                raise ValueError(
                    f"Missing 'transform' key in transformation config: {cfg}"
                )
            transform_fn = utils.get_func_from_str(cfg["transform"])()
            params = cfg.get("params", {})
            instances.append((transform_fn, params))
        return instances

    def _apply_transform_sequence(
        self, transform_cfgs: list[dict[str, Any]], data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Apply a sequence of transformations to a list of records.
        """
        transform_instances = self._get_transform_instances(transform_cfgs)
        data_input = copy.deepcopy(data)
        for instance, params in transform_instances:
            logger.info(f"Applying transform: {instance.name}")
            data_input = instance.transform(data_input, params)
        return data_input

    def _apply_transforms_iterable(
        self, transform_cfgs: list[dict[str, Any]], data: datasets.IterableDataset
    ) -> datasets.IterableDataset:
        """
        Apply a sequence of transformations to an IterableDataset.
        """
        transform_instances = self._get_transform_instances(transform_cfgs)

        def _apply_transform_record(record: dict[str, Any]) -> dict[str, Any]:
            for instance, params in transform_instances:
                record = instance.transform([record], params)[
                    0
                ]  # Apply transform to single record
            return record

        return data.map(_apply_transform_record)

    def add_id(self, record) -> dict[str, Any]:
        """
        Add an "id" to the record. If the id_column is specified, use that value.
        If the id_column is not specified and the record does not have an "id", generate a hash of the record.

        Args:
            record: The input record (dict) to which the id will be added.
        Returns:
            dict: The record with the added "id" field.
        """

        if self.id_column:
            id_column = self.id_column
            if id_column in record and record[id_column] is not None:
                record["id"] = record[id_column]
            return record
        elif "id" in record and record["id"] is not None:
            return record
        else:
            record_for_hash = record.copy()
            record_str = json.dumps(record_for_hash, sort_keys=True)
            content_hash = hashlib.sha256(record_str.encode()).hexdigest()
            record["id"] = content_hash
            return record

    # Function to assign "id" to every record of full_data
    def assign_ids(self, full_data, features: Optional[datasets.Features] = None):
        """Assign unique IDs to dataset records. Features should be inferred beforehand."""
        if isinstance(full_data, datasets.IterableDataset):
            if features is None:
                raise ValueError("Features must be provided for IterableDataset")

            if "id" not in features:
                features = features.copy()
                features["id"] = datasets.Value("string")

            return full_data.map(self.add_id, features=features)

        # Handle list/dict data
        if full_data and full_data[0].get("id"):
            return full_data

        for i in range(len(full_data)):
            record = full_data[i]
            if isinstance(record, dict):
                full_data[i] = self.add_id(record)
            elif isinstance(record, list):
                for j in range(len(record)):
                    if isinstance(record[j], dict):
                        record[j] = self.add_id(record[j])
            else:
                raise ValueError(
                    f"Unsupported data format: {type(record)}. Expected dict or list."
                )

        return full_data

    # Mapping function to convert input record to the format expected by the graph
    def input_record_generator(self, record: dict[str, Any]) -> dict[str, Any]:
        return record

    # Mapping function to convert the output state of the graph to the output results format
    def output_record_generator(self, state: GraspState) -> GraspState:
        """
        Convert the output state of the graph to the output results format.

        Args:
            state: GraspState object

        Returns:
            GraspState: The output state
        """
        if self.output_generator:
            return self.output_generator.generate(state)
        else:
            return state

    def execute(self):
        graph = self.init_graph()
        compiled_graph = graph.compile()
        logger.info("Graph compiled successfully")
        logger.info("\n" + compiled_graph.get_graph().draw_ascii())

        ts_suffix = (
            ""
            if not self.args.output_with_ts
            else "_" + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        )

        num_records_total = self.args.num_records
        if isinstance(self.dataset, list):
            num_records_total = (
                min(self.args.num_records, len(self.dataset))
                if self.args.num_records
                else len(self.dataset)
            )

        metadata_path = utils.get_file_in_task_dir(self.args.task, "metadata.json")

        existing_output_file = None

        if self.resumable and os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    if metadata.get("task_name") == self.task_name:
                        existing_output_file = metadata.get("output_file")
            except Exception as e:
                logger.warning(f"Error reading metadata file: {e}")

        if (
            self.resumable
            and existing_output_file
            and os.path.exists(existing_output_file)
        ):
            out_file = existing_output_file
            out_file_type = os.path.splitext(existing_output_file)[1].lstrip(".")
            logger.info(f"Resuming with existing output file: {out_file}")
        else:
            # output file type is jsonl if num_records_total > 25k
            # since the output file will also be big and its efficient to append to jsonl
            out_file_type = "jsonl" if num_records_total > 25000 else "json"
            run_name_prefix = f"{self.args.run_name}_" if self.args.run_name else ""
            # Create a subdirectory for outputs with the name format
            output_subdir_name = f"{run_name_prefix}output{ts_suffix}"
            
            if self.output_dir:
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                
                output_subdir_path = os.path.join(self.output_dir, output_subdir_name)
                if not os.path.exists(output_subdir_path):
                    os.makedirs(output_subdir_path)
                
                out_file = os.path.join(output_subdir_path, f"output.{out_file_type}")
            else:
                # Get the task directory path
                task_dir = utils.get_task_dir(self.task_name)

                output_subdir_path = os.path.join(task_dir, output_subdir_name)
                if not os.path.exists(output_subdir_path):
                    os.makedirs(output_subdir_path)

                out_file = os.path.join(output_subdir_path, f"output.{out_file_type}")
        if not self.resumable and os.path.exists(out_file):
            logger.info(
                f"Deleting existing output file since resumable=False: {out_file}"
            )
            utils.delete_file(out_file)

            if os.path.exists(metadata_path):
                logger.info(f"Removing metadata file: {metadata_path}")
                utils.delete_file(metadata_path)

        if self.args.start_index != 0:
            logger.info(
                f"Creating a subset of the dataset starting from index {self.args.start_index}"
            )
            if isinstance(self.dataset, list):
                self.dataset = self.dataset[self.args.start_index :]
            else:
                self.dataset = self.dataset.skip(self.args.start_index)

        if self.args.num_records:
            logger.info(f"Setting target to process {self.args.num_records} records")
            if isinstance(self.dataset, list):
                self.dataset = self.dataset[: self.args.num_records]

        dataset_processor = DatasetProcessor(
            self.dataset,
            compiled_graph,
            self.graph_config,
            out_file,
            num_records_total=num_records_total,
            start_index=self.args.start_index,
            batch_size=self.args.batch_size,
            checkpoint_interval=self.args.checkpoint_interval,
            debug=self.args.debug,
            input_record_generator=self.input_record_generator,
            output_record_generator=self.output_record_generator,
            resumable=self.resumable,
            task_name=self.task_name,
        )
        dataset_processor.process_and_store_results()

        if "data_quality" in self.graph_config.config.get("output_config", {}):
            logger.info("Performing data quality checks")
            data_quality_processor = DataQuality(
                self.graph_config.config["output_config"].get("data_quality", {})
            )
            data_quality_processor.process(input_path=out_file, output_path=out_file)

        # Write to sink if configured
        if self.output_config and dataset_processor.is_valid_schema:
            try:
                with open(out_file, "r") as f:
                    data = (
                        json.load(f)
                        if out_file_type == "json"
                        else [json.loads(line) for line in f]
                    )
                if self.output_config.type == OutputType.HUGGINGFACE:
                    writer = HuggingFaceHandler(
                        source_config=self.source_config,
                        output_config=self.output_config,
                    )
                    writer.write(data)
                else:
                    writer = FileHandler(
                        source_config=self.source_config,
                        output_config=self.output_config,
                    )
                    writer.write(data, path=self.output_config.file_path)
                logger.info(
                    f"Successfully wrote output to sink: {self.output_config.type.value}, {self.output_config.model_dump()}"
                )
            except Exception as e:
                logger.error(f"Error writing to sink: {e}")

        if dataset_processor.resume_manager:
            dataset_processor.resume_manager.force_save_state(is_final=True)


class DefaultTaskExecutor(BaseTaskExecutor):
    """
    A universal executor for tasks that only need the YAML config
    and do NOT define their own TaskExecutor class.
    If the user doesn't define grasp.tasks.<task_name>.task_executor.TaskExecutor,
    we fall back to this class by default.
    """

    def __init__(self, args):
        super().__init__(args)
        logger.info("Using DefaultTaskExecutor for task: %s", self.task_name)
