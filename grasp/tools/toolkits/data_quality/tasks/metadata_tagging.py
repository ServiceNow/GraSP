import os
import logging
from argparse import Namespace
from grasp.utils import utils
from grasp.internal_tasks.data_quality.metadata_tagging.filter_tags import (
    PipelineConfig,
    extract_instag_stats,
)

logger = logging.getLogger(__name__)


class MetadataTaggingTask:
    """
    A task for tagging metadata in a dataset.

    This task processes input data, applies metadata tagging transformations, and executes
    a tagging pipeline using a task executor. The results are saved to an output file.

    Args:
        input_file (str): Path to the input file containing the dataset.
        output_dir (str): Directory where the output files will be saved.
        num_records (int): Total number of records to process.
        **kwargs: Additional task-specific parameters.
    """

    def __init__(self, input_file: str, output_dir: str, num_records: int, **kwargs):
        self.input_file = input_file
        self.output_dir = output_dir
        self.num_records = num_records
        self.task_params = kwargs

    def execute(self) -> str:
        """
        Executes the metadata tagging task.

        Returns:
            str: Path to the output file containing the results.
        """
        from grasp.core.base_task_executor import BaseTaskExecutor

        args = self._construct_args()
        data_config = self._construct_data_config()
        graph_config_dict = self._load_and_update_graph_config(data_config)
        BaseTaskExecutor(args, graph_config_dict).execute()

        output_file = os.path.join(self.output_dir, "metadata_tagging_output", "output.jsonl")
        if not os.path.exists(output_file):
            output_file = os.path.join(self.output_dir, "metadata_tagging_output", "output.json")

        # Run filter_tags on the output file if it exists
        if os.path.exists(output_file):
            self._run_filter_tags(output_file)
        else:
            logger.warning(f"Output file {output_file} does not exist. Skipping tags normalization and filtering.")

        return output_file

    def _construct_args(self) -> Namespace:
        """
        Constructs the arguments required for the task execution.

        Returns:
            Namespace: A namespace object containing task arguments.
        """
        args = {
            "task": "grasp.internal_tasks.data_quality.metadata_tagging",
            "start_index": 0,
            "num_records": self.num_records,
            "run_name": "metadata_tagging",
            "batch_size": self.task_params.get("batch_size", 25),
            "checkpoint_interval": 100,
            "debug": self.task_params.get("debug", False),
            "output_with_ts": self.task_params.get("output_with_ts", False),
            "output_dir": self.output_dir,
            "oasst": False,
            "quality": False,
        }
        return Namespace(**args)

    def _construct_data_config(self) -> dict:
        """
        Constructs the data configuration for the task.

        Returns:
            dict: A dictionary containing the data configuration.
        """
        return {"source": {"type": "disk", "file_path": self.input_file}}

    def _load_and_update_graph_config(self, data_config: dict) -> dict:
        """
        Loads and updates the graph configuration with the provided data configuration.

        Args:
            data_config (dict): The data configuration to merge into the graph configuration.

        Returns:
            dict: The updated graph configuration.
        """
        graph_config = utils.load_yaml_file(
            filepath=utils.get_file_in_task_dir(
                "grasp.internal_tasks.data_quality.metadata_tagging", "graph_config.yaml"
            )
        )
        transformations = (
            graph_config.get("data_config", {})
            .get("source", {})
            .get("transformations", [])
        )
        graph_config.update({"data_config": data_config})
        graph_config["data_config"]["source"]["transformations"] = transformations
        return graph_config

    def _run_filter_tags(self, input_file: str) -> None:
        """
        Runs the filter_tags.py script on the given input file to normalize instruction tags.

        Args:
            input_file (str): Path to the input file to process.
        """
        try:
            import json

            with open(input_file, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    logger.info(
                        f"Loaded {len(data)} records from {input_file} for tag filtering"
                    )
                except json.JSONDecodeError:
                    f.seek(0)
                    data = [json.loads(line) for line in f if line.strip()]
                    logger.info(
                        f"Loaded {len(data)} records from JSONL file {input_file} for tag filtering"
                    )

            cfg = PipelineConfig()

            logger.info("Running instruction tag filtering and normalization...")
            stats = extract_instag_stats(data, cfg)

            with open(input_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.info(
                f"Tag filtering complete. Stats: {stats['instag_stats']['num_unique_tags']} unique tags after processing"
            )
            logger.info(f"Updated data written back to {input_file}")

        except Exception as e:
            logger.error(f"Error running filter_tags: {e}")
            logger.exception("Filter tags processing failed")
