import os
from argparse import Namespace
from grasp.utils import utils


class LLMBasedQualityTask:
    """
    A task for evaluating data quality using a judge model-based approach.

    This task processes input data, applies transformations, and executes a quality evaluation
    pipeline using a judge task executor. The results are saved to an output file.

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
        Executes the LLM-based quality task.

        Returns:
            str: Path to the output file containing the results.
        """
        from grasp.core.base_task_executor import BaseTaskExecutor

        args = self._construct_args()
        data_config = self._construct_data_config()
        graph_config_dict = self._load_and_update_graph_config(data_config)

        BaseTaskExecutor(args, graph_config_dict).execute()

        output_file = os.path.join(self.output_dir, "llm_based_quality_output", "output.jsonl")
        if os.path.exists(output_file):
            return output_file
        return os.path.join(self.output_dir, "llm_based_quality_output", "output.json")

    def _construct_args(self) -> Namespace:
        """
        Constructs the arguments required for the task execution.

        Returns:
            Namespace: A namespace object containing task arguments.
        """
        args = {
            "task": "grasp.internal_tasks.data_quality.llm_based",
            "start_index": 0,
            "num_records": self.num_records,
            "run_name": "llm_based_quality",
            "batch_size": self.task_params.get("batch_size", 25),
            "checkpoint_interval": self.task_params.get("checkpoint_interval", 100),
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
        return {
            "source": {
                "type": "disk",
                "file_path": self.input_file,
                "transformations": [
                    {
                        "transform": "grasp.processors.data_transform.AddNewFieldTransform",
                        "params": {
                            "mapping": {
                                "category": self.task_params.get("category", "Generic")
                            }
                        },
                    }
                ],
            }
        }

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
                "grasp.internal_tasks.data_quality.llm_based", "graph_config.yaml"
            )
        )
        transformations = (
            graph_config.get("data_config", {})
            .get("source", {})
            .get("transformations", [])
        )
        graph_config.update({"data_config": data_config})
        graph_config["data_config"]["source"]["transformations"].extend(transformations)
        return graph_config
