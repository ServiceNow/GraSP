import os
import tempfile
from typing import Any, Dict, Optional, Type

import ujson

from grasp.logger.logger_config import logger
from grasp.tools.base_tool import BaseTool
from grasp.tools.registry import register_tool
from grasp.utils import utils
from grasp.utils.dotenv import load_dotenv

load_dotenv()

OASST_REPRESENTATIVE_FIELDS = [
    "message_id",
    "parent_id",
    "root_message_id",
    "role",
    "content",
]

TASK_REGISTRY: dict[str, Type] = {
    "conversation_pretokenization": utils.get_class_from(
        "grasp.tools.toolkits.data_quality.tasks.conversation_pretokenization_task.ConversationPreTokenizationTask"
    ),
    "metadata_tagging": utils.get_class_from(
        "grasp.tools.toolkits.data_quality.tasks.metadata_tagging.MetadataTaggingTask"
    ),
    "llm_based_quality": utils.get_class_from(
        "grasp.tools.toolkits.data_quality.tasks.llm_based_quality.LLMBasedQualityTask"
    ),
    "data_characteristics": utils.get_class_from(
        "grasp.tools.toolkits.data_quality.tasks.data_characteristics.DataCharacteristicsTask"
    ),
    "language_tagging": utils.get_class_from(
        "grasp.tools.toolkits.data_quality.tasks.language_tagging.LanguageTaggingTask"
    ),
    "lexical_diversity": utils.get_class_from(
        "grasp.tools.toolkits.data_quality.tasks.ttr_tagging.TTRTaggingTask"
    ),
    "ppl_score": utils.get_class_from(
        "grasp.tools.toolkits.data_quality.tasks.ppl_ifd_scorer.PPLInferenceTask"
    ),
    "reward_score": utils.get_class_from(
        "grasp.tools.toolkits.data_quality.tasks.reward_score.RewardScoringTask"
    ),
}


@register_tool
class DataQuality(BaseTool):
    name = "data_quality"
    description = "Processes datasets to add quality metadata"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.task_list = self.config.get("tasks", [])
        self.parent_output_file_format = "conversation"
        self.num_records_total = 0
        self.input: Optional[str] = None
        self.parent_output_file: Optional[str] = None
        self.output_dir: Optional[str] = None

    def process(self, input_path: Optional[str], output_path: Optional[str]) -> str:
        logger.info("====================================")
        logger.info(" STARTING DATA QUALITY TAGGING ")
        logger.info("====================================")

        if not input_path:
            raise ValueError("input_path must be provided")
        if not output_path:
            raise ValueError("output_path must be provided")

        logger.info(f"Running DataQualityProcessor on: {input_path}")
        self.parent_output_file = input_path

        if self.parent_output_file is None:
            raise ValueError("parent_output_file is not set")
        self.get_data_quality_input_file(self.parent_output_file)
        self.execute_data_quality_tasks()
        self.merge_results(input_path, output_path)

        logger.info("====================================")
        logger.info(" DATA QUALITY TAGGING COMPLETED ")
        logger.info("====================================")

        return output_path

    def execute_data_quality_tasks(self) -> None:
        if self.input is None:
            raise ValueError("Input path for data quality tasks is not set.")
        input_file: str = self.input
        if not os.path.exists(input_file):
            raise ValueError(f"Input file {input_file} does not exist.")

        for task_config in self.task_list:
            task_name = task_config.get("name")
            task_params = task_config.get("params", {})
            task_cls = TASK_REGISTRY.get(task_name)

            if not task_cls:
                raise ValueError(f"Task '{task_name}' is not registered.")

            logger.info(f"Executing task: {task_name}")
            try:
                task_instance = task_cls(
                    input_file=self.input,
                    output_dir=self.output_dir,
                    num_records=self.num_records_total,
                    **task_params,
                )
                task_output = task_instance.execute()
                if task_output:
                    self.input = task_output
                else:
                    logger.warning(f"Task {task_name} returned no output; reusing previous input.")
            except Exception as e:
                logger.error(f"Task {task_name} failed: {e}", exc_info=True)
                if self.config.get("skip_failed_tasks", True):
                    logger.info("Continuing with next task...")
                else:
                    raise e

    def merge_results(self, input_path: str, output_path: str) -> None:
        if self.input is None:
            raise ValueError("Intermediate input path is not set before merging results.")
        if self.parent_output_file is None:
            raise ValueError("Parent output file path is not set before merging results.")

        if self.input.endswith(".json"):
            quality_data = utils.load_json_file(self.input)
        elif self.input.endswith(".jsonl"):
            quality_data = utils.load_jsonl_file(self.input)
        else:
            raise ValueError("Unsupported input format.")

        if self.parent_output_file.endswith(".jsonl"):
            original_data = utils.load_jsonl_file(self.parent_output_file)
        elif self.parent_output_file.endswith(".json"):
            original_data = utils.load_json_file(self.parent_output_file)
        else:
            raise ValueError("Unsupported parent file format.")

        quality_lookup = {record["id"]: record for record in quality_data}

        if self.parent_output_file_format == "oasst":
            for original in original_data:
                q_record = quality_lookup.get(original.get("message_id"))
                if not q_record:
                    continue

                metadata = q_record.get("metadata", {})
                taxonomy = metadata.get("data_taxonomy", {})

                original.setdefault("quality", {}).update(
                    utils.flatten_dict(
                        {"quality_characteristics": metadata.get("quality_characteristics", {})}
                    )
                )
                original.setdefault("data_characteristics", {}).update(
                    metadata.get("data_characteristics", {})
                )
                original.setdefault("categories", []).append(taxonomy.get("category", ""))
                original["instruction_tags"] = taxonomy.get("instruction_tags", [])
                original["languages"] = metadata.get("languages", [])
        else:
            for original in original_data:
                q_record = quality_lookup.get(original.get("id"))
                if q_record:
                    original["quality_metadata"] = q_record.get("metadata", {})

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if output_path.endswith(".jsonl"):
            utils.save_jsonl_file(output_path, original_data)
        elif output_path.endswith(".json"):
            utils.save_json_file(output_path, original_data)
        else:
            raise ValueError("Unsupported output file format.")

        logger.info("Merged data quality results saved to %s", output_path)

    def get_data_quality_input_file(self, parent_output_file: str) -> None:
        if parent_output_file.endswith(".json"):
            data = utils.load_json_file(parent_output_file)
        elif parent_output_file.endswith(".jsonl"):
            data = utils.load_jsonl_file(parent_output_file)
        else:
            raise ValueError(f"Unsupported file format: {parent_output_file}")

        if all(field in data[0] for field in OASST_REPRESENTATIVE_FIELDS):
            self.construct_conversation_tree(data)
            self.parent_output_file_format = "oasst"
        else:
            self.output_dir = tempfile.TemporaryDirectory().name
            os.makedirs(self.output_dir, exist_ok=True)
            self.input = os.path.join(self.output_dir, "conversation.jsonl")
            with open(self.input, "w") as f:
                for record in data:
                    f.write(ujson.dumps(record) + "\n")
            self.num_records_total = len(data)

    def construct_conversation_tree(self, data: list[Dict[str, Any]]) -> None:
        from collections import defaultdict

        node_map = {}
        children_map = defaultdict(set)
        root_ids = set()
        formatted_conversations = []

        for record in data:
            message_id = record["message_id"]
            parent_id = record["parent_id"]
            node_map[message_id] = record
            if parent_id is None:
                root_ids.add(message_id)
            else:
                children_map[parent_id].add(message_id)

        def dfs_path(path: list[str], visited: set[str]) -> None:
            current_id = path[-1]
            if current_id in visited:
                return
            visited.add(current_id)

            current_node = node_map.get(current_id)
            if not current_node:
                return

            if current_node["role"] == "assistant":
                conversation = [
                    {"role": node_map[mid]["role"], "content": node_map[mid]["content"]}
                    for mid in path
                    if mid in node_map
                ]
                if conversation:
                    formatted_conversations.append({"id": current_id, "conversation": conversation})

            for child_id in children_map.get(current_id, set()):
                if child_id not in visited:
                    dfs_path(path + [child_id], visited)

        visited: set[str] = set()
        for root_id in root_ids:
            if root_id not in visited:
                dfs_path([root_id], visited)

        self.output_dir = tempfile.TemporaryDirectory().name
        os.makedirs(self.output_dir, exist_ok=True)
        self.input = os.path.join(self.output_dir, "split_conversation.jsonl")
        with open(self.input, "w") as f:
            for convo in formatted_conversations:
                f.write(ujson.dumps(convo) + "\n")
        self.num_records_total = len(formatted_conversations)
