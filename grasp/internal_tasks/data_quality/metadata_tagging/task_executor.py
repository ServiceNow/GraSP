import json
from typing import Any

from grasp.core.graph.functions.node_processor import (
    NodePreProcessor,
    NodePostProcessorWithState,
)
from grasp.core.graph.grasp_message import GraspMessage
from grasp.core.graph.grasp_state import GraspState
from grasp.logger.logger_config import logger
import pandas as pd

from grasp.processors.data_transform import DataTransform
from grasp.processors.output_record_generator import BaseOutputGenerator
from grasp.utils import utils


class ExtractCategoryPreProcess(NodePreProcessor):
    def apply(self, state: GraspState) -> GraspState:
        state["messages"] = []
        question = state.get("question")
        if len(question) > 4000:
            question = question[:500] + "\n\n...\n\n" + question[-500:]
        state.update({"question": question})
        return state


class ExtractCategoryPostProcess(NodePostProcessorWithState):
    def apply(self, resp: GraspMessage, state: GraspState) -> GraspState:
        try:
            category = resp.message.content
            category = category if category in state.get("categories") else ""
            state.update({"category": category})
            state["messages"] = []
        except Exception:
            logger.error(
                f"Error in extract_category_postprocess with content {resp.message.content}"
            )
        return state


class ExtractTagsPostProcess(NodePostProcessorWithState):
    def apply(self, resp: GraspMessage, state: GraspState) -> GraspState:
        tags_list_str = resp.message.content
        if tags_list_str is None:
            state["instruction_tags"] = []
            return state
        tag_list = tags_list_str.split(",")
        tag_list = [tag.strip() for tag in tag_list]
        state["instruction_tags"] = tag_list
        state["messages"] = []
        return state


class MetadataTaggingDataTransform(DataTransform):
    """Transform to prepare data for metadata tagging task.
    It extracts the question from each record and initializes necessary fields.
    """

    @property
    def name(self) -> str:
        """Get the name of the metadata tagging data transform.

        Returns:
            str: The identifier 'metadata_tagging_transform'.
        """
        return "metadata_tagging_transform"

    def transform(
        self, data: list[dict[str, Any]], params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Apply the transformation to a list of records.

        Args:
            data (list[dict[str, Any]]): List of dictionary records to transform.
            params (dict[str, Any]): Parameters controlling the transformation.

        Returns:
            list[dict[str, Any]]: Transformed list of dictionary records.
        """
        # Load taxonomy metadata and prepare category descriptions
        sub_tasks = self.get_data(
            utils.get_file_in_dir(
                "internal_tasks.data_quality.metadata_tagging.taxonomy", "taxonomy.json"
            )
        )
        task_category = {cat: sub_tasks[cat]["Description"] for cat in sub_tasks}

        # Extract run parameters
        data_type = params.get("data_type", "sft")

        # Convert input list of dicts to DataFrame
        data_df = pd.DataFrame(data)
        data_df["existing_data"] = data_df.apply(lambda row: row.to_dict(), axis=1)

        # Handle transformation for "sft" type
        if data_type == "sft":

            def extract_question_answer(conversation):
                question = ""
                answer = ""
                for i, turn in enumerate(conversation):
                    if i < len(conversation) - 1:
                        question += f"{turn['role']}: {turn['content']}\n"
                    else:
                        answer += f"{turn['content']}"
                return question.strip(), answer.strip()

            data_df[["question", "response"]] = data_df["conversation"].apply(
                lambda msgs: pd.Series(
                    extract_question_answer(msgs) if msgs else ("", "")
                )
            )

        # Handle transformation for "generic" type
        elif data_type == "generic":
            question_field = params.get("question_field", "")
            response_field = params.get("response_field", "")
            if not question_field and not response_field:
                raise ValueError(
                    "Question and Response fields are missing for generic type"
                )
            data_df["question"] = data_df[question_field]
            data_df["response"] = data_df[response_field]

        # Add taxonomy categories and placeholders
        data_df["categories"] = data_df.apply(lambda row: task_category, axis=1)
        data_df["category"] = ""
        data_df["instruction_tags"] = ""

        # Return transformed records
        return data_df.to_dict(orient="records")

    def get_data(self, file_path):
        with open(file_path) as f:
            data = json.load(f)
        return data


class MetaTaggingOutputGenerator(BaseOutputGenerator):
    @staticmethod
    def build_metadata(category: str, state: GraspState) -> dict:
        return {
            "data_taxonomy": {
                "category": state.get("category", ""),
                "instruction_tags": state.get("instruction_tags", []),
            }
        }
