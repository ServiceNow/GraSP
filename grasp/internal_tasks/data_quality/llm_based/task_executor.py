import json
from typing import Any

import yaml
import regex
from grasp.logger.logger_config import logger
from grasp.core.graph.grasp_message import GraspMessage
from grasp.core.graph.grasp_state import GraspState
from grasp.core.graph.functions.edge_condition import EdgeCondition
from grasp.core.graph.functions.node_processor import (
    NodePostProcessorWithState,
)

from grasp.processors.data_transform import DataTransform
from grasp.processors.output_record_generator import BaseOutputGenerator
from grasp.utils import utils


def parse_response_as_json(s: str):
    JSON_REGEX_PATTERN = regex.compile(r"\{(?:[^{}]|(?R))*\}")
    try:
        return json.loads(s)
    except json.decoder.JSONDecodeError as e:
        p = JSON_REGEX_PATTERN.search(s)
        if not p:
            logger.error("No json string found: " + e.msg)
            logger.error(s)
            return None
        try:
            return json.loads(p[0])
        except json.decoder.JSONDecodeError as e:
            logger.error("Unable to parse json string: " + e.msg)
            logger.error(s)
            return None


def load_prompt_config(prompt_config_file: str) -> dict:
    with open(prompt_config_file, "r") as file:
        config = yaml.safe_load(file)
    category_prompt_map = config.get("Mapping")
    # small case and remove spaces with underscore
    category_prompt_map = {
        k.lower().replace(" ", "_"): v for k, v in category_prompt_map.items()
    }
    return category_prompt_map


class ConvertToQuestionAnswerTransform(DataTransform):
    """Transform to convert conversation into question-answer pairs.
    Eg: If a conversation is:
    conversation = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you! How can I help?"},
        {"role": "user", "content": "What's the weather today?"},
        {"role": "assistant", "content": "Today's weather is sunny with a high of 75°F."}
    ]
    the output will be:
    Question: "User: Hello, how are you?\nAssistant: I'm fine, thank you! How can I help?\nUser: What's the weather today?"
    Answer: "Assistant: Today's weather is sunny with a high of 75°F."
    """

    @property
    def name(self) -> str:
        """Get the name of the convert to question-answer transform.

        Returns:
            str: The identifier 'convert_to_question_answer'.
        """
        return "convert_to_question_answer"

    def transform(
        self, data: list[dict[str, Any]], params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Convert each record's conversation into question-answer pairs.

        Args:
            data (list[dict[str, Any]]): List of dictionary records to transform.
            params (dict[str, Any]): None

        Returns:
            list[dict[str, Any]]: Transformed records with question-answer pairs.
        """
        transformed_data = []
        for record in data:
            conversation = record.get("conversation", [])
            question, answer = self._convert_to_question_answer(conversation)
            transformed_record = record.copy()
            transformed_record["question"] = question
            transformed_record["answer"] = answer
            transformed_data.append(transformed_record)
        return transformed_data

    @staticmethod
    def _convert_to_question_answer(
        conversation: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """Convert a conversation into question-answer pairs.

        Args:
            conversation (list[dict[str, Any]]): List of dictionary records representing the conversation.

        Returns:
            tuple: A tuple containing the question and answer strings.
        """
        question = ""
        answer = ""
        for i, turn in enumerate(conversation):
            if i < len(conversation) - 1:
                question += f"{turn['role']}: {turn['content']}\n"
            else:
                answer += f"{turn['content']}"

        return question, answer


# --- Edge Condition ---


class DataQualityCategoryCondition(EdgeCondition):
    @staticmethod
    def apply(state: GraspState) -> str:
        # Retrieve the category from state and check against allowed categories
        category = state.get("category", "").lower().replace(" ", "_")
        prompt_config = load_prompt_config(
            utils.get_file_in_dir("internal_tasks.data_quality.llm_based", "prompt_config.yaml")
        )

        if category not in prompt_config:
            logger.warning(f"Category '{category}' not found. Defaulting to 'generic'.")
            return "generic"
        return category


# --- Node Post Processors ---


class DataQualityQuestionQualityPostProcessor(NodePostProcessorWithState):
    def apply(self, response: GraspMessage, state: GraspState) -> GraspState:
        response_str = response.message.content.replace("\\", "\\\\")
        response_json = parse_response_as_json(response_str) or {}
        score_json = {
            "question_quality_score": response_json.get("QUALITY_SCORE", ""),
            "explanation_question_quality": response_json.get(
                "QUALITY_EXPLANATION", ""
            ),
        }
        state.setdefault("scores", {}).update(score_json)
        return state


class GenericPromptPostProcessor(NodePostProcessorWithState):
    def apply(self, response: GraspMessage, state: GraspState) -> GraspState:
        response_str = response.message.content.replace("\\", "\\\\")
        json_obj = parse_response_as_json(response_str) or {}
        state["scores"].update({"response_quality": json_obj})
        # Compute overall quality score if possible
        score_keys = [k for k in json_obj.keys() if "explanation" not in k]
        scores = [json_obj[k] for k in score_keys if json_obj[k] != -1]
        if scores:
            state["scores"]["response_quality"]["overall_score"] = round(
                sum(scores) / len(scores), 2
            )
        return state


# --- Output Generator ---


class DataQualityOutputGenerator(BaseOutputGenerator):
    @staticmethod
    def update_metadata(metadata: dict[str, Any], state: GraspState) -> dict[str, Any]:
        """Update the metadata with the scores from the state.

        Args:
            metadata (dict[str, Any]): The original metadata.
            state (GraspState): The state containing the scores.

        Returns:
            dict[str, Any]: The updated metadata.
        """
        if "scores" in state:
            metadata_llm_based = {
                "quality_characteristics": {"LLM_based": state["scores"]}
            }
            utils.deep_update(metadata, metadata_llm_based)
        return metadata
