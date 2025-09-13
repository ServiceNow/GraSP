"""Module for DPO samples data transformation operations."""

from typing import Any

from grasp.logger.logger_config import logger
from grasp.processors.data_transform import DataTransform


class ExtractUserPromptTransform(DataTransform):
    """
    Transform that extracts user_prompt from conversation field.

    This transformer extracts the content of the first user message in the conversation
    and adds it as a separate user_prompt field.
    """

    @property
    def name(self) -> str:
        """Get the name of the transformation.

        Returns:
            str: Unique identifier for this transformation type.
        """
        return "extract_user_prompt"

    def transform(
        self, data: list[dict[str, Any]], params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Apply the transformation to extract user_prompt from conversation.

        Args:
            data (list[dict[str, Any]]): List of dictionary records to transform.
            params (dict[str, Any]): Parameters controlling the transformation.

        Returns:
            list[dict[str, Any]]: Transformed records with user_prompt field.
        """
        transformed_data = []

        for record in data:
            # Create a copy of the record to avoid modifying the original
            transformed_record = record.copy()

            # Extract user_prompt from conversation if it exists
            if "conversation" in record and isinstance(record["conversation"], list):
                user_messages = [
                    msg for msg in record["conversation"] if msg.get("role") == "user"
                ]
                if user_messages:
                    transformed_record["user_prompt"] = user_messages[0].get(
                        "content", ""
                    )
                    logger.info(
                        f"Extracted user_prompt: {transformed_record['user_prompt'][:50]}..."
                    )
                else:
                    logger.warning(
                        f"No user message found in conversation for record {record.get('id', 'unknown')}"
                    )
                    transformed_record["user_prompt"] = ""
            else:
                logger.warning(
                    f"No conversation field found in record {record.get('id', 'unknown')}"
                )
                transformed_record["user_prompt"] = ""

            transformed_data.append(transformed_record)

        return transformed_data


class ExtractResponseScaleTransform(DataTransform):
    """
    Transform that extracts response_scale from conversation field.

    This transformer extracts the content of the first assistant message in the conversation
    and adds it as a separate response_scale field.
    """

    @property
    def name(self) -> str:
        """Get the name of the transformation.

        Returns:
            str: Unique identifier for this transformation type.
        """
        return "extract_response_scale"

    def transform(
        self, data: list[dict[str, Any]], params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Apply the transformation to extract response_scale from conversation.

        Args:
            data (list[dict[str, Any]]): List of dictionary records to transform.
            params (dict[str, Any]): Parameters controlling the transformation.

        Returns:
            list[dict[str, Any]]: Transformed records with response_scale field.
        """
        transformed_data = []

        for record in data:
            # Create a copy of the record to avoid modifying the original
            transformed_record = record.copy()

            # Extract response_scale from conversation if it exists
            if "conversation" in record and isinstance(record["conversation"], list):
                assistant_messages = [
                    msg
                    for msg in record["conversation"]
                    if msg.get("role") == "assistant"
                ]
                if assistant_messages:
                    transformed_record["response_scale"] = assistant_messages[0].get(
                        "content", ""
                    )
                    logger.info(
                        f"Extracted response_scale: {transformed_record['response_scale'][:50]}..."
                    )
                else:
                    logger.warning(
                        f"No assistant message found in conversation for record {record.get('id', 'unknown')}"
                    )
                    transformed_record["response_scale"] = ""
            else:
                logger.warning(
                    f"No conversation field found in record {record.get('id', 'unknown')}"
                )
                transformed_record["response_scale"] = ""

            transformed_data.append(transformed_record)

        return transformed_data


class InitializeStateVariablesTransform(DataTransform):
    """
    Transform that initializes state variables needed for the DPO samples task.

    This transformer adds necessary state variables with default values to ensure
    they persist throughout the task execution.
    """

    @property
    def name(self) -> str:
        """Get the name of the transformation.

        Returns:
            str: Unique identifier for this transformation type.
        """
        return "initialize_state_variables"

    def transform(
        self, data: list[dict[str, Any]], params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Apply the transformation to initialize state variables.

        Args:
            data (list[dict[str, Any]]): List of dictionary records to transform.
            params (dict[str, Any]): Parameters controlling the transformation.

        Returns:
            list[dict[str, Any]]: Transformed records with initialized state variables.
        """
        transformed_data = []

        for record in data:
            # Create a copy of the record to avoid modifying the original
            transformed_record = record.copy()

            # Initialize samples_ratings with an empty list
            transformed_record["samples_ratings"] = []

            transformed_data.append(transformed_record)

        return transformed_data
