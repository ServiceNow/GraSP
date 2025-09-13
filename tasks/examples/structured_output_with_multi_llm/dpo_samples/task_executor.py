import json
from typing import Any, Dict, List


from grasp.core.graph.functions.edge_condition import EdgeCondition
from grasp.core.graph.functions.node_processor import (
    NodePostProcessor,
    NodePreProcessor,
)
from grasp.core.graph.grasp_state import GraspState
from grasp.logger.logger_config import logger
from grasp.processors.output_record_generator import BaseOutputGenerator
from grasp.utils import constants, utils


class GenerateSamplesPreProcessor(NodePreProcessor):
    """
    Pre-processor for the generate_samples node.
    Initializes state variables and prepares for model response collection.
    """

    def apply(self, state: GraspState) -> GraspState:
        """
        Initialize state variables needed for the task.

        Args:
            state: The current graph state

        Returns:
            Updated graph state
        """
        # Extract user prompt from conversation in the state
        if "user_prompt" not in state and "conversation" in state:
            user_messages = [
                msg for msg in state["conversation"] if msg["role"] == "user"
            ]
            if user_messages:
                state["user_prompt"] = user_messages[0]["content"]

        # Extract response_scale from conversation in the state
        if "response_scale" not in state and "conversation" in state:
            assistant_messages = [
                msg for msg in state["conversation"] if msg["role"] == "assistant"
            ]
            if assistant_messages:
                state["response_scale"] = assistant_messages[0]["content"]

        if "taxonomy" not in state and "taxonomy" in state:
            state["taxonomy"] = state["taxonomy"]

        # Safety check - initialize samples_ratings if not present
        if "samples_ratings" not in state:
            logger.warning(
                "samples_ratings not found in state, initializing as empty list"
            )
            state["samples_ratings"] = []

        return state


class RateSamplesPreProcessor(NodePreProcessor):
    """
    Pre-processor for the rate_samples node.
    Prepares model responses for rating.
    """

    def apply(self, state: GraspState) -> GraspState:
        """
        Format model responses for rating.

        Args:
            state: The current graph state

        Returns:
            Updated graph state
        """
        asst_responses = ""

        # Get the most recent model responses
        model_responses = state["model_responses"]
        logger.info(f"RateSamplesPreProcessor - Got model_responses: {model_responses}")

        # Convert model_responses to a dictionary for easier processing
        model_responses_dict = {}

        # Check the structure of model_responses and convert accordingly
        if isinstance(model_responses, list):
            # Each item in the list contains model-response pairs
            for item in model_responses:
                for key, value in item.items():
                    model_responses_dict[key] = value
        elif isinstance(model_responses, dict):
            # It's already a dict with model names as keys
            model_responses_dict = model_responses

        logger.info(
            f"RateSamplesPreProcessor - Converted to model_responses_dict: {model_responses_dict}"
        )

        # Store the current model responses in the state for later reference
        state["current_model_responses"] = model_responses_dict

        # Include scale response for judgment only once
        if len(state.get("samples_ratings", [])) == 0 and "response_scale" in state:
            model_responses_dict["scale"] = state["response_scale"]

        # Format each model response for the judge to evaluate
        for model, response in model_responses_dict.items():
            content = self._extract_content(response)
            asst_responses += (
                f"[Start of {model} Answer]\n{content}\n[End of {model} Answer]\n\n"
            )

        # Set assistant_responses for the rating node
        state["assistant_responses"] = asst_responses
        logger.info(
            f"RateSamplesPreProcessor - Set assistant_responses: {asst_responses[:100]}..."
        )

        return state

    def _extract_content(self, response: Any) -> str:
        """
        Extract content from structured output for display to the judge.

        Args:
            response: The model response, could be in various formats

        Returns:
            Extracted content as string
        """
        try:
            if isinstance(response, str):
                # Try to parse JSON string from structured output
                try:
                    json_response = json.loads(response)
                    if isinstance(json_response, dict) and "message" in json_response:
                        return json_response["message"]
                    return response
                except json.JSONDecodeError:
                    return response
            elif isinstance(response, dict) and "message" in response:
                # It's already a structured output dict
                return response["message"]
            elif (
                isinstance(response, list)
                and response
                and hasattr(response[0], "content")
            ):
                # It's a list of AIMessage objects
                return response[0].content
            elif hasattr(response, "content"):
                # It's a single message object
                return response.content
            else:
                # It's some other format
                return str(response)
        except Exception as e:
            logger.warning(f"Error extracting content from response: {e}")
            return str(response)


class RateSamplesPostProcessor(NodePostProcessor):
    """
    Post-processor for the rate_samples node.
    Processes rating results and organizes them in the state.
    """

    def apply(self, response: Any, state: GraspState) -> GraspState:
        """
        Process rating results.

        Args:
            response: The response from the rate_samples node
            state: The current graph state

        Returns:
            Updated graph state
        """
        message_content = response.message.content

        logger.info(
            f"RateSamplesPostProcessor - Raw response: {message_content[:200]}..."
        )

        # Extract JSON ratings from response content
        samples_ratings = utils.extract_and_load_json(message_content)

        if not samples_ratings:
            logger.warning(
                f"Failed to extract JSON ratings from response: {message_content[:100]}..."
            )
            return state

        logger.info(f"RateSamplesPostProcessor - Extracted ratings: {samples_ratings}")

        # Add the model responses to each rating for easier reference later
        current_model_responses = state.get("current_model_responses", {})
        for rating in samples_ratings:
            if "assistant" in rating and rating["assistant"] in current_model_responses:
                rating["response"] = current_model_responses[rating["assistant"]]

        # Add the ratings to the state - samples_ratings should already be initialized
        if "samples_ratings" not in state:
            # Fallback initialization in case the data transform didn't run
            logger.warning(
                "samples_ratings not found in state, initializing as empty list"
            )
            state["samples_ratings"] = []

        state["samples_ratings"].append(samples_ratings)
        logger.info(
            f"RateSamplesPostProcessor - Updated samples_ratings: {state['samples_ratings']}"
        )

        return state


class ShouldContinueCondition(EdgeCondition):
    """
    Edge condition to determine whether to continue generating samples or end.
    """

    @staticmethod
    def apply(state: GraspState) -> str:
        """
        Check if we should continue generating samples.

        Args:
            state: The current graph state

        Returns:
            Next node name or END
        """
        samples_ratings = state.get("samples_ratings", [])

        # Hard safety limit - never go beyond 5 iterations
        if len(samples_ratings) >= 5:
            logger.warning(
                f"ShouldContinueCondition - Reached hard iteration limit: {len(samples_ratings)}"
            )
            return constants.GRASP_END

        # Log the current state for debugging
        logger.info(
            f"ShouldContinueCondition - Current samples_ratings length: {len(samples_ratings)}"
        )

        buckets = {}

        for samples_rating in samples_ratings:
            logger.info(f"Processing rating batch: {samples_rating}")
            for sample_rating in samples_rating:
                if not isinstance(sample_rating, dict) or "rating" not in sample_rating:
                    logger.warning(f"Invalid rating format: {sample_rating}")
                    continue

                try:
                    rating = float(sample_rating["rating"])
                    logger.info(f"Processing rating: {rating}")
                    if rating <= 4:  # 1,2,3,4
                        buckets[0] = 1
                    elif rating <= 7:  # 5,6,7
                        buckets[1] = 1
                    else:  # 8,9,10
                        buckets[2] = 1
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Invalid rating format: {sample_rating}, error: {e}"
                    )

        logger.info(f"ShouldContinueCondition - Buckets: {buckets}")

        # End after 3 iterations or if we have collected samples for all the buckets
        if len(buckets) == 3 or len(samples_ratings) >= 3:
            logger.info(
                f"ShouldContinueCondition - Ending flow: buckets={len(buckets)}, iterations={len(samples_ratings)}"
            )
            return constants.GRASP_END

        logger.info("ShouldContinueCondition - Continuing to generate more samples")
        return "generate_samples"


class DpoSamplesOutputGenerator(BaseOutputGenerator):
    """
    Output generator for the DPO samples task.
    """

    @staticmethod
    def build_conversation(data: Any, state: GraspState) -> list[dict]:
        """
        Build conversation from state data.

        Args:
            data: Data from state
            state: The graph state

        Returns:
            Conversation in the required format
        """
        conversation = [{"role": "user", "content": state.get("user_prompt", "")}]

        # Generate the assistant response with ratings and explanations
        assistant_content = DpoSamplesOutputGenerator._generate_asst_content(state)
        if assistant_content:
            conversation.append({"role": "assistant", "content": assistant_content})

        return conversation

    @staticmethod
    def _generate_asst_content(state: GraspState) -> List[Dict[str, Any]]:
        """
        Generate assistant content with ratings and explanations.

        Args:
            state: The graph state

        Returns:
            List of dictionaries with model responses and ratings
        """
        contents = []
        samples_ratings_list = state.get("samples_ratings", [])

        logger.info(
            f"_generate_asst_content - samples_ratings_list: {samples_ratings_list}"
        )

        # Process all ratings
        for samples_rating in samples_ratings_list:
            for rating in samples_rating:
                if not isinstance(rating, dict):
                    continue

                model = rating.get("assistant")
                if not model:
                    continue

                response = rating.get("response")
                if not response:
                    continue

                contents.append(
                    {
                        "generation": response,
                        "model": model,
                        "judge_rating": rating.get("rating", 0),
                        "judge_explanation": rating.get("explanation", ""),
                    }
                )

        # Sort by rating in descending order
        contents = sorted(
            contents,
            key=lambda x: (
                float(x["judge_rating"])
                if isinstance(x["judge_rating"], (int, float, str))
                else 0
            ),
            reverse=True,
        )

        return contents
